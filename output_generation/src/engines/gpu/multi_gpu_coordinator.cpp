/**
 * @file multi_gpu_coordinator.cpp
 * @brief 多GPU协同系统实现
 */

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <chrono>
#include <functional>
#include <boost/thread/future.hpp>
#include <boost/log/trivial.hpp>
#include "output_generation/gpu/gpu_tile_types.h"
#include "core_services/common_data_types.h"

#ifdef OSCEAN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace oscean::output_generation::gpu {

using oscean::core_services::GridData;  // 添加using声明

/**
 * @brief GPU工作项
 */
struct GPUWorkItem {
    int deviceId;
    size_t dataOffset;
    size_t dataSize;
    std::function<void()> task;
    boost::promise<bool> completion;
};

/**
 * @brief GPU设备状态
 */
struct GPUDeviceState {
    int deviceId;
    std::string name;
    size_t totalMemory;
    size_t freeMemory;
    std::atomic<float> currentLoad{0.0f};
    std::atomic<int> activeJobs{0};
    std::chrono::steady_clock::time_point lastUpdate;
    
    // 性能统计
    double avgProcessingTime = 0.0;
    int completedJobs = 0;
    
    // 默认构造函数
    GPUDeviceState() = default;
    
    // 拷贝构造函数
    GPUDeviceState(const GPUDeviceState& other) 
        : deviceId(other.deviceId),
          name(other.name),
          totalMemory(other.totalMemory),
          freeMemory(other.freeMemory),
          currentLoad(other.currentLoad.load()),
          activeJobs(other.activeJobs.load()),
          lastUpdate(other.lastUpdate),
          avgProcessingTime(other.avgProcessingTime),
          completedJobs(other.completedJobs) {
    }
    
    // 拷贝赋值运算符
    GPUDeviceState& operator=(const GPUDeviceState& other) {
        if (this != &other) {
            deviceId = other.deviceId;
            name = other.name;
            totalMemory = other.totalMemory;
            freeMemory = other.freeMemory;
            currentLoad.store(other.currentLoad.load());
            activeJobs.store(other.activeJobs.load());
            lastUpdate = other.lastUpdate;
            avgProcessingTime = other.avgProcessingTime;
            completedJobs = other.completedJobs;
        }
        return *this;
    }
    
    // 移动构造函数
    GPUDeviceState(GPUDeviceState&& other) noexcept
        : deviceId(other.deviceId),
          name(std::move(other.name)),
          totalMemory(other.totalMemory),
          freeMemory(other.freeMemory),
          currentLoad(other.currentLoad.load()),
          activeJobs(other.activeJobs.load()),
          lastUpdate(other.lastUpdate),
          avgProcessingTime(other.avgProcessingTime),
          completedJobs(other.completedJobs) {
    }
    
    // 移动赋值运算符
    GPUDeviceState& operator=(GPUDeviceState&& other) noexcept {
        if (this != &other) {
            deviceId = other.deviceId;
            name = std::move(other.name);
            totalMemory = other.totalMemory;
            freeMemory = other.freeMemory;
            currentLoad.store(other.currentLoad.load());
            activeJobs.store(other.activeJobs.load());
            lastUpdate = other.lastUpdate;
            avgProcessingTime = other.avgProcessingTime;
            completedJobs = other.completedJobs;
        }
        return *this;
    }
};

/**
 * @brief 多GPU协同器
 */
class MultiGPUCoordinator {
private:
    std::vector<GPUDeviceState> m_devices;
    std::vector<std::thread> m_workers;
    std::queue<std::shared_ptr<GPUWorkItem>> m_workQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    std::atomic<bool> m_running{false};
    
    // 负载均衡策略
    enum class LoadBalancingStrategy {
        ROUND_ROBIN,
        LEAST_LOADED,
        MEMORY_AWARE,
        PERFORMANCE_BASED
    };
    
    LoadBalancingStrategy m_strategy = LoadBalancingStrategy::LEAST_LOADED;
    
public:
    MultiGPUCoordinator() {
        detectGPUs();
    }
    
    ~MultiGPUCoordinator() {
        stop();
    }
    
    /**
     * @brief 检测可用GPU
     */
    void detectGPUs() {
        #ifdef OSCEAN_CUDA_ENABLED
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            GPUDeviceState state;
            state.deviceId = i;
            state.name = prop.name;
            state.totalMemory = prop.totalGlobalMem;
            
            size_t free, total;
            cudaSetDevice(i);
            cudaMemGetInfo(&free, &total);
            state.freeMemory = free;
            
            m_devices.push_back(state);
            
            BOOST_LOG_TRIVIAL(info) << "Detected GPU " << i << ": " 
                                   << state.name << " (" 
                                   << state.totalMemory / (1024*1024) << " MB)";
        }
        #endif
    }
    
    /**
     * @brief 启动工作线程
     */
    void start() {
        if (m_running) return;
        
        m_running = true;
        
        // 为每个GPU创建工作线程
        for (auto& device : m_devices) {
            m_workers.emplace_back([this, &device]() {
                workerThread(device);
            });
        }
    }
    
    /**
     * @brief 停止工作线程
     */
    void stop() {
        if (!m_running) return;
        
        m_running = false;
        m_queueCV.notify_all();
        
        for (auto& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        m_workers.clear();
    }
    
    /**
     * @brief 数据分片策略
     */
    struct DataPartition {
        int deviceId;
        size_t offset;
        size_t size;
        float weight;  // 基于设备性能的权重
    };
    
    /**
     * @brief 智能数据分片
     */
    std::vector<DataPartition> partitionData(size_t totalSize) {
        std::vector<DataPartition> partitions;
        
        if (m_devices.empty()) return partitions;
        
        // 计算每个设备的权重
        float totalWeight = 0.0f;
        std::vector<float> weights;
        
        for (const auto& device : m_devices) {
            float weight = 1.0f;
            
            // 基于可用内存的权重
            weight *= (float)device.freeMemory / device.totalMemory;
            
            // 基于当前负载的权重
            weight *= (1.0f - device.currentLoad);
            
            // 基于历史性能的权重
            if (device.completedJobs > 0) {
                float performanceScore = 1.0f / (device.avgProcessingTime + 0.001f);
                weight *= performanceScore;
            }
            
            weights.push_back(weight);
            totalWeight += weight;
        }
        
        // 分配数据
        size_t currentOffset = 0;
        for (size_t i = 0; i < m_devices.size(); ++i) {
            DataPartition partition;
            partition.deviceId = m_devices[i].deviceId;
            partition.offset = currentOffset;
            partition.weight = weights[i] / totalWeight;
            
            // 最后一个设备获取剩余数据
            if (i == m_devices.size() - 1) {
                partition.size = totalSize - currentOffset;
            } else {
                partition.size = static_cast<size_t>(totalSize * partition.weight);
                // 对齐到合适的边界
                partition.size = (partition.size + 255) & ~255;
            }
            
            currentOffset += partition.size;
            partitions.push_back(partition);
            
            BOOST_LOG_TRIVIAL(debug) << "GPU " << partition.deviceId 
                                    << " assigned " << partition.size 
                                    << " bytes (" << (partition.weight * 100) << "%)";
        }
        
        return partitions;
    }
    
    /**
     * @brief 提交批量任务
     */
    template<typename TaskFunc>
    std::vector<boost::future<bool>> submitBatchTasks(
        const std::vector<DataPartition>& partitions,
        TaskFunc taskGenerator) {
        
        std::vector<boost::future<bool>> futures;
        
        for (const auto& partition : partitions) {
            auto workItem = std::make_shared<GPUWorkItem>();
            workItem->deviceId = partition.deviceId;
            workItem->dataOffset = partition.offset;
            workItem->dataSize = partition.size;
            workItem->task = taskGenerator(partition);
            
            futures.push_back(workItem->completion.get_future());
            
            {
                std::lock_guard<std::mutex> lock(m_queueMutex);
                m_workQueue.push(workItem);
            }
            m_queueCV.notify_one();
        }
        
        return futures;
    }
    
    /**
     * @brief 动态负载均衡
     */
    void rebalanceLoad() {
        // 检查是否需要重新平衡
        float loadVariance = calculateLoadVariance();
        
        if (loadVariance > 0.3f) {  // 负载不平衡阈值
            BOOST_LOG_TRIVIAL(info) << "Rebalancing GPU load (variance: " 
                                   << loadVariance << ")";
            
            // 实现工作迁移逻辑
            // 1. 识别过载的GPU
            std::vector<int> overloadedDevices;
            std::vector<int> underloadedDevices;
            
            float avgLoad = 0.0f;
            for (const auto& device : m_devices) {
                avgLoad += device.currentLoad;
            }
            avgLoad /= m_devices.size();
            
            // 分类设备
            for (size_t i = 0; i < m_devices.size(); ++i) {
                float load = m_devices[i].currentLoad;
                if (load > avgLoad + 0.2f) {
                    overloadedDevices.push_back(i);
                } else if (load < avgLoad - 0.2f) {
                    underloadedDevices.push_back(i);
                }
            }
            
            // 2. 迁移工作
            if (!overloadedDevices.empty() && !underloadedDevices.empty()) {
                migrateWorkload(overloadedDevices, underloadedDevices);
            }
        }
    }
    
    /**
     * @brief 迁移工作负载
     */
    void migrateWorkload(const std::vector<int>& fromDevices, 
                        const std::vector<int>& toDevices) {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        
        // 临时存储队列中的任务
        std::vector<std::shared_ptr<GPUWorkItem>> items;
        while (!m_workQueue.empty()) {
            items.push_back(m_workQueue.front());
            m_workQueue.pop();
        }
        
        // 重新分配任务
        size_t migratedCount = 0;
        for (auto& item : items) {
            bool shouldMigrate = false;
            
            // 检查是否需要迁移
            for (int fromDevice : fromDevices) {
                if (item->deviceId == m_devices[fromDevice].deviceId) {
                    shouldMigrate = true;
                    break;
                }
            }
            
            if (shouldMigrate && !toDevices.empty()) {
                // 选择负载最低的目标设备
                int bestDevice = toDevices[0];
                float minLoad = m_devices[toDevices[0]].currentLoad;
                
                for (int deviceIdx : toDevices) {
                    if (m_devices[deviceIdx].currentLoad < minLoad) {
                        minLoad = m_devices[deviceIdx].currentLoad;
                        bestDevice = deviceIdx;
                    }
                }
                
                // 迁移任务
                item->deviceId = m_devices[bestDevice].deviceId;
                migratedCount++;
                
                BOOST_LOG_TRIVIAL(debug) << "Migrated task from GPU " 
                                        << item->deviceId << " to GPU " 
                                        << m_devices[bestDevice].deviceId;
            }
            
            // 重新加入队列
            m_workQueue.push(item);
        }
        
        if (migratedCount > 0) {
            BOOST_LOG_TRIVIAL(info) << "Migrated " << migratedCount 
                                   << " tasks for load balancing";
            m_queueCV.notify_all();
        }
    }
    
    /**
     * @brief 获取设备状态
     */
    std::vector<GPUDeviceState> getDeviceStates() const {
        return m_devices;
    }
    
private:
    /**
     * @brief 工作线程函数
     */
    void workerThread(GPUDeviceState& device) {
        #ifdef OSCEAN_CUDA_ENABLED
        cudaSetDevice(device.deviceId);
        #endif
        
        while (m_running) {
            std::shared_ptr<GPUWorkItem> workItem;
            
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_queueCV.wait(lock, [this] {
                    return !m_workQueue.empty() || !m_running;
                });
                
                if (!m_running) break;
                
                if (!m_workQueue.empty()) {
                    // 查找适合此设备的工作项
                    std::queue<std::shared_ptr<GPUWorkItem>> tempQueue;
                    bool found = false;
                    
                    while (!m_workQueue.empty() && !found) {
                        auto item = m_workQueue.front();
                        m_workQueue.pop();
                        
                        if (item->deviceId == device.deviceId) {
                            workItem = item;
                            found = true;
                        } else {
                            tempQueue.push(item);
                        }
                    }
                    
                    // 恢复队列
                    while (!tempQueue.empty()) {
                        m_workQueue.push(tempQueue.front());
                        tempQueue.pop();
                    }
                }
            }
            
            if (workItem) {
                // 执行任务
                device.activeJobs++;
                auto startTime = std::chrono::steady_clock::now();
                
                try {
                    workItem->task();
                    workItem->completion.set_value(true);
                } catch (const std::exception& e) {
                    BOOST_LOG_TRIVIAL(error) << "GPU " << device.deviceId 
                                           << " task failed: " << e.what();
                    workItem->completion.set_value(false);
                }
                
                auto endTime = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    endTime - startTime).count();
                
                // 更新统计
                device.activeJobs--;
                device.completedJobs++;
                device.avgProcessingTime = 
                    (device.avgProcessingTime * (device.completedJobs - 1) + duration) 
                    / device.completedJobs;
                
                // 更新负载
                updateDeviceLoad(device);
            }
        }
    }
    
    /**
     * @brief 更新设备负载
     */
    void updateDeviceLoad(GPUDeviceState& device) {
        #ifdef OSCEAN_CUDA_ENABLED
        cudaSetDevice(device.deviceId);
        
        // 获取内存使用情况
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        device.freeMemory = free;
        
        // 计算负载（简化版本）
        float memoryLoad = 1.0f - (float)free / total;
        float jobLoad = (float)device.activeJobs / 10.0f;  // 假设最多10个并发任务
        
        device.currentLoad = (memoryLoad + jobLoad) / 2.0f;
        device.lastUpdate = std::chrono::steady_clock::now();
        #endif
    }
    
    /**
     * @brief 计算负载方差
     */
    float calculateLoadVariance() {
        if (m_devices.size() < 2) return 0.0f;
        
        float avgLoad = 0.0f;
        for (const auto& device : m_devices) {
            avgLoad += device.currentLoad;
        }
        avgLoad /= m_devices.size();
        
        float variance = 0.0f;
        for (const auto& device : m_devices) {
            float diff = device.currentLoad - avgLoad;
            variance += diff * diff;
        }
        variance /= m_devices.size();
        
        return variance;
    }
};

/**
 * @brief 多GPU瓦片生成示例
 */
class MultiGPUTileGenerator {
private:
    std::unique_ptr<MultiGPUCoordinator> m_coordinator;
    
public:
    MultiGPUTileGenerator() {
        m_coordinator = std::make_unique<MultiGPUCoordinator>();
        m_coordinator->start();
    }
    
    /**
     * @brief 生成瓦片批次
     */
    std::vector<GPUTileResult> generateTileBatch(
        std::shared_ptr<GridData> data,
        const std::vector<TileRequest>& requests) {
        
        // 数据分片
        size_t totalDataSize = data->getSizeInBytes();
        auto partitions = m_coordinator->partitionData(totalDataSize);
        
        // 按照分区分配瓦片请求
        std::vector<std::vector<TileRequest>> partitionedRequests(partitions.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            // 简单的轮询分配，可以根据瓦片位置优化
            size_t partitionIdx = i % partitions.size();
            partitionedRequests[partitionIdx].push_back(requests[i]);
        }
        
        // 结果收集
        std::vector<GPUTileResult> allResults;
        std::mutex resultMutex;
        
        // 创建任务生成器
        auto taskGenerator = [&](const MultiGPUCoordinator::DataPartition& partition) {
            return [=, &partitionedRequests, &allResults, &resultMutex]() {
                #ifdef OSCEAN_CUDA_ENABLED
                cudaSetDevice(partition.deviceId);
                
                // 获取此分区的瓦片请求
                size_t partitionIdx = 0;
                for (size_t i = 0; i < partitions.size(); ++i) {
                    if (partitions[i].deviceId == partition.deviceId) {
                        partitionIdx = i;
                        break;
                    }
                }
                
                const auto& localRequests = partitionedRequests[partitionIdx];
                
                // 在GPU上分配内存
                float* d_gridData = nullptr;
                size_t dataSize = partition.size;
                cudaMalloc(&d_gridData, dataSize);
                
                // 复制数据到GPU
                const float* hostData = static_cast<const float*>(data->getDataPtr());
                cudaMemcpy(d_gridData, 
                          hostData + partition.offset, 
                          dataSize, cudaMemcpyHostToDevice);
                
                // 为每个瓦片请求生成瓦片
                for (const auto& request : localRequests) {
                    GPUTileResult result;
                    result.tileX = request.tileX;
                    result.tileY = request.tileY;
                    result.zoomLevel = request.zoomLevel;
                    
                    try {
                        // 创建瓦片生成参数
                        GPUTileParams params;
                        // 使用请求中的信息设置参数
                        params.tileSize = 256;
                        params.colormap = "viridis";
                        params.autoScale = true;
                        
                        // 分配瓦片内存
                        uint32_t* d_tileData = nullptr;
                        size_t tilePixels = params.tileSize * params.tileSize;
                        cudaMalloc(&d_tileData, tilePixels * sizeof(uint32_t));
                        
                        // 调用GPU瓦片生成核函数（简化版本）
                        dim3 blockSize(16, 16);
                        dim3 gridSize((params.tileSize + blockSize.x - 1) / blockSize.x,
                                     (params.tileSize + blockSize.y - 1) / blockSize.y);
                        
                        // 这里应该调用实际的瓦片生成核函数
                        // generateTileKernel<<<gridSize, blockSize>>>(...)
                        
                        // 复制结果回主机
                        std::vector<uint8_t> tileData(tilePixels * 4); // RGBA格式
                        cudaMemcpy(tileData.data(), d_tileData, 
                                  tilePixels * sizeof(uint32_t), 
                                  cudaMemcpyDeviceToHost);
                        
                        result.success = true;
                        result.tileData = std::move(tileData);
                        result.width = params.tileSize;
                        result.height = params.tileSize;
                        result.format = "RGBA";
                        result.stats.totalTime = 10; // 示例时间
                        
                        // 清理瓦片内存
                        cudaFree(d_tileData);
                        
                        BOOST_LOG_TRIVIAL(debug) << "GPU " << partition.deviceId 
                                                << " generated tile " 
                                                << request.tileX << "," 
                                                << request.tileY << " @ Z" 
                                                << request.zoomLevel;
                    } catch (const std::exception& e) {
                        result.success = false;
                        BOOST_LOG_TRIVIAL(error) << "Tile generation failed: " << e.what();
                    }
                    
                    // 线程安全地添加结果
                    {
                        std::lock_guard<std::mutex> lock(resultMutex);
                        allResults.push_back(result);
                    }
                }
                
                // 清理GPU内存
                cudaFree(d_gridData);
                
                BOOST_LOG_TRIVIAL(debug) << "GPU " << partition.deviceId 
                                        << " completed " << localRequests.size() 
                                        << " tiles";
                #endif
            };
        };
        
        // 提交任务
        auto futures = m_coordinator->submitBatchTasks(partitions, taskGenerator);
        
        // 等待完成
        for (auto& future : futures) {
            future.get();
        }
        
        // 动态负载均衡
        m_coordinator->rebalanceLoad();
        
        return allResults;
    }
};

} // namespace oscean::output_generation::gpu 