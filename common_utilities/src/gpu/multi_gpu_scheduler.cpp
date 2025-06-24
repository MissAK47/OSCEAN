/**
 * @file multi_gpu_scheduler.cpp
 * @brief 多GPU负载均衡调度器实现
 */

#include "common_utils/gpu/multi_gpu_scheduler.h"
#include <memory>
#include "common_utils/utilities/logging_utils.h"
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/thread.hpp>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace oscean::common_utils::gpu {

// 全局调度器实例
std::unique_ptr<MultiGPUScheduler> GlobalSchedulerManager::s_instance;
boost::mutex GlobalSchedulerManager::s_mutex;

/**
 * @brief 多GPU调度器内部实现
 */
class MultiGPUScheduler::Impl {
public:
    Impl(const std::vector<GPUDeviceInfo>& devices, const SchedulerConfig& config)
        : m_config(config), m_roundRobinIndex(0), m_isRunning(true) {
        
        // 初始化GPU负载信息
        for (const auto& device : devices) {
            auto workload = std::make_shared<GPUWorkload>();
            workload->deviceId = device.deviceId;
            workload->deviceInfo = device;
            workload->lastUpdate = boost::chrono::steady_clock::now();
            // 直接存储shared_ptr
            m_workloads[device.deviceId] = workload;
        }
        
        // 启动监控线程
        m_monitorThread = boost::thread(&Impl::monitoringLoop, this);
        
        // OSCEAN_LOG_INFO("MultiGPUScheduler", "Initialized with {} GPU devices", 
        //                devices.size()); // TODO: Fix log format
    }
    
    ~Impl() {
        m_isRunning = false;
        if (m_monitorThread.joinable()) {
            m_monitorThread.join();
        }
    }
    
    SchedulingDecision selectOptimalGPU(const GPUTaskInfo& taskInfo) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        SchedulingDecision decision;
        
        // 如果有偏好设备且可用，优先考虑
        if (taskInfo.preferredDeviceId && 
            m_workloads.find(*taskInfo.preferredDeviceId) != m_workloads.end()) {
            
            const auto& workload = m_workloads[*taskInfo.preferredDeviceId];
            if (canAcceptTask(*workload, taskInfo)) {
                decision.selectedDeviceId = *taskInfo.preferredDeviceId;
                decision.confidenceScore = 0.9f;
                decision.reason = "Preferred device available";
                return decision;
            }
        }
        
        // 根据调度策略选择设备
        switch (m_config.strategy) {
            case SchedulingStrategy::ROUND_ROBIN:
                decision = selectRoundRobin(taskInfo);
                break;
                
            case SchedulingStrategy::LEAST_LOADED:
                decision = selectLeastLoaded(taskInfo);
                break;
                
            case SchedulingStrategy::PERFORMANCE_BASED:
                decision = selectPerformanceBased(taskInfo);
                break;
                
            case SchedulingStrategy::MEMORY_AWARE:
                decision = selectMemoryAware(taskInfo);
                break;
                
            case SchedulingStrategy::AFFINITY_BASED:
                decision = selectAffinityBased(taskInfo);
                break;
                
            case SchedulingStrategy::POWER_EFFICIENT:
                decision = selectPowerEfficient(taskInfo);
                break;
                
            default:
                decision = selectLeastLoaded(taskInfo);
                break;
        }
        
        // 生成备选设备列表
        if (decision.selectedDeviceId >= 0) {
            for (const auto& [deviceId, workloadPtr] : m_workloads) {
                if (deviceId != decision.selectedDeviceId && 
                    canAcceptTask(*workloadPtr, taskInfo)) {
                    decision.alternativeDevices.push_back(deviceId);
                }
            }
        }
        
        return decision;
    }
    
    bool submitTask(int deviceId, const GPUTaskInfo& taskInfo) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        auto it = m_workloads.find(deviceId);
        if (it == m_workloads.end()) {
            // OSCEAN_LOG_ERROR("MultiGPUScheduler", "Invalid device ID: {}", deviceId); // TODO: Fix log format
            return false;
        }
        
        auto& workload = *it->second;
        
        // 检查是否可以接受任务
        if (!canAcceptTask(workload, taskInfo)) {
            // OSCEAN_LOG_WARN("MultiGPUScheduler", "Device {} cannot accept task {}", 
            //                deviceId, taskInfo.taskId); // TODO: Fix log format
            return false;
        }
        
        // 更新负载信息
        workload.queuedTasks++;
        workload.allocatedMemory += taskInfo.memoryRequirement;
        updateLoad(workload);
        
        // 记录任务信息
        m_taskDeviceMap[taskInfo.taskId] = deviceId;
        m_taskInfoMap[taskInfo.taskId] = taskInfo;
        
        // 触发事件
        fireEvent(SchedulerEventType::TASK_SCHEDULED, deviceId, taskInfo.taskId,
                 "Task scheduled to device " + std::to_string(deviceId));
        
        return true;
    }
    
    void taskStarted(int deviceId, const std::string& taskId) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        auto it = m_workloads.find(deviceId);
        if (it == m_workloads.end()) return;
        
        auto& workload = *it->second;
        if (workload.queuedTasks > 0) {
            workload.queuedTasks--;
        }
        workload.runningTasks++;
        
        // 记录开始时间
        m_taskStartTimes[taskId] = boost::chrono::steady_clock::now();
        
        updateLoad(workload);
        
        fireEvent(SchedulerEventType::TASK_STARTED, deviceId, taskId,
                 "Task started on device " + std::to_string(deviceId));
    }
    
    void taskCompleted(int deviceId, const std::string& taskId, 
                      boost::chrono::milliseconds executionTime) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        auto it = m_workloads.find(deviceId);
        if (it == m_workloads.end()) return;
        
        auto& workload = *it->second;
        if (workload.runningTasks > 0) {
            workload.runningTasks--;
        }
        workload.completedTasks++;
        
        // 释放内存
        auto taskIt = m_taskInfoMap.find(taskId);
        if (taskIt != m_taskInfoMap.end()) {
            if (workload.allocatedMemory >= taskIt->second.memoryRequirement) {
                workload.allocatedMemory -= taskIt->second.memoryRequirement;
            }
            m_taskInfoMap.erase(taskIt);
        }
        
        // 更新性能统计
        updatePerformanceStats(workload, executionTime.count());
        
        // 清理记录
        m_taskDeviceMap.erase(taskId);
        m_taskStartTimes.erase(taskId);
        
        updateLoad(workload);
        
        fireEvent(SchedulerEventType::TASK_COMPLETED, deviceId, taskId,
                 "Task completed in " + std::to_string(executionTime.count()) + "ms");
    }
    
    void taskFailed(int deviceId, const std::string& taskId, const std::string& reason) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        auto it = m_workloads.find(deviceId);
        if (it == m_workloads.end()) return;
        
        auto& workload = *it->second;
        if (workload.runningTasks > 0) {
            workload.runningTasks--;
        }
        workload.failedTasks++;
        
        // 释放内存
        auto taskIt = m_taskInfoMap.find(taskId);
        if (taskIt != m_taskInfoMap.end()) {
            if (workload.allocatedMemory >= taskIt->second.memoryRequirement) {
                workload.allocatedMemory -= taskIt->second.memoryRequirement;
            }
            m_taskInfoMap.erase(taskIt);
        }
        
        // 清理记录
        m_taskDeviceMap.erase(taskId);
        m_taskStartTimes.erase(taskId);
        
        updateLoad(workload);
        
        fireEvent(SchedulerEventType::TASK_FAILED, deviceId, taskId,
                 "Task failed: " + reason);
    }
    
    void updateMemoryUsage(int deviceId, size_t usedMemory) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        auto it = m_workloads.find(deviceId);
        if (it == m_workloads.end()) return;
        
        auto& workload = *it->second;
        workload.deviceInfo.memoryDetails.freeGlobalMemory = 
            workload.deviceInfo.memoryDetails.totalGlobalMemory - usedMemory;
        
        updateLoad(workload);
    }
    
    boost::optional<GPUWorkloadSnapshot> getWorkload(int deviceId) const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        auto it = m_workloads.find(deviceId);
        if (it != m_workloads.end()) {
            return GPUWorkloadSnapshot(*it->second);
        }
        return boost::none;
    }
    
    std::vector<GPUWorkloadSnapshot> getAllWorkloads() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        std::vector<GPUWorkloadSnapshot> result;
        for (const auto& [_, workloadPtr] : m_workloads) {
            result.push_back(GPUWorkloadSnapshot(*workloadPtr));
        }
        return result;
    }
    
    std::string getStatistics() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        std::stringstream ss;
        ss << "=== GPU Scheduler Statistics ===\n";
        ss << "Strategy: " << strategyToString(m_config.strategy) << "\n";
        ss << "Total Devices: " << m_workloads.size() << "\n\n";
        
        for (const auto& [deviceId, workloadPtr] : m_workloads) {
            ss << "Device " << deviceId << " (" << workloadPtr->deviceInfo.name << "):\n";
            ss << "  Current Load: " << std::fixed << std::setprecision(1) 
               << (workloadPtr->currentLoad * 100) << "%\n";
            ss << "  Queued Tasks: " << workloadPtr->queuedTasks.load() << "\n";
            ss << "  Running Tasks: " << workloadPtr->runningTasks.load() << "\n";
            ss << "  Completed Tasks: " << workloadPtr->completedTasks.load() << "\n";
            ss << "  Failed Tasks: " << workloadPtr->failedTasks.load() << "\n";
            ss << "  Memory Used: " << (workloadPtr->allocatedMemory.load() / (1024*1024)) << " MB / "
               << (workloadPtr->deviceInfo.memoryDetails.totalGlobalMemory / (1024*1024)) << " MB\n";
            ss << "  Avg Task Duration: " << std::fixed << std::setprecision(2) 
               << workloadPtr->avgTaskDuration.load() << " ms\n";
            ss << "  Throughput: " << std::fixed << std::setprecision(2) 
               << workloadPtr->throughput.load() << " tasks/sec\n\n";
        }
        
        return ss.str();
    }
    
    void registerEventCallback(SchedulerEventCallback callback) {
        boost::lock_guard<boost::mutex> lock(m_callbackMutex);
        m_eventCallbacks.push_back(callback);
    }
    
    void setSchedulingStrategy(SchedulingStrategy strategy) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        m_config.strategy = strategy;
        // OSCEAN_LOG_INFO("MultiGPUScheduler", "Scheduling strategy changed to: {}", 
        //                strategyToString(strategy)); // TODO: Fix log format
    }
    
    SchedulingStrategy getSchedulingStrategy() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        return m_config.strategy;
    }
    
    bool performLoadBalancing() {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        if (!m_config.enableDynamicBalancing) {
            return false;
        }
        
        // 计算平均负载
        float totalLoad = 0.0f;
        for (const auto& [_, workloadPtr] : m_workloads) {
            totalLoad += workloadPtr->currentLoad;
        }
        float avgLoad = totalLoad / m_workloads.size();
        
        // 检查是否需要负载均衡
        bool needsBalancing = false;
        for (const auto& [_, workloadPtr] : m_workloads) {
            if (std::abs(workloadPtr->currentLoad - avgLoad) > 0.2f) {
                needsBalancing = true;
                break;
            }
        }
        
        if (needsBalancing) {
            // OSCEAN_LOG_INFO("MultiGPUScheduler", "Load balancing triggered"); // TODO: Fix log format
            fireEvent(SchedulerEventType::LOAD_BALANCED, -1, "",
                     "Load balancing performed");
            return true;
        }
        
        return false;
    }
    
    void reset() {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        for (auto& [_, workloadPtr] : m_workloads) {
            workloadPtr->currentLoad = 0.0f;
            workloadPtr->queuedTasks = 0;
            workloadPtr->runningTasks = 0;
            workloadPtr->completedTasks = 0;
            workloadPtr->failedTasks = 0;
            workloadPtr->allocatedMemory = 0;
            workloadPtr->avgTaskDuration = 0.0;
            workloadPtr->throughput = 0.0;
        }
        
        m_taskDeviceMap.clear();
        m_taskInfoMap.clear();
        m_taskStartTimes.clear();
        m_roundRobinIndex = 0;
        
        // OSCEAN_LOG_INFO("MultiGPUScheduler", "Reset completed"); // TODO: Fix log format
    }
    
private:
    // 配置和状态
    SchedulerConfig m_config;
    mutable boost::mutex m_mutex;
    mutable boost::mutex m_callbackMutex;
    boost::atomic<bool> m_isRunning;
    boost::thread m_monitorThread;
    
    // GPU负载信息
    std::map<int, std::shared_ptr<GPUWorkload>> m_workloads;
    
    // 任务跟踪
    std::map<std::string, int> m_taskDeviceMap;
    std::map<std::string, GPUTaskInfo> m_taskInfoMap;
    std::map<std::string, boost::chrono::steady_clock::time_point> m_taskStartTimes;
    
    // 调度状态
    boost::atomic<int> m_roundRobinIndex;
    
    // 事件回调
    std::vector<SchedulerEventCallback> m_eventCallbacks;
    
    // 辅助函数
    bool canAcceptTask(const GPUWorkload& workload, const GPUTaskInfo& taskInfo) const {
        // 检查内存是否足够
        size_t requiredMemory = workload.allocatedMemory + taskInfo.memoryRequirement;
        size_t availableMemory = workload.deviceInfo.memoryDetails.freeGlobalMemory;
        
        if (requiredMemory > availableMemory * m_config.memoryThreshold) {
            return false;
        }
        
        // 检查负载是否过高
        if (workload.currentLoad > m_config.loadThreshold) {
            return false;
        }
        
        // 检查队列是否已满
        if (workload.queuedTasks >= m_config.maxQueuedTasksPerDevice) {
            return false;
        }
        
        return true;
    }
    
    void updateLoad(GPUWorkload& workload) {
        // 计算综合负载
        float queueLoad = static_cast<float>(workload.queuedTasks) / 
                         m_config.maxQueuedTasksPerDevice;
        float runningLoad = static_cast<float>(workload.runningTasks) / 10.0f; // 假设最多10个并行任务
        float memoryLoad = static_cast<float>(workload.allocatedMemory) / 
                          workload.deviceInfo.memoryDetails.totalGlobalMemory;
        
        // 加权平均
        workload.currentLoad = queueLoad * 0.3f + runningLoad * 0.5f + memoryLoad * 0.2f;
        workload.currentLoad = std::min(1.0f, workload.currentLoad.load());
        
        workload.lastUpdate = boost::chrono::steady_clock::now();
        
        // 检查是否过载
        if (workload.currentLoad > m_config.loadThreshold) {
            fireEvent(SchedulerEventType::DEVICE_OVERLOADED, workload.deviceId, "",
                     "Device overloaded: " + std::to_string(workload.currentLoad.load() * 100) + "%");
        }
    }
    
    void updatePerformanceStats(GPUWorkload& workload, double executionTime) {
        // 更新平均任务时长（指数移动平均）
        double alpha = 0.1; // 平滑因子
        double currentAvg = workload.avgTaskDuration.load();
        workload.avgTaskDuration = currentAvg * (1 - alpha) + executionTime * alpha;
        
        // 更新吞吐量
        auto now = boost::chrono::steady_clock::now();
        auto duration = boost::chrono::duration_cast<boost::chrono::seconds>(
            now - workload.lastUpdate).count();
        
        if (duration > 0) {
            workload.throughput = static_cast<double>(workload.completedTasks) / duration;
        }
    }
    
    // 调度策略实现
    SchedulingDecision selectRoundRobin(const GPUTaskInfo& taskInfo) {
        SchedulingDecision decision;
        
        int startIndex = m_roundRobinIndex.load();
        int deviceCount = m_workloads.size();
        
        for (int i = 0; i < deviceCount; ++i) {
            int index = (startIndex + i) % deviceCount;
            auto it = std::next(m_workloads.begin(), index);
            
            if (canAcceptTask(*it->second, taskInfo)) {
                decision.selectedDeviceId = it->first;
                decision.confidenceScore = 0.8f;
                decision.reason = "Round-robin selection";
                m_roundRobinIndex = (index + 1) % deviceCount;
                break;
            }
        }
        
        return decision;
    }
    
    SchedulingDecision selectLeastLoaded(const GPUTaskInfo& taskInfo) {
        SchedulingDecision decision;
        
        float minLoad = 1.0f;
        int selectedDevice = -1;
        
        for (const auto& [deviceId, workloadPtr] : m_workloads) {
            if (canAcceptTask(*workloadPtr, taskInfo) && workloadPtr->currentLoad < minLoad) {
                minLoad = workloadPtr->currentLoad;
                selectedDevice = deviceId;
            }
        }
        
        if (selectedDevice >= 0) {
            decision.selectedDeviceId = selectedDevice;
            decision.confidenceScore = 1.0f - minLoad;
            decision.reason = "Least loaded device (load: " + 
                            std::to_string(minLoad * 100) + "%)";
        }
        
        return decision;
    }
    
    SchedulingDecision selectPerformanceBased(const GPUTaskInfo& taskInfo) {
        SchedulingDecision decision;
        
        float bestScore = -1.0f;
        int selectedDevice = -1;
        
        for (const auto& [deviceId, workloadPtr] : m_workloads) {
            if (!canAcceptTask(*workloadPtr, taskInfo)) continue;
            
            // 计算性能得分
            float performanceScore = workloadPtr->deviceInfo.performanceScore / 100.0f;
            float loadPenalty = 1.0f - workloadPtr->currentLoad;
            float score = performanceScore * loadPenalty;
            
            if (score > bestScore) {
                bestScore = score;
                selectedDevice = deviceId;
            }
        }
        
        if (selectedDevice >= 0) {
            decision.selectedDeviceId = selectedDevice;
            decision.confidenceScore = bestScore;
            decision.reason = "Performance-based selection (score: " + 
                            std::to_string(bestScore) + ")";
        }
        
        return decision;
    }
    
    SchedulingDecision selectMemoryAware(const GPUTaskInfo& taskInfo) {
        SchedulingDecision decision;
        
        size_t maxFreeMemory = 0;
        int selectedDevice = -1;
        
        for (const auto& [deviceId, workloadPtr] : m_workloads) {
            if (!canAcceptTask(*workloadPtr, taskInfo)) continue;
            
            size_t freeMemory = workloadPtr->deviceInfo.memoryDetails.freeGlobalMemory - 
                               workloadPtr->allocatedMemory;
            
            if (freeMemory > maxFreeMemory && freeMemory >= taskInfo.memoryRequirement) {
                maxFreeMemory = freeMemory;
                selectedDevice = deviceId;
            }
        }
        
        if (selectedDevice >= 0) {
            decision.selectedDeviceId = selectedDevice;
            decision.confidenceScore = 0.9f;
            decision.reason = "Memory-aware selection (free: " + 
                            std::to_string(maxFreeMemory / (1024*1024)) + " MB)";
        }
        
        return decision;
    }
    
    SchedulingDecision selectAffinityBased(const GPUTaskInfo& taskInfo) {
        // 暂时使用性能优先策略
        return selectPerformanceBased(taskInfo);
    }
    
    SchedulingDecision selectPowerEfficient(const GPUTaskInfo& taskInfo) {
        // 选择功耗效率最高的设备（暂时使用负载最低的设备）
        return selectLeastLoaded(taskInfo);
    }
    
    void fireEvent(SchedulerEventType type, int deviceId, const std::string& taskId,
                  const std::string& message) {
        SchedulerEvent event;
        event.type = type;
        event.deviceId = deviceId;
        event.taskId = taskId;
        event.message = message;
        event.timestamp = boost::chrono::steady_clock::now();
        
        boost::lock_guard<boost::mutex> lock(m_callbackMutex);
        for (const auto& callback : m_eventCallbacks) {
            callback(event);
        }
    }
    
    void monitoringLoop() {
        while (m_isRunning) {
            boost::this_thread::sleep_for(m_config.updateInterval);
            
            // 执行负载均衡检查
            if (m_config.enableDynamicBalancing) {
                performLoadBalancing();
            }
            
            // 更新设备状态
            {
                boost::lock_guard<boost::mutex> lock(m_mutex);
                for (auto& [_, workloadPtr] : m_workloads) {
                    updateLoad(*workloadPtr);
                }
            }
        }
    }
    
    std::string strategyToString(SchedulingStrategy strategy) const {
        switch (strategy) {
            case SchedulingStrategy::ROUND_ROBIN: return "Round Robin";
            case SchedulingStrategy::LEAST_LOADED: return "Least Loaded";
            case SchedulingStrategy::PERFORMANCE_BASED: return "Performance Based";
            case SchedulingStrategy::MEMORY_AWARE: return "Memory Aware";
            case SchedulingStrategy::AFFINITY_BASED: return "Affinity Based";
            case SchedulingStrategy::POWER_EFFICIENT: return "Power Efficient";
            default: return "Unknown";
        }
    }
};

// MultiGPUScheduler实现
MultiGPUScheduler::MultiGPUScheduler(const std::vector<GPUDeviceInfo>& devices,
                                   const SchedulerConfig& config)
    : m_impl(std::make_unique<Impl>(devices, config)) {
}

MultiGPUScheduler::~MultiGPUScheduler() = default;

SchedulingDecision MultiGPUScheduler::selectOptimalGPU(const GPUTaskInfo& taskInfo) {
    return m_impl->selectOptimalGPU(taskInfo);
}

bool MultiGPUScheduler::submitTask(int deviceId, const GPUTaskInfo& taskInfo) {
    return m_impl->submitTask(deviceId, taskInfo);
}

void MultiGPUScheduler::taskStarted(int deviceId, const std::string& taskId) {
    m_impl->taskStarted(deviceId, taskId);
}

void MultiGPUScheduler::taskCompleted(int deviceId, const std::string& taskId,
                                    boost::chrono::milliseconds executionTime) {
    m_impl->taskCompleted(deviceId, taskId, executionTime);
}

void MultiGPUScheduler::taskFailed(int deviceId, const std::string& taskId,
                                  const std::string& reason) {
    m_impl->taskFailed(deviceId, taskId, reason);
}

void MultiGPUScheduler::updateMemoryUsage(int deviceId, size_t usedMemory) {
    m_impl->updateMemoryUsage(deviceId, usedMemory);
}

boost::optional<GPUWorkloadSnapshot> MultiGPUScheduler::getWorkload(int deviceId) const {
    return m_impl->getWorkload(deviceId);
}

std::vector<GPUWorkloadSnapshot> MultiGPUScheduler::getAllWorkloads() const {
    return m_impl->getAllWorkloads();
}

std::string MultiGPUScheduler::getStatistics() const {
    return m_impl->getStatistics();
}

void MultiGPUScheduler::registerEventCallback(SchedulerEventCallback callback) {
    m_impl->registerEventCallback(callback);
}

void MultiGPUScheduler::setSchedulingStrategy(SchedulingStrategy strategy) {
    m_impl->setSchedulingStrategy(strategy);
}

SchedulingStrategy MultiGPUScheduler::getSchedulingStrategy() const {
    return m_impl->getSchedulingStrategy();
}

bool MultiGPUScheduler::performLoadBalancing() {
    return m_impl->performLoadBalancing();
}

void MultiGPUScheduler::reset() {
    m_impl->reset();
}

SchedulerStatistics MultiGPUScheduler::getStatisticsData() const {
    SchedulerStatistics stats;
    
    // 获取所有工作负载信息
    auto workloads = getAllWorkloads();
    
    for (const auto& workload : workloads) {
        stats.totalTasksScheduled += workload.completedTasks + workload.failedTasks + 
                                   workload.runningTasks + workload.queuedTasks;
        stats.completedTasks += workload.completedTasks;
        stats.failedTasks += workload.failedTasks;
        stats.runningTasks += workload.runningTasks;
        stats.queuedTasks += workload.queuedTasks;
        
        // 累加平均执行时间（简单平均）
        if (workload.completedTasks > 0) {
            stats.averageExecutionTime += workload.avgTaskDuration * workload.completedTasks;
        }
    }
    
    // 计算平均执行时间
    if (stats.completedTasks > 0) {
        stats.averageExecutionTime /= stats.completedTasks;
    }
    
    // 计算负载均衡效率（基于负载标准差）
    if (!workloads.empty()) {
        double avgLoad = 0.0;
        for (const auto& workload : workloads) {
            avgLoad += workload.currentLoad;
        }
        avgLoad /= workloads.size();
        
        double variance = 0.0;
        for (const auto& workload : workloads) {
            double diff = workload.currentLoad - avgLoad;
            variance += diff * diff;
        }
        variance /= workloads.size();
        
        // 效率 = 1 - 标准差（归一化到0-1）
        double stdDev = std::sqrt(variance);
        stats.loadBalanceEfficiency = 1.0 - std::min(1.0, stdDev);
    }
    
    return stats;
}

// GlobalSchedulerManager实现
MultiGPUScheduler& GlobalSchedulerManager::getInstance() {
    boost::lock_guard<boost::mutex> lock(s_mutex);
    if (!s_instance) {
        throw std::runtime_error("Global GPU scheduler not initialized");
    }
    return *s_instance;
}

void GlobalSchedulerManager::initialize(const std::vector<GPUDeviceInfo>& devices,
                                      const SchedulerConfig& config) {
    boost::lock_guard<boost::mutex> lock(s_mutex);
    s_instance = std::make_unique<MultiGPUScheduler>(devices, config);
}

void GlobalSchedulerManager::destroy() {
    boost::lock_guard<boost::mutex> lock(s_mutex);
    s_instance.reset();
}

} // namespace oscean::common_utils::gpu 