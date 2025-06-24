/**
 * @file gpu_utils.h
 * @brief 通用GPU工具函数和辅助类
 */

#pragma once

#include "gpu_common.h"
#include "gpu_device_info.h"
#include <chrono>
#include <functional>
#include <memory>

namespace oscean::common_utils::gpu {

/**
 * @brief GPU计时器基类
 */
class GPUTimer {
public:
    virtual ~GPUTimer() = default;
    
    /**
     * @brief 开始计时
     */
    virtual void start() = 0;
    
    /**
     * @brief 停止计时
     */
    virtual void stop() = 0;
    
    /**
     * @brief 获取经过的时间（毫秒）
     */
    virtual float getElapsedMilliseconds() = 0;
};

/**
 * @brief CPU计时器（作为后备）
 */
class CPUTimer : public GPUTimer {
private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_stop;
    
public:
    void start() override {
        m_start = std::chrono::high_resolution_clock::now();
    }
    
    void stop() override {
        m_stop = std::chrono::high_resolution_clock::now();
    }
    
    float getElapsedMilliseconds() override {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(m_stop - m_start);
        return duration.count() / 1000.0f;
    }
};

/**
 * @brief GPU性能计数器
 */
class GPUPerformanceCounter {
private:
    std::string m_name;
    size_t m_count = 0;
    double m_totalTime = 0.0;
    double m_minTime = std::numeric_limits<double>::max();
    double m_maxTime = 0.0;
    
public:
    explicit GPUPerformanceCounter(const std::string& name) : m_name(name) {}
    
    void record(double timeMs) {
        m_count++;
        m_totalTime += timeMs;
        m_minTime = std::min(m_minTime, timeMs);
        m_maxTime = std::max(m_maxTime, timeMs);
    }
    
    double getAverageTime() const {
        return m_count > 0 ? m_totalTime / m_count : 0.0;
    }
    
    double getMinTime() const { return m_minTime; }
    double getMaxTime() const { return m_maxTime; }
    size_t getCount() const { return m_count; }
    const std::string& getName() const { return m_name; }
    
    void reset() {
        m_count = 0;
        m_totalTime = 0.0;
        m_minTime = std::numeric_limits<double>::max();
        m_maxTime = 0.0;
    }
};

/**
 * @brief GPU内存使用跟踪器
 */
class GPUMemoryTracker {
private:
    size_t m_currentUsage = 0;
    size_t m_peakUsage = 0;
    size_t m_totalAllocated = 0;
    size_t m_totalDeallocated = 0;
    
public:
    void recordAllocation(size_t bytes) {
        m_currentUsage += bytes;
        m_totalAllocated += bytes;
        m_peakUsage = std::max(m_peakUsage, m_currentUsage);
    }
    
    void recordDeallocation(size_t bytes) {
        m_currentUsage = (bytes <= m_currentUsage) ? m_currentUsage - bytes : 0;
        m_totalDeallocated += bytes;
    }
    
    size_t getCurrentUsage() const { return m_currentUsage; }
    size_t getPeakUsage() const { return m_peakUsage; }
    size_t getTotalAllocated() const { return m_totalAllocated; }
    size_t getTotalDeallocated() const { return m_totalDeallocated; }
    
    void reset() {
        m_currentUsage = 0;
        m_peakUsage = 0;
        m_totalAllocated = 0;
        m_totalDeallocated = 0;
    }
};

/**
 * @brief GPU错误处理辅助函数
 */
inline std::string gpuErrorToString(GPUError error) {
    switch (error) {
        case GPUError::SUCCESS: return "Success";
        case GPUError::INVALID_DEVICE: return "Invalid device";
        case GPUError::OUT_OF_MEMORY: return "Out of memory";
        case GPUError::INVALID_INPUT: return "Invalid input";
        case GPUError::KERNEL_LAUNCH_ERROR: return "Kernel launch error";
        case GPUError::MEMORY_TRANSFER_ERROR: return "Memory transfer error";
        case GPUError::UNSUPPORTED_OPERATION: return "Unsupported operation";
        case GPUError::INITIALIZATION_ERROR: return "Initialization error";
        case GPUError::CUDA_ERROR: return "CUDA error";
        case GPUError::OPENCL_ERROR: return "OpenCL error";
        case GPUError::UNKNOWN_ERROR: return "Unknown error";
        default: return "Unrecognized error";
    }
}

/**
 * @brief GPU内存对齐辅助函数
 */
inline size_t alignSize(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

/**
 * @brief 计算网格和块大小的辅助函数
 */
struct LaunchConfig {
    int gridSize;
    int blockSize;
};

inline LaunchConfig calculateLaunchConfig(
    size_t totalElements,
    int maxBlockSize = 256,
    int minBlocksPerMultiprocessor = 2) {
    
    LaunchConfig config;
    config.blockSize = (totalElements < maxBlockSize) ? totalElements : maxBlockSize;
    config.gridSize = (totalElements + config.blockSize - 1) / config.blockSize;
    return config;
}

/**
 * @brief 2D启动配置
 */
struct LaunchConfig2D {
    struct Dim2 {
        int x, y;
    };
    Dim2 gridSize;
    Dim2 blockSize;
};

inline LaunchConfig2D calculateLaunchConfig2D(
    int width, int height,
    int blockX = 16, int blockY = 16) {
    
    LaunchConfig2D config;
    config.blockSize = {blockX, blockY};
    config.gridSize = {
        (width + blockX - 1) / blockX,
        (height + blockY - 1) / blockY
    };
    return config;
}

/**
 * @brief GPU任务包装器
 */
template<typename ResultType>
class GPUTask {
private:
    std::function<ResultType()> m_task;
    std::string m_name;
    int m_deviceId;
    
public:
    GPUTask(const std::string& name, int deviceId, std::function<ResultType()> task)
        : m_name(name), m_deviceId(deviceId), m_task(std::move(task)) {}
    
    ResultType execute() {
        return m_task();
    }
    
    const std::string& getName() const { return m_name; }
    int getDeviceId() const { return m_deviceId; }
};

/**
 * @brief GPU资源作用域管理器
 */
class GPUResourceScope {
private:
    std::function<void()> m_cleanup;
    
public:
    explicit GPUResourceScope(std::function<void()> cleanup)
        : m_cleanup(std::move(cleanup)) {}
    
    ~GPUResourceScope() {
        if (m_cleanup) {
            m_cleanup();
        }
    }
    
    // 禁止拷贝
    GPUResourceScope(const GPUResourceScope&) = delete;
    GPUResourceScope& operator=(const GPUResourceScope&) = delete;
    
    // 允许移动
    GPUResourceScope(GPUResourceScope&& other) noexcept
        : m_cleanup(std::move(other.m_cleanup)) {
        other.m_cleanup = nullptr;
    }
};

/**
 * @brief 数据类型大小辅助函数
 */
inline size_t getDataTypeSize(DataType type) {
    switch (type) {
        case DataType::INT8: return 1;
        case DataType::UINT8: return 1;
        case DataType::INT16: return 2;
        case DataType::UINT16: return 2;
        case DataType::INT32: return 4;
        case DataType::UINT32: return 4;
        case DataType::INT64: return 8;
        case DataType::UINT64: return 8;
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT64: return 8;
        default: return 0;
    }
}

/**
 * @brief 格式化内存大小
 */
inline std::string formatMemorySize(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unitIndex < 4) {
        size /= 1024.0;
        unitIndex++;
    }
    
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unitIndex]);
    return std::string(buffer);
}

/**
 * @brief GPU设备选择策略
 */
enum class DeviceSelectionStrategy {
    FIRST_AVAILABLE,      // 选择第一个可用设备
    HIGHEST_PERFORMANCE,  // 选择性能最高的设备
    MOST_MEMORY,         // 选择内存最大的设备
    LEAST_LOADED,        // 选择负载最低的设备
    ROUND_ROBIN          // 轮询选择
};

/**
 * @brief 根据策略选择GPU设备
 */
inline int selectDevice(
    const std::vector<GPUDeviceInfo>& devices,
    DeviceSelectionStrategy strategy,
    size_t memoryRequirement = 0) {
    
    if (devices.empty()) return -1;
    
    switch (strategy) {
        case DeviceSelectionStrategy::FIRST_AVAILABLE:
            for (size_t i = 0; i < devices.size(); ++i) {
                if (devices[i].freeMemory >= memoryRequirement) {
                    return static_cast<int>(i);
                }
            }
            break;
            
        case DeviceSelectionStrategy::HIGHEST_PERFORMANCE: {
            int bestDevice = -1;
            int bestScore = -1;
            for (size_t i = 0; i < devices.size(); ++i) {
                if (devices[i].freeMemory >= memoryRequirement &&
                    devices[i].performanceScore > bestScore) {
                    bestDevice = static_cast<int>(i);
                    bestScore = devices[i].performanceScore;
                }
            }
            return bestDevice;
        }
        
        case DeviceSelectionStrategy::MOST_MEMORY: {
            int bestDevice = -1;
            size_t maxMemory = memoryRequirement;
            for (size_t i = 0; i < devices.size(); ++i) {
                if (devices[i].freeMemory > maxMemory) {
                    bestDevice = static_cast<int>(i);
                    maxMemory = devices[i].freeMemory;
                }
            }
            return bestDevice;
        }
        
        case DeviceSelectionStrategy::LEAST_LOADED:
            // 这需要实时负载信息，暂时返回第一个可用设备
            return selectDevice(devices, DeviceSelectionStrategy::FIRST_AVAILABLE, memoryRequirement);
            
        case DeviceSelectionStrategy::ROUND_ROBIN:
            // 简单的轮询实现
            static int lastDevice = -1;
            lastDevice = (lastDevice + 1) % devices.size();
            return lastDevice;
    }
    
    return -1;
}

} // namespace oscean::common_utils::gpu 