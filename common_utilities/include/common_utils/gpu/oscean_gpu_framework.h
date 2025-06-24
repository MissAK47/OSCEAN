/**
 * @file oscean_gpu_framework.h
 * @brief OSCEAN统一GPU框架接口
 */

#pragma once

#include "unified_gpu_manager.h"
#include "multi_gpu_scheduler.h"
#include "multi_gpu_memory_manager.h"
#include <boost/thread/mutex.hpp>
#include <boost/thread/once.hpp>
#include <boost/any.hpp>
#include <memory>
#include <functional>

namespace oscean {
namespace common_utils {
namespace gpu {

/**
 * @brief GPU框架配置
 */
struct GPUFrameworkConfig {
    // GPU检测配置
    bool autoDetectGPUs = true;              // 自动检测所有GPU
    bool preferHighPerformanceGPU = true;    // 优先使用高性能GPU
    bool enableMultiGPU = true;              // 启用多GPU支持
    
    // 调度器配置
    SchedulerConfig schedulerConfig;
    
    // 内存管理配置
    MemoryPoolConfig memoryPoolConfig;
    MemoryTransferConfig transferConfig;
    
    // 性能监控配置
    bool enablePerformanceMonitoring = true;
    boost::chrono::milliseconds monitoringInterval{1000}; // 监控间隔
    
    // 错误处理配置
    bool enableAutoRecovery = true;          // 启用自动恢复
    int maxRetryAttempts = 3;                // 最大重试次数
    
    // 日志配置
    bool enableDetailedLogging = false;      // 详细日志
    std::string logPrefix = "[GPU Framework]";
};

/**
 * @brief GPU框架状态
 */
struct GPUFrameworkStatus {
    bool initialized = false;
    size_t availableDevices = 0;
    size_t activeDevices = 0;
    std::vector<GPUDeviceInfo> devices;
    
    // 性能指标
    double averageGPUUtilization = 0.0;
    double averageMemoryUtilization = 0.0;
    size_t totalTasksProcessed = 0;
    size_t failedTasks = 0;
    
    // 内存状态
    size_t totalMemoryAllocated = 0;
    size_t totalMemoryAvailable = 0;
    
    std::string toString() const;
};

/**
 * @brief GPU任务执行上下文
 */
struct GPUTaskExecutionContext {
    int deviceId = -1;                       // 执行设备ID
    GPUMemoryHandle inputMemory;             // 输入内存
    GPUMemoryHandle outputMemory;            // 输出内存
    std::map<std::string, void*> parameters; // 额外参数
    
    // 性能度量
    boost::chrono::steady_clock::time_point startTime;
    boost::chrono::steady_clock::time_point endTime;
    double executionTimeMs = 0.0;
    
    // 结果存储
    std::map<std::string, boost::any> results;
    
    // 设置结果
    template<typename T>
    void setResult(const std::string& key, const T& value) {
        results[key] = value;
    }
    
    // 获取结果
    template<typename T>
    bool getResult(const std::string& key, T& value) const {
        auto it = results.find(key);
        if (it != results.end()) {
            try {
                value = boost::any_cast<T>(it->second);
                return true;
            } catch (const boost::bad_any_cast&) {
                return false;
            }
        }
        return false;
    }
};

/**
 * @brief GPU任务接口
 */
class IGPUTask {
public:
    virtual ~IGPUTask() = default;
    
    /**
     * @brief 获取任务名称
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief 估算内存需求
     */
    virtual size_t estimateMemoryRequirement() const = 0;
    
    /**
     * @brief 估算计算复杂度
     */
    virtual double estimateComputeComplexity() const = 0;
    
    /**
     * @brief 执行任务
     */
    virtual bool execute(GPUTaskExecutionContext& context) = 0;
    
    /**
     * @brief 验证执行结果
     */
    virtual bool validate(const GPUTaskExecutionContext& context) const { return true; }
};

/**
 * @brief GPU框架事件类型
 */
enum class FrameworkEventType {
    INITIALIZED,
    SHUTDOWN,
    DEVICE_ADDED,
    DEVICE_REMOVED,
    DEVICE_ERROR,
    TASK_SUBMITTED,
    TASK_COMPLETED,
    TASK_FAILED,
    PERFORMANCE_WARNING,
    MEMORY_WARNING,
    RECOVERY_ATTEMPTED,
    RECOVERY_SUCCESS,
    RECOVERY_FAILED
};

/**
 * @brief GPU框架事件
 */
struct FrameworkEvent {
    FrameworkEventType type;
    std::string message;
    int deviceId = -1;
    boost::chrono::steady_clock::time_point timestamp;
    std::map<std::string, std::string> details;
};

/**
 * @brief 框架事件回调
 */
typedef std::function<void(const FrameworkEvent&)> FrameworkEventCallback;

/**
 * @brief OSCEAN统一GPU框架
 */
class OSCEANGPUFramework {
public:
    /**
     * @brief 获取框架单例实例
     */
    static OSCEANGPUFramework& getInstance();
    
    /**
     * @brief 初始化GPU框架
     * @param config 框架配置
     * @return 是否成功初始化
     */
    static bool initialize(const GPUFrameworkConfig& config = GPUFrameworkConfig());
    
    /**
     * @brief 关闭GPU框架
     */
    static void shutdown();
    
    /**
     * @brief 销毁单例实例
     * 
     * 在程序退出前调用此方法以确保正确清理
     */
    static void destroy();
    
    /**
     * @brief 检查框架是否已初始化
     */
    static bool isInitialized();
    
    /**
     * @brief 获取框架状态
     */
    static GPUFrameworkStatus getStatus();
    
    /**
     * @brief 获取GPU管理器
     */
    static UnifiedGPUManager& getGPUManager();
    
    /**
     * @brief 获取调度器
     */
    static MultiGPUScheduler& getScheduler();
    
    /**
     * @brief 获取内存管理器
     */
    static MultiGPUMemoryManager& getMemoryManager();
    
    /**
     * @brief 获取可用设备列表
     */
    static const std::vector<GPUDeviceInfo>& getAvailableDevices();
    
    /**
     * @brief 提交GPU任务
     * @param task 任务对象
     * @param priority 任务优先级
     * @return 任务ID
     */
    std::string submitTask(std::shared_ptr<IGPUTask> task,
                          GPUTaskPriority priority = GPUTaskPriority::NORMAL);
    
    /**
     * @brief 等待任务完成
     * @param taskId 任务ID
     * @param timeoutMs 超时时间（毫秒）
     * @return 是否成功完成
     */
    bool waitForTask(const std::string& taskId, int timeoutMs = -1);
    
    /**
     * @brief 取消任务
     * @param taskId 任务ID
     * @return 是否成功取消
     */
    bool cancelTask(const std::string& taskId);
    
    /**
     * @brief 获取任务状态
     * @param taskId 任务ID
     * @return 任务执行上下文
     */
    boost::optional<GPUTaskExecutionContext> getTaskStatus(const std::string& taskId) const;
    
    /**
     * @brief 分配GPU内存
     * @param size 内存大小
     * @param deviceId 设备ID（-1表示自动选择）
     * @return 内存句柄
     */
    GPUMemoryHandle allocateMemory(size_t size, int deviceId = -1);
    
    /**
     * @brief 释放GPU内存
     * @param handle 内存句柄
     */
    void deallocateMemory(const GPUMemoryHandle& handle);
    
    /**
     * @brief 传输数据
     * @param source 源内存
     * @param destination 目标内存
     * @param async 是否异步传输
     * @return 是否成功启动传输
     */
    bool transferData(const GPUMemoryHandle& source,
                     const GPUMemoryHandle& destination,
                     bool async = true);
    
    /**
     * @brief 注册事件回调
     * @param callback 事件回调函数
     */
    void registerEventCallback(FrameworkEventCallback callback);
    
    /**
     * @brief 设置性能监控回调
     * @param callback 监控回调函数
     */
    void setPerformanceCallback(std::function<void(const GPUFrameworkStatus&)> callback);
    
    /**
     * @brief 手动触发设备检测
     */
    void refreshDevices();
    
    /**
     * @brief 获取推荐的设备配置
     * @param taskType 任务类型
     * @return 推荐的设备ID列表
     */
    std::vector<int> getRecommendedDevices(const std::string& taskType) const;
    
    /**
     * @brief 优化框架配置
     * @param workloadProfile 工作负载特征
     */
    void optimizeConfiguration(const std::string& workloadProfile);
    
    /**
     * @brief 导出性能报告
     * @return 性能报告字符串
     */
    std::string generatePerformanceReport() const;
    
    /**
     * @brief 重置框架状态
     */
    void reset();
    
    /**
     * @brief 析构函数
     */
    ~OSCEANGPUFramework();
    
private:
    /**
     * @brief 构造函数 (私有，单例模式)
     */
    OSCEANGPUFramework();
    
    // 禁止拷贝
    OSCEANGPUFramework(const OSCEANGPUFramework&) = delete;
    OSCEANGPUFramework& operator=(const OSCEANGPUFramework&) = delete;
    
    // 私有实现
    class Impl;
    std::unique_ptr<Impl> m_impl;
    
    // 单例相关
    static std::unique_ptr<OSCEANGPUFramework> s_instance;
    static boost::mutex s_mutex;
    static boost::once_flag s_onceFlag;
    
    static void createInstance();
};

/**
 * @brief GPU任务构建器
 */
class GPUTaskBuilder {
public:
    GPUTaskBuilder& withName(const std::string& name);
    GPUTaskBuilder& withMemoryRequirement(size_t size);
    GPUTaskBuilder& withComputeComplexity(double complexity);
    GPUTaskBuilder& withExecutor(std::function<bool(GPUTaskExecutionContext&)> executor);
    GPUTaskBuilder& withValidator(std::function<bool(const GPUTaskExecutionContext&)> validator);
    
    std::shared_ptr<IGPUTask> build();
    
private:
    class SimpleGPUTask;
    std::shared_ptr<SimpleGPUTask> m_task;
};

/**
 * @brief GPU框架工具函数
 */
namespace GPUFrameworkUtils {
    /**
     * @brief 检查GPU支持
     */
    bool checkGPUSupport();
    
    /**
     * @brief 获取最佳GPU配置建议
     */
    GPUFrameworkConfig getBestConfiguration();
    
    /**
     * @brief 验证框架配置
     */
    bool validateConfiguration(const GPUFrameworkConfig& config);
    
    /**
     * @brief 格式化设备信息
     */
    std::string formatDeviceInfo(const GPUDeviceInfo& device);
    
    /**
     * @brief 估算任务执行时间
     */
    double estimateExecutionTime(const IGPUTask& task, int deviceId);
}

} // namespace gpu
} // namespace common_utils
} // namespace oscean 