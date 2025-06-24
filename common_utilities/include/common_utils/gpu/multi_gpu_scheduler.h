/**
 * @file multi_gpu_scheduler.h
 * @brief 多GPU负载均衡调度器接口
 */

#pragma once

#include "gpu_device_info.h"
#include <boost/thread/mutex.hpp>
#include <boost/atomic.hpp>
#include <boost/chrono.hpp>
#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <vector>
#include <queue>
#include <map>

namespace oscean::common_utils::gpu {

/**
 * @brief GPU任务优先级
 */
enum class GPUTaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief GPU任务信息
 */
struct GPUTaskInfo {
    std::string taskId;                    // 任务唯一标识
    size_t memoryRequirement;              // 内存需求（字节）
    double computeComplexity;              // 计算复杂度（0.0-1.0）
    GPUTaskPriority priority;              // 任务优先级
    boost::optional<int> preferredDeviceId; // 偏好的设备ID
    boost::chrono::milliseconds estimatedDuration; // 预估执行时间
    
    GPUTaskInfo() : memoryRequirement(0), computeComplexity(0.5), 
                   priority(GPUTaskPriority::NORMAL) {}
};

/**
 * @brief GPU负载信息
 */
struct GPUWorkload {
    int deviceId;
    GPUDeviceInfo deviceInfo;
    boost::atomic<float> currentLoad{0.0f};        // 当前负载（0.0-1.0）
    boost::atomic<size_t> queuedTasks{0};          // 排队任务数
    boost::atomic<size_t> runningTasks{0};         // 运行中任务数
    boost::atomic<size_t> completedTasks{0};       // 已完成任务数
    boost::atomic<size_t> failedTasks{0};          // 失败任务数
    boost::chrono::steady_clock::time_point lastUpdate;
    boost::atomic<size_t> allocatedMemory{0};      // 已分配内存
    
    // 性能统计
    boost::atomic<double> avgTaskDuration{0.0};    // 平均任务时长（毫秒）
    boost::atomic<double> throughput{0.0};         // 吞吐量（任务/秒）
};

/**
 * @brief GPU负载信息快照（可拷贝）
 */
struct GPUWorkloadSnapshot {
    int deviceId;
    GPUDeviceInfo deviceInfo;
    float currentLoad;
    size_t queuedTasks;
    size_t runningTasks;
    size_t completedTasks;
    size_t failedTasks;
    boost::chrono::steady_clock::time_point lastUpdate;
    size_t allocatedMemory;
    double avgTaskDuration;
    double throughput;
    
    // 从GPUWorkload创建快照
    explicit GPUWorkloadSnapshot(const GPUWorkload& workload)
        : deviceId(workload.deviceId)
        , deviceInfo(workload.deviceInfo)
        , currentLoad(workload.currentLoad.load())
        , queuedTasks(workload.queuedTasks.load())
        , runningTasks(workload.runningTasks.load())
        , completedTasks(workload.completedTasks.load())
        , failedTasks(workload.failedTasks.load())
        , lastUpdate(workload.lastUpdate)
        , allocatedMemory(workload.allocatedMemory.load())
        , avgTaskDuration(workload.avgTaskDuration.load())
        , throughput(workload.throughput.load()) {}
};

/**
 * @brief 调度策略
 */
enum class SchedulingStrategy {
    ROUND_ROBIN,           // 轮询调度
    LEAST_LOADED,         // 最低负载优先
    PERFORMANCE_BASED,    // 基于性能评分
    MEMORY_AWARE,         // 内存感知调度
    AFFINITY_BASED,       // 亲和性调度
    POWER_EFFICIENT       // 功耗优化调度
};

/**
 * @brief 调度器配置
 */
struct SchedulerConfig {
    SchedulingStrategy strategy = SchedulingStrategy::LEAST_LOADED;
    bool enableDynamicBalancing = true;      // 启用动态负载均衡
    bool enableTaskMigration = false;        // 启用任务迁移
    float loadThreshold = 0.8f;              // 负载阈值
    float memoryThreshold = 0.9f;            // 内存使用阈值
    boost::chrono::milliseconds updateInterval{100}; // 状态更新间隔
    size_t maxQueuedTasksPerDevice = 100;    // 每设备最大排队任务数
    bool preferHighPerformanceDevices = true; // 优先使用高性能设备
};

/**
 * @brief 调度决策结果
 */
struct SchedulingDecision {
    int selectedDeviceId;                    // 选中的设备ID
    float confidenceScore;                   // 决策置信度（0.0-1.0）
    std::string reason;                      // 决策原因
    std::vector<int> alternativeDevices;     // 备选设备列表
    
    SchedulingDecision() : selectedDeviceId(-1), confidenceScore(0.0f) {}
};

/**
 * @brief 调度器事件类型
 */
enum class SchedulerEventType {
    TASK_SCHEDULED,
    TASK_STARTED,
    TASK_COMPLETED,
    TASK_FAILED,
    DEVICE_OVERLOADED,
    DEVICE_RECOVERED,
    LOAD_BALANCED
};

/**
 * @brief 调度器事件
 */
struct SchedulerEvent {
    SchedulerEventType type;
    int deviceId;
    std::string taskId;
    std::string message;
    boost::chrono::steady_clock::time_point timestamp;
};

/**
 * @brief 调度器事件回调
 */
typedef boost::function<void(const SchedulerEvent&)> SchedulerEventCallback;

/**
 * @brief 调度器统计信息
 */
struct SchedulerStatistics {
    size_t totalTasksScheduled = 0;      // 总调度任务数
    size_t completedTasks = 0;           // 完成任务数
    size_t failedTasks = 0;              // 失败任务数
    size_t runningTasks = 0;             // 运行中任务数
    size_t queuedTasks = 0;              // 排队任务数
    double averageWaitTime = 0.0;        // 平均等待时间（毫秒）
    double averageExecutionTime = 0.0;   // 平均执行时间（毫秒）
    double loadBalanceEfficiency = 0.0;  // 负载均衡效率（0.0-1.0）
};

/**
 * @brief 多GPU负载均衡调度器
 */
class MultiGPUScheduler {
public:
    /**
     * @brief 构造函数
     * @param devices GPU设备列表
     * @param config 调度器配置
     */
    MultiGPUScheduler(const std::vector<GPUDeviceInfo>& devices, 
                     const SchedulerConfig& config = SchedulerConfig());
    
    /**
     * @brief 析构函数
     */
    ~MultiGPUScheduler();
    
    /**
     * @brief 根据任务需求选择最优GPU
     * @param taskInfo 任务信息
     * @return 调度决策
     */
    SchedulingDecision selectOptimalGPU(const GPUTaskInfo& taskInfo);
    
    /**
     * @brief 提交任务到指定GPU
     * @param deviceId GPU设备ID
     * @param taskInfo 任务信息
     * @return 是否成功提交
     */
    bool submitTask(int deviceId, const GPUTaskInfo& taskInfo);
    
    /**
     * @brief 任务开始执行
     * @param deviceId GPU设备ID
     * @param taskId 任务ID
     */
    void taskStarted(int deviceId, const std::string& taskId);
    
    /**
     * @brief 任务完成
     * @param deviceId GPU设备ID
     * @param taskId 任务ID
     * @param executionTime 实际执行时间
     */
    void taskCompleted(int deviceId, const std::string& taskId, 
                      boost::chrono::milliseconds executionTime);
    
    /**
     * @brief 任务失败
     * @param deviceId GPU设备ID
     * @param taskId 任务ID
     * @param reason 失败原因
     */
    void taskFailed(int deviceId, const std::string& taskId, const std::string& reason);
    
    /**
     * @brief 更新GPU内存使用情况
     * @param deviceId GPU设备ID
     * @param usedMemory 已使用内存
     */
    void updateMemoryUsage(int deviceId, size_t usedMemory);
    
    /**
     * @brief 获取GPU负载信息
     * @param deviceId GPU设备ID
     * @return GPU负载信息
     */
    boost::optional<GPUWorkloadSnapshot> getWorkload(int deviceId) const;
    
    /**
     * @brief 获取所有GPU负载信息
     * @return GPU负载信息列表
     */
    std::vector<GPUWorkloadSnapshot> getAllWorkloads() const;
    
    /**
     * @brief 获取调度统计信息
     * @return 统计信息字符串
     */
    std::string getStatistics() const;
    
    /**
     * @brief 获取调度统计数据
     * @return 统计数据结构
     */
    SchedulerStatistics getStatisticsData() const;
    
    /**
     * @brief 注册事件回调
     * @param callback 事件回调函数
     */
    void registerEventCallback(SchedulerEventCallback callback);
    
    /**
     * @brief 设置调度策略
     * @param strategy 新的调度策略
     */
    void setSchedulingStrategy(SchedulingStrategy strategy);
    
    /**
     * @brief 获取当前调度策略
     * @return 当前调度策略
     */
    SchedulingStrategy getSchedulingStrategy() const;
    
    /**
     * @brief 执行负载均衡
     * @return 是否执行了负载均衡操作
     */
    bool performLoadBalancing();
    
    /**
     * @brief 重置调度器状态
     */
    void reset();
    
private:
    // 私有实现
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief 全局多GPU调度器管理
 */
class GlobalSchedulerManager {
public:
    /**
     * @brief 获取全局调度器实例
     * @return 调度器实例
     */
    static MultiGPUScheduler& getInstance();
    
    /**
     * @brief 初始化全局调度器
     * @param devices GPU设备列表
     * @param config 调度器配置
     */
    static void initialize(const std::vector<GPUDeviceInfo>& devices,
                          const SchedulerConfig& config = SchedulerConfig());
    
    /**
     * @brief 销毁全局调度器
     */
    static void destroy();
    
private:
    static std::unique_ptr<MultiGPUScheduler> s_instance;
    static boost::mutex s_mutex;
};

} // namespace oscean::common_utils::gpu 