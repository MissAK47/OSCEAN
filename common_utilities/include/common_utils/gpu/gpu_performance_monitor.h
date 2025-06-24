/**
 * @file gpu_performance_monitor.h
 * @brief GPU性能监控器接口定义
 * 
 * 提供GPU性能监控、分析和优化建议功能
 */

#pragma once

#include "gpu_common.h"
#include "gpu_device_info.h"
#include <boost/circular_buffer.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/chrono.hpp>
#include <boost/optional.hpp>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace oscean::common_utils::gpu {

/**
 * @brief GPU性能指标
 */
struct GPUPerformanceMetrics {
    // 设备利用率
    struct Utilization {
        float gpu = 0.0f;           // GPU核心利用率 (0-100%)
        float memory = 0.0f;        // 内存控制器利用率 (0-100%)
        float encoder = 0.0f;       // 编码器利用率 (0-100%)
        float decoder = 0.0f;       // 解码器利用率 (0-100%)
    } utilization;
    
    // 内存信息
    struct Memory {
        size_t total = 0;           // 总内存 (bytes)
        size_t used = 0;            // 已用内存 (bytes)
        size_t free = 0;            // 空闲内存 (bytes)
        float bandwidth = 0.0f;     // 内存带宽利用率 (0-100%)
    } memory;
    
    // 温度和功耗
    struct ThermalPower {
        int temperature = 0;        // GPU温度 (摄氏度)
        float power = 0.0f;         // 功耗 (瓦特)
        float powerLimit = 0.0f;    // 功耗限制 (瓦特)
        int fanSpeed = 0;           // 风扇转速 (0-100%)
    } thermal;
    
    // 时钟频率
    struct Clocks {
        int graphics = 0;           // 图形时钟 (MHz)
        int memory = 0;             // 内存时钟 (MHz)
        int sm = 0;                 // SM时钟 (MHz)
    } clocks;
    
    // 时间戳
    boost::chrono::steady_clock::time_point timestamp;
};

/**
 * @brief 核函数性能数据
 */
struct KernelPerformanceData {
    std::string kernelName;                    // 核函数名称
    int deviceId = 0;                          // 设备ID
    
    // 执行时间
    struct Timing {
        double kernelTime = 0.0;               // 核函数执行时间 (ms)
        double memcpyH2D = 0.0;                // 主机到设备传输时间 (ms)
        double memcpyD2H = 0.0;                // 设备到主机传输时间 (ms)
        double totalTime = 0.0;                // 总时间 (ms)
    } timing;
    
    // 内存传输
    struct MemoryTransfer {
        size_t bytesH2D = 0;                   // 主机到设备传输字节数
        size_t bytesD2H = 0;                   // 设备到主机传输字节数
        double bandwidthH2D = 0.0;             // H2D带宽 (GB/s)
        double bandwidthD2H = 0.0;             // D2H带宽 (GB/s)
    } memory;
    
    // 占用率和效率
    struct Efficiency {
        float occupancy = 0.0f;                // 占用率 (0-100%)
        float smEfficiency = 0.0f;             // SM效率 (0-100%)
        float memoryEfficiency = 0.0f;         // 内存效率 (0-100%)
        size_t registersPerThread = 0;         // 每线程寄存器数
        size_t sharedMemoryPerBlock = 0;       // 每块共享内存
    } efficiency;
    
    // 工作负载
    struct Workload {
        size_t gridSize[3] = {0, 0, 0};        // Grid维度
        size_t blockSize[3] = {0, 0, 0};       // Block维度
        size_t totalThreads = 0;               // 总线程数
    } workload;
    
    boost::chrono::steady_clock::time_point timestamp;
};

/**
 * @brief 性能事件类型
 */
enum class PerformanceEventType {
    KERNEL_LAUNCH,          // 核函数启动
    KERNEL_COMPLETE,        // 核函数完成
    MEMORY_TRANSFER,        // 内存传输
    ALLOCATION,             // 内存分配
    DEALLOCATION,           // 内存释放
    SYNC_POINT,             // 同步点
    BOTTLENECK_DETECTED,    // 检测到瓶颈
    OPTIMIZATION_HINT       // 优化建议
};

/**
 * @brief 性能事件
 */
struct PerformanceEvent {
    PerformanceEventType type;
    std::string description;
    int deviceId;
    boost::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> metadata;
};

/**
 * @brief 性能监控器接口
 */
class IGPUPerformanceMonitor {
public:
    virtual ~IGPUPerformanceMonitor() = default;
    
    /**
     * @brief 开始监控
     * @param deviceIds 要监控的设备ID列表（空表示所有设备）
     * @return 是否成功开始
     */
    virtual bool startMonitoring(const std::vector<int>& deviceIds = {}) = 0;
    
    /**
     * @brief 停止监控
     */
    virtual void stopMonitoring() = 0;
    
    /**
     * @brief 是否正在监控
     * @return 监控状态
     */
    virtual bool isMonitoring() const = 0;
    
    /**
     * @brief 获取设备性能指标
     * @param deviceId 设备ID
     * @return 性能指标
     */
    virtual GPUPerformanceMetrics getDeviceMetrics(int deviceId) const = 0;
    
    /**
     * @brief 获取所有设备的性能指标
     * @return 设备ID到性能指标的映射
     */
    virtual std::unordered_map<int, GPUPerformanceMetrics> getAllMetrics() const = 0;
    
    /**
     * @brief 记录核函数性能
     * @param data 核函数性能数据
     */
    virtual void recordKernelPerformance(const KernelPerformanceData& data) = 0;
    
    /**
     * @brief 获取核函数性能历史
     * @param kernelName 核函数名称（空表示所有）
     * @param maxRecords 最大记录数
     * @return 性能数据列表
     */
    virtual std::vector<KernelPerformanceData> getKernelHistory(
        const std::string& kernelName = "",
        size_t maxRecords = 100) const = 0;
    
    /**
     * @brief 记录性能事件
     * @param event 性能事件
     */
    virtual void recordEvent(const PerformanceEvent& event) = 0;
    
    /**
     * @brief 获取性能事件历史
     * @param type 事件类型（可选）
     * @param maxRecords 最大记录数
     * @return 事件列表
     */
    virtual std::vector<PerformanceEvent> getEventHistory(
        boost::optional<PerformanceEventType> type = boost::none,
        size_t maxRecords = 100) const = 0;
};

/**
 * @brief 性能分析器
 * 
 * 分析性能数据并提供优化建议
 */
class GPUPerformanceAnalyzer {
public:
    /**
     * @brief 瓶颈类型
     */
    enum class BottleneckType {
        NONE,                   // 无瓶颈
        COMPUTE_BOUND,          // 计算受限
        MEMORY_BOUND,           // 内存受限
        LATENCY_BOUND,          // 延迟受限
        THERMAL_THROTTLE,       // 温度限制
        POWER_THROTTLE          // 功耗限制
    };
    
    /**
     * @brief 优化建议
     */
    struct OptimizationHint {
        std::string category;           // 建议类别
        std::string description;        // 建议描述
        int priority;                   // 优先级 (1-10, 10最高)
        double expectedImprovement;     // 预期改进 (%)
    };
    
    /**
     * @brief 分析结果
     */
    struct AnalysisResult {
        BottleneckType primaryBottleneck = BottleneckType::NONE;
        BottleneckType secondaryBottleneck = BottleneckType::NONE;
        std::vector<OptimizationHint> hints;
        double overallEfficiency = 0.0;  // 整体效率 (0-100%)
        
        // 详细分析
        struct DetailedAnalysis {
            double computeUtilization = 0.0;
            double memoryUtilization = 0.0;
            double transferEfficiency = 0.0;
            double kernelEfficiency = 0.0;
            double occupancyRatio = 0.0;
        } details;
    };
    
    /**
     * @brief 分析核函数性能
     * @param kernelData 核函数性能数据
     * @param deviceInfo GPU设备信息
     * @return 分析结果
     */
    static AnalysisResult analyzeKernelPerformance(
        const KernelPerformanceData& kernelData,
        const GPUDeviceInfo& deviceInfo);
    
    /**
     * @brief 分析设备性能
     * @param metrics 设备性能指标
     * @param history 历史数据（用于趋势分析）
     * @return 分析结果
     */
    static AnalysisResult analyzeDevicePerformance(
        const GPUPerformanceMetrics& metrics,
        const std::vector<GPUPerformanceMetrics>& history = {});
    
    /**
     * @brief 生成性能报告
     * @param monitor 性能监控器
     * @param deviceId 设备ID
     * @return 性能报告（Markdown格式）
     */
    static std::string generatePerformanceReport(
        const IGPUPerformanceMonitor& monitor,
        int deviceId);
};

/**
 * @brief 性能监控器工厂
 */
class GPUPerformanceMonitorFactory {
public:
    /**
     * @brief 创建性能监控器
     * @param api GPU API类型
     * @return 性能监控器实例
     */
    static std::unique_ptr<IGPUPerformanceMonitor> createMonitor(ComputeAPI api);
    
    /**
     * @brief 创建最优性能监控器
     * @param device GPU设备信息
     * @return 性能监控器实例
     */
    static std::unique_ptr<IGPUPerformanceMonitor> createOptimalMonitor(
        const GPUDeviceInfo& device);
};

/**
 * @brief 性能计时器
 * 
 * RAII风格的性能计时工具
 */
class GPUPerformanceTimer {
public:
    /**
     * @brief 构造函数
     * @param kernelName 核函数名称
     * @param monitor 性能监控器（可选）
     */
    explicit GPUPerformanceTimer(
        const std::string& kernelName,
        IGPUPerformanceMonitor* monitor = nullptr);
    
    /**
     * @brief 析构函数 - 自动记录性能
     */
    ~GPUPerformanceTimer();
    
    /**
     * @brief 设置核函数执行时间
     * @param ms 毫秒
     */
    void setKernelTime(double ms);
    
    /**
     * @brief 设置内存传输信息
     * @param bytesH2D 主机到设备字节数
     * @param bytesD2H 设备到主机字节数
     * @param timeH2D H2D时间(ms)
     * @param timeD2H D2H时间(ms)
     */
    void setMemoryTransfer(size_t bytesH2D, size_t bytesD2H,
                          double timeH2D, double timeD2H);
    
    /**
     * @brief 设置工作负载信息
     * @param gridSize Grid维度
     * @param blockSize Block维度
     */
    void setWorkload(const size_t gridSize[3], const size_t blockSize[3]);
    
private:
    std::string m_kernelName;
    IGPUPerformanceMonitor* m_monitor;
    boost::chrono::steady_clock::time_point m_startTime;
    std::unique_ptr<KernelPerformanceData> m_data;
};

/**
 * @brief 性能监控回调
 */
using PerformanceCallback = std::function<void(const GPUPerformanceMetrics&)>;
using KernelCallback = std::function<void(const KernelPerformanceData&)>;
using EventCallback = std::function<void(const PerformanceEvent&)>;

/**
 * @brief 全局性能监控管理器
 */
class GlobalPerformanceManager {
public:
    static GlobalPerformanceManager& getInstance() {
        static GlobalPerformanceManager instance;
        return instance;
    }
    
    /**
     * @brief 设置全局性能监控器
     * @param monitor 性能监控器
     */
    void setGlobalMonitor(std::shared_ptr<IGPUPerformanceMonitor> monitor);
    
    /**
     * @brief 获取全局性能监控器
     * @return 性能监控器
     */
    std::shared_ptr<IGPUPerformanceMonitor> getGlobalMonitor() const;
    
    /**
     * @brief 注册性能回调
     * @param callback 回调函数
     * @return 回调ID（用于注销）
     */
    size_t registerPerformanceCallback(PerformanceCallback callback);
    
    /**
     * @brief 注销性能回调
     * @param callbackId 回调ID
     */
    void unregisterPerformanceCallback(size_t callbackId);
    
private:
    GlobalPerformanceManager() = default;
    ~GlobalPerformanceManager() = default;
    GlobalPerformanceManager(const GlobalPerformanceManager&) = delete;
    GlobalPerformanceManager& operator=(const GlobalPerformanceManager&) = delete;
    
    mutable boost::mutex m_mutex;
    std::shared_ptr<IGPUPerformanceMonitor> m_globalMonitor;
    std::unordered_map<size_t, PerformanceCallback> m_callbacks;
    size_t m_nextCallbackId = 1;
};

} // namespace oscean::common_utils::gpu 