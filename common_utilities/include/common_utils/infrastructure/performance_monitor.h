/**
 * @file performance_monitor.h
 * @brief 统一性能监控器 - 消除8个模块重复监控实现
 * 
 * 🎯 重构目标：
 * ✅ 统一各模块的性能监控，消除8处重复实现
 * ✅ 提供实时性能统计和预警
 * ✅ 支持多层级监控（系统、模块、任务）
 * ✅ 集成优化建议和自动调整
 */

#pragma once

#include "../utilities/boost_config.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <functional>
#include <shared_mutex>

namespace oscean::common_utils::infrastructure {

/**
 * @brief 性能指标类型
 */
enum class MetricType {
    TIMING,          // 时间统计
    MEMORY,          // 内存使用
    THROUGHPUT,      // 吞吐量
    LATENCY,         // 延迟
    CACHE_HIT_RATE,  // 缓存命中率
    CPU_USAGE,       // CPU使用率
    CUSTOM           // 自定义指标
};

/**
 * @brief 监控级别
 */
enum class MonitorLevel {
    SYSTEM,          // 系统级监控
    MODULE,          // 模块级监控
    COMPONENT,       // 组件级监控
    FUNCTION,        // 函数级监控
    TASK             // 任务级监控
};

/**
 * @brief 性能指标值
 */
struct MetricValue {
    double value = 0.0;
    std::chrono::steady_clock::time_point timestamp;
    std::string unit;
    std::map<std::string, std::string> labels;
    
    MetricValue() : timestamp(std::chrono::steady_clock::now()) {}
    explicit MetricValue(double val, const std::string& u = "") 
        : value(val), timestamp(std::chrono::steady_clock::now()), unit(u) {}
};

/**
 * @brief 性能统计信息
 */
struct PerformanceStats {
    std::string metricName;
    MetricType type;
    double currentValue = 0.0;
    double averageValue = 0.0;
    double minValue = 0.0;
    double maxValue = 0.0;
    double standardDeviation = 0.0;
    size_t sampleCount = 0;
    std::chrono::steady_clock::time_point firstSample;
    std::chrono::steady_clock::time_point lastSample;
    std::string unit;
    
    double getRate() const;
    std::string toString() const;
};

/**
 * @brief 性能警报配置
 */
struct AlertConfig {
    std::string metricName;
    double threshold;
    bool greaterThan = true;  // true: >threshold, false: <threshold
    std::chrono::seconds cooldownPeriod{60};
    std::function<void(const std::string&, double)> callback;
    bool enabled = true;
};

/**
 * @brief 统一性能监控器
 */
class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    // 禁用拷贝，允许移动
    PerformanceMonitor(const PerformanceMonitor&) = delete;
    PerformanceMonitor& operator=(const PerformanceMonitor&) = delete;
    PerformanceMonitor(PerformanceMonitor&&) = default;
    PerformanceMonitor& operator=(PerformanceMonitor&&) = default;
    
    // === 监控控制 ===
    
    void startMonitoring();
    void stopMonitoring();
    void pauseMonitoring();
    void resumeMonitoring();
    bool isMonitoring() const;
    
    // === 指标记录 ===
    
    /**
     * @brief 记录时间指标
     */
    void recordTiming(const std::string& name, std::chrono::milliseconds duration);
    void recordTiming(const std::string& name, double durationMs);
    
    /**
     * @brief 记录内存指标
     */
    void recordMemoryUsage(const std::string& name, size_t bytesUsed);
    void recordMemoryPeak(const std::string& name, size_t peakBytes);
    
    /**
     * @brief 记录吞吐量指标
     */
    void recordThroughput(const std::string& name, double itemsPerSecond);
    void recordThroughput(const std::string& name, size_t itemCount, std::chrono::milliseconds duration);
    
    /**
     * @brief 记录延迟指标
     */
    void recordLatency(const std::string& name, std::chrono::microseconds latency);
    
    /**
     * @brief 记录缓存命中率
     */
    void recordCacheHitRate(const std::string& name, double hitRate);
    void recordCacheStats(const std::string& name, size_t hits, size_t misses);
    
    /**
     * @brief 记录CPU使用率
     */
    void recordCpuUsage(const std::string& name, double cpuPercent);
    
    /**
     * @brief 记录自定义指标
     */
    void recordCustomMetric(const std::string& name, double value, const std::string& unit = "");
    void recordCustomMetric(const std::string& name, const MetricValue& metric);
    
    // === 计时器工具 ===
    
    /**
     * @brief RAII计时器
     */
    class Timer {
    public:
        explicit Timer(PerformanceMonitor& monitor, const std::string& name);
        ~Timer();
        
        void stop();
        std::chrono::milliseconds getElapsed() const;
        
    private:
        PerformanceMonitor& monitor_;
        std::string name_;
        std::chrono::steady_clock::time_point startTime_;
        bool stopped_;
    };
    
    std::unique_ptr<Timer> createTimer(const std::string& name);
    
    // === 作用域计时 (便捷宏支持) ===
    
    /**
     * @brief 函数作用域计时
     */
    class ScopedTimer {
    public:
        ScopedTimer(PerformanceMonitor& monitor, const std::string& name);
        ~ScopedTimer();
    private:
        std::unique_ptr<Timer> timer_;
    };
    
    ScopedTimer createScopedTimer(const std::string& name);
    
    // === 统计查询 ===
    
    PerformanceStats getStats(const std::string& metricName) const;
    std::vector<PerformanceStats> getAllStats() const;
    std::vector<PerformanceStats> getStatsByType(MetricType type) const;
    std::vector<PerformanceStats> getStatsByLevel(MonitorLevel level) const;
    
    // === 实时监控 ===
    
    double getCurrentValue(const std::string& metricName) const;
    std::map<std::string, double> getCurrentValues() const;
    
    // === 警报系统 ===
    
    void addAlert(const AlertConfig& config);
    void removeAlert(const std::string& metricName);
    void enableAlert(const std::string& metricName);
    void disableAlert(const std::string& metricName);
    std::vector<AlertConfig> getActiveAlerts() const;
    
    // === 性能报告 ===
    
    std::string generatePerformanceReport() const;
    std::string generateSummaryReport() const;
    std::string generateDetailedReport() const;
    std::string generateAlertReport() const;
    
    // === 性能分析 ===
    
    struct PerformanceAnalysis {
        std::string componentName;
        std::vector<std::string> bottlenecks;
        std::vector<std::string> optimizationSuggestions;
        double overallScore;  // 0-100
        std::map<std::string, double> metricScores;
    };
    
    PerformanceAnalysis analyzePerformance(const std::string& componentName = "") const;
    std::vector<std::string> getOptimizationSuggestions() const;
    
    // === 基准测试支持 ===
    
    struct BenchmarkResult {
        std::string name;
        double averageTime;
        double minTime;
        double maxTime;
        double throughput;
        size_t iterations;
        std::string status;
    };
    
    void startBenchmark(const std::string& name);
    void endBenchmark(const std::string& name);
    BenchmarkResult getBenchmarkResult(const std::string& name) const;
    std::vector<BenchmarkResult> getAllBenchmarkResults() const;
    
    // === 数据导出 ===
    
    void exportToCSV(const std::string& filename) const;
    void exportToJSON(const std::string& filename) const;
    std::string exportToPrometheus() const;  // Prometheus格式
    
    // === 配置管理 ===
    
    void setUpdateInterval(std::chrono::seconds interval);
    void setMaxHistorySize(size_t maxSize);
    void setMonitorLevel(MonitorLevel level);
    void enableAutoCleanup(bool enabled);
    
    // === 内存管理 ===
    
    void clearHistory();
    void clearOldData(std::chrono::hours olderThan = std::chrono::hours{24});
    size_t getHistorySize() const;
    size_t getMemoryUsage() const;

private:
    // === 内部状态 ===
    
    std::atomic<bool> monitoring_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> shuttingDown_{false};
    
    MonitorLevel currentLevel_{MonitorLevel::MODULE};
    std::chrono::seconds updateInterval_{1};
    size_t maxHistorySize_{10000};
    bool autoCleanupEnabled_{true};
    
    // === 数据存储 ===
    
    mutable std::shared_mutex dataMutex_;
    std::map<std::string, std::vector<MetricValue>> metricHistory_;
    std::map<std::string, PerformanceStats> currentStats_;
    std::map<std::string, MetricType> metricTypes_;
    
    // === 警报系统 ===
    
    mutable std::mutex alertsMutex_;
    std::map<std::string, AlertConfig> alerts_;
    std::map<std::string, std::chrono::steady_clock::time_point> lastAlertTimes_;
    
    // === 基准测试 ===
    
    mutable std::mutex benchmarkMutex_;
    std::map<std::string, std::chrono::steady_clock::time_point> benchmarkStartTimes_;
    std::map<std::string, BenchmarkResult> benchmarkResults_;
    
    // === 后台线程 ===
    
    std::unique_ptr<std::thread> monitoringThread_;
    std::unique_ptr<std::thread> cleanupThread_;
    
    // === 内部方法 ===
    
    void monitoringLoop();
    void cleanupLoop();
    void updateStatistics();
    void checkAlerts();
    void processMetric(const std::string& name, const MetricValue& value);
    void updateStats(const std::string& name, const MetricValue& value);
    void triggerAlert(const std::string& metricName, double value);
    std::string formatMetricName(const std::string& baseName, MonitorLevel level) const;
    void cleanupOldData();
    
    // === 静态工厂方法 ===
    
    static std::unique_ptr<PerformanceMonitor> createForDevelopment();
    static std::unique_ptr<PerformanceMonitor> createForTesting();
    static std::unique_ptr<PerformanceMonitor> createForProduction();
    static std::unique_ptr<PerformanceMonitor> createForHPC();
    
    static std::unique_ptr<PerformanceMonitor> createWithConfig(
        MonitorLevel level,
        std::chrono::seconds updateInterval = std::chrono::seconds{1},
        size_t maxHistorySize = 10000
    );
};

// === 便捷宏定义 ===

#define PERF_TIMER(monitor, name) \
    auto timer_##__LINE__ = (monitor).createScopedTimer(name)

#define PERF_FUNCTION_TIMER(monitor) \
    PERF_TIMER(monitor, __FUNCTION__)

#define PERF_RECORD_TIMING(monitor, name, duration) \
    (monitor).recordTiming(name, duration)

#define PERF_RECORD_MEMORY(monitor, name, bytes) \
    (monitor).recordMemoryUsage(name, bytes)

#define PERF_BENCHMARK_START(monitor, name) \
    (monitor).startBenchmark(name)

#define PERF_BENCHMARK_END(monitor, name) \
    (monitor).endBenchmark(name)

} // namespace oscean::common_utils::infrastructure 