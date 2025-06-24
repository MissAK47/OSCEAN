#pragma once

/**
 * @file memory_statistics.h
 * @brief 内存统计和监控模块 - 集中统计管理
 * 
 * 重构目标：
 * ✅ 集中管理所有内存统计功能
 * ✅ 提供性能监控和分析
 * ✅ 支持内存压力检测
 * ✅ 兼容现有统计接口
 */

#include "memory_config.h"
#include "memory_interfaces.h"
#include <atomic>
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <functional>
#include <future>
#include <unordered_map>

namespace oscean::common_utils::memory {

/**
 * @brief 内存统计收集器
 */
class MemoryStatisticsCollector {
public:
    /**
     * @brief 分配事件记录
     */
    struct AllocationEvent {
        std::chrono::steady_clock::time_point timestamp;
        size_t size;
        size_t alignment;
        MemoryPoolType poolType;
        std::string tag;
        void* address;
        bool success;
        
        AllocationEvent(size_t s, size_t a, MemoryPoolType pt, const std::string& t, void* addr, bool succ)
            : timestamp(std::chrono::steady_clock::now()), size(s), alignment(a), 
              poolType(pt), tag(t), address(addr), success(succ) {}
    };
    
    /**
     * @brief 释放事件记录
     */
    struct DeallocationEvent {
        std::chrono::steady_clock::time_point timestamp;
        void* address;
        size_t size;
        MemoryPoolType poolType;
        
        DeallocationEvent(void* addr, size_t s, MemoryPoolType pt)
            : timestamp(std::chrono::steady_clock::now()), address(addr), size(s), poolType(pt) {}
    };
    
    MemoryStatisticsCollector() = default;
    ~MemoryStatisticsCollector() = default;
    
    /**
     * @brief 记录分配事件
     */
    void recordAllocation(size_t size, const std::string& tag = "");
    
    /**
     * @brief 记录释放事件
     */
    void recordDeallocation(size_t size, const std::string& tag = "");
    
    /**
     * @brief 获取统计汇总
     */
    MemoryUsageStats getOverallStats() const;
    
    /**
     * @brief 重置统计
     */
    void reset();
    
    /**
     * @brief 获取分配历史
     */
    std::vector<AllocationEvent> getAllocationHistory(size_t maxEvents = 1000) const;
    
    /**
     * @brief 获取释放历史
     */
    std::vector<DeallocationEvent> getDeallocationHistory(size_t maxEvents = 1000) const;
    
    /**
     * @brief 检测内存泄漏
     */
    std::vector<AllocationEvent> detectMemoryLeaks() const;
    
    /**
     * @brief 计算内存碎片率
     */
    double calculateFragmentationRatio() const;
    
    /**
     * @brief 获取分配时间统计
     */
    struct TimingStats {
        double averageAllocationTime;
        double averageDeallocationTime;
        double maxAllocationTime;
        double maxDeallocationTime;
        double minAllocationTime;
        double minDeallocationTime;
    };
    
    TimingStats getTimingStatistics() const;

    /**
     * @brief 生成报告
     */
    std::string generateReport() const;

private:
    mutable std::mutex statsMutex_;
    
    MemoryUsageStats overallStats_;
    
    // 事件历史 (环形缓冲区)
    static constexpr size_t MAX_EVENTS = 10000;
    std::vector<AllocationEvent> allocationEvents_;
    std::vector<DeallocationEvent> deallocationEvents_;
    size_t allocationEventIndex_{0};
    size_t deallocationEventIndex_{0};
    
    // 活跃分配跟踪
    std::map<void*, AllocationEvent> activeAllocations_;
    
    // 时间统计
    mutable TimingStats timingStats_{};
    
    void updateGlobalStats(size_t size, bool allocation);
    void updatePoolStats(MemoryPoolType poolType, size_t size, bool allocation);
    void updateTimingStats(const AllocationEvent& event);
    void updateTimingStats(const DeallocationEvent& event);
};

/**
 * @brief 内存压力监控器
 */
class MemoryPressureMonitor {
public:
    /**
     * @brief 内存压力级别
     */
    enum class PressureLevel { LOW, MEDIUM, HIGH, CRITICAL };
    
    /**
     * @brief 压力阈值配置
     */
    struct PressureThresholds {
        size_t mediumThresholdMB = 150;    // 中等压力阈值
        size_t highThresholdMB = 200;      // 高压力阈值  
        size_t criticalThresholdMB = 240;  // 临界压力阈值
        double fragmentationWarning = 0.3; // 碎片化警告阈值
        double fragmentationCritical = 0.5; // 碎片化临界阈值
    };
    
    explicit MemoryPressureMonitor(const PressureThresholds& thresholds = PressureThresholds{});
    ~MemoryPressureMonitor() = default;
    
    /**
     * @brief 获取当前压力级别
     */
    PressureLevel getCurrentPressure() const noexcept;
    
    /**
     * @brief 更新内存使用情况
     */
    void updateMemoryUsage(const MemoryUsageStats& stats);
    
    /**
     * @brief 设置压力回调
     */
    void setPressureCallback(std::function<void(PressureLevel)> callback);
    
    /**
     * @brief 获取压力历史
     */
    struct PressureHistory {
        std::chrono::steady_clock::time_point timestamp;
        PressureLevel level;
        size_t memoryUsageMB;
        double fragmentationRatio;
    };
    
    std::vector<PressureHistory> getPressureHistory(size_t maxEntries = 100) const;
    
    /**
     * @brief 获取建议操作
     */
    enum class RecommendedAction {
        NONE,
        GARBAGE_COLLECT,
        DEFRAGMENT,
        REDUCE_ALLOCATION,
        EMERGENCY_CLEANUP
    };
    
    RecommendedAction getRecommendedAction() const noexcept;
    
    /**
     * @brief 设置阈值
     */
    void setThresholds(const PressureThresholds& thresholds);
    
    /**
     * @brief 获取当前阈值
     */
    const PressureThresholds& getThresholds() const noexcept;

private:
    PressureThresholds thresholds_;
    mutable std::atomic<PressureLevel> currentPressure_{PressureLevel::LOW};
    std::function<void(PressureLevel)> pressureCallback_;
    
    mutable std::mutex historyMutex_;
    std::vector<PressureHistory> pressureHistory_;
    static constexpr size_t MAX_HISTORY = 1000;
    size_t historyIndex_{0};
    
    PressureLevel calculatePressureLevel(const MemoryUsageStats& stats) const noexcept;
    void recordPressure(PressureLevel level, const MemoryUsageStats& stats);
};

/**
 * @brief 性能分析器
 */
class MemoryPerformanceAnalyzer {
public:
    /**
     * @brief 性能指标
     */
    struct PerformanceMetrics {
        double allocationThroughput;    // 分配吞吐量 (MB/s)
        double deallocationThroughput;  // 释放吞吐量 (MB/s)
        double averageLatency;          // 平均延迟 (ms)
        double peakLatency;             // 峰值延迟 (ms)
        double cacheHitRatio;           // 缓存命中率
        double fragmentationLevel;      // 碎片化程度
        size_t activeAllocations;       // 活跃分配数
        size_t totalMemoryFootprint;    // 总内存占用
    };
    
    MemoryPerformanceAnalyzer() = default;
    ~MemoryPerformanceAnalyzer() = default;
    
    /**
     * @brief 分析性能指标
     */
    PerformanceMetrics analyzePerformance(const MemoryUsageStats& stats) const;
    
    /**
     * @brief 生成性能报告
     */
    std::string generatePerformanceReport(const PerformanceMetrics& metrics) const;
    
    /**
     * @brief 检测性能瓶颈
     */
    enum class BottleneckType {
        NONE,
        HIGH_FRAGMENTATION,
        SLOW_ALLOCATION,
        MEMORY_PRESSURE,
        CACHE_MISSES,
        THREAD_CONTENTION
    };
    
    std::vector<BottleneckType> detectBottlenecks(const PerformanceMetrics& metrics) const;
    
    /**
     * @brief 获取优化建议
     */
    struct OptimizationSuggestion {
        std::string description;
        BottleneckType targetBottleneck;
        double expectedImprovement;
        bool isAutoApplicable;
    };
    
    std::vector<OptimizationSuggestion> getOptimizationSuggestions(
        const std::vector<BottleneckType>& bottlenecks
    ) const;
    
    /**
     * @brief 基准测试
     */
    struct BenchmarkResult {
        double allocationSpeed;     // 分配速度 (ops/sec)
        double deallocationSpeed;   // 释放速度 (ops/sec)
        double memoryEfficiency;    // 内存效率 (0-1)
        double threadSafety;        // 线程安全性能 (0-1)
    };
    
    BenchmarkResult runBenchmark(size_t iterations = 10000) const;

private:
    // 性能基线值
    static constexpr double BASELINE_ALLOCATION_SPEED = 1000000.0; // 1M ops/sec
    static constexpr double BASELINE_DEALLOCATION_SPEED = 1200000.0; // 1.2M ops/sec
    static constexpr double ACCEPTABLE_LATENCY = 0.001; // 1ms
    static constexpr double ACCEPTABLE_FRAGMENTATION = 0.1; // 10%
    
    double calculateThroughput(size_t bytes, double timeSeconds) const noexcept;
    double calculateEfficiency(const PerformanceMetrics& metrics) const noexcept;
};

} // namespace oscean::common_utils::memory 