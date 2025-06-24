#pragma once

/**
 * @file unified_advanced_reader.h
 * @brief 统一高级数据读取器 - 完整实现版本
 */

#include "common_utils/utilities/boost_config.h"
#include "../unified_data_reader.h"
#include "core_services/common_data_types.h"

// Common Utilities 组件头文件
#include "common_utils/simd/simd_manager_unified.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/async/async_framework.h"
#include "common_utils/async/async_task.h"
#include "common_utils/cache/icache_manager.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/time/time_services.h"
#include "common_utils/utilities/logging_utils.h"

#include <memory>
#include <string>
#include <optional>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <shared_mutex>

namespace oscean::core_services::data_access::readers::impl {

/**
 * @brief 流式处理配置
 */
struct StreamingConfig {
    size_t chunkSize = 1024 * 1024;                ///< 块大小（字节）
    size_t maxConcurrentChunks = 4;                ///< 最大并发块数
    bool enableBackpressure = true;                ///< 启用背压控制
    double memoryThreshold = 0.8;                  ///< 内存阈值
    size_t bufferSize = 8 * 1024 * 1024;           ///< 缓冲区大小
    std::chrono::milliseconds chunkTimeout{5000};   ///< 块处理超时时间
    bool enableCompression = false;                 ///< 启用数据压缩
    
    /**
     * @brief 创建海洋数据专用配置
     */
    static StreamingConfig createForOceanData() {
        StreamingConfig config;
        config.chunkSize = 4 * 1024 * 1024;        // 4MB块，适合海洋栅格数据
        config.maxConcurrentChunks = 8;            // 更多并发，提高吞吐量
        config.memoryThreshold = 0.7;              // 保守的内存使用
        config.bufferSize = 32 * 1024 * 1024;      // 32MB缓冲区
        config.enableBackpressure = true;          // 必须启用背压
        config.enableCompression = true;           // 海洋数据通常可压缩
        return config;
    }
};

/**
 * @brief 缓存配置
 */
struct CacheConfig {
    size_t maxSize = 100 * 1024 * 1024;            ///< 最大缓存大小（100MB）
    bool enabled = true;                            ///< 是否启用
    std::chrono::minutes ttl{30};                  ///< 缓存生存时间
    size_t maxEntries = 1000;                      ///< 最大缓存条目数
    double evictionThreshold = 0.9;                ///< 逐出阈值
    bool enablePersistent = false;                 ///< 启用持久化缓存
    
    /**
     * @brief 创建大数据专用缓存配置
     */
    static CacheConfig createForLargeData() {
        CacheConfig config;
        config.maxSize = 500 * 1024 * 1024;        // 500MB
        config.maxEntries = 500;                   // 减少条目数，每个条目更大
        config.ttl = std::chrono::hours(2);        // 更长的缓存时间
        config.evictionThreshold = 0.8;            // 更早逐出
        config.enablePersistent = true;            // 启用持久化
        return config;
    }
};

/**
 * @brief SIMD优化配置
 */
struct SIMDOptimizationConfig {
    bool enabled = true;                            ///< 是否启用SIMD
    oscean::common_utils::simd::SIMDImplementation preferredImplementation = 
        oscean::common_utils::simd::SIMDImplementation::AUTO_DETECT;
    size_t batchSize = 1024;                       ///< 批处理大小
    bool enableAutoTuning = true;                  ///< 自动调优
    std::vector<std::string> optimizedOperations = {
        "vectorAdd", "vectorMul", "interpolation", "statistics"
    };                                             ///< 优化的操作列表
};

/**
 * @brief 性能统计信息
 */
struct PerformanceStats {
    std::atomic<uint64_t> totalBytesRead{0};           ///< 总读取字节数
    std::atomic<uint64_t> totalOperations{0};          ///< 总操作数
    std::atomic<uint64_t> cacheHits{0};                ///< 缓存命中数
    std::atomic<uint64_t> cacheMisses{0};              ///< 缓存未命中数
    std::atomic<uint64_t> simdOperations{0};           ///< SIMD操作数
    std::atomic<uint64_t> streamingChunks{0};          ///< 流式处理块数
    
    std::chrono::steady_clock::time_point startTime = 
        std::chrono::steady_clock::now();              ///< 开始时间
    std::chrono::steady_clock::time_point lastAccessTime = 
        std::chrono::steady_clock::now();              ///< 最后访问时间
    
    // 延迟统计
    std::atomic<double> averageReadLatency{0.0};       ///< 平均读取延迟（毫秒）
    std::atomic<double> averageProcessingTime{0.0};    ///< 平均处理时间（毫秒）
    std::atomic<uint64_t> totalReadOperations{0};      ///< 总读取操作数
    
    /**
     * @brief 计算缓存命中率
     */
    double getCacheHitRate() const {
        uint64_t total = cacheHits.load() + cacheMisses.load();
        return total > 0 ? static_cast<double>(cacheHits.load()) / total : 0.0;
    }
    
    /**
     * @brief 计算运行时间（秒）
     */
    double getUptimeSeconds() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - startTime);
        return static_cast<double>(duration.count());
    }
    
    /**
     * @brief 重置统计信息
     */
    void reset() {
        totalBytesRead.store(0);
        totalOperations.store(0);
        cacheHits.store(0);
        cacheMisses.store(0);
        simdOperations.store(0);
        streamingChunks.store(0);
        averageReadLatency.store(0.0);
        averageProcessingTime.store(0.0);
        totalReadOperations.store(0);
        
        startTime = std::chrono::steady_clock::now();
        lastAccessTime = startTime;
    }
};

/**
 * @brief 统一高级数据读取器基类 - 完整实现版本
 * 
 * 提供完整的高级功能集成：
 * - SIMD优化（通过UnifiedSIMDManager）
 * - 内存管理（通过UnifiedMemoryManager）
 * - 异步处理（通过AsyncFramework）
 * - 缓存管理（通过ICacheManager）
 * - 性能监控和统计
 */
class UnifiedAdvancedReader {
public:
    /**
     * @brief 构造函数
     * @param commonServices Common服务工厂
     */
    explicit UnifiedAdvancedReader(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices = nullptr);
    
    /**
     * @brief 析构函数
     */
    virtual ~UnifiedAdvancedReader();
    
    // 禁用拷贝，允许移动
    UnifiedAdvancedReader(const UnifiedAdvancedReader&) = delete;
    UnifiedAdvancedReader& operator=(const UnifiedAdvancedReader&) = delete;
    UnifiedAdvancedReader(UnifiedAdvancedReader&&) = default;
    UnifiedAdvancedReader& operator=(UnifiedAdvancedReader&&) = default;
    
    // =============================================================================
    // 高级功能接口（完整实现）
    // =============================================================================
    
    /**
     * @brief 启用SIMD优化
     * @param config SIMD配置
     */
    void enableSIMDOptimization(const SIMDOptimizationConfig& config = {});
    
    /**
     * @brief 禁用SIMD优化
     */
    void disableSIMDOptimization();
    
    /**
     * @brief 启用内存优化
     * @param enablePooling 是否启用内存池
     * @param maxPoolSize 最大池大小
     */
    void enableMemoryOptimization(bool enablePooling = true, size_t maxPoolSize = 1024 * 1024 * 100);
    
    /**
     * @brief 启用高级缓存
     * @param config 缓存配置
     */
    void enableCaching(const CacheConfig& config);
    
    /**
     * @brief 禁用缓存
     */
    void disableCaching();
    
    /**
     * @brief 启用流式处理模式
     * @param config 流式处理配置
     */
    void enableStreamingMode(const StreamingConfig& config);
    
    /**
     * @brief 禁用流式处理模式
     */
    void disableStreamingMode();
    
    /**
     * @brief 启用异步处理
     * @param threadPoolSize 线程池大小（0表示使用硬件并发数）
     */
    void enableAsyncProcessing(size_t threadPoolSize = 0);
    
    /**
     * @brief 禁用异步处理
     */
    void disableAsyncProcessing();
    
    /**
     * @brief 启用性能监控
     * @param detailed 是否启用详细监控
     */
    void enablePerformanceMonitoring(bool detailed = true);
    
    /**
     * @brief 禁用性能监控
     */
    void disablePerformanceMonitoring();
    
    // =============================================================================
    // 状态查询接口
    // =============================================================================
    
    /**
     * @brief 获取性能统计信息
     */
    const PerformanceStats& getPerformanceStats() const;
    
    /**
     * @brief 获取详细性能报告
     */
    std::string getPerformanceReport() const;
    
    /**
     * @brief 获取系统资源使用情况
     */
    std::string getResourceUsageReport() const;
    
    /**
     * @brief 重置性能统计
     */
    void resetPerformanceStats();
    
    /**
     * @brief 检查高级功能状态
     */
    bool isSIMDEnabled() const noexcept { return simdEnabled_.load(); }
    bool isMemoryOptimizationEnabled() const noexcept { return memoryOptimizationEnabled_.load(); }
    bool isCachingEnabled() const noexcept { return cachingEnabled_.load(); }
    bool isStreamingEnabled() const noexcept { return streamingEnabled_.load(); }
    bool isAsyncEnabled() const noexcept { return asyncEnabled_.load(); }
    bool isPerformanceMonitoringEnabled() const noexcept { return performanceMonitoringEnabled_.load(); }
    
    // =============================================================================
    // 高级数据处理接口 - 使用AsyncTask
    // =============================================================================
    
    /**
     * @brief SIMD优化的数据转换
     * @param input 输入数据
     * @param output 输出数据
     * @param size 数据大小
     * @param operation 操作类型
     */
    oscean::common_utils::async::AsyncTask<void> simdTransformAsync(
        const float* input, float* output, size_t size, 
        const std::string& operation);
    
    /**
     * @brief 并行数据聚合
     * @param data 输入数据
     * @param aggregationType 聚合类型（"sum", "mean", "min", "max"）
     * @return 聚合结果
     */
    oscean::common_utils::async::AsyncTask<double> parallelAggregateAsync(
        const std::vector<double>& data,
        const std::string& aggregationType);
    
    /**
     * @brief 缓存预热
     * @param keys 要预热的缓存键列表
     */
    oscean::common_utils::async::AsyncTask<void> warmupCacheAsync(const std::vector<std::string>& keys);
    
    /**
     * @brief 获取缓存统计信息
     */
    std::unordered_map<std::string, uint64_t> getCacheStatistics() const;
    
    // =============================================================================
    // 性能优化辅助方法
    // =============================================================================
    
    /**
     * @brief 自动调优性能参数
     */
    oscean::common_utils::async::AsyncTask<void> autoTunePerformanceAsync();
    
    /**
     * @brief 预热所有组件
     */
    oscean::common_utils::async::AsyncTask<void> warmupAllComponentsAsync();
    
    /**
     * @brief 优化内存使用
     */
    void optimizeMemoryUsage();
    
    /**
     * @brief 清理资源
     */
    virtual void cleanup();

protected:
    // =============================================================================
    // Common Utilities 组件
    // =============================================================================
    
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices_;
    std::shared_ptr<oscean::common_utils::simd::UnifiedSIMDManager> simdManager_;
    std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager_;
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework_;
    std::shared_ptr<oscean::common_utils::cache::ICacheManager<std::string, std::vector<unsigned char>>> cacheManager_;
    
    // =============================================================================
    // 配置和状态
    // =============================================================================
    
    // 功能启用状态
    std::atomic<bool> simdEnabled_{false};
    std::atomic<bool> memoryOptimizationEnabled_{false};
    std::atomic<bool> cachingEnabled_{false};
    std::atomic<bool> streamingEnabled_{false};
    std::atomic<bool> asyncEnabled_{false};
    std::atomic<bool> performanceMonitoringEnabled_{false};
    
    // 配置信息
    SIMDOptimizationConfig simdConfig_;
    CacheConfig cacheConfig_;
    StreamingConfig streamingConfig_;
    
    // 性能统计
    mutable PerformanceStats performanceStats_;
    
    // 同步原语
    mutable std::shared_mutex configMutex_;
    mutable std::mutex statsMutex_;
    
    // =============================================================================
    // 内部辅助方法
    // =============================================================================
    
    /**
     * @brief 初始化Common组件
     */
    virtual void initializeCommonComponents();
    
    /**
     * @brief 配置SIMD管理器
     */
    void configureSIMDManager();
    
    /**
     * @brief 配置内存管理器
     */
    void configureMemoryManager();
    
    /**
     * @brief 配置异步框架
     */
    void configureAsyncFramework();
    
    /**
     * @brief 配置缓存管理器
     */
    void configureCacheManager();
    
    /**
     * @brief 更新性能统计
     * @param bytesProcessed 处理的字节数
     * @param operationType 操作类型
     * @param duration 操作耗时
     */
    void updatePerformanceStats(
        size_t bytesProcessed, 
        const std::string& operationType,
        std::chrono::milliseconds duration) const;
    
    /**
     * @brief 记录操作开始时间
     * @param operationId 操作ID
     */
    std::chrono::steady_clock::time_point recordOperationStart(const std::string& operationId) const;
    
    /**
     * @brief 记录操作结束时间
     * @param operationId 操作ID
     * @param startTime 开始时间
     * @param bytesProcessed 处理的字节数
     */
    void recordOperationEnd(
        const std::string& operationId,
        std::chrono::steady_clock::time_point startTime,
        size_t bytesProcessed) const;
    
    /**
     * @brief 检查内存使用情况
     */
    bool checkMemoryUsage() const;
    
    /**
     * @brief 获取组件健康状态
     */
    std::unordered_map<std::string, bool> getComponentHealth() const;

private:
    // 操作计时器
    mutable std::unordered_map<std::string, std::chrono::steady_clock::time_point> operationTimers_;
    mutable std::mutex timerMutex_;
    
    // 组件初始化状态
    std::atomic<bool> componentsInitialized_{false};
};

} // namespace oscean::core_services::data_access::readers::impl 