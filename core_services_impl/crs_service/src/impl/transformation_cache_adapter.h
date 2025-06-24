/**
 * @file transformation_cache_adapter.h
 * @brief 高性能转换缓存实现 - 集成Common模块全套功能
 * 
 * 🎯 实现目标：
 * ✅ 实现ITransformationCache接口
 * ✅ 集成Common模块的智能缓存、SIMD优化、性能监控
 * ✅ 支持大批量高并发坐标转换
 * ✅ 内存高效的流式处理
 * ✅ 统一使用boost::future异步接口
 */

#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 缓存适配器只使用boost::future，不使用boost::asio

#include "transformation_cache.h"
#include "common_utils/cache/icache_manager.h"
#include "common_utils/infrastructure/performance_monitor.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/simd/isimd_manager.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <vector>
#include <boost/thread/future.hpp>
#include <functional>
#include <proj.h>

namespace oscean::core_services::crs {

/**
 * @brief 高性能坐标转换缓存实现
 * 
 * 核心特性：
 * - 基于Common模块的多级缓存策略
 * - SIMD优化的批量坐标转换
 * - 实时性能监控和自适应优化
 * - 内存高效的流式处理能力
 * - 线程安全的并发访问
 * - 统一boost::future异步接口
 */
class HighPerformanceTransformationCache : public ITransformationCache {
public:
    /**
     * @brief 构造函数
     */
    explicit HighPerformanceTransformationCache(
        std::shared_ptr<oscean::common_utils::cache::ICacheManager<std::string, std::string>> cacheManager,
        std::shared_ptr<oscean::common_utils::infrastructure::PerformanceMonitor> perfMonitor,
        std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager,
        std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
        std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> threadManager,
        size_t maxTransformers = 1000
    );
    
    ~HighPerformanceTransformationCache() override;

    // 禁用复制和移动
    HighPerformanceTransformationCache(const HighPerformanceTransformationCache&) = delete;
    HighPerformanceTransformationCache& operator=(const HighPerformanceTransformationCache&) = delete;
    HighPerformanceTransformationCache(HighPerformanceTransformationCache&&) = delete;
    HighPerformanceTransformationCache& operator=(HighPerformanceTransformationCache&&) = delete;

    // === ITransformationCache接口实现 (统一boost::future) ===
    
    boost::future<std::shared_ptr<TransformationInfo>> getTransformerAsync(
        const std::string& sourceCRS,
        const std::string& targetCRS
    ) override;

    boost::future<std::vector<std::shared_ptr<TransformationInfo>>> getTransformersAsync(
        const std::vector<std::pair<std::string, std::string>>& crsPairs
    ) override;

    std::optional<std::pair<double, double>> transformPoint(
        const std::shared_ptr<TransformationInfo>& transformer,
        double x, double y
    ) override;

    boost::future<std::vector<std::optional<std::pair<double, double>>>> transformPointsBatch(
        const std::shared_ptr<TransformationInfo>& transformer,
        const std::vector<std::pair<double, double>>& points
    ) override;

    boost::future<void> transformPointsStream(
        const std::shared_ptr<TransformationInfo>& transformer,
        const std::vector<std::pair<double, double>>& points,
        std::function<void(const std::vector<std::optional<std::pair<double, double>>>&)> callback,
        size_t batchSize = 10000
    ) override;

    void warmupCache(
        const std::vector<std::pair<std::string, std::string>>& commonTransformations
    ) override;

    void optimizeCache() override;

    size_t cleanupUnusedTransformers(
        std::chrono::minutes maxIdleTime = std::chrono::minutes(30)
    ) override;

    CacheStatistics getStatistics() const override;

    void resetStatistics() override;

private:
    // === Common模块服务实例 ===
    std::shared_ptr<oscean::common_utils::cache::ICacheManager<std::string, std::string>> cacheManager_;
    std::shared_ptr<oscean::common_utils::infrastructure::PerformanceMonitor> perfMonitor_;
    std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager_;
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;
    std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> threadManager_;

    // === 本地转换器缓存 ===
    mutable std::shared_mutex transformerMutex_;
    std::unordered_map<std::string, std::shared_ptr<TransformationInfo>> transformerCache_;
    
    // === 配置和统计 ===
    const size_t maxTransformers_;
    mutable std::atomic<size_t> totalRequests_{0};
    mutable std::atomic<size_t> cacheHits_{0};
    mutable std::atomic<size_t> totalPointsTransformed_{0};
    mutable std::atomic<double> totalTransformTime_{0.0};
    
    // === PROJ上下文管理 ===
    PJ_CONTEXT* projContext_;
    mutable std::mutex projMutex_;

    // === 私有辅助方法 ===
    
    /**
     * @brief 生成缓存键
     */
    std::string generateCacheKey(const std::string& sourceCRS, const std::string& targetCRS) const;
    
    /**
     * @brief 创建PROJ转换器
     */
    PJ* createProjTransformer(const std::string& sourceCRS, const std::string& targetCRS);
    
    /**
     * @brief 获取或创建转换器（内部实现）
     */
    std::shared_ptr<TransformationInfo> getOrCreateTransformer(
        const std::string& sourceCRS, 
        const std::string& targetCRS
    );
    
    /**
     * @brief SIMD优化的批量转换核心实现
     */
    std::vector<std::optional<std::pair<double, double>>> transformPointsSIMDImpl(
        PJ* transformer,
        const std::vector<std::pair<double, double>>& points
    );
    
    /**
     * @brief 传统的批量转换实现（回退方案）
     */
    std::vector<std::optional<std::pair<double, double>>> transformPointsStandardImpl(
        PJ* transformer,
        const std::vector<std::pair<double, double>>& points
    );
    
    /**
     * @brief 更新性能统计
     */
    void updateStatistics(
        const std::string& operation,
        size_t pointCount,
        double durationMs,
        bool cacheHit = false
    );
    
    /**
     * @brief 实施LRU策略
     */
    void enforceMaxSize();
    
    /**
     * @brief 清理过期条目
     */
    void cleanupExpiredEntries(std::chrono::minutes maxIdleTime);
    
    /**
     * @brief 优化SIMD处理参数
     */
    void optimizeSIMDParameters();
    
    /**
     * @brief 自适应缓存策略调整
     */
    void adaptiveCacheOptimization();
};

/**
 * @brief 转换缓存工厂实现
 */
class TransformationCacheFactoryImpl {
public:
    static std::unique_ptr<ITransformationCache> createStandardCache(
        size_t maxTransformers
    );

    static std::unique_ptr<ITransformationCache> createHighPerformanceCache(
        size_t maxTransformers,
        size_t simdBatchSize
    );

    static std::unique_ptr<ITransformationCache> createLowMemoryCache(
        size_t maxTransformers,
        size_t maxMemoryMB
    );

    static std::unique_ptr<ITransformationCache> createCache(
        std::shared_ptr<oscean::common_utils::cache::ICacheManager<std::string, std::string>> cacheManager,
        std::shared_ptr<oscean::common_utils::infrastructure::PerformanceMonitor> perfMonitor,
        std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager,
        std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager
    );

private:
    /**
     * @brief 获取Common服务工厂实例
     */
    static oscean::common_utils::infrastructure::CommonServicesFactory& getCommonFactory();
};

} // namespace oscean::core_services::crs 