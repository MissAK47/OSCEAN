/**
 * @file transformation_cache.h
 * @brief 现代化坐标转换缓存接口 - 基于Common模块优化
 * 
 * 🎯 设计目标：
 * ✅ 基于Common模块的智能缓存系统
 * ✅ 支持大批量高并发转换
 * ✅ 集成SIMD优化和性能监控
 * ✅ 内存高效和线程安全
 * ✅ 统一使用boost::future异步接口
 */

#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 转换缓存只使用boost::future，不使用boost::asio

#include "common_utils/cache/icache_manager.h"
#include "common_utils/infrastructure/performance_monitor.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/simd/isimd_manager.h"

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <atomic>
#include <boost/thread/future.hpp>
#include <functional>
#include <proj.h>

namespace oscean::core_services::crs {

/**
 * @brief 坐标转换器信息
 */
struct TransformationInfo {
    std::string sourceId;
    std::string targetId;
    PJ* projTransformer = nullptr;
    std::chrono::steady_clock::time_point createdAt;
    std::chrono::steady_clock::time_point lastUsed;
    std::atomic<size_t> usageCount{0};
    std::atomic<double> totalTransformTime{0.0};
    std::atomic<size_t> totalPoints{0};
    
    // 性能统计
    double averageLatencyMs() const {
        return totalPoints > 0 ? totalTransformTime.load() / totalPoints.load() : 0.0;
    }
    
    double throughputPointsPerSecond() const {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - createdAt
        ).count();
        return elapsed > 0 ? (totalPoints.load() * 1000.0) / elapsed : 0.0;
    }
    
    ~TransformationInfo() {
        if (projTransformer) {
            proj_destroy(projTransformer);
            projTransformer = nullptr;
        }
    }
};

/**
 * @brief 现代化坐标转换缓存
 * 
 * 核心特性：
 * - 基于Common模块的智能缓存管理
 * - 支持大批量并行转换（SIMD优化）
 * - 实时性能监控和自适应优化
 * - 内存高效的流式处理
 * - 统一boost::future异步接口
 */
class ITransformationCache {
public:
    virtual ~ITransformationCache() = default;

    /**
     * @brief 缓存统计信息
     */
    struct CacheStatistics {
        size_t totalTransformers = 0;
        size_t activeTransformers = 0;
        size_t totalRequests = 0;
        size_t cacheHits = 0;
        double hitRatio = 0.0;
        size_t memoryUsageBytes = 0;
        double averageAccessTimeMs = 0.0;
        size_t totalPointsTransformed = 0;
        double averageThroughput = 0.0;
    };

    // === 核心转换接口 (统一boost::future) ===
    
    /**
     * @brief 获取或创建转换器（异步）
     */
    virtual boost::future<std::shared_ptr<TransformationInfo>> getTransformerAsync(
        const std::string& sourceCRS,
        const std::string& targetCRS
    ) = 0;

    /**
     * @brief 批量获取转换器（并行优化）
     */
    virtual boost::future<std::vector<std::shared_ptr<TransformationInfo>>> getTransformersAsync(
        const std::vector<std::pair<std::string, std::string>>& crsPairs
    ) = 0;

    /**
     * @brief 单点坐标转换（同步）
     */
    virtual std::optional<std::pair<double, double>> transformPoint(
        const std::shared_ptr<TransformationInfo>& transformer,
        double x, double y
    ) = 0;

    /**
     * @brief 批量坐标转换（SIMD优化）
     */
    virtual boost::future<std::vector<std::optional<std::pair<double, double>>>> transformPointsBatch(
        const std::shared_ptr<TransformationInfo>& transformer,
        const std::vector<std::pair<double, double>>& points
    ) = 0;

    /**
     * @brief 流式大批量转换（内存优化）
     */
    virtual boost::future<void> transformPointsStream(
        const std::shared_ptr<TransformationInfo>& transformer,
        const std::vector<std::pair<double, double>>& points,
        std::function<void(const std::vector<std::optional<std::pair<double, double>>>&)> callback,
        size_t batchSize = 10000
    ) = 0;

    // === 缓存管理接口 ===
    
    /**
     * @brief 预热缓存
     */
    virtual void warmupCache(
        const std::vector<std::pair<std::string, std::string>>& commonTransformations
    ) = 0;

    /**
     * @brief 优化缓存策略
     */
    virtual void optimizeCache() = 0;

    /**
     * @brief 清理未使用的转换器
     */
    virtual size_t cleanupUnusedTransformers(
        std::chrono::minutes maxIdleTime = std::chrono::minutes(30)
    ) = 0;

    /**
     * @brief 获取缓存统计信息
     */
    virtual CacheStatistics getStatistics() const = 0;

    /**
     * @brief 重置所有统计信息
     */
    virtual void resetStatistics() = 0;
};

/**
 * @brief 高性能转换缓存实现工厂
 */
class TransformationCacheFactory {
public:
    /**
     * @brief 创建标准性能缓存
     */
    static std::unique_ptr<ITransformationCache> createStandardCache(
        size_t maxTransformers = 1000
    );

    /**
     * @brief 创建高性能缓存（SIMD优化）
     */
    static std::unique_ptr<ITransformationCache> createHighPerformanceCache(
        size_t maxTransformers = 5000,
        size_t simdBatchSize = 1000
    );

    /**
     * @brief 创建低内存缓存
     */
    static std::unique_ptr<ITransformationCache> createLowMemoryCache(
        size_t maxTransformers = 100,
        size_t maxMemoryMB = 64
    );

    /**
     * @brief 创建自定义缓存
     */
    static std::unique_ptr<ITransformationCache> createCache(
        std::shared_ptr<oscean::common_utils::cache::ICacheManager<std::string, std::string>> cacheManager,
        std::shared_ptr<oscean::common_utils::infrastructure::PerformanceMonitor> perfMonitor,
        std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager,
        std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager
    );
};

} // namespace oscean::core_services::crs 