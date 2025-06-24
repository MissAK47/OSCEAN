/**
 * @file cache_config.h
 * @brief 缓存配置和策略定义
 * @author OSCEAN Team
 * @date 2024
 */

#pragma once

#include <chrono>
#include <string>
#include <map>

namespace oscean::common_utils::cache {

/**
 * @brief 缓存策略类型
 */
enum class CacheStrategy {
    LRU,              // Least Recently Used
    LFU,              // Least Frequently Used
    FIFO,             // First In First Out
    TTL,              // Time To Live
    ADAPTIVE,         // 自适应策略
    HIERARCHICAL,     // 分层缓存
    COMPRESSED,       // 压缩缓存
    DISK_BASED,       // 磁盘缓存
    SPATIAL_AWARE,    // 空间感知缓存
    TEMPORAL_AWARE    // 时间感知缓存
};

/**
 * @brief 缓存级别
 */
enum class CacheLevel {
    L1_MEMORY,        // L1内存缓存（最快）
    L2_MEMORY,        // L2内存缓存（较快）
    L3_DISK,          // L3磁盘缓存（中等）
    L4_NETWORK        // L4网络缓存（较慢）
};

/**
 * @brief 环境类型
 */
enum class Environment {
    DEVELOPMENT,      // 开发环境
    TESTING,          // 测试环境
    PRODUCTION,       // 生产环境
    HPC              // 高性能计算环境
};

/**
 * @brief 缓存配置
 */
struct CacheConfig {
    CacheStrategy strategy = CacheStrategy::ADAPTIVE;
    size_t capacity = 100000;
    std::chrono::seconds ttl{3600};
    bool enableCompression = false;
    bool enablePersistence = false;
    std::string persistencePath;
    double evictionThreshold = 0.9;    // 达到90%开始清理
    size_t maxMemoryMB = 1024;
    bool enableStatistics = true;
    
    // 海洋数据专用配置
    bool enableSpatialIndexing = false;
    bool enableTemporalIndexing = false;
    size_t spatialGridSize = 1000;     // 空间网格大小
    std::chrono::seconds temporalResolution{300}; // 5分钟时间分辨率
    
    // 预定义配置工厂方法
    static CacheConfig forSpatialData();
    static CacheConfig forTemporalData();
    static CacheConfig forInterpolationResults();
    static CacheConfig forMetadata();
    static CacheConfig forDataAccess();
    static CacheConfig forCRSTransformations();
    static CacheConfig forEnvironment(Environment env);
};

} // namespace oscean::common_utils::cache 