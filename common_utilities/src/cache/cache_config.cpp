/**
 * @file cache_config.cpp
 * @brief 缓存配置实现
 * @author OSCEAN Team
 * @date 2024
 */

#include "common_utils/cache/cache_config.h"

namespace oscean::common_utils::cache {

CacheConfig CacheConfig::forSpatialData() {
    CacheConfig config;
    config.strategy = CacheStrategy::SPATIAL_AWARE;
    config.capacity = 50000;
    config.ttl = std::chrono::seconds(7200); // 2小时
    config.enableCompression = true;
    config.maxMemoryMB = 512;
    config.enableSpatialIndexing = true;
    config.spatialGridSize = 1000;
    return config;
}

CacheConfig CacheConfig::forTemporalData() {
    CacheConfig config;
    config.strategy = CacheStrategy::TEMPORAL_AWARE;
    config.capacity = 100000;
    config.ttl = std::chrono::seconds(3600); // 1小时
    config.enableCompression = false;
    config.maxMemoryMB = 1024;
    config.enableTemporalIndexing = true;
    config.temporalResolution = std::chrono::seconds(300); // 5分钟
    return config;
}

CacheConfig CacheConfig::forInterpolationResults() {
    CacheConfig config;
    config.strategy = CacheStrategy::ADAPTIVE;
    config.capacity = 100000;
    config.ttl = std::chrono::seconds(3600); // 1小时
    config.enableCompression = false;
    config.maxMemoryMB = 1024;
    config.enableSpatialIndexing = true;
    config.enableTemporalIndexing = true;
    return config;
}

CacheConfig CacheConfig::forMetadata() {
    CacheConfig config;
    config.strategy = CacheStrategy::TTL;
    config.capacity = 10000;
    config.ttl = std::chrono::seconds(14400); // 4小时
    config.enablePersistence = true;
    config.maxMemoryMB = 128;
    config.enableStatistics = true;
    return config;
}

CacheConfig CacheConfig::forDataAccess() {
    CacheConfig config;
    config.strategy = CacheStrategy::LRU;
    config.capacity = 200000;
    config.ttl = std::chrono::seconds(1800); // 30分钟
    config.enableCompression = true;
    config.maxMemoryMB = 2048;
    config.evictionThreshold = 0.85;
    return config;
}

CacheConfig CacheConfig::forCRSTransformations() {
    CacheConfig config;
    config.strategy = CacheStrategy::LFU;
    config.capacity = 5000;
    config.ttl = std::chrono::seconds(86400); // 24小时
    config.enablePersistence = true;
    config.maxMemoryMB = 64;
    config.evictionThreshold = 0.95;
    return config;
}

CacheConfig CacheConfig::forEnvironment(Environment env) {
    CacheConfig config;
    
    switch (env) {
        case Environment::DEVELOPMENT:
            config.strategy = CacheStrategy::LRU;
            config.capacity = 1000;
            config.ttl = std::chrono::seconds(300); // 5分钟
            config.maxMemoryMB = 64;
            config.enableStatistics = true;
            break;
            
        case Environment::TESTING:
            config.strategy = CacheStrategy::ADAPTIVE;
            config.capacity = 10000;
            config.ttl = std::chrono::seconds(600); // 10分钟
            config.maxMemoryMB = 256;
            config.enableStatistics = true;
            break;
            
        case Environment::PRODUCTION:
            config.strategy = CacheStrategy::ADAPTIVE;
            config.capacity = 100000;
            config.ttl = std::chrono::seconds(3600); // 1小时
            config.maxMemoryMB = 1024;
            config.enableCompression = true;
            config.enablePersistence = true;
            config.enableStatistics = true;
            break;
            
        case Environment::HPC:
            config.strategy = CacheStrategy::HIERARCHICAL;
            config.capacity = 1000000;
            config.ttl = std::chrono::seconds(7200); // 2小时
            config.maxMemoryMB = 8192;
            config.enableCompression = true;
            config.enablePersistence = true;
            config.enableSpatialIndexing = true;
            config.enableTemporalIndexing = true;
            config.evictionThreshold = 0.9;
            break;
    }
    
    return config;
}

} // namespace oscean::common_utils::cache 