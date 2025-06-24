#include "common_utils/memory/memory_config.h"
#include <algorithm>
#include <thread>

namespace oscean::common_utils::memory {

bool Config::isValid() const noexcept {
    // 基础验证
    if (maxTotalMemoryMB == 0 || chunkSizeMB == 0 || alignmentSize == 0) {
        return false;
    }
    
    // 对齐大小必须是2的幂
    if ((alignmentSize & (alignmentSize - 1)) != 0) {
        return false;
    }
    
    // 块大小不能超过总内存
    if (chunkSizeMB > maxTotalMemoryMB) {
        return false;
    }
    
    // 内存压力阈值应该小于总内存
    if (memoryPressureThresholdMB >= maxTotalMemoryMB) {
        return false;
    }
    
    // 流式缓冲区数量合理范围
    if (streamingBufferCount == 0 || streamingBufferCount > 16) {
        return false;
    }
    
    return true;
}

Config Config::optimizeForEnvironment(Environment env) noexcept {
    Config config;
    
    switch (env) {
        case Environment::DEVELOPMENT:
            config.maxTotalMemoryMB = 128;
            config.chunkSizeMB = 8;
            config.concurrentThreads = 4;
            config.enableStatistics = true;
            config.enableLeakDetection = true;
            config.enableBoundsChecking = true;
            break;
            
        case Environment::TESTING:
            config.maxTotalMemoryMB = 256;
            config.chunkSizeMB = 16;
            config.concurrentThreads = 8;
            config.enableStatistics = true;
            config.enableLeakDetection = true;
            config.enableBoundsChecking = false;
            break;
            
        case Environment::PRODUCTION:
            config.maxTotalMemoryMB = 512;
            config.chunkSizeMB = 32;
            config.concurrentThreads = std::thread::hardware_concurrency();
            config.enableStatistics = true;
            config.enableLeakDetection = false;
            config.enableBoundsChecking = false;
            config.enableSIMDOptimization = true;
            config.enableCacheFriendly = true;
            break;
            
        case Environment::HPC:
            config.maxTotalMemoryMB = 1024;
            config.chunkSizeMB = 64;
            config.concurrentThreads = std::thread::hardware_concurrency() * 2;
            config.enableStatistics = false; // 性能优先
            config.enableSIMDOptimization = true;
            config.enableNUMAOptimization = true;
            config.enableCacheFriendly = true;
            config.enableLargeDataSupport = true;
            config.largeDataThresholdMB = 1000;
            break;
    }
    
    config.environment = env;
    return config;
}

Config Config::optimizeForLargeData(size_t expectedFileSizeGB) noexcept {
    Config config;
    
    // 基于文件大小调整内存配置
    if (expectedFileSizeGB <= 1) {
        config.maxTotalMemoryMB = 256;
        config.chunkSizeMB = 16;
    } else if (expectedFileSizeGB <= 5) {
        config.maxTotalMemoryMB = 512;
        config.chunkSizeMB = 32;
    } else if (expectedFileSizeGB <= 10) {
        config.maxTotalMemoryMB = 1024;
        config.chunkSizeMB = 64;
    } else {
        config.maxTotalMemoryMB = 2048;
        config.chunkSizeMB = 128;
    }
    
    // 大数据特化配置
    config.enableLargeDataSupport = true;
    config.largeDataThresholdMB = std::min(expectedFileSizeGB * 100, config.maxTotalMemoryMB / 2);
    config.streamingBufferCount = 8; // 更多缓冲区用于预读
    config.memoryPressureThresholdMB = config.maxTotalMemoryMB * 3 / 4; // 75%阈值
    config.enableAutoGarbageCollection = true;
    
    // 性能优化
    config.enableSIMDOptimization = true;
    config.enableCacheFriendly = true;
    config.concurrentThreads = std::thread::hardware_concurrency();
    
    return config;
}

Config Config::optimizeForHighConcurrency(size_t maxThreads) noexcept {
    Config config;
    
    // 并发优化配置
    config.concurrentThreads = maxThreads;
    config.enableThreadLocalCache = true;
    
    // 内存配置基于线程数
    config.maxTotalMemoryMB = std::max(static_cast<size_t>(256), maxThreads * 32); // 每线程32MB
    config.chunkSizeMB = std::max(static_cast<size_t>(8), maxThreads / 4); // 动态块大小
    
    // 并发优化
    config.enableSIMDOptimization = true;
    config.enableCacheFriendly = true;
    config.enableNUMAOptimization = maxThreads > 16; // 高并发启用NUMA
    
    // 内存压力管理
    config.memoryPressureThresholdMB = config.maxTotalMemoryMB * 2 / 3; // 67%阈值
    config.enableAutoGarbageCollection = true;
    
    // 减少统计开销
    config.enableStatistics = maxThreads <= 32; // 超高并发时禁用统计
    
    return config;
}

} // namespace oscean::common_utils::memory 