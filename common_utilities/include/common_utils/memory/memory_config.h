#pragma once

/**
 * @file memory_config.h
 * @brief 内存配置和环境管理 - 统一的配置接口
 * 
 * 重构目标：
 * ✅ 简化memory_traits.h中的复杂配置
 * ✅ 环境感知的内存配置
 * ✅ 大数据处理优化配置
 * ✅ 向后兼容所有现有配置
 */

#include <chrono>
#include <string>
#include <cstddef>

namespace oscean::common_utils::memory {

/**
 * @brief 环境类型 - 兼容现有定义
 */
enum class Environment {
    DEVELOPMENT,
    TESTING, 
    PRODUCTION,
    HPC
};

/**
 * @brief 内存管理器类型 - 兼容现有定义
 */
enum class MemoryManagerType {
    STANDARD,
    HIGH_PERFORMANCE,
    LOW_MEMORY,
    DEBUG,
    NUMA_AWARE,
    SIMD_OPTIMIZED,
    STREAMING,
    CACHE_OPTIMIZED
};

/**
 * @brief 分配策略 - 兼容现有定义
 */
enum class AllocationStrategy {
    FIRST_FIT,
    BEST_FIT,
    WORST_FIT,
    NEXT_FIT,
    POOL_BASED,
    ALIGNED,
    NUMA_AWARE,
    CACHE_FRIENDLY,
    SEQUENTIAL,
    RANDOM
};

/**
 * @brief 内存池类型 - 兼容现有定义
 */
enum class MemoryPoolType {
    GENERAL_PURPOSE,
    SMALL_OBJECTS,
    LARGE_OBJECTS,
    TEMPORARY,
    SIMD_ALIGNED,
    STREAMING_BUFFER,
    CACHE_STORAGE,
    STRING_POOL
};

/**
 * @brief 统一内存配置 - 整合所有配置选项
 */
struct Config {
    // === 环境配置 ===
    Environment environment = Environment::PRODUCTION;
    
    // === 基础内存限制 ===
    size_t maxTotalMemoryMB = 256;           // 总内存限制
    size_t chunkSizeMB = 16;                 // 流式处理块大小
    size_t alignmentSize = 64;               // 缓存行对齐
    
    // === 并发配置 ===
    size_t concurrentThreads = 8;           // 并发线程数
    bool enableThreadLocalCache = true;      // 线程本地缓存
    
    // === 性能优化 ===
    bool enableSIMDOptimization = true;      // SIMD优化
    bool enableNUMAOptimization = false;     // NUMA优化
    bool enableCacheFriendly = true;         // 缓存友好
    
    // === 大数据支持 ===
    bool enableLargeDataSupport = true;      // 大数据流式支持
    size_t largeDataThresholdMB = 100;       // 大数据阈值
    size_t streamingBufferCount = 4;         // 流式缓冲区数量
    
    // === 调试配置 ===
    bool enableStatistics = true;           // 统计信息
    bool enableLeakDetection = false;       // 内存泄漏检测（仅调试）
    bool enableBoundsChecking = false;      // 边界检查（仅调试）
    
    // === 压力管理 ===
    size_t memoryPressureThresholdMB = 200; // 内存压力阈值
    bool enableAutoGarbageCollection = true; // 自动垃圾回收
    
    /**
     * @brief 验证配置有效性
     */
    bool isValid() const noexcept;
    
    /**
     * @brief 针对环境优化配置
     */
    static Config optimizeForEnvironment(Environment env) noexcept;
    
    /**
     * @brief 针对大数据优化配置
     */
    static Config optimizeForLargeData(size_t expectedFileSizeGB = 5) noexcept;
    
    /**
     * @brief 针对高并发优化配置  
     */
    static Config optimizeForHighConcurrency(size_t maxThreads = 32) noexcept;
};

/**
 * @brief 内存特性 - 兼容现有MemoryTraits
 */
struct MemoryTraits {
    size_t alignment = 0;                    // 自定义对齐要求
    bool zeroInitialized = false;            // 零初始化
    bool enablePrefault = false;             // 预分配物理页
    bool enableHugePage = false;             // 大页支持
    int numaNode = -1;                       // NUMA节点
    bool enableMemoryAdvise = true;          // 内存建议
    int memoryAdvice = 0;                    // 内存使用建议
    bool enableBoundsChecking = false;       // 边界检查
    bool enableAccessTracking = false;       // 访问跟踪
    std::string memoryTag;                   // 内存标签
    
    MemoryTraits() = default;
};

} // namespace oscean::common_utils::memory 