#pragma once

/**
 * @file memory_interfaces.h
 * @brief 内存管理器接口定义 - 兼容现有接口
 * 
 * 重构目标：
 * ✅ 保持与现有IMemoryManager 100%兼容
 * ✅ 添加大数据流式处理接口
 * ✅ 简化接口层次结构
 * ✅ 支持高并发分配
 */

#include "memory_config.h"
#include <memory>
#include <functional>
#include <string>
#include <atomic>
#include <chrono>

namespace oscean::common_utils::memory {

/**
 * @brief 内存使用统计信息
 */
struct MemoryUsageStats {
    size_t currentUsed = 0;        // 当前使用的内存量（字节）
    size_t totalAllocated = 0;     // 总分配的内存量（字节）
    size_t peakUsage = 0;          // 峰值使用量（字节）
    size_t allocationCount = 0;    // 分配次数
    size_t deallocationCount = 0;  // 释放次数
    double fragmentationRatio = 0.0; // 碎片化率
    
    // 允许默认拷贝和移动
    MemoryUsageStats() = default;
    MemoryUsageStats(const MemoryUsageStats&) = default;
    MemoryUsageStats& operator=(const MemoryUsageStats&) = default;
    MemoryUsageStats(MemoryUsageStats&&) = default;
    MemoryUsageStats& operator=(MemoryUsageStats&&) = default;
};

/**
 * @brief 内存管理器基础接口 - 与现有IMemoryManager兼容
 */
class IMemoryManager {
public:
    virtual ~IMemoryManager() = default;
    
    // === 基础接口 (完全兼容现有) ===
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    
    virtual void* allocate(size_t size, size_t alignment = 0, const std::string& tag = "") = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void* reallocate(void* ptr, size_t newSize, const std::string& tag = "") = 0;
    
    virtual bool isManaged(void* ptr) const = 0;
    virtual size_t getPoolSize() const = 0;
    virtual size_t getAvailableMemory() const = 0;
    virtual MemoryUsageStats getUsageStats() const = 0;
    
    virtual void setAllocationStrategy(AllocationStrategy strategy) = 0;
    virtual void setMemoryUsageCallback(std::function<void(const MemoryUsageStats&)> callback) = 0;
};

/**
 * @brief 流式缓冲区接口 - 大数据处理专用
 */
class IStreamingBuffer {
public:
    virtual ~IStreamingBuffer() = default;
    
    /**
     * @brief 获取写入缓冲区
     */
    virtual void* getWriteBuffer(size_t size) = 0;
    
    /**
     * @brief 提交实际写入大小
     */
    virtual void commitBuffer(size_t actualSize) = 0;
    
    /**
     * @brief 获取读取缓冲区
     */
    virtual const void* getReadBuffer() const = 0;
    
    /**
     * @brief 获取缓冲区大小
     */
    virtual size_t getBufferSize() const noexcept = 0;
    
    /**
     * @brief 重置缓冲区
     */
    virtual void resetBuffer() noexcept = 0;
};

/**
 * @brief 并发分配器接口 - 高并发场景专用
 */
class IConcurrentAllocator {
public:
    virtual ~IConcurrentAllocator() = default;
    
    /**
     * @brief 线程安全的分配
     */
    virtual void* allocate(size_t size) noexcept = 0;
    
    /**
     * @brief 线程安全的释放
     */
    virtual void deallocate(void* ptr) noexcept = 0;
    
    /**
     * @brief 获取线程统计
     */
    virtual MemoryUsageStats getThreadStats() const noexcept = 0;
};

/**
 * @brief STL兼容分配器接口
 */
template<typename T>
class ISTLAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    virtual ~ISTLAllocator() = default;
    
    virtual T* allocate(size_t count) = 0;
    virtual void deallocate(T* ptr, size_t count) noexcept = 0;
    
    template<typename U>
    struct rebind {
        using other = ISTLAllocator<U>;
    };
};

/**
 * @brief 内存池配置 - 兼容现有MemoryPoolConfig
 */
struct MemoryPoolConfig {
    // 基础配置
    size_t initialSize = 64 * 1024 * 1024;  // 64MB
    size_t maxSize = 512 * 1024 * 1024;     // 512MB
    size_t blockSize = 4 * 1024;            // 4KB
    size_t alignmentSize = 16;              // 16字节对齐
    bool enableGrowth = true;
    double growthFactor = 1.5;
    size_t maxBlocks = 10000;
    std::chrono::seconds cleanupInterval{300}; // 5分钟
    bool enableStatistics = true;
    
    // 高性能配置
    size_t fastPathThreshold = 4096;        // 快速路径阈值
    bool enableFastPath = true;              // 启用快速路径
    bool enableNUMA = false;                 // 启用NUMA优化
    int numaNode = -1;                       // NUMA节点
    size_t maxPoolCount = 16;                // 最大内存池数量
    
    // 缓存配置
    bool enableCacheFriendly = true;         // 缓存友好分配
    size_t cacheLineSize = 64;               // 缓存行大小
    
    // 调试配置
    bool enableMemoryTracking = false;       // 启用内存跟踪
    bool enableLeakDetection = false;        // 启用内存泄漏检测
    size_t memoryFillPattern = 0xDEADBEEF;   // 内存填充模式
    
    MemoryPoolConfig() = default;
    
    // 预设配置
    static MemoryPoolConfig forDevelopment();
    static MemoryPoolConfig forTesting();
    static MemoryPoolConfig forHPC();
    static MemoryPoolConfig forLowMemory();
    static MemoryPoolConfig forStreaming();
};

/**
 * @brief 内存块信息 - 兼容现有MemoryBlock
 */
struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    std::atomic<bool> inUse{false};
    std::chrono::steady_clock::time_point allocTime;
    std::chrono::steady_clock::time_point lastAccess;
    std::string tag;
    bool isAlignedAllocation = false;
    
    // 对齐分配支持
    void* originalPtr = nullptr;
    size_t alignmentOffset = 0;
    
    MemoryBlock() {
        auto now = std::chrono::steady_clock::now();
        allocTime = now;
        lastAccess = now;
    }
    
    MemoryBlock(void* p, size_t s, const std::string& t, bool aligned = false);
    MemoryBlock(const MemoryBlock& other);
    MemoryBlock& operator=(const MemoryBlock& other);
};

} // namespace oscean::common_utils::memory 