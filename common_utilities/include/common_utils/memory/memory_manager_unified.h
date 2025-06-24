#pragma once

/**
 * @file memory_manager_unified.h
 * @brief 统一内存管理器 - 整合所有功能的单一实现
 * 
 * 重构目标：
 * ✅ 整合5个管理器：BaseMemoryManager, HighPerformanceMemoryManager, 
 *                  SIMDOptimizedMemoryManager, DebugMemoryManager, GlobalMemoryManager
 * ✅ 保持与现有UnifiedMemoryManager接口100%兼容
 * ✅ 新增大数据流式处理能力 (GB级数据<256MB内存)
 * ✅ 新增高并发分配优化 (万级并发支持)
 * ✅ 简化实现复杂度，提升可维护性
 */

#include "memory_interfaces.h"
#include "memory_config.h"
#include <memory>
#include <atomic>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <functional>

namespace oscean::common_utils::memory {

/**
 * @brief 统一内存管理器 - 所有功能的集成实现
 */
class UnifiedMemoryManager : public IMemoryManager {
public:
    /**
     * @brief 构造函数 - 支持配置定制
     */
    explicit UnifiedMemoryManager(const Config& config = Config{});
    virtual ~UnifiedMemoryManager();
    
    // 禁用拷贝，允许移动
    UnifiedMemoryManager(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager& operator=(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager(UnifiedMemoryManager&&) = default;
    UnifiedMemoryManager& operator=(UnifiedMemoryManager&&) = default;
    
    // === IMemoryManager接口实现 (完全兼容) ===
    bool initialize() override;
    void shutdown() override;
    
    void* allocate(size_t size, size_t alignment = 0, const std::string& tag = "") override;
    void deallocate(void* ptr) override;
    void* reallocate(void* ptr, size_t newSize, const std::string& tag = "") override;
    
    bool isManaged(void* ptr) const override;
    size_t getPoolSize() const override;
    size_t getAvailableMemory() const override;
    MemoryUsageStats getUsageStats() const override;
    
    void setAllocationStrategy(AllocationStrategy strategy) override;
    void setMemoryUsageCallback(std::function<void(const MemoryUsageStats&)> callback) override;
    
    // === 大数据流式处理支持 ===
    
    /**
     * @brief 流式缓冲区实现
     */
    class StreamingBuffer : public IStreamingBuffer {
    public:
        explicit StreamingBuffer(UnifiedMemoryManager& manager, size_t maxSize);
        ~StreamingBuffer();
        
        void* getWriteBuffer(size_t size) override;
        void commitBuffer(size_t actualSize) override;
        const void* getReadBuffer() const override;
        size_t getBufferSize() const noexcept override;
        void resetBuffer() noexcept override;
        
    private:
        UnifiedMemoryManager& manager_;
        void* buffer_;
        size_t maxSize_;
        size_t currentSize_;
        std::mutex bufferMutex_;
    };
    
    /**
     * @brief 创建流式缓冲区
     */
    std::unique_ptr<StreamingBuffer> createStreamingBuffer(size_t maxSize);
    
    // === 高并发分配支持 ===
    
    /**
     * @brief 并发分配器实现
     */
    class ConcurrentAllocator : public IConcurrentAllocator {
    public:
        explicit ConcurrentAllocator(UnifiedMemoryManager& manager);
        ~ConcurrentAllocator();
        
        void* allocate(size_t size) noexcept override;
        void deallocate(void* ptr) noexcept override;
        MemoryUsageStats getThreadStats() const noexcept override;
        
    private:
        UnifiedMemoryManager& manager_;
        MemoryUsageStats threadStats_;
        mutable std::mutex statsMutex_;
    };
    
    /**
     * @brief 创建并发分配器
     */
    std::unique_ptr<ConcurrentAllocator> createConcurrentAllocator();
    
    // === STL兼容分配器 ===
    
    /**
     * @brief STL兼容分配器实现
     */
    template<typename T>
    class STLAllocator : public ISTLAllocator<T> {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        
        explicit STLAllocator(UnifiedMemoryManager& manager) : manager_(manager) {}
        
        template<typename U>
        STLAllocator(const STLAllocator<U>& other) : manager_(other.manager_) {}
        
        T* allocate(size_t count) override {
            if (count == 0) return nullptr;
            
            size_t size = count * sizeof(T);
            void* ptr = manager_.allocate(size, alignof(T));
            if (!ptr) {
                throw std::bad_alloc();
            }
            return static_cast<T*>(ptr);
        }
        
        void deallocate(T* ptr, size_t) noexcept override {
            if (ptr) {
                manager_.deallocate(ptr);
            }
        }
        
        // STL兼容性要求
        template<typename U>
        bool operator==(const STLAllocator<U>& other) const noexcept {
            return &manager_ == &other.manager_;
        }
        
        template<typename U>
        bool operator!=(const STLAllocator<U>& other) const noexcept {
            return !(*this == other);
        }
        
        template<typename U>
        struct rebind {
            using other = STLAllocator<U>;
        };
        
    private:
        UnifiedMemoryManager& manager_;
        
        template<typename> friend class STLAllocator;
    };
    
    /**
     * @brief 获取STL分配器
     */
    template<typename T>
    STLAllocator<T> getSTLAllocator() {
        return STLAllocator<T>(*this);
    }
    
    // === 类型化分配支持 (兼容现有UnifiedMemoryManager) ===
    
    /**
     * @brief 类型化分配
     */
    template<typename T>
    T* allocateTyped(size_t count, const MemoryTraits& traits = MemoryTraits{}) {
        return allocateTyped<T>(count, MemoryPoolType::GENERAL_PURPOSE, traits);
    }
    
    template<typename T>
    T* allocateTyped(size_t count, MemoryPoolType poolType, const MemoryTraits& traits = MemoryTraits{}) {
        if (count == 0) return nullptr;
        
        size_t size = count * sizeof(T);
        size_t alignment = traits.alignment > 0 ? traits.alignment : alignof(T);
        
        void* ptr = allocateWithTraits(size, alignment, poolType, traits);
        return static_cast<T*>(ptr);
    }
    
    /**
     * @brief 类型化释放
     */
    template<typename T>
    void deallocateTyped(T* ptr, size_t count, MemoryPoolType poolType = MemoryPoolType::GENERAL_PURPOSE, 
                        const MemoryTraits& traits = MemoryTraits{}) {
        if (ptr) {
            deallocateWithTraits(ptr, count * sizeof(T), poolType, traits);
        }
    }
    
    // === 批量分配支持 ===
    
    /**
     * @brief 批量分配
     */
    std::vector<void*> allocateBatch(const std::vector<size_t>& sizes);
    void deallocateBatch(const std::vector<std::pair<void*, size_t>>& allocations);
    
    template<typename T>
    std::vector<T*> allocateTypedBatch(const std::vector<size_t>& counts, 
                                      MemoryPoolType poolType = MemoryPoolType::GENERAL_PURPOSE,
                                      const MemoryTraits& traits = MemoryTraits{}) {
        std::vector<T*> results;
        results.reserve(counts.size());
        
        for (size_t count : counts) {
            results.push_back(allocateTyped<T>(count, poolType, traits));
        }
        
        return results;
    }
    
    // === 内存压力监控 ===
    
    /**
     * @brief 内存压力级别
     */
    enum class MemoryPressureLevel { LOW, MEDIUM, HIGH, CRITICAL };
    
    /**
     * @brief 获取当前内存压力
     */
    MemoryPressureLevel getMemoryPressure() const noexcept;
    
    /**
     * @brief 设置内存压力回调
     */
    void setMemoryPressureCallback(std::function<void(MemoryPressureLevel)> callback);
    
    // === 内存优化功能 ===
    
    /**
     * @brief 触发垃圾回收
     */
    void triggerGarbageCollection();
    
    /**
     * @brief 优化内存布局
     */
    void optimizeMemoryLayout();
    
    /**
     * @brief 预分配内存池
     */
    void preAllocatePool(MemoryPoolType poolType, size_t sizeMB);
    
    // === 高级功能 (兼容现有接口) ===
    
    /**
     * @brief NUMA感知分配
     */
    void* allocateOnNUMANode(size_t size, int node);
    
    /**
     * @brief SIMD对齐分配
     */
    void* allocateSIMDAligned(size_t size, size_t vectorWidth = 64);
    
    /**
     * @brief 内存预取
     */
    void prefetchMemory(void* ptr, size_t size);
    
    // === 调试和统计 ===
    
    /**
     * @brief 转储分配信息 (调试模式)
     */
    void dumpAllocations() const;
    
    /**
     * @brief 内存健康检查
     */
    bool performHealthCheck() const;
    
    /**
     * @brief 获取详细统计
     */
    struct DetailedStats {
        MemoryUsageStats overall;
        std::unordered_map<MemoryPoolType, MemoryUsageStats> poolStats;
        size_t threadLocalCacheHits;
        size_t threadLocalCacheMisses;
        double averageAllocationTime;
        double averageDeallocationTime;
    };
    
    DetailedStats getDetailedStatistics() const;
    
    // === 工厂方法 ===
    
    /**
     * @brief 环境特定工厂方法
     */
    static std::unique_ptr<UnifiedMemoryManager> createForEnvironment(const std::string& environment);
    
    /**
     * @brief 获取详细统计信息
     */
    struct Statistics {
        size_t currentlyAllocated = 0;
        size_t maxAllocation = 0;
        size_t totalAllocations = 0;
        size_t totalDeallocations = 0;
        double averageAllocationSize = 0.0;
    };
    
    Statistics getStatistics() const;

private:
    // === 内部实现 ===
    
    struct MemoryPool;
    struct ThreadLocalCache;
    struct LargeDataManager;
    struct ConcurrentManager;
    struct AllocationRegistry;
    
    Config config_;
    AllocationStrategy allocationStrategy_;
    
    // 统计信息和线程安全
    MemoryUsageStats stats_;
    mutable std::mutex statsMutex_;
    
    // 多池管理
    std::vector<std::unique_ptr<MemoryPool>> pools_;
    std::atomic<size_t> currentPoolIndex_{0};
    
    // 专用管理器
    std::unique_ptr<LargeDataManager> largeDataManager_;
    std::unique_ptr<ConcurrentManager> concurrentManager_;
    std::unique_ptr<AllocationRegistry> allocationRegistry_;
    
    // 线程本地缓存
    static thread_local ThreadLocalCache tlsCache_;
    
    // 同步控制
    mutable std::shared_mutex poolMutex_;
    
    // 回调函数
    std::function<void(const MemoryUsageStats&)> memoryUsageCallback_;
    std::function<void(MemoryPressureLevel)> memoryPressureCallback_;
    
    // 内部方法
    void* allocateWithTraits(size_t size, size_t alignment, MemoryPoolType poolType, const MemoryTraits& traits);
    void deallocateWithTraits(void* ptr, size_t size, MemoryPoolType poolType, const MemoryTraits& traits);
    
    void* allocateFromPool(size_t size, size_t alignment);
    void* allocateFromSystem(size_t size, size_t alignment);
    void* allocateFromThreadCache(size_t size);
    
    bool deallocateToPool(void* ptr, size_t size);
    bool deallocateToThreadCache(void* ptr, size_t size);
    
    void updateStatistics(const std::string& operation, size_t size);
    void checkMemoryPressure();
    
    MemoryPool* getOrCreatePool(MemoryPoolType poolType);
    void initializePools();
    void shutdownPools();
};

} // namespace oscean::common_utils::memory 