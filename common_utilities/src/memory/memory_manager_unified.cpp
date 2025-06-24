#include "common_utils/memory/memory_manager_unified.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>

#ifdef _WIN32
#include <malloc.h>
namespace {
    inline void* aligned_alloc_compat(size_t alignment, size_t size) {
        // 确保大小是对齐值的倍数
        size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
        return _aligned_malloc(aligned_size, alignment);
    }
    
    inline void aligned_free_compat(void* ptr) {
        _aligned_free(ptr);
    }
}
#else
#include <cstdlib>
namespace {
    inline void* aligned_alloc_compat(size_t alignment, size_t size) {
        // 确保大小是对齐值的倍数
        size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
        return aligned_alloc(alignment, aligned_size);
    }
    
    inline void aligned_free_compat(void* ptr) {
        std::free(ptr);
    }
}
#endif

namespace oscean::common_utils::memory {

// === 内存管理常量定义 ===
namespace {
    // 🔧 统一的快速路径阈值：256字节以下的小对象使用快速分配，不记录统计信息
    constexpr size_t FAST_PATH_THRESHOLD = 256;
    // 🔧 中等对象阈值：2048字节以下的对象使用直接系统分配
    constexpr size_t MEDIUM_PATH_THRESHOLD = 2048;
    // 🔧 线程缓存阈值：1024字节以下的对象可以尝试线程缓存
    constexpr size_t THREAD_CACHE_THRESHOLD = 1024;
}

// === 内部结构体定义 ===

struct UnifiedMemoryManager::MemoryPool {
    void* memory;
    size_t size;
    size_t used;
    std::vector<void*> freeBlocks;
    MemoryPoolType poolType;
    std::mutex poolMutex;
    
    MemoryPool(size_t s, MemoryPoolType type) 
        : memory(nullptr), size(s), used(0), poolType(type) {
        memory = aligned_alloc_compat(64, size);
        if (!memory) {
            throw std::bad_alloc();
        }
        std::memset(memory, 0, size);
    }
    
    ~MemoryPool() {
        if (memory) {
            aligned_free_compat(memory);
        }
    }
};

struct UnifiedMemoryManager::ThreadLocalCache {
    static constexpr size_t CACHE_SIZE = 16;
    struct CacheEntry {
        void* ptr;
        size_t size;
        bool inUse;
    };
    
    CacheEntry entries[CACHE_SIZE];
    size_t nextIndex = 0;
    
    ThreadLocalCache() {
        for (auto& entry : entries) {
            entry.ptr = nullptr;
            entry.size = 0;
            entry.inUse = false;
        }
    }
};

struct UnifiedMemoryManager::LargeDataManager {
    std::unordered_map<void*, size_t> largeAllocations;
    std::mutex largeMutex;
    size_t totalLargeMemory = 0;
    
    void* allocateLarge(size_t size) {
        void* ptr = aligned_alloc_compat(64, size);
        if (ptr) {
            std::lock_guard<std::mutex> lock(largeMutex);
            largeAllocations[ptr] = size;
            totalLargeMemory += size;
        }
        return ptr;
    }
    
    bool deallocateLarge(void* ptr) {
        std::lock_guard<std::mutex> lock(largeMutex);
        auto it = largeAllocations.find(ptr);
        if (it != largeAllocations.end()) {
            totalLargeMemory -= it->second;
            largeAllocations.erase(it);
            aligned_free_compat(ptr);
            return true;
        }
        return false;
    }
};

struct UnifiedMemoryManager::AllocationRegistry {
    std::unordered_map<void*, size_t> allocations;
    std::mutex registryMutex;
    
    void recordAllocation(void* ptr, size_t size) {
        if (ptr) {
            std::lock_guard<std::mutex> lock(registryMutex);
            allocations[ptr] = size;
        }
    }
    
    size_t getAllocationSize(void* ptr) {
        if (!ptr) return 0;
        std::lock_guard<std::mutex> lock(registryMutex);
        auto it = allocations.find(ptr);
        return (it != allocations.end()) ? it->second : 0;
    }
    
    bool removeAllocation(void* ptr) {
        if (!ptr) return false;
        std::lock_guard<std::mutex> lock(registryMutex);
        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            allocations.erase(it);
            return true;
        }
        return false;
    }
};

struct UnifiedMemoryManager::ConcurrentManager {
    std::atomic<size_t> concurrentAllocations{0};
    std::atomic<size_t> maxConcurrentAllocations{0};
    
    void recordAllocation() {
        size_t current = concurrentAllocations.fetch_add(1) + 1;
        size_t maxVal = maxConcurrentAllocations.load();
        while (current > maxVal && 
               !maxConcurrentAllocations.compare_exchange_weak(maxVal, current)) {
            // 空循环，直到成功更新最大值
        }
    }
    
    void recordDeallocation() {
        concurrentAllocations.fetch_sub(1);
    }
};

// === 线程本地缓存 ===
thread_local UnifiedMemoryManager::ThreadLocalCache UnifiedMemoryManager::tlsCache_;

// === UnifiedMemoryManager实现 ===

UnifiedMemoryManager::UnifiedMemoryManager(const Config& config) 
    : config_(config), allocationStrategy_(AllocationStrategy::BEST_FIT) {
    largeDataManager_ = std::make_unique<LargeDataManager>();
    concurrentManager_ = std::make_unique<ConcurrentManager>();
    allocationRegistry_ = std::make_unique<AllocationRegistry>();
}

UnifiedMemoryManager::~UnifiedMemoryManager() {
    shutdown();
}

bool UnifiedMemoryManager::initialize() {
    try {
        initializePools();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize memory manager: " << e.what() << std::endl;
        return false;
    }
}

void UnifiedMemoryManager::shutdown() {
    shutdownPools();
}

void* UnifiedMemoryManager::allocate(size_t size, size_t alignment, const std::string& tag) {
    if (size == 0) return nullptr;
    
    // 如果指定了对齐要求，不使用快速路径
    if (alignment > 8) {
        // 使用标准路径确保正确对齐
        void* ptr = allocateFromSystem(size, alignment);
        if (ptr) {
            allocationRegistry_->recordAllocation(ptr, size);
            updateStatistics("aligned_allocation", size);
        }
        return ptr;
    }
    
    // 🚀 超快速路径：256字节以下的小对象直接分配，完全跳过所有管理开销
    if (size <= FAST_PATH_THRESHOLD) {
        // 🚀 内联最快的分配路径，避免函数调用开销
        size_t useAlignment = (alignment == 0) ? 8 : alignment;
#ifdef _WIN32
        return _aligned_malloc(size, useAlignment);  // 直接调用最快的Windows API
#else
        return aligned_alloc(useAlignment, size);    // 直接调用最快的Linux API
#endif
    }
    
    // 🔧 标准路径：需要完整的管理和统计（仅用于大对象）
    concurrentManager_->recordAllocation();
    
    void* ptr = nullptr;
    
    // 🏊‍♂️ 中等对象优先尝试池分配
    if (size <= MEDIUM_PATH_THRESHOLD) {
        ptr = allocateFromPool(size, alignment);
        if (ptr) {
            allocationRegistry_->recordAllocation(ptr, size);
            updateStatistics("pool_allocation", size);
            return ptr;
        }
        
        // 池分配失败，降级到系统分配
        size_t useAlignment = (alignment == 0 || alignment > 64) ? 64 : alignment;
        ptr = aligned_alloc_compat(useAlignment, size);
        if (ptr) {
            allocationRegistry_->recordAllocation(ptr, size);
            updateStatistics("system_allocation", size);
        }
        return ptr;
    }
    
    // 大对象路径
    ptr = allocateFromSystem(size, alignment);
    if (ptr) {
        allocationRegistry_->recordAllocation(ptr, size);
        updateStatistics("system_allocation", size);
    }
    
    return ptr;
}

void UnifiedMemoryManager::deallocate(void* ptr) {
    if (!ptr) return;
    
    // 🚀 超快速路径检测：如果不在注册表中，很可能是快速路径分配的小对象
    size_t allocSize = 0;
    {
        std::lock_guard<std::mutex> lock(allocationRegistry_->registryMutex);
        auto it = allocationRegistry_->allocations.find(ptr);
        if (it != allocationRegistry_->allocations.end()) {
            allocSize = it->second;
            allocationRegistry_->allocations.erase(it);
        }
    }
    
    if (allocSize > 0) {
        // 🔧 标准路径：有记录的对象，完整释放流程
        concurrentManager_->recordDeallocation();
        
        // 尝试放入线程缓存
        if (allocSize <= THREAD_CACHE_THRESHOLD && deallocateToThreadCache(ptr, allocSize)) {
            updateStatistics("thread_cache_deallocation", allocSize);
            return;
        }
        
        // 返回池中
        if (deallocateToPool(ptr, allocSize)) {
            updateStatistics("pool_deallocation", allocSize);
            return;
        }
        
        // 系统释放
        aligned_free_compat(ptr);
        updateStatistics("system_deallocation", allocSize);
        return;
    }
    
    // 🚀 快速路径释放：可能是256字节以下的小对象，直接释放
    // 简单的有效性检查
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr < 0x1000) {  // 只排除明显无效的地址
        return;
    }
    
    // 🚀 直接释放，最小化开销
#ifdef _WIN32
    _aligned_free(ptr);  // 直接调用最快的Windows API
#else
    std::free(ptr);      // 直接调用最快的Linux API
#endif
}

void* UnifiedMemoryManager::reallocate(void* ptr, size_t newSize, const std::string& tag) {
    if (!ptr) {
        return allocate(newSize, 0, tag);
    }
    
    if (newSize == 0) {
        deallocate(ptr);
        return nullptr;
    }
    
    // 🔧 修复：获取原始分配大小
    size_t oldSize = allocationRegistry_->getAllocationSize(ptr);
    if (oldSize == 0) {
        // 检查是否为大对象
        std::lock_guard<std::mutex> lock(largeDataManager_->largeMutex);
        auto it = largeDataManager_->largeAllocations.find(ptr);
        if (it != largeDataManager_->largeAllocations.end()) {
            oldSize = it->second;
        }
    }
    
    // 分配新内存
    void* newPtr = allocate(newSize, 0, tag);
    if (newPtr && ptr && oldSize > 0) {
        // 🔧 修复：使用正确的拷贝大小 - min(oldSize, newSize)
        size_t copySize = std::min(oldSize, newSize);
        std::memcpy(newPtr, ptr, copySize);
        deallocate(ptr);
    } else if (!newPtr) {
        // 分配失败，不释放原内存
        return nullptr;
    }
    
    return newPtr;
}

bool UnifiedMemoryManager::isManaged(void* ptr) const {
    // 检查是否在任何池中
    std::shared_lock<std::shared_mutex> lock(poolMutex_);
    for (const auto& pool : pools_) {
        char* poolStart = static_cast<char*>(pool->memory);
        char* poolEnd = poolStart + pool->size;
        char* ptrAddr = static_cast<char*>(ptr);
        
        if (ptrAddr >= poolStart && ptrAddr < poolEnd) {
            return true;
        }
    }
    
    // 检查大对象
    std::lock_guard<std::mutex> largeLock(largeDataManager_->largeMutex);
    return largeDataManager_->largeAllocations.find(ptr) != 
           largeDataManager_->largeAllocations.end();
}

size_t UnifiedMemoryManager::getPoolSize() const {
    std::shared_lock<std::shared_mutex> lock(poolMutex_);
    size_t totalSize = 0;
    for (const auto& pool : pools_) {
        totalSize += pool->size;
    }
    return totalSize + largeDataManager_->totalLargeMemory;
}

size_t UnifiedMemoryManager::getAvailableMemory() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    size_t totalUsed = stats_.currentUsed;
    size_t maxMemory = config_.maxTotalMemoryMB * 1024 * 1024;
    return maxMemory > totalUsed ? maxMemory - totalUsed : 0;
}

MemoryUsageStats UnifiedMemoryManager::getUsageStats() const {
    return stats_;
}

void UnifiedMemoryManager::setAllocationStrategy(AllocationStrategy strategy) {
    allocationStrategy_ = strategy;
}

void UnifiedMemoryManager::setMemoryUsageCallback(
    std::function<void(const MemoryUsageStats&)> callback) {
    memoryUsageCallback_ = callback;
}

// === 流式缓冲区实现 ===

UnifiedMemoryManager::StreamingBuffer::StreamingBuffer(
    UnifiedMemoryManager& manager, size_t maxSize)
    : manager_(manager), maxSize_(maxSize), currentSize_(0) {
    buffer_ = manager_.allocate(maxSize);
}

UnifiedMemoryManager::StreamingBuffer::~StreamingBuffer() {
    if (buffer_) {
        manager_.deallocate(buffer_);
    }
}

void* UnifiedMemoryManager::StreamingBuffer::getWriteBuffer(size_t size) {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    if (size > maxSize_) return nullptr;
    
    currentSize_ = size;
    return buffer_;
}

void UnifiedMemoryManager::StreamingBuffer::commitBuffer(size_t actualSize) {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    currentSize_ = std::min(actualSize, maxSize_);
}

const void* UnifiedMemoryManager::StreamingBuffer::getReadBuffer() const {
    return buffer_;
}

size_t UnifiedMemoryManager::StreamingBuffer::getBufferSize() const noexcept {
    return currentSize_;
}

void UnifiedMemoryManager::StreamingBuffer::resetBuffer() noexcept {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    currentSize_ = 0;
}

std::unique_ptr<UnifiedMemoryManager::StreamingBuffer> 
UnifiedMemoryManager::createStreamingBuffer(size_t maxSize) {
    return std::make_unique<StreamingBuffer>(*this, maxSize);
}

// === 并发分配器实现 ===

UnifiedMemoryManager::ConcurrentAllocator::ConcurrentAllocator(
    UnifiedMemoryManager& manager) : manager_(manager) {}

UnifiedMemoryManager::ConcurrentAllocator::~ConcurrentAllocator() = default;

void* UnifiedMemoryManager::ConcurrentAllocator::allocate(size_t size) noexcept {
    try {
        return manager_.allocate(size);
    } catch (...) {
        return nullptr;
    }
}

void UnifiedMemoryManager::ConcurrentAllocator::deallocate(void* ptr) noexcept {
    manager_.deallocate(ptr);
}

MemoryUsageStats UnifiedMemoryManager::ConcurrentAllocator::getThreadStats() const noexcept {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return threadStats_;
}

std::unique_ptr<UnifiedMemoryManager::ConcurrentAllocator> 
UnifiedMemoryManager::createConcurrentAllocator() {
    return std::make_unique<ConcurrentAllocator>(*this);
}

// === 内部实现方法 ===

void UnifiedMemoryManager::initializePools() {
    std::lock_guard<std::shared_mutex> lock(poolMutex_);
    
    // 创建基础池
    size_t poolSize = config_.chunkSizeMB * 1024 * 1024;
    pools_.push_back(std::make_unique<MemoryPool>(poolSize, MemoryPoolType::GENERAL_PURPOSE));
    
    // 根据配置创建其他池
    if (config_.enableSIMDOptimization) {
        pools_.push_back(std::make_unique<MemoryPool>(poolSize / 2, MemoryPoolType::SIMD_ALIGNED));
    }
    
    if (config_.enableLargeDataSupport) {
        pools_.push_back(std::make_unique<MemoryPool>(poolSize * 2, MemoryPoolType::LARGE_OBJECTS));
    }
}

void UnifiedMemoryManager::shutdownPools() {
    std::lock_guard<std::shared_mutex> lock(poolMutex_);
    pools_.clear();
}

void* UnifiedMemoryManager::allocateFromPool(size_t size, size_t alignment) {
    std::shared_lock<std::shared_mutex> lock(poolMutex_);
    
    for (auto& pool : pools_) {
        std::lock_guard<std::mutex> poolLock(pool->poolMutex);
        
        // 🚀 超快速查找：优先查看freeBlocks中的可用内存块
        for (auto it = pool->freeBlocks.begin(); it != pool->freeBlocks.end(); ++it) {
            void* freePtr = *it;
            
            // 简单对齐检查
            uintptr_t addr = reinterpret_cast<uintptr_t>(freePtr);
            if (alignment > 0 && (addr % alignment) != 0) {
                continue;  // 对齐不满足，继续查找
            }
            
            // 找到可用块，立即返回
            pool->freeBlocks.erase(it);
            return freePtr;
        }
        
        // 🚀 如果freeBlocks中没有，从池尾分配新内存
        if (pool->used + size <= pool->size) {
            char* basePtr = static_cast<char*>(pool->memory) + pool->used;
            
            // 处理对齐
            uintptr_t addr = reinterpret_cast<uintptr_t>(basePtr);
            size_t alignedSize = size;
            
            if (alignment > 0) {
                uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
                alignedSize = size + (aligned - addr);
                basePtr = reinterpret_cast<char*>(aligned);
            }
            
            if (pool->used + alignedSize <= pool->size) {
                pool->used += alignedSize;
                return basePtr;
            }
        }
    }
    
    return nullptr;  // 所有池都满了
}

void* UnifiedMemoryManager::allocateFromSystem(size_t size, size_t alignment) {
    // 🔧 修复：确保对齐参数是2的幂次且至少为8
    if (alignment == 0) {
        alignment = config_.alignmentSize; // 使用配置的默认对齐
    }
    
    // 🔧 修复：只有当对齐值不是2的幂次时才调整
    if ((alignment & (alignment - 1)) != 0) {
        // 向上调整到最近的2的幂次
        size_t powerOfTwo = 1;
        while (powerOfTwo < alignment) {
            powerOfTwo <<= 1;
        }
        alignment = powerOfTwo;
    }
    
    // 最小对齐为8字节
    if (alignment < 8) {
        alignment = 8;
    }
    
    return aligned_alloc_compat(alignment, size);
}

void* UnifiedMemoryManager::allocateFromThreadCache(size_t size) {
    // 简化的线程缓存实现
    for (auto& entry : tlsCache_.entries) {
        if (!entry.inUse && entry.size >= size) {
            entry.inUse = true;
            return entry.ptr;
        }
    }
    return nullptr;
}

bool UnifiedMemoryManager::deallocateToPool(void* ptr, size_t size) {
    // 🔧 实现真正的池释放功能而不是简化实现
    if (!ptr || size == 0) return false;
    
    std::shared_lock<std::shared_mutex> lock(poolMutex_);
    
    for (auto& pool : pools_) {
        std::lock_guard<std::mutex> poolLock(pool->poolMutex);
        
        // 检查指针是否在此池的范围内
        char* poolStart = static_cast<char*>(pool->memory);
        char* poolEnd = poolStart + pool->size;
        char* ptrAddr = static_cast<char*>(ptr);
        
        if (ptrAddr >= poolStart && ptrAddr < poolEnd) {
            // 将释放的内存块添加到自由块列表
            pool->freeBlocks.push_back(ptr);
            
            // 🔧 可选优化：如果释放的是池末尾的内存，直接减少used
            if (ptrAddr + size == poolStart + pool->used) {
                pool->used -= size;
                // 从freeBlocks中移除刚添加的指针
                if (!pool->freeBlocks.empty()) {
                    pool->freeBlocks.pop_back();
                }
            }
            
            return true;  // 成功释放到池
        }
    }
    
    return false;  // 不在任何池中
}

bool UnifiedMemoryManager::deallocateToThreadCache(void* ptr, size_t size) {
    // 简化的线程缓存实现
    for (auto& entry : tlsCache_.entries) {
        if (entry.ptr == ptr) {
            entry.inUse = false;
            return true;
        }
    }
    
    // 尝试添加到空闲槽
    for (auto& entry : tlsCache_.entries) {
        if (entry.ptr == nullptr) {
            entry.ptr = ptr;
            entry.size = size;
            entry.inUse = false;
            return true;
        }
    }
    
    return false;
}

void UnifiedMemoryManager::updateStatistics(const std::string& operation, size_t size) {
    std::lock_guard<std::mutex> lock(statsMutex_);  // 添加mutex保护
    
    // 🔧 修复：先检查deallocation，避免与allocation冲突
    if (operation.find("deallocation") != std::string::npos) {
        stats_.deallocationCount++;
        if (size > 0 && stats_.currentUsed >= size) {
            stats_.currentUsed -= size;
        }
    } else if (operation.find("allocation") != std::string::npos) {
        stats_.allocationCount++;
        stats_.currentUsed += size;
        stats_.totalAllocated += size;
        
        // 更新峰值使用量
        if (stats_.currentUsed > stats_.peakUsage) {
            stats_.peakUsage = stats_.currentUsed;
        }
    }
    
    // 调用用户回调
    if (memoryUsageCallback_) {
        memoryUsageCallback_(stats_);
    }
}

void UnifiedMemoryManager::checkMemoryPressure() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    size_t currentUsageMB = stats_.currentUsed / (1024 * 1024);
    if (currentUsageMB > config_.memoryPressureThresholdMB) {
        if (memoryPressureCallback_) {
            memoryPressureCallback_(MemoryPressureLevel::HIGH);
        }
    }
}

UnifiedMemoryManager::MemoryPressureLevel 
UnifiedMemoryManager::getMemoryPressure() const noexcept {
    std::lock_guard<std::mutex> lock(statsMutex_);
    size_t currentUsageMB = stats_.currentUsed / (1024 * 1024);
    size_t threshold = config_.memoryPressureThresholdMB;
    
    if (currentUsageMB < threshold / 2) {
        return MemoryPressureLevel::LOW;
    } else if (currentUsageMB < threshold * 3 / 4) {
        return MemoryPressureLevel::MEDIUM;
    } else if (currentUsageMB < threshold) {
        return MemoryPressureLevel::HIGH;
    } else {
        return MemoryPressureLevel::CRITICAL;
    }
}

void UnifiedMemoryManager::setMemoryPressureCallback(
    std::function<void(MemoryPressureLevel)> callback) {
    memoryPressureCallback_ = callback;
}

void UnifiedMemoryManager::triggerGarbageCollection() {
    // 🔧 修复：安全的垃圾回收实现，避免双重释放
    for (auto& entry : tlsCache_.entries) {
        if (!entry.inUse && entry.ptr) {
            // 🔧 检查指针是否在分配注册表中，避免双重释放
            size_t allocSize = allocationRegistry_->getAllocationSize(entry.ptr);
            if (allocSize > 0) {
                // 从注册表中移除
                allocationRegistry_->removeAllocation(entry.ptr);
                // 释放内存
                aligned_free_compat(entry.ptr);
                // 更新统计
                updateStatistics("garbage_collection_deallocation", allocSize);
            }
            // 清理缓存条目
            entry.ptr = nullptr;
            entry.size = 0;
        }
    }
}

void UnifiedMemoryManager::optimizeMemoryLayout() {
    // 简化实现：触发垃圾回收
    triggerGarbageCollection();
}

void UnifiedMemoryManager::dumpAllocations() const {
    std::cout << "=== Memory Manager Statistics ===" << std::endl;
    
    // 🔧 手动输出统计信息，因为toString()方法不存在
    std::lock_guard<std::mutex> lock(statsMutex_);
    std::cout << "Total Allocated: " << stats_.totalAllocated << " bytes" << std::endl;
    std::cout << "Currently Used: " << stats_.currentUsed << " bytes" << std::endl;
    std::cout << "Peak Usage: " << stats_.peakUsage << " bytes" << std::endl;
    std::cout << "Allocation Count: " << stats_.allocationCount << std::endl;
    std::cout << "Deallocation Count: " << stats_.deallocationCount << std::endl;
    std::cout << "Fragmentation Ratio: " << stats_.fragmentationRatio << std::endl;
    
    std::cout << "\n=== Pool Information ===" << std::endl;
    std::shared_lock<std::shared_mutex> poolLock(poolMutex_);
    for (size_t i = 0; i < pools_.size(); ++i) {
        const auto& pool = pools_[i];
        std::lock_guard<std::mutex> poolMutexLock(pool->poolMutex);
        std::cout << "Pool " << i << ": " << pool->used << "/" << pool->size 
                  << " bytes used" << std::endl;
    }
    
    std::cout << "\n=== Large Objects ===" << std::endl;
    std::lock_guard<std::mutex> largeLock(largeDataManager_->largeMutex);
    std::cout << "Large objects count: " << largeDataManager_->largeAllocations.size() << std::endl;
    std::cout << "Total large memory: " << largeDataManager_->totalLargeMemory << " bytes" << std::endl;
}

bool UnifiedMemoryManager::performHealthCheck() const {
    // 🔧 简化健康检查实现
    std::lock_guard<std::mutex> lock(statsMutex_);
    bool memoryHealthy = (stats_.fragmentationRatio < 0.5) && 
                        (getMemoryPressure() != MemoryPressureLevel::CRITICAL);
    return memoryHealthy;
}

// === 类型化分配支持实现 ===

void* UnifiedMemoryManager::allocateWithTraits(size_t size, size_t alignment, 
                                              MemoryPoolType poolType, const MemoryTraits& traits) {
    // 简化实现：使用标准分配，考虑traits中的特殊要求
    if (traits.zeroInitialized) {
        void* ptr = allocate(size, alignment);
        if (ptr) {
            std::memset(ptr, 0, size);
        }
        return ptr;
    }
    
    return allocate(size, alignment, traits.memoryTag);
}

void UnifiedMemoryManager::deallocateWithTraits(void* ptr, size_t size, 
                                               MemoryPoolType poolType, const MemoryTraits& traits) {
    deallocate(ptr);
}

// === 批量分配支持实现 ===

std::vector<void*> UnifiedMemoryManager::allocateBatch(const std::vector<size_t>& sizes) {
    std::vector<void*> results;
    results.reserve(sizes.size());
    
    if (sizes.empty()) {
        return results;
    }
    
    // 🚀 极简化批量分配：对于小对象，直接使用最快路径，完全跳过记录
    for (size_t size : sizes) {
        if (size <= FAST_PATH_THRESHOLD) {
            // 🚀 快速路径：直接分配，无任何记录或统计开销
#ifdef _WIN32
            results.push_back(_aligned_malloc(size, 8));
#else
            results.push_back(aligned_alloc(8, size));
#endif
        } else {
            // 大对象：使用标准路径
            results.push_back(allocate(size));
        }
    }
    
    return results;
}

void UnifiedMemoryManager::deallocateBatch(const std::vector<std::pair<void*, size_t>>& allocations) {
    if (allocations.empty()) return;
    
    size_t fastPathCount = 0;
    size_t trackedDeallocations = 0;
    size_t totalDeallocatedTracked = 0;
    
    for (const auto& alloc : allocations) {
        void* ptr = alloc.first;
        size_t size = alloc.second;
        
        if (!ptr) continue;
        
        if (size <= FAST_PATH_THRESHOLD) {
            // 🔧 快速路径对象：直接释放，不做记录（与分配时一致）
            aligned_free_compat(ptr);
            fastPathCount++;
        } else {
            // 🔧 跟踪对象：使用标准释放路径
            deallocate(ptr);
            trackedDeallocations++;
            totalDeallocatedTracked += size;
        }
    }
    
    // 🔧 批量统计更新（只统计释放计数，不重复统计内存）
    if (fastPathCount > 0) {
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        // 只增加释放计数，不调整内存使用量（因为快速路径分配时也没统计）
        stats_.deallocationCount += fastPathCount;
    }
    
    // 注意：trackedDeallocations 的统计已经在 deallocate() 调用中处理了
}

// === 高级功能实现 ===

void* UnifiedMemoryManager::allocateOnNUMANode(size_t size, int node) {
    // 简化实现：忽略NUMA节点，使用标准分配
    return allocate(size);
}

void* UnifiedMemoryManager::allocateSIMDAligned(size_t size, size_t vectorWidth) {
    // 确保vectorWidth是有效的对齐值（2的幂次）
    if (vectorWidth == 0 || (vectorWidth & (vectorWidth - 1)) != 0) {
        vectorWidth = 32; // 默认AVX2对齐
    }
    
    // 强制使用系统分配确保正确对齐
    return allocateFromSystem(size, vectorWidth);
}

void UnifiedMemoryManager::prefetchMemory(void* ptr, size_t size) {
    // 简化实现：在实际项目中可以使用__builtin_prefetch
    (void)ptr;
    (void)size;
}

void UnifiedMemoryManager::preAllocatePool(MemoryPoolType poolType, size_t sizeMB) {
    // 简化实现：创建指定大小的池
    std::lock_guard<std::shared_mutex> lock(poolMutex_);
    size_t poolSize = sizeMB * 1024 * 1024;
    pools_.push_back(std::make_unique<MemoryPool>(poolSize, poolType));
}

// === 详细统计实现 ===

UnifiedMemoryManager::DetailedStats UnifiedMemoryManager::getDetailedStatistics() const {
    DetailedStats detailedStats;
    detailedStats.overall = stats_;
    
    // 收集各池的统计信息
    std::shared_lock<std::shared_mutex> lock(poolMutex_);
    for (const auto& pool : pools_) {
        std::lock_guard<std::mutex> poolLock(pool->poolMutex);
        
        MemoryUsageStats poolStats;
        poolStats.totalAllocated = pool->size;
        poolStats.currentUsed = pool->used;
        poolStats.peakUsage = pool->used; // 简化
        
        detailedStats.poolStats[pool->poolType] = poolStats;
    }
    
    // 线程本地缓存统计
    detailedStats.threadLocalCacheHits = 0;
    detailedStats.threadLocalCacheMisses = 0;
    
    // 性能统计
    detailedStats.averageAllocationTime = 0.001; // 1ms 简化值
    detailedStats.averageDeallocationTime = 0.0005; // 0.5ms 简化值
    
    return detailedStats;
}

UnifiedMemoryManager::MemoryPool* UnifiedMemoryManager::getOrCreatePool(MemoryPoolType poolType) {
    std::shared_lock<std::shared_mutex> lock(poolMutex_);
    
    // 查找现有池
    for (const auto& pool : pools_) {
        if (pool->poolType == poolType) {
            return pool.get();
        }
    }
    
    // 如果没找到，创建新池
    lock.unlock();
    std::lock_guard<std::shared_mutex> writeLock(poolMutex_);
    
    // 双重检查
    for (const auto& pool : pools_) {
        if (pool->poolType == poolType) {
            return pool.get();
        }
    }
    
    // 创建新池
    size_t poolSize = config_.chunkSizeMB * 1024 * 1024;
    pools_.push_back(std::make_unique<MemoryPool>(poolSize, poolType));
    return pools_.back().get();
}

// === 工厂方法实现 ===

std::unique_ptr<UnifiedMemoryManager> UnifiedMemoryManager::createForEnvironment(const std::string& environment) {
    Config config;
    
    if (environment == "hpc") {
        config.chunkSizeMB = 2048;  // 2GB for HPC
        config.largeDataThresholdMB = 128;
        config.enableThreadLocalCache = true;
        config.enableSIMDOptimization = true;
    } else if (environment == "production") {
        config.chunkSizeMB = 1024;  // 1GB for production
        config.largeDataThresholdMB = 64;
        config.enableThreadLocalCache = true;
        config.enableSIMDOptimization = false;
    } else if (environment == "testing") {
        config.chunkSizeMB = 128;   // 128MB for testing
        config.largeDataThresholdMB = 16;
        config.enableThreadLocalCache = false;
        config.enableSIMDOptimization = false;
    } else { // development
        config.chunkSizeMB = 256;   // 256MB for development
        config.largeDataThresholdMB = 32;
        config.enableThreadLocalCache = true;
        config.enableSIMDOptimization = false;
    }
    
    auto manager = std::make_unique<UnifiedMemoryManager>(config);
    if (!manager->initialize()) {
        throw std::runtime_error("Failed to initialize memory manager for " + environment);
    }
    
    return manager;
}

UnifiedMemoryManager::Statistics UnifiedMemoryManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    Statistics stats;
    stats.currentlyAllocated = stats_.currentUsed;
    stats.maxAllocation = stats_.peakUsage;
    stats.totalAllocations = stats_.allocationCount;
    stats.totalDeallocations = stats_.deallocationCount;
    
    if (stats.totalAllocations > 0) {
        stats.averageAllocationSize = static_cast<double>(stats.currentlyAllocated) / stats.totalAllocations;
    }
    
    return stats;
}

} // namespace oscean::common_utils::memory 
