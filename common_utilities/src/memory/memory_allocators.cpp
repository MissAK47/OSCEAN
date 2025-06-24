#include "common_utils/memory/memory_allocators.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/memory/memory_config.h"
#include <stdexcept>
#include <limits>
#include <new>
#include <mutex>
#include <memory>

namespace oscean::common_utils::memory {

namespace {
    // 全局默认内存管理器
    std::unique_ptr<UnifiedMemoryManager> g_defaultManager;
    std::mutex g_defaultManagerMutex;
    
    UnifiedMemoryManager* getOrCreateDefaultManager() {
        std::lock_guard<std::mutex> lock(g_defaultManagerMutex);
        if (!g_defaultManager) {
            Config config;
            config.environment = Environment::PRODUCTION;
            g_defaultManager = std::make_unique<UnifiedMemoryManager>(config);
        }
        return g_defaultManager.get();
    }
}

// === STLAllocator实现 ===

template<typename T>
STLAllocator<T>::STLAllocator() noexcept 
    : manager_(getOrCreateDefaultManager()) {
}

template<typename T>
STLAllocator<T>::STLAllocator(UnifiedMemoryManager& manager) noexcept 
    : manager_(&manager) {
}

template<typename T>
template<typename U>
STLAllocator<T>::STLAllocator(const STLAllocator<U>& other) noexcept 
    : manager_(other.manager_) {
}

template<typename T>
typename STLAllocator<T>::pointer STLAllocator<T>::allocate(size_type count) {
    if (count == 0) return nullptr;
    
    if (count > max_size()) {
        throw std::bad_alloc();
    }
    
    size_t totalSize = count * sizeof(T);
    void* ptr = manager_->allocate(totalSize, alignof(T));
    
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    return static_cast<pointer>(ptr);
}

template<typename T>
void STLAllocator<T>::deallocate(pointer ptr, size_type) noexcept {
    if (ptr) {
        manager_->deallocate(ptr);
    }
}

template<typename T>
template<typename U, typename... Args>
void STLAllocator<T>::construct(U* ptr, Args&&... args) {
    ::new(static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
}

template<typename T>
template<typename U>
void STLAllocator<T>::destroy(U* ptr) {
    ptr->~U();
}

template<typename T>
typename STLAllocator<T>::size_type STLAllocator<T>::max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(T);
}

template<typename T>
template<typename U>
bool STLAllocator<T>::operator==(const STLAllocator<U>& other) const noexcept {
    return manager_ == other.manager_;
}

template<typename T>
template<typename U>
bool STLAllocator<T>::operator!=(const STLAllocator<U>& other) const noexcept {
    return !(*this == other);
}

template<typename T>
UnifiedMemoryManager* STLAllocator<T>::getDefaultManager() {
    return getOrCreateDefaultManager();
}

// === SIMDAllocator实现 ===

template<typename T, size_t Alignment>
SIMDAllocator<T, Alignment>::SIMDAllocator(UnifiedMemoryManager& manager) noexcept 
    : manager_(&manager) {
}

template<typename T, size_t Alignment>
template<typename U>
SIMDAllocator<T, Alignment>::SIMDAllocator(const SIMDAllocator<U, Alignment>& other) noexcept 
    : manager_(other.manager_) {
}

template<typename T, size_t Alignment>
typename SIMDAllocator<T, Alignment>::pointer 
SIMDAllocator<T, Alignment>::allocate(size_type count) {
    if (count == 0) return nullptr;
    
    if (count > max_size()) {
        throw std::bad_alloc();
    }
    
    size_t totalSize = count * sizeof(T);
    size_t actualAlignment = std::max(Alignment, alignof(T));
    
    UnifiedMemoryManager* mgr = manager_ ? manager_ : getOrCreateDefaultManager();
    void* ptr = mgr->allocate(totalSize, actualAlignment);
    
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    return static_cast<pointer>(ptr);
}

template<typename T, size_t Alignment>
void SIMDAllocator<T, Alignment>::deallocate(pointer ptr, size_type) noexcept {
    if (ptr) {
        UnifiedMemoryManager* mgr = manager_ ? manager_ : getOrCreateDefaultManager();
        mgr->deallocate(ptr);
    }
}

template<typename T, size_t Alignment>
typename SIMDAllocator<T, Alignment>::size_type 
SIMDAllocator<T, Alignment>::max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(T);
}

template<typename T, size_t Alignment>
template<typename U>
bool SIMDAllocator<T, Alignment>::operator==(const SIMDAllocator<U, Alignment>& other) const noexcept {
    return manager_ == other.manager_;
}

template<typename T, size_t Alignment>
template<typename U>
bool SIMDAllocator<T, Alignment>::operator!=(const SIMDAllocator<U, Alignment>& other) const noexcept {
    return !(*this == other);
}

// === PoolAllocator实现 ===

template<typename T>
PoolAllocator<T>::PoolAllocator(MemoryPoolType poolType) noexcept 
    : manager_(getOrCreateDefaultManager()), poolType_(poolType) {
}

template<typename T>
PoolAllocator<T>::PoolAllocator(UnifiedMemoryManager& manager, MemoryPoolType poolType) noexcept 
    : manager_(&manager), poolType_(poolType) {
}

template<typename T>
template<typename U>
PoolAllocator<T>::PoolAllocator(const PoolAllocator<U>& other) noexcept 
    : manager_(other.manager_), poolType_(other.poolType_) {
}

template<typename T>
typename PoolAllocator<T>::pointer PoolAllocator<T>::allocate(size_type count) {
    if (count == 0) return nullptr;
    
    if (count > max_size()) {
        throw std::bad_alloc();
    }
    
    size_t totalSize = count * sizeof(T);
    void* ptr = manager_->allocate(totalSize, alignof(T));
    
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    return static_cast<pointer>(ptr);
}

template<typename T>
void PoolAllocator<T>::deallocate(pointer ptr, size_type) noexcept {
    if (ptr) {
        manager_->deallocate(ptr);
    }
}

template<typename T>
typename PoolAllocator<T>::size_type PoolAllocator<T>::max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(T);
}

template<typename T>
template<typename U>
bool PoolAllocator<T>::operator==(const PoolAllocator<U>& other) const noexcept {
    return manager_ == other.manager_ && poolType_ == other.poolType_;
}

template<typename T>
template<typename U>
bool PoolAllocator<T>::operator!=(const PoolAllocator<U>& other) const noexcept {
    return !(*this == other);
}

// === AllocatorFactory实现 ===

template<typename T>
STLAllocator<T> AllocatorFactory::createSTLAllocator() {
    return STLAllocator<T>();
}

template<typename T>
STLAllocator<T> AllocatorFactory::createSTLAllocator(UnifiedMemoryManager& manager) {
    return STLAllocator<T>(manager);
}

template<typename T, size_t Alignment>
SIMDAllocator<T, Alignment> AllocatorFactory::createSIMDAllocator() {
    return SIMDAllocator<T, Alignment>();
}

template<typename T, size_t Alignment>
SIMDAllocator<T, Alignment> AllocatorFactory::createSIMDAllocator(UnifiedMemoryManager& manager) {
    return SIMDAllocator<T, Alignment>(manager);
}

template<typename T>
PoolAllocator<T> AllocatorFactory::createPoolAllocator(MemoryPoolType poolType) {
    return PoolAllocator<T>(poolType);
}

template<typename T>
PoolAllocator<T> AllocatorFactory::createPoolAllocator(UnifiedMemoryManager& manager, MemoryPoolType poolType) {
    return PoolAllocator<T>(manager, poolType);
}

// === 显式模板实例化（常用类型） ===

// STLAllocator实例化
template class STLAllocator<char>;
template class STLAllocator<int>;
template class STLAllocator<float>;
template class STLAllocator<double>;
template class STLAllocator<void*>;

// SIMDAllocator实例化  
template class SIMDAllocator<float, 64>;
template class SIMDAllocator<double, 64>;
template class SIMDAllocator<int, 64>;

// PoolAllocator实例化
template class PoolAllocator<char>;
template class PoolAllocator<int>;
template class PoolAllocator<float>;
template class PoolAllocator<double>;

// === 获取默认内存管理器 ===
UnifiedMemoryManager* getDefaultManager() {
    static std::unique_ptr<UnifiedMemoryManager> defaultManager;
    static std::once_flag initFlag;
    
    std::call_once(initFlag, []() {
        // 🔧 直接创建UnifiedMemoryManager
        Config config = Config::optimizeForEnvironment(Environment::PRODUCTION);
        defaultManager = std::make_unique<UnifiedMemoryManager>(config);
        if (defaultManager) {
            defaultManager->initialize();
        }
    });
    
    return defaultManager.get();
}

} // namespace oscean::common_utils::memory 