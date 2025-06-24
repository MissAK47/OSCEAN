#pragma once

/**
 * @file memory_allocators.h
 * @brief STL兼容分配器 - 统一的STL分配器实现
 * 
 * 重构目标：
 * ✅ 提供STL兼容的分配器接口
 * ✅ 支持自定义内存池分配
 * ✅ 统一所有STL容器的内存管理
 * ✅ 高性能SIMD对齐分配
 */

#include "memory_config.h"
#include "memory_interfaces.h"
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>
#include <map>
#include <deque>
#include <list>
#include <algorithm>
#include <chrono>
#include <functional>
#include <type_traits>

namespace oscean::common_utils::memory {

// 前向声明
class UnifiedMemoryManager;

/**
 * @brief STL兼容的通用分配器
 */
template<typename T>
class STLAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = STLAllocator<U>;
    };
    
    STLAllocator() noexcept;
    explicit STLAllocator(UnifiedMemoryManager& manager) noexcept;
    
    template<typename U>
    STLAllocator(const STLAllocator<U>& other) noexcept;
    
    ~STLAllocator() = default;
    
    pointer allocate(size_type count);
    void deallocate(pointer ptr, size_type count) noexcept;
    
    template<typename U, typename... Args>
    void construct(U* ptr, Args&&... args);
    
    template<typename U>
    void destroy(U* ptr);
    
    size_type max_size() const noexcept;
    
    template<typename U>
    bool operator==(const STLAllocator<U>& other) const noexcept;
    
    template<typename U>
    bool operator!=(const STLAllocator<U>& other) const noexcept;

private:
    UnifiedMemoryManager* manager_;
    static UnifiedMemoryManager* getDefaultManager();
};

/**
 * @brief SIMD对齐分配器
 */
template<typename T, size_t Alignment = 64>
class SIMDAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    static constexpr size_t alignment = Alignment;
    
    template<typename U>
    struct rebind {
        using other = SIMDAllocator<U, Alignment>;
    };
    
    SIMDAllocator() noexcept = default;
    explicit SIMDAllocator(UnifiedMemoryManager& manager) noexcept;
    
    template<typename U>
    SIMDAllocator(const SIMDAllocator<U, Alignment>& other) noexcept;
    
    ~SIMDAllocator() = default;
    
    pointer allocate(size_type count);
    void deallocate(pointer ptr, size_type count) noexcept;
    
    size_type max_size() const noexcept;
    
    template<typename U>
    bool operator==(const SIMDAllocator<U, Alignment>& other) const noexcept;
    
    template<typename U>
    bool operator!=(const SIMDAllocator<U, Alignment>& other) const noexcept;

private:
    UnifiedMemoryManager* manager_;
};

/**
 * @brief 池特化分配器
 */
template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = PoolAllocator<U>;
    };
    
    explicit PoolAllocator(MemoryPoolType poolType = MemoryPoolType::GENERAL_PURPOSE) noexcept;
    explicit PoolAllocator(UnifiedMemoryManager& manager, MemoryPoolType poolType = MemoryPoolType::GENERAL_PURPOSE) noexcept;
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) noexcept;
    
    ~PoolAllocator() = default;
    
    pointer allocate(size_type count);
    void deallocate(pointer ptr, size_type count) noexcept;
    
    size_type max_size() const noexcept;
    
    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const noexcept;
    
    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const noexcept;

private:
    UnifiedMemoryManager* manager_;
    MemoryPoolType poolType_;
};

/**
 * @brief 分配器工厂
 */
class AllocatorFactory {
public:
    template<typename T>
    static STLAllocator<T> createSTLAllocator();
    
    template<typename T>
    static STLAllocator<T> createSTLAllocator(UnifiedMemoryManager& manager);
    
    template<typename T, size_t Alignment = 64>
    static SIMDAllocator<T, Alignment> createSIMDAllocator();
    
    template<typename T, size_t Alignment = 64>
    static SIMDAllocator<T, Alignment> createSIMDAllocator(UnifiedMemoryManager& manager);
    
    template<typename T>
    static PoolAllocator<T> createPoolAllocator(MemoryPoolType poolType = MemoryPoolType::GENERAL_PURPOSE);
    
    template<typename T>
    static PoolAllocator<T> createPoolAllocator(UnifiedMemoryManager& manager, MemoryPoolType poolType = MemoryPoolType::GENERAL_PURPOSE);
};

// === 便利类型别名 ===

template<typename T>
using vector = std::vector<T, STLAllocator<T>>;

template<typename T>
using simd_vector = std::vector<T, SIMDAllocator<T>>;

template<typename Key, typename Value>
using map = std::map<Key, Value, std::less<Key>, STLAllocator<std::pair<const Key, Value>>>;

template<typename Key, typename Value>
using unordered_map = std::unordered_map<Key, Value, std::hash<Key>, std::equal_to<Key>, STLAllocator<std::pair<const Key, Value>>>;

template<typename T>
using list = std::list<T, STLAllocator<T>>;

template<typename T>
using deque = std::deque<T, STLAllocator<T>>;

} // namespace oscean::common_utils::memory 