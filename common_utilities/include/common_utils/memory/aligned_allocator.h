/**
 * @file aligned_allocator.h
 * @brief 内存对齐分配器 - 为SIMD优化和缓存友好访问提供支持
 * @author OSCEAN Team
 * @date 2024
 */

#pragma once

#include <memory>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace oscean {
namespace common_utils {
namespace memory {

/**
 * @brief 内存对齐常量定义
 */
namespace alignment {
    constexpr size_t CACHE_LINE_SIZE = 64;      ///< CPU缓存行大小 (64字节)
    constexpr size_t SIMD_AVX_ALIGNMENT = 32;   ///< AVX指令集要求的32字节对齐
    constexpr size_t SIMD_SSE_ALIGNMENT = 16;   ///< SSE指令集要求的16字节对齐
    constexpr size_t DEFAULT_ALIGNMENT = 32;    ///< 默认对齐大小 (适合大多数SIMD操作)
}

/**
 * @brief 对齐内存分配器模板类
 * @tparam T 元素类型
 * @tparam Alignment 对齐字节数，必须是2的幂
 */
template<typename T, size_t Alignment = alignment::DEFAULT_ALIGNMENT>
class AlignedAllocator {
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
        using other = AlignedAllocator<U, Alignment>;
    };

    /**
     * @brief 默认构造函数
     */
    AlignedAllocator() noexcept = default;

    /**
     * @brief 复制构造函数模板
     */
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    /**
     * @brief 分配对齐内存
     * @param n 要分配的元素数量
     * @return 对齐的内存指针
     */
    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        
        // 检查对齐是否是2的幂
        static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of 2");
        
        size_type bytes = n * sizeof(T);
        
        // 确保分配的字节数至少是对齐要求的倍数
        size_type aligned_bytes = (bytes + Alignment - 1) & ~(Alignment - 1);
        
        void* ptr = nullptr;
        
#ifdef _WIN32
        // Windows下使用_aligned_malloc
        ptr = _aligned_malloc(aligned_bytes, Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        // Unix/Linux下使用posix_memalign
        int result = posix_memalign(&ptr, Alignment, aligned_bytes);
        if (result != 0 || !ptr) {
            throw std::bad_alloc();
        }
#endif

        return static_cast<pointer>(ptr);
    }

    /**
     * @brief 释放对齐内存
     * @param ptr 要释放的内存指针
     * @param n 元素数量（未使用，但STL要求）
     */
    void deallocate(pointer ptr, size_type n) noexcept {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
    }

    /**
     * @brief 构造对象
     */
    template<typename U, typename... Args>
    void construct(U* ptr, Args&&... args) {
        new(ptr) U(std::forward<Args>(args)...);
    }

    /**
     * @brief 析构对象
     */
    template<typename U>
    void destroy(U* ptr) {
        ptr->~U();
    }

    /**
     * @brief 获取最大可分配元素数量
     */
    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    /**
     * @brief 比较操作符
     */
    template<typename U, size_t OtherAlignment>
    bool operator==(const AlignedAllocator<U, OtherAlignment>&) const noexcept {
        return Alignment == OtherAlignment;
    }

    template<typename U, size_t OtherAlignment>
    bool operator!=(const AlignedAllocator<U, OtherAlignment>&) const noexcept {
        return !(*this == AlignedAllocator<U, OtherAlignment>());
    }
};

/**
 * @brief 便捷类型别名
 */
template<typename T>
using CacheAlignedAllocator = AlignedAllocator<T, alignment::CACHE_LINE_SIZE>;

template<typename T>
using SIMDAlignedAllocator = AlignedAllocator<T, alignment::SIMD_AVX_ALIGNMENT>;

template<typename T>
using SSEAlignedAllocator = AlignedAllocator<T, alignment::SIMD_SSE_ALIGNMENT>;

/**
 * @brief 对齐向量类型别名
 */
template<typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

template<typename T>
using cache_aligned_vector = std::vector<T, CacheAlignedAllocator<T>>;

template<typename T>
using simd_aligned_vector = std::vector<T, SIMDAlignedAllocator<T>>;

/**
 * @brief 检查指针是否按指定对齐
 * @param ptr 要检查的指针
 * @param alignment 对齐要求
 * @return 如果对齐则返回true
 */
template<typename T>
constexpr bool is_aligned(const T* ptr, size_t alignment) noexcept {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

/**
 * @brief 检查指针是否按缓存行对齐
 */
template<typename T>
constexpr bool is_cache_aligned(const T* ptr) noexcept {
    return is_aligned(ptr, alignment::CACHE_LINE_SIZE);
}

/**
 * @brief 检查指针是否按SIMD对齐
 */
template<typename T>
constexpr bool is_simd_aligned(const T* ptr) noexcept {
    return is_aligned(ptr, alignment::SIMD_AVX_ALIGNMENT);
}

/**
 * @brief 分配对齐的原始内存块
 * @param size 字节数
 * @param alignment 对齐要求
 * @return 对齐的内存指针
 */
inline void* aligned_alloc(size_t size, size_t alignment) {
    // 检查对齐是否是2的幂
    if ((alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    
    // 确保分配的字节数至少是对齐要求的倍数
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(aligned_size, alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }
#else
    int result = posix_memalign(&ptr, alignment, aligned_size);
    if (result != 0 || !ptr) {
        throw std::bad_alloc();
    }
#endif

    return ptr;
}

/**
 * @brief 释放对齐分配的内存
 * @param ptr 要释放的内存指针
 */
inline void aligned_free(void* ptr) noexcept {
    if (ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

} // namespace memory
} // namespace common_utils
} // namespace oscean 