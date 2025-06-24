/**
 * @file cuda_memory_manager.cpp
 * @brief CUDA内存管理器实现
 */

#include "common_utils/gpu/multi_gpu_memory_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include <cuda_runtime.h>
#include <map>
#include <queue>
#include <mutex>

namespace oscean::common_utils::gpu::cuda {

/**
 * @brief CUDA内存块信息
 */
struct CUDAMemoryBlock {
    void* devicePtr = nullptr;
    size_t size = 0;
    int deviceId = -1;
    bool inUse = false;
    cudaStream_t stream = nullptr;
};

/**
 * @brief CUDA内存池实现
 */
class CUDAMemoryPool {
private:
    int m_deviceId;
    size_t m_totalAllocated = 0;
    size_t m_totalUsed = 0;
    
    // 按大小组织的空闲块队列
    std::map<size_t, std::queue<CUDAMemoryBlock*>> m_freeBlocks;
    
    // 所有分配的块
    std::vector<std::unique_ptr<CUDAMemoryBlock>> m_allBlocks;
    
    // 最小分配大小（避免碎片）
    static constexpr size_t MIN_BLOCK_SIZE = 1024 * 1024;  // 1MB
    
    // 对齐要求
    static constexpr size_t ALIGNMENT = 256;
    
    mutable std::mutex m_mutex;
    
public:
    explicit CUDAMemoryPool(int deviceId) : m_deviceId(deviceId) {
        cudaSetDevice(deviceId);
    }
    
    ~CUDAMemoryPool() {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 释放所有CUDA内存
        cudaSetDevice(m_deviceId);
        for (auto& block : m_allBlocks) {
            if (block->devicePtr) {
                cudaFree(block->devicePtr);
            }
        }
    }
    
    /**
     * @brief 分配内存
     */
    CUDAMemoryBlock* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 对齐大小
        size = alignSize(size);
        
        // 查找合适的空闲块
        CUDAMemoryBlock* block = findFreeBlock(size);
        
        if (!block) {
            // 分配新块
            block = allocateNewBlock(size);
        }
        
        if (block) {
            block->inUse = true;
            m_totalUsed += block->size;
        }
        
        return block;
    }
    
    /**
     * @brief 释放内存
     */
    void deallocate(CUDAMemoryBlock* block) {
        if (!block) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        
        block->inUse = false;
        m_totalUsed -= block->size;
        
        // 返回到空闲队列
        m_freeBlocks[block->size].push(block);
    }
    
    /**
     * @brief 获取内存使用统计
     */
    void getMemoryStats(size_t& totalAllocated, size_t& totalUsed) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        totalAllocated = m_totalAllocated;
        totalUsed = m_totalUsed;
    }
    
    /**
     * @brief 碎片整理
     */
    void defragment() {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 简单策略：释放所有未使用的块
        std::vector<std::unique_ptr<CUDAMemoryBlock>> newBlocks;
        m_freeBlocks.clear();
        
        for (auto& block : m_allBlocks) {
            if (block->inUse) {
                newBlocks.push_back(std::move(block));
            } else {
                if (block->devicePtr) {
                    cudaSetDevice(m_deviceId);
                    cudaFree(block->devicePtr);
                    m_totalAllocated -= block->size;
                }
            }
        }
        
        m_allBlocks = std::move(newBlocks);
    }
    
private:
    /**
     * @brief 对齐内存大小
     */
    size_t alignSize(size_t size) const {
        return ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }
    
    /**
     * @brief 查找空闲块
     */
    CUDAMemoryBlock* findFreeBlock(size_t size) {
        // 查找大小完全匹配的块
        auto it = m_freeBlocks.find(size);
        if (it != m_freeBlocks.end() && !it->second.empty()) {
            CUDAMemoryBlock* block = it->second.front();
            it->second.pop();
            return block;
        }
        
        // 查找更大的块（简单的首次适配）
        for (auto& [blockSize, queue] : m_freeBlocks) {
            if (blockSize >= size && !queue.empty()) {
                CUDAMemoryBlock* block = queue.front();
                queue.pop();
                // 注意：这里浪费了一些内存，实际应用中可以分割块
                return block;
            }
        }
        
        return nullptr;
    }
    
    /**
     * @brief 分配新的内存块
     */
    CUDAMemoryBlock* allocateNewBlock(size_t size) {
        // 确保最小分配大小
        size = std::max(size, MIN_BLOCK_SIZE);
        
        cudaSetDevice(m_deviceId);
        
        void* devicePtr = nullptr;
        cudaError_t err = cudaMalloc(&devicePtr, size);
        
        if (err != cudaSuccess) {
            OSCEAN_LOG_ERROR("CUDA内存分配失败: {}", cudaGetErrorString(err));
            return nullptr;
        }
        
        auto block = std::make_unique<CUDAMemoryBlock>();
        block->devicePtr = devicePtr;
        block->size = size;
        block->deviceId = m_deviceId;
        block->inUse = false;
        
        CUDAMemoryBlock* blockPtr = block.get();
        m_allBlocks.push_back(std::move(block));
        m_totalAllocated += size;
        
        return blockPtr;
    }
};

/**
 * @brief CUDA内存管理器
 */
class CUDAMemoryManager {
private:
    std::map<int, std::unique_ptr<CUDAMemoryPool>> m_devicePools;
    mutable std::mutex m_mutex;
    
public:
    static CUDAMemoryManager& getInstance() {
        static CUDAMemoryManager instance;
        return instance;
    }
    
    /**
     * @brief 在指定设备上分配内存
     */
    void* allocateOnDevice(int deviceId, size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 获取或创建设备内存池
        if (m_devicePools.find(deviceId) == m_devicePools.end()) {
            m_devicePools[deviceId] = std::make_unique<CUDAMemoryPool>(deviceId);
        }
        
        CUDAMemoryBlock* block = m_devicePools[deviceId]->allocate(size);
        return block ? block->devicePtr : nullptr;
    }
    
    /**
     * @brief 释放设备内存
     */
    void deallocateOnDevice(int deviceId, void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        
        auto it = m_devicePools.find(deviceId);
        if (it != m_devicePools.end()) {
            // 简化实现：需要跟踪ptr到block的映射
            // 实际实现中应该维护一个ptr到block的映射表
        }
    }
    
    /**
     * @brief 跨设备复制数据
     */
    cudaError_t copyBetweenDevices(
        void* dst, int dstDevice,
        const void* src, int srcDevice,
        size_t size,
        cudaStream_t stream = nullptr) {
        
        // 启用对等访问（如果可能）
        int canAccess = 0;
        cudaDeviceCanAccessPeer(&canAccess, dstDevice, srcDevice);
        
        if (canAccess) {
            cudaDeviceEnablePeerAccess(srcDevice, 0);
            cudaDeviceEnablePeerAccess(dstDevice, 0);
        }
        
        // 执行复制
        cudaError_t err;
        if (stream) {
            err = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, size, stream);
        } else {
            err = cudaMemcpyPeer(dst, dstDevice, src, srcDevice, size);
        }
        
        return err;
    }
    
    /**
     * @brief 获取设备内存统计
     */
    void getDeviceMemoryInfo(int deviceId, size_t& free, size_t& total) {
        cudaSetDevice(deviceId);
        cudaMemGetInfo(&free, &total);
    }
    
    /**
     * @brief 执行所有设备的碎片整理
     */
    void defragmentAll() {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        for (auto& [deviceId, pool] : m_devicePools) {
            pool->defragment();
        }
    }
    
private:
    CUDAMemoryManager() = default;
    ~CUDAMemoryManager() = default;
    CUDAMemoryManager(const CUDAMemoryManager&) = delete;
    CUDAMemoryManager& operator=(const CUDAMemoryManager&) = delete;
};

// 导出的C接口函数
extern "C" {

void* cudaAllocateMemory(int deviceId, size_t size) {
    return CUDAMemoryManager::getInstance().allocateOnDevice(deviceId, size);
}

void cudaDeallocateMemory(int deviceId, void* ptr) {
    CUDAMemoryManager::getInstance().deallocateOnDevice(deviceId, ptr);
}

cudaError_t cudaCopyBetweenDevices(
    void* dst, int dstDevice,
    const void* src, int srcDevice,
    size_t size,
    cudaStream_t stream) {
    return CUDAMemoryManager::getInstance().copyBetweenDevices(
        dst, dstDevice, src, srcDevice, size, stream);
}

void cudaGetDeviceMemoryInfo(int deviceId, size_t* free, size_t* total) {
    CUDAMemoryManager::getInstance().getDeviceMemoryInfo(deviceId, *free, *total);
}

void cudaDefragmentMemory() {
    CUDAMemoryManager::getInstance().defragmentAll();
}

} // extern "C"

} // namespace oscean::common_utils::gpu::cuda 