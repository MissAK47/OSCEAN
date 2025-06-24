/**
 * @file multi_gpu_memory_manager.cpp
 * @brief 多GPU统一内存管理器实现
 */

#include "common_utils/gpu/multi_gpu_memory_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/thread.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <memory>

#ifdef OSCEAN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace oscean::common_utils::gpu {

// 全局内存管理器实例
std::unique_ptr<MultiGPUMemoryManager> GlobalMemoryManager::s_instance;
boost::mutex GlobalMemoryManager::s_mutex;

/**
 * @brief 内存池实现
 */
class MemoryPool {
public:
    MemoryPool(int deviceId, size_t initialSize, size_t blockSize)
        : m_deviceId(deviceId), m_blockSize(blockSize), m_totalSize(0), 
          m_usedSize(0), m_fragmentedSize(0) {
        
        // 不再需要预留容量，因为使用指针
        // m_blocks.reserve(1000);
        
        // 初始化内存池
        if (initialSize > 0) {
            grow(initialSize);
        }
    }
    
    ~MemoryPool() {
        // 释放所有内存块
        for (auto& blockPtr : m_blocks) {
            if (blockPtr && blockPtr->ptr) {
                freeDeviceMemory(blockPtr->ptr, m_deviceId);
            }
        }
    }
    
    MemoryBlock* allocate(size_t size, size_t alignment) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        // 对齐大小
        size_t alignedSize = alignUp(size, alignment);
        
        // 查找合适的空闲块
        MemoryBlock* bestBlock = nullptr;
        size_t bestWaste = std::numeric_limits<size_t>::max();
        
        for (auto& blockPtr : m_blocks) {
            if (blockPtr->isFree && blockPtr->size >= alignedSize) {
                size_t waste = blockPtr->size - alignedSize;
                if (waste < bestWaste) {
                    bestWaste = waste;
                    bestBlock = blockPtr.get();
                }
            }
        }
        
        if (bestBlock) {
            // 分配内存块
            bestBlock->isFree = false;
            bestBlock->usedSize = alignedSize;
            bestBlock->allocTime = boost::chrono::steady_clock::now();
            bestBlock->allocationId = generateAllocationId();
            
            m_usedSize += alignedSize;
            
            // 如果剩余空间足够大，分割块
            if (bestBlock->size - alignedSize >= m_blockSize) {
                splitBlock(bestBlock, alignedSize);
            }
            
            return bestBlock;
        }
        
        return nullptr;
    }
    
    bool deallocate(MemoryBlock* block) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        if (!block || block->isFree) {
            return false;
        }
        
        // 标记为空闲
        block->isFree = true;
        m_usedSize -= block->usedSize;
        block->usedSize = 0;
        block->allocationId.clear();
        
        // 尝试合并相邻的空闲块
        mergeAdjacentBlocks();
        
        return true;
    }
    
    bool deallocateByAllocationId(const std::string& allocationId) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        // 查找具有指定allocationId的块
        for (auto& blockPtr : m_blocks) {
            if (blockPtr->allocationId == allocationId && !blockPtr->isFree) {
                // 标记为空闲
                blockPtr->isFree = true;
                m_usedSize -= blockPtr->usedSize;
                blockPtr->usedSize = 0;
                blockPtr->allocationId.clear();
                
                // 尝试合并相邻的空闲块
                mergeAdjacentBlocks();
                
                return true;
            }
        }
        
        return false;
    }
    
    bool grow(size_t additionalSize) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        // 分配新的设备内存
        void* newPtr = allocateDeviceMemory(additionalSize, m_deviceId);
        if (!newPtr) {
            return false;
        }
        
        // 创建新的内存块
        auto newBlock = std::make_unique<MemoryBlock>();
        newBlock->ptr = newPtr;
        newBlock->size = additionalSize;
        newBlock->deviceId = m_deviceId;
        newBlock->isFree = true;
        newBlock->usedSize = 0;
        newBlock->allocationId = "";  // 显式初始化为空字符串
        newBlock->allocTime = boost::chrono::steady_clock::now();
        
        m_blocks.push_back(std::move(newBlock));
        m_totalSize += additionalSize;
        
        // 合并相邻块
        mergeAdjacentBlocks();
        
        return true;
    }
    
    size_t shrink() {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        size_t freedSize = 0;
        
        // 查找并释放完全空闲的大块
        auto it = m_blocks.begin();
        while (it != m_blocks.end()) {
            if (it->get()->isFree && it->get()->size >= m_blockSize * 4) {
                freeDeviceMemory(it->get()->ptr, m_deviceId);
                freedSize += it->get()->size;
                m_totalSize -= it->get()->size;
                it = m_blocks.erase(it);
            } else {
                ++it;
            }
        }
        
        return freedSize;
    }
    
    size_t defragment() {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        // 计算碎片大小
        calculateFragmentation();
        
        if (m_fragmentedSize < m_blockSize * 10) {
            return 0; // 碎片太少，不值得整理
        }
        
        // 实现内存碎片整理
        size_t totalReclaimed = 0;
        
        // 1. 收集所有已分配的块
        std::vector<MemoryBlock*> allocatedBlocks;
        for (auto& blockPtr : m_blocks) {
            if (!blockPtr->isFree) {
                allocatedBlocks.push_back(blockPtr.get());
            }
        }
        
        // 2. 按地址排序已分配的块
        std::sort(allocatedBlocks.begin(), allocatedBlocks.end(),
                 [](const MemoryBlock* a, const MemoryBlock* b) {
                     return a->ptr < b->ptr;
                 });
        
        // 3. 计算紧凑布局
        void* currentPtr = m_blocks.front()->ptr;
        std::vector<std::pair<MemoryBlock*, void*>> relocations;
        
        for (auto* block : allocatedBlocks) {
            if (block->ptr != currentPtr) {
                // 需要移动这个块
                relocations.push_back({block, currentPtr});
            }
            currentPtr = static_cast<char*>(currentPtr) + block->size;
        }
        
        // 4. 执行内存移动（从后向前，避免覆盖）
        for (auto it = relocations.rbegin(); it != relocations.rend(); ++it) {
            auto* block = it->first;
            void* newPtr = it->second;
            
            // 移动数据
            std::memmove(newPtr, block->ptr, block->usedSize);
            
            // 更新块信息
            block->ptr = newPtr;
        }
        
        // 5. 重新组织块列表
        std::vector<std::unique_ptr<MemoryBlock>> newBlocks;
        
        // 添加所有已分配的块（已经紧凑排列）
        for (auto* block : allocatedBlocks) {
            auto it = std::find_if(m_blocks.begin(), m_blocks.end(),
                                  [block](const std::unique_ptr<MemoryBlock>& b) {
                                      return b.get() == block;
                                  });
            if (it != m_blocks.end()) {
                newBlocks.push_back(std::move(*it));
            }
        }
        
        // 创建一个大的空闲块
        if (!allocatedBlocks.empty()) {
            auto* lastBlock = allocatedBlocks.back();
            void* freeStart = static_cast<char*>(lastBlock->ptr) + lastBlock->size;
            void* poolEnd = static_cast<char*>(m_blocks.front()->ptr) + m_totalSize;
            size_t freeSize = static_cast<char*>(poolEnd) - static_cast<char*>(freeStart);
            
            if (freeSize > 0) {
                auto freeBlock = std::make_unique<MemoryBlock>();
                freeBlock->ptr = freeStart;
                freeBlock->size = freeSize;
                freeBlock->deviceId = m_deviceId;
                freeBlock->isFree = true;
                newBlocks.push_back(std::move(freeBlock));
                totalReclaimed = freeSize;
            }
        }
        
        // 替换块列表
        m_blocks = std::move(newBlocks);
        
        // 重新计算碎片
        calculateFragmentation();
        
        return totalReclaimed;
    }
    
    size_t getTotalSize() const { return m_totalSize; }
    size_t getUsedSize() const { return m_usedSize; }
    size_t getFreeSize() const { return m_totalSize - m_usedSize; }
    size_t getFragmentedSize() const { return m_fragmentedSize; }
    
private:
    int m_deviceId;
    size_t m_blockSize;
    size_t m_totalSize;
    size_t m_usedSize;
    size_t m_fragmentedSize;
    std::vector<std::unique_ptr<MemoryBlock>> m_blocks;
    mutable boost::mutex m_mutex;
    
    static size_t alignUp(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    static std::string generateAllocationId() {
        // 使用简单的原子计数器代替UUID，避免boost::uuids::to_string的潜在问题
        static boost::atomic<uint64_t> counter(0);
        uint64_t id = counter.fetch_add(1);
        return "alloc_" + std::to_string(id);
    }
    
    void splitBlock(MemoryBlock* block, size_t usedSize) {
        if (block->size <= usedSize) return;
        
        // 创建新的空闲块
        MemoryBlock newBlock;
        newBlock.ptr = static_cast<char*>(block->ptr) + usedSize;
        newBlock.size = block->size - usedSize;
        newBlock.deviceId = block->deviceId;
        newBlock.isFree = true;
        
        // 调整原块大小
        block->size = usedSize;
        
        // 插入新块
        auto it = std::find_if(m_blocks.begin(), m_blocks.end(),
                              [block](const std::unique_ptr<MemoryBlock>& b) { return b.get() == block; });
        if (it != m_blocks.end()) {
            m_blocks.insert(it + 1, std::make_unique<MemoryBlock>(newBlock));
        }
    }
    
    void mergeAdjacentBlocks() {
        // 按地址排序
        std::sort(m_blocks.begin(), m_blocks.end(),
                 [](const std::unique_ptr<MemoryBlock>& a, const std::unique_ptr<MemoryBlock>& b) {
                     return a->ptr < b->ptr;
                 });
        
        // 合并相邻的空闲块
        for (size_t i = 0; i < m_blocks.size() - 1; ) {
            if (m_blocks[i]->isFree && m_blocks[i + 1]->isFree &&
                static_cast<char*>(m_blocks[i]->ptr) + m_blocks[i]->size == m_blocks[i + 1]->ptr) {
                // 合并块
                m_blocks[i]->size += m_blocks[i + 1]->size;
                m_blocks.erase(m_blocks.begin() + i + 1);
            } else {
                ++i;
            }
        }
    }
    
    void calculateFragmentation() {
        m_fragmentedSize = 0;
        for (const auto& blockPtr : m_blocks) {
            if (blockPtr->isFree && blockPtr->size < m_blockSize) {
                m_fragmentedSize += blockPtr->size;
            }
        }
    }
    
    // 平台特定的内存分配/释放
    void* allocateDeviceMemory(size_t size, int deviceId) {
        void* ptr = nullptr;
        
        // 根据设备的API类型选择合适的分配方法
        auto deviceIt = std::find_if(m_blocks.begin(), m_blocks.end(),
                                   [deviceId](const std::unique_ptr<MemoryBlock>& b) {
                                       return b->deviceId == deviceId;
                                   });
        
        // 获取设备信息（需要访问父类的设备信息）
        // 这里暂时使用主机内存作为后备方案
        
#ifdef OSCEAN_CUDA_ENABLED
        // 尝试CUDA分配
        if (isCUDADevice(deviceId)) {
            cudaError_t err = cudaSetDevice(deviceId);
            if (err == cudaSuccess) {
                err = cudaMalloc(&ptr, size);
                if (err == cudaSuccess) {
                    // 初始化内存
                    cudaMemset(ptr, 0, size);
                    try {
                        OSCEAN_LOG_DEBUG("MultiGPUMemoryManager", 
                                       "Allocated " + std::to_string(size) + 
                                       " bytes on CUDA device " + std::to_string(deviceId));
                    } catch (...) {}
                    return ptr;
                } else {
                    // CUDA 分配失败，记录详细错误
                    try {
                        OSCEAN_LOG_WARN("MultiGPUMemoryManager", 
                                       "CUDA malloc failed for " + std::to_string(size) + 
                                       " bytes on device " + std::to_string(deviceId) + 
                                       ": " + std::string(cudaGetErrorString(err)));
                        
                        // 获取当前 GPU 内存状态
                        size_t free_mem = 0, total_mem = 0;
                        cudaError_t memErr = cudaMemGetInfo(&free_mem, &total_mem);
                        if (memErr == cudaSuccess) {
                            OSCEAN_LOG_WARN("MultiGPUMemoryManager", 
                                           "GPU memory status: " + std::to_string(free_mem / 1024 / 1024) + 
                                           " MB free / " + std::to_string(total_mem / 1024 / 1024) + " MB total");
                        }
                    } catch (...) {}
                }
            } else {
                try {
                    OSCEAN_LOG_WARN("MultiGPUMemoryManager", 
                                   "Failed to set CUDA device " + std::to_string(deviceId) + 
                                   ": " + std::string(cudaGetErrorString(err)));
                } catch (...) {}
            }
        }
#endif
        
#ifdef OSCEAN_OPENCL_ENABLED
        // 尝试OpenCL分配
        if (isOpenCLDevice(deviceId)) {
            // OpenCL分配需要context和command queue
            // 这里简化处理，实际应该维护设备的context
            try {
                OSCEAN_LOG_DEBUG("MultiGPUMemoryManager", 
                               "OpenCL allocation not yet implemented for device " + 
                               std::to_string(deviceId));
            } catch (...) {}
        }
#endif
        
        // 后备方案：使用主机内存
#ifdef _WIN32
        ptr = _aligned_malloc(size, 256);
#else
        ptr = std::aligned_alloc(256, size);
#endif
        
        if (ptr) {
            std::memset(ptr, 0, size);
            try {
                OSCEAN_LOG_DEBUG("MultiGPUMemoryManager", 
                               "Allocated " + std::to_string(size) + 
                               " bytes of host memory for device " + std::to_string(deviceId));
            } catch (...) {}
        }
        
        return ptr;
    }
    
    void freeDeviceMemory(void* ptr, int deviceId) {
        if (!ptr) return;
        
#ifdef OSCEAN_CUDA_ENABLED
        // 尝试CUDA释放
        if (isCUDADevice(deviceId)) {
            cudaError_t err = cudaSetDevice(deviceId);
            if (err == cudaSuccess) {
                err = cudaFree(ptr);
                if (err == cudaSuccess) {
                    try {
                        OSCEAN_LOG_DEBUG("MultiGPUMemoryManager", 
                                       "Freed CUDA memory on device " + 
                                       std::to_string(deviceId));
                    } catch (...) {}
                    return;
                }
            }
        }
#endif
        
#ifdef OSCEAN_OPENCL_ENABLED
        // 尝试OpenCL释放
        if (isOpenCLDevice(deviceId)) {
            // OpenCL释放需要相应的API
            try {
                OSCEAN_LOG_DEBUG("MultiGPUMemoryManager", 
                               "OpenCL deallocation not yet implemented for device " + 
                               std::to_string(deviceId));
            } catch (...) {}
        }
#endif
        
        // 后备方案：释放主机内存
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
        
        try {
            OSCEAN_LOG_DEBUG("MultiGPUMemoryManager", 
                           "Freed host memory for device " + 
                           std::to_string(deviceId));
        } catch (...) {}
    }
    
    // 辅助函数：检查是否是CUDA设备
    bool isCUDADevice(int deviceId) const {
        // 简化实现，假设deviceId >= 0的都是潜在的CUDA设备
        // 实际实现应该查询设备信息
        return deviceId >= 0;
    }
    
    // 辅助函数：检查是否是OpenCL设备
    bool isOpenCLDevice(int deviceId) const {
        // 简化实现，暂时返回false
        // 实际实现应该查询设备信息
        return false;
    }
};

/**
 * @brief 传输任务
 */
struct TransferTask {
    TransferRequest request;
    boost::chrono::steady_clock::time_point startTime;
    boost::atomic<bool> completed{false};
    boost::atomic<bool> success{false};
};

/**
 * @brief 多GPU内存管理器内部实现
 */
class MultiGPUMemoryManager::Impl {
public:
    Impl(const std::vector<GPUDeviceInfo>& devices,
         const MemoryPoolConfig& poolConfig,
         const MemoryTransferConfig& transferConfig)
        : m_poolConfig(poolConfig), m_transferConfig(transferConfig),
          m_allocationStrategy(AllocationStrategy::BEST_FIT),
          m_isRunning(true) {
        
        // 初始化设备信息
        for (const auto& device : devices) {
            m_devices[device.deviceId] = device;
            
            // 创建内存池（暂时设置初始大小为0以避免日志问题）
            auto pool = std::make_unique<MemoryPool>(
                device.deviceId, 
                0,  // 暂时设置为0，稍后通过preallocatePool来分配
                poolConfig.blockSize
            );
            m_pools[device.deviceId] = std::move(pool);
            
            // 初始化设备统计
            m_usageStats.deviceUsage[device.deviceId] = 0;
            m_usageStats.devicePeakUsage[device.deviceId] = 0;
        }
        
        // 启动传输线程
        if (transferConfig.enableAsyncTransfer) {
            for (int i = 0; i < transferConfig.maxConcurrentTransfers; ++i) {
                m_transferThreads.emplace_back(&Impl::transferWorker, this);
            }
        }
        
        try {
            OSCEAN_LOG_INFO("MultiGPUMemoryManager", "Initialized with " + 
                           std::to_string(devices.size()) + " devices");
        } catch (...) {
            // 忽略日志错误
        }
    }
    
    ~Impl() {
        m_isRunning = false;
        m_transferCondition.notify_all();
        
        for (auto& thread : m_transferThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        // 检查内存泄漏
        if (m_allocations.size() > 0) {
            try {
                OSCEAN_LOG_WARN("MultiGPUMemoryManager", "Detected " + 
                               std::to_string(m_allocations.size()) + " unreleased allocations");
            } catch (...) {
                // 忽略日志错误
            }
            fireEvent(MemoryEventType::MEMORY_LEAK_DETECTED, -1, 0,
                     "Memory leaks detected on shutdown");
        }
    }
    
    GPUMemoryHandle allocate(const AllocationRequest& request) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        GPUMemoryHandle handle;
        
        // 选择设备
        int deviceId = request.preferredDeviceId;
        if (deviceId < 0 || m_devices.find(deviceId) == m_devices.end()) {
            deviceId = selectDeviceForAllocation(request);
        }
        
        if (deviceId < 0) {
            fireEvent(MemoryEventType::ALLOCATION_FAILED, -1, request.size,
                     "No suitable device found");
            return handle;
        }
        
        // 从内存池分配
        auto& pool = m_pools[deviceId];
        MemoryBlock* block = pool->allocate(request.size, request.alignment);
        
        if (!block && m_poolConfig.enableGrowth) {
            // 尝试增长内存池
            size_t growSize = std::max(request.size, 
                static_cast<size_t>(pool->getTotalSize() * m_poolConfig.growthFactor));
            growSize = std::min(growSize, m_poolConfig.maxPoolSize - pool->getTotalSize());
            
            if (growSize > 0 && pool->grow(growSize)) {
                fireEvent(MemoryEventType::POOL_GROWN, deviceId, growSize,
                         "Pool grown by " + std::to_string(growSize) + " bytes");
                block = pool->allocate(request.size, request.alignment);
            }
        }
        
        if (!block) {
            // 尝试其他设备
            if (request.allowFallback) {
                for (auto& [devId, devPool] : m_pools) {
                    if (devId != deviceId) {
                        block = devPool->allocate(request.size, request.alignment);
                        if (block) {
                            deviceId = devId;
                            break;
                        }
                    }
                }
            }
        }
        
        if (block) {
            // 创建内存句柄
            handle.deviceId = deviceId;
            handle.devicePtr = block->ptr;
            handle.size = request.size;
            // 使用块的实际内存类型，而不是请求的类型
            // 检查内存是否真的在 GPU 上
            GPUMemoryType actualType = request.memoryType;
#ifdef OSCEAN_CUDA_ENABLED
            if (deviceId >= 0) {
                cudaPointerAttributes attrs;
                cudaError_t err = cudaPointerGetAttributes(&attrs, block->ptr);
                if (err == cudaSuccess) {
                    if (attrs.type == cudaMemoryTypeDevice) {
                        actualType = GPUMemoryType::DEVICE;
                    } else {
                        actualType = GPUMemoryType::HOST;
                        try {
                            OSCEAN_LOG_WARN("MultiGPUMemoryManager", 
                                          "Allocated host memory for device " + 
                                          std::to_string(deviceId) + 
                                          " (requested GPU memory)");
                        } catch (...) {}
                    }
                } else {
                    // 如果无法查询，假设是主机内存
                    actualType = GPUMemoryType::HOST;
                }
            }
#endif
            handle.memoryType = actualType;
            handle.isPooled = true;
            handle.allocationId = block->allocationId;
            
            // 记录分配
            m_allocations[handle.allocationId] = handle;
            
            // 更新统计
            m_usageStats.totalAllocated += request.size;
            m_usageStats.currentUsage += request.size;
            m_usageStats.allocationCount++;
            m_usageStats.deviceUsage[deviceId] += request.size;
            
            if (m_usageStats.currentUsage > m_usageStats.peakUsage) {
                m_usageStats.peakUsage = m_usageStats.currentUsage;
            }
            if (m_usageStats.deviceUsage[deviceId] > m_usageStats.devicePeakUsage[deviceId]) {
                m_usageStats.devicePeakUsage[deviceId] = m_usageStats.deviceUsage[deviceId];
            }
            
            updateAverageAllocationSize();
            
            fireEvent(MemoryEventType::ALLOCATION_SUCCESS, deviceId, request.size,
                     "Allocated " + std::to_string(request.size) + " bytes");
        } else {
            fireEvent(MemoryEventType::OUT_OF_MEMORY, deviceId, request.size,
                     "Failed to allocate " + std::to_string(request.size) + " bytes");
        }
        
        return handle;
    }
    
    bool deallocate(const GPUMemoryHandle& handle) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        if (!handle.isValid()) {
            return false;
        }
        
        // 查找分配记录
        auto it = m_allocations.find(handle.allocationId);
        if (it == m_allocations.end()) {
            try {
                OSCEAN_LOG_WARN("MultiGPUMemoryManager", "Allocation not found: " + handle.allocationId);
            } catch (...) {
                // 忽略日志错误
            }
            return false;
        }
        
        // 释放到内存池
        if (handle.isPooled) {
            auto poolIt = m_pools.find(handle.deviceId);
            if (poolIt != m_pools.end()) {
                // 通过allocationId释放内存块
                bool freed = poolIt->second->deallocateByAllocationId(handle.allocationId);
                
                if (freed) {
                    // 更新统计
                    m_usageStats.totalFreed += handle.size;
                    m_usageStats.currentUsage -= handle.size;
                    m_usageStats.deallocationCount++;
                    m_usageStats.deviceUsage[handle.deviceId] -= handle.size;
                    
                    // 移除分配记录
                    m_allocations.erase(it);
                    
                    fireEvent(MemoryEventType::DEALLOCATION, handle.deviceId, handle.size,
                             "Deallocated " + std::to_string(handle.size) + " bytes");
                    
                    return true;
                }
            }
        }
        
        return false;
    }
    
    bool transfer(const TransferRequest& request) {
        if (!request.source.isValid() || !request.destination.isValid()) {
            return false;
        }
        
        // 确定传输类型
        TransferType type = determineTransferType(request.source, request.destination);
        
        // 创建传输任务
        auto task = std::make_shared<TransferTask>();
        task->request = request;
        task->startTime = boost::chrono::steady_clock::now();
        
        if (request.async && m_transferConfig.enableAsyncTransfer) {
            // 异步传输
            boost::lock_guard<boost::mutex> lock(m_transferMutex);
            m_transferQueue.push(task);
            m_transferCondition.notify_one();
            
            fireEvent(MemoryEventType::TRANSFER_STARTED, 
                     request.source.deviceId, request.size,
                     "Async transfer queued");
            
            return true;
        } else {
            // 同步传输
            bool success = performTransfer(task);
            
            if (success) {
                fireEvent(MemoryEventType::TRANSFER_COMPLETED,
                         request.source.deviceId, request.size,
                         "Sync transfer completed");
            } else {
                fireEvent(MemoryEventType::TRANSFER_FAILED,
                         request.source.deviceId, request.size,
                         "Sync transfer failed");
            }
            
            return success;
        }
    }
    
    bool synchronize(int timeoutMs) {
        auto startTime = boost::chrono::steady_clock::now();
        
        while (true) {
            {
                boost::lock_guard<boost::mutex> lock(m_transferMutex);
                if (m_transferQueue.empty() && m_activeTransfers == 0) {
                    return true;
                }
            }
            
            if (timeoutMs >= 0) {
                auto elapsed = boost::chrono::duration_cast<boost::chrono::milliseconds>(
                    boost::chrono::steady_clock::now() - startTime).count();
                if (elapsed >= timeoutMs) {
                    return false;
                }
            }
            
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        }
    }
    
    bool preallocatePool(int deviceId, size_t size) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        auto it = m_pools.find(deviceId);
        if (it == m_pools.end()) {
            return false;
        }
        
        return it->second->grow(size);
    }
    
    void shrinkPools(int deviceId) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        if (deviceId >= 0) {
            auto it = m_pools.find(deviceId);
            if (it != m_pools.end()) {
                size_t freed = it->second->shrink();
                if (freed > 0) {
                    fireEvent(MemoryEventType::POOL_SHRUNK, deviceId, freed,
                             "Pool shrunk by " + std::to_string(freed) + " bytes");
                }
            }
        } else {
            // 收缩所有池
            for (auto& [devId, pool] : m_pools) {
                size_t freed = pool->shrink();
                if (freed > 0) {
                    fireEvent(MemoryEventType::POOL_SHRUNK, devId, freed,
                             "Pool shrunk by " + std::to_string(freed) + " bytes");
                }
            }
        }
    }
    
    size_t defragment(int deviceId) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        size_t totalDefragmented = 0;
        
        if (deviceId >= 0) {
            auto it = m_pools.find(deviceId);
            if (it != m_pools.end()) {
                fireEvent(MemoryEventType::DEFRAGMENTATION_STARTED, deviceId, 0,
                         "Defragmentation started");
                
                totalDefragmented = it->second->defragment();
                
                fireEvent(MemoryEventType::DEFRAGMENTATION_COMPLETED, deviceId, 
                         totalDefragmented, "Defragmentation completed");
            }
        } else {
            // 整理所有池
            for (auto& [devId, pool] : m_pools) {
                fireEvent(MemoryEventType::DEFRAGMENTATION_STARTED, devId, 0,
                         "Defragmentation started");
                
                size_t defragmented = pool->defragment();
                totalDefragmented += defragmented;
                
                fireEvent(MemoryEventType::DEFRAGMENTATION_COMPLETED, devId,
                         defragmented, "Defragmentation completed");
            }
        }
        
        return totalDefragmented;
    }
    
    MemoryUsageStats getUsageStats() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        // 更新池化内存统计
        MemoryUsageStats stats = m_usageStats;
        stats.pooledMemory = 0;
        stats.fragmentedMemory = 0;
        
        for (const auto& [_, pool] : m_pools) {
            stats.pooledMemory += pool->getTotalSize();
            stats.fragmentedMemory += pool->getFragmentedSize();
        }
        
        return stats;
    }
    
    TransferStats getTransferStats() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        return m_transferStats;
    }
    
    size_t getAvailableMemory(int deviceId) const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        auto poolIt = m_pools.find(deviceId);
        if (poolIt != m_pools.end()) {
            return poolIt->second->getFreeSize();
        }
        
        auto devIt = m_devices.find(deviceId);
        if (devIt != m_devices.end()) {
            return devIt->second.memoryDetails.freeGlobalMemory;
        }
        
        return 0;
    }
    
    void registerEventCallback(MemoryEventCallback callback) {
        boost::lock_guard<boost::mutex> lock(m_callbackMutex);
        m_eventCallbacks.push_back(callback);
    }
    
    void setAllocationStrategy(AllocationStrategy strategy) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        m_allocationStrategy = strategy;
    }
    
    AllocationStrategy getAllocationStrategy() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        return m_allocationStrategy;
    }
    
    std::string generateMemoryReport() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        std::stringstream ss;
        ss << "=== GPU Memory Report ===\n\n";
        
        // 总体统计
        ss << "Overall Statistics:\n";
        ss << "  Total Allocated: " << formatBytes(m_usageStats.totalAllocated) << "\n";
        ss << "  Total Freed: " << formatBytes(m_usageStats.totalFreed) << "\n";
        ss << "  Current Usage: " << formatBytes(m_usageStats.currentUsage) << "\n";
        ss << "  Peak Usage: " << formatBytes(m_usageStats.peakUsage) << "\n";
        ss << "  Allocations: " << m_usageStats.allocationCount << "\n";
        ss << "  Deallocations: " << m_usageStats.deallocationCount << "\n";
        ss << "  Average Allocation: " << formatBytes(m_usageStats.averageAllocationSize) << "\n\n";
        
        // 设备统计
        ss << "Per-Device Statistics:\n";
        for (const auto& [deviceId, device] : m_devices) {
            ss << "\nDevice " << deviceId << " (" << device.name << "):\n";
            
            auto poolIt = m_pools.find(deviceId);
            if (poolIt != m_pools.end()) {
                ss << "  Pool Size: " << formatBytes(poolIt->second->getTotalSize()) << "\n";
                ss << "  Pool Used: " << formatBytes(poolIt->second->getUsedSize()) << "\n";
                ss << "  Pool Free: " << formatBytes(poolIt->second->getFreeSize()) << "\n";
                ss << "  Fragmented: " << formatBytes(poolIt->second->getFragmentedSize()) << "\n";
            }
            
            ss << "  Current Usage: " << formatBytes(m_usageStats.deviceUsage.at(deviceId)) << "\n";
            ss << "  Peak Usage: " << formatBytes(m_usageStats.devicePeakUsage.at(deviceId)) << "\n";
        }
        
        // 传输统计
        ss << "\nTransfer Statistics:\n";
        ss << "  Total Transfers: " << m_transferStats.totalTransfers << "\n";
        ss << "  Total Bytes: " << formatBytes(m_transferStats.totalBytesTransferred) << "\n";
        ss << "  Average Size: " << formatBytes(m_transferStats.averageTransferSize) << "\n";
        ss << "  Average Time: " << std::fixed << std::setprecision(2) 
           << m_transferStats.averageTransferTime << " ms\n";
        ss << "  Peak Bandwidth: " << std::fixed << std::setprecision(2)
           << m_transferStats.peakBandwidth << " GB/s\n";
        
        return ss.str();
    }
    
    void reset(bool releaseAll) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        if (releaseAll) {
            // 释放所有分配
            m_allocations.clear();
            
            // 重置内存池
            for (auto& [deviceId, pool] : m_pools) {
                pool.reset(new MemoryPool(deviceId, 0, m_poolConfig.blockSize));
            }
        }
        
        // 重置统计
        m_usageStats = MemoryUsageStats();
        m_transferStats = TransferStats();
        
        for (const auto& [deviceId, _] : m_devices) {
            m_usageStats.deviceUsage[deviceId] = 0;
            m_usageStats.devicePeakUsage[deviceId] = 0;
        }
        
        try {
            OSCEAN_LOG_INFO("MultiGPUMemoryManager", "Memory manager reset");
        } catch (...) {
            // 忽略日志错误
        }
    }
    
    // 添加公共方法来访问设备信息
    bool hasDevice(int deviceId) const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        return m_devices.find(deviceId) != m_devices.end();
    }
    
    bool deviceSupportsCUDA(int deviceId) const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        auto it = m_devices.find(deviceId);
        if (it != m_devices.end()) {
            return std::find(it->second.supportedAPIs.begin(),
                           it->second.supportedAPIs.end(),
                           ComputeAPI::CUDA) != it->second.supportedAPIs.end();
        }
        return false;
    }
    
private:
    // 配置
    MemoryPoolConfig m_poolConfig;
    MemoryTransferConfig m_transferConfig;
    AllocationStrategy m_allocationStrategy;
    
    // 设备和内存池
    std::map<int, GPUDeviceInfo> m_devices;
    std::map<int, std::unique_ptr<MemoryPool>> m_pools;
    
    // 分配跟踪
    std::map<std::string, GPUMemoryHandle> m_allocations;
    
    // 统计信息
    MemoryUsageStats m_usageStats;
    TransferStats m_transferStats;
    
    // 传输管理
    std::queue<std::shared_ptr<TransferTask>> m_transferQueue;
    std::vector<boost::thread> m_transferThreads;
    boost::atomic<int> m_activeTransfers{0};
    boost::condition_variable m_transferCondition;
    
    // 同步
    mutable boost::mutex m_mutex;
    mutable boost::mutex m_transferMutex;
    mutable boost::mutex m_callbackMutex;
    boost::atomic<bool> m_isRunning;
    
    // 事件回调
    std::vector<MemoryEventCallback> m_eventCallbacks;
    
    // 辅助函数
    int selectDeviceForAllocation(const AllocationRequest& request) {
        int bestDevice = -1;
        size_t maxAvailable = 0;
        
        for (const auto& [deviceId, pool] : m_pools) {
            size_t available = pool->getFreeSize();
            if (available >= request.size && available > maxAvailable) {
                maxAvailable = available;
                bestDevice = deviceId;
            }
        }
        
        return bestDevice;
    }
    
    TransferType determineTransferType(const GPUMemoryHandle& src, 
                                      const GPUMemoryHandle& dst) {
        if (src.memoryType == GPUMemoryType::HOST) {
            return TransferType::HOST_TO_DEVICE;
        } else if (dst.memoryType == GPUMemoryType::HOST) {
            return TransferType::DEVICE_TO_HOST;
        } else if (src.deviceId == dst.deviceId) {
            return TransferType::DEVICE_TO_DEVICE;
        } else {
            return TransferType::PEER_TO_PEER;
        }
    }
    
    // GPU数据传输实现方法
    bool performHostToDeviceTransfer(const TransferRequest& request) {
        size_t size = request.size ? request.size : request.source.size;
        
#ifdef OSCEAN_CUDA_ENABLED
        if (isCUDADevice(request.destination.deviceId)) {
            cudaError_t err = cudaSetDevice(request.destination.deviceId);
            if (err == cudaSuccess) {
                err = cudaMemcpy(
                    static_cast<char*>(request.destination.devicePtr) + request.offset,
                    static_cast<const char*>(request.source.devicePtr) + request.offset,
                    size, cudaMemcpyHostToDevice);
                if (err == cudaSuccess) {
                    return true;
                }
                // CUDA传输失败，记录错误但继续尝试后备方案
                try {
                    OSCEAN_LOG_WARN("MultiGPUMemoryManager", 
                                   "CUDA memcpy failed: " + std::string(cudaGetErrorString(err)) + 
                                   ", falling back to standard memcpy");
                } catch (...) {}
            }
        }
#endif
        
        // 后备方案：使用memcpy
        std::memcpy(
            static_cast<char*>(request.destination.devicePtr) + request.offset,
            static_cast<const char*>(request.source.devicePtr) + request.offset,
            size);
        return true;
    }
    
    bool performDeviceToHostTransfer(const TransferRequest& request) {
        size_t size = request.size ? request.size : request.source.size;
        
#ifdef OSCEAN_CUDA_ENABLED
        if (isCUDADevice(request.source.deviceId)) {
            cudaError_t err = cudaSetDevice(request.source.deviceId);
            if (err == cudaSuccess) {
                err = cudaMemcpy(
                    static_cast<char*>(request.destination.devicePtr) + request.offset,
                    static_cast<const char*>(request.source.devicePtr) + request.offset,
                    size, cudaMemcpyDeviceToHost);
                if (err == cudaSuccess) {
                    return true;
                }
                // CUDA传输失败，记录错误但继续尝试后备方案
                try {
                    OSCEAN_LOG_WARN("MultiGPUMemoryManager", 
                                   "CUDA memcpy failed: " + std::string(cudaGetErrorString(err)) + 
                                   ", falling back to standard memcpy");
                } catch (...) {}
            }
        }
#endif
        
        // 后备方案：使用memcpy
        std::memcpy(
            static_cast<char*>(request.destination.devicePtr) + request.offset,
            static_cast<const char*>(request.source.devicePtr) + request.offset,
            size);
        return true;
    }
    
    bool performDeviceToDeviceTransfer(const TransferRequest& request) {
        size_t size = request.size ? request.size : request.source.size;
        
#ifdef OSCEAN_CUDA_ENABLED
        if (isCUDADevice(request.source.deviceId)) {
            cudaError_t err = cudaSetDevice(request.source.deviceId);
            if (err == cudaSuccess) {
                err = cudaMemcpy(
                    static_cast<char*>(request.destination.devicePtr) + request.offset,
                    static_cast<const char*>(request.source.devicePtr) + request.offset,
                    size, cudaMemcpyDeviceToDevice);
                if (err == cudaSuccess) {
                    return true;
                }
                // CUDA传输失败，记录错误但继续尝试后备方案
                try {
                    OSCEAN_LOG_WARN("MultiGPUMemoryManager", 
                                   "CUDA device-to-device memcpy failed: " + std::string(cudaGetErrorString(err)) + 
                                   ", falling back to standard memcpy");
                } catch (...) {}
            }
        }
#endif
        
        // 后备方案：使用memcpy
        std::memcpy(
            static_cast<char*>(request.destination.devicePtr) + request.offset,
            static_cast<const char*>(request.source.devicePtr) + request.offset,
            size);
        return true;
    }
    
    bool performPeerToPeerTransfer(const TransferRequest& request) {
        size_t size = request.size ? request.size : request.source.size;
        
#ifdef OSCEAN_CUDA_ENABLED
        if (isCUDADevice(request.source.deviceId) && isCUDADevice(request.destination.deviceId)) {
            // 检查是否支持点对点访问
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, request.destination.deviceId, request.source.deviceId);
            
            if (canAccess) {
                // 启用点对点访问
                cudaSetDevice(request.destination.deviceId);
                cudaDeviceEnablePeerAccess(request.source.deviceId, 0);
                
                // 执行点对点传输
                cudaError_t err = cudaMemcpyPeer(
                    static_cast<char*>(request.destination.devicePtr) + request.offset,
                    request.destination.deviceId,
                    static_cast<const char*>(request.source.devicePtr) + request.offset,
                    request.source.deviceId,
                    size);
                
                return err == cudaSuccess;
            }
        }
#endif
        
        // 后备方案：通过主机内存中转
        return performFallbackTransfer(request);
    }
    
    bool performFallbackTransfer(const TransferRequest& request) {
        size_t size = request.size ? request.size : request.source.size;
        
        // 分配临时主机内存
        void* tempBuffer = std::malloc(size);
        if (!tempBuffer) {
            return false;
        }
        
        bool success = true;
        
        // 从源设备复制到主机
        TransferRequest toHost;
        toHost.source = request.source;
        toHost.destination.devicePtr = tempBuffer;
        toHost.destination.memoryType = GPUMemoryType::HOST;
        toHost.size = size;
        toHost.offset = 0;
        
        if (request.source.memoryType != GPUMemoryType::HOST) {
            success = performDeviceToHostTransfer(toHost);
        } else {
            std::memcpy(tempBuffer, 
                       static_cast<const char*>(request.source.devicePtr) + request.offset, 
                       size);
        }
        
        // 从主机复制到目标设备
        if (success) {
            TransferRequest fromHost;
            fromHost.source.devicePtr = tempBuffer;
            fromHost.source.memoryType = GPUMemoryType::HOST;
            fromHost.destination = request.destination;
            fromHost.size = size;
            fromHost.offset = 0;
            
            if (request.destination.memoryType != GPUMemoryType::HOST) {
                success = performHostToDeviceTransfer(fromHost);
            } else {
                std::memcpy(static_cast<char*>(request.destination.devicePtr) + request.offset,
                           tempBuffer, size);
            }
        }
        
        std::free(tempBuffer);
        return success;
    }
    
    bool performTransfer(std::shared_ptr<TransferTask> task) {
        m_activeTransfers++;
        
        // 计算传输大小
        size_t transferSize = task->request.size;
        if (transferSize == 0) {
            transferSize = task->request.source.size;
        }
        
        // 确定传输类型
        TransferType type = determineTransferType(task->request.source, 
                                                 task->request.destination);
        
        // 实现真实的GPU数据传输
        bool success = false;
        
        try {
            switch (type) {
            case TransferType::HOST_TO_DEVICE:
                success = performHostToDeviceTransfer(task->request);
                break;
                
            case TransferType::DEVICE_TO_HOST:
                success = performDeviceToHostTransfer(task->request);
                break;
                
            case TransferType::DEVICE_TO_DEVICE:
                success = performDeviceToDeviceTransfer(task->request);
                break;
                
            case TransferType::PEER_TO_PEER:
                success = performPeerToPeerTransfer(task->request);
                break;
                
            default:
                // 后备方案：使用主机内存中转
                success = performFallbackTransfer(task->request);
                break;
            }
        } catch (const std::exception& e) {
            try {
                OSCEAN_LOG_ERROR("MultiGPUMemoryManager", 
                               "Transfer failed: " + std::string(e.what()));
            } catch (...) {}
            success = false;
        }
        
        task->success = success;
        
        // 计算传输时间
        auto endTime = boost::chrono::steady_clock::now();
        auto duration = boost::chrono::duration_cast<boost::chrono::microseconds>(
            endTime - task->startTime).count() / 1000.0; // 转换为毫秒
        
        // 更新统计
        {
            boost::lock_guard<boost::mutex> lock(m_mutex);
            
            m_transferStats.totalTransfers++;
            m_transferStats.totalBytesTransferred += transferSize;
            
            // 更新平均值
            double alpha = 0.1; // 指数移动平均系数
            m_transferStats.averageTransferSize = 
                m_transferStats.averageTransferSize * (1 - alpha) + transferSize * alpha;
            m_transferStats.averageTransferTime = 
                m_transferStats.averageTransferTime * (1 - alpha) + duration * alpha;
            
            // 计算带宽
            if (duration > 0) {
                double bandwidth = (transferSize / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000.0);
                if (bandwidth > m_transferStats.peakBandwidth) {
                    m_transferStats.peakBandwidth = bandwidth;
                }
            }
            
            // 按类型统计
            TransferType type = determineTransferType(task->request.source, 
                                                     task->request.destination);
            m_transferStats.transfersByType[type]++;
            m_transferStats.bytesByType[type] += transferSize;
        }
        
        task->completed = true;
        m_activeTransfers--;
        
        // 调用回调
        if (task->request.callback) {
            task->request.callback(task->success);
        }
        
        return task->success.load();
    }
    
    void transferWorker() {
        while (m_isRunning) {
            std::shared_ptr<TransferTask> task;
            
            {
                boost::unique_lock<boost::mutex> lock(m_transferMutex);
                m_transferCondition.wait(lock, [this] {
                    return !m_transferQueue.empty() || !m_isRunning;
                });
                
                if (!m_isRunning) break;
                
                if (!m_transferQueue.empty()) {
                    task = m_transferQueue.front();
                    m_transferQueue.pop();
                }
            }
            
            if (task) {
                bool success = performTransfer(task);
                
                if (success) {
                    fireEvent(MemoryEventType::TRANSFER_COMPLETED,
                             task->request.source.deviceId, task->request.size,
                             "Transfer completed");
                } else {
                    fireEvent(MemoryEventType::TRANSFER_FAILED,
                             task->request.source.deviceId, task->request.size,
                             "Transfer failed");
                }
            }
        }
    }
    
    void updateAverageAllocationSize() {
        if (m_usageStats.allocationCount > 0) {
            m_usageStats.averageAllocationSize = 
                static_cast<double>(m_usageStats.totalAllocated) / 
                m_usageStats.allocationCount;
        }
    }
    
    void fireEvent(MemoryEventType type, int deviceId, size_t size,
                  const std::string& message) {
        try {
            MemoryEvent event;
            event.type = type;
            event.deviceId = deviceId;
            event.size = size;
            event.message = message;
            event.timestamp = boost::chrono::steady_clock::now();
            
            boost::lock_guard<boost::mutex> lock(m_callbackMutex);
            for (const auto& callback : m_eventCallbacks) {
                try {
                    callback(event);
                } catch (...) {
                    // 忽略回调中的异常
                }
            }
        } catch (...) {
            // 忽略事件触发中的异常，避免影响内存管理操作
        }
    }
    
    static std::string formatBytes(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unitIndex = 0;
        double size = static_cast<double>(bytes);
        
        while (size >= 1024.0 && unitIndex < 4) {
            size /= 1024.0;
            unitIndex++;
        }
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
        return ss.str();
    }
    
    // 辅助函数：检查是否是CUDA设备
    bool isCUDADevice(int deviceId) const {
        auto it = m_devices.find(deviceId);
        if (it != m_devices.end()) {
            // 检查设备是否支持CUDA
            return std::find(it->second.supportedAPIs.begin(), 
                           it->second.supportedAPIs.end(),
                           ComputeAPI::CUDA) != it->second.supportedAPIs.end();
        }
        return false;
    }
};

// MultiGPUMemoryManager实现
MultiGPUMemoryManager::MultiGPUMemoryManager(const std::vector<GPUDeviceInfo>& devices,
                                           const MemoryPoolConfig& poolConfig,
                                           const MemoryTransferConfig& transferConfig)
    : m_impl(std::make_unique<Impl>(devices, poolConfig, transferConfig)) {
}

MultiGPUMemoryManager::~MultiGPUMemoryManager() = default;

GPUMemoryHandle MultiGPUMemoryManager::allocate(const AllocationRequest& request) {
    return m_impl->allocate(request);
}

bool MultiGPUMemoryManager::deallocate(const GPUMemoryHandle& handle) {
    return m_impl->deallocate(handle);
}

bool MultiGPUMemoryManager::transfer(const TransferRequest& request) {
    return m_impl->transfer(request);
}

bool MultiGPUMemoryManager::synchronize(int timeoutMs) {
    return m_impl->synchronize(timeoutMs);
}

bool MultiGPUMemoryManager::preallocatePool(int deviceId, size_t size) {
    return m_impl->preallocatePool(deviceId, size);
}

void MultiGPUMemoryManager::shrinkPools(int deviceId) {
    m_impl->shrinkPools(deviceId);
}

size_t MultiGPUMemoryManager::defragment(int deviceId) {
    return m_impl->defragment(deviceId);
}

MemoryUsageStats MultiGPUMemoryManager::getUsageStats() const {
    return m_impl->getUsageStats();
}

TransferStats MultiGPUMemoryManager::getTransferStats() const {
    return m_impl->getTransferStats();
}

size_t MultiGPUMemoryManager::getAvailableMemory(int deviceId) const {
    return m_impl->getAvailableMemory(deviceId);
}

bool MultiGPUMemoryManager::canAccessPeer(int srcDevice, int dstDevice) const {
#ifdef OSCEAN_CUDA_ENABLED
    // 检查两个设备是否都支持CUDA
    if (m_impl->deviceSupportsCUDA(srcDevice) && m_impl->deviceSupportsCUDA(dstDevice)) {
        int canAccess = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(&canAccess, dstDevice, srcDevice);
        return err == cudaSuccess && canAccess != 0;
    }
#endif
    return false;
}

bool MultiGPUMemoryManager::enablePeerAccess(int srcDevice, int dstDevice) {
#ifdef OSCEAN_CUDA_ENABLED
    if (canAccessPeer(srcDevice, dstDevice)) {
        cudaError_t err = cudaSetDevice(dstDevice);
        if (err == cudaSuccess) {
            err = cudaDeviceEnablePeerAccess(srcDevice, 0);
            if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                try {
                    OSCEAN_LOG_INFO("MultiGPUMemoryManager",
                                   "Enabled peer access from device " + std::to_string(srcDevice) +
                                   " to device " + std::to_string(dstDevice));
                } catch (...) {}
                return true;
            }
        }
    }
#endif
    return false;
}

void MultiGPUMemoryManager::disablePeerAccess(int srcDevice, int dstDevice) {
#ifdef OSCEAN_CUDA_ENABLED
    cudaError_t err = cudaSetDevice(dstDevice);
    if (err == cudaSuccess) {
        err = cudaDeviceDisablePeerAccess(srcDevice);
        if (err == cudaSuccess || err == cudaErrorPeerAccessNotEnabled) {
            try {
                OSCEAN_LOG_INFO("MultiGPUMemoryManager",
                               "Disabled peer access from device " + std::to_string(srcDevice) +
                               " to device " + std::to_string(dstDevice));
            } catch (...) {}
        }
    }
#endif
}

void MultiGPUMemoryManager::registerEventCallback(MemoryEventCallback callback) {
    m_impl->registerEventCallback(callback);
}

void MultiGPUMemoryManager::setAllocationStrategy(AllocationStrategy strategy) {
    m_impl->setAllocationStrategy(strategy);
}

AllocationStrategy MultiGPUMemoryManager::getAllocationStrategy() const {
    return m_impl->getAllocationStrategy();
}

std::vector<std::string> MultiGPUMemoryManager::validateMemoryIntegrity() const {
    std::vector<std::string> issues;
    
    // 检查内存泄漏
    auto stats = m_impl->getUsageStats();
    if (stats.allocationCount != stats.deallocationCount) {
        issues.push_back("Memory leak detected: " + 
                        std::to_string(stats.allocationCount - stats.deallocationCount) + 
                        " unfreed allocations");
    }
    
    // 检查内存使用不一致
    if (stats.totalAllocated < stats.totalFreed) {
        issues.push_back("Memory accounting error: freed more than allocated");
    }
    
    // 检查当前使用量
    size_t expectedUsage = stats.totalAllocated - stats.totalFreed;
    if (stats.currentUsage != expectedUsage) {
        issues.push_back("Memory usage mismatch: expected " + 
                        std::to_string(expectedUsage) + 
                        " bytes, but tracking " + 
                        std::to_string(stats.currentUsage) + " bytes");
    }
    
    // 检查设备内存使用
    size_t totalDeviceUsage = 0;
    for (const auto& [deviceId, usage] : stats.deviceUsage) {
        totalDeviceUsage += usage;
        
        // 检查设备使用量是否超过峰值
        auto peakIt = stats.devicePeakUsage.find(deviceId);
        if (peakIt != stats.devicePeakUsage.end() && usage > peakIt->second) {
            issues.push_back("Device " + std::to_string(deviceId) + 
                           " current usage exceeds recorded peak");
        }
    }
    
    // 检查总设备使用量
    if (totalDeviceUsage != stats.currentUsage) {
        issues.push_back("Device usage sum mismatch: " + 
                        std::to_string(totalDeviceUsage) + 
                        " vs total " + std::to_string(stats.currentUsage));
    }
    
    // 检查内存池完整性
    size_t totalPoolMemory = 0;
    size_t totalPoolUsed = 0;
    
    // 需要访问内部池信息（这里简化处理）
    if (stats.pooledMemory > 0) {
        totalPoolMemory = stats.pooledMemory;
        
        // 检查碎片率
        if (stats.fragmentedMemory > stats.pooledMemory * 0.5) {
            issues.push_back("High memory fragmentation: " + 
                           std::to_string(stats.fragmentedMemory * 100 / stats.pooledMemory) + 
                           "% of pool memory is fragmented");
        }
    }
    
    return issues;
}

std::string MultiGPUMemoryManager::generateMemoryReport() const {
    return m_impl->generateMemoryReport();
}

void MultiGPUMemoryManager::reset(bool releaseAll) {
    m_impl->reset(releaseAll);
}

GPUMemoryHandle MultiGPUMemoryManager::mapHostMemory(void* hostPtr, size_t size, int deviceId) {
    GPUMemoryHandle handle;
    
    if (!hostPtr || size == 0) {
        return handle; // 返回无效句柄
    }
    
#ifdef OSCEAN_CUDA_ENABLED
    // 检查设备是否支持CUDA
    if (m_impl->deviceSupportsCUDA(deviceId)) {
        // 注册主机内存为固定内存
        cudaError_t err = cudaHostRegister(hostPtr, size, cudaHostRegisterMapped);
        if (err == cudaSuccess) {
            // 获取设备指针
            void* devicePtr = nullptr;
            err = cudaHostGetDevicePointer(&devicePtr, hostPtr, 0);
            if (err == cudaSuccess) {
                handle.deviceId = deviceId;
                handle.devicePtr = devicePtr;
                handle.size = size;
                handle.memoryType = GPUMemoryType::PINNED;
                handle.isPooled = false;
                handle.allocationId = "mapped_" + std::to_string(reinterpret_cast<uintptr_t>(hostPtr));
                
                // 记录映射（需要通过Impl提供的方法）
                // 暂时跳过记录，因为需要在Impl中添加方法
                
                try {
                    OSCEAN_LOG_DEBUG("MultiGPUMemoryManager",
                                   "Mapped " + std::to_string(size) + 
                                   " bytes of host memory to device " + std::to_string(deviceId));
                } catch (...) {}
                
                return handle;
            }
            
            // 如果获取设备指针失败，取消注册
            cudaHostUnregister(hostPtr);
        }
    }
#endif
    
    // 后备方案：返回主机内存句柄
    handle.deviceId = deviceId;
    handle.devicePtr = hostPtr;
    handle.size = size;
    handle.memoryType = GPUMemoryType::HOST;
    handle.isPooled = false;
    handle.allocationId = "host_" + std::to_string(reinterpret_cast<uintptr_t>(hostPtr));
    
    return handle;
}

bool MultiGPUMemoryManager::unmapHostMemory(const GPUMemoryHandle& handle) {
    if (!handle.isValid() || handle.memoryType != GPUMemoryType::PINNED) {
        return false;
    }
    
    bool success = false;
    
#ifdef OSCEAN_CUDA_ENABLED
    // 查找原始主机指针
    // 在实际实现中，应该维护设备指针到主机指针的映射
    // 这里简化处理，假设可以从allocationId推断
    if (handle.allocationId.substr(0, 7) == "mapped_") {
        // 从allocationId提取主机指针地址
        std::string addrStr = handle.allocationId.substr(7);
        uintptr_t addr = std::stoull(addrStr);
        void* hostPtr = reinterpret_cast<void*>(addr);
        
        cudaError_t err = cudaHostUnregister(hostPtr);
        if (err == cudaSuccess || err == cudaErrorHostMemoryNotRegistered) {
            success = true;
            
            try {
                OSCEAN_LOG_DEBUG("MultiGPUMemoryManager",
                               "Unmapped host memory from device " + 
                               std::to_string(handle.deviceId));
            } catch (...) {}
        }
    }
#endif
    
    // 移除分配记录（需要通过Impl提供的方法）
    // 暂时跳过，因为需要在Impl中添加方法
    
    return success;
}

// GlobalMemoryManager实现
MultiGPUMemoryManager& GlobalMemoryManager::getInstance() {
    boost::lock_guard<boost::mutex> lock(s_mutex);
    if (!s_instance) {
        throw std::runtime_error("Global GPU memory manager not initialized");
    }
    return *s_instance;
}

void GlobalMemoryManager::initialize(const std::vector<GPUDeviceInfo>& devices,
                                    const MemoryPoolConfig& poolConfig,
                                    const MemoryTransferConfig& transferConfig) {
    boost::lock_guard<boost::mutex> lock(s_mutex);
    s_instance = std::make_unique<MultiGPUMemoryManager>(devices, poolConfig, transferConfig);
}

void GlobalMemoryManager::destroy() {
    boost::lock_guard<boost::mutex> lock(s_mutex);
    s_instance.reset();
}

} // namespace oscean::common_utils::gpu