/**
 * @file opencl_memory_manager.cpp
 * @brief OpenCL内存管理器实现
 */

#include "common_utils/gpu/multi_gpu_memory_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include "opencl_utils.h"
#include <map>
#include <queue>
#include <mutex>

namespace oscean::common_utils::gpu::opencl {

/**
 * @brief OpenCL内存块信息
 */
struct CLMemoryBlock {
    cl_mem buffer = nullptr;
    size_t size = 0;
    int deviceId = -1;
    bool inUse = false;
    cl_context context = nullptr;
};

/**
 * @brief OpenCL内存池实现
 */
class CLMemoryPool {
private:
    int m_deviceId;
    cl_context m_context;
    cl_device_id m_device;
    size_t m_totalAllocated = 0;
    size_t m_totalUsed = 0;
    
    // 按大小组织的空闲块队列
    std::map<size_t, std::queue<CLMemoryBlock*>> m_freeBlocks;
    
    // 所有分配的块
    std::vector<std::unique_ptr<CLMemoryBlock>> m_allBlocks;
    
    // 最小分配大小（避免碎片）
    static constexpr size_t MIN_BLOCK_SIZE = 1024 * 1024;  // 1MB
    
    // 对齐要求
    static constexpr size_t ALIGNMENT = 256;
    
    mutable std::mutex m_mutex;
    
public:
    CLMemoryPool(int deviceId, cl_context context, cl_device_id device) 
        : m_deviceId(deviceId), m_context(context), m_device(device) {
    }
    
    ~CLMemoryPool() {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 释放所有OpenCL内存
        for (auto& block : m_allBlocks) {
            if (block->buffer) {
                clReleaseMemObject(block->buffer);
            }
        }
    }
    
    /**
     * @brief 分配内存
     */
    CLMemoryBlock* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 对齐大小
        size = alignSize(size);
        
        // 查找合适的空闲块
        CLMemoryBlock* block = findFreeBlock(size);
        
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
    void deallocate(CLMemoryBlock* block) {
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
        std::vector<std::unique_ptr<CLMemoryBlock>> newBlocks;
        m_freeBlocks.clear();
        
        for (auto& block : m_allBlocks) {
            if (block->inUse) {
                newBlocks.push_back(std::move(block));
            } else {
                if (block->buffer) {
                    clReleaseMemObject(block->buffer);
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
    CLMemoryBlock* findFreeBlock(size_t size) {
        // 查找大小完全匹配的块
        auto it = m_freeBlocks.find(size);
        if (it != m_freeBlocks.end() && !it->second.empty()) {
            CLMemoryBlock* block = it->second.front();
            it->second.pop();
            return block;
        }
        
        // 查找更大的块
        for (auto& [blockSize, queue] : m_freeBlocks) {
            if (blockSize >= size && !queue.empty()) {
                CLMemoryBlock* block = queue.front();
                queue.pop();
                return block;
            }
        }
        
        return nullptr;
    }
    
    /**
     * @brief 分配新的内存块
     */
    CLMemoryBlock* allocateNewBlock(size_t size) {
        // 确保最小分配大小
        size = std::max(size, MIN_BLOCK_SIZE);
        
        cl_int err;
        cl_mem buffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, size, nullptr, &err);
        
        if (err != CL_SUCCESS) {
            OSCEAN_LOG_ERROR("OpenCL内存分配失败: {}", getOpenCLErrorString(err));
            return nullptr;
        }
        
        auto block = std::make_unique<CLMemoryBlock>();
        block->buffer = buffer;
        block->size = size;
        block->deviceId = m_deviceId;
        block->context = m_context;
        block->inUse = false;
        
        CLMemoryBlock* blockPtr = block.get();
        m_allBlocks.push_back(std::move(block));
        m_totalAllocated += size;
        
        return blockPtr;
    }
};

/**
 * @brief OpenCL内存管理器
 */
class CLMemoryManager {
private:
    struct DeviceContext {
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        std::unique_ptr<CLMemoryPool> pool;
    };
    
    std::map<int, DeviceContext> m_deviceContexts;
    mutable std::mutex m_mutex;
    
public:
    static CLMemoryManager& getInstance() {
        static CLMemoryManager instance;
        return instance;
    }
    
    /**
     * @brief 初始化设备上下文
     */
    bool initializeDevice(int deviceId, cl_platform_id platform, cl_device_id device) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (m_deviceContexts.find(deviceId) != m_deviceContexts.end()) {
            return true;  // 已经初始化
        }
        
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
            0
        };
        
        cl_int err;
        cl_context context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            OSCEAN_LOG_ERROR("创建OpenCL上下文失败: {}", getOpenCLErrorString(err));
            return false;
        }
        
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            OSCEAN_LOG_ERROR("创建OpenCL命令队列失败: {}", getOpenCLErrorString(err));
            return false;
        }
        
        DeviceContext ctx;
        ctx.platform = platform;
        ctx.device = device;
        ctx.context = context;
        ctx.queue = queue;
        ctx.pool = std::make_unique<CLMemoryPool>(deviceId, context, device);
        
        m_deviceContexts[deviceId] = std::move(ctx);
        return true;
    }
    
    /**
     * @brief 在指定设备上分配内存
     */
    cl_mem allocateOnDevice(int deviceId, size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        auto it = m_deviceContexts.find(deviceId);
        if (it == m_deviceContexts.end()) {
            OSCEAN_LOG_ERROR("设备 {} 未初始化", deviceId);
            return nullptr;
        }
        
        CLMemoryBlock* block = it->second.pool->allocate(size);
        return block ? block->buffer : nullptr;
    }
    
    /**
     * @brief 释放设备内存
     */
    void deallocateOnDevice(int deviceId, cl_mem buffer) {
        if (!buffer) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        
        auto it = m_deviceContexts.find(deviceId);
        if (it != m_deviceContexts.end()) {
            // 简化实现：需要跟踪buffer到block的映射
            // 实际实现中应该维护一个buffer到block的映射表
        }
    }
    
    /**
     * @brief 跨设备复制数据
     */
    cl_int copyBetweenDevices(
        cl_mem dst, int dstDevice,
        cl_mem src, int srcDevice,
        size_t size) {
        
        std::lock_guard<std::mutex> lock(m_mutex);
        
        auto srcIt = m_deviceContexts.find(srcDevice);
        auto dstIt = m_deviceContexts.find(dstDevice);
        
        if (srcIt == m_deviceContexts.end() || dstIt == m_deviceContexts.end()) {
            return CL_INVALID_DEVICE;
        }
        
        // 如果是同一设备，直接复制
        if (srcDevice == dstDevice) {
            return clEnqueueCopyBuffer(srcIt->second.queue, src, dst, 0, 0, size, 0, nullptr, nullptr);
        }
        
        // 跨设备复制：通过主机内存中转
        std::vector<unsigned char> hostBuffer(size);
        
        // 从源设备读取
        cl_int err = clEnqueueReadBuffer(srcIt->second.queue, src, CL_TRUE, 0, size, 
                                        hostBuffer.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return err;
        
        // 写入目标设备
        err = clEnqueueWriteBuffer(dstIt->second.queue, dst, CL_TRUE, 0, size,
                                  hostBuffer.data(), 0, nullptr, nullptr);
        
        return err;
    }
    
    /**
     * @brief 获取设备内存信息
     */
    void getDeviceMemoryInfo(int deviceId, size_t& total, size_t& available) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        auto it = m_deviceContexts.find(deviceId);
        if (it == m_deviceContexts.end()) {
            total = 0;
            available = 0;
            return;
        }
        
        // 获取总内存
        clGetDeviceInfo(it->second.device, CL_DEVICE_GLOBAL_MEM_SIZE, 
                       sizeof(size_t), &total, nullptr);
        
        // OpenCL没有直接获取可用内存的API，使用内存池统计
        size_t allocated, used;
        it->second.pool->getMemoryStats(allocated, used);
        available = total - allocated;
    }
    
    /**
     * @brief 执行所有设备的碎片整理
     */
    void defragmentAll() {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        for (auto& [deviceId, ctx] : m_deviceContexts) {
            ctx.pool->defragment();
        }
    }
    
    /**
     * @brief 清理资源
     */
    ~CLMemoryManager() {
        for (auto& [deviceId, ctx] : m_deviceContexts) {
            ctx.pool.reset();  // 先释放内存池
            clReleaseCommandQueue(ctx.queue);
            clReleaseContext(ctx.context);
        }
    }
    
private:
    CLMemoryManager() = default;
    CLMemoryManager(const CLMemoryManager&) = delete;
    CLMemoryManager& operator=(const CLMemoryManager&) = delete;
};

// 导出的C接口函数
extern "C" {

bool clInitializeDevice(int deviceId, void* platform, void* device) {
    return CLMemoryManager::getInstance().initializeDevice(
        deviceId, 
        static_cast<cl_platform_id>(platform),
        static_cast<cl_device_id>(device));
}

void* clAllocateMemory(int deviceId, size_t size) {
    return CLMemoryManager::getInstance().allocateOnDevice(deviceId, size);
}

void clDeallocateMemory(int deviceId, void* buffer) {
    CLMemoryManager::getInstance().deallocateOnDevice(deviceId, static_cast<cl_mem>(buffer));
}

cl_int clCopyBetweenDevices(
    void* dst, int dstDevice,
    void* src, int srcDevice,
    size_t size) {
    return CLMemoryManager::getInstance().copyBetweenDevices(
        static_cast<cl_mem>(dst), dstDevice,
        static_cast<cl_mem>(src), srcDevice,
        size);
}

void clGetDeviceMemoryInfo(int deviceId, size_t* total, size_t* available) {
    CLMemoryManager::getInstance().getDeviceMemoryInfo(deviceId, *total, *available);
}

void clDefragmentMemory() {
    CLMemoryManager::getInstance().defragmentAll();
}

} // extern "C"

} // namespace oscean::common_utils::gpu::opencl 