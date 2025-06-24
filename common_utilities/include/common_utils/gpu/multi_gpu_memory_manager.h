/**
 * @file multi_gpu_memory_manager.h
 * @brief 多GPU统一内存管理器接口
 */

#pragma once

#include "gpu_common.h"
#include "gpu_device_info.h"
#include <boost/thread/mutex.hpp>
#include <boost/atomic.hpp>
#include <boost/optional.hpp>
#include <boost/function.hpp>
#include <memory>
#include <map>
#include <queue>
#include <vector>

namespace oscean::common_utils::gpu {

/**
 * @brief GPU内存句柄
 */
struct GPUMemoryHandle {
    int deviceId = -1;                     // 设备ID
    void* devicePtr = nullptr;             // 设备内存指针
    size_t size = 0;                       // 内存大小
    GPUMemoryType memoryType = GPUMemoryType::DEVICE;  // 内存类型
    ComputeAPI api = ComputeAPI::AUTO_DETECT;          // 使用的API
    bool isPooled = false;                 // 是否来自内存池
    std::string allocationId;              // 分配ID
    
    bool isValid() const { return devicePtr != nullptr && size > 0; }
};

/**
 * @brief GPU内存池配置
 */
struct MemoryPoolConfig {
    size_t initialPoolSize = 256 * 1024 * 1024;    // 初始池大小 (256MB)
    size_t maxPoolSize = 2ULL * 1024 * 1024 * 1024; // 最大池大小 (2GB)
    size_t blockSize = 1024 * 1024;                // 块大小 (1MB)
    bool enableGrowth = true;                       // 允许池增长
    float growthFactor = 1.5f;                      // 增长因子
    bool enableDefragmentation = true;              // 启用碎片整理
    size_t defragmentationThreshold = 100 * 1024 * 1024; // 碎片整理阈值
};

/**
 * @brief 内存传输配置
 */
struct MemoryTransferConfig {
    bool enablePeerAccess = true;           // 启用GPU间直接访问
    bool enableAsyncTransfer = true;        // 启用异步传输
    bool enableCompression = false;         // 启用压缩传输
    size_t transferBufferSize = 64 * 1024 * 1024; // 传输缓冲区大小
    int maxConcurrentTransfers = 4;         // 最大并发传输数
};

/**
 * @brief 内存分配策略
 */
enum class AllocationStrategy {
    FIRST_FIT,          // 首次适配
    BEST_FIT,           // 最佳适配
    WORST_FIT,          // 最差适配
    BUDDY_SYSTEM,       // 伙伴系统
    SLAB_ALLOCATION     // SLAB分配
};

/**
 * @brief 内存传输类型
 */
enum class TransferType {
    HOST_TO_DEVICE,     // 主机到设备
    DEVICE_TO_HOST,     // 设备到主机
    DEVICE_TO_DEVICE,   // 设备到设备（同GPU）
    PEER_TO_PEER,       // GPU间直接传输
    UNIFIED_ACCESS      // 统一内存访问
};

/**
 * @brief 内存使用统计
 */
struct MemoryUsageStats {
    size_t totalAllocated = 0;          // 总分配量
    size_t totalFreed = 0;              // 总释放量
    size_t currentUsage = 0;            // 当前使用量
    size_t peakUsage = 0;               // 峰值使用量
    size_t pooledMemory = 0;            // 池化内存量
    size_t fragmentedMemory = 0;        // 碎片内存量
    size_t allocationCount = 0;         // 分配次数
    size_t deallocationCount = 0;       // 释放次数
    double averageAllocationSize = 0.0; // 平均分配大小
    
    // 按设备统计
    std::map<int, size_t> deviceUsage;
    std::map<int, size_t> devicePeakUsage;
};

/**
 * @brief 内存传输统计
 */
struct TransferStats {
    size_t totalTransfers = 0;          // 总传输次数
    size_t totalBytesTransferred = 0;   // 总传输字节数
    double averageTransferSize = 0.0;   // 平均传输大小
    double averageTransferTime = 0.0;   // 平均传输时间(ms)
    double peakBandwidth = 0.0;         // 峰值带宽(GB/s)
    
    // 按传输类型统计
    std::map<TransferType, size_t> transfersByType;
    std::map<TransferType, size_t> bytesByType;
};

/**
 * @brief 内存分配请求
 */
struct AllocationRequest {
    size_t size;                        // 请求大小
    int preferredDeviceId = -1;         // 偏好设备
    GPUMemoryType memoryType = GPUMemoryType::DEVICE;
    AllocationStrategy strategy = AllocationStrategy::BEST_FIT;
    bool allowFallback = true;          // 允许降级到其他设备
    size_t alignment = 256;             // 内存对齐
    std::string tag;                    // 分配标签（用于调试）
};

/**
 * @brief 内存传输请求
 */
struct TransferRequest {
    GPUMemoryHandle source;             // 源内存
    GPUMemoryHandle destination;        // 目标内存
    size_t offset = 0;                  // 偏移量
    size_t size = 0;                    // 传输大小（0表示全部）
    bool async = true;                  // 异步传输
    boost::function<void(bool)> callback; // 完成回调
};

/**
 * @brief 内存事件类型
 */
enum class MemoryEventType {
    ALLOCATION_SUCCESS,
    ALLOCATION_FAILED,
    DEALLOCATION,
    TRANSFER_STARTED,
    TRANSFER_COMPLETED,
    TRANSFER_FAILED,
    POOL_GROWN,
    POOL_SHRUNK,
    DEFRAGMENTATION_STARTED,
    DEFRAGMENTATION_COMPLETED,
    OUT_OF_MEMORY,
    MEMORY_LEAK_DETECTED
};

/**
 * @brief 内存事件
 */
struct MemoryEvent {
    MemoryEventType type;
    int deviceId;
    size_t size;
    std::string message;
    boost::chrono::steady_clock::time_point timestamp;
};

/**
 * @brief 内存事件回调
 */
typedef boost::function<void(const MemoryEvent&)> MemoryEventCallback;

/**
 * @brief GPU内存块（内部使用）
 */
struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    size_t usedSize = 0;
    bool isFree = true;
    int deviceId = -1;
    std::string allocationId;
    boost::chrono::steady_clock::time_point allocTime;
};

/**
 * @brief 多GPU统一内存管理器
 */
class MultiGPUMemoryManager {
public:
    /**
     * @brief 构造函数
     * @param devices GPU设备列表
     * @param poolConfig 内存池配置
     * @param transferConfig 传输配置
     */
    MultiGPUMemoryManager(const std::vector<GPUDeviceInfo>& devices,
                         const MemoryPoolConfig& poolConfig = MemoryPoolConfig(),
                         const MemoryTransferConfig& transferConfig = MemoryTransferConfig());
    
    /**
     * @brief 析构函数
     */
    ~MultiGPUMemoryManager();
    
    /**
     * @brief 分配GPU内存
     * @param request 分配请求
     * @return 内存句柄
     */
    GPUMemoryHandle allocate(const AllocationRequest& request);
    
    /**
     * @brief 释放GPU内存
     * @param handle 内存句柄
     * @return 是否成功
     */
    bool deallocate(const GPUMemoryHandle& handle);
    
    /**
     * @brief 跨GPU数据传输
     * @param request 传输请求
     * @return 是否成功启动传输
     */
    bool transfer(const TransferRequest& request);
    
    /**
     * @brief 同步等待所有传输完成
     * @param timeoutMs 超时时间（毫秒）
     * @return 是否全部完成
     */
    bool synchronize(int timeoutMs = -1);
    
    /**
     * @brief 预分配内存池
     * @param deviceId 设备ID
     * @param size 预分配大小
     * @return 是否成功
     */
    bool preallocatePool(int deviceId, size_t size);
    
    /**
     * @brief 收缩内存池
     * @param deviceId 设备ID（-1表示所有设备）
     */
    void shrinkPools(int deviceId = -1);
    
    /**
     * @brief 执行内存碎片整理
     * @param deviceId 设备ID（-1表示所有设备）
     * @return 整理出的空闲内存大小
     */
    size_t defragment(int deviceId = -1);
    
    /**
     * @brief 获取内存使用统计
     * @return 内存使用统计
     */
    MemoryUsageStats getUsageStats() const;
    
    /**
     * @brief 获取传输统计
     * @return 传输统计
     */
    TransferStats getTransferStats() const;
    
    /**
     * @brief 获取设备可用内存
     * @param deviceId 设备ID
     * @return 可用内存大小
     */
    size_t getAvailableMemory(int deviceId) const;
    
    /**
     * @brief 检查GPU间是否可以直接访问
     * @param srcDevice 源设备ID
     * @param dstDevice 目标设备ID
     * @return 是否支持直接访问
     */
    bool canAccessPeer(int srcDevice, int dstDevice) const;
    
    /**
     * @brief 启用GPU间直接访问
     * @param srcDevice 源设备ID
     * @param dstDevice 目标设备ID
     * @return 是否成功
     */
    bool enablePeerAccess(int srcDevice, int dstDevice);
    
    /**
     * @brief 禁用GPU间直接访问
     * @param srcDevice 源设备ID
     * @param dstDevice 目标设备ID
     */
    void disablePeerAccess(int srcDevice, int dstDevice);
    
    /**
     * @brief 注册内存事件回调
     * @param callback 事件回调函数
     */
    void registerEventCallback(MemoryEventCallback callback);
    
    /**
     * @brief 设置内存分配策略
     * @param strategy 新的分配策略
     */
    void setAllocationStrategy(AllocationStrategy strategy);
    
    /**
     * @brief 获取当前分配策略
     * @return 当前分配策略
     */
    AllocationStrategy getAllocationStrategy() const;
    
    /**
     * @brief 验证内存完整性
     * @return 检测到的问题列表
     */
    std::vector<std::string> validateMemoryIntegrity() const;
    
    /**
     * @brief 导出内存使用报告
     * @return 详细的内存使用报告
     */
    std::string generateMemoryReport() const;
    
    /**
     * @brief 重置内存管理器
     * @param releaseAll 是否释放所有内存
     */
    void reset(bool releaseAll = true);
    
    /**
     * @brief 创建内存映射（零拷贝）
     * @param hostPtr 主机内存指针
     * @param size 内存大小
     * @param deviceId 设备ID
     * @return GPU内存句柄
     */
    GPUMemoryHandle mapHostMemory(void* hostPtr, size_t size, int deviceId);
    
    /**
     * @brief 取消内存映射
     * @param handle 内存句柄
     * @return 是否成功
     */
    bool unmapHostMemory(const GPUMemoryHandle& handle);
    
private:
    // 私有实现
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief 全局内存管理器
 */
class GlobalMemoryManager {
public:
    /**
     * @brief 获取全局内存管理器实例
     * @return 内存管理器实例
     */
    static MultiGPUMemoryManager& getInstance();
    
    /**
     * @brief 初始化全局内存管理器
     * @param devices GPU设备列表
     * @param poolConfig 内存池配置
     * @param transferConfig 传输配置
     */
    static void initialize(const std::vector<GPUDeviceInfo>& devices,
                          const MemoryPoolConfig& poolConfig = MemoryPoolConfig(),
                          const MemoryTransferConfig& transferConfig = MemoryTransferConfig());
    
    /**
     * @brief 销毁全局内存管理器
     */
    static void destroy();
    
private:
    static std::unique_ptr<MultiGPUMemoryManager> s_instance;
    static boost::mutex s_mutex;
};

/**
 * @brief 智能GPU内存指针（RAII）
 */
template<typename T>
class GPUMemoryPtr {
public:
    GPUMemoryPtr() = default;
    
    explicit GPUMemoryPtr(const GPUMemoryHandle& handle, MultiGPUMemoryManager* manager)
        : m_handle(handle), m_manager(manager) {}
    
    ~GPUMemoryPtr() {
        release();
    }
    
    // 禁止拷贝
    GPUMemoryPtr(const GPUMemoryPtr&) = delete;
    GPUMemoryPtr& operator=(const GPUMemoryPtr&) = delete;
    
    // 移动语义
    GPUMemoryPtr(GPUMemoryPtr&& other) noexcept
        : m_handle(other.m_handle), m_manager(other.m_manager) {
        other.m_handle = GPUMemoryHandle();
        other.m_manager = nullptr;
    }
    
    GPUMemoryPtr& operator=(GPUMemoryPtr&& other) noexcept {
        if (this != &other) {
            release();
            m_handle = other.m_handle;
            m_manager = other.m_manager;
            other.m_handle = GPUMemoryHandle();
            other.m_manager = nullptr;
        }
        return *this;
    }
    
    T* get() const { return static_cast<T*>(m_handle.devicePtr); }
    size_t size() const { return m_handle.size; }
    int deviceId() const { return m_handle.deviceId; }
    bool isValid() const { return m_handle.isValid(); }
    
    const GPUMemoryHandle& handle() const { return m_handle; }
    
    void release() {
        if (m_manager && m_handle.isValid()) {
            m_manager->deallocate(m_handle);
            m_handle = GPUMemoryHandle();
        }
    }
    
private:
    GPUMemoryHandle m_handle;
    MultiGPUMemoryManager* m_manager = nullptr;
};

} // namespace oscean::common_utils::gpu 