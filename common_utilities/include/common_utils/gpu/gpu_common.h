/**
 * @file gpu_common.h
 * @brief GPU通用定义和宏 - OSCEAN GPU框架基础
 * 
 * 提供跨平台GPU开发的通用定义、宏和基础类型
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace oscean::common_utils::gpu {

// === GPU平台检测宏 ===

// CUDA检测
#ifdef __CUDACC__
    #define OSCEAN_CUDA_AVAILABLE 1
#else
    #define OSCEAN_CUDA_AVAILABLE 0
#endif

// OpenCL检测
#ifdef CL_VERSION_1_0
    #define OSCEAN_OPENCL_AVAILABLE 1
#else
    #define OSCEAN_OPENCL_AVAILABLE 0
#endif

// ROCm/HIP检测
#ifdef __HIP__
    #define OSCEAN_ROCM_AVAILABLE 1
#else
    #define OSCEAN_ROCM_AVAILABLE 0
#endif

// Metal检测（仅macOS/iOS）
#ifdef __APPLE__
    #include <TargetConditionals.h>
    #if TARGET_OS_MAC || TARGET_OS_IPHONE
        #define OSCEAN_METAL_AVAILABLE 1
    #else
        #define OSCEAN_METAL_AVAILABLE 0
    #endif
#else
    #define OSCEAN_METAL_AVAILABLE 0
#endif

// === GPU错误码定义 ===

/**
 * @brief GPU操作错误码
 */
enum class GPUError {
    SUCCESS = 0,                    ///< 操作成功
    DEVICE_NOT_FOUND,              ///< 未找到GPU设备
    OUT_OF_MEMORY,                 ///< GPU内存不足
    INVALID_DEVICE,                ///< 无效的设备ID
    INVALID_CONTEXT,               ///< 无效的上下文
    INVALID_KERNEL,                ///< 无效的核函数
    KERNEL_LAUNCH_FAILED,          ///< 核函数启动失败
    SYNCHRONIZATION_FAILED,        ///< 同步失败
    TRANSFER_FAILED,               ///< 数据传输失败
    COMPILATION_FAILED,            ///< 核函数编译失败
    NOT_SUPPORTED,                 ///< 功能不支持
    INITIALIZATION_FAILED,         ///< 初始化失败
    UNKNOWN_ERROR                  ///< 未知错误
};

/**
 * @brief 将GPU错误码转换为字符串
 */
inline std::string gpuErrorToString(GPUError error) {
    switch (error) {
        case GPUError::SUCCESS: return "Success";
        case GPUError::DEVICE_NOT_FOUND: return "Device not found";
        case GPUError::OUT_OF_MEMORY: return "Out of GPU memory";
        case GPUError::INVALID_DEVICE: return "Invalid device";
        case GPUError::INVALID_CONTEXT: return "Invalid context";
        case GPUError::INVALID_KERNEL: return "Invalid kernel";
        case GPUError::KERNEL_LAUNCH_FAILED: return "Kernel launch failed";
        case GPUError::SYNCHRONIZATION_FAILED: return "Synchronization failed";
        case GPUError::TRANSFER_FAILED: return "Transfer failed";
        case GPUError::COMPILATION_FAILED: return "Compilation failed";
        case GPUError::NOT_SUPPORTED: return "Not supported";
        case GPUError::INITIALIZATION_FAILED: return "Initialization failed";
        case GPUError::UNKNOWN_ERROR: return "Unknown error";
        default: return "Unrecognized error";
    }
}

// === GPU内存类型 ===

/**
 * @brief GPU内存类型枚举
 */
enum class GPUMemoryType {
    DEVICE,         ///< 设备内存（GPU全局内存）
    HOST,           ///< 主机内存（CPU内存）
    PINNED,         ///< 固定内存（页锁定内存）
    UNIFIED,        ///< 统一内存（CUDA统一内存/OpenCL SVM）
    CONSTANT,       ///< 常量内存
    TEXTURE,        ///< 纹理内存
    SHARED          ///< 共享内存（块内共享）
};

// === GPU数据传输方向 ===

/**
 * @brief 数据传输方向
 */
enum class GPUTransferDirection {
    HOST_TO_DEVICE,     ///< 从主机到设备
    DEVICE_TO_HOST,     ///< 从设备到主机
    DEVICE_TO_DEVICE,   ///< 设备间传输
    HOST_TO_HOST        ///< 主机间传输（用于固定内存）
};

// === GPU核函数维度 ===

/**
 * @brief GPU核函数执行维度
 */
struct GPUDimension {
    size_t x;
    size_t y;
    size_t z;
    
    GPUDimension(size_t x_val = 1, size_t y_val = 1, size_t z_val = 1)
        : x(x_val), y(y_val), z(z_val) {}
    
    size_t total() const { return x * y * z; }
};

// === GPU性能计时器 ===

/**
 * @brief GPU性能计时器接口
 */
class IGPUTimer {
public:
    virtual ~IGPUTimer() = default;
    
    /**
     * @brief 开始计时
     */
    virtual void start() = 0;
    
    /**
     * @brief 停止计时
     */
    virtual void stop() = 0;
    
    /**
     * @brief 获取经过的时间（毫秒）
     */
    virtual float getElapsedTime() const = 0;
    
    /**
     * @brief 重置计时器
     */
    virtual void reset() = 0;
};

// === GPU内存信息 ===

/**
 * @brief GPU内存使用信息
 */
struct GPUMemoryInfo {
    size_t totalMemory;      ///< 总内存（字节）
    size_t freeMemory;       ///< 空闲内存（字节）
    size_t usedMemory;       ///< 已用内存（字节）
    
    double getUsagePercent() const {
        return totalMemory > 0 ? (100.0 * usedMemory / totalMemory) : 0.0;
    }
};

// === GPU设备能力 ===

/**
 * @brief GPU设备能力标志
 */
struct GPUCapabilities {
    bool supportsDoublePrecision;    ///< 支持双精度浮点
    bool supportsAtomics;            ///< 支持原子操作
    bool supportsSharedMemory;       ///< 支持共享内存
    bool supportsTextureMemory;      ///< 支持纹理内存
    bool supportsUnifiedMemory;      ///< 支持统一内存
    bool supportsConcurrentKernels;  ///< 支持并发核函数
    bool supportsAsyncTransfer;      ///< 支持异步传输
    bool supportsDynamicParallelism; ///< 支持动态并行
    bool supportsTensorCores;        ///< 支持Tensor核心
    bool supportsRayTracing;         ///< 支持光线追踪
};

// === 便捷宏定义 ===

/**
 * @brief GPU错误检查宏
 */
#define GPU_CHECK_ERROR(error) \
    do { \
        if ((error) != GPUError::SUCCESS) { \
            throw std::runtime_error("GPU Error: " + gpuErrorToString(error) + \
                                   " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

/**
 * @brief GPU安全删除宏
 */
#define GPU_SAFE_DELETE(ptr) \
    do { \
        if (ptr) { \
            delete ptr; \
            ptr = nullptr; \
        } \
    } while(0)

/**
 * @brief GPU安全释放宏
 */
#define GPU_SAFE_FREE(ptr, freeFunc) \
    do { \
        if (ptr) { \
            freeFunc(ptr); \
            ptr = nullptr; \
        } \
    } while(0)

// === 对齐宏 ===

/**
 * @brief 内存对齐宏（用于GPU内存分配）
 */
#define GPU_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

/**
 * @brief 默认GPU内存对齐（256字节）
 */
#define GPU_DEFAULT_ALIGNMENT 256

// === 核函数配置 ===

/**
 * @brief 默认块大小（线程数）
 */
constexpr size_t DEFAULT_BLOCK_SIZE = 256;

/**
 * @brief 最大块大小
 */
constexpr size_t MAX_BLOCK_SIZE = 1024;

/**
 * @brief Warp/Wavefront大小
 */
constexpr size_t WARP_SIZE = 32;  // NVIDIA
constexpr size_t WAVEFRONT_SIZE = 64;  // AMD

} // namespace oscean::common_utils::gpu 