/**
 * @file gpu_device_info.h
 * @brief GPU设备信息结构定义
 * 
 * 定义了GPU设备的详细信息结构，支持多厂商GPU
 */

#pragma once

#include "gpu_common.h"
#include <string>
#include <vector>
#include <boost/optional.hpp>

namespace oscean::common_utils::gpu {

/**
 * @brief GPU厂商类型枚举
 */
enum class GPUVendor {
    NVIDIA,     ///< NVIDIA GPU (CUDA支持)
    AMD,        ///< AMD GPU (ROCm/HIP支持)
    INTEL,      ///< Intel GPU (Level Zero支持)
    APPLE,      ///< Apple GPU (Metal支持)
    QUALCOMM,   ///< Qualcomm GPU (移动设备)
    ARM,        ///< ARM GPU (Mali等)
    UNKNOWN     ///< 未知厂商
};

/**
 * @brief GPU计算API类型
 */
enum class ComputeAPI {
    CUDA,           ///< NVIDIA CUDA
    ROCM_HIP,       ///< AMD ROCm/HIP
    OPENCL,         ///< OpenCL (跨平台)
    LEVEL_ZERO,     ///< Intel Level Zero
    METAL,          ///< Apple Metal
    DIRECTCOMPUTE,  ///< DirectCompute (Windows)
    VULKAN_COMPUTE, ///< Vulkan Compute
    AUTO_DETECT     ///< 自动检测最优API
};

/**
 * @brief GPU架构信息
 */
struct GPUArchitecture {
    std::string name;           ///< 架构名称 (如 "Ampere", "RDNA 3")
    int majorVersion;           ///< 主版本号
    int minorVersion;           ///< 次版本号
    int computeCapability;      ///< 计算能力 (CUDA概念，其他API映射)
};

/**
 * @brief GPU时钟信息
 */
struct GPUClockInfo {
    int baseClock;              ///< 基础时钟频率 (MHz)
    int boostClock;             ///< 加速时钟频率 (MHz)
    int memoryClock;            ///< 内存时钟频率 (MHz)
    int currentCoreClock;       ///< 当前核心时钟 (MHz)
    int currentMemoryClock;     ///< 当前内存时钟 (MHz)
};

/**
 * @brief GPU温度信息
 */
struct GPUThermalInfo {
    float currentTemp;          ///< 当前温度 (°C)
    float maxTemp;              ///< 最高温度限制 (°C)
    float throttleTemp;         ///< 降频温度阈值 (°C)
    int fanSpeed;               ///< 风扇转速 (%)
};

/**
 * @brief GPU功耗信息
 */
struct GPUPowerInfo {
    float currentPower;         ///< 当前功耗 (W)
    float maxPower;             ///< 最大功耗 (W)
    float powerLimit;           ///< 功耗限制 (W)
    float powerEfficiency;      ///< 功耗效率评分
};

/**
 * @brief GPU计算单元信息
 */
struct GPUComputeUnits {
    int multiprocessorCount;    ///< 多处理器数量 (SM/CU)
    int coresPerMP;             ///< 每个多处理器的核心数
    int totalCores;             ///< 总核心数
    int tensorCores;            ///< Tensor核心数 (如果有)
    int rtCores;                ///< RT核心数 (如果有)
};

/**
 * @brief GPU内存详细信息
 */
struct GPUMemoryDetails {
    size_t totalGlobalMemory;   ///< 总全局内存 (字节)
    size_t freeGlobalMemory;    ///< 空闲全局内存 (字节)
    size_t l2CacheSize;         ///< L2缓存大小 (字节)
    size_t sharedMemoryPerBlock;///< 每个块的共享内存 (字节)
    size_t constantMemory;      ///< 常量内存大小 (字节)
    int memoryBusWidth;         ///< 内存总线宽度 (位)
    double memoryBandwidth;     ///< 内存带宽 (GB/s)
};

/**
 * @brief GPU执行限制
 */
struct GPUExecutionLimits {
    GPUDimension maxThreadsPerBlock;    ///< 每个块的最大线程数
    GPUDimension maxBlockDimension;     ///< 最大块维度
    GPUDimension maxGridDimension;      ///< 最大网格维度
    int maxRegistersPerBlock;           ///< 每个块的最大寄存器数
    int maxRegistersPerThread;          ///< 每个线程的最大寄存器数
    int warpSize;                       ///< Warp/Wavefront大小
    int maxWarpsPerMP;                  ///< 每个多处理器的最大Warp数
};

/**
 * @brief GPU设备信息结构
 */
struct GPUDeviceInfo {
    // 基本信息
    int deviceId;                       ///< 设备ID
    std::string name;                   ///< 设备名称
    GPUVendor vendor;                   ///< 厂商
    std::vector<ComputeAPI> supportedAPIs; ///< 支持的计算API列表
    std::string driverVersion;          ///< 驱动版本
    std::string pcieBusId;              ///< PCIe总线ID
    
    // 架构信息
    GPUArchitecture architecture;       ///< GPU架构
    
    // 性能相关
    GPUClockInfo clockInfo;             ///< 时钟信息
    GPUComputeUnits computeUnits;       ///< 计算单元信息
    GPUMemoryDetails memoryDetails;     ///< 内存详情
    GPUExecutionLimits executionLimits; ///< 执行限制
    
    // 能力标志
    GPUCapabilities capabilities;       ///< 设备能力
    
    // 状态信息
    GPUThermalInfo thermalInfo;         ///< 温度信息
    GPUPowerInfo powerInfo;             ///< 功耗信息
    
    // 性能评分
    int performanceScore;               ///< 性能评分 (0-100)
    
    // 扩展信息
    std::vector<std::pair<std::string, std::string>> extendedInfo; ///< 扩展属性
    
    /**
     * @brief 自动选择最优API
     * @return 推荐的计算API
     */
    ComputeAPI getBestAPI() const {
        // NVIDIA GPU优先使用CUDA
        if (vendor == GPUVendor::NVIDIA && hasAPI(ComputeAPI::CUDA)) {
            return ComputeAPI::CUDA;
        }
        // AMD GPU优先使用ROCm/HIP
        else if (vendor == GPUVendor::AMD && hasAPI(ComputeAPI::ROCM_HIP)) {
            return ComputeAPI::ROCM_HIP;
        }
        // Intel GPU优先使用Level Zero
        else if (vendor == GPUVendor::INTEL && hasAPI(ComputeAPI::LEVEL_ZERO)) {
            return ComputeAPI::LEVEL_ZERO;
        }
        // Apple GPU使用Metal
        else if (vendor == GPUVendor::APPLE && hasAPI(ComputeAPI::METAL)) {
            return ComputeAPI::METAL;
        }
        // 其他情况使用OpenCL作为通用后备
        else if (hasAPI(ComputeAPI::OPENCL)) {
            return ComputeAPI::OPENCL;
        }
        // 如果都不支持，返回自动检测
        return ComputeAPI::AUTO_DETECT;
    }
    
    /**
     * @brief 检查是否支持特定API
     * @param api 计算API类型
     * @return 如果支持返回true
     */
    bool hasAPI(ComputeAPI api) const {
        return std::find(supportedAPIs.begin(), supportedAPIs.end(), api) 
               != supportedAPIs.end();
    }
    
    /**
     * @brief 获取可用内存百分比
     * @return 可用内存百分比 (0-100)
     */
    double getMemoryAvailablePercent() const {
        if (memoryDetails.totalGlobalMemory == 0) return 0.0;
        return 100.0 * memoryDetails.freeGlobalMemory / memoryDetails.totalGlobalMemory;
    }
    
    /**
     * @brief 获取内存使用量
     * @return 已使用的内存（字节）
     */
    size_t getMemoryUsed() const {
        return memoryDetails.totalGlobalMemory - memoryDetails.freeGlobalMemory;
    }
    
    /**
     * @brief 检查是否为高性能GPU
     * @return 如果性能评分>=70返回true
     */
    bool isHighPerformance() const {
        return performanceScore >= 70;
    }
    
    /**
     * @brief 检查是否支持AI/ML加速
     * @return 如果有Tensor核心返回true
     */
    bool hasAIAcceleration() const {
        return computeUnits.tensorCores > 0 || capabilities.supportsTensorCores;
    }
    
    /**
     * @brief 获取设备描述字符串
     * @return 格式化的设备描述
     */
    std::string getDescription() const {
        return name + " (" + vendorToString(vendor) + ", " + 
               std::to_string(memoryDetails.totalGlobalMemory / (1024*1024*1024)) + " GB)";
    }
    
    /**
     * @brief 将厂商枚举转换为字符串
     */
    static std::string vendorToString(GPUVendor vendor) {
        switch (vendor) {
            case GPUVendor::NVIDIA: return "NVIDIA";
            case GPUVendor::AMD: return "AMD";
            case GPUVendor::INTEL: return "Intel";
            case GPUVendor::APPLE: return "Apple";
            case GPUVendor::QUALCOMM: return "Qualcomm";
            case GPUVendor::ARM: return "ARM";
            default: return "Unknown";
        }
    }
};

/**
 * @brief GPU配置结构
 */
struct GPUConfiguration {
    GPUDeviceInfo primaryDevice;                    ///< 主GPU设备
    std::vector<GPUDeviceInfo> secondaryDevices;   ///< 辅助GPU设备列表
    ComputeAPI computeAPI;                          ///< 选定的计算API
    bool enableMultiGPU;                            ///< 是否启用多GPU
    bool enablePeerAccess;                          ///< 是否启用GPU间直接访问
    size_t memoryPoolSize;                          ///< 内存池大小
    int maxConcurrentKernels;                       ///< 最大并发核函数数
};

/**
 * @brief 将ComputeAPI转换为字符串
 */
inline std::string computeAPIToString(ComputeAPI api) {
    switch (api) {
        case ComputeAPI::CUDA: return "CUDA";
        case ComputeAPI::ROCM_HIP: return "ROCm/HIP";
        case ComputeAPI::OPENCL: return "OpenCL";
        case ComputeAPI::LEVEL_ZERO: return "Level Zero";
        case ComputeAPI::METAL: return "Metal";
        case ComputeAPI::DIRECTCOMPUTE: return "DirectCompute";
        case ComputeAPI::VULKAN_COMPUTE: return "Vulkan Compute";
        case ComputeAPI::AUTO_DETECT: return "Auto Detect";
        default: return "Unknown";
    }
}

} // namespace oscean::common_utils::gpu 