/**
 * @file cuda_device_detector.cpp
 * @brief CUDA设备检测实现
 */

#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include "cuda_utils.h"
#include <algorithm>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

#include <sstream>
#include <iomanip>

namespace oscean::common_utils::gpu {

// CUDA Runtime API函数指针类型定义
typedef int (*cudaGetDeviceCount_t)(int*);
typedef int (*cudaGetDeviceProperties_t)(void*, int);
typedef int (*cudaSetDevice_t)(int);
typedef int (*cudaGetDevice_t)(int*);
typedef int (*cudaMemGetInfo_t)(size_t*, size_t*);
typedef int (*cudaDriverGetVersion_t)(int*);
typedef int (*cudaRuntimeGetVersion_t)(int*);
typedef const char* (*cudaGetErrorString_t)(int);

// CUDA设备属性结构（简化版本）
struct CudaDeviceProp {
    char name[256];
    char uuid[16];                        // 添加UUID字段
    char luid[8];                         // 添加LUID字段  
    unsigned int luidDeviceNodeMask;      // 添加LUID mask
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;         // 添加持久化L2缓存
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;        // 移到正确位置
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;       // 移到正确位置
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int sparseHipMappedArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingCudaArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
    int reserved2[2];
    int reserved[61];
    int reserved1[16];
    int reserved3[63];
};

// CUDA错误码
enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
};

class CUDADetector {
private:
    void* cudaLibrary = nullptr;
    
    // CUDA Runtime API函数指针
    cudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
    cudaGetDeviceProperties_t cudaGetDeviceProperties = nullptr;
    cudaSetDevice_t cudaSetDevice = nullptr;
    cudaGetDevice_t cudaGetDevice = nullptr;
    cudaMemGetInfo_t cudaMemGetInfo = nullptr;
    cudaDriverGetVersion_t cudaDriverGetVersion = nullptr;
    cudaRuntimeGetVersion_t cudaRuntimeGetVersion = nullptr;
    cudaGetErrorString_t cudaGetErrorString = nullptr;
    
    bool loadCUDALibrary() {
        #ifdef _WIN32
            // Windows: 尝试加载不同版本的CUDA Runtime
            const char* libraryNames[] = {
                "cudart64_12.dll",
                "cudart64_11.dll", 
                "cudart64_10.dll",
                "cudart64_90.dll",
                "cudart.dll"
            };
            
            for (const auto& libName : libraryNames) {
                cudaLibrary = LoadLibraryA(libName);
                if (cudaLibrary) {
                    OSCEAN_LOG_INFO("CUDADeviceDetector", std::string("Loaded CUDA library: ") + libName);
                    break;
                }
            }
        #else
            // Linux/macOS
            const char* libraryNames[] = {
                "libcudart.so.12",
                "libcudart.so.11", 
                "libcudart.so.10",
                "libcudart.so",
                "libcudart.dylib"
            };
            
            for (const auto& libName : libraryNames) {
                cudaLibrary = dlopen(libName, RTLD_LAZY);
                if (cudaLibrary) {
                    // OSCEAN_LOG_INFO("CUDADeviceDetector", "Loaded CUDA library: {}", libName); // TODO: Fix log format
                    break;
                }
            }
        #endif
        
        if (!cudaLibrary) {
            OSCEAN_LOG_DEBUG("CUDADeviceDetector", "CUDA runtime library not found");
            return false;
        }
        
        // 加载函数指针
        #ifdef _WIN32
            #define GET_PROC(name) name = (name##_t)GetProcAddress((HMODULE)cudaLibrary, #name)
        #else
            #define GET_PROC(name) name = (name##_t)dlsym(cudaLibrary, #name)
        #endif
        
        GET_PROC(cudaGetDeviceCount);
        GET_PROC(cudaGetDeviceProperties);
        GET_PROC(cudaSetDevice);
        GET_PROC(cudaGetDevice);
        GET_PROC(cudaMemGetInfo);
        GET_PROC(cudaDriverGetVersion);
        GET_PROC(cudaRuntimeGetVersion);
        GET_PROC(cudaGetErrorString);
        
        #undef GET_PROC
        
        // 验证所有必需的函数都已加载
        if (!cudaGetDeviceCount || !cudaGetDeviceProperties || 
            !cudaSetDevice || !cudaMemGetInfo) {
            OSCEAN_LOG_ERROR("CUDADeviceDetector", "Failed to load required CUDA functions");
            return false;
        }
        
        return true;
    }
    
    void unloadCUDALibrary() {
        if (cudaLibrary) {
            #ifdef _WIN32
                FreeLibrary((HMODULE)cudaLibrary);
            #else
                dlclose(cudaLibrary);
            #endif
            cudaLibrary = nullptr;
        }
    }
    
    std::string getCUDAErrorString(int error) {
        if (cudaGetErrorString) {
            return std::string(cudaGetErrorString(error));
        }
        return "Unknown CUDA error " + std::to_string(error);
    }
    
    GPUArchitecture getArchitectureInfo(int major, int minor) {
        GPUArchitecture arch;
        arch.majorVersion = major;
        arch.minorVersion = minor;
        arch.computeCapability = major * 10 + minor;
        
        // 根据计算能力确定架构名称
        if (major == 9) {
            arch.name = "Hopper";
        } else if (major == 8) {
            if (minor >= 6) arch.name = "Ada Lovelace";
            else arch.name = "Ampere";
        } else if (major == 7) {
            if (minor >= 5) arch.name = "Turing";
            else arch.name = "Volta";
        } else if (major == 6) {
            arch.name = "Pascal";
        } else if (major == 5) {
            arch.name = "Maxwell";
        } else if (major == 3) {
            arch.name = "Kepler";
        } else {
            arch.name = "Unknown";
        }
        
        return arch;
    }
    
    int calculateCoresPerSM(int major, int minor) {
        // 根据计算能力返回每个SM的CUDA核心数
        switch (major) {
            case 9: return 128;  // Hopper
            case 8: 
                if (minor >= 6) return 128;  // Ada Lovelace
                else return 64;  // Ampere
            case 7:
                if (minor >= 5) return 64;  // Turing
                else return 64;  // Volta
            case 6:
                if (minor == 0) return 64;  // Pascal GP100
                else return 128;  // Pascal GP102-GP108
            case 5: return 128;  // Maxwell
            case 3: return 192;  // Kepler
            default: return 32;
        }
    }
    
public:
    std::vector<GPUDeviceInfo> detect() {
        std::vector<GPUDeviceInfo> devices;
        
        OSCEAN_LOG_DEBUG("CUDADeviceDetector", "Starting CUDA detection...");
        
        if (!loadCUDALibrary()) {
            OSCEAN_LOG_DEBUG("CUDADeviceDetector", "Failed to load CUDA library");
            return devices;
        }
        
        OSCEAN_LOG_DEBUG("CUDADeviceDetector", "CUDA library loaded successfully");
        
        int deviceCount = 0;
        int result = cudaGetDeviceCount(&deviceCount);
        
        if (result != CUDA_SUCCESS) {
            OSCEAN_LOG_DEBUG("CUDADeviceDetector", "cudaGetDeviceCount failed: " + getCUDAErrorString(result));
            unloadCUDALibrary();
            return devices;
        }
        
        if (deviceCount == 0) {
            OSCEAN_LOG_DEBUG("CUDADeviceDetector", "No CUDA devices found");
            unloadCUDALibrary();
            return devices;
        }
        
        OSCEAN_LOG_INFO("CUDADeviceDetector", "Found " + std::to_string(deviceCount) + " CUDA device(s)");
        
        // 获取驱动版本
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        int driverMajor = driverVersion / 1000;
        int driverMinor = (driverVersion % 1000) / 10;
        
        // 获取运行时版本
        int runtimeVersion = 0;
        cudaRuntimeGetVersion(&runtimeVersion);
        int runtimeMajor = runtimeVersion / 1000;
        int runtimeMinor = (runtimeVersion % 1000) / 10;
        
        OSCEAN_LOG_INFO("CUDADeviceDetector", "CUDA Driver Version: " + std::to_string(driverMajor) + "." + std::to_string(driverMinor));
        OSCEAN_LOG_INFO("CUDADeviceDetector", "CUDA Runtime Version: " + std::to_string(runtimeMajor) + "." + std::to_string(runtimeMinor));
        
        // 枚举所有CUDA设备
        for (int i = 0; i < deviceCount; ++i) {
            OSCEAN_LOG_DEBUG("CUDADeviceDetector", "Getting properties for device " + std::to_string(i));
            
            CudaDeviceProp prop;
            memset(&prop, 0, sizeof(prop));  // 初始化结构体
            
            // 尝试使用较小的结构体大小，兼容旧版本
            size_t propSize = sizeof(prop);
            if (propSize > 840) {  // CUDA 9.0的大小约为840字节
                propSize = 840;
            }
            
            result = cudaGetDeviceProperties(&prop, i);
            
            if (result != CUDA_SUCCESS) {
                OSCEAN_LOG_WARN("CUDADeviceDetector", "Failed to get properties for device " + std::to_string(i));
                continue;
            }
            
            OSCEAN_LOG_DEBUG("CUDADeviceDetector", "Device " + std::to_string(i) + " name: " + std::string(prop.name));
            
            GPUDeviceInfo device;
            device.deviceId = i;
            device.name = prop.name;
            device.vendor = GPUVendor::NVIDIA;
            device.supportedAPIs.push_back(ComputeAPI::CUDA);
            device.supportedAPIs.push_back(ComputeAPI::OPENCL); // NVIDIA也支持OpenCL
            device.driverVersion = "CUDA " + std::to_string(driverMajor) + "." + std::to_string(driverMinor);
            
            // PCIe信息
            std::stringstream pcieSS;
            pcieSS << std::hex << std::setfill('0');
            pcieSS << std::setw(4) << prop.pciDomainID << ":"
                   << std::setw(2) << prop.pciBusID << ":"
                   << std::setw(2) << prop.pciDeviceID << ".0";
            device.pcieBusId = pcieSS.str();
            
            // 架构信息
            device.architecture = getArchitectureInfo(prop.major, prop.minor);
            
            // 时钟信息
            device.clockInfo.baseClock = prop.clockRate / 1000; // KHz to MHz
            device.clockInfo.boostClock = prop.clockRate / 1000; // CUDA不区分base和boost
            device.clockInfo.memoryClock = prop.memoryClockRate / 1000;
            device.clockInfo.currentCoreClock = prop.clockRate / 1000;
            device.clockInfo.currentMemoryClock = prop.memoryClockRate / 1000;
            
            // 计算单元信息
            device.computeUnits.multiprocessorCount = prop.multiProcessorCount;
            device.computeUnits.coresPerMP = calculateCoresPerSM(prop.major, prop.minor);
            device.computeUnits.totalCores = device.computeUnits.multiprocessorCount * 
                                           device.computeUnits.coresPerMP;
            
            // Tensor核心和RT核心（根据架构推测）
            if (prop.major >= 7) { // Volta及以上
                device.computeUnits.tensorCores = device.computeUnits.multiprocessorCount * 8;
                device.capabilities.supportsTensorCores = true;
            }
            if (prop.major >= 7 && prop.minor >= 5) { // Turing及以上
                device.computeUnits.rtCores = device.computeUnits.multiprocessorCount;
                device.capabilities.supportsRayTracing = true;
            }
            
            // 内存信息
            device.memoryDetails.totalGlobalMemory = prop.totalGlobalMem;
            device.memoryDetails.l2CacheSize = prop.l2CacheSize;
            device.memoryDetails.sharedMemoryPerBlock = prop.sharedMemPerBlock;
            device.memoryDetails.constantMemory = prop.totalConstMem;
            device.memoryDetails.memoryBusWidth = prop.memoryBusWidth;
            
            // 计算内存带宽 (GB/s)
            device.memoryDetails.memoryBandwidth = 
                (prop.memoryClockRate * 1000.0 * prop.memoryBusWidth * 2) / (8.0 * 1e9);
            
            // 获取当前空闲内存
            result = cudaSetDevice(i);
            if (result == CUDA_SUCCESS) {
                size_t freeMem = 0, totalMem = 0;
                if (cudaMemGetInfo(&freeMem, &totalMem) == CUDA_SUCCESS) {
                    device.memoryDetails.freeGlobalMemory = freeMem;
                } else {
                    device.memoryDetails.freeGlobalMemory = device.memoryDetails.totalGlobalMemory;
                }
            } else {
                // 设置设备失败，使用默认值
                device.memoryDetails.freeGlobalMemory = device.memoryDetails.totalGlobalMemory;
            }
            
            // 执行限制
            device.executionLimits.maxThreadsPerBlock = GPUDimension(
                prop.maxThreadsPerBlock, 1, 1);
            device.executionLimits.maxBlockDimension = GPUDimension(
                prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            device.executionLimits.maxGridDimension = GPUDimension(
                prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            device.executionLimits.maxRegistersPerBlock = prop.regsPerBlock;
            device.executionLimits.warpSize = prop.warpSize;
            device.executionLimits.maxWarpsPerMP = prop.maxThreadsPerMultiProcessor / prop.warpSize;
            
            // 能力标志
            device.capabilities.supportsDoublePrecision = (prop.major >= 2);
            device.capabilities.supportsAtomics = true;
            device.capabilities.supportsSharedMemory = true;
            device.capabilities.supportsTextureMemory = true;
            device.capabilities.supportsUnifiedMemory = prop.managedMemory;
            device.capabilities.supportsConcurrentKernels = prop.concurrentKernels;
            device.capabilities.supportsAsyncTransfer = prop.asyncEngineCount > 0;
            device.capabilities.supportsDynamicParallelism = (prop.major >= 3 && prop.minor >= 5);
            
            // 温度和功耗信息（CUDA API不直接提供，使用默认值）
            device.thermalInfo.currentTemp = 0.0f;
            device.thermalInfo.maxTemp = 85.0f;
            device.thermalInfo.throttleTemp = 80.0f;
            device.thermalInfo.fanSpeed = 0;
            
            device.powerInfo.currentPower = 0.0f;
            device.powerInfo.maxPower = 0.0f;
            device.powerInfo.powerLimit = 0.0f;
            device.powerInfo.powerEfficiency = 0.0f;
            
            // 性能评分将由UnifiedGPUManager计算
            device.performanceScore = 0;
            
            // 扩展信息
            device.extendedInfo.push_back({"Compute Mode", 
                prop.computeMode == 0 ? "Default" : 
                prop.computeMode == 1 ? "Exclusive Thread" :
                prop.computeMode == 2 ? "Prohibited" : "Exclusive Process"});
            device.extendedInfo.push_back({"ECC Enabled", prop.ECCEnabled ? "Yes" : "No"});
            device.extendedInfo.push_back({"TCC Driver", prop.tccDriver ? "Yes" : "No"});
            device.extendedInfo.push_back({"Multi-GPU Board", prop.isMultiGpuBoard ? "Yes" : "No"});
            
            devices.push_back(device);
        }
        
        unloadCUDALibrary();
        return devices;
    }
};

// 检测NVIDIA GPU的实现函数
std::vector<GPUDeviceInfo> detectNVIDIAGPUsImpl() {
    CUDADetector detector;
    return detector.detect();
}

} // namespace oscean::common_utils::gpu 