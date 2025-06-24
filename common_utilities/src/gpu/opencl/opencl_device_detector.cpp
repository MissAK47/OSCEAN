/**
 * @file opencl_device_detector.cpp
 * @brief OpenCL设备检测实现
 */

#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <boost/log/trivial.hpp>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

#include <cstring>

namespace oscean::common_utils::gpu {

// OpenCL类型定义
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef size_t cl_size_t;
typedef int cl_int;
typedef unsigned int cl_bitfield;
typedef char cl_char;

// OpenCL常量
#define CL_SUCCESS                          0
#define CL_DEVICE_NOT_FOUND                -1
#define CL_DEVICE_NOT_AVAILABLE            -2
#define CL_PLATFORM_PROFILE                 0x0900
#define CL_PLATFORM_VERSION                 0x0901
#define CL_PLATFORM_NAME                    0x0902
#define CL_PLATFORM_VENDOR                  0x0903
#define CL_PLATFORM_EXTENSIONS              0x0904
#define CL_DEVICE_TYPE                      0x1000
#define CL_DEVICE_VENDOR_ID                 0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS         0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS  0x1003
#define CL_DEVICE_MAX_WORK_GROUP_SIZE       0x1004
#define CL_DEVICE_MAX_WORK_ITEM_SIZES       0x1005
#define CL_DEVICE_MAX_CLOCK_FREQUENCY       0x100C
#define CL_DEVICE_GLOBAL_MEM_SIZE          0x101F
#define CL_DEVICE_ERROR_CORRECTION_SUPPORT  0x1024
#define CL_DEVICE_LOCAL_MEM_SIZE           0x1023
#define CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE  0x1020
#define CL_DEVICE_QUEUE_PROPERTIES         0x102A
#define CL_DEVICE_NAME                     0x102B
#define CL_DEVICE_VENDOR                   0x102C
#define CL_DRIVER_VERSION                  0x102D
#define CL_DEVICE_PROFILE                  0x102E
#define CL_DEVICE_VERSION                  0x102F
#define CL_DEVICE_EXTENSIONS               0x1030
#define CL_DEVICE_TYPE_GPU                 (1 << 2)
#define CL_DEVICE_TYPE_CPU                 (1 << 1)
#define CL_DEVICE_TYPE_ACCELERATOR         (1 << 3)
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE    0x101E
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE       0x1010
#define CL_DEVICE_IMAGE_SUPPORT            0x1016
#define CL_DEVICE_MAX_SAMPLERS             0x1018
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT 0x103A
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT 0x1024
#define CL_DEVICE_DOUBLE_FP_CONFIG         0x1032
#define CL_DEVICE_SINGLE_FP_CONFIG         0x101B
#define CL_DEVICE_EXECUTION_CAPABILITIES   0x1029
#define CL_DEVICE_QUEUE_ON_HOST_PROPERTIES 0x102A
#define CL_DEVICE_BUILT_IN_KERNELS         0x103F
#define CL_DEVICE_PLATFORM                 0x1031
#define CL_DEVICE_MAX_PARAMETER_SIZE       0x1017
#define CL_DEVICE_MEM_BASE_ADDR_ALIGN      0x1019
#define CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE 0x101A
#define CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE 0x101D
#define CL_DEVICE_MAX_CONSTANT_ARGS        0x1021
#define CL_DEVICE_LOCAL_MEM_TYPE           0x1022
#define CL_DEVICE_ENDIAN_LITTLE            0x1026
#define CL_DEVICE_AVAILABLE                0x1027
#define CL_DEVICE_COMPILER_AVAILABLE       0x1028

// OpenCL 2.0+
#define CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE 0x104D
#define CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE 0x1050

// OpenCL函数指针类型
typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*clGetPlatformInfo_t)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*clGetDeviceInfo_t)(cl_device_id, cl_uint, size_t, void*, size_t*);

class OpenCLDetector {
private:
    void* openclLibrary = nullptr;
    
    // OpenCL函数指针
    clGetPlatformIDs_t clGetPlatformIDs = nullptr;
    clGetPlatformInfo_t clGetPlatformInfo = nullptr;
    clGetDeviceIDs_t clGetDeviceIDs = nullptr;
    clGetDeviceInfo_t clGetDeviceInfo = nullptr;
    
    bool loadOpenCLLibrary() {
        #ifdef _WIN32
            // Windows
            const char* libraryNames[] = {
                "OpenCL.dll",
                "opencl.dll"
            };
            
            for (const auto& libName : libraryNames) {
                openclLibrary = LoadLibraryA(libName);
                if (openclLibrary) {
                    OSCEAN_LOG_INFO("OpenCLDetector", std::string("Loaded OpenCL library: ") + libName);
                    break;
                }
            }
        #else
            // Linux/macOS
            const char* libraryNames[] = {
                "libOpenCL.so.1",
                "libOpenCL.so",
                "libOpenCL.dylib",
                "/System/Library/Frameworks/OpenCL.framework/OpenCL"
            };
            
            for (const auto& libName : libraryNames) {
                openclLibrary = dlopen(libName, RTLD_LAZY);
                if (openclLibrary) {
                    OSCEAN_LOG_INFO("OpenCLDetector", std::string("Loaded OpenCL library: ") + libName);
                    break;
                }
            }
        #endif
        
        if (!openclLibrary) {
            OSCEAN_LOG_DEBUG("OpenCLDetector", "OpenCL library not found");
            return false;
        }
        
        // 加载函数指针
        #ifdef _WIN32
            #define GET_PROC(name) name = (name##_t)GetProcAddress((HMODULE)openclLibrary, #name)
        #else
            #define GET_PROC(name) name = (name##_t)dlsym(openclLibrary, #name)
        #endif
        
        GET_PROC(clGetPlatformIDs);
        GET_PROC(clGetPlatformInfo);
        GET_PROC(clGetDeviceIDs);
        GET_PROC(clGetDeviceInfo);
        
        #undef GET_PROC
        
        // 验证所有必需的函数都已加载
        if (!clGetPlatformIDs || !clGetPlatformInfo || 
            !clGetDeviceIDs || !clGetDeviceInfo) {
            OSCEAN_LOG_ERROR("OpenCLDetector", "Failed to load required OpenCL functions");
            return false;
        }
        
        return true;
    }
    
    void unloadOpenCLLibrary() {
        if (openclLibrary) {
            #ifdef _WIN32
                FreeLibrary((HMODULE)openclLibrary);
            #else
                dlclose(openclLibrary);
            #endif
            openclLibrary = nullptr;
        }
    }
    
    template<typename T>
    bool getDeviceInfo(cl_device_id device, cl_uint param, T& value) {
        return clGetDeviceInfo(device, param, sizeof(T), &value, nullptr) == CL_SUCCESS;
    }
    
    std::string getDeviceString(cl_device_id device, cl_uint param) {
        size_t size = 0;
        if (clGetDeviceInfo(device, param, 0, nullptr, &size) != CL_SUCCESS) {
            return "";
        }
        
        std::vector<char> buffer(size);
        if (clGetDeviceInfo(device, param, size, buffer.data(), nullptr) != CL_SUCCESS) {
            return "";
        }
        
        return std::string(buffer.data());
    }
    
    GPUVendor detectVendor(const std::string& vendorName) {
        std::string vendor = vendorName;
        // 转换为小写进行比较
        std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);
        
        if (vendor.find("nvidia") != std::string::npos) {
            return GPUVendor::NVIDIA;
        } else if (vendor.find("amd") != std::string::npos || 
                   vendor.find("advanced micro devices") != std::string::npos) {
            return GPUVendor::AMD;
        } else if (vendor.find("intel") != std::string::npos) {
            return GPUVendor::INTEL;
        } else if (vendor.find("apple") != std::string::npos) {
            return GPUVendor::APPLE;
        }
        
        return GPUVendor::UNKNOWN;
    }
    
    GPUArchitecture parseArchitecture(const std::string& deviceName, GPUVendor vendor) {
        GPUArchitecture arch;
        arch.name = "Unknown";
        arch.majorVersion = 0;
        arch.minorVersion = 0;
        arch.computeCapability = 0;
        
        // 根据设备名称推测架构
        if (vendor == GPUVendor::NVIDIA) {
            if (deviceName.find("RTX 40") != std::string::npos) {
                arch.name = "Ada Lovelace";
                arch.majorVersion = 8;
                arch.minorVersion = 9;
            } else if (deviceName.find("RTX 30") != std::string::npos) {
                arch.name = "Ampere";
                arch.majorVersion = 8;
                arch.minorVersion = 6;
            } else if (deviceName.find("RTX 20") != std::string::npos || 
                       deviceName.find("GTX 16") != std::string::npos) {
                arch.name = "Turing";
                arch.majorVersion = 7;
                arch.minorVersion = 5;
            }
        } else if (vendor == GPUVendor::AMD) {
            if (deviceName.find("RX 7") != std::string::npos) {
                arch.name = "RDNA 3";
                arch.majorVersion = 11;
                arch.minorVersion = 0;
            } else if (deviceName.find("RX 6") != std::string::npos) {
                arch.name = "RDNA 2";
                arch.majorVersion = 10;
                arch.minorVersion = 3;
            }
        } else if (vendor == GPUVendor::INTEL) {
            if (deviceName.find("Arc A7") != std::string::npos) {
                arch.name = "Alchemist";
                arch.majorVersion = 12;
                arch.minorVersion = 7;
            }
        }
        
        arch.computeCapability = arch.majorVersion * 10 + arch.minorVersion;
        return arch;
    }
    
public:
    std::vector<GPUDeviceInfo> detect() {
        std::vector<GPUDeviceInfo> devices;
        
        if (!loadOpenCLLibrary()) {
            return devices;
        }
        
        // 获取平台数量
        cl_uint platformCount = 0;
        if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0) {
            OSCEAN_LOG_DEBUG("OpenCLDetector", "No OpenCL platforms found");
            unloadOpenCLLibrary();
            return devices;
        }
        
        // 获取所有平台
        std::vector<cl_platform_id> platforms(platformCount);
        if (clGetPlatformIDs(platformCount, platforms.data(), nullptr) != CL_SUCCESS) {
            OSCEAN_LOG_ERROR("OpenCLDetector", "Failed to get OpenCL platforms");
            unloadOpenCLLibrary();
            return devices;
        }
        
        int globalDeviceId = 0;
        
        // 遍历每个平台
        for (cl_uint p = 0; p < platformCount; ++p) {
            // 获取平台信息
            char platformName[256] = {0};
            char platformVendor[256] = {0};
            char platformVersion[256] = {0};
            
            clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
            clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, nullptr);
            clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, sizeof(platformVersion), platformVersion, nullptr);
            
            OSCEAN_LOG_INFO("OpenCLDetector", std::string("OpenCL Platform ") + std::to_string(p) + ": " + platformName);
            
            // 获取GPU设备数量
            cl_uint deviceCount = 0;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) != CL_SUCCESS || 
                deviceCount == 0) {
                continue;
            }
            
            // 获取所有GPU设备
            std::vector<cl_device_id> clDevices(deviceCount);
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, deviceCount, clDevices.data(), nullptr) != CL_SUCCESS) {
                continue;
            }
            
            // 处理每个设备
            for (cl_uint d = 0; d < deviceCount; ++d) {
                OSCEAN_LOG_DEBUG("OpenCLDetector", "Processing device " + std::to_string(d) + " on platform " + std::to_string(p));
                
                GPUDeviceInfo device;
                device.deviceId = globalDeviceId++;
                
                // 基本信息
                device.name = getDeviceString(clDevices[d], CL_DEVICE_NAME);
                if (device.name.empty()) {
                    OSCEAN_LOG_WARN("OpenCLDetector", "Failed to get device name, skipping device");
                    continue;
                }
                
                std::string vendorName = getDeviceString(clDevices[d], CL_DEVICE_VENDOR);
                device.vendor = detectVendor(vendorName);
                
                // 暂时跳过Intel集成显卡，避免检测问题
                if (device.vendor == GPUVendor::INTEL && device.name.find("Graphics") != std::string::npos) {
                    OSCEAN_LOG_INFO("OpenCLDetector", "Skipping Intel integrated graphics: " + device.name);
                    continue;
                }
                
                device.supportedAPIs.push_back(ComputeAPI::OPENCL);
                
                // 如果是NVIDIA设备，也支持CUDA
                if (device.vendor == GPUVendor::NVIDIA) {
                    device.supportedAPIs.push_back(ComputeAPI::CUDA);
                }
                
                // 驱动版本
                device.driverVersion = getDeviceString(clDevices[d], CL_DRIVER_VERSION);
                
                // 架构信息
                device.architecture = parseArchitecture(device.name, device.vendor);
                
                // 计算单元
                cl_uint computeUnits = 0;
                getDeviceInfo(clDevices[d], CL_DEVICE_MAX_COMPUTE_UNITS, computeUnits);
                device.computeUnits.multiprocessorCount = computeUnits;
                
                // 根据厂商估算核心数
                if (device.vendor == GPUVendor::NVIDIA) {
                    device.computeUnits.coresPerMP = 128; // 估算值
                } else if (device.vendor == GPUVendor::AMD) {
                    device.computeUnits.coresPerMP = 64;  // 估算值
                } else {
                    device.computeUnits.coresPerMP = 8;   // 默认值
                }
                device.computeUnits.totalCores = computeUnits * device.computeUnits.coresPerMP;
                
                // 时钟频率
                cl_uint maxFreq = 0;
                getDeviceInfo(clDevices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, maxFreq);
                device.clockInfo.baseClock = maxFreq;
                device.clockInfo.boostClock = maxFreq;
                device.clockInfo.currentCoreClock = maxFreq;
                
                // 内存信息
                cl_ulong globalMem = 0, localMem = 0, constantMem = 0;
                cl_ulong cacheSize = 0, maxAlloc = 0;
                cl_uint cacheLineSize = 0, memAlign = 0;
                
                getDeviceInfo(clDevices[d], CL_DEVICE_GLOBAL_MEM_SIZE, globalMem);
                getDeviceInfo(clDevices[d], CL_DEVICE_LOCAL_MEM_SIZE, localMem);
                getDeviceInfo(clDevices[d], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, constantMem);
                getDeviceInfo(clDevices[d], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cacheSize);
                getDeviceInfo(clDevices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, maxAlloc);
                getDeviceInfo(clDevices[d], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cacheLineSize);
                getDeviceInfo(clDevices[d], CL_DEVICE_MEM_BASE_ADDR_ALIGN, memAlign);
                
                device.memoryDetails.totalGlobalMemory = globalMem;
                device.memoryDetails.freeGlobalMemory = globalMem; // OpenCL不提供空闲内存信息
                device.memoryDetails.sharedMemoryPerBlock = localMem;
                device.memoryDetails.constantMemory = constantMem;
                device.memoryDetails.l2CacheSize = cacheSize;
                
                // 执行限制
                size_t maxWorkGroupSize = 0;
                size_t maxWorkItemSizes[3] = {0};
                cl_uint maxWorkItemDims = 0;
                
                getDeviceInfo(clDevices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, maxWorkGroupSize);
                getDeviceInfo(clDevices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, maxWorkItemDims);
                clGetDeviceInfo(clDevices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, 
                               sizeof(maxWorkItemSizes), maxWorkItemSizes, nullptr);
                
                device.executionLimits.maxThreadsPerBlock = GPUDimension(maxWorkGroupSize, 1, 1);
                device.executionLimits.maxBlockDimension = GPUDimension(
                    maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);
                device.executionLimits.warpSize = 32; // 默认值，实际值依赖于硬件
                
                // 能力标志
                cl_bitfield fpConfig = 0;
                getDeviceInfo(clDevices[d], CL_DEVICE_DOUBLE_FP_CONFIG, fpConfig);
                device.capabilities.supportsDoublePrecision = (fpConfig != 0);
                
                cl_uint imageSupport = 0;
                getDeviceInfo(clDevices[d], CL_DEVICE_IMAGE_SUPPORT, imageSupport);
                device.capabilities.supportsTextureMemory = (imageSupport != 0);
                
                device.capabilities.supportsAtomics = true;  // OpenCL 1.1+都支持
                device.capabilities.supportsSharedMemory = true;
                device.capabilities.supportsAsyncTransfer = true;
                
                // 性能评分（基于计算单元和频率的简单估算）
                device.performanceScore = (computeUnits * maxFreq) / 1000;
                if (device.performanceScore > 100) device.performanceScore = 100;
                
                // 扩展信息
                std::string extensions = getDeviceString(clDevices[d], CL_DEVICE_EXTENSIONS);
                device.extendedInfo.push_back({"Platform", platformName});
                device.extendedInfo.push_back({"OpenCL Version", getDeviceString(clDevices[d], CL_DEVICE_VERSION)});
                device.extendedInfo.push_back({"Profile", getDeviceString(clDevices[d], CL_DEVICE_PROFILE)});
                
                // 检查特定扩展
                if (extensions.find("cl_khr_fp64") != std::string::npos) {
                    device.extendedInfo.push_back({"Double Precision", "Supported"});
                }
                if (extensions.find("cl_khr_fp16") != std::string::npos) {
                    device.extendedInfo.push_back({"Half Precision", "Supported"});
                }
                
                devices.push_back(device);
            }
        }
        
        unloadOpenCLLibrary();
        return devices;
    }
};

// 检测OpenCL GPU的实现函数
std::vector<GPUDeviceInfo> detectOpenCLGPUsImpl() {
    OpenCLDetector detector;
    return detector.detect();
}

} // namespace oscean::common_utils::gpu 