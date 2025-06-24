/**
 * @file rocm_utils.h
 * @brief ROCm/HIP工具函数占位符
 */

#pragma once

#include <string>

namespace oscean::common_utils::gpu::rocm {

// ROCm/HIP API占位符
// 实际实现需要ROCm SDK

struct hipDeviceProp_t {
    char name[256];
    size_t totalGlobalMem;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int clockRate;
    int major;
    int minor;
};

enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidDevice = 1,
    hipErrorNoDevice = 2
};

// 占位符函数
inline hipError_t hipGetDeviceCount(int* count) {
    *count = 0;
    return hipErrorNoDevice;
}

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device) {
    return hipErrorInvalidDevice;
}

inline const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess: return "Success";
        case hipErrorInvalidDevice: return "Invalid device";
        case hipErrorNoDevice: return "No device";
        default: return "Unknown error";
    }
}

} // namespace oscean::common_utils::gpu::rocm 