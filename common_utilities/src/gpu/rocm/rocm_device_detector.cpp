/**
 * @file rocm_device_detector.cpp
 * @brief AMD ROCm设备检测实现（占位符）
 */

#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/utilities/logging_utils.h"

namespace oscean::common_utils::gpu {

// 检测AMD GPU的实现函数
std::vector<GPUDeviceInfo> detectAMDGPUsImpl() {
    // ROCm/HIP检测目前返回空列表
    // 实际实现需要ROCm SDK
    OSCEAN_LOG_DEBUG("UnifiedGPUManager", "ROCm detection not implemented yet");
    return std::vector<GPUDeviceInfo>();
}

} // namespace oscean::common_utils::gpu 