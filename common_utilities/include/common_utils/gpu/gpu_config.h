#pragma once

/**
 * @file gpu_config.h
 * @brief GPU Configuration and Conditional Compilation Header
 * 
 * This header provides macros and utilities for conditional compilation
 * based on GPU availability. It ensures the project can compile and run
 * even when no GPU is available.
 */

// =============================================================================
// GPU Platform Detection Macros
// =============================================================================

// These macros are defined by CMake based on GPU detection results
// OSCEAN_GPU_AVAILABLE    - 1 if any GPU support is available, 0 otherwise
// OSCEAN_CUDA_ENABLED     - 1 if CUDA is available, 0 otherwise
// OSCEAN_OPENCL_ENABLED   - 1 if OpenCL is available, 0 otherwise
// OSCEAN_ROCM_ENABLED     - 1 if ROCm is available, 0 otherwise
// OSCEAN_ONEAPI_ENABLED   - 1 if Intel oneAPI is available, 0 otherwise

// =============================================================================
// Convenience Macros for Conditional Compilation
// =============================================================================

// Use these macros to conditionally compile GPU-specific code
#if OSCEAN_GPU_AVAILABLE
    #define OSCEAN_HAS_GPU 1
#else
    #define OSCEAN_HAS_GPU 0
#endif

#if OSCEAN_CUDA_ENABLED
    #define OSCEAN_HAS_CUDA 1
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
#else
    #define OSCEAN_HAS_CUDA 0
#endif

#if OSCEAN_OPENCL_ENABLED
    #define OSCEAN_HAS_OPENCL 1
    #ifdef __APPLE__
        #include <OpenCL/opencl.h>
    #else
        #include <CL/cl.h>
    #endif
#else
    #define OSCEAN_HAS_OPENCL 0
#endif

// =============================================================================
// GPU Selection Priority
// =============================================================================

namespace oscean {
namespace gpu {

enum class GpuBackend {
    None = 0,
    CUDA = 1,
    OpenCL = 2,
    ROCm = 3,
    OneAPI = 4,
    CPU_Fallback = 5
};

/**
 * @brief Get the preferred GPU backend based on availability
 */
inline GpuBackend getPreferredBackend() {
#if OSCEAN_CUDA_ENABLED
    return GpuBackend::CUDA;
#elif OSCEAN_OPENCL_ENABLED
    return GpuBackend::OpenCL;
#elif OSCEAN_ROCM_ENABLED
    return GpuBackend::ROCm;
#elif OSCEAN_ONEAPI_ENABLED
    return GpuBackend::OneAPI;
#else
    return GpuBackend::CPU_Fallback;
#endif
}

/**
 * @brief Check if any GPU backend is available
 */
inline bool isGpuAvailable() {
    return OSCEAN_GPU_AVAILABLE != 0;
}

/**
 * @brief Get a string representation of the GPU backend
 */
inline const char* getBackendName(GpuBackend backend) {
    switch (backend) {
        case GpuBackend::CUDA: return "CUDA";
        case GpuBackend::OpenCL: return "OpenCL";
        case GpuBackend::ROCm: return "ROCm";
        case GpuBackend::OneAPI: return "Intel oneAPI";
        case GpuBackend::CPU_Fallback: return "CPU";
        default: return "None";
    }
}

} // namespace gpu
} // namespace oscean

// =============================================================================
// Conditional Compilation Helpers
// =============================================================================

/**
 * @brief Macro for GPU-specific code blocks
 * 
 * Usage:
 * OSCEAN_GPU_CODE(
 *     // GPU-specific code here
 *     gpuFunction();
 * ,
 *     // CPU fallback code here
 *     cpuFunction();
 * )
 */
#if OSCEAN_GPU_AVAILABLE
    #define OSCEAN_GPU_CODE(gpu_code, cpu_code) gpu_code
#else
    #define OSCEAN_GPU_CODE(gpu_code, cpu_code) cpu_code
#endif

/**
 * @brief Macro for CUDA-specific code blocks
 */
#if OSCEAN_CUDA_ENABLED
    #define OSCEAN_CUDA_CODE(cuda_code, fallback_code) cuda_code
#else
    #define OSCEAN_CUDA_CODE(cuda_code, fallback_code) fallback_code
#endif

/**
 * @brief Macro for OpenCL-specific code blocks
 */
#if OSCEAN_OPENCL_ENABLED
    #define OSCEAN_OPENCL_CODE(opencl_code, fallback_code) opencl_code
#else
    #define OSCEAN_OPENCL_CODE(opencl_code, fallback_code) fallback_code
#endif

// =============================================================================
// GPU Function Availability Macros
// =============================================================================

/**
 * @brief Declare a function that has both GPU and CPU implementations
 * 
 * Usage:
 * OSCEAN_GPU_FUNCTION(void processData(float* data, size_t size));
 */
#if OSCEAN_GPU_AVAILABLE
    #define OSCEAN_GPU_FUNCTION(func) func
#else
    #define OSCEAN_GPU_FUNCTION(func) func
#endif

/**
 * @brief Mark a class member as GPU-only (excluded in CPU-only builds)
 */
#if OSCEAN_GPU_AVAILABLE
    #define OSCEAN_GPU_MEMBER(member) member
#else
    #define OSCEAN_GPU_MEMBER(member)
#endif

// =============================================================================
// Runtime GPU Detection
// =============================================================================

namespace oscean {
namespace gpu {

/**
 * @brief Runtime GPU device information
 */
struct GpuDeviceInfo {
    std::string name;
    size_t totalMemory;
    size_t availableMemory;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    GpuBackend backend;
};

/**
 * @brief Get GPU device count at runtime
 */
inline int getDeviceCount() {
#if OSCEAN_CUDA_ENABLED
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) ? count : 0;
#elif OSCEAN_OPENCL_ENABLED
    // OpenCL device enumeration
    cl_uint count = 0;
    cl_platform_id platform;
    if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
    }
    return static_cast<int>(count);
#else
    return 0;
#endif
}

/**
 * @brief Check if GPU is available at runtime (handles driver issues)
 */
inline bool isGpuRuntimeAvailable() {
    return getDeviceCount() > 0;
}

} // namespace gpu
} // namespace oscean

// =============================================================================
// Example Usage Patterns
// =============================================================================

/*
// Example 1: Conditional class members
class InterpolationEngine {
public:
    void interpolate(const float* input, float* output, size_t size) {
        OSCEAN_GPU_CODE(
            // GPU implementation
            interpolateGPU(input, output, size);
        ,
            // CPU fallback
            interpolateCPU(input, output, size);
        )
    }

private:
    OSCEAN_GPU_MEMBER(void* d_workspace;)  // Only included if GPU available
    
    void interpolateCPU(const float* input, float* output, size_t size);
    OSCEAN_GPU_MEMBER(void interpolateGPU(const float* input, float* output, size_t size);)
};

// Example 2: Platform-specific implementations
void processOceanData(OceanGrid& grid) {
    #if OSCEAN_CUDA_ENABLED
        processCUDA(grid);
    #elif OSCEAN_OPENCL_ENABLED
        processOpenCL(grid);
    #else
        processCPU(grid);
    #endif
}

// Example 3: Runtime selection
auto backend = oscean::gpu::getPreferredBackend();
if (backend != oscean::gpu::GpuBackend::CPU_Fallback && 
    oscean::gpu::isGpuRuntimeAvailable()) {
    std::cout << "Using GPU backend: " << oscean::gpu::getBackendName(backend) << std::endl;
} else {
    std::cout << "Using CPU fallback" << std::endl;
}
*/

#endif // OSCEAN_GPU_CONFIG_H 