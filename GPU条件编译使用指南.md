# OSCEAN项目GPU条件编译使用指南

## 概述

本指南说明如何在OSCEAN项目中使用GPU条件编译系统，使项目能够在有GPU和无GPU的环境下都能正常编译和运行。

## 1. CMake配置

### 1.1 自动GPU检测

项目会自动检测以下GPU平台：
- NVIDIA CUDA
- OpenCL
- AMD ROCm/HIP
- Intel oneAPI/DPC++

在配置CMake时，系统会自动检测可用的GPU支持：

```bash
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="C:/Users/Administrator/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

### 1.2 强制CPU模式

如果需要在有GPU的机器上强制使用CPU模式，可以使用：

```bash
cmake .. -DOSCEAN_FORCE_CPU_ONLY=ON
```

### 1.3 查看GPU配置状态

配置完成后，CMake会输出GPU检测结果：

```
========== GPU Support Detection ==========
✅ CUDA Toolkit found: 12.3
  CUDA Compiler: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe
  CUDA Architectures: 75;80;86;89
❌ OpenCL not available
❌ AMD ROCm/HIP not available
❌ Intel oneAPI/DPC++ not available
✅ GPU support available
=========================================

GPU Configuration Summary:
  Any GPU Available: TRUE
  CUDA Enabled: TRUE
    Version: 12.3
    Architectures: 75;80;86;89
  OpenCL Enabled: FALSE
  ROCm Enabled: FALSE
  oneAPI Enabled: FALSE
```

## 2. 在代码中使用条件编译

### 2.1 包含GPU配置头文件

```cpp
#include <common_utils/gpu/gpu_config.h>
```

### 2.2 基本条件编译模式

#### 模式1：简单的GPU/CPU分支

```cpp
void processData(float* data, size_t size) {
    OSCEAN_GPU_CODE(
        // GPU实现
        launchGPUKernel(data, size);
    ,
        // CPU后备实现
        processCPU(data, size);
    )
}
```

#### 模式2：平台特定实现

```cpp
void interpolate(const GridData& input, GridData& output) {
    #if OSCEAN_CUDA_ENABLED
        interpolateCUDA(input, output);
    #elif OSCEAN_OPENCL_ENABLED
        interpolateOpenCL(input, output);
    #else
        interpolateCPU(input, output);
    #endif
}
```

#### 模式3：条件成员变量

```cpp
class InterpolationEngine {
private:
    // 仅在GPU可用时包含的成员
    OSCEAN_GPU_MEMBER(void* d_deviceMemory;)
    OSCEAN_GPU_MEMBER(cudaStream_t stream;)
    
    // 总是包含的成员
    size_t dataSize;
    float* hostData;
};
```

### 2.3 运行时GPU检测

```cpp
#include <common_utils/gpu/gpu_config.h>

void initializeEngine() {
    auto backend = oscean::gpu::getPreferredBackend();
    
    if (backend != oscean::gpu::GpuBackend::CPU_Fallback) {
        // 检查是否真的有GPU设备（处理驱动问题）
        if (oscean::gpu::isGpuRuntimeAvailable()) {
            std::cout << "Using GPU backend: " 
                      << oscean::gpu::getBackendName(backend) 
                      << std::endl;
            std::cout << "GPU device count: " 
                      << oscean::gpu::getDeviceCount() 
                      << std::endl;
        } else {
            std::cout << "GPU backend available but no devices found" << std::endl;
            std::cout << "Falling back to CPU" << std::endl;
        }
    } else {
        std::cout << "Using CPU backend" << std::endl;
    }
}
```

## 3. 在CMakeLists.txt中使用

### 3.1 为目标应用GPU配置

```cmake
# 创建库或可执行文件
add_library(my_module ${SOURCES})

# 应用GPU配置（自动添加编译定义和链接库）
oscean_configure_gpu_target(my_module)
```

### 3.2 条件添加GPU源文件

```cmake
# 使用提供的函数条件添加源文件
oscean_add_gpu_sources(my_module
    CUDA_SOURCES
        src/cuda/kernel1.cu
        src/cuda/kernel2.cu
    OPENCL_SOURCES
        src/opencl/kernel1.cl
        src/opencl/kernel2.cl
    CPU_FALLBACK_SOURCES
        src/cpu/fallback1.cpp
        src/cpu/fallback2.cpp
)
```

### 3.3 手动条件编译

```cmake
if(OSCEAN_GPU_AVAILABLE)
    target_sources(my_module PRIVATE
        src/gpu/gpu_manager.cpp
        src/gpu/gpu_memory.cpp
    )
else()
    target_sources(my_module PRIVATE
        src/cpu/cpu_only_impl.cpp
    )
endif()
```

## 4. 预定义的宏

| 宏名称 | 值 | 说明 |
|--------|-----|------|
| `OSCEAN_GPU_AVAILABLE` | 0或1 | 是否有任何GPU支持 |
| `OSCEAN_CUDA_ENABLED` | 0或1 | 是否启用CUDA |
| `OSCEAN_OPENCL_ENABLED` | 0或1 | 是否启用OpenCL |
| `OSCEAN_ROCM_ENABLED` | 0或1 | 是否启用ROCm |
| `OSCEAN_ONEAPI_ENABLED` | 0或1 | 是否启用Intel oneAPI |
| `OSCEAN_CUDA_VERSION_MAJOR` | 数字 | CUDA主版本号 |
| `OSCEAN_CUDA_VERSION_MINOR` | 数字 | CUDA次版本号 |

## 5. 最佳实践

### 5.1 总是提供CPU后备实现

```cpp
class DataProcessor {
public:
    void process(const Data& input, Data& output) {
        #if OSCEAN_GPU_AVAILABLE
            if (useGPU && oscean::gpu::isGpuRuntimeAvailable()) {
                processGPU(input, output);
            } else {
                processCPU(input, output);
            }
        #else
            processCPU(input, output);
        #endif
    }
    
private:
    void processCPU(const Data& input, Data& output);
    
    #if OSCEAN_GPU_AVAILABLE
    void processGPU(const Data& input, Data& output);
    #endif
};
```

### 5.2 避免GPU特定头文件泄露

```cpp
// gpu_impl.h - 仅在实现文件中包含
#pragma once

#if OSCEAN_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// 使用前向声明而不是包含GPU头文件
class GPUImpl {
    struct Impl;  // 隐藏GPU特定类型
    std::unique_ptr<Impl> pImpl;
public:
    GPUImpl();
    ~GPUImpl();
    void compute(float* data, size_t size);
};
```

### 5.3 错误处理

```cpp
bool initializeGPU() {
    #if OSCEAN_GPU_AVAILABLE
        if (!oscean::gpu::isGpuRuntimeAvailable()) {
            OSCEAN_LOG_WARNING("GPU support compiled but no devices found");
            return false;
        }
        
        try {
            // 初始化GPU资源
            return true;
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("GPU initialization failed: {}", e.what());
            return false;
        }
    #else
        OSCEAN_LOG_INFO("GPU support not compiled");
        return false;
    #endif
}
```

## 6. 故障排除

### 6.1 编译错误：未定义的宏

确保：
1. 包含了 `gpu_detection.cmake`
2. 调用了 `oscean_detect_gpu_support()`
3. 对目标调用了 `oscean_configure_gpu_target()`

### 6.2 运行时找不到GPU

可能的原因：
- GPU驱动未安装或版本过旧
- CUDA/OpenCL运行时未安装
- 权限问题（特别是在Linux上）

### 6.3 链接错误

确保所有GPU相关的库都通过vcpkg安装：
```bash
vcpkg install cuda:x64-windows
vcpkg install opencl:x64-windows
```

## 7. 示例项目

参考以下模块的实现：
- `output_generation` - GPU加速的图像生成
- `core_services_impl/interpolation_service` - GPU加速的插值计算
- `common_utilities` - GPU管理基础设施

## 8. 性能建议

1. **批处理**: GPU适合大批量数据处理，小数据量使用CPU可能更快
2. **内存传输**: 最小化CPU-GPU内存传输
3. **异步执行**: 使用流（streams）实现CPU-GPU并行
4. **自动选择**: 根据数据量和可用资源自动选择CPU或GPU

```cpp
bool shouldUseGPU(size_t dataSize) {
    const size_t GPU_THRESHOLD = 1024 * 1024;  // 1MB
    return OSCEAN_GPU_AVAILABLE && 
           oscean::gpu::isGpuRuntimeAvailable() &&
           dataSize >= GPU_THRESHOLD;
}
```

## 总结

OSCEAN的GPU条件编译系统提供了灵活的方式来支持GPU加速，同时确保在没有GPU的环境下也能正常工作。通过合理使用这些工具和模式，可以构建高性能且具有良好兼容性的海洋科学计算应用。 