# GPU模块源文件
set(GPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unified_gpu_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_memory_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_performance_monitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/oscean_gpu_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_algorithm_examples.cpp
)

# CUDA特定源文件
if(CUDAToolkit_FOUND)
    list(APPEND GPU_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_device_detector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_memory_manager.cpp
    )
endif()

# OpenCL特定源文件
if(OpenCL_FOUND)
    list(APPEND GPU_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/opencl/opencl_device_detector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/opencl/opencl_memory_manager.cpp
    )
endif()

# ROCm特定源文件（占位符）
list(APPEND GPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm/rocm_device_detector.cpp
)

# 将GPU源文件添加到父作用域
set(COMMON_UTILS_GPU_SOURCES ${GPU_SOURCES} PARENT_SCOPE) 