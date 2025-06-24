# GPU Detection and Configuration Module for OSCEAN Project
# 
# This module detects GPU capabilities and sets appropriate compile definitions
# to enable conditional compilation when GPU is not available

# =============================================================================
# GPU Detection Functions
# =============================================================================

# Function to detect CUDA capability
function(oscean_detect_cuda)
    find_package(CUDAToolkit QUIET)
    
    if(CUDAToolkit_FOUND)
        message(STATUS "✅ CUDA Toolkit found: ${CUDAToolkit_VERSION}")
        
        # Check if we can actually compile CUDA code
        include(CheckLanguage)
        check_language(CUDA)
        
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            
            # Set CUDA architectures based on available GPUs
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                # Default to common architectures
                set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")
            endif()
            
            set(OSCEAN_CUDA_AVAILABLE TRUE PARENT_SCOPE)
            set(OSCEAN_CUDA_VERSION ${CUDAToolkit_VERSION} PARENT_SCOPE)
            
            # Set include directories
            set(OSCEAN_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE)
            
            # Check for specific CUDA libraries
            if(TARGET CUDA::cudart)
                set(OSCEAN_CUDA_RUNTIME_FOUND TRUE PARENT_SCOPE)
            endif()
            
            if(TARGET CUDA::cublas)
                set(OSCEAN_CUDA_CUBLAS_FOUND TRUE PARENT_SCOPE)
            endif()
            
            if(TARGET CUDA::cufft)
                set(OSCEAN_CUDA_CUFFT_FOUND TRUE PARENT_SCOPE)
            endif()
            
            message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
            message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
            
            return()
        else()
            message(STATUS "⚠️  CUDA Toolkit found but CUDA compiler not available")
        endif()
    endif()
    
    # CUDA not available
    set(OSCEAN_CUDA_AVAILABLE FALSE PARENT_SCOPE)
    message(STATUS "❌ CUDA not available - GPU features will be disabled")
endfunction()

# Function to detect OpenCL capability
function(oscean_detect_opencl)
    find_package(OpenCL QUIET)
    
    if(OpenCL_FOUND)
        message(STATUS "✅ OpenCL found: ${OpenCL_VERSION_STRING}")
        set(OSCEAN_OPENCL_AVAILABLE TRUE PARENT_SCOPE)
        set(OSCEAN_OPENCL_VERSION ${OpenCL_VERSION_STRING} PARENT_SCOPE)
        set(OSCEAN_OPENCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} PARENT_SCOPE)
        
        # Check OpenCL version
        if(OpenCL_VERSION_STRING VERSION_GREATER_EQUAL "2.0")
            set(OSCEAN_OPENCL_2_0_AVAILABLE TRUE PARENT_SCOPE)
        endif()
    else()
        set(OSCEAN_OPENCL_AVAILABLE FALSE PARENT_SCOPE)
        message(STATUS "❌ OpenCL not available")
    endif()
endfunction()

# Function to detect AMD ROCm/HIP
function(oscean_detect_rocm)
    find_package(hip QUIET)
    
    if(hip_FOUND)
        message(STATUS "✅ AMD ROCm/HIP found")
        set(OSCEAN_ROCM_AVAILABLE TRUE PARENT_SCOPE)
    else()
        set(OSCEAN_ROCM_AVAILABLE FALSE PARENT_SCOPE)
        message(STATUS "❌ AMD ROCm/HIP not available")
    endif()
endfunction()

# Function to detect Intel oneAPI/DPC++
function(oscean_detect_oneapi)
    find_package(IntelDPCPP QUIET)
    
    if(IntelDPCPP_FOUND)
        message(STATUS "✅ Intel oneAPI/DPC++ found")
        set(OSCEAN_ONEAPI_AVAILABLE TRUE PARENT_SCOPE)
    else()
        set(OSCEAN_ONEAPI_AVAILABLE FALSE PARENT_SCOPE)
        message(STATUS "❌ Intel oneAPI/DPC++ not available")
    endif()
endfunction()

# =============================================================================
# Option to force CPU-only mode
# =============================================================================

option(OSCEAN_FORCE_CPU_ONLY "Force CPU-only mode even if GPU is available" OFF)

# =============================================================================
# Main GPU Detection Function
# =============================================================================

function(oscean_detect_gpu_support)
    # Check if force CPU-only mode is enabled first
    if(OSCEAN_FORCE_CPU_ONLY)
        message(STATUS "")
        message(STATUS "========== GPU Support Detection ==========")
        message(STATUS "⚠️  OSCEAN_FORCE_CPU_ONLY is ON - Skipping GPU detection")
        message(STATUS "All GPU support disabled by user request")
        message(STATUS "=========================================")
        message(STATUS "")
        
        # Set all GPU flags to FALSE
        set(OSCEAN_GPU_AVAILABLE FALSE PARENT_SCOPE)
        set(OSCEAN_CUDA_AVAILABLE FALSE PARENT_SCOPE)
        set(OSCEAN_CUDA_ENABLED FALSE PARENT_SCOPE)
        set(OSCEAN_OPENCL_AVAILABLE FALSE PARENT_SCOPE)
        set(OSCEAN_OPENCL_ENABLED FALSE PARENT_SCOPE)
        set(OSCEAN_ROCM_AVAILABLE FALSE PARENT_SCOPE)
        set(OSCEAN_ROCM_ENABLED FALSE PARENT_SCOPE)
        set(OSCEAN_ONEAPI_AVAILABLE FALSE PARENT_SCOPE)
        set(OSCEAN_ONEAPI_ENABLED FALSE PARENT_SCOPE)
        return()
    endif()
    
    message(STATUS "")
    message(STATUS "========== GPU Support Detection ==========")
    
    # Detect various GPU platforms
    oscean_detect_cuda()
    oscean_detect_opencl()
    oscean_detect_rocm()
    oscean_detect_oneapi()
    
    # Determine if any GPU support is available
    if(OSCEAN_CUDA_AVAILABLE OR OSCEAN_OPENCL_AVAILABLE OR 
       OSCEAN_ROCM_AVAILABLE OR OSCEAN_ONEAPI_AVAILABLE)
        set(OSCEAN_GPU_AVAILABLE TRUE PARENT_SCOPE)
        message(STATUS "✅ GPU support available")
    else()
        set(OSCEAN_GPU_AVAILABLE FALSE PARENT_SCOPE)
        message(STATUS "⚠️  No GPU support detected - CPU-only mode enabled")
    endif()
    
    # Set convenience flags
    set(OSCEAN_CUDA_ENABLED ${OSCEAN_CUDA_AVAILABLE} PARENT_SCOPE)
    set(OSCEAN_OPENCL_ENABLED ${OSCEAN_OPENCL_AVAILABLE} PARENT_SCOPE)
    set(OSCEAN_ROCM_ENABLED ${OSCEAN_ROCM_AVAILABLE} PARENT_SCOPE)
    set(OSCEAN_ONEAPI_ENABLED ${OSCEAN_ONEAPI_AVAILABLE} PARENT_SCOPE)
    
    message(STATUS "=========================================")
    message(STATUS "")
endfunction()

# =============================================================================
# GPU Configuration Function
# =============================================================================

function(oscean_configure_gpu_target target)
    # Add compile definitions based on GPU availability
    if(OSCEAN_GPU_AVAILABLE)
        target_compile_definitions(${target} PUBLIC OSCEAN_GPU_AVAILABLE=1)
    else()
        target_compile_definitions(${target} PUBLIC OSCEAN_GPU_AVAILABLE=0)
    endif()
    
    # CUDA-specific configuration
    if(OSCEAN_CUDA_ENABLED)
        target_compile_definitions(${target} PUBLIC 
            OSCEAN_CUDA_ENABLED=1
            OSCEAN_CUDA_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
            OSCEAN_CUDA_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}
        )
        
        if(OSCEAN_CUDA_INCLUDE_DIRS)
            target_include_directories(${target} PUBLIC ${OSCEAN_CUDA_INCLUDE_DIRS})
        endif()
        
        # Link CUDA libraries if needed
        if(TARGET CUDA::cudart)
            target_link_libraries(${target} PUBLIC CUDA::cudart)
        endif()
    else()
        target_compile_definitions(${target} PUBLIC OSCEAN_CUDA_ENABLED=0)
    endif()
    
    # OpenCL-specific configuration
    if(OSCEAN_OPENCL_ENABLED)
        target_compile_definitions(${target} PUBLIC OSCEAN_OPENCL_ENABLED=1)
        
        if(OSCEAN_OPENCL_INCLUDE_DIRS)
            target_include_directories(${target} PUBLIC ${OSCEAN_OPENCL_INCLUDE_DIRS})
        endif()
        
        if(TARGET OpenCL::OpenCL)
            target_link_libraries(${target} PUBLIC OpenCL::OpenCL)
        endif()
    else()
        target_compile_definitions(${target} PUBLIC OSCEAN_OPENCL_ENABLED=0)
    endif()
    
    # ROCm-specific configuration
    if(OSCEAN_ROCM_ENABLED)
        target_compile_definitions(${target} PUBLIC OSCEAN_ROCM_ENABLED=1)
    else()
        target_compile_definitions(${target} PUBLIC OSCEAN_ROCM_ENABLED=0)
    endif()
    
    # oneAPI-specific configuration
    if(OSCEAN_ONEAPI_ENABLED)
        target_compile_definitions(${target} PUBLIC OSCEAN_ONEAPI_ENABLED=1)
    else()
        target_compile_definitions(${target} PUBLIC OSCEAN_ONEAPI_ENABLED=0)
    endif()
endfunction()

# =============================================================================
# Conditional GPU Source Management
# =============================================================================

function(oscean_add_gpu_sources target)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs CUDA_SOURCES OPENCL_SOURCES ROCM_SOURCES ONEAPI_SOURCES CPU_FALLBACK_SOURCES)
    cmake_parse_arguments(GPU "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Always add CPU fallback sources
    if(GPU_CPU_FALLBACK_SOURCES)
        target_sources(${target} PRIVATE ${GPU_CPU_FALLBACK_SOURCES})
    endif()
    
    # Add platform-specific sources only if available
    if(OSCEAN_CUDA_ENABLED AND GPU_CUDA_SOURCES)
        target_sources(${target} PRIVATE ${GPU_CUDA_SOURCES})
    endif()
    
    if(OSCEAN_OPENCL_ENABLED AND GPU_OPENCL_SOURCES)
        target_sources(${target} PRIVATE ${GPU_OPENCL_SOURCES})
    endif()
    
    if(OSCEAN_ROCM_ENABLED AND GPU_ROCM_SOURCES)
        target_sources(${target} PRIVATE ${GPU_ROCM_SOURCES})
    endif()
    
    if(OSCEAN_ONEAPI_ENABLED AND GPU_ONEAPI_SOURCES)
        target_sources(${target} PRIVATE ${GPU_ONEAPI_SOURCES})
    endif()
endfunction()

# =============================================================================
# GPU Feature Summary
# =============================================================================

function(oscean_print_gpu_summary)
    message(STATUS "")
    message(STATUS "GPU Configuration Summary:")
    
    if(OSCEAN_FORCE_CPU_ONLY)
        message(STATUS "  CPU-Only Mode: FORCED")
        message(STATUS "  Any GPU Available: FALSE (disabled by user)")
    else()
        message(STATUS "  Any GPU Available: ${OSCEAN_GPU_AVAILABLE}")
    endif()
    
    message(STATUS "  CUDA Enabled: ${OSCEAN_CUDA_ENABLED}")
    if(OSCEAN_CUDA_ENABLED)
        message(STATUS "    Version: ${OSCEAN_CUDA_VERSION}")
        message(STATUS "    Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    message(STATUS "  OpenCL Enabled: ${OSCEAN_OPENCL_ENABLED}")
    if(OSCEAN_OPENCL_ENABLED)
        message(STATUS "    Version: ${OSCEAN_OPENCL_VERSION}")
    endif()
    message(STATUS "  ROCm Enabled: ${OSCEAN_ROCM_ENABLED}")
    message(STATUS "  oneAPI Enabled: ${OSCEAN_ONEAPI_ENABLED}")
    message(STATUS "")
endfunction() 