cmake_minimum_required(VERSION 3.20)

# 设置CMake策略以避免警告
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)  # 使用新的Boost查找模块
endif()

project(interpolation_service VERSION 2.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 启用GPU支持
set(ENABLE_CUDA ON)
set(ENABLE_OPENCL OFF)
set(OSCEAN_GPU_ENABLED ON)

# 检测GPU支持
if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        # 设置CUDA架构
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
        endif()
        
        # 简化CUDA编译标志
        set(CMAKE_CUDA_FLAGS "-Xcompiler=/W3")
        set(CMAKE_CUDA_FLAGS_DEBUG "-G")
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
        
        add_definitions(-DOSCEAN_CUDA_ENABLED=1)
        message(STATUS "[interpolation_service] CUDA support enabled")
    else()
        add_definitions(-DOSCEAN_CUDA_ENABLED=0)
    endif()
else()
    add_definitions(-DOSCEAN_CUDA_ENABLED=0)
endif()

if(ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        add_definitions(-DOSCEAN_OPENCL_ENABLED=1)
        message(STATUS "[interpolation_service] OpenCL support enabled")
    else()
        add_definitions(-DOSCEAN_OPENCL_ENABLED=0)
    endif()
else()
    add_definitions(-DOSCEAN_OPENCL_ENABLED=0)
endif()

#--------------------------------------------------------------------
# 1. 查找依赖（参考data_access服务的处理方式）
#--------------------------------------------------------------------

# 查找common_utilities模块（参考data_access服务）
find_package(common_utilities CONFIG QUIET)
if(NOT common_utilities_FOUND)
    # 检查是否已经作为子目录添加
    if(NOT TARGET common_utilities)
        # 如果没有安装，尝试使用项目根目录的common_utilities
        set(COMMON_UTILITIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../common_utilities")
        if(EXISTS "${COMMON_UTILITIES_ROOT}/CMakeLists.txt")
            message(STATUS "[interpolation_service] Using common_utilities from: ${COMMON_UTILITIES_ROOT}")
            add_subdirectory("${COMMON_UTILITIES_ROOT}" common_utilities_build)
            set(common_utilities_FOUND TRUE)
            # 添加包含目录
            list(APPEND INTERPOLATION_INCLUDE_DIRS "${COMMON_UTILITIES_ROOT}/include")
            include_directories("${COMMON_UTILITIES_ROOT}/include")
        else()
            message(FATAL_ERROR "[interpolation_service] common_utilities not found at ${COMMON_UTILITIES_ROOT}. This is required for SIMD support.")
        endif()
    else()
        message(STATUS "[interpolation_service] common_utilities target already exists, skipping add_subdirectory")
        # 仍然添加包含目录
        set(COMMON_UTILITIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../common_utilities")
        list(APPEND INTERPOLATION_INCLUDE_DIRS "${COMMON_UTILITIES_ROOT}/include")
        include_directories("${COMMON_UTILITIES_ROOT}/include")
        set(common_utilities_FOUND TRUE)
    endif()
else()
    message(STATUS "[interpolation_service] Found common_utilities package")
endif()

# 查找core_service_interfaces模块
if(NOT TARGET core_service_interfaces)
    set(CORE_SERVICE_INTERFACES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces")
    if(EXISTS "${CORE_SERVICE_INTERFACES_ROOT}/CMakeLists.txt")
        message(STATUS "[interpolation_service] Using core_service_interfaces from: ${CORE_SERVICE_INTERFACES_ROOT}")
        add_subdirectory("${CORE_SERVICE_INTERFACES_ROOT}" core_service_interfaces_build)
        # 添加包含目录
        list(APPEND INTERPOLATION_INCLUDE_DIRS "${CORE_SERVICE_INTERFACES_ROOT}/include")
        include_directories("${CORE_SERVICE_INTERFACES_ROOT}/include")
    else()
        message(FATAL_ERROR "[interpolation_service] core_service_interfaces not found at ${CORE_SERVICE_INTERFACES_ROOT}.")
    endif()
else()
    message(STATUS "[interpolation_service] core_service_interfaces target already exists, skipping add_subdirectory")
    # 仍然添加包含目录
    set(CORE_SERVICE_INTERFACES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces")
    list(APPEND INTERPOLATION_INCLUDE_DIRS "${CORE_SERVICE_INTERFACES_ROOT}/include")
    include_directories("${CORE_SERVICE_INTERFACES_ROOT}/include")
endif()

find_package(Boost REQUIRED COMPONENTS thread system filesystem log)
if(Boost_FOUND)
    message(STATUS "[interpolation_service] Found Boost: ${Boost_VERSION}")
    list(APPEND INTERPOLATION_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "[interpolation_service] Boost (thread, system, filesystem, log) libraries not found. These are required dependencies.")
endif()

#--------------------------------------------------------------------
# 2. 定义源文件
#--------------------------------------------------------------------

# 服务实现和工厂
set(SERVICE_CORE_SOURCES
    src/impl/interpolation_service_impl.cpp
    src/factory/interpolation_service_factory.cpp
)

# 算法实现 - 修正路径
set(ALGORITHM_SOURCES
    src/impl/algorithms/bilinear_interpolator.cpp
    src/impl/algorithms/cubic_spline_interpolator.cpp
    src/impl/algorithms/linear_1d_interpolator.cpp
    src/impl/algorithms/nearest_neighbor_interpolator.cpp
    src/impl/algorithms/pchip_interpolator.cpp
    src/impl/algorithms/trilinear_interpolator.cpp
    src/impl/algorithms/layout_aware_interpolator_base.cpp
    src/impl/algorithms/recursive_ndim_pchip_interpolator.cpp
    src/impl/algorithms/complex_field_interpolator.cpp
    # 内核函数
    src/impl/algorithms/kernels/interpolation_kernels.cpp
)

# 检查额外的算法实现
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/impl/algorithms/fast_pchip_interpolator_2d.cpp")
    list(APPEND ALGORITHM_SOURCES
        src/impl/algorithms/fast_pchip_interpolator_2d.cpp
        src/impl/algorithms/fast_pchip_interpolator_3d.cpp
        src/impl/algorithms/pchip_interpolator_2d_bathy.cpp
    )
endif()

# GPU源文件
set(GPU_SOURCES
    src/gpu/gpu_interpolation_engine.cpp
    src/gpu/optimized_batch_engine.cpp  # 优化的批量处理引擎
    src/impl/layout_converter.cpp  # 添加布局转换器
)

# CUDA核函数
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    # CUDA核函数文件
    set(CUDA_SOURCES
        src/gpu/cuda/bilinear_interpolation.cu
        src/gpu/cuda/bicubic_interpolation.cu
        src/gpu/cuda/trilinear_interpolation.cu
        src/gpu/cuda/pchip_interpolation_optimized.cu  # 优化的PCHIP版本（包含一体化实现）
        src/gpu/cuda/nearest_neighbor_interpolation.cu
        src/gpu/cuda/batch_interpolation.cu
        src/gpu/cuda/batch_interpolation_optimized.cu  # 优化的批量处理核函数
    )
    
    # 创建CUDA对象库以避免命令行过长问题
    add_library(interpolation_cuda_kernels OBJECT ${CUDA_SOURCES})
    
    # 设置CUDA编译选项
    set_target_properties(interpolation_cuda_kernels PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "75;80;86;89"
    )
    
    # 为CUDA对象库设置包含目录
    target_include_directories(interpolation_cuda_kernels
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/impl
            ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/algorithms
            ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms
            ${INTERPOLATION_INCLUDE_DIRS}
    )
    
    # 最小化CUDA编译定义，避免重复
    target_compile_definitions(interpolation_cuda_kernels
        PRIVATE
            OSCEAN_CUDA_ENABLED=1
    )
endif()

# OpenCL源文件
if(ENABLE_OPENCL AND OpenCL_FOUND)
    list(APPEND GPU_SOURCES
        src/gpu/opencl/opencl_kernels.cpp
    )
endif()

# 注释掉暂时不需要的CUDA文件
# if(ENABLE_CUDA AND CUDAToolkit_FOUND)
#     set(CUDA_SOURCES
#         src/gpu/cuda/bilinear_interpolation.cu
#         src/gpu/cuda/bicubic_interpolation.cu
#         src/gpu/cuda/trilinear_interpolation.cu
#         src/gpu/cuda/pchip_interpolation.cu
#     )
#     set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
#     list(APPEND GPU_SOURCES ${CUDA_SOURCES})
# endif()

# 收集所有源文件
set(ALL_SOURCES
    ${SERVICE_CORE_SOURCES}
    ${ALGORITHM_SOURCES}
    ${GPU_SOURCES}
)

#--------------------------------------------------------------------
# 3. 创建库
#--------------------------------------------------------------------
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    # 包含CUDA对象文件
    add_library(interpolation_service ${ALL_SOURCES} $<TARGET_OBJECTS:interpolation_cuda_kernels>)
else()
    add_library(interpolation_service ${ALL_SOURCES})
endif()

# 创建别名
add_library(OSCEAN::interpolation_service ALIAS interpolation_service)

#--------------------------------------------------------------------
# 4. 设置包含目录
#--------------------------------------------------------------------
target_include_directories(interpolation_service
    PUBLIC
        # 公共接口
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # 私有实现
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/algorithms
        ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms
)

#--------------------------------------------------------------------
# 5. 链接依赖（直接使用目标，不查找包）
#--------------------------------------------------------------------
target_link_libraries(interpolation_service
    PUBLIC
        core_service_interfaces
        common_utilities
        Boost::thread
        Boost::system
        Boost::filesystem
        Boost::log
)

# 链接GPU库
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(interpolation_service PRIVATE CUDA::cudart)
endif()

if(ENABLE_OPENCL AND OpenCL_FOUND)
    target_link_libraries(interpolation_service PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(interpolation_service PRIVATE ${OpenCL_INCLUDE_DIRS})
endif()

#--------------------------------------------------------------------
# 6. 编译选项
#--------------------------------------------------------------------
if(MSVC)
    target_compile_options(interpolation_service PRIVATE /W4 /EHsc /FS)
    target_compile_definitions(interpolation_service PRIVATE NOMINMAX)
    # 启用SIMD优化
    target_compile_options(interpolation_service PRIVATE /arch:AVX2)
    
    # CUDA特殊处理
    if(ENABLE_CUDA AND CUDAToolkit_FOUND)
        # 为CUDA文件设置更简单的编译选项
        set_source_files_properties(src/gpu/gpu_interpolation_kernels.cu
            PROPERTIES COMPILE_OPTIONS "")
    endif()
else()
    target_compile_options(interpolation_service PRIVATE -Wall -Wextra -Wpedantic)
    # 启用SIMD优化
    target_compile_options(interpolation_service PRIVATE -march=native)
endif()

# 设置编译特性
target_compile_features(interpolation_service PUBLIC cxx_std_17)

#--------------------------------------------------------------------
# 7. 输出目录
#--------------------------------------------------------------------
set_target_properties(interpolation_service PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

#--------------------------------------------------------------------
# 8. 测试
#--------------------------------------------------------------------
if(BUILD_TESTING)
    add_subdirectory(tests)
endif() 