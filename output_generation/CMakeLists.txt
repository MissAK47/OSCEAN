cmake_minimum_required(VERSION 3.10)

project(output_generation)

# 设置库名称
set(LIB_NAME "output_generation")

# ====================================================================
# 1. 定义源文件
# ====================================================================
# 明确列出所有源文件，这是比 GLOB 更推荐的做法
set(SOURCES
    "src/output_service_impl.cpp"
    "src/output_service_factory.cpp"
    "src/profiles/output_profile_manager.cpp"
    "src/engines/data_export_engine.cpp"
    "src/engines/in_memory_data_reader.cpp"
    "src/engines/visualization_engine.cpp"
    "src/engines/font_renderer.cpp"
    "src/writers/writer_factory.cpp"
    "src/writers/text_based_writer.cpp"
    "src/writers/netcdf_writer.cpp"
    "src/writers/gdal_raster_writer.cpp"
    "src/writers/gdal_vector_writer.cpp"
)

# GPU相关源文件（如果启用）
if(OSCEAN_GPU_AVAILABLE)
    list(APPEND SOURCES
        "src/engines/gpu/gpu_visualization_engine.cpp"
        "src/engines/gpu/gpu_color_mapper.cpp"
        "src/engines/gpu/gpu_tile_generator_enhanced.cpp"
        "src/engines/gpu/multi_gpu_coordinator.cpp"
    )
endif()

# CUDA特定文件
if(OSCEAN_CUDA_ENABLED AND CUDAToolkit_FOUND)
    # 启用CUDA语言支持
    enable_language(CUDA)
    
    list(APPEND SOURCES
        "src/engines/gpu/gpu_color_mapper_cuda.cpp"
    )
    
    # CUDA核函数文件
    set(CUDA_SOURCES
        "src/engines/gpu/cuda/color_mapping.cu"
        "src/engines/gpu/cuda/tile_generation.cu"
        "src/engines/gpu/cuda/min_max_reduction.cu"
        "src/engines/gpu/cuda/statistics.cu"
        "src/engines/gpu/cuda/image_resampling.cu"
        "src/engines/gpu/cuda/marching_squares.cu"
        "src/engines/gpu/cuda/image_compression.cu"
    )
    
    # 创建CUDA对象库以避免命令行过长问题
    add_library(${LIB_NAME}_cuda_kernels OBJECT ${CUDA_SOURCES})
    
    # 设置CUDA编译选项
    set_target_properties(${LIB_NAME}_cuda_kernels PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    
    # 为CUDA对象库设置包含目录
    target_include_directories(${LIB_NAME}_cuda_kernels
        PUBLIC  
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    
    # 应用GPU配置到CUDA对象库
    oscean_configure_gpu_target(${LIB_NAME}_cuda_kernels)
endif()

# OpenCL特定文件
if(OSCEAN_OPENCL_ENABLED)
    list(APPEND SOURCES
        "src/engines/gpu/opencl/opencl_kernels.cpp"
    )
    
    # OpenCL kernel文件（作为资源）
    set(OPENCL_KERNELS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/engines/gpu/opencl/kernels/color_mapping.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/engines/gpu/opencl/kernels/tile_generation.cl"
    )
    
    # 将OpenCL kernel文件路径传递给编译器
    foreach(kernel ${OPENCL_KERNELS})
        if(EXISTS ${kernel})
            message(STATUS "Found OpenCL kernel: ${kernel}")
        endif()
    endforeach()
endif()

# ====================================================================
# 2. 创建静态库
# ====================================================================
if(OSCEAN_CUDA_ENABLED AND CUDAToolkit_FOUND)
    # 包含CUDA对象文件
    add_library(${LIB_NAME} ${SOURCES} $<TARGET_OBJECTS:${LIB_NAME}_cuda_kernels>)
else()
    add_library(${LIB_NAME} ${SOURCES})
endif()

# 应用GPU配置到主库
oscean_configure_gpu_target(${LIB_NAME})

# 设置标准和属性
set_target_properties(${LIB_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# ====================================================================
# Boost.Thread 宏定义 - 必需用于 boost::future 支持
# ====================================================================
target_compile_definitions(${LIB_NAME} PRIVATE
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

# ====================================================================
# 3. 设置包含目录和依赖关系
# ====================================================================
target_include_directories(${LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # 项目内构建时使用
        $<INSTALL_INTERFACE:include>                          # 安装后使用
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/../core_service_interfaces/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../common_utilities/include"  # 确保可以访问
)

# ====================================================================
# 查找Boost依赖
# ====================================================================
find_package(Boost REQUIRED COMPONENTS system thread filesystem log unit_test_framework)
find_package(Freetype QUIET)  # 查找FreeType库（可选）

# ====================================================================
# 查找NetCDF依赖 - 使用vcpkg兼容的CONFIG模式
# ====================================================================
find_package(netCDF CONFIG REQUIRED)
message(STATUS "NetCDF Found: ${netCDF_FOUND}")
message(STATUS "NetCDF Version: ${netCDF_VERSION}")

# 查找NetCDF C++接口库 - 注意vcpkg中的包名区分大小写
find_package(netcdf-cxx4 CONFIG QUIET)
if(NOT netcdf-cxx4_FOUND)
    # 尝试使用不同的名称形式
    find_package(netCDF-cxx4 CONFIG QUIET)
    if(NOT netCDF-cxx4_FOUND)
        message(STATUS "未找到netcdf-cxx4/netCDF-cxx4包，尝试手动配置...")
        # 创建一个接口库，只链接到netCDF C库
        if(TARGET netCDF::netcdf)
            message(STATUS "找到netCDF C库，创建netcdf-cxx4接口库")
            # 如果只有C库找到，创建一个接口库
            add_library(netcdf_cxx4 INTERFACE IMPORTED)
            set_target_properties(netcdf_cxx4 PROPERTIES
                INTERFACE_LINK_LIBRARIES "netCDF::netcdf"
            )
            set(netcdf-cxx4_FOUND TRUE)
        else()
            message(FATAL_ERROR "无法找到netCDF C库或C++库，请安装vcpkg netcdf-cxx4包")
        endif()
    endif()
endif()
message(STATUS "NetCDF C++ 接口配置完成")

# 确定要链接的正确目标名称
if(TARGET netcdf-cxx4)
    set(NETCDF_CXX4_TARGET netcdf-cxx4)
elseif(TARGET netCDF-cxx4)
    set(NETCDF_CXX4_TARGET netCDF-cxx4)
elseif(TARGET netcdf_cxx4)
    set(NETCDF_CXX4_TARGET netcdf_cxx4)
else()
    message(FATAL_ERROR "无法找到可用的NetCDF C++目标")
endif()

# 链接核心依赖
target_link_libraries(${LIB_NAME}
    PUBLIC
        common_utilities
        core_service_interfaces
        netCDF::netcdf
        ${NETCDF_CXX4_TARGET}
        spdlog::spdlog
        GDAL::GDAL
        Boost::system
        Boost::thread
        Boost::filesystem
        Boost::log
)

# 添加FreeType支持（如果找到）
if(Freetype_FOUND)
    target_link_libraries(${LIB_NAME} PUBLIC Freetype::Freetype)
    target_compile_definitions(${LIB_NAME} PUBLIC HAVE_FREETYPE=1)
    message(STATUS "FreeType support enabled for font rendering")
else()
    message(STATUS "FreeType not found, font rendering will be limited")
endif()

# GPU相关链接
if(OSCEAN_CUDA_ENABLED AND CUDAToolkit_FOUND)
    # CUDA库链接由oscean_configure_gpu_target处理
    # 这里只需要添加额外的CUDA库（如果需要）
    if(TARGET CUDA::cublas)
        target_link_libraries(${LIB_NAME} PRIVATE CUDA::cublas)
    endif()
    if(TARGET CUDA::cufft)
        target_link_libraries(${LIB_NAME} PRIVATE CUDA::cufft)
    endif()
endif()

if(OSCEAN_OPENCL_ENABLED)
    # OpenCL链接由oscean_configure_gpu_target处理
endif()

# 提供一个宏用于条件编译
target_compile_definitions(${LIB_NAME} PUBLIC
    OSCEAN_OUTPUT_GENERATION_EXPORTS  # 用于DLL导出
)

# ====================================================================
# 4. 安装规则
# ====================================================================
install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ====================================================================
# 5. 测试配置
# ====================================================================
if(BUILD_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)

    # GDAL诊断工具已移除 - 不再需要

    set(TEST_SOURCES
        "tests/test_visualization_engine.cpp"
        "tests/test_data_export_engine.cpp"
        "tests/test_simd_performance.cpp"
        "tests/detailed_performance_benchmark.cpp"
        # 在后续阶段添加其他测试文件
        # "tests/test_output_profile_manager.cpp"
    )
    
    # GPU测试（如果启用）
    if(OSCEAN_CUDA_ENABLED OR OSCEAN_OPENCL_ENABLED)
        list(APPEND TEST_SOURCES
            # "tests/gpu/test_gpu_color_mapping.cpp"  # 暂时禁用，需要修复API调用
            "tests/test_gpu_visualization.cpp"
        )
        
        # 添加简单的GPU测试程序（不使用gtest）
        add_executable(test_gpu_simple tests/test_gpu_simple.cpp)
        target_link_libraries(test_gpu_simple PRIVATE
            ${LIB_NAME}
            core_service_interfaces
            common_utilities
            spdlog::spdlog
        )
        set_target_properties(test_gpu_simple PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        target_compile_definitions(test_gpu_simple PRIVATE
            BOOST_THREAD_PROVIDES_FUTURE=1
            BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
            BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
        )
        
        # 添加GPU性能基准测试
        add_executable(test_gpu_performance_benchmark tests/test_gpu_performance_benchmark.cpp)
        target_link_libraries(test_gpu_performance_benchmark PRIVATE
            ${LIB_NAME}
            core_service_interfaces
            common_utilities
            spdlog::spdlog
        )
        set_target_properties(test_gpu_performance_benchmark PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        target_compile_definitions(test_gpu_performance_benchmark PRIVATE
            BOOST_THREAD_PROVIDES_FUTURE=1
            BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
            BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
        )
        
        # 添加GPU颜色映射测试
        add_executable(test_gpu_color_maps tests/test_gpu_color_maps.cpp)
        target_link_libraries(test_gpu_color_maps PRIVATE
            ${LIB_NAME}
            core_service_interfaces
            common_utilities
            spdlog::spdlog
            Boost::filesystem
        )
        set_target_properties(test_gpu_color_maps PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        target_compile_definitions(test_gpu_color_maps PRIVATE
            BOOST_THREAD_PROVIDES_FUTURE=1
            BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
            BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
        )
        
        # 添加GPU批处理性能测试
        add_executable(test_gpu_batch_performance tests/test_gpu_batch_performance.cpp)
        target_link_libraries(test_gpu_batch_performance PRIVATE
            ${LIB_NAME}
            core_service_interfaces
            common_utilities
            spdlog::spdlog
        )
        set_target_properties(test_gpu_batch_performance PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        target_compile_definitions(test_gpu_batch_performance PRIVATE
            BOOST_THREAD_PROVIDES_FUTURE=1
            BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
            BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
        )
        
        # GPU瓦片生成测试
        add_executable(test_gpu_tile_generation
            tests/test_gpu_tile_generation_complete.cpp
        )
        target_link_libraries(test_gpu_tile_generation
            PRIVATE
            output_generation
            core_service_interfaces
            common_utilities
            ${Boost_LIBRARIES}
            spdlog::spdlog
        )
        if(CUDAToolkit_FOUND)
            target_link_libraries(test_gpu_tile_generation PRIVATE CUDA::cudart)
        endif()
        set_target_properties(test_gpu_tile_generation PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        target_compile_definitions(test_gpu_tile_generation PRIVATE
            BOOST_THREAD_PROVIDES_FUTURE=1
            BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
            BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
        )
        
        # GPU优化功能测试
        add_executable(test_gpu_optimizations
            tests/test_gpu_optimizations.cpp
        )
        target_link_libraries(test_gpu_optimizations
            PRIVATE
            output_generation
            core_service_interfaces
            common_utilities
            ${Boost_LIBRARIES}
            spdlog::spdlog
        )
        if(CUDAToolkit_FOUND)
            target_link_libraries(test_gpu_optimizations PRIVATE CUDA::cudart)
        endif()
        
        # GPU图像重采样测试
        if(CUDAToolkit_FOUND)
            add_executable(test_gpu_image_resampling
                tests/gpu/test_gpu_image_resampling.cpp
            )
            target_link_libraries(test_gpu_image_resampling
                PRIVATE
                output_generation_cuda_kernels
                ${Boost_LIBRARIES}
                CUDA::cudart
            )
            set_target_properties(test_gpu_image_resampling PROPERTIES
                CXX_STANDARD 17
                CXX_STANDARD_REQUIRED ON
                CXX_EXTENSIONS OFF
            )
            target_compile_definitions(test_gpu_image_resampling PRIVATE
                BOOST_THREAD_PROVIDES_FUTURE=1
                BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
                BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
            )
        endif()
    endif()

    add_executable(test_${LIB_NAME} ${TEST_SOURCES})

    # 设置测试目标的C++17标准和Boost.Thread宏定义
    set_target_properties(test_${LIB_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    target_compile_definitions(test_${LIB_NAME} PRIVATE
        BOOST_THREAD_PROVIDES_FUTURE=1
        BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
        BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
    )

    # 为测试添加必要的包含目录 - 测试暴露了这个配置缺失
    target_include_directories(test_${LIB_NAME}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"  # 访问实现文件中的头文件
            "${CMAKE_CURRENT_SOURCE_DIR}/include"  # 访问公开头文件
    )

    target_link_libraries(test_${LIB_NAME}
        PRIVATE
            ${LIB_NAME}
            GTest::gtest
            GTest::gtest_main
            Boost::log
            Boost::unit_test_framework  # 添加Boost.Test库
    )

    include(GoogleTest)
    gtest_discover_tests(test_${LIB_NAME})
endif()

# ====================================================================
# 6. 编译器特定设置
# ====================================================================
if(MSVC)
    # 修复并行编译时的PDB文件冲突问题
    target_compile_options(${LIB_NAME} PRIVATE /FS)
endif()

# 查找依赖项
find_package(GDAL REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(spdlog REQUIRED)
find_package(Freetype QUIET)  # 查找FreeType库（可选） 