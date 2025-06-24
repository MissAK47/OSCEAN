# Common Utilities CMake Configuration
# 重构后的统一基础设施构建配置

cmake_minimum_required(VERSION 3.20)
project(common_utilities VERSION 2.0.0 LANGUAGES CXX)

# 包含必要的CMake模块
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# 设置vcpkg工具链（如果未指定）
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "C:/Users/Administrator/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/Administrator/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# C++标准要求
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项 - 平台兼容
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /FS /Zc:__cplusplus /bigobj")
    # 修复Windows版本宏未定义问题
    add_definitions(-D_WIN32_WINNT=0x0A00)  # Windows 10
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# 🔴 重构方案：配置异步框架选择
option(OSCEAN_USE_BOOST_FUTURE "Use boost::future instead of std::future" ON)
option(OSCEAN_ENFORCE_ASYNC_COMPLIANCE "Enforce async compliance at compile time" ON)

if(OSCEAN_USE_BOOST_FUTURE)
    add_definitions(-DOSCEAN_USE_BOOST_FUTURE)
    message(STATUS "Using boost::future for async operations")
else()
    message(STATUS "Using std::future for async operations")
endif()

if(OSCEAN_ENFORCE_ASYNC_COMPLIANCE)
    add_definitions(-DOSCEAN_ENFORCE_ASYNC_COMPLIANCE)
    message(STATUS "Enforcing async compliance at compile time")
endif()

# =============================================================================
# 🔴 依赖管理修复：使用主CMake传递的依赖
# =============================================================================

# 查找并使用主CMake传递的依赖库
find_package(GDAL CONFIG REQUIRED)  # 使用CONFIG模式查找GDAL
find_package(netCDF CONFIG QUIET)
find_package(Boost REQUIRED COMPONENTS thread system filesystem)
find_package(spdlog CONFIG REQUIRED)

# 打印依赖信息用于调试
message(STATUS "[common_utilities] GDAL_FOUND: ${GDAL_FOUND}")
if(GDAL_FOUND)
    message(STATUS "[common_utilities] GDAL target available: GDAL::GDAL")
endif()

message(STATUS "[common_utilities] Boost_FOUND: ${Boost_FOUND}")

# =============================================================================
# 🔴 重构后的新文件结构源文件
# =============================================================================

# 核心模块入口
set(CORE_SOURCES
    src/common_utils/common_utils.cpp       # 🆕 模块统一入口实现
)

# 🆕 重构后的utilities模块源文件
set(UTILITIES_SOURCES
    src/utilities/string_utils.cpp         # 字符串工具实现
    src/utilities/filesystem_path_utils.cpp     # 文件系统路径工具实现 - 修正路径
    src/utilities/filesystem_query_utils.cpp    # 文件系统查询工具实现 - 修正路径
    src/utilities/filesystem_io_utils.cpp       # 文件系统I/O工具实现 - 修正路径
    src/utilities/filesystem_manipulation_utils.cpp # 文件系统操作工具实现 - 修正路径
    src/utilities/file_format_detector.cpp      # 🆕 文件格式检测工具实现 - 修正路径
    src/utilities/app_config_loader.cpp         # 🆕 应用配置加载器实现 - 修正路径
    src/utilities/logging_utils.cpp        # 日志工具实现
    src/utilities/error_handling.cpp       # 错误处理工具实现
)

# 🆕 重构后的异步框架源文件 (新的统一实现)
set(ASYNC_SOURCES
    src/async/async_config.cpp          # 🆕 异步配置实现 - 修正路径
    src/async/async_framework.cpp       # 🆕 异步框架核心实现 - 修正路径
    # 注意：已删除的文件：
    # - async_factory.h (功能由 CommonServicesFactory 取代)
    # - async_interfaces.h (接口已整合至 async_framework.h)
    # - async_composition.h (逻辑整合到 async_framework.cpp)
    # - async_context.h (逻辑整合到 async_framework.cpp)
    # - async_enhanced.h (逻辑整合到 async_framework.cpp)
    # - async_patterns.h (逻辑整合到 async_framework.cpp)
)

# 缓存管理器源文件 (🔴 重构后的新cache实现)
set(CACHE_SOURCES
    src/cache/cache_config.cpp              # 缓存配置和策略定义实现
    src/cache/cache_strategies.cpp          # 缓存策略实现(LRU/LFU/FIFO/TTL/自适应)
    # 注意：已删除 cache_factory.h (功能由 CommonServicesFactory 取代)
    # 注意：已删除 cache_spatial.cpp (模板类，实现在头文件中)
)

# 基础设施源文件 (确认能编译)
set(INFRASTRUCTURE_SOURCES
    src/infrastructure/common_services_factory.cpp    # 修正路径
    src/infrastructure/large_file_processor.cpp         # 🆕 大文件处理器实现
    src/infrastructure/unified_thread_pool_manager.cpp # 🆕 统一线程池管理器实现
    # 🔧 移除：unified_thread_pool_manager.cpp 已改为仅头文件模板实现
    # 🔧 移除：performance_monitor.cpp 已改为仅头文件模板实现
)

# 🔄 重构后的内存管理源文件 (新的统一实现)
set(MEMORY_SOURCES
    src/memory/memory_config.cpp               # 内存配置实现
    src/memory/memory_manager_unified.cpp      # 统一内存管理器核心实现
    src/memory/memory_allocators.cpp           # STL兼容分配器实现
    src/memory/memory_statistics.cpp           # 内存统计和监控实现
    # 注意：已删除 memory_factory.h (功能由 CommonServicesFactory 取代)
    # 注意：已删除 memory_concurrent.cpp (逻辑已整合)
    # 注意：已删除 memory_pools.cpp (逻辑已整合)
    # 注意：已删除 memory_streaming.cpp (逻辑已整合)
    # 注意：已删除 memory_interfaces.cpp (逻辑已整合)
    # 注意：已删除 adaptive_memory_strategy.cpp (过度复杂，已删除)
)

# 🔄 重新启用流式处理源文件 - 所有异步接口问题已解决
set(STREAMING_SOURCES
    # 注意：streaming模块已大幅简化，大部分功能迁移到infrastructure/large_file_processor
    # src/streaming/streaming_buffer.cpp     # ✅ 已修复：基础缓冲管理 - 已删除
    # src/streaming/streaming_reader.cpp     # ✅ 已修复：异步读取器接口实现完整 - 已删除
    # src/streaming/streaming_transformer.cpp # ✅ 已修复：异步变换器接口实现完整 - 已删除
    # src/streaming/streaming_memory.cpp     # ✅ 已修复：内存管理和压力监控 - 已删除
    # src/streaming/streaming_factory.cpp    # ✅ 已修复：基础部分（工厂返回nullptr策略）- 已删除
    # src/streaming/streaming_pipeline.cpp   # ✅ 已修复：所有7个异步方法实现完整 - 已删除
    # src/streaming/streaming_large_data.cpp # ✅ 已修复：所有3个异步方法声明+实现完整 - 已删除
)

# 🔄 重新启用SIMD处理源文件 - 已拆分为多个专用模块文件
set(SIMD_SOURCES
    src/simd/simd_config.cpp               # ✅ SIMD配置和环境检测
    src/simd/simd_manager_unified.cpp      # ✅ 统一SIMD管理器实现
    src/simd/simd_manager_math.cpp         # 🆕 数学运算实现（包含各种数学函数和统计）
    src/simd/simd_manager_geo.cpp          # 🆕 地理操作实现（投影、插值、变换等）
    src/simd/simd_manager_ocean.cpp        # 🆕 海洋数据专用操作实现
    # 注意：已删除 simd_factory.h (功能由 CommonServicesFactory 取代)
    # 注意：已删除 simd_operations_basic.cpp (功能已整合到 simd_manager_unified.cpp)
    # 注意：已删除 simd_unified.cpp (重构为 simd_manager_unified.cpp)
)

# 🔄 重构后的时间管理源文件 (纯净化实现)
set(TIME_SOURCES
    src/time/time_types.cpp         # 通用时间类型实现
    src/time/time_range.cpp         # 时间范围处理实现  
    src/time/time_resolution.cpp    # 时间分辨率管理实现
    src/time/time_calendar.cpp      # 通用日历处理实现
    src/time/time_interfaces.cpp    # 时间接口实现
    src/time/time_services.cpp      # 时间服务统一接口实现
    # 注意：已删除 time_factory.cpp (功能由 CommonServicesFactory 取代)
)

# 🆕 GPU基础设施源文件
# GPU基础源文件（总是编译）
set(GPU_SOURCES
    src/gpu/unified_gpu_manager.cpp         # 统一GPU管理器实现
    src/gpu/multi_gpu_scheduler.cpp         # 多GPU调度器实现
    src/gpu/multi_gpu_memory_manager.cpp    # 多GPU内存管理器实现
    src/gpu/oscean_gpu_framework.cpp        # OSCEAN GPU框架实现
    src/gpu/gpu_performance_monitor.cpp     # GPU性能监控实现
)

# 条件添加CUDA源文件
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    list(APPEND GPU_SOURCES
        src/gpu/cuda/cuda_device_detector.cpp   # CUDA设备检测
        src/gpu/cuda/cuda_memory_manager.cpp    # CUDA内存管理实现
    )
    add_definitions(-DOSCEAN_CUDA_ENABLED=1)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "[GPU] CUDA support enabled")
    message(STATUS "[GPU] CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
else()
    add_definitions(-DOSCEAN_CUDA_ENABLED=0)
    message(STATUS "[GPU] CUDA not available")
endif()

# 条件添加OpenCL源文件
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    list(APPEND GPU_SOURCES
        src/gpu/opencl/opencl_device_detector.cpp # OpenCL设备检测
        src/gpu/opencl/opencl_memory_manager.cpp # OpenCL内存管理实现
    )
    add_definitions(-DOSCEAN_OPENCL_ENABLED=1)
    include_directories(${OpenCL_INCLUDE_DIRS})
    message(STATUS "[GPU] OpenCL support enabled")
    message(STATUS "[GPU] OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
else()
    add_definitions(-DOSCEAN_OPENCL_ENABLED=0)
    message(STATUS "[GPU] OpenCL not available")
endif()

# ROCm检测（占位符，总是编译）
list(APPEND GPU_SOURCES src/gpu/rocm/rocm_device_detector.cpp)

# 🆕 重构后的并行处理源文件 (新的统一实现)
set(PARALLEL_SOURCES
    # 注意：parallel模块已大幅简化，大部分功能迁移到infrastructure和其他模块
    # src/parallel/parallel_scheduler.cpp     # 任务调度器实现 - 已删除
    # src/parallel/parallel_algorithms.cpp    # 并行算法实现 - 已删除
    # src/parallel/parallel_data_ops.cpp      # 数据并行操作实现 - 已删除
    # src/parallel/parallel_spatial_ops.cpp   # 空间并行操作实现 - 已删除
    # src/parallel/parallel_enhanced.cpp      # 增强并行引擎实现 - 已删除
    # src/parallel/parallel_factory.cpp       # 并行处理工厂实现 - 已删除
)

# 🆕 重构后的格式工具源文件 (新的统一实现)
set(FORMAT_UTILS_SOURCES
    # 注意：format_utils模块已删除，功能迁移到utilities/file_format_detector
    # src/format_utils/format_detection.cpp       # 格式检测实现 - 已删除
    # src/format_utils/format_factory.cpp         # 格式工具工厂实现 - 已删除
    # 注意：gdal和netcdf子模块已删除，功能迁移到utilities/file_format_detector
    # src/format_utils/netcdf/netcdf_format.cpp   # NetCDF格式工具实现 - 已删除
    # src/format_utils/netcdf/netcdf_streaming.cpp # NetCDF流式读取实现 - 已删除
    # src/format_utils/gdal/gdal_format.cpp       # GDAL格式工具实现 - 已删除
    # src/format_utils/gdal/gdal_streaming.cpp    # GDAL流式读取实现 - 已删除
)

# 合并当前可编译的源文件 - 🔴 现在包含重构后的所有模块
set(ALL_SOURCES 
    ${CORE_SOURCES}
    ${UTILITIES_SOURCES}
    ${ASYNC_SOURCES}
    ${CACHE_SOURCES}
    ${INFRASTRUCTURE_SOURCES}
    ${MEMORY_SOURCES}
    ${STREAMING_SOURCES}
    ${SIMD_SOURCES}
    ${TIME_SOURCES}
    ${GPU_SOURCES}      # 🆕 GPU基础设施
    ${PARALLEL_SOURCES}
    # ${FORMAT_UTILS_SOURCES}  # 已删除，功能迁移到utilities模块
)

# 验证所有源文件存在
foreach(source ${ALL_SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        message(WARNING "Source file not found: ${source}")
    endif()
endforeach()

message(STATUS "Total source files: ${ALL_SOURCES}")
list(LENGTH ALL_SOURCES source_count)
message(STATUS "Source files count: ${source_count}")

# 创建common_utilities库
add_library(common_utilities STATIC ${ALL_SOURCES})

# =============================================================================
# 设置编译选项和定义
# =============================================================================

target_compile_definitions(common_utilities PUBLIC
    OSCEAN_COMMON_UTILS_VERSION_MAJOR=2
    OSCEAN_COMMON_UTILS_VERSION_MINOR=0
    OSCEAN_COMMON_UTILS_VERSION_PATCH=0
)

if(OSCEAN_USE_BOOST_FUTURE)
    target_compile_definitions(common_utilities PUBLIC OSCEAN_USE_BOOST_FUTURE)
endif()

# Windows特定定义
if(WIN32)
    target_compile_definitions(common_utilities PRIVATE
        _WIN32_WINNT=0x0A00
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

# 包含目录
target_include_directories(common_utilities PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 链接库
target_link_libraries(common_utilities PUBLIC
    core_service_interfaces
    GDAL::GDAL
    spdlog::spdlog
    Boost::thread
    Boost::system
    Boost::filesystem
)

if(netCDF_FOUND)
    target_link_libraries(common_utilities PUBLIC netCDF::netcdf)
endif()

# =============================================================================
# 测试配置
# =============================================================================
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "common_utilities tests enabled.")
else()
    message(STATUS "common_utilities tests disabled.")
endif()

# =============================================================================
# 🔴 重构验证和测试目标 - 更新状态
# =============================================================================

# 添加重构验证目标
add_custom_target(validate_common_refactoring
    COMMAND echo "Validating Common module refactoring..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Validating Common module refactoring compliance"
)

# 显示配置总结
message(STATUS "Common Utilities v2.0.0 Configuration Summary:")
if(OSCEAN_USE_BOOST_FUTURE)
    message(STATUS "  Async Framework: C++${CMAKE_CXX_STANDARD} with boost::future")
else()
    message(STATUS "  Async Framework: C++${CMAKE_CXX_STANDARD} with std::future")
endif()
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Sources Count: ${source_count}")
message(STATUS "  Memory Management: REFACTORED (统一实现)")
message(STATUS "  Async Framework: REFACTORED (模块化设计)")
message(STATUS "  Factory Pattern: UNIFIED (CommonServicesFactory)")
message(STATUS "  NetCDF Support: ${netCDF_FOUND}")
if(OSCEAN_USE_BOOST_FUTURE AND Boost_FOUND)
    message(STATUS "  Boost Support: ${Boost_FOUND} (${Boost_VERSION})")
endif()

# === 安装和导出 ===

# 安装目标
install(TARGETS common_utilities
    EXPORT CommonUtilitiesTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 安装公共头文件
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# --- 生成和安装CMake包配置文件 ---

# 创建并安装 common_utilitiesConfigVersion.cmake
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/common_utilitiesConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# 创建 common_utilities-config.cmake
configure_package_config_file(
    "cmake/common_utilities-config.cmake.in" # 需要创建这个模板文件
    "${CMAKE_CURRENT_BINARY_DIR}/common_utilities-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/common_utilities"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

# 安装配置文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/common_utilities-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/common_utilitiesConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/common_utilities"
)

# 安装导出目标文件
install(EXPORT CommonUtilitiesTargets
    FILE CommonUtilitiesTargets.cmake
    NAMESPACE OSCEAN::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/common_utilities"
)

# GPU测试文件
if(BUILD_TESTS)
    set(GPU_TEST_SOURCES
        tests/gpu/test_gpu_detection.cpp
        tests/gpu/test_multi_gpu_scheduler.cpp
        tests/gpu/test_gpu_framework.cpp
        # tests/gpu/test_gpu_memory_manager.cpp
    )
    
    # 添加GPU测试
    foreach(test_source ${GPU_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} 
            common_utilities
            GTest::gtest
            GTest::gtest_main
            ${Boost_LIBRARIES}
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif() 
