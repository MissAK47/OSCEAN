cmake_minimum_required(VERSION 3.17)

# 🚀 现代化CMake配置 - 无后向兼容负担
project(CrsService VERSION 1.0.0 LANGUAGES CXX)

message(STATUS "[CRS Service] 🚀 Configuring Modern CRS Service Module v1.0")

# 🎯 C++17标准 + 现代化优化（与主项目保持一致）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === 🚀 高性能编译优化 ===
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
    # SIMD优化标志
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    endif()
endif()

# 设置CMake策略
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# === 🔧 修复依赖传递 - 使用顶层配置的依赖 ===

# 🎯 从顶层继承依赖，不重复查找
# PROJ, GDAL, Boost, spdlog 等依赖由顶层CMakeLists.txt管理

# 验证关键依赖是否已在顶层找到
if(NOT TARGET PROJ::proj)
    message(FATAL_ERROR "[CRS Service] ❌ PROJ::proj target not found. Please ensure PROJ is found in top-level CMakeLists.txt")
endif()

if(NOT TARGET GDAL::GDAL)
    message(FATAL_ERROR "[CRS Service] ❌ GDAL::GDAL target not found. Please ensure GDAL is found in top-level CMakeLists.txt")
endif()

if(NOT TARGET spdlog::spdlog)
    message(FATAL_ERROR "[CRS Service] ❌ spdlog::spdlog target not found. Please ensure spdlog is found in top-level CMakeLists.txt")
endif()

message(STATUS "[CRS Service] ✅ Using dependencies from top-level configuration")
message(STATUS "[CRS Service] ✅ PROJ target: PROJ::proj")
message(STATUS "[CRS Service] ✅ GDAL target: GDAL::GDAL")
message(STATUS "[CRS Service] ✅ spdlog target: spdlog::spdlog")

set(TARGET_NAME crs_service)

# === 现代化源文件配置 ===

set(CRS_SERVICE_SOURCES
    # 核心实现 - 只保留优化版本
    src/impl/optimized_crs_service_impl.cpp
    src/impl/transformation_cache_adapter.h
    src/impl/transformation_cache_impl.cpp
    
    # 🔧 架构修复：智能GDAL管理器
    src/impl/gdal_manager.cpp
    
    # 服务组件
    src/crs_service_factory.cpp
    src/crs_transformer.cpp
    src/crs_inspector.cpp
    src/crs_parser.cpp
)

set(CRS_SERVICE_HEADERS
    # 公共接口
    include/core_services/crs/crs_service_factory.h
    include/core_services/crs/internal/crs_service_extended.h
    
    # 优化实现
    src/impl/optimized_crs_service_impl.h
    src/impl/transformation_cache_adapter.h
    src/impl/transformation_cache.h
    src/impl/transformation_cache_pimpl.h
    
    # 🔧 架构修复：智能GDAL管理器
    src/impl/gdal_manager.h
    
    # 内部组件
    src/crs_transformer.h
    src/crs_inspector.h
    src/crs_parser.h
)

# 创建现代化静态库
add_library(${TARGET_NAME} STATIC ${CRS_SERVICE_SOURCES} ${CRS_SERVICE_HEADERS})

# MSVC: 启用UTF-8支持并禁用特定警告
if(MSVC)
    # 移除 /utf-8 以遵循顶层配置
    # add_compile_options(/utf-8) 
    add_compile_options(/W4 /WX- /Zc:preprocessor /permissive- /Zc:__cplusplus /Zc:throwingNew)
    add_compile_options(/wd4251 /wd4275 /wd4996)
endif()

# === 现代化编译定义 ===

target_compile_definitions(${TARGET_NAME} PUBLIC 
    OSCEAN_CRS_VERSION_MAJOR=1
    OSCEAN_CRS_VERSION_MINOR=0
    OSCEAN_CRS_VERSION_PATCH=0
    
    # 🚀 启用现代化功能
    OSCEAN_CRS_MODERN_ARCH=1
    OSCEAN_CRS_SIMD_ENABLED=1
    OSCEAN_CRS_PERFORMANCE_MONITORING=1
    OSCEAN_CRS_STREAMING_ENABLED=1
    
    # 外部库支持
    OSCEAN_HAS_PROJ=1
)

# === 现代化依赖链接 ===

target_link_libraries(${TARGET_NAME}
    PUBLIC 
        # 🎯 核心接口依赖
        core_service_interfaces
        
        # 🚀 Common模块现代化集成
        common_utilities
        
        # 📍 地理空间库 - 只使用PROJ进行坐标转换
        PROJ::proj 
        
        # 🚀 现代化日志系统
        spdlog::spdlog
        
        # Boost库（异步框架）
        Boost::system
        Boost::thread
        Boost::chrono
        
    PRIVATE 
        # 🔧 Boost现代化组件
        Boost::system 
        Boost::thread 
        Boost::exception
)

# === 现代化包含目录 ===

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
        $<INSTALL_INTERFACE:include>
        
        # 🎯 核心服务接口
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/core_service_interfaces/include>
        
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl
)

# === 现代化编译选项 ===

if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE 
        /EHsc           # C++异常处理
        /permissive-    # 严格标准符合性
        /Zc:__cplusplus # 正确的__cplusplus宏
        
        # 🚀 现代化优化选项
        $<$<CONFIG:Release>:/O2>        # 最大速度优化
        $<$<CONFIG:Release>:/Ot>        # 优先速度
        $<$<CONFIG:Release>:/GL>        # 全程序优化
        $<$<CONFIG:Release>:/favor:AMD64> # x64优化
        
        # 🛡️ 安全性增强
        /GS             # 缓冲区安全检查
        /sdl            # 额外安全检查
        /arch:AVX2       # SIMD优化
    )
    
    target_compile_definitions(${TARGET_NAME} PRIVATE
        _WIN32_WINNT=0x0A00     # Windows 10+
        NOMINMAX                # 禁用min/max宏
        WIN32_LEAN_AND_MEAN     # 减少包含
        _CRT_SECURE_NO_WARNINGS # 抑制安全警告
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        
        # 🚀 现代化优化选项
        $<$<CONFIG:Release>:-O3>        # 最大优化
        $<$<CONFIG:Release>:-march=native> # 本机优化
        $<$<CONFIG:Release>:-flto>      # 链接时优化
        -mavx2 -mfma                    # SIMD优化
    )
endif()

# === 现代化目标属性 ===

set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    
    # 输出目录
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    
    # 版本信息
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    
    # 现代化输出名称
    OUTPUT_NAME "crs_service"
    
    # 🚀 现代化符号可见性
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# === 🧪 启用测试 ===
enable_testing()

# 🧪 添加测试子目录
if(BUILD_TESTING OR CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_subdirectory(tests)
endif()

# === 现代化安装配置 ===

include(GNUInstallDirs)

install(TARGETS ${TARGET_NAME}
    EXPORT CrsServiceTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# 现代化配置文件
install(EXPORT CrsServiceTargets
    FILE CrsServiceTargets.cmake
    NAMESPACE Oscean::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CrsService
)

message(STATUS "[CRS Service] ✅ Configuration complete - Modern high-performance CRS service ready!")

# === 现代化特性报告 ===

message(STATUS "[CRS Service] 🎯 Modern CRS Service Configuration Complete")
message(STATUS "  📊 Version: ${PROJECT_VERSION}")
message(STATUS "  🚀 C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  🔧 PROJ Support: ${PROJ_VERSION}")
message(STATUS "  🗺️ 专注坐标转换，无数据读取功能")
message(STATUS "  ⚡ Boost Support: ${Boost_VERSION}")
message(STATUS "  📝 Logging: spdlog ${spdlog_VERSION}")
message(STATUS "  🧪 Testing: ${BUILD_TESTING}")
message(STATUS "  🎭 Build Type: ${CMAKE_BUILD_TYPE}")

# 功能特性报告
message(STATUS "  🎯 Modern Features Enabled:")
message(STATUS "    ✅ SIMD Optimization")
message(STATUS "    ✅ Performance Monitoring")
message(STATUS "    ✅ Streaming Processing")
message(STATUS "    ✅ Smart Caching")
message(STATUS "    ✅ Async/Future Support")
message(STATUS "    ✅ boost::future Support") 