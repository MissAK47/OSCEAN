cmake_minimum_required(VERSION 3.17)
project(crs_service_tests LANGUAGES CXX)

message(STATUS "[CRS Tests] 🧪 Configuring modern CRS service test suite")

# 🎯 C++17标准（与主项目保持一致）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === 🔍 依赖查找 ===

# GTest是必需的
find_package(GTest CONFIG REQUIRED)
if(NOT TARGET GTest::gtest_main OR NOT TARGET GTest::gmock)
    message(FATAL_ERROR "GTest targets not found. Ensure GTest is properly installed.")
endif()

# 确保核心依赖存在
if(NOT TARGET crs_service)
    message(FATAL_ERROR "crs_service target not found.")
endif()

if(NOT TARGET common_utilities) 
    message(FATAL_ERROR "common_utilities target not found.")
endif()

# === 📋 测试源文件 ===

set(TEST_SOURCES
    # 🔄 现代化测试套件（使用依赖注入，避免单例模式）
    modern_crs_service_tests.cpp          # 主要的现代化测试文件
    
    # 🎯 全面功能测试套件
    comprehensive_crs_tests.cpp           # 全面的CRS功能和性能测试
    
    # 🗺️ GDAL/OGR集成测试
    crs_gdal_integration_tests.cpp        # CRS与GDAL/OGR的集成测试
    
    # 📊 性能基准测试（可选）
    $<$<BOOL:${ENABLE_BENCHMARKS}>:benchmark_main.cpp>
    $<$<BOOL:${ENABLE_BENCHMARKS}>:crs_benchmarks.cpp>
)

# === 🎯 多个测试可执行文件 ===

# 基础现代化测试
set(MODERN_TEST_TARGET_NAME crs_service_tests)
add_executable(${MODERN_TEST_TARGET_NAME} 
    modern_crs_service_tests.cpp
)

# 全面功能测试
set(COMPREHENSIVE_TEST_TARGET_NAME crs_comprehensive_tests)
add_executable(${COMPREHENSIVE_TEST_TARGET_NAME} 
    comprehensive_crs_tests.cpp
)

# GDAL集成测试
set(GDAL_TEST_TARGET_NAME crs_gdal_integration_tests)
add_executable(${GDAL_TEST_TARGET_NAME} 
    crs_gdal_integration_tests.cpp
)

# PROJ库诊断测试
set(PROJ_DIAGNOSTICS_TARGET_NAME crs_proj_diagnostics)
add_executable(${PROJ_DIAGNOSTICS_TARGET_NAME} 
    crs_proj_diagnostics.cpp
)

# 大规模压力测试
set(STRESS_TEST_TARGET_NAME crs_stress_tests)
add_executable(${STRESS_TEST_TARGET_NAME} 
    crs_stress_tests.cpp
)

# 极地投影增强测试
set(POLAR_ENHANCED_TEST_TARGET_NAME crs_polar_enhanced_tests)
add_executable(${POLAR_ENHANCED_TEST_TARGET_NAME} 
    polar_projection_enhanced_tests.cpp
)

# 简化的NetCDF测试
set(SIMPLE_NETCDF_TEST_TARGET_NAME simple_netcdf_test)
add_executable(${SIMPLE_NETCDF_TEST_TARGET_NAME} 
    simple_netcdf_test.cpp
)

# === 🧪 CTest集成配置 ===
include(GoogleTest)

# === 🎯 为所有测试目标配置通用函数 ===

function(configure_crs_test_target TARGET_NAME)
    # === 📦 依赖库链接 ===
    target_link_libraries(${TARGET_NAME} PRIVATE
        # 🎯 被测试的库
        crs_service
        
        # 🚀 Common模块依赖
        common_utilities
        
        # 🔗 核心接口
        core_service_interfaces
        
        # 🧪 测试框架
        GTest::gtest_main
        GTest::gmock
        
        # 📍 地理空间库
        PROJ::proj
        GDAL::GDAL
        
        # 🚀 日志系统
        spdlog::spdlog
        
        # Boost异步支持
        Boost::system
        Boost::thread
        Boost::chrono
    )

    # === 📂 包含目录 ===
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src         # CRS内部源码
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/impl    # 实现细节
        ${CMAKE_CURRENT_SOURCE_DIR}/../include     # 公共接口
    )

    # === 🏗️ 编译选项 ===
    target_compile_definitions(${TARGET_NAME} PRIVATE
        # 🧪 测试环境定义
        OSCEAN_CRS_TESTING_MODE=1
        OSCEAN_TESTING_ENABLED=1
        
        # 🚀 启用所有现代化特性
        OSCEAN_CRS_SIMD_ENABLED=1
        OSCEAN_CRS_PERFORMANCE_MONITORING=1
        OSCEAN_CRS_STREAMING_ENABLED=1
        
        # 🔧 日志级别（测试时更详细）
        SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
    )

    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE 
            /EHsc /permissive- /FS              # 移除/utf-8避免冲突，添加/FS避免PDB冲突
            $<$<CONFIG:Debug>:/Od /RTC1>        # Debug优化
            $<$<CONFIG:Release>:/O2 /arch:AVX2> # Release优化
        )
        
        target_compile_definitions(${TARGET_NAME} PRIVATE
            _WIN32_WINNT=0x0A00 NOMINMAX WIN32_LEAN_AND_MEAN
            _CRT_SECURE_NO_WARNINGS
        )
    endif()

    # === 🎯 目标属性 ===
    set_target_properties(${TARGET_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        
        # 输出目录
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        
        # 测试执行属性
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # === 🧪 CTest集成 ===
    gtest_discover_tests(${TARGET_NAME}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# 配置所有测试目标
configure_crs_test_target(${MODERN_TEST_TARGET_NAME})
configure_crs_test_target(${COMPREHENSIVE_TEST_TARGET_NAME})
configure_crs_test_target(${GDAL_TEST_TARGET_NAME})
configure_crs_test_target(${PROJ_DIAGNOSTICS_TARGET_NAME})
configure_crs_test_target(${STRESS_TEST_TARGET_NAME})
configure_crs_test_target(${POLAR_ENHANCED_TEST_TARGET_NAME})
configure_crs_test_target(${SIMPLE_NETCDF_TEST_TARGET_NAME})

# === 为兼容性保留原始目标 ===
set(TEST_TARGET_NAME ${MODERN_TEST_TARGET_NAME})  # 兼容性别名

# === 📄 DLL复制（Windows）- 为所有测试目标 ===

if(WIN32)
    set(VCPKG_BIN_DIR "C:/Users/Administrator/vcpkg/installed/x64-windows/bin")
    set(VCPKG_DEBUG_BIN_DIR "C:/Users/Administrator/vcpkg/installed/x64-windows/debug/bin")
    
    set(OUTPUT_DEBUG_DIR "${CMAKE_BINARY_DIR}/bin/Debug")
    set(OUTPUT_RELEASE_DIR "${CMAKE_BINARY_DIR}/bin/Release")
    
    # 创建输出目录
    file(MAKE_DIRECTORY ${OUTPUT_DEBUG_DIR})
    file(MAKE_DIRECTORY ${OUTPUT_RELEASE_DIR})
    
    # 复制必需的DLL - 对所有测试目标应用
    set(ALL_TEST_TARGETS 
        ${MODERN_TEST_TARGET_NAME}
        ${COMPREHENSIVE_TEST_TARGET_NAME}
        ${GDAL_TEST_TARGET_NAME}
        ${PROJ_DIAGNOSTICS_TARGET_NAME}
        ${STRESS_TEST_TARGET_NAME}
        ${POLAR_ENHANCED_TEST_TARGET_NAME}
    )
    
    foreach(TARGET_NAME ${ALL_TEST_TARGETS})
        foreach(DLL_NAME "proj_9.dll" "gdal.dll")
            if(EXISTS "${VCPKG_BIN_DIR}/${DLL_NAME}")
                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_BIN_DIR}/${DLL_NAME}"
                    "${OUTPUT_RELEASE_DIR}/${DLL_NAME}"
                    COMMENT "Copying ${DLL_NAME} to release directory for ${TARGET_NAME}"
                )
                
                # Debug版本
                if(EXISTS "${VCPKG_DEBUG_BIN_DIR}/${DLL_NAME}")
                    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${VCPKG_DEBUG_BIN_DIR}/${DLL_NAME}"
                        "${OUTPUT_DEBUG_DIR}/${DLL_NAME}"
                        COMMENT "Copying debug ${DLL_NAME} to debug directory for ${TARGET_NAME}"
                    )
                else()
                    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${VCPKG_BIN_DIR}/${DLL_NAME}"
                        "${OUTPUT_DEBUG_DIR}/${DLL_NAME}"
                        COMMENT "Copying release ${DLL_NAME} to debug directory for ${TARGET_NAME}"
                    )
                endif()
            endif()
        endforeach()
    endforeach()
endif()

# === 📊 性能基准测试（可选）===

option(ENABLE_BENCHMARKS "Enable performance benchmarks" OFF)

if(ENABLE_BENCHMARKS)
    find_package(benchmark CONFIG)
    if(TARGET benchmark::benchmark)
        message(STATUS "[CRS Tests] 📊 Performance benchmarks enabled")
        
        # 为所有测试目标启用基准测试
        foreach(TARGET_NAME ${ALL_TEST_TARGETS})
            target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_BENCHMARKS=1)
            target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark)
        endforeach()
    else()
        message(WARNING "[CRS Tests] ⚠️ Google Benchmark not found, benchmarks disabled")
    endif()
endif()

message(STATUS "[CRS Tests] ✅ Modern test suite configuration complete!") 
message(STATUS "[CRS Tests] 🎯 Test targets created:")
message(STATUS "  📝 ${MODERN_TEST_TARGET_NAME} - Basic modern CRS tests")
message(STATUS "  🎯 ${COMPREHENSIVE_TEST_TARGET_NAME} - Comprehensive CRS test suite")
message(STATUS "  🗺️ ${GDAL_TEST_TARGET_NAME} - GDAL/OGR integration tests")
message(STATUS "  🎯 ${PROJ_DIAGNOSTICS_TARGET_NAME} - PROJ library diagnostics")
message(STATUS "  🎯 ${STRESS_TEST_TARGET_NAME} - Stress tests") 