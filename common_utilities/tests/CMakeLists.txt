cmake_minimum_required(VERSION 3.20)
project(common_utilities_tests VERSION 2.0.0 LANGUAGES CXX)

# =============================================================================
# 📋 Common Utilities 测试架构
# =============================================================================

message(STATUS "[common_utilities/tests] 🎯 构建测试架构...")

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# 🔧 依赖管理
# =============================================================================

find_package(GTest CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system filesystem chrono)
find_package(GDAL CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

if(NOT TARGET common_utilities)
    message(FATAL_ERROR "common_utilities target not found. Make sure this is called from parent CMakeLists.txt")
endif()

message(STATUS "[common_utilities/tests] ✅ 依赖验证:")
message(STATUS "  📦 GTest: ${GTest_FOUND}")
message(STATUS "  📦 Boost: ${Boost_FOUND}")
message(STATUS "  📦 GDAL: ${GDAL_FOUND}")
message(STATUS "  📦 spdlog: ${spdlog_FOUND}")

# =============================================================================
# 🔧 DLL自动复制功能
# =============================================================================

# 🧹 自动清理旧DLL，防止Debug/Release混用
file(GLOB OLD_DLLS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/*.dll")
file(REMOVE ${OLD_DLLS})

function(setup_dll_copying target_name)
    if(WIN32)
        # 检测vcpkg安装目录
        if(DEFINED ENV{VCPKG_ROOT})
            set(VCPKG_ROOT $ENV{VCPKG_ROOT})
        elseif(EXISTS "C:/Users/Administrator/vcpkg")
            set(VCPKG_ROOT "C:/Users/Administrator/vcpkg")
        elseif(EXISTS "C:/vcpkg")
            set(VCPKG_ROOT "C:/vcpkg")
        else()
            message(WARNING "Could not find vcpkg installation")
            return()
        endif()
        set(VCPKG_TARGET_TRIPLET "x64-windows")
        set(NO_GDAL_TESTS "async_framework_tests" "unified_memory_manager_tests" "cache_strategies_tests")
        set(needs_gdal TRUE)
        if(target_name IN_LIST NO_GDAL_TESTS)
            set(needs_gdal FALSE)
        endif()

        # Debug 配置的 DLL 列表
        set(DEBUG_DLL_LIST
            "gtest.dll"
            "gtest_main.dll"
            "boost_thread-vc143-mt-gd-x64-1_87.dll"
            "boost_system-vc143-mt-gd-x64-1_87.dll"
            "boost_filesystem-vc143-mt-gd-x64-1_87.dll"
            "boost_chrono-vc143-mt-gd-x64-1_87.dll"
            "spdlogd.dll"
            "fmtd.dll"
        )

        # Release 配置的 DLL 列表
        set(RELEASE_DLL_LIST
            "gtest.dll"
            "gtest_main.dll"
            "boost_thread-vc143-mt-x64-1_87.dll"
            "boost_system-vc143-mt-x64-1_87.dll"
            "boost_filesystem-vc143-mt-x64-1_87.dll"
            "boost_chrono-vc143-mt-x64-1_87.dll"
            "spdlog.dll"
            "fmt.dll"
        )

        # 如果需要 GDAL，添加到相应的 DLL 列表
        if(needs_gdal)
            list(APPEND DEBUG_DLL_LIST
                "gdald.dll"
                "proj_9_d.dll"
                "zlibd1.dll"
                "libexpatd.dll"
            )
            list(APPEND RELEASE_DLL_LIST
                "gdal.dll"
                "proj_9.dll"
                "zlib1.dll"
                "libexpat.dll"
            )
        endif()

        # 为 Debug 配置设置 DLL 复制
        foreach(DLL ${DEBUG_DLL_LIST})
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/debug/bin/${DLL}"
                $<TARGET_FILE_DIR:${target_name}>
                COMMENT "Copying ${DLL} to debug directory for ${target_name}"
            )
        endforeach()

        # 为 Release 配置设置 DLL 复制
        foreach(DLL ${RELEASE_DLL_LIST})
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/bin/${DLL}"
                $<TARGET_FILE_DIR:${target_name}>
                COMMENT "Copying ${DLL} to release directory for ${target_name}"
            )
        endforeach()
    endif()
endfunction()

# =============================================================================
# 🔧 通用测试配置函数
# =============================================================================

function(create_test target_name source_file needs_gdal)
    # 创建可执行文件
    add_executable(${target_name} ${source_file})
    
    # 设置C++标准
    set_target_properties(${target_name} PROPERTIES 
        CXX_STANDARD 17 
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    # 包含目录
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    
    # 链接基础库
    target_link_libraries(${target_name} PRIVATE
        GTest::gtest_main
        GTest::gtest
        common_utilities
        Boost::thread
        Boost::system
        Boost::filesystem
        Boost::chrono
        spdlog::spdlog
    )
    
    # 可选链接GDAL
    if(needs_gdal AND GDAL_FOUND)
        target_link_libraries(${target_name} PRIVATE GDAL::GDAL)
    endif()
    
    # Windows特定配置
    if(WIN32 AND MSVC)
        target_compile_definitions(${target_name} PRIVATE
            _WIN32_WINNT=0x0A00
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        )
        target_compile_options(${target_name} PRIVATE /W4 /FS /Zc:__cplusplus /bigobj)
    endif()
    
    # 设置DLL复制
    setup_dll_copying(${target_name})
    
    message(STATUS "[测试配置] ✅ ${target_name} 配置完成")
endfunction()

# =============================================================================
# 🎯 核心测试定义
# =============================================================================

# 基础单元测试 (不需要GDAL)
create_test(common_unit_tests common_unit_tests.cpp FALSE)

# 集成验证测试 (需要GDAL)
create_test(integration_verification_test integration_verification_test.cpp TRUE)

# 异步框架测试 (不需要GDAL)
create_test(async_framework_tests async_framework_tests.cpp FALSE)

# SIMD管理器测试 (需要GDAL)
create_test(simd_manager_tests simd_manager_tests.cpp TRUE)

# 统一内存管理器测试 (不需要GDAL)
create_test(unified_memory_manager_tests unified_memory_manager_tests.cpp FALSE)

# 缓存系统测试 (不需要GDAL)
create_test(cache_strategies_tests cache_strategies_tests.cpp FALSE)

# 缓存系统核心测试 - 按照测试计划1.2 (不需要GDAL) - [FIXME] 暂时禁用，此测试使用的API已过时
# create_test(cache_system_core_tests test_cache_system_core.cpp FALSE)

# 大文件处理器测试 (需要GDAL)
create_test(large_file_processor_tests large_file_processor_tests.cpp TRUE)

# 性能基准测试 (需要GDAL)
create_test(performance_benchmark_test performance_benchmark_test.cpp TRUE)

# 复杂集成测试 (需要GDAL)
create_test(complex_integration_test complex_integration_test.cpp TRUE)

# =============================================================================
# 🧪 CTest集成
# =============================================================================

enable_testing()

# 注册测试
add_test(NAME CommonUnitTests COMMAND common_unit_tests)
add_test(NAME IntegrationVerification COMMAND integration_verification_test)
add_test(NAME AsyncFrameworkTests COMMAND async_framework_tests)
add_test(NAME SIMDManagerTests COMMAND simd_manager_tests)
add_test(NAME UnifiedMemoryManagerTests COMMAND unified_memory_manager_tests)
add_test(NAME CacheStrategiesTests COMMAND cache_strategies_tests)
add_test(NAME CacheSystemCoreTests COMMAND cache_system_core_tests)
add_test(NAME LargeFileProcessorTests COMMAND large_file_processor_tests)
add_test(NAME PerformanceBenchmark COMMAND performance_benchmark_test)
add_test(NAME ComplexIntegrationScenarios COMMAND complex_integration_test)

# 设置测试属性
set_tests_properties(
    CommonUnitTests IntegrationVerification
    PROPERTIES LABELS "unit;fast" TIMEOUT 120
)

set_tests_properties(
    AsyncFrameworkTests SIMDManagerTests UnifiedMemoryManagerTests CacheStrategiesTests CacheSystemCoreTests LargeFileProcessorTests
    PROPERTIES LABELS "unit;medium" TIMEOUT 180
)

set_tests_properties(
    PerformanceBenchmark ComplexIntegrationScenarios
    PROPERTIES LABELS "performance;slow" TIMEOUT 300
)

# =============================================================================
# 🚀 自定义目标
# =============================================================================

# 所有测试目标列表
set(ALL_TEST_TARGETS
    common_unit_tests
    integration_verification_test
    async_framework_tests
    simd_manager_tests
    unified_memory_manager_tests
    cache_strategies_tests
    cache_system_core_tests
    large_file_processor_tests
    performance_benchmark_test
    complex_integration_test
)

# 运行所有测试
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 4
    DEPENDS ${ALL_TEST_TARGETS}
    COMMENT "运行所有Common Utilities测试"
)

# 运行快速测试
add_custom_target(run_fast_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --label-regex "fast|medium"
    DEPENDS common_unit_tests integration_verification_test async_framework_tests simd_manager_tests unified_memory_manager_tests cache_strategies_tests cache_system_core_tests large_file_processor_tests
    COMMENT "运行快速和中等速度的测试"
)

# 运行性能测试
add_custom_target(run_performance_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --label-regex "performance"
    DEPENDS performance_benchmark_test complex_integration_test
    COMMENT "运行性能基准测试"
)

# 运行核心功能测试
add_custom_target(run_core_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --label-regex "unit"
    DEPENDS async_framework_tests simd_manager_tests unified_memory_manager_tests cache_strategies_tests cache_system_core_tests large_file_processor_tests
    COMMENT "运行核心功能测试"
)

# =============================================================================
# 📄 状态报告
# =============================================================================

list(LENGTH ALL_TEST_TARGETS test_count)
message(STATUS "[common_utilities/tests] ✅ 测试架构配置完成:")
message(STATUS "  🎯 核心测试数量: ${test_count}")
message(STATUS "  📦 DLL自动复制: 已配置 (vcpkg管理)")
message(STATUS "  🛠️ 自定义目标: run_all_tests, run_fast_tests, run_performance_tests, run_core_tests")
message(STATUS "  📊 测试覆盖: 异步|SIMD|内存|缓存|集成|性能")
message(STATUS "  🆕 新增测试: 统一内存管理器, 缓存系统策略, 大文件处理器")
message(STATUS "  ♻️ 架构优化: 并行功能已整合到异步框架中")

# =============================================================================
# 🎮 GPU测试
# =============================================================================

# 添加GPU测试子目录
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu)
    add_subdirectory(gpu)
    message(STATUS "  🎮 GPU测试: 已启用")
endif()