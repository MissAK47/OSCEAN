# 元数据服务真实功能测试
cmake_minimum_required(VERSION 3.20)

# 如果不是子项目，设置项目名称
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(metadata_service_tests LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === 🔍 查找依赖包 ===

# Google Test
find_package(GTest REQUIRED)

# Boost（用于异步操作）
find_package(Boost REQUIRED COMPONENTS 
    system 
    thread 
    filesystem
    chrono
)

# 添加缺少的依赖包
find_package(spdlog CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# 使用与顶层一致的SQLite3查找逻辑
if(NOT TARGET SQLite::SQLite3)
    message(STATUS "[metadata_service/tests] 正在查找 SQLite3...")
    find_package(SQLite3 QUIET)
    if(NOT SQLite3_FOUND)
        message(STATUS "[metadata_service/tests] 标准 SQLite3 未找到，尝试 vcpkg 的 unofficial-sqlite3...")
        find_package(unofficial-sqlite3 CONFIG REQUIRED)
        message(STATUS "[metadata_service/tests] 找到 unofficial-sqlite3")
        # 创建 SQLite::SQLite3 别名指向 unofficial::sqlite3::sqlite3
        add_library(SQLite::SQLite3 ALIAS unofficial::sqlite3::sqlite3)
    endif()
else()
    message(STATUS "[metadata_service/tests] SQLite::SQLite3 target already available")
endif()

# === 📁 包含目录 ===

# 元数据服务相关头文件
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../core_service_interfaces/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common_utilities/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../core_services_impl/data_access_service/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# === 🧪 定义真实功能测试 ===

# 🔧 添加配置文件加载真实功能测试
add_executable(config_loading_test
    unit/config_loading_test.cpp
)

# 添加MSVC特定的编译选项
if(MSVC)
    target_compile_options(config_loading_test PRIVATE
        /EHsc           # 异常处理
        /W3             # 警告级别
        /utf-8          # UTF-8编码支持
        /wd4819         # 忽略编码警告
        /wd4996         # 忽略弃用警告
        /bigobj         # 大对象文件支持
    )
    
    # 添加预处理器定义
    target_compile_definitions(config_loading_test PRIVATE
        _CRT_SECURE_NO_WARNINGS
        UNICODE
        _UNICODE
        NOMINMAX
        BOOST_ALL_NO_LIB
        BOOST_THREAD_USE_LIB
    )
else()
    # 非MSVC编译器的定义
    target_compile_definitions(config_loading_test PRIVATE
        BOOST_ALL_NO_LIB
        BOOST_THREAD_USE_LIB
    )
endif()

# 链接库
target_link_libraries(config_loading_test
    GTest::gtest
    GTest::gtest_main
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    metadata_service
    common_utilities
    core_service_interfaces
)

# 🔧 添加真实配置文件验证测试
add_executable(real_config_verification_test
    unit/real_config_verification_test.cpp
)

# 添加MSVC特定的编译选项
if(MSVC)
    target_compile_options(real_config_verification_test PRIVATE
        /EHsc           # 异常处理
        /W3             # 警告级别
        /utf-8          # UTF-8编码支持
        /wd4819         # 忽略编码警告
        /wd4996         # 忽略弃用警告
        /bigobj         # 大对象文件支持
    )
    
    # 添加预处理器定义
    target_compile_definitions(real_config_verification_test PRIVATE
        _CRT_SECURE_NO_WARNINGS
        UNICODE
        _UNICODE
        NOMINMAX
        BOOST_ALL_NO_LIB
        BOOST_THREAD_USE_LIB
    )
else()
    # 非MSVC编译器的定义
    target_compile_definitions(real_config_verification_test PRIVATE
        BOOST_ALL_NO_LIB
        BOOST_THREAD_USE_LIB
    )
endif()

# 链接库
target_link_libraries(real_config_verification_test
    GTest::gtest
    GTest::gtest_main
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    metadata_service
    common_utilities
    core_service_interfaces
)

# 🔧 添加真实NetCDF元数据测试
add_executable(real_netcdf_metadata_test
    unit/real_netcdf_metadata_test.cpp
)

# 添加MSVC特定的编译选项
if(MSVC)
    target_compile_options(real_netcdf_metadata_test PRIVATE
        /EHsc           # 异常处理
        /W3             # 警告级别
        /utf-8          # UTF-8编码支持
        /wd4819         # 忽略编码警告
        /wd4996         # 忽略弃用警告
        /bigobj         # 大对象文件支持
    )
    
    # 添加预处理器定义
    target_compile_definitions(real_netcdf_metadata_test PRIVATE
        _CRT_SECURE_NO_WARNINGS
        UNICODE
        _UNICODE
        NOMINMAX
        BOOST_ALL_NO_LIB
        BOOST_THREAD_USE_LIB
    )
else()
    # 非MSVC编译器的定义
    target_compile_definitions(real_netcdf_metadata_test PRIVATE
        BOOST_ALL_NO_LIB
        BOOST_THREAD_USE_LIB
    )
endif()

# 链接库
target_link_libraries(real_netcdf_metadata_test
    metadata_service
    core_service_interfaces
    common_utilities
    GTest::gtest
    GTest::gtest_main
    Boost::system
    Boost::thread
    Boost::filesystem
    Boost::chrono
    spdlog::spdlog
)

# === 🔧 测试配置 ===

# 启用测试
enable_testing()

# 添加配置文件加载测试
add_test(
    NAME ConfigLoadingTests
    COMMAND config_loading_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 添加真实配置验证测试
add_test(
    NAME RealConfigVerificationTests
    COMMAND real_config_verification_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 添加真实数据处理测试
add_test(
    NAME RealDataProcessingTests
    COMMAND real_netcdf_metadata_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 设置测试超时和标签
set_tests_properties(ConfigLoadingTests PROPERTIES 
    TIMEOUT 300
    LABELS "config;functional"
)

set_tests_properties(RealConfigVerificationTests PROPERTIES 
    TIMEOUT 300
    LABELS "config;real_files;verification"
)

set_tests_properties(RealDataProcessingTests PROPERTIES 
    TIMEOUT 600
    LABELS "data_processing;real_files;functional"
)

# === 📝 测试报告 ===

# 创建测试输出目录
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_reports)

# 设置测试输出文件
set_tests_properties(ConfigLoadingTests PROPERTIES
    ENVIRONMENT "GTEST_OUTPUT=xml:${CMAKE_CURRENT_BINARY_DIR}/test_reports/config_loading_test.xml"
)

set_tests_properties(RealConfigVerificationTests PROPERTIES
    ENVIRONMENT "GTEST_OUTPUT=xml:${CMAKE_CURRENT_BINARY_DIR}/test_reports/real_config_verification_test.xml"
)

set_tests_properties(RealDataProcessingTests PROPERTIES
    ENVIRONMENT "GTEST_OUTPUT=xml:${CMAKE_CURRENT_BINARY_DIR}/test_reports/real_netcdf_metadata_test.xml"
)

# === 🎨 自定义测试目标 ===

# 快速测试（配置文件测试）
add_custom_target(metadata_quick_test
    COMMAND ${CMAKE_CTEST_COMMAND} -L "config" --output-on-failure
    DEPENDS config_loading_test real_config_verification_test
    COMMENT "运行配置文件相关测试"
)

# 完整测试（所有真实功能测试）
add_custom_target(metadata_full_test
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS config_loading_test real_config_verification_test real_netcdf_metadata_test
    COMMENT "运行所有真实功能测试"
)

# 清理测试数据
add_custom_target(clean_test_data
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/test_data
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/test_reports
    COMMENT "清理测试数据和报告"
)

# 添加工具目录
add_subdirectory(utils)

# 测试程序配置
if(BUILD_TESTING)
    message(STATUS "📋 元数据服务真实功能测试配置完成")
    message(STATUS "   - 配置加载测试: config_loading_test")
    message(STATUS "   - 配置验证测试: real_config_verification_test")
    message(STATUS "   - 数据处理测试: real_netcdf_metadata_test")
endif()

# 配置验证测试
add_executable(configuration_validation_test
    configuration_validation_test.cpp
)

target_link_libraries(configuration_validation_test
    PRIVATE
        metadata_service
        common_utilities
        core_service_interfaces
        GTest::gtest
        GTest::gtest_main
        SQLite::SQLite3
        Boost::thread
        Boost::system
        Boost::filesystem
        spdlog::spdlog
)

target_include_directories(configuration_validation_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_SOURCE_DIR}/core_service_interfaces/include
        ${CMAKE_SOURCE_DIR}/common_utilities/include
)

# 设置C++标准
target_compile_features(configuration_validation_test PRIVATE cxx_std_17)

# 添加boost::future所需的宏定义
target_compile_definitions(configuration_validation_test PRIVATE
    BOOST_THREAD_PROVIDES_FUTURE
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
    BOOST_THREAD_PROVIDES_FUTURE_ASYNC
)

# 添加MSVC特定的编译选项
if(MSVC)
    target_compile_options(configuration_validation_test PRIVATE
        /utf-8
        /wd4996
        /wd4251
        /FS
        /EHsc
    )
    
    target_compile_definitions(configuration_validation_test PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _WIN32_WINNT=0x0A00
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# 设置输出目录
set_target_properties(configuration_validation_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests"
)

# 添加到测试
add_test(NAME ConfigurationValidationTest COMMAND configuration_validation_test)

message(STATUS "[metadata_service] 配置验证测试已添加") 