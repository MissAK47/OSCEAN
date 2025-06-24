cmake_minimum_required(VERSION 3.15)

project(core_service_interfaces VERSION 0.1.0 LANGUAGES CXX)

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加编译选项
if(MSVC)
    add_compile_options(/W4 /wd4100 /wd4996)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置安装目录
include(GNUInstallDirs)

# 创建接口库（纯头文件库）
add_library(core_service_interfaces INTERFACE)

# 查找核心依赖
find_package(Boost CONFIG REQUIRED COMPONENTS thread asio)

# 🆕 可选高性能优化支持
option(OSCEAN_ENABLE_HIGH_PERFORMANCE_INTERFACES "Enable high-performance optimizations in interfaces" ON)

if(OSCEAN_ENABLE_HIGH_PERFORMANCE_INTERFACES)
    message(STATUS "[core_service_interfaces] High-performance optimizations enabled")
    target_compile_definitions(core_service_interfaces
        INTERFACE
            OSCEAN_HAS_HIGH_PERFORMANCE_INTERFACES
    )
    # 仅在需要时链接common_utilities
    target_link_libraries(core_service_interfaces
        INTERFACE
            common_utilities
    )
else()
    message(STATUS "[core_service_interfaces] High-performance optimizations explicitly disabled")
endif()

# 设置包含目录
target_include_directories(core_service_interfaces
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 链接接口依赖
target_link_libraries(core_service_interfaces
    INTERFACE
        Boost::thread # for boost::future
        Boost::asio   # for potential transitive dependencies or definitions needed by asio users
        common_utilities # 核心数据结构依赖
)

# 设置必要的编译定义
target_compile_definitions(core_service_interfaces
    INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
)

# 安装接口头文件
install(DIRECTORY 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/core_service_interfaces
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# 安装目标
install(TARGETS core_service_interfaces
    EXPORT core_service_interfaces_targets
)

# 导出目标
install(EXPORT core_service_interfaces_targets
    FILE core_service_interfaces-targets.cmake
    NAMESPACE OSCEAN::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/core_service_interfaces
)

# 创建配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/core_service_interfaces-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/core_service_interfaces
)

# 生成版本文件
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/core_service_interfaces-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# 安装配置文件
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/core_service_interfaces-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/core_service_interfaces-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/core_service_interfaces
)

# 添加简单测试项目（如果存在）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/simple_test" AND BUILD_TESTING AND BUILD_TESTS)
    add_subdirectory(simple_test)
endif()