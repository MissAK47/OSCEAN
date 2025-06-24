cmake_minimum_required(VERSION 3.16)

project(network_service VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 Boost 策略以避免警告
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# Windows 平台特定配置 - 简化为避免Boost.Asio冲突
if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0601)  # Windows 7 兼容
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
    # 移除所有会导致Boost.Asio冲突的宏定义
endif()

# 添加编译选项
if(MSVC)
    add_compile_options(/utf-8)  # 强制使用 UTF-8 编码
endif()

# Gather all source files
add_library(network_service STATIC
    src/http_connection.cpp
    src/listener.cpp
    src/network_server.cpp
    src/request_router.cpp
    src/network_runner_impl.cpp
)

# Include directories for this project
target_include_directories(network_service PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    src
)

# 添加到 application 的接口头文件目录
target_include_directories(network_service PRIVATE
    ${CMAKE_SOURCE_DIR}/application/data_workflow_app/include
)

# Find and link against dependencies
find_package(Boost REQUIRED COMPONENTS system thread) # Beast需要这些组件
find_package(nlohmann_json REQUIRED)

# 尝试查找内部项目依赖，如果在独立编译环境中找不到则跳过
find_package(common_utilities QUIET)
find_package(core_service_interfaces QUIET)
find_package(workflow_engine QUIET)

# 确保可以找到 workflow_engine 的头文件
find_package(workflow_engine_core QUIET)

# 基础依赖库
target_link_libraries(network_service PRIVATE
    Boost::system
    Boost::thread
    nlohmann_json::nlohmann_json
)

# Windows specific libraries
if(WIN32)
    target_link_libraries(network_service PRIVATE ws2_32)
endif()

# 只有在找到内部依赖时才链接
if(TARGET common_utilities)
    target_link_libraries(network_service PRIVATE common_utilities)
    message(STATUS "Found and linking common_utilities")
else()
    message(WARNING "common_utilities not found - network_service may need manual integration")
endif()

if(TARGET core_service_interfaces)
    target_link_libraries(network_service PRIVATE core_service_interfaces)
    message(STATUS "Found and linking core_service_interfaces")
else()
    message(WARNING "core_service_interfaces not found - network_service may need manual integration")
endif()

if(TARGET workflow_engine_core)
    target_link_libraries(network_service PRIVATE workflow_engine_core)
    message(STATUS "Found and linking workflow_engine_core")
else()
    message(WARNING "workflow_engine_core not found - network_service may need manual integration")
endif()

# 注意：在主项目构建环境中，不需要导出配置
# 导出配置仅用于独立使用或安装后的包发现 