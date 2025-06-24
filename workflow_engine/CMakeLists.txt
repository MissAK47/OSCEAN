cmake_minimum_required(VERSION 3.16)
project(WorkflowEngine VERSION 3.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core_service_interfaces/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common_utilities/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core_services_impl/metadata_service/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core_services_impl/data_access_service/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core_services_impl/crs_service/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core_services_impl/spatial_ops_service/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core_services_impl/interpolation_service/include
)

# 查找所需的库
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system container)

# 查找SQLite3以支持元数据服务
find_package(SQLite3 QUIET)
if(NOT SQLite3_FOUND)
    find_package(unofficial-sqlite3 CONFIG QUIET)
    if(unofficial-sqlite3_FOUND AND NOT TARGET SQLite::SQLite3)
        if(TARGET unofficial::sqlite3::sqlite3)
            add_library(SQLite::SQLite3 ALIAS unofficial::sqlite3::sqlite3)
        endif()
    endif()
endif()

# 首先检查是否存在common_utilities目标，如果不存在则添加
if(NOT TARGET common_utilities)
    # 添加common_utilities子目录
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../common_utilities common_utilities_build)
endif()

# 源文件 - 核心框架源文件
set(WORKFLOW_ENGINE_SOURCES
    src/workflow_engine.cpp
    # src/workflow_registry.cpp        # 🆕 工作流注册表 - 暂时移除，解决命名空间冲突
    proxies/src/core_service_proxy.cpp
    # src/workflow_types.cpp # 暂时移除，类型定义应在头文件中
    # src/workflow_base.cpp    # 暂时移除，待确认实现

    # 🆕 添加服务管理器实现
    src/service_management/service_manager_impl.cpp
)

# 创建静态库
add_library(workflow_engine_core STATIC "") # Initially empty, sources added below

# 链接库
target_link_libraries(workflow_engine_core PUBLIC
    # External dependencies
    Boost::headers
    Boost::thread
    Boost::system
    Boost::container

    # Internal project dependencies
    common_utilities
    core_service_interfaces
)

# 🆕 添加核心服务实现的条件链接 - 修复SpatialOpsServiceFactory链接错误
# 检查各服务目标是否存在并进行链接
if(TARGET data_access_service)
    target_link_libraries(workflow_engine_core PRIVATE data_access_service)
    message(STATUS "workflow_engine_core: 已链接 data_access_service")
endif()

if(TARGET metadata_service)
    target_link_libraries(workflow_engine_core PRIVATE metadata_service)
    message(STATUS "workflow_engine_core: 已链接 metadata_service")
endif()

if(TARGET crs_service)
    target_link_libraries(workflow_engine_core PRIVATE crs_service)
    message(STATUS "workflow_engine_core: 已链接 crs_service")
endif()

if(TARGET spatial_ops_service)
    target_link_libraries(workflow_engine_core PRIVATE spatial_ops_service)
    message(STATUS "workflow_engine_core: 已链接 spatial_ops_service")
endif()

if(TARGET interpolation_service)
    target_link_libraries(workflow_engine_core PRIVATE interpolation_service)
    message(STATUS "workflow_engine_core: 已链接 interpolation_service")
endif()

if(TARGET output_generation)
    target_link_libraries(workflow_engine_core PRIVATE output_generation)
    message(STATUS "workflow_engine_core: 已链接 output_generation")
endif()

# 如果找不到服务目标，尝试直接添加依赖项目
if(NOT TARGET spatial_ops_service)
    message(STATUS "workflow_engine_core: 未找到 spatial_ops_service 目标，尝试添加依赖项目")
    # 首先确保构建了spatial_ops_service
    if(NOT TARGET spatial_ops_service)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core_services_impl/spatial_ops_service core_services_impl/spatial_ops_service)
    endif()
    if(TARGET spatial_ops_service)
        target_link_libraries(workflow_engine_core PRIVATE spatial_ops_service)
        message(STATUS "workflow_engine_core: 成功链接 spatial_ops_service")
    endif()
endif()

# Add compile definitions for the library
if(WIN32)
    target_compile_definitions(workflow_engine_core PRIVATE BOOST_USE_WINDOWS_H)
endif()

# 包含目录
target_include_directories(workflow_engine_core PUBLIC
    # Current public interface - 使用生成器表达式
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    
    # Add proxies include directory - 使用生成器表达式
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/proxies/include>

    # Inherit include directories from dependencies
    $<TARGET_PROPERTY:common_utilities,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:core_service_interfaces,INTERFACE_INCLUDE_DIRECTORIES>
)

# --- 暂时注释掉所有测试目标，直到核心库可以编译 ---
# add_executable(workflow_test_existing tests/test_workflow_functionality.cpp)
# target_link_libraries(workflow_test_existing 
#     workflow_engine_core
#     Threads::Threads
#     Boost::thread
# )
# 
# add_executable(workflow_integration_test tests/test_integration_full.cpp)
# target_link_libraries(workflow_integration_test 
#     workflow_engine_core
#     Threads::Threads
#     Boost::thread
# )
# 
# add_executable(workflow_integration_with_services tests/test_integration_with_services.cpp)
# target_link_libraries(workflow_integration_with_services 
#     workflow_engine_core
#     data_access_service
#     crs_service
#     Threads::Threads
#     Boost::thread
# )
# 
# add_executable(workflow_complete_integration tests/test_complete_integration.cpp)
# target_link_libraries(workflow_complete_integration 
#     workflow_engine_core
#     data_access_service
#     crs_service
#     metadata_service
#     Threads::Threads
#     Boost::thread
# )
# ---------------------------------------------

# --- 🆕 工作流模块化架构 ---
# 各工作流作为独立的可选组件，不强制链接到core

# 1. 数据管理工作流（独立）
add_subdirectory(data_management)

# 2. 数据处理工作流（独立）
add_subdirectory(data_workflow)

# 3. 未来的其他工作流可以在这里添加
# add_subdirectory(spatial_analysis)   # 空间分析工作流
# add_subdirectory(time_series)        # 时间序列工作流  
# add_subdirectory(multi_fusion)       # 多源融合工作流

# 🔧 核心框架不再自动链接具体工作流
# 让用户和服务层按需选择和链接工作流
# target_link_libraries(workflow_engine_core PUBLIC workflow_engine_data_management)  # 移除自动链接

# Add the main workflow engine sources to the core library
target_sources(workflow_engine_core PRIVATE
    ${WORKFLOW_ENGINE_SOURCES}
)
# ------------------------------------

# 设置输出目录
set_target_properties(workflow_engine_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# --- 暂时注释掉所有测试目标的输出目录设置 ---
# set_target_properties(workflow_test_existing PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )
# 
# set_target_properties(workflow_integration_test PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )
# 
# set_target_properties(workflow_integration_with_services PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )
# 
# set_target_properties(workflow_complete_integration PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# === 添加测试目录 ===
# 确保测试目录总是被添加，便于独立测试
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# 🔧 核心修复：将 workflow_engine_core 添加到导出集
# 暂时注释掉，解决导出目标冲突问题
# install(TARGETS workflow_engine_core
#     EXPORT WorkflowEngineTargets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#     INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# --- 安装和导出 ---
# 确保所有公共头文件都包含在内
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 创建并安装CMake配置文件，以便其他项目可以使用find_package(workflow_engine)
# 暂时注释掉，解决导出目标冲突问题
# install(EXPORT WorkflowEngineTargets
#     FILE WorkflowEngineTargets.cmake
#     NAMESPACE oscean::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/workflow_engine
# )

# 暂时注释掉，解决导出目标冲突问题
# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/cmake/WorkflowEngineConfig.cmake.in
#     ${CMAKE_CURRENT_BINARY_DIR}/WorkflowEngineConfig.cmake
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/workflow_engine
#     PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
# )

# 创建版本文件
# 暂时注释掉，解决导出目标冲突问题
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/WorkflowEngineConfigVersion.cmake"
#     VERSION "1.0.0"
#     COMPATIBILITY SameMajorVersion
# )

# 安装配置文件和版本文件
# 暂时注释掉，解决导出目标冲突问题
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/WorkflowEngineConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/WorkflowEngineConfigVersion.cmake"
#     DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/workflow_engine"
# )