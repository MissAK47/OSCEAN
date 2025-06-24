# core_services_impl/CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
# project()命令应该在顶层CMakeLists.txt中定义，这里不需要重复

# 设置组件版本（与顶层项目保持一致）
set(CORE_SERVICES_IMPL_VERSION "1.0.0")

# 设置CMake策略以避免Boost查找警告
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)  # 使用新的Boost查找模块
endif()

# 查找所有依赖包
find_package(Boost 1.71.0 REQUIRED COMPONENTS system thread filesystem property_tree log program_options serialization date_time)
find_package(GDAL CONFIG QUIET)
if(NOT GDAL_FOUND)
    find_package(GDAL MODULE REQUIRED)
endif()

# SQLite3查找 - 与主项目保持一致，但设为可选
find_package(SQLite3 QUIET)
if(SQLite3_FOUND)
    message(STATUS "[core_services_impl] 找到标准 SQLite3")
else()
    message(STATUS "[core_services_impl] 标准 SQLite3 未找到，尝试 vcpkg 的 unofficial-sqlite3...")
    find_package(unofficial-sqlite3 CONFIG QUIET)
    
    if(unofficial-sqlite3_FOUND)
        message(STATUS "[core_services_impl] 找到 unofficial-sqlite3")
        # 创建兼容的 SQLite::SQLite3 目标别名
        if(NOT TARGET SQLite::SQLite3)
            if(TARGET unofficial::sqlite3::sqlite3)
                add_library(SQLite::SQLite3 ALIAS unofficial::sqlite3::sqlite3)
                message(STATUS "[core_services_impl] 创建 SQLite::SQLite3 别名")
            else()
                add_library(SQLite::SQLite3 INTERFACE IMPORTED)
                set_target_properties(SQLite::SQLite3 PROPERTIES
                    INTERFACE_LINK_LIBRARIES "unofficial::sqlite3::sqlite3"
                )
                message(STATUS "[core_services_impl] 创建 SQLite::SQLite3 接口库")
            endif()
        endif()
        set(SQLite3_FOUND TRUE)
    else()
        message(WARNING "[core_services_impl] 无法找到 SQLite3！某些功能将被禁用。")
        set(SQLite3_FOUND FALSE)
    endif()
endif()

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    find_package(nlohmann_json MODULE QUIET)
    if(NOT nlohmann_json_FOUND)
        message(WARNING "nlohmann_json not found. Some functionality may be limited.")
    endif()
endif()

# 设置全局编译选项 (可以由顶层继承，或在此处覆盖/添加)
# if(MSVC)
#     add_compile_options(/W4 /wd4100 /wd4996)
# else()
#     add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# 添加各服务实现子目录
add_subdirectory(crs_service)  # 首先构建crs_service，因为其他服务依赖它
add_subdirectory(data_access_service)
add_subdirectory(metadata_service)
add_subdirectory(interpolation_service)
add_subdirectory(spatial_ops_service)  # 启用空间服务

# 创建一个整合所有核心服务实现的 INTERFACE 库 (用于链接)
add_library(core_services_impl INTERFACE)

# 链接所有核心服务实现
target_link_libraries(core_services_impl INTERFACE
    crs_service
    data_access_service
    metadata_service
    interpolation_service
    spatial_ops_service  # 启用空间服务
)

# 设置所有核心服务实现的通用属性
set(CORE_SERVICES_TARGETS
    crs_service
    data_access_service
    metadata_service
    interpolation_service
    spatial_ops_service  # 启用空间服务
)

foreach(target ${CORE_SERVICES_TARGETS})
    if(TARGET ${target})
        set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
        set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
        set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)

        # 统一为所有核心服务目标设置干净的编译选项
        if(MSVC)
            target_compile_options(${target} PRIVATE 
                /W4 
                /WX- 
                /Zc:preprocessor 
                /permissive- 
                /Zc:__cplusplus 
                /Zc:throwingNew 
                /EHsc
                /wd4251 
                /wd4275 
                /wd4996
            )
        endif()
    endif()
endforeach()

# 安装所有核心服务实现
install(TARGETS 
    crs_service
    data_access_service
    metadata_service
    interpolation_service
    spatial_ops_service  # 启用空间服务
    EXPORT core_services_impl_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 统一使用一个导出集
# INTERFACE库的安装主要通过导出其目标
# 暂时注释掉，解决导出目标循环依赖问题
# install(TARGETS core_services_impl crs_service data_access_service metadata_service interpolation_service spatial_ops_service # 使用正确的目标名称
#     EXPORT CoreServicesImplTargets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )

# 导出包配置
# 暂时注释掉，解决导出目标循环依赖问题
# include(CMakePackageConfigHelpers)

# 创建并安装 CoreServicesImplConfigVersion.cmake
# 暂时注释掉，解决导出目标循环依赖问题
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/CoreServicesImplConfigVersion.cmake"
#     VERSION ${CORE_SERVICES_IMPL_VERSION} 
#     COMPATIBILITY SameMajorVersion
# )

# 创建 core_services_impl-config.cmake 文件
# 这个文件将由 configure_package_config_file 生成
# 需要一个 core_services_impl/cmake/config.cmake.in 模板文件

# 确保 cmake 目录存在
# 暂时注释掉，解决导出目标循环依赖问题
# file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 暂时注释掉，解决导出目标循环依赖问题
# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/CoreServicesImplConfig.cmake"
#     INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CoreServicesImpl"
#     PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
# )

# 安装配置文件和版本文件
# 暂时注释掉，解决导出目标循环依赖问题
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/CoreServicesImplConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/CoreServicesImplConfigVersion.cmake"
#     DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CoreServicesImpl"
# )

# 安装导出目标文件 CoreServicesImplTargets.cmake
# 暂时注释掉，解决导出目标循环依赖问题
# install(EXPORT CoreServicesImplTargets
#     FILE CoreServicesImplTargets.cmake
#     NAMESPACE OSCEAN::
#     DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CoreServicesImpl"
# )

# 注意：移除了全局include_directories调用，改为在各个目标中使用target_include_directories
# 这是现代CMake的最佳实践，避免全局污染包含路径 

# 🔧 修复：在同一项目中直接使用目标，无需find_package
# find_package(common_utilities REQUIRED CONFIG)  # 移除，因为在同一项目中
# find_package(core_service_interfaces REQUIRED CONFIG)  # 移除，因为在同一项目中
# find_package(workflow_engine REQUIRED CONFIG) # 移除，因为在同一项目中

# 添加所有子目录已在前面完成，不需要重复
# ... existing code ... 