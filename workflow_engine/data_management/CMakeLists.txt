cmake_minimum_required(VERSION 3.20)
project(workflow_engine_data_management)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 🔧 修复字符编码问题 - 添加UTF-8编码支持
if(MSVC)
    # 添加UTF-8编码支持，解决中文字符编译问题
    add_compile_options(/utf-8)
    # 禁用特定警告
    add_compile_options(/wd4819)  # 禁用字符编码警告
    add_compile_options(/wd4996)  # 禁用已弃用函数警告
    # 添加Unicode支持
    add_compile_definitions(UNICODE _UNICODE)
endif()

# 🚀 强制禁用boost::asio - 防止命名空间冲突
add_compile_definitions(
    OSCEAN_NO_BOOST_ASIO_MODULE=1
    BOOST_ASIO_DISABLE_BOOST_ARRAY=1
    BOOST_ASIO_DISABLE_BOOST_ASSERT=1
    BOOST_ASIO_DISABLE_BOOST_BIND=1
    BOOST_ASIO_DISABLE_BOOST_CHRONO=1
    BOOST_ASIO_DISABLE_BOOST_DATE_TIME=1
    BOOST_ASIO_DISABLE_BOOST_REGEX=1
)

# 查找依赖
find_package(Boost REQUIRED COMPONENTS thread system chrono)
find_package(Threads REQUIRED)

# 🆕 添加SQLite3支持（用于数据库操作）
find_package(SQLite3 QUIET)
if(NOT SQLite3_FOUND)
    find_package(unofficial-sqlite3 CONFIG QUIET)
    if(unofficial-sqlite3_FOUND AND NOT TARGET SQLite::SQLite3)
        if(TARGET unofficial::sqlite3::sqlite3)
            add_library(SQLite::SQLite3 ALIAS unofficial::sqlite3::sqlite3)
            message(STATUS "[data_management] 创建 SQLite::SQLite3 别名")
        endif()
    endif()
endif()

if(TARGET SQLite::SQLite3)
    message(STATUS "[data_management] SQLite3 支持已启用")
    set(HAS_SQLITE3 TRUE)
else()
    message(WARNING "[data_management] SQLite3 未找到，数据库功能将受限")
    set(HAS_SQLITE3 FALSE)
endif()

# 🔧 查找和配置核心服务依赖
find_package(common_utilities CONFIG QUIET)
find_package(core_service_interfaces CONFIG QUIET)

# 🎯 在主build环境中，直接使用已构建的目标而不是查找包
# 检查是否在主build环境中（通过检查目标是否存在）
if(TARGET metadata_service)
    message(STATUS "找到metadata_service目标（主build环境）")
    set(HAS_METADATA_SERVICE TRUE)
else()
    message(STATUS "未找到metadata_service目标，尝试查找包")
    find_package(metadata_service CONFIG QUIET)
    set(HAS_METADATA_SERVICE ${metadata_service_FOUND})
endif()

if(TARGET data_access_service)
    message(STATUS "找到data_access_service目标（主build环境）")
    set(HAS_DATA_ACCESS_SERVICE TRUE)
else()
    message(STATUS "未找到data_access_service目标，尝试查找包")
    find_package(data_access_service CONFIG QUIET)
    set(HAS_DATA_ACCESS_SERVICE ${data_access_service_FOUND})
endif()

if(TARGET crs_service)
    message(STATUS "找到crs_service目标（主build环境）")
    set(HAS_CRS_SERVICE TRUE)
else()
    message(STATUS "未找到crs_service目标，尝试查找包")
    find_package(crs_service CONFIG QUIET)
    set(HAS_CRS_SERVICE ${crs_service_FOUND})
endif()

# 🆕 新增：检查插值服务
if(TARGET interpolation_service)
    message(STATUS "找到interpolation_service目标（主build环境）")
    set(HAS_INTERPOLATION_SERVICE TRUE)
else()
    message(STATUS "未找到interpolation_service目标，尝试查找包")
    find_package(interpolation_service CONFIG QUIET)
    set(HAS_INTERPOLATION_SERVICE ${interpolation_service_FOUND})
endif()

# 🆕 新增：检查空间计算服务
if(TARGET spatial_ops_service)
    message(STATUS "找到spatial_ops_service目标（主build环境）")
    set(HAS_SPATIAL_OPS_SERVICE TRUE)
else()
    message(STATUS "未找到spatial_ops_service目标，尝试查找包")
    find_package(spatial_ops_service CONFIG QUIET)
    set(HAS_SPATIAL_OPS_SERVICE ${spatial_ops_service_FOUND})
endif()

# 🆕 新增：检查输出服务
if(TARGET output_generation)
    message(STATUS "找到output_generation目标（主build环境）")
    set(HAS_OUTPUT_SERVICE TRUE)
else()
    message(STATUS "未找到output_generation目标，尝试查找包")
    find_package(output_generation CONFIG QUIET)
    set(HAS_OUTPUT_SERVICE ${output_generation_FOUND})
endif()

# 🎯 确定是否有完整服务支持
if(HAS_METADATA_SERVICE AND HAS_DATA_ACCESS_SERVICE)
    message(STATUS "找到必需的核心服务，启用完整功能")
    set(HAS_FULL_SERVICES TRUE)
else()
    message(STATUS "缺少核心服务实现，将仅编译基础功能")
    message(STATUS "  - metadata_service: ${HAS_METADATA_SERVICE}")
    message(STATUS "  - data_access_service: ${HAS_DATA_ACCESS_SERVICE}")
    set(HAS_FULL_SERVICES FALSE)
endif()

# 🆕 扩展服务支持检查
message(STATUS "扩展服务支持状态:")
message(STATUS "  - crs_service: ${HAS_CRS_SERVICE}")
message(STATUS "  - interpolation_service: ${HAS_INTERPOLATION_SERVICE}")
message(STATUS "  - spatial_ops_service: ${HAS_SPATIAL_OPS_SERVICE}")
message(STATUS "  - output_service: ${HAS_OUTPUT_SERVICE}")

if(HAS_INTERPOLATION_SERVICE AND HAS_SPATIAL_OPS_SERVICE AND HAS_OUTPUT_SERVICE)
    message(STATUS "✅ 所有扩展服务可用，启用增强数据处理功能")
    set(HAS_ENHANCED_SERVICES TRUE)
else()
    message(STATUS "⚠️ 部分扩展服务不可用，某些增强功能将被禁用")
    set(HAS_ENHANCED_SERVICES FALSE)
endif()

# 设置包含路径 - 使用统一的方式
set(INCLUDE_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 添加common_utilities包含路径
if(common_utilities_FOUND)
    get_target_property(COMMON_INCLUDE_DIRS common_utilities INTERFACE_INCLUDE_DIRECTORIES)
    if(COMMON_INCLUDE_DIRS)
        list(APPEND INCLUDE_PATHS ${COMMON_INCLUDE_DIRS})
    endif()
else()
    # 回退到本地路径
    list(APPEND INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../common_utilities/include")
endif()

# 添加core_service_interfaces包含路径
if(core_service_interfaces_FOUND)
    get_target_property(CORE_INTERFACES_INCLUDE_DIRS core_service_interfaces INTERFACE_INCLUDE_DIRECTORIES)
    if(CORE_INTERFACES_INCLUDE_DIRS)
        list(APPEND INCLUDE_PATHS ${CORE_INTERFACES_INCLUDE_DIRS})
    endif()
else()
    # 回退到本地路径
    list(APPEND INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces/include")
endif()

# 添加服务实现的包含路径
if(HAS_FULL_SERVICES)
    if(core_services_impl_FOUND)
        get_target_property(CORE_IMPL_INCLUDE_DIRS core_services_impl INTERFACE_INCLUDE_DIRECTORIES)
        if(CORE_IMPL_INCLUDE_DIRS)
            list(APPEND INCLUDE_PATHS ${CORE_IMPL_INCLUDE_DIRS})
        endif()
    else()
        # 添加独立服务的包含路径
        list(APPEND INCLUDE_PATHS 
            "${CMAKE_CURRENT_SOURCE_DIR}/../../core_services_impl/data_access_service/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../core_services_impl/data_access_service/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../core_services_impl/metadata_service/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../core_services_impl/metadata_service/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../core_services_impl/crs_service/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../core_services_impl/crs_service/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../core_services_impl/crs_service/src/impl"
        )
    endif()
endif()

# 🔧 查找spdlog包 - 解决链接冲突（可选）
# 在主build环境中，spdlog已通过common_utilities链接，无需独立链接
set(USE_SPDLOG_UNIFIED FALSE)  # 在主build环境中禁用独立spdlog链接

# 链接目录 - 仅在必要时使用
if(NOT common_utilities_FOUND OR NOT core_service_interfaces_FOUND)
    link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../common_utilities/Debug
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces/Debug
    )
endif()

# 源文件
set(SOURCES
    src/data_management_service.cpp
    src/data_management_workflow.cpp    # 🆕 符合工作流引擎标准的工作流类
    # src/data_management_workflow_registry.cpp    # 🆕 工作流自动注册 - 暂时移除，避免命名空间冲突
)

# 头文件
set(HEADERS
    include/workflow_engine/data_management/data_management_service.h
    include/workflow_engine/data_management/data_management_workflow.h    # 🆕 符合工作流引擎标准的工作流类
)

# 创建库
add_library(workflow_engine_data_management STATIC ${SOURCES} ${HEADERS})

# 设置包含目录
target_include_directories(workflow_engine_data_management PRIVATE ${INCLUDE_PATHS})

# 设置编译定义
target_compile_definitions(workflow_engine_data_management PRIVATE
    OSCEAN_DATA_MANAGEMENT_EXPORTS
    OSCEAN_NO_BOOST_ASIO_MODULE=1
    # 🔧 启用Boost.Thread的future扩展功能
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

# 根据是否有完整服务设置编译定义
if(HAS_FULL_SERVICES)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        OSCEAN_HAS_FULL_SERVICES=1  # 启用完整服务实现
    )
    message(STATUS "启用完整服务功能")
else()
    message(STATUS "仅启用基础功能")
endif()

# 🔧 解决spdlog链接冲突 - 统一使用共享库版本（如果可用）
if(USE_SPDLOG_UNIFIED)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        SPDLOG_SHARED_LIB  # 使用共享库版本
        SPDLOG_COMPILED_LIB  # 使用编译版本而非header-only
    )
endif()

# 🆕 添加SQLite支持定义
if(HAS_SQLITE3)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        OSCEAN_HAS_SQLITE3=1  # 启用SQLite数据库支持
    )
    message(STATUS "启用SQLite数据库功能")
endif()

# 🆕 根据扩展服务可用性设置编译定义
if(HAS_ENHANCED_SERVICES)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        OSCEAN_HAS_ENHANCED_SERVICES=1  # 启用增强服务实现
    )
    message(STATUS "启用增强服务功能（插值、空间计算、输出）")
endif()

# 🆕 单独设置各扩展服务的编译定义
if(HAS_INTERPOLATION_SERVICE)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        OSCEAN_HAS_INTERPOLATION_SERVICE=1
    )
    message(STATUS "启用插值服务支持")
endif()

if(HAS_SPATIAL_OPS_SERVICE)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        OSCEAN_HAS_SPATIAL_OPS_SERVICE=1
    )
    message(STATUS "启用空间计算服务支持")
endif()

if(HAS_OUTPUT_SERVICE)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        OSCEAN_HAS_OUTPUT_SERVICE=1
    )
    message(STATUS "启用输出服务支持")
endif()

if(HAS_CRS_SERVICE)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        OSCEAN_HAS_CRS_SERVICE=1
    )
    message(STATUS "启用CRS服务支持")
endif()

# 基础库链接
set(BASE_LIBRARIES
    ${Boost_LIBRARIES}
    Threads::Threads
    Boost::thread
    Boost::chrono
    Boost::system
)

# 🆕 添加SQLite3库（如果可用）
if(HAS_SQLITE3)
    list(APPEND BASE_LIBRARIES SQLite::SQLite3)
    message(STATUS "[data_management] 添加SQLite3库到链接")
endif()

# 核心依赖库链接
set(CORE_LIBRARIES)

if(common_utilities_FOUND)
    list(APPEND CORE_LIBRARIES common_utilities)
else()
    message(STATUS "使用本地common_utilities链接")
    list(APPEND CORE_LIBRARIES common_utilities)  # 假设通过link_directories可以找到
endif()

if(core_service_interfaces_FOUND)
    list(APPEND CORE_LIBRARIES core_service_interfaces)
else()
    message(STATUS "使用本地core_service_interfaces链接")
    list(APPEND CORE_LIBRARIES core_service_interfaces)  # 假设通过link_directories可以找到
endif()

# 完整服务库链接
set(SERVICE_LIBRARIES)
if(HAS_FULL_SERVICES)
    # 🎯 在主build环境中，直接链接到目标而不是通过包
    if(TARGET metadata_service)
        list(APPEND SERVICE_LIBRARIES metadata_service)
        message(STATUS "添加metadata_service目标到链接库")
    endif()
    
    if(TARGET data_access_service)
        list(APPEND SERVICE_LIBRARIES data_access_service)
        message(STATUS "添加data_access_service目标到链接库")
    endif()
    
    if(TARGET crs_service)
        list(APPEND SERVICE_LIBRARIES crs_service)
        message(STATUS "添加crs_service目标到链接库")
    endif()
    
    # 🆕 恢复spatial_ops_service链接 - workflow_engine_core现已正确链接
    if(TARGET spatial_ops_service)
        list(APPEND SERVICE_LIBRARIES spatial_ops_service)
        message(STATUS "添加spatial_ops_service目标到链接库")
    endif()
    
    if(TARGET interpolation_service)
        list(APPEND SERVICE_LIBRARIES interpolation_service)
        message(STATUS "添加interpolation_service目标到链接库")
    endif()
    
    if(TARGET output_generation)
        list(APPEND SERVICE_LIBRARIES output_generation)
        message(STATUS "添加output_generation目标到链接库")
    else()
        # 如果output_generation目标还不存在，添加库路径
        if(EXISTS "${CMAKE_BINARY_DIR}/output_generation/Debug/output_generation.lib")
            list(APPEND SERVICE_LIBRARIES "${CMAKE_BINARY_DIR}/output_generation/Debug/output_generation.lib")
            message(STATUS "添加output_generation库文件到链接库")
        endif()
    endif()
    
    # 如果没有找到目标，尝试通过包链接
    if(NOT SERVICE_LIBRARIES)
        if(data_access_service_FOUND)
            list(APPEND SERVICE_LIBRARIES data_access_service)
        endif()
        if(metadata_service_FOUND)
            list(APPEND SERVICE_LIBRARIES metadata_service)
        endif()
        if(crs_service_FOUND)
            list(APPEND SERVICE_LIBRARIES crs_service)
        endif()
    endif()
endif()

# 执行链接
target_link_libraries(workflow_engine_data_management
    ${BASE_LIBRARIES}
    ${CORE_LIBRARIES}
    ${SERVICE_LIBRARIES}
)

# 创建测试可执行文件
add_executable(test_data_management test_data_management.cpp)

# 🎯 设置测试程序输出到主build目录的bin目录，便于统一管理
set_target_properties(test_data_management PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
)

# 设置测试程序的包含目录
target_include_directories(test_data_management PRIVATE ${INCLUDE_PATHS})

# 🔧 测试可执行文件 - 使用相同的spdlog链接方式（如果可用）
if(USE_SPDLOG_UNIFIED)
    target_compile_definitions(test_data_management PRIVATE
        SPDLOG_SHARED_LIB  # 使用共享库版本
        SPDLOG_COMPILED_LIB  # 使用编译版本而非header-only
    )
endif()

# 设置测试程序的编译定义
target_compile_definitions(test_data_management PRIVATE
    OSCEAN_NO_BOOST_ASIO_MODULE=1
    # 🔧 启用Boost.Thread的future扩展功能
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

# 根据是否有完整服务设置测试程序的编译定义
if(HAS_FULL_SERVICES)
    target_compile_definitions(test_data_management PRIVATE
        OSCEAN_HAS_FULL_SERVICES=1  # 启用完整服务实现
    )
endif()

# 链接测试可执行文件 - 修复链接顺序
target_link_libraries(test_data_management
    workflow_engine_data_management
    workflow_engine_core  # 🆕 添加workflow_engine_core库以使用ServiceManagerImpl
    ${SERVICE_LIBRARIES}  # 添加服务库，确保符号可用
    ${CORE_LIBRARIES}     # 添加核心库
    ${BASE_LIBRARIES}     # 添加基础库
    # 注意：依赖会自动传递，但显式添加确保链接顺序正确
)

# 设置目标属性
set_target_properties(workflow_engine_data_management PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(test_data_management PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 🔧 Windows特定设置
if(WIN32)
    target_compile_definitions(workflow_engine_data_management PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0601
    )
    
    target_compile_definitions(test_data_management PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0601
    )
endif()

# 设置公共包含目录（用于其他模块链接此库时）
target_include_directories(workflow_engine_data_management PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 输出配置信息
message(STATUS "=== 数据管理模块配置总结 ===")
message(STATUS "完整服务支持: ${HAS_FULL_SERVICES}")
message(STATUS "Common工具包: ${common_utilities_FOUND}")
message(STATUS "核心服务接口: ${core_service_interfaces_FOUND}")
if(HAS_FULL_SERVICES)
    message(STATUS "核心服务实现: ${core_services_impl_FOUND}")
    message(STATUS "数据访问服务: ${data_access_service_FOUND}")
    message(STATUS "元数据服务: ${metadata_service_FOUND}")
    message(STATUS "CRS服务: ${crs_service_FOUND}")
endif()
message(STATUS "================================")

# === 数据管理模块配置总结 ===
# 完整服务支持: TRUE
# Common工具包: 0
# 核心服务接口: 0
# 核心服务实现:
# 数据访问服务:
# 元数据服务:
# CRS服务:
# ================================

# 🔧 添加服务管理器诊断程序
add_executable(service_manager_diagnostics service_manager_diagnostics.cpp)

# 🔬 添加完整元数据处理测试程序
# add_executable(test_complete_metadata_processing ${CMAKE_CURRENT_SOURCE_DIR}/../../test_complete_metadata_processing.cpp)
# target_link_libraries(test_complete_metadata_processing PRIVATE
#     data_management_core
#     GTest::gtest GTest::gtest_main
# )
# gtest_discover_tests(test_complete_metadata_processing)

# 🔍 添加CRS服务深度诊断测试程序
add_executable(crs_diagnostic_test crs_diagnostic_test.cpp)

# 🎯 设置诊断程序输出到主build目录的bin目录
set_target_properties(service_manager_diagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
)

# 🎯 设置完整元数据处理测试程序输出目录
# set_target_properties(test_complete_metadata_processing PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#     RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
#     RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
#     RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
#     RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
# )

# 🎯 设置CRS诊断测试程序输出目录
set_target_properties(crs_diagnostic_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
)

# 设置诊断程序的包含目录
target_include_directories(service_manager_diagnostics PRIVATE ${INCLUDE_PATHS})

# 设置完整元数据处理测试程序的包含目录
# target_include_directories(test_complete_metadata_processing PRIVATE ${INCLUDE_PATHS})

# 设置CRS诊断测试程序的包含目录
target_include_directories(crs_diagnostic_test PRIVATE ${INCLUDE_PATHS})

# 设置诊断程序的编译定义
target_compile_definitions(service_manager_diagnostics PRIVATE
    OSCEAN_NO_BOOST_ASIO_MODULE=1
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

# 设置完整元数据处理测试程序的编译定义
# target_compile_definitions(test_complete_metadata_processing PRIVATE
#     OSCEAN_NO_BOOST_ASIO_MODULE=1
#     BOOST_THREAD_PROVIDES_FUTURE=1
#     BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
#     BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
# )

# 设置CRS诊断测试程序的编译定义
target_compile_definitions(crs_diagnostic_test PRIVATE
    OSCEAN_NO_BOOST_ASIO_MODULE=1
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

if(HAS_FULL_SERVICES)
    target_compile_definitions(service_manager_diagnostics PRIVATE
        OSCEAN_HAS_FULL_SERVICES=1
    )
    # target_compile_definitions(test_complete_metadata_processing PRIVATE
    #     OSCEAN_HAS_FULL_SERVICES=1
    # )
    target_compile_definitions(crs_diagnostic_test PRIVATE
        OSCEAN_HAS_FULL_SERVICES=1
    )
endif()

# 链接诊断程序
target_link_libraries(service_manager_diagnostics
    workflow_engine_core
    ${SERVICE_LIBRARIES}
    ${CORE_LIBRARIES}
    ${BASE_LIBRARIES}
)

# 链接完整元数据处理测试程序
# target_link_libraries(test_complete_metadata_processing
#     workflow_engine_core
#     ${SERVICE_LIBRARIES}
#     ${CORE_LIBRARIES}
#     ${BASE_LIBRARIES}
# )

# 链接CRS诊断测试程序
target_link_libraries(crs_diagnostic_test
    workflow_engine_core
    ${SERVICE_LIBRARIES}
    ${CORE_LIBRARIES}
    ${BASE_LIBRARIES}
)

set_target_properties(service_manager_diagnostics PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# set_target_properties(test_complete_metadata_processing PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED ON
# )

set_target_properties(crs_diagnostic_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(WIN32)
    target_compile_definitions(service_manager_diagnostics PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0601
    )
endif()

# ===============================================================================
# 🆕 增强数据管理工作流测试
# ===============================================================================

# 创建增强数据管理工作流测试可执行文件
add_executable(test_enhanced_data_management test_enhanced_data_management.cpp)

# 设置测试程序输出到主build目录的bin目录
set_target_properties(test_enhanced_data_management PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
)

# 设置测试程序的包含目录
target_include_directories(test_enhanced_data_management PRIVATE ${INCLUDE_PATHS})

# 设置测试程序的编译定义
target_compile_definitions(test_enhanced_data_management PRIVATE
    OSCEAN_NO_BOOST_ASIO_MODULE=1
    # 启用Boost.Thread的future扩展功能
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

# 根据服务可用性设置编译定义
if(HAS_FULL_SERVICES)
    target_compile_definitions(test_enhanced_data_management PRIVATE
        OSCEAN_HAS_FULL_SERVICES=1
    )
endif()

if(HAS_ENHANCED_SERVICES)
    target_compile_definitions(test_enhanced_data_management PRIVATE
        OSCEAN_HAS_ENHANCED_SERVICES=1
    )
endif()

if(HAS_INTERPOLATION_SERVICE)
    target_compile_definitions(test_enhanced_data_management PRIVATE
        OSCEAN_HAS_INTERPOLATION_SERVICE=1
    )
endif()

if(HAS_SPATIAL_OPS_SERVICE)
    target_compile_definitions(test_enhanced_data_management PRIVATE
        OSCEAN_HAS_SPATIAL_OPS_SERVICE=1
    )
endif()

if(HAS_OUTPUT_SERVICE)
    target_compile_definitions(test_enhanced_data_management PRIVATE
        OSCEAN_HAS_OUTPUT_SERVICE=1
    )
endif()

if(HAS_CRS_SERVICE)
    target_compile_definitions(test_enhanced_data_management PRIVATE
        OSCEAN_HAS_CRS_SERVICE=1
    )
endif()

# 链接增强测试可执行文件
target_link_libraries(test_enhanced_data_management
    workflow_engine_data_management
    workflow_engine_core  # 添加workflow_engine_core库以使用ServiceManagerImpl
    ${SERVICE_LIBRARIES}  # 添加服务库，确保符号可用
    ${CORE_LIBRARIES}     # 添加核心库
    ${BASE_LIBRARIES}     # 添加基础库
)

# 显式链接output_generation（如果存在）
if(TARGET output_generation)
    target_link_libraries(test_enhanced_data_management output_generation)
    message(STATUS "显式链接output_generation到test_enhanced_data_management")
endif()

# 添加Boost::log支持（output_generation需要）
find_package(Boost REQUIRED COMPONENTS log)
if(Boost_LOG_FOUND)
    target_link_libraries(test_enhanced_data_management Boost::log)
    message(STATUS "添加Boost::log到test_enhanced_data_management")
endif()

# 添加最小线程池测试
add_executable(test_minimal_threadpool test_minimal_threadpool.cpp)
target_link_libraries(test_minimal_threadpool
    common_utilities
    ${Boost_LIBRARIES}
)
target_compile_definitions(test_minimal_threadpool PRIVATE
    OSCEAN_ENABLE_BOOST_ASIO
)

# 设置目标属性
set_target_properties(test_enhanced_data_management PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# ===============================================================================
# 📊 构建总结
# ===============================================================================

message(STATUS "=== 数据管理工作流构建配置总结 ===")
message(STATUS "✅ 主库: workflow_engine_data_management")
message(STATUS "✅ 基础测试: test_data_management")
message(STATUS "✅ 增强测试: test_enhanced_data_management")
message(STATUS "")
message(STATUS "📋 服务支持状态:")
message(STATUS "  - 核心服务: ${HAS_FULL_SERVICES}")
message(STATUS "  - 扩展服务: ${HAS_ENHANCED_SERVICES}")
message(STATUS "  - 插值服务: ${HAS_INTERPOLATION_SERVICE}")
message(STATUS "  - 空间计算: ${HAS_SPATIAL_OPS_SERVICE}")
message(STATUS "  - 输出服务: ${HAS_OUTPUT_SERVICE}")
message(STATUS "  - CRS服务: ${HAS_CRS_SERVICE}")
message(STATUS "")
message(STATUS "🎯 输出目录: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "============================================")

# 统一服务管理测试
# ==============================================================================

# 删除旧的、分散的测试目标
# remove_executable(test_enhanced_data_management)

# 创建统一服务管理测试
add_executable(test_unified_service_management test_unified_service_management.cpp)

target_link_libraries(test_unified_service_management
    workflow_engine_data_management
    workflow_engine_core
    ${SERVICE_LIBRARIES}
    ${CORE_LIBRARIES}
    ${BASE_LIBRARIES}
)

target_compile_definitions(test_unified_service_management PRIVATE
    ${BASE_DEFINITIONS}
)

# 显式链接output_generation（如果存在）
if(TARGET output_generation)
    target_link_libraries(test_unified_service_management output_generation)
    message(STATUS "显式链接output_generation到test_unified_service_management")
endif()

# 添加Boost::log支持
if(Boost_LOG_FOUND)
    target_link_libraries(test_unified_service_management Boost::log)
    message(STATUS "添加Boost::log到test_unified_service_management")
endif()

# 关键修复：将配置文件复制到 build/config 目录
add_custom_command(
    TARGET test_unified_service_management POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/config" # 源目录
        "${CMAKE_BINARY_DIR}/config" # 目标目录
    COMMENT "将配置文件复制到主构建输出目录"
)

