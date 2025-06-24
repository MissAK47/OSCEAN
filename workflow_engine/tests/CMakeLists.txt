# workflow_engine tests CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# 查找必要的依赖
find_package(Boost REQUIRED COMPONENTS thread chrono system)

# === 解决spdlog链接冲突 ===
# 统一使用动态库版本，避免重复定义
find_package(spdlog CONFIG REQUIRED)
if(WIN32)
    # 在Windows上确保使用动态库版本
    set_target_properties(spdlog::spdlog PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "SPDLOG_SHARED_LIB"
    )
endif()


# 设置C++标准
# set_target_properties(test_real_ocean_data_processing PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED ON
#     CXX_EXTENSIONS OFF
# )

# set_target_properties(simple_metadata_test PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED ON
#     CXX_EXTENSIONS OFF
# )

# 检查元数据服务是否可用
if(TARGET metadata_service)
    set(METADATA_SERVICE_AVAILABLE TRUE)
    message(STATUS "测试程序: 元数据服务目标已存在")
else()
    set(METADATA_SERVICE_AVAILABLE FALSE)
    message(WARNING "测试程序: 元数据服务目标不存在")
endif()

# === 修复的链接配置 ===
# 首先链接核心依赖，确保spdlog不被重复链接
# target_link_libraries(test_real_ocean_data_processing PRIVATE
#     # 核心依赖 - 顺序很重要
#     common_utilities
#     workflow_engine_core
#     
#     # Boost库
#     Boost::thread
#     Boost::chrono
#     Boost::system
#     
#     # 显式链接spdlog，确保使用统一版本
#     spdlog::spdlog
# )

# 如果元数据服务可用，则链接它
# if(METADATA_SERVICE_AVAILABLE)
#     target_link_libraries(test_real_ocean_data_processing PRIVATE
#         metadata_service
#     )
#     target_compile_definitions(test_real_ocean_data_processing PRIVATE
#         METADATA_SERVICE_AVAILABLE=1
#     )
# endif()

# 添加SQLite3支持
if(TARGET SQLite::SQLite3)
    # target_link_libraries(test_real_ocean_data_processing PRIVATE
    #     SQLite::SQLite3
    # )
    # target_compile_definitions(test_real_ocean_data_processing PRIVATE
    #     SQLITE3_AVAILABLE=1
    # )
    message(STATUS "测试程序: 添加SQLite3支持")
else()
    message(WARNING "测试程序: SQLite3不可用，数据库功能将受限")
endif()

# 包含目录
# target_include_directories(test_real_ocean_data_processing PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../data_management/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../common_utilities/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../core_services_impl/metadata_service/include
# )

# === 修复的编译定义 ===
# target_compile_definitions(test_real_ocean_data_processing PRIVATE
#     BOOST_ALL_NO_LIB
#     BOOST_ALL_DYN_LINK
#     # 确保spdlog使用动态库
#     SPDLOG_SHARED_LIB
#     # 避免符号重复定义
#     SPDLOG_COMPILED_LIB
# )

# === 链接器特定设置修复spdlog重复定义问题 ===
# if(MSVC)
#     # 在MSVC下，强制忽略重复的符号定义警告并使用第一个遇到的符号
#     target_link_options(test_real_ocean_data_processing PRIVATE
#         /FORCE:MULTIPLE
#         /IGNORE:4006,4088
#     )
#     
#     # 设置spdlog使用动态库
#     target_compile_definitions(test_real_ocean_data_processing PRIVATE
#         SPDLOG_SHARED_LIB
#     )
# endif()

# 设置运行时输出目录
# set_target_properties(test_real_ocean_data_processing PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# 复制到测试运行目录
# install(TARGETS test_real_ocean_data_processing
#     RUNTIME DESTINATION bin
# )

# 如果需要，可以添加测试发现
if(BUILD_TESTING)
    # add_test(NAME RealOceanDataProcessingTest 
    #          COMMAND test_real_ocean_data_processing)
    
    # 设置测试属性
    # set_tests_properties(RealOceanDataProcessingTest PROPERTIES
    #     TIMEOUT 300  # 5分钟超时
    #     LABELS "integration;ocean_data"
    # )
    
    message(STATUS "测试已配置 - 临时注释掉缺失的测试文件")
endif()

find_package(GTest REQUIRED)

# 阶段1测试: 工厂注册模式和模块独立调用
add_executable(test_ice_thickness_workflow_stage1 
    test_ice_thickness_workflow_stage1.cpp
)

target_link_libraries(test_ice_thickness_workflow_stage1
    PRIVATE
    # 核心服务实现 - 使用正确的目标名称
    data_access_service      # 不是OSCEAN::data_access_service
    crs_service             # 不是OSCEAN::crs_service  
    spatial_ops_service     # 不是OSCEAN::spatial_ops_service
    
    # 通用工具
    common_utilities        # 不是OSCEAN::common_utilities
    
    # 测试框架
    GTest::gtest
    GTest::gtest_main
    
    # Boost库
    Boost::thread
    Boost::chrono
    Boost::system
)

target_include_directories(test_ice_thickness_workflow_stage1
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core_service_interfaces/include
    ${PROJECT_SOURCE_DIR}/core_services_impl
    ${PROJECT_SOURCE_DIR}/common_utilities/include
)

# 设置测试属性
set_target_properties(test_ice_thickness_workflow_stage1 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 添加编译定义
target_compile_definitions(test_ice_thickness_workflow_stage1
    PRIVATE
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

# 注册为测试
add_test(NAME IceThicknessWorkflowStage1 COMMAND test_ice_thickness_workflow_stage1)

# 创建快速测试目标
add_custom_target(run_ice_thickness_stage1_test
    COMMAND test_ice_thickness_workflow_stage1
    DEPENDS test_ice_thickness_workflow_stage1
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "运行冰厚度工作流阶段1测试"
)

# 设置测试工作目录
set_tests_properties(IceThicknessWorkflowStage1 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# === 新增：NetCDF Reader 直接测试 ===
# test_netcdf_reader_direct - 暂时移除，等文件创建后重新启用
# add_executable(test_netcdf_reader_direct
#     test_netcdf_reader_direct.cpp
# )

# 暂时注释掉test_netcdf_reader_direct的所有配置
# target_link_libraries(test_netcdf_reader_direct
#     PRIVATE
#     # 核心数据访问服务
#     data_access_service
#     
#     # 通用工具  
#     common_utilities
#     
#     # 测试框架
#     GTest::gtest
#     GTest::gtest_main
#     
#     # Boost库
#     Boost::thread
#     Boost::chrono
#     Boost::system
# )

# target_include_directories(test_netcdf_reader_direct
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${PROJECT_SOURCE_DIR}/core_service_interfaces/include
#     ${PROJECT_SOURCE_DIR}/core_services_impl
#     ${PROJECT_SOURCE_DIR}/common_utilities/include
# )

# # 设置测试属性
# set_target_properties(test_netcdf_reader_direct PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED ON
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# # 添加编译定义
# target_compile_definitions(test_netcdf_reader_direct
#     PRIVATE
#     BOOST_THREAD_PROVIDES_FUTURE=1
#     BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
#     BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
# )

# # 注册为测试
# add_test(NAME NetCDFReaderDirectTest COMMAND test_netcdf_reader_direct)

# # 创建快速测试目标
# add_custom_target(run_netcdf_reader_direct_test
#     COMMAND test_netcdf_reader_direct
#     DEPENDS test_netcdf_reader_direct
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#     COMMENT "运行NetCDF Reader直接测试"
# )

message(STATUS "配置冰厚度工作流测试 - 阶段1")

# ================================
# 修正版测试程序 - 解决重复测试、CF参数硬编码和析构问题
# ================================

add_executable(test_ice_thickness_workflow_stage1_fixed 
    test_ice_thickness_workflow_stage1_fixed.cpp
)

target_link_libraries(test_ice_thickness_workflow_stage1_fixed
    PRIVATE
    # 核心服务实现
    data_access_service
    crs_service
    spatial_ops_service
    
    # 通用工具
    common_utilities
    
    # 测试框架
    GTest::gtest
    GTest::gtest_main
    
    # Boost库
    Boost::thread
    Boost::chrono
    Boost::system
)

target_include_directories(test_ice_thickness_workflow_stage1_fixed
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core_service_interfaces/include
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common_utilities/include
)

# 设置测试属性
set_target_properties(test_ice_thickness_workflow_stage1_fixed PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 添加编译定义
target_compile_definitions(test_ice_thickness_workflow_stage1_fixed
    PRIVATE
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

# 注册为测试
add_test(NAME IceThicknessWorkflowStage1Fixed COMMAND test_ice_thickness_workflow_stage1_fixed)

# 创建快速测试目标
add_custom_target(run_ice_thickness_stage1_fixed_test
    COMMAND test_ice_thickness_workflow_stage1_fixed
    DEPENDS test_ice_thickness_workflow_stage1_fixed
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "运行冰厚度工作流阶段1修正版测试"
)

# 设置测试工作目录
set_tests_properties(IceThicknessWorkflowStage1Fixed PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "配置冰厚度工作流测试 - 修正版完成")

# 复制测试数据文件(如果存在)
if(EXISTS "${CMAKE_SOURCE_DIR}/test_data")
    file(COPY "${CMAKE_SOURCE_DIR}/test_data" 
         DESTINATION "${CMAKE_BINARY_DIR}")
endif()

message(STATUS "配置完所有测试目标")

# === 添加生产级ServiceManager测试程序 ===
add_executable(test_service_manager_production 
    test_service_manager_production.cpp
)

target_link_libraries(test_service_manager_production
    PRIVATE
    # 工作流引擎核心
    workflow_engine_core
    
    # 核心服务实现
    data_access_service
    metadata_service
    crs_service
    interpolation_service
    
    # 通用工具
    common_utilities
    core_service_interfaces
    
    # Boost库
    Boost::thread
    Boost::chrono
    Boost::system
)

target_include_directories(test_service_manager_production
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/workflow_engine/include
    ${PROJECT_SOURCE_DIR}/workflow_engine/src
    ${PROJECT_SOURCE_DIR}/core_service_interfaces/include
    ${PROJECT_SOURCE_DIR}/core_services_impl
    ${PROJECT_SOURCE_DIR}/common_utilities/include
)

# 设置测试属性
set_target_properties(test_service_manager_production PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 添加编译定义
target_compile_definitions(test_service_manager_production
    PRIVATE
    OSCEAN_NO_BOOST_ASIO_MODULE=1
    BOOST_THREAD_PROVIDES_FUTURE=1
    BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
    BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
    OSCEAN_HAS_FULL_SERVICES=1
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0601
)

message(STATUS "配置生产级ServiceManager测试程序")
