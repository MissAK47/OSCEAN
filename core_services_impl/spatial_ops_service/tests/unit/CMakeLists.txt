cmake_minimum_required(VERSION 3.15)

# ===================================================================
# 空间服务单元测试套件 - 基于真实功能的测试
# ===================================================================

message(STATUS "🚀 配置空间服务单元测试套件...")

# 通用设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread filesystem)
find_package(GDAL CONFIG REQUIRED)
find_package(geos CONFIG REQUIRED)

# 通用包含目录
set(COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../core_service_interfaces/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common_utilities/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../crs_service/include
)

# 通用链接库
set(COMMON_LINK_LIBS
    spatial_ops_service
    common_utilities
    GTest::gtest
    GTest::gtest_main
    GDAL::GDAL
    GEOS::geos_c
    Boost::system
    Boost::thread
    Boost::filesystem
)

# ===================================================================
# 1. 几何引擎测试 (真实GEOS库测试)
# ===================================================================

add_executable(test_geometry_engine test_geometry_engine.cpp)
target_include_directories(test_geometry_engine PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_geometry_engine PRIVATE ${COMMON_LINK_LIBS})
target_compile_definitions(test_geometry_engine PRIVATE GEOS_FOUND=1)

gtest_discover_tests(test_geometry_engine
    TEST_PREFIX "GeometryEngine::"
    PROPERTIES LABELS "geometry;geos;core"
)

# ===================================================================
# 2. 栅格引擎测试 (真实GDAL库测试)
# ===================================================================

add_executable(test_raster_engine test_raster_engine.cpp)
target_include_directories(test_raster_engine PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_raster_engine PRIVATE ${COMMON_LINK_LIBS})
target_compile_definitions(test_raster_engine PRIVATE GDAL_FOUND=1)

gtest_discover_tests(test_raster_engine
    TEST_PREFIX "RasterEngine::"
    PROPERTIES LABELS "raster;gdal;core"
)

# ===================================================================
# 3. 空间工具测试 (数学和几何计算)
# ===================================================================

add_executable(test_spatial_utils test_spatial_utils.cpp)
target_include_directories(test_spatial_utils PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_spatial_utils PRIVATE ${COMMON_LINK_LIBS})

gtest_discover_tests(test_spatial_utils
    TEST_PREFIX "SpatialUtils::"
    PROPERTIES LABELS "utils;math;geometry"
)

# ===================================================================
# 4. 几何转换测试 (格式转换)
# ===================================================================

add_executable(test_geometry_converter test_geometry_converter.cpp)
target_include_directories(test_geometry_converter PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_geometry_converter PRIVATE ${COMMON_LINK_LIBS})

gtest_discover_tests(test_geometry_converter
    TEST_PREFIX "GeometryConverter::"
    PROPERTIES LABELS "converter;format;utils"
)

# ===================================================================
# 5. 坐标验证测试 (坐标有效性检查) - 🆕 支持CRS服务集成
# ===================================================================

add_executable(test_coordinate_validator test_coordinate_validator.cpp)
target_include_directories(test_coordinate_validator PRIVATE ${COMMON_INCLUDE_DIRS})

# 🆕 为坐标验证测试配置特殊的链接库列表（包含CRS服务）
set(COORDINATE_VALIDATOR_LIBS
    spatial_ops_service
    common_utilities
    GTest::gtest
    GTest::gtest_main
    GDAL::GDAL
    GEOS::geos_c
    Boost::system
    Boost::thread
    Boost::filesystem
)

# 🆕 添加CRS服务可用性检查和条件链接
if(TARGET crs_service)
    # 将CRS服务添加到链接库列表（放在common_utilities之后）
    list(APPEND COORDINATE_VALIDATOR_LIBS crs_service)
    target_compile_definitions(test_coordinate_validator PRIVATE OSCEAN_HAS_CRS_SERVICE=1)
    message(STATUS "✅ 坐标验证测试: CRS服务集成已启用")
else()
    target_compile_definitions(test_coordinate_validator PRIVATE OSCEAN_HAS_CRS_SERVICE=0)
    message(STATUS "⚠️ 坐标验证测试: CRS服务暂时禁用，使用基础验证模式")
endif()

target_link_libraries(test_coordinate_validator PRIVATE ${COORDINATE_VALIDATOR_LIBS})

gtest_discover_tests(test_coordinate_validator
    TEST_PREFIX "CoordinateValidator::"
    PROPERTIES LABELS "validator;coordinates;utils;crs"
)

# ===================================================================
# 6. 空间索引测试 (索引算法)
# ===================================================================

add_executable(test_spatial_indexes test_spatial_indexes.cpp)
target_include_directories(test_spatial_indexes PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_spatial_indexes PRIVATE ${COMMON_LINK_LIBS})

gtest_discover_tests(test_spatial_indexes
    TEST_PREFIX "SpatialIndexes::"
    PROPERTIES LABELS "indexes;algorithms;performance"
)

# ===================================================================
# 7. 综合集成测试 (多模块协作) - 暂时禁用，测试文件不存在
# ===================================================================

# add_executable(test_integration test_integration.cpp)
# target_include_directories(test_integration PRIVATE ${COMMON_INCLUDE_DIRS})
# target_link_libraries(test_integration PRIVATE ${COMMON_LINK_LIBS})

# gtest_discover_tests(test_integration
#     TEST_PREFIX "Integration::"
#     PROPERTIES LABELS "integration;end-to-end;workflow"
# )

# ===================================================================
# 测试数据文件复制
# ===================================================================

# 复制测试数据文件到构建目录
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../test_data")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../../../test_data
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
    )
    message(STATUS "📁 复制测试数据目录到: ${CMAKE_CURRENT_BINARY_DIR}/test_data")
else()
    message(WARNING "⚠️ 测试数据目录不存在: ${CMAKE_CURRENT_SOURCE_DIR}/../../../../test_data")
endif()

# ===================================================================
# 自定义测试目标
# ===================================================================

# 快速测试 (仅核心功能)
add_custom_target(run_spatial_quick_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L "core" --output-on-failure
    DEPENDS test_geometry_engine test_raster_engine test_spatial_utils test_geometry_converter test_coordinate_validator test_spatial_indexes
    COMMENT "🚀 运行空间服务快速核心测试..."
)

# 全量测试
add_custom_target(run_all_spatial_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_geometry_engine test_raster_engine test_spatial_utils test_geometry_converter test_coordinate_validator test_spatial_indexes
    COMMENT "🎯 运行空间服务全量测试..."
)

# 性能测试
add_custom_target(run_spatial_performance_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L "performance" --output-on-failure
    COMMENT "⚡ 运行空间服务性能相关测试..."
)

message(STATUS "✅ 单元测试套件配置完成")
message(STATUS "📊 测试目标: 4个测试模块")
message(STATUS "🎯 自定义目标: run_spatial_quick_tests, run_all_spatial_tests, run_spatial_performance_tests") 