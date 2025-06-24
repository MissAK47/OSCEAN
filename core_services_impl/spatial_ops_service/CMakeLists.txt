# core_services_impl/spatial_ops_service/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# 设置CMake策略以避免Boost查找警告
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)  # 使用新的Boost查找模块
endif()

# 设置项目名称和版本
project(spatial_ops_service VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库名称
set(SPATIAL_OPS_SERVICE_LIB_NAME "spatial_ops_service")

# 查找依赖包
find_package(Boost REQUIRED COMPONENTS system thread filesystem)

# 查找GDAL
find_package(GDAL CONFIG QUIET)
if(NOT GDAL_FOUND)
    find_package(GDAL MODULE REQUIRED)
endif()

# 查找GEOS
find_package(geos CONFIG QUIET)
if(NOT geos_FOUND)
    # 如果CONFIG模式失败，尝试MODULE模式
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GEOS geos)
    endif()
    
    if(NOT GEOS_FOUND)
        # 手动查找GEOS
        find_path(GEOS_INCLUDE_DIR geos_c.h
            PATHS 
                "${CMAKE_PREFIX_PATH}/include"
                "C:/Users/Administrator/vcpkg/installed/x64-windows/include"
                /usr/include /usr/local/include
            PATH_SUFFIXES geos)
        
        find_library(GEOS_C_LIBRARY NAMES geos_c
            PATHS 
                "${CMAKE_PREFIX_PATH}/lib"
                "C:/Users/Administrator/vcpkg/installed/x64-windows/lib"
                /usr/lib /usr/local/lib)
        
        find_library(GEOS_LIBRARY NAMES geos
            PATHS 
                "${CMAKE_PREFIX_PATH}/lib"
                "C:/Users/Administrator/vcpkg/installed/x64-windows/lib"
                /usr/lib /usr/local/lib)
        
        if(GEOS_INCLUDE_DIR AND GEOS_C_LIBRARY AND GEOS_LIBRARY)
            set(GEOS_FOUND TRUE)
            set(GEOS_INCLUDE_DIRS ${GEOS_INCLUDE_DIR})
            set(GEOS_LIBRARIES ${GEOS_C_LIBRARY} ${GEOS_LIBRARY})
            message(STATUS "Found GEOS manually:")
            message(STATUS "  GEOS_INCLUDE_DIRS: ${GEOS_INCLUDE_DIRS}")
            message(STATUS "  GEOS_LIBRARIES: ${GEOS_LIBRARIES}")
        endif()
    endif()
else()
    set(GEOS_FOUND TRUE)
    message(STATUS "Found GEOS via CONFIG mode")
endif()

if(NOT GEOS_FOUND)
    message(WARNING "GEOS not found. Some spatial operations may not be available.")
endif()

# 定义核心必要的源文件
set(CORE_SOURCES
    src/impl/spatial_ops_service_impl.cpp
    src/impl/spatial_ops_service_factory.cpp
    src/impl/spatial_config_manager.cpp
)

# 定义栅格处理源文件
set(RASTER_SOURCES
    src/engine/raster_engine.cpp
    src/raster/raster_clipping.cpp
    src/raster/raster_algebra.cpp
    src/raster/raster_statistics.cpp
    src/raster/raster_vectorization.cpp
)

# 定义几何处理源文件
set(GEOMETRY_SOURCES
    src/engine/geometry_engine.cpp
)

# 定义查询源文件
set(QUERY_SOURCES
    src/engine/query_engine.cpp
)

# 定义工具源文件
set(UTILS_SOURCES
    src/utils/spatial_utils.cpp
    src/utils/geometry_converter.cpp
    src/utils/coordinate_validator.cpp
)

# 定义算法支持源文件
set(ALGORITHM_SOURCES
    # src/algorithms/interpolation_spatial_support_impl.cpp  # 文件不存在，暂时注释
    src/algorithms/tile_spatial_support_impl.cpp
    src/algorithms/modeling_spatial_support_impl.cpp
)

# 定义引擎源文件
set(ENGINE_SOURCES
    src/engine/spatial_index_manager.cpp
    src/engine/spatial_parallel_coordinator.cpp
)

# 定义索引源文件
set(INDEX_SOURCES
    src/index/quad_tree_index.cpp
    src/index/grid_index.cpp
    src/index/r_tree_index.cpp
    src/index/index_builder.cpp
)

# 定义基础设施源文件（移除不存在的performance_monitor.cpp）
set(INFRASTRUCTURE_SOURCES
    # src/infrastructure/performance_monitor_ext.cpp
    # src/infrastructure/gdal_performance_manager.cpp
)

# 创建静态库
add_library(${SPATIAL_OPS_SERVICE_LIB_NAME} STATIC 
    ${CORE_SOURCES}
    ${RASTER_SOURCES}
    ${GEOMETRY_SOURCES}
    ${QUERY_SOURCES}
    ${UTILS_SOURCES}
    ${ALGORITHM_SOURCES}
    ${ENGINE_SOURCES}
    ${INDEX_SOURCES}
    ${INFRASTRUCTURE_SOURCES}
)

# 🎯 解决UTF-8编译选项冲突
if(MSVC)
    get_target_property(SPS_COMPILE_OPTIONS ${SPATIAL_OPS_SERVICE_LIB_NAME} COMPILE_OPTIONS)
    if(SPS_COMPILE_OPTIONS)
        string(REPLACE "/utf-8" "" SPS_COMPILE_OPTIONS "${SPS_COMPILE_OPTIONS}")
        set_target_properties(${SPATIAL_OPS_SERVICE_LIB_NAME} PROPERTIES COMPILE_OPTIONS "${SPS_COMPILE_OPTIONS}")
        message(STATUS "[Spatial Ops Service] Removed conflicting /utf-8 option to defer to top-level setting.")
    endif()
endif()

# 设置库的别名
add_library(OSCEAN::${SPATIAL_OPS_SERVICE_LIB_NAME} ALIAS ${SPATIAL_OPS_SERVICE_LIB_NAME})

# 包含目录
target_include_directories(${SPATIAL_OPS_SERVICE_LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common_utilities/include
)

# 链接依赖库
target_link_libraries(${SPATIAL_OPS_SERVICE_LIB_NAME}
    PUBLIC
        GDAL::GDAL
        Boost::system
        Boost::thread
        Boost::filesystem
    PRIVATE
        common_utilities
)

# 如果找到GEOS，添加GEOS支持
if(GEOS_FOUND)
    if(TARGET GEOS::geos_c)
        # 使用vcpkg提供的CMake目标
        target_link_libraries(${SPATIAL_OPS_SERVICE_LIB_NAME} PRIVATE GEOS::geos_c)
        message(STATUS "Linking GEOS via CMake target: GEOS::geos_c")
    elseif(TARGET geos_c)
        # 备选目标名称
        target_link_libraries(${SPATIAL_OPS_SERVICE_LIB_NAME} PRIVATE geos_c)
        message(STATUS "Linking GEOS via CMake target: geos_c")
    elseif(GEOS_LIBRARIES)
        # 使用手动查找的库
        target_include_directories(${SPATIAL_OPS_SERVICE_LIB_NAME} PRIVATE ${GEOS_INCLUDE_DIRS})
        target_link_libraries(${SPATIAL_OPS_SERVICE_LIB_NAME} PRIVATE ${GEOS_LIBRARIES})
        message(STATUS "Linking GEOS via manual libraries: ${GEOS_LIBRARIES}")
    endif()
    
    # 添加GEOS编译定义
    target_compile_definitions(${SPATIAL_OPS_SERVICE_LIB_NAME} PRIVATE GEOS_FOUND=1)
endif()

# 添加编译定义
target_compile_definitions(${SPATIAL_OPS_SERVICE_LIB_NAME}
    PRIVATE
        GDAL_FOUND=1
        BOOST_THREAD_PROVIDES_FUTURE=1
        BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION=1
        BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY=1
)

# 设置编译选项
if(MSVC)
    target_compile_options(${SPATIAL_OPS_SERVICE_LIB_NAME} PRIVATE
        /W3
        /wd4996  # 禁用已弃用函数警告
        /wd4251  # 禁用DLL接口警告
    )
else()
    target_compile_options(${SPATIAL_OPS_SERVICE_LIB_NAME} PRIVATE
        -Wall
        -Wextra
        -Wno-deprecated-declarations
    )
endif()

# 安装配置
install(TARGETS ${SPATIAL_OPS_SERVICE_LIB_NAME}
    EXPORT SpatialOpsServiceTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# ===================================================================
# 测试配置
# ===================================================================

# 启用测试并检查依赖
if(BUILD_TESTING)
    message(STATUS "🧪 [空间服务] 启用测试套件")
    
    # 查找Google Test
    find_package(GTest REQUIRED)
    
    if(GTest_FOUND)
        message(STATUS "✅ [空间服务] 找到 Google Test")
        
        # 检查测试目录是否存在
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
            message(STATUS "📁 [空间服务] 添加测试目录")
            add_subdirectory(tests)
        else()
            message(WARNING "⚠️ [空间服务] 测试目录不存在: ${CMAKE_CURRENT_SOURCE_DIR}/tests")
        endif()
    else()
        message(WARNING "⚠️ [空间服务] Google Test 未找到，跳过测试")
    endif()
else()
    message(STATUS "🔇 [空间服务] 测试被禁用")
endif()

# ===================================================================

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(EXPORT SpatialOpsServiceTargets
    FILE SpatialOpsServiceTargets.cmake
    NAMESPACE OSCEAN::
    DESTINATION lib/cmake/SpatialOpsService
) 