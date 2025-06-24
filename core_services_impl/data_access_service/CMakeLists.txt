cmake_minimum_required(VERSION 3.20)

# 设置CMake策略以避免警告
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)  # 使用新的Boost查找模块
endif()

project(data_access_service VERSION 1.0.0 LANGUAGES CXX)
# No project() command in sub-module

message(STATUS "[data_access_service] Configuring data_access service module.")

# 设置组件版本
set(DATA_ACCESS_SERVICE_VERSION "1.0.0")

SET(VAPKG_DIR "C:/Users/Administrator/vcpkg")

# 从 DATA_ACCESS_SERVICE_VERSION 提取主版本号
string(REGEX MATCH "^([0-9]+)\." _dummy ${DATA_ACCESS_SERVICE_VERSION})
if(CMAKE_MATCH_COUNT GREATER 0)
    set(DATA_ACCESS_SERVICE_VERSION_MAJOR ${CMAKE_MATCH_1})
else()
    set(DATA_ACCESS_SERVICE_VERSION_MAJOR "0") # 提供一个默认值以防万一
    message(WARNING "[data_access_service] Could not parse MAJOR version from DATA_ACCESS_SERVICE_VERSION: ${DATA_ACCESS_SERVICE_VERSION}. Using default SOVERSION ${DATA_ACCESS_SERVICE_VERSION_MAJOR}.")
endif()
message(STATUS "[data_access_service] DATA_ACCESS_SERVICE_VERSION_MAJOR set to: ${DATA_ACCESS_SERVICE_VERSION_MAJOR}")




# --- 查找必需的依赖包 ---
# P1异步框架集成：添加common_utilities依赖
find_package(common_utilities CONFIG QUIET)
if(NOT common_utilities_FOUND)
    # 检查是否已经作为子目录添加
    if(NOT TARGET common_utilities)
        # 如果没有安装，尝试使用项目根目录的common_utilities
        set(COMMON_UTILITIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../common_utilities")
        if(EXISTS "${COMMON_UTILITIES_ROOT}/CMakeLists.txt")
            message(STATUS "[data_access_service] Using common_utilities from: ${COMMON_UTILITIES_ROOT}")
            add_subdirectory("${COMMON_UTILITIES_ROOT}" common_utilities_build)
            set(common_utilities_FOUND TRUE)
            # 添加包含目录
            list(APPEND DATA_ACCESS_INCLUDE_DIRS "${COMMON_UTILITIES_ROOT}/include")
            include_directories("${COMMON_UTILITIES_ROOT}/include")
        else()
            message(FATAL_ERROR "[data_access_service] common_utilities not found at ${COMMON_UTILITIES_ROOT}. This is required for P1 async framework integration.")
        endif()
    else()
        message(STATUS "[data_access_service] common_utilities target already exists, skipping add_subdirectory")
        # 仍然添加包含目录
        set(COMMON_UTILITIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../common_utilities")
        list(APPEND DATA_ACCESS_INCLUDE_DIRS "${COMMON_UTILITIES_ROOT}/include")
        include_directories("${COMMON_UTILITIES_ROOT}/include")
        set(common_utilities_FOUND TRUE)
    endif()
else()
    message(STATUS "[data_access_service] Found common_utilities package")
endif()

# 查找core_service_interfaces模块
if(NOT TARGET core_service_interfaces)
    set(CORE_SERVICE_INTERFACES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces")
    if(EXISTS "${CORE_SERVICE_INTERFACES_ROOT}/CMakeLists.txt")
        message(STATUS "[data_access_service] Using core_service_interfaces from: ${CORE_SERVICE_INTERFACES_ROOT}")
        add_subdirectory("${CORE_SERVICE_INTERFACES_ROOT}" core_service_interfaces_build)
        # 添加包含目录
        list(APPEND DATA_ACCESS_INCLUDE_DIRS "${CORE_SERVICE_INTERFACES_ROOT}/include")
        include_directories("${CORE_SERVICE_INTERFACES_ROOT}/include")
    else()
        message(FATAL_ERROR "[data_access_service] core_service_interfaces not found at ${CORE_SERVICE_INTERFACES_ROOT}.")
    endif()
else()
    message(STATUS "[data_access_service] core_service_interfaces target already exists, skipping add_subdirectory")
    # 仍然添加包含目录
    set(CORE_SERVICE_INTERFACES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces")
    list(APPEND DATA_ACCESS_INCLUDE_DIRS "${CORE_SERVICE_INTERFACES_ROOT}/include")
    include_directories("${CORE_SERVICE_INTERFACES_ROOT}/include")
endif()

# 首先尝试从环境变量或缓存中获取GDAL路径
if(NOT DEFINED GDAL_ROOT)
    if(DEFINED ENV{GDAL_ROOT})
        set(GDAL_ROOT $ENV{GDAL_ROOT} CACHE PATH "Path to GDAL installation")
    endif()
endif()

# 打印GDAL_ROOT信息
if(DEFINED GDAL_ROOT)
    message(STATUS "[data_access_service] Using GDAL_ROOT: ${GDAL_ROOT}")
endif()

# 找到GDAL库
find_package(GDAL CONFIG REQUIRED)

if(GDAL_FOUND)
    message(STATUS "[data_access_service] GDAL include directories: ${GDAL_INCLUDE_DIRS}")
    message(STATUS "[data_access_service] GDAL libraries: ${GDAL_LIBRARIES}")
    
    # 添加GDAL_DATA_DIR宏
    if(DEFINED ENV{GDAL_DATA})
        message(STATUS "[data_access_service] GDAL_DATA environment variable: $ENV{GDAL_DATA}")
        add_compile_definitions(GDAL_DATA_DIR="$ENV{GDAL_DATA}")
    elseif(DEFINED GDAL_DATA_PATH)
        message(STATUS "[data_access_service] Using GDAL_DATA_PATH: ${GDAL_DATA_PATH}")
        add_compile_definitions(GDAL_DATA_DIR="${GDAL_DATA_PATH}")
    endif()
    
    # 定义GDAL版本宏
    if(DEFINED GDAL_VERSION)
        # 将版本字符串拆分为组件
        string(REPLACE "." ";" GDAL_VERSION_LIST ${GDAL_VERSION})
        list(GET GDAL_VERSION_LIST 0 GDAL_VERSION_MAJOR)
        list(GET GDAL_VERSION_LIST 1 GDAL_VERSION_MINOR)
        
        # 定义版本宏供代码使用
        add_compile_definitions(
            GDAL_VERSION_MAJOR=${GDAL_VERSION_MAJOR}
            GDAL_VERSION_MINOR=${GDAL_VERSION_MINOR}
            GDAL_VERSION_NUM=${GDAL_VERSION_MAJOR}${GDAL_VERSION_MINOR}00
        )
        
        message(STATUS "[data_access_service] Defined GDAL version macros: GDAL_VERSION_MAJOR=${GDAL_VERSION_MAJOR}, GDAL_VERSION_MINOR=${GDAL_VERSION_MINOR}")
    endif()
    
    # 定义编译宏启用GDAL读取器
    add_compile_definitions(OSCEAN_HAS_GDAL_READER)
else()
    message(WARNING "[data_access_service] GDAL libraries not found. GDAL support will be disabled.")
endif()

# 查找PROJ库（可选）
find_package(PROJ CONFIG QUIET)
if(NOT PROJ_FOUND)
    message(WARNING "[data_access_service] PROJ library not found. Using stub implementation for coordinate transformation.")
    set(OSCEAN_HAS_PROJ FALSE)
    add_compile_definitions(OSCEAN_HAS_PROJ=0)
else()
    message(STATUS "[data_access_service] Found PROJ: ${PROJ_LIBRARIES}")
    set(OSCEAN_HAS_PROJ TRUE)
    add_compile_definitions(OSCEAN_HAS_PROJ=1)
endif()

# 查找NetCDF库
find_package(netCDF CONFIG QUIET)
if(NOT netCDF_FOUND)
    find_package(netCDF MODULE QUIET)
endif()

if(netCDF_FOUND)
    message(STATUS "[data_access_service] Found NetCDF: ${netCDF_VERSION}")
    set(OSCEAN_HAS_NETCDF TRUE)
    add_definitions(-DOSCEAN_HAS_NETCDF)
    
    # 添加NetCDF的包含目录
    if(DEFINED netCDF_INCLUDE_DIRS)
        message(STATUS "[data_access_service] Using NetCDF include dirs: ${netCDF_INCLUDE_DIRS}")
        list(APPEND DATA_ACCESS_INCLUDE_DIRS ${netCDF_INCLUDE_DIRS})
        include_directories(${netCDF_INCLUDE_DIRS})
    elseif(TARGET netCDF::netcdf)
        get_target_property(NETCDF_INTERFACE_INCLUDES netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(NETCDF_INTERFACE_INCLUDES)
            message(STATUS "[data_access_service] Using NetCDF include dirs from target: ${NETCDF_INTERFACE_INCLUDES}")
            list(APPEND DATA_ACCESS_INCLUDE_DIRS ${NETCDF_INTERFACE_INCLUDES})
            include_directories(${NETCDF_INTERFACE_INCLUDES})
        endif()
    endif()
else()
    message(STATUS "[data_access_service] NetCDF not found, disabling NetCDF support")
    set(OSCEAN_HAS_NETCDF FALSE)
endif()

# 查找spdlog库
find_package(spdlog CONFIG REQUIRED)
if(spdlog_FOUND)
    message(STATUS "[data_access_service] Found spdlog: ${spdlog_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "[data_access_service] spdlog library not found. This is a required dependency.")
endif()

# 查找fmt库 (spdlog 的依赖)
find_package(fmt CONFIG REQUIRED)
if(fmt_FOUND)
    message(STATUS "[data_access_service] Found fmt: ${fmt_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "[data_access_service] fmt library not found. This is a required dependency for spdlog.")
endif()

# 设置CMake策略以避免Boost查找警告
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)  # 使用新的Boost查找模块
endif()

# 查找Boost库 (asio 和 thread 组件)
find_package(Boost REQUIRED COMPONENTS thread system filesystem)
if(Boost_FOUND)
    message(STATUS "[data_access_service] Found Boost: ${Boost_VERSION}")
    list(APPEND DATA_ACCESS_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "[data_access_service] Boost (thread, system, filesystem) libraries not found. These are required dependencies.")
endif()

# 查找nlohmann_json库
find_package(nlohmann_json CONFIG QUIET) # 使用QUIET因为一些项目可能通过子模块或直接包含头文件
if(nlohmann_json_FOUND)
    message(STATUS "[data_access_service] Found nlohmann_json: ${nlohmann_json_INCLUDE_DIRS}")
    # 通常 nlohmann_json 是一个头文件库，但 vcpkg 可能会提供一个 target
    if(TARGET nlohmann_json::nlohmann_json)
        message(STATUS "[data_access_service] nlohmann_json target found: nlohmann_json::nlohmann_json")
    endif()
else()
    message(WARNING "[data_access_service] nlohmann_json library/target not found. Assuming header-only or included elsewhere.")
    # 如果您的项目依赖于通过find_package找到nlohmann_json，这里可能需要改为FATAL_ERROR
endif()

# 查找GTest库（仅在需要构建测试时）
if(BUILD_TESTING)
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "[data_access_service] GTest not found via CONFIG, trying MODULE mode")
        find_package(GTest MODULE QUIET)
    endif()
    
    if(GTest_FOUND)
        message(STATUS "[data_access_service] Found GTest: ${GTEST_VERSION}")
        if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
            message(STATUS "[data_access_service] GTest targets available: GTest::gtest, GTest::gtest_main")
        endif()
    else()
        message(WARNING "[data_access_service] GTest not found. Tests will be disabled.")
    endif()
endif()

# 设置编译标志以启用GDAL读取器
add_compile_definitions(OSCEAN_HAS_GDAL_READER=1)

# 添加 netcdf_reader 子模块
# 暂时禁用NetCDF模块以专注于Phase 3异步执行器编译
# if(OSCEAN_HAS_NETCDF)
#     add_subdirectory(src/impl/readers/netcdf)
# endif()

# 内存对齐优化配置
# option(ENABLE_MEMORY_ALIGNMENT "Enable SIMD memory alignment optimization" ON)

# if(ENABLE_MEMORY_ALIGNMENT)
#     message(STATUS "[data_access_service] Memory alignment optimization enabled")
#     add_compile_definitions(OSCEAN_HAS_HIGH_PERFORMANCE_INTERFACES)
#     # 启用额外的SIMD优化标志
#     if(MSVC)
#         add_compile_options(/arch:AVX2)
#     else()
#         add_compile_options(-mavx2 -mfma)
#     endif()
# else()
#     message(STATUS "[data_access_service] Memory alignment optimization disabled")
# endif()

# 目标名称定义
set(TARGET_NAME "data_access_service")

# 添加MSVC特定的编译选项
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 设置源文件列表 - 清理后的版本（只包含实际存在的文件）
set(DATA_ACCESS_SOURCES
    # 核心服务实现 - 实际存在的文件
    src/unified_data_access_service_impl.cpp
    src/data_access_service_factory_impl.cpp
    
    # 🔧 修复：添加并发优化组件实现
    src/concurrent_optimization_components.cpp
    
    # 读取器架构 - 实际存在的文件
    src/readers/core/reader_registry.cpp
    src/readers/core/unified_data_reader.cpp               # 添加统一数据读取器基类实现
    src/readers/core/impl/unified_advanced_reader.cpp
    
    # GDAL读取器实现 - 清理后的文件列表
    src/readers/core/impl/gdal/gdal_raster_reader.cpp
    src/readers/core/impl/gdal/gdal_raster_processor.cpp
    src/readers/core/impl/gdal/gdal_vector_reader.cpp      # 矢量读取器
    src/readers/core/impl/gdal/gdal_vector_processor.cpp   # 矢量处理器
    src/readers/core/impl/gdal/gdal_format_handler.cpp
    # ✅ 已删除冗余文件:
    # - gdal_advanced_reader.h/.cpp (功能已整合到专用读取器)
    # - gdal_base_reader.h/.cpp (功能已分散到专用读取器)
    
    # NetCDF读取器实现 - 清理后的文件（删除了重复的文件）
    src/readers/core/impl/netcdf/netcdf_advanced_reader.cpp
    src/readers/core/impl/netcdf/netcdf_coordinate_system.cpp
    src/readers/core/impl/netcdf/netcdf_variable_processor.cpp
    src/readers/core/impl/netcdf/netcdf_utils.cpp
    src/readers/core/impl/netcdf/memory_layout_analyzer.cpp  # 新增：内存布局分析器
    
    # 缓存系统 - 已移除，使用 common_utilities 提供的缓存功能
    # src/cache/unified_data_access_cache.cpp - 已删除，功能重复
    
    # 流式处理（已完整实现）
    src/streaming/streaming_processor.cpp
    src/streaming/data_streaming_coordinator.cpp
    src/streaming/performance_manager.cpp
    
    # 时间处理（已完整实现）
    # src/time/cf_time_extractor.cpp - 已删除，功能合并到 netcdf_time_processor
)

# Phase 1: 性能基线测试文件
set(PERFORMANCE_BASELINE_TESTS
    tests/performance_baseline_tests.cpp
)

# 包含目录
set(DATA_ACCESS_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GDAL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core_service_interfaces/include
    ${CMAKE_SOURCE_DIR}/common_utilities/include
)

# 如果找到NetCDF库，添加其包含目录
if(OSCEAN_HAS_NETCDF)
    list(APPEND DATA_ACCESS_INCLUDE_DIRS ${netCDF_INCLUDE_DIRS})
endif()

# 链接库
set(DATA_ACCESS_LIBS
    ${GDAL_LIBRARIES}
)

# 如果core_service_interfaces作为子目录存在，添加依赖
if(TARGET core_service_interfaces)
    list(APPEND DATA_ACCESS_LIBS core_service_interfaces)
endif()

# 如果common_utilities作为子目录存在，添加依赖
if(TARGET common_utilities)
    list(APPEND DATA_ACCESS_LIBS common_utilities)
endif()

# 如果找到NetCDF库，添加相应的链接
if(OSCEAN_HAS_NETCDF)
    list(APPEND DATA_ACCESS_LIBS netCDF::netcdf)
endif()

# 添加spdlog和Boost库
# list(APPEND DATA_ACCESS_LIBS spdlog::spdlog Boost::asio Boost::thread)

# 添加 nlohmann_json 目标（如果存在）
# if(TARGET nlohmann_json::nlohmann_json)
# list(APPEND DATA_ACCESS_LIBS nlohmann_json::nlohmann_json)
# endif()

# 创建库
add_library(${TARGET_NAME} STATIC ${DATA_ACCESS_SOURCES})

# 设置目标属性
set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION ${DATA_ACCESS_SERVICE_VERSION}
    SOVERSION ${DATA_ACCESS_SERVICE_VERSION_MAJOR}
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)


# 为目标添加包含目录
target_include_directories(${TARGET_NAME}
    PUBLIC
        # 接口相关的头文件目录，使其对链接到此库的其他目标可见
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../core_service_interfaces/include> # 修正相对路径
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common_utilities/include>      # 修正相对路径
        "${VAPKG_DIR}/installed/x64-windows/include"
    PRIVATE
        # 实现相关的头文件目录
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GDAL_INCLUDE_DIRS} # GDAL的头文件
        ${Boost_INCLUDE_DIRS} # Boost的头文件
        # spdlog 和 fmt 的包含目录通常由其 CMake target 自动处理
        # ${spdlog_INCLUDE_DIRS} # 通常不需要手动添加
        # ${fmt_INCLUDE_DIRS}    # 通常不需要手动添加
)

# 如果找到NetCDF库，添加其包含目录到目标
if(OSCEAN_HAS_NETCDF AND TARGET netCDF::netcdf)
    target_include_directories(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:netCDF::netcdf,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# 如果启用了NetCDF读取器并且netcdf_reader目标存在，则链接它
if(OSCEAN_HAS_NETCDF AND TARGET netcdf_reader)
    list(APPEND DATA_ACCESS_LIBS netcdf_reader)
    # 确保在目标链接中也添加netcdf_reader
    target_link_libraries(${TARGET_NAME} PUBLIC netcdf_reader)
    message(STATUS "[data_access_service] Linked against netcdf_reader.")
endif()

# 链接库到目标
target_link_libraries(${TARGET_NAME}
    PUBLIC
        core_service_interfaces # 移至 PUBLIC
        common_utilities      # 移至 PUBLIC
        spdlog::spdlog
        # netCDF::netcdf # 保持 PUBLIC，如果接口使用了 NetCDF 类型
    PRIVATE
        ${GDAL_LIBRARIES} # GDAL_LIBRARIES 已包含在 DATA_ACCESS_LIBS 中，但这里直接使用目标更清晰
        # ${DATA_ACCESS_LIBS} # 不再需要，因为主要依赖已移至PUBLIC或直接在此处列出
        Boost::thread
        Boost::system
        Boost::filesystem
        nlohmann_json::nlohmann_json
)

# 检查Boost::asio是否可用，如果可用则链接
if(TARGET Boost::asio)
    target_link_libraries(${TARGET_NAME} PRIVATE Boost::asio)
endif()

if(OSCEAN_HAS_NETCDF AND TARGET netCDF::netcdf)
    target_link_libraries(${TARGET_NAME} PUBLIC netCDF::netcdf) # 确保netCDF是PUBLIC如果需要
endif()

# 如果 PROJ 找到并且是一个 target, 链接它
if(PROJ_FOUND AND TARGET PROJ::proj)
    target_link_libraries(${TARGET_NAME} PRIVATE PROJ::proj)
endif()

# 设置C++标准
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)

# 添加boost::future所需的宏定义
add_definitions(-DBOOST_THREAD_PROVIDES_FUTURE)
add_definitions(-DBOOST_THREAD_PROVIDES_FUTURE_CONTINUATION)
add_definitions(-DBOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY)
add_definitions(-DBOOST_THREAD_PROVIDES_FUTURE_ASYNC)

# 添加MSVC特定的编译选项
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE
        /wd4996  # 禁用已弃用函数警告
        /wd4251  # 禁用DLL接口警告
        /FS      # 强制同步PDB写入
        /utf-8   # 使用UTF-8编码
        /EHsc    # 启用C++异常处理
        /wd4834  # 禁用[[nodiscard]]警告
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        _WIN32_WINNT=0x0A00     # Windows 10支持
        NOMINMAX                # 禁用min/max宏
        WIN32_LEAN_AND_MEAN     # 减少Windows头文件包含
    )
endif()

# 数据访问服务测试选项
option(BUILD_DATA_ACCESS_TESTS "Build data access service tests" ON)

# 添加测试 - 简化条件判断
if(BUILD_TESTING)
    message(STATUS "[data_access_service] Adding tests subdirectory.")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        message(WARNING "[data_access_service] Tests directory not found in ${CMAKE_CURRENT_SOURCE_DIR}/tests")
    endif()
else()
    message(STATUS "[data_access_service] Skipping tests (BUILD_TESTING=${BUILD_TESTING})")
endif()

# --- 安装配置 ---
include(GNUInstallDirs)

# 安装头文件
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/data_access_service
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# 查找spdlog（日志库，可能被common_utils/logging.h使用）
find_package(spdlog CONFIG QUIET)
if(NOT spdlog_FOUND)
    # 尝试使用模块模式
    find_package(spdlog MODULE QUIET)
endif()

if(spdlog_FOUND)
    message(STATUS "[data_access_service] Found spdlog: ${spdlog_VERSION}")
    if(TARGET spdlog::spdlog)
        get_target_property(SPDLOG_INTERFACE_INCLUDES spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
        if(SPDLOG_INTERFACE_INCLUDES)
            list(APPEND DATA_ACCESS_INCLUDE_DIRS ${SPDLOG_INTERFACE_INCLUDES})
            include_directories(${SPDLOG_INTERFACE_INCLUDES})
        endif()
    endif()
else()
    message(WARNING "[data_access_service] spdlog not found. Logging functionality may be limited.")
endif()

# 设置目标属性
set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION ${DATA_ACCESS_SERVICE_VERSION}
    SOVERSION ${DATA_ACCESS_SERVICE_VERSION_MAJOR}
)

message(STATUS "[data_access_service] 数据访问服务库配置完成")
message(STATUS "[data_access_service] 已清理过时的源文件和测试配置") 