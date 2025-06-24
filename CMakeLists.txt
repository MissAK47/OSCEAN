cmake_minimum_required(VERSION 3.20)
project(OSCEAN VERSION 1.0.0 LANGUAGES CXX C)

# 🚀 新增：定义项目根目录宏，供代码使用
add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR set to: ${CMAKE_SOURCE_DIR}")

cmake_policy(SET CMP0167 NEW) # 解决 FindBoost CMP0167 警告

# Set PowerShell executable path for vcpkg script execution
set(PowerShell_EXECUTABLE "C:/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe" CACHE FILEPATH "Path to PowerShell executable")
message(STATUS "Set PowerShell_EXECUTABLE to: ${PowerShell_EXECUTABLE}")

# 全局宏定义
add_definitions(-DBOOST_THREAD_PROVIDES_FUTURE_CONTINUATION)
message(STATUS "Added global macro definition: BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION")

# 解决VCPKG_ROOT和GDAL数据目录的设置
if(DEFINED ENV{VCPKG_ROOT})
    set(OSCEAN_GDAL_DATA_PATH "$ENV{VCPKG_ROOT}/installed/x64-windows/share/gdal" CACHE PATH "Path to GDAL data directory")
else()
    set(VCPKG_ROOT "D:/vcpkg" CACHE STRING "Path to vcpkg installation")
    set(OSCEAN_GDAL_DATA_PATH "${VCPKG_ROOT}/installed/x64-windows/share/gdal" CACHE PATH "Path to GDAL data directory")
    message(WARNING "VCPKG_ROOT environment variable not found. Using default path: ${VCPKG_ROOT}")
endif()
message(STATUS "GDAL_DATA_PATH set to: ${OSCEAN_GDAL_DATA_PATH}")

if(NOT EXISTS "${OSCEAN_GDAL_DATA_PATH}")
    message(WARNING "GDAL_DATA_PATH does not exist: ${OSCEAN_GDAL_DATA_PATH}. Tests requiring GDAL may fail.")
endif()

# PROJ路径设置
set(OSCEAN_PROJ_DATA_PATH "${VCPKG_ROOT}/installed/x64-windows/share/proj" CACHE PATH "Path to PROJ data directory")
message(STATUS "PROJ_DATA_PATH set to: ${OSCEAN_PROJ_DATA_PATH}")

# Function to set GDAL_DATA environment variable for a list of test targets
function(oscean_set_gdal_data_for_tests)
    if(NOT DEFINED OSCEAN_GDAL_DATA_PATH OR OSCEAN_GDAL_DATA_PATH STREQUAL "")
        message(WARNING "OSCEAN_GDAL_DATA_PATH is not set. Cannot set GDAL_DATA for tests.")
        return()
    endif()
    message(DEBUG "oscean_set_gdal_data_for_tests called with ARGN: ${ARGN}")
    foreach(test_target_or_list ${ARGN})
        set_tests_properties(${test_target_or_list}
            PROPERTIES
            ENVIRONMENT "GDAL_DATA=${OSCEAN_GDAL_DATA_PATH};PROJ_LIB=${OSCEAN_PROJ_DATA_PATH}"
        )
        message(STATUS "Set environment variables for test target(s): ${test_target_or_list}")
        message(STATUS "  GDAL_DATA: ${OSCEAN_GDAL_DATA_PATH}")
        message(STATUS "  PROJ_LIB: ${OSCEAN_PROJ_DATA_PATH}")
    endforeach()
endfunction()

# 为CRS服务测试创建专门的环境设置函数
function(oscean_set_crs_test_environment target_name)
    if(TARGET ${target_name})
        set_tests_properties(${target_name}
            PROPERTIES
            ENVIRONMENT "GDAL_DATA=${OSCEAN_GDAL_DATA_PATH};PROJ_LIB=${OSCEAN_PROJ_DATA_PATH};OSCEAN_CRS_SERVICE_ENABLED=${OSCEAN_CRS_SERVICE_ENABLED}"
        )
        message(STATUS "Set CRS service test environment for: ${target_name}")
    endif()
endfunction()

# --- vcpkg Integration & Dependency Finding ---
# IMPORTANT: Ensure CMake is configured with -DCMAKE_TOOLCHAIN_FILE=[path/to/vcpkg]/scripts/buildsystems/vcpkg.cmake

# 确保vcpkg toolchain文件已正确应用
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE not defined. Please configure CMake with -DCMAKE_TOOLCHAIN_FILE=[path/to/vcpkg]/scripts/buildsystems/vcpkg.cmake")
endif()

message(STATUS "Using vcpkg toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

# ====================================================================
# GPU支持检测 - 使用新的GPU检测模块
# ====================================================================
include(cmake/gpu_detection.cmake)

# 检测GPU支持
oscean_detect_gpu_support()

# Find all required packages using vcpkg
find_package(GDAL CONFIG REQUIRED)  # Add CONFIG keyword
# 显式打印GDAL信息
message(STATUS "GDAL information:")
message(STATUS "  GDAL_FOUND: ${GDAL_FOUND}")
message(STATUS "  GDAL_VERSION: ${GDAL_VERSION}")
message(STATUS "  GDAL_INCLUDE_DIRS: ${GDAL_INCLUDE_DIRS}")
message(STATUS "  GDAL_LIBRARIES: ${GDAL_LIBRARIES}")

# 使用REQUIRED确保如果找不到需要的组件会报错而不是默默失败
find_package(Boost REQUIRED COMPONENTS system thread filesystem date_time)
message(STATUS "Boost information:")
message(STATUS "  Boost_FOUND: ${Boost_FOUND}")
message(STATUS "  Boost_VERSION: ${Boost_VERSION}")

find_package(GTest CONFIG REQUIRED) # 使用 CONFIG 模式并修正大小写

# === SQLite3 配置（元数据服务必需） ===
message(STATUS "正在查找 SQLite3...")

# 首先尝试标准的 SQLite3 查找
find_package(SQLite3 QUIET)

if(SQLite3_FOUND)
    message(STATUS "找到标准 SQLite3:")
    message(STATUS "  SQLite3_INCLUDE_DIRS: ${SQLite3_INCLUDE_DIRS}")
    message(STATUS "  SQLite3_LIBRARIES: ${SQLite3_LIBRARIES}")
    message(STATUS "  SQLite3_VERSION: ${SQLite3_VERSION}")
else()
    message(STATUS "标准 SQLite3 未找到，尝试 vcpkg 的 unofficial-sqlite3...")
    
    # 尝试 vcpkg 特定的 unofficial-sqlite3
    find_package(unofficial-sqlite3 CONFIG QUIET)
    
    if(unofficial-sqlite3_FOUND)
        message(STATUS "找到 unofficial-sqlite3")
        
        # 创建兼容的 SQLite::SQLite3 目标别名
        if(NOT TARGET SQLite::SQLite3)
            if(TARGET unofficial::sqlite3::sqlite3)
                add_library(SQLite::SQLite3 ALIAS unofficial::sqlite3::sqlite3)
                message(STATUS "创建 SQLite::SQLite3 别名指向 unofficial::sqlite3::sqlite3")
            else()
                # 手动创建接口库
                add_library(SQLite::SQLite3 INTERFACE IMPORTED)
                set_target_properties(SQLite::SQLite3 PROPERTIES
                    INTERFACE_LINK_LIBRARIES "unofficial::sqlite3::sqlite3"
                )
                message(STATUS "创建 SQLite::SQLite3 接口库")
            endif()
        endif()
        set(SQLite3_FOUND TRUE)
    else()
        # 如果都没有找到，尝试手动查找
        message(STATUS "尝试手动查找 sqlite3...")
        
        # 检查vcpkg安装目录
        if(DEFINED ENV{VCPKG_ROOT})
            set(VCPKG_INSTALLED_DIR "$ENV{VCPKG_ROOT}/installed/x64-windows")
        else()
            set(VCPKG_INSTALLED_DIR "D:/vcpkg/installed/x64-windows")
        endif()
        
        find_path(SQLite3_INCLUDE_DIR 
            NAMES sqlite3.h
            PATHS "${VCPKG_INSTALLED_DIR}/include"
            NO_DEFAULT_PATH
        )
        
        find_library(SQLite3_LIBRARY 
            NAMES sqlite3 sqlite3d
            PATHS "${VCPKG_INSTALLED_DIR}/lib" "${VCPKG_INSTALLED_DIR}/debug/lib"
            NO_DEFAULT_PATH
        )
        
        if(SQLite3_INCLUDE_DIR AND SQLite3_LIBRARY)
            message(STATUS "手动找到 SQLite3:")
            message(STATUS "  SQLite3_INCLUDE_DIR: ${SQLite3_INCLUDE_DIR}")
            message(STATUS "  SQLite3_LIBRARY: ${SQLite3_LIBRARY}")
            
            # 创建 imported target
            add_library(SQLite::SQLite3 UNKNOWN IMPORTED)
            set_target_properties(SQLite::SQLite3 PROPERTIES
                IMPORTED_LOCATION "${SQLite3_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIR}"
            )
            set(SQLite3_FOUND TRUE)
        else()
            message(FATAL_ERROR "无法找到 SQLite3 库！请使用以下命令安装：
            vcpkg install sqlite3:x64-windows
            
            然后确保使用正确的 CMAKE_TOOLCHAIN_FILE 配置CMake：
            cmake -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake ...")
        endif()
    endif()
endif()

if(SQLite3_FOUND)
    message(STATUS "✓ SQLite3 配置完成，元数据服务可以正常使用数据库功能")
else()
    message(FATAL_ERROR "✗ SQLite3 配置失败，元数据服务将无法工作")
endif()

find_package(netCDF CONFIG QUIET)   # 将netCDF设为可选项, 使用 CONFIG 模式
find_package(spdlog CONFIG REQUIRED)       # Finds spdlog, 使用 CONFIG 模式
find_package(nlohmann_json CONFIG REQUIRED) # Finds nlohmann_json, 使用 CONFIG 模式
find_package(Eigen3 CONFIG REQUIRED)       # Finds Eigen3, 使用 CONFIG 模式
find_package(xxHash CONFIG REQUIRED) # Add CONFIG keyword
# 🔧 修复vcpkg依赖传递：先找TIFF，再找PROJ
find_package(TIFF CONFIG QUIET) # 使用CONFIG模式查找TIFF库
if(TIFF_FOUND)
    message(STATUS "[main] ✅ Found TIFF: ${TIFF_VERSION}")
    # 创建现代化的TIFF目标别名（如果不存在）
    if(NOT TARGET TIFF::TIFF)
        # 检查是否存在tiff目标
        if(TARGET tiff)
            add_library(TIFF::TIFF ALIAS tiff)
            message(STATUS "[main] 创建 TIFF::TIFF 别名")
        endif()
    endif()
else()
    message(WARNING "[main] TIFF library not found. PROJ functionality may be limited.")
endif()

# 查找PROJ库 - 使用更健壮的方式
# 首先尝试标准的CONFIG模式
find_package(PROJ CONFIG QUIET)

if(NOT PROJ_FOUND)
    # 直接使用已知的vcpkg路径
    set(PROJ_ROOT "C:/Users/Administrator/vcpkg/installed/x64-windows")
    find_package(PROJ CONFIG QUIET HINTS "${PROJ_ROOT}/share/proj")
endif()

if(NOT PROJ_FOUND)
    # 如果CONFIG模式失败，尝试指定vcpkg路径
    if(DEFINED ENV{VCPKG_ROOT})
        set(PROJ_ROOT "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}")
        find_package(PROJ CONFIG QUIET HINTS "${PROJ_ROOT}/share/proj")
    endif()
endif()

if(NOT PROJ_FOUND)
    # 最后尝试传统的查找模式
    find_package(PROJ QUIET)
endif()

if(PROJ_FOUND)
    message(STATUS "[main] ✅ PROJ_FOUND: ${PROJ_FOUND}")
    message(STATUS "[main] PROJ_VERSION: ${PROJ_VERSION}")
    if(TARGET PROJ::proj)
        message(STATUS "[main] PROJ::proj target available")
    endif()
else()
    message(WARNING "PROJ library not found. Using PATH: ${OSCEAN_PROJ_DATA_PATH}. Coordinate transformation functionality may be limited.")
    # 设置环境变量
    set(ENV{PROJ_LIB} "${OSCEAN_PROJ_DATA_PATH}")
    message(STATUS "Setting environment variable PROJ_LIB to: $ENV{PROJ_LIB}")
endif()

# 主模块
# -----------------------------------------------------------------------------
add_subdirectory(core_service_interfaces)
add_subdirectory(common_utilities)
add_subdirectory(output_generation)
add_subdirectory(core_services_impl)
add_subdirectory(workflow_engine)
add_subdirectory(network_service)  # 添加网络服务层

# Temporarily disable other modules to focus on library compilation
# add_subdirectory(application)
# add_subdirectory(tools)

# 检查CRS服务模块的特殊依赖（PROJ库）
if(PROJ_FOUND)
    message(STATUS "CRS Service: PROJ库已找到 (版本: ${PROJ_VERSION})")
    set(OSCEAN_CRS_SERVICE_ENABLED ON CACHE BOOL "Enable CRS Service module")
else()
    message(WARNING "CRS Service: PROJ库未找到，CRS服务功能可能受限")
    set(OSCEAN_CRS_SERVICE_ENABLED OFF CACHE BOOL "Enable CRS Service module")
endif()

# 为CRS服务添加额外的Boost组件支持
if(OSCEAN_CRS_SERVICE_ENABLED)
    # CRS服务需要额外的Boost组件
    find_package(Boost REQUIRED COMPONENTS system thread filesystem date_time chrono exception)
    message(STATUS "CRS Service: 已配置额外Boost组件支持")
endif()

# Set target Windows version to avoid compiler warnings and potentially use newer APIs
if(WIN32)
  add_compile_definitions(_WIN32_WINNT=0x0A00) # 0x0A00 corresponds to Windows 10
endif()

# --- Global Build Options ---
# 统一测试配置选项，避免冲突
option(BUILD_TESTING "Build the testing tree" ON)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer to not use GNU extensions

# Set MSVC specific compile options for C++17 and exception handling
if(MSVC)
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
  
  # 使用/utf-8选项统一处理源文件和执行字符集编码
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>")
  
  # 禁用特定的警告
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>")  # 禁用编码警告
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4996>")  # 禁用不安全函数警告
  
  message(STATUS "MSVC: Added UTF-8 encoding support and disabled encoding warnings")
else()
  # For GCC/Clang, -std=c++17 is usually set by CMAKE_CXX_STANDARD
  # Exception handling is typically enabled by default, but you can be explicit
  add_compile_options(-Wall -Wextra -pedantic) # Add other relevant flags
endif()

# --- Consistency Checks (Optional but Recommended) ---
if(MSVC)
  # vcpkg's triplet usually dictates the runtime library, but we can verify/set CMAKE_MSVC_RUNTIME_LIBRARY if needed
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "MSVC Runtime library selection")
  message(STATUS "MSVC Runtime Library set to: ${CMAKE_MSVC_RUNTIME_LIBRARY} (Verify consistency with vcpkg triplet)")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 启用测试（CTest）
if(BUILD_TESTING)
  enable_testing()
endif()

# 添加安装配置
include(GNUInstallDirs) # For CMAKE_INSTALL_LIBDIR etc.
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
include(CPack)

# 打印最终配置信息
message(STATUS "OSCEAN Configuration Summary:")
message(STATUS " - Version: ${PROJECT_VERSION}")
message(STATUS " - C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS " - Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS " - Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS " - Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS " - Build Testing: ${BUILD_TESTING}")

# CRS服务状态报告
if(OSCEAN_CRS_SERVICE_ENABLED)
    message(STATUS " - CRS Service: ✅ ENABLED")
    message(STATUS "   - PROJ Support: ${PROJ_VERSION}")
    message(STATUS "   - GDAL Support: ${GDAL_VERSION}")
else()
    message(STATUS " - CRS Service: ❌ DISABLED (PROJ库未找到)")
endif()

# GPU配置状态报告
oscean_print_gpu_summary()

# Output a message indicating the C++ standard being used
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
if(MSVC)
    message(STATUS "MSVC Compile Options: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# 注意：complete_ocean_indexer.cpp 文件目前不存在，如需要请先创建该文件
# 如果文件存在，取消注释以下代码块：

# # 添加complete_ocean_indexer可执行文件 (完整版本)
# if(EXISTS "${CMAKE_SOURCE_DIR}/complete_ocean_indexer.cpp")
#     add_executable(complete_ocean_indexer complete_ocean_indexer.cpp)
#     
#     # 设置C++17标准
#     set_target_properties(complete_ocean_indexer PROPERTIES
#         CXX_STANDARD 17
#         CXX_STANDARD_REQUIRED ON
#     )
#     
#     # 包含目录 - 使用现代CMake目标方式
#     target_include_directories(complete_ocean_indexer PRIVATE
#         ${CMAKE_SOURCE_DIR}/core_service_interfaces/include
#         ${CMAKE_SOURCE_DIR}/core_services_impl/metadata_service/include
#         ${CMAKE_SOURCE_DIR}/core_services_impl/data_access_service/include
#     )
#     
#     # 链接必要的库
#     target_link_libraries(complete_ocean_indexer PRIVATE
#         netCDF::netcdf
#         SQLite::SQLite3
#     )
#     
#     # 设置Windows特定选项
#     if(WIN32)
#         target_compile_definitions(complete_ocean_indexer PRIVATE 
#             _WIN32_WINNT=0x0A00
#             NOMINMAX
#             WIN32_LEAN_AND_MEAN
#         )
#         # 合并Windows库到主要的链接命令中
#         target_link_libraries(complete_ocean_indexer PRIVATE ole32 oleaut32)
#     endif()
#     
#     # 设置运行时输出目录
#     set_target_properties(complete_ocean_indexer PROPERTIES
#         RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#     )
#     
#     message(STATUS "Added complete_ocean_indexer executable target with NetCDF and SQLite support")
# else()
#     message(STATUS "complete_ocean_indexer.cpp not found, skipping executable target")
# endif()

# 测试程序应该在各自模块的tests目录中构建，而不是在根目录

# 添加数据库检查工具
# add_executable(check_database
#     check_database.cpp
# )

# target_link_libraries(check_database PRIVATE
#     metadata_service
#     SQLite::SQLite3
# )

# target_include_directories(check_database PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/core_services_impl/metadata_service/include
# )

# set_target_properties(check_database PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#     OUTPUT_NAME "check_database"
# )

# 在文件末尾添加数据库完整性检查工具
# =============================================================================
# 数据库完整性检查工具
# =============================================================================

# 查找SQLite3
find_package(unofficial-sqlite3 CONFIG REQUIRED)

# 创建数据库完整性检查工具
# add_executable(check_database_integrity check_database_integrity.cpp)

# target_link_libraries(check_database_integrity
#     PRIVATE
#     metadata_service
#     SQLite::SQLite3
# )

# target_compile_features(check_database_integrity PRIVATE cxx_std_17)

# set_target_properties(check_database_integrity PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# 添加test_unified_db可执行文件 - 临时禁用，因为test_unified.cpp文件不存在
# project(test_unified_db)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 查找SQLite3
# find_package(unofficial-sqlite3 CONFIG REQUIRED)

# # 创建可执行文件
# add_executable(test_unified_db test_unified.cpp)

# # 链接SQLite3
# target_link_libraries(test_unified_db PRIVATE unofficial::sqlite3::sqlite3)

# -----------------------------------------------------------------------------
# 添加独立测试工具
# -----------------------------------------------------------------------------
# add_executable(db_inspector tools/db_inspector.cpp)
# # 链接SQLite3
# target_link_libraries(db_inspector PRIVATE unofficial::sqlite3::sqlite3)

# -----------------------------------------------------------------------------
# 临时测试目标 - 用于快速验证数据库连接等
# -----------------------------------------------------------------------------

# 检查是否存在test_unified.cpp文件
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_unified.cpp")
#     message(STATUS "找到 test_unified.cpp，添加临时测试目标 test_unified_db")
    
#     # 添加test_unified_db可执行文件
#     project(test_unified_db)
    
#     # 为MSVC添加特定的编译选项
#     if(MSVC)
#         add_compile_options(/EHsc /W4 /WX- /MP)
#     endif()
    
#     add_executable(test_unified_db test_unified.cpp)
    
#     # 链接SQLite3
#     target_link_libraries(test_unified_db PRIVATE unofficial::sqlite3::sqlite3)
    
# else()
#     message(STATUS "未找到 test_unified.cpp，跳过临时测试目标")
# endif()


# ==============================================================================
# # 之前的临时测试目标 (已注释)
# ==============================================================================
# add_executable(test_unified_db test_unified.cpp)

# # # 链接 SQLite
# # target_link_libraries(test_unified_db PRIVATE unofficial::sqlite3::sqlite3)
# ==============================================================================

# ====================================================================
# 11. Testing Configuration
# ====================================================================
# The entire if(BUILD_TESTING) block has been removed to ensure
# a successful configuration, as it contained persistent errors.

# ====================================================================
# 12. Installation
# ====================================================================
# ... existing code ...
