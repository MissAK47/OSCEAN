# 数据工作流模块 CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# 设置项目名称
set(MODULE_NAME data_workflow)
project(${MODULE_NAME})

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(Boost REQUIRED COMPONENTS system thread)

# 定义头文件目录
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 收集头文件
file(GLOB_RECURSE HEADER_FILES 
    "${INCLUDE_DIR}/*.h"
    "${INCLUDE_DIR}/*.hpp"
)

# 收集源文件 - 暂时排除有问题的enhanced文件
file(GLOB_RECURSE SOURCE_FILES 
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.cc"
)

# 重新启用增强功能文件
# list(FILTER SOURCE_FILES EXCLUDE REGEX ".*enhanced_data_workflow_service_impl\\.cpp$")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*enhanced_workflow_analyzers\\.cpp$")

# 创建静态库
add_library(${MODULE_NAME} STATIC ${SOURCE_FILES} ${HEADER_FILES})

# 设置包含目录
target_include_directories(${MODULE_NAME} 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        # 显式添加依赖项的包含目录，这是最可靠的方式
        ${CMAKE_SOURCE_DIR}/core_service_interfaces/include
        ${CMAKE_SOURCE_DIR}/common_utilities/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接依赖库
target_link_libraries(${MODULE_NAME}
    PUBLIC
        Boost::system
        Boost::thread
        # 链接目标以传递链接时依赖
        core_service_interfaces
        spatial_ops_service
        crs_service
        common_utilities
)

# 设置编译选项
target_compile_options(${MODULE_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

# 设置预处理器定义
target_compile_definitions(${MODULE_NAME} PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# 安装规则
install(TARGETS ${MODULE_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${INCLUDE_DIR}/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# 测试目标
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# 显示构建信息
message(STATUS "数据工作流模块配置完成")
message(STATUS "  - 包含目录: ${INCLUDE_DIR}")
message(STATUS "  - 源文件目录: ${SRC_DIR}")
message(STATUS "  - 头文件数量: ${HEADER_FILES}")
message(STATUS "  - 源文件数量: ${SOURCE_FILES}")
message(STATUS "  - 增强类型支持: 已启用")
message(STATUS "  - 智能策略选择: 已启用") 