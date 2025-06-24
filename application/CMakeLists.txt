cmake_minimum_required(VERSION 3.16)
project(oscean_applications)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4996)
    add_compile_definitions(UNICODE _UNICODE)
endif()

# =============================================================================
# 数据工作流应用程序 - 作为独立子项目
# =============================================================================
add_subdirectory(data_workflow_app)

# =============================================================================
# 安装规则
# =============================================================================
# 安装规则由各个子项目自己管理 