cmake_minimum_required(VERSION 3.16)
project(data_workflow_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找网络服务和工作流所需的依赖
find_package(Boost REQUIRED COMPONENTS thread system chrono filesystem)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

# 创建可执行文件
add_executable(data_workflow_app 
    src/main.cpp
)

# Windows 平台特定配置 - 精确解决Boost.Asio与WinRT冲突
if(WIN32)
    target_compile_definitions(data_workflow_app PRIVATE
        _WIN32_WINNT=0x0601          # Windows 7 兼容
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        BOOST_ASIO_DISABLE_WINRT
    )
endif()

if(MSVC)
    target_compile_options(data_workflow_app PRIVATE /utf-8 /wd4819 /wd4996)
    target_compile_definitions(data_workflow_app PRIVATE UNICODE _UNICODE)
endif()

# 添加必要的包含目录 - 现在只依赖我们自己的纯接口
target_include_directories(data_workflow_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 基础依赖库
target_link_libraries(data_workflow_app PRIVATE
    Boost::thread # for std::thread wrapper
    Boost::system # for boost::asio::signal_set
    # 其余依赖由链接的库传递
    Boost::chrono
    Boost::filesystem
    Threads::Threads
    nlohmann_json::nlohmann_json
)

# Windows specific libraries
if(WIN32)
    target_link_libraries(data_workflow_app PRIVATE ws2_32)
endif()

# 内部依赖
target_link_libraries(data_workflow_app PRIVATE
    common_utilities
    core_service_interfaces
    workflow_engine_core
    network_service
)

# 设置输出目录
set_target_properties(data_workflow_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装规则
install(TARGETS data_workflow_app
    RUNTIME DESTINATION bin
) 