cmake_minimum_required(VERSION 3.15)
project(spatial_ops_service_tests VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Google Test
find_package(GTest REQUIRED)

# 查找依赖包
find_package(Boost REQUIRED COMPONENTS system thread filesystem)
find_package(GDAL CONFIG REQUIRED)
find_package(geos CONFIG REQUIRED)

# 添加单元测试子目录
add_subdirectory(unit)

# 启用测试
enable_testing()

# 添加测试发现
include(GoogleTest)

message(STATUS "✅ 空间服务单元测试配置完成")
message(STATUS "🎯 测试覆盖: 几何引擎|栅格引擎|工具函数|索引算法")
message(STATUS "📊 测试策略: 真实库集成测试，无Mock依赖") 