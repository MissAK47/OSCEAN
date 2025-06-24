cmake_minimum_required(VERSION 3.20)

# 创建数据库检查工具
add_executable(db_inspector db_inspector.cpp)

# 链接SQLite3
find_package(unofficial-sqlite3 CONFIG REQUIRED)
target_link_libraries(db_inspector PRIVATE unofficial::sqlite3::sqlite3)

# 设置输出目录
set_target_properties(db_inspector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
) 