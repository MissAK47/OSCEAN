# 测试目录占位符

# 数据工作流测试

# 添加简单工作流CRS测试
add_executable(test_simple_workflow 
    ../test_simple_workflow.cpp
)

target_link_libraries(test_simple_workflow
    PRIVATE
    workflow_engine_core
    common_utilities
)

target_include_directories(test_simple_workflow
    PRIVATE
    ${CMAKE_SOURCE_DIR}/workflow_engine/include
    ${CMAKE_SOURCE_DIR}/workflow_engine/data_workflow/include
    ${CMAKE_SOURCE_DIR}/common_utilities/include
    ${CMAKE_SOURCE_DIR}/core_service_interfaces/include
)
