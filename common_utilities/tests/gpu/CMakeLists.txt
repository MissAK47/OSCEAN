cmake_minimum_required(VERSION 3.20)
project(gpu_tests VERSION 1.0.0 LANGUAGES CXX)

message(STATUS "[GPU Tests] 🎮 构建GPU测试...")

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# 🔧 依赖管理
# =============================================================================

find_package(GTest CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system filesystem chrono uuid)
find_package(spdlog CONFIG REQUIRED)

# 检查CUDA支持（可选）
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "[GPU Tests] ✅ CUDA support found")
    add_definitions(-DOSCEAN_CUDA_ENABLED=1)
else()
    message(STATUS "[GPU Tests] ⚠️ CUDA not found, GPU tests will run in CPU fallback mode")
    add_definitions(-DOSCEAN_CUDA_ENABLED=0)
endif()

# 检查OpenCL支持（可选）
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "[GPU Tests] ✅ OpenCL support found")
    add_definitions(-DOSCEAN_OPENCL_ENABLED=1)
else()
    message(STATUS "[GPU Tests] ⚠️ OpenCL not found")
    add_definitions(-DOSCEAN_OPENCL_ENABLED=0)
endif()

# =============================================================================
# 🎯 GPU测试定义
# =============================================================================

# GPU检测测试
add_executable(test_gpu_detection test_gpu_detection.cpp)
target_link_libraries(test_gpu_detection PRIVATE
    GTest::gtest_main
    GTest::gtest
    common_utilities
    Boost::thread
    Boost::system
    Boost::chrono
    spdlog::spdlog
)

# 多GPU调度器测试
add_executable(test_multi_gpu_scheduler test_multi_gpu_scheduler.cpp)
target_link_libraries(test_multi_gpu_scheduler PRIVATE
    GTest::gtest_main
    GTest::gtest
    common_utilities
    Boost::thread
    Boost::system
    Boost::chrono
    spdlog::spdlog
)

# GPU内存管理器测试
add_executable(test_gpu_memory_manager test_gpu_memory_manager.cpp)
target_link_libraries(test_gpu_memory_manager PRIVATE
    GTest::gtest_main
    GTest::gtest
    common_utilities
    Boost::thread
    Boost::system
    Boost::chrono
    spdlog::spdlog
)

# GPU框架集成测试
add_executable(test_gpu_framework test_gpu_framework.cpp)
target_link_libraries(test_gpu_framework PRIVATE
    GTest::gtest_main
    GTest::gtest
    common_utilities
    Boost::thread
    Boost::system
    Boost::chrono
    spdlog::spdlog
)

# GPU算法基础实现测试
add_executable(test_gpu_algorithm_base test_gpu_algorithm_base.cpp)
target_link_libraries(test_gpu_algorithm_base PRIVATE
    GTest::gtest_main
    GTest::gtest
    common_utilities
    Boost::thread
    Boost::system
    Boost::chrono
    spdlog::spdlog
)

# 链接CUDA库（如果可用）
if(CUDAToolkit_FOUND)
    set(GPU_TEST_TARGETS_WITH_CUDA
        test_gpu_detection
        test_gpu_memory_manager
        test_gpu_framework
        test_gpu_algorithm_base
    )
    
    foreach(target ${GPU_TEST_TARGETS_WITH_CUDA})
        target_link_libraries(${target} PRIVATE CUDA::cudart)
    endforeach()
endif()

# 链接OpenCL库（如果可用）
if(OpenCL_FOUND)
    set(GPU_TEST_TARGETS_WITH_OPENCL
        test_gpu_detection
        test_gpu_memory_manager
        test_gpu_framework
        test_gpu_algorithm_base
    )
    
    foreach(target ${GPU_TEST_TARGETS_WITH_OPENCL})
        target_link_libraries(${target} PRIVATE ${OpenCL_LIBRARIES})
        target_include_directories(${target} PRIVATE ${OpenCL_INCLUDE_DIRS})
    endforeach()
endif()

# =============================================================================
# 🔧 通用配置
# =============================================================================

set(GPU_TEST_TARGETS
    test_gpu_detection
    test_multi_gpu_scheduler
    test_gpu_memory_manager
    test_gpu_framework
    test_gpu_algorithm_base
)

foreach(target ${GPU_TEST_TARGETS})
    # 设置C++标准
    set_target_properties(${target} PROPERTIES 
        CXX_STANDARD 17 
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    # 包含目录
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    
    # Windows特定配置
    if(WIN32 AND MSVC)
        target_compile_definitions(${target} PRIVATE
            _WIN32_WINNT=0x0A00
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        )
        target_compile_options(${target} PRIVATE /W4 /FS /Zc:__cplusplus /bigobj)
    endif()
    
    # 设置输出目录
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/gpu_tests"
    )
endforeach()

# =============================================================================
# 🧪 CTest集成
# =============================================================================

enable_testing()

# 注册测试
add_test(NAME GPUDetectionTest COMMAND test_gpu_detection)
add_test(NAME MultiGPUSchedulerTest COMMAND test_multi_gpu_scheduler)
add_test(NAME GPUMemoryManagerTest COMMAND test_gpu_memory_manager)
add_test(NAME GPUFrameworkTest COMMAND test_gpu_framework)
add_test(NAME GPUAlgorithmBaseTest COMMAND test_gpu_algorithm_base)

# 设置测试属性
set_tests_properties(
    GPUDetectionTest
    PROPERTIES LABELS "gpu;unit;fast" TIMEOUT 60
)

set_tests_properties(
    MultiGPUSchedulerTest GPUMemoryManagerTest
    PROPERTIES LABELS "gpu;unit;medium" TIMEOUT 120
)

set_tests_properties(
    GPUFrameworkTest GPUAlgorithmBaseTest
    PROPERTIES LABELS "gpu;integration;medium" TIMEOUT 180
)

# =============================================================================
# 🚀 自定义目标
# =============================================================================

# 运行所有GPU测试
add_custom_target(run_gpu_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "gpu"
    DEPENDS ${GPU_TEST_TARGETS}
    COMMENT "运行所有GPU测试"
)

# 运行快速GPU测试
add_custom_target(run_gpu_tests_fast
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "gpu" -L "fast"
    DEPENDS ${GPU_TEST_TARGETS}
    COMMENT "运行快速GPU测试"
)

# =============================================================================
# 📄 状态报告
# =============================================================================

message(STATUS "[GPU Tests] ✅ GPU测试配置完成:")
message(STATUS "  🎯 测试数量: ${CMAKE_CURRENT_LIST_LINE}")
list(LENGTH GPU_TEST_TARGETS num_tests)
message(STATUS "  🎯 测试数量: ${num_tests}")
message(STATUS "  🎮 测试类型: 检测|调度器|内存管理|框架集成|算法基础")
message(STATUS "  🔧 CUDA支持: ${CUDAToolkit_FOUND}")
message(STATUS "  🔧 OpenCL支持: ${OpenCL_FOUND}")
message(STATUS "  🚀 自定义目标: run_gpu_tests, run_gpu_tests_fast") 