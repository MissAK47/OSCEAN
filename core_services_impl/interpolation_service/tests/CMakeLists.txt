cmake_minimum_required(VERSION 3.20)

# 查找测试框架
find_package(GTest REQUIRED)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_SOURCE_DIR}/core_service_interfaces/include
    ${CMAKE_SOURCE_DIR}/common_utilities/include
)

# ==============================================================================
# 单元测试 (Unit Tests)
# ==============================================================================

# 基础算法测试
add_executable(test_interpolation_algorithms
    unit/test_interpolation_algorithms.cpp
)
target_link_libraries(test_interpolation_algorithms
    GTest::gtest
    GTest::gtest_main
    interpolation_service
    common_utilities
)
add_test(NAME InterpolationAlgorithmsTest COMMAND test_interpolation_algorithms)

# 双线性插值测试
add_executable(test_bilinear_interpolator
    unit/test_bilinear_interpolator.cpp
)
target_link_libraries(test_bilinear_interpolator
    GTest::gtest
    GTest::gtest_main
    interpolation_service
    common_utilities
)
add_test(NAME BilinearInterpolatorTest COMMAND test_bilinear_interpolator)

# 双线性SIMD性能测试
add_executable(test_bilinear_simd_performance
    test_bilinear_simd_performance.cpp
)
target_link_libraries(test_bilinear_simd_performance
    GTest::gtest
    GTest::gtest_main
    interpolation_service
    common_utilities
)
# 添加编译选项以启用SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_bilinear_simd_performance PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_bilinear_simd_performance PRIVATE /arch:AVX2)
endif()
add_test(NAME BilinearSIMDPerformanceTest COMMAND test_bilinear_simd_performance)

# 线性1D和三线性SIMD性能测试
add_executable(test_linear_trilinear_simd_performance
    test_linear_trilinear_simd_performance.cpp
)
target_link_libraries(test_linear_trilinear_simd_performance
    GTest::gtest
    GTest::gtest_main
    interpolation_service
    common_utilities
)
# 添加编译选项以启用SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_linear_trilinear_simd_performance PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_linear_trilinear_simd_performance PRIVATE /arch:AVX2)
endif()
add_test(NAME LinearTrilinearSIMDPerformanceTest COMMAND test_linear_trilinear_simd_performance)

# 立方样条和最近邻SIMD性能测试
add_executable(test_cubic_nearest_simd_performance
    test_cubic_nearest_simd_performance.cpp
)
target_link_libraries(test_cubic_nearest_simd_performance PRIVATE
    interpolation_service
    common_utilities
    core_service_interfaces
    gtest
    gtest_main
)
# 添加编译选项以启用SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_cubic_nearest_simd_performance PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_cubic_nearest_simd_performance PRIVATE /arch:AVX2)
endif()
add_test(NAME CubicNearestSIMDPerformanceTest COMMAND test_cubic_nearest_simd_performance)

# PCHIP 2D SIMD性能测试
add_executable(test_pchip_2d_simd_performance
    test_pchip_2d_simd_performance.cpp
)

target_link_libraries(test_pchip_2d_simd_performance PRIVATE
    interpolation_service
    common_utilities
    core_service_interfaces
    GTest::gtest
    GTest::gtest_main
)

# 添加编译选项以启用SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_pchip_2d_simd_performance PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_pchip_2d_simd_performance PRIVATE /arch:AVX2)
endif()
add_test(NAME PCHIP2DSIMDPerformanceTest COMMAND test_pchip_2d_simd_performance)

# PCHIP 3D SIMD性能测试
add_executable(test_pchip_3d_simd_performance
    test_pchip_3d_simd_performance.cpp
)

target_link_libraries(test_pchip_3d_simd_performance PRIVATE
    interpolation_service
    common_utilities
    core_service_interfaces
    GTest::gtest
    GTest::gtest_main
)

# 添加编译选项以启用SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_pchip_3d_simd_performance PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_pchip_3d_simd_performance PRIVATE /arch:AVX2)
endif()
add_test(NAME FastPCHIP3DSIMDPerformanceTest COMMAND test_pchip_3d_simd_performance)

# 复数场插值SIMD性能测试
add_executable(test_complex_field_simd_performance
    test_complex_field_simd_performance.cpp
)

target_link_libraries(test_complex_field_simd_performance PRIVATE
    interpolation_service
    common_utilities
    core_service_interfaces
    GTest::gtest
    GTest::gtest_main
)

# 添加编译选项以启用SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_complex_field_simd_performance PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_complex_field_simd_performance PRIVATE /arch:AVX2)
endif()
add_test(NAME ComplexFieldSIMDPerformanceTest COMMAND test_complex_field_simd_performance)

# 基础PCHIP SIMD性能测试
add_executable(test_pchip_basic_simd_performance
    test_pchip_basic_simd_performance.cpp
)

target_link_libraries(test_pchip_basic_simd_performance PRIVATE
    interpolation_service
    common_utilities
    core_service_interfaces
    GTest::gtest
    GTest::gtest_main
)

# 添加编译选项以启用SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_pchip_basic_simd_performance PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_pchip_basic_simd_performance PRIVATE /arch:AVX2)
endif()
add_test(NAME PCHIPBasicSIMDPerformanceTest COMMAND test_pchip_basic_simd_performance)

# 服务集成测试
add_executable(test_interpolation_service_integration
    unit/test_interpolation_service_integration.cpp
)
target_link_libraries(test_interpolation_service_integration
    GTest::gtest
    GTest::gtest_main
    interpolation_service
    common_utilities
    Boost::thread
    Boost::system
)
add_test(NAME InterpolationServiceIntegrationTest COMMAND test_interpolation_service_integration)

# 性能基准测试
add_executable(test_interpolation_performance_benchmark
    unit/test_interpolation_performance_benchmark.cpp
)
target_link_libraries(test_interpolation_performance_benchmark
    GTest::gtest
    GTest::gtest_main
    interpolation_service
    common_utilities
    Boost::thread
    Boost::system
)
add_test(NAME InterpolationPerformanceBenchmarkTest COMMAND test_interpolation_performance_benchmark)

# ==============================================================================
# GPU测试 (GPU Tests)
# ==============================================================================

if(OSCEAN_USE_CUDA OR OSCEAN_USE_OPENCL)
    # GPU插值测试
    add_executable(test_gpu_interpolation
        gpu/test_gpu_interpolation.cpp
    )
    target_link_libraries(test_gpu_interpolation
        GTest::gtest
        GTest::gtest_main
        interpolation_service
        common_utilities
    )
    add_test(NAME GPUInterpolationTest COMMAND test_gpu_interpolation)
    
    # GPU性能测试
    add_executable(test_gpu_interpolation_performance
        gpu/test_gpu_interpolation_performance.cpp
    )
    target_link_libraries(test_gpu_interpolation_performance
        GTest::gtest
        GTest::gtest_main
        interpolation_service
        common_utilities
    )
    add_test(NAME GPUInterpolationPerformanceTest COMMAND test_gpu_interpolation_performance)
endif()

# ==============================================================================
# 集成测试 (Integration Tests)
# ==============================================================================

# GPU插值集成测试
add_executable(test_gpu_interpolation_integration
    test_gpu_interpolation_integration.cpp
)

# 为测试目标添加头文件目录
target_include_directories(test_gpu_interpolation_integration PRIVATE
    "${CMAKE_SOURCE_DIR}/common_utilities/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_SOURCE_DIR}/core_service_interfaces/include"
)

# 链接GTest和其他依赖
target_link_libraries(test_gpu_interpolation_integration PRIVATE 
    GTest::gtest
    GTest::gtest_main
    interpolation_service
    common_utilities
    Boost::thread
    Boost::system
)
add_test(NAME GPUInterpolationIntegrationTest COMMAND test_gpu_interpolation_integration)

# 布局感知插值测试
add_executable(test_layout_aware_interpolation
    test_layout_aware_interpolation.cpp
)
target_link_libraries(test_layout_aware_interpolation
    GTest::gtest
    GTest::gtest_main
    interpolation_service
    common_utilities
)
add_test(NAME LayoutAwareInterpolationTest COMMAND test_layout_aware_interpolation)

# ==============================================================================
# 测试工具和设置
# ==============================================================================

# 启用CTest
enable_testing()

# 设置测试超时时间
set_tests_properties(
    InterpolationAlgorithmsTest
    BilinearInterpolatorTest
    BilinearSIMDPerformanceTest
    LinearTrilinearSIMDPerformanceTest
    CubicNearestSIMDPerformanceTest
    InterpolationServiceIntegrationTest
    InterpolationPerformanceBenchmarkTest
    GPUInterpolationIntegrationTest
    LayoutAwareInterpolationTest
    PROPERTIES TIMEOUT 60
)

# GPU测试可能需要更长时间
if(OSCEAN_USE_CUDA OR OSCEAN_USE_OPENCL)
    set_tests_properties(
        GPUInterpolationTest
        GPUInterpolationPerformanceTest
        PROPERTIES TIMEOUT 180
    )
endif()

# 添加测试标签
set_tests_properties(
    InterpolationAlgorithmsTest
    BilinearInterpolatorTest
    BilinearSIMDPerformanceTest
    LinearTrilinearSIMDPerformanceTest
    CubicNearestSIMDPerformanceTest
    InterpolationServiceIntegrationTest
    InterpolationPerformanceBenchmarkTest
    LayoutAwareInterpolationTest
    PROPERTIES LABELS "unit"
)

set_tests_properties(
    GPUInterpolationIntegrationTest
    PROPERTIES LABELS "integration"
)

if(OSCEAN_USE_CUDA OR OSCEAN_USE_OPENCL)
    set_tests_properties(
        GPUInterpolationTest
        GPUInterpolationPerformanceTest
        PROPERTIES LABELS "gpu"
    )
endif()

# 添加所有测试源文件
set(TEST_SOURCES
    test_interpolation_algorithms.cpp
    test_interpolation_service.cpp
    test_bilinear_simd_performance.cpp
    test_linear_trilinear_simd_performance.cpp
    test_cubic_nearest_simd_performance.cpp
    # test_gpu_interpolation_integration.cpp  # 暂时禁用，等GPU支持完善后再启用
    # test_layout_aware_interpolation.cpp
    # test_pchip_interpolation.cpp
)

# GPU性能测试（如果启用了GPU）
if(OSCEAN_GPU_ENABLED)
    add_executable(test_gpu_basic_performance
        test_gpu_basic_performance.cpp
    )
    
    target_link_libraries(test_gpu_basic_performance
        PRIVATE
            interpolation_service
            common_utilities
            GTest::gtest
            GTest::gtest_main
    )
    
    # 添加CUDA支持
    if(CUDA_FOUND)
        target_link_libraries(test_gpu_basic_performance
            PRIVATE
                ${CUDA_LIBRARIES}
                ${CUDA_CUDART_LIBRARY}
        )
    endif()
    
    add_test(
        NAME test_gpu_basic_performance
        COMMAND test_gpu_basic_performance
    )
    
    # GPU综合测试
    add_executable(test_gpu_interpolation_comprehensive
        test_gpu_interpolation_comprehensive.cpp
    )
    
    target_link_libraries(test_gpu_interpolation_comprehensive
        PRIVATE
            interpolation_service
            common_utilities
            GTest::gtest
            GTest::gtest_main
    )
    
    # 添加CUDA支持
    if(CUDA_FOUND)
        target_link_libraries(test_gpu_interpolation_comprehensive
            PRIVATE
                ${CUDA_LIBRARIES}
                ${CUDA_CUDART_LIBRARY}
        )
    endif()
    
    add_test(
        NAME test_gpu_interpolation_comprehensive
        COMMAND test_gpu_interpolation_comprehensive
    )
    
    # 设置测试超时时间
    set_tests_properties(test_gpu_interpolation_comprehensive
        PROPERTIES 
        TIMEOUT 300
        LABELS "gpu;comprehensive"
    )
    
    # 综合性能对比测试
    add_executable(test_performance_comparison
        test_performance_comparison.cpp
    )
    
    target_link_libraries(test_performance_comparison
        PRIVATE
            interpolation_service
            common_utilities
            GTest::gtest
            GTest::gtest_main
    )
    
    # 添加CUDA支持
    if(CUDA_FOUND)
        target_link_libraries(test_performance_comparison
            PRIVATE
                ${CUDA_LIBRARIES}
                ${CUDA_CUDART_LIBRARY}
        )
    endif()
    
    add_test(
        NAME test_performance_comparison
        COMMAND test_performance_comparison
    )
    
    # 设置测试超时时间
    set_tests_properties(test_performance_comparison
        PROPERTIES 
        TIMEOUT 300
        LABELS "performance;comparison"
    )
    
    # 批量GPU优化测试
    add_executable(test_batch_gpu_optimization
        test_batch_gpu_optimization.cpp
    )
    
    target_link_libraries(test_batch_gpu_optimization
        PRIVATE
            interpolation_service
            common_utilities
            GTest::gtest
            GTest::gtest_main
    )
    
    # 添加CUDA支持
    if(CUDA_FOUND)
        target_link_libraries(test_batch_gpu_optimization
            PRIVATE
                ${CUDA_LIBRARIES}
                ${CUDA_CUDART_LIBRARY}
        )
    endif()
    
    add_test(
        NAME test_batch_gpu_optimization
        COMMAND test_batch_gpu_optimization
    )
    
    # 设置测试超时时间
    set_tests_properties(test_batch_gpu_optimization
        PROPERTIES 
        TIMEOUT 300
        LABELS "gpu;batch;optimization"
    )
    
    # 综合性能对比测试（CPU vs SIMD vs GPU）
    add_executable(test_comprehensive_performance
        test_comprehensive_performance.cpp
    )
    
    target_link_libraries(test_comprehensive_performance
        PRIVATE
            interpolation_service
            common_utilities
            GTest::gtest
            GTest::gtest_main
    )
    
    # 添加CUDA支持
    if(CUDA_FOUND)
        target_link_libraries(test_comprehensive_performance
            PRIVATE
                ${CUDA_LIBRARIES}
                ${CUDA_CUDART_LIBRARY}
        )
    endif()
    
    add_test(
        NAME test_comprehensive_performance
        COMMAND test_comprehensive_performance
    )
    
    # 设置测试超时时间
    set_tests_properties(test_comprehensive_performance
        PROPERTIES 
        TIMEOUT 300
        LABELS "performance;comprehensive;comparison"
    )
    
    # CPU vs GPU 性能对比测试
    add_executable(test_cpu_vs_gpu_performance
        test_cpu_vs_gpu_performance.cpp
    )
    
    target_link_libraries(test_cpu_vs_gpu_performance
        PRIVATE
            interpolation_service
            common_utilities
            GTest::gtest
            GTest::gtest_main
    )
    
    # 添加CUDA支持
    if(CUDA_FOUND)
        target_link_libraries(test_cpu_vs_gpu_performance
            PRIVATE
                ${CUDA_LIBRARIES}
                ${CUDA_CUDART_LIBRARY}
        )
    endif()
    
    add_test(
        NAME test_cpu_vs_gpu_performance
        COMMAND test_cpu_vs_gpu_performance
    )
    
    # 设置测试超时时间
    set_tests_properties(test_cpu_vs_gpu_performance
        PROPERTIES 
        TIMEOUT 300
        LABELS "performance;cpu_vs_gpu"
    )
    
    # PCHIP GPU性能测试
    add_executable(test_pchip_gpu_performance
        test_pchip_gpu_performance.cpp
    )
    
    target_link_libraries(test_pchip_gpu_performance
        PRIVATE
            interpolation_service
            common_utilities
            GTest::gtest
            GTest::gtest_main
    )
    
    # 添加CUDA支持
    if(CUDA_FOUND)
        target_link_libraries(test_pchip_gpu_performance
            PRIVATE
                ${CUDA_LIBRARIES}
                ${CUDA_CUDART_LIBRARY}
        )
    endif()
    
    add_test(
        NAME test_pchip_gpu_performance
        COMMAND test_pchip_gpu_performance
    )
    
    # 设置测试超时时间
    set_tests_properties(test_pchip_gpu_performance
        PROPERTIES 
        TIMEOUT 300
        LABELS "performance;pchip;gpu"
    )
endif() 