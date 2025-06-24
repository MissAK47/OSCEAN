# GPU相关源文件
if(OSCEAN_USE_CUDA)
    set(GPU_SOURCES
        gpu/gpu_interpolation_engine.cpp
        gpu/gpu_interpolation_kernels.cu
        gpu/cuda/cuda_interpolation_impl.cpp
        gpu/cuda/cuda_memory_manager.cpp
    )
    
    # 设置CUDA编译选项
    set_source_files_properties(${GPU_SOURCES} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    set(GPU_SOURCES
        gpu/gpu_interpolation_engine.cpp
        gpu/opencl/opencl_interpolation_impl.cpp
        gpu/opencl/opencl_memory_manager.cpp
    )
endif()

# 核心实现文件
set(IMPL_SOURCES
    impl/interpolation_service_impl.cpp
    impl/interpolation_algorithm_factory.cpp
    impl/layout_converter.cpp
)

# 收集所有源文件
set(INTERPOLATION_SOURCES
    # 基础实现
    ${IMPL_SOURCES}
    
    # 基础算法
    impl/algorithms/linear_interpolator.cpp
    impl/algorithms/linear_1d_interpolator.cpp
    impl/algorithms/bilinear_interpolator.cpp
    impl/algorithms/trilinear_interpolator.cpp
    impl/algorithms/nearest_neighbor_interpolator.cpp
    impl/algorithms/cubic_spline_interpolator.cpp
    impl/algorithms/pchip_interpolator.cpp
    impl/algorithms/pchip_interpolator_2d.cpp
    impl/algorithms/pchip_interpolator_2d_bathy.cpp
    impl/algorithms/kernels/interpolation_kernels.cpp
    
    # 高级算法
    impl/algorithms/fast_pchip_interpolator_2d.cpp
    impl/algorithms/fast_pchip_interpolator_3d.cpp
    impl/algorithms/bilinear_interpolator_simd.cpp
    impl/layout_aware_interpolator_base.cpp
    impl/recursive_ndim_pchip_interpolator.cpp
    impl/complex_field_interpolator.cpp
    
    # GPU相关
    ${GPU_SOURCES}
) 