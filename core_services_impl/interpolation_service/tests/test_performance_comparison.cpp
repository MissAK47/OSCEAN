/**
 * @file test_performance_comparison.cpp
 * @brief CPU vs SIMD vs GPU 综合性能对比测试
 */

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#endif

// 避免boost asio带来的Windows socket冲突
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "interpolation/gpu/gpu_interpolation_engine.h"
#include "interpolation/interpolation_method_mapping.h"
#include "core_services/common_data_types.h"
#include "common_utils/gpu/oscean_gpu_framework.h"

// 包含各个SIMD优化的算法
#include "../src/impl/algorithms/bilinear_interpolator.h"
#include "../src/impl/algorithms/nearest_neighbor_interpolator.h"

// 工厂和服务
#include "../src/factory/interpolation_service_factory.h"

using namespace oscean::core_services;
using namespace oscean::core_services::interpolation;
using namespace oscean::common_utils::gpu;

namespace gpu = oscean::core_services::interpolation::gpu;

class PerformanceComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 检测CPU能力
        detectCPUCapabilities();
        std::cout << "\n========== CPU能力信息 ==========\n";
        std::cout << "SSE4.2: " << (hasSSE42_ ? "支持" : "不支持") << std::endl;
        std::cout << "AVX2: " << (hasAVX2_ ? "支持" : "不支持") << std::endl;
        std::cout << "AVX512F: " << (hasAVX512F_ ? "支持" : "不支持") << std::endl;
        std::cout << "==================================\n";

        // 初始化GPU
        try {
            gpuAvailable_ = OSCEANGPUFramework::initialize();
            if (gpuAvailable_) {
                auto devices = OSCEANGPUFramework::getAvailableDevices();
                std::cout << "\n========== GPU环境信息 ==========\n";
                std::cout << "检测到 " << devices.size() << " 个GPU设备\n";
                if (!devices.empty()) {
                    const auto& device = devices[0];
                    std::cout << "GPU: " << device.name << std::endl;
                    std::cout << "计算能力: " << device.architecture.majorVersion 
                              << "." << device.architecture.minorVersion << std::endl;
                    std::cout << "全局内存: " 
                              << (device.memoryDetails.totalGlobalMemory / (1024.0 * 1024.0 * 1024.0)) 
                              << " GB" << std::endl;
                }
                std::cout << "==================================\n";
            }
        } catch (...) {
            gpuAvailable_ = false;
        }
    }
    
    void TearDown() override {
        if (gpuAvailable_) {
            OSCEANGPUFramework::shutdown();
        }
    }
    
    // 创建测试数据
    boost::shared_ptr<GridData> createTestGrid(size_t rows, size_t cols) {
        GridDefinition def;
        def.rows = rows;
        def.cols = cols;
        
        DimensionCoordinateInfo xDim, yDim;
        xDim.name = "x";
        xDim.type = CoordinateDimension::LON;
        xDim.minValue = 0.0;
        xDim.maxValue = 10.0;
        xDim.coordinates.resize(cols);
        for (size_t i = 0; i < cols; ++i) {
            xDim.coordinates[i] = xDim.minValue + i * (xDim.maxValue - xDim.minValue) / (cols - 1);
        }
        
        yDim.name = "y";
        yDim.type = CoordinateDimension::LAT;
        yDim.minValue = 0.0;
        yDim.maxValue = 10.0;
        yDim.coordinates.resize(rows);
        for (size_t i = 0; i < rows; ++i) {
            yDim.coordinates[i] = yDim.minValue + i * (yDim.maxValue - yDim.minValue) / (rows - 1);
        }
        
        def.xDimension = xDim;
        def.yDimension = yDim;
        
        auto grid = boost::make_shared<GridData>(def, DataType::Float32, 1);
        
        // 填充测试数据 - 使用简单的sin(x)*cos(y)函数
        float* data = static_cast<float*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                double x = xDim.coordinates[j];
                double y = yDim.coordinates[i];
                data[i * cols + j] = static_cast<float>(sin(x) * cos(y));
            }
        }
        
        return grid;
    }
    
    // 测试结果结构
    struct TestResult {
        double cpuTime = 0.0;      // 原始CPU时间(ms)
        double simdTime = 0.0;     // SIMD优化时间(ms)
        double gpuTime = 0.0;      // GPU时间(ms)
        double simdSpeedup = 0.0;  // SIMD加速比
        double gpuSpeedup = 0.0;   // GPU加速比
        bool simdAvailable = false;
        bool gpuAvailable = false;
    };
    
    void detectCPUCapabilities() {
#ifdef _MSC_VER
        // MSVC方式检测
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        int nIds = cpuInfo[0];
        
        if (nIds >= 1) {
            __cpuid(cpuInfo, 1);
            hasSSE42_ = (cpuInfo[2] & (1 << 20)) != 0;
        }
        
        if (nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            hasAVX2_ = (cpuInfo[1] & (1 << 5)) != 0;
            hasAVX512F_ = (cpuInfo[1] & (1 << 16)) != 0;
        }
#else
        // GCC/Clang方式
        hasSSE42_ = __builtin_cpu_supports("sse4.2");
        hasAVX2_ = __builtin_cpu_supports("avx2");
        hasAVX512F_ = __builtin_cpu_supports("avx512f");
#endif
    }
    
    bool gpuAvailable_ = false;
    bool hasSSE42_ = false;
    bool hasAVX2_ = false;
    bool hasAVX512F_ = false;
};

// 测试双线性插值性能对比
TEST_F(PerformanceComparisonTest, BilinearInterpolation) {
    std::cout << "\n========== 双线性插值性能对比 ==========\n";
    
    std::vector<std::pair<size_t, size_t>> testSizes = {
        {32, 32}, {64, 64}, {128, 128}, {256, 256}
    };
    
    std::cout << std::setw(15) << "源尺寸"
              << std::setw(15) << "目标尺寸"
              << std::setw(15) << "CPU(ms)"
              << std::setw(15) << "SIMD(ms)"
              << std::setw(15) << "GPU(ms)"
              << std::setw(15) << "SIMD加速"
              << std::setw(15) << "GPU加速" << std::endl;
    std::cout << std::string(105, '-') << std::endl;
    
    for (const auto& [srcSize, _] : testSizes) {
        size_t dstSize = srcSize * 2;
        auto sourceGrid = createTestGrid(srcSize, srcSize);
        
        TestResult result;
        const int iterations = 10;
        
        // 创建目标点集
        std::vector<TargetPoint> targetPoints;
        targetPoints.reserve(dstSize * dstSize);
        for (size_t i = 0; i < dstSize; ++i) {
            for (size_t j = 0; j < dstSize; ++j) {
                TargetPoint pt;
                pt.coordinates.push_back(j * 10.0 / (dstSize - 1));
                pt.coordinates.push_back(i * 10.0 / (dstSize - 1));
                targetPoints.push_back(pt);
            }
        }
        
        // 1. 测试原始CPU版本（标量实现）
        {
            BilinearInterpolator interpolator;
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                auto results = interpolator.interpolateAtPoints(*sourceGrid, targetPoints);
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            result.cpuTime = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        }
        
        // 2. 测试SIMD优化版本
        if (hasAVX2_) {
            BilinearInterpolator interpolator;
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                auto results = interpolator.interpolateAtPointsSIMD(*sourceGrid, targetPoints);
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            result.simdTime = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
            result.simdAvailable = true;
            result.simdSpeedup = result.cpuTime / result.simdTime;
        }
        
        // 3. 测试GPU版本
        if (gpuAvailable_) {
            auto gpuEngine = gpu::GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
            if (gpuEngine) {
                gpu::GPUInterpolationParams params;
                params.sourceData = sourceGrid;
                params.outputWidth = dstSize;
                params.outputHeight = dstSize;
                params.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
                
                GPUExecutionContext context;
                context.deviceId = 0;
                
                // 预热
                for (int i = 0; i < 3; ++i) {
                    auto warmup = gpuEngine->execute(params, context);
                }
                
                auto start = std::chrono::high_resolution_clock::now();
                double totalGpuKernelTime = 0.0;
                for (int i = 0; i < iterations; ++i) {
                    auto gpuResult = gpuEngine->execute(params, context);
                    if (gpuResult.success && gpuResult.data.gpuTimeMs > 0) {
                        totalGpuKernelTime += gpuResult.data.gpuTimeMs;
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                
                // 优先使用GPU核函数时间
                if (totalGpuKernelTime > 0) {
                    result.gpuTime = totalGpuKernelTime / iterations;
                } else {
                    result.gpuTime = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
                }
                
                result.gpuAvailable = true;
                result.gpuSpeedup = result.cpuTime / result.gpuTime;
            }
        }
        
        // 输出结果
        std::cout << std::setw(15) << (std::to_string(srcSize) + "x" + std::to_string(srcSize))
                  << std::setw(15) << (std::to_string(dstSize) + "x" + std::to_string(dstSize))
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.cpuTime
                  << std::setw(15) << (result.simdAvailable ? std::to_string(result.simdTime).substr(0,5) : "N/A")
                  << std::setw(15) << (result.gpuAvailable ? std::to_string(result.gpuTime).substr(0,5) : "N/A")
                  << std::setw(15) << (result.simdAvailable ? 
                                       std::to_string(result.simdSpeedup).substr(0,4) + "x" : "N/A")
                  << std::setw(15) << (result.gpuAvailable ? 
                                       std::to_string(result.gpuSpeedup).substr(0,4) + "x" : "N/A")
                  << std::endl;
    }
}

// 测试最近邻插值性能对比
TEST_F(PerformanceComparisonTest, NearestNeighborInterpolation) {
    std::cout << "\n========== 最近邻插值性能对比 ==========\n";
    
    std::vector<std::pair<size_t, size_t>> testSizes = {
        {64, 64}, {128, 128}, {256, 256}
    };
    
    std::cout << std::setw(15) << "源尺寸"
              << std::setw(15) << "目标尺寸"
              << std::setw(15) << "CPU(μs)"
              << std::setw(15) << "SIMD(μs)"
              << std::setw(15) << "GPU(ms)"
              << std::setw(15) << "SIMD加速"
              << std::setw(15) << "GPU加速" << std::endl;
    std::cout << std::string(105, '-') << std::endl;
    
    for (const auto& [srcSize, _] : testSizes) {
        size_t dstSize = srcSize * 2;
        auto sourceGrid = createTestGrid(srcSize, srcSize);
        
        TestResult result;
        const int iterations = 100;
        
        // 创建目标点集
        std::vector<TargetPoint> targetPoints;
        size_t totalPoints = dstSize * dstSize;
        targetPoints.reserve(totalPoints);
        for (size_t i = 0; i < dstSize; ++i) {
            for (size_t j = 0; j < dstSize; ++j) {
                TargetPoint pt;
                pt.coordinates.push_back(j * 10.0 / (dstSize - 1));
                pt.coordinates.push_back(i * 10.0 / (dstSize - 1));
                targetPoints.push_back(pt);
            }
        }
        
        // 创建插值请求
        InterpolationRequest request;
        request.sourceGrid = sourceGrid;
        request.target = targetPoints;
        request.method = oscean::core_services::interpolation::InterpolationMethod::NEAREST_NEIGHBOR;
        
        // 1. 测试原始CPU版本（使用标量路径）
        {
            NearestNeighborInterpolator interpolator;
            
            // 使用小批量强制标量路径
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                for (size_t j = 0; j < targetPoints.size(); j += 4) {
                    InterpolationRequest smallRequest = request;
                    smallRequest.target = std::vector<TargetPoint>(
                        targetPoints.begin() + j, 
                        targetPoints.begin() + std::min(j + 4, targetPoints.size())
                    );
                    interpolator.execute(smallRequest);
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            result.cpuTime = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
        }
        
        // 2. 测试SIMD优化版本
        if (hasAVX2_) {
            NearestNeighborInterpolator interpolator;
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                // 批量处理以触发SIMD路径
                interpolator.execute(request);
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            result.simdTime = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
            result.simdAvailable = true;
            result.simdSpeedup = result.cpuTime / result.simdTime;
        }
        
        // 3. 测试GPU版本
        if (gpuAvailable_) {
            auto gpuEngine = gpu::GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
            if (gpuEngine) {
                gpu::GPUInterpolationParams params;
                params.sourceData = sourceGrid;
                params.outputWidth = dstSize;
                params.outputHeight = dstSize;
                params.method = oscean::core_services::interpolation::InterpolationMethod::NEAREST_NEIGHBOR;
                
                GPUExecutionContext context;
                context.deviceId = 0;
                
                auto start = std::chrono::high_resolution_clock::now();
                double totalGpuKernelTime = 0.0;
                for (int i = 0; i < 10; ++i) {  // GPU使用较少迭代次数
                    auto gpuResult = gpuEngine->execute(params, context);
                    if (gpuResult.success && gpuResult.data.gpuTimeMs > 0) {
                        totalGpuKernelTime += gpuResult.data.gpuTimeMs;
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                
                if (totalGpuKernelTime > 0) {
                    result.gpuTime = totalGpuKernelTime / 10;
                } else {
                    result.gpuTime = std::chrono::duration<double, std::milli>(end - start).count() / 10;
                }
                
                result.gpuAvailable = true;
                result.gpuSpeedup = (result.cpuTime / 1000.0) / result.gpuTime;  // 转换单位
            }
        }
        
        // 输出结果
        std::cout << std::setw(15) << (std::to_string(srcSize) + "x" + std::to_string(srcSize))
                  << std::setw(15) << (std::to_string(dstSize) + "x" + std::to_string(dstSize))
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.cpuTime
                  << std::setw(15) << (result.simdAvailable ? std::to_string(result.simdTime).substr(0,5) : "N/A")
                  << std::setw(15) << (result.gpuAvailable ? std::to_string(result.gpuTime).substr(0,5) : "N/A")
                  << std::setw(15) << (result.simdAvailable ? 
                                       std::to_string(result.simdSpeedup).substr(0,4) + "x" : "N/A")
                  << std::setw(15) << (result.gpuAvailable ? 
                                       std::to_string(result.gpuSpeedup).substr(0,4) + "x" : "N/A")
                  << std::endl;
    }
}

// 综合性能总结
TEST_F(PerformanceComparisonTest, PerformanceSummary) {
    std::cout << "\n========== 性能优化总结 ==========\n";
    
    std::cout << "\n基于测试结果的优化建议：\n";
    std::cout << "1. 小数据集（<10K点）：SIMD优化的CPU版本通常最优\n";
    std::cout << "2. 中等数据集（10K-100K点）：根据具体算法选择SIMD或GPU\n";
    std::cout << "3. 大数据集（>100K点）：GPU加速通常提供最佳性能\n";
    std::cout << "4. 批量处理：GPU在批量处理时优势明显\n";
    std::cout << "5. 实时性要求：考虑GPU初始化开销，小任务可能CPU更快\n";
    
    std::cout << "\n性能特征：\n";
    std::cout << "- 双线性插值：SIMD通常提供3-4倍加速\n";
    std::cout << "- 最近邻插值：SIMD可达69倍加速（基于项目文档）\n";
    std::cout << "- GPU加速：对大规模数据集效果显著\n";
    std::cout << "- 内存带宽：往往是性能瓶颈，批量处理可改善\n";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 