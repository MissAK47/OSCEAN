/**
 * @file test_gpu_interpolation_comprehensive.cpp
 * @brief GPU插值综合测试程序
 */

// Windows头文件顺序修复
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <memory>

#include "interpolation/gpu/gpu_interpolation_engine.h"
#include "interpolation/interpolation_method_mapping.h"
#include "core_services/common_data_types.h"
#include "common_utils/gpu/oscean_gpu_framework.h"

// 只使用接口，不包含实现细节
#include "core_services/interpolation/i_interpolation_service.h"
#include "../src/factory/interpolation_service_factory.h"

// 使用具体的类型而不是using namespace避免冲突
using oscean::common_utils::gpu::OSCEANGPUFramework;
using oscean::common_utils::gpu::ComputeAPI;
using oscean::common_utils::gpu::GPUExecutionContext;

// 核心服务类型
using oscean::core_services::GridData;
using oscean::core_services::GridDefinition;
using oscean::core_services::DataType;
using oscean::core_services::DimensionCoordinateInfo;
using oscean::core_services::CoordinateDimension;
using oscean::core_services::interpolation::IInterpolationService;
using oscean::core_services::interpolation::InterpolationMethodMapping;
using oscean::core_services::interpolation::InterpolationServiceFactory;
using oscean::core_services::interpolation::InterpolationRequest;
using oscean::core_services::interpolation::InterpolationResult;
using oscean::core_services::interpolation::TargetGridDefinition;

// GPU插值类型（使用完整路径避免与基础InterpolationMethod冲突）
namespace gpu = oscean::core_services::interpolation::gpu;

// 使用插值服务的InterpolationMethod而不是common_data_types中的
using InterpolationMethod = oscean::core_services::interpolation::InterpolationMethod;

class GPUInterpolationComprehensiveTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化GPU框架
        try {
            gpuAvailable_ = OSCEANGPUFramework::initialize();
            if (gpuAvailable_) {
                auto devices = OSCEANGPUFramework::getAvailableDevices();
                std::cout << "\n========== GPU环境信息 ==========\n";
                std::cout << "检测到 " << devices.size() << " 个GPU设备:\n";
                for (size_t i = 0; i < devices.size(); ++i) {
                    const auto& device = devices[i];
                    std::cout << "\n设备 " << i << ": " << device.name << std::endl;
                    std::cout << "  - 计算能力: " << device.architecture.majorVersion 
                              << "." << device.architecture.minorVersion << std::endl;
                    std::cout << "  - 全局内存: " 
                              << (device.memoryDetails.totalGlobalMemory / (1024.0 * 1024.0 * 1024.0)) 
                              << " GB" << std::endl;
                    std::cout << "  - SM数量: " << device.computeUnits.multiprocessorCount << std::endl;
                }
                std::cout << "==================================\n\n";
            } else {
                std::cout << "GPU不可用，将跳过GPU测试\n";
            }
        } catch (const std::exception& e) {
            std::cout << "GPU初始化失败: " << e.what() << std::endl;
            gpuAvailable_ = false;
        }
        
        // 创建CPU插值服务用于对比
        cpuService_ = InterpolationServiceFactory::createDefault();
    }
    
    void TearDown() override {
        if (gpuAvailable_) {
            OSCEANGPUFramework::shutdown();
        }
    }
    
    // 创建测试网格数据
    boost::shared_ptr<GridData> createTestGrid(size_t rows, size_t cols, 
                                               const std::string& pattern = "sincos") {
        GridDefinition def;
        def.rows = rows;
        def.cols = cols;
        
        // 设置维度信息
        oscean::core_services::DimensionCoordinateInfo xDim;
        xDim.name = "x";
        xDim.type = oscean::core_services::CoordinateDimension::LON;
        xDim.minValue = 0.0;
        xDim.maxValue = 10.0;
        xDim.coordinates.resize(cols);
        for (size_t i = 0; i < cols; ++i) {
            xDim.coordinates[i] = xDim.minValue + i * (xDim.maxValue - xDim.minValue) / (cols - 1);
        }
        
        oscean::core_services::DimensionCoordinateInfo yDim;
        yDim.name = "y";
        yDim.type = oscean::core_services::CoordinateDimension::LAT;
        yDim.minValue = 0.0;
        yDim.maxValue = 10.0;
        yDim.coordinates.resize(rows);
        for (size_t i = 0; i < rows; ++i) {
            yDim.coordinates[i] = yDim.minValue + i * (yDim.maxValue - yDim.minValue) / (rows - 1);
        }
        
        def.xDimension = xDim;
        def.yDimension = yDim;
        
        auto grid = boost::make_shared<GridData>(def, DataType::Float32, 1);
        
        // 填充测试数据
        float* data = static_cast<float*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                double x = xDim.coordinates[j];
                double y = yDim.coordinates[i];
                
                if (pattern == "sincos") {
                    data[i * cols + j] = static_cast<float>(sin(x) * cos(y));
                } else if (pattern == "peaks") {
                    // MATLAB peaks函数
                    double z = 3 * pow(1 - x, 2) * exp(-pow(x, 2) - pow(y + 1, 2))
                             - 10 * (x / 5 - pow(x, 3) - pow(y, 5)) * exp(-pow(x, 2) - pow(y, 2))
                             - 1.0 / 3 * exp(-pow(x + 1, 2) - pow(y, 2));
                    data[i * cols + j] = static_cast<float>(z);
                } else if (pattern == "linear") {
                    data[i * cols + j] = static_cast<float>(x + y);
                }
            }
        }
        
        return grid;
    }
    
    // 计算误差统计
    struct ErrorStats {
        double maxError = 0.0;
        double avgError = 0.0;
        double rmse = 0.0;
        size_t nanCount = 0;
    };
    
    ErrorStats calculateError(const std::vector<float>& gpu, const std::vector<float>& cpu) {
        ErrorStats stats;
        if (gpu.size() != cpu.size()) {
            std::cerr << "Size mismatch: GPU=" << gpu.size() << ", CPU=" << cpu.size() << std::endl;
            return stats;
        }
        
        double sumError = 0.0;
        double sumSquaredError = 0.0;
        size_t validCount = 0;
        
        for (size_t i = 0; i < gpu.size(); ++i) {
            if (std::isnan(gpu[i]) || std::isnan(cpu[i])) {
                stats.nanCount++;
                continue;
            }
            
            double error = std::abs(gpu[i] - cpu[i]);
            stats.maxError = std::max(stats.maxError, error);
            sumError += error;
            sumSquaredError += error * error;
            validCount++;
        }
        
        if (validCount > 0) {
            stats.avgError = sumError / validCount;
            stats.rmse = std::sqrt(sumSquaredError / validCount);
        }
        
        return stats;
    }
    
    bool gpuAvailable_ = false;
    boost::shared_ptr<IInterpolationService> cpuService_;
};

// 测试所有GPU支持的插值方法
TEST_F(GPUInterpolationComprehensiveTest, AllGPUSupportedMethods) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU不可用，跳过测试";
    }
    
    // 创建测试数据
    auto sourceGrid = createTestGrid(64, 64, "peaks");
    
    // 获取GPU支持的所有方法
    auto gpuMethods = InterpolationMethodMapping::getGPUSupportedMethods();
    
    std::cout << "\n========== GPU插值方法测试 ==========\n";
    std::cout << "源网格: 64x64, 目标网格: 128x128\n";
    std::cout << "=====================================\n\n";
    
    // 创建GPU引擎
    auto gpuEngine = gpu::GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    ASSERT_NE(gpuEngine, nullptr);
    
    // 测试每种GPU方法
    for (auto method : gpuMethods) {
        std::cout << "测试方法: " << InterpolationMethodMapping::toString(method) << std::endl;
        
        // 设置GPU参数
        gpu::GPUInterpolationParams gpuParams;
        gpuParams.sourceData = sourceGrid;
        gpuParams.outputWidth = 128;
        gpuParams.outputHeight = 128;
        gpuParams.method = method;
        gpuParams.fillValue = 0.0f;
        
        // 执行GPU插值
        GPUExecutionContext context;
        context.deviceId = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = gpuEngine->execute(gpuParams, context);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.success) {
            std::cout << "  ✓ 成功" << std::endl;
            std::cout << "    - 总时间: " << duration.count() / 1000.0 << " ms" << std::endl;
            std::cout << "    - GPU时间: " << result.data.gpuTimeMs << " ms" << std::endl;
            std::cout << "    - 传输时间: " << result.data.memoryTransferTimeMs << " ms" << std::endl;
            std::cout << "    - 内存使用: " << result.data.memoryUsedBytes / 1024.0 / 1024.0 << " MB" << std::endl;
            std::cout << "    - 数据范围: [" << result.data.minValue 
                      << ", " << result.data.maxValue << "]" << std::endl;
        } else {
            std::cout << "  ✗ 失败: " << result.errorMessage << std::endl;
        }
        std::cout << std::endl;
    }
}

// GPU vs CPU性能和精度对比测试
TEST_F(GPUInterpolationComprehensiveTest, GPUvsCPUComparison) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU不可用，跳过测试";
    }
    
    std::vector<std::pair<size_t, size_t>> testSizes = {
        {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512}
    };
    
    std::cout << "\n========== GPU vs CPU 性能对比 ==========\n";
    std::cout << std::setw(12) << "源尺寸" 
              << std::setw(12) << "目标尺寸"
              << std::setw(12) << "CPU(ms)"
              << std::setw(12) << "GPU(ms)"
              << std::setw(12) << "加速比"
              << std::setw(12) << "最大误差"
              << std::setw(12) << "RMSE" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    auto gpuEngine = gpu::GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    
    for (const auto& [srcSize, dstSize] : testSizes) {
        auto sourceGrid = createTestGrid(srcSize, srcSize, "sincos");
        
        // GPU插值
        gpu::GPUInterpolationParams gpuParams;
        gpuParams.sourceData = sourceGrid;
        gpuParams.outputWidth = dstSize * 2;
        gpuParams.outputHeight = dstSize * 2;
        gpuParams.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        auto gpuStart = std::chrono::high_resolution_clock::now();
        auto gpuResult = gpuEngine->execute(gpuParams, context);
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        
        if (!gpuResult.success) {
            std::cout << srcSize << "x" << srcSize << " GPU失败: " 
                      << gpuResult.errorMessage << std::endl;
            continue;
        }
        
        auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(gpuEnd - gpuStart).count() / 1000.0;
        
        // CPU插值
        InterpolationRequest cpuRequest;
        cpuRequest.sourceGrid = sourceGrid;
        cpuRequest.method = InterpolationMethod::BILINEAR;
        
        // 创建目标网格定义
        TargetGridDefinition targetGrid;
        DimensionCoordinateInfo targetX, targetY;
        targetX.name = "x";
        targetX.type = CoordinateDimension::LON;
        targetX.minValue = 0.0;
        targetX.maxValue = 10.0;
        targetX.coordinates.resize(dstSize * 2);
        for (size_t i = 0; i < dstSize * 2; ++i) {
            targetX.coordinates[i] = targetX.minValue + 
                i * (targetX.maxValue - targetX.minValue) / (dstSize * 2 - 1);
        }
        targetY = targetX;
        targetY.name = "y";
        targetY.type = CoordinateDimension::LAT;
        targetGrid.dimensions = {targetX, targetY};
        cpuRequest.target = targetGrid;
        
        auto cpuStart = std::chrono::high_resolution_clock::now();
        auto cpuFuture = cpuService_->interpolateAsync(cpuRequest);
        auto cpuResult = cpuFuture.get();
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        
        auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count() / 1000.0;
        
        // 计算误差（如果CPU结果可用）
        ErrorStats errorStats;
        if (cpuResult.statusCode == 0 && std::holds_alternative<GridData>(cpuResult.data)) {
            const auto& cpuGrid = std::get<GridData>(cpuResult.data);
            std::vector<float> cpuData;
            cpuData.reserve(cpuGrid.getUnifiedBufferSize() / sizeof(float));
            
            const float* cpuPtr = static_cast<const float*>(cpuGrid.getDataPtr());
            for (size_t i = 0; i < cpuGrid.getUnifiedBufferSize() / sizeof(float); ++i) {
                cpuData.push_back(cpuPtr[i]);
            }
            
            errorStats = calculateError(gpuResult.data.interpolatedData, cpuData);
        }
        
        // 输出结果
        std::cout << std::setw(12) << (std::to_string(srcSize) + "x" + std::to_string(srcSize))
                  << std::setw(12) << (std::to_string(dstSize*2) + "x" + std::to_string(dstSize*2))
                  << std::setw(12) << std::fixed << std::setprecision(2) << cpuTime
                  << std::setw(12) << gpuTime
                  << std::setw(12) << std::setprecision(1) << (cpuTime / gpuTime) << "x"
                  << std::setw(12) << std::scientific << std::setprecision(2) << errorStats.maxError
                  << std::setw(12) << errorStats.rmse << std::endl;
    }
    std::cout << std::string(84, '-') << std::endl;
}

// 测试GPU批量插值
TEST_F(GPUInterpolationComprehensiveTest, BatchInterpolation) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU不可用，跳过测试";
    }
    
    std::cout << "\n========== GPU批量插值测试 ==========\n";
    
    // 创建批量插值引擎
    auto batchEngine = gpu::GPUInterpolationEngineFactory::createBatch(ComputeAPI::CUDA);
    ASSERT_NE(batchEngine, nullptr);
    
    // 准备批量数据
    std::vector<gpu::GPUInterpolationParams> batchParams;
    for (int i = 0; i < 10; ++i) {
        auto grid = createTestGrid(32 + i * 8, 32 + i * 8, "linear");
        
        gpu::GPUInterpolationParams params;
        params.sourceData = grid;
        params.outputWidth = 64;
        params.outputHeight = 64;
        params.method = InterpolationMethod::BILINEAR;
        
        batchParams.push_back(params);
    }
    
    // 执行批量插值
    GPUExecutionContext context;
    context.deviceId = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = batchEngine->execute(batchParams, context);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "批量大小: " << batchParams.size() << std::endl;
    std::cout << "总时间: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "平均时间: " << duration.count() / 1000.0 / batchParams.size() << " ms/item" << std::endl;
    
    if (result.success) {
        std::cout << "成功处理: " << result.data.size() << " 个插值任务" << std::endl;
    } else {
        std::cout << "批量插值失败: " << result.errorMessage << std::endl;
    }
}

// 测试大规模数据插值
TEST_F(GPUInterpolationComprehensiveTest, LargeScaleInterpolation) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU不可用，跳过测试";
    }
    
    std::cout << "\n========== 大规模数据GPU插值测试 ==========\n";
    
    // 测试不同大小的数据
    std::vector<size_t> sizes = {1024, 2048};
    
    auto gpuEngine = gpu::GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    
    for (auto size : sizes) {
        std::cout << "\n测试 " << size << "x" << size << " -> " 
                  << size*2 << "x" << size*2 << " 插值" << std::endl;
        
        auto sourceGrid = createTestGrid(size, size, "peaks");
        
        // 估算内存需求
        size_t memRequired = gpuEngine->estimateInterpolationMemory(
            size, size, size*2, size*2, InterpolationMethod::BILINEAR);
        
        std::cout << "预计GPU内存需求: " << memRequired / 1024.0 / 1024.0 << " MB" << std::endl;
        
        gpu::GPUInterpolationParams params;
        params.sourceData = sourceGrid;
        params.outputWidth = size * 2;
        params.outputHeight = size * 2;
        params.method = InterpolationMethod::BILINEAR;
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = gpuEngine->execute(params, context);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (result.success) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            size_t totalPixels = params.outputWidth * params.outputHeight;
            double throughput = totalPixels / (duration.count() / 1000.0) / 1e6; // 百万像素/秒
            
            std::cout << "  ✓ 成功" << std::endl;
            std::cout << "    - 总时间: " << duration.count() << " ms" << std::endl;
            std::cout << "    - GPU核心时间: " << result.data.gpuTimeMs << " ms" << std::endl;
            std::cout << "    - 吞吐量: " << std::fixed << std::setprecision(1) 
                      << throughput << " Mpixels/s" << std::endl;
            std::cout << "    - 实际内存使用: " << result.data.memoryUsedBytes / 1024.0 / 1024.0 
                      << " MB" << std::endl;
        } else {
            std::cout << "  ✗ 失败: " << result.errorMessage << std::endl;
        }
    }
}

// 测试特殊情况和边界条件
TEST_F(GPUInterpolationComprehensiveTest, EdgeCases) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU不可用，跳过测试";
    }
    
    std::cout << "\n========== GPU插值边界条件测试 ==========\n";
    
    auto gpuEngine = gpu::GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    
    // 测试1: 非常小的网格
    {
        std::cout << "测试1: 最小网格 (2x2 -> 4x4)" << std::endl;
        auto smallGrid = createTestGrid(2, 2, "linear");
        
        gpu::GPUInterpolationParams params;
        params.sourceData = smallGrid;
        params.outputWidth = 4;
        params.outputHeight = 4;
        params.method = InterpolationMethod::BILINEAR;
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        auto result = gpuEngine->execute(params, context);
        EXPECT_TRUE(result.success) << "小网格插值应该成功";
        if (result.success) {
            std::cout << "  ✓ 成功处理最小网格" << std::endl;
        }
    }
    
    // 测试2: 非方形网格
    {
        std::cout << "\n测试2: 非方形网格 (128x32 -> 256x64)" << std::endl;
        auto rectGrid = createTestGrid(32, 128, "sincos");
        
        gpu::GPUInterpolationParams params;
        params.sourceData = rectGrid;
        params.outputWidth = 256;
        params.outputHeight = 64;
        params.method = InterpolationMethod::BILINEAR;
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        auto result = gpuEngine->execute(params, context);
        EXPECT_TRUE(result.success) << "非方形网格插值应该成功";
        if (result.success) {
            std::cout << "  ✓ 成功处理非方形网格" << std::endl;
        }
    }
    
    // 测试3: 不同插值方法的切换
    {
        std::cout << "\n测试3: 动态切换插值方法" << std::endl;
        auto grid = createTestGrid(64, 64, "peaks");
        
        std::vector<InterpolationMethod> methods = {
            InterpolationMethod::NEAREST_NEIGHBOR,
            InterpolationMethod::BILINEAR,
            InterpolationMethod::BICUBIC
        };
        
        for (auto method : methods) {
            gpuEngine->setInterpolationMethod(method);
            
            gpu::GPUInterpolationParams params;
            params.sourceData = grid;
            params.outputWidth = 128;
            params.outputHeight = 128;
            params.method = method;
            
            GPUExecutionContext context;
            context.deviceId = 0;
            
            auto result = gpuEngine->execute(params, context);
            std::cout << "  " << InterpolationMethodMapping::toString(method) 
                      << ": " << (result.success ? "✓ 成功" : "✗ 失败") << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 