#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include "../src/gpu/gpu_interpolation_engine.cpp"
#include "core_services/common_data_types.h"
#include "common_utils/gpu/oscean_gpu_framework.h"

using namespace oscean::core_services::interpolation::gpu;
using namespace oscean::core_services;
using namespace oscean::common_utils::gpu;

class GPUBasicPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 尝试初始化GPU框架
        try {
            gpuAvailable_ = OSCEANGPUFramework::initialize();
            if (gpuAvailable_) {
                auto devices = OSCEANGPUFramework::getAvailableDevices();
                std::cout << "检测到 " << devices.size() << " 个GPU设备" << std::endl;
                for (const auto& device : devices) {
                    std::cout << "  - " << device.name << std::endl;
                }
            } else {
                std::cout << "GPU不可用，将使用模拟测试" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "GPU初始化失败: " << e.what() << std::endl;
            gpuAvailable_ = false;
        }
    }
    
    void TearDown() override {
        if (gpuAvailable_) {
            OSCEANGPUFramework::shutdown();
        }
    }
    
    // 创建测试网格数据
    boost::shared_ptr<GridData> createTestGrid(size_t rows, size_t cols) {
        GridDefinition def;
        def.rows = rows;
        def.cols = cols;
        
        // 创建GridData，使用默认的DataType和bandCount
        auto grid = boost::make_shared<GridData>(def, DataType::Float64, 1);
        
        // 填充测试数据
        double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t i = 0; i < rows * cols; ++i) {
            data[i] = static_cast<double>(i) / (rows * cols);
        }
        
        return grid;
    }
    
    bool gpuAvailable_ = false;
};

// 测试GPU插值引擎的基本功能
TEST_F(GPUBasicPerformanceTest, BasicInterpolation) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU不可用，跳过测试";
    }
    
    // 创建GPU插值引擎
    auto engine = std::make_unique<GPUInterpolationEngine>();
    
    // 创建测试数据
    auto grid = createTestGrid(100, 100);

    // 设置插值参数
    GPUInterpolationParams params;
    params.sourceData = grid;
    params.outputWidth = 200;
    params.outputHeight = 200;
    params.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;

    // 设置执行上下文
    GPUExecutionContext context;
    context.deviceId = 0;

    // 执行插值
    auto start = std::chrono::high_resolution_clock::now();
    auto result = engine->execute(params, context);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "GPU插值耗时: " << duration.count() << " 微秒" << std::endl;
    
    // 验证结果
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.data.width, params.outputWidth);
    EXPECT_EQ(result.data.height, params.outputHeight);
    EXPECT_EQ(result.data.interpolatedData.size(), params.outputWidth * params.outputHeight);
}

// 测试不同大小的数据集性能
TEST_F(GPUBasicPerformanceTest, PerformanceScaling) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU不可用，跳过测试";
    }
    
    auto engine = std::make_unique<GPUInterpolationEngine>();
    
    std::vector<size_t> gridSizes = {50, 100, 200, 500};
    std::vector<size_t> outputSizes = {100, 200, 400, 1000};
    
    std::cout << "\n性能测试结果:" << std::endl;
    std::cout << std::setw(15) << "输入网格大小" 
              << std::setw(15) << "输出网格大小" 
              << std::setw(15) << "耗时(ms)" 
              << std::setw(20) << "吞吐量(MB/s)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    GPUExecutionContext context;
    context.deviceId = 0;

    for (size_t i = 0; i < gridSizes.size(); ++i) {
        auto gridSize = gridSizes[i];
        auto outputSize = outputSizes[i];
        
        auto grid = createTestGrid(gridSize, gridSize);

        GPUInterpolationParams params;
        params.sourceData = grid;
        params.outputWidth = outputSize;
        params.outputHeight = outputSize;
        params.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine->execute(params, context);
        auto end = std::chrono::high_resolution_clock::now();

        if (result.success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double ms = duration.count() / 1000.0;
            double throughput = (outputSize * outputSize * sizeof(float)) / (ms / 1000.0) / (1024 * 1024);

            std::cout << std::setw(15) << gridSize << "x" << gridSize
                      << std::setw(15) << outputSize << "x" << outputSize
                      << std::setw(15) << std::fixed << std::setprecision(2) << ms
                      << std::setw(20) << std::fixed << std::setprecision(2) 
                      << throughput << std::endl;
        } else {
            std::cout << std::setw(15) << gridSize << "x" << gridSize
                      << std::setw(15) << outputSize << "x" << outputSize
                      << std::setw(15) << "失败"
                      << std::setw(20) << "-" << std::endl;
        }
    }
}

// 测试GPU与CPU性能对比（模拟）
TEST_F(GPUBasicPerformanceTest, GPUvsCPUComparison) {
    std::cout << "\nGPU vs CPU 性能对比（模拟）:" << std::endl;
    
    // 创建测试数据
    auto grid = createTestGrid(500, 500);

    // 设置插值参数
    GPUInterpolationParams params;
    params.sourceData = grid;
    params.outputWidth = 1000;
    params.outputHeight = 1000;
    params.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;

    // 模拟CPU时间（假设CPU单线程处理）
    auto outputPixels = params.outputWidth * params.outputHeight;
    auto cpuTimePerPixel = 0.1; // 微秒
    auto estimatedCPUTime = outputPixels * cpuTimePerPixel / 1000.0; // 毫秒

    if (gpuAvailable_) {
        auto engine = std::make_unique<GPUInterpolationEngine>();
        
        GPUExecutionContext context;
        context.deviceId = 0;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine->execute(params, context);
        auto end = std::chrono::high_resolution_clock::now();

        if (result.success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double gpuTime = duration.count() / 1000.0; // 毫秒

            std::cout << "输出大小: " << params.outputWidth << "x" << params.outputHeight << std::endl;
            std::cout << "CPU预估时间: " << estimatedCPUTime << " ms" << std::endl;
            std::cout << "GPU实际时间: " << gpuTime << " ms" << std::endl;
            std::cout << "GPU内核时间: " << result.data.gpuTimeMs << " ms" << std::endl;
            std::cout << "内存传输时间: " << result.data.memoryTransferTimeMs << " ms" << std::endl;
            std::cout << "加速比: " << estimatedCPUTime / gpuTime << "x" << std::endl;
        } else {
            std::cout << "GPU插值失败: " << result.errorMessage << std::endl;
        }
    } else {
        std::cout << "GPU不可用，仅显示CPU预估时间: " << estimatedCPUTime << " ms" << std::endl;
    }
}

// 测试不同插值方法的性能
TEST_F(GPUBasicPerformanceTest, DifferentMethods) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU不可用，跳过测试";
    }

    auto engine = std::make_unique<GPUInterpolationEngine>();
    
    // 创建测试数据
    auto grid = createTestGrid(256, 256);
    
    std::vector<oscean::core_services::interpolation::InterpolationMethod> methods = {
        oscean::core_services::interpolation::InterpolationMethod::BILINEAR,
        oscean::core_services::interpolation::InterpolationMethod::NEAREST_NEIGHBOR,
        oscean::core_services::interpolation::InterpolationMethod::CUBIC_SPLINE_1D
    };
    
    std::vector<std::string> methodNames = {
        "双线性插值",
        "最近邻插值",
        "立方样条插值"
    };
    
    std::cout << "\n不同插值方法性能对比:" << std::endl;
    std::cout << std::setw(20) << "插值方法" 
              << std::setw(15) << "耗时(ms)" 
              << std::setw(15) << "GPU时间(ms)" 
              << std::setw(20) << "内存传输时间(ms)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    GPUExecutionContext context;
    context.deviceId = 0;

    for (size_t i = 0; i < methods.size(); ++i) {
        GPUInterpolationParams params;
        params.sourceData = grid;
        params.outputWidth = 512;
        params.outputHeight = 512;
        params.method = methods[i];
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine->execute(params, context);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (result.success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double totalMs = duration.count() / 1000.0;
            
            std::cout << std::setw(20) << methodNames[i]
                      << std::setw(15) << std::fixed << std::setprecision(2) << totalMs
                      << std::setw(15) << result.data.gpuTimeMs
                      << std::setw(20) << result.data.memoryTransferTimeMs << std::endl;
        } else {
            std::cout << std::setw(20) << methodNames[i]
                      << std::setw(15) << "失败"
                      << std::setw(15) << "-"
                      << std::setw(20) << "-" << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 