/**
 * @file test_gpu_interpolation.cpp
 * @brief GPU插值功能测试
 */

#include <gtest/gtest.h>
#include <interpolation/gpu/gpu_interpolation_engine.h>
#include <common_utils/gpu/oscean_gpu_framework.h>
#include <common_data_types.h>
#include <memory>
#include <chrono>
#include <random>
#include <cmath>
#include <iostream>

using namespace oscean;
using namespace oscean::interpolation::gpu;
using namespace oscean::common_utils::gpu;

class GPUInterpolationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化GPU框架
        if (!OSCEANGPUFramework::initialize()) {
            GTEST_SKIP() << "GPU framework initialization failed";
        }
    }
    
    void TearDown() override {
        // 清理
    }
    
    // 辅助函数：创建测试网格数据
    std::shared_ptr<GridData> createTestGridData(int width, int height) {
        auto gridData = std::make_shared<GridData>();
        
        GridDefinition def;
        def.rows = height;
        def.cols = width;
        def.minX = -180.0;
        def.maxX = 180.0;
        def.minY = -90.0;
        def.maxY = 90.0;
        
        gridData->setDefinition(def);
        
        // 创建测试数据（正弦波模式）
        std::vector<float> data(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float fx = static_cast<float>(x) / width;
                float fy = static_cast<float>(y) / height;
                data[y * width + x] = std::sin(fx * 2 * M_PI) * std::cos(fy * 2 * M_PI);
            }
        }
        
        gridData->setData(data.data(), data.size());
        return gridData;
    }
};

// 测试GPU插值引擎创建
TEST_F(GPUInterpolationTest, EngineCreation) {
    std::cout << "Testing GPU interpolation engine creation..." << std::endl;
    
    // 创建CUDA插值引擎
    auto cudaEngine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    EXPECT_NE(cudaEngine, nullptr);
    
    // 创建OpenCL插值引擎
    auto openclEngine = GPUInterpolationEngineFactory::create(ComputeAPI::OPENCL);
    EXPECT_NE(openclEngine, nullptr);
    
    // 获取支持的插值方法
    if (cudaEngine) {
        auto methods = cudaEngine->getSupportedMethods();
        EXPECT_FALSE(methods.empty());
        EXPECT_TRUE(std::find(methods.begin(), methods.end(), 
                             InterpolationMethod::BILINEAR) != methods.end());
    }
}

// 测试双线性插值
TEST_F(GPUInterpolationTest, BilinearInterpolation) {
    std::cout << "Testing GPU bilinear interpolation..." << std::endl;
    
    // 创建测试数据
    const int sourceWidth = 100;
    const int sourceHeight = 100;
    auto sourceData = createTestGridData(sourceWidth, sourceHeight);
    
    // 创建插值参数
    GPUInterpolationParams params;
    params.sourceData = sourceData;
    params.outputWidth = 200;
    params.outputHeight = 200;
    params.outputBounds = sourceData->getDefinition().getBounds();
    params.method = InterpolationMethod::BILINEAR;
    params.enableCaching = true;
    params.useTextureMemory = false;
    params.fillValue = 0.0f;
    
    // 创建GPU插值引擎
    auto engine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    ASSERT_NE(engine, nullptr);
    
    // 验证参数
    EXPECT_TRUE(engine->validateParams(params));
    
    // 执行插值
    GPUExecutionContext context;
    context.deviceId = 0;
    context.stream = nullptr;
    
    auto future = engine->executeAsync(params, context);
    auto result = future.get();
    
    // 检查结果
    EXPECT_EQ(result.status, GPUError::SUCCESS);
    EXPECT_EQ(result.width, params.outputWidth);
    EXPECT_EQ(result.height, params.outputHeight);
    EXPECT_EQ(result.interpolatedData.size(), 
              params.outputWidth * params.outputHeight);
    
    // 验证插值质量
    EXPECT_GE(result.minValue, -1.0f);
    EXPECT_LE(result.minValue, 1.0f);
    EXPECT_GE(result.maxValue, -1.0f);
    EXPECT_LE(result.maxValue, 1.0f);
    EXPECT_EQ(result.nanCount, 0u);
    
    std::cout << "Interpolation completed in " 
              << result.gpuTimeMs << " ms" << std::endl;
}

// 测试不同插值方法
TEST_F(GPUInterpolationTest, MultipleInterpolationMethods) {
    std::cout << "Testing multiple interpolation methods..." << std::endl;
    
    auto sourceData = createTestGridData(50, 50);
    auto engine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    
    if (!engine) {
        GTEST_SKIP() << "GPU engine not available";
    }
    
    std::vector<InterpolationMethod> methods = {
        InterpolationMethod::NEAREST,
        InterpolationMethod::BILINEAR,
        InterpolationMethod::BICUBIC
    };
    
    for (auto method : methods) {
        GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = 100;
        params.outputHeight = 100;
        params.outputBounds = sourceData->getDefinition().getBounds();
        params.method = method;
        params.fillValue = 0.0f;
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        auto future = engine->executeAsync(params, context);
        auto result = future.get();
        
        EXPECT_EQ(result.status, GPUError::SUCCESS);
        std::cout << "Method: " << static_cast<int>(method)
                  << ", Time: " << result.gpuTimeMs << " ms" << std::endl;
    }
}

// 测试批量插值
TEST_F(GPUInterpolationTest, BatchInterpolation) {
    std::cout << "Testing batch GPU interpolation..." << std::endl;
    
    auto batchEngine = GPUInterpolationEngineFactory::createBatch(ComputeAPI::CUDA);
    ASSERT_NE(batchEngine, nullptr);
    
    // 创建批量任务
    std::vector<GPUInterpolationParams> batch;
    for (int i = 0; i < 4; ++i) {
        auto sourceData = createTestGridData(50 + i * 10, 50 + i * 10);
        
        GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = 100;
        params.outputHeight = 100;
        params.outputBounds = sourceData->getDefinition().getBounds();
        params.method = InterpolationMethod::BILINEAR;
        params.fillValue = 0.0f;
        
        batch.push_back(params);
    }
    
    // 执行批量插值
    GPUExecutionContext context;
    context.deviceId = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto futures = batchEngine->executeBatchAsync(batch, context);
    
    // 收集结果
    std::vector<GPUInterpolationResult> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count();
    
    // 验证结果
    EXPECT_EQ(results.size(), batch.size());
    for (const auto& result : results) {
        EXPECT_EQ(result.status, GPUError::SUCCESS);
    }
    
    std::cout << "Batch interpolation completed in " 
              << totalTime << " ms" << std::endl;
}

// 性能基准测试
TEST_F(GPUInterpolationTest, PerformanceBenchmark) {
    std::cout << "Running interpolation performance benchmark..." << std::endl;
    
    struct TestCase {
        int sourceSize;
        int targetSize;
        const char* name;
    };
    
    std::vector<TestCase> testCases = {
        {100, 200, "Small (100x100 -> 200x200)"},
        {500, 1000, "Medium (500x500 -> 1000x1000)"},
        {1000, 2000, "Large (1000x1000 -> 2000x2000)"}
    };
    
    auto engine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    
    if (!engine) {
        GTEST_SKIP() << "GPU engine not available";
    }
    
    for (const auto& testCase : testCases) {
        auto sourceData = createTestGridData(testCase.sourceSize, testCase.sourceSize);
        
        GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = testCase.targetSize;
        params.outputHeight = testCase.targetSize;
        params.outputBounds = sourceData->getDefinition().getBounds();
        params.method = InterpolationMethod::BILINEAR;
        params.fillValue = 0.0f;
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        // 预热
        engine->executeAsync(params, context).get();
        
        // 测量性能
        const int iterations = 10;
        double totalTime = 0.0;
        
        for (int i = 0; i < iterations; ++i) {
            auto future = engine->executeAsync(params, context);
            auto result = future.get();
            totalTime += result.gpuTimeMs;
        }
        
        double avgTime = totalTime / iterations;
        double throughput = (testCase.targetSize * testCase.targetSize) / 
                          (avgTime * 1000.0); // pixels per second
        
        std::cout << testCase.name 
                  << " - Avg time: " << avgTime << " ms"
                  << ", Throughput: " << throughput / 1e6 << " Mpixels/s" << std::endl;
    }
}

// 测试内存需求估算
TEST_F(GPUInterpolationTest, MemoryEstimation) {
    std::cout << "Testing memory requirement estimation..." << std::endl;
    
    auto engine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    
    if (!engine) {
        GTEST_SKIP() << "GPU engine not available";
    }
    
    struct TestCase {
        int sourceWidth, sourceHeight;
        int targetWidth, targetHeight;
        InterpolationMethod method;
    };
    
    std::vector<TestCase> testCases = {
        {100, 100, 200, 200, InterpolationMethod::BILINEAR},
        {100, 100, 200, 200, InterpolationMethod::BICUBIC},
        {500, 500, 1000, 1000, InterpolationMethod::BILINEAR}
    };
    
    for (const auto& tc : testCases) {
        size_t estimated = engine->estimateInterpolationMemory(
            tc.sourceWidth, tc.sourceHeight,
            tc.targetWidth, tc.targetHeight,
            tc.method);
        
        size_t expectedMin = tc.sourceWidth * tc.sourceHeight * sizeof(float) +
                           tc.targetWidth * tc.targetHeight * sizeof(float);
        
        EXPECT_GE(estimated, expectedMin);
        
        std::cout << "Method: " << static_cast<int>(tc.method)
                  << ", Size: " << tc.sourceWidth << "x" << tc.sourceHeight
                  << " -> " << tc.targetWidth << "x" << tc.targetHeight
                  << ", Memory: " << estimated / (1024.0 * 1024.0) << " MB" << std::endl;
    }
}

// 测试边界情况
TEST_F(GPUInterpolationTest, EdgeCases) {
    std::cout << "Testing edge cases..." << std::endl;
    
    auto engine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    
    if (!engine) {
        GTEST_SKIP() << "GPU engine not available";
    }
    
    // 测试空数据
    {
        GPUInterpolationParams params;
        params.sourceData = nullptr;
        EXPECT_FALSE(engine->validateParams(params));
    }
    
    // 测试无效输出尺寸
    {
        auto sourceData = createTestGridData(10, 10);
        GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = 0;
        params.outputHeight = 100;
        EXPECT_FALSE(engine->validateParams(params));
    }
    
    // 测试极小数据
    {
        auto sourceData = createTestGridData(2, 2);
        GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = 10;
        params.outputHeight = 10;
        params.outputBounds = sourceData->getDefinition().getBounds();
        params.method = InterpolationMethod::BILINEAR;
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        auto future = engine->executeAsync(params, context);
        auto result = future.get();
        
        EXPECT_EQ(result.status, GPUError::SUCCESS);
    }
} 