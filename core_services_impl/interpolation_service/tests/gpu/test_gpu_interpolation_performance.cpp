/**
 * @file test_gpu_interpolation_performance.cpp
 * @brief GPU插值性能测试
 */

#include <gtest/gtest.h>
#include <interpolation/gpu/gpu_interpolation_engine.h>
#include <common_utils/gpu/oscean_gpu_framework.h>
#include <common_utils/gpu/multi_gpu_scheduler.h>
#include <common_data_types.h>
#include <memory>
#include <chrono>
#include <random>
#include <numeric>
#include <iostream>

using namespace oscean;
using namespace oscean::interpolation::gpu;
using namespace oscean::common_utils::gpu;

class GPUInterpolationPerformanceTest : public ::testing::Test {
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
    
    // 辅助函数：创建大规模测试数据
    std::shared_ptr<GridData> createLargeTestData(int width, int height) {
        auto gridData = std::make_shared<GridData>();
        
        GridDefinition def;
        def.rows = height;
        def.cols = width;
        def.minX = -180.0;
        def.maxX = 180.0;
        def.minY = -90.0;
        def.maxY = 90.0;
        
        gridData->setDefinition(def);
        
        // 创建复杂的测试数据模式
        std::vector<float> data(width * height);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float fx = static_cast<float>(x) / width;
                float fy = static_cast<float>(y) / height;
                // 复杂函数 + 噪声
                data[y * width + x] = std::sin(fx * 10) * std::cos(fy * 10) + 
                                     0.1f * dis(gen);
            }
        }
        
        gridData->setData(data.data(), data.size());
        return gridData;
    }
};

// 测试GPU框架初始化
TEST_F(GPUInterpolationPerformanceTest, GPUFrameworkInitialization) {
    std::cout << "Testing GPU framework initialization..." << std::endl;
    
    auto deviceInfo = OSCEANGPUFramework::getDeviceInfo(0);
    EXPECT_TRUE(deviceInfo.isValid);
    
    if (deviceInfo.isValid) {
        std::cout << "GPU Device: " << deviceInfo.name << std::endl;
        std::cout << "Compute Capability: " << deviceInfo.major << "." 
                  << deviceInfo.minor << std::endl;
        std::cout << "Global Memory: " << deviceInfo.totalGlobalMem / (1024*1024) 
                  << " MB" << std::endl;
    }
}

// 双线性插值性能对比测试
TEST_F(GPUInterpolationPerformanceTest, BilinearPerformanceComparison) {
    std::cout << "Testing bilinear interpolation performance..." << std::endl;
    
    struct TestCase {
        int sourceSize;
        int targetSize;
        const char* description;
    };
    
    std::vector<TestCase> testCases = {
        {512, 1024, "512x512 -> 1024x1024"},
        {1024, 2048, "1024x1024 -> 2048x2048"},
        {2048, 4096, "2048x2048 -> 4096x4096"}
    };
    
    auto engine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    ASSERT_NE(engine, nullptr);
    
    for (const auto& tc : testCases) {
        auto sourceData = createLargeTestData(tc.sourceSize, tc.sourceSize);
        
        GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = tc.targetSize;
        params.outputHeight = tc.targetSize;
        params.outputBounds = sourceData->getDefinition().getBounds();
        params.method = InterpolationMethod::BILINEAR;
        params.fillValue = 0.0f;
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        // 预热
        for (int i = 0; i < 3; ++i) {
            engine->executeAsync(params, context).get();
        }
        
        // 性能测试
        const int iterations = 10;
        std::vector<double> times;
        
        for (int i = 0; i < iterations; ++i) {
            auto future = engine->executeAsync(params, context);
            auto result = future.get();
            times.push_back(result.gpuTimeMs);
        }
        
        // 计算统计数据
        double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double minTime = *std::min_element(times.begin(), times.end());
        double maxTime = *std::max_element(times.begin(), times.end());
        
        double throughput = (tc.targetSize * tc.targetSize) / (avgTime * 1000.0);
        
        std::cout << tc.description << ":" << std::endl;
        std::cout << "  Average time: " << avgTime << " ms" << std::endl;
        std::cout << "  Min/Max time: " << minTime << "/" << maxTime << " ms" << std::endl;
        std::cout << "  Throughput: " << throughput / 1e6 << " Mpixels/s" << std::endl;
    }
}

// 不同插值方法的性能对比
TEST_F(GPUInterpolationPerformanceTest, DifferentInterpolationMethods) {
    std::cout << "Comparing different interpolation methods..." << std::endl;
    
    const int sourceSize = 1024;
    const int targetSize = 2048;
    auto sourceData = createLargeTestData(sourceSize, sourceSize);
    
    std::vector<InterpolationMethod> methods = {
        InterpolationMethod::NEAREST,
        InterpolationMethod::BILINEAR,
        InterpolationMethod::BICUBIC
    };
    
    auto engine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    ASSERT_NE(engine, nullptr);
    
    for (auto method : methods) {
        GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = targetSize;
        params.outputHeight = targetSize;
        params.outputBounds = sourceData->getDefinition().getBounds();
        params.method = method;
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
        double throughput = (targetSize * targetSize) / (avgTime * 1000.0);
        
        std::cout << "Method: " << static_cast<int>(method) << std::endl;
        std::cout << "  Average time: " << avgTime << " ms" << std::endl;
        std::cout << "  Throughput: " << throughput / 1e6 << " Mpixels/s" << std::endl;
    }
}

// 多GPU调度测试
TEST_F(GPUInterpolationPerformanceTest, MultiGPUScheduling) {
    std::cout << "Testing multi-GPU scheduling..." << std::endl;
    
    auto scheduler = MultiGPUScheduler::getInstance();
    int numGPUs = scheduler->getDeviceCount();
    
    std::cout << "Number of GPUs available: " << numGPUs << std::endl;
    
    if (numGPUs < 2) {
        GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs";
    }
    
    // 创建多个任务
    std::vector<std::shared_ptr<GridData>> datasets;
    for (int i = 0; i < 4; ++i) {
        datasets.push_back(createLargeTestData(1024, 1024));
    }
    
    auto batchEngine = GPUInterpolationEngineFactory::createBatch(ComputeAPI::CUDA);
    ASSERT_NE(batchEngine, nullptr);
    
    // 准备批量参数
    std::vector<GPUInterpolationParams> batch;
    for (const auto& data : datasets) {
        GPUInterpolationParams params;
        params.sourceData = data;
        params.outputWidth = 2048;
        params.outputHeight = 2048;
        params.outputBounds = data->getDefinition().getBounds();
        params.method = InterpolationMethod::BILINEAR;
        params.fillValue = 0.0f;
        batch.push_back(params);
    }
    
    // 执行多GPU批处理
    auto startTime = std::chrono::high_resolution_clock::now();
    
    GPUExecutionContext context;
    context.enableMultiGPU = true;
    
    auto futures = batchEngine->executeBatchAsync(batch, context);
    
    // 等待所有任务完成
    for (auto& future : futures) {
        auto result = future.get();
        EXPECT_EQ(result.status, GPUError::SUCCESS);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count();
    
    std::cout << "Multi-GPU batch processing completed in " << totalTime << " ms" << std::endl;
    std::cout << "Average time per task: " << totalTime / batch.size() << " ms" << std::endl;
}

// 内存效率测试
TEST_F(GPUInterpolationPerformanceTest, MemoryEfficiency) {
    std::cout << "Testing memory efficiency..." << std::endl;
    
    auto engine = GPUInterpolationEngineFactory::create(ComputeAPI::CUDA);
    ASSERT_NE(engine, nullptr);
    
    // 测试不同大小的数据集
    std::vector<int> sizes = {256, 512, 1024, 2048};
    
    for (int size : sizes) {
        auto sourceData = createLargeTestData(size, size);
        
        GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = size * 2;
        params.outputHeight = size * 2;
        params.outputBounds = sourceData->getDefinition().getBounds();
        params.method = InterpolationMethod::BILINEAR;
        params.fillValue = 0.0f;
        
        size_t estimatedMemory = engine->estimateInterpolationMemory(
            size, size, size * 2, size * 2, params.method);
        
        GPUExecutionContext context;
        context.deviceId = 0;
        
        auto future = engine->executeAsync(params, context);
        auto result = future.get();
        
        EXPECT_EQ(result.status, GPUError::SUCCESS);
        
        std::cout << "Size: " << size << "x" << size << " -> " 
                  << size * 2 << "x" << size * 2 << std::endl;
        std::cout << "  Estimated memory: " << estimatedMemory / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Actual GPU time: " << result.gpuTimeMs << " ms" << std::endl;
        std::cout << "  Memory efficiency: " << 
                  (result.interpolatedData.size() * sizeof(float)) / 
                  static_cast<double>(estimatedMemory) * 100 << "%" << std::endl;
    }
} 