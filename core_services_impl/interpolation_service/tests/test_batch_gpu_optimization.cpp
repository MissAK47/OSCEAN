/**
 * @file test_batch_gpu_optimization.cpp
 * @brief 测试优化的批量GPU插值性能
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>

#include "interpolation/gpu/gpu_interpolation_engine.h"
#include "core_services/common_data_types.h"
#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/gpu/gpu_algorithm_interface.h"
#include "common_utils/utilities/logging_utils.h"

using namespace std::chrono;
using namespace oscean::core_services::interpolation;
using namespace oscean::core_services::interpolation::gpu;
using namespace oscean::core_services;
using namespace oscean::common_utils::gpu;

/**
 * @brief 生成测试数据
 */
boost::shared_ptr<GridData> generateTestData(int width, int height) {
    GridDefinition def;
    def.cols = width;
    def.rows = height;
    def.extent.minX = 0.0;
    def.extent.maxX = width - 1.0;
    def.extent.minY = 0.0;
    def.extent.maxY = height - 1.0;
    def.xResolution = 1.0;
    def.yResolution = 1.0;
    def.crs.isGeographic = false;
    def.crs.isProjected = true;
    def.crs.wkt = "PROJCS[\"Test\"]";
    
    auto gridData = boost::make_shared<GridData>(def, DataType::Float32, 1);
    
    // 填充随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    float* buffer = reinterpret_cast<float*>(gridData->getUnifiedBufferData());
    for (int i = 0; i < width * height; ++i) {
        buffer[i] = dis(gen);
    }
    
    return gridData;
}

/**
 * @brief 测试批量处理性能
 */
void testBatchProcessing(int batchSize, int srcSize, int dstSize) {
    std::cout << "\n=== 测试批量处理性能 ===" << std::endl;
    std::cout << "批大小: " << batchSize << std::endl;
    std::cout << "输入尺寸: " << srcSize << "x" << srcSize << std::endl;
    std::cout << "输出尺寸: " << dstSize << "x" << dstSize << std::endl;
    std::cout << "总数据量: " << (batchSize * srcSize * srcSize * sizeof(float) / (1024.0 * 1024.0)) << " MB (输入)" << std::endl;
    std::cout << "         " << (batchSize * dstSize * dstSize * sizeof(float) / (1024.0 * 1024.0)) << " MB (输出)" << std::endl;
    
    // 准备批数据
    std::vector<GPUInterpolationParams> batchParams;
    for (int i = 0; i < batchSize; ++i) {
        GPUInterpolationParams params;
        params.sourceData = generateTestData(srcSize, srcSize);
        params.outputWidth = dstSize;
        params.outputHeight = dstSize;
        params.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
        params.useTextureMemory = false;
        batchParams.push_back(params);
    }
    
    // 获取GPU设备信息
    auto& deviceManager = oscean::common_utils::gpu::UnifiedGPUManager::getInstance();
    auto devices = deviceManager.getAllDeviceInfo();
    if (devices.empty()) {
        std::cerr << "没有找到GPU设备" << std::endl;
        return;
    }
    
    GPUExecutionContext context;
    context.deviceId = 0;
    
    // 测试简化版批处理
    {
        std::cout << "\n--- 测试简化版批处理 ---" << std::endl;
        
        auto factory = boost::make_shared<oscean::core_services::interpolation::gpu::GPUInterpolationEngineFactory>();
        auto batchEngine = factory->createBatch(ComputeAPI::CUDA);
        
        // 预热
        batchEngine->execute(batchParams, context);
        
        auto start = high_resolution_clock::now();
        
        auto result = batchEngine->execute(batchParams, context);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        
        if (result.success) {
            std::cout << "简化版批处理成功" << std::endl;
            std::cout << "总时间: " << duration << " ms" << std::endl;
            std::cout << "平均每张: " << (float)duration / batchSize << " ms" << std::endl;
            
            std::cout << "GPU核函数时间: " << result.stats.kernelTime << " ms" << std::endl;
            std::cout << "数据传输时间: " << result.stats.transferTime << " ms" << std::endl;
            
            double processedMB = (batchSize * srcSize * srcSize + batchSize * dstSize * dstSize) * sizeof(float) / (1024.0 * 1024.0);
            double throughputMBps = processedMB / (duration / 1000.0);
            std::cout << "吞吐量: " << throughputMBps << " MB/s" << std::endl;
        } else {
            std::cerr << "批处理失败: " << result.errorMessage << std::endl;
        }
    }
    
    // 测试优化版批处理（如果可用）
    {
        std::cout << "\n--- 测试优化版批处理 ---" << std::endl;
        
        // 创建优化的批处理引擎（需要确保已经注册）
        auto batchEngine = GPUInterpolationEngineFactory::createOptimizedBatch();
        if (!batchEngine) {
            std::cout << "优化版批处理引擎不可用" << std::endl;
            return;
        }
        
        // 预热
        batchEngine->execute(batchParams, context);
        
        auto start = high_resolution_clock::now();
        
        auto result = batchEngine->execute(batchParams, context);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        
        if (result.success) {
            std::cout << "优化版批处理成功" << std::endl;
            std::cout << "总时间: " << duration << " ms" << std::endl;
            std::cout << "平均每张: " << (float)duration / batchSize << " ms" << std::endl;
            
            std::cout << "GPU核函数时间: " << result.stats.kernelTime << " ms" << std::endl;
            std::cout << "数据传输时间: " << result.stats.transferTime << " ms" << std::endl;
            
            double processedMB = (batchSize * srcSize * srcSize + batchSize * dstSize * dstSize) * sizeof(float) / (1024.0 * 1024.0);
            double throughputMBps = processedMB / (duration / 1000.0);
            std::cout << "吞吐量: " << throughputMBps << " MB/s" << std::endl;
        } else {
            std::cerr << "批处理失败: " << result.errorMessage << std::endl;
        }
    }
}

/**
 * @brief 测试异步处理
 */
void testAsyncProcessing() {
    std::cout << "\n=== 测试异步处理 ===" << std::endl;
    
    auto factory = boost::make_shared<GPUInterpolationEngineFactory>();
    auto batchEngine = factory->createBatch(ComputeAPI::CUDA);
    
    GPUExecutionContext context;
    context.deviceId = 0;
    
    // 准备多个批次
    std::vector<boost::future<GPUAlgorithmResult<std::vector<GPUInterpolationResult>>>> futures;
    
    for (int batch = 0; batch < 4; ++batch) {
        std::vector<GPUInterpolationParams> batchParams;
        
        for (int i = 0; i < 8; ++i) {
            GPUInterpolationParams params;
            params.sourceData = generateTestData(512, 512);
            params.outputWidth = 1024;
            params.outputHeight = 1024;
            params.method = (batch % 2 == 0) ? oscean::core_services::interpolation::InterpolationMethod::BILINEAR : oscean::core_services::interpolation::InterpolationMethod::BICUBIC;
            params.useTextureMemory = false;
            batchParams.push_back(params);
        }
        
        // 异步提交
        auto future = batchEngine->executeAsync(batchParams, context);
        futures.push_back(std::move(future));
        
        std::cout << "提交批次 " << batch << " (8张图片)" << std::endl;
    }
    
    // 等待所有批次完成
    std::cout << "\n等待所有批次完成..." << std::endl;
    
    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        if (result.success) {
            std::cout << "批次 " << i << " 完成" << std::endl;
        } else {
            std::cerr << "批次 " << i << " 失败: " << result.errorMessage << std::endl;
        }
    }
}

/**
 * @brief 测试不同批大小的性能
 */
void testBatchSizeScaling() {
    std::cout << "\n=== 测试批大小缩放性能 ===" << std::endl;
    
    // 小规模测试
    std::cout << "\n### 小规模数据测试 (512x512 -> 1024x1024) ###" << std::endl;
    std::vector<int> smallBatchSizes = {1, 4, 8, 16, 32, 64};
    for (int batchSize : smallBatchSizes) {
        testBatchProcessing(batchSize, 512, 1024);
    }
    
    // 中等规模测试
    std::cout << "\n### 中等规模数据测试 (1024x1024 -> 2048x2048) ###" << std::endl;
    std::vector<int> mediumBatchSizes = {1, 8, 16, 32, 64};
    for (int batchSize : mediumBatchSizes) {
        testBatchProcessing(batchSize, 1024, 2048);
    }
    
    // 大规模测试
    std::cout << "\n### 大规模数据测试 (2048x2048 -> 4096x4096) ###" << std::endl;
    std::vector<int> largeBatchSizes = {1, 4, 8, 16, 32};
    for (int batchSize : largeBatchSizes) {
        testBatchProcessing(batchSize, 2048, 4096);
    }
}

/**
 * @brief 测试长时间运行性能
 */
void testLongRunningPerformance() {
    std::cout << "\n=== 测试长时间运行性能 ===" << std::endl;
    
    auto factory = boost::make_shared<GPUInterpolationEngineFactory>();
    auto simpleBatch = factory->createBatch(ComputeAPI::CUDA);
    auto optimizedBatch = GPUInterpolationEngineFactory::createOptimizedBatch();
    
    if (!optimizedBatch) {
        std::cout << "优化版批处理引擎不可用" << std::endl;
        return;
    }
    
    GPUExecutionContext context;
    context.deviceId = 0;
    
    const int numIterations = 100;
    const int batchSize = 32;
    const int srcSize = 1024;
    const int dstSize = 2048;
    
    std::cout << "测试配置: " << numIterations << " 次迭代, 批大小=" << batchSize 
              << ", " << srcSize << "x" << srcSize << " -> " << dstSize << "x" << dstSize << std::endl;
    
    // 准备批数据
    std::vector<GPUInterpolationParams> batchParams;
    for (int i = 0; i < batchSize; ++i) {
        GPUInterpolationParams params;
        params.sourceData = generateTestData(srcSize, srcSize);
        params.outputWidth = dstSize;
        params.outputHeight = dstSize;
        params.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
        params.useTextureMemory = false;
        batchParams.push_back(params);
    }
    
    // 测试简化版
    {
        std::cout << "\n--- 简化版长时间运行测试 ---" << std::endl;
        
        // 预热
        for (int i = 0; i < 5; ++i) {
            simpleBatch->execute(batchParams, context);
        }
        
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < numIterations; ++i) {
            auto result = simpleBatch->execute(batchParams, context);
            if (!result.success) {
                std::cerr << "迭代 " << i << " 失败" << std::endl;
                break;
            }
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        
        std::cout << "总时间: " << duration << " ms" << std::endl;
        std::cout << "平均每批: " << (float)duration / numIterations << " ms" << std::endl;
        std::cout << "平均每张: " << (float)duration / (numIterations * batchSize) << " ms" << std::endl;
        
        double totalDataMB = numIterations * batchSize * (srcSize * srcSize + dstSize * dstSize) * sizeof(float) / (1024.0 * 1024.0);
        std::cout << "总处理数据: " << totalDataMB << " MB" << std::endl;
        std::cout << "平均吞吐量: " << totalDataMB / (duration / 1000.0) << " MB/s" << std::endl;
    }
    
    // 测试优化版
    {
        std::cout << "\n--- 优化版长时间运行测试 ---" << std::endl;
        
        // 预热
        for (int i = 0; i < 5; ++i) {
            optimizedBatch->execute(batchParams, context);
        }
        
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < numIterations; ++i) {
            auto result = optimizedBatch->execute(batchParams, context);
            if (!result.success) {
                std::cerr << "迭代 " << i << " 失败" << std::endl;
                break;
            }
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        
        std::cout << "总时间: " << duration << " ms" << std::endl;
        std::cout << "平均每批: " << (float)duration / numIterations << " ms" << std::endl;
        std::cout << "平均每张: " << (float)duration / (numIterations * batchSize) << " ms" << std::endl;
        
        double totalDataMB = numIterations * batchSize * (srcSize * srcSize + dstSize * dstSize) * sizeof(float) / (1024.0 * 1024.0);
        std::cout << "总处理数据: " << totalDataMB << " MB" << std::endl;
        std::cout << "平均吞吐量: " << totalDataMB / (duration / 1000.0) << " MB/s" << std::endl;
    }
}

/**
 * @brief 主函数
 */
int main() {
    try {
        std::cout << "=== 批量GPU优化测试程序 (大规模数据版本) ===" << std::endl;
        
        // 初始化日志
        oscean::common_utils::LoggingConfig config;
        config.console_level = "info";
        config.enable_console = true;
        oscean::common_utils::LoggingManager::configureGlobal(config);
        
        std::cout << "初始化GPU管理器..." << std::endl;
        
        // 检查GPU设备
        auto& deviceManager = oscean::common_utils::gpu::UnifiedGPUManager::getInstance();
        deviceManager.initialize();
        
        std::cout << "检测GPU设备..." << std::endl;
        auto devices = deviceManager.getAllDeviceInfo();
        
        if (devices.empty()) {
            std::cerr << "没有找到GPU设备，测试退出" << std::endl;
            return 1;
        }
        
        std::cout << "找到 " << devices.size() << " 个GPU设备" << std::endl;
        for (const auto& device : devices) {
            std::cout << "  - " << device.name 
                     << " (计算能力: " << device.architecture.majorVersion << "." 
                     << device.architecture.minorVersion 
                     << ", 显存: " << device.memoryDetails.totalGlobalMemory / (1024*1024) << " MB"
                     << ", 空闲: " << device.memoryDetails.freeGlobalMemory / (1024*1024) << " MB)" 
                     << std::endl;
        }
        
        // 运行测试
        // testBatchProcessing(16, 512, 1024);  // 原始测试
        // testAsyncProcessing();  // 跳过异步测试，专注于性能对比
        testBatchSizeScaling();    // 不同规模的测试
        testLongRunningPerformance();  // 长时间运行测试
        
        std::cout << "\n所有测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "未知错误" << std::endl;
        return 1;
    }
    
    return 0;
} 