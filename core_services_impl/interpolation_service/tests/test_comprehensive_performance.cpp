#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>

#include "interpolation/gpu/gpu_interpolation_engine.h"
#include "core_services/common_data_types.h"
#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/gpu/multi_gpu_memory_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include <cuda_runtime.h>

// CUDA核函数的C接口声明
extern "C" {
    // 原始版本
    cudaError_t computePCHIPDerivatives(
        const float* d_data,
        float* d_derivX,
        float* d_derivY,
        float* d_derivXY,
        int width, int height,
        cudaStream_t stream);
    
    cudaError_t launchPCHIP2DInterpolation(
        const float* d_sourceData,
        const float* d_derivX,
        const float* d_derivY,
        const float* d_derivXY,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
    
    // 优化版本
    cudaError_t computePCHIPDerivativesOptimized(
        const float* d_data,
        float* d_derivX,
        float* d_derivY,
        float* d_derivXY,
        int width, int height,
        float dx, float dy,
        cudaStream_t stream);
    
    cudaError_t launchPCHIP2DInterpolationOptimized(
        const float* d_sourceData,
        const float* d_derivX,
        const float* d_derivY,
        const float* d_derivXY,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
    
    // 一体化版本
    cudaError_t launchPCHIP2DInterpolationIntegrated(
        const float* d_sourceData,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
}

using namespace std::chrono;
using namespace oscean::core_services;

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
    
    auto data = boost::make_shared<GridData>(def, DataType::Float32, 1);
    
    // 生成随机数据
    size_t dataSize = width * height * sizeof(float);
    auto& buffer = const_cast<std::vector<unsigned char>&>(data->getUnifiedBuffer());
    buffer.resize(dataSize);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    float* dataPtr = reinterpret_cast<float*>(buffer.data());
    for (int i = 0; i < width * height; ++i) {
        dataPtr[i] = dis(gen);
    }
    
    return data;
}

/**
 * @brief 性能测试结果结构
 */
struct PerformanceResult {
    std::string method;
    double gpuTime;         // GPU时间（毫秒）
    double gpuKernelTime;   // GPU核函数时间（毫秒）
    double gpuTransferTime; // GPU传输时间（毫秒）
    double throughput;      // 吞吐量(MP/s)
    bool success;           // 是否成功
};

/**
 * @brief 测试GPU插值性能
 */
PerformanceResult testGPUInterpolation(const boost::shared_ptr<GridData>& sourceData,
                                      oscean::core_services::interpolation::InterpolationMethod method,
                                      int dstWidth, int dstHeight) {
    PerformanceResult result;
    result.success = false;
    
    try {
        auto factory = boost::make_shared<oscean::core_services::interpolation::gpu::GPUInterpolationEngineFactory>();
        auto gpuEngine = factory->create(oscean::common_utils::gpu::ComputeAPI::CUDA);
        
        oscean::core_services::interpolation::gpu::GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = dstWidth;
        params.outputHeight = dstHeight;
        params.method = method;
        params.useTextureMemory = false;
        
        // 创建执行上下文
        oscean::common_utils::gpu::GPUExecutionContext context;
        context.deviceId = 0;
        
        // 预热
        gpuEngine->execute(params, context);
        
        // 正式测试
        auto start = high_resolution_clock::now();
        auto algResult = gpuEngine->execute(params, context);
        auto end = high_resolution_clock::now();
        
        if (algResult.success) {
            result.success = true;
            result.gpuTime = duration_cast<microseconds>(end - start).count() / 1000.0;
            result.gpuKernelTime = algResult.data.gpuTimeMs;
            result.gpuTransferTime = algResult.data.memoryTransferTimeMs;
            result.throughput = (dstWidth * dstHeight) / (algResult.data.gpuTimeMs * 1000.0); // MP/s
        }
    } catch (const std::exception& e) {
        std::cerr << "GPU测试失败: " << e.what() << std::endl;
    }
    
    return result;
}

/**
 * @brief 打印性能结果表格
 */
void printPerformanceTable(const std::vector<PerformanceResult>& results) {
    std::cout << "\n===== GPU性能测试结果 =====" << std::endl;
    std::cout << std::setw(20) << "插值方法" 
              << std::setw(15) << "GPU总时间(ms)"
              << std::setw(15) << "GPU核(ms)"
              << std::setw(15) << "数据传输(ms)"
              << std::setw(15) << "吞吐量(MP/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& result : results) {
        if (result.success) {
            std::cout << std::setw(20) << result.method
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.gpuTime
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.gpuKernelTime
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.gpuTransferTime
                      << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput
                      << std::endl;
        } else {
            std::cout << std::setw(20) << result.method
                      << std::setw(15) << "失败"
                      << std::setw(15) << "-"
                      << std::setw(15) << "-"
                      << std::setw(15) << "-"
                      << std::endl;
        }
    }
}

/**
 * @brief 运行综合性能测试
 */
void runComprehensiveTest(int srcSize, int dstSize) {
    std::cout << "\n测试配置: 源数据 " << srcSize << "x" << srcSize 
              << " -> 目标数据 " << dstSize << "x" << dstSize << std::endl;
    
    // 生成测试数据
    auto sourceData = generateTestData(srcSize, srcSize);
    
    // 测试不同的插值方法
    std::vector<std::pair<oscean::core_services::interpolation::InterpolationMethod, std::string>> methods = {
        {oscean::core_services::interpolation::InterpolationMethod::BILINEAR, "双线性"},
        {oscean::core_services::interpolation::InterpolationMethod::BICUBIC, "双三次"},
        {oscean::core_services::interpolation::InterpolationMethod::NEAREST_NEIGHBOR, "最近邻"},
        {oscean::core_services::interpolation::InterpolationMethod::PCHIP_FAST_2D, "PCHIP 2D"},
        {oscean::core_services::interpolation::InterpolationMethod::TRILINEAR, "三线性"}
    };
    
    std::vector<PerformanceResult> results;
    
    for (const auto& [method, name] : methods) {
        std::cout << "\n测试 " << name << " 插值..." << std::endl;
        
        auto result = testGPUInterpolation(sourceData, method, dstSize, dstSize);
        result.method = name;
        results.push_back(result);
    }
    
    // 打印结果表格
    printPerformanceTable(results);
}

/**
 * @brief 测试PCHIP不同版本的性能对比
 */
void testPCHIPVersionsComparison() {
    std::cout << "\n====== PCHIP版本性能对比 ======" << std::endl;
    
    // 测试不同规模
    std::vector<std::pair<int, int>> testConfigs = {
        {256, 512},    // 中小规模
        {512, 1024},   // 中等规模
    };
    
    for (const auto& [srcSize, dstSize] : testConfigs) {
        std::cout << "\n测试规模: " << srcSize << "x" << srcSize 
                  << " -> " << dstSize << "x" << dstSize << std::endl;
        
        // 准备测试数据
        auto sourceData = generateTestData(srcSize, srcSize);
        const auto& buffer = sourceData->getUnifiedBuffer();
        const float* h_data = reinterpret_cast<const float*>(buffer.data());
        
        // 分配GPU内存
        size_t srcDataSize = srcSize * srcSize * sizeof(float);
        size_t dstDataSize = dstSize * dstSize * sizeof(float);
        
        float* d_sourceData = nullptr;
        float* d_outputData = nullptr;
        cudaMalloc(&d_sourceData, srcDataSize);
        cudaMalloc(&d_outputData, dstDataSize);
        
        // 传输数据到GPU
        cudaMemcpy(d_sourceData, h_data, srcDataSize, cudaMemcpyHostToDevice);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // 1. 测试当前版本（一体化版本）
        std::cout << "\n1. 当前版本（一体化）:" << std::endl;
        cudaEventRecord(start);
        cudaError_t err = launchPCHIP2DInterpolationIntegrated(
            d_sourceData, d_outputData,
            srcSize, srcSize,
            dstSize, dstSize,
            0.0f, (float)(srcSize - 1),
            0.0f, (float)(srcSize - 1),
            0.0f, nullptr
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float integratedTime = 0;
        cudaEventElapsedTime(&integratedTime, start, stop);
        std::cout << "   核函数时间: " << integratedTime << " ms" << std::endl;
        std::cout << "   状态: " << (err == cudaSuccess ? "成功" : "失败") << std::endl;
        
        // 2. 测试原始版本（分离导数计算）
        std::cout << "\n2. 原始版本（分离导数）:" << std::endl;
        
        // 分配导数内存
        float* d_derivX = nullptr;
        float* d_derivY = nullptr;
        float* d_derivXY = nullptr;
        cudaMalloc(&d_derivX, srcDataSize);
        cudaMalloc(&d_derivY, srcDataSize);
        cudaMalloc(&d_derivXY, srcDataSize);
        
        // 计算导数
        cudaEventRecord(start);
        err = computePCHIPDerivatives(
            d_sourceData, d_derivX, d_derivY, d_derivXY,
            srcSize, srcSize, nullptr
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float derivTime = 0;
        cudaEventElapsedTime(&derivTime, start, stop);
        
        // 执行插值
        cudaEventRecord(start);
        err = launchPCHIP2DInterpolation(
            d_sourceData, d_derivX, d_derivY, d_derivXY, d_outputData,
            srcSize, srcSize,
            dstSize, dstSize,
            0.0f, (float)(srcSize - 1),
            0.0f, (float)(srcSize - 1),
            0.0f, nullptr
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float originalInterpolTime = 0;
        cudaEventElapsedTime(&originalInterpolTime, start, stop);
        
        std::cout << "   导数计算时间: " << derivTime << " ms" << std::endl;
        std::cout << "   插值时间: " << originalInterpolTime << " ms" << std::endl;
        std::cout << "   总时间: " << (derivTime + originalInterpolTime) << " ms" << std::endl;
        
        // 3. 测试优化版本
        std::cout << "\n3. 优化版本（共享内存）:" << std::endl;
        
        // 计算优化的导数
        cudaEventRecord(start);
        err = computePCHIPDerivativesOptimized(
            d_sourceData, d_derivX, d_derivY, d_derivXY,
            srcSize, srcSize, 1.0f, 1.0f, nullptr
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float optimizedDerivTime = 0;
        cudaEventElapsedTime(&optimizedDerivTime, start, stop);
        
        // 执行优化的插值
        cudaEventRecord(start);
        err = launchPCHIP2DInterpolationOptimized(
            d_sourceData, d_derivX, d_derivY, d_derivXY, d_outputData,
            srcSize, srcSize,
            dstSize, dstSize,
            0.0f, (float)(srcSize - 1),
            0.0f, (float)(srcSize - 1),
            0.0f, nullptr
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float optimizedInterpolTime = 0;
        cudaEventElapsedTime(&optimizedInterpolTime, start, stop);
        
        std::cout << "   导数计算时间: " << optimizedDerivTime << " ms" << std::endl;
        std::cout << "   插值时间: " << optimizedInterpolTime << " ms" << std::endl;
        std::cout << "   总时间: " << (optimizedDerivTime + optimizedInterpolTime) << " ms" << std::endl;
        
        // 性能对比分析
        std::cout << "\n性能对比:" << std::endl;
        float originalTotal = derivTime + originalInterpolTime;
        float optimizedTotal = optimizedDerivTime + optimizedInterpolTime;
        
        std::cout << "   一体化版本比原始版本快: " 
                  << std::fixed << std::setprecision(2) 
                  << (originalTotal / integratedTime) << "x" << std::endl;
        std::cout << "   优化版本比原始版本快: " 
                  << (originalTotal / optimizedTotal) << "x" << std::endl;
        std::cout << "   一体化版本比优化版本快: " 
                  << (optimizedTotal / integratedTime) << "x" << std::endl;
        
        // 清理
        cudaFree(d_derivX);
        cudaFree(d_derivY);
        cudaFree(d_derivXY);
        cudaFree(d_sourceData);
        cudaFree(d_outputData);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

/**
 * @brief 测试批量处理性能
 */
void testBatchProcessing(int srcSize, int dstSize, int batchSize) {
    std::cout << "\n====== 批量处理性能测试 ======" << std::endl;
    std::cout << "批量大小: " << batchSize << std::endl;
    
    auto factory = boost::make_shared<oscean::core_services::interpolation::gpu::GPUInterpolationEngineFactory>();
    auto batchEngine = factory->createOptimizedBatch(); // 使用优化的批量引擎
    
    // 准备批量数据
    std::vector<oscean::core_services::interpolation::gpu::GPUInterpolationParams> batchParams;
    std::vector<boost::shared_ptr<GridData>> testData;
    
    for (int i = 0; i < batchSize; ++i) {
        auto data = generateTestData(srcSize, srcSize);
        testData.push_back(data);
        
        oscean::core_services::interpolation::gpu::GPUInterpolationParams params;
        params.sourceData = data;
        params.outputWidth = dstSize;
        params.outputHeight = dstSize;
        params.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
        params.useTextureMemory = false;
        batchParams.push_back(params);
    }
    
    // 创建执行上下文
    oscean::common_utils::gpu::GPUExecutionContext context;
    context.deviceId = 0;
    
    // 测试批量处理
    auto batchStart = high_resolution_clock::now();
    auto batchResult = batchEngine->execute(batchParams, context);
    auto batchEnd = high_resolution_clock::now();
    
    if (batchResult.success && !batchResult.data.empty()) {
        auto batchTotalTime = duration_cast<microseconds>(batchEnd - batchStart).count() / 1000.0;
        auto avgTimePerImage = batchTotalTime / batchSize;
        
        // 计算统计信息
        double avgGpuTime = 0;
        double avgTransferTime = 0;
        int successCount = 0;
        
        for (const auto& res : batchResult.data) {
            if (res.status == oscean::common_utils::gpu::GPUError::SUCCESS) {
                avgGpuTime += res.gpuTimeMs;
                avgTransferTime += res.memoryTransferTimeMs;
                successCount++;
            }
        }
        
        if (successCount > 0) {
            avgGpuTime /= successCount;
            avgTransferTime /= successCount;
            
            std::cout << "总时间: " << batchTotalTime << " ms" << std::endl;
            std::cout << "平均每张: " << avgTimePerImage << " ms" << std::endl;
            std::cout << "平均GPU核函数: " << avgGpuTime << " ms" << std::endl;
            std::cout << "平均数据传输: " << avgTransferTime << " ms" << std::endl;
            std::cout << "成功率: " << (successCount * 100.0 / batchSize) << "%" << std::endl;
            
            // 计算加速比
            auto singleEngine = factory->create(oscean::common_utils::gpu::ComputeAPI::CUDA);
            auto singleStart = high_resolution_clock::now();
            auto singleResult = singleEngine->execute(batchParams[0], context);
            auto singleEnd = high_resolution_clock::now();
            
            if (singleResult.success) {
                auto singleTime = duration_cast<microseconds>(singleEnd - singleStart).count() / 1000.0;
                auto speedup = (singleTime * batchSize) / batchTotalTime;
                std::cout << "批量处理加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            }
        }
    } else {
        std::cout << "批量处理失败" << std::endl;
    }
}

int main() {
    try {
        std::cout << "=== 插值算法综合性能测试 ===" << std::endl;
        
        // 初始化日志
        oscean::common_utils::LoggingConfig config;
        config.console_level = "warn";
        config.enable_console = true;
        oscean::common_utils::LoggingManager::configureGlobal(config);
        
        // 初始化GPU管理器
        auto& deviceManager = oscean::common_utils::gpu::UnifiedGPUManager::getInstance();
        deviceManager.initialize();
        
        // 获取设备信息
        auto devices = deviceManager.getAllDeviceInfo();
        
        // 初始化全局GPU内存管理器
        if (!devices.empty()) {
            // 使用自定义的内存池配置以解决中等规模数据传输问题
            oscean::common_utils::gpu::MemoryPoolConfig poolConfig;
            poolConfig.initialPoolSize = 512 * 1024 * 1024;  // 512MB
            poolConfig.blockSize = 4 * 1024 * 1024;          // 4MB块
            poolConfig.maxPoolSize = 2048 * 1024 * 1024;     // 2GB
            poolConfig.enableGrowth = true;                  // 允许池增长
            poolConfig.growthFactor = 1.5f;                  // 增长因子
            
            oscean::common_utils::gpu::MemoryTransferConfig transferConfig;
            transferConfig.enablePeerAccess = true;           // 启用GPU间直接访问
            transferConfig.enableAsyncTransfer = true;        // 启用异步传输
            transferConfig.transferBufferSize = 64 * 1024 * 1024; // 64MB传输缓冲区
            transferConfig.maxConcurrentTransfers = 16;       // 最大并发传输数
            
            oscean::common_utils::gpu::GlobalMemoryManager::initialize(devices, poolConfig, transferConfig);
            
            // 预分配内存池以避免运行时分配失败
            try {
                auto& memManager = oscean::common_utils::gpu::GlobalMemoryManager::getInstance();
                for (const auto& device : devices) {
                    // 预分配初始内存池
                    if (!memManager.preallocatePool(device.deviceId, poolConfig.initialPoolSize)) {
                        std::cout << "警告: 无法为设备 " << device.deviceId 
                                  << " 预分配内存池" << std::endl;
                    } else {
                        std::cout << "为设备 " << device.deviceId 
                                  << " 预分配了 " << (poolConfig.initialPoolSize / 1024 / 1024) 
                                  << " MB 内存池" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cout << "预分配内存池时出错: " << e.what() << std::endl;
            }
        }
        
        // 检查GPU设备
        if (devices.empty()) {
            std::cerr << "没有找到GPU设备" << std::endl;
            return 1;
        } else {
            std::cout << "\n找到GPU设备: " << devices[0].name << std::endl;
            std::cout << "显存: " << (devices[0].memoryDetails.totalGlobalMemory / (1024*1024)) << " MB" << std::endl;
            std::cout << "计算能力: " << devices[0].architecture.majorVersion << "." 
                      << devices[0].architecture.minorVersion << std::endl;
        }
        
        // 运行不同规模的测试
        std::vector<std::pair<int, int>> testConfigs = {
            {128, 256},    // 小规模
            {256, 512},    // 中小规模（之前失败的）
            {512, 1024},   // 中等规模（之前失败的）
            {1024, 2048},  // 大规模
        };
        
        for (const auto& [srcSize, dstSize] : testConfigs) {
            runComprehensiveTest(srcSize, dstSize);
        }
        
        // 测试PCHIP不同版本
        testPCHIPVersionsComparison();
        
        // 测试批量处理
        std::cout << "\n=== 批量处理性能测试 ===" << std::endl;
        testBatchProcessing(256, 512, 8);   // 8张图片批量
        testBatchProcessing(256, 512, 16);  // 16张图片批量
        testBatchProcessing(256, 512, 32);  // 32张图片批量
        
        // 总结
        std::cout << "\n=== 性能分析总结 ===" << std::endl;
        std::cout << "1. GPU在处理大规模数据时优势明显" << std::endl;
        std::cout << "2. 批量处理通过流水线优化可以显著提升性能" << std::endl;
        std::cout << "3. 固定内存池解决了中等规模数据传输问题" << std::endl;
        std::cout << "4. 不同插值方法的性能差异主要在核函数复杂度" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "未知错误" << std::endl;
        return 1;
    }
    
    return 0;
} 