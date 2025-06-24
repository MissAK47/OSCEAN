#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
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

/**
 * @brief 生成测试数据（模拟海洋深度）
 */
void generateTestData(float* data, int width, int height) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-5000.0f, 0.0f);
    
    for (int i = 0; i < width * height; ++i) {
        data[i] = dis(gen);
    }
}

/**
 * @brief 性能测试结果
 */
struct PerformanceResult {
    std::string version;
    double totalTime;      // 总时间（包括导数计算）
    double kernelTime;     // 插值核函数时间
    double derivTime;      // 导数计算时间（如果有）
    bool success;
};

/**
 * @brief 测试原始版本PCHIP
 */
PerformanceResult testOriginalPCHIP(const float* d_sourceData,
                                   float* d_outputData,
                                   int srcWidth, int srcHeight,
                                   int dstWidth, int dstHeight) {
    PerformanceResult result;
    result.version = "原始版本";
    result.success = false;
    result.derivTime = 0.0;
    
    // 分配导数内存
    size_t derivSize = srcWidth * srcHeight * sizeof(float);
    float* d_derivX = nullptr;
    float* d_derivY = nullptr;
    float* d_derivXY = nullptr;
    
    cudaMalloc(&d_derivX, derivSize);
    cudaMalloc(&d_derivY, derivSize);
    cudaMalloc(&d_derivXY, derivSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 计算导数
    cudaEventRecord(start);
    cudaError_t err = computePCHIPDerivatives(
        d_sourceData, d_derivX, d_derivY, d_derivXY,
        srcWidth, srcHeight, nullptr
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float derivTimeMs = 0;
    cudaEventElapsedTime(&derivTimeMs, start, stop);
    result.derivTime = derivTimeMs;
    
    if (err == cudaSuccess) {
        // 执行插值
        cudaEventRecord(start);
        err = launchPCHIP2DInterpolation(
            d_sourceData, d_derivX, d_derivY, d_derivXY, d_outputData,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            0.0f, (float)(srcWidth - 1),
            0.0f, (float)(srcHeight - 1),
            0.0f, nullptr
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float kernelTimeMs = 0;
        cudaEventElapsedTime(&kernelTimeMs, start, stop);
        result.kernelTime = kernelTimeMs;
        result.totalTime = derivTimeMs + kernelTimeMs;
        result.success = (err == cudaSuccess);
    }
    
    // 清理
    cudaFree(d_derivX);
    cudaFree(d_derivY);
    cudaFree(d_derivXY);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

/**
 * @brief 测试优化版本PCHIP
 */
PerformanceResult testOptimizedPCHIP(const float* d_sourceData,
                                    float* d_outputData,
                                    int srcWidth, int srcHeight,
                                    int dstWidth, int dstHeight) {
    PerformanceResult result;
    result.version = "优化版本（预计算导数）";
    result.success = false;
    result.derivTime = 0.0;
    
    // 分配导数内存
    size_t derivSize = srcWidth * srcHeight * sizeof(float);
    float* d_derivX = nullptr;
    float* d_derivY = nullptr;
    float* d_derivXY = nullptr;
    
    cudaMalloc(&d_derivX, derivSize);
    cudaMalloc(&d_derivY, derivSize);
    cudaMalloc(&d_derivXY, derivSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 计算导数（优化版本）
    float dx = 1.0f;  // 假设单位网格间距
    float dy = 1.0f;
    
    cudaEventRecord(start);
    cudaError_t err = computePCHIPDerivativesOptimized(
        d_sourceData, d_derivX, d_derivY, d_derivXY,
        srcWidth, srcHeight, dx, dy, nullptr
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float derivTimeMs = 0;
    cudaEventElapsedTime(&derivTimeMs, start, stop);
    result.derivTime = derivTimeMs;
    
    if (err == cudaSuccess) {
        // 执行优化的插值
        cudaEventRecord(start);
        err = launchPCHIP2DInterpolationOptimized(
            d_sourceData, d_derivX, d_derivY, d_derivXY, d_outputData,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            0.0f, (float)(srcWidth - 1),
            0.0f, (float)(srcHeight - 1),
            0.0f, nullptr
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float kernelTimeMs = 0;
        cudaEventElapsedTime(&kernelTimeMs, start, stop);
        result.kernelTime = kernelTimeMs;
        result.totalTime = derivTimeMs + kernelTimeMs;
        result.success = (err == cudaSuccess);
    }
    
    // 清理
    cudaFree(d_derivX);
    cudaFree(d_derivY);
    cudaFree(d_derivXY);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

/**
 * @brief 测试一体化版本PCHIP
 */
PerformanceResult testIntegratedPCHIP(const float* d_sourceData,
                                     float* d_outputData,
                                     int srcWidth, int srcHeight,
                                     int dstWidth, int dstHeight) {
    PerformanceResult result;
    result.version = "一体化版本（即时导数）";
    result.success = false;
    result.derivTime = 0.0;  // 包含在核函数中
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 执行一体化插值
    cudaEventRecord(start);
    cudaError_t err = launchPCHIP2DInterpolationIntegrated(
        d_sourceData, d_outputData,
        srcWidth, srcHeight,
        dstWidth, dstHeight,
        0.0f, (float)(srcWidth - 1),
        0.0f, (float)(srcHeight - 1),
        0.0f, nullptr
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float totalTimeMs = 0;
    cudaEventElapsedTime(&totalTimeMs, start, stop);
    result.kernelTime = totalTimeMs;
    result.totalTime = totalTimeMs;
    result.success = (err == cudaSuccess);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

/**
 * @brief 打印性能对比结果
 */
void printResults(const std::vector<PerformanceResult>& results, 
                 int srcSize, int dstSize) {
    std::cout << "\n=== PCHIP版本性能对比 (" << srcSize << "x" << srcSize 
              << " -> " << dstSize << "x" << dstSize << ") ===" << std::endl;
    std::cout << std::setw(25) << "版本" 
              << std::setw(15) << "总时间(ms)"
              << std::setw(15) << "核函数(ms)"
              << std::setw(15) << "导数计算(ms)"
              << std::setw(15) << "状态"
              << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(25) << result.version
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.totalTime
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.kernelTime
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.derivTime
                  << std::setw(15) << (result.success ? "成功" : "失败")
                  << std::endl;
    }
    
    // 找出最快的版本
    auto minIt = std::min_element(results.begin(), results.end(),
        [](const PerformanceResult& a, const PerformanceResult& b) {
            return a.success && (!b.success || a.totalTime < b.totalTime);
        });
    
    if (minIt != results.end() && minIt->success) {
        std::cout << "\n最快版本: " << minIt->version 
                  << " (" << minIt->totalTime << " ms)" << std::endl;
        
        // 计算相对性能
        for (const auto& result : results) {
            if (result.success && &result != &(*minIt)) {
                double speedup = result.totalTime / minIt->totalTime;
                std::cout << minIt->version << " 比 " << result.version 
                          << " 快 " << std::fixed << std::setprecision(2) 
                          << speedup << "x" << std::endl;
            }
        }
    }
}

int main() {
    try {
        std::cout << "=== PCHIP GPU版本性能对比测试 ===" << std::endl;
        
        // 检查CUDA设备
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            std::cerr << "没有找到CUDA设备" << std::endl;
            return 1;
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU设备: " << prop.name << std::endl;
        std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
        
        // 测试配置
        std::vector<std::pair<int, int>> testConfigs = {
            {128, 256},    // 小规模
            {256, 512},    // 中等规模
            {512, 1024},   // 大规模
            {1024, 2048}   // 超大规模
        };
        
        for (const auto& [srcSize, dstSize] : testConfigs) {
            // 准备测试数据
            size_t srcDataSize = srcSize * srcSize * sizeof(float);
            size_t dstDataSize = dstSize * dstSize * sizeof(float);
            
            // 主机数据
            float* h_sourceData = new float[srcSize * srcSize];
            generateTestData(h_sourceData, srcSize, srcSize);
            
            // GPU数据
            float* d_sourceData = nullptr;
            float* d_outputData1 = nullptr;
            float* d_outputData2 = nullptr;
            float* d_outputData3 = nullptr;
            
            cudaMalloc(&d_sourceData, srcDataSize);
            cudaMalloc(&d_outputData1, dstDataSize);
            cudaMalloc(&d_outputData2, dstDataSize);
            cudaMalloc(&d_outputData3, dstDataSize);
            
            // 传输数据到GPU
            cudaMemcpy(d_sourceData, h_sourceData, srcDataSize, cudaMemcpyHostToDevice);
            
            // 预热
            testIntegratedPCHIP(d_sourceData, d_outputData1, srcSize, srcSize, dstSize, dstSize);
            
            // 测试不同版本
            std::vector<PerformanceResult> results;
            
            // 1. 原始版本
            results.push_back(testOriginalPCHIP(
                d_sourceData, d_outputData1, srcSize, srcSize, dstSize, dstSize));
            
            // 2. 优化版本
            results.push_back(testOptimizedPCHIP(
                d_sourceData, d_outputData2, srcSize, srcSize, dstSize, dstSize));
            
            // 3. 一体化版本
            results.push_back(testIntegratedPCHIP(
                d_sourceData, d_outputData3, srcSize, srcSize, dstSize, dstSize));
            
            // 打印结果
            printResults(results, srcSize, dstSize);
            
            // 清理
            delete[] h_sourceData;
            cudaFree(d_sourceData);
            cudaFree(d_outputData1);
            cudaFree(d_outputData2);
            cudaFree(d_outputData3);
        }
        
        std::cout << "\n=== 测试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 