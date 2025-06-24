/**
 * @file gpu_algorithm_examples.cpp
 * @brief GPU算法示例实现
 * 
 * 展示如何使用GPUAlgorithmBase创建具体的GPU算法
 */

#include "common_utils/gpu/gpu_algorithm_base.h"
#include "common_utils/gpu/multi_gpu_memory_manager.h"
#include "core_services/common_data_types.h"  // 使用项目通用数据结构
#include "common_utils/utilities/logging_utils.h"
#include <algorithm>
#include <cmath>
#include <chrono>

#ifdef OSCEAN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#ifdef OSCEAN_OPENCL_ENABLED
#include <CL/cl.h>
#endif

namespace oscean::common_utils::gpu {

using oscean::core_services::GridData;
using oscean::core_services::DataType;

/**
 * @brief 向量加法GPU算法示例
 * 
 * 实现简单的向量加法：C = A + B
 */
class VectorAddGPUAlgorithm : public GPUAlgorithmBase<
    std::pair<std::vector<float>, std::vector<float>>,  // 输入：两个向量
    std::vector<float>                                   // 输出：结果向量
> {
public:
    VectorAddGPUAlgorithm() 
        : GPUAlgorithmBase(
            "VectorAdd",
            "1.0.0",
            {ComputeAPI::CUDA, ComputeAPI::OPENCL}
        ) {
    }
    
    /**
     * @brief 估算内存需求
     */
    size_t estimateMemoryRequirement(
        const std::pair<std::vector<float>, std::vector<float>>& input) const override {
        
        size_t vectorSize = std::max(input.first.size(), input.second.size());
        return vectorSize * sizeof(float) * 3;  // 输入A + 输入B + 输出C
    }
    
    /**
     * @brief 验证输入数据
     */
    std::pair<bool, std::string> validateInput(
        const std::pair<std::vector<float>, std::vector<float>>& input) const override {
        
        if (input.first.empty() || input.second.empty()) {
            return {false, "Input vectors cannot be empty"};
        }
        
        if (input.first.size() != input.second.size()) {
            return {false, "Input vectors must have the same size"};
        }
        
        return {true, ""};
    }
    
protected:
    /**
     * @brief 内部执行实现
     */
    GPUAlgorithmResult<std::vector<float>> executeInternal(
        const std::pair<std::vector<float>, std::vector<float>>& input,
        const GPUExecutionContext& context) override {
        
        // 验证输入
        auto [valid, errorMsg] = validateInput(input);
        if (!valid) {
            return createErrorResult(GPUError::INVALID_INPUT, errorMsg);
        }
        
        const auto& vecA = input.first;
        const auto& vecB = input.second;
        size_t size = vecA.size();
        
        // 性能统计
        typename GPUAlgorithmResult<std::vector<float>>::PerformanceStats stats;
        stats.memoryUsed = estimateMemoryRequirement(input);
        
        try {
            // 获取内存管理器
            auto& memManager = OSCEANGPUFramework::getMemoryManager();
            
            // 分配GPU内存
            ScopedTimer transferTimer(stats.transferTime);
            
            auto gpuA = memManager.allocate(size * sizeof(float), context.deviceId);
            auto gpuB = memManager.allocate(size * sizeof(float), context.deviceId);
            auto gpuC = memManager.allocate(size * sizeof(float), context.deviceId);
            
            if (!gpuA.isValid() || !gpuB.isValid() || !gpuC.isValid()) {
                return createErrorResult(GPUError::OUT_OF_MEMORY, "Failed to allocate GPU memory");
            }
            
            // 传输数据到GPU
            memManager.copyHostToDevice(gpuA, vecA.data(), size * sizeof(float));
            memManager.copyHostToDevice(gpuB, vecB.data(), size * sizeof(float));
            
            transferTimer.~ScopedTimer();  // 结束传输计时
            
            // 执行GPU计算
            ScopedTimer kernelTimer(stats.kernelTime);
            
            bool success = false;
            auto devices = OSCEANGPUFramework::getAvailableDevices();
            if (context.deviceId < devices.size()) {
                auto& device = devices[context.deviceId];
                
                // 根据设备API选择实现
                if (std::find(device.supportedAPIs.begin(), device.supportedAPIs.end(), 
                             ComputeAPI::CUDA) != device.supportedAPIs.end()) {
                    success = executeCUDA(gpuA, gpuB, gpuC, size, context);
                } else if (std::find(device.supportedAPIs.begin(), device.supportedAPIs.end(),
                                    ComputeAPI::OPENCL) != device.supportedAPIs.end()) {
                    success = executeOpenCL(gpuA, gpuB, gpuC, size, context);
                } else {
                    // CPU后备实现
                    success = executeCPU(gpuA, gpuB, gpuC, size, context);
                }
            }
            
            kernelTimer.~ScopedTimer();  // 结束核函数计时
            
            if (!success) {
                memManager.deallocate(gpuA);
                memManager.deallocate(gpuB);
                memManager.deallocate(gpuC);
                return createErrorResult(GPUError::KERNEL_LAUNCH_ERROR, "Failed to execute kernel");
            }
            
            // 传输结果回主机
            ScopedTimer transferBackTimer(stats.transferTime);
            std::vector<float> result(size);
            memManager.copyDeviceToHost(result.data(), gpuC, size * sizeof(float));
            
            // 释放GPU内存
            memManager.deallocate(gpuA);
            memManager.deallocate(gpuB);
            memManager.deallocate(gpuC);
            
            // 计算吞吐量
            stats.totalTime = stats.kernelTime + stats.transferTime;
            stats.throughput = (size * sizeof(float) * 3) / (stats.totalTime * 1e6);  // GB/s
            
            return createSuccessResult(std::move(result), stats);
            
        } catch (const std::exception& e) {
            return createErrorResult(GPUError::UNKNOWN_ERROR, e.what());
        }
    }
    
private:
    /**
     * @brief CUDA实现
     */
    bool executeCUDA(
        const GPUMemoryHandle& gpuA,
        const GPUMemoryHandle& gpuB,
        const GPUMemoryHandle& gpuC,
        size_t size,
        const GPUExecutionContext& context) {
        
#ifdef OSCEAN_CUDA_ENABLED
        // 这里应该调用CUDA kernel
        // 为了示例，使用简单的CPU实现
        return executeCPU(gpuA, gpuB, gpuC, size, context);
#else
        return false;
#endif
    }
    
    /**
     * @brief OpenCL实现
     */
    bool executeOpenCL(
        const GPUMemoryHandle& gpuA,
        const GPUMemoryHandle& gpuB,
        const GPUMemoryHandle& gpuC,
        size_t size,
        const GPUExecutionContext& context) {
        
#ifdef OSCEAN_OPENCL_ENABLED
        // 这里应该调用OpenCL kernel
        // 为了示例，使用简单的CPU实现
        return executeCPU(gpuA, gpuB, gpuC, size, context);
#else
        return false;
#endif
    }
    
    /**
     * @brief CPU后备实现
     */
    bool executeCPU(
        const GPUMemoryHandle& gpuA,
        const GPUMemoryHandle& gpuB,
        const GPUMemoryHandle& gpuC,
        size_t size,
        const GPUExecutionContext& context) {
        
        // 在实际实现中，这里应该操作GPU内存
        // 为了示例，进行简单的模拟
        float* a = static_cast<float*>(gpuA.devicePtr);
        float* b = static_cast<float*>(gpuB.devicePtr);
        float* c = static_cast<float*>(gpuC.devicePtr);
        
        if (!a || !b || !c) return false;
        
        // 简单的向量加法
        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] + b[i];
        }
        
        return true;
    }
};

/**
 * @brief 矩阵乘法GPU算法示例（批处理版本）
 */
class MatrixMultiplyBatchGPUAlgorithm : public BatchGPUAlgorithmBase<
    std::pair<std::vector<float>, std::vector<float>>,  // 输入：矩阵A和B的数据
    std::vector<float>                                   // 输出：矩阵C的数据
> {
private:
    int m_M, m_N, m_K;  // 矩阵维度：A(M×K), B(K×N), C(M×N)
    
public:
    MatrixMultiplyBatchGPUAlgorithm(int M, int N, int K)
        : BatchGPUAlgorithmBase(
            "MatrixMultiplyBatch",
            "1.0.0",
            {ComputeAPI::CUDA, ComputeAPI::OPENCL}
        )
        , m_M(M), m_N(N), m_K(K) {
    }
    
protected:
    /**
     * @brief 估算单个矩阵乘法的内存大小
     */
    size_t estimateItemSize() const override {
        return sizeof(float) * (m_M * m_K + m_K * m_N + m_M * m_N);
    }
    
    /**
     * @brief 执行批量矩阵乘法
     */
    GPUAlgorithmResult<std::vector<std::vector<float>>> executeInternal(
        const std::vector<std::pair<std::vector<float>, std::vector<float>>>& batch,
        const GPUExecutionContext& context) override {
        
        // 这里应该实现批量矩阵乘法
        // 为了示例，返回空结果
        std::vector<std::vector<float>> results;
        results.reserve(batch.size());
        
        for (const auto& item : batch) {
            results.push_back(std::vector<float>(m_M * m_N, 0.0f));
        }
        
        typename GPUAlgorithmResult<std::vector<std::vector<float>>>::PerformanceStats stats;
        stats.kernelTime = 1.0;  // 模拟执行时间
        stats.transferTime = 0.5;
        stats.totalTime = 1.5;
        stats.memoryUsed = batch.size() * estimateItemSize();
        stats.throughput = stats.memoryUsed / (stats.totalTime * 1e6);
        
        return createSuccessResult(std::move(results), stats);
    }
};

/**
 * @brief 矩阵乘法GPU算法示例
 * 
 * 实现矩阵乘法：C = A * B
 * 使用GridData作为矩阵表示
 */
class MatrixMultiplyGPUAlgorithm : public GPUAlgorithmBase<
    std::pair<std::shared_ptr<GridData>, std::shared_ptr<GridData>>,  // 输入：两个矩阵
    std::shared_ptr<GridData>                                          // 输出：结果矩阵
> {
public:
    MatrixMultiplyGPUAlgorithm()
        : GPUAlgorithmBase(
            "MatrixMultiply",
            "1.0.0",
            {ComputeAPI::CUDA, ComputeAPI::OPENCL}
        ) {
    }
    
    size_t estimateMemoryRequirement(
        const std::pair<std::shared_ptr<GridData>, std::shared_ptr<GridData>>& input) const override {
        const auto& [A, B] = input;
        // A矩阵 + B矩阵 + C矩阵
        return A->getSizeInBytes() + B->getSizeInBytes() + 
               A->getRows() * B->getCols() * sizeof(float);
    }
    
    std::pair<bool, std::string> validateInput(
        const std::pair<std::shared_ptr<GridData>, std::shared_ptr<GridData>>& input) const override {
        const auto& [A, B] = input;
        
        if (!A || !B) {
            return {false, "Input matrices cannot be null"};
        }
        
        if (A->isEmpty() || B->isEmpty()) {
            return {false, "Input matrices cannot be empty"};
        }
        
        if (A->getCols() != B->getRows()) {
            return {false, "Matrix dimensions incompatible for multiplication"};
        }
        
        if (A->getDataType() != DataType::FLOAT32 || B->getDataType() != DataType::FLOAT32) {
            return {false, "Only FLOAT32 matrices are supported"};
        }
        
        return {true, ""};
    }
    
protected:
    GPUAlgorithmResult<std::shared_ptr<GridData>> executeInternal(
        const std::pair<std::shared_ptr<GridData>, std::shared_ptr<GridData>>& input,
        const GPUExecutionContext& context) override {
        
        const auto& [A, B] = input;
        
        // 验证输入
        auto [valid, errorMsg] = validateInput(input);
        if (!valid) {
            return createErrorResult(GPUError::INVALID_KERNEL, errorMsg);
        }
        
        // 准备输出矩阵
        auto C = std::make_shared<GridData>(A->getRows(), B->getCols(), 1, DataType::FLOAT32);
        
        // 选择执行路径
        if (context.computeAPI == ComputeAPI::CUDA) {
            return executeCUDA(A, B, C, context);
        } else if (context.computeAPI == ComputeAPI::OPENCL) {
            return executeOpenCL(A, B, C, context);
        } else {
            // CPU后备实现
            return executeCPU(A, B, C);
        }
    }
    
private:
    GPUAlgorithmResult<std::shared_ptr<GridData>> executeCUDA(
        const std::shared_ptr<GridData>& A,
        const std::shared_ptr<GridData>& B,
        std::shared_ptr<GridData>& C,
        const GPUExecutionContext& context) {
        
#ifdef OSCEAN_CUDA_ENABLED
        // TODO: 实现CUDA矩阵乘法
        OSCEAN_LOG_INFO("MatrixMultiply", "CUDA implementation not yet available, using CPU fallback");
#endif
        return executeCPU(A, B, C);
    }
    
    GPUAlgorithmResult<std::shared_ptr<GridData>> executeOpenCL(
        const std::shared_ptr<GridData>& A,
        const std::shared_ptr<GridData>& B,
        std::shared_ptr<GridData>& C,
        const GPUExecutionContext& context) {
        
#ifdef OSCEAN_OPENCL_ENABLED
        // TODO: 实现OpenCL矩阵乘法
        OSCEAN_LOG_INFO("MatrixMultiply", "OpenCL implementation not yet available, using CPU fallback");
#endif
        return executeCPU(A, B, C);
    }
    
    GPUAlgorithmResult<std::shared_ptr<GridData>> executeCPU(
        const std::shared_ptr<GridData>& A,
        const std::shared_ptr<GridData>& B,
        std::shared_ptr<GridData>& C) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 简单的CPU矩阵乘法实现
        size_t M = A->getRows();
        size_t K = A->getCols();
        size_t N = B->getCols();
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A->getValue<float>(i, k, 0) * B->getValue<float>(k, j, 0);
                }
                C->setValue(i, j, 0, sum);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        typename GPUAlgorithmResult<std::shared_ptr<GridData>>::PerformanceStats stats;
        stats.kernelTime = duration.count() / 1000.0;
        stats.transferTime = 0.0;
        stats.totalTime = stats.kernelTime;
        stats.memoryUsed = estimateMemoryRequirement({A, B});
        stats.throughput = (2.0 * M * K * N) / (stats.totalTime * 1e6);  // GFLOPS
        
        return createSuccessResult(std::move(C), stats);
    }
};

/**
 * @brief 图像模糊GPU算法示例
 * 
 * 使用高斯模糊处理图像
 * 使用GridData表示图像（支持多通道）
 */
class ImageBlurGPUAlgorithm : public GPUAlgorithmBase<
    std::shared_ptr<GridData>,    // 输入：原始图像
    std::shared_ptr<GridData>     // 输出：模糊后的图像
> {
public:
    ImageBlurGPUAlgorithm()
        : GPUAlgorithmBase(
            "ImageBlur",
            "1.0.0",
            {ComputeAPI::CUDA, ComputeAPI::OPENCL, ComputeAPI::VULKAN_COMPUTE}
        ) {
        generateGaussianKernel();
    }
    
    void setKernelSize(int size) {
        m_kernelSize = size;
        generateGaussianKernel();
    }
    
    void setSigma(float sigma) {
        m_sigma = sigma;
        generateGaussianKernel();
    }
    
    size_t estimateMemoryRequirement(const std::shared_ptr<GridData>& input) const override {
        if (!input) return 0;
        // 输入图像 + 输出图像 + 高斯核
        return input->getSizeInBytes() * 2 +
               m_kernelSize * m_kernelSize * sizeof(float);
    }
    
    std::pair<bool, std::string> validateInput(const std::shared_ptr<GridData>& input) const override {
        if (!input) {
            return {false, "Input image cannot be null"};
        }
        
        if (input->isEmpty()) {
            return {false, "Input image cannot be empty"};
        }
        
        if (input->getDataType() != DataType::UINT8) {
            return {false, "Only UINT8 images are supported"};
        }
        
        size_t bands = input->getBandCount();
        if (bands != 1 && bands != 3 && bands != 4) {
            return {false, "Unsupported number of channels: " + std::to_string(bands)};
        }
        
        return {true, ""};
    }
    
protected:
    GPUAlgorithmResult<std::shared_ptr<GridData>> executeInternal(
        const std::shared_ptr<GridData>& input,
        const GPUExecutionContext& context) override {
        
        // 验证输入
        auto [valid, errorMsg] = validateInput(input);
        if (!valid) {
            return createErrorResult(GPUError::INVALID_KERNEL, errorMsg);
        }
        
        // 准备输出图像
        auto output = std::make_shared<GridData>(
            input->getCols(),  // width
            input->getRows(),  // height
            input->getBandCount(),
            input->getDataType()
        );
        
        // CPU实现（作为示例）
        return executeCPU(input, output);
    }
    
private:
    int m_kernelSize = 5;
    float m_sigma = 1.0f;
    std::vector<float> m_gaussianKernel;
    
    void generateGaussianKernel() {
        m_gaussianKernel.resize(m_kernelSize * m_kernelSize);
        
        int halfSize = m_kernelSize / 2;
        float sum = 0.0f;
        
        // 生成高斯核
        for (int y = -halfSize; y <= halfSize; ++y) {
            for (int x = -halfSize; x <= halfSize; ++x) {
                float value = std::exp(-(x*x + y*y) / (2.0f * m_sigma * m_sigma));
                m_gaussianKernel[(y + halfSize) * m_kernelSize + (x + halfSize)] = value;
                sum += value;
            }
        }
        
        // 归一化
        for (auto& val : m_gaussianKernel) {
            val /= sum;
        }
    }
    
    GPUAlgorithmResult<std::shared_ptr<GridData>> executeCPU(
        const std::shared_ptr<GridData>& input,
        std::shared_ptr<GridData>& output) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        int halfSize = m_kernelSize / 2;
        size_t width = input->getCols();
        size_t height = input->getRows();
        size_t bands = input->getBandCount();
        
        // 对每个像素应用高斯模糊
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                for (size_t b = 0; b < bands; ++b) {
                    float sum = 0.0f;
                    
                    // 应用卷积核
                    for (int ky = -halfSize; ky <= halfSize; ++ky) {
                        for (int kx = -halfSize; kx <= halfSize; ++kx) {
                            size_t px = std::clamp(static_cast<size_t>(x + kx), size_t(0), width - 1);
                            size_t py = std::clamp(static_cast<size_t>(y + ky), size_t(0), height - 1);
                            
                            int kernelIdx = (ky + halfSize) * m_kernelSize + (kx + halfSize);
                            uint8_t pixelValue = input->getValue<uint8_t>(py, px, b);
                            
                            sum += pixelValue * m_gaussianKernel[kernelIdx];
                        }
                    }
                    
                    output->setValue(y, x, b, static_cast<uint8_t>(std::clamp(sum, 0.0f, 255.0f)));
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        typename GPUAlgorithmResult<std::shared_ptr<GridData>>::PerformanceStats stats;
        stats.kernelTime = duration.count() / 1000.0;
        stats.transferTime = 0.0;
        stats.totalTime = stats.kernelTime;
        stats.memoryUsed = estimateMemoryRequirement(input);
        stats.throughput = (width * height * bands) / (stats.totalTime * 1e6);
        
        return createSuccessResult(std::move(output), stats);
    }
};

/**
 * @brief 注册示例算法到注册表
 */
void registerExampleAlgorithms() {
    auto& registry = GPUAlgorithmRegistry::getInstance();
    
    // 注册向量加法算法
    registry.registerAlgorithm<
        std::pair<std::vector<float>, std::vector<float>>,
        std::vector<float>
    >(
        "VectorAdd",
        std::make_shared<SimpleGPUAlgorithmFactory<
            VectorAddGPUAlgorithm,
            std::pair<std::vector<float>, std::vector<float>>,
            std::vector<float>
        >>()
    );
    
    OSCEAN_LOG_INFO("GPUAlgorithmExamples", "Registered example GPU algorithms");
}

} // namespace oscean::common_utils::gpu 