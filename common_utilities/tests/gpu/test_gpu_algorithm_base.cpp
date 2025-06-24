/**
 * @file test_gpu_algorithm_base.cpp
 * @brief GPU算法基础实现测试
 */

#include <gtest/gtest.h>
#include "common_utils/gpu/gpu_algorithm_base.h"
#include "common_utils/gpu/oscean_gpu_framework.h"
#include "common_utils/utilities/logging_utils.h"
#include <numeric>
#include <algorithm>

using namespace oscean::common_utils::gpu;

/**
 * @brief 简单的测试算法：向量元素平方
 */
class SquareVectorGPUAlgorithm : public GPUAlgorithmBase<
    std::vector<float>,    // 输入
    std::vector<float>     // 输出
> {
public:
    SquareVectorGPUAlgorithm()
        : GPUAlgorithmBase(
            "SquareVector",
            "1.0.0",
            {ComputeAPI::CUDA, ComputeAPI::OPENCL}
        ) {
    }
    
    size_t estimateMemoryRequirement(const std::vector<float>& input) const override {
        return input.size() * sizeof(float) * 2;  // 输入 + 输出
    }
    
    std::pair<bool, std::string> validateInput(const std::vector<float>& input) const override {
        if (input.empty()) {
            return {false, "Input vector cannot be empty"};
        }
        return {true, ""};
    }
    
protected:
    GPUAlgorithmResult<std::vector<float>> executeInternal(
        const std::vector<float>& input,
        const GPUExecutionContext& context) override {
        
        // 验证输入
        auto [valid, errorMsg] = validateInput(input);
        if (!valid) {
            return createErrorResult(GPUError::INVALID_KERNEL, errorMsg);
        }
        
        // 模拟GPU执行
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = input[i] * input[i];
        }
        
        // 创建性能统计
        typename GPUAlgorithmResult<std::vector<float>>::PerformanceStats stats;
        stats.kernelTime = 0.5;  // 模拟执行时间
        stats.transferTime = 0.2;
        stats.totalTime = 0.7;
        stats.memoryUsed = estimateMemoryRequirement(input);
        stats.throughput = stats.memoryUsed / (stats.totalTime * 1e6);
        
        return createSuccessResult(std::move(output), stats);
    }
};

/**
 * @brief 批处理测试算法：批量向量归一化
 */
class NormalizeVectorBatchGPUAlgorithm : public BatchGPUAlgorithmBase<float, float> {
public:
    NormalizeVectorBatchGPUAlgorithm()
        : BatchGPUAlgorithmBase(
            "NormalizeVectorBatch",
            "1.0.0",
            {ComputeAPI::CUDA, ComputeAPI::OPENCL}
        ) {
    }
    
protected:
    size_t estimateItemSize() const override {
        return sizeof(float) * 2;  // 输入 + 输出
    }
    
    GPUAlgorithmResult<std::vector<float>> executeInternal(
        const std::vector<float>& batch,
        const GPUExecutionContext& context) override {
        
        if (batch.empty()) {
            return createErrorResult(GPUError::INVALID_KERNEL, "Batch cannot be empty");
        }
        
        // 计算最大值和最小值
        float minVal = *std::min_element(batch.begin(), batch.end());
        float maxVal = *std::max_element(batch.begin(), batch.end());
        float range = maxVal - minVal;
        
        // 归一化到[0, 1]
        std::vector<float> normalized(batch.size());
        if (range > 0) {
            for (size_t i = 0; i < batch.size(); ++i) {
                normalized[i] = (batch[i] - minVal) / range;
            }
        } else {
            // 所有值相同，返回0.5
            std::fill(normalized.begin(), normalized.end(), 0.5f);
        }
        
        typename GPUAlgorithmResult<std::vector<float>>::PerformanceStats stats;
        stats.kernelTime = 1.0;
        stats.transferTime = 0.5;
        stats.totalTime = 1.5;
        stats.memoryUsed = batch.size() * sizeof(float) * 2;
        stats.throughput = stats.memoryUsed / (stats.totalTime * 1e6);
        
        return createSuccessResult(std::move(normalized), stats);
    }
};

class GPUAlgorithmBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化GPU框架
        OSCEANGPUFramework::initialize();
    }
    
    void TearDown() override {
        // 关闭GPU框架
        OSCEANGPUFramework::shutdown();
    }
};

TEST_F(GPUAlgorithmBaseTest, BasicAlgorithmExecution) {
    SquareVectorGPUAlgorithm algorithm;
    
    // 测试算法属性
    EXPECT_EQ(algorithm.getAlgorithmName(), "SquareVector");
    EXPECT_EQ(algorithm.getVersion(), "1.0.0");
    
    auto supportedAPIs = algorithm.getSupportedAPIs();
    EXPECT_EQ(supportedAPIs.size(), 2);
    EXPECT_TRUE(std::find(supportedAPIs.begin(), supportedAPIs.end(), 
                         ComputeAPI::CUDA) != supportedAPIs.end());
    EXPECT_TRUE(std::find(supportedAPIs.begin(), supportedAPIs.end(), 
                         ComputeAPI::OPENCL) != supportedAPIs.end());
}

TEST_F(GPUAlgorithmBaseTest, SynchronousExecution) {
    SquareVectorGPUAlgorithm algorithm;
    
    // 准备输入数据
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    // 创建执行上下文
    GPUExecutionContext context;
    context.deviceId = 0;
    context.workGroupSize = 256;
    
    // 同步执行
    auto result = algorithm.execute(input, context);
    
    // 验证结果
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.error, GPUError::SUCCESS);
    EXPECT_TRUE(result.errorMessage.empty());
    
    // 验证输出数据
    ASSERT_EQ(result.data.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(result.data[i], input[i] * input[i]);
    }
    
    // 验证性能统计
    EXPECT_GT(result.stats.kernelTime, 0);
    EXPECT_GT(result.stats.transferTime, 0);
    EXPECT_GT(result.stats.totalTime, 0);
    EXPECT_GT(result.stats.throughput, 0);
}

TEST_F(GPUAlgorithmBaseTest, AsynchronousExecution) {
    SquareVectorGPUAlgorithm algorithm;
    
    // 准备输入数据
    std::vector<float> input = {2.0f, 4.0f, 6.0f, 8.0f};
    
    // 创建执行上下文
    GPUExecutionContext context;
    context.deviceId = 0;
    
    // 异步执行
    auto future = algorithm.executeAsync(input, context);
    
    // 等待结果
    auto result = future.get();
    
    // 验证结果
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.data.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(result.data[i], input[i] * input[i]);
    }
}

TEST_F(GPUAlgorithmBaseTest, InputValidation) {
    SquareVectorGPUAlgorithm algorithm;
    
    // 测试空输入
    std::vector<float> emptyInput;
    GPUExecutionContext context;
    
    auto result = algorithm.execute(emptyInput, context);
    
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error, GPUError::INVALID_KERNEL);
    EXPECT_FALSE(result.errorMessage.empty());
}

TEST_F(GPUAlgorithmBaseTest, MemoryEstimation) {
    SquareVectorGPUAlgorithm algorithm;
    
    std::vector<float> input(1000);
    size_t estimated = algorithm.estimateMemoryRequirement(input);
    
    // 输入 + 输出
    size_t expected = 1000 * sizeof(float) * 2;
    EXPECT_EQ(estimated, expected);
}

TEST_F(GPUAlgorithmBaseTest, DeviceSupport) {
    SquareVectorGPUAlgorithm algorithm;
    
    // 创建测试设备信息
    GPUDeviceInfo cudaDevice;
    cudaDevice.name = "Test CUDA Device";
    cudaDevice.vendor = GPUVendor::NVIDIA;
    cudaDevice.supportedAPIs = {ComputeAPI::CUDA};
    
    GPUDeviceInfo openclDevice;
    openclDevice.name = "Test OpenCL Device";
    openclDevice.vendor = GPUVendor::AMD;
    openclDevice.supportedAPIs = {ComputeAPI::OPENCL};
    
    GPUDeviceInfo unsupportedDevice;
    unsupportedDevice.name = "Test Unsupported Device";
    unsupportedDevice.vendor = GPUVendor::UNKNOWN;
    unsupportedDevice.supportedAPIs = {ComputeAPI::DIRECTCOMPUTE};
    
    // 测试设备支持
    EXPECT_TRUE(algorithm.supportsDevice(cudaDevice));
    EXPECT_TRUE(algorithm.supportsDevice(openclDevice));
    EXPECT_FALSE(algorithm.supportsDevice(unsupportedDevice));
}

TEST_F(GPUAlgorithmBaseTest, BatchAlgorithmExecution) {
    NormalizeVectorBatchGPUAlgorithm algorithm;
    
    // 准备批量数据
    std::vector<float> batch = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 7.0f};
    
    GPUExecutionContext context;
    context.deviceId = 0;
    
    auto result = algorithm.execute(batch, context);
    
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.data.size(), batch.size());
    
    // 验证归一化结果
    for (const auto& val : result.data) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
    
    // 验证最小值和最大值
    auto minIt = std::min_element(result.data.begin(), result.data.end());
    auto maxIt = std::max_element(result.data.begin(), result.data.end());
    EXPECT_FLOAT_EQ(*minIt, 0.0f);
    EXPECT_FLOAT_EQ(*maxIt, 1.0f);
}

TEST_F(GPUAlgorithmBaseTest, BatchSizeOptimization) {
    NormalizeVectorBatchGPUAlgorithm algorithm;
    
    // 创建测试设备
    GPUDeviceInfo device;
    device.memoryDetails.freeGlobalMemory = 1024 * 1024 * 1024;  // 1GB
    device.computeUnits.totalCores = 4608;  // RTX 4070的核心数量
    device.vendor = GPUVendor::NVIDIA;
    
    // 测试不同数据量的批处理大小
    size_t smallBatch = algorithm.getOptimalBatchSize(100, device);
    size_t mediumBatch = algorithm.getOptimalBatchSize(10000, device);
    size_t largeBatch = algorithm.getOptimalBatchSize(1000000, device);
    
    // 验证批处理大小合理
    EXPECT_GT(smallBatch, 0);
    EXPECT_GE(mediumBatch, smallBatch);
    EXPECT_GE(largeBatch, mediumBatch);
    
    // 验证对齐到warp大小
    EXPECT_EQ(smallBatch % 32, 0);
    EXPECT_EQ(mediumBatch % 32, 0);
    EXPECT_EQ(largeBatch % 32, 0);
}

TEST_F(GPUAlgorithmBaseTest, PerformanceMetrics) {
    SquareVectorGPUAlgorithm algorithm;
    
    // 重置统计
    algorithm.resetMetrics();
    
    // 执行多次
    std::vector<float> input(1000, 1.0f);
    GPUExecutionContext context;
    
    for (int i = 0; i < 5; ++i) {
        algorithm.execute(input, context);
    }
    
    // 获取性能统计
    auto metrics = algorithm.getPerformanceMetrics();
    
    EXPECT_EQ(metrics.executionCount, 5);
    EXPECT_GT(metrics.totalKernelTime, 0);
    EXPECT_GT(metrics.totalTransferTime, 0);
    EXPECT_LE(metrics.minKernelTime, metrics.maxKernelTime);
}

TEST_F(GPUAlgorithmBaseTest, SimpleFactory) {
    using VectorAlgorithm = IGPUAlgorithm<std::vector<float>, std::vector<float>>;
    using VectorFactory = SimpleGPUAlgorithmFactory<
        SquareVectorGPUAlgorithm,
        std::vector<float>,
        std::vector<float>
    >;
    
    VectorFactory factory;
    
    // 创建CUDA版本
    auto cudaAlgorithm = factory.createAlgorithm(ComputeAPI::CUDA);
    ASSERT_NE(cudaAlgorithm, nullptr);
    EXPECT_EQ(cudaAlgorithm->getAlgorithmName(), "SquareVector");
    
    // 创建OpenCL版本
    auto openclAlgorithm = factory.createAlgorithm(ComputeAPI::OPENCL);
    ASSERT_NE(openclAlgorithm, nullptr);
    EXPECT_EQ(openclAlgorithm->getAlgorithmName(), "SquareVector");
    
    // 测试设备最优算法选择
    GPUDeviceInfo device;
    device.vendor = GPUVendor::NVIDIA;
    device.supportedAPIs = {ComputeAPI::CUDA, ComputeAPI::OPENCL};
    
    auto optimalAlgorithm = factory.createOptimalAlgorithm(device);
    ASSERT_NE(optimalAlgorithm, nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // 初始化日志系统
    OSCEAN_LOG_INFO("GPUAlgorithmBaseTest", "Starting GPU algorithm base tests");
    
    return RUN_ALL_TESTS();
} 