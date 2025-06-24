/**
 * @file gpu_algorithm_base.h
 * @brief GPU算法基础实现类
 * 
 * 提供IGPUAlgorithm接口的基础实现，简化具体算法的开发
 */

#pragma once

#include "gpu_algorithm_interface.h"
#include "multi_gpu_scheduler.h"
#include "multi_gpu_memory_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include <chrono>
#include <mutex>

// 前向声明，避免循环包含
namespace oscean::common_utils::gpu {
    class OSCEANGPUFramework;
}

namespace oscean::common_utils::gpu {

/**
 * @brief GPU算法基础实现类
 * 
 * 提供通用功能实现，子类只需要实现核心算法逻辑
 */
template<typename InputType, typename OutputType>
class GPUAlgorithmBase : public IGPUAlgorithm<InputType, OutputType> {
protected:
    std::string m_algorithmName;
    std::string m_version;
    std::vector<ComputeAPI> m_supportedAPIs;
    mutable std::mutex m_mutex;
    
    // 性能统计
    struct PerformanceMetrics {
        size_t executionCount = 0;
        double totalKernelTime = 0.0;
        double totalTransferTime = 0.0;
        double minKernelTime = std::numeric_limits<double>::max();
        double maxKernelTime = 0.0;
    } m_metrics;
    
public:
    /**
     * @brief 构造函数
     * @param name 算法名称
     * @param version 算法版本
     * @param supportedAPIs 支持的GPU API列表
     */
    GPUAlgorithmBase(
        const std::string& name,
        const std::string& version,
        const std::vector<ComputeAPI>& supportedAPIs)
        : m_algorithmName(name)
        , m_version(version)
        , m_supportedAPIs(supportedAPIs) {
    }
    
    virtual ~GPUAlgorithmBase() = default;
    
    /**
     * @brief 异步执行GPU算法
     */
    boost::future<GPUAlgorithmResult<OutputType>> executeAsync(
        const InputType& input,
        const GPUExecutionContext& context) override {
        
        // 创建promise用于返回结果
        auto promise = std::make_shared<boost::promise<GPUAlgorithmResult<OutputType>>>();
        auto future = promise->get_future();
        
        // 在线程中执行，避免包含oscean_gpu_framework.h
        boost::thread([this, input, context, promise]() {
            try {
                // 执行算法
                auto result = this->executeInternal(input, context);
                promise->set_value(result);
            } catch (const std::exception& e) {
                GPUAlgorithmResult<OutputType> errorResult;
                errorResult.success = false;
                errorResult.error = GPUError::UNKNOWN_ERROR;
                errorResult.errorMessage = e.what();
                promise->set_value(errorResult);
            }
        }).detach();
        
        return future;
    }
    
    /**
     * @brief 同步执行（使用基类默认实现）
     */
    using IGPUAlgorithm<InputType, OutputType>::execute;
    
    /**
     * @brief 获取支持的GPU API列表
     */
    std::vector<ComputeAPI> getSupportedAPIs() const override {
        return m_supportedAPIs;
    }
    
    /**
     * @brief 检查是否支持指定的GPU设备
     */
    bool supportsDevice(const GPUDeviceInfo& device) const override {
        // 检查设备是否支持任一API
        for (const auto& api : m_supportedAPIs) {
            auto it = std::find(
                device.supportedAPIs.begin(),
                device.supportedAPIs.end(),
                api
            );
            if (it != device.supportedAPIs.end()) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief 获取算法名称
     */
    std::string getAlgorithmName() const override {
        return m_algorithmName;
    }
    
    /**
     * @brief 获取算法版本
     */
    std::string getVersion() const override {
        return m_version;
    }
    
    /**
     * @brief 获取性能统计信息
     */
    PerformanceMetrics getPerformanceMetrics() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_metrics;
    }
    
    /**
     * @brief 重置性能统计
     */
    void resetMetrics() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_metrics = PerformanceMetrics();
    }
    
protected:
    /**
     * @brief 内部执行方法，子类必须实现
     * @param input 输入数据
     * @param context GPU执行上下文
     * @return 执行结果
     */
    virtual GPUAlgorithmResult<OutputType> executeInternal(
        const InputType& input,
        const GPUExecutionContext& context) = 0;
    
    /**
     * @brief 更新性能统计
     * @param kernelTime 核函数执行时间
     * @param transferTime 数据传输时间
     */
    void updateMetrics(double kernelTime, double transferTime) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_metrics.executionCount++;
        m_metrics.totalKernelTime += kernelTime;
        m_metrics.totalTransferTime += transferTime;
        m_metrics.minKernelTime = std::min(m_metrics.minKernelTime, kernelTime);
        m_metrics.maxKernelTime = std::max(m_metrics.maxKernelTime, kernelTime);
    }
    
    /**
     * @brief 创建成功结果
     * @param data 结果数据
     * @param stats 性能统计
     * @return 成功的结果对象
     */
    GPUAlgorithmResult<OutputType> createSuccessResult(
        OutputType&& data,
        const typename GPUAlgorithmResult<OutputType>::PerformanceStats& stats) {
        
        GPUAlgorithmResult<OutputType> result;
        result.success = true;
        result.data = std::move(data);
        result.error = GPUError::SUCCESS;
        result.stats = stats;
        
        // 更新内部统计
        updateMetrics(stats.kernelTime, stats.transferTime);
        
        return result;
    }
    
    /**
     * @brief 创建错误结果
     * @param error 错误码
     * @param message 错误信息
     * @return 错误的结果对象
     */
    GPUAlgorithmResult<OutputType> createErrorResult(
        GPUError error,
        const std::string& message) {
        
        GPUAlgorithmResult<OutputType> result;
        result.success = false;
        result.error = error;
        result.errorMessage = message;
        
        OSCEAN_LOG_ERROR(m_algorithmName, message);
        
        return result;
    }
    
    /**
     * @brief 测量执行时间的辅助类
     */
    class ScopedTimer {
    private:
        std::chrono::high_resolution_clock::time_point m_start;
        double& m_target;
        
    public:
        explicit ScopedTimer(double& target)
            : m_start(std::chrono::high_resolution_clock::now())
            , m_target(target) {
        }
        
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
            m_target = duration.count() / 1000.0;  // 转换为毫秒
        }
    };
};

/**
 * @brief 批处理GPU算法基础实现
 */
template<typename InputType, typename OutputType>
class BatchGPUAlgorithmBase 
    : public GPUAlgorithmBase<std::vector<InputType>, std::vector<OutputType>> {
    
protected:
    size_t m_defaultBatchSize = 1024;
    
public:
    using GPUAlgorithmBase<std::vector<InputType>, std::vector<OutputType>>::GPUAlgorithmBase;
    
    /**
     * @brief 获取最优批处理大小
     */
    size_t getOptimalBatchSize(
        size_t totalItems,
        const GPUDeviceInfo& device) const {
        
        // 基于设备内存和计算能力估算
        size_t itemSize = estimateItemSize();
        size_t availableMemory = device.memoryDetails.freeGlobalMemory * 0.8;  // 使用80%的可用内存
        size_t maxBatchFromMemory = availableMemory / itemSize;
        
        // 基于计算单元数量估算
        size_t maxBatchFromCompute = device.computeUnits.totalCores * 256;  // 每个核心256个项目
        
        // 取较小值，并限制在合理范围内
        size_t optimalBatch = std::min(maxBatchFromMemory, maxBatchFromCompute);
        optimalBatch = std::min(optimalBatch, totalItems);
        optimalBatch = std::max(optimalBatch, size_t(1));
        
        // 对齐到warp/wavefront大小
        size_t alignment = (device.vendor == GPUVendor::NVIDIA) ? 32 : 64;
        optimalBatch = (optimalBatch / alignment) * alignment;
        
        return optimalBatch > 0 ? optimalBatch : m_defaultBatchSize;
    }
    
    /**
     * @brief 默认不支持流式处理
     */
    bool supportsStreaming() const {
        return false;
    }
    
    /**
     * @brief 估算内存需求
     */
    size_t estimateMemoryRequirement(
        const std::vector<InputType>& input) const override {
        return input.size() * estimateItemSize();
    }
    
protected:
    /**
     * @brief 估算单个数据项的大小，子类应该重写
     */
    virtual size_t estimateItemSize() const {
        return sizeof(InputType) + sizeof(OutputType);
    }
};

/**
 * @brief 简单GPU算法工厂基类
 */
template<typename AlgorithmType, typename InputType, typename OutputType>
class SimpleGPUAlgorithmFactory : public IGPUAlgorithmFactory<InputType, OutputType> {
public:
    /**
     * @brief 创建算法实例
     */
    std::unique_ptr<IGPUAlgorithm<InputType, OutputType>> 
        createAlgorithm(ComputeAPI api) override {
        
        // 检查算法是否支持请求的API
        auto algorithm = std::make_unique<AlgorithmType>();
        auto supportedAPIs = algorithm->getSupportedAPIs();
        
        auto it = std::find(supportedAPIs.begin(), supportedAPIs.end(), api);
        if (it == supportedAPIs.end()) {
            OSCEAN_LOG_WARN("GPUAlgorithmFactory", 
                "Requested API not supported by algorithm, using default");
        }
        
        return algorithm;
    }
};

} // namespace oscean::common_utils::gpu 