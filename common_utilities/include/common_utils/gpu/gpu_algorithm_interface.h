/**
 * @file gpu_algorithm_interface.h
 * @brief GPU算法统一接口定义
 * 
 * 提供跨平台GPU算法的抽象接口，支持CUDA、OpenCL、ROCm等多种GPU API
 */

#pragma once

#include "gpu_common.h"
#include "gpu_device_info.h"
#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <boost/optional.hpp>
#include <memory>
#include <vector>
#include <string>

namespace oscean::common_utils::gpu {

/**
 * @brief GPU执行上下文
 */
struct GPUExecutionContext {
    int deviceId = 0;                          // GPU设备ID
    size_t workGroupSize = 256;                // 工作组大小
    size_t globalWorkSize = 0;                 // 全局工作项数量
    GPUMemoryType memoryType = GPUMemoryType::DEVICE;  // 内存类型
    bool enableProfiling = false;              // 是否启用性能分析
    boost::optional<size_t> sharedMemorySize;  // 共享内存大小（可选）
    
    // 性能提示
    struct PerformanceHints {
        bool preferCoalesced = true;           // 优先合并内存访问
        bool preferCached = true;              // 优先使用缓存
        int occupancyTarget = 75;              // 目标占用率(%)
    } hints;
};

/**
 * @brief GPU算法执行结果
 */
template<typename T>
struct GPUAlgorithmResult {
    bool success = false;                      // 执行是否成功
    T data;                                    // 结果数据
    GPUError error = GPUError::SUCCESS;        // 错误码
    std::string errorMessage;                  // 错误信息
    
    // 性能统计
    struct PerformanceStats {
        double kernelTime = 0.0;               // 核函数执行时间(ms)
        double transferTime = 0.0;             // 数据传输时间(ms)
        double totalTime = 0.0;                // 总时间(ms)
        size_t memoryUsed = 0;                 // 使用的内存(bytes)
        double throughput = 0.0;               // 吞吐量(GB/s)
    } stats;
};

/**
 * @brief GPU算法基础接口
 * 
 * 所有GPU算法必须实现此接口，提供统一的执行模型
 */
template<typename InputType, typename OutputType>
class IGPUAlgorithm {
public:
    virtual ~IGPUAlgorithm() = default;
    
    /**
     * @brief 异步执行GPU算法
     * @param input 输入数据
     * @param context GPU执行上下文
     * @return 异步执行结果
     */
    virtual boost::future<GPUAlgorithmResult<OutputType>> executeAsync(
        const InputType& input,
        const GPUExecutionContext& context) = 0;
    
    /**
     * @brief 同步执行GPU算法
     * @param input 输入数据
     * @param context GPU执行上下文
     * @return 执行结果
     */
    virtual GPUAlgorithmResult<OutputType> execute(
        const InputType& input,
        const GPUExecutionContext& context) {
        // 默认实现：调用异步版本并等待
        return executeAsync(input, context).get();
    }
    
    /**
     * @brief 获取支持的GPU API列表
     * @return 支持的API列表
     */
    virtual std::vector<ComputeAPI> getSupportedAPIs() const = 0;
    
    /**
     * @brief 检查是否支持指定的GPU设备
     * @param device GPU设备信息
     * @return 是否支持
     */
    virtual bool supportsDevice(const GPUDeviceInfo& device) const = 0;
    
    /**
     * @brief 估算内存需求
     * @param input 输入数据
     * @return 预计需要的GPU内存(bytes)
     */
    virtual size_t estimateMemoryRequirement(const InputType& input) const = 0;
    
    /**
     * @brief 获取算法名称
     * @return 算法名称
     */
    virtual std::string getAlgorithmName() const = 0;
    
    /**
     * @brief 获取算法版本
     * @return 版本字符串
     */
    virtual std::string getVersion() const = 0;
    
    /**
     * @brief 验证输入数据
     * @param input 输入数据
     * @return 验证结果和错误信息
     */
    virtual std::pair<bool, std::string> validateInput(const InputType& input) const {
        // 默认实现：总是返回有效
        return {true, ""};
    }
    
    /**
     * @brief 获取最优工作组大小
     * @param input 输入数据
     * @param device GPU设备信息
     * @return 推荐的工作组大小
     */
    virtual size_t getOptimalWorkGroupSize(
        const InputType& input,
        const GPUDeviceInfo& device) const {
        // 默认实现：返回设备的warp/wavefront大小的倍数
        if (device.vendor == GPUVendor::NVIDIA) {
            return 256;  // NVIDIA通常使用256
        } else if (device.vendor == GPUVendor::AMD) {
            return 256;  // AMD也常用256
        } else {
            return 128;  // 其他设备使用较小值
        }
    }
};

/**
 * @brief GPU算法工厂接口
 */
template<typename InputType, typename OutputType>
class IGPUAlgorithmFactory {
public:
    virtual ~IGPUAlgorithmFactory() = default;
    
    /**
     * @brief 创建GPU算法实例
     * @param api 目标GPU API
     * @return 算法实例
     */
    virtual std::unique_ptr<IGPUAlgorithm<InputType, OutputType>> 
        createAlgorithm(ComputeAPI api) = 0;
    
    /**
     * @brief 根据设备自动选择最优算法
     * @param device GPU设备信息
     * @return 最优算法实例
     */
    virtual std::unique_ptr<IGPUAlgorithm<InputType, OutputType>> 
        createOptimalAlgorithm(const GPUDeviceInfo& device) {
        // 默认实现：选择设备最优API
        auto api = device.getBestAPI();
        return createAlgorithm(api);
    }
};

/**
 * @brief 批处理GPU算法接口
 * 
 * 支持批量数据处理的GPU算法
 */
template<typename InputType, typename OutputType>
class IBatchGPUAlgorithm : public IGPUAlgorithm<std::vector<InputType>, std::vector<OutputType>> {
public:
    /**
     * @brief 获取最优批处理大小
     * @param totalItems 总数据项数
     * @param device GPU设备信息
     * @return 推荐的批处理大小
     */
    virtual size_t getOptimalBatchSize(
        size_t totalItems,
        const GPUDeviceInfo& device) const = 0;
    
    /**
     * @brief 是否支持流式处理
     * @return 是否支持
     */
    virtual bool supportsStreaming() const = 0;
};

/**
 * @brief 流式GPU算法接口
 * 
 * 支持流式数据处理，适合大数据集
 */
template<typename InputType, typename OutputType>
class IStreamingGPUAlgorithm {
public:
    virtual ~IStreamingGPUAlgorithm() = default;
    
    /**
     * @brief 开始流式处理
     * @param context GPU执行上下文
     * @return 是否成功开始
     */
    virtual bool beginStreaming(const GPUExecutionContext& context) = 0;
    
    /**
     * @brief 处理一批数据
     * @param batch 输入批次
     * @return 处理结果
     */
    virtual boost::future<GPUAlgorithmResult<std::vector<OutputType>>> 
        processBatch(const std::vector<InputType>& batch) = 0;
    
    /**
     * @brief 结束流式处理
     */
    virtual void endStreaming() = 0;
    
    /**
     * @brief 获取流式处理状态
     */
    struct StreamingStatus {
        size_t itemsProcessed = 0;
        size_t batchesProcessed = 0;
        double averageLatency = 0.0;
        double throughput = 0.0;
    };
    
    virtual StreamingStatus getStreamingStatus() const = 0;
};

/**
 * @brief GPU算法注册表
 * 
 * 管理所有可用的GPU算法
 */
class GPUAlgorithmRegistry {
public:
    static GPUAlgorithmRegistry& getInstance() {
        static GPUAlgorithmRegistry instance;
        return instance;
    }
    
    /**
     * @brief 注册算法工厂
     * @param name 算法名称
     * @param factory 算法工厂
     */
    template<typename InputType, typename OutputType>
    void registerAlgorithm(
        const std::string& name,
        std::shared_ptr<IGPUAlgorithmFactory<InputType, OutputType>> factory) {
        // 实现略
    }
    
    /**
     * @brief 获取算法工厂
     * @param name 算法名称
     * @return 算法工厂
     */
    template<typename InputType, typename OutputType>
    std::shared_ptr<IGPUAlgorithmFactory<InputType, OutputType>> 
        getAlgorithmFactory(const std::string& name) {
        // 实现略
        return nullptr;
    }
    
private:
    GPUAlgorithmRegistry() = default;
    ~GPUAlgorithmRegistry() = default;
    GPUAlgorithmRegistry(const GPUAlgorithmRegistry&) = delete;
    GPUAlgorithmRegistry& operator=(const GPUAlgorithmRegistry&) = delete;
};

} // namespace oscean::common_utils::gpu 