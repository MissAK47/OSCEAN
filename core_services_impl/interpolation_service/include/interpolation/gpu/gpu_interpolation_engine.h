/**
 * @file gpu_interpolation_engine.h
 * @brief GPU加速的插值引擎接口
 */

#pragma once

#include <common_utils/gpu/gpu_algorithm_base.h>
#include <common_utils/gpu/gpu_device_info.h>
#include <core_services/common_data_types.h>
#include <core_services/interpolation/i_interpolation_service.h>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace oscean {
namespace core_services {
namespace interpolation {
namespace gpu {

// 使用标准接口中的InterpolationMethod枚举
using InterpolationMethod = oscean::core_services::interpolation::InterpolationMethod;

/**
 * @brief GPU插值参数
 */
struct GPUInterpolationParams {
    // 输入数据
    boost::shared_ptr<GridData> sourceData;
    
    // 输出网格定义
    int outputWidth;
    int outputHeight;
    BoundingBox outputBounds;
    
    // 插值方法
    InterpolationMethod method;
    
    // 性能参数
    bool enableCaching = false;
    bool useTextureMemory = false;
    int maxThreadsPerBlock = 256;
    
    // 质量参数
    float smoothingFactor = 1.0f;
    bool handleBoundaries = true;
    float fillValue = 0.0f;
};

/**
 * @brief GPU插值结果
 */
struct GPUInterpolationResult {
    std::vector<float> interpolatedData;
    int width;
    int height;
    
    // 性能统计
    double gpuTimeMs = 0.0;
    double memoryTransferTimeMs = 0.0;
    size_t memoryUsedBytes = 0;
    
    // 质量指标
    float minValue = 0.0f;
    float maxValue = 0.0f;
    float meanValue = 0.0f;
    int nanCount = 0;
    
    // 错误状态
    common_utils::gpu::GPUError status = common_utils::gpu::GPUError::SUCCESS;
};

/**
 * @brief GPU插值引擎接口
 */
class IGPUInterpolationEngine {
public:
    virtual ~IGPUInterpolationEngine() = default;
    
    /**
     * @brief 设置插值方法
     */
    virtual void setInterpolationMethod(InterpolationMethod method) = 0;
    
    /**
     * @brief 获取支持的插值方法
     */
    virtual std::vector<InterpolationMethod> getSupportedMethods() const = 0;
    
    /**
     * @brief 验证插值参数
     */
    virtual bool validateParams(const GPUInterpolationParams& params) const = 0;
    
    /**
     * @brief 估算插值的内存需求
     */
    virtual size_t estimateInterpolationMemory(
        int sourceWidth, int sourceHeight,
        int targetWidth, int targetHeight,
        InterpolationMethod method) const = 0;
    
    /**
     * @brief 执行GPU插值
     */
    virtual common_utils::gpu::GPUAlgorithmResult<GPUInterpolationResult> execute(
        const GPUInterpolationParams& params,
        const common_utils::gpu::GPUExecutionContext& context) = 0;
};

/**
 * @brief 批量GPU插值引擎接口
 */
class IBatchGPUInterpolationEngine : public common_utils::gpu::IBatchGPUAlgorithm<
    GPUInterpolationParams,
    GPUInterpolationResult> {
public:
    virtual ~IBatchGPUInterpolationEngine() = default;
    
    /**
     * @brief 设置批处理大小
     */
    virtual void setBatchSize(int size) = 0;
    
    /**
     * @brief 获取最优批处理大小
     */
    virtual int getOptimalBatchSize(const common_utils::gpu::GPUDeviceInfo& device) const = 0;
};

/**
 * @brief GPU插值引擎工厂
 */
class GPUInterpolationEngineFactory {
public:
    /**
     * @brief 创建GPU插值引擎
     * @param api 计算API类型
     * @return 插值引擎实例
     */
    static boost::shared_ptr<IGPUInterpolationEngine> create(
        common_utils::gpu::ComputeAPI api);
    
    /**
     * @brief 创建最优的GPU插值引擎
     * @param device GPU设备信息
     * @return 插值引擎实例
     */
    static boost::shared_ptr<IGPUInterpolationEngine> createOptimal(
        const common_utils::gpu::GPUDeviceInfo& device);
    
    /**
     * @brief 创建批量GPU插值引擎
     * @param api 计算API类型
     * @return 批量插值引擎实例
     */
    static boost::shared_ptr<IBatchGPUInterpolationEngine> createBatch(
        common_utils::gpu::ComputeAPI api);
    
    /**
     * @brief 创建优化的批量GPU插值引擎
     * @return 优化的批量插值引擎实例
     */
    static boost::shared_ptr<IBatchGPUInterpolationEngine> createOptimizedBatch();
};

} // namespace gpu
} // namespace interpolation
} // namespace core_services
} // namespace oscean 