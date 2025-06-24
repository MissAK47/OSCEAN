#pragma once

#include <core_services/interpolation/i_interpolation_service.h>
#include <memory>

namespace oscean::core_services::interpolation {

/**
 * @brief 插值算法接口
 */
class IInterpolationAlgorithm {
public:
    virtual ~IInterpolationAlgorithm() = default;
    
    /**
     * @brief 执行插值
     */
    virtual InterpolationResult interpolate(const InterpolationRequest& request) = 0;
    
    /**
     * @brief 获取算法名称
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief 是否支持GPU加速
     */
    virtual bool supportsGPU() const { return false; }
};

/**
 * @brief 插值算法工厂
 */
class InterpolationAlgorithmFactory {
public:
    /**
     * @brief 创建插值算法实例
     * @param method 插值方法
     * @return 算法实例
     */
    static std::unique_ptr<IInterpolationAlgorithm> create(InterpolationMethod method);
};

} // namespace oscean::core_services::interpolation 