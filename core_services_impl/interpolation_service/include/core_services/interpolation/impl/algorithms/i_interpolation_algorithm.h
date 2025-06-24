#pragma once

// 直接使用接口中的类型定义
#include "core_services/interpolation/i_interpolation_service.h"
#include <string>
#include <memory>

namespace oscean::core_services::interpolation {

// 预计算数据，用于优化
struct PrecomputedData {};

/**
 * @brief 插值算法策略接口
 * @details 与core_service_interfaces标准一致的算法接口
 */
class IInterpolationAlgorithm {
public:
    virtual ~IInterpolationAlgorithm() = default;

    /**
     * @brief 执行插值计算
     * @param request 插值请求
     * @param precomputed 可选的预计算数据
     * @return 插值结果
     */
    virtual InterpolationResult execute(
        const InterpolationRequest& request,
        const PrecomputedData* precomputed = nullptr
    ) const = 0;
    
    /**
     * @brief 获取算法支持的插值方法
     */
    virtual InterpolationMethod getMethodType() const = 0;

    /**
     * @brief 预计算步骤（可选）
     * @param sourceGrid 源网格数据
     * @return 预计算结果的共享指针
     */
    virtual std::shared_ptr<PrecomputedData> precompute(
        std::shared_ptr<const GridData> sourceGrid
    ) const {
        // 默认实现为空，因为并非所有算法都需要预计算
        return nullptr;
    }
};

} // namespace oscean::core_services::interpolation 