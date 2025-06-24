#pragma once

#include "core_services/interpolation/i_interpolation_service.h"
#include "../../../include/core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"
#include "layout_aware_interpolator_base.h"
#include <vector>
#include <memory>
#include <optional>
#include <boost/shared_ptr.hpp>

namespace oscean::core_services::interpolation {

/**
 * @brief 递归N维PCHIP插值器（USML兼容版本）
 * @details 支持任意维度的递归PCHIP插值，兼容USML的data_grid接口
 *          每个维度可以独立选择插值方法（PCHIP、线性等）
 */
class RecursiveNDimPCHIPInterpolator : public IInterpolationAlgorithm,
                                      public LayoutAwareInterpolatorBase {
public:
    /**
     * @brief 每个维度的插值方法
     */
    enum class DimensionMethod {
        PCHIP,      // 分段三次Hermite插值
        LINEAR,     // 线性插值
        NEAREST     // 最近邻
    };
    
    /**
     * @brief 构造函数
     * @param simdManager SIMD管理器（可选）
     * @param methods 每个维度的插值方法（默认全部PCHIP）
     */
    RecursiveNDimPCHIPInterpolator(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr,
        const std::vector<DimensionMethod>& methods = {});
    
    ~RecursiveNDimPCHIPInterpolator() override = default;
    
    // IInterpolationAlgorithm接口实现
    InterpolationResult execute(const InterpolationRequest& request, 
                              const PrecomputedData* precomputed) const override;
    
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::PCHIP_RECURSIVE_NDIM;
    }
    
    /**
     * @brief USML兼容接口
     */
    double interpolate(
        const GridData& grid,
        const std::vector<double>& location,
        const std::vector<DimensionMethod>& methods) const;
    
    /**
     * @brief 设置维度插值方法
     */
    void setDimensionMethods(const std::vector<DimensionMethod>& methods) {
        dimensionMethods_ = methods;
    }
    
private:
    std::vector<DimensionMethod> dimensionMethods_;
    mutable std::mutex cacheMutex_;
    
    /**
     * @brief 递归插值核心函数
     * @param grid 数据网格
     * @param dim 当前处理的维度（从最高维开始）
     * @param indices 当前的索引位置
     * @param coords 目标坐标（网格坐标）
     * @param accessor 布局感知访问器
     * @return 插值结果
     */
    std::optional<double> interpRecursive(
        const GridData& grid,
        int dim,
        std::vector<size_t>& indices,
        const std::vector<double>& coords,
        const LayoutAwareAccessor& accessor) const;
    
    /**
     * @brief 1D插值（根据方法选择）
     */
    double interp1D(
        const GridData& grid,
        int dim,
        std::vector<size_t>& indices,
        double coord,
        DimensionMethod method,
        const LayoutAwareAccessor& accessor) const;
    
    /**
     * @brief 1D PCHIP插值实现
     */
    double pchip1D(
        const std::vector<double>& x,
        const std::vector<double>& y,
        double xi) const;
    
    /**
     * @brief 计算PCHIP导数
     */
    std::vector<double> computePCHIPDerivatives(
        const std::vector<double>& x,
        const std::vector<double>& y) const;
    
    /**
     * @brief 获取维度数据
     */
    void extractDimensionData(
        const GridData& grid,
        int dim,
        const std::vector<size_t>& indices,
        std::vector<double>& values,
        const LayoutAwareAccessor& accessor) const;
    
    /**
     * @brief 获取维度坐标
     */
    std::vector<double> getDimensionCoordinates(
        const GridData& grid,
        int dim) const;
    
    /**
     * @brief 查找插值区间
     */
    std::pair<size_t, size_t> findInterval(
        const std::vector<double>& coords,
        double target) const;
};

} // namespace oscean::core_services::interpolation 