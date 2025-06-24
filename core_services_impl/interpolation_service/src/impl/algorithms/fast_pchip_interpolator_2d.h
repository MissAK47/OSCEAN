#pragma once

#include "core_services/interpolation/i_interpolation_service.h"
#include "core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"
#include "common_utils/simd/isimd_manager.h"
#include <vector>
#include <memory>
#include <boost/shared_ptr.hpp>

namespace oscean::core_services::interpolation {

/**
 * @brief 高性能、非递归的2D PCHIP插值器
 * @details 专为2D网格设计，通过在构造时进行大量的预计算（整个网格的导数），
 *          换取后续插值操作的极高速度。其思想源于USML库中的data_grid_bathy实现。
 *          这是一种典型的空间换时间优化策略。
 */
class FastPchipInterpolator2D : public IInterpolationAlgorithm {
public:
    /**
     * @brief 构造函数，执行所有预计算
     * @param sourceGrid 用于插值的源2D网格数据。
     * @param simdManager (可选) SIMD管理器。
     */
    explicit FastPchipInterpolator2D(
        const std::shared_ptr<const GridData>& sourceGrid,
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr
    );
    
    ~FastPchipInterpolator2D() override = default;

    // IInterpolationAlgorithm接口实现
    InterpolationResult execute(const InterpolationRequest& request, const PrecomputedData* precomputed) const override;
    
    InterpolationMethod getMethodType() const override {
        // 注意：这里我们可能需要一个新的方法类型来区分
        // 或者在服务层进行选择。暂时返回一个现有的。
        return InterpolationMethod::PCHIP_RECURSIVE_NDIM; 
    }

private:
    /**
     * @brief 内部非递归的PCHIP插值函数
     */
    double fastPchip(const double location[2], double* derivative) const;

    /**
     * @brief 从网格安全地获取一个值
     */
    double getGridValue(size_t col, size_t row) const;
    
    /**
     * @brief 内联SIMD优化的批量插值
     * @param targetPoints 目标点集合
     * @return 插值结果
     */
    std::vector<std::optional<double>> interpolateAtPointsSIMD(
        const std::vector<TargetPoint>& targetPoints) const;
    
    /**
     * @brief SIMD优化的PCHIP核心计算
     * @param gridX 网格X坐标
     * @param gridY 网格Y坐标
     * @param c0 列索引
     * @param r0 行索引
     * @return 插值结果
     */
    double fastPchipSIMD(double gridX, double gridY, size_t c0, size_t r0) const;

    std::shared_ptr<const GridData> sourceGrid_;
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;

    // 预计算数据
    std::vector<double> dervX_;      // X方向的一阶导数
    std::vector<double> dervY_;      // Y方向的一阶导数
    std::vector<double> dervXY_;     // XY方向的混合导数

    // 网格尺寸
    size_t cols_{0};
    size_t rows_{0};
};

} // namespace oscean::core_services::interpolation 