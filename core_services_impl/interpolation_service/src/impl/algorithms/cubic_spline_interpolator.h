#pragma once

// 🚀 使用Common模块的统一boost配置（参考CRS服务）
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 插值算法只使用boost::future，不使用boost::asio

// 直接使用接口中的类型定义
#include "core_services/interpolation/i_interpolation_service.h"
#include "../../../include/core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"
#include "common_utils/simd/isimd_manager.h"
#include <vector>
#include <optional>
#include <memory>

namespace oscean::core_services::interpolation {

/**
 * @brief 立方样条插值算法实现
 * @details 支持SIMD加速的高精度立方样条插值算法
 */
class CubicSplineInterpolator : public IInterpolationAlgorithm {
public:
    /**
     * @brief 构造函数
     * @param simdManager SIMD管理器（可选）
     */
    explicit CubicSplineInterpolator(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr);
    
    ~CubicSplineInterpolator() override = default;

    // IInterpolationAlgorithm接口实现
    InterpolationResult execute(
        const InterpolationRequest& request,
        const PrecomputedData* precomputed = nullptr
    ) const override;
    
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::CUBIC_SPLINE_1D;
    }

private:
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;

    /**
     * @brief 在目标点集合执行插值
     * @param sourceGrid 源网格数据
     * @param targetPoints 目标点集合
     * @return 插值结果
     */
    std::vector<std::optional<double>> interpolateAtPoints(
        const GridData& sourceGrid,
        const std::vector<TargetPoint>& targetPoints) const;

    /**
     * @brief 网格到网格插值
     * @param sourceGrid 源网格数据
     * @param targetGridDef 目标网格定义
     * @return 插值后的网格数据
     */
    GridData interpolateToGrid(
        const GridData& sourceGrid,
        const TargetGridDefinition& targetGridDef) const;

    /**
     * @brief 在单个点执行立方样条插值
     * @param grid 源网格数据
     * @param worldX 世界坐标X
     * @param worldY 世界坐标Y
     * @return 插值结果
     */
    std::optional<double> interpolateAtPoint(
        const GridData& grid, 
        double worldX, 
        double worldY) const;

    /**
     * @brief 获取网格数据值（安全访问）
     * @param grid 源网格数据
     * @param col 列索引
     * @param row 行索引
     * @param band 波段索引
     * @return 数据值
     */
    std::optional<double> getGridValue(
        const GridData& grid,
        int col, int row, int band = 0) const;

    /**
     * @brief 边界区域的双线性插值回退
     * @param grid 源网格数据
     * @param gridX 网格X坐标
     * @param gridY 网格Y坐标
     * @return 插值结果
     */
    std::optional<double> fallbackBilinearInterpolation(
        const GridData& grid, 
        double gridX, 
        double gridY) const;

    /**
     * @brief SIMD优化的批量插值
     * @param grid 源网格数据
     * @param points 目标点集合
     * @return 插值结果
     */
    std::vector<std::optional<double>> simdBatchInterpolate(
        const GridData& grid,
        const std::vector<TargetPoint>& points) const;
        
    /**
     * @brief 内联SIMD优化的批量插值（AVX2/AVX512）
     * @param sourceGrid 源网格数据
     * @param targetPoints 目标点集合
     * @return 插值结果
     */
    std::vector<std::optional<double>> interpolateAtPointsSIMD(
        const GridData& sourceGrid,
        const std::vector<TargetPoint>& targetPoints) const;
        
    /**
     * @brief SIMD优化的双三次插值核心计算
     * @param values 16个控制点的值
     * @param fx X方向的小数部分
     * @param fy Y方向的小数部分
     * @return 插值结果
     */
    double bicubicInterpolateSIMD(
        const float values[16], float fx, float fy) const;
        
#ifdef __AVX2__
    /**
     * @brief SIMD计算立方插值权重
     * @param t 插值参数
     * @return 4个权重值
     */
    __m128 computeCubicWeightsSIMD(__m128 t) const;
#endif
};

} // namespace oscean::core_services::interpolation 