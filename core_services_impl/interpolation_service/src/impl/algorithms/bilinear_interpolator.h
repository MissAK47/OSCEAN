#pragma once

// 🚀 使用Common模块的统一boost配置（参考CRS服务）
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 插值算法只使用boost::future，不使用boost::asio

// 直接使用接口中的类型定义
#include "core_services/interpolation/i_interpolation_service.h"
#include "../../../include/core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"
#include "common_utils/simd/isimd_manager.h"
#include <boost/smart_ptr/shared_ptr.hpp>
#include <vector>
#include <optional>
#include <memory>

namespace oscean::core_services::interpolation {

/**
 * @brief 双线性插值算法实现
 * @details 与标准接口兼容的算法实现，支持SIMD加速
 */
class BilinearInterpolator : public IInterpolationAlgorithm {
public:
    /**
     * @brief 构造函数
     * @param simdManager SIMD管理器（可选）
     */
    explicit BilinearInterpolator(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr);
     
    ~BilinearInterpolator() override = default;

    // IInterpolationAlgorithm接口实现
    InterpolationResult execute(
        const InterpolationRequest& request,
        const PrecomputedData* precomputed = nullptr
    ) const override;
    
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::BILINEAR;
    }

    /**
     * @brief 内联SIMD优化的批量插值
     * @param sourceGrid 源网格数据
     * @param targetPoints 目标点列表
     * @return 插值结果列表
     * @details 直接使用CPU指令集，避免函数调用开销
     */
    std::vector<std::optional<double>> interpolateAtPointsSIMD(
        const GridData& sourceGrid,
        const std::vector<TargetPoint>& targetPoints) const;

    /**
     * @brief SIMD优化的批量插值
     * @param grid 源网格数据
     * @param points 目标点列表
     * @return 插值结果列表
     */
    std::vector<std::optional<double>> simdBatchInterpolate(
        const GridData& grid,
        const std::vector<TargetPoint>& points) const;

    /**
     * @brief AVX优化的批量插值
     * @param grid 源网格数据
     * @param points 目标点列表
     * @param results 输出结果数组
     */
    void batchInterpolateSIMD(
        const GridData& grid,
        const std::vector<TargetPoint>& points,
        double* results) const;

#ifdef __AVX512F__
    /**
     * @brief AVX-512优化的批量插值
     * @param grid 源网格数据
     * @param points 目标点列表
     * @param results 输出结果数组
     */
    void batchInterpolateAVX512(
        const GridData& grid,
        const std::vector<TargetPoint>& points,
        double* results) const;
#endif

    /**
     * @brief 在目标点集合执行插值
     * @param sourceGrid 源网格数据
     * @param targetPoints 目标点集合
     * @return 插值结果向量
     */
    std::vector<std::optional<double>> interpolateAtPoints(
        const GridData& sourceGrid,
        const std::vector<TargetPoint>& targetPoints
    ) const;

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
     * @brief 在单个目标点执行插值
     * @param grid 源网格数据
     * @param worldX 世界坐标X
     * @param worldY 世界坐标Y
     * @return 插值结果，如果失败返回nullopt
     */
    std::optional<double> interpolateAtPoint(
        const GridData& grid, 
        double worldX, 
        double worldY
    ) const;

private:
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;

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
};

} // namespace oscean::core_services::interpolation 