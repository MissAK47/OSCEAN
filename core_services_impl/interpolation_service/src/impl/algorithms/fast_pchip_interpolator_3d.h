#pragma once

#include "core_services/interpolation/i_interpolation_service.h"
#include "core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"
#include "common_utils/simd/isimd_manager.h"
#include <vector>
#include <memory>
#include <boost/shared_ptr.hpp>

namespace oscean::core_services::interpolation {

/**
 * @brief 高性能、非递归的3D PCHIP插值器
 * @details 专为3D网格设计，通过预计算整个网格的导数来加速插值。
 *          特别优化了声速剖面等深度优先的数据访问模式。
 */
class FastPchipInterpolator3D : public IInterpolationAlgorithm {
public:
    /**
     * @brief 构造函数，执行所有预计算
     * @param sourceGrid 用于插值的源3D网格数据
     * @param simdManager (可选) SIMD管理器
     */
    explicit FastPchipInterpolator3D(
        const std::shared_ptr<const GridData>& sourceGrid,
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr
    );
    
    ~FastPchipInterpolator3D() override = default;

    /**
     * @brief 执行插值计算
     * @param request 插值请求，包含目标点等信息
     * @param precomputed 预计算数据（在此实现中未使用）
     * @return 插值结果
     */
    InterpolationResult execute(const InterpolationRequest& request, const PrecomputedData* precomputed) const override;
    
    /**
     * @brief 获取此算法的类型
     * @return InterpolationMethod 枚举值
     */
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::PCHIP_FAST_3D;
    }
    
    /**
     * @brief 预计算数据接口（此实现不使用）
     * @param sourceGrid 要为其预计算数据的源网格
     * @return 预计算结果的共享指针（此处为空）
     */
    std::shared_ptr<PrecomputedData> precompute(
        std::shared_ptr<const GridData> sourceGrid
    ) const override {
        return nullptr;
    }

private:
    /**
     * @brief 内部非递归的3D PCHIP插值函数
     */
    double fastPchip3D(const double location[3], double* derivative) const;

    /**
     * @brief 从3D网格安全地获取一个值
     */
    double getGridValue(size_t x, size_t y, size_t z) const;
    
    /**
     * @brief 内联SIMD优化的批量插值
     * @param targetPoints 目标点集合
     * @return 插值结果
     */
    std::vector<std::optional<double>> interpolateAtPointsSIMD(
        const std::vector<TargetPoint>& targetPoints) const;
    
    /**
     * @brief SIMD优化的3D插值核心计算
     * @param gridX 网格X坐标
     * @param gridY 网格Y坐标  
     * @param gridZ 网格Z坐标
     * @return 插值结果
     */
    double interpolate3DSIMD(double gridX, double gridY, double gridZ) const;

    std::shared_ptr<const GridData> sourceGrid_;
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;

    // 预计算数据
    std::vector<double> dervX_;      // X方向的一阶导数
    std::vector<double> dervY_;      // Y方向的一阶导数
    std::vector<double> dervZ_;      // Z方向的一阶导数
    std::vector<double> dervXY_;     // XY方向的混合导数
    std::vector<double> dervXZ_;     // XZ方向的混合导数
    std::vector<double> dervYZ_;     // YZ方向的混合导数
    std::vector<double> dervXYZ_;    // XYZ方向的混合导数

    // 网格尺寸
    size_t dimX_{0};
    size_t dimY_{0};
    size_t dimZ_{0};
    
    /**
     * @brief 预计算所有导数
     */
    void computeDerivatives();
    
    /**
     * @brief 执行3D插值
     */
    double interpolate3D(double gridX, double gridY, double gridZ) const;
};

} // namespace oscean::core_services::interpolation 