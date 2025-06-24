#pragma once

// 🚀 使用Common模块的统一boost配置（参考CRS服务）
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 插值算法只使用boost::future，不使用boost::asio

// 直接使用接口中的类型定义
#include "core_services/interpolation/i_interpolation_service.h"
#include "../../../include/core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"
#include "layout_aware_interpolator_base.h"
#include "common_utils/simd/isimd_manager.h"
#include <vector>
#include <optional>
#include <memory>
#include <functional>
#include <mutex>
#include <map>
#include <numeric>
#include <complex>

namespace oscean::core_services::interpolation {

/**
 * @brief PCHIP插值算法实现 (分段三次Hermite插值)
 * @details 采用与USML库类似的递归、动态计算方法，实现N维PCHIP插值。
 *          此实现不预先计算全局导数，而是在插值时动态计算，保证了算法的准确性。
 *          支持内存布局感知和SIMD优化。
 */
class PCHIPInterpolator : public IInterpolationAlgorithm, 
                         public LayoutAwareInterpolatorBase {
public:
    /**
     * @brief 构造函数
     * @param simdManager SIMD管理器
     */
    explicit PCHIPInterpolator(boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager);
    
    ~PCHIPInterpolator() override = default;

    // IInterpolationAlgorithm接口实现
    InterpolationResult execute(const InterpolationRequest& request, const PrecomputedData* precomputed) const override;
    
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::PCHIP_RECURSIVE_NDIM;
    }

    /**
     * @brief 预计算导数（可选的性能优化）
     */
    struct PCHIPDerivatives {
        std::vector<std::vector<double>> firstDerivatives;  // 各维度的一阶导数
        std::vector<std::vector<double>> crossDerivatives;  // 交叉导数
        bool isComputed = false;
        
        // 内存布局信息
        GridData::MemoryLayout layout = GridData::MemoryLayout::ROW_MAJOR;
        std::vector<size_t> dimensions;
    };
    
    /**
     * @brief 预计算导数以加速后续插值
     */
    std::unique_ptr<PCHIPDerivatives> precomputeDerivatives(const GridData& grid) const;

    /**
     * @brief 支持复数插值（为RAM模块）
     */
    std::complex<double> interpolateComplex(
        const GridData& realGrid,
        const GridData& imagGrid,
        const std::vector<double>& worldCoords) const;

    /**
     * @brief 静态辅助函数：计算PCHIP导数
     */
    static std::vector<double> computePCHIPDerivatives(
        const std::vector<double>& x,
        const std::vector<double>& y);

private:
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;

    // 布局感知的递归插值函数
    std::optional<double> interpRecursive(
        const GridData& grid,
        int dim,
        std::vector<size_t>& indices,
        const std::vector<double>& gridCoords,
        const LayoutAwareAccessor& accessor
    ) const;
    
    // 布局优化的1D PCHIP核心算法
    double pchip1D(
        const GridData& grid,
        int dim,
        std::vector<size_t>& indices,
        const std::vector<double>& gridCoords,
        const LayoutAwareAccessor& accessor
    ) const;
    
    // SIMD优化的批量PCHIP计算
    void pchip1DSIMD(
        const float* data,
        const float* coords,
        float* results,
        size_t numPoints,
        size_t dataSize
    ) const;

    // 公共入口点
    std::optional<double> interpolateAtPoint(const GridData& grid, const std::vector<double>& worldCoords) const;
    std::vector<std::optional<double>> batchInterpolate(const GridData& grid, const std::vector<TargetPoint>& points) const;
    std::vector<std::optional<double>> batchInterpolateSIMD(const GridData& grid, const std::vector<TargetPoint>& points) const;
    GridData interpolateToGrid(const GridData& sourceGrid, const TargetGridDefinition& targetGridDef) const;
    
    // 布局感知的辅助函数
    std::optional<double> getGridValue(
        const GridData& grid, 
        const std::vector<size_t>& indices, 
        size_t band,
        const LayoutAwareAccessor& accessor
    ) const;
    
    // 计算PCHIP导数
    double computePCHIPDerivative(
        double h1, double h2,
        double delta1, double delta2
    ) const;
    
    // 评估Hermite多项式
    double evaluateHermite(
        double x0, double x1,
        double y0, double y1,
        double d0, double d1,
        double x
    ) const;
    
    // 缓存管理
    mutable std::mutex cacheMutex_;
    mutable std::map<size_t, std::unique_ptr<PCHIPDerivatives>> derivativesCache_;
    
    // 获取或计算导数
    const PCHIPDerivatives* getOrComputeDerivatives(const GridData& grid) const;
};

/**
 * @brief 复数PCHIP插值器（专门为RAM优化）
 */
class ComplexPCHIPInterpolator : public PCHIPInterpolator {
public:
    explicit ComplexPCHIPInterpolator(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
        : PCHIPInterpolator(simdManager) {}
    
    /**
     * @brief 执行复数插值
     */
    std::vector<std::complex<double>> interpolateComplexBatch(
        const GridData& complexGrid,  // 复数数据网格
        const std::vector<TargetPoint>& points
    ) const;
    
    /**
     * @brief 从实部和虚部网格插值
     */
    std::vector<std::complex<double>> interpolateFromRealImag(
        const GridData& realGrid,
        const GridData& imagGrid,
        const std::vector<TargetPoint>& points
    ) const;
};

} // namespace oscean::core_services::interpolation 