#pragma once

#include "../../include/core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"
#include "common_utils/simd/isimd_manager.h"
#include <memory>
#include <boost/shared_ptr.hpp>

namespace oscean::core_services::interpolation {

/**
 * @class PCHIPInterpolator2DBathy
 * @brief 针对2D水深数据优化的PCHIP插值算法。
 * @details
 * 该算法专门为处理水深（Bathymetry）数据而设计。相比于通用的PCHIP实现，
 * 它可能在以下方面进行了优化：
 * - **性能**: 采用非递归、迭代的方式计算，减少函数调用开销。
 * - **内存**: 优化内存访问模式，提高缓存命中率。
 * - **精度**: 可能针对水深数据的特点（大片平缓区域与局部陡峭变化）调整斜率计算，
 *           以在保持形状的同时提高保真度。
 * 这是一个高性能的"即时"插值器，不需要像FastPCHIP那样进行全局预计算。
 */
class PCHIPInterpolator2DBathy : public IInterpolationAlgorithm {
public:
    /**
     * @brief 构造函数
     * @param simdManager SIMD管理器（可选）
     */
    explicit PCHIPInterpolator2DBathy(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr
    );

    InterpolationResult execute(
        const InterpolationRequest& request,
        const PrecomputedData* precomputed
    ) const override;
    
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY;
    }
    
    // 此算法不使用预计算
    std::shared_ptr<PrecomputedData> precompute(
        std::shared_ptr<const GridData> sourceGrid
    ) const override {
        return nullptr;
    }

private:
    double pchipSlope(double h1, double h2, double m1, double m2) const;
    double interpolateCubic(double x, double x0, double x1, 
                            double y0, double y1, 
                            double d0, double d1) const;

    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;
};

} // namespace oscean::core_services::interpolation 