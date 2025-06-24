#pragma once

#include "core_services/interpolation/i_interpolation_service.h"
#include "../../../include/core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"
#include "layout_aware_interpolator_base.h"
#include <boost/smart_ptr/shared_ptr.hpp>
#include <complex>
#include <vector>
#include <memory>

namespace oscean::core_services::interpolation {

/**
 * @brief 复数场插值器（为RAM声场数据优化）
 * @details 支持复数值网格数据的插值，实部和虚部分别插值
 *          特别优化了声场传播数据的处理
 */
class ComplexFieldInterpolator : public IInterpolationAlgorithm,
                                public LayoutAwareInterpolatorBase {
public:
    /**
     * @brief 构造函数
     * @param simdManager SIMD管理器（可选）
     * @param baseMethod 基础插值方法（默认双线性）
     */
    explicit ComplexFieldInterpolator(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr,
        InterpolationMethod baseMethod = InterpolationMethod::BILINEAR);
    
    ~ComplexFieldInterpolator() override = default;
    
    // IInterpolationAlgorithm接口实现
    InterpolationResult execute(const InterpolationRequest& request,
                              const PrecomputedData* precomputed) const override;
    
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::BILINEAR;  // 复数场默认使用双线性
    }
    
    /**
     * @brief 插值单个复数值
     */
    std::complex<double> interpolateComplex(
        const GridData& realGrid,
        const GridData& imagGrid,
        double x, double y, double z = 0) const;
    
    /**
     * @brief 批量复数插值
     */
    std::vector<std::complex<double>> interpolateComplexBatch(
        const GridData& realGrid,
        const GridData& imagGrid,
        const std::vector<TargetPoint>& points) const;
    
    /**
     * @brief 从交错存储的复数数据插值
     * @param complexGrid 复数数据，格式为[real0, imag0, real1, imag1, ...]
     */
    std::complex<double> interpolateInterleavedComplex(
        const GridData& complexGrid,
        double x, double y, double z = 0) const;
    
    /**
     * @brief 设置基础插值方法
     */
    void setBaseMethod(InterpolationMethod method) {
        baseMethod_ = method;
    }
    
private:
    InterpolationMethod baseMethod_;
    
    /**
     * @brief SIMD优化的批量复数插值
     */
    std::vector<std::complex<double>> interpolateComplexBatchSIMD(
        const GridData& realGrid,
        const GridData& imagGrid,
        const std::vector<TargetPoint>& points) const;
    
    /**
     * @brief SIMD优化的2D值插值
     */
    double interpolate2DValueSIMD(
        const GridData& grid,
        double gridX, double gridY,
        size_t band) const;
    
    /**
     * @brief 执行2D复数场插值
     */
    std::complex<double> interpolate2DComplex(
        const GridData& realGrid,
        const GridData& imagGrid,
        double x, double y,
        const LayoutAwareAccessor& realAccessor,
        const LayoutAwareAccessor& imagAccessor) const;
    
    /**
     * @brief 执行3D复数场插值
     */
    std::complex<double> interpolate3DComplex(
        const GridData& realGrid,
        const GridData& imagGrid,
        double x, double y, double z,
        const LayoutAwareAccessor& realAccessor,
        const LayoutAwareAccessor& imagAccessor) const;
    
    /**
     * @brief 双线性插值核心实现
     */
    template<typename T>
    T bilinearInterpolate(
        T v00, T v10, T v01, T v11,
        double fx, double fy) const {
        T v0 = v00 * (1 - fx) + v10 * fx;
        T v1 = v01 * (1 - fx) + v11 * fx;
        return v0 * (1 - fy) + v1 * fy;
    }
    
    /**
     * @brief PCHIP插值（用于高精度需求）
     */
    double pchipInterpolate2D(
        const GridData& grid,
        double x, double y,
        const LayoutAwareAccessor& accessor) const;
    
    /**
     * @brief 2D插值辅助方法
     */
    double interpolate2DValue(
        const GridData& grid,
        double gridX, double gridY,
        size_t band,
        const LayoutAwareAccessor& accessor) const;
};

/**
 * @brief RAM声场数据适配器
 * @details 将RAM的声场格式转换为OSCEAN的GridData格式
 */
class RAMFieldAdapter {
public:
    /**
     * @brief 从RAM声场数据创建GridData
     * @param pressureField 声压场数据（复数）
     * @param ranges 距离坐标（米）
     * @param depths 深度坐标（米）
     * @return 包含实部和虚部的GridData对
     */
    static std::pair<std::shared_ptr<GridData>, std::shared_ptr<GridData>>
    createFromRAMField(
        const std::vector<std::complex<double>>& pressureField,
        const std::vector<double>& ranges,
        const std::vector<double>& depths);
    
    /**
     * @brief 创建交错存储的复数GridData
     */
    static std::shared_ptr<GridData> createInterleavedComplexGrid(
        const std::vector<std::complex<double>>& pressureField,
        const std::vector<double>& ranges,
        const std::vector<double>& depths);
};

} // namespace oscean::core_services::interpolation 