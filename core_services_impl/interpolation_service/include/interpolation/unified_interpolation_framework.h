#pragma once

#include <core_services/interpolation/i_interpolation_service.h>
#include <core_services/common_data_types.h>
#include <array>
#include <vector>
#include <memory>
#include <boost/optional.hpp>

namespace oscean {
namespace core_services {
namespace interpolation {

/**
 * @brief 统一的多维插值框架
 * @details 基于USML设计理念，支持任意维度的递归插值
 */
template<size_t NUM_DIMS>
class UnifiedInterpolationFramework {
public:
    /**
     * @brief 预计算的导数信息
     */
    struct PrecomputedDerivatives {
        std::array<std::vector<double>, NUM_DIMS> firstDerivatives;
        std::array<std::vector<double>, NUM_DIMS*(NUM_DIMS-1)/2> crossDerivatives;
        bool isComputed = false;
    };

    /**
     * @brief 插值配置
     */
    struct InterpolationConfig {
        InterpolationMethod method = InterpolationMethod::BILINEAR;
        bool enablePrecomputation = true;
        bool enableGPUAcceleration = true;
        size_t cacheSize = 1000;
    };

    /**
     * @brief 多维插值接口
     */
    template<typename DATA_TYPE>
    class IMultiDimInterpolator {
    public:
        virtual ~IMultiDimInterpolator() = default;
        
        /**
         * @brief 递归多维插值
         * @param coordinates 各维度坐标
         * @return 插值结果
         */
        virtual boost::optional<DATA_TYPE> interpolate(
            const std::array<double, NUM_DIMS>& coordinates) const = 0;
        
        /**
         * @brief 批量插值
         */
        virtual std::vector<boost::optional<DATA_TYPE>> interpolateBatch(
            const std::vector<std::array<double, NUM_DIMS>>& coordinatesBatch) const = 0;
        
        /**
         * @brief 预计算导数
         */
        virtual void precomputeDerivatives() = 0;
    };

    /**
     * @brief PCHIP多维插值器
     */
    template<typename DATA_TYPE>
    class PCHIPMultiDimInterpolator : public IMultiDimInterpolator<DATA_TYPE> {
    private:
        std::shared_ptr<GridData> sourceData_;
        PrecomputedDerivatives derivatives_;
        InterpolationConfig config_;
        
        // 递归插值实现
        template<size_t DIM>
        DATA_TYPE interpolateRecursive(
            const std::array<double, NUM_DIMS>& coords,
            const std::array<size_t, NUM_DIMS>& indices) const;
        
    public:
        explicit PCHIPMultiDimInterpolator(
            std::shared_ptr<GridData> data,
            const InterpolationConfig& config = {});
        
        boost::optional<DATA_TYPE> interpolate(
            const std::array<double, NUM_DIMS>& coordinates) const override;
        
        std::vector<boost::optional<DATA_TYPE>> interpolateBatch(
            const std::vector<std::array<double, NUM_DIMS>>& coordinatesBatch) const override;
        
        void precomputeDerivatives() override;
    };

    /**
     * @brief 工厂方法
     */
    template<typename DATA_TYPE>
    static std::unique_ptr<IMultiDimInterpolator<DATA_TYPE>> createInterpolator(
        InterpolationMethod method,
        std::shared_ptr<GridData> data,
        const InterpolationConfig& config = {});
};

/**
 * @brief 2D特化版本（向后兼容）
 */
template<>
class UnifiedInterpolationFramework<2> {
public:
    // 2D特化实现，与现有代码兼容
    using Point2D = std::array<double, 2>;
    
    template<typename DATA_TYPE>
    class Interpolator2D : public UnifiedInterpolationFramework<2>::IMultiDimInterpolator<DATA_TYPE> {
        // 实现细节...
    };
};

/**
 * @brief 3D特化版本
 */
template<>
class UnifiedInterpolationFramework<3> {
public:
    // 3D特化实现
    using Point3D = std::array<double, 3>;
    
    template<typename DATA_TYPE>
    class Interpolator3D : public UnifiedInterpolationFramework<3>::IMultiDimInterpolator<DATA_TYPE> {
        // 实现细节...
    };
};

} // namespace interpolation
} // namespace core_services
} // namespace oscean 