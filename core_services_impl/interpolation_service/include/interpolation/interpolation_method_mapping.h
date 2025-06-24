#pragma once

#include <core_services/interpolation/i_interpolation_service.h>
#include <map>
#include <vector>
#include <algorithm>
#include <string>

namespace oscean {
namespace core_services {
namespace interpolation {

/**
 * @brief 插值方法映射工具类
 * @details 统一管理CPU和GPU支持的插值方法，确保一致性
 */
class InterpolationMethodMapping {
public:
    /**
     * @brief 检查GPU是否支持指定的插值方法
     */
    static bool isGPUSupported(InterpolationMethod method) {
        static const std::vector<InterpolationMethod> gpuSupportedMethods = {
            InterpolationMethod::BILINEAR,
            InterpolationMethod::BICUBIC,
            InterpolationMethod::TRILINEAR,
            InterpolationMethod::PCHIP_FAST_2D,
            InterpolationMethod::NEAREST_NEIGHBOR
        };
        
        return std::find(gpuSupportedMethods.begin(), 
                        gpuSupportedMethods.end(), 
                        method) != gpuSupportedMethods.end();
    }
    
    /**
     * @brief 获取所有GPU支持的插值方法
     */
    static std::vector<InterpolationMethod> getGPUSupportedMethods() {
        return {
            InterpolationMethod::BILINEAR,
            InterpolationMethod::BICUBIC,
            InterpolationMethod::TRILINEAR,
            InterpolationMethod::PCHIP_FAST_2D,
            InterpolationMethod::NEAREST_NEIGHBOR
        };
    }
    
    /**
     * @brief 获取所有CPU支持的插值方法
     */
    static std::vector<InterpolationMethod> getCPUSupportedMethods() {
        return {
            InterpolationMethod::LINEAR_1D,
            InterpolationMethod::CUBIC_SPLINE_1D,
            InterpolationMethod::NEAREST_NEIGHBOR,
            InterpolationMethod::BILINEAR,
            InterpolationMethod::BICUBIC,
            InterpolationMethod::TRILINEAR,
            InterpolationMethod::TRICUBIC,
            InterpolationMethod::PCHIP_RECURSIVE_NDIM,
            InterpolationMethod::PCHIP_MULTIGRID_NDIM,
            InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY,
            InterpolationMethod::PCHIP_OPTIMIZED_3D_SVP,
            InterpolationMethod::PCHIP_FAST_2D,
            InterpolationMethod::PCHIP_FAST_3D,
            InterpolationMethod::COMPLEX_FIELD_BILINEAR,
            InterpolationMethod::COMPLEX_FIELD_BICUBIC,
            InterpolationMethod::COMPLEX_FIELD_TRILINEAR,
            InterpolationMethod::COMPLEX_FIELD_PCHIP
        };
    }
    
    /**
     * @brief 获取插值方法的字符串表示
     */
    static std::string toString(InterpolationMethod method) {
        static const std::map<InterpolationMethod, std::string> methodNames = {
            {InterpolationMethod::UNKNOWN, "UNKNOWN"},
            {InterpolationMethod::LINEAR_1D, "LINEAR_1D"},
            {InterpolationMethod::CUBIC_SPLINE_1D, "CUBIC_SPLINE_1D"},
            {InterpolationMethod::NEAREST_NEIGHBOR, "NEAREST_NEIGHBOR"},
            {InterpolationMethod::BILINEAR, "BILINEAR"},
            {InterpolationMethod::BICUBIC, "BICUBIC"},
            {InterpolationMethod::TRILINEAR, "TRILINEAR"},
            {InterpolationMethod::TRICUBIC, "TRICUBIC"},
            {InterpolationMethod::PCHIP_RECURSIVE_NDIM, "PCHIP_RECURSIVE_NDIM"},
            {InterpolationMethod::PCHIP_MULTIGRID_NDIM, "PCHIP_MULTIGRID_NDIM"},
            {InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY, "PCHIP_OPTIMIZED_2D_BATHY"},
            {InterpolationMethod::PCHIP_OPTIMIZED_3D_SVP, "PCHIP_OPTIMIZED_3D_SVP"},
            {InterpolationMethod::PCHIP_FAST_2D, "PCHIP_FAST_2D"},
            {InterpolationMethod::PCHIP_FAST_3D, "PCHIP_FAST_3D"},
            {InterpolationMethod::COMPLEX_FIELD_BILINEAR, "COMPLEX_FIELD_BILINEAR"},
            {InterpolationMethod::COMPLEX_FIELD_BICUBIC, "COMPLEX_FIELD_BICUBIC"},
            {InterpolationMethod::COMPLEX_FIELD_TRILINEAR, "COMPLEX_FIELD_TRILINEAR"},
            {InterpolationMethod::COMPLEX_FIELD_PCHIP, "COMPLEX_FIELD_PCHIP"}
        };
        
        auto it = methodNames.find(method);
        return (it != methodNames.end()) ? it->second : "UNKNOWN";
    }
    
    /**
     * @brief 从字符串解析插值方法
     */
    static InterpolationMethod fromString(const std::string& name) {
        static const std::map<std::string, InterpolationMethod> nameToMethod = {
            {"UNKNOWN", InterpolationMethod::UNKNOWN},
            {"LINEAR_1D", InterpolationMethod::LINEAR_1D},
            {"CUBIC_SPLINE_1D", InterpolationMethod::CUBIC_SPLINE_1D},
            {"NEAREST_NEIGHBOR", InterpolationMethod::NEAREST_NEIGHBOR},
            {"BILINEAR", InterpolationMethod::BILINEAR},
            {"BICUBIC", InterpolationMethod::BICUBIC},
            {"TRILINEAR", InterpolationMethod::TRILINEAR},
            {"TRICUBIC", InterpolationMethod::TRICUBIC},
            {"PCHIP_RECURSIVE_NDIM", InterpolationMethod::PCHIP_RECURSIVE_NDIM},
            {"PCHIP_MULTIGRID_NDIM", InterpolationMethod::PCHIP_MULTIGRID_NDIM},
            {"PCHIP_OPTIMIZED_2D_BATHY", InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY},
            {"PCHIP_OPTIMIZED_3D_SVP", InterpolationMethod::PCHIP_OPTIMIZED_3D_SVP},
            {"PCHIP_FAST_2D", InterpolationMethod::PCHIP_FAST_2D},
            {"PCHIP_FAST_3D", InterpolationMethod::PCHIP_FAST_3D},
            {"COMPLEX_FIELD_BILINEAR", InterpolationMethod::COMPLEX_FIELD_BILINEAR},
            {"COMPLEX_FIELD_BICUBIC", InterpolationMethod::COMPLEX_FIELD_BICUBIC},
            {"COMPLEX_FIELD_TRILINEAR", InterpolationMethod::COMPLEX_FIELD_TRILINEAR},
            {"COMPLEX_FIELD_PCHIP", InterpolationMethod::COMPLEX_FIELD_PCHIP}
        };
        
        auto it = nameToMethod.find(name);
        return (it != nameToMethod.end()) ? it->second : InterpolationMethod::UNKNOWN;
    }
};

} // namespace interpolation
} // namespace core_services
} // namespace oscean 