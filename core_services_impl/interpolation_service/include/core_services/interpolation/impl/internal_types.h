#pragma once

#include "core_services/interpolation/i_interpolation_service.h"
#include <map>
#include <any>

namespace oscean::core_services::interpolation {

// 使用接口中定义的类型
using oscean::core_services::interpolation::InterpolationMethod;
using oscean::core_services::interpolation::TargetPoint;
using oscean::core_services::interpolation::TargetGridDefinition;
using oscean::core_services::interpolation::InterpolationRequest;
using oscean::core_services::interpolation::InterpolationResult;
using oscean::core_services::interpolation::InterpolationResultData;
using oscean::core_services::interpolation::AlgorithmParameters;

/**
 * @brief 网格坐标结构体（内部实现使用）
 * @details 表示网格中的一个坐标点
 */
struct GridCoordinate {
    double x = 0.0;  ///< X坐标
    double y = 0.0;  ///< Y坐标
    std::optional<double> z = std::nullopt;  ///< Z坐标（可选）
    std::optional<double> t = std::nullopt;  ///< 时间坐标（可选）
    
    /**
     * @brief 构造函数
     */
    GridCoordinate() = default;
    
    /**
     * @brief 参数构造函数
     */
    GridCoordinate(double x_val, double y_val, 
                   std::optional<double> z_val = std::nullopt,
                   std::optional<double> t_val = std::nullopt)
        : x(x_val), y(y_val), z(z_val), t(t_val) {}
    
    /**
     * @brief 等值运算符
     */
    bool operator==(const GridCoordinate& other) const {
        return x == other.x && y == other.y && z == other.z && t == other.t;
    }
};

/**
 * @brief 算法类型枚举（兼容旧代码）
 */
enum class AlgorithmType {
    BILINEAR = static_cast<int>(InterpolationMethod::BILINEAR),
    CUBIC_SPLINE = static_cast<int>(InterpolationMethod::CUBIC_SPLINE_1D),
    LINEAR_1D = static_cast<int>(InterpolationMethod::LINEAR_1D),
    NEAREST_NEIGHBOR = static_cast<int>(InterpolationMethod::NEAREST_NEIGHBOR),
    TRILINEAR = static_cast<int>(InterpolationMethod::TRILINEAR),
    PCHIP = static_cast<int>(InterpolationMethod::PCHIP_RECURSIVE_NDIM)
};

/**
 * @brief 预计算数据结构（内部实现使用）
 * @details 用于存储算法预计算的数据以提高性能
 */
struct PrecomputedData {
    std::string algorithmType;  ///< 算法类型
    std::vector<double> weights;  ///< 预计算的权重
    std::vector<size_t> indices;  ///< 预计算的索引
    std::map<std::string, std::any> customData;  ///< 自定义数据
    
    /**
     * @brief 检查是否有效
     */
    bool isValid() const {
        return !algorithmType.empty();
    }
};

/**
 * @brief 性能预测结果（内部实现使用）
 * @details 用于预测插值操作的性能
 */
struct PerformancePrediction {
    double estimatedTimeMs = 0.0;  ///< 预估时间（毫秒）
    size_t estimatedMemoryBytes = 0;  ///< 预估内存使用（字节）
    double complexity = 1.0;  ///< 复杂度因子
    std::string recommendedMethod;  ///< 推荐的插值方法
    
    /**
     * @brief 构造函数
     */
    PerformancePrediction() = default;
    
    /**
     * @brief 参数构造函数
     */
    PerformancePrediction(double time_ms, size_t memory_bytes, 
                         double complexity_factor = 1.0,
                         const std::string& method = "")
        : estimatedTimeMs(time_ms), estimatedMemoryBytes(memory_bytes),
          complexity(complexity_factor), recommendedMethod(method) {}
};

/**
 * @brief 插值质量指标（内部实现使用）
 * @details 用于评估插值结果的质量
 */
struct InterpolationQuality {
    double accuracy = 0.0;  ///< 精度（0-1）
    double smoothness = 0.0;  ///< 平滑度（0-1）
    double continuity = 0.0;  ///< 连续性（0-1）
    size_t validPointsCount = 0;  ///< 有效点数量
    size_t totalPointsCount = 0;  ///< 总点数量
    
    /**
     * @brief 计算成功率
     */
    double getSuccessRate() const {
        if (totalPointsCount == 0) return 0.0;
        return static_cast<double>(validPointsCount) / totalPointsCount;
    }
    
    /**
     * @brief 计算总体质量分数
     */
    double getOverallScore() const {
        return (accuracy + smoothness + continuity + getSuccessRate()) / 4.0;
    }
};

/**
 * @brief 插值统计信息（内部实现使用）
 * @details 用于收集插值操作的统计数据
 */
struct InterpolationStatistics {
    size_t totalOperations = 0;  ///< 总操作数
    double totalTimeMs = 0.0;  ///< 总时间（毫秒）
    size_t totalMemoryBytes = 0;  ///< 总内存使用（字节）
    size_t successfulOperations = 0;  ///< 成功操作数
    size_t failedOperations = 0;  ///< 失败操作数
    
    std::map<InterpolationMethod, size_t> methodUsageCount;  ///< 方法使用统计
    std::map<InterpolationMethod, double> methodAverageTime;  ///< 方法平均时间
    
    /**
     * @brief 计算平均时间
     */
    double getAverageTimeMs() const {
        if (totalOperations == 0) return 0.0;
        return totalTimeMs / totalOperations;
    }
    
    /**
     * @brief 计算成功率
     */
    double getSuccessRate() const {
        if (totalOperations == 0) return 0.0;
        return static_cast<double>(successfulOperations) / totalOperations;
    }
};

} // namespace oscean::core_services::interpolation 