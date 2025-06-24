#pragma once

/**
 * @file time_resolution.h
 * @brief 时间分辨率管理 - 时间精度和规律性分析
 * 
 * 提供时间分辨率检测、分析和优化功能
 */

#include "time_types.h"
#include <vector>
#include <chrono>
#include <string>

namespace oscean::common_utils::time {

/**
 * @brief 时间分辨率分析器
 */
class TimeResolutionAnalyzer {
public:
    /**
     * @brief 分析时间序列的分辨率
     */
    static TimeResolutionInfo analyzeResolution(const std::vector<CalendarTime>& timePoints);
    
    /**
     * @brief 检测时间间隔规律性
     */
    static double calculateRegularity(const std::vector<CalendarTime>& timePoints);
    
    /**
     * @brief 查找时间间隙
     */
    static std::vector<std::chrono::seconds> findTimeGaps(
        const std::vector<CalendarTime>& timePoints,
        std::chrono::seconds expectedInterval
    );
    
    /**
     * @brief 推断名义分辨率
     */
    static std::chrono::seconds inferNominalResolution(const std::vector<CalendarTime>& timePoints);
    
    /**
     * @brief 计算实际分辨率
     */
    static std::chrono::seconds calculateActualResolution(const std::vector<CalendarTime>& timePoints);
    
    /**
     * @brief 验证时间序列质量
     */
    static bool validateTimeSequenceQuality(const std::vector<CalendarTime>& timePoints);
    
    /**
     * @brief 检测时间序列中的异常值
     */
    static std::vector<CalendarTime> detectOutliers(
        const std::vector<CalendarTime>& timePoints,
        double standardDeviations = 2.0
    );
    
    /**
     * @brief 带异常值过滤的分辨率分析
     */
    static TimeResolutionInfo analyzeResolutionWithFiltering(
        const std::vector<CalendarTime>& timePoints,
        double outlierThreshold = 2.0
    );
};

/**
 * @brief 时间分辨率配置
 */
struct TimeResolutionConfig {
    std::chrono::seconds expectedResolution{3600};  // 期望分辨率
    double regularityThreshold = 0.95;              // 规律性阈值
    std::chrono::seconds maxAllowedGap{7200};        // 最大允许间隙
    bool strictValidation = false;                   // 严格验证
    size_t minSampleSize = 10;                      // 最小样本大小
    
    // 验证配置
    bool isValid() const;
    std::string toString() const;
    
    /**
     * @brief 为特定数据类型创建配置
     */
    static TimeResolutionConfig createForDataType(const std::string& dataType);
};

/**
 * @brief 时间分辨率优化器
 */
class TimeResolutionOptimizer {
public:
    /**
     * @brief 优化策略枚举
     */
    enum class OptimizationStrategy {
        NONE,              // 无需优化
        OUTLIER_REMOVAL,   // 移除异常值
        GAP_FILLING,       // 填充间隙
        INTERPOLATION      // 插值平滑
    };
    
    explicit TimeResolutionOptimizer(const TimeResolutionConfig& config = {});
    
    /**
     * @brief 优化时间分辨率检测
     */
    TimeResolutionInfo optimizeResolution(const std::vector<CalendarTime>& timePoints) const;
    
    /**
     * @brief 建议最佳采样间隔
     */
    std::chrono::seconds suggestOptimalInterval(const std::vector<CalendarTime>& timePoints) const;
    
    /**
     * @brief 评估时间序列质量
     */
    std::string assessTimeQuality(const TimeResolutionInfo& resolution) const;
    
    /**
     * @brief 生成最优时间网格
     */
    std::vector<CalendarTime> generateOptimalTimeGrid(
        const TimeRange& timeRange,
        std::chrono::seconds interval
    ) const;

private:
    TimeResolutionConfig config_;
    
    /**
     * @brief 使用过滤策略优化
     */
    TimeResolutionInfo optimizeWithFiltering(const std::vector<CalendarTime>& timePoints) const;
    
    /**
     * @brief 选择优化策略
     */
    OptimizationStrategy selectOptimizationStrategy(const TimeResolutionInfo& info) const;
};

} // namespace oscean::common_utils::time 