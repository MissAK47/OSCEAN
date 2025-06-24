#pragma once

/**
 * @file time_range.h
 * @brief 时间范围处理 - 时间区间操作和管理
 * 
 * 提供时间范围的基础操作：包含、重叠、交集、并集等
 * 以及高级分析功能：分割、合并、覆盖分析等
 * 🔴 注意: 类型定义在time_types.h中，这里只提供额外的工具函数
 */

#include "time_types.h"
#include <vector>
#include <string>

namespace oscean::common_utils::time {

/**
 * @brief 时间范围分析器 - 高级时间范围分析功能
 */
class TimeRangeAnalyzer {
public:
    /**
     * @brief 从时间点集合创建时间范围
     */
    static TimeRange fromTimePoints(const std::vector<CalendarTime>& timePoints);
    
    /**
     * @brief 按时长分割时间范围
     */
    static std::vector<TimeRange> splitByDuration(
        const TimeRange& range, 
        std::chrono::seconds segmentDuration
    );
    
    /**
     * @brief 按数量分割时间范围
     */
    static std::vector<TimeRange> splitByCount(const TimeRange& range, size_t segmentCount);
    
    /**
     * @brief 查找两个范围的重叠部分
     */
    static TimeRange findOverlap(const TimeRange& range1, const TimeRange& range2);
    
    /**
     * @brief 查找两个范围的并集
     */
    static TimeRange findUnion(const TimeRange& range1, const TimeRange& range2);
    
    /**
     * @brief 查找时间范围间的间隙
     */
    static std::vector<TimeRange> findGaps(
        const std::vector<TimeRange>& ranges,
        const TimeRange& searchRange
    );
    
    /**
     * @brief 计算覆盖率
     */
    static double calculateCoverage(
        const std::vector<TimeRange>& ranges,
        const TimeRange& totalRange
    );
    
    /**
     * @brief 检查范围是否连续
     */
    static bool areContiguous(const std::vector<TimeRange>& ranges);
    
    /**
     * @brief 合并重叠的范围
     */
    static std::vector<TimeRange> mergeOverlapping(const std::vector<TimeRange>& ranges);
    
    /**
     * @brief 计算总时长
     */
    static std::chrono::seconds calculateTotalDuration(const std::vector<TimeRange>& ranges);
    
    /**
     * @brief 自适应分段（根据数据密度）
     */
    static std::vector<TimeRange> adaptiveSegmentation(
        const TimeRange& range,
        const std::vector<CalendarTime>& timePoints,
        size_t targetSegments
    );
    
    /**
     * @brief 查找数据范围（排除异常值）
     */
    static TimeRange findDataRange(
        const std::vector<CalendarTime>& timePoints,
        double percentile = 0.95
    );
    
    /**
     * @brief 识别密集时间聚类
     */
    static std::vector<TimeRange> identifyDenseClusters(
        const std::vector<CalendarTime>& timePoints,
        std::chrono::seconds clusterWindow,
        size_t minClusterSize
    );
};

/**
 * @brief 时间范围验证器
 */
class TimeRangeValidator {
public:
    /**
     * @brief 验证时间范围是否有效
     */
    static bool isValid(const TimeRange& range);
    
    /**
     * @brief 检查最小时长要求
     */
    static bool hasMinimumDuration(
        const TimeRange& range,
        std::chrono::seconds minimumDuration
    );
    
    /**
     * @brief 检查是否在边界内
     */
    static bool isWithinBounds(
        const TimeRange& range,
        const TimeRange& bounds
    );
    
    /**
     * @brief 验证并生成报告
     */
    static std::string validateAndReport(const TimeRange& range);
    
    /**
     * @brief 检查合理的时长
     */
    static bool hasReasonableDuration(
        const TimeRange& range,
        std::chrono::seconds maxExpected
    );
    
    /**
     * @brief 查找异常范围
     */
    static std::vector<std::string> findAnomalies(const std::vector<TimeRange>& ranges);
};

/**
 * @brief 时间范围工具类 - 提供额外的时间范围操作（保持向后兼容）
 */
class TimeRangeUtils {
public:
    /**
     * @brief 合并重叠的时间范围
     */
    static std::vector<TimeRange> mergeOverlappingRanges(const std::vector<TimeRange>& ranges);
    
    /**
     * @brief 找出范围间的间隙
     */
    static std::vector<TimeRange> findGapsBetweenRanges(const std::vector<TimeRange>& ranges);
    
    /**
     * @brief 验证时间范围序列
     */
    static bool validateRangeSequence(const std::vector<TimeRange>& ranges);
    
    /**
     * @brief 计算总覆盖时间
     */
    static std::chrono::seconds calculateTotalCoverage(const std::vector<TimeRange>& ranges);
    
    /**
     * @brief 按时间排序范围
     */
    static std::vector<TimeRange> sortRangesByTime(const std::vector<TimeRange>& ranges);
    
    /**
     * @brief 查找包含指定时间的范围
     */
    static std::vector<size_t> findRangesContaining(const std::vector<TimeRange>& ranges, 
                                                   const CalendarTime& time);
};

} // namespace oscean::common_utils::time 