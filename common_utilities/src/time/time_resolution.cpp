/**
 * @file time_resolution.cpp
 * @brief 时间分辨率管理实现 - 完整版
 */

#include "common_utils/time/time_resolution.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <set>

namespace oscean::common_utils::time {

// =============================================================================
// TimeResolutionAnalyzer 实现
// =============================================================================

TimeResolutionInfo TimeResolutionAnalyzer::analyzeResolution(const std::vector<CalendarTime>& timePoints) {
    TimeResolutionInfo info;
    
    if (timePoints.size() < 2) {
        return info;
    }
    
    info.totalTimePoints = timePoints.size();
    
    // 计算时间间隔
    std::vector<std::chrono::seconds> intervals;
    for (size_t i = 1; i < timePoints.size(); ++i) {
        auto interval = timePoints[i] - timePoints[i-1];
        intervals.push_back(interval);
    }
    
    // 计算分辨率统计
    auto minInterval = *std::min_element(intervals.begin(), intervals.end());
    auto maxInterval = *std::max_element(intervals.begin(), intervals.end());
    
    info.minInterval = minInterval;
    info.maxInterval = maxInterval;
    info.nominalResolution = inferNominalResolution(timePoints);
    info.actualResolution = calculateActualResolution(timePoints);
    info.regularityRatio = calculateRegularity(timePoints);
    info.isRegular = info.regularityRatio > 0.95;
    
    // 查找间隙
    info.gaps = findTimeGaps(timePoints, info.nominalResolution);
    
    return info;
}

double TimeResolutionAnalyzer::calculateRegularity(const std::vector<CalendarTime>& timePoints) {
    if (timePoints.size() < 3) {
        return 1.0;
    }
    
    // 计算所有时间间隔
    std::vector<std::chrono::seconds> intervals;
    for (size_t i = 1; i < timePoints.size(); ++i) {
        intervals.push_back(timePoints[i] - timePoints[i-1]);
    }
    
    // 使用更精确的统计方法计算规律性
    std::unordered_map<long long, size_t> intervalCounts;
    for (const auto& interval : intervals) {
        intervalCounts[interval.count()]++;
    }
    
    // 找到最常见的间隔
    auto mostCommon = std::max_element(intervalCounts.begin(), intervalCounts.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    if (mostCommon != intervalCounts.end()) {
        double regularRatio = static_cast<double>(mostCommon->second) / intervals.size();
        return regularRatio;
    }
    
    // 备用方法：计算标准差
    auto mean = std::accumulate(intervals.begin(), intervals.end(), std::chrono::seconds(0)) / 
                static_cast<double>(intervals.size());
    
    double variance = 0.0;
    for (const auto& interval : intervals) {
        double diff = interval.count() - mean.count();
        variance += diff * diff;
    }
    variance /= intervals.size();
    
    double stdDev = std::sqrt(variance);
    
    // 变异系数：标准差/平均值，越小越规律
    if (mean.count() > 0) {
        double coefficientOfVariation = stdDev / mean.count();
        return std::max(0.0, 1.0 - coefficientOfVariation);
    }
    
    return 0.0;
}

std::vector<std::chrono::seconds> TimeResolutionAnalyzer::findTimeGaps(
    const std::vector<CalendarTime>& timePoints, 
    std::chrono::seconds expectedInterval) {
    
    std::vector<std::chrono::seconds> gaps;
    
    if (expectedInterval.count() <= 0) {
        return gaps;
    }
    
    for (size_t i = 1; i < timePoints.size(); ++i) {
        auto actualInterval = timePoints[i] - timePoints[i-1];
        
        // 如果实际间隔超过期望间隔的阈值，认为是间隙
        double gapThreshold = 1.5; // 可调参数
        if (actualInterval > expectedInterval * gapThreshold) {
            gaps.push_back(actualInterval - expectedInterval);
        }
    }
    
    return gaps;
}

std::chrono::seconds TimeResolutionAnalyzer::inferNominalResolution(const std::vector<CalendarTime>& timePoints) {
    if (timePoints.size() < 2) {
        return std::chrono::seconds(0);
    }
    
    // 计算所有时间间隔并统计频率
    std::unordered_map<long long, size_t> intervalCounts;
    for (size_t i = 1; i < timePoints.size(); ++i) {
        auto interval = timePoints[i] - timePoints[i-1];
        intervalCounts[interval.count()]++;
    }
    
    // 找到最常见的间隔作为名义分辨率
    auto mostCommon = std::max_element(intervalCounts.begin(), intervalCounts.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    if (mostCommon != intervalCounts.end()) {
        return std::chrono::seconds(mostCommon->first);
    }
    
    // 备用方法：使用最小间隔
    std::vector<std::chrono::seconds> intervals;
    for (size_t i = 1; i < timePoints.size(); ++i) {
        intervals.push_back(timePoints[i] - timePoints[i-1]);
    }
    
    return *std::min_element(intervals.begin(), intervals.end());
}

std::chrono::seconds TimeResolutionAnalyzer::calculateActualResolution(const std::vector<CalendarTime>& timePoints) {
    if (timePoints.size() < 2) {
        return std::chrono::seconds(0);
    }
    
    // 计算中位数间隔作为实际分辨率（更稳定）
    std::vector<std::chrono::seconds> intervals;
    for (size_t i = 1; i < timePoints.size(); ++i) {
        intervals.push_back(timePoints[i] - timePoints[i-1]);
    }
    
    std::sort(intervals.begin(), intervals.end());
    
    if (intervals.size() % 2 == 0) {
        // 偶数个间隔，取中间两个的平均值
        size_t mid1 = intervals.size() / 2 - 1;
        size_t mid2 = intervals.size() / 2;
        return std::chrono::seconds((intervals[mid1].count() + intervals[mid2].count()) / 2);
    } else {
        // 奇数个间隔，取中位数
        size_t mid = intervals.size() / 2;
        return intervals[mid];
    }
}

bool TimeResolutionAnalyzer::validateTimeSequenceQuality(const std::vector<CalendarTime>& timePoints) {
    if (timePoints.size() < 2) {
        return false;
    }
    
    auto info = analyzeResolution(timePoints);
    
    // 质量标准
    const double MIN_REGULARITY = 0.8;
    const size_t MAX_GAPS = timePoints.size() / 10; // 最多10%的间隙
    const double MAX_RESOLUTION_VARIANCE = 2.0; // 最大分辨率方差比
    
    if (info.regularityRatio < MIN_REGULARITY) {
        return false;
    }
    
    if (info.gaps.size() > MAX_GAPS) {
        return false;
    }
    
    if (info.maxInterval.count() > 0 && info.minInterval.count() > 0) {
        double resolutionVariance = static_cast<double>(info.maxInterval.count()) / info.minInterval.count();
        if (resolutionVariance > MAX_RESOLUTION_VARIANCE) {
            return false;
        }
    }
    
    return true;
}

std::vector<CalendarTime> TimeResolutionAnalyzer::detectOutliers(
    const std::vector<CalendarTime>& timePoints,
    double standardDeviations) {
    
    std::vector<CalendarTime> outliers;
    
    if (timePoints.size() < 3) {
        return outliers;
    }
    
    // 计算时间间隔
    std::vector<std::chrono::seconds> intervals;
    for (size_t i = 1; i < timePoints.size(); ++i) {
        intervals.push_back(timePoints[i] - timePoints[i-1]);
    }
    
    // 计算平均值和标准差
    double mean = std::accumulate(intervals.begin(), intervals.end(), std::chrono::seconds(0)).count() 
                  / static_cast<double>(intervals.size());
    
    double variance = 0.0;
    for (const auto& interval : intervals) {
        double diff = interval.count() - mean;
        variance += diff * diff;
    }
    variance /= intervals.size();
    double stdDev = std::sqrt(variance);
    
    // 找到异常点
    double threshold = standardDeviations * stdDev;
    for (size_t i = 0; i < intervals.size(); ++i) {
        if (std::abs(intervals[i].count() - mean) > threshold) {
            outliers.push_back(timePoints[i + 1]); // 异常间隔对应的第二个时间点
        }
    }
    
    return outliers;
}

TimeResolutionInfo TimeResolutionAnalyzer::analyzeResolutionWithFiltering(
    const std::vector<CalendarTime>& timePoints,
    double outlierThreshold) {
    
    // 先进行异常值检测
    auto outliers = detectOutliers(timePoints, outlierThreshold);
    
    if (outliers.empty()) {
        return analyzeResolution(timePoints);
    }
    
    // 创建过滤后的时间点序列
    std::set<CalendarTime> outlierSet(outliers.begin(), outliers.end());
    std::vector<CalendarTime> filteredTimePoints;
    
    for (const auto& timePoint : timePoints) {
        if (outlierSet.find(timePoint) == outlierSet.end()) {
            filteredTimePoints.push_back(timePoint);
        }
    }
    
    return analyzeResolution(filteredTimePoints);
}

// =============================================================================
// TimeResolutionConfig 实现
// =============================================================================

bool TimeResolutionConfig::isValid() const {
    return expectedResolution.count() > 0 && 
           regularityThreshold >= 0.0 && regularityThreshold <= 1.0 &&
           maxAllowedGap.count() >= 0 &&
           minSampleSize > 0;
}

std::string TimeResolutionConfig::toString() const {
    std::ostringstream oss;
    oss << "TimeResolutionConfig{"
        << "expectedResolution=" << expectedResolution.count() << "s"
        << ", regularityThreshold=" << regularityThreshold
        << ", maxAllowedGap=" << maxAllowedGap.count() << "s"
        << ", strictValidation=" << (strictValidation ? "true" : "false")
        << ", minSampleSize=" << minSampleSize
        << "}";
    return oss.str();
}

TimeResolutionConfig TimeResolutionConfig::createForDataType(const std::string& dataType) {
    TimeResolutionConfig config;
    
    if (dataType == "meteorological") {
        config.expectedResolution = std::chrono::hours(1); // 1小时
        config.regularityThreshold = 0.9;
        config.maxAllowedGap = std::chrono::hours(6);
        config.strictValidation = true;
        config.minSampleSize = 24; // 至少一天的数据
    } else if (dataType == "oceanographic") {
        config.expectedResolution = std::chrono::minutes(30); // 30分钟
        config.regularityThreshold = 0.85;
        config.maxAllowedGap = std::chrono::hours(3);
        config.strictValidation = false;
        config.minSampleSize = 48; // 至少一天的数据
    } else if (dataType == "satellite") {
        config.expectedResolution = std::chrono::hours(24); // 1天
        config.regularityThreshold = 0.95;
        config.maxAllowedGap = std::chrono::hours(72);
        config.strictValidation = true;
        config.minSampleSize = 10; // 至少10天的数据
    } else if (dataType == "climate") {
        config.expectedResolution = std::chrono::hours(24 * 30); // 1月
        config.regularityThreshold = 0.8;
        config.maxAllowedGap = std::chrono::hours(24 * 90);
        config.strictValidation = false;
        config.minSampleSize = 12; // 至少一年的数据
    } else {
        // 默认配置
        config.expectedResolution = std::chrono::hours(1);
        config.regularityThreshold = 0.8;
        config.maxAllowedGap = std::chrono::hours(24);
        config.strictValidation = false;
        config.minSampleSize = 10;
    }
    
    return config;
}

// =============================================================================
// TimeResolutionOptimizer 实现
// =============================================================================

TimeResolutionOptimizer::TimeResolutionOptimizer(const TimeResolutionConfig& config)
    : config_(config) {
}

TimeResolutionInfo TimeResolutionOptimizer::optimizeResolution(const std::vector<CalendarTime>& timePoints) const {
    // 先进行基础分析
    auto basicInfo = TimeResolutionAnalyzer::analyzeResolution(timePoints);
    
    // 如果质量不达标，尝试优化
    if (basicInfo.regularityRatio < config_.regularityThreshold) {
        return optimizeWithFiltering(timePoints);
    }
    
    return basicInfo;
}

std::chrono::seconds TimeResolutionOptimizer::suggestOptimalInterval(const std::vector<CalendarTime>& timePoints) const {
    if (timePoints.size() < 2) {
        return config_.expectedResolution;
    }
    
    auto info = TimeResolutionAnalyzer::analyzeResolution(timePoints);
    
    // 智能选择最优间隔
    std::vector<std::chrono::seconds> candidates = {
        info.nominalResolution,
        info.actualResolution,
        config_.expectedResolution
    };
    
    // 移除无效候选
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [](const std::chrono::seconds& s) { return s.count() <= 0; }), candidates.end());
    
    if (candidates.empty()) {
        return config_.expectedResolution;
    }
    
    // 选择最接近期望分辨率的候选
    auto best = std::min_element(candidates.begin(), candidates.end(),
        [this](const std::chrono::seconds& a, const std::chrono::seconds& b) {
            auto diffA = std::abs(a.count() - config_.expectedResolution.count());
            auto diffB = std::abs(b.count() - config_.expectedResolution.count());
            return diffA < diffB;
        });
    
    return *best;
}

std::string TimeResolutionOptimizer::assessTimeQuality(const TimeResolutionInfo& resolution) const {
    std::ostringstream oss;
    
    if (resolution.isHighQuality()) {
        oss << "优秀: 高质量规律时间序列";
    } else if (resolution.regularityRatio >= config_.regularityThreshold) {
        oss << "良好: 满足质量要求";
    } else if (resolution.regularityRatio > 0.5) {
        oss << "一般: 存在一些不规律性";
    } else {
        oss << "较差: 高度不规律的时间序列";
    }
    
    if (resolution.hasSignificantGaps()) {
        oss << " (包含间隙)";
    }
    
    // 添加详细统计
    oss << " [规律性: " << std::fixed << std::setprecision(1) << (resolution.regularityRatio * 100) << "%";
    oss << ", 间隙: " << resolution.gaps.size() << "个";
    oss << ", 点数: " << resolution.totalTimePoints << "]";
    
    return oss.str();
}

std::vector<CalendarTime> TimeResolutionOptimizer::generateOptimalTimeGrid(
    const TimeRange& timeRange,
    std::chrono::seconds interval) const {
    
    std::vector<CalendarTime> timeGrid;
    
    if (interval.count() <= 0) {
        return timeGrid;
    }
    
    auto current = timeRange.startTime;
    while (current <= timeRange.endTime) {
        timeGrid.push_back(current);
        current = current + interval;
    }
    
    return timeGrid;
}

TimeResolutionInfo TimeResolutionOptimizer::optimizeWithFiltering(const std::vector<CalendarTime>& timePoints) const {
    // 尝试不同的过滤策略
    std::vector<double> outlierThresholds = {3.0, 2.5, 2.0, 1.5};
    
    for (double threshold : outlierThresholds) {
        auto filteredInfo = TimeResolutionAnalyzer::analyzeResolutionWithFiltering(timePoints, threshold);
        
        if (filteredInfo.regularityRatio >= config_.regularityThreshold) {
            return filteredInfo;
        }
    }
    
    // 如果过滤后仍不满足要求，返回原始分析结果
    return TimeResolutionAnalyzer::analyzeResolution(timePoints);
}

TimeResolutionOptimizer::OptimizationStrategy TimeResolutionOptimizer::selectOptimizationStrategy(
    const TimeResolutionInfo& info) const {
    
    if (info.regularityRatio >= 0.9) {
        return OptimizationStrategy::NONE; // 已经很好，无需优化
    } else if (info.gaps.size() > info.totalTimePoints / 5) {
        return OptimizationStrategy::GAP_FILLING; // 间隙太多，需要填充
    } else if (info.regularityRatio < 0.5) {
        return OptimizationStrategy::OUTLIER_REMOVAL; // 不规律，移除异常值
    } else {
        return OptimizationStrategy::INTERPOLATION; // 插值平滑
    }
}

} // namespace oscean::common_utils::time 