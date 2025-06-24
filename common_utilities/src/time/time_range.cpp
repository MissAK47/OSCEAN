/**
 * @file time_range.cpp
 * @brief 时间范围处理实现 - 完整版
 */

#include "common_utils/time/time_range.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace oscean::common_utils::time {

// =============================================================================
// TimeRange 实现 - 只实现在头文件中声明但未定义的方法
// =============================================================================

std::chrono::seconds TimeRange::getDuration() const {
    return endTime - startTime;
}

bool TimeRange::contains(const CalendarTime& time) const {
    return time >= startTime && time <= endTime;
}

bool TimeRange::overlaps(const TimeRange& other) const {
    return !(endTime < other.startTime || startTime > other.endTime);
}

bool TimeRange::isEmpty() const {
    return startTime == endTime;
}

std::string TimeRange::toString() const {
    std::ostringstream oss;
    oss << "[" << startTime.toISOString() << " - " << endTime.toISOString() << "]";
    return oss.str();
}

// =============================================================================
// TimeIndex 实现
// =============================================================================

TimeIndex::TimeIndex(const std::vector<CalendarTime>& times) 
    : timePoints(times) {
    buildIndex();
}

std::optional<size_t> TimeIndex::findIndex(const CalendarTime& time) const {
    auto it = timeToIndex.find(time);
    if (it != timeToIndex.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<CalendarTime> TimeIndex::getTime(size_t index) const {
    auto it = indexToTime.find(index);
    if (it != indexToTime.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<size_t> TimeIndex::findIndicesInRange(const TimeRange& range) const {
    std::vector<size_t> indices;
    for (size_t i = 0; i < timePoints.size(); ++i) {
        if (range.contains(timePoints[i])) {
            indices.push_back(i);
        }
    }
    return indices;
}

void TimeIndex::buildIndex() {
    timeToIndex.clear();
    indexToTime.clear();
    
    for (size_t i = 0; i < timePoints.size(); ++i) {
        timeToIndex[timePoints[i]] = i;
        indexToTime[i] = timePoints[i];
    }
    
    if (!timePoints.empty()) {
        auto minMax = std::minmax_element(timePoints.begin(), timePoints.end());
        coverageRange = TimeRange(*minMax.first, *minMax.second);
    }
}

void TimeIndex::addTimePoint(const CalendarTime& time) {
    size_t index = timePoints.size();
    timePoints.push_back(time);
    timeToIndex[time] = index;
    indexToTime[index] = time;
    
    // Update coverage range
    if (timePoints.size() == 1) {
        coverageRange = TimeRange(time, time);
    } else {
        if (time < coverageRange.startTime) {
            coverageRange.startTime = time;
        }
        if (time > coverageRange.endTime) {
            coverageRange.endTime = time;
        }
    }
}

void TimeIndex::clear() {
    timePoints.clear();
    timeToIndex.clear();
    indexToTime.clear();
    coverageRange = TimeRange{};
}

bool TimeIndex::isValid() const {
    return !timePoints.empty() && 
           timeToIndex.size() == timePoints.size() &&
           indexToTime.size() == timePoints.size();
}

// =============================================================================
// TimeRangeAnalyzer 实现
// =============================================================================

TimeRange TimeRangeAnalyzer::fromTimePoints(const std::vector<CalendarTime>& timePoints) {
    if (timePoints.empty()) {
        return TimeRange{};
    }
    
    auto minmaxPair = std::minmax_element(timePoints.begin(), timePoints.end());
    return TimeRange(*minmaxPair.first, *minmaxPair.second);
}

std::vector<TimeRange> TimeRangeAnalyzer::splitByDuration(
    const TimeRange& range, 
    std::chrono::seconds segmentDuration) {
    
    std::vector<TimeRange> segments;
    
    if (segmentDuration.count() <= 0) {
        return segments;
    }
    
    auto current = range.startTime;
    while (current < range.endTime) {
        auto segmentEnd = current + segmentDuration;
        if (segmentEnd > range.endTime) {
            segmentEnd = range.endTime;
        }
        
        segments.emplace_back(current, segmentEnd);
        current = segmentEnd;
    }
    
    return segments;
}

std::vector<TimeRange> TimeRangeAnalyzer::splitByCount(const TimeRange& range, size_t segmentCount) {
    std::vector<TimeRange> segments;
    
    if (segmentCount == 0 || range.isEmpty()) {
        return segments;
    }
    
    auto totalDuration = range.getDuration();
    auto segmentDuration = std::chrono::seconds(totalDuration.count() / static_cast<long long>(segmentCount));
    
    // 处理除不尽的情况
    auto remainder = std::chrono::seconds(totalDuration.count() % static_cast<long long>(segmentCount));
    
    auto current = range.startTime;
    for (size_t i = 0; i < segmentCount; ++i) {
        auto segmentEnd = current + segmentDuration;
        
        // 将余数分布到前几个段中
        if (i < static_cast<size_t>(remainder.count())) {
            segmentEnd = segmentEnd + std::chrono::seconds(1);
        }
        
        if (i == segmentCount - 1) {
            segmentEnd = range.endTime; // 确保最后一段精确到结束时间
        }
        
        segments.emplace_back(current, segmentEnd);
        current = segmentEnd;
    }
    
    return segments;
}

TimeRange TimeRangeAnalyzer::findOverlap(const TimeRange& range1, const TimeRange& range2) {
    if (!range1.overlaps(range2)) {
        return TimeRange{}; // 空范围
    }
    
    auto overlapStart = std::max(range1.startTime, range2.startTime);
    auto overlapEnd = std::min(range1.endTime, range2.endTime);
    
    return TimeRange(overlapStart, overlapEnd);
}

TimeRange TimeRangeAnalyzer::findUnion(const TimeRange& range1, const TimeRange& range2) {
    auto unionStart = std::min(range1.startTime, range2.startTime);
    auto unionEnd = std::max(range1.endTime, range2.endTime);
    
    return TimeRange(unionStart, unionEnd);
}

std::vector<TimeRange> TimeRangeAnalyzer::findGaps(
    const std::vector<TimeRange>& ranges,
    const TimeRange& searchRange) {
    
    std::vector<TimeRange> gaps;
    
    if (ranges.empty()) {
        gaps.push_back(searchRange);
        return gaps;
    }
    
    // 对范围按开始时间排序
    auto sortedRanges = ranges;
    std::sort(sortedRanges.begin(), sortedRanges.end(),
        [](const TimeRange& a, const TimeRange& b) {
            return a.startTime < b.startTime;
        });
    
    // 合并重叠的范围
    std::vector<TimeRange> mergedRanges;
    mergedRanges.push_back(sortedRanges[0]);
    
    for (size_t i = 1; i < sortedRanges.size(); ++i) {
        auto& lastMerged = mergedRanges.back();
        const auto& current = sortedRanges[i];
        
        if (lastMerged.overlaps(current) || lastMerged.endTime == current.startTime) {
            // 合并范围
            lastMerged.endTime = std::max(lastMerged.endTime, current.endTime);
        } else {
            mergedRanges.push_back(current);
        }
    }
    
    // 查找间隙
    auto current = searchRange.startTime;
    for (const auto& range : mergedRanges) {
        if (range.startTime > current && range.startTime <= searchRange.endTime) {
            auto gapEnd = std::min(range.startTime, searchRange.endTime);
            if (current < gapEnd) {
                gaps.emplace_back(current, gapEnd);
            }
        }
        current = std::max(current, range.endTime);
        
        if (current >= searchRange.endTime) {
            break;
        }
    }
    
    // 检查最后的间隙
    if (current < searchRange.endTime) {
        gaps.emplace_back(current, searchRange.endTime);
    }
    
    return gaps;
}

double TimeRangeAnalyzer::calculateCoverage(
    const std::vector<TimeRange>& ranges,
    const TimeRange& totalRange) {
    
    if (totalRange.isEmpty()) {
        return 0.0;
    }
    
    auto coveredTime = std::chrono::seconds(0);
    
    for (const auto& range : ranges) {
        auto overlap = findOverlap(range, totalRange);
        if (!overlap.isEmpty()) {
            coveredTime += overlap.getDuration();
        }
    }
    
    auto totalTime = totalRange.getDuration();
    if (totalTime.count() == 0) {
        return 0.0;
    }
    
    return static_cast<double>(coveredTime.count()) / totalTime.count();
}

bool TimeRangeAnalyzer::areContiguous(const std::vector<TimeRange>& ranges) {
    if (ranges.size() < 2) {
        return true;
    }
    
    // 对范围按开始时间排序
    auto sortedRanges = ranges;
    std::sort(sortedRanges.begin(), sortedRanges.end(),
        [](const TimeRange& a, const TimeRange& b) {
            return a.startTime < b.startTime;
        });
    
    // 检查相邻范围是否连续
    for (size_t i = 1; i < sortedRanges.size(); ++i) {
        if (sortedRanges[i-1].endTime != sortedRanges[i].startTime) {
            return false;
        }
    }
    
    return true;
}

std::vector<TimeRange> TimeRangeAnalyzer::mergeOverlapping(const std::vector<TimeRange>& ranges) {
    if (ranges.empty()) {
        return {};
    }
    
    // 对范围按开始时间排序
    auto sortedRanges = ranges;
    std::sort(sortedRanges.begin(), sortedRanges.end(),
        [](const TimeRange& a, const TimeRange& b) {
            return a.startTime < b.startTime;
        });
    
    std::vector<TimeRange> merged;
    merged.push_back(sortedRanges[0]);
    
    for (size_t i = 1; i < sortedRanges.size(); ++i) {
        auto& lastMerged = merged.back();
        const auto& current = sortedRanges[i];
        
        if (lastMerged.overlaps(current) || lastMerged.endTime == current.startTime) {
            // 合并范围
            lastMerged.endTime = std::max(lastMerged.endTime, current.endTime);
        } else {
            merged.push_back(current);
        }
    }
    
    return merged;
}

std::chrono::seconds TimeRangeAnalyzer::calculateTotalDuration(const std::vector<TimeRange>& ranges) {
    // 先合并重叠的范围，然后计算总时长
    auto merged = mergeOverlapping(ranges);
    
    auto totalDuration = std::chrono::seconds(0);
    for (const auto& range : merged) {
        totalDuration += range.getDuration();
    }
    
    return totalDuration;
}

// =============================================================================
// TimeRangeValidator 实现
// =============================================================================

bool TimeRangeValidator::isValid(const TimeRange& range) {
    return range.startTime <= range.endTime;
}

bool TimeRangeValidator::hasMinimumDuration(
    const TimeRange& range,
    std::chrono::seconds minimumDuration) {
    
    return range.getDuration() >= minimumDuration;
}

bool TimeRangeValidator::isWithinBounds(
    const TimeRange& range,
    const TimeRange& bounds) {
    
    return bounds.contains(range.startTime) && bounds.contains(range.endTime);
}

std::string TimeRangeValidator::validateAndReport(const TimeRange& range) {
    std::ostringstream report;
    
    if (!isValid(range)) {
        report << "ERROR: 无效时间范围 - 开始时间晚于结束时间\\n";
    }
    
    if (range.isEmpty()) {
        report << "WARNING: 空时间范围\\n";
    }
    
    auto duration = range.getDuration();
    if (duration.count() < 0) {
        report << "ERROR: 负时长: " << duration.count() << " 秒\\n";
    } else if (duration.count() == 0) {
        report << "INFO: 零时长范围\\n";
    } else {
        report << "INFO: 有效时间范围，时长: " << duration.count() << " 秒\\n";
    }
    
    if (report.str().empty()) {
        report << "OK: 时间范围验证通过";
    }
    
    return report.str();
}

bool TimeRangeValidator::hasReasonableDuration(
    const TimeRange& range,
    std::chrono::seconds maxExpected) {
    
    auto duration = range.getDuration();
    return duration.count() > 0 && duration <= maxExpected;
}

std::vector<std::string> TimeRangeValidator::findAnomalies(const std::vector<TimeRange>& ranges) {
    std::vector<std::string> anomalies;
    
    if (ranges.empty()) {
        anomalies.push_back("空范围列表");
        return anomalies;
    }
    
    // 检查重叠
    for (size_t i = 0; i < ranges.size(); ++i) {
        for (size_t j = i + 1; j < ranges.size(); ++j) {
            if (ranges[i].overlaps(ranges[j])) {
                std::ostringstream oss;
                oss << "范围重叠: [" << i << "] 和 [" << j << "]";
                anomalies.push_back(oss.str());
            }
        }
    }
    
    // 检查无效范围
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (!isValid(ranges[i])) {
            std::ostringstream oss;
            oss << "无效范围 [" << i << "]: " << ranges[i].toString();
            anomalies.push_back(oss.str());
        }
    }
    
    // 检查时长异常
    if (ranges.size() > 1) {
        std::vector<std::chrono::seconds> durations;
        for (const auto& range : ranges) {
            durations.push_back(range.getDuration());
        }
        
        // 计算时长的平均值和标准差
        auto meanDuration = std::chrono::seconds(0);
        for (const auto& duration : durations) {
            meanDuration += duration;
        }
        meanDuration = std::chrono::seconds(meanDuration.count() / static_cast<long long>(durations.size()));
        
        double variance = 0.0;
        for (const auto& duration : durations) {
            double diff = duration.count() - meanDuration.count();
            variance += diff * diff;
        }
        variance /= durations.size();
        double stdDev = std::sqrt(variance);
        
        // 标记异常时长 (超过2个标准差)
        for (size_t i = 0; i < ranges.size(); ++i) {
            double diff = std::abs(ranges[i].getDuration().count() - meanDuration.count());
            if (diff > 2 * stdDev) {
                std::ostringstream oss;
                oss << "异常时长 [" << i << "]: " << ranges[i].getDuration().count() 
                    << " 秒 (平均值: " << meanDuration.count() << " 秒)";
                anomalies.push_back(oss.str());
            }
        }
    }
    
    return anomalies;
}

// =============================================================================
// 高级时间范围操作
// =============================================================================

std::vector<TimeRange> TimeRangeAnalyzer::adaptiveSegmentation(
    const TimeRange& range,
    const std::vector<CalendarTime>& timePoints,
    size_t targetSegments) {
    
    std::vector<TimeRange> segments;
    
    if (targetSegments == 0 || timePoints.empty()) {
        return segments;
    }
    
    // 根据数据密度进行自适应分段
    auto pointsInRange = std::count_if(timePoints.begin(), timePoints.end(),
        [&range](const CalendarTime& time) {
            return range.contains(time);
        });
    
    if (pointsInRange <= targetSegments) {
        // 数据点较少，直接分段
        return splitByCount(range, targetSegments);
    }
    
    // 计算数据密度分布
    auto preliminarySegments = splitByCount(range, targetSegments * 2); // 先细分
    std::vector<size_t> densities;
    
    for (const auto& segment : preliminarySegments) {
        auto density = std::count_if(timePoints.begin(), timePoints.end(),
            [&segment](const CalendarTime& time) {
                return segment.contains(time);
            });
        densities.push_back(density);
    }
    
    // 根据密度合并段
    segments.push_back(preliminarySegments[0]);
    size_t currentDensity = densities[0];
    
    for (size_t i = 1; i < preliminarySegments.size(); ++i) {
        if (segments.size() >= targetSegments) {
            // 达到目标段数，合并剩余段到最后一段
            segments.back().endTime = preliminarySegments.back().endTime;
            break;
        }
        
        currentDensity += densities[i];
        
        // 决定是否开始新段
        double avgDensity = static_cast<double>(pointsInRange) / targetSegments;
        if (currentDensity >= avgDensity || segments.size() == targetSegments - 1) {
            segments.back().endTime = preliminarySegments[i].endTime;
            if (i < preliminarySegments.size() - 1) {
                segments.push_back(preliminarySegments[i + 1]);
                currentDensity = 0;
            }
        } else {
            // 扩展当前段
            segments.back().endTime = preliminarySegments[i].endTime;
        }
    }
    
    return segments;
}

TimeRange TimeRangeAnalyzer::findDataRange(
    const std::vector<CalendarTime>& timePoints,
    double percentile) {
    
    if (timePoints.empty()) {
        return TimeRange{};
    }
    
    if (percentile >= 1.0) {
        return fromTimePoints(timePoints);
    }
    
    auto sortedTimes = timePoints;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    
    double excludeRatio = (1.0 - percentile) / 2.0;
    size_t startIndex = static_cast<size_t>(sortedTimes.size() * excludeRatio);
    size_t endIndex = static_cast<size_t>(sortedTimes.size() * (1.0 - excludeRatio));
    
    endIndex = std::min(endIndex, sortedTimes.size() - 1);
    
    return TimeRange(sortedTimes[startIndex], sortedTimes[endIndex]);
}

std::vector<TimeRange> TimeRangeAnalyzer::identifyDenseClusters(
    const std::vector<CalendarTime>& timePoints,
    std::chrono::seconds clusterWindow,
    size_t minClusterSize) {
    
    std::vector<TimeRange> clusters;
    
    if (timePoints.size() < minClusterSize) {
        return clusters;
    }
    
    auto sortedTimes = timePoints;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    
    size_t clusterStart = 0;
    for (size_t i = 1; i < sortedTimes.size(); ++i) {
        auto gap = sortedTimes[i] - sortedTimes[i-1];
        
        if (gap > clusterWindow) {
            // 检查当前聚类是否满足最小大小
            if (i - clusterStart >= minClusterSize) {
                clusters.emplace_back(sortedTimes[clusterStart], sortedTimes[i-1]);
            }
            clusterStart = i;
        }
    }
    
    // 检查最后一个聚类
    if (sortedTimes.size() - clusterStart >= minClusterSize) {
        clusters.emplace_back(sortedTimes[clusterStart], sortedTimes.back());
    }
    
    return clusters;
}

} // namespace oscean::common_utils::time 