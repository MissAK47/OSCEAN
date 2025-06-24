/**
 * @file time_types.cpp
 * @brief 通用时间类型实现 - 基础类型和操作
 */

#include "common_utils/time/time_types.h"
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <stdexcept>

namespace oscean::common_utils::time {

// =============================================================================
// CalendarTime 完整实现
// =============================================================================

std::string CalendarTime::toISOString() const {
    if (!isValid()) {
        return "INVALID_TIME";
    }
    
    auto time_t_val = std::chrono::system_clock::to_time_t(timePoint);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        timePoint.time_since_epoch()
    ) % 1000;
    
    std::ostringstream oss;
    
    // 使用UTC时间
    std::tm utc_tm;
#ifdef _WIN32
    gmtime_s(&utc_tm, &time_t_val);
#else
    gmtime_r(&time_t_val, &utc_tm);
#endif
    
    oss << std::put_time(&utc_tm, "%Y-%m-%dT%H:%M:%S");
    
    // 添加毫秒
    if (milliseconds.count() > 0) {
        oss << "." << std::setfill('0') << std::setw(3) << milliseconds.count();
    }
    
    oss << "Z"; // UTC标识
    return oss.str();
}

double CalendarTime::toJulianDay() const {
    // 儒略日计算：从公元前4713年1月1日正午开始的天数
    auto seconds_since_epoch = std::chrono::duration_cast<std::chrono::seconds>(timePoint.time_since_epoch()).count();
    const double UNIX_EPOCH_JULIAN_DAY = 2440587.5; // 1970年1月1日的儒略日
    const double SECONDS_PER_DAY = 86400.0;
    
    return UNIX_EPOCH_JULIAN_DAY + (seconds_since_epoch / SECONDS_PER_DAY);
}

CalendarTime CalendarTime::operator+(std::chrono::seconds duration) const {
    CalendarTime result = *this;
    result.timePoint += duration;
    return result;
}

CalendarTime CalendarTime::operator-(std::chrono::seconds duration) const {
    CalendarTime result = *this;
    result.timePoint -= duration;
    return result;
}

std::chrono::seconds CalendarTime::operator-(const CalendarTime& other) const {
    return std::chrono::duration_cast<std::chrono::seconds>(timePoint - other.timePoint);
}

bool CalendarTime::operator<(const CalendarTime& other) const {
    return timePoint < other.timePoint;
}

bool CalendarTime::operator==(const CalendarTime& other) const {
    return timePoint == other.timePoint && calendarType == other.calendarType;
}

bool CalendarTime::operator<=(const CalendarTime& other) const {
    return timePoint <= other.timePoint;
}

bool CalendarTime::operator>=(const CalendarTime& other) const {
    return timePoint >= other.timePoint;
}

bool CalendarTime::operator>(const CalendarTime& other) const {
    return timePoint > other.timePoint;
}

bool CalendarTime::operator!=(const CalendarTime& other) const {
    return !(*this == other);
}

std::string CalendarTime::serialize() const {
    std::ostringstream oss;
    auto milliseconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
        timePoint.time_since_epoch()
    ).count();
    
    oss << "{"
        << "\"timePoint\":" << milliseconds_since_epoch << ","
        << "\"calendarType\":\"" << calendarType << "\","
        << "\"julianDay\":" << std::fixed << std::setprecision(6) << julianDay
        << "}";
    
    return oss.str();
}

CalendarTime CalendarTime::deserialize(const std::string& data) {
    // 简化的JSON解析
    CalendarTime result;
    
    try {
        // 查找时间点
        auto timePointPos = data.find("\"timePoint\":");
        if (timePointPos != std::string::npos) {
            auto valueStart = data.find(':', timePointPos) + 1;
            auto valueEnd = data.find(',', valueStart);
            if (valueEnd == std::string::npos) {
                valueEnd = data.find('}', valueStart);
            }
            
            if (valueEnd != std::string::npos) {
                auto valueStr = data.substr(valueStart, valueEnd - valueStart);
                long long milliseconds = std::stoll(valueStr);
                result.timePoint = std::chrono::system_clock::time_point(
                    std::chrono::milliseconds(milliseconds)
                );
            }
        }
        
        // 查找日历类型
        auto calendarPos = data.find("\"calendarType\":");
        if (calendarPos != std::string::npos) {
            auto valueStart = data.find('\"', calendarPos + 15) + 1;
            auto valueEnd = data.find('\"', valueStart);
            if (valueEnd != std::string::npos) {
                result.calendarType = data.substr(valueStart, valueEnd - valueStart);
            }
        }
        
        // 查找儒略日
        auto julianPos = data.find("\"julianDay\":");
        if (julianPos != std::string::npos) {
            auto valueStart = data.find(':', julianPos) + 1;
            auto valueEnd = data.find('}', valueStart);
            if (valueEnd != std::string::npos) {
                auto valueStr = data.substr(valueStart, valueEnd - valueStart);
                result.julianDay = std::stod(valueStr);
            }
        }
        
    } catch (const std::exception&) {
        // 解析失败，返回默认时间
        result = CalendarTime{};
    }
    
    return result;
}

bool CalendarTime::isValid() const {
    // 检查时间点是否在合理范围内 - 🔧 修复：扩大范围以支持地质时间尺度
    const auto MIN_TIME_POINT = std::chrono::system_clock::from_time_t(-147465600000L); // 约公元前2670年（更大范围）
    const auto MAX_TIME_POINT = std::chrono::system_clock::from_time_t(253402300799L);  // 9999年12月31日
    
    return timePoint >= MIN_TIME_POINT && timePoint <= MAX_TIME_POINT && !calendarType.empty();
}

// =============================================================================
// TimeRange 额外方法实现
// =============================================================================

TimeRange TimeRange::intersection(const TimeRange& other) const {
    if (!overlaps(other)) {
        return TimeRange{}; // 空范围
    }
    
    auto intersectionStart = std::max(startTime, other.startTime);
    auto intersectionEnd = std::min(endTime, other.endTime);
    
    return TimeRange(intersectionStart, intersectionEnd);
}

TimeRange TimeRange::union_range(const TimeRange& other) const {
    auto unionStart = std::min(startTime, other.startTime);
    auto unionEnd = std::max(endTime, other.endTime);
    
    return TimeRange(unionStart, unionEnd);
}

bool TimeRange::isValid() const {
    return startTime.isValid() && endTime.isValid() && startTime <= endTime;
}

std::vector<TimeRange> TimeRange::split(std::chrono::seconds interval) const {
    std::vector<TimeRange> result;
    
    if (!isValid() || interval.count() <= 0) {
        return result;
    }
    
    auto current = startTime;
    while (current < endTime) {
        auto next = current + interval;
        if (next > endTime) {
            next = endTime;
        }
        
        result.emplace_back(current, next);
        current = next;
    }
    
    return result;
}

std::vector<TimeRange> TimeRange::splitToChunks(size_t chunkCount) const {
    std::vector<TimeRange> result;
    
    if (!isValid() || chunkCount == 0) {
        return result;
    }
    
    auto totalDuration = getDuration();
    auto chunkDuration = std::chrono::seconds(totalDuration.count() / static_cast<long long>(chunkCount));
    
    auto current = startTime;
    for (size_t i = 0; i < chunkCount; ++i) {
        auto next = (i == chunkCount - 1) ? endTime : current + chunkDuration;
        result.emplace_back(current, next);
        current = next;
    }
    
    return result;
}

std::string TimeRange::serialize() const {
    return startTime.serialize() + "~" + endTime.serialize();
}

TimeRange TimeRange::deserialize(const std::string& data) {
    auto pos = data.find('~');
    if (pos == std::string::npos) {
        return TimeRange{};
    }
    
    auto startStr = data.substr(0, pos);
    auto endStr = data.substr(pos + 1);
    
    return TimeRange{
        CalendarTime::deserialize(startStr),
        CalendarTime::deserialize(endStr)
    };
}

bool TimeRange::operator==(const TimeRange& other) const {
    return startTime == other.startTime && endTime == other.endTime;
}

bool TimeRange::operator!=(const TimeRange& other) const {
    return !(*this == other);
}

bool TimeRange::operator<(const TimeRange& other) const {
    return startTime < other.startTime;
}

// =============================================================================
// TimeResolutionInfo 实现
// =============================================================================

std::string TimeResolutionInfo::getQualityDescription() const {
    std::ostringstream oss;
    
    if (isHighQuality()) {
        oss << "优秀：高质量规律时间序列";
    } else if (regularityRatio > 0.8) {
        oss << "良好：时间序列基本规律";
    } else if (regularityRatio > 0.5) {
        oss << "一般：时间序列存在一些不规律性";
    } else {
        oss << "较差：时间序列高度不规律";
    }
    
    oss << " (规律性: " << std::fixed << std::setprecision(1) << (regularityRatio * 100) << "%)";
    
    if (hasSignificantGaps()) {
        oss << ", 包含 " << gaps.size() << " 个时间间隙";
    }
    
    return oss.str();
}

std::string TimeResolutionInfo::serialize() const {
    std::ostringstream oss;
    oss << "{"
        << "\"nominalResolution\":" << nominalResolution.count() << ","
        << "\"actualResolution\":" << actualResolution.count() << ","
        << "\"minInterval\":" << minInterval.count() << ","
        << "\"maxInterval\":" << maxInterval.count() << ","
        << "\"regularityRatio\":" << regularityRatio << ","
        << "\"isRegular\":" << (isRegular ? "true" : "false") << ","
        << "\"totalTimePoints\":" << totalTimePoints << ","
        << "\"gapCount\":" << gaps.size()
        << "}";
    return oss.str();
}

TimeResolutionInfo TimeResolutionInfo::deserialize(const std::string& data) {
    // 简化的JSON解析实现
    TimeResolutionInfo result;
    
    try {
        // 解析各个字段
        auto parseField = [&data](const std::string& fieldName) -> std::string {
            auto pos = data.find("\"" + fieldName + "\":");
            if (pos != std::string::npos) {
                auto valueStart = data.find(':', pos) + 1;
                auto valueEnd = data.find(',', valueStart);
                if (valueEnd == std::string::npos) {
                    valueEnd = data.find('}', valueStart);
                }
                if (valueEnd != std::string::npos) {
                    return data.substr(valueStart, valueEnd - valueStart);
                }
            }
            return "";
        };
        
        auto nominalStr = parseField("nominalResolution");
        if (!nominalStr.empty()) {
            result.nominalResolution = std::chrono::seconds(std::stoll(nominalStr));
        }
        
        auto actualStr = parseField("actualResolution");
        if (!actualStr.empty()) {
            result.actualResolution = std::chrono::seconds(std::stoll(actualStr));
        }
        
        auto regularityStr = parseField("regularityRatio");
        if (!regularityStr.empty()) {
            result.regularityRatio = std::stod(regularityStr);
        }
        
        auto totalPointsStr = parseField("totalTimePoints");
        if (!totalPointsStr.empty()) {
            result.totalTimePoints = std::stoull(totalPointsStr);
        }
        
        auto isRegularStr = parseField("isRegular");
        result.isRegular = (isRegularStr == "true");
        
    } catch (const std::exception&) {
        // 解析失败，返回默认值
        result = TimeResolutionInfo{};
    }
    
    return result;
}

// =============================================================================
// TimeAxisInfo 实现
// =============================================================================

bool TimeAxisInfo::isValid() const {
    return !name.empty() && 
           !units.empty() && 
           referenceTime.isValid() && 
           timeRange.isValid() && 
           numberOfTimePoints > 0;
}

std::string TimeAxisInfo::getDescription() const {
    std::ostringstream oss;
    oss << "时间轴 '" << name << "': "
        << numberOfTimePoints << " 个时间点, "
        << "单位: " << units << ", "
        << "日历: " << calendarType << ", "
        << "范围: " << timeRange.toString();
    
    if (isUnlimited) {
        oss << " (无限维)";
    }
    
    return oss.str();
}

// =============================================================================
// TimeCoverageAnalysis 实现
// =============================================================================

std::string TimeCoverageAnalysis::getQualityReport() const {
    std::ostringstream oss;
    
    oss << "时间覆盖分析报告:\n";
    oss << "  总覆盖范围: " << totalCoverage.toString() << "\n";
    oss << "  覆盖率: " << std::fixed << std::setprecision(1) << (coverageRatio * 100) << "%\n";
    oss << "  数据源数量: " << totalDataSources << "\n";
    oss << "  时间间隙数量: " << gaps.size() << "\n";
    oss << "  重叠区域数量: " << overlaps.size() << "\n";
    
    if (hasGoodCoverage()) {
        oss << "  质量评级: 优秀\n";
    } else if (coverageRatio > 0.6) {
        oss << "  质量评级: 良好\n";
    } else if (coverageRatio > 0.4) {
        oss << "  质量评级: 一般\n";
    } else {
        oss << "  质量评级: 较差\n";
    }
    
    oss << "  整体分辨率: " << overallResolution.getQualityDescription();
    
    return oss.str();
}

// =============================================================================
// TimeExtractionOptions 实现
// =============================================================================

bool TimeExtractionOptions::isValid() const {
    return !preferredCalendar.empty() && 
           !timeZone.empty() && 
           maxTimeGap.count() > 0 && 
           maxTimePoints > 0 && 
           batchSize > 0;
}

std::string TimeExtractionOptions::toString() const {
    std::ostringstream oss;
    oss << "TimeExtractionOptions{"
        << "calendar='" << preferredCalendar << "'"
        << ", timezone='" << timeZone << "'"
        << ", strict=" << (strictMode ? "true" : "false")
        << ", maxGap=" << maxTimeGap.count() << "s"
        << ", maxPoints=" << maxTimePoints
        << ", batchSize=" << batchSize
        << "}";
    return oss.str();
}

} // namespace oscean::common_utils::time 