/**
 * @file time_services.cpp
 * @brief 时间处理模块统一服务接口实现
 */

#include "common_utils/time/time_services.h"
#include "common_utils/time/time_types.h"
#include "common_utils/time/time_interfaces.h"
#include "common_utils/time/time_resolution.h"
#include "common_utils/time/time_range.h"
#include "common_utils/time/time_calendar.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/utilities/string_utils.h"
#include <boost/optional.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <numeric>

#include <mutex>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <regex>
#include <cstring>  // for memset
#include <map>

#ifdef _WIN32
#include <time.h>
#else
#include <time.h>
#endif

namespace oscean::common_utils::time {

// === TimeSeriesStatistics 方法实现 ===

std::string TimeSeriesStatistics::getQualityDescription() const {
    if (count == 0) {
        return "无数据";
    }
    
    std::ostringstream oss;
    if (regularityScore > 0.9) {
        oss << "高质量时间序列";
    } else if (regularityScore > 0.7) {
        oss << "良好质量时间序列";
    } else if (regularityScore > 0.5) {
        oss << "中等质量时间序列";
    } else {
        oss << "低质量时间序列";
    }
    
    oss << " (规律性: " << std::fixed << std::setprecision(1) << (regularityScore * 100) << "%)";
    
    if (hasGaps()) {
        oss << ", 存在 " << gapCount << " 个间隙";
    }
    
    return oss.str();
}

// === 时间服务配置实现 ===

TimeServiceConfiguration TimeServiceConfiguration::createDefault() {
    TimeServiceConfiguration config;
    config.defaultCalendar = "gregorian";
    config.defaultTimeZone = "UTC";
    config.defaultLanguage = "zh_CN";
    config.enableCaching = true;
    config.strictMode = false;
    config.maxTimePoints = 100000;
    config.cacheExpiry = std::chrono::seconds{3600};
    return config;
}

TimeServiceConfiguration TimeServiceConfiguration::createForTesting() {
    auto config = createDefault();
    config.enableCaching = false;
    config.strictMode = true;
    config.maxTimePoints = 1000;
    return config;
}

TimeServiceConfiguration TimeServiceConfiguration::createForPerformance() {
    auto config = createDefault();
    config.enableCaching = true;
    config.maxTimePoints = 1000000;
    config.cacheExpiry = std::chrono::seconds{7200};
    return config;
}

// === 时间服务实现类 ===

class TimeServiceImpl : public ITimeService {
private:
    TimeServiceConfiguration config_;
    mutable TimeServiceStatistics stats_;
    mutable std::mutex mutex_;
    std::chrono::steady_clock::time_point startTime_;
    
    // 缓存
    std::unordered_map<std::string, CalendarTime> parseCache_;
    std::unordered_map<std::string, std::string> formatCache_;
    
    // 内部服务组件
    std::unique_ptr<TimeResolutionAnalyzer> resolutionAnalyzer_;

public:
    explicit TimeServiceImpl(const TimeServiceConfiguration& config)
        : config_(config), startTime_(std::chrono::steady_clock::now()) {
        
        // 初始化内部组件
        resolutionAnalyzer_ = std::make_unique<TimeResolutionAnalyzer>();
        
        LOG_INFO("时间服务已初始化，语言: " + config_.defaultLanguage);
    }
    
    // === 核心时间操作实现 ===
    
    boost::optional<CalendarTime> parseTime(const std::string& timeString, 
                                          const std::string& format) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.totalTimesParsed++;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // 检查缓存
            if (config_.enableCaching) {
                auto cacheKey = timeString + "|" + format;
                auto it = parseCache_.find(cacheKey);
                if (it != parseCache_.end()) {
                    return it->second;
                }
            }
            
            boost::optional<CalendarTime> result;
            CalendarType calendarType = CalendarConverter::parseCalendarType(config_.defaultCalendar);
            
            if (format.empty()) {
                // 尝试自动格式检测
                auto std_result = CalendarUtils::fromISO8601(timeString, calendarType);
                if (std_result) { result = *std_result; } else { result = boost::none; }

                if (!result.has_value()) {
                    // 如果ISO8601失败，尝试其他常见格式
                    std::vector<std::string> commonFormats = {
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d",
                        "%Y/%m/%d %H:%M:%S",
                        "%Y/%m/%d",
                        "%d/%m/%Y %H:%M:%S",
                        "%d/%m/%Y"
                    };
                    
                    for (const auto& fmt : commonFormats) {
                        auto std_res_loop = CalendarUtils::parseTime(timeString, fmt, calendarType);
                        if (std_res_loop) { result = *std_res_loop; } else { result = boost::none; }
                        if (result.has_value()) {
                            break;
                        }
                    }
                }
            } else {
                // 使用指定格式解析
                auto std_result = CalendarUtils::parseTime(timeString, format, calendarType);
                if (std_result) { result = *std_result; } else { result = boost::none; }
            }
            
            // 更新缓存
            if (config_.enableCaching && result.has_value()) {
                auto cacheKey = timeString + "|" + format;
                parseCache_[cacheKey] = result.value();
            }
            
            // 更新统计
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            stats_.averageParsingTime = (stats_.averageParsingTime + duration.count() / 1000.0) / 2.0;
            
            return result;
            
        } catch (const std::exception& e) {
            stats_.errorCount++;
            LOG_ERROR("时间解析失败: " + std::string(e.what()));
            return boost::none;
        }
    }
    
    std::string formatToChinese(const CalendarTime& time) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.totalTimesFormatted++;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // 转换为本地时间
            auto tt = std::chrono::system_clock::to_time_t(time.timePoint);
            auto tm = *std::localtime(&tt);
            
            // 格式化为中文
            std::ostringstream oss;
            oss << (tm.tm_year + 1900) << "年"
                << std::setfill('0') << std::setw(2) << (tm.tm_mon + 1) << "月"
                << std::setfill('0') << std::setw(2) << tm.tm_mday << "日 "
                << std::setfill('0') << std::setw(2) << tm.tm_hour << "时"
                << std::setfill('0') << std::setw(2) << tm.tm_min << "分"
                << std::setfill('0') << std::setw(2) << tm.tm_sec << "秒";
            
            // 更新统计
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            stats_.averageFormattingTime = (stats_.averageFormattingTime + duration.count() / 1000.0) / 2.0;
            
            return oss.str();
            
        } catch (const std::exception& e) {
            stats_.errorCount++;
            LOG_ERROR("时间格式化失败: " + std::string(e.what()));
            return "时间格式化失败";
        }
    }
    
    std::string formatToChineseFull(const CalendarTime& time) override {
        std::lock_guard<std::mutex> lock(mutex_);
        
        try {
            auto tt = std::chrono::system_clock::to_time_t(time.timePoint);
            auto tm = *std::localtime(&tt);
            
            // 添加星期和更多信息
            const char* weekdays[] = {"星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"};
            
            std::ostringstream oss;
            oss << (tm.tm_year + 1900) << "年"
                << std::setfill('0') << std::setw(2) << (tm.tm_mon + 1) << "月"
                << std::setfill('0') << std::setw(2) << tm.tm_mday << "日"
                << " " << weekdays[tm.tm_wday] << " "
                << std::setfill('0') << std::setw(2) << tm.tm_hour << "时"
                << std::setfill('0') << std::setw(2) << tm.tm_min << "分"
                << std::setfill('0') << std::setw(2) << tm.tm_sec << "秒";
            
            if (!time.calendarType.empty() && time.calendarType != "gregorian") {
                oss << " (" << time.calendarType << "历)";
            }
            
            return oss.str();
            
        } catch (const std::exception& e) {
            stats_.errorCount++;
            return "完整时间格式化失败";
        }
    }
    
    std::string formatDurationToChinese(std::chrono::seconds duration) override {
        std::lock_guard<std::mutex> lock(mutex_);
        
        try {
            auto totalSeconds = duration.count();
            
            if (totalSeconds < 60) {
                return std::to_string(totalSeconds) + "秒";
            } else if (totalSeconds < 3600) {
                auto minutes = totalSeconds / 60;
                auto seconds = totalSeconds % 60;
                return std::to_string(minutes) + "分" + std::to_string(seconds) + "秒";
            } else if (totalSeconds < 86400) {
                auto hours = totalSeconds / 3600;
                auto minutes = (totalSeconds % 3600) / 60;
                auto seconds = totalSeconds % 60;
                return std::to_string(hours) + "小时" + std::to_string(minutes) + "分" + std::to_string(seconds) + "秒";
            } else {
                auto days = totalSeconds / 86400;
                auto hours = (totalSeconds % 86400) / 3600;
                auto minutes = (totalSeconds % 3600) / 60;
                return std::to_string(days) + "天" + std::to_string(hours) + "小时" + std::to_string(minutes) + "分";
            }
            
        } catch (const std::exception& e) {
            stats_.errorCount++;
            return "持续时间格式化失败";
        }
    }
    
    std::string formatTimeRangeToChinese(const TimeRange& range) override {
        try {
            std::string startStr = formatToChinese(range.startTime);
            std::string endStr = formatToChinese(range.endTime);
            auto duration = range.getDuration();
            std::string durationStr = formatDurationToChinese(duration);
            
            return "从 " + startStr + " 到 " + endStr + "，持续 " + durationStr;
            
        } catch (const std::exception& e) {
            stats_.errorCount++;
            return "时间范围格式化失败";
        }
    }
    
    // === 时间分辨率分析实现 ===
    
    TimeResolutionInfo analyzeTimeResolution(const std::vector<CalendarTime>& times) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.totalResolutionAnalyses++;
        
        try {
            // 使用正确的方法名
            return resolutionAnalyzer_->analyzeResolution(times);
        } catch (const std::exception& e) {
            stats_.errorCount++;
            LOG_ERROR("时间分辨率分析失败: " + std::string(e.what()));
            return TimeResolutionInfo{};
        }
    }
    
    std::string formatResolutionToChinese(const TimeResolutionInfo& info) override {
        try {
            std::string nominalStr = formatDurationToChinese(info.nominalResolution);
            std::string actualStr = formatDurationToChinese(info.actualResolution);
            
            std::ostringstream oss;
            oss << "时间分辨率分析报告:\n";
            oss << "  名义分辨率: " << nominalStr << "\n";
            oss << "  实际分辨率: " << actualStr << "\n";
            oss << "  规律性比率: " << std::fixed << std::setprecision(2) << (info.regularityRatio * 100) << "%\n";
            oss << "  数据质量: " << (info.isHighQuality() ? "高质量" : "需要改进") << "\n";
            oss << "  总时间点数: " << info.totalTimePoints << "\n";
            
            if (info.hasSignificantGaps()) {
                oss << "  检测到 " << info.gaps.size() << " 个时间间隙\n";
            }
            
            return oss.str();
            
        } catch (const std::exception& e) {
            stats_.errorCount++;
            return "分辨率信息格式化失败";
        }
    }
    
    TimeResolutionInfo optimizeTimeResolution(const std::vector<CalendarTime>& times) override {
        try {
            // 使用TimeResolutionOptimizer来优化分辨率
            TimeResolutionOptimizer optimizer;
            return optimizer.optimizeResolution(times);
        } catch (const std::exception& e) {
            stats_.errorCount++;
            return TimeResolutionInfo{};
        }
    }
    
    // === 其他方法的完整实现 ===
    
    TimeRange analyzeTimeRange(const std::vector<CalendarTime>& times) override {
        if (times.empty()) return TimeRange{};
        
        auto minTime = *std::min_element(times.begin(), times.end());
        auto maxTime = *std::max_element(times.begin(), times.end());
        
        return TimeRange{minTime, maxTime};
    }
    
    bool validateTimeRange(const TimeRange& range) override {
        return range.isValid() && !range.isEmpty();
    }
    
    std::vector<TimeRange> splitTimeRange(const TimeRange& range, size_t segments) override {
        return range.splitToChunks(segments);
    }
    
    TimeSeriesStatistics analyzeTimeSeries(const std::vector<CalendarTime>& times) override {
        // 实现时间序列统计分析
        TimeSeriesStatistics stats;
        if (times.empty()) {
            return stats;
        }
        
        try {
            stats.count = times.size();
            
            // 计算时间间隔
            std::vector<std::chrono::seconds> intervals;
            for (size_t i = 1; i < times.size(); ++i) {
                auto interval = times[i].timePoint - times[i-1].timePoint;
                auto intervalSeconds = std::chrono::duration_cast<std::chrono::seconds>(interval);
                intervals.push_back(intervalSeconds);
            }
            
            if (!intervals.empty()) {
                // 计算统计信息
                auto minInterval = *std::min_element(intervals.begin(), intervals.end());
                auto maxInterval = *std::max_element(intervals.begin(), intervals.end());
                
                auto totalSeconds = std::accumulate(intervals.begin(), intervals.end(), 
                                                   std::chrono::seconds{0});
                auto avgInterval = totalSeconds / static_cast<int64_t>(intervals.size());
                
                stats.minInterval = minInterval;
                stats.maxInterval = maxInterval;
                stats.averageInterval = avgInterval;
                
                // 计算规律性评分
                auto avgCount = avgInterval.count();
                if (avgCount > 0) {
                    double regularitySum = 0.0;
                    for (const auto& interval : intervals) {
                        double deviation = std::abs(interval.count() - avgCount) / static_cast<double>(avgCount);
                        regularitySum += std::max(0.0, 1.0 - deviation);
                    }
                    stats.regularityScore = regularitySum / intervals.size();
                }
                
                // 检测间隙
                auto threshold = avgInterval * 2; // 2倍平均间隔视为间隙
                for (const auto& interval : intervals) {
                    if (interval > threshold) {
                        stats.gaps.push_back(interval);
                    }
                }
                stats.gapCount = stats.gaps.size();
            }
            
            // 设置覆盖范围
            stats.coverage = TimeRange{times.front(), times.back()};
            
        } catch (const std::exception& e) {
            LOG_ERROR("时间序列分析失败: " + std::string(e.what()));
        }
        
        return stats;
    }
    
    std::string assessTimeQuality(const std::vector<CalendarTime>& times) override {
        auto resolution = analyzeTimeResolution(times);
        return resolution.getQualityDescription();
    }
    
    boost::optional<std::vector<CalendarTime>> extractTimesFromNetCDF(
        const std::string& filePath, const std::string& timeVarName = "") override {
        try {
            // 使用正确的参数调用TimeExtractorFactory
            auto extractor = TimeExtractorFactory::createNetCDFExtractor(filePath, timeVarName);
            if (extractor) {
                return extractor->extractAllTimePoints();
            }
            return boost::none;
        } catch (const std::exception& e) {
            stats_.errorCount++;
            LOG_ERROR("NetCDF时间提取失败: " + std::string(e.what()));
            return boost::none;
        }
    }
    
    boost::optional<std::vector<CalendarTime>> extractTimesFromCSV(
        const std::string& filePath, const std::string& timeColumnName = "") override {
        try {
            // 使用正确的参数调用TimeExtractorFactory
            auto extractor = TimeExtractorFactory::createCSVExtractor(filePath, timeColumnName);
            if (extractor) {
                return extractor->extractAllTimePoints();
            }
            return boost::none;
        } catch (const std::exception& e) {
            stats_.errorCount++;
            LOG_ERROR("CSV时间提取失败: " + std::string(e.what()));
            return boost::none;
        }
    }
    
    boost::optional<CalendarTime> convertCFTime(double cfValue, 
                                                const std::string& units) override {
        return CFTimeConverter::convertCFTime(cfValue, units);
    }
    
    boost::optional<double> convertToCFTime(const CalendarTime& time, 
                                            const std::string& units) override {
        return CFTimeConverter::convertToCFTime(time, units);
    }
    
    // === 服务管理实现 ===
    
    TimeServiceStatistics getStatistics() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto currentStats = stats_;
        currentStats.uptime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startTime_);
        currentStats.cacheHitRatio = calculateCacheHitRatio();
        return currentStats;
    }
    
    void resetStatistics() override {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = TimeServiceStatistics{};
        startTime_ = std::chrono::steady_clock::now();
    }
    
    void clearCache() override {
        std::lock_guard<std::mutex> lock(mutex_);
        parseCache_.clear();
        formatCache_.clear();
    }
    
    void setConfiguration(const TimeServiceConfiguration& config) override {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
    }
    
    TimeServiceConfiguration getConfiguration() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return config_;
    }

private:
    double calculateCacheHitRatio() const {
        if (stats_.totalTimesParsed == 0) return 0.0;
        // 简化的缓存命中率计算
        return static_cast<double>(parseCache_.size()) / stats_.totalTimesParsed;
    }
};

// === 工厂实现 ===

std::unique_ptr<ITimeService> TimeServicesFactory::createDefault() {
    return std::make_unique<TimeServiceImpl>(TimeServiceConfiguration::createDefault());
}

std::unique_ptr<ITimeService> TimeServicesFactory::createWithConfiguration(
    const TimeServiceConfiguration& config) {
    return std::make_unique<TimeServiceImpl>(config);
}

std::unique_ptr<ITimeService> TimeServicesFactory::createForTesting() {
    return std::make_unique<TimeServiceImpl>(TimeServiceConfiguration::createForTesting());
}

std::unique_ptr<ITimeService> TimeServicesFactory::createForPerformance() {
    return std::make_unique<TimeServiceImpl>(TimeServiceConfiguration::createForPerformance());
}

std::unique_ptr<ITimeService> TimeServicesFactory::createForOceanData() {
    auto config = TimeServiceConfiguration::createForPerformance();
    config.defaultCalendar = "gregorian";
    config.maxTimePoints = 1000000;  // 海洋数据通常有大量时间点
    return std::make_unique<TimeServiceImpl>(config);
}

// 全局单例实现
static std::unique_ptr<ITimeService> globalTimeService_;
static std::mutex globalMutex_;

ITimeService& TimeServicesFactory::getGlobalInstance() {
    std::lock_guard<std::mutex> lock(globalMutex_);
    if (!globalTimeService_) {
        globalTimeService_ = createDefault();
    }
    return *globalTimeService_;
}

void TimeServicesFactory::shutdownGlobalInstance() {
    std::lock_guard<std::mutex> lock(globalMutex_);
    globalTimeService_.reset();
}

// === 便捷工具类实现 ===

std::string TimeUtils::toChinese(const CalendarTime& time) {
    return TimeServicesFactory::getGlobalInstance().formatToChinese(time);
}

std::string TimeUtils::durationToChinese(std::chrono::seconds duration) {
    return TimeServicesFactory::getGlobalInstance().formatDurationToChinese(duration);
}

boost::optional<CalendarTime> TimeUtils::parseQuick(const std::string& timeString) {
    return TimeServicesFactory::getGlobalInstance().parseTime(timeString);
}

TimeResolutionInfo TimeUtils::analyzeResolutionQuick(const std::vector<CalendarTime>& times) {
    return TimeServicesFactory::getGlobalInstance().analyzeTimeResolution(times);
}

std::string TimeUtils::assessQualityQuick(const std::vector<CalendarTime>& times) {
    return TimeServicesFactory::getGlobalInstance().assessTimeQuality(times);
}

CalendarTime TimeUtils::fromTimePoint(const std::chrono::system_clock::time_point& tp) {
    return CalendarTime{tp};
}

std::chrono::system_clock::time_point TimeUtils::toTimePoint(const CalendarTime& time) {
    return time.timePoint;
}

CalendarTime TimeUtils::now() {
    return CalendarTime{std::chrono::system_clock::now()};
}

TimeRange TimeUtils::createRange(const CalendarTime& start, const CalendarTime& end) {
    return TimeRange{start, end};
}

// 🎯 新增：中文时间格式转换实现
const std::map<std::string, double> ChineseTimeFormatter::chineseResolutionMap_ = {
    {"秒", 1.0}, {"分", 60.0}, {"时", 3600.0},
    {"日", 86400.0}, {"逐日", 86400.0}, {"天", 86400.0},
    {"周", 604800.0}, {"旬", 864000.0}, {"月", 2592000.0},
    {"季", 7776000.0}, {"年", 31536000.0}
};

const std::map<double, std::string> ChineseTimeFormatter::resolutionChineseMap_ = {
    {1.0, "秒"}, {60.0, "分"}, {3600.0, "时"},
    {86400.0, "日"}, {604800.0, "周"}, {864000.0, "旬"},
    {2592000.0, "月"}, {7776000.0, "季"}, {31536000.0, "年"}
};

std::string ChineseTimeFormatter::formatChineseTime(const std::string& isoTime) {
    auto calTime = CalendarUtils::fromISO8601(isoTime);
    if (!calTime.has_value()) {
        return "无效时间";
    }

    auto timePoint = calTime.value().timePoint;
    std::time_t timestamp = std::chrono::system_clock::to_time_t(timePoint);

    tm localTime;
#ifdef _WIN32
    localtime_s(&localTime, &timestamp);
#else
    localtime_r(&timestamp, &localTime);
#endif

    std::ostringstream oss;
    oss << (localTime.tm_year + 1900) << "年"
        << std::setfill('0') << std::setw(2) << (localTime.tm_mon + 1) << "月"
        << std::setfill('0') << std::setw(2) << localTime.tm_mday << "日 "
        << std::setfill('0') << std::setw(2) << localTime.tm_hour << "时"
        << std::setfill('0') << std::setw(2) << localTime.tm_min << "分"
        << std::setfill('0') << std::setw(2) << localTime.tm_sec << "秒";
    return oss.str();
}

std::time_t ChineseTimeFormatter::parseChineseTime(const std::string& chineseTime) {
    std::tm tm = {};
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;

    // The format is "YYYY年MM月DD日 HH时MM分SS秒"
    // sscanf is perfect for this.
    int parsed = std::sscanf(chineseTime.c_str(), "%d年%d月%d日 %d时%d分%d秒",
                            &year, &month, &day, &hour, &minute, &second);

    if (parsed == 6) {
        tm.tm_year = year - 1900;
        tm.tm_mon = month - 1;
        tm.tm_mday = day;
        tm.tm_hour = hour;
        tm.tm_min = minute;
        tm.tm_sec = second;
        tm.tm_isdst = -1; // Let mktime figure out DST
        return std::mktime(&tm);
    }

    return -1; // Parsing failed
}

std::string ChineseTimeFormatter::formatTemporalResolutionChinese(double seconds) {
    if (seconds <= 0) return "无效分辨率";

    // 找到最接近的时间分辨率
    double bestMatch = 1.0;
    double minDiff = std::abs(seconds - 1.0);
    
    for (const auto& [value, chinese] : resolutionChineseMap_) {
        double diff = std::abs(seconds - value);
        if (diff < minDiff) {
            minDiff = diff;
            bestMatch = value;
        }
    }
    
    auto it = resolutionChineseMap_.find(bestMatch);
    if (it != resolutionChineseMap_.end()) {
        return it->second;
    }
    
    // 如果没有精确匹配，返回近似描述
    if (seconds < 60) {
        return "秒";
    } else if (seconds < 3600) {
        return "分";
    } else if (seconds < 86400) {
        return "时";
    } else if (seconds < 604800) {
        return "日";
    } else if (seconds < 2592000) {
        return "周";
    } else if (seconds < 31536000) {
        return "月";
    } else {
        return "年";
    }
}

double ChineseTimeFormatter::parseTemporalResolutionChinese(const std::string& chineseResolution) {
    auto it = chineseResolutionMap_.find(chineseResolution);
    if (it != chineseResolutionMap_.end()) {
        return it->second;
    }
    return -1.0; // 未找到
}

namespace {
// 内部辅助函数，用于解析CF时间单位字符串
struct CFTimeUnits {
    // ... existing code ...
};
}

} // namespace oscean::common_utils::time

// =============================================================================
// CFTimeConverter 实现
// =============================================================================

namespace oscean::common_utils::time {

boost::optional<CalendarTime> CFTimeConverter::parseReferenceTime(const std::string& units) {
    // 🎯 第一步：解析CF时间单位格式，提取参考时间字符串
    // 例如: "days since 1970-01-01 00:00:00" -> "1970-01-01 00:00:00"
    std::regex cfPattern(R"(\w+\s+since\s+(.+))", std::regex_constants::icase);
    std::smatch matches;
    
    if (!std::regex_search(units, matches, cfPattern) || matches.size() < 2) {
        LOG_ERROR("CF时间单位格式不匹配: " + units);
        return boost::none;
    }
    
    std::string referenceTimeStr = matches[1].str();
    LOG_DEBUG("提取的参考时间字符串: " + referenceTimeStr);
    
    // 🎯 第二步：解析参考时间字符串，增强对特殊格式的支持
    std::tm refTime = {};
    bool parsed = false;
    
    // 🎯 增强的正则表达式，支持更多CF时间格式
    // 支持格式：YYYY-MM-DD HH:MM:SS.sss, YYYY-MM-DD HH:MM:S.s 等
    std::regex fullPattern(R"((\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?))?\s*(?:UTC?)?(?:\s*[+-]\d{2}:?\d{2})?)", std::regex_constants::icase);
    std::smatch timeMatches;
    
    if (std::regex_search(referenceTimeStr, timeMatches, fullPattern)) {
        try {
            int year = std::stoi(timeMatches[1].str());
            int month = std::stoi(timeMatches[2].str());
            int day = std::stoi(timeMatches[3].str());
            
            // 对于早期年份（如1900年），需要特殊处理
            if (year < 1970) {
                LOG_DEBUG("检测到早期年份: " + std::to_string(year) + "，使用特殊处理逻辑");
            }
            
            // 验证日期范围 - 🔧 修复：扩大年份范围以支持地质时间尺度
            if (year < -4713 || year > 9999 || month < 1 || month > 12 || day < 1 || day > 31) {
                LOG_ERROR("日期参数超出合理范围: {}-{}-{}", year, month, day);
                return boost::none;
            }
            
            refTime.tm_year = year - 1900;
            refTime.tm_mon = month - 1;
            refTime.tm_mday = day;
            
            // 解析时间部分（如果存在）
            if (timeMatches[4].matched && timeMatches[5].matched && timeMatches[6].matched) {
                int hour = std::stoi(timeMatches[4].str());
                int minute = std::stoi(timeMatches[5].str());
                
                // 🎯 处理秒数部分，去掉小数部分
                std::string secondsStr = timeMatches[6].str();
                size_t dotPos = secondsStr.find('.');
                if (dotPos != std::string::npos) {
                    secondsStr = secondsStr.substr(0, dotPos);  // 去掉小数部分
                }
                int second = std::stoi(secondsStr);
                
                // 验证时间范围
                if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59) {
                    LOG_ERROR("时间部分超出有效范围: " + std::to_string(hour) + ":" + 
                             std::to_string(minute) + ":" + std::to_string(second));
                    return boost::none;
                }
                
                refTime.tm_hour = hour;
                refTime.tm_min = minute;
                refTime.tm_sec = second;
                
                LOG_DEBUG("成功解析时间部分: " + std::to_string(hour) + ":" + 
                         std::to_string(minute) + ":" + std::to_string(second));
            } else {
                // 如果没有时间部分，设置为00:00:00
                refTime.tm_hour = 0;
                refTime.tm_min = 0;
                refTime.tm_sec = 0;
            }
            
            refTime.tm_isdst = -1; // 让系统决定是否为夏令时
            parsed = true;
            
            std::string parsedStr = "CF时间解析成功: " + std::to_string(refTime.tm_year + 1900) + "-" + 
                                   std::to_string(refTime.tm_mon + 1) + "-" + std::to_string(refTime.tm_mday) + 
                                   " " + std::to_string(refTime.tm_hour) + ":" + 
                                   std::to_string(refTime.tm_min) + ":" + std::to_string(refTime.tm_sec);
            LOG_INFO(parsedStr);
            
        } catch (const std::exception& e) {
            LOG_ERROR("CF时间数值解析失败: " + std::string(e.what()));
            parsed = false;
        }
    }
    
    if (!parsed) {
        LOG_ERROR("所有时间解析方法都失败了，输入: " + referenceTimeStr);
        return boost::none;
    }
    
    // 转换为time_t
    std::time_t refTimeT;
    
    // 🎯 修复早期年份处理逻辑
    if (refTime.tm_year + 1900 < 1970) {
        // 对于1970年之前的日期，使用更精确的计算方法
        LOG_DEBUG("处理早期年份: {}", refTime.tm_year + 1900);
        
        // 使用标准库的mktime进行转换，然后调整为UTC
        std::tm utcTime = refTime;
        utcTime.tm_isdst = 0; // 明确设置为非夏令时
        
        // 先尝试使用标准转换
#ifdef _WIN32
        refTimeT = _mkgmtime(&utcTime);
#else
        refTimeT = timegm(&utcTime);
#endif
        
        // 如果标准转换失败（返回-1），使用手动计算
        if (refTimeT == static_cast<std::time_t>(-1)) {
            LOG_DEBUG("标准转换失败，使用手动计算");
            
            // 手动计算从1970年1月1日开始的秒数偏移
            int year = refTime.tm_year + 1900;
            
            // 计算从1970年到目标年份的天数
            int64_t totalDays = 0;
            
            if (year < 1970) {
                // 计算1970年之前的年份
                for (int y = year; y < 1970; ++y) {
                    bool isLeap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
                    totalDays -= isLeap ? 366 : 365;
                }
            } else {
                // 计算1970年之后的年份
                for (int y = 1970; y < year; ++y) {
                    bool isLeap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
                    totalDays += isLeap ? 366 : 365;
                }
            }
            
            // 加上年内的天数
            // 计算月份天数
            static const int daysInMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            for (int m = 0; m < refTime.tm_mon; ++m) {
                totalDays += daysInMonth[m];
                // 处理闰年的2月
                if (m == 1) { // 2月
                    bool isLeap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                    if (isLeap) totalDays += 1;
                }
            }
            totalDays += refTime.tm_mday - 1; // tm_mday是1-based
            
            // 转换为秒数
            int64_t totalSeconds = totalDays * 24 * 3600;
            totalSeconds += refTime.tm_hour * 3600;
            totalSeconds += refTime.tm_min * 60;
            totalSeconds += refTime.tm_sec;
            
            refTimeT = static_cast<std::time_t>(totalSeconds);
            
            LOG_DEBUG("手动计算结果: 年份={}, 总天数={}, 总秒数={}, time_t={}", 
                     year, totalDays, totalSeconds, refTimeT);
        } else {
            LOG_DEBUG("标准转换成功: time_t={}", refTimeT);
        }
        
    } else {
        // 1970年及之后的年份，使用标准转换
#ifdef _WIN32
        refTimeT = _mkgmtime(&refTime);
#else
        refTimeT = timegm(&refTime);
#endif
        
        if (refTimeT == static_cast<std::time_t>(-1)) {
            LOG_ERROR("标准时间转换失败: {}-{}-{} {}:{}:{}", 
                     refTime.tm_year + 1900, refTime.tm_mon + 1, refTime.tm_mday,
                     refTime.tm_hour, refTime.tm_min, refTime.tm_sec);
            return boost::none;
        }
        
        LOG_DEBUG("标准年份转换成功: {} -> time_t={}", refTime.tm_year + 1900, refTimeT);
    }
    
    // 验证转换结果的合理性
    if (refTimeT < -2147483648LL || refTimeT > 2147483647LL) {
        LOG_WARN("时间转换结果超出合理范围: time_t={}, 可能存在计算错误", refTimeT);
        // 不返回错误，继续使用，但记录警告
    }
    
    auto resultTimePoint = std::chrono::system_clock::from_time_t(refTimeT);
    CalendarTime result(resultTimePoint);
    result.calendarType = "gregorian";
    
    // 验证结果
    auto verifyTime = std::chrono::system_clock::to_time_t(result.timePoint);
    LOG_DEBUG("CF时间解析完成: 输入='{}', 解析年份={}, time_t={}, 验证time_t={}", 
             referenceTimeStr, refTime.tm_year + 1900, refTimeT, verifyTime);
    
    return result;
}

std::chrono::seconds CFTimeConverter::parseTimeUnit(const std::string& units) {
    // 提取单位类型
    std::regex unitPattern(R"((\w+)\s+since)");
    std::smatch matches;
    
    if (!std::regex_search(units, matches, unitPattern) || matches.size() < 2) {
        return std::chrono::seconds(0);
    }
    
    std::string unitType = matches[1].str();
    std::transform(unitType.begin(), unitType.end(), unitType.begin(), ::tolower);
    
    // 转换为秒数
    if (unitType == "seconds" || unitType == "second") {
        return std::chrono::seconds(1);
    } else if (unitType == "minutes" || unitType == "minute") {
        return std::chrono::seconds(60);
    } else if (unitType == "hours" || unitType == "hour") {
        return std::chrono::seconds(3600);
    } else if (unitType == "days" || unitType == "day") {
        return std::chrono::seconds(86400); // 24 * 3600
    } else {
        return std::chrono::seconds(1); // 默认为秒
    }
}

bool CFTimeConverter::isValidCFTimeUnit(const std::string& units) {
    std::regex cfPattern(R"((\w+)\s+since\s+.+)");
    return std::regex_match(units, cfPattern);
}

std::vector<std::string> CFTimeConverter::getSupportedCalendars() {
    return {"gregorian", "standard", "proleptic_gregorian", "julian", "360_day", "365_day", "366_day"};
}

boost::optional<CalendarTime> CFTimeConverter::convertCFTime(double cfValue, 
                                                           const std::string& units) {
    try {
        // 解析参考时间
        auto referenceTime = parseReferenceTime(units);
        if (!referenceTime.has_value()) {
            return boost::none;
        }
        
        // 解析时间单位
        auto timeUnit = parseTimeUnit(units);
        if (timeUnit.count() == 0) {
            return boost::none;
        }
        
        // 计算实际时间
        auto offsetSeconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::duration<double>(cfValue) * timeUnit.count()
        );
        
        auto resultTime = referenceTime->timePoint + offsetSeconds;
        
        CalendarTime result(resultTime);
        result.calendarType = "gregorian";
        return result;
        
    } catch (const std::exception&) {
        return boost::none;
    }
}

boost::optional<double> CFTimeConverter::convertToCFTime(const CalendarTime& time, 
                                                      const std::string& units) {
    try {
        // 解析参考时间
        auto referenceTime = parseReferenceTime(units);
        if (!referenceTime.has_value()) {
            return boost::none;
        }
        
        // 解析时间单位
        auto timeUnit = parseTimeUnit(units);
        if (timeUnit.count() == 0) {
            return boost::none;
        }
        
        // 计算CF值
        auto timeDiff = time.timePoint - referenceTime->timePoint;
        auto secondsDiff = std::chrono::duration_cast<std::chrono::seconds>(timeDiff);
        
        double cfValue = static_cast<double>(secondsDiff.count()) / timeUnit.count();
        return cfValue;
        
    } catch (const std::exception&) {
        return boost::none;
    }
}

boost::optional<double> CFTimeConverter::calculateTimeResolution(const std::vector<double>& timeValues, const std::string& units) {
    if (timeValues.size() < 2) {
        return boost::none;
    }

    std::vector<double> diffs;
    diffs.reserve(timeValues.size() - 1);
    for (size_t i = 1; i < timeValues.size(); ++i) {
        diffs.push_back(timeValues[i] - timeValues[i - 1]);
    }

    if (diffs.empty()) {
        return boost::none;
    }

    double average_diff = std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();

    // 假设单位字符串格式为 "unit since YYYY-MM-DD HH:MM:SS"
    std::string unit_part = units.substr(0, units.find(' '));
    double multiplier = 1.0;
    if (unit_part == "days" || unit_part == "day") {
        multiplier = 24.0 * 3600.0;
    } else if (unit_part == "hours" || unit_part == "hour") {
        multiplier = 3600.0;
    } else if (unit_part == "minutes" || unit_part == "minute") {
        multiplier = 60.0;
    } else if (unit_part == "seconds" || unit_part == "second") {
        // multiplier is 1.0
    } else {
        // 不支持的单位
        return boost::none;
    }

    return average_diff * multiplier;
}

} // namespace oscean::common_utils::time 