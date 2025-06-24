#pragma once

/**
 * @file time_interfaces.h
 * @brief 时间提取器接口 - 格式无关的时间抽象
 * 
 * 提供完整的时间提取和处理接口
 */

#include "time_types.h"
#include "time_range.h"
#include "time_calendar.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <map>

// C++17兼容：统一使用boost::optional
#include <boost/optional.hpp>

namespace oscean::common_utils::time {

/**
 * @brief 时间提取器接口 - 格式特定的时间提取
 */
class ITimeExtractor {
public:
    virtual ~ITimeExtractor() = default;
    
    // 基础时间提取
    virtual std::optional<TimeAxisInfo> extractTimeAxis() = 0;
    virtual std::vector<CalendarTime> extractAllTimePoints() = 0;
    virtual std::optional<TimeRange> extractTimeRange() = 0;
    
    // 配置和选项
    virtual TimeExtractionOptions getDefaultOptions() const = 0;
};

/**
 * @brief 时间提取器工厂
 */
class TimeExtractorFactory {
public:
    TimeExtractorFactory() = default;
    virtual ~TimeExtractorFactory() = default;
    
    // 自动检测并创建提取器
    static std::unique_ptr<ITimeExtractor> createAutoExtractor(const std::string& source);
    
    // 格式特定的提取器创建
    static std::unique_ptr<ITimeExtractor> createExtractor(const std::string& format, const std::string& source);
    static std::unique_ptr<ITimeExtractor> createNetCDFExtractor(const std::string& source, 
                                                                 const std::string& timeVariable = "time", 
                                                                 CalendarType calendar = CalendarType::GREGORIAN);
    static std::unique_ptr<ITimeExtractor> createCSVExtractor(const std::string& source, 
                                                             const std::string& timeColumn = "time",
                                                             const std::string& timeFormat = "%Y-%m-%d %H:%M:%S",
                                                             CalendarType calendar = CalendarType::GREGORIAN);
    
    // 格式检测和支持
    std::vector<std::string> getSupportedFormats() const;
    bool isFormatSupported(const std::string& format) const;
    std::vector<std::string> detectPossibleFormats(const std::string& source) const;

private:
    // 私有辅助方法
    static std::unique_ptr<ITimeExtractor> createGenericExtractor(const std::string& source);
};

/**
 * @brief 时间元数据提取器接口 - 高级时间分析
 */
class ITimeMetadataExtractor {
public:
    virtual ~ITimeMetadataExtractor() = default;
    
    // 基础时间提取
    virtual std::optional<TimeRange> extractTimeRange() const = 0;
    virtual std::optional<TimeResolutionInfo> calculateTimeResolution() const = 0;
    virtual std::vector<CalendarTime> extractAllTimePoints() const = 0;
    
    // 验证和检查
    virtual bool hasValidTimeDimension() const = 0;
    virtual std::string getTimeDimensionName() const = 0;
    virtual std::string getFormatType() const = 0;
    virtual std::vector<std::string> getSupportedCalendars() const = 0;
    
    // 流式时间提取 (大数据支持)
    virtual void extractTimePointsStreaming(
        std::function<void(const std::vector<CalendarTime>&)> callback,
        size_t batchSize = 1000
    ) const = 0;
    
    // 时间质量评估
    virtual std::optional<TimeResolutionInfo> analyzeTimeQuality() const = 0;
    
    // 元数据获取
    virtual std::map<std::string, std::string> getTimeMetadata() const = 0;
    virtual std::string getCalendarType() const = 0;
    virtual std::string getTimeUnits() const = 0;
};

/**
 * @class CFTimeConverter
 * @brief 提供CF约定时间转换的静态工具函数
 */
class CFTimeConverter {
public:
    static boost::optional<CalendarTime> convertCFTime(double value, const std::string& units);

    static boost::optional<double> calculateTimeResolution(const std::vector<double>& timeValues, const std::string& units);

    static boost::optional<double> convertToCFTime(const CalendarTime& time, const std::string& units);
    
    static boost::optional<CalendarTime> parseReferenceTime(const std::string& units);
    static std::chrono::seconds parseTimeUnit(const std::string& units);
    
    static bool isValidCFTimeUnit(const std::string& units);
    static std::vector<std::string> getSupportedCalendars();
};

/**
 * @brief 时间格式检测器
 */
class TimeFormatDetector {
public:
    struct DetectedFormat {
        std::string format;
        double confidence;
        std::string description;
        std::vector<std::string> examples;
    };
    
    static std::vector<DetectedFormat> detectTimeFormats(const std::vector<std::string>& timeStrings);
    static std::optional<std::string> detectBestFormat(const std::vector<std::string>& timeStrings);
    static bool validateTimeFormat(const std::string& timeString, const std::string& format);
    
    static std::vector<std::string> getCommonFormats();
    static std::vector<std::string> getISO8601Formats();
    static std::vector<std::string> getCFCompatibleFormats();
};

/**
 * @brief 时间序列分析器
 */
class TimeSeriesAnalyzer {
public:
    struct TimeSeriesStats {
        size_t totalPoints;
        std::chrono::seconds minInterval;
        std::chrono::seconds maxInterval;
        std::chrono::seconds averageInterval;
        double regularityScore; // 0-1, 1为完全规律
        size_t gapCount;
        std::vector<std::chrono::seconds> gaps;
        TimeRange coverage;
    };
    
    static TimeSeriesStats analyzeTimeSeries(const std::vector<CalendarTime>& timePoints);
    static std::vector<TimeRange> findTimeGaps(const std::vector<CalendarTime>& timePoints, 
                                              std::chrono::seconds maxGap);
    static bool isRegularTimeSeries(const std::vector<CalendarTime>& timePoints, 
                                   double tolerance = 0.1);
    static std::chrono::seconds inferTimeStep(const std::vector<CalendarTime>& timePoints);
};

} // namespace oscean::common_utils::time 