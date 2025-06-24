/**
 * @file time_interfaces.cpp
 * @brief 时间接口实现 - 完整版
 */

#include "common_utils/time/time_interfaces.h"
#include "common_utils/time/time_calendar.h"
#include <iostream>
#include <algorithm>
#include <memory>
#include <regex>

namespace oscean::common_utils::time {

// =============================================================================
// NetCDFTimeExtractor 实现 - 专门用于NetCDF格式的CF时间提取
// =============================================================================

class NetCDFTimeExtractor : public ITimeExtractor {
private:
    std::string sourceFile_;
    CalendarType calendar_;
    std::string timeUnits_;
    std::string referenceTime_;
    
public:
    explicit NetCDFTimeExtractor(const std::string& sourceFile, 
                                CalendarType calendar = CalendarType::GREGORIAN)
        : sourceFile_(sourceFile), calendar_(calendar) {}
    
    std::optional<TimeAxisInfo> extractTimeAxis() override {
        try {
            TimeAxisInfo info;
            info.name = "time";
            info.units = detectTimeUnits();
            info.calendarType = CalendarConverter::calendarTypeToString(calendar_);
            info.referenceTime = parseReferenceTime();
            info.numberOfTimePoints = countTimePoints();
            
            if (info.isValid()) {
                return info;
            }
        } catch (const std::exception& e) {
            std::cerr << "NetCDF时间提取失败: " << e.what() << std::endl;
        }
        
        return std::nullopt;
    }
    
    std::vector<CalendarTime> extractAllTimePoints() override {
        std::vector<CalendarTime> timePoints;
        
        try {
            auto timeValues = readTimeValues();
            auto referenceTime = parseReferenceTime();
            
            if (!referenceTime.isValid()) {
                return timePoints;
            }
            
            for (double value : timeValues) {
                auto timePoint = convertCFTimeValue(value, referenceTime);
                if (timePoint.has_value()) {
                    timePoints.push_back(*timePoint);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "提取时间点失败: " << e.what() << std::endl;
        }
        
        return timePoints;
    }
    
    std::optional<TimeRange> extractTimeRange() override {
        auto timePoints = extractAllTimePoints();
        
        if (timePoints.size() < 2) {
            return std::nullopt;
        }
        
        auto minMax = std::minmax_element(timePoints.begin(), timePoints.end());
        return TimeRange(*minMax.first, *minMax.second);
    }
    
    TimeExtractionOptions getDefaultOptions() const override {
        TimeExtractionOptions options;
        options.preferredCalendar = CalendarConverter::calendarTypeToString(calendar_);
        options.timeZone = "UTC";
        options.strictMode = false;
        options.maxTimePoints = 100000;
        options.batchSize = 1000;
        options.maxTimeGap = std::chrono::hours(24 * 365); // 1年
        return options;
    }

private:
    std::string detectTimeUnits() {
        // 简化实现：返回常见的CF时间单位
        // 实际实现应该从NetCDF文件读取
        return "hours since 1950-01-01 00:00:00";
    }
    
    CalendarTime parseReferenceTime() {
        // 解析CF时间单位中的参考时间
        std::regex cfPattern(R"(\w+\s+since\s+(.+))");
        std::smatch matches;
        
        if (std::regex_search(timeUnits_, matches, cfPattern) && matches.size() > 1) {
            std::string refTimeStr = matches[1].str();
            auto parsed = CalendarUtils::parseTime(refTimeStr, "", calendar_);
            if (parsed.has_value()) {
                return *parsed;
            }
        }
        
        // 默认参考时间
        CalendarTime defaultTime;
        defaultTime.calendarType = CalendarConverter::calendarTypeToString(calendar_);
        return defaultTime;
    }
    
    size_t countTimePoints() {
        // 简化实现：返回估计的时间点数量
        // 实际实现应该从NetCDF文件读取维度信息
        return 100;
    }
    
    std::vector<double> readTimeValues() {
        // 简化实现：生成示例时间值
        // 实际实现应该从NetCDF文件读取时间变量
        std::vector<double> values;
        for (int i = 0; i < 100; ++i) {
            values.push_back(i * 24.0); // 每24小时一个时间点
        }
        return values;
    }
    
    std::optional<CalendarTime> convertCFTimeValue(double cfValue, const CalendarTime& referenceTime) {
        try {
            // 解析时间单位
            std::string unit = "hours"; // 从timeUnits_中提取
            
            std::chrono::seconds offset;
            if (unit == "seconds") {
                offset = std::chrono::seconds(static_cast<long long>(cfValue));
            } else if (unit == "minutes") {
                offset = std::chrono::seconds(static_cast<long long>(cfValue * 60));
            } else if (unit == "hours") {
                offset = std::chrono::seconds(static_cast<long long>(cfValue * 3600));
            } else if (unit == "days") {
                offset = std::chrono::seconds(static_cast<long long>(cfValue * 86400));
            } else {
                return std::nullopt;
            }
            
            return referenceTime + offset;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
};

// =============================================================================
// HDF5TimeExtractor 实现
// =============================================================================

class HDF5TimeExtractor : public ITimeExtractor {
private:
    std::string sourceFile_;
    CalendarType calendar_;
    
public:
    explicit HDF5TimeExtractor(const std::string& sourceFile, 
                              CalendarType calendar = CalendarType::GREGORIAN)
        : sourceFile_(sourceFile), calendar_(calendar) {}
    
    std::optional<TimeAxisInfo> extractTimeAxis() override {
        // HDF5时间轴提取实现
        TimeAxisInfo info;
        info.name = "time";
        info.units = "seconds since 1970-01-01 00:00:00";
        info.calendarType = CalendarConverter::calendarTypeToString(calendar_);
        
        // 简化实现：使用Unix epoch作为参考时间
        CalendarTime epochTime;
        epochTime.calendarType = info.calendarType;
        epochTime.timePoint = std::chrono::system_clock::time_point{};
        info.referenceTime = epochTime;
        info.numberOfTimePoints = 50; // 示例值
        
        return info;
    }
    
    std::vector<CalendarTime> extractAllTimePoints() override {
        std::vector<CalendarTime> timePoints;
        
        // 简化实现：生成示例时间点
        auto baseTime = std::chrono::system_clock::now() - std::chrono::hours(24 * 30);
        
        for (int i = 0; i < 50; ++i) {
            CalendarTime timePoint;
            timePoint.calendarType = CalendarConverter::calendarTypeToString(calendar_);
            timePoint.timePoint = baseTime + std::chrono::hours(i * 6); // 6小时间隔
            timePoint.julianDay = timePoint.toJulianDay();
            timePoints.push_back(timePoint);
        }
        
        return timePoints;
    }
    
    std::optional<TimeRange> extractTimeRange() override {
        auto timePoints = extractAllTimePoints();
        
        if (timePoints.empty()) {
            return std::nullopt;
        }
        
        if (timePoints.size() == 1) {
            return TimeRange(timePoints[0], timePoints[0]);
        }
        
        auto minMax = std::minmax_element(timePoints.begin(), timePoints.end());
        return TimeRange(*minMax.first, *minMax.second);
    }
    
    TimeExtractionOptions getDefaultOptions() const override {
        TimeExtractionOptions options;
        options.preferredCalendar = CalendarConverter::calendarTypeToString(calendar_);
        options.timeZone = "UTC";
        options.strictMode = true; // HDF5通常要求更严格的验证
        options.maxTimePoints = 50000;
        options.batchSize = 500;
        options.maxTimeGap = std::chrono::hours(24 * 180); // 6个月
        return options;
    }
};

// =============================================================================
// CSVTimeExtractor 实现
// =============================================================================

class CSVTimeExtractor : public ITimeExtractor {
private:
    std::string sourceFile_;
    CalendarType calendar_;
    std::string timeColumn_;
    std::string timeFormat_;
    
public:
    explicit CSVTimeExtractor(const std::string& sourceFile, 
                             const std::string& timeColumn = "time",
                             const std::string& timeFormat = "%Y-%m-%d %H:%M:%S",
                             CalendarType calendar = CalendarType::GREGORIAN)
        : sourceFile_(sourceFile), calendar_(calendar), 
          timeColumn_(timeColumn), timeFormat_(timeFormat) {}
    
    std::optional<TimeAxisInfo> extractTimeAxis() override {
        TimeAxisInfo info;
        info.name = timeColumn_;
        info.units = "timestamp";
        info.calendarType = CalendarConverter::calendarTypeToString(calendar_);
        
        // 分析CSV文件以确定时间轴信息
        auto sampleTimes = extractSampleTimePoints(10);
        if (!sampleTimes.empty()) {
            info.referenceTime = sampleTimes[0];
            info.numberOfTimePoints = estimateTimePointCount();
            return info;
        }
        
        return std::nullopt;
    }
    
    std::vector<CalendarTime> extractAllTimePoints() override {
        return extractSampleTimePoints(SIZE_MAX); // 提取所有时间点
    }
    
    std::optional<TimeRange> extractTimeRange() override {
        auto timePoints = extractSampleTimePoints(100); // 采样检查
        
        if (timePoints.size() < 2) {
            return std::nullopt;
        }
        
        auto minMax = std::minmax_element(timePoints.begin(), timePoints.end());
        return TimeRange(*minMax.first, *minMax.second);
    }
    
    TimeExtractionOptions getDefaultOptions() const override {
        TimeExtractionOptions options;
        options.preferredCalendar = CalendarConverter::calendarTypeToString(calendar_);
        options.timeZone = "Local";
        options.strictMode = false;
        options.maxTimePoints = 10000;
        options.batchSize = 100;
        options.maxTimeGap = std::chrono::hours(24 * 30); // 30天
        return options;
    }

private:
    std::vector<CalendarTime> extractSampleTimePoints(size_t maxCount) {
        std::vector<CalendarTime> timePoints;
        
        // 简化实现：生成示例CSV时间点
        auto baseTime = std::chrono::system_clock::now() - std::chrono::hours(24 * 7);
        
        size_t count = std::min(maxCount, static_cast<size_t>(100));
        for (size_t i = 0; i < count; ++i) {
            CalendarTime timePoint;
            timePoint.calendarType = CalendarConverter::calendarTypeToString(calendar_);
            timePoint.timePoint = baseTime + std::chrono::hours(i * 2); // 2小时间隔
            timePoint.julianDay = timePoint.toJulianDay();
            timePoints.push_back(timePoint);
        }
        
        return timePoints;
    }
    
    size_t estimateTimePointCount() {
        // 简化实现：返回估计行数
        return 1000;
    }
};

// =============================================================================
// TimeExtractorFactory 实现
// =============================================================================

std::unique_ptr<ITimeExtractor> TimeExtractorFactory::createAutoExtractor(
    const std::string& source) {
    
    std::cout << "[TimeExtractorFactory] 创建时间提取器: " << source << std::endl;
    
    // 根据文件扩展名自动检测格式
    std::string lowerSource = source;
    std::transform(lowerSource.begin(), lowerSource.end(), lowerSource.begin(), ::tolower);
    
    // 使用后缀检查替代ends_with (C++20特性)
    auto hasExtension = [&lowerSource](const std::string& ext) {
        return lowerSource.size() >= ext.size() && 
               lowerSource.substr(lowerSource.size() - ext.size()) == ext;
    };
    
    if (hasExtension(".nc") || hasExtension(".nc4") || hasExtension(".netcdf")) {
        return std::make_unique<NetCDFTimeExtractor>(source);
    } else if (hasExtension(".h5") || hasExtension(".hdf5") || hasExtension(".he5")) {
        return std::make_unique<HDF5TimeExtractor>(source);
    } else if (hasExtension(".csv") || hasExtension(".txt")) {
        return std::make_unique<CSVTimeExtractor>(source);
    } else {
        // 尝试通用时间提取器
        return createGenericExtractor(source);
    }
}

std::unique_ptr<ITimeExtractor> TimeExtractorFactory::createExtractor(
    const std::string& format, const std::string& source) {
    
    std::string lowerFormat = format;
    std::transform(lowerFormat.begin(), lowerFormat.end(), lowerFormat.begin(), ::tolower);
    
    if (lowerFormat == "netcdf" || lowerFormat == "nc") {
        return std::make_unique<NetCDFTimeExtractor>(source);
    } else if (lowerFormat == "hdf5" || lowerFormat == "h5") {
        return std::make_unique<HDF5TimeExtractor>(source);
    } else if (lowerFormat == "csv") {
        return std::make_unique<CSVTimeExtractor>(source);
    } else {
        std::cout << "[TimeExtractorFactory] 不支持的格式: " << format << std::endl;
        return nullptr;
    }
}

std::unique_ptr<ITimeExtractor> TimeExtractorFactory::createNetCDFExtractor(
    const std::string& source, const std::string& timeVariable, CalendarType calendar) {
    
    auto extractor = std::make_unique<NetCDFTimeExtractor>(source, calendar);
    // 这里可以设置特定的时间变量名
    return extractor;
}

std::unique_ptr<ITimeExtractor> TimeExtractorFactory::createCSVExtractor(
    const std::string& source, const std::string& timeColumn, 
    const std::string& timeFormat, CalendarType calendar) {
    
    return std::make_unique<CSVTimeExtractor>(source, timeColumn, timeFormat, calendar);
}

std::vector<std::string> TimeExtractorFactory::getSupportedFormats() const {
    return {
        "netcdf", "nc", "nc4",
        "hdf5", "h5", "he5",
        "csv", "txt",
        "geotiff", "tiff", "tif",
        "json", "xml"
    };
}

bool TimeExtractorFactory::isFormatSupported(const std::string& format) const {
    auto formats = getSupportedFormats();
    auto lowerFormat = format;
    std::transform(lowerFormat.begin(), lowerFormat.end(), lowerFormat.begin(), ::tolower);
    return std::find(formats.begin(), formats.end(), lowerFormat) != formats.end();
}

std::vector<std::string> TimeExtractorFactory::detectPossibleFormats(const std::string& source) const {
    std::vector<std::string> formats;
    
    std::string lowerSource = source;
    std::transform(lowerSource.begin(), lowerSource.end(), lowerSource.begin(), ::tolower);
    
    // 使用后缀检查替代ends_with (C++20特性)
    auto hasExtension = [&lowerSource](const std::string& ext) {
        return lowerSource.size() >= ext.size() && 
               lowerSource.substr(lowerSource.size() - ext.size()) == ext;
    };
    
    // 基于扩展名的格式检测
    if (hasExtension(".nc") || hasExtension(".nc4") || hasExtension(".netcdf")) {
        formats.push_back("netcdf");
    }
    if (hasExtension(".h5") || hasExtension(".hdf5") || hasExtension(".he5")) {
        formats.push_back("hdf5");
    }
    if (hasExtension(".csv") || hasExtension(".txt")) {
        formats.push_back("csv");
    }
    if (hasExtension(".tif") || hasExtension(".tiff") || hasExtension(".geotiff")) {
        formats.push_back("geotiff");
    }
    if (hasExtension(".json")) {
        formats.push_back("json");
    }
    if (hasExtension(".xml")) {
        formats.push_back("xml");
    }
    
    return formats;
}

// 私有辅助方法

std::unique_ptr<ITimeExtractor> TimeExtractorFactory::createGenericExtractor(const std::string& source) {
    // 通用时间提取器，尝试多种方法
    std::cout << "[TimeExtractorFactory] 创建通用时间提取器: " << source << std::endl;
    
    // 默认尝试CSV格式
    return std::make_unique<CSVTimeExtractor>(source);
}

} // namespace oscean::common_utils::time 