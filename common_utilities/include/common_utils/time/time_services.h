/**
 * @file time_services.h
 * @brief 时间处理模块统一对外服务接口
 * 
 * 🎯 设计原则：
 * ✅ 唯一对外接口 - 外部只需包含此文件
 * ✅ 隐藏内部实现 - 不暴露具体实现类
 * ✅ 统一时间服务 - 所有时间操作通过此接口
 * ✅ 支持依赖注入 - 支持外部时间源
 * ✅ 环境适配 - 根据环境自动优化配置
 * 
 * 🚨 重要：这是时间处理模块的唯一推荐对外接口
 * 
 * ❌ 禁止直接使用的类：
 * - TimeExtractorFactory（通过 getTimeExtractor() 获取）
 * - TimeResolutionAnalyzer（通过 analyzeTimeResolution() 获取）
 * - CalendarUtils（通过 formatTime() 等方法获取）
 * - 其他任何具体实现类
 * 
 * ✅ 正确使用方式：
 * @code
 * // 创建时间服务
 * auto timeService = TimeServicesFactory::createDefault();
 * 
 * // 解析和格式化时间
 * auto calTime = timeService->parseTime("2024-01-01", "yyyy-MM-dd");
 * std::string chineseTime = timeService->formatToChinese(calTime);
 * 
 * // 分析时间分辨率
 * std::vector<CalendarTime> times = {...};
 * auto resolution = timeService->analyzeTimeResolution(times);
 * std::string resolutionDesc = timeService->formatResolutionToChinese(resolution);
 * 
 * // 格式化时间持续时间
 * std::chrono::seconds duration{3661};
 * std::string durationDesc = timeService->formatDurationToChinese(duration);
 * @endcode
 */

#pragma once

#include "time_types.h"
#include "time_interfaces.h"
#include "time_resolution.h"
#include "time_range.h"
#include "time_calendar.h"

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <map>
#include <boost/optional.hpp>

namespace oscean::common_utils::time {

/**
 * @brief 时间序列统计信息
 */
struct TimeSeriesStatistics {
    size_t count = 0;                              // 时间点总数
    std::chrono::seconds averageInterval{0};       // 平均间隔
    std::chrono::seconds minInterval{0};           // 最小间隔
    std::chrono::seconds maxInterval{0};           // 最大间隔
    double regularityScore = 0.0;                  // 规律性评分 [0-1]
    size_t gapCount = 0;                          // 间隙数量
    std::vector<std::chrono::seconds> gaps;        // 间隙列表
    TimeRange coverage;                           // 覆盖范围
    
    // 质量评估
    bool isHighRegularity() const { return regularityScore > 0.9; }
    bool hasGaps() const { return gapCount > 0; }
    std::string getQualityDescription() const;
};

/**
 * @brief 时间服务配置
 */
struct TimeServiceConfiguration {
    std::string defaultCalendar = "gregorian";     // 默认日历类型
    std::string defaultTimeZone = "UTC";           // 默认时区
    std::string defaultLanguage = "zh_CN";         // 默认语言（中文）
    bool enableCaching = true;                     // 启用缓存
    bool strictMode = false;                       // 严格模式
    size_t maxTimePoints = 100000;                // 最大时间点数
    std::chrono::seconds cacheExpiry{3600};       // 缓存过期时间
    
    static TimeServiceConfiguration createDefault();
    static TimeServiceConfiguration createForTesting();
    static TimeServiceConfiguration createForPerformance();
};

/**
 * @brief 时间服务统计信息
 */
struct TimeServiceStatistics {
    size_t totalTimesParsed = 0;                   // 解析时间总数
    size_t totalTimesFormatted = 0;                // 格式化时间总数
    size_t totalResolutionAnalyses = 0;            // 分辨率分析总数
    double averageParsingTime = 0.0;               // 平均解析时间(ms)
    double averageFormattingTime = 0.0;            // 平均格式化时间(ms)
    double cacheHitRatio = 0.0;                    // 缓存命中率
    size_t errorCount = 0;                         // 错误总数
    std::chrono::seconds uptime{0};                // 运行时间
};

/**
 * @brief 时间服务主接口 - 提供所有时间处理功能的统一访问点
 */
class ITimeService {
public:
    virtual ~ITimeService() = default;
    
    // === 核心时间操作 ===
    
    /**
     * @brief 解析时间字符串为CalendarTime
     */
    virtual boost::optional<CalendarTime> parseTime(const std::string& timeString, 
                                                  const std::string& format = "") = 0;
    
    /**
     * @brief 将CalendarTime格式化为中文字符串
     */
    virtual std::string formatToChinese(const CalendarTime& time) = 0;
    
    /**
     * @brief 将CalendarTime格式化为完整中文字符串
     */
    virtual std::string formatToChineseFull(const CalendarTime& time) = 0;
    
    /**
     * @brief 格式化时间持续时间为中文字符串
     */
    virtual std::string formatDurationToChinese(std::chrono::seconds duration) = 0;
    
    /**
     * @brief 格式化时间范围为中文字符串
     */
    virtual std::string formatTimeRangeToChinese(const TimeRange& range) = 0;
    
    // === 时间分辨率分析 ===
    
    /**
     * @brief 分析时间序列分辨率
     */
    virtual TimeResolutionInfo analyzeTimeResolution(const std::vector<CalendarTime>& times) = 0;
    
    /**
     * @brief 格式化时间分辨率信息为中文字符串
     */
    virtual std::string formatResolutionToChinese(const TimeResolutionInfo& info) = 0;
    
    /**
     * @brief 优化时间分辨率
     */
    virtual TimeResolutionInfo optimizeTimeResolution(const std::vector<CalendarTime>& times) = 0;
    
    // === 时间范围操作 ===
    
    /**
     * @brief 分析时间范围
     */
    virtual TimeRange analyzeTimeRange(const std::vector<CalendarTime>& times) = 0;
    
    /**
     * @brief 验证时间范围
     */
    virtual bool validateTimeRange(const TimeRange& range) = 0;
    
    /**
     * @brief 分割时间范围
     */
    virtual std::vector<TimeRange> splitTimeRange(const TimeRange& range, size_t segments) = 0;
    
    // === 时间序列分析 ===
    
    /**
     * @brief 分析时间序列统计信息
     */
    virtual TimeSeriesStatistics analyzeTimeSeries(const std::vector<CalendarTime>& times) = 0;
    
    /**
     * @brief 检测时间序列质量
     */
    virtual std::string assessTimeQuality(const std::vector<CalendarTime>& times) = 0;
    
    // === 多格式时间提取 ===
    
    /**
     * @brief 从NetCDF文件提取时间信息
     */
    virtual boost::optional<std::vector<CalendarTime>> extractTimesFromNetCDF(
        const std::string& filePath, const std::string& timeVarName = "") = 0;
    
    /**
     * @brief 从CSV文件提取时间信息
     */
    virtual boost::optional<std::vector<CalendarTime>> extractTimesFromCSV(
        const std::string& filePath, const std::string& timeColumnName = "") = 0;
    
    // === CF时间转换 ===
    
    /**
     * @brief CF时间转换为CalendarTime
     */
    virtual boost::optional<CalendarTime> convertCFTime(double cfValue, 
                                                      const std::string& units) = 0;
    
    /**
     * @brief CalendarTime转换为CF时间
     */
    virtual boost::optional<double> convertToCFTime(const CalendarTime& time, 
                                                  const std::string& units) = 0;
    
    // === 服务管理 ===
    
    /**
     * @brief 获取服务统计信息
     */
    virtual TimeServiceStatistics getStatistics() const = 0;
    
    /**
     * @brief 重置统计信息
     */
    virtual void resetStatistics() = 0;
    
    /**
     * @brief 清理缓存
     */
    virtual void clearCache() = 0;
    
    /**
     * @brief 设置配置
     */
    virtual void setConfiguration(const TimeServiceConfiguration& config) = 0;
    
    /**
     * @brief 获取配置
     */
    virtual TimeServiceConfiguration getConfiguration() const = 0;
};

/**
 * @brief 时间服务工厂 - 创建和管理时间服务实例
 */
class TimeServicesFactory {
public:
    /**
     * @brief 创建默认时间服务实例
     */
    static std::unique_ptr<ITimeService> createDefault();
    
    /**
     * @brief 创建针对特定配置的时间服务实例
     */
    static std::unique_ptr<ITimeService> createWithConfiguration(const TimeServiceConfiguration& config);
    
    /**
     * @brief 创建测试用时间服务实例
     */
    static std::unique_ptr<ITimeService> createForTesting();
    
    /**
     * @brief 创建高性能时间服务实例
     */
    static std::unique_ptr<ITimeService> createForPerformance();
    
    /**
     * @brief 创建专门用于海洋数据的时间服务实例
     */
    static std::unique_ptr<ITimeService> createForOceanData();
    
    /**
     * @brief 获取全局单例时间服务实例
     */
    static ITimeService& getGlobalInstance();
    
    /**
     * @brief 关闭全局实例
     */
    static void shutdownGlobalInstance();
    
    static boost::optional<CalendarTime> parseQuick(const std::string& timeString);
};

/**
 * @brief 便捷的时间处理工具类 - 提供静态方法快速访问常用功能
 */
class TimeUtils {
public:
    // === 快速格式化方法 ===
    
    /**
     * @brief 快速将时间格式化为中文字符串
     */
    static std::string toChinese(const CalendarTime& time);
    
    /**
     * @brief 快速将持续时间格式化为中文字符串
     */
    static std::string durationToChinese(std::chrono::seconds duration);
    
    /**
     * @brief 快速解析时间字符串（使用全局服务实例）
     */
    static boost::optional<CalendarTime> parseQuick(const std::string& timeString);
    
    /**
     * @brief 快速分析时间分辨率
     */
    static TimeResolutionInfo analyzeResolutionQuick(const std::vector<CalendarTime>& times);
    
    /**
     * @brief 快速质量评估
     */
    static std::string assessQualityQuick(const std::vector<CalendarTime>& times);
    
    // === 快速转换方法 ===
    
    /**
     * @brief 系统时间点转CalendarTime
     */
    static CalendarTime fromTimePoint(const std::chrono::system_clock::time_point& tp);
    
    /**
     * @brief CalendarTime转系统时间点
     */
    static std::chrono::system_clock::time_point toTimePoint(const CalendarTime& time);
    
    /**
     * @brief 当前时间为CalendarTime
     */
    static CalendarTime now();
    
    /**
     * @brief 创建时间范围
     */
    static TimeRange createRange(const CalendarTime& start, const CalendarTime& end);
};

/**
 * @brief 时间分辨率枚举 - 支持中文
 */
enum class TemporalResolution {
    SECOND = 1,      // 秒
    MINUTE = 60,     // 分
    HOUR = 3600,     // 时
    DAY = 86400,     // 日
    WEEK = 604800,   // 周
    MONTH = 2592000, // 月（30天）
    YEAR = 31536000  // 年（365天）
};

/**
 * @brief 中文时间格式转换工具类
 */
class ChineseTimeFormatter {
public:
    /**
     * @brief 将ISO时间字符串转换为中文格式
     * @param isoTime ISO格式时间字符串，如："2024-01-01T08:30:00Z"
     * @return 中文格式时间字符串
     */
    static std::string formatChineseTime(const std::string& isoTime);
    
    /**
     * @brief 将中文时间字符串转换为Unix时间戳
     * @param chineseTime 中文格式时间字符串
     * @return Unix时间戳
     */
    static std::time_t parseChineseTime(const std::string& chineseTime);
    
    /**
     * @brief 将时间分辨率秒数转换为中文描述
     * @param seconds 时间分辨率（秒）
     * @return 中文描述，如："日"、"月"、"年"等
     */
    static std::string formatTemporalResolutionChinese(double seconds);
    
    /**
     * @brief 将中文时间分辨率转换为秒数
     * @param chineseResolution 中文时间分辨率，如："日"、"月"、"年"
     * @return 对应的秒数
     */
    static double parseTemporalResolutionChinese(const std::string& chineseResolution);

private:
    static const std::map<std::string, double> chineseResolutionMap_;
    static const std::map<double, std::string> resolutionChineseMap_;
};

} // namespace oscean::common_utils::time 