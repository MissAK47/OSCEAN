#pragma once

/**
 * @file time_calendar.h
 * @brief 通用日历处理 - 支持多种日历系统
 * 
 * 提供格里高利历、儒略历等多种日历系统的转换和计算
 */

#include "time_types.h"
#include <string>
#include <vector>
#include <map>

namespace oscean::common_utils::time {

/**
 * @brief 支持的日历类型
 */
enum class CalendarType {
    GREGORIAN,      // 格里高利历
    JULIAN,         // 儒略历
    NOLEAP,         // 无闰年日历
    ALL_LEAP,       // 全闰年日历
    NONE,           // 无日历
    STANDARD,       // 标准日历
    PROLEPTIC_GREGORIAN // 向前推格里高利历
};

/**
 * @brief 日历信息结构
 */
struct CalendarInfo {
    CalendarType type;
    std::string name;
    std::string description;
    bool hasLeapYears;
    int daysInYear;             // 标准年天数
    int daysInLeapYear;         // 闰年天数
    std::vector<int> daysInMonth; // 各月天数
    
    // 构造函数
    CalendarInfo() : type(CalendarType::GREGORIAN), hasLeapYears(true), 
                     daysInYear(365), daysInLeapYear(366) {}
};

/**
 * @brief 日期组件结构
 */
struct DateComponents {
    int year;
    int month;      // 1-12
    int day;        // 1-31
    int hour;       // 0-23
    int minute;     // 0-59
    int second;     // 0-59
    int millisecond; // 0-999
    
    // 构造函数
    DateComponents() : year(1970), month(1), day(1), hour(0), 
                      minute(0), second(0), millisecond(0) {}
    
    // 验证
    bool isValid() const;
    std::string toString() const;
    
    // 静态工具方法
    static int getDaysInMonth(int year, int month);
    static bool isLeapYear(int year);
};

/**
 * @brief 日历转换器
 */
class CalendarConverter {
public:
    /**
     * @brief 获取日历信息
     */
    static CalendarInfo getCalendarInfo(CalendarType type);
    static CalendarInfo getCalendarInfo(const std::string& calendarName);
    
    /**
     * @brief 日历类型转换
     */
    static CalendarType parseCalendarType(const std::string& calendarName);
    static std::string calendarTypeToString(CalendarType type);
    
    /**
     * @brief 时间点转换
     */
    static CalendarTime convertToCalendar(const CalendarTime& time, CalendarType targetType);
    static CalendarTime convertFromCalendar(const CalendarTime& time, CalendarType sourceType);
    
    /**
     * @brief 日期组件转换
     */
    static DateComponents timePointToComponents(const CalendarTime& time, CalendarType calendar);
    static CalendarTime componentsToTimePoint(const DateComponents& components, CalendarType calendar);
    
    /**
     * @brief 儒略日转换
     */
    static double timePointToJulianDay(const CalendarTime& time);
    static CalendarTime julianDayToTimePoint(double julianDay);
    
    /**
     * @brief 闰年判断
     */
    static bool isLeapYear(int year, CalendarType calendar);
    static int getDaysInYear(int year, CalendarType calendar);
    static int getDaysInMonth(int year, int month, CalendarType calendar);
    
    /**
     * @brief 获取支持的日历列表
     */
    static std::vector<std::string> getSupportedCalendars();
    static bool isCalendarSupported(const std::string& calendarName);

private:
    /**
     * @brief 私有日历转换方法
     */
    static std::chrono::system_clock::time_point convertGregorianToJulian(
        const std::chrono::system_clock::time_point& gregorianTime);
    static std::chrono::system_clock::time_point convertJulianToGregorian(
        const std::chrono::system_clock::time_point& julianTime);
    static std::chrono::system_clock::time_point convertToNoLeapCalendar(
        const std::chrono::system_clock::time_point& time);
    static std::chrono::system_clock::time_point convertToAllLeapCalendar(
        const std::chrono::system_clock::time_point& time);
};

/**
 * @brief 日历工具类
 */
class CalendarUtils {
public:
    /**
     * @brief 时间格式化
     */
    static std::string formatTime(const CalendarTime& time, const std::string& format, 
                                CalendarType calendar = CalendarType::GREGORIAN);
    
    /**
     * @brief 时间解析
     */
    static std::optional<CalendarTime> parseTime(const std::string& timeString, 
                                                const std::string& format,
                                                CalendarType calendar = CalendarType::GREGORIAN);
    
    /**
     * @brief ISO8601格式支持
     */
    static std::string toISO8601(const CalendarTime& time, CalendarType calendar = CalendarType::GREGORIAN);
    static std::optional<CalendarTime> fromISO8601(const std::string& isoString, 
                                                   CalendarType calendar = CalendarType::GREGORIAN);
    
    /**
     * @brief 时间差计算
     */
    static std::chrono::seconds calculateDifference(const CalendarTime& start, const CalendarTime& end,
                                                   CalendarType calendar = CalendarType::GREGORIAN);
    
    /**
     * @brief 时间比较
     */
    static bool isEqual(const CalendarTime& time1, const CalendarTime& time2, 
                       CalendarType calendar, std::chrono::seconds tolerance = std::chrono::seconds(0));
    
    /**
     * @brief 时间验证
     */
    static bool validateTime(const CalendarTime& time, CalendarType calendar);
    static std::string getValidationError(const CalendarTime& time, CalendarType calendar);
    
    /**
     * @brief 获取时间精度
     */
    static std::chrono::seconds getMinimumResolution(CalendarType calendar);
    static std::chrono::seconds getMaximumResolution(CalendarType calendar);

private:
    /**
     * @brief 私有解析辅助方法
     */
    static std::optional<CalendarTime> parseISODate(const std::string& dateString, CalendarType calendar);
    static std::optional<CalendarTime> parseWithFormat(const std::string& timeString, 
                                                      const std::string& format, 
                                                      CalendarType calendar);
    static std::optional<CalendarTime> parseWithCommonFormats(const std::string& timeString, 
                                                             CalendarType calendar);
    static std::optional<CalendarTime> parseRelativeTime(const std::string& timeString, 
                                                        CalendarType calendar);
    static std::optional<CalendarTime> parseTimeOffset(const std::string& offsetString, 
                                                      const CalendarTime& baseTime, 
                                                      CalendarType calendar);
    static CalendarTime addTimeUnit(const CalendarTime& baseTime, 
                                   const std::string& unit, 
                                   int amount, 
                                   CalendarType calendar);
    static std::chrono::system_clock::time_point calculateTimePointForEarlyDate(int year, int month, int day);
};

/**
 * @brief 日历常量
 */
namespace CalendarConstants {
    // 儒略日相关常量
    constexpr double JULIAN_DAY_EPOCH = 2440588.0;  // Unix时间起点的儒略日
    constexpr double SECONDS_PER_DAY = 86400.0;     // 每天秒数
    
    // 日历转换常量
    constexpr int GREGORIAN_EPOCH_YEAR = 1582;      // 格里高利历开始年份
    constexpr int JULIAN_EPOCH_YEAR = -4712;        // 儒略历开始年份
    
    // 常用格式字符串
    const std::string ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%SZ";
    const std::string DATE_FORMAT = "%Y-%m-%d";
    const std::string TIME_FORMAT = "%H:%M:%S";
}

} // namespace oscean::common_utils::time 