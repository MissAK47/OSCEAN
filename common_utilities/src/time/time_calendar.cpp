/**
 * @file time_calendar.cpp
 * @brief 通用日历处理实现 - 完整版
 */

#include "common_utils/time/time_calendar.h"
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <array>

namespace oscean::common_utils::time {

// =============================================================================
// DateComponents 实现
// =============================================================================

bool DateComponents::isValid() const {
    return year >= 1 && year <= 9999 &&
           month >= 1 && month <= 12 &&
           day >= 1 && day <= getDaysInMonth(year, month) &&
           hour >= 0 && hour <= 23 &&
           minute >= 0 && minute <= 59 &&
           second >= 0 && second <= 59 &&
           millisecond >= 0 && millisecond <= 999;
}

std::string DateComponents::toString() const {
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(4) << year << "-"
        << std::setw(2) << month << "-"
        << std::setw(2) << day << " "
        << std::setw(2) << hour << ":"
        << std::setw(2) << minute << ":"
        << std::setw(2) << second << "."
        << std::setw(3) << millisecond;
    return oss.str();
}

int DateComponents::getDaysInMonth(int year, int month) {
    static const std::array<int, 12> daysInMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    
    if (month < 1 || month > 12) {
        return 0;
    }
    
    int days = daysInMonth[month - 1];
    
    // 2月闰年处理
    if (month == 2 && isLeapYear(year)) {
        days = 29;
    }
    
    return days;
}

bool DateComponents::isLeapYear(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

// =============================================================================
// CalendarConverter 实现
// =============================================================================

CalendarInfo CalendarConverter::getCalendarInfo(CalendarType type) {
    CalendarInfo info;
    info.type = type;
    
    switch (type) {
        case CalendarType::GREGORIAN:
            info.name = "gregorian";
            info.description = "Gregorian calendar with leap years";
            info.hasLeapYears = true;
            info.daysInYear = 365;
            info.daysInLeapYear = 366;
            info.daysInMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            break;
            
        case CalendarType::JULIAN:
            info.name = "julian";
            info.description = "Julian calendar";
            info.hasLeapYears = true;
            info.daysInYear = 365;
            info.daysInLeapYear = 366;
            info.daysInMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            break;
            
        case CalendarType::NOLEAP:
            info.name = "noleap";
            info.description = "Calendar without leap years (365_day)";
            info.hasLeapYears = false;
            info.daysInYear = 365;
            info.daysInLeapYear = 365;
            info.daysInMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            break;
            
        case CalendarType::ALL_LEAP:
            info.name = "all_leap";
            info.description = "Calendar with all leap years (366_day)";
            info.hasLeapYears = false; // 总是闰年，所以不需要判断
            info.daysInYear = 366;
            info.daysInLeapYear = 366;
            info.daysInMonth = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            break;
            
        case CalendarType::STANDARD:
            info.name = "standard";
            info.description = "Standard calendar (same as Gregorian)";
            info.hasLeapYears = true;
            info.daysInYear = 365;
            info.daysInLeapYear = 366;
            info.daysInMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            break;
            
        case CalendarType::PROLEPTIC_GREGORIAN:
            info.name = "proleptic_gregorian";
            info.description = "Proleptic Gregorian calendar";
            info.hasLeapYears = true;
            info.daysInYear = 365;
            info.daysInLeapYear = 366;
            info.daysInMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            break;
            
        default:
            info.name = "none";
            info.description = "No calendar specified";
            info.hasLeapYears = false;
            info.daysInYear = 365;
            info.daysInLeapYear = 365;
            break;
    }
    
    return info;
}

CalendarInfo CalendarConverter::getCalendarInfo(const std::string& calendarName) {
    auto type = parseCalendarType(calendarName);
    return getCalendarInfo(type);
}

CalendarType CalendarConverter::parseCalendarType(const std::string& calendarName) {
    std::string lowerName = calendarName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    
    if (lowerName == "gregorian" || lowerName == "standard") {
        return CalendarType::GREGORIAN;
    } else if (lowerName == "julian") {
        return CalendarType::JULIAN;
    } else if (lowerName == "noleap" || lowerName == "365_day") {
        return CalendarType::NOLEAP;
    } else if (lowerName == "all_leap" || lowerName == "366_day") {
        return CalendarType::ALL_LEAP;
    } else if (lowerName == "proleptic_gregorian") {
        return CalendarType::PROLEPTIC_GREGORIAN;
    } else {
        return CalendarType::NONE;
    }
}

std::string CalendarConverter::calendarTypeToString(CalendarType type) {
    auto info = getCalendarInfo(type);
    return info.name;
}

CalendarTime CalendarConverter::convertToCalendar(const CalendarTime& time, CalendarType targetType) {
    // 完整的日历转换实现
    if (time.calendarType == calendarTypeToString(targetType)) {
        return time; // 已经是目标日历类型
    }
    
    CalendarTime result = time;
    result.calendarType = calendarTypeToString(targetType);
    
    // 根据源日历和目标日历进行转换
    auto sourceType = parseCalendarType(time.calendarType);
    
    if (sourceType == CalendarType::GREGORIAN && targetType == CalendarType::JULIAN) {
        // 格里高利历到儒略历的转换
        result.timePoint = convertGregorianToJulian(time.timePoint);
    } else if (sourceType == CalendarType::JULIAN && targetType == CalendarType::GREGORIAN) {
        // 儒略历到格里高利历的转换
        result.timePoint = convertJulianToGregorian(time.timePoint);
    } else if (targetType == CalendarType::NOLEAP) {
        // 转换到无闰年日历：移除2月29日
        result.timePoint = convertToNoLeapCalendar(time.timePoint);
    } else if (targetType == CalendarType::ALL_LEAP) {
        // 转换到全闰年日历：确保每年都有2月29日
        result.timePoint = convertToAllLeapCalendar(time.timePoint);
    }
    
    // 更新儒略日
    result.julianDay = result.toJulianDay();
    
    return result;
}

CalendarTime CalendarConverter::convertFromCalendar(const CalendarTime& time, CalendarType sourceType) {
    // 从指定日历类型转换为标准格里高利历
    return convertToCalendar(time, CalendarType::GREGORIAN);
}

bool CalendarConverter::isLeapYear(int year, CalendarType calendar) {
    switch (calendar) {
        case CalendarType::GREGORIAN:
        case CalendarType::STANDARD:
        case CalendarType::PROLEPTIC_GREGORIAN:
            return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            
        case CalendarType::JULIAN:
            return year % 4 == 0;
            
        case CalendarType::NOLEAP:
            return false;
            
        case CalendarType::ALL_LEAP:
            return true;
            
        default:
            return false;
    }
}

int CalendarConverter::getDaysInYear(int year, CalendarType calendar) {
    return isLeapYear(year, calendar) ? 366 : 365;
}

int CalendarConverter::getDaysInMonth(int year, int month, CalendarType calendar) {
    if (month < 1 || month > 12) {
        return 0;
    }
    
    static const std::array<int, 12> daysInMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int days = daysInMonth[month - 1];
    
    // 2月特殊处理
    if (month == 2) {
        if (calendar == CalendarType::ALL_LEAP) {
            days = 29;
        } else if (calendar == CalendarType::NOLEAP) {
            days = 28;
        } else if (isLeapYear(year, calendar)) {
            days = 29;
        }
    }
    
    return days;
}

std::vector<std::string> CalendarConverter::getSupportedCalendars() {
    return {
        "gregorian", "julian", "noleap", "all_leap", 
        "standard", "proleptic_gregorian", "none"
    };
}

bool CalendarConverter::isCalendarSupported(const std::string& calendarName) {
    auto supported = getSupportedCalendars();
    auto lowerName = calendarName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    return std::find(supported.begin(), supported.end(), lowerName) != supported.end();
}

// 私有方法：具体的日历转换实现

std::chrono::system_clock::time_point CalendarConverter::convertGregorianToJulian(
    const std::chrono::system_clock::time_point& gregorianTime) {
    
    // 格里高利历和儒略历的差异主要在1582年10月4日之后
    auto time_t_val = std::chrono::system_clock::to_time_t(gregorianTime);
    auto tm_val = *std::gmtime(&time_t_val);
    
    int year = tm_val.tm_year + 1900;
    
    // 1582年10月15日之前，格里高利历和儒略历是相同的
    if (year < 1582 || (year == 1582 && tm_val.tm_mon < 9) || 
        (year == 1582 && tm_val.tm_mon == 9 && tm_val.tm_mday < 15)) {
        return gregorianTime;
    }
    
    // 计算格里高利历和儒略历的天数差异
    int centuryYears = (year / 100) - 15; // 从1500年开始计算
    int leapCenturies = centuryYears / 4; // 400年闰年规则
    int daysDifference = centuryYears - leapCenturies;
    
    // 调整时间点
    return gregorianTime + std::chrono::hours(24 * daysDifference);
}

std::chrono::system_clock::time_point CalendarConverter::convertJulianToGregorian(
    const std::chrono::system_clock::time_point& julianTime) {
    
    auto time_t_val = std::chrono::system_clock::to_time_t(julianTime);
    auto tm_val = *std::gmtime(&time_t_val);
    
    int year = tm_val.tm_year + 1900;
    
    if (year < 1582 || (year == 1582 && tm_val.tm_mon < 9) || 
        (year == 1582 && tm_val.tm_mon == 9 && tm_val.tm_mday < 15)) {
        return julianTime;
    }
    
    // 计算差异并调整
    int centuryYears = (year / 100) - 15;
    int leapCenturies = centuryYears / 4;
    int daysDifference = centuryYears - leapCenturies;
    
    return julianTime - std::chrono::hours(24 * daysDifference);
}

std::chrono::system_clock::time_point CalendarConverter::convertToNoLeapCalendar(
    const std::chrono::system_clock::time_point& time) {
    
    auto time_t_val = std::chrono::system_clock::to_time_t(time);
    auto tm_val = *std::gmtime(&time_t_val);
    
    // 如果是2月29日，调整为2月28日
    if (tm_val.tm_mon == 1 && tm_val.tm_mday == 29) { // 2月29日
        tm_val.tm_mday = 28;
        auto adjusted_time = std::mktime(&tm_val);
        return std::chrono::system_clock::from_time_t(adjusted_time);
    }
    
    // 如果是2月29日之后的日期，减去一天
    if (tm_val.tm_mon > 1 && isLeapYear(tm_val.tm_year + 1900, CalendarType::GREGORIAN)) {
        return time - std::chrono::hours(24);
    }
    
    return time;
}

std::chrono::system_clock::time_point CalendarConverter::convertToAllLeapCalendar(
    const std::chrono::system_clock::time_point& time) {
    
    auto time_t_val = std::chrono::system_clock::to_time_t(time);
    auto tm_val = *std::gmtime(&time_t_val);
    
    int year = tm_val.tm_year + 1900;
    
    // 如果原本不是闰年，但在2月28日之后，需要添加一天
    if (!isLeapYear(year, CalendarType::GREGORIAN) && 
        (tm_val.tm_mon > 1 || (tm_val.tm_mon == 1 && tm_val.tm_mday > 28))) {
        return time + std::chrono::hours(24);
    }
    
    return time;
}

// =============================================================================
// CalendarUtils 实现
// =============================================================================

std::string CalendarUtils::formatTime(const CalendarTime& time, const std::string& format, CalendarType calendar) {
    // 完整的时间格式化实现，考虑日历类型
    auto adjustedTime = CalendarConverter::convertToCalendar(time, calendar);
    
    auto time_t_value = std::chrono::system_clock::to_time_t(adjustedTime.timePoint);
    auto tm_value = *std::gmtime(&time_t_value);
    
    std::ostringstream ss;
    ss << std::put_time(&tm_value, format.c_str());
    return ss.str();
}

std::optional<CalendarTime> CalendarUtils::parseTime(const std::string& timeString, 
                                                    const std::string& format,
                                                    CalendarType calendar) {
    try {
        CalendarTime result;
        result.calendarType = CalendarConverter::calendarTypeToString(calendar);
        
        // 特殊处理："YYYY-MM-DD"格式的直接解析
        if (timeString.size() == 10 && timeString[4] == '-' && timeString[7] == '-') {
            auto parsedTime = parseISODate(timeString, calendar);
            if (parsedTime.has_value()) {
                return parsedTime;
            }
        }
        
        // 如果提供了格式，使用指定格式解析
        if (!format.empty()) {
            auto parsedTime = parseWithFormat(timeString, format, calendar);
            if (parsedTime.has_value()) {
                return parsedTime;
            }
        }
        
        // 尝试多种常见格式
        auto parsedTime = parseWithCommonFormats(timeString, calendar);
        if (parsedTime.has_value()) {
            return parsedTime;
        }
        
        // 尝试解析相对时间表达式
        auto relativeTime = parseRelativeTime(timeString, calendar);
        if (relativeTime.has_value()) {
            return relativeTime;
        }
        
        return std::nullopt;
        
    } catch (const std::exception& e) {
        return std::nullopt;
    }
}

std::string CalendarUtils::toISO8601(const CalendarTime& time, CalendarType calendar) {
    return formatTime(time, CalendarConstants::ISO8601_FORMAT, calendar);
}

std::optional<CalendarTime> CalendarUtils::fromISO8601(const std::string& isoString, CalendarType calendar) {
    return parseTime(isoString, CalendarConstants::ISO8601_FORMAT, calendar);
}

std::chrono::seconds CalendarUtils::calculateDifference(const CalendarTime& start, const CalendarTime& end,
                                                       CalendarType calendar) {
    // 完整的时间差计算，考虑日历差异
    auto startAdjusted = CalendarConverter::convertToCalendar(start, calendar);
    auto endAdjusted = CalendarConverter::convertToCalendar(end, calendar);
    
    return endAdjusted - startAdjusted;
}

bool CalendarUtils::validateTime(const CalendarTime& time, CalendarType calendar) {
    if (!time.isValid()) {
        return false;
    }
    
    // 检查时间是否在指定日历中有效
    auto time_t_val = std::chrono::system_clock::to_time_t(time.timePoint);
    auto tm_val = *std::gmtime(&time_t_val);
    
    int year = tm_val.tm_year + 1900;
    int month = tm_val.tm_mon + 1;
    int day = tm_val.tm_mday;
    
    // 检查日期在指定日历中是否有效
    if (month == 2 && day == 29) {
        // 2月29日需要特殊检查
        switch (calendar) {
            case CalendarType::NOLEAP:
                return false; // 无闰年日历不允许2月29日
            case CalendarType::ALL_LEAP:
                return true;  // 全闰年日历总是允许2月29日
            default:
                return CalendarConverter::isLeapYear(year, calendar);
        }
    }
    
    return true;
}

std::string CalendarUtils::getValidationError(const CalendarTime& time, CalendarType calendar) {
    if (!time.isValid()) {
        return "Invalid time point";
    }
    
    if (!validateTime(time, calendar)) {
        auto time_t_val = std::chrono::system_clock::to_time_t(time.timePoint);
        auto tm_val = *std::gmtime(&time_t_val);
        
        if (tm_val.tm_mon == 1 && tm_val.tm_mday == 29) { // 2月29日
            return "February 29 is not valid in " + CalendarConverter::calendarTypeToString(calendar) + " calendar";
        }
        
        return "Time is not valid in specified calendar";
    }
    
    return "";
}

// 私有辅助方法实现

std::optional<CalendarTime> CalendarUtils::parseISODate(const std::string& dateString, CalendarType calendar) {
    try {
        int year = std::stoi(dateString.substr(0, 4));
        int month = std::stoi(dateString.substr(5, 2));
        int day = std::stoi(dateString.substr(8, 2));
        
        if (year >= 1 && year <= 9999 && 
            month >= 1 && month <= 12 && 
            day >= 1 && day <= 31) {
            
            // 验证日期在指定日历中是否有效
            if (day > CalendarConverter::getDaysInMonth(year, month, calendar)) {
                return std::nullopt;
            }
            
            CalendarTime result;
            result.calendarType = CalendarConverter::calendarTypeToString(calendar);
            
            // 对于1970年之前的日期，使用直接计算
            if (year < 1970) {
                result.timePoint = calculateTimePointForEarlyDate(year, month, day);
            } else {
                std::tm tm_value = {};
                tm_value.tm_year = year - 1900;
                tm_value.tm_mon = month - 1;
                tm_value.tm_mday = day;
                tm_value.tm_hour = 0;
                tm_value.tm_min = 0;
                tm_value.tm_sec = 0;
                tm_value.tm_isdst = -1;
                
                auto time_t_value = std::mktime(&tm_value);
                if (time_t_value != -1) {
                    result.timePoint = std::chrono::system_clock::from_time_t(time_t_value);
                } else {
                    return std::nullopt;
                }
            }
            
            result.julianDay = result.toJulianDay();
            return result;
        }
    } catch (const std::exception&) {
        // 解析失败
    }
    
    return std::nullopt;
}

std::optional<CalendarTime> CalendarUtils::parseWithFormat(const std::string& timeString, 
                                                          const std::string& format, 
                                                          CalendarType calendar) {
    try {
        std::tm tm_value = {};
        std::istringstream ss(timeString);
        ss >> std::get_time(&tm_value, format.c_str());
        
        if (!ss.fail()) {
            // 验证解析结果
            if (tm_value.tm_year >= 0 && tm_value.tm_year <= 300 &&
                tm_value.tm_mon >= 0 && tm_value.tm_mon < 12 &&
                tm_value.tm_mday >= 1 && tm_value.tm_mday <= 31) {
                
                tm_value.tm_isdst = -1;
                auto time_t_value = std::mktime(&tm_value);
                
                if (time_t_value != -1) {
                    CalendarTime result;
                    result.calendarType = CalendarConverter::calendarTypeToString(calendar);
                    result.timePoint = std::chrono::system_clock::from_time_t(time_t_value);
                    result.julianDay = result.toJulianDay();
                    
                    // 验证在指定日历中是否有效
                    if (validateTime(result, calendar)) {
                        return result;
                    }
                }
            }
        }
    } catch (const std::exception&) {
        // 解析失败
    }
    
    return std::nullopt;
}

std::optional<CalendarTime> CalendarUtils::parseWithCommonFormats(const std::string& timeString, 
                                                                 CalendarType calendar) {
    std::vector<std::string> formats = {
        "%Y-%m-%d %H:%M:%S",    // 1950-01-01 00:00:00
        "%Y-%m-%d",             // 1950-01-01
        "%Y/%m/%d %H:%M:%S",    // 1950/01/01 00:00:00
        "%Y/%m/%d",             // 1950/01/01
        "%Y-%m-%dT%H:%M:%S",    // 1950-01-01T00:00:00 (ISO 8601)
        "%Y-%m-%dT%H:%M:%SZ",   // 1950-01-01T00:00:00Z (ISO 8601 UTC)
        "%Y%m%d",               // 19500101
        "%Y-%j",                // 1950-001 (年-儒略日)
        "%d/%m/%Y",             // 01/01/1950 (欧洲格式)
        "%m/%d/%Y",             // 01/01/1950 (美国格式)
        "%d-%m-%Y",             // 01-01-1950
        "%Y.%m.%d",             // 1950.01.01
        "%d.%m.%Y",             // 01.01.1950
    };
    
    for (const auto& format : formats) {
        auto result = parseWithFormat(timeString, format, calendar);
        if (result.has_value()) {
            return result;
        }
    }
    
    return std::nullopt;
}

std::optional<CalendarTime> CalendarUtils::parseRelativeTime(const std::string& timeString, 
                                                           CalendarType calendar) {
    // 解析相对时间表达式，如 "now", "today", "yesterday", "+1 day", "-3 months" 等
    std::string lowerStr = timeString;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    
    auto now = std::chrono::system_clock::now();
    CalendarTime baseTime;
    baseTime.calendarType = CalendarConverter::calendarTypeToString(calendar);
    baseTime.timePoint = now;
    baseTime.julianDay = baseTime.toJulianDay();
    
    if (lowerStr == "now") {
        return baseTime;
    } else if (lowerStr == "today") {
        // 今天的开始时间 (00:00:00)
        auto time_t_val = std::chrono::system_clock::to_time_t(now);
        auto tm_val = *std::gmtime(&time_t_val);
        tm_val.tm_hour = 0;
        tm_val.tm_min = 0;
        tm_val.tm_sec = 0;
        
        baseTime.timePoint = std::chrono::system_clock::from_time_t(std::mktime(&tm_val));
        baseTime.julianDay = baseTime.toJulianDay();
        return baseTime;
    } else if (lowerStr == "yesterday") {
        auto yesterday = now - std::chrono::hours(24);
        baseTime.timePoint = yesterday;
        baseTime.julianDay = baseTime.toJulianDay();
        return baseTime;
    } else if (lowerStr == "tomorrow") {
        auto tomorrow = now + std::chrono::hours(24);
        baseTime.timePoint = tomorrow;
        baseTime.julianDay = baseTime.toJulianDay();
        return baseTime;
    }
    
    // 解析相对时间偏移，如 "+1 day", "-3 months"
    return parseTimeOffset(timeString, baseTime, calendar);
}

std::optional<CalendarTime> CalendarUtils::parseTimeOffset(const std::string& offsetString, 
                                                          const CalendarTime& baseTime, 
                                                          CalendarType calendar) {
    // 简化的时间偏移解析
    // 支持格式: [+/-]数字 单位
    // 单位: second(s), minute(s), hour(s), day(s), week(s), month(s), year(s)
    
    std::istringstream iss(offsetString);
    std::string sign;
    int amount;
    std::string unit;
    
    if (!(iss >> sign >> amount >> unit)) {
        return std::nullopt;
    }
    
    bool negative = (sign == "-");
    if (!negative && sign != "+") {
        return std::nullopt;
    }
    
    // 标准化单位
    std::transform(unit.begin(), unit.end(), unit.begin(), ::tolower);
    if (unit.back() == 's') {
        unit.pop_back(); // 移除复数形式的's'
    }
    
    CalendarTime result = baseTime;
    
    try {
        if (unit == "second") {
            auto offset = std::chrono::seconds(negative ? -amount : amount);
            result.timePoint += offset;
        } else if (unit == "minute") {
            auto offset = std::chrono::minutes(negative ? -amount : amount);
            result.timePoint += offset;
        } else if (unit == "hour") {
            auto offset = std::chrono::hours(negative ? -amount : amount);
            result.timePoint += offset;
        } else if (unit == "day") {
            auto offset = std::chrono::hours(24 * (negative ? -amount : amount));
            result.timePoint += offset;
        } else if (unit == "week") {
            auto offset = std::chrono::hours(24 * 7 * (negative ? -amount : amount));
            result.timePoint += offset;
        } else if (unit == "month" || unit == "year") {
            // 月和年的计算需要考虑日历的复杂性
            result = addTimeUnit(baseTime, unit, negative ? -amount : amount, calendar);
        } else {
            return std::nullopt;
        }
        
        result.julianDay = result.toJulianDay();
        return result;
        
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

CalendarTime CalendarUtils::addTimeUnit(const CalendarTime& baseTime, 
                                       const std::string& unit, 
                                       int amount, 
                                       CalendarType calendar) {
    auto time_t_val = std::chrono::system_clock::to_time_t(baseTime.timePoint);
    auto tm_val = *std::gmtime(&time_t_val);
    
    if (unit == "month") {
        tm_val.tm_mon += amount;
        
        // 处理月份溢出
        while (tm_val.tm_mon >= 12) {
            tm_val.tm_mon -= 12;
            tm_val.tm_year++;
        }
        while (tm_val.tm_mon < 0) {
            tm_val.tm_mon += 12;
            tm_val.tm_year--;
        }
        
        // 调整日期以确保在月份范围内
        int maxDay = CalendarConverter::getDaysInMonth(tm_val.tm_year + 1900, 
                                                      tm_val.tm_mon + 1, 
                                                      calendar);
        if (tm_val.tm_mday > maxDay) {
            tm_val.tm_mday = maxDay;
        }
        
    } else if (unit == "year") {
        tm_val.tm_year += amount;
        
        // 处理闰年的2月29日
        if (tm_val.tm_mon == 1 && tm_val.tm_mday == 29) { // 2月29日
            if (!CalendarConverter::isLeapYear(tm_val.tm_year + 1900, calendar)) {
                tm_val.tm_mday = 28; // 调整为2月28日
            }
        }
    }
    
    CalendarTime result;
    result.calendarType = CalendarConverter::calendarTypeToString(calendar);
    result.timePoint = std::chrono::system_clock::from_time_t(std::mktime(&tm_val));
    result.julianDay = result.toJulianDay();
    
    return result;
}

std::chrono::system_clock::time_point CalendarUtils::calculateTimePointForEarlyDate(int year, int month, int day) {
    // 为1970年之前的日期计算时间点
    // 使用天数累加的方法，避免mktime的限制
    
    const int REFERENCE_YEAR = 1900;
    const auto REFERENCE_TIME_POINT = std::chrono::system_clock::time_point{} - 
                                     std::chrono::hours(24 * 25567); // 1900-01-01的偏移
    
    int totalDays = 0;
    
    // 计算年份的天数
    for (int y = REFERENCE_YEAR; y < year; ++y) {
        totalDays += CalendarConverter::isLeapYear(y, CalendarType::GREGORIAN) ? 366 : 365;
    }
    
    // 计算月份的天数
    for (int m = 1; m < month; ++m) {
        totalDays += CalendarConverter::getDaysInMonth(year, m, CalendarType::GREGORIAN);
    }
    
    // 加上天数 (减1因为1月1日是第0天)
    totalDays += day - 1;
    
    return REFERENCE_TIME_POINT + std::chrono::hours(24 * totalDays);
}

} // namespace oscean::common_utils::time 