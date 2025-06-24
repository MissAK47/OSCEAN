# NetCDF文件CF标准时间处理设计方案

## 概述

海洋环境NetCDF文件遵循CF（Climate and Forecast）标准，包含两种主要的时间表示方式：单时间文件和多时间序列文件。本方案设计了统一的时间解析和转换机制，将CF时间标准转换为清晰的日历时间格式。

## CF时间标准分析

### 1. CF时间表示方式

#### 1.1 单时间文件
```
// 示例：单个时间点的海洋数据
time = 1234567890 ;
time:units = "seconds since 1970-01-01 00:00:00" ;
time:calendar = "gregorian" ;
time:standard_name = "time" ;
```

#### 1.2 多时间序列文件
```
// 示例：时间序列数据
time = 0, 3600, 7200, 10800, ... ;
time:units = "hours since 2023-01-01 00:00:00" ;
time:calendar = "gregorian" ;
time:standard_name = "time" ;
time:long_name = "time" ;
```

### 2. 常见时间单位类型
- `seconds since YYYY-MM-DD HH:MM:SS`
- `minutes since YYYY-MM-DD HH:MM:SS`
- `hours since YYYY-MM-DD HH:MM:SS`
- `days since YYYY-MM-DD HH:MM:SS`
- `months since YYYY-MM-DD HH:MM:SS`
- `years since YYYY-MM-DD HH:MM:SS`

### 3. 日历类型
- `gregorian` / `standard`: 标准公历
- `proleptic_gregorian`: 扩展公历
- `noleap` / `365_day`: 无闰年日历
- `all_leap` / `366_day`: 全闰年日历
- `360_day`: 360天日历
- `julian`: 儒略历

## 数据库时间字段设计

### 1. 扩展数据集表结构
```sql
-- 在现有datasets表中添加时间相关字段
ALTER TABLE datasets ADD COLUMN time_type TEXT DEFAULT 'single';  -- 'single' 或 'series'
ALTER TABLE datasets ADD COLUMN time_units TEXT;                  -- CF时间单位
ALTER TABLE datasets ADD COLUMN time_calendar TEXT DEFAULT 'gregorian'; -- 日历类型
ALTER TABLE datasets ADD COLUMN reference_time TEXT;              -- 参考时间 (ISO 8601)
ALTER TABLE datasets ADD COLUMN start_time TEXT;                  -- 开始时间 (ISO 8601)
ALTER TABLE datasets ADD COLUMN end_time TEXT;                    -- 结束时间 (ISO 8601)
ALTER TABLE datasets ADD COLUMN time_step_seconds INTEGER;        -- 时间步长(秒)
ALTER TABLE datasets ADD COLUMN time_count INTEGER DEFAULT 1;     -- 时间点数量
```

### 2. 时间索引表
```sql
CREATE TABLE dataset_time_indices (
    time_index_id TEXT PRIMARY KEY,        -- 时间索引ID
    dataset_id TEXT NOT NULL,              -- 数据集ID
    time_index INTEGER NOT NULL,           -- 时间索引位置
    cf_time_value REAL NOT NULL,           -- CF时间值
    calendar_time TEXT NOT NULL,           -- 日历时间 (ISO 8601)
    year INTEGER NOT NULL,                 -- 年
    month INTEGER NOT NULL,                -- 月
    day INTEGER NOT NULL,                  -- 日
    hour INTEGER NOT NULL,                 -- 时
    minute INTEGER NOT NULL,               -- 分
    second INTEGER NOT NULL,               -- 秒
    day_of_year INTEGER,                   -- 年内第几天
    week_of_year INTEGER,                  -- 年内第几周
    
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- 时间查询优化索引
CREATE INDEX idx_time_dataset ON dataset_time_indices(dataset_id);
CREATE INDEX idx_time_calendar ON dataset_time_indices(calendar_time);
CREATE INDEX idx_time_year_month ON dataset_time_indices(year, month);
CREATE INDEX idx_time_components ON dataset_time_indices(year, month, day, hour);
```

## 时间解析服务设计

### 1. CF时间解析器
```cpp
// common_utilities/include/common_utils/netcdf/cf_time_parser.h
#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>

namespace common_utils::netcdf {

enum class CalendarType {
    Gregorian,
    ProlepticGregorian,
    NoLeap,
    AllLeap,
    Day360,
    Julian
};

enum class TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Months,
    Years
};

struct CFTimeInfo {
    std::string units;                      // 原始单位字符串
    std::string calendar;                   // 日历类型
    TimeUnit timeUnit;                      // 时间单位
    CalendarType calendarType;              // 日历类型枚举
    std::chrono::system_clock::time_point referenceTime; // 参考时间
};

struct CalendarTime {
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
    int dayOfYear;
    int weekOfYear;
    std::string isoString;                  // ISO 8601格式
    std::chrono::system_clock::time_point timePoint;
};

class CFTimeParser {
public:
    // 解析CF时间单位字符串
    static std::optional<CFTimeInfo> parseCFTimeUnits(const std::string& units);
    
    // 解析日历类型
    static CalendarType parseCalendarType(const std::string& calendar);
    
    // 转换CF时间值到日历时间
    static CalendarTime convertCFTimeToCalendar(
        double cfTimeValue,
        const CFTimeInfo& timeInfo);
    
    // 批量转换时间序列
    static std::vector<CalendarTime> convertCFTimeSeries(
        const std::vector<double>& cfTimeValues,
        const CFTimeInfo& timeInfo);
    
    // 计算时间步长
    static std::chrono::seconds calculateTimeStep(
        const std::vector<double>& cfTimeValues,
        const CFTimeInfo& timeInfo);
    
    // 验证时间连续性
    static bool validateTimeContinuity(
        const std::vector<double>& cfTimeValues,
        const CFTimeInfo& timeInfo,
        double toleranceSeconds = 1.0);

private:
    // 解析参考时间字符串
    static std::chrono::system_clock::time_point parseReferenceTime(
        const std::string& referenceTimeStr);
    
    // 不同日历类型的时间计算
    static CalendarTime calculateGregorianTime(
        double cfTimeValue,
        const CFTimeInfo& timeInfo);
    
    static CalendarTime calculateNoLeapTime(
        double cfTimeValue,
        const CFTimeInfo& timeInfo);
    
    static CalendarTime calculate360DayTime(
        double cfTimeValue,
        const CFTimeInfo& timeInfo);
    
    // 计算年内天数和周数
    static void calculateDayAndWeekOfYear(CalendarTime& calTime);
};

} // namespace common_utils::netcdf
```

### 2. NetCDF时间提取器
```cpp
// common_utilities/include/common_utils/netcdf/netcdf_time_extractor.h
#pragma once

#include "cf_time_parser.h"
#include <netcdf>
#include <memory>

namespace common_utils::netcdf {

enum class TimeFileType {
    SingleTime,     // 单时间文件
    TimeSeries      // 时间序列文件
};

struct NetCDFTimeInfo {
    TimeFileType fileType;
    CFTimeInfo cfInfo;
    std::vector<double> timeValues;
    std::vector<CalendarTime> calendarTimes;
    std::optional<std::chrono::seconds> timeStep;
    bool isRegular;                         // 是否规律时间间隔
};

class NetCDFTimeExtractor {
public:
    explicit NetCDFTimeExtractor(const std::string& filePath);
    ~NetCDFTimeExtractor();
    
    // 提取时间信息
    std::optional<NetCDFTimeInfo> extractTimeInfo();
    
    // 获取时间维度信息
    struct TimeDimensionInfo {
        std::string dimName;
        size_t dimSize;
        bool isUnlimited;
    };
    std::optional<TimeDimensionInfo> getTimeDimensionInfo();
    
    // 检查是否为单时间文件
    bool isSingleTimeFile();
    
    // 获取时间变量属性
    std::map<std::string, std::string> getTimeAttributes();

private:
    std::unique_ptr<netCDF::NcFile> ncFile_;
    std::string filePath_;
    
    // 查找时间变量
    std::optional<netCDF::NcVar> findTimeVariable();
    
    // 读取时间数据
    std::vector<double> readTimeValues(const netCDF::NcVar& timeVar);
    
    // 读取时间属性
    CFTimeInfo readTimeAttributes(const netCDF::NcVar& timeVar);
};

} // namespace common_utils::netcdf
```

### 3. 时间数据库服务
```cpp
// core_services_impl/metadata_service/include/core_services/metadata/time_metadata_service.h
#pragma once

#include "common_utils/netcdf/netcdf_time_extractor.h"
#include <memory>

namespace core_services::metadata {

struct DatasetTimeMetadata {
    std::string datasetId;
    common_utils::netcdf::TimeFileType timeType;
    std::string timeUnits;
    std::string timeCalendar;
    std::string referenceTime;
    std::string startTime;
    std::string endTime;
    std::optional<int> timeStepSeconds;
    int timeCount;
    bool isRegular;
};

class TimeMetadataService {
public:
    explicit TimeMetadataService(std::shared_ptr<IMetadataDatabase> db);
    
    // 从NetCDF文件提取并存储时间元数据
    bool extractAndStoreTimeMetadata(
        const std::string& datasetId,
        const std::string& filePath);
    
    // 获取数据集时间信息
    std::optional<DatasetTimeMetadata> getDatasetTimeMetadata(
        const std::string& datasetId);
    
    // 查询指定时间范围的数据集
    std::vector<std::string> findDatasetsInTimeRange(
        const std::string& startTime,
        const std::string& endTime,
        const std::optional<std::string>& dataType = std::nullopt);
    
    // 查询指定时间点的数据集
    std::vector<std::string> findDatasetsAtTime(
        const std::string& targetTime,
        double toleranceHours = 1.0);
    
    // 获取时间序列数据集的所有时间点
    std::vector<std::string> getDatasetTimePoints(
        const std::string& datasetId);
    
    // 验证时间数据完整性
    struct TimeValidationResult {
        bool isValid;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        std::optional<std::chrono::seconds> detectedTimeStep;
    };
    TimeValidationResult validateDatasetTime(const std::string& datasetId);

private:
    std::shared_ptr<IMetadataDatabase> db_;
    
    // 存储时间索引
    bool storeTimeIndices(
        const std::string& datasetId,
        const common_utils::netcdf::NetCDFTimeInfo& timeInfo);
    
    // 检测时间步长
    std::optional<std::chrono::seconds> detectTimeStep(
        const std::vector<common_utils::netcdf::CalendarTime>& times);
};

} // namespace core_services::metadata
```

## 时间查询视图和索引

### 1. 时间查询视图
```sql
-- 数据集时间概览视图
CREATE VIEW v_dataset_time_summary AS
SELECT 
    d.dataset_id,
    d.dataset_name,
    d.data_type,
    d.time_type,
    d.time_calendar,
    d.start_time,
    d.end_time,
    d.time_count,
    d.time_step_seconds,
    CASE 
        WHEN d.time_step_seconds IS NOT NULL THEN 'regular'
        ELSE 'irregular'
    END as time_regularity,
    -- 计算时间跨度
    CASE 
        WHEN d.start_time IS NOT NULL AND d.end_time IS NOT NULL THEN
            (julianday(d.end_time) - julianday(d.start_time)) * 24 * 3600
        ELSE NULL
    END as duration_seconds
FROM datasets d
WHERE d.start_time IS NOT NULL;

-- 月度数据统计视图
CREATE VIEW v_monthly_data_coverage AS
SELECT 
    substr(dti.calendar_time, 1, 7) as year_month,
    COUNT(DISTINCT dti.dataset_id) as dataset_count,
    COUNT(dti.time_index_id) as time_point_count,
    GROUP_CONCAT(DISTINCT d.data_type) as data_types
FROM dataset_time_indices dti
JOIN datasets d ON dti.dataset_id = d.dataset_id
GROUP BY substr(dti.calendar_time, 1, 7)
ORDER BY year_month;

-- 年度数据统计视图
CREATE VIEW v_yearly_data_coverage AS
SELECT 
    dti.year,
    COUNT(DISTINCT dti.dataset_id) as dataset_count,
    COUNT(dti.time_index_id) as time_point_count,
    MIN(dti.calendar_time) as earliest_time,
    MAX(dti.calendar_time) as latest_time,
    GROUP_CONCAT(DISTINCT d.data_type) as data_types
FROM dataset_time_indices dti
JOIN datasets d ON dti.dataset_id = d.dataset_id
GROUP BY dti.year
ORDER BY dti.year;
```

### 2. 时间查询优化索引
```sql
-- 复合时间索引
CREATE INDEX idx_time_range_query ON dataset_time_indices(calendar_time, dataset_id);
CREATE INDEX idx_time_year_month_day ON dataset_time_indices(year, month, day);
CREATE INDEX idx_time_dataset_index ON dataset_time_indices(dataset_id, time_index);

-- 数据集时间范围索引
CREATE INDEX idx_dataset_time_range ON datasets(start_time, end_time);
CREATE INDEX idx_dataset_time_type ON datasets(time_type, data_type);
```

## 实际应用示例

### 1. 时间范围查询
```sql
-- 查询指定时间范围内的海洋数据
SELECT 
    dts.dataset_id,
    dts.dataset_name,
    dts.data_type,
    dts.start_time,
    dts.end_time,
    dts.time_count,
    dts.time_regularity
FROM v_dataset_time_summary dts
WHERE dts.start_time <= '2023-12-31T23:59:59Z'
  AND dts.end_time >= '2023-01-01T00:00:00Z'
  AND dts.data_type = 'ocean_temperature_salinity'
ORDER BY dts.start_time;
```

### 2. 特定时间点查询
```sql
-- 查询最接近指定时间的数据
SELECT 
    dti.dataset_id,
    d.dataset_name,
    dti.calendar_time,
    ABS(julianday(dti.calendar_time) - julianday('2023-06-15T12:00:00Z')) * 24 as hour_diff
FROM dataset_time_indices dti
JOIN datasets d ON dti.dataset_id = d.dataset_id
WHERE d.data_type = 'sea_surface_temperature'
  AND ABS(julianday(dti.calendar_time) - julianday('2023-06-15T12:00:00Z')) * 24 <= 6
ORDER BY hour_diff ASC
LIMIT 10;
```

### 3. 时间序列分析
```sql
-- 分析数据集的时间连续性
SELECT 
    dataset_id,
    COUNT(*) as time_points,
    MIN(calendar_time) as start_time,
    MAX(calendar_time) as end_time,
    -- 检查时间间隔规律性
    COUNT(DISTINCT 
        ROUND((julianday(calendar_time) - julianday(MIN(calendar_time) OVER (PARTITION BY dataset_id))) * 24 * 3600)
    ) as unique_intervals
FROM dataset_time_indices
WHERE dataset_id = 'specific_dataset_id'
GROUP BY dataset_id;
```

## C++实现示例

### 1. CF时间解析实现
```cpp
// common_utilities/src/netcdf/cf_time_parser.cpp
#include "common_utils/netcdf/cf_time_parser.h"
#include <regex>
#include <sstream>
#include <iomanip>

namespace common_utils::netcdf {

std::optional<CFTimeInfo> CFTimeParser::parseCFTimeUnits(const std::string& units) {
    // 解析CF时间单位字符串，如 "hours since 2023-01-01 00:00:00"
    std::regex unitsRegex(R"((\w+)\s+since\s+(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?))");
    std::smatch match;
    
    if (!std::regex_match(units, match, unitsRegex)) {
        return std::nullopt;
    }
    
    CFTimeInfo info;
    info.units = units;
    
    // 解析时间单位
    std::string unitStr = match[1].str();
    if (unitStr == "seconds") {
        info.timeUnit = TimeUnit::Seconds;
    } else if (unitStr == "minutes") {
        info.timeUnit = TimeUnit::Minutes;
    } else if (unitStr == "hours") {
        info.timeUnit = TimeUnit::Hours;
    } else if (unitStr == "days") {
        info.timeUnit = TimeUnit::Days;
    } else if (unitStr == "months") {
        info.timeUnit = TimeUnit::Months;
    } else if (unitStr == "years") {
        info.timeUnit = TimeUnit::Years;
    } else {
        return std::nullopt;
    }
    
    // 解析参考时间
    std::string refTimeStr = match[2].str();
    info.referenceTime = parseReferenceTime(refTimeStr);
    
    return info;
}

CalendarTime CFTimeParser::convertCFTimeToCalendar(
    double cfTimeValue,
    const CFTimeInfo& timeInfo) {
    
    switch (timeInfo.calendarType) {
        case CalendarType::Gregorian:
        case CalendarType::ProlepticGregorian:
            return calculateGregorianTime(cfTimeValue, timeInfo);
        case CalendarType::NoLeap:
            return calculateNoLeapTime(cfTimeValue, timeInfo);
        case CalendarType::Day360:
            return calculate360DayTime(cfTimeValue, timeInfo);
        default:
            return calculateGregorianTime(cfTimeValue, timeInfo);
    }
}

CalendarTime CFTimeParser::calculateGregorianTime(
    double cfTimeValue,
    const CFTimeInfo& timeInfo) {
    
    // 计算时间偏移量（秒）
    std::chrono::seconds offset;
    switch (timeInfo.timeUnit) {
        case TimeUnit::Seconds:
            offset = std::chrono::seconds(static_cast<long long>(cfTimeValue));
            break;
        case TimeUnit::Minutes:
            offset = std::chrono::seconds(static_cast<long long>(cfTimeValue * 60));
            break;
        case TimeUnit::Hours:
            offset = std::chrono::seconds(static_cast<long long>(cfTimeValue * 3600));
            break;
        case TimeUnit::Days:
            offset = std::chrono::seconds(static_cast<long long>(cfTimeValue * 86400));
            break;
        default:
            offset = std::chrono::seconds(0);
    }
    
    // 计算目标时间点
    auto targetTime = timeInfo.referenceTime + offset;
    
    // 转换为日历时间
    auto timeT = std::chrono::system_clock::to_time_t(targetTime);
    auto tm = *std::gmtime(&timeT);
    
    CalendarTime calTime;
    calTime.year = tm.tm_year + 1900;
    calTime.month = tm.tm_mon + 1;
    calTime.day = tm.tm_mday;
    calTime.hour = tm.tm_hour;
    calTime.minute = tm.tm_min;
    calTime.second = tm.tm_sec;
    calTime.timePoint = targetTime;
    
    // 生成ISO 8601字符串
    std::ostringstream oss;
    oss << std::setfill('0') 
        << std::setw(4) << calTime.year << "-"
        << std::setw(2) << calTime.month << "-"
        << std::setw(2) << calTime.day << "T"
        << std::setw(2) << calTime.hour << ":"
        << std::setw(2) << calTime.minute << ":"
        << std::setw(2) << calTime.second << "Z";
    calTime.isoString = oss.str();
    
    // 计算年内天数和周数
    calculateDayAndWeekOfYear(calTime);
    
    return calTime;
}

} // namespace common_utils::netcdf
```

## 总结

### **核心功能**
1. **CF标准解析**: 完整支持CF时间单位和日历类型
2. **双模式处理**: 统一处理单时间和时间序列文件
3. **日历转换**: 精确转换为ISO 8601标准时间格式
4. **时间索引**: 高效的时间查询和范围检索
5. **数据验证**: 时间连续性和完整性验证

### **设计优势**
- **标准兼容**: 完全遵循CF约定标准
- **高性能**: 优化的时间索引和查询机制
- **易用性**: 清晰的日历时间表示
- **扩展性**: 支持多种日历类型和时间单位
- **可靠性**: 完整的错误处理和验证机制

这个方案将为OSCEAN项目提供强大的NetCDF时间处理能力，确保海洋环境数据的时间信息得到准确解析和高效管理。 