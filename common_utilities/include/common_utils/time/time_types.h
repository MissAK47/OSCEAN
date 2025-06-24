/**
 * @file time_types.h
 * @brief 通用时间类型定义 - 时间处理模块核心类型
 * 
 * 🔴 Critical: 只包含通用时间类型，禁止包含格式专用解析代码
 * 设计原则：纯净化、抽象化、可扩展
 */

#pragma once

/**
 * @file time_types.h
 * @brief 通用时间类型定义 - Common层只定义抽象类型和接口
 * @author OSCEAN Team
 * @date 2024
 * 
 * 🔴 Critical: 此文件只包含通用时间类型，禁止包含任何格式专用解析代码
 */

#include "../utilities/boost_config.h"
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <functional>
#include <memory>

namespace oscean::common_utils::time {

/**
 * @brief 通用日历时间类型 - 所有模块统一使用
 */
struct CalendarTime {
    std::chrono::system_clock::time_point timePoint;
    std::string calendarType = "gregorian";  // 支持多种日历系统
    double julianDay = 0.0;                  // 儒略日
    
    // 构造函数
    CalendarTime() = default;
    explicit CalendarTime(const std::chrono::system_clock::time_point& tp) : timePoint(tp) {}
    
    // 转换方法
    std::string toISOString() const;
    std::string toIsoString() const { return toISOString(); }  ///< 兼容别名
    std::chrono::system_clock::time_point toTimePoint() const { return timePoint; }  ///< 兼容方法
    double toJulianDay() const;
    
    // ✅ 新增中文格式化方法
    std::string toChineseString() const;
    std::string toChineseFullString() const;
    
    // 数学运算
    CalendarTime operator+(std::chrono::seconds duration) const;
    CalendarTime operator-(std::chrono::seconds duration) const;
    std::chrono::seconds operator-(const CalendarTime& other) const;
    
    // 比较操作
    bool operator<(const CalendarTime& other) const;
    bool operator==(const CalendarTime& other) const;
    bool operator<=(const CalendarTime& other) const;
    bool operator>=(const CalendarTime& other) const;
    bool operator>(const CalendarTime& other) const;
    bool operator!=(const CalendarTime& other) const;
    
    // 序列化支持
    std::string serialize() const;
    static CalendarTime deserialize(const std::string& data);
    
    // 有效性检查
    bool isValid() const;
};

/**
 * @brief 时间范围定义
 */
struct TimeRange {
    CalendarTime startTime;
    CalendarTime endTime;
    
    // 构造函数
    TimeRange() = default;
    TimeRange(const CalendarTime& start, const CalendarTime& end) 
        : startTime(start), endTime(end) {}
    
    // 范围操作
    bool contains(const CalendarTime& time) const;
    bool overlaps(const TimeRange& other) const;
    TimeRange intersection(const TimeRange& other) const;
    TimeRange union_range(const TimeRange& other) const;
    
    // 时间计算
    std::chrono::seconds getDuration() const;
    bool isValid() const;
    bool isEmpty() const;
    
    // 分割时间范围
    std::vector<TimeRange> split(std::chrono::seconds interval) const;
    std::vector<TimeRange> splitToChunks(size_t chunkCount) const;
    
    // 字符串表示
    std::string toString() const;
    
    // 序列化
    std::string serialize() const;
    static TimeRange deserialize(const std::string& data);
    
    // 比较操作符
    bool operator==(const TimeRange& other) const;
    bool operator!=(const TimeRange& other) const;
    bool operator<(const TimeRange& other) const;
};

/**
 * @brief 时间分辨率信息
 */
struct TimeResolutionInfo {
    std::chrono::seconds nominalResolution{0};    // 名义分辨率
    std::chrono::seconds actualResolution{0};     // 实际分辨率
    std::chrono::seconds minInterval{0};          // 最小间隔
    std::chrono::seconds maxInterval{0};          // 最大间隔
    double regularityRatio = 0.0;                 // 规律性比率 [0-1]
    bool isRegular = false;                       // 是否规律
    size_t totalTimePoints = 0;                   // 总时间点数
    std::vector<std::chrono::seconds> gaps;       // 时间间隙
    
    // 质量评估
    bool isHighQuality() const { return regularityRatio > 0.95 && isRegular; }
    bool hasSignificantGaps() const { return !gaps.empty(); }
    std::string getQualityDescription() const;
    
    // 序列化
    std::string serialize() const;
    static TimeResolutionInfo deserialize(const std::string& data);
};

/**
 * @brief 时间轴信息
 */
struct TimeAxisInfo {
    std::string name;                           // 时间轴名称
    std::string units;                          // 时间单位
    std::string calendarType;                   // 日历类型
    CalendarTime referenceTime;                 // 参考时间
    TimeRange timeRange;                        // 时间范围
    TimeResolutionInfo resolution;              // 分辨率信息
    std::vector<std::string> attributes;        // 属性列表
    size_t numberOfTimePoints;                  // 时间点数量
    bool isUnlimited;                          // 是否无限维
    
    // 构造函数
    TimeAxisInfo() : numberOfTimePoints(0), isUnlimited(false) {}
    
    // 验证
    bool isValid() const;
    std::string getDescription() const;
};

/**
 * @brief 时间索引结构
 */
struct TimeIndex {
    std::vector<CalendarTime> timePoints;       // 时间点列表
    std::map<CalendarTime, size_t> timeToIndex; // 时间到索引的映射
    std::map<size_t, CalendarTime> indexToTime; // 索引到时间的映射
    TimeRange coverageRange;                    // 覆盖范围
    TimeResolutionInfo resolution;              // 分辨率信息
    
    // 构造函数
    TimeIndex() = default;
    explicit TimeIndex(const std::vector<CalendarTime>& times);
    
    // 查询方法
    std::optional<size_t> findIndex(const CalendarTime& time) const;
    std::optional<CalendarTime> getTime(size_t index) const;
    std::vector<size_t> findIndicesInRange(const TimeRange& range) const;
    
    // 构建方法
    void buildIndex();
    void addTimePoint(const CalendarTime& time);
    void clear();
    
    // 验证
    bool isValid() const;
    size_t size() const { return timePoints.size(); }
    bool empty() const { return timePoints.empty(); }
};

/**
 * @brief 时间覆盖分析结果
 */
struct TimeCoverageAnalysis {
    TimeRange totalCoverage;                    // 总覆盖范围
    std::vector<TimeRange> gaps;                // 时间间隙
    std::vector<TimeRange> overlaps;            // 重叠区域
    double coverageRatio;                       // 覆盖率 (0-1)
    TimeResolutionInfo overallResolution;       // 整体分辨率
    size_t totalDataSources;                    // 数据源总数
    std::map<std::string, TimeRange> sourceCoverage; // 各数据源覆盖
    
    // 构造函数
    TimeCoverageAnalysis() : coverageRatio(0.0), totalDataSources(0) {}
    
    // 质量评估
    bool hasGoodCoverage() const { return coverageRatio > 0.8; }
    bool hasSignificantGaps() const { return !gaps.empty(); }
    std::string getQualityReport() const;
};

/**
 * @brief 时间提取选项
 */
struct TimeExtractionOptions {
    std::string preferredCalendar = "gregorian"; // 首选日历类型
    std::string timeZone = "UTC";                // 时区
    bool strictMode = false;                     // 严格模式
    bool allowApproximation = true;              // 允许近似
    std::chrono::seconds maxTimeGap{3600};       // 最大时间间隙
    size_t maxTimePoints = 10000;               // 最大时间点数
    bool enableCaching = true;                   // 启用缓存
    bool validateTimes = true;                   // 验证时间
    
    // 解析选项
    std::vector<std::string> supportedFormats;  // 支持的格式
    std::map<std::string, std::string> customAttributes; // 自定义属性
    
    // 性能选项
    size_t batchSize = 1000;                    // 批处理大小
    bool useParallelProcessing = true;           // 使用并行处理
    
    // 验证
    bool isValid() const;
    std::string toString() const;
};

// 🔴 编译期检查 - 禁止在Common层包含格式专用代码
#ifdef NETCDF_TIME_PARSER_INCLUDED
    #error "❌ FORBIDDEN: Common层禁止包含NetCDF专用时间解析代码！应在数据访问服务层实现"
#endif

#ifdef GDAL_TIME_PARSER_INCLUDED  
    #error "❌ FORBIDDEN: Common层禁止包含GDAL专用时间解析代码！应在数据访问服务层实现"
#endif

#ifdef CF_TIME_STANDARDS_INCLUDED
    #error "❌ FORBIDDEN: Common层禁止包含CF时间标准解析代码！应在数据访问服务层实现"
#endif

/**
 * @brief 时间格式化工具类 - 提供统一的中文格式化接口
 */
class TimeFormatUtils {
public:
    /**
     * @brief 格式化时间持续时间为中文字符串
     */
    static std::string formatDurationToChinese(std::chrono::seconds duration);
    
    /**
     * @brief 格式化时间持续时间为简短中文字符串
     */
    static std::string formatDurationToChineseShort(std::chrono::seconds duration);
    
    /**
     * @brief 格式化CalendarTime为中文字符串
     */
    static std::string formatCalendarTimeToChinese(const CalendarTime& time);
    
    /**
     * @brief 格式化CalendarTime为完整中文字符串
     */
    static std::string formatCalendarTimeToChineseFull(const CalendarTime& time);
    
    /**
     * @brief 格式化TimeRange为中文字符串
     */
    static std::string formatTimeRangeToChinese(const TimeRange& range);
    
    /**
     * @brief 格式化时间分辨率为中文字符串
     */
    static std::string formatResolutionToChinese(const TimeResolutionInfo& info);
};

} // namespace oscean::common_utils::time 