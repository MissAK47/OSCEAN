#pragma once

#ifndef OSCEAN_CORE_SERVICES_METADATA_UNIFIED_METADATA_SERVICE_H
#define OSCEAN_CORE_SERVICES_METADATA_UNIFIED_METADATA_SERVICE_H

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <boost/thread/future.hpp>
#include "core_services/common_data_types.h"  // 🔧 引入VariableMeta定义

// 前向声明
namespace oscean {
namespace common_utils {
namespace infrastructure {
    class CommonServicesFactory;
}
}
}

namespace oscean::core_services::metadata {

// === 🎯 基础枚举类型 ===

/**
 * @brief 数据类型枚举
 */
enum class DataType {
    OCEAN_ENVIRONMENT,        // 动态海洋环境数据（有时间分辨率）
    TOPOGRAPHY_BATHYMETRY,    // 地形底质数据（静态数据）
    BOUNDARY_LINES,           // 边界线、沉船等约束数据
    SONAR_PROPAGATION,        // 声纳传播数据  
    TACTICAL_ENVIRONMENT,     // 战术环境数据（层深、汇聚区等）
    UNKNOWN
};

/**
 * @brief 数据库类型枚举
 */
enum class DatabaseType {
    OCEAN_ENVIRONMENT,        // 动态海洋环境数据库
    TOPOGRAPHY_BATHYMETRY,    // 地形底质数据库（包含静态海洋数据）
    BOUNDARY_LINES,           // 边界约束数据库（边界线、沉船等）
    SONAR_PROPAGATION,        // 声纳传播数据库
    TACTICAL_ENVIRONMENT      // 战术环境数据库
};

/**
 * @brief 时间分辨率类型
 */
enum class TemporalResolutionType {
    UNKNOWN,          // 未知分辨率
    SECOND,           // 秒级数据
    MINUTE,           // 分钟级数据  
    HOUR,             // 小时级数据
    DAILY,            // 日数据
    WEEKLY,           // 周数据
    MONTHLY,          // 月数据
    SEASONAL,         // 季数据
    YEARLY,           // 年数据
    IRREGULAR         // 不规则时间间隔
};

// === 🏗️ 核心数据结构 ===

/**
 * @brief 空间边界
 */
struct SpatialBounds {
    double minLongitude = 0.0;
    double maxLongitude = 0.0;
    double minLatitude = 0.0;
    double maxLatitude = 0.0;
    std::optional<std::string> coordinateSystem;
    
    SpatialBounds() = default;
    SpatialBounds(double minLon, double minLat, double maxLon, double maxLat)
        : minLongitude(minLon), maxLongitude(maxLon), minLatitude(minLat), maxLatitude(maxLat) {}
};

/**
 * @brief 空间信息
 */
struct SpatialInfo {
    SpatialBounds bounds;
    std::optional<double> spatialResolution;
    std::string coordinateSystem = "WGS84";
};

/**
 * @brief 时间信息
 * 🎯 统一使用标准ISO格式字符串，所有时间转换由common_utilities处理
 */
struct TemporalInfo {
    struct TimeRange {
        std::string startTime;  ///< ISO格式时间字符串，如"2023-01-01T00:00:00Z"
        std::string endTime;    ///< ISO格式时间字符串，如"2023-12-31T23:59:59Z"
        std::string timeUnits = "ISO8601";  ///< 时间单位，统一使用ISO8601
    } timeRange;
    
    std::optional<int> temporalResolutionSeconds;
    TemporalResolutionType temporalResolutionType = TemporalResolutionType::UNKNOWN;
    std::string calendar = "gregorian";
};

// === 💾 核心元数据类型 ===

/**
 * @brief 存储选项
 */
struct StorageOptions {
    bool forceOverwrite = false;
    bool enableVersioning = true;
    std::map<std::string, std::string> customAttributes;
};

/**
 * @brief 元数据更新
 */
struct MetadataUpdate {
    std::optional<double> dataQuality;
    std::optional<double> completeness;
    std::map<std::string, std::string> updatedAttributes;
    std::vector<oscean::core_services::VariableMeta> updatedVariables; // 🔧 统一使用VariableMeta
};

// === 🔍 基础查询类型 ===

/**
 * @brief 基础查询条件
 */
struct QueryCriteria {
    std::optional<TemporalInfo::TimeRange> timeRange;  // 时间范围
    std::optional<SpatialBounds> spatialBounds;        // 空间范围
    std::vector<std::string> variablesInclude;         // 包含变量
    std::vector<std::string> variablesExclude;         // 排除变量
    std::vector<DataType> dataTypes;                   // 数据类型
    std::optional<double> minDataQuality;              // 最小数据质量
    std::optional<size_t> limit;                       // 结果数量限制
    std::optional<size_t> offset;                      // 偏移量
    
    std::string toString() const {
        // 生成查询字符串表示，用于缓存等
        std::string result = "query_";
        result += (timeRange ? "T" : "");
        result += (spatialBounds ? "S" : "");
        result += std::to_string(variablesInclude.size());
        result += "_";
        result += std::to_string(dataTypes.size());
        return result;
    }
};

// === ⚙️ 配置类型 ===

/**
 * @brief 变量分类配置
 */
struct VariableClassificationConfig {
    std::map<std::string, std::vector<std::string>> oceanVariables;
    std::map<std::string, std::vector<std::string>> topographyVariables;
    std::map<std::string, std::vector<std::string>> boundaryVariables;
    std::map<std::string, std::vector<std::string>> sonarVariables;
    bool enableFuzzyMatching = true;
    double fuzzyMatchingThreshold = 0.8;
    std::vector<std::string> priorityVariables;
    bool loadClassificationRules = true;  ///< 是否加载分类规则（数据管理工作流=true，数据处理工作流=false）
};

/**
 * @brief 数据库配置
 */
struct DatabaseConfiguration {
    std::string basePath = "./databases";
    std::map<DataType, std::string> databasePaths;
    std::map<DataType, size_t> maxConnections;
    std::chrono::seconds connectionTimeout{30};
    bool enableWALMode = true;
    size_t cacheSize = 1000;
};

/**
 * @brief 元数据服务配置
 */
struct MetadataServiceConfiguration {
    std::string config_base_path = "config"; // ✅ 新增：配置文件根目录
    DatabaseConfiguration databaseConfig;
    VariableClassificationConfig classificationConfig;
    size_t metadataCacheSize = 1000;
    size_t queryCacheSize = 500;
    std::chrono::minutes cacheExpiryTime{30};
    size_t maxConcurrentQueries = 10;
    std::chrono::milliseconds queryTimeout{5000};
    size_t maxBatchSize = 100;
};

// === 🏭 统一异步结果类型 ===

/**
 * @brief 统一异步结果类型
 */
template<typename T>
class AsyncResult {
public:
    AsyncResult() = default;
    explicit AsyncResult(T data) : data_(std::move(data)), success_(true) {}
    
    // 明确的错误构造函数
    static AsyncResult<T> failure(const std::string& error) { 
        AsyncResult<T> result;
        result.error_ = error;
        result.success_ = false;
        return result;
    }
    
    bool isSuccess() const { return success_; }
    
    const T& getData() const { 
        if (!success_) {
            throw std::runtime_error("Cannot get data from failed result: " + error_);
        }
        return data_;
    }
    
    const std::string& getError() const { return error_; }
    
    static AsyncResult<T> success(T data) { return AsyncResult<T>(std::move(data)); }
    
private:
    // 私有错误构造函数
    explicit AsyncResult(const std::string& error, bool) : error_(error), success_(false) {}
    
    T data_{};
    std::string error_;
    bool success_ = false;
};

/**
 * @brief void 类型的特化
 */
template<>
class AsyncResult<void> {
public:
    AsyncResult() : success_(true) {}
    explicit AsyncResult(const std::string& error) : error_(error), success_(false) {}
    
    bool isSuccess() const { return success_; }
    const std::string& getError() const { return error_; }
    
    static AsyncResult<void> success() { return AsyncResult<void>(); }
    static AsyncResult<void> failure(const std::string& error) { return AsyncResult<void>(error); }
    
private:
    std::string error_;
    bool success_ = false;
};

} // namespace oscean::core_services::metadata

#endif // OSCEAN_CORE_SERVICES_METADATA_UNIFIED_METADATA_SERVICE_H 