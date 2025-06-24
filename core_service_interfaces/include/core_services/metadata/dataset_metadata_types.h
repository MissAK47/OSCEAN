#pragma once

#include "core_services/common_data_types.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <variant>

namespace oscean {
namespace core_services {
namespace metadata {

/**
 * @struct DatasetMetadataEntry
 * @brief 数据集元数据条目 - 数据集级别的元数据表示
 */
struct DatasetMetadataEntry {
    std::string id;                             ///< 唯一标识符
    std::string logicalName;                    ///< 用户友好名称
    std::vector<std::string> filePaths;         ///< 文件路径列表
    TimeRange timeCoverage;                     ///< 时间覆盖范围 - 使用统一定义
    BoundingBox spatialCoverage;                ///< 空间覆盖范围
    std::vector<std::string> variables;         ///< 变量名列表
    std::string format;                         ///< 数据格式
    std::string dataSource;                     ///< 数据源
    std::string processingLevel;                ///< 处理级别
    std::map<std::string, AttributeValue> attributes; ///< 自定义属性
    Timestamp lastIndexedTime = 0;              ///< 最后索引时间
    std::optional<std::string> notes;           ///< 备注

    DatasetMetadataEntry() = default;
};

/**
 * @struct MetadataQueryCriteria  
 * @brief 数据集级别的查询条件
 */
struct MetadataQueryCriteria {
    std::optional<std::string> textFilter;          ///< 文本过滤
    std::optional<TimeRange> timeRange;             ///< 时间范围 - 使用统一定义
    std::optional<BoundingBox> spatialExtent;       ///< 空间范围
    std::vector<std::string> variablesInclude;      ///< 必须包含的变量
    std::vector<std::string> variablesExclude;      ///< 必须排除的变量
    std::optional<std::string> formatFilter;        ///< 格式过滤
    std::map<std::string, std::string> attributeFilters; ///< 属性过滤
    std::optional<std::string> dataSourceFilter;    ///< 数据源过滤

    MetadataQueryCriteria() = default;
};

/**
 * @struct ExtractionOptions
 * @brief 元数据提取选项
 */
struct ExtractionOptions {
    bool recursiveScan = true;                  ///< 递归扫描
    std::vector<std::string> fileExtensions;    ///< 文件扩展名
    bool extractDetailedVariables = true;       ///< 提取详细变量信息
    bool parallelExtraction = false;            ///< 并行提取
    std::optional<CRSInfo> targetCRS = std::nullopt; ///< 目标坐标参考系统

    ExtractionOptions() {
        fileExtensions = {".nc", ".hdf", ".h5", ".grib", ".grib2", ".tif", ".tiff", ".img"};
    }
};

} // namespace metadata
} // namespace core_services
} // namespace oscean 