#pragma once

#include "common_utils/utilities/boost_config.h"
#include "core_services/common_data_types.h"
#include "unified_metadata_service.h"  // 包含完整的元数据类型定义
#include <boost/thread/future.hpp>
#include <string>
#include <vector>
#include <memory>

namespace oscean::core_services::metadata {

// 🔧 第三阶段：统一使用core_services::FileMetadata

/**
 * @brief 元数据提取器接口
 * 
 * 🎯 负责从各种格式的数据文件中提取元数据信息
 * ✅ 支持NetCDF, GeoTIFF, Shapefile等多种格式
 * ✅ 异步操作设计
 * ✅ 标准化错误处理
 */
class IMetadataExtractor {
public:
    virtual ~IMetadataExtractor() = default;

    /**
     * @brief 🔧 第三阶段：从单个文件提取标准化元数据
     * @param filePath 文件路径
     * @return 异步标准化文件元数据提取结果
     */
    virtual boost::future<oscean::core_services::metadata::AsyncResult<oscean::core_services::FileMetadata>> extractFileMetadataAsync(
        const std::string& filePath
    ) = 0;

    /**
     * @brief 🔧 第三阶段：批量提取标准化元数据
     * @param filePaths 文件路径列表
     * @return 异步批量标准化文件元数据提取结果
     */
    virtual boost::future<oscean::core_services::metadata::AsyncResult<std::vector<oscean::core_services::FileMetadata>>> extractBatchFileMetadataAsync(
        const std::vector<std::string>& filePaths
    ) = 0;

    /**
     * @brief 检查文件是否支持元数据提取
     * @param filePath 文件路径
     * @return 是否支持
     */
    virtual bool isSupportedFormat(const std::string& filePath) const = 0;

    /**
     * @brief 获取支持的文件格式列表
     * @return 支持的格式列表
     */
    virtual std::vector<std::string> getSupportedFormats() const = 0;

    /**
     * @brief 验证文件的完整性
     * @param filePath 文件路径
     * @return 异步验证结果
     */
    virtual boost::future<oscean::core_services::metadata::AsyncResult<bool>> validateFileIntegrityAsync(
        const std::string& filePath
    ) = 0;
};

} // namespace oscean::core_services::metadata 