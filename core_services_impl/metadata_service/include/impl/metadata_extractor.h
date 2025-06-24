#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "core_services/metadata/unified_metadata_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/utilities/logging_utils.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <vector>

namespace oscean::core_services::metadata::impl {

/**
 * @brief 元数据管理器（重命名自MetadataExtractor）
 * @note 负责元数据的存储、管理、查询和索引，不进行文件提取
 * 
 * 正确的架构设计：
 * - metadata_service 专注于元数据的存储、管理、查询和索引
 * - 接收由 data_access_service 提取的元数据
 * - 不直接操作文件或提取原始数据
 * - 保持与 data_access_service 的解耦
 */
class MetadataExtractor {
public:
    /**
     * @brief 构造函数
     * @param commonServices 通用服务工厂
     */
    explicit MetadataExtractor(
        std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices
    );

    /**
     * @brief 析构函数
     */
    ~MetadataExtractor();

    /**
     * @brief 🔧 第三阶段：异步存储标准化文件元数据
     * @param metadata 标准化文件元数据
     * @return 异步结果，包含存储的元数据ID
     */
    boost::future<AsyncResult<std::string>> storeFileMetadataAsync(
        const ::oscean::core_services::FileMetadata& metadata
    );
    
    /**
     * @brief 🔧 第三阶段：异步批量存储文件元数据
     * @param metadataList 文件元数据列表
     * @return 异步结果，包含存储的元数据ID列表
     */
    boost::future<AsyncResult<std::vector<std::string>>> storeBatchFileMetadataAsync(
        const std::vector<::oscean::core_services::FileMetadata>& metadataList
    );

    /**
     * @brief 🔧 第三阶段：异步查询存储的文件元数据
     * @param queryFilter 查询过滤条件
     * @return 异步结果，包含查询到的文件元数据列表
     */
    boost::future<AsyncResult<std::vector<::oscean::core_services::FileMetadata>>> queryFileMetadataAsync(
        const std::string& queryFilter
    );

    /**
     * @brief 🔧 第三阶段：异步更新文件元数据
     * @param filePath 文件路径
     * @param updatedMetadata 更新的文件元数据
     * @return 异步结果，包含更新后的完整文件元数据
     */
    boost::future<AsyncResult<::oscean::core_services::FileMetadata>> updateFileMetadataAsync(
        const std::string& filePath, 
        const ::oscean::core_services::FileMetadata& updatedMetadata
    );

    /**
     * @brief 🔧 第三阶段：异步删除文件元数据
     * @param filePath 文件路径
     * @return 异步结果，表示删除是否成功
     */
    boost::future<AsyncResult<bool>> deleteFileMetadataAsync(
        const std::string& filePath
    );

private:
    // PIMPL 模式
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace oscean::core_services::metadata::impl 