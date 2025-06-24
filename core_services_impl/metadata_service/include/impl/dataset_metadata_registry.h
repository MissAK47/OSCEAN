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
#include <map>
#include <mutex>

namespace oscean::core_services::metadata::impl {

// 前向声明
class MultiDatabaseManager;

/**
 * @brief 数据集元数据条目
 */
struct DatasetMetadataEntry {
    std::string datasetId;
    std::string filePath;
    DataType dataType;
    SpatialInfo spatialInfo;
    TemporalInfo temporalInfo;
    std::vector<oscean::core_services::VariableMeta> variables;
    double dataQuality;
    double completeness;
    std::string registrationTime;
    std::map<std::string, std::string> attributes;
};

/**
 * @brief 内部数据集元数据注册器实现
 * @note 此头文件仅供内部使用，不对外暴露
 */
class DatasetMetadataRegistry {
public:
    /**
     * @brief 构造函数
     * @param dbManager 数据库管理器
     * @param commonServices 通用服务工厂
     */
    DatasetMetadataRegistry(
        std::shared_ptr<MultiDatabaseManager> dbManager,
        std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices
    );

    /**
     * @brief 🔧 第三阶段：注册文件元数据数据集
     */
    boost::future<AsyncResult<std::string>> registerDatasetAsync(
        const ::oscean::core_services::FileMetadata& metadata
    );
    
    /**
     * @brief 更新数据集元数据
     */
    boost::future<AsyncResult<void>> updateDatasetAsync(
        const std::string& datasetId,
        const MetadataUpdate& update
    );
    
    /**
     * @brief 删除数据集元数据
     */
    boost::future<AsyncResult<void>> unregisterDatasetAsync(
        const std::string& datasetId
    );
    
    /**
     * @brief 查询数据集元数据
     */
    boost::future<AsyncResult<std::vector<oscean::core_services::FileMetadata>>> queryDatasetsAsync(
        const QueryCriteria& criteria
    );

private:
    std::shared_ptr<MultiDatabaseManager> dbManager_;
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices_;
    
    // 内部存储
    std::map<std::string, DatasetMetadataEntry> registry_;
    std::mutex registryMutex_;
    
    // 🔧 第三阶段：辅助方法
    std::string generateDatasetId(const ::oscean::core_services::FileMetadata& metadata);
    bool validateFileMetadata(const ::oscean::core_services::FileMetadata& metadata);
    DatabaseType determineTargetDatabase(const ::oscean::core_services::FileMetadata& metadata);
    
    // 🔧 第三阶段：新增数据类型推断方法
    DataType inferDataTypeFromFileMetadata(const ::oscean::core_services::FileMetadata& metadata) const;
    
    // 🔧 第三阶段：新增转换方法
    void convertSpatialInfo(const ::oscean::core_services::FileMetadata& source, SpatialInfo& target) const;
    void convertTemporalInfo(const ::oscean::core_services::FileMetadata& source, TemporalInfo& target) const;
    std::vector<DatabaseType> determineDatabasesForQuery(const QueryCriteria& criteria);
    std::vector<DatabaseType> getAllDatabaseTypes();
};

} // namespace oscean::core_services::metadata::impl 