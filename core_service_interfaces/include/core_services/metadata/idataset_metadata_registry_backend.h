#pragma once

#include "dataset_metadata_types.h"
#include <functional>
#include <memory>

namespace oscean {
namespace core_services {
namespace metadata {

/**
 * @class IDatasetMetadataRegistryBackend
 * @brief 数据集元数据注册表后端接口
 * 
 * 定义了数据集元数据存储后端的抽象接口，支持各种存储实现
 * (如SQLite、内存、Redis等)
 */
class IDatasetMetadataRegistryBackend {
public:
    virtual ~IDatasetMetadataRegistryBackend() = default;

    /**
     * @brief 初始化后端存储
     * @return 是否成功初始化
     */
    virtual bool initialize() = 0;

    /**
     * @brief 添加或更新数据集
     * @param entry 数据集元数据条目
     * @return 是否成功
     */
    virtual bool addOrUpdateDataset(const DatasetMetadataEntry& entry) = 0;

    /**
     * @brief 获取数据集
     * @param datasetId 数据集ID
     * @return 数据集元数据，如果不存在则返回空
     */
    virtual std::optional<DatasetMetadataEntry> getDataset(const std::string& datasetId) const = 0;

    /**
     * @brief 删除数据集
     * @param datasetId 数据集ID  
     * @return 是否成功
     */
    virtual bool removeDataset(const std::string& datasetId) = 0;

    /**
     * @brief 查找数据集
     * @param criteria 查询条件
     * @return 匹配的数据集列表
     */
    virtual std::vector<DatasetMetadataEntry> findDatasets(const MetadataQueryCriteria& criteria) const = 0;

    /**
     * @brief 获取所有数据集ID
     * @return 数据集ID列表
     */
    virtual std::vector<std::string> getAllDatasetIds() const = 0;

    /**
     * @brief 获取数据集总数
     * @return 数据集数量
     */
    virtual size_t getDatasetCount() const = 0;

    /**
     * @brief 清空所有数据集
     * @return 是否成功
     */
    virtual bool clearAllDatasets() = 0;

    /**
     * @brief 在事务中执行操作
     * @param transactionBody 事务体函数
     * @return 是否成功
     */
    virtual bool executeInTransaction(const std::function<bool()>& transactionBody) = 0;

    /**
     * @brief 替换所有数据集(原子操作)
     * @param entries 新的数据集列表
     * @return 是否成功
     */
    virtual bool replaceAllDatasets(const std::vector<DatasetMetadataEntry>& entries) = 0;
};

} // namespace metadata
} // namespace core_services  
} // namespace oscean 