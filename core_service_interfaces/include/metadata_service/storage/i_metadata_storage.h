/**
 * @file i_metadata_storage.h
 * @brief 元数据存储接口
 */

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <utility> // For std::pair
#include <functional> // *** ADDED: Include for std::function ***

// Include necessary data types from the common header
#include "core_services/common_data_types.h"

namespace oscean::core_services {
namespace metadata {
namespace storage {

// *** ADDED: Definition for TransactionBody ***
using TransactionBody = std::function<bool()>;

// 前向声明MetadataEntry结构
struct MetadataEntry {
    std::string key;
    std::string value;
    std::string category;
    std::string lastModified;
};

/**
 * @class IMetadataStorage
 * @brief 元数据持久化存储接口 (扩展以支持文件元数据)
 * 该接口封装了元数据的持久化存储操作
 */
class IMetadataStorage {
public:
    /**
     * @brief 析构函数
     */
    virtual ~IMetadataStorage() = default;

    /**
     * @brief 初始化存储
     * @return 是否成功初始化
     */
    virtual bool initialize() = 0;

    /**
     * @brief 根据键获取值
     * @param key 键
     * @return 值，如果不存在则返回空
     */
    virtual std::optional<std::string> getValue(const std::string& key) const = 0;

    /**
     * @brief 设置键值对
     * @param key 键
     * @param value 值
     * @param category 分类
     * @return 是否成功设置
     */
    virtual bool setValue(const std::string& key, const std::string& value, const std::string& category) = 0;

    /**
     * @brief 删除键值对
     * @param key 键
     * @return 是否成功删除
     */
    virtual bool removeKey(const std::string& key) = 0;

    /**
     * @brief 获取所有元数据项
     * @return 元数据项列表
     */
    virtual std::vector<MetadataEntry> getAllMetadata() const = 0;

    /**
     * @brief 获取特定分类的元数据项
     * @param category 分类
     * @return 元数据项列表
     */
    virtual std::vector<MetadataEntry> getMetadataByCategory(const std::string& category) const = 0;

    /**
     * @brief 根据查询条件查找文件信息
     * @param criteria 查询条件
     * @return 文件信息列表
     */
    virtual std::vector<FileInfo> findFiles(const QueryCriteria& criteria) const = 0;

    /**
     * @brief 获取单个文件的详细元数据
     * @param fileId 文件唯一标识符 (例如路径)
     * @return 文件元数据, 如果不存在则返回空
     */
    virtual std::optional<FileMetadata> getFileMetadata(const std::string& fileId) const = 0;

    /**
     * @brief 添加或更新文件元数据
     * @param metadata 文件元数据
     * @return 是否成功
     */
    virtual bool addOrUpdateFileMetadata(const FileMetadata& metadata) = 0;

    /**
     * @brief 删除文件元数据
     * @param fileId 要删除的文件ID
     * @return 是否成功
     */
    virtual bool removeFileMetadata(const std::string& fileId) = 0;

    /**
     * @brief 按时间范围查找文件元数据
     * @param start 起始时间戳
     * @param end 结束时间戳
     * @return 匹配的文件元数据列表
     */
    virtual std::vector<FileMetadata> findFilesByTimeRange(const Timestamp start, const Timestamp end) const = 0;

    /**
     * @brief 按空间包围盒查找文件元数据
     * @param bbox 空间包围盒
     * @return 匹配的文件元数据列表
     */
    virtual std::vector<FileMetadata> findFilesByBBox(const BoundingBox& bbox) const = 0;

    /**
     * @brief 获取所有可用的变量名列表 (Distinct)
     * @return 变量名列表
     */
    virtual std::vector<std::string> getAvailableVariables() const = 0;

    /**
     * @brief 获取指定变量或全局的时间范围
     * @param variableName (可选) 变量名，为空则获取全局范围
     * @return 时间范围 (start, end)，如果无数据则返回空
     */
    virtual std::optional<std::pair<Timestamp, Timestamp>> getTimeRange(
        const std::optional<std::string>& variableName = std::nullopt) const = 0;

    /**
     * @brief 获取指定变量或全局的空间范围
     * @param variableName (可选) 变量名，为空则获取全局范围
     * @param targetCrs (可选) 目标 CRS，用于返回 BoundingBox
     * @return 空间范围，如果无数据则返回空
     */
    virtual std::optional<BoundingBox> getSpatialExtent(
        const std::optional<std::string>& variableName = std::nullopt,
        const std::optional<CRSInfo>& targetCrs = std::nullopt) const = 0;

    /**
     * @brief 完全替换存储中的索引数据 (原子操作)
     * @param metadataList 新的完整元数据列表
     * @return 是否成功替换
     */
    virtual bool replaceIndexData(const std::vector<FileMetadata>& metadataList) = 0;

    /**
     * @brief 在事务中执行操作
     * @param transactionBody 一个返回bool表示成功/失败的函数对象
     * @return 事务是否成功提交
     */
    virtual bool executeInTransaction(const TransactionBody& transactionBody) = 0;

    /**
     * @brief 批量插入元数据 (如果需要暴露给 Service 层)
     * @param metadataList 元数据列表
     * @return 是否成功
     */
    // virtual bool batchInsertMetadata(const std::vector<FileMetadata>& metadataList) = 0;

    /**
     * @brief 批量更新元数据 (如果需要暴露给 Service 层)
     * @param metadataList 元数据列表
     * @return 是否成功
     */
    // virtual bool batchUpdateMetadata(const std::vector<FileMetadata>& metadataList) = 0;

    /**
     * @brief 批量删除元数据 (如果需要暴露给 Service 层)
     * @param fileIds 文件ID列表
     * @return 是否成功
     */
    // virtual bool batchDeleteMetadata(const std::vector<std::string>& fileIds) = 0;

    /**
     * @brief 关闭存储
     */
    virtual void close() = 0;
};

} // namespace storage
} // namespace metadata
} // namespace oscean::core_services 