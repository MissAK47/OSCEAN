/**
 * @file sqlite_storage.h
 * @brief SQLite实现的元数据存储
 */

#pragma once

// 修复接口头文件路径 - 使用正确的绝对路径
#include "core_services/metadata/unified_metadata_service.h"
#include "metadata_service/storage/i_metadata_storage.h"

// Other necessary headers
#include "core_services/common_data_types.h"
#include "common_utils/utilities/logging_utils.h"

#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <functional>
#include <filesystem>
#include <utility> // For std::pair

// Include filesystem for path operations if needed here
#include <filesystem>

// ADD new interface and types
#include "core_services/metadata/idataset_metadata_registry_backend.h"
#include "core_services/metadata/dataset_metadata_types.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// SQLite3 是必需的依赖，不使用条件编译
// Forward declaration is better than including sqlite3.h in a header
struct sqlite3;
struct sqlite3_stmt; // Also forward declare statement handle

namespace oscean { // Added oscean namespace
namespace core_services {
namespace metadata {
namespace storage {

// Forward declarations (no longer needed if included above)
// struct MetadataEntry; // Defined in i_metadata_storage.h
// struct FileInfo; // Defined in common_data_types.h
// struct FileMetadata; // Defined in common_data_types.h
// struct QueryCriteria; // Defined in common_data_types.h
// struct BoundingBox; // Defined in common_data_types.h
// using Timestamp = uint64_t; // Defined in common_data_types.h
// struct CRSInfo; // Defined in common_data_types.h

// 前向声明SQLite相关类型
class SQLiteConnection;
class SQLiteStatement;

/**
 * @class SQLiteStorage
 * @brief 使用SQLite实现的元数据持久化存储 (优化版)
 */
class SQLiteStorage : public IMetadataStorage, public IDatasetMetadataRegistryBackend { 
public:
    /**
     * @brief 构造函数
     * @param dbPath SQLite数据库文件路径
     */
    explicit SQLiteStorage(const std::string& dbPath);
    
    /**
     * @brief 析构函数
     */
    ~SQLiteStorage() override;

    // 禁止拷贝构造和赋值
    SQLiteStorage(const SQLiteStorage&) = delete;
    SQLiteStorage& operator=(const SQLiteStorage&) = delete;
    SQLiteStorage(SQLiteStorage&&) = delete;
    SQLiteStorage& operator=(SQLiteStorage&&) = delete;

    /**
     * @brief 初始化数据库
     * @return 是否成功
     */
    bool initialize() override;

    /**
     * @brief 获取指定键的元数据值
     * @param key 元数据键
     * @return 如果找到则返回元数据值，否则返回空
     */
    std::optional<std::string> getValue(const std::string& key) const override;

    /**
     * @brief 设置元数据键值对
     * @param key 元数据键
     * @param value 元数据值
     * @param category 可选类别
     * @return 操作是否成功
     */
    bool setValue(const std::string& key, 
                const std::string& value, 
                const std::string& category) override;

    /**
     * @brief 删除元数据项
     * @param key 要删除的元数据键
     * @return 操作是否成功
     */
    bool removeKey(const std::string& key) override;

    /**
     * @brief 获取所有元数据项
     * @return 元数据项列表
     */
    std::vector<MetadataEntry> getAllMetadata() const override;

    /**
     * @brief 获取指定类别的所有元数据项
     * @param category 类别名称
     * @return 元数据项列表
     */
    std::vector<MetadataEntry> getMetadataByCategory(const std::string& category) const override;

    /**
     * @brief 关闭数据库连接
     */
    void close() override;

    /**
     * @brief 查询文件元数据
     * @param criteria 查询条件
     * @return 文件信息列表
     */
    std::vector<FileInfo> findFiles(const QueryCriteria& criteria) const override;
    
    /**
     * @brief 获取单个文件的元数据
     * @param fileId 文件ID
     * @return 文件元数据, 如果不存在则返回空
     */
    std::optional<FileMetadata> getFileMetadata(const std::string& fileId) const override;
    
    /**
     * @brief 更新索引
     * @return 是否成功
     */
    bool updateIndex();
    
    /**
     * @brief 批量插入元数据
     * @param metadataList 元数据列表
     * @return 是否成功
     */
    bool batchInsertMetadata(const std::vector<FileMetadata>& metadataList);
    
    /**
     * @brief 批量更新元数据
     * @param metadataList 元数据列表
     * @return 是否成功
     */
    bool batchUpdateMetadata(const std::vector<FileMetadata>& metadataList);
    
    /**
     * @brief 批量删除元数据
     * @param fileIds 文件ID列表
     * @return 是否成功
     */
    bool batchDeleteMetadata(const std::vector<std::string>& fileIds);

    /**
     * @brief 添加或更新文件元数据
     * @param metadata 文件元数据
     * @return 是否成功
     */
    bool addOrUpdateFileMetadata(const FileMetadata& metadata) override;

    /**
     * @brief 删除文件元数据
     * @param fileId 文件ID
     * @return 是否成功
     */
    bool removeFileMetadata(const std::string& fileId) override;

    /**
     * @brief 查找时间范围内的文件元数据
     * @param start 开始时间
     * @param end 结束时间
     * @return 文件元数据列表
     */
    std::vector<FileMetadata> findFilesByTimeRange(const Timestamp start, const Timestamp end) const override;

    /**
     * @brief 查找包围盒内的文件元数据
     * @param bbox 包围盒
     * @return 文件元数据列表
     */
    std::vector<FileMetadata> findFilesByBBox(const BoundingBox& bbox) const override;

    /**
     * @brief 获取所有可用的变量名列表 (Distinct)
     * @return 变量名列表
     */
    std::vector<std::string> getAvailableVariables() const override;

    /**
     * @brief 获取指定变量或全局的时间范围
     * @param variableName (可选) 变量名，为空则获取全局范围
     * @return 时间范围 (start, end)，如果无数据则返回空
     */
    std::optional<std::pair<Timestamp, Timestamp>> getTimeRange(
        const std::optional<std::string>& variableName = std::nullopt) const override;

    /**
     * @brief 获取指定变量或全局的空间范围
     * @param variableName (可选) 变量名，为空则获取全局范围
     * @param targetCrs (可选) 目标 CRS，用于返回 BoundingBox
     * @return 空间范围，如果无数据则返回空
     */
    std::optional<BoundingBox> getSpatialExtent(
        const std::optional<std::string>& variableName = std::nullopt,
        const std::optional<CRSInfo>& targetCrs = std::nullopt) const override;

    /**
     * @brief 完全替换存储中的索引数据 (原子操作)
     * @param metadataList 新的完整元数据列表
     * @return 是否成功替换
     */
    bool replaceIndexData(const std::vector<FileMetadata>& metadataList) override;

    // --- IDatasetMetadataRegistryBackend Methods ---
    bool addOrUpdateDataset(const oscean::core_services::metadata::DatasetMetadataEntry& entry) override;
    std::optional<oscean::core_services::metadata::DatasetMetadataEntry> getDataset(const std::string& datasetId) const override;
    bool removeDataset(const std::string& datasetId) override;
    std::vector<oscean::core_services::metadata::DatasetMetadataEntry> findDatasets(const oscean::core_services::metadata::MetadataQueryCriteria& criteria) const override;
    std::vector<std::string> getAllDatasetIds() const override;
    size_t getDatasetCount() const override;
    bool clearAllDatasets() override;
    bool executeInTransaction(const std::function<bool()>& transactionBody) override;
    bool replaceAllDatasets(const std::vector<oscean::core_services::metadata::DatasetMetadataEntry>& entries) override;

private:
    // Database and connection management
    std::string dbPath_;
    sqlite3* db_ = nullptr;
    mutable std::mutex mutex_;
    bool isInitialized_ = false;

    // Prepared statements for efficient database operations
    struct PreparedStatements {
        sqlite3_stmt* insertFile = nullptr;
        sqlite3_stmt* updateFile = nullptr;
        sqlite3_stmt* deleteFile = nullptr;
        sqlite3_stmt* insertVariable = nullptr;
        sqlite3_stmt* deleteVariablesForFile = nullptr;
        sqlite3_stmt* findFilesBase = nullptr; // Need dynamic part handling
        sqlite3_stmt* getFile = nullptr;
        sqlite3_stmt* getTimeRange = nullptr;
        sqlite3_stmt* getSpatialExtent = nullptr;
        sqlite3_stmt* getVariables = nullptr;
        sqlite3_stmt* insertKeyValue = nullptr;
        sqlite3_stmt* getKeyValue = nullptr;
        sqlite3_stmt* deleteKeyValue = nullptr;
        // NEW statements for DatasetMetadataEntry
        sqlite3_stmt* insertDataset = nullptr;
        sqlite3_stmt* updateDataset = nullptr;
        sqlite3_stmt* getDatasetById = nullptr;
        sqlite3_stmt* deleteDatasetById = nullptr;
        sqlite3_stmt* getAllDatasetIds = nullptr;
        sqlite3_stmt* getDatasetCount = nullptr;
        sqlite3_stmt* clearAllDatasets = nullptr;
        sqlite3_stmt* findDatasets = nullptr;
        // 🆕 CRS信息相关语句
        sqlite3_stmt* insertCrsInfo = nullptr;
        sqlite3_stmt* getCrsInfo = nullptr;
        sqlite3_stmt* deleteCrsInfo = nullptr;
        ~PreparedStatements(); // Destructor to finalize statements
    };
    PreparedStatements statements_;

    // In-memory datasets storage for RegistryBackend interface
    mutable std::unordered_map<std::string, oscean::core_services::metadata::DatasetMetadataEntry> datasets_;

    // Helper methods for dataset management
    bool addOrUpdateDatasetInMemory(const oscean::core_services::metadata::DatasetMetadataEntry& entry);
    bool removeDatasetInMemory(const std::string& datasetId);
    std::vector<oscean::core_services::metadata::DatasetMetadataEntry> findDatasetsInMemory(const oscean::core_services::metadata::MetadataQueryCriteria& criteria) const;

    // Helper for in-memory FileMetadata (old, to be phased out or adapted)
    std::optional<FileMetadata> getFileMetadataFromMemory(const std::string& fileId);
    bool addOrUpdateFileMetadataInMemory(const FileMetadata& metadata);
    bool removeFileMetadataFromMemory(const std::string& fileId);
    std::vector<FileInfo> findFileInfosInMemory(const QueryCriteria& criteria);
    void clearAllFileMetadataFromMemory();

    // 🆕 CRS信息处理helper函数
    CRSInfo getCRSInfoForFile(const std::string& fileId) const;
    bool saveCRSInfoForFile(const std::string& fileId, const CRSInfo& crsInfo);
    bool deleteCRSInfoForFile(const std::string& fileId);

    // Helper for in-memory DatasetMetadataEntry operations

    // New Dataset methods with in-memory management
    std::optional<DatasetMetadataEntry> getDatasetFromMemory(const std::string& datasetId);
    bool removeDatasetFromMemory(const std::string& datasetId);
    std::vector<DatasetMetadataEntry> findDatasetsInMemory(const MetadataQueryCriteria& criteria);
    std::vector<std::string> getAllDatasetIdsInMemory();
    int64_t getDatasetCountInMemory();
    void clearAllDatasetsInMemory();
    bool replaceAllDatasetsInMemory(const std::vector<DatasetMetadataEntry>& datasets);

    /**
     * @brief 创建数据库表 (private helper)
     * @return 是否成功
     */
    bool createTables();

    /**
     * @brief 准备SQL语句 (private helper)
     * @return 是否成功
     */
    bool prepareStatements();

    /**
     * @brief 释放准备的SQL语句 (private helper)
     */
    void finalizeStatements();

    /**
     * @brief 获取当前时间戳字符串 (private helper)
     * @return 时间戳字符串 (YYYY-MM-DD HH:MM:SS)
     */
    std::string getCurrentTimestamp() const;

    /**
     * @brief 帮助方法
     */
    void initDatabase();
    std::string buildQuerySql(const QueryCriteria& criteria);
    
    // 在事务中执行批量操作
    bool batchInsertMetadataInternal(const std::vector<FileMetadata>& metadataList);
    bool batchUpdateMetadataInternal(const std::vector<FileMetadata>& metadataList);
    bool batchDeleteMetadataInternal(const std::vector<std::string>& fileIds);

    // NEW internal helpers for DatasetMetadataEntry if needed
    bool addOrUpdateDatasetInternal(const oscean::core_services::metadata::DatasetMetadataEntry& entry);
    bool removeDatasetInternal(const std::string& datasetId);
    std::string buildFindDatasetsSql(const oscean::core_services::metadata::MetadataQueryCriteria& criteria, std::vector<std::variant<std::string, int64_t, double, oscean::core_services::Timestamp>>& params) const;

    // Private helper for clearing dataset tables, to be called within a transaction if needed
    bool clearAllDatasetsInternal() noexcept; 

    /**
     * @brief 获取指定文件的变量列表
     * @param fileId 文件ID
     * @return 变量元数据列表
     */
    std::vector<VariableMeta> getVariablesForFile(const std::string& fileId) const;
};

} // namespace storage
} // namespace metadata
} // namespace core_services
} // namespace oscean // Added oscean namespace 