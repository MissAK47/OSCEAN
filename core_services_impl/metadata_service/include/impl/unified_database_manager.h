#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <sqlite3.h>
#include <boost/thread/future.hpp>

// 包含工厂，它会带来所有需要的接口定义
#include "common_utils/infrastructure/common_services_factory.h"
#include "core_services/metadata/unified_metadata_service.h" // 包含 AsyncResult 定义
#include "core_services/common_data_types.h"

// 命名空间别名，简化后续使用
namespace infra = oscean::common_utils::infrastructure;

namespace oscean::core_services::metadata::impl {

// 前向声明，因为 ClassificationResult 在这个头文件中没有完整定义
struct ClassificationResult;

/**
 * @class UnifiedDatabaseManager
 * @brief 负责管理与统一元数据SQLite数据库的交互。
 * @note 该类是线程安全的。
 */
class UnifiedDatabaseManager {
public:
    explicit UnifiedDatabaseManager(
        const std::string& dbPath,
        std::shared_ptr<infra::CommonServicesFactory> injectedCommonFactory);
    ~UnifiedDatabaseManager();

    UnifiedDatabaseManager(const UnifiedDatabaseManager&) = delete;
    UnifiedDatabaseManager& operator=(const UnifiedDatabaseManager&) = delete;

    /**
     * @brief 初始化数据库连接并应用schema。
     * @return 如果成功则返回true。
     */
    bool initialize();

    /**
     * @brief 关闭数据库连接。
     */
    void close();

    /**
     * @brief 🔧 第二阶段简化：直接存储FileMetadata
     * @param metadata 标准化文件元数据
     * @return 生成的元数据ID
     */
    boost::future<AsyncResult<std::string>> storeFileMetadataAsync(const core_services::FileMetadata& metadata);

    /**
     * @brief 🔧 第二阶段简化：直接查询FileMetadata
     * @param criteria 查询条件
     * @return 标准化文件元数据查询结果
     */
    boost::future<AsyncResult<std::vector<core_services::FileMetadata>>> queryMetadataAsync(const QueryCriteria& criteria);

    /**
     * @brief 🔧 第二阶段简化：按文件路径直接查询FileMetadata
     * @param filePath 文件路径
     * @return 标准化文件元数据查询结果
     */
    boost::future<AsyncResult<FileMetadata>> queryByFilePathAsync(const std::string& filePath);

    /**
     * @brief 🚀 [新] 批量过滤数据库中已存在的文件路径
     * @param filePaths 待检查的文件路径列表
     * @return 一个future，其结果是数据库中不存在的文件路径列表
     */
    boost::future<std::vector<std::string>> filterExistingFiles(const std::vector<std::string>& filePaths);

    /**
     * @brief 执行匿名的SQL查询
     * @param sql 要执行的SQL查询语句
     * @return 查询结果，为字符串向量的向量
     */
    boost::future<AsyncResult<std::vector<std::vector<std::string>>>> queryAsync(const std::string& sql);

    /**
     * @brief 删除元数据
     * @param metadataId 元数据ID
     * @return 是否成功
     */
    boost::future<AsyncResult<bool>> deleteMetadataAsync(const std::string& metadataId);

    /**
     * @brief 更新元数据
     * @param metadataId 元数据ID
     * @param update 更新内容
     * @return 是否成功
     */
    boost::future<AsyncResult<bool>> updateMetadataAsync(
        const std::string& metadataId, 
        const MetadataUpdate& update
    );

    /**
     * @brief 检查数据库连接状态
     */
    bool isConnected() const;

    boost::future<AsyncResult<int>> countMetadataAsync(const QueryCriteria& criteria);
    boost::future<AsyncResult<bool>> metadataExistsAsync(const std::string& metadataId);
    boost::future<AsyncResult<std::vector<std::string>>> getDistinctValuesAsync(const std::string& fieldName);
    boost::future<AsyncResult<void>> bulkInsertAsync(const std::vector<FileMetadata>& metadataList);
    boost::future<AsyncResult<std::string>> getDatabaseStatisticsAsync();

private:
    // 私有数据成员
    std::string dbPath_;
    sqlite3* db_ = nullptr;
    std::shared_ptr<infra::CommonServicesFactory> commonServices_;
    std::shared_ptr<infra::utilities::IConfigurationLoader> configLoader_;
    std::shared_ptr<infra::logging::ILogger> logger_;
    bool initialized_ = false;
    mutable std::mutex mutex_;
    
    // 🔧 SQL查询配置映射 - 避免硬编码
    std::map<std::string, std::string> sqlQueries_;

    // 私有方法
    bool executeSql(const std::string& sql);
    bool createBuiltinTables();
    bool createIndexes();
    bool executeSqlFile(const std::string& schemaPath);
    std::vector<core_services::FileMetadata> executeQuery(const std::string& sql, const std::vector<std::string>& params);
    std::string buildQuerySql(const QueryCriteria& criteria, std::vector<std::string>& params);
    core_services::FileMetadata buildFileMetadata(sqlite3_stmt* stmt);
    void insertAttributesFiltered(const std::string& metadataId, const std::map<std::string, std::string>& attributes);
    std::string getSchemaFilePath() const;
    int64_t convertTimeStringToTimestamp(const std::string& timeStr);
    std::string convertTimestampToTimeString(int64_t timestamp);
    bool tableExists(const std::string& tableName);
    std::string inferStandardName(const std::string& variableName);
    std::string inferLongName(const std::string& variableName, const std::string& standardName);
    bool inferIsPrimaryVariable(const std::string& variableName);
    std::string generateUniqueId(const std::string& input) const;
    bool createSchemaFromSqlFile();
    void applyDatabaseConfiguration();

    // 数据库插入帮助函数 (void返回，通过异常报告错误)
    void insertBasicInfo(const std::string& metadataId, const FileMetadata& metadata);
    void insertSpatialInfoFromCoverage(const std::string& metadataId, const FileMetadata& metadata);
    void insertTemporalInfoFromRange(const std::string& metadataId, const FileMetadata& metadata);
    void insertVariables(const std::string& metadataId, const std::vector<VariableMeta>& variables);
    void insertDataTypes(const std::string& metadataId, const FileMetadata& metadata);
    std::string generateMetadataId(const FileMetadata& metadata);
    
    // 🚀 新增：配置驱动查询系统支持方法
    void loadTableMappings();
    void initializeFallbackQueries();
};

} // namespace oscean::core_services::metadata::impl