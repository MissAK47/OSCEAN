#pragma once

#include "core_services/metadata/unified_metadata_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/utilities/logging_utils.h"
#include <sqlite3.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace spdlog {
    class logger;
}

namespace oscean::core_services::metadata::impl {

/**
 * @brief 数据库Schema版本信息
 */
struct SchemaVersion {
    std::string version;
    std::string description;
    std::chrono::system_clock::time_point upgradeDate;
    std::vector<std::string> migrationScripts;
};

/**
 * @brief 数据库迁移结果
 */
struct MigrationResult {
    bool success = false;
    std::string fromVersion;
    std::string toVersion;
    std::string errorMessage;
    std::chrono::milliseconds duration{0};
    std::vector<std::string> executedScripts;
};

/**
 * @brief 数据库迁移管理器
 * 
 * 负责数据库Schema版本管理、升级和数据迁移
 */
class DatabaseMigrationManager {
public:
    /**
     * @brief 构造函数
     */
    explicit DatabaseMigrationManager(
        std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices);
    
    ~DatabaseMigrationManager();

    /**
     * @brief 检查数据库Schema版本
     * @param dbPath 数据库文件路径
     * @return 当前版本，如果无法确定则返回空字符串
     */
    std::string checkSchemaVersion(const std::string& dbPath);

    /**
     * @brief 检查指定类型数据库的Schema版本
     * @param dbType 数据库类型
     * @param dbPath 数据库文件路径
     * @return 当前版本
     */
    std::string checkSchemaVersion(DatabaseType dbType, const std::string& dbPath);

    /**
     * @brief 执行Schema升级
     * @param dbType 数据库类型
     * @param dbPath 数据库文件路径
     * @param targetVersion 目标版本
     * @return 迁移结果
     */
    MigrationResult upgradeSchema(DatabaseType dbType, const std::string& dbPath, 
                                 const std::string& targetVersion);

    /**
     * @brief 清理旧数据（可选）
     * @param dbType 数据库类型
     * @param dbPath 数据库文件路径
     * @param olderThanDays 清理多少天前的数据
     * @return 是否成功
     */
    bool cleanupOldData(DatabaseType dbType, const std::string& dbPath, int olderThanDays = 30);

    /**
     * @brief 备份数据库
     * @param dbPath 数据库文件路径
     * @param backupPath 备份文件路径
     * @return 是否成功
     */
    bool backupDatabase(const std::string& dbPath, const std::string& backupPath);

    /**
     * @brief 验证数据库完整性
     * @param dbPath 数据库文件路径
     * @return 是否完整
     */
    bool validateDatabaseIntegrity(const std::string& dbPath);

    /**
     * @brief 获取支持的Schema版本列表
     * @return 版本列表
     */
    std::vector<SchemaVersion> getSupportedVersions() const;

    /**
     * @brief 获取迁移历史
     * @param dbPath 数据库文件路径
     * @return 迁移历史记录
     */
    std::vector<MigrationResult> getMigrationHistory(const std::string& dbPath);

private:
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices_;
    std::shared_ptr<common_utils::infrastructure::logging::ILogger> logger_;

    // 版本管理
    std::map<std::string, SchemaVersion> supportedVersions_;
    std::map<DatabaseType, std::string> currentSchemaVersions_;

    /**
     * @brief 初始化支持的版本信息
     */
    void initializeSupportedVersions();

    /**
     * @brief 打开数据库连接
     */
    sqlite3* openDatabase(const std::string& dbPath);

    /**
     * @brief 关闭数据库连接
     */
    void closeDatabase(sqlite3* db);

    /**
     * @brief 创建版本管理表
     */
    bool createVersionTable(sqlite3* db);

    /**
     * @brief 执行迁移脚本
     */
    bool executeMigrationScript(sqlite3* db, const std::string& script);

    /**
     * @brief 记录迁移历史
     */
    bool recordMigration(sqlite3* db, const MigrationResult& result);

    /**
     * @brief 获取数据库类型对应的Schema文件路径
     */
    std::string getSchemaFilePath(DatabaseType dbType, const std::string& version);

    /**
     * @brief 生成备份文件名
     */
    std::string generateBackupFileName(const std::string& dbPath);
};

} // namespace oscean::core_services::metadata::impl 