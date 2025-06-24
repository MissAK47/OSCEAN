#include "impl/database_migration_manager.h"
#include "common_utils/utilities/string_utils.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;
using namespace oscean::core_services::metadata::impl;

DatabaseMigrationManager::DatabaseMigrationManager(
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices)
    : commonServices_(commonServices) {
    
    logger_ = commonServices_->getLogger();
    initializeSupportedVersions();
    
    // 设置当前Schema版本
    currentSchemaVersions_[DatabaseType::OCEAN_ENVIRONMENT] = "3.1";
    currentSchemaVersions_[DatabaseType::TOPOGRAPHY_BATHYMETRY] = "3.1";
    currentSchemaVersions_[DatabaseType::BOUNDARY_LINES] = "3.1";
    currentSchemaVersions_[DatabaseType::SONAR_PROPAGATION] = "3.1";
    currentSchemaVersions_[DatabaseType::TACTICAL_ENVIRONMENT] = "3.1";
}

DatabaseMigrationManager::~DatabaseMigrationManager() = default;

void DatabaseMigrationManager::initializeSupportedVersions() {
    // 版本1.0 - 基础版本
    SchemaVersion v1_0;
    v1_0.version = "1.0";
    v1_0.description = "基础Schema版本";
    v1_0.migrationScripts = {};
    supportedVersions_["1.0"] = v1_0;

    // 版本2.0 - 增加Geohash和全文搜索
    SchemaVersion v2_0;
    v2_0.version = "2.0";
    v2_0.description = "增加Geohash索引和全文搜索支持";
    v2_0.migrationScripts = {
        "ALTER TABLE spatial_coverage ADD COLUMN geohash_6 TEXT;",
        "ALTER TABLE spatial_coverage ADD COLUMN geohash_8 TEXT;",
        "CREATE TABLE IF NOT EXISTS fts_content (file_id TEXT PRIMARY KEY, content TEXT);"
    };
    supportedVersions_["2.0"] = v2_0;

    // 版本3.0 - 增强CRS支持
    SchemaVersion v3_0;
    v3_0.version = "3.0";
    v3_0.description = "增强CRS支持和空间分辨率";
    v3_0.migrationScripts = {
        "ALTER TABLE spatial_coverage ADD COLUMN crs_proj4 TEXT;",
        "ALTER TABLE spatial_coverage ADD COLUMN crs_epsg_code INTEGER;",
        "ALTER TABLE spatial_coverage ADD COLUMN spatial_resolution_x REAL;",
        "ALTER TABLE spatial_coverage ADD COLUMN spatial_resolution_y REAL;"
    };
    supportedVersions_["3.0"] = v3_0;

    // 版本3.1 - 统一Schema版本
    SchemaVersion v3_1;
    v3_1.version = "3.1";
    v3_1.description = "统一Schema结构，支持所有新字段";
    v3_1.migrationScripts = {
        "ALTER TABLE temporal_coverage ADD COLUMN time_resolution_seconds REAL;",
        "ALTER TABLE variable_info ADD COLUMN min_value REAL;",
        "ALTER TABLE variable_info ADD COLUMN max_value REAL;",
        "ALTER TABLE variable_info ADD COLUMN scale_factor REAL;",
        "ALTER TABLE variable_info ADD COLUMN add_offset REAL;"
    };
    supportedVersions_["3.1"] = v3_1;
}

std::string DatabaseMigrationManager::checkSchemaVersion(const std::string& dbPath) {
    sqlite3* db = openDatabase(dbPath);
    if (!db) {
        return "";
    }

    std::string version;
    const char* sql = "SELECT version FROM schema_version ORDER BY upgrade_date DESC LIMIT 1;";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* versionStr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if (versionStr) {
                version = versionStr;
            }
        }
        sqlite3_finalize(stmt);
    }

    closeDatabase(db);
    
    // 如果没有版本表或版本信息，假设是1.0版本
    return version.empty() ? "1.0" : version;
}

std::string DatabaseMigrationManager::checkSchemaVersion(DatabaseType dbType, const std::string& dbPath) {
    return checkSchemaVersion(dbPath);
}

MigrationResult DatabaseMigrationManager::upgradeSchema(DatabaseType dbType, const std::string& dbPath, 
                                                       const std::string& targetVersion) {
    MigrationResult result;
    auto startTime = std::chrono::steady_clock::now();
    
    result.fromVersion = checkSchemaVersion(dbPath);
    result.toVersion = targetVersion;
    
    logger_->info("开始数据库迁移: " + result.fromVersion + " -> " + targetVersion + ", 文件: " + dbPath);

    // 检查目标版本是否支持
    if (supportedVersions_.find(targetVersion) == supportedVersions_.end()) {
        result.errorMessage = "不支持的目标版本: " + targetVersion;
        logger_->error(result.errorMessage);
        return result;
    }

    // 如果已经是目标版本，直接返回成功
    if (result.fromVersion == targetVersion) {
        result.success = true;
        logger_->info("数据库已经是目标版本 " + targetVersion);
        return result;
    }

    // 备份数据库
    std::string backupPath = generateBackupFileName(dbPath);
    if (!backupDatabase(dbPath, backupPath)) {
        result.errorMessage = "数据库备份失败";
        logger_->error(result.errorMessage);
        return result;
    }

    sqlite3* db = openDatabase(dbPath);
    if (!db) {
        result.errorMessage = "无法打开数据库";
        logger_->error(result.errorMessage);
        return result;
    }

    // 创建版本管理表
    if (!createVersionTable(db)) {
        result.errorMessage = "无法创建版本管理表";
        closeDatabase(db);
        return result;
    }

    // 开始事务
    if (sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        result.errorMessage = "无法开始事务";
        closeDatabase(db);
        return result;
    }

    try {
        // 执行迁移脚本
        std::vector<std::string> versionsToApply;
        
        // 确定需要应用的版本
        if (result.fromVersion == "1.0" && targetVersion == "3.1") {
            versionsToApply = {"2.0", "3.0", "3.1"};
        } else if (result.fromVersion == "2.0" && targetVersion == "3.1") {
            versionsToApply = {"3.0", "3.1"};
        } else if (result.fromVersion == "3.0" && targetVersion == "3.1") {
            versionsToApply = {"3.1"};
        }

        for (const auto& version : versionsToApply) {
            const auto& schemaVersion = supportedVersions_[version];
            for (const auto& script : schemaVersion.migrationScripts) {
                if (!executeMigrationScript(db, script)) {
                    throw std::runtime_error("迁移脚本执行失败: " + script);
                }
                result.executedScripts.push_back(script);
            }
        }

        // 更新版本信息
        const char* updateVersionSql = "INSERT OR REPLACE INTO schema_version (version, upgrade_date) VALUES (?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, updateVersionSql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, targetVersion.c_str(), -1, SQLITE_STATIC);
            
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S");
            std::string timeStr = ss.str();
            
            sqlite3_bind_text(stmt, 2, timeStr.c_str(), -1, SQLITE_STATIC);
            
            if (sqlite3_step(stmt) == SQLITE_DONE) {
                result.success = true;
                auto endTime = std::chrono::steady_clock::now();
                result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
                
                logger_->info("数据库迁移成功: " + result.fromVersion + " -> " + targetVersion);
            } else {
                throw std::runtime_error("无法更新版本信息");
            }
            sqlite3_finalize(stmt);
        }

        // 提交事务
        if (sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) {
            result.errorMessage = "无法提交事务";
            logger_->error(result.errorMessage);
            closeDatabase(db);
            return result;
        }

        // 记录迁移历史
        recordMigration(db, result);
        
        closeDatabase(db);
        
        logger_->info("数据库Schema升级完成");
        return result;

    } catch (const std::exception& e) {
        // 回滚事务
        sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
        
        result.success = false;
        result.errorMessage = e.what();
        auto endTime = std::chrono::steady_clock::now();
        result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        logger_->error("数据库迁移失败: " + result.errorMessage);
        closeDatabase(db);
        return result;
    }
}

bool DatabaseMigrationManager::cleanupOldData(DatabaseType dbType, const std::string& dbPath, int olderThanDays) {
    logger_->info("开始清理 " + std::to_string(olderThanDays) + " 天前的旧数据");

    sqlite3* db = openDatabase(dbPath);
    if (!db) {
        return false;
    }

    // 计算截止时间戳
    auto cutoffTime = std::chrono::system_clock::now() - std::chrono::hours(24 * olderThanDays);
    auto cutoffTimestamp = std::chrono::duration_cast<std::chrono::seconds>(cutoffTime.time_since_epoch()).count();

    const char* sql = "DELETE FROM file_info WHERE created_at < datetime(?, 'unixepoch');";
    sqlite3_stmt* stmt;
    
    bool success = false;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int64(stmt, 1, cutoffTimestamp);
        success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
    }

    closeDatabase(db);
    
    if (success) {
        logger_->info("清理了 " + std::to_string(olderThanDays) + " 天前的旧数据");
    }
    
    return success;
}

bool DatabaseMigrationManager::backupDatabase(const std::string& dbPath, const std::string& backupPath) {
    try {
        fs::copy_file(dbPath, backupPath, fs::copy_options::overwrite_existing);
        logger_->info("数据库备份成功: " + dbPath + " -> " + backupPath);
        return true;
    } catch (const std::exception& e) {
        logger_->error("数据库备份失败: " + std::string(e.what()));
        return false;
    }
}

bool DatabaseMigrationManager::validateDatabaseIntegrity(const std::string& dbPath) {
    sqlite3* db = openDatabase(dbPath);
    if (!db) {
        return false;
    }

    const char* sql = "PRAGMA integrity_check;";
    sqlite3_stmt* stmt;
    bool isValid = false;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            isValid = (result && std::string(result) == "ok");
        }
        sqlite3_finalize(stmt);
    }

    closeDatabase(db);
    return isValid;
}

std::vector<SchemaVersion> DatabaseMigrationManager::getSupportedVersions() const {
    std::vector<SchemaVersion> versions;
    for (const auto& pair : supportedVersions_) {
        versions.push_back(pair.second);
    }
    return versions;
}

std::vector<MigrationResult> DatabaseMigrationManager::getMigrationHistory(const std::string& dbPath) {
    std::vector<MigrationResult> history;
    
    sqlite3* db = openDatabase(dbPath);
    if (!db) {
        return history;
    }

    const char* sql = "SELECT from_version, to_version, success, error_message, duration_ms, upgrade_date FROM migration_history ORDER BY upgrade_date DESC;";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            MigrationResult result;
            result.fromVersion = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            result.toVersion = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            result.success = sqlite3_column_int(stmt, 2) != 0;
            
            const char* errorMsg = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            if (errorMsg) {
                result.errorMessage = errorMsg;
            }
            
            result.duration = std::chrono::milliseconds(sqlite3_column_int64(stmt, 4));
            history.push_back(result);
        }
        sqlite3_finalize(stmt);
    }

    closeDatabase(db);
    return history;
}

sqlite3* DatabaseMigrationManager::openDatabase(const std::string& dbPath) {
    sqlite3* db = nullptr;
    if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
        logger_->error("无法打开数据库: " + dbPath);
        return nullptr;
    }
    return db;
}

void DatabaseMigrationManager::closeDatabase(sqlite3* db) {
    if (db) {
        sqlite3_close(db);
    }
}

bool DatabaseMigrationManager::createVersionTable(sqlite3* db) {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS schema_version (
            version TEXT PRIMARY KEY,
            upgrade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS migration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_version TEXT,
            to_version TEXT,
            success INTEGER,
            error_message TEXT,
            duration_ms INTEGER,
            upgrade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    )";
    
    return sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK;
}

bool DatabaseMigrationManager::executeMigrationScript(sqlite3* db, const std::string& script) {
    char* errorMsg = nullptr;
    int result = sqlite3_exec(db, script.c_str(), nullptr, nullptr, &errorMsg);
    
    if (result != SQLITE_OK) {
        if (errorMsg) {
            logger_->error("迁移脚本执行失败: " + script + ", 错误: " + std::string(errorMsg));
            sqlite3_free(errorMsg);
        }
        return false;
    }
    
    logger_->debug("迁移脚本执行成功: " + script);
    return true;
}

bool DatabaseMigrationManager::recordMigration(sqlite3* db, const MigrationResult& result) {
    const char* sql = "INSERT INTO migration_history (from_version, to_version, success, error_message, duration_ms) VALUES (?, ?, ?, ?, ?);";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, result.fromVersion.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, result.toVersion.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 3, result.success ? 1 : 0);
    sqlite3_bind_text(stmt, 4, result.errorMessage.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 5, result.duration.count());

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    
    return success;
}

std::string DatabaseMigrationManager::getSchemaFilePath(DatabaseType dbType, const std::string& version) {
    std::string dbTypeName;
    switch (dbType) {
        case DatabaseType::OCEAN_ENVIRONMENT:
            dbTypeName = "ocean_environment";
            break;
        case DatabaseType::TOPOGRAPHY_BATHYMETRY:
            dbTypeName = "topography_bathymetry";
            break;
        case DatabaseType::BOUNDARY_LINES:
            dbTypeName = "boundary_lines";
            break;
        case DatabaseType::SONAR_PROPAGATION:
            dbTypeName = "sonar_propagation";
            break;
        case DatabaseType::TACTICAL_ENVIRONMENT:
            dbTypeName = "tactical_environment";
            break;
        default:
            dbTypeName = "unknown";
    }
    
    return "src/database/schema/" + dbTypeName + "_schema_v" + version + ".sql";
}

std::string DatabaseMigrationManager::generateBackupFileName(const std::string& dbPath) {
    fs::path path(dbPath);
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << path.stem().string() << "_backup_" << std::put_time(std::gmtime(&time_t), "%Y%m%d_%H%M%S") << path.extension().string();
    
    return (path.parent_path() / ss.str()).string();
} 