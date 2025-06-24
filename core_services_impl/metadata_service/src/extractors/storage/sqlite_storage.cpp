/**
 * @file sqlite_storage.cpp
 * @brief SQLite实现的元数据存储 (优化版)
 */

#include "sqlite_storage.h" // Include own header first

// SQLite3 是必需的依赖
#include <sqlite3.h> // Include sqlite3 header only in cpp file

// Standard Library Includes
#include <algorithm> // for std::find, std::min/max
#include <chrono>
#include <cmath>   // for std::abs
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip> // Added for std::put_time
#include <iterator> // for std::istream_iterator
#include <memory>
#include <mutex>
#include <optional>
#include <set>       // for getAvailableVariables
#include <sstream> // Added for timestamp stringstream
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant> // Added for parameter binding
#include <limits> // Added for std::numeric_limits

// Project Includes
#include "core_services/common_data_types.h"
#include "common_utils/utilities/logging_utils.h"
#include "core_services/metadata/idataset_metadata_registry_backend.h" // For new interface
#include "core_services/metadata/dataset_metadata_types.h"     // For new types

// Include nlohmann/json
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// 修复日志宏定义
#ifndef LOG_MODULE_ERROR
#define LOG_MODULE_ERROR(module, ...) OSCEAN_LOG_ERROR("[" #module "] " __VA_ARGS__)
#define LOG_MODULE_WARN(module, ...) OSCEAN_LOG_WARN("[" #module "] " __VA_ARGS__)
#define LOG_MODULE_INFO(module, ...) OSCEAN_LOG_INFO("[" #module "] " __VA_ARGS__)
#define LOG_MODULE_DEBUG(module, ...) OSCEAN_LOG_DEBUG("[" #module "] " __VA_ARGS__)
#endif

// Logging Macros
#define STORAGE_LOG_ERROR(...) LOG_MODULE_ERROR("SQLiteStorage", __VA_ARGS__)
#define STORAGE_LOG_WARN(...) LOG_MODULE_WARN("SQLiteStorage", __VA_ARGS__)
#define STORAGE_LOG_INFO(...) LOG_MODULE_INFO("SQLiteStorage", __VA_ARGS__)
#define STORAGE_LOG_DEBUG(...) LOG_MODULE_DEBUG("SQLiteStorage", __VA_ARGS__)

namespace oscean {
namespace core_services {
namespace metadata {
namespace storage {

// 添加变体访问运算符
template<typename T>
struct VariantVisitor {
    std::stringstream& ss;
    VariantVisitor(std::stringstream& ss) : ss(ss) {}
    
    void operator()(const T& value) {
        ss << value;
    }
};

// SQLiteStorage::PreparedStatements 析构函数实现
SQLiteStorage::PreparedStatements::~PreparedStatements() {
    if (insertFile && insertFile != nullptr) sqlite3_finalize(insertFile);
    if (updateFile && updateFile != nullptr) sqlite3_finalize(updateFile);
    if (deleteFile && deleteFile != nullptr) sqlite3_finalize(deleteFile);
    if (insertVariable && insertVariable != nullptr) sqlite3_finalize(insertVariable);
    if (deleteVariablesForFile && deleteVariablesForFile != nullptr) sqlite3_finalize(deleteVariablesForFile);
    if (findFilesBase && findFilesBase != nullptr) sqlite3_finalize(findFilesBase);
    if (getFile && getFile != nullptr) sqlite3_finalize(getFile);
    if (getTimeRange && getTimeRange != nullptr) sqlite3_finalize(getTimeRange);
    if (getSpatialExtent && getSpatialExtent != nullptr) sqlite3_finalize(getSpatialExtent);
    if (getVariables && getVariables != nullptr) sqlite3_finalize(getVariables);
    if (insertKeyValue && insertKeyValue != nullptr) sqlite3_finalize(insertKeyValue);
    if (getKeyValue && getKeyValue != nullptr) sqlite3_finalize(getKeyValue);
    if (deleteKeyValue && deleteKeyValue != nullptr) sqlite3_finalize(deleteKeyValue);
    if (insertDataset && insertDataset != nullptr) sqlite3_finalize(insertDataset);
    if (updateDataset && updateDataset != nullptr) sqlite3_finalize(updateDataset);
    if (getDatasetById && getDatasetById != nullptr) sqlite3_finalize(getDatasetById);
    if (deleteDatasetById && deleteDatasetById != nullptr) sqlite3_finalize(deleteDatasetById);
    if (getAllDatasetIds && getAllDatasetIds != nullptr) sqlite3_finalize(getAllDatasetIds);
    if (getDatasetCount && getDatasetCount != nullptr) sqlite3_finalize(getDatasetCount);
    if (clearAllDatasets && clearAllDatasets != nullptr) sqlite3_finalize(clearAllDatasets);
    if (findDatasets && findDatasets != nullptr) sqlite3_finalize(findDatasets);
}

// 帮助函数
std::string getStringFromColumn(sqlite3_stmt* stmt, int colIndex) {
    const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, colIndex));
    return text ? std::string(text) : std::string();
}

double getDoubleFromColumn(sqlite3_stmt* stmt, int colIndex, double defaultValue = 0.0) {
    if (sqlite3_column_type(stmt, colIndex) == SQLITE_NULL) {
        return defaultValue;
    }
    return sqlite3_column_double(stmt, colIndex);
}

Timestamp getTimestampFromColumn(sqlite3_stmt* stmt, int colIndex, Timestamp defaultValue = 0) {
    if (sqlite3_column_type(stmt, colIndex) == SQLITE_NULL) {
        return defaultValue;
    }
    return static_cast<Timestamp>(sqlite3_column_int64(stmt, colIndex));
}

json attributeValueToJson(const AttributeValue& value) {
    return std::visit([](const auto& v) -> json {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return json(nullptr);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return json(v);
        } else if constexpr (std::is_same_v<T, int>) {
            return json(v);
        } else if constexpr (std::is_same_v<T, double>) {
            return json(v);
        } else if constexpr (std::is_same_v<T, bool>) {
            return json(v);
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            return json(v);
        } else if constexpr (std::is_same_v<T, std::vector<int>>) {
            return json(v);
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            return json(v);
        } else {
            return json(nullptr);
        }
    }, value);
}

AttributeValue jsonToAttributeValue(const json& j) {
    if (j.is_string()) {
        return j.get<std::string>();
    } else if (j.is_number_integer()) {
        return j.get<int>();
    } else if (j.is_number_float()) {
        return j.get<double>();
    } else if (j.is_boolean()) {
        return j.get<bool>();
    }
    return std::string();
}

DatasetMetadataEntry mapRowToDatasetMetadataEntry(sqlite3_stmt* stmt) {
    DatasetMetadataEntry entry;
    
    entry.id = getStringFromColumn(stmt, 0);
    entry.logicalName = getStringFromColumn(stmt, 1);
    
    // 文件路径列表（从JSON解析）
    std::string filePathsJson = getStringFromColumn(stmt, 2);
    if (!filePathsJson.empty()) {
        try {
            json j = json::parse(filePathsJson);
            for (const auto& path : j) {
                entry.filePaths.push_back(path.get<std::string>());
            }
        } catch (const json::exception& e) {
            STORAGE_LOG_WARN("Failed to parse file paths JSON for dataset {}: {}", entry.id, e.what());
        }
    }
    
    // 时间覆盖范围
    auto startTimestamp = getTimestampFromColumn(stmt, 3);
    auto endTimestamp = getTimestampFromColumn(stmt, 4);
    entry.timeCoverage = TimeRange(
        std::chrono::system_clock::time_point(std::chrono::milliseconds(startTimestamp)),
        std::chrono::system_clock::time_point(std::chrono::milliseconds(endTimestamp))
    );
    
    // 空间覆盖范围
    entry.spatialCoverage.minX = getDoubleFromColumn(stmt, 5);
    entry.spatialCoverage.minY = getDoubleFromColumn(stmt, 6);
    entry.spatialCoverage.maxX = getDoubleFromColumn(stmt, 7);
    entry.spatialCoverage.maxY = getDoubleFromColumn(stmt, 8);
    entry.spatialCoverage.crsId = getStringFromColumn(stmt, 9);
    
    // 格式和数据源
    entry.format = getStringFromColumn(stmt, 10);
    entry.dataSource = getStringFromColumn(stmt, 11);
    entry.processingLevel = getStringFromColumn(stmt, 12);
    
    // 元数据属性
    std::string metadataJson = getStringFromColumn(stmt, 13);
    if (!metadataJson.empty()) {
        try {
            json j = json::parse(metadataJson);
            for (auto it = j.begin(); it != j.end(); ++it) {
                entry.attributes[it.key()] = jsonToAttributeValue(it.value());
            }
        } catch (const json::exception& e) {
            STORAGE_LOG_WARN("Failed to parse metadata JSON for dataset {}: {}", entry.id, e.what());
        }
    }
    
    // 变量列表（需要单独查询）
    // 这里只设置基本信息，变量信息需要调用者单独获取
    
    // 时间戳
    entry.lastIndexedTime = getTimestampFromColumn(stmt, 14);
    
    // 备注
    std::string notes = getStringFromColumn(stmt, 15);
    if (!notes.empty()) {
        entry.notes = notes;
    }
    
    return entry;
}

FileMetadata mapRowToFileMetadataBase(sqlite3_stmt* stmt) {
    FileMetadata metadata;
    
    metadata.fileId = getStringFromColumn(stmt, 0);
    metadata.filePath = getStringFromColumn(stmt, 1);
    metadata.fileName = getStringFromColumn(stmt, 2);
    metadata.format = getStringFromColumn(stmt, 3);
    
    // 注意：CRS信息需要通过SQLiteStorage实例的getCRSInfoForFile方法获取
    // 这里暂时不加载CRS信息，由调用者负责补充
    
    // 如果专用表中没有CRS信息，回退到旧的crs_definition字段
    if (metadata.crs.wktext.empty() && metadata.crs.projString.empty()) {
        std::string crsDefinition = getStringFromColumn(stmt, 4);
        if (!crsDefinition.empty()) {
            metadata.crs.wktext = crsDefinition;
            metadata.crs.wkt = crsDefinition; // 兼容
            
            // 尝试检测格式
            if (crsDefinition.find("EPSG:") == 0) {
                metadata.crs.authorityName = "EPSG";
                metadata.crs.authority = "EPSG";
                try {
                    int epsgCode = std::stoi(crsDefinition.substr(5));
                    metadata.crs.epsgCode = epsgCode;
                    metadata.crs.authorityCode = std::to_string(epsgCode);
                    metadata.crs.code = metadata.crs.authorityCode;
                    metadata.crs.id = crsDefinition;
                } catch (...) {
                    // 解析失败，保持原始字符串
                }
            } else if (crsDefinition.find("+proj=") != std::string::npos) {
                // PROJ格式
                metadata.crs.projString = crsDefinition;
                metadata.crs.proj4text = crsDefinition;
            }
        }
    }
    
    // 空间边界
    metadata.spatialCoverage.minX = getDoubleFromColumn(stmt, 5);
    metadata.spatialCoverage.minY = getDoubleFromColumn(stmt, 6);
    metadata.spatialCoverage.maxX = getDoubleFromColumn(stmt, 7);
    metadata.spatialCoverage.maxY = getDoubleFromColumn(stmt, 8);
    
    // 时间范围 - 从毫秒时间戳转换为时间点
    auto startTimestamp = getTimestampFromColumn(stmt, 9);
    auto endTimestamp = getTimestampFromColumn(stmt, 10);
    
    // 转换毫秒时间戳为标准时间点
    metadata.timeRange.startTime = std::chrono::system_clock::time_point(
        std::chrono::milliseconds(startTimestamp));
    metadata.timeRange.endTime = std::chrono::system_clock::time_point(
        std::chrono::milliseconds(endTimestamp));
    
    // 元数据 JSON
    std::string metadataJson = getStringFromColumn(stmt, 11);
    if (!metadataJson.empty()) {
        try {
            json j = json::parse(metadataJson);
            for (auto it = j.begin(); it != j.end(); ++it) {
                metadata.metadata[it.key()] = it.value().get<std::string>();
            }
        } catch (const json::exception& e) {
            STORAGE_LOG_WARN("Failed to parse metadata JSON for file {}: {}", metadata.fileId, e.what());
        }
    }
    
    // 注意: variables 字段需要单独查询，这里不填充
    
    return metadata;
}

// SQLiteStorage 构造函数
SQLiteStorage::SQLiteStorage(const std::string& dbPath) 
    : dbPath_(dbPath), db_(nullptr)
{
    STORAGE_LOG_INFO("Creating SQLiteStorage with database path: {}", dbPath_);
}

// SQLiteStorage 析构函数
SQLiteStorage::~SQLiteStorage() {
    close(); // 确保数据库关闭和语句最终化
}

// executeInTransaction 实现 - 修复参数类型
bool SQLiteStorage::executeInTransaction(const std::function<bool()>& transactionBody) {
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Transaction failed: Storage not initialized or DB connection is null.");
        return false;
    }

    char* errMsg = nullptr;
    // Begin transaction
    int rc = sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("Failed to begin transaction: {}", errMsg ? errMsg : "Unknown error");
        sqlite3_free(errMsg);
        return false;
    }
    STORAGE_LOG_DEBUG("Transaction started.");

    bool success = false;
    try {
        // Execute the provided transaction body
        success = transactionBody();
    } catch (const std::exception& e) {
        STORAGE_LOG_ERROR("Exception caught during transaction: {}", e.what());
        success = false;
    } catch (...) {
        STORAGE_LOG_ERROR("Unknown exception caught during transaction.");
        success = false;
    }

    // Commit or rollback based on success
    if (success) {
        rc = sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            STORAGE_LOG_ERROR("Failed to commit transaction: {}", errMsg ? errMsg : "Unknown error");
            sqlite3_free(errMsg);
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }
        STORAGE_LOG_DEBUG("Transaction committed successfully.");
        return true;
    } else {
        rc = sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            STORAGE_LOG_ERROR("Failed to rollback transaction: {}", errMsg ? errMsg : "Unknown error");
            sqlite3_free(errMsg);
            return false;
        }
        STORAGE_LOG_DEBUG("Transaction rolled back.");
        return false;
    }
}

bool SQLiteStorage::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (isInitialized_) {
        return true;
    }

    // 确保父目录存在
    try {
        std::filesystem::path dbFilePath(dbPath_);
        if (dbFilePath.has_parent_path()) {
            std::filesystem::create_directories(dbFilePath.parent_path());
        }
    } catch (const std::filesystem::filesystem_error& e) {
        STORAGE_LOG_ERROR("无法创建数据库目录 {}: {}", dbPath_, e.what());
        return false;
    }

    int rc = sqlite3_open_v2(dbPath_.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("无法打开数据库 '{}': {}", dbPath_, sqlite3_errmsg(db_));
        sqlite3_close(db_); 
        db_ = nullptr;
        return false;
    }

    STORAGE_LOG_INFO("数据库已打开: {}", dbPath_);

    // Create tables
    if (!this->createTables()) {
        STORAGE_LOG_ERROR("创建数据库表失败: {}", dbPath_);
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }

    // Prepare statements
    if (!this->prepareStatements()) {
        STORAGE_LOG_ERROR("准备 SQL 语句失败: {}", dbPath_);
        this->finalizeStatements();
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    STORAGE_LOG_INFO("SQL 语句准备完毕.");

    isInitialized_ = true;
    return true;
}

// finalizeStatements
void SQLiteStorage::finalizeStatements() {
    // 手动清理每个语句指针，然后重置为nullptr
    if (statements_.insertFile) sqlite3_finalize(statements_.insertFile);
    if (statements_.updateFile) sqlite3_finalize(statements_.updateFile);
    if (statements_.deleteFile) sqlite3_finalize(statements_.deleteFile);
    if (statements_.insertVariable) sqlite3_finalize(statements_.insertVariable);
    if (statements_.deleteVariablesForFile) sqlite3_finalize(statements_.deleteVariablesForFile);
    if (statements_.findFilesBase) sqlite3_finalize(statements_.findFilesBase);
    if (statements_.getFile) sqlite3_finalize(statements_.getFile);
    if (statements_.getTimeRange) sqlite3_finalize(statements_.getTimeRange);
    if (statements_.getSpatialExtent) sqlite3_finalize(statements_.getSpatialExtent);
    if (statements_.getVariables) sqlite3_finalize(statements_.getVariables);
    if (statements_.insertKeyValue) sqlite3_finalize(statements_.insertKeyValue);
    if (statements_.getKeyValue) sqlite3_finalize(statements_.getKeyValue);
    if (statements_.deleteKeyValue) sqlite3_finalize(statements_.deleteKeyValue);
    if (statements_.insertDataset) sqlite3_finalize(statements_.insertDataset);
    if (statements_.updateDataset) sqlite3_finalize(statements_.updateDataset);
    if (statements_.getDatasetById) sqlite3_finalize(statements_.getDatasetById);
    if (statements_.deleteDatasetById) sqlite3_finalize(statements_.deleteDatasetById);
    if (statements_.getAllDatasetIds) sqlite3_finalize(statements_.getAllDatasetIds);
    if (statements_.getDatasetCount) sqlite3_finalize(statements_.getDatasetCount);
    if (statements_.clearAllDatasets) sqlite3_finalize(statements_.clearAllDatasets);
    if (statements_.findDatasets) sqlite3_finalize(statements_.findDatasets);
    
    // 重置所有指针为nullptr
    statements_ = {};
    
    STORAGE_LOG_DEBUG("Prepared statements finalized.");
}

// prepareStatements
bool SQLiteStorage::prepareStatements() {
    if (!db_) return false;

    const char* sql;
    int rc;

    // KeyValue Statements
    sql = "REPLACE INTO metadata (key, value, category) VALUES (?, ?, ?);";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.insertKeyValue, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (insertKeyValue): {}", sqlite3_errmsg(db_)); return false; }

    sql = "SELECT value FROM metadata WHERE key = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.getKeyValue, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (getKeyValue): {}", sqlite3_errmsg(db_)); return false; }

    sql = "DELETE FROM metadata WHERE key = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.deleteKeyValue, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (deleteKeyValue): {}", sqlite3_errmsg(db_)); return false; }

    // Files Table Statements
    sql = "REPLACE INTO files (file_id, file_path, file_name, format, crs_definition, "
          "bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, "
          "time_start, time_end, metadata_json, last_indexed_time) "
          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.insertFile, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (insertFile): {}", sqlite3_errmsg(db_)); return false; }

    sql = "UPDATE files SET file_path = ?, file_name = ?, format = ?, crs_definition = ?, "
          "bbox_min_x = ?, bbox_min_y = ?, bbox_max_x = ?, bbox_max_y = ?, "
          "time_start = ?, time_end = ?, metadata_json = ?, last_indexed_time = ? "
          "WHERE file_id = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.updateFile, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (updateFile): {}", sqlite3_errmsg(db_)); return false; }

    sql = "DELETE FROM files WHERE file_id = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.deleteFile, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (deleteFile): {}", sqlite3_errmsg(db_)); return false; }

    // File Variables Table Statements
    sql = "INSERT OR IGNORE INTO file_variables (file_id, variable_name) VALUES (?, ?);";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.insertVariable, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (insertVariable): {}", sqlite3_errmsg(db_)); return false; }

    sql = "DELETE FROM file_variables WHERE file_id = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.deleteVariablesForFile, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (deleteVariablesForFile): {}", sqlite3_errmsg(db_)); return false; }

    // 🆕 CRS信息表相关语句
    sql = "REPLACE INTO crs_info (file_id, wkt_definition, proj_string, epsg_code, "
          "authority_name, authority_code, is_geographic, is_projected, "
          "linear_unit_name, linear_unit_to_meter, angular_unit_name, angular_unit_to_radian, "
          "authority, code, id, wkt, proj4text, crs_json) "
          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.insertCrsInfo, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (insertCrsInfo): {}", sqlite3_errmsg(db_)); return false; }

    sql = "SELECT wkt_definition, proj_string, epsg_code, authority_name, authority_code, "
          "is_geographic, is_projected, linear_unit_name, linear_unit_to_meter, "
          "angular_unit_name, angular_unit_to_radian, authority, code, id, wkt, proj4text, crs_json "
          "FROM crs_info WHERE file_id = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.getCrsInfo, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (getCrsInfo): {}", sqlite3_errmsg(db_)); return false; }

    sql = "DELETE FROM crs_info WHERE file_id = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.deleteCrsInfo, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (deleteCrsInfo): {}", sqlite3_errmsg(db_)); return false; }

    // Get Statements
    sql = "SELECT file_id, file_path, file_name, format, crs_definition, "
          "bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, "
          "time_start, time_end, metadata_json, last_indexed_time "
          "FROM files WHERE file_id = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.getFile, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (getFile): {}", sqlite3_errmsg(db_)); return false; }

    // Dataset statements - 修复字段映射
    sql = "REPLACE INTO datasets (dataset_id, logical_name, file_paths_json, "
          "temporal_start, temporal_end, spatial_min_x, spatial_min_y, spatial_max_x, spatial_max_y, "
          "spatial_crs_id, format, data_source, processing_level, metadata_json, "
          "last_indexed_time, notes) "
          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.insertDataset, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (insertDataset): {}", sqlite3_errmsg(db_)); return false; }

    sql = "SELECT dataset_id, logical_name, file_paths_json, temporal_start, temporal_end, "
          "spatial_min_x, spatial_min_y, spatial_max_x, spatial_max_y, spatial_crs_id, "
          "format, data_source, processing_level, metadata_json, last_indexed_time, notes "
          "FROM datasets WHERE dataset_id = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.getDatasetById, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (getDatasetById): {}", sqlite3_errmsg(db_)); return false; }

    sql = "DELETE FROM datasets WHERE dataset_id = ?;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.deleteDatasetById, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (deleteDatasetById): {}", sqlite3_errmsg(db_)); return false; }

    sql = "SELECT dataset_id FROM datasets;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.getAllDatasetIds, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (getAllDatasetIds): {}", sqlite3_errmsg(db_)); return false; }

    sql = "SELECT COUNT(*) FROM datasets;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.getDatasetCount, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (getDatasetCount): {}", sqlite3_errmsg(db_)); return false; }

    sql = "DELETE FROM datasets;";
    rc = sqlite3_prepare_v2(db_, sql, -1, &statements_.clearAllDatasets, nullptr);
    if (rc != SQLITE_OK) { STORAGE_LOG_ERROR("Prepare failed (clearAllDatasets): {}", sqlite3_errmsg(db_)); return false; }

    return true;
}

// createTables 实现
bool SQLiteStorage::createTables() {
    if (!db_) return false;

    const char* createTablesSql = R"(
        -- 基础元数据表
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 文件主表 - 增强版，支持完整元数据
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            file_name TEXT,
            format TEXT,
            file_size INTEGER,
            creation_time INTEGER,
            modification_time INTEGER,
            extraction_timestamp INTEGER,
            data_quality REAL DEFAULT 1.0,
            completeness REAL DEFAULT 1.0,
            checksum TEXT,
            crs_definition TEXT,
            bbox_min_x REAL,
            bbox_min_y REAL,
            bbox_max_x REAL,
            bbox_max_y REAL,
            time_start TEXT,  -- 中文格式时间字符串
            time_end TEXT,    -- 中文格式时间字符串
            temporal_resolution_text TEXT,  -- 中文时间分辨率：年、月、周、日、时、分、秒
            temporal_resolution_seconds REAL,
            metadata_json TEXT,
            last_indexed_time INTEGER
        );

        -- 数据来源信息表
        CREATE TABLE IF NOT EXISTS data_source_info (
            file_id TEXT PRIMARY KEY,
            source_id TEXT,
            institution TEXT,
            processing_level TEXT,
            source_url TEXT,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        -- 空间信息表
        CREATE TABLE IF NOT EXISTS spatial_info (
            file_id TEXT PRIMARY KEY,
            min_longitude REAL,
            max_longitude REAL,
            min_latitude REAL,
            max_latitude REAL,
            min_elevation REAL,
            max_elevation REAL,
            min_depth REAL,
            max_depth REAL,
            center_longitude REAL,
            center_latitude REAL,
            coordinate_system TEXT,
            spatial_resolution_lon REAL,
            spatial_resolution_lat REAL,
            vertical_resolution REAL,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        -- 🆕 CRS多格式支持表
        CREATE TABLE IF NOT EXISTS crs_info (
            file_id TEXT PRIMARY KEY,
            wkt_definition TEXT,           -- WKT格式定义
            proj_string TEXT,              -- PROJ字符串
            epsg_code INTEGER,             -- EPSG代码
            authority_name TEXT,           -- 权威机构名称 (EPSG, ESRI等)
            authority_code TEXT,           -- 权威代码
            is_geographic INTEGER DEFAULT 0,  -- 是否地理坐标系
            is_projected INTEGER DEFAULT 0,   -- 是否投影坐标系
            linear_unit_name TEXT,         -- 线性单位名称
            linear_unit_to_meter REAL,     -- 线性单位到米转换系数
            angular_unit_name TEXT,        -- 角度单位名称
            angular_unit_to_radian REAL,   -- 角度单位到弧度转换系数
            -- 兼容字段（向后兼容）
            authority TEXT,                -- 兼容：authorityName
            code TEXT,                     -- 兼容：authorityCode
            id TEXT,                       -- 兼容：authority:code格式
            wkt TEXT,                      -- 兼容：wkt定义
            proj4text TEXT,                -- 兼容：PROJ字符串
            crs_json TEXT,                 -- 完整CRS信息JSON存储
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        -- 时间信息表 - 支持中文格式
        CREATE TABLE IF NOT EXISTS temporal_info (
            file_id TEXT PRIMARY KEY,
            start_time_chinese TEXT,  -- 中文格式：2024年1月1日 08时30分00秒
            end_time_chinese TEXT,    -- 中文格式：2024年12月31日 18时45分30秒
            start_time_iso TEXT,      -- ISO格式：2024-01-01T08:30:00Z
            end_time_iso TEXT,        -- ISO格式：2024-12-31T18:45:30Z
            temporal_resolution_chinese TEXT,  -- 中文：日、月、年等
            temporal_resolution_seconds REAL,
            time_units TEXT,
            calendar TEXT DEFAULT 'gregorian',
            is_regular INTEGER DEFAULT 1,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        -- 变量信息表 - 完整版
        CREATE TABLE IF NOT EXISTS variable_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            name TEXT NOT NULL,
            standard_name TEXT,
            long_name TEXT,
            units TEXT,
            data_type TEXT,
            dimensions TEXT,  -- JSON array of dimension names
            shape TEXT,       -- JSON array of dimension sizes
            fill_value TEXT,
            valid_range TEXT, -- JSON array [min, max]
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        -- 🎯 核心新增：多重分类标签表（最多3个分类）
        CREATE TABLE IF NOT EXISTS variable_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            variable_id INTEGER NOT NULL,
            file_id TEXT NOT NULL,
            variable_name TEXT NOT NULL,
            classification_1 TEXT,      -- 主分类：海洋流、温度、盐度、海平面等
            classification_2 TEXT,      -- 次分类：表层、深层、异常等
            classification_3 TEXT,      -- 细分类：东西向、南北向、季节性等
            classification_1_chinese TEXT,  -- 中文主分类
            classification_2_chinese TEXT,  -- 中文次分类
            classification_3_chinese TEXT,  -- 中文细分类
            confidence_1 REAL DEFAULT 1.0,  -- 主分类置信度
            confidence_2 REAL DEFAULT 1.0,  -- 次分类置信度
            confidence_3 REAL DEFAULT 1.0,  -- 细分类置信度
            classification_method TEXT DEFAULT 'intelligent',  -- 分类方法：intelligent, manual, rule_based
            classification_timestamp INTEGER,  -- 分类时间戳
            FOREIGN KEY (variable_id) REFERENCES variable_info(id) ON DELETE CASCADE,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        -- 海洋环境专用变量表 - 增强版
        CREATE TABLE IF NOT EXISTS ocean_variables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            variable_id INTEGER NOT NULL,
            product_category TEXT,      -- 'SeaLevel', 'OceanCurrents', 'Temperature', 'Salinity'
            product_category_chinese TEXT,  -- 中文：海平面、海洋流、温度、盐度
            model_name TEXT,            -- 'HYCOM', 'NEMO', 'SatelliteObservation'
            measurement_type TEXT,      -- observation, model, analysis
            measurement_type_chinese TEXT,  -- 中文：观测、模型、分析
            quality_flag INTEGER DEFAULT 0,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE,
            FOREIGN KEY (variable_id) REFERENCES variable_info(id) ON DELETE CASCADE
        );

        -- 地形底质专用变量表
        CREATE TABLE IF NOT EXISTS topography_variables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            variable_id INTEGER NOT NULL,
            topo_parameter TEXT,        -- elevation, depth, bathymetry, slope, aspect
            topo_parameter_chinese TEXT,  -- 中文：高程、深度、水深、坡度、坡向
            vertical_datum TEXT,        -- WGS84, MSL, etc.
            measurement_method TEXT,    -- lidar, sonar, satellite, survey
            measurement_method_chinese TEXT,  -- 中文：激光雷达、声纳、卫星、测量
            accuracy_meters REAL,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE,
            FOREIGN KEY (variable_id) REFERENCES variable_info(id) ON DELETE CASCADE
        );

        -- 边界线专用变量表
        CREATE TABLE IF NOT EXISTS boundary_variables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            variable_id INTEGER NOT NULL,
            boundary_type TEXT,         -- coastline, country_border, maritime_boundary
            boundary_type_chinese TEXT, -- 中文：海岸线、国界、海域边界
            line_classification TEXT,   -- high_water_line, low_water_line
            line_classification_chinese TEXT,  -- 中文：高潮线、低潮线
            survey_scale INTEGER,       -- 1:50000, 1:250000, etc.
            accuracy_meters REAL,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE,
            FOREIGN KEY (variable_id) REFERENCES variable_info(id) ON DELETE CASCADE
        );

        -- 声纳传播专用表
        CREATE TABLE IF NOT EXISTS propagation_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            platform_name TEXT,
            platform_name_chinese TEXT,  -- 中文平台名称
            platform_id TEXT,
            sensor_name TEXT,
            sensor_name_chinese TEXT,    -- 中文传感器名称
            sensor_id TEXT,
            sensor_working_mode TEXT,
            sensor_working_mode_chinese TEXT,  -- 中文工作模式
            sensor_working_frequency_hz REAL,
            propagation_model TEXT,
            propagation_model_chinese TEXT,    -- 中文传播模型
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        -- 属性表 - 支持中文
        CREATE TABLE IF NOT EXISTS attributes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            attribute_name TEXT NOT NULL,
            attribute_name_chinese TEXT,  -- 中文属性名
            attribute_value TEXT,
            attribute_value_chinese TEXT, -- 中文属性值
            attribute_type TEXT DEFAULT 'string',
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        -- 保留原有的简化表结构以兼容现有代码
        CREATE TABLE IF NOT EXISTS file_variables (
            file_id TEXT,
            variable_name TEXT,
            PRIMARY KEY (file_id, variable_name),
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            logical_name TEXT NOT NULL,
            file_paths_json TEXT NOT NULL,
            temporal_start TEXT,
            temporal_end TEXT,
            spatial_min_x REAL,
            spatial_min_y REAL,
            spatial_max_x REAL,
            spatial_max_y REAL,
            spatial_crs_id TEXT,
            format TEXT,
            data_source TEXT,
            processing_level TEXT,
            metadata_json TEXT,
            last_indexed_time INTEGER,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS dataset_variables (
            dataset_id TEXT,
            variable_name TEXT,
            variable_standard_name TEXT,
            variable_units TEXT,
            variable_description TEXT,
            PRIMARY KEY (dataset_id, variable_name),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );

        -- 🎯 核心索引 - 优化查询性能
        CREATE INDEX IF NOT EXISTS idx_files_path ON files(file_path);
        CREATE INDEX IF NOT EXISTS idx_files_time_chinese ON files(time_start, time_end);
        CREATE INDEX IF NOT EXISTS idx_files_bbox ON files(bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y);
        CREATE INDEX IF NOT EXISTS idx_files_format ON files(format);
        CREATE INDEX IF NOT EXISTS idx_files_quality ON files(data_quality, completeness);
        
        CREATE INDEX IF NOT EXISTS idx_variable_info_name ON variable_info(name);
        CREATE INDEX IF NOT EXISTS idx_variable_info_file ON variable_info(file_id);
        
        CREATE INDEX IF NOT EXISTS idx_classifications_primary ON variable_classifications(classification_1, classification_1_chinese);
        CREATE INDEX IF NOT EXISTS idx_classifications_secondary ON variable_classifications(classification_2, classification_2_chinese);
        CREATE INDEX IF NOT EXISTS idx_classifications_tertiary ON variable_classifications(classification_3, classification_3_chinese);
        CREATE INDEX IF NOT EXISTS idx_classifications_confidence ON variable_classifications(confidence_1, confidence_2, confidence_3);
        CREATE INDEX IF NOT EXISTS idx_classifications_method ON variable_classifications(classification_method);
        
        CREATE INDEX IF NOT EXISTS idx_ocean_category ON ocean_variables(product_category, product_category_chinese);
        CREATE INDEX IF NOT EXISTS idx_ocean_model ON ocean_variables(model_name);
        CREATE INDEX IF NOT EXISTS idx_ocean_measurement ON ocean_variables(measurement_type, measurement_type_chinese);
        
        CREATE INDEX IF NOT EXISTS idx_topo_parameter ON topography_variables(topo_parameter, topo_parameter_chinese);
        CREATE INDEX IF NOT EXISTS idx_topo_method ON topography_variables(measurement_method, measurement_method_chinese);
        
        CREATE INDEX IF NOT EXISTS idx_boundary_type ON boundary_variables(boundary_type, boundary_type_chinese);
        CREATE INDEX IF NOT EXISTS idx_boundary_classification ON boundary_variables(line_classification, line_classification_chinese);
        
        CREATE INDEX IF NOT EXISTS idx_propagation_platform ON propagation_context(platform_name, platform_name_chinese);
        CREATE INDEX IF NOT EXISTS idx_propagation_sensor ON propagation_context(sensor_name, sensor_name_chinese);
        CREATE INDEX IF NOT EXISTS idx_propagation_frequency ON propagation_context(sensor_working_frequency_hz);
        
        CREATE INDEX IF NOT EXISTS idx_spatial_bounds ON spatial_info(min_longitude, max_longitude, min_latitude, max_latitude);
        CREATE INDEX IF NOT EXISTS idx_spatial_elevation ON spatial_info(min_elevation, max_elevation);
        CREATE INDEX IF NOT EXISTS idx_spatial_depth ON spatial_info(min_depth, max_depth);
        
        CREATE INDEX IF NOT EXISTS idx_temporal_chinese ON temporal_info(start_time_chinese, end_time_chinese);
        CREATE INDEX IF NOT EXISTS idx_temporal_iso ON temporal_info(start_time_iso, end_time_iso);
        CREATE INDEX IF NOT EXISTS idx_temporal_resolution ON temporal_info(temporal_resolution_chinese);
        
        CREATE INDEX IF NOT EXISTS idx_attributes_name ON attributes(attribute_name, attribute_name_chinese);
        
        -- 兼容性索引
        CREATE INDEX IF NOT EXISTS idx_file_variables_name ON file_variables(variable_name);
        CREATE INDEX IF NOT EXISTS idx_datasets_temporal ON datasets(temporal_start, temporal_end);
        CREATE INDEX IF NOT EXISTS idx_datasets_spatial ON datasets(spatial_min_x, spatial_min_y, spatial_max_x, spatial_max_y);
        CREATE INDEX IF NOT EXISTS idx_datasets_format ON datasets(format);
        CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets(data_source);
    )";

    char* errMsg = nullptr;
    int rc = sqlite3_exec(db_, createTablesSql, nullptr, nullptr, &errMsg);
    
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("创建数据库表失败: {}", errMsg ? errMsg : "未知错误");
        if (errMsg) {
            sqlite3_free(errMsg);
        }
        return false;
    }
    
    STORAGE_LOG_INFO("✅ 数据库表创建成功，支持多重分类标签和中文时间格式");
    return true;
}

// 实现IMetadataStorage接口方法
std::optional<std::string> SQLiteStorage::getValue(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_ || !statements_.getKeyValue) {
        STORAGE_LOG_ERROR("Storage not initialized or statements not prepared");
        return std::nullopt;
    }
    
    sqlite3_bind_text(statements_.getKeyValue, 1, key.c_str(), -1, SQLITE_STATIC);
    
    int rc = sqlite3_step(statements_.getKeyValue);
    std::optional<std::string> result;
    
    if (rc == SQLITE_ROW) {
        result = getStringFromColumn(statements_.getKeyValue, 0);
    } else if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Failed to get value for key {}: {}", key, sqlite3_errmsg(db_));
    }
    
    sqlite3_reset(statements_.getKeyValue);
    return result;
}

bool SQLiteStorage::setValue(const std::string& key, const std::string& value, const std::string& category) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_ || !statements_.insertKeyValue) {
        STORAGE_LOG_ERROR("Storage not initialized or statements not prepared");
        return false;
    }
    
    sqlite3_bind_text(statements_.insertKeyValue, 1, key.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(statements_.insertKeyValue, 2, value.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(statements_.insertKeyValue, 3, category.c_str(), -1, SQLITE_STATIC);
    
    int rc = sqlite3_step(statements_.insertKeyValue);
    sqlite3_reset(statements_.insertKeyValue);
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Failed to set value for key {}: {}", key, sqlite3_errmsg(db_));
        return false;
    }
    
    return true;
}

bool SQLiteStorage::removeKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_ || !statements_.deleteKeyValue) {
        STORAGE_LOG_ERROR("Storage not initialized or statements not prepared");
        return false;
    }
    
    sqlite3_bind_text(statements_.deleteKeyValue, 1, key.c_str(), -1, SQLITE_STATIC);
    
    int rc = sqlite3_step(statements_.deleteKeyValue);
    sqlite3_reset(statements_.deleteKeyValue);
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Failed to remove key {}: {}", key, sqlite3_errmsg(db_));
        return false;
    }
    
    return true;
}

std::vector<MetadataEntry> SQLiteStorage::getAllMetadata() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MetadataEntry> entries;
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return entries;
    }
    
    const char* sql = "SELECT key, value, category FROM metadata ORDER BY key;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("Failed to prepare getAllMetadata query: {}", sqlite3_errmsg(db_));
        return entries;
    }
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        MetadataEntry entry;
        entry.key = getStringFromColumn(stmt, 0);
        entry.value = getStringFromColumn(stmt, 1);
        entry.category = getStringFromColumn(stmt, 2);
        entries.push_back(entry);
    }
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Error reading metadata: {}", sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    return entries;
}

std::vector<MetadataEntry> SQLiteStorage::getMetadataByCategory(const std::string& category) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MetadataEntry> entries;
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return entries;
    }
    
    const char* sql = "SELECT key, value, category FROM metadata WHERE category = ? ORDER BY key;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("Failed to prepare getMetadataByCategory query: {}", sqlite3_errmsg(db_));
        return entries;
    }
    
    sqlite3_bind_text(stmt, 1, category.c_str(), -1, SQLITE_STATIC);
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        MetadataEntry entry;
        entry.key = getStringFromColumn(stmt, 0);
        entry.value = getStringFromColumn(stmt, 1);
        entry.category = getStringFromColumn(stmt, 2);
        entries.push_back(entry);
    }
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Error reading metadata by category {}: {}", category, sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    return entries;
}

void SQLiteStorage::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (db_) {
        finalizeStatements();
        int rc = sqlite3_close(db_);
        if (rc != SQLITE_OK) {
            STORAGE_LOG_ERROR("Failed to close database: {}", sqlite3_errmsg(db_));
        }
        db_ = nullptr;
    }
    
    isInitialized_ = false;
}

// 实现文件查找功能
std::vector<FileInfo> SQLiteStorage::findFiles(const QueryCriteria& criteria) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FileInfo> results;
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return results;
    }
    
    // 首先检查数据库中是否有数据
    const char* countSql = "SELECT COUNT(*) FROM files";
    sqlite3_stmt* countStmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, countSql, -1, &countStmt, nullptr);
    if (rc == SQLITE_OK) {
        if (sqlite3_step(countStmt) == SQLITE_ROW) {
            int count = sqlite3_column_int(countStmt, 0);
            STORAGE_LOG_INFO("数据库中总共有 {} 条文件记录", count);
        }
        sqlite3_finalize(countStmt);
    }
    
    // 基础查询，查询所有文件
    std::string sql = "SELECT file_id, file_path, file_name, format, last_indexed_time FROM files WHERE 1=1";
    std::vector<std::string> params;
    
    // 构建查询条件 - 使用正确的字段名和宽松的匹配
    if (!criteria.dataTypes.empty()) {
        sql += " AND format IN (";
        for (size_t i = 0; i < criteria.dataTypes.size(); ++i) {
            if (i > 0) sql += ", ";
            sql += "?";
            params.push_back(std::to_string(static_cast<int>(criteria.dataTypes[i])));
        }
        sql += ")";
        STORAGE_LOG_INFO("添加数据类型过滤条件: {} 个类型", criteria.dataTypes.size());
    }
    
    // 时间范围查询 - 简化处理
    if (criteria.timeRange) {
        sql += " AND last_indexed_time IS NOT NULL";
        STORAGE_LOG_INFO("添加时间范围过滤条件");
    }
    
    // 变量过滤 - 如果指定了变量，通过JOIN查询
    if (!criteria.variablesInclude.empty()) {
        sql += " AND file_id IN (SELECT DISTINCT file_id FROM file_variables WHERE variable_name IN (";
        for (size_t i = 0; i < criteria.variablesInclude.size(); ++i) {
            if (i > 0) sql += ", ";
            sql += "?";
            params.push_back(criteria.variablesInclude[i]);
        }
        sql += "))";
        STORAGE_LOG_INFO("添加变量过滤条件: {} 个变量", criteria.variablesInclude.size());
    }
    
    sql += " ORDER BY last_indexed_time DESC LIMIT 1000"; // 限制结果数量
    
    STORAGE_LOG_INFO("执行查询SQL: {}", sql);
    STORAGE_LOG_INFO("查询参数数量: {}", params.size());
    
    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("Failed to prepare query: {}", sqlite3_errmsg(db_));
        return results;
    }
    
    // 绑定参数
    for (size_t i = 0; i < params.size(); ++i) {
        sqlite3_bind_text(stmt, static_cast<int>(i + 1), params[i].c_str(), -1, SQLITE_STATIC);
        STORAGE_LOG_INFO("绑定参数 {}: {}", i + 1, params[i]);
    }
    
    // 执行查询并收集结果
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        FileInfo fileInfo;
        fileInfo.id = getStringFromColumn(stmt, 0);         // file_id
        fileInfo.path = getStringFromColumn(stmt, 1);       // file_path
        
        STORAGE_LOG_INFO("找到文件: ID={}, Path={}", fileInfo.id, fileInfo.path);
        
        results.push_back(fileInfo);
    }
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Query execution failed: {}", sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    
    STORAGE_LOG_INFO("findFiles query returned {} results", results.size());
    return results;
}

std::optional<FileMetadata> SQLiteStorage::getFileMetadata(const std::string& fileId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_ || !statements_.getFile) {
        STORAGE_LOG_ERROR("Storage not initialized or statements not prepared");
        return std::nullopt;
    }
    
    sqlite3_bind_text(statements_.getFile, 1, fileId.c_str(), -1, SQLITE_STATIC);
    
    int rc = sqlite3_step(statements_.getFile);
    std::optional<FileMetadata> result;
    
    if (rc == SQLITE_ROW) {
        FileMetadata metadata = mapRowToFileMetadataBase(statements_.getFile);
        
        // 获取变量列表
        metadata.variables = getVariablesForFile(fileId);
        
        result = metadata;
    } else if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Failed to get file metadata for {}: {}", fileId, sqlite3_errmsg(db_));
    }
    
    sqlite3_reset(statements_.getFile);
    return result;
}

bool SQLiteStorage::addOrUpdateFileMetadata(const FileMetadata& metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return false;
    }
    
    return executeInTransaction([this, &metadata]() -> bool {
        // 删除旧的变量信息
        if (statements_.deleteVariablesForFile) {
            sqlite3_bind_text(statements_.deleteVariablesForFile, 1, metadata.fileId.c_str(), -1, SQLITE_STATIC);
            sqlite3_step(statements_.deleteVariablesForFile);
            sqlite3_reset(statements_.deleteVariablesForFile);
        }
        
        // 插入/更新文件信息
        if (!statements_.insertFile) {
            STORAGE_LOG_ERROR("insertFile statement not prepared");
            return false;
        }
        
        sqlite3_bind_text(statements_.insertFile, 1, metadata.fileId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(statements_.insertFile, 2, metadata.filePath.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(statements_.insertFile, 3, metadata.fileName.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(statements_.insertFile, 4, metadata.format.c_str(), -1, SQLITE_STATIC);
        
        // 🔧 修复CRS信息存储 - 序列化为完整的JSON格式
        json crsJson;
        if (!metadata.crs.authorityName.empty()) {
            crsJson["authorityName"] = metadata.crs.authorityName;
        }
        if (!metadata.crs.authorityCode.empty()) {
            crsJson["authorityCode"] = metadata.crs.authorityCode;
        }
        if (!metadata.crs.wktext.empty()) {
            crsJson["wktext"] = metadata.crs.wktext;
        }
        if (!metadata.crs.projString.empty()) {
            crsJson["projString"] = metadata.crs.projString;
        }
        if (metadata.crs.epsgCode.has_value()) {
            crsJson["epsgCode"] = metadata.crs.epsgCode.value();
        }
        crsJson["isGeographic"] = metadata.crs.isGeographic;
        crsJson["isProjected"] = metadata.crs.isProjected;
        if (!metadata.crs.linearUnitName.empty()) {
            crsJson["linearUnitName"] = metadata.crs.linearUnitName;
        }
        crsJson["linearUnitToMeter"] = metadata.crs.linearUnitToMeter;
        if (!metadata.crs.angularUnitName.empty()) {
            crsJson["angularUnitName"] = metadata.crs.angularUnitName;
        }
        crsJson["angularUnitToRadian"] = metadata.crs.angularUnitToRadian;
        
        std::string crsJsonStr = crsJson.empty() ? "" : crsJson.dump();
        sqlite3_bind_text(statements_.insertFile, 5, crsJsonStr.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(statements_.insertFile, 6, metadata.spatialCoverage.minX);
        sqlite3_bind_double(statements_.insertFile, 7, metadata.spatialCoverage.minY);
        sqlite3_bind_double(statements_.insertFile, 8, metadata.spatialCoverage.maxX);
        sqlite3_bind_double(statements_.insertFile, 9, metadata.spatialCoverage.maxY);
        
        // 时间范围 - 转换为毫秒时间戳存储
        auto startTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            metadata.timeRange.startTime.time_since_epoch()).count();
        auto endTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            metadata.timeRange.endTime.time_since_epoch()).count();
        sqlite3_bind_int64(statements_.insertFile, 10, startTime);
        sqlite3_bind_int64(statements_.insertFile, 11, endTime);
        
        // 元数据JSON
        json metadataJson;
        for (const auto& [key, value] : metadata.metadata) {
            metadataJson[key] = value;
        }
        std::string metadataStr = metadataJson.dump();
        sqlite3_bind_text(statements_.insertFile, 12, metadataStr.c_str(), -1, SQLITE_STATIC);
        
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        sqlite3_bind_int64(statements_.insertFile, 13, now);
        
        int rc = sqlite3_step(statements_.insertFile);
        sqlite3_reset(statements_.insertFile);
        
        if (rc != SQLITE_DONE) {
            STORAGE_LOG_ERROR("Failed to insert/update file metadata: {}", sqlite3_errmsg(db_));
            return false;
        }
        
        // 🆕 保存CRS信息到专用表
        if (!saveCRSInfoForFile(metadata.fileId, metadata.crs)) {
            STORAGE_LOG_WARN("Failed to save CRS info for file {}", metadata.fileId);
            // 不返回失败，因为主要数据已经保存成功
        }
        
        // 插入变量信息
        if (statements_.insertVariable) {
            for (const auto& variable : metadata.variables) {
                sqlite3_bind_text(statements_.insertVariable, 1, metadata.fileId.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(statements_.insertVariable, 2, variable.name.c_str(), -1, SQLITE_STATIC);
                
                rc = sqlite3_step(statements_.insertVariable);
                sqlite3_reset(statements_.insertVariable);
                
                if (rc != SQLITE_DONE) {
                    STORAGE_LOG_ERROR("Failed to insert variable {}: {}", variable.name, sqlite3_errmsg(db_));
                    return false;
                }
            }
        }
        
        return true;
    });
}

bool SQLiteStorage::removeFileMetadata(const std::string& fileId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_ || !statements_.deleteFile) {
        STORAGE_LOG_ERROR("Storage not initialized or statements not prepared");
        return false;
    }
    
    return executeInTransaction([this, &fileId]() -> bool {
        // 🆕 删除CRS信息
        if (!deleteCRSInfoForFile(fileId)) {
            STORAGE_LOG_WARN("Failed to delete CRS info for file {}", fileId);
            // 继续执行，不因此失败
        }
        
        // 删除变量信息（外键约束会自动处理，但手动删除更明确）
        if (statements_.deleteVariablesForFile) {
            sqlite3_bind_text(statements_.deleteVariablesForFile, 1, fileId.c_str(), -1, SQLITE_STATIC);
            sqlite3_step(statements_.deleteVariablesForFile);
            sqlite3_reset(statements_.deleteVariablesForFile);
        }
        
        // 删除文件信息
        sqlite3_bind_text(statements_.deleteFile, 1, fileId.c_str(), -1, SQLITE_STATIC);
        int rc = sqlite3_step(statements_.deleteFile);
        sqlite3_reset(statements_.deleteFile);
        
        if (rc != SQLITE_DONE) {
            STORAGE_LOG_ERROR("Failed to remove file metadata for {}: {}", fileId, sqlite3_errmsg(db_));
            return false;
        }
        
        return true;
    });
}

std::vector<FileMetadata> SQLiteStorage::findFilesByTimeRange(const Timestamp start, const Timestamp end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FileMetadata> results;
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return results;
    }
    
    const char* sql = "SELECT file_id, file_path, file_name, format, crs_definition, "
                      "bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, "
                      "time_start, time_end, metadata_json "
                      "FROM files WHERE time_start <= ? AND time_end >= ? "
                      "ORDER BY time_start;";
    
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("Failed to prepare findFilesByTimeRange query: {}", sqlite3_errmsg(db_));
        return results;
    }
    
    sqlite3_bind_int64(stmt, 1, end);
    sqlite3_bind_int64(stmt, 2, start);
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        FileMetadata metadata = mapRowToFileMetadataBase(stmt);
        // 获取变量列表
        metadata.variables = getVariablesForFile(metadata.fileId);
        results.push_back(metadata);
    }
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Error in findFilesByTimeRange query: {}", sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    return results;
}

std::vector<FileMetadata> SQLiteStorage::findFilesByBBox(const BoundingBox& bbox) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FileMetadata> results;
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return results;
    }
    
    const char* sql = "SELECT file_id, file_path, file_name, format, crs_definition, "
                      "bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, "
                      "time_start, time_end, metadata_json "
                      "FROM files WHERE bbox_min_x <= ? AND bbox_max_x >= ? AND "
                      "bbox_min_y <= ? AND bbox_max_y >= ? "
                      "ORDER BY file_path;";
    
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("Failed to prepare findFilesByBBox query: {}", sqlite3_errmsg(db_));
        return results;
    }
    
    sqlite3_bind_double(stmt, 1, bbox.maxX);
    sqlite3_bind_double(stmt, 2, bbox.minX);
    sqlite3_bind_double(stmt, 3, bbox.maxY);
    sqlite3_bind_double(stmt, 4, bbox.minY);
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        FileMetadata metadata = mapRowToFileMetadataBase(stmt);
        // 获取变量列表
        metadata.variables = getVariablesForFile(metadata.fileId);
        results.push_back(metadata);
    }
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Error in findFilesByBBox query: {}", sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    return results;
}

std::vector<std::string> SQLiteStorage::getAvailableVariables() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> variables;
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return variables;
    }
    
    const char* sql = "SELECT DISTINCT variable_name FROM file_variables ORDER BY variable_name;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("Failed to prepare getAvailableVariables query: {}", sqlite3_errmsg(db_));
        return variables;
    }
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        variables.push_back(getStringFromColumn(stmt, 0));
    }
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Error in getAvailableVariables query: {}", sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    return variables;
}

std::optional<std::pair<Timestamp, Timestamp>> SQLiteStorage::getTimeRange(
    const std::optional<std::string>& variableName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return std::nullopt;
    }
    
    std::string sql;
    sqlite3_stmt* stmt = nullptr;
    
    if (variableName) {
        sql = "SELECT MIN(f.time_start), MAX(f.time_end) FROM files f "
              "JOIN file_variables fv ON f.file_id = fv.file_id "
              "WHERE fv.variable_name = ?;";
        
        int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            STORAGE_LOG_ERROR("Failed to prepare getTimeRange query: {}", sqlite3_errmsg(db_));
            return std::nullopt;
        }
        
        sqlite3_bind_text(stmt, 1, variableName->c_str(), -1, SQLITE_STATIC);
    } else {
        sql = "SELECT MIN(time_start), MAX(time_end) FROM files;";
        
        int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            STORAGE_LOG_ERROR("Failed to prepare getTimeRange query: {}", sqlite3_errmsg(db_));
            return std::nullopt;
        }
    }
    
    int rc = sqlite3_step(stmt);
    std::optional<std::pair<Timestamp, Timestamp>> result;
    
    if (rc == SQLITE_ROW) {
        auto minTime = getTimestampFromColumn(stmt, 0);
        auto maxTime = getTimestampFromColumn(stmt, 1);
        if (minTime != 0 && maxTime != 0) {
            result = std::make_pair(minTime, maxTime);
        }
    } else if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Error in getTimeRange query: {}", sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    return result;
}

std::optional<BoundingBox> SQLiteStorage::getSpatialExtent(
    const std::optional<std::string>& variableName,
    const std::optional<CRSInfo>& /*targetCrs*/) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return std::nullopt;
    }
    
    std::string sql;
    sqlite3_stmt* stmt = nullptr;
    
    if (variableName) {
        sql = "SELECT MIN(f.bbox_min_x), MIN(f.bbox_min_y), MAX(f.bbox_max_x), MAX(f.bbox_max_y) "
              "FROM files f JOIN file_variables fv ON f.file_id = fv.file_id "
              "WHERE fv.variable_name = ?;";
        
        int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            STORAGE_LOG_ERROR("Failed to prepare getSpatialExtent query: {}", sqlite3_errmsg(db_));
            return std::nullopt;
        }
        
        sqlite3_bind_text(stmt, 1, variableName->c_str(), -1, SQLITE_STATIC);
    } else {
        sql = "SELECT MIN(bbox_min_x), MIN(bbox_min_y), MAX(bbox_max_x), MAX(bbox_max_y) FROM files;";
        
        int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            STORAGE_LOG_ERROR("Failed to prepare getSpatialExtent query: {}", sqlite3_errmsg(db_));
            return std::nullopt;
        }
    }
    
    int rc = sqlite3_step(stmt);
    std::optional<BoundingBox> result;
    
    if (rc == SQLITE_ROW) {
        BoundingBox bbox;
        bbox.minX = getDoubleFromColumn(stmt, 0, std::numeric_limits<double>::max());
        bbox.minY = getDoubleFromColumn(stmt, 1, std::numeric_limits<double>::max());
        bbox.maxX = getDoubleFromColumn(stmt, 2, std::numeric_limits<double>::lowest());
        bbox.maxY = getDoubleFromColumn(stmt, 3, std::numeric_limits<double>::lowest());
        
        // 检查是否有有效的边界框
        if (bbox.minX != std::numeric_limits<double>::max() && 
            bbox.minY != std::numeric_limits<double>::max() &&
            bbox.maxX != std::numeric_limits<double>::lowest() && 
            bbox.maxY != std::numeric_limits<double>::lowest()) {
            result = bbox;
        }
    } else if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Error in getSpatialExtent query: {}", sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    return result;
}

bool SQLiteStorage::replaceIndexData(const std::vector<FileMetadata>& metadataList) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return false;
    }
    
    return executeInTransaction([this, &metadataList]() -> bool {
        // 清空现有数据
        char* errMsg = nullptr;
        int rc = sqlite3_exec(db_, "DELETE FROM file_variables; DELETE FROM files;", nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            STORAGE_LOG_ERROR("Failed to clear existing data: {}", errMsg ? errMsg : "Unknown error");
            sqlite3_free(errMsg);
            return false;
        }
        
        // 重新插入所有数据
        for (const auto& metadata : metadataList) {
            // 临时解锁以调用addOrUpdateFileMetadata（它有自己的锁）
            mutex_.unlock();
            bool success = addOrUpdateFileMetadata(metadata);
            mutex_.lock();
            
            if (!success) {
                STORAGE_LOG_ERROR("Failed to insert metadata for file: {}", metadata.fileId);
                return false;
            }
        }
        
        return true;
    });
}

// 实现IDatasetMetadataRegistryBackend接口方法
bool SQLiteStorage::addOrUpdateDataset(const oscean::core_services::metadata::DatasetMetadataEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    datasets_[entry.id] = entry;
    return true;
}

std::optional<oscean::core_services::metadata::DatasetMetadataEntry> SQLiteStorage::getDataset(const std::string& datasetId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = datasets_.find(datasetId);
    if (it != datasets_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool SQLiteStorage::removeDataset(const std::string& datasetId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = datasets_.find(datasetId);
    if (it != datasets_.end()) {
        datasets_.erase(it);
        return true;
    }
    return false;
}

std::vector<oscean::core_services::metadata::DatasetMetadataEntry> SQLiteStorage::findDatasets(const oscean::core_services::metadata::MetadataQueryCriteria& /*criteria*/) const {
    std::vector<oscean::core_services::metadata::DatasetMetadataEntry> results;
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& pair : datasets_) {
        // 简单的实现，实际应该根据criteria进行过滤
        results.push_back(pair.second);
    }
    
    return results;
}

std::vector<std::string> SQLiteStorage::getAllDatasetIds() const {
    std::vector<std::string> ids;
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& pair : datasets_) {
        ids.push_back(pair.first);
    }
    
    return ids;
}

size_t SQLiteStorage::getDatasetCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return datasets_.size();
}

bool SQLiteStorage::clearAllDatasets() {
    std::lock_guard<std::mutex> lock(mutex_);
    datasets_.clear();
    return true;
}

bool SQLiteStorage::replaceAllDatasets(const std::vector<oscean::core_services::metadata::DatasetMetadataEntry>& entries) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    datasets_.clear();
    for (const auto& entry : entries) {
        datasets_[entry.id] = entry;
    }
    
    return true;
}

// 添加缺失的辅助方法实现
bool SQLiteStorage::addOrUpdateDatasetInMemory(const oscean::core_services::metadata::DatasetMetadataEntry& entry) {
    datasets_[entry.id] = entry;
    return true;
}

bool SQLiteStorage::removeDatasetInMemory(const std::string& datasetId) {
    auto it = datasets_.find(datasetId);
    if (it != datasets_.end()) {
        datasets_.erase(it);
        return true;
    }
    return false;
}

std::vector<oscean::core_services::metadata::DatasetMetadataEntry> SQLiteStorage::findDatasetsInMemory(const oscean::core_services::metadata::MetadataQueryCriteria& /*criteria*/) const {
    std::vector<oscean::core_services::metadata::DatasetMetadataEntry> results;
    
    for (const auto& pair : datasets_) {
        results.push_back(pair.second);
    }
    
    return results;
}

std::string SQLiteStorage::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// 添加getVariablesForFile辅助方法
std::vector<VariableMeta> SQLiteStorage::getVariablesForFile(const std::string& fileId) const {
    std::vector<VariableMeta> variables;
    
    if (!isInitialized_ || !db_) {
        STORAGE_LOG_ERROR("Storage not initialized");
        return variables;
    }
    
    const char* sql = "SELECT variable_name FROM file_variables WHERE file_id = ? ORDER BY variable_name;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        STORAGE_LOG_ERROR("Failed to prepare getVariablesForFile query: {}", sqlite3_errmsg(db_));
        return variables;
    }
    
    sqlite3_bind_text(stmt, 1, fileId.c_str(), -1, SQLITE_STATIC);
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        VariableMeta variable;
        variable.name = getStringFromColumn(stmt, 0);
        // 可以从其他表或字段获取更多变量信息
        variables.push_back(variable);
    }
    
    if (rc != SQLITE_DONE) {
        STORAGE_LOG_ERROR("Error in getVariablesForFile query: {}", sqlite3_errmsg(db_));
    }
    
    sqlite3_finalize(stmt);
    return variables;
}


// =============================================================================
// 🆕 CRS信息处理helper函数实现
// =============================================================================

CRSInfo SQLiteStorage::getCRSInfoForFile(const std::string& fileId) const {
    CRSInfo crsInfo;
    
    if (!statements_.getCrsInfo) {
        STORAGE_LOG_WARN("getCrsInfo statement not prepared");
        return crsInfo;
    }
    
    // 绑定参数
    sqlite3_bind_text(statements_.getCrsInfo, 1, fileId.c_str(), -1, SQLITE_STATIC);
    
    // 执行查询
    int rc = sqlite3_step(statements_.getCrsInfo);
    if (rc == SQLITE_ROW) {
        // 读取所有CRS字段
        crsInfo.wktext = getStringFromColumn(statements_.getCrsInfo, 0);
        crsInfo.projString = getStringFromColumn(statements_.getCrsInfo, 1);
        
        int epsgCode = sqlite3_column_int(statements_.getCrsInfo, 2);
        if (epsgCode != 0) {
            crsInfo.epsgCode = epsgCode;
        }
        
        crsInfo.authorityName = getStringFromColumn(statements_.getCrsInfo, 3);
        crsInfo.authorityCode = getStringFromColumn(statements_.getCrsInfo, 4);
        crsInfo.isGeographic = sqlite3_column_int(statements_.getCrsInfo, 5) != 0;
        crsInfo.isProjected = sqlite3_column_int(statements_.getCrsInfo, 6) != 0;
        crsInfo.linearUnitName = getStringFromColumn(statements_.getCrsInfo, 7);
        crsInfo.linearUnitToMeter = getDoubleFromColumn(statements_.getCrsInfo, 8, 1.0);
        crsInfo.angularUnitName = getStringFromColumn(statements_.getCrsInfo, 9);
        crsInfo.angularUnitToRadian = getDoubleFromColumn(statements_.getCrsInfo, 10, 1.0);
        
        // 兼容字段
        crsInfo.authority = getStringFromColumn(statements_.getCrsInfo, 11);
        crsInfo.code = getStringFromColumn(statements_.getCrsInfo, 12);
        crsInfo.id = getStringFromColumn(statements_.getCrsInfo, 13);
        crsInfo.wkt = getStringFromColumn(statements_.getCrsInfo, 14);
        crsInfo.proj4text = getStringFromColumn(statements_.getCrsInfo, 15);
        
        // 🆕 增强：从JSON中反序列化CFProjectionParameters和扩展信息
        std::string crsJson = getStringFromColumn(statements_.getCrsInfo, 16);
        if (!crsJson.empty()) {
            try {
                json j = json::parse(crsJson);
                
                // 反序列化CFProjectionParameters
                if (j.contains("cfParameters") && j["cfParameters"].is_object()) {
                    CFProjectionParameters cfParams;
                    auto cfParamsJson = j["cfParameters"];
                    
                    if (cfParamsJson.contains("gridMappingName") && cfParamsJson["gridMappingName"].is_string()) {
                        cfParams.gridMappingName = cfParamsJson["gridMappingName"];
                    }
                    
                    // 反序列化数值参数
                    if (cfParamsJson.contains("numericParameters") && cfParamsJson["numericParameters"].is_object()) {
                        for (auto& [key, value] : cfParamsJson["numericParameters"].items()) {
                            if (value.is_number()) {
                                cfParams.numericParameters[key] = value.get<double>();
                            }
                        }
                    }
                    
                    // 反序列化字符串参数
                    if (cfParamsJson.contains("stringParameters") && cfParamsJson["stringParameters"].is_object()) {
                        for (auto& [key, value] : cfParamsJson["stringParameters"].items()) {
                            if (value.is_string()) {
                                cfParams.stringParameters[key] = value.get<std::string>();
                            }
                        }
                    }
                    
                    crsInfo.cfParameters = cfParams;
                    
                    STORAGE_LOG_DEBUG("加载CF投影参数: gridMapping={}, 数值参数{}个, 字符串参数{}个", 
                                     cfParams.gridMappingName,
                                     cfParams.numericParameters.size(),
                                     cfParams.stringParameters.size());
                }
                
                // 反序列化扩展参数
                if (j.contains("parameters") && j["parameters"].is_object()) {
                    for (auto& [key, value] : j["parameters"].items()) {
                        if (value.is_string()) {
                            crsInfo.parameters[key] = value.get<std::string>();
                        }
                    }
                }
                
                // 验证数据完整性
                if (j.contains("dataIntegrity") && j["dataIntegrity"].is_object()) {
                    auto integrity = j["dataIntegrity"];
                    bool hasCleanedProjection = integrity.value("cleanedProjection", false);
                    bool hasCFParams = integrity.value("hasCFParams", false);
                    
                    STORAGE_LOG_DEBUG("CRS数据完整性: 清理投影={}, CF参数={}", 
                                     hasCleanedProjection, hasCFParams);
                }
                
            } catch (const json::exception& e) {
                STORAGE_LOG_WARN("解析CRS JSON失败 for file {}: {}", fileId, e.what());
                // 继续处理，只是不加载扩展信息
            }
        }
        
        STORAGE_LOG_DEBUG("加载CRS信息: file_id={}, authority={}:{}, 类型={}", 
                         fileId, crsInfo.authorityName, crsInfo.authorityCode,
                         crsInfo.isGeographic ? "地理坐标系" : (crsInfo.isProjected ? "投影坐标系" : "未知"));
    }
    
    // 重置语句
    sqlite3_reset(statements_.getCrsInfo);
    sqlite3_clear_bindings(statements_.getCrsInfo);
    
    return crsInfo;
}

bool SQLiteStorage::saveCRSInfoForFile(const std::string& fileId, const CRSInfo& crsInfo) {
    if (!statements_.insertCrsInfo) {
        STORAGE_LOG_ERROR("insertCrsInfo statement not prepared");
        return false;
    }
    
    // 创建完整的CRS JSON，包含CFProjectionParameters
    json crsJson;
    crsJson["wktext"] = crsInfo.wktext;
    crsJson["projString"] = crsInfo.projString;
    if (crsInfo.epsgCode.has_value()) {
        crsJson["epsgCode"] = crsInfo.epsgCode.value();
    }
    crsJson["authorityName"] = crsInfo.authorityName;
    crsJson["authorityCode"] = crsInfo.authorityCode;
    crsJson["isGeographic"] = crsInfo.isGeographic;
    crsJson["isProjected"] = crsInfo.isProjected;
    crsJson["linearUnitName"] = crsInfo.linearUnitName;
    crsJson["linearUnitToMeter"] = crsInfo.linearUnitToMeter;
    crsJson["angularUnitName"] = crsInfo.angularUnitName;
    crsJson["angularUnitToRadian"] = crsInfo.angularUnitToRadian;
    
    // 🆕 增强：添加CFProjectionParameters完整序列化
    if (crsInfo.cfParameters.has_value()) {
        json cfParamsJson;
        cfParamsJson["gridMappingName"] = crsInfo.cfParameters->gridMappingName;
        
        // 序列化数值参数
        json numericParamsJson = json::object();
        for (const auto& [key, value] : crsInfo.cfParameters->numericParameters) {
            numericParamsJson[key] = value;
        }
        cfParamsJson["numericParameters"] = numericParamsJson;
        
        // 序列化字符串参数
        json stringParamsJson = json::object();
        for (const auto& [key, value] : crsInfo.cfParameters->stringParameters) {
            stringParamsJson[key] = value;
        }
        cfParamsJson["stringParameters"] = stringParamsJson;
        
        crsJson["cfParameters"] = cfParamsJson;
        
        STORAGE_LOG_DEBUG("存储CF投影参数: gridMapping={}, 数值参数{}个, 字符串参数{}个", 
                         crsInfo.cfParameters->gridMappingName,
                         crsInfo.cfParameters->numericParameters.size(),
                         crsInfo.cfParameters->stringParameters.size());
    }
    
    // 存储扩展参数
    if (!crsInfo.parameters.empty()) {
        json parametersJson = json::object();
        for (const auto& [key, value] : crsInfo.parameters) {
            parametersJson[key] = value;
        }
        crsJson["parameters"] = parametersJson;
    }
    
    // 添加完整性验证信息
    crsJson["dataIntegrity"] = {
        {"hasWKT", !crsInfo.wktext.empty()},
        {"hasPROJ", !crsInfo.projString.empty()},
        {"hasEPSG", crsInfo.epsgCode.has_value()},
        {"hasCFParams", crsInfo.cfParameters.has_value()},
        {"cleanedProjection", crsInfo.projString.find("EPSG:") != std::string::npos || 
                              crsInfo.projString.find("+proj=") != std::string::npos}
    };
    
    std::string jsonStr = crsJson.dump();
    
    // 绑定参数 - 按SQL语句的参数顺序
    int paramIndex = 1;
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, fileId.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.wktext.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.projString.c_str(), -1, SQLITE_STATIC);
    
    if (crsInfo.epsgCode.has_value()) {
        sqlite3_bind_int(statements_.insertCrsInfo, paramIndex++, crsInfo.epsgCode.value());
    } else {
        sqlite3_bind_null(statements_.insertCrsInfo, paramIndex++);
    }
    
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.authorityName.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.authorityCode.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(statements_.insertCrsInfo, paramIndex++, crsInfo.isGeographic ? 1 : 0);
    sqlite3_bind_int(statements_.insertCrsInfo, paramIndex++, crsInfo.isProjected ? 1 : 0);
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.linearUnitName.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_double(statements_.insertCrsInfo, paramIndex++, crsInfo.linearUnitToMeter);
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.angularUnitName.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_double(statements_.insertCrsInfo, paramIndex++, crsInfo.angularUnitToRadian);
    
    // 兼容字段
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.authorityName.c_str(), -1, SQLITE_STATIC); // authority
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.authorityCode.c_str(), -1, SQLITE_STATIC); // code
    
    // 构造ID字段
    std::string id = crsInfo.authorityName.empty() || crsInfo.authorityCode.empty() ? 
                     "" : crsInfo.authorityName + ":" + crsInfo.authorityCode;
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, id.c_str(), -1, SQLITE_STATIC);
    
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.wktext.c_str(), -1, SQLITE_STATIC); // wkt
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, crsInfo.projString.c_str(), -1, SQLITE_STATIC); // proj4text
    sqlite3_bind_text(statements_.insertCrsInfo, paramIndex++, jsonStr.c_str(), -1, SQLITE_STATIC); // crs_json
    
    // 执行语句
    int rc = sqlite3_step(statements_.insertCrsInfo);
    bool success = (rc == SQLITE_DONE);
    
    if (!success) {
        STORAGE_LOG_ERROR("Failed to insert CRS info for file {}: {}", fileId, sqlite3_errmsg(db_));
    } else {
        STORAGE_LOG_DEBUG("成功存储CRS信息: file_id={}, authority={}:{}, 类型={}", 
                         fileId, crsInfo.authorityName, crsInfo.authorityCode,
                         crsInfo.isGeographic ? "地理坐标系" : (crsInfo.isProjected ? "投影坐标系" : "未知"));
    }
    
    // 重置语句
    sqlite3_reset(statements_.insertCrsInfo);
    sqlite3_clear_bindings(statements_.insertCrsInfo);
    
    return success;
}

bool SQLiteStorage::deleteCRSInfoForFile(const std::string& fileId) {
    if (!statements_.deleteCrsInfo) {
        STORAGE_LOG_ERROR("deleteCrsInfo statement not prepared");
        return false;
    }
    
    sqlite3_bind_text(statements_.deleteCrsInfo, 1, fileId.c_str(), -1, SQLITE_STATIC);
    
    int rc = sqlite3_step(statements_.deleteCrsInfo);
    bool success = (rc == SQLITE_DONE);
    
    sqlite3_reset(statements_.deleteCrsInfo);
    sqlite3_clear_bindings(statements_.deleteCrsInfo);
    
    return success;
}

} // namespace storage
} // namespace metadata  
} // namespace core_services
} // namespace oscean
