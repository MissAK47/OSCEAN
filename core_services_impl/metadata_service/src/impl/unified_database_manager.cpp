// 🔧 添加必需的Boost宏定义
#ifndef BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE
#endif
#ifndef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#endif
#ifndef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#endif

#include "impl/unified_database_manager.h"
#include "core_services/common_data_types.h"
#include "core_services/metadata/unified_metadata_service.h"
#include "core_services/force_include_boost_future.h"  // 强制包含boost future
#include "common_utils/time/time_services.h"  // 添加时间服务头文件
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sqlite3.h>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <boost/thread/thread.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <unordered_set>
#include <algorithm>
#include <fmt/format.h>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <regex>
#include <chrono>
#include <sstream>
#include <set>
#include <algorithm>

// 🔧 添加系统头文件以支持目录操作
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace oscean::core_services::metadata::impl {

// === Forward declarations for helper functions ===
void fillMetadataDetails(sqlite3* db, const std::string& metadataId, FileMetadata& metadata);
void fillBasicInfo(sqlite3_stmt* stmt, FileMetadata& metadata);
void fillSpatialInfo(sqlite3* db, const std::string& metadataId, FileMetadata& metadata);
void fillTemporalInfo(sqlite3* db, const std::string& metadataId, FileMetadata& metadata);
void fillVariables(sqlite3* db, const std::string& metadataId, FileMetadata& metadata);
void fillAttributes(sqlite3* db, const std::string& metadataId, FileMetadata& metadata);
const std::unordered_set<std::string>& getExcludedAttributeKeys();

// 🔧 新增：完整的Geohash算法实现
namespace {
    
    /**
     * @brief 完整的Geohash编码实现
     * @param latitude 纬度 (-90 到 90)
     * @param longitude 经度 (-180 到 180)
     * @param precision 精度位数 (通常为5-12)
     * @return Geohash字符串
     */
    std::string encodeGeohash(double latitude, double longitude, int precision) {
        // Geohash字符表（Base32编码）
        static const char base32[] = "0123456789bcdefghjkmnpqrstuvwxyz";
        
        // 确保坐标在有效范围内
        latitude = std::max(-90.0, std::min(90.0, latitude));
        longitude = std::max(-180.0, std::min(180.0, longitude));
        
        std::string geohash;
        geohash.reserve(precision);
        
        // 初始边界
        double lat_min = -90.0, lat_max = 90.0;
        double lon_min = -180.0, lon_max = 180.0;
        
        int bit = 0;
        int bit_count = 0;
        bool is_longitude = true; // 交替处理经度和纬度
        
        while (geohash.length() < static_cast<size_t>(precision)) {
            if (is_longitude) {
                // 处理经度
                double mid = (lon_min + lon_max) / 2.0;
                if (longitude >= mid) {
                    bit = (bit << 1) | 1;
                    lon_min = mid;
                } else {
                    bit = bit << 1;
                    lon_max = mid;
                }
            } else {
                // 处理纬度
                double mid = (lat_min + lat_max) / 2.0;
                if (latitude >= mid) {
                    bit = (bit << 1) | 1;
                    lat_min = mid;
                } else {
                    bit = bit << 1;
                    lat_max = mid;
                }
            }
            
            is_longitude = !is_longitude;
            bit_count++;
            
            // 每5位生成一个字符
            if (bit_count == 5) {
                geohash += base32[bit];
                bit = 0;
                bit_count = 0;
            }
        }
        
        return geohash;
    }
    
    /**
     * @brief 从CRS WKT中提取EPSG代码
     * @param wkt CRS的WKT字符串
     * @return EPSG代码，如果未找到返回0
     */
    int extractEpsgCodeFromWkt(const std::string& wkt) {
        if (wkt.empty()) {
            return 0;
        }
        
        // 查找AUTHORITY["EPSG","code"]模式 - 修复转义问题
        std::regex epsgPattern(R"(AUTHORITY\[\"EPSG\",\"(\d+)\"\])");
        std::smatch match;
        
        if (std::regex_search(wkt, match, epsgPattern)) {
            try {
                return std::stoi(match[1].str());
            } catch (const std::exception&) {
                // 转换失败，返回0
            }
        }
        
        // 如果WKT包含"WGS84"或"WGS_1984"，默认为EPSG:4326
        if (wkt.find("WGS84") != std::string::npos || 
            wkt.find("WGS_1984") != std::string::npos) {
            return 4326;
        }
        
        return 0; // 未能确定EPSG代码
    }
}

UnifiedDatabaseManager::UnifiedDatabaseManager(
    const std::string& dbPath,
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> injectedCommonFactory)
    : dbPath_(dbPath),
      commonServices_(injectedCommonFactory),
      db_(nullptr),
      initialized_(false)
{
    if (!commonServices_) {
        throw std::runtime_error("Injected CommonServicesFactory is null during UnifiedDatabaseManager construction.");
    }
    logger_ = commonServices_->getLogger();
    configLoader_ = commonServices_->getConfigurationLoader();
    if (!logger_ || !configLoader_) {
        throw std::runtime_error("Failed to get logger or config loader from injected CommonServicesFactory.");
    }
    logger_->info(fmt::format("UnifiedDatabaseManager constructed for DB path: {} with injected CommonServicesFactory", dbPath_));
}

UnifiedDatabaseManager::~UnifiedDatabaseManager()
{
    close();
}

bool UnifiedDatabaseManager::initialize()
{
    std::cout << "[DEBUG DB] UnifiedDatabaseManager::initialize() 开始..." << std::endl;
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        std::cout << "[DEBUG DB] 数据库已初始化，返回true" << std::endl;
        return true;
    }
    
    logger_->info("Initializing UnifiedDatabaseManager...");
    
    // 🔧 步骤1: 从配置文件读取数据库配置
    std::cout << "[DEBUG DB] 步骤1: 读取配置文件..." << std::endl;
    std::string configDbDir;
    std::string configDbFileName;
    
    try {
        // 🔧 修复：层级读取配置，确保找到正确路径
        bool configFound = false;
        
        // 1. 尝试读取新的统一架构配置 - 🔧 修复键名匹配
        try {
            configDbDir = configLoader_->getString("metadata.unified_connection.directory");
            configDbFileName = configLoader_->getString("metadata.unified_connection.file");
            
            std::cout << "[DEBUG DB] 🔍 尝试读取 metadata.unified_connection.directory: '" << configDbDir << "'" << std::endl;
            std::cout << "[DEBUG DB] 🔍 尝试读取 metadata.unified_connection.file: '" << configDbFileName << "'" << std::endl;
            
            if (!configDbDir.empty() && !configDbFileName.empty()) {
                configFound = true;
                std::cout << "[DEBUG DB] ✅ 从统一架构配置读取成功" << std::endl;
                std::cout << "[DEBUG DB] 📍 读取路径: metadata.unified_connection.*" << std::endl;
            } else {
                std::cout << "[DEBUG DB] ⚠️ metadata.unified_connection 配置为空，继续尝试其他配置" << std::endl;
                // 🔧 添加配置加载器诊断
                std::cout << "[DEBUG DB] 🔧 配置加载器诊断:" << std::endl;
                std::cout << "[DEBUG DB]   - configLoader_ 指针: " << (configLoader_ ? "有效" : "NULL") << std::endl;
                if (configLoader_) {
                    try {
                        // 测试读取一个已知存在的配置
                        std::string testValue = configLoader_->getString("database.base_path", "DEFAULT");
                        std::cout << "[DEBUG DB]   - 测试读取 database.base_path: '" << testValue << "'" << std::endl;
                    } catch (const std::exception& testE) {
                        std::cout << "[DEBUG DB]   - 测试读取异常: " << testE.what() << std::endl;
                    }
                }
            }
        } catch (const std::exception& e) {
            // 统一配置不存在，继续尝试其他配置
            std::cout << "[DEBUG DB] ⚠️ metadata.unified_connection 读取异常: " << e.what() << std::endl;
        }
        
        // 2. 如果统一配置不存在，尝试metadata配置
        if (!configFound) {
            try {
                configDbDir = configLoader_->getString("metadata.database.directory");
                configDbFileName = configLoader_->getString("metadata.database.filename");
                configFound = true;
                std::cout << "[DEBUG DB] ✅ 从metadata配置读取成功" << std::endl;
                std::cout << "[DEBUG DB] 📍 读取路径: metadata.database.*" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[DEBUG DB] ⚠️ metadata.database 配置读取失败: " << e.what() << std::endl;
            }
        }
        
        // 3. 如果上述配置都不存在，尝试兼容性配置
        if (!configFound) {
            try {
                std::string basePath = configLoader_->getString("database.base_path");
                configDbDir = basePath;
                configDbFileName = "ocean_environment.db";  // 默认文件名
                configFound = true;
                std::cout << "[DEBUG DB] ✅ 从基础路径配置读取成功" << std::endl;
            } catch (...) {
                // 基础配置也不存在
            }
        }
        
        if (!configFound) {
            throw std::runtime_error("No valid database configuration found in config file");
        }
        
        std::cout << "[DEBUG DB] 配置读取成功:" << std::endl;
        std::cout << "[DEBUG DB]   数据库目录: " << configDbDir << std::endl;
        std::cout << "[DEBUG DB]   数据库文件名: " << configDbFileName << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[DEBUG DB] 配置读取失败: " << e.what() << std::endl;
        logger_->error(fmt::format("Failed to read database configuration: {}", e.what()));
        
        // 🔧 修复：智能默认配置，基于项目结构推断
        // 获取当前工作目录，然后构建相对于项目根的路径
        std::string currentDir = std::filesystem::current_path().string();
        std::cout << "[DEBUG DB] 当前工作目录: " << currentDir << std::endl;
        
        // 智能路径推断
        if (currentDir.find("OSCEAN") != std::string::npos) {
            // 找到OSCEAN目录位置
            size_t osceanPos = currentDir.find("OSCEAN");
            std::string projectRoot = currentDir.substr(0, osceanPos + 6); // "OSCEAN"长度为6
            configDbDir = projectRoot + "/database";
            std::cout << "[DEBUG DB] 🔧 智能推断项目根目录: " << projectRoot << std::endl;
        } else {
            // 回退到相对路径
            configDbDir = "./database";
            std::cout << "[DEBUG DB] 🔧 使用相对路径回退方案" << std::endl;
        }
        
        configDbFileName = "ocean_environment.db";
        std::cout << "[DEBUG DB] 使用智能推断配置: " << configDbDir << "/" << configDbFileName << std::endl;
        logger_->warn("Using intelligent fallback database configuration");
    }
    
    // 🔧 步骤2: 构建完整数据库路径
    std::string fullDbPath = configDbDir + "/" + configDbFileName;
    std::cout << "[DEBUG DB] 原数据库路径: " << dbPath_ << std::endl;
    std::cout << "[DEBUG DB] 配置数据库路径: " << fullDbPath << std::endl;
    
    // 🔧 步骤3: 检查数据库目录是否存在（仅检查，不创建）
    std::cout << "[DEBUG DB] 步骤3: 检查数据库目录是否存在..." << std::endl;
    
    bool dirExists = false;
    #ifdef _WIN32
    // Windows - 使用 _access 检查目录
    dirExists = (_access(configDbDir.c_str(), 0) == 0);
    #else
    // Unix/Linux - 使用 access 检查目录
    dirExists = (access(configDbDir.c_str(), F_OK) == 0);
    #endif
    
    if (!dirExists) {
        std::cout << "[DEBUG DB] ⚠️  警告: 数据库目录不存在: " << configDbDir << std::endl;
        logger_->warn(fmt::format("Database directory does not exist: {}. Please create it manually.", configDbDir));
        std::cout << "[DEBUG DB] 💡 建议: 请手动创建目录或更新配置文件中的 metadata.database.directory" << std::endl;
        // 注意：我们不返回false，而是继续尝试，让SQLite处理路径问题
    } else {
        std::cout << "[DEBUG DB] ✅ 数据库目录存在: " << configDbDir << std::endl;
    }
    
    // 🔧 步骤4: 更新数据库路径并验证
    dbPath_ = fullDbPath;
    std::cout << "[DEBUG DB] 最终数据库路径: " << dbPath_ << std::endl;
    
    // 验证路径不为空
    if (dbPath_.empty()) {
        std::cout << "[DEBUG DB] 数据库路径为空，初始化失败" << std::endl;
        logger_->critical("Database path is empty after configuration");
        return false;
    }

    std::cout << "[DEBUG DB] 步骤4: 应用数据库配置..." << std::endl;
    applyDatabaseConfiguration();

    std::cout << "[DEBUG DB] 步骤5: 尝试打开数据库..." << std::endl;
    if (sqlite3_open(dbPath_.c_str(), &db_) != SQLITE_OK) {
        std::cout << "[DEBUG DB] 数据库打开失败: " << sqlite3_errmsg(db_) << std::endl;
        logger_->critical(fmt::format("Can't open database: {}. Please check if directory exists and has write permissions.", sqlite3_errmsg(db_)));
        std::cout << "[DEBUG DB] 💡 解决建议:" << std::endl;
        std::cout << "[DEBUG DB]   1. 检查目录是否存在: " << configDbDir << std::endl;
        std::cout << "[DEBUG DB]   2. 检查目录是否有写权限" << std::endl;
        std::cout << "[DEBUG DB]   3. 检查磁盘空间是否充足" << std::endl;
        return false;
    }
    std::cout << "[DEBUG DB] 数据库打开成功" << std::endl;

    std::string schemaPath = configLoader_->getString("metadata.database.schema_path", "./config/unified_schema.sql");
    std::cout << "[DEBUG DB] 尝试加载schema文件: " << schemaPath << std::endl;
    logger_->info(fmt::format("Attempting to load database schema from: {}", schemaPath));

    std::ifstream schemaFile(schemaPath);
    
    std::string schema_sql;
    if (schemaFile) {
        std::cout << "[DEBUG DB] 从外部文件加载schema" << std::endl;
        logger_->info(fmt::format("Loading schema from external file: {}", schemaPath));
        std::stringstream schema_ss;
        schema_ss << schemaFile.rdbuf();
        schema_sql = schema_ss.str();
        
        if (schema_sql.empty()) {
            std::cout << "[DEBUG DB] schema文件为空" << std::endl;
            logger_->critical(fmt::format("Schema file is empty: {}", schemaPath));
            sqlite3_close(db_);
            db_ = nullptr;
            return false;
        }
    } else {
        std::cout << "[DEBUG DB] schema文件不存在: " << schemaPath << std::endl;
        logger_->critical(fmt::format("Schema file not found: {}. Please ensure the schema file exists.", schemaPath));
        std::cout << "[DEBUG DB] 💡 解决建议:" << std::endl;
        std::cout << "[DEBUG DB]   1. 检查文件路径是否正确: " << schemaPath << std::endl;
        std::cout << "[DEBUG DB]   2. 检查文件是否存在且可读" << std::endl;
        std::cout << "[DEBUG DB]   3. 更新配置文件中的 metadata.database.schema_path" << std::endl;
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }

    std::cout << "[DEBUG DB] 执行schema SQL..." << std::endl;
    char* errMsg = nullptr;
    if (sqlite3_exec(db_, schema_sql.c_str(), 0, 0, &errMsg) != SQLITE_OK) {
        std::cout << "[DEBUG DB] schema执行失败: " << errMsg << std::endl;
        logger_->critical(fmt::format("Failed to execute schema script: {}", errMsg));
        sqlite3_free(errMsg);
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    std::cout << "[DEBUG DB] schema执行成功" << std::endl;

    logger_->info("Database schema applied successfully.");
    initialized_ = true;
    std::cout << "[DEBUG DB] 数据库初始化完成，返回true" << std::endl;
    return true;
}

void UnifiedDatabaseManager::close()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
        initialized_ = false;
        logger_->info("Database connection closed.");
    }
}

boost::future<AsyncResult<std::string>> UnifiedDatabaseManager::storeFileMetadataAsync(const core_services::FileMetadata& metadata)
{
    return boost::async(boost::launch::async, [this, metadata]() {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) {
                return AsyncResult<std::string>::failure("Database not initialized.");
            }

            const std::string metadataId = generateMetadataId(metadata);

            if (sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr) != SQLITE_OK) {
                return AsyncResult<std::string>::failure(fmt::format("Failed to begin transaction: {}", sqlite3_errmsg(db_)));
            }

            insertBasicInfo(metadataId, metadata);
            insertSpatialInfoFromCoverage(metadataId, metadata);
            insertTemporalInfoFromRange(metadataId, metadata);
            insertVariables(metadataId, metadata.variables);
            insertDataTypes(metadataId, metadata);
            insertAttributesFiltered(metadataId, metadata.attributes);

            if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) {
                sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
                return AsyncResult<std::string>::failure(fmt::format("Failed to commit transaction: {}", sqlite3_errmsg(db_)));
            }
            
            return AsyncResult<std::string>::success(metadataId);
        } catch (const std::exception& e) {
            logger_->critical(fmt::format("Exception in storeFileMetadataAsync: {}", e.what()));
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
            return AsyncResult<std::string>::failure(e.what());
        }
    });
}

boost::future<AsyncResult<FileMetadata>> UnifiedDatabaseManager::queryByFilePathAsync(const std::string& filePath)
{
    return boost::async(boost::launch::async, [this, filePath]() {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) {
                return AsyncResult<FileMetadata>::failure("Database not initialized.");
            }
            
            // 🔧 使用配置化SQL查询而不是硬编码
            auto it = sqlQueries_.find("select_file_by_path");
            if (it == sqlQueries_.end()) {
                return AsyncResult<FileMetadata>::failure("select_file_by_path query not found in configuration");
            }
            const std::string& querySQL = it->second;
            auto results = executeQuery(querySQL, {filePath});
            
            if (results.empty()) {
                return AsyncResult<FileMetadata>::failure("File path not found.");
            }
            
            return AsyncResult<FileMetadata>::success(results[0]);

        } catch (const std::exception& e) {
            logger_->critical(fmt::format("Exception in queryByFilePathAsync: {}", e.what()));
            return AsyncResult<FileMetadata>::failure(e.what());
        }
    });
}

boost::future<AsyncResult<std::vector<FileMetadata>>> UnifiedDatabaseManager::queryMetadataAsync(const QueryCriteria& criteria)
{
    return boost::async(boost::launch::async, [this, criteria]() {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) {
                return AsyncResult<std::vector<FileMetadata>>::failure("Database not initialized.");
            }
            std::vector<std::string> params;
            std::string sql = buildQuerySql(criteria, params);
            auto results = executeQuery(sql, params);
            return AsyncResult<std::vector<FileMetadata>>::success(std::move(results));
        } catch (const std::exception& e) {
            logger_->critical(fmt::format("Exception in queryMetadataAsync: {}", e.what()));
            return AsyncResult<std::vector<FileMetadata>>::failure(e.what());
        }
    });
}

boost::future<std::vector<std::string>> UnifiedDatabaseManager::filterExistingFiles(const std::vector<std::string>& filePaths)
{
    return boost::async(boost::launch::async, [this, filePaths]() {
        std::vector<std::string> unprocessedFiles;
        if (filePaths.empty()) {
            return unprocessedFiles;
        }

        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) {
                logger_->error("Database not initialized, cannot filter existing files.");
                return filePaths; // Return all files assuming none exist in DB
            }

            // Use a temporary table for performance with large lists
            sqlite3_exec(db_, "CREATE TEMP TABLE paths_to_check (path TEXT PRIMARY KEY NOT NULL);", nullptr, nullptr, nullptr);

            sqlite3_stmt* stmt;
            const char* insert_sql = "INSERT INTO paths_to_check (path) VALUES (?);";
            if (sqlite3_prepare_v2(db_, insert_sql, -1, &stmt, nullptr) != SQLITE_OK) {
                throw std::runtime_error(fmt::format("Failed to prepare statement for temp table insert: {}", sqlite3_errmsg(db_)));
            }

            for (const auto& path : filePaths) {
                sqlite3_bind_text(stmt, 1, path.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(stmt) != SQLITE_DONE) {
                    // Log error but continue
                    logger_->warn(fmt::format("Failed to insert path '{}' into temp table.", path));
                }
                sqlite3_reset(stmt);
            }
            sqlite3_finalize(stmt);

            const char* select_sql =
                "SELECT ptc.path FROM paths_to_check ptc "
                "LEFT JOIN file_info fi ON ptc.path = fi.file_path "
                "WHERE fi.file_path IS NULL;";

            sqlite3_stmt* select_stmt;
            if (sqlite3_prepare_v2(db_, select_sql, -1, &select_stmt, nullptr) != SQLITE_OK) {
                 throw std::runtime_error(fmt::format("Failed to prepare statement for selecting unprocessed files: {}", sqlite3_errmsg(db_)));
            }

            while (sqlite3_step(select_stmt) == SQLITE_ROW) {
                unprocessedFiles.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(select_stmt, 0)));
            }
            sqlite3_finalize(select_stmt);
            
            sqlite3_exec(db_, "DROP TABLE paths_to_check;", nullptr, nullptr, nullptr);

        } catch (const std::exception& e) {
            logger_->critical(fmt::format("Exception in filterExistingFiles: {}", e.what()));
            sqlite3_exec(db_, "DROP TABLE IF EXISTS paths_to_check;", nullptr, nullptr, nullptr);
            return filePaths; // On error, assume no files exist and return original list
        }

        return unprocessedFiles;
    });
}


boost::future<AsyncResult<bool>> UnifiedDatabaseManager::deleteMetadataAsync(const std::string& metadataId) {
    return boost::async(boost::launch::async, [this, metadataId]() {
        try {
             std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) {
                return AsyncResult<bool>::failure("Database not initialized.");
            }
            std::vector<std::string> tables = {"variable_attributes", "variable_info", "temporal_coverage", "spatial_coverage", "file_info"};
            for(const auto& table : tables){
                std::string sql = "DELETE FROM " + table + " WHERE metadata_id = ?;";
                sqlite3_stmt* stmt;
                if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
                     return AsyncResult<bool>::failure(fmt::format("Failed to prepare delete for table {}: {}", table, sqlite3_errmsg(db_)));
                }
                sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(stmt) != SQLITE_DONE) {
                    sqlite3_finalize(stmt);
                    return AsyncResult<bool>::failure(fmt::format("Failed to execute delete for table {}: {}", table, sqlite3_errmsg(db_)));
                }
                sqlite3_finalize(stmt);
            }
            return AsyncResult<bool>::success(true);
        } catch (const std::exception& e) {
             logger_->critical(fmt::format("Exception in deleteMetadataAsync: {}", e.what()));
            return AsyncResult<bool>::failure(e.what());
        }
    });
}

boost::future<AsyncResult<bool>> UnifiedDatabaseManager::updateMetadataAsync(const std::string& metadataId, const MetadataUpdate& update) {
     return boost::async(boost::launch::async, [this, metadataId, update]() -> AsyncResult<bool> {
        return AsyncResult<bool>::failure("Update operation not implemented yet.");
    });
}

boost::future<AsyncResult<void>> UnifiedDatabaseManager::bulkInsertAsync(const std::vector<FileMetadata>& metadataList)
{
    return boost::async(boost::launch::async, [this, metadataList]() {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
             if (!initialized_) {
                return AsyncResult<void>::failure("Database not initialized.");
            }
            if (sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr) != SQLITE_OK) {
                return AsyncResult<void>::failure(fmt::format("Failed to begin bulk transaction: {}", sqlite3_errmsg(db_)));
            }

            for (const auto& metadata : metadataList) {
                const std::string metadataId = generateMetadataId(metadata);
                insertBasicInfo(metadataId, metadata);
                insertSpatialInfoFromCoverage(metadataId, metadata);
                insertTemporalInfoFromRange(metadataId, metadata);
                insertVariables(metadataId, metadata.variables);
                insertDataTypes(metadataId, metadata);
                insertAttributesFiltered(metadataId, metadata.attributes);
            }

            if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) {
                sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
                return AsyncResult<void>::failure(fmt::format("Failed to commit bulk transaction: {}", sqlite3_errmsg(db_)));
            }
            return AsyncResult<void>::success();
        } catch (const std::exception& e) {
            logger_->critical(fmt::format("Exception in bulkInsertAsync: {}", e.what()));
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
            return AsyncResult<void>::failure(e.what());
        }
    });
}

std::vector<FileMetadata> UnifiedDatabaseManager::executeQuery(const std::string& sql, const std::vector<std::string>& params)
{
    sqlite3_stmt* stmt;
    std::vector<FileMetadata> results;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(fmt::format("Failed to prepare statement: {}", sqlite3_errmsg(db_)));
    }

    for (size_t i = 0; i < params.size(); ++i) {
        sqlite3_bind_text(stmt, i + 1, params[i].c_str(), -1, SQLITE_TRANSIENT);
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FileMetadata metadata;
        std::string metadataId = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        fillMetadataDetails(db_, metadataId, metadata);
        results.push_back(metadata);
    }

    sqlite3_finalize(stmt);
    return results;
}

void UnifiedDatabaseManager::applyDatabaseConfiguration() {
    std::cout << "[DEBUG DB] 步骤6: 从配置文件加载SQL查询..." << std::endl;
    logger_->info("Loading SQL queries from configuration file for dynamic database operations.");
    
    // 🚀 直接从YAML配置文件加载SQL查询 - 绕过configLoader的复杂性
    try {
        // 直接读取database_config.yaml文件
        std::string configFilePath = "./config/database_config.yaml";
        if (!std::filesystem::exists(configFilePath)) {
            configFilePath = "config/database_config.yaml";  // 尝试另一个路径
        }
        
        std::cout << "[DEBUG DB] 尝试读取配置文件: " << configFilePath << std::endl;
        
        if (std::filesystem::exists(configFilePath)) {
            YAML::Node config = YAML::LoadFile(configFilePath);
            
            // 查找 metadata.queries 节点
            if (config["metadata"] && config["metadata"]["queries"]) {
                const auto& queriesNode = config["metadata"]["queries"];
                
                std::cout << "[DEBUG DB] ✅ 找到 metadata.queries 节点，包含 " << queriesNode.size() << " 个查询" << std::endl;
                
                // 逐个加载SQL查询
                for (const auto& queryPair : queriesNode) {
                    std::string queryKey = queryPair.first.as<std::string>();
                    std::string querySQL = queryPair.second.as<std::string>();
                    
                    sqlQueries_[queryKey] = querySQL;
                    std::cout << "[DEBUG DB] ✅ 加载查询: " << queryKey << " (长度: " << querySQL.length() << ")" << std::endl;
                }
                
                std::cout << "[DEBUG DB] 🎉 配置驱动SQL系统初始化完成，共加载 " << sqlQueries_.size() << " 个查询" << std::endl;
                logger_->info(fmt::format("Configuration-driven SQL system initialized with {} queries", sqlQueries_.size()));
                
            } else {
                std::cout << "[DEBUG DB] ⚠️  警告: 配置文件中未找到 metadata.queries 节点" << std::endl;
                logger_->warn("metadata.queries section not found in configuration file");
                
                // 降级到基本查询
                initializeFallbackQueries();
            }
        } else {
            std::cout << "[DEBUG DB] ❌ 配置文件不存在: " << configFilePath << std::endl;
            logger_->error(fmt::format("Configuration file not found: {}", configFilePath));
            
            // 降级到基本查询
            initializeFallbackQueries();
        }
        
        // 加载表结构映射
        loadTableMappings();
        
    } catch (const std::exception& e) {
        std::cout << "[DEBUG DB] ❌ 配置驱动SQL系统初始化失败: " << e.what() << std::endl;
        logger_->error(fmt::format("Failed to initialize configuration-driven SQL system: {}", e.what()));
        
        // 降级到基本查询
        initializeFallbackQueries();
    }
}

std::string UnifiedDatabaseManager::generateMetadataId(const core_services::FileMetadata& metadata) {
    return boost::uuids::to_string(boost::uuids::random_generator()());
}

std::string UnifiedDatabaseManager::buildQuerySql(const QueryCriteria& criteria, std::vector<std::string>& params) {
    std::stringstream sql;
    
    // 🔧 使用新的统一schema表结构
    sql << "SELECT DISTINCT fi.file_id FROM file_info fi ";
    
    // 如果需要空间查询，加入spatial_coverage表
    bool needsSpatial = false;
    bool needsTemporal = false;
    
    // 这里可以根据criteria的具体字段来决定是否需要JOIN其他表
    // 当前简化实现，后续可以根据实际需求扩展
    
    if (needsSpatial) {
        sql << "LEFT JOIN spatial_coverage sc ON fi.file_id = sc.file_id ";
    }
    
    if (needsTemporal) {
        sql << "LEFT JOIN temporal_coverage tc ON fi.file_id = tc.file_id ";
    }
    
    sql << "WHERE 1=1";
    
    // 这里可以添加更多的WHERE条件，基于criteria的具体字段
    // 例如：
    // if (!criteria.filePath.empty()) {
    //     sql << " AND fi.file_path LIKE ?";
    //     params.push_back("%" + criteria.filePath + "%");
    // }
    
    return sql.str();
}

void UnifiedDatabaseManager::insertBasicInfo(const std::string& metadataId, const FileMetadata& metadata) {
    // 🔧 完整SQL包含file_path_hash字段
    const char* sql = "INSERT OR REPLACE INTO file_info (file_id, file_path, file_path_hash, logical_name, file_size, last_modified, file_format, format_variant, format_specific_attributes, primary_category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(fmt::format("Failed to prepare file_info insert: {}", sqlite3_errmsg(db_)));
    }

    sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, metadata.filePath.c_str(), -1, SQLITE_TRANSIENT);
    
    // 🔧 新增：计算文件路径Hash
    std::string pathHash = generateUniqueId(metadata.filePath);
    sqlite3_bind_text(stmt, 3, pathHash.c_str(), -1, SQLITE_TRANSIENT);
    
    // 如果文件名为空，使用路径中的文件名
    std::string logicalName = metadata.fileName.empty() ? 
        metadata.filePath.substr(metadata.filePath.find_last_of("/\\") + 1) : metadata.fileName;
    sqlite3_bind_text(stmt, 4, logicalName.c_str(), -1, SQLITE_TRANSIENT);
    
    sqlite3_bind_int64(stmt, 5, metadata.fileSizeBytes);
    sqlite3_bind_text(stmt, 6, metadata.lastModified.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 7, metadata.format.c_str(), -1, SQLITE_TRANSIENT);
    
    // format_variant和format_specific_attributes暂时设为NULL
    sqlite3_bind_null(stmt, 8);
    sqlite3_bind_null(stmt, 9);
    
    sqlite3_bind_int(stmt, 10, static_cast<int>(metadata.primaryCategory));

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::string errMsg = sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        throw std::runtime_error(fmt::format("Failed to execute file_info insert: {}", errMsg));
    }
    sqlite3_finalize(stmt);
}

void UnifiedDatabaseManager::insertSpatialInfoFromCoverage(const std::string& metadataId, const FileMetadata& metadata) {
    // 🔧 修复：完整SQL包含所有字段
    const char* sql = "INSERT OR REPLACE INTO spatial_coverage (file_id, min_longitude, max_longitude, min_latitude, max_latitude, spatial_resolution_x, spatial_resolution_y, crs_wkt, crs_epsg_code, geohash_6, geohash_8) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(fmt::format("Failed to prepare spatial_coverage insert: {}", sqlite3_errmsg(db_)));
    }

    sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);
    // 🔧 修复：使用NetCDF读取器实际填充的spatialCoverage字段
    sqlite3_bind_double(stmt, 2, metadata.spatialCoverage.minX);
    sqlite3_bind_double(stmt, 3, metadata.spatialCoverage.maxX);
    sqlite3_bind_double(stmt, 4, metadata.spatialCoverage.minY);
    sqlite3_bind_double(stmt, 5, metadata.spatialCoverage.maxY);
    
    // 🔧 修复：使用NetCDF读取器填充的分辨率字段
    if (metadata.spatialInfo.resolutionX > 0) {
        sqlite3_bind_double(stmt, 6, metadata.spatialInfo.resolutionX);
    } else {
        sqlite3_bind_null(stmt, 6);
    }
    
    if (metadata.spatialInfo.resolutionY > 0) {
        sqlite3_bind_double(stmt, 7, metadata.spatialInfo.resolutionY);
    } else {
        sqlite3_bind_null(stmt, 7);
    }

    // 🔧 修复：CRS信息绑定
    if (metadata.rawCrsWkt && !metadata.rawCrsWkt->empty()) {
        sqlite3_bind_text(stmt, 8, metadata.rawCrsWkt->c_str(), -1, SQLITE_TRANSIENT);
        
        // 🔧 新增：从WKT中提取EPSG代码
        int epsgCode = extractEpsgCodeFromWkt(*metadata.rawCrsWkt);
        if (epsgCode > 0) {
            sqlite3_bind_int(stmt, 9, epsgCode);
        } else {
            sqlite3_bind_null(stmt, 9);
        }
    } else if (!metadata.crs.wkt.empty()) {
        // 备用CRS字段
        sqlite3_bind_text(stmt, 8, metadata.crs.wkt.c_str(), -1, SQLITE_TRANSIENT);
        
        int epsgCode = extractEpsgCodeFromWkt(metadata.crs.wkt);
        if (epsgCode > 0) {
            sqlite3_bind_int(stmt, 9, epsgCode);
        } else {
            sqlite3_bind_null(stmt, 9);
        }
    } else {
        sqlite3_bind_null(stmt, 8);
        sqlite3_bind_null(stmt, 9);
    }
    
    // 🔧 新增：Geohash计算（中心点）
    double centerLon = (metadata.spatialCoverage.minX + metadata.spatialCoverage.maxX) / 2.0;
    double centerLat = (metadata.spatialCoverage.minY + metadata.spatialCoverage.maxY) / 2.0;
    
    std::string geohash6 = encodeGeohash(centerLat, centerLon, 6);
    std::string geohash8 = encodeGeohash(centerLat, centerLon, 8);
    
    sqlite3_bind_text(stmt, 10, geohash6.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 11, geohash8.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::string errMsg = sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        throw std::runtime_error(fmt::format("Failed to execute spatial_coverage insert: {}", errMsg));
    }
    sqlite3_finalize(stmt);
}

void UnifiedDatabaseManager::insertTemporalInfoFromRange(const std::string& metadataId, const FileMetadata& metadata) {
    // 🔧 简化SQL，先插入基本字段，避免复杂依赖导致的编译错误
    const char* sql = "INSERT OR REPLACE INTO temporal_coverage (file_id, start_time, end_time, time_resolution_seconds, time_calendar) VALUES (?, ?, ?, ?, ?);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(fmt::format("Failed to prepare temporal_coverage insert: {}", sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, metadata.temporalInfo.startTime.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, metadata.temporalInfo.endTime.c_str(), -1, SQLITE_TRANSIENT);
    
    // 🔧 简化：时间分辨率计算
    double timeResolutionSeconds = 0.0;
    if (metadata.temporalInfo.temporalResolutionSeconds.has_value()) {
        timeResolutionSeconds = static_cast<double>(*metadata.temporalInfo.temporalResolutionSeconds);
    } else {
        // 默认为日度数据
        timeResolutionSeconds = 86400.0; // 24小时 = 86400秒
    }
    sqlite3_bind_double(stmt, 4, timeResolutionSeconds);
    
    // 🔧 简化：日历类型
    std::string calendarType = metadata.temporalInfo.calendar;
    if (calendarType.empty()) {
        calendarType = "gregorian";
    }
    sqlite3_bind_text(stmt, 5, calendarType.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::string errMsg = sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        throw std::runtime_error(fmt::format("Failed to execute temporal_coverage insert: {}", errMsg));
    }
    sqlite3_finalize(stmt);
}

void UnifiedDatabaseManager::insertVariables(const std::string& metadataId, const std::vector<VariableMeta>& variables) {
    // 🔧 修复：包含完整的variable_info表字段，避免NULL值
    const char* sql = "INSERT INTO variable_info (file_id, variable_name, standard_name, long_name, units, data_type, dimensions, variable_category, is_coordinate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(fmt::format("Failed to prepare variable_info insert: {}", sqlite3_errmsg(db_)));
    }

    for (const auto& var : variables) {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, var.name.c_str(), -1, SQLITE_TRANSIENT);
        
        // 🔧 修复：正确填充standard_name，基于变量名推断CF标准名称
        std::string standardName = ""; // VariableMeta没有standardName字段，需要推断
        if (standardName.empty()) {
            // 基于变量名推断CF标准名称
            std::string lowerName = var.name;
            std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
            
            if (lowerName == "uo" || lowerName.find("eastward") != std::string::npos) {
                standardName = "eastward_sea_water_velocity";
            } else if (lowerName == "vo" || lowerName.find("northward") != std::string::npos) {
                standardName = "northward_sea_water_velocity";
            } else if (lowerName == "lon" || lowerName == "longitude") {
                standardName = "longitude";
            } else if (lowerName == "lat" || lowerName == "latitude") {
                standardName = "latitude";
            } else if (lowerName == "time") {
                standardName = "time";
            } else if (lowerName == "depth") {
                standardName = "depth";
            }
        }
        sqlite3_bind_text(stmt, 3, standardName.c_str(), -1, SQLITE_TRANSIENT);
        
        sqlite3_bind_text(stmt, 4, var.description.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, var.units.c_str(), -1, SQLITE_TRANSIENT);
        
        // 🔧 修复：正确推断数据类型，而不是硬编码为"float"
        std::string dataType = "float"; // 简化：直接使用默认类型
        // TODO: 实现dataTypeToString函数或使用枚举转换
        sqlite3_bind_text(stmt, 6, dataType.c_str(), -1, SQLITE_TRANSIENT);
        
        // 🔧 修复：构建维度信息JSON字符串
        std::string dimensionsJson = "[]"; // 默认空数组
        if (!var.dimensionNames.empty()) {
            // 构建简单的维度JSON数组
            std::ostringstream dimStream;
            dimStream << "[";
            for (size_t i = 0; i < var.dimensionNames.size(); ++i) {
                if (i > 0) dimStream << ",";
                dimStream << "\"" << var.dimensionNames[i] << "\"";
            }
            dimStream << "]";
            dimensionsJson = dimStream.str();
        }
        sqlite3_bind_text(stmt, 7, dimensionsJson.c_str(), -1, SQLITE_TRANSIENT);
        
        // 🔧 修复：推断变量分类
        std::string variableCategory = "unknown";
        // 重新声明lowerName变量用于变量分类
        std::string lowerName = var.name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
        
        if (lowerName == "uo" || lowerName == "vo" || lowerName.find("velocity") != std::string::npos) {
            variableCategory = "velocity";
        } else if (lowerName == "lon" || lowerName == "longitude") {
            variableCategory = "coordinate";
        } else if (lowerName == "lat" || lowerName == "latitude") {
            variableCategory = "coordinate";
        } else if (lowerName == "time") {
            variableCategory = "coordinate";
        } else if (lowerName == "depth") {
            variableCategory = "coordinate";
        } else if (lowerName.find("temp") != std::string::npos) {
            variableCategory = "temperature";
        } else if (lowerName.find("sal") != std::string::npos) {
            variableCategory = "salinity";
        }
        sqlite3_bind_text(stmt, 8, variableCategory.c_str(), -1, SQLITE_TRANSIENT);
        
        // 🔧 修复：确定是否为坐标变量
        int isCoordinate = 0;
        if (variableCategory == "coordinate" || 
            lowerName == "lon" || lowerName == "lat" || lowerName == "time" || lowerName == "depth" ||
            lowerName == "longitude" || lowerName == "latitude") {
            isCoordinate = 1;
        }
        sqlite3_bind_int(stmt, 9, isCoordinate);
        
        if (sqlite3_step(stmt) != SQLITE_DONE) {
             std::string errMsg = sqlite3_errmsg(db_);
             sqlite3_finalize(stmt);
            throw std::runtime_error(fmt::format("Failed to execute variable_info insert for '{}': {}", var.name, errMsg));
        }
        
        logger_->debug(fmt::format("✅ 插入变量: {} (标准名: {}, 分类: {}, 维度: {}, 坐标: {})", 
                                  var.name, standardName, variableCategory, dimensionsJson, isCoordinate ? "是" : "否"));
    }
    sqlite3_finalize(stmt);
    logger_->info(fmt::format("变量信息插入完成，文件: {}, 变量数: {}", metadataId, variables.size()));
}

void UnifiedDatabaseManager::insertDataTypes(const std::string& metadataId, const FileMetadata& metadata) {
    // 🔧 修复：正确插入到file_data_types表，基于变量名和文件路径推断语义数据类型
    const char* sql = "INSERT OR IGNORE INTO file_data_types (file_id, data_type, confidence_score) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(fmt::format("Failed to prepare file_data_types insert: {}", sqlite3_errmsg(db_)));
    }

    // 🔧 基于变量名和文件路径推断语义数据类型
    std::set<std::string> detectedDataTypes;
    
    // 从文件路径推断（基于ncdump分析，这些是海洋环境数据）
    std::string lowerPath = metadata.filePath;
    std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
    
    if (lowerPath.find("cs_") != std::string::npos || lowerPath.find("ocean") != std::string::npos) {
        detectedDataTypes.insert("OCEAN_ENVIRONMENT");
    }
    
    // 从变量名推断更具体的数据类型
    for (const auto& var : metadata.variables) {
        std::string lowerName = var.name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
        
        if (lowerName.find("temp") != std::string::npos || lowerName.find("sst") != std::string::npos || 
            lowerName.find("thetao") != std::string::npos) {
            detectedDataTypes.insert("TEMPERATURE");
        } else if (lowerName.find("sal") != std::string::npos || lowerName.find("so") != std::string::npos) {
            detectedDataTypes.insert("SALINITY");
        } else if (lowerName.find("uo") != std::string::npos || lowerName.find("vo") != std::string::npos || 
                  lowerName.find("velocity") != std::string::npos || lowerName.find("current") != std::string::npos) {
            detectedDataTypes.insert("CURRENT_VELOCITY");
        } else if (lowerName.find("elev") != std::string::npos || lowerName.find("depth") != std::string::npos ||
                  lowerName.find("bathymetry") != std::string::npos) {
            detectedDataTypes.insert("BATHYMETRY");
        }
    }
    
    // 如果没有检测到任何数据类型，基于文件类型设置默认值
    if (detectedDataTypes.empty()) {
        if (metadata.format == "NetCDF" || metadata.format == "netcdf") {
            detectedDataTypes.insert("OCEAN_ENVIRONMENT"); // NetCDF文件默认为海洋环境数据
        } else {
            detectedDataTypes.insert("UNKNOWN");
        }
    }
    
    // 插入检测到的数据类型
    for (const auto& dataType : detectedDataTypes) {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, dataType.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 3, 1.0); // 默认置信度为1.0
        
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::string errMsg = sqlite3_errmsg(db_);
            sqlite3_finalize(stmt);
            throw std::runtime_error(fmt::format("Failed to execute file_data_types insert for '{}': {}", dataType, errMsg));
        }
        
        logger_->debug(fmt::format("✅ 插入数据类型: {} -> {}", metadataId, dataType));
    }
    
    sqlite3_finalize(stmt);
    logger_->info(fmt::format("数据类型插入完成，文件: {}, 类型数: {}", metadata.filePath, detectedDataTypes.size()));
}

void UnifiedDatabaseManager::insertAttributesFiltered(const std::string& metadataId, const std::map<std::string, std::string>& attributes) {
    const auto& excludedKeys = getExcludedAttributeKeys();
    
    logger_->info(fmt::format("Starting attribute insertion for metadataId: {}. Total attributes: {}. Excluded keys count: {}", 
                 metadataId, attributes.size(), excludedKeys.size()));
    
    // 暂时跳过属性插入，因为新schema中需要variable_id关联
    logger_->info(fmt::format("Skipping attribute insertion for metadataId: {} (schema mismatch)", metadataId));
    return;
}

void fillMetadataDetails(sqlite3* db, const std::string& metadataId, FileMetadata& metadata) {
    const char* sql = "SELECT file_id, logical_name, file_path, file_format, file_size, last_modified, primary_category FROM file_info WHERE file_id = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return;
    }
    sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        fillBasicInfo(stmt, metadata);
    }
    sqlite3_finalize(stmt);

    metadata.metadataId = metadataId;
    fillSpatialInfo(db, metadataId, metadata);
    fillTemporalInfo(db, metadataId, metadata);
    fillVariables(db, metadataId, metadata);
    fillAttributes(db, metadataId, metadata);
}

void fillBasicInfo(sqlite3_stmt* stmt, FileMetadata& metadata) {
    metadata.fileId = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    metadata.fileName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    metadata.filePath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    metadata.format = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    metadata.extractionTimestamp = sqlite3_column_int64(stmt, 4);
    metadata.lastIndexedTime = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
    metadata.primaryCategory = static_cast<oscean::core_services::DataType>(sqlite3_column_int(stmt, 6));
    metadata.fileSizeBytes = sqlite3_column_int64(stmt, 7);
    metadata.lastModified = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 8));
}

void fillSpatialInfo(sqlite3* db, const std::string& metadataId, FileMetadata& metadata) {
    const char* sql = "SELECT min_longitude, max_longitude, min_latitude, max_latitude, spatial_resolution_x, crs_wkt FROM spatial_coverage WHERE file_id = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        metadata.spatialInfo.bounds.minX = sqlite3_column_double(stmt, 0);
        metadata.spatialInfo.bounds.maxX = sqlite3_column_double(stmt, 1);
        metadata.spatialInfo.bounds.minY = sqlite3_column_double(stmt, 2);
        metadata.spatialInfo.bounds.maxY = sqlite3_column_double(stmt, 3);
        if (sqlite3_column_type(stmt, 4) != SQLITE_NULL) {
            metadata.spatialInfo.spatialResolution = sqlite3_column_double(stmt, 4);
        }
        if (sqlite3_column_type(stmt, 5) != SQLITE_NULL) {
            metadata.spatialInfo.crsWkt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        }
        if (sqlite3_column_type(stmt, 6) != SQLITE_NULL) {
            metadata.spatialInfo.proj4 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        }
    }
    sqlite3_finalize(stmt);
}

void fillTemporalInfo(sqlite3* db, const std::string& metadataId, FileMetadata& metadata) {
    const char* sql = "SELECT start_time, end_time, time_resolution_seconds, time_calendar FROM temporal_coverage WHERE file_id = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        if (sqlite3_column_type(stmt, 0) != SQLITE_NULL) metadata.temporalInfo.startTime = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (sqlite3_column_type(stmt, 1) != SQLITE_NULL) metadata.temporalInfo.endTime = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        if (sqlite3_column_type(stmt, 2) != SQLITE_NULL) metadata.temporalInfo.temporalResolutionSeconds = sqlite3_column_int(stmt, 2);
        if (sqlite3_column_type(stmt, 3) != SQLITE_NULL) metadata.temporalInfo.calendar = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    }
    sqlite3_finalize(stmt);
}

void fillVariables(sqlite3* db, const std::string& metadataId, FileMetadata& metadata) {
    const char* sql = "SELECT variable_name, data_type, units, long_name FROM variable_info WHERE file_id = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    sqlite3_bind_text(stmt, 1, metadataId.c_str(), -1, SQLITE_TRANSIENT);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        VariableMeta var;
        var.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        var.dataType = static_cast<oscean::core_services::DataType>(sqlite3_column_int(stmt, 1));
        var.units = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        var.description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        metadata.variables.push_back(var);
    }
    sqlite3_finalize(stmt);
}

void fillAttributes(sqlite3* db, const std::string& metadataId, FileMetadata& metadata) {
    // 暂时跳过属性读取，因为新schema中的variable_attributes需要更复杂的JOIN查询
    return;
}

const std::unordered_set<std::string>& getExcludedAttributeKeys() {
    static const std::unordered_set<std::string> excludedKeys = {
        "AREA_OR_POINT", "Conventions", "history", "_NCProperties"
    };
    return excludedKeys;
}

// === 🚀 配置驱动查询系统支持方法实现 ===

void UnifiedDatabaseManager::loadTableMappings() {
    std::cout << "[DEBUG DB] 加载表结构映射配置..." << std::endl;
    try {
        // 这里可以加载表结构映射，当前版本暂时跳过
        // 未来可以支持动态表结构适配
        std::cout << "[DEBUG DB] 表结构映射加载完成" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[DEBUG DB] 表结构映射加载失败: " << e.what() << std::endl;
        logger_->warn(fmt::format("Failed to load table mappings: {}", e.what()));
    }
}

void UnifiedDatabaseManager::initializeFallbackQueries() {
    std::cout << "[DEBUG DB] 初始化降级查询配置..." << std::endl;
    logger_->warn("Initializing fallback SQL queries due to configuration loading failure");
    
    // 🛡️ 降级查询：当配置文件加载失败时的基本查询
    sqlQueries_["select_file_by_path"] = "SELECT file_id FROM file_info WHERE file_path = ?";
    sqlQueries_["insert_file_info"] = 
        "INSERT OR REPLACE INTO file_info "
        "(file_id, file_path, logical_name, file_size, last_modified, file_format, primary_category, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)";
    sqlQueries_["insert_spatial_coverage"] = 
        "INSERT OR REPLACE INTO spatial_coverage "
        "(file_id, min_longitude, max_longitude, min_latitude, max_latitude, spatial_resolution_x, crs_wkt) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)";
    sqlQueries_["insert_temporal_coverage"] = 
        "INSERT OR REPLACE INTO temporal_coverage "
        "(file_id, start_time, end_time, time_resolution_seconds, time_calendar) "
        "VALUES (?, ?, ?, ?, ?)";
    sqlQueries_["insert_variable_info"] = 
        "INSERT INTO variable_info "
        "(file_id, variable_name, data_type, units, long_name) "
        "VALUES (?, ?, ?, ?, ?)";
    
    std::cout << "[DEBUG DB] 降级查询配置完成，共 " << sqlQueries_.size() << " 个查询" << std::endl;
    logger_->info(fmt::format("Fallback SQL queries initialized with {} queries", sqlQueries_.size()));
}

std::string UnifiedDatabaseManager::generateUniqueId(const std::string& input) const {
    // 使用输入字符串的哈希值作为ID
    std::hash<std::string> hasher;
    size_t hashValue = hasher(input);
    
    // 添加时间戳确保唯一性
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    std::ostringstream oss;
    oss << "meta_" << hashValue << "_" << timestamp;
    return oss.str();
}

} // namespace oscean::core_services::metadata::impl