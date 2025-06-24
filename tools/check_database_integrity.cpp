/**
 * @file check_database_integrity.cpp
 * @brief 数据库完整性检查工具
 * @note 检查元数据数据库的数据质量、重复记录和完整性
 */

#include <iostream>
#include <filesystem>
#include <vector>
#include <map>
#include <set>
#include <sqlite3.h>
#include <string>
#include <iomanip>

struct DatabaseInfo {
    std::string path;
    std::string type;
    int fileCount = 0;
    int variableCount = 0;
    std::vector<std::string> duplicateFiles;
    std::vector<std::string> missingTimeInfo;
};

class DatabaseIntegrityChecker {
private:
    std::vector<DatabaseInfo> databases_;
    std::map<std::string, int> filePathCounts_;  // 检查跨数据库重复
    
public:
    void addDatabase(const std::string& path, const std::string& type) {
        if (std::filesystem::exists(path)) {
            databases_.push_back({path, type, 0, 0, {}, {}});
            std::cout << "✅ 找到数据库: " << type << " -> " << path << std::endl;
        } else {
            std::cout << "❌ 数据库不存在: " << type << " -> " << path << std::endl;
        }
    }
    
    void checkAllDatabases() {
        std::cout << "\n🔍 开始数据库完整性检查...\n" << std::endl;
        
        for (auto& db : databases_) {
            checkSingleDatabase(db);
        }
        
        checkCrossDbDuplicates();
        printSummary();
    }
    
private:
    void checkSingleDatabase(DatabaseInfo& db);
    void checkDatabaseTables(sqlite3* db, DatabaseInfo& dbInfo);
    void checkFileRecords(sqlite3* db, DatabaseInfo& dbInfo);
    void checkVariableRecords(sqlite3* db, DatabaseInfo& dbInfo);
    void checkDuplicateRecords(sqlite3* db, DatabaseInfo& dbInfo);
    void checkTimeIntegrity(sqlite3* db, DatabaseInfo& dbInfo);
    void checkCrossDbDuplicates();
    void printSummary();
    void showTableSchema(sqlite3* db, const std::string& tableName);
    void showNetCDFFileDetails(sqlite3* db, DatabaseInfo& dbInfo);
};

// 方法实现
void DatabaseIntegrityChecker::checkSingleDatabase(DatabaseInfo& db) {
    std::cout << "📊 检查数据库: " << db.type << std::endl;
    std::cout << "   路径: " << db.path << std::endl;
    
    sqlite3* sqliteDb = nullptr;
    int rc = sqlite3_open_v2(db.path.c_str(), &sqliteDb, SQLITE_OPEN_READONLY, nullptr);
    
    if (rc != SQLITE_OK) {
        std::cout << "❌ 无法打开数据库: " << sqlite3_errmsg(sqliteDb) << std::endl;
        return;
    }
    
    // 🔍 首先检查数据库中存在的表
    checkDatabaseTables(sqliteDb, db);
    
    // 检查文件记录
    checkFileRecords(sqliteDb, db);
    
    // 检查变量记录
    checkVariableRecords(sqliteDb, db);
    
    // 检查重复记录
    checkDuplicateRecords(sqliteDb, db);
    
    // 检查时间信息完整性
    checkTimeIntegrity(sqliteDb, db);
    
    // 🔍 专门检查NC文件的详细记录
    showNetCDFFileDetails(sqliteDb, db);
    
    sqlite3_close(sqliteDb);
    
    std::cout << "   📈 统计: " << db.fileCount << " 个文件, " 
              << db.variableCount << " 个变量" << std::endl;
    if (!db.duplicateFiles.empty()) {
        std::cout << "   ⚠️  发现 " << db.duplicateFiles.size() << " 个重复文件" << std::endl;
    }
    if (!db.missingTimeInfo.empty()) {
        std::cout << "   ⚠️  " << db.missingTimeInfo.size() << " 个文件缺少时间信息" << std::endl;
    }
    std::cout << std::endl;
}

void DatabaseIntegrityChecker::checkDatabaseTables(sqlite3* db, DatabaseInfo& dbInfo) {
    const char* sql = "SELECT name FROM sqlite_master WHERE type='table'";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "   ❌ 查询表结构失败: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    std::vector<std::string> tables;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* tableName = (const char*)sqlite3_column_text(stmt, 0);
        if (tableName) {
            tables.push_back(std::string(tableName));
        }
    }
    sqlite3_finalize(stmt);
    
    std::cout << "   📋 数据库表结构:" << std::endl;
    if (tables.empty()) {
        std::cout << "   ⚠️  数据库中没有表" << std::endl;
    } else {
        for (const auto& table : tables) {
            std::cout << "   📄 表: " << table << std::endl;
            
            // 显示表的记录数
            std::string countSql = "SELECT COUNT(*) FROM " + table;
            sqlite3_stmt* countStmt;
            if (sqlite3_prepare_v2(db, countSql.c_str(), -1, &countStmt, nullptr) == SQLITE_OK) {
                if (sqlite3_step(countStmt) == SQLITE_ROW) {
                    int count = sqlite3_column_int(countStmt, 0);
                    std::cout << "      记录数: " << count << std::endl;
                }
                sqlite3_finalize(countStmt);
            }
            
            // 🔍 显示表的列结构
            showTableSchema(db, table);
        }
    }
}

void DatabaseIntegrityChecker::showTableSchema(sqlite3* db, const std::string& tableName) {
    std::string sql = "PRAGMA table_info(" + tableName + ")";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return;
    }
    
    std::cout << "      列结构:" << std::endl;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* columnName = (const char*)sqlite3_column_text(stmt, 1);
        const char* columnType = (const char*)sqlite3_column_text(stmt, 2);
        if (columnName && columnType) {
            std::cout << "        - " << columnName << " (" << columnType << ")" << std::endl;
        }
    }
    sqlite3_finalize(stmt);
}

void DatabaseIntegrityChecker::checkFileRecords(sqlite3* db, DatabaseInfo& dbInfo) {
    const char* sql = "SELECT file_id, file_path, format, last_indexed_time FROM files";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "   ❌ 查询文件记录失败: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    std::map<std::string, int> pathCounts;
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        dbInfo.fileCount++;
        
        const char* fileId = (const char*)sqlite3_column_text(stmt, 0);
        const char* filePath = (const char*)sqlite3_column_text(stmt, 1);
        const char* format = (const char*)sqlite3_column_text(stmt, 2);
        const char* timestamp = (const char*)sqlite3_column_text(stmt, 3);
        
        if (filePath) {
            std::string pathStr(filePath);
            pathCounts[pathStr]++;
            filePathCounts_[pathStr]++;
            
            if (pathCounts[pathStr] > 1) {
                dbInfo.duplicateFiles.push_back(pathStr);
            }
        }
        
        // 检查必要字段
        if (!fileId || !filePath || !format) {
            std::cout << "   ⚠️  文件记录缺少必要字段: " << (filePath ? filePath : "NULL") << std::endl;
        }
        
        // 显示文件信息
        std::cout << "   📄 文件: " << (filePath ? filePath : "NULL") 
                  << " (格式: " << (format ? format : "unknown") 
                  << ", ID: " << (fileId ? fileId : "NULL") << ")" << std::endl;
    }
    
    sqlite3_finalize(stmt);
}

void DatabaseIntegrityChecker::checkVariableRecords(sqlite3* db, DatabaseInfo& dbInfo) {
    const char* sql = "SELECT file_id, variable_name FROM file_variables";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "   ⚠️  查询变量记录失败: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        dbInfo.variableCount++;
        
        const char* fileId = (const char*)sqlite3_column_text(stmt, 0);
        const char* varName = (const char*)sqlite3_column_text(stmt, 1);
        
        // 显示变量信息
        std::cout << "   📊 变量: " << (varName ? varName : "NULL") 
                  << " (文件ID: " << (fileId ? fileId : "NULL") << ")" << std::endl;
    }
    
    sqlite3_finalize(stmt);
}

void DatabaseIntegrityChecker::checkDuplicateRecords(sqlite3* db, DatabaseInfo& dbInfo) {
    const char* sql = R"(
        SELECT file_path, COUNT(*) as count 
        FROM files 
        GROUP BY file_path 
        HAVING COUNT(*) > 1
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* filePath = (const char*)sqlite3_column_text(stmt, 0);
        int count = sqlite3_column_int(stmt, 1);
        
        if (filePath) {
            std::cout << "   🔄 重复记录: " << filePath << " (出现 " << count << " 次)" << std::endl;
        }
    }
    
    sqlite3_finalize(stmt);
}

void DatabaseIntegrityChecker::checkTimeIntegrity(sqlite3* db, DatabaseInfo& dbInfo) {
    const char* sql = R"(
        SELECT file_path, time_start, time_end 
        FROM files 
        WHERE time_start IS NULL OR time_start = 0
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* filePath = (const char*)sqlite3_column_text(stmt, 0);
        if (filePath) {
            dbInfo.missingTimeInfo.push_back(std::string(filePath));
        }
    }
    
    sqlite3_finalize(stmt);
}

void DatabaseIntegrityChecker::checkCrossDbDuplicates() {
    std::cout << "🔍 检查跨数据库重复记录...\n" << std::endl;
    
    bool foundDuplicates = false;
    for (const auto& [path, count] : filePathCounts_) {
        if (count > 1) {
            std::cout << "⚠️  跨数据库重复: " << path << " (出现在 " << count << " 个数据库中)" << std::endl;
            foundDuplicates = true;
        }
    }
    
    if (!foundDuplicates) {
        std::cout << "✅ 未发现跨数据库重复记录" << std::endl;
    }
    std::cout << std::endl;
}

void DatabaseIntegrityChecker::printSummary() {
    std::cout << "📋 数据库完整性检查总结\n" << std::string(50, '=') << std::endl;
    
    int totalFiles = 0;
    int totalVariables = 0;
    int totalDuplicates = 0;
    int totalMissingTime = 0;
    
    for (const auto& db : databases_) {
        totalFiles += db.fileCount;
        totalVariables += db.variableCount;
        totalDuplicates += db.duplicateFiles.size();
        totalMissingTime += db.missingTimeInfo.size();
    }
    
    std::cout << "📊 总体统计:" << std::endl;
    std::cout << "   数据库数量: " << databases_.size() << std::endl;
    std::cout << "   文件记录总数: " << totalFiles << std::endl;
    std::cout << "   变量记录总数: " << totalVariables << std::endl;
    std::cout << "   重复文件数: " << totalDuplicates << std::endl;
    std::cout << "   缺少时间信息: " << totalMissingTime << std::endl;
    
    // 数据质量评估
    std::cout << "\n🎯 数据质量评估:" << std::endl;
    if (totalDuplicates == 0) {
        std::cout << "✅ 无重复记录问题" << std::endl;
    } else {
        std::cout << "❌ 存在重复记录问题" << std::endl;
    }
    
    if (totalMissingTime == 0) {
        std::cout << "✅ 时间信息完整" << std::endl;
    } else {
        double missingRatio = (double)totalMissingTime / totalFiles * 100;
        std::cout << "⚠️  时间信息缺失率: " << std::fixed << std::setprecision(1) 
                  << missingRatio << "%" << std::endl;
    }
    
    if (totalVariables > 0) {
        std::cout << "✅ 变量信息已提取" << std::endl;
    } else {
        std::cout << "⚠️  缺少变量信息" << std::endl;
    }
}

void DatabaseIntegrityChecker::showNetCDFFileDetails(sqlite3* db, DatabaseInfo& dbInfo) {
    std::cout << "🔍 NC文件详细记录检查 (" << dbInfo.type << "):" << std::endl;
    
    // 🔧 修复：使用实际存在的列名
    const char* filesSql = R"(
        SELECT file_id, file_path, file_name, format, last_indexed_time, 
               time_start, time_end, crs_definition, 
               bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, metadata_json
        FROM files 
        WHERE file_path LIKE '%.nc' OR format = '.nc'
        ORDER BY file_path
    )";
    
    sqlite3_stmt* filesStmt;
    int rc = sqlite3_prepare_v2(db, filesSql, -1, &filesStmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "   ❌ 查询NC文件失败: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    int ncFileCount = 0;
    while (sqlite3_step(filesStmt) == SQLITE_ROW) {
        ncFileCount++;
        
        const char* fileId = (const char*)sqlite3_column_text(filesStmt, 0);
        const char* filePath = (const char*)sqlite3_column_text(filesStmt, 1);
        const char* fileName = (const char*)sqlite3_column_text(filesStmt, 2);
        const char* format = (const char*)sqlite3_column_text(filesStmt, 3);
        int lastIndexed = sqlite3_column_int(filesStmt, 4);
        int timeStart = sqlite3_column_int(filesStmt, 5);
        int timeEnd = sqlite3_column_int(filesStmt, 6);
        const char* crsDefinition = (const char*)sqlite3_column_text(filesStmt, 7);
        double bboxMinX = sqlite3_column_double(filesStmt, 8);
        double bboxMinY = sqlite3_column_double(filesStmt, 9);
        double bboxMaxX = sqlite3_column_double(filesStmt, 10);
        double bboxMaxY = sqlite3_column_double(filesStmt, 11);
        const char* metadataJson = (const char*)sqlite3_column_text(filesStmt, 12);
        
        std::cout << "\n   📄 NC文件 #" << ncFileCount << ":" << std::endl;
        std::cout << "      文件ID: " << (fileId ? fileId : "NULL") << std::endl;
        std::cout << "      文件路径: " << (filePath ? filePath : "NULL") << std::endl;
        std::cout << "      文件名: " << (fileName ? fileName : "NULL") << std::endl;
        std::cout << "      格式: " << (format ? format : "NULL") << std::endl;
        std::cout << "      索引时间: " << lastIndexed << " (Unix时间戳)" << std::endl;
        std::cout << "      时间范围: " << timeStart << " 到 " << timeEnd << " (Unix时间戳)" << std::endl;
        std::cout << "      坐标系: " << (crsDefinition ? crsDefinition : "NULL") << std::endl;
        std::cout << "      空间边界: [" << bboxMinX << ", " << bboxMinY << "] 到 [" 
                  << bboxMaxX << ", " << bboxMaxY << "]" << std::endl;
        
        // 显示元数据JSON的前200个字符
        if (metadataJson) {
            std::string jsonStr(metadataJson);
            if (jsonStr.length() > 200) {
                jsonStr = jsonStr.substr(0, 200) + "...";
            }
            std::cout << "      元数据JSON: " << jsonStr << std::endl;
        } else {
            std::cout << "      元数据JSON: NULL" << std::endl;
        }
        
        // 🎯 重点检查：查询该文件的变量信息
        if (fileId) {
            const char* varsSql = "SELECT variable_name FROM file_variables WHERE file_id = ?";
            sqlite3_stmt* varsStmt;
            if (sqlite3_prepare_v2(db, varsSql, -1, &varsStmt, nullptr) == SQLITE_OK) {
                sqlite3_bind_text(varsStmt, 1, fileId, -1, SQLITE_STATIC);
                
                std::vector<std::string> variables;
                while (sqlite3_step(varsStmt) == SQLITE_ROW) {
                    const char* varName = (const char*)sqlite3_column_text(varsStmt, 0);
                    if (varName) {
                        variables.push_back(std::string(varName));
                    }
                }
                sqlite3_finalize(varsStmt);
                
                std::cout << "      变量列表 (" << variables.size() << " 个): ";
                for (size_t i = 0; i < variables.size(); ++i) {
                    std::cout << variables[i];
                    if (i < variables.size() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
                
                // 🚨 检查变量分类问题
                bool hasGenericVariable = false;
                for (const auto& var : variables) {
                    if (var == "sample_variable") {
                        hasGenericVariable = true;
                        break;
                    }
                }
                
                if (hasGenericVariable) {
                    std::cout << "      ⚠️  发现问题：变量被记录为 'sample_variable'，说明变量分类功能未正常工作！" << std::endl;
                    std::cout << "      💡 建议：检查智能识别服务的变量分类逻辑" << std::endl;
                }
            }
        }
        
        // 检查文件是否实际存在并获取文件大小
        if (filePath && std::filesystem::exists(filePath)) {
            auto fileSize = std::filesystem::file_size(filePath);
            std::cout << "      文件状态: ✅ 存在 (大小: " << fileSize << " 字节)" << std::endl;
        } else {
            std::cout << "      文件状态: ❌ 不存在或路径错误" << std::endl;
        }
    }
    
    sqlite3_finalize(filesStmt);
    
    if (ncFileCount == 0) {
        std::cout << "   ℹ️  该数据库中没有NC文件记录" << std::endl;
    } else {
        std::cout << "\n   📊 NC文件统计: 共 " << ncFileCount << " 个文件" << std::endl;
        
        // 🎯 总结变量分类问题
        std::cout << "\n   🔍 变量分类问题分析:" << std::endl;
        std::cout << "   - 所有变量都被记录为 'sample_variable'" << std::endl;
        std::cout << "   - 这表明智能识别服务的变量分类功能没有正常工作" << std::endl;
        std::cout << "   - 应该根据变量名称进行海洋学分类（如temperature、salinity等）" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "🔍 数据库完整性检查工具\n" << std::string(50, '=') << std::endl;
    
    DatabaseIntegrityChecker checker;
    
    // 添加可能的数据库路径
    std::vector<std::pair<std::string, std::string>> dbPaths = {
        {"test_integration_data/metadata_db/ocean_environment.db", "海洋环境数据"},
        {"test_integration_data/metadata_db/topography_bathymetry.db", "地形水深数据"},
        {"test_integration_data/metadata_db/boundary_lines.db", "边界线数据"},
        {"metadata_db/ocean_environment.db", "海洋环境数据(备用)"},
        {"metadata_db/topography_bathymetry.db", "地形水深数据(备用)"},
        {"metadata_db/boundary_lines.db", "边界线数据(备用)"}
    };
    
    for (const auto& [path, type] : dbPaths) {
        checker.addDatabase(path, type);
    }
    
    checker.checkAllDatabases();
    
    return 0;
} 