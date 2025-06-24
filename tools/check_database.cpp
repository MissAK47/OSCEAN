#include <iostream>
#include <sqlite3.h>
#include <string>
#include <iomanip>
#include <ctime>
#include <vector>

int callback(void* data, int argc, char** argv, char** azColName) {
    for (int i = 0; i < argc; i++) {
        std::cout << azColName[i] << ": " << (argv[i] ? argv[i] : "NULL") << " | ";
    }
    std::cout << std::endl;
    return 0;
}

void queryDatabase(const std::string& dbPath, const std::string& query, const std::string& description = "") {
    sqlite3* db;
    char* errMsg = nullptr;
    
    int rc = sqlite3_open(dbPath.c_str(), &db);
    if (rc) {
        std::cerr << "无法打开数据库: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    if (!description.empty()) {
        std::cout << "\n--- " << description << " ---" << std::endl;
    }
    std::cout << "数据库: " << dbPath << std::endl;
    std::cout << "SQL: " << query << std::endl;
    std::cout << "结果:" << std::endl;
    
    rc = sqlite3_exec(db, query.c_str(), callback, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL错误: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    } else {
        std::cout << "(查询完成)" << std::endl;
    }
    
    sqlite3_close(db);
}

void checkFileMetadata(const std::string& dbPath, const std::string& dbName) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "检查数据库: " << dbName << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 基本统计
    queryDatabase(dbPath, "SELECT COUNT(*) as total_files FROM files;", "1. 文件总数");
    
    // 检查元数据JSON内容
    queryDatabase(dbPath, "SELECT file_name, metadata_json FROM files WHERE metadata_json IS NOT NULL AND metadata_json != '' LIMIT 3;", "2. 元数据JSON内容样本");
    
    // 检查文件路径和时间信息
    queryDatabase(dbPath, "SELECT file_name, file_path, time_start, time_end, last_indexed_time FROM files LIMIT 5;", "3. 文件路径和时间信息");
    
    // 检查CRS信息
    queryDatabase(dbPath, "SELECT file_name, crs_definition FROM files WHERE crs_definition IS NOT NULL AND crs_definition != '' LIMIT 3;", "4. CRS坐标系信息");
    
    // 检查边界框完整性
    queryDatabase(dbPath, "SELECT COUNT(*) as valid_bbox FROM files WHERE bbox_min_x IS NOT NULL AND bbox_min_y IS NOT NULL AND bbox_max_x IS NOT NULL AND bbox_max_y IS NOT NULL;", "5. 有效边界框数量");
    
    // 检查变量映射
    queryDatabase(dbPath, "SELECT COUNT(DISTINCT file_id) as files_with_variables FROM file_variables;", "6. 有变量映射的文件数");
    
    // 详细变量分布
    queryDatabase(dbPath, "SELECT variable_name, COUNT(*) as count FROM file_variables GROUP BY variable_name ORDER BY count DESC;", "7. 变量分布详细统计");
    
    // 检查重复文件
    queryDatabase(dbPath, "SELECT file_name, COUNT(*) as count FROM files GROUP BY file_name HAVING count > 1;", "8. 重复文件检查");
    
    // 文件格式分布
    queryDatabase(dbPath, "SELECT format, COUNT(*) as count FROM files GROUP BY format;", "9. 文件格式分布");
    
    // 检查空值情况
    queryDatabase(dbPath, "SELECT 'metadata_json' as field, COUNT(*) as null_count FROM files WHERE metadata_json IS NULL OR metadata_json = '' UNION SELECT 'crs_definition', COUNT(*) FROM files WHERE crs_definition IS NULL OR crs_definition = '' UNION SELECT 'time_start', COUNT(*) FROM files WHERE time_start IS NULL OR time_start = 0;", "10. 空值字段统计");
}

void printDatabaseInfo(const std::string& dbPath) {
    sqlite3* db;
    int rc = sqlite3_open(dbPath.c_str(), &db);
    
    if (rc) {
        std::cerr << "无法打开数据库: " << dbPath << " - " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    std::cout << "\n=== 数据库: " << dbPath << " ===" << std::endl;
    
    // 查询所有表
    const char* sql = "SELECT name FROM sqlite_master WHERE type='table';";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "📋 表列表:" << std::endl;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* tableName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            std::cout << "  - " << tableName << std::endl;
            
            // 查询每个表的记录数
            std::string countSql = "SELECT COUNT(*) FROM " + std::string(tableName) + ";";
            sqlite3_stmt* countStmt;
            if (sqlite3_prepare_v2(db, countSql.c_str(), -1, &countStmt, nullptr) == SQLITE_OK) {
                if (sqlite3_step(countStmt) == SQLITE_ROW) {
                    int count = sqlite3_column_int(countStmt, 0);
                    std::cout << "    记录数: " << count << std::endl;
                }
            }
            sqlite3_finalize(countStmt);
        }
        sqlite3_finalize(stmt);
    }
    
    // 查询元数据表的详细内容（如果存在）
    const char* metadataQuery = "SELECT id, file_path, data_type, creation_time FROM comprehensive_metadata LIMIT 5;";
    if (sqlite3_prepare_v2(db, metadataQuery, -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "\n📊 元数据样本 (前5条):" << std::endl;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* filePath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* dataType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            const char* creationTime = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            
            std::cout << "  ID: " << (id ? id : "NULL") << std::endl;
            std::cout << "  文件: " << (filePath ? filePath : "NULL") << std::endl;
            std::cout << "  类型: " << (dataType ? dataType : "NULL") << std::endl;
            std::cout << "  时间: " << (creationTime ? creationTime : "NULL") << std::endl;
            std::cout << "  ---" << std::endl;
        }
        sqlite3_finalize(stmt);
    }
    
    sqlite3_close(db);
}

int main() {
    std::cout << "🔍 OSCEAN 数据库内容检查工具" << std::endl;
    
    std::vector<std::string> databases = {
        "E:/Ocean_data/cs/internal_function_tests/ocean_environment.db",
        "E:/Ocean_data/cs/internal_function_tests/topography_bathymetry.db", 
        "E:/Ocean_data/cs/internal_function_tests/boundary_lines.db",
        "E:/Ocean_data/cs/internal_function_tests/sonar_propagation.db"
    };
    
    for (const auto& dbPath : databases) {
        printDatabaseInfo(dbPath);
    }
    
    return 0;
} 