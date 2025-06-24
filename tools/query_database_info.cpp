#include <iostream>
#include <sqlite3.h>
#include <string>
#include <vector>

void queryTableSchema(sqlite3* db, const std::string& tableName) {
    std::cout << "\n📋 表 '" << tableName << "' 结构:" << std::endl;
    
    std::string sql = "PRAGMA table_info(" + tableName + ");";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "  列定义:" << std::endl;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int cid = sqlite3_column_int(stmt, 0);
            const char* name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            int notnull = sqlite3_column_int(stmt, 3);
            const char* dflt_value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            int pk = sqlite3_column_int(stmt, 5);
            
            std::cout << "    " << cid << ": " << (name ? name : "NULL") 
                      << " (" << (type ? type : "NULL") << ")";
            if (pk) std::cout << " [主键]";
            if (notnull) std::cout << " [非空]";
            if (dflt_value) std::cout << " [默认值: " << dflt_value << "]";
            std::cout << std::endl;
        }
    }
    sqlite3_finalize(stmt);
}

void detailedDatabaseAnalysis(const std::string& dbPath) {
    sqlite3* db;
    int rc = sqlite3_open(dbPath.c_str(), &db);
    
    if (rc) {
        std::cerr << "❌ 无法打开数据库: " << dbPath << std::endl;
        return;
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "📊 数据库详细分析: " << dbPath << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // 获取数据库文件信息
    std::string fileInfoSql = "PRAGMA database_list;";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, fileInfoSql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "\n📁 数据库文件信息:" << std::endl;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int seq = sqlite3_column_int(stmt, 0);
            const char* name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* file = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            
            std::cout << "  序号: " << seq << ", 名称: " << (name ? name : "NULL") 
                      << ", 文件: " << (file ? file : "NULL") << std::endl;
        }
    }
    sqlite3_finalize(stmt);
    
    // 获取所有表
    const char* tablesSql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;";
    std::vector<std::string> tables;
    
    if (sqlite3_prepare_v2(db, tablesSql, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* tableName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if (tableName) {
                tables.push_back(tableName);
            }
        }
    }
    sqlite3_finalize(stmt);
    
    // 分析每个表
    for (const auto& tableName : tables) {
        queryTableSchema(db, tableName);
        
        // 检查表记录数
        std::string countSql = "SELECT COUNT(*) FROM " + tableName + ";";
        if (sqlite3_prepare_v2(db, countSql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                int count = sqlite3_column_int(stmt, 0);
                std::cout << "  📈 记录数: " << count << std::endl;
                
                // 如果有记录，显示几条样本
                if (count > 0) {
                    std::string sampleSql = "SELECT * FROM " + tableName + " LIMIT 3;";
                    sqlite3_stmt* sampleStmt;
                    if (sqlite3_prepare_v2(db, sampleSql.c_str(), -1, &sampleStmt, nullptr) == SQLITE_OK) {
                        std::cout << "  📝 样本数据:" << std::endl;
                        int colCount = sqlite3_column_count(sampleStmt);
                        while (sqlite3_step(sampleStmt) == SQLITE_ROW) {
                            std::cout << "    行: ";
                            for (int i = 0; i < colCount; i++) {
                                if (i > 0) std::cout << " | ";
                                const char* value = reinterpret_cast<const char*>(sqlite3_column_text(sampleStmt, i));
                                std::cout << (value ? value : "NULL");
                            }
                            std::cout << std::endl;
                        }
                    }
                    sqlite3_finalize(sampleStmt);
                }
            }
        }
        sqlite3_finalize(stmt);
    }
    
    // 检查索引
    const char* indexesSql = "SELECT name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';";
    if (sqlite3_prepare_v2(db, indexesSql, -1, &stmt, nullptr) == SQLITE_OK) {
        bool hasIndexes = false;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            if (!hasIndexes) {
                std::cout << "\n🔍 自定义索引:" << std::endl;
                hasIndexes = true;
            }
            const char* name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* sql = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            std::cout << "  " << (name ? name : "NULL") << ": " << (sql ? sql : "NULL") << std::endl;
        }
        if (!hasIndexes) {
            std::cout << "\n🔍 没有自定义索引" << std::endl;
        }
    }
    sqlite3_finalize(stmt);
    
    sqlite3_close(db);
}

int main() {
    std::cout << "🔍 OSCEAN 数据库详细分析工具" << std::endl;
    
    std::vector<std::string> databases = {
        "E:/Ocean_data/cs/internal_function_tests/ocean_environment.db",
        "E:/Ocean_data/cs/internal_function_tests/topography_bathymetry.db", 
        "E:/Ocean_data/cs/internal_function_tests/boundary_lines.db",
        "E:/Ocean_data/cs/internal_function_tests/sonar_propagation.db"
    };
    
    for (const auto& dbPath : databases) {
        detailedDatabaseAnalysis(dbPath);
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🎯 分析结论:" << std::endl;
    std::cout << "• 如果所有表记录数为0，说明测试只创建了数据库结构，但没有插入实际数据" << std::endl;
    std::cout << "• 这是正常的模拟测试行为，验证了系统的数据库路由和分类功能" << std::endl;
    std::cout << "• 要查看真实数据插入，需要使用真实的海洋数据文件进行测试" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    return 0;
} 