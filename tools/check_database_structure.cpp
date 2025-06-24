#include <iostream>
#include <filesystem>
#include <sqlite3.h>
#include <vector>
#include <string>

// 简单的数据库检查工具
class DatabaseChecker {
private:
    sqlite3* db = nullptr;
    
public:
    bool openDatabase(const std::string& dbPath) {
        int rc = sqlite3_open(dbPath.c_str(), &db);
        if (rc != SQLITE_OK) {
            std::cerr << "无法打开数据库: " << sqlite3_errmsg(db) << std::endl;
            return false;
        }
        std::cout << "成功打开数据库: " << dbPath << std::endl;
        return true;
    }
    
    void showTableStructure(const std::string& tableName) {
        std::cout << "\n=== 表结构: " << tableName << " ===" << std::endl;
        
        std::string sql = "PRAGMA table_info(" + tableName + ");";
        sqlite3_stmt* stmt;
        
        int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            std::cerr << "查询失败: " << sqlite3_errmsg(db) << std::endl;
            return;
        }
        
        std::cout << "列ID | 列名 | 类型 | 非空 | 默认值 | 主键" << std::endl;
        std::cout << "------|------|------|------|--------|------" << std::endl;
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int cid = sqlite3_column_int(stmt, 0);
            const char* name = (const char*)sqlite3_column_text(stmt, 1);
            const char* type = (const char*)sqlite3_column_text(stmt, 2);
            int notnull = sqlite3_column_int(stmt, 3);
            const char* dflt_value = (const char*)sqlite3_column_text(stmt, 4);
            int pk = sqlite3_column_int(stmt, 5);
            
            std::cout << cid << " | " << (name ? name : "NULL") << " | " 
                      << (type ? type : "NULL") << " | " << notnull << " | "
                      << (dflt_value ? dflt_value : "NULL") << " | " << pk << std::endl;
        }
        
        sqlite3_finalize(stmt);
    }
    
    void showTableData(const std::string& tableName, int limit = 10) {
        std::cout << "\n=== 表数据: " << tableName << " (前" << limit << "条) ===" << std::endl;
        
        std::string sql = "SELECT * FROM " + tableName + " LIMIT " + std::to_string(limit) + ";";
        sqlite3_stmt* stmt;
        
        int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            std::cerr << "查询失败: " << sqlite3_errmsg(db) << std::endl;
            return;
        }
        
        // 显示列名
        int columnCount = sqlite3_column_count(stmt);
        for (int i = 0; i < columnCount; i++) {
            std::cout << sqlite3_column_name(stmt, i);
            if (i < columnCount - 1) std::cout << " | ";
        }
        std::cout << std::endl;
        
        // 显示分隔线
        for (int i = 0; i < columnCount; i++) {
            std::cout << "--------";
            if (i < columnCount - 1) std::cout << "-|-";
        }
        std::cout << std::endl;
        
        // 显示数据
        int rowCount = 0;
        while (sqlite3_step(stmt) == SQLITE_ROW && rowCount < limit) {
            for (int i = 0; i < columnCount; i++) {
                const char* text = (const char*)sqlite3_column_text(stmt, i);
                std::cout << (text ? text : "NULL");
                if (i < columnCount - 1) std::cout << " | ";
            }
            std::cout << std::endl;
            rowCount++;
        }
        
        if (rowCount == 0) {
            std::cout << "(表为空)" << std::endl;
        }
        
        sqlite3_finalize(stmt);
    }
    
    void showTableCount(const std::string& tableName) {
        std::string sql = "SELECT COUNT(*) FROM " + tableName + ";";
        sqlite3_stmt* stmt;
        
        int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
            int count = sqlite3_column_int(stmt, 0);
            std::cout << "表 " << tableName << " 记录数: " << count << std::endl;
        }
        sqlite3_finalize(stmt);
    }
    
    void checkDatabase() {
        if (!db) {
            std::cerr << "数据库未打开" << std::endl;
            return;
        }
        
        // 检查所有表
        std::vector<std::string> tables = {"metadata", "files", "file_variables", "datasets", "dataset_variables"};
        
        std::cout << "\n=== 数据库概览 ===" << std::endl;
        for (const auto& table : tables) {
            showTableCount(table);
        }
        
        // 显示每个表的结构和数据
        for (const auto& table : tables) {
            showTableStructure(table);
            showTableData(table, 5);
        }
    }
    
    ~DatabaseChecker() {
        if (db) {
            sqlite3_close(db);
        }
    }
};

void checkTableStructure(sqlite3* db, const std::string& tableName) {
    std::cout << "\n=== 表结构: " << tableName << " ===" << std::endl;
    
    std::string sql = "PRAGMA table_info(" + tableName + ");";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "字段名\t\t类型\t\t非空\t默认值\t主键" << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int cid = sqlite3_column_int(stmt, 0);
            const char* name = (const char*)sqlite3_column_text(stmt, 1);
            const char* type = (const char*)sqlite3_column_text(stmt, 2);
            int notnull = sqlite3_column_int(stmt, 3);
            const char* dflt_value = (const char*)sqlite3_column_text(stmt, 4);
            int pk = sqlite3_column_int(stmt, 5);
            
            std::cout << (name ? name : "NULL") << "\t\t"
                     << (type ? type : "NULL") << "\t\t"
                     << (notnull ? "是" : "否") << "\t\t"
                     << (dflt_value ? dflt_value : "NULL") << "\t"
                     << (pk ? "是" : "否") << std::endl;
        }
    }
    sqlite3_finalize(stmt);
}

void checkVariableClassification(sqlite3* db) {
    std::cout << "\n=== 变量分类信息检查 ===" << std::endl;
    
    // 检查ocean_variables表中的分类信息
    const char* sql = R"(
        SELECT 
            ov.product_category,
            ov.model_name,
            ov.measurement_type,
            vi.name,
            vi.standard_name,
            vi.units,
            COUNT(*) as count
        FROM ocean_variables ov
        JOIN variable_info vi ON ov.variable_id = vi.id
        GROUP BY ov.product_category, ov.model_name, ov.measurement_type, vi.name
        ORDER BY ov.product_category, count DESC;
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "产品类别\t\t模型名称\t\t测量类型\t\t变量名\t\t标准名\t\t单位\t\t数量" << std::endl;
        std::cout << "-------------------------------------------------------------------------------------" << std::endl;
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* product_category = (const char*)sqlite3_column_text(stmt, 0);
            const char* model_name = (const char*)sqlite3_column_text(stmt, 1);
            const char* measurement_type = (const char*)sqlite3_column_text(stmt, 2);
            const char* name = (const char*)sqlite3_column_text(stmt, 3);
            const char* standard_name = (const char*)sqlite3_column_text(stmt, 4);
            const char* units = (const char*)sqlite3_column_text(stmt, 5);
            int count = sqlite3_column_int(stmt, 6);
            
            std::cout << (product_category ? product_category : "NULL") << "\t\t"
                     << (model_name ? model_name : "NULL") << "\t\t"
                     << (measurement_type ? measurement_type : "NULL") << "\t\t"
                     << (name ? name : "NULL") << "\t\t"
                     << (standard_name ? standard_name : "NULL") << "\t\t"
                     << (units ? units : "NULL") << "\t\t"
                     << count << std::endl;
        }
    } else {
        std::cout << "查询ocean_variables表失败: " << sqlite3_errmsg(db) << std::endl;
    }
    sqlite3_finalize(stmt);
}

void checkTimeFormat(sqlite3* db) {
    std::cout << "\n=== 时间格式信息检查 ===" << std::endl;
    
    const char* sql = R"(
        SELECT 
            me.file_name,
            ti.start_time,
            ti.end_time,
            ti.temporal_resolution_seconds,
            ti.time_units,
            ti.calendar,
            ti.is_regular
        FROM metadata_entries me
        LEFT JOIN temporal_info ti ON me.id = ti.metadata_id
        ORDER BY me.file_name;
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "文件名\t\t\t开始时间\t\t\t结束时间\t\t\t时间分辨率(秒)\t时间单位\t历法\t\t规律性" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* file_name = (const char*)sqlite3_column_text(stmt, 0);
            const char* start_time = (const char*)sqlite3_column_text(stmt, 1);
            const char* end_time = (const char*)sqlite3_column_text(stmt, 2);
            double temporal_resolution = sqlite3_column_double(stmt, 3);
            const char* time_units = (const char*)sqlite3_column_text(stmt, 4);
            const char* calendar = (const char*)sqlite3_column_text(stmt, 5);
            int is_regular = sqlite3_column_int(stmt, 6);
            
            std::cout << (file_name ? file_name : "NULL") << "\t\t"
                     << (start_time ? start_time : "NULL") << "\t\t"
                     << (end_time ? end_time : "NULL") << "\t\t"
                     << temporal_resolution << "\t\t\t"
                     << (time_units ? time_units : "NULL") << "\t\t"
                     << (calendar ? calendar : "NULL") << "\t\t"
                     << (is_regular ? "是" : "否") << std::endl;
        }
    } else {
        std::cout << "查询时间信息失败: " << sqlite3_errmsg(db) << std::endl;
    }
    sqlite3_finalize(stmt);
}

int main() {
    std::cout << "=== 数据库结构检查工具 ===" << std::endl;
    
    // 检查数据库文件是否存在
    std::vector<std::string> dbPaths = {
        "test_integration_data/metadata_db/ocean_environment.db",
        "test_integration_data/metadata_db/topography_bathymetry.db", 
        "test_integration_data/metadata_db/boundary_lines.db",
        "test_integration_data/metadata_db/sonar_propagation.db"
    };
    
    for (const auto& dbPath : dbPaths) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "检查数据库: " << dbPath << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        if (!std::filesystem::exists(dbPath)) {
            std::cout << "数据库文件不存在: " << dbPath << std::endl;
            continue;
        }
        
        DatabaseChecker checker;
        if (checker.openDatabase(dbPath)) {
            checker.checkDatabase();
            
            // 如果是海洋环境数据库，检查分类信息
            if (dbPath.find("ocean_environment") != std::string::npos) {
                checkVariableClassification(checker.db);
                checkTimeFormat(checker.db);
            }
        }
    }
    
    return 0;
} 