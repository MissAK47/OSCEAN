#include <iostream>
#include <sqlite3.h>
#include <string>
#include <iomanip>

int callback(void* data, int argc, char** argv, char** azColName) {
    for (int i = 0; i < argc; i++) {
        std::cout << azColName[i] << " = " << (argv[i] ? argv[i] : "NULL") << " | ";
    }
    std::cout << std::endl;
    return 0;
}

void printTableData(sqlite3* db, const std::string& tableName) {
    std::cout << "\n=== " << tableName << " 表数据 ===" << std::endl;
    
    std::string sql = "SELECT * FROM " + tableName;
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "查询失败: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    // 打印列名
    int columnCount = sqlite3_column_count(stmt);
    for (int i = 0; i < columnCount; i++) {
        std::cout << std::setw(20) << sqlite3_column_name(stmt, i);
    }
    std::cout << std::endl;
    
    // 打印分隔线
    for (int i = 0; i < columnCount; i++) {
        std::cout << std::setw(20) << "--------------------";
    }
    std::cout << std::endl;
    
    // 打印数据
    int rowCount = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        for (int i = 0; i < columnCount; i++) {
            const char* value = (const char*)sqlite3_column_text(stmt, i);
            std::cout << std::setw(20) << (value ? value : "NULL");
        }
        std::cout << std::endl;
        rowCount++;
    }
    
    std::cout << "总计: " << rowCount << " 条记录" << std::endl;
    sqlite3_finalize(stmt);
}

void printTemporalResolutionInfo(sqlite3* db) {
    std::cout << "\n=== 时间分辨率详细信息 ===" << std::endl;
    
    const char* sql = R"(
        SELECT 
            m.file_path,
            t.start_time,
            t.end_time,
            t.temporal_resolution_seconds,
            t.temporal_resolution_type,
            t.time_units
        FROM metadata_entries m
        JOIN temporal_info t ON m.id = t.metadata_id
        ORDER BY m.file_path
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "查询失败: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    std::cout << std::setw(30) << "文件路径" 
              << std::setw(20) << "开始时间"
              << std::setw(20) << "结束时间"
              << std::setw(15) << "分辨率(秒)"
              << std::setw(15) << "分辨率类型"
              << std::setw(15) << "时间单位" << std::endl;
    
    for (int i = 0; i < 6; i++) {
        std::cout << std::setw(30) << "------------------------------";
    }
    std::cout << std::endl;
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* filePath = (const char*)sqlite3_column_text(stmt, 0);
        const char* startTime = (const char*)sqlite3_column_text(stmt, 1);
        const char* endTime = (const char*)sqlite3_column_text(stmt, 2);
        const char* resolutionSeconds = (const char*)sqlite3_column_text(stmt, 3);
        const char* resolutionType = (const char*)sqlite3_column_text(stmt, 4);
        const char* timeUnits = (const char*)sqlite3_column_text(stmt, 5);
        
        // 提取文件名
        std::string fileName = filePath ? filePath : "NULL";
        size_t lastSlash = fileName.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            fileName = fileName.substr(lastSlash + 1);
        }
        
        std::cout << std::setw(30) << fileName
                  << std::setw(20) << (startTime ? startTime : "NULL")
                  << std::setw(20) << (endTime ? endTime : "NULL")
                  << std::setw(15) << (resolutionSeconds ? resolutionSeconds : "NULL")
                  << std::setw(15) << (resolutionType ? resolutionType : "NULL")
                  << std::setw(15) << (timeUnits ? timeUnits : "NULL") << std::endl;
    }
    
    sqlite3_finalize(stmt);
}

int main() {
    sqlite3* db;
    char* zErrMsg = 0;
    int rc;
    
    // 打开海洋环境数据库  
    rc = sqlite3_open("databases/ocean_environment.db", &db);
    
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    } else {
        std::cout << "✅ 海洋环境数据库打开成功" << std::endl;
    }
    
    // 查询表列表
    std::cout << "\n=== 数据库表列表 ===" << std::endl;
    const char* sqlTables = "SELECT name FROM sqlite_master WHERE type='table';";
    rc = sqlite3_exec(db, sqlTables, callback, 0, &zErrMsg);
    
    // 查询元数据记录总数
    std::cout << "\n=== 元数据记录总数 ===" << std::endl;
    const char* sql1 = "SELECT COUNT(*) as total_records FROM metadata_entries;";
    rc = sqlite3_exec(db, sql1, callback, 0, &zErrMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << zErrMsg << std::endl;
        sqlite3_free(zErrMsg);
    }
    
    // 查询metadata_entries表结构
    std::cout << "\n=== metadata_entries表结构 ===" << std::endl;
    const char* sqlSchema = "PRAGMA table_info(metadata_entries);";
    rc = sqlite3_exec(db, sqlSchema, callback, 0, &zErrMsg);
    
    // 查询所有元数据记录的基础信息
    std::cout << "\n=== 元数据记录详情 ===" << std::endl;
    const char* sql2 = "SELECT * FROM metadata_entries;";
    rc = sqlite3_exec(db, sql2, callback, 0, &zErrMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << zErrMsg << std::endl;
        sqlite3_free(zErrMsg);
    }
    
    // 查询variable_info表结构
    std::cout << "\n=== variable_info表结构 ===" << std::endl;
    const char* sqlVarSchema = "PRAGMA table_info(variable_info);";
    rc = sqlite3_exec(db, sqlVarSchema, callback, 0, &zErrMsg);
    
    // 查询变量信息
    std::cout << "\n=== 变量信息 ===" << std::endl;
    const char* sql3 = "SELECT * FROM variable_info ORDER BY id;";
    rc = sqlite3_exec(db, sql3, callback, 0, &zErrMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << zErrMsg << std::endl;
        sqlite3_free(zErrMsg);
    }
    
    // 查询空间信息
    std::cout << "\n=== 空间信息 ===" << std::endl;
    const char* sql4 = "SELECT * FROM spatial_info;";
    rc = sqlite3_exec(db, sql4, callback, 0, &zErrMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << zErrMsg << std::endl;
        sqlite3_free(zErrMsg);
    }
    
    // 查询时间信息
    std::cout << "\n=== 时间信息 ===" << std::endl;
    const char* sql5 = "SELECT * FROM temporal_info;";
    rc = sqlite3_exec(db, sql5, callback, 0, &zErrMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << zErrMsg << std::endl;
        sqlite3_free(zErrMsg);
    }
    
    // 查询属性信息
    std::cout << "\n=== 属性信息 ===" << std::endl;
    const char* sql6 = "SELECT * FROM attributes ORDER BY id;";
    rc = sqlite3_exec(db, sql6, callback, 0, &zErrMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << zErrMsg << std::endl;
        sqlite3_free(zErrMsg);
    }
    
    // 检查时间分辨率信息
    printTemporalResolutionInfo(db);
    
    // 检查temporal_info表的完整数据
    printTableData(db, "temporal_info");
    
    // 专门查询SP文件的时间信息
    std::cout << "\n=== SP文件时间分辨率详细信息 ===" << std::endl;
    const char* sqlSP = R"(
        SELECT 
            m.file_path,
            t.start_time,
            t.end_time,
            t.temporal_resolution_seconds,
            t.temporal_resolution_type,
            t.time_units,
            t.calendar
        FROM temporal_info t 
        JOIN metadata_entries m ON t.metadata_id = m.id 
        WHERE m.file_path LIKE '%sp_%'
        ORDER BY m.file_path;
    )";
    rc = sqlite3_exec(db, sqlSP, callback, 0, &zErrMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SP文件时间信息查询失败: " << zErrMsg << std::endl;
        sqlite3_free(zErrMsg);
    }
    
    sqlite3_close(db);
    std::cout << "\n🔍 数据库检查完成" << std::endl;
    return 0;
} 