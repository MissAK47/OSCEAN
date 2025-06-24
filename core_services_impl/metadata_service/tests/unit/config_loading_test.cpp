/**
 * @file config_loading_test.cpp
 * @brief 配置文件加载真实功能测试 - 简化版
 * @note 直接测试配置文件的解析功能，避免复杂依赖
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <iostream>

// 直接包含yaml-cpp和spdlog
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>

using namespace std;

/**
 * @brief 简化的配置文件加载测试
 */
class SimpleConfigLoadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        cout << "=== 简化配置文件测试初始化 ===" << endl;
        
        // 创建测试配置目录
        testConfigDir_ = filesystem::temp_directory_path() / "simple_config_test";
        filesystem::create_directories(testConfigDir_);
        cout << "测试目录: " << testConfigDir_ << endl;
        
        // 设置工作目录
        originalWorkingDir_ = filesystem::current_path();
        filesystem::current_path(testConfigDir_);
        
        // 创建config子目录
        filesystem::create_directories("config");
    }
    
    void TearDown() override {
        cout << "=== 清理测试环境 ===" << endl;
        
        // 恢复原始工作目录
        filesystem::current_path(originalWorkingDir_);
        
        // 清理测试目录
        error_code ec;
        filesystem::remove_all(testConfigDir_, ec);
        if (ec) {
            cout << "清理失败: " << ec.message() << endl;
        }
    }
    
    /**
     * @brief 创建简单的variable_classification.yaml测试文件
     */
    void createSimpleVariableConfig() {
        string configContent = 
R"(# 简化版变量分类配置
variable_classification:
  ocean_variables:
    temperature:
      - "temperature"
      - "temp"
      - "sst"
    salinity:
      - "salinity"
      - "sal"
      - "so"
    current:
      - "u"
      - "v"
      - "uo"
      - "vo"

# 模糊匹配配置
fuzzy_matching:
  enable: true
  threshold: 0.8
  max_suggestions: 3
)";
        
        ofstream file(testConfigDir_ / "config" / "variable_classification.yaml");
        file << configContent;
        file.close();
        
        cout << "创建简化版variable_classification.yaml" << endl;
    }
    
    /**
     * @brief 创建简单的database_config.yaml测试文件
     */
    void createSimpleDatabaseConfig() {
        string configContent = 
R"(# 简化版数据库配置
database:
  base_path: "./test_databases"
  connections:
    ocean_environment:
      file: "ocean_test.db"
      max_connections: 5
      timeout_seconds: 15
    topography_bathymetry:
      file: "topo_test.db"
      max_connections: 3
      timeout_seconds: 20
)";
        
        ofstream file(testConfigDir_ / "config" / "database_config.yaml");
        file << configContent;
        file.close();
        
        cout << "创建简化版database_config.yaml" << endl;
    }

protected:
    filesystem::path testConfigDir_;
    filesystem::path originalWorkingDir_;
};

/**
 * @brief 测试variable_classification.yaml文件的直接解析
 */
TEST_F(SimpleConfigLoadingTest, DirectVariableConfigParsing) {
    cout << "=== 测试变量分类配置文件直接解析 ===" << endl;
    
    // 创建测试配置文件
    createSimpleVariableConfig();
    
    // 直接使用yaml-cpp解析
    try {
        YAML::Node config = YAML::LoadFile("config/variable_classification.yaml");
        
        // 验证基本结构
        ASSERT_TRUE(config["variable_classification"]) << "缺少variable_classification节点";
        
        const auto& varClass = config["variable_classification"];
        ASSERT_TRUE(varClass["ocean_variables"]) << "缺少ocean_variables节点";
        
        const auto& oceanVars = varClass["ocean_variables"];
        ASSERT_TRUE(oceanVars["temperature"]) << "缺少temperature节点";
        ASSERT_TRUE(oceanVars["salinity"]) << "缺少salinity节点";
        ASSERT_TRUE(oceanVars["current"]) << "缺少current节点";
        
        // 验证temperature变量列表
        const auto& tempVars = oceanVars["temperature"];
        ASSERT_TRUE(tempVars.IsSequence()) << "temperature应该是一个序列";
        
        vector<string> expectedTempVars = {"temperature", "temp", "sst"};
        ASSERT_EQ(tempVars.size(), expectedTempVars.size()) << "temperature变量数量不匹配";
        
        for (size_t i = 0; i < expectedTempVars.size(); ++i) {
            EXPECT_EQ(tempVars[i].as<string>(), expectedTempVars[i]) 
                << "temperature变量 " << i << " 不匹配";
        }
        
        // 验证fuzzy_matching配置
        ASSERT_TRUE(config["fuzzy_matching"]) << "缺少fuzzy_matching节点";
        const auto& fuzzyConfig = config["fuzzy_matching"];
        EXPECT_TRUE(fuzzyConfig["enable"].as<bool>()) << "fuzzy_matching应该启用";
        EXPECT_EQ(fuzzyConfig["threshold"].as<double>(), 0.8) << "threshold不匹配";
        EXPECT_EQ(fuzzyConfig["max_suggestions"].as<int>(), 3) << "max_suggestions不匹配";
        
        cout << "✅ variable_classification.yaml解析成功" << endl;
        
    } catch (const YAML::Exception& e) {
        FAIL() << "YAML解析失败: " << e.what();
    } catch (const exception& e) {
        FAIL() << "解析异常: " << e.what();
    }
}

/**
 * @brief 测试database_config.yaml文件的直接解析
 */
TEST_F(SimpleConfigLoadingTest, DirectDatabaseConfigParsing) {
    cout << "=== 测试数据库配置文件直接解析 ===" << endl;
    
    // 创建测试配置文件
    createSimpleDatabaseConfig();
    
    // 直接使用yaml-cpp解析
    try {
        YAML::Node config = YAML::LoadFile("config/database_config.yaml");
        
        // 验证根节点
        ASSERT_TRUE(config["database"]) << "缺少database根节点";
        
        const auto& dbConfig = config["database"];
        ASSERT_TRUE(dbConfig["base_path"]) << "缺少base_path节点";
        ASSERT_TRUE(dbConfig["connections"]) << "缺少connections节点";
        
        // 验证基础路径
        EXPECT_EQ(dbConfig["base_path"].as<string>(), "./test_databases") 
            << "base_path不匹配";
        
        // 验证连接配置
        const auto& connections = dbConfig["connections"];
        ASSERT_TRUE(connections["ocean_environment"]) << "缺少ocean_environment连接";
        ASSERT_TRUE(connections["topography_bathymetry"]) << "缺少topography_bathymetry连接";
        
        // 验证ocean_environment配置
        const auto& oceanConn = connections["ocean_environment"];
        EXPECT_EQ(oceanConn["file"].as<string>(), "ocean_test.db") 
            << "ocean数据库文件名不匹配";
        EXPECT_EQ(oceanConn["max_connections"].as<int>(), 5) 
            << "ocean最大连接数不匹配";
        EXPECT_EQ(oceanConn["timeout_seconds"].as<int>(), 15) 
            << "ocean超时时间不匹配";
        
        // 验证topography_bathymetry配置
        const auto& topoConn = connections["topography_bathymetry"];
        EXPECT_EQ(topoConn["file"].as<string>(), "topo_test.db") 
            << "地形数据库文件名不匹配";
        EXPECT_EQ(topoConn["max_connections"].as<int>(), 3) 
            << "地形最大连接数不匹配";
        EXPECT_EQ(topoConn["timeout_seconds"].as<int>(), 20) 
            << "地形超时时间不匹配";
        
        cout << "✅ database_config.yaml解析成功" << endl;
        
    } catch (const YAML::Exception& e) {
        FAIL() << "YAML解析失败: " << e.what();
    } catch (const exception& e) {
        FAIL() << "解析异常: " << e.what();
    }
}

/**
 * @brief 测试配置文件不存在的处理
 */
TEST_F(SimpleConfigLoadingTest, ConfigFileNotFound) {
    cout << "=== 测试配置文件不存在处理 ===" << endl;
    
    // 尝试加载不存在的文件
    try {
        YAML::Node config = YAML::LoadFile("config/nonexistent.yaml");
        FAIL() << "应该抛出异常，因为文件不存在";
    } catch (const YAML::BadFile& e) {
        cout << "✅ 正确捕获了文件不存在异常: " << e.what() << endl;
        SUCCEED();
    } catch (const exception& e) {
        FAIL() << "捕获了错误类型的异常: " << e.what();
    }
}

/**
 * @brief 测试无效配置文件的处理
 */
TEST_F(SimpleConfigLoadingTest, InvalidConfigFile) {
    cout << "=== 测试无效配置文件处理 ===" << endl;
    
    // 创建无效的YAML文件
    ofstream invalidFile(testConfigDir_ / "config" / "invalid.yaml");
    invalidFile << "invalid: yaml: content:\n  - broken\n    structure";
    invalidFile.close();
    
    // 尝试解析无效文件
    try {
        YAML::Node config = YAML::LoadFile("config/invalid.yaml");
        FAIL() << "应该抛出解析异常";
    } catch (const YAML::ParserException& e) {
        cout << "✅ 正确捕获了解析异常: " << e.what() << endl;
        SUCCEED();
    } catch (const exception& e) {
        cout << "✅ 捕获了异常: " << e.what() << endl;
        SUCCEED();
    }
} 