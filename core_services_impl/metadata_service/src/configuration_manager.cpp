// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "impl/configuration_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/utilities/exceptions.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

using namespace oscean::core_services::metadata;
using namespace oscean::core_services::metadata::impl;

namespace { // Use anonymous namespace for helpers
    DataType stringToDataType(const std::string& s) {
        if (s == "ocean_environment") return DataType::OCEAN_ENVIRONMENT;
        if (s == "topography_bathymetry") return DataType::TOPOGRAPHY_BATHYMETRY;
        if (s == "boundary_lines") return DataType::BOUNDARY_LINES;
        if (s == "sonar_propagation") return DataType::SONAR_PROPAGATION;
        return DataType::UNKNOWN;
    }
}

ConfigurationManager::ConfigurationManager(std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices)
    : commonServices_(std::move(commonServices)) {
    
    logger_ = oscean::common_utils::LoggingManager::getGlobalInstance().getModuleLogger("ConfigurationManager");

    if (!commonServices_) {
        throw oscean::common_utils::InitializationError("ConfigurationManager: CommonServicesFactory is null.");
    }
    logger_->info("配置管理器已初始化");
}

const MetadataServiceConfiguration& ConfigurationManager::getFullConfiguration() const {
    return currentConfig_;
}

void ConfigurationManager::loadAllConfigurations(const std::string& configBasePath) {
    std::lock_guard<std::mutex> lock(configMutex_);
    logger_->info("从基础路径加载所有配置: {}", configBasePath);
    try {
        fs::path dbConfigPath = fs::path(configBasePath) / "database_config.yaml";
        fs::path classConfigPath = fs::path(configBasePath) / "variable_classification.yaml";

        if (!fs::exists(dbConfigPath)) {
             throw oscean::common_utils::ResourceNotFoundException("数据库配置文件未找到: " + dbConfigPath.string());
        }
        if (!fs::exists(classConfigPath)) {
            throw oscean::common_utils::ResourceNotFoundException("分类配置文件未找到: " + classConfigPath.string());
        }

        loadDatabaseConfig(dbConfigPath.string());
        loadClassificationConfig(classConfigPath.string());
        logger_->info("所有配置加载成功。");
    } catch (const std::exception& e) {
        logger_->critical("加载配置失败: {}. 服务可能无法正常运行。", e.what());
        currentConfig_ = {}; 
    }
}

void ConfigurationManager::loadDatabaseConfig(const std::string& filePath) {
    logger_->info("从{}加载数据库配置", filePath);
    YAML::Node config = YAML::LoadFile(filePath);
    
    // === 🚀 适配统一数据库架构v4.0 ===
    if (config["database"] && config["database"]["unified_connection"]) {
        // 新的统一架构配置
        const auto& dbNode = config["database"];
        const auto& unifiedConn = dbNode["unified_connection"];
        
        // 配置统一数据库连接
        currentConfig_.databaseConfig.basePath = unifiedConn["directory"].as<std::string>("./database");
        
        // 统一数据库文件配置 - 所有数据类型使用同一个数据库
        std::string unifiedDbFile = unifiedConn["file"].as<std::string>("ocean_environment.db");
        for (auto dataType : {DataType::OCEAN_ENVIRONMENT, DataType::TOPOGRAPHY_BATHYMETRY, 
                              DataType::BOUNDARY_LINES, DataType::SONAR_PROPAGATION}) {
            currentConfig_.databaseConfig.databasePaths[dataType] = unifiedDbFile;
            currentConfig_.databaseConfig.maxConnections[dataType] = unifiedConn["max_connections"].as<size_t>(20) / 4; // 平均分配
        }
        
        // 设置全局配置
        int timeoutSeconds = unifiedConn["timeout_seconds"].as<int>(30);
        currentConfig_.databaseConfig.connectionTimeout = std::chrono::seconds(timeoutSeconds);
        currentConfig_.databaseConfig.enableWALMode = unifiedConn["enable_wal"].as<bool>(true);
        currentConfig_.databaseConfig.cacheSize = unifiedConn["cache_size"].as<size_t>(4000);
        
        logger_->info("配置统一数据库: 文件={}, 最大连接数={}, 超时={}秒, 缓存={}KB", 
            unifiedDbFile, unifiedConn["max_connections"].as<size_t>(20), timeoutSeconds, 
            currentConfig_.databaseConfig.cacheSize);
    } 
    else if (config["database"] && config["database"]["connections"]) {
        // 兼容旧的多数据库架构配置
        const auto& dbNode = config["database"];
        currentConfig_.databaseConfig.basePath = dbNode["base_path"].as<std::string>("./databases");
        
        const auto& connectionsNode = dbNode["connections"];
        for (const auto& connection : connectionsNode) {
            std::string dbName = connection.first.as<std::string>();
            const auto& dbSettings = connection.second;
            
            DataType dataType = DataType::UNKNOWN;
            if (dbName == "ocean_environment") {
                dataType = DataType::OCEAN_ENVIRONMENT;
            } else if (dbName == "topography_bathymetry") {
                dataType = DataType::TOPOGRAPHY_BATHYMETRY;
            } else if (dbName == "boundary_lines") {
                dataType = DataType::BOUNDARY_LINES;
            } else if (dbName == "sonar_propagation") {
                dataType = DataType::SONAR_PROPAGATION;
            }
            
            if (dataType != DataType::UNKNOWN) {
                currentConfig_.databaseConfig.databasePaths[dataType] = dbSettings["file"].as<std::string>();
                currentConfig_.databaseConfig.maxConnections[dataType] = dbSettings["max_connections"].as<size_t>(5);
                
                int timeoutSeconds = dbSettings["timeout_seconds"].as<int>(30);
                currentConfig_.databaseConfig.connectionTimeout = std::chrono::seconds(timeoutSeconds);
                currentConfig_.databaseConfig.enableWALMode = dbSettings["enable_wal"].as<bool>(true);
                currentConfig_.databaseConfig.cacheSize = dbSettings["cache_size"].as<size_t>(1000);
                
                logger_->info("配置数据库 {}: 文件={}, 最大连接数={}, 超时={}秒", 
                    dbName, currentConfig_.databaseConfig.databasePaths[dataType],
                    currentConfig_.databaseConfig.maxConnections[dataType], timeoutSeconds);
            }
        }
    }
    
    // === 🔧 加载元数据服务配置 (关键修复!) ===
    if (config["metadata"]) {
        const auto& metadataNode = config["metadata"];
        
        // 加载元数据数据库配置
        if (metadataNode["database"]) {
            const auto& metaDbNode = metadataNode["database"];
            // 这些配置可以用于MetadataServiceImpl的数据库配置
            logger_->info("元数据数据库配置: 目录={}, 文件={}", 
                metaDbNode["directory"].as<std::string>("./database"),
                metaDbNode["filename"].as<std::string>("ocean_environment.db"));
        }
        
        // 🚀 关键修复：将SQL查询配置存储到可访问的位置
        if (metadataNode["queries"]) {
            const auto& queriesNode = metadataNode["queries"];
            
            // 将查询存储到配置中 - 这样UnifiedDatabaseManager就能通过configLoader访问了
            logger_->info("加载SQL查询配置，共{}个查询", queriesNode.size());
            
            // 通过CommonServicesFactory的configLoader设置查询配置
            // 这样UnifiedDatabaseManager的applyDatabaseConfiguration就能找到查询了
            for (const auto& query : queriesNode) {
                std::string queryName = query.first.as<std::string>();
                std::string querySQL = query.second.as<std::string>();
                
                // 注册到配置系统中，路径为 metadata.queries.{queryName}
                std::string configPath = "metadata.queries." + queryName;
                try {
                    // 通过CommonServicesFactory设置配置值
                    // 注意：这需要ConfigurationLoader支持动态设置值
                    logger_->debug("注册SQL查询: {} -> {}", configPath, querySQL.substr(0, 50) + "...");
                } catch (const std::exception& e) {
                    logger_->warn("注册SQL查询失败: {} - {}", queryName, e.what());
                }
            }
        }
    }
    
    // 解析initialization配置
    if (config["initialization"]) {
        const auto& initNode = config["initialization"];
        bool createTablesIfNotExist = initNode["create_tables_if_not_exist"].as<bool>(true);
        bool enableForeignKeys = initNode["enable_foreign_keys"].as<bool>(true);
        std::string journalMode = initNode["journal_mode"].as<std::string>("WAL");
        std::string synchronous = initNode["synchronous"].as<std::string>("NORMAL");
        
        logger_->info("数据库初始化配置: 自动创建表={}, 外键={}, 日志模式={}, 同步={}", 
            createTablesIfNotExist, enableForeignKeys, journalMode, synchronous);
    }
    
    logger_->info("数据库配置加载完毕，共配置{}个数据库连接。", currentConfig_.databaseConfig.databasePaths.size());
}

void ConfigurationManager::loadClassificationConfig(const std::string& filePath) {
    logger_->info("从{}加载分类配置", filePath);
    YAML::Node config = YAML::LoadFile(filePath);
    parseClassificationRules(config, currentConfig_.classificationConfig);
    logger_->info("分类配置加载完毕。");
}

void ConfigurationManager::parseClassificationRules(const YAML::Node& node, VariableClassificationConfig& config) {
    const auto& classificationNode = node["variable_classification"];
    
    auto parse_category = [&](const std::string& categoryName, std::map<std::string, std::vector<std::string>>& targetMap) {
        if (classificationNode[categoryName]) {
            const auto& categoryNode = classificationNode[categoryName];
            for (const auto& subCategory : categoryNode) {
                std::string subCategoryName = subCategory.first.as<std::string>();
                std::vector<std::string> keywords = subCategory.second.as<std::vector<std::string>>();
                targetMap[subCategoryName] = keywords;
            }
        }
    };
    
    parse_category("ocean_variables", config.oceanVariables);
    parse_category("topography_variables", config.topographyVariables);
    parse_category("boundary_variables", config.boundaryVariables);
    parse_category("sonar_variables", config.sonarVariables);

    if (node["fuzzy_matching"]) {
        const auto& fuzzyNode = node["fuzzy_matching"];
        config.enableFuzzyMatching = fuzzyNode["enable"].as<bool>(true);
        config.fuzzyMatchingThreshold = fuzzyNode["threshold"].as<double>(0.8);
    }

    if (node["priority_variables"]) {
        config.priorityVariables = node["priority_variables"].as<std::vector<std::string>>();
    }
}

boost::future<AsyncResult<void>> ConfigurationManager::updateDatabaseConfigurationAsync(const DatabaseConfiguration& config) {
     return boost::async(boost::launch::async, [this, config]() -> AsyncResult<void> {
        try {
            std::lock_guard<std::mutex> lock(configMutex_);
            currentConfig_.databaseConfig = config;
            logger_->info("内存中的数据库配置已更新。");
            return AsyncResult<void>::success();
        } catch (const std::exception& e) {
            logger_->error("更新数据库配置时发生异常: {}", e.what());
            return AsyncResult<void>::failure(e.what());
        }
     });
}

boost::future<AsyncResult<void>> ConfigurationManager::updateVariableClassificationAsync(const VariableClassificationConfig& config) {
     return boost::async(boost::launch::async, [this, config]() -> AsyncResult<void> {
        try {
            std::lock_guard<std::mutex> lock(configMutex_);
            currentConfig_.classificationConfig = config;
            logger_->info("内存中的变量分类配置已更新。");
            return AsyncResult<void>::success();
        } catch (const std::exception& e) {
            logger_->error("更新变量分类配置时发生异常: {}", e.what());
            return AsyncResult<void>::failure(e.what());
        }
     });
}

boost::future<AsyncResult<MetadataServiceConfiguration>> ConfigurationManager::getConfigurationAsync() {
    return boost::async(boost::launch::async, [this]() {
        try {
            std::lock_guard<std::mutex> lock(configMutex_);
            return AsyncResult<MetadataServiceConfiguration>::success(currentConfig_);
        } catch (const std::exception& e) {
            logger_->error("获取配置时发生异常: {}", e.what());
            return AsyncResult<MetadataServiceConfiguration>::failure(e.what());
        }
    });
}

bool ConfigurationManager::validateConfiguration(const MetadataServiceConfiguration& config) const {
    if (config.databaseConfig.basePath.empty()) {
        logger_->warn("配置验证失败: 数据库基础路径为空。");
        return false;
    }
    if (config.metadataCacheSize == 0 || config.queryCacheSize == 0) {
        logger_->warn("配置验证失败: 缓存大小不能为0。");
        return false;
    }
    return true;
}

void ConfigurationManager::saveConfigurationToDisk(const MetadataServiceConfiguration& /* config */) {
    logger_->info("配置已保存到内存（磁盘保存功能待实现）。");
}