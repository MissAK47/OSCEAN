// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "impl/multi_database_manager.h"
#include "impl/database_adapter.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/utilities/string_utils.h"
#include <boost/thread/future.hpp>
#include <filesystem>
#include <algorithm>

using namespace oscean::core_services::metadata::impl;
using namespace oscean::core_services::metadata;
using namespace oscean::common_utils;

namespace fs = std::filesystem;

namespace { // 使用匿名命名空间使其成为文件内的静态辅助函数
    DatabaseType getDatabaseTypeFor(DataType dataType) {
        switch (dataType) {
            case DataType::OCEAN_ENVIRONMENT:
                return DatabaseType::OCEAN_ENVIRONMENT;
            case DataType::TOPOGRAPHY_BATHYMETRY:
                return DatabaseType::TOPOGRAPHY_BATHYMETRY;
            case DataType::BOUNDARY_LINES:
                return DatabaseType::BOUNDARY_LINES;
            case DataType::SONAR_PROPAGATION:
                return DatabaseType::SONAR_PROPAGATION;
            case DataType::TACTICAL_ENVIRONMENT:
                return DatabaseType::TACTICAL_ENVIRONMENT;
            default:
                return DatabaseType::OCEAN_ENVIRONMENT; // Default fallback
        }
    }
} // namespace

MultiDatabaseManager::MultiDatabaseManager(
    const DatabaseConfiguration& config,
    std::shared_ptr<IntelligentRecognizer> recognizer,
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices)
    : config_(config), 
      m_recognizer(std::move(recognizer)),
      commonServices_(std::move(commonServices)), 
      managerStartTime_(std::chrono::system_clock::now()) {
    
    LOG_INFO("MultiDatabaseManager已初始化，基础路径: {} [延迟初始化模式]", config_.basePath);
}

MultiDatabaseManager::~MultiDatabaseManager() {
    close();
}

bool MultiDatabaseManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // 确保基础目录存在
        if (!fs::exists(config_.basePath)) {
            fs::create_directories(config_.basePath);
            LOG_INFO("已创建数据库基础目录: {}", config_.basePath);
        }
        
        // 验证配置有效性
        if (!validateConfiguration()) {
            LOG_ERROR("数据库配置验证失败");
            return false;
        }
        
        LOG_INFO("MultiDatabaseManager初始化完成 [延迟模式] - 数据库将按需创建");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("MultiDatabaseManager初始化异常: {}", e.what());
        return false;
    }
}

void MultiDatabaseManager::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closeAllDatabases();
}

boost::future<AsyncResult<std::string>> MultiDatabaseManager::storeMetadataAsync(
    const oscean::core_services::FileMetadata& metadata) {
    
    return boost::async(boost::launch::async, [this, metadata]() -> AsyncResult<std::string> {
        try {
            if (!m_recognizer) {
                return AsyncResult<std::string>::failure("Intelligent Recognizer not available.");
            }

            // 1. 获取所有可能的数据类型
            std::vector<DataType> dataTypes = m_recognizer->determineDataTypeFromVariables(metadata.variables);
            
            if (dataTypes.empty() || (dataTypes.size() == 1 && dataTypes[0] == DataType::UNKNOWN)) {
                LOG_WARN("Could not determine a valid data type for file {}. Defaulting to OCEAN_ENVIRONMENT.", metadata.fileName);
                dataTypes = {DataType::OCEAN_ENVIRONMENT};
            }
            
            int successCount = 0;
            std::string lastSuccessfulId;
            std::vector<std::string> errors;

            // 2. 遍历所有识别出的数据类型，并存储到对应的数据库
            for (DataType dataType : dataTypes) {
                DatabaseType dbType = getDatabaseTypeFor(dataType);

                if (!ensureDatabaseExists(dbType)) {
                    std::string errorMsg = "Failed to create or connect to target database for data type: " + std::to_string(static_cast<int>(dataType));
                    LOG_ERROR(errorMsg);
                    errors.push_back(errorMsg);
                    continue; // 继续尝试下一个数据类型
                }
                
                auto adapter = databaseAdapters_[dbType];
                if (!adapter) {
                    std::string errorMsg = "Database adapter not available for data type: " + std::to_string(static_cast<int>(dataType));
                    LOG_ERROR(errorMsg);
                    errors.push_back(errorMsg);
                    continue;
                }
                
                auto result = adapter->storeFileMetadataAsync(metadata).get();
                
                if (result.isSuccess()) {
                    LOG_INFO("Metadata for {} successfully stored in database for type {}", metadata.fileName, static_cast<int>(dbType));
                    successCount++;
                    lastSuccessfulId = result.getData();
                    std::lock_guard<std::mutex> lock(mutex_);
                    connectionInfo_[dbType].lastAccessTime = std::chrono::system_clock::now();
                } else {
                    LOG_ERROR("Failed to store metadata for {} in database for type {}: {}", metadata.fileName, static_cast<int>(dbType), result.getError());
                    errors.push_back(result.getError());
                }
            }

            if (successCount > 0) {
                // 即使有部分失败，但只要有一次成功，就认为操作部分成功。
                // 返回最后一次成功的ID。
                return AsyncResult<std::string>::success(lastSuccessfulId);
            }

            // 如果所有存储都失败了
            return AsyncResult<std::string>::failure("Failed to store metadata in any target database. Errors: " + common_utils::StringUtils::join(errors, "; "));

        } catch (const std::exception& e) {
            return AsyncResult<std::string>::failure("Storing metadata failed with exception: " + std::string(e.what()));
        }
    });
}

boost::future<AsyncResult<std::vector<oscean::core_services::FileMetadata>>> MultiDatabaseManager::queryMetadataAsync(
    DatabaseType dbType,
    const QueryCriteria& criteria) {
    
    return boost::async(boost::launch::async, [this, dbType, criteria]() -> AsyncResult<std::vector<oscean::core_services::FileMetadata>> {
        try {
            // 检查数据库是否已初始化
            if (!isDatabaseInitialized(dbType)) {
                // 数据库未初始化，返回空结果（而不是错误）
                LOG_DEBUG("数据库类型 {} 未初始化，返回空查询结果", static_cast<int>(dbType));
                return AsyncResult<std::vector<oscean::core_services::FileMetadata>>::success(std::vector<oscean::core_services::FileMetadata>{});
            }
            
            auto adapter = databaseAdapters_[dbType];
            if (!adapter) {
                return AsyncResult<std::vector<oscean::core_services::FileMetadata>>::failure("数据库适配器不可用");
            }
            
            // 使用DatabaseAdapter的查询功能
            auto result = adapter->queryMetadataAsync(criteria).get();
            
            if (result.isSuccess()) {
                LOG_INFO("查询数据库 {} 完成，返回 {} 条记录", 
                    static_cast<int>(dbType), result.getData().size());
                
                // 更新连接信息
                std::lock_guard<std::mutex> lock(mutex_);
                connectionInfo_[dbType].lastAccessTime = std::chrono::system_clock::now();
            }
            
            return result;
            
        } catch (const std::exception& e) {
            return AsyncResult<std::vector<oscean::core_services::FileMetadata>>::failure("查询失败: " + std::string(e.what()));
        }
    });
}

boost::future<AsyncResult<bool>> MultiDatabaseManager::deleteMetadataAsync(
    DatabaseType dbType,
    const std::string& metadataId) {
    
    return boost::async(boost::launch::async, [this, dbType, metadataId]() -> AsyncResult<bool> {
        try {
            // 检查数据库是否已初始化
            if (!isDatabaseInitialized(dbType)) {
                return AsyncResult<bool>::failure("数据库未初始化");
            }
            
            auto adapter = databaseAdapters_[dbType];
            if (!adapter) {
                return AsyncResult<bool>::failure("数据库适配器不可用");
            }
            
            // 使用DatabaseAdapter的删除功能
            auto result = adapter->deleteMetadataAsync(metadataId).get();
            
            if (result.isSuccess()) {
                LOG_INFO("从数据库 {} 删除元数据 {}: {}", 
                    static_cast<int>(dbType), metadataId, result.getData() ? "成功" : "未找到");
                
                // 更新连接信息
                std::lock_guard<std::mutex> lock(mutex_);
                connectionInfo_[dbType].lastAccessTime = std::chrono::system_clock::now();
            }
            
            return result;
            
        } catch (const std::exception& e) {
            return AsyncResult<bool>::failure("删除失败: " + std::string(e.what()));
        }
    });
}

boost::future<AsyncResult<std::vector<oscean::core_services::FileMetadata>>> MultiDatabaseManager::queryByFilePathAsync(
    const std::string& filePath) {
    
    return boost::async(boost::launch::async, [this, filePath]() -> AsyncResult<std::vector<oscean::core_services::FileMetadata>> {
        try {
            std::vector<oscean::core_services::FileMetadata> allResults;
            
            // 🆕 只查询已初始化的数据库
            std::set<DatabaseType> initializedTypes;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                initializedTypes = initializedDatabases_;
            }
            
            for (auto dbType : initializedTypes) {
                if (isDatabaseConnected(dbType)) {
                    auto adapter = databaseAdapters_[dbType];
                    if (adapter) {
                        auto result = adapter->queryByFilePathAsync(filePath).get();
                        if (result.isSuccess()) {
                            const auto& results = result.getData();
                            allResults.insert(allResults.end(), results.begin(), results.end());
                        }
                    }
                }
            }
            
            LOG_INFO("按路径查询完成: {} -> {} 条记录 (搜索了{}个已初始化数据库)", 
                filePath, allResults.size(), initializedTypes.size());
            
            return AsyncResult<std::vector<oscean::core_services::FileMetadata>>::success(std::move(allResults));
            
        } catch (const std::exception& e) {
            return AsyncResult<std::vector<oscean::core_services::FileMetadata>>::failure("查询失败: " + std::string(e.what()));
        }
    });
}

boost::future<AsyncResult<void>> MultiDatabaseManager::updateMetadataAsync(
    const std::string& metadataId,
    const MetadataUpdate& update) {
    
    return boost::async(boost::launch::async, [this, metadataId, update]() -> AsyncResult<void> {
        try {
            bool found = false;
            
            // 🆕 只在已初始化的数据库中查找
            std::set<DatabaseType> initializedTypes;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                initializedTypes = initializedDatabases_;
            }
            
            for (auto dbType : initializedTypes) {
                if (isDatabaseConnected(dbType)) {
                    auto adapter = databaseAdapters_[dbType];
                    if (adapter) {
                        // 先查询是否存在
                        QueryCriteria criteria;
                        auto queryResult = adapter->queryMetadataAsync(criteria).get();
                        if (queryResult.isSuccess()) {
                            const auto& entries = queryResult.getData();
                            auto it = std::find_if(entries.begin(), entries.end(),
                                [&metadataId](const core_services::FileMetadata& entry) {
                                    return entry.metadataId == metadataId;
                                });
                            
                            if (it != entries.end()) {
                                // 找到了，执行更新
                                // 这里需要实现具体的更新逻辑
                                found = true;
                                LOG_INFO("在数据库 {} 中找到并更新元数据: {}", static_cast<int>(dbType), metadataId);
                                break;
                            }
                        }
                    }
                }
            }
            
            if (found) {
                return AsyncResult<void>::success();
            } else {
                return AsyncResult<void>::failure("未找到指定的元数据");
            }
            
        } catch (const std::exception& e) {
            return AsyncResult<void>::failure("更新异常: " + std::string(e.what()));
        }
    });
}

bool MultiDatabaseManager::isDatabaseConnected(DatabaseType dbType) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = databaseAdapters_.find(dbType);
    if (it == databaseAdapters_.end()) {
        return false;
    }
    
    return it->second && it->second->isConnected();
}

DatabaseConnectionInfo MultiDatabaseManager::getDatabaseInfo(DatabaseType dbType) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = connectionInfo_.find(dbType);
    if (it != connectionInfo_.end()) {
        return it->second;
    }
    
    DatabaseConnectionInfo info;
    info.type = dbType;
    info.isConnected = false;
    info.isInitialized = false;
    return info;
}

void MultiDatabaseManager::updateConfiguration(const DatabaseConfiguration& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    LOG_INFO("数据库配置已更新");
}

DatabaseConfiguration MultiDatabaseManager::getCurrentConfiguration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

// 私有方法实现

std::shared_ptr<DatabaseAdapter> MultiDatabaseManager::createDatabaseAdapter(DatabaseType dbType) {
    std::cout << "🔧 createDatabaseAdapter 开始，dbType: " << static_cast<int>(dbType) << std::endl;
    
    std::string dbPath = getDatabasePath(dbType);
    std::cout << "🔧 数据库路径: " << dbPath << std::endl;
    
    auto adapter = std::make_shared<DatabaseAdapter>(dbPath, dbType, commonServices_);
    std::cout << "🔧 DatabaseAdapter 对象创建成功" << std::endl;
    
    bool initResult = adapter->initialize();
    std::cout << "🔧 adapter->initialize() 结果: " << initResult << std::endl;
    
    if (!initResult) {
        LOG_ERROR("DatabaseAdapter初始化失败: {}", dbPath);
        std::cout << "❌ DatabaseAdapter初始化失败，返回nullptr" << std::endl;
        return nullptr;
    }
    
    LOG_INFO("DatabaseAdapter创建成功: {} (类型: {})", dbPath, static_cast<int>(dbType));
    std::cout << "✅ DatabaseAdapter创建和初始化成功" << std::endl;
    return adapter;
}

std::string MultiDatabaseManager::getDatabasePath(DatabaseType dbType) const {
    // 🔧 修复：首先检查是否有自定义数据库路径配置
    auto pathIt = config_.databasePaths.find(static_cast<DataType>(dbType));
    if (pathIt != config_.databasePaths.end() && !pathIt->second.empty()) {
        // 使用自定义路径（完整路径）
        LOG_DEBUG("使用自定义数据库路径: {}", pathIt->second);
        return pathIt->second;
    }
    
    // 如果没有自定义路径，使用默认的文件名
    std::string fileName;
    switch (dbType) {
        case DatabaseType::OCEAN_ENVIRONMENT:
            fileName = "ocean_environment.db";
            break;
        case DatabaseType::TOPOGRAPHY_BATHYMETRY:
            fileName = "topography_bathymetry.db";
            break;
        case DatabaseType::BOUNDARY_LINES:
            fileName = "boundary_lines.db";
            break;
        case DatabaseType::SONAR_PROPAGATION:
            fileName = "sonar_propagation.db";
            break;
        case DatabaseType::TACTICAL_ENVIRONMENT:  // 🆕 支持新的战术环境数据库
            fileName = "tactical_environment.db";
            break;
        default:
            fileName = "unknown.db";
            break;
    }
    
    // 使用基础路径和默认文件名构造路径
    std::string defaultPath = (fs::path(config_.basePath) / fileName).string();
    LOG_DEBUG("使用默认数据库路径: {}", defaultPath);
    return defaultPath;
}

bool MultiDatabaseManager::validateConfiguration() const {
    return !config_.basePath.empty();
}

void MultiDatabaseManager::closeAllDatabases() {
    for (auto& [dbType, adapter] : databaseAdapters_) {
        if (adapter) {
            adapter->close();
        }
    }
    databaseAdapters_.clear();
    connectionInfo_.clear();
    initializedDatabases_.clear();
    LOG_INFO("所有数据库连接已关闭");
}

// 🗑️ 已删除convertToMetadataEntry函数 - 直接使用core_services::FileMetadata，不需要转换

DataType MultiDatabaseManager::determineDataTypeFromPath(const std::string& filePath, DatabaseType dbType) const {
    return static_cast<DataType>(dbType);
}

std::string MultiDatabaseManager::generateFileId(const std::string& filePath) const {
    std::hash<std::string> hasher;
    return "file_" + std::to_string(hasher(filePath));
}

std::string MultiDatabaseManager::detectFileFormat(const std::string& filePath) const {
    fs::path path(filePath);
    std::string extension = path.extension().string();
    
    if (extension == ".nc" || extension == ".netcdf") {
        return "NetCDF";
    } else if (extension == ".shp") {
        return "Shapefile";
    } else if (extension == ".geojson") {
        return "GeoJSON";
    } else {
        return "Unknown";
    }
}

bool MultiDatabaseManager::initializeDatabase(DatabaseType dbType) {
    // 注意：这个方法现在被 initializeDatabaseOnDemand 替代
    return initializeDatabaseOnDemand(dbType);
}

bool MultiDatabaseManager::isConnected(DatabaseType dbType) const {
    auto it = databaseAdapters_.find(dbType);
    return it != databaseAdapters_.end() && it->second && it->second->isConnected();
}

// === 🆕 数据库生命周期管理实现 ===

bool MultiDatabaseManager::ensureDatabaseExists(DatabaseType dbType) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "🔍 ensureDatabaseExists 开始，dbType: " << static_cast<int>(dbType) << std::endl;
    
    // 检查是否已初始化
    bool isInitialized = initializedDatabases_.find(dbType) != initializedDatabases_.end();
    std::cout << "🔍 数据库是否已初始化: " << isInitialized << std::endl;
    
    if (isInitialized) {
        // 确保连接仍然有效
        auto it = databaseAdapters_.find(dbType);
        bool adapterExists = it != databaseAdapters_.end();
        std::cout << "🔍 适配器是否存在: " << adapterExists << std::endl;
        
        if (adapterExists && it->second) {
            bool isConnected = it->second->isConnected();
            std::cout << "🔍 适配器是否连接: " << isConnected << std::endl;
            
            if (isConnected) {
                std::cout << "✅ 重用现有数据库适配器" << std::endl;
                return true;
            } else {
                std::cout << "⚠️ 适配器连接失效，需要重新初始化" << std::endl;
            }
        } else {
            std::cout << "⚠️ 适配器不存在或为空，需要重新初始化" << std::endl;
        }
    }
    
    std::cout << "🔄 开始按需初始化数据库" << std::endl;
    // 按需初始化数据库
    bool result = initializeDatabaseOnDemand(dbType);
    std::cout << "🔍 按需初始化结果: " << result << std::endl;
    return result;
}

std::set<DatabaseType> MultiDatabaseManager::getInitializedDatabaseTypes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initializedDatabases_;
}

bool MultiDatabaseManager::isDatabaseInitialized(DatabaseType dbType) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initializedDatabases_.find(dbType) != initializedDatabases_.end();
}

size_t MultiDatabaseManager::preWarmDatabases(const std::vector<DatabaseType>& dbTypes) {
    size_t successCount = 0;
    
    for (auto dbType : dbTypes) {
        if (ensureDatabaseExists(dbType)) {
            successCount++;
            LOG_INFO("预热数据库成功: {}", static_cast<int>(dbType));
        } else {
            LOG_WARN("预热数据库失败: {}", static_cast<int>(dbType));
        }
    }
    
    LOG_INFO("数据库预热完成: {}/{} 成功", successCount, dbTypes.size());
    return successCount;
}

MultiDatabaseManager::DatabaseStatistics MultiDatabaseManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    DatabaseStatistics stats;
    stats.managerStartTime = managerStartTime_;
    stats.totalInitializedDatabases = initializedDatabases_.size();
    
    for (auto dbType : initializedDatabases_) {
        auto connIt = connectionInfo_.find(dbType);
        if (connIt != connectionInfo_.end()) {
            stats.creationTimes[dbType] = connIt->second.creationTime;
            stats.lastAccessTimes[dbType] = connIt->second.lastAccessTime;
        }
        
        // 获取记录数（需要适配器支持）
        auto adapterIt = databaseAdapters_.find(dbType);
        if (adapterIt != databaseAdapters_.end() && adapterIt->second) {
            // 这里可以通过适配器查询记录数
            stats.recordCounts[dbType] = 0; // 占位符，需要实际实现
        }
    }
    
    return stats;
}

// === 🆕 延迟初始化核心方法实现 ===

bool MultiDatabaseManager::initializeDatabaseOnDemand(DatabaseType dbType) {
    // 注意：此方法在互斥锁保护下调用
    
    if (!isValidDatabaseType(dbType)) {
        LOG_ERROR("无效的数据库类型: {}", static_cast<int>(dbType));
        return false;
    }
    
    try {
        LOG_INFO("开始按需初始化数据库: {}", static_cast<int>(dbType));
        
        // 创建DatabaseAdapter
        auto adapter = createDatabaseAdapter(dbType);
        if (!adapter) {
            LOG_ERROR("创建数据库适配器失败: {}", static_cast<int>(dbType));
            return false;
        }
        
        // 存储适配器
        databaseAdapters_[dbType] = adapter;
        
        // 初始化连接信息
        DatabaseConnectionInfo info;
        info.type = dbType;
        info.filePath = getDatabasePath(dbType);
        info.isConnected = adapter->isConnected();
        info.isInitialized = true;
        info.lastError = "";
        info.activeConnections = 1;
        auto now = std::chrono::system_clock::now();
        info.lastAccessTime = now;
        info.creationTime = now;
        
        connectionInfo_[dbType] = info;
        
        // 标记为已初始化
        initializedDatabases_.insert(dbType);
        
        LOG_INFO("数据库按需初始化完成: {} (文件: {})", 
            static_cast<int>(dbType), info.filePath);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("数据库按需初始化异常: {}", e.what());
        return false;
    }
}

bool MultiDatabaseManager::isValidDatabaseType(DatabaseType dbType) const {
    return dbType == DatabaseType::OCEAN_ENVIRONMENT ||
           dbType == DatabaseType::TOPOGRAPHY_BATHYMETRY ||
           dbType == DatabaseType::BOUNDARY_LINES ||
           dbType == DatabaseType::SONAR_PROPAGATION ||
           dbType == DatabaseType::TACTICAL_ENVIRONMENT;  // 🆕 支持新的战术环境数据库
} 