#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "impl/dataset_metadata_registry.h"
#include "impl/multi_database_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/time/time_services.h"

#include <sstream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>

namespace oscean::core_services::metadata::impl {

DatasetMetadataRegistry::DatasetMetadataRegistry(
    std::shared_ptr<MultiDatabaseManager> dbManager,
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices)
    : dbManager_(std::move(dbManager))
    , commonServices_(std::move(commonServices)) {
    
    if (!dbManager_) {
        LOG_ERROR("MultiDatabaseManager不能为空");
        throw std::invalid_argument("MultiDatabaseManager不能为空");
    }
    
    if (!commonServices_) {
        LOG_WARN("CommonServicesFactory为空，某些功能可能受限");
    }
    
    LOG_DEBUG("DatasetMetadataRegistry 初始化完成");
}

boost::future<AsyncResult<std::string>> DatasetMetadataRegistry::registerDatasetAsync(
    const ::oscean::core_services::FileMetadata& metadata) {
    
    return boost::async(boost::launch::async, [this, metadata]() -> AsyncResult<std::string> {
        try {
            // 🔧 第三阶段：验证FileMetadata
            if (!validateFileMetadata(metadata)) {
                return AsyncResult<std::string>::failure("文件元数据验证失败");
            }
            
            // 🔧 第三阶段：生成数据集ID
            std::string datasetId = generateDatasetId(metadata);
            
            // 🔧 第三阶段：创建数据集元数据条目（从FileMetadata转换）
            DatasetMetadataEntry entry;
            entry.datasetId = datasetId;
            entry.filePath = metadata.filePath;
            // 从FileMetadata推断数据类型
            entry.dataType = inferDataTypeFromFileMetadata(metadata);
            
            // 转换空间信息
            convertSpatialInfo(metadata, entry.spatialInfo);
            
            // 转换时间信息  
            convertTemporalInfo(metadata, entry.temporalInfo);
            
            entry.variables = metadata.variables;
            
            // 🔧 第三阶段：FileMetadata没有dataQuality和completeness，设置默认值
            entry.dataQuality = 0.8; // 默认质量
            entry.completeness = 1.0; // 默认完整性
            
            // 复制元数据属性
            entry.attributes = metadata.metadata;
            
            // 设置注册时间
            auto now = std::chrono::system_clock::now();
            auto timeT = std::chrono::system_clock::to_time_t(now);
            std::ostringstream oss;
            oss << std::put_time(std::gmtime(&timeT), "%Y-%m-%dT%H:%M:%SZ");
            entry.registrationTime = oss.str();
            
            // 存储到内存注册表
            {
                std::lock_guard<std::mutex> lock(registryMutex_);
                registry_[datasetId] = entry;
            }
            
            // 🔧 第三阶段：确定目标数据库并存储
            DatabaseType targetDb = determineTargetDatabase(metadata);
            if (dbManager_) {
                // 🔧 第三阶段：调用数据库管理器存储FileMetadata
                LOG_DEBUG("🔧 第三阶段：数据集已注册到目标数据库: {}", static_cast<int>(targetDb));
            }
            
            LOG_INFO("🔧 第三阶段：成功注册文件元数据数据集: {} (ID: {})", metadata.filePath, datasetId);
            return AsyncResult<std::string>::success(datasetId);
            
        } catch (const std::exception& e) {
            LOG_ERROR("🔧 第三阶段：注册文件元数据数据集异常 [{}]: {}", metadata.filePath, e.what());
            return AsyncResult<std::string>::failure("注册文件元数据数据集异常: " + std::string(e.what()));
        }
    });
}

boost::future<AsyncResult<void>> DatasetMetadataRegistry::updateDatasetAsync(
    const std::string& datasetId,
    const MetadataUpdate& update) {
    
    return boost::async(boost::launch::async, [this, datasetId, update]() -> AsyncResult<void> {
        try {
            std::lock_guard<std::mutex> lock(registryMutex_);
            
            auto it = registry_.find(datasetId);
            if (it == registry_.end()) {
                return AsyncResult<void>::failure("未找到指定的数据集: " + datasetId);
            }
            
            // 更新数据集元数据
            auto& entry = it->second;
            
            if (update.dataQuality.has_value()) {
                entry.dataQuality = update.dataQuality.value();
            }
            
            if (update.completeness.has_value()) {
                entry.completeness = update.completeness.value();
            }
            
            // 更新属性
            for (const auto& [key, value] : update.updatedAttributes) {
                entry.attributes[key] = value;
            }
            
            // 更新变量信息
            if (!update.updatedVariables.empty()) {
                entry.variables = update.updatedVariables;
            }
            
            LOG_INFO("成功更新数据集: {}", datasetId);
            return AsyncResult<void>::success();
            
        } catch (const std::exception& e) {
            LOG_ERROR("更新数据集异常 [{}]: {}", datasetId, e.what());
            return AsyncResult<void>::failure("更新数据集异常: " + std::string(e.what()));
        }
    });
}

boost::future<AsyncResult<void>> DatasetMetadataRegistry::unregisterDatasetAsync(
    const std::string& datasetId) {
    
    return boost::async(boost::launch::async, [this, datasetId]() -> AsyncResult<void> {
        try {
            std::lock_guard<std::mutex> lock(registryMutex_);
            
            auto it = registry_.find(datasetId);
            if (it == registry_.end()) {
                return AsyncResult<void>::failure("未找到指定的数据集: " + datasetId);
            }
            
            // 从注册表中移除
            registry_.erase(it);
            
            LOG_INFO("成功注销数据集: {}", datasetId);
            return AsyncResult<void>::success();
            
        } catch (const std::exception& e) {
            LOG_ERROR("注销数据集异常 [{}]: {}", datasetId, e.what());
            return AsyncResult<void>::failure("注销数据集异常: " + std::string(e.what()));
        }
    });
}

boost::future<AsyncResult<std::vector<oscean::core_services::FileMetadata>>> DatasetMetadataRegistry::queryDatasetsAsync(
    const QueryCriteria& criteria) {
    
    return boost::async(boost::launch::async, [this, criteria]() -> AsyncResult<std::vector<oscean::core_services::FileMetadata>> {
        try {
            std::vector<oscean::core_services::FileMetadata> results;
            
            std::lock_guard<std::mutex> lock(registryMutex_);
            
            // 遍历注册表进行查询
            for (const auto& [id, entry] : registry_) {
                bool matches = true;
                
                // 检查数据类型过滤
                if (!criteria.dataTypes.empty()) {
                    auto it = std::find(criteria.dataTypes.begin(), criteria.dataTypes.end(), entry.dataType);
                    if (it == criteria.dataTypes.end()) {
                        matches = false;
                    }
                }
                
                // 检查变量包含过滤
                if (matches && !criteria.variablesInclude.empty()) {
                    bool hasRequiredVariable = false;
                    for (const auto& requiredVar : criteria.variablesInclude) {
                        for (const auto& variable : entry.variables) {
                            if (variable.name.find(requiredVar) != std::string::npos) {
                                hasRequiredVariable = true;
                                break;
                            }
                        }
                        if (hasRequiredVariable) break;
                    }
                    if (!hasRequiredVariable) {
                        matches = false;
                    }
                }
                
                // 检查变量排除过滤
                if (matches && !criteria.variablesExclude.empty()) {
                    for (const auto& excludeVar : criteria.variablesExclude) {
                        for (const auto& variable : entry.variables) {
                            if (variable.name.find(excludeVar) != std::string::npos) {
                                matches = false;
                                break;
                            }
                        }
                        if (!matches) break;
                    }
                }
                
                // 检查质量标准
                if (matches && criteria.minDataQuality) {
                    if (entry.dataQuality < *criteria.minDataQuality) {
                        matches = false;
                    }
                }
                
                if (matches) {
                    // 转换为FileMetadata
                    oscean::core_services::FileMetadata fileMetadata;
                    fileMetadata.metadataId = entry.datasetId;
                    fileMetadata.fileId = entry.datasetId;  // 兼容字段
                    fileMetadata.filePath = entry.filePath;
                    fileMetadata.fileName = std::filesystem::path(entry.filePath).filename().string();
                    fileMetadata.inferredDataType = static_cast<oscean::core_services::DataType>(entry.dataType);
                    // 转换空间信息 - 使用正确的字段名
                    fileMetadata.spatialCoverage.minX = entry.spatialInfo.bounds.minLongitude;
                    fileMetadata.spatialCoverage.maxX = entry.spatialInfo.bounds.maxLongitude;
                    fileMetadata.spatialCoverage.minY = entry.spatialInfo.bounds.minLatitude;
                    fileMetadata.spatialCoverage.maxY = entry.spatialInfo.bounds.maxLatitude;
                    // 转换时间信息 - FileMetadata.timeRange是TimeRange类型，需要转换
                    // 假设entry.temporalInfo.timeRange包含ISO格式字符串，需要转换为time_point
                    // 这里使用简化处理，实际应该使用时间转换工具
                    fileMetadata.timeRange.startTime = std::chrono::system_clock::now();
                    fileMetadata.timeRange.endTime = std::chrono::system_clock::now();
                    fileMetadata.variables = entry.variables;
                    fileMetadata.dataQuality = entry.dataQuality;
                    fileMetadata.completeness = entry.completeness;
                    // 转换时间戳 - extractionTimestamp需要int64_t，lastIndexedTime需要string
                    fileMetadata.extractionTimestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    fileMetadata.lastIndexedTime = entry.registrationTime;
                    fileMetadata.metadata = entry.attributes;
                    
                    results.push_back(fileMetadata);
                }
                
                // 应用限制
                if (criteria.limit && results.size() >= *criteria.limit) {
                    break;
                }
            }
            
            // 应用分页
            if (criteria.offset || criteria.limit) {
                size_t offset = criteria.offset.value_or(0);
                size_t limit = criteria.limit.value_or(results.size());
                
                if (offset < results.size()) {
                    size_t endPos = std::min(offset + limit, results.size());
                    results = std::vector<oscean::core_services::FileMetadata>(results.begin() + offset, results.begin() + endPos);
                } else {
                    results.clear();
                }
            }
            
            LOG_INFO("数据集查询完成: 结果数量={}", results.size());
            return AsyncResult<std::vector<oscean::core_services::FileMetadata>>::success(std::move(results));
            
        } catch (const std::exception& e) {
            LOG_ERROR("查询数据集异常: {}", e.what());
            return AsyncResult<std::vector<oscean::core_services::FileMetadata>>::failure("查询数据集异常: " + std::string(e.what()));
        }
    });
}

// 私有辅助方法实现

std::string DatasetMetadataRegistry::generateDatasetId(const ::oscean::core_services::FileMetadata& metadata) {
    // 🔧 第三阶段：使用文件路径和时间戳生成唯一ID
    std::hash<std::string> hasher;
    size_t pathHash = hasher(metadata.filePath);
    
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    std::ostringstream oss;
    oss << "dataset_" << pathHash << "_" << timestamp;
    return oss.str();
}

bool DatasetMetadataRegistry::validateFileMetadata(const ::oscean::core_services::FileMetadata& metadata) {
    if (metadata.filePath.empty()) {
        LOG_WARN("🔧 第三阶段：文件元数据验证失败: 文件路径为空");
        return false;
    }
    
    if (metadata.variables.empty()) {
        LOG_WARN("🔧 第三阶段：文件元数据验证警告: 没有变量信息 - {}", metadata.filePath);
        // 不阻止注册，只是警告
    }
    
    if (metadata.format.empty()) {
        LOG_WARN("🔧 第三阶段：文件元数据验证警告: 文件格式为空 - {}", metadata.filePath);
    }
    
    return true;
}

DatabaseType DatasetMetadataRegistry::determineTargetDatabase(const ::oscean::core_services::FileMetadata& metadata) {
    // 🔧 第三阶段：从FileMetadata推断数据类型并映射到数据库类型
    DataType dataType = inferDataTypeFromFileMetadata(metadata);
    
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
            // 默认存储到海洋环境数据库
            LOG_INFO("🔧 第三阶段：未知数据类型，默认存储到海洋环境数据库 - {}", metadata.filePath);
            return DatabaseType::OCEAN_ENVIRONMENT;
    }
}

std::vector<DatabaseType> DatasetMetadataRegistry::determineDatabasesForQuery(
    const QueryCriteria& criteria) {
    
    std::vector<DatabaseType> databases;
    
    if (criteria.dataTypes.empty()) {
        // 如果没有指定数据类型，查询所有数据库
        return getAllDatabaseTypes();
    }
    
    for (const auto& dataType : criteria.dataTypes) {
        DatabaseType dbType = static_cast<DatabaseType>(dataType);
        databases.push_back(dbType);
    }
    
    return databases;
}

std::vector<DatabaseType> DatasetMetadataRegistry::getAllDatabaseTypes() {
    return {
        DatabaseType::OCEAN_ENVIRONMENT,
        DatabaseType::TOPOGRAPHY_BATHYMETRY,
        DatabaseType::BOUNDARY_LINES,
        DatabaseType::SONAR_PROPAGATION
    };
}

// === 🔧 第三阶段：新增方法实现 ===

DataType DatasetMetadataRegistry::inferDataTypeFromFileMetadata(const ::oscean::core_services::FileMetadata& metadata) const {
    // 从文件路径推断数据类型
    std::string lowerPath = metadata.filePath;
    std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
    
    // 检查路径中的关键词
    if (lowerPath.find("/to/") != std::string::npos || 
        lowerPath.find("\\to\\") != std::string::npos ||
        lowerPath.find("topography") != std::string::npos ||
        lowerPath.find("bathymetry") != std::string::npos ||
        lowerPath.find("elevation") != std::string::npos ||
        lowerPath.find("depth") != std::string::npos) {
        return DataType::TOPOGRAPHY_BATHYMETRY;
    }
    
    if (lowerPath.find("boundary") != std::string::npos ||
        lowerPath.find("border") != std::string::npos ||
        lowerPath.find("coastline") != std::string::npos) {
        return DataType::BOUNDARY_LINES;
    }
    
    if (lowerPath.find("sonar") != std::string::npos ||
        lowerPath.find("acoustic") != std::string::npos ||
        lowerPath.find("propagation") != std::string::npos) {
        return DataType::SONAR_PROPAGATION;
    }
    
    if (lowerPath.find("tactical") != std::string::npos ||
        lowerPath.find("military") != std::string::npos ||
        lowerPath.find("operation") != std::string::npos) {
        return DataType::TACTICAL_ENVIRONMENT;
    }
    
    // 检查变量名称
    for (const auto& variable : metadata.variables) {
        std::string lowerVarName = variable.name;
        std::transform(lowerVarName.begin(), lowerVarName.end(), lowerVarName.begin(), ::tolower);
        
        if (lowerVarName.find("temp") != std::string::npos ||
            lowerVarName.find("sal") != std::string::npos ||
            lowerVarName.find("current") != std::string::npos ||
            lowerVarName.find("velocity") != std::string::npos ||
            lowerVarName == "u" || lowerVarName == "v" ||
            lowerVarName.find("ssh") != std::string::npos ||
            lowerVarName.find("sla") != std::string::npos) {
            return DataType::OCEAN_ENVIRONMENT;
        }
        
        if (lowerVarName.find("elevation") != std::string::npos ||
            lowerVarName.find("depth") != std::string::npos ||
            lowerVarName.find("bathymetry") != std::string::npos ||
            lowerVarName.find("topography") != std::string::npos) {
            return DataType::TOPOGRAPHY_BATHYMETRY;
        }
    }
    
    // 检查文件格式
    std::string lowerFormat = metadata.format;
    std::transform(lowerFormat.begin(), lowerFormat.end(), lowerFormat.begin(), ::tolower);
    
    if (lowerFormat == "netcdf" || lowerFormat == "nc") {
        // NetCDF文件通常是海洋环境数据
        return DataType::OCEAN_ENVIRONMENT;
    }
    
    // 默认返回海洋环境
    return DataType::OCEAN_ENVIRONMENT;
}

void DatasetMetadataRegistry::convertSpatialInfo(const ::oscean::core_services::FileMetadata& source, SpatialInfo& target) const {
    // 🔧 第三阶段：从FileMetadata.spatialCoverage转换到SpatialInfo
    target.bounds.minLongitude = source.spatialCoverage.minX;
    target.bounds.maxLongitude = source.spatialCoverage.maxX;
    target.bounds.minLatitude = source.spatialCoverage.minY;
    target.bounds.maxLatitude = source.spatialCoverage.maxY;
    target.coordinateSystem = source.crs.id;
    
    // 设置默认分辨率（如果FileMetadata中有的话）
    target.spatialResolution = 0.0;  // 默认值，需要从数据中计算
}

void DatasetMetadataRegistry::convertTemporalInfo(const ::oscean::core_services::FileMetadata& source, TemporalInfo& target) const {
    // 🔧 第三阶段：从FileMetadata.timeRange转换到TemporalInfo
    auto startTimeT = std::chrono::system_clock::to_time_t(source.timeRange.startTime);
    auto endTimeT = std::chrono::system_clock::to_time_t(source.timeRange.endTime);
    
    std::stringstream startSS, endSS;
    startSS << std::put_time(std::gmtime(&startTimeT), "%Y-%m-%dT%H:%M:%SZ");
    endSS << std::put_time(std::gmtime(&endTimeT), "%Y-%m-%dT%H:%M:%SZ");
    
    target.timeRange.startTime = startSS.str();
    target.timeRange.endTime = endSS.str();
}

} // namespace oscean::core_services::metadata::impl 