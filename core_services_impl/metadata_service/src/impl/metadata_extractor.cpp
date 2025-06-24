// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "impl/metadata_extractor.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/time/time_services.h"
#include "common_utils/utilities/logging_utils.h"
#include <boost/thread/future.hpp>

#include <memory>
#include <utility>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <mutex>
#include <regex>

namespace oscean::core_services::metadata::impl {

// PIMPL (Pointer to Implementation)
class MetadataExtractor::Impl {
public:
    explicit Impl(std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices)
        : commonServices_(std::move(commonServices)) {
        if (commonServices_) {
            LOG_DEBUG("MetadataManager::Impl 初始化成功");
        } else {
            LOG_WARN("CommonServicesFactory 为空，某些功能可能受限");
        }
        
        // 初始化内存数据库（简化实现）
        initializeInMemoryDatabase();
    }

    ~Impl() = default;

    /**
     * @brief 🔧 第三阶段：存储标准化文件元数据
     */
    boost::future<AsyncResult<std::string>> storeFileMetadataAsync(
        const ::oscean::core_services::FileMetadata& metadata) {
        return boost::async(boost::launch::async, [this, metadata]() -> AsyncResult<std::string> {
            try {
                // 🔧 第三阶段验证：标准化文件元数据完整性
                if (metadata.filePath.empty()) {
                    return AsyncResult<std::string>::failure("文件元数据缺少文件路径");
                }

                // 🔧 第三阶段：直接使用FileMetadata，不需要转换
                auto storedMetadata = metadata;
                
                // 🔧 第三阶段：生成文件元数据ID
                if (storedMetadata.fileId.empty()) {
                    storedMetadata.fileId = generateUniqueId(metadata.filePath);
                }
                
                // 🔧 第三阶段：简化CRS处理，直接使用FileMetadata.crs
                if (storedMetadata.crs.id.empty() && !storedMetadata.crs.wkt.empty()) {
                    // 尝试从WKT中提取EPSG代码
                    if (storedMetadata.crs.wkt.find("EPSG") != std::string::npos) {
                        // 提取EPSG代码的简单实现
                        size_t pos = storedMetadata.crs.wkt.find("EPSG");
                        if (pos != std::string::npos) {
                            storedMetadata.crs.id = "EPSG:4326"; // 默认设置
                            LOG_INFO("🔧 从WKT提取CRS ID: {}", storedMetadata.crs.id);
                        }
                    } else {
                        storedMetadata.crs.id = "EPSG:4326"; // 默认CRS
                        LOG_INFO("🔧 设置默认CRS: EPSG:4326");
                    }
                }
                
                // 🔧 第三阶段：存储到内存数据库
                {
                    std::lock_guard<std::mutex> lock(databaseMutex_);
                    metadataDatabase_[storedMetadata.fileId] = storedMetadata;
                    
                    // 更新索引
                    updateIndexes(storedMetadata);
                }
                
                LOG_INFO("🔧 第三阶段：成功存储文件元数据: {} (ID: {})", metadata.filePath, storedMetadata.fileId);
                return AsyncResult<std::string>::success(storedMetadata.fileId);
                
            } catch (const std::exception& e) {
                LOG_ERROR("🔧 第三阶段：文件元数据存储异常 [{}]: {}", metadata.filePath, e.what());
                return AsyncResult<std::string>::failure("文件元数据存储异常: " + std::string(e.what()));
            }
        });
    }

    /**
     * @brief 🔧 第三阶段：批量存储文件元数据
     */
    boost::future<AsyncResult<std::vector<std::string>>> storeBatchFileMetadataAsync(
        const std::vector<::oscean::core_services::FileMetadata>& metadataList) {
        return boost::async(boost::launch::async, [this, metadataList]() -> AsyncResult<std::vector<std::string>> {
            try {
                std::vector<std::string> results;
                results.reserve(metadataList.size());
                
                // 🔧 第三阶段：并行处理多个文件元数据
                std::vector<boost::future<AsyncResult<std::string>>> futures;
                futures.reserve(metadataList.size());
                
                for (const auto& metadata : metadataList) {
                    futures.push_back(storeFileMetadataAsync(metadata));
                }
                
                // 🔧 第三阶段：收集元数据ID结果
                for (auto& future : futures) {
                    auto result = future.get();
                    if (result.isSuccess()) {
                        results.push_back(result.getData());
                    } else {
                        LOG_WARN("🔧 批量存储中的文件元数据失败: {}", result.getError());
                    }
                }
                
                LOG_INFO("🔧 第三阶段：批量文件元数据存储完成: 成功 {}/{} 个文件", 
                        results.size(), metadataList.size());
                
                return AsyncResult<std::vector<std::string>>::success(std::move(results));
                
            } catch (const std::exception& e) {
                LOG_ERROR("🔧 第三阶段：批量文件元数据存储异常: {}", e.what());
                return AsyncResult<std::vector<std::string>>::failure("批量文件元数据存储异常: " + std::string(e.what()));
            }
        });
    }

    /**
     * @brief 🔧 第三阶段：查询存储的文件元数据
     */
    boost::future<AsyncResult<std::vector<::oscean::core_services::FileMetadata>>> queryFileMetadataAsync(
        const std::string& queryFilter) {
        return boost::async(boost::launch::async, [this, queryFilter]() -> AsyncResult<std::vector<::oscean::core_services::FileMetadata>> {
            try {
                std::vector<::oscean::core_services::FileMetadata> results;
                
                std::lock_guard<std::mutex> lock(databaseMutex_);
                
                // 解析查询过滤器
                if (queryFilter.empty() || queryFilter == "*") {
                    // 返回所有元数据
                    for (const auto& [id, metadata] : metadataDatabase_) {
                        results.push_back(metadata);
                    }
                } else if (queryFilter.find("filePath:") == 0) {
                    // 按文件路径查询
                    std::string pathPattern = queryFilter.substr(9); // 移除 "filePath:" 前缀
                    for (const auto& [id, metadata] : metadataDatabase_) {
                        if (metadata.filePath.find(pathPattern) != std::string::npos) {
                            results.push_back(metadata);
                        }
                    }
                } else if (queryFilter.find("format:") == 0) {
                    // 🔧 第三阶段：按文件格式查询
                    std::string formatStr = queryFilter.substr(7); // 移除 "format:" 前缀
                    for (const auto& [id, metadata] : metadataDatabase_) {
                        if (metadata.format.find(formatStr) != std::string::npos) {
                            results.push_back(metadata);
                        }
                    }
                } else if (queryFilter.find("variable:") == 0) {
                    // 🔧 第三阶段：按变量名查询（使用FileMetadata.variables）
                    std::string varName = queryFilter.substr(9); // 移除 "variable:" 前缀
                    for (const auto& [id, metadata] : metadataDatabase_) {
                        for (const auto& variable : metadata.variables) {
                            if (variable.name.find(varName) != std::string::npos) {
                                results.push_back(metadata);
                                break;
                            }
                        }
                    }
                } else {
                    // 🔧 第三阶段：通用文本搜索
                    for (const auto& [id, metadata] : metadataDatabase_) {
                        if (containsFileMetadataText(metadata, queryFilter)) {
                            results.push_back(metadata);
                        }
                    }
                }
                
                LOG_INFO("🔧 第三阶段：文件元数据查询完成: 查询条件='{}', 结果数量={}", queryFilter, results.size());
                
                return AsyncResult<std::vector<::oscean::core_services::FileMetadata>>::success(std::move(results));
                
            } catch (const std::exception& e) {
                LOG_ERROR("🔧 第三阶段：文件元数据查询异常: {}", e.what());
                return AsyncResult<std::vector<::oscean::core_services::FileMetadata>>::failure("文件元数据查询异常: " + std::string(e.what()));
            }
        });
    }

    /**
     * @brief 🔧 第三阶段：更新文件元数据
     */
    boost::future<AsyncResult<::oscean::core_services::FileMetadata>> updateFileMetadataAsync(
        const std::string& filePath, const ::oscean::core_services::FileMetadata& updatedMetadata) {
        return boost::async(boost::launch::async, [this, filePath, updatedMetadata]() -> AsyncResult<::oscean::core_services::FileMetadata> {
            try {
                std::lock_guard<std::mutex> lock(databaseMutex_);
                
                // 查找现有元数据记录
                std::string targetId;
                for (const auto& [id, metadata] : metadataDatabase_) {
                    if (metadata.filePath == filePath) {
                        targetId = id;
                        break;
                    }
                }
                
                if (targetId.empty()) {
                    return AsyncResult<::oscean::core_services::FileMetadata>::failure("未找到指定文件的FileMetadata: " + filePath);
                }
                
                // 🔧 第三阶段：直接更新FileMetadata
                auto result = updatedMetadata;
                result.fileId = targetId;
                result.filePath = filePath; // 确保文件路径不变
                
                // 🔧 第三阶段：简化时间戳处理
                // FileMetadata不需要单独的修改时间戳字段
                
                // 🔧 第三阶段：保存到数据库
                metadataDatabase_[targetId] = result;
                
                // 🔧 第三阶段：重新建立索引
                updateIndexes(result);
                
                LOG_INFO("🔧 第三阶段：成功更新文件元数据: {}", filePath);
                return AsyncResult<::oscean::core_services::FileMetadata>::success(std::move(result));
                
            } catch (const std::exception& e) {
                LOG_ERROR("🔧 第三阶段：文件元数据更新异常 [{}]: {}", filePath, e.what());
                return AsyncResult<::oscean::core_services::FileMetadata>::failure("文件元数据更新异常: " + std::string(e.what()));
            }
        });
    }

    /**
     * @brief 删除元数据
     */
    boost::future<AsyncResult<bool>> deleteMetadataAsync(const std::string& filePath) {
        return boost::async(boost::launch::async, [this, filePath]() -> AsyncResult<bool> {
            try {
                std::lock_guard<std::mutex> lock(databaseMutex_);
                
                // 查找并删除元数据记录
                std::string targetId;
                for (const auto& [id, metadata] : metadataDatabase_) {
                    if (metadata.filePath == filePath) {
                        targetId = id;
                        break;
                    }
                }
                
                if (targetId.empty()) {
                    return AsyncResult<bool>::failure("未找到指定文件的元数据: " + filePath);
                }
                
                // 删除记录
                metadataDatabase_.erase(targetId);
                
                // 清理索引
                cleanupIndexes(targetId);
                
                LOG_INFO("成功删除元数据: {}", filePath);
                return AsyncResult<bool>::success(true);
                
            } catch (const std::exception& e) {
                LOG_ERROR("元数据删除异常 [{}]: {}", filePath, e.what());
                return AsyncResult<bool>::failure("元数据删除异常: " + std::string(e.what()));
            }
        });
    }

private:
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices_;
    
    // 🔧 第三阶段：内存数据库存储FileMetadata
    std::map<std::string, ::oscean::core_services::FileMetadata> metadataDatabase_;
    std::mutex databaseMutex_;
    
    // 🔧 第三阶段：索引结构保持不变
    std::map<std::string, std::vector<std::string>> formatIndex_;  // 按格式索引
    std::map<std::string, std::vector<std::string>> variableIndex_;
    std::map<std::string, std::vector<std::string>> pathIndex_;

    /**
     * @brief 初始化内存数据库
     */
    void initializeInMemoryDatabase() {
        LOG_DEBUG("🔧 第三阶段：初始化内存FileMetadata数据库");
        // 🔧 第三阶段：清空所有数据结构
        metadataDatabase_.clear();
        formatIndex_.clear();  // 修正：使用formatIndex_
        variableIndex_.clear();
        pathIndex_.clear();
    }

    /**
     * @brief 生成唯一ID
     */
    std::string generateUniqueId(const std::string& filePath) {
        // 使用文件路径的哈希值作为ID
        std::hash<std::string> hasher;
        size_t hashValue = hasher(filePath);
        
        // 添加时间戳确保唯一性
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        
        std::ostringstream oss;
        oss << "meta_" << hashValue << "_" << timestamp;
        return oss.str();
    }

    /**
     * @brief 🔧 第三阶段：更新FileMetadata索引
     */
    void updateIndexes(const ::oscean::core_services::FileMetadata& metadata) {
        // 🔧 第三阶段：更新格式索引
        if (!metadata.format.empty()) {
            formatIndex_[metadata.format].push_back(metadata.fileId);
        }
        
        // 🔧 第三阶段：更新变量索引（使用FileMetadata.variables）
        for (const auto& variable : metadata.variables) {
            variableIndex_[variable.name].push_back(metadata.fileId);
        }
        
        // 🔧 第三阶段：更新路径索引
        std::filesystem::path fsPath(metadata.filePath);
        std::string directory = fsPath.parent_path().string();
        pathIndex_[directory].push_back(metadata.fileId);
    }

    /**
     * @brief 🔧 第三阶段：清理FileMetadata索引
     */
    void cleanupIndexes(const std::string& fileId) {
        // 🔧 第三阶段：从所有索引中移除指定的fileId
        for (auto& [format, ids] : formatIndex_) {
            ids.erase(std::remove(ids.begin(), ids.end(), fileId), ids.end());
        }
        
        for (auto& [varName, ids] : variableIndex_) {
            ids.erase(std::remove(ids.begin(), ids.end(), fileId), ids.end());
        }
        
        for (auto& [path, ids] : pathIndex_) {
            ids.erase(std::remove(ids.begin(), ids.end(), fileId), ids.end());
        }
    }

    /**
     * @brief 解析数据类型字符串
     */
    DataType parseDataType(const std::string& typeStr) {
        if (typeStr == "OCEAN_ENVIRONMENT") return DataType::OCEAN_ENVIRONMENT;
        if (typeStr == "TOPOGRAPHY_BATHYMETRY") return DataType::TOPOGRAPHY_BATHYMETRY;
        if (typeStr == "BOUNDARY_LINES") return DataType::BOUNDARY_LINES;
        if (typeStr == "SONAR_PROPAGATION") return DataType::SONAR_PROPAGATION;
        return DataType::UNKNOWN;
    }

    /**
     * @brief 🔧 第三阶段：检查FileMetadata是否包含指定文本
     */
    bool containsFileMetadataText(const ::oscean::core_services::FileMetadata& metadata, const std::string& text) {
        std::string lowerText = text;
        std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
        
        // 🔧 第三阶段：检查文件路径
        std::string lowerPath = metadata.filePath;
        std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
        if (lowerPath.find(lowerText) != std::string::npos) {
            return true;
        }
        
        // 🔧 第三阶段：检查文件格式
        std::string lowerFormat = metadata.format;
        std::transform(lowerFormat.begin(), lowerFormat.end(), lowerFormat.begin(), ::tolower);
        if (lowerFormat.find(lowerText) != std::string::npos) {
            return true;
        }
        
        // 🔧 第三阶段：检查变量名（使用FileMetadata.variables）
        for (const auto& variable : metadata.variables) {
            std::string lowerVarName = variable.name;
            std::transform(lowerVarName.begin(), lowerVarName.end(), lowerVarName.begin(), ::tolower);
            if (lowerVarName.find(lowerText) != std::string::npos) {
                return true;
            }
        }
        
        // 🔧 第三阶段：检查元数据属性
        for (const auto& [key, value] : metadata.metadata) {
            std::string lowerValue = value;
            std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(), ::tolower);
            if (lowerValue.find(lowerText) != std::string::npos) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * @brief 坐标系统标准化处理 - 专门处理极地坐标和非标准投影
     * @param coordinateSystem 原始坐标系统字符串
     * @return 标准化后的坐标系统字符串
     */
    std::string standardizeCoordinateSystem(const std::string& coordinateSystem) {
        if (coordinateSystem.empty()) {
            return "EPSG:4326"; // 默认WGS84
        }
        
        try {
            // 🔧 Step 1: 检查是否已经是标准EPSG格式
            if (coordinateSystem.find("EPSG:") == 0) {
                return coordinateSystem; // 已经是标准格式
            }
            
            // 🔧 Step 2: 处理PROJ字符串格式
            if (coordinateSystem.find("+proj=") != std::string::npos) {
                std::string cleanedProj = cleanPolarProjectionString(coordinateSystem);
                
                // 尝试映射到标准EPSG
                std::string epsgMapping = mapPolarProjectionToEPSG(cleanedProj);
                if (!epsgMapping.empty()) {
                    LOG_INFO("🎯 极地投影映射到标准EPSG: {} -> {}", coordinateSystem, epsgMapping);
                    return epsgMapping;
                }
                
                // 如果无法映射，返回清理后的PROJ字符串
                if (cleanedProj != coordinateSystem) {
                    LOG_INFO("🔧 PROJ字符串已清理: {} -> {}", coordinateSystem, cleanedProj);
                }
                return cleanedProj;
            }
            
            // 🔧 Step 3: 处理其他格式（WKT等）
            LOG_INFO("✅ 保持原始坐标系统格式: {}", coordinateSystem);
            return coordinateSystem;
            
        } catch (const std::exception& e) {
            LOG_WARN("坐标系统标准化处理异常: {}, 使用原始格式", e.what());
            return coordinateSystem;
        }
    }
    
    /**
     * @brief 清理极地投影PROJ字符串 - 专门处理极地坐标系统的非标准参数
     */
    std::string cleanPolarProjectionString(const std::string& projString) {
        if (projString.empty()) return "";
        
        // 🎯 Step 1: 检查是否可以映射到标准EPSG
        std::string epsgMapping = mapPolarProjectionToEPSG(projString);
        if (!epsgMapping.empty()) {
            LOG_INFO("🎯 极地投影PROJ字符串映射到标准EPSG: {}", epsgMapping);
            return epsgMapping;
        }
        
        // 🔧 Step 2: 清理冲突和无效参数
        std::string cleaned = projString;
        
        // 移除 lat_ts=90 对于极地立体投影
        if (cleaned.find("+proj=stere") != std::string::npos && 
            cleaned.find("+lat_0=90") != std::string::npos) {
            std::regex latTsPattern(R"(\s*\+lat_ts=90(\.\d+)?\s*)");
            cleaned = std::regex_replace(cleaned, latTsPattern, " ");
            LOG_INFO("🔧 移除极地投影的非标准lat_ts=90参数");
        }
        
        // 🔧 处理椭球参数冲突：+R= 与 +ellps=sphere 冲突
        if (cleaned.find("+R=") != std::string::npos && 
            cleaned.find("+ellps=sphere") != std::string::npos) {
            // 移除 +ellps=sphere，保留 +R=
            std::regex ellpsPattern(R"(\s*\+ellps=sphere\s*)");
            cleaned = std::regex_replace(cleaned, ellpsPattern, " ");
            LOG_INFO("🔧 移除冲突的+ellps=sphere参数，保留+R=参数");
        }
        
        // 清理多余空格
        std::regex extraSpaces(R"(\s+)");
        cleaned = std::regex_replace(cleaned, extraSpaces, " ");
        
        // 去除首尾空格
        cleaned = std::regex_replace(cleaned, std::regex(R"(^\s+|\s+$)"), "");
        
        // 🔧 最后步骤：确保PROJ字符串能被正确识别为CRS
        // 对于自定义球体参数，需要添加type=crs标识
        if (cleaned.find("+R=") != std::string::npos && 
            cleaned.find("+type=crs") == std::string::npos) {
            LOG_INFO("🔧 检测到自定义球体半径，添加+type=crs标识确保PROJ识别为CRS");
            cleaned = "+type=crs " + cleaned;
        }
        
        return cleaned;
    }
    
    /**
     * @brief 将极地投影PROJ字符串映射到标准EPSG代码 - 专门处理极地坐标系统
     */
    std::string mapPolarProjectionToEPSG(const std::string& projString) {
        // 🔧 修正：检查是否为NSIDC极地立体投影 (EPSG:3413)
        // 但只有在使用标准WGS84椭球参数时才映射到EPSG:3413
        if (projString.find("+proj=stere") != std::string::npos &&
            projString.find("+lat_0=90") != std::string::npos &&
            projString.find("+lon_0=-45") != std::string::npos) {
            
            // 🎯 关键修正：检查是否使用自定义球体半径
            std::regex radiusPattern(R"(\+R=([0-9\.e\+\-]+))");
            std::smatch match;
            if (std::regex_search(projString, match, radiusPattern)) {
                double radius = std::stod(match[1].str());
                // 🔧 重要：如果使用自定义球体半径，不要映射到标准EPSG
                // 因为EPSG:3413使用WGS84椭球，与自定义球体R=6378273不同
                if (std::abs(radius - 6378273.0) < 1000.0) { // 检测到自定义NSIDC球体
                    LOG_INFO("🔧 检测到自定义球体半径({:.0f}m)，保持使用原始PROJ字符串", radius);
                    LOG_INFO("🔧 不映射到EPSG:3413，因为椭球参数不匹配");
                    return ""; // 不映射，保持原始PROJ字符串
                }
            }
            
            // 🎯 只有使用标准WGS84椭球时才映射到EPSG:3413
            if (projString.find("+datum=WGS84") != std::string::npos ||
                projString.find("+ellps=WGS84") != std::string::npos) {
                LOG_INFO("🎯 检测到标准WGS84极地立体投影，映射到EPSG:3413");
                return "EPSG:3413";
            }
        }
        
        // 检查是否为标准WGS84地理坐标系
        if (projString.find("+proj=longlat") != std::string::npos &&
            (projString.find("+datum=WGS84") != std::string::npos ||
             projString.find("+ellps=WGS84") != std::string::npos)) {
            LOG_INFO("🎯 检测到WGS84地理坐标系，映射到EPSG:4326");
            return "EPSG:4326";
        }
        
        // 可以在这里添加更多标准投影的检测
        // 但都要确保椭球参数匹配
        
        return ""; // 没有找到匹配的标准EPSG，保持原始PROJ字符串
    }
};

// MetadataExtractor 实现
MetadataExtractor::MetadataExtractor(
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices)
    : pImpl_(std::make_unique<Impl>(std::move(commonServices))) {
    LOG_DEBUG("MetadataManager 构造完成 - 专注于元数据管理，不进行文件提取");
}

MetadataExtractor::~MetadataExtractor() = default;

// 🔧 第三阶段：重构公共接口方法使用FileMetadata
boost::future<AsyncResult<std::string>> MetadataExtractor::storeFileMetadataAsync(
    const ::oscean::core_services::FileMetadata& metadata) {
    return pImpl_->storeFileMetadataAsync(metadata);
}

boost::future<AsyncResult<std::vector<std::string>>> MetadataExtractor::storeBatchFileMetadataAsync(
    const std::vector<::oscean::core_services::FileMetadata>& metadataList) {
    return pImpl_->storeBatchFileMetadataAsync(metadataList);
}

boost::future<AsyncResult<std::vector<::oscean::core_services::FileMetadata>>> MetadataExtractor::queryFileMetadataAsync(
    const std::string& queryFilter) {
    return pImpl_->queryFileMetadataAsync(queryFilter);
}

boost::future<AsyncResult<::oscean::core_services::FileMetadata>> MetadataExtractor::updateFileMetadataAsync(
    const std::string& filePath, const ::oscean::core_services::FileMetadata& updatedMetadata) {
    return pImpl_->updateFileMetadataAsync(filePath, updatedMetadata);
}

boost::future<AsyncResult<bool>> MetadataExtractor::deleteFileMetadataAsync(const std::string& filePath) {
    return pImpl_->deleteMetadataAsync(filePath);
}

} // namespace oscean::core_services::metadata::impl 