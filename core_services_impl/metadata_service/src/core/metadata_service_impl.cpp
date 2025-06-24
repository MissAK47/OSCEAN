/*
 * metadata_service_impl.cpp
 *
 *  Created on: 2024年7月2日
 *      Author: Administrator
 */
#include "impl/metadata_service_impl.h"
#include "impl/metadata_standardizer.h"
#include "common_utils/utilities/logging_utils.h"
#include "core_services/common_data_types.h"
#include "impl/metadata_extractor.h"
#include "workflow_engine/service_management/i_service_manager.h"
#include "core_services/crs/i_crs_service.h"

#include <boost/thread/future.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <algorithm>

namespace oscean {
namespace core_services {
namespace metadata {
namespace impl {

MetadataServiceImpl::MetadataServiceImpl(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServicesFactory,
    std::shared_ptr<UnifiedDatabaseManager> dbManager,
    std::shared_ptr<IntelligentRecognizer> recognizer,
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager)
    : commonServices_(commonServicesFactory),
      dbManager_(dbManager),
      recognizer_(recognizer),
      serviceManager_(serviceManager),
      isInitialized_(false) {
    
    // 初始化日志器
    logger_ = oscean::common_utils::getModuleLogger("MetadataService");
    if (logger_) {
        LOG_INFO("MetadataServiceImpl 构造完成，CRS服务将延迟加载");
    } else {
        std::cout << "[DEBUG] MetadataServiceImpl logger初始化失败，使用std::cout替代" << std::endl;
    }
}

MetadataServiceImpl::~MetadataServiceImpl() {
    LOG_INFO("MetadataServiceImpl destructor called.");
}

bool MetadataServiceImpl::initialize() {
    std::cout << "[DEBUG] MetadataServiceImpl::initialize() 开始..." << std::endl;
    
    if (!logger_) {
        std::cerr << "Logger not available for MetadataServiceImpl initialization." << std::endl;
        std::cout << "[DEBUG] logger_为null，初始化失败" << std::endl;
        return false;
    }
    std::cout << "[DEBUG] logger_检查通过" << std::endl;

    LOG_INFO("Initializing MetadataService with unified architecture...");

    std::cout << "[DEBUG] 检查核心服务依赖..." << std::endl;
    if (!commonServices_ || !dbManager_ || !recognizer_ || !serviceManager_) {
        std::cout << "[DEBUG] 核心服务依赖检查失败:" << std::endl;
        std::cout << "[DEBUG]   commonServices_: " << (commonServices_ ? "OK" : "NULL") << std::endl;
        std::cout << "[DEBUG]   dbManager_: " << (dbManager_ ? "OK" : "NULL") << std::endl;
        std::cout << "[DEBUG]   recognizer_: " << (recognizer_ ? "OK" : "NULL") << std::endl;
        std::cout << "[DEBUG]   serviceManager_: " << (serviceManager_ ? "OK" : "NULL") << std::endl;
        
        LOG_ERROR("One or more core service dependencies are null. Initialization failed.");
        return false;
    }
    std::cout << "[DEBUG] 核心服务依赖检查通过" << std::endl;

    // 初始化数据库管理器
    std::cout << "[DEBUG] 开始初始化数据库管理器..." << std::endl;
    bool dbInitialized = dbManager_->initialize();
    if (!dbInitialized) {
        std::cout << "[DEBUG] 数据库管理器初始化失败！" << std::endl;
        LOG_ERROR("Database manager initialization failed.");
        return false;
    }
    std::cout << "[DEBUG] 数据库管理器初始化成功" << std::endl;

    // 🔧 初始化元数据标准化器
    std::cout << "[DEBUG] 开始初始化元数据标准化器..." << std::endl;
    try {
        // 获取CRS服务用于初始化标准化器（可以为null，标准化器会处理）
        auto crsService = getCrsService();
        standardizer_ = std::make_shared<MetadataStandardizer>(crsService);
        std::cout << "[DEBUG] 元数据标准化器初始化成功" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[DEBUG] 元数据标准化器初始化失败: " << e.what() << std::endl;
        LOG_WARN("MetadataStandardizer initialization failed: {}", e.what());
        // 不阻止服务初始化，标准化器可选
    }

    LOG_INFO("MetadataService initialized successfully with all dependencies.");
    isInitialized_ = true;
    std::cout << "[DEBUG] MetadataServiceImpl::initialize() 完成，返回true" << std::endl;
    return true;
}

boost::future<AsyncResult<std::string>> MetadataServiceImpl::processFile(const std::string& filePath) {
    auto promise = std::make_shared<boost::promise<AsyncResult<std::string>>>();
    auto future = promise->get_future();
    
    // 使用统一异步框架提交任务
    if (commonServices_) {
        try {
            // 通过 CommonServicesFactory 获取异步执行器
            auto asyncExecutor = commonServices_->getAsyncExecutor();
            
            // 使用 std::thread 实现异步（保持架构兼容性）
            std::thread([this, filePath, promise]() {
                try {
                    auto result = processFileInternal(filePath);
                    promise->set_value(result);
                } catch (const std::exception& e) {
                    promise->set_value(AsyncResult<std::string>::failure(e.what()));
                }
            }).detach();
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to submit async task: {}", e.what());
            promise->set_value(AsyncResult<std::string>::failure(e.what()));
        }
    } else {
        // 同步回调作为备选方案
        try {
            auto result = processFileInternal(filePath);
            promise->set_value(result);
        } catch (const std::exception& e) {
            promise->set_value(AsyncResult<std::string>::failure(e.what()));
        }
    }
    
    return future;
}

boost::future<AsyncResult<void>> MetadataServiceImpl::receiveFileMetadataAsync(FileMetadata metadata) {
    auto promise = std::make_shared<boost::promise<AsyncResult<void>>>();
    auto future = promise->get_future();
    
    // 异步处理元数据接收
    std::thread([this, metadata, promise]() mutable {
        try {
            std::cout << "[DEBUG METADATA] 开始处理文件元数据: " << metadata.filePath << std::endl;
            
            // 🔧 关键修复：添加元数据标准化和分析步骤
            
            // 1. 首先进行元数据标准化（空间、时间、CRS分析）
            std::cout << "[DEBUG METADATA] 检查standardizer_指针: " << (standardizer_ ? "有效" : "空") << std::endl;
            if (standardizer_) {
                std::cout << "[DEBUG METADATA] 步骤1: 应用元数据标准化..." << std::endl;
                std::cout << "[DEBUG METADATA] 调用前 - 元数据格式: " << metadata.format << std::endl;
                std::cout << "[DEBUG METADATA] 调用前 - geographicDimensions数量: " << metadata.geographicDimensions.size() << std::endl;
                try {
                    std::cout << "[DEBUG METADATA] 即将调用 standardizer_->standardizeMetadata..." << std::endl;
                    auto result = standardizer_->standardizeMetadata(metadata, metadata.format);
                    std::cout << "[DEBUG METADATA] standardizeMetadata调用成功，准备赋值..." << std::endl;
                    metadata = result;
                    std::cout << "[DEBUG METADATA] 元数据标准化完成" << std::endl;
                    std::cout << "[DEBUG METADATA] 调用后 - geographicDimensions数量: " << metadata.geographicDimensions.size() << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "[DEBUG METADATA] 标准化失败: " << e.what() << std::endl;
                    LOG_WARN("元数据标准化失败: {}", e.what());
                }
            } else {
                std::cout << "[DEBUG METADATA] 警告：标准化器未初始化，跳过标准化步骤" << std::endl;
            }
            
            // 2. 填充缺失的关键字段
            std::cout << "[DEBUG METADATA] 步骤2: 填充基础元数据字段..." << std::endl;
            
            // 确保文件ID存在
            if (metadata.fileId.empty()) {
                metadata.fileId = generateMetadataId();
            }
            
            // 确保metadataId存在
            if (metadata.metadataId.empty()) {
                metadata.metadataId = metadata.fileId;
            }
            
            // 从文件路径提取文件名（如果缺失）
            if (metadata.fileName.empty() && !metadata.filePath.empty()) {
                metadata.fileName = std::filesystem::path(metadata.filePath).filename().string();
            }
            
            // 设置处理时间戳
            metadata.extractionTimestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            std::cout << "[DEBUG METADATA] 基础字段填充完成" << std::endl;
            
            // 3. 进行智能分类
            std::cout << "[DEBUG METADATA] 步骤3: 进行文件智能分类..." << std::endl;
            ClassificationResult classificationResult = recognizer_->classifyFile(metadata);
            metadata.primaryCategory = static_cast<oscean::core_services::DataType>(classificationResult.primaryCategory);
            metadata.classifications = classificationResult.tags;
            std::cout << "[DEBUG METADATA] 智能分类完成，主要类别: " << static_cast<int>(metadata.primaryCategory) << std::endl;
            
            // 4. 打印调试信息
            std::cout << "[DEBUG METADATA] 步骤4: 元数据处理结果摘要" << std::endl;
            std::cout << "  - 文件路径: " << metadata.filePath << std::endl;
            std::cout << "  - 变量数量: " << metadata.variables.size() << std::endl;
            std::cout << "  - 空间范围: [" << metadata.spatialCoverage.minX << ", " << metadata.spatialCoverage.minY 
                      << "] - [" << metadata.spatialCoverage.maxX << ", " << metadata.spatialCoverage.maxY << "]" << std::endl;
            std::cout << "  - 时间范围: " << metadata.temporalInfo.startTime << " - " << metadata.temporalInfo.endTime << std::endl;
            std::cout << "  - CRS信息: " << (metadata.crs.wkt.empty() ? "未设置" : "已设置") << std::endl;
            
            // 5. 将处理后的元数据保存到数据库
            std::cout << "[DEBUG METADATA] 步骤5: 保存到数据库..." << std::endl;
            auto saveFuture = dbManager_->storeFileMetadataAsync(metadata);
            auto saveResult = saveFuture.get();
            
            if(saveResult.isSuccess()){
                std::cout << "[DEBUG METADATA] 元数据保存成功: " << metadata.filePath << std::endl;
                promise->set_value(AsyncResult<void>::success());
            } else {
                std::cout << "[DEBUG METADATA] 元数据保存失败: " << saveResult.getError() << std::endl;
                promise->set_value(AsyncResult<void>::failure(saveResult.getError()));
            }
            
        } catch (const std::exception& e) {
            std::cout << "[DEBUG METADATA] 异常: " << e.what() << std::endl;
            LOG_ERROR("Failed to receive file metadata: {}", e.what());
            promise->set_value(AsyncResult<void>::failure(e.what()));
        }
    }).detach();
    
    return future;
}

boost::future<AsyncResult<std::vector<FileMetadata>>> MetadataServiceImpl::queryMetadataAsync(const QueryCriteria& criteria) {
    LOG_INFO("Forwarding queryMetadataAsync to dbManager");
    return dbManager_->queryMetadataAsync(criteria);
}

boost::future<AsyncResult<FileMetadata>> MetadataServiceImpl::queryByFilePathAsync(const std::string& filePath) {
    LOG_INFO("Forwarding queryByFilePathAsync to dbManager for path: {}", filePath);
    return dbManager_->queryByFilePathAsync(filePath);
}

boost::future<AsyncResult<std::vector<FileMetadata>>> MetadataServiceImpl::queryByCategoryAsync(
    DataType category,
    const std::optional<QueryCriteria>& additionalCriteria) {
    
    LOG_INFO("Forwarding queryByCategoryAsync to dbManager for category: {}", static_cast<int>(category));
    QueryCriteria criteria = additionalCriteria.value_or(QueryCriteria{});
    criteria.dataTypes.push_back(category);
    return dbManager_->queryMetadataAsync(criteria);
}

boost::future<AsyncResult<bool>> MetadataServiceImpl::deleteMetadataAsync(const std::string& metadataId) {
    LOG_INFO("Forwarding deleteMetadataAsync to dbManager for ID: {}", metadataId);
    return dbManager_->deleteMetadataAsync(metadataId);
}

boost::future<AsyncResult<bool>> MetadataServiceImpl::updateMetadataAsync(
    const std::string& metadataId, 
    const MetadataUpdate& update) {
    
    LOG_INFO("Forwarding updateMetadataAsync to dbManager for ID: {}", metadataId);
    return dbManager_->updateMetadataAsync(metadataId, update);
}

boost::future<AsyncResult<void>> MetadataServiceImpl::updateConfigurationAsync(const MetadataServiceConfiguration& config) {
    return boost::async(boost::launch::async, [this, config]() {
        try {
            updateConfiguration(config);
            return AsyncResult<void>::success();
        } catch (const std::exception& e) {
            LOG_ERROR("Update configuration failed: {}", e.what());
            return AsyncResult<void>::failure(e.what());
        }
    });
}

boost::future<AsyncResult<MetadataServiceConfiguration>> MetadataServiceImpl::getConfigurationAsync() {
    return boost::async(boost::launch::async, [this]() {
        try {
            MetadataServiceConfiguration config;
            // TODO: Implement actual configuration retrieval logic
            return AsyncResult<MetadataServiceConfiguration>::success(std::move(config));
        } catch (const std::exception& e) {
            LOG_ERROR("Get configuration failed: {}", e.what());
            return AsyncResult<MetadataServiceConfiguration>::failure(e.what());
        }
    });
}

std::string MetadataServiceImpl::getVersion() const {
    return "1.0.0";
}

bool MetadataServiceImpl::isReady() const {
    return isInitialized_ && dbManager_ && logger_;
}

// 添加缺失的方法实现
boost::future<std::vector<std::string>> MetadataServiceImpl::filterUnprocessedFilesAsync(
    const std::vector<std::string>& filePaths) {
    
    LOG_INFO("Filtering {} file paths against the database.", filePaths.size());
    return dbManager_->filterExistingFiles(filePaths);
}

boost::future<FileMetadata> MetadataServiceImpl::classifyAndEnrichAsync(
    const FileMetadata& metadata) {
    
    return boost::async(boost::launch::async, [this, metadata]() {
        LOG_INFO("Classifying and enriching metadata for file: {}", metadata.filePath);
        auto mutableMetadata = metadata;
        
        try {
            auto classificationResult = recognizer_->classifyFile(mutableMetadata);
            
            // 智能识别器的输出是语义分类，不直接影响primaryCategory（技术数据类型）
            // primaryCategory应该由数据读取器根据实际数据类型设置
            // 这里我们只是记录分类已完成
            
            LOG_DEBUG("File classification completed for: {}", mutableMetadata.filePath);
            return mutableMetadata;

        } catch (const std::exception& e) {
            LOG_ERROR("Exception during classification for {}: {}", metadata.filePath, e.what());
            return mutableMetadata;
        }
    });
}

boost::future<AsyncResult<bool>> MetadataServiceImpl::saveMetadataAsync(
    const FileMetadata& metadata) {
    
    LOG_INFO("Saving final metadata to database for file: {}", metadata.filePath);
    
    return boost::async(boost::launch::async, [this, metadata]() -> AsyncResult<bool> {
        try {
            auto storeFuture = dbManager_->storeFileMetadataAsync(metadata);
            auto storeResult = storeFuture.get();
            
            if (storeResult.isSuccess()) {
                return AsyncResult<bool>::success(true);
            } else {
                return AsyncResult<bool>::failure("Failed to store metadata: " + storeResult.getError());
            }
        } catch (const std::exception& e) {
            return AsyncResult<bool>::failure(e.what());
        }
    });
}

AsyncResult<std::string> MetadataServiceImpl::processFileInternal(const std::string& filePath) {
    if (!isInitialized_) {
        return AsyncResult<std::string>::failure("服务未初始化");
    }

    LOG_INFO("开始内部处理文件: {}", filePath);

    try {
        // 1. 提取元数据 - 修复构造函数调用（只需要1个参数）
        auto extractor = MetadataExtractor(commonServices_);
        // 注意：extract方法需要异步调用，这里需要修复为从data_access_service获取元数据
        // 在正确的架构中，metadata_service应该接收由data_access_service提取的元数据
        // 这里作为临时修复，我们创建一个基本的FileMetadata对象
        FileMetadata metadata;
        metadata.filePath = filePath;
        metadata.metadataId = generateMetadataId();
        LOG_DEBUG("步骤1/4: 原始元数据提取完成（临时实现）");

        // 2. CRS信息处理 (延迟获取CRS服务)
        auto crsService = getCrsService();
        if (crsService) {
            // ... (如果需要，可以在这里调用CRS服务)
            // 示例: metadata = crsService->enrich(metadata).get();
            LOG_DEBUG("步骤2/4: CRS服务可用，已处理CRS信息 (当前为占位符)");
        } else {
            LOG_WARN("CRS服务不可用，跳过CRS信息处理步骤");
        }

        // 3. 智能分类
        ClassificationResult classificationResult = recognizer_->classifyFile(metadata);
        // 修复DataType转换：metadata服务使用自己的DataType枚举
        metadata.primaryCategory = static_cast<::oscean::core_services::DataType>(classificationResult.primaryCategory);
        metadata.classifications = classificationResult.tags;
        LOG_DEBUG("步骤3/4: 文件智能分类完成");

        // 4. 存储到数据库
        auto storeFuture = dbManager_->storeFileMetadataAsync(metadata);
        auto storeResult = storeFuture.get();
        if (!storeResult.isSuccess()) {
            return AsyncResult<std::string>::failure("存储元数据失败: " + storeResult.getError());
        }
        LOG_DEBUG("步骤4/4: 元数据存储完成");

        LOG_INFO("文件处理成功: {}", filePath);
        return AsyncResult<std::string>::success(metadata.metadataId);

    } catch (const std::exception& e) {
        LOG_ERROR("处理文件时发生异常 {}: {}", filePath, e.what());
        return AsyncResult<std::string>::failure(e.what());
    }
}

void MetadataServiceImpl::updateConfiguration(const MetadataServiceConfiguration& config) {
    // TODO: 实现配置更新逻辑
    LOG_INFO("Configuration updated.");
}

std::string MetadataServiceImpl::generateMetadataId() const {
    // 使用boost::uuid生成唯一ID
    boost::uuids::random_generator gen;
    boost::uuids::uuid id = gen();
    return boost::uuids::to_string(id);
}

std::shared_ptr<ICrsService> MetadataServiceImpl::getCrsService() const {
    if (!crsService_) {
        if (!serviceManager_) {
            LOG_ERROR("服务管理器未初始化，无法获取CRS服务");
            return nullptr;
        }
        try {
            LOG_INFO("首次请求，正在延迟加载CRS服务...");
            crsService_ = serviceManager_->getService<ICrsService>();
            LOG_INFO("CRS服务延迟加载成功");
        } catch (const std::exception& e) {
            LOG_ERROR("延迟加载CRS服务失败: {}", e.what());
            // 返回nullptr，让调用者处理服务不可用的情况
            return nullptr;
        }
    }
    return crsService_;
}

// === 补全缺失的方法 ===

AsyncResult<std::vector<FileMetadata>> MetadataServiceImpl::queryByCategoryAsync(const std::string& category, const std::string& value) {
    LOG_INFO("Querying metadata by category: {} = {}", category, value);
    
    try {
        // 使用metadata服务的QueryCriteria，根据分类进行简单查询
        QueryCriteria criteria;
        // 注意：这里是简化实现，实际可能需要扩展QueryCriteria来支持自定义属性过滤
        // 当前设计中QueryCriteria主要支持标准查询字段
        
        auto queryFuture = dbManager_->queryMetadataAsync(criteria);
        auto queryResult = queryFuture.get();
        
        if (queryResult.isSuccess()) {
            // 在结果中进一步过滤指定的category和value
            auto allResults = queryResult.getData();
            std::vector<FileMetadata> filteredResults;
            
            for (const auto& metadata : allResults) {
                auto it = metadata.metadata.find(category);
                if (it != metadata.metadata.end() && it->second == value) {
                    filteredResults.push_back(metadata);
                }
            }
            
            return AsyncResult<std::vector<FileMetadata>>::success(filteredResults);
        } else {
            return AsyncResult<std::vector<FileMetadata>>::failure(queryResult.getError());
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Category query failed: {}", e.what());
        return AsyncResult<std::vector<FileMetadata>>::failure(e.what());
    }
}

AsyncResult<bool> MetadataServiceImpl::deleteMetadataByFilePathAsync(const std::string& filePath) {
    LOG_INFO("Deleting metadata by file path: {}", filePath);
    
    try {
        // 先根据文件路径查询到metadataId
        auto queryFuture = dbManager_->queryByFilePathAsync(filePath);
        auto queryResult = queryFuture.get();
        
        if (!queryResult.isSuccess()) {
            return AsyncResult<bool>::failure("File not found: " + filePath);
        }
        
        auto metadata = queryResult.getData();
        auto deleteFuture = dbManager_->deleteMetadataAsync(metadata.metadataId);
        auto deleteResult = deleteFuture.get();
        
        if (deleteResult.isSuccess()) {
            return AsyncResult<bool>::success(true);
        } else {
            return AsyncResult<bool>::failure(deleteResult.getError());
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Delete by file path failed: {}", e.what());
        return AsyncResult<bool>::failure(e.what());
    }
}

// 移除了类型转换函数 - 架构修正：语义分类和技术数据类型应该分开处理

} // namespace impl
} // namespace metadata
} // namespace core_services
} // namespace oscean 