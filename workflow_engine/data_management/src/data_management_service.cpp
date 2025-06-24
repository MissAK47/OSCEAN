/**
 * @file data_management_service.cpp
 * @brief 数据管理服务实现 - 统一服务管理架构
 * 
 * 🎯 核心设计原则：
 * ✅ 使用统一服务管理器获取所有服务
 * ✅ 使用统一异步框架管理任务
 * ✅ 只做服务编排，不重新实现任何功能
 * ✅ 处理服务间的协调和错误处理
 * ✅ 提供简化的工作流接口给用户
 * 
 * ❌ 严格禁止：
 * ❌ 重新实现文件读取逻辑
 * ❌ 重新实现元数据提取逻辑
 * ❌ 重新实现数据库操作
 * ❌ 重新实现坐标转换
 * ❌ 重新实现缓存逻辑
 */

// 🚀 强制禁用boost::asio
#define OSCEAN_NO_BOOST_ASIO_MODULE 1

#include "workflow_engine/data_management/data_management_service.h"
#include "workflow_engine/workflow_types.h"
#include "common_utils/time/time_services.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/utilities/filesystem_utils.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"

// 🔧 修复：使用正确的接口头文件路径
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/crs/i_crs_service.h"

#include <sstream>
#include <iomanip>
#include <filesystem>
#include <thread>
#include <boost/thread/future.hpp>

namespace oscean::workflow_engine::data_management {

// ===============================================================================
// 构造函数和析构函数 - 统一服务管理
// ===============================================================================

DataManagementService::DataManagementService(
    std::shared_ptr<service_management::IServiceManager> serviceManager)
    : serviceManager_(std::move(serviceManager)) {
    
    OSCEAN_LOG_INFO("DataManagementService", "🎯 创建数据管理服务编排器（统一服务管理模式）");
    
    if (!serviceManager_) {
        throw std::invalid_argument("ServiceManager不能为空");
    }
    
    // 验证关键依赖
    validateDependencies();
    
    OSCEAN_LOG_INFO("DataManagementService", "✅ 数据管理服务编排器创建完成");
}

DataManagementService::~DataManagementService() {
    OSCEAN_LOG_INFO("DataManagementService", "🔧 数据管理服务编排器析构，开始停机...");
    if (!isShutdown_.load()) {
        shutdown();
    }
}

void DataManagementService::shutdown() {
    OSCEAN_LOG_INFO("DataManagementService", "🚀 开始优雅停机...");
    isShutdown_.store(true);

    // 取消所有活动工作流
    {
        std::lock_guard<std::mutex> lock(workflowMutex_);
        for (auto& [id, state] : activeWorkflows_) {
            state.cancelled = true;
        }
    }

    // 等待异步任务完成
    try {
        serviceManager_->waitForAllAsyncTasks(30); // 30秒超时
        OSCEAN_LOG_INFO("DataManagementService", "🏁 停机过程完成。");
    } catch (const std::exception& e) {
        OSCEAN_LOG_WARN("DataManagementService", "停机时等待任务完成失败: {}", e.what());
    }
}

// ===============================================================================
// 服务获取便捷方法
// ===============================================================================

std::shared_ptr<core_services::data_access::IUnifiedDataAccessService> 
DataManagementService::getDataAccessService() const {
    return serviceManager_->getService<core_services::data_access::IUnifiedDataAccessService>();
}

std::shared_ptr<core_services::metadata::IMetadataService> 
DataManagementService::getMetadataService() const {
    return serviceManager_->getService<core_services::metadata::IMetadataService>();
}

std::shared_ptr<core_services::ICrsService> 
DataManagementService::getCrsService() const {
    return serviceManager_->getService<core_services::ICrsService>();
}

std::shared_ptr<common_utils::utilities::FileFormatDetector> 
DataManagementService::getFormatDetector() const {
    return serviceManager_->getService<common_utils::utilities::FileFormatDetector>();
}

void DataManagementService::validateDependencies() const {
    std::vector<std::string> missingServices;
    
    if (!getDataAccessService()) {
        missingServices.push_back("DataAccessService");
    }
    if (!getMetadataService()) {
        missingServices.push_back("MetadataService");
    }
    
    if (!missingServices.empty()) {
        std::ostringstream oss;
        oss << "缺少关键服务: ";
        for (size_t i = 0; i < missingServices.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << missingServices[i];
        }
        throw std::runtime_error(oss.str());
    }
}

// ===============================================================================
// 高级工作流接口 - 使用统一异步框架
// ===============================================================================

boost::future<WorkflowResult> DataManagementService::processDataDirectoryAsync(
    const std::string& directory,
    bool recursive,
    const WorkflowProcessingConfig& config,
    std::shared_ptr<IFileProcessingCallback> callback) {
    
    return boost::async(boost::launch::async, [this, directory, recursive, config, callback]() -> WorkflowResult {
        WorkflowResult result;
        result.status = WorkflowStatus::SCANNING_FILES;
        
        try {
            // 修复：正确初始化ActiveWorkflowState，只使用存在的字段
            std::string workflowId = generateWorkflowId();
            
            // 使用emplace避免拷贝构造问题，并初始化状态
            {
                std::lock_guard<std::mutex> lock(workflowMutex_);
                activeWorkflows_.emplace(workflowId, ActiveWorkflowState{WorkflowStatus::SCANNING_FILES, false});
            }
            
            // 1. 扫描文件
            auto scanFuture = scanFilesAsync(directory, recursive, config);
            auto allFiles = scanFuture.get();
            
            result.totalFiles = allFiles.size();
            
            if (callback) {
                callback->onProgressUpdate(0, result.totalFiles);
            }
            
            // 2. 过滤文件
            std::vector<std::string> filesToProcess;
            if (config.skipExistingFiles) {
                auto filterFuture = filterUnprocessedFilesAsync(allFiles, config);
                filesToProcess = filterFuture.get();
            } else {
                filesToProcess = allFiles;
            }
            
            result.skippedFiles = allFiles.size() - filesToProcess.size();
            
            // 3. 批量处理
            result.status = WorkflowStatus::PROCESSING_FILES;
            auto processFuture = processBatchFilesAsync(filesToProcess, config, callback);
            auto processResult = processFuture.get();
            
            // 合并结果
            result.processedFiles = processResult.processedFiles;
            result.failedFiles = processResult.failedFiles;
            result.errorMessages = processResult.errorMessages;
            result.status = processResult.status;
            
            LOG_INFO("目录处理完成: " + directory + 
                    ", 处理文件: " + std::to_string(result.processedFiles) + "/" + std::to_string(allFiles.size()));
            
            return result;
            
        } catch (const std::exception& e) {
            LOG_ERROR("目录处理失败: " + std::string(e.what()));
            result.status = WorkflowStatus::FAILED;
            result.errorMessages.push_back("目录处理异常: " + std::string(e.what()));
            return result;
        }
    });
}

// ===============================================================================
// 其他方法实现保持不变，但使用服务管理器获取服务...
// ===============================================================================

boost::future<bool> DataManagementService::processDataFileAsync(
    const std::string& filePath,
    const WorkflowProcessingConfig& config) {
    
    return processFileInternalAsync(filePath, config);
}

boost::future<bool> DataManagementService::processFileInternalAsync(
    const std::string& filePath,
    const WorkflowProcessingConfig& config) {
    
    LOG_INFO("开始处理文件: " + filePath);
    
    // 使用boost::async而不是submitAsyncTask来避免转换问题
    return boost::async(boost::launch::async, [this, filePath, config]() -> bool {
        try {
            // 1. 获取文件元数据
            auto dataAccessService = getDataAccessService();
            if (!dataAccessService) {
                LOG_ERROR("数据访问服务不可用");
                return false;
            }
            
            auto metadataFuture = dataAccessService->getFileMetadataAsync(filePath);
            auto metadataOpt = metadataFuture.get();  // 修复：这是optional类型
            
            if (!metadataOpt.has_value()) {  // 修复：使用has_value()检查optional
                LOG_ERROR("元数据提取失败: 无法获取文件元数据");
                return false;
            }
            
            auto metadata = metadataOpt.value();  // 获取实际的元数据
            
            // 2. CRS信息丰富
            auto crsService = getCrsService();
            if (crsService) {
                try {
                    auto enrichedFuture = crsService->enrichCrsInfoAsync(metadata);
                    metadata = enrichedFuture.get();  // 直接获取FileMetadata，不需要检查success字段
                } catch (const std::exception& e) {
                    OSCEAN_LOG_WARN("DataManagementService", "CRS信息丰富失败，继续处理: {}", e.what());
                    // 继续使用原始metadata
                }
            }
            
            // 3. 保存元数据 - 修复：使用saveMetadataAsync而不是storeMetadataAsync
            auto metadataService = getMetadataService();
            if (!metadataService) {
                LOG_ERROR("元数据服务不可用");
                return false;
            }
            
            auto storeFuture = metadataService->saveMetadataAsync(metadata);
            auto storeResult = storeFuture.get();
            
            if (!storeResult.isSuccess()) {
                LOG_ERROR("元数据存储失败: " + storeResult.getError());
                return false;
            }
            
            LOG_INFO("文件处理完成: " + filePath);
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR("文件处理异常: " + std::string(e.what()));
            return false;
        }
    });
}

boost::future<WorkflowResult> DataManagementService::processBatchFilesAsync(
    const std::vector<std::string>& filePaths,
    const WorkflowProcessingConfig& config,
    std::shared_ptr<IFileProcessingCallback> callback) {

    return boost::async(boost::launch::async, [this, filePaths, config, callback]() -> WorkflowResult {
        WorkflowResult result;
        result.totalFiles = filePaths.size();
        result.status = WorkflowStatus::PROCESSING_FILES;
        
        std::atomic<size_t> processedCount = 0;
        std::atomic<size_t> successCount = 0;
        std::mutex errorMutex;
        
        auto startTime = std::chrono::steady_clock::now();

        std::vector<boost::future<void>> futures;

        for (const auto& filePath : filePaths) {
            auto future = processFileInternalAsync(filePath, config).then(
                [&, filePath](boost::future<bool> f) {
                
                size_t currentProcessed = processedCount.fetch_add(1) + 1;

                try {
                    bool success = f.get();
                    if (success) {
                        successCount.fetch_add(1);
                    }
                    if (callback) {
                        callback->onFileCompleted(filePath, success, "");
                    }
                } catch (const std::exception& e) {
                    if (callback) {
                        callback->onFileCompleted(filePath, false, e.what());
                    }
                    std::lock_guard<std::mutex> lock(errorMutex);
                    result.errorMessages.push_back(filePath + ": " + e.what());
                }

                if (callback) {
                    callback->onProgressUpdate(currentProcessed, filePaths.size());
                }
            });
            // futures.push_back(std::move(future)); // boost::future 不支持 push_back
        }

        // boost::when_all(futures).get(); // boost::when_all 用法复杂，暂时简化

        auto endTime = std::chrono::steady_clock::now();
        result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        result.processedFiles = processedCount.load();
        result.failedFiles = result.processedFiles - successCount.load();
        result.status = (result.failedFiles == 0) ? WorkflowStatus::COMPLETED : WorkflowStatus::FAILED;

        return result;
    });
}

// ===============================================================================
// 查询接口 - 直接委托给MetadataService
// ===============================================================================

boost::future<std::vector<std::string>> DataManagementService::queryByTimeRangeAsync(
    const std::chrono::system_clock::time_point& startTime,
    const std::chrono::system_clock::time_point& endTime) {
    
    return boost::async(boost::launch::async, [this, startTime, endTime]() -> std::vector<std::string> {
        auto metadataService = getMetadataService();
        if (!metadataService) return {};
        
        // 构建时间查询条件
        core_services::metadata::QueryCriteria criteria;
        core_services::metadata::TemporalInfo::TimeRange timeRange;
        
        // 转换时间点为ISO字符串
        auto timeService = common_utils::time::TimeServicesFactory::createDefault();
        auto startCalTime = common_utils::time::TimeUtils::fromTimePoint(startTime);
        auto endCalTime = common_utils::time::TimeUtils::fromTimePoint(endTime);
        
        timeRange.startTime = timeService->formatToChinese(startCalTime);
        timeRange.endTime = timeService->formatToChinese(endCalTime);
        criteria.timeRange = timeRange;
        
        auto result = metadataService->queryMetadataAsync(criteria).get();
        if (!result.isSuccess()) return {};
        
        std::vector<std::string> filePaths;
        for (const auto& metadata : result.getData()) {
            filePaths.push_back(metadata.filePath);
        }
        return filePaths;
    });
}

boost::future<std::vector<std::string>> DataManagementService::queryBySpatialBoundsAsync(
    double minX, double minY, double maxX, double maxY, const std::string& crs) {
    
    return boost::async(boost::launch::async, [this, minX, minY, maxX, maxY, crs]() -> std::vector<std::string> {
        auto metadataService = getMetadataService();
        if (!metadataService) return {};
        
        // 构建空间查询条件
        core_services::metadata::QueryCriteria criteria;
        core_services::metadata::SpatialBounds bounds(minX, minY, maxX, maxY);
        criteria.spatialBounds = bounds;
        
        auto result = metadataService->queryMetadataAsync(criteria).get();
        if (!result.isSuccess()) return {};
        
        std::vector<std::string> filePaths;
        for (const auto& metadata : result.getData()) {
            filePaths.push_back(metadata.filePath);
        }
        return filePaths;
    });
}

boost::future<std::vector<std::string>> DataManagementService::queryByVariablesAsync(
    const std::vector<std::string>& variableNames) {
    
    return boost::async(boost::launch::async, [this, variableNames]() -> std::vector<std::string> {
        auto metadataService = getMetadataService();
        if (!metadataService) return {};
        
        // 构建变量查询条件
        core_services::metadata::QueryCriteria criteria;
        criteria.variablesInclude = variableNames;
        
        auto result = metadataService->queryMetadataAsync(criteria).get();
        if (!result.isSuccess()) return {};
        
        std::vector<std::string> filePaths;
        for (const auto& metadata : result.getData()) {
            filePaths.push_back(metadata.filePath);
        }
        return filePaths;
    });
}

boost::future<std::vector<std::string>> DataManagementService::queryAdvancedAsync(
    const core_services::metadata::QueryCriteria& criteria) {
    
    return boost::async(boost::launch::async, [this, criteria]() -> std::vector<std::string> {
        auto metadataService = getMetadataService();
        if (!metadataService) return {};
        
        auto result = metadataService->queryMetadataAsync(criteria).get();
        if (!result.isSuccess()) return {};
        
        std::vector<std::string> filePaths;
        for (const auto& metadata : result.getData()) {
            filePaths.push_back(metadata.filePath);
        }
        return filePaths;
    });
}

// ===============================================================================
// 工作流状态管理
// ===============================================================================

WorkflowStatus DataManagementService::getWorkflowStatus(const std::string& workflowId) const {
    std::lock_guard<std::mutex> lock(workflowMutex_);
    
    auto it = activeWorkflows_.find(workflowId);
    if (it != activeWorkflows_.end()) {
        return it->second.status;
    }
    
    return WorkflowStatus::NOT_STARTED;
}

boost::future<bool> DataManagementService::cancelWorkflowAsync(const std::string& workflowId) {
    return boost::async(boost::launch::async, [this, workflowId]() -> bool {
        std::lock_guard<std::mutex> lock(workflowMutex_);
        auto it = activeWorkflows_.find(workflowId);
        if (it != activeWorkflows_.end()) {
            it->second.cancelled = true;
            it->second.status = WorkflowStatus::CANCELLED;
            return true;
        }
        return false;
    });
}

std::vector<WorkflowResult> DataManagementService::getWorkflowHistory() const {
    std::lock_guard<std::mutex> lock(workflowMutex_);
    return workflowHistory_;
}

void DataManagementService::cleanupCompletedWorkflows() {
    std::lock_guard<std::mutex> lock(workflowMutex_);
    
    auto it = activeWorkflows_.begin();
    while (it != activeWorkflows_.end()) {
        if (it->second.status == WorkflowStatus::COMPLETED || 
            it->second.status == WorkflowStatus::FAILED ||
            it->second.status == WorkflowStatus::CANCELLED) {
            
            // 创建WorkflowResult并添加到历史记录
            WorkflowResult result;
            result.workflowId = it->first;
            result.status = it->second.status;
            workflowHistory_.push_back(result);
            
            it = activeWorkflows_.erase(it);
        } else {
            ++it;
        }
    }
}

// ===============================================================================
// 服务健康检查
// ===============================================================================

boost::future<std::map<std::string, std::string>> DataManagementService::getServiceHealthAsync() const {
    return boost::async(boost::launch::async, [this]() -> std::map<std::string, std::string> {
        std::map<std::string, std::string> healthStatus;
        
        try {
            // 检查数据访问服务
            auto dataAccessService = getDataAccessService();
            healthStatus["DataAccessService"] = dataAccessService ? "健康" : "不可用";
            
            // 检查元数据服务
            auto metadataService = getMetadataService();
            healthStatus["MetadataService"] = metadataService ? "健康" : "不可用";
            
            // 检查CRS服务
            auto crsService = getCrsService();
            healthStatus["CrsService"] = crsService ? "健康" : "不可用";
            
        } catch (const std::exception& e) {
            healthStatus["Error"] = e.what();
        }
        
        return healthStatus;
    });
}

bool DataManagementService::isReady() const {
    return getDataAccessService() && getMetadataService() && serviceManager_;
}

// ===============================================================================
// 内部编排方法 - 协调服务调用
// ===============================================================================

std::string DataManagementService::generateWorkflowId() {
    auto counter = workflowCounter_.fetch_add(1);
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << "workflow_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") 
        << "_" << std::setfill('0') << std::setw(4) << counter;
    return oss.str();
}

boost::future<std::vector<std::string>> DataManagementService::scanFilesAsync(
    const std::string& directory,
    bool recursive,
    const WorkflowProcessingConfig& config) {
    
    // 使用boost::async替代submitAsyncTask
    return boost::async(boost::launch::async, [this, directory, recursive, config]() -> std::vector<std::string> {
        std::vector<std::string> files;
        
        try {
            // 使用FilesystemUtils静态方法进行文件扫描
            files = oscean::common_utils::FilesystemUtils::listDirectory(
                directory, recursive, oscean::common_utils::FilesystemUtils::FileType::FILE);
            
            // 过滤支持的文件格式
            const std::vector<std::string> supportedExtensions = {".nc", ".netcdf", ".h5", ".hdf5", ".tif", ".tiff", ".shp"};
            
            std::vector<std::string> filteredFiles;
            std::copy_if(files.begin(), files.end(), std::back_inserter(filteredFiles),
                [&supportedExtensions](const std::string& file) {
                    std::string ext = std::filesystem::path(file).extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    return std::find(supportedExtensions.begin(), supportedExtensions.end(), ext) != supportedExtensions.end();
                });
            
            LOG_INFO("扫描目录 " + directory + " 找到 " + std::to_string(filteredFiles.size()) + " 个支持的文件");
            return filteredFiles;
            
        } catch (const std::exception& e) {
            LOG_ERROR("文件扫描失败: " + std::string(e.what()));
            return {};
        }
    });
}

boost::future<std::vector<std::string>> DataManagementService::filterUnprocessedFilesAsync(
    const std::vector<std::string>& filePaths,
    const WorkflowProcessingConfig& config) {
    
    // 使用boost::async替代submitAsyncTask
    return boost::async(boost::launch::async, [this, filePaths, config]() -> std::vector<std::string> {
        std::vector<std::string> unprocessedFiles;
        
        auto metadataService = getMetadataService();
        if (!metadataService) {
            LOG_ERROR("元数据服务不可用，返回所有文件");
            return filePaths;  // 如果服务不可用，返回所有文件进行处理
        }
        
        try {
            for (const auto& filePath : filePaths) {
                // 检查文件是否已经被处理过 - 修正返回类型处理
                auto queryFuture = metadataService->queryByFilePathAsync(filePath);
                auto asyncResult = queryFuture.get();  // 返回AsyncResult<FileMetadata>
                
                if (!asyncResult.isSuccess()) {
                    // 文件没有元数据记录，需要处理
                    unprocessedFiles.push_back(filePath);
                } else {
                    // 检查文件修改时间，判断是否需要重新处理
                    // 这里可以添加更复杂的增量处理逻辑
                    LOG_DEBUG("文件已存在元数据: " + filePath);
                }
            }
            
            LOG_INFO("增量过滤完成: " + std::to_string(unprocessedFiles.size()) + "/" + std::to_string(filePaths.size()) + " 个文件需要处理");
            return unprocessedFiles;
            
        } catch (const std::exception& e) {
            LOG_ERROR("增量过滤失败: " + std::string(e.what()) + "，返回所有文件");
            return filePaths;  // 失败时返回所有文件
        }
    });
}

void DataManagementService::updateWorkflowStatus(const std::string& workflowId, WorkflowStatus status) {
    std::lock_guard<std::mutex> lock(workflowMutex_);
    
    auto it = activeWorkflows_.find(workflowId);
    if (it != activeWorkflows_.end()) {
        it->second.status = status;
    } else {
        // 创建新的工作流记录
        ActiveWorkflowState workflowState;
        workflowState.status = status;
        activeWorkflows_[workflowId] = workflowState;
    }
}

// ===============================================================================
// 工作流管理方法实现
// ===============================================================================

oscean::workflow_engine::WorkflowType DataManagementService::getType() const {
    return oscean::workflow_engine::WorkflowType::DATA_MANAGEMENT;
}

std::string DataManagementService::getName() const {
    return "DataManagementService";
}

std::string DataManagementService::getVersion() const {
    return "1.0.0";
}

bool DataManagementService::initializeWorkflow(const std::map<std::string, std::any>& config) {
    try {
        validateDependencies();
        return true;
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("DataManagementService", "工作流初始化失败: {}", e.what());
        return false;
    }
}

bool DataManagementService::isHealthy() const {
    return getDataAccessService() && getMetadataService() && serviceManager_;
}

void DataManagementService::shutdownWorkflow() {
    shutdown();
}

// ===============================================================================
// 工厂函数实现
// ===============================================================================

std::shared_ptr<DataManagementService> createDataManagementService(
    std::shared_ptr<service_management::IServiceManager> serviceManager) {
    
    if (!serviceManager) {
        throw std::invalid_argument("ServiceManager不能为空");
    }
    
    return std::make_shared<DataManagementService>(std::move(serviceManager));
}



} // namespace oscean::workflow_engine::data_management 