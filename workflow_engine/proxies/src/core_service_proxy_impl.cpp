/**
 * @file core_service_proxy_impl.cpp
 * @brief 核心服务代理具体实现
 * @author OSCEAN Team
 * @date 2024
 */

#include "core_service_proxy_impl.h"
#include "common_utils/utilities/logging.h"
#include <boost/thread/future.hpp>
#include <boost/thread/thread_pool.hpp>
#include <chrono>
#include <algorithm>

namespace oscean::workflow_engine::proxies {

CoreServiceProxyImpl::CoreServiceProxyImpl()
    : factory_(std::make_shared<CoreServiceFactory>())
    , isServiceHealthy_(true)
    , circuitBreakerThreshold_(5)
    , failureCount_(0) {
    // 🔧 不再创建独立线程池，改为从环境变量检查模式
    const char* runMode = std::getenv("OSCEAN_RUN_MODE");
    if (runMode && std::string(runMode) == "SINGLE_THREAD") {
        // 单线程模式：不创建线程池
        threadPool_ = nullptr;
        LOG_INFO("CoreServiceProxy运行在单线程模式");
    } else {
        // 生产模式：使用小型线程池
        threadPool_ = std::make_unique<boost::thread_pool>(2);  // 减少线程数
        LOG_INFO("CoreServiceProxy创建了最小线程池(2线程)");
    }
}

CoreServiceProxyImpl::~CoreServiceProxyImpl() {
    if (threadPool_) {
        threadPool_->close();
        threadPool_->join();
    }
}

// === 服务管理 ===

bool CoreServiceProxyImpl::initialize(const std::map<std::string, std::any>& config) {
    try {
        // 初始化服务连接配置
        loadServiceConfiguration(config);
        
        // 初始化统计信息
        serviceStats_["data_access_service"] = ServiceCallStats{};
        serviceStats_["metadata_service"] = ServiceCallStats{};
        
        // 设置默认超时和重试策略
        serviceTimeouts_["data_access_service"] = std::chrono::milliseconds(30000);
        serviceTimeouts_["metadata_service"] = std::chrono::milliseconds(15000);
        
        retryPolicies_["data_access_service"] = {3, std::chrono::milliseconds(1000)};
        retryPolicies_["metadata_service"] = {3, std::chrono::milliseconds(500)};
        
        initialized_ = true;
        OSCEAN_LOG_INFO("CoreServiceProxy", "核心服务代理初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("CoreServiceProxy", "初始化失败: {}", e.what());
        return false;
    }
}

ServiceAvailability CoreServiceProxyImpl::checkServiceAvailability(const std::string& serviceName) {
    std::lock_guard<std::mutex> lock(serviceStatusMutex_);
    
    try {
        if (serviceName == "data_access_service") {
            return checkDASAvailability();
        } else if (serviceName == "metadata_service") {
            return checkMDSAvailability();
        } else {
            OSCEAN_LOG_WARN("CoreServiceProxy", "未知服务: {}", serviceName);
            return ServiceAvailability::UNKNOWN;
        }
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("CoreServiceProxy", "检查服务可用性失败 [{}]: {}", serviceName, e.what());
        return ServiceAvailability::UNAVAILABLE;
    }
}

ServiceCallStats CoreServiceProxyImpl::getServiceStats(const std::string& serviceName) const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    auto it = serviceStats_.find(serviceName);
    return (it != serviceStats_.end()) ? it->second : ServiceCallStats{};
}

// === Data Access Service 代理方法 ===

boost::future<std::optional<oscean::core_services::FileMetadata>> 
CoreServiceProxyImpl::getFileMetadataFromDAS(const std::string& filePath) {
    return executeServiceCall<std::optional<oscean::core_services::FileMetadata>>(
        "data_access_service", 
        "getFileMetadata",
        [this, filePath]() -> std::optional<oscean::core_services::FileMetadata> {
            if (!dataAccessService_) {
                throw std::runtime_error("Data Access Service 未连接");
            }
            
            try {
                // 调用DAS获取文件元数据
                auto metadata = dataAccessService_->extractFileMetadata(filePath);
                return metadata;
            } catch (const std::exception& e) {
                OSCEAN_LOG_WARN("CoreServiceProxy", "DAS获取文件元数据失败 [{}]: {}", filePath, e.what());
                return std::nullopt;
            }
        });
}

boost::future<std::shared_ptr<oscean::core_services::data_access::api::GridData>>
CoreServiceProxyImpl::readGridDataFromDAS(const oscean::core_services::data_access::api::ReadGridDataRequest& request) {
    return executeServiceCall<std::shared_ptr<oscean::core_services::data_access::api::GridData>>(
        "data_access_service",
        "readGridData",
        [this, request]() -> std::shared_ptr<oscean::core_services::data_access::api::GridData> {
            if (!dataAccessService_) {
                throw std::runtime_error("Data Access Service 未连接");
            }
            
            return dataAccessService_->readGridData(request);
        });
}

boost::future<oscean::core_services::data_access::api::FeatureCollection>
CoreServiceProxyImpl::readFeatureCollectionFromDAS(const oscean::core_services::data_access::api::ReadFeatureCollectionRequest& request) {
    return executeServiceCall<oscean::core_services::data_access::api::FeatureCollection>(
        "data_access_service",
        "readFeatureCollection",
        [this, request]() -> oscean::core_services::data_access::api::FeatureCollection {
            if (!dataAccessService_) {
                throw std::runtime_error("Data Access Service 未连接");
            }
            
            return dataAccessService_->readFeatureCollection(request);
        });
}

boost::future<bool> 
CoreServiceProxyImpl::checkVariableExistsInDAS(const std::string& filePath, const std::string& variableName) {
    return executeServiceCall<bool>(
        "data_access_service",
        "checkVariableExists",
        [this, filePath, variableName]() -> bool {
            if (!dataAccessService_) {
                throw std::runtime_error("Data Access Service 未连接");
            }
            
            return dataAccessService_->hasVariable(filePath, variableName);
        });
}

boost::future<std::vector<oscean::core_services::MetadataEntry>>
CoreServiceProxyImpl::getVariableAttributesFromDAS(const std::string& filePath, const std::string& variableName) {
    return executeServiceCall<std::vector<oscean::core_services::MetadataEntry>>(
        "data_access_service",
        "getVariableAttributes",
        [this, filePath, variableName]() -> std::vector<oscean::core_services::MetadataEntry> {
            if (!dataAccessService_) {
                throw std::runtime_error("Data Access Service 未连接");
            }
            
            return dataAccessService_->getVariableAttributes(filePath, variableName);
        });
}

boost::future<std::vector<oscean::core_services::MetadataEntry>>
CoreServiceProxyImpl::getGlobalAttributesFromDAS(const std::string& filePath) {
    return executeServiceCall<std::vector<oscean::core_services::MetadataEntry>>(
        "data_access_service",
        "getGlobalAttributes",
        [this, filePath]() -> std::vector<oscean::core_services::MetadataEntry> {
            if (!dataAccessService_) {
                throw std::runtime_error("Data Access Service 未连接");
            }
            
            return dataAccessService_->getGlobalAttributes(filePath);
        });
}

// === Metadata Service 代理方法 ===

boost::future<std::string> 
CoreServiceProxyImpl::recognizeFileWithMDS(const std::string& filePath) {
    return executeServiceCall<std::string>(
        "metadata_service",
        "recognizeFile",
        [this, filePath]() -> std::string {
            if (!metadataService_) {
                throw std::runtime_error("Metadata Service 未连接");
            }
            
            return metadataService_->recognizeFileAsync(filePath).get();
        });
}

boost::future<std::string>
CoreServiceProxyImpl::storeMetadataToMDS(const oscean::core_services::metadata::ExtractedMetadata& metadata) {
    return executeServiceCall<std::string>(
        "metadata_service",
        "storeMetadata",
        [this, metadata]() -> std::string {
            if (!metadataService_) {
                throw std::runtime_error("Metadata Service 未连接");
            }
            
            return metadataService_->storeMetadata(metadata);
        });
}

boost::future<std::vector<oscean::core_services::metadata::MetadataEntry>>
CoreServiceProxyImpl::queryMetadataFromMDS(const oscean::core_services::metadata::MultiDimensionalQueryCriteria& criteria) {
    return executeServiceCall<std::vector<oscean::core_services::metadata::MetadataEntry>>(
        "metadata_service",
        "queryMetadata",
        [this, criteria]() -> std::vector<oscean::core_services::metadata::MetadataEntry> {
            if (!metadataService_) {
                throw std::runtime_error("Metadata Service 未连接");
            }
            
            return metadataService_->queryMetadata(criteria);
        });
}

boost::future<bool>
CoreServiceProxyImpl::updateMetadataInMDS(const std::string& metadataId, 
                                         const oscean::core_services::metadata::MetadataUpdate& update) {
    return executeServiceCall<bool>(
        "metadata_service",
        "updateMetadata",
        [this, metadataId, update]() -> bool {
            if (!metadataService_) {
                throw std::runtime_error("Metadata Service 未连接");
            }
            
            return metadataService_->updateMetadata(metadataId, update);
        });
}

boost::future<bool>
CoreServiceProxyImpl::deleteMetadataFromMDS(const std::string& metadataId) {
    return executeServiceCall<bool>(
        "metadata_service",
        "deleteMetadata",
        [this, metadataId]() -> bool {
            if (!metadataService_) {
                throw std::runtime_error("Metadata Service 未连接");
            }
            
            return metadataService_->deleteMetadata(metadataId);
        });
}

// === 批量操作 ===

boost::future<std::vector<std::optional<oscean::core_services::FileMetadata>>>
CoreServiceProxyImpl::getFileMetadataBatchFromDAS(const std::vector<std::string>& filePaths) {
    return executeServiceCall<std::vector<std::optional<oscean::core_services::FileMetadata>>>(
        "data_access_service",
        "getFileMetadataBatch",
        [this, filePaths]() -> std::vector<std::optional<oscean::core_services::FileMetadata>> {
            std::vector<std::optional<oscean::core_services::FileMetadata>> results;
            results.reserve(filePaths.size());
            
            for (const auto& filePath : filePaths) {
                try {
                    auto metadata = getFileMetadataFromDAS(filePath).get();
                    results.push_back(metadata);
                } catch (const std::exception& e) {
                    OSCEAN_LOG_WARN("CoreServiceProxy", "批量获取元数据失败 [{}]: {}", filePath, e.what());
                    results.push_back(std::nullopt);
                }
            }
            
            return results;
        });
}

boost::future<std::vector<std::string>>
CoreServiceProxyImpl::recognizeFilesBatchWithMDS(const std::vector<std::string>& filePaths) {
    return executeServiceCall<std::vector<std::string>>(
        "metadata_service",
        "recognizeFilesBatch",
        [this, filePaths]() -> std::vector<std::string> {
            std::vector<std::string> results;
            results.reserve(filePaths.size());
            
            for (const auto& filePath : filePaths) {
                try {
                    auto recognitionId = recognizeFileWithMDS(filePath).get();
                    results.push_back(recognitionId);
                } catch (const std::exception& e) {
                    OSCEAN_LOG_WARN("CoreServiceProxy", "批量文件识别失败 [{}]: {}", filePath, e.what());
                    results.push_back("");
                }
            }
            
            return results;
        });
}

boost::future<std::vector<std::string>>
CoreServiceProxyImpl::storeMetadataBatchToMDS(const std::vector<oscean::core_services::metadata::ExtractedMetadata>& metadataList) {
    return executeServiceCall<std::vector<std::string>>(
        "metadata_service",
        "storeMetadataBatch",
        [this, metadataList]() -> std::vector<std::string> {
            std::vector<std::string> results;
            results.reserve(metadataList.size());
            
            for (const auto& metadata : metadataList) {
                try {
                    auto metadataId = storeMetadataToMDS(metadata).get();
                    results.push_back(metadataId);
                } catch (const std::exception& e) {
                    OSCEAN_LOG_WARN("CoreServiceProxy", "批量存储元数据失败: {}", e.what());
                    results.push_back("");
                }
            }
            
            return results;
        });
}

// === 高级功能 ===

void CoreServiceProxyImpl::setServiceTimeout(const std::string& serviceName, std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> lock(configMutex_);
    serviceTimeouts_[serviceName] = timeout;
}

void CoreServiceProxyImpl::setRetryPolicy(const std::string& serviceName, 
                                         int maxRetries, 
                                         std::chrono::milliseconds retryDelay) {
    std::lock_guard<std::mutex> lock(configMutex_);
    retryPolicies_[serviceName] = {maxRetries, retryDelay};
}

void CoreServiceProxyImpl::registerServiceUnavailableCallback(
    const std::string& serviceName,
    std::function<void(const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    serviceUnavailableCallbacks_[serviceName] = callback;
}

std::vector<std::string> CoreServiceProxyImpl::getAvailableServices() const {
    std::vector<std::string> services;
    
    if (checkServiceAvailability("data_access_service") == ServiceAvailability::AVAILABLE) {
        services.push_back("data_access_service");
    }
    
    if (checkServiceAvailability("metadata_service") == ServiceAvailability::AVAILABLE) {
        services.push_back("metadata_service");
    }
    
    return services;
}

bool CoreServiceProxyImpl::refreshServiceConnections(const std::string& serviceName) {
    try {
        if (serviceName.empty() || serviceName == "data_access_service") {
            // 重新连接DAS
            // 实际实现中，这里会重新建立服务连接
            OSCEAN_LOG_INFO("CoreServiceProxy", "已刷新 Data Access Service 连接");
        }
        
        if (serviceName.empty() || serviceName == "metadata_service") {
            // 重新连接MDS
            // 实际实现中，这里会重新建立服务连接
            OSCEAN_LOG_INFO("CoreServiceProxy", "已刷新 Metadata Service 连接");
        }
        
        return true;
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("CoreServiceProxy", "刷新服务连接失败: {}", e.what());
        return false;
    }
}

// === 私有方法 ===

template<typename T>
boost::future<T> CoreServiceProxyImpl::executeServiceCall(
    const std::string& serviceName,
    const std::string& operation,
    std::function<T()> serviceCall) {
    
    auto promise = std::make_shared<boost::promise<T>>();
    auto future = promise->get_future();
    
    boost::asio::post(*threadPool_, [this, serviceName, operation, serviceCall, promise]() {
        auto startTime = std::chrono::steady_clock::now();
        
        try {
            // 执行服务调用
            T result = serviceCall();
            
            // 记录成功统计
            auto duration = std::chrono::steady_clock::now() - startTime;
            logServiceCall(serviceName, operation, true, duration);
            
            promise->set_value(std::move(result));
            
        } catch (const std::exception& e) {
            // 记录失败统计
            auto duration = std::chrono::steady_clock::now() - startTime;
            logServiceCall(serviceName, operation, false, duration);
            
            // 处理异常和重试逻辑
            bool shouldRetry = handleServiceException(serviceName, operation, e);
            if (shouldRetry) {
                // 实现重试逻辑
                // 这里简化处理，实际实现会根据重试策略进行重试
            }
            
            promise->set_exception(std::current_exception());
        }
    });
    
    return future;
}

void CoreServiceProxyImpl::loadServiceConfiguration(const std::map<std::string, std::any>& config) {
    // 从配置中加载服务连接信息
    // 实际实现中，这里会解析配置并建立服务连接
    
    // 模拟服务连接状态
    // dataAccessService_ = ...; // 实际的服务连接
    // metadataService_ = ...; // 实际的服务连接
    
    OSCEAN_LOG_INFO("CoreServiceProxy", "服务配置已加载");
}

ServiceAvailability CoreServiceProxyImpl::checkDASAvailability() const {
    // 实际实现中，这里会检查DAS的健康状态
    return dataAccessService_ ? ServiceAvailability::AVAILABLE : ServiceAvailability::UNAVAILABLE;
}

ServiceAvailability CoreServiceProxyImpl::checkMDSAvailability() const {
    // 实际实现中，这里会检查MDS的健康状态
    return metadataService_ ? ServiceAvailability::AVAILABLE : ServiceAvailability::UNAVAILABLE;
}

void CoreServiceProxyImpl::logServiceCall(const std::string& serviceName,
                                        const std::string& operation,
                                        bool success,
                                        std::chrono::duration<double> duration) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    auto& stats = serviceStats_[serviceName];
    stats.totalCalls++;
    stats.lastCallTime = std::chrono::system_clock::now();
    
    if (success) {
        stats.successfulCalls++;
    } else {
        stats.failedCalls++;
    }
    
    // 更新平均响应时间
    auto totalDuration = stats.averageResponseTime * (stats.totalCalls - 1) + duration;
    stats.averageResponseTime = totalDuration / stats.totalCalls;
    
    OSCEAN_LOG_DEBUG("CoreServiceProxy", 
                     "服务调用 [{}:{}] - 成功: {}, 耗时: {:.3f}ms",
                     serviceName, operation, success, duration.count() * 1000);
}

bool CoreServiceProxyImpl::handleServiceException(const std::string& serviceName,
                                                 const std::string& operation,
                                                 const std::exception& exception) {
    OSCEAN_LOG_ERROR("CoreServiceProxy", 
                     "服务调用异常 [{}:{}]: {}", 
                     serviceName, operation, exception.what());
    
    // 检查是否需要触发服务不可用回调
    std::lock_guard<std::mutex> lock(callbackMutex_);
    auto it = serviceUnavailableCallbacks_.find(serviceName);
    if (it != serviceUnavailableCallbacks_.end()) {
        it->second(serviceName);
    }
    
    // 简化的重试判断逻辑
    // 实际实现中会根据异常类型和重试策略决定是否重试
    return false;
}

} // namespace oscean::workflow_engine::proxies 