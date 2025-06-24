#pragma once

/**
 * @file core_service_proxy_impl.h
 * @brief 核心服务代理具体实现类
 * @author OSCEAN Team
 * @date 2024
 */

#include "core_service_proxy.h"
#include "core_services/data_access/i_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include <boost/thread/thread_pool.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <memory>
#include <mutex>
#include <map>
#include <functional>

namespace oscean::workflow_engine::proxies {

/**
 * @brief 重试策略配置
 */
struct RetryPolicy {
    int maxRetries = 3;
    std::chrono::milliseconds retryDelay{1000};
};

/**
 * @brief 核心服务代理具体实现类
 */
class CoreServiceProxyImpl : public CoreServiceProxy {
public:
    CoreServiceProxyImpl();
    ~CoreServiceProxyImpl() override;

    // === CoreServiceProxy 接口实现 ===

    bool initialize(const std::map<std::string, std::any>& config) override;

    ServiceAvailability checkServiceAvailability(const std::string& serviceName) override;

    ServiceCallStats getServiceStats(const std::string& serviceName) const override;

    // Data Access Service 代理方法
    boost::future<std::optional<oscean::core_services::FileMetadata>> 
    getFileMetadataFromDAS(const std::string& filePath) override;

    boost::future<std::shared_ptr<oscean::core_services::data_access::api::GridData>>
    readGridDataFromDAS(const oscean::core_services::data_access::api::ReadGridDataRequest& request) override;

    boost::future<oscean::core_services::data_access::api::FeatureCollection>
    readFeatureCollectionFromDAS(const oscean::core_services::data_access::api::ReadFeatureCollectionRequest& request) override;

    boost::future<bool> 
    checkVariableExistsInDAS(const std::string& filePath, const std::string& variableName) override;

    boost::future<std::vector<oscean::core_services::MetadataEntry>>
    getVariableAttributesFromDAS(const std::string& filePath, const std::string& variableName) override;

    boost::future<std::vector<oscean::core_services::MetadataEntry>>
    getGlobalAttributesFromDAS(const std::string& filePath) override;

    // Metadata Service 代理方法
    boost::future<std::string> 
    recognizeFileWithMDS(const std::string& filePath) override;

    boost::future<std::string>
    storeMetadataToMDS(const oscean::core_services::metadata::ExtractedMetadata& metadata) override;

    boost::future<std::vector<oscean::core_services::metadata::MetadataEntry>>
    queryMetadataFromMDS(const oscean::core_services::metadata::MultiDimensionalQueryCriteria& criteria) override;

    boost::future<bool>
    updateMetadataInMDS(const std::string& metadataId, 
                       const oscean::core_services::metadata::MetadataUpdate& update) override;

    boost::future<bool>
    deleteMetadataFromMDS(const std::string& metadataId) override;

    // 批量操作
    boost::future<std::vector<std::optional<oscean::core_services::FileMetadata>>>
    getFileMetadataBatchFromDAS(const std::vector<std::string>& filePaths) override;

    boost::future<std::vector<std::string>>
    recognizeFilesBatchWithMDS(const std::vector<std::string>& filePaths) override;

    boost::future<std::vector<std::string>>
    storeMetadataBatchToMDS(const std::vector<oscean::core_services::metadata::ExtractedMetadata>& metadataList) override;

    // 高级功能
    void setServiceTimeout(const std::string& serviceName, std::chrono::milliseconds timeout) override;

    void setRetryPolicy(const std::string& serviceName, 
                       int maxRetries, 
                       std::chrono::milliseconds retryDelay) override;

    void registerServiceUnavailableCallback(
        const std::string& serviceName,
        std::function<void(const std::string&)> callback) override;

    std::vector<std::string> getAvailableServices() const override;

    bool refreshServiceConnections(const std::string& serviceName = "") override;

protected:
    void logServiceCall(const std::string& serviceName,
                       const std::string& operation,
                       bool success,
                       std::chrono::duration<double> duration) override;

    bool handleServiceException(const std::string& serviceName,
                               const std::string& operation,
                               const std::exception& exception) override;

private:
    // 基本状态
    bool initialized_;
    
    // 异步执行
    boost::scoped_ptr<boost::thread_pool> threadPool_;
    
    // 服务连接
    std::shared_ptr<oscean::core_services::data_access::IDataAccessService> dataAccessService_;
    std::shared_ptr<oscean::core_services::metadata::IMetadataService> metadataService_;
    
    // 线程安全
    mutable std::mutex serviceStatusMutex_;
    mutable std::mutex statsMutex_;
    mutable std::mutex configMutex_;
    mutable std::mutex callbackMutex_;
    
    // 配置和状态
    std::map<std::string, ServiceCallStats> serviceStats_;
    std::map<std::string, std::chrono::milliseconds> serviceTimeouts_;
    std::map<std::string, RetryPolicy> retryPolicies_;
    std::map<std::string, std::function<void(const std::string&)>> serviceUnavailableCallbacks_;

    /**
     * @brief 执行异步服务调用的模板方法
     * @tparam T 返回值类型
     * @param serviceName 服务名称
     * @param operation 操作名称
     * @param serviceCall 服务调用函数
     * @return 异步结果
     */
    template<typename T>
    boost::future<T> executeServiceCall(
        const std::string& serviceName,
        const std::string& operation,
        std::function<T()> serviceCall);

    /**
     * @brief 加载服务配置
     * @param config 配置信息
     */
    void loadServiceConfiguration(const std::map<std::string, std::any>& config);

    /**
     * @brief 检查DAS可用性
     * @return 可用性状态
     */
    ServiceAvailability checkDASAvailability() const;

    /**
     * @brief 检查MDS可用性
     * @return 可用性状态
     */
    ServiceAvailability checkMDSAvailability() const;
};

/**
 * @brief 核心服务代理工厂实现
 */
class CoreServiceProxyFactoryImpl : public CoreServiceProxyFactory {
public:
    std::shared_ptr<CoreServiceProxy> createProxy(
        const std::map<std::string, std::any>& config) override;

    std::shared_ptr<CoreServiceProxy> createMockProxy() override;
};

} // namespace oscean::workflow_engine::proxies 