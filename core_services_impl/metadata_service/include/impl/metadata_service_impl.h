#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "core_services/metadata/i_metadata_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/utilities/logging_utils.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <chrono>
#include <mutex>
#include "impl/unified_database_manager.h"
#include "impl/intelligent_recognizer.h"
#include "impl/metadata_extractor.h"
#include "common_utils/async/async_framework.h"

namespace oscean::core_services {
    // 前向声明CRS服务接口，避免在头文件中引入依赖
    class ICrsService;
}

namespace oscean::workflow_engine::service_management {
    // 前向声明服务管理器接口
    class IServiceManager;
}

namespace oscean::core_services::metadata::impl {

// 前向声明
class QueryEngine;
class ConfigurationManager;
class MetadataStandardizer;

/**
 * @brief 精简的元数据服务实现
 * @note 这是IMetadataService的核心实现类，只包含基础CRUD功能
 */
class MetadataServiceImpl : public IMetadataService, 
                            public std::enable_shared_from_this<MetadataServiceImpl> {
public:
    /**
     * @brief 构造函数 - 采用依赖注入模式
     */
    MetadataServiceImpl(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServicesFactory,
        std::shared_ptr<UnifiedDatabaseManager> dbManager,
        std::shared_ptr<IntelligentRecognizer> recognizer,
        std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager
    );

    /**
     * @brief 析构函数
     */
    ~MetadataServiceImpl() override;

    // 删除拷贝和移动构造函数/赋值运算符
    MetadataServiceImpl(const MetadataServiceImpl&) = delete;
    MetadataServiceImpl& operator=(const MetadataServiceImpl&) = delete;
    MetadataServiceImpl(MetadataServiceImpl&&) = delete;
    MetadataServiceImpl& operator=(MetadataServiceImpl&&) = delete;

    /**
     * @brief 初始化服务，加载配置等
     */
    bool initialize() override;

    /**
     * @brief 核心功能：处理单个文件，提取、标准化和存储元数据
     */
    boost::future<AsyncResult<std::string>> processFile(const std::string& filePath) override;

    /**
     * @brief 🚀 [新] 批量过滤未处理的文件
     */
    boost::future<std::vector<std::string>> filterUnprocessedFilesAsync(
        const std::vector<std::string>& filePaths) override;

    /**
     * @brief 🚀 [新] 对元数据进行分类和最终丰富
     */
    boost::future<FileMetadata> classifyAndEnrichAsync(
        const FileMetadata& metadata) override;

    /**
     * @brief 🚀 [新] 异步保存元数据到持久化存储
     */
    boost::future<AsyncResult<bool>> saveMetadataAsync(
        const FileMetadata& metadata) override;

    /**
     * @brief 异步接收文件元数据，用于外部推送
     * @note 签名与 IMetadataService 接口保持一致
     */
    boost::future<AsyncResult<void>> receiveFileMetadataAsync(FileMetadata metadata) override;

    /**
     * @brief 根据通用查询条件异步执行查询
     */
    boost::future<AsyncResult<std::vector<FileMetadata>>> queryMetadataAsync(const QueryCriteria& criteria) override;

    /**
     * @brief 根据文件路径精确匹配查询
     */
    boost::future<AsyncResult<FileMetadata>> queryByFilePathAsync(const std::string& filePath) override;
    
    /**
     * @brief 根据数据主类别进行查询
     */
    boost::future<AsyncResult<std::vector<FileMetadata>>> queryByCategoryAsync(
        DataType category,
        const std::optional<QueryCriteria>& additionalCriteria = std::nullopt
    ) override;

    /**
     * @brief 根据分类键值对进行查询
     */
    AsyncResult<std::vector<FileMetadata>> queryByCategoryAsync(const std::string& category, const std::string& value);

    /**
     * @brief 根据文件路径删除元数据
     */
    AsyncResult<bool> deleteMetadataByFilePathAsync(const std::string& filePath);

    /**
     * @brief 删除元数据
     */
    boost::future<AsyncResult<bool>> deleteMetadataAsync(const std::string& metadataId) override;

    /**
     * @brief 更新元数据
     */
    boost::future<AsyncResult<bool>> updateMetadataAsync(
        const std::string& metadataId, 
        const MetadataUpdate& update
    ) override;

    /**
     * @brief 更新服务配置
     */
    boost::future<AsyncResult<void>> updateConfigurationAsync(const MetadataServiceConfiguration& config) override;

    // === 🔧 服务管理接口实现 ===
    
    std::string getVersion() const override;
    bool isReady() const override;

    // === ⚙️ 配置管理接口实现 ===
    
    boost::future<AsyncResult<MetadataServiceConfiguration>> getConfigurationAsync() override;

    /**
     * @brief 延迟获取CRS服务
     */
    std::shared_ptr<ICrsService> getCrsService() const;

private:
    // === 内部辅助方法 ===
    
    /**
     * @brief 执行实际的文件处理流程
     */
    AsyncResult<std::string> processFileInternal(const std::string& filePath);

    /**
     * @brief 更新服务配置
     */
    void updateConfiguration(const MetadataServiceConfiguration& config);
    
    /**
     * @brief 生成唯一的元数据ID
     */
    std::string generateMetadataId() const;

    // === 依赖注入的服务和组件 ===
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices_;
    std::shared_ptr<UnifiedDatabaseManager> dbManager_;
    std::shared_ptr<IntelligentRecognizer> recognizer_;
    std::shared_ptr<MetadataStandardizer> standardizer_;
    mutable std::shared_ptr<ICrsService> crsService_;
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager_;
    std::shared_ptr<spdlog::logger> logger_;

    bool isInitialized_ = false;
};

} // namespace oscean::core_services::metadata::impl 