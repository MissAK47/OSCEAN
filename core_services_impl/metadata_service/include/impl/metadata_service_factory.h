#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "core_services/metadata/i_metadata_service.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/data_access/i_unified_data_access_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include <memory>
#include <stdexcept>
#include "impl/metadata_service_impl.h"
#include "impl/unified_database_manager.h"
#include "impl/intelligent_recognizer.h"
#include "impl/configuration_manager.h"

namespace oscean::core_services::metadata {
    class IMetadataService;
    struct MetadataServiceConfiguration;
}

namespace oscean::core_services {
    class ICrsService;
}

namespace oscean::core_services::metadata::impl {

// 前向声明内部实现类
class MetadataServiceImpl;
class UnifiedDatabaseManager;
class IntelligentRecognizer;
class QueryEngine;
class ConfigurationManager;

/**
 * @brief 元数据工厂异常类
 */
class MetadataFactoryException : public std::runtime_error {
public:
    explicit MetadataFactoryException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @class MetadataServiceFactory
 * @brief 纯工厂模式的元数据服务创建器 - 支持完全依赖注入
 *
 * 🎯 核心设计原则：
 * ✅ 无全局状态，完全无状态工厂
 * ✅ 支持依赖注入和控制反转
 * ✅ 可测试和可配置
 * ✅ 集成CommonServicesFactory
 * ✅ 明确的生命周期管理
 */
class MetadataServiceFactory {
public:
    /**
     * @brief 构造函数 - 完全基于依赖注入的设计
     * @param commonFactory 从外部注入的CommonServicesFactory实例（不在此处创建）
     * @param serviceManager 从外部注入的ServiceManager实例（用于延迟加载依赖服务）
     * @param config 元数据服务配置
     * 
     * @note 重要：此构造函数遵循依赖注入原则，不创建任何CommonServicesFactory实例
     *       所有依赖都从外部注入，确保单一职责和可测试性
     */
    explicit MetadataServiceFactory(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory,
        std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager,
        const MetadataServiceConfiguration& config = getDefaultConfiguration()
    );
    
    /**
     * @brief 移动构造函数
     */
    MetadataServiceFactory(MetadataServiceFactory&& other) noexcept;
    
    /**
     * @brief 移动赋值运算符
     */
    MetadataServiceFactory& operator=(MetadataServiceFactory&& other) noexcept;
    
    /**
     * @brief 删除拷贝构造和赋值
     */
    MetadataServiceFactory(const MetadataServiceFactory&) = delete;
    MetadataServiceFactory& operator=(const MetadataServiceFactory&) = delete;
    
    /**
     * @brief 析构函数
     */
    ~MetadataServiceFactory();
    
    // === 🏭 静态工厂方法 ===
    
    /**
     * @brief [静态] 创建用于测试的工厂实例
     */
    static std::unique_ptr<MetadataServiceFactory> createForTesting(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory,
        std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager
    );
    
    /**
     * @brief [静态] 创建用于高性能场景的工厂实例
     */
    static std::unique_ptr<MetadataServiceFactory> createHighPerformance(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory,
        std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager
    );
    
    // === 🎯 核心服务创建 ===
    
    /**
     * @brief 创建标准元数据服务
     */
    std::unique_ptr<IMetadataService> createMetadataService();
    
    /**
     * @brief 创建带配置的元数据服务
     */
    std::unique_ptr<IMetadataService> createMetadataService(const MetadataServiceConfiguration& config);
    
    /**
     * @brief 创建高性能元数据服务
     */
    std::unique_ptr<IMetadataService> createHighPerformanceMetadataService();
    
    /**
     * @brief 创建用于测试的元数据服务
     */
    std::unique_ptr<IMetadataService> createTestingMetadataService();
    
    /**
     * @brief 创建流式处理元数据服务
     */
    std::unique_ptr<IMetadataService> createStreamingMetadataService();
    
    /**
     * @brief 创建批处理元数据服务
     */
    std::unique_ptr<IMetadataService> createBatchProcessingMetadataService();
    
    /**
     * @brief 创建低内存元数据服务
     */
    std::unique_ptr<IMetadataService> createLowMemoryMetadataService();
    
    /**
     * @brief 创建模拟服务（用于测试）
     */
    std::unique_ptr<IMetadataService> createMockService();
    
    // === 🔧 配置访问 ===
    
    /**
     * @brief 获取当前配置
     */
    const MetadataServiceConfiguration& getConfiguration() const { return config_; }
    
    /**
     * @brief 更新配置
     */
    void updateConfiguration(const MetadataServiceConfiguration& config);
    
    /**
     * @brief 获取Common服务工厂
     */
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> getCommonFactory() const {
        return commonFactory_;
    }
    
    // === 📊 状态和健康检查 ===
    
    /**
     * @brief 检查工厂是否健康
     */
    bool isHealthy() const;
    
    /**
     * @brief 获取诊断信息
     */
    std::vector<std::string> getDiagnosticMessages() const;
    
    /**
     * @brief 验证依赖服务
     */
    bool validateDependencies() const;
    
    /**
     * @brief 验证配置有效性
     */
    static bool validateConfig(const MetadataServiceConfiguration& config);
    
    /**
     * @brief 获取默认配置
     */
    static MetadataServiceConfiguration getDefaultConfiguration();
    
    /**
     * @brief 获取最优配置
     */
    static MetadataServiceConfiguration getOptimalConfiguration();
    
    /**
     * @brief 检查资源可用性
     */
    static bool checkResourceAvailability(const MetadataServiceConfiguration& config);

private:
    // === 配置和依赖（通过构造函数注入，不是内部创建）===
    MetadataServiceConfiguration config_;
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory_;  // 注入的CommonServicesFactory实例
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager_;  // 注入的ServiceManager实例
    
    // === 内部辅助方法 ===
    void validateConfiguration(const MetadataServiceConfiguration& config);
    std::unique_ptr<IMetadataService> createMetadataServiceWithConfig(const MetadataServiceConfiguration& config);
    void setupLogging();
    void validateCommonFactory();
};

} // namespace oscean::core_services::metadata::impl 