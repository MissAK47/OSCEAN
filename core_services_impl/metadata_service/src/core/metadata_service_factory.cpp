// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "impl/metadata_service_factory.h"
#include "impl/metadata_service_impl.h"
#include "impl/unified_database_manager.h"
#include "impl/intelligent_recognizer.h"
#include "common_utils/utilities/logging_utils.h"
#include "workflow_engine/service_management/i_service_manager.h" // 包含服务管理器接口
#include <boost/thread/future.hpp>
#include <filesystem>
#include <stdexcept>

using namespace oscean::core_services::metadata::impl;
using namespace oscean::core_services::metadata;

// === 构造函数和析构函数 ===

MetadataServiceFactory::MetadataServiceFactory(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory,
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager,
    const MetadataServiceConfiguration& config)
    : config_(config), 
      commonFactory_(std::move(commonFactory)),
      serviceManager_(std::move(serviceManager)) {
    
    if (!commonFactory_) {
        throw MetadataFactoryException("CommonServicesFactory不能为空");
    }
    if (!serviceManager_) {
        throw MetadataFactoryException("ServiceManager不能为空");
    }
    
    validateConfiguration(config_);
    
    // LOG_INFO("MetadataServiceFactory constructed. Service dependencies will be resolved on-demand.");
    std::cout << "[DEBUG] MetadataServiceFactory构造完成，依赖将按需解析" << std::endl;
}

MetadataServiceFactory::MetadataServiceFactory(MetadataServiceFactory&& other) noexcept
    : config_(std::move(other.config_))
    , commonFactory_(std::move(other.commonFactory_))
    , serviceManager_(std::move(other.serviceManager_)) {
    // LOG_INFO("MetadataServiceFactory move constructed.");
    std::cout << "[DEBUG] MetadataServiceFactory移动构造完成" << std::endl;
}

MetadataServiceFactory& MetadataServiceFactory::operator=(MetadataServiceFactory&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        commonFactory_ = std::move(other.commonFactory_);
        serviceManager_ = std::move(other.serviceManager_);
        // LOG_INFO("MetadataServiceFactory move assigned.");
        std::cout << "[DEBUG] MetadataServiceFactory移动赋值完成" << std::endl;
    }
    return *this;
}

MetadataServiceFactory::~MetadataServiceFactory() {
    // LOG_INFO("MetadataServiceFactory destructed.");
    std::cout << "[DEBUG] MetadataServiceFactory析构" << std::endl;
}

// === 静态工厂方法 ===

std::unique_ptr<MetadataServiceFactory> MetadataServiceFactory::createForTesting(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory,
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager) {
    
    auto config = getDefaultConfiguration();
    // 测试配置调整
    config.metadataCacheSize = 100;
    config.queryCacheSize = 50;
    config.maxConcurrentQueries = 2;
    config.maxBatchSize = 10;
    
    return std::make_unique<MetadataServiceFactory>(commonFactory, serviceManager, config);
}

std::unique_ptr<MetadataServiceFactory> MetadataServiceFactory::createHighPerformance(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory,
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager) {
    
    auto config = getOptimalConfiguration();
    return std::make_unique<MetadataServiceFactory>(commonFactory, serviceManager, config);
}

// === 核心服务创建 ===

std::unique_ptr<IMetadataService> MetadataServiceFactory::createMetadataService() {
    return createMetadataServiceWithConfig(config_);
}

std::unique_ptr<IMetadataService> MetadataServiceFactory::createMetadataService(
    const MetadataServiceConfiguration& config) {
    return createMetadataServiceWithConfig(config);
}

std::unique_ptr<IMetadataService> MetadataServiceFactory::createHighPerformanceMetadataService() {
    auto config = getOptimalConfiguration();
    return createMetadataServiceWithConfig(config);
}

std::unique_ptr<IMetadataService> MetadataServiceFactory::createTestingMetadataService() {
    auto config = getDefaultConfiguration();
    // 测试特定配置
    config.metadataCacheSize = 100;
    config.queryCacheSize = 50;
    config.maxConcurrentQueries = 2;
    // 数据处理工作流测试：禁用分类规则加载
    config.classificationConfig.loadClassificationRules = false;
    
    return createMetadataServiceWithConfig(config);
}

std::unique_ptr<IMetadataService> MetadataServiceFactory::createStreamingMetadataService() {
    auto config = config_;
    // 流式处理配置
    config.maxBatchSize = 1000;
    config.queryCacheSize = 2000;
    
    return createMetadataServiceWithConfig(config);
}

std::unique_ptr<IMetadataService> MetadataServiceFactory::createBatchProcessingMetadataService() {
    auto config = config_;
    // 批处理配置
    config.maxBatchSize = 5000;
    config.maxConcurrentQueries = 20;
    
    return createMetadataServiceWithConfig(config);
}

std::unique_ptr<IMetadataService> MetadataServiceFactory::createLowMemoryMetadataService() {
    auto config = config_;
    // 低内存配置
    config.metadataCacheSize = 200;
    config.queryCacheSize = 100;
    config.maxConcurrentQueries = 3;
    config.maxBatchSize = 50;
    
    return createMetadataServiceWithConfig(config);
}

std::unique_ptr<IMetadataService> MetadataServiceFactory::createMockService() {
    // 创建模拟服务用于测试
    LOG_WARN("创建模拟元数据服务 - 仅用于测试");
    return createTestingMetadataService();
}

// === 配置访问 ===

void MetadataServiceFactory::updateConfiguration(const MetadataServiceConfiguration& config) {
    validateConfiguration(config);
    config_ = config;
    LOG_INFO("MetadataServiceFactory配置已更新");
}

// === 状态和健康检查 ===

bool MetadataServiceFactory::isHealthy() const {
    return commonFactory_ != nullptr && validateConfig(config_);
}

std::vector<std::string> MetadataServiceFactory::getDiagnosticMessages() const {
    std::vector<std::string> messages;
    
    if (!commonFactory_) {
        messages.push_back("CommonServicesFactory为空");
    }
    
    if (!validateConfig(config_)) {
        messages.push_back("配置验证失败");
    }
    
    if (messages.empty()) {
        messages.push_back("工厂状态正常");
    }
    
    return messages;
}

bool MetadataServiceFactory::validateDependencies() const {
    return commonFactory_ != nullptr;
}

bool MetadataServiceFactory::validateConfig(const MetadataServiceConfiguration& config) {
    return !config.databaseConfig.basePath.empty()
        && config.metadataCacheSize > 0
        && config.queryCacheSize > 0
        && config.maxConcurrentQueries > 0
        && config.maxBatchSize > 0;
}

MetadataServiceConfiguration MetadataServiceFactory::getDefaultConfiguration() {
    MetadataServiceConfiguration config;
    
    // 数据库配置
    config.databaseConfig.basePath = "./databases";
    config.databaseConfig.enableWALMode = true;
    config.databaseConfig.cacheSize = 1000;
    config.databaseConfig.connectionTimeout = std::chrono::seconds(30);
    
    // 分类配置
    config.classificationConfig.enableFuzzyMatching = true;
    config.classificationConfig.fuzzyMatchingThreshold = 0.8;
    
    // 缓存配置
    config.metadataCacheSize = 1000;
    config.queryCacheSize = 500;
    config.cacheExpiryTime = std::chrono::minutes(30);
    
    // 性能配置
    config.maxConcurrentQueries = 10;
    config.queryTimeout = std::chrono::milliseconds(5000);
    config.maxBatchSize = 100;
    
    return config;
}

MetadataServiceConfiguration MetadataServiceFactory::getOptimalConfiguration() {
    auto config = getDefaultConfiguration();
    
    // 高性能配置
    config.metadataCacheSize = 5000;
    config.queryCacheSize = 2000;
    config.maxConcurrentQueries = 20;
    config.maxBatchSize = 1000;
    config.databaseConfig.cacheSize = 5000;
    
    return config;
}

bool MetadataServiceFactory::checkResourceAvailability(const MetadataServiceConfiguration& config) {
    // 检查基础路径是否可写
    std::filesystem::path basePath(config.databaseConfig.basePath);
    return std::filesystem::exists(basePath.parent_path()) || 
           std::filesystem::create_directories(basePath.parent_path());
}

// === 私有方法 ===

void MetadataServiceFactory::validateConfiguration(const MetadataServiceConfiguration& config) {
    if (!validateConfig(config)) {
        throw MetadataFactoryException("配置验证失败");
    }
}

std::unique_ptr<IMetadataService> MetadataServiceFactory::createMetadataServiceWithConfig(
    const MetadataServiceConfiguration& config) {
    
    LOG_INFO("开始创建MetadataService实例...");
    
    try {
        // 1. 创建数据库管理器 - 注入已有的CommonServicesFactory
        // 🔧 重要说明：传入占位路径，真实路径将在UnifiedDatabaseManager::initialize()中从配置文件读取
        // 传递注入的commonFactory_，而不是创建新的CommonServicesFactory实例
        auto dbManager = std::make_shared<UnifiedDatabaseManager>(
            "./databases",  // 临时占位路径，将被initialize()中的配置覆盖
            commonFactory_  // 传递注入的CommonServicesFactory实例
        );
        LOG_INFO("  - 数据库管理器(UnifiedDatabaseManager)已创建");

        // 2. 创建智能识别器 - 修复构造函数参数
        auto recognizer = std::make_shared<IntelligentRecognizer>(
            commonFactory_->getLogger(),
            nullptr,  // dataAccessService - 在metadata服务中不需要
            nullptr,  // crsService - 延迟加载
            config.classificationConfig.loadClassificationRules  // 根据配置决定是否加载规则
        );
        LOG_INFO("  - 智能识别器(IntelligentRecognizer)已创建");
        
        // 3. 创建元数据服务实例，注入服务管理器和已有的CommonServicesFactory
        LOG_INFO("  - 准备创建MetadataServiceImpl，注入服务管理器和CommonServicesFactory...");
        auto serviceImpl = std::make_unique<MetadataServiceImpl>(
            commonFactory_,  // 传递注入的CommonServicesFactory实例
            dbManager,
            recognizer,
            serviceManager_  // 注入服务管理器
        );
        LOG_INFO("  - MetadataServiceImpl实例已创建");

        // 4. 初始化服务
        if (!serviceImpl->initialize()) {
            throw MetadataFactoryException("元数据服务初始化失败");
        }
        LOG_INFO("  - MetadataServiceImpl初始化完成");
        
        LOG_INFO("✅ MetadataService实例创建并初始化成功");
        
        // 🔧 修复编译错误：直接返回unique_ptr
        return std::move(serviceImpl);

    } catch (const std::exception& e) {
        LOG_ERROR("创建元数据服务失败: {}", e.what());
        throw MetadataFactoryException(std::string("创建元数据服务失败: ") + e.what());
    }
}

void MetadataServiceFactory::setupLogging() {
    // 设置日志
    LOG_INFO("MetadataServiceFactory日志已设置");
}

void MetadataServiceFactory::validateCommonFactory() {
    if (!commonFactory_) {
        throw MetadataFactoryException("CommonServicesFactory验证失败");
    }
}

// createService方法已删除，该功能由createMetadataServiceWithConfig提供 