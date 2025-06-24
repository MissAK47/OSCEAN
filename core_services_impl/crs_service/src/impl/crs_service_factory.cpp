/**
 * @file crs_service_factory.cpp
 * @brief CRS服务工厂实现 - 纯工厂+依赖注入模式
 */

#include "core_services/crs/crs_service_factory.h"
#include "core_services/crs/impl/optimized_crs_service.h"
#include "common_utils/infrastructure/common_services_factory.h"

#include <stdexcept>
#include <algorithm>

namespace oscean::core_services::crs {

// === 🏗️ 构造函数实现 ===

CrsServiceFactory::CrsServiceFactory(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory,
    const CrsServiceConfig& config
) : config_(config), commonFactory_(std::move(commonFactory)) {
    validateCommonFactory();
    validateConfiguration(config_);
    setupLogging();
}

CrsServiceFactory::CrsServiceFactory(CrsServiceFactory&& other) noexcept
    : config_(std::move(other.config_)),
      commonFactory_(std::move(other.commonFactory_)) {
}

CrsServiceFactory& CrsServiceFactory::operator=(CrsServiceFactory&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        commonFactory_ = std::move(other.commonFactory_);
    }
    return *this;
}

CrsServiceFactory::~CrsServiceFactory() = default;

// === 🏭 静态工厂方法 ===

std::unique_ptr<CrsServiceFactory> CrsServiceFactory::createForTesting(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory
) {
    if (!commonFactory) {
        // 为测试创建一个简化的CommonServicesFactory
        commonFactory = oscean::common_utils::infrastructure::CommonServicesFactory::createForTesting();
    }
    
    auto config = CrsServiceConfig::createForTesting();
    return std::make_unique<CrsServiceFactory>(std::move(commonFactory), config);
}

std::unique_ptr<CrsServiceFactory> CrsServiceFactory::createHighPerformance(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory
) {
    if (!commonFactory) {
        throw CrsFactoryException("CommonServicesFactory is required for high performance CRS factory");
    }
    
    auto config = CrsServiceConfig::createHighPerformance();
    return std::make_unique<CrsServiceFactory>(std::move(commonFactory), config);
}

// === 🎯 核心服务创建 ===

std::unique_ptr<ICrsService> CrsServiceFactory::createCrsService() {
    return createCrsServiceWithConfig(config_);
}

std::unique_ptr<ICrsService> CrsServiceFactory::createCrsService(const CrsServiceConfig& config) {
    validateConfiguration(config);
    return createCrsServiceWithConfig(config);
}

std::unique_ptr<ICrsService> CrsServiceFactory::createHighPerformanceCrsService() {
    auto config = CrsServiceConfig::createHighPerformance();
    return createCrsServiceWithConfig(config);
}

std::unique_ptr<ICrsService> CrsServiceFactory::createOptimizedCrsService() {
    // 优化版本使用当前配置，但启用所有优化选项
    auto config = config_;
    config.enableSIMD = true;
    config.enableBatchProcessing = true;
    config.enableMemoryMappedProcessing = true;
    config.enablePerformanceOptimization = true;
    
    return createCrsServiceWithConfig(config);
}

std::unique_ptr<ICrsService> CrsServiceFactory::createTestingCrsService() {
    auto config = CrsServiceConfig::createForTesting();
    return createCrsServiceWithConfig(config);
}

std::unique_ptr<ICrsService> CrsServiceFactory::createStreamingCrsService() {
    auto config = config_;
    config.enableMemoryMappedProcessing = true;
    config.streamingBufferSize = std::max(config.streamingBufferSize, static_cast<size_t>(4096));
    
    return createCrsServiceWithConfig(config);
}

std::unique_ptr<ICrsService> CrsServiceFactory::createBatchProcessingCrsService() {
    auto config = config_;
    config.enableBatchProcessing = true;
    config.threadPoolSize = std::max(config.threadPoolSize, static_cast<size_t>(8));
    config.maxCacheSize = std::max(config.maxCacheSize, static_cast<size_t>(50000));
    
    return createCrsServiceWithConfig(config);
}

// === 🔧 配置访问 ===

void CrsServiceFactory::updateConfiguration(const CrsServiceConfig& config) {
    validateConfiguration(config);
    config_ = config;
}

// === 📊 状态和健康检查 ===

bool CrsServiceFactory::isHealthy() const {
    if (!commonFactory_) {
        return false;
    }
    
    if (!commonFactory_->isHealthy()) {
        return false;
    }
    
    return validateDependencies();
}

std::vector<std::string> CrsServiceFactory::getDiagnosticMessages() const {
    std::vector<std::string> messages;
    
    if (!commonFactory_) {
        messages.push_back("CommonServicesFactory not provided");
        return messages;
    }
    
    // 检查Common服务的健康状态
    auto commonDiagnostics = commonFactory_->getDiagnosticMessages();
    for (const auto& msg : commonDiagnostics) {
        messages.push_back("Common: " + msg);
    }
    
    // 检查CRS特定配置
    if (config_.maxCacheSize == 0) {
        messages.push_back("Cache disabled - may impact performance");
    }
    
    if (config_.threadPoolSize < 2) {
        messages.push_back("Low thread pool size - may limit concurrency");
    }
    
    if (!config_.enableSIMD) {
        messages.push_back("SIMD disabled - may reduce performance");
    }
    
    return messages;
}

bool CrsServiceFactory::validateDependencies() const {
    if (!commonFactory_) {
        return false;
    }
    
    try {
        // 验证必需的Common服务
        auto memoryManager = commonFactory_->getMemoryManager();
        if (!memoryManager) {
            return false;
        }
        
        auto threadPoolManager = commonFactory_->getThreadPoolManager();
        if (!threadPoolManager) {
            return false;
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// === 🔧 内部辅助方法 ===

void CrsServiceFactory::validateConfiguration(const CrsServiceConfig& config) {
    if (config.maxCacheSize > 1000000) {
        throw CrsFactoryException("Cache size too large: " + std::to_string(config.maxCacheSize));
    }
    
    if (config.threadPoolSize > 64) {
        throw CrsFactoryException("Thread pool size too large: " + std::to_string(config.threadPoolSize));
    }
    
    if (config.streamingBufferSize == 0) {
        throw CrsFactoryException("Streaming buffer size cannot be zero");
    }
    
    if (config.transformationTolerance <= 0.0) {
        throw CrsFactoryException("Transformation tolerance must be positive");
    }
}

std::unique_ptr<ICrsService> CrsServiceFactory::createCrsServiceWithConfig(const CrsServiceConfig& config) {
    if (!commonFactory_) {
        throw CrsFactoryException("CommonServicesFactory not available");
    }
    
    try {
        // 创建CRS服务，注入Common服务依赖
        auto crsService = std::make_unique<OptimizedCrsService>(
            commonFactory_,
            config
        );
        
        return crsService;
    } catch (const std::exception& e) {
        throw CrsFactoryException("Failed to create CRS service: " + std::string(e.what()));
    }
}

void CrsServiceFactory::setupLogging() {
    // TODO: 使用CommonServicesFactory的日志服务进行日志设置
    // auto loggingService = commonFactory_->getLoggingService();
    // if (loggingService) {
    //     loggingService->setLogLevel(config_.logLevel);
    // }
}

void CrsServiceFactory::validateCommonFactory() {
    if (!commonFactory_) {
        throw CrsFactoryException("CommonServicesFactory cannot be null");
    }
    
    if (!commonFactory_->isHealthy()) {
        throw CrsFactoryException("CommonServicesFactory is not healthy");
    }
}

} // namespace oscean::core_services::crs 