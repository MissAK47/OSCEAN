/**
 * @file data_access_service_factory_impl.cpp
 * @brief 数据访问服务工厂实现 - 使用统一标准接口
 * 
 * 🎯 参考crs模块的标准工厂模式：
 * ✅ 统一使用core_service_interfaces接口
 * ✅ 支持完全依赖注入
 * ✅ 集成CommonServicesFactory
 * ✅ 标准化配置管理
 * ✅ 简化构造函数设计 - 内部自动管理并发优化组件
 */

#include "core_services/data_access/i_data_access_service_factory.h"
#include "unified_data_access_service_impl.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/infrastructure/common_services_factory.h"

namespace oscean::core_services::data_access {

/**
 * @brief 数据访问服务工厂实现 - 使用统一标准接口
 */
class DataAccessServiceFactoryImpl : public IDataAccessServiceFactory {
private:
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServicesFactory_;
    bool isInitialized_;

public:
    explicit DataAccessServiceFactoryImpl(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory = nullptr)
        : commonServicesFactory_(std::move(commonFactory))
        , isInitialized_(false) {
        
        if (!commonServicesFactory_) {
            // 创建默认的common services factory - 参考crs模块模式
            commonServicesFactory_ = std::make_shared<oscean::common_utils::infrastructure::CommonServicesFactory>(
                oscean::common_utils::infrastructure::ServiceConfiguration::createDefault()
            );
        }
    }

    ~DataAccessServiceFactoryImpl() override = default;

    // =============================================================================
    // IDataAccessServiceFactory 标准接口实现
    // =============================================================================
    
    std::shared_ptr<IUnifiedDataAccessService> createDataAccessService(
        const api::DataAccessConfiguration& config) override {
        
        OSCEAN_LOG_INFO("DataAccessServiceFactory", std::string("创建统一数据访问服务 - 配置: ") + config.serviceName);
        
        try {
            if (!isInitialized_) {
                if (!initialize()) {
                    throw std::runtime_error("工厂初始化失败");
                }
            }

            // 验证配置
            if (!validateConfiguration(config)) {
                throw std::invalid_argument("无效的数据访问服务配置");
            }
            
            // 使用简化构造函数创建统一数据访问服务实现
            // 内部会自动创建并管理所有并发优化组件
            return std::make_shared<oscean::core_services::UnifiedDataAccessServiceImpl>(
                commonServicesFactory_,
                config
            );
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建数据访问服务失败: ") + e.what());
            throw;
        }
    }
    
    std::shared_ptr<IUnifiedDataAccessService> createDataAccessServiceWithDependencies(
        const api::DataAccessConfiguration& config,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServicesFactory) override {
        
        try {
            OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "创建带依赖注入的数据访问服务");
            
            if (!commonServicesFactory) {
                OSCEAN_LOG_ERROR("DataAccessServiceFactory", "Common services factory is null");
                throw std::invalid_argument("Common services factory cannot be null");
            }
            
            // 验证配置
            if (!validateConfiguration(config)) {
                OSCEAN_LOG_ERROR("DataAccessServiceFactory", "Invalid configuration for dependency injection");
                throw std::invalid_argument("Invalid configuration");
            }
            
            // 使用简化构造函数创建统一数据访问服务实现
            // 内部会自动创建并管理所有并发优化组件
            auto serviceImpl = std::make_shared<oscean::core_services::UnifiedDataAccessServiceImpl>(
                commonServicesFactory, config);
            
            OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "带依赖注入的数据访问服务创建成功");
            return serviceImpl;
            
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建带依赖注入的数据访问服务失败: ") + e.what());
            throw;
        }
    }
    
    std::shared_ptr<IUnifiedDataAccessService> createForProduction() override {
        try {
            OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "创建生产环境数据访问服务");
            auto config = api::DataAccessConfiguration::createForProduction();
            return createDataAccessService(config);
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建生产环境数据访问服务失败: ") + e.what());
            throw;
        }
    }
    
    std::shared_ptr<IUnifiedDataAccessService> createForTesting() override {
        try {
            OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "创建测试环境数据访问服务");
            auto config = api::DataAccessConfiguration::createForTesting();
            return createDataAccessService(config);
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建测试环境数据访问服务失败: ") + e.what());
            throw;
        }
    }
    
    std::shared_ptr<IUnifiedDataAccessService> createForDevelopment() override {
        try {
            OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "创建开发环境数据访问服务");
            auto config = api::DataAccessConfiguration::createForDevelopment();
            return createDataAccessService(config);
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建开发环境数据访问服务失败: ") + e.what());
            throw;
        }
    }
    
    std::shared_ptr<IUnifiedDataAccessService> createForHPC() override {
        try {
            OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "创建HPC环境数据访问服务");
            auto config = api::DataAccessConfiguration::createForHPC();
            return createDataAccessService(config);
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建HPC环境数据访问服务失败: ") + e.what());
            throw;
        }
    }
    
    /**
     * @brief 创建生产环境优化的数据访问服务
     * 使用简化构造函数，内部自动创建GDAL预热、文件锁定、读取器池化等并发优化组件
     */
    std::shared_ptr<IUnifiedDataAccessService> createForProductionWithConcurrencyOptimization(
        const api::DataAccessConfiguration& config) override {
        
        try {
            OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "创建生产环境并发优化的数据访问服务");
            
            // 验证配置
            if (!validateConfiguration(config)) {
                throw std::invalid_argument("Invalid configuration for production with concurrency optimization");
            }
            
            // 使用简化构造函数，内部会自动创建并管理所有并发优化组件
            auto serviceImpl = std::make_shared<oscean::core_services::UnifiedDataAccessServiceImpl>(
                commonServicesFactory_, config);
            
            OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "生产环境并发优化的数据访问服务创建成功");
            return serviceImpl;
            
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建生产环境并发优化服务失败: ") + e.what());
            throw;
        }
    }
    
    api::DataAccessConfiguration getDefaultConfiguration() const override {
        return api::DataAccessConfiguration::createDefault();
    }
    
    bool validateConfiguration(const api::DataAccessConfiguration& config) const override {
        try {
            // 使用内置的配置验证方法
            return config.isValid();
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("配置验证异常: ") + e.what());
            return false;
        }
    }
    
    std::vector<std::string> getSupportedFormats() const override {
        return {
            "NetCDF",
            "HDF5", 
            "GeoTIFF",
            "Shapefile",
            "GeoJSON",
            "ESRI Geodatabase",
            "CSV",
            "GML"
        };
    }
    
    api::DataAccessMetrics getFactoryMetrics() const override {
        api::DataAccessMetrics metrics;
        // 简化实现 - 实际应该从内部统计收集
        return metrics;
    }
    
    bool initialize() override {
        try {
            OSCEAN_LOG_INFO("DataAccessServiceFactory", "初始化数据访问服务工厂");
            
            if (!commonServicesFactory_) {
                OSCEAN_LOG_ERROR("DataAccessServiceFactory", "CommonServicesFactory未设置");
                return false;
            }
            
            isInitialized_ = true;
            OSCEAN_LOG_INFO("DataAccessServiceFactory", "数据访问服务工厂初始化成功");
            return true;
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("工厂初始化失败: ") + e.what());
            return false;
        }
    }
    
    void shutdown() override {
        try {
            OSCEAN_LOG_INFO("DataAccessServiceFactory", "关闭数据访问服务工厂");
            isInitialized_ = false;
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("工厂关闭异常: ") + e.what());
        }
    }
    
    bool isHealthy() const override {
        return isInitialized_ && commonServicesFactory_ != nullptr;
    }
};

// =============================================================================
// 工厂创建函数实现 - 标准接口
// =============================================================================

/**
 * @brief 创建数据访问服务工厂 - 标准接口入口
 * 
 * 🎯 参考crs模块的工厂创建模式
 */
std::shared_ptr<IDataAccessServiceFactory> createDataAccessServiceFactory() {
    try {
        OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "创建数据访问服务工厂");
        return std::make_shared<DataAccessServiceFactoryImpl>();
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建数据访问服务工厂失败: ") + e.what());
        throw;
    }
}

/**
 * @brief 创建带依赖注入的数据访问服务工厂
 * 
 * 🎯 支持完全依赖注入，参考crs模块模式
 */
std::shared_ptr<IDataAccessServiceFactory> createDataAccessServiceFactoryWithDependencies(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServicesFactory) {
    
    try {
        OSCEAN_LOG_DEBUG("DataAccessServiceFactory", "创建带依赖注入的数据访问服务工厂");
        
        if (!commonServicesFactory) {
            throw std::invalid_argument("CommonServicesFactory不能为空");
        }
        
        return std::make_shared<DataAccessServiceFactoryImpl>(commonServicesFactory);
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("DataAccessServiceFactory", std::string("创建带依赖注入的数据访问服务工厂失败: ") + e.what());
        throw;
    }
}

} // namespace oscean::core_services::data_access 