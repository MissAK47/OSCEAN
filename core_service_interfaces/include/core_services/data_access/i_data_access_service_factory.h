/**
 * @file i_data_access_service_factory.h
 * @brief 数据访问服务工厂接口 - 阶段1：统一对外接口架构
 * 
 * 🎯 重构目标：
 * ✅ 统一创建管理 - 所有数据访问服务通过工厂创建
 * ✅ 依赖注入支持 - 支持外部依赖注入
 * ✅ 配置管理 - 统一配置管理和环境适配
 * ✅ 生命周期管理 - 统一服务生命周期管理
 */

#pragma once

#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <map>
#include "i_unified_data_access_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "unified_data_types.h"
#include "../common_data_types.h"

// 前向声明 - 添加统一线程池管理器
namespace oscean::common_utils::infrastructure {
    class CommonServicesFactory;
    class UnifiedThreadPoolManager;  // 🔧 新增前向声明
}

namespace oscean::core_services::data_access {

// 前向声明
namespace api {
    struct DataAccessMetrics;
}

// =============================================================================
// 配置结构体 - 移到前面避免前向声明问题
// =============================================================================

namespace api {

/**
 * @brief 数据访问配置
 */
struct DataAccessConfiguration {
    // 基础配置
    std::string serviceName = "DataAccessService";                 ///< 服务名称
    std::string version = "1.0.0";                                 ///< 版本号
    
    // 性能配置
    size_t maxConcurrentRequests = 100;                           ///< 最大并发请求数
    size_t threadPoolSize = 0;                                    ///< 线程池大小（0=自动检测）
    size_t maxMemoryUsageMB = 1024;                               ///< 最大内存使用量（MB）
    
    // 🔧 新增：共享线程池管理器支持
    std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> sharedThreadPoolManager = nullptr;
    
    // 缓存配置
    bool enableCaching = true;                                     ///< 是否启用缓存
    size_t gridDataCacheSize = 10000;                            ///< 格点数据缓存大小
    size_t metadataCacheSize = 5000;                              ///< 元数据缓存大小
    size_t readerCacheSize = 100;                                ///< 读取器缓存大小
    std::string cacheStrategy = "LRU";                           ///< 缓存策略
    
    // 日志配置
    std::string logLevel = "info";                                ///< 日志级别
    bool enablePerformanceLogging = true;                        ///< 是否启用性能日志
    
    // 数据处理配置
    bool enableSIMD = true;                                       ///< 是否启用SIMD优化
    bool enableCompression = false;                               ///< 是否启用数据压缩
    size_t defaultChunkSize = 1024 * 1024;                       ///< 默认数据块大小（字节）
    
    // 超时配置
    double requestTimeoutSeconds = 30.0;                         ///< 请求超时时间（秒）
    double connectionTimeoutSeconds = 10.0;                      ///< 连接超时时间（秒）
    
    // 重试配置
    size_t maxRetryAttempts = 3;                                  ///< 最大重试次数
    double retryDelaySeconds = 1.0;                              ///< 重试延迟（秒）
    
    // 支持的格式配置
    std::vector<std::string> enabledFormats = {                  ///< 启用的数据格式
        "NetCDF", "HDF5", "GeoTIFF", "Shapefile", "GeoJSON"
    };
    
    // 环境特定配置
    std::map<std::string, std::string> environmentSettings;      ///< 环境特定设置
    
    /**
     * @brief 创建默认配置
     */
    static DataAccessConfiguration createDefault() {
        return DataAccessConfiguration{};
    }
    
    /**
     * @brief 创建生产环境配置
     */
    static DataAccessConfiguration createForProduction() {
        DataAccessConfiguration config = createDefault();
        config.maxConcurrentRequests = 200;
        config.maxMemoryUsageMB = 2048;
        config.logLevel = "warn";
        config.enablePerformanceLogging = true;
        config.requestTimeoutSeconds = 60.0;
        return config;
    }
    
    /**
     * @brief 创建测试环境配置
     */
    static DataAccessConfiguration createForTesting() {
        DataAccessConfiguration config = createDefault();
        config.maxConcurrentRequests = 10;
        config.maxMemoryUsageMB = 256;
        config.logLevel = "debug";
        config.enableCaching = false;  // 测试时禁用缓存
        config.requestTimeoutSeconds = 5.0;
        return config;
    }
    
    /**
     * @brief 创建开发环境配置
     */
    static DataAccessConfiguration createForDevelopment() {
        DataAccessConfiguration config = createDefault();
        config.maxConcurrentRequests = 20;
        config.maxMemoryUsageMB = 512;
        config.logLevel = "debug";
        config.enablePerformanceLogging = true;
        config.requestTimeoutSeconds = 10.0;
        return config;
    }
    
    /**
     * @brief 创建HPC环境配置
     */
    static DataAccessConfiguration createForHPC() {
        DataAccessConfiguration config = createDefault();
        config.maxConcurrentRequests = 500;
        config.threadPoolSize = 32;  // 高性能环境使用更多线程
        config.maxMemoryUsageMB = 8192;
        config.logLevel = "error";  // 减少日志开销
        config.enableSIMD = true;
        config.enableCompression = true;
        config.defaultChunkSize = 4 * 1024 * 1024;  // 更大的数据块
        return config;
    }
    
    /**
     * @brief 验证配置有效性
     */
    bool isValid() const {
        return maxConcurrentRequests > 0 &&
               maxMemoryUsageMB > 0 &&
               requestTimeoutSeconds > 0 &&
               connectionTimeoutSeconds > 0 &&
               !serviceName.empty() &&
               !version.empty();
    }
};

} // namespace api

/**
 * @brief 数据访问服务工厂接口
 * 
 * 🎯 设计原则：
 * ✅ 统一创建入口 - 所有数据访问服务通过工厂创建
 * ✅ 依赖注入支持 - 支持Common模块和其他服务的注入
 * ✅ 配置驱动 - 通过配置控制服务行为
 * ✅ 环境适配 - 支持开发、测试、生产环境
 */
class IDataAccessServiceFactory {
public:
    virtual ~IDataAccessServiceFactory() = default;

    // =============================================================================
    // 服务创建方法
    // =============================================================================

    /**
     * @brief 创建统一数据访问服务
     * 
     * @param config 服务配置
     * @return 数据访问服务实例
     */
    virtual std::shared_ptr<IUnifiedDataAccessService> createDataAccessService(
        const api::DataAccessConfiguration& config = api::DataAccessConfiguration::createDefault()) = 0;

    /**
     * @brief 创建带依赖注入的数据访问服务
     * 
     * @param config 服务配置
     * @param commonServicesFactory Common模块服务工厂
     * @return 数据访问服务实例
     */
    virtual std::shared_ptr<IUnifiedDataAccessService> createDataAccessServiceWithDependencies(
        const api::DataAccessConfiguration& config,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServicesFactory) = 0;

    // =============================================================================
    // 环境特定创建方法
    // =============================================================================

    /**
     * @brief 创建生产环境数据访问服务
     * 
     * @return 生产环境优化的服务实例
     */
    virtual std::shared_ptr<IUnifiedDataAccessService> createForProduction() = 0;

    /**
     * @brief 创建测试环境数据访问服务
     * 
     * @return 测试环境优化的服务实例
     */
    virtual std::shared_ptr<IUnifiedDataAccessService> createForTesting() = 0;

    /**
     * @brief 创建开发环境数据访问服务
     * 
     * @return 开发环境优化的服务实例
     */
    virtual std::shared_ptr<IUnifiedDataAccessService> createForDevelopment() = 0;

    /**
     * @brief 创建高性能计算环境数据访问服务
     * 
     * @return HPC环境优化的服务实例
     */
    virtual std::shared_ptr<IUnifiedDataAccessService> createForHPC() = 0;

    /**
     * @brief 创建生产环境并发优化的数据访问服务
     * 
     * 🎯 新增功能：
     * ✅ GDAL多线程预热初始化
     * ✅ 文件级锁定机制（避免文件竞争）
     * ✅ 读取器池化系统（提升性能）
     * ✅ 完全基于依赖注入（可测试、可配置）
     * 
     * @param config 数据访问配置
     * @return 生产环境并发优化的服务实例
     */
    virtual std::shared_ptr<IUnifiedDataAccessService> createForProductionWithConcurrencyOptimization(
        const api::DataAccessConfiguration& config) = 0;

    // =============================================================================
    // 配置和管理方法
    // =============================================================================

    /**
     * @brief 获取默认配置
     * 
     * @return 默认配置
     */
    virtual api::DataAccessConfiguration getDefaultConfiguration() const = 0;

    /**
     * @brief 验证配置
     * 
     * @param config 要验证的配置
     * @return 配置是否有效
     */
    virtual bool validateConfiguration(const api::DataAccessConfiguration& config) const = 0;

    /**
     * @brief 获取支持的数据格式列表
     * 
     * @return 支持的格式列表
     */
    virtual std::vector<std::string> getSupportedFormats() const = 0;

    /**
     * @brief 获取工厂统计信息
     * 
     * @return 工厂统计信息
     */
    virtual api::DataAccessMetrics getFactoryMetrics() const = 0;

    // =============================================================================
    // 生命周期管理
    // =============================================================================

    /**
     * @brief 初始化工厂
     * 
     * @return 是否初始化成功
     */
    virtual bool initialize() = 0;

    /**
     * @brief 关闭工厂
     */
    virtual void shutdown() = 0;

    /**
     * @brief 检查工厂健康状态
     * 
     * @return 是否健康
     */
    virtual bool isHealthy() const = 0;
};

} // namespace oscean::core_services::data_access

// =============================================================================
// 🏭 标准工厂创建函数声明 - 参考crs模块模式
// =============================================================================

namespace oscean::core_services::data_access {

/**
 * @brief 创建数据访问服务工厂 - 标准接口入口
 * 
 * 🎯 参考crs模块的工厂创建模式
 * 
 * @return 数据访问服务工厂实例
 */
std::shared_ptr<IDataAccessServiceFactory> createDataAccessServiceFactory();

/**
 * @brief 创建带依赖注入的数据访问服务工厂
 * 
 * 🎯 支持完全依赖注入，参考crs模块模式
 * 
 * @param commonServicesFactory Common模块服务工厂
 * @return 数据访问服务工厂实例
 */
std::shared_ptr<IDataAccessServiceFactory> createDataAccessServiceFactoryWithDependencies(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServicesFactory);

} // namespace oscean::core_services::data_access 