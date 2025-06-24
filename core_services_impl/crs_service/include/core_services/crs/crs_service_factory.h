/**
 * @file crs_service_factory.h
 * @brief CRS服务工厂 - 纯工厂+依赖注入模式
 * 
 * 🎯 重构核心：
 * ✅ 移除单例模式，改为纯工厂
 * ✅ 支持完全的依赖注入
 * ✅ 集成CommonServicesFactory依赖
 * ✅ 可测试和可配置
 * ✅ 明确的生命周期管理
 */

#pragma once

#include "core_services/crs/i_crs_service.h"
#include "common_utils/infrastructure/common_services_factory.h"

#include <memory>
#include <string>
#include <stdexcept>

namespace oscean::core_services::crs {

/**
 * @brief CRS服务配置
 */
struct CrsServiceConfig {
    // 基础配置
    std::string defaultSourceEpsg = "EPSG:4326";
    std::string defaultTargetEpsg = "EPSG:3857";
    bool enableCaching = true;
    bool enablePerformanceOptimization = true;
    
    // 性能配置
    size_t maxCacheSize = 10000;
    size_t threadPoolSize = 4;
    bool enableSIMD = true;
    bool enableBatchProcessing = true;
    
    // 流式处理配置
    size_t streamingBufferSize = 1024;
    bool enableMemoryMappedProcessing = true;
    
    // 质量保证配置
    double transformationTolerance = 1e-6;
    bool enableStrictValidation = true;
    
    // 🔧 修正：添加缺失的配置字段
    bool enableSIMDOptimization = true;
    size_t batchSize = 1000;
    size_t maxMemoryUsageMB = 512;
    
    static CrsServiceConfig createDefault() {
        return CrsServiceConfig{};
    }
    
    static CrsServiceConfig createForTesting() {
        CrsServiceConfig config;
        config.maxCacheSize = 100;
        config.threadPoolSize = 2;
        config.enableStrictValidation = false;
        config.transformationTolerance = 1e-3;
        config.maxMemoryUsageMB = 64;
        config.batchSize = 100;
        return config;
    }
    
    static CrsServiceConfig createHighPerformance() {
        CrsServiceConfig config;
        config.maxCacheSize = 50000;
        config.threadPoolSize = 8;
        config.enableSIMD = true;
        config.enableBatchProcessing = true;
        config.streamingBufferSize = 4096;
        config.maxMemoryUsageMB = 1024;
        config.batchSize = 5000;
        return config;
    }
    
    // 🆕 添加低内存配置
    static CrsServiceConfig createLowMemory() {
        CrsServiceConfig config;
        config.maxCacheSize = 1000;
        config.threadPoolSize = 2;
        config.enableSIMD = false;
        config.enableBatchProcessing = false;
        config.streamingBufferSize = 256;
        config.maxMemoryUsageMB = 128;
        config.batchSize = 100;
        return config;
    }
};

/**
 * @brief CRS工厂异常类
 */
class CrsFactoryException : public std::runtime_error {
public:
    explicit CrsFactoryException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @class CrsServiceFactory
 * @brief 纯工厂模式的CRS服务创建器 - 支持完全依赖注入
 *
 * 🎯 核心设计原则：
 * ✅ 无全局状态，完全无状态工厂
 * ✅ 支持依赖注入和控制反转
 * ✅ 可测试和可配置
 * ✅ 集成CommonServicesFactory
 * ✅ 明确的生命周期管理
 */
class CrsServiceFactory {
public:
    /**
     * @brief 构造函数 - 支持依赖注入
     * @param commonFactory Common服务工厂实例
     * @param config CRS服务配置
     */
    explicit CrsServiceFactory(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory,
        const CrsServiceConfig& config = CrsServiceConfig::createDefault()
    );
    
    /**
     * @brief 移动构造函数
     */
    CrsServiceFactory(CrsServiceFactory&& other) noexcept;
    
    /**
     * @brief 移动赋值运算符
     */
    CrsServiceFactory& operator=(CrsServiceFactory&& other) noexcept;
    
    /**
     * @brief 删除拷贝构造和赋值
     */
    CrsServiceFactory(const CrsServiceFactory&) = delete;
    CrsServiceFactory& operator=(const CrsServiceFactory&) = delete;
    
    /**
     * @brief 析构函数
     */
    ~CrsServiceFactory();
    
    // === 🏭 静态工厂方法 ===
    
    /**
     * @brief 创建用于测试的CRS工厂
     */
    static std::unique_ptr<CrsServiceFactory> createForTesting(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory = nullptr
    );
    
    /**
     * @brief 创建高性能CRS工厂
     */
    static std::unique_ptr<CrsServiceFactory> createHighPerformance(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory
    );
    
    // === 🎯 核心服务创建 ===
    
    /**
     * @brief 创建标准CRS服务
     */
    std::unique_ptr<ICrsService> createCrsService();
    
    /**
     * @brief 创建带配置的CRS服务
     */
    std::unique_ptr<ICrsService> createCrsService(const CrsServiceConfig& config);
    
    /**
     * @brief 创建高性能CRS服务
     */
    std::unique_ptr<ICrsService> createHighPerformanceCrsService();
    
    /**
     * @brief 创建优化的CRS服务
     */
    std::unique_ptr<ICrsService> createOptimizedCrsService();
    
    /**
     * @brief 创建用于测试的CRS服务
     */
    std::unique_ptr<ICrsService> createTestingCrsService();
    
    /**
     * @brief 创建流式处理CRS服务
     */
    std::unique_ptr<ICrsService> createStreamingCrsService();
    
    /**
     * @brief 创建批处理CRS服务
     */
    std::unique_ptr<ICrsService> createBatchProcessingCrsService();
    
    /**
     * @brief 创建低内存CRS服务
     */
    std::unique_ptr<ICrsService> createLowMemoryCrsService();
    
    /**
     * @brief 创建模拟服务（用于测试）
     */
    std::unique_ptr<ICrsService> createMockService();
    
    // === 🔧 配置访问 ===
    
    /**
     * @brief 获取当前配置
     */
    const CrsServiceConfig& getConfiguration() const { return config_; }
    
    /**
     * @brief 更新配置
     */
    void updateConfiguration(const CrsServiceConfig& config);
    
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
    static bool validateConfig(const CrsServiceConfig& config);
    
    /**
     * @brief 获取最优配置
     */
    static CrsServiceConfig getOptimalConfig();
    
    /**
     * @brief 检查资源可用性
     */
    static bool checkResourceAvailability(const CrsServiceConfig& config);

private:
    // === 配置和依赖 ===
    CrsServiceConfig config_;
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory_;
    
    // === 内部辅助方法 ===
    void validateConfiguration(const CrsServiceConfig& config);
    std::unique_ptr<ICrsService> createCrsServiceWithConfig(const CrsServiceConfig& config);
    void setupLogging();
    void validateCommonFactory();
};

} // namespace oscean::core_services::crs 