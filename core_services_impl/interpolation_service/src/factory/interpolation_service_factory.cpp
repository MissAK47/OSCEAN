#include "interpolation_service_factory.h"
#include "../impl/interpolation_service_impl.cpp"  // 直接包含实现，因为类定义在cpp文件中
#include "common_utils/simd/simd_manager_unified.h"

namespace oscean::core_services::interpolation {

std::unique_ptr<IInterpolationService> InterpolationServiceFactory::createDefault() {
    // 创建默认的SIMD管理器
    auto simdManager = boost::make_shared<common_utils::simd::UnifiedSIMDManager>();
    return createWithSIMDManager(simdManager, true);
}

std::unique_ptr<IInterpolationService> InterpolationServiceFactory::createService() {
    return createDefault();
}

std::unique_ptr<IInterpolationService> InterpolationServiceFactory::createWithSIMDManager(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    bool enableSmartSelection) {
    
    return std::make_unique<InterpolationServiceImpl>(
        std::move(simdManager),
        enableSmartSelection,
        true  // enableGPUAcceleration
    );
}

std::unique_ptr<IInterpolationService> InterpolationServiceFactory::createFromConfig(
    const std::string& configPath,
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager) {
    
    // TODO: 实现从配置文件加载
    // 暂时返回默认实现
    if (!simdManager) {
        simdManager = boost::make_shared<common_utils::simd::UnifiedSIMDManager>();
    }
    return createWithSIMDManager(simdManager, true);
}

std::unique_ptr<IInterpolationService> InterpolationServiceFactory::createService(
    const InterpolationServiceConfig& config) {
    
    auto simdManager = boost::make_shared<common_utils::simd::UnifiedSIMDManager>();
    return std::make_unique<InterpolationServiceImpl>(
        simdManager,
        config.enableSmartSelection,
        true  // enableGPUAcceleration
    );
}

std::unique_ptr<IInterpolationService> InterpolationServiceFactory::createHighPerformanceService(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager) {
    
    auto config = getHighPerformanceConfig();
    return std::make_unique<InterpolationServiceImpl>(
        std::move(simdManager),
        config.enableSmartSelection,
        true  // enableGPUAcceleration
    );
}

std::unique_ptr<IInterpolationService> InterpolationServiceFactory::createHighAccuracyService() {
    auto config = getHighAccuracyConfig();
    auto simdManager = boost::make_shared<common_utils::simd::UnifiedSIMDManager>();
    return std::make_unique<InterpolationServiceImpl>(
        simdManager,
        config.enableSmartSelection,
        false  // 禁用GPU以确保精度
    );
}

InterpolationServiceConfig InterpolationServiceFactory::getDefaultConfig() {
    InterpolationServiceConfig config;
    config.enableSmartSelection = true;
    config.enableSIMDOptimization = true;
    config.maxCacheSize = 1000;
    config.performanceThreshold = 100.0;
    return config;
}

InterpolationServiceConfig InterpolationServiceFactory::getHighPerformanceConfig() {
    InterpolationServiceConfig config;
    config.enableSmartSelection = true;
    config.enableSIMDOptimization = true;
    config.maxCacheSize = 5000;        // 更大的缓存
    config.performanceThreshold = 50.0; // 更严格的性能要求
    return config;
}

InterpolationServiceConfig InterpolationServiceFactory::getHighAccuracyConfig() {
    InterpolationServiceConfig config;
    config.enableSmartSelection = true;
    config.enableSIMDOptimization = false; // 禁用SIMD以确保精度
    config.maxCacheSize = 500;             // 较小的缓存
    config.performanceThreshold = 500.0;   // 放宽性能要求
    return config;
}

} // namespace oscean::core_services::interpolation 