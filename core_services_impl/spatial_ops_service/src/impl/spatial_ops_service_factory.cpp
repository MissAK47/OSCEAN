#include "core_services/spatial_ops/spatial_ops_service_factory.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "spatial_ops_service_impl.h"
#include "spatial_config_manager.h"
#include "core_services/exceptions.h"

// GDAL includes
#ifdef GDAL_FOUND
#include "gdal_priv.h"
#include "ogrsf_frmts.h"
#include "cpl_conv.h"
#include "cpl_error.h"
#endif

#include <iostream>
#include <memory>
#include <stdexcept>

namespace oscean::core_services::spatial_ops {

std::unique_ptr<ISpatialOpsService> SpatialOpsServiceFactory::createService() {
    auto config = getDefaultConfig();
    return createService(config);
}

std::unique_ptr<ISpatialOpsService> SpatialOpsServiceFactory::createService(
    const SpatialOpsConfig& config) {
    try {
        // 验证配置
        if (!validateConfig(config)) {
            throw oscean::core_services::SpatialOpsException("Invalid spatial ops configuration");
        }
        
        // 创建服务实例
        return std::make_unique<impl::SpatialOpsServiceImpl>(config);
        
    } catch (const std::bad_alloc& ba_ex) {
        throw oscean::core_services::SpatialOpsException("Failed to allocate memory for SpatialOpsService: " + std::string(ba_ex.what()));
    } catch (const oscean::core_services::SpatialOpsException&) {
        throw; // 重新抛出空间服务异常
    } catch (const std::exception& ex) {
        throw oscean::core_services::SpatialOpsException("An unexpected error occurred during SpatialOpsService creation: " + std::string(ex.what()));
    }
}

std::unique_ptr<ISpatialOpsService> SpatialOpsServiceFactory::createService(
    const SpatialOpsConfig& config,
    std::shared_ptr<oscean::core_services::ICrsService> crsService,
    std::shared_ptr<oscean::core_services::IRawDataAccessService> dataAccessService,
    std::shared_ptr<oscean::core_services::interpolation::IInterpolationService> interpolationService) {
    
    try {
        // 验证配置
        if (!validateConfig(config)) {
            throw oscean::core_services::SpatialOpsException("Invalid configuration provided");
        }
        
        // ❌ 删除依赖验证，因为空间服务现在是独立的
        // validateDependencies(crsService, dataAccessService, interpolationService);
        
        // 创建服务实例
        return std::make_unique<impl::SpatialOpsServiceImpl>(config);
        
    } catch (const std::bad_alloc& ba_ex) {
        throw oscean::core_services::SpatialOpsException("Failed to allocate memory for SpatialOpsService: " + std::string(ba_ex.what()));
    } catch (const oscean::core_services::SpatialOpsException&) {
        throw; // 重新抛出空间服务异常
    } catch (const std::exception& ex) {
        throw oscean::core_services::SpatialOpsException("An unexpected error occurred during SpatialOpsService creation: " + std::string(ex.what()));
    }
}

std::unique_ptr<ISpatialOpsService> SpatialOpsServiceFactory::createMockService() {
    // 创建测试配置
    SpatialOpsConfig config = getDefaultConfig();
    
    // 设置Mock模式的特殊配置
    config.enableDebugMode = true;
    
    // 创建Mock服务实例
    try {
        // 简化的GDAL初始化（仅用于测试）
        initializeGDALForTesting(config);
        
        return std::make_unique<impl::SpatialOpsServiceImpl>(config);
    } catch (const std::exception& ex) {
        throw oscean::core_services::SpatialOpsException("Failed to create mock service: " + std::string(ex.what()));
    }
}

bool SpatialOpsServiceFactory::validateConfig(const SpatialOpsConfig& config) {
    try {
        // 使用配置管理器进行验证
        SpatialConfigManager configManager;
        auto mutableConfig = config; // 创建可变副本
        return configManager.validateAndFix(mutableConfig);
    } catch (const std::exception&) {
        return false;
    }
}

SpatialOpsConfig SpatialOpsServiceFactory::getDefaultConfig() {
    return SpatialConfigManager::getDefaultConfig();
}

void SpatialOpsServiceFactory::initializeGDALForTesting(const SpatialOpsConfig& config) {
#ifdef GDAL_FOUND
    static bool testGdalInitialized = false;
    
    if (!testGdalInitialized) {
        // 检查GDAL是否已由全局初始化器初始化
        if (GDALGetDriverCount() == 0) {
            throw std::runtime_error("GDAL未初始化！请确保在main函数中调用了GdalGlobalInitializer::initialize()");
        }
        
        // 移除所有分散的GDAL初始化和全局配置调用
        // GDALAllRegister(); // ❌ 已移除 - 现在由GdalGlobalInitializer统一管理
        // OGRRegisterAll();  // ❌ 已移除
        // CPLSetConfigOption("GDAL_NUM_THREADS", "1"); // ❌ 已移除
        // CPLSetConfigOption("GDAL_CACHEMAX", "64000000"); // ❌ 已移除
        // CPLSetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "YES"); // ❌ 已移除
        // CPLSetConfigOption("CPL_CURL_ENABLE_VSIMEM", "NO"); // ❌ 已移除
        
        testGdalInitialized = true;
    }
#endif
}

void SpatialOpsServiceFactory::validateDependencies(
    std::shared_ptr<oscean::core_services::ICrsService> crsService,
    std::shared_ptr<oscean::core_services::IRawDataAccessService> dataAccessService,
    std::shared_ptr<oscean::core_services::interpolation::IInterpolationService> interpolationService) {
    
    // ❌ 删除依赖验证逻辑，因为空间服务现在是完全独立的
    // 不再需要验证其他服务的依赖
    
    // 输出信息表明空间服务运行在独立模式
    std::cout << "SpatialOpsService: Running in independent mode - no external service dependencies" << std::endl;
    
    /*
    // 以下代码已删除 - 空间服务不再依赖其他服务
    if (!crsService) {
        throw oscean::core_services::SpatialOpsException("CRS service dependency is required but not provided");
    }
    
    if (!dataAccessService) {
        // DataAccess服务是可选的，但如果提供了应该是有效的
        std::cout << "Warning: DataAccess service not provided. Some data reading features may not be available." << std::endl;
    }
    
    if (!interpolationService) {
        // Interpolation服务是可选的
        std::cout << "Info: Interpolation service not provided. Complex interpolation features will use basic implementations." << std::endl;
    }
    
    // 验证服务状态
    if (!crsService->isReady()) {
        throw oscean::core_services::SpatialOpsException("CRS service is not ready");
    }
    
    if (dataAccessService && !dataAccessService->isReady()) {
        std::cout << "Warning: DataAccess service is not ready. Some features may not work correctly." << std::endl;
    }
    
    if (interpolationService && !interpolationService->isReady()) {
        std::cout << "Warning: Interpolation service is not ready. Some features may not work correctly." << std::endl;
    }
    */
}

#ifdef GDAL_FOUND
void SpatialOpsServiceFactory::gdalErrorHandler(CPLErr eErrClass, CPLErrorNum nError, const char* pszErrorMsg) {
    // 自定义GDAL错误处理
    switch (eErrClass) {
        case CE_None:
            // 无错误，不需要处理
            break;
        case CE_Debug:
            std::cout << "GDAL Debug: " << pszErrorMsg << std::endl;
            break;
        case CE_Warning:
            std::cerr << "GDAL Warning (" << nError << "): " << pszErrorMsg << std::endl;
            break;
        case CE_Failure:
            std::cerr << "GDAL Error (" << nError << "): " << pszErrorMsg << std::endl;
            break;
        case CE_Fatal:
            std::cerr << "GDAL Fatal Error (" << nError << "): " << pszErrorMsg << std::endl;
            throw oscean::core_services::SpatialOpsException("GDAL Fatal Error: " + std::string(pszErrorMsg));
            break;
    }
}
#endif

void SpatialOpsServiceFactory::cleanup() {
#ifdef GDAL_FOUND
    // 清理GDAL资源
    GDALDestroyDriverManager();
    OGRCleanupAll();
    CPLCleanupTLS();
#endif
}

std::string SpatialOpsServiceFactory::getGDALVersion() {
#ifdef GDAL_FOUND
    return std::string(GDALVersionInfo("RELEASE_NAME"));
#else
    return "GDAL not available";
#endif
}

bool SpatialOpsServiceFactory::isGDALAvailable() {
#ifdef GDAL_FOUND
    return true;
#else
    return false;
#endif
}

std::vector<std::string> SpatialOpsServiceFactory::getSupportedFormats() {
    std::vector<std::string> formats;
    
#ifdef GDAL_FOUND
    // 获取支持的栅格格式
    GDALDriverManager* driverManager = GetGDALDriverManager();
    for (int i = 0; i < driverManager->GetDriverCount(); ++i) {
        GDALDriver* driver = driverManager->GetDriver(i);
        if (driver) {
            formats.push_back(std::string("Raster: ") + driver->GetDescription());
        }
    }
    
    // 获取支持的矢量格式
    for (int i = 0; i < OGRGetDriverCount(); ++i) {
        auto* driver = static_cast<OGRSFDriver*>(OGRGetDriver(i));
        if (driver) {
            formats.push_back(std::string("Vector: ") + driver->GetName());
        }
    }
#else
    formats.push_back("GDAL not available - limited format support");
#endif
    
    return formats;
}

} // namespace oscean::core_services::spatial_ops 