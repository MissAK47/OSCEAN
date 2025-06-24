#include "gdal_manager.h"

// GDAL/PROJ 头文件
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <cpl_conv.h>
#include <proj.h>

// 标准库
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <iostream>
#include <filesystem>

// 日志系统
#include "common_utils/utilities/logging_utils.h"

namespace oscean::core_services::crs::impl {

// 静态成员变量
static std::once_flag g_gdalInitFlag;
static std::atomic<bool> g_gdalInitialized{false};

bool GDALManager::ensureInitialized() {
    bool success = true;
    
    std::call_once(g_gdalInitFlag, [&success]() {
        try {
            OSCEAN_LOG_INFO("GDALManager", "🔧 开始GDAL环境初始化...");
            
            success = performInitialization();
            
            if (success) {
                g_gdalInitialized.store(true);
                OSCEAN_LOG_INFO("GDALManager", "✅ GDAL环境初始化成功，驱动程序数量: {}", GDALGetDriverCount());
            } else {
                OSCEAN_LOG_ERROR("GDALManager", "❌ GDAL环境初始化失败");
            }
            
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("GDALManager", "❌ GDAL环境初始化异常: {}", e.what());
            success = false;
        }
    });
    
    return success && g_gdalInitialized.load();
}

bool GDALManager::isInitialized() {
    return g_gdalInitialized.load() && (GDALGetDriverCount() > 0);
}

int GDALManager::getDriverCount() {
    return GDALGetDriverCount();
}

bool GDALManager::performInitialization() {
    try {
        // 🔧 检查GDAL是否已由全局初始化器初始化
        OSCEAN_LOG_DEBUG("GDALManager", "📍 检查GDAL全局初始化状态...");
        
        if (GDALGetDriverCount() == 0) {
            OSCEAN_LOG_ERROR("GDALManager", "❌ GDAL未初始化！请确保在main函数中调用了GdalGlobalInitializer::initialize()");
            return false;
        }
        
        OSCEAN_LOG_INFO("GDALManager", "✅ GDAL已由全局初始化器初始化，驱动数量: {}", GDALGetDriverCount());
        
        // 移除所有分散的GDAL初始化和全局配置调用
        // CPLSetConfigOption("GDAL_PAM_ENABLED", "NO"); // ❌ 已移除 - 现在由GdalGlobalInitializer统一管理
        // CPLSetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"); // ❌ 已移除
        // CPLSetConfigOption("GDAL_CACHEMAX", "256"); // ❌ 已移除
        // GDALAllRegister(); // ❌ 已移除
        // OGRRegisterAll();  // ❌ 已移除
        
        // 🔧 PROJ配置现在由GdalGlobalInitializer统一管理
        OSCEAN_LOG_DEBUG("GDALManager", "📍 PROJ配置由全局初始化器管理...");
        
        // 检查PROJ数据路径（仅用于验证）
        const char* projDataPath = std::getenv("PROJ_LIB");
        if (projDataPath) {
            OSCEAN_LOG_DEBUG("GDALManager", "📍 PROJ数据路径: {}", projDataPath);
        } else {
            OSCEAN_LOG_DEBUG("GDALManager", "📍 PROJ_LIB环境变量未设置");
        }
        
        // 移除全局配置调用
        // CPLSetConfigOption("PROJ_LIB", projDataPath); // ❌ 已移除
        
        // 🔧 第五步：验证初始化结果
        int gdalDrivers = GDALGetDriverCount();
        int ogrDrivers = OGRGetDriverCount();
        
        OSCEAN_LOG_INFO("GDALManager", "📊 GDAL驱动程序: {}, OGR驱动程序: {}", gdalDrivers, ogrDrivers);
        
        if (gdalDrivers == 0) {
            OSCEAN_LOG_ERROR("GDALManager", "❌ GDAL驱动程序注册失败");
            return false;
        }
        
        // 🔧 第六步：测试基本功能
        OSCEAN_LOG_DEBUG("GDALManager", "📍 测试GDAL基本功能...");
        
        // 尝试获取一个常见的驱动程序
        GDALDriverH memDriver = GDALGetDriverByName("MEM");
        if (!memDriver) {
            OSCEAN_LOG_WARN("GDALManager", "⚠️ 内存驱动程序不可用");
        } else {
            OSCEAN_LOG_DEBUG("GDALManager", "✅ 内存驱动程序可用");
        }
        
        OSCEAN_LOG_INFO("GDALManager", "🎉 GDAL环境初始化完成");
        return true;
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("GDALManager", "❌ GDAL初始化过程中发生异常: {}", e.what());
        return false;
    } catch (...) {
        OSCEAN_LOG_ERROR("GDALManager", "❌ GDAL初始化过程中发生未知异常");
        return false;
    }
}

} // namespace oscean::core_services::crs::impl 