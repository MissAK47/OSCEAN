/**
 * @file common_utils.cpp
 * @brief Common Utilities 模块实现
 * 
 * 实现模块的初始化、清理和版本管理功能
 */

#include "common_utils/common_utils.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include <atomic>
#include <mutex>
#include <iostream>

namespace oscean::common_utils {

namespace {
    std::atomic<bool> g_initialized{false};
    std::mutex g_initMutex;
    
    constexpr const char* VERSION = "2.0.0";
    constexpr const char* BUILD_DATE = __DATE__;
    constexpr const char* BUILD_TIME = __TIME__;
    
    // 全局工厂实例，仅在初始化时使用
    std::unique_ptr<infrastructure::CommonServicesFactory> g_factory;
}

bool initialize(const std::string& config) {
    std::lock_guard<std::mutex> lock(g_initMutex);
    
    if (g_initialized.load()) {
        return true;  // 已经初始化
    }
    
    try {
        // 1. 初始化日志系统
        auto logger = getLogger(); // 使用无参数版本
        
        if (logger) {
            logger->info("Initializing Common Utilities module v{}", VERSION);
            logger->info("Build: {} {}", BUILD_DATE, BUILD_TIME);
        }
        
        // 2. 初始化服务工厂
        g_factory = std::make_unique<infrastructure::CommonServicesFactory>();
        
        // 3. 处理配置参数
        if (!config.empty()) {
            if (logger) {
                logger->info("Applying configuration: {}", config);
            }
            // 这里可以解析和应用配置
        }
        
        g_initialized.store(true);
        
        if (logger) {
            logger->info("Common Utilities module initialized successfully");
        }
        
        return true;
        
    } catch (const std::exception& e) {
        auto logger = getLogger();
        if (logger) {
            logger->error("Failed to initialize Common Utilities module: {}", e.what());
        }
        return false;
    }
}

void cleanup() {
    std::lock_guard<std::mutex> lock(g_initMutex);
    
    if (!g_initialized.load()) {
        return;  // 未初始化或已清理
    }
    
    try {
        auto logger = getLogger();
        if (logger) {
            logger->info("Cleaning up Common Utilities module");
        }
        
        // 1. 关闭服务工厂
        if (g_factory) {
            g_factory->shutdown();
            g_factory.reset();
        }
        
        // 2. 清理日志系统（最后清理）
        if (logger) {
            logger->info("Common Utilities module cleanup completed");
        }
        
        g_initialized.store(false);
        
    } catch (const std::exception& e) {
        // 清理过程中的异常只能记录到标准错误输出
        std::cerr << "Error during Common Utilities cleanup: " << e.what() << std::endl;
    }
}

std::string getVersion() {
    return std::string(VERSION) + " (built " + BUILD_DATE + " " + BUILD_TIME + ")";
}

} // namespace oscean::common_utils 