#include "../src/service_management/service_manager_impl.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/crs/crs_service_factory.h"
#include <iostream>
#include <chrono>
#include <thread>


void testService(const std::string& serviceName, std::function<void()> testFunc) {
    std::cout << "🔍 测试服务: " << serviceName << "..." << std::flush;
    auto start = std::chrono::steady_clock::now();
    
    try {
        testFunc();
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << " ✅ 成功 (" << duration.count() << "ms)" << std::endl;
    } catch (const std::exception& e) {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << " ❌ 失败 (" << duration.count() << "ms)" << std::endl;
        std::cout << "     错误详情: " << e.what() << std::endl;
    }
}

void testCommonServicesFactory(std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory) {
    std::cout << "\n🔧 详细测试CommonServicesFactory各组件:" << std::endl;
    
    testService("CommonFactory健康检查", [&]() {
        bool healthy = commonFactory->isHealthy();
        if (!healthy) {
            auto messages = commonFactory->getDiagnosticMessages();
            std::cout << "\n       健康检查失败，诊断信息:" << std::endl;
            for (const auto& msg : messages) {
                std::cout << "         - " << msg << std::endl;
            }
            throw std::runtime_error("CommonServicesFactory健康检查失败");
        }
    });
    
    testService("MemoryManager", [&]() {
        auto memoryManager = commonFactory->getMemoryManager();
        if (!memoryManager) throw std::runtime_error("MemoryManager is nullptr");
    });
    
    testService("ThreadPoolManager", [&]() {
        auto threadPoolManager = commonFactory->getThreadPoolManager();
        if (!threadPoolManager) throw std::runtime_error("ThreadPoolManager is nullptr");
    });
    
    testService("SIMDManager", [&]() {
        auto simdManager = commonFactory->getSIMDManager();
        if (!simdManager) throw std::runtime_error("SIMDManager is nullptr");
    });
    
    testService("Logger", [&]() {
        auto logger = commonFactory->getLogger();
        if (!logger) throw std::runtime_error("Logger is nullptr");
    });
    
    testService("PerformanceMonitor", [&]() {
        auto performanceMonitor = commonFactory->getPerformanceMonitor();
        if (!performanceMonitor) throw std::runtime_error("PerformanceMonitor is nullptr");
    });
}

void testCrsServiceFactory(std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory) {
    std::cout << "\n🎯 详细测试CRS服务工厂:" << std::endl;
    
    std::unique_ptr<oscean::core_services::crs::CrsServiceFactory> crsFactory;
    
    testService("CRS工厂创建", [&]() {
        auto config = oscean::core_services::crs::CrsServiceConfig::createDefault();
        crsFactory = std::make_unique<oscean::core_services::crs::CrsServiceFactory>(commonFactory, config);
        if (!crsFactory) throw std::runtime_error("CrsServiceFactory创建失败");
    });
    
    if (crsFactory) {
        testService("CRS工厂健康检查", [&]() {
            bool healthy = crsFactory->isHealthy();
            if (!healthy) {
                auto messages = crsFactory->getDiagnosticMessages();
                std::cout << "\n       CRS工厂健康检查失败，诊断信息:" << std::endl;
                for (const auto& msg : messages) {
                    std::cout << "         - " << msg << std::endl;
                }
                throw std::runtime_error("CRS工厂健康检查失败");
            }
        });
        
        testService("CRS服务创建", [&]() {
            auto crsService = crsFactory->createCrsService();
            if (!crsService) throw std::runtime_error("CRS服务创建失败 - 返回nullptr");
        });
    }
}

int main() {
    // 设置控制台UTF-8编码
    system("chcp 65001 > nul");
    
    // 初始化日志系统
    try {
        oscean::common_utils::LoggingManager::configureGlobal(
            oscean::common_utils::LoggingConfig{}
        );
    } catch (const std::exception& e) {
        std::cerr << "⚠️ 日志系统初始化失败: " << e.what() << std::endl;
    }
    
    std::cout << "\n🔧 OSCEAN 服务管理器诊断程序" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    try {
        // 1. 创建生产配置的CommonServicesFactory
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory;
        testService("CommonServicesFactory", [&]() {
            auto productionConfig = oscean::common_utils::infrastructure::ServiceConfiguration::createDefault();
            commonFactory = std::make_shared<oscean::common_utils::infrastructure::CommonServicesFactory>(productionConfig);
            if (!commonFactory) throw std::runtime_error("CommonServicesFactory创建失败");
        });
        
        if (commonFactory) {
            // 详细测试CommonServicesFactory
            testCommonServicesFactory(commonFactory);
            
            // 详细测试CRS服务工厂
            testCrsServiceFactory(commonFactory);
        }
        
        // 2. 创建ServiceManager
        std::shared_ptr<oscean::workflow_engine::service_management::ServiceManagerImpl> serviceManager;
        testService("ServiceManager", [&]() {
            auto productionConfig = oscean::common_utils::infrastructure::ServiceConfiguration::createDefault();
            auto commonFactory = std::make_shared<oscean::common_utils::infrastructure::CommonServicesFactory>(productionConfig);
            auto threadPoolManager = commonFactory->getUnifiedThreadPoolManager();
            serviceManager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
            if (!serviceManager) throw std::runtime_error("ServiceManager创建失败");
        });
        
        if (!serviceManager) {
            std::cout << "❌ ServiceManager创建失败，无法继续测试" << std::endl;
            return 1;
        }
        
        // 3. 测试各个服务的获取
        testService("数据访问服务", [&]() {
            auto service = serviceManager->getService<oscean::core_services::data_access::IUnifiedDataAccessService>();
            if (!service) throw std::runtime_error("数据访问服务获取失败");
        });
        
        testService("CRS服务", [&]() {
            try {
                std::cout << "\n      → 正在调用getService<ICrsService>()..." << std::endl;
                auto service = serviceManager->getService<oscean::core_services::ICrsService>();
                std::cout << "      → getService调用完成，检查结果..." << std::endl;
                if (!service) {
                    throw std::runtime_error("CRS服务返回nullptr");
                }
                std::cout << "      → CRS服务获取成功！" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "      → CRS服务获取过程中发生异常: " << e.what() << std::endl;
                throw;
            }
        });
        
        testService("元数据服务", [&]() {
            try {
                std::cout << "\n      → 正在调用getService<IMetadataService>()..." << std::endl;
                auto service = serviceManager->getService<oscean::core_services::metadata::IMetadataService>();
                std::cout << "      → getService调用完成，检查结果..." << std::endl;
                if (!service) {
                    throw std::runtime_error("元数据服务返回nullptr");
                }
                std::cout << "      → 元数据服务获取成功！" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "      → 元数据服务获取过程中发生异常: " << e.what() << std::endl;
                throw;
            }
        });
        
        std::cout << "\n✅ 所有服务测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ 诊断程序异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 