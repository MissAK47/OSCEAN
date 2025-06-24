#define OSCEAN_ENABLE_BOOST_ASIO
#include "common_utils/utilities/boost_config.h"
OSCEAN_ENABLE_BOOST_ASIO_IN_MODULE();  // 工作流引擎间接使用boost::asio（通过线程池管理器）

#include "../src/service_management/service_manager_impl.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/crs/crs_service_factory.h"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <thread>
#include <atomic>
#include <future>


#ifdef _WIN32
#include <windows.h>
#endif

class CRSDiagnosticTester {
private:
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory_;
    std::shared_ptr<oscean::workflow_engine::service_management::ServiceManagerImpl> serviceManager_;

public:
    explicit CRSDiagnosticTester() {
        setupLogging();
        initializeCommonServices();
    }

    void setupLogging() {
        try {
            oscean::common_utils::LoggingManager::configureGlobal(
                oscean::common_utils::LoggingConfig{}
            );
            std::cout << "✅ 日志系统初始化成功" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "⚠️ 日志系统初始化失败: " << e.what() << std::endl;
        }
    }

    void initializeCommonServices() {
        std::cout << "\n🔧 初始化Common服务..." << std::endl;
        
        // 创建线程池管理器
        oscean::common_utils::infrastructure::UnifiedThreadPoolManager::PoolConfiguration poolConfig;
        poolConfig.minThreads = 1;
        poolConfig.maxThreads = 16;
        auto threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>(poolConfig);
        
        // 创建CommonServicesFactory
        std::string configPath = "config/database_config.yaml";
        if (std::filesystem::exists(configPath)) {
            std::cout << "✅ 使用配置文件: " << configPath << std::endl;
            commonFactory_ = std::make_shared<oscean::common_utils::infrastructure::CommonServicesFactory>(configPath);
        } else {
            std::cout << "⚠️ 使用默认配置" << std::endl;
            oscean::common_utils::infrastructure::ServiceConfiguration config;
            config.sharedThreadPoolManager = threadPoolManager;
            commonFactory_ = std::make_shared<oscean::common_utils::infrastructure::CommonServicesFactory>(config);
        }
        
        // 创建服务管理器
        serviceManager_ = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        
        std::cout << "✅ Common服务初始化完成" << std::endl;
    }

    bool testCommonServicesDependencies() {
        std::cout << "\n🔍 测试1: Common服务依赖检查" << std::endl;
        
        try {
            auto memoryManager = commonFactory_->getMemoryManager();
            std::cout << "内存管理器: " << (memoryManager ? "✅ 可用" : "❌ 不可用") << std::endl;
            
            auto threadPoolManager = commonFactory_->getThreadPoolManager();
            std::cout << "线程池管理器: " << (threadPoolManager ? "✅ 可用" : "❌ 不可用") << std::endl;
            
            auto simdManager = commonFactory_->getSIMDManager();
            std::cout << "SIMD管理器: " << (simdManager ? "✅ 可用" : "❌ 不可用") << std::endl;
            if (simdManager) {
                std::cout << "  - SIMD实现类型: " << simdManager->getImplementationName() << std::endl;
                auto features = simdManager->getFeatures();
                std::cout << "  - SIMD特性摘要: " << features.toString() << std::endl;
            }
            
            auto logger = commonFactory_->getLogger();
            std::cout << "日志管理器: " << (logger ? "✅ 可用" : "❌ 不可用") << std::endl;
            
            auto perfMonitor = commonFactory_->getPerformanceMonitor();
            std::cout << "性能监控器: " << (perfMonitor ? "✅ 可用" : "❌ 不可用") << std::endl;
            
            return memoryManager && threadPoolManager && logger;
            
        } catch (const std::exception& e) {
            std::cout << "❌ 依赖检查异常: " << e.what() << std::endl;
            return false;
        }
    }

    bool testCRSConfigCreation() {
        std::cout << "\n🔍 测试2: CRS配置创建" << std::endl;
        
        try {
            std::cout << "创建测试配置..." << std::endl;
            auto testConfig = oscean::core_services::crs::CrsServiceConfig::createForTesting();
            std::cout << "✅ 测试配置创建成功" << std::endl;
            
            std::cout << "创建高性能配置..." << std::endl;
            auto perfConfig = oscean::core_services::crs::CrsServiceConfig::createHighPerformance();
            std::cout << "✅ 高性能配置创建成功" << std::endl;
            
            std::cout << "创建低内存配置..." << std::endl;
            auto lowMemConfig = oscean::core_services::crs::CrsServiceConfig::createLowMemory();
            std::cout << "✅ 低内存配置创建成功" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ CRS配置创建异常: " << e.what() << std::endl;
            return false;
        }
    }

    bool testCRSFactoryCreation() {
        std::cout << "\n🔍 测试3: CRS工厂创建" << std::endl;
        
        try {
            auto config = oscean::core_services::crs::CrsServiceConfig::createForTesting();
            
            std::cout << "创建CRS工厂..." << std::endl;
            auto factory = std::make_unique<oscean::core_services::crs::CrsServiceFactory>(commonFactory_, config);
            std::cout << "✅ CRS工厂创建成功" << std::endl;
            
            std::cout << "检查工厂健康状态..." << std::endl;
            bool isHealthy = factory->isHealthy();
            std::cout << "工厂健康状态: " << (isHealthy ? "✅ 健康" : "❌ 不健康") << std::endl;
            
            if (!isHealthy) {
                auto diagnostics = factory->getDiagnosticMessages();
                std::cout << "诊断信息:" << std::endl;
                for (const auto& msg : diagnostics) {
                    std::cout << "  - " << msg << std::endl;
                }
            }
            
            return isHealthy;
            
        } catch (const std::exception& e) {
            std::cout << "❌ CRS工厂创建异常: " << e.what() << std::endl;
            return false;
        }
    }

    bool testDirectCRSServiceCreation() {
        std::cout << "\n🔍 测试4: 直接CRS服务创建（无超时）" << std::endl;
        
        try {
            auto config = oscean::core_services::crs::CrsServiceConfig::createForTesting();
            auto factory = std::make_unique<oscean::core_services::crs::CrsServiceFactory>(commonFactory_, config);
            
            std::cout << "直接调用createCrsService()..." << std::endl;
            auto start_time = std::chrono::steady_clock::now();
            
            auto crsService = factory->createCrsService();
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "✅ CRS服务直接创建成功，耗时: " << duration.count() << "ms" << std::endl;
            std::cout << "服务指针: " << crsService.get() << std::endl;
            
            return crsService != nullptr;
            
        } catch (const std::exception& e) {
            std::cout << "❌ 直接CRS服务创建异常: " << e.what() << std::endl;
            return false;
        }
    }

    bool testTimeoutCRSServiceCreation() {
        std::cout << "\n🔍 测试5: 超时保护CRS服务创建" << std::endl;
        
        try {
            auto config = oscean::core_services::crs::CrsServiceConfig::createForTesting();
            auto factory = std::make_unique<oscean::core_services::crs::CrsServiceFactory>(commonFactory_, config);
            
            std::unique_ptr<oscean::core_services::ICrsService> crsService;
            std::exception_ptr creation_exception = nullptr;
            std::atomic<bool> creation_complete{false};
            
            std::cout << "在独立线程中创建CRS服务..." << std::endl;
            std::thread creation_thread([&]() {
                try {
                    std::cout << "[创建线程] 开始..." << std::endl;
                    crsService = factory->createCrsService();
                    std::cout << "[创建线程] 完成" << std::endl;
                    creation_complete.store(true);
                } catch (...) {
                    std::cout << "[创建线程] 异常" << std::endl;
                    creation_exception = std::current_exception();
                    creation_complete.store(true);
                }
            });
            
            // 等待10秒
            const int timeout_seconds = 10;
            auto start_time = std::chrono::steady_clock::now();
            
            while (!creation_complete.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed > std::chrono::seconds(timeout_seconds)) {
                    std::cout << "⏰ 创建超时(" << timeout_seconds << "秒)" << std::endl;
                    creation_thread.detach();
                    return false;
                }
            }
            
            if (creation_thread.joinable()) {
                creation_thread.join();
            }
            
            if (creation_exception) {
                std::rethrow_exception(creation_exception);
            }
            
            std::cout << "✅ 超时保护CRS服务创建成功" << std::endl;
            return crsService != nullptr;
            
        } catch (const std::exception& e) {
            std::cout << "❌ 超时保护CRS服务创建异常: " << e.what() << std::endl;
            return false;
        }
    }

    bool testServiceManagerCRSService() {
        std::cout << "\n🔍 测试6: 通过ServiceManager获取CRS服务" << std::endl;
        
        try {
            std::cout << "调用serviceManager->getService<ICrsService>()..." << std::endl;
            auto start_time = std::chrono::steady_clock::now();
            
            auto crsService = serviceManager_->getService<oscean::core_services::ICrsService>();
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "ServiceManager调用完成，耗时: " << duration.count() << "ms" << std::endl;
            std::cout << "CRS服务获取结果: " << (crsService ? "✅ 成功" : "❌ 失败") << std::endl;
            
            return crsService != nullptr;
            
        } catch (const std::exception& e) {
            std::cout << "❌ ServiceManager获取CRS服务异常: " << e.what() << std::endl;
            return false;
        }
    }

    void runAllTests() {
        std::cout << "===========================================" << std::endl;
        std::cout << "  OSCEAN CRS服务深度诊断测试" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        bool test1 = testCommonServicesDependencies();
        bool test2 = testCRSConfigCreation();
        bool test3 = testCRSFactoryCreation();
        bool test4 = false, test5 = false, test6 = false;
        
        if (test1 && test2 && test3) {
            test4 = testDirectCRSServiceCreation();
            test5 = testTimeoutCRSServiceCreation();
            test6 = testServiceManagerCRSService();
        }
        
        std::cout << "\n🎯 测试结果摘要:" << std::endl;
        std::cout << "1. Common服务依赖: " << (test1 ? "✅ 通过" : "❌ 失败") << std::endl;
        std::cout << "2. CRS配置创建: " << (test2 ? "✅ 通过" : "❌ 失败") << std::endl;
        std::cout << "3. CRS工厂创建: " << (test3 ? "✅ 通过" : "❌ 失败") << std::endl;
        std::cout << "4. 直接CRS服务创建: " << (test4 ? "✅ 通过" : "❌ 失败") << std::endl;
        std::cout << "5. 超时保护创建: " << (test5 ? "✅ 通过" : "❌ 失败") << std::endl;
        std::cout << "6. ServiceManager创建: " << (test6 ? "✅ 通过" : "❌ 失败") << std::endl;
        
        bool allPassed = test1 && test2 && test3 && test4 && test5 && test6;
        std::cout << "\n🏆 总体结果: " << (allPassed ? "✅ 全部通过" : "❌ 部分失败") << std::endl;
        
        if (!allPassed) {
            std::cout << "\n🔍 故障排除建议:" << std::endl;
            if (!test1) std::cout << "- 检查Common服务配置和依赖" << std::endl;
            if (!test2) std::cout << "- 检查CRS配置类定义" << std::endl;
            if (!test3) std::cout << "- 检查CRS工厂实现和依赖注入" << std::endl;
            if (!test4) std::cout << "- 检查CRS服务实现的构造函数和初始化逻辑" << std::endl;
            if (!test5) std::cout << "- 可能存在死锁或长时间阻塞" << std::endl;
            if (!test6) std::cout << "- 检查ServiceManager的CRS服务工厂注册" << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    // 设置控制台UTF-8编码
    system("chcp 65001 > nul");
    
    CRSDiagnosticTester tester;
    tester.runAllTests();
    
    return 0;
} 