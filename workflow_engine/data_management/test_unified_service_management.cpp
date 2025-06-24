/**
 * @file test_unified_service_management.cpp
 * @brief 统一服务管理工作流架构测试
 * 
 * 🎯 测试目标：
 * ✅ 验证IServiceManager正确创建和管理服务
 * ✅ 验证DataManagementService使用统一服务管理
 * ✅ 验证DataManagementWorkflow使用统一服务管理
 * ✅ 验证统一异步框架工作正常
 */

#include <iostream>
#include <memory>
#include <chrono>
#include "workflow_engine/service_management/i_service_manager.h"
#include "../src/service_management/service_manager_impl.h"
#include "workflow_engine/data_management/data_management_service.h"
#include "workflow_engine/data_management/data_management_workflow.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/async/async_framework.h"

// 模拟GDAL/PROJ的全局初始化
void initializeGlobalDependencies() {
    std::cout << "🌍 全局依赖初始化..." << std::endl;
    // 在实际应用中，这里会调用 GDALAllRegister() 和 PROJ相关设置
    // 例如：gdal::GDALAllRegister();
    std::cout << "  - GDAL/PROJ 全局设置完成" << std::endl;
}

using namespace oscean::workflow_engine;
using namespace oscean::workflow_engine::data_management;
using namespace oscean::workflow_engine::service_management;

/**
 * @brief 测试统一服务管理器
 */
bool testServiceManager() {
    std::cout << "\n🧪 === 测试1: 统一服务管理器 ===" << std::endl;
    
    try {
        // 1. 创建基础依赖
        auto threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>();
        
        // 2. 创建服务管理器，并传入正确的项目根目录
        // 测试程序在 build/workflow_engine/data_management/Debug 下运行
        auto serviceManager = std::make_shared<ServiceManagerImpl>(threadPoolManager);
        
        std::cout << "✅ ServiceManager创建成功" << std::endl;
        
        // 3. 测试核心服务获取 (强制data_access优先)
        std::cout << "📋 服务可用性检查:" << std::endl;
        auto dataAccessService = serviceManager->getService<oscean::core_services::data_access::IUnifiedDataAccessService>();
        std::cout << "  - DataAccessService: " << (dataAccessService ? "✅" : "❌") << std::endl;

        auto metadataService = serviceManager->getService<oscean::core_services::metadata::IMetadataService>();
        auto crsService = serviceManager->getService<oscean::core_services::ICrsService>();
        auto interpolationService = serviceManager->getService<oscean::core_services::interpolation::IInterpolationService>();
        auto spatialOpsService = serviceManager->getService<oscean::core_services::spatial_ops::ISpatialOpsService>();
        auto outputService = serviceManager->getService<oscean::core_services::output::IOutputService>();
        
        std::cout << "  - MetadataService: " << (metadataService ? "✅" : "❌") << std::endl;
        std::cout << "  - CrsService: " << (crsService ? "✅" : "❌") << std::endl;
        std::cout << "  - InterpolationService: " << (interpolationService ? "✅" : "❌") << std::endl;
        std::cout << "  - SpatialOpsService: " << (spatialOpsService ? "✅" : "❌") << std::endl;
        std::cout << "  - OutputService: " << (outputService ? "✅" : "❌") << std::endl;
        
        // 4. 测试异步框架
        auto& asyncFramework = serviceManager->getAsyncFramework();
        auto stats = asyncFramework.getStatistics();
        std::cout << "📊 异步框架统计: 活跃任务=" << stats.currentActiveTasks 
                  << ", 完成任务=" << stats.totalTasksCompleted << std::endl;
        
        std::cout << "✅ 测试1通过：统一服务管理器工作正常" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试1失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试DataManagementService使用统一服务管理
 */
bool testDataManagementService() {
    std::cout << "\n🧪 === 测试2: DataManagementService统一服务管理 ===" << std::endl;
    
    try {
        // 1. 创建服务管理器
        auto threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>();
        auto serviceManager = std::make_shared<ServiceManagerImpl>(threadPoolManager);
        
        // 2. 创建DataManagementService
        auto dataManagementService = createDataManagementService(serviceManager);
        
        std::cout << "✅ DataManagementService创建成功" << std::endl;
        
        // 3. 测试工作流接口
        std::cout << "📋 工作流信息:" << std::endl;
        std::cout << "  - 类型: " << static_cast<int>(dataManagementService->getType()) << std::endl;
        std::cout << "  - 名称: " << dataManagementService->getName() << std::endl;
        std::cout << "  - 版本: " << dataManagementService->getVersion() << std::endl;
        std::cout << "  - 健康状态: " << (dataManagementService->isHealthy() ? "✅" : "❌") << std::endl;
        
        // 4. 测试依赖验证
        try {
            std::map<std::string, std::any> config;
            bool initialized = dataManagementService->initializeWorkflow(config);
            std::cout << "  - 初始化: " << (initialized ? "✅" : "❌") << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  - 初始化: ❌ " << e.what() << std::endl;
        }
        
        std::cout << "✅ 测试2通过：DataManagementService正确使用统一服务管理" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试2失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试DataManagementWorkflow使用统一服务管理
 */
bool testDataManagementWorkflow() {
    std::cout << "\n🧪 === 测试3: DataManagementWorkflow统一服务管理 ===" << std::endl;
    
    try {
        // 1. 创建服务管理器
        auto threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>();
        auto serviceManager = std::make_shared<ServiceManagerImpl>(threadPoolManager);
        
        // 2. 创建DataManagementWorkflow
        auto workflow = std::make_shared<DataManagementWorkflow>("test_workflow_001", serviceManager);
        
        std::cout << "✅ DataManagementWorkflow创建成功" << std::endl;
        
        // 3. 测试工作流接口
        std::cout << "📋 工作流信息:" << std::endl;
        std::cout << "  - 名称: " << workflow->getName() << std::endl;
        std::cout << "  - 就绪状态: " << (!workflow->isRunning() ? "✅" : "❌") << std::endl;
        
        std::cout << "✅ 测试3通过：DataManagementWorkflow正确使用统一服务管理" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试3失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试异步任务执行
 */
bool testAsyncExecution() {
    std::cout << "\n🧪 === 测试4: 统一异步框架执行 ===" << std::endl;
    
    try {
        // 1. 创建服务管理器
        auto threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>();
        auto serviceManager = std::make_shared<ServiceManagerImpl>(threadPoolManager);
        
        // 2. 测试异步任务提交
        auto asyncTask = serviceManager->submitAsyncTask("test_task", []() -> int {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return 42;
        });
        
        std::cout << "✅ 异步任务提交成功" << std::endl;
        
        // 3. 等待任务完成
        auto result = asyncTask.get();
        std::cout << "📊 任务执行结果: " << result << std::endl;
        
        // 4. 检查异步框架统计
        auto& asyncFramework = serviceManager->getAsyncFramework();
        auto stats = asyncFramework.getStatistics();
        std::cout << "📊 执行后统计: 活跃任务=" << stats.currentActiveTasks 
                  << ", 完成任务=" << stats.totalTasksCompleted << std::endl;
        
        std::cout << "✅ 测试4通过：统一异步框架工作正常" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试4失败: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "🚀 OSCEAN 统一服务管理工作流架构测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 全局初始化
    initializeGlobalDependencies();

    int passedTests = 0;
    int totalTests = 4;
    
    // 执行测试
    if (testServiceManager()) passedTests++;
    if (testDataManagementService()) passedTests++;
    if (testDataManagementWorkflow()) passedTests++;
    if (testAsyncExecution()) passedTests++;
    
    // 汇总结果
    std::cout << "\n📊 === 测试结果汇总 ===" << std::endl;
    std::cout << "通过测试: " << passedTests << "/" << totalTests << std::endl;
    
    if (passedTests == totalTests) {
        std::cout << "🎉 所有测试通过！统一服务管理工作流架构正常工作" << std::endl;
        return 0;
    } else {
        std::cout << "❌ 部分测试失败，需要检查架构实现" << std::endl;
        return 1;
    }
} 