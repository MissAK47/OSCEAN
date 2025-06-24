/**
 * @file simple_test.cpp
 * @brief 简化的数据管理工作流测试
 */

#include <iostream>
#include <memory>
#include <string>

// 工作流引擎核心
#include "workflow_engine/data_management/data_management_workflow.h"
#include "workflow_engine/service_management/service_manager_impl.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"

// 通用工具
#include "common_utils/utilities/logging_utils.h"

using namespace oscean::workflow_engine::data_management;

/**
 * @brief 简单测试函数
 */
int main() {
    std::cout << "🚀 简化数据管理工作流测试开始" << std::endl;
    
    try {
        // 创建线程池管理器
        auto threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>();
        
        // 创建服务管理器
        auto serviceManager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        
        // 创建工作流
        auto workflow = std::make_unique<DataManagementWorkflow>("simple-test", serviceManager);
        
        // 测试基本功能
        std::cout << "工作流名称: " << workflow->getName() << std::endl;
        std::cout << "工作流ID: " << workflow->getWorkflowId() << std::endl;
        std::cout << "是否运行中: " << (workflow->isRunning() ? "是" : "否") << std::endl;
        
        // 测试同步执行
        std::cout << "执行工作流..." << std::endl;
        workflow->execute();
        std::cout << "工作流执行完成" << std::endl;
        
        std::cout << "✅ 简化测试通过" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
} 