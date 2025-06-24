/**
 * @file test_simple_workflow.cpp
 * @brief 数据工作流CRS服务问题测试
 */

#include <iostream>
#include <memory>

// 服务管理器相关
#include "workflow_engine/service_management/service_manager_impl.h"
#include "workflow_engine/data_workflow/data_workflow_service_impl.h"
#include "workflow_engine/data_workflow/data_workflow_types.h"

// 通用工具
#include "common_utils/utilities/logging_utils.h"

using namespace oscean::workflow_engine;

int main() {
    try {
        std::cout << "=== 数据工作流CRS服务测试 ===" << std::endl;
        
        // 步骤1：创建ServiceManager
        std::cout << "步骤1: 创建ServiceManager..." << std::endl;
        auto serviceManager = std::make_shared<service_management::ServiceManagerImpl>();
        std::cout << "ServiceManager创建成功" << std::endl;
        
        // 步骤2：创建数据工作流服务
        std::cout << "步骤2: 创建数据工作流服务..." << std::endl;
        auto workflow = std::make_unique<data_workflow::DataWorkflowServiceImpl>(serviceManager);
        std::cout << "数据工作流服务创建成功" << std::endl;
        
        // 步骤3：测试isReady() - 这会触发CRS服务创建
        std::cout << "步骤3: 调用workflow->isReady()..." << std::endl;
        std::cout << "这会触发CRS服务创建，观察是否卡住..." << std::endl;
        
        bool ready = workflow->isReady();
        
        std::cout << "步骤3完成: workflow->isReady() = " << (ready ? "true" : "false") << std::endl;
        
        std::cout << "=== 测试完成，CRS服务问题验证结束 ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return 1;
    }
} 