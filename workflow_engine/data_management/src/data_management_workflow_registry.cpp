/**
 * @file data_management_workflow_registry.cpp
 * @brief 数据管理工作流自动注册实现
 */

#include "workflow_engine/workflow_registry.h"
#include "workflow_engine/data_management/data_management_service.h"
#include "common_utils/utilities/logging_utils.h"

namespace {

/**
 * @brief 数据管理工作流适配器
 * 
 * 将DataManagementService适配为IWorkflow接口
 */
class DataManagementWorkflowAdapter : public oscean::workflow_engine::IWorkflow {
public:
    DataManagementWorkflowAdapter(
        std::shared_ptr<oscean::workflow_engine::data_management::DataManagementService> service)
        : dataManagementService_(service) {
    }

    oscean::workflow_engine::WorkflowType getType() const override {
        return oscean::workflow_engine::WorkflowType::DATA_MANAGEMENT;
    }

    std::string getName() const override {
        return "DataManagement";
    }

    std::string getVersion() const override {
        return "1.0.0";
    }

    bool initialize(const std::map<std::string, std::any>& config) override {
        if (dataManagementService_) {
            return dataManagementService_->initializeWorkflow(config);
        }
        return false;
    }

    bool isHealthy() const override {
        return dataManagementService_ && dataManagementService_->isHealthy();
    }

    void shutdown() override {
        if (dataManagementService_) {
            dataManagementService_->shutdownWorkflow();
        }
    }

    /**
     * @brief 获取底层的数据管理服务
     */
    std::shared_ptr<oscean::workflow_engine::data_management::DataManagementService> 
    getDataManagementService() const {
        return dataManagementService_;
    }

private:
    std::shared_ptr<oscean::workflow_engine::data_management::DataManagementService> dataManagementService_;
};

/**
 * @brief 数据管理工作流工厂函数
 */
std::shared_ptr<oscean::workflow_engine::IWorkflow> createDataManagementWorkflow() {
    try {
        OSCEAN_LOG_INFO("DataManagementWorkflowFactory", "创建数据管理工作流实例");
        
        // 使用工厂函数创建数据管理服务
        auto dataManagementService = oscean::workflow_engine::data_management::createDataManagementService();
        
        if (!dataManagementService) {
            OSCEAN_LOG_ERROR("DataManagementWorkflowFactory", "数据管理服务创建失败");
            return nullptr;
        }

        // 创建适配器
        auto adapter = std::make_shared<DataManagementWorkflowAdapter>(dataManagementService);
        
        OSCEAN_LOG_INFO("DataManagementWorkflowFactory", "数据管理工作流实例创建成功");
        return adapter;
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("DataManagementWorkflowFactory", "创建数据管理工作流异常: {}", e.what());
        return nullptr;
    }
}

} // anonymous namespace

// 🚀 自动注册数据管理工作流
REGISTER_WORKFLOW(
    DATA_MANAGEMENT,
    "DataManagement",
    "1.0.0",
    "数据管理工作流 - 提供文件扫描、元数据提取、分类存储等数据管理功能",
    createDataManagementWorkflow
); 