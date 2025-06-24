#pragma once

#include "core_services/common_data_types.h"
#include "workflow_engine/service_management/i_service_manager.h"
#include "workflow_engine/data_workflow/data_workflow_types.h"
#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <string>
#include <vector>
#include <optional>

namespace oscean::workflow_engine::data_workflow {

/**
 * @brief 数据处理工作流服务接口
 */
class IDataWorkflowService {
public:
    virtual ~IDataWorkflowService() = default;

    /**
     * @brief 异步执行工作流
     * @param request 工作流请求
     * @return 工作流结果的future
     */
    virtual boost::future<WorkflowResult> executeWorkflowAsync(const WorkflowRequest& request) = 0;

    /**
     * @brief 获取工作流名称
     * @return 工作流名称
     */
    virtual std::string getWorkflowName() const = 0;

    /**
     * @brief 检查服务是否就绪
     * @return 如果服务就绪返回true
     */
    virtual bool isReady() const = 0;
};

} // namespace oscean::workflow_engine::data_workflow 