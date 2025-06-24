#pragma once

#include <memory>

// Forward declare the service manager interface to avoid circular dependencies in headers
namespace oscean::workflow_engine::service_management {
    class IServiceManager;
}

namespace oscean::workflow_engine {

    /**
     * @brief 所有动态加载的工作流模块都需要实现的通用接口。
     */
    class IWorkflow {
    public:
        virtual ~IWorkflow() = default;
        
        /**
         * @brief 获取工作流的唯一名称。
         * @return const char* 工作流名称。
         */
        virtual const char* getName() const = 0;

        /**
         * @brief 执行工作流的核心方法。
         * @details 具体的参数和返回值将根据工作流的实际需求而变化，
         *          这里只是一个示例。在实际应用中，可能会使用一个
         *          包含所有参数的请求对象。
         */
        virtual void execute(/* TBD: Workflow-specific request object or parameters */) = 0;
    };

} // namespace oscean::workflow_engine 