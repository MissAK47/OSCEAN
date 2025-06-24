#pragma once

#include "i_workflow.h"
#include <memory>

// Forward declare to break header cycle
namespace oscean::workflow_engine::service_management {
    class IServiceManager;
}

namespace oscean::workflow_engine {

    /**
     * @brief 定义了所有工作流库必须实现的工厂函数的签名。
     * @param serviceManager 一个指向共享的、唯一的服务管理器的指针。
     * @return std::shared_ptr<IWorkflow> 一个指向新创建的工作流实例的指针。
     */
    using WorkflowFactoryFunc = std::shared_ptr<IWorkflow>(
        std::shared_ptr<service_management::IServiceManager> serviceManager
    );

    /**
     * @brief 定义所有工作流动态库必须导出的C风格工厂函数的标准名称。
     */
    constexpr const char* WORKFLOW_FACTORY_FUNCTION_NAME = "create_workflow";

    // 在 MSVC 编译器上定义 dllexport 宏
    #if defined(_MSC_VER)
        #define WORKFLOW_API __declspec(dllexport)
    #else
        #define WORKFLOW_API
    #endif

} // namespace oscean::workflow_engine 