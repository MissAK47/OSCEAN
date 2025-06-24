#pragma once

#include "workflow_engine/workflow_parameters.h"
#include "workflow_engine/workflow_types.h"
// 🔧 修复：暂时注释掉代理头文件，在新架构中不需要
// #include "workflow_engine/proxies/core_service_proxy.h"

#include <string>
#include <memory>
#include <atomic>

namespace ocean {
namespace workflow_engine {

enum class WorkflowStatus {
    NotStarted,
    Running,
    Completed,
    Failed,
    Cancelled
};

/**
 * @class IWorkflow
 * @brief An interface representing the basic operations for any workflow.
 */
class IWorkflow {
public:
    virtual ~IWorkflow() = default;

    virtual void run() = 0;
    virtual void cancel() = 0;
    virtual WorkflowStatus getStatus() const = 0;
    virtual std::string getId() const = 0;
};

/**
 * @class WorkflowBase
 * @brief An abstract base class for all concrete workflow implementations.
 *
 * It provides a common structure including an ID, status management, and access
 * to core services. Concrete workflows must implement the `runInternal` method
 * which contains the main workflow logic.
 */
class WorkflowBase : public IWorkflow, public std::enable_shared_from_this<WorkflowBase> {
public:
    /**
     * @brief Constructs a WorkflowBase.
     * @param workflowId A unique identifier for this workflow instance.
     * @param params The initial parameters for the workflow.
     * @param serviceProxy A proxy to access core services (optional, for backward compatibility).
     */
    WorkflowBase(std::string workflowId,
                 WorkflowParameters params,
                 void* serviceProxy = nullptr);  // 🔧 修复：使用void*以支持可选代理

    virtual ~WorkflowBase() = default;

    // --- IWorkflow Implementation ---
    void run() final;
    void cancel() final;
    WorkflowStatus getStatus() const final;
    std::string getId() const final;

protected:
    /**
     * @brief The main logic of the workflow must be implemented here by subclasses.
     */
    virtual void runInternal() = 0;

    // --- Member Variables Accessible by Subclasses ---
    std::string id_;
    WorkflowParameters params_;
    void* service_proxy_;  // 🔧 修复：使用void*以支持可选代理
    std::atomic<WorkflowStatus> status_;

protected:
    void setStatus(WorkflowStatus newStatus);

private:
    // No longer private
};

// Convenience type alias for a shared pointer to a workflow.
using IWorkflowPtr = std::shared_ptr<IWorkflow>;

} // namespace workflow_engine
} // namespace ocean 