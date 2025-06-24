#include "workflow_engine/workflow_base.h"
#include <stdexcept>

namespace ocean {
namespace workflow_engine {

WorkflowBase::WorkflowBase(std::string workflowId,
                           WorkflowParameters params,
                           ICoreServiceProxyPtr serviceProxy)
    : id_(std::move(workflowId)),
      params_(std::move(params)),
      service_proxy_(std::move(serviceProxy)),
      status_(WorkflowStatus::NotStarted) {
    if (id_.empty()) {
        throw std::invalid_argument("Workflow ID cannot be empty.");
    }
    if (!service_proxy_) {
        throw std::invalid_argument("CoreServiceProxy cannot be null.");
    }
}

void WorkflowBase::run() {
    if (status_.load() != WorkflowStatus::NotStarted) {
        // Or log a warning: "Workflow already started."
        return;
    }

    try {
        setStatus(WorkflowStatus::Running);
        runInternal();
        // If runInternal completes without throwing, it's considered successful.
        // The final status (Completed/Failed) might be set within runInternal for async workflows.
        // For simple, synchronous workflows, we can set it here.
        if(getStatus() == WorkflowStatus::Running) {
             setStatus(WorkflowStatus::Completed);
        }
    } catch (...) {
        // Catch any exception from the implementation and mark as failed.
        setStatus(WorkflowStatus::Failed);
        // It's good practice to log the exception here.
        // Re-throw if the caller needs to be aware of it.
        throw;
    }
}

void WorkflowBase::cancel() {
    // This is a basic implementation.
    // Subclasses might need to override this to handle cancellation more gracefully
    // (e.g., by interrupting a long-running task).
    if (status_ == WorkflowStatus::Running) {
        setStatus(WorkflowStatus::Cancelled);
    }
}

WorkflowStatus WorkflowBase::getStatus() const {
    return status_.load();
}

std::string WorkflowBase::getId() const {
    return id_;
}

void WorkflowBase::setStatus(WorkflowStatus newStatus) {
    status_.store(newStatus);
    // Here you could also emit signals or log status changes.
}

} // namespace workflow_engine
} // namespace ocean 