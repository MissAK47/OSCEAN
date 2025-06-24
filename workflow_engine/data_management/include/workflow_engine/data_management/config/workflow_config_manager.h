#pragma once

#include <string>
#include <memory>
#include <any>
#include <map>

namespace oscean {
namespace workflow_engine {

/**
 * @class WorkflowConfigManager
 * @brief Manages configuration loading and parsing for data management workflows.
 * 
 * This class is responsible for loading workflow configurations from JSON/YAML files
 * and providing configuration data to the DataManagementWorkflow and its stage handlers.
 */
class WorkflowConfigManager {
public:
    WorkflowConfigManager() = default;
    virtual ~WorkflowConfigManager() = default;

    /**
     * @brief Loads configuration from a file.
     * @param configPath Path to the configuration file.
     * @return True if configuration was loaded successfully, false otherwise.
     */
    bool loadConfig(const std::string& configPath);

    /**
     * @brief Gets the workflow configuration.
     * @param workflowName Name of the workflow to get configuration for.
     * @return Configuration data as std::any.
     */
    std::any getWorkflowConfig(const std::string& workflowName);

    /**
     * @brief Gets configuration for a specific stage handler.
     * @param handlerName Name of the handler.
     * @param configIdentifier Optional identifier for specific configuration.
     * @return Configuration data as std::any.
     */
    std::any getStageHandlerConfig(const std::string& handlerName, const std::string& configIdentifier = "");

private:
    bool m_configLoaded = false;
    std::string m_configPath;
    std::map<std::string, std::string> m_configData;  ///< 存储配置键值对
};

} // namespace workflow_engine
} // namespace oscean
