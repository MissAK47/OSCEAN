#pragma once

#include "../common_data_types.h"
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 模型服务不使用boost::asio，只使用boost::future

#include <boost/thread/future.hpp>
#include <vector>
#include <string>
#include <memory> // For shared_ptr
#include <map> // For model parameters

// 定义OSCEAN_FUTURE宏为boost::future
#define OSCEAN_FUTURE(T) boost::future<T>

namespace oscean::core_services {

// Forward declaration for plugin interface
class IComputationModel;

// 🔧 修复重定义错误：使用common_data_types.h中的定义
// 这些类型已在common_data_types.h中定义为boost::any版本
// 因此这里不重复定义，直接使用前向声明或引用

/**
 * @brief Interface for the Modeling Service.
 *
 * Responsible for loading, managing, and executing computation models (plugins).
 */
class IModelingService {
public:
    virtual ~IModelingService() = default;

    /**
     * @brief Executes a named computation model.
     *
     * @param modelName The unique name of the model to execute.
     * @param inputData A structure containing the input data required by the model.
     * @return A future containing the ModelOutput structure with the model results.
     *         The future might contain an exception if the model is not found, 
     *         initialization fails, or execution fails.
     */
    virtual OSCEAN_FUTURE(ModelOutput) executeModelAsync(
        const std::string& modelName,
        const ModelInput& inputData
    ) = 0;

    /**
     * @brief Lists the names of all currently available (loaded) computation models.
     *
     * @return A vector of strings containing model names.
     */
    virtual std::vector<std::string> listAvailableModels() = 0;

    /**
     * @brief Loads computation models from a specified plugin file (e.g., .so, .dll).
     *
     * Assumes the plugin exports a known registration function (e.g., registerModels).
     *
     * @param pluginPath Path to the plugin shared library file.
     * @return True if the plugin was loaded and models were registered successfully, false otherwise.
     *         Note: This might not be async if loading needs to be immediate for subsequent calls.
     *         Consider an async version if plugin loading is slow and can happen in the background.
     */
    virtual bool loadModelsFromPlugin(const std::string& pluginPath) = 0;

    /**
     * @brief (Optional) Unloads a previously loaded plugin and its associated models.
     *
     * Careful implementation needed to handle potential ongoing model executions.
     *
     * @param pluginPath Path or identifier of the plugin to unload.
     * @return True if unloading was successful.
     */
    // virtual bool unloadPlugin(const std::string& pluginPath) = 0;

    /**
     * @brief (Optional) Initializes a specific model instance if needed before execution.
     * 
     * Some models might require explicit initialization with configuration.
     * 
     * @param modelName Name of the model.
     * @param config Configuration parameters for the model.
     * @return Future indicating success or failure of initialization.
     */
    // virtual OSCEAN_FUTURE(bool) initializeModelAsync(const std::string& modelName, const ModelConfig& config) = 0;

};

} // namespace oscean::core_services 