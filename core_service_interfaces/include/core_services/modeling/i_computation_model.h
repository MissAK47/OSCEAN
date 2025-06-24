#pragma once

#include "../common_data_types.h"
#include <string>
#include <memory>

namespace oscean::core_services {

// Forward declare config type if needed (e.g., from common_utils)
// namespace common_utils { class Config; }

/**
 * @brief Abstract interface for a computation model plugin.
 *
 * All dynamically loaded computation models must inherit from this class
 * and implement its pure virtual methods.
 */
class IComputationModel {
public:
    virtual ~IComputationModel() = default;

    /**
     * @brief Initializes the model instance.
     *
     * Called potentially once after loading or before the first execution.
     * Allows the model to load resources, set up internal state based on configuration.
     *
     * @param config A configuration object (e.g., a boost::property_tree::ptree view
     *               or a custom Config struct) containing model-specific settings.
     * @return True if initialization was successful, false otherwise.
     */
    virtual bool initialize(const std::any& config) = 0; // Use std::any for flexibility, or define a Config struct

    /**
     * @brief Executes the computation model.
     *
     * This is the core method where the model performs its calculation.
     * It should be designed to be thread-safe if multiple instances might be executed concurrently,
     * or the Modeling Service must ensure sequential execution per instance if needed.
     *
     * @param input Input data for the model, structured as defined by ModelInput.
     *              The model is responsible for validating and casting std::any values.
     * @param output Output data structure to be populated by the model.
     * @return True if execution was successful, false otherwise.
     */
    virtual bool execute(const ModelInput& input, ModelOutput& output) = 0;

    /**
     * @brief Gets the unique name of the computation model.
     *
     * This name is used by the Modeling Service to identify and execute the model.
     *
     * @return const std::string& The name of the model.
     */
    virtual const std::string& getName() const = 0;

    /**
     * @brief (Optional) Provides information about the expected inputs.
     *
     * @return A description of inputs (e.g., map<string, ExpectedTypeInfo>).
     */
    // virtual ModelInputSchema getInputSchema() const = 0;

    /**
     * @brief (Optional) Provides information about the produced outputs.
     *
     * @return A description of outputs.
     */
    // virtual ModelOutputSchema getOutputSchema() const = 0;

    // Add other methods if needed, e.g., for progress reporting, cancellation support.

}; // class IComputationModel

/**
 * @brief Defines the expected signature for the model registration function.
 *
 * Each computation model plugin (.so/.dll) must export a C-style function 
 * with this signature, typically named "registerModels". The Modeling Service 
 * will call this function after loading the plugin library.
 *
 * @param registry A pointer to an object (e.g., ModelRegistry) that the plugin 
 *                 can use to register its model factory or instances.
 */
// Define ModelRegistry interface or use a std::function based approach if preferred.
class IModelRegistry; // Forward declaration
using RegisterModelsFunc = void (*)(IModelRegistry* registry);

} // namespace oscean::core_services 