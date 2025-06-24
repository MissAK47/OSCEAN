#pragma once

#include "core_services/output/i_output_service.h"
#include <memory>
#include <set>
#include <string>
#include <algorithm>
#include <boost/filesystem.hpp>

// Forward declarations for internal components.
// This avoids pulling in their full headers, reducing compile times and dependencies.
// This is possible because we only hold std::shared_ptr to these types in the header.
namespace oscean {
namespace output {
class OutputProfileManager;
class DataExportEngine;
class VisualizationEngine;
} // namespace output
} // namespace oscean


namespace oscean {
namespace output {

/**
 * @class OutputServiceImpl
 * @brief The concrete implementation of the IOutputService interface.
 *
 * This class acts as a Facade, receiving all output requests and delegating the
 * complex work to a set of specialized internal engines and managers. It is
 * responsible for orchestrating the overall output generation workflow.
 */
class OutputServiceImpl : public core_services::output::IOutputService {
public:
    /**
     * @brief Constructs the OutputServiceImpl using Dependency Injection.
     * @param profileManager The manager for handling output profiles.
     * @param exportEngine The engine for handling data file exports (e.g., CSV, NetCDF).
     * @param visualizationEngine The engine for rendering visual outputs (e.g., PNG, JPEG).
     *
     * This constructor receives all its collaborators as shared pointers. This
     * decouples the service from the concrete creation of its dependencies,
     * making the class more modular and easier to test.
     */
    OutputServiceImpl(
        std::shared_ptr<OutputProfileManager> profileManager,
        std::shared_ptr<DataExportEngine> exportEngine,
        std::shared_ptr<VisualizationEngine> visualizationEngine
    );

    ~OutputServiceImpl() override = default;

    // --- IOutputService interface implementation ---

    boost::future<core_services::output::OutputResult> processFromProfile(
        const core_services::output::ProfiledRequest& request) override;

    boost::future<core_services::output::OutputResult> processRequest(
        const core_services::output::OutputRequest& request) override;

    boost::future<void> writeGridAsync(
        std::shared_ptr<const core_services::GridData> gridDataPtr,
        const std::string& filePath,
        const core_services::output::WriteOptions& options) override;

private:
    std::shared_ptr<OutputProfileManager> m_profileManager;
    std::shared_ptr<DataExportEngine> m_exportEngine;
    std::shared_ptr<VisualizationEngine> m_visualizationEngine;
    
    /**
     * @brief Determines if a given format requires visualization processing.
     * @param format The format string (e.g., "png", "csv").
     * @return True if the format is a visualization format.
     */
    bool isVisualizationFormat(const std::string& format) const;
};

} // namespace output
} // namespace oscean 