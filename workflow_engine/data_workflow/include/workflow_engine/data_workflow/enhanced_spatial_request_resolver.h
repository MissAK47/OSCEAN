#pragma once

/**
 * @file enhanced_spatial_request_resolver.h
 * @brief Defines the resolver for complex spatial requests in the enhanced data workflow.
 * @author OSCEAN Team
 * @date 2024
 */

#include "workflow_engine/data_workflow/enhanced_workflow_types.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include <boost/thread/future.hpp>
#include <memory>

namespace oscean::workflow_engine::data_workflow {

class EnhancedDataWorkflowServiceImpl; // Forward declaration

/**
 * @class EnhancedSpatialRequestResolver
 * @brief Handles the complex logic of interpreting spatial requests.
 *
 * This class validates CRS, handles transformations, determines optimal
 * resolution, and calculates the final grid definition for data extraction.
 */
class EnhancedSpatialRequestResolver {
public:
    /**
     * @brief Constructor.
     * @param owner Pointer to the parent service to access other core services.
     */
    explicit EnhancedSpatialRequestResolver(EnhancedDataWorkflowServiceImpl* owner);

    /**
     * @brief Asynchronously resolves a spatial request into detailed grid and metadata.
     * @param request The full enhanced data workflow request.
     * @return A future containing the resolved spatial query metadata.
     */
    boost::future<EnhancedSpatialQueryMetadata> resolveAsync(
        const EnhancedDataWorkflowRequest& request);

private:
    /// Pointer to the parent service implementation to access helpers like getCrsService().
    EnhancedDataWorkflowServiceImpl* owner_;

    /**
     * @brief Gets the CRS service from the service manager.
     * @return A shared pointer to the CRS service.
     * @throws std::runtime_error if the service is not available.
     */
    std::shared_ptr<core_services::ICrsService> getCrsService();

    /**
     * @brief Gets the Spatial Operations service from the service manager.
     * @return A shared pointer to the Spatial Operations service.
     * @throws std::runtime_error if the service is not available.
     */
    std::shared_ptr<oscean::core_services::spatial_ops::ISpatialOpsService> getSpatialOpsService();
};

} // namespace oscean::workflow_engine::data_workflow 