/**
 * @file enhanced_spatial_request_resolver.cpp
 * @brief Implementation of the resolver for complex spatial requests.
 * @author OSCEAN Team
 * @date 2024
 */

#include "workflow_engine/data_workflow/enhanced_spatial_request_resolver.h"
#include "workflow_engine/data_workflow/enhanced_data_workflow_service_impl.h"
#include "workflow_engine/data_workflow/enhanced_workflow_types.h"
#include "common_utils/utilities/logging_utils.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "core_services/crs/i_crs_service.h"
#include <boost/thread/future.hpp>
#include <cmath>

namespace oscean::workflow_engine::data_workflow {

// Helper function to calculate complexity
double calculateComplexity(const SpatialRequest& spatialRequest, double area) {
    double complexity = 0.0;
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, Point>) {
            complexity = 0.0; // Point is simplest
        } else if constexpr (std::is_same_v<T, BoundingBox>) {
            complexity = 0.1; // BBox is simple
        } else if constexpr (std::is_same_v<T, LineString>) {
            complexity = std::min(1.0, 0.1 + static_cast<double>(arg.points.size()) / 100.0);
        } else if constexpr (std::is_same_v<T, Polygon>) {
            complexity = std::min(1.0, 0.2 + static_cast<double>(arg.shell.points.size()) / 200.0 + area / 1e10);
        } else if constexpr (std::is_same_v<T, BearingDistanceRequest>) {
            complexity = 0.05; // Simple calculation
        }
    }, spatialRequest);
    return complexity;
}

// Helper function to recommend access pattern
std::string recommendPattern(const SpatialRequest& spatialRequest, double complexity) {
    if (std::holds_alternative<Point>(spatialRequest)) {
        return "random_access";
    }
    if (complexity < 0.3) {
        return "sequential_scan";
    } else if (complexity < 0.7) {
        return "chunked_reading";
    } else {
        return "streaming_processing";
    }
}

EnhancedSpatialRequestResolver::EnhancedSpatialRequestResolver(EnhancedDataWorkflowServiceImpl* owner)
    : owner_(owner) {
    LOG_MODULE_INFO("EnhancedSpatialRequestResolver", "Initialized.");
}

std::shared_ptr<core_services::ICrsService> EnhancedSpatialRequestResolver::getCrsService() {
    auto crsService = owner_->getCrsService();
    if (!crsService) {
        throw std::runtime_error("CRS Service is not available.");
    }
    return crsService;
}

// Add accessor for SpatialOpsService
std::shared_ptr<oscean::core_services::spatial_ops::ISpatialOpsService> EnhancedSpatialRequestResolver::getSpatialOpsService() {
    auto spatialOpsService = owner_->getSpatialOpsService();
    if (!spatialOpsService) {
        throw std::runtime_error("Spatial Operations Service is not available.");
    }
    return spatialOpsService;
}

boost::future<EnhancedSpatialQueryMetadata> EnhancedSpatialRequestResolver::resolveAsync(
    const EnhancedDataWorkflowRequest& request) {
    
    auto promise = std::make_shared<boost::promise<EnhancedSpatialQueryMetadata>>();
    auto future = promise->get_future();

    auto taskFuture = boost::async(boost::launch::async, [this, promise, request]() {
        try {
            LOG_DEBUG("EnhancedSpatialRequestResolver", "Starting spatial request resolution...");
            
            auto spatialOpsService = getSpatialOpsService();
            auto crsService = getCrsService();

            const auto& spatialRequest = request.spatialRequest;
            const auto& processingOptions = request.processingOptions;
            
            EnhancedSpatialQueryMetadata metadata;

            // Step 1: Get original bounds and CRS
            metadata.originalRequestCRS = std::visit([](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, Point>) return arg.crsId.value_or("EPSG:4326");
                if constexpr (std::is_same_v<T, BoundingBox>) return arg.crsId;
                if constexpr (std::is_same_v<T, Polygon>) return arg.crsId.value_or("EPSG:4326");
                if constexpr (std::is_same_v<T, LineString>) return arg.crsId.value_or("EPSG:4326");
                if constexpr (std::is_same_v<T, BearingDistanceRequest>) return arg.crsId.value_or("EPSG:4326");
                return std::string("EPSG:4326");
            }, spatialRequest);
            
            // Convert SpatialRequest to Geometry for spatial operations
            oscean::core_services::Geometry requestGeometry;
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, Point>) {
                    requestGeometry = oscean::core_services::Geometry(oscean::core_services::Geometry::Type::POINT);
                    requestGeometry.wkt = "POINT(" + std::to_string(arg.x) + " " + std::to_string(arg.y) + ")";
                } else if constexpr (std::is_same_v<T, BoundingBox>) {
                    requestGeometry = oscean::core_services::Geometry(oscean::core_services::Geometry::Type::POLYGON);
                    requestGeometry.wkt = "POLYGON((" + 
                        std::to_string(arg.minX) + " " + std::to_string(arg.minY) + "," +
                        std::to_string(arg.maxX) + " " + std::to_string(arg.minY) + "," +
                        std::to_string(arg.maxX) + " " + std::to_string(arg.maxY) + "," +
                        std::to_string(arg.minX) + " " + std::to_string(arg.maxY) + "," +
                        std::to_string(arg.minX) + " " + std::to_string(arg.minY) + "))";
                } else if constexpr (std::is_same_v<T, LineString>) {
                    requestGeometry = arg.toGeometry();
                } else if constexpr (std::is_same_v<T, Polygon>) {
                    requestGeometry = arg.toGeometry();
                } else if constexpr (std::is_same_v<T, BearingDistanceRequest>) {
                    // For BearingDistanceRequest, create a point geometry
                    requestGeometry = oscean::core_services::Geometry(oscean::core_services::Geometry::Type::POINT);
                    requestGeometry.wkt = "POINT(" + std::to_string(arg.startPoint.x) + " " + std::to_string(arg.startPoint.y) + ")";
                }
            }, spatialRequest);
            
            metadata.originalRequestBounds = spatialOpsService->getBoundingBoxForGeometry(requestGeometry).get();

            // Step 2: Determine target CRS and transform bounds if necessary
            metadata.gridDefinition.targetCRS = processingOptions.targetCRS.value_or(metadata.originalRequestCRS);
            metadata.gridDefinition.isTransformed = (metadata.originalRequestCRS != metadata.gridDefinition.targetCRS);

            if (metadata.gridDefinition.isTransformed) {
                // Create CRSInfo for target CRS
                oscean::core_services::CRSInfo targetCRSInfo;
                targetCRSInfo.id = metadata.gridDefinition.targetCRS;
                targetCRSInfo.authorityName = "EPSG";
                targetCRSInfo.authorityCode = metadata.gridDefinition.targetCRS.substr(5); // Remove "EPSG:" prefix
                
                metadata.gridDefinition.targetBounds = crsService->transformBoundingBoxAsync(
                    metadata.originalRequestBounds, targetCRSInfo).get();
            } else {
                metadata.gridDefinition.targetBounds = metadata.originalRequestBounds;
            }

            // Step 3: Determine Resolution and Grid Size (using placeholder logic for now)
            // This part should be enhanced with logic from SpatialResolutionConfig
            metadata.gridDefinition.xResolution = 0.1;
            metadata.gridDefinition.yResolution = 0.1;
            metadata.gridDefinition.width = static_cast<size_t>(
                (metadata.gridDefinition.targetBounds.maxX - metadata.gridDefinition.targetBounds.minX) / metadata.gridDefinition.xResolution);
            metadata.gridDefinition.height = static_cast<size_t>(
                (metadata.gridDefinition.targetBounds.maxY - metadata.gridDefinition.targetBounds.minY) / metadata.gridDefinition.yResolution);
            
            // Step 4: Calculate complexity and recommend access pattern
            // Calculate area from bounding box
            double area = (metadata.gridDefinition.targetBounds.maxX - metadata.gridDefinition.targetBounds.minX) *
                         (metadata.gridDefinition.targetBounds.maxY - metadata.gridDefinition.targetBounds.minY);
            metadata.spatialComplexity = calculateComplexity(spatialRequest, area);
            metadata.recommendedAccessPattern = recommendPattern(spatialRequest, metadata.spatialComplexity);

            LOG_INFO("EnhancedSpatialRequestResolver", "Spatial resolution completed. Complexity: {:.2f}, Pattern: {}",
                metadata.spatialComplexity, metadata.recommendedAccessPattern);

            promise->set_value(metadata);

        } catch (const std::exception& e) {
            LOG_ERROR("EnhancedSpatialRequestResolver", "Error during spatial resolution: {}", e.what());
            promise->set_exception(std::current_exception());
        }
    });

    // 保持taskFuture存活
    (void)taskFuture;

    return future;
}

} // namespace oscean::workflow_engine::data_workflow 