/**
 * @file raster_vectorization.h
 * @brief RasterVectorization class for raster-vector conversion operations
 */

#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include <boost/optional.hpp>

namespace oscean::core_services::spatial_ops::raster {

/**
 * @brief Handles raster-vector conversion operations
 */
class RasterVectorization {
public:
    explicit RasterVectorization(const SpatialOpsConfig& config);
    ~RasterVectorization() = default;

    // Non-copyable, non-movable
    RasterVectorization(const RasterVectorization&) = delete;
    RasterVectorization& operator=(const RasterVectorization&) = delete;
    RasterVectorization(RasterVectorization&&) = delete;
    RasterVectorization& operator=(RasterVectorization&&) = delete;

    /**
     * @brief Rasterizes a collection of vector features onto a target grid
     * @param features The input feature collection
     * @param targetGridDef The definition of the target raster
     * @param options Rasterization options
     * @return A new GridData object representing the rasterized features
     */
    oscean::core_services::GridData rasterizeFeatures(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::GridDefinition& targetGridDef,
        const RasterizeOptions& options) const;

    /**
     * @brief Generates contour lines or polygons from a raster
     * @param raster Input raster data
     * @param options Contour generation options
     * @return FeatureCollection containing contour lines or polygons
     */
    oscean::core_services::FeatureCollection generateContours(
        const oscean::core_services::GridData& raster,
        const ContourOptions& options) const;

    /**
     * @brief Converts raster to polygons (vectorization)
     * @param raster Input raster data
     * @param noDataValue Optional NoData value to ignore
     * @return FeatureCollection containing polygons
     */
    oscean::core_services::FeatureCollection vectorizeRaster(
        const oscean::core_services::GridData& raster,
        boost::optional<double> noDataValue = boost::none) const;

    /**
     * @brief Traces raster boundaries to create line features
     * @param raster Input raster data
     * @param threshold Value threshold for boundary tracing
     * @param noDataValue Optional NoData value to ignore
     * @return Vector of WKT strings representing boundaries
     */
    std::vector<std::string> traceBoundaries(
        const oscean::core_services::GridData& raster,
        double threshold,
        boost::optional<double> noDataValue = boost::none) const;

    /**
     * @brief Converts raster to points at cell centers
     * @param raster Input raster data
     * @param noDataValue Optional NoData value to ignore
     * @return FeatureCollection containing point features
     */
    oscean::core_services::FeatureCollection rasterToPoints(
        const oscean::core_services::GridData& raster,
        boost::optional<double> noDataValue = boost::none) const;

    /**
     * @brief Creates isolines (contour lines) at specified values
     * @param raster Input raster data
     * @param levels Vector of values to create contours for
     * @param noDataValue Optional NoData value to ignore
     * @return Vector of WKT strings representing isolines
     */
    std::vector<std::string> createIsolines(
        const oscean::core_services::GridData& raster,
        const std::vector<double>& levels,
        boost::optional<double> noDataValue = boost::none) const;

private:
    const SpatialOpsConfig& m_config;

    /**
     * @brief Validates grid definition parameters
     */
    void validateGridDefinition(
        const oscean::core_services::GridDefinition& gridDef) const;

    /**
     * @brief Rasterizes a single feature
     */
    void rasterizeFeature(
        const oscean::core_services::Feature& feature,
        const oscean::core_services::GridDefinition& gridDef,
        float* data,
        const RasterizeOptions& options) const;

    /**
     * @brief Rasterizes a point from WKT
     */
    void rasterizePointFromWKT(
        const std::string& wkt,
        const oscean::core_services::GridDefinition& gridDef,
        float* data,
        float burnValue) const;

    /**
     * @brief Rasterizes a linestring from WKT
     */
    void rasterizeLineStringFromWKT(
        const std::string& wkt,
        const oscean::core_services::GridDefinition& gridDef,
        float* data,
        float burnValue) const;

    /**
     * @brief Rasterizes a polygon from WKT
     */
    void rasterizePolygonFromWKT(
        const std::string& wkt,
        const oscean::core_services::GridDefinition& gridDef,
        float* data,
        float burnValue) const;
};

} // namespace oscean::core_services::spatial_ops::raster 
