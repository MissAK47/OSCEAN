/**
 * @file raster_statistics.h
 * @brief RasterStatistics class for raster statistical analysis operations
 */

#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include <vector>
#include <map>
#include <boost/optional.hpp>

namespace oscean::core_services::spatial_ops::raster {

/**
 * @brief Handles raster statistical analysis operations
 */
class RasterStatistics {
public:
    explicit RasterStatistics(const SpatialOpsConfig& config);
    ~RasterStatistics() = default;

    // Non-copyable, non-movable
    RasterStatistics(const RasterStatistics&) = delete;
    RasterStatistics& operator=(const RasterStatistics&) = delete;
    RasterStatistics(RasterStatistics&&) = delete;
    RasterStatistics& operator=(RasterStatistics&&) = delete;

    /**
     * @brief Computes pixel-wise statistics across multiple rasters
     * @param inputRasters Vector of input rasters. Must have same dimensions and CRS
     * @param statisticType Type of statistic to compute
     * @return A new GridData object representing the computed statistical raster
     */
    oscean::core_services::GridData computePixelwiseStatistics(
        const std::vector<oscean::core_services::GridData>& inputRasters,
        StatisticalMeasure statisticType) const;

    /**
     * @brief Calculates zonal statistics for a single geometry
     * @param valueRaster The raster containing values to analyze
     * @param zoneGeometry The geometry defining the zone
     * @param options Zonal statistics options
     * @return Vector of statistics results
     */
    std::vector<StatisticsResult> calculateZonalStatistics(
        const oscean::core_services::GridData& valueRaster,
        const oscean::core_services::Geometry& zoneGeometry,
        const ZonalStatisticsOptions& options) const;

    /**
     * @brief Calculates zonal statistics for multiple features
     * @param valueRaster The raster containing values to analyze
     * @param zoneFeatures The feature collection defining zones
     * @param options Zonal statistics options
     * @return Map of feature IDs to statistics results
     */
    std::map<FeatureId, StatisticsResult> calculateZonalStatistics(
        const oscean::core_services::GridData& valueRaster,
        const oscean::core_services::FeatureCollection& zoneFeatures,
        const ZonalStatisticsOptions& options) const;

    /**
     * @brief Calculates zonal statistics using a zone raster
     * @param valueRaster The raster containing values to analyze
     * @param zoneRaster The raster where each cell value represents a zone ID
     * @param options Zonal statistics options
     * @return Map of zone IDs to statistics results
     */
    std::map<int, StatisticsResult> calculateZonalStatistics(
        const oscean::core_services::GridData& valueRaster,
        const oscean::core_services::GridData& zoneRaster,
        const ZonalStatisticsOptions& options) const;

    /**
     * @brief Computes basic statistics for a raster.
     * @param raster Input raster data.
     * @param excludeNoData Whether to exclude NoData values from calculation.
     * @return StatisticsResult containing computed statistics.
     */
    StatisticsResult computeBasicStatistics(
        const oscean::core_services::GridData& raster,
        bool excludeNoData = true) const;

    /**
     * @brief Computes histogram for a raster.
     * @param raster Input raster data.
     * @param numBins Number of histogram bins.
     * @param minValue Optional minimum value for histogram range.
     * @param maxValue Optional maximum value for histogram range.
     * @return Vector of pairs containing bin center values and counts.
     */
    std::vector<std::pair<double, int>> computeHistogram(
        const oscean::core_services::GridData& raster,
        int numBins = 256,
        boost::optional<double> minValue = boost::none,
        boost::optional<double> maxValue = boost::none) const;

    /**
     * @brief Computes percentiles for a raster
     * @param raster Input raster data
     * @param percentiles Vector of percentile values (0-100)
     * @return Vector of percentile values
     */
    std::vector<double> computePercentiles(
        const oscean::core_services::GridData& raster,
        const std::vector<double>& percentiles) const;

private:
    const SpatialOpsConfig& m_config;

    /**
     * @brief Helper method to extract valid pixel values from a raster
     */
    std::vector<double> extractValidPixels(
        const oscean::core_services::GridData& raster,
        boost::optional<double> noDataValue = boost::none) const;

    /**
     * @brief Helper method to compute statistics from a vector of values
     */
    StatisticsResult computeStatisticsFromValues(
        const std::vector<double>& values,
        const std::vector<StatisticalMeasure>& measures) const;

    /**
     * @brief Helper method to validate inputs for zonal statistics
     */
    void validateZonalStatisticsInputs(
        const oscean::core_services::GridData& valueRaster,
        const ZonalStatisticsOptions& options) const;

    /**
     * @brief Helper method to rasterize geometry for zonal analysis
     */
    oscean::core_services::GridData rasterizeGeometryForZones(
        const oscean::core_services::Geometry& geometry,
        const oscean::core_services::GridDefinition& targetGridDef) const;
};

} // namespace oscean::core_services::spatial_ops::raster 