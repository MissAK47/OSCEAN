/**
 * @file raster_clipping.h
 * @brief RasterClipping class for raster clipping and masking operations
 */

#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include <optional>

namespace oscean::core_services::spatial_ops::raster {

/**
 * @brief Handles raster clipping and masking operations
 */
class RasterClipping {
public:
    explicit RasterClipping(const SpatialOpsConfig& config);
    ~RasterClipping() = default;

    // Non-copyable, non-movable
    RasterClipping(const RasterClipping&) = delete;
    RasterClipping& operator=(const RasterClipping&) = delete;
    RasterClipping(RasterClipping&&) = delete;
    RasterClipping& operator=(RasterClipping&&) = delete;

    /**
     * @brief Clips a raster by a given bounding box
     * @param inputRaster The input grid data
     * @param bbox The bounding box to clip against
     * @param noDataValue Optional NoData value for areas outside the clip extent
     * @return A new GridData object representing the clipped raster
     */
    oscean::core_services::GridData clipByBoundingBox(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::BoundingBox& bbox,
        std::optional<double> noDataValue = std::nullopt) const;

    /**
     * @brief Clips a raster by a given geometry
     * @param inputRaster The input grid data
     * @param clipGeom The geometry to clip against
     * @param options Masking and clipping options
     * @return A new GridData object representing the clipped raster
     */
    oscean::core_services::GridData clipByGeometry(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::Geometry& clipGeom,
        const MaskOptions& options) const;

    /**
     * @brief Applies a mask raster to an input raster
     * @param inputRaster The raster to be masked
     * @param maskRaster The raster to use as a mask
     * @param options Masking options
     * @return A new GridData object representing the masked raster
     */
    oscean::core_services::GridData applyRasterMask(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::GridData& maskRaster,
        const MaskOptions& options) const;

    /**
     * @brief Crops a raster to a specified bounding box (alias for clipByBoundingBox)
     * @param inputRaster The input raster to crop
     * @param cropBounds The bounding box to crop to
     * @return A new GridData object representing the cropped raster
     */
    oscean::core_services::GridData cropRaster(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::BoundingBox& cropBounds) const;

    /**
     * @brief Extracts a sub-region (window) from a raster
     * @param raster Input raster data
     * @param startCol Starting column index (0-based)
     * @param startRow Starting row index (0-based)
     * @param numCols Number of columns in the sub-region
     * @param numRows Number of rows in the sub-region
     * @return GridData object for the sub-region
     */
    oscean::core_services::GridData extractSubRegion(
        const oscean::core_services::GridData& raster,
        std::size_t startCol, std::size_t startRow,
        std::size_t numCols, std::size_t numRows) const;

private:
    const SpatialOpsConfig& m_config;

    /**
     * @brief Helper method to validate clipping parameters
     */
    void validateClippingInputs(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::BoundingBox& bbox) const;

    /**
     * @brief Helper method to calculate pixel coordinates from geographic coordinates
     */
    std::tuple<int, int, int, int> calculatePixelBounds(
        const oscean::core_services::GridDefinition& gridDef,
        const oscean::core_services::BoundingBox& bbox) const;
};

} // namespace oscean::core_services::spatial_ops::raster 