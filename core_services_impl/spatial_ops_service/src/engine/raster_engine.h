#pragma once

#include "core_services/common_data_types.h" // For GridData, BoundingBox, Point, FeatureCollection, CRSInfo, GridIndex
#include "core_services/spatial_ops/spatial_types.h" // For MaskOptions, ResampleOptions, RasterizeOptions, ZonalStatisticsOptions, ContourOptions etc.
#include "core_services/spatial_ops/spatial_config.h" // For SpatialOpsConfig (maybe needed for GDAL paths etc.)

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <optional>
#include <variant>
#include <gdal.h> // Include GDAL header for GDALDataType definition

// Forward declare GDAL types to avoid including heavy GDAL headers here.
class GDALDataset;

// Forward declarations for modular components
namespace oscean::core_services::spatial_ops::raster {
    class RasterClipping;
    class RasterAlgebra;
    class RasterStatistics;
    class RasterVectorization;
    class GridIndex;
}

// Namespace for the engine layer
namespace oscean::core_services::spatial_ops::engine {

class RasterEngine { // Renamed from RasterProcessor
public:
    /**
     * @brief Constructor for RasterEngine.
     * @param config Service configuration, which might contain GDAL specific settings.
     */
    explicit RasterEngine(const oscean::core_services::spatial_ops::SpatialOpsConfig& config);
    ~RasterEngine();

    // Disable copy and move semantics for now, can be enabled if needed
    RasterEngine(const RasterEngine&) = delete;
    RasterEngine& operator=(const RasterEngine&) = delete;
    RasterEngine(RasterEngine&&) = delete;
    RasterEngine& operator=(RasterEngine&&) = delete;

    /**
     * @brief Clips a raster by a given bounding box.
     * @param inputRaster The input grid data.
     * @param bbox The bounding box to clip against.
     * @param noDataValue Optional NoData value to set for areas outside the clip extent in the output.
     * @return A new GridData object representing the clipped raster.
     * @throw SpatialOpsException if the operation fails.
     */
    oscean::core_services::GridData clipRasterByBoundingBox(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::BoundingBox& bbox,
        std::optional<double> noDataValue = std::nullopt) const;

    /**
     * @brief Clips a raster by a given geometry.
     * @param inputRaster The input grid data.
     * @param clipGeom The geometry to clip against.
     * @param options Masking and clipping options.
     * @return A new GridData object representing the clipped raster.
     * @throw SpatialOpsException if the operation fails or inputs are invalid.
     */
    oscean::core_services::GridData clipRasterByGeometry(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::Geometry& clipGeom,
        const oscean::core_services::spatial_ops::MaskOptions& options) const;

    /**
     * @brief Rasterizes a collection of vector features onto a target grid.
     * @param features The input feature collection.
     * @param targetGridDef The definition of the target raster (resolution, extent, CRS).
     * @param options Rasterization options, including burn value, attribute field, etc.
     * @return A new GridData object representing the rasterized features.
     * @throw SpatialOpsException if the operation fails or inputs are invalid.
     */
    oscean::core_services::GridData rasterizeFeatures(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::GridDefinition& targetGridDef,
        const oscean::core_services::spatial_ops::RasterizeOptions& options) const;

    /**
     * @brief Applies a mask raster to an input raster.
     * @param inputRaster The raster to be masked.
     * @param maskRaster The raster to use as a mask.
     * @param options Masking options, including mask value and invert settings.
     * @return A new GridData object representing the masked raster.
     * @throw SpatialOpsException if the operation fails or inputs are invalid.
     */
    oscean::core_services::GridData applyRasterMask(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::GridData& maskRaster,
        const oscean::core_services::spatial_ops::MaskOptions& options) const;

    /**
     * @brief Performs raster algebra based on an expression and named input rasters.
     * @param expression The mathematical expression to evaluate.
     * @param namedRasters A map of names to GridData objects, used as variables in the expression.
     * @param targetGridDef Optional target grid definition for the output raster. If not provided, derived from inputs.
     * @param noDataValue Optional NoData value for the output raster.
     * @return A new GridData object representing the result of the raster algebra.
     * @throw SpatialOpsException if the operation fails, expression is invalid, or inputs are invalid.
     */
    oscean::core_services::GridData performRasterAlgebra(
        const std::string& expression,
        const std::map<std::string, oscean::core_services::GridData>& namedRasters,
        std::optional<oscean::core_services::GridDefinition> targetGridDef = std::nullopt,
        std::optional<double> noDataValue = std::nullopt) const;

    /**
     * @brief Crops a raster to a specified bounding box.
     * @param inputRaster The input raster to crop.
     * @param cropBounds The bounding box to crop to.
     * @return A new GridData object representing the cropped raster.
     * @throw SpatialOpsException if the operation fails or inputs are invalid.
     */
    oscean::core_services::GridData cropRaster(
        const oscean::core_services::GridData& inputRaster,
        const oscean::core_services::BoundingBox& cropBounds) const;

    /**
     * @brief Extracts a sub-region (window) from a raster.
     * @param raster Input raster data.
     * @param startCol Starting column index (0-based).
     * @param startRow Starting row index (0-based).
     * @param numCols Number of columns in the sub-region.
     * @param numRows Number of rows in the sub-region.
     * @return GridData object for the sub-region.
     * @throw SpatialOpsException if indices are out of bounds or dimensions are invalid.
     */
    oscean::core_services::GridData extractSubRegion(
        const oscean::core_services::GridData& raster,
        std::size_t startCol, std::size_t startRow,
        std::size_t numCols, std::size_t numRows) const;

    /**
     * @brief Computes pixel-wise statistics across multiple rasters (e.g., for time series analysis).
     * @param inputRasters Vector of input rasters. Must have same dimensions and CRS.
     * @param statisticType Type of statistic to compute (e.g., MIN, MAX, MEAN, SUM, STDDEV from StatisticalMeasure enum).
     * @return A new GridData object representing the computed statistical raster.
     * @throw SpatialOpsException if operation fails or inputs are invalid.
     */
    oscean::core_services::GridData computePixelwiseStatistics(
        const std::vector<oscean::core_services::GridData>& inputRasters,
        oscean::core_services::spatial_ops::StatisticalMeasure statisticType) const;

    /**
     * @brief Converts a geographic point to a grid cell index.
     * @param point The geographic point.
     * @param gridDef The definition of the grid.
     * @return Optional GridIndex. Returns std::nullopt if point is outside grid extent.
     */
    std::optional<oscean::core_services::GridIndex> pointToGridCell(
        const oscean::core_services::Point& point,
        const oscean::core_services::GridDefinition& gridDef) const;

    // Zonal Statistics overloads
    std::vector<oscean::core_services::spatial_ops::StatisticsResult> calculateZonalStatistics(
        const oscean::core_services::GridData& valueRaster,
        const oscean::core_services::Geometry& zoneGeometry,
        const oscean::core_services::spatial_ops::ZonalStatisticsOptions& options) const;

    std::map<oscean::core_services::spatial_ops::FeatureId, oscean::core_services::spatial_ops::StatisticsResult> calculateZonalStatistics(
        const oscean::core_services::GridData& valueRaster,
        const oscean::core_services::FeatureCollection& zoneFeatures,
        const oscean::core_services::spatial_ops::ZonalStatisticsOptions& options) const;
    
    std::map<int, oscean::core_services::spatial_ops::StatisticsResult> calculateZonalStatistics(
        const oscean::core_services::GridData& valueRaster,
        const oscean::core_services::GridData& zoneRaster, // Zone raster where each cell value represents a zone ID
        const oscean::core_services::spatial_ops::ZonalStatisticsOptions& options) const;

    /**
     * @brief Generates contour lines or polygons from a raster.
     * @param raster Input raster data.
     * @param options Contour generation options (interval, levels, output attribute etc.).
     * @return FeatureCollection containing contour lines or polygons.
     * @throw SpatialOpsException if operation fails.
     */
    oscean::core_services::FeatureCollection generateContours(
        const oscean::core_services::GridData& raster,
        const oscean::core_services::spatial_ops::ContourOptions& options) const;

    GridData applyMask(
        const GridData& raster,
        const GridData& maskRaster,
        const MaskOptions& options) const;

    GridData mosaic(
        const std::vector<std::shared_ptr<const GridData>>& sources,
        const MosaicOptions& options) const;

private:
    const oscean::core_services::spatial_ops::SpatialOpsConfig& m_config;
    
    // Modular components for different raster operations
    std::unique_ptr<oscean::core_services::spatial_ops::raster::RasterClipping> m_clipping;
    std::unique_ptr<oscean::core_services::spatial_ops::raster::RasterAlgebra> m_algebra;
    std::unique_ptr<oscean::core_services::spatial_ops::raster::RasterStatistics> m_statistics;
    std::unique_ptr<oscean::core_services::spatial_ops::raster::RasterVectorization> m_vectorization;

    // ✅ 添加辅助方法声明
    /**
     * @brief 获取数据类型对应的字节大小
     */
    size_t getSizeForDataType(oscean::core_services::DataType dataType) const;
    
    /**
     * @brief 设置栅格指定位置的像素值
     */
    void setPixelValue(oscean::core_services::GridData& raster, int row, int col, double value) const;

    /**
     * @brief 简化的要素栅格化辅助方法
     * @param feature 要栅格化的要素
     * @param gridDef 目标栅格定义
     * @param data 栅格数据指针
     * @param burnValue 燃烧值
     * @param allTouched 是否栅格化所有接触的像素
     */
    void rasterizeFeatureSimple(
        const oscean::core_services::Feature& feature,
        const oscean::core_services::GridDefinition& gridDef,
        float* data,
        float burnValue,
        bool allTouched) const;

    // Helper methods if needed, e.g., for GDAL dataset handling
    // void* openGdalDataset(const std::string& path) const;
    // void closeGdalDataset(void* hDataset) const;
    
    GDALDataset* createGdalDatasetFromGrid(const oscean::core_services::GridData& grid) const;
    GDALDataType toGdalDataType(oscean::core_services::DataType dt) const;
};

} // namespace oscean::core_services::spatial_ops::engine 