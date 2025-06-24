#pragma once

#include "tile_spatial_support.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/spatial_ops/spatial_exceptions.h"

namespace oscean::core_services::spatial_ops::algorithms {

/**
 * @brief Implementation of spatial support for tile service
 */
class TileSpatialSupportImpl : public ITileSpatialSupport {
public:
    explicit TileSpatialSupportImpl(const SpatialOpsConfig& config);
    ~TileSpatialSupportImpl() override = default;

    // --- Tile Boundary Operations ---
    std::future<std::vector<TileBoundary>> calculateTileBoundaries(
        const BoundingBox& extent,
        int zoomLevel,
        TileScheme tileScheme = TileScheme::XYZ) const override;

    std::future<std::vector<TileIndex>> getTileIndices(
        const BoundingBox& extent,
        int zoomLevel,
        TileScheme tileScheme = TileScheme::XYZ) const override;

    std::future<TileIndex> convertTileCoordinates(
        const TileIndex& tileIndex,
        TileScheme sourceScheme,
        TileScheme targetScheme) const override;

    std::future<int> calculateOptimalZoomLevel(
        double targetResolution,
        double latitude = 0.0) const override;

    // --- Raster Preprocessing for Tiles ---
    std::future<GridData> preprocessRasterForTiling(
        const GridData& raster,
        int tileSize,
        const TilePreprocessingOptions& options = {}) const override;

    std::future<std::vector<GridData>> createRasterOverviews(
        const GridData& raster,
        int overviewLevels,
        ResamplingMethod resamplingMethod = ResamplingMethod::AVERAGE) const override;

    std::future<GridData> optimizeRasterForWebTiles(
        const GridData& raster,
        TileOptimizationLevel optimizationLevel,
        const WebTileOptions& options = {}) const override;

    // --- Fast Clipping Operations ---
    std::future<GridData> fastClipRasterToTile(
        const GridData& raster,
        const TileBoundary& tileBoundary,
        const FastClipOptions& options = {}) const override;

    std::future<std::vector<GridData>> batchClipRasterToTiles(
        const GridData& raster,
        const std::vector<TileBoundary>& tileBoundaries,
        const BatchClipOptions& options = {}) const override;

    std::future<FeatureCollection> clipFeaturesToTile(
        const FeatureCollection& features,
        const TileBoundary& tileBoundary,
        const VectorClipOptions& options = {}) const override;

    // --- Tile Cache Management ---
    std::future<std::string> calculateTileCacheKey(
        const TileIndex& tileIndex,
        const std::string& layerId,
        const std::string& styleId = "",
        TileFormat format = TileFormat::PNG) const override;

    std::future<std::vector<TileCacheValidationResult>> validateTileCache(
        const std::vector<std::string>& cacheKeys,
        const CacheValidationOptions& validationOptions = {}) const override;

    std::future<std::vector<TileIndex>> calculateTileDependencies(
        const BoundingBox& changedExtent,
        const std::vector<std::string>& affectedLayers,
        int maxZoomLevel) const override;

    // --- Tile Quality and Validation ---
    std::future<TileQualityMetrics> assessTileQuality(
        const GridData& tileData,
        const std::vector<QualityMetric>& qualityMetrics) const override;

    std::future<std::vector<bool>> detectEmptyTiles(
        const std::vector<GridData>& tiles,
        const EmptyTileDetectionOptions& detectionOptions = {}) const override;

    std::future<TileConsistencyResult> validateTileConsistency(
        const std::map<TilePosition, GridData>& neighboringTiles,
        const ConsistencyOptions& consistencyOptions = {}) const override;

    // --- Coordinate Transformation ---
    std::future<GridData> reprojectTile(
        const GridData& tileData,
        const std::string& sourceCRS,
        const std::string& targetCRS,
        const TileReprojectionOptions& reprojectionOptions = {}) const override;

    std::future<GridData> transformTileScheme(
        const GridData& tileData,
        TileScheme sourceScheme,
        TileScheme targetScheme,
        const SchemeTransformOptions& transformOptions = {}) const override;

    // --- Tile Merging and Mosaicking ---
    std::future<GridData> mergeTiles(
        const std::vector<std::pair<TilePosition, GridData>>& tiles,
        const TileBoundary& targetBoundary,
        const TileMergeOptions& mergeOptions = {}) const override;

    std::future<GridData> createTileMosaic(
        const std::map<TilePosition, GridData>& tiles,
        const BoundingBox& mosaicBoundary,
        const MosaicOptions& mosaicOptions = {}) const override;

    // --- Configuration ---
    std::future<void> setConfiguration(
        const TileSpatialSupportConfig& config) override;

    std::future<TileSpatialSupportConfig> getConfiguration() const override;

    std::future<std::vector<std::string>> getSupportedTileSchemes() const override;

    std::future<PerformanceMetrics> getPerformanceMetrics() const override;

private:
    const SpatialOpsConfig& m_config;
    TileSpatialSupportConfig m_supportConfig;
    
    // Helper methods
    TileBoundary calculateSingleTileBoundary(
        const BoundingBox& extent, int x, int y, int z, TileScheme scheme) const;
    
    double calculatePixelSize(int zoomLevel, double latitude = 0.0) const;
    
    std::pair<int, int> lonLatToTileXY(double lon, double lat, int zoom) const;
    
    BoundingBox tileXYToBounds(int x, int y, int zoom) const;
    
    std::string generateCacheKey(const std::vector<std::string>& components) const;
    
    bool isEmptyTile(const GridData& tile, double threshold = 0.01) const;
    
    double calculateEdgeMatch(const GridData& tile1, const GridData& tile2, 
                             const std::string& edge) const;
};

} // namespace oscean::core_services::spatial_ops::algorithms 