#pragma once

#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "core_services/common_data_types.h"
#include <memory>
#include <future>
#include <vector>
#include <optional>
#include <map>
#include <functional>
#include <chrono>

namespace oscean::core_services::spatial_ops::algorithms {

// 使用已定义的类型别名
using Geometry = oscean::core_services::Geometry;
using Point = oscean::core_services::Point;
using BoundingBox = oscean::core_services::BoundingBox;
using GridData = oscean::core_services::GridData;
using FeatureCollection = oscean::core_services::FeatureCollection;

// --- 枚举定义 - 移到接口声明之前 ---

/**
 * @brief 瓦片方案枚举
 */
enum class TileScheme {
    XYZ,        ///< XYZ tile scheme (Google/OSM style)
    TMS,        ///< TMS tile scheme (TileMapService)
    WMTS,       ///< WMTS tile scheme
    QUADKEY     ///< QuadKey tile scheme (Bing Maps style)
};

/**
 * @brief 瓦片格式枚举
 */
enum class TileFormat {
    PNG,        ///< PNG format
    JPEG,       ///< JPEG format
    WEBP,       ///< WebP format
    TIFF,       ///< TIFF format
    MVT         ///< Mapbox Vector Tiles
};

/**
 * @brief 瓦片优化级别
 */
enum class TileOptimizationLevel {
    NONE,       ///< No optimization
    BASIC,      ///< Basic optimization
    STANDARD,   ///< Standard optimization
    AGGRESSIVE  ///< Aggressive optimization
};

/**
 * @brief 质量度量类型
 */
enum class QualityMetric {
    COMPLETENESS,   ///< Data completeness
    SHARPNESS,      ///< Image sharpness
    CONTRAST,       ///< Image contrast
    NOISE_LEVEL,    ///< Noise level
    ARTIFACTS       ///< Artifact detection
};

/**
 * @brief 压缩类型
 */
enum class CompressionType {
    NONE,       ///< No compression
    LZW,        ///< LZW compression
    DEFLATE,    ///< Deflate compression
    JPEG,       ///< JPEG compression
    WEBP        ///< WebP compression
};

// 使用跨模块共用的重采样算法
using ResamplingMethod = oscean::core_services::ResampleAlgorithm;

// --- 结构体定义 - 移到接口声明之前 ---

/**
 * @brief 瓦片索引结构
 */
struct TileIndex {
    int x;                                  ///< Tile X coordinate
    int y;                                  ///< Tile Y coordinate
    int z;                                  ///< Zoom level
    
    bool operator<(const TileIndex& other) const {
        if (z != other.z) return z < other.z;
        if (y != other.y) return y < other.y;
        return x < other.x;
    }
    
    bool operator==(const TileIndex& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * @brief 瓦片位置结构
 */
struct TilePosition {
    int row;                                ///< Row position
    int col;                                ///< Column position
    
    bool operator<(const TilePosition& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
    
    bool operator==(const TilePosition& other) const {
        return row == other.row && col == other.col;
    }
};

/**
 * @brief 瓦片边界结构
 */
struct TileBoundary {
    BoundingBox geographicBounds;           ///< Geographic bounds in CRS units
    TileIndex tileIndex;                    ///< Tile index (x, y, z)
    int pixelWidth = 256;                   ///< Tile width in pixels
    int pixelHeight = 256;                  ///< Tile height in pixels
    std::string crs = "EPSG:3857";          ///< Coordinate reference system
    double pixelSizeX;                      ///< Pixel size in X direction
    double pixelSizeY;                      ///< Pixel size in Y direction
};

// 前向声明进度回调类型
using ProgressCallback = std::function<void(double)>;

/**
 * @brief 瓦片预处理选项
 */
struct TilePreprocessingOptions {
    bool createOverviews = true;            ///< Create overview pyramids
    bool optimizeForWeb = true;             ///< Optimize for web serving
    bool enableCompression = true;          ///< Enable tile compression
    CompressionType compressionType = CompressionType::LZW;
    int compressionQuality = 85;            ///< Compression quality (0-100)
    bool addAlphaChannel = false;           ///< Add alpha channel for transparency
    std::optional<double> noDataValue;      ///< NoData value to use
};

/**
 * @brief Web瓦片选项
 */
struct WebTileOptions {
    TileFormat outputFormat = TileFormat::PNG;
    bool enableCOG = true;                  ///< Enable Cloud Optimized GeoTIFF
    bool enableTiling = true;               ///< Enable internal tiling
    int blockSize = 512;                    ///< Internal block size
    bool addOverviews = true;               ///< Add overview levels
    std::vector<int> overviewFactors = {2, 4, 8, 16}; ///< Overview factors
};

/**
 * @brief 快速裁剪选项
 */
struct FastClipOptions {
    bool useApproximation = false;          ///< Use approximation for speed
    bool maintainDataType = true;           ///< Maintain original data type
    bool cropToExtent = true;               ///< Crop to exact extent
    std::optional<double> fillValue;        ///< Fill value for areas outside
    bool enableMultithreading = true;       ///< Enable multithreading
};

/**
 * @brief 批量裁剪选项
 */
struct BatchClipOptions {
    bool useParallelProcessing = true;      ///< Enable parallel processing
    std::size_t maxConcurrentTiles = 0;     ///< Maximum concurrent tiles (0 = auto)
    bool optimizeMemoryUsage = true;        ///< Optimize memory usage
    bool enableProgressReporting = false;   ///< Enable progress callbacks
    std::optional<ProgressCallback> progressCallback;
};

/**
 * @brief 矢量裁剪选项
 */
struct VectorClipOptions {
    bool preserveTopology = true;           ///< Preserve topology during clipping
    bool includePartialFeatures = true;     ///< Include partially intersecting features
    bool simplifyGeometry = false;          ///< Simplify clipped geometry
    double simplificationTolerance = 0.0;   ///< Simplification tolerance
    bool addTileAttributes = false;         ///< Add tile boundary attributes
};

/**
 * @brief 瓦片缓存验证结果
 */
struct TileCacheValidationResult {
    std::string cacheKey;                   ///< Cache key
    bool isValid;                           ///< Whether cache entry is valid
    std::optional<std::string> errorMessage; ///< Error message if invalid
    std::chrono::system_clock::time_point lastModified; ///< Last modification time
    std::size_t dataSize;                   ///< Size of cached data
};

/**
 * @brief 缓存验证选项
 */
struct CacheValidationOptions {
    bool checkTimestamp = true;             ///< Check timestamp validity
    bool checkDataIntegrity = true;         ///< Check data integrity
    bool enableDeepValidation = false;      ///< Enable deep validation
    std::chrono::seconds maxAge{3600};      ///< Maximum cache age
};

/**
 * @brief 瓦片质量度量
 */
struct TileQualityMetrics {
    double completeness;                    ///< Data completeness (0.0-1.0)
    double sharpness;                       ///< Image sharpness metric
    double contrast;                        ///< Image contrast metric
    double noiseLevel;                      ///< Noise level assessment
    bool hasArtifacts;                      ///< Whether artifacts are detected
    std::vector<std::string> qualityIssues; ///< Detailed quality issues
    double overallScore;                    ///< Overall quality score (0.0-1.0)
};

/**
 * @brief 瓦片一致性结果
 */
struct TileConsistencyResult {
    bool isConsistent;                      ///< Whether tiles are consistent
    std::vector<std::string> inconsistencies; ///< Detected inconsistencies
    std::map<TilePosition, double> edgeMatchScores; ///< Edge matching scores
    double overallConsistencyScore;         ///< Overall consistency score
};

/**
 * @brief 空瓦片检测选项
 */
struct EmptyTileDetectionOptions {
    double emptyThreshold = 0.01;           ///< Threshold for considering tile empty
    bool checkNoDataValues = true;          ///< Check for NoData values
    bool checkTransparency = true;          ///< Check for transparency
    bool enableStatisticalAnalysis = false; ///< Enable statistical analysis
};

/**
 * @brief 一致性选项
 */
struct ConsistencyOptions {
    double edgeMatchTolerance = 0.1;        ///< Edge matching tolerance
    bool checkColorConsistency = true;      ///< Check color consistency
    bool checkGeometricConsistency = true;  ///< Check geometric consistency
    bool enableDetailedAnalysis = false;    ///< Enable detailed analysis
};

/**
 * @brief 瓦片重投影选项
 */
struct TileReprojectionOptions {
    ResamplingMethod resamplingMethod = ResamplingMethod::BILINEAR;
    bool maintainPixelSize = false;         ///< Maintain pixel size during reprojection
    std::optional<double> targetResolution; ///< Target resolution
    bool cropToExtent = true;               ///< Crop to target extent
};

/**
 * @brief 方案转换选项
 */
struct SchemeTransformOptions {
    bool maintainQuality = true;            ///< Maintain quality during transformation
    bool enableOptimization = true;         ///< Enable optimization
    ResamplingMethod resamplingMethod = ResamplingMethod::BILINEAR;
};

/**
 * @brief 瓦片合并选项
 */
struct TileMergeOptions {
    bool enableBlending = true;             ///< Enable edge blending
    double blendingRadius = 2.0;            ///< Blending radius in pixels
    bool maintainDataType = true;           ///< Maintain original data type
    std::optional<double> noDataValue;      ///< NoData value for output
};

/**
 * @brief 镶嵌选项
 */
struct MosaicOptions {
    bool enableSeamlineOptimization = true; ///< Enable seamline optimization
    bool enableColorMatching = true;        ///< Enable color matching
    ResamplingMethod resamplingMethod = ResamplingMethod::BILINEAR;
    bool enableFeathering = true;           ///< Enable edge feathering
    double featheringRadius = 5.0;          ///< Feathering radius in pixels
};

/**
 * @brief 瓦片空间支持配置
 */
struct TileSpatialSupportConfig {
    int defaultTileSize = 256;              ///< Default tile size in pixels
    TileScheme defaultTileScheme = TileScheme::XYZ;
    std::string defaultCRS = "EPSG:3857";   ///< Default coordinate reference system
    bool enableCaching = true;              ///< Enable operation caching
    std::size_t cacheSize = 1000;           ///< Cache size
    bool enableParallelProcessing = true;   ///< Enable parallel processing
    std::size_t maxThreads = 0;             ///< Maximum threads (0 = auto)
    bool enableProgressReporting = false;   ///< Enable progress reporting
    double qualityThreshold = 0.8;          ///< Quality threshold for validation
};

// --- 接口定义 ---

/**
 * @brief Interface for spatial support to tile service
 * 
 * Provides spatial operations specifically designed to support
 * tile generation, caching, and rendering workflows.
 */
class ITileSpatialSupport {
public:
    virtual ~ITileSpatialSupport() = default;

    // --- Tile Boundary Operations ---

    /**
     * @brief Calculate tile boundaries for given zoom level and extent
     * @param extent Geographic extent
     * @param zoomLevel Tile zoom level
     * @param tileScheme Tile scheme (TMS, XYZ, etc.)
     * @return Future containing tile boundary definitions
     */
    virtual std::future<std::vector<TileBoundary>> calculateTileBoundaries(
        const BoundingBox& extent,
        int zoomLevel,
        TileScheme tileScheme = TileScheme::XYZ) const = 0;

    /**
     * @brief Get tile indices for geographic extent
     * @param extent Geographic extent
     * @param zoomLevel Tile zoom level
     * @param tileScheme Tile scheme
     * @return Future containing tile indices
     */
    virtual std::future<std::vector<TileIndex>> getTileIndices(
        const BoundingBox& extent,
        int zoomLevel,
        TileScheme tileScheme = TileScheme::XYZ) const = 0;

    /**
     * @brief Convert between different tile coordinate systems
     * @param tileIndex Source tile index
     * @param sourceScheme Source tile scheme
     * @param targetScheme Target tile scheme
     * @return Future containing converted tile index
     */
    virtual std::future<TileIndex> convertTileCoordinates(
        const TileIndex& tileIndex,
        TileScheme sourceScheme,
        TileScheme targetScheme) const = 0;

    /**
     * @brief Calculate optimal zoom level for given resolution
     * @param targetResolution Target resolution in meters per pixel
     * @param latitude Reference latitude for calculation
     * @return Future containing optimal zoom level
     */
    virtual std::future<int> calculateOptimalZoomLevel(
        double targetResolution,
        double latitude = 0.0) const = 0;

    // --- Raster Preprocessing for Tiles ---

    /**
     * @brief Preprocess raster data for tile generation
     * @param raster Input raster data
     * @param tileSize Target tile size in pixels
     * @param options Preprocessing options
     * @return Future containing preprocessed raster
     */
    virtual std::future<GridData> preprocessRasterForTiling(
        const GridData& raster,
        int tileSize,
        const TilePreprocessingOptions& options = {}) const = 0;

    /**
     * @brief Create overview pyramids for raster data
     * @param raster Input raster data
     * @param overviewLevels Number of overview levels
     * @param resamplingMethod Resampling method for overviews
     * @return Future containing raster with overviews
     */
    virtual std::future<std::vector<GridData>> createRasterOverviews(
        const GridData& raster,
        int overviewLevels,
        ResamplingMethod resamplingMethod = ResamplingMethod::AVERAGE) const = 0;

    /**
     * @brief Optimize raster for web tile serving
     * @param raster Input raster data
     * @param optimizationLevel Optimization level
     * @param options Optimization options
     * @return Future containing optimized raster
     */
    virtual std::future<GridData> optimizeRasterForWebTiles(
        const GridData& raster,
        TileOptimizationLevel optimizationLevel,
        const WebTileOptions& options = {}) const = 0;

    // --- Fast Clipping Operations ---

    /**
     * @brief Perform fast raster clipping for tile boundaries
     * @param raster Input raster data
     * @param tileBoundary Tile boundary for clipping
     * @param options Fast clipping options
     * @return Future containing clipped raster tile
     */
    virtual std::future<GridData> fastClipRasterToTile(
        const GridData& raster,
        const TileBoundary& tileBoundary,
        const FastClipOptions& options = {}) const = 0;

    /**
     * @brief Batch clip raster to multiple tiles
     * @param raster Input raster data
     * @param tileBoundaries Vector of tile boundaries
     * @param options Batch clipping options
     * @return Future containing vector of clipped tiles
     */
    virtual std::future<std::vector<GridData>> batchClipRasterToTiles(
        const GridData& raster,
        const std::vector<TileBoundary>& tileBoundaries,
        const BatchClipOptions& options = {}) const = 0;

    /**
     * @brief Clip vector features to tile boundaries
     * @param features Input feature collection
     * @param tileBoundary Tile boundary for clipping
     * @param options Vector clipping options
     * @return Future containing clipped features
     */
    virtual std::future<FeatureCollection> clipFeaturesToTile(
        const FeatureCollection& features,
        const TileBoundary& tileBoundary,
        const VectorClipOptions& options = {}) const = 0;

    // --- Tile Cache Management ---

    /**
     * @brief Calculate tile cache key for given parameters
     * @param tileIndex Tile index
     * @param layerId Layer identifier
     * @param styleId Style identifier
     * @param format Output format
     * @return Future containing cache key
     */
    virtual std::future<std::string> calculateTileCacheKey(
        const TileIndex& tileIndex,
        const std::string& layerId,
        const std::string& styleId = "",
        TileFormat format = TileFormat::PNG) const = 0;

    /**
     * @brief Validate tile cache consistency
     * @param cacheKeys Vector of cache keys to validate
     * @param validationOptions Validation options
     * @return Future containing validation results
     */
    virtual std::future<std::vector<TileCacheValidationResult>> validateTileCache(
        const std::vector<std::string>& cacheKeys,
        const CacheValidationOptions& validationOptions = {}) const = 0;

    /**
     * @brief Calculate tile dependencies for cache invalidation
     * @param changedExtent Geographic extent that changed
     * @param affectedLayers Layers affected by the change
     * @param maxZoomLevel Maximum zoom level to consider
     * @return Future containing tiles that need cache invalidation
     */
    virtual std::future<std::vector<TileIndex>> calculateTileDependencies(
        const BoundingBox& changedExtent,
        const std::vector<std::string>& affectedLayers,
        int maxZoomLevel) const = 0;

    // --- Tile Quality and Validation ---

    /**
     * @brief Assess tile quality metrics
     * @param tileData Tile raster data
     * @param qualityMetrics Metrics to calculate
     * @return Future containing quality assessment results
     */
    virtual std::future<TileQualityMetrics> assessTileQuality(
        const GridData& tileData,
        const std::vector<QualityMetric>& qualityMetrics) const = 0;

    /**
     * @brief Detect empty or invalid tiles
     * @param tiles Vector of tile data
     * @param detectionOptions Detection options
     * @return Future containing validity flags for each tile
     */
    virtual std::future<std::vector<bool>> detectEmptyTiles(
        const std::vector<GridData>& tiles,
        const EmptyTileDetectionOptions& detectionOptions = {}) const = 0;

    /**
     * @brief Validate tile spatial consistency
     * @param neighboringTiles Map of tile positions to tile data
     * @param consistencyOptions Consistency check options
     * @return Future containing consistency validation results
     */
    virtual std::future<TileConsistencyResult> validateTileConsistency(
        const std::map<TilePosition, GridData>& neighboringTiles,
        const ConsistencyOptions& consistencyOptions = {}) const = 0;

    // --- Tile Transformation and Reprojection ---

    /**
     * @brief Reproject tile to different coordinate system
     * @param tileData Input tile data
     * @param sourceCRS Source coordinate reference system
     * @param targetCRS Target coordinate reference system
     * @param reprojectionOptions Reprojection options
     * @return Future containing reprojected tile
     */
    virtual std::future<GridData> reprojectTile(
        const GridData& tileData,
        const std::string& sourceCRS,
        const std::string& targetCRS,
        const TileReprojectionOptions& reprojectionOptions = {}) const = 0;

    /**
     * @brief Transform tile to match target tile scheme
     * @param tileData Input tile data
     * @param sourceScheme Source tile scheme
     * @param targetScheme Target tile scheme
     * @param transformOptions Transformation options
     * @return Future containing transformed tile
     */
    virtual std::future<GridData> transformTileScheme(
        const GridData& tileData,
        TileScheme sourceScheme,
        TileScheme targetScheme,
        const SchemeTransformOptions& transformOptions = {}) const = 0;

    // --- Tile Merging and Mosaicking ---

    /**
     * @brief Merge multiple tiles into single tile
     * @param tiles Vector of input tiles with positions
     * @param targetBoundary Target tile boundary
     * @param mergeOptions Merge options
     * @return Future containing merged tile
     */
    virtual std::future<GridData> mergeTiles(
        const std::vector<std::pair<TilePosition, GridData>>& tiles,
        const TileBoundary& targetBoundary,
        const TileMergeOptions& mergeOptions = {}) const = 0;

    /**
     * @brief Create mosaic from tile collection
     * @param tiles Map of tile positions to tile data
     * @param mosaicBoundary Boundary for mosaic
     * @param mosaicOptions Mosaic creation options
     * @return Future containing mosaic raster
     */
    virtual std::future<GridData> createTileMosaic(
        const std::map<TilePosition, GridData>& tiles,
        const BoundingBox& mosaicBoundary,
        const MosaicOptions& mosaicOptions = {}) const = 0;

    // --- Configuration and Performance ---

    /**
     * @brief Set tile spatial support configuration
     * @param config Configuration parameters
     * @return Future indicating completion
     */
    virtual std::future<void> setConfiguration(
        const TileSpatialSupportConfig& config) = 0;

    /**
     * @brief Get current configuration
     * @return Future containing current configuration
     */
    virtual std::future<TileSpatialSupportConfig> getConfiguration() const = 0;

    /**
     * @brief Get supported tile schemes
     * @return Future containing list of supported tile schemes
     */
    virtual std::future<std::vector<std::string>> getSupportedTileSchemes() const = 0;

    /**
     * @brief Get performance metrics for tile operations
     * @return Future containing performance metrics
     */
    virtual std::future<PerformanceMetrics> getPerformanceMetrics() const = 0;
};

} // namespace oscean::core_services::spatial_ops::algorithms 