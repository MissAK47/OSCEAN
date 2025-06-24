/**
 * @file spatial_types.h
 * @brief Spatial operations service type definitions
 * 
 * 本文件定义空间服务专用的数据结构，不包含跨模块共用的结构。
 * 跨模块共用结构请参考 common_data_types.h
 */

#pragma once

// 防止Windows API宏干扰
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// 在任何Windows头文件被包含前取消这些宏定义
#undef min
#undef max
#undef DOMAIN
#undef KEY
#undef TYPE
#undef VALUE
#undef OPTIONAL
#undef ERROR
#undef SUCCESS
#undef FAILED

#include <string>
#include <vector>
#include <variant>
#include <map>
#include <chrono>
#include <functional>
#include <memory>
#include <boost/optional.hpp>

#include "core_services/common_data_types.h" // For oscean::core_services::DataType, oscean::core_services::GridData etc.

// 前向声明，避免循环依赖
namespace oscean::core_services {
    class GridData;
}

namespace oscean::core_services::spatial_ops {

// 🔧 使用跨模块共用结构的别名
using Geometry = oscean::core_services::Geometry;
using Point = oscean::core_services::Point;
using BoundingBox = oscean::core_services::BoundingBox;
using Feature = oscean::core_services::Feature;
using FeatureCollection = oscean::core_services::FeatureCollection;
using GridData = oscean::core_services::GridData;
using GridDefinition = oscean::core_services::GridDefinition;
using GridIndex = oscean::core_services::GridIndex;
using DataType = oscean::core_services::DataType;
using ResampleAlgorithm = oscean::core_services::ResampleAlgorithm;

// 🔧 空间服务专用枚举（仅空间服务使用）
/**
 * @enum SpatialPredicate
 * @brief Defines spatial predicates for spatial operations.
 */
enum class SpatialPredicate {
    UNKNOWN = 0,
    INTERSECTS,
    CONTAINS,
    WITHIN,
    OVERLAPS,
    CROSSES,
    DISJOINT,
    TOUCHES,
    EQUALS
};

/**
 * @enum DistanceType
 * @brief Defines distance calculation types.
 */
enum class DistanceType {
    EUCLIDEAN,
    GEODESIC
};

/**
 * @enum OverlayType
 * @brief Defines overlay operation types.
 */
enum class OverlayType {
    UNKNOWN = 0,
    INTERSECTION,
    UNION,
    DIFFERENCE_OP,
    SYM_DIFFERENCE_OP,
    CLIP,
    ERASE
};

/**
 * @enum RasterAlgebraOp
 * @brief Defines raster algebra operations.
 */
enum class RasterAlgebraOp {
    ADD, SUBTRACT, MULTIPLY, DIVIDE, POWER, MIN, MAX, MODULO,
    AND, OR, XOR, EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL,
    ABS, SQRT, LOG, LOG10, EXP, SIN, COS, TAN, ASIN, ACOS, ATAN,
    FLOOR, CEIL, ROUND, NEGATE, NOT
};

/**
 * @enum SpatialIndexType
 * @brief Defines spatial index types.
 */
enum class SpatialIndexType {
    NONE = 0,
    RTREE,
    QUADTREE,
    GRID,
    HASH,
    ADAPTIVE
};

/**
 * @enum BufferCapStyle
 * @brief Defines buffer cap styles.
 */
enum class BufferCapStyle {
    ROUND = 1,
    FLAT = 2,
    SQUARE = 3
};

/**
 * @enum BufferJoinStyle
 * @brief Defines buffer join styles.
 */
enum class BufferJoinStyle {
    ROUND = 1,
    MITRE = 2,
    BEVEL = 3
};

/**
 * @enum SimplificationAlgorithm
 * @brief Defines geometry simplification algorithms.
 */
enum class SimplificationAlgorithm {
    DOUGLAS_PEUCKER,
    VISVALINGAM_WHYATT,
    TOPOLOGY_PRESERVING
};

/**
 * @enum StatisticalMeasure
 * @brief Defines statistical measures for analysis.
 */
enum class StatisticalMeasure {
    MIN,
    MAX,
    MEAN,
    MEDIAN,
    MODE,
    SUM,
    COUNT,
    STDDEV,
    VARIANCE,
    RANGE
};

/**
 * @enum SpatialRelation
 * @brief Extended spatial relations for complex queries.
 */
enum class SpatialRelation {
    INTERSECTS,
    CONTAINS,
    WITHIN,
    OVERLAPS,
    CROSSES,
    DISJOINT,
    TOUCHES,
    EQUALS,
    COVERS,
    COVERED_BY,
    WITHIN_DISTANCE,
    BEYOND_DISTANCE
};

// 🔧 空间服务专用选项结构（仅空间服务使用）

/**
 * @struct BufferOptions
 * @brief Options for buffer analysis.
 */
struct BufferOptions {
    int quadrantSegments = 8;
    BufferCapStyle capStyle = BufferCapStyle::ROUND;
    BufferJoinStyle joinStyle = BufferJoinStyle::ROUND;
    double mitreLimit = 5.0;
    bool singleSided = false;
};

/**
 * @struct MaskOptions
 * @brief Options for raster masking or clipping by geometry.
 */
struct MaskOptions {
    bool invertMask = false;
    boost::optional<double> maskValue = 1.0;
    bool allTouched = false;
    boost::optional<double> outputNoDataValue = boost::none;
};

/**
 * @struct ZonalStatisticsOptions
 * @brief Options for zonal statistics.
 */
struct ZonalStatisticsOptions {
    std::vector<StatisticalMeasure> statistics;
    boost::optional<double> noDataValueToIgnore = boost::none;
};

/**
 * @brief 🆕 栅格融合（镶嵌）操作的选项。
 */
struct MosaicOptions {
    /**
     * @brief 定义如何处理重叠区域的像素值。
     */
    enum class MosaicOverlapMethod {
        FIRST,      // 使用第一个栅格的值
        LAST,       // 使用最后一个栅格的值 (默认)
        MIN,        // 使用所有重叠像素中的最小值
        MAX,        // 使用所有重叠像素中的最大值
        MEAN,       // 使用所有重叠像素的平均值
        SUM         // 使用所有重叠像素的总和
    };

    MosaicOverlapMethod method = MosaicOverlapMethod::LAST;
    boost::optional<double> outputNoDataValue = boost::none; // 输出栅格的NoData值
    // 目标分辨率，如果未指定，则使用第一个栅格的分辨率
    boost::optional<double> targetResolution = boost::none;
};

/**
 * @struct ContourOptions
 * @brief Options for contour generation.
 */
struct ContourOptions {
    std::variant<double, std::vector<double>> intervalOrLevels;
    std::string outputAttributeName = "level";
    boost::optional<double> noDataValueToIgnore = boost::none;
};

/**
 * @struct RasterizeOptions
 * @brief Options for rasterizing vector features.
 */
struct RasterizeOptions {
    boost::optional<std::string> attributeField = boost::none;
    boost::optional<double> burnValue = 1.0;
    boost::optional<double> backgroundValue = 0.0;
    bool allTouched = false;
    boost::optional<double> noDataValue = boost::none;
};

/**
 * @struct SimplificationOptions
 * @brief Options for geometry simplification.
 */
struct SimplificationOptions {
    double tolerance = 0.0;
    SimplificationAlgorithm algorithm = SimplificationAlgorithm::DOUGLAS_PEUCKER;
    bool preserveTopology = true;
    bool preserveEndpoints = true;
};

/**
 * @struct ResampleOptions
 * @brief Options for raster resampling operations.
 */
struct ResampleOptions {
    ResampleAlgorithm method = ResampleAlgorithm::NEAREST;
    DataType outputDataType = DataType::Unknown; // Unknown means same as input
    boost::optional<double> noDataValue = boost::none;
    bool maintainAspectRatio = true;
    bool useMultithreading = true;
    double targetResolutionX = 0.0;
    double targetResolutionY = 0.0;
    boost::optional<std::string> targetCRS;
    boost::optional<GridDefinition> targetGrid; // Alternative to resolution-based resampling
};

/**
 * @struct StatisticsOptions
 * @brief Options for statistical calculations.
 */
struct StatisticsOptions {
    std::vector<StatisticalMeasure> measures = {StatisticalMeasure::MEAN, StatisticalMeasure::MIN, StatisticalMeasure::MAX};
    boost::optional<double> noDataValue = boost::none;
    bool ignoreNoData = true;
    bool computeHistogram = false;
    int histogramBins = 256;
    boost::optional<double> histogramMin = boost::none;
    boost::optional<double> histogramMax = boost::none;
};

/**
 * @struct SpatialQueryOptions
 * @brief Options for spatial queries.
 */
struct SpatialQueryOptions {
    SpatialRelation relation = SpatialRelation::INTERSECTS;
    boost::optional<double> distance; // For distance-based queries
    bool useIndex = true;
    std::size_t maxResults = 0; // 0 = no limit
    bool returnGeometry = true;
    std::vector<std::string> attributesToReturn; // Empty = return all
};

/**
 * @struct ValidationOptions
 * @brief Options for geometry validation.
 */
struct ValidationOptions {
    bool fixInvalidGeometries = false;
    double tolerance = 1e-10;
    bool checkSelfIntersections = true;
    bool checkRingOrientation = true;
    bool checkDuplicatePoints = true;
};

// 🔧 空间服务专用结果结构（仅空间服务使用）

/**
 * @struct StatisticsResult
 * @brief Result structure for statistical operations.
 */
struct StatisticsResult {
    std::map<StatisticalMeasure, double> values;
    boost::optional<std::string> zoneIdentifier;
};

/**
 * @struct SpatialQueryResult
 * @brief Result structure for spatial queries.
 */
struct SpatialQueryResult {
    std::vector<std::variant<int, std::string>> featureIds;
    std::vector<std::string> geometries; // WKT format
    std::map<std::string, std::vector<std::variant<int, double, std::string>>> attributes;
    std::size_t totalCount; // Total matching features (may be > returned if limited)
    std::chrono::milliseconds executionTime;
};

/**
 * @struct ValidationResult
 * @brief Result structure for geometry validation.
 */
struct ValidationResult {
    bool isValid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    boost::optional<std::string> fixedGeometry; // WKT of fixed geometry if applicable
};

/**
 * @struct PerformanceMetrics
 * @brief Performance metrics for spatial operations.
 */
struct PerformanceMetrics {
    std::chrono::milliseconds executionTime;
    std::size_t memoryUsed; // Bytes
    std::size_t peakMemoryUsed; // Bytes
    std::size_t inputDataSize; // Bytes
    std::size_t outputDataSize; // Bytes
    std::size_t threadsUsed;
    bool indexUsed = false;
    std::string operationType;
    std::map<std::string, std::variant<int, double, std::string>> customMetrics;
};

// === ADVANCED SPATIAL ANALYSIS TYPES ===

/**
 * @struct RasterExpression
 * @brief Represents a custom raster algebra expression.
 */
struct RasterExpression {
    std::string expression;
    std::map<std::string, GridData> inputs;
    std::map<std::string, double> constants; // Named constants in expression
    boost::optional<double> noDataValue;
    bool useParallelProcessing = true;
};

/**
 * @struct SpatialOperationContext
 * @brief Context information for spatial operations
 */
struct SpatialOperationContext {
    std::string operationId;
    std::string userId;
    std::chrono::system_clock::time_point startTime;
    std::map<std::string, std::variant<int, double, std::string>> parameters;
    boost::optional<std::function<void(double)>> progressCallback; // Progress 0.0-1.0
    bool enableCaching = true;
    boost::optional<std::string> cacheKey;
};

// --- Type Aliases for Convenience ---

using GeometryWKT = std::string;
using FeatureId = std::variant<int, std::string>;
using AttributeValue = std::variant<int, double, std::string>;
using AttributeMap = std::map<std::string, AttributeValue>;
using StatisticsCollection = std::vector<StatisticsResult>;

// --- Function Type Definitions ---

using ProgressCallback = std::function<void(double progress, const std::string& message)>;
using ValidationCallback = std::function<bool(const std::string& warning)>;
using ErrorCallback = std::function<void(const std::string& error)>;

// Type aliases for convenience
using GeometryPtr = std::unique_ptr<Geometry>;
using FeaturePtr = std::unique_ptr<Feature>;
using GridDataPtr = std::unique_ptr<GridData>;

// === CONTAINMENT ANALYSIS TYPES ===

/**
 * @enum ContainmentType
 * @brief 包含关系类型
 */
enum class ContainmentType {
    DISJOINT,      ///< 分离
    TOUCHES,       ///< 接触
    INTERSECTS,    ///< 相交
    OVERLAPS,      ///< 重叠
    CONTAINS,      ///< 包含
    WITHIN         ///< 在内部
};

/**
 * @enum BoundaryIntersectionType
 * @brief 边界相交类型
 */
enum class BoundaryIntersectionType {
    NONE,          ///< 无相交
    POINT,         ///< 点相交
    LINE,          ///< 线相交
    COMPLEX        ///< 复杂相交
};

/**
 * @struct AdaptiveBufferParams
 * @brief 自适应缓冲区参数
 */
struct AdaptiveBufferParams {
    int baseQuadrantSegments = 8;
    int minQuadrantSegments = 4;
    int maxQuadrantSegments = 32;
    int complexityThreshold = 100;
    double complexityFactor = 1.5;
    bool useDistanceAdaptation = false;
    double referenceSizeThreshold = 1000.0;
    double sizeAdaptationFactor = 0.1;
};

/**
 * @struct BufferQualityMetrics
 * @brief 缓冲区质量指标
 */
struct BufferQualityMetrics {
    double areaRatio = 0.0;
    double perimeterRatio = 0.0;
    double distanceAccuracy = 0.0;
    int vertexCount = 0;
    bool isValid = false;
    double complexityScore = 0.0;
};

/**
 * @struct ContainmentResult
 * @brief 包含关系结果
 */
struct ContainmentResult {
    ContainmentType relationship = ContainmentType::DISJOINT;
    bool isOnBoundary = false;
    double distanceToBoundary = -1.0;
};

/**
 * @struct DetailedContainmentResult
 * @brief 详细包含关系结果
 */
struct DetailedContainmentResult {
    ContainmentType primaryRelationship = ContainmentType::DISJOINT;
    bool contains = false;
    bool within = false;
    bool touches = false;
    bool overlaps = false;
    bool intersects = false;
    bool covers = false;
    bool coveredBy = false;
    double intersectionRatio = 0.0;
    double distance = 0.0;
    double boundaryDistance = 0.0;
};

/**
 * @struct ContainmentNode
 * @brief 包含关系节点
 */
struct ContainmentNode {
    size_t geometryIndex = 0;
    int level = 0;
    boost::optional<size_t> parentIndex;
    std::vector<size_t> childrenIndices;
};

/**
 * @struct HierarchicalContainment
 * @brief 层次包含关系
 */
struct HierarchicalContainment {
    std::vector<std::vector<bool>> containmentMatrix;
    std::vector<int> levels;
    std::vector<ContainmentNode> nodes;
    std::vector<size_t> rootIndices;
};

/**
 * @struct BoundaryRelationship
 * @brief 边界关系信息
 */
struct BoundaryRelationship {
    bool boundariesIntersect = false;
    BoundaryIntersectionType intersectionType = BoundaryIntersectionType::NONE;
    int intersectionPointCount = 0;
    double intersectionLength = 0.0;
    double boundary1Length = 0.0;
    double boundary2Length = 0.0;
    double intersectionRatio = 0.0;
    bool interior1OnBoundary2 = false;
    bool interior2OnBoundary1 = false;
};

/**
 * @struct ContainmentStatistics
 * @brief 包含关系统计信息
 */
struct ContainmentStatistics {
    size_t totalCount = 0;
    size_t containsCount = 0;
    size_t withinCount = 0;
    size_t touchesCount = 0;
    size_t overlapsCount = 0;
    size_t intersectsCount = 0;
    size_t disjointCount = 0;
    size_t onBoundaryCount = 0;
    size_t validDistanceCount = 0;
    double totalDistance = 0.0;
    double averageDistance = 0.0;
    double containsPercentage = 0.0;
    double withinPercentage = 0.0;
    double touchesPercentage = 0.0;
    double overlapsPercentage = 0.0;
    double intersectsPercentage = 0.0;
    double disjointPercentage = 0.0;
    double onBoundaryPercentage = 0.0;
};

// === INTERSECTION ANALYSIS TYPES ===

/**
 * @struct IntersectionResult
 * @brief 相交分析结果
 */
struct IntersectionResult {
    size_t index = 0;
    bool intersects = false;
    void* intersection = nullptr; // GEOSGeometry*
    double intersectionArea = 0.0;
};

/**
 * @struct IntersectionStatistics
 * @brief 相交统计信息
 */
struct IntersectionStatistics {
    int geometryType = 0;
    double area = 0.0;
    double length = 0.0;
    int vertexCount = 0;
    int componentCount = 0;
    bool isValid = false;
    bool isEmpty = false;
};

// === DISTANCE ANALYSIS TYPES ===

/**
 * @struct NearestPointsResult
 * @brief 最近点结果
 */
struct NearestPointsResult {
    void* point1 = nullptr; // GEOSGeometry*
    void* point2 = nullptr; // GEOSGeometry*
    double distance = 0.0;
};

/**
 * @struct DistanceResult
 * @brief 距离计算结果
 */
struct DistanceResult {
    size_t index = 0;
    double distance = 0.0;
    bool isValid = false;
    bool intersects = false;
};

/**
 * @struct LineDistanceResult
 * @brief 线距离结果
 */
struct LineDistanceResult {
    double distance = 0.0;
    void* nearestPointOnGeom = nullptr; // GEOSGeometry*
    void* nearestPointOnLine = nullptr; // GEOSGeometry*
    double lineParameter = 0.0;
    bool intersects = false;
    void* intersectionGeometry = nullptr; // GEOSGeometry*
    double intersectionLength = 0.0;
};

/**
 * @struct DistanceStatistics
 * @brief 距离统计信息
 */
struct DistanceStatistics {
    size_t count = 0;
    double minDistance = 0.0;
    double maxDistance = 0.0;
    double meanDistance = 0.0;
    double medianDistance = 0.0;
    double standardDeviation = 0.0;
    double firstQuartile = 0.0;
    double thirdQuartile = 0.0;
};

// === GEOMETRY ANALYSIS TYPES ===

/**
 * @struct GeometryStatistics
 * @brief 几何统计信息
 */
struct GeometryStatistics {
    std::size_t totalGeometries = 0;        ///< Total number of geometries
    std::size_t validGeometries = 0;        ///< Number of valid geometries
    std::size_t invalidGeometries = 0;      ///< Number of invalid geometries
    
    std::map<Geometry::Type, std::size_t> geometryTypeCounts; ///< Count by geometry type
    
    BoundingBox overallBounds;           ///< Bounding box of all geometries
    double totalArea = 0.0;                 ///< Total area of all geometries
    double totalLength = 0.0;               ///< Total length of all linear geometries
    
    std::chrono::milliseconds processingTime{0}; ///< Time taken to compute statistics
};

/**
 * @struct GeometryQualityMetrics
 * @brief 几何质量指标
 */
struct GeometryQualityMetrics {
    bool isValid = true;                    ///< Overall validity
    bool isSimple = true;                   ///< Geometry is simple (no self-intersections)
    bool isClosed = true;                   ///< Geometry is closed (for applicable types)
    bool hasCorrectOrientation = true;      ///< Correct ring orientation
    
    std::size_t selfIntersectionCount = 0;  ///< Number of self-intersections
    std::size_t duplicatePointCount = 0;    ///< Number of duplicate consecutive points
    std::size_t degenerateSegmentCount = 0; ///< Number of degenerate segments
    
    double minimumSegmentLength = 0.0;      ///< Shortest segment length
    double maximumSegmentLength = 0.0;      ///< Longest segment length
    double averageSegmentLength = 0.0;      ///< Average segment length
    
    std::vector<std::string> qualityIssues; ///< Detailed quality issue descriptions
};

// === RASTER ANALYSIS TYPES ===

/**
 * @struct RasterClipOptions
 * @brief Options for raster clipping operations
 */
struct RasterClipOptions {
    bool cropToExtent = true;
    bool maintainResolution = true;
    boost::optional<double> noDataValue;
    bool compressOutput = false;
};

/**
 * @struct RasterAlgebraOptions
 * @brief Options for raster algebra operations
 */
struct RasterAlgebraOptions {
    boost::optional<double> noDataValue;
    bool ignoreNoData = true;
    DataType outputDataType = DataType::Float32;
    bool useParallelProcessing = true;
    std::size_t blockSize = 512;
};

// === QUERY OPTIMIZATION TYPES ===

/**
 * @enum QueryOptimizationLevel
 * @brief Defines optimization levels for spatial queries
 */
enum class QueryOptimizationLevel {
    NONE, BASIC, STANDARD, AGGRESSIVE
};

/**
 * @enum QueryExecutionStrategy
 * @brief Defines execution strategies for spatial queries
 */
enum class QueryExecutionStrategy {
    AUTO, SEQUENTIAL, PARALLEL, INDEXED, HYBRID
};

/**
 * @enum NearestNeighborAlgorithm
 * @brief Defines algorithms for nearest neighbor search
 */
enum class NearestNeighborAlgorithm {
    BRUTE_FORCE, KD_TREE, R_TREE, GRID_INDEX, ADAPTIVE
};

/**
 * @enum DistanceMetric
 * @brief Defines distance metrics for spatial queries
 */
enum class DistanceMetric {
    EUCLIDEAN, MANHATTAN, CHEBYSHEV, GEODESIC, HAVERSINE
};

/**
 * @struct NearestNeighborOptions
 * @brief Options for nearest neighbor queries
 */
struct NearestNeighborOptions {
    DistanceMetric distanceMetric = DistanceMetric::EUCLIDEAN;
    NearestNeighborAlgorithm algorithm = NearestNeighborAlgorithm::ADAPTIVE;
    boost::optional<double> maxDistance;
    bool returnDistance = false;
    bool sortByDistance = true;
    boost::optional<std::string> distanceField = "distance";
};

/**
 * @struct GridQueryOptions
 * @brief Options for grid cell queries
 */
struct GridQueryOptions {
    bool handleEdgeCases = true;
    bool useApproximation = false;
    double tolerance = 1e-10;
    bool validateInput = true;
    bool transformCoordinates = true;
};

// === PARALLEL EXECUTION TYPES ===

/**
 * @enum TaskPriority
 * @brief Defines priority levels for parallel tasks
 */
enum class TaskPriority {
    LOW = 0, NORMAL = 1, HIGH = 2, CRITICAL = 3
};

/**
 * @enum TaskState
 * @brief Defines states for parallel tasks
 */
enum class TaskState {
    PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT
};

/**
 * @enum LoadBalancingStrategy
 * @brief Defines load balancing strategies for parallel execution
 */
enum class LoadBalancingStrategy {
    ROUND_ROBIN, LEAST_LOADED, WORK_STEALING, DYNAMIC, CUSTOM
};

/**
 * @enum ParallelizationStrategy
 * @brief Defines parallelization strategies
 */
enum class ParallelizationStrategy {
    DATA_PARALLEL, TASK_PARALLEL, PIPELINE, HYBRID, AUTO
};

/**
 * @struct TaskExecutionOptions
 * @brief Options for task execution
 */
struct TaskExecutionOptions {
    TaskPriority priority = TaskPriority::NORMAL;
    boost::optional<std::chrono::milliseconds> timeout;
    std::size_t maxRetries = 0;
    bool enableProgressReporting = false;
    std::map<std::string, std::string> customOptions;
};

// === INDEX MANAGEMENT TYPES ===

/**
 * @enum IndexStorageType
 * @brief Defines storage types for spatial indexes
 */
enum class IndexStorageType {
    MEMORY, DISK, HYBRID, DISTRIBUTED
};

/**
 * @enum IndexCompressionType
 * @brief Defines compression types for spatial indexes
 */
enum class IndexCompressionType {
    NONE, LZ4, ZSTD, GZIP, CUSTOM
};

/**
 * @enum IndexUpdateStrategy
 * @brief Defines update strategies for spatial indexes
 */
enum class IndexUpdateStrategy {
    IMMEDIATE, BATCH, LAZY, SCHEDULED, ADAPTIVE
};

/**
 * @struct IndexCreationOptions
 * @brief Options for creating spatial indexes
 */
struct IndexCreationOptions {
    std::size_t maxDepth = 10;
    std::size_t maxItemsPerNode = 10;
    std::size_t minItemsPerNode = 2;
    bool enableBulkLoading = true;
    bool enableCompression = false;
    boost::optional<BoundingBox> bounds;
    std::map<std::string, std::string> customOptions;
};

/**
 * @struct IndexQueryOptions
 * @brief Options for querying spatial indexes
 */
struct IndexQueryOptions {
    std::size_t maxResults = 0;
    bool returnGeometry = true;
    bool returnAttributes = true;
    std::vector<std::string> attributeFilter;
    bool useApproximation = false;
    double tolerance = 1e-10;
};

/**
 * @struct ParallelExecutionOptions
 * @brief Options for parallel execution
 */
struct ParallelExecutionOptions {
    bool preferParallel = true;
    std::size_t estimatedDataSize = 0;
};

} // namespace oscean::core_services::spatial_ops

// 在所有包含之后再次取消定义可能的宏
#undef min
#undef max
#undef DOMAIN
#undef KEY
#undef TYPE
#undef VALUE
#undef OPTIONAL
#undef ERROR
#undef SUCCESS
#undef FAILED 
