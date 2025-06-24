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

// 枚举类型定义
enum class MeshType {
    TRIANGULAR,     ///< 三角形网格
    QUADRILATERAL,  ///< 四边形网格
    TETRAHEDRAL,    ///< 四面体网格（3D）
    HEXAHEDRAL,     ///< 六面体网格（3D）
    MIXED           ///< 混合网格
};

enum class BoundaryType {
    INLET,      ///< 入口边界
    OUTLET,     ///< 出口边界
    WALL,       ///< 壁面边界
    SYMMETRY,   ///< 对称边界
    PERIODIC,   ///< 周期边界
    OPEN,       ///< 开放边界
    FIXED,      ///< 固定边界
    FREE        ///< 自由边界
};

enum class FieldType { 
    SCALAR,     ///< 标量场
    VECTOR,     ///< 矢量场
    TENSOR      ///< 张量场
};

enum class FieldLocation { 
    NODE,           ///< 节点位置
    ELEMENT_CENTER, ///< 单元中心
    FACE_CENTER,    ///< 面中心
    EDGE_CENTER     ///< 边中心
};

// 使用跨模块共用的插值方法
// 注意：这里保留了一个本地的InterpolationMethod用于建模特定的插值方法
// 如果需要与通用插值方法保持一致，可以使用别名
// enum class InterpolationMethod { 
//     NEAREST,    ///< 最近邻插值
//     BILINEAR,   ///< 双线性插值
//     BICUBIC,    ///< 双三次插值
//     KRIGING     ///< 克里金插值（建模专用）
// };

enum class GradientCalculationMethod { 
    FINITE_DIFFERENCE,  ///< 有限差分
    FINITE_ELEMENT,     ///< 有限元
    LEAST_SQUARES       ///< 最小二乘
};

enum class SpatialOperatorType {
    GRADIENT,       ///< 梯度算子
    DIVERGENCE,     ///< 散度算子
    CURL,           ///< 旋度算子
    LAPLACIAN,      ///< 拉普拉斯算子
    BIHARMONIC,     ///< 双调和算子
    CONVECTION,     ///< 对流算子
    DIFFUSION       ///< 扩散算子
};

enum class SpatialStatisticsType { 
    BASIC,              ///< 基本统计
    ADVANCED,           ///< 高级统计
    AUTOCORRELATION,    ///< 自相关
    VARIOGRAM           ///< 变异函数
};

enum class CorrelationMethod { 
    PEARSON,                ///< 皮尔逊相关
    SPEARMAN,               ///< 斯皮尔曼相关
    SPATIAL_CORRELATION     ///< 空间相关
};

enum class SpatialPatternType { 
    HOTSPOTS,   ///< 热点
    CLUSTERS,   ///< 聚类
    TRENDS,     ///< 趋势
    ANOMALIES   ///< 异常
};

enum class SamplingStrategy {
    RANDOM,         ///< 随机采样
    SYSTEMATIC,     ///< 系统采样
    STRATIFIED,     ///< 分层采样
    ADAPTIVE,       ///< 自适应采样
    OPTIMAL         ///< 最优采样
};

enum class FieldExtractionMethod { 
    NEAREST_NODE,       ///< 最近节点
    INTERPOLATED,       ///< 插值
    ELEMENT_AVERAGE     ///< 单元平均
};

enum class BoundaryConditionType { 
    DIRICHLET,  ///< 第一类边界条件
    NEUMANN,    ///< 第二类边界条件
    ROBIN,      ///< 第三类边界条件
    MIXED       ///< 混合边界条件
};

enum class RefinementType { 
    UNIFORM,        ///< 均匀细化
    ADAPTIVE,       ///< 自适应细化
    GRADIENT_BASED, ///< 基于梯度的细化
    ERROR_BASED     ///< 基于误差的细化
};

enum class InitializationMethod { 
    CONSTANT,       ///< 常数初始化
    FUNCTION,       ///< 函数初始化
    INTERPOLATION,  ///< 插值初始化
    RANDOM          ///< 随机初始化
};

enum class MeshQualityMetric { 
    ASPECT_RATIO,   ///< 长宽比
    SKEWNESS,       ///< 偏斜度
    ORTHOGONALITY,  ///< 正交性
    SMOOTHNESS      ///< 平滑性
};

enum class SensitivityMetric { 
    LOCAL_SENSITIVITY,  ///< 局部敏感性
    GLOBAL_SENSITIVITY, ///< 全局敏感性
    SOBOL_INDICES       ///< Sobol指数
};

enum class MatrixFormat { 
    DENSE,  ///< 稠密矩阵
    SPARSE, ///< 稀疏矩阵
    BANDED  ///< 带状矩阵
};

// 结构体定义 - 移到接口声明之前
struct DomainCreationOptions {
    bool simplifyBoundary = true;           ///< 是否简化边界
    double simplificationTolerance = 0.1;   ///< 简化容差
    std::string crs = "EPSG:4326";          ///< 坐标参考系统
};

struct ComputationalDomain {
    Geometry boundary;                      ///< 域边界几何
    double resolution;                      ///< 空间分辨率
    std::string crs;                        ///< 坐标参考系统
    BoundingBox bounds;                     ///< 域边界框
    std::vector<Geometry> holes;            ///< 内部孔洞/排除区域
    std::map<std::string, std::string> properties; ///< 域属性
};

struct MeshGenerationOptions {
    double minElementSize = 0.0;            ///< 最小单元尺寸
    double maxElementSize = 0.0;            ///< 最大单元尺寸
    bool smoothMesh = true;                 ///< 是否平滑网格
};

struct MeshQualityMetrics {
    double minAngle = 0.0;                  ///< 最小角度
    double maxAngle = 0.0;                  ///< 最大角度
    double minAspectRatio = 0.0;            ///< 最小长宽比
    double avgAspectRatio = 0.0;            ///< 平均长宽比
};

struct BoundarySegment {
    std::vector<Point> points;              ///< 定义段的点
    BoundaryType boundaryType;              ///< 边界类型
    std::string boundaryId;                 ///< 边界标识符
    std::map<std::string, double> properties; ///< 边界属性
};

struct SpatialMesh {
    std::vector<Point> nodes;               ///< 网格节点
    std::vector<std::vector<std::size_t>> elements; ///< 单元连接性
    MeshType meshType;                      ///< 网格类型
    std::vector<BoundarySegment> boundaries; ///< 边界段
    std::map<std::string, std::vector<double>> nodeData; ///< 节点关联数据
    std::map<std::string, std::vector<double>> elementData; ///< 单元关联数据
    MeshQualityMetrics qualityMetrics;      ///< 网格质量指标
};

struct FieldDefinition {
    std::string fieldName;                  ///< 场名称
    FieldType fieldType;                    ///< 场类型
    FieldLocation fieldLocation;           ///< 场位置
    InitializationMethod initMethod;        ///< 初始化方法
    std::map<std::string, double> parameters; ///< 初始化参数
    std::optional<GridData> initialData;    ///< 来自栅格的初始数据
    std::string units;                      ///< 场单位
};

struct SpatialField {
    std::string fieldName;                  ///< 场名称
    std::vector<double> values;             ///< 节点/单元处的场值
    FieldType fieldType;                    ///< 场类型（标量、矢量、张量）
    FieldLocation fieldLocation;           ///< 场位置（节点、单元、面）
    std::string units;                      ///< 场单位
    std::optional<double> noDataValue;      ///< 无数据值
    std::chrono::system_clock::time_point timestamp; ///< 场时间戳
    std::map<std::string, std::string> metadata; ///< 场元数据
};

struct VectorField {
    std::string fieldName;                  ///< 场名称
    std::vector<std::vector<double>> vectors; ///< 每个位置的矢量分量
    std::size_t dimensions;                 ///< 维数（2D或3D）
    FieldLocation fieldLocation;           ///< 场位置
    std::string units;                      ///< 场单位
    std::map<std::string, std::string> metadata; ///< 场元数据
};

struct SpatialStatistics {
    double mean;                            ///< 平均值
    double variance;                        ///< 方差
    double standardDeviation;               ///< 标准差
    double minimum;                         ///< 最小值
    double maximum;                         ///< 最大值
    double spatialAutocorrelation;          ///< 空间自相关（Moran's I）
    std::vector<double> percentiles;        ///< 百分位值
    std::map<std::string, double> customMetrics; ///< 自定义统计指标
};

struct SpatialBoundaryCondition {
    std::string boundaryId;                 ///< 边界标识符
    BoundaryConditionType conditionType;   ///< 边界条件类型
    std::vector<double> values;             ///< 边界条件值
    std::optional<std::string> expression;  ///< 边界条件的数学表达式
    std::map<std::string, double> parameters; ///< 边界条件参数
};

struct RefinementCriterion {
    RefinementType refinementType;          ///< 细化类型
    double threshold;                       ///< 细化阈值
    std::optional<Geometry> region;         ///< 细化区域
    std::map<std::string, double> parameters; ///< 细化参数
};

struct ModelingSpatialSupportConfig {
    double defaultResolution = 100.0;       ///< 默认空间分辨率
    MeshType defaultMeshType = MeshType::TRIANGULAR; ///< 默认网格类型
    bool enableMeshValidation = true;       ///< 启用网格验证
    bool enableParallelProcessing = true;   ///< 启用并行处理
    std::size_t maxThreads = 0;             ///< 最大线程数（0 = 自动）
    bool enableCaching = true;              ///< 启用结果缓存
    std::size_t cacheSize = 1000;           ///< 缓存大小
    double qualityThreshold = 0.8;          ///< 验证的质量阈值
    std::map<std::string, std::string> customSettings; ///< 自定义设置
};

// 简化的选项结构体
struct BoundaryExtractionOptions {};
struct BoundaryApplicationOptions {};
struct FieldInitializationOptions {};
struct FieldInterpolationOptions {};
struct GradientOptions {};
struct OperatorParameters {};
struct CorrelationOptions {};
struct CorrelationAnalysisResult {};
struct PatternDetectionParameters {};
struct PatternDetectionResult {};
struct SamplingOptions {};
struct AdequacyCriteria {};
struct SamplingAdequacyResult {};
struct MonitoringResult {};
struct ObjectiveFunction {};
struct PerturbationParameters {};
struct SensitivityAnalysisResult {};
struct SpatialConstraint {};
struct SpatialConstraintMatrices {};
struct MeshRefinementOptions {};
struct MeshValidationResult {};

/**
 * @brief Interface for spatial support to modeling service
 * 
 * Provides spatial operations specifically designed to support
 * environmental modeling, simulation, and analysis workflows.
 */
class IModelingSpatialSupport {
public:
    virtual ~IModelingSpatialSupport() = default;

    // --- Spatial Domain Operations ---

    /**
     * @brief Create computational domain from geometry
     * @param domainGeometry Geometry defining the computational domain
     * @param resolution Spatial resolution for discretization
     * @param options Domain creation options
     * @return Future containing computational domain definition
     */
    virtual std::future<ComputationalDomain> createComputationalDomain(
        const Geometry& domainGeometry,
        double resolution,
        const DomainCreationOptions& options = {}) const = 0;

    /**
     * @brief Generate mesh for computational domain
     * @param domain Computational domain
     * @param meshType Type of mesh to generate
     * @param options Mesh generation options
     * @return Future containing generated mesh
     */
    virtual std::future<SpatialMesh> generateSpatialMesh(
        const ComputationalDomain& domain,
        MeshType meshType,
        const MeshGenerationOptions& options = {}) const = 0;

    /**
     * @brief Refine mesh based on spatial criteria
     * @param mesh Input mesh
     * @param refinementCriteria Criteria for mesh refinement
     * @param options Refinement options
     * @return Future containing refined mesh
     */
    virtual std::future<SpatialMesh> refineMesh(
        const SpatialMesh& mesh,
        const std::vector<RefinementCriterion>& refinementCriteria,
        const MeshRefinementOptions& options = {}) const = 0;

    /**
     * @brief Validate mesh quality and topology
     * @param mesh Mesh to validate
     * @param qualityMetrics Quality metrics to check
     * @return Future containing mesh validation results
     */
    virtual std::future<MeshValidationResult> validateMesh(
        const SpatialMesh& mesh,
        const std::vector<MeshQualityMetric>& qualityMetrics) const = 0;

    // --- Boundary Condition Support ---

    /**
     * @brief Extract boundary segments from domain
     * @param domain Computational domain
     * @param boundaryType Type of boundary to extract
     * @param options Boundary extraction options
     * @return Future containing boundary segments
     */
    virtual std::future<std::vector<BoundarySegment>> extractBoundarySegments(
        const ComputationalDomain& domain,
        BoundaryType boundaryType,
        const BoundaryExtractionOptions& options = {}) const = 0;

    /**
     * @brief Apply spatial boundary conditions
     * @param mesh Computational mesh
     * @param boundaryConditions Boundary conditions to apply
     * @param options Application options
     * @return Future containing mesh with applied boundary conditions
     */
    virtual std::future<SpatialMesh> applyBoundaryConditions(
        const SpatialMesh& mesh,
        const std::vector<SpatialBoundaryCondition>& boundaryConditions,
        const BoundaryApplicationOptions& options = {}) const = 0;

    /**
     * @brief Interpolate boundary values using spatial data
     * ❌ 注意：插值功能应该通过工作流引擎调用插值服务，空间服务不直接提供插值
     * @param boundarySegments Boundary segments to interpolate
     * @param spatialData Source spatial data
     * @return Future containing interpolated boundary values
     */
    virtual std::future<std::vector<double>> interpolateBoundaryValues(
        const std::vector<BoundarySegment>& boundarySegments,
        const GridData& spatialData) const = 0;

    // --- Spatial Field Operations ---

    /**
     * @brief Initialize spatial field on mesh
     * @param mesh Computational mesh
     * @param fieldDefinition Field definition and initial conditions
     * @param options Field initialization options
     * @return Future containing initialized spatial field
     */
    virtual std::future<SpatialField> initializeSpatialField(
        const SpatialMesh& mesh,
        const FieldDefinition& fieldDefinition,
        const FieldInitializationOptions& options = {}) const = 0;

    /**
     * @brief Interpolate field values between meshes
     * ❌ 注意：复杂插值功能应该通过工作流引擎调用插值服务
     * @param sourceField Source spatial field
     * @param targetMesh Target mesh
     * @param options Interpolation options
     * @return Future containing interpolated field on target mesh
     */
    virtual std::future<SpatialField> interpolateFieldBetweenMeshes(
        const SpatialField& sourceField,
        const SpatialMesh& targetMesh,
        const FieldInterpolationOptions& options = {}) const = 0;

    /**
     * @brief Calculate spatial gradients of field
     * @param field Spatial field
     * @param gradientMethod Method for gradient calculation
     * @param options Gradient calculation options
     * @return Future containing gradient field
     */
    virtual std::future<VectorField> calculateSpatialGradients(
        const SpatialField& field,
        GradientCalculationMethod gradientMethod,
        const GradientOptions& options = {}) const = 0;

    /**
     * @brief Apply spatial operators to field
     * @param field Input spatial field
     * @param spatialOperator Spatial operator to apply
     * @param operatorParameters Operator-specific parameters
     * @return Future containing result field
     */
    virtual std::future<SpatialField> applySpatialOperator(
        const SpatialField& field,
        SpatialOperatorType spatialOperator,
        const OperatorParameters& operatorParameters = {}) const = 0;

    // --- Spatial Statistics and Analysis ---

    /**
     * @brief Calculate spatial statistics for field
     * @param field Spatial field
     * @param statisticsType Type of statistics to calculate
     * @param analysisRegion Optional region for analysis
     * @return Future containing spatial statistics
     */
    virtual std::future<SpatialStatistics> calculateSpatialStatistics(
        const SpatialField& field,
        SpatialStatisticsType statisticsType,
        const std::optional<Geometry>& analysisRegion = std::nullopt) const = 0;

    /**
     * @brief Perform spatial correlation analysis
     * @param field1 First spatial field
     * @param field2 Second spatial field
     * @param correlationMethod Correlation analysis method
     * @param options Analysis options
     * @return Future containing correlation analysis results
     */
    virtual std::future<CorrelationAnalysisResult> performSpatialCorrelation(
        const SpatialField& field1,
        const SpatialField& field2,
        CorrelationMethod correlationMethod,
        const CorrelationOptions& options = {}) const = 0;

    /**
     * @brief Detect spatial patterns in field
     * @param field Spatial field
     * @param patternType Type of pattern to detect
     * @param detectionParameters Pattern detection parameters
     * @return Future containing pattern detection results
     */
    virtual std::future<PatternDetectionResult> detectSpatialPatterns(
        const SpatialField& field,
        SpatialPatternType patternType,
        const PatternDetectionParameters& detectionParameters = {}) const = 0;

    // --- Spatial Sampling and Monitoring ---

    /**
     * @brief Generate optimal sampling locations
     * @param domain Computational domain
     * @param samplingStrategy Sampling strategy
     * @param numberOfSamples Number of samples to generate
     * @param options Sampling options
     * @return Future containing optimal sampling locations
     */
    virtual std::future<std::vector<Point>> generateOptimalSamplingLocations(
        const ComputationalDomain& domain,
        SamplingStrategy samplingStrategy,
        std::size_t numberOfSamples,
        const SamplingOptions& options = {}) const = 0;

    /**
     * @brief Evaluate sampling network adequacy
     * @param existingSamples Existing sampling locations
     * @param domain Computational domain
     * @param adequacyCriteria Adequacy evaluation criteria
     * @return Future containing adequacy evaluation results
     */
    virtual std::future<SamplingAdequacyResult> evaluateSamplingAdequacy(
        const std::vector<Point>& existingSamples,
        const ComputationalDomain& domain,
        const AdequacyCriteria& adequacyCriteria) const = 0;

    /**
     * @brief Extract field values at monitoring points
     * @param field Spatial field
     * @param monitoringPoints Monitoring point locations
     * @param extractionMethod Value extraction method
     * @return Future containing extracted values with metadata
     */
    virtual std::future<std::vector<MonitoringResult>> extractFieldAtPoints(
        const SpatialField& field,
        const std::vector<Point>& monitoringPoints,
        FieldExtractionMethod extractionMethod = FieldExtractionMethod::INTERPOLATED) const = 0;

    // --- Spatial Optimization Support ---

    /**
     * @brief Calculate spatial objective function
     * @param field Spatial field
     * @param objectiveFunction Objective function definition
     * @param constraintRegions Optional constraint regions
     * @return Future containing objective function value
     */
    virtual std::future<double> calculateSpatialObjectiveFunction(
        const SpatialField& field,
        const ObjectiveFunction& objectiveFunction,
        const std::vector<Geometry>& constraintRegions = {}) const = 0;

    /**
     * @brief Perform spatial sensitivity analysis
     * @param baseField Base spatial field
     * @param perturbationParameters Parameters for field perturbation
     * @param sensitivityMetrics Metrics for sensitivity analysis
     * @return Future containing sensitivity analysis results
     */
    virtual std::future<SensitivityAnalysisResult> performSensitivityAnalysis(
        const SpatialField& baseField,
        const PerturbationParameters& perturbationParameters,
        const std::vector<SensitivityMetric>& sensitivityMetrics) const = 0;

    /**
     * @brief Generate spatial constraint matrices for optimization
     * @param mesh Computational mesh
     * @param constraints Spatial constraints
     * @param matrixFormat Format of the output matrices
     * @return Future containing constraint matrices
     */
    virtual std::future<SpatialConstraintMatrices> generateConstraintMatrices(
        const SpatialMesh& mesh,
        const std::vector<SpatialConstraint>& constraints,
        MatrixFormat matrixFormat = MatrixFormat::SPARSE) const = 0;

    // --- Configuration ---

    /**
     * @brief Set the configuration for modeling spatial support
     * @param config Configuration settings
     */
    virtual void setConfiguration(const ModelingSpatialSupportConfig& config) = 0;

    /**
     * @brief Get the current configuration
     * @return Current configuration settings
     */
    virtual ModelingSpatialSupportConfig getConfiguration() const = 0;

    /**
     * @brief Get the capabilities of this support module
     * @return List of capability strings
     */
    virtual std::vector<std::string> getCapabilities() const = 0;
};

} // namespace oscean::core_services::spatial_ops::algorithms
 