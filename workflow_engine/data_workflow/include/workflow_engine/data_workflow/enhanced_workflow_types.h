#pragma once

/**
 * @file enhanced_workflow_types.h
 * @brief 增强的数据工作流类型定义 - 基于高级优化方案
 * @author OSCEAN Team
 * @date 2024
 */

#include "core_services/common_data_types.h"
#include "workflow_engine/data_workflow/data_workflow_types.h"
#include <variant>
#include <vector>
#include <string>
#include <optional>
#include <chrono>
#include <map>
#include <functional>

namespace oscean::workflow_engine::data_workflow {

// 复用common_data_types.h中的基础类型
using Point = oscean::core_services::Point;
using BoundingBox = oscean::core_services::BoundingBox;
using TimeRange = oscean::core_services::TimeRange;
using Geometry = oscean::core_services::Geometry;

/**
 * @brief 空间精度配置 - 支持水平和垂直精度的独立控制
 */
struct SpatialResolutionConfig {
    // 🎯 水平精度控制
    struct HorizontalResolution {
        enum class Mode {
            ORIGINAL,           // 使用数据原始精度
            SPECIFIED           // 使用指定精度
        } mode = Mode::ORIGINAL;
        
        std::optional<double> targetResolutionMeters;  // 目标水平精度（米）
        bool enableAnisotropicHandling = false;        // 启用各向异性处理
        std::optional<double> targetResolutionX;       // X方向精度（米）
        std::optional<double> targetResolutionY;       // Y方向精度（米）
    } horizontalResolution;
    
    // 🎯 垂直精度控制
    struct VerticalResolution {
        enum class Mode {
            ORIGINAL,           // 使用数据原始精度
            SPECIFIED,          // 使用指定精度
            ADAPTIVE            // 自适应精度（基于深度范围）
        } mode = Mode::ORIGINAL;
        
        std::optional<double> targetResolutionMeters;  // 目标垂直精度（米）
        std::optional<double> minResolutionMeters;     // 最小精度限制
        std::optional<double> maxResolutionMeters;     // 最大精度限制
        std::string depthUnit = "meters";              // 深度单位
        std::string depthPositive = "down";            // 深度正方向
    } verticalResolution;
    
    // 🎯 插值算法选择
    enum class InterpolationAlgorithm {
        AUTO,               // 自动选择
        NEAREST,            // 最近邻
        BILINEAR,           // 双线性
        BICUBIC,            // 双三次
        LANCZOS            // Lanczos
    } interpolationAlgorithm = InterpolationAlgorithm::AUTO;
    
    // 🎯 精度阈值设置
    struct ResolutionThresholds {
        double horizontalImprovementThreshold = 0.5;   // 水平精度提升阈值
        double verticalImprovementThreshold = 0.5;     // 垂直精度提升阈值
        double maxInterpolationRatio = 10.0;           // 最大插值比例
    } thresholds;
};

/**
 * @brief 增强的数据工作流请求 - 基于高级优化方案
 */
struct EnhancedDataWorkflowRequest {
    // =============================================================================
    // 🎯 空间查询请求（必需）
    // =============================================================================
    SpatialRequest spatialRequest;                  // 复用现有定义
    
    // =============================================================================
    // 🎯 数据源模式选择
    // =============================================================================
    enum class DataSourceMode {
        DIRECT_FILES,           // 用户直接指定文件
        DATABASE_QUERY          // 通过数据库查询获取文件
    } dataSourceMode = DataSourceMode::DATABASE_QUERY;
    
    // =============================================================================
    // 🎯 直接文件模式参数
    // =============================================================================
    struct DirectFileParams {
        struct FileSpec {
            std::string filePath;
            std::vector<std::string> variableNames;
            
            // 深度维度参数
            struct DepthDimension {
                std::optional<std::vector<double>> specificDepths;     // 指定深度（米）
                std::optional<double> minDepth, maxDepth;              // 深度范围（米）
                std::optional<std::vector<int>> layerIndices;          // 层索引
                std::string depthUnit = "meters";
                std::string depthPositive = "down";
            };
            std::optional<DepthDimension> depthDimension;
            
            // CRS处理配置
            struct CRSHandling {
                std::string userCRS = "EPSG:4326";                     // 用户坐标CRS
                std::optional<std::string> expectedDataCRS;            // 预期数据CRS
                bool enableAutoDetection = true;                       // 启用CRS自动检测
                bool enableTransformation = true;                      // 启用CRS转换
                std::string preferredOutputCRS = "EPSG:4326";         // 首选输出CRS
            } crsHandling;
            
            // 空间精度配置
            SpatialResolutionConfig spatialResolution;
            
            // 其他参数
            std::optional<TimeRange> expectedTimeRange;
            std::optional<std::string> preferredAccessPattern;
        };
        std::vector<FileSpec> fileSpecs;
    };
    std::optional<DirectFileParams> directFileParams;
    
    // =============================================================================
    // 🎯 数据库查询模式参数
    // =============================================================================
    struct DatabaseQueryParams {
        std::vector<std::string> variableNames;                      // 全局变量列表
        std::optional<TimeRange> timeRange;                         // 时间范围
        std::optional<std::string> datasetType;                     // 数据集类型
        std::optional<double> minQualityScore;                      // 最低质量分数
        
        // 深度查询参数
        std::optional<double> minDepth, maxDepth;                   // 深度范围过滤
        std::optional<std::vector<std::string>> depthLayers;        // 深度层名称
        
        // 空间精度需求
        SpatialResolutionConfig spatialResolution;
        
        // 高级查询选项
        bool enableGeohashOptimization = true;                      // 启用Geohash优化
        bool preferHighQuality = true;                              // 优先高质量数据
        int maxResults = 100;                                       // 最大结果数
    };
    std::optional<DatabaseQueryParams> databaseQueryParams;
    
    // =============================================================================
    // 🎯 全局处理选项
    // =============================================================================
    struct ProcessingOptions {
        // 坐标转换选项
        std::optional<std::string> targetCRS;                       // 最终输出CRS
        bool enableCrsTransformation = true;
        
        // 插值处理选项
        bool enableAdvancedInterpolation = true;
        std::optional<double> interpolationTolerance;               // 插值容差
        
        // 数据融合选项
        enum class FusionStrategy {
            MOSAIC,             // 镶嵌
            AVERAGE,            // 平均
            WEIGHTED_AVERAGE,   // 加权平均
            LATEST_PRIORITY     // 最新优先
        } fusionStrategy = FusionStrategy::MOSAIC;
        
        // 质量控制
        bool enableQualityCheck = true;
        double minDataQuality = 0.5;
    } processingOptions;
    
    // =============================================================================
    // 🎯 输出配置
    // =============================================================================
    struct OutputOptions {
        enum class Format {
            NETCDF,             // NetCDF格式
            GEOTIFF,            // GeoTIFF格式
            CSV,                // CSV格式
            JSON,               // JSON格式
            MEMORY_OBJECT       // 内存对象（用于链式处理）
        } format = Format::NETCDF;
        
        std::optional<std::string> outputPath;                      // 输出路径
        std::optional<int> maxFileSizeMB;                          // 最大文件大小
        bool enableCompression = true;                              // 启用压缩
        
        // 元数据输出
        bool includeMetadata = true;                                // 包含元数据
        bool includeProcessingHistory = true;                      // 包含处理历史
    };
    std::optional<OutputOptions> outputOptions;
    
    // =============================================================================
    // 🎯 工作流控制选项
    // =============================================================================
    struct WorkflowOptions {
        std::string workflowId;                                     // 工作流ID
        int priority = 5;                                          // 优先级(1-10)
        std::chrono::seconds timeout = std::chrono::seconds(300);  // 超时时间
        bool enableProgressCallback = true;                        // 启用进度回调
        bool enableErrorRecovery = true;                          // 启用错误恢复
        int maxRetries = 3;                                        // 最大重试次数
    } workflowOptions;
    
    /**
     * @brief 从现有WorkflowRequest转换
     */
    static EnhancedDataWorkflowRequest fromLegacyRequest(const WorkflowRequest& legacyRequest);
    
    /**
     * @brief 转换为现有WorkflowRequest（向后兼容）
     */
    WorkflowRequest toLegacyRequest() const;
    
    /**
     * @brief 验证请求有效性
     */
    bool isValid() const;
};

/**
 * @brief 智能读取策略选择结果
 */
struct IntelligentReadingStrategy {
    enum class AccessPattern {
        SEQUENTIAL_SCAN,        // 顺序扫描（大区域）
        RANDOM_ACCESS,          // 随机访问（点查询）
        CHUNKED_READING,        // 分块读取（中等区域）
        STREAMING_PROCESSING,   // 流式处理（超大文件）
        MEMORY_MAPPED,          // 内存映射（频繁访问）
        PARALLEL_READING        // 并行读取（多文件）
    } accessPattern;
    
    std::string strategyName;                       // 策略名称
    std::string selectionReasoning;                 // 选择理由
    
    // 性能优化配置
    struct PerformanceConfig {
        bool enableSIMD = true;                     // 启用SIMD优化
        bool enableCaching = true;                  // 启用智能缓存
        bool enableMemoryOptimization = true;       // 启用内存优化
        bool enableAsyncProcessing = true;          // 启用异步处理
        
        // 资源限制
        size_t maxMemoryUsageMB = 1024;            // 最大内存使用（MB）
        size_t maxConcurrentOperations = 8;        // 最大并发操作数
        double timeoutSeconds = 300.0;             // 超时时间（秒）
        
        // 流式处理配置
        struct StreamingConfig {
            size_t chunkSizeMB = 64;               // 数据块大小（MB）
            size_t maxConcurrentChunks = 4;        // 最大并发块数
            bool enableBackpressure = true;        // 启用背压控制
        } streamingConfig;
    } performanceConfig;
    
    // 预期性能指标
    struct PerformanceExpectation {
        double estimatedProcessingTimeSeconds;      // 预估处理时间
        double estimatedMemoryUsageMB;             // 预估内存使用
        double estimatedIOOperations;              // 预估IO操作数
        double confidenceLevel;                    // 预估置信度
    } performanceExpectation;
    
    // 风险评估
    std::vector<std::string> potentialRisks;       // 潜在风险
    std::vector<std::string> mitigationStrategies; // 缓解策略
};

/**
 * @brief 工作流执行上下文
 */
struct WorkflowExecutionContext {
    std::string executionId;                        // 执行ID
    std::chrono::system_clock::time_point startTime; // 开始时间
    
    // 进度回调
    std::function<void(double progress, const std::string& status)> progressCallback;
    
    // 取消控制
    std::shared_ptr<std::atomic<bool>> cancellationToken;
    
    // 执行统计
    struct ExecutionStats {
        std::chrono::milliseconds totalTime{0};
        std::chrono::milliseconds spatialResolutionTime{0};
        std::chrono::milliseconds dataSourceDiscoveryTime{0};
        std::chrono::milliseconds dataReadingTime{0};
        std::chrono::milliseconds dataProcessingTime{0};
        std::chrono::milliseconds outputGenerationTime{0};
        
        size_t bytesRead = 0;
        size_t dataPointsProcessed = 0;
        size_t filesProcessed = 0;
        
        std::vector<std::string> optimizationsUsed;
    } executionStats;
};

// =============================================================================
// 🎯 增强的空间分析结果
// =============================================================================

/**
 * @brief Defines the target grid for data processing, resolved from a spatial request.
 */
struct GridDefinition {
    BoundingBox targetBounds;           ///< The final bounding box in the target CRS.
    std::string targetCRS;              ///< The target Coordinate Reference System (e.g., "EPSG:4326").
    size_t width;                       ///< Grid width in pixels.
    size_t height;                      ///< Grid height in pixels.
    double xResolution;                 ///< Horizontal resolution in target CRS units.
    double yResolution;                 ///< Vertical resolution in target CRS units.
    bool isTransformed;                 ///< True if the grid required CRS transformation.
};

/**
 * @brief Metadata and results generated from resolving a spatial request.
 * This is a more advanced version of SpatialAnalysisResult.
 */
struct EnhancedSpatialQueryMetadata {
    GridDefinition gridDefinition;      ///< The resolved target grid definition.
    BoundingBox originalRequestBounds;  ///< Bounding box of the original request in its original CRS.
    std::string originalRequestCRS;     ///< CRS of the original request.
    double spatialComplexity;           ///< Calculated complexity score (0.0 to 1.0).
    std::string recommendedAccessPattern; ///< Recommended data access pattern (e.g., "streaming", "chunked").
    std::vector<std::string> warnings;  ///< Any warnings generated during resolution.
};

/**
 * @brief 🎯 修正工作流的新数据结构 - 基于正确的业务逻辑
 */

/**
 * @brief CRS转换结果
 */
struct CRSTransformationResult {
    bool needsTransformation = false;           // 是否需要坐标转换
    std::string sourceCRS;                      // 源坐标系
    std::string targetCRS;                      // 目标坐标系
    std::string transformationPipeline;         // PROJ转换管道
    double transformationAccuracy = 0.0;        // 转换精度（米）
    std::vector<std::string> warnings;         // 转换警告
    
    // 转换后的几何
    std::optional<Point> transformedPoint;      // 转换后的点（如果适用）
    std::optional<BoundingBox> transformedBounds; // 转换后的边界框
};

/**
 * @brief 空间分析结果
 */
struct SpatialAnalysisResult {
    // 空间分辨率信息
    struct ResolutionInfo {
        double actualHorizontalResolution = 0.0;   // 实际水平分辨率（米）
        double actualVerticalResolution = 0.0;     // 实际垂直分辨率（米）
        double targetHorizontalResolution = 0.0;   // 目标水平分辨率（米）
        double targetVerticalResolution = 0.0;     // 目标垂直分辨率（米）
        bool needsInterpolation = false;           // 是否需要插值
        double interpolationRatio = 1.0;           // 插值比例
    } resolutionInfo;
    
    // 空间子集信息
    struct SubsetInfo {
        BoundingBox calculatedSubset;              // 计算的空间子集
        double subsetArea = 0.0;                   // 子集面积（平方米）
        size_t estimatedGridPoints = 0;            // 预估网格点数
        std::string subsetStrategy;                // 子集策略
    } subsetInfo;
    
    // 网格配置
    struct GridConfig {
        size_t gridWidth = 0;                      // 网格宽度
        size_t gridHeight = 0;                     // 网格高度
        size_t gridDepth = 1;                      // 网格深度（层数）
        double gridSpacingX = 0.0;                 // X方向网格间距
        double gridSpacingY = 0.0;                 // Y方向网格间距
        std::vector<double> depthLayers;           // 深度层定义
    } gridConfig;
    
    std::vector<std::string> analysisWarnings;    // 分析警告
};

/**
 * @brief 智能读取决策结果
 */
struct IntelligentReadingDecision {
    // 读取模式决策
    enum class ReadingMode {
        POINT_INTERPOLATION,        // 点插值读取
        GRID_EXTRACTION,           // 网格提取读取
        STREAMING_PROCESSING,      // 流式处理读取
        CHUNKED_READING,          // 分块读取
        MEMORY_MAPPED_ACCESS      // 内存映射访问
    } readingMode;
    
    // 插值决策（如果需要）
    struct InterpolationDecision {
        bool enableInterpolation = false;          // 是否启用插值
        SpatialResolutionConfig::InterpolationAlgorithm algorithm; // 插值算法
        double searchRadius = 5000.0;              // 搜索半径（米）
        int maxSearchPoints = 4;                   // 最大搜索点数
        std::string decisionReasoning;             // 决策理由
    } interpolationDecision;
    
    // 性能优化决策
    struct PerformanceDecision {
        bool enableSIMD = true;                    // 启用SIMD优化
        bool enableCaching = true;                 // 启用缓存
        bool enableParallelProcessing = true;      // 启用并行处理
        size_t recommendedChunkSize = 64;          // 推荐块大小（MB）
        size_t recommendedConcurrency = 4;         // 推荐并发数
    } performanceDecision;
    
    // 数据质量决策
    struct QualityDecision {
        bool enableQualityCheck = true;            // 启用质量检查
        double minAcceptableQuality = 0.5;         // 最低可接受质量
        bool skipInvalidData = true;               // 跳过无效数据
        std::string qualityStrategy;               // 质量策略
    } qualityDecision;
    
    std::string decisionSummary;                   // 决策摘要
    std::vector<std::string> decisionReasons;      // 决策原因列表
    double confidenceLevel = 0.8;                 // 决策置信度
};

} // namespace oscean::workflow_engine::data_workflow 