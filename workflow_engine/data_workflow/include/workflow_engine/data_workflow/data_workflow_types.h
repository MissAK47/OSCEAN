#pragma once

/**
 * @file data_workflow_types.h
 * @brief Data types specific to the intelligent data processing workflow
 * @author OSCEAN Team
 * @date 2024
 */

#include "core_services/common_data_types.h"
#include <variant>
#include <vector>
#include <string>
#include <optional>
#include <chrono>
#include <sstream>
#include <map>

namespace oscean::workflow_engine::data_workflow {

// 使用core_services中已定义的基础类型，避免重复定义
using Point = oscean::core_services::Point;
using BoundingBox = oscean::core_services::BoundingBox;
using Geometry = oscean::core_services::Geometry;
using TimeRange = oscean::core_services::TimeRange;

/**
 * @brief 线串几何类型
 * 扩展基础Point类型为线串，重用已有数据结构
 */
struct LineString {
    std::vector<Point> points;                    ///< 构成线串的点集合
    std::optional<std::string> crsId;             ///< 坐标参考系统标识符
    
    LineString() = default;
    LineString(std::vector<Point> pts, std::optional<std::string> crs = std::nullopt)
        : points(std::move(pts)), crsId(std::move(crs)) {}
        
    /**
     * @brief 转换为WKT格式几何对象
     */
    Geometry toGeometry() const {
        Geometry geom(Geometry::Type::LINESTRING);
        std::stringstream wkt;
        wkt << "LINESTRING (";
        for (size_t i = 0; i < points.size(); ++i) {
            wkt << points[i].x << " " << points[i].y;
            if (points[i].z) wkt << " " << *points[i].z;
            if (i < points.size() - 1) wkt << ", ";
        }
        wkt << ")";
        geom.wkt = wkt.str();
        return geom;
    }
};

/**
 * @brief 多边形几何类型
 * 扩展基础类型为多边形，重用已有数据结构
 */
struct Polygon {
    LineString shell;                             ///< 外环
    std::vector<LineString> holes;                ///< 内环（洞）
    std::optional<std::string> crsId;             ///< 坐标参考系统标识符
    
    Polygon() = default;
    Polygon(LineString outer_ring, std::optional<std::string> crs = std::nullopt)
        : shell(std::move(outer_ring)), crsId(std::move(crs)) {}
        
    /**
     * @brief 转换为WKT格式几何对象
     */
    Geometry toGeometry() const {
        Geometry geom(Geometry::Type::POLYGON);
        std::stringstream wkt;
        wkt << "POLYGON ((";
        for (size_t i = 0; i < shell.points.size(); ++i) {
            wkt << shell.points[i].x << " " << shell.points[i].y;
            if (shell.points[i].z) wkt << " " << *shell.points[i].z;
            if (i < shell.points.size() - 1) wkt << ", ";
        }
        wkt << "))";
        geom.wkt = wkt.str();
        return geom;
    }
};

/**
 * @brief 方位角距离请求类型
 */
struct BearingDistanceRequest {
    Point startPoint;                             ///< 起始点
    double bearing;                               ///< 方位角（度）
    double distance;                              ///< 距离（米）
    std::optional<std::string> crsId;             ///< 坐标参考系统标识符
    
    BearingDistanceRequest() = default;
    BearingDistanceRequest(Point start, double bear, double dist, std::optional<std::string> crs = std::nullopt)
        : startPoint(std::move(start)), bearing(bear), distance(dist), crsId(std::move(crs)) {}
};

/**
 * @brief 空间请求类型变体
 * 统一不同空间查询类型
 */
using SpatialRequest = std::variant<
    Point,
    LineString,
    Polygon,
    BoundingBox,
    BearingDistanceRequest
>;

/**
 * @brief 工作流状态枚举
 */
enum class WorkflowStatus {
    NOT_STARTED,                                  ///< 未开始
    INITIALIZING,                                 ///< 初始化中
    RESOLVING_SPATIAL_REQUEST,                    ///< 解析空间请求
    FINDING_DATA_SOURCES,                         ///< 查找数据源
    PROCESSING_DATA_SOURCES,                      ///< 处理数据源
    FUSING_DATA,                                  ///< 数据融合
    POST_PROCESSING,                              ///< 后处理
    COMPLETED,                                    ///< 完成
    COMPLETED_EMPTY,                              ///< 完成但无结果
    FAILED,                                       ///< 失败
    CANCELLED                                     ///< 已取消
};

/**
 * @brief 输出格式枚举
 */
enum class OutputFormat {
    NETCDF,                                       ///< NetCDF格式
    GEOTIFF,                                      ///< GeoTIFF格式
    TEXT,                                         ///< 文本格式
    JSON,                                         ///< JSON格式
    BINARY                                        ///< 二进制格式
};

/**
 * @brief 处理选项结构
 */
struct ProcessingOptions {
    std::optional<double> targetSpatialResolution;  ///< 目标空间分辨率（米）
    std::optional<std::string> targetCRS;           ///< 目标坐标参考系统
    bool enableAdvancedInterpolation = false;       ///< 启用高级插值算法
    bool enableQualityControl = true;               ///< 启用质量控制
    double qualityThreshold = 0.8;                  ///< 质量阈值
    std::optional<int> maxConcurrentJobs;           ///< 最大并发作业数
};

/**
 * @brief 输出选项结构
 */
struct OutputOptions {
    OutputFormat format = OutputFormat::NETCDF;     ///< 输出格式
    std::string outputPath;                         ///< 输出路径
    std::optional<int> maxFileSizeMB;               ///< 最大文件大小（MB）
    bool compressOutput = true;                     ///< 是否压缩输出
    std::optional<std::vector<std::string>> metadata; ///< 附加元数据
};

/**
 * @brief 处理模式枚举
 */
enum class ProcessingMode {
    DATABASE_QUERY,                               ///< 数据库查询模式（默认）
    DIRECT_FILES                                  ///< 直接文件模式
};

/**
 * @brief 维度范围选择结构
 */
struct DimensionSelection {
    std::string dimensionName;                    ///< 维度名称（如"depth", "height", "time", "level"）
    std::optional<std::pair<double, double>> valueRange; ///< 值范围（实际物理值，如深度米数）
    std::optional<std::pair<size_t, size_t>> indexRange; ///< 索引范围（数组索引）
    std::optional<std::vector<double>> specificValues;   ///< 指定特定值
    std::optional<std::vector<size_t>> specificIndices;  ///< 指定特定索引
    
    DimensionSelection() = default;
    DimensionSelection(std::string name) : dimensionName(std::move(name)) {}
    
    /**
     * @brief 设置深度范围（米）
     */
    static DimensionSelection createDepthRange(double minDepth, double maxDepth) {
        DimensionSelection sel("depth");
        sel.valueRange = std::make_pair(minDepth, maxDepth);
        return sel;
    }
    
    /**
     * @brief 设置高度范围（米）
     */
    static DimensionSelection createHeightRange(double minHeight, double maxHeight) {
        DimensionSelection sel("height");
        sel.valueRange = std::make_pair(minHeight, maxHeight);
        return sel;
    }
    
    /**
     * @brief 设置垂直层级范围
     */
    static DimensionSelection createLevelRange(size_t minLevel, size_t maxLevel) {
        DimensionSelection sel("level");
        sel.indexRange = std::make_pair(minLevel, maxLevel);
        return sel;
    }
    
    /**
     * @brief 设置特定深度值
     */
    static DimensionSelection createSpecificDepths(const std::vector<double>& depths) {
        DimensionSelection sel("depth");
        sel.specificValues = depths;
        return sel;
    }
};

/**
 * @brief 直接文件规范结构
 * 用于直接文件模式时指定文件和相关信息
 */
struct DirectFileSpec {
    std::string filePath;                         ///< 文件路径
    std::vector<std::string> variableNames;       ///< 该文件中要读取的变量名
    std::optional<BoundingBox> spatialBounds;     ///< 空间边界（可选，如不指定则读取全部）
    std::optional<TimeRange> timeRange;           ///< 时间范围（可选）
    std::optional<std::string> crsId;             ///< 坐标参考系统（可选）
    
    // 新增：维度选择支持
    std::vector<DimensionSelection> dimensionSelections; ///< 维度选择列表（深度、高度、层级等）
    bool readAllDepths = true;                    ///< 是否读取所有深度（默认true）
    
    DirectFileSpec() = default;
    DirectFileSpec(std::string path, std::vector<std::string> vars)
        : filePath(std::move(path)), variableNames(std::move(vars)) {}
        
    /**
     * @brief 添加深度选择
     */
    void addDepthRange(double minDepth, double maxDepth) {
        dimensionSelections.push_back(DimensionSelection::createDepthRange(minDepth, maxDepth));
        readAllDepths = false;
    }
    
    /**
     * @brief 添加高度选择
     */
    void addHeightRange(double minHeight, double maxHeight) {
        dimensionSelections.push_back(DimensionSelection::createHeightRange(minHeight, maxHeight));
    }
    
    /**
     * @brief 添加特定深度值
     */
    void addSpecificDepths(const std::vector<double>& depths) {
        dimensionSelections.push_back(DimensionSelection::createSpecificDepths(depths));
        readAllDepths = false;
    }
    
    /**
     * @brief 检查是否有深度/高度选择
     */
    bool hasDimensionSelections() const {
        return !dimensionSelections.empty();
    }
};

/**
 * @brief 多变量处理配置
 */
struct MultiVariableConfig {
    bool enableParallelReading = true;            ///< 启用并行读取
    bool fuseVariablesIntoSingleGrid = false;     ///< 将多变量融合到单一网格中
    std::optional<size_t> maxConcurrentVariables; ///< 最大并行变量数
    bool keepVariablesSeparate = true;            ///< 保持变量分离（默认）
};

/**
 * @brief 工作流请求结构
 */
struct WorkflowRequest {
    // 0. 处理模式选择（新增）
    ProcessingMode processingMode = ProcessingMode::DATABASE_QUERY;  ///< 处理模式
    
    // 1. 空间请求 (必需)：定义了要查询的空间位置和形态
    SpatialRequest spatialRequest;                  ///< 空间请求
    
    // 2a. 数据库查询模式的数据内容请求
    std::vector<std::string> variableNames;         ///< 变量名称列表（数据库查询模式）
    std::optional<TimeRange> timeRange;             ///< 时间范围（数据库查询模式）
    std::optional<std::vector<std::string>> dataSources;  ///< 指定的数据源（数据库查询模式）
    
    // 新增：数据库查询模式的维度选择
    std::vector<DimensionSelection> globalDimensionSelections; ///< 全局维度选择（应用于所有文件）
    bool readAllDepthsByDefault = true;             ///< 默认读取所有深度
    
    // 2b. 直接文件模式的数据规范（新增）
    std::vector<DirectFileSpec> directFiles;        ///< 直接指定的文件列表
    
    // 3. 多变量处理配置（新增）
    MultiVariableConfig multiVariableConfig;        ///< 多变量处理配置
    
    // 4. 处理选项 (可选)：定义了对提取数据的处理要求
    std::optional<ProcessingOptions> processingOptions; ///< 处理选项
    
    // 5. 输出选项 (可选)：定义了最终结果的输出形式
    std::optional<OutputOptions> outputOptions;     ///< 输出选项
    
    // 兼容性字段（保持向后兼容）
    std::string outputFormat = "netcdf";             ///< 输出格式（字符串）
    std::string outputPath;                          ///< 输出路径
    bool enableInterpolation = false;                ///< 是否启用插值
    bool enableCrsTransformation = true;             ///< 是否启用坐标转换
    std::optional<std::string> targetCrs;            ///< 目标坐标系统
    double tolerance = 0.001;                        ///< 容差值
    
    /**
     * @brief 验证请求有效性
     */
    bool isValid() const {
        if (processingMode == ProcessingMode::DATABASE_QUERY) {
            return !variableNames.empty();
        } else {
            return !directFiles.empty() && 
                   std::all_of(directFiles.begin(), directFiles.end(),
                       [](const DirectFileSpec& spec) {
                           return !spec.filePath.empty() && !spec.variableNames.empty();
                       });
        }
    }
    
    /**
     * @brief 获取所有要处理的变量名称
     */
    std::vector<std::string> getAllVariableNames() const {
        std::vector<std::string> allVars;
        
        if (processingMode == ProcessingMode::DATABASE_QUERY) {
            allVars = variableNames;
        } else {
            for (const auto& fileSpec : directFiles) {
                allVars.insert(allVars.end(), fileSpec.variableNames.begin(), fileSpec.variableNames.end());
            }
        }
        
        // 去重
        std::sort(allVars.begin(), allVars.end());
        allVars.erase(std::unique(allVars.begin(), allVars.end()), allVars.end());
        
        return allVars;
    }
    
    /**
     * @brief 添加全局深度范围选择（数据库查询模式）
     */
    void addGlobalDepthRange(double minDepth, double maxDepth) {
        globalDimensionSelections.push_back(DimensionSelection::createDepthRange(minDepth, maxDepth));
        readAllDepthsByDefault = false;
    }
    
    /**
     * @brief 添加全局高度范围选择（数据库查询模式）
     */
    void addGlobalHeightRange(double minHeight, double maxHeight) {
        globalDimensionSelections.push_back(DimensionSelection::createHeightRange(minHeight, maxHeight));
    }
    
    /**
     * @brief 添加全局层级范围选择（数据库查询模式）
     */
    void addGlobalLevelRange(size_t minLevel, size_t maxLevel) {
        globalDimensionSelections.push_back(DimensionSelection::createLevelRange(minLevel, maxLevel));
        readAllDepthsByDefault = false;
    }
    
    /**
     * @brief 添加全局特定深度选择（数据库查询模式）
     */
    void addGlobalSpecificDepths(const std::vector<double>& depths) {
        globalDimensionSelections.push_back(DimensionSelection::createSpecificDepths(depths));
        readAllDepthsByDefault = false;
    }
    
    /**
     * @brief 检查是否有维度选择
     */
    bool hasDimensionSelections() const {
        if (processingMode == ProcessingMode::DATABASE_QUERY) {
            return !globalDimensionSelections.empty();
        } else {
            return std::any_of(directFiles.begin(), directFiles.end(),
                             [](const DirectFileSpec& spec) { return spec.hasDimensionSelections(); });
        }
    }
    
    /**
     * @brief 获取维度选择摘要信息
     */
    std::string getDimensionSelectionSummary() const {
        std::stringstream summary;
        
        if (processingMode == ProcessingMode::DATABASE_QUERY) {
            if (globalDimensionSelections.empty()) {
                summary << "Global: Read all depths/levels";
            } else {
                summary << "Global: " << globalDimensionSelections.size() << " dimension selections";
                for (const auto& dimSel : globalDimensionSelections) {
                    summary << " [" << dimSel.dimensionName;
                    if (dimSel.valueRange.has_value()) {
                        summary << ":" << dimSel.valueRange->first << "-" << dimSel.valueRange->second;
                    } else if (dimSel.indexRange.has_value()) {
                        summary << ":idx:" << dimSel.indexRange->first << "-" << dimSel.indexRange->second;
                    }
                    summary << "]";
                }
            }
        } else {
            summary << "Per-file selections: ";
            for (const auto& fileSpec : directFiles) {
                if (fileSpec.hasDimensionSelections()) {
                    summary << fileSpec.dimensionSelections.size() << " dims for " << fileSpec.filePath << "; ";
                }
            }
        }
        
        return summary.str();
    }
    
    /**
     * @brief 获取有效的处理选项
     */
    ProcessingOptions getEffectiveProcessingOptions() const {
        if (processingOptions.has_value()) {
            return *processingOptions;
        }
        
        // 从兼容性字段构建处理选项
        ProcessingOptions opts;
        if (targetCrs.has_value()) {
            opts.targetCRS = *targetCrs;
        }
        opts.enableAdvancedInterpolation = enableInterpolation;
        return opts;
    }
    
    /**
     * @brief 获取有效的输出选项
     */
    OutputOptions getEffectiveOutputOptions() const {
        if (outputOptions.has_value()) {
            return *outputOptions;
        }
        
        // 从兼容性字段构建输出选项
        OutputOptions opts;
        opts.outputPath = outputPath;
        
        // 转换输出格式
        if (outputFormat == "netcdf" || outputFormat == "nc") {
            opts.format = OutputFormat::NETCDF;
        } else if (outputFormat == "geotiff" || outputFormat == "tiff") {
            opts.format = OutputFormat::GEOTIFF;
        } else if (outputFormat == "text" || outputFormat == "txt") {
            opts.format = OutputFormat::TEXT;
        } else if (outputFormat == "json") {
            opts.format = OutputFormat::JSON;
        } else {
            opts.format = OutputFormat::NETCDF; // 默认
        }
        
        return opts;
    }
};

/**
 * @brief 工作流结果结构
 */
struct WorkflowResult {
    bool success = false;                         ///< 是否成功
    WorkflowStatus status = WorkflowStatus::NOT_STARTED;  ///< 工作流状态
    std::string message;                          ///< 结果消息
    std::optional<std::string> error;             ///< 错误信息
    std::chrono::milliseconds duration{0};       ///< 执行时间

    // 统计信息
    size_t processedDataSources = 0;              ///< 处理的数据源数量
    size_t totalDataPoints = 0;                   ///< 总数据点数
    double dataVolumeMB = 0.0;                    ///< 数据体积（MB）

    // 输出信息
    std::optional<std::string> outputLocation;   ///< 输出位置
    std::optional<std::string> outputFormat;     ///< 输出格式
    
    // 🎯 数据访问：供应用层使用的实际数据
    std::shared_ptr<core_services::GridData> gridData; ///< 处理后的网格数据（供应用层访问）
    
    // 多变量处理结果（新增）
    std::vector<std::string> processedVariables; ///< 成功处理的变量列表
    std::vector<std::string> failedVariables;    ///< 处理失败的变量列表
    std::map<std::string, std::string> variableOutputPaths; ///< 每个变量的输出路径
    
    // 文件处理统计（新增）
    size_t totalFilesProcessed = 0;               ///< 总处理文件数
    size_t successfulFilesProcessed = 0;          ///< 成功处理文件数
    std::vector<std::string> failedFiles;        ///< 处理失败的文件列表
    
    /**
     * @brief 获取处理成功率
     */
    double getSuccessRate() const {
        if (totalFilesProcessed == 0) return 0.0;
        return static_cast<double>(successfulFilesProcessed) / totalFilesProcessed * 100.0;
    }
    
    /**
     * @brief 获取变量处理成功率
     */
    double getVariableSuccessRate() const {
        size_t totalVars = processedVariables.size() + failedVariables.size();
        if (totalVars == 0) return 0.0;
        return static_cast<double>(processedVariables.size()) / totalVars * 100.0;
    }
};

} // namespace oscean::workflow_engine::data_workflow 