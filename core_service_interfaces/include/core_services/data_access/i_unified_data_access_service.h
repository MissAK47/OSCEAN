/**
 * @file i_unified_data_access_service.h
 * @brief 统一数据访问服务接口 - 阶段1：统一对外接口架构
 * 
 * 🎯 重构目标：
 * ✅ 统一对外接口 - 外部只需要这一个接口
 * ✅ 简化API设计 - 10个核心方法替代50+个分散方法
 * ✅ 统一请求响应 - 避免多套重复的请求响应类型
 * ✅ 移除CRS依赖 - 只返回原生CRS信息
 * ✅ 异步优先设计 - 所有操作都是异步的
 * 🆕 真正的流式处理 - 支持大文件分块读取
 * 🆕 垂直剖面读取 - 支持海洋3D数据查询
 * 🆕 点查询功能 - 支持指定坐标点数据查询
 */

#pragma once

#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <chrono>
#include "core_services/common_data_types.h"
#include "unified_data_types.h"  // 包含完整的类型定义

namespace oscean::core_services::data_access {

/**
 * @brief 🆕 流式数据处理器接口
 */
class IStreamProcessor {
public:
    virtual ~IStreamProcessor() = default;
    
    /**
     * @brief 处理数据块
     * @param chunk 数据块
     * @param chunkInfo 块信息（索引、形状等）
     * @return true继续处理，false停止流式读取
     */
    virtual bool processChunk(
        const std::vector<double>& chunk, 
        const std::map<std::string, std::any>& chunkInfo) = 0;
        
    /**
     * @brief 流式处理完成回调
     */
    virtual void onStreamComplete() = 0;
    
    /**
     * @brief 流式处理错误回调
     */
    virtual void onStreamError(const std::string& error) = 0;
};

/**
 * @brief 🆕 大文件读取配置
 */
struct LargeFileReadConfig {
    size_t chunkSizeBytes = 64 * 1024 * 1024;  ///< 64MB 默认块大小
    size_t maxMemoryUsageBytes = 512 * 1024 * 1024;  ///< 512MB 最大内存使用
    bool enableProgressCallback = true;        ///< 启用进度回调
    bool enableMemoryOptimization = true;      ///< 启用内存优化
    bool enableParallelReading = false;        ///< 启用并行读取（谨慎使用）
};

/**
 * @brief 统一数据访问服务接口
 * 
 * 🎯 设计原则：
 * ✅ 单一对外接口 - 外部只需要这一个接口
 * ✅ 统一请求响应 - 避免多套重复的请求响应类型
 * ✅ 职责清晰 - 只负责数据读取，不处理CRS转换
 * ✅ 异步优先 - 所有操作都是异步的
 * ✅ 可扩展性 - 通过请求类型扩展新功能
 * 🆕 真正流式处理 - 支持大文件优化读取
 * 🆕 3D数据支持 - 支持垂直剖面等海洋数据
 */
class IUnifiedDataAccessService {
public:
    virtual ~IUnifiedDataAccessService() = default;

    // =============================================================================
    // 核心数据访问方法 - 统一接口
    // =============================================================================

    /**
     * @brief 统一数据访问方法 - 核心接口
     * 
     * 支持的请求类型：
     * - FILE_METADATA: 获取文件元数据
     * - GRID_DATA: 读取格点数据
     * - FEATURE_COLLECTION: 读取要素集合
     * - TIME_SERIES: 读取时间序列
     * - VERTICAL_PROFILE: 读取垂直剖面
     * - VARIABLE_ATTRIBUTES: 获取变量属性
     * - GLOBAL_ATTRIBUTES: 获取全局属性
     * - FIELD_DEFINITIONS: 获取字段定义
     * 🆕 POINT_QUERY: 点数据查询
     * 🆕 STREAMING_DATA: 流式数据读取
     * 
     * @param request 统一数据请求
     * @return 统一数据响应的future
     */
    virtual boost::future<api::UnifiedDataResponse> processDataRequestAsync(
        const api::UnifiedDataRequest& request) = 0;

    /**
     * @brief 批量数据访问方法
     * 
     * @param requests 批量请求列表
     * @return 批量响应的future
     */
    virtual boost::future<std::vector<api::UnifiedDataResponse>> processBatchRequestsAsync(
        const std::vector<api::UnifiedDataRequest>& requests) = 0;

    // =============================================================================
    // 便捷方法 - 简化常用操作
    // =============================================================================

    /**
     * @brief 快速获取文件元数据
     * 
     * @param filePath 文件路径
     * @return 文件元数据的future
     */
    virtual boost::future<std::optional<::oscean::core_services::FileMetadata>> getFileMetadataAsync(
        const std::string& filePath) = 0;

    /**
     * @brief 🔧 第三阶段：批量提取文件元数据
     * 
     * @param filePaths 文件路径列表
     * @param maxConcurrency 最大并发数（默认4）
     * @return 文件元数据列表的future
     */
    virtual boost::future<std::vector<::oscean::core_services::FileMetadata>> extractBatchMetadataAsync(
        const std::vector<std::string>& filePaths,
        size_t maxConcurrency = 4
    ) = 0;

    /**
     * @brief 快速读取格点数据
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param bounds 边界框（可选，原生坐标系）
     * @return 格点数据的future
     */
    virtual boost::future<std::shared_ptr<::oscean::core_services::GridData>> readGridDataAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::optional<::oscean::core_services::BoundingBox>& bounds = std::nullopt) = 0;

    /**
     * @brief 🆕 读取格点数据并支持坐标转换
     * 
     * 为工作流层提供便利，DataAccess会协调CRS服务进行坐标转换
     * 注意：坐标转换的具体实现由CRS服务负责，DataAccess只负责协调
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param bounds 边界框（目标坐标系）
     * @param targetCRS 目标坐标系（如 "EPSG:4326"）
     * @return 转换后的格点数据future
     */
    virtual boost::future<std::shared_ptr<::oscean::core_services::GridData>> readGridDataWithCRSAsync(
        const std::string& filePath,
        const std::string& variableName,
        const ::oscean::core_services::BoundingBox& bounds,
        const std::string& targetCRS) = 0;

    /**
     * @brief 🆕 读取点数据并支持坐标转换
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param point 目标点（目标坐标系）
     * @param targetCRS 目标坐标系（如 "EPSG:4326"）
     * @return 点数据值的future
     */
    virtual boost::future<std::optional<double>> readPointDataWithCRSAsync(
        const std::string& filePath,
        const std::string& variableName,
        const ::oscean::core_services::Point& point,
        const std::string& targetCRS) = 0;

    // =============================================================================
    // 🆕 3D数据和垂直剖面支持
    // =============================================================================

    /**
     * @brief 读取垂直剖面数据（如海洋密度剖面）
     * 
     * @param filePath 文件路径
     * @param variableName 变量名（如 "rho"）
     * @param longitude 经度
     * @param latitude 纬度
     * @param timePoint 时间点（可选）
     * @return 垂直剖面数据的future
     */
    virtual boost::future<std::shared_ptr<::oscean::core_services::VerticalProfileData>> readVerticalProfileAsync(
        const std::string& filePath,
        const std::string& variableName,
        double longitude,
        double latitude,
        const std::optional<std::chrono::system_clock::time_point>& timePoint = std::nullopt) = 0;

    /**
     * @brief 读取时间序列数据
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param longitude 经度
     * @param latitude 纬度
     * @param depth 深度（可选）
     * @param timeRange 时间范围（可选）
     * @return 时间序列数据的future
     */
    virtual boost::future<std::shared_ptr<::oscean::core_services::TimeSeriesData>> readTimeSeriesAsync(
        const std::string& filePath,
        const std::string& variableName,
        double longitude,
        double latitude,
        const std::optional<double>& depth = std::nullopt,
        const std::optional<std::pair<std::chrono::system_clock::time_point,
                                      std::chrono::system_clock::time_point>>& timeRange = std::nullopt) = 0;

    /**
     * @brief 读取指定点的数据值
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param longitude 经度
     * @param latitude 纬度
     * @param depth 深度（可选，用于3D数据）
     * @param timePoint 时间点（可选）
     * @return 点数据值的future
     */
    virtual boost::future<std::optional<double>> readPointValueAsync(
        const std::string& filePath,
        const std::string& variableName,
        double longitude,
        double latitude,
        const std::optional<double>& depth = std::nullopt,
        const std::optional<std::chrono::system_clock::time_point>& timePoint = std::nullopt) = 0;

    /**
     * @brief 获取垂直层信息
     * 
     * @param filePath 文件路径
     * @return 垂直层深度/高度的future
     */
    virtual boost::future<std::vector<double>> getVerticalLevelsAsync(
        const std::string& filePath) = 0;

    // =============================================================================
    // 🆕 真正的流式处理 - 大文件优化
    // =============================================================================

    /**
     * @brief 启动真正的流式数据读取
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param processor 流式数据处理器
     * @param config 大文件读取配置
     * @return 流式处理完成的future
     */
    virtual boost::future<void> startAdvancedStreamingAsync(
        const std::string& filePath,
        const std::string& variableName,
        std::shared_ptr<IStreamProcessor> processor,
        const LargeFileReadConfig& config = LargeFileReadConfig{}) = 0;

    /**
     * @brief 流式读取带边界限制的数据
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param bounds 空间边界
     * @param chunkProcessor 数据块处理器
     * @param progressCallback 进度回调（可选）
     * @return 流式处理完成的future
     */
    virtual boost::future<void> streamBoundedDataAsync(
        const std::string& filePath,
        const std::string& variableName,
        const ::oscean::core_services::BoundingBox& bounds,
        std::function<bool(const std::vector<double>&, const std::vector<size_t>&)> chunkProcessor,
        std::function<void(double)> progressCallback = nullptr) = 0;

    /**
     * @brief 内存优化的大文件读取
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param bounds 空间边界（可选）
     * @param config 大文件读取配置
     * @return 优化后的数据future
     */
    virtual boost::future<std::shared_ptr<::oscean::core_services::GridData>> readLargeFileOptimizedAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::optional<::oscean::core_services::BoundingBox>& bounds = std::nullopt,
        const LargeFileReadConfig& config = LargeFileReadConfig{}) = 0;

    // =============================================================================
    // 传统方法保持兼容性
    // =============================================================================

    /**
     * @brief 检查变量是否存在
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @return 是否存在的future
     */
    virtual boost::future<bool> checkVariableExistsAsync(
        const std::string& filePath,
        const std::string& variableName) = 0;

    /**
     * @brief 获取文件中的所有变量名
     * 
     * @param filePath 文件路径
     * @return 变量名列表的future
     */
    virtual boost::future<std::vector<std::string>> getVariableNamesAsync(
        const std::string& filePath) = 0;

    /**
     * @brief 启动流式数据读取（简化版本，保持兼容性）
     * 
     * @param filePath 文件路径
     * @param variableName 变量名
     * @param chunkProcessor 数据块处理器
     * @return 流式处理完成的future
     */
    virtual boost::future<void> startStreamingAsync(
        const std::string& filePath,
        const std::string& variableName,
        std::function<bool(const std::vector<double>&)> chunkProcessor) = 0;

    // =============================================================================
    // 性能监控和管理
    // =============================================================================

    /**
     * @brief 获取性能指标
     * 
     * @return 性能指标
     */
    virtual api::DataAccessMetrics getPerformanceMetrics() const = 0;

    /**
     * @brief 配置性能优化目标
     * 
     * @param targets 性能目标
     */
    virtual void configurePerformanceTargets(const api::DataAccessPerformanceTargets& targets) = 0;

    /**
     * @brief 清理缓存
     */
    virtual void clearCache() = 0;

    /**
     * @brief 获取服务健康状态
     * 
     * @return 是否健康
     */
    virtual bool isHealthy() const = 0;
};

} // namespace oscean::core_services::data_access 