#pragma once

/**
 * @file i_enhanced_data_workflow_service.h
 * @brief 增强的数据工作流服务接口 - 基于高级优化方案
 * @author OSCEAN Team
 * @date 2024
 */

#include "workflow_engine/data_workflow/i_data_workflow_service.h"
#include "workflow_engine/data_workflow/enhanced_workflow_types.h"
#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <functional>
#include <optional>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <atomic>

namespace oscean::workflow_engine::data_workflow {

/**
 * @brief 增强的数据工作流服务接口
 * 
 * 🎯 核心职责：
 * ✅ 服务编排 - 协调空间服务、数据访问服务、插值服务等
 * ✅ 策略选择 - 根据请求特征选择最优处理策略
 * ✅ 参数传递 - 将用户请求转换为各服务能理解的参数
 * ✅ 结果整合 - 将各服务的结果整合为最终输出
 * 
 * ❌ 不负责：
 * ❌ 空间计算 - 由空间服务完成
 * ❌ 数据读取优化 - 由数据访问服务内部完成
 * ❌ 插值计算 - 由插值服务完成
 */
class IEnhancedDataWorkflowService : public virtual IDataWorkflowService {
public:
    virtual ~IEnhancedDataWorkflowService() = default;

    // =============================================================================
    // 🎯 嵌套类型定义 - 必须在方法声明之前定义
    // =============================================================================

    /**
     * @brief 请求分析结果
     */
    struct RequestAnalysisResult {
        EnhancedDataWorkflowRequest validatedRequest;
        std::vector<std::string> warnings;
        std::vector<std::string> optimizationSuggestions;
        bool isValid;
    };

    /**
     * @brief 数据源发现结果
     */
    struct DataSourceDiscoveryResult {
        std::vector<std::string> matchedFiles;     // 匹配的文件列表
        std::map<std::string, std::string> fileMetadata; // 文件元数据
        std::vector<std::string> qualityAssessments; // 质量评估
        std::string recommendedProcessingOrder;    // 推荐处理顺序
    };

    /**
     * @brief 智能数据读取结果
     */
    struct IntelligentDataReadingResult {
        std::shared_ptr<oscean::core_services::GridData> gridData; // 读取的网格数据
        std::map<std::string, std::string> processingMetadata;     // 处理元数据
        std::vector<std::string> appliedOptimizations;            // 应用的优化
        IntelligentReadingStrategy::PerformanceExpectation actualPerformance; // 实际性能
    };

    /**
     * @brief 增强工作流服务配置
     */
    struct EnhancedWorkflowServiceConfig {
        // 性能配置
        size_t maxConcurrentWorkflows = 10;        // 最大并发工作流数
        size_t maxMemoryUsagePerWorkflowMB = 2048; // 每个工作流最大内存使用
        std::chrono::seconds defaultTimeout{300};  // 默认超时时间
        
        // 策略选择配置
        bool enableIntelligentStrategySelection = true; // 启用智能策略选择
        bool enablePerformancePrediction = true;        // 启用性能预测
        bool enableAdaptiveOptimization = true;         // 启用自适应优化
        
        // 缓存配置
        bool enableResultCaching = true;               // 启用结果缓存
        size_t maxCachedResults = 100;                 // 最大缓存结果数
        std::chrono::minutes cacheExpirationTime{60};  // 缓存过期时间
        
        // 监控配置
        bool enableDetailedLogging = true;             // 启用详细日志
        bool enablePerformanceMetrics = true;          // 启用性能指标
        bool enableResourceMonitoring = true;          // 启用资源监控
    };

    /**
     * @brief 服务性能指标
     */
    struct ServicePerformanceMetrics {
        size_t totalWorkflowsExecuted = 0;             // 总执行工作流数
        size_t successfulWorkflows = 0;                // 成功工作流数
        size_t failedWorkflows = 0;                    // 失败工作流数
        std::chrono::milliseconds averageExecutionTime{0}; // 平均执行时间
        double averageMemoryUsageMB = 0.0;             // 平均内存使用
        std::map<std::string, size_t> strategyUsageCount; // 策略使用统计
    };

    // =============================================================================
    // 🎯 增强的工作流执行接口
    // =============================================================================

    /**
     * @brief 执行增强的数据工作流
     * @param request 增强的工作流请求
     * @param context 执行上下文（可选）
     * @return 工作流结果的Future
     */
    virtual boost::future<WorkflowResult> executeEnhancedWorkflowAsync(
        const EnhancedDataWorkflowRequest& request,
        std::optional<WorkflowExecutionContext> context = std::nullopt) = 0;

    /**
     * @brief 智能策略选择 - 分析请求并选择最优处理策略
     * @param request 工作流请求
     * @param spatialMetadata 空间分析元数据
     * @param dataSourceResult 数据源发现结果
     * @return 智能读取策略
     */
    virtual boost::future<IntelligentReadingStrategy> selectOptimalStrategyAsync(
        const EnhancedDataWorkflowRequest& request,
        const EnhancedSpatialQueryMetadata& spatialMetadata,
        const DataSourceDiscoveryResult& dataSourceResult) = 0;

    /**
     * @brief 预估工作流性能 - 在执行前预估资源需求和执行时间
     * @param request 工作流请求
     * @return 性能预估结果
     */
    virtual boost::future<IntelligentReadingStrategy::PerformanceExpectation> estimatePerformanceAsync(
        const EnhancedDataWorkflowRequest& request) = 0;

    // =============================================================================
    // 🎯 分阶段执行接口 - 支持细粒度控制
    // =============================================================================

    /**
     * @brief 第一步：请求解析和验证
     * @param request 原始请求
     * @return 解析后的请求和验证结果
     */
    virtual boost::future<RequestAnalysisResult> analyzeRequestAsync(
        const EnhancedDataWorkflowRequest& request) = 0;

    /**
     * @brief 第二步：空间请求解析（此功能已移至EnhancedSpatialRequestResolver）
     *       此接口方法已废弃，将在未来移除。
     */
    // virtual boost::future<EnhancedSpatialQueryMetadata> analyzeSpatialRequestAsync(
    //     const EnhancedDataWorkflowRequest& request) = 0;

    /**
     * @brief 第三步：数据源发现（委托给元数据服务）
     * @param request 工作流请求
     * @param spatialResult 空间分析结果
     * @return 数据源发现结果
     */
    virtual boost::future<DataSourceDiscoveryResult> discoverDataSourcesAsync(
        const EnhancedDataWorkflowRequest& request,
        const EnhancedSpatialQueryMetadata& spatialMetadata) = 0;

    /**
     * @brief 第四步：智能数据读取（委托给数据访问服务）
     * @param request 工作流请求
     * @param strategy 选择的策略
     * @param dataSources 数据源信息
     * @param spatialMetadata 空间分析元数据
     * @return 数据读取结果
     */
    virtual boost::future<IntelligentDataReadingResult> executeIntelligentDataReadingAsync(
        const EnhancedDataWorkflowRequest& request,
        const IntelligentReadingStrategy& strategy,
        const DataSourceDiscoveryResult& dataSources,
        const EnhancedSpatialQueryMetadata& spatialMetadata) = 0;

    // =============================================================================
    // 🎯 工作流管理和监控接口
    // =============================================================================

    /**
     * @brief 获取工作流执行状态
     * @param executionId 执行ID
     * @return 执行状态
     */
    virtual boost::future<WorkflowStatus> getWorkflowStatusAsync(const std::string& executionId) = 0;

    /**
     * @brief 取消工作流执行
     * @param executionId 执行ID
     * @return 取消是否成功
     */
    virtual boost::future<bool> cancelWorkflowAsync(const std::string& executionId) = 0;

    /**
     * @brief 获取工作流执行统计
     * @param executionId 执行ID
     * @return 执行统计信息
     */
    virtual boost::future<WorkflowExecutionContext::ExecutionStats> getExecutionStatsAsync(
        const std::string& executionId) = 0;

    // =============================================================================
    // 🎯 向后兼容接口
    // =============================================================================

    /**
     * @brief 从传统请求执行工作流（向后兼容）
     * @param legacyRequest 传统工作流请求
     * @return 工作流结果
     */
    virtual boost::future<WorkflowResult> executeLegacyWorkflowAsync(
        const WorkflowRequest& legacyRequest) = 0;

    // =============================================================================
    // 🎯 配置和优化接口
    // =============================================================================

    /**
     * @brief 配置工作流服务
     * @param config 服务配置
     */
    virtual void configure(const EnhancedWorkflowServiceConfig& config) = 0;

    /**
     * @brief 获取服务性能指标
     * @return 性能指标
     */
    virtual ServicePerformanceMetrics getPerformanceMetrics() const = 0;

    /**
     * @brief 重置性能指标
     */
    virtual void resetPerformanceMetrics() = 0;
};

// SpatialAnalysisResult is now defined in enhanced_workflow_types.h

} // namespace oscean::workflow_engine::data_workflow 