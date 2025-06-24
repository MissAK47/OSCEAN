#pragma once

/**
 * @file enhanced_data_workflow_service_impl.h
 * @brief 增强数据工作流服务实现类头文件
 * @author OSCEAN Team
 * @date 2024
 */

#include "workflow_engine/data_workflow/i_enhanced_data_workflow_service.h"
#include "workflow_engine/data_workflow/data_workflow_service_impl.h"
#include "workflow_engine/data_workflow/enhanced_workflow_types.h"
#include "workflow_engine/data_workflow/enhanced_spatial_request_resolver.h"
#include "workflow_engine/service_management/i_service_manager.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "core_services/data_access/i_raw_data_access_service.h"
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/crs/i_crs_service.h"
#include <boost/thread/future.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <map>

namespace oscean::service_management {
    class IServiceManager;
}

// Forward declarations
namespace oscean::core_services {
    class IRawDataAccessService;
    class ISpatialOpsService;
    class ICrsService;
}

namespace oscean::workflow_engine::data_workflow {

class EnhancedSpatialRequestResolver;

// Type aliases for cleaner code
using RequestAnalysisResult = IEnhancedDataWorkflowService::RequestAnalysisResult;
using DataSourceDiscoveryResult = IEnhancedDataWorkflowService::DataSourceDiscoveryResult;
using IntelligentDataReadingResult = IEnhancedDataWorkflowService::IntelligentDataReadingResult;
using EnhancedWorkflowServiceConfig = IEnhancedDataWorkflowService::EnhancedWorkflowServiceConfig;
using ServicePerformanceMetrics = IEnhancedDataWorkflowService::ServicePerformanceMetrics;

/**
 * @class EnhancedDataWorkflowServiceImpl
 * @brief Implements the enhanced data workflow service, orchestrating complex data processing tasks.
 */
class EnhancedDataWorkflowServiceImpl : public DataWorkflowServiceImpl, public virtual IEnhancedDataWorkflowService {
public:
    explicit EnhancedDataWorkflowServiceImpl(std::shared_ptr<service_management::IServiceManager> serviceManager);
    ~EnhancedDataWorkflowServiceImpl() override;

    // IEnhancedDataWorkflowService interface implementation
    boost::future<WorkflowResult> executeEnhancedWorkflowAsync(
        const EnhancedDataWorkflowRequest& request,
        std::optional<WorkflowExecutionContext> context) override;

    boost::future<IntelligentReadingStrategy> selectOptimalStrategyAsync(
        const EnhancedDataWorkflowRequest& request,
        const EnhancedSpatialQueryMetadata& spatialMetadata,
        const DataSourceDiscoveryResult& dataSourceResult) override;

    boost::future<IntelligentReadingStrategy::PerformanceExpectation> estimatePerformanceAsync(
        const EnhancedDataWorkflowRequest& request) override;
    
    // Staged execution methods
    boost::future<RequestAnalysisResult> analyzeRequestAsync(
        const EnhancedDataWorkflowRequest& request) override;
        
    boost::future<DataSourceDiscoveryResult> discoverDataSourcesAsync(
        const EnhancedDataWorkflowRequest& request,
        const EnhancedSpatialQueryMetadata& spatialMetadata) override;

    boost::future<IntelligentDataReadingResult> executeIntelligentDataReadingAsync(
        const EnhancedDataWorkflowRequest& request,
        const IntelligentReadingStrategy& strategy,
        const DataSourceDiscoveryResult& dataSources,
        const EnhancedSpatialQueryMetadata& spatialMetadata) override;

    // Management and monitoring methods
    boost::future<WorkflowStatus> getWorkflowStatusAsync(const std::string& executionId) override;
    boost::future<bool> cancelWorkflowAsync(const std::string& executionId) override;
    boost::future<WorkflowExecutionContext::ExecutionStats> getExecutionStatsAsync(
        const std::string& executionId) override;

    // Configuration
    void configure(const EnhancedWorkflowServiceConfig& config) override;
    
    // Additional IEnhancedDataWorkflowService methods
    boost::future<WorkflowResult> executeLegacyWorkflowAsync(const WorkflowRequest& legacyRequest) override;
    ServicePerformanceMetrics getPerformanceMetrics() const override;
    void resetPerformanceMetrics() override;

    // IDataWorkflowService interface implementation
    boost::future<WorkflowResult> executeWorkflowAsync(const WorkflowRequest& request) override;

protected:
    friend class EnhancedSpatialRequestResolver;
    std::shared_ptr<oscean::core_services::ICrsService> getCrsService();
    std::shared_ptr<oscean::core_services::data_access::IUnifiedDataAccessService> getUnifiedDataAccessService();

private:
    friend class EnhancedSpatialRequestResolver;
    
    // Forward declarations of internal classes
    class IntelligentStrategySelector;
    class InterpolationNeedsAnalyzer;
    class ExecutionContextManager;
    class HybridLandMaskProcessor;

    // PImpl classes for internal logic - 声明但不初始化，在构造函数中初始化 - 使用boost::scoped_ptr替代std::unique_ptr
    boost::scoped_ptr<IntelligentStrategySelector> strategySelector_;
    boost::scoped_ptr<InterpolationNeedsAnalyzer> interpolationAnalyzer_;
    boost::scoped_ptr<ExecutionContextManager> contextManager_;
    boost::scoped_ptr<HybridLandMaskProcessor> landMaskProcessor_;
    boost::scoped_ptr<EnhancedSpatialRequestResolver> spatialRequestResolver_;

    // Configuration and state
    EnhancedWorkflowServiceConfig config_;
    std::atomic<bool> isInitialized_{false};
    std::atomic<int> activeWorkflowCount_{0};

    // Performance metrics
    mutable std::mutex metricsMutex_;
    mutable std::mutex initializationMutex_;
    ServicePerformanceMetrics performanceMetrics_;

    // Private helper methods from the implementation file
    std::map<std::string, std::string> extractFileMetadata(const std::string& filePath);
    bool validateVariableAvailability(const std::string& filePath, const std::vector<std::string>& variables);
    void configureMemoryOptimization(const EnhancedDataWorkflowRequest& request);
    boost::future<WorkflowResult> handleWorkflowError(
        const std::string& executionId,
        const std::exception& error,
        const EnhancedDataWorkflowRequest& request);
    
    // Staged execution helpers
    boost::future<DataSourceDiscoveryResult> processDirectFileMode(
        const EnhancedDataWorkflowRequest::DirectFileParams& params,
        const EnhancedSpatialQueryMetadata& spatialMetadata);
        
    boost::future<DataSourceDiscoveryResult> processDatabaseQueryMode(
        const EnhancedDataWorkflowRequest::DatabaseQueryParams& params,
        const EnhancedSpatialQueryMetadata& spatialMetadata);

    double calculateEstimatedTime(const EnhancedDataWorkflowRequest& request, const IntelligentReadingStrategy& strategy);
    double calculateEstimatedMemory(const EnhancedDataWorkflowRequest& request, const IntelligentReadingStrategy& strategy);
    double calculateEstimatedIO(const EnhancedDataWorkflowRequest& request, const IntelligentReadingStrategy& strategy);
    double calculateConfidenceLevel(const EnhancedDataWorkflowRequest& request);

    // 🎯 修正后的工作流步骤方法 - 基于正确的业务逻辑
    boost::future<core_services::FileMetadata> extractFileMetadataAsync(
        const EnhancedDataWorkflowRequest& request);
    
    boost::future<CRSTransformationResult> checkAndTransformCRSAsync(
        const EnhancedDataWorkflowRequest& request,
        const core_services::FileMetadata& fileMetadata);
    
    boost::future<SpatialAnalysisResult> analyzeSpatialResolutionAndCalculateSubsetAsync(
        const EnhancedDataWorkflowRequest& request,
        const core_services::FileMetadata& fileMetadata,
        const CRSTransformationResult& crsResult);
    
    boost::future<IntelligentReadingDecision> makeIntelligentReadingDecisionAsync(
        const EnhancedDataWorkflowRequest& request,
        const core_services::FileMetadata& fileMetadata,
        const CRSTransformationResult& crsResult,
        const SpatialAnalysisResult& spatialResult);
    
    boost::future<IEnhancedDataWorkflowService::IntelligentDataReadingResult> executeDataReadingWithDecisionAsync(
        const EnhancedDataWorkflowRequest& request,
        const IntelligentReadingDecision& decision,
        const IEnhancedDataWorkflowService::DataSourceDiscoveryResult& dataSourceResult,
        const core_services::FileMetadata& fileMetadata,
        const CRSTransformationResult& crsResult,
        const SpatialAnalysisResult& spatialResult);

    // 🎯 同步执行方法，避免Promise生命周期问题
    IEnhancedDataWorkflowService::RequestAnalysisResult executeRequestAnalysisSync(
        const EnhancedDataWorkflowRequest& request);
    EnhancedSpatialQueryMetadata executeSpatialRequestResolverSync(
        const EnhancedDataWorkflowRequest& request);
    IEnhancedDataWorkflowService::DataSourceDiscoveryResult executeDataSourceDiscoverySync(
        const EnhancedDataWorkflowRequest& request,
        const EnhancedSpatialQueryMetadata& spatialMetadata);
    IntelligentReadingStrategy executeStrategySelectionSync(
        const EnhancedDataWorkflowRequest& request,
        const EnhancedSpatialQueryMetadata& spatialMetadata,
        const IEnhancedDataWorkflowService::DataSourceDiscoveryResult& dataSourceResult);
    IEnhancedDataWorkflowService::IntelligentDataReadingResult executeIntelligentDataReadingSync(
        const EnhancedDataWorkflowRequest& request,
        const IntelligentReadingStrategy& strategy,
        const IEnhancedDataWorkflowService::DataSourceDiscoveryResult& dataSources,
        const EnhancedSpatialQueryMetadata& spatialMetadata);

    // Service accessors
    std::shared_ptr<oscean::core_services::spatial_ops::ISpatialOpsService> getSpatialOpsService();
    std::shared_ptr<oscean::core_services::metadata::IMetadataService> getMetadataService();
    
    // Internal initialization
    void initializeInternalComponents();
};

} // namespace oscean::workflow_engine::data_workflow 