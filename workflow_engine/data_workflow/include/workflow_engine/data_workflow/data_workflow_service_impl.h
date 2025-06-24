#pragma once

/**
 * @file data_workflow_service_impl.h
 * @brief Implementation header for the intelligent data processing workflow service.
 * @author OSCEAN Team
 * @date 2024
 */

#include "workflow_engine/data_workflow/i_data_workflow_service.h"
#include "workflow_engine/data_workflow/data_workflow_types.h"
#include "workflow_engine/data_workflow/enhanced_workflow_types.h"
#include "workflow_engine/service_management/i_service_manager.h"
#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/interpolation/i_interpolation_service.h"
#include "core_services/output/i_output_service.h"
// 移除有问题的基础设施依赖
// #include "common_utils/infrastructure/common_services_factory.h"
// #include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace oscean::workflow_engine::data_workflow {

/**
 * @brief 多变量数据处理结果
 */
struct MultiVariableDataResult {
    bool success = false;                         ///< 是否成功
    std::string filePath;                         ///< 处理的文件路径
    std::vector<std::string> processedVariables; ///< 成功处理的变量
    std::vector<std::string> failedVariables;    ///< 处理失败的变量
    
    // 统计信息
    size_t totalVariables = 0;                   ///< 总变量数
    size_t successCount = 0;                     ///< 成功处理的变量数
    size_t failureCount = 0;                     ///< 失败的变量数
    
    // 每个变量对应的数据
    std::map<std::string, std::shared_ptr<core_services::GridData>> variableData;
    
    // 错误信息
    std::string errorMessage;                     ///< 错误消息
};

/**
 * @brief 融合后的多变量数据
 */
struct FusedMultiVariableData {
    bool success = false;                         ///< 是否成功
    
    // 融合策略结果
    std::shared_ptr<core_services::GridData> singleFusedGrid;  ///< 单一融合网格（如果启用融合）
    std::map<std::string, std::shared_ptr<core_services::GridData>> separateVariableGrids; ///< 分离的变量网格
    
    std::vector<std::string> availableVariables; ///< 可用的变量列表
    std::string errorMessage;                     ///< 错误消息
};

/**
 * @class DataWorkflowServiceImpl
 * @brief Implements the IDataWorkflowService interface, orchestrating various core
 *        services to fulfill a WorkflowRequest.
 */
class DataWorkflowServiceImpl : public virtual IDataWorkflowService {
public:
    /**
     * @brief Constructor using unified service manager.
     * @param serviceManager Shared pointer to the service manager
     */
    explicit DataWorkflowServiceImpl(std::shared_ptr<service_management::IServiceManager> serviceManager);

    ~DataWorkflowServiceImpl() override;

    // --- IDataWorkflowService Implementation ---

    boost::future<WorkflowResult> executeWorkflowAsync(
        const WorkflowRequest& request) override;

    std::string getWorkflowName() const override;

    bool isReady() const override;

protected:
    void initialize();
    
    // --- Service Getter Helper Methods ---
    std::shared_ptr<core_services::spatial_ops::ISpatialOpsService> getSpatialOpsService() const;
    std::shared_ptr<core_services::data_access::IUnifiedDataAccessService> getDataAccessService() const;
    std::shared_ptr<core_services::metadata::IMetadataService> getMetadataService() const;
    std::shared_ptr<core_services::ICrsService> getCrsService() const;
    std::shared_ptr<core_services::interpolation::IInterpolationService> getInterpolationService() const;
    std::shared_ptr<core_services::output::IOutputService> getOutputService() const;
    // 简化线程池管理 - 移除复杂的基础设施依赖
    // std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> getThreadPoolManager() const;

    // --- Helper methods for workflow steps ---
    boost::future<core_services::Geometry> resolveSpatialRequestAsync(
        const SpatialRequest& spatialRequest);

    boost::future<std::vector<std::string>> findDataSourcesAsync(
        const core_services::Geometry& queryGeometry, 
        const WorkflowRequest& request);

    // --- 智能读取策略方法 (根据优化方案添加) ---
    IntelligentReadingStrategy selectOptimalReadingStrategy(
        const WorkflowRequest& request,
        const std::vector<std::string>& dataSources);

    boost::future<std::vector<std::shared_ptr<core_services::GridData>>> executeDataReadingAsync(
        const std::vector<std::string>& dataSources,
        const core_services::Geometry& queryGeometry,
        const WorkflowRequest& request,
        const IntelligentReadingStrategy& strategy);

    boost::future<std::shared_ptr<core_services::GridData>> executeProcessingPipelineAsync(
        const std::vector<std::shared_ptr<core_services::GridData>>& rawData,
        const WorkflowRequest& request);

    boost::future<WorkflowResult> generateOutputAsync(
        std::shared_ptr<core_services::GridData> processedData,
        const WorkflowRequest& request);

    // --- 辅助方法 ---
    std::shared_ptr<core_services::GridData> executeSmartDataReadingForFile(
        const std::string& filePath,
        const std::string& variableName,
        const WorkflowRequest& request);

    boost::future<std::shared_ptr<core_services::GridData>> processSingleDataSourceAsync(
        const std::string& filePath,
        const core_services::Geometry& originalQueryGeometry,
        const WorkflowRequest& request
    );

    boost::future<MultiVariableDataResult> processMultiVariableDataSourceAsync(
        const std::string& filePath,
        const std::vector<std::string>& variableNames,
        const core_services::Geometry& queryGeometry,
        const WorkflowRequest& request);

    boost::future<std::shared_ptr<core_services::GridData>> processSingleVariableAsync(
        const std::string& filePath,
        const std::string& variableName,
        const core_services::Geometry& queryGeometry,
        const WorkflowRequest& request);

    boost::future<std::shared_ptr<core_services::GridData>> fuseDataAsync(
        std::vector<std::shared_ptr<core_services::GridData>>& allData);

    boost::future<FusedMultiVariableData> fuseMultiVariableDataAsync(
        const std::vector<MultiVariableDataResult>& allResults,
        const WorkflowRequest& request);

    boost::future<WorkflowResult> postProcessDataAsync(
        std::shared_ptr<core_services::GridData> finalData,
        const WorkflowRequest& request);

    boost::future<WorkflowResult> postProcessMultiVariableDataAsync(
        const FusedMultiVariableData& fusedData,
        const WorkflowRequest& request);

    // --- Unified Service Manager ---
    std::shared_ptr<service_management::IServiceManager> serviceManager_;
    
private:
    // --- Cached Service Pointers (Lazy-loaded) ---
    mutable std::shared_ptr<core_services::spatial_ops::ISpatialOpsService> spatialOpsService_;
    mutable std::shared_ptr<core_services::data_access::IUnifiedDataAccessService> dataAccessService_;
    mutable std::shared_ptr<core_services::metadata::IMetadataService> metadataService_;
    mutable std::shared_ptr<core_services::ICrsService> crsService_;
    mutable std::shared_ptr<core_services::interpolation::IInterpolationService> interpolationService_;
    mutable std::shared_ptr<core_services::output::IOutputService> outputService_;

    const std::string m_workflowName = "IntelligentDataProcessingWorkflow";
};

} // namespace oscean::workflow_engine::data_workflow 