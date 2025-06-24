/**
 * @file data_management_workflow.h
 * @brief 统一的数据管理工作流接口 - 最终正确版（扩展插值、空间计算、输出服务）
 */
#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <chrono>
#include <boost/optional.hpp>
#include <boost/thread/future.hpp>

#include "workflow_engine/i_workflow.h"
#include "workflow_engine/service_management/i_service_manager.h"

// 包含所有必需的服务接口和通用数据类型
#include "core_services/common_data_types.h"
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/interpolation/i_interpolation_service.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "core_services/output/i_output_service.h"

namespace oscean::workflow_engine::data_management {

/**
 * @brief 批处理配置结构体 - 扩展版本
 */
struct BatchProcessingConfig {
    // 基础配置
    std::string inputDirectory;
    std::string outputDirectory;
    std::vector<std::string> filePatterns;
    bool enableParallelProcessing = true;
    size_t maxConcurrentTasks = 4;
    
    // 🆕 插值配置
    bool enableInterpolation = false;
    oscean::core_services::InterpolationMethod interpolationMethod = oscean::core_services::InterpolationMethod::BILINEAR;
    boost::optional<oscean::core_services::GridDefinition> targetGrid;
    
    // 🆕 空间计算配置
    bool enableSpatialOps = false;
    boost::optional<oscean::core_services::BoundingBox> clipBounds;
    boost::optional<std::string> targetCRS;
    
    // 🆕 输出配置
    bool enableOutput = false;
    oscean::core_services::output::OutputFormat outputFormat = oscean::core_services::output::OutputFormat::NETCDF;
    std::string outputFileTemplate = "processed_{{filename}}";
    
    // 质量控制配置
    bool enableQualityCheck = true;
    double qualityThreshold = 0.8;
};

/**
 * @brief 工作流状态枚举
 */
enum class WorkflowState {
    IDLE,
    RUNNING,
    PAUSED,
    COMPLETED,
    FAILED,
    CANCELLED
};

/**
 * @brief 处理结果结构体
 */
struct ProcessingResult {
    bool success = false;
    std::string filePath;
    std::string errorMessage;
    std::chrono::milliseconds processingTime{0};
    size_t dataPointsProcessed = 0;
    double qualityScore = 0.0;
    
    // 🆕 扩展结果信息
    boost::optional<std::string> interpolatedDataPath;
    boost::optional<std::string> spatialProcessedDataPath;
    boost::optional<std::string> outputFilePath;
};

/**
 * @brief 活动工作流状态
 */
struct ActiveWorkflowState {
    bool isRunning = false;
    mutable std::mutex stateMutex;  // 修复：添加mutable以允许const方法中使用
    WorkflowState currentState = WorkflowState::IDLE;
    std::chrono::steady_clock::time_point startTime;
    size_t totalFiles = 0;
    size_t processedFiles = 0;
    std::string currentFile;
    std::vector<ProcessingResult> results;
};

/**
 * @brief 数据管理工作流类 - 扩展版本
 */
class DataManagementWorkflow : public IWorkflow {
public:
    /**
     * @brief 构造函数
     */
    DataManagementWorkflow(
        std::string workflowId,
        std::shared_ptr<service_management::IServiceManager> serviceManager);

    /**
     * @brief 析构函数
     */
    ~DataManagementWorkflow() = default;

    // IWorkflow接口实现
    const char* getName() const override;
    void execute() override;

    // 🆕 扩展的工作流控制方法（非override）
    boost::future<void> executeAsync();
    void pause();
    void resume();
    void cancel();
    std::string getWorkflowId() const;
    bool isRunning() const;

    // 🆕 扩展的批处理方法
    boost::future<std::vector<ProcessingResult>> processBatchAsync(
        const BatchProcessingConfig& config);

    // 🆕 单文件处理方法（支持完整管道）
    boost::future<ProcessingResult> processFileAsync(
        const std::string& filePath,
        const BatchProcessingConfig& config);

private:
    // 🆕 插值处理方法
    boost::future<std::shared_ptr<oscean::core_services::GridData>> performInterpolation(
        std::shared_ptr<oscean::core_services::GridData> sourceData,
        const oscean::core_services::GridDefinition& targetGrid,
        oscean::core_services::InterpolationMethod method);

    // 🆕 空间计算处理方法
    boost::future<std::shared_ptr<oscean::core_services::GridData>> performSpatialOps(
        std::shared_ptr<oscean::core_services::GridData> sourceData,
        const boost::optional<oscean::core_services::BoundingBox>& clipBounds,
        const boost::optional<oscean::core_services::CRSInfo>& targetCRS);

    // 🆕 输出生成方法
    boost::future<oscean::core_services::output::OutputResult> generateOutput(
        std::shared_ptr<oscean::core_services::GridData> processedData,
        const oscean::core_services::output::OutputRequest& outputRequest);

    // 私有辅助方法
    std::vector<std::string> scanInputFiles(const BatchProcessingConfig& config);
    boost::future<std::shared_ptr<oscean::core_services::GridData>> loadDataAsync(
        const std::string& filePath);
    bool performQualityCheck(std::shared_ptr<oscean::core_services::GridData> data) const;
    void updateProgress(const std::string& currentFile, size_t processed, size_t total);

    // 成员变量
    std::string m_workflowId;
    std::shared_ptr<service_management::IServiceManager> m_serviceManager;
    ActiveWorkflowState m_state;
    
    // 🆕 扩展服务缓存
    std::shared_ptr<oscean::core_services::interpolation::IInterpolationService> m_interpolationService;
    std::shared_ptr<oscean::core_services::spatial_ops::ISpatialOpsService> m_spatialOpsService;
    std::shared_ptr<oscean::core_services::output::IOutputService> m_outputService;
    
    // 统计信息
    mutable std::mutex m_statsMutex;
    size_t m_totalFilesProcessed = 0;
    std::chrono::milliseconds m_totalProcessingTime{0};
};

} // namespace oscean::workflow_engine::data_management

// 工厂函数
std::shared_ptr<oscean::workflow_engine::IWorkflow> 
create_workflow(std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> serviceManager);