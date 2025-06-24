#pragma once

// 🚀 强制禁用boost::asio - 必须在所有其他包含之前
#define OSCEAN_NO_BOOST_ASIO_MODULE 1

#include "common_utils/utilities/boost_config.h"
#pragma message("This module does not use boost::asio")

// 🆕 工作流接口支持 - 使用前向声明避免命名空间冲突
namespace oscean::workflow_engine {
    class IWorkflow;
    enum class WorkflowType;
}

// 🎯 只包含接口，不包含实现 - 纯编排层设计
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/crs/i_crs_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/utilities/file_format_detector.h"

#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <optional>
#include <atomic>

#include "workflow_engine/service_management/i_service_manager.h"

namespace oscean::workflow_engine::data_management {

// ===============================================================================
// 工作流配置和状态类型 - 仅用于编排层
// ===============================================================================

/**
 * @brief 工作流处理配置 - 编排层配置，不重复核心服务配置
 */
struct WorkflowProcessingConfig {
    bool enableProgressCallback = true;
    bool skipExistingFiles = true;        // 通过元数据服务检查
    bool enableParallelProcessing = true; // 使用统一线程池
    bool enableErrorRecovery = true;
    std::chrono::seconds timeout = std::chrono::seconds(300);
    size_t maxConcurrentFiles = 4;       // 由CommonServicesFactory管理
};

/**
 * @brief 工作流状态 - 编排层状态管理
 */
enum class WorkflowStatus {
    NOT_STARTED,
    INITIALIZING,
    SCANNING_FILES,
    PROCESSING_FILES,
    COMPLETED,
    FAILED,
    CANCELLED
};

/**
 * @brief 工作流结果 - 编排结果汇总
 */
struct WorkflowResult {
    WorkflowStatus status = WorkflowStatus::NOT_STARTED;
    size_t totalFiles = 0;
    size_t processedFiles = 0;
    size_t skippedFiles = 0;
    size_t failedFiles = 0;
    std::chrono::milliseconds duration{0};
    std::vector<std::string> errorMessages;
    std::string workflowId;
};

/**
 * @brief 文件处理回调接口 - 用于进度通知
 */
class IFileProcessingCallback {
public:
    virtual ~IFileProcessingCallback() = default;
    virtual void onFileStarted(const std::string& filePath) = 0;
    virtual void onFileCompleted(const std::string& filePath, bool success, const std::string& error = "") = 0;
    virtual void onProgressUpdate(size_t processed, size_t total) = 0;
};

// ===============================================================================
// DataManagementService - 纯编排层，调用现有服务
// ===============================================================================

/**
 * @brief 数据管理服务 - 纯服务编排层
 * 
 * 🎯 设计原则：
 * ✅ 只做服务编排，不重新实现功能
 * ✅ 使用统一服务管理器获取所有服务
 * ✅ 使用统一异步框架管理任务  
 * ✅ 提供简化的工作流接口
 * ✅ 处理服务间的协调和错误处理
 * 
 * ❌ 不做的事情：
 * ❌ 不重新实现文件读取
 * ❌ 不重新实现元数据提取
 * ❌ 不重新实现数据库操作
 * ❌ 不重新实现坐标转换
 */
class DataManagementService : public std::enable_shared_from_this<DataManagementService> {
public:
    /**
     * @brief 构造函数 - 使用统一服务管理器
     * @param serviceManager 统一服务管理器，提供所有核心服务
     */
    explicit DataManagementService(
        std::shared_ptr<service_management::IServiceManager> serviceManager
    );

    ~DataManagementService();

    /**
     * @brief 优雅地关闭服务，等待所有任务完成
     */
    void shutdown();

    // ===============================================================================
    // 工作流注册支持方法 - 供工作流注册表使用
    // ===============================================================================
    
    /**
     * @brief 获取工作流类型
     */
    oscean::workflow_engine::WorkflowType getType() const;
    
    /**
     * @brief 获取工作流名称
     */
    std::string getName() const;
    
    /**
     * @brief 获取工作流版本
     */
    std::string getVersion() const;
    
    /**
     * @brief 初始化工作流（注册接口）
     */
    bool initializeWorkflow(const std::map<std::string, std::any>& config);
    
    /**
     * @brief 检查工作流是否健康（注册接口）
     */
    bool isHealthy() const;
    
    /**
     * @brief 关闭工作流（注册接口）
     */
    void shutdownWorkflow();

    // ===============================================================================
    // 高级工作流接口 - 编排多个服务调用
    // ===============================================================================

    /**
     * @brief 处理数据目录 - 编排完整的处理工作流
     * 
     * 工作流步骤：
     * 1. 调用FilesystemUtils扫描文件
     * 2. 调用MetadataService检查已处理文件
     * 3. 调用DataAccessService处理新文件
     * 4. 调用MetadataService存储元数据
     * 
     * @param directory 目录路径
     * @param recursive 是否递归扫描
     * @param config 工作流配置
     * @param callback 进度回调
     * @return 工作流结果
     */
    boost::future<WorkflowResult> processDataDirectoryAsync(
        const std::string& directory,
        bool recursive = true,
        const WorkflowProcessingConfig& config = {},
        std::shared_ptr<IFileProcessingCallback> callback = nullptr
    );

    /**
     * @brief 处理单个文件 - 简化的单文件工作流
     * 
     * 工作流步骤：
     * 1. 调用DataAccessService提取元数据
     * 2. 调用MetadataService分类和存储
     * 3. 可选：调用CrsService处理坐标转换
     * 
     * @param filePath 文件路径
     * @param config 工作流配置
     * @return 处理是否成功
     */
    boost::future<bool> processDataFileAsync(
        const std::string& filePath,
        const WorkflowProcessingConfig& config = {}
    );

    /**
     * @brief 批量文件处理 - 并行编排工作流
     * 
     * @param filePaths 文件路径列表
     * @param config 工作流配置
     * @param callback 进度回调
     * @return 工作流结果
     */
    boost::future<WorkflowResult> processBatchFilesAsync(
        const std::vector<std::string>& filePaths,
        const WorkflowProcessingConfig& config = {},
        std::shared_ptr<IFileProcessingCallback> callback = nullptr
    );

    // ===============================================================================
    // 查询接口 - 直接委托给MetadataService
    // ===============================================================================

    /**
     * @brief 时间范围查询 - 直接调用MetadataService
     */
    boost::future<std::vector<std::string>> queryByTimeRangeAsync(
        const std::chrono::system_clock::time_point& startTime,
        const std::chrono::system_clock::time_point& endTime
    );

    /**
     * @brief 空间范围查询 - 直接调用MetadataService  
     */
    boost::future<std::vector<std::string>> queryBySpatialBoundsAsync(
        double minX, double minY, double maxX, double maxY,
        const std::string& crs = "EPSG:4326"
    );

    /**
     * @brief 变量查询 - 直接调用MetadataService
     */
    boost::future<std::vector<std::string>> queryByVariablesAsync(
        const std::vector<std::string>& variableNames
    );

    /**
     * @brief 高级查询 - 直接委托给MetadataService的复杂查询接口
     */
    boost::future<std::vector<std::string>> queryAdvancedAsync(
        const core_services::metadata::QueryCriteria& criteria
    );

    // ===============================================================================
    // 工作流状态管理 - 编排层状态，不是业务状态
    // ===============================================================================

    /**
     * @brief 获取工作流状态
     */
    WorkflowStatus getWorkflowStatus(const std::string& workflowId) const;

    /**
     * @brief 取消工作流
     */
    boost::future<bool> cancelWorkflowAsync(const std::string& workflowId);

    /**
     * @brief 获取工作流历史
     */
    std::vector<WorkflowResult> getWorkflowHistory() const;

    /**
     * @brief 清理已完成的工作流记录
     */
    void cleanupCompletedWorkflows();

    // ===============================================================================
    // 服务健康检查 - 检查依赖服务状态
    // ===============================================================================

    /**
     * @brief 检查所有依赖服务的健康状态
     */
    boost::future<std::map<std::string, std::string>> getServiceHealthAsync() const;

    /**
     * @brief 检查数据管理服务是否就绪
     */
    bool isReady() const;

private:
    /**
     * @brief 验证核心服务依赖是否有效
     */
    void validateDependencies() const;

    /**
     * @brief 获取服务的便捷方法
     */
    std::shared_ptr<core_services::data_access::IUnifiedDataAccessService> getDataAccessService() const;
    std::shared_ptr<core_services::metadata::IMetadataService> getMetadataService() const;
    std::shared_ptr<core_services::ICrsService> getCrsService() const;
    std::shared_ptr<common_utils::utilities::FileFormatDetector> getFormatDetector() const;

    // ===============================================================================
    // 内部编排方法 - 协调服务调用
    // ===============================================================================

    /**
     * @brief 生成工作流ID
     */
    std::string generateWorkflowId();

    /**
     * @brief 扫描文件 - 调用CommonUtilities的文件扫描功能
     */
    boost::future<std::vector<std::string>> scanFilesAsync(
        const std::string& directory, 
        bool recursive,
        const WorkflowProcessingConfig& config
    );

    /**
     * @brief 过滤已处理文件 - 调用MetadataService查询
     */
    boost::future<std::vector<std::string>> filterUnprocessedFilesAsync(
        const std::vector<std::string>& filePaths,
        const WorkflowProcessingConfig& config
    );

    /**
     * @brief 处理单个文件的内部实现 - 编排服务调用
     */
    boost::future<bool> processFileInternalAsync(
        const std::string& filePath,
        const WorkflowProcessingConfig& config
    );

    /**
     * @brief 更新工作流状态
     */
    void updateWorkflowStatus(const std::string& workflowId, WorkflowStatus status);

    // 核心依赖 - 统一服务管理
    std::shared_ptr<service_management::IServiceManager> serviceManager_;

    // 工作流状态管理
    struct ActiveWorkflowState {
        WorkflowStatus status;
        bool cancelled = false;  // 移除atomic，使用mutex保护
        
        // 提供拷贝构造和赋值
        ActiveWorkflowState() = default;
        ActiveWorkflowState(WorkflowStatus s, bool c = false) : status(s), cancelled(c) {}
        ActiveWorkflowState(const ActiveWorkflowState&) = default;
        ActiveWorkflowState& operator=(const ActiveWorkflowState&) = default;
        ActiveWorkflowState(ActiveWorkflowState&&) = default;
        ActiveWorkflowState& operator=(ActiveWorkflowState&&) = default;
    };

    mutable std::mutex workflowMutex_;
    std::map<std::string, ActiveWorkflowState> activeWorkflows_;
    std::vector<WorkflowResult> workflowHistory_;
    std::atomic<size_t> workflowCounter_{0};
    
    // 优雅停机支持
    std::atomic<bool> isShutdown_{false};
};

/**
 * @brief DataManagementService 工厂函数 - 使用服务管理器
 * @param serviceManager 统一服务管理器
 * @return 数据管理服务实例
 */
std::shared_ptr<DataManagementService> createDataManagementService(
    std::shared_ptr<service_management::IServiceManager> serviceManager
);

} // namespace oscean::workflow_engine::data_management