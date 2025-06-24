/**
 * @file main.cpp
 * @brief OSCEAN海洋数据读取应用程序主入口
 * @author OSCEAN Team
 * @date 2024
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>

// 服务管理
#include "workflow_engine/service_management/i_service_manager.h"
#include "../../workflow_engine/src/service_management/service_manager_impl.h"

// 工作流服务
#include "workflow_engine/data_workflow/i_enhanced_data_workflow_service.h"
#include "workflow_engine/data_workflow/enhanced_data_workflow_service_impl.h"

// 数据访问服务
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/data_access/i_data_access_service_factory.h"
#include "common_utils/infrastructure/common_services_factory.h"

// 数据类型
#include "workflow_engine/data_workflow/enhanced_workflow_types.h"
#include "workflow_engine/data_workflow/data_workflow_types.h"
#include "core_services/common_data_types.h"

// 基础设施
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "common_utils/infrastructure/common_services_factory.h"

// 日志
#include "common_utils/utilities/logging_utils.h"

using namespace oscean;
using namespace oscean::workflow_engine::data_workflow;
using namespace oscean::workflow_engine::service_management;
using namespace oscean::core_services;

// 前向声明
namespace oscean::core_services {
    class UnifiedDataAccessServiceImpl;
}

/**
 * @brief OSCEAN海洋数据读取应用程序
 */
class OceanDataReaderApp {
private:
    std::shared_ptr<IServiceManager> serviceManager_;
    std::shared_ptr<IEnhancedDataWorkflowService> workflowService_;
    bool serviceRegistered_ = false;

public:
    /**
     * @brief 构造函数 - 初始化服务管理器
     */
    OceanDataReaderApp() {
        std::cout << "🌊 OSCEAN海洋数据读取应用程序启动\n";
        
        // 创建线程池管理器
        auto threadPoolManager = std::make_shared<common_utils::infrastructure::UnifiedThreadPoolManager>();
        
        // 创建服务管理器
        serviceManager_ = std::make_shared<ServiceManagerImpl>(threadPoolManager);
        
        std::cout << "📋 统一服务管理架构已加载\n";
    }

    /**
     * @brief 注册工作流服务到服务管理器
     */
    void registerWorkflowService() {
        if (serviceRegistered_) return;
        
        std::cout << "🔧 注册增强数据工作流服务...\n";
        
        // 类型转换为ServiceManagerImpl以访问registerServiceFactory方法
        auto serviceManagerImpl = std::dynamic_pointer_cast<ServiceManagerImpl>(serviceManager_);
        if (!serviceManagerImpl) {
            throw std::runtime_error("服务管理器类型转换失败");
        }
        
        // 🎯 核心服务已经在ServiceManagerImpl中注册，无需在应用层重复注册
        std::cout << "✅ 核心服务由服务管理器统一管理\n";
        
        // 注册增强数据工作流服务工厂
        serviceManagerImpl->registerServiceFactory<IEnhancedDataWorkflowService>(
            [this]() -> std::shared_ptr<IEnhancedDataWorkflowService> {
                std::cout << "🏗️ 创建增强数据工作流服务实例...\n";
                return std::make_shared<EnhancedDataWorkflowServiceImpl>(serviceManager_);
            }
        );
        
        serviceRegistered_ = true;
        std::cout << "✅ 工作流服务注册完成\n";
    }

    /**
     * @brief 获取工作流服务（懒加载）
     */
    std::shared_ptr<IEnhancedDataWorkflowService> getWorkflowService() {
        if (!workflowService_) {
            // 先注册增强数据工作流服务
            registerWorkflowService();
            
            std::cout << "🔍 获取工作流服务实例...\n";
            workflowService_ = serviceManager_->getService<IEnhancedDataWorkflowService>();
            if (!workflowService_) {
                throw std::runtime_error("无法获取增强数据工作流服务");
            }
            std::cout << "✅ 工作流服务获取成功\n";
        }
        return workflowService_;
    }

    /**
     * @brief 显示欢迎界面
     */
    void showWelcome() {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "    OSCEAN 海洋数据读取应用程序\n";
        std::cout << "========================================\n";
        std::cout << "支持功能:\n";
        std::cout << "• 单点数据查询 - 指定经纬度获取垂直剖面数据\n";
        std::cout << "• 多变量支持 - 同时查询多个海洋变量\n";
        std::cout << "• NetCDF文件读取 - 支持标准海洋数据格式\n";
        std::cout << "========================================\n\n";
    }

    /**
     * @brief 获取用户输入的查询参数
     */
    struct QueryParams {
        double longitude;
        double latitude;
        std::string filePath;
        std::vector<std::string> variableNames;
    };

    QueryParams getUserInput() {
        QueryParams params;
        
        std::cout << "📍 请输入查询位置:\n";
        std::cout << "经度 (度): ";
        std::cin >> params.longitude;
        std::cout << "纬度 (度): ";
        std::cin >> params.latitude;
        
        std::cout << "\n📁 请输入NetCDF文件路径:\n";
        std::cout << "文件路径: ";
        std::cin.ignore(); // 清除输入缓冲区
        std::getline(std::cin, params.filePath);
        
        std::cout << "\n🔢 请输入要查询的变量 (用空格分隔):\n";
        std::cout << "变量名: ";
        std::string variablesInput;
        std::getline(std::cin, variablesInput);
        
        // 解析变量列表
        std::istringstream iss(variablesInput);
        std::string variable;
        while (iss >> variable) {
            params.variableNames.push_back(variable);
        }
        
        return params;
    }

    /**
     * @brief 创建工作流请求
     */
    EnhancedDataWorkflowRequest createWorkflowRequest(const QueryParams& params) {
        std::cout << "\n🔄 创建工作流请求...\n";
        std::cout << "📍 查询位置: " << params.longitude << "°E, " << params.latitude << "°N\n";
        std::cout << "📁 数据文件: " << params.filePath << "\n";
        std::cout << "🔢 查询变量: ";
        for (size_t i = 0; i < params.variableNames.size(); ++i) {
            std::cout << params.variableNames[i];
            if (i < params.variableNames.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
        
        // 🎯 添加详细的坐标调试信息
        std::cout << "\n🔍 详细坐标信息:\n";
        std::cout << "  • 查询经度: " << std::fixed << std::setprecision(6) << params.longitude << "°E\n";
        std::cout << "  • 查询纬度: " << std::fixed << std::setprecision(6) << params.latitude << "°N\n";
        std::cout << "  • 预期数据: 根据您提供的图片，该点vo第一层应为 -0.2 m/s\n";
        std::cout << "  • 调试目标: 验证坐标定位和数据读取的准确性\n";

        // 创建点空间请求
        Point queryPoint{params.longitude, params.latitude, 0.0}; // 使用聚合初始化

        // 创建增强工作流请求 - 使用聚合初始化
        EnhancedDataWorkflowRequest request{
            queryPoint,  // spatialRequest
            EnhancedDataWorkflowRequest::DataSourceMode::DIRECT_FILES  // dataSourceMode
        };

        // 配置直接文件参数
        EnhancedDataWorkflowRequest::DirectFileParams fileParams;
        EnhancedDataWorkflowRequest::DirectFileParams::FileSpec fileSpec;
        fileSpec.filePath = params.filePath;
        fileSpec.variableNames = params.variableNames;

        // 配置深度维度 - 读取垂直有效层所有数据
        EnhancedDataWorkflowRequest::DirectFileParams::FileSpec::DepthDimension depthDim;
        depthDim.depthUnit = "meters";
        depthDim.depthPositive = "down";
        fileSpec.depthDimension = depthDim;

        // 配置CRS处理
        EnhancedDataWorkflowRequest::DirectFileParams::FileSpec::CRSHandling crsHandling;
        crsHandling.userCRS = "EPSG:4326";
        crsHandling.enableAutoDetection = true;
        crsHandling.enableTransformation = true;
        crsHandling.preferredOutputCRS = "EPSG:4326";
        fileSpec.crsHandling = crsHandling;

        fileParams.fileSpecs.push_back(fileSpec);
        request.directFileParams = fileParams;

        // 配置输出选项
        EnhancedDataWorkflowRequest::OutputOptions outputOptions;
        outputOptions.format = EnhancedDataWorkflowRequest::OutputOptions::Format::CSV;
        outputOptions.outputPath = "ocean_data_query_result.txt";
        outputOptions.includeMetadata = true;
        outputOptions.includeProcessingHistory = true;
        request.outputOptions = outputOptions;

        // 配置工作流选项
        EnhancedDataWorkflowRequest::WorkflowOptions workflowOptions;
        workflowOptions.workflowId = "ocean_data_query";
        workflowOptions.priority = 8;
        workflowOptions.timeout = std::chrono::seconds(120);
        workflowOptions.enableProgressCallback = true;
        workflowOptions.enableErrorRecovery = true;
        workflowOptions.maxRetries = 2;
        request.workflowOptions = workflowOptions;

        std::cout << "✅ 工作流请求创建完成\n";
        return request;
    }

    /**
     * @brief 执行工作流并跟踪执行状态
     */
    bool executeWorkflowWithTracking(const EnhancedDataWorkflowRequest& request) {
        try {
            std::cout << "\n🚀 开始执行海洋数据工作流...\n";
            std::cout << "=====================================\n";
            
            auto workflow = getWorkflowService();
            
            std::cout << "📋 工作流执行步骤:\n";
            std::cout << "1️⃣ 请求分析和验证\n";
            std::cout << "2️⃣ 空间请求解析\n";
            std::cout << "3️⃣ 数据源发现\n";
            std::cout << "4️⃣ 策略选择\n";
            std::cout << "5️⃣ 智能数据读取\n";
            std::cout << "6️⃣ 数据处理\n";
            std::cout << "7️⃣ 生成输出\n\n";
            
            std::cout << "⏳ 提交工作流请求...\n";
            std::cout << "🔍 调用 workflow->executeEnhancedWorkflowAsync...\n";
            auto resultFuture = workflow->executeEnhancedWorkflowAsync(request);
            std::cout << "✅ executeEnhancedWorkflowAsync调用返回\n";
            
            std::cout << "⌛ 等待工作流执行完成 (最大等待时间: 120秒)...\n";
            
            // 使用带超时的等待
            auto status = resultFuture.wait_for(boost::chrono::seconds(120));
            
            if (status == boost::future_status::timeout) {
                std::cout << "⏰ 工作流执行超时 (120秒)\n";
                return false;
            } else if (status == boost::future_status::ready) {
                std::cout << "📊 工作流执行完成，获取结果...\n";
                auto result = resultFuture.get();
                processWorkflowResult(result);
                return result.success;
            } else {
                std::cout << "❓ 工作流状态未知\n";
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cout << "❌ 工作流执行异常: " << e.what() << "\n";
            std::cout << "🔍 异常类型: " << typeid(e).name() << "\n";
            
            // 添加详细的错误诊断
            std::cout << "\n🔧 错误诊断:\n";
            std::cout << "• 检查文件路径是否正确\n";
            std::cout << "• 检查文件是否存在且可访问\n";
            std::cout << "• 检查变量名是否在文件中存在\n";
            std::cout << "• 检查坐标是否在数据范围内\n";
            
            return false;
        }
    }

    /**
     * @brief 处理工作流结果 - 增强版，详细打印所有数据信息并生成CSV文件
     */
    void processWorkflowResult(const WorkflowResult& result) {
        std::cout << "\n📊 工作流执行结果详细报告:\n";
        std::cout << "=========================================\n";
        
        // 基本执行状态
        std::cout << "🔍 基本执行信息:\n";
        std::cout << "  ✓ 执行成功: " << (result.success ? "是" : "否") << "\n";
        std::cout << "  📈 执行状态: " << getStatusText(result.status) << " (" << static_cast<int>(result.status) << ")\n";
        std::cout << "  ⏱️  执行耗时: " << result.duration.count() << " ms\n";
        
        if (!result.message.empty()) {
            std::cout << "  💬 状态消息: " << result.message << "\n";
        }
        
        if (result.error.has_value()) {
            std::cout << "  🔥 错误信息: " << result.error.value() << "\n";
        }
        
        std::cout << "\n";
        
        // 数据处理统计
        std::cout << "📊 数据处理统计:\n";
        std::cout << "  📂 处理数据源数量: " << result.processedDataSources << "\n";
        std::cout << "  📁 总处理文件数: " << result.totalFilesProcessed << "\n";
        std::cout << "  ✅ 成功处理文件数: " << result.successfulFilesProcessed << "\n";
        std::cout << "  📈 文件处理成功率: " << result.getSuccessRate() << "%\n";
        std::cout << "  🔢 总数据点数: " << result.totalDataPoints << "\n";
        std::cout << "  💾 数据体积: " << result.dataVolumeMB << " MB\n";
        std::cout << "\n";
        
        // 🎯 应用层数据处理：从工作流结果中获取数据并显示
        if (result.success && result.gridData) {
            std::cout << "🎯 从工作流服务获取数据，开始在应用层显示和处理...\n";
            
            try {
                // 显示数据信息
                displayGridDataInfo(*result.gridData);
                
                // 生成TXT文件
                std::string txtPath = "ocean_data_query_result.txt";
                std::cout << "📝 开始生成TXT文件: " << txtPath << "\n";
                
                // generateDataReport(txtPath, result, *result.gridData); // 暂时注释掉
                
            } catch (const std::exception& e) {
                std::cout << "❌ 应用层数据处理失败: " << e.what() << "\n";
            }
        } else if (result.success) {
            std::cout << "🎯 工作流执行成功但无数据，生成基本报告...\n";
            try {
                std::string txtPath = "ocean_data_query_result.txt";
                // generateSimpleDataReport(txtPath, result); // 暂时注释掉
            } catch (const std::exception& e) {
                std::cout << "❌ TXT文件生成失败: " << e.what() << "\n";
            }
        }
        
        // 🔍 详细诊断：分析为什么没有数据
        std::cout << "🔍 详细诊断分析:\n";
        if (result.success && result.totalDataPoints == 0) {
            std::cout << "  ⚠️  执行成功但无数据 - 可能的原因:\n";
            std::cout << "    1. 空间坐标不在NetCDF文件的数据范围内\n";
            std::cout << "    2. 变量名在文件中不存在或拼写错误\n";
            std::cout << "    3. 文件损坏或格式不正确\n";
            std::cout << "    4. 数据访问服务配置问题\n";
            std::cout << "    5. 工作流内部数据传递问题\n";
        }
        
        if (result.processedDataSources > 0 && result.totalFilesProcessed == 0) {
            std::cout << "  ⚠️  发现数据源但未处理文件 - 可能的原因:\n";
            std::cout << "    1. 数据源发现逻辑与文件处理逻辑不一致\n";
            std::cout << "    2. 文件路径转换或访问权限问题\n";
            std::cout << "    3. 工作流执行步骤之间的数据传递中断\n";
        }
        
        // 变量处理详情
        std::cout << "\n🔢 变量处理详情:\n";
        if (!result.processedVariables.empty()) {
            std::cout << "  ✅ 成功处理变量 (" << result.processedVariables.size() << " 个):\n";
            for (const auto& var : result.processedVariables) {
                std::cout << "    - " << var << "\n";
            }
        } else {
            std::cout << "  ❌ 未处理任何变量\n";
            std::cout << "  💡 检查建议:\n";
            std::cout << "    - 确认变量名 'vo', 'uo' 在NetCDF文件中存在\n";
            std::cout << "    - 检查变量名大小写是否正确\n";
            std::cout << "    - 验证文件是否使用标准NetCDF格式\n";
        }
        
        if (!result.failedVariables.empty()) {
            std::cout << "  ❌ 处理失败变量 (" << result.failedVariables.size() << " 个):\n";
            for (const auto& var : result.failedVariables) {
                std::cout << "    - " << var << "\n";
            }
        }
        
        std::cout << "  📊 变量处理成功率: " << result.getVariableSuccessRate() << "%\n";
        std::cout << "\n";
        
        // 输出文件信息
        std::cout << "📄 输出文件信息:\n";
        if (result.outputLocation.has_value()) {
            std::cout << "  📍 主输出位置: " << result.outputLocation.value() << "\n";
            
            // 🔍 特殊检查："no_data" 结果
            if (result.outputLocation.value() == "no_data") {
                std::cout << "  ⚠️  输出位置显示 'no_data' - 这是工作流未生成实际数据的标识\n";
                std::cout << "  💡 这通常意味着:\n";
                std::cout << "    - 数据读取步骤没有找到匹配的数据\n";
                std::cout << "    - 空间查询范围与数据文件范围不重叠\n";
                std::cout << "    - 数据服务返回了空结果\n";
                std::cout << "  🔧 建议检查:\n";
                std::cout << "    - 文件坐标系统与查询坐标系统是否匹配\n";
                std::cout << "    - 查询点 (116.59°E, 15.9°N) 是否在文件的地理范围内\n";
                std::cout << "    - NetCDF文件的维度和坐标变量是否正确\n";
            } else {
                // 检查文件是否实际存在
                std::cout << "  🔍 检查输出文件是否存在...\n";
                checkOutputFile(result.outputLocation.value());
            }
        } else {
            std::cout << "  ⚠️ 未指定输出位置\n";
        }
        
        if (result.outputFormat.has_value()) {
            std::cout << "  📋 输出格式: " << result.outputFormat.value() << "\n";
            
            // 🔍 特殊检查："none" 格式
            if (result.outputFormat.value() == "none") {
                std::cout << "  ⚠️  输出格式显示 'none' - 这表明工作流没有生成任何格式化输出\n";
            }
        }
        
        // 单独变量输出路径
        if (!result.variableOutputPaths.empty()) {
            std::cout << "  📂 各变量输出文件:\n";
            for (const auto& [var, path] : result.variableOutputPaths) {
                std::cout << "    - " << var << ": " << path << "\n";
                checkOutputFile(path);
            }
        } else {
            std::cout << "  📂 无单独变量输出文件\n";
        }
        
        std::cout << "\n";
        
        // 失败文件列表
        if (!result.failedFiles.empty()) {
            std::cout << "⚠️ 处理失败的文件:\n";
            for (const auto& file : result.failedFiles) {
                std::cout << "  - " << file << "\n";
            }
            std::cout << "\n";
        }
        
        // 🎯 重要提示和下一步建议
        std::cout << "🎯 问题诊断和建议:\n";
        if (!result.success) {
            std::cout << "  ❌ 工作流执行失败 - 检查错误信息\n";
        } else if (result.totalDataPoints == 0) {
            std::cout << "  🔍 工作流执行成功但无数据输出 - 这是当前的主要问题\n";
            std::cout << "  📋 建议的调试步骤:\n";
            std::cout << "    1. 使用ncdump或其他NetCDF工具检查文件结构\n";
            std::cout << "    2. 验证文件中的坐标范围和变量名\n";
            std::cout << "    3. 检查日志中是否有被忽略的错误信息\n";
            std::cout << "    4. 尝试使用更简单的测试用例（如读取文件的任意一点）\n";
            std::cout << "    5. 检查NetCDF库的版本和配置\n";
        } else {
            std::cout << "  ✅ 工作流执行成功并生成了数据\n";
        }
        
        std::cout << "=========================================\n";
        
        // 总结和建议
        if (result.success) {
            std::cout << "🎉 工作流执行成功完成！\n";
            if (result.outputLocation.has_value() && result.outputLocation.value() != "no_data") {
                std::cout << "💡 查看输出文件获取详细数据: " << result.outputLocation.value() << "\n";
            } else {
                std::cout << "⚠️  虽然执行成功，但未生成实际数据输出\n";
                std::cout << "💡 这可能是配置或数据范围问题，请按照上述建议进行排查\n";
            }
        } else {
            std::cout << "⚠️ 工作流执行失败\n";
            std::cout << "💡 问题诊断建议:\n";
            std::cout << "  • 检查输入文件路径和权限\n";
            std::cout << "  • 验证变量名在NetCDF文件中是否存在\n";
            std::cout << "  • 确认查询坐标在数据覆盖范围内\n";
            std::cout << "  • 检查系统资源和磁盘空间\n";
        }
    }

private:
    /**
     * @brief 获取工作流状态的文本描述
     */
    std::string getStatusText(WorkflowStatus status) const {
        switch (status) {
            case WorkflowStatus::NOT_STARTED: return "未开始";
            case WorkflowStatus::INITIALIZING: return "初始化中";
            case WorkflowStatus::RESOLVING_SPATIAL_REQUEST: return "解析空间请求";
            case WorkflowStatus::FINDING_DATA_SOURCES: return "查找数据源";
            case WorkflowStatus::PROCESSING_DATA_SOURCES: return "处理数据源";
            case WorkflowStatus::FUSING_DATA: return "数据融合";
            case WorkflowStatus::POST_PROCESSING: return "后处理";
            case WorkflowStatus::COMPLETED: return "完成";
            case WorkflowStatus::COMPLETED_EMPTY: return "完成但无结果";
            case WorkflowStatus::FAILED: return "失败";
            case WorkflowStatus::CANCELLED: return "已取消";
            default: return "未知状态";
        }
    }
    
    /**
     * @brief 检查输出文件是否存在并显示文件信息
     */
    void checkOutputFile(const std::string& filePath) {
        std::cout << "    🔍 检查文件: " << filePath << "\n";
        
        // 这里可以添加文件系统检查
        // 由于使用的是文件系统API，暂时用简单的提示
        std::cout << "    💡 请手动检查文件是否存在并包含预期数据\n";
        
        // TODO: 可以添加更详细的文件内容检查
        // 比如文件大小、创建时间、前几行内容等
    }

    /**
     * @brief 应用层数据显示方法 - 显示从工作流服务获取的GridData
     */
    void displayGridDataInfo(const core_services::GridData& gridData) {
        std::cout << "\n📊 ===== 海洋数据详细信息 =====" << std::endl;
        std::cout << "📐 数据维度: " << gridData.definition.cols << "x" << gridData.definition.rows 
                  << "x" << gridData.getData().size() << std::endl;
        std::cout << "🌍 空间范围: [" << std::fixed << std::setprecision(3) 
                  << gridData.definition.extent.minX << ", " << gridData.definition.extent.maxX 
                  << "] x [" << gridData.definition.extent.minY << ", " << gridData.definition.extent.maxY 
                  << "]" << std::endl;
        std::cout << "📏 分辨率: " << std::fixed << std::setprecision(6)
                  << gridData.definition.xResolution << " x " << gridData.definition.yResolution << std::endl;
        
        // 显示数据类型
        std::string dataTypeStr;
        switch (gridData.dataType) {
            case core_services::DataType::Float32: dataTypeStr = "Float32"; break;
            case core_services::DataType::Float64: dataTypeStr = "Float64"; break;
            case core_services::DataType::Int32: dataTypeStr = "Int32"; break;
            case core_services::DataType::Int16: dataTypeStr = "Int16"; break;
            default: dataTypeStr = "Unknown"; break;
        }
        std::cout << "🔢 数据类型: " << dataTypeStr << std::endl;
        
        // 显示元数据
        if (!gridData.metadata.empty()) {
            std::cout << "📋 元数据信息:" << std::endl;
            for (const auto& [key, value] : gridData.metadata) {
                std::cout << "  • " << key << ": " << value << std::endl;
            }
        }
        
        // 🎯 诊断数据问题：检查数据类型和大小匹配
        std::cout << "🔍 数据诊断信息:" << std::endl;
        std::cout << "  • 数据缓冲区大小: " << gridData.getData().size() << " bytes" << std::endl;
        std::cout << "  • 数据类型: " << dataTypeStr << std::endl;
        
        // 🎯 检查是否为多变量合并数据
        bool isMergedVariables = false;
        size_t variableCount = 1;
        size_t depthLevels = 50; // 默认50层
        
        // 从元数据中获取变量信息
        if (gridData.metadata.find("merged_variables") != gridData.metadata.end() && 
            gridData.metadata.at("merged_variables") == "true") {
            isMergedVariables = true;
            if (gridData.metadata.find("variable_count") != gridData.metadata.end()) {
                variableCount = std::stoul(gridData.metadata.at("variable_count"));
            }
        }
        
        std::cout << "  • 多变量合并: " << (isMergedVariables ? "是" : "否") << std::endl;
        if (isMergedVariables) {
            std::cout << "  • 变量数量: " << variableCount << std::endl;
        }
        
        // 🎯 根据实际情况修复数据类型解析
        if (gridData.getData().size() == 800) {
            std::cout << "  • 推断：800字节 = 100个double值 (8字节/double)" << std::endl;
            std::cout << "  • 结构：50层深度 × 2变量 = 100个数据点" << std::endl;
            std::cout << "  • 修正：按double类型解析数据" << std::endl;
            
            // 🎯 按double解析800字节数据
            size_t numDoubles = gridData.getData().size() / sizeof(double);
            const double* doubleData = reinterpret_cast<const double*>(gridData.getData().data());
            
            std::cout << "\n🌊 多变量垂直剖面数据 (按double解析):" << std::endl;
            std::cout << "  • 总数据点数: " << numDoubles << " (应为100)" << std::endl;
            
            if (numDoubles == 100 && isMergedVariables && variableCount == 2) {
                // 🎯 正确分离vo和uo变量数据
                std::cout << "\n📊 vo变量 (海洋北向流速) - 前50个数据点:" << std::endl;
                std::vector<double> voValues;
                for (size_t i = 0; i < 50 && i < numDoubles; ++i) {
                    double value = doubleData[i];
                    voValues.push_back(value);
                    if (std::isfinite(value)) {
                        std::cout << "  深度层[" << std::setw(2) << i+1 << "] = " << std::fixed << std::setprecision(6) 
                                  << value << " m/s" << std::endl;
                    } else {
                        std::cout << "  深度层[" << std::setw(2) << i+1 << "] = 无效值" << std::endl;
                    }
                }
                
                std::cout << "\n📊 uo变量 (海洋东向流速) - 后50个数据点:" << std::endl;
                std::vector<double> uoValues;
                for (size_t i = 50; i < 100 && i < numDoubles; ++i) {
                    double value = doubleData[i];
                    uoValues.push_back(value);
                    if (std::isfinite(value)) {
                        std::cout << "  深度层[" << std::setw(2) << (i-49) << "] = " << std::fixed << std::setprecision(6) 
                                  << value << " m/s" << std::endl;
                    } else {
                        std::cout << "  深度层[" << std::setw(2) << (i-49) << "] = 无效值" << std::endl;
                    }
                }
                
                // 🎯 计算各变量统计信息
                std::cout << "\n📈 vo变量统计信息:" << std::endl;
                if (!voValues.empty()) {
                    std::vector<double> voValidValues;
                    for (double val : voValues) {
                        if (std::isfinite(val)) voValidValues.push_back(val);
                    }
                    
                    if (!voValidValues.empty()) {
                        double voMin = *std::min_element(voValidValues.begin(), voValidValues.end());
                        double voMax = *std::max_element(voValidValues.begin(), voValidValues.end());
                        double voSum = std::accumulate(voValidValues.begin(), voValidValues.end(), 0.0);
                        double voMean = voSum / voValidValues.size();
                        
                        std::cout << "  • 有效数据点数: " << voValidValues.size() << " / 50" << std::endl;
                        std::cout << "  • 最小值: " << std::fixed << std::setprecision(6) << voMin << " m/s" << std::endl;
                        std::cout << "  • 最大值: " << std::fixed << std::setprecision(6) << voMax << " m/s" << std::endl;
                        std::cout << "  • 平均值: " << std::fixed << std::setprecision(6) << voMean << " m/s" << std::endl;
                    } else {
                        std::cout << "  • 所有vo数据都无效" << std::endl;
                    }
                }
                
                std::cout << "\n📈 uo变量统计信息:" << std::endl;
                if (!uoValues.empty()) {
                    std::vector<double> uoValidValues;
                    for (double val : uoValues) {
                        if (std::isfinite(val)) uoValidValues.push_back(val);
                    }
                    
                    if (!uoValidValues.empty()) {
                        double uoMin = *std::min_element(uoValidValues.begin(), uoValidValues.end());
                        double uoMax = *std::max_element(uoValidValues.begin(), uoValidValues.end());
                        double uoSum = std::accumulate(uoValidValues.begin(), uoValidValues.end(), 0.0);
                        double uoMean = uoSum / uoValidValues.size();
                        
                        std::cout << "  • 有效数据点数: " << uoValidValues.size() << " / 50" << std::endl;
                        std::cout << "  • 最小值: " << std::fixed << std::setprecision(6) << uoMin << " m/s" << std::endl;
                        std::cout << "  • 最大值: " << std::fixed << std::setprecision(6) << uoMax << " m/s" << std::endl;
                        std::cout << "  • 平均值: " << std::fixed << std::setprecision(6) << uoMean << " m/s" << std::endl;
                    } else {
                        std::cout << "  • 所有uo数据都无效" << std::endl;
                    }
                }
                
            } else {
                // 通用double数据显示
                std::cout << "\n📊 垂直剖面数据 (通用double格式):" << std::endl;
                size_t maxShow = std::min(static_cast<size_t>(50), numDoubles);
                for (size_t i = 0; i < maxShow; ++i) {
                    double value = doubleData[i];
                    if (std::isfinite(value)) {
                        std::cout << "  数据点[" << std::setw(2) << i+1 << "] = " << std::fixed << std::setprecision(6) 
                                  << value << std::endl;
                    } else {
                        std::cout << "  数据点[" << std::setw(2) << i+1 << "] = 无效值" << std::endl;
                    }
                }
            }
            
        } else if (gridData.dataType == core_services::DataType::Float64) {
            // Float64 类型 (8字节) - 正确的数据类型处理
            size_t numDoubles = gridData.getData().size() / sizeof(double);
            std::cout << "  • Float64数据点数: " << numDoubles << std::endl;
            
            const double* doubleData = reinterpret_cast<const double*>(gridData.getData().data());
            std::cout << "\n📊 垂直剖面数据 (Float64):" << std::endl;
            
            size_t maxShow = std::min(static_cast<size_t>(50), numDoubles);
            for (size_t i = 0; i < maxShow; ++i) {
                double value = doubleData[i];
                if (std::isfinite(value)) {
                    std::cout << "  深度层[" << std::setw(2) << i+1 << "] = " << std::fixed << std::setprecision(6) 
                              << value << " m/s" << std::endl;
                } else {
                    std::cout << "  深度层[" << std::setw(2) << i+1 << "] = 无效值" << std::endl;
                }
            }
            
        } else if (gridData.dataType == core_services::DataType::Float32) {
            // 正常的Float32处理
            size_t numFloats = gridData.getData().size() / sizeof(float);
            std::cout << "  • Float32数据点数: " << numFloats << std::endl;
            
            const float* floatData = reinterpret_cast<const float*>(gridData.getData().data());
            std::cout << "\n📊 垂直剖面数据 (Float32):" << std::endl;
            
            size_t maxShow = std::min(static_cast<size_t>(50), numFloats);
            for (size_t i = 0; i < maxShow; ++i) {
                float value = floatData[i];
                if (std::isfinite(value)) {
                    std::cout << "  深度层[" << std::setw(2) << i+1 << "] = " << std::fixed << std::setprecision(6) 
                              << value << " m/s" << std::endl;
                } else {
                    std::cout << "  深度层[" << std::setw(2) << i+1 << "] = 无效值" << std::endl;
                }
            }
            
        } else {
            // 其他数据类型
            std::cout << "  • 未支持的数据类型，显示原始字节数据" << std::endl;
            const auto& dataBuffer = gridData.getData();
            size_t maxShow = std::min(static_cast<size_t>(50), dataBuffer.size());
            for (size_t i = 0; i < maxShow; ++i) {
                std::cout << "  字节[" << std::setw(2) << i << "] = " << static_cast<int>(dataBuffer[i]) << std::endl;
            }
        }
        
        // 🎯 显示深度信息
        std::cout << "\n🌊 深度层信息:" << std::endl;
        
        // 从元数据中查找深度信息
        bool foundDepthInfo = false;
        for (const auto& [key, value] : gridData.metadata) {
            if (key.find("depth") != std::string::npos || key.find("level") != std::string::npos) {
                std::cout << "  • " << key << ": " << value << std::endl;
                foundDepthInfo = true;
            }
        }
        
        if (!foundDepthInfo) {
            std::cout << "  • 深度信息: 未在元数据中找到深度坐标信息" << std::endl;
            std::cout << "  • 建议: 工作流服务应读取NetCDF文件中的depth坐标变量" << std::endl;
            std::cout << "  • 说明: 通常NetCDF文件包含名为'depth'的坐标变量，包含实际深度值(米)" << std::endl;
        }
        
        // 🎯 显示变量信息
        std::cout << "\n🔢 变量信息:" << std::endl;
        for (const auto& [key, value] : gridData.metadata) {
            if (key.find("variable") != std::string::npos) {
                std::cout << "  • " << key << ": " << value << std::endl;
            }
        }
        
        std::cout << "📊 ========================\n" << std::endl;
    }

public:

    /**
     * @brief 运行应用程序主循环
     */
    void run() {
        try {
            showWelcome();
            
            std::cout << "🎯 使用指定参数执行海洋数据查询\n";
            std::cout << "=====================================\n";
            
            // 使用用户指定的实际参数
            QueryParams params;
            params.longitude = 116.59;  // 东经116.59度
            params.latitude = 15.9;     // 北纬15.9度
            params.filePath = "E:/Ocean_data/cs/cs_2023_01_00_00.nc";  // 用户指定文件
            params.variableNames = {"vo", "uo"};  // 查询变量
            
            std::cout << "📋 查询参数:\n";
            std::cout << "📍 位置: " << params.longitude << "°E, " << params.latitude << "°N\n";
            std::cout << "📁 文件: " << params.filePath << "\n";
            std::cout << "🔢 变量: vo, uo (垂直剖面数据)\n";
            std::cout << "=====================================\n";
            
            auto request = createWorkflowRequest(params);
            bool success = executeWorkflowWithTracking(request);
            
            if (success) {
                std::cout << "\n🎉 海洋数据查询成功完成！\n";
                std::cout << "📄 输出文件: ocean_data_query_result.txt\n";
            } else {
                std::cout << "\n⚠️ 海洋数据查询失败\n";
                std::cout << "💡 这是工作流调试的重要信息 - 帮助定位Promise生命周期问题\n";
            }
        
        } catch (const std::exception& e) {
            std::cout << "💥 应用程序异常: " << e.what() << "\n";
        }
    }
};

/**
 * @brief 主函数
 */
int main() {
    try {
        std::cout << "🚀 程序开始启动..." << std::endl;
        
        // 🔧 GDAL全局初始化 - 必须在任何GDAL功能使用前调用
        std::cout << "🌍 初始化GDAL全局环境..." << std::endl;
        oscean::common_utils::infrastructure::GdalGlobalInitializer::getInstance().initialize();
        std::cout << "✅ GDAL全局环境初始化完成" << std::endl;
        
        // 创建应用程序实例
        std::cout << "📦 创建应用程序实例..." << std::endl;
        OceanDataReaderApp app;
        std::cout << "✅ 应用程序实例创建成功" << std::endl;
        
        // 运行应用程序
        std::cout << "🏃 开始运行应用程序..." << std::endl;
        app.run();
        std::cout << "✅ 应用程序运行完成" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "💥 程序启动失败: " << e.what() << "\n";
        std::cout << "🔍 异常类型: " << typeid(e).name() << "\n";
        return 1;
    } catch (...) {
        std::cout << "💥 程序启动失败: 未知异常\n";
        return 1;
    }
} 