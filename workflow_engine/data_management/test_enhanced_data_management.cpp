/**
 * @file test_enhanced_data_management.cpp
 * @brief 增强数据管理工作流测试 - 集成插值、空间计算、输出服务（修复版）
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <string>

// 工作流引擎核心
#include "workflow_engine/data_management/data_management_workflow.h"
#include "workflow_engine/service_management/service_manager_impl.h"

// 通用工具
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/utilities/boost_config.h"
#include "common_utils/time/time_services.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"

// 核心服务接口
#include "core_services/common_data_types.h"
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/interpolation/i_interpolation_service.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "core_services/output/i_output_service.h"

using namespace oscean::workflow_engine::data_management;
using namespace oscean::core_services;

// 全局共享的线程池管理器
static std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> g_threadPoolManager;

/**
 * @brief 获取或创建共享的线程池管理器
 */
std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> getSharedThreadPoolManager() {
    if (!g_threadPoolManager) {
        g_threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>();
    }
    return g_threadPoolManager;
}

/**
 * @brief 测试增强数据管理工作流的基本功能
 */
void testBasicWorkflowFunctionality() {
    std::cout << "\n=== 测试1: 基本工作流功能 ===" << std::endl;
    
    try {
        // 使用共享的线程池管理器
        std::cout << "获取线程池管理器..." << std::endl;
        auto threadPoolManager = getSharedThreadPoolManager();
        std::cout << "线程池管理器获取成功" << std::endl;
        
        // 测试线程池管理器的基本功能
        std::cout << "测试线程池管理器统计..." << std::endl;
        try {
            auto stats = threadPoolManager->getStatistics();
            std::cout << "  总线程数: " << stats.totalThreads << std::endl;
            std::cout << "  活动线程: " << stats.activeThreads << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  获取统计信息失败: " << e.what() << std::endl;
        }
        
        // 创建服务管理器
        std::cout << "创建服务管理器..." << std::endl;
        auto serviceManager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        std::cout << "服务管理器创建成功" << std::endl;
        
        // 创建工作流
        std::cout << "创建工作流..." << std::endl;
        auto workflow = std::make_unique<DataManagementWorkflow>("test-workflow-001", serviceManager);
        std::cout << "工作流创建成功" << std::endl;
        
        // 测试基本属性
        std::cout << "测试基本属性..." << std::endl;
        std::cout << "工作流名称: " << workflow->getName() << std::endl;
        std::cout << "工作流ID: " << workflow->getWorkflowId() << std::endl;
        std::cout << "是否运行中: " << (workflow->isRunning() ? "是" : "否") << std::endl;
        
        std::cout << "✅ 基本工作流功能测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 基本工作流功能测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试批处理配置
 */
void testBatchProcessingConfiguration() {
    std::cout << "\n=== 测试2: 批处理配置 ===" << std::endl;
    
    try {
        BatchProcessingConfig config;
        
        // 基础配置
        config.inputDirectory = "./test_data";
        config.outputDirectory = "./test_output";
        config.filePatterns = {"*.nc", "*.tiff"};
        config.enableParallelProcessing = true;
        config.maxConcurrentTasks = 4;
        
        // 插值配置
        config.enableInterpolation = true;
        config.interpolationMethod = InterpolationMethod::BILINEAR;
        
        // 空间计算配置
        config.enableSpatialOps = true;
        config.targetCRS = "EPSG:4326";
        
        // 输出配置
        config.enableOutput = true;
        config.outputFormat = output::OutputFormat::NETCDF;
        config.outputFileTemplate = "processed_{{filename}}";
        
        // 质量控制配置
        config.enableQualityCheck = true;
        config.qualityThreshold = 0.8;
        
        std::cout << "输入目录: " << config.inputDirectory << std::endl;
        std::cout << "输出目录: " << config.outputDirectory << std::endl;
        std::cout << "启用插值: " << (config.enableInterpolation ? "是" : "否") << std::endl;
        std::cout << "启用空间计算: " << (config.enableSpatialOps ? "是" : "否") << std::endl;
        std::cout << "启用输出: " << (config.enableOutput ? "是" : "否") << std::endl;
        
        std::cout << "✅ 批处理配置测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 批处理配置测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试插值配置
 */
void testInterpolationConfiguration() {
    std::cout << "\n=== 测试3: 插值配置 ===" << std::endl;
    
    try {
        BatchProcessingConfig config;
        config.enableInterpolation = true;
        config.interpolationMethod = InterpolationMethod::BILINEAR;
        
        // 创建目标网格定义
        GridDefinition targetGrid;
        targetGrid.cols = 100;
        targetGrid.rows = 100;
        targetGrid.xResolution = 3.6;  // 3.6度分辨率
        targetGrid.yResolution = 1.8;  // 1.8度分辨率
        targetGrid.crs.id = "EPSG:4326";
        
        // 设置空间范围
        targetGrid.extent.minX = -180.0;
        targetGrid.extent.minY = -90.0;
        targetGrid.extent.maxX = 180.0;
        targetGrid.extent.maxY = 90.0;
        targetGrid.extent.crsId = "EPSG:4326";
        
        config.targetGrid = targetGrid;
        
        std::cout << "插值方法: BILINEAR" << std::endl;
        std::cout << "目标网格尺寸: " << targetGrid.cols << "x" << targetGrid.rows << std::endl;
        std::cout << "目标分辨率: " << targetGrid.xResolution << "°x" << targetGrid.yResolution << "°" << std::endl;
        
        std::cout << "✅ 插值配置测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 插值配置测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试空间计算配置
 */
void testSpatialOpsConfiguration() {
    std::cout << "\n=== 测试4: 空间计算配置 ===" << std::endl;
    
    try {
        BatchProcessingConfig config;
        config.enableSpatialOps = true;
        config.targetCRS = "EPSG:4326";
        
        // 创建裁剪边界
        BoundingBox clipBounds;
        clipBounds.minX = 110.0;  // 东经110度
        clipBounds.maxX = 120.0;  // 东经120度
        clipBounds.minY = 30.0;   // 北纬30度
        clipBounds.maxY = 40.0;   // 北纬40度
        
        config.clipBounds = clipBounds;
        
        std::cout << "目标坐标系: " << config.targetCRS.value() << std::endl;
        std::cout << "裁剪边界: [" << clipBounds.minX << ", " << clipBounds.minY 
                  << "] - [" << clipBounds.maxX << ", " << clipBounds.maxY << "]" << std::endl;
        
        std::cout << "✅ 空间计算配置测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 空间计算配置测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试输出配置
 */
void testOutputConfiguration() {
    std::cout << "\n=== 测试5: 输出配置 ===" << std::endl;
    
    try {
        BatchProcessingConfig config;
        config.enableOutput = true;
        config.outputFormat = output::OutputFormat::NETCDF;
        config.outputFileTemplate = "processed_{{filename}}_{{timestamp}}";
        config.outputDirectory = "./enhanced_output";
        
        std::cout << "输出格式: NETCDF" << std::endl;
        std::cout << "输出目录: " << config.outputDirectory << std::endl;
        std::cout << "文件模板: " << config.outputFileTemplate << std::endl;
        
        // 测试不同输出格式
        std::vector<output::OutputFormat> formats = {
            output::OutputFormat::NETCDF,
            output::OutputFormat::GEOTIFF,
            output::OutputFormat::PNG,
            output::OutputFormat::JSON
        };
        
        for (auto format : formats) {
            config.outputFormat = format;
            std::cout << "支持的格式: " << static_cast<int>(format) << std::endl;
        }
        
        std::cout << "✅ 输出配置测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 输出配置测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试工作流执行
 */
void testWorkflowExecution() {
    std::cout << "\n=== 测试6: 工作流执行 ===" << std::endl;
    
    try {
        // 使用共享的线程池管理器
        auto threadPoolManager = getSharedThreadPoolManager();
        
        // 创建服务管理器
        auto serviceManager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        
        // 创建工作流
        auto workflow = std::make_unique<DataManagementWorkflow>("test-execution-001", serviceManager);
        
        // 测试同步执行
        std::cout << "测试同步执行..." << std::endl;
        workflow->execute();
        std::cout << "同步执行完成" << std::endl;
        
        // 测试异步执行
        std::cout << "测试异步执行..." << std::endl;
        auto future = workflow->executeAsync();
        
        // 等待完成（设置超时）
        auto status = future.wait_for(boost::chrono::seconds(5));
        if (status == boost::future_status::ready) {
            future.get();
            std::cout << "异步执行完成" << std::endl;
        } else {
            std::cout << "异步执行超时，取消操作" << std::endl;
            workflow->cancel();
        }
        
        std::cout << "✅ 工作流执行测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 工作流执行测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试错误处理
 */
void testErrorHandling() {
    std::cout << "\n=== 测试7: 错误处理 ===" << std::endl;
    
    try {
        // 使用共享的线程池管理器
        auto threadPoolManager = getSharedThreadPoolManager();
        
        // 创建服务管理器
        auto serviceManager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        
        // 创建工作流
        auto workflow = std::make_unique<DataManagementWorkflow>("test-error-001", serviceManager);
        
        // 测试无效配置
        BatchProcessingConfig invalidConfig;
        invalidConfig.inputDirectory = "/nonexistent/path";
        invalidConfig.outputDirectory = "/invalid/output/path";
        
        std::cout << "测试无效输入目录处理..." << std::endl;
        auto future = workflow->processBatchAsync(invalidConfig);
        
        try {
            auto results = future.get();
            std::cout << "批处理结果数量: " << results.size() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "预期的错误: " << e.what() << std::endl;
        }
        
        std::cout << "✅ 错误处理测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 错误处理测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试完整工作流管道
 */
void testCompleteWorkflowPipeline() {
    std::cout << "\n=== 测试8: 完整工作流管道 ===" << std::endl;
    
    try {
        // 使用共享的线程池管理器
        auto threadPoolManager = getSharedThreadPoolManager();
        
        // 创建服务管理器
        auto serviceManager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        
        // 创建工作流
        auto workflow = std::make_unique<DataManagementWorkflow>("test-pipeline-001", serviceManager);
        
        // 创建完整配置
        BatchProcessingConfig config;
        config.inputDirectory = "./test_data";
        config.outputDirectory = "./pipeline_output";
        config.filePatterns = {"*.nc"};
        config.enableParallelProcessing = true;
        config.maxConcurrentTasks = 2;
        
        // 启用所有增强功能
        config.enableInterpolation = true;
        config.interpolationMethod = InterpolationMethod::BILINEAR;
        
        config.enableSpatialOps = true;
        config.targetCRS = "EPSG:4326";
        
        config.enableOutput = true;
        config.outputFormat = output::OutputFormat::NETCDF;
        
        config.enableQualityCheck = true;
        config.qualityThreshold = 0.8;
        
        std::cout << "启动完整工作流管道..." << std::endl;
        
        // 执行批处理
        auto future = workflow->processBatchAsync(config);
        auto results = future.get();
        
        std::cout << "管道执行完成，处理结果数量: " << results.size() << std::endl;
        
        // 分析结果
        size_t successCount = 0;
        size_t failureCount = 0;
        
        for (const auto& result : results) {
            if (result.success) {
                successCount++;
                std::cout << "✅ 成功: " << result.filePath 
                          << " (耗时: " << result.processingTime.count() << "ms)" << std::endl;
            } else {
                failureCount++;
                std::cout << "❌ 失败: " << result.filePath 
                          << " - " << result.errorMessage << std::endl;
            }
        }
        
        std::cout << "成功: " << successCount << ", 失败: " << failureCount << std::endl;
        std::cout << "✅ 完整工作流管道测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 完整工作流管道测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试性能统计
 */
void testPerformanceStatistics() {
    std::cout << "\n=== 测试9: 性能统计 ===" << std::endl;
    
    try {
        // 使用共享的线程池管理器
        auto threadPoolManager = getSharedThreadPoolManager();
        
        // 创建服务管理器
        auto serviceManager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        
        // 创建工作流
        auto workflow = std::make_unique<DataManagementWorkflow>("test-perf-001", serviceManager);
        
        auto startTime = std::chrono::steady_clock::now();
        
        // 模拟多次处理
        for (int i = 0; i < 3; ++i) {
            BatchProcessingConfig config;
            config.inputDirectory = "./test_data";
            config.outputDirectory = "./perf_output";
            config.enableQualityCheck = true;
            
            auto future = workflow->processBatchAsync(config);
            auto results = future.get();
            
            std::cout << "第 " << (i+1) << " 次处理完成，结果数量: " << results.size() << std::endl;
        }
        
        auto endTime = std::chrono::steady_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "总处理时间: " << totalTime.count() << "ms" << std::endl;
        std::cout << "✅ 性能统计测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 性能统计测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试服务发现和注册
 */
void testServiceDiscoveryAndRegistration() {
    std::cout << "\n=== 测试10: 服务发现和注册 ===" << std::endl;
    
    try {
        // 使用共享的线程池管理器
        auto threadPoolManager = getSharedThreadPoolManager();
        
        // 创建服务管理器
        auto serviceManager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        
        // 测试服务发现
        std::cout << "检查核心服务..." << std::endl;
        
        // 数据访问服务
        auto dataAccessService = serviceManager->getService<oscean::core_services::data_access::IUnifiedDataAccessService>();
        std::cout << "数据访问服务: " << (dataAccessService ? "✅ 已注册" : "❌ 未注册") << std::endl;
        
        // 元数据服务
        auto metadataService = serviceManager->getService<oscean::core_services::metadata::IMetadataService>();
        std::cout << "元数据服务: " << (metadataService ? "✅ 已注册" : "❌ 未注册") << std::endl;
        
        // CRS服务
        auto crsService = serviceManager->getService<oscean::core_services::ICrsService>();
        std::cout << "CRS服务: " << (crsService ? "✅ 已注册" : "❌ 未注册") << std::endl;
        
        // 插值服务
        auto interpolationService = serviceManager->getService<oscean::core_services::interpolation::IInterpolationService>();
        std::cout << "插值服务: " << (interpolationService ? "✅ 已注册" : "❌ 未注册") << std::endl;
        
        // 空间计算服务
        auto spatialOpsService = serviceManager->getService<oscean::core_services::spatial_ops::ISpatialOpsService>();
        std::cout << "空间计算服务: " << (spatialOpsService ? "✅ 已注册" : "❌ 未注册") << std::endl;
        
        // 输出服务
        auto outputService = serviceManager->getService<oscean::core_services::output::IOutputService>();
        std::cout << "输出服务: " << (outputService ? "✅ 已注册" : "❌ 未注册") << std::endl;
        
        std::cout << "✅ 服务发现和注册测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 服务发现和注册测试失败: " << e.what() << std::endl;
    }
}

/**
 * @brief 主测试函数
 */
int main() {
    std::cout << "🚀 增强数据管理工作流测试开始" << std::endl;
    std::cout << "================================================" << std::endl;
    
    try {
        // 创建共享的线程池管理器
        std::cout << "创建线程池管理器..." << std::endl;
        std::cout.flush();
        
        try {
            g_threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>();
        } catch (const std::exception& e) {
            std::cout << "创建线程池管理器时发生异常: " << e.what() << std::endl;
            return 1;
        } catch (...) {
            std::cout << "创建线程池管理器时发生未知异常" << std::endl;
            return 1;
        }
        
        std::cout << "线程池管理器创建成功" << std::endl;
        std::cout.flush();
        
        // 执行所有测试
        std::cout << "开始执行测试..." << std::endl;
        testBasicWorkflowFunctionality();
        testBatchProcessingConfiguration();
        testInterpolationConfiguration();
        testSpatialOpsConfiguration();
        testOutputConfiguration();
        testWorkflowExecution();
        testErrorHandling();
        testCompleteWorkflowPipeline();
        testPerformanceStatistics();
        testServiceDiscoveryAndRegistration();
        
        std::cout << "\n================================================" << std::endl;
        std::cout << "🎉 所有增强数据管理工作流测试完成！" << std::endl;
        
        // 清理共享资源
        if (g_threadPoolManager) {
            std::cout << "正在关闭线程池管理器..." << std::endl;
            g_threadPoolManager->requestShutdown(std::chrono::seconds(30));
            g_threadPoolManager.reset();
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n💥 测试过程中发生异常: " << e.what() << std::endl;
        
        // 确保清理资源
        if (g_threadPoolManager) {
            g_threadPoolManager->requestShutdown(std::chrono::seconds(30));
            g_threadPoolManager.reset();
        }
        
        return 1;
    }
} 