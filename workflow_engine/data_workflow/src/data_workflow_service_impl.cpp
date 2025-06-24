/**
 * @file data_workflow_service_impl.cpp
 * @brief 精简的数据处理工作流服务实现 - 专注于策略选择和服务编排
 * @author OSCEAN Team
 * @date 2024
 */

#include "workflow_engine/data_workflow/data_workflow_service_impl.h"
#include "common_utils/utilities/logging_utils.h"
#include "workflow_engine/data_workflow/data_workflow_types.h"
#include "workflow_engine/data_workflow/enhanced_workflow_types.h"
#include "core_services/metadata/dataset_metadata_types.h"
// 移除有问题的基础设施依赖
// #include "common_utils/infrastructure/unified_thread_pool_manager.h"

#include <variant>
#include <sstream>
#include <iomanip>

namespace oscean::workflow_engine::data_workflow {

// =============================================================================
// 构造函数和基础方法
// =============================================================================

DataWorkflowServiceImpl::DataWorkflowServiceImpl(std::shared_ptr<service_management::IServiceManager> serviceManager)
    : serviceManager_(std::move(serviceManager)) {
    if (!serviceManager_) {
        throw std::invalid_argument("Service manager cannot be null.");
    }
    initialize();
}

DataWorkflowServiceImpl::~DataWorkflowServiceImpl() {
    OSCEAN_LOG_INFO("DataWorkflowService", "Data Workflow Service destroyed.");
}

void DataWorkflowServiceImpl::initialize() {
    OSCEAN_LOG_INFO("DataWorkflowService", "Initializing DataWorkflowService with unified service manager");
    // 服务将在首次使用时懒加载
}

// =============================================================================
// 服务获取辅助方法
// =============================================================================

std::shared_ptr<core_services::spatial_ops::ISpatialOpsService> DataWorkflowServiceImpl::getSpatialOpsService() const {
    if (!spatialOpsService_) {
        spatialOpsService_ = serviceManager_->getService<core_services::spatial_ops::ISpatialOpsService>();
    }
    return spatialOpsService_;
}

std::shared_ptr<core_services::data_access::IUnifiedDataAccessService> DataWorkflowServiceImpl::getDataAccessService() const {
    if (!dataAccessService_) {
        dataAccessService_ = serviceManager_->getService<core_services::data_access::IUnifiedDataAccessService>();
    }
    return dataAccessService_;
}

std::shared_ptr<core_services::metadata::IMetadataService> DataWorkflowServiceImpl::getMetadataService() const {
    if (!metadataService_) {
        metadataService_ = serviceManager_->getService<core_services::metadata::IMetadataService>();
    }
    return metadataService_;
}

std::shared_ptr<core_services::ICrsService> DataWorkflowServiceImpl::getCrsService() const {
    if (!crsService_) {
        crsService_ = serviceManager_->getService<core_services::ICrsService>();
    }
    return crsService_;
}

std::shared_ptr<core_services::interpolation::IInterpolationService> DataWorkflowServiceImpl::getInterpolationService() const {
    if (!interpolationService_) {
        try {
            interpolationService_ = serviceManager_->getService<core_services::interpolation::IInterpolationService>();
        } catch (const std::exception&) {
            // 插值服务是可选的
            return nullptr;
        }
    }
    return interpolationService_;
}

std::shared_ptr<core_services::output::IOutputService> DataWorkflowServiceImpl::getOutputService() const {
    if (!outputService_) {
        try {
            outputService_ = serviceManager_->getService<core_services::output::IOutputService>();
        } catch (const std::exception&) {
            // 输出服务是可选的
            return nullptr;
        }
    }
    return outputService_;
}

// 简化线程池管理 - 移除复杂的基础设施依赖
// std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> DataWorkflowServiceImpl::getThreadPoolManager() const {
//     return serviceManager_->getService<common_utils::infrastructure::UnifiedThreadPoolManager>();
// }

// =============================================================================
// IDataWorkflowService 接口实现
// =============================================================================

std::string DataWorkflowServiceImpl::getWorkflowName() const {
    return m_workflowName;
}

bool DataWorkflowServiceImpl::isReady() const {
    if (!serviceManager_) {
        return false;
    }
    
    try {
        // 检查核心服务是否就绪
        auto spatialOps = getSpatialOpsService();
        auto dataAccess = getDataAccessService();
        auto metadata = getMetadataService();
        auto crs = getCrsService();
        
        return spatialOps && spatialOps->isReady() &&
               dataAccess && 
               metadata && metadata->isReady() &&
               crs;
               
                } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("DataWorkflowService", "Error checking service readiness: {}", e.what());
        return false;
    }
}

boost::future<WorkflowResult> DataWorkflowServiceImpl::executeWorkflowAsync(const WorkflowRequest& request) {
    if (!isReady()) {
        return boost::make_ready_future(WorkflowResult{false, WorkflowStatus::FAILED, "服务未就绪"});
    }
    
    // 使用boost::async替代线程池管理器
    return boost::async(boost::launch::async, [this, request]() -> WorkflowResult {
        auto startTime = std::chrono::steady_clock::now();
        
        try {
            OSCEAN_LOG_INFO("DataWorkflowService", "🚀 开始执行工作流");
            
            // 步骤1：解析空间请求 - 调用空间操作服务
            auto queryGeometry = resolveSpatialRequestAsync(request.spatialRequest).get();
            
            // 步骤2：查找数据源 - 调用元数据服务
            auto dataSources = findDataSourcesAsync(queryGeometry, request).get();
            if (dataSources.empty()) {
                return WorkflowResult{false, WorkflowStatus::COMPLETED_EMPTY, "未找到匹配的数据源"};
            }
            
            // 步骤3：智能数据读取策略选择 (工作流层职责)
            auto readingStrategy = selectOptimalReadingStrategy(request, dataSources);
            OSCEAN_LOG_INFO("DataWorkflowService", "🎯 选择读取策略: {}", readingStrategy.strategyName.c_str());
            
            // 步骤4：执行数据读取 - 调用数据访问服务
            auto rawData = executeDataReadingAsync(dataSources, queryGeometry, request, readingStrategy).get();
            
            // 步骤5：数据处理流水线 - 协调各个服务
            auto processedData = executeProcessingPipelineAsync(rawData, request).get();
            
            // 步骤6：生成输出 - 调用输出服务
            auto result = generateOutputAsync(processedData, request).get();
            
            auto endTime = std::chrono::steady_clock::now();
            result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            OSCEAN_LOG_INFO("DataWorkflowService", "✅ 工作流执行完成，耗时: {} ms", static_cast<long long>(result.duration.count()));
            return result;

        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("DataWorkflowService", "❌ 工作流执行失败: {}", e.what());
            
            WorkflowResult result;
            result.success = false;
            result.status = WorkflowStatus::FAILED;
            result.error = e.what();
            
            auto endTime = std::chrono::steady_clock::now();
            result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            return result;
        }
    });
}

// =============================================================================
// 工作流步骤实现 - 专注于服务编排
// =============================================================================

boost::future<core_services::Geometry> DataWorkflowServiceImpl::resolveSpatialRequestAsync(
    const SpatialRequest& spatialRequest) {
    
    // 使用boost::async替代线程池管理器
    return boost::async(boost::launch::async, [this, spatialRequest]() -> core_services::Geometry {
        OSCEAN_LOG_DEBUG("DataWorkflowService", "解析空间请求");
        
        return std::visit([this](auto&& arg) -> core_services::Geometry {
                using T = std::decay_t<decltype(arg)>;
            
                if constexpr (std::is_same_v<T, Point>) {
                // 调用空间操作服务处理点几何
                core_services::Geometry geom;
                    std::stringstream wkt;
                    wkt << "POINT (" << arg.x << " " << arg.y;
                    if (arg.z) wkt << " " << *arg.z;
                    wkt << ")";
                    geom.wkt = wkt.str();
                    return geom;
                
            } else if constexpr (std::is_same_v<T, BoundingBox>) {
                // 调用空间操作服务处理边界框
                    core_services::Geometry geom;
                    std::stringstream wkt;
                    wkt << "POLYGON (("
                        << arg.minX << " " << arg.minY << ", "
                        << arg.maxX << " " << arg.minY << ", "
                        << arg.maxX << " " << arg.maxY << ", "
                        << arg.minX << " " << arg.maxY << ", "
                        << arg.minX << " " << arg.minY << "))";
                    geom.wkt = wkt.str();
                    return geom;
                
                } else if constexpr (std::is_same_v<T, BearingDistanceRequest>) {
                // 调用空间操作服务计算目标点
                auto spatialOps = getSpatialOpsService();
                auto destPointFuture = spatialOps->calculateDestinationPointAsync(arg.startPoint, arg.bearing, arg.distance);
                auto destPoint = destPointFuture.get();
                
                core_services::Geometry geom;
                    std::stringstream wkt;
                    wkt << "POINT (" << destPoint.x << " " << destPoint.y;
                    if (destPoint.z) wkt << " " << *destPoint.z;
                    wkt << ")";
                    geom.wkt = wkt.str();
                    return geom;
                
                } else {
                throw std::runtime_error("不支持的空间请求类型");
                }
        }, spatialRequest);
    });
}

boost::future<std::vector<std::string>> DataWorkflowServiceImpl::findDataSourcesAsync(
    const core_services::Geometry& queryGeometry, const WorkflowRequest& request) {
    
    // 使用boost::async替代线程池管理器
    return boost::async(boost::launch::async, [this, queryGeometry, request]() -> std::vector<std::string> {
        OSCEAN_LOG_DEBUG("DataWorkflowService", "查找数据源，模式: {}", 
            request.processingMode == ProcessingMode::DIRECT_FILES ? "DIRECT_FILES" : "DATABASE_QUERY");
        
        // 🔧 修复：正确处理 DIRECT_FILES 模式
        if (request.processingMode == ProcessingMode::DIRECT_FILES) {
            std::vector<std::string> filePaths;
            for (const auto& fileSpec : request.directFiles) {
                filePaths.push_back(fileSpec.filePath);
            }
            OSCEAN_LOG_INFO("DataWorkflowService", "直接文件模式：找到 {} 个数据源", filePaths.size());
            return filePaths;
        }
        
        // 数据库查询模式的原有逻辑
        if (request.dataSources) {
            return *request.dataSources;
        }
        
        // 调用空间操作服务获取边界框
        auto spatialOps = getSpatialOpsService();
        auto bboxFuture = spatialOps->getBoundingBoxForGeometry(queryGeometry);
        auto bbox = bboxFuture.get();
        
        // 调用元数据服务查询
        auto metadataService = getMetadataService();
        core_services::metadata::QueryCriteria criteria;
        
        // 设置空间边界
        core_services::metadata::SpatialBounds spatialBounds;
        spatialBounds.minLongitude = bbox.minX;
        spatialBounds.maxLongitude = bbox.maxX;
        spatialBounds.minLatitude = bbox.minY;
        spatialBounds.maxLatitude = bbox.maxY;
        criteria.spatialBounds = spatialBounds;

        // 设置时间范围
        if (request.timeRange) {
            core_services::metadata::TemporalInfo::TimeRange timeRange;
            auto timeToString = [](const std::chrono::system_clock::time_point& tp) -> std::string {
                auto time_t = std::chrono::system_clock::to_time_t(tp);
                std::stringstream ss;
                ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
                return ss.str();
            };
            timeRange.startTime = timeToString(request.timeRange->startTime);
            timeRange.endTime = timeToString(request.timeRange->endTime);
            criteria.timeRange = timeRange;
        }
        
        // 设置变量名
        if (!request.variableNames.empty()) {
            criteria.variablesInclude = request.variableNames;
        }

        auto queryResult = metadataService->queryMetadataAsync(criteria).get();

        std::vector<std::string> filePaths;
        if (queryResult.isSuccess()) {
            for (const auto& entry : queryResult.getData()) {
                filePaths.push_back(entry.filePath);
            }
        }
        
        OSCEAN_LOG_INFO("DataWorkflowService", "数据库查询模式：找到 {} 个数据源", filePaths.size());
        return filePaths;
    });
}

// =============================================================================
// 🚀 智能数据读取策略选择 (工作流层核心职责)
// =============================================================================

IntelligentReadingStrategy DataWorkflowServiceImpl::selectOptimalReadingStrategy(
    const WorkflowRequest& request, const std::vector<std::string>& dataSources) {
    
    IntelligentReadingStrategy strategy;
    
    // 🔧 修复：根据处理模式计算变量数量
    size_t totalFiles = dataSources.size();
    size_t totalVariables = 0;
    
    if (request.processingMode == ProcessingMode::DIRECT_FILES) {
        // 直接文件模式：统计所有文件的变量总数
        for (const auto& fileSpec : request.directFiles) {
            totalVariables += fileSpec.variableNames.size();
        }
    } else {
        // 数据库查询模式：使用全局变量列表
        totalVariables = request.variableNames.size();
    }
    
    bool hasTimeRange = request.timeRange.has_value();
    bool isMultiVariable = totalVariables > 1;
    
    // 估算数据量
    double estimatedDataSizeMB = totalFiles * totalVariables * 50.0; // 简化估算
    
    // 根据数据特征选择策略
    if (estimatedDataSizeMB < 100.0 && totalFiles <= 5) {
        // 小数据集 - 缓存优化策略
        strategy.accessPattern = IntelligentReadingStrategy::AccessPattern::RANDOM_ACCESS;
        strategy.strategyName = "小数据集缓存优化";
        strategy.selectionReasoning = "数据量小，适合随机访问和缓存";
        strategy.performanceConfig.enableCaching = true;
        strategy.performanceConfig.streamingConfig.chunkSizeMB = 4; // 4MB
        strategy.performanceConfig.maxConcurrentOperations = 1;
        
    } else if (estimatedDataSizeMB > 500.0 || totalFiles > 20) {
        // 大数据集 - 流式处理策略
        strategy.accessPattern = IntelligentReadingStrategy::AccessPattern::STREAMING_PROCESSING;
        strategy.strategyName = "大数据集流式处理";
        strategy.selectionReasoning = "数据量大，使用流式处理避免内存溢出";
        strategy.performanceConfig.enableCaching = false;
        strategy.performanceConfig.streamingConfig.chunkSizeMB = 64; // 64MB
        strategy.performanceConfig.maxConcurrentOperations = 4;
        
    } else if (isMultiVariable) {
        // 多变量 - 并行读取策略
        strategy.accessPattern = IntelligentReadingStrategy::AccessPattern::PARALLEL_READING;
        strategy.strategyName = "多变量并行读取";
        strategy.selectionReasoning = "多变量数据，使用并行读取提高效率";
        strategy.performanceConfig.enableCaching = true;
        strategy.performanceConfig.streamingConfig.chunkSizeMB = 16; // 16MB
        strategy.performanceConfig.maxConcurrentOperations = 2;
        
    } else {
        // 标准策略
        strategy.accessPattern = IntelligentReadingStrategy::AccessPattern::SEQUENTIAL_SCAN;
        strategy.strategyName = "标准读取";
        strategy.selectionReasoning = "标准数据集，使用顺序扫描";
        strategy.performanceConfig.enableCaching = true;
        strategy.performanceConfig.streamingConfig.chunkSizeMB = 8; // 8MB
        strategy.performanceConfig.maxConcurrentOperations = 1;
    }
    
    // 设置性能预期
    strategy.performanceExpectation.estimatedProcessingTimeSeconds = estimatedDataSizeMB / 10.0; // 简化估算
    strategy.performanceExpectation.estimatedMemoryUsageMB = estimatedDataSizeMB * 1.5;
    strategy.performanceExpectation.estimatedIOOperations = totalFiles * totalVariables * 2;
    strategy.performanceExpectation.confidenceLevel = 0.8;
    
    return strategy;
}

boost::future<std::vector<std::shared_ptr<core_services::GridData>>> DataWorkflowServiceImpl::executeDataReadingAsync(
    const std::vector<std::string>& dataSources,
    const core_services::Geometry& queryGeometry,
    const WorkflowRequest& request,
    const IntelligentReadingStrategy& strategy) {
    
    // 使用boost::async替代线程池管理器
    return boost::async(boost::launch::async, [this, dataSources, queryGeometry, request, strategy]() -> std::vector<std::shared_ptr<core_services::GridData>> {
        OSCEAN_LOG_DEBUG("DataWorkflowService", "执行数据读取，策略: {}", strategy.strategyName);
        
        std::vector<std::shared_ptr<core_services::GridData>> allData;
        auto dataAccessService = getDataAccessService();
        
        // 获取查询边界框 - 调用空间操作服务
        auto spatialOps = getSpatialOpsService();
        auto bboxFuture = spatialOps->getBoundingBoxForGeometry(queryGeometry);
        auto bbox = bboxFuture.get();
        
                // 🔧 修复：将智能读取逻辑提取为成员方法调用
        // 这样避免了复杂的 lambda 捕获问题
        
        // 🔧 修复：根据处理模式获取正确的变量列表
        auto getVariablesForFile = [&request](const std::string& filePath) -> std::vector<std::string> {
            if (request.processingMode == ProcessingMode::DIRECT_FILES) {
                for (const auto& fileSpec : request.directFiles) {
                    if (fileSpec.filePath == filePath) {
                        return fileSpec.variableNames;
                    }
                }
                return {};
            } else {
                return request.variableNames;
            }
        };
        
        // 根据策略执行智能数据读取
        bool shouldUseParallelReading = (strategy.accessPattern == IntelligentReadingStrategy::AccessPattern::PARALLEL_READING ||
                                       strategy.accessPattern == IntelligentReadingStrategy::AccessPattern::STREAMING_PROCESSING) &&
                                      dataSources.size() > 1;
        if (shouldUseParallelReading) {
            // 并行读取多个文件
            std::vector<boost::future<std::shared_ptr<core_services::GridData>>> futures;
            
            for (const auto& filePath : dataSources) {
                auto variables = getVariablesForFile(filePath);
                for (const auto& variableName : variables) {
                    OSCEAN_LOG_DEBUG("DataWorkflowService", "并行智能读取: {} - {}", filePath, variableName);
                    
                    // 🔧 修复：使用成员方法调用避免 lambda 捕获问题
                    auto task = [this, filePath, variableName, request]() -> std::shared_ptr<core_services::GridData> {
                        return executeSmartDataReadingForFile(filePath, variableName, request);
                    };
                    
                    // 使用boost::async替代线程池管理器
                    futures.push_back(
                        boost::async(boost::launch::async, std::move(task))
                    );
                }
            }
            
            // 收集结果
            for (auto& future : futures) {
                try {
                    auto data = future.get();
                    if (data) {
                        allData.push_back(data);
                    }
                } catch (const std::exception& e) {
                    OSCEAN_LOG_WARN("DataWorkflowService", "并行数据读取失败: {}", e.what());
                }
            }
            
        } else {
            // 串行智能读取
            for (const auto& filePath : dataSources) {
                auto variables = getVariablesForFile(filePath);
                OSCEAN_LOG_DEBUG("DataWorkflowService", "串行智能读取文件: {}，变量数: {}", filePath, variables.size());
                
                for (const auto& variableName : variables) {
                    try {
                        OSCEAN_LOG_DEBUG("DataWorkflowService", "智能读取变量: {} 从文件: {}", variableName, filePath);
                        auto data = executeSmartDataReadingForFile(filePath, variableName, request);
                        if (data) {
                            allData.push_back(data);
                            OSCEAN_LOG_DEBUG("DataWorkflowService", "成功智能读取变量: {}", variableName);
                        } else {
                            OSCEAN_LOG_WARN("DataWorkflowService", "变量 {} 智能读取返回空数据", variableName);
                        }
                    } catch (const std::exception& e) {
                        OSCEAN_LOG_WARN("DataWorkflowService", "智能数据读取失败 {} - {}: {}", filePath, variableName, e.what());
                    }
                }
            }
        }
        
        OSCEAN_LOG_INFO("DataWorkflowService", "成功读取 {} 个数据集", allData.size());
        return allData;
    });
}

boost::future<std::shared_ptr<core_services::GridData>> DataWorkflowServiceImpl::executeProcessingPipelineAsync(
    const std::vector<std::shared_ptr<core_services::GridData>>& rawData,
    const WorkflowRequest& request) {
    
    // 使用boost::async替代线程池管理器
    return boost::async(boost::launch::async, [this, rawData, request]() -> std::shared_ptr<core_services::GridData> {
        OSCEAN_LOG_DEBUG("DataWorkflowService", "执行数据处理流水线");
        
        if (rawData.empty()) {
            return nullptr;
        }
        
        // 步骤1：数据融合 - 调用空间操作服务
        std::shared_ptr<core_services::GridData> fusedData;
        if (rawData.size() == 1) {
            fusedData = rawData[0];
            } else {
            // 调用空间操作服务进行数据镶嵌
            auto spatialOps = getSpatialOpsService();
            std::vector<std::shared_ptr<const core_services::GridData>> constData;
            for (const auto& data : rawData) {
                constData.push_back(data);
            }
            auto mosaicFuture = spatialOps->mosaicRastersAsync(constData);
            fusedData = mosaicFuture.get();
        }
        
        if (!fusedData) {
            OSCEAN_LOG_ERROR("DataWorkflowService", "数据融合失败");
                return nullptr;
            }

        // 步骤2：坐标转换 - 调用CRS服务
        auto processingOptions = request.getEffectiveProcessingOptions();
        if (request.enableCrsTransformation && processingOptions.targetCRS.has_value()) {
            auto crsService = getCrsService();
            core_services::CRSInfo targetCrsInfo;
            targetCrsInfo.id = *processingOptions.targetCRS;
            
            try {
                auto reprojFuture = crsService->reprojectGridAsync(*fusedData, targetCrsInfo);
                auto reprojectedGrid = reprojFuture.get();
                fusedData = std::make_shared<core_services::GridData>(std::move(reprojectedGrid));
                OSCEAN_LOG_INFO("DataWorkflowService", "坐标转换完成: {}", *processingOptions.targetCRS);
                    } catch (const std::exception& e) {
                OSCEAN_LOG_WARN("DataWorkflowService", "坐标转换失败: {}", e.what());
            }
        }
        
        // 步骤3：插值处理 - 调用插值服务
        if (processingOptions.enableAdvancedInterpolation) {
            auto interpolationService = getInterpolationService();
            if (interpolationService && processingOptions.targetSpatialResolution.has_value()) {
                try {
                        using namespace core_services::interpolation;
                        InterpolationRequest interpRequest;
                    interpRequest.sourceGrid = boost::shared_ptr<core_services::GridData>(fusedData.get(), [fusedData](core_services::GridData*){});
                        interpRequest.method = InterpolationMethod::BILINEAR;
                        
                    auto interpFuture = interpolationService->interpolateAsync(interpRequest);
                        auto interpResult = interpFuture.get();
                        
                    if (interpResult.statusCode == 0 && std::holds_alternative<core_services::GridData>(interpResult.data)) {
                        fusedData = std::make_shared<core_services::GridData>(
                            std::move(std::get<core_services::GridData>(interpResult.data)));
                        OSCEAN_LOG_INFO("DataWorkflowService", "插值处理完成");
                    }
                } catch (const std::exception& e) {
                    OSCEAN_LOG_WARN("DataWorkflowService", "插值处理失败: {}", e.what());
                }
            }
        }
        
        return fusedData;
    });
}

boost::future<WorkflowResult> DataWorkflowServiceImpl::generateOutputAsync(
    std::shared_ptr<core_services::GridData> processedData,
    const WorkflowRequest& request) {
    
    // 使用boost::async替代线程池管理器
    return boost::async(boost::launch::async, [this, processedData, request]() -> WorkflowResult {
        OSCEAN_LOG_DEBUG("DataWorkflowService", "生成输出");
        
        WorkflowResult result;
        
        if (!processedData) {
                result.success = false;
                result.status = WorkflowStatus::FAILED;
            result.error = "没有可用的处理数据";
                return result;
            }
            
        // 设置基本统计信息
            result.totalDataPoints = processedData->getWidth() * processedData->getHeight() * processedData->getBandCount();
            result.dataVolumeMB = static_cast<double>(processedData->getSizeInBytes()) / (1024.0 * 1024.0);
            
        // 生成输出文件 - 调用输出服务
            if (!request.outputPath.empty()) {
            auto outputService = getOutputService();
            if (outputService) {
                try {
                    auto outputOptions = request.getEffectiveOutputOptions();
                    
                            core_services::output::WriteOptions writeOptions;
                            writeOptions.overwrite = true;
                            
                            if (outputOptions.format == OutputFormat::NETCDF) {
                                writeOptions.format = "NetCDF";
                            } else if (outputOptions.format == OutputFormat::GEOTIFF) {
                                writeOptions.format = "GTiff";
                            } else {
                        writeOptions.format = "GTiff";
                            }
                            
                    auto outputFuture = outputService->writeGridAsync(processedData, request.outputPath, writeOptions);
                            outputFuture.get();
                            
                    result.outputLocation = request.outputPath;
                    OSCEAN_LOG_INFO("DataWorkflowService", "输出文件生成成功: {}", request.outputPath);
                    
                } catch (const std::exception& e) {
                    OSCEAN_LOG_WARN("DataWorkflowService", "输出文件生成失败: {}", e.what());
                }
            }
        }
        
            result.success = true;
            result.status = WorkflowStatus::COMPLETED;
        result.message = "工作流执行成功";
        
        return result;
    });
}

// 重复定义已删除 - 使用第325行的定义

std::shared_ptr<core_services::GridData> DataWorkflowServiceImpl::executeSmartDataReadingForFile(
    const std::string& filePath,
    const std::string& variableName,
    const WorkflowRequest& request) {
    
    try {
        auto dataAccessService = getDataAccessService();
        
        // 🎯 创建统一数据请求
        core_services::data_access::api::UnifiedDataRequest dataRequest;
        dataRequest.requestType = core_services::data_access::api::UnifiedRequestType::GRID_DATA;
        dataRequest.filePath = filePath;
        dataRequest.variableName = variableName;
        dataRequest.includeMetadata = true;
        
        // 🎯 设置空间边界（如果是区域查询）
        if (std::holds_alternative<BoundingBox>(request.spatialRequest)) {
            auto bbox = std::get<BoundingBox>(request.spatialRequest);
            core_services::BoundingBox spatialBounds;
            spatialBounds.minX = bbox.minX;
            spatialBounds.maxX = bbox.maxX;
            spatialBounds.minY = bbox.minY;
            spatialBounds.maxY = bbox.maxY;
            dataRequest.spatialBounds = spatialBounds;
        }
        
        // 🎯 设置时间范围
        if (request.timeRange.has_value()) {
            core_services::TimeRange timeRange;
            timeRange.startTime = request.timeRange->startTime;
            timeRange.endTime = request.timeRange->endTime;
            dataRequest.timeRange = timeRange;
        }
        
        // 🎯 执行数据读取
        auto response = dataAccessService->processDataRequestAsync(dataRequest).get();
        
        if (response.status == core_services::data_access::api::UnifiedResponseStatus::SUCCESS) {
            if (std::holds_alternative<std::shared_ptr<core_services::GridData>>(response.data)) {
                return std::get<std::shared_ptr<core_services::GridData>>(response.data);
            }
        }
        
        OSCEAN_LOG_WARN("DataWorkflowService", "智能数据读取失败: {} - {}", filePath, variableName);
        return nullptr;
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("DataWorkflowService", "智能数据读取异常: {} - {}: {}", 
                         filePath, variableName, e.what());
        return nullptr;
    }
}

} // namespace oscean::workflow_engine::data_workflow 