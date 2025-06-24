# 其他模块调用Common功能示例

## 📋 **1. 数据访问服务调用示例**

### **1.1 服务初始化**

```cpp
// 文件: core_services_impl/data_access_service/src/data_access_service_impl.cpp

class DataAccessServiceImpl {
private:
    std::shared_ptr<CommonServicesFactory> commonServices_;
    CommonServicesFactory::LargeFileServices largeFileServices_;
    std::unique_ptr<PerformanceMonitor::DataAccessMonitor> performanceMonitor_;
    
public:
    // 通过依赖注入获取Common服务
    explicit DataAccessServiceImpl(std::shared_ptr<CommonServicesFactory> commonServices)
        : commonServices_(commonServices)
        , largeFileServices_(commonServices->getLargeFileServices())
        , performanceMonitor_(largeFileServices_.performanceMonitor->createDataAccessMonitor()) {
        
        // 配置大文件处理专用设置
        auto config = LargeFileConfig::createOptimal();
        config.maxMemoryUsageMB = 256;  // 严格内存限制
        config.chunkSizeMB = 16;        // NetCDF优化块大小
        config.enableSIMDOptimization = true;
        
        largeFileServices_.fileProcessor->updateConfig(config);
        
        // 设置性能监控
        largeFileServices_.performanceMonitor->startMonitoring();
    }
    
    // === 大文件读取接口 ===
    
    OSCEAN_FUTURE(GridData) readLargeNetCDFAsync(const std::string& filePath, 
                                                const std::string& variableName) {
        // 创建针对该文件的专用处理器
        auto processor = commonServices_->createFileProcessor(filePath);
        
        // 定义NetCDF数据处理逻辑
        auto netcdfHandler = [this, variableName](const DataChunk& chunk) -> bool {
            // 1. 性能监控
            performanceMonitor_->recordFileRead(chunk.metadata.at("file_name"), 
                                              chunk.size, 
                                              std::chrono::milliseconds(chunk.metadata.count("duration")));
            
            // 2. 解析NetCDF数据块
            auto ncData = parseNetCDFChunk(chunk, variableName);
            
            // 3. 缓存解析结果
            auto cacheKey = generateCacheKey(filePath, chunk.chunkId, variableName);
            largeFileServices_.dataCache->put(cacheKey, ncData.rawData);
            
            // 4. 记录缓存使用
            performanceMonitor_->recordCacheHit("netcdf_data_cache");
            
            return true;
        };
        
        // 启动异步处理
        return processor->processFileAsync(filePath, netcdfHandler)
            .then([this](const ProcessingResult& result) -> GridData {
                if (!result.success) {
                    throw std::runtime_error("NetCDF处理失败: " + result.errorMessage);
                }
                
                // 汇总处理结果，构建GridData
                return assembleGridData(result);
            });
    }
    
    // === 批量文件处理 ===
    
    OSCEAN_FUTURE(std::vector<MetadataInfo>) processBatchFilesAsync(
        const std::vector<std::string>& filePaths) {
        
        // 使用Common的并行处理能力
        auto& asyncFramework = commonServices_->getAsyncFramework();
        
        return asyncFramework.parallelMap(filePaths, 
            [this](const std::string& filePath) -> OSCEAN_FUTURE(MetadataInfo) {
                return extractMetadataAsync(filePath);
            }, 
            4  // 最大4个并发处理
        );
    }
    
private:
    // 解析NetCDF数据块
    NetCDFData parseNetCDFChunk(const DataChunk& chunk, const std::string& variableName) {
        NetCDFData result;
        
        // 1. 检查缓存
        auto cacheKey = generateCacheKey(chunk.metadata.at("file_path"), chunk.chunkId, variableName);
        auto cachedData = largeFileServices_.dataCache->get(cacheKey);
        
        if (cachedData) {
            performanceMonitor_->recordCacheHit("netcdf_parse_cache");
            result.rawData = *cachedData;
            return result;
        }
        
        performanceMonitor_->recordCacheMiss("netcdf_parse_cache");
        
        // 2. 实际解析 (使用SIMD优化)
        {
            OSCEAN_PERFORMANCE_TIMER(*largeFileServices_.performanceMonitor, "netcdf_parsing");
            
            // 使用内存管理器分配解析缓冲区
            auto parseBuffer = largeFileServices_.memoryManager->allocateSIMDAligned(
                chunk.size * 2  // 解析可能需要额外空间
            );
            
            // 调用NetCDF解析库 (这里是具体的解析逻辑)
            result = parseNetCDFChunkData(chunk.data.data(), chunk.size, variableName, parseBuffer);
            
            // 缓存解析结果
            largeFileServices_.dataCache->put(cacheKey, result.rawData);
        }
        
        return result;
    }
    
    // 组装最终的GridData
    GridData assembleGridData(const ProcessingResult& result) {
        GridData gridData;
        
        // 从缓存中收集所有处理好的数据块
        for (size_t chunkId = 0; chunkId < result.totalChunks; ++chunkId) {
            auto cacheKey = generateChunkCacheKey(chunkId);
            auto chunkData = largeFileServices_.dataCache->get(cacheKey);
            
            if (chunkData) {
                gridData.appendChunk(*chunkData);
            }
        }
        
        return gridData;
    }
};
```

## 📊 **2. 空间操作服务调用示例**

```cpp
// 文件: core_services_impl/spatial_ops_service/src/spatial_ops_service_impl.cpp

class SpatialOpsServiceImpl {
private:
    std::shared_ptr<CommonServicesFactory> commonServices_;
    CommonServicesFactory::ComputeServices computeServices_;
    std::unique_ptr<PerformanceMonitor::SpatialOpsMonitor> performanceMonitor_;
    
public:
    explicit SpatialOpsServiceImpl(std::shared_ptr<CommonServicesFactory> commonServices)
        : commonServices_(commonServices)
        , computeServices_(commonServices->getComputeServices())
        , performanceMonitor_(computeServices_.resultCache->createSpatialOpsMonitor()) {
    }
    
    // === SIMD优化的几何运算 ===
    
    OSCEAN_FUTURE(std::vector<GeometryResult>) processGeometriesBatchAsync(
        const std::vector<Geometry>& geometries) {
        
        // 使用Common的异步批处理能力
        return computeServices_.asyncFramework->processBatch(
            geometries.begin(), 
            geometries.end(),
            [this](const Geometry& geometry) -> GeometryResult {
                return processGeometryWithSIMD(geometry);
            },
            10  // 批大小
        );
    }
    
private:
    GeometryResult processGeometryWithSIMD(const Geometry& geometry) {
        GeometryResult result;
        
        // 1. 检查结果缓存
        auto cacheKey = generateGeometryCacheKey(geometry);
        auto cachedResult = computeServices_.resultCache->get(cacheKey);
        
        if (cachedResult) {
            performanceMonitor_->recordCacheHit("geometry_result_cache");
            return *cachedResult;
        }
        
        // 2. 使用SIMD优化计算
        {
            OSCEAN_PERFORMANCE_TIMER(*computeServices_.performanceMonitor, "simd_geometry_processing");
            
            // 分配SIMD对齐的工作内存
            size_t workBufferSize = geometry.getPointCount() * sizeof(Point3D);
            auto workBuffer = computeServices_.memoryManager->allocateSIMDAligned(workBufferSize);
            
            // 应用SIMD优化的几何算法
            if (computeServices_.simdManager->hasAVX2()) {
                result = processGeometryAVX2(geometry, workBuffer);
                performanceMonitor_->recordSIMDUsage("geometry_processing", 2.8);  // 2.8x加速
            } else if (computeServices_.simdManager->hasSSE4_1()) {
                result = processGeometrySSE41(geometry, workBuffer);
                performanceMonitor_->recordSIMDUsage("geometry_processing", 1.6);  // 1.6x加速
            } else {
                result = processGeometryScalar(geometry);
                performanceMonitor_->recordSIMDUsage("geometry_processing", 1.0);  // 无加速
            }
            
            // 释放工作内存
            computeServices_.memoryManager->deallocate(workBuffer);
        }
        
        // 3. 缓存计算结果
        computeServices_.resultCache->put(cacheKey, result);
        
        // 4. 记录性能指标
        performanceMonitor_->recordGeometryProcessing(1, result.processingTime);
        
        return result;
    }
    
    // AVX2优化的几何处理
    GeometryResult processGeometryAVX2(const Geometry& geometry, void* workBuffer) {
        GeometryResult result;
        
        // 获取几何点数据
        const auto& points = geometry.getPoints();
        size_t pointCount = points.size();
        
        // SIMD处理：计算几何中心
        __m256 sumX = _mm256_setzero_ps();
        __m256 sumY = _mm256_setzero_ps();
        __m256 sumZ = _mm256_setzero_ps();
        
        size_t simdCount = pointCount / 8;  // AVX2处理8个float
        
        for (size_t i = 0; i < simdCount; ++i) {
            // 加载8个点的X坐标
            __m256 x = _mm256_load_ps(&points[i * 8].x);
            __m256 y = _mm256_load_ps(&points[i * 8].y);
            __m256 z = _mm256_load_ps(&points[i * 8].z);
            
            sumX = _mm256_add_ps(sumX, x);
            sumY = _mm256_add_ps(sumY, y);
            sumZ = _mm256_add_ps(sumZ, z);
        }
        
        // 水平求和并处理剩余点
        float totalX = horizontalSum(sumX);
        float totalY = horizontalSum(sumY);
        float totalZ = horizontalSum(sumZ);
        
        // 处理剩余点
        for (size_t i = simdCount * 8; i < pointCount; ++i) {
            totalX += points[i].x;
            totalY += points[i].y;
            totalZ += points[i].z;
        }
        
        // 计算中心点
        result.center = Point3D{
            totalX / pointCount,
            totalY / pointCount,
            totalZ / pointCount
        };
        
        result.processingTime = std::chrono::milliseconds(10);  // 示例时间
        return result;
    }
};
```

## 🧮 **3. 插值服务调用示例**

```cpp
// 文件: core_services_impl/interpolation_service/src/interpolation_service_impl.cpp

class InterpolationServiceImpl {
private:
    std::shared_ptr<CommonServicesFactory> commonServices_;
    CommonServicesFactory::ComputeServices computeServices_;
    std::unique_ptr<PerformanceMonitor::InterpolationMonitor> performanceMonitor_;
    
public:
    explicit InterpolationServiceImpl(std::shared_ptr<CommonServicesFactory> commonServices)
        : commonServices_(commonServices)
        , computeServices_(commonServices->getComputeServices())
        , performanceMonitor_(computeServices_.resultCache->createInterpolationMonitor()) {
    }
    
    // === 大规模插值计算 ===
    
    OSCEAN_FUTURE(InterpolationResult) interpolateLargeDatasetAsync(
        const DataGrid& sourceGrid,
        const std::vector<Point>& targetPoints,
        InterpolationMethod method) {
        
        // 1. 检查缓存
        auto cacheKey = generateInterpolationCacheKey(sourceGrid, targetPoints, method);
        auto cachedResult = computeServices_.resultCache->get(cacheKey);
        
        if (cachedResult) {
            performanceMonitor_->recordCacheUsage(1, 0);  // 1个缓存命中
            return computeServices_.asyncFramework->makeReadyFuture(*cachedResult);
        }
        
        // 2. 大数据集分块处理
        if (targetPoints.size() > 10000) {
            return processLargeInterpolationAsync(sourceGrid, targetPoints, method);
        } else {
            return processSmallInterpolationAsync(sourceGrid, targetPoints, method);
        }
    }
    
private:
    OSCEAN_FUTURE(InterpolationResult) processLargeInterpolationAsync(
        const DataGrid& sourceGrid,
        const std::vector<Point>& targetPoints,
        InterpolationMethod method) {
        
        // 分块处理大数据集
        const size_t chunkSize = 1000;  // 每块1000个点
        size_t chunkCount = (targetPoints.size() + chunkSize - 1) / chunkSize;
        
        std::vector<OSCEAN_FUTURE(InterpolationChunkResult)> chunkFutures;
        
        for (size_t i = 0; i < chunkCount; ++i) {
            size_t startIdx = i * chunkSize;
            size_t endIdx = std::min(startIdx + chunkSize, targetPoints.size());
            
            // 创建点的子集
            std::vector<Point> chunkPoints(
                targetPoints.begin() + startIdx,
                targetPoints.begin() + endIdx
            );
            
            // 提交并行处理任务
            auto chunkFuture = computeServices_.threadPoolManager->submitTaskWithResult(
                [this, sourceGrid, chunkPoints, method, i]() -> InterpolationChunkResult {
                    return processInterpolationChunk(sourceGrid, chunkPoints, method, i);
                }
            );
            
            chunkFutures.push_back(std::move(chunkFuture));
        }
        
        // 等待所有块完成并合并结果
        return computeServices_.asyncFramework->whenAll(std::move(chunkFutures))
            .then([this](const std::vector<InterpolationChunkResult>& chunkResults) {
                return mergeInterpolationResults(chunkResults);
            });
    }
    
    InterpolationChunkResult processInterpolationChunk(
        const DataGrid& sourceGrid,
        const std::vector<Point>& chunkPoints,
        InterpolationMethod method,
        size_t chunkId) {
        
        InterpolationChunkResult result;
        result.chunkId = chunkId;
        
        {
            OSCEAN_PERFORMANCE_TIMER(*computeServices_.performanceMonitor, "interpolation_chunk");
            
            // 分配SIMD对齐的工作内存
            size_t workBufferSize = chunkPoints.size() * sizeof(double) * 4;  // 4倍安全系数
            auto workBuffer = computeServices_.memoryManager->allocateSIMDAligned(workBufferSize);
            
            // 根据插值方法选择SIMD优化实现
            switch (method) {
                case InterpolationMethod::BILINEAR:
                    result = performBilinearInterpolationSIMD(sourceGrid, chunkPoints, workBuffer);
                    break;
                case InterpolationMethod::BICUBIC:
                    result = performBicubicInterpolationSIMD(sourceGrid, chunkPoints, workBuffer);
                    break;
                case InterpolationMethod::KRIGING:
                    result = performKrigingInterpolation(sourceGrid, chunkPoints, workBuffer);
                    break;
            }
            
            // 释放工作内存
            computeServices_.memoryManager->deallocate(workBuffer);
        }
        
        // 记录性能指标
        performanceMonitor_->recordInterpolation(
            getMethodName(method),
            chunkPoints.size(),
            result.processingTime
        );
        
        return result;
    }
    
    InterpolationChunkResult performBilinearInterpolationSIMD(
        const DataGrid& sourceGrid,
        const std::vector<Point>& points,
        void* workBuffer) {
        
        InterpolationChunkResult result;
        result.values.reserve(points.size());
        
        // 使用AVX2进行双线性插值
        if (computeServices_.simdManager->hasAVX2()) {
            const size_t simdWidth = 4;  // AVX2处理4个double
            size_t simdCount = points.size() / simdWidth;
            
            for (size_t i = 0; i < simdCount; ++i) {
                // 加载4个插值点
                __m256d x = _mm256_set_pd(points[i*4+3].x, points[i*4+2].x, 
                                         points[i*4+1].x, points[i*4].x);
                __m256d y = _mm256_set_pd(points[i*4+3].y, points[i*4+2].y,
                                         points[i*4+1].y, points[i*4].y);
                
                // 执行SIMD双线性插值
                __m256d interpolatedValues = performBilinearSIMD(sourceGrid, x, y);
                
                // 存储结果
                double values[4];
                _mm256_storeu_pd(values, interpolatedValues);
                
                for (int j = 0; j < 4; ++j) {
                    result.values.push_back(values[j]);
                }
            }
            
            // 处理剩余点
            for (size_t i = simdCount * simdWidth; i < points.size(); ++i) {
                double value = performBilinearScalar(sourceGrid, points[i]);
                result.values.push_back(value);
            }
        } else {
            // 标量回退实现
            for (const auto& point : points) {
                double value = performBilinearScalar(sourceGrid, point);
                result.values.push_back(value);
            }
        }
        
        result.processingTime = std::chrono::milliseconds(50);  // 示例时间
        return result;
    }
};
```

## 🏗️ **4. 应用层统一调用示例**

```cpp
// 文件: workflow_engine/src/workflow_engine_impl.cpp

class WorkflowEngineImpl {
private:
    // 唯一的Common服务工厂实例
    std::shared_ptr<CommonServicesFactory> commonServices_;
    
    // 各个服务实例（通过依赖注入获得）
    std::unique_ptr<DataAccessServiceImpl> dataAccessService_;
    std::unique_ptr<SpatialOpsServiceImpl> spatialOpsService_;
    std::unique_ptr<InterpolationServiceImpl> interpolationService_;
    std::unique_ptr<MetadataServiceImpl> metadataService_;
    std::unique_ptr<CRSServiceImpl> crsService_;
    
public:
    explicit WorkflowEngineImpl(const WorkflowConfig& config) {
        // 1. 创建Common服务工厂 (整个系统的基础)
        commonServices_ = CommonServicesFactory::createForEnvironment(
            config.environment
        );
        
        // 2. 通过依赖注入创建各个服务
        dataAccessService_ = std::make_unique<DataAccessServiceImpl>(commonServices_);
        spatialOpsService_ = std::make_unique<SpatialOpsServiceImpl>(commonServices_);
        interpolationService_ = std::make_unique<InterpolationServiceImpl>(commonServices_);
        metadataService_ = std::make_unique<MetadataServiceImpl>(commonServices_);
        crsService_ = std::make_unique<CRSServiceImpl>(commonServices_);
        
        // 3. 启动全局性能监控
        commonServices_->getPerformanceMonitor().startMonitoring();
        commonServices_->getPerformanceMonitor().setAlertCallback([this](const auto& alert) {
            handlePerformanceAlert(alert);
        });
    }
    
    // === 海洋数据处理工作流 ===
    
    OSCEAN_FUTURE(WorkflowResult) processOceanDataWorkflowAsync(
        const std::string& netcdfFilePath,
        const BoundingBox& targetRegion,
        const std::vector<Point>& interpolationPoints) {
        
        // 获取异步框架以编排工作流
        auto& asyncFramework = commonServices_->getAsyncFramework();
        
        // 第1步：并行读取数据和提取元数据
        auto dataFuture = dataAccessService_->readLargeNetCDFAsync(netcdfFilePath, "temperature");
        auto metadataFuture = metadataService_->extractMetadataAsync(netcdfFilePath);
        
        // 第2步：等待数据和元数据就绪
        return asyncFramework.whenAll(std::move(dataFuture), std::move(metadataFuture))
            .then([this, targetRegion, interpolationPoints](const auto& results) {
                auto [gridData, metadata] = results;
                
                // 第3步：空间操作 - 裁剪到目标区域
                return spatialOpsService_->clipToRegionAsync(gridData, targetRegion);
            })
            .then([this, interpolationPoints](const GridData& clippedData) {
                // 第4步：插值到目标点
                return interpolationService_->interpolateLargeDatasetAsync(
                    clippedData.toDataGrid(),
                    interpolationPoints,
                    InterpolationMethod::BILINEAR
                );
            })
            .then([this](const InterpolationResult& interpolationResult) {
                // 第5步：生成最终结果
                WorkflowResult finalResult;
                finalResult.interpolatedValues = interpolationResult.values;
                finalResult.processingStats = gatherProcessingStatistics();
                return finalResult;
            });
    }
    
private:
    WorkflowStats gatherProcessingStatistics() {
        WorkflowStats stats;
        
        // 从Common服务工厂获取全局统计
        auto systemStats = commonServices_->getSystemStatistics();
        
        stats.totalMemoryUsageMB = systemStats.totalMemoryUsageMB;
        stats.threadPoolUtilization = systemStats.threadPoolUtilization;
        stats.cacheHitRates = systemStats.cacheHitRates;
        stats.averageProcessingSpeedMBps = systemStats.averageProcessingSpeedMBps;
        
        // 生成性能报告
        stats.performanceReport = commonServices_->generateSystemReport();
        
        return stats;
    }
    
    void handlePerformanceAlert(const PerformanceAlert& alert) {
        std::cout << "工作流性能预警: " << alert.toString() << std::endl;
        
        if (alert.level == AlertLevel::CRITICAL) {
            // 关键预警：应用自动优化
            commonServices_->applyAutomaticOptimizations();
        }
    }
    
public:
    ~WorkflowEngineImpl() {
        // 安全关闭所有服务
        commonServices_->shutdown();
    }
};

// === 应用程序入口点 ===

int main() {
    try {
        // 配置工作流
        WorkflowConfig config;
        config.environment = Environment::PRODUCTION;
        
        // 创建工作流引擎
        auto workflowEngine = std::make_unique<WorkflowEngineImpl>(config);
        
        // 定义处理参数
        std::string dataFile = "data/global_ocean_temperature_8gb.nc";
        BoundingBox region{-180.0, -90.0, 180.0, 90.0};  // 全球范围
        
        // 生成插值目标点
        std::vector<Point> targetPoints = generateInterpolationGrid(region, 0.1);  // 0.1度网格
        
        std::cout << "开始处理8GB海洋数据，目标点数: " << targetPoints.size() << std::endl;
        
        // 启动异步处理
        auto resultFuture = workflowEngine->processOceanDataWorkflowAsync(
            dataFile, region, targetPoints
        );
        
        // 等待结果
        auto result = resultFuture.get();
        
        std::cout << "处理完成！" << std::endl;
        std::cout << "插值结果数量: " << result.interpolatedValues.size() << std::endl;
        std::cout << "性能统计:\n" << result.processingStats.performanceReport << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "工作流执行错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## 🎯 **核心优势总结**

### **统一的依赖管理**
- ✅ **单一入口**：CommonServicesFactory是所有模块的唯一依赖
- ✅ **清晰注入**：每个服务明确声明其对Common的依赖
- ✅ **生命周期管理**：工厂负责所有服务的创建和销毁
- ✅ **配置一致性**：所有模块使用相同的基础配置

### **性能优化共享**
- ✅ **SIMD加速**：所有模块共享SIMD优化能力
- ✅ **内存管理**：统一的内存池和压力监控
- ✅ **缓存共享**：智能缓存策略在模块间共享
- ✅ **并行处理**：统一的线程池和异步框架

### **监控和优化**
- ✅ **统一监控**：所有模块的性能指标集中管理
- ✅ **全局优化**：基于整体性能的自动优化建议
- ✅ **预警系统**：跨模块的性能预警和处理
- ✅ **资源协调**：避免模块间的资源竞争

### **开发效率**
- ✅ **接口一致**：所有模块使用相同的Common接口
- ✅ **易于测试**：依赖注入便于单元测试和集成测试
- ✅ **代码复用**：消除重复实现，提高代码质量
- ✅ **维护性**：统一的架构降低维护成本 