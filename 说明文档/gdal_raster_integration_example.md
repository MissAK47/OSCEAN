# GDAL栅格模块集成示例

## 重构后的架构

### 模块职责分工

```cpp
┌─────────────────────────────────────┐
│     Spatial Operations Service      │ ← 复杂空间分析
│  - RasterResampling                 │
│  - RasterStatistics                 │
│  - RasterAlgebra                    │
├─────────────────────────────────────┤
│      GDALRasterProcessor            │ ← 格式转换和处理
│  - Format Conversion                │
│  - NoData Processing                │
│  - Scale/Offset Application         │
│  - Basic Coordinate Transform       │
├─────────────────────────────────────┤
│       GDALRasterReader              │ ← 原始数据访问
│  - File I/O Operations              │
│  - Raw Data Reading                 │
│  - Variable Discovery               │
│  - Basic Validation                 │
└─────────────────────────────────────┘
```

## 使用示例

### 1. 基础数据读取

```cpp
#include "gdal_raster_reader.h"
#include "gdal_raster_processor.h"

// 数据访问层：读取原始数据
auto reader = std::make_unique<GdalRasterReader>("raster.tif");
if (!reader->validateFile()) {
    throw std::runtime_error("Invalid raster file");
}

auto variableNames = reader->getVariableNames();
std::cout << "Available variables: ";
for (const auto& name : variableNames) {
    std::cout << name << " ";
}
std::cout << std::endl;
```

### 2. 数据处理和转换

```cpp
// 数据处理层：转换格式并处理GDAL特性
auto processor = std::make_unique<GDALRasterProcessor>(reader->getDataset());

// 获取波段信息
auto bandInfo = processor->getBandInfo(1);
if (bandInfo.has_value()) {
    std::cout << "Band description: " << bandInfo->description << std::endl;
    std::cout << "Has NoData: " << bandInfo->hasNoDataValue << std::endl;
    std::cout << "Scale factor: " << bandInfo->scaleFactor << std::endl;
}

// 读取数据并应用GDAL特定处理
oscean::core_services::BoundingBox bounds{10.0, 20.0, 30.0, 40.0};
auto gridData = processor->readBandData(1, bounds);

if (gridData) {
    std::cout << "Data size: " << gridData->data.size() << std::endl;
    std::cout << "Dimensions: " << gridData->width << "x" << gridData->height << std::endl;
}
```

### 3. 空间分析操作

```cpp
#include "core_services/spatial_ops/raster_engine.h"

// 空间分析层：复杂操作
auto spatialEngine = std::make_unique<RasterEngine>(config);

// 重采样操作
oscean::core_services::GridDefinition targetGrid;
targetGrid.xResolution = 0.1;
targetGrid.yResolution = 0.1;
targetGrid.extent = bounds;

oscean::core_services::spatial_ops::ResampleOptions resampleOpts;
resampleOpts.method = oscean::core_services::spatial_ops::ResamplingMethod::BILINEAR;

auto resampledData = spatialEngine->resampleRaster(*gridData, targetGrid, resampleOpts);

// 统计计算
auto stats = spatialEngine->calculateStatistics(resampledData);
std::cout << "Min: " << stats.min << ", Max: " << stats.max << std::endl;
```

## 协同工作流程

### 完整的数据处理管道

```cpp
class RasterDataPipeline {
private:
    std::unique_ptr<GdalRasterReader> reader_;
    std::unique_ptr<GDALRasterProcessor> processor_;
    std::unique_ptr<RasterEngine> spatialEngine_;

public:
    RasterDataPipeline(const std::string& filePath, const SpatialOpsConfig& config)
        : reader_(std::make_unique<GdalRasterReader>(filePath))
        , processor_(std::make_unique<GDALRasterProcessor>(reader_->getDataset()))
        , spatialEngine_(std::make_unique<RasterEngine>(config)) {
    }
    
    /**
     * @brief 完整的数据处理流程
     */
    ProcessedRasterData processRaster(const RasterProcessingRequest& request) {
        ProcessedRasterData result;
        
        // 第1层：数据访问 - 验证和基础信息
        if (!reader_->validateFile()) {
            throw std::runtime_error("Invalid file: " + request.filePath);
        }
        
        auto variables = reader_->getVariableNames();
        if (std::find(variables.begin(), variables.end(), request.variableName) == variables.end()) {
            throw std::runtime_error("Variable not found: " + request.variableName);
        }
        
        // 第2层：数据处理 - 格式转换和GDAL特性处理
        auto bandInfo = processor_->getBandInfo(request.bandNumber);
        if (!bandInfo.has_value()) {
            throw std::runtime_error("Band not found: " + std::to_string(request.bandNumber));
        }
        
        // 计算读取区域（使用统一的ReadRegion）
        std::optional<ReadRegion> region;
        if (request.spatialBounds.has_value()) {
            region = processor_->calculateReadRegion(request.spatialBounds.value());
        }
        
        // 读取并处理数据
        auto gridData = processor_->readBandData(request.bandNumber, request.spatialBounds);
        if (!gridData) {
            throw std::runtime_error("Failed to read raster data");
        }
        
        result.rawData = gridData;
        result.bandInfo = bandInfo.value();
        
        // 第3层：空间分析 - 复杂操作
        if (request.needsResampling) {
            result.resampledData = spatialEngine_->resampleRaster(
                *gridData, request.targetGrid, request.resampleOptions);
        }
        
        if (request.calculateStatistics) {
            result.statistics = spatialEngine_->calculateStatistics(
                result.resampledData ? *result.resampledData : *gridData);
        }
        
        if (request.needsReprojection) {
            auto dataToReproject = result.resampledData ? result.resampledData : gridData;
            result.reprojectedData = spatialEngine_->reprojectRaster(
                *dataToReproject, request.targetCRS, request.resampleOptions);
        }
        
        return result;
    }
    
    /**
     * @brief 流式处理大文件
     */
    void processStreamingRaster(const StreamingRequest& request) {
        auto processor = [this, &request](const std::vector<double>& data, 
                                        int x, int y, int width, int height) -> bool {
            // 创建临时GridData
            auto tempGrid = std::make_shared<oscean::core_services::GridData>();
            tempGrid->data = data;
            tempGrid->width = width;
            tempGrid->height = height;
            
            // 应用空间操作
            if (request.applyFilter) {
                // 使用spatial_ops_service进行滤波
                auto filtered = spatialEngine_->applyFilter(*tempGrid, request.filter);
                return request.chunkProcessor(filtered);
            } else {
                return request.chunkProcessor(*tempGrid);
            }
        };
        
        // 启动流式读取
        auto future = reader_->streamRasterDataAsync(
            request.variableName, request.bounds, processor);
        
        future.wait(); // 等待完成
    }
};
```

## 性能优化

### 内存管理

```cpp
// 使用统一的内存管理策略
class OptimizedRasterProcessor {
private:
    std::unique_ptr<GDALRasterProcessor> processor_;
    oscean::common_utils::memory::MemoryManager memoryManager_;
    
public:
    auto processWithMemoryOptimization(const RasterRequest& request) {
        // 获取处理器的内存建议
        auto memHints = processor_->getMemoryHints(request.bandNumber);
        
        // 配置内存管理器
        memoryManager_.configurePool(memHints);
        
        // 检查内存可用性
        if (!processor_->checkMemoryUsage(memHints.expectedDataSize)) {
            // 使用分块处理
            return processInChunks(request);
        } else {
            // 直接处理
            return processor_->readBandData(request.bandNumber, request.bounds);
        }
    }
};
```

### SIMD优化集成

```cpp
// 集成SIMD优化
class SIMDOptimizedPipeline {
public:
    auto processWithSIMD(const RasterRequest& request) {
        // 获取SIMD优化建议
        auto simdHints = processor_->getSIMDHints(request.bandNumber);
        
        // 配置SIMD处理器
        auto simdProcessor = oscean::common_utils::simd::createOptimizedProcessor(simdHints);
        
        // 读取数据
        auto data = processor_->readBandData(request.bandNumber, request.bounds);
        
        // 应用SIMD优化处理
        if (simdProcessor && data) {
            simdProcessor->optimizeDataProcessing(data->data);
        }
        
        return data;
    }
};
```

## 错误处理和日志

### 统一错误处理

```cpp
class RasterProcessingError : public std::runtime_error {
public:
    enum class ErrorType {
        FILE_ACCESS,
        FORMAT_CONVERSION, 
        SPATIAL_OPERATION,
        MEMORY_ALLOCATION
    };
    
    RasterProcessingError(ErrorType type, const std::string& msg)
        : std::runtime_error(msg), type_(type) {}
    
    ErrorType getType() const { return type_; }
    
private:
    ErrorType type_;
};

class ErrorHandlingPipeline {
public:
    auto safeProcessRaster(const RasterRequest& request) -> std::expected<GridData, RasterProcessingError> {
        try {
            // 数据访问层错误
            if (!reader_->validateFile()) {
                return std::unexpected(RasterProcessingError(
                    RasterProcessingError::ErrorType::FILE_ACCESS, 
                    "文件访问失败: " + request.filePath));
            }
            
            // 数据处理层错误
            auto gridData = processor_->readBandData(request.bandNumber, request.bounds);
            if (!gridData) {
                return std::unexpected(RasterProcessingError(
                    RasterProcessingError::ErrorType::FORMAT_CONVERSION,
                    "格式转换失败"));
            }
            
            // 空间操作层错误
            if (request.needsProcessing) {
                auto processed = spatialEngine_->processRaster(*gridData, request.options);
                return processed;
            }
            
            return *gridData;
            
        } catch (const std::bad_alloc& e) {
            return std::unexpected(RasterProcessingError(
                RasterProcessingError::ErrorType::MEMORY_ALLOCATION,
                "内存分配失败: " + std::string(e.what())));
        } catch (const std::exception& e) {
            return std::unexpected(RasterProcessingError(
                RasterProcessingError::ErrorType::SPATIAL_OPERATION,
                "空间操作失败: " + std::string(e.what())));
        }
    }
};
```

## 总结

重构后的GDAL栅格模块具有以下优势：

### ✅ 解决的问题
1. **消除功能重复**: 统一ReadRegion定义，整合坐标转换功能
2. **明确职责边界**: Reader专注数据访问，Processor专注格式转换，SpatialOps专注分析
3. **提高代码复用**: 避免在多个地方实现相同功能
4. **改善维护性**: 清晰的分层架构便于测试和维护

### ✅ 性能提升
1. **减少重复计算**: 统一的坐标转换和区域计算
2. **优化内存使用**: 更好的内存管理和缓存策略
3. **SIMD集成**: 统一的性能优化策略

### ✅ 架构改进
1. **分层清晰**: 每层有明确的职责和接口
2. **松耦合**: 层间依赖关系清晰，便于测试和替换
3. **可扩展**: 新功能可以在合适的层次添加 