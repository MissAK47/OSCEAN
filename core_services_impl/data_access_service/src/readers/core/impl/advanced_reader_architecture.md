# 高级数据读取器架构设计（修正版）

## 🎯 核心原则

**复用common_utilities高级功能，专注数据访问流式处理**

## 📁 推荐架构

```
core_services_impl/data_access_service/src/readers/core/impl/
├── unified_advanced_reader.h              # 统一高级读取器基类
├── unified_advanced_reader.cpp            # 集成common功能 + 流式处理
├── 
├── gdal/                                   # GDAL专用模块
│   ├── gdal_format_handler.h              # GDAL格式处理器
│   ├── gdal_format_handler.cpp            # GDAL具体实现
│   ├── gdal_raster_processor.h            # 栅格专用处理
│   ├── gdal_raster_processor.cpp          
│   ├── gdal_vector_processor.h            # 矢量专用处理
│   └── gdal_vector_processor.cpp          
├── 
├── netcdf/                                 # NetCDF专用模块
│   ├── netcdf_format_handler.h            # NetCDF格式处理器
│   ├── netcdf_format_handler.cpp          # NetCDF具体实现
│   ├── netcdf_variable_processor.h        # 变量专用处理
│   ├── netcdf_variable_processor.cpp      
│   ├── netcdf_time_processor.h            # 时间维度处理
│   └── netcdf_time_processor.cpp          
├── 
└── streaming/                              # 数据访问专用流式处理
    ├── data_streaming_coordinator.h       # 数据流式处理协调器
    ├── data_streaming_coordinator.cpp     
    ├── format_streaming_adapter.h         # 格式流式适配器
    └── format_streaming_adapter.cpp       
```

## 🏗️ 核心设计

### 1. 统一高级读取器基类（集成common功能）

```cpp
#include "common_utils/simd/simd_manager_unified.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/async/async_framework.h"
#include "common_utils/cache/icache_manager.h"
#include "streaming/data_streaming_coordinator.h"

class UnifiedAdvancedReader : public UnifiedDataReader {
public:
    UnifiedAdvancedReader(const std::string& filePath);
    
    // 高级功能接口（直接使用common组件）
    void enableSIMDOptimization(bool enable = true);
    void enableMemoryOptimization(const MemoryConfig& config);
    void enableCaching(const CacheConfig& config);
    
    // 数据访问专用的流式处理
    void enableStreamingMode(const StreamingConfig& config);
    
    // 性能分析（使用common组件）
    PerformanceReport getPerformanceReport() const;
    
protected:
    // 直接使用common_utilities中的高级功能组件
    std::shared_ptr<oscean::common_utils::simd::UnifiedSIMDManager> simdManager_;
    std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager_;
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework_;
    std::shared_ptr<oscean::common_utils::cache::ICacheManager> cacheManager_;
    
    // 数据访问专用的流式处理组件
    std::shared_ptr<DataStreamingCoordinator> streamingCoordinator_;
    
    // 格式特定的处理器接口（由子类实现）
    virtual std::unique_ptr<FormatHandler> createFormatHandler() = 0;
    virtual void configureFormatSpecificOptimizations() = 0;
    
private:
    void initializeCommonComponents();
};
```

### 2. 格式处理器接口（简化版）

```cpp
class FormatHandler {
public:
    virtual ~FormatHandler() = default;
    
    // 格式特定的核心功能
    virtual bool openFile(const std::string& filePath) = 0;
    virtual std::vector<std::string> getVariableNames() = 0;
    virtual std::shared_ptr<GridData> readVariable(const std::string& name) = 0;
    
    // 与流式处理的集成点（其他高级功能由common提供）
    virtual void configureStreamingAdapter(DataStreamingCoordinator& coordinator) = 0;
    
    // 获取格式特定的优化参数
    virtual SIMDOptimizationHints getSIMDHints() const = 0;
    virtual MemoryOptimizationHints getMemoryHints() const = 0;
};
```

### 3. GDAL实现（使用common功能）

```cpp
class GdalAdvancedReader : public UnifiedAdvancedReader {
public:
    GdalAdvancedReader(const std::string& filePath, GdalReaderType type);
    
protected:
    std::unique_ptr<FormatHandler> createFormatHandler() override {
        if (readerType_ == GdalReaderType::RASTER) {
            return std::make_unique<GdalRasterProcessor>(gdalDataset_);
        } else {
            return std::make_unique<GdalVectorProcessor>(gdalDataset_);
        }
    }
    
    void configureFormatSpecificOptimizations() override {
        auto handler = createFormatHandler();
        
        // 配置SIMD优化（使用common的SIMDManager）
        auto simdHints = handler->getSIMDHints();
        simdManager_->configureForDataType(simdHints.dataType);
        simdManager_->setOptimizationLevel(simdHints.level);
        
        // 配置内存优化（使用common的MemoryManager）
        auto memoryHints = handler->getMemoryHints();
        memoryManager_->setChunkSize(memoryHints.optimalChunkSize);
        
        // 配置数据访问专用的流式处理
        handler->configureStreamingAdapter(*streamingCoordinator_);
    }
    
private:
    GDALDataset* gdalDataset_;
    GdalReaderType readerType_;
};
```

### 4. NetCDF实现（使用common功能）

```cpp
class NetCDFAdvancedReader : public UnifiedAdvancedReader {
public:
    NetCDFAdvancedReader(const std::string& filePath);
    
protected:
    std::unique_ptr<FormatHandler> createFormatHandler() override {
        return std::make_unique<NetCDFFormatHandler>(ncid_);
    }
    
    void configureFormatSpecificOptimizations() override {
        auto handler = createFormatHandler();
        
        // 配置SIMD优化（使用common的SIMDManager）
        auto simdHints = handler->getSIMDHints();
        simdManager_->configureForDataType(simdHints.dataType);
        
        // 配置内存优化（使用common的MemoryManager）
        auto memoryHints = handler->getMemoryHints();
        memoryManager_->setChunkSize(memoryHints.optimalChunkSize);
        
        // 配置NetCDF专用的流式处理
        handler->configureStreamingAdapter(*streamingCoordinator_);
    }
    
private:
    int ncid_;
};
```

## 🚀 数据访问专用流式处理

### 数据流式处理协调器

```cpp
class DataStreamingCoordinator {
public:
    DataStreamingCoordinator(
        std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager,
        std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework
    );
    
    // 数据访问专用的流式处理接口
    void configureStreaming(const StreamingConfig& config);
    
    // 格式特定的流式适配
    void configureForGdal(GDALDataset* dataset);
    void configureForNetCDF(int ncid);
    
    // 流式读取（使用common的内存管理和异步框架）
    boost::future<void> streamVariable(
        const std::string& variableName,
        std::function<bool(DataChunk)> processor,
        FormatHandler& formatHandler
    );
    
    // 背压控制（基于common的内存监控）
    bool shouldApplyBackpressure() const;
    void waitForBackpressureRelief();
    
private:
    std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager_;
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework_;
    
    // 流式处理状态
    std::atomic<size_t> activeChunks_{0};
    std::atomic<size_t> maxConcurrentChunks_{4};
};
```

## 📊 优势分析

### ✅ 复用common_utilities的优势
1. **SIMD优化**：直接使用`common_utils::simd::UnifiedSIMDManager`
2. **内存管理**：直接使用`common_utils::memory::UnifiedMemoryManager`
3. **异步处理**：直接使用`common_utils::async::AsyncFramework`
4. **缓存管理**：直接使用`common_utils::cache::ICacheManager`
5. **性能监控**：直接使用common中的性能分析工具

### ✅ 数据访问专用功能
1. **流式处理协调**：专门针对数据文件读取的流式处理逻辑
2. **格式适配**：不同数据格式的流式读取适配
3. **背压控制**：基于数据访问特点的背压机制

### ✅ 架构清晰度
1. **职责明确**：common负责通用高级功能，data_access负责数据专用逻辑
2. **避免重复**：不重新实现已有的高级功能
3. **易于维护**：高级功能的bug修复和优化在common中统一进行

## 🔧 实施建议

1. **第一阶段**：重构现有代码，移除重复的高级功能实现，改为使用common组件
2. **第二阶段**：专注实现数据访问专用的流式处理功能
3. **第三阶段**：创建格式处理器接口，实现GDAL和NetCDF的具体处理器

这样既避免了重复造轮子，又保持了模块化的清晰架构，专注于数据访问领域的核心功能。 