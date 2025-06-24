# GDAL栅格模块完全解耦重构方案

## 🎯 设计原则：服务完全解耦

### 核心理念
- **独立服务原则**: 每个模块作为独立服务，不直接依赖其他服务
- **接口通信**: 模块间只能通过标准化接口进行通信
- **功能内聚**: 每个服务只负责自己核心领域的功能
- **服务自治**: 每个服务可以独立部署、升级和扩展

## 🏗️ 完全解耦的架构设计

### 1. 服务边界重新定义

```
┌─────────────────────────────────────┐
│          应用层 / 客户端              │
└─────────────┬───────────────────────┘
              │ 标准API调用
┌─────────────▼───────────────────────┐
│        服务协调层（可选）             │
└─┬─────┬─────┬─────┬─────┬───────────┘
  │     │     │     │     │
  │     │     │     │     │
┌─▼──┐ ┌▼──┐ ┌▼──┐ ┌▼──┐ ┌▼────────┐
│CRS │ │空间│ │数据│ │插值│ │元数据   │
│服务│ │服务│ │访问│ │服务│ │服务     │
│    │ │    │ │服务│ │    │ │(可选)   │
└────┘ └───┘ └────┘ └────┘ └─────────┘
  ▲      ▲      ▲      ▲       ▲
  │      │      │      │       │
  └──────┴──────┴──────┴───────┘
        标准化接口通信
```

### 2. 各服务独立职责

#### 🗺️ CRS服务 (独立)
**核心职责**: 坐标系统管理
- 坐标系统解析 (WKT, EPSG, PROJ)
- 坐标点转换
- 边界框转换  
- 几何对象转换
- CRS信息查询

**❌ 不负责**: 
- 数据读取
- 空间分析
- 文件I/O

#### 📊 数据访问服务 (独立)
**核心职责**: 原始数据访问
- 文件读取 (NetCDF, GeoTIFF, Shapefile等)
- 原始数据提取
- 基本元数据读取
- 文件格式验证

**❌ 不负责**:
- 坐标转换 (调用CRS服务API)
- 空间分析 (调用空间服务API)
- 复杂数据处理

#### 🧮 空间操作服务 (独立)
**核心职责**: 空间分析和计算
- 几何运算 (缓冲区、相交、并集等)
- 栅格分析 (重采样、统计、代数运算)
- 空间查询
- 拓扑分析

**❌ 不负责**:
- 数据读取 (调用数据访问服务API)
- 坐标转换 (调用CRS服务API)

#### 🔢 插值服务 (独立)
**核心职责**: 数值插值计算
- 各种插值算法
- 重采样算法
- 数值计算优化

**❌ 不负责**:
- 数据读取
- 坐标转换

## 🔧 GDAL模块重构详细方案

### 问题诊断

当前GDAL模块存在的功能重复：
1. **内部重复**: `gdal_raster_processor` 和 `gdal_raster_reader` 功能重叠
2. **外部重复**: 与CRS服务、空间服务功能重复
3. **职责混乱**: 数据访问层实现了处理和转换功能

### 重构目标

#### GDALRasterReader (纯数据访问)
```cpp
namespace oscean::core_services::data_access::readers::impl::gdal {

/**
 * @brief 纯粹的GDAL栅格数据读取器
 * 只负责原始数据访问，不做任何处理
 */
class GDALRasterReader : public GdalBaseReader {
public:
    explicit GDALRasterReader(const std::string& filePath);
    
    // ✅ 保留：原始数据读取
    std::shared_ptr<GridData> readRawData(
        const std::string& variableName,
        const std::optional<BoundingBox>& bounds = std::nullopt) const;
    
    // ✅ 保留：基本信息获取
    std::vector<std::string> getVariableNames() const;
    RasterInfo getRasterInfo() const;
    std::string getProjectionWKT() const; // 只返回原始WKT
    std::array<double, 6> getGeoTransform() const; // 只返回原始变换参数
    
    // ✅ 保留：文件验证
    bool validateFile() const;
    bool variableExists(const std::string& name) const;
    
    // ❌ 移除：所有坐标转换功能
    // ❌ 移除：数据处理功能
    // ❌ 移除：复杂计算功能
};

}
```

#### GDALRasterProcessor (格式转换器)
```cpp
namespace oscean::core_services::data_access::processors::gdal {

/**
 * @brief GDAL格式转换器
 * 专注于GDAL特定格式到标准格式的转换
 */
class GDALRasterProcessor {
public:
    explicit GDALRasterProcessor(GDALDataset* dataset);
    
    // ✅ 保留：GDAL特定处理
    void processNoDataValues(std::vector<double>& data, const RasterBandInfo& info) const;
    void applyScaleAndOffset(std::vector<double>& data, const RasterBandInfo& info) const;
    GridData convertToStandardFormat(const GDALData& gdalData) const;
    
    // ✅ 保留：格式特定信息
    RasterBandInfo extractBandInfo(int bandNumber) const;
    std::vector<MetadataEntry> extractMetadata() const;
    
    // ❌ 移除：坐标转换 (客户端调用CRS服务)
    // ❌ 移除：空间分析 (客户端调用空间服务)
    // ❌ 移除：统计计算 (客户端调用空间服务)
};

}
```

### 服务间通信方式

#### 1. 标准API调用模式
```cpp
// 客户端代码示例：读取栅格数据并转换坐标
class ClientApplication {
private:
    std::shared_ptr<IRawDataAccessService> dataService_;
    std::shared_ptr<ICrsService> crsService_;
    std::shared_ptr<ISpatialOpsService> spatialService_;

public:
    async auto processRasterData(const std::string& filePath, 
                                const BoundingBox& targetBounds) {
        // 1. 通过数据访问服务读取原始数据
        auto rawData = co_await dataService_->readGridVariableSubsetAsync(
            filePath, "temperature", std::nullopt, targetBounds);
        
        // 2. 获取数据的CRS信息
        auto metadata = co_await dataService_->extractFileMetadataSummaryAsync(filePath);
        auto sourceCRS = metadata->spatialReference;
        
        // 3. 通过CRS服务转换坐标
        auto targetCRS = CRSInfo::fromEPSG(4326); // WGS84
        auto transformedBounds = co_await crsService_->transformBoundingBoxAsync(
            targetBounds, targetCRS);
        
        // 4. 通过空间服务进行重采样
        ResampleOptions options;
        options.method = ResamplingMethod::BILINEAR;
        auto resampledData = co_await spatialService_->resample(
            std::make_shared<GridData>(rawData), options);
        
        return resampledData;
    }
};
```

#### 2. 事件驱动模式 (高级)
```cpp
// 服务间通过事件总线通信
class ServiceEventBus {
public:
    // 数据访问服务发布数据读取完成事件
    void publishDataReadComplete(const DataReadEvent& event);
    
    // CRS服务订阅坐标转换请求事件
    void subscribeToCoordinateTransformRequest(
        std::function<void(const CoordinateTransformEvent&)> handler);
};
```

## 📋 重构实施计划

### 第一阶段：内部功能清理

#### 1.1 移除GDAL模块内部重复
```cpp
// 统一ReadRegion定义到core_services
// 移除gdal_raster_processor.h和gdal_raster_reader.h中的重复定义
using ReadRegion = oscean::core_services::data_access::api::ReadRegion;

// 清理重复的calculateReadRegion方法
// 只在GDALRasterProcessor中保留，作为纯粹的像素计算
```

#### 1.2 功能职责重新分配
```cpp
// GDALRasterReader - 只负责读取
class GDALRasterReader {
    // ✅ 保留
    std::shared_ptr<GridData> readDataImpl(...);
    std::vector<std::string> loadVariableNamesImpl();
    bool validateFileImpl();
    
    // ❌ 移除 - 转移到Processor
    // calculateReadRegion -> GDALRasterProcessor
    // 复杂数据处理 -> GDALRasterProcessor
};

// GDALRasterProcessor - 只负责格式转换
class GDALRasterProcessor {
    // ✅ 保留格式特定处理
    void processNoDataValues(...);
    void applyScaleAndOffset(...);
    ReadRegion calculatePixelRegion(...); // 重命名，明确只做像素计算
    
    // ❌ 移除 - 不再负责
    // 坐标转换 -> 客户端调用CRS服务
    // 空间分析 -> 客户端调用空间服务
};
```

### 第二阶段：移除外部依赖

#### 2.1 移除对CRS服务的直接依赖
```cpp
// ❌ 移除：直接注入CRS服务
// class GDALRasterProcessor {
//     std::shared_ptr<ICrsService> crsService_;
// };

// ✅ 替换：提供原始信息，让客户端调用CRS服务
class GDALRasterProcessor {
public:
    // 只提供原始投影信息
    std::string getProjectionWKT() const;
    std::array<double, 6> getGeoTransform() const;
    
    // 客户端负责调用CRS服务进行转换
    // auto crsInfo = crsService->parseFromWKTAsync(processor.getProjectionWKT());
};
```

#### 2.2 移除对空间服务的功能重复
```cpp
// 从GDALRasterProcessor移除空间分析功能
// ❌ 移除：
// - resampleRaster -> 客户端调用空间服务
// - calculateStatistics -> 客户端调用空间服务  
// - getHistogram -> 客户端调用空间服务

// 从SpatialUtils移除坐标转换功能
// ❌ 移除：
// - geoToPixel -> 客户端调用CRS服务
// - pixelToGeo -> 客户端调用CRS服务
```

### 第三阶段：接口标准化

#### 3.1 标准化数据访问接口
```cpp
// 确保所有GDAL功能都通过IRawDataAccessService接口暴露
class RawDataAccessServiceImpl : public IRawDataAccessService {
public:
    // 标准化的栅格数据读取
    boost::future<GridData> readGridVariableSubsetAsync(...) override;
    
    // 标准化的矢量数据读取
    boost::future<std::vector<Feature>> readFeaturesAsync(...) override;
    
    // 标准化的元数据提取
    boost::future<std::optional<FileMetadata>> extractFileMetadataSummaryAsync(...) override;
};
```

#### 3.2 标准化服务工厂
```cpp
// 独立的服务工厂，不相互依赖
class DataAccessServiceFactory {
public:
    static std::unique_ptr<IRawDataAccessService> createService(
        const DataAccessConfig& config); // 不需要其他服务依赖
};

class CrsServiceFactory {
public:
    static std::unique_ptr<ICrsService> createService(
        const CrsConfig& config); // 完全独立
};

class SpatialOpsServiceFactory {
public:
    static std::unique_ptr<ISpatialOpsService> createService(
        const SpatialOpsConfig& config); // 完全独立
};
```

## 🎯 解耦后的使用模式

### 客户端组合服务使用
```cpp
class OceanDataProcessor {
private:
    std::unique_ptr<IRawDataAccessService> dataService_;
    std::unique_ptr<ICrsService> crsService_;
    std::unique_ptr<ISpatialOpsService> spatialService_;

public:
    OceanDataProcessor() {
        // 独立创建各服务
        dataService_ = DataAccessServiceFactory::createService(dataConfig_);
        crsService_ = CrsServiceFactory::createService(crsConfig_);
        spatialService_ = SpatialOpsServiceFactory::createService(spatialConfig_);
    }
    
    async auto processOceanData(const ProcessingRequest& request) {
        // 1. 数据读取
        auto rawData = co_await dataService_->readGridVariableSubsetAsync(
            request.filePath, request.variable, request.timeRange, request.bounds);
        
        // 2. 坐标转换 (如果需要)
        if (request.targetCRS.has_value()) {
            auto sourceCRS = co_await getCRSFromMetadata(request.filePath);
            auto transformedBounds = co_await crsService_->transformBoundingBoxAsync(
                request.bounds, request.targetCRS.value());
            // 使用转换后的边界重新读取数据...
        }
        
        // 3. 空间处理 (如果需要)
        if (request.needsResampling) {
            auto resampledData = co_await spatialService_->resample(
                std::make_shared<GridData>(rawData), request.resampleOptions);
            return resampledData;
        }
        
        return rawData;
    }
};
```

### 服务编排模式
```cpp
// 高级用法：服务编排器
class ServiceOrchestrator {
public:
    template<typename... Services>
    auto orchestrate(Services&&... services) {
        return Pipeline()
            .then([&](auto&& input) { return dataService_.process(input); })
            .then([&](auto&& data) { return crsService_.transform(data); })
            .then([&](auto&& transformed) { return spatialService_.analyze(transformed); });
    }
};
```

## 📊 解耦后的收益

### 1. 架构清晰
- ✅ 每个服务职责单一明确
- ✅ 服务间无直接依赖
- ✅ 可以独立测试和部署
- ✅ 便于水平扩展

### 2. 维护性提升
- ✅ 修改一个服务不影响其他服务
- ✅ 可以独立升级各个服务
- ✅ 更容易进行单元测试
- ✅ Bug修复范围明确

### 3. 扩展性增强
- ✅ 可以插拔式替换服务实现
- ✅ 支持多种后端 (GDAL, 其他库)
- ✅ 可以添加新的服务类型
- ✅ 支持分布式部署

### 4. 性能优化
- ✅ 各服务可以独立优化
- ✅ 避免不必要的服务调用
- ✅ 可以实现服务级别的缓存
- ✅ 支持异步非阻塞调用

## ⚠️ 注意事项

### 1. 服务调用开销
由于服务间通过API调用，可能有一定的调用开销。可以通过以下方式缓解：
- 批量API调用
- 智能缓存
- 连接池
- 异步处理

### 2. 数据传输
服务间需要传输数据，要注意：
- 使用高效的序列化格式
- 避免大数据在服务间传输
- 考虑使用共享内存或数据引用

### 3. 错误处理
分布式服务的错误处理更复杂：
- 需要统一的错误码和异常体系
- 要处理网络失败和服务不可用
- 需要实现重试和降级机制

### 4. 配置管理
多个独立服务的配置管理：
- 统一的配置中心
- 环境特定的配置
- 配置变更的影响范围控制

## 🚀 总结

这个完全解耦的重构方案实现了：

1. **清晰的服务边界**: 每个服务有明确的职责，不依赖其他服务
2. **标准化接口**: 通过标准API进行服务间通信  
3. **独立部署**: 每个服务可以独立开发、测试、部署
4. **高可扩展性**: 支持水平扩展和功能扩展
5. **低耦合高内聚**: 服务内部功能内聚，服务间松耦合

重构后的系统将更加灵活、可维护，并为未来的扩展提供了坚实的架构基础。 