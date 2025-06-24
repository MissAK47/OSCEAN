# GDAL栅格模块重构方案

## 问题分析

### 功能重复问题

#### 1. 文件间重复功能
- **坐标转换**: `gdal_raster_processor` 和 `gdal_raster_reader` 都实现了像素-地理坐标转换
- **读取区域计算**: 两个文件都有 `calculateReadRegion()` 方法
- **数据读取**: 栅格数据读取逻辑分散在两个文件中
- **结构体重复**: `ReadRegion` 结构体在两个文件中重复定义

#### 2. 与spatial_ops_service模块功能重复
- **重采样功能**: GDAL模块和spatial_ops都实现重采样
- **坐标转换**: 与spatial_ops的 `SpatialUtils` 功能重复
- **统计计算**: 与spatial_ops的 `RasterStatistics` 模块重复

#### 3. 🚨 与CRS服务模块功能重复（新发现）
经过分析CRS模块，发现以下严重功能重复：

**坐标变换功能重复**:
- **CRS服务**: `ICrsService::transformPointAsync()`, `ICrsService::transformBoundingBoxAsync()`
- **GDAL Processor**: `pixelToGeo()`, `geoToPixel()`, `calculateReadRegion()`
- **Spatial Ops**: `SpatialUtils::geoToPixel()`, `SpatialUtils::pixelToGeo()`
- **Geometry Engine**: `GeometryEngine::transform()`, `GeometryEngine::transformToEPSG()`

**CRS信息处理重复**:
- **CRS服务**: `parseFromWKTAsync()`, `parseFromEpsgCodeAsync()`, `getProjectionMethodAsync()`
- **GDAL Processor**: `getProjectionWKT()`, CRS信息缓存
- **NetCDF读取器**: `NetCDFCoordinateSystem::detectCRS()`

**空间参考对象重复**:
- **CRS服务**: `createOgrSrsAsync()`, `transformGeometryAsync()`
- **GDAL模块**: 内部OGR对象处理
- **Spatial Ops**: OGR几何对象转换

### 架构问题
- **职责边界不清**: Reader和Processor功能高度重叠
- **分层违反**: 数据访问层实现了处理层功能
- **代码重复**: 相同功能在多处实现
- **🔴 跨模块重复**: CRS、GDAL、Spatial Ops三个模块都实现坐标转换

## 重构原则

### 1. 单一职责原则
- **CRS Service**: 专注坐标系统定义、解析和转换
- **GDALRasterReader**: 专注于数据读取和文件访问
- **GDALRasterProcessor**: 专注于数据格式处理和转换
- **Spatial Ops Service**: 专注于空间计算和分析

### 2. 分层架构
```
Application Layer
├── Spatial Operations (spatial_ops_service) ← 复杂空间分析
├── Data Processing (gdal_raster_processor) ← 格式转换和处理  
├── Data Access (gdal_raster_reader) ← 原始数据访问
└── CRS Service (crs_service) ← 坐标系统和变换
```

### 3. 依赖方向 - 修正版
- **Reader** 不依赖 Processor
- **Processor** 可以使用 Reader 和 **CRS Service**
- **Spatial Ops** 使用 Processor 和 **CRS Service**
- **所有模块** 统一使用 **CRS Service** 进行坐标变换

### 4. 🎯 统一CRS服务使用原则
- **禁止重复实现**: 所有坐标变换必须通过CRS服务
- **统一接口**: 使用CRS服务的异步接口
- **缓存共享**: CRS变换缓存统一管理

## 重构方案

### 文件职责重新定义

#### GDALRasterReader (数据访问层)
**保留功能**:
- ✅ 文件打开和验证
- ✅ 原始数据读取 (`readDataImpl`)
- ✅ 变量名获取 (`loadVariableNamesImpl`) 
- ✅ 基本元数据读取
- ✅ 文件格式验证

**移除功能**:
- ❌ `calculateReadRegion` → 移至GDALRasterProcessor
- ❌ 复杂数据处理逻辑 → 移至GDALRasterProcessor
- ❌ SIMD优化 → 移至common_utilities或processor
- ❌ 统计计算 → 移至spatial_ops_service
- ❌ **坐标转换** → 移至CRS服务

**新增功能**:
- ✅ 简单的原始数据获取接口
- ✅ 流式数据读取基础设施

#### GDALRasterProcessor (数据处理层)
**保留功能**:
- ✅ 波段信息管理
- ✅ 数据类型转换
- ✅ NoData值处理
- ✅ 缩放和偏移应用
- ✅ 格式标准化
- ✅ 读取区域计算（但**不再自己实现坐标转换**）

**移除功能**:
- ❌ `resampleRaster` → 移至spatial_ops_service
- ❌ `calculateStatistics` → 移至spatial_ops_service
- ❌ `getHistogram` → 移至spatial_ops_service
- ❌ 复杂空间分析 → 移至spatial_ops_service
- ❌ **`pixelToGeo()`, `geoToPixel()`** → 使用CRS服务

**修改功能**:
- 🔄 `calculateReadRegion()` → 内部使用CRS服务进行坐标转换
- 🔄 `getProjectionWKT()` → 仅返回原始信息，不做解析

**新增依赖**:
- ✅ 注入CRS服务接口
- ✅ 使用CRS服务进行所有坐标变换

#### CRS Service (坐标系统服务) - 统一坐标变换中心
**核心功能**:
- ✅ 坐标系统解析 (`parseFromWKTAsync`, `parseFromEpsgCodeAsync`)
- ✅ 点坐标转换 (`transformPointAsync`)
- ✅ 边界框转换 (`transformBoundingBoxAsync`)
- ✅ 几何对象转换 (`transformGeometryAsync`)
- ✅ 批量坐标转换 (`transformPointsBatchSIMDAsync`)
- ✅ 流式坐标转换 (`createCoordinateStreamAsync`)
- ✅ 栅格重投影 (`reprojectGridAsync`)

#### Spatial Ops Service (空间分析层)
**保留功能**:
- ✅ 栅格重采样 (`RasterResampling`)
- ✅ 栅格统计 (`RasterStatistics`)  
- ✅ 栅格代数运算
- ✅ 空间分析操作

**移除功能**:
- ❌ **坐标转换逻辑** → 全部使用CRS服务
- ❌ **`SpatialUtils::geoToPixel()`, `SpatialUtils::pixelToGeo()`** → 使用CRS服务
- ❌ **`GeometryEngine::transform()`** → 使用CRS服务

**修改功能**:
- 🔄 所有需要坐标转换的地方改为调用CRS服务

### 🎯 CRS功能重复清理方案

#### 第一阶段：移除重复的坐标转换实现

1. **GDAL模块清理**
```cpp
// 移除 gdal_raster_processor.h 中的：
// ❌ pixelToGeo()
// ❌ geoToPixel() 
// ❌ 内部坐标转换逻辑

// 替换为 CRS 服务调用：
class GDALRasterProcessor {
private:
    std::shared_ptr<oscean::core_services::ICrsService> crsService_;
    
public:
    // 新的实现方式
    boost::future<std::optional<std::pair<double, double>>> 
    pixelToGeoAsync(int x, int y) const {
        return crsService_->transformPointAsync(x, y, pixelCRS_, geoCRS_);
    }
};
```

2. **Spatial Ops模块清理**
```cpp
// 移除 spatial_utils.h 中的：
// ❌ SpatialUtils::geoToPixel()
// ❌ SpatialUtils::pixelToGeo()

// 移除 geometry_engine.cpp 中的：
// ❌ GeometryEngine::transform()
// ❌ GeometryEngine::transformToEPSG()

// 替换为 CRS 服务调用
```

3. **统一CRS信息处理**
```cpp
// 移除各模块中的CRS解析代码：
// ❌ NetCDFCoordinateSystem::detectCRS() 中的解析逻辑
// ❌ GDAL模块中的投影信息缓存
// ❌ 重复的OGR对象创建

// 统一使用CRS服务：
auto crsInfo = crsService_->parseFromWKTAsync(wktString).get();
```

#### 第二阶段：建立统一的坐标转换接口

1. **定义标准转换接口**
```cpp
namespace oscean::core_services::data_access {
    /**
     * @brief 数据访问层的CRS转换适配器
     * 为数据读取器提供简化的CRS转换接口
     */
    class DataAccessCRSAdapter {
    public:
        explicit DataAccessCRSAdapter(std::shared_ptr<ICrsService> crsService);
        
        // 像素-地理坐标转换（GDAL模块专用）
        boost::future<std::pair<double, double>> 
        pixelToGeo(int x, int y, const std::vector<double>& geoTransform) const;
        
        boost::future<std::pair<int, int>> 
        geoToPixel(double x, double y, const std::vector<double>& geoTransform) const;
        
        // 读取区域计算（GDAL模块专用）
        boost::future<ReadRegion> 
        calculateReadRegion(const BoundingBox& bounds, 
                           const CRSInfo& sourceCRS,
                           const CRSInfo& targetCRS) const;
    };
}
```

2. **修改GDAL处理器依赖注入**
```cpp
class GDALRasterProcessor {
private:
    std::shared_ptr<DataAccessCRSAdapter> crsAdapter_;
    
public:
    explicit GDALRasterProcessor(
        GDALDataset* dataset,
        std::shared_ptr<ICrsService> crsService) 
        : dataset_(dataset)
        , crsAdapter_(std::make_shared<DataAccessCRSAdapter>(crsService)) {
    }
    
    // 使用适配器进行坐标转换
    boost::future<std::optional<ReadRegion>> calculateReadRegion(
        const BoundingBox& bounds) const {
        return crsAdapter_->calculateReadRegion(bounds, sourceCRS_, targetCRS_);
    }
};
```

#### 第三阶段：性能优化和缓存统一

1. **统一缓存策略**
- 所有坐标转换缓存由CRS服务管理
- 移除各模块中的重复缓存实现
- 使用CRS服务的高性能批量转换接口

2. **SIMD优化集成**
```cpp
// 使用CRS服务的SIMD优化接口
auto future = crsService_->transformPointsBatchSIMDAsync(
    points, sourceCRS, targetCRS, simdBatchSize);
```

### 结构体和类型统一

#### 移除重复定义
```cpp
// ✅ 统一的ReadRegion定义 - 已移至core_services
namespace oscean::core_services::data_access::api {
    struct ReadRegion {
        // ... 统一定义
    };
}

// ❌ 移除GDAL模块中的重复定义
// ❌ 移除Spatial Ops模块中的坐标转换结构
```

#### 🎯 CRS类型统一
```cpp
// 确保所有模块使用统一的CRS类型
using CRSInfo = oscean::core_services::CRSInfo;
using TransformedPoint = oscean::core_services::TransformedPoint;
using BoundingBox = oscean::core_services::BoundingBox;
```

### 具体重构步骤

#### 第一阶段：移除功能重复

1. **统一ReadRegion定义**
   - ✅ 已在 `core_services/data_access/unified_data_types.h` 中定义
   - 移除两个文件中的重复定义

2. **移除坐标转换功能重复**
   - 移除 `GDALRasterProcessor::pixelToGeo()`, `GDALRasterProcessor::geoToPixel()`
   - 移除 `SpatialUtils::geoToPixel()`, `SpatialUtils::pixelToGeo()`
   - 移除 `GeometryEngine::transform()` 中的坐标转换逻辑

3. **移除CRS解析重复**
   - 统一使用CRS服务的解析接口
   - 移除各模块中的投影信息缓存

#### 第二阶段：建立CRS服务依赖

1. **修改GDAL模块构造函数**
   - 注入CRS服务依赖
   - 使用CRS服务进行所有坐标变换

2. **修改Spatial Ops模块**
   - 注入CRS服务依赖
   - 移除内部坐标转换实现

3. **创建CRS适配器**
   - 为数据访问层提供简化接口
   - 封装常用的坐标转换操作

#### 第三阶段：接口优化和性能提升

1. **异步接口统一**
   - 所有坐标转换使用异步接口
   - 支持批量处理和流式转换

2. **性能优化**
   - 使用CRS服务的SIMD优化
   - 统一的缓存策略
   - 减少重复计算

### 重构后的架构

```
┌─────────────────────────────────────┐
│        Application Layer            │
├─────────────────────────────────────┤
│     Spatial Operations Service      │
│  - RasterResampling                 │
│  - RasterStatistics                 │  
│  - RasterAlgebra                    │
│  ↓ 使用CRS服务                        │
├─────────────────────────────────────┤
│      Data Processing Layer          │
│  GDALRasterProcessor:               │
│  - Format Conversion                │
│  - Data Type Handling               │
│  - NoData Processing                │
│  ↓ 使用CRS服务                        │
├─────────────────────────────────────┤
│       Data Access Layer             │
│  GDALRasterReader:                  │
│  - File I/O Operations              │
│  - Raw Data Reading                 │
│  - Variable Discovery               │
│  - Basic Validation                 │
├─────────────────────────────────────┤
│          CRS Service                │  ← 🎯 统一坐标变换中心
│  - Coordinate Parsing               │
│  - Point/Bbox Transformation        │
│  - Geometry Transformation          │
│  - Batch/Stream Processing          │
│  - SIMD Optimization                │
│  - Caching & Performance            │
└─────────────────────────────────────┘
```

### 🔄 迁移和兼容性

#### 渐进式迁移策略
1. **第一步**: 添加CRS服务依赖，保持原有接口
2. **第二步**: 内部实现切换到CRS服务
3. **第三步**: 移除重复实现，清理接口
4. **第四步**: 性能优化和测试验证

#### 向后兼容保证
- 保持公共接口不变（内部使用CRS服务）
- 逐步废弃重复的API
- 提供迁移指南和示例

### 性能和维护性提升

#### 性能提升
- **消除重复计算**: 统一的坐标转换实现
- **优化缓存策略**: CRS服务统一管理转换缓存
- **SIMD加速**: 统一的高性能批量转换
- **异步处理**: 所有转换都支持异步操作

#### 维护性提升
- **单一职责**: 每个模块有明确的功能边界
- **依赖清晰**: CRS服务作为唯一的坐标转换提供者
- **测试简化**: 统一的转换逻辑便于测试
- **扩展容易**: 新的坐标系统支持只需在CRS服务中添加

#### 风险控制
- **分阶段重构**: 降低迁移风险
- **双轨运行**: 保持兼容期，逐步切换
- **性能基准**: 确保重构后性能不降低
- **充分测试**: 覆盖所有坐标转换场景

## 总结

这次重构将解决以下关键问题：
1. ✅ 消除功能重复，提高代码复用
2. ✅ 明确职责边界，符合单一职责原则  
3. ✅ 遵循分层架构，提高系统可维护性
4. ✅ 统一接口设计，提高系统一致性
5. ✅ 优化性能表现，减少不必要开销
6. ✅ **统一CRS功能，消除跨模块重复**
7. ✅ **建立清晰的依赖关系，提高系统稳定性**

重构后的系统将具有更好的可维护性、可扩展性和性能表现，同时彻底解决了CRS功能在多个模块间的重复实现问题。 