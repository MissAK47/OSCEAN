# OSCEAN 空间服务实施最终版

**版本**: 2.0  
**日期**: 2025年5月24日  0900
**状态**: 实施准备阶段

## 0. 设计总结与核心定位

经过与OSCEAN顶层需求和设计文档的深度对比分析，重新明确空间服务的准确定位和职责边界。

### 0.1 OSCEAN架构中的准确定位

```
OSCEAN分层架构:
┌─────────────────────────────────────────────────────────┐
│  Layer 4: 输出生成与交付层 (Output Generation Layer)   │
│  ├── 瓦片服务 (Tile Service)                          │
│  ├── 文件生成器 (File Generator)                       │
│  └── 图像生成器 (Image Generator)                      │
└─────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────┐
│  Layer 3: 核心服务层 (Core Services Layer)             │
│  ├── 3a. 元数据与索引服务 (Metadata) ✓                │
│  ├── 3b. 原始数据访问服务 (Data Access) ✓              │
│  ├── 3c. 空间处理 (Spatial Ops) ← 当前设计目标        │
│  ├── 3d. 插值服务 (Interpolation) ⏳                  │
│  ├── 3e. 模型计算服务 (Modeling) ⏳                   │
│  └── 3f. 坐标转换 (CRS Engine) ✓                      │
└─────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────┐
│  Layer 2: 任务调度与工作流引擎                          │
└─────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────┐
│  Layer 1: 网络服务层                                   │
└─────────────────────────────────────────────────────────┘
```

**关键认识**：空间服务是Layer 3的**基础计算服务**，不是应用层业务逻辑！

### 0.2 OSCEAN需求文档中的空间功能需求

基于需求文档分析，空间服务需要支持的**明确功能**：

```
直接需求（必须实现）:
✅ FR-PROC-001: CRS坐标转换功能（已在CRS服务中实现）
✅ FR-PROC-004: 空间处理操作
   - 点、线、面要素的空间查询
   - 使用Shapefile对栅格数据进行掩膜 (Masking)
✅ FR-OUT-002: 瓦片服务坐标重投影支持

间接需求（支持其他服务）:
⚪ 为插值服务提供空间约束计算
⚪ 为瓦片服务提供空间数据预处理
⚪ 为建模服务提供空间几何运算
```

**重要发现**：需求文档中**没有要求**复杂的空间分析、工作流编排、高级拓扑分析等功能！

### 0.3 正确的职责边界

```cpp
// ✅ 空间服务应该做什么（基础计算）
namespace oscean::core_services::spatial_ops {

class ISpatialOpsService {
    // 基础几何运算
    virtual Geometry computeBuffer(const Geometry& geom, double distance) = 0;
    virtual bool intersects(const Geometry& g1, const Geometry& g2) = 0;
    
    // 空间查询和过滤
    virtual std::vector<Feature> spatialQuery(
        const std::vector<Feature>& features,
        const Geometry& queryGeometry) = 0;
    
    // 栅格掩膜操作（FR-PROC-004）
    virtual GridData applyMask(
        const GridData& raster,
        const std::vector<Feature>& maskFeatures) = 0;
    
    // 基础栅格操作
    virtual GridData resampleRaster(
        const GridData& input,
        const GridDefinition& targetGrid) = 0;
    
    // 为其他服务提供的工具函数
    virtual BoundingBox calculateTileBounds(int z, int x, int y) = 0;
    virtual std::vector<Point> findPointsInPolygon(
        const std::vector<Point>& points,
        const Polygon& polygon) = 0;
};

}

// ❌ 空间服务不应该做什么（应用层功能）
// - 复杂的空间分析工作流
// - 多步骤的叠加分析
// - 空间数据质量控制
// - 高级拓扑分析
// - 空间统计分析
// - 空间建模和预测
```

## 1. 空间服务核心职责定义

### 1.1 服务职责边界

**核心职责**：为OSCEAN系统提供**基础空间计算能力**

```cpp
namespace oscean::core_services::spatial_ops {

/**
 * @brief 空间服务核心职责定义
 * 
 * 定位：Layer 3核心服务层的基础计算服务
 * 职责：提供空间几何运算、查询、转换等基础计算能力
 * 服务对象：插值服务、瓦片服务、建模服务等其他核心服务
 */
class ISpatialOpsService {
public:
    // ============ 核心功能区域1：基础几何运算 ============
    
    /**
     * @brief 几何缓冲区计算
     * 为插值服务、建模服务等提供基础几何运算支持
     */
    virtual std::future<Geometry> computeBuffer(
        const Geometry& geometry,
        double distance,
        const BufferOptions& options = BufferOptions()) = 0;
    
    /**
     * @brief 几何相交检测
     * 为空间查询和过滤提供基础判断能力
     */
    virtual std::future<bool> intersects(
        const Geometry& geom1,
        const Geometry& geom2) = 0;
    
    /**
     * @brief 几何包含检测
     */
    virtual std::future<bool> contains(
        const Geometry& container,
        const Geometry& contained) = 0;
    
    // ============ 核心功能区域2：空间查询（FR-PROC-004）============
    
    /**
     * @brief 空间要素查询
     * 实现FR-PROC-004：空间查询功能
     */
    virtual std::future<std::vector<Feature>> spatialQuery(
        const std::vector<Feature>& features,
        const Geometry& queryGeometry,
        SpatialPredicate predicate = SpatialPredicate::INTERSECTS) = 0;
    
    /**
     * @brief 点在多边形内判断
     * 为插值服务提供空间约束判断
     */
    virtual std::future<std::vector<bool>> pointsInPolygon(
        const std::vector<Point>& points,
        const Polygon& polygon) = 0;
    
    // ============ 核心功能区域3：栅格掩膜（FR-PROC-004）============
    
    /**
     * @brief 使用矢量要素对栅格数据进行掩膜
     * 实现FR-PROC-004：使用Shapefile对栅格数据进行掩膜
     */
    virtual std::future<GridData> applyVectorMask(
        const GridData& raster,
        const std::vector<Feature>& maskFeatures,
        const MaskOptions& options = MaskOptions()) = 0;
    
    /**
     * @brief 创建布尔掩膜网格
     * 为插值服务提供约束掩膜
     */
    virtual std::future<GridData> createBooleanMask(
        const GridDefinition& targetGrid,
        const std::vector<Feature>& features,
        bool inverseMask = false) = 0;
    
    // ============ 核心功能区域4：基础栅格操作 ============
    
    /**
     * @brief 栅格重采样
     * 为瓦片服务和其他服务提供栅格预处理能力
     */
    virtual std::future<GridData> resampleRaster(
        const GridData& input,
        const GridDefinition& targetGrid,
        ResamplingMethod method = ResamplingMethod::BILINEAR) = 0;
    
    /**
     * @brief 栅格裁剪
     * 为瓦片服务提供数据裁剪能力
     */
    virtual std::future<GridData> cropRaster(
        const GridData& raster,
        const BoundingBox& cropBox) = 0;
    
    // ============ 核心功能区域5：为其他服务提供的工具函数 ============
    
    /**
     * @brief 计算瓦片边界框
     * 为瓦片服务（FR-OUT-002）提供空间计算支持
     */
    virtual BoundingBox calculateTileBounds(
        int z, int x, int y,
        const CRSInfo& targetCRS = CRSInfo("EPSG:3857")) = 0;
    
    /**
     * @brief 查找网格单元索引
     * 为插值服务提供空间定位能力
     */
    virtual std::future<std::vector<std::optional<GridIndex>>> findGridCellsForPoints(
        const std::vector<Point>& points,
        const GridDefinition& gridDef) = 0;
    
    /**
     * @brief 计算空间距离
     * 为插值服务和其他服务提供距离计算
     */
    virtual std::future<double> computeDistance(
        const Point& point1,
        const Point& point2,
        DistanceType type = DistanceType::EUCLIDEAN) = 0;
};

/**
 * @brief 空间谓词枚举
 */
enum class SpatialPredicate {
    INTERSECTS,     // 相交
    CONTAINS,       // 包含  
    WITHIN,         // 被包含
    TOUCHES,        // 相切
    DISJOINT        // 分离
};

/**
 * @brief 掩膜选项
 */
struct MaskOptions {
    bool inverseMask = false;           // 是否反转掩膜
    double noDataValue = -9999.0;       // NoData值
    bool allTouched = false;            // 是否包含所有接触的像素
};

}
```

### 1.2 不属于空间服务的功能

**移除过度设计的功能**：

```cpp
// ❌ 这些功能不属于空间服务（过度设计）：

// 1. 高级空间分析（应在应用层实现）
class AdvancedSpatialAnalysis {
    WatershedAnalysis performWatershedAnalysis();        // -> 应用层
    ConnectivityGraph analyzeConnectivity();            // -> 应用层
    HotspotAnalysis detectHotspots();                   // -> 应用层
};

// 2. 工作流编排（应在TaskDispatcher层实现）
class SpatialWorkflowOrchestrator {
    void orchestrateComplexAnalysis();                  // -> TaskDispatcher
    void manageAnalysisWorkflow();                      // -> TaskDispatcher
};

// 3. 性能管理（应在common_utils实现）
class PerformanceManager {
    void optimizeMemoryUsage();                         // -> common_utils
    void manageParallelProcessing();                    // -> common_utils
    void handleCaching();                               // -> common_utils
};

// 4. 数据访问（已在data_access_service实现）
class SpatialDataAccess {
    GridData readSpatialData();                         // -> data_access_service
    std::vector<Feature> loadShapefiles();              // -> data_access_service
};
```

## 2. 与已完成模块的接口关系

### 2.1 依赖关系图

```
空间服务的依赖关系：

┌─────────────────────────────────────────────────────────┐
│                    已完成模块                           │
├─────────────────────────────────────────────────────────┤
│ ✓ common_utilities: 线程池、日志、错误处理             │
│ ✓ crs_service: 坐标转换、CRS解析                       │
│ ✓ metadata_service: 元数据查询                         │  
│ ✓ data_access_service: 数据读取                        │
└─────────────────────────────────────────────────────────┘
                           ↓ 依赖
┌─────────────────────────────────────────────────────────┐
│               空间服务 (spatial_ops_service)           │
├─────────────────────────────────────────────────────────┤
│ • 使用CRS服务进行坐标转换                              │
│ • 使用DataAccess服务读取空间数据                       │
│ • 使用CommonUtils的线程池和日志                        │
│ • 集成GDAL/OGR进行空间计算                             │
└─────────────────────────────────────────────────────────┘
                           ↓ 被依赖
┌─────────────────────────────────────────────────────────┐
│                   未来模块                              │
├─────────────────────────────────────────────────────────┤
│ ⏳ interpolation_service: 使用空间约束和掩膜功能       │
│ ⏳ modeling_service: 使用几何运算和空间查询             │
│ ⏳ tile_service: 使用栅格处理和坐标计算                │
└─────────────────────────────────────────────────────────┘
```

### 2.2 与CRS服务的接口

```cpp
/**
 * @brief 空间服务与CRS服务的集成
 */
class SpatialOpsServiceImpl {
private:
    std::shared_ptr<ICrsService> crsService_;
    
public:
    /**
     * @brief 空间查询中的CRS处理示例
     */
    std::future<std::vector<Feature>> spatialQuery(
        const std::vector<Feature>& features,
        const Geometry& queryGeometry,
        SpatialPredicate predicate) override {
        
        return std::async(std::launch::async, [=]() {
            
            // 1. 检查CRS一致性
            auto featureCRS = features[0].geometry.crs;
            auto queryCRS = queryGeometry.crs;
            
            // 2. 如果CRS不同，使用CRS服务进行转换
            Geometry normalizedQuery = queryGeometry;
            if (!crsService_->areEquivalentCRS(featureCRS, queryCRS)) {
                auto transformedPoint = crsService_->transformPoint(
                    queryGeometry.centroid.x,
                    queryGeometry.centroid.y,
                    queryCRS,
                    featureCRS
                );
                normalizedQuery = transformQueryGeometry(queryGeometry, transformedPoint);
            }
            
            // 3. 执行空间查询
            return performSpatialQueryWithGDAL(features, normalizedQuery, predicate);
        });
    }
};
```

### 2.3 与DataAccess服务的接口

```cpp
/**
 * @brief 空间服务与数据访问服务的集成
 */
class SpatialOpsServiceImpl {
private:
    std::shared_ptr<IDataAccessService> dataAccessService_;
    
public:
    /**
     * @brief 栅格掩膜操作中的数据访问示例
     */
    std::future<GridData> applyVectorMask(
        const GridData& raster,
        const std::vector<Feature>& maskFeatures,
        const MaskOptions& options) override {
        
        return std::async(std::launch::async, [=]() {
            
            // 1. 验证输入数据有效性
            if (raster.data.empty() || maskFeatures.empty()) {
                throw SpatialOpsException("Invalid input data for masking");
            }
            
            // 2. 如果需要，可以通过DataAccess服务获取额外的矢量数据
            // （在这个例子中，矢量数据已经作为参数传入）
            
            // 3. 使用GDAL进行栅格化和掩膜操作
            return performVectorMaskingWithGDAL(raster, maskFeatures, options);
        });
    }
    
    /**
     * @brief 为插值服务提供的空间约束生成
     */
    std::future<GridData> generateSpatialConstraints(
        const std::string& constraintDataPath,
        const GridDefinition& targetGrid) {
        
        return std::async(std::launch::async, [=]() {
            
            // 1. 使用DataAccess服务读取约束数据
            auto constraintFeatures = dataAccessService_->readFeaturesAsync(
                constraintDataPath).get();
            
            // 2. 生成约束掩膜
            return createBooleanMask(targetGrid, constraintFeatures).get();
        });
    }
};
```

### 2.4 与CommonUtils的集成

```cpp
/**
 * @brief 空间服务与通用工具的集成
 */
class SpatialOpsServiceImpl {
private:
    std::shared_ptr<common_utils::ThreadPool> threadPool_;
    std::shared_ptr<common_utils::Logger> logger_;
    
public:
    SpatialOpsServiceImpl(
        std::shared_ptr<ICrsService> crsService,
        std::shared_ptr<IDataAccessService> dataAccessService,
        std::shared_ptr<common_utils::ThreadPool> threadPool,
        std::shared_ptr<common_utils::Logger> logger)
        : crsService_(crsService)
        , dataAccessService_(dataAccessService)
        , threadPool_(threadPool)
        , logger_(logger) {
        
        logger_->info("SpatialOpsService initialized");
    }
    
    /**
     * @brief 使用线程池的并行空间查询
     */
    std::future<std::vector<Feature>> spatialQueryParallel(
        const std::vector<Feature>& features,
        const Geometry& queryGeometry,
        SpatialPredicate predicate) {
        
        // 将大任务分解为小任务，提交给线程池
        return threadPool_->submit([=]() {
            logger_->debug("Starting parallel spatial query for {} features", 
                          features.size());
            
            auto result = performSpatialQuery(features, queryGeometry, predicate);
            
            logger_->debug("Spatial query completed, found {} matching features", 
                          result.size());
            return result;
        });
    }
};
```

## 3. 内部架构设计

### 3.1 简化的内部架构

基于职责重新定义，空间服务的内部架构应该**大幅简化**：

```cpp
/**
 * @brief 精简的空间服务内部架构
 */
namespace oscean::core_services::spatial_ops {

class SpatialOpsServiceImpl : public ISpatialOpsService {
private:
    // ============ 核心计算引擎（简化版）============
    std::unique_ptr<GeometryEngine> geometryEngine_;      // 基础几何运算
    std::unique_ptr<RasterProcessor> rasterProcessor_;    // 基础栅格处理  
    std::unique_ptr<SpatialQueryEngine> queryEngine_;     // 空间查询引擎
    
    // ============ GDAL集成层 ============
    std::unique_ptr<GDALSpatialWrapper> gdalWrapper_;     // GDAL操作封装
    
    // ============ 依赖的核心服务 ============
    std::shared_ptr<ICrsService> crsService_;             // CRS服务
    
    // ============ 基础工具 ============
    std::shared_ptr<common_utils::ThreadPool> threadPool_;
    std::shared_ptr<common_utils::Logger> logger_;
    
    // ============ 配置 ============
    SpatialOpsConfig config_;
};

/**
 * @brief 几何运算引擎（简化版）
 */
class GeometryEngine {
public:
    // 基础几何运算
    Geometry computeBuffer(const Geometry& geom, double distance, const BufferOptions& options);
    bool intersects(const Geometry& g1, const Geometry& g2);
    bool contains(const Geometry& container, const Geometry& contained);
    double computeDistance(const Point& p1, const Point& p2, DistanceType type);
    
private:
    // 直接使用GDAL/OGR，不做过度封装
    std::unique_ptr<GDALSpatialWrapper> gdalWrapper_;
};

/**
 * @brief 栅格处理引擎（简化版）
 */
class RasterProcessor {
public:
    // 基础栅格操作
    GridData resampleRaster(const GridData& input, const GridDefinition& target, ResamplingMethod method);
    GridData cropRaster(const GridData& raster, const BoundingBox& cropBox);
    GridData applyVectorMask(const GridData& raster, const std::vector<Feature>& mask, const MaskOptions& options);
    GridData createBooleanMask(const GridDefinition& grid, const std::vector<Feature>& features, bool inverse);
    
private:
    std::unique_ptr<GDALSpatialWrapper> gdalWrapper_;
};

/**
 * @brief 空间查询引擎（简化版）
 */
class SpatialQueryEngine {
public:
    // 空间查询功能
    std::vector<Feature> spatialQuery(const std::vector<Feature>& features, const Geometry& query, SpatialPredicate predicate);
    std::vector<bool> pointsInPolygon(const std::vector<Point>& points, const Polygon& polygon);
    std::vector<std::optional<GridIndex>> findGridCellsForPoints(const std::vector<Point>& points, const GridDefinition& grid);
    
    // 工具函数
    BoundingBox calculateTileBounds(int z, int x, int y, const CRSInfo& crs);
    
private:
    std::shared_ptr<ICrsService> crsService_;
    std::unique_ptr<GDALSpatialWrapper> gdalWrapper_;
};

}
```

### 3.2 核心数据结构

```cpp
/**
 * @brief 空间服务配置
 */
struct SpatialOpsConfig {
    // 基础配置
    double geometryTolerance = 1e-10;              // 几何运算容差
    ResamplingMethod defaultResampling = ResamplingMethod::BILINEAR;
    
    // 性能配置
    size_t maxThreads = std::thread::hardware_concurrency();
    size_t maxMemoryMB = 512;                      // 限制内存使用
    bool enableParallelProcessing = true;
    
    // GDAL配置
    std::string gdalCacheSize = "256MB";
    std::vector<std::string> gdalOptions;
};

/**
 * @brief 重采样方法枚举
 */
enum class ResamplingMethod {
    NEAREST,        // 最近邻
    BILINEAR,       // 双线性
    CUBIC,          // 三次卷积
    AVERAGE,        // 平均值
    MODE,           // 众数
    MAX,            // 最大值
    MIN             // 最小值
};

/**
 * @brief 距离类型枚举
 */
enum class DistanceType {
    EUCLIDEAN,      // 欧几里得距离
    GEODESIC,       // 大地测量距离
    MANHATTAN       // 曼哈顿距离
};

/**
 * @brief 缓冲区选项
 */
struct BufferOptions {
    int segments = 8;                              // 圆弧分段数（简化）
    bool singleSided = false;                      // 单边缓冲区
    double flatness = 0.25;                        // 曲线平坦度
};
```

## 4. GDAL集成策略

### 4.0 重要架构调整：避免重复开发

**发现的问题**：空间服务最初设计的GDAL包装器与data_access模块中已有的GDAL集成存在**严重重叠**。

#### 4.0.1 已有GDAL集成分析

Data Access模块中已实现的GDAL功能：

```cpp
// 已存在的GDAL组件架构：
namespace oscean::core_services::data_access::readers::gdal {

├── GDALDatasetHandler          // 数据集生命周期管理、线程安全访问
├── GDALRasterIO               // 栅格数据读写操作
├── GDALVectorIO               // 矢量数据读写、OGR集成
├── GDALRasterReader           // 高级栅格读取接口
├── GDALVectorReader           // 高级矢量读取接口
├── metadata/
│   ├── GDALMetadataExtractor  // 元数据提取基类
│   ├── GDALRasterMetadataExtractor // 栅格元数据
│   └── GDALVectorMetadataExtractor // 矢量元数据
├── utils/
│   ├── GDALCommonUtils        // GDAL初始化、通用工具
│   └── GDALTransformationUtils // 坐标变换工具
└── crs_service/
    └── GDALCrsServiceImpl     // GDAL-CRS服务集成

}
```

#### 4.0.2 功能重叠度分析

| 计划功能 | Data Access已有 | 重叠程度 | 决策 |
|----------|----------------|----------|------|
| GDAL初始化 | ✅ GDALCommonUtils::initializeGDALOnce | 100% | **重用** |
| 数据集管理 | ✅ GDALDatasetHandler | 90% | **重用** |
| OGR几何转换 | ✅ convertOGRGeometryToOsceanGeometry | 85% | **重用** |
| 栅格I/O | ✅ GDALRasterIO | 70% | **扩展重用** |
| 空间参考 | ✅ GDALCrsServiceImpl | 80% | **重用** |
| 矢量I/O | ✅ GDALVectorIO | 60% | **扩展重用** |
| 纯空间计算 | ❌ 无 | 0% | **新开发** |

#### 4.0.3 修正的架构策略

**新策略**：**重用 + 扩展**而非重新开发

```cpp
/**
 * @brief 修正的空间服务GDAL集成策略
 */
namespace oscean::core_services::spatial_ops {

// ✅ 重用Data Access的GDAL组件
using GDALDatasetHandler = data_access::readers::gdal::GDALDatasetHandler;
using GDALCommonUtils = data_access::readers::gdal::utils::GDALCommonUtils;

/**
 * @brief 空间操作GDAL适配器（轻量级扩展）
 */
class SpatialOpsGDALAdapter {
public:
    SpatialOpsGDALAdapter(
        std::shared_ptr<GDALDatasetHandler> datasetHandler,
        std::shared_ptr<ICrsService> crsService);
    
    // ============ 新增：纯空间计算功能 ============
    
    /**
     * @brief 几何缓冲区计算（新功能）
     */
    std::unique_ptr<OGRGeometry> computeGeometryBuffer(
        OGRGeometry* geometry,
        double distance,
        int segments = 8);
    
    /**
     * @brief 几何空间关系判断（新功能）
     */
    bool geometryIntersects(OGRGeometry* geom1, OGRGeometry* geom2);
    bool geometryContains(OGRGeometry* container, OGRGeometry* contained);
    
    /**
     * @brief 栅格化矢量要素（新功能）
     */
    std::unique_ptr<GDALDataset> rasterizeFeatures(
        const std::vector<OGRGeometry*>& geometries,
        const GridDefinition& targetGrid,
        double burnValue = 1.0);
    
    /**
     * @brief 栅格掩膜操作（新功能）
     */
    std::unique_ptr<GDALDataset> applyVectorMask(
        GDALDataset* rasterDataset,
        const std::vector<OGRGeometry*>& maskGeometries,
        double noDataValue = -9999.0);
    
    // ============ 重用：委托给已有组件 ============
    
    /**
     * @brief 重用Data Access的几何转换
     */
    oscean::core_services::Geometry convertToOsceanGeometry(OGRGeometry* ogrGeom) {
        return dataAccessConverter_->convertOGRGeometryToOsceanGeometry(ogrGeom);
    }
    
    /**
     * @brief 重用Data Access的栅格读取
     */
    GridData readRasterData(const std::string& filePath, int bandNumber) {
        // 委托给GDALRasterIO
        return rasterIO_->readRasterData(bandNumber, gridData);
    }

private:
    // 重用已有组件
    std::shared_ptr<GDALDatasetHandler> datasetHandler_;
    std::shared_ptr<data_access::readers::gdal::io::GDALRasterIO> rasterIO_;
    std::shared_ptr<data_access::readers::gdal::io::GDALVectorIO> vectorIO_;
    std::shared_ptr<ICrsService> crsService_;
    
    // 仅为空间计算新增的轻量级工具
    std::unique_ptr<OGRSpatialReference> workingSRS_;
};

}
```

### 4.1 重用现有GDAL集成

**核心原则**：最大化重用，最小化重复开发

```cpp
/**
 * @brief 空间服务实现（重用版本）
 */
class SpatialOpsServiceImpl : public ISpatialOpsService {
private:
    // ============ 重用已有服务 ============
    std::shared_ptr<IDataAccessService> dataAccessService_;    // 重用数据读写
    std::shared_ptr<ICrsService> crsService_;                  // 重用CRS服务
    
    // ============ 轻量级空间计算扩展 ============
    std::unique_ptr<SpatialOpsGDALAdapter> gdalAdapter_;       // 轻量级GDAL适配器
    std::unique_ptr<GeometryComputeEngine> geometryEngine_;    // 纯几何计算
    std::unique_ptr<RasterSpatialProcessor> rasterProcessor_; // 栅格空间处理
    
    // ============ 基础工具（重用）============
    std::shared_ptr<common_utils::ThreadPool> threadPool_;
    std::shared_ptr<common_utils::Logger> logger_;

public:
    /**
     * @brief 空间查询实现（重用DataAccess + 增强）
     */
    std::future<std::vector<Feature>> spatialQuery(
        const std::vector<Feature>& features,
        const Geometry& queryGeometry,
        SpatialPredicate predicate) override {
        
        return std::async(std::launch::async, [=]() {
            // 1. 重用DataAccess的矢量处理能力
            // 2. 仅添加空间关系判断逻辑
            
            std::vector<Feature> results;
            for (const auto& feature : features) {
                // 使用GDAL适配器进行空间关系判断
                auto ogrGeom1 = convertToOGRGeometry(feature.geometry);
                auto ogrGeom2 = convertToOGRGeometry(queryGeometry);
                
                bool matches = false;
                switch (predicate) {
                    case SpatialPredicate::INTERSECTS:
                        matches = gdalAdapter_->geometryIntersects(ogrGeom1.get(), ogrGeom2.get());
                        break;
                    case SpatialPredicate::CONTAINS:
                        matches = gdalAdapter_->geometryContains(ogrGeom1.get(), ogrGeom2.get());
                        break;
                    // ... 其他谓词
                }
                
                if (matches) {
                    results.push_back(feature);
                }
            }
            return results;
        });
    }
    
    /**
     * @brief 栅格掩膜实现（重用DataAccess + 扩展空间操作）
     */
    std::future<GridData> applyVectorMask(
        const GridData& raster,
        const std::vector<Feature>& maskFeatures,
        const MaskOptions& options) override {
        
        return std::async(std::launch::async, [=]() {
            // 1. 使用轻量级适配器转换矢量要素
            std::vector<std::unique_ptr<OGRGeometry>> ogrGeometries;
            for (const auto& feature : maskFeatures) {
                ogrGeometries.push_back(convertToOGRGeometry(feature.geometry));
            }
            
            // 2. 使用GDAL适配器执行栅格化掩膜
            auto maskedDataset = gdalAdapter_->applyVectorMask(
                convertToGDALDataset(raster).get(),
                extractRawPointers(ogrGeometries),
                options.noDataValue
            );
            
            // 3. 重用DataAccess的转换功能
            return convertFromGDALDataset(maskedDataset.get());
        });
    }

private:
    /**
     * @brief 重用DataAccess的几何转换
     */
    std::unique_ptr<OGRGeometry> convertToOGRGeometry(const Geometry& osceanGeom) {
        // 委托给DataAccess中已有的转换函数
        return dataAccessConverter_.convertOsceanGeometryToOGR(osceanGeom);
    }
};
```

## 5. 为其他服务提供的接口

### 5.1 为插值服务提供的支持

```cpp
/**
 * @brief 空间服务为插值服务提供的专用接口
 */
namespace oscean::core_services::spatial_ops {

class InterpolationSpatialSupport {
public:
    /**
     * @brief 生成插值约束掩膜
     * 为插值服务提供空间约束信息
     */
    std::future<GridData> generateInterpolationConstraints(
        const GridDefinition& interpolationGrid,
        const std::vector<Feature>& constraintFeatures,
        const InterpolationConstraintConfig& config);
    
    /**
     * @brief 查找插值点的空间邻近信息
     * 为插值算法提供空间近邻数据
     */
    std::future<std::vector<SpatialNeighborInfo>> findInterpolationNeighbors(
        const Point& interpolationPoint,
        const std::vector<Point>& dataPoints,
        double maxDistance,
        int maxCount);
    
    /**
     * @brief 验证插值点是否在有效区域内
     * 为插值服务提供空间有效性检查
     */
    std::future<std::vector<bool>> validateInterpolationPoints(
        const std::vector<Point>& interpolationPoints,
        const std::vector<Feature>& validRegions);
    
    /**
     * @brief 生成插值网格的空间索引
     * 为插值服务提供高效的空间查找能力
     */
    std::future<GridSpatialIndex> createInterpolationSpatialIndex(
        const GridDefinition& interpolationGrid,
        const std::vector<Point>& dataPoints);
};

/**
 * @brief 插值约束配置
 */
struct InterpolationConstraintConfig {
    ConstraintType type = ConstraintType::LAND_SEA_MASK;
    double bufferDistance = 0.0;           // 约束缓冲距离
    bool inverseMask = false;              // 是否反转掩膜
    double constraintValue = 0.0;          // 约束值
};

/**
 * @brief 空间邻近信息
 */
struct SpatialNeighborInfo {
    Point neighborPoint;                   // 邻近点
    double distance;                       // 距离
    size_t originalIndex;                  // 原始索引
    double weight;                         // 空间权重
};

}
```

### 5.2 为瓦片服务提供的支持

```cpp
/**
 * @brief 空间服务为瓦片服务提供的专用接口
 */
namespace oscean::core_services::spatial_ops {

class TileSpatialSupport {
public:
    /**
     * @brief 计算瓦片的地理边界框（FR-OUT-002支持）
     */
    BoundingBox calculateTileBounds(
        int z, int x, int y,
        const CRSInfo& targetCRS = CRSInfo("EPSG:3857"));
    
    /**
     * @brief 为瓦片准备栅格数据
     * 包括重投影、重采样、裁剪等操作
     */
    std::future<GridData> prepareRasterForTile(
        const GridData& sourceRaster,
        const TileRequest& tileRequest);
    
    /**
     * @brief 为瓦片栅格化矢量数据
     */
    std::future<GridData> rasterizeVectorForTile(
        const std::vector<Feature>& vectorFeatures,
        const TileRequest& tileRequest,
        const RasterizationOptions& options);
    
    /**
     * @brief 判断瓦片是否与数据范围相交
     * 用于瓦片服务的快速过滤
     */
    bool tileIntersectsDataExtent(
        int z, int x, int y,
        const BoundingBox& dataExtent,
        const CRSInfo& dataCRS);
};

/**
 * @brief 瓦片请求结构
 */
struct TileRequest {
    int z, x, y;                           // 瓦片坐标
    int tileSize = 256;                    // 瓦片大小（像素）
    CRSInfo targetCRS = CRSInfo("EPSG:3857");  // 目标CRS
    std::optional<CRSInfo> sourceCRS;      // 源数据CRS
};

/**
 * @brief 栅格化选项
 */
struct RasterizationOptions {
    double burnValue = 1.0;                // 烧录值
    bool allTouched = false;               // 是否包含所有接触像素
    std::string attributeField;           // 属性字段名（用于取值）
};

}
```

### 5.3 为建模服务提供的支持

```cpp
/**
 * @brief 空间服务为建模服务提供的专用接口
 */
namespace oscean::core_services::spatial_ops {

class ModelingSpatialSupport {
public:
    /**
     * @brief 为建模计算提供空间几何运算
     */
    std::future<Geometry> computeModelingGeometry(
        const Geometry& inputGeometry,
        ModelingGeometryOperation operation,
        const ModelingGeometryParams& params);
    
    /**
     * @brief 为建模结果提供空间验证
     */
    std::future<ModelingSpatialValidationResult> validateModelingResults(
        const std::vector<Point>& modelingPoints,
        const std::vector<double>& modelingValues,
        const SpatialValidationCriteria& criteria);
    
    /**
     * @brief 为建模计算提供区域统计
     */
    std::future<std::map<std::string, double>> computeRegionalStatistics(
        const GridData& modelingResults,
        const std::vector<Feature>& regions,
        const std::vector<std::string>& statistics);
};

}
```

## 6. 实施计划

### 6.1 开发阶段规划

```
第一阶段：核心架构和基础功能 (3周)
├── Week 1: 核心架构搭建
│   ├── 接口定义完善
│   ├── 基础数据结构实现
│   ├── GDAL集成层搭建
│   └── 基础单元测试框架
├── Week 2: 基础几何运算实现
│   ├── GeometryEngine实现
│   ├── 缓冲区、相交、包含等基础运算
│   ├── 单元测试编写
│   └── 与CRS服务集成测试
└── Week 3: 基础栅格操作实现
    ├── RasterProcessor实现
    ├── 重采样、裁剪、掩膜功能
    ├── 单元测试编写
    └── 与DataAccess服务集成测试

第二阶段：空间查询和高级功能 (2周)
├── Week 4: 空间查询引擎
│   ├── SpatialQueryEngine实现
│   ├── 空间谓词查询功能
│   ├── 点在多边形判断
│   └── 集成测试
└── Week 5: 服务支持接口
    ├── 为插值服务提供的接口实现
    ├── 为瓦片服务提供的接口实现
    ├── 性能优化和并行处理
    └── 完整的集成测试

第三阶段：优化和部署准备 (1周)
└── Week 6: 优化和部署
    ├── 性能基准测试和优化
    ├── 错误处理完善
    ├── 文档完善
    └── 部署准备

总计：6周
```

### 6.2 优先级矩阵

```
功能优先级分级:

P0 (必须实现) - Week 1-3:
✅ 基础几何运算 (缓冲区、相交、包含)
✅ 栅格掩膜操作 (FR-PROC-004)
✅ 空间查询功能 (FR-PROC-004)
✅ 栅格重采样和裁剪
✅ 与CRS服务的集成
✅ 与DataAccess服务的集成

P1 (高优先级) - Week 4-5:
⚪ 为插值服务提供的约束掩膜功能
⚪ 为瓦片服务提供的边界计算功能
⚪ 高效的空间查询算法
⚪ 并行处理支持

P2 (中优先级) - Week 6:
⚪ 性能优化和内存管理
⚪ 完善的错误处理
⚪ 全面的单元测试覆盖

P3 (低优先级) - 未来版本:
○ 高级几何运算
○ 复杂的栅格代数
○ 高级空间索引
```

### 6.3 测试策略

```cpp
/**
 * @brief 空间服务测试策略
 */
namespace oscean::core_services::spatial_ops::tests {

/**
 * @brief 单元测试计划
 */
class SpatialOpsUnitTests {
public:
    // 基础几何运算测试
    void testGeometryBuffer();
    void testGeometryIntersection();
    void testGeometryContainment();
    void testDistanceCalculation();
    
    // 栅格操作测试
    void testRasterResampling();
    void testRasterCropping();
    void testVectorMasking();
    void testBooleanMaskCreation();
    
    // 空间查询测试
    void testSpatialQuery();
    void testPointsInPolygon();
    void testGridCellFinding();
    
    // GDAL集成测试
    void testGDALWrapper();
    void testDataTypeConversion();
    void testCRSTransformation();
};

/**
 * @brief 集成测试计划
 */
class SpatialOpsIntegrationTests {
public:
    // 服务间集成测试
    void testCRSServiceIntegration();
    void testDataAccessServiceIntegration();
    void testCommonUtilsIntegration();
    
    // 端到端测试
    void testRealWorldSpatialQuery();
    void testLargeRasterProcessing();
    void testComplexGeometryOperations();
    
    // 性能测试
    void testPerformanceBenchmarks();
    void testMemoryUsage();
    void testConcurrentAccess();
};

/**
 * @brief 验收测试计划
 */
class SpatialOpsAcceptanceTests {
public:
    // FR-PROC-004验收测试
    void testShapefileMaskingRequirement();
    void testSpatialQueryRequirement();
    
    // FR-OUT-002支持测试
    void testTileCoordinateSupport();
    
    // 插值服务支持测试
    void testInterpolationConstraintGeneration();
    void testSpatialNeighborFinding();
    
    // 瓦片服务支持测试
    void testTileBoundaryCalculation();
    void testRasterPreparationForTiles();
};

}
```

## 7. 部署和配置

### 7.1 依赖管理

```cmake
# CMakeLists.txt 配置
cmake_minimum_required(VERSION 3.16)
project(spatial_ops_service)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 核心依赖
find_package(GDAL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

# OSCEAN内部依赖
find_package(oscean_core_service_interfaces REQUIRED)
find_package(oscean_common_utils REQUIRED)

# 空间服务库
add_library(spatial_ops_service
    src/spatial_ops_service_impl.cpp
    src/geometry_engine.cpp
    src/raster_processor.cpp
    src/spatial_query_engine.cpp
    src/gdal_spatial_wrapper.cpp
    src/oscean_gdal_converter.cpp
)

target_link_libraries(spatial_ops_service
    PUBLIC
    oscean::core_service_interfaces
    oscean::common_utils
    GDAL::GDAL
    Boost::system
)

target_include_directories(spatial_ops_service
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 测试
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
```

### 7.2 配置文件

```json
{
  "spatial_ops_service": {
    "geometry": {
      "tolerance": 1e-10,
      "buffer_segments": 8,
      "simplification_tolerance": 0.001
    },
    "raster": {
      "default_resampling": "bilinear",
      "max_memory_mb": 512,
      "tile_cache_size": 100,
      "nodata_value": -9999.0
    },
    "performance": {
      "max_threads": 0,
      "enable_parallel": true,
      "chunk_size": 1024
    },
    "gdal": {
      "cache_size": "256MB",
      "options": [
        "GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR",
        "GDAL_NUM_THREADS=ALL_CPUS"
      ]
    },
    "logging": {
      "level": "info",
      "file": "spatial_ops.log"
    }
  }
}
```

## 8. 总结

### 8.1 关键设计原则

1. **职责明确**：专注于基础空间计算，不承担高级分析和工作流功能
2. **依赖合理**：充分利用已完成的CRS和DataAccess服务
3. **GDAL集成**：最大化利用GDAL/OGR的成熟功能，避免重复造轮子
4. **服务导向**：为插值服务、瓦片服务等提供必要的空间计算支持
5. **简单高效**：保持内部架构简洁，专注性能优化

### 8.2 成功标准

```
功能完整性：
✅ 实现FR-PROC-004的所有空间处理需求
✅ 为FR-OUT-002提供坐标计算支持
✅ 为插值服务提供约束掩膜功能
✅ 为瓦片服务提供空间预处理功能

技术质量：
✅ 代码覆盖率 > 85%
✅ 单元测试通过率 100%
✅ 集成测试通过率 > 95%
✅ 性能基准达标（处理1GB栅格 < 30秒）

架构合规性：
✅ 符合OSCEAN分层架构定位
✅ 与已完成模块正确集成
✅ 接口设计清晰合理
✅ 错误处理完善

可维护性：
✅ 代码结构清晰
✅ 文档完善
✅ 配置灵活
✅ 易于扩展
```

这个重新设计的空间服务**完全符合OSCEAN架构要求**，专注于提供**基础空间计算能力**，为系统中的其他服务提供必要的空间几何和栅格处理支持，既实用又可靠！ 

### 4.1 修正的架构定位：核心空间计算的正确职责

经过深入分析，发现之前的简化策略过度削弱了空间服务的核心职责。**空间服务必须承担核心的空间计算功能**，而不能简单地委托给其他层。

#### 4.1.1 正确的职责分工

```cpp
/**
 * @brief 修正的职责分工 - 明确各层的空间功能边界
 */

// ============ DataAccess层的空间功能（已有）============
namespace oscean::core_services::data_access {
    // ✅ 基础数据读取和简单过滤
    FeatureCollection readFeaturesByBoundingBox();      // 边界框过滤
    bool applySpatialFilter();                          // 简单空间过滤
    Geometry convertOGRGeometryToOsceanGeometry();      // 几何格式转换
    
    // ❌ 不应该做复杂的空间关系计算
    // ❌ 不应该做掩膜算法
    // ❌ 不应该做几何运算算法
}

// ============ 空间服务层的核心职责（应该做的）============
namespace oscean::core_services::spatial_ops {
    // ✅ 空间关系计算 - 核心算法职责
    virtual std::future<bool> intersects(const Geometry& g1, const Geometry& g2) = 0;
    virtual std::future<bool> contains(const Geometry& container, const Geometry& contained) = 0;
    virtual std::future<bool> within(const Geometry& inner, const Geometry& outer) = 0;
    virtual std::future<bool> touches(const Geometry& g1, const Geometry& g2) = 0;
    virtual std::future<double> distance(const Geometry& g1, const Geometry& g2) = 0;
    
    // ✅ 复杂空间查询算法 - 超越简单边界框过滤
    virtual std::future<std::vector<Feature>> spatialQuery(
        const std::vector<Feature>& features,
        const Geometry& queryGeometry,
        SpatialPredicate predicate) = 0;
    
    virtual std::future<std::vector<Feature>> findFeaturesNearPoint(
        const std::vector<Feature>& features,
        const Point& queryPoint,
        double radius) = 0;
    
    // ✅ 几何运算算法 - 核心空间计算
    virtual std::future<Geometry> computeBuffer(
        const Geometry& geometry, double distance) = 0;
    virtual std::future<std::optional<Geometry>> computeIntersection(
        const Geometry& g1, const Geometry& g2) = 0;
    virtual std::future<std::optional<Geometry>> computeUnion(
        const std::vector<Geometry>& geometries) = 0;
    virtual std::future<std::optional<Geometry>> computeDifference(
        const Geometry& g1, const Geometry& g2) = 0;
    
    // ✅ 掩膜计算算法 - 复杂的空间处理
    virtual std::future<GridData> applyVectorMask(
        const GridData& raster,
        const std::vector<Feature>& maskFeatures,
        const MaskOptions& options) = 0;
    
    virtual std::future<GridData> createBooleanMask(
        const GridDefinition& targetGrid,
        const std::vector<Feature>& features) = 0;
    
    // ✅ 栅格化算法 - 矢量到栅格转换
    virtual std::future<GridData> rasterizeFeatures(
        const std::vector<Feature>& features,
        const GridDefinition& targetGrid,
        const RasterizationOptions& options) = 0;
    
    // ✅ 空间索引和优化算法
    virtual std::future<SpatialIndex> createSpatialIndex(
        const std::vector<Feature>& features) = 0;
    
    virtual std::future<std::vector<Feature>> queryWithIndex(
        const SpatialIndex& index,
        const Geometry& queryGeometry) = 0;
}

// ============ 上层服务的职责（插值、建模等）============
namespace oscean::core_services::interpolation {
    // ✅ 使用空间服务提供的基础能力
    auto constraintMask = spatialOpsService->createBooleanMask(grid, constraintFeatures);
    auto nearbyPoints = spatialOpsService->findFeaturesNearPoint(dataPoints, interpolationPoint, radius);
    
    // ❌ 不应该自己实现空间关系判断
    // ❌ 不应该自己实现掩膜算法
}
```

#### 4.1.2 重用策略的正确平衡

```cpp
/**
 * @brief 正确的重用策略：重用基础设施，实现核心算法
 */
class SpatialOpsServiceImpl : public ISpatialOpsService {
private:
    // ============ 重用：基础设施组件 ============
    std::shared_ptr<IDataAccessService> dataAccessService_;           // 重用数据读取
    std::shared_ptr<ICrsService> crsService_;                        // 重用坐标转换
    std::shared_ptr<data_access::GDALDatasetHandler> gdalHandler_;    // 重用GDAL管理
    std::shared_ptr<data_access::GDALCommonUtils> gdalUtils_;        // 重用GDAL工具
    
    // ============ 新实现：核心空间算法 ============
    std::unique_ptr<SpatialRelationshipEngine> relationshipEngine_;  // 空间关系计算
    std::unique_ptr<GeometryOperationsEngine> geometryEngine_;       // 几何运算
    std::unique_ptr<VectorRasterizer> vectorRasterizer_;            // 矢量栅格化
    std::unique_ptr<MaskingEngine> maskingEngine_;                  // 掩膜算法
    std::unique_ptr<SpatialQueryEngine> queryEngine_;               // 空间查询算法
    std::unique_ptr<SpatialIndexManager> indexManager_;             // 空间索引
    
public:
    /**
     * @brief 空间关系计算 - 核心实现，不委托
     */
    std::future<bool> intersects(const Geometry& g1, const Geometry& g2) override {
        return std::async(std::launch::async, [=]() {
            // 使用重用的GDAL基础设施，但实现自己的算法逻辑
            auto ogr1 = gdalUtils_->convertToOGRGeometry(g1);
            auto ogr2 = gdalUtils_->convertToOGRGeometry(g2);
            
            // 核心算法：空间关系判断
            return relationshipEngine_->computeIntersects(ogr1.get(), ogr2.get());
        });
    }
    
    /**
     * @brief 复杂空间查询 - 超越DataAccess的简单过滤
     */
    std::future<std::vector<Feature>> spatialQuery(
        const std::vector<Feature>& features,
        const Geometry& queryGeometry,
        SpatialPredicate predicate) override {
        
        return std::async(std::launch::async, [=]() {
            // 不是简单委托，而是实现复杂的空间查询算法
            
            // 1. 可选的空间索引优化
            auto index = indexManager_->buildTempIndex(features);
            
            // 2. 精确的空间关系计算
            std::vector<Feature> results;
            for (const auto& feature : features) {
                bool matches = false;
                switch (predicate) {
                    case SpatialPredicate::INTERSECTS:
                        matches = relationshipEngine_->computeIntersects(
                            feature.geometry, queryGeometry);
                        break;
                    case SpatialPredicate::CONTAINS:
                        matches = relationshipEngine_->computeContains(
                            feature.geometry, queryGeometry);
                        break;
                    case SpatialPredicate::WITHIN:
                        matches = relationshipEngine_->computeWithin(
                            feature.geometry, queryGeometry);
                        break;
                    // ... 其他复杂谓词
                }
                
                if (matches) {
                    results.push_back(feature);
                }
            }
            return results;
        });
    }
    
    /**
     * @brief 掩膜计算 - 复杂的空间处理算法
     */
    std::future<GridData> applyVectorMask(
        const GridData& raster,
        const std::vector<Feature>& maskFeatures,
        const MaskOptions& options) override {
        
        return std::async(std::launch::async, [=]() {
            // 复杂的掩膜算法，不是简单委托
            
            // 1. 矢量栅格化
            auto maskGrid = vectorRasterizer_->rasterizeFeatures(
                maskFeatures, raster.definition);
            
            // 2. 掩膜应用算法
            return maskingEngine_->applyMask(raster, maskGrid, options);
        });
    }
};
```

#### 4.1.3 核心空间算法引擎设计

```cpp
/**
 * @brief 空间关系计算引擎 - 核心算法实现
 */
class SpatialRelationshipEngine {
public:
    /**
     * @brief 高精度相交判断
     * 超越简单的边界框检查，进行精确的几何相交计算
     */
    bool computeIntersects(OGRGeometry* geom1, OGRGeometry* geom2);
    
    /**
     * @brief 包含关系判断
     * 实现完整的拓扑包含算法
     */
    bool computeContains(OGRGeometry* container, OGRGeometry* contained);
    
    /**
     * @brief 距离计算
     * 支持欧几里得距离和大地测量距离
     */
    double computeDistance(OGRGeometry* geom1, OGRGeometry* geom2, DistanceType type);
    
    /**
     * @brief 拓扑关系判断
     * 实现DE-9IM模型的完整拓扑关系
     */
    TopologicalRelation computeTopologicalRelation(OGRGeometry* geom1, OGRGeometry* geom2);

private:
    // 重用GDAL基础设施，但实现自己的算法逻辑
    std::shared_ptr<data_access::GDALCommonUtils> gdalUtils_;
    
    // 算法优化组件
    std::unique_ptr<GeometryValidator> validator_;
    std::unique_ptr<TopologyOptimizer> optimizer_;
};

/**
 * @brief 几何运算引擎 - 核心算法实现
 */
class GeometryOperationsEngine {
public:
    /**
     * @brief 缓冲区计算
     * 实现精确的几何缓冲区算法
     */
    std::unique_ptr<OGRGeometry> computeBuffer(
        OGRGeometry* geometry, double distance, const BufferOptions& options);
    
    /**
     * @brief 几何相交运算
     * 计算两个几何体的精确相交结果
     */
    std::unique_ptr<OGRGeometry> computeIntersection(
        OGRGeometry* geom1, OGRGeometry* geom2);
    
    /**
     * @brief 几何联合运算
     * 计算多个几何体的联合
     */
    std::unique_ptr<OGRGeometry> computeUnion(
        const std::vector<OGRGeometry*>& geometries);
    
    /**
     * @brief 几何差运算
     * 计算两个几何体的差集
     */
    std::unique_ptr<OGRGeometry> computeDifference(
        OGRGeometry* geom1, OGRGeometry* geom2);

private:
    std::shared_ptr<data_access::GDALCommonUtils> gdalUtils_;
    std::unique_ptr<GeometrySimplifier> simplifier_;
    std::unique_ptr<TopologyProcessor> topologyProcessor_;
};

/**
 * @brief 掩膜处理引擎 - 复杂掩膜算法
 */
class MaskingEngine {
public:
    /**
     * @brief 应用矢量掩膜
     * 实现高效的矢量掩膜算法
     */
    GridData applyVectorMask(
        const GridData& raster,
        const GridData& maskGrid,
        const MaskOptions& options);
    
    /**
     * @brief 创建布尔掩膜
     * 基于空间关系创建布尔掩膜
     */
    GridData createBooleanMask(
        const GridDefinition& targetGrid,
        const std::vector<Feature>& features,
        MaskRule rule);
    
    /**
     * @brief 多层掩膜组合
     * 实现复杂的多层掩膜算法
     */
    GridData combineMasks(
        const std::vector<GridData>& masks,
        MaskCombineOperation operation);

private:
    std::unique_ptr<RasterProcessor> rasterProcessor_;
    std::unique_ptr<VectorRasterizer> vectorRasterizer_;
};
```

#### 4.1.4 修正的开发工作量估算

```
核心空间算法实现（必须要做的）：
├── SpatialRelationshipEngine    (2-3周)  ✅ 核心空间关系算法
├── GeometryOperationsEngine     (2-3周)  ✅ 几何运算算法  
├── MaskingEngine               (2-3周)  ✅ 掩膜处理算法
├── VectorRasterizer            (1-2周)  ✅ 矢量栅格化算法
├── SpatialQueryEngine          (1-2周)  ✅ 复杂查询算法
├── SpatialIndexManager         (1-2周)  ✅ 空间索引优化
└── 集成和测试                  (2-3周)  ✅ 完整测试

重用基础设施（避免重复造轮子）：
├── GDAL初始化和管理            (0周)    ✅ 重用DataAccess
├── 几何格式转换                (0周)    ✅ 重用DataAccess  
├── 坐标系转换                  (0周)    ✅ 重用CRS服务
├── 数据读取和简单过滤           (0周)    ✅ 重用DataAccess
└── 通用工具和线程池             (0周)    ✅ 重用CommonUtils

总计：9-16周 (合理的工作量，专注核心算法)
```

### 4.2 与已有功能的正确边界划分

```cpp
/**
 * @brief 明确的功能边界 - 避免重叠但不过度简化
 */

// ============ DataAccess层边界 ============
namespace data_access {
    // ✅ 负责：数据I/O和基础转换
    FeatureCollection readFeaturesByBoundingBox();      // 边界框过滤（简单）
    Geometry convertOGRToOscean();                      // 格式转换
    
    // ❌ 不负责：复杂空间算法
    // 不实现：intersects, contains, buffer, mask等算法
}

// ============ 空间服务层边界 ============  
namespace spatial_ops {
    // ✅ 负责：核心空间算法
    bool intersects();                                  // 精确相交判断
    GridData applyVectorMask();                        // 掩膜算法
    
    // ❌ 不负责：数据I/O
    // 不重复：readFeatures, writeFeatures等
    
    // ✅ 可以调用：DataAccess的数据读取功能
    auto features = dataAccessService_->readFeatures(path);
    auto result = this->spatialQuery(features, queryGeom);
}

// ============ 上层服务边界 ============
namespace interpolation {
    // ✅ 负责：业务逻辑和算法
    GridData performInterpolation();
    
    // ❌ 不负责：空间关系判断
    // 调用空间服务：spatialOpsService_->intersects()
    
    // ❌ 不负责：数据读取
    // 调用数据服务：dataAccessService_->readGridData()
}
```

// ... existing code ... 