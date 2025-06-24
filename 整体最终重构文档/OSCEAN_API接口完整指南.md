# OSCEAN 项目 API 接口完整指南

## 概述

本文档提供OSCEAN (Ocean Simulation Computation Environment And Network-service) 项目的完整API接口指南。基于当前项目实现状态，将API接口按功能模块进行分类描述。

## 项目架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    网络服务层 (Layer 1)                 │
│                   network_service                       │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│              任务调度与工作流引擎 (Layer 2)              │
│            task_dispatcher & workflow_engine            │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   核心服务层 (Layer 3)                  │
│  ┌─────────────┬─────────────┬─────────────┬──────────┐  │
│  │ 数据访问服务 │ 元数据服务   │ 空间操作服务 │ 插值服务  │  │
│  │ DataAccess  │ Metadata    │ SpatialOps  │Interpolat│  │
│  └─────────────┴─────────────┴─────────────┴──────────┘  │
│  ┌─────────────┬─────────────┬─────────────┬──────────┐  │
│  │ CRS服务     │ 建模服务     │ 缓存服务     │ 其他服务  │  │
│  │ CRS        │ Modeling    │ Cache       │ Others   │  │
│  └─────────────┴─────────────┴─────────────┴──────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                 输出生成与交付层 (Layer 4)               │
│                   output_generation                     │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                共享基础库与工具集 (Layer 5)              │
│                   common_utilities                      │
└─────────────────────────────────────────────────────────┘
```

## 1. 核心数据类型 (Core Data Types)

### 1.1 基础数据类型

```cpp
namespace oscean::core_services {

// 数据类型枚举
enum class DataType {
    Unknown, Byte, Int8, UInt8, Int16, UInt16, Int32, UInt32, 
    Int64, UInt64, Float32, Float64, String, Boolean, Binary,
    Complex16, Complex32, Complex64
};

// 坐标维度类型
enum class CoordinateDimension {
    LON, LAT, VERTICAL, TIME, SPECTRAL, BAND, 
    INSTANCE, FEATURE_ID, STRING_CHAR, OTHER, NONE
};

// 几何类型
enum class GeometryType {
    Unknown, Point, LineString, Polygon, 
    MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
};

// 属性值变体类型
using AttributeValue = std::variant<
    std::monostate, std::string, bool, char, long long, 
    unsigned long long, double, std::vector<char>, 
    std::vector<long long>, std::vector<unsigned long long>,
    std::vector<double>, std::vector<std::string>
>;

}
```

### 1.2 空间数据类型

```cpp
// 点坐标
struct Point {
    double x, y;
    std::optional<double> z;
    std::optional<std::string> crsId;
    
    Point(double x_val, double y_val, 
          std::optional<double> z_val = std::nullopt,
          std::optional<std::string> crs_id_val = std::nullopt);
};

// 边界框
struct BoundingBox {
    double minX, minY, maxX, maxY;
    std::optional<double> minZ, maxZ;
    std::string crsId;
    
    BoundingBox(double min_x, double min_y, double max_x, double max_y,
                std::optional<double> min_z = std::nullopt,
                std::optional<double> max_z = std::nullopt,
                std::string crs_id = "");
    bool isValid() const;
};

// CRS信息
struct CRSInfo {
    std::string id;                    // 唯一标识符
    std::string name;                  // 名称描述
    std::string wkt;                   // WKT格式描述
    std::string projString;            // PROJ格式字符串
    bool isGeographic = false;         // 是否为地理坐标系
    bool isProjected = false;          // 是否为投影坐标系
    std::optional<int> epsgCode;       // EPSG代码
};
```

### 1.3 网格数据类型

```cpp
// 网格定义
struct GridDefinition {
    size_t rows = 0, cols = 0;
    BoundingBox extent;
    double xResolution = 0.0, yResolution = 0.0;
    CRSInfo crs;
    std::string gridName;
    std::vector<DimensionCoordinateInfo> dimensions;
    DataType originalDataType = DataType::Unknown;
    std::vector<CoordinateDimension> dimensionOrderInDataLayout;
    std::map<std::string, std::string> globalAttributes;
};

// 网格数据
class GridData {
public:
    GridData(const GridDefinition& gridDef, DataType dataType, size_t bandCount = 1);
    
    const GridDefinition& getDefinition() const;
    DataType getDataType() const;
    size_t getElementSizeBytes() const;
    size_t getTotalDataSize() const;
    const void* getDataPtr() const;
    
    template<typename T>
    T getValue(const std::vector<size_t>& indices) const;
    
    template<typename T>
    void setValue(const std::vector<size_t>& indices, T value);
};
```

## 2. 数据访问服务 API (Data Access Service)

### 2.1 接口定义

```cpp
namespace oscean::core_services {

class IRawDataAccessService {
public:
    virtual ~IRawDataAccessService() = default;
    
    // 基础信息
    virtual std::string getVersion() const = 0;
    virtual std::vector<std::string> getSupportedFormats() const = 0;
    
    // 网格数据读取
    virtual boost::future<GridData> readGridVariableSubsetAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::optional<IndexRange>& timeRange = std::nullopt,
        const std::optional<BoundingBox>& spatialExtent = std::nullopt,
        const std::optional<IndexRange>& levelRange = std::nullopt
    ) = 0;
    
    // 矢量数据读取
    virtual boost::future<std::vector<Feature>> readFeaturesAsync(
        const std::string& filePath,
        const std::string& layerName = "",
        const std::optional<BoundingBox>& spatialFilter = std::nullopt,
        const std::optional<AttributeFilter>& attributeFilter = std::nullopt,
        const std::optional<CRSInfo>& targetCRS = std::nullopt
    ) = 0;
    
    // 时间序列数据读取
    virtual boost::future<TimeSeriesData> readTimeSeriesAtPointAsync(
        const std::string& filePath,
        const std::string& varName,
        const Point& point,
        const std::string& method = "nearest"
    ) = 0;
    
    // 垂直剖面数据读取
    virtual boost::future<VerticalProfileData> readVerticalProfileAsync(
        const std::string& filePath,
        const std::string& varName,
        const Point& point,
        const Timestamp& time,
        const std::string& spatialMethod = "nearest",
        const std::string& timeMethod = "nearest"
    ) = 0;
    
    // 元数据提取
    virtual boost::future<std::optional<FileMetadata>> extractFileMetadataAsync(
        const std::string& filePath,
        const std::optional<CRSInfo>& targetCrs = std::nullopt
    ) = 0;
    
    // 变量检查
    virtual boost::future<bool> checkVariableExistsAsync(
        const std::string& filePath,
        const std::string& varName
    ) = 0;
};

}
```

### 2.2 使用示例

```cpp
// 创建数据访问服务实例
auto dataService = std::make_shared<RawDataAccessServiceImpl>();

// 读取网格数据子集
IndexRange timeRange(0, 10);  // 读取前10个时间步
BoundingBox bbox(110.0, 30.0, 120.0, 40.0, "EPSG:4326");

auto gridFuture = dataService->readGridVariableSubsetAsync(
    "/path/to/ocean_data.nc",
    "sea_surface_temperature",
    timeRange,
    bbox
);

GridData gridData = gridFuture.get();
std::cout << "读取到网格数据: " << gridData.getDefinition().cols 
          << "x" << gridData.getDefinition().rows << std::endl;

// 读取矢量要素
auto featuresFuture = dataService->readFeaturesAsync(
    "/path/to/coastline.shp",
    "",  // 默认图层
    bbox  // 空间过滤
);

auto features = featuresFuture.get();
std::cout << "读取到 " << features.size() << " 个要素" << std::endl;
```

## 3. 元数据服务 API (Metadata Service)

### 3.1 接口定义

```cpp
namespace oscean::core_services {

class IMetadataService {
public:
    virtual ~IMetadataService() = default;
    
    // 基础信息
    virtual std::string getVersion() const = 0;
    
    // 文件查找
    virtual std::future<std::vector<FileInfo>> findFilesAsync(
        const QueryCriteria& criteria
    ) = 0;
    
    // 元数据管理
    virtual std::future<std::optional<FileMetadata>> getFileMetadataAsync(
        const std::string& fileId
    ) = 0;
    
    virtual std::future<bool> addOrUpdateFileMetadataAsync(
        const FileMetadata& metadata
    ) = 0;
    
    virtual std::future<bool> removeFileMetadataAsync(
        const std::string& fileId
    ) = 0;
    
    // 时空查询
    virtual std::future<std::vector<FileMetadata>> findFilesByTimeRangeAsync(
        const Timestamp& start,
        const Timestamp& end
    ) = 0;
    
    virtual std::future<std::vector<FileMetadata>> findFilesByBBoxAsync(
        const BoundingBox& bbox
    ) = 0;
    
    // 变量信息
    virtual std::future<std::vector<std::string>> getAvailableVariablesAsync() = 0;
    
    virtual std::future<std::optional<std::pair<Timestamp, Timestamp>>> 
    getTimeRangeAsync(const std::optional<std::string>& variableName = std::nullopt) = 0;
    
    virtual std::future<std::optional<BoundingBox>> 
    getSpatialExtentAsync(
        const std::optional<std::string>& variableName = std::nullopt,
        const std::optional<CRSInfo>& targetCrs = std::nullopt
    ) = 0;
    
    // 索引管理
    virtual std::future<IndexingStatus> startIndexingAsync(
        const std::vector<std::string>& directories
    ) = 0;
    
    virtual IndexingStatus getIndexingStatus() const = 0;
    virtual std::future<void> stopIndexingAsync() = 0;
};

}
```

### 3.2 查询条件

```cpp
struct QueryCriteria {
    std::optional<BoundingBox> spatialExtent;
    std::optional<std::pair<Timestamp, Timestamp>> timeRange;
    std::vector<std::string> variables;
    std::string textFilter;
    std::string formatFilter;
    std::map<std::string, std::string> metadataFilters;
};
```

### 3.3 使用示例

```cpp
// 创建元数据服务实例
auto metadataService = std::make_shared<MetadataServiceImpl>();

// 构建查询条件
QueryCriteria criteria;
criteria.spatialExtent = BoundingBox(110.0, 30.0, 120.0, 40.0, "EPSG:4326");
criteria.variables = {"sea_surface_temperature", "sea_surface_salinity"};

// 查找文件
auto filesFuture = metadataService->findFilesAsync(criteria);
auto files = filesFuture.get();

std::cout << "找到 " << files.size() << " 个匹配文件" << std::endl;

// 获取时间范围
auto timeRangeFuture = metadataService->getTimeRangeAsync("sea_surface_temperature");
auto timeRange = timeRangeFuture.get();

if (timeRange) {
    std::cout << "时间范围: " << timeRange->first << " - " << timeRange->second << std::endl;
}
```

## 4. 空间操作服务 API (Spatial Operations Service)

### 4.1 接口定义

```cpp
namespace oscean::core_services::spatial_ops {

class ISpatialOpsService {
public:
    virtual ~ISpatialOpsService() = default;
    
    // 服务管理
    virtual std::future<void> setConfiguration(const SpatialOpsConfig& config) = 0;
    virtual std::future<SpatialOpsConfig> getConfiguration() const = 0;
    virtual std::future<std::vector<std::string>> getCapabilities() const = 0;
    virtual std::string getVersion() const = 0;
    virtual bool isReady() const = 0;
    
    // 基础几何运算
    virtual std::future<Geometry> buffer(
        const Geometry& geom,
        double distance,
        const BufferOptions& options = {}
    ) const = 0;
    
    virtual std::future<Geometry> intersection(
        const Geometry& geom1,
        const Geometry& geom2
    ) const = 0;
    
    virtual std::future<Geometry> difference(
        const Geometry& geom1,
        const Geometry& geom2
    ) const = 0;
    
    virtual std::future<Geometry> union_(
        const Geometry& geom1,
        const Geometry& geom2
    ) const = 0;
    
    // 空间关系查询
    virtual std::future<bool> intersects(
        const Geometry& geom1,
        const Geometry& geom2
    ) const = 0;
    
    virtual std::future<bool> contains(
        const Geometry& geom1,
        const Geometry& geom2
    ) const = 0;
    
    virtual std::future<double> distance(
        const Geometry& geom1,
        const Geometry& geom2
    ) const = 0;
    
    // 栅格操作
    virtual std::future<GridData> clipRasterByGeometry(
        const GridData& raster,
        const Geometry& clipGeometry,
        const RasterClipOptions& options = {}
    ) const = 0;
    
    virtual std::future<GridData> resampleRaster(
        const GridData& input,
        const GridDefinition& targetGrid,
        const ResampleOptions& options = {}
    ) const = 0;
    
    // 空间查询
    virtual std::future<std::vector<Feature>> spatialQuery(
        const std::vector<Feature>& features,
        const Geometry& queryGeometry,
        const SpatialQueryOptions& options = {}
    ) const = 0;
    
    virtual std::future<std::vector<GridIndex>> findGridCellsInGeometry(
        const GridDefinition& grid,
        const Geometry& geometry
    ) const = 0;
};

}
```

### 4.2 配置选项

```cpp
struct SpatialOpsConfig {
    // 并行设置
    size_t maxThreads = 0;  // 0表示自动检测
    bool enableParallelProcessing = true;
    
    // 性能设置
    bool enablePerformanceMonitoring = false;
    bool enableExperimentalFeatures = false;
    
    // 内存设置
    size_t maxMemoryUsageMB = 1024;
    bool enableMemoryOptimization = true;
    
    // GDAL设置
    std::map<std::string, std::string> gdalOptions;
};
```

### 4.3 使用示例

```cpp
// 创建空间操作服务实例
auto spatialService = std::make_shared<SpatialOpsServiceImpl>();

// 设置配置
SpatialOpsConfig config;
config.maxThreads = 4;
config.enablePerformanceMonitoring = true;
spatialService->setConfiguration(config).get();

// 几何缓冲区计算
Geometry polygon; // 假设已初始化
BufferOptions bufferOpts;
bufferOpts.distance = 1000.0;  // 1000米
bufferOpts.segments = 32;

auto bufferFuture = spatialService->buffer(polygon, 1000.0, bufferOpts);
Geometry bufferedGeom = bufferFuture.get();

// 栅格裁剪
GridData rasterData; // 假设已加载
Geometry clipGeom;   // 裁剪几何

auto clippedFuture = spatialService->clipRasterByGeometry(rasterData, clipGeom);
GridData clippedRaster = clippedFuture.get();
```

## 5. 插值服务 API (Interpolation Service)

### 5.1 接口定义

```cpp
namespace oscean::core_services {

class IInterpolationService {
public:
    virtual ~IInterpolationService() = default;
    
    // 基础信息
    virtual std::string getVersion() const = 0;
    virtual std::vector<std::string> getSupportedMethods() const = 0;
    
    // 点插值
    virtual std::future<InterpolationResult> interpolateAtPointsAsync(
        const InterpolationRequest& request
    ) = 0;
    
    // 网格插值
    virtual std::future<InterpolationResult> interpolateToGridAsync(
        const InterpolationRequest& request
    ) = 0;
    
    // 批量插值
    virtual std::future<std::vector<InterpolationResult>> batchInterpolateAsync(
        const std::vector<InterpolationRequest>& requests
    ) = 0;
    
    // 算法信息
    virtual AlgorithmInfo getAlgorithmInfo(InterpolationMethod method) const = 0;
    virtual std::vector<AlgorithmInfo> getAllAlgorithmInfo() const = 0;
};

}
```

### 5.2 插值请求和结果

```cpp
// 插值方法
enum class InterpolationMethod {
    UNKNOWN, NEAREST_NEIGHBOR, BILINEAR, BICUBIC, 
    INVERSE_DISTANCE_WEIGHTING, KRIGING, SPLINE, 
    NATURAL_NEIGHBOR, LINEAR_TIME, CUBIC_TIME
};

// 插值请求
struct InterpolationRequest {
    std::shared_ptr<GridData> sourceGrid;
    std::variant<std::vector<TargetPoint>, TargetGridDefinition> target;
    InterpolationMethod method = InterpolationMethod::UNKNOWN;
    AlgorithmParameters algorithmParams;
    DataType desiredOutputValueType = DataType::Float32;
};

// 插值结果
using InterpolationResultData = std::variant<
    std::monostate,
    GridData,
    std::vector<std::optional<double>>
>;

struct InterpolationResult {
    InterpolationResultData data;
    int statusCode = 0;
    std::string message;
};
```

### 5.3 使用示例

```cpp
// 创建插值服务实例
auto interpolationService = std::make_shared<InterpolationServiceImpl>();

// 准备插值请求
InterpolationRequest request;
request.sourceGrid = std::make_shared<GridData>(sourceData);
request.method = InterpolationMethod::BILINEAR;

// 目标点
std::vector<TargetPoint> targetPoints = {
    {{116.3974, 39.9093}, "EPSG:4326"},
    {{121.4737, 31.2304}, "EPSG:4326"}
};
request.target = targetPoints;

// 执行插值
auto resultFuture = interpolationService->interpolateAtPointsAsync(request);
auto result = resultFuture.get();

if (result.statusCode == 0) {
    auto values = std::get<std::vector<std::optional<double>>>(result.data);
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i]) {
            std::cout << "点 " << i << " 插值结果: " << *values[i] << std::endl;
        }
    }
}
```

## 6. CRS服务 API (Coordinate Reference System Service)

### 6.1 接口定义

```cpp
namespace oscean::core_services {

class ICrsService {
public:
    virtual ~ICrsService() = default;
    
    // 基础信息
    virtual std::string getVersion() const = 0;
    
    // CRS解析
    virtual std::optional<CRSInfo> parseFromWKT(const std::string& wkt) = 0;
    virtual std::optional<CRSInfo> parseFromEpsgCode(int epsgCode) = 0;
    virtual std::optional<CRSInfo> parseFromProj4(const std::string& proj4) = 0;
    
    // 坐标转换
    virtual TransformResult transformPoint(
        double x, double y,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS
    ) = 0;
    
    virtual std::future<TransformResult> transformPointAsync(
        double x, double y,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS
    ) = 0;
    
    virtual BoundingBox transformBoundingBox(
        const BoundingBox& bbox,
        const CRSInfo& targetCRS
    ) = 0;
    
    virtual std::vector<TransformResult> transformPoints(
        const std::vector<std::pair<double, double>>& points,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS
    ) = 0;
    
    // 性能统计
    virtual PerformanceStats getPerformanceStats() const = 0;
    virtual void resetPerformanceStats() = 0;
};

}
```

### 6.2 转换结果

```cpp
enum class TransformStatus {
    Success, InvalidSourceCRS, InvalidTargetCRS, 
    TransformationFailed, OutOfBounds
};

struct TransformResult {
    double x, y;
    std::optional<double> z;
    TransformStatus status;
    std::string errorMessage;
};

struct PerformanceStats {
    size_t totalParsings = 0;
    size_t totalTransformations = 0;
    double averageParsingTime = 0.0;
    double averageTransformTime = 0.0;
    size_t cacheHits = 0;
    size_t cacheMisses = 0;
    double cacheHitRatio = 0.0;
};
```

### 6.3 使用示例

```cpp
// 创建CRS服务实例
auto crsService = std::make_shared<CrsServiceImpl>();

// 解析CRS
auto wgs84 = crsService->parseFromEpsgCode(4326);
auto webMercator = crsService->parseFromEpsgCode(3857);

if (wgs84 && webMercator) {
    // 坐标转换
    auto result = crsService->transformPoint(
        116.3974, 39.9093,  // 北京坐标
        *wgs84, *webMercator
    );
    
    if (result.status == TransformStatus::Success) {
        std::cout << "转换结果: (" << result.x << ", " << result.y << ")" << std::endl;
    }
    
    // 异步转换
    auto asyncFuture = crsService->transformPointAsync(
        121.4737, 31.2304,  // 上海坐标
        *wgs84, *webMercator
    );
    
    auto asyncResult = asyncFuture.get();
    if (asyncResult.status == TransformStatus::Success) {
        std::cout << "异步转换结果: (" << asyncResult.x << ", " << asyncResult.y << ")" << std::endl;
    }
}
```

## 7. 通用工具库 API (Common Utilities)

### 7.1 性能监控

```cpp
namespace oscean::common_utils::performance {

class PerformanceMonitor {
public:
    // 计时器管理
    static void startTimer(const std::string& name);
    static void stopTimer(const std::string& name);
    static double getTimerValue(const std::string& name);
    
    // 内存监控
    static size_t getCurrentMemoryUsage();
    static size_t getPeakMemoryUsage();
    
    // 报告生成
    static PerformanceReport generateReport();
};

// 作用域计时器
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();
};

}
```

### 7.2 缓存系统

```cpp
namespace oscean::common_utils::cache {

template<typename Key, typename Value>
class LRUCache {
public:
    explicit LRUCache(size_t capacity);
    
    bool put(const Key& key, const Value& value);
    std::optional<Value> get(const Key& key);
    void clear();
    size_t size() const;
    double getHitRatio() const;
};

class DiskCache {
public:
    struct CacheConfig {
        std::string cacheDirectory;
        size_t maxSizeBytes;
        size_t maxFileAge = 3600; // 秒
    };
    
    explicit DiskCache(const CacheConfig& config);
    
    bool put(const std::string& key, const std::vector<uint8_t>& data);
    std::optional<std::vector<uint8_t>> get(const std::string& key);
    void cleanup();
};

}
```

### 7.3 并行处理

```cpp
namespace oscean::common_utils::parallel {

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads);
    ~ThreadPool();
    
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;
    
    void shutdown();
    size_t getThreadCount() const;
};

class GlobalThreadPoolRegistry {
public:
    static std::shared_ptr<ThreadPoolManager> getGlobalManager();
    static void initializeThreadPool(const std::string& name, size_t numThreads);
    static std::shared_ptr<ThreadPool> getThreadPool(const std::string& name);
};

}
```

## 8. 输出生成服务 API (Output Generation Service)

### 8.1 瓦片服务

```cpp
namespace oscean::output_generation::tile_service {

class ITileService {
public:
    virtual ~ITileService() = default;
    
    // 瓦片生成
    virtual std::future<TileData> generateTileAsync(
        const TileRequest& request
    ) = 0;
    
    // 批量瓦片生成
    virtual std::future<std::vector<TileData>> generateTilesAsync(
        const std::vector<TileRequest>& requests
    ) = 0;
    
    // 缓存管理
    virtual void clearCache() = 0;
    virtual CacheStats getCacheStats() const = 0;
};

struct TileRequest {
    int z, x, y;                    // 瓦片坐标
    std::string layerName;          // 图层名称
    std::string format = "png";     // 输出格式
    std::map<std::string, std::string> parameters; // 额外参数
};

struct TileData {
    std::vector<uint8_t> data;      // 瓦片数据
    std::string mimeType;           // MIME类型
    std::string format;             // 格式
    size_t sizeBytes;               // 大小
};

}
```

### 8.2 文件生成服务

```cpp
namespace oscean::output_generation {

class IFileGenerator {
public:
    virtual ~IFileGenerator() = default;
    
    // NetCDF文件生成
    virtual std::future<bool> generateNetCDFAsync(
        const GridData& data,
        const std::string& outputPath,
        const NetCDFOptions& options = {}
    ) = 0;
    
    // GeoTIFF文件生成
    virtual std::future<bool> generateGeoTIFFAsync(
        const GridData& data,
        const std::string& outputPath,
        const GeoTIFFOptions& options = {}
    ) = 0;
    
    // CSV文件生成
    virtual std::future<bool> generateCSVAsync(
        const std::vector<Feature>& features,
        const std::string& outputPath,
        const CSVOptions& options = {}
    ) = 0;
};

}
```

## 9. 错误处理和异常

### 9.1 异常类型

```cpp
namespace oscean::core_services {

// 基础异常类
class OSCEANException : public std::exception {
public:
    explicit OSCEANException(const std::string& message);
    const char* what() const noexcept override;
};

// 数据访问异常
class DataAccessException : public OSCEANException {
public:
    explicit DataAccessException(const std::string& message);
};

// 空间操作异常
class SpatialOperationException : public OSCEANException {
public:
    explicit SpatialOperationException(const std::string& message);
};

// CRS异常
class CRSException : public OSCEANException {
public:
    explicit CRSException(const std::string& message);
};

// 插值异常
class InterpolationException : public OSCEANException {
public:
    explicit InterpolationException(const std::string& message);
};

}
```

### 9.2 错误码

```cpp
enum class ErrorCode {
    Success = 0,
    
    // 通用错误 (1-99)
    UnknownError = 1,
    InvalidParameter = 2,
    OutOfMemory = 3,
    Timeout = 4,
    
    // 数据访问错误 (100-199)
    FileNotFound = 100,
    FileFormatNotSupported = 101,
    VariableNotFound = 102,
    InvalidDataRange = 103,
    
    // 空间操作错误 (200-299)
    InvalidGeometry = 200,
    SpatialOperationFailed = 201,
    CRSTransformationFailed = 202,
    
    // 插值错误 (300-399)
    InterpolationMethodNotSupported = 300,
    InsufficientDataPoints = 301,
    InterpolationFailed = 302
};
```

## 10. 配置和初始化

### 10.1 全局配置

```cpp
namespace oscean {

struct OSCEANConfig {
    // 线程池配置
    size_t defaultThreadPoolSize = 0;  // 0表示自动检测
    
    // 缓存配置
    size_t memoryCacheSizeMB = 512;
    std::string diskCacheDirectory = "./cache";
    size_t diskCacheSizeMB = 2048;
    
    // 日志配置
    std::string logLevel = "INFO";
    std::string logFile = "";
    
    // GDAL配置
    std::map<std::string, std::string> gdalOptions;
    
    // 性能配置
    bool enablePerformanceMonitoring = false;
    bool enableMemoryOptimization = true;
};

class OSCEAN {
public:
    static bool initialize(const OSCEANConfig& config = {});
    static void shutdown();
    static bool isInitialized();
    static OSCEANConfig getConfig();
};

}
```

### 10.2 服务工厂

```cpp
namespace oscean::core_services {

class ServiceFactory {
public:
    // 数据访问服务
    static std::shared_ptr<IRawDataAccessService> createDataAccessService();
    
    // 元数据服务
    static std::shared_ptr<IMetadataService> createMetadataService();
    
    // 空间操作服务
    static std::shared_ptr<spatial_ops::ISpatialOpsService> createSpatialOpsService();
    
    // 插值服务
    static std::shared_ptr<IInterpolationService> createInterpolationService();
    
    // CRS服务
    static std::shared_ptr<ICrsService> createCrsService();
};

}
```

## 11. 使用示例和最佳实践

### 11.1 完整使用示例

```cpp
#include "oscean/oscean.h"
#include "core_services/service_factory.h"

int main() {
    try {
        // 1. 初始化OSCEAN
        oscean::OSCEANConfig config;
        config.defaultThreadPoolSize = 4;
        config.memoryCacheSizeMB = 1024;
        config.enablePerformanceMonitoring = true;
        
        if (!oscean::OSCEAN::initialize(config)) {
            std::cerr << "OSCEAN初始化失败" << std::endl;
            return 1;
        }
        
        // 2. 创建服务实例
        auto dataService = oscean::core_services::ServiceFactory::createDataAccessService();
        auto metadataService = oscean::core_services::ServiceFactory::createMetadataService();
        auto spatialService = oscean::core_services::ServiceFactory::createSpatialOpsService();
        auto interpolationService = oscean::core_services::ServiceFactory::createInterpolationService();
        auto crsService = oscean::core_services::ServiceFactory::createCrsService();
        
        // 3. 执行数据处理工作流
        
        // 3.1 查找数据文件
        oscean::core_services::QueryCriteria criteria;
        criteria.spatialExtent = oscean::core_services::BoundingBox(
            110.0, 30.0, 120.0, 40.0, "EPSG:4326"
        );
        criteria.variables = {"sea_surface_temperature"};
        
        auto filesFuture = metadataService->findFilesAsync(criteria);
        auto files = filesFuture.get();
        
        if (files.empty()) {
            std::cout << "未找到匹配的数据文件" << std::endl;
            return 0;
        }
        
        // 3.2 读取网格数据
        auto gridFuture = dataService->readGridVariableSubsetAsync(
            files[0].path,
            "sea_surface_temperature",
            std::nullopt,  // 所有时间
            criteria.spatialExtent
        );
        
        auto gridData = gridFuture.get();
        std::cout << "读取网格数据: " << gridData.getDefinition().cols 
                  << "x" << gridData.getDefinition().rows << std::endl;
        
        // 3.3 执行插值
        oscean::core_services::InterpolationRequest interpRequest;
        interpRequest.sourceGrid = std::make_shared<oscean::core_services::GridData>(gridData);
        interpRequest.method = oscean::core_services::InterpolationMethod::BILINEAR;
        
        std::vector<oscean::core_services::TargetPoint> targetPoints = {
            {{116.3974, 39.9093}, "EPSG:4326"},  // 北京
            {{121.4737, 31.2304}, "EPSG:4326"}   // 上海
        };
        interpRequest.target = targetPoints;
        
        auto interpFuture = interpolationService->interpolateAtPointsAsync(interpRequest);
        auto interpResult = interpFuture.get();
        
        if (interpResult.statusCode == 0) {
            auto values = std::get<std::vector<std::optional<double>>>(interpResult.data);
            for (size_t i = 0; i < values.size(); ++i) {
                if (values[i]) {
                    std::cout << "点 " << i << " 的海表温度: " << *values[i] << "°C" << std::endl;
                }
            }
        }
        
        // 4. 清理资源
        oscean::OSCEAN::shutdown();
        
    } catch (const std::exception& e) {
        std::cerr << "处理过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### 11.2 最佳实践

1. **资源管理**
   - 始终使用智能指针管理服务实例
   - 在程序结束时调用`OSCEAN::shutdown()`
   - 合理设置缓存大小以平衡性能和内存使用

2. **异步操作**
   - 优先使用异步API以提高并发性能
   - 合理处理future的异常
   - 避免阻塞主线程

3. **错误处理**
   - 总是检查操作结果的状态码
   - 使用try-catch捕获异常
   - 提供有意义的错误信息

4. **性能优化**
   - 启用性能监控以识别瓶颈
   - 合理使用缓存
   - 根据数据大小调整线程池大小

## 12. 版本信息和兼容性

- **当前版本**: 1.0.0
- **API稳定性**: 核心API已稳定，扩展API可能会有变化
- **向后兼容性**: 主版本号变更时可能破坏兼容性
- **依赖要求**: 
  - C++17或更高版本
  - Boost 1.70+
  - GDAL 3.0+
  - NetCDF 4.6+

## 总结

本文档提供了OSCEAN项目的完整API接口指南。各个模块的API设计遵循现代C++最佳实践，提供了同步和异步两种接口形式，支持灵活的配置和扩展。通过合理使用这些API，可以构建高性能的海洋数据处理和分析应用。 