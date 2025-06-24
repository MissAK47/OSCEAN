# OSCEAN 统一服务模式重构指导文档 - 设计文档符合版本

## 0. 重要说明：与设计文档的一致性

### 0.1 设计文档要求回顾
根据《设计.md》文档的明确要求：
- **异步模型**：基于Boost.Asio的异步I/O模型 + ThreadPool处理CPU密集任务
- **异步技术**：使用`std::async`, `std::packaged_task`, `std::future`或ThreadPool
- **错误处理**：优先使用`std::error_code`和自定义`std::error_category`进行错误传递
- **数据流**：优先考虑流式处理或分块处理大数据

### 0.2 重构方案调整
**原方案问题**：选择boost::future违背了设计文档要求
**修正方案**：回归std::future，但统一异步模式和错误处理架构

## 1. 现状分析

### 1.1 当前服务模式不一致性问题

| 模块 | 当前创建模式 | 当前接口模式 | 异步技术 | 错误处理 | 问题 |
|------|-------------|-------------|---------|---------|------|
| **CRS服务** | 简单工厂函数 | 混合模式 | 混用 | 不统一 | 需要完全重构 |
| **数据访问服务** | 参数化工厂函数 | boost::future | boost | 不统一 | 需要改为std::future |
| **空间操作服务** | **工厂类** | std::future | std | 不统一 | 工厂符合，需要统一错误处理 |
| **插值服务** | 构造函数注入 | std::future | std | 不统一 | 需要工厂类和错误处理 |
| **元数据服务** | 构造函数注入 | 混合模式 | 混用 | 不统一 | 需要完全重构 |
| **模拟服务** | 构造函数注入 | std::future | std | 不统一 | 需要工厂类和错误处理 |
| **工作流引擎** | 直接构造 | 混合模式 | 混用 | 不统一 | 严重混用问题 |

### 1.2 目标统一标准（符合设计文档）

- **创建模式**: 统一使用**工厂类模式**
- **接口模式**: 统一使用**std::future全异步模式**
- **异步技术**: 统一使用**std::async/std::packaged_task/ThreadPool**
- **错误处理**: 统一使用**std::error_code + 自定义error_category**
- **线程模型**: **Boost.Asio I/O线程 + std::future + ThreadPool**

## 2. 统一异步架构标准（设计文档兼容）

### 2.1 全局异步配置

```cpp
// 文件: core_service_interfaces/include/core_services/unified_async_config.h
#pragma once

#ifndef OSCEAN_UNIFIED_ASYNC_CONFIG_H
#define OSCEAN_UNIFIED_ASYNC_CONFIG_H

// 标准库异步支持
#include <future>
#include <async>
#include <thread>
#include <functional>

// Boost.Asio集成支持（网络层）
#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>

// 错误处理支持
#include <system_error>

namespace oscean::core_services {

// ============================================================================
// 统一的异步类型别名（符合设计文档）
// ============================================================================

template<typename T>
using Future = std::future<T>;

template<typename T>
using Promise = std::promise<T>;

template<typename T>
using PackagedTask = std::packaged_task<T>;

// ============================================================================
// 统一的错误处理架构（设计文档要求）
// ============================================================================

// 核心服务错误类别
enum class CoreServiceError {
    Success = 0,
    InvalidInput,
    ServiceUnavailable,
    OperationFailed,
    DataAccessError,
    NetworkError,
    ConfigurationError,
    InternalError
};

// 自定义错误类别（符合设计文档要求）
class CoreServiceErrorCategory : public std::error_category {
public:
    const char* name() const noexcept override;
    std::string message(int ev) const override;
};

// 错误码生成函数
std::error_code make_error_code(CoreServiceError e);

// ============================================================================
// 异步结果包装器（统一错误处理）
// ============================================================================

template<typename T>
struct AsyncResult {
    T data;
    std::error_code error;
    
    bool has_value() const { return !error; }
    bool has_error() const { return static_cast<bool>(error); }
    
    // 如果有错误则抛出异常
    T& value() {
        if (error) {
            throw std::system_error(error);
        }
        return data;
    }
    
    // 安全访问，返回optional
    std::optional<T> value_or_none() const {
        return error ? std::nullopt : std::make_optional(data);
    }
};

// ============================================================================
// 流式数据处理支持（设计文档要求）
// ============================================================================

// 数据块类型定义
template<typename T>
struct DataChunk {
    T data;
    size_t chunkIndex;
    size_t totalChunks;
    bool isLast;
    std::error_code error;
    
    bool has_error() const { return static_cast<bool>(error); }
    bool is_complete() const { return isLast && !has_error(); }
};

// 流式数据回调类型
template<typename T>
using StreamCallback = std::function<void(DataChunk<T>)>;

// 流式数据接口
template<typename T>
class DataStream {
public:
    virtual ~DataStream() = default;
    
    // 设置数据块回调
    virtual void setChunkCallback(StreamCallback<T> callback) = 0;
    
    // 开始流式读取
    virtual Future<AsyncResult<void>> startStreaming() = 0;
    
    // 取消流式读取
    virtual void cancel() = 0;
    
    // 获取流状态
    virtual bool isActive() const = 0;
    virtual size_t getBytesRead() const = 0;
    virtual size_t getTotalSize() const = 0;
};

// 流式结果包装器
template<typename T>
struct StreamingResult {
    std::shared_ptr<DataStream<T>> stream;
    std::error_code error;
    
    bool has_value() const { return stream && !error; }
    bool has_error() const { return static_cast<bool>(error); }
    
    std::shared_ptr<DataStream<T>>& value() {
        if (error) {
            throw std::system_error(error);
        }
        return stream;
    }
};

// ============================================================================
// 大数据处理工具函数
// ============================================================================

// 创建流式数据处理任务
template<typename T, typename ReaderFunc>
auto create_streaming_task(boost::asio::thread_pool& pool, 
                          ReaderFunc&& reader,
                          size_t chunkSize = 1024 * 1024) // 默认1MB块
    -> Future<StreamingResult<T>> {
    
    auto promise = std::make_shared<std::promise<StreamingResult<T>>>();
    auto future = promise->get_future();
    
    // 在线程池中执行流式读取
    boost::asio::post(pool, [promise, reader = std::forward<ReaderFunc>(reader), chunkSize]() mutable {
        try {
            auto stream = std::make_shared<DefaultDataStream<T>>(std::move(reader), chunkSize);
            promise->set_value(StreamingResult<T>{stream, {}});
        } catch (const std::system_error& e) {
            promise->set_value(StreamingResult<T>{nullptr, e.code()});
        } catch (const std::exception& e) {
            promise->set_value(StreamingResult<T>{nullptr, make_error_code(CoreServiceError::InternalError)});
        }
    });
    
    return future;
}

// 分块数据处理器
template<typename InputType, typename OutputType>
class ChunkProcessor {
public:
    virtual ~ChunkProcessor() = default;
    
    // 处理单个数据块
    virtual Future<AsyncResult<OutputType>> processChunk(
        const DataChunk<InputType>& chunk) = 0;
    
    // 聚合处理结果
    virtual Future<AsyncResult<OutputType>> finalizeResult(
        const std::vector<OutputType>& chunkResults) = 0;
};

// ============================================================================
// 网络层流式集成（Boost.Asio兼容）
// ============================================================================

// HTTP流式响应生成器
class StreamingResponseGenerator {
public:
    virtual ~StreamingResponseGenerator() = default;
    
    // 开始流式响应
    virtual void startStreaming(
        boost::asio::io_context& ioc,
        std::function<void(const std::vector<char>&)> writeCallback,
        std::function<void(std::error_code)> completeCallback) = 0;
    
    // 设置响应头
    virtual void setHeaders(const std::map<std::string, std::string>& headers) = 0;
    
    // 设置内容类型
    virtual void setContentType(const std::string& contentType) = 0;
    
    // 启用分块传输编码
    virtual void enableChunkedTransfer(bool enable = true) = 0;
};

// 创建流式响应
template<typename T>
std::unique_ptr<StreamingResponseGenerator> create_streaming_response(
    std::shared_ptr<DataStream<T>> dataStream,
    std::function<std::vector<char>(const T&)> serializer) {
    
    return std::make_unique<DefaultStreamingResponseGenerator<T>>(
        std::move(dataStream), 
        std::move(serializer)
    );
}

// ============================================================================
// 大文件处理专用接口
// ============================================================================

// 大型NetCDF文件流式读取器
class NetCDFStreamReader : public DataStream<GridData> {
private:
    std::string filePath_;
    std::string variableName_;
    size_t chunkSize_;
    
public:
    NetCDFStreamReader(const std::string& filePath, 
                      const std::string& variableName,
                      size_t chunkSize = 1024 * 1024);
    
    void setChunkCallback(StreamCallback<GridData> callback) override;
    Future<AsyncResult<void>> startStreaming() override;
    void cancel() override;
    bool isActive() const override;
    size_t getBytesRead() const override;
    size_t getTotalSize() const override;
};

// 大型栅格瓦片流式生成器
class RasterTileStreamGenerator : public DataStream<std::vector<unsigned char>> {
private:
    GridData sourceData_;
    TileRequest request_;
    size_t tileSize_;
    
public:
    RasterTileStreamGenerator(const GridData& sourceData, 
                            const TileRequest& request);
    
    void setChunkCallback(StreamCallback<std::vector<unsigned char>> callback) override;
    Future<AsyncResult<void>> startStreaming() override;
    void cancel() override;
    bool isActive() const override;
    size_t getBytesRead() const override;
    size_t getTotalSize() const override;
};

// ============================================================================
// 内存优化和资源管理
// ============================================================================

// 内存使用监控器
class MemoryMonitor {
public:
    static size_t getCurrentMemoryUsage();
    static size_t getMaxMemoryUsage();
    static bool isMemoryPressure();
    static void setMemoryLimit(size_t limitBytes);
};

// 自适应块大小管理器
class AdaptiveChunkSizeManager {
private:
    size_t baseChunkSize_;
    size_t maxChunkSize_;
    size_t minChunkSize_;
    
public:
    AdaptiveChunkSizeManager(size_t baseSize = 1024 * 1024);
    
    // 根据当前内存状况调整块大小
    size_t getOptimalChunkSize();
    
    // 根据处理性能调整块大小
    void adjustBasedOnPerformance(std::chrono::milliseconds processingTime);
};

} // namespace oscean::core_services

// 使std::error_code支持我们的枚举
namespace std {
    template<>
    struct is_error_code_enum<oscean::core_services::CoreServiceError> : true_type {};
}

#endif // OSCEAN_UNIFIED_ASYNC_CONFIG_H
```

### 2.2 强制包含规则

**所有服务相关文件必须首先包含**：
```cpp
#include "core_services/unified_async_config.h"
// 然后才能包含其他头文件
```

## 3. 统一工厂类标准（设计文档兼容版本）

### 3.1 工厂类标准结构

```cpp
#pragma once

// 必须首先包含统一配置
#include "core_services/unified_async_config.h"
#include "i_[service]_service.h"
#include "[service]_config.h"
#include <memory>

namespace oscean::core_services::[service] {

/**
 * @brief [Service]服务工厂类 - 设计文档兼容版本
 * 负责创建和配置[Service]服务实例，管理依赖注入和初始化过程
 * 所有创建的服务都使用std::future异步接口和统一错误处理
 */
class [Service]ServiceFactory {
public:
    /**
     * @brief 创建服务实例（使用默认配置）
     * @return 服务实例的智能指针，使用std::future异步接口
     */
    static std::unique_ptr<I[Service]Service> createService();
    
    /**
     * @brief 创建服务实例（使用指定配置）
     * @param config 服务配置
     * @return 服务实例的智能指针，使用std::future异步接口
     */
    static std::unique_ptr<I[Service]Service> createService(
        const [Service]Config& config);
    
    /**
     * @brief 创建服务实例（完整依赖注入）
     * @param config 服务配置
     * @param dependencies 依赖服务实例（必须也是std::future版本）
     * @param threadPool 共享线程池（符合设计文档）
     * @return 服务实例的智能指针，使用std::future异步接口
     */
    static std::unique_ptr<I[Service]Service> createService(
        const [Service]Config& config,
        const [Service]Dependencies& dependencies,
        std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);
    
    /**
     * @brief 创建用于测试的Mock服务实例
     * @return Mock服务实例的智能指针，使用std::future异步接口
     */
    static std::unique_ptr<I[Service]Service> createMockService();
    
    /**
     * @brief 验证配置的有效性
     * @param config 要验证的配置
     * @return Future<AsyncResult<bool>> 异步验证结果，包含错误信息
     */
    static Future<AsyncResult<bool>> validateConfigAsync(const [Service]Config& config);

private:
    // 禁止实例化
    [Service]ServiceFactory() = delete;
    ~[Service]ServiceFactory() = delete;
    [Service]ServiceFactory(const [Service]ServiceFactory&) = delete;
    [Service]ServiceFactory& operator=(const [Service]ServiceFactory&) = delete;
};

} // namespace oscean::core_services::[service]
```

## 4. 统一异步接口标准（设计文档兼容版本）

### 4.1 异步方法标准格式

```cpp
/**
 * @brief 异步操作方法 - 设计文档兼容版本
 * @param param 操作参数
 * @return std::future<AsyncResult<Result>> 异步结果，包含统一错误处理
 * @note 使用std::error_code进行错误传递，符合设计文档要求
 */
virtual Future<AsyncResult<Result>> operationAsync(const Param& param) = 0;
```

### 4.2 与Boost.Asio集成的异步模式

```cpp
// 1. 在服务实现中使用ThreadPool（符合设计文档）
class CrsServiceImpl : public ICrsService {
private:
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    
public:
    Future<AsyncResult<TransformedPoint>> transformPointAsync(
        double x, double y, 
        const CRSInfo& sourceCRS, 
        const CRSInfo& targetCRS) override {
        
        return submit_to_pool(*threadPool_, [=]() -> TransformedPoint {
            // 实际的坐标转换逻辑
            return performTransformation(x, y, sourceCRS, targetCRS);
        });
    }
};

// 2. 与网络层的集成（Boost.Asio兼容）
class TaskDispatcher {
    Future<AsyncResult<ResponseData>> processRequest(const RequestDTO& request) {
        // 调用核心服务的std::future接口
        auto crsResult = crsService_->transformPointAsync(x, y, src, dst);
        
        // 结果处理和错误传播
        return crsResult.then([](auto future) {
            auto result = future.get();
            if (result.has_error()) {
                return make_error_future<ResponseData>(result.error);
            }
            return make_ready_future(createResponse(result.data));
        });
    }
};
```

### 4.3 统一错误处理模式（设计文档要求）

```cpp
// 使用std::error_code进行错误传递
auto future = service->operationAsync(params);
auto result = future.get();

if (result.has_error()) {
    // 统一的错误处理
    std::error_code ec = result.error;
    if (ec == CoreServiceError::InvalidInput) {
        // 处理输入错误
    } else if (ec == CoreServiceError::ServiceUnavailable) {
        // 处理服务不可用
    }
} else {
    // 处理成功结果
    auto data = result.value();
}

// 错误传播和转换
auto chainedResult = service1->operationAsync(params)
    .then([this](auto future) -> Future<AsyncResult<FinalResult>> {
        auto result = future.get();
        if (result.has_error()) {
            return make_error_future<FinalResult>(result.error);
        }
        return service2->processAsync(result.data);
    });
```

### 4.4 流式处理接口标准（设计文档核心要求）

```cpp
/**
 * @brief 流式数据访问接口 - 解决大数据处理核心问题
 * 支持分块读取、避免一次性加载大文件
 */
class IRawDataAccessService {
public:
    // 传统完整数据读取（小文件）
    virtual Future<AsyncResult<GridData>> readGridVariableAsync(
        const std::string& filePath,
        const std::string& variableName) = 0;
    
    // 流式数据读取（大文件） - 核心新增功能
    virtual Future<StreamingResult<GridData>> readGridVariableStreamAsync(
        const std::string& filePath,
        const std::string& variableName,
        size_t chunkSizeBytes = 1024 * 1024) = 0;
    
    // 分块范围读取（精确控制）
    virtual Future<AsyncResult<GridData>> readGridVariableChunkAsync(
        const std::string& filePath,
        const std::string& variableName,
        const ChunkRange& chunkRange) = 0;
    
    // 渐进式瓦片生成（栅格服务核心）
    virtual Future<StreamingResult<std::vector<unsigned char>>> generateTileStreamAsync(
        const GridData& sourceData,
        const TileRequest& request,
        const TileFormat& format) = 0;
};

/**
 * @brief 空间操作流式接口 - 处理大数据集空间运算
 */
class ISpatialOpsService {
public:
    // 流式空间查询（避免一次性加载所有要素）
    virtual Future<StreamingResult<Feature>> spatialQueryStreamAsync(
        const std::vector<std::string>& layerPaths,
        const SpatialQuery& query,
        size_t featureBatchSize = 1000) = 0;
    
    // 分块栅格运算（大栅格数据处理）
    virtual Future<StreamingResult<GridData>> rasterOperationStreamAsync(
        const std::vector<GridData>& inputs,
        const RasterOperation& operation,
        size_t processingChunkSize = 1024 * 1024) = 0;
    
    // 渐进式缓冲区分析（大数据集）
    virtual Future<StreamingResult<Feature>> bufferAnalysisStreamAsync(
        const std::vector<Feature>& features,
        double bufferDistance,
        size_t processingBatchSize = 100) = 0;
};

/**
 * @brief 插值服务流式接口 - 大规模插值计算
 */
class IInterpolationService {
public:
    // 流式插值计算（避免内存溢出）
    virtual Future<StreamingResult<GridData>> interpolateStreamAsync(
        const std::vector<DataPoint>& inputPoints,
        const GridDefinition& outputGrid,
        const InterpolationMethod& method,
        size_t calculationChunkSize = 10000) = 0;
    
    // 分区域插值（超大区域处理）
    virtual Future<StreamingResult<GridData>> interpolateByRegionStreamAsync(
        const std::vector<DataPoint>& inputPoints,
        const std::vector<BoundingBox>& regions,
        const InterpolationMethod& method) = 0;
};
```

### 4.5 网络层流式集成模式（Chunked Transfer Encoding）

```cpp
/**
 * @brief 网络层流式响应控制器 - 实现Chunked Encoding
 * 与核心服务的DataStream无缝集成
 */
class StreamingController {
private:
    boost::asio::io_context& ioContext_;
    std::shared_ptr<boost::asio::thread_pool> workerPool_;
    
public:
    // 处理大文件下载请求
    template<typename T>
    void handleStreamingRequest(
        const HttpRequest& request,
        HttpResponse& response,
        std::shared_ptr<DataStream<T>> dataStream) {
        
        // 设置Chunked Transfer Encoding
        response.set_header("Transfer-Encoding", "chunked");
        response.set_header("Content-Type", determineContentType<T>());
        
        // 设置流式数据回调
        dataStream->setChunkCallback([&response](const DataChunk<T>& chunk) {
            if (chunk.has_error()) {
                response.write_error_chunk(chunk.error);
                return;
            }
            
            // 序列化数据块
            auto serializedData = serialize(chunk.data);
            
            // 写入HTTP块
            response.write_chunk(serializedData);
            
            if (chunk.isLast) {
                response.write_final_chunk();
            }
        });
        
        // 在IO context中启动流式传输
        submit_to_io_context(ioContext_, [dataStream]() {
            return dataStream->startStreaming();
        });
    }
    
    // 处理瓦片流式生成
    void handleTileStreamRequest(
        const TileRequest& tileRequest,
        HttpResponse& response,
        std::shared_ptr<IRawDataAccessService> dataService) {
        
        // 获取流式瓦片生成器
        auto tileStreamFuture = dataService->generateTileStreamAsync(
            tileRequest.sourceData,
            tileRequest,
            TileFormat::PNG
        );
        
        // 异步处理结果
        tileStreamFuture.then([this, &response](auto future) {
            auto streamResult = future.get();
            if (streamResult.has_error()) {
                response.write_error(streamResult.error);
                return;
            }
            
            handleStreamingRequest(
                HttpRequest{}, 
                response, 
                streamResult.value()
            );
        });
    }
};
```

## 5. 各模块重构方案（设计文档兼容）

### 5.1 数据访问服务重构 (优先级: **最高** - 流式处理核心)

**现状**: 使用boost::future，缺乏流式处理能力
**重构任务**:

1. **接口全面改造**: `IRawDataAccessService` 
```cpp
class IRawDataAccessService {
public:
    virtual ~IRawDataAccessService() = default;
    
    // 保留传统完整读取（小文件，<100MB）
    virtual Future<AsyncResult<GridData>> readGridVariableAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::optional<IndexRange>& timeRange = std::nullopt,
        const std::optional<BoundingBox>& spatialExtent = std::nullopt,
        const std::optional<IndexRange>& levelRange = std::nullopt
    ) = 0;
    
    // 新增：流式数据读取（大文件，>100MB）- 设计文档核心要求
    virtual Future<StreamingResult<GridData>> readGridVariableStreamAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::optional<IndexRange>& timeRange = std::nullopt,
        const std::optional<BoundingBox>& spatialExtent = std::nullopt,
        const std::optional<IndexRange>& levelRange = std::nullopt,
        size_t chunkSizeBytes = 1024 * 1024 // 默认1MB块
    ) = 0;
    
    // 新增：精确分块读取
    virtual Future<AsyncResult<GridData>> readGridVariableChunkAsync(
        const std::string& filePath,
        const std::string& variableName,
        const ChunkRange& chunkRange,
        const ChunkCoordinates& coordinates
    ) = 0;
    
    // 新增：流式要素读取（大矢量文件）
    virtual Future<StreamingResult<Feature>> readFeaturesStreamAsync(
        const std::string& filePath,
        const std::string& layerName = "",
        const std::optional<BoundingBox>& spatialFilter = std::nullopt,
        const std::optional<AttributeFilter>& attributeFilter = std::nullopt,
        const std::optional<CRSInfo>& targetCRS = std::nullopt,
        size_t featureBatchSize = 1000 // 每批次要素数量
    ) = 0;
    
    // 新增：瓦片流式生成（栅格服务核心功能）
    virtual Future<StreamingResult<std::vector<unsigned char>>> generateTileStreamAsync(
        const GridData& sourceData,
        const TileRequest& request,
        const TileFormat& format
    ) = 0;
    
    // 新增：文件信息获取（用于流式处理决策）
    virtual Future<AsyncResult<FileMetadata>> getFileMetadataAsync(
        const std::string& filePath
    ) = 0;
};
```

2. **实现层改造**: 
   - **NetCDF流式读取器**: 实现基于HDF5分块的真正流式读取
   - **GDAL流式适配器**: 使用GDAL的分块读取API
   - **内存管理**: 实现自适应块大小和内存监控

3. **工厂类更新**: `DataAccessServiceFactory`
```cpp
class DataAccessServiceFactory {
public:
    // 创建支持流式处理的服务实例
    static std::unique_ptr<IRawDataAccessService> createStreamingService(
        const DataAccessConfig& config,
        std::shared_ptr<boost::asio::thread_pool> threadPool,
        size_t maxMemoryUsage = 512 * 1024 * 1024 // 默认512MB内存限制
    );
    
    // 创建针对大数据优化的服务实例
    static std::unique_ptr<IRawDataAccessService> createBigDataService(
        const BigDataConfig& config,
        std::shared_ptr<boost::asio::thread_pool> threadPool
    );
};
```

4. **流式处理实现细节**:
```cpp
// NetCDF流式读取实现示例
class NetCDFStreamingReader : public NetCDFStreamReader {
private:
    // HDF5分块读取配置
    std::array<size_t, 4> optimalChunkDims_; // time, level, lat, lon
    size_t memoryBudget_;
    
public:
    Future<StreamingResult<GridData>> readVariableStreamAsync(
        const std::string& filePath,
        const std::string& variableName,
        size_t chunkSizeBytes) override {
        
        return create_streaming_task<GridData>(*threadPool_, 
            [=](size_t chunkSize) -> std::unique_ptr<DataStream<GridData>> {
                // 1. 打开NetCDF文件并分析变量结构
                auto ncFile = openNetCDFFile(filePath);
                auto variable = ncFile.getVariable(variableName);
                
                // 2. 计算最优分块策略
                auto chunkStrategy = calculateOptimalChunking(
                    variable.getDimensions(), 
                    chunkSizeBytes,
                    memoryBudget_
                );
                
                // 3. 创建流式读取器
                return std::make_unique<NetCDFVariableStream>(
                    std::move(ncFile),
                    variable,
                    chunkStrategy
                );
            },
            chunkSizeBytes
        );
    }
};

// 大文件自动检测和流式处理决策
class SmartDataAccessService : public IRawDataAccessService {
private:
    static constexpr size_t LARGE_FILE_THRESHOLD = 100 * 1024 * 1024; // 100MB
    
public:
    Future<AsyncResult<GridData>> readGridVariableAsync(/*...*/) override {
        // 自动选择读取策略
        return getFileMetadataAsync(filePath)
            .then([=](auto future) -> Future<AsyncResult<GridData>> {
                auto metadata = future.get();
                if (metadata.has_error()) {
                    return make_error_future<GridData>(metadata.error);
                }
                
                if (metadata.data.estimatedSizeBytes > LARGE_FILE_THRESHOLD) {
                    // 自动切换到流式处理，然后聚合结果
                    return readStreamAndAggregate(filePath, variableName, timeRange, spatialExtent, levelRange);
                } else {
                    // 使用传统的一次性读取
                    return readTraditionalAsync(filePath, variableName, timeRange, spatialExtent, levelRange);
                }
            });
    }
};
```

### 5.2 CRS服务重构 (优先级: **高**)

**现状**: 接口混乱，需要完全重构
**重构任务**:

1. **重写接口**: `ICrsService` -> 完全std::future + 错误处理
2. **ThreadPool集成**: 与设计文档的线程模型一致
3. **创建工厂类**: `CrsServiceFactory`

### 5.3 空间操作服务标准化 (优先级: **高**)

**现状**: 接口已使用std::future，但缺乏错误处理
**重构任务**:

1. **添加错误处理**: 所有方法返回`AsyncResult<T>`包装
2. **ThreadPool集成**: 确保与设计文档一致
3. **工厂类已存在**: 验证是否符合标准

### 5.4 其他服务重构 (优先级: **中**)

**元数据、插值、模拟服务**：
1. **统一接口**: std::future + AsyncResult包装
2. **创建工厂类**: 标准工厂模式
3. **错误处理**: 统一std::error_code

### 5.5 工作流引擎重构 (优先级: **高**)

**现状**: 严重混用问题
**重构任务**:

1. **统一异步接口**: 全部使用std::future
2. **Boost.Asio集成**: 与网络层无缝集成
3. **错误处理统一**: 使用std::error_code

## 6. 实施计划（流式处理优先版本）

### 6.0 关键问题解决优先级 ⭐
**设计文档核心要求**: "数据流：优先考虑流式处理或分块处理大数据，避免一次性加载"

### 6.1 阶段零: 流式处理架构基础 (1天) - **最高优先级**
1. **创建流式处理配置**：
   ```bash
   # 创建核心流式处理头文件
   core_service_interfaces/include/core_services/unified_async_config.h
   
   # 包含：DataStream接口、StreamingResult、ChunkProcessor等
   ```

2. **实现内存监控机制**：
   ```cpp
   // 内存使用监控和自适应块大小管理
   class MemoryMonitor {
       static size_t getCurrentMemoryUsage();
       static bool isMemoryPressure();
       static void setMemoryLimit(size_t limitBytes);
   };
   ```

3. **Boost.Asio集成准备**：
   ```cpp
   // ThreadPool和IO Context的协调工作机制
   // Chunked Transfer Encoding基础设施
   ```

### 6.2 阶段一: 数据访问服务流式重构 (3天) - **最高优先级**

#### 6.2.1 NetCDF流式读取实现 (1天)
```bash
# 重点文件修改：
core_services_impl/data_access_service/src/impl/readers/netcdf/
├── streaming_netcdf_reader.cpp  # 新建
├── chunked_variable_reader.cpp  # 新建
└── netcdf_memory_manager.cpp    # 新建

# 验证标准：
- 能够流式读取>1GB的NetCDF文件而不超过512MB内存使用
- 支持时间序列、空间范围、多层级的分块读取
- 错误处理和进度报告正确
```

#### 6.2.2 GDAL流式适配 (1天)
```bash
# 重点文件修改：
core_services_impl/data_access_service/src/impl/readers/gdal/
├── streaming_gdal_reader.cpp    # 新建
├── raster_tile_generator.cpp    # 新建
└── gdal_chunk_optimizer.cpp     # 新建

# 验证标准：
- 大型GeoTIFF/栅格文件的分块读取
- 瓦片生成的流式输出
- 与GDAL块大小的最优协调
```

#### 6.2.3 接口统一和工厂更新 (1天)
```bash
# 接口更新：
core_service_interfaces/include/core_services/data_access/i_raw_data_access_service.h

# 工厂类更新：
core_services_impl/data_access_service/src/impl/factory/data_access_service_factory.cpp

# 验证标准：
- 所有新增流式接口可用
- 自动大小文件检测和策略选择
- 向后兼容性保持
```

### 6.3 阶段二: 网络层流式集成 (1天) - **高优先级**

#### 6.3.1 HTTP Chunked Transfer实现
```bash
# 修改文件：
network_service/src/streaming_controller.cpp     # 新建
output_generation/src/tile_service/streaming_tile_handler.cpp  # 修改

# 验证标准：
- HTTP响应使用正确的Chunked Encoding
- 大文件下载不占用过多服务器内存
- 客户端能够正常接收流式数据
```

#### 6.3.2 瓦片服务流式集成
```bash
# 重点集成：
- DataStream<GridData> -> HTTP Chunked Response
- 瓦片生成的流式输出
- 错误处理和客户端通知

# 验证标准：
- 大型栅格瓦片的流式生成和传输
- 多个并发瓦片请求的资源管理
- 错误恢复和重试机制
```

### 6.4 阶段三: 其他核心服务流式适配 (2天) - **中优先级**

#### 6.4.1 空间操作服务 (1天)
```bash
# 流式空间查询实现
core_services_impl/spatial_ops_service/src/impl/spatial_ops_service_impl.cpp

# 验证标准：
- 大数据集的流式空间查询
- 分块栅格运算
- 内存使用控制在合理范围
```

#### 6.4.2 插值服务 (1天)
```bash
# 流式插值计算实现
core_services_impl/interpolation_service/src/algorithms/streaming_interpolator.cpp

# 验证标准：
- 大规模点数据的分块插值
- 内存使用优化
- 计算精度保持
```

### 6.5 阶段四: 工作流引擎统一 (1天) - **中优先级**
```bash
# 统一异步模式，std::future集成
workflow_engine/src/workflow_executor.cpp

# 与流式处理的协调：
- 工作流中的大数据步骤自动使用流式处理
- 错误处理统一
- 进度报告和监控
```

### 6.6 阶段五: 验证和优化 (1天) - **最高优先级**

#### 6.6.1 流式处理验证标准

**内存使用验证**:
```cpp
// 验证测试用例
class StreamingMemoryTest : public ::testing::Test {
public:
    void testLargeFileProcessing() {
        // 处理2GB NetCDF文件，内存使用不超过256MB
        auto service = DataAccessServiceFactory::createStreamingService(
            config, threadPool, 256 * 1024 * 1024
        );
        
        auto streamResult = service->readGridVariableStreamAsync(
            "large_file_2gb.nc", "temperature"
        );
        
        // 监控内存使用
        size_t maxMemory = 0;
        streamResult.value()->setChunkCallback([&](auto chunk) {
            maxMemory = std::max(maxMemory, MemoryMonitor::getCurrentMemoryUsage());
        });
        
        EXPECT_LT(maxMemory, 256 * 1024 * 1024);
    }
};
```

**网络传输验证**:
```bash
# HTTP流式传输测试
curl -v "http://localhost:8080/api/tiles/large_raster/z/x/y" \
  --header "Accept-Encoding: chunked" \
  --output large_tile.png

# 验证：
- Transfer-Encoding: chunked头存在
- 数据分块正确传输
- 文件完整性保持
```

**性能验证**:
```cpp
// 性能对比测试
void performanceComparison() {
    // 传统方式：一次性加载
    auto start1 = std::chrono::high_resolution_clock::now();
    auto traditionalResult = service->readGridVariableAsync(filePath, varName);
    auto end1 = std::chrono::high_resolution_clock::now();
    
    // 流式方式：分块处理
    auto start2 = std::chrono::high_resolution_clock::now();
    auto streamResult = service->readGridVariableStreamAsync(filePath, varName);
    auto end2 = std::chrono::high_resolution_clock::now();
    
    // 内存使用对比
    // 首字节时间对比 (TTFB)
    // 总传输时间对比
}
```

### 6.7 关键成功指标 (KPI)

#### 6.7.1 内存效率指标
- [ ] **大文件处理**: 2GB文件处理内存使用<256MB
- [ ] **并发处理**: 10个并发大文件请求内存总使用<2GB
- [ ] **内存稳定性**: 长时间运行无内存泄漏

#### 6.7.2 性能指标
- [ ] **首字节时间**: 大文件流式响应TTFB<2秒
- [ ] **吞吐量**: 流式传输速度>50MB/s
- [ ] **并发能力**: 支持>100个并发流式连接

#### 6.7.3 可靠性指标
- [ ] **错误恢复**: 网络中断后能够正确恢复流式传输
- [ ] **资源清理**: 取消的流式操作能够正确释放资源
- [ ] **错误传播**: 流式处理中的错误能够正确传播到客户端

---

**流式处理实施原则**: 
1. **内存第一**: 任何时候内存使用都不能超过配置限制
2. **渐进式**: 允许传统方式和流式方式并存，逐步迁移
3. **性能监控**: 实时监控内存、网络、处理性能
4. **向后兼容**: 现有API保持兼容，新增流式API

## 7. 设计文档兼容性验证

### 7.1 异步模型验证清单
- [ ] **网络层**: 使用Boost.Asio异步I/O
- [ ] **核心服务**: 使用std::future + ThreadPool
- [ ] **集成**: 两层无缝协作
- [ ] **性能**: 少量I/O线程 + 固定工作线程池

### 7.2 错误处理验证清单
- [ ] **统一**: 所有服务使用std::error_code
- [ ] **异步**: 异步回调中正确传递错误
- [ ] **分类**: 自定义error_category
- [ ] **传播**: 错误在调用链中正确传播

### 7.3 架构一致性验证清单
- [ ] **分层**: 严格遵循5层架构
- [ ] **依赖**: 高层依赖低层，无循环依赖
- [ ] **接口**: 清晰的服务接口定义
- [ ] **数据流**: 支持流式处理大数据

## 8. 与设计文档的关键对齐

### 8.1 技术选型对齐
- ✅ **异步技术**: std::async/std::future/ThreadPool
- ✅ **网络层**: Boost.Asio + Boost.Beast
- ✅ **错误处理**: std::error_code + 自定义category
- ✅ **线程模型**: I/O线程 + 工作线程池

### 8.2 架构模式对齐
- ✅ **分层架构**: 严格的5层结构
- ✅ **依赖注入**: 工厂类 + 构造函数注入
- ✅ **模块化**: 清晰的模块边界
- ✅ **异步流水线**: 端到端异步处理

---

**重构方针**: 严格符合设计文档要求，使用std::future + 统一错误处理，确保与Boost.Asio的完美集成。 