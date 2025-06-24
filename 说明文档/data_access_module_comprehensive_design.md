# Data Access模块全面分析与重构设计方案

## 1. 🔍 深度功能分析

### 1.1 核心组件架构

**主服务类：RawDataAccessServiceImpl (约270行)**
- 现有17个异步接口方法，公共API保持不变。
- 内部实现将全面重构，依赖新的统一缓存、异步执行器、读取器架构和流式处理器。
- 支持GridData、FeatureCollection的读取（TimeSeriesData的读取方式需确认是否保留或如何整合）。
- 元数据提取、文件验证、变量查询功能将通过新的读取器接口实现。

**读取器体系：旧 `IDataReaderImpl` 接口**
- 包含约20个纯虚函数，职责宽泛，违反接口隔离原则（ISP）。
- 具体格式实现：`NetCDFCfReader` (约402行), `GdalRasterReader`, `GdalVectorReader`。
- 使用 `std::variant<std::shared_ptr<NetCDFCfReader>, ...>` (`SharedReaderVariant`) 管理读取器类型，扩展性差。

### 1.2 缓存子系统（重复功能严重）

**现有四套独立缓存实现：**
1.  `DataChunkCache` (约597行) - 基于自定义LRU算法的数据块（`GridData`子集）缓存。键为 `DataChunkKey`，值为 `std::shared_ptr<CacheableData>` (包装 `GridData`)。容量和大小通过构造函数参数配置。
2.  `ReaderCache` (约424行) - 基于自定义LRU和"热度图"驱逐逻辑的读取器实例（`IDataReaderImpl` 实现）缓存。键为文件路径 `std::string`，值为 `ReaderCacheEntry` (含 `SharedReaderVariant` 和访问统计)。
3.  `MetadataCache` (约253行) - 基于TTL（Time-To-Live）的元数据字符串缓存。键为 `std::string`，值为序列化元数据 `std::string`。主要用于 `data_access_service` 内部，与 `metadata_service` 的元数据缓存可能存在功能重叠。
4.  `NetCDFCacheManager` (约429行) - 针对NetCDF数据切片的专用缓存，内部为不同数据类型（float, double, int, char的vector）聚合了多个 `CacheManager` 模板实例（一个简单的基于大小的LRU缓存）。键为 `NCDataSliceKey`。

**问题严重性：**
- 总计约2400+行重复的缓存逻辑和管理代码。
- **内存利用率低下：** 估计仅60-70%。各缓存独立管理内存，无法全局优化，导致碎片化和资源浪费。目标应提升至85-90%。
- **缓存命中率不高：** 估计45-60%。缺乏高级缓存策略（如LFU、自适应、预取）和全局视角。统一后利用 `common_utils` 高级缓存有望达到80-85%。
- **锁竞争与性能瓶颈：** 4套独立的锁机制增加了并发控制的复杂性和潜在的竞争点，影响整体吞吐量。
- **缺乏灵活性和可扩展性：** 硬编码的缓存策略和类型限制了对不同数据特征的适应能力。
- **与 `common_utils::cache` 重复：** `common_utils` 提供了基于 `ICacheManager` 接口的、包含多种可配置策略（LRU, LFU, FIFO, TTL, Adaptive, Hierarchical）的先进缓存框架。

### 1.3 时间处理重复（与Common模块约90%重叠）

**Data Access模块中的独立实现：**
- `NetCDFCoordinateSystemParser` (约268行)：包含大量CF时间约定解析逻辑（如单位、日历类型、偏移量计算），以及非时间相关的坐标系元数据解析。
- 可能存在独立的 `TimeProcessor` 类或散落在各处的自定义时间转换函数。
- 可能存在独立的时间单位枚举、时间范围结构。

**与Common模块 `common_utils::time` 重复：**
- `common_utils::time::CalendarTime`: 提供标准化的时间表示。
- `common_utils::time::TimeRange`: 提供标准化的时间范围表示。
- `common_utils::time::CalendarConverter`: 提供不同日历（Gregorian, Julian, Noleap, 360_day等）之间的转换及CF时间字符串解析。
- `common_utils::time::ITimeExtractor`: 定义了从数据源提取时间的标准接口。
- `common_utils::time::time_calendar.h` 提供了对多种日历系统的支持。

## 2. 🔴 重复功能深度分析 与 现有问题剖析

### 2.1 缓存功能重复度：~95%

```cpp
// Data Access模块：众多独立的LRU或特定缓存逻辑片段
// 例如 DataChunkCache::moveToFrontLocked() 采用经典双向链表实现LRU
// if (_cacheMap.find(key) != _cacheMap.end()) {
//     _lruList.splice(_lruList.begin(), _lruList, _cacheMap[key].lruIterator);
//     _cacheMap[key].lruIterator = _lruList.begin();
//     // ...
// }

// VS Common统一实现：common_utils::cache::ICacheManager 及其策略
// 例如 common_utils::cache::LRUCacheStrategy<Key, Value>
// - 内部同样使用 std::unordered_map 和 std::list 实现LRU
// - 更重要的是提供了统一接口和多种其他策略 (LFU, Adaptive等)
// - 支持通过 CacheConfig 进行细粒度配置 (容量、TTL、驱逐策略等)
// - 某些高级策略可能包含智能预取、机器学习优化（如文档所述，需确认common_utils中具体实现程度）
// - 统一管理能实现全局内存协调，避免各缓存争抢。
```
**现有缓存问题剖析：**
- **`DataChunkCache`**: 自定义LRU，无高级策略。`CacheableData` 接口用于获取大小，但缓存对象类型单一。
- **`ReaderCache`**: "热度图"驱逐逻辑 (`evictHeatMapLowest`) 是一种尝试性的LFU变体，但可能不如标准LFU高效或灵活。缓存的 `ReaderCacheEntry` 结构复杂。关键问题是读取器关闭（资源释放）与缓存驱逐的联动。
- **`MetadataCache`**: 简单的TTL策略。对于频繁访问的元数据，可能不是最优。键的构造（如文件路径+查询条件哈希）需要统一规范。
- **`NetCDFCacheManager`**: 为特定数据类型（vector<float>等）实例化多个简单LRU缓存，缺乏整体协调，内存分配固定。

### 2.2 异步框架碎片化：~70%重复

```cpp
// Data Access中17+个异步方法普遍采用以下重复模式 (总计1300+行样板代码)
boost::future<ReturnType> MethodNameAsync(ArgType1 arg1, ArgType2 arg2, ...) {
    // 1. 异步获取Reader (通常涉及ReaderCache和ReaderFactory)
    return getReaderAsync(filePath, targetCRS) 
        .then(boost::launch::deferred, 
              [this, arg1, arg2, ...](boost::future<std::optional<readers::SharedReaderVariant>> futureReaderOpt) -> ReturnType {
            // 2. 从future获取Reader实例，处理潜在的nullopt
            std::optional<readers::SharedReaderVariant> readerVariantOpt = futureReaderOpt.get();
            if (!readerVariantOpt) {
                // 3. 错误处理和异常抛出
                throw oscean::common_utils::ResourceNotFoundError("Failed to get reader for " + filePath);
            }
            readers::SharedReaderVariant& readerVariant = *readerVariantOpt;
            if (std::holds_alternative<std::shared_ptr<std::monostate>>(readerVariant)) {
                 throw oscean::common_utils::AppBaseException("Reader is monostate for file: " + filePath);
            }

            // 4. 使用std::visit分发到具体的Reader类型
            return std::visit(
                [this, arg1, arg2, ...](auto&& concreteReaderPtr) -> ReturnType {
                    // 5. 调用具体Reader的同步方法执行实际操作
                    if (concreteReaderPtr && concreteReaderPtr->isOpen()) {
                        return concreteReaderPtr->doActualWork(arg1, arg2, ...);
                    } else {
                        // 6. 更多的错误处理
                        throw oscean::common_utils::AppBaseException("Reader not open or null for " + filePath);
                    }
                },
                readerVariant);
        });
}
// VS Common统一实现: common_utils::async::TaskManager
// - 统一线程池管理 (通过 GlobalThreadPoolRegistry)
// - 支持任务优先级、重试策略、批量执行、任务链 (then)
// - 封装了 boost::future 和 boost::promise 的使用细节
// - 提供了更简洁的API来提交和管理异步任务
```
**现有异步框架问题剖析：**
- **样板代码冗余：** `getReaderAsync().then(...)` 结构在每个异步方法中重复，包括错误检查、`std::visit` 的使用等。
- **错误处理不一致：** 虽然都抛出异常，但异常类型、错误消息的构造可能存在细微差别。
- **缺乏高级功能：** 没有统一的重试机制、优先级调度、超时控制。
- **维护困难：** 修改异步处理逻辑（例如，增加统一的日志记录或性能追踪）需要在所有17+个地方进行。
- **潜在阻塞点：** 部分异步链中可能隐藏同步操作，例如在 `then` 的回调中执行了耗时的同步 `getReaderSync` 或其他阻塞调用。

### 2.3 架构问题总结

**Critical级别问题：**
- **缓存系统碎片化与低效：** 80MB+（估计值，实际可能更高）独立内存分配，缺乏全局协调和高级策略，导致内存利用率和命中率低下。
- **异步接口实现冗余：** 1300+行重复的异步调用模式代码，难以维护和扩展，缺乏统一的错误处理和高级异步控制。
- **时间处理逻辑重复：** 500+行与 `common_utils::time` 重复的时间解析和转换代码，增加了维护负担和不一致的风险。

**Medium级别问题：**
- **读取器接口 (`IDataReaderImpl`) 过大：** 包含约20个纯虚函数，涵盖了文件操作、元数据提取、多种数据读取等多个方面，违反接口隔离原则，使得具体读取器实现复杂。
- **硬编码的读取器变体类型 (`SharedReaderVariant`)：** 严重限制了对新数据格式读取器的扩展能力，每次增加新格式都需要修改核心代码。
- **缺乏流式处理能力：** 现有读取接口设计为一次性加载数据到内存，无法高效处理超出可用内存的大型文件（例如GB级、TB级遥感影像或气候模型输出）。
- **配置不灵活：** 缓存大小、策略等参数多为硬编码或通过构造函数传递，缺乏统一的外部配置机制。

## 3. 🎯 一次性重构方案

### 3.1 重构策略

**核心原则：**
- **全面采用 `common_utilities`：** 最大限度复用 `common_utils` 中成熟的缓存、异步、时间处理、日志、错误处理等基础组件。
- **直接替换重复功能：** 对于 `data_access` 中与 `common_utils` 重复的功能，采用直接替换策略，避免引入不必要的适配器层以保证性能和可维护性。
- **保持公共API兼容性：** `i_data_access_service.h` 中定义的公共接口方法签名和核心行为保持不变，确保对上层服务的兼容性。
- **内部实现彻底重写：** `RawDataAccessServiceImpl` 及其依赖的内部组件将进行大规模重写和替换。
- **面向接口编程：** 新的读取器架构将严格遵循接口隔离原则，定义清晰、职责单一的接口。
- **提升可扩展性：** 新的读取器注册机制和流式处理框架将显著提升模块处理新格式和大数据场景的能力。
- **目标性能提升：** 综合各项优化，预期在缓存命中率、内存使用效率、并发处理能力及大文件处理方面获得30-50%的性能提升。

### 3.2 详细设计方案

#### 第一阶段：缓存系统统一 (预计 Week 1-2)

**新架构设计：`UnifiedDataAccessCache`**
- **目的：** 替换现有的四个独立缓存，提供统一的、基于 `common_utils::cache` 的缓存服务。
- **核心组件：**
    ```cpp
    // common_utils::cache::ICacheManager (接口)
    // common_utils::cache::LRUCacheStrategy, LFUCacheStrategy, AdaptiveCacheStrategy 等 (策略实现)
    // common_utils::cache::CacheConfig (配置结构)
    
    // In: core_services_impl/data_access_service/include/core_services/data_access/cache/unified_data_access_cache.h
    namespace oscean::core_services::data_access::cache {
    
    /**
     * @brief 统一数据访问缓存值类型。
     * 可以存储不同类型的可缓存对象，并处理其生命周期（特别是资源释放）。
     */
    class CacheableValue {
    public:
        virtual ~CacheableValue() = default;
        virtual size_t getSizeInBytes() const = 0; // 用于缓存大小统计和策略决策
        // 可选：virtual void onEviction() {} // 用于特殊资源释放，如关闭文件句柄
    };
    
    // 示例：包装读取器实例
    class CachedReaderWrapper : public CacheableValue {
    public:
        explicit CachedReaderWrapper(std::shared_ptr<readers::IDataSource> reader) 
            : reader_(std::move(reader)) {}
        
        ~CachedReaderWrapper() override {
            // 确保读取器在被包装器销毁时（可能在缓存驱逐后）被关闭
            if (reader_ && reader_->isOpen()) {
                try {
                    reader_->close();
                } catch (const std::exception& e) {
                    // Log error: Failed to close reader on eviction
                }
            }
        }
        
        std::shared_ptr<readers::IDataSource> getReader() const { return reader_; }
        size_t getSizeInBytes() const override { /* 估算读取器对象大小或返回固定值 */ return 1024; } // 示例大小
    
    private:
        std::shared_ptr<readers::IDataSource> reader_;
    };
    
    // 其他 CacheableValue 实现，如 CachedDataChunk, CachedMetadataString
    
    class UnifiedDataAccessCache {
    public:
        UnifiedDataAccessCache(std::shared_ptr<common_utils::logging::Logger> logger);
    
        // 初始化缓存，可接受更细致的配置
        void initialize(const common_utils::cache::CacheConfig& globalConfig,
                        const std::map<std::string, common_utils::cache::CacheConfig>& specificConfigs = {});
    
        // 获取或创建特定用途的缓存实例 (例如，一个用于数据块，一个用于读取器)
        // KeyType 和 ValueType 应为具体类型，ValueType 通常是 std::shared_ptr<CacheableValue的子类>
        template<typename KeyType, typename ValueType>
        std::shared_ptr<common_utils::cache::ICacheManager<KeyType, ValueType>>
        getOrCreateCacheRegion(const std::string& regionName, 
                               const common_utils::cache::CacheConfig& regionConfig);
    
        // 通用获取/放置接口，可能内部路由到不同区域的缓存
        template<typename KeyType, typename ValueType>
        boost::future<std::optional<ValueType>> getAsync(const std::string& regionName, const KeyType& key);
    
        template<typename KeyType, typename ValueType>
        boost::future<bool> putAsync(const std::string& regionName, const KeyType& key, ValueType&& data, 
                                     std::optional<std::chrono::seconds> ttl = std::nullopt);
        
        // ... 其他必要的管理方法，如 clearRegion, getRegionStats ...
    
    private:
        std::shared_ptr<common_utils::logging::Logger> logger_;
        // 可以用一个map管理不同区域的ICacheManager实例
        std::map<std::string, std::shared_ptr<void>> cacheRegions_; // 使用void*配合类型擦除，或更复杂的模板管理
        std::mutex regionsMutex_;
        
        // 辅助函数创建和配置 common_utils::cache::ICacheManager 实例
        template<typename KeyType, typename ValueType>
        std::shared_ptr<common_utils::cache::ICacheManager<KeyType, ValueType>>
        createCacheInstance(const common_utils::cache::CacheConfig& config);
    };
    } // namespace
    ```
- **增强和修改点：**
    1.  **缓存区域 (Cache Regions)：** 引入"缓存区域"概念。不同的数据类型（如数据块、读取器实例、元数据字符串）可以有独立的缓存区域，每个区域可以配置不同的 `common_utils::cache::CacheConfig`（例如，不同的策略、容量、TTL）。`UnifiedDataAccessCache` 负责管理这些区域。
    2.  **结构化缓存键：**
        *   为数据块设计 `StructuredDataChunkKey` (可包含数据源标识、变量名、时间戳、空间范围哈希等)。
        *   读取器缓存键仍可为文件路径 `std::string`，但考虑规范化。
        *   元数据键可为 `StructuredMetadataKey` (可包含数据源标识、查询参数等)。
    3.  **`CacheableValue` 接口：** 定义一个 `CacheableValue` 接口/基类，所有希望被缓存的对象（或其包装器）都实现此接口。它至少包含 `getSizeInBytes()` 方法。对于需要特殊处理的缓存项（如 `ReaderCache` 中的读取器实例），可以增加 `onEviction()` 方法，在缓存项被驱逐时调用，以执行资源清理（如关闭文件句柄）。`CachedReaderWrapper` 将包装 `std::shared_ptr<readers::IDataSource>` 并实现此逻辑。
    4.  **配置灵活性：** `UnifiedDataAccessCache::initialize` 方法应允许传入全局配置和针对特定区域的配置。配置应来源于外部文件或配置服务，而不是硬编码。
    5.  **利用 `common_utils::cache` 高级特性：** 根据 `common_utils::cache::CacheConfig` 提供的选项，启用如自适应策略、分层缓存（如果 `HierarchicalCacheStrategy` 可用）、压缩（如果支持）等。
- **删除重复代码：**
    - ❌ `DataChunkCache` (及其所有代码和依赖 ~597行)
    - ❌ `ReaderCache` (及其所有代码和依赖 ~424行)
    - ❌ `MetadataCache` (及其所有代码和依赖 ~253行)
    - ❌ `NetCDFCacheManager` (及其所有代码和依赖 ~429行)
    - ❌ `cache_manager_template.h` (如果仅被 `NetCDFCacheManager` 使用)
    - **总计删除：约1703+行重复代码。**

#### 第二阶段：时间处理统一 (预计 Week 2-3)

**新架构设计：直接集成 `common_utils::time`**
- **目的：** 消除 `data_access` 模块中所有自定义的时间解析、转换和表示逻辑。
- **核心组件 (来自 `common_utils::time`)：**
    - `common_utils::time::CalendarTime`
    - `common_utils::time::TimeRange`
    - `common_utils::time::CalendarConverter` (及其内部支持的日历类型和CF解析)
    - `common_utils::time::ITimeExtractor`
    - `common_utils::time::time_calendar.h` (提供不同日历的核心逻辑)
- **适配与集成：**
    ```cpp
    // In: core_services_impl/data_access_service/include/core_services/data_access/time/cf_time_extractor.h
    namespace oscean::core_services::data_access::time {
    
    class CFTimeExtractor : public common_utils::time::ITimeExtractor {
    public:
        // filePath 和 variableName/timeCoordinateName 用于从数据源获取时间单位和日历类型元数据
        CFTimeExtractor(std::weak_ptr<readers::IMetadataProvider> metadataProvider, 
                        const std::string& timeCoordinateName,
                        std::shared_ptr<common_utils::logging::Logger> logger);
    
        // 解析单个时间字符串 (例如 "2023-10-26T12:00:00Z" 或 "100 days since 2000-01-01")
        boost::future<std::optional<common_utils::time::CalendarTime>> extractTimeAsync(
            const std::string& timeValueString) override;
        
        // 解析时间范围字符串 (特定于数据格式的表示)
        boost::future<std::optional<common_utils::time::TimeRange>> extractTimeRangeAsync(
            const std::string& timeRangeString) override;
        
        // 新增：直接从时间坐标值（通常是double类型）和其单位、日历进行转换
        boost::future<std::optional<common_utils::time::CalendarTime>> convertCoordinateToCalendarTimeAsync(
            double timeCoordinateValue);
            
        // 新增：批量转换时间坐标值
        boost::future<std::vector<std::optional<common_utils::time::CalendarTime>>> convertCoordinatesToCalendarTimesAsync(
            const std::vector<double>& timeCoordinateValues);
    
    private:
        // 辅助方法，异步获取时间单位和日历类型
        boost::future<std::pair<std::string, common_utils::time::CalendarType>> getTimeUnitsAndCalendarAsync();
    
        std::weak_ptr<readers::IMetadataProvider> metadataProvider_; // 用于获取时间元数据
        std::string timeCoordinateName_; // 时间坐标变量名
        std::shared_ptr<common_utils::logging::Logger> logger_;
        
        // 缓存获取到的时间单位和日历，避免重复提取
        std::optional<std::pair<std::string, common_utils::time::CalendarType>> cachedTimeAttrs_;
        boost::shared_mutex timeAttrsMutex_; 
    };
    
    } // namespace
    ```
- **增强和修改点：**
    1.  **`CFTimeExtractor` 增强：**
        *   构造时传入 `std::weak_ptr<readers::IMetadataProvider>` 和时间坐标变量名，使其能够按需从数据源（通过新的读取器元数据接口）异步获取关键的时间元数据（如 `units: "days since YYYY-MM-DD"` 和 `calendar: "noleap"`）。
        *   实现 `extractTimeAsync` 时，利用获取到的单位和日历调用 `common_utils::time::CalendarConverter`。
        *   增加 `convertCoordinateToCalendarTimeAsync` 和 `convertCoordinatesToCalendarTimesAsync` 方法，直接处理从NetCDF等文件读取到的数值型时间坐标。
        *   内部缓存从元数据中获取的时间单位和日历类型，避免重复IO。
    2.  **迁移非时间逻辑：** `NetCDFCoordinateSystemParser` 中除时间解析外的其他元数据解析逻辑（如解析投影字符串、地理边界、变量属性等）需要完整地迁移到新的 `readers::netcdf::NetCdfUnifiedReader` 的元数据提取方法中。
    3.  **全面替换：** `data_access_service` 内部所有使用自定义时间类型或逻辑的地方，都替换为 `common_utils::time` 的类型和 `CFTimeExtractor` (或直接使用 `CalendarConverter`，如果上下文信息足够)。
- **删除重复代码：**
    - ❌ `NetCDFCoordinateSystemParser` (约268行，其时间解析逻辑被取代，其他元数据逻辑迁移)
    - ❌ 任何独立的 `TimeProcessor` 类或自定义时间工具函数 (估计约200+行)
    - ❌ 任何自定义的时间单位枚举、时间结构体
    - **总计删除：约500+行重复代码。**

#### 第三阶段：异步框架统一 (预计 Week 3-4)

**新架构设计：`UnifiedAsyncExecutor`**
- **目的：** 消除所有异步方法中重复的 `boost::future` 处理、错误检查和 `std::visit` 样板代码。提供统一的异步任务执行、错误处理和高级控制（如重试、优先级）。
- **核心组件 (来自 `common_utils::async`)：**
    - `common_utils::async::TaskManager` (管理线程池、任务队列、执行逻辑)
    - `common_utils::async::GlobalThreadPoolRegistry` (获取全局线程池)
    - `common_utils::async::RetryPolicy`
    - `common_utils::async::AsyncPriority`
- **适配与集成：**
    ```cpp
    // In: core_services_impl/data_access_service/include/core_services/data_access/async/unified_async_executor.h
    namespace oscean::core_services::data_access::async {
    
    class UnifiedAsyncExecutor {
    public:
        UnifiedAsyncExecutor(std::shared_ptr<common_utils::async::TaskManager> taskManager,
                             std::shared_ptr<common_utils::logging::Logger> logger);
    
        // 执行单个异步任务
        template<typename ReturnType, typename TaskFunc>
        boost::future<ReturnType> executeAsync(
            TaskFunc&& task, 
            common_utils::async::AsyncPriority priority = common_utils::async::AsyncPriority::NORMAL,
            std::optional<common_utils::async::RetryPolicy> retryPolicy = std::nullopt,
            const std::string& taskDescription = "DataAccessAsyncTask") {
            
            // TaskFunc 应该返回 ReturnType 或 boost::future<ReturnType>
            // TaskManager 应该能处理这两种情况
            return taskManager_->execute<ReturnType>( // 或 executeWithRetry, 取决于TaskManager API
                std::forward<TaskFunc>(task), 
                priority, 
                retryPolicy,
                taskDescription
            );
        }
        
        // 专门用于执行数据读取器操作的模板方法，封装了获取读取器和std::visit的逻辑
        template<typename ReturnType, typename ReaderOperationFunc>
        boost::future<ReturnType> executeReaderOperationAsync(
            const std::string& filePath,
            const std::optional<oscean::core_services::CRSInfo>& targetCrs, // 用于获取Reader
            ReaderOperationFunc&& operation, // Func taking std::shared_ptr<readers::IDataReaderImplConcept>
            common_utils::async::AsyncPriority priority = common_utils::async::AsyncPriority::NORMAL,
            std::optional<common_utils::async::RetryPolicy> retryPolicy = std::nullopt,
            const std::string& taskDescription = "DataReaderOperation") {
            
            // 1. 异步获取统一读取器 (IDataSource, IMetadataProvider etc. 的组合)
            //    这需要 RawDataAccessServiceImpl 提供一个 getUnifiedReaderAsync 方法
            auto futureReader = ownerServiceImpl_->getUnifiedReaderAsync(filePath, targetCrs); 
                                                                      // ^ ownerServiceImpl_ 需要注入
    
            return futureReader.then(priority_as_launch_policy(priority), // launch policy mapping
                [this, operation = std::forward<ReaderOperationFunc>(operation), filePath, taskDescription, retryPolicy]
                (boost::future<std::shared_ptr<readers::UnifiedDataReader>> fReader) -> boost::future<ReturnType> {
                    try {
                        std::shared_ptr<readers::UnifiedDataReader> reader = fReader.get(); // May throw
                        if (!reader || !reader->isOpen()) {
                            throw common_utils::ResourceNotFoundError("Failed to get open reader for: " + filePath);
                        }
                        
                        // 实际的操作 (operation) 现在接收一个 UnifiedDataReader 实例
                        // TaskFunc 本身可以返回 future<ReturnType> 或 ReturnType
                        // 如果 operation 返回 ReturnType，需要包装成 future
                        // 如果 TaskManager::execute 能处理 TaskFunc 返回 future 的情况，这里可以简化
                        return taskManager_->execute<ReturnType>(
                            [reader, operation]() { return operation(reader); }, // operation现在直接用reader
                            priority, // Pass original priority or NORMAL for the inner task
                            std::nullopt, // Retry is handled coisasde fora
                            taskDescription + " [inner]"
                        );
    
                    } catch (const std::exception& e) {
                        logger_->error("Error in executeReaderOperationAsync for {}: {}", taskDescription, e.what());
                        // 重新抛出或包装成特定异常
                        return boost::make_exceptional_future<ReturnType>(e);
                    }
                }).unwrap(); // unwrap if the then lambda returns a future
        }
    
    private:
        std::shared_ptr<common_utils::async::TaskManager> taskManager_;
        std::shared_ptr<common_utils::logging::Logger> logger_;
        class RawDataAccessServiceImpl* ownerServiceImpl_ = nullptr; // 用于调用 getUnifiedReaderAsync
        
        // Helper for launch policy (example)
        boost::launch priority_as_launch_policy(common_utils::async::AsyncPriority prio) {
            if (prio == common_utils::async::AsyncPriority::CRITICAL_REALTIME) return boost::launch::async;
            return boost::launch::deferred; // default
        }
    public:
        void setOwner(RawDataAccessServiceImpl* owner) { ownerServiceImpl_ = owner; }
    };
    } // namespace
    
    // RawDataAccessServiceImpl 中的异步方法重写示例:
    // boost::future<std::optional<FileMetadata>> RawDataAccessServiceImpl::extractFileMetadataAsync(...) {
    //     return asyncExecutor_.executeReaderOperationAsync<std::optional<FileMetadata>>(
    //         filePath, targetCrs,
    //         [this](std::shared_ptr<readers::UnifiedDataReader> reader) -> boost::future<std::optional<FileMetadata>> {
    //             // UnifiedDataReader 同时是 IMetadataProvider
    //             return reader->extractFileMetadataAsync(); // 假设 UnifiedDataReader 有此便捷方法
    //                                                      // 或直接调用 IMetadataProvider 的方法组合
    //         },
    //         common_utils::async::AsyncPriority::HIGH
    //     );
    // }
    ```
- **增强和修改点：**
    1.  **`UnifiedAsyncExecutor` 职责：**
        *   主要职责是作为 `common_utils::async::TaskManager` 的门面，简化任务提交。
        *   新增 `executeReaderOperationAsync` 模板方法，此方法将封装：
            *   异步获取新的 `UnifiedDataReader` 实例（通过调用 `RawDataAccessServiceImpl` 的一个新方法，如 `getUnifiedReaderAsync`）。
            *   处理读取器获取失败的情况。
            *   将获取到的 `UnifiedDataReader` 实例传递给调用者提供的具体操作函数 (`ReaderOperationFunc`)。
            *   这样，17个异步方法的样板代码（获取读取器、检查、传递给操作）可以集中到 `executeReaderOperationAsync` 中。
    2.  **`RawDataAccessServiceImpl::getUnifiedReaderAsync`：** `RawDataAccessServiceImpl` 需要提供一个新的私有异步方法，如 `getUnifiedReaderAsync(filePath, targetCrs)`，它负责通过新的 `ReaderRegistry` 和 `UnifiedDataAccessCache`（用于缓存读取器实例）来异步获取一个 `std::shared_ptr<readers::UnifiedDataReader>`。
    3.  **任务取消与超时：** `TaskManager` 应支持任务取消和超时。`UnifiedAsyncExecutor` 提交任务时应能传递这些参数。
    4.  **错误处理统一：** `UnifiedAsyncExecutor` (特别是 `executeReaderOperationAsync`) 将提供集中的错误捕获和日志记录点，可以将底层异常统一包装成 `data_access` 相关的特定异常类型。
    5.  **避免阻塞：** 确保所有IO操作（包括获取读取器实例）在异步链中都是非阻塞的。
    6.  **日志集成：** `UnifiedAsyncExecutor` 中集成日志记录，用于追踪任务执行、错误等。
- **消除重复：**
    - 17个异步方法中重复的读取器获取、`std::visit` 调用、错误检查等逻辑将被封装。
    - **消除约1300+行重复代码。**
    - 提供统一的错误处理、重试策略（通过 `TaskManager`）、优先级管理。

#### 第四阶段：读取器架构重构 (预计 Week 4-5)

**新接口和类设计（遵循ISP）：**
- **目的：** 拆分庞大的 `IDataReaderImpl`，用 `ReaderRegistry` 替换 `SharedReaderVariant`，实现可插拔的读取器架构。
- **核心接口定义 (在 `core_services_impl/data_access_service/include/core_services/data_access/readers/`)**
    ```cpp
    // --- i_data_source.h ---
    class IDataSource { /* ...如文档定义: open, close, isOpen, getFilePath... */ };
    
    // --- i_metadata_provider.h ---
    class IMetadataProvider { 
    public:
        virtual ~IMetadataProvider() = default;
        // 方法签名更新为直接返回具体类型或其future，减少可选参数的层层传递
        virtual boost::future<std::vector<std::string>> listDataVariableNamesAsync() const = 0;
        virtual boost::future<std::vector<std::string>> getVariableNamesAsync() const = 0; // 所有变量
        virtual boost::future<std::optional<CRSInfo>> getNativeCrsAsync() const = 0;
        virtual boost::future<BoundingBox> getNativeBoundingBoxAsync() const = 0;
        virtual boost::future<std::optional<TimeRange>> getNativeTimeRangeAsync(const std::string& timeVariableName = "") = 0; // 可选时间变量名
        virtual boost::future<std::vector<double>> getVerticalLevelsAsync(const std::string& verticalCoordName = "") = 0; // 可选垂直坐标名
        virtual boost::future<std::vector<MetadataEntry>> getGlobalAttributesAsync() const = 0;
        virtual boost::future<std::optional<std::vector<DimensionDefinition>>> getVariableDimensionsAsync(
            const std::string& variableName) const = 0;
        virtual boost::future<std::optional<std::vector<MetadataEntry>>> getVariableMetadataAsync(
            const std::string& variableName) const = 0;
        virtual boost::future<std::vector<DatasetIssue>> validateDatasetAsync(bool comprehensive = false) const = 0;
        // 新增：便捷方法获取文件元数据摘要
        virtual boost::future<std::optional<FileMetadata>> extractFileMetadataSummaryAsync() = 0; 
    };
    
    // --- i_data_provider.h ---
    // 定义清晰的请求结构体
    struct GridReadRequest {
        std::string variableName;
        std::vector<IndexRange> sliceRanges; // 可选，默认为整个范围
        std::optional<std::vector<double>> targetResolution;
        std::optional<CRSInfo> targetCRS;
        ResampleAlgorithm resampleAlgorithm = ResampleAlgorithm::NEAREST;
        std::optional<BoundingBox> outputBounds;
        // ...可扩展: std::map<std::string, std::any> processingParams;
    };
    
    struct FeatureReadRequest { /* ...类似定义... */ };
    
    struct RawVariableReadRequest { // 用于替代旧的 readVariableData
        std::string variableName;
        std::vector<size_t> startIndices;
        std::vector<size_t> counts;
        // ...可扩展
    };
    
    class IDataProvider {
    public:
        virtual ~IDataProvider() = default;
        virtual boost::future<std::shared_ptr<GridData>> readGridDataAsync(
            const GridReadRequest& request) = 0;
        virtual boost::future<FeatureCollection> readFeatureCollectionAsync( // 注意原文档用Features，此处用FeatureCollection
            const FeatureReadRequest& request) = 0;
    };
    
    // --- i_streaming_data_provider.h ---
    struct DataChunk { // 定义在 common_types.h 或此处
        VariableDataVariant data; // 或更具体的类型如 std::vector<char> + TypeInfo
        std::vector<DimensionDefinition> chunkDimensions; // 描述此块的维度
        std::vector<IndexRange> chunkCoverageInVariable; // 此块在整个变量中的索引范围
        bool isLastChunk = false;
        size_t chunkSizeBytes = 0;
        // ...其他元数据，如时间戳、空间范围（如果适用）
    };
    
    class BackpressureControl { /* ...接口待定义，如 requestMore(int n), pause()... */ };
    
    struct StreamingOptions {
        size_t desiredChunkSizeHintBytes = 1024 * 1024; // 1MB
        // ...其他选项，如预取数量
    };
    
    class IStreamingDataProvider { /* ...如文档定义，回调中包含DataChunk和BackpressureControl... */ 
    public:
        virtual ~IStreamingDataProvider() = default;
        // 拉模式 (pull-based)
        virtual boost::future<std::optional<DataChunk>> getNextDataChunkAsync(
            const std::string& variableName, 
            const StreamingOptions& options = {}) = 0; 
            // 返回 nullopt 表示流结束或错误
            // 需要一种方式来初始化/启动流，或在第一次调用时隐式启动

        // 推模式 (push-based) with backpressure
        virtual boost::future<void> streamVariableDataAsync(
            const std::string& variableName,
            std::function<boost::future<bool>(DataChunk)> chunkProcessor, // 返回 future<bool> 以支持异步处理块和背压
            const StreamingOptions& options = {}) = 0; 
            // chunkProcessor 返回 false 停止流
    };
    ```
- **核心实现类 (在 `core_services_impl/data_access_service/src/impl/readers/`)**
    ```cpp
    // --- unified_data_reader.h ---
    // 统一读取器基类 (抽象类)
    class UnifiedDataReader : public IDataSource, 
                              public IMetadataProvider,
                              public IDataProvider,
                              public IStreamingDataProvider { // 可选实现流式接口
    protected:
        std::string filePath_;
        bool isOpen_ = false;
        std::shared_ptr<common_utils::logging::Logger> logger_;
        // 注入CRS服务等依赖
        std::shared_ptr<oscean::core_services::ICrsService> crsService_; 
    public:
        UnifiedDataReader(const std::string& filePath, 
                          std::shared_ptr<common_utils::logging::Logger> logger,
                          std::shared_ptr<oscean::core_services::ICrsService> crsService)
            : filePath_(filePath), logger_(logger), crsService_(crsService) {}
        
        // IDataSource 实现 (部分可提供默认实现)
        bool open(const std::string& filePath) override { filePath_ = filePath; /*子类实现打开逻辑*/ return false; }
        void close() override { isOpen_ = false; /*子类实现关闭逻辑*/ }
        bool isOpen() const override { return isOpen_; }
        std::string getFilePath() const override { return filePath_; }
        
        // 其他接口由子类具体实现
    };
    
    // --- netcdf/netcdf_unified_reader.h / .cpp ---
    class NetCdfUnifiedReader : public UnifiedDataReader { /* ...实现所有接口方法... */ };
    // --- gdal/gdal_raster_unified_reader.h / .cpp ---
    class GdalRasterUnifiedReader : public UnifiedDataReader { /* ...实现所有接口方法... */ };
    // --- gdal/gdal_vector_unified_reader.h / .cpp ---
    class GdalVectorUnifiedReader : public UnifiedDataReader { /* ...实现所有接口方法... */ };
    
    // --- reader_registry.h / .cpp ---
    class ReaderRegistry {
    public:
        using ReaderFactoryFn = std::function<std::unique_ptr<UnifiedDataReader>(
            const std::string& filePath, 
            std::shared_ptr<common_utils::logging::Logger>,
            std::shared_ptr<oscean::core_services::ICrsService>
            /* 其他通用依赖 */
        )>;
        
        ReaderRegistry(std::shared_ptr<FormatDetector> detector, 
                       std::shared_ptr<common_utils::logging::Logger> logger,
                       std::shared_ptr<oscean::core_services::ICrsService> crsService);
        
        void registerReader(const std::string& formatName, uint32_t priority, ReaderFactoryFn factory);
        
        // createReader 会使用 FormatDetector 判断格式，然后调用对应 factory
        boost::future<std::unique_ptr<UnifiedDataReader>> createReaderAsync(const std::string& filePath);
        
    private:
        struct RegisteredFactory {
            ReaderFactoryFn factory;
            uint32_t priority; // 用于处理多种factory都能处理同一格式的情况
        };
        std::map<std::string, std::vector<RegisteredFactory>> factoriesByFormatName_;
        std::vector<std::pair<std::string, RegisteredFactory>> genericFactories_; // 不基于特定名称的探测
        std::shared_ptr<FormatDetector> detector_;
        std::shared_ptr<common_utils::logging::Logger> logger_;
        std::shared_ptr<oscean::core_services::ICrsService> crsService_;
        // ...其他通用依赖，传递给工厂函数...
    };
    
    // --- format_detector.h / .cpp ---
    class FormatDetector {
    public:
        virtual ~FormatDetector() = default;
        // 返回最可能的格式名 (如 "NetCDF", "GeoTIFF", "Shapefile") 和置信度
        virtual boost::future<std::pair<std::string, double>> detectFormatAsync(const std::string& filePath) = 0;
    };
    // 具体实现如: ExtensionBasedFormatDetector, MagicNumberFormatDetector, CompositeFormatDetector
    ```
- **增强和修改点：**
    1.  **接口细化与异步化：** `IMetadataProvider` 和 `IDataProvider` 的所有方法都应设计为异步返回 `boost::future`，以完全支持非阻塞IO。
    2.  **请求结构体：** `GridReadRequest`, `FeatureReadRequest`, `RawVariableReadRequest` 将封装所有读取参数，使API更清晰、易扩展。
    3.  **`UnifiedDataReader`：** 作为新的读取器基类，继承所有四个核心接口。具体格式的读取器（如 `NetCdfUnifiedReader`）将继承它。
    4.  **`ReaderRegistry`：**
        *   支持按格式名注册工厂函数。工厂函数负责创建特定类型的 `UnifiedDataReader` 实例并注入其依赖（如logger, CRS服务）。
        *   `createReaderAsync` 方法将是异步的，因为它可能涉及 `FormatDetector` 的IO操作。
        *   支持工厂优先级，以便在多种探测器都能识别同一文件时选择最优的。
    5.  **`FormatDetector`：** 定义清晰的接口，允许有多种实现策略（基于扩展名、文件头魔数、内容探测等）。其 `detectFormatAsync` 方法也应是异步的。
    6.  **移除 `SharedReaderVariant`：** 代码中所有使用 `SharedReaderVariant` 和 `std::visit` 的地方都将被新的注册表和接口调用取代。
- **主要变化：**
    - ❌ `IDataReaderImpl.h` (被新接口集取代)
    - ❌ `netcdf_cf_reader.h/cpp`, `gdal_raster_reader.h/cpp`, `gdal_vector_reader.h/cpp` (被新的UnifiedReader实现取代)
    - ❌ `data_reader_common.h` 中的 `SharedReaderVariant` 定义。
    - ❌ `factory/reader_factory.h/cpp` (被 `ReaderRegistry` 和 `FormatDetector` 取代)。
    - ✨ 新增上述所有接口和类。

#### 第五阶段：流式处理功能 (预计 Week 5-6)

**新架构设计：`StreamingDataProcessor` 与 `IStreamingDataProvider` 实现**
- **目的：** 为 `data_access_service` 添加对大规模数据集的高效、内存可控的流式读取能力。
- **核心组件：**
    - `IStreamingDataProvider` 接口 (已在第四阶段定义)
    - `DataChunk` 结构体 (已在第四阶段定义)
    - `StreamingOptions` 结构体 (已在第四阶段定义)
    - `BackpressureControl` 接口 (已在第四阶段定义，具体实现待定)
    ```cpp
    // In: core_services_impl/data_access_service/include/core_services/data_access/streaming/streaming_data_processor.h
    namespace oscean::core_services::data_access::streaming {
    
    // (AdaptiveChunkingConfig 和 DataGeometryType 枚举定义如文档所述)
    // AdaptiveChunkingConfig 将作为 StreamingOptions 的一部分或独立配置
    
    class StreamingDataProcessor {
    public:
        StreamingDataProcessor(
            std::shared_ptr<UnifiedAsyncExecutor> asyncExecutor, // 复用第三阶段的异步执行器
            // ReaderRegistry 用于获取 UnifiedDataReader, 它实现了 IStreamingDataProvider
            std::shared_ptr<readers::ReaderRegistry> readerRegistry, 
            std::shared_ptr<common_utils::MemoryMonitor> memoryMonitor, // 来自 common_utils 或新建
            std::shared_ptr<common_utils::PerformanceProfiler> profiler,  // 来自 common_utils 或新建
            std::shared_ptr<common_utils::logging::Logger> logger);
    
        // 主流式读取接口 (推模式示例)
        boost::future<void> processStreamAsync(
            const std::string& filePath,
            const std::string& variableName,
            // 回调函数，处理每个数据块，并通过返回的future<bool>实现异步背压
            std::function<boost::future<bool>(DataChunk)> chunkProcessorCallback,
            StreamingOptions options = {}); 
            // options 中可包含 AdaptiveChunkingConfig
    
        // 批量文件流式处理 (概念性)
        // boost::future<BatchProcessingSummary> processBatchOfStreamsAsync(
        //    const std::vector<StreamProcessRequest>& requests,
        //    const BatchStreamingConfig& config);
    
    private:
        // 辅助方法，用于单个文件流的处理
        boost::future<void> performSingleStreamProcessing(
            std::shared_ptr<readers::IStreamingDataProvider> streamingReader,
            const std::string& variableName,
            std::function<boost::future<bool>(DataChunk)> chunkProcessorCallback,
            StreamingOptions options);
            
        // 根据DataGeometryType和options调整StreamingOptions，特别是块大小
        StreamingOptions determineAdaptiveStreamingOptions(
            // readers::DataGeometryType geometryType, // 需要从reader获取
            const StreamingOptions& initialOptions); 
    
        std::shared_ptr<UnifiedAsyncExecutor> asyncExecutor_;
        std::shared_ptr<readers::ReaderRegistry> readerRegistry_;
        std::shared_ptr<common_utils::MemoryMonitor> memoryMonitor_;
        std::shared_ptr<common_utils::PerformanceProfiler> profiler_;
        std::shared_ptr<common_utils::logging::Logger> logger_;
    };
    
    } // namespace
    
    // 具体读取器 (如 NetCdfUnifiedReader) 中 IStreamingDataProvider 的实现:
    // boost::future<std::optional<DataChunk>> NetCdfUnifiedReader::getNextDataChunkAsync(...) {
    //     // 实现从NetCDF文件按需读取下一个数据块的逻辑
    //     // 考虑使用NetCDF库的分块读取API（如nc_get_vara_xxx）
    //     // 更新内部状态以记住当前流的位置
    //     // 填充DataChunk结构体
    // }
    // boost::future<void> NetCdfUnifiedReader::streamVariableDataAsync(...) {
    //    // 循环调用 getNextDataChunkAsync 或直接实现推模式逻辑
    //    // 调用 chunkProcessor(chunk)，并根据其返回的 future<bool> 控制流程 (处理背压)
    // }
    ```
- **增强和修改点：**
    1.  **`IStreamingDataProvider` 实现：**
        *   每个支持流式的 `UnifiedDataReader`子类 (如 `NetCdfUnifiedReader`) 都需要具体实现 `IStreamingDataProvider` 的接口方法。这通常涉及到底层文件格式库（如NetCDF, GDAL）的分块读取API。
        *   需要管理流的状态（例如，当前读取到文件的哪个位置，下一个块是什么）。
    2.  **`StreamingDataProcessor`：**
        *   **获取流式读取器：** 通过 `ReaderRegistry` 获取 `UnifiedDataReader` 实例，并确保它实现了 `IStreamingDataProvider` 接口（可以通过 `dynamic_cast` 或在注册时标记）。
        *   **自适应分块逻辑 (`determineAdaptiveStreamingOptions`)：**
            *   需要明确 `DataGeometryType` 如何从读取器或元数据中获取。
            *   集成 `MemoryMonitor` 和 `PerformanceProfiler` (如果这些组件在 `common_utils` 中不可用，则需要设计和实现其接口和基本功能)。根据监控数据动态调整 `StreamingOptions` 中的 `desiredChunkSizeHintBytes` 等参数。
        *   **背压实现：** `processStreamAsync` 中的 `chunkProcessorCallback` 返回 `boost::future<bool>`。`StreamingDataProcessor` 在获取到数据块后调用此回调，并等待其 `future` 完成。如果回调处理数据块耗时较长，则自然形成了背压，因为在 `future` 完成前不会请求（或处理）下一个数据块。如果回调返回的 `future` 解析为 `false`，则停止流。
        *   **错误处理：** 流式处理中任何步骤（获取读取器、读取块、处理块）发生的错误都需要被捕获、记录，并以适当的方式（如中断流并向上抛出异常，或调用错误处理回调）通知调用者。
    3.  **`DataChunk` 结构完善：** 确保 `DataChunk` 包含足够的信息供消费者使用，例如每个维度的起始索引和计数，以及全局变量中的总维度大小，以便消费者了解块在整体数据中的位置。
    4.  **`BackpressureControl` (如果采用更主动的控制方式)：** 如果回调函数需要更主动地控制流（例如，处理完一批块后才请求更多），则 `BackpressureControl` 接口需要定义如 `requestChunks(n)` 的方法，并通过 `StreamingDataProcessor` 传递给回调。
    5.  **与异步执行器集成：** 所有耗时操作（文件IO、数据处理回调）都应通过 `UnifiedAsyncExecutor` 提交到线程池执行。
- **主要变化：**
    - ✨ `IStreamingDataProvider` 接口在各个 `UnifiedDataReader` 子类中得到具体实现。
    - ✨ 新增 `StreamingDataProcessor` 类及其辅助逻辑。
    - ✨ 可能需要在 `common_utils` 中新增或完善 `MemoryMonitor` 和 `PerformanceProfiler` 组件，或在 `data_access` 中实现简化版本。

## 4. 📊 重构收益分析 (保持不变或微调)

## 5. 📋 实施计划 (保持不变或微调)

## 6. 🎯 总结 (保持不变或微调)

这份更新后的文档应该更详细地阐述了每个阶段的重构目标、设计思路、关键组件的接口和实现要点，以及与 `common_utils` 的集成方式。它也更明确地指出了需要修改或增强的具体功能点。 


## 3. 重构方案 (Refined Refactoring Plan)

基于对 `data_access` 模块及其所有主要组件（包括 `cache`、`readers` 下的 `gdal` 和 `netcdf` 子模块、`factory` 以及 `crs_service`）的深入代码级分析，现对重构方案进行如下细化。此次分析覆盖了您提供的所有相关头文件和源文件。

### 3.1 阶段一：缓存系统统一 (Cache System Unification)

**目标：** 将现有的多个独立缓存（`DataChunkCache`, `ReaderCache`, `MetadataCache`, `NetCDFCacheManager`）整合为基于 `common_utils::cache::ICacheManager` 的 `UnifiedDataAccessCache`。

**具体实施：**

1.  **引入 `UnifiedDataAccessCache`**:
    *   创建一个新类 `UnifiedDataAccessCache`，其内部将使用 `common_utils::cache::ICacheManager` 的一个或多个实例。
    *   考虑为不同类型的数据（数据块、读取器实例、元数据）配置不同的"缓存区域 (Cache Regions)"。每个区域可以有其特定的 `common_utils::cache::CacheConfig`（例如，不同的`CacheStrategyType`如 `LRU`, `LFU`, `TTL`, `Adaptive`，以及不同的容量、压缩、持久化设置）。

2.  **定义 `CacheableValue` 接口**:
    *   在 `data_access` 模块内部或 `common_utils` (如果更通用) 中定义一个接口，例如 `CacheableValue`，包含：
        *   `virtual size_t getSizeInBytes() const = 0;` 用于缓存大小管理。
        *   `virtual void onEviction() {}` 回调函数，用于在条目被逐出时执行清理操作（例如，对于缓存的读取器，需要关闭文件句柄）。

3.  **改造现有缓存逻辑**:
    *   **`DataChunkCache`** (涉及 `core_services_impl/data_access_service/include/core_services/data_access/cache/data_chunk_cache.h`, `core_services_impl/data_access_service/src/impl/cache/data_chunk_cache.h`, `core_services_impl/data_access_service/src/impl/cache/data_chunk_cache.cpp`):
        *   其缓存的 `GridData` (通过 `CacheableData` 或 `GridDataAdapter` 包装) 将实现 `CacheableValue`。
        *   `DataChunkKey` 将作为缓存键。
        *   配置一个专门的缓存区域，可能使用 LRU 或 Adaptive 策略。
    *   **`ReaderCache`** (涉及 `core_services_impl/data_access_service/src/impl/cache/reader_cache.h`, `core_services_impl/data_access_service/src/impl/cache/reader_cache.cpp`):
        *   缓存的读取器实例 (`SharedReaderVariant` 将被 `CachedReaderWrapper` 替代，该包装器包含一个 `std::shared_ptr<UnifiedDataReader>` 并实现 `CacheableValue`)。
        *   `CachedReaderWrapper::onEviction()` 将负责调用读取器的 `close()` 方法。
        *   缓存键可以是文件路径字符串或更结构化的键。
        *   配置一个专门的缓存区域，可以考虑 LFU 或 Adaptive 策略，因为某些读取器可能比其他读取器更常用。其现有的 "heat map" 逻辑可以被 `common_utils` 的高级缓存策略（如 LFU 或 Adaptive）所取代或在其基础上构建。
    *   **`MetadataCache`** (涉及 `core_services_impl/data_access_service/src/impl/cache/metadata_cache.h`, `core_services_impl/data_access_service/src/impl/cache/metadata_cache.cpp`):
        *   缓存的元数据（通常是序列化字符串或结构化对象）将实现 `CacheableValue`。
        *   缓存键可以是文件路径或特定的元数据请求键。
        *   配置一个 TTL 策略的缓存区域。
    *   **`NetCDFCacheManager`** (涉及 `core_services_impl/data_access_service/src/impl/cache/netcdf_cache_manager.h`, `core_services_impl/data_access_service/src/impl/cache/netcdf_cache_manager.cpp`) 及其使用的 `cache_manager_template.h`:
        *   这个聚合缓存将被 `UnifiedDataAccessCache` 中的不同区域或配置所取代。
        *   `NCDataSliceKey` 将作为数据块缓存的键。
        *   NetCDF 数据切片将实现 `CacheableValue`。

4.  **代码清理**:
    *   删除 `core_services_impl/data_access_service/include/core_services/data_access/cache/data_chunk_cache.h`。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/data_chunk_cache.h` (旧的实现头文件)。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/data_chunk_cache.cpp`。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/reader_cache.h`。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/reader_cache.cpp`。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/metadata_cache.h`。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/metadata_cache.cpp`。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/netcdf_cache_manager.h`。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/netcdf_cache_manager.cpp`。
    *   删除 `core_services_impl/data_access_service/src/impl/cache/cache_manager_template.h`。
    *   预计删除约 2000-2400 行代码。

### 3.2 阶段二：时间处理统一 (Time Processing Unification)

**目标：** 移除 `data_access` 模块内（特别是 NetCDF 读取器中）的自定义时间处理逻辑，全面采用 `common_utils::time` 模块。

**具体实施：**

1.  **直接使用 `common_utils::time` 类型**:
    *   代码中所有表示时间点、时间范围、日历等的自定义结构或基本类型，都替换为 `common_utils::time` 提供的相应类型（如 `CalendarTime`, `TimeRange`）。

2.  **实现 `CFTimeExtractor`**:
    *   创建一个 `CFTimeExtractor` 类，实现 `common_utils::time::ITimeExtractor` 接口。
    *   此类将负责从实现了新 `IMetadataProvider` 接口的读取器中异步获取时间单位、日历等元数据。
    *   处理 CF 规范下的各种时间坐标（如 "days since YYYY-MM-DD")。
    *   `core_services_impl/data_access_service/src/impl/readers/netcdf/parsing/netcdf_time_processor.h` 和 `.cpp` 的逻辑将被这个新的提取器取代。
    *   `core_services_impl/data_access_service/src/impl/readers/netcdf/parsing/netcdf_coordinate_system_parser.h` 和 `.cpp` 中的部分时间解析逻辑（例如 `parseTimeUnits`, `parseTimeCoordinateVariable`）也将移至此处或被 `common_utils::time` 的功能替代。

3.  **代码清理**:
    *   删除 `core_services_impl/data_access_service/src/impl/readers/netcdf/parsing/netcdf_time_processor.h`。
    *   删除 `core_services_impl/data_access_service/src/impl/readers/netcdf/parsing/netcdf_time_processor.cpp`。
    *   `core_services_impl/data_access_service/src/impl/readers/netcdf/parsing/netcdf_coordinate_system_parser.h/cpp` 中的相关时间处理代码将被大幅削减或移除。
    *   预计删除约 400-500 行代码。

### 3.3 阶段三：异步框架统一 (Asynchronous Framework Unification)

**目标：** 统一异步操作的执行方式，使用 `common_utils::async::TaskManager` 替代散布在代码中的 `boost::future` 和手动线程池管理。

**具体实施：**

1.  **引入 `UnifiedAsyncExecutor`**:
    *   创建 `UnifiedAsyncExecutor` 类，作为 `common_utils::async::TaskManager` 的外观或包装器。
    *   提供统一的异步任务提交接口，支持优先级、取消、超时等。
    *   `core_services_impl/data_access_service/include/core_services/data_access/boost_future_config.h` 可能不再需要。

2.  **重构 `RawDataAccessServiceImpl` (将被 `UnifiedDataAccessServiceImpl` 替代)**:
    *   位于 `core_services_impl/data_access_service/src/impl/raw_data_access_service_impl.h` 和 `.cpp` 中的所有异步方法（当前返回 `boost::future<T>`）将通过 `UnifiedAsyncExecutor` 执行。
    *   创建 `executeReaderOperationAsync` 模板方法：此方法封装了异步获取读取器（通过新的 `ReaderRegistry` 和 `IMetadataProvider::extractFileMetadataSummaryAsync` 进行格式探测和元数据预取，然后通过 `ReaderCache` 获取或创建读取器实例）并异步执行具体操作的通用逻辑。这将极大地减少 `raw_data_access_service_impl.cpp` 中的重复代码（例如，`readGridDataAsync`, `readFeatureCollectionAsync` 等方法的实现中获取和使用reader的部分）。

3.  **代码清理**:
    *   `core_services_impl/data_access_service/src/impl/raw_data_access_service_impl.cpp` 中的大量 `boost::future` 相关模板代码和重复的异步流程控制代码将被删除。
    *   预计删除约 1000-1300 行代码。

### 3.4 阶段四：读取器架构重构 (Reader Architecture Refactoring)

**目标：** 拆分庞大的 `IDataReaderImpl` 接口，引入更灵活的读取器注册和发现机制，消除 `SharedReaderVariant` 的硬编码限制。

**具体实施：**

1.  **拆分 `IDataReaderImpl`**:
    *   `core_services_impl/data_access_service/include/core_services/data_access/i_data_reader_impl.h` 将被以下符合接口隔离原则 (ISP) 的新接口取代：
        *   `IDataSource`: 管理数据源生命周期 (`openAsync`, `closeAsync`, `isOpenAsync`, `getFilePath`).
        *   `IMetadataProvider`: 提供异步元数据提取方法 (`getVariableNamesAsync`, `getNativeCrsAsync`, `getSpatialExtentAsync`, `getTimeExtentAsync`, `getGlobalAttributesAsync`, `getVariableAttributesAsync(varName)`, `getGridDefinitionAsync(varName)`, `getLayerNamesAsync`, `getLayerMetadataAsync(layerName)`, `getFieldDefinitionsAsync(layerName)`,以及新增的 `extractFileMetadataSummaryAsync` 用于快速获取文件摘要以支持格式探测和缓存决策）。
        *   `IDataProvider`: 提供异步数据读取方法，使用新的请求/响应结构体 (如 `GridReadRequest`, `GridDataResponse`, `FeatureReadRequest`, `FeatureCollectionResponse`, `RawVariableReadRequest`, `RawVariableDataResponse`)。例如: `readGridDataAsync(const GridReadRequest& request)`, `readFeatureCollectionAsync(const FeatureReadRequest& request)`.
        *   `IStreamingDataProvider` (详见阶段五)。
    *   这些新接口将明确使用 `boost::future` 或等效的 `std::future` (如果 `common_utils::async` 支持) 并遵循异步模式。

2.  **引入 `UnifiedDataReader`**:
    *   创建一个抽象基类 `UnifiedDataReader`，它将实现上述所有新接口 (`IDataSource`, `IMetadataProvider`, `IDataProvider`, `IStreamingDataProvider`)。
    *   具体的读取器，如 `NetCdfUnifiedReader`, `GdalRasterUnifiedReader`, `GdalVectorUnifiedReader`，将继承自 `UnifiedDataReader`。
    *   现有读取器实现将被重构：
        *   **GDAL Readers**:
            *   `core_services_impl/data_access_service/include/core_services/data_access/readers/gdal/gdal_raster_reader.h` (及其 `.cpp`)
            *   `core_services_impl/data_access_service/include/core_services/data_access/readers/gdal/gdal_vector_reader.h` (及其 `.cpp`)
            *   `core_services_impl/data_access_service/include/core_services/data_access/readers/gdal/gdal_reader.h` (及其 `.cpp` - 通用 GDAL 基类)
            *   内部组件如 `gdal_dataset_handler.h/.cpp`, `gdal_raster_io.h/.cpp`, `gdal_vector_io.h/.cpp`, `gdal_metadata_extractor.h/.cpp` (及其派生类) 将被调整以支持新的接口和异步操作。例如，元数据提取方法将变为异步。
        *   **NetCDF Reader**:
            *   `core_services_impl/data_access_service/src/impl/readers/netcdf/netcdf_cf_reader.h` (及其 `.cpp`)
            *   其大量的内部辅助类 (如 `io` 下的 `netcdf_attribute_io`, `netcdf_dimension_io`, `netcdf_variable_io`; `parsing` 下的 `netcdf_cf_conventions`, `netcdf_coordinate_decoder`, `netcdf_coordinate_system_parser`, `netcdf_grid_mapping_parser`, `netcdf_metadata_parser`; 以及 `netcdf_file_processor`, `netcdf_metadata_manager`) 将被重组和调整以适应新的异步接口和 `UnifiedDataReader` 结构。

3.  **替换 `SharedReaderVariant` 和 `ReaderFactory`**:
    *   `core_services_impl/data_access_service/include/core_services/data_access/readers/data_reader_common.h` 中的 `SharedReaderVariant` 将被移除。
    *   `core_services_impl/data_access_service/src/impl/factory/reader_factory.h` 和 `.cpp` 的功能将由以下组件替代：
        *   **`ReaderRegistry`**: 负责注册和创建读取器实例。
            *   提供 `registerReaderFactory(formatName, priority, ReaderFactoryFn)`。
            *   `ReaderFactoryFn` 是一个函数，例如 `std::function<std::shared_ptr<UnifiedDataReader>(const std::string& filePath, std::shared_ptr<ICrsService> crsService, ...)>`。
            *   `createReaderAsync(filePath, detectedFormatHint)`: 根据格式提示创建读取器。
        *   **`FormatDetector`**: 定义一个接口 `IFormatDetector`，包含 `detectFormatAsync(filePath, metadataSummary)` 方法。
            *   可以有多个实现，如 `ExtensionBasedFormatDetector`, `MagicNumberFormatDetector`, `MetadataBasedFormatDetector` (利用 `IMetadataProvider::extractFileMetadataSummaryAsync` 的结果)。
            *   `ReaderRegistry` 将使用一个或多个 `IFormatDetector` 来确定文件格式。

4.  **代码清理**:
    *   删除 `core_services_impl/data_access_service/include/core_services/data_access/i_data_reader_impl.h`。
    *   `core_services_impl/data_access_service/include/core_services/data_access/readers/data_reader_common.h` 将被大幅修改或其内容分散到新的类型定义中。
    *   删除 `core_services_impl/data_access_service/src/impl/factory/reader_factory.h` 和 `.cpp`。
    *   GDAL 和 NetCDF 读取器的头文件和源文件将经历显著重构。
    *   `core_services_impl/data_access_service/src/impl/raw_data_access_service_impl.h` 和 `.cpp` 将被新的 `UnifiedDataAccessServiceImpl` 取代，其内部逻辑大量改变。

### 3.5 阶段五：流式特性实现 (Streaming Feature Implementation)

**目标：** 为数据读取添加流式处理能力，以支持大数据集和低延迟场景。

**具体实施：**

1.  **定义 `IStreamingDataProvider` 接口** (已在阶段四中提及):
    *   `openStreamAsync(const StreamingOptions& options) -> boost::future<StreamHandle>`
    *   `readNextChunkAsync(StreamHandle handle) -> boost::future<std::optional<DataChunk>>` (拉模式)
    *   `streamVariableDataAsync(const StreamingReadRequest& request, std::function<boost::future<bool>(DataChunk)> chunkProcessor) -> boost::future<void>` (推模式，`chunkProcessor` 返回 `false` 来施加反压)。
    *   `closeStreamAsync(StreamHandle handle) -> boost::future<void>`
    *   定义 `DataChunk` 结构 (包含数据、元数据、块索引等)。
    *   定义 `StreamingOptions` (包含块大小、数据类型、变量名、过滤条件、反压参数等)。
    *   定义 `StreamingReadRequest` (包含变量名、过滤条件、可选的 `AdaptiveChunkingConfig` 等)。
    *   `BackpressureControl` 机制将通过 `chunkProcessor` 的返回值实现。

2.  **在 `UnifiedDataReader` 子类中实现接口**:
    *   `NetCdfUnifiedReader`, `GdalRasterUnifiedReader` 等将实现 `IStreamingDataProvider` 接口。
    *   这将涉及修改其内部的 IO 逻辑 (如 `gdal_raster_io.cpp`, `netcdf_variable_io.cpp`) 以支持分块读取和按需加载。

3.  **创建 `StreamingDataProcessor`**:
    *   此类将作为流式处理的协调器。
    *   方法：`processStreamAsync(const std::string& filePath, const StreamingReadRequest& request, std::function<boost::future<bool>(DataChunk)> chunkProcessor)`。
    *   内部使用 `ReaderRegistry` 获取 `IStreamingDataProvider`。
    *   管理流的生命周期、错误处理。
    *   集成 `AdaptiveChunkingConfig`（来自 `StreamingOptions`），并根据 `DataGeometryType`（从读取器元数据获取）和可选的 `common_utils::MemoryMonitor` 及 `common_utils::PerformanceProfiler` (如果可用) 来动态调整块大小和读取策略 (`determineAdaptiveStreamingOptions`)。
    *   所有异步操作通过 `UnifiedAsyncExecutor` 执行。

### 3.6 提议的新目录结构 (`core_services_impl/data_access_service/`)
```
core_services_impl/data_access_service/
├── include/core_services/data_access/  # 公共接口和核心类型
│   ├── api/                            # 服务接口和主要数据结构
│   │   ├── i_data_access_service.h     # (新的) 主服务接口
│   │   ├── i_data_source.h             # 数据源生命周期接口
│   │   ├── i_metadata_provider.h       # 元数据提供接口
│   │   ├── i_data_provider.h           # 数据读取接口
│   │   ├── i_streaming_data_provider.h # 流式数据读取接口
│   │   ├── data_access_requests.h      # (GridReadRequest, FeatureReadRequest, etc.)
│   │   ├── data_access_responses.h     # (GridDataResponse, FeatureCollectionResponse, etc.)
│   │   └── data_chunk.h                # 流式数据块定义
│   ├── common/                         # 模块内共享的类型定义
│   │   ├── data_access_common_types.h  # (替代旧的 data_reader_common.h 部分内容, 如 DataChunkKey, ReaderType)
│   │   ├── data_access_error_codes.h
│   │   └── streaming_options.h
│   └── cache/                          # 如果需要暴露特定的缓存接口或类型
│       └── cacheable_value.h           # CacheableValue 接口定义
│
└── src/                                # 实现代码
    ├── main/                           # 主服务实现
    │   ├── unified_data_access_service_impl.h
    │   └── unified_data_access_service_impl.cpp
    ├── cache/                          # 统一缓存实现
    │   ├── unified_data_access_cache.h
    │   └── unified_data_access_cache.cpp
    ├── async/                          # 统一异步执行器
    │   ├── unified_async_executor.h
    │   └── unified_async_executor.cpp
    ├── readers/                        # 读取器相关实现
    │   ├── core/                       # 读取器核心组件
    │   │   ├── unified_data_reader.h         # (新的) 抽象基类
    │   │   ├── unified_data_reader.cpp
    │   │   ├── reader_registry.h
    │   │   ├── reader_registry.cpp
    │   │   ├── format_detector.h           # (IFormatDetector 及其实现)
    │   │   └── format_detector.cpp
    │   ├── gdal/                       # GDAL 读取器实现
    │   │   ├── gdal_unified_reader.h       # (新的) 统一GDAL读取器
    │   │   ├── gdal_unified_reader.cpp
    │   │   └── internal/                 # GDAL内部辅助组件 (保持封装)
    │   │   │   ├── gdal_dataset_handler.h
    │   │   │   ├── gdal_dataset_handler.cpp
    │   │   │   ├── gdal_metadata_extractor.h  // (refactored)
    │   │   │   ├── gdal_metadata_extractor.cpp // (refactored)
    │   │   │   ├── gdal_raster_metadata_extractor.h // (refactored)
    │   │   │   ├── gdal_raster_metadata_extractor.cpp // (refactored)
    │   │   │   ├── gdal_vector_metadata_extractor.h // (refactored)
    │   │   │   ├── gdal_vector_metadata_extractor.cpp // (refactored)
    │   │   │   ├── gdal_raster_io.h          // (refactored)
    │   │   │   ├── gdal_raster_io.cpp        // (refactored)
    │   │   │   ├── gdal_vector_io.h          // (refactored)
    │   │   │   ├── gdal_vector_io.cpp        // (refactored)
    │   │   │   └── utils/                  # (gdal_common_utils, gdal_api_compat, etc. - refactored as needed)
    │   │   │       ├── gdal_common_utils.h
    │   │   │       ├── gdal_common_utils.cpp
    │   │   │       ├── gdal_api_compat.h
    │   │   │       ├── gdal_type_conversion.h
    │   │   │       ├── gdal_type_conversion.cpp
    │   │   │       ├── gdal_transformation_utils.h
    │   │   │       └── gdal_transformation_utils.cpp
    │   ├── netcdf/                     # NetCDF 读取器实现
    │   │   ├── netcdf_unified_reader.h     # (新的) 统一NetCDF读取器
    │   │   ├── netcdf_unified_reader.cpp
    │   │   └── internal/                 # NetCDF内部辅助组件 (保持封装)
    │   │       ├── netcdf_file_processor.h     // (refactored)
    │   │       ├── netcdf_file_processor.cpp   // (refactored)
    │   │       ├── netcdf_metadata_manager.h   // (refactored)
    │   │       ├── netcdf_metadata_manager.cpp // (refactored)
    │   │       ├── io/                       // (refactored io helpers)
    │   │       │   ├── netcdf_attribute_io.h
    │   │       │   ├── netcdf_attribute_io.cpp
    │   │       │   ├── netcdf_dimension_io.h
    │   │       │   ├── netcdf_dimension_io.cpp
    │   │       │   ├── netcdf_variable_io.h
    │   │       │   └── netcdf_variable_io.cpp
    │   │       ├── parsing/                  // (refactored parsing helpers)
    │   │       │   ├── netcdf_cf_conventions.h
    │   │       │   ├── netcdf_cf_conventions.cpp
    │   │       │   ├── netcdf_coordinate_decoder.h
    │   │       │   ├── netcdf_coordinate_decoder.cpp
    │   │       │   ├── netcdf_coordinate_system_parser.h // (time logic removed/reduced)
    │   │       │   ├── netcdf_coordinate_system_parser.cpp // (time logic removed/reduced)
    │   │       │   ├── netcdf_grid_mapping_parser.h
    │   │       │   ├── netcdf_grid_mapping_parser.cpp
    │   │       │   └── netcdf_metadata_parser.h
    │   │       │   └── netcdf_metadata_parser.cpp
    │   │       └── utils/                    // (refactored netcdf_reader_utils)
    │   │           ├── netcdf_reader_utils.h
    │   │           └── netcdf_reader_utils.cpp
    │   └── streaming/                  # 流式处理实现
    │       ├── streaming_data_processor.h
    │       └── streaming_data_processor.cpp
    ├── time_processing/                # 时间处理相关 (如果 CFTimeExtractor 很复杂)
    │   ├── cf_time_extractor.h
    │   └── cf_time_extractor.cpp
    ├── crs/                            # CRS 相关服务 (如果 gdal_crs_service_impl 仍在此模块)
    │   ├── gdal_crs_service_impl.h       # (might move to a common_services_impl if used more broadly)
    │   └── gdal_crs_service_impl.cpp
    └── utils/                          # 模块内其他工具类
        ├── data_access_utils.h         # (e.g., dimension_converter.h, data_type_converters.h refactored contents)
        └── data_access_utils.cpp
```

### 3.7 重构阶段细化与实施计划

基于对现有代码的深入分析和"最小化目录改动"的原则，我们对重构方案进行以下细化：

#
#### 3.7.3 详细重构阶段

**阶段 0：准备阶段 (Week 0)**
- **备份现有代码**：完整备份当前 `data_access_service` 模块
- **建立测试基准**：确保现有所有测试都通过，记录性能基准
- **依赖分析**：确认 `common_utils` 模块的可用功能和接口

**阶段 1：缓存系统统一 (Week 1-2)**

*目标：消除4套独立缓存，统一使用 `common_utils::cache`*

**具体操作：**
1. **创建最小适配层**：
   ```cpp
   // src/impl/cache/unified_data_access_cache.h/cpp
   class UnifiedDataAccessCache {
   private:
       // 仅使用 common_utils::cache::ICacheManager 的实例
       std::shared_ptr<common_utils::cache::ICacheManager<DataChunkKey, std::shared_ptr<GridData>>> dataChunkCache_;
       std::shared_ptr<common_utils::cache::ICacheManager<std::string, std::shared_ptr<IDataReaderImpl>>> readerCache_;
       std::shared_ptr<common_utils::cache::ICacheManager<std::string, std::string>> metadataCache_;
   public:
       // 提供领域特定的缓存接口
       boost::future<std::optional<std::shared_ptr<GridData>>> getDataChunkAsync(const DataChunkKey& key);
       boost::future<std::optional<std::shared_ptr<IDataReaderImpl>>> getReaderAsync(const std::string& filePath);
   };
   ```

2. **文件删除**：
   - ❌ `src/impl/cache/data_chunk_cache.h/cpp`
   - ❌ `src/impl/cache/reader_cache.h/cpp`
   - ❌ `src/impl/cache/metadata_cache.h/cpp`
   - ❌ `src/impl/cache/netcdf_cache_manager.h/cpp`
   - ❌ `src/impl/cache/cache_manager_template.h`

3. **集成点修改**：
   - 修改 `raw_data_access_service_impl.cpp` 中的缓存调用
   - 替换所有独立缓存访问为 `UnifiedDataAccessCache` 调用

**预期收益：删除约 2000 行重复代码，内存利用率提升至 85%+**

**阶段 2：异步框架统一 (Week 2-3)**

*目标：统一异步操作，消除重复的 boost::future 模板代码*

**具体操作：**
1. **创建异步执行器**：
   ```cpp
   // src/impl/async/unified_async_executor.h/cpp
   class UnifiedAsyncExecutor {
   private:
       std::shared_ptr<common_utils::async::TaskManager> taskManager_;
   public:
       template<typename ReturnType, typename ReaderOperation>
       boost::future<ReturnType> executeReaderOperationAsync(
           const std::string& filePath,
           ReaderOperation&& operation
       );
   };
   ```

2. **重构 RawDataAccessServiceImpl**：
   - 保持 `raw_data_access_service_impl.h/cpp` 文件名
   - 重构内部实现，使用 `UnifiedAsyncExecutor`
   - 消除 17 个异步方法中的重复代码

**预期收益：删除约 1300 行重复异步代码**

**阶段 3：时间处理统一 (Week 3-4)**

*目标：消除自定义时间处理，全面使用 `common_utils::time`*

**具体操作：**
1. **创建CF时间提取器**：
   ```cpp
   // src/impl/time/cf_time_extractor.h/cpp
   class CFTimeExtractor : public common_utils::time::ITimeExtractor {
       // 实现CF约定的时间提取逻辑
   };
   ```

2. **文件清理**：
   - ❌ `src/impl/readers/netcdf/parsing/netcdf_time_processor.h/cpp`
   - 🔄 重构 `netcdf_coordinate_system_parser.h/cpp`（移除时间逻辑，保留其他功能）

3. **全面替换**：
   - 所有时间相关代码使用 `common_utils::time` 类型
   - NetCDF 读取器使用 `CFTimeExtractor`

**预期收益：删除约 500 行重复时间处理代码**

**阶段 4：读取器架构重构 (Week 4-5)**

*目标：拆分 IDataReaderImpl，引入 UnifiedDataReader 架构*

**具体操作：**
1. **新接口定义**：
   ```cpp
   // include/core_services/data_access/api/
   // i_data_source.h, i_metadata_provider.h, i_data_provider.h
   ```

2. **重构现有读取器**（保持文件名，重构内容）：
   - 🔄 `gdal_raster_reader.h/cpp` → 实现新接口集
   - 🔄 `gdal_vector_reader.h/cpp` → 实现新接口集  
   - 🔄 `netcdf_cf_reader.h/cpp` → 实现新接口集

3. **工厂模式重构**：
   - 🔄 `factory/reader_factory.h/cpp` → 重构为 `ReaderRegistry` 实现
   - 保持文件名，完全重写内容

4. **逐步替换**：
   - 🔄 `data_reader_common.h` → 移除 `SharedReaderVariant`，保留其他类型定义
   - 分阶段替换 `raw_data_access_service_impl.cpp` 中的读取器使用方式

**预期收益：解决ISP违规，提升扩展性**

**阶段 5：流式特性实现 (Week 5-6)**

*目标：为大数据场景添加流式处理能力*

**具体操作：**
1. **流式接口实现**：
   - 在现有读取器中实现 `IStreamingDataProvider`
   - 修改 `gdal_raster_io.cpp`, `netcdf_variable_io.cpp` 支持分块读取

2.  **流式处理器**：
   ```cpp
   // src/impl/streaming/streaming_data_processor.h/cpp （新增）
   class StreamingDataProcessor {
       // 协调流式数据处理
   };
   ```

**预期收益：支持TB级数据文件的内存高效处理**

#### 3.7.4 风险控制与验证

**每阶段验证要求：**
1. **单元测试通过率 100%**
2. **集成测试通过率 100%**
3. **性能不低于重构前基准**
4. **内存泄漏检测通过**

**回滚方案：**
- 每个阶段完成后创建Git分支
- 提供详细的回滚文档
- 保留原始备份直到整个重构完成

#### 3.7.5 资源与时间估算

**总工期：6-7周**
- 开发时间：5-6周
- 测试验证：1-2周（与开发并行）
- 文档更新：0.5周

**团队配置建议：**
- 核心开发工程师：2人
- 测试工程师：1人
- 代码审查：1人（兼职）

**关键里程碑：**
- Week 2：缓存统一完成，性能测试通过
- Week 4：异步+时间处理完成，所有原有功能正常
- Week 6：读取器架构重构完成，新架构验证通过
- Week 7：流式特性完成，整体验收通过

### 3.8 代码清理和迁移总结

**删除文件清单：**
```
core_services_impl/data_access_service/src/impl/cache/
├── ❌ data_chunk_cache.h (597行)
├── ❌ data_chunk_cache.cpp 
├── ❌ reader_cache.h (424行)
├── ❌ reader_cache.cpp
├── ❌ metadata_cache.h (253行)
├── ❌ metadata_cache.cpp
├── ❌ netcdf_cache_manager.h (429行)
├── ❌ netcdf_cache_manager.cpp
└── ❌ cache_manager_template.h

core_services_impl/data_access_service/src/impl/readers/netcdf/parsing/
├── ❌ netcdf_time_processor.h (~200行)
└── ❌ netcdf_time_processor.cpp

include/core_services/data_access/
└── ❌ boost_future_config.h （可能不再需要）
```

**新增文件清单：**
```
include/core_services/data_access/
├── cache/unified_data_access_cache.h
└── api/
    ├── i_data_source.h
    ├── i_metadata_provider.h  
    ├── i_data_provider.h
    └── i_streaming_data_provider.h

src/impl/
├── cache/unified_data_access_cache.cpp
├── async/unified_async_executor.h/cpp
├── time/cf_time_extractor.h/cpp
└── streaming/streaming_data_processor.h/cpp
```

**重构文件清单：**
```
✏️ src/impl/raw_data_access_service_impl.h/cpp        # 核心服务重构
✏️ src/impl/factory/reader_factory.h/cpp              # 重构为ReaderRegistry
✏️ src/impl/readers/gdal/gdal_raster_reader.h/cpp     # 实现新接口
✏️ src/impl/readers/gdal/gdal_vector_reader.h/cpp     # 实现新接口
✏️ src/impl/readers/netcdf/netcdf_cf_reader.h/cpp     # 实现新接口
✏️ include/core_services/data_access/readers/data_reader_common.h  # 移除SharedReaderVariant
✏️ src/impl/readers/netcdf/parsing/netcdf_coordinate_system_parser.h/cpp  # 移除时间逻辑
```

**代码统计：**
- 删除代码：约 3,200+ 行（主要是重复的缓存、异步、时间处理代码）
- 新增代码：约 1,800 行（新架构、流式处理、适配层）
- 重构代码：约 2,000 行（现有读取器和服务类）
- **净减少：约 1,400 行代码**

**预期性能提升：**
- 缓存命中率：45-60% → 80-85%
- 内存利用率：60-70% → 85-90%  
- 异步并发能力：提升 30-50%
- 大文件处理：支持流式处理，理论上无文件大小限制

// ... existing code ...