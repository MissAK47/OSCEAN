# OSCEAN 数据访问与核心模块统一重构方案 (最终实施版)

## 0. 核心原则回顾

*   **不保留向后兼容性**: 直接采用新设计，以避免技术债务。
*   **统一核心数据结构**: 严格遵循 `core_service_interfaces/include/core_services/common_data_types.h` 中的新定义（详见 `重构API接口.md`）。
*   **模块化与高内聚**: 将大文件和复杂类拆分为职责单一的小组件。
*   **简化并发控制**: 移除 `LockLevel`，采用局部化、RAII锁。

## 1. 引言与目标

本文档为OSCEAN项目的数据访问层及相关核心模块提供最终的、代码级的重构方案。目标是解决现有代码库的结构性问题，构建一个清晰、模块化、可维护和可扩展的系统。

**核心目标**：

1.  **统一数据结构**: 全面采用标准化的 `GridData`、`GridDefinition` 和 `DimensionCoordinateInfo` 结构。
2.  **模块化与内聚**: 将大型复杂类拆分为职责更单一的小型模块，解决超大文件问题。
3.  **消除冗余**: 移除重复的类型定义和逻辑。
4.  **简化并发控制**: 改进锁机制。
5.  **提升代码质量**: 遵循C++最佳实践和项目编码规范。

## 2. 核心数据结构定义

**主要影响文件**: `core_service_interfaces/include/core_services/common_data_types.h`
**API详细文档**: `重构API接口.md`

本项目将采用一组统一的核心数据结构来表示地理空间数据。这些结构旨在提供清晰、一致且可扩展的数据模型。详细的C++定义和Doxygen文档位于 `重构API接口.md`，并最终在 `common_data_types.h` 中实现。

核心结构摘要：

*   **`CoordinateDimension` 枚举**: 定义标准的坐标维度类型 (`LON`, `LAT`, `VERTICAL`, `TIME`, `SPECTRAL`, `OTHER`, `NONE`)。
*   **`AttributeValue` 类型定义**: 使用 `std::variant` 存储多种类型的元数据属性值。
*   **`ValueRange` 结构体**: 表示可选最小/最大值的范围。
*   **`DimensionCoordinateInfo` 结构体**: 全面描述单个坐标维度，包括标识、坐标数据、特性、质量、变换、特定信息 (通过 `std::variant` 实现时间、垂直、光谱的特有属性) 及元数据。
*   **`GridDefinition` 类**: 定义N维格网的结构和元数据，使用 `DimensionCoordinateInfo` 描述各主要维度，并通过 `dimensionOrderInDataLayout` 指定数据物理存储顺序。
*   **`GridData` 类**: 持有 `GridDefinition` 和实际数据缓冲区 (`std::vector<unsigned char>`)。

这些统一的结构是本次重构的基础，旨在提升代码的清晰度与可维护性。

## 3. `readers` 子模块代码级重构 (重点关注文件拆分与职责)

**目标**: 将超大文件拆分为更小、职责更单一的模块/类，遵循高内聚低耦合原则。行数目标为建议值，旨在控制模块复杂度。

### 3.1. 建议目录结构 (源自 `重构方案讨论.md`)
```
core_services_impl/data_access_service/src/impl/readers/
├── common/                             # 通用读取器辅助组件
│   ├── abstract_reader_utils.h/.cpp    # (可选，如果需要通用基类或工具)
│   └── data_processing_utils.h/.cpp  # (如数据类型转换、解包逻辑等)
│
├── netcdf/
│   ├── netcdf_cf_reader.h/.cpp         # 主NetCDF CF读取器 (高层逻辑)
│   ├── netcdf_file_processor.h/.cpp    # 底层NetCDF文件打开/关闭，ncid管理
│   ├── io/                               # IO操作层 (直接与NetCDF C API交互)
│   │   ├── netcdf_attribute_io.h/.cpp      # 属性读写
│   │   ├── netcdf_dimension_io.h/.cpp      # 维度定义读取
│   │   └── netcdf_variable_io.h/.cpp       # 变量原始数据块读写
│   ├── parsing/                          # 解析逻辑层 (将原始信息转换为模型)
│   │   ├── netcdf_metadata_parser.h/.cpp   # 解析通用元数据到DimensionCoordinateInfo/GridDefinition
│   │   ├── netcdf_coordinate_decoder.h/.cpp # 解码坐标变量数据
│   │   ├── netcdf_cf_conventions.h/.cpp    # CF约定处理 (核心语义解析)
│   │   └── netcdf_time_processor.h/.cpp    # CF时间处理
│   └── utils/                              # NetCDF特定辅助工具
│       └── netcdf_reader_utils.h/.cpp    # (如NetCDF数据类型映射等)
│
├── gdal/                                 # GDAL读取器模块
│   ├── gdal_raster_reader.h/.cpp         # 主GDAL栅格读取器
│   ├── gdal_vector_reader.h/.cpp         # 主GDAL矢量读取器
│   ├── gdal_dataset_handler.h/.cpp       # 封装GDALDataset* 打开/关闭
│   ├── metadata/                         # GDAL元数据提取
│   │   ├── gdal_raster_metadata_extractor.h/.cpp
│   │   └── gdal_vector_metadata_extractor.h/.cpp
│   ├── io/                               # GDAL IO操作
│   │   ├── gdal_raster_io.h/.cpp         # RasterIO等
│   │   └── gdal_vector_feature_io.h/.cpp # Feature读取
│   └── utils/                              # GDAL特定辅助工具
│       └── gdal_common_utils.h/.cpp
└── ... (其他读取器，如HDF等，可遵循类似结构)
```

### 3.2. NetCDF 组件职责 (源自 `重构方案讨论.md` 并整合)

*   **`netcdf_file_processor.h/.cpp`**:
    *   **职责**: 仅封装 `ncid` 管理 (open, close, inquire format, isOpen) 和最基本的文件级查询。确保其操作线程安全。
    *   **行数目标**: < 200行。
*   **`io/netcdf_attribute_io.h/.cpp`**:
    *   **职责**: 负责全局属性和变量属性的纯粹NetCDF API读写。
    *   **行数目标**: ~150-200行。
*   **`io/netcdf_dimension_io.h/.cpp`**:
    *   **职责**: 负责维度定义（名称、长度）的纯粹NetCDF API读取。
    *   **行数目标**: ~100-150行。
*   **`io/netcdf_variable_io.h/.cpp`**:
    *   **职责**: 负责变量元数据（类型、维度ID列表）和原始数据块（`nc_get_vara_uchar` 等）的纯粹NetCDF API读写。不处理数据解包（scale/offset）。
    *   **行数目标**: ~200-300行。
*   **`parsing/netcdf_metadata_parser.h/.cpp`**:
    *   **职责**: 从IO组件获取原始属性、维度、变量信息，进行初步解析，填充 `DimensionCoordinateInfo` 和 `GridDefinition` 中的非CF特定、非坐标数据本身的字段。
    *   **行数目标**: ~200-250行。
*   **`parsing/netcdf_coordinate_decoder.h/.cpp`**:
    *   **职责**: 识别潜在的坐标变量，调用 `NetCDFVariableIO` 读取其数据和 `NetCDFAttributeIO` 读取其属性，填充 `DimensionCoordinateInfo` 的 `coordinates`, `coordinateLabels`, `coordinateBounds` 等坐标数据相关字段。
    *   **行数目标**: ~250-300行。
*   **`parsing/netcdf_cf_conventions.h/.cpp`**:
    *   **职责**: **核心CF逻辑处理**。接收已部分填充的 `DimensionCoordinateInfo` 对象集合及变量元数据。应用CF规则（如 `axis`, `standard_name`, `units`, `positive`, `_CoordinateAxisType`, `grid_mapping`, `formula_terms` 等）来最终确定每个 `DimensionCoordinateInfo` 的 `type` (e.g., `CoordinateDimension::LON`), 正确解析并填充 `specificInfo` (如 `TimeSpecificInfo`, `VerticalSpecificInfo`), 解析 `grid_mapping` 变量以填充 `CRSInfo`。
    *   **行数目标**: ~300-450行 (可能较大，因CF约定复杂)。
*   **`parsing/netcdf_time_processor.h/.cpp`**:
    *   **职责**: 专门处理CF时间维度。输入时间坐标变量的 `units` 和 `calendar` 属性（来自 `DimensionCoordinateInfo`），将其解析为 `TimeSpecificInfo::referenceEpochString` 和 `TimeSpecificInfo::calendar`。处理时间坐标值的转换（如果需要从数值转换为标准时间表示或反之）。
    *   **行数目标**: ~150-200行。
*   **`utils/netcdf_reader_utils.h/.cpp`**:
    *   **职责**: 存放NetCDF数据类型到内部 `DataType` 的映射，错误处理辅助函数等NetCDF相关的通用工具。
    *   **行数目标**: ~100行。
*   **`netcdf_cf_reader.h/.cpp` (主读取器)**:
    *   **职责**: 实现 `AbstractReader` 接口 (如果存在)。作为**编排者 (Orchestrator)** 调用上述所有 `io` 和 `parsing` 组件。
    *   `openImpl()` (或类似方法):
        1.  使用 `NetCDFFileProcessor` 打开文件。
        2.  调用 `NetCDFDimensionIO`, `NetCDFAttributeIO`, `NetCDFVariableIO` 获取原始维度、属性、变量定义信息。
        3.  使用 `NetCDFMetadataParser` 进行初步元数据解析。
        4.  使用 `NetCDFCoordinateDecoder` 解码所有潜在坐标变量的数据。
        5.  使用 `NetCDFCFConventions` 和 `NetCDFTimeProcessor` 应用CF语义并最终完成所有相关变量的 `GridDefinition` (包含所有 `DimensionCoordinateInfo`) 的构建。
        6.  缓存构建好的 `GridDefinition` 对象。
    *   `readGridDataImpl()` (或类似方法):
        1.  获取缓存的或按需构建的 `GridDefinition`。
        2.  根据请求范围和 `GridDefinition` 确定读取策略。
        3.  调用 `NetCDFVariableIO` 读取请求变量的原始数据块到 `std::vector<unsigned char>`。
        4.  **数据解包与处理**: 在 `common/data_processing_utils.cpp` (或直接在Reader中) 实现逻辑，应用 `DimensionCoordinateInfo` 中的 `scale_factor`, `add_offset`。处理 `_FillValue`/`missingValue` (例如替换为 `std::nan` for float/double)。
        5.  根据 `GridDefinition::dimensionOrderInDataLayout` 将数据正确排列（如果需要转置）到 `GridData::_buffer`。
    *   **行数目标**: ~400-600行（主要是流程控制、错误处理和组件调用）。

### 3.3. GDAL 组件职责 (类似拆分思路，源自 `重构方案讨论.md`)
*   **`gdal_dataset_handler.h/.cpp`**: 封装 `GDALDataset*` 的打开、关闭、基本信息查询。线程安全。
*   **`metadata/gdal_raster_metadata_extractor.h/.cpp`**: 从 `GDALDataset` 和 `GDALRasterBand` 提取元数据，构建 `GridDefinition`，填充各 `DimensionCoordinateInfo` (X, Y 来自地理参考和尺寸，Z 来自波段信息，T 可能需要特定逻辑)。
*   **`metadata/gdal_vector_metadata_extractor.h/.cpp`**: 从 `OGRLayer` 提取元数据，构建适合矢量数据的 `FeatureCollectionDefinition` (如果需要，或适配 `GridDefinition` 用于点集等)。
*   **`io/gdal_raster_io.h/.cpp`**: 执行 `GDALDataset::RasterIO` 或 `GDALRasterBand::RasterIO` 读取数据块。
*   **`io/gdal_vector_feature_io.h/.cpp`**: 迭代读取 `OGRFeature`。
*   **`utils/gdal_common_utils.h/.cpp`**: GDAL错误处理、数据类型映射、CRS处理辅助。
*   **`gdal_raster_reader.h/.cpp` (主栅格读取器)**: 编排以上组件，实现读取栅格数据的接口。
*   **`gdal_vector_reader.h/.cpp` (主矢量读取器)**: 编排以上组件，实现读取矢量数据的接口。

## 4. 缓存模块 (`impl/cache/` - 源自 `重构方案讨论.md`)

1.  **`DataChunkCache`**:
    *   `estimateGridDataSize(const GridData& gridData)`: **必须重写**。需要递归估算 `GridData` -> `GridDefinition` -> (所有 `DimensionCoordinateInfo` 成员) -> (所有 `std::vector`, `std::map`, `std::string` 的 `capacity()` 或实际占用)。这是确保缓存大小限制有效的关键。
2.  **`MetadataCache` (建议)**:
    *   可以考虑一个独立的元数据缓存，用于缓存 `std::shared_ptr<GridDefinition>` 对象。键可以是 `filePath#variableName` 或类似的唯一标识符。这可以避免重复解析元数据。
3.  **`NetCDFCacheManager`**: **建议废弃**。其功能应由通用的 `DataChunkCache` (用于数据块) 和新增的 `MetadataCache` (用于 `GridDefinition`) 替代，以简化缓存管理。
4.  **`ReaderCache`**:
    *   如果存在，确保其能正确缓存和管理新的、拆分后的读取器实例或其句柄（如 `NetCDFFileProcessor` 实例）。
    *   更新 `SharedReaderVariant` (如果使用) 以包含新读取器的完全限定类型。

## 5. 服务与工厂 (`impl/` - 源自 `重构方案讨论.md`)

1.  **`ReaderFactory`**:
    *   更新 `createReaderInternal` (或类似方法) 以能够实例化新的、拆分后的、完全限定命名空间的读取器类（如 `NetCDFCFReader`, `GDALRasterReader`）。
    *   可能需要传递配置给这些读取器，用于初始化其子组件。
2.  **`RawDataAccessServiceImpl` (或类似服务实现)**:
    *   更新所有与数据读取器交互的方法，以使用新的读取器接口。
    *   确保返回给调用方的核心数据结构（如 `GridData`）是按照新定义构建的。

## 6. 锁机制最终方案 (源自 `重构方案讨论.md`)

1.  **彻底移除 `LockLevel` 枚举及所有相关检查逻辑。**
2.  **RAII**: 强制在所有需要同步的地方使用 `std::lock_guard` 或 `std::unique_lock`。
3.  **锁粒度与位置**:
    *   **底层文件/数据集处理器 (`NetCDFFileProcessor`, `GDALDatasetHandler`)**:
        *   每个处理器实例内部应持有一个 `std::mutex` (如果方法不互相调用) 或 `std::recursive_mutex` (如果其公共方法可能直接或间接重入调用同一实例的其他公共方法)。
        *   所有直接与C API交互（如 `nc_*` 函数, `GDAL*` 函数）的公共方法都必须在开始时获取此锁，结束时释放（通过RAII）。这使得这些处理器对象本身的操作是线程安全的。
    *   **主读取器 (`NetCDFCFReader`, `GDALRasterReader`, etc.)**:
        *   **元数据加载/`openImpl()`**: 此方法通常涉及多次调用底层处理器，并构建和缓存 `GridDefinition`。
            *   对内部缓存的 `GridDefinition` 的首次填充和后续只读访问，推荐使用 `std::call_once` 配合 `std::once_flag` 和一个 `std::mutex`（如果填充过程本身复杂且需要临时共享状态）来确保线程安全和仅初始化一次。
            *   在调用底层文件处理器（如 `NetCDFFileProcessor`）时，由于处理器本身已保证线程安全，主读取器层面不需要为这些调用额外加锁。
        *   **数据读取/`readGridDataImpl()`**:
            *   此方法理想情况下应该是可重入且无副作用的（不修改读取器实例的共享状态，除了可能的内部统计或日志记录）。它依赖线程安全的底层处理器和IO组件，并为每个请求创建新的 `GridData` 对象。
            *   如果读取器自身确实需要在实例级别维护一些可变状态（应尽量避免，例如管理一个共享的临时缓冲区），则需要为这些状态的访问配备单独的 `std::mutex`。
    *   **缓存类 (`DataChunkCache`, `MetadataCache`)**: 保持其内部现有的线程安全机制（如读写锁 `std::shared_mutex`），确保并发访问安全。

## 7. 内存池 (`impl/memory/memory_pool.h/.cpp` - 源自 `重构方案讨论.md`)

*   当前重构主要关注数据结构和模块逻辑的清晰性。`MemoryPool` 的直接改动不是本次重构的首要任务。
*   **后续评估**: 在重构稳定后，可以评估 `GridData::_buffer`（即 `std::vector<unsigned char>`）以及其他频繁分配和释放的大块内存（如某些IO操作的临时缓冲区）是否可以从自定义内存池中分配以获得显著的性能优势。
    *   如果评估结果表明有益，则需要确保通过自定义分配器与 `std::vector` 等标准容器良好集成，或为特定用途提供池化分配接口。
    *   如果性能提升不明显，或集成复杂性过高，则标准库的默认分配器通常已足够高效。

## 8. 测试策略 (整合自两方案)

*   **单元测试**:
    *   对每个新拆分的类（IO、Parsing、Utils、底层处理器、主读取器的核心逻辑单元）编写详尽的单元测试。
    *   测试 `DimensionCoordinateInfo` 的所有字段能否被正确解析和填充，包括 `specificInfo` 的各种情况。
    *   测试 `GridDefinition` 对各种维度组合（包括缺失维度、0级维度）的正确表示和 `getTotalElements()` 的准确性。
    *   测试 `GridData::calculateByteOffset` 对不同 `dimensionOrderInDataLayout` 的计算正确性，以及 `getValue`/`setValue`。
    *   测试CF约定解析的各种情况：如 `grid_mapping`，`axis`/`standard_name` 对维度类型的判断，`formula_terms`，`scale_factor`/`add_offset` 应用等。
    *   测试时间处理，覆盖不同日历和 `units` 字符串。
*   **集成测试**:
    *   从 `RawDataAccessServiceImpl` (或等效服务层) 发起请求，覆盖整个数据读取链路。
    *   使用包含各种复杂情况的NetCDF和GDAL文件（例如，您提供的 `cs_2023_01_00_00.nc` 文件作为良好用例，以及不规则坐标、多垂直层、多波段、复杂时间、不同数据类型、压缩等其他文件）。
    *   验证数据子集读取、空间/时间范围查询的准确性。
*   **性能测试**:
    *   对比重构前后读取相同文件和数据块的耗时和内存占用。
    *   重点关注 `DataChunkCache::estimateGridDataSize` 的准确性和效率对缓存性能的影响。
    *   测试高并发场景下的稳定性和吞吐量。

## 9. 实施阶段建议 (整合)

1.  **核心结构定义与实现**: 在 `common_data_types.h` 中最终确定并实现新的数据结构 (`CoordinateDimension`, `AttributeValue`, `ValueRange`, `DimensionCoordinateInfo`, `GridDefinition`, `GridData`)。确保API稳定。
2.  **底层处理器与IO模块**: 优先实现 `NetCDFFileProcessor`, `GDALDatasetHandler` 以及 `netcdf/io/` 和 `gdal/io/` 中的纯IO模块。这些是上层逻辑的基础。
3.  **解析与语义模块**: 逐步实现 `netcdf/parsing/` 和 `gdal/metadata/` 中的各个解析和约定处理模块。这是重构的核心和难点。
4.  **主读取器编排**: 实现 `NetCDFCFReader`, `GDALRasterReader` 等，将各子模块功能编排起来。
5.  **缓存和服务层更新**: 更新缓存机制和相关服务实现。
6.  **全面测试**: 持续进行单元测试、集成测试和性能测试。

---

此最终整合方案提供了明确的重构路径和详细的模块职责划分。实施时，建议从小处着手，先完成核心数据结构的定义和底层IO模块，然后逐步构建解析和上层逻辑，并始终伴随严格的测试。
