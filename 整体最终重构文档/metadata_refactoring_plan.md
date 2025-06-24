# 元数据服务 (`metadata_service`) 重构方案

## 1. 核心思想

鉴于详细的、与具体变量/格网相关的元数据（如单位、标准名、维度属性等）已整合进核心数据结构 `GridDefinition` 和 `DimensionCoordinateInfo`，重构后的 `metadata_service` 应专注于**更高层次、跨数据集的元数据管理、数据发现以及可选的元数据标准符合性**。它将从一个可能也处理细粒度元数据的角色，转变为一个更宏观的数据集目录和发现服务。

## 2. 主要职责

*   **数据集级元数据注册与管理**:
    *   提供机制来注册数据集（一个数据集可以是一个单独的NetCDF文件、一个相关NetCDF文件系列、一个GDAL支持的栅格文件等）。
    *   存储与整个数据集相关的元数据，例如：
        *   数据集标识符（唯一ID、逻辑名称）。
        *   文件位置（单个路径、路径列表、路径模式）。
        *   总体时间覆盖范围（`start_datetime`, `end_datetime`）。
        *   总体空间覆盖范围（`BoundingBox`）。
        *   包含的主要变量或参数列表。
        *   数据来源、所有权、版权、许可信息。
        *   数据处理历史或血缘关系的简要描述或链接。
        *   数据质量摘要或链接到详细质量报告。
        *   用户定义的标签或分类。
    *   支持更新和删除已注册的数据集元数据。

*   **数据发现与查询**:
    *   提供强大的查询接口，允许用户根据多种元数据条件搜索和发现数据集。
    *   查询条件可以包括：时间范围、空间范围（与查询边界框相交、包含等）、变量名、关键词、来源等。
    *   查询结果应返回匹配数据集的标识符和关键元数据，足以让用户决定是否进一步访问该数据（例如，返回文件路径供 `data_access_service` 使用）。

*   **元数据提取与索引构建 (核心功能)**:
    *   实现机制来从物理数据文件（如NetCDF）中自动或半自动提取上述数据集级元数据，以填充注册表。这可能涉及：
        *   扫描指定的文件系统目录。
        *   轻量级读取文件头部或全局属性（避免加载完整数据）。
        *   解析文件名约定以提取时间等信息。
    *   构建和维护一个内部索引（"数据库"）以支持高效的元数据查询。

*   **元数据标准符合性 (可选)**:
    *   如果项目需要，可以支持特定的元数据标准（如ISO 19115, CF Conventions全局属性部分, Dublin Core等）。
    *   提供元数据验证工具，或在注册时进行标准映射/转换。

## 3. 建议组件/类

*   **`DatasetMetadataEntry` (或 `DatasetRecord`) (数据结构)**:
    *   **职责**: 定义存储在注册表中的单个数据集的元数据结构。应包含上述"数据集级元数据注册与管理"中列出的字段。
    *   例如: `std::string id; std::vector<std::string> filePaths; TimeRange timeCoverage; BoundingBox spatialCoverage; std::vector<std::string> variables; std::map<std::string, AttributeValue> customAttributes;` 等。

*   **`IDatasetMetadataRegistryBackend` (接口) & 实现类 (如 `InMemoryRegistryBackend`, `SQLiteRegistryBackend`)**:
    *   **职责**: 定义元数据存储和检索的底层接口。具体的实现类将负责实际的持久化（或内存存储）和索引机制。
    *   **方法**: `addRecord(const DatasetMetadataEntry& entry)`, `updateRecord(const std::string& id, const DatasetMetadataEntry& entry)`, `deleteRecord(const std::string& id)`, `getRecord(const std::string& id) -> std::optional<DatasetMetadataEntry>`, `queryRecords(const MetadataQueryCriteria& criteria) -> std::vector<DatasetMetadataEntry>`.
    *   这允许未来根据需要更换后端存储（例如，从内存缓存升级到数据库）。

*   **`DatasetMetadataRegistry` (`dataset_metadata_registry.h/.cpp`)**:
    *   **职责**: 作为元数据注册和查询的主要逻辑处理单元。它使用 `IDatasetMetadataRegistryBackend` 进行数据存取。
    *   **主要方法**:
        *   `bool registerDataset(const DatasetMetadataEntry& datasetInfo);`
        *   `std::optional<DatasetMetadataEntry> getDatasetInfo(const std::string& datasetId);`
        *   `std::vector<DatasetMetadataEntry> findDatasets(const MetadataQueryCriteria& criteria);`
        *   `bool updateDatasetInfo(const std::string& datasetId, const DatasetMetadataEntry& updatedInfo);`
        *   `bool unregisterDataset(const std::string& datasetId);`

*   **`MetadataExtractor` (`metadata_extractor.h/.cpp`)**:
    *   **职责**: 负责从给定的文件路径（如NetCDF文件）中提取用于填充 `DatasetMetadataEntry` 的高层元数据。它可能会轻量级地使用 `data_access_service` 的能力来读取文件头或必要的全局属性，但避免加载大数据块。
    *   **主要方法**: `std::optional<DatasetMetadataEntry> extractMetadataFromFile(const std::string& filePath, const ExtractionOptions& options);`
    *   `std::vector<DatasetMetadataEntry> scanAndExtractMetadataFromDirectory(const std::string& directoryPath, bool recursive, const ExtractionOptions& options);`

*   **`MetadataQueryParser` (`metadata_query_parser.h/.cpp`)**:
    *   **职责**: (同之前方案) 解析用户输入的元数据查询请求（可能是结构化的对象，或简单的字符串）为内部的 `MetadataQueryCriteria` 对象，供 `DatasetMetadataRegistry` 使用。

*   **`MetadataServiceImpl` (`metadata_service_impl.h/.cpp`)**:
    *   **职责**: 实现对外暴露的 `IMetadataService` 接口。编排 `DatasetMetadataRegistry`、`MetadataExtractor` 和 `MetadataQueryParser` 等组件，提供统一的数据集注册、发现和管理服务。

## 4. 核心数据结构交互

*   `oscean::core_services::AttributeValue` (定义于 `common_data_types.h`): 可用于 `DatasetMetadataEntry` 中存储用户自定义的、类型灵活的元数据属性。
*   `oscean::core_services::BoundingBox`, `oscean::core_services::TimeRange` (假设已定义或将定义): 用于在 `DatasetMetadataEntry` 和查询条件中表示空间和时间范围。
*   `metadata_service` **不直接修改或深度依赖** `GridDefinition` 或 `DimensionCoordinateInfo` 的内部。它关注的是这些对象所代表的整个数据集的"外部"或"摘要"元数据。当 `MetadataExtractor` 工作时，它可能会读取一个临时的 `GridDefinition` 来获取全局属性或维度范围，但这些详细结构不会被 `metadata_service` 长期存储为其主要索引的一部分。

## 5. 预期目录结构 (示例)

```
OSCEAN/
├── metadata_refactoring_plan.md  <-- 本文件
├── core_service_interfaces/
│   └── include/core_services/
│       ├── common_data_types.h     (包含 AttributeValue, BoundingBox 等)
│       └── metadata/
│           ├── imetadata_service.h           # 服务接口定义 (可选,推荐)
│           ├── dataset_metadata_types.h    # 定义如 DatasetMetadataEntry, MetadataQueryCriteria, ExtractionOptions
│           └── idataset_metadata_registry_backend.h # 后端存储接口
└── core_services_impl/
    └── metadata_service/
        ├── include/core_services/metadata/impl/ # 内部头文件 (如果需要)
        ├── src/
        │   ├── metadata_service_impl.h/.cpp
        │   ├── dataset_metadata_registry.h/.cpp
        │   ├── metadata_extractor.h/.cpp
        │   ├── metadata_query_parser.h/.cpp
        │   └── backends/                       # 后端实现
        │       ├── in_memory_registry_backend.h/.cpp
        │       └── sqlite_registry_backend.h/.cpp (示例)
        └── tests/
            ├── dataset_metadata_registry_tests.cpp
            ├── metadata_extractor_tests.cpp
            └── metadata_service_integration_tests.cpp
```

## 6. 注意事项与挑战

*   **索引的持久化与规模**: 需要决定元数据索引是仅在内存中，还是需要持久化到文件或数据库（如SQLite）。这取决于数据集的数量和对服务重启后数据保留的需求。
*   **提取元数据的效率与准确性**: 从大量文件中自动提取元数据可能耗时。提取逻辑需要健壮，能处理各种文件格式的差异和不规范的元数据。
*   **查询语言的灵活性与复杂性**: 需要在查询的易用性和功能强大性之间找到平衡。
*   **更新与同步**: 当物理文件发生变化（添加、删除、修改）时，元数据索引需要有机制进行更新或重新同步。
*   **并发访问**: 如果服务被多个客户端并发访问，需要确保元数据注册表和索引的线程安全。

通过此方案，`metadata_service` 将演变成一个强大的数据发现和目录服务，能够有效地管理和查询跨文件和目录的数据集级元数据，从而极大地支持数据的查找和后续读取操作。 