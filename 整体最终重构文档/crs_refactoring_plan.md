# CRS 服务 (`crs_service`) 重构方案

## 1. 核心思想

新的 `crs_service` 应专注于坐标参考系统相关的**操作和转换**，而不是简单地存储现在已部分集成到 `GridDefinition` 中的CRS信息。它将成为一个专业的CRS处理工具集，为OSCEAN项目提供统一、准确的地理空间坐标处理能力。

## 2. 主要职责

*   **CRS定义解析与验证**:
    *   能够解析多种格式的CRS定义，包括但不限于EPSG代码、WKT (Well-Known Text) 字符串、Proj (PROJ string) 字符串。
    *   验证解析后的CRS定义的有效性和完整性。
    *   将不同的输入格式转换为内部统一的 `CRSInfo` 对象表示。
*   **坐标转换**:
    *   在不同的坐标参考系统之间准确转换单个点、点序列。
    *   支持对 `BoundingBox`（边界框）对象的整体坐标转换。
    *   应考虑三维坐标的转换（如果Z坐标也与CRS相关）。
*   **CRS信息查询与提取**:
    *   从给定的 `CRSInfo` 对象或CRS定义中提取详细参数，如基准面 (Datum)、椭球体 (Ellipsoid)、投影方法 (Projection Method)、中央经线、标准纬线、单位 (Units) 等。
*   **CRS比较与等效性判断**:
    *   提供功能来判断两个不同的CRS定义（即使表示方式不同，如EPSG代码与WKT字符串）是否在地理空间上等效。
*   **地理计算 (可选扩展)**:
    *   提供基于特定CRS的地理计算功能，如两点间的距离、方位角计算等。

## 3. 建议组件/类

*   **`CRSParser` (`crs_parser.h/.cpp`)**:
    *   **职责**: 负责解析各种格式的CRS输入，并将其转换为标准化的 `CRSInfo` 对象。封装CRS定义的语法解析和初步验证逻辑。
    *   **主要方法**:
        *   `std::optional<CRSInfo> parseFromWKT(const std::string& wktString);`
        *   `std::optional<CRSInfo> parseFromProjString(const std::string& projString);`
        *   `std::optional<CRSInfo> parseFromEpsgCode(int epsgCode);`
        *   `bool isValid(const CRSInfo& crsInfo);`
    *   **依赖**: 可能需要链接到PROJ库或其他GIS库进行实际的解析和验证。

*   **`CRSTransformer` (`crs_transformer.h/.cpp`)**:
    *   **职责**: 实现不同 `CRSInfo` 对象所定义的坐标参考系统之间的坐标点、坐标序列和 `BoundingBox` 的转换。这是服务的核心计算组件。
    *   **主要方法**:
        *   `TransformedPoint transformPoint(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS);`
        *   `TransformedPoint transformPoint(double x, double y, double z, const CRSInfo& sourceCRS, const CRSInfo& targetCRS);` (支持3D)
        *   `std::vector<TransformedPoint> transformPoints(const std::vector<Point>& points, const CRSInfo& sourceCRS, const CRSInfo& targetCRS);`
        *   `BoundingBox transformBoundingBox(const BoundingBox& sourceBbox, const CRSInfo& targetCRS);` (假设`sourceBbox`自身包含其源CRS信息或通过参数传入)
    *   **依赖**: 强烈依赖PROJ库或类似库进行精确的坐标转换计算。

*   **`CRSInspector` (或 `CRSQueryEngine`) (`crs_inspector.h/.cpp`)**:
    *   **职责**: 提供从 `CRSInfo` 对象中查询和提取详细CRS参数的功能。
    *   **主要方法**:
        *   `CRSDetailedParameters getDetailedParameters(const CRSInfo& crsInfo);`
        *   `std::string getUnit(const CRSInfo& crsInfo);`
        *   `std::string getProjectionMethod(const CRSInfo& crsInfo);`
        *   `bool areEquivalent(const CRSInfo& crsInfo1, const CRSInfo& crsInfo2);`
    *   **依赖**: 可能需要查询PROJ库的内部数据库或解析WKT来获取这些信息。

*   **`CRSServiceImpl` (`crs_service_impl.h/.cpp`)**:
    *   **职责**: 作为 `crs_service` 的主实现类，实现对外暴露的 `ICRSService` 接口（如果定义）。它将编排 `CRSParser`、`CRSTransformer` 和 `CRSInspector` 等组件的功能，为上层应用提供统一和简化的服务调用。
    *   它不直接存储CRS定义状态，而是操作调用时传入的 `CRSInfo` 对象或CRS定义字符串/代码。

## 4. 核心数据结构交互

*   `oscean::core_services::CRSInfo` (定义于 `common_data_types.h`): 这是CRS信息在系统内部流转的核心结构。`crs_service` 的所有组件都将围绕它进行操作（接收、处理、返回）。
    *   `CRSInfo` 应该设计得足够灵活，能够存储如WKT字符串、Proj字符串或EPSG代码等原始定义，并可能包含一个解析后的标准化表示。
*   `oscean::core_services::BoundingBox` (定义于 `common_data_types.h`): `CRSTransformer` 会操作此结构进行边界框的整体转换。
*   自定义类型如 `TransformedPoint`, `CRSDetailedParameters` 需要在 `crs_service` 的接口头文件中定义。

## 5. 预期目录结构 (示例)

```
OSCEAN/
├── crs_refactoring_plan.md  <-- 本文件
├── core_service_interfaces/
│   └── include/core_services/
│       ├── common_data_types.h  (包含 CRSInfo, BoundingBox 定义)
│       └── crs/
│           ├── icrs_service.h              # 服务接口定义 (可选,推荐)
│           └── crs_operation_types.h       # 定义如 TransformedPoint, CRSDetailedParameters 等
└── core_services_impl/
    └── crs_service/
        ├── include/core_services/crs/impl/ # 内部头文件 (如果需要)
        ├── src/                            # 实现文件
        │   ├── crs_service_impl.h/.cpp
        │   ├── crs_parser.h/.cpp
        │   ├── crs_transformer.h/.cpp
        │   └── crs_inspector.h/.cpp        # (替代之前的 crs_utils.h/.cpp)
        └── tests/                          # 单元测试和集成测试
            ├── crs_parser_tests.cpp
            ├── crs_transformer_tests.cpp
            └── crs_inspector_tests.cpp
```

## 6. 外部依赖

*   **PROJ**: 这是一个强大且广泛使用的地理坐标转换库。强烈建议 `crs_service` 的核心转换和解析功能基于PROJ库进行封装。CMake或项目的构建系统需要正确配置PROJ库的链接。

## 7. 注意事项与挑战

*   **精度与性能**: 坐标转换可能涉及复杂的数学运算。需要确保转换的精度，并关注高频调用场景下的性能。
*   **错误处理**: 对于无效的CRS定义、不支持的转换或PROJ库返回的错误，需要设计健壮的错误处理和报告机制。
*   **PROJ版本兼容性**: PROJ库的API在不同版本间可能有变化，需要注意使用的PROJ版本并确保API调用的兼容性。
*   **线程安全**: 如果 `crs_service` 的实例会在多线程环境中使用，需要确保其内部组件（尤其是与PROJ库交互的部分）是线程安全的，或者在服务层面进行适当的同步控制。

通过这样的重构，`crs_service` 将成为一个功能内聚、职责清晰、易于维护和扩展的专业模块，为整个OSCEAN项目提供可靠的坐标参考系统服务。 