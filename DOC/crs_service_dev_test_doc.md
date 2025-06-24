# 坐标转换服务 (crs_service) 开发与测试文档
2025-5-11-0028
## 1. 概述

坐标转换服务 (`crs_service`) 是核心服务层 (Layer 3) 的关键组件之一，负责提供地理空间坐标参考系统 (Coordinate Reference System, CRS) 的定义、解析和点坐标转换功能。它封装了 GDAL/OGR 库的复杂性，并提供了一个带有缓存机制的高性能接口，供系统其他部分（如工作流引擎、空间处理服务等）使用。

主要目标：
*   提供基于 WKT、PROJ 字符串或 EPSG 代码的 CRS 之间的坐标转换。
*   缓存 `OGRCoordinateTransformation` 对象以提高重复转换的性能。
*   确保线程安全，支持并发访问。

## 2. 当前状态

*   **实现完成度**:
    *   `ICrsEngine` 接口已在 `core_service_interfaces` 中定义。
    *   `CrsEngineImpl` 作为接口的主要实现已完成，位于 `core_services_impl/crs_service/src/impl`。
    *   `TransformationCache` 类已实现，位于 `core_services_impl/crs_service/src/cache`，用于缓存坐标转换对象。
    *   依赖 `common_utils::GdalInit` 来确保 GDAL 初始化。
*   **测试覆盖**:
    *   针对 `TransformationCache` 的单元测试已编写，使用 Google Test 框架，测试固件为 `TransformationCacheTest`。
    *   测试用例覆盖了缓存的基本功能（获取、相同/不同键、清除、无效CRS处理）和并发访问。
*   **已知问题**:
    *   **严重**: 当前运行测试时，GDAL/PROJ 无法找到核心数据库文件 `proj.db`，导致所有需要创建有效 `OGRCoordinateTransformation` 的操作失败，`getTransformation` 返回 `nullptr`。错误信息为 `ERROR 1: PROJ: proj_create_from_database: Cannot find proj.db`。这使得大部分测试用例（除了处理无效 CRS 和并发空指针访问的测试）失败。
    *   `TransformationAccuracy_WGS84_To_Polar` 和 `TransformationAccuracy_Polar_To_WGS84` 两个精度测试用例暂时被搁置，需要等待 `proj.db` 问题解决后重新评估和修正。

## 3. 设计与实现

*   **接口**: `core_services::ICrsEngine` (位于 `core_service_interfaces`) 定义了服务的公共契约。
*   **实现**: `core_services::crs::CrsEngineImpl` 实现了 `ICrsEngine` 接口。它依赖 `TransformationCache` 来获取或创建坐标转换对象。
*   **缓存**: `core_services::crs::TransformationCache`
    *   使用 `std::map<std::string, std::shared_ptr<OGRCoordinateTransformation>>` 存储缓存项。
    *   键 (`std::string`) 由源 CRS 和目标 CRS 字符串拼接而成，用特殊分隔符隔开。
    *   使用 `std::shared_mutex` 保护缓存映射，允许多线程并发读取，单线程写入（在缓存未命中时创建并插入新对象）。
    *   提供 `getTransformation(sourceCRS, targetCRS)` 和 `clear()` 方法。
*   **核心逻辑**:
    *   使用 `OGRSpatialReference::SetFromUserInput()` 来解析输入的 CRS 字符串（如 "EPSG:4326"）。
    *   使用 `OGRCreateCoordinateTransformation()` 创建转换对象。
*   **依赖**:
    *   `core_service_interfaces` (获取接口定义和 `common_data_types`)
    *   `common_utils` (主要是 `GdalInit`)
    *   GDAL/OGR 库

## 4. 构建流程

`crs_service` 模块作为整个 `core_services_impl` 的一部分进行构建。

1.  确保 CMake 配置能够正确找到 GDAL 库和头文件。
2.  在项目根目录下的 `build` 目录中执行：
    ```bash
    # 1. 配置项目 (在 build 目录下)
    cmake .. 

    # 2. 构建项目 (包括 crs_service 及其测试)
    cmake --build . --config Debug 
    ```
3.  编译成功后，测试可执行文件将位于 `build/bin/Debug/crs_service_tests.exe` (或类似路径)。

## 5. 测试策略

*   **框架**: Google Test
*   **测试类型**: 单元测试 (当前主要针对 `TransformationCache`)
*   **测试固件**: `TransformationCacheTest` (位于 `tests/test_transformation_cache.cpp`)
    *   `SetUpTestSuite()`: 确保 GDAL 初始化。**注意**: 当前临时加入了 `OSRSetPROJSearchPaths` 来尝试解决 `proj.db` 问题，但这只是临时措施。
    *   `SetUp()`: 每个测试前清空缓存 `cache.clear()`。
*   **测试用例**:
    *   `GetTransformationReturnsValidPointerOnSuccess`: (当前失败) 测试有效 CRS 输入能否获得非空指针。
    *   `GetTransformationReturnsSamePointerForSameKey`: (当前失败) 测试相同输入是否返回缓存的相同指针。
    *   `GetTransformationReturnsDifferentPointersForDifferentKeys`: (当前失败) 测试不同输入是否通常返回不同指针。
    *   `GetTransformationReturnsNullptrForInvalidCRS`: (当前通过) 测试无效 CRS 输入是否返回空指针。
    *   `ClearRemovesCachedObjects`: (当前失败) 测试 `clear()` 后是否会创建新对象。
    *   `TransformationAccuracy_WGS84_To_Polar`: (当前失败, **暂缓处理**) 测试特定点从 WGS84 到极坐标转换的精度。
    *   `TransformationAccuracy_Polar_To_WGS84`: (当前失败, **暂缓处理**) 测试特定点从极坐标到 WGS84 转换的精度。
    *   `ConcurrentAccess`: (当前通过) 测试多线程并发访问缓存的基本安全性（主要是 `getTransformation` 调用）。

## 6. 运行测试

1.  确保项目已成功构建 (Debug 模式)。
2.  打开终端，导航到测试可执行文件所在的目录：
    ```powershell
    cd D:\OSCEAN\build\bin\Debug 
    ```
3.  运行测试：
    ```powershell
    .\crs_service_tests.exe
    ```
4.  观察输出，注意 `FAILED` 的测试用例和 `ERROR 1: PROJ: ... Cannot find proj.db` 错误信息。

## 7. 已知问题与故障排查

*   **核心问题**: `proj.db` 未找到。
    *   **根本原因**: 程序运行时无法定位 PROJ 库所需的数据文件。GDAL/PROJ 通常通过 `PROJ_LIB` 环境变量查找此文件。
    *   **排查步骤**:
        1.  **确认 `PROJ_LIB` 环境变量**:
            *   检查系统环境变量或用户环境变量是否设置了 `PROJ_LIB`。
            *   **关键**: 确认运行 `crs_service_tests.exe` 的**确切环境**（例如，直接从 PowerShell 运行 vs 从 Visual Studio 内部运行）是否继承或设置了正确的 `PROJ_LIB` 值。在运行测试的终端执行 `echo $env:PROJ_LIB` (PowerShell) 或 `echo %PROJ_LIB%` (CMD)。
            *   **推荐**: 优先使用系统级环境变量或在 CMake 构建时配置 GDAL 的数据路径，而不是依赖临时会话设置。
        2.  **验证路径内容**: 检查 `PROJ_LIB` 指向的目录下是否确实存在 `proj.db` 文件。
        3.  **检查 GDAL/PROJ 安装**: 确保 GDAL 及其依赖（特别是 PROJ）安装完整，并且数据文件 (`share/proj` 目录下的内容) 存在且未损坏。
        4.  **临时代码修复检查**: 确认 `test_transformation_cache.cpp` 中 `SetUpTestSuite` 内 `OSRSetPROJSearchPaths` 设置的路径 (`D:/OSCEAN/3rdParty/gdal/share/proj`) **绝对正确**。这只是临时方案，最终应通过环境配置解决。
        5.  **IDE 环境**: 如果从 IDE (如 Visual Studio) 运行，检查其项目设置或启动配置中是否正确传递了环境变量或配置了调试环境。

## 8. 后续工作与计划

1.  **最高优先级**: 彻底解决 `proj.db` 加载问题，确保 GDAL/PROJ 运行时能稳定找到其数据文件（最好是通过环境配置而非代码写死路径）。
2.  **精度测试**: 待 `proj.db` 问题解决后，重新审视 `TransformationAccuracy_...` 测试用例，确认预期值是否正确，调整容差 (`EXPECT_NEAR` 的第三个参数)，确保测试能在合理精度内通过。
3.  **补充测试**:
    *   为 `CrsEngineImpl` 本身编写单元测试，可能需要 Mock `TransformationCache` 或直接测试其与真实缓存的交互。
    *   测试使用不同格式 CRS 输入（WKT, PROJ String, EPSG Code）的行为。
4.  **集成测试**: 开发集成测试，模拟工作流引擎或其他服务调用 `ICrsEngine` 接口的场景，例如，结合数据访问服务读取带有 CRS 的数据，然后进行转换。
5.  **功能完善**: 根据后续需求，实现 `transformGridAsync` 等可能需要的接口。

## 9. 依赖项

*   C++17 标准库
*   CMake (构建系统)
*   GDAL/OGR (核心库 >= 3.x 版本)
*   Boost (可能通过其他模块间接依赖，如 Asio 用于线程池)
*   spdlog (通过 `common_utils` 依赖，用于日志)
*   Google Test (测试框架)
*   `core_service_interfaces` (项目内部模块)
*   `common_utils` (项目内部模块)

## 坐标转换

坐标参考系统（Coordinate Reference System，简称CRS）是空间数据表示的基础。

CRS服务需要能够在不同坐标系统之间转换点、线、多边形等空间要素。

## 依赖库分析

GDAL（Geospatial Data Abstraction Library）是一个开源的地理空间数据处理库，我们主要使用其子库OGR处理矢量数据和坐标转换。

GDAL依赖PROJ库进行坐标系统定义和转换。

## 已知问题与解决方案

### 1. `proj.db`路径问题

**症状**：测试过程中出现错误 `ERROR 1: PROJ: proj_create_from_database: Cannot find proj.db`

**原因**：PROJ 7.0及以上版本依赖proj.db数据库文件来进行坐标系统定义和转换操作。测试时无法找到这个文件。

**解决方案**：

1. 测试初始化时设置PROJ_LIB环境变量或使用API设置搜索路径
   ```cpp
   // 在测试前设置PROJ库搜索路径
   const char* projSearchPaths[] = { "C:/Users/flyfox/vcpkg/installed/x64-windows/share/proj", nullptr };
   OSRSetPROJSearchPaths(projSearchPaths);

   // 设置GDAL_DATA路径
   CPLSetConfigOption("GDAL_DATA", "C:/Users/flyfox/vcpkg/installed/x64-windows/share/gdal");

   // 使用环境变量方式设置（双保险）
   #ifdef _WIN32
   _putenv_s("PROJ_LIB", "C:/Users/flyfox/vcpkg/installed/x64-windows/share/proj");
   _putenv_s("GDAL_DATA", "C:/Users/flyfox/vcpkg/installed/x64-windows/share/gdal");
   #else
   setenv("PROJ_LIB", "C:/Users/flyfox/vcpkg/installed/x64-windows/share/proj", 1);
   setenv("GDAL_DATA", "C:/Users/flyfox/vcpkg/installed/x64-windows/share/gdal", 1);
   #endif
   ```

2. 在实际部署时，应当提供统一的环境配置，确保所有服务能找到正确的PROJ数据文件。可以考虑以下方案：
   - 使用安装脚本自动设置环境变量
   - 在服务启动脚本中设置环境变量
   - 编程方式指定PROJ搜索路径
   - 将proj.db数据文件与应用程序打包在一起，并在运行时配置PROJ搜索路径指向此位置 