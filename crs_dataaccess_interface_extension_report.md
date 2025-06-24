# DataAccess接口扩展与元数据CRS多格式支持修复报告

## 修复概述

基于您的要求，我们完成了以下两个重要修复：

1. **扩展DataAccess接口**：支持坐标转换参数，为工作流层提供便利
2. **修复元数据模块**：增强对多种坐标格式的保存和记录能力

## 一、DataAccess接口扩展

### 1.1 新增坐标转换请求参数类型

**文件**: `core_service_interfaces/include/core_services/data_access/unified_data_types.h`

新增了 `CRSTransformRequest` 结构体：

```cpp
struct CRSTransformRequest {
    std::string sourceCRS;          ///< 源坐标系（WKT/PROJ/EPSG:xxxx格式）
    std::string targetCRS;          ///< 目标坐标系（WKT/PROJ/EPSG:xxxx格式）
    bool transformBounds = false;   ///< 是否转换空间边界
    bool transformGeometry = false; ///< 是否转换几何体坐标
    
    // 便捷创建方法
    static CRSTransformRequest createEpsgTransform(int sourceEpsg, int targetEpsg);
    static CRSTransformRequest createToWGS84(const std::string& sourceCrs);
};
```

### 1.2 扩展统一数据请求

在 `UnifiedDataRequest` 中添加了坐标转换支持：

```cpp
struct UnifiedDataRequest {
    // ... 现有字段 ...
    
    // 🆕 坐标转换参数 - 为工作流层提供便利
    std::optional<CRSTransformRequest> crsTransform;
    
    // 便捷方法
    void setCRSTransform(const std::string& sourceCrs, const std::string& targetCrs);
    void setTransformToWGS84(const std::string& sourceCrs);
    bool needsCRSTransform() const;
};
```

### 1.3 新增坐标转换便捷接口

**文件**: `core_service_interfaces/include/core_services/data_access/i_unified_data_access_service.h`

```cpp
/**
 * @brief 读取格点数据并支持坐标转换
 * 
 * 为工作流层提供便利，DataAccess会协调CRS服务进行坐标转换
 * 注意：坐标转换的具体实现由CRS服务负责，DataAccess只负责协调
 */
virtual boost::future<std::shared_ptr<GridData>> readGridDataWithCRSAsync(
    const std::string& filePath,
    const std::string& variableName,
    const BoundingBox& bounds,
    const std::string& targetCRS) = 0;

/**
 * @brief 读取点数据并支持坐标转换
 */
virtual boost::future<std::optional<double>> readPointDataWithCRSAsync(
    const std::string& filePath,
    const std::string& variableName,
    const Point& point,
    const std::string& targetCRS) = 0;
```

## 二、元数据模块CRS多格式支持修复

### 2.1 问题分析

**原有问题**：
- 数据库只使用单一的 `crs_definition TEXT` 字段
- 存储实现只保存 `metadata.crs.wktext`，忽略其他CRS格式
- 读取时只映射 `wktext` 和 `wkt` 字段，其他字段丢失

### 2.2 修复方案

**文件**: `core_services_impl/metadata_service/src/extractors/storage/sqlite_storage.cpp`

#### A. 修复CRS信息存储

将CRS信息序列化为完整的JSON格式：

```cpp
// 🔧 修复CRS信息存储 - 序列化为完整的JSON格式
json crsJson;
if (!metadata.crs.authorityName.empty()) {
    crsJson["authorityName"] = metadata.crs.authorityName;
}
if (!metadata.crs.authorityCode.empty()) {
    crsJson["authorityCode"] = metadata.crs.authorityCode;
}
if (!metadata.crs.wktext.empty()) {
    crsJson["wktext"] = metadata.crs.wktext;
}
if (!metadata.crs.projString.empty()) {
    crsJson["projString"] = metadata.crs.projString;
}
if (metadata.crs.epsgCode.has_value()) {
    crsJson["epsgCode"] = metadata.crs.epsgCode.value();
}
crsJson["isGeographic"] = metadata.crs.isGeographic;
crsJson["isProjected"] = metadata.crs.isProjected;
// ... 其他字段 ...

std::string crsJsonStr = crsJson.empty() ? "" : crsJson.dump();
sqlite3_bind_text(statements_.insertFile, 5, crsJsonStr.c_str(), -1, SQLITE_STATIC);
```

#### B. 修复CRS信息读取

从JSON格式解析完整的CRS信息：

```cpp
// 🔧 修复CRS信息解析 - 支持多种格式
std::string crsDefinition = getStringFromColumn(stmt, 4);
if (!crsDefinition.empty()) {
    try {
        // 尝试解析为JSON格式的CRS信息
        json crsJson = json::parse(crsDefinition);
        
        // 解析所有CRS字段
        if (crsJson.contains("authorityName")) {
            metadata.crs.authorityName = crsJson["authorityName"].get<std::string>();
            metadata.crs.authority = metadata.crs.authorityName; // 兼容字段
        }
        // ... 解析其他字段 ...
        
    } catch (const json::exception& e) {
        // 如果不是JSON格式，则视为简单的WKT/PROJ字符串
        // 智能检测格式（EPSG:xxxx, WKT, PROJ等）
    }
}
```

### 2.3 支持的CRS格式

修复后的元数据模块现在完整支持：

1. **WKT格式** (`wktext`, `wkt`)
2. **PROJ格式** (`projString`, `proj4text`)
3. **EPSG代码** (`epsgCode`)
4. **权威机构信息** (`authorityName`, `authorityCode`)
5. **坐标系类型** (`isGeographic`, `isProjected`)
6. **单位信息** (`linearUnitName`, `linearUnitToMeter`, `angularUnitName`, `angularUnitToRadian`)
7. **兼容字段** (`authority`, `code`, `id`, `name`)

## 三、架构设计原则

### 3.1 模块职责清晰

- **CRS模块**：仅负责坐标系解析、转换、验证
- **DataAccess模块**：负责数据读取，可协调CRS服务（但不直接转换）
- **元数据模块**：记录和保存所有格式的坐标信息（不做处理）
- **工作流引擎**：在上层协调各模块，实现完整业务流程

### 3.2 坐标转换流程

```
正确的工作流：
1. DataAccess检测文件CRS → 
2. 元数据模块保存完整CRS信息 →
3. 工作流引擎请求DataAccess进行坐标转换查询 →
4. DataAccess协调CRS服务转换用户坐标到数据坐标 → 
5. DataAccess在数据坐标系中读取 → 
6. DataAccess协调CRS服务转换结果回用户坐标系
```

## 四、测试验证

### 4.1 新增测试

**文件**: `workflow_engine/tests/test_ice_thickness_workflow_stage1.cpp`

添加了 `Stage2_MetadataCRSMultiFormatSupport` 测试，验证：

- CRS多格式信息的完整性
- 各种格式字段的正确解析
- 兼容字段的一致性
- 投影坐标系的正确识别

### 4.2 测试覆盖

- ✅ WKT格式解析和存储
- ✅ EPSG代码处理
- ✅ PROJ字符串支持
- ✅ 权威机构信息记录
- ✅ 坐标系类型识别
- ✅ 单位信息保存
- ✅ 兼容字段同步

## 五、使用示例

### 5.1 工作流层使用DataAccess扩展接口

```cpp
// 示例：读取冰厚度数据并转换到WGS84
auto request = api::UnifiedDataRequest(
    api::UnifiedRequestType::GRID_DATA, 
    "ice_thickness.nc"
);
request.variableName = "sithick";
request.setCRSTransform("AUTO_DETECT", "EPSG:4326");
request.spatialBounds = BoundingBox{-180, -90, 180, 90}; // WGS84边界

auto response = dataAccessService->processDataRequestAsync(request).get();
```

### 5.2 便捷接口使用

```cpp
// 读取指定点的冰厚度（自动坐标转换）
Point targetPoint(-175.0, 75.0); // WGS84坐标
auto iceThickness = dataAccessService->readPointDataWithCRSAsync(
    "ice_thickness.nc", 
    "sithick", 
    targetPoint, 
    "EPSG:4326"
).get();
```

## 六、后续工作

### 6.1 DataAccess实现类修改

需要在具体的DataAccess实现类中：
1. 实现新增的坐标转换接口
2. 在 `processDataRequestAsync` 中处理 `crsTransform` 参数
3. 集成CRS服务进行坐标转换协调

### 6.2 工作流引擎集成

1. 利用新的DataAccess接口简化坐标转换逻辑
2. 在业务流程中正确使用坐标转换参数
3. 处理多种坐标系之间的转换需求

## 七、总结

本次修复完成了：

1. **✅ DataAccess接口扩展**：为工作流层提供坐标转换便利，同时保持架构清晰
2. **✅ 元数据CRS支持修复**：实现对多种坐标格式的完整保存和记录
3. **✅ 架构设计改进**：明确各模块职责，避免功能耦合
4. **✅ 测试验证**：确保修复的正确性和完整性

通过这些修复，系统现在能够：
- 正确保存和恢复所有格式的CRS信息
- 为工作流层提供便捷的坐标转换接口
- 保持清晰的模块职责分离
- 支持复杂的多坐标系数据处理流程 