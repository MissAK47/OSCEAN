# 空间服务单元测试套件

## 📋 测试概述

本测试套件专门针对空间服务的核心功能进行全面测试，基于真实的GEOS/GDAL库实现，不使用Mock对象。

### 🎯 测试原则

- ✅ **真实库集成**: 直接测试GEOS和GDAL库的实际功能
- ✅ **功能完整性**: 覆盖所有核心空间操作
- ✅ **边界条件**: 测试错误处理和极端情况
- ✅ **性能验证**: 包含基准测试验证性能表现
- ❌ **无Mock依赖**: 不使用Mock对象，测试真实功能

## 📊 测试模块

### 1. 几何引擎测试 (`test_geometry_engine.cpp`)
**目标**: 验证GEOS几何引擎的核心功能

- **缓冲区操作**: buffer()
- **拓扑运算**: intersection(), union(), difference()
- **空间谓词**: contains(), intersects(), disjoint()
- **距离计算**: calculateDistance()
- **几何处理**: simplify(), convexHull()
- **有效性检查**: isValid(), makeValid()

### 2. 栅格引擎测试 (`test_raster_engine.cpp`)
**目标**: 验证GDAL栅格引擎的核心功能

- **栅格裁剪**: clipRasterByGeometry(), clipRasterByBoundingBox()
- **要素栅格化**: rasterizeFeatures()
- **栅格掩膜**: applyRasterMask()
- **栅格变换**: 重投影和坐标转换

### 3. 空间工具测试 (`test_spatial_utils.cpp`)
**目标**: 验证空间计算和验证工具

- **几何验证**: isValidWKT(), isValidBoundingBox(), isValidPoint()
- **空间计算**: calculateHaversineDistance(), calculatePolygonArea()
- **边界框操作**: boundingBoxesIntersect(), expandBoundingBox()
- **栅格工具**: calculateRasterResolution(), calculateRasterBounds()

### 4. 几何转换测试 (`test_geometry_converter.cpp`)
**目标**: 验证几何格式转换功能

- **WKT转换**: pointToWKT(), boundingBoxToWKT(), parsePointFromWKT()
- **GeoJSON转换**: pointToGeoJSON(), boundingBoxToGeoJSON()
- **坐标转换**: transformCoordinates()

### 5. 坐标验证测试 (`test_coordinate_validator.cpp`)
**目标**: 验证坐标有效性检查

- **基础验证**: isValidPoint(), isValidBoundingBox()
- **地理坐标**: isValidLongitude(), isValidLatitude()
- **CRS验证**: isValidEPSGCode(), isValidCRS()

### 6. 空间索引测试 (`test_spatial_indexes.cpp`)
**目标**: 验证空间索引算法

- **四叉树索引**: QuadTreeIndex
- **R树索引**: RTreeIndex
- **网格索引**: GridIndex
- **索引操作**: build(), query(), insert(), remove()

### 7. 集成测试 (`test_integration.cpp`)
**目标**: 验证多模块协作工作流

- **端到端流程**: 完整的空间分析工作流
- **模块协作**: 多个引擎协同工作
- **性能集成**: 整体性能表现

## 🚀 运行测试

### 编译测试

```bash
# 编译所有测试
cmake --build build --target spatial_ops_service_tests

# 编译特定测试
cmake --build build --target test_geometry_engine
```

### 运行测试

```bash
# 运行快速核心测试
cmake --build build --target run_quick_tests

# 运行全量测试
cmake --build build --target run_all_spatial_tests

# 运行性能测试
cmake --build build --target run_performance_tests

# 使用CTest运行
cd build
ctest --output-on-failure

# 运行特定标签的测试
ctest -L "geometry" --output-on-failure
ctest -L "core" --output-on-failure
ctest -L "performance" --output-on-failure
```

### 调试测试

```bash
# 运行单个测试并查看详细输出
./test_geometry_engine --gtest_filter="GeometryEngineTest.BufferOperation_Point_Success"

# 运行测试并生成详细报告
./test_geometry_engine --gtest_output=xml:test_results.xml
```

## 📈 测试覆盖

### 功能覆盖率
- ✅ **几何引擎**: 100% 核心API覆盖
- ✅ **栅格引擎**: 100% 核心API覆盖
- ✅ **工具函数**: 90% 函数覆盖
- ✅ **索引算法**: 95% 算法覆盖

### 测试类型分布
- 🧪 **单元测试**: 85%
- 🔗 **集成测试**: 10%
- ⚡ **性能测试**: 5%

## 🛠️ 测试环境要求

### 必需依赖
- **Google Test**: >=1.10.0
- **GEOS**: >=3.8.0 (真实几何计算)
- **GDAL**: >=3.0.0 (真实栅格处理)
- **Boost**: >=1.70.0 (线程和系统支持)

### 推荐配置
- **C++标准**: C++17
- **编译器**: GCC 9+ 或 MSVC 2019+
- **内存**: 至少4GB（用于大数据测试）
- **磁盘**: 至少1GB（用于测试数据）

## 📊 性能基准

### 几何操作基准
- **缓冲区计算**: <1ms per 简单多边形
- **拓扑运算**: <5ms per 中等复杂度
- **空间查询**: <10ms per 1000要素

### 栅格操作基准
- **栅格裁剪**: <100ms per 1024x1024像素
- **要素栅格化**: <200ms per 1000要素
- **栅格掩膜**: <50ms per 1024x1024像素

## 🐛 故障排除

### 常见问题

1. **GEOS库未找到**
   ```
   解决方案: 确保vcpkg中安装了GEOS: vcpkg install geos
   ```

2. **GDAL库未找到**
   ```
   解决方案: 确保vcpkg中安装了GDAL: vcpkg install gdal
   ```

3. **测试数据缺失**
   ```
   解决方案: 确保test_data目录已正确复制到构建目录
   ```

### 调试技巧

1. **使用详细输出**
   ```bash
   ctest --verbose --output-on-failure
   ```

2. **单独运行失败测试**
   ```bash
   ./test_geometry_engine --gtest_filter="*FailingTest*"
   ```

3. **检查测试日志**
   ```bash
   cat Testing/Temporary/LastTest.log
   ```

## 🔄 持续集成

### 自动化测试流程
1. **编译验证**: 确保所有测试可以编译
2. **快速测试**: 运行核心功能测试(~5分钟)
3. **全量测试**: 运行完整测试套件(~15分钟)
4. **性能回归**: 检查性能基准是否下降

### 测试报告
- 测试结果自动生成XML格式报告
- 性能数据自动记录和比较
- 失败测试自动标记和跟踪

---

**📧 维护者**: OSCEAN空间服务团队  
**📅 最后更新**: 2024年6月  
**🔧 版本**: 1.0.0 