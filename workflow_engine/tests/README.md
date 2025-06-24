# Workflow Engine 测试

## 测试目录组织原则

### ✅ 正确做法

1. **测试文件放在 `tests/` 目录**
   - 所有测试文件都应该在适当的测试目录中
   - 遵循标准的项目结构约定

2. **基于真实API编写测试**
   - 测试代码**必须**基于实际存在的接口和数据结构
   - **禁止**假设或虚构API
   - 在编写测试前**必须**先读取和理解实际的代码接口

3. **一个测试文件，持续改进**
   - 修正现有文件而不是创建新文件
   - 避免重复和混乱

### ❌ 错误做法（之前的问题）

1. **测试文件散落在根目录**
   - ~~`workflow_engine/test_ocean_data_processing_real.cpp`~~ ❌
   - ~~`workflow_engine/test_e2e_ocean_data_processing.cpp`~~ ❌

2. **基于假设的API编写代码**
   - 假设不存在的工厂函数
   - 使用不存在的方法调用
   - 导致大量编译错误

3. **重复创建测试文件**
   - 创建多个类似功能的文件
   - 造成维护负担和混乱

## 当前测试文件

### `test_real_ocean_data_processing.cpp`

**功能：**
- 基于真实的 `core_services` API
- 处理 `E:\Ocean_data` 目录中的实际数据文件
- 使用真实的元数据服务进行文件识别和存储
- 提供基础的查询功能验证

**使用的真实API：**
- `metadata::IUnifiedMetadataService::create()` - 实际的工厂函数
- `metadataService_->recognizeFileAsync()` - 真实的文件识别API
- `metadataService_->storeMetadataAsync()` - 真实的存储API
- `metadataService_->getStatisticsAsync()` - 真实的查询API

**编译要求：**
- C++17
- Boost libraries (thread, chrono, system)
- OSCEAN core services
- 所有依赖项必须正确构建

## 运行测试

### 编译
```bash
cd build
cmake --build . --target test_real_ocean_data_processing
```

### 运行
```bash
cd build/bin
./test_real_ocean_data_processing
```

### 前提条件
1. `E:\Ocean_data` 目录存在且包含海洋数据文件
2. 支持的文件格式：`.nc`, `.nc4`, `.netcdf`, `.tif`, `.tiff`, `.geotiff`, `.shp`, `.json`, `.geojson`, `.grd`
3. 所有核心服务正确构建和链接

## 测试功能

### 核心功能测试
- [x] 服务初始化
- [x] 目录扫描和文件发现
- [x] 文件类型识别
- [x] 元数据提取
- [x] 数据库存储
- [x] 基础查询验证

### 输出示例
```
=== 基于真实API的海洋数据处理测试程序 ===
版本: 1.0
目标: E:\Ocean_data 目录处理
=========================================
正在初始化服务...
✓ 初始化成功
发现 15 个数据文件
[1/15] 处理: ocean_temp.nc
  ✓ 成功
[2/15] 处理: bathymetry.tif
  ✓ 成功
...
进度: 15/15 (成功: 14, 失败: 1)

=== 处理完成 ===
总文件数: 15
成功处理: 14
处理失败: 1
成功率: 93.3%

=== 元数据查询测试 ===
✓ 查询成功
  总数据集: 14
  平均质量: 85.6%
  数据类型 1: 8 个
  数据类型 2: 6 个
```

## 后续改进

1. **添加更多测试场景**
   - 异常情况处理
   - 性能基准测试
   - 并发处理测试

2. **完善错误处理**
   - 网络连接异常
   - 文件损坏情况
   - 内存不足情况

3. **增加断言验证**
   - 元数据完整性检查
   - 数据质量验证
   - 时空范围准确性

## 经验教训

> **核心教训：** 永远不要基于假设编写代码。必须先理解实际的API和数据结构，然后编写测试。

1. **API理解优先**：在编写任何代码前，必须先阅读和理解实际的接口定义
2. **避免重复文件**：修正现有代码而不是创建新文件
3. **正确的目录结构**：遵循标准的项目组织约定
4. **真实数据测试**：使用实际的数据和服务，避免mock和简化 