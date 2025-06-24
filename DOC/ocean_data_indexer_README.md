# 🌊 OSCEAN 海洋数据索引测试程序

## 📖 程序概述

这是一个独立的海洋数据索引测试程序，用于扫描 `E:\Ocean_data` 目录下的海洋数据文件并自动建立元数据索引数据库。

### 🎯 主要功能

- **📁 自动文件扫描**: 递归扫描指定目录下的所有数据文件
- **🔍 格式自动识别**: 支持 NetCDF、GeoTIFF、HDF、GRIB、Shapefile 等格式
- **⚡ 元数据提取**: 自动提取时间、空间、坐标、变量等元数据信息  
- **💾 数据库索引**: 建立SQLite索引数据库，支持快速查询
- **📊 实时进度显示**: 显示处理进度和详细统计信息
- **🔍 查询验证**: 自动验证索引结果并执行测试查询

### 📋 支持的文件格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| NetCDF | `.nc`, `.nc4`, `.netcdf` | 网络通用数据格式 |
| GeoTIFF | `.tif`, `.tiff`, `.geotiff` | 地理标记TIFF |
| HDF | `.hdf`, `.h5`, `.hdf5` | 分层数据格式 |
| GRIB | `.grib`, `.grb`, `.grib2` | 气象数据格式 |
| Shapefile | `.shp` | 矢量数据格式 |

## 🛠️ 编译和运行

### 前置要求

- **编译器**: 支持 C++20 的编译器 (Visual Studio 2019+ 或 GCC 10+)
- **CMake**: 版本 3.20 或更高
- **依赖库**: 
  - SQLite3 (通过 vcpkg 或系统安装)
  - Threads (标准库)

### 编译步骤

#### 方法1: 使用提供的CMakeLists.txt

```bash
# 1. 创建构建目录
mkdir build_indexer_test
cd build_indexer_test

# 2. 配置项目 (如果使用vcpkg)
cmake -S .. -B . -DCMAKE_TOOLCHAIN_FILE=<vcpkg_path>/scripts/buildsystems/vcpkg.cmake

# 3. 编译
cmake --build . --config Release

# 4. 运行
./bin/ocean_data_indexer_test
```

#### 方法2: 直接编译 (简化版)

```bash
# 确保有必要的包含路径和链接库
g++ -std=c++20 -O2 \
    -I core_service_interfaces/include \
    -I core_services_impl/metadata_service/include \
    ocean_data_indexer_test.cpp \
    -lsqlite3 -lpthread \
    -o ocean_data_indexer_test
```

### 运行程序

#### 基本运行

```bash
# 使用默认路径 E:\Ocean_data
./ocean_data_indexer_test

# 指定自定义数据路径
./ocean_data_indexer_test "D:\MyOceanData"

# 指定数据路径和数据库路径
./ocean_data_indexer_test "D:\MyOceanData" "my_ocean_index.db"
```

#### Windows 批处理文件

```batch
REM 编译后会自动生成以下批处理文件:
build_test.bat   - 编译程序
run_test.bat     - 运行程序
```

## 📊 程序输出

### 典型运行输出示例

```
🌊 OSCEAN 海洋数据索引测试程序
版本: 1.0.0
功能: 扫描海洋数据文件并建立元数据索引数据库

📋 测试参数:
  数据目录: E:\Ocean_data
  索引数据库: ocean_metadata_index.db

是否开始测试？(y/n): y

=== 海洋数据索引测试器 ===
数据目录: E:\Ocean_data
数据库路径: ocean_metadata_index.db

开始执行完整的海洋数据索引测试...

📁 第1步：扫描数据文件...
📊 扫描结果:
  - 总文件数: 156
  - 支持的数据文件: 43
  - 文件类型分布:
    .nc: 38 个
    .tif: 3 个
    .h5: 2 个

🔄 第2步：批量索引数据文件...
[数据索引] 开始处理 43 个项目...
[████████████████████████████████████████████████████] 100.0% (43/43)
[数据索引] 完成! 耗时: 125 秒

📊 索引结果:
  - 成功索引: 41 个文件
  - 索引失败: 2 个文件

✅ 第3步：验证索引结果...
📊 数据库验证结果:
  - 数据库中的数据集总数: 41
  - 总变量数: 287
  - 唯一变量名: 45
  - 坐标系统: 3 种
  - 变量示例: temperature, salinity, u, v, ssh...

🔍 第4步：执行查询测试...
  测试1: 按文件路径模式查询...
    - NetCDF文件: 38 个数据集
  测试2: 按变量名查询...
    - 变量 'temperature': 15 个数据集
    - 变量 'salinity': 12 个数据集
    - 变量 'u': 8 个数据集
    - 变量 'v': 8 个数据集
  测试3: 空间范围查询...
    - 空间范围内: 23 个数据集

============================================================
📊 海洋数据索引测试 - 完整统计报告
============================================================

📁 文件扫描统计:
  总文件数: 156
  支持的数据文件: 43
  成功索引文件: 41
  失败文件: 2
  成功率: 95.3%

📋 文件类型分布:
        .nc:     38 个
       .tif:      3 个
        .h5:      2 个

📊 数据内容统计:
  总变量数: 287
  时间步数: 41

⏱️  性能统计:
  总处理时间: 127.45 秒
  元数据提取时间: 125.23 秒
  平均每文件处理时间: 3.054 秒

💾 数据库信息:
  数据库路径: ocean_metadata_index.db
  数据库大小: 2.3 MB

============================================================

✅ 所有测试完成，索引建立成功！

🎉 测试成功完成！
索引数据库已创建: ocean_metadata_index.db
您现在可以使用该数据库进行海洋数据查询。
```

## 🗃️ 生成的数据库

### 数据库结构

程序会创建一个SQLite数据库，包含以下主要表：

- **datasets**: 数据集基本信息 (文件路径、大小、修改时间等)
- **variables**: 变量信息 (名称、类型、维度、单位等)  
- **spatial_info**: 空间范围信息 (经纬度范围、坐标系统等)
- **temporal_info**: 时间范围信息 (开始/结束时间、时间步长等)
- **metadata_cache**: 元数据缓存表

### 数据库查询示例

```sql
-- 查询所有包含温度变量的数据集
SELECT DISTINCT d.file_path, d.dataset_name 
FROM datasets d 
JOIN variables v ON d.id = v.dataset_id 
WHERE v.variable_name LIKE '%temp%' OR v.variable_name LIKE '%sst%';

-- 查询指定空间范围内的数据集  
SELECT d.file_path, s.min_longitude, s.max_longitude, s.min_latitude, s.max_latitude
FROM datasets d
JOIN spatial_info s ON d.id = s.dataset_id
WHERE s.min_longitude >= 100 AND s.max_longitude <= 150 
  AND s.min_latitude >= 0 AND s.max_latitude <= 50;

-- 查询指定时间范围内的数据集
SELECT d.file_path, t.start_time, t.end_time
FROM datasets d
JOIN temporal_info t ON d.id = t.dataset_id  
WHERE t.start_time >= '2023-01-01' AND t.end_time <= '2023-12-31';
```

## 🔧 故障排除

### 常见问题

1. **编译错误: 找不到SQLite3**
   ```bash
   # 使用vcpkg安装
   vcpkg install sqlite3
   
   # 或在Ubuntu/Debian上
   sudo apt-get install libsqlite3-dev
   
   # 或在CentOS/RHEL上  
   sudo yum install sqlite-devel
   ```

2. **运行时错误: 无法打开数据目录**
   - 检查路径是否正确
   - 确保有读取权限
   - 使用绝对路径

3. **索引失败率过高**
   - 检查文件是否损坏
   - 确认文件格式是否受支持
   - 查看错误日志了解具体原因

4. **中文编码问题 (Windows)**
   - 程序已配置UTF-8支持
   - 如仍有问题，检查系统区域设置

### 性能优化建议

- **并行处理**: 可修改代码启用多线程处理
- **内存限制**: 大型文件可能需要更多内存
- **磁盘空间**: 确保有足够空间存储索引数据库
- **网络驱动器**: 避免在网络驱动器上运行，会影响性能

## 📞 技术支持

如果遇到问题，请提供以下信息：

1. 操作系统版本
2. 编译器版本
3. 数据文件类型和大小
4. 完整的错误信息
5. 程序运行日志

## 🔄 版本历史

- **v1.0.0**: 初始版本，支持基本的文件扫描和索引功能

---

*该程序是OSCEAN项目metadata_service模块的独立测试工具，用于验证元数据索引功能的完整性和性能。* 