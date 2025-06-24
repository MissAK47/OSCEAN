# OSCEAN项目vcpkg配置说明

## 概述

OSCEAN项目使用vcpkg作为C++包管理器来管理第三方依赖。本文档说明如何正确配置和使用vcpkg。

## 前置要求

- Visual Studio 2022 (推荐)
- CMake 3.20或更高版本
- PowerShell 7.x
- Git

## vcpkg安装和配置

### 1. 安装vcpkg

```powershell
# 克隆vcpkg仓库
git clone https://github.com/Microsoft/vcpkg.git C:\Users\flyfox\vcpkg

# 进入vcpkg目录
cd C:\Users\flyfox\vcpkg

# 运行bootstrap脚本
.\bootstrap-vcpkg.bat

# 集成到Visual Studio (可选)
.\vcpkg integrate install
```

### 2. 安装项目依赖

项目依赖已在`vcpkg.json`文件中定义。vcpkg会自动安装这些依赖：

```json
{
  "dependencies": [
    "gdal",
    "boost-system",
    "boost-thread", 
    "boost-filesystem",
    "boost-date-time",
    "gtest",
    "sqlite3",
    "netcdf-c",
    "spdlog",
    "nlohmann-json",
    "eigen3",
    "xxhash",
    "tiff",
    "proj"
  ]
}
```

### 3. 手动安装依赖 (如果需要)

```powershell
# 安装x64-windows平台的依赖
.\vcpkg install gdal:x64-windows
.\vcpkg install boost-system:x64-windows
.\vcpkg install boost-thread:x64-windows
.\vcpkg install boost-filesystem:x64-windows
.\vcpkg install boost-date-time:x64-windows
.\vcpkg install gtest:x64-windows
.\vcpkg install sqlite3:x64-windows
.\vcpkg install netcdf-c:x64-windows
.\vcpkg install spdlog:x64-windows
.\vcpkg install nlohmann-json:x64-windows
.\vcpkg install eigen3:x64-windows
.\vcpkg install xxhash:x64-windows
.\vcpkg install tiff:x64-windows
.\vcpkg install proj:x64-windows
```

## 构建项目

### 使用PowerShell脚本 (推荐)

```powershell
# 基本构建
.\build_with_vcpkg.ps1

# 清理并重新构建
.\build_with_vcpkg.ps1 -Clean

# 构建并运行测试
.\build_with_vcpkg.ps1 -RunTests

# 指定构建类型
.\build_with_vcpkg.ps1 -BuildType Release

# 指定自定义vcpkg路径
.\build_with_vcpkg.ps1 -VcpkgRoot "C:\custom\vcpkg\path"
```

### 手动CMake命令

```powershell
# 创建构建目录
mkdir build
cd build

# 配置CMake
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:\Users\flyfox\vcpkg\scripts\buildsystems\vcpkg.cmake -DBUILD_TESTING=ON

# 构建项目
cmake --build . --config Debug --verbose

# 运行测试
ctest --output-on-failure --build-config Debug
```

## 环境变量设置

可以设置以下环境变量来简化配置：

```powershell
# 设置VCPKG_ROOT环境变量
$env:VCPKG_ROOT = "C:\Users\flyfox\vcpkg"

# 设置GDAL_DATA路径
$env:GDAL_DATA = "C:\Users\flyfox\vcpkg\installed\x64-windows\share\gdal"

# 设置PROJ_LIB路径
$env:PROJ_LIB = "C:\Users\flyfox\vcpkg\installed\x64-windows\share\proj"
```

## 故障排除

### 常见问题

1. **vcpkg路径错误**
   - 确保vcpkg安装在正确路径：`C:\Users\flyfox\vcpkg`
   - 检查CMakeLists.txt中的VCPKG_ROOT设置

2. **依赖包未找到**
   - 运行`vcpkg list`检查已安装的包
   - 确保使用正确的triplet (x64-windows)

3. **编译错误**
   - 清理构建目录：`Remove-Item -Recurse -Force build`
   - 重新配置CMake

4. **GDAL数据文件未找到**
   - 检查GDAL_DATA环境变量设置
   - 确保路径存在：`C:\Users\flyfox\vcpkg\installed\x64-windows\share\gdal`

### 调试命令

```powershell
# 检查vcpkg状态
.\vcpkg list

# 检查特定包的信息
.\vcpkg search gdal

# 重新安装包
.\vcpkg remove gdal:x64-windows
.\vcpkg install gdal:x64-windows
```

## 更新依赖

```powershell
# 更新vcpkg本身
cd C:\Users\flyfox\vcpkg
git pull
.\bootstrap-vcpkg.bat

# 更新所有包
.\vcpkg upgrade --no-dry-run
```

## 注意事项

1. 确保vcpkg路径中没有中文字符
2. 使用管理员权限运行PowerShell可能会避免某些权限问题
3. 首次安装依赖可能需要较长时间，请耐心等待
4. 建议定期更新vcpkg和依赖包以获得最新的bug修复和功能 