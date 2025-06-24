# NetCDF-CF Reader 实现指南

## 总体流程

根据重构方案，我们将分阶段实现新的模块化 NetCDF-CF Reader。本文档提供每个模块实现的顺序和指导。

## 实现顺序

按照依赖关系，建议按以下顺序进行实现：

1. **工具类和公共组件**
   - `netcdf_common_types.h` (已完成)
   - `netcdf_utils.h` (已完成)
   - `netcdf_utils.cpp`

2. **缓存管理器**
   - `netcdf_cache_manager.h` (已完成)
   - `netcdf_cache_manager.cpp`

3. **文件处理器**
   - `netcdf_file_handler.h`
   - `netcdf_file_handler.cpp`

4. **元数据管理器**
   - `netcdf_metadata_manager.h`
   - `netcdf_metadata_manager.cpp`

5. **坐标系统**
   - `netcdf_coordinate_system.h`
   - `netcdf_coordinate_system.cpp`

6. **时间工具**
   - `netcdf_time_utils.h`
   - `netcdf_time_utils.cpp`

7. **数据读取器**
   - `netcdf_data_reader.h`
   - `netcdf_data_reader.cpp`

8. **主类**
   - `netcdf_cf_reader.h`
   - `netcdf_cf_reader.cpp`

## 实现注意事项

### 1. 锁机制使用原则

- **锁层次规则**
  ```
  _mutex (全局状态)
  ↓
  _metadataMutex (元数据)
  ↓
  _cacheMutex (缓存)
  ↓
  _ncAccessMutex (NetCDF API)
  ```

- **双重检查锁定模式 (DCLP)**
  在缓存检查和更新时，必须使用正确的DCLP实现：
  ```cpp
  // 检查缓存
  {
      std::shared_lock<std::shared_mutex> sharedLock(_cacheMutex);
      if (inCache) return cachedValue;
  }
  
  // 更新缓存
  {
      std::unique_lock<std::shared_mutex> exclusiveLock(_cacheMutex);
      // 再次检查
      if (inCache) return cachedValue;
      // 更新缓存
      // ...
  }
  ```

- **最小化锁范围**
  获取数据副本后尽快释放锁：
  ```cpp
  DataCopy localCopy;
  {
      std::shared_lock<std::shared_mutex> lock(_mutex);
      localCopy = _data;
  }
  // 在锁外处理数据
  processData(localCopy);
  ```

### 2. 错误处理统一

- 对于可预期的失败，使用 `std::optional` 或 `NetCDFResult<T>` 返回
- 对于不可恢复的错误，使用异常
- 保持日志记录一致性

### 3. 内存管理

- 使用 RAII 和智能指针管理资源
- 避免不必要的数据复制
- 针对大数据集使用分块处理和流式传输

### 4. 测试策略

- 为每个模块编写单元测试
- 为关键功能创建集成测试
- 包含边界条件和错误情况

## 实现模板

### 代码文件头部模板

```cpp
/**
 * @file [文件名]
 * @brief [简短描述]
 * @note [详细说明]
 */

#pragma once

// 包含必要的头文件
#include <...>

namespace oscean::core_services::data_access::readers {

// 类/函数实现

} // namespace oscean::core_services::data_access::readers
```

### 类实现模板

```cpp
/**
 * @class [类名]
 * @brief [简短描述]
 * @note [详细说明]
 */
class ClassName {
public:
    /**
     * @brief 构造函数
     * @param [参数说明]
     */
    ClassName([参数]);
    
    /**
     * @brief 方法描述
     * @param [参数说明]
     * @return [返回值说明]
     */
    ReturnType methodName(Parameters);
    
private:
    // 私有成员变量
    Type _memberVariable;
    
    // 私有辅助方法
    void helperMethod();
};
```

## 依赖管理

确保在 CMake 文件中正确设置包含路径和链接库：

```cmake
target_include_directories(netcdf_reader
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${NETCDF_INCLUDE_DIRS}
)

target_link_libraries(netcdf_reader
    PRIVATE
        ${NETCDF_LIBRARIES}
        common_utilities
        core_service_interfaces
)
```

## 异常处理

自定义异常类示例：

```cpp
class NetCDFException : public std::runtime_error {
public:
    explicit NetCDFException(const std::string& message) 
        : std::runtime_error(message) {}
};

class NetCDFTimeoutException : public NetCDFException {
public:
    explicit NetCDFTimeoutException(const std::string& message) 
        : NetCDFException(message) {}
};
```

## 并发控制

使用合适的锁保护共享资源：

```cpp
// 多线程读取场景
{
    std::shared_lock<std::shared_mutex> lock(_mutex);
    // 读取操作
}

// 修改场景
{
    std::unique_lock<std::shared_mutex> lock(_mutex);
    // 修改操作
}
```

## 性能优化技巧

1. **内存预分配**
   ```cpp
   std::vector<double> results;
   results.reserve(expectedSize); // 预分配内存
   ```

2. **减少锁竞争**
   ```cpp
   // 获取共享数据副本
   DataType localCopy;
   {
       std::shared_lock<std::shared_mutex> lock(_mutex);
       localCopy = _sharedData;
   }
   // 在锁外处理数据
   processData(localCopy);
   ```

3. **批量处理**
   ```cpp
   // 一次性处理多个元素
   template<typename T>
   void processDataBatch(const std::vector<T>& data, size_t batchSize) {
       for (size_t i = 0; i < data.size(); i += batchSize) {
           size_t currentBatchSize = std::min(batchSize, data.size() - i);
           processBatch(data.data() + i, currentBatchSize);
       }
   }
   ```

4. **避免频繁分配/释放内存**
   ```cpp
   // 重用缓冲区
   class DataProcessor {
   private:
       std::vector<float> _buffer;
   
   public:
       void process(size_t size) {
           if (_buffer.size() < size) {
               _buffer.resize(size);
           }
           // 使用 _buffer 处理数据
       }
   };
   ```

## 代码迁移策略

1. 从原始文件提取方法时：
   - 保留原始逻辑
   - 改进锁机制
   - 统一错误处理
   - 增强注释

2. 测试先行：
   - 为每个提取的组件编写单元测试
   - 确保行为与原始实现一致

3. 逐步替换：
   - 先实现基础组件
   - 进行单元测试
   - 然后实现依赖这些组件的高级功能

## 下一步行动

1. 实现 `netcdf_utils.cpp`
2. 实现 `netcdf_cache_manager.cpp`
3. 继续按上述顺序实现其他组件

这个指南将帮助确保重构过程中的一致性和质量，同时减少引入新错误的风险。 