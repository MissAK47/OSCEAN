# CRS服务统一重构实施方案

## 1. 现状分析

### 1.1 当前CRS服务结构

```
core_services_impl/crs_service/
├── include/core_services/crs/
│   ├── crs_service_factory.h        # 现有简单工厂
│   └── internal/
│       └── crs_service_extended.h
├── src/
│   ├── crs_service_factory.cpp      # 实现简单工厂函数
│   ├── crs_service_impl.h           # 服务实现头文件
│   └── crs_service_impl.cpp         # 服务实现
└── tests/
```

### 1.2 当前接口 (全同步)

```cpp
class ICrsService {
    // 同步方法示例
    virtual TransformedPoint transformPoint(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual std::vector<TransformedPoint> transformPoints(const std::vector<Point>& points, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual BoundingBox transformBoundingBox(const BoundingBox& sourceBbox, const CRSInfo& targetCRS) = 0;
    // ... 其他同步方法
};
```

### 1.3 当前工厂函数

```cpp
namespace oscean::core_services {
    std::shared_ptr<ICrsService> createCrsService();
    std::shared_ptr<ICrsServiceGdalExtended> createCrsServiceExtended();
}
```

## 2. 目标架构

### 2.1 新目录结构

```
core_services_impl/crs_service/
├── include/core_services/crs/
│   ├── crs_service_factory.h        # 新工厂类
│   ├── crs_config.h                 # 配置结构
│   └── internal/
│       └── crs_service_extended.h
├── src/
│   ├── crs_service_factory.cpp      # 工厂类实现
│   ├── crs_config.cpp               # 配置实现
│   ├── crs_service_impl.h           # 异步服务实现头文件
│   └── crs_service_impl.cpp         # 异步服务实现
└── tests/
    ├── test_crs_factory.cpp         # 工厂测试
    └── test_crs_config.cpp          # 配置测试
```

### 2.2 新接口 (全异步)

```cpp
class ICrsService {
    // 异步方法
    virtual std::future<TransformedPoint> transformPointAsync(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual std::future<std::vector<TransformedPoint>> transformPointsAsync(const std::vector<Point>& points, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual std::future<BoundingBox> transformBoundingBoxAsync(const BoundingBox& sourceBbox, const CRSInfo& targetCRS) = 0;
    // ... 其他异步方法
};
```

## 3. 实施步骤

### 3.1 步骤一: 创建配置结构

#### 文件: `core_services_impl/crs_service/include/core_services/crs/crs_config.h`

```cpp
#pragma once

#include <string>
#include <chrono>
#include <memory>

namespace spdlog { class logger; }
namespace boost::asio { class thread_pool; }

namespace oscean::core_services::crs {

/**
 * @brief CRS服务配置结构
 */
struct CrsConfig {
    // 基础配置
    std::string serviceName = "CrsService";
    std::string version = "1.0.0";
    
    // 性能配置
    size_t maxConcurrentOperations = 100;
    std::chrono::milliseconds operationTimeout{5000};
    
    // 缓存配置
    bool enableTransformationCaching = true;
    size_t transformationCacheMaxSize = 10000;
    std::chrono::minutes cacheExpiration{60};
    
    // 精度配置
    double transformationTolerance = 1e-9;
    bool enableHighPrecisionMode = false;
    
    // 日志配置
    std::string logLevel = "info";
    bool enableDetailedLogging = false;
    
    // GDAL配置
    std::string gdalDataPath = "";
    bool enableGdalErrorHandler = true;
    
    /**
     * @brief 验证配置有效性
     * @return true if valid, false otherwise
     */
    bool isValid() const;
    
    /**
     * @brief 获取默认配置
     * @return 默认配置实例
     */
    static CrsConfig getDefault();
};

/**
 * @brief CRS服务依赖项结构
 */
struct CrsDependencies {
    // 可选依赖
    std::shared_ptr<spdlog::logger> logger = nullptr;
    std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr;
    
    /**
     * @brief 验证依赖项有效性
     * @return true if all required dependencies are valid
     */
    bool isValid() const;
};

} // namespace oscean::core_services::crs
```

#### 文件: `core_services_impl/crs_service/src/crs_config.cpp`

```cpp
#include "core_services/crs/crs_config.h"

namespace oscean::core_services::crs {

bool CrsConfig::isValid() const {
    return !serviceName.empty() &&
           !version.empty() &&
           maxConcurrentOperations > 0 &&
           operationTimeout.count() > 0 &&
           transformationCacheMaxSize > 0 &&
           transformationTolerance > 0.0;
}

CrsConfig CrsConfig::getDefault() {
    return CrsConfig{};
}

bool CrsDependencies::isValid() const {
    // CRS服务没有必需依赖，只有可选依赖
    return true;
}

} // namespace oscean::core_services::crs
```

### 3.2 步骤二: 创建新工厂类

#### 文件: `core_services_impl/crs_service/include/core_services/crs/crs_service_factory.h`

```cpp
#pragma once

#include "core_services/crs/i_crs_service.h"
#include "core_services/crs/crs_config.h"
#include <memory>

namespace oscean::core_services::crs {

/**
 * @brief CRS服务工厂类
 * 负责创建和配置CRS服务实例，管理依赖注入和初始化过程
 */
class CrsServiceFactory {
public:
    /**
     * @brief 创建CRS服务实例（使用默认配置）
     * @return CRS服务实例的智能指针
     */
    static std::unique_ptr<ICrsService> createService();
    
    /**
     * @brief 创建CRS服务实例（使用指定配置）
     * @param config CRS服务配置
     * @return CRS服务实例的智能指针
     */
    static std::unique_ptr<ICrsService> createService(const CrsConfig& config);
    
    /**
     * @brief 创建CRS服务实例（完整依赖注入）
     * @param config CRS服务配置
     * @param dependencies 依赖服务实例
     * @return CRS服务实例的智能指针
     */
    static std::unique_ptr<ICrsService> createService(
        const CrsConfig& config,
        const CrsDependencies& dependencies);
    
    /**
     * @brief 创建用于测试的Mock CRS服务实例
     * @return Mock CRS服务实例的智能指针
     */
    static std::unique_ptr<ICrsService> createMockService();
    
    /**
     * @brief 验证配置的有效性
     * @param config 要验证的配置
     * @return true if valid, false otherwise
     */
    static bool validateConfig(const CrsConfig& config);

private:
    // 禁止实例化
    CrsServiceFactory() = delete;
    ~CrsServiceFactory() = delete;
    CrsServiceFactory(const CrsServiceFactory&) = delete;
    CrsServiceFactory& operator=(const CrsServiceFactory&) = delete;
};

} // namespace oscean::core_services::crs

// 保持向后兼容的全局工厂函数（标记为废弃）
namespace oscean::core_services {

/**
 * @deprecated 请使用 oscean::core_services::crs::CrsServiceFactory::createService()
 */
[[deprecated("Use oscean::core_services::crs::CrsServiceFactory::createService()")]]
std::shared_ptr<ICrsService> createCrsService();

/**
 * @deprecated 请使用 oscean::core_services::crs::CrsServiceFactory::createService()
 */
[[deprecated("Use oscean::core_services::crs::CrsServiceFactory::createService()")]]
std::shared_ptr<ICrsServiceGdalExtended> createCrsServiceExtended();

} // namespace oscean::core_services
```

#### 文件: `core_services_impl/crs_service/src/crs_service_factory.cpp`

```cpp
#include "core_services/crs/crs_service_factory.h"
#include "crs_service_impl.h"
#include "common_utils/logging.h"
#include <stdexcept>

namespace oscean::core_services::crs {

std::unique_ptr<ICrsService> CrsServiceFactory::createService() {
    return createService(CrsConfig::getDefault());
}

std::unique_ptr<ICrsService> CrsServiceFactory::createService(const CrsConfig& config) {
    CrsDependencies deps;
    return createService(config, deps);
}

std::unique_ptr<ICrsService> CrsServiceFactory::createService(
    const CrsConfig& config,
    const CrsDependencies& dependencies) {
    
    if (!validateConfig(config)) {
        throw oscean::core_services::ServiceCreationException(
            "Invalid CRS service configuration");
    }
    
    if (!dependencies.isValid()) {
        throw oscean::core_services::ServiceCreationException(
            "Invalid CRS service dependencies");
    }
    
    // 创建日志器
    auto logger = dependencies.logger;
    if (!logger) {
        logger = common_utils::getModuleLogger("CrsService");
    }
    
    // 创建线程池
    auto threadPool = dependencies.threadPool;
    if (!threadPool) {
        threadPool = std::make_shared<boost::asio::thread_pool>(
            config.maxConcurrentOperations);
    }
    
    return std::make_unique<impl::CrsServiceImpl>(config, logger, threadPool);
}

std::unique_ptr<ICrsService> CrsServiceFactory::createMockService() {
    // 创建Mock实现或返回测试专用的实现
    auto config = CrsConfig::getDefault();
    config.enableTransformationCaching = false; // 简化测试
    return createService(config);
}

bool CrsServiceFactory::validateConfig(const CrsConfig& config) {
    return config.isValid();
}

} // namespace oscean::core_services::crs

// 向后兼容实现
namespace oscean::core_services {

std::shared_ptr<ICrsService> createCrsService() {
    return oscean::core_services::crs::CrsServiceFactory::createService();
}

std::shared_ptr<ICrsServiceGdalExtended> createCrsServiceExtended() {
    auto service = oscean::core_services::crs::CrsServiceFactory::createService();
    // 假设CrsServiceImpl同时实现了ICrsServiceGdalExtended
    return std::dynamic_pointer_cast<ICrsServiceGdalExtended>(service);
}

} // namespace oscean::core_services
```

### 3.3 步骤三: 修改服务接口为异步

#### 文件: `core_service_interfaces/include/core_services/crs/i_crs_service.h`

```cpp
#pragma once

#include "../common_data_types.h"
#include "crs_operation_types.h"
#include <vector>
#include <optional>
#include <string>
#include <memory>
#include <future>

namespace oscean::core_services {

/**
 * @interface ICrsService
 * @brief 坐标参考系统服务接口 (全异步版本)
 */
class ICrsService {
public:
    virtual ~ICrsService() = default;

    // Parser相关异步方法
    virtual std::future<std::optional<CRSInfo>> parseFromWKTAsync(const std::string& wktString) = 0;
    virtual std::future<std::optional<CRSInfo>> parseFromProjStringAsync(const std::string& projString) = 0;
    virtual std::future<std::optional<CRSInfo>> parseFromEpsgCodeAsync(int epsgCode) = 0;

    // Transformer相关异步方法
    virtual std::future<TransformedPoint> transformPointAsync(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual std::future<TransformedPoint> transformPointAsync(double x, double y, double z, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual std::future<std::vector<TransformedPoint>> transformPointsAsync(const std::vector<Point>& points, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual std::future<BoundingBox> transformBoundingBoxAsync(const BoundingBox& sourceBbox, const CRSInfo& targetCRS) = 0;

    // Inspector相关异步方法
    virtual std::future<std::optional<CRSDetailedParameters>> getDetailedParametersAsync(const CRSInfo& crsInfo) = 0;
    virtual std::future<std::optional<std::string>> getUnitAsync(const CRSInfo& crsInfo) = 0;
    virtual std::future<std::optional<std::string>> getProjectionMethodAsync(const CRSInfo& crsInfo) = 0;
    virtual std::future<bool> areEquivalentCRSAsync(const CRSInfo& crsInfo1, const CRSInfo& crsInfo2) = 0;

    /**
     * @brief 获取服务版本
     * @return 服务版本字符串
     */
    virtual std::string getVersion() const = 0;

    // === 向后兼容的同步方法 (标记为废弃) ===
    
    /**
     * @deprecated 请使用 parseFromWKTAsync()
     */
    [[deprecated("Use parseFromWKTAsync()")]]
    virtual std::optional<CRSInfo> parseFromWKT(const std::string& wktString) {
        try {
            return parseFromWKTAsync(wktString).get();
        } catch (...) {
            return std::nullopt;
        }
    }
    
    /**
     * @deprecated 请使用 transformPointAsync()
     */
    [[deprecated("Use transformPointAsync()")]]
    virtual TransformedPoint transformPoint(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) {
        return transformPointAsync(x, y, sourceCRS, targetCRS).get();
    }
    
    // ... 其他向后兼容方法 ...
};

} // namespace oscean::core_services
```

### 3.4 步骤四: 更新服务实现

#### 文件: `core_services_impl/crs_service/src/crs_service_impl.h`

```cpp
#pragma once

#include "core_services/crs/i_crs_service.h"
#include "core_services/crs/crs_config.h"
#include <memory>
#include <boost/asio/thread_pool.hpp>

namespace spdlog { class logger; }

namespace oscean::core_services::crs::impl {

class CrsServiceImpl : public ICrsService {
public:
    explicit CrsServiceImpl(
        const CrsConfig& config,
        std::shared_ptr<spdlog::logger> logger,
        std::shared_ptr<boost::asio::thread_pool> threadPool);
    
    ~CrsServiceImpl() override;

    // 实现异步接口
    std::future<std::optional<CRSInfo>> parseFromWKTAsync(const std::string& wktString) override;
    std::future<std::optional<CRSInfo>> parseFromProjStringAsync(const std::string& projString) override;
    std::future<std::optional<CRSInfo>> parseFromEpsgCodeAsync(int epsgCode) override;
    
    std::future<TransformedPoint> transformPointAsync(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) override;
    std::future<TransformedPoint> transformPointAsync(double x, double y, double z, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) override;
    std::future<std::vector<TransformedPoint>> transformPointsAsync(const std::vector<Point>& points, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) override;
    std::future<BoundingBox> transformBoundingBoxAsync(const BoundingBox& sourceBbox, const CRSInfo& targetCRS) override;
    
    std::future<std::optional<CRSDetailedParameters>> getDetailedParametersAsync(const CRSInfo& crsInfo) override;
    std::future<std::optional<std::string>> getUnitAsync(const CRSInfo& crsInfo) override;
    std::future<std::optional<std::string>> getProjectionMethodAsync(const CRSInfo& crsInfo) override;
    std::future<bool> areEquivalentCRSAsync(const CRSInfo& crsInfo1, const CRSInfo& crsInfo2) override;
    
    std::string getVersion() const override;

private:
    CrsConfig config_;
    std::shared_ptr<spdlog::logger> logger_;
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    
    // 内部同步实现方法（复用现有代码）
    std::optional<CRSInfo> parseFromWKTSync(const std::string& wktString);
    TransformedPoint transformPointSync(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS);
    // ... 其他同步方法
};

} // namespace oscean::core_services::crs::impl
```

## 4. 迁移策略

### 4.1 阶段性迁移

1. **第一阶段**: 保持向后兼容
   - 新接口与旧接口共存
   - 废弃标记指导用户迁移
   - 完整测试覆盖

2. **第二阶段**: 逐步替换
   - 更新调用方代码使用新接口
   - 每个模块测试通过后继续下一个

3. **第三阶段**: 清理旧接口
   - 移除废弃的同步方法
   - 移除全局工厂函数

### 4.2 调用方代码迁移示例

#### 旧代码:
```cpp
auto crsService = oscean::core_services::createCrsService();
auto result = crsService->transformPoint(x, y, sourceCRS, targetCRS);
```

#### 新代码:
```cpp
auto crsService = oscean::core_services::crs::CrsServiceFactory::createService();
auto future = crsService->transformPointAsync(x, y, sourceCRS, targetCRS);
auto result = future.get();
```

#### 或使用配置:
```cpp
auto config = oscean::core_services::crs::CrsConfig::getDefault();
config.maxConcurrentOperations = 200;
auto crsService = oscean::core_services::crs::CrsServiceFactory::createService(config);
```

## 5. 测试计划

### 5.1 单元测试

1. **工厂类测试** (`test_crs_factory.cpp`):
   - 测试所有工厂方法
   - 测试配置验证
   - 测试错误处理

2. **配置测试** (`test_crs_config.cpp`):
   - 测试配置验证逻辑
   - 测试默认配置
   - 测试依赖验证

3. **异步接口测试**:
   - 测试所有异步方法
   - 测试并发操作
   - 测试超时处理

### 5.2 集成测试

1. **向后兼容测试**:
   - 验证废弃方法仍然工作
   - 验证迁移路径

2. **性能测试**:
   - 比较异步vs同步性能
   - 测试并发处理能力

## 6. 完成验证清单

- [ ] `CrsConfig` 和 `CrsDependencies` 结构创建
- [ ] `CrsServiceFactory` 工厂类实现
- [ ] `ICrsService` 接口异步化
- [ ] `CrsServiceImpl` 异步实现
- [ ] 向后兼容层实现
- [ ] 完整单元测试覆盖
- [ ] 集成测试通过
- [ ] 性能测试验证
- [ ] 文档更新

## 7. 风险和缓解措施

### 7.1 主要风险

1. **性能退化**: 异步化可能引入开销
2. **兼容性问题**: 现有代码可能需要大量修改
3. **复杂性增加**: 异步代码调试困难

### 7.2 缓解措施

1. **性能监控**: 建立基准测试，持续监控
2. **渐进迁移**: 保持向后兼容，逐步迁移
3. **充分测试**: 增加单元测试和集成测试覆盖

---

**准备就绪**: 此方案详细规划了CRS服务的完整重构过程。请确认后开始实施第一步：创建配置结构。 