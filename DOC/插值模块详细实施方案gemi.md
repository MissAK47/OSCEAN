# OSCEAN 插值服务模块详细实施方案（修订版）

## 1. 引言与设计原则

### 1.1 模块定位
插值服务模块（`InterpolationService`）位于OSCEAN架构的**核心服务层 (Layer 3)**。它接收来自上层的插值请求，利用底层的通用工具和数据结构，执行数学插值计算，并将结果返回。

### 1.2 设计原则
*   **数学纯粹性**: 模块核心职责是执行与领域无关的数学插值计算
*   **接口驱动与模块化**: `IInterpolationService` 定义服务契约，`IInterpolationAlgorithm` 定义具体算法插件的契约
*   **职责分离**:
    *   `InterpolationServiceImpl`: 负责请求解析、算法选择、异步任务调度、参数校验及缓存管理
    *   `IInterpolationAlgorithm` 实现类: 封装特定插值算法的数学逻辑
    *   `InterpolationGridUtils`: 提供插值专用的 `GridData` 操作和坐标/索引计算
    *   `InterpolationKernels`: 提供与数据结构解耦的、核心的数值计算函数
*   **使用现有数据结构**: 严格使用项目中已定义的数据结构，不重复定义
*   **利用现有基础设施**: 充分使用已有的线程池、缓存、内存管理等基础设施
*   **高性能**: 通过异步处理、利用公共线程池、高效内存管理和缓存来保证性能
*   **可扩展性**: 易于添加新的插值算法

## 2. 现有数据结构使用

### 2.1 核心数据类型（直接使用现有定义）

插值模块将直接使用`core_service_interfaces/include/core_services/common_data_types.h`中已定义的类型：

```cpp
// 在插值模块中使用现有类型
namespace oscean::core_services::interpolation {
    // 直接使用现有类型，不重新定义
    using GridData = oscean::core_services::GridData;
    using Point = oscean::core_services::Point;
    using CRSInfo = oscean::core_services::CRSInfo;
    using DataType = oscean::core_services::DataType;
    using GridDefinition = oscean::core_services::GridDefinition;
    using DimensionCoordinateInfo = oscean::core_services::DimensionCoordinateInfo;
    using BoundingBox = oscean::core_services::BoundingBox;
}
```

### 2.2 现有GridData接口使用

```cpp
// 使用现有GridData的接口
const GridData& grid = sourceGrid;

// 获取网格基本信息
size_t rows = grid.getDefinition().rows;
size_t cols = grid.getDefinition().cols;
DataType dataType = grid.getDataType();

// 数据访问（使用现有模板方法）
auto value = grid.getValue<double>(row, col, band);
grid.setValue<double>(row, col, band, newValue);

// 获取地理变换和CRS
const auto& geoTransform = grid.getGeoTransform();
const auto& crs = grid.getCRS();
```

## 3. 插值专用工具类设计

### 3.1 `InterpolationGridUtils` 工具类

**位置**: `core_services_impl/interpolation_service/src/interpolation_grid_utils.h`
**命名空间**: `oscean::core_services::interpolation::utils`

```cpp
// 在 interpolation_grid_utils.h 中
#pragma once
#include "core_services/common_data_types.h"
#include <vector>
#include <optional>
#include <array>

namespace oscean::core_services::interpolation::utils {

/**
 * @brief 插值专用网格工具类
 * 基于现有GridData接口提供插值算法需要的辅助功能
 */
class InterpolationGridUtils {
public:
    /**
     * @brief 将世界坐标转换为分数网格索引
     * @param grid 网格数据
     * @param worldPoint 世界坐标点
     * @return 如果点在网格范围内返回分数索引，否则返回std::nullopt
     */
    static std::optional<std::vector<double>> worldToFractionalGridIndices(
        const oscean::core_services::GridData& grid,
        const oscean::core_services::Point& worldPoint);

    /**
     * @brief 获取双线性插值的四个角点索引和权重
     * @param grid 网格数据
     * @param fractionalIndices 分数索引（x, y）
     * @param corners 输出：四个角点的索引 [左下, 右下, 左上, 右上]
     * @param weights 输出：对应的权重
     * @return 成功返回true
     */
    static bool getBilinearCorners(
        const oscean::core_services::GridData& grid,
        const std::vector<double>& fractionalIndices,
        std::array<std::pair<size_t, size_t>, 4>& corners,
        std::array<double, 4>& weights);

    /**
     * @brief 安全获取网格值（处理边界和NoData）
     * @param grid 网格数据
     * @param row 行索引
     * @param col 列索引
     * @param band 波段索引
     * @return 如果有效且非NoData返回值，否则返回std::nullopt
     */
    template<typename T>
    static std::optional<T> safeGetValue(
        const oscean::core_services::GridData& grid,
        size_t row, size_t col, size_t band = 0);

    /**
     * @brief 检查两个网格是否具有兼容的维度结构
     * @param grid1 第一个网格
     * @param grid2 第二个网格
     * @return 如果兼容返回true
     */
    static bool areGridsCompatible(
        const oscean::core_services::GridData& grid1,
        const oscean::core_services::GridData& grid2);
};

} // namespace oscean::core_services::interpolation::utils
```

### 3.2 `InterpolationKernels` 数学内核

**位置**: `core_services_impl/interpolation_service/src/algorithms/kernels/interpolation_kernels.h`
**命名空间**: `oscean::core_services::interpolation::kernels`

```cpp
// 在 interpolation_kernels.h 中
#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <optional>

namespace oscean::core_services::interpolation::kernels {

/**
 * @brief 双线性插值
 * @param values 四个角点的值 [左下, 右下, 左上, 右上]
 * @param wx X方向权重 (0到1)
 * @param wy Y方向权重 (0到1)
 * @return 插值结果
 */
double bilinear(const std::array<double, 4>& values, double wx, double wy);

/**
 * @brief 三线性插值
 * @param values 八个角点的值
 * @param wx X方向权重
 * @param wy Y方向权重
 * @param wz Z方向权重
 * @return 插值结果
 */
double trilinear(const std::array<double, 8>& values, double wx, double wy, double wz);

/**
 * @brief 计算PCHIP斜率
 * @param y_prev 前一个点的值
 * @param y_curr 当前点的值
 * @param y_next 下一个点的值
 * @param h_prev 前一段的长度
 * @param h_curr 当前段的长度
 * @return 计算的斜率
 */
double pchipSlope(double y_prev, double y_curr, double y_next, 
                  double h_prev, double h_curr);

/**
 * @brief 计算埃尔米特多项式值
 * @param t_normalized 归一化参数 (0到1)
 * @param y_k 起点值
 * @param y_k_plus_1 终点值
 * @param slope_k 起点斜率
 * @param slope_k_plus_1 终点斜率
 * @param h_k 段长度
 * @return 插值结果
 */
double evaluateHermitePolynomial(
    double t_normalized,
    double y_k, double y_k_plus_1,
    double slope_k, double slope_k_plus_1,
    double h_k);

} // namespace oscean::core_services::interpolation::kernels
```

## 4. 插值服务接口（使用现有定义）

插值服务将使用已存在的接口定义：
- `core_service_interfaces/include/core_services/interpolation/i_interpolation_service.h`
- `core_services_impl/interpolation_service/algorithms/i_interpolation_algorithm.h`

### 4.1 现有接口概览

```cpp
// 现有的插值方法枚举（来自现有代码）
enum class InterpolationMethod {
    UNKNOWN,
    LINEAR_1D,
    CUBIC_SPLINE_1D,
    NEAREST_NEIGHBOR,
    BILINEAR,
    BICUBIC,
    TRILINEAR,
    TRICUBIC,
    PCHIP_RECURSIVE_NDIM,
    PCHIP_OPTIMIZED_2D_BATHY,
    PCHIP_OPTIMIZED_3D_SVP
};

// 现有的插值请求结构
struct InterpolationRequest {
    std::shared_ptr<GridData> sourceGrid;
    std::variant<std::vector<TargetPoint>, TargetGridDefinition> target;
    InterpolationMethod method = InterpolationMethod::UNKNOWN;
    AlgorithmParameters algorithmParams;
    DataType desiredOutputValueType = DataType::Float32;
};
```

## 5. 插值服务实现

### 5.1 `InterpolationServiceImpl` 核心实现

**位置**: `core_services_impl/interpolation_service/src/interpolation_service_impl.h`

```cpp
// 在 interpolation_service_impl.h 中
#pragma once
#include "core_services/interpolation/i_interpolation_service.h"
#include "common_utils/thread_pool_manager.h"
#include "common_utils/cache/interpolation_cache.h"
#include "algorithms/i_interpolation_algorithm.h"
#include <unordered_map>
#include <memory>

namespace oscean::core_services::interpolation {

class InterpolationServiceImpl : public IInterpolationService {
public:
    InterpolationServiceImpl(
        std::shared_ptr<oscean::common_utils::IThreadPoolManager> threadPoolManager,
        std::shared_ptr<oscean::common_utils::cache::InterpolationCache> cache);

    ~InterpolationServiceImpl() override = default;

    std::future<InterpolationResult> interpolateAsync(
        const InterpolationRequest& request) override;

    std::vector<InterpolationMethod> getSupportedMethods() const override;

private:
    void registerAlgorithms();
    std::unique_ptr<IInterpolationAlgorithm> getAlgorithm(InterpolationMethod method);
    std::string generateCacheKey(const InterpolationRequest& request);

    std::shared_ptr<oscean::common_utils::IThreadPoolManager> threadPoolManager_;
    std::shared_ptr<oscean::common_utils::cache::InterpolationCache> interpolationCache_;
    std::unordered_map<InterpolationMethod, std::unique_ptr<IInterpolationAlgorithm>> algorithms_;
};

} // namespace oscean::core_services::interpolation
```

### 5.2 核心实现逻辑

```cpp
// 在 interpolation_service_impl.cpp 中
std::future<InterpolationResult> InterpolationServiceImpl::interpolateAsync(
    const InterpolationRequest& request) {
    
    return std::async(std::launch::async, [this, request]() {
        // 1. 检查缓存
        auto cacheKey = generateCacheKey(request);
        if (auto cached = interpolationCache_->getInterpolationResult(cacheKey)) {
            return *cached;
        }
        
        // 2. 选择算法
        auto algorithm = getAlgorithm(request.method);
        if (!algorithm) {
            InterpolationResult result;
            result.statusCode = -1;
            result.message = "不支持的插值方法";
            return result;
        }
        
        // 3. 执行插值
        auto result = algorithm->execute(request, interpolationCache_.get());
        
        // 4. 缓存结果
        if (result.statusCode == 0) {
            interpolationCache_->cacheInterpolationResult(cacheKey, result);
        }
        
        return result;
    });
}

void InterpolationServiceImpl::registerAlgorithms() {
    // 注册基础算法
    algorithms_[InterpolationMethod::NEAREST_NEIGHBOR] = 
        std::make_unique<NearestNeighborInterpolator>();
    
    algorithms_[InterpolationMethod::LINEAR_1D] = 
        std::make_unique<Linear1DInterpolator>();
    
    algorithms_[InterpolationMethod::BILINEAR] = 
        std::make_unique<BilinearInterpolator>();
    
    algorithms_[InterpolationMethod::TRILINEAR] = 
        std::make_unique<TrilinearInterpolator>();
    
    // 统一PCHIP算法
    algorithms_[InterpolationMethod::PCHIP_RECURSIVE_NDIM] = 
        std::make_unique<PCHIPInterpolator>();
}
```

## 6. 插值算法实现

### 6.1 统一PCHIP插值器

**位置**: `core_services_impl/interpolation_service/src/algorithms/pchip_interpolator.h`

```cpp
// 在 pchip_interpolator.h 中
#pragma once
#include "i_interpolation_algorithm.h"
#include "../interpolation_grid_utils.h"

namespace oscean::core_services::interpolation {

class PCHIPInterpolator : public IInterpolationAlgorithm {
public:
    enum class Mode {
        PURE_1D,        // 纯1D插值
        SEPARABLE_2D,   // 2D可分离插值
        SEPARABLE_3D    // 3D可分离插值
    };
    
    struct Parameters {
        Mode mode = Mode::SEPARABLE_2D;
        bool enablePrecomputation = false;
        double tolerance = 1e-10;
    };
    
    explicit PCHIPInterpolator(const Parameters& params = {});
    
    InterpolationResult execute(
        const InterpolationRequest& request,
        PrecomputedDataCache* cache = nullptr) override;
    
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::PCHIP_RECURSIVE_NDIM;
    }

private:
    Parameters params_;
    
    // 核心算法实现
    std::vector<double> interpolate1D(
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& xi) const;
    
    InterpolationResult interpolateSeparable2D(
        const oscean::core_services::GridData& sourceGrid,
        const std::vector<TargetPoint>& targets) const;
    
    InterpolationResult interpolateSeparable3D(
        const oscean::core_services::GridData& sourceGrid,
        const std::vector<TargetPoint>& targets) const;
};

} // namespace oscean::core_services::interpolation
```

### 6.2 双线性插值器

```cpp
// 在 bilinear_interpolator.h 中
#pragma once
#include "i_interpolation_algorithm.h"

namespace oscean::core_services::interpolation {

class BilinearInterpolator : public IInterpolationAlgorithm {
public:
    InterpolationResult execute(
        const InterpolationRequest& request,
        PrecomputedDataCache* cache = nullptr) override;
    
    InterpolationMethod getMethodType() const override {
        return InterpolationMethod::BILINEAR;
    }

private:
    double interpolateAtPoint(
        const oscean::core_services::GridData& grid,
        const TargetPoint& point) const;
};

} // namespace oscean::core_services::interpolation
```

## 7. 文件与目录结构

### 7.1 最终目录结构

```
core_services_impl/
└── interpolation_service/
    ├── include/core_services/interpolation/impl/
    │   └── interpolation_service_impl.h
    ├── src/
    │   ├── interpolation_service_impl.cpp
    │   ├── interpolation_grid_utils.h
    │   ├── interpolation_grid_utils.cpp
    │   └── algorithms/
    │       ├── i_interpolation_algorithm.h          # 已存在
    │       ├── nearest_neighbor_interpolator.h
    │       ├── nearest_neighbor_interpolator.cpp
    │       ├── linear_1d_interpolator.h
    │       ├── linear_1d_interpolator.cpp
    │       ├── bilinear_interpolator.h
    │       ├── bilinear_interpolator.cpp
    │       ├── trilinear_interpolator.h
    │       ├── trilinear_interpolator.cpp
    │       ├── pchip_interpolator.h
    │       ├── pchip_interpolator.cpp
    │       └── kernels/
    │           ├── interpolation_kernels.h
    │           └── interpolation_kernels.cpp
    └── tests/
        ├── test_interpolation_service.cpp
        ├── test_interpolation_algorithms.cpp
        ├── test_interpolation_utils.cpp
        └── test_interpolation_kernels.cpp
```

### 7.2 CMakeLists.txt 配置

```cmake
# 在 core_services_impl/interpolation_service/CMakeLists.txt 中
cmake_minimum_required(VERSION 3.16)

# 插值服务库
add_library(interpolation_service
    src/interpolation_service_impl.cpp
    src/interpolation_grid_utils.cpp
    src/algorithms/nearest_neighbor_interpolator.cpp
    src/algorithms/linear_1d_interpolator.cpp
    src/algorithms/bilinear_interpolator.cpp
    src/algorithms/trilinear_interpolator.cpp
    src/algorithms/pchip_interpolator.cpp
    src/algorithms/kernels/interpolation_kernels.cpp
)

target_include_directories(interpolation_service
    PUBLIC
        include
    PRIVATE
        src
)

target_link_libraries(interpolation_service
    PUBLIC
        core_service_interfaces
        common_utilities
)

# 测试
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
```

## 8. 实施计划

### 8.1 开发阶段（总计5-6周）

**阶段1：基础框架（2周）**
- 实现`InterpolationServiceImpl`基础框架
- 集成现有的线程池和缓存系统
- 实现`InterpolationGridUtils`工具类
- 实现最近邻和线性插值算法

**阶段2：核心算法（2周）**
- 实现双线性、三线性插值算法
- 实现`InterpolationKernels`数学内核
- 实现统一PCHIP算法
- 完善错误处理和边界情况

**阶段3：测试与优化（1-2周）**
- 完整的单元测试覆盖
- 集成测试
- 性能测试和优化
- 文档完善

### 8.2 关键里程碑

1. **里程碑1**：基础框架完成，能够执行简单插值
2. **里程碑2**：所有核心算法实现完成
3. **里程碑3**：测试覆盖率达到90%以上，性能达标

## 9. 质量保证

### 9.1 测试策略

```cpp
// 测试示例
TEST(InterpolationServiceTest, BilinearInterpolation) {
    // 创建测试网格
    auto grid = createTestGrid();
    
    // 创建插值请求
    InterpolationRequest request;
    request.sourceGrid = grid;
    request.method = InterpolationMethod::BILINEAR;
    
    std::vector<TargetPoint> targets = {
        {{1.5, 1.5}}, // 网格中心点
        {{0.0, 0.0}}, // 边界点
    };
    request.target = targets;
    
    // 执行插值
    auto service = createInterpolationService();
    auto result = service->interpolateAsync(request).get();
    
    // 验证结果
    EXPECT_EQ(result.statusCode, 0);
    EXPECT_FALSE(std::get<std::vector<std::optional<double>>>(result.data).empty());
}
```

### 9.2 性能要求

- 单点插值延迟 < 1ms
- 1000点批量插值 < 100ms
- 内存使用效率 > 80%
- 缓存命中率 > 70%

## 10. 总结

本修订方案的主要改进：

1. **完全基于现有代码结构**：不重复定义数据类型，充分利用现有基础设施
2. **大幅简化设计**：从原方案的20+个文件减少到12个核心文件
3. **缩短开发周期**：从9-13周缩短到5-6周
4. **提高可维护性**：统一PCHIP实现，清晰的模块划分
5. **确保正确性**：基于实际代码分析，避免设计错误

该方案现在可以直接指导开发过程，无需额外的架构调整。 