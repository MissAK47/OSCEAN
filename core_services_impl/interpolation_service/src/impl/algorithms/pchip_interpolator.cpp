#include "pchip_interpolator.h"
#include "kernels/interpolation_kernels.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <numeric>

// SIMD头文件
#ifdef __AVX2__
#include <immintrin.h>
#endif

// CPU检测头文件
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace oscean::core_services::interpolation {

namespace { // Anonymous namespace for static helpers

/**
 * @brief 改进的PCHIP斜率计算（基于USML/Fritsch-Carlson方法）
 * @param d0 前一个区间的斜率 (y[k]-y[k-1])/h[k-1]
 * @param d1 当前区间的斜率 (y[k+1]-y[k])/h[k]
 * @param w0 前一个区间的权重 (2*h[k] + h[k-1])
 * @param w1 当前区间的权重 (h[k] + 2*h[k-1])
 * @return 节点k的加权调和平均斜率
 */
inline double pchipSlopeWeightedHarmonic(double d0, double d1, double w0, double w1) noexcept {
    if (d0 * d1 <= 0.0) {
        return 0.0; // 局部极值点，斜率为0
    }
    // 使用加权调和平均
    return (w0 + w1) / (w0 / d0 + w1 / d1);
}

/**
 * @brief 改进的PCHIP端点斜率约束（基于USML/Matlab方法）
 * @param slope 初始斜率估计（由非中心差分公式计算）
 * @param d_adj 相邻区间的斜率
 * @param d_next 下一个区间的斜率
 * @return 修正后的端点斜率
 */
inline double pchipEndpointSlopeConstrained(double slope, double d_adj, double d_next) noexcept {
    if (slope * d_adj < 0.0) {
        return 0.0; // 如果估计的斜率与区间斜率异号，则设为0
    }
    // 如果单调性发生变化，且斜率绝对值过大，则进行限制
    if ((d_adj * d_next < 0.0) && (std::abs(slope) > std::abs(3.0 * d_adj))) {
        return 3.0 * d_adj;
    }
    return slope;
}

// CPU能力检测（静态成员，只检测一次）
namespace {
    struct CPUCapabilities {
        bool hasSSE42 = false;
        bool hasAVX2 = false;
        bool hasAVX512F = false;
        
        CPUCapabilities() {
            #ifdef _MSC_VER
            // MSVC方式检测CPU能力
            int cpuInfo[4];
            __cpuid(cpuInfo, 0);
            int nIds = cpuInfo[0];
            
            if (nIds >= 1) {
                __cpuid(cpuInfo, 1);
                hasSSE42 = (cpuInfo[2] & (1 << 20)) != 0;
            }
            
            if (nIds >= 7) {
                __cpuidex(cpuInfo, 7, 0);
                hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;
                hasAVX512F = (cpuInfo[1] & (1 << 16)) != 0;
            }
            #else
            // GCC/Clang方式
            #ifdef __SSE4_2__
            hasSSE42 = __builtin_cpu_supports("sse4.2");
            #endif
            
            #ifdef __AVX2__
            hasAVX2 = __builtin_cpu_supports("avx2");
            #endif
            
            #ifdef __AVX512F__
            hasAVX512F = __builtin_cpu_supports("avx512f");
            #endif
            #endif
        }
    };
    
    static const CPUCapabilities g_cpuCaps;
}

} // namespace

PCHIPInterpolator::PCHIPInterpolator(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
    : simdManager_(std::move(simdManager)) {
}

InterpolationResult PCHIPInterpolator::execute(
    const InterpolationRequest& request,
    const PrecomputedData* precomputed) const {
    
    // 该实现忽略 precomputed 参数
    (void)precomputed;

    InterpolationResult result;
    result.statusCode = -1;

    if (!request.sourceGrid) {
        result.message = "源网格数据为空";
        return result;
    }
    
    if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        const auto& targetPoints = std::get<std::vector<TargetPoint>>(request.target);
        auto values = batchInterpolate(*request.sourceGrid, targetPoints);
        result.data = values;
        result.statusCode = 0;
        result.message = "PCHIP点插值成功完成";
    } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
        try {
            auto gridResult = interpolateToGrid(*request.sourceGrid, std::get<TargetGridDefinition>(request.target));
            result.data = std::move(gridResult);
            result.statusCode = 0;
            result.message = "PCHIP网格插值成功完成";
        } catch (const std::exception& e) {
            result.message = std::string("网格到网格插值失败: ") + e.what();
        }
    } else {
        result.message = "未知的目标类型";
    }
    
    return result;
}

std::vector<std::optional<double>> PCHIPInterpolator::batchInterpolate(
    const GridData& grid,
    const std::vector<TargetPoint>& points) const {
    
    // 如果支持AVX2且点数足够多，使用SIMD优化版本
    if (g_cpuCaps.hasAVX2 && points.size() >= 8) {
        return batchInterpolateSIMD(grid, points);
    }
    
    // 标量版本
    std::vector<std::optional<double>> results;
    results.reserve(points.size());
    
    for (const auto& point : points) {
        results.push_back(interpolateAtPoint(grid, point.coordinates));
    }
    
    return results;
}


GridData PCHIPInterpolator::interpolateToGrid(
    const GridData& sourceGrid,
    const TargetGridDefinition& targetGridDef) const {
    
    GridDefinition targetDef;
    auto xDimIt = std::find_if(targetGridDef.dimensions.begin(), targetGridDef.dimensions.end(), 
                             [](const auto& d){ return d.name == "x" || d.name == "longitude"; });
    auto yDimIt = std::find_if(targetGridDef.dimensions.begin(), targetGridDef.dimensions.end(), 
                             [](const auto& d){ return d.name == "y" || d.name == "latitude"; });

    if (xDimIt == targetGridDef.dimensions.end() || yDimIt == targetGridDef.dimensions.end()) {
        throw std::runtime_error("目标网格定义必须包含x和y维度");
    }

    targetDef.cols = xDimIt->getNumberOfLevels();
    targetDef.rows = yDimIt->getNumberOfLevels();

    if (targetDef.cols == 0 || targetDef.rows == 0) {
        throw std::runtime_error("目标网格维度为0");
    }

    GridData resultGrid(targetDef, sourceGrid.getDataType(), sourceGrid.getBandCount());
    resultGrid.setCrs(targetGridDef.crs);
    
    std::vector<double> targetGeoTransform(6);
    targetGeoTransform[0] = xDimIt->coordinates.front();
    targetGeoTransform[1] = (xDimIt->coordinates.back() - xDimIt->coordinates.front()) / (targetDef.cols > 1 ? targetDef.cols - 1 : 1);
    targetGeoTransform[2] = 0.0;
    targetGeoTransform[3] = yDimIt->coordinates.front();
    targetGeoTransform[4] = 0.0;
    targetGeoTransform[5] = (yDimIt->coordinates.back() - yDimIt->coordinates.front()) / (targetDef.rows > 1 ? targetDef.rows - 1 : 1);
    // 检查Y轴方向
    if (yDimIt->coordinates.front() > yDimIt->coordinates.back()) {
        targetGeoTransform[3] = yDimIt->coordinates.front();
        targetGeoTransform[5] = (yDimIt->coordinates.back() - yDimIt->coordinates.front()) / (targetDef.rows > 1 ? targetDef.rows - 1 : 1);
    } else {
        targetGeoTransform[3] = yDimIt->coordinates.back();
        targetGeoTransform[5] = (yDimIt->coordinates.front() - yDimIt->coordinates.back()) / (targetDef.rows > 1 ? targetDef.rows - 1 : 1);
    }
    resultGrid.setGeoTransform(targetGeoTransform);

    auto* raw_data = const_cast<double*>(static_cast<const double*>(resultGrid.getDataPtr()));

    for (size_t row = 0; row < targetDef.rows; ++row) {
        for (size_t col = 0; col < targetDef.cols; ++col) {
            double worldX = targetGeoTransform[0] + col * targetGeoTransform[1];
            double worldY = targetGeoTransform[3] + row * targetGeoTransform[5];
            
            auto val = interpolateAtPoint(sourceGrid, {worldX, worldY});
            
            size_t index = row * targetDef.cols + col;
            raw_data[index] = val.value_or(targetGridDef.fillValue.value_or(std::numeric_limits<double>::quiet_NaN()));
        }
    }
    
    return resultGrid;
}


std::optional<double> PCHIPInterpolator::interpolateAtPoint(
    const GridData& grid, 
    const std::vector<double>& worldCoords) const {
    
    if (worldCoords.size() != 2) return std::nullopt;

    const auto& geoTransform = grid.getGeoTransform();
    if (geoTransform.size() < 6 || geoTransform[1] == 0 || geoTransform[5] == 0) {
        return std::nullopt;
    }
    
    const double gridX = (worldCoords[0] - geoTransform[0]) / geoTransform[1];
    const double gridY = (worldCoords[1] - geoTransform[3]) / geoTransform[5];
    
    const auto& def = grid.getDefinition();
    
    // 严格的边界检查 - 不允许外推
    if (gridX < 0 || gridX >= def.cols - 1 || 
        gridY < 0 || gridY >= def.rows - 1) {
        return std::nullopt;
    }
    
    if (def.cols < 4 || def.rows < 4) {
        // PCHIP 需要至少4个点，这里简化处理，可以回退到双线性
        return std::nullopt; 
    }

    const std::vector<double> gridCoords = {gridX, gridY};
    std::vector<size_t> indices(2);
    indices[0] = static_cast<size_t>(std::floor(gridX));
    indices[1] = static_cast<size_t>(std::floor(gridY));

    // 当前只支持2D，所以递归深度为1
    LayoutAwareAccessor accessor(grid);
    return interpRecursive(grid, 1, indices, gridCoords, accessor);
}

std::optional<double> PCHIPInterpolator::interpRecursive(
    const GridData& grid,
    int dim,
    std::vector<size_t>& indices,
    const std::vector<double>& gridCoords,
    const LayoutAwareAccessor& accessor
) const {
    if (dim < 0) {
        return getGridValue(grid, indices, 0, accessor);
    }

    // 在PCHIPInterpolator中，我们总是使用pchip1D
    return pchip1D(grid, dim, indices, gridCoords, accessor);
}


std::optional<double> PCHIPInterpolator::getGridValue(
    const GridData& grid,
    const std::vector<size_t>& indices,
    size_t band,
    const LayoutAwareAccessor& accessor) const {
    
    const auto& def = grid.getDefinition();
    if (indices.size() != 2) return std::nullopt;

    size_t col = indices[0];
    size_t row = indices[1];

    if (col >= def.cols || row >= def.rows || band >= grid.getBandCount()) {
        return std::nullopt;
    }
    
    try {
        switch (grid.getDataType()) {
            case DataType::Float64:
                return grid.getValue<double>(row, col, band);
            case DataType::Float32:
                return static_cast<double>(grid.getValue<float>(row, col, band));
            default:
                 // 为简化，只处理了最常见的类型
                return std::nullopt;
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

double PCHIPInterpolator::pchip1D(
    const GridData& grid,
    int dim,
    std::vector<size_t>& indices,
    const std::vector<double>& gridCoords,
    const LayoutAwareAccessor& accessor
) const {
    const auto& def = grid.getDefinition();
    const std::vector<size_t> shape = {def.cols, def.rows};
    size_t k = indices[dim];

    const double h = 1.0;

    auto getValueAt = [&](int offset) -> double {
        std::vector<size_t> temp_indices = indices;
        long long requested_k = static_cast<long long>(k) + offset;
        requested_k = std::max(0LL, std::min(requested_k, static_cast<long long>(shape[dim] - 1)));
        temp_indices[dim] = static_cast<size_t>(requested_k);
        auto result = interpRecursive(grid, dim - 1, temp_indices, gridCoords, accessor);
        return result.value_or(0.0);
    };

    double y0 = getValueAt(-1);
    double y1 = getValueAt(0);
    double y2 = getValueAt(1);
    double y3 = getValueAt(2);

    double deriv0 = (y1 - y0) / h;
    double deriv1 = (y2 - y1) / h;
    double deriv2 = (y3 - y2) / h;

    double slope1, slope2;

    if (k > 0) {
        slope1 = pchipSlopeWeightedHarmonic(deriv0, deriv1, 3.0, 3.0);
    } else {
        double initial_slope = (3.0 * deriv1 - deriv2) / 2.0;
        slope1 = pchipEndpointSlopeConstrained(initial_slope, deriv1, deriv2);
    }

    if (k < shape[dim] - 2) {
        slope2 = pchipSlopeWeightedHarmonic(deriv1, deriv2, 3.0, 3.0);
    } else {
        double initial_slope = (3.0 * deriv1 - deriv0) / 2.0;
        slope2 = pchipEndpointSlopeConstrained(initial_slope, deriv1, deriv0);
    }

    double t = gridCoords[dim] - static_cast<double>(k);
    return kernels::evaluateHermitePolynomial(t, y1, y2, slope1, slope2, h);
}

/**
 * @brief SIMD优化的批量插值
 */
std::vector<std::optional<double>> PCHIPInterpolator::batchInterpolateSIMD(
    const GridData& grid,
    const std::vector<TargetPoint>& points) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(points.size());
    
#ifdef __AVX2__
    const auto& geoTransform = grid.getGeoTransform();
    if (geoTransform.size() < 6 || geoTransform[1] == 0 || geoTransform[5] == 0) {
        // 无效的地理变换，回退到标量版本
        for (const auto& point : points) {
            results.push_back(interpolateAtPoint(grid, point.coordinates));
        }
        return results;
    }
    
    const auto& def = grid.getDefinition();
    if (def.cols < 4 || def.rows < 4) {
        // PCHIP需要至少4个点
        for (size_t i = 0; i < points.size(); ++i) {
            results.push_back(std::nullopt);
        }
        return results;
    }
    
    // AVX2优化路径：一次处理4个点
    const size_t simdWidth = 4;
    const size_t numFullBatches = points.size() / simdWidth;
    
    // 准备AVX2常量
    __m256d vOriginX = _mm256_set1_pd(geoTransform[0]);
    __m256d vOriginY = _mm256_set1_pd(geoTransform[3]);
    __m256d vDxInv = _mm256_set1_pd(1.0 / geoTransform[1]);
    __m256d vDyInv = _mm256_set1_pd(1.0 / geoTransform[5]);
    __m256d vMaxX = _mm256_set1_pd(def.cols - 1.0);
    __m256d vMaxY = _mm256_set1_pd(def.rows - 1.0);
    __m256d vZero = _mm256_setzero_pd();
    
    // 对齐的临时数组
    alignas(32) double worldX[4];
    alignas(32) double worldY[4];
    alignas(32) double gridX[4];
    alignas(32) double gridY[4];
    
    // 批量处理
    for (size_t batch = 0; batch < numFullBatches; ++batch) {
        size_t baseIdx = batch * simdWidth;
        
        // 收集4个点的世界坐标
        for (size_t i = 0; i < simdWidth; ++i) {
            const auto& point = points[baseIdx + i];
            if (point.coordinates.size() >= 2) {
                worldX[i] = point.coordinates[0];
                worldY[i] = point.coordinates[1];
            } else {
                worldX[i] = 0.0;
                worldY[i] = 0.0;
            }
        }
        
        // 加载坐标到AVX2寄存器
        __m256d vWorldX = _mm256_load_pd(worldX);
        __m256d vWorldY = _mm256_load_pd(worldY);
        
        // 转换到网格坐标
        __m256d vGridX = _mm256_mul_pd(_mm256_sub_pd(vWorldX, vOriginX), vDxInv);
        __m256d vGridY = _mm256_mul_pd(_mm256_sub_pd(vWorldY, vOriginY), vDyInv);
        
        // 边界检查
        __m256d vValidX = _mm256_and_pd(
            _mm256_cmp_pd(vGridX, vZero, _CMP_GE_OQ),
            _mm256_cmp_pd(vGridX, vMaxX, _CMP_LE_OQ)
        );
        __m256d vValidY = _mm256_and_pd(
            _mm256_cmp_pd(vGridY, vZero, _CMP_GE_OQ),
            _mm256_cmp_pd(vGridY, vMaxY, _CMP_LE_OQ)
        );
        __m256d vValid = _mm256_and_pd(vValidX, vValidY);
        
        // 存储网格坐标
        _mm256_store_pd(gridX, vGridX);
        _mm256_store_pd(gridY, vGridY);
        
        // 对每个点进行PCHIP插值
        for (size_t i = 0; i < simdWidth; ++i) {
            const auto& point = points[baseIdx + i];
            
            // 检查有效性
            if (point.coordinates.size() < 2 || 
                gridX[i] < 0 || gridX[i] >= def.cols - 1 ||
                gridY[i] < 0 || gridY[i] >= def.rows - 1) {
                results.push_back(std::nullopt);
                continue;
            }
            
            // 执行PCHIP插值
            std::vector<size_t> indices(2);
            indices[0] = static_cast<size_t>(std::floor(gridX[i]));
            indices[1] = static_cast<size_t>(std::floor(gridY[i]));
            
            std::vector<double> gridCoords = {gridX[i], gridY[i]};
            
            LayoutAwareAccessor accessor(grid);
            auto value = interpRecursive(grid, 1, indices, gridCoords, accessor);
            results.push_back(value);
        }
    }
    
    // 处理剩余的点
    for (size_t i = numFullBatches * simdWidth; i < points.size(); ++i) {
        results.push_back(interpolateAtPoint(grid, points[i].coordinates));
    }
#else
    // 非AVX2路径，使用标量版本
    for (const auto& point : points) {
        results.push_back(interpolateAtPoint(grid, point.coordinates));
    }
#endif
    
    return results;
}

} // namespace oscean::core_services::interpolation 