#include "complex_field_interpolator.h"
#include <algorithm>
#include <cmath>

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

ComplexFieldInterpolator::ComplexFieldInterpolator(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    InterpolationMethod baseMethod)
    : LayoutAwareInterpolatorBase(simdManager)
    , baseMethod_(baseMethod) {
}

InterpolationResult ComplexFieldInterpolator::execute(
    const InterpolationRequest& request,
    const PrecomputedData* precomputed) const {
    
    InterpolationResult result;
    
    if (!request.sourceGrid) {
        result.statusCode = -1;
        result.message = "Source grid is null";
        return result;
    }
    
    // 检查是否为复数数据（波段数应为2：实部和虚部）
    if (request.sourceGrid->getBandCount() != 2) {
        result.statusCode = -1;
        result.message = "Complex field interpolation requires 2 bands (real and imaginary)";
        return result;
    }
    
    // 处理点插值
    if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        const auto& points = std::get<std::vector<TargetPoint>>(request.target);
        std::vector<std::optional<double>> values;
        values.reserve(points.size() * 2);  // 实部和虚部
        
        // 创建访问器
        LayoutAwareAccessor accessor(*request.sourceGrid);
        
        for (const auto& point : points) {
            // 转换坐标
            double x = point.coordinates[0];
            double y = point.coordinates.size() > 1 ? point.coordinates[1] : 0;
            double z = point.coordinates.size() > 2 ? point.coordinates[2] : 0;
            
            // 分别插值实部和虚部
            double real = 0, imag = 0;
            
            if (point.coordinates.size() == 2) {
                // 2D插值
                const auto& def = request.sourceGrid->getDefinition();
                auto gridX = (x - def.xDimension.minValue) /
                            (def.xDimension.maxValue - def.xDimension.minValue) *
                            (def.xDimension.coordinates.size() - 1);
                
                auto gridY = (y - def.yDimension.minValue) /
                            (def.yDimension.maxValue - def.yDimension.minValue) *
                            (def.yDimension.coordinates.size() - 1);
                
                // 实部（band 0）
                real = interpolate2DValue(*request.sourceGrid, gridX, gridY, 0, accessor);
                // 虚部（band 1）
                imag = interpolate2DValue(*request.sourceGrid, gridX, gridY, 1, accessor);
            }
            
            values.push_back(real);
            values.push_back(imag);
        }
        
        result.data = std::move(values);
        result.statusCode = 0;
    }
    
    return result;
}

std::complex<double> ComplexFieldInterpolator::interpolateComplex(
    const GridData& realGrid,
    const GridData& imagGrid,
    double x, double y, double z) const {
    
    LayoutAwareAccessor realAccessor(realGrid);
    LayoutAwareAccessor imagAccessor(imagGrid);
    
    if (z == 0 && realGrid.getDefinition().zDimension.coordinates.empty()) {
        return interpolate2DComplex(realGrid, imagGrid, x, y, realAccessor, imagAccessor);
    } else {
        return interpolate3DComplex(realGrid, imagGrid, x, y, z, realAccessor, imagAccessor);
    }
}

std::vector<std::complex<double>> ComplexFieldInterpolator::interpolateComplexBatch(
    const GridData& realGrid,
    const GridData& imagGrid,
    const std::vector<TargetPoint>& points) const {
    
    // 如果支持AVX2且点数足够多，使用SIMD优化版本
    if (g_cpuCaps.hasAVX2 && points.size() >= 4) {
        return interpolateComplexBatchSIMD(realGrid, imagGrid, points);
    }
    
    // 标量版本
    std::vector<std::complex<double>> results;
    results.reserve(points.size());
    
    LayoutAwareAccessor realAccessor(realGrid);
    LayoutAwareAccessor imagAccessor(imagGrid);
    
    for (const auto& point : points) {
        double x = point.coordinates[0];
        double y = point.coordinates.size() > 1 ? point.coordinates[1] : 0;
        double z = point.coordinates.size() > 2 ? point.coordinates[2] : 0;
        
        if (point.coordinates.size() == 2) {
            results.push_back(interpolate2DComplex(realGrid, imagGrid, x, y, 
                                                  realAccessor, imagAccessor));
        } else {
            results.push_back(interpolate3DComplex(realGrid, imagGrid, x, y, z,
                                                  realAccessor, imagAccessor));
        }
    }
    
    return results;
}

/**
 * @brief SIMD优化的批量复数插值
 */
std::vector<std::complex<double>> ComplexFieldInterpolator::interpolateComplexBatchSIMD(
    const GridData& realGrid,
    const GridData& imagGrid,
    const std::vector<TargetPoint>& points) const {
    
    std::vector<std::complex<double>> results;
    results.reserve(points.size());
    
    const auto& def = realGrid.getDefinition();
    double xMin = def.xDimension.minValue;
    double xMax = def.xDimension.maxValue;
    double yMin = def.yDimension.minValue;
    double yMax = def.yDimension.maxValue;
    double xScale = (def.xDimension.coordinates.size() - 1) / (xMax - xMin);
    double yScale = (def.yDimension.coordinates.size() - 1) / (yMax - yMin);
    
#ifdef __AVX2__
    // AVX2优化路径：一次处理4个点
    const size_t simdWidth = 4;
    const size_t numFullBatches = points.size() / simdWidth;
    
    // 准备AVX2常量
    __m256d vXMin = _mm256_set1_pd(xMin);
    __m256d vYMin = _mm256_set1_pd(yMin);
    __m256d vXScale = _mm256_set1_pd(xScale);
    __m256d vYScale = _mm256_set1_pd(yScale);
    __m256d vMaxX = _mm256_set1_pd(def.xDimension.coordinates.size() - 1.0);
    __m256d vMaxY = _mm256_set1_pd(def.yDimension.coordinates.size() - 1.0);
    __m256d vZero = _mm256_setzero_pd();
    __m256d vOne = _mm256_set1_pd(1.0);
    
    // 对齐的临时数组
    alignas(32) double xCoords[4];
    alignas(32) double yCoords[4];
    alignas(32) double gridXs[4];
    alignas(32) double gridYs[4];
    alignas(32) double realResults[4];
    alignas(32) double imagResults[4];
    
    // 批量处理
    for (size_t batch = 0; batch < numFullBatches; ++batch) {
        size_t baseIdx = batch * simdWidth;
        
        // 收集4个点的坐标
        for (size_t i = 0; i < simdWidth; ++i) {
            const auto& point = points[baseIdx + i];
            xCoords[i] = point.coordinates[0];
            yCoords[i] = point.coordinates.size() > 1 ? point.coordinates[1] : 0;
        }
        
        // 加载坐标到AVX2寄存器
        __m256d vX = _mm256_load_pd(xCoords);
        __m256d vY = _mm256_load_pd(yCoords);
        
        // 转换到网格坐标
        __m256d vGridX = _mm256_mul_pd(_mm256_sub_pd(vX, vXMin), vXScale);
        __m256d vGridY = _mm256_mul_pd(_mm256_sub_pd(vY, vYMin), vYScale);
        
        // 边界钳制
        vGridX = _mm256_min_pd(_mm256_max_pd(vGridX, vZero), vMaxX);
        vGridY = _mm256_min_pd(_mm256_max_pd(vGridY, vZero), vMaxY);
        
        // 存储网格坐标
        _mm256_store_pd(gridXs, vGridX);
        _mm256_store_pd(gridYs, vGridY);
        
        // 对每个点进行插值（这部分仍然是标量，但坐标转换已经向量化）
        for (size_t i = 0; i < simdWidth; ++i) {
            // 检查原始坐标是否在边界内
            const auto& point = points[baseIdx + i];
            if (point.coordinates[0] < xMin || point.coordinates[0] > xMax ||
                (point.coordinates.size() > 1 && 
                 (point.coordinates[1] < yMin || point.coordinates[1] > yMax))) {
                // 越界点返回0
                realResults[i] = 0.0;
                imagResults[i] = 0.0;
            } else {
                // 双线性插值实部和虚部
                double real = interpolate2DValueSIMD(realGrid, gridXs[i], gridYs[i], 0);
                double imag = interpolate2DValueSIMD(imagGrid, gridXs[i], gridYs[i], 0);
                realResults[i] = real;
                imagResults[i] = imag;
            }
        }
        
        // 存储结果
        for (size_t i = 0; i < simdWidth; ++i) {
            results.emplace_back(realResults[i], imagResults[i]);
        }
    }
    
    // 处理剩余的点
    LayoutAwareAccessor realAccessor(realGrid);
    LayoutAwareAccessor imagAccessor(imagGrid);
    
    for (size_t i = numFullBatches * simdWidth; i < points.size(); ++i) {
        const auto& point = points[i];
        double x = point.coordinates[0];
        double y = point.coordinates.size() > 1 ? point.coordinates[1] : 0;
        
        results.push_back(interpolate2DComplex(realGrid, imagGrid, x, y,
                                              realAccessor, imagAccessor));
    }
#else
    // 非AVX2路径，使用标量版本
    LayoutAwareAccessor realAccessor(realGrid);
    LayoutAwareAccessor imagAccessor(imagGrid);
    
    for (const auto& point : points) {
        double x = point.coordinates[0];
        double y = point.coordinates.size() > 1 ? point.coordinates[1] : 0;
        
        results.push_back(interpolate2DComplex(realGrid, imagGrid, x, y,
                                              realAccessor, imagAccessor));
    }
#endif
    
    return results;
}

/**
 * @brief SIMD优化的2D值插值
 */
double ComplexFieldInterpolator::interpolate2DValueSIMD(
    const GridData& grid,
    double gridX, double gridY,
    size_t band) const {
    
    // 边界检查
    const auto& def = grid.getDefinition();
    if (gridX < 0 || gridX >= def.xDimension.coordinates.size() - 1 ||
        gridY < 0 || gridY >= def.yDimension.coordinates.size() - 1) {
        return 0.0;
    }
    
    // 计算整数索引
    int x0 = static_cast<int>(gridX);
    int y0 = static_cast<int>(gridY);
    int x1 = std::min(x0 + 1, static_cast<int>(def.xDimension.coordinates.size() - 1));
    int y1 = std::min(y0 + 1, static_cast<int>(def.yDimension.coordinates.size() - 1));
    
    // 计算插值权重
    double fx = gridX - x0;
    double fy = gridY - y0;
    
    // 获取四个角点的值
    const double* data = static_cast<const double*>(grid.getDataPtr());
    size_t stride = def.cols;
    
    double v00 = data[(y0 * stride + x0) * grid.getBandCount() + band];
    double v10 = data[(y0 * stride + x1) * grid.getBandCount() + band];
    double v01 = data[(y1 * stride + x0) * grid.getBandCount() + band];
    double v11 = data[(y1 * stride + x1) * grid.getBandCount() + band];
    
    // 双线性插值
    return (1 - fx) * (1 - fy) * v00 +
           fx * (1 - fy) * v10 +
           (1 - fx) * fy * v01 +
           fx * fy * v11;
}

std::complex<double> ComplexFieldInterpolator::interpolateInterleavedComplex(
    const GridData& complexGrid,
    double x, double y, double z) const {
    
    LayoutAwareAccessor accessor(complexGrid);
    const auto& def = complexGrid.getDefinition();
    
    // 转换到网格坐标
    double gridX = (x - def.xDimension.minValue) /
                  (def.xDimension.maxValue - def.xDimension.minValue) *
                  (def.xDimension.coordinates.size() - 1);
    
    double gridY = (y - def.yDimension.minValue) /
                  (def.yDimension.maxValue - def.yDimension.minValue) *
                  (def.yDimension.coordinates.size() - 1);
    
    // 边界检查
    if (gridX < 0 || gridX >= def.xDimension.coordinates.size() - 1 ||
        gridY < 0 || gridY >= def.yDimension.coordinates.size() - 1) {
        return std::complex<double>(0, 0);
    }
    
    // 计算索引
    int x0 = static_cast<int>(gridX);
    int y0 = static_cast<int>(gridY);
    int x1 = std::min(x0 + 1, static_cast<int>(def.xDimension.coordinates.size() - 1));
    int y1 = std::min(y0 + 1, static_cast<int>(def.yDimension.coordinates.size() - 1));
    
    double fx = gridX - x0;
    double fy = gridY - y0;
    
    // 获取四个角点的复数值（交错存储）
    auto getComplexValue = [&](int row, int col) -> std::complex<double> {
        double real = accessor.getValue<double>(row, col * 2, 0, 0);
        double imag = accessor.getValue<double>(row, col * 2 + 1, 0, 0);
        return std::complex<double>(real, imag);
    };
    
    auto v00 = getComplexValue(y0, x0);
    auto v10 = getComplexValue(y0, x1);
    auto v01 = getComplexValue(y1, x0);
    auto v11 = getComplexValue(y1, x1);
    
    // 双线性插值
    return bilinearInterpolate(v00, v10, v01, v11, fx, fy);
}

std::complex<double> ComplexFieldInterpolator::interpolate2DComplex(
    const GridData& realGrid,
    const GridData& imagGrid,
    double x, double y,
    const LayoutAwareAccessor& realAccessor,
    const LayoutAwareAccessor& imagAccessor) const {
    
    const auto& def = realGrid.getDefinition();
    
    // 转换到网格坐标
    double gridX = (x - def.xDimension.minValue) /
                  (def.xDimension.maxValue - def.xDimension.minValue) *
                  (def.xDimension.coordinates.size() - 1);
    
    double gridY = (y - def.yDimension.minValue) /
                  (def.yDimension.maxValue - def.yDimension.minValue) *
                  (def.yDimension.coordinates.size() - 1);
    
    double real = 0, imag = 0;
    
    switch (baseMethod_) {
        case InterpolationMethod::BILINEAR: {
            real = interpolate2DValue(realGrid, gridX, gridY, 0, realAccessor);
            imag = interpolate2DValue(imagGrid, gridX, gridY, 0, imagAccessor);
            break;
        }
        
        case InterpolationMethod::PCHIP_FAST_2D: {
            real = pchipInterpolate2D(realGrid, gridX, gridY, realAccessor);
            imag = pchipInterpolate2D(imagGrid, gridX, gridY, imagAccessor);
            break;
        }
        
        default: {
            // 默认使用双线性
            real = interpolate2DValue(realGrid, gridX, gridY, 0, realAccessor);
            imag = interpolate2DValue(imagGrid, gridX, gridY, 0, imagAccessor);
        }
    }
    
    return std::complex<double>(real, imag);
}

std::complex<double> ComplexFieldInterpolator::interpolate3DComplex(
    const GridData& realGrid,
    const GridData& imagGrid,
    double x, double y, double z,
    const LayoutAwareAccessor& realAccessor,
    const LayoutAwareAccessor& imagAccessor) const {
    
    // TODO: 实现3D复数插值
    return std::complex<double>(0, 0);
}

double ComplexFieldInterpolator::pchipInterpolate2D(
    const GridData& grid,
    double x, double y,
    const LayoutAwareAccessor& accessor) const {
    
    // TODO: 实现2D PCHIP插值
    // 暂时使用双线性插值
    return interpolate2DValue(grid, x, y, 0, accessor);
}

// RAMFieldAdapter实现
std::pair<std::shared_ptr<GridData>, std::shared_ptr<GridData>>
RAMFieldAdapter::createFromRAMField(
    const std::vector<std::complex<double>>& pressureField,
    const std::vector<double>& ranges,
    const std::vector<double>& depths) {
    
    // 创建网格定义
    GridDefinition def;
    def.rows = depths.size();
    def.cols = ranges.size();
    
    // 设置维度信息
    def.xDimension.coordinates = ranges;
    def.xDimension.minValue = *std::min_element(ranges.begin(), ranges.end());
    def.xDimension.maxValue = *std::max_element(ranges.begin(), ranges.end());
    def.xDimension.name = "range";
    def.xDimension.units = "m";
    
    def.yDimension.coordinates = depths;
    def.yDimension.minValue = *std::min_element(depths.begin(), depths.end());
    def.yDimension.maxValue = *std::max_element(depths.begin(), depths.end());
    def.yDimension.name = "depth";
    def.yDimension.units = "m";
    
    // 创建实部和虚部网格
    auto realGrid = std::make_shared<GridData>(def, DataType::Float64, 1);
    auto imagGrid = std::make_shared<GridData>(def, DataType::Float64, 1);
    
    // 填充数据
    auto* realData = static_cast<double*>(const_cast<void*>(realGrid->getDataPtr()));
    auto* imagData = static_cast<double*>(const_cast<void*>(imagGrid->getDataPtr()));
    
    for (size_t i = 0; i < pressureField.size(); ++i) {
        realData[i] = pressureField[i].real();
        imagData[i] = pressureField[i].imag();
    }
    
    return {realGrid, imagGrid};
}

std::shared_ptr<GridData> RAMFieldAdapter::createInterleavedComplexGrid(
    const std::vector<std::complex<double>>& pressureField,
    const std::vector<double>& ranges,
    const std::vector<double>& depths) {
    
    // 创建网格定义（列数翻倍用于存储实部和虚部）
    GridDefinition def;
    def.rows = depths.size();
    def.cols = ranges.size() * 2;  // 交错存储
    
    // 设置维度信息
    def.xDimension.coordinates.reserve(ranges.size() * 2);
    for (size_t i = 0; i < ranges.size(); ++i) {
        def.xDimension.coordinates.push_back(ranges[i]);
        def.xDimension.coordinates.push_back(ranges[i]);  // 重复，表示实部和虚部
    }
    def.xDimension.minValue = *std::min_element(ranges.begin(), ranges.end());
    def.xDimension.maxValue = *std::max_element(ranges.begin(), ranges.end());
    
    def.yDimension = def.yDimension;
    def.yDimension.coordinates = depths;
    def.yDimension.minValue = *std::min_element(depths.begin(), depths.end());
    def.yDimension.maxValue = *std::max_element(depths.begin(), depths.end());
    
    // 创建网格
    auto grid = std::make_shared<GridData>(def, DataType::Float64, 1);
    auto* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
    
    // 填充交错数据
    size_t idx = 0;
    for (size_t d = 0; d < depths.size(); ++d) {
        for (size_t r = 0; r < ranges.size(); ++r) {
            size_t fieldIdx = d * ranges.size() + r;
            data[idx++] = pressureField[fieldIdx].real();
            data[idx++] = pressureField[fieldIdx].imag();
        }
    }
    
    return grid;
}

double ComplexFieldInterpolator::interpolate2DValue(
    const GridData& grid,
    double gridX, double gridY,
    size_t band,
    const LayoutAwareAccessor& accessor) const {
    
    // 边界检查
    const auto& def = grid.getDefinition();
    if (gridX < 0 || gridX >= def.xDimension.coordinates.size() - 1 ||
        gridY < 0 || gridY >= def.yDimension.coordinates.size() - 1) {
        return 0.0;
    }
    
    // 计算整数索引
    int x0 = static_cast<int>(gridX);
    int y0 = static_cast<int>(gridY);
    int x1 = std::min(x0 + 1, static_cast<int>(def.xDimension.coordinates.size() - 1));
    int y1 = std::min(y0 + 1, static_cast<int>(def.yDimension.coordinates.size() - 1));
    
    // 计算插值权重
    double fx = gridX - x0;
    double fy = gridY - y0;
    
    // 获取四个角点的值
    double v00 = accessor.getValue<double>(x0, y0, 0, band);
    double v10 = accessor.getValue<double>(x1, y0, 0, band);
    double v01 = accessor.getValue<double>(x0, y1, 0, band);
    double v11 = accessor.getValue<double>(x1, y1, 0, band);
    
    // 双线性插值
    return bilinearInterpolate(v00, v10, v01, v11, fx, fy);
}

} // namespace oscean::core_services::interpolation 