#include "cubic_spline_interpolator.h"
#include "kernels/interpolation_kernels.h"
#include <cmath>
#include <algorithm>
#include <array>
#include <variant>
#include <vector>

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

CubicSplineInterpolator::CubicSplineInterpolator(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
    : simdManager_(simdManager) {
}

InterpolationResult CubicSplineInterpolator::execute(
    const InterpolationRequest& request,
    const PrecomputedData* precomputed) const {
    
    InterpolationResult result;
    result.statusCode = -1; // 默认失败
    
    if (!request.sourceGrid) {
        result.message = "源网格数据为空";
        return result;
    }
    
    // 检查目标类型
    if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        // 点插值
        const auto& targetPoints = std::get<std::vector<TargetPoint>>(request.target);
        
        // 使用内联SIMD优化的批量插值
        std::vector<std::optional<double>> values;
        if (targetPoints.size() > 8 && g_cpuCaps.hasAVX2) {
            // 使用内联SIMD优化
            values = interpolateAtPointsSIMD(*request.sourceGrid, targetPoints);
        } else if (simdManager_ && targetPoints.size() > 4) {
            // 使用SIMD管理器
            values = simdBatchInterpolate(*request.sourceGrid, targetPoints);
        } else {
            // 标量实现
            values = interpolateAtPoints(*request.sourceGrid, targetPoints);
        }
        
        result.data = values;
        result.statusCode = 0;
        result.message = "立方样条插值成功完成";
    } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
        // 网格到网格插值
        const auto& targetGridDef = std::get<TargetGridDefinition>(request.target);
        try {
            auto gridResult = interpolateToGrid(*request.sourceGrid, targetGridDef);
            result.data = std::move(gridResult);
            result.statusCode = 0;
            result.message = "网格到网格立方样条插值成功完成";
        } catch (const std::exception& e) {
            result.message = std::string("网格到网格插值失败: ") + e.what();
            return result;
        }
    } else {
        result.message = "未知的目标类型";
        return result;
    }
    
    return result;
}

std::vector<std::optional<double>> CubicSplineInterpolator::interpolateAtPoints(
    const GridData& sourceGrid,
    const std::vector<TargetPoint>& targetPoints) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());
    
    for (const auto& point : targetPoints) {
        double worldX = point.coordinates.size() > 0 ? point.coordinates[0] : 0.0;
        double worldY = point.coordinates.size() > 1 ? point.coordinates[1] : 0.0;
        auto result = interpolateAtPoint(sourceGrid, worldX, worldY);
        results.push_back(result);
    }
    
    return results;
}

GridData CubicSplineInterpolator::interpolateToGrid(
    const GridData& sourceGrid,
    const TargetGridDefinition& targetGridDef) const {
    
    // 创建目标网格定义
    GridDefinition targetDef;
    targetDef.cols = 0;
    targetDef.rows = 0;
    
    // 从目标网格定义中提取维度信息
    for (const auto& dim : targetGridDef.dimensions) {
        if (dim.name == "x" || dim.name == "longitude") {
            targetDef.cols = dim.getNumberOfLevels();
        } else if (dim.name == "y" || dim.name == "latitude") {
            targetDef.rows = dim.getNumberOfLevels();
        }
    }
    
    if (targetDef.cols == 0 || targetDef.rows == 0) {
        throw std::runtime_error("无效的目标网格维度");
    }
    
    // 创建目标网格数据
    size_t bandCount = sourceGrid.getBandCount();
    std::vector<double> targetData(targetDef.cols * targetDef.rows * bandCount);
    
    // 计算目标网格的地理变换参数
    std::vector<double> targetGeoTransform(6);
    
    // 获取目标网格的坐标范围
    auto xDim = std::find_if(targetGridDef.dimensions.begin(), targetGridDef.dimensions.end(),
        [](const auto& dim) { return dim.name == "x" || dim.name == "longitude"; });
    auto yDim = std::find_if(targetGridDef.dimensions.begin(), targetGridDef.dimensions.end(),
        [](const auto& dim) { return dim.name == "y" || dim.name == "latitude"; });
    
    if (xDim != targetGridDef.dimensions.end() && yDim != targetGridDef.dimensions.end()) {
        double xMin = xDim->coordinates.front();
        double xMax = xDim->coordinates.back();
        double yMin = yDim->coordinates.front();
        double yMax = yDim->coordinates.back();
        
        targetGeoTransform[0] = xMin;  // 左上角X坐标
        targetGeoTransform[1] = (xMax - xMin) / (targetDef.cols - 1);  // X方向像素大小
        targetGeoTransform[2] = 0.0;   // X方向旋转
        targetGeoTransform[3] = yMax;  // 左上角Y坐标
        targetGeoTransform[4] = 0.0;   // Y方向旋转
        targetGeoTransform[5] = -(yMax - yMin) / (targetDef.rows - 1);  // Y方向像素大小（负值）
    }
    
    // 执行立方样条插值
    for (size_t row = 0; row < targetDef.rows; ++row) {
        for (size_t col = 0; col < targetDef.cols; ++col) {
            // 计算目标点的世界坐标
            double worldX = targetGeoTransform[0] + col * targetGeoTransform[1];
            double worldY = targetGeoTransform[3] + row * targetGeoTransform[5];
            
            // 执行立方样条插值
            auto interpolatedValue = interpolateAtPoint(sourceGrid, worldX, worldY);
            
            for (size_t band = 0; band < bandCount; ++band) {
                size_t index = band * targetDef.rows * targetDef.cols + row * targetDef.cols + col;
                if (interpolatedValue.has_value()) {
                    targetData[index] = interpolatedValue.value();
                } else {
                    targetData[index] = targetGridDef.fillValue.value_or(std::numeric_limits<double>::quiet_NaN());
                }
            }
        }
    }
    
    // 创建并返回GridData对象
    GridData result(targetDef, DataType::Float64, bandCount);
    result.setGeoTransform(targetGeoTransform);
    result.setCrs(targetGridDef.crs);
    
    // 复制数据
    std::memcpy(const_cast<void*>(result.getDataPtr()), targetData.data(), 
                targetData.size() * sizeof(double));
    
    return result;
}

std::optional<double> CubicSplineInterpolator::interpolateAtPoint(
    const GridData& grid, 
    double worldX, 
    double worldY) const {
    
    const auto& def = grid.getDefinition();
    const auto& geoTransform = grid.getGeoTransform();
    
    // 检查地理变换是否有效
    if (geoTransform.size() < 6) {
        return std::nullopt;
    }
    
    // 将世界坐标转换为网格坐标
    double gridX = (worldX - geoTransform[0]) / geoTransform[1];
    double gridY = (worldY - geoTransform[3]) / geoTransform[5];
    
    // 检查边界（立方样条需要更大的边界）
    if (gridX < 1 || gridX >= def.cols - 2 || 
        gridY < 1 || gridY >= def.rows - 2) {
        // 边界区域使用双线性插值
        return fallbackBilinearInterpolation(grid, gridX, gridY);
    }
    
    // 获取16个控制点的整数坐标（4x4网格）
    int x0 = static_cast<int>(std::floor(gridX)) - 1;
    int y0 = static_cast<int>(std::floor(gridY)) - 1;
    
    // 获取16个控制点的值
    std::array<double, 16> values;
    bool allValid = true;
    
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            auto value = getGridValue(grid, x0 + i, y0 + j);
            if (value.has_value()) {
                values[j * 4 + i] = value.value();
            } else {
                allValid = false;
                break;
            }
        }
        if (!allValid) break;
    }
    
    if (!allValid) {
        // 如果有无效值，回退到双线性插值
        return fallbackBilinearInterpolation(grid, gridX, gridY);
    }
    
    // 计算插值权重
    double fx = gridX - std::floor(gridX);
    double fy = gridY - std::floor(gridY);
    
    // 使用双三次插值内核
    return kernels::bicubic(values, fx, fy);
}

std::optional<double> CubicSplineInterpolator::fallbackBilinearInterpolation(
    const GridData& grid, 
    double gridX, 
    double gridY) const {
    
    const auto& def = grid.getDefinition();
    
    // 边界检查
    if (gridX < 0 || gridX >= def.cols - 1 || 
        gridY < 0 || gridY >= def.rows - 1) {
        return std::nullopt;
    }
    
    // 获取四个角点的整数坐标
    int x0 = static_cast<int>(std::floor(gridX));
    int y0 = static_cast<int>(std::floor(gridY));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // 获取四个角点的值
    auto v00 = getGridValue(grid, x0, y0);
    auto v10 = getGridValue(grid, x1, y0);
    auto v01 = getGridValue(grid, x0, y1);
    auto v11 = getGridValue(grid, x1, y1);
    
    // 检查所有角点值是否有效
    if (!v00.has_value() || !v10.has_value() || !v01.has_value() || !v11.has_value()) {
        return std::nullopt;
    }
    
    // 计算插值权重
    double fx = gridX - x0;
    double fy = gridY - y0;
    
    // 使用双线性插值
    std::array<double, 4> values = {v00.value(), v10.value(), v01.value(), v11.value()};
    return kernels::bilinear(values, fx, fy);
}

std::optional<double> CubicSplineInterpolator::getGridValue(
    const GridData& grid,
    int col, int row, int band) const {
    
    const auto& def = grid.getDefinition();
    
    // 边界检查
    if (col < 0 || col >= static_cast<int>(def.cols) || 
        row < 0 || row >= static_cast<int>(def.rows) || 
        band < 0 || band >= static_cast<int>(grid.getBandCount())) {
        return std::nullopt;
    }
    
    try {
        // 根据数据类型获取值
        switch (grid.getDataType()) {
            case DataType::Float32:
                return static_cast<double>(grid.getValue<float>(static_cast<size_t>(row), 
                                                              static_cast<size_t>(col), 
                                                              static_cast<size_t>(band)));
            case DataType::Float64:
                return grid.getValue<double>(static_cast<size_t>(row), 
                                           static_cast<size_t>(col), 
                                           static_cast<size_t>(band));
            case DataType::Int16:
                return static_cast<double>(grid.getValue<int16_t>(static_cast<size_t>(row), 
                                                                 static_cast<size_t>(col), 
                                                                 static_cast<size_t>(band)));
            case DataType::Int32:
                return static_cast<double>(grid.getValue<int32_t>(static_cast<size_t>(row), 
                                                                 static_cast<size_t>(col), 
                                                                 static_cast<size_t>(band)));
            case DataType::UInt16:
                return static_cast<double>(grid.getValue<uint16_t>(static_cast<size_t>(row), 
                                                                  static_cast<size_t>(col), 
                                                                  static_cast<size_t>(band)));
            case DataType::UInt32:
                return static_cast<double>(grid.getValue<uint32_t>(static_cast<size_t>(row), 
                                                                  static_cast<size_t>(col), 
                                                                  static_cast<size_t>(band)));
            case DataType::Byte:
                return static_cast<double>(grid.getValue<uint8_t>(static_cast<size_t>(row), 
                                                                 static_cast<size_t>(col), 
                                                                 static_cast<size_t>(band)));
            default:
                return std::nullopt;
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<std::optional<double>> CubicSplineInterpolator::simdBatchInterpolate(
    const GridData& grid,
    const std::vector<TargetPoint>& points) const {
    
    if (!simdManager_) {
        return interpolateAtPoints(grid, points);
    }
    
    std::vector<std::optional<double>> results;
    results.reserve(points.size());
    
    const auto& def = grid.getDefinition();
    const auto& geoTransform = grid.getGeoTransform();
    
    // 检查地理变换是否有效
    if (geoTransform.size() < 6) {
        return interpolateAtPoints(grid, points);
    }
    
    // SIMD批量处理
    const size_t batchSize = simdManager_->getOptimalBatchSize();
    const size_t numBatches = (points.size() + batchSize - 1) / batchSize;
    
    for (size_t batch = 0; batch < numBatches; ++batch) {
        size_t startIdx = batch * batchSize;
        size_t endIdx = std::min(startIdx + batchSize, points.size());
        size_t currentBatchSize = endIdx - startIdx;
        
        // 预分配SIMD对齐的内存缓冲区
        std::vector<float> xCoords, yCoords;
        std::vector<float> gridXCoords, gridYCoords;
        std::vector<float> interpolatedValues;
        
        xCoords.reserve(currentBatchSize);
        yCoords.reserve(currentBatchSize);
        gridXCoords.reserve(currentBatchSize);
        gridYCoords.reserve(currentBatchSize);
        interpolatedValues.resize(currentBatchSize);
        
        // 准备SIMD输入数据
        for (size_t i = startIdx; i < endIdx; ++i) {
            const auto& point = points[i];
            float worldX = point.coordinates.size() > 0 ? static_cast<float>(point.coordinates[0]) : 0.0f;
            float worldY = point.coordinates.size() > 1 ? static_cast<float>(point.coordinates[1]) : 0.0f;
            
            xCoords.push_back(worldX);
            yCoords.push_back(worldY);
            
            // 将世界坐标转换为网格坐标
            float gridX = (worldX - static_cast<float>(geoTransform[0])) / static_cast<float>(geoTransform[1]);
            float gridY = (worldY - static_cast<float>(geoTransform[3])) / static_cast<float>(geoTransform[5]);
            
            gridXCoords.push_back(gridX);
            gridYCoords.push_back(gridY);
        }
        
        try {
            // 使用SIMD管理器执行批量双三次插值
            simdManager_->bicubicInterpolate(
                reinterpret_cast<const float*>(grid.getDataPtr()),
                gridXCoords.data(),
                gridYCoords.data(),
                interpolatedValues.data(),
                currentBatchSize,
                def.cols,
                def.rows
            );
            
            // 处理结果
            for (size_t i = 0; i < currentBatchSize; ++i) {
                float value = interpolatedValues[i];
                if (std::isfinite(value)) {
                    results.push_back(static_cast<double>(value));
                } else {
                    results.push_back(std::nullopt);
                }
            }
            
        } catch (const std::exception& e) {
            // SIMD失败时回退到标量计算
            for (size_t i = startIdx; i < endIdx; ++i) {
                const auto& point = points[i];
                double worldX = point.coordinates.size() > 0 ? point.coordinates[0] : 0.0;
                double worldY = point.coordinates.size() > 1 ? point.coordinates[1] : 0.0;
                results.push_back(interpolateAtPoint(grid, worldX, worldY));
            }
        }
    }
    
    return results;
}

// 新增：内联SIMD优化的批量插值实现
std::vector<std::optional<double>> CubicSplineInterpolator::interpolateAtPointsSIMD(
    const GridData& sourceGrid,
    const std::vector<TargetPoint>& targetPoints) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());
    
    const auto& def = sourceGrid.getDefinition();
    const auto& geoTransform = sourceGrid.getGeoTransform();
    
    if (geoTransform.size() < 6) {
        return interpolateAtPoints(sourceGrid, targetPoints);
    }
    
    const int width = static_cast<int>(def.cols);
    const int height = static_cast<int>(def.rows);
    const double originX = geoTransform[0];
    const double dx = geoTransform[1];
    const double originY = geoTransform[3];
    const double dy = geoTransform[5];
    
    DataType dataType = sourceGrid.getDataType();
    
    #ifdef __AVX2__
    if (g_cpuCaps.hasAVX2 && dataType == DataType::Float32) {
        // AVX2优化路径（处理float类型）
        const float* dataPtr = static_cast<const float*>(sourceGrid.getDataPtr());
        
        size_t i = 0;
        // 处理4个点一组（由于立方样条需要4x4=16个点，复杂度较高）
        for (; i + 3 < targetPoints.size(); i += 4) {
            // 加载4个点的坐标
            alignas(32) float x_arr[4], y_arr[4];
            for (int j = 0; j < 4; ++j) {
                x_arr[j] = static_cast<float>(targetPoints[i+j].coordinates[0]);
                y_arr[j] = static_cast<float>(targetPoints[i+j].coordinates[1]);
            }
            
            __m128 x_coords = _mm_load_ps(x_arr);
            __m128 y_coords = _mm_load_ps(y_arr);
            
            // 转换到网格坐标
            __m128 origin_x = _mm_set1_ps(static_cast<float>(originX));
            __m128 dx_vec = _mm_set1_ps(static_cast<float>(dx));
            __m128 origin_y = _mm_set1_ps(static_cast<float>(originY));
            __m128 dy_vec = _mm_set1_ps(static_cast<float>(dy));
            
            __m128 grid_x = _mm_div_ps(_mm_sub_ps(x_coords, origin_x), dx_vec);
            __m128 grid_y = _mm_div_ps(_mm_sub_ps(y_coords, origin_y), dy_vec);
            
            // 提取网格坐标
            alignas(16) float grid_x_arr[4], grid_y_arr[4];
            _mm_store_ps(grid_x_arr, grid_x);
            _mm_store_ps(grid_y_arr, grid_y);
            
            // 处理每个点（立方样条需要逐点处理，因为需要访问16个点）
            for (int j = 0; j < 4; ++j) {
                float gx = grid_x_arr[j];
                float gy = grid_y_arr[j];
                
                // 边界检查
                if (gx < 1 || gx >= width - 2 || gy < 1 || gy >= height - 2) {
                    // 边界使用双线性插值
                    results.push_back(fallbackBilinearInterpolation(sourceGrid, gx, gy));
                    continue;
                }
                
                int x0 = static_cast<int>(std::floor(gx)) - 1;
                int y0 = static_cast<int>(std::floor(gy)) - 1;
                float fx = gx - std::floor(gx);
                float fy = gy - std::floor(gy);
                
                // 使用SIMD加速16个点的数据收集
                alignas(32) float values[16];
                
                // 一次加载4行数据
                for (int row = 0; row < 4; ++row) {
                    int idx = (y0 + row) * width + x0;
                    __m128 row_data = _mm_loadu_ps(&dataPtr[idx]);
                    _mm_store_ps(&values[row * 4], row_data);
                }
                
                // 执行双三次插值（可以进一步SIMD优化）
                double result = bicubicInterpolateSIMD(values, fx, fy);
                results.push_back(result);
            }
        }
        
        // 处理剩余的点
        for (; i < targetPoints.size(); ++i) {
            results.push_back(interpolateAtPoint(sourceGrid, 
                targetPoints[i].coordinates[0], 
                targetPoints[i].coordinates[1]));
        }
        
        return results;
    }
    #endif
    
    // 非SIMD路径或不支持的数据类型
    return interpolateAtPoints(sourceGrid, targetPoints);
}

// 新增：SIMD优化的双三次插值核心计算
double CubicSplineInterpolator::bicubicInterpolateSIMD(
    const float values[16], float fx, float fy) const {
    
    #ifdef __AVX2__
    if (g_cpuCaps.hasAVX2) {
        // 使用AVX2加速立方插值权重计算
        __m128 fx_vec = _mm_set1_ps(fx);
        __m128 fy_vec = _mm_set1_ps(fy);
        
        // 计算水平方向的立方权重
        __m128 wx = computeCubicWeightsSIMD(fx_vec);
        
        // 对每行进行水平插值
        alignas(16) float row_results[4];
        for (int i = 0; i < 4; ++i) {
            __m128 row_values = _mm_load_ps(&values[i * 4]);
            __m128 weighted = _mm_mul_ps(row_values, wx);
            
            // 水平求和
            __m128 sum1 = _mm_hadd_ps(weighted, weighted);
            __m128 sum2 = _mm_hadd_ps(sum1, sum1);
            row_results[i] = _mm_cvtss_f32(sum2);
        }
        
        // 垂直方向插值
        __m128 col_values = _mm_load_ps(row_results);
        __m128 wy = computeCubicWeightsSIMD(fy_vec);
        __m128 final_weighted = _mm_mul_ps(col_values, wy);
        
        // 最终求和
        __m128 sum1 = _mm_hadd_ps(final_weighted, final_weighted);
        __m128 sum2 = _mm_hadd_ps(sum1, sum1);
        
        return static_cast<double>(_mm_cvtss_f32(sum2));
    }
    #endif
    
    // 标量实现
    std::array<double, 16> dvalues;
    for (int i = 0; i < 16; ++i) {
        dvalues[i] = static_cast<double>(values[i]);
    }
    return kernels::bicubic(dvalues, fx, fy);
}

// 新增：SIMD计算立方插值权重
#ifdef __AVX2__
__m128 CubicSplineInterpolator::computeCubicWeightsSIMD(__m128 t) const {
    // Catmull-Rom样条权重计算
    // w0 = -0.5t³ + t² - 0.5t
    // w1 = 1.5t³ - 2.5t² + 1
    // w2 = -1.5t³ + 2t² + 0.5t
    // w3 = 0.5t³ - 0.5t²
    
    __m128 t2 = _mm_mul_ps(t, t);
    __m128 t3 = _mm_mul_ps(t2, t);
    
    __m128 half = _mm_set1_ps(0.5f);
    __m128 one = _mm_set1_ps(1.0f);
    __m128 one_half = _mm_set1_ps(1.5f);
    __m128 two = _mm_set1_ps(2.0f);
    __m128 two_half = _mm_set1_ps(2.5f);
    
    // w0 = -0.5t³ + t² - 0.5t
    __m128 w0 = _mm_sub_ps(t2, _mm_mul_ps(half, t3));
    w0 = _mm_sub_ps(w0, _mm_mul_ps(half, t));
    
    // w1 = 1.5t³ - 2.5t² + 1
    __m128 w1 = _mm_sub_ps(_mm_mul_ps(one_half, t3), _mm_mul_ps(two_half, t2));
    w1 = _mm_add_ps(w1, one);
    
    // w2 = -1.5t³ + 2t² + 0.5t
    __m128 w2 = _mm_add_ps(_mm_mul_ps(two, t2), _mm_mul_ps(half, t));
    w2 = _mm_sub_ps(w2, _mm_mul_ps(one_half, t3));
    
    // w3 = 0.5t³ - 0.5t²
    __m128 w3 = _mm_sub_ps(_mm_mul_ps(half, t3), _mm_mul_ps(half, t2));
    
    // 将4个权重打包成一个向量
    __m128 weights = _mm_set_ps(
        _mm_cvtss_f32(w3),
        _mm_cvtss_f32(w2),
        _mm_cvtss_f32(w1),
        _mm_cvtss_f32(w0)
    );
    
    return weights;
}
#endif

} // namespace oscean::core_services::interpolation