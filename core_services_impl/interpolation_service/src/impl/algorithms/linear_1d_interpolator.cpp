#include "linear_1d_interpolator.h"
#include "kernels/interpolation_kernels.h"
#include <cmath>
#include <algorithm>
#include <variant>

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

Linear1DInterpolator::Linear1DInterpolator(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
    : simdManager_(simdManager) {
}

InterpolationResult Linear1DInterpolator::execute(
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
        
        // 根据数据大小和CPU能力选择最优实现
        std::vector<std::optional<double>> values;
        
        if (targetPoints.size() >= 8 && g_cpuCaps.hasAVX2) {
            // 使用内联SIMD优化
            values = interpolateAtPointsSIMD(*request.sourceGrid, targetPoints);
        } else if (simdManager_ && targetPoints.size() > 4) {
            // 使用SIMD管理器（向后兼容）
            values = simdBatchInterpolate(*request.sourceGrid, targetPoints);
        } else {
            // 标量实现
            values = interpolateAtPoints(*request.sourceGrid, targetPoints);
        }
        
        result.data = values;
        result.statusCode = 0;
        result.message = "1D线性插值成功完成";
    } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
        // 网格到网格插值
        const auto& targetGridDef = std::get<TargetGridDefinition>(request.target);
        try {
            auto gridResult = interpolateToGrid(*request.sourceGrid, targetGridDef);
            result.data = std::move(gridResult);
            result.statusCode = 0;
            result.message = "网格到网格1D线性插值成功完成";
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

// 新增：内联SIMD优化的批量插值实现
std::vector<std::optional<double>> Linear1DInterpolator::interpolateAtPointsSIMD(
    const GridData& sourceGrid,
    const std::vector<TargetPoint>& targetPoints) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());
    
    const auto& def = sourceGrid.getDefinition();
    const auto& geoTransform = sourceGrid.getGeoTransform();
    
    // 检查地理变换是否有效
    if (geoTransform.size() < 6) {
        return interpolateAtPoints(sourceGrid, targetPoints);
    }
    
    // 预计算常用值
    const double originX = geoTransform[0];
    const double dx = geoTransform[1];
    const double originY = geoTransform[3];
    const double dy = geoTransform[5];
    const int width = static_cast<int>(def.cols);
    const int height = static_cast<int>(def.rows);
    
    // 根据数据类型选择优化路径
    DataType dataType = sourceGrid.getDataType();
    
    #ifdef __AVX2__
    if (g_cpuCaps.hasAVX2 && dataType == DataType::Float64) {
        // AVX2优化路径（处理double类型）
        const double* dataPtr = static_cast<const double*>(sourceGrid.getDataPtr());
        
        size_t i = 0;
        // 处理4个点一组（AVX2可以处理4个double）
        for (; i + 3 < targetPoints.size(); i += 4) {
            // 加载4个点的坐标
            __m256d x_coords = _mm256_set_pd(
                targetPoints[i+3].coordinates[0],
                targetPoints[i+2].coordinates[0],
                targetPoints[i+1].coordinates[0],
                targetPoints[i].coordinates[0]
            );
            
            __m256d y_coords = _mm256_set_pd(
                targetPoints[i+3].coordinates[1],
                targetPoints[i+2].coordinates[1],
                targetPoints[i+1].coordinates[1],
                targetPoints[i].coordinates[1]
            );
            
            // 转换到网格坐标
            __m256d origin_x = _mm256_set1_pd(originX);
            __m256d dx_vec = _mm256_set1_pd(dx);
            __m256d origin_y = _mm256_set1_pd(originY);
            __m256d dy_vec = _mm256_set1_pd(dy);
            
            __m256d grid_x = _mm256_div_pd(_mm256_sub_pd(x_coords, origin_x), dx_vec);
            __m256d grid_y = _mm256_div_pd(_mm256_sub_pd(y_coords, origin_y), dy_vec);
            
            // 提取整数部分和小数部分
            __m256d x0_d = _mm256_floor_pd(grid_x);
            __m256d y_rounded = _mm256_round_pd(grid_y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            
            __m256d fx = _mm256_sub_pd(grid_x, x0_d);
            
            // 转换为整数索引
            alignas(32) double x0_arr[4], y_arr[4], fx_arr[4];
            _mm256_store_pd(x0_arr, x0_d);
            _mm256_store_pd(y_arr, y_rounded);
            _mm256_store_pd(fx_arr, fx);
            
            // 处理每个点
            for (int j = 0; j < 4; ++j) {
                int x0 = static_cast<int>(x0_arr[j]);
                int y = static_cast<int>(y_arr[j]);
                
                // 边界检查
                if (x0 < 0 || x0 >= width - 1 || y < 0 || y >= height) {
                    results.push_back(std::nullopt);
                    continue;
                }
                
                int x1 = x0 + 1;
                
                // 获取两个点的值
                double v0 = dataPtr[y * width + x0];
                double v1 = dataPtr[y * width + x1];
                
                // 1D线性插值
                double weight = fx_arr[j];
                results.push_back(v0 * (1 - weight) + v1 * weight);
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
    else if (g_cpuCaps.hasAVX2 && dataType == DataType::Float32) {
        // AVX2优化路径（处理float类型，可以一次处理8个）
        const float* dataPtr = static_cast<const float*>(sourceGrid.getDataPtr());
        
        size_t i = 0;
        // 处理8个点一组
        for (; i + 7 < targetPoints.size(); i += 8) {
            alignas(32) float x_arr[8], y_arr[8];
            for (int j = 0; j < 8; ++j) {
                x_arr[j] = static_cast<float>(targetPoints[i+j].coordinates[0]);
                y_arr[j] = static_cast<float>(targetPoints[i+j].coordinates[1]);
            }
            
            __m256 x_coords = _mm256_load_ps(x_arr);
            __m256 y_coords = _mm256_load_ps(y_arr);
            
            // 转换到网格坐标
            __m256 origin_x = _mm256_set1_ps(static_cast<float>(originX));
            __m256 dx_vec = _mm256_set1_ps(static_cast<float>(dx));
            __m256 origin_y = _mm256_set1_ps(static_cast<float>(originY));
            __m256 dy_vec = _mm256_set1_ps(static_cast<float>(dy));
            
            __m256 grid_x = _mm256_div_ps(_mm256_sub_ps(x_coords, origin_x), dx_vec);
            __m256 grid_y = _mm256_div_ps(_mm256_sub_ps(y_coords, origin_y), dy_vec);
            
            // 提取整数部分和小数部分
            __m256 x0_f = _mm256_floor_ps(grid_x);
            __m256 y_rounded = _mm256_round_ps(grid_y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            
            __m256 fx = _mm256_sub_ps(grid_x, x0_f);
            
            // 转换为整数索引并处理
            alignas(32) float x0_arr[8], y_arr2[8], fx_arr[8];
            _mm256_store_ps(x0_arr, x0_f);
            _mm256_store_ps(y_arr2, y_rounded);
            _mm256_store_ps(fx_arr, fx);
            
            for (int j = 0; j < 8; ++j) {
                int x0 = static_cast<int>(x0_arr[j]);
                int y = static_cast<int>(y_arr2[j]);
                
                if (x0 < 0 || x0 >= width - 1 || y < 0 || y >= height) {
                    results.push_back(std::nullopt);
                    continue;
                }
                
                int idx0 = y * width + x0;
                int idx1 = idx0 + 1;
                
                float v0 = dataPtr[idx0];
                float v1 = dataPtr[idx1];
                
                float weight = fx_arr[j];
                results.push_back(static_cast<double>(v0 * (1 - weight) + v1 * weight));
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

std::vector<std::optional<double>> Linear1DInterpolator::interpolateAtPoints(
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

GridData Linear1DInterpolator::interpolateToGrid(
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
    
    // 执行1D线性插值
    for (size_t row = 0; row < targetDef.rows; ++row) {
        for (size_t col = 0; col < targetDef.cols; ++col) {
            // 计算目标点的世界坐标
            double worldX = targetGeoTransform[0] + col * targetGeoTransform[1];
            double worldY = targetGeoTransform[3] + row * targetGeoTransform[5];
            
            // 执行1D线性插值
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

std::optional<double> Linear1DInterpolator::interpolateAtPoint(
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
    
    // 检查边界
    if (gridX < 0 || gridX >= def.cols - 1 || 
        gridY < 0 || gridY >= def.rows - 1) {
        return std::nullopt;
    }
    
    // 1D线性插值：优先在X方向进行插值
    int x0 = static_cast<int>(std::floor(gridX));
    int x1 = x0 + 1;
    int y = static_cast<int>(std::round(gridY)); // 固定Y坐标到最近的整数
    
    // 如果Y坐标超出边界，尝试在Y方向进行1D插值
    if (y < 0 || y >= static_cast<int>(def.rows)) {
        return interpolate1DInYDirection(grid, gridX, gridY);
    }
    
    // 获取两个点的值（X方向）
    auto v0 = getGridValue(grid, x0, y);
    auto v1 = getGridValue(grid, x1, y);
    
    // 检查值是否有效
    if (!v0.has_value() || !v1.has_value()) {
        // 如果X方向插值失败，尝试Y方向
        return interpolate1DInYDirection(grid, gridX, gridY);
    }
    
    // 计算插值权重
    double fx = gridX - x0;
    
    // 使用1D线性插值
    return kernels::linear1D(v0.value(), v1.value(), fx);
}

std::optional<double> Linear1DInterpolator::interpolate1DInYDirection(
    const GridData& grid, 
    double gridX, 
    double gridY) const {
    
    const auto& def = grid.getDefinition();
    
    // 在Y方向进行1D线性插值
    int y0 = static_cast<int>(std::floor(gridY));
    int y1 = y0 + 1;
    int x = static_cast<int>(std::round(gridX)); // 固定X坐标到最近的整数
    
    // 边界检查
    if (x < 0 || x >= static_cast<int>(def.cols) ||
        y0 < 0 || y1 >= static_cast<int>(def.rows)) {
        return std::nullopt;
    }
    
    // 获取两个点的值（Y方向）
    auto v0 = getGridValue(grid, x, y0);
    auto v1 = getGridValue(grid, x, y1);
    
    // 检查值是否有效
    if (!v0.has_value() || !v1.has_value()) {
        return std::nullopt;
    }
    
    // 计算插值权重
    double fy = gridY - y0;
    
    // 使用1D线性插值
    return kernels::linear1D(v0.value(), v1.value(), fy);
}

std::optional<double> Linear1DInterpolator::getGridValue(
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
        DataType dataType = grid.getDataType();
        switch (dataType) {
            case DataType::Float32: {
                float value = grid.getValue<float>(static_cast<size_t>(row), 
                                                 static_cast<size_t>(col), 
                                                 static_cast<size_t>(band));
                return static_cast<double>(value);
            }
            case DataType::Float64: {
                return grid.getValue<double>(static_cast<size_t>(row), 
                                           static_cast<size_t>(col), 
                                           static_cast<size_t>(band));
            }
            case DataType::Int16: {
                int16_t value = grid.getValue<int16_t>(static_cast<size_t>(row), 
                                                      static_cast<size_t>(col), 
                                                      static_cast<size_t>(band));
                return static_cast<double>(value);
            }
            case DataType::Int32: {
                int32_t value = grid.getValue<int32_t>(static_cast<size_t>(row), 
                                                      static_cast<size_t>(col), 
                                                      static_cast<size_t>(band));
                return static_cast<double>(value);
            }
            case DataType::UInt16: {
                uint16_t value = grid.getValue<uint16_t>(static_cast<size_t>(row), 
                                                        static_cast<size_t>(col), 
                                                        static_cast<size_t>(band));
                return static_cast<double>(value);
            }
            case DataType::UInt32: {
                uint32_t value = grid.getValue<uint32_t>(static_cast<size_t>(row), 
                                                        static_cast<size_t>(col), 
                                                        static_cast<size_t>(band));
                return static_cast<double>(value);
            }
            case DataType::Byte: {
                uint8_t value = grid.getValue<uint8_t>(static_cast<size_t>(row), 
                                                      static_cast<size_t>(col), 
                                                      static_cast<size_t>(band));
                return static_cast<double>(value);
            }
            default:
                // 默认尝试使用double类型
                return grid.getValue<double>(static_cast<size_t>(row), 
                                           static_cast<size_t>(col), 
                                           static_cast<size_t>(band));
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<std::optional<double>> Linear1DInterpolator::simdBatchInterpolate(
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
        std::vector<float> weights;
        
        xCoords.reserve(currentBatchSize);
        yCoords.reserve(currentBatchSize);
        gridXCoords.reserve(currentBatchSize);
        gridYCoords.reserve(currentBatchSize);
        interpolatedValues.resize(currentBatchSize);
        weights.resize(currentBatchSize * 2); // 2个权重值（1D插值）
        
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
            
            // 计算1D线性插值权重
            float fx = gridX - std::floor(gridX);
            
            size_t weightIdx = (i - startIdx) * 2;
            weights[weightIdx + 0] = 1.0f - fx;
            weights[weightIdx + 1] = fx;
        }
        
        try {
            // 使用SIMD管理器执行批量线性插值
            simdManager_->linearInterpolate(
                reinterpret_cast<const float*>(grid.getDataPtr()),
                weights.data(),
                interpolatedValues.data(),
                currentBatchSize
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

} // namespace oscean::core_services::interpolation 