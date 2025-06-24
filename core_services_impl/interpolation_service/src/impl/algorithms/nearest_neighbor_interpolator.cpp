#include "nearest_neighbor_interpolator.h"
#include <cmath>
#include <algorithm>

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

NearestNeighborInterpolator::NearestNeighborInterpolator(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
    : simdManager_(simdManager) {
}

InterpolationResult NearestNeighborInterpolator::execute(
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
        result.message = "最近邻插值成功完成";
    } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
        // 网格到网格插值
        const auto& targetGridDef = std::get<TargetGridDefinition>(request.target);
        try {
            auto gridResult = interpolateToGrid(*request.sourceGrid, targetGridDef);
            result.data = std::move(gridResult);
            result.statusCode = 0;
            result.message = "网格到网格最近邻插值成功完成";
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

std::vector<std::optional<double>> NearestNeighborInterpolator::interpolateAtPoints(
    const GridData& sourceGrid,
    const std::vector<TargetPoint>& targetPoints) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());
    
    for (const auto& point : targetPoints) {
        if (point.coordinates.size() < 2) {
            results.push_back(std::nullopt);
            continue;
        }
        
        double worldX = point.coordinates[0];
        double worldY = point.coordinates[1];
        
        auto result = interpolateAtPoint(sourceGrid, worldX, worldY);
        results.push_back(result);
    }
    
    return results;
}

GridData NearestNeighborInterpolator::interpolateToGrid(
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
        targetGeoTransform[1] = (xMax - xMin) / (targetDef.cols > 1 ? targetDef.cols - 1 : 1);  // X方向像素大小
        targetGeoTransform[2] = 0.0;   // X方向旋转
        targetGeoTransform[3] = yMax;  // 左上角Y坐标
        targetGeoTransform[4] = 0.0;   // Y方向旋转
        targetGeoTransform[5] = -(yMax - yMin) / (targetDef.rows > 1 ? targetDef.rows - 1 : 1);  // Y方向像素大小（负值）
    }
    
    // 执行最近邻插值
    for (size_t row = 0; row < targetDef.rows; ++row) {
        for (size_t col = 0; col < targetDef.cols; ++col) {
            // 计算目标点的世界坐标
            double worldX = targetGeoTransform[0] + col * targetGeoTransform[1];
            double worldY = targetGeoTransform[3] + row * targetGeoTransform[5];
            
            // 执行最近邻插值
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

std::optional<double> NearestNeighborInterpolator::interpolateAtPoint(
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
    
    // 找到最近的网格点
    int nearestCol = static_cast<int>(std::round(gridX));
    int nearestRow = static_cast<int>(std::round(gridY));
    
    // 检查边界
    if (nearestCol < 0 || nearestCol >= static_cast<int>(def.cols) || 
        nearestRow < 0 || nearestRow >= static_cast<int>(def.rows)) {
        return std::nullopt;
    }
    
    // 返回最近邻的值
    return getGridValue(grid, nearestCol, nearestRow);
}

std::optional<double> NearestNeighborInterpolator::getGridValue(
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

std::vector<std::optional<double>> NearestNeighborInterpolator::simdBatchInterpolate(
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
        
        // 对当前批次的点进行处理
        for (size_t i = startIdx; i < endIdx; ++i) {
            const auto& point = points[i];
            if (point.coordinates.size() < 2) {
                results.push_back(std::nullopt);
                continue;
            }
            
            double worldX = point.coordinates[0];
            double worldY = point.coordinates[1];
            
            // 将世界坐标转换为网格坐标
            double gridX = (worldX - geoTransform[0]) / geoTransform[1];
            double gridY = (worldY - geoTransform[3]) / geoTransform[5];
            
            // 找到最近的网格点
            int nearestCol = static_cast<int>(std::round(gridX));
            int nearestRow = static_cast<int>(std::round(gridY));
            
            // 检查边界并获取值
            if (nearestCol >= 0 && nearestCol < static_cast<int>(def.cols) && 
                nearestRow >= 0 && nearestRow < static_cast<int>(def.rows)) {
                auto value = getGridValue(grid, nearestCol, nearestRow);
                results.push_back(value);
            } else {
                results.push_back(std::nullopt);
            }
        }
    }
    
    return results;
}

// 新增：内联SIMD优化的批量插值实现
std::vector<std::optional<double>> NearestNeighborInterpolator::interpolateAtPointsSIMD(
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
        // 处理8个点一组（最近邻插值简单，可以处理更多点）
        for (; i + 7 < targetPoints.size(); i += 8) {
            // 加载8个点的坐标
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
            
            // 四舍五入到最近的整数
            __m256 rounded_x = _mm256_round_ps(grid_x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 rounded_y = _mm256_round_ps(grid_y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            
            // 转换为整数索引
            __m256i idx_x = _mm256_cvtps_epi32(rounded_x);
            __m256i idx_y = _mm256_cvtps_epi32(rounded_y);
            
            // 边界检查
            __m256i zero = _mm256_setzero_si256();
            __m256i width_vec = _mm256_set1_epi32(width);
            __m256i height_vec = _mm256_set1_epi32(height);
            
            __m256i valid_x = _mm256_and_si256(
                _mm256_cmpgt_epi32(idx_x, _mm256_set1_epi32(-1)),
                _mm256_cmpgt_epi32(width_vec, idx_x)
            );
            
            __m256i valid_y = _mm256_and_si256(
                _mm256_cmpgt_epi32(idx_y, _mm256_set1_epi32(-1)),
                _mm256_cmpgt_epi32(height_vec, idx_y)
            );
            
            __m256i valid = _mm256_and_si256(valid_x, valid_y);
            
            // 计算线性索引
            __m256i linear_idx = _mm256_add_epi32(
                _mm256_mullo_epi32(idx_y, _mm256_set1_epi32(width)),
                idx_x
            );
            
            // 提取索引和有效性标志
            alignas(32) int idx_arr[8];
            alignas(32) int valid_arr[8];
            _mm256_store_si256((__m256i*)idx_arr, linear_idx);
            _mm256_store_si256((__m256i*)valid_arr, valid);
            
            // 收集数据并存储结果
            for (int j = 0; j < 8; ++j) {
                if (valid_arr[j] != 0) {
                    results.push_back(static_cast<double>(dataPtr[idx_arr[j]]));
                } else {
                    results.push_back(std::nullopt);
                }
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
    else if (g_cpuCaps.hasAVX2 && dataType == DataType::Float64) {
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
            
            // 四舍五入到最近的整数
            __m256d rounded_x = _mm256_round_pd(grid_x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d rounded_y = _mm256_round_pd(grid_y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            
            // 提取索引
            alignas(32) double x_arr[4], y_arr[4];
            _mm256_store_pd(x_arr, rounded_x);
            _mm256_store_pd(y_arr, rounded_y);
            
            // 处理每个点
            for (int j = 0; j < 4; ++j) {
                int col = static_cast<int>(x_arr[j]);
                int row = static_cast<int>(y_arr[j]);
                
                // 边界检查
                if (col >= 0 && col < width && row >= 0 && row < height) {
                    int idx = row * width + col;
                    results.push_back(dataPtr[idx]);
                } else {
                    results.push_back(std::nullopt);
                }
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

} // namespace oscean::core_services::interpolation 