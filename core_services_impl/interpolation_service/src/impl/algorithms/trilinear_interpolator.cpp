#include "trilinear_interpolator.h"
#include "kernels/interpolation_kernels.h"
#include <cmath>
#include <algorithm>
#include <variant>
#include <limits>

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

TrilinearInterpolator::TrilinearInterpolator(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
    : simdManager_(simdManager) {
}

InterpolationResult TrilinearInterpolator::execute(
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
        
        if (targetPoints.size() >= 4 && g_cpuCaps.hasAVX2) {
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
        result.message = "三线性插值成功完成";
    } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
        // 网格到网格插值
        const auto& targetGridDef = std::get<TargetGridDefinition>(request.target);
        try {
            auto gridResult = interpolateToGrid(*request.sourceGrid, targetGridDef);
            result.data = std::move(gridResult);
            result.statusCode = 0;
            result.message = "网格到网格三线性插值成功完成";
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
std::vector<std::optional<double>> TrilinearInterpolator::interpolateAtPointsSIMD(
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
    const int bands = static_cast<int>(sourceGrid.getBandCount());
    
    // 如果只有一个波段，使用双线性插值的SIMD优化
    if (bands == 1) {
        return interpolateAtPointsSIMD2D(sourceGrid, targetPoints);
    }
    
    // 根据数据类型选择优化路径
    DataType dataType = sourceGrid.getDataType();
    
    #ifdef __AVX2__
    if (g_cpuCaps.hasAVX2 && dataType == DataType::Float32) {
        // AVX2优化路径（处理float类型）
        const float* dataPtr = static_cast<const float*>(sourceGrid.getDataPtr());
        
        size_t i = 0;
        // 处理4个点一组（三线性插值计算量大，一次处理较少点）
        for (; i + 3 < targetPoints.size(); i += 4) {
            alignas(32) float x_arr[4], y_arr[4], z_arr[4];
            for (int j = 0; j < 4; ++j) {
                x_arr[j] = static_cast<float>(targetPoints[i+j].coordinates[0]);
                y_arr[j] = static_cast<float>(targetPoints[i+j].coordinates[1]);
                z_arr[j] = targetPoints[i+j].coordinates.size() > 2 ? 
                          static_cast<float>(targetPoints[i+j].coordinates[2]) : 0.0f;
            }
            
            // 转换到网格坐标
            __m128 x_coords = _mm_load_ps(x_arr);
            __m128 y_coords = _mm_load_ps(y_arr);
            __m128 z_coords = _mm_load_ps(z_arr);
            
            __m128 origin_x = _mm_set1_ps(static_cast<float>(originX));
            __m128 dx_vec = _mm_set1_ps(static_cast<float>(dx));
            __m128 origin_y = _mm_set1_ps(static_cast<float>(originY));
            __m128 dy_vec = _mm_set1_ps(static_cast<float>(dy));
            
            __m128 grid_x = _mm_div_ps(_mm_sub_ps(x_coords, origin_x), dx_vec);
            __m128 grid_y = _mm_div_ps(_mm_sub_ps(y_coords, origin_y), dy_vec);
            __m128 grid_z = z_coords; // 简化处理
            
            // 提取整数部分和小数部分
            __m128 x0_f = _mm_floor_ps(grid_x);
            __m128 y0_f = _mm_floor_ps(grid_y);
            __m128 z0_f = _mm_floor_ps(grid_z);
            
            __m128 fx = _mm_sub_ps(grid_x, x0_f);
            __m128 fy = _mm_sub_ps(grid_y, y0_f);
            __m128 fz = _mm_sub_ps(grid_z, z0_f);
            
            // 转换为整数索引并处理每个点
            alignas(16) float x0_arr[4], y0_arr[4], z0_arr[4];
            alignas(16) float fx_arr[4], fy_arr[4], fz_arr[4];
            
            _mm_store_ps(x0_arr, x0_f);
            _mm_store_ps(y0_arr, y0_f);
            _mm_store_ps(z0_arr, z0_f);
            _mm_store_ps(fx_arr, fx);
            _mm_store_ps(fy_arr, fy);
            _mm_store_ps(fz_arr, fz);
            
            for (int j = 0; j < 4; ++j) {
                int x0 = static_cast<int>(x0_arr[j]);
                int y0 = static_cast<int>(y0_arr[j]);
                int z0 = static_cast<int>(z0_arr[j]);
                
                // 边界检查
                if (x0 < 0 || x0 >= width - 1 || 
                    y0 < 0 || y0 >= height - 1 ||
                    z0 < 0 || z0 >= bands - 1) {
                    results.push_back(std::nullopt);
                    continue;
                }
                
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                int z1 = z0 + 1;
                
                // 使用AVX2加速8个角点值的收集
                int planeSize = width * height;
                
                // 底层平面的4个点
                __m128 v_bottom = _mm_set_ps(
                    dataPtr[z0 * planeSize + y1 * width + x1],  // v110
                    dataPtr[z0 * planeSize + y1 * width + x0],  // v010
                    dataPtr[z0 * planeSize + y0 * width + x1],  // v100
                    dataPtr[z0 * planeSize + y0 * width + x0]   // v000
                );
                
                // 顶层平面的4个点
                __m128 v_top = _mm_set_ps(
                    dataPtr[z1 * planeSize + y1 * width + x1],  // v111
                    dataPtr[z1 * planeSize + y1 * width + x0],  // v011
                    dataPtr[z1 * planeSize + y0 * width + x1],  // v101
                    dataPtr[z1 * planeSize + y0 * width + x0]   // v001
                );
                
                // 在Z方向插值
                __m128 fz_vec = _mm_set1_ps(fz_arr[j]);
                __m128 one_minus_fz = _mm_sub_ps(_mm_set1_ps(1.0f), fz_vec);
                
                __m128 v_interp = _mm_add_ps(
                    _mm_mul_ps(v_bottom, one_minus_fz),
                    _mm_mul_ps(v_top, fz_vec)
                );
                
                // 提取4个值进行双线性插值
                alignas(16) float v_arr[4];
                _mm_store_ps(v_arr, v_interp);
                
                // 双线性插值
                float fx_val = fx_arr[j];
                float fy_val = fy_arr[j];
                
                float v0 = v_arr[0] * (1 - fx_val) + v_arr[2] * fx_val;
                float v1 = v_arr[1] * (1 - fx_val) + v_arr[3] * fx_val;
                float result = v0 * (1 - fy_val) + v1 * fy_val;
                
                results.push_back(static_cast<double>(result));
            }
        }
        
        // 处理剩余的点
        for (; i < targetPoints.size(); ++i) {
            const auto& point = targetPoints[i];
            double worldX = point.coordinates[0];
            double worldY = point.coordinates[1];
            double worldZ = point.coordinates.size() > 2 ? point.coordinates[2] : 0.0;
            results.push_back(interpolateAtPoint(sourceGrid, worldX, worldY, worldZ));
        }
        
        return results;
    }
    #endif
    
    // 非SIMD路径或不支持的数据类型
    return interpolateAtPoints(sourceGrid, targetPoints);
}

// 新增：2D情况下的SIMD优化（当只有一个波段时）
std::vector<std::optional<double>> TrilinearInterpolator::interpolateAtPointsSIMD2D(
    const GridData& sourceGrid,
    const std::vector<TargetPoint>& targetPoints) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());
    
    const auto& def = sourceGrid.getDefinition();
    const auto& geoTransform = sourceGrid.getGeoTransform();
    
    const double originX = geoTransform[0];
    const double dx = geoTransform[1];
    const double originY = geoTransform[3];
    const double dy = geoTransform[5];
    const int width = static_cast<int>(def.cols);
    const int height = static_cast<int>(def.rows);
    
    DataType dataType = sourceGrid.getDataType();
    
    #ifdef __AVX2__
    if (g_cpuCaps.hasAVX2 && dataType == DataType::Float32) {
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
            __m256 y0_f = _mm256_floor_ps(grid_y);
            
            __m256 fx = _mm256_sub_ps(grid_x, x0_f);
            __m256 fy = _mm256_sub_ps(grid_y, y0_f);
            
            // 转换为整数索引
            alignas(32) float x0_arr[8], y0_arr[8], fx_arr[8], fy_arr[8];
            _mm256_store_ps(x0_arr, x0_f);
            _mm256_store_ps(y0_arr, y0_f);
            _mm256_store_ps(fx_arr, fx);
            _mm256_store_ps(fy_arr, fy);
            
            // 处理每个点
            for (int j = 0; j < 8; ++j) {
                int x0 = static_cast<int>(x0_arr[j]);
                int y0 = static_cast<int>(y0_arr[j]);
                
                if (x0 < 0 || x0 >= width - 1 || y0 < 0 || y0 >= height - 1) {
                    results.push_back(std::nullopt);
                    continue;
                }
                
                int idx00 = y0 * width + x0;
                int idx10 = idx00 + 1;
                int idx01 = idx00 + width;
                int idx11 = idx01 + 1;
                
                float v00 = dataPtr[idx00];
                float v10 = dataPtr[idx10];
                float v01 = dataPtr[idx01];
                float v11 = dataPtr[idx11];
                
                float fx_val = fx_arr[j];
                float fy_val = fy_arr[j];
                
                float v0 = v00 * (1 - fx_val) + v10 * fx_val;
                float v1 = v01 * (1 - fx_val) + v11 * fx_val;
                float result = v0 * (1 - fy_val) + v1 * fy_val;
                
                results.push_back(static_cast<double>(result));
            }
        }
        
        // 处理剩余的点
        for (; i < targetPoints.size(); ++i) {
            const auto& point = targetPoints[i];
            results.push_back(interpolateAtPoint(sourceGrid, 
                point.coordinates[0], 
                point.coordinates[1], 
                0.0));
        }
        
        return results;
    }
    #endif
    
    // 非SIMD路径
    return interpolateAtPoints(sourceGrid, targetPoints);
}

std::vector<std::optional<double>> TrilinearInterpolator::interpolateAtPoints(
    const GridData& sourceGrid,
    const std::vector<TargetPoint>& targetPoints) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());
    
    for (const auto& point : targetPoints) {
        if (point.coordinates.size() < 3) {
            // 如果只有2D坐标，使用Z=0
            double worldX = point.coordinates.size() > 0 ? point.coordinates[0] : 0.0;
            double worldY = point.coordinates.size() > 1 ? point.coordinates[1] : 0.0;
            double worldZ = 0.0;
            auto result = interpolateAtPoint(sourceGrid, worldX, worldY, worldZ);
            results.push_back(result);
        } else {
            double worldX = point.coordinates[0];
            double worldY = point.coordinates[1];
            double worldZ = point.coordinates[2];
            auto result = interpolateAtPoint(sourceGrid, worldX, worldY, worldZ);
            results.push_back(result);
        }
    }
    
    return results;
}

GridData TrilinearInterpolator::interpolateToGrid(
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
    
    // 简化实现：假设目标网格是规则网格
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
    
    // 执行插值
    for (size_t row = 0; row < targetDef.rows; ++row) {
        for (size_t col = 0; col < targetDef.cols; ++col) {
            // 计算目标点的世界坐标
            double worldX = targetGeoTransform[0] + col * targetGeoTransform[1];
            double worldY = targetGeoTransform[3] + row * targetGeoTransform[5];
            double worldZ = 0.0; // 假设Z=0
            
            // 执行插值
            auto interpolatedValue = interpolateAtPoint(sourceGrid, worldX, worldY, worldZ);
            
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

std::optional<double> TrilinearInterpolator::interpolateAtPoint(
    const GridData& grid, 
    double worldX, 
    double worldY,
    double worldZ) const {
    
    const auto& def = grid.getDefinition();
    const auto& geoTransform = grid.getGeoTransform();
    
    // 检查地理变换是否有效
    if (geoTransform.size() < 6) {
        return std::nullopt;
    }
    
    // 将世界坐标转换为网格坐标
    double gridX = (worldX - geoTransform[0]) / geoTransform[1];
    double gridY = (worldY - geoTransform[3]) / geoTransform[5];
    
    // 对于3D数据，Z坐标处理（简化：假设Z对应波段）
    double gridZ = worldZ; // 可以根据实际需要调整
    
    // 检查边界
    if (gridX < 0 || gridX >= def.cols - 1 || 
        gridY < 0 || gridY >= def.rows - 1) {
        return std::nullopt;
    }
    
    // 如果只有一个波段，使用双线性插值
    if (grid.getBandCount() == 1) {
        // 获取四个角点的整数坐标
        int x0 = static_cast<int>(std::floor(gridX));
        int y0 = static_cast<int>(std::floor(gridY));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        // 获取四个角点的值
        auto v00 = getGridValue(grid, x0, y0, 0);
        auto v10 = getGridValue(grid, x1, y0, 0);
        auto v01 = getGridValue(grid, x0, y1, 0);
        auto v11 = getGridValue(grid, x1, y1, 0);
        
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
    
    // 三线性插值需要至少2个波段
    if (grid.getBandCount() < 2) {
        return std::nullopt;
    }
    
    // 确定Z方向的插值区间
    int z0 = static_cast<int>(std::floor(gridZ));
    int z1 = z0 + 1;
    
    // 检查Z方向边界
    if (z0 < 0 || z1 >= static_cast<int>(grid.getBandCount())) {
        // 如果超出Z边界，使用最近的波段进行双线性插值
        int bandIdx = std::max(0, std::min(static_cast<int>(std::round(gridZ)), 
                                          static_cast<int>(grid.getBandCount()) - 1));
        
        int x0 = static_cast<int>(std::floor(gridX));
        int y0 = static_cast<int>(std::floor(gridY));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        auto v00 = getGridValue(grid, x0, y0, bandIdx);
        auto v10 = getGridValue(grid, x1, y0, bandIdx);
        auto v01 = getGridValue(grid, x0, y1, bandIdx);
        auto v11 = getGridValue(grid, x1, y1, bandIdx);
        
        if (!v00.has_value() || !v10.has_value() || !v01.has_value() || !v11.has_value()) {
            return std::nullopt;
        }
        
        double fx = gridX - x0;
        double fy = gridY - y0;
        
        std::array<double, 4> values = {v00.value(), v10.value(), v01.value(), v11.value()};
        return kernels::bilinear(values, fx, fy);
    }
    
    // 获取八个角点的整数坐标
    int x0 = static_cast<int>(std::floor(gridX));
    int y0 = static_cast<int>(std::floor(gridY));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // 获取八个角点的值 (2x2x2立方体)
    auto v000 = getGridValue(grid, x0, y0, z0);
    auto v100 = getGridValue(grid, x1, y0, z0);
    auto v010 = getGridValue(grid, x0, y1, z0);
    auto v110 = getGridValue(grid, x1, y1, z0);
    auto v001 = getGridValue(grid, x0, y0, z1);
    auto v101 = getGridValue(grid, x1, y0, z1);
    auto v011 = getGridValue(grid, x0, y1, z1);
    auto v111 = getGridValue(grid, x1, y1, z1);
    
    // 检查所有角点值是否有效
    if (!v000.has_value() || !v100.has_value() || !v010.has_value() || !v110.has_value() ||
        !v001.has_value() || !v101.has_value() || !v011.has_value() || !v111.has_value()) {
        return std::nullopt;
    }
    
    // 计算插值权重
    double fx = gridX - x0;
    double fy = gridY - y0;
    double fz = gridZ - z0;
    
    // 使用三线性插值
    std::array<double, 8> values = {
        v000.value(), v100.value(), v010.value(), v110.value(),
        v001.value(), v101.value(), v011.value(), v111.value()
    };
    
    return kernels::trilinear(values, fx, fy, fz);
}

std::vector<std::optional<double>> TrilinearInterpolator::simdBatchInterpolate(
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
    
    // 预分配SIMD对齐的内存缓冲区
    const size_t alignment = simdManager_->getAlignment();
    std::vector<float> xCoords, yCoords, zCoords;
    std::vector<float> gridXCoords, gridYCoords, gridZCoords;
    std::vector<float> interpolatedValues;
    
    for (size_t batch = 0; batch < numBatches; ++batch) {
        size_t startIdx = batch * batchSize;
        size_t endIdx = std::min(startIdx + batchSize, points.size());
        size_t currentBatchSize = endIdx - startIdx;
        
        // 清空并预分配缓冲区
        xCoords.clear(); xCoords.reserve(currentBatchSize);
        yCoords.clear(); yCoords.reserve(currentBatchSize);
        zCoords.clear(); zCoords.reserve(currentBatchSize);
        gridXCoords.clear(); gridXCoords.reserve(currentBatchSize);
        gridYCoords.clear(); gridYCoords.reserve(currentBatchSize);
        gridZCoords.clear(); gridZCoords.reserve(currentBatchSize);
        interpolatedValues.clear(); interpolatedValues.resize(currentBatchSize);
        
        // 准备SIMD输入数据
        for (size_t i = startIdx; i < endIdx; ++i) {
            const auto& point = points[i];
            float worldX = point.coordinates.size() > 0 ? static_cast<float>(point.coordinates[0]) : 0.0f;
            float worldY = point.coordinates.size() > 1 ? static_cast<float>(point.coordinates[1]) : 0.0f;
            float worldZ = point.coordinates.size() > 2 ? static_cast<float>(point.coordinates[2]) : 0.0f;
            
            xCoords.push_back(worldX);
            yCoords.push_back(worldY);
            zCoords.push_back(worldZ);
            
            // 将世界坐标转换为网格坐标
            float gridX = (worldX - static_cast<float>(geoTransform[0])) / static_cast<float>(geoTransform[1]);
            float gridY = (worldY - static_cast<float>(geoTransform[3])) / static_cast<float>(geoTransform[5]);
            float gridZ = worldZ; // 简化：Z坐标直接使用
            
            gridXCoords.push_back(gridX);
            gridYCoords.push_back(gridY);
            gridZCoords.push_back(gridZ);
        }
        
        try {
            // 使用SIMD管理器执行批量三线性插值
            simdManager_->bilinearInterpolate(
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
                double worldZ = point.coordinates.size() > 2 ? point.coordinates[2] : 0.0;
                results.push_back(interpolateAtPoint(grid, worldX, worldY, worldZ));
            }
        }
    }
    
    return results;
}

std::optional<double> TrilinearInterpolator::getGridValue(
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

} // namespace oscean::core_services::interpolation 