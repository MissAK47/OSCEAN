#include "interpolation_kernels.h"
#include "common_utils/simd/isimd_manager.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>

// 添加boost future支持
#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace oscean::core_services::interpolation::kernels {

// bilinear和trilinear函数已在头文件中内联定义

double nearestNeighbor(const std::array<double, 4>& values, double wx, double wy) {
    // 选择最近的邻居
    int x_idx = (wx < 0.5) ? 0 : 1;
    int y_idx = (wy < 0.5) ? 0 : 1;
    
    // values: [左下, 右下, 左上, 右上]
    int index = y_idx * 2 + x_idx;
    return values[index];
}

double linear1D(double v0, double v1, double t) {
    return v0 * (1.0 - t) + v1 * t;
}

double pchipSlope(double y_prev, double y_curr, double y_next, 
                  double h_prev, double h_curr) {
    // 计算相邻段的斜率
    double s_prev = (y_curr - y_prev) / h_prev;
    double s_curr = (y_next - y_curr) / h_curr;
    
    // 如果斜率符号不同，则斜率为0（保持单调性）
    if (s_prev * s_curr <= 0.0) {
        return 0.0;
    }
    
    // 使用调和平均数
    double w1 = 2.0 * h_curr + h_prev;
    double w2 = h_curr + 2.0 * h_prev;
    return (w1 + w2) / (w1 / s_prev + w2 / s_curr);
}

double pchipEndpointSlopeStart(double y0, double y1, double h0) {
    // 起始点的斜率：使用一阶差分
    return (y1 - y0) / h0;
}

double pchipEndpointSlopeEnd(double yn_1, double yn, double hn_1) {
    // 结束点的斜率：使用一阶差分
    return (yn - yn_1) / hn_1;
}

std::optional<double> inverseDistanceWeighting(
    const std::vector<double>& distances,
    const std::vector<double>& values,
    double power) {
    
    if (distances.size() != values.size() || distances.empty()) {
        return std::nullopt;
    }
    
    double weightSum = 0.0;
    double valueSum = 0.0;
    
    for (size_t i = 0; i < distances.size(); ++i) {
        if (!isValidValue(distances[i]) || !isValidValue(values[i])) {
            continue;
        }
        
        if (distances[i] < 1e-10) {
            // 如果距离非常小，直接返回该点的值
            return values[i];
        }
        
        double weight = 1.0 / std::pow(distances[i], power);
        weightSum += weight;
        valueSum += weight * values[i];
    }
    
    if (weightSum < 1e-10) {
        return std::nullopt;
    }
    
    return valueSum / weightSum;
}

bool isValidValue(double value) {
    return std::isfinite(value);
}

double bicubic(const std::array<double, 16>& values, double wx, double wy) {
    // 优化的双三次插值实现
    // values是4x4网格的值，按行主序排列
    
    // 优化的三次插值权重函数（减少计算量）
    auto cubicWeight = [](double t, std::array<double, 4>& weights) {
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double mt = 1.0 - t;
        
        // Catmull-Rom样条权重
        weights[0] = -0.5 * t * mt * mt;
        weights[1] = 1.0 + t2 * (1.5 * t - 2.5);
        weights[2] = t * (1.0 + mt * (1.0 + mt));
        weights[3] = -0.5 * t2 * mt;
    };
    
    std::array<double, 4> wx_weights, wy_weights;
    cubicWeight(wx, wx_weights);
    cubicWeight(wy, wy_weights);
    
    // 分离的行列计算，改善缓存局部性
    std::array<double, 4> row_results = {0.0, 0.0, 0.0, 0.0};
    
    // 先沿X方向插值每一行
    for (int j = 0; j < 4; ++j) {
        const int row_offset = j * 4;
        row_results[j] = values[row_offset] * wx_weights[0] +
                        values[row_offset + 1] * wx_weights[1] +
                        values[row_offset + 2] * wx_weights[2] +
                        values[row_offset + 3] * wx_weights[3];
    }
    
    // 再沿Y方向插值
    return row_results[0] * wy_weights[0] +
           row_results[1] * wy_weights[1] +
           row_results[2] * wy_weights[2] +
           row_results[3] * wy_weights[3];
}

double cubicSpline(double t, double p0, double p1, double p2, double p3) {
    // Catmull-Rom样条插值
    double t2 = t * t;
    double t3 = t2 * t;
    
    return 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    );
}

// SIMD优化的批量插值函数
namespace simd {

std::vector<double> batchBilinear(
    const std::vector<std::array<double, 4>>& valuesBatch,
    const std::vector<double>& wxBatch,
    const std::vector<double>& wyBatch) {
    
    const size_t count = valuesBatch.size();
    std::vector<double> results(count);
    
    // 批量处理，利用现代CPU的流水线和预取
    #pragma omp parallel for simd
    for (size_t i = 0; i < count; ++i) {
        const double wx = wxBatch[i];
        const double wy = wyBatch[i];
        const double wx_inv = 1.0 - wx;
        const double wy_inv = 1.0 - wy;
        
        const auto& vals = valuesBatch[i];
        results[i] = (vals[0] * wx_inv + vals[1] * wx) * wy_inv +
                     (vals[2] * wx_inv + vals[3] * wx) * wy;
    }
    
    return results;
}

std::vector<double> batchTrilinear(
    const std::vector<std::array<double, 8>>& valuesBatch,
    const std::vector<double>& wxBatch,
    const std::vector<double>& wyBatch,
    const std::vector<double>& wzBatch) {
    
    std::vector<double> results;
    results.reserve(valuesBatch.size());
    
    for (size_t i = 0; i < valuesBatch.size(); ++i) {
        results.push_back(trilinear(valuesBatch[i], wxBatch[i], wyBatch[i], wzBatch[i]));
    }
    
    return results;
}

std::vector<double> batchNearestNeighbor(
    const std::vector<std::array<double, 4>>& valuesBatch,
    const std::vector<double>& wxBatch,
    const std::vector<double>& wyBatch) {
    
    std::vector<double> results;
    results.reserve(valuesBatch.size());
    
    for (size_t i = 0; i < valuesBatch.size(); ++i) {
        std::array<double, 4> values = {
            valuesBatch[i][0], valuesBatch[i][1], 
            valuesBatch[i][2], valuesBatch[i][3]
        };
        results.push_back(nearestNeighbor(values, wxBatch[i], wyBatch[i]));
    }
    
    return results;
}

void performanceGridInterpolation(
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    const float* gridData,
    const float* xCoords,
    const float* yCoords,
    float* results,
    size_t count,
    size_t gridWidth,
    size_t gridHeight,
    const std::string& method) {
    
    if (!simdManager) {
        // 回退到标量实现
        for (size_t i = 0; i < count; ++i) {
            float x = xCoords[i];
            float y = yCoords[i];
            
            // 边界检查
            if (x < 0 || x >= gridWidth - 1 || y < 0 || y >= gridHeight - 1) {
                results[i] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // 简单双线性插值
            int x0 = static_cast<int>(std::floor(x));
            int y0 = static_cast<int>(std::floor(y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float fx = x - x0;
            float fy = y - y0;
            
            float v00 = gridData[y0 * gridWidth + x0];
            float v10 = gridData[y0 * gridWidth + x1];
            float v01 = gridData[y1 * gridWidth + x0];
            float v11 = gridData[y1 * gridWidth + x1];
            
            std::array<float, 4> values = {v00, v10, v01, v11};
            results[i] = static_cast<float>(bilinear(
                std::array<double, 4>{
                    static_cast<double>(v00), static_cast<double>(v10), 
                    static_cast<double>(v01), static_cast<double>(v11)
                }, 
                static_cast<double>(fx), static_cast<double>(fy)
            ));
        }
        return;
    }
    
    try {
        if (method == "bilinear") {
            simdManager->bilinearInterpolate(
                gridData, xCoords, yCoords, results, count, gridWidth, gridHeight);
        } else if (method == "bicubic") {
            simdManager->bicubicInterpolate(
                gridData, xCoords, yCoords, results, count, gridWidth, gridHeight);
        } else {
            // 默认使用双线性插值
            simdManager->bilinearInterpolate(
                gridData, xCoords, yCoords, results, count, gridWidth, gridHeight);
        }
    } catch (const std::exception& e) {
        // SIMD失败时回退到标量实现
        performanceGridInterpolation(nullptr, gridData, xCoords, yCoords, 
                                    results, count, gridWidth, gridHeight, method);
    }
}

boost::future<std::vector<double>> asyncBatchInterpolation(
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    const float* gridData,
    const std::vector<std::pair<float, float>>& points,
    size_t gridWidth,
    size_t gridHeight,
    const std::string& method) {
    
    return boost::async(boost::launch::async, [=]() -> std::vector<double> {
        std::vector<double> results(points.size());
        
        // 准备坐标数组
        std::vector<float> xCoords, yCoords;
        xCoords.reserve(points.size());
        yCoords.reserve(points.size());
        
        for (const auto& point : points) {
            xCoords.push_back(point.first);
            yCoords.push_back(point.second);
        }
        
        // 准备结果数组
        std::vector<float> floatResults(points.size());
        
        // 执行插值
        performanceGridInterpolation(
            simdManager,
            gridData,
            xCoords.data(),
            yCoords.data(),
            floatResults.data(),
            points.size(),
            gridWidth,
            gridHeight,
            method
        );
        
        // 转换结果
        for (size_t i = 0; i < points.size(); ++i) {
            results[i] = static_cast<double>(floatResults[i]);
        }
        
        return results;
    });
}

size_t getOptimalBatchSize(
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    size_t dataSize,
    double complexity) {
    
    if (!simdManager) {
        return std::min(dataSize, size_t(1000)); // 默认批量大小
    }
    
    size_t baseBatchSize = simdManager->getOptimalBatchSize();
    
    // 根据数据大小和复杂度调整
    if (dataSize < 100) {
        return dataSize; // 小数据集不分批
    } else if (dataSize < 10000) {
        return std::min(baseBatchSize * 2, dataSize);
    } else {
        // 大数据集，考虑复杂度
        size_t adjustedSize = static_cast<size_t>(baseBatchSize / complexity);
        return std::max(adjustedSize, baseBatchSize / 4); // 最小为基础大小的1/4
    }
}

std::unique_ptr<float[], void(*)(float*)> prepareAlignedData(
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    const float* inputData,
    size_t size) {
    
    size_t alignment = simdManager ? simdManager->getAlignment() : 32;
    
    // 分配对齐内存
    size_t alignedSize = ((size * sizeof(float) + alignment - 1) / alignment) * alignment;
    
#ifdef _WIN32
    float* alignedPtr = static_cast<float*>(_aligned_malloc(alignedSize, alignment));
#else
    float* alignedPtr = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&alignedPtr), alignment, alignedSize) != 0) {
        alignedPtr = nullptr;
    }
#endif
    
    if (!alignedPtr) {
        throw std::bad_alloc();
    }
    
    // 复制数据
    std::memcpy(alignedPtr, inputData, size * sizeof(float));
    
    // 返回带自定义删除器的unique_ptr
    auto deleter = [](float* ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    };
    
    return std::unique_ptr<float[], void(*)(float*)>(alignedPtr, deleter);
}

} // namespace simd

} // namespace oscean::core_services::interpolation::kernels 