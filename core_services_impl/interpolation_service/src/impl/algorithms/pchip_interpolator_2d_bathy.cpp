#include "pchip_interpolator_2d_bathy.h"
#include "core_services/common_data_types.h"
#include <stdexcept>
#include <vector>
#include <cmath>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace oscean::core_services::interpolation {

PCHIPInterpolator2DBathy::PCHIPInterpolator2DBathy(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
    : simdManager_(simdManager) {
}

// AVX2优化的PCHIP斜率计算
__m256d pchipSlopeAVX2(__m256d h1, __m256d h2, __m256d m1, __m256d m2) {
    // 检查m1 * m2 <= 0的情况
    __m256d prod = _mm256_mul_pd(m1, m2);
    __m256d zero = _mm256_setzero_pd();
    __m256d mask = _mm256_cmp_pd(prod, zero, _CMP_LE_OQ);
    
    // 计算加权调和平均
    __m256d two = _mm256_set1_pd(2.0);
    __m256d wh1 = _mm256_fmadd_pd(two, h1, h2);
    __m256d wh2 = _mm256_fmadd_pd(two, h2, h1);
    
    // 避免除零
    __m256d m1_safe = _mm256_blendv_pd(m1, _mm256_set1_pd(1.0), 
                                       _mm256_cmp_pd(m1, zero, _CMP_EQ_OQ));
    __m256d m2_safe = _mm256_blendv_pd(m2, _mm256_set1_pd(1.0), 
                                       _mm256_cmp_pd(m2, zero, _CMP_EQ_OQ));
    
    __m256d inv_m1 = _mm256_div_pd(wh1, m1_safe);
    __m256d inv_m2 = _mm256_div_pd(wh2, m2_safe);
    __m256d sum_wh = _mm256_add_pd(wh1, wh2);
    __m256d sum_inv = _mm256_add_pd(inv_m1, inv_m2);
    __m256d result = _mm256_div_pd(sum_wh, sum_inv);
    
    // 应用掩码，如果m1*m2 <= 0则返回0
    return _mm256_blendv_pd(result, zero, mask);
}

// AVX2优化的Hermite插值
__m256d interpolateCubicAVX2(__m256d t, __m256d y0, __m256d y1, 
                             __m256d d0, __m256d d1, __m256d h) {
    __m256d t2 = _mm256_mul_pd(t, t);
    __m256d t3 = _mm256_mul_pd(t2, t);
    
    // Hermite基函数
    __m256d zero = _mm256_setzero_pd();
    __m256d two = _mm256_set1_pd(2.0);
    __m256d three = _mm256_set1_pd(3.0);
    __m256d one = _mm256_set1_pd(1.0);
    
    // h00 = 2*t^3 - 3*t^2 + 1
    __m256d h00 = _mm256_fmadd_pd(two, t3, _mm256_fnmadd_pd(three, t2, one));
    
    // h10 = t^3 - 2*t^2 + t
    __m256d h10 = _mm256_fmadd_pd(t3, one, _mm256_fnmadd_pd(two, t2, t));
    
    // h01 = -2*t^3 + 3*t^2
    __m256d neg_two = _mm256_sub_pd(zero, two);
    __m256d h01 = _mm256_fmadd_pd(neg_two, t3, _mm256_mul_pd(three, t2));
    
    // h11 = t^3 - t^2
    __m256d h11 = _mm256_sub_pd(t3, t2);
    
    // 组合结果
    __m256d result = _mm256_mul_pd(h00, y0);
    result = _mm256_fmadd_pd(h10, _mm256_mul_pd(h, d0), result);
    result = _mm256_fmadd_pd(h01, y1, result);
    result = _mm256_fmadd_pd(h11, _mm256_mul_pd(h, d1), result);
    
    return result;
}

InterpolationResult PCHIPInterpolator2DBathy::execute(
    const InterpolationRequest& request,
    const PrecomputedData* precomputed) const {

    if (!request.sourceGrid || !std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        return { {}, -1, "无效的输入：需要源网格和目标点列表。" };
    }

    const auto& sourceGrid = *request.sourceGrid;
    const auto& targetPoints = std::get<std::vector<TargetPoint>>(request.target);
    const auto& def = sourceGrid.getDefinition();

    if (def.cols < 4 || def.rows < 4) {
        return { {}, -1, "PCHIP 2D Bathy优化算法需要至少4x4的网格。" };
    }

    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());

    // 假设为规则网格，获取地理变换参数
    const auto& geoTransform = sourceGrid.getGeoTransform();
    const double originX = geoTransform[0];
    const double pixelSizeX = geoTransform[1];
    const double originY = geoTransform[3];
    const double pixelSizeY = geoTransform[5];

    // 检查是否可以使用SIMD优化
    bool useAVX2 = simdManager_ && simdManager_->getFeatures().hasAVX2;
    
    // 如果有足够的点并且支持AVX2，则批量处理
    if (useAVX2 && targetPoints.size() >= 4) {
        // 批量处理4个点
        for (size_t idx = 0; idx + 3 < targetPoints.size(); idx += 4) {
            alignas(32) double targetX[4], targetY[4];
            for (int i = 0; i < 4; ++i) {
                if (targetPoints[idx + i].coordinates.size() >= 2) {
                    targetX[i] = targetPoints[idx + i].coordinates[0];
                    targetY[i] = targetPoints[idx + i].coordinates[1];
                } else {
                    targetX[i] = 0.0;
                    targetY[i] = 0.0;
                }
            }
            
            __m256d vTargetX = _mm256_load_pd(targetX);
            __m256d vTargetY = _mm256_load_pd(targetY);
            __m256d vOriginX = _mm256_set1_pd(originX);
            __m256d vOriginY = _mm256_set1_pd(originY);
            __m256d vPixelSizeX = _mm256_set1_pd(pixelSizeX);
            __m256d vPixelSizeY = _mm256_set1_pd(pixelSizeY);
            
            // 转换为像素坐标
            __m256d vPixX = _mm256_div_pd(_mm256_sub_pd(vTargetX, vOriginX), vPixelSizeX);
            __m256d vPixY = _mm256_div_pd(_mm256_sub_pd(vTargetY, vOriginY), vPixelSizeY);
            
            // 对每个点进行插值（这部分仍需要串行处理，因为内存访问模式不规则）
            alignas(32) double pixX[4], pixY[4];
            _mm256_store_pd(pixX, vPixX);
            _mm256_store_pd(pixY, vPixY);
            
            for (int i = 0; i < 4; ++i) {
                if (targetPoints[idx + i].coordinates.size() < 2) {
                    results.push_back(std::nullopt);
                    continue;
                }
                
                // 确定插值所需的4x4网格的左上角索引
                int x0 = static_cast<int>(std::floor(pixX[i])) - 1;
                int y0 = static_cast<int>(std::floor(pixY[i])) - 1;
                
                // 边界检查
                if (x0 < 0 || y0 < 0 || x0 + 3 >= def.cols || y0 + 3 >= def.rows) {
                    results.push_back(std::nullopt);
                    continue;
                }
                
                // 双三次PCHIP插值核心逻辑
                std::vector<double> y_interp(4);
                
                // 沿X方向对4行数据进行插值
                for (int j = 0; j < 4; ++j) {
                    int current_y = y0 + j;
                    
                    // 获取行数据
                    double v[4], m[3], h[3], d[4];
                    for (int k = 0; k < 4; ++k) {
                        if (sourceGrid.getDataType() == DataType::Float32) {
                            v[k] = static_cast<double>(sourceGrid.getValue<float>(current_y, x0 + k, 0));
                        } else if (sourceGrid.getDataType() == DataType::Float64) {
                            v[k] = sourceGrid.getValue<double>(current_y, x0 + k, 0);
                        } else {
                            v[k] = static_cast<double>(sourceGrid.getValue<float>(current_y, x0 + k, 0));
                        }
                    }
                    
                    // 计算差分
                    for (int k = 0; k < 3; ++k) {
                        h[k] = 1.0;
                        m[k] = v[k+1] - v[k];
                    }
                    
                    // 计算导数
                    d[0] = (2 * m[0] * h[1] + m[1] * h[0]) / (h[0] + h[1]);
                    d[1] = pchipSlope(h[0], h[1], m[0], m[1]);
                    d[2] = pchipSlope(h[1], h[2], m[1], m[2]);
                    d[3] = (2 * m[2] * h[1] + m[1] * h[2]) / (h[1] + h[2]);
                    
                    // 在目标x位置进行三次插值
                    y_interp[j] = interpolateCubic(pixX[i], x0 + 1.0, x0 + 2.0, v[1], v[2], d[1], d[2]);
                }
                
                // 沿Y方向对插值后的数据进行插值
                double m_y[3], h_y[3], d_y[4];
                for (int j = 0; j < 3; ++j) {
                    h_y[j] = 1.0;
                    m_y[j] = y_interp[j+1] - y_interp[j];
                }
                
                // 同样为Y方向的端点使用非中心差分
                d_y[0] = (2 * m_y[0] * h_y[1] + m_y[1] * h_y[0]) / (h_y[0] + h_y[1]);
                d_y[1] = pchipSlope(h_y[0], h_y[1], m_y[0], m_y[1]);
                d_y[2] = pchipSlope(h_y[1], h_y[2], m_y[1], m_y[2]);
                d_y[3] = (2 * m_y[2] * h_y[1] + m_y[1] * h_y[2]) / (h_y[1] + h_y[2]);
                
                double final_value = interpolateCubic(pixY[i], y0 + 1.0, y0 + 2.0, 
                                                    y_interp[1], y_interp[2], d_y[1], d_y[2]);
                results.push_back(final_value);
            }
        }
        
        // 处理剩余的点
        for (size_t idx = (targetPoints.size() / 4) * 4; idx < targetPoints.size(); ++idx) {
            const auto& point = targetPoints[idx];
            if (point.coordinates.size() < 2) {
                results.push_back(std::nullopt);
                continue;
            }
            
            const double targetX = point.coordinates[0];
            const double targetY = point.coordinates[1];
            
            // 将地理坐标转换为像素索引
            double pix_x = (targetX - originX) / pixelSizeX;
            double pix_y = (targetY - originY) / pixelSizeY;
            
            // 确定插值所需的4x4网格的左上角索引
            int x0 = static_cast<int>(std::floor(pix_x)) - 1;
            int y0 = static_cast<int>(std::floor(pix_y)) - 1;
            
            // 边界检查
            if (x0 < 0 || y0 < 0 || x0 + 3 >= def.cols || y0 + 3 >= def.rows) {
                results.push_back(std::nullopt);
                continue;
            }
            
            // --- 双三次PCHIP插值核心逻辑 ---
            std::vector<double> y_interp(4);
            
            // 1. 沿X方向对4行数据进行插值
            for (int j = 0; j < 4; ++j) {
                int current_y = y0 + j;
                
                // 获取行数据和计算导数
                double v[4], m[3], h[3], d[4];
                for (int i = 0; i < 4; ++i) {
                    // 支持不同的数据类型
                    if (sourceGrid.getDataType() == DataType::Float32) {
                        v[i] = static_cast<double>(sourceGrid.getValue<float>(current_y, x0 + i, 0));
                    } else if (sourceGrid.getDataType() == DataType::Float64) {
                        v[i] = sourceGrid.getValue<double>(current_y, x0 + i, 0);
                    } else {
                        // 其他数据类型，尝试转换
                        v[i] = static_cast<double>(sourceGrid.getValue<float>(current_y, x0 + i, 0));
                    }
                }
                for (int i = 0; i < 3; ++i) {
                    h[i] = 1.0; // 假设网格间距为1
                    m[i] = v[i+1] - v[i];
                }
                
                // 为端点使用更稳健的非中心差分，中间点使用标准PCHIP
                d[0] = (2 * m[0] * h[1] + m[1] * h[0]) / (h[0] + h[1]);
                d[1] = pchipSlope(h[0], h[1], m[0], m[1]);
                d[2] = pchipSlope(h[1], h[2], m[1], m[2]);
                d[3] = (2 * m[2] * h[1] + m[1] * h[2]) / (h[1] + h[2]);
                
                // 在目标x位置进行三次插值
                y_interp[j] = interpolateCubic(pix_x, x0 + 1.0, x0 + 2.0, v[1], v[2], d[1], d[2]);
            }
            
            // 2. 沿Y方向对插值后的数据进行插值
            double m_y[3], h_y[3], d_y[4];
            for (int j = 0; j < 3; ++j) {
                h_y[j] = 1.0;
                m_y[j] = y_interp[j+1] - y_interp[j];
            }
            
            // 同样为Y方向的端点使用非中心差分
            d_y[0] = (2 * m_y[0] * h_y[1] + m_y[1] * h_y[0]) / (h_y[0] + h_y[1]);
            d_y[1] = pchipSlope(h_y[0], h_y[1], m_y[0], m_y[1]);
            d_y[2] = pchipSlope(h_y[1], h_y[2], m_y[1], m_y[2]);
            d_y[3] = (2 * m_y[2] * h_y[1] + m_y[1] * h_y[2]) / (h_y[1] + h_y[2]);
            
            double final_value = interpolateCubic(pix_y, y0 + 1.0, y0 + 2.0, y_interp[1], y_interp[2], d_y[1], d_y[2]);
            results.push_back(final_value);
        }
    } else {
        // 非SIMD路径（原始实现）
        for (const auto& point : targetPoints) {
            if (point.coordinates.size() < 2) {
                results.push_back(std::nullopt);
                continue;
            }

            const double targetX = point.coordinates[0];
            const double targetY = point.coordinates[1];

            // 将地理坐标转换为像素索引
            double pix_x = (targetX - originX) / pixelSizeX;
            double pix_y = (targetY - originY) / pixelSizeY;

            // 确定插值所需的4x4网格的左上角索引 (cell's top-left)
            int x0 = static_cast<int>(std::floor(pix_x)) - 1;
            int y0 = static_cast<int>(std::floor(pix_y)) - 1;

            // 边界检查
            if (x0 < 0 || y0 < 0 || x0 + 3 >= def.cols || y0 + 3 >= def.rows) {
                results.push_back(std::nullopt);
                continue;
            }
            
            // --- 双三次PCHIP插值核心逻辑 ---
            std::vector<double> y_interp(4);

            // 1. 沿X方向对4行数据进行插值
            for (int j = 0; j < 4; ++j) {
                int current_y = y0 + j;
                
                // 获取行数据和计算导数
                double v[4], m[3], h[3], d[4];
                for (int i = 0; i < 4; ++i) {
                    // 支持不同的数据类型
                    if (sourceGrid.getDataType() == DataType::Float32) {
                        v[i] = static_cast<double>(sourceGrid.getValue<float>(current_y, x0 + i, 0));
                    } else if (sourceGrid.getDataType() == DataType::Float64) {
                        v[i] = sourceGrid.getValue<double>(current_y, x0 + i, 0);
                    } else {
                        // 其他数据类型，尝试转换
                        v[i] = static_cast<double>(sourceGrid.getValue<float>(current_y, x0 + i, 0));
                    }
                }
                for (int i = 0; i < 3; ++i) {
                    h[i] = 1.0; // 假设网格间距为1
                    m[i] = v[i+1] - v[i];
                }
                
                // 为端点使用更稳健的非中心差分，中间点使用标准PCHIP
                d[0] = (2 * m[0] * h[1] + m[1] * h[0]) / (h[0] + h[1]);
                d[1] = pchipSlope(h[0], h[1], m[0], m[1]);
                d[2] = pchipSlope(h[1], h[2], m[1], m[2]);
                d[3] = (2 * m[2] * h[1] + m[1] * h[2]) / (h[1] + h[2]);

                // 在目标x位置进行三次插值
                y_interp[j] = interpolateCubic(pix_x, x0 + 1.0, x0 + 2.0, v[1], v[2], d[1], d[2]);
            }

            // 2. 沿Y方向对插值后的数据进行插值
            double m_y[3], h_y[3], d_y[4];
            for (int j = 0; j < 3; ++j) {
                h_y[j] = 1.0;
                m_y[j] = y_interp[j+1] - y_interp[j];
            }
            
            // 同样为Y方向的端点使用非中心差分
            d_y[0] = (2 * m_y[0] * h_y[1] + m_y[1] * h_y[0]) / (h_y[0] + h_y[1]);
            d_y[1] = pchipSlope(h_y[0], h_y[1], m_y[0], m_y[1]);
            d_y[2] = pchipSlope(h_y[1], h_y[2], m_y[1], m_y[2]);
            d_y[3] = (2 * m_y[2] * h_y[1] + m_y[1] * h_y[2]) / (h_y[1] + h_y[2]);
            
            double final_value = interpolateCubic(pix_y, y0 + 1.0, y0 + 2.0, y_interp[1], y_interp[2], d_y[1], d_y[2]);
            results.push_back(final_value);
        }
    }
    
    return { results, 0, "成功" };
}

double PCHIPInterpolator2DBathy::pchipSlope(double h1, double h2, double m1, double m2) const {
    // 这是标准的PCHIP斜率计算，后续可以根据水深特性进行微调
    if (m1 * m2 <= 0.0) {
        return 0.0;
    }
    // 处理斜率为零的情况，防止除以零
    if (m1 == 0.0 || m2 == 0.0) {
        return 0.0;
    }
    // 加权调和平均
    double wh1 = 2.0 * h1 + h2;
    double wh2 = h1 + 2.0 * h2;
    return (wh1 + wh2) / (wh1 / m1 + wh2 / m2);
}

double PCHIPInterpolator2DBathy::interpolateCubic(double x, double x0, double x1,
                                                double y0, double y1,
                                                double d0, double d1) const {
    double h = x1 - x0;
    if (h <= 0.0) return y0; // 避免除以零
    double t = (x - x0) / h;
    double t2 = t * t;
    double t3 = t2 * t;

    // Hermite基函数
    double h00 = 2 * t3 - 3 * t2 + 1;
    double h10 = t3 - 2 * t2 + t;
    double h01 = -2 * t3 + 3 * t2;
    double h11 = t3 - t2;

    return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1;
}

} // namespace oscean::core_services::interpolation 