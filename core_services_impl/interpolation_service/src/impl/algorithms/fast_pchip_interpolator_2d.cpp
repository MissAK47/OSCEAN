#include "fast_pchip_interpolator_2d.h"
#include <stdexcept>
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

// 临时包含，用于访问我们自己的PCHIP实现的一些内部逻辑或类型
// 实际应用中可能需要重构
#include "pchip_interpolator.h" 

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

// Helper to access 1D vector as a 2D matrix
#define IDX(col, row, width) ((row) * (width) + (col))

FastPchipInterpolator2D::FastPchipInterpolator2D(
    const std::shared_ptr<const GridData>& sourceGrid,
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
    : sourceGrid_(sourceGrid)
    , simdManager_(simdManager) {

    if (!sourceGrid_ || sourceGrid_->getBandCount() != 1 || sourceGrid_->getDataType() != DataType::Float64) {
        throw std::invalid_argument("FastPchipInterpolator2D requires a valid, single-band, Float64 GridData.");
    }
    
    const auto& def = sourceGrid_->getDefinition();
    cols_ = def.cols;
    rows_ = def.rows;

    if (cols_ < 4 || rows_ < 4) {
        throw std::invalid_argument("FastPchipInterpolator2D requires grid dimensions to be at least 4x4.");
    }

    dervX_.resize(cols_ * rows_);
    dervY_.resize(cols_ * rows_);
    dervXY_.resize(cols_ * rows_);

    auto geoTransform = sourceGrid_->getGeoTransform();
    double dx = geoTransform[1];
    double dy = geoTransform[5];
    
    if (std::abs(dx) < 1e-9 || std::abs(dy) < 1e-9) {
        throw std::invalid_argument("Grid spacing (dx, dy) cannot be zero.");
    }

    // Pre-construct all derivatives and cross-derivatives once to save time
    // This logic is adapted from USML's data_grid_bathy.h
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = 0; c < cols_; ++c) {
            double inc_x_prev = (c > 0) ? dx : dx;
            double inc_x_next = (c < cols_ - 1) ? dx : dx;
            double hx = inc_x_prev + inc_x_next;

            double inc_y_prev = (r > 0) ? dy : dy;
            double inc_y_next = (r < rows_ - 1) ? dy : dy;
            double hy = inc_y_prev + inc_y_next;

            // X Derivative (Centered difference, with forward/backward at edges)
            if (c == 0) {
                dervX_[IDX(c, r, cols_)] = (getGridValue(c + 1, r) - getGridValue(c, r)) / inc_x_next;
            } else if (c == cols_ - 1) {
                dervX_[IDX(c, r, cols_)] = (getGridValue(c, r) - getGridValue(c - 1, r)) / inc_x_prev;
            } else {
                dervX_[IDX(c, r, cols_)] = (getGridValue(c + 1, r) - getGridValue(c - 1, r)) / hx;
            }

            // Y Derivative (Centered difference, with forward/backward at edges)
            if (r == 0) {
                dervY_[IDX(c, r, cols_)] = (getGridValue(c, r + 1) - getGridValue(c, r)) / inc_y_next;
            } else if (r == rows_ - 1) {
                dervY_[IDX(c, r, cols_)] = (getGridValue(c, r) - getGridValue(c, r - 1)) / inc_y_prev;
            } else {
                dervY_[IDX(c, r, cols_)] = (getGridValue(c, r + 1) - getGridValue(c, r - 1)) / hy;
            }

            // XY Mixed Derivative
            double f_c_p1_r_p1 = getGridValue(std::min(c + 1, cols_ - 1), std::min(r + 1, rows_ - 1));
            double f_c_p1_r_m1 = getGridValue(std::min(c + 1, cols_ - 1), std::max(0, (int)r - 1));
            double f_c_m1_r_p1 = getGridValue(std::max(0, (int)c - 1), std::min(r + 1, rows_ - 1));
            double f_c_m1_r_m1 = getGridValue(std::max(0, (int)c - 1), std::max(0, (int)r - 1));
            
            dervXY_[IDX(c, r, cols_)] = (f_c_p1_r_p1 - f_c_p1_r_m1 - f_c_m1_r_p1 + f_c_m1_r_m1) / (hx * hy);
        }
    }
}

InterpolationResult FastPchipInterpolator2D::execute(const InterpolationRequest& request, const PrecomputedData* precomputed) const {
    // FastPchipInterpolator2D使用构造时传入的sourceGrid_，忽略request中的sourceGrid
    // 这是因为导数已经预计算好了
    if (!sourceGrid_) {
        InterpolationResult result;
        result.data = std::monostate{};
        result.statusCode = -1;
        result.message = "No source grid available in FastPchipInterpolator2D.";
        return result;
    }

    if (!std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        InterpolationResult result;
        result.data = std::monostate{};
        result.statusCode = -1;
        result.message = "FastPchipInterpolator2D only supports point interpolation.";
        return result;
    }

    const auto& points = std::get<std::vector<TargetPoint>>(request.target);
    std::vector<std::optional<double>> results;
    
    // 使用SIMD优化的批量插值
    if (points.size() > 4 && g_cpuCaps.hasAVX2) {
        results = interpolateAtPointsSIMD(points);
    } else {
        results.reserve(points.size());
        for (const auto& point : points) {
            if (point.coordinates.size() < 2) {
                results.push_back(std::nullopt);
                continue;
            }

            double location[2] = { point.coordinates[0], point.coordinates[1] };
            // Derivatives are not yet implemented for the fast version
            results.push_back(fastPchip(location, nullptr));
        }
    }

    InterpolationResult result;
    result.data = results;
    result.statusCode = 0;
    result.message = "Success";
    return result;
}

// 新增：内联SIMD优化的批量插值实现
std::vector<std::optional<double>> FastPchipInterpolator2D::interpolateAtPointsSIMD(
    const std::vector<TargetPoint>& targetPoints) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());
    
    auto geoTransform = sourceGrid_->getGeoTransform();
    const double originX = geoTransform[0];
    const double dx = geoTransform[1];
    const double originY = geoTransform[3];
    const double dy = geoTransform[5];
    
    const double* dataPtr = static_cast<const double*>(sourceGrid_->getDataPtr());
    
    #ifdef __AVX2__
    if (g_cpuCaps.hasAVX2) {
        // AVX2优化路径（处理double类型，一次4个点）
        size_t i = 0;
        
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
            __m256d origin_x_vec = _mm256_set1_pd(originX);
            __m256d dx_vec = _mm256_set1_pd(dx);
            __m256d origin_y_vec = _mm256_set1_pd(originY);
            __m256d dy_vec = _mm256_set1_pd(dy);
            
            __m256d grid_x = _mm256_div_pd(_mm256_sub_pd(x_coords, origin_x_vec), dx_vec);
            __m256d grid_y = _mm256_div_pd(_mm256_sub_pd(y_coords, origin_y_vec), dy_vec);
            
            // 提取网格坐标
            alignas(32) double grid_x_arr[4], grid_y_arr[4];
            _mm256_store_pd(grid_x_arr, grid_x);
            _mm256_store_pd(grid_y_arr, grid_y);
            
            // 处理每个点（PCHIP需要访问16个值，难以完全向量化）
            for (int j = 0; j < 4; ++j) {
                double gx = grid_x_arr[j];
                double gy = grid_y_arr[j];
                
                // 边界检查
                size_t c0 = static_cast<size_t>(gx);
                size_t r0 = static_cast<size_t>(gy);
                
                if (gx < 0 || gy < 0 || c0 >= cols_ - 1 || r0 >= rows_ - 1) {
                    results.push_back(std::nullopt);
                    continue;
                }
                
                // 使用SIMD优化的PCHIP计算
                double result = fastPchipSIMD(gx, gy, c0, r0);
                results.push_back(result);
            }
        }
        
        // 处理剩余的点
        for (; i < targetPoints.size(); ++i) {
            if (targetPoints[i].coordinates.size() < 2) {
                results.push_back(std::nullopt);
                continue;
            }
            
            double location[2] = { targetPoints[i].coordinates[0], targetPoints[i].coordinates[1] };
            results.push_back(fastPchip(location, nullptr));
        }
        
        return results;
    }
    #endif
    
    // 非SIMD路径
    for (const auto& point : targetPoints) {
        if (point.coordinates.size() < 2) {
            results.push_back(std::nullopt);
            continue;
        }
        
        double location[2] = { point.coordinates[0], point.coordinates[1] };
        results.push_back(fastPchip(location, nullptr));
    }
    
    return results;
}

// 新增：SIMD优化的PCHIP核心计算
double FastPchipInterpolator2D::fastPchipSIMD(double gridX, double gridY, size_t c0, size_t r0) const {
    #ifdef __AVX2__
    if (g_cpuCaps.hasAVX2) {
        auto geoTransform = sourceGrid_->getGeoTransform();
        const double dx = geoTransform[1];
        const double dy = geoTransform[5];
        
        size_t c1 = c0 + 1;
        size_t r1 = r0 + 1;
        
        // 归一化坐标
        double t_x = gridX - c0;
        double t_y = gridY - r0;
        
        // 使用AVX2加载4个角点的值和导数
        __m256d p_vec = _mm256_set_pd(
            getGridValue(c1, r1),
            getGridValue(c0, r1),
            getGridValue(c1, r0),
            getGridValue(c0, r0)
        );
        
        __m256d fx_vec = _mm256_set_pd(
            dervX_[IDX(c1, r1, cols_)] * dx,
            dervX_[IDX(c0, r1, cols_)] * dx,
            dervX_[IDX(c1, r0, cols_)] * dx,
            dervX_[IDX(c0, r0, cols_)] * dx
        );
        
        __m256d fy_vec = _mm256_set_pd(
            dervY_[IDX(c1, r1, cols_)] * dy,
            dervY_[IDX(c0, r1, cols_)] * dy,
            dervY_[IDX(c1, r0, cols_)] * dy,
            dervY_[IDX(c0, r0, cols_)] * dy
        );
        
        __m256d fxy_vec = _mm256_set_pd(
            dervXY_[IDX(c1, r1, cols_)] * dx * dy,
            dervXY_[IDX(c0, r1, cols_)] * dx * dy,
            dervXY_[IDX(c1, r0, cols_)] * dx * dy,
            dervXY_[IDX(c0, r0, cols_)] * dx * dy
        );
        
        // 提取数组用于计算
        alignas(32) double p[4], fx[4], fy[4], fxy[4];
        _mm256_store_pd(p, p_vec);
        _mm256_store_pd(fx, fx_vec);
        _mm256_store_pd(fy, fy_vec);
        _mm256_store_pd(fxy, fxy_vec);
        
        // 使用SIMD优化的Hermite基函数
        auto pchip1dSIMD = [](double t, double p0, double p1, double m0, double m1) -> double {
            __m128d t_vec = _mm_set1_pd(t);
            __m128d t2_vec = _mm_mul_pd(t_vec, t_vec);
            __m128d one = _mm_set1_pd(1.0);
            __m128d two = _mm_set1_pd(2.0);
            __m128d three = _mm_set1_pd(3.0);
            
            __m128d mt = _mm_sub_pd(one, t_vec);
            __m128d mt2 = _mm_mul_pd(mt, mt);
            
            // h00 = (1 + 2t) * (1-t)^2
            __m128d h00 = _mm_mul_pd(
                _mm_add_pd(one, _mm_mul_pd(two, t_vec)),
                mt2
            );
            
            // h10 = t * (1-t)^2
            __m128d h10 = _mm_mul_pd(t_vec, mt2);
            
            // h01 = t^2 * (3 - 2t)
            __m128d h01 = _mm_mul_pd(
                t2_vec,
                _mm_sub_pd(three, _mm_mul_pd(two, t_vec))
            );
            
            // h11 = t^2 * (t - 1)
            __m128d h11 = _mm_mul_pd(t2_vec, _mm_sub_pd(t_vec, one));
            
            // 计算结果
            __m128d result = _mm_add_pd(
                _mm_add_pd(
                    _mm_mul_pd(_mm_set1_pd(p0), h00),
                    _mm_mul_pd(_mm_set1_pd(m0), h10)
                ),
                _mm_add_pd(
                    _mm_mul_pd(_mm_set1_pd(p1), h01),
                    _mm_mul_pd(_mm_set1_pd(m1), h11)
                )
            );
            
            return _mm_cvtsd_f64(result);
        };
        
        // X方向的PCHIP插值
        double v0 = pchip1dSIMD(t_x, p[0], p[1], fx[0], fx[1]);
        double v1 = pchip1dSIMD(t_x, p[2], p[3], fx[2], fx[3]);
        
        // Y方向导数的插值
        double m0y = pchip1dSIMD(t_x, fy[0], fy[1], fxy[0], fxy[1]);
        double m1y = pchip1dSIMD(t_x, fy[2], fy[3], fxy[2], fxy[3]);
        
        // 最终Y方向的PCHIP插值
        return pchip1dSIMD(t_y, v0, v1, m0y, m1y);
    }
    #endif
    
    // 回退到标量实现
    double location[2] = { gridX, gridY };
    return fastPchip(location, nullptr);
}

double FastPchipInterpolator2D::fastPchip(const double location[2], double* derivative) const {
    auto geoTransform = sourceGrid_->getGeoTransform();
    double originX = geoTransform[0];
    double dx = geoTransform[1];
    double originY = geoTransform[3];
    double dy = geoTransform[5];

    // World to grid coordinates
    double gridX = (location[0] - originX) / dx;
    double gridY = (location[1] - originY) / dy;

    // 边界检查（修复：gridX和gridY可能为负数）
    if (gridX < 0 || gridY < 0 || gridX >= cols_ - 1 || gridY >= rows_ - 1) {
        // 边界外的点返回NaN
        return std::nan(""); 
    }

    // Find indices
    size_t c0 = static_cast<size_t>(gridX);
    size_t r0 = static_cast<size_t>(gridY);
    size_t c1 = c0 + 1;
    size_t r1 = r0 + 1;

    // Normalized coordinates
    double t_x = gridX - c0;
    double t_y = gridY - r0;
    
    // Get values and derivatives at the 4 corners of the cell
    double p[4] = {
        getGridValue(c0, r0), getGridValue(c1, r0),
        getGridValue(c0, r1), getGridValue(c1, r1)
    };
    double fx[4] = {
        dervX_[IDX(c0, r0, cols_)] * dx, dervX_[IDX(c1, r0, cols_)] * dx,
        dervX_[IDX(c0, r1, cols_)] * dx, dervX_[IDX(c1, r1, cols_)] * dx,
    };
    double fy[4] = {
        dervY_[IDX(c0, r0, cols_)] * dy, dervY_[IDX(c1, r0, cols_)] * dy,
        dervY_[IDX(c0, r1, cols_)] * dy, dervY_[IDX(c1, r1, cols_)] * dy,
    };
    double fxy[4] = {
        dervXY_[IDX(c0, r0, cols_)] * dx * dy, dervXY_[IDX(c1, r0, cols_)] * dx * dy,
        dervXY_[IDX(c0, r1, cols_)] * dx * dy, dervXY_[IDX(c1, r1, cols_)] * dx * dy,
    };

    // 1D PCHIP in x-direction for y=r0 and y=r1
    // 优化：使用Horner方法减少乘法运算
    auto pchip1d = [](double t, double p0, double p1, double m0, double m1) {
        const double t2 = t * t;
        const double mt = 1.0 - t;
        const double mt2 = mt * mt;
        
        // 使用更高效的Hermite基函数计算
        const double h00 = (1.0 + 2.0 * t) * mt2;
        const double h10 = t * mt2;
        const double h01 = t2 * (3.0 - 2.0 * t);
        const double h11 = t2 * (t - 1.0);
        
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
    };

    double v0 = pchip1d(t_x, p[0], p[1], fx[0], fx[1]);
    double v1 = pchip1d(t_x, p[2], p[3], fx[2], fx[3]);
    
    // 1D PCHIP for derivatives in y-direction
    double m0y = pchip1d(t_x, fy[0], fy[1], fxy[0], fxy[1]);
    double m1y = pchip1d(t_x, fy[2], fy[3], fxy[2], fxy[3]);
    
    // Final 1D PCHIP in y-direction
    double result = pchip1d(t_y, v0, v1, m0y, m1y);

    if(derivative) {
        // Derivative calculation not fully implemented yet, similar to USML it requires more terms.
        derivative[0] = 0.0;
        derivative[1] = 0.0;
    }

    return result;
}


double FastPchipInterpolator2D::getGridValue(size_t col, size_t row) const {
    // Clamp coordinates to be within grid bounds
    size_t c = std::min(col, cols_ - 1);
    size_t r = std::min(row, rows_ - 1);
    
    const double* dataPtr = static_cast<const double*>(sourceGrid_->getDataPtr());
    return dataPtr[IDX(c, r, cols_)];
}


} // namespace oscean::core_services::interpolation 