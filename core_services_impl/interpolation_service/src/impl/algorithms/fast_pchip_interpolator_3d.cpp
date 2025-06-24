#include "fast_pchip_interpolator_3d.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
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

// 宏，用于将3D索引转换为1D向量索引
#define IDX3D(x, y, z, width, depth) ((z) * (width) * (depth) + (y) * (width) + (x))

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

// PCHIP斜率计算的核心辅助函数 (从2D版本移植)
double pchipSlope(double h1, double h2, double m1, double m2) {
    // ... PCHIP斜率计算的完整逻辑 ...
    if (m1 * m2 <= 0.0) {
        return 0.0;
    }
    double wh1 = 2.0 * h1 + h2;
    double wh2 = h1 + 2.0 * h2;
    return (wh1 + wh2) / (wh1 / m1 + wh2 / m2);
}

FastPchipInterpolator3D::FastPchipInterpolator3D(
    const std::shared_ptr<const GridData>& sourceGrid,
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager)
    : sourceGrid_(sourceGrid)
    , simdManager_(simdManager) {

    if (!sourceGrid_ || sourceGrid_->getDataType() != DataType::Float64) {
        throw std::invalid_argument("FastPchipInterpolator3D requires a valid, Float64 GridData.");
    }
    
    const auto& def = sourceGrid_->getDefinition();
    dimX_ = def.cols;
    dimY_ = def.rows;
    dimZ_ = sourceGrid_->getBandCount();

    if (dimX_ < 4 || dimY_ < 4 || dimZ_ < 4) {
        throw std::invalid_argument("FastPchipInterpolator3D requires grid dimensions to be at least 4x4x4.");
    }

    size_t total_points = dimX_ * dimY_ * dimZ_;
    
    // 初始化导数向量
    dervX_.resize(total_points);
    dervY_.resize(total_points);
    dervZ_.resize(total_points);
    dervXY_.resize(total_points);
    dervXZ_.resize(total_points);
    dervYZ_.resize(total_points);
    dervXYZ_.resize(total_points);

    // 预计算所有导数
    computeDerivatives();
}

void FastPchipInterpolator3D::computeDerivatives() {
    // 1. 计算 dF/dx (沿X轴)
    std::vector<double> h(std::max({dimX_, dimY_, dimZ_}));
    std::vector<double> m(std::max({dimX_, dimY_, dimZ_}));
    
    for (size_t z = 0; z < dimZ_; ++z) {
        for (size_t y = 0; y < dimY_; ++y) {
            // 提取当前扫描线的数据
            std::vector<double> line_data(dimX_);
            for (size_t x = 0; x < dimX_; ++x) {
                line_data[x] = sourceGrid_->getValue<double>(y, x, z);
            }

            // 计算h和m
            for (size_t x = 0; x < dimX_ - 1; ++x) {
                h[x] = 1.0; // 假设网格间距为1
                m[x] = line_data[x+1] - line_data[x];
            }

            // 计算端点导数
            if (dimX_ >= 2) {
                dervX_[IDX3D(0, y, z, dimX_, dimY_)] = (dimX_ >= 3) ? pchipSlope(h[0], h[1], m[0], m[1]) : m[0];
                dervX_[IDX3D(dimX_ - 1, y, z, dimX_, dimY_)] = (dimX_ >= 3) ? pchipSlope(h[dimX_ - 2], h[dimX_ - 3], m[dimX_ - 2], m[dimX_ - 3]) : m[dimX_ - 2];
            }

            // 计算中间点导数
            for (size_t x = 1; x < dimX_ - 1; ++x) {
                dervX_[IDX3D(x, y, z, dimX_, dimY_)] = pchipSlope(h[x - 1], h[x], m[x - 1], m[x]);
            }
        }
    }

    // 2. 计算 dF/dy (沿Y轴)
    for (size_t z = 0; z < dimZ_; ++z) {
        for (size_t x = 0; x < dimX_; ++x) {
            // 提取当前扫描线的数据
            std::vector<double> line_data(dimY_);
            for (size_t y = 0; y < dimY_; ++y) {
                line_data[y] = sourceGrid_->getValue<double>(y, x, z);
            }

            // 计算h和m
            for (size_t y = 0; y < dimY_ - 1; ++y) {
                h[y] = 1.0;
                m[y] = line_data[y+1] - line_data[y];
            }

            // 计算端点导数
            if (dimY_ >= 2) {
                dervY_[IDX3D(x, 0, z, dimX_, dimY_)] = (dimY_ >= 3) ? pchipSlope(h[0], h[1], m[0], m[1]) : m[0];
                dervY_[IDX3D(x, dimY_ - 1, z, dimX_, dimY_)] = (dimY_ >= 3) ? pchipSlope(h[dimY_ - 2], h[dimY_ - 3], m[dimY_ - 2], m[dimY_ - 3]) : m[dimY_ - 2];
            }

            // 计算中间点导数
            for (size_t y = 1; y < dimY_ - 1; ++y) {
                dervY_[IDX3D(x, y, z, dimX_, dimY_)] = pchipSlope(h[y - 1], h[y], m[y - 1], m[y]);
            }
        }
    }

    // 3. 计算 dF/dz (沿Z轴)
    for (size_t y = 0; y < dimY_; ++y) {
        for (size_t x = 0; x < dimX_; ++x) {
            // 提取当前扫描线的数据
            std::vector<double> line_data(dimZ_);
            for (size_t z = 0; z < dimZ_; ++z) {
                line_data[z] = sourceGrid_->getValue<double>(y, x, z);
            }

            // 计算h和m
            for (size_t z = 0; z < dimZ_ - 1; ++z) {
                h[z] = 1.0;
                m[z] = line_data[z+1] - line_data[z];
            }

            // 计算端点导数
            if (dimZ_ >= 2) {
                dervZ_[IDX3D(x, y, 0, dimX_, dimY_)] = (dimZ_ >= 3) ? pchipSlope(h[0], h[1], m[0], m[1]) : m[0];
                dervZ_[IDX3D(x, y, dimZ_ - 1, dimX_, dimY_)] = (dimZ_ >= 3) ? pchipSlope(h[dimZ_ - 2], h[dimZ_ - 3], m[dimZ_ - 2], m[dimZ_ - 3]) : m[dimZ_ - 2];
            }

            // 计算中间点导数
            for (size_t z = 1; z < dimZ_ - 1; ++z) {
                dervZ_[IDX3D(x, y, z, dimX_, dimY_)] = pchipSlope(h[z - 1], h[z], m[z - 1], m[z]);
            }
        }
    }

    // 4. 计算交叉导数 (简化版本，使用中心差分)
    // 注意：完整的PCHIP交叉导数计算更复杂，这里使用简化版本
    for (size_t z = 0; z < dimZ_; ++z) {
        for (size_t y = 0; y < dimY_; ++y) {
            for (size_t x = 0; x < dimX_; ++x) {
                // d²F/dxdy
                double dxy = 0.0;
                if (x > 0 && x < dimX_ - 1 && y > 0 && y < dimY_ - 1) {
                    double f_xp_yp = sourceGrid_->getValue<double>(y + 1, x + 1, z);
                    double f_xp_ym = sourceGrid_->getValue<double>(y - 1, x + 1, z);
                    double f_xm_yp = sourceGrid_->getValue<double>(y + 1, x - 1, z);
                    double f_xm_ym = sourceGrid_->getValue<double>(y - 1, x - 1, z);
                    dxy = (f_xp_yp - f_xp_ym - f_xm_yp + f_xm_ym) / 4.0;
                }
                dervXY_[IDX3D(x, y, z, dimX_, dimY_)] = dxy;

                // d²F/dxdz
                double dxz = 0.0;
                if (x > 0 && x < dimX_ - 1 && z > 0 && z < dimZ_ - 1) {
                    double f_xp_zp = sourceGrid_->getValue<double>(y, x + 1, z + 1);
                    double f_xp_zm = sourceGrid_->getValue<double>(y, x + 1, z - 1);
                    double f_xm_zp = sourceGrid_->getValue<double>(y, x - 1, z + 1);
                    double f_xm_zm = sourceGrid_->getValue<double>(y, x - 1, z - 1);
                    dxz = (f_xp_zp - f_xp_zm - f_xm_zp + f_xm_zm) / 4.0;
                }
                dervXZ_[IDX3D(x, y, z, dimX_, dimY_)] = dxz;

                // d²F/dydz
                double dyz = 0.0;
                if (y > 0 && y < dimY_ - 1 && z > 0 && z < dimZ_ - 1) {
                    double f_yp_zp = sourceGrid_->getValue<double>(y + 1, x, z + 1);
                    double f_yp_zm = sourceGrid_->getValue<double>(y + 1, x, z - 1);
                    double f_ym_zp = sourceGrid_->getValue<double>(y - 1, x, z + 1);
                    double f_ym_zm = sourceGrid_->getValue<double>(y - 1, x, z - 1);
                    dyz = (f_yp_zp - f_yp_zm - f_ym_zp + f_ym_zm) / 4.0;
                }
                dervYZ_[IDX3D(x, y, z, dimX_, dimY_)] = dyz;

                // d³F/dxdydz (简化版本)
                dervXYZ_[IDX3D(x, y, z, dimX_, dimY_)] = 0.0;
            }
        }
    }
}

InterpolationResult FastPchipInterpolator3D::execute(const InterpolationRequest& request, const PrecomputedData* precomputed) const {
    if (!request.sourceGrid || request.sourceGrid.get() != sourceGrid_.get()) {
        InterpolationResult result;
        result.data = std::monostate{};
        result.statusCode = -1;
        result.message = "Mismatched source grid for FastPchipInterpolator3D.";
        return result;
    }

    if (!std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        InterpolationResult result;
        result.data = std::monostate{};
        result.statusCode = -1;
        result.message = "FastPchipInterpolator3D only supports point interpolation.";
        return result;
    }

    const auto& points = std::get<std::vector<TargetPoint>>(request.target);
    std::vector<std::optional<double>> results;
    
    // 使用SIMD优化的批量插值（如果可用且点数足够多）
    if (points.size() > 8 && g_cpuCaps.hasAVX2) {
        results = interpolateAtPointsSIMD(points);
    } else {
        // 标量路径
        results.reserve(points.size());

        auto geoTransform = sourceGrid_->getGeoTransform();
        double originX = geoTransform[0];
        double dx = geoTransform[1];
        double originY = geoTransform[3];
        double dy = geoTransform[5];
        
        // 假设Z方向也有均匀间距
        double originZ = 0.0;
        double dz = 1.0;
        if (sourceGrid_->getDefinition().zDimension.coordinates.size() > 1) {
            originZ = sourceGrid_->getDefinition().zDimension.coordinates[0];
            dz = sourceGrid_->getDefinition().zDimension.coordinates[1] - originZ;
        }

        for (const auto& point : points) {
            if (point.coordinates.size() < 3) {
                results.push_back(std::nullopt);
                continue;
            }

            // 世界坐标到网格坐标
            double gridX = (point.coordinates[0] - originX) / dx;
            double gridY = (point.coordinates[1] - originY) / dy;
            double gridZ = (point.coordinates[2] - originZ) / dz;

            // 边界检查
            if (gridX < 0 || gridX >= dimX_ - 1 ||
                gridY < 0 || gridY >= dimY_ - 1 ||
                gridZ < 0 || gridZ >= dimZ_ - 1) {
                results.push_back(std::nullopt);
                continue;
            }

            // 执行3D PCHIP插值
            double result = interpolate3D(gridX, gridY, gridZ);
            results.push_back(result);
        }
    }

    InterpolationResult result;
    result.data = results;
    result.statusCode = 0;
    result.message = "Success";
    return result;
}

double FastPchipInterpolator3D::interpolate3D(double gridX, double gridY, double gridZ) const {
    // 找到包含点的单元格
    size_t x0 = static_cast<size_t>(gridX);
    size_t y0 = static_cast<size_t>(gridY);
    size_t z0 = static_cast<size_t>(gridZ);
    
    size_t x1 = std::min(x0 + 1, dimX_ - 1);
    size_t y1 = std::min(y0 + 1, dimY_ - 1);
    size_t z1 = std::min(z0 + 1, dimZ_ - 1);

    // 归一化坐标
    double tx = gridX - x0;
    double ty = gridY - y0;
    double tz = gridZ - z0;

    // 获取8个角点的值和导数
    double values[8];
    double dx_vals[8], dy_vals[8], dz_vals[8];
    
    int idx = 0;
    for (size_t z = z0; z <= z1; ++z) {
        for (size_t y = y0; y <= y1; ++y) {
            for (size_t x = x0; x <= x1; ++x) {
                values[idx] = sourceGrid_->getValue<double>(y, x, z);
                dx_vals[idx] = dervX_[IDX3D(x, y, z, dimX_, dimY_)];
                dy_vals[idx] = dervY_[IDX3D(x, y, z, dimX_, dimY_)];
                dz_vals[idx] = dervZ_[IDX3D(x, y, z, dimX_, dimY_)];
                idx++;
            }
        }
    }

    // 三次Hermite插值的辅助函数
    auto hermite = [](double t, double p0, double p1, double m0, double m1) {
        double t2 = t * t;
        double t3 = t2 * t;
        double h00 = 2*t3 - 3*t2 + 1;
        double h10 = t3 - 2*t2 + t;
        double h01 = -2*t3 + 3*t2;
        double h11 = t3 - t2;
        return h00*p0 + h10*m0 + h01*p1 + h11*m1;
    };

    // 先在X方向插值（4条线）
    double vx[4], mx[4];
    for (int i = 0; i < 4; ++i) {
        int base = i * 2;
        vx[i] = hermite(tx, values[base], values[base+1], dx_vals[base], dx_vals[base+1]);
        mx[i] = hermite(tx, dy_vals[base], dy_vals[base+1], 
                        dervXY_[IDX3D(x0, y0 + (i/2), z0 + (i%2), dimX_, dimY_)],
                        dervXY_[IDX3D(x1, y0 + (i/2), z0 + (i%2), dimX_, dimY_)]);
    }

    // 然后在Y方向插值（2条线）
    double vy[2], my[2];
    vy[0] = hermite(ty, vx[0], vx[2], mx[0], mx[2]);
    vy[1] = hermite(ty, vx[1], vx[3], mx[1], mx[3]);
    
    // 计算Z方向的导数
    my[0] = hermite(tx, dz_vals[0], dz_vals[1],
                    dervXZ_[IDX3D(x0, y0, z0, dimX_, dimY_)],
                    dervXZ_[IDX3D(x1, y0, z0, dimX_, dimY_)]);
    my[1] = hermite(tx, dz_vals[4], dz_vals[5],
                    dervXZ_[IDX3D(x0, y0, z1, dimX_, dimY_)],
                    dervXZ_[IDX3D(x1, y0, z1, dimX_, dimY_)]);

    // 最后在Z方向插值
    return hermite(tz, vy[0], vy[1], my[0], my[1]);
}

/**
 * @brief SIMD优化的批量插值实现
 */
std::vector<std::optional<double>> FastPchipInterpolator3D::interpolateAtPointsSIMD(
    const std::vector<TargetPoint>& targetPoints) const {
    
    std::vector<std::optional<double>> results;
    results.reserve(targetPoints.size());
    
    auto geoTransform = sourceGrid_->getGeoTransform();
    double originX = geoTransform[0];
    double dx = geoTransform[1];
    double originY = geoTransform[3];
    double dy = geoTransform[5];
    
    // 假设Z方向也有均匀间距
    double originZ = 0.0;
    double dz = 1.0;
    if (sourceGrid_->getDefinition().zDimension.coordinates.size() > 1) {
        originZ = sourceGrid_->getDefinition().zDimension.coordinates[0];
        dz = sourceGrid_->getDefinition().zDimension.coordinates[1] - originZ;
    }
    
#ifdef __AVX2__
    // AVX2优化路径：一次处理4个点
    const size_t simdWidth = 4;
    const size_t numFullBatches = targetPoints.size() / simdWidth;
    
    // 准备AVX2常量
    __m256d vOriginX = _mm256_set1_pd(originX);
    __m256d vOriginY = _mm256_set1_pd(originY);
    __m256d vOriginZ = _mm256_set1_pd(originZ);
    __m256d vDxInv = _mm256_set1_pd(1.0 / dx);
    __m256d vDyInv = _mm256_set1_pd(1.0 / dy);
    __m256d vDzInv = _mm256_set1_pd(1.0 / dz);
    __m256d vMaxX = _mm256_set1_pd(dimX_ - 1.0);
    __m256d vMaxY = _mm256_set1_pd(dimY_ - 1.0);
    __m256d vMaxZ = _mm256_set1_pd(dimZ_ - 1.0);
    __m256d vZero = _mm256_setzero_pd();
    
    // 对齐的临时数组
    alignas(32) double xCoords[4];
    alignas(32) double yCoords[4];
    alignas(32) double zCoords[4];
    alignas(32) double gridXs[4];
    alignas(32) double gridYs[4];
    alignas(32) double gridZs[4];
    
    // 批量处理
    for (size_t batch = 0; batch < numFullBatches; ++batch) {
        size_t baseIdx = batch * simdWidth;
        
        // 收集4个点的坐标
        bool allValid = true;
        for (size_t i = 0; i < simdWidth; ++i) {
            const auto& point = targetPoints[baseIdx + i];
            if (point.coordinates.size() < 3) {
                allValid = false;
                break;
            }
            xCoords[i] = point.coordinates[0];
            yCoords[i] = point.coordinates[1];
            zCoords[i] = point.coordinates[2];
        }
        
        if (!allValid) {
            // 如果有无效点，回退到标量处理
            for (size_t i = 0; i < simdWidth; ++i) {
                const auto& point = targetPoints[baseIdx + i];
                if (point.coordinates.size() < 3) {
                    results.push_back(std::nullopt);
                } else {
                    double gridX = (point.coordinates[0] - originX) / dx;
                    double gridY = (point.coordinates[1] - originY) / dy;
                    double gridZ = (point.coordinates[2] - originZ) / dz;
                    
                    if (gridX < 0 || gridX >= dimX_ - 1 ||
                        gridY < 0 || gridY >= dimY_ - 1 ||
                        gridZ < 0 || gridZ >= dimZ_ - 1) {
                        results.push_back(std::nullopt);
                    } else {
                        results.push_back(interpolate3D(gridX, gridY, gridZ));
                    }
                }
            }
            continue;
        }
        
        // 加载坐标到AVX2寄存器
        __m256d vX = _mm256_load_pd(xCoords);
        __m256d vY = _mm256_load_pd(yCoords);
        __m256d vZ = _mm256_load_pd(zCoords);
        
        // 转换到网格坐标
        __m256d vGridX = _mm256_mul_pd(_mm256_sub_pd(vX, vOriginX), vDxInv);
        __m256d vGridY = _mm256_mul_pd(_mm256_sub_pd(vY, vOriginY), vDyInv);
        __m256d vGridZ = _mm256_mul_pd(_mm256_sub_pd(vZ, vOriginZ), vDzInv);
        
        // 边界检查
        __m256d vValidX = _mm256_and_pd(
            _mm256_cmp_pd(vGridX, vZero, _CMP_GE_OQ),
            _mm256_cmp_pd(vGridX, vMaxX, _CMP_LT_OQ)
        );
        __m256d vValidY = _mm256_and_pd(
            _mm256_cmp_pd(vGridY, vZero, _CMP_GE_OQ),
            _mm256_cmp_pd(vGridY, vMaxY, _CMP_LT_OQ)
        );
        __m256d vValidZ = _mm256_and_pd(
            _mm256_cmp_pd(vGridZ, vZero, _CMP_GE_OQ),
            _mm256_cmp_pd(vGridZ, vMaxZ, _CMP_LT_OQ)
        );
        __m256d vValid = _mm256_and_pd(vValidX, _mm256_and_pd(vValidY, vValidZ));
        
        // 存储网格坐标
        _mm256_store_pd(gridXs, vGridX);
        _mm256_store_pd(gridYs, vGridY);
        _mm256_store_pd(gridZs, vGridZ);
        
        // 检查有效性掩码
        int validMask = _mm256_movemask_pd(vValid);
        
        // 对每个点进行插值
        for (size_t i = 0; i < simdWidth; ++i) {
            if (validMask & (1 << i)) {
                // 有效点，执行插值
                results.push_back(interpolate3D(gridXs[i], gridYs[i], gridZs[i]));
            } else {
                // 无效点
                results.push_back(std::nullopt);
            }
        }
    }
    
    // 处理剩余的点
    for (size_t i = numFullBatches * simdWidth; i < targetPoints.size(); ++i) {
        const auto& point = targetPoints[i];
        if (point.coordinates.size() < 3) {
            results.push_back(std::nullopt);
            continue;
        }
        
        double gridX = (point.coordinates[0] - originX) / dx;
        double gridY = (point.coordinates[1] - originY) / dy;
        double gridZ = (point.coordinates[2] - originZ) / dz;
        
        if (gridX < 0 || gridX >= dimX_ - 1 ||
            gridY < 0 || gridY >= dimY_ - 1 ||
            gridZ < 0 || gridZ >= dimZ_ - 1) {
            results.push_back(std::nullopt);
        } else {
            results.push_back(interpolate3D(gridX, gridY, gridZ));
        }
    }
#else
    // 非AVX2路径
    for (const auto& point : targetPoints) {
        if (point.coordinates.size() < 3) {
            results.push_back(std::nullopt);
            continue;
        }
        
        double gridX = (point.coordinates[0] - originX) / dx;
        double gridY = (point.coordinates[1] - originY) / dy;
        double gridZ = (point.coordinates[2] - originZ) / dz;
        
        if (gridX < 0 || gridX >= dimX_ - 1 ||
            gridY < 0 || gridY >= dimY_ - 1 ||
            gridZ < 0 || gridZ >= dimZ_ - 1) {
            results.push_back(std::nullopt);
        } else {
            results.push_back(interpolate3D(gridX, gridY, gridZ));
        }
    }
#endif
    
    return results;
}

} // namespace oscean::core_services::interpolation 