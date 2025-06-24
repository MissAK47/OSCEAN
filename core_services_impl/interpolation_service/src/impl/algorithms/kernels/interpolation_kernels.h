#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <optional>
#include <string>
#include <memory>
#include <map>

// 前向声明
namespace oscean::common_utils::simd {
    class ISIMDManager;
}

// 前向声明boost future，避免包含冲突
namespace boost {
    template<typename T>
    class future;
}

namespace oscean::core_services::interpolation::kernels {

/**
 * @brief 双线性插值核心计算
 * @param values 四个角点的值 [v00, v10, v01, v11]
 * @param fx X方向的插值权重 (0-1)
 * @param fy Y方向的插值权重 (0-1)
 * @return 插值结果
 */
inline double bilinear(const std::array<double, 4>& values, double fx, double fy) {
    double v0 = values[0] * (1 - fx) + values[1] * fx;
    double v1 = values[2] * (1 - fx) + values[3] * fx;
    return v0 * (1 - fy) + v1 * fy;
}

/**
 * @brief 三线性插值核心计算
 * @param values 八个角点的值
 * @param fx X方向的插值权重
 * @param fy Y方向的插值权重
 * @param fz Z方向的插值权重
 * @return 插值结果
 */
inline double trilinear(const std::array<double, 8>& values, double fx, double fy, double fz) {
    // 先在Z=0平面做双线性插值
    double v00 = values[0] * (1 - fx) + values[1] * fx;
    double v01 = values[2] * (1 - fx) + values[3] * fx;
    double v0 = v00 * (1 - fy) + v01 * fy;
    
    // 再在Z=1平面做双线性插值
    double v10 = values[4] * (1 - fx) + values[5] * fx;
    double v11 = values[6] * (1 - fx) + values[7] * fx;
    double v1 = v10 * (1 - fy) + v11 * fy;
    
    // 最后在Z方向插值
    return v0 * (1 - fz) + v1 * fz;
}

/**
 * @brief PCHIP斜率计算
 * @param h1 第一个区间长度
 * @param h2 第二个区间长度
 * @param m1 第一个区间斜率
 * @param m2 第二个区间斜率
 * @return PCHIP导数
 */
inline double pchipSlope(double h1, double h2, double m1, double m2) {
    if (m1 * m2 <= 0.0) {
        return 0.0;
    }
    double wh1 = 2.0 * h1 + h2;
    double wh2 = h1 + 2.0 * h2;
    return (wh1 + wh2) / (wh1 / m1 + wh2 / m2);
}

/**
 * @brief 立方Hermite插值
 * @param p0 起点值
 * @param p1 终点值
 * @param m0 起点导数
 * @param m1 终点导数
 * @param t 插值参数 (0-1)
 * @return 插值结果
 */
inline double cubicHermite(double p0, double p1, double m0, double m1, double t) {
    double t2 = t * t;
    double t3 = t2 * t;
    
    double h00 = 2 * t3 - 3 * t2 + 1;
    double h10 = t3 - 2 * t2 + t;
    double h01 = -2 * t3 + 3 * t2;
    double h11 = t3 - t2;
    
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
}

/**
 * @brief 最近邻插值
 * @param values 四个角点的值
 * @param wx X方向权重
 * @param wy Y方向权重
 * @return 最近邻的值
 */
double nearestNeighbor(const std::array<double, 4>& values, double wx, double wy);

/**
 * @brief 1D线性插值
 * @param v0 起点值
 * @param v1 终点值
 * @param t 插值参数 (0到1)
 * @return 插值结果
 */
double linear1D(double v0, double v1, double t);

/**
 * @brief 计算埃尔米特多项式值
 * @details P(t) = (2t^3 - 3t^2 + 1)y_k + (t^3 - 2t^2 + t)h_k*m_k + (-2t^3 + 3t^2)y_{k+1} + (t^3 - t^2)h_k*m_{k+1}
 * @param t_normalized 归一化参数 (0到1)
 * @param y_k 起点值
 * @param y_k_plus_1 终点值
 * @param slope_k 起点斜率
 * @param slope_k_plus_1 终点斜率
 * @param h_k 段长度
 * @return 插值结果
 */
inline double evaluateHermitePolynomial(
    double t_normalized,
    double y_k, double y_k_plus_1,
    double slope_k, double slope_k_plus_1,
    double h_k) {

    double t2 = t_normalized * t_normalized;
    double t3 = t2 * t_normalized;

    double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    double h10 = t3 - 2.0 * t2 + t_normalized;
    double h01 = -2.0 * t3 + 3.0 * t2;
    double h11 = t3 - t2;

    return h00 * y_k + h10 * h_k * slope_k + h01 * y_k_plus_1 + h11 * h_k * slope_k_plus_1;
}

/**
 * @brief 反距离权重插值
 * @param distances 到各点的距离
 * @param values 各点的值
 * @param power 权重指数（通常为2）
 * @return 插值结果，如果计算失败返回nullopt
 */
std::optional<double> inverseDistanceWeighting(
    const std::vector<double>& distances,
    const std::vector<double>& values,
    double power = 2.0);

/**
 * @brief 检查值是否有效（非NaN、非Inf）
 * @param value 要检查的值
 * @return 如果有效返回true
 */
bool isValidValue(double value);

/**
 * @brief 安全的插值计算（处理无效值）
 * @param values 输入值数组
 * @param weights 权重数组
 * @return 如果计算成功返回结果，否则返回nullopt
 */
template<size_t N>
std::optional<double> safeWeightedSum(
    const std::array<double, N>& values,
    const std::array<double, N>& weights);

/**
 * @brief 双三次插值（Bicubic）
 * @param values 16个控制点的值（4x4网格）
 * @param wx X方向权重
 * @param wy Y方向权重
 * @return 插值结果
 */
double bicubic(const std::array<double, 16>& values, double wx, double wy);

/**
 * @brief 三次样条插值的基函数
 * @param t 参数 (0到1)
 * @param p0 控制点0
 * @param p1 控制点1
 * @param p2 控制点2
 * @param p3 控制点3
 * @return 插值结果
 */
double cubicSpline(double t, double p0, double p1, double p2, double p3);

// SIMD优化的批量插值函数
namespace simd {

/**
 * @brief SIMD优化的批量双线性插值
 * @param valuesBatch 批量的四角点值数组
 * @param wxBatch X方向权重批量
 * @param wyBatch Y方向权重批量
 * @return 批量插值结果
 */
std::vector<double> batchBilinear(
    const std::vector<std::array<double, 4>>& valuesBatch,
    const std::vector<double>& wxBatch,
    const std::vector<double>& wyBatch);

/**
 * @brief SIMD优化的批量三线性插值
 * @param valuesBatch 批量的八角点值数组
 * @param wxBatch X方向权重批量
 * @param wyBatch Y方向权重批量
 * @param wzBatch Z方向权重批量
 * @return 批量插值结果
 */
std::vector<double> batchTrilinear(
    const std::vector<std::array<double, 8>>& valuesBatch,
    const std::vector<double>& wxBatch,
    const std::vector<double>& wyBatch,
    const std::vector<double>& wzBatch);

/**
 * @brief SIMD优化的批量最近邻插值
 * @param valuesBatch 批量的四角点值数组
 * @param wxBatch X方向权重批量
 * @param wyBatch Y方向权重批量
 * @return 批量插值结果
 */
std::vector<double> batchNearestNeighbor(
    const std::vector<std::array<double, 4>>& valuesBatch,
    const std::vector<double>& wxBatch,
    const std::vector<double>& wyBatch);

/**
 * @brief 高性能网格插值（使用SIMD管理器）
 * @param simdManager SIMD管理器
 * @param gridData 网格数据
 * @param xCoords X坐标数组
 * @param yCoords Y坐标数组
 * @param results 结果数组
 * @param count 点数量
 * @param gridWidth 网格宽度
 * @param gridHeight 网格高度
 * @param method 插值方法
 */
void performanceGridInterpolation(
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    const float* gridData,
    const float* xCoords,
    const float* yCoords,
    float* results,
    size_t count,
    size_t gridWidth,
    size_t gridHeight,
    const std::string& method = "bilinear");

/**
 * @brief 异步批量插值处理
 * @param simdManager SIMD管理器
 * @param gridData 网格数据
 * @param points 目标点集合
 * @param method 插值方法
 * @return 异步插值结果
 */
boost::future<std::vector<double>> asyncBatchInterpolation(
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    const float* gridData,
    const std::vector<std::pair<float, float>>& points,
    size_t gridWidth,
    size_t gridHeight,
    const std::string& method = "bilinear");

/**
 * @brief 自适应批量大小优化
 * @param simdManager SIMD管理器
 * @param dataSize 数据大小
 * @param complexity 算法复杂度
 * @return 推荐的批量大小
 */
size_t getOptimalBatchSize(
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    size_t dataSize,
    double complexity = 1.0);

/**
 * @brief 内存对齐优化的数据准备
 * @param simdManager SIMD管理器
 * @param inputData 输入数据
 * @param size 数据大小
 * @return 对齐后的数据指针
 */
std::unique_ptr<float[], void(*)(float*)> prepareAlignedData(
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    const float* inputData,
    size_t size);

} // namespace simd

// 模板函数实现
template<size_t N>
std::optional<double> safeWeightedSum(
    const std::array<double, N>& values,
    const std::array<double, N>& weights) {
    
    double sum = 0.0;
    double weightSum = 0.0;
    
    for (size_t i = 0; i < N; ++i) {
        if (isValidValue(values[i]) && isValidValue(weights[i])) {
            sum += values[i] * weights[i];
            weightSum += weights[i];
        }
    }
    
    if (weightSum < 1e-10) {
        return std::nullopt;
    }
    
    return sum / weightSum;
}

/**
 * @brief 双线性插值
 * @param v00 左上角值
 * @param v10 右上角值  
 * @param v01 左下角值
 * @param v11 右下角值
 * @param fx X方向插值因子 [0,1]
 * @param fy Y方向插值因子 [0,1]
 * @return 插值结果
 */
inline double bilinearInterpolate(double v00, double v10, double v01, double v11, 
                                 double fx, double fy) noexcept {
    return v00 * (1.0 - fx) * (1.0 - fy) + 
           v10 * fx * (1.0 - fy) + 
           v01 * (1.0 - fx) * fy + 
           v11 * fx * fy;
}

} // namespace oscean::core_services::interpolation::kernels 