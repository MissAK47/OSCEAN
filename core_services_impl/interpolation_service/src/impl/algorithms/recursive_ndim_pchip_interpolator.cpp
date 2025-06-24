#include "recursive_ndim_pchip_interpolator.h"
#include <algorithm>
#include <cmath>

namespace oscean::core_services::interpolation {

RecursiveNDimPCHIPInterpolator::RecursiveNDimPCHIPInterpolator(
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
    const std::vector<DimensionMethod>& methods)
    : LayoutAwareInterpolatorBase(simdManager)
    , dimensionMethods_(methods) {
}

InterpolationResult RecursiveNDimPCHIPInterpolator::execute(
    const InterpolationRequest& request,
    const PrecomputedData* precomputed) const {
    
    InterpolationResult result;
    
    if (!request.sourceGrid) {
        result.statusCode = -1;
        result.message = "Source grid is null";
        return result;
    }
    
    const auto& grid = *request.sourceGrid;
    
    // 确定维度数
    size_t numDims = 0;
    const auto& def = grid.getDefinition();
    if (!def.xDimension.coordinates.empty()) numDims++;
    if (!def.yDimension.coordinates.empty()) numDims++;
    if (!def.zDimension.coordinates.empty()) numDims++;
    
    // 设置默认方法（如果未指定）
    if (dimensionMethods_.empty()) {
        // 创建临时变量来修改
        std::vector<DimensionMethod> tempMethods(numDims, DimensionMethod::PCHIP);
        const_cast<std::vector<DimensionMethod>&>(dimensionMethods_) = tempMethods;
    }
    
    // 创建布局感知访问器
    LayoutAwareAccessor accessor(grid);
    
    // 处理不同类型的目标
    if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        const auto& points = std::get<std::vector<TargetPoint>>(request.target);
        std::vector<std::optional<double>> values;
        values.reserve(points.size());
        
        for (const auto& point : points) {
            // 转换世界坐标到网格坐标
            std::vector<double> gridCoords;
            
            if (point.coordinates.size() > 0) {
                double gridX = (point.coordinates[0] - def.xDimension.minValue) /
                              (def.xDimension.maxValue - def.xDimension.minValue) *
                              (def.xDimension.coordinates.size() - 1);
                gridCoords.push_back(gridX);
            }
            
            if (point.coordinates.size() > 1) {
                double gridY = (point.coordinates[1] - def.yDimension.minValue) /
                              (def.yDimension.maxValue - def.yDimension.minValue) *
                              (def.yDimension.coordinates.size() - 1);
                gridCoords.push_back(gridY);
            }
            
            if (point.coordinates.size() > 2 && !def.zDimension.coordinates.empty()) {
                double gridZ = (point.coordinates[2] - def.zDimension.minValue) /
                              (def.zDimension.maxValue - def.zDimension.minValue) *
                              (def.zDimension.coordinates.size() - 1);
                gridCoords.push_back(gridZ);
            }
            
            // 执行递归插值
            std::vector<size_t> indices(numDims, 0);
            auto value = interpRecursive(grid, numDims - 1, indices, gridCoords, accessor);
            values.push_back(value);
        }
        
        result.data = std::move(values);
        result.statusCode = 0;
        
    } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
        // TODO: 实现网格到网格的插值
        result.statusCode = -1;
        result.message = "Grid-to-grid interpolation not yet implemented";
    }
    
    return result;
}

double RecursiveNDimPCHIPInterpolator::interpolate(
    const GridData& grid,
    const std::vector<double>& location,
    const std::vector<DimensionMethod>& methods) const {
    
    // 临时设置方法
    auto savedMethods = dimensionMethods_;
    const_cast<std::vector<DimensionMethod>&>(dimensionMethods_) = methods;
    
    // 创建访问器
    LayoutAwareAccessor accessor(grid);
    
    // 执行插值
    std::vector<size_t> indices(location.size(), 0);
    auto result = interpRecursive(grid, location.size() - 1, indices, location, accessor);
    
    // 恢复方法
    const_cast<std::vector<DimensionMethod>&>(dimensionMethods_) = savedMethods;
    
    return result.value_or(std::numeric_limits<double>::quiet_NaN());
}

std::optional<double> RecursiveNDimPCHIPInterpolator::interpRecursive(
    const GridData& grid,
    int dim,
    std::vector<size_t>& indices,
    const std::vector<double>& coords,
    const LayoutAwareAccessor& accessor) const {
    
    if (dim < 0) {
        // 基础情况：所有维度都已处理，返回数据值
        return accessor.getValue<double>(indices[0], indices[1], 
                                       indices.size() > 2 ? indices[2] : 0, 0);
    }
    
    // 获取当前维度的方法
    DimensionMethod method = (dim < dimensionMethods_.size()) ? 
                            dimensionMethods_[dim] : DimensionMethod::PCHIP;
    
    // 在当前维度进行插值
    return interp1D(grid, dim, indices, coords[dim], method, accessor);
}

double RecursiveNDimPCHIPInterpolator::interp1D(
    const GridData& grid,
    int dim,
    std::vector<size_t>& indices,
    double coord,
    DimensionMethod method,
    const LayoutAwareAccessor& accessor) const {
    
    // 获取维度坐标
    auto dimCoords = getDimensionCoordinates(grid, dim);
    if (dimCoords.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // 查找插值区间
    auto [idx0, idx1] = findInterval(dimCoords, coord);
    
    switch (method) {
        case DimensionMethod::NEAREST: {
            // 最近邻插值
            indices[dim] = (coord - dimCoords[idx0] < dimCoords[idx1] - coord) ? idx0 : idx1;
            auto result = interpRecursive(grid, dim - 1, indices, 
                                        std::vector<double>(indices.size(), 0), accessor);
            return result.value_or(std::numeric_limits<double>::quiet_NaN());
        }
        
        case DimensionMethod::LINEAR: {
            // 线性插值
            if (idx0 == idx1) {
                indices[dim] = idx0;
                auto result = interpRecursive(grid, dim - 1, indices,
                                            std::vector<double>(indices.size(), 0), accessor);
                return result.value_or(std::numeric_limits<double>::quiet_NaN());
            }
            
            // 在两个点进行递归插值
            indices[dim] = idx0;
            auto v0 = interpRecursive(grid, dim - 1, indices,
                                    std::vector<double>(indices.size(), 0), accessor);
            
            indices[dim] = idx1;
            auto v1 = interpRecursive(grid, dim - 1, indices,
                                    std::vector<double>(indices.size(), 0), accessor);
            
            if (!v0.has_value() || !v1.has_value()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            
            // 线性插值
            double t = (coord - dimCoords[idx0]) / (dimCoords[idx1] - dimCoords[idx0]);
            return v0.value() * (1 - t) + v1.value() * t;
        }
        
        case DimensionMethod::PCHIP: {
            // PCHIP插值需要4个点
            size_t n = dimCoords.size();
            if (n < 4) {
                // 退化到线性插值
                return interp1D(grid, dim, indices, coord, DimensionMethod::LINEAR, accessor);
            }
            
            // 确定4个点的索引
            size_t i0 = (idx0 > 0) ? idx0 - 1 : 0;
            size_t i1 = idx0;
            size_t i2 = idx1;
            size_t i3 = (idx1 < n - 1) ? idx1 + 1 : n - 1;
            
            // 获取4个点的值
            std::vector<double> values(4);
            for (size_t i = 0; i < 4; ++i) {
                indices[dim] = (i == 0) ? i0 : (i == 1) ? i1 : (i == 2) ? i2 : i3;
                auto v = interpRecursive(grid, dim - 1, indices,
                                       std::vector<double>(indices.size(), 0), accessor);
                if (!v.has_value()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                values[i] = v.value();
            }
            
            // 提取坐标
            std::vector<double> x = {dimCoords[i0], dimCoords[i1], 
                                    dimCoords[i2], dimCoords[i3]};
            
            // 执行PCHIP插值
            return pchip1D(x, values, coord);
        }
        
        default:
            return std::numeric_limits<double>::quiet_NaN();
    }
}

double RecursiveNDimPCHIPInterpolator::pchip1D(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double xi) const {
    
    size_t n = x.size();
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();
    
    // 查找包含xi的区间
    size_t k = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        if (xi >= x[i] && xi <= x[i + 1]) {
            k = i;
            break;
        }
    }
    
    if (k == n - 1) k = n - 2;
    
    // 计算导数
    auto derivatives = computePCHIPDerivatives(x, y);
    
    // Hermite插值
    double h = x[k + 1] - x[k];
    double t = (xi - x[k]) / h;
    double t2 = t * t;
    double t3 = t2 * t;
    
    double h00 = 2 * t3 - 3 * t2 + 1;
    double h10 = t3 - 2 * t2 + t;
    double h01 = -2 * t3 + 3 * t2;
    double h11 = t3 - t2;
    
    return h00 * y[k] + h10 * h * derivatives[k] + 
           h01 * y[k + 1] + h11 * h * derivatives[k + 1];
}

std::vector<double> RecursiveNDimPCHIPInterpolator::computePCHIPDerivatives(
    const std::vector<double>& x,
    const std::vector<double>& y) const {
    
    size_t n = x.size();
    std::vector<double> d(n);
    
    if (n < 2) return d;
    
    // 计算差分
    std::vector<double> h(n - 1), delta(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }
    
    // 计算导数
    d[0] = delta[0];
    d[n - 1] = delta[n - 2];
    
    for (size_t i = 1; i < n - 1; ++i) {
        if (delta[i - 1] * delta[i] <= 0) {
            d[i] = 0;
        } else {
            double w1 = 2 * h[i] + h[i - 1];
            double w2 = h[i] + 2 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }
    
    return d;
}

std::vector<double> RecursiveNDimPCHIPInterpolator::getDimensionCoordinates(
    const GridData& grid,
    int dim) const {
    
    const auto& def = grid.getDefinition();
    switch (dim) {
        case 0:
            return def.xDimension.coordinates;
        case 1:
            return def.yDimension.coordinates;
        case 2:
            return def.zDimension.coordinates;
        default:
            return {};
    }
}

std::pair<size_t, size_t> RecursiveNDimPCHIPInterpolator::findInterval(
    const std::vector<double>& coords,
    double target) const {
    
    if (coords.empty()) return {0, 0};
    
    // 二分查找
    auto it = std::lower_bound(coords.begin(), coords.end(), target);
    
    if (it == coords.begin()) {
        return {0, 0};
    } else if (it == coords.end()) {
        return {coords.size() - 1, coords.size() - 1};
    } else {
        size_t idx1 = std::distance(coords.begin(), it);
        return {idx1 - 1, idx1};
    }
}

} // namespace oscean::core_services::interpolation 