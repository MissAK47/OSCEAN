#pragma once

/**
 * @file spatial_utils.h
 * @brief 空间工具函数集合 - CRS服务集成重构版本
 * 
 * 🎯 职责重新定义：
 * ✅ 专注于空间计算、验证和几何操作工具
 * ✅ 数学计算和几何算法实用函数
 * ✅ 空间数据验证和质量检查
 * ❌ 不再包含坐标转换功能（统一使用CRS服务）
 * ❌ 不再实现像素-地理坐标转换（使用CRS服务）
 */

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "core_services/crs/i_crs_service.h"
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <cmath>

// 定义PI常量（Windows上M_PI可能未定义）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace oscean::core_services::spatial_ops::utils {

/**
 * @brief 空间工具函数集合 - 重构版本
 * 专注于空间计算、验证和几何操作，坐标转换功能移至CRS服务
 */
class SpatialUtils {
public:
    // === 常量定义 ===
    static constexpr double EARTH_RADIUS_METERS = 6378137.0;  // WGS84椭球体长半轴
    static constexpr double DEG_TO_RAD = M_PI / 180.0;        // 度到弧度转换
    static constexpr double RAD_TO_DEG = 180.0 / M_PI;        // 弧度到度转换
    static constexpr double EPSILON = 1e-10;                  // 浮点数比较精度
    
    // =============================================================================
    // 🚫 移除的坐标转换功能 - 现在使用CRS服务
    // =============================================================================
    
    // ❌ geoToPixel() → 使用CRS服务的transformPointAsync()
    // ❌ pixelToGeo() → 使用CRS服务的transformPointAsync()
    
    // =============================================================================
    // 几何验证工具 - 保留并增强
    // =============================================================================
    
    /**
     * @brief 验证WKT字符串格式是否有效
     * @param wkt WKT字符串
     * @return 如果格式有效返回true
     */
    static bool isValidWKT(const std::string& wkt);
    
    /**
     * @brief 验证边界框是否有效
     * @param bbox 边界框
     * @return 如果有效返回true
     */
    static bool isValidBoundingBox(const oscean::core_services::BoundingBox& bbox);
    
    /**
     * @brief 验证点坐标是否有效
     * @param point 点坐标
     * @return 如果有效返回true
     */
    static bool isValidPoint(const oscean::core_services::Point& point);
    
    /**
     * @brief 验证几何对象是否有效
     * @param geometry 几何对象
     * @return 如果有效返回true
     */
    static bool isValidGeometry(const oscean::core_services::Geometry& geometry);
    
    // =============================================================================
    // 几何计算工具 - 保留
    // =============================================================================
    
    /**
     * @brief 计算两点间的大圆距离（Haversine公式）
     * @param lat1 点1纬度（度）
     * @param lon1 点1经度（度）
     * @param lat2 点2纬度（度）
     * @param lon2 点2经度（度）
     * @return 距离（米）
     */
    static double calculateHaversineDistance(double lat1, double lon1, double lat2, double lon2);
    
    /**
     * @brief 计算点到线段的距离
     * @param point 查询点
     * @param lineStart 线段起点
     * @param lineEnd 线段终点
     * @return 距离
     */
    static double pointToLineDistance(const oscean::core_services::Point& point,
                                    const oscean::core_services::Point& lineStart,
                                    const oscean::core_services::Point& lineEnd);
    
    /**
     * @brief 计算多边形面积（适用于地理坐标）
     * @param points 多边形顶点列表
     * @return 面积（平方米）
     */
    static double calculatePolygonArea(const std::vector<oscean::core_services::Point>& points);
    
    /**
     * @brief 计算边界框的中心点
     * @param bbox 边界框
     * @return 中心点
     */
    static oscean::core_services::Point calculateBoundingBoxCenter(
        const oscean::core_services::BoundingBox& bbox);
    
    /**
     * @brief 扩展边界框
     * @param bbox 原始边界框
     * @param margin 扩展边距
     * @return 扩展后的边界框
     */
    static oscean::core_services::BoundingBox expandBoundingBox(
        const oscean::core_services::BoundingBox& bbox, double margin);
    
    /**
     * @brief 检查两个边界框是否相交
     * @param bbox1 边界框1
     * @param bbox2 边界框2
     * @return 如果相交返回true
     */
    static bool boundingBoxesIntersect(const oscean::core_services::BoundingBox& bbox1,
                                      const oscean::core_services::BoundingBox& bbox2);
    
    /**
     * @brief 计算两个边界框的交集
     * @param bbox1 边界框1
     * @param bbox2 边界框2
     * @return 交集边界框，如果不相交则返回nullopt
     */
    static std::optional<oscean::core_services::BoundingBox> intersectBoundingBoxes(
        const oscean::core_services::BoundingBox& bbox1,
        const oscean::core_services::BoundingBox& bbox2);
    
    // =============================================================================
    // 栅格工具函数 - 保留
    // =============================================================================
    
    /**
     * @brief 计算栅格分辨率
     * @param geoTransform GDAL地理变换参数
     * @return 分辨率对(x分辨率, y分辨率)
     */
    static std::pair<double, double> calculateRasterResolution(
        const std::vector<double>& geoTransform);
    
    /**
     * @brief 验证地理变换参数
     * @param geoTransform 地理变换参数
     * @return 如果有效返回true
     */
    static bool isValidGeoTransform(const std::vector<double>& geoTransform);
    
    /**
     * @brief 计算栅格边界框
     * @param geoTransform 地理变换参数
     * @param width 栅格宽度
     * @param height 栅格高度
     * @return 边界框
     */
    static oscean::core_services::BoundingBox calculateRasterBounds(
        const std::vector<double>& geoTransform, int width, int height);
    
    // =============================================================================
    // 数学工具函数 - 保留
    // =============================================================================
    
    /**
     * @brief 安全的浮点数比较
     * @param a 数值a
     * @param b 数值b
     * @param epsilon 精度阈值
     * @return 如果相等返回true
     */
    static bool doubleEqual(double a, double b, double epsilon = EPSILON);
    
    /**
     * @brief 将角度规范化到[0, 360)范围
     * @param degrees 角度值
     * @return 规范化后的角度
     */
    static double normalizeDegrees(double degrees);
    
    /**
     * @brief 将弧度规范化到[0, 2π)范围
     * @param radians 弧度值
     * @return 规范化后的弧度
     */
    static double normalizeRadians(double radians);
    
    /**
     * @brief 计算两个角度间的最小夹角
     * @param angle1 角度1（度）
     * @param angle2 角度2（度）
     * @return 最小夹角（度）
     */
    static double calculateAngleDifference(double angle1, double angle2);
    
    // =============================================================================
    // 插值工具函数 - 保留
    // =============================================================================
    
    /**
     * @brief 双线性插值
     * @param x 查询点x坐标
     * @param y 查询点y坐标
     * @param x1 左下角x坐标
     * @param y1 左下角y坐标
     * @param x2 右上角x坐标
     * @param y2 右上角y坐标
     * @param q11 左下角值
     * @param q12 左上角值
     * @param q21 右下角值
     * @param q22 右上角值
     * @return 插值结果
     */
    static double bilinearInterpolation(double x, double y,
                                       double x1, double y1, double x2, double y2,
                                       double q11, double q12, double q21, double q22);
    
    /**
     * @brief 最近邻插值
     * @param x 查询点x坐标
     * @param y 查询点y坐标
     * @param gridX 网格x坐标数组
     * @param gridY 网格y坐标数组
     * @param values 网格值数组
     * @return 插值结果，如果查询点超出范围则返回nullopt
     */
    static std::optional<double> nearestNeighborInterpolation(
        double x, double y,
        const std::vector<double>& gridX,
        const std::vector<double>& gridY,
        const std::vector<std::vector<double>>& values);
    
    // =============================================================================
    // 🎯 CRS服务集成辅助函数 - 新增
    // =============================================================================
    
    /**
     * @brief 使用CRS服务验证坐标转换的可行性
     * @param crsService CRS服务接口
     * @param sourceCRS 源坐标系
     * @param targetCRS 目标坐标系
     * @return 如果可以转换返回true
     */
    static boost::future<bool> canTransformAsync(
        std::shared_ptr<oscean::core_services::ICrsService> crsService,
        const oscean::core_services::CRSInfo& sourceCRS,
        const oscean::core_services::CRSInfo& targetCRS);
    
    /**
     * @brief 使用CRS服务批量验证点坐标
     * @param crsService CRS服务接口
     * @param points 点坐标列表
     * @param sourceCRS 坐标系
     * @return 验证结果，对应每个点的有效性
     */
    static boost::future<std::vector<bool>> validatePointsAsync(
        std::shared_ptr<oscean::core_services::ICrsService> crsService,
        const std::vector<oscean::core_services::Point>& points,
        const oscean::core_services::CRSInfo& sourceCRS);

private:
    // 私有工具函数
    static double radians(double degrees) { return degrees * DEG_TO_RAD; }
    static double degrees(double radians) { return radians * RAD_TO_DEG; }
    
    // 验证辅助函数
    static bool isFiniteNumber(double value);
    static bool isValidLatitude(double lat);
    static bool isValidLongitude(double lon);
};

} // namespace oscean::core_services::spatial_ops::utils
