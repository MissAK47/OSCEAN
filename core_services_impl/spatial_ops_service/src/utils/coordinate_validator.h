#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include <string>
#include <vector>
#include <optional>
#include <limits>

namespace oscean::core_services::spatial_ops::utils {

/**
 * @brief 坐标验证工具类
 * 提供各种坐标系统和几何对象的验证功能
 */
class CoordinateValidator {
public:
    // === 基础坐标验证 ===
    
    /**
     * @brief 验证点坐标是否有效
     * @param point 点对象
     * @return 有效返回true
     */
    static bool isValidPoint(const oscean::core_services::Point& point);
    
    /**
     * @brief 验证边界框是否有效
     * @param bbox 边界框对象
     * @return 有效返回true
     */
    static bool isValidBoundingBox(const oscean::core_services::BoundingBox& bbox);
    
    /**
     * @brief 验证网格定义是否有效
     * @param gridDef 网格定义对象
     * @return 有效返回true
     */
    static bool isValidGridDefinition(const oscean::core_services::GridDefinition& gridDef);
    
    // === 地理坐标验证 ===
    
    /**
     * @brief 验证经度值是否在有效范围内
     * @param longitude 经度值
     * @return 有效返回true（-180到180度）
     */
    static bool isValidLongitude(double longitude);
    
    /**
     * @brief 验证纬度值是否在有效范围内
     * @param latitude 纬度值
     * @return 有效返回true（-90到90度）
     */
    static bool isValidLatitude(double latitude);
    
    /**
     * @brief 验证地理坐标点是否有效
     * @param point 地理坐标点
     * @return 有效返回true
     */
    static bool isValidGeographicPoint(const oscean::core_services::Point& point);
    
    /**
     * @brief 验证地理边界框是否有效
     * @param bbox 地理边界框
     * @return 有效返回true
     */
    static bool isValidGeographicBoundingBox(const oscean::core_services::BoundingBox& bbox);
    
    // === 投影坐标验证 ===
    
    /**
     * @brief 验证投影坐标点是否有效
     * @param point 投影坐标点
     * @param crs 坐标参考系统信息
     * @return 有效返回true
     */
    static bool isValidProjectedPoint(const oscean::core_services::Point& point, 
                                     const oscean::core_services::CRSInfo& crs);
    
    /**
     * @brief 验证投影边界框是否有效
     * @param bbox 投影边界框
     * @param crs 坐标参考系统信息
     * @return 有效返回true
     */
    static bool isValidProjectedBoundingBox(const oscean::core_services::BoundingBox& bbox,
                                           const oscean::core_services::CRSInfo& crs);
    
    // ❌ CRS验证功能已移除 - 使用CRS服务进行验证
    // 使用 ICrsService::validateCRSAsync() 替代
    // 使用 ICrsService::parseFromEpsgCodeAsync() 验证EPSG代码
    // 使用 ICrsService::parseFromWktAsync() 验证WKT
    
    // === 几何验证 ===
    
    /**
     * @brief 验证多边形顶点数组是否有效
     * @param polygon 多边形顶点数组
     * @return 有效返回true
     */
    static bool isValidPolygon(const std::vector<oscean::core_services::Point>& polygon);
    
    /**
     * @brief 验证线串顶点数组是否有效
     * @param lineString 线串顶点数组
     * @return 有效返回true
     */
    static bool isValidLineString(const std::vector<oscean::core_services::Point>& lineString);
    
    /**
     * @brief 验证要素集合是否有效
     * @param features 要素集合
     * @return 有效返回true
     */
    static bool isValidFeatureCollection(const oscean::core_services::FeatureCollection& features);
    
    // === 数值验证 ===
    
    /**
     * @brief 检查浮点数是否为有限值（非NaN、非无穷大）
     * @param value 浮点数值
     * @return 有限值返回true
     */
    static bool isFiniteNumber(double value);
    
    /**
     * @brief 检查浮点数是否在指定范围内
     * @param value 浮点数值
     * @param minValue 最小值
     * @param maxValue 最大值
     * @return 在范围内返回true
     */
    static bool isInRange(double value, double minValue, double maxValue);
    
    /**
     * @brief 检查两个浮点数是否近似相等
     * @param a 第一个数值
     * @param b 第二个数值
     * @param epsilon 容差值
     * @return 近似相等返回true
     */
    static bool isApproximatelyEqual(double a, double b, double epsilon = EPSILON);
    
    // === 拓扑验证 ===
    
    /**
     * @brief 检查点是否在边界框内
     * @param point 点坐标
     * @param bbox 边界框
     * @return 在边界框内返回true
     */
    static bool isPointInBoundingBox(const oscean::core_services::Point& point,
                                    const oscean::core_services::BoundingBox& bbox);
    
    /**
     * @brief 检查两个边界框是否相交
     * @param bbox1 第一个边界框
     * @param bbox2 第二个边界框
     * @return 相交返回true
     */
    static bool doBoundingBoxesIntersect(const oscean::core_services::BoundingBox& bbox1,
                                        const oscean::core_services::BoundingBox& bbox2);
    
    /**
     * @brief 检查边界框是否包含另一个边界框
     * @param container 容器边界框
     * @param contained 被包含的边界框
     * @return 包含返回true
     */
    static bool doesBoundingBoxContain(const oscean::core_services::BoundingBox& container,
                                      const oscean::core_services::BoundingBox& contained);
    
    // === 网格验证 ===
    
    /**
     * @brief 验证网格索引是否有效
     * @param gridIndex 网格索引
     * @param gridDef 网格定义
     * @return 有效返回true
     */
    static bool isValidGridIndex(const oscean::core_services::GridIndex& gridIndex,
                                const oscean::core_services::GridDefinition& gridDef);
    
    /**
     * @brief 验证网格分辨率是否合理
     * @param xResolution X方向分辨率
     * @param yResolution Y方向分辨率
     * @return 合理返回true
     */
    static bool isValidGridResolution(double xResolution, double yResolution);
    
    // === 错误诊断 ===
    
    /**
     * @brief 获取点坐标的详细验证结果
     * @param point 点对象
     * @return 验证结果描述
     */
    static std::string validatePointDetailed(const oscean::core_services::Point& point);
    
    /**
     * @brief 获取边界框的详细验证结果
     * @param bbox 边界框对象
     * @return 验证结果描述
     */
    static std::string validateBoundingBoxDetailed(const oscean::core_services::BoundingBox& bbox);
    
    /**
     * @brief 获取CRS的详细验证结果
     * @param crs 坐标参考系统信息
     * @return 验证结果描述
     */
    static std::string validateCRSDetailed(const oscean::core_services::CRSInfo& crs);
    
    // === 常量定义 ===
    static constexpr double EPSILON = 1e-10;                    // 浮点数比较精度
    static constexpr double MIN_LONGITUDE = -180.0;             // 最小经度
    static constexpr double MAX_LONGITUDE = 180.0;              // 最大经度
    static constexpr double MIN_LATITUDE = -90.0;               // 最小纬度
    static constexpr double MAX_LATITUDE = 90.0;                // 最大纬度
    static constexpr double MIN_RESOLUTION = 1e-12;             // 最小分辨率
    static constexpr double MAX_COORDINATE_VALUE = 1e15;        // 最大坐标值
    static constexpr int MIN_EPSG_CODE = 1000;                  // 最小EPSG代码
    static constexpr int MAX_EPSG_CODE = 32767;                 // 最大EPSG代码
    static constexpr size_t MIN_POLYGON_VERTICES = 3;           // 多边形最少顶点数
    static constexpr size_t MIN_LINESTRING_VERTICES = 2;        // 线串最少顶点数

private:
    // === 内部辅助方法 ===
    
    /**
     * @brief 检查多边形是否自相交
     * @param polygon 多边形顶点数组
     * @return 自相交返回true
     */
    static bool hasPolygonSelfIntersection(const std::vector<oscean::core_services::Point>& polygon);
    
    /**
     * @brief 检查多边形是否闭合
     * @param polygon 多边形顶点数组
     * @return 闭合返回true
     */
    static bool isPolygonClosed(const std::vector<oscean::core_services::Point>& polygon);
    
    /**
     * @brief 计算多边形的有向面积
     * @param polygon 多边形顶点数组
     * @return 有向面积值
     */
    static double calculateSignedArea(const std::vector<oscean::core_services::Point>& polygon);
};

} // namespace oscean::core_services::spatial_ops::utils 