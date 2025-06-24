#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include <string>
#include <vector>
#include <optional>
#include <memory>

// 定义PI常量（Windows上M_PI可能未定义）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace oscean::core_services::spatial_ops::utils {

/**
 * @brief 几何格式转换工具类
 * 提供WKT、WKB、GeoJSON等格式之间的转换功能
 */
class GeometryConverter {
public:
    // === WKT转换功能 ===
    
    /**
     * @brief 将Point转换为WKT格式
     * @param point 点对象
     * @return WKT字符串
     */
    static std::string pointToWKT(const oscean::core_services::Point& point);
    
    /**
     * @brief 将BoundingBox转换为WKT POLYGON格式
     * @param bbox 边界框对象
     * @return WKT POLYGON字符串
     */
    static std::string boundingBoxToWKT(const oscean::core_services::BoundingBox& bbox);
    
    /**
     * @brief 从WKT字符串解析Point
     * @param wkt WKT字符串
     * @return 解析成功返回Point，失败返回nullopt
     */
    static std::optional<oscean::core_services::Point> parsePointFromWKT(const std::string& wkt);
    
    /**
     * @brief 从WKT字符串解析BoundingBox
     * @param wkt WKT POLYGON字符串
     * @return 解析成功返回BoundingBox，失败返回nullopt
     */
    static std::optional<oscean::core_services::BoundingBox> parseBoundingBoxFromWKT(const std::string& wkt);
    
    /**
     * @brief 从WKT字符串提取边界框
     * @param wkt 任意几何体的WKT字符串
     * @return 几何体的边界框
     */
    static oscean::core_services::BoundingBox extractBoundingBoxFromWKT(const std::string& wkt);
    
    /**
     * @brief 解析完整的WKT几何体
     * @param wkt WKT字符串
     * @return 解析后的几何体信息
     */
    struct ParsedWKTGeometry {
        enum Type { POINT, LINESTRING, POLYGON, MULTIPOINT, MULTILINESTRING, MULTIPOLYGON, UNKNOWN };
        Type type = UNKNOWN;
        std::vector<oscean::core_services::Point> points;
        std::vector<std::vector<oscean::core_services::Point>> rings;
        oscean::core_services::BoundingBox boundingBox;
        bool isEmpty = false;
    };
    
    static ParsedWKTGeometry parseWKTGeometry(const std::string& wkt);
    
    /**
     * @brief 计算点到WKT几何体的距离
     * @param point 查询点
     * @param wkt 目标几何体WKT
     * @return 距离值
     */
    static double calculateDistanceToWKTGeometry(const oscean::core_services::Point& point, const std::string& wkt);
    
    /**
     * @brief 判断点是否在WKT多边形内
     * @param point 查询点
     * @param polygonWkt 多边形WKT
     * @return 是否在多边形内
     */
    static bool isPointInWKTPolygon(const oscean::core_services::Point& point, const std::string& polygonWkt);
    
    /**
     * @brief 判断WKT几何体是否与边界框相交
     * @param wkt 几何体WKT
     * @param bbox 边界框
     * @return 是否相交
     */
    static bool wktIntersectsBoundingBox(const std::string& wkt, const oscean::core_services::BoundingBox& bbox);
    
    // === GeoJSON转换功能 ===
    
    /**
     * @brief 将Point转换为GeoJSON格式
     * @param point 点对象
     * @return GeoJSON字符串
     */
    static std::string pointToGeoJSON(const oscean::core_services::Point& point);
    
    /**
     * @brief 将BoundingBox转换为GeoJSON Polygon格式
     * @param bbox 边界框对象
     * @return GeoJSON字符串
     */
    static std::string boundingBoxToGeoJSON(const oscean::core_services::BoundingBox& bbox);
    
    /**
     * @brief 将FeatureCollection转换为GeoJSON格式
     * @param features 要素集合
     * @return GeoJSON字符串
     */
    static std::string featureCollectionToGeoJSON(const oscean::core_services::FeatureCollection& features);
    
    /**
     * @brief 从GeoJSON字符串解析Point
     * @param geoJson GeoJSON字符串
     * @return 解析成功返回Point，失败返回nullopt
     */
    static std::optional<oscean::core_services::Point> parsePointFromGeoJSON(const std::string& geoJson);
    
    /**
     * @brief 从GeoJSON字符串解析FeatureCollection
     * @param geoJson GeoJSON字符串
     * @return 解析成功返回FeatureCollection，失败返回nullopt
     */
    static std::optional<oscean::core_services::FeatureCollection> parseFeatureCollectionFromGeoJSON(const std::string& geoJson);
    
    // === 几何验证功能 ===
    
    /**
     * @brief 验证WKT格式是否正确
     * @param wkt WKT字符串
     * @return 格式正确返回true
     */
    static bool isValidWKT(const std::string& wkt);
    
    /**
     * @brief 验证GeoJSON格式是否正确
     * @param geoJson GeoJSON字符串
     * @return 格式正确返回true
     */
    static bool isValidGeoJSON(const std::string& geoJson);
    
    /**
     * @brief 标准化WKT字符串（去除多余空格、统一大小写等）
     * @param wkt 原始WKT字符串
     * @return 标准化后的WKT字符串
     */
    static std::string normalizeWKT(const std::string& wkt);
    
    // === 几何计算辅助功能 ===
    
    /**
     * @brief 计算两点间的方位角（以北为0度，顺时针）
     * @param from 起始点
     * @param to 目标点
     * @return 方位角（度）
     */
    static double calculateBearing(const oscean::core_services::Point& from, 
                                  const oscean::core_services::Point& to);
    
    /**
     * @brief 根据起点、方位角和距离计算目标点
     * @param start 起始点
     * @param bearing 方位角（度）
     * @param distance 距离（米）
     * @return 目标点坐标
     */
    static oscean::core_services::Point calculateDestination(
        const oscean::core_services::Point& start,
        double bearing,
        double distance);
    
    /**
     * @brief 计算多边形的重心
     * @param polygon 多边形顶点数组
     * @return 重心坐标
     */
    static oscean::core_services::Point calculateCentroid(
        const std::vector<oscean::core_services::Point>& polygon);
    
    /**
     * @brief 计算多边形面积（使用Shoelace公式）
     * @param polygon 多边形顶点数组
     * @return 面积值
     */
    static double calculatePolygonArea(
        const std::vector<oscean::core_services::Point>& polygon);
    
    // === 常量定义 ===
    static constexpr double EARTH_RADIUS_METERS = 6378137.0;  // WGS84椭球体长半轴
    static constexpr double DEG_TO_RAD = M_PI / 180.0;        // 度到弧度转换
    static constexpr double RAD_TO_DEG = 180.0 / M_PI;        // 弧度到度转换
    static constexpr double EPSILON = 1e-10;                  // 浮点数比较精度

private:
    // === 内部辅助方法 ===
    
    /**
     * @brief 解析WKT中的坐标对
     * @param coordStr 坐标字符串
     * @return 坐标对数组
     */
    static std::vector<std::pair<double, double>> parseCoordinatePairs(const std::string& coordStr);
    
    /**
     * @brief 清理和标准化字符串
     * @param str 输入字符串
     * @return 清理后的字符串
     */
    static std::string cleanString(const std::string& str);
    
    /**
     * @brief 判断字符串是否为数字
     * @param str 待检查的字符串
     * @return 是否为数字
     */
    static bool isNumeric(const std::string& str);
    
    // === WKT解析辅助方法 ===
    
    /**
     * @brief 解析POINT坐标
     */
    static std::optional<oscean::core_services::Point> parsePointCoordinates(const std::string& wkt);
    
    /**
     * @brief 解析LINESTRING坐标
     */
    static std::vector<oscean::core_services::Point> parseLineStringCoordinates(const std::string& wkt);
    
    /**
     * @brief 解析POLYGON环
     */
    static std::vector<std::vector<oscean::core_services::Point>> parsePolygonRings(const std::string& wkt);
    
    /**
     * @brief 解析坐标序列
     */
    static std::vector<oscean::core_services::Point> parseCoordinateSequence(const std::string& coordsStr);
    
    /**
     * @brief 从点集合计算边界框
     */
    static oscean::core_services::BoundingBox calculateBoundingBoxFromPoints(const std::vector<oscean::core_services::Point>& points);
    
    /**
     * @brief 计算点到线串的距离
     */
    static double calculateDistanceToLineString(const oscean::core_services::Point& point, const std::vector<oscean::core_services::Point>& linePoints);
    
    /**
     * @brief 计算点到多边形的距离
     */
    static double calculateDistanceToPolygon(const oscean::core_services::Point& point, const std::vector<std::vector<oscean::core_services::Point>>& rings);
    
    /**
     * @brief 判断点是否在多边形环内
     */
    static bool isPointInPolygonRing(const oscean::core_services::Point& point, const std::vector<oscean::core_services::Point>& ring);
    
    /**
     * @brief 计算点到线段的距离
     */
    static double calculateDistanceToSegment(const oscean::core_services::Point& point, const oscean::core_services::Point& segStart, const oscean::core_services::Point& segEnd);
    
    /**
     * @brief 计算两点间距离
     */
    static double calculateDistance(const oscean::core_services::Point& p1, const oscean::core_services::Point& p2);
    
    /**
     * @brief 解析MULTIPOINT坐标
     */
    static std::vector<oscean::core_services::Point> parseMultiPointCoordinates(const std::string& wkt);
    
    /**
     * @brief 解析MULTILINESTRING坐标
     */
    static std::vector<oscean::core_services::Point> parseMultiLineStringCoordinates(const std::string& wkt);
    
    /**
     * @brief 解析MULTIPOLYGON环
     */
    static std::vector<std::vector<oscean::core_services::Point>> parseMultiPolygonRings(const std::string& wkt);
    
    /**
     * @brief 计算点到多点的距离
     */
    static double calculateDistanceToMultiPoint(const oscean::core_services::Point& point, const std::vector<oscean::core_services::Point>& multiPoints);
};

} // namespace oscean::core_services::spatial_ops::utils 