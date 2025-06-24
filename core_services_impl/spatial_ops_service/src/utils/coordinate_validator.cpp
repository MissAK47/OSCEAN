#include "utils/coordinate_validator.h"
#include <cmath>
#include <sstream>
#include <algorithm>
#include <regex>

namespace oscean::core_services::spatial_ops::utils {

// === 基础坐标验证实现 ===

bool CoordinateValidator::isValidPoint(const oscean::core_services::Point& point) {
    // 检查X和Y坐标是否为有限数值
    if (!isFiniteNumber(point.x) || !isFiniteNumber(point.y)) {
        return false;
    }
    
    // 检查坐标值是否在合理范围内
    if (std::abs(point.x) > MAX_COORDINATE_VALUE || std::abs(point.y) > MAX_COORDINATE_VALUE) {
        return false;
    }
    
    // 如果有Z坐标，也需要验证
    if (point.z.has_value()) {
        if (!isFiniteNumber(point.z.value()) || std::abs(point.z.value()) > MAX_COORDINATE_VALUE) {
            return false;
        }
    }
    
    return true;
}

bool CoordinateValidator::isValidBoundingBox(const oscean::core_services::BoundingBox& bbox) {
    // 检查所有坐标值是否有效
    if (!isFiniteNumber(bbox.minX) || !isFiniteNumber(bbox.minY) ||
        !isFiniteNumber(bbox.maxX) || !isFiniteNumber(bbox.maxY)) {
        return false;
    }
    
    // 检查最小值是否小于等于最大值
    if (bbox.minX > bbox.maxX || bbox.minY > bbox.maxY) {
        return false;
    }
    
    // 检查Z坐标（如果存在）
    if (bbox.minZ.has_value() && bbox.maxZ.has_value()) {
        if (!isFiniteNumber(bbox.minZ.value()) || !isFiniteNumber(bbox.maxZ.value())) {
            return false;
        }
        if (bbox.minZ.value() > bbox.maxZ.value()) {
            return false;
        }
    }
    
    // 检查坐标值是否在合理范围内
    if (std::abs(bbox.minX) > MAX_COORDINATE_VALUE || std::abs(bbox.minY) > MAX_COORDINATE_VALUE ||
        std::abs(bbox.maxX) > MAX_COORDINATE_VALUE || std::abs(bbox.maxY) > MAX_COORDINATE_VALUE) {
        return false;
    }
    
    return true;
}

bool CoordinateValidator::isValidGridDefinition(const oscean::core_services::GridDefinition& gridDef) {
    // 检查网格尺寸
    if (gridDef.rows == 0 || gridDef.cols == 0) {
        return false;
    }
    
    // 检查边界框
    if (!isValidBoundingBox(gridDef.extent)) {
        return false;
    }
    
    // 检查分辨率
    if (!isValidGridResolution(gridDef.xResolution, gridDef.yResolution)) {
        return false;
    }
    
    // 检查CRS
    // TODO: 使用CRS服务验证 - ICrsService::validateCRSAsync(gridDef.crs)
    // 临时跳过CRS验证，避免与CRS服务功能重复
    
    return true;
}

// === 地理坐标验证实现 ===

bool CoordinateValidator::isValidLongitude(double longitude) {
    return isFiniteNumber(longitude) && 
           isInRange(longitude, MIN_LONGITUDE, MAX_LONGITUDE);
}

bool CoordinateValidator::isValidLatitude(double latitude) {
    return isFiniteNumber(latitude) && 
           isInRange(latitude, MIN_LATITUDE, MAX_LATITUDE);
}

bool CoordinateValidator::isValidGeographicPoint(const oscean::core_services::Point& point) {
    return isValidLongitude(point.x) && isValidLatitude(point.y) &&
           (!point.z.has_value() || isFiniteNumber(point.z.value()));
}

bool CoordinateValidator::isValidGeographicBoundingBox(const oscean::core_services::BoundingBox& bbox) {
    return isValidLongitude(bbox.minX) && isValidLongitude(bbox.maxX) &&
           isValidLatitude(bbox.minY) && isValidLatitude(bbox.maxY) &&
           bbox.minX <= bbox.maxX && bbox.minY <= bbox.maxY;
}

// === 投影坐标验证实现 ===

bool CoordinateValidator::isValidProjectedPoint(const oscean::core_services::Point& point, 
                                               const oscean::core_services::CRSInfo& crs) {
    // 基础点验证
    if (!isValidPoint(point)) {
        return false;
    }
    
    // TODO: 使用CRS服务验证 - ICrsService::validateCRSAsync(crs)
    // 临时跳过CRS验证，避免与CRS服务功能重复
    
    // 如果是地理坐标系，使用地理坐标验证
    if (crs.isGeographic) {
        return isValidGeographicPoint(point);
    }
    
    // 对于投影坐标系，检查坐标值是否在合理范围内
    // 大多数投影坐标系的坐标值不应超过地球周长的数量级
    const double MAX_PROJECTED_COORDINATE = 50000000.0; // 50,000 km
    
    return std::abs(point.x) <= MAX_PROJECTED_COORDINATE && 
           std::abs(point.y) <= MAX_PROJECTED_COORDINATE;
}

bool CoordinateValidator::isValidProjectedBoundingBox(const oscean::core_services::BoundingBox& bbox,
                                                     const oscean::core_services::CRSInfo& crs) {
    // 基础边界框验证
    if (!isValidBoundingBox(bbox)) {
        return false;
    }
    
    // TODO: 使用CRS服务验证 - ICrsService::validateCRSAsync(crs)
    // 临时跳过CRS验证，避免与CRS服务功能重复
    
    // 如果是地理坐标系，使用地理坐标验证
    if (crs.isGeographic) {
        return isValidGeographicBoundingBox(bbox);
    }
    
    // 对于投影坐标系，检查边界框的四个角点
    oscean::core_services::Point corners[4] = {
        oscean::core_services::Point(bbox.minX, bbox.minY),
        oscean::core_services::Point(bbox.maxX, bbox.minY),
        oscean::core_services::Point(bbox.maxX, bbox.maxY),
        oscean::core_services::Point(bbox.minX, bbox.maxY)
    };
    
    for (const auto& corner : corners) {
        if (!isValidProjectedPoint(corner, crs)) {
            return false;
        }
    }
    
    return true;
}

// ❌ CRS验证功能已移除 - 避免与CRS服务功能重复
// 
// 坐标验证器专注于几何和坐标数值验证，CRS定义验证由CRS服务负责：
// - CRS有效性验证：使用 ICrsService::validateCRSAsync()
// - EPSG代码验证：使用 ICrsService::parseFromEpsgCodeAsync()
// - WKT定义验证：使用 ICrsService::parseFromWktAsync()
// - PROJ字符串验证：使用 ICrsService::parseFromProjStringAsync()

// === 几何验证实现 ===

bool CoordinateValidator::isValidPolygon(const std::vector<oscean::core_services::Point>& polygon) {
    // 检查顶点数量
    if (polygon.size() < MIN_POLYGON_VERTICES) {
        return false;
    }
    
    // 检查每个顶点是否有效
    for (const auto& point : polygon) {
        if (!isValidPoint(point)) {
            return false;
        }
    }
    
    // 检查多边形是否闭合
    if (!isPolygonClosed(polygon)) {
        return false;
    }
    
    // 检查是否有足够的面积（避免退化多边形）
    double area = std::abs(calculateSignedArea(polygon));
    if (area < EPSILON) {
        return false;
    }
    
    // 简化的自相交检查（完整实现需要更复杂的算法）
    if (polygon.size() > 4 && hasPolygonSelfIntersection(polygon)) {
        return false;
    }
    
    return true;
}

bool CoordinateValidator::isValidLineString(const std::vector<oscean::core_services::Point>& lineString) {
    // 检查顶点数量
    if (lineString.size() < MIN_LINESTRING_VERTICES) {
        return false;
    }
    
    // 检查每个顶点是否有效
    for (const auto& point : lineString) {
        if (!isValidPoint(point)) {
            return false;
        }
    }
    
    // 检查是否有重复的连续顶点
    for (size_t i = 1; i < lineString.size(); ++i) {
        if (isApproximatelyEqual(lineString[i-1].x, lineString[i].x) &&
            isApproximatelyEqual(lineString[i-1].y, lineString[i].y)) {
            return false;
        }
    }
    
    return true;
}

bool CoordinateValidator::isValidFeatureCollection(const oscean::core_services::FeatureCollection& features) {
    // 空的要素集合是有效的
    if (features.empty()) {
        return true;
    }
    
    // 检查每个要素
    for (const auto& feature : features.getFeatures()) {
        // 检查几何是否有效（简化检查）
        if (feature.geometryWkt.empty()) {
            return false;
        }
        
        // 可以添加更多的要素验证逻辑
    }
    
    return true;
}

// === 数值验证实现 ===

bool CoordinateValidator::isFiniteNumber(double value) {
    return std::isfinite(value);
}

bool CoordinateValidator::isInRange(double value, double minValue, double maxValue) {
    return isFiniteNumber(value) && value >= minValue && value <= maxValue;
}

bool CoordinateValidator::isApproximatelyEqual(double a, double b, double epsilon) {
    return std::abs(a - b) <= epsilon;
}

// === 拓扑验证实现 ===

bool CoordinateValidator::isPointInBoundingBox(const oscean::core_services::Point& point,
                                              const oscean::core_services::BoundingBox& bbox) {
    return point.x >= bbox.minX && point.x <= bbox.maxX &&
           point.y >= bbox.minY && point.y <= bbox.maxY;
}

bool CoordinateValidator::doBoundingBoxesIntersect(const oscean::core_services::BoundingBox& bbox1,
                                                  const oscean::core_services::BoundingBox& bbox2) {
    return !(bbox1.maxX < bbox2.minX || bbox2.maxX < bbox1.minX ||
             bbox1.maxY < bbox2.minY || bbox2.maxY < bbox1.minY);
}

bool CoordinateValidator::doesBoundingBoxContain(const oscean::core_services::BoundingBox& container,
                                                const oscean::core_services::BoundingBox& contained) {
    return container.minX <= contained.minX && container.maxX >= contained.maxX &&
           container.minY <= contained.minY && container.maxY >= contained.maxY;
}

// === 网格验证实现 ===

bool CoordinateValidator::isValidGridIndex(const oscean::core_services::GridIndex& gridIndex,
                                          const oscean::core_services::GridDefinition& gridDef) {
    // 检查X和Y索引是否在有效范围内
    if (gridIndex.x < 0 || gridIndex.y < 0) {
        return false;
    }
    
    if (static_cast<size_t>(gridIndex.x) >= gridDef.cols || 
        static_cast<size_t>(gridIndex.y) >= gridDef.rows) {
        return false;
    }
    
    // 检查Z索引（如果存在）
    if (gridIndex.z.has_value()) {
        if (gridIndex.z.value() < 0) {
            return false;
        }
        // 可以添加Z维度范围检查
    }
    
    // 检查时间索引（如果存在）
    if (gridIndex.t.has_value()) {
        if (gridIndex.t.value() < 0) {
            return false;
        }
        // 可以添加时间维度范围检查
    }
    
    return true;
}

bool CoordinateValidator::isValidGridResolution(double xResolution, double yResolution) {
    return isFiniteNumber(xResolution) && isFiniteNumber(yResolution) &&
           xResolution > MIN_RESOLUTION && yResolution > MIN_RESOLUTION;
}

// === 错误诊断实现 ===

std::string CoordinateValidator::validatePointDetailed(const oscean::core_services::Point& point) {
    std::ostringstream oss;
    
    if (!isFiniteNumber(point.x)) {
        oss << "X coordinate is not finite (NaN or infinite); ";
    }
    if (!isFiniteNumber(point.y)) {
        oss << "Y coordinate is not finite (NaN or infinite); ";
    }
    if (point.z.has_value() && !isFiniteNumber(point.z.value())) {
        oss << "Z coordinate is not finite (NaN or infinite); ";
    }
    
    if (std::abs(point.x) > MAX_COORDINATE_VALUE) {
        oss << "X coordinate exceeds maximum allowed value; ";
    }
    if (std::abs(point.y) > MAX_COORDINATE_VALUE) {
        oss << "Y coordinate exceeds maximum allowed value; ";
    }
    if (point.z.has_value() && std::abs(point.z.value()) > MAX_COORDINATE_VALUE) {
        oss << "Z coordinate exceeds maximum allowed value; ";
    }
    
    std::string result = oss.str();
    if (result.empty()) {
        return "Point is valid";
    } else {
        // 移除最后的分号和空格
        if (result.length() >= 2) {
            result = result.substr(0, result.length() - 2);
        }
        return result;
    }
}

std::string CoordinateValidator::validateBoundingBoxDetailed(const oscean::core_services::BoundingBox& bbox) {
    std::ostringstream oss;
    
    if (!isFiniteNumber(bbox.minX) || !isFiniteNumber(bbox.minY) ||
        !isFiniteNumber(bbox.maxX) || !isFiniteNumber(bbox.maxY)) {
        oss << "One or more coordinates are not finite; ";
    }
    
    if (bbox.minX > bbox.maxX) {
        oss << "minX (" << bbox.minX << ") > maxX (" << bbox.maxX << "); ";
    }
    if (bbox.minY > bbox.maxY) {
        oss << "minY (" << bbox.minY << ") > maxY (" << bbox.maxY << "); ";
    }
    
    if (bbox.minZ.has_value() && bbox.maxZ.has_value() && 
        bbox.minZ.value() > bbox.maxZ.value()) {
        oss << "minZ > maxZ; ";
    }
    
    std::string result = oss.str();
    if (result.empty()) {
        return "BoundingBox is valid";
    } else {
        if (result.length() >= 2) {
            result = result.substr(0, result.length() - 2);
        }
        return result;
    }
}

std::string CoordinateValidator::validateCRSDetailed(const oscean::core_services::CRSInfo& crs) {
    std::ostringstream oss;
    
    // 基础CRS信息验证
    if (crs.id.empty() && crs.authorityName.empty() && crs.wktext.empty()) {
        oss << "CRS has no valid identifier (id, authority, or WKT); ";
    }
    
    // 检查EPSG代码有效性
    if (crs.epsgCode.has_value() && crs.epsgCode.value() <= 0) {
        oss << "Invalid EPSG code (" << crs.epsgCode.value() << "); ";
    }
    
    // 检查WKT格式基础有效性
    if (!crs.wktext.empty()) {
        if (crs.wktext.find("GEOGCS") == std::string::npos && 
            crs.wktext.find("PROJCS") == std::string::npos &&
            crs.wktext.find("GEOCCS") == std::string::npos) {
            oss << "WKT does not contain valid CRS definition; ";
        }
    }
    
    // TODO: 使用CRS服务进行详细验证
    // - ICrsService::parseFromEpsgCodeAsync() 验证EPSG代码
    // - ICrsService::parseFromWktAsync() 验证WKT格式
    // - ICrsService::parseFromProjStringAsync() 验证PROJ字符串
    
    std::string result = oss.str();
    if (result.empty()) {
        return ""; // 有效时返回空字符串
    } else {
        if (result.length() >= 2) {
            result = result.substr(0, result.length() - 2);
        }
        return result;
    }
}

// === 内部辅助方法实现 ===

bool CoordinateValidator::hasPolygonSelfIntersection(const std::vector<oscean::core_services::Point>& polygon) {
    // 简化的自相交检查
    // 完整实现需要使用Bentley-Ottmann算法或类似的复杂算法
    
    size_t n = polygon.size();
    if (n < 4) return false;
    
    // 检查相邻边是否相交（除了在端点处）
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = i + 2; j < n - 1; ++j) {
            // 避免检查相邻边
            if (j == i + 1 || (i == 0 && j == n - 2)) continue;
            
            // 简化的线段相交检查
            // 这里只是一个占位符，实际需要实现完整的线段相交算法
        }
    }
    
    return false; // 简化实现，总是返回false
}

bool CoordinateValidator::isPolygonClosed(const std::vector<oscean::core_services::Point>& polygon) {
    if (polygon.size() < 3) return false;
    
    const auto& first = polygon.front();
    const auto& last = polygon.back();
    
    return isApproximatelyEqual(first.x, last.x) && 
           isApproximatelyEqual(first.y, last.y);
}

double CoordinateValidator::calculateSignedArea(const std::vector<oscean::core_services::Point>& polygon) {
    if (polygon.size() < 3) return 0.0;
    
    double area = 0.0;
    size_t n = polygon.size();
    
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += polygon[i].x * polygon[j].y;
        area -= polygon[j].x * polygon[i].y;
    }
    
    return area * 0.5;
}

} // namespace oscean::core_services::spatial_ops::utils 