// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 空间工具不使用boost::asio

#include "spatial_utils.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <regex>
#include <boost/thread/future.hpp>

namespace oscean::core_services::spatial_ops::utils {

// =============================================================================
// 几何验证工具 - 实现头文件中声明的方法
// =============================================================================

bool SpatialUtils::isValidWKT(const std::string& wkt) {
    if (wkt.empty()) {
        return false;
    }
    
    // 简单的WKT格式验证
    std::regex wktPattern(R"(^(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*\(.+\)$)", 
                         std::regex_constants::icase);
    return std::regex_match(wkt, wktPattern);
}

bool SpatialUtils::isValidBoundingBox(const oscean::core_services::BoundingBox& bbox) {
    return bbox.isValid();
}

bool SpatialUtils::isValidPoint(const oscean::core_services::Point& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && 
           (!point.z.has_value() || std::isfinite(point.z.value()));
}

bool SpatialUtils::isValidGeometry(const oscean::core_services::Geometry& geometry) {
    return !geometry.wkt.empty() && isValidWKT(geometry.wkt);
}

// =============================================================================
// 几何计算工具 - 实现头文件中声明的方法
// =============================================================================

double SpatialUtils::calculateHaversineDistance(double lat1, double lon1, double lat2, double lon2) {
    // 转换为弧度
    double lat1Rad = lat1 * DEG_TO_RAD;
    double lon1Rad = lon1 * DEG_TO_RAD;
    double lat2Rad = lat2 * DEG_TO_RAD;
    double lon2Rad = lon2 * DEG_TO_RAD;
    
    // Haversine公式
    double dLat = lat2Rad - lat1Rad;
    double dLon = lon2Rad - lon1Rad;
    
    double a = std::sin(dLat / 2) * std::sin(dLat / 2) +
               std::cos(lat1Rad) * std::cos(lat2Rad) *
               std::sin(dLon / 2) * std::sin(dLon / 2);
    
    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
    
    return EARTH_RADIUS_METERS * c;
}

double SpatialUtils::pointToLineDistance(const oscean::core_services::Point& point,
                                        const oscean::core_services::Point& lineStart,
                                        const oscean::core_services::Point& lineEnd) {
    // 计算点到线段的距离
    double A = point.x - lineStart.x;
    double B = point.y - lineStart.y;
    double C = lineEnd.x - lineStart.x;
    double D = lineEnd.y - lineStart.y;
    
    double dot = A * C + B * D;
    double lenSq = C * C + D * D;
    
    if (lenSq == 0.0) {
        // 线段退化为点
        return std::sqrt(A * A + B * B);
    }
    
    double param = dot / lenSq;
    
    double xx, yy;
    if (param < 0) {
        xx = lineStart.x;
        yy = lineStart.y;
    } else if (param > 1) {
        xx = lineEnd.x;
        yy = lineEnd.y;
    } else {
        xx = lineStart.x + param * C;
        yy = lineStart.y + param * D;
    }
    
    double dx = point.x - xx;
    double dy = point.y - yy;
    return std::sqrt(dx * dx + dy * dy);
}

double SpatialUtils::calculatePolygonArea(const std::vector<oscean::core_services::Point>& points) {
    if (points.size() < 3) {
        return 0.0;
    }
    
    double area = 0.0;
    size_t n = points.size();
    
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[j].x * points[i].y;
    }
    
    return std::abs(area) / 2.0;
}

oscean::core_services::Point SpatialUtils::calculateBoundingBoxCenter(
    const oscean::core_services::BoundingBox& bbox) {
    
    double centerX = (bbox.minX + bbox.maxX) / 2.0;
    double centerY = (bbox.minY + bbox.maxY) / 2.0;
    
    oscean::core_services::Point center(centerX, centerY);
    
    if (bbox.minZ.has_value() && bbox.maxZ.has_value()) {
        center.z = (bbox.minZ.value() + bbox.maxZ.value()) / 2.0;
    }
    
    return center;
}

oscean::core_services::BoundingBox SpatialUtils::expandBoundingBox(
    const oscean::core_services::BoundingBox& bbox, double margin) {
    
    oscean::core_services::BoundingBox expanded = bbox;
    expanded.minX -= margin;
    expanded.minY -= margin;
    expanded.maxX += margin;
    expanded.maxY += margin;
    
    if (expanded.minZ.has_value()) {
        expanded.minZ = expanded.minZ.value() - margin;
    }
    if (expanded.maxZ.has_value()) {
        expanded.maxZ = expanded.maxZ.value() + margin;
    }
    
    return expanded;
}

bool SpatialUtils::boundingBoxesIntersect(const oscean::core_services::BoundingBox& bbox1,
                                          const oscean::core_services::BoundingBox& bbox2) {
    return !(bbox1.maxX < bbox2.minX || bbox2.maxX < bbox1.minX ||
             bbox1.maxY < bbox2.minY || bbox2.maxY < bbox1.minY);
}

std::optional<oscean::core_services::BoundingBox> SpatialUtils::intersectBoundingBoxes(
    const oscean::core_services::BoundingBox& bbox1,
    const oscean::core_services::BoundingBox& bbox2) {
    
    if (!boundingBoxesIntersect(bbox1, bbox2)) {
        return std::nullopt;
    }
    
    oscean::core_services::BoundingBox intersection;
    intersection.minX = std::max(bbox1.minX, bbox2.minX);
    intersection.minY = std::max(bbox1.minY, bbox2.minY);
    intersection.maxX = std::min(bbox1.maxX, bbox2.maxX);
    intersection.maxY = std::min(bbox1.maxY, bbox2.maxY);
    
    if (bbox1.minZ.has_value() && bbox2.minZ.has_value()) {
        intersection.minZ = std::max(bbox1.minZ.value(), bbox2.minZ.value());
    }
    if (bbox1.maxZ.has_value() && bbox2.maxZ.has_value()) {
        intersection.maxZ = std::min(bbox1.maxZ.value(), bbox2.maxZ.value());
    }
    
    return intersection;
}

// =============================================================================
// 栅格工具函数 - 实现头文件中声明的方法
// =============================================================================

std::pair<double, double> SpatialUtils::calculateRasterResolution(
    const std::vector<double>& geoTransform) {
    
    if (geoTransform.size() < 6) {
        return {0.0, 0.0};
    }
    
    double xRes = std::abs(geoTransform[1]);
    double yRes = std::abs(geoTransform[5]);
    
    return {xRes, yRes};
}

bool SpatialUtils::isValidGeoTransform(const std::vector<double>& geoTransform) {
    if (geoTransform.size() != 6) {
        return false;
    }
    
    // 检查像素大小不为零
    return geoTransform[1] != 0.0 && geoTransform[5] != 0.0;
}

oscean::core_services::BoundingBox SpatialUtils::calculateRasterBounds(
    const std::vector<double>& geoTransform, int width, int height) {
    
    if (!isValidGeoTransform(geoTransform)) {
        return oscean::core_services::BoundingBox{};
    }
    
    double minX = geoTransform[0];
    double maxY = geoTransform[3];
    double maxX = minX + width * geoTransform[1];
    double minY = maxY + height * geoTransform[5];
    
    oscean::core_services::BoundingBox bounds;
    bounds.minX = std::min(minX, maxX);
    bounds.maxX = std::max(minX, maxX);
    bounds.minY = std::min(minY, maxY);
    bounds.maxY = std::max(minY, maxY);
    
    return bounds;
}

// =============================================================================
// 数学工具函数 - 实现头文件中声明的方法
// =============================================================================

bool SpatialUtils::doubleEqual(double a, double b, double epsilon) {
    return std::abs(a - b) < epsilon;
}

double SpatialUtils::normalizeDegrees(double degrees) {
    while (degrees < 0.0) {
        degrees += 360.0;
    }
    while (degrees >= 360.0) {
        degrees -= 360.0;
    }
    return degrees;
}

double SpatialUtils::normalizeRadians(double radians) {
    while (radians < 0.0) {
        radians += 2.0 * M_PI;
    }
    while (radians >= 2.0 * M_PI) {
        radians -= 2.0 * M_PI;
    }
    return radians;
}

double SpatialUtils::calculateAngleDifference(double angle1, double angle2) {
    double diff = std::abs(angle1 - angle2);
    return std::min(diff, 360.0 - diff);
}

double SpatialUtils::bilinearInterpolation(double x, double y,
                                          double x1, double y1, double x2, double y2,
                                          double q11, double q12, double q21, double q22) {
    double x2x1 = x2 - x1;
    double y2y1 = y2 - y1;
    double x2x = x2 - x;
    double y2y = y2 - y;
    double yy1 = y - y1;
    double xx1 = x - x1;
    
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}

std::optional<double> SpatialUtils::nearestNeighborInterpolation(
    double x, double y,
    const std::vector<double>& gridX,
    const std::vector<double>& gridY,
    const std::vector<std::vector<double>>& values) {
    
    if (gridX.empty() || gridY.empty() || values.empty()) {
        return std::nullopt;
    }
    
    // 找到最近的网格点
    auto xIt = std::lower_bound(gridX.begin(), gridX.end(), x);
    auto yIt = std::lower_bound(gridY.begin(), gridY.end(), y);
    
    size_t xIdx = std::distance(gridX.begin(), xIt);
    size_t yIdx = std::distance(gridY.begin(), yIt);
    
    // 调整索引到有效范围
    if (xIdx >= gridX.size()) xIdx = gridX.size() - 1;
    if (yIdx >= gridY.size()) yIdx = gridY.size() - 1;
    
    if (xIdx < values.size() && yIdx < values[xIdx].size()) {
        return values[xIdx][yIdx];
    }
    
    return std::nullopt;
}

// =============================================================================
// CRS相关异步方法 - 实现头文件中声明的方法
// =============================================================================

boost::future<bool> SpatialUtils::canTransformAsync(
    std::shared_ptr<oscean::core_services::ICrsService> crsService,
    const oscean::core_services::CRSInfo& sourceCRS,
    const oscean::core_services::CRSInfo& targetCRS) {
    
    return boost::async(boost::launch::async, [crsService, sourceCRS, targetCRS]() {
        if (!crsService) {
            return false;
        }
        
        // 简化实现：假设总是可以转换
        return true;
    });
}

boost::future<std::vector<bool>> SpatialUtils::validatePointsAsync(
    std::shared_ptr<oscean::core_services::ICrsService> crsService,
    const std::vector<oscean::core_services::Point>& points,
    const oscean::core_services::CRSInfo& sourceCRS) {
    
    return boost::async(boost::launch::async, [points, sourceCRS](auto) {
        std::vector<bool> results;
        results.reserve(points.size());
        
        for (const auto& point : points) {
            results.push_back(isValidPoint(point));
        }
        
        return results;
    }, crsService);
}

// =============================================================================
// 辅助函数 - 实现头文件中声明的方法
// =============================================================================

bool SpatialUtils::isFiniteNumber(double value) {
    return std::isfinite(value);
}

bool SpatialUtils::isValidLatitude(double lat) {
    return lat >= -90.0 && lat <= 90.0;
}

bool SpatialUtils::isValidLongitude(double lon) {
    return lon >= -180.0 && lon <= 180.0;
}

} // namespace oscean::core_services::spatial_ops::utils 