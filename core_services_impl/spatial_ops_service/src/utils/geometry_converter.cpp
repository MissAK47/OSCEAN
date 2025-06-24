#include "geometry_converter.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <cmath>
#include <limits>
#include <variant>

using namespace oscean::core_services;

namespace oscean::core_services::spatial_ops::utils {

// === WKT转换实现 ===
    
std::string GeometryConverter::pointToWKT(const Point& point) {
    auto formatCoordinate = [](double coord) -> std::string {
        // 如果是整数，不显示小数部分
        if (coord == std::floor(coord)) {
            return std::to_string(static_cast<long long>(coord));
        } else {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(10) << coord;
            std::string result = oss.str();
            // 移除尾随的零
            result.erase(result.find_last_not_of('0') + 1, std::string::npos);
            result.erase(result.find_last_not_of('.') + 1, std::string::npos);
            return result;
        }
    };
    
    if (point.z.has_value()) {
        return "POINT Z (" + formatCoordinate(point.x) + " " + 
               formatCoordinate(point.y) + " " + formatCoordinate(point.z.value()) + ")";
    } else {
        return "POINT(" + formatCoordinate(point.x) + " " + formatCoordinate(point.y) + ")";
    }
}

std::string GeometryConverter::boundingBoxToWKT(const BoundingBox& bbox) {
    auto formatCoordinate = [](double coord) -> std::string {
    std::ostringstream oss;
        oss << std::fixed << std::setprecision(10) << coord;
        return oss.str();
    };
    
    return "POLYGON ((" + 
           formatCoordinate(bbox.minX) + " " + formatCoordinate(bbox.minY) + ", " +
           formatCoordinate(bbox.maxX) + " " + formatCoordinate(bbox.minY) + ", " +
           formatCoordinate(bbox.maxX) + " " + formatCoordinate(bbox.maxY) + ", " +
           formatCoordinate(bbox.minX) + " " + formatCoordinate(bbox.maxY) + ", " +
           formatCoordinate(bbox.minX) + " " + formatCoordinate(bbox.minY) + "))";
}

std::optional<Point> GeometryConverter::parsePointFromWKT(const std::string& wkt) {
    try {
        ParsedWKTGeometry parsed = parseWKTGeometry(wkt);
        if (parsed.type == ParsedWKTGeometry::POINT && !parsed.points.empty()) {
            return parsed.points[0];
        }
        return std::nullopt;
    } catch (const std::exception&) {
        return std::nullopt;
    }
    }
    
std::optional<BoundingBox> GeometryConverter::parseBoundingBoxFromWKT(const std::string& wkt) {
    try {
        BoundingBox bbox = extractBoundingBoxFromWKT(wkt);
        return bbox;
    } catch (const std::exception&) {
    return std::nullopt;
}
}

BoundingBox GeometryConverter::extractBoundingBoxFromWKT(const std::string& wkt) {
    try {
        ParsedWKTGeometry parsed = parseWKTGeometry(wkt);
        return parsed.boundingBox;
    } catch (const std::exception&) {
        // 解析失败，返回默认边界框
        return BoundingBox(-180.0, -90.0, 180.0, 90.0);
    }
}

GeometryConverter::ParsedWKTGeometry GeometryConverter::parseWKTGeometry(const std::string& wkt) {
    ParsedWKTGeometry result;
    
    try {
        std::string cleanWkt = cleanString(wkt);
        std::transform(cleanWkt.begin(), cleanWkt.end(), cleanWkt.begin(), ::toupper);
        
        // 检测几何类型
        if (cleanWkt.find("POINT") == 0) {
            result.type = ParsedWKTGeometry::POINT;
            auto point = parsePointCoordinates(cleanWkt);
            if (point.has_value()) {
                result.points.push_back(point.value());
                result.boundingBox = BoundingBox(point->x, point->y, point->x, point->y);
            }
        } else if (cleanWkt.find("LINESTRING") == 0) {
            result.type = ParsedWKTGeometry::LINESTRING;
            result.points = parseLineStringCoordinates(cleanWkt);
            if (!result.points.empty()) {
                result.boundingBox = calculateBoundingBoxFromPoints(result.points);
            }
        } else if (cleanWkt.find("POLYGON") == 0) {
            result.type = ParsedWKTGeometry::POLYGON;
            result.rings = parsePolygonRings(cleanWkt);
            if (!result.rings.empty() && !result.rings[0].empty()) {
                result.boundingBox = calculateBoundingBoxFromPoints(result.rings[0]);
            }
        } else if (cleanWkt.find("MULTIPOINT") == 0) {
            result.type = ParsedWKTGeometry::MULTIPOINT;
            result.points = parseMultiPointCoordinates(cleanWkt);
            if (!result.points.empty()) {
                result.boundingBox = calculateBoundingBoxFromPoints(result.points);
            }
        } else if (cleanWkt.find("MULTILINESTRING") == 0) {
            result.type = ParsedWKTGeometry::MULTILINESTRING;
            result.points = parseMultiLineStringCoordinates(cleanWkt);
            if (!result.points.empty()) {
                result.boundingBox = calculateBoundingBoxFromPoints(result.points);
            }
        } else if (cleanWkt.find("MULTIPOLYGON") == 0) {
            result.type = ParsedWKTGeometry::MULTIPOLYGON;
            result.rings = parseMultiPolygonRings(cleanWkt);
            if (!result.rings.empty() && !result.rings[0].empty()) {
                result.boundingBox = calculateBoundingBoxFromPoints(result.rings[0]);
            }
        }
        
        result.isEmpty = result.points.empty() && result.rings.empty();
        
    } catch (const std::exception&) {
        result.type = ParsedWKTGeometry::UNKNOWN;
        result.isEmpty = true;
    }
    
    return result;
}

double GeometryConverter::calculateDistanceToWKTGeometry(const Point& point, const std::string& geometryWkt) {
    try {
        ParsedWKTGeometry parsed = parseWKTGeometry(geometryWkt);
        
        switch (parsed.type) {
            case ParsedWKTGeometry::POINT:
                if (!parsed.points.empty()) {
                    return calculateDistance(point, parsed.points[0]);
                }
                break;
                
            case ParsedWKTGeometry::LINESTRING:
                return calculateDistanceToLineString(point, parsed.points);
                
            case ParsedWKTGeometry::POLYGON:
                if (!parsed.rings.empty()) {
                    return calculateDistanceToPolygon(point, parsed.rings);
                }
                break;
                
            default:
                break;
        }
        
        return std::numeric_limits<double>::max();
        
    } catch (const std::exception&) {
        return std::numeric_limits<double>::max();
    }
}

bool GeometryConverter::isPointInWKTPolygon(const Point& point, const std::string& polygonWkt) {
    try {
        ParsedWKTGeometry parsed = parseWKTGeometry(polygonWkt);
        if (parsed.type == ParsedWKTGeometry::POLYGON && !parsed.rings.empty()) {
            // 使用第一个环（外环）进行点在多边形内判断
            const auto& outerRing = parsed.rings[0];
            return isPointInPolygonRing(point, outerRing);
        }
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

bool GeometryConverter::wktIntersectsBoundingBox(const std::string& wkt, const BoundingBox& bbox) {
    try {
        BoundingBox geomBbox = extractBoundingBoxFromWKT(wkt);
        
        // 检查边界框是否相交
        return !(geomBbox.maxX < bbox.minX || geomBbox.minX > bbox.maxX ||
                 geomBbox.maxY < bbox.minY || geomBbox.minY > bbox.maxY);
                 
    } catch (const std::exception&) {
        return false;
    }
}

// === 几何验证功能 ===

bool GeometryConverter::isValidWKT(const std::string& wkt) {
    if (wkt.empty()) return false;
    
    std::string cleanWkt = cleanString(wkt);
    std::transform(cleanWkt.begin(), cleanWkt.end(), cleanWkt.begin(), ::toupper);
    
    // 基本的WKT格式验证
    std::vector<std::string> validTypes = {
        "POINT", "LINESTRING", "POLYGON", "MULTIPOINT", 
        "MULTILINESTRING", "MULTIPOLYGON", "GEOMETRYCOLLECTION"
    };
    
    bool hasValidType = false;
    for (const auto& type : validTypes) {
        if (cleanWkt.find(type) == 0) {
            hasValidType = true;
            break;
        }
    }
    
    if (!hasValidType) return false;
    
    // 检查括号匹配
    int openCount = 0;
    for (char c : cleanWkt) {
        if (c == '(') openCount++;
        else if (c == ')') openCount--;
        if (openCount < 0) return false;
    }
    
    if (openCount != 0) return false;
    
    // 提取括号内容进行进一步验证
    size_t firstParen = cleanWkt.find('(');
    size_t lastParen = cleanWkt.rfind(')');
    
    if (firstParen == std::string::npos || lastParen == std::string::npos || firstParen >= lastParen) {
        return false;
    }
    
    std::string content = cleanWkt.substr(firstParen + 1, lastParen - firstParen - 1);
    content = cleanString(content);
    
    // 对于POINT类型，进行更严格的坐标检查
    if (cleanWkt.find("POINT") == 0) {
        // 简单检查是否包含至少两个数字
        std::regex coordRegex(R"([+-]?\d+\.?\d*\s+[+-]?\d+\.?\d*)");
        if (!std::regex_search(content, coordRegex)) {
            return false;
        }
    }
    
    // 对于LINESTRING类型，检查是否至少有两个点
    if (cleanWkt.find("LINESTRING") == 0) {
        // LINESTRING至少需要两个点，用逗号分隔
        std::regex lineRegex(R"([+-]?\d+\.?\d*\s+[+-]?\d+\.?\d*\s*,\s*[+-]?\d+\.?\d*\s+[+-]?\d+\.?\d*)");
        if (!std::regex_search(content, lineRegex)) {
            return false;
        }
    }
    
    return true;
}

// === 私有辅助方法实现 ===

std::optional<Point> GeometryConverter::parsePointCoordinates(const std::string& wkt) {
    std::regex pointRegex(R"(POINT\s*(?:Z\s*)?\(\s*([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)(?:\s+([+-]?\d+\.?\d*))?\s*\))");
    std::smatch matches;
    
    if (std::regex_search(wkt, matches, pointRegex)) {
        double x = std::stod(matches[1].str());
        double y = std::stod(matches[2].str());
        
        if (matches[3].matched) {
            double z = std::stod(matches[3].str());
            return Point(x, y, z);
        } else {
            return Point(x, y);
        }
    }
    
    return std::nullopt;
}

std::vector<Point> GeometryConverter::parseLineStringCoordinates(const std::string& wkt) {
    std::vector<Point> points;
    
    // 提取LINESTRING括号内的坐标
    std::regex lineRegex(R"(LINESTRING\s*\(([^)]+)\))");
    std::smatch matches;
    
    if (std::regex_search(wkt, matches, lineRegex)) {
        std::string coordsStr = matches[1].str();
        points = parseCoordinateSequence(coordsStr);
    }
    
    return points;
}

std::vector<std::vector<Point>> GeometryConverter::parsePolygonRings(const std::string& wkt) {
    std::vector<std::vector<Point>> rings;
    
    // 提取POLYGON括号内的环
    std::regex polygonRegex(R"(POLYGON\s*\((.+)\))");
    std::smatch matches;
    
    if (std::regex_search(wkt, matches, polygonRegex)) {
        std::string ringsStr = matches[1].str();
        
        // 分割各个环（用括号分组）
        std::regex ringRegex(R"(\(([^)]+)\))");
        std::sregex_iterator iter(ringsStr.begin(), ringsStr.end(), ringRegex);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            std::string ringCoords = (*iter)[1].str();
            auto ringPoints = parseCoordinateSequence(ringCoords);
            if (!ringPoints.empty()) {
                rings.push_back(ringPoints);
            }
        }
    }
    
    return rings;
}

std::vector<Point> GeometryConverter::parseCoordinateSequence(const std::string& coordsStr) {
    std::vector<Point> points;
    
    if (coordsStr.empty()) {
        return points;
    }
    
    // 分割坐标对（用逗号分隔）
    std::istringstream iss(coordsStr);
    std::string coordPair;
    
    while (std::getline(iss, coordPair, ',')) {
        coordPair = cleanString(coordPair);
        if (!coordPair.empty()) {
            std::istringstream coordIss(coordPair);
            double x, y, z;
            
            if (coordIss >> x >> y) {
                if (coordIss >> z) {
                    points.emplace_back(x, y, z);
                } else {
                    points.emplace_back(x, y);
                }
            }
        }
    }
    
    return points;
}

BoundingBox GeometryConverter::calculateBoundingBoxFromPoints(const std::vector<Point>& points) {
    if (points.empty()) {
        return BoundingBox(0, 0, 0, 0);
    }
    
    double minX = points[0].x, maxX = points[0].x;
    double minY = points[0].y, maxY = points[0].y;
    
    for (const auto& point : points) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }
    
    return BoundingBox(minX, minY, maxX, maxY);
}

double GeometryConverter::calculateDistanceToLineString(const Point& point, const std::vector<Point>& linePoints) {
    if (linePoints.size() < 2) {
        return std::numeric_limits<double>::max();
    }
    
    double minDistance = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < linePoints.size() - 1; ++i) {
        double segmentDistance = calculateDistanceToSegment(point, linePoints[i], linePoints[i + 1]);
        minDistance = std::min(minDistance, segmentDistance);
    }
    
    return minDistance;
}

double GeometryConverter::calculateDistanceToPolygon(const Point& point, const std::vector<std::vector<Point>>& rings) {
    if (rings.empty() || rings[0].empty()) {
        return std::numeric_limits<double>::max();
    }
    
    // 检查点是否在多边形内
    if (isPointInPolygonRing(point, rings[0])) {
        return 0.0; // 点在多边形内，距离为0
    }
    
    // 计算到边界的最短距离
    return calculateDistanceToLineString(point, rings[0]);
}

bool GeometryConverter::isPointInPolygonRing(const Point& point, const std::vector<Point>& ring) {
    if (ring.size() < 3) {
        return false;
    }
    
    // 使用射线投射算法
    bool inside = false;
    size_t j = ring.size() - 1;
    
    for (size_t i = 0; i < ring.size(); j = i++) {
        if (((ring[i].y > point.y) != (ring[j].y > point.y)) &&
            (point.x < (ring[j].x - ring[i].x) * (point.y - ring[i].y) / (ring[j].y - ring[i].y) + ring[i].x)) {
            inside = !inside;
        }
    }
    
    return inside;
}

double GeometryConverter::calculateDistanceToSegment(const Point& point, const Point& segStart, const Point& segEnd) {
    double A = point.x - segStart.x;
    double B = point.y - segStart.y;
    double C = segEnd.x - segStart.x;
    double D = segEnd.y - segStart.y;
    
    double dot = A * C + B * D;
    double lenSq = C * C + D * D;
    
    if (lenSq == 0.0) {
        // 线段退化为点
        return calculateDistance(point, segStart);
    }
    
    double param = dot / lenSq;
    
    double xx, yy;
    
    if (param < 0.0) {
        xx = segStart.x;
        yy = segStart.y;
    } else if (param > 1.0) {
        xx = segEnd.x;
        yy = segEnd.y;
    } else {
        xx = segStart.x + param * C;
        yy = segStart.y + param * D;
    }
    
    double dx = point.x - xx;
    double dy = point.y - yy;
    return std::sqrt(dx * dx + dy * dy);
}

double GeometryConverter::calculateDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 简化实现的多点和多线解析方法
std::vector<Point> GeometryConverter::parseMultiPointCoordinates(const std::string& wkt) {
    // 简化实现：提取所有坐标对
    return parseCoordinateSequence(wkt);
}

std::vector<Point> GeometryConverter::parseMultiLineStringCoordinates(const std::string& wkt) {
    // 简化实现：只返回第一条线的坐标
    return parseLineStringCoordinates(wkt);
}

std::vector<std::vector<Point>> GeometryConverter::parseMultiPolygonRings(const std::string& wkt) {
    // 简化实现：只返回第一个多边形的环
    return parsePolygonRings(wkt);
}

double GeometryConverter::calculateDistanceToMultiPoint(const Point& point, const std::vector<Point>& multiPoints) {
    double minDistance = std::numeric_limits<double>::max();
    
    for (const auto& mp : multiPoints) {
        double distance = calculateDistance(point, mp);
        minDistance = std::min(minDistance, distance);
    }
    
    return minDistance;
}

std::string GeometryConverter::cleanString(const std::string& str) {
    std::string result = str;
    
    // 移除前导和尾随空格
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    result.erase(std::find_if(result.rbegin(), result.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), result.end());
    
    return result;
}

bool GeometryConverter::isNumeric(const std::string& str) {
    std::istringstream iss(str);
    double d;
    return iss >> d && iss.eof();
}

// === GeoJSON转换功能实现 ===

std::string GeometryConverter::pointToGeoJSON(const Point& point) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10);
    
    oss << "{\"type\":\"Point\",\"coordinates\":[" << point.x << "," << point.y;
    if (point.z.has_value()) {
        oss << "," << point.z.value();
    }
    oss << "]}";
    
    return oss.str();
}

std::string GeometryConverter::boundingBoxToGeoJSON(const BoundingBox& bbox) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10);
    
    oss << "{\"type\":\"Polygon\",\"coordinates\":[[[" 
        << bbox.minX << "," << bbox.minY << "],["
        << bbox.maxX << "," << bbox.minY << "],["
        << bbox.maxX << "," << bbox.maxY << "],["
        << bbox.minX << "," << bbox.maxY << "],["
        << bbox.minX << "," << bbox.minY << "]]]}";
    
    return oss.str();
}

std::string GeometryConverter::featureCollectionToGeoJSON(const FeatureCollection& features) {
    std::ostringstream oss;
    oss << "{\"type\":\"FeatureCollection\",\"features\":[";
    
    const auto& featureList = features.getFeatures();
    for (size_t i = 0; i < featureList.size(); ++i) {
        if (i > 0) oss << ",";
        
        const auto& feature = featureList[i];
        oss << "{\"type\":\"Feature\",\"id\":\"" << feature.id << "\",";
        oss << "\"geometry\":" << feature.geometryWkt << ","; // 简化：假设geometryWkt已经是GeoJSON格式
        oss << "\"properties\":{";
        
        // 输出属性
        bool firstAttr = true;
        for (const auto& attr : feature.attributes) {
            if (!firstAttr) oss << ",";
            firstAttr = false;
            
            oss << "\"" << attr.first << "\":";
            std::visit([&oss](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    oss << "null";
                } else if constexpr (std::is_same_v<T, std::string>) {
                    oss << "\"" << value << "\"";
                } else if constexpr (std::is_same_v<T, bool>) {
                    oss << (value ? "true" : "false");
                } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    oss << "[";
                    for (size_t i = 0; i < value.size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << "\"" << value[i] << "\"";
                    }
                    oss << "]";
                } else if constexpr (std::is_same_v<T, std::vector<int>>) {
                    oss << "[";
                    for (size_t i = 0; i < value.size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << value[i];
                    }
                    oss << "]";
                } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                    oss << "[";
                    for (size_t i = 0; i < value.size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << value[i];
                    }
                    oss << "]";
                } else {
                    oss << value;
                }
            }, attr.second);
        }
        
        oss << "}}";
    }
    
    oss << "]}";
    return oss.str();
}

std::optional<Point> GeometryConverter::parsePointFromGeoJSON(const std::string& geoJson) {
    // 简化实现：使用正则表达式解析
    std::regex pointRegex(R"("coordinates"\s*:\s*\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)(?:\s*,\s*([+-]?\d+\.?\d*))?\s*\])");
    std::smatch matches;
    
    if (std::regex_search(geoJson, matches, pointRegex)) {
        double x = std::stod(matches[1].str());
        double y = std::stod(matches[2].str());
        
        if (matches[3].matched) {
            double z = std::stod(matches[3].str());
            return Point(x, y, z);
        } else {
            return Point(x, y);
        }
    }
    
    return std::nullopt;
}

std::optional<FeatureCollection> GeometryConverter::parseFeatureCollectionFromGeoJSON(const std::string& geoJson) {
    // 简化实现：返回空的FeatureCollection
    // 实际实现需要完整的JSON解析
    FeatureCollection collection;
    return collection;
}

// === 几何验证功能实现 ===

bool GeometryConverter::isValidGeoJSON(const std::string& geoJson) {
    // 简化实现：检查基本的GeoJSON结构
    if (geoJson.empty()) return false;
    
    // 检查是否包含必要的字段
    return geoJson.find("\"type\"") != std::string::npos &&
           (geoJson.find("\"Point\"") != std::string::npos ||
            geoJson.find("\"LineString\"") != std::string::npos ||
            geoJson.find("\"Polygon\"") != std::string::npos ||
            geoJson.find("\"Feature\"") != std::string::npos ||
            geoJson.find("\"FeatureCollection\"") != std::string::npos);
}

std::string GeometryConverter::normalizeWKT(const std::string& wkt) {
    std::string result = cleanString(wkt);
    
    // 转换为大写
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    
    // 移除多余的空格
    std::regex multipleSpaces(R"(\s+)");
    result = std::regex_replace(result, multipleSpaces, " ");
    
    // 标准化括号周围的空格
    std::regex spacesAroundParens(R"(\s*\(\s*)");
    result = std::regex_replace(result, spacesAroundParens, "(");
    
    spacesAroundParens = std::regex(R"(\s*\)\s*)");
    result = std::regex_replace(result, spacesAroundParens, ")");
    
    return result;
}

// === 几何计算辅助功能实现 ===

double GeometryConverter::calculateBearing(const Point& from, const Point& to) {
    double deltaLon = (to.x - from.x) * DEG_TO_RAD;
    double lat1 = from.y * DEG_TO_RAD;
    double lat2 = to.y * DEG_TO_RAD;
    
    double y = std::sin(deltaLon) * std::cos(lat2);
    double x = std::cos(lat1) * std::sin(lat2) - std::sin(lat1) * std::cos(lat2) * std::cos(deltaLon);
    
    double bearing = std::atan2(y, x) * RAD_TO_DEG;
    
    // 转换为0-360度范围
    return std::fmod(bearing + 360.0, 360.0);
}

Point GeometryConverter::calculateDestination(
    const Point& start,
    double bearing,
    double distance) {
    
    double bearingRad = bearing * DEG_TO_RAD;
    double lat1 = start.y * DEG_TO_RAD;
    double lon1 = start.x * DEG_TO_RAD;
    double angularDistance = distance / EARTH_RADIUS_METERS;
    
    double lat2 = std::asin(std::sin(lat1) * std::cos(angularDistance) +
                           std::cos(lat1) * std::sin(angularDistance) * std::cos(bearingRad));
    
    double lon2 = lon1 + std::atan2(std::sin(bearingRad) * std::sin(angularDistance) * std::cos(lat1),
                                   std::cos(angularDistance) - std::sin(lat1) * std::sin(lat2));
    
    return Point(lon2 * RAD_TO_DEG, lat2 * RAD_TO_DEG);
}

Point GeometryConverter::calculateCentroid(const std::vector<Point>& polygon) {
    if (polygon.empty()) {
        return Point(0, 0);
    }
    
    if (polygon.size() == 1) {
        return polygon[0];
    }
    
    double area = 0.0;
    double centroidX = 0.0;
    double centroidY = 0.0;
    
    size_t n = polygon.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        double crossProduct = polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
        area += crossProduct;
        centroidX += (polygon[i].x + polygon[j].x) * crossProduct;
        centroidY += (polygon[i].y + polygon[j].y) * crossProduct;
    }
    
    area *= 0.5;
    if (std::abs(area) < EPSILON) {
        // 退化多边形，返回第一个点
        return polygon[0];
    }
    
    centroidX /= (6.0 * area);
    centroidY /= (6.0 * area);
    
    return Point(centroidX, centroidY);
}

double GeometryConverter::calculatePolygonArea(const std::vector<Point>& polygon) {
    if (polygon.size() < 3) {
        return 0.0;
    }
    
    double area = 0.0;
    size_t n = polygon.size();
    
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += polygon[i].x * polygon[j].y;
        area -= polygon[j].x * polygon[i].y;
    }
    
    return std::abs(area) * 0.5;
}

} // namespace oscean::core_services::spatial_ops::utils 