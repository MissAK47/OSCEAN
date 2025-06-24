/**
 * @file query_engine.cpp
 * @brief QueryEngine class implementation
 */

#include "query_engine.h"
#include "core_services/spatial_ops/spatial_exceptions.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <sstream>

namespace oscean::core_services::spatial_ops::engine {

QueryEngine::QueryEngine(const oscean::core_services::spatial_ops::SpatialOpsConfig& config)
    : m_config(config)
    , m_useIndex(config.indexSettings.strategy != oscean::core_services::spatial_ops::IndexStrategy::NONE)
    , m_indexThreshold(config.indexSettings.indexThreshold) {
}

QueryEngine::~QueryEngine() = default;

oscean::core_services::FeatureCollection QueryEngine::queryByBoundingBox(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::BoundingBox& bbox) const {
    
    if (features.getFeatures().empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Feature collection is empty");
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    
    oscean::core_services::FeatureCollection result;
    
    // 遍历所有要素，检查是否与边界框相交
    for (const auto& feature : features.getFeatures()) {
        if (intersectsBoundingBox(feature.geometryWkt, bbox)) {
            result.addFeature(feature);
        }
    }
    
    updateQueryStats(startTime);
    return result;
}

oscean::core_services::FeatureCollection QueryEngine::queryByGeometry(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::Geometry& queryGeom,
    oscean::core_services::spatial_ops::SpatialPredicate predicate) const {
    
    if (features.getFeatures().empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Feature collection is empty");
    }
    if (queryGeom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Query geometry WKT is empty");
    }

    // Placeholder implementation
    oscean::core_services::FeatureCollection result;
    return result;
}

oscean::core_services::Feature QueryEngine::findNearestNeighbor(
    const oscean::core_services::Point& point,
    const oscean::core_services::FeatureCollection& candidates) const {
    
    if (candidates.getFeatures().empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Candidate feature collection is empty");
    }

    // Placeholder implementation - return first feature
    return candidates.getFeatures()[0];
}

std::optional<oscean::core_services::GridIndex> QueryEngine::findGridCell(
    const oscean::core_services::Point& point,
    const oscean::core_services::GridDefinition& gridDef) const {
    
    // Simple implementation
    double col = (point.x - gridDef.extent.minX) / gridDef.xResolution;
    double row = (point.y - gridDef.extent.minY) / gridDef.yResolution;  // 从minY开始计算
    
    if (col < 0 || col >= static_cast<double>(gridDef.cols) || 
        row < 0 || row >= static_cast<double>(gridDef.rows)) {
        return std::nullopt;
    }
    
    return oscean::core_services::GridIndex{
        static_cast<int>(col),  // x对应col
        static_cast<int>(row)   // y对应row
    };
}

std::vector<std::optional<oscean::core_services::GridIndex>> QueryEngine::findGridCellsForPoints(
    const std::vector<oscean::core_services::Point>& points,
    const oscean::core_services::GridDefinition& gridDef) const {
    
    std::vector<std::optional<oscean::core_services::GridIndex>> result;
    result.reserve(points.size());
    
    for (const auto& point : points) {
        result.push_back(findGridCell(point, gridDef));
    }
    
    return result;
}

oscean::core_services::FeatureCollection QueryEngine::queryByRadius(
    const oscean::core_services::Point& center,
    double radius,
    const oscean::core_services::FeatureCollection& candidates) const {
    
    if (candidates.getFeatures().empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Candidate feature collection is empty");
    }
    if (radius <= 0.0) {
        throw oscean::core_services::spatial_ops::InvalidParameterException("Radius must be positive");
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    
    oscean::core_services::FeatureCollection result;
    
    // 遍历所有候选要素，计算距离
    for (const auto& feature : candidates.getFeatures()) {
        double distance = calculateDistanceToGeometry(center, feature.geometryWkt);
        if (distance <= radius) {
            result.addFeature(feature);
        }
    }
    
    updateQueryStats(startTime);
    return result;
}

oscean::core_services::FeatureCollection QueryEngine::findKNearestNeighbors(
    const oscean::core_services::Point& point,
    int k,
    const oscean::core_services::FeatureCollection& candidates) const {
    
    if (candidates.getFeatures().empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Candidate feature collection is empty");
    }
    if (k <= 0) {
        throw oscean::core_services::spatial_ops::InvalidParameterException("K must be positive");
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 创建距离-要素对的向量
    std::vector<std::pair<double, oscean::core_services::Feature>> distanceFeaturePairs;
    
    // 计算所有要素的距离
    for (const auto& feature : candidates.getFeatures()) {
        double distance = calculateDistanceToGeometry(point, feature.geometryWkt);
        distanceFeaturePairs.emplace_back(distance, feature);
    }
    
    // 按距离排序
    std::sort(distanceFeaturePairs.begin(), distanceFeaturePairs.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });
    
    // 取前K个要素
    oscean::core_services::FeatureCollection result;
    int count = std::min(k, static_cast<int>(distanceFeaturePairs.size()));
    for (int i = 0; i < count; ++i) {
        result.addFeature(distanceFeaturePairs[i].second);
    }
    
    updateQueryStats(startTime);
    return result;
}

void QueryEngine::setUseIndex(bool useIndex) {
    m_useIndex = useIndex;
}

QueryEngine::QueryStats QueryEngine::getQueryStats() const {
    return m_stats;
}

bool QueryEngine::shouldUseIndex(size_t featureCount) const {
    return m_useIndex && featureCount >= m_indexThreshold;
}

oscean::core_services::FeatureCollection QueryEngine::filterByPredicate(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::Geometry& queryGeom,
    oscean::core_services::spatial_ops::SpatialPredicate predicate) const {
    
    // Placeholder implementation
    oscean::core_services::FeatureCollection result;
    return result;
}

bool QueryEngine::intersectsBoundingBox(const std::string& geometryWkt, const oscean::core_services::BoundingBox& bbox) const {
    // 从WKT中提取边界框
    auto geomBbox = extractBoundingBoxFromWKT(geometryWkt);
    
    // 检查两个边界框是否相交
    return !(geomBbox.maxX < bbox.minX || geomBbox.minX > bbox.maxX ||
             geomBbox.maxY < bbox.minY || geomBbox.minY > bbox.maxY);
}

oscean::core_services::FeatureCollection QueryEngine::filterByIntersection(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::Geometry& queryGeom) const {
    
    // Placeholder implementation
    oscean::core_services::FeatureCollection result;
    return result;
}

oscean::core_services::FeatureCollection QueryEngine::filterByContainment(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::Geometry& queryGeom,
    bool featureContainsQuery) const {
    
    // Placeholder implementation
    oscean::core_services::FeatureCollection result;
    return result;
}

oscean::core_services::FeatureCollection QueryEngine::filterByDisjoint(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::Geometry& queryGeom) const {
    
    // Placeholder implementation
    oscean::core_services::FeatureCollection result;
    return result;
}

double QueryEngine::calculateDistanceToGeometry(const oscean::core_services::Point& point, const std::string& geometryWkt) const {
    if (geometryWkt.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    // 处理POINT类型
    if (geometryWkt.find("POINT") != std::string::npos) {
        // 提取坐标 "POINT(x y)"
        size_t start = geometryWkt.find('(');
        size_t end = geometryWkt.find(')');
        if (start != std::string::npos && end != std::string::npos) {
            std::string coords = geometryWkt.substr(start + 1, end - start - 1);
            size_t spacePos = coords.find(' ');
            if (spacePos != std::string::npos) {
                double x = std::stod(coords.substr(0, spacePos));
                double y = std::stod(coords.substr(spacePos + 1));
                
                // 计算欧几里得距离
                double dx = point.x - x;
                double dy = point.y - y;
                return std::sqrt(dx * dx + dy * dy);
            }
        }
    }
    
    // 处理POLYGON类型 - 简化为点在多边形内则距离为0，否则计算到边界的距离
    if (geometryWkt.find("POLYGON") != std::string::npos) {
        // 简化实现：提取多边形的边界框，如果点在边界框内则认为距离为0
        auto bbox = extractBoundingBoxFromWKT(geometryWkt);
        
        // 检查点是否在边界框内
        if (point.x >= bbox.minX && point.x <= bbox.maxX &&
            point.y >= bbox.minY && point.y <= bbox.maxY) {
            return 0.0; // 简化：认为在多边形内
        }
        
        // 计算到边界框的最小距离
        double dx = std::max(0.0, std::max(bbox.minX - point.x, point.x - bbox.maxX));
        double dy = std::max(0.0, std::max(bbox.minY - point.y, point.y - bbox.maxY));
        return std::sqrt(dx * dx + dy * dy);
    }
    
    // 默认返回最大距离
    return std::numeric_limits<double>::max();
}

double QueryEngine::calculatePointToSegmentDistance(const oscean::core_services::Point& point, const oscean::core_services::Point& segmentStart, const oscean::core_services::Point& segmentEnd) const {
    // Placeholder implementation
    return 0.0;
}

std::optional<oscean::core_services::GridIndex> QueryEngine::findGridCellForPoint(
    const oscean::core_services::Point& point,
    const oscean::core_services::GridDefinition& gridDef,
    double cellWidth,
    double cellHeight) const {
    
    return findGridCell(point, gridDef);
}

oscean::core_services::BoundingBox QueryEngine::extractBoundingBoxFromWKT(const std::string& wkt) const {
    // 简单的WKT解析实现
    if (wkt.empty()) {
        return oscean::core_services::BoundingBox(0.0, 0.0, 0.0, 0.0);
    }
    
    // 查找POINT类型
    if (wkt.find("POINT") != std::string::npos) {
        // 提取坐标 "POINT(x y)"
        size_t start = wkt.find('(');
        size_t end = wkt.find(')');
        if (start != std::string::npos && end != std::string::npos) {
            std::string coords = wkt.substr(start + 1, end - start - 1);
            size_t spacePos = coords.find(' ');
            if (spacePos != std::string::npos) {
                double x = std::stod(coords.substr(0, spacePos));
                double y = std::stod(coords.substr(spacePos + 1));
                return oscean::core_services::BoundingBox(x, y, x, y);
            }
        }
    }
    
    // 查找POLYGON类型
    if (wkt.find("POLYGON") != std::string::npos) {
        // 简化处理：提取所有数字并找到最小最大值
        std::vector<double> coords;
        std::string numbers = wkt;
        
        // 移除非数字字符（保留数字、小数点、负号、空格）
        for (char& c : numbers) {
            if (!std::isdigit(c) && c != '.' && c != '-' && c != ' ' && c != '\t') {
                c = ' ';
            }
        }
        
        std::istringstream iss(numbers);
        std::string token;
        while (iss >> token) {
            try {
                coords.push_back(std::stod(token));
            } catch (...) {
                // 忽略无效数字
            }
        }
        
        if (coords.size() >= 4) {
            double minX = coords[0], maxX = coords[0];
            double minY = coords[1], maxY = coords[1];
            
            for (size_t i = 0; i < coords.size(); i += 2) {
                if (i + 1 < coords.size()) {
                    minX = std::min(minX, coords[i]);
                    maxX = std::max(maxX, coords[i]);
                    minY = std::min(minY, coords[i + 1]);
                    maxY = std::max(maxY, coords[i + 1]);
                }
            }
            
            return oscean::core_services::BoundingBox(minX, minY, maxX, maxY);
        }
    }
    
    // 默认返回空边界框
    return oscean::core_services::BoundingBox(0.0, 0.0, 0.0, 0.0);
}

void QueryEngine::updateQueryStats(
    const std::chrono::high_resolution_clock::time_point& startTime) const {
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    // 使用const_cast来修改mutable统计数据
    auto& stats = const_cast<QueryStats&>(m_stats);
    
    stats.totalQueries++;
    double queryTime = duration.count() / 1000.0; // Convert to milliseconds
    stats.averageQueryTime = (stats.averageQueryTime * (stats.totalQueries - 1) + queryTime) / stats.totalQueries;
}

} // namespace oscean::core_services::spatial_ops::engine 