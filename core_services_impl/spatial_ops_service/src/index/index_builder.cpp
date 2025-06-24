#include "../engine/spatial_index_manager.h"
#include "r_tree_index.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "../utils/geometry_converter.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <future>
#include <map>

using namespace oscean::core_services;
using namespace oscean::core_services::spatial_ops;

namespace oscean::core_services::spatial_ops::engine {

// === IndexBuilder 静态方法实现 ===

std::unique_ptr<ISpatialIndex> IndexBuilder::buildRTreeIndex(
    const FeatureCollection& features,
    const SpatialOpsConfig& config) {
    
    auto rtreeIndex = std::make_unique<index::RTreeIndex>(16, 4);
    
    try {
        // 设置并行构建参数
        if (config.parallelSettings.strategy != ParallelStrategy::NONE) {
            rtreeIndex->setParallelBuild(true, config.parallelSettings.maxThreads);
        }
        
        // 构建索引
        rtreeIndex->build(features);
        
        return std::move(rtreeIndex);
        
    } catch (const std::exception& e) {
        throw SpatialOpsException("R-tree索引构建失败: " + std::string(e.what()));
    }
}

BoundingBox IndexBuilder::calculateBounds(const FeatureCollection& features) {
    const auto& featureList = features.getFeatures();
    
    if (featureList.empty()) {
        return BoundingBox{0.0, 0.0, 0.0, 0.0};
    }
    
    // 初始化边界框
    BoundingBox bounds = utils::GeometryConverter::extractBoundingBoxFromWKT(featureList[0].geometryWkt);
    
    // 扩展边界框以包含所有要素
    for (size_t i = 1; i < featureList.size(); ++i) {
        BoundingBox featureBounds = utils::GeometryConverter::extractBoundingBoxFromWKT(featureList[i].geometryWkt);
        
        bounds.minX = std::min(bounds.minX, featureBounds.minX);
        bounds.minY = std::min(bounds.minY, featureBounds.minY);
        bounds.maxX = std::max(bounds.maxX, featureBounds.maxX);
        bounds.maxY = std::max(bounds.maxY, featureBounds.maxY);
    }
    
    return bounds;
}

std::map<std::string, double> IndexBuilder::analyzeFeatureDistribution(const FeatureCollection& features) {
    std::map<std::string, double> analysis;
    const auto& featureList = features.getFeatures();
    
    analysis["feature_count"] = static_cast<double>(featureList.size());
    
    if (featureList.empty()) {
        return analysis;
    }
    
    // 计算整体边界框
    BoundingBox bounds = calculateBounds(features);
    double totalArea = (bounds.maxX - bounds.minX) * (bounds.maxY - bounds.minY);
    analysis["total_area"] = totalArea;
    
    // 计算要素密度
    analysis["feature_density"] = featureList.size() / std::max(totalArea, 1.0);
    
    // 分析几何类型分布
    size_t pointCount = 0;
    size_t lineCount = 0;
    size_t polygonCount = 0;
    
    for (const auto& feature : featureList) {
        const std::string& wkt = feature.geometryWkt;
        if (wkt.find("POINT") != std::string::npos) {
            pointCount++;
        } else if (wkt.find("LINESTRING") != std::string::npos || wkt.find("MULTILINESTRING") != std::string::npos) {
            lineCount++;
        } else if (wkt.find("POLYGON") != std::string::npos || wkt.find("MULTIPOLYGON") != std::string::npos) {
            polygonCount++;
        }
    }
    
    analysis["point_ratio"] = static_cast<double>(pointCount) / featureList.size();
    analysis["line_ratio"] = static_cast<double>(lineCount) / featureList.size();
    analysis["polygon_ratio"] = static_cast<double>(polygonCount) / featureList.size();
    
    // 计算空间分布的均匀性
    double width = bounds.maxX - bounds.minX;
    double height = bounds.maxY - bounds.minY;
    analysis["aspect_ratio"] = (height > 0) ? width / height : 1.0;
    
    return analysis;
}

} // namespace oscean::core_services::spatial_ops::engine 