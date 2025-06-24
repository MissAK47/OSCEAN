#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include <vector>
#include <optional>
#include <memory>
#include <chrono>
#include <mutex>

namespace oscean::core_services::spatial_ops::engine {

/**
 * @class QueryEngine
 * @brief 高性能空间查询引擎
 * 
 * 提供各种空间查询算法，包括边界框查询、几何查询、最近邻查找等。
 * 支持空间索引加速和并行查询。
 */
class QueryEngine {
public:
    /**
     * @brief 构造函数
     * @param config 空间服务配置
     */
    explicit QueryEngine(const oscean::core_services::spatial_ops::SpatialOpsConfig& config);
    
    ~QueryEngine();

    // --- 基础查询功能 ---
    
    /**
     * @brief 根据边界框查询要素集合
     * @param features 要查询的要素集合
     * @param bbox 查询边界框
     * @return 与边界框相交的要素子集
     */
    oscean::core_services::FeatureCollection queryByBoundingBox(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::BoundingBox& bbox) const;

    /**
     * @brief 根据几何体和空间谓词查询要素集合
     * @param features 要查询的要素集合
     * @param queryGeom 查询几何体
     * @param predicate 空间谓词
     * @return 满足条件的要素子集
     */
    oscean::core_services::FeatureCollection queryByGeometry(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::Geometry& queryGeom,
        oscean::core_services::spatial_ops::SpatialPredicate predicate) const;

    /**
     * @brief 查找最近邻要素
     * @param point 参考点
     * @param candidates 候选要素集合
     * @return 最近的要素
     */
    oscean::core_services::Feature findNearestNeighbor(
        const oscean::core_services::Point& point,
        const oscean::core_services::FeatureCollection& candidates) const;

    // --- 网格查询功能 ---
    
    /**
     * @brief 查找点所在的网格单元
     * @param point 查询点
     * @param gridDef 网格定义
     * @return 网格索引，如果点在网格外则返回nullopt
     */
    std::optional<oscean::core_services::GridIndex> findGridCell(
        const oscean::core_services::Point& point,
        const oscean::core_services::GridDefinition& gridDef) const;

    /**
     * @brief 批量查找点所在的网格单元
     * @param points 查询点集合
     * @param gridDef 网格定义
     * @return 网格索引集合
     */
    std::vector<std::optional<oscean::core_services::GridIndex>> findGridCellsForPoints(
        const std::vector<oscean::core_services::Point>& points,
        const oscean::core_services::GridDefinition& gridDef) const;

    // --- 高级查询功能 ---
    
    /**
     * @brief 半径查询 - 查找指定半径内的要素
     * @param center 中心点
     * @param radius 查询半径
     * @param candidates 候选要素集合
     * @return 半径内的要素集合
     */
    oscean::core_services::FeatureCollection queryByRadius(
        const oscean::core_services::Point& center,
        double radius,
        const oscean::core_services::FeatureCollection& candidates) const;

    /**
     * @brief K最近邻查询
     * @param point 参考点
     * @param k 邻居数量
     * @param candidates 候选要素集合
     * @return K个最近的要素
     */
    oscean::core_services::FeatureCollection findKNearestNeighbors(
        const oscean::core_services::Point& point,
        int k,
        const oscean::core_services::FeatureCollection& candidates) const;

    // --- 性能优化功能 ---
    
    /**
     * @brief 设置是否使用空间索引
     * @param useIndex 是否使用索引
     */
    void setUseIndex(bool useIndex);

    /**
     * @brief 获取查询统计信息
     * @return 查询性能统计
     */
    struct QueryStats {
        size_t totalQueries = 0;
        double averageQueryTime = 0.0;
        size_t indexHits = 0;
        size_t indexMisses = 0;
    };
    
    QueryStats getQueryStats() const;

private:
    // 配置参数
    oscean::core_services::spatial_ops::SpatialOpsConfig m_config;
    bool m_useIndex;
    size_t m_indexThreshold;
    
    // 性能统计
    mutable QueryStats m_stats;
    
    // 内部辅助方法
    bool shouldUseIndex(size_t featureCount) const;
    oscean::core_services::FeatureCollection filterByPredicate(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::Geometry& queryGeom,
        oscean::core_services::spatial_ops::SpatialPredicate predicate) const;
    
    // 几何查询辅助方法
    bool intersectsBoundingBox(const std::string& geometryWkt, const oscean::core_services::BoundingBox& bbox) const;
    oscean::core_services::FeatureCollection filterByIntersection(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::Geometry& queryGeom) const;
    oscean::core_services::FeatureCollection filterByContainment(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::Geometry& queryGeom,
        bool featureContainsQuery) const;
    oscean::core_services::FeatureCollection filterByDisjoint(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::Geometry& queryGeom) const;
    
    // 距离计算辅助方法
    double calculateDistanceToGeometry(const oscean::core_services::Point& point, const std::string& geometryWkt) const;
    double calculatePointToSegmentDistance(const oscean::core_services::Point& point, const oscean::core_services::Point& segmentStart, const oscean::core_services::Point& segmentEnd) const;
    
    // 网格查询辅助方法
    std::optional<oscean::core_services::GridIndex> findGridCellForPoint(
        const oscean::core_services::Point& point,
        const oscean::core_services::GridDefinition& gridDef,
        double cellWidth,
        double cellHeight) const;
    
    // WKT处理辅助方法
    oscean::core_services::BoundingBox extractBoundingBoxFromWKT(const std::string& wkt) const;
    
    // 性能统计辅助方法
    void updateQueryStats(
        const std::chrono::high_resolution_clock::time_point& startTime) const;
};

} // namespace oscean::core_services::spatial_ops::engine 