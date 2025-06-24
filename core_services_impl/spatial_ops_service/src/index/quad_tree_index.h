#pragma once

#include "../engine/spatial_index_manager.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "../utils/geometry_converter.h"
#include <memory>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <set>

namespace oscean::core_services::spatial_ops::index {

// 使用命名空间别名简化代码
using ISpatialIndex = oscean::core_services::spatial_ops::engine::ISpatialIndex;
using IndexStats = oscean::core_services::spatial_ops::engine::IndexStats;
using IndexType = oscean::core_services::spatial_ops::engine::IndexType;

/**
 * @brief 四叉树空间索引实现
 * 
 * 提供基于四叉树的空间数据索引和查询功能，适用于点数据和小规模几何体。
 * 特别适合均匀分布的空间数据。
 */
class QuadTreeIndex : public ISpatialIndex {
public:
    /**
     * @brief 构造函数
     * @param maxCapacity 叶子节点最大容量（默认10）
     * @param maxDepth 树的最大深度（默认8）
     */
    explicit QuadTreeIndex(size_t maxCapacity = 10, size_t maxDepth = 8);

    /**
     * @brief 析构函数
     */
    ~QuadTreeIndex() override;

    /**
     * @brief 构建索引
     * @param features 要素集合
     */
    void build(const oscean::core_services::FeatureCollection& features);

    // === ISpatialIndex接口实现 ===
    
    std::vector<size_t> query(const oscean::core_services::BoundingBox& bbox) override;
    std::vector<size_t> query(const oscean::core_services::Point& point) override;
    std::vector<size_t> query(const oscean::core_services::Geometry& geom) const override;
    std::vector<size_t> radiusQuery(const oscean::core_services::Point& center, double radius) override;
    std::vector<size_t> nearestNeighbors(const oscean::core_services::Point& point, size_t k) override;
    void insert(size_t featureId, const oscean::core_services::BoundingBox& bbox) override;
    void remove(size_t featureId) override;
    void update(size_t featureId, const oscean::core_services::BoundingBox& newBbox) override;
    IndexStats getStats() const override;
    IndexType getType() const override { return IndexType::QUADTREE; }
    void clear() override;
    bool empty() const override;
    size_t size() const override;

    /**
     * @brief 获取树的深度
     * @return 树的最大深度
     */
    size_t getTreeDepth() const;

    /**
     * @brief 获取节点数量
     * @return 总节点数量
     */
    size_t getNodeCount() const;

    /**
     * @brief 获取叶子节点数量
     * @return 叶子节点数量
     */
    size_t getLeafCount() const;

    /**
     * @brief 更新查询统计信息
     * @param startTime 查询开始时间
     */
    void updateQueryStats(const std::chrono::high_resolution_clock::time_point& startTime);

private:
    struct QuadTreeNode;
    std::unique_ptr<QuadTreeNode> root_;
    std::vector<size_t> featureIds_;
    std::vector<oscean::core_services::BoundingBox> boundingBoxes_;
    std::unordered_map<size_t, size_t> featureIdToIndex_;
    IndexStats stats_;
    oscean::core_services::BoundingBox worldBounds_;
    size_t maxCapacity_;
    size_t maxDepth_;

    /**
     * @brief 计算点到边界框的距离
     * @param point 查询点
     * @param bbox 边界框
     * @return 最小距离
     */
    double calculateDistanceToBBox(const oscean::core_services::Point& point, 
                                  const oscean::core_services::BoundingBox& bbox) const;

    /**
     * @brief 递归计算树的深度
     * @param node 节点指针
     * @return 从该节点开始的深度
     */
    size_t getTreeDepthRecursive(const QuadTreeNode* node) const;
    
    /**
     * @brief 计算树的平均深度
     * @param node 节点指针
     * @return 平均深度
     */
    double calculateAverageDepth(const QuadTreeNode* node) const;
    
    /**
     * @brief 检查是否需要重建树
     * @param newBbox 新的边界框
     * @return 是否需要重建
     */
    bool needsRebuild(const oscean::core_services::BoundingBox& newBbox) const;
    
    /**
     * @brief 重建整个树结构
     */
    void rebuildTree();
    
    /**
     * @brief 优化的范围查询方法（避免重复）
     * @param node 当前节点
     * @param queryBounds 查询边界框
     * @param results 结果集合（使用set避免重复）
     */
    void queryRangeOptimized(const QuadTreeNode* node, const oscean::core_services::BoundingBox& queryBounds, std::set<size_t>& results) const;
    
    /**
     * @brief 检查两个边界框是否相交
     * @param bbox1 第一个边界框
     * @param bbox2 第二个边界框
     * @return 是否相交
     */
    bool intersects(const oscean::core_services::BoundingBox& bbox1, const oscean::core_services::BoundingBox& bbox2) const;
    
    /**
     * @brief 优化的半径查询方法（避免重复）
     * @param node 当前节点
     * @param center 查询中心点
     * @param radius 查询半径
     * @param results 结果集合（使用set避免重复）
     */
    void radiusQueryOptimized(const QuadTreeNode* node, const oscean::core_services::Point& center, double radius, std::set<size_t>& results) const;
};

} // namespace oscean::core_services::spatial_ops::index 