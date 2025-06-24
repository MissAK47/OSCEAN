#pragma once

#include "../engine/spatial_index_manager.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "../utils/geometry_converter.h"
#include <memory>
#include <vector>
#include <chrono>

namespace oscean::core_services::spatial_ops::index {

// 使用命名空间别名简化代码
using ISpatialIndex = oscean::core_services::spatial_ops::engine::ISpatialIndex;
using IndexStats = oscean::core_services::spatial_ops::engine::IndexStats;
using IndexType = oscean::core_services::spatial_ops::engine::IndexType;

// 前向声明
struct RTreeNode;

/**
 * @brief R-tree空间索引实现
 * 
 * 提供高效的空间数据索引和查询功能，适用于大规模空间数据集。
 * 支持动态插入、删除和更新操作。
 */
class RTreeIndex : public ISpatialIndex {
public:
    /**
     * @brief 构造函数
     * @param maxEntries 节点最大条目数（默认16）
     * @param minEntries 节点最小条目数（默认4）
     */
    explicit RTreeIndex(size_t maxEntries = 16, size_t minEntries = 4);

    /**
     * @brief 析构函数
     */
    ~RTreeIndex() override;

    /**
     * @brief 构建索引
     * @param features 要素集合
     */
    void build(const oscean::core_services::FeatureCollection& features);

    /**
     * @brief 设置并行构建参数
     * @param useParallel 是否使用并行构建
     * @param numThreads 线程数量（0表示自动检测）
     */
    void setParallelBuild(bool useParallel, size_t numThreads = 0);

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
    IndexType getType() const override { return IndexType::RTREE; }
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
     * @brief 优化索引结构
     * @return 优化是否成功
     */
    bool optimize();

    /**
     * @brief 强制完成索引构建
     * 确保所有插入的要素都被正确索引
     */
    void finalizeBuild();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    /**
     * @brief 计算点到边界框的距离
     * @param point 查询点
     * @param bbox 边界框
     * @return 最小距离
     */
    double calculateDistanceToBBox(const oscean::core_services::Point& point, 
                                  const oscean::core_services::BoundingBox& bbox) const;

    /**
     * @brief 更新查询统计信息
     * @param startTime 查询开始时间
     */
    void updateQueryStats(const std::chrono::high_resolution_clock::time_point& startTime);
    
    /**
     * @brief 计算树的深度
     * @param node 节点指针
     * @return 从该节点开始的深度
     */
    size_t calculateTreeDepth(const RTreeNode* node) const;
    
    /**
     * @brief 递归计算节点数量
     * @param node 节点指针
     * @return 从该节点开始的总节点数
     */
    size_t countNodesRecursive(const RTreeNode* node) const;
    
    /**
     * @brief 递归计算叶子节点数量
     * @param node 节点指针
     * @return 从该节点开始的叶子节点数
     */
    size_t countLeavesRecursive(const RTreeNode* node) const;
    
    /**
     * @brief 计算平均深度
     * @param node 节点指针
     * @return 平均深度
     */
    double calculateAverageDepth(const RTreeNode* node) const;
    
    /**
     * @brief 递归计算深度总和（辅助方法）
     * @param node 节点指针
     * @param currentDepth 当前深度
     * @param totalDepth 深度总和（输出参数）
     * @param leafCount 叶子节点数量（输出参数）
     */
    void calculateDepthSum(const RTreeNode* node, size_t currentDepth, 
                          size_t& totalDepth, size_t& leafCount) const;
};

} // namespace oscean::core_services::spatial_ops::index 