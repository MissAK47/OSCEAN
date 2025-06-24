#pragma once

#include "../engine/spatial_index_manager.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "../utils/geometry_converter.h"
#include <memory>
#include <vector>
#include <chrono>
#include <unordered_map>

namespace oscean::core_services::spatial_ops::index {

// 使用命名空间别名简化代码
using ISpatialIndex = oscean::core_services::spatial_ops::engine::ISpatialIndex;
using IndexStats = oscean::core_services::spatial_ops::engine::IndexStats;
using IndexType = oscean::core_services::spatial_ops::engine::IndexType;

/**
 * @brief 网格空间索引实现
 * 
 * 提供基于规则网格的空间数据索引和查询功能，适用于均匀分布的大规模数据集。
 * 查询性能稳定，内存使用可预测。
 */
class GridIndex : public ISpatialIndex {
public:
    /**
     * @brief 构造函数
     * @param width 网格宽度（默认100）
     * @param height 网格高度（默认100）
     */
    explicit GridIndex(size_t width = 100, size_t height = 100);

    /**
     * @brief 析构函数
     */
    ~GridIndex() override;

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
    IndexType getType() const override { return IndexType::GRID; }
    void clear() override;
    bool empty() const override;
    size_t size() const override;

    /**
     * @brief 获取网格尺寸
     * @return std::pair<width, height>
     */
    std::pair<size_t, size_t> getGridSize() const;

    /**
     * @brief 获取单元格尺寸
     * @return std::pair<cellWidth, cellHeight>
     */
    std::pair<double, double> getCellSize() const;

    /**
     * @brief 获取网格边界
     * @return 网格覆盖的边界框
     */
    oscean::core_services::BoundingBox getGridBounds() const;

private:
    struct GridCell;
    std::vector<std::vector<GridCell>> grid;
    size_t gridWidth;
    size_t gridHeight;
    oscean::core_services::BoundingBox worldBounds;
    double cellWidth;
    double cellHeight;
    std::vector<size_t> featureIds;
    std::vector<oscean::core_services::BoundingBox> boundingBoxes;
    std::unordered_map<size_t, size_t> featureIdToIndexMap; // 高效映射表
    IndexStats stats;

    /**
     * @brief 将要素插入到网格中
     * @param featureId 要素ID
     * @param bbox 要素边界框
     */
    void insertFeatureToGrid(size_t featureId, const oscean::core_services::BoundingBox& bbox);

    /**
     * @brief 计算点到边界框的距离
     * @param point 查询点
     * @param bbox 边界框
     * @return 最小距离
     */
    double calculateDistanceToBBox(const oscean::core_services::Point& point, 
                                  const oscean::core_services::BoundingBox& bbox) const;
};

} // namespace oscean::core_services::spatial_ops::index 