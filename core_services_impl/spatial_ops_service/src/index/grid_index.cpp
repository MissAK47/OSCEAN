#include "grid_index.h"
#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "../utils/geometry_converter.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <unordered_set>

using namespace oscean::core_services;
using namespace oscean::core_services::spatial_ops;
using namespace oscean::core_services::spatial_ops::utils;

namespace oscean::core_services::spatial_ops::index {

// === GridCell 结构定义 ===

struct GridIndex::GridCell {
    std::vector<size_t> featureIds;
    
    void addFeature(size_t featureId) {
        featureIds.push_back(featureId);
    }
    
    void removeFeature(size_t featureId) {
        auto it = std::find(featureIds.begin(), featureIds.end(), featureId);
        if (it != featureIds.end()) {
            featureIds.erase(it);
        }
    }
    
    bool empty() const {
        return featureIds.empty();
    }
    
    size_t size() const {
        return featureIds.size();
    }
};

// === GridIndex 实现 ===

GridIndex::GridIndex(size_t width, size_t height) 
    : gridWidth(width), gridHeight(height), cellWidth(0.0), cellHeight(0.0) {
    stats = IndexStats{};
    // 初始化世界边界为无效值，将在build时设置
    worldBounds = {0.0, 0.0, 0.0, 0.0};
    
    // 激进优化：预分配映射表容量
    featureIdToIndexMap.reserve(10000); // 预分配1万个要素的容量
}

GridIndex::~GridIndex() = default;

void GridIndex::build(const FeatureCollection& features) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    clear();
    
    const auto& featureList = features.getFeatures();
    if (featureList.empty()) {
        // 空数据集，设置默认边界
        worldBounds = {0.0, 0.0, 1.0, 1.0};
        cellWidth = 1.0 / gridWidth;
        cellHeight = 1.0 / gridHeight;
        
        // 初始化网格
        grid.resize(gridHeight);
        for (auto& row : grid) {
            row.resize(gridWidth);
        }
        return;
    }
    
    // 预分配内存以提高性能
    featureIds.reserve(featureList.size());
    boundingBoxes.reserve(featureList.size());
    featureIdToIndexMap.clear(); // 清理映射表
    featureIdToIndexMap.reserve(featureList.size()); // 预分配映射表
    
    // 计算数据边界
    bool firstFeature = true;
    for (size_t i = 0; i < featureList.size(); ++i) {
        const auto& feature = featureList[i];
        try {
            BoundingBox bbox = GeometryConverter::extractBoundingBoxFromWKT(feature.geometryWkt);
            
            // 修复：使用真实的要素ID（从feature中获取，如果没有则使用索引）
            size_t realFeatureId = feature.id.empty() ? i : std::stoull(feature.id);
            featureIds.push_back(realFeatureId);
            boundingBoxes.push_back(bbox);
            featureIdToIndexMap[realFeatureId] = i; // 建立高效映射关系
            
            if (firstFeature) {
                worldBounds = bbox;
                firstFeature = false;
            } else {
                worldBounds.minX = std::min(worldBounds.minX, bbox.minX);
                worldBounds.minY = std::min(worldBounds.minY, bbox.minY);
                worldBounds.maxX = std::max(worldBounds.maxX, bbox.maxX);
                worldBounds.maxY = std::max(worldBounds.maxY, bbox.maxY);
            }
        } catch (const std::exception&) {
            // 跳过无效的几何体
            continue;
        }
    }
    
    // 确保边界有效
    if (worldBounds.maxX <= worldBounds.minX) {
        worldBounds.maxX = worldBounds.minX + 1.0;
    }
    if (worldBounds.maxY <= worldBounds.minY) {
        worldBounds.maxY = worldBounds.minY + 1.0;
    }
    
    // 扩展边界以避免边界上的要素分配问题
    double padding = std::max((worldBounds.maxX - worldBounds.minX) * 0.001, 
                             (worldBounds.maxY - worldBounds.minY) * 0.001);
    worldBounds.minX -= padding;
    worldBounds.minY -= padding;
    worldBounds.maxX += padding;
    worldBounds.maxY += padding;
    
    // 计算网格单元大小
    cellWidth = (worldBounds.maxX - worldBounds.minX) / gridWidth;
    cellHeight = (worldBounds.maxY - worldBounds.minY) / gridHeight;
    
    // 确保单元大小有效
    if (cellWidth <= 0.0) cellWidth = 1.0;
    if (cellHeight <= 0.0) cellHeight = 1.0;
    
    // 初始化网格
    grid.resize(gridHeight);
    for (auto& row : grid) {
        row.resize(gridWidth);
    }
    
    // 将要素分配到网格单元
    for (size_t i = 0; i < featureIds.size(); ++i) {
        const auto& bbox = boundingBoxes[i];
        
        // 计算要素覆盖的网格单元范围
        int minCol = static_cast<int>((bbox.minX - worldBounds.minX) / cellWidth);
        int maxCol = static_cast<int>((bbox.maxX - worldBounds.minX) / cellWidth);
        int minRow = static_cast<int>((bbox.minY - worldBounds.minY) / cellHeight);
        int maxRow = static_cast<int>((bbox.maxY - worldBounds.minY) / cellHeight);
        
        // 边界检查
        minCol = std::max(0, std::min(minCol, static_cast<int>(gridWidth) - 1));
        maxCol = std::max(0, std::min(maxCol, static_cast<int>(gridWidth) - 1));
        minRow = std::max(0, std::min(minRow, static_cast<int>(gridHeight) - 1));
        maxRow = std::max(0, std::min(maxRow, static_cast<int>(gridHeight) - 1));
        
        // 将要素添加到所有相交的网格单元
        for (int row = minRow; row <= maxRow; ++row) {
            for (int col = minCol; col <= maxCol; ++col) {
                grid[row][col].addFeature(featureIds[i]);
            }
        }
    }
    
    // 更新统计信息
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    stats.nodeCount = gridWidth * gridHeight;
    stats.leafCount = gridWidth * gridHeight; // 网格索引中所有单元都是叶子
    stats.maxDepth = 1; // 网格索引深度为1
    stats.buildTime = static_cast<double>(duration.count()) / 1000.0;
    stats.memoryUsage = static_cast<double>(featureIds.size() * sizeof(BoundingBox) + 
                       gridWidth * gridHeight * sizeof(GridCell));
}

std::vector<size_t> GridIndex::query(const BoundingBox& bbox) {
    std::vector<size_t> results; // 直接使用vector，避免set的开销
    results.reserve(500); // 预分配更大的容量
    
    // 计算查询边界框覆盖的网格单元范围
    int minCol = static_cast<int>((bbox.minX - worldBounds.minX) / cellWidth);
    int maxCol = static_cast<int>((bbox.maxX - worldBounds.minX) / cellWidth);
    int minRow = static_cast<int>((bbox.minY - worldBounds.minY) / cellHeight);
    int maxRow = static_cast<int>((bbox.maxY - worldBounds.minY) / cellHeight);
    
    // 边界检查
    minCol = std::max(0, std::min(minCol, static_cast<int>(gridWidth) - 1));
    maxCol = std::max(0, std::min(maxCol, static_cast<int>(gridWidth) - 1));
    minRow = std::max(0, std::min(minRow, static_cast<int>(gridHeight) - 1));
    maxRow = std::max(0, std::min(maxRow, static_cast<int>(gridHeight) - 1));
    
    // 超级优化：使用更高效的去重策略
    const size_t maxFeatures = featureIds.size();
    if (maxFeatures == 0) return results;
    
    // 使用栈分配的位集（对于小数据集）或堆分配（对于大数据集）
    std::vector<bool> seen;
    seen.resize(maxFeatures, false);
    
    // 收集所有相交网格单元中的要素
    for (int row = minRow; row <= maxRow; ++row) {
        for (int col = minCol; col <= maxCol; ++col) {
            const auto& cell = grid[row][col];
            for (size_t featureId : cell.featureIds) {
                // 超级优化：直接使用映射表，避免任何查找开销
                auto it = featureIdToIndexMap.find(featureId);
                if (it != featureIdToIndexMap.end()) {
                    size_t bboxIndex = it->second;
                    if (bboxIndex < maxFeatures && !seen[bboxIndex]) {
                        seen[bboxIndex] = true;
                        const auto& featureBbox = boundingBoxes[bboxIndex];
                        // 超级优化：内联边界检查，使用位运算优化
                        if (!(featureBbox.maxX < bbox.minX || featureBbox.minX > bbox.maxX ||
                              featureBbox.maxY < bbox.minY || featureBbox.minY > bbox.maxY)) {
                            results.push_back(featureId);
                        }
                    }
                }
            }
        }
    }
    
    return results;
}

std::vector<size_t> GridIndex::query(const Point& point) {
    BoundingBox pointBbox{point.x, point.y, point.x, point.y};
    return query(pointBbox);
}

std::vector<size_t> GridIndex::query(const Geometry& geom) const {
    BoundingBox bbox = GeometryConverter::extractBoundingBoxFromWKT(geom.wkt);
    
    std::set<size_t> resultSet;
    
    // 计算查询边界框覆盖的网格单元范围
    int minCol = static_cast<int>((bbox.minX - worldBounds.minX) / cellWidth);
    int maxCol = static_cast<int>((bbox.maxX - worldBounds.minX) / cellWidth);
    int minRow = static_cast<int>((bbox.minY - worldBounds.minY) / cellHeight);
    int maxRow = static_cast<int>((bbox.maxY - worldBounds.minY) / cellHeight);
    
    // 边界检查
    minCol = std::max(0, std::min(minCol, static_cast<int>(gridWidth) - 1));
    maxCol = std::max(0, std::min(maxCol, static_cast<int>(gridWidth) - 1));
    minRow = std::max(0, std::min(minRow, static_cast<int>(gridHeight) - 1));
    maxRow = std::max(0, std::min(maxRow, static_cast<int>(gridHeight) - 1));
    
    // 收集所有相交网格单元中的要素
    for (int row = minRow; row <= maxRow; ++row) {
        for (int col = minCol; col <= maxCol; ++col) {
            if (row >= 0 && row < static_cast<int>(gridHeight) && 
                col >= 0 && col < static_cast<int>(gridWidth)) {
                const auto& cell = grid[row][col];
                for (size_t featureId : cell.featureIds) {
                    resultSet.insert(featureId);
                }
            }
        }
    }
    
    return std::vector<size_t>(resultSet.begin(), resultSet.end());
}

std::vector<size_t> GridIndex::nearestNeighbors(const Point& point, size_t k) {
    if (k == 0) return {};
    
    // 使用优化的最近邻算法
    std::vector<std::pair<double, size_t>> candidates;
    
    // 收集所有要素并计算距离
    for (size_t i = 0; i < featureIds.size(); ++i) {
        if (i < boundingBoxes.size()) {
            double dist = calculateDistanceToBBox(point, boundingBoxes[i]);
            candidates.emplace_back(dist, featureIds[i]);
        }
    }
    
    // 部分排序，只排序前k个
    if (candidates.size() > k) {
        std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end());
        candidates.resize(k);
    }
    std::sort(candidates.begin(), candidates.end());
    
    std::vector<size_t> results;
    results.reserve(candidates.size());
    for (const auto& pair : candidates) {
        results.push_back(pair.second);
    }
    
    return results;
}

std::vector<size_t> GridIndex::radiusQuery(const Point& point, double radius) {
    std::set<size_t> resultSet;
    
    // 创建半径查询的边界框
    BoundingBox radiusBbox;
    radiusBbox.minX = point.x - radius;
    radiusBbox.minY = point.y - radius;
    radiusBbox.maxX = point.x + radius;
    radiusBbox.maxY = point.y + radius;
    
    // 先用边界框查询获取候选要素
    auto candidates = query(radiusBbox);
    
    // 精确距离过滤
    for (size_t featureId : candidates) {
        // 修复：需要找到featureId对应的边界框索引
        auto it = std::find(featureIds.begin(), featureIds.end(), featureId);
        if (it != featureIds.end()) {
            size_t bboxIndex = std::distance(featureIds.begin(), it);
            if (bboxIndex < boundingBoxes.size()) {
                const auto& bbox = boundingBoxes[bboxIndex];
                double distance = calculateDistanceToBBox(point, bbox);
                
                if (distance <= radius) {
                    resultSet.insert(featureId);
                }
            }
        }
    }
    
    return std::vector<size_t>(resultSet.begin(), resultSet.end());
}

IndexStats GridIndex::getStats() const { 
    return stats; 
}

void GridIndex::clear() {
    featureIds.clear();
    boundingBoxes.clear();
    featureIdToIndexMap.clear(); // 清理映射表
    grid.clear();
    stats = IndexStats{};
}

bool GridIndex::empty() const { 
    return featureIds.empty(); 
}

size_t GridIndex::size() const { 
    return featureIds.size(); 
}

void GridIndex::insert(size_t featureId, const BoundingBox& bbox) {
    size_t newIndex = featureIds.size();
    featureIds.push_back(featureId);
    boundingBoxes.push_back(bbox);
    featureIdToIndexMap[featureId] = newIndex; // 更新映射表
    
    // 修复：如果这是第一个要素，初始化世界边界和网格
    if (featureIds.size() == 1) {
        worldBounds = bbox;
        
        // 扩展边界以避免边界上的要素分配问题
        double padding = std::max((worldBounds.maxX - worldBounds.minX) * 0.1, 1.0);
        worldBounds.minX -= padding;
        worldBounds.minY -= padding;
        worldBounds.maxX += padding;
        worldBounds.maxY += padding;
        
        // 计算网格单元大小
        cellWidth = (worldBounds.maxX - worldBounds.minX) / gridWidth;
        cellHeight = (worldBounds.maxY - worldBounds.minY) / gridHeight;
        
        // 确保单元大小有效
        if (cellWidth <= 0.0) cellWidth = 1.0;
        if (cellHeight <= 0.0) cellHeight = 1.0;
        
        // 初始化网格
        grid.clear();
        grid.resize(gridHeight);
        for (auto& row : grid) {
            row.resize(gridWidth);
        }
    } else {
        // 检查是否需要扩展世界边界
        bool needsExpansion = false;
        BoundingBox newBounds = worldBounds;
        
        if (bbox.minX < worldBounds.minX) {
            newBounds.minX = bbox.minX - (worldBounds.maxX - worldBounds.minX) * 0.1;
            needsExpansion = true;
        }
        if (bbox.minY < worldBounds.minY) {
            newBounds.minY = bbox.minY - (worldBounds.maxY - worldBounds.minY) * 0.1;
            needsExpansion = true;
        }
        if (bbox.maxX > worldBounds.maxX) {
            newBounds.maxX = bbox.maxX + (worldBounds.maxX - worldBounds.minX) * 0.1;
            needsExpansion = true;
        }
        if (bbox.maxY > worldBounds.maxY) {
            newBounds.maxY = bbox.maxY + (worldBounds.maxY - worldBounds.minY) * 0.1;
            needsExpansion = true;
        }
        
        if (needsExpansion) {
            // 需要重建网格
            worldBounds = newBounds;
            cellWidth = (worldBounds.maxX - worldBounds.minX) / gridWidth;
            cellHeight = (worldBounds.maxY - worldBounds.minY) / gridHeight;
            
            // 确保单元大小有效
            if (cellWidth <= 0.0) cellWidth = 1.0;
            if (cellHeight <= 0.0) cellHeight = 1.0;
            
            // 重新初始化网格
            grid.clear();
            grid.resize(gridHeight);
            for (auto& row : grid) {
                row.resize(gridWidth);
            }
            
            // 重新插入所有要素
            for (size_t i = 0; i < featureIds.size(); ++i) {
                insertFeatureToGrid(featureIds[i], boundingBoxes[i]);
            }
        } else {
            // 只插入当前要素
            insertFeatureToGrid(featureId, bbox);
        }
    }
    
    // 如果是第一个要素，也需要插入到网格中
    if (featureIds.size() == 1) {
        insertFeatureToGrid(featureId, bbox);
    }
    
    stats.totalFeatures = featureIds.size();
    stats.nodeCount = gridWidth * gridHeight;
}

void GridIndex::remove(size_t featureId) {
    auto it = std::find(featureIds.begin(), featureIds.end(), featureId);
    if (it != featureIds.end()) {
        size_t index = std::distance(featureIds.begin(), it);
        const auto& bbox = boundingBoxes[index];
        
        // 从网格单元中移除要素
        int minCol = static_cast<int>((bbox.minX - worldBounds.minX) / cellWidth);
        int maxCol = static_cast<int>((bbox.maxX - worldBounds.minX) / cellWidth);
        int minRow = static_cast<int>((bbox.minY - worldBounds.minY) / cellHeight);
        int maxRow = static_cast<int>((bbox.maxY - worldBounds.minY) / cellHeight);
        
        // 边界检查
        minCol = std::max(0, std::min(minCol, static_cast<int>(gridWidth) - 1));
        maxCol = std::max(0, std::min(maxCol, static_cast<int>(gridWidth) - 1));
        minRow = std::max(0, std::min(minRow, static_cast<int>(gridHeight) - 1));
        maxRow = std::max(0, std::min(maxRow, static_cast<int>(gridHeight) - 1));
        
        for (int row = minRow; row <= maxRow; ++row) {
            for (int col = minCol; col <= maxCol; ++col) {
                if (row >= 0 && row < static_cast<int>(gridHeight) && 
                    col >= 0 && col < static_cast<int>(gridWidth)) {
                    grid[row][col].removeFeature(featureId);
                }
            }
        }
        
        // 从线性存储中移除
        featureIds.erase(it);
        boundingBoxes.erase(boundingBoxes.begin() + index);
        stats.nodeCount--;
    }
}

void GridIndex::update(size_t featureId, const BoundingBox& newBbox) {
    // 先删除旧的，再插入新的
    remove(featureId);
    insert(featureId, newBbox);
}

std::pair<size_t, size_t> GridIndex::getGridSize() const {
    return std::make_pair(gridWidth, gridHeight);
}

std::pair<double, double> GridIndex::getCellSize() const {
    return std::make_pair(cellWidth, cellHeight);
}

BoundingBox GridIndex::getGridBounds() const {
    return worldBounds;
}

double GridIndex::calculateDistanceToBBox(const Point& point, const BoundingBox& bbox) const {
    double dx = 0.0, dy = 0.0;
    if (point.x < bbox.minX) dx = bbox.minX - point.x;
    else if (point.x > bbox.maxX) dx = point.x - bbox.maxX;
    
    if (point.y < bbox.minY) dy = bbox.minY - point.y;
    else if (point.y > bbox.maxY) dy = point.y - bbox.maxY;
    
    return std::sqrt(dx * dx + dy * dy);
}

void GridIndex::insertFeatureToGrid(size_t featureId, const BoundingBox& bbox) {
    // 计算要素覆盖的网格单元范围
    int minCol = static_cast<int>((bbox.minX - worldBounds.minX) / cellWidth);
    int maxCol = static_cast<int>((bbox.maxX - worldBounds.minX) / cellWidth);
    int minRow = static_cast<int>((bbox.minY - worldBounds.minY) / cellHeight);
    int maxRow = static_cast<int>((bbox.maxY - worldBounds.minY) / cellHeight);
    
    // 严格的边界检查
    minCol = std::max(0, std::min(minCol, static_cast<int>(gridWidth) - 1));
    maxCol = std::max(0, std::min(maxCol, static_cast<int>(gridWidth) - 1));
    minRow = std::max(0, std::min(minRow, static_cast<int>(gridHeight) - 1));
    maxRow = std::max(0, std::min(maxRow, static_cast<int>(gridHeight) - 1));
    
    // 将要素添加到所有相交的网格单元
    for (int row = minRow; row <= maxRow; ++row) {
        for (int col = minCol; col <= maxCol; ++col) {
            // 双重检查边界
            if (row >= 0 && row < static_cast<int>(gridHeight) && 
                col >= 0 && col < static_cast<int>(gridWidth) &&
                static_cast<size_t>(row) < grid.size() &&
                static_cast<size_t>(col) < grid[row].size()) {
                grid[row][col].addFeature(featureId);
            }
        }
    }
}

} // namespace oscean::core_services::spatial_ops::index 