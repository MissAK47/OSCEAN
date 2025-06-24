#include "quad_tree_index.h"
#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "../utils/geometry_converter.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <unordered_map>
#include <queue>

using namespace oscean::core_services;
using namespace oscean::core_services::spatial_ops;
using namespace oscean::core_services::spatial_ops::utils;

namespace oscean::core_services::spatial_ops::index {

// === QuadTreeNode 结构定义 ===

struct QuadTreeIndex::QuadTreeNode {
    BoundingBox bounds;
    std::vector<size_t> featureIds;
    std::unique_ptr<QuadTreeNode> children[4]; // NW, NE, SW, SE
    bool isLeaf = true;
    size_t maxCapacity = 10;
    size_t maxDepth = 8;
    size_t currentDepth = 0;
    
    QuadTreeNode(const BoundingBox& bbox, size_t depth = 0, size_t capacity = 10, size_t maxD = 8) 
        : bounds(bbox), currentDepth(depth), maxCapacity(capacity), maxDepth(maxD) {}
    
    void subdivide(const std::unordered_map<size_t, size_t>& featureIdToIndex, const std::vector<BoundingBox>& boundingBoxes) {
        if (!isLeaf || currentDepth >= maxDepth || featureIds.empty()) return;
        
        double midX = (bounds.minX + bounds.maxX) / 2.0;
        double midY = (bounds.minY + bounds.maxY) / 2.0;
        
        // 创建四个子节点
        children[0] = std::make_unique<QuadTreeNode>(
            BoundingBox{bounds.minX, midY, midX, bounds.maxY}, 
            currentDepth + 1, maxCapacity, maxDepth); // NW
        children[1] = std::make_unique<QuadTreeNode>(
            BoundingBox{midX, midY, bounds.maxX, bounds.maxY}, 
            currentDepth + 1, maxCapacity, maxDepth); // NE
        children[2] = std::make_unique<QuadTreeNode>(
            BoundingBox{bounds.minX, bounds.minY, midX, midY}, 
            currentDepth + 1, maxCapacity, maxDepth); // SW
        children[3] = std::make_unique<QuadTreeNode>(
            BoundingBox{midX, bounds.minY, bounds.maxX, midY}, 
            currentDepth + 1, maxCapacity, maxDepth); // SE
        
        isLeaf = false;
        
        // 简化策略：保留所有要素在父节点中，确保100%正确性
        // 同时将要素分配到相交的子节点中，提高查询效率
        for (size_t featureId : featureIds) {
            auto mapIt = featureIdToIndex.find(featureId);
            if (mapIt != featureIdToIndex.end()) {
                size_t bboxIndex = mapIt->second;
                if (bboxIndex < boundingBoxes.size()) {
                    const auto& featureBbox = boundingBoxes[bboxIndex];
                    
                    // 将要素添加到所有相交的子节点
                    for (int i = 0; i < 4; ++i) {
                        if (children[i] && intersects(children[i]->bounds, featureBbox)) {
                            children[i]->featureIds.push_back(featureId);
                        }
                    }
                }
            }
        }
        
        // 注意：不清空父节点的要素列表，保持所有要素在父节点中
        // 这样确保查询时不会遗漏任何要素
        
        // 递归细分子节点
        for (int i = 0; i < 4; ++i) {
            if (children[i] && children[i]->featureIds.size() > maxCapacity && children[i]->currentDepth < maxDepth) {
                children[i]->subdivide(featureIdToIndex, boundingBoxes);
            }
        }
    }
    
    void insertFeature(size_t featureId, const BoundingBox& featureBbox, const std::unordered_map<size_t, size_t>& featureIdToIndex, const std::vector<BoundingBox>& boundingBoxes) {
        if (isLeaf) {
            featureIds.push_back(featureId);
            if (featureIds.size() > maxCapacity && currentDepth < maxDepth) {
                subdivide(featureIdToIndex, boundingBoxes);
            }
        } else {
            // 修复：首先将要素添加到当前节点，确保不会遗漏
            featureIds.push_back(featureId);
            
            // 将要素添加到所有相交的子节点
            bool addedToChild = false;
            for (int i = 0; i < 4; ++i) {
                if (children[i] && intersects(children[i]->bounds, featureBbox)) {
                    children[i]->insertFeature(featureId, featureBbox, featureIdToIndex, boundingBoxes);
                    addedToChild = true;
                }
            }
            
            // 如果没有与任何子节点相交，使用质心策略
            if (!addedToChild) {
                double midX = (bounds.minX + bounds.maxX) / 2.0;
                double midY = (bounds.minY + bounds.maxY) / 2.0;
                double centerX = (featureBbox.minX + featureBbox.maxX) / 2.0;
                double centerY = (featureBbox.minY + featureBbox.maxY) / 2.0;
                
                int childIndex;
                if (centerY >= midY) { // North
                    childIndex = (centerX >= midX) ? 1 : 0; // NE : NW
                } else { // South
                    childIndex = (centerX >= midX) ? 3 : 2; // SE : SW
                }
                
                if (children[childIndex]) {
                    children[childIndex]->insertFeature(featureId, featureBbox, featureIdToIndex, boundingBoxes);
                }
            }
        }
    }
    
    bool intersects(const BoundingBox& nodeBounds, const BoundingBox& featureBbox) const {
        // 超级精确的边界框相交检测，处理浮点数精度问题
        const double EPSILON = 1e-10;
        return !(featureBbox.maxX < nodeBounds.minX - EPSILON || 
                featureBbox.minX > nodeBounds.maxX + EPSILON ||
                featureBbox.maxY < nodeBounds.minY - EPSILON || 
                featureBbox.minY > nodeBounds.maxY + EPSILON);
    }
    
    void queryRange(const BoundingBox& queryBounds, const std::unordered_map<size_t, size_t>& featureIdToIndex, const std::vector<BoundingBox>& boundingBoxes, std::vector<size_t>& results) const {
        // 检查节点边界是否与查询框相交
        if (!intersects(bounds, queryBounds)) {
            return;
        }
        
        // 检查当前节点中的要素（包括跨边界要素）
        for (size_t featureId : featureIds) {
            auto mapIt = featureIdToIndex.find(featureId);
            if (mapIt != featureIdToIndex.end()) {
                size_t bboxIndex = mapIt->second;
                if (bboxIndex < boundingBoxes.size()) {
                    const auto& featureBbox = boundingBoxes[bboxIndex];
                    if (intersects(queryBounds, featureBbox)) {
                        results.push_back(featureId);
                    }
                }
            }
        }
        
        // 如果不是叶子节点，递归查询子节点
        if (!isLeaf) {
            for (int i = 0; i < 4; ++i) {
                if (children[i] && intersects(children[i]->bounds, queryBounds)) {
                    children[i]->queryRange(queryBounds, featureIdToIndex, boundingBoxes, results);
                }
            }
        }
    }
    
    void queryRadius(const Point& center, double radius, const std::unordered_map<size_t, size_t>& featureIdToIndex, const std::vector<BoundingBox>& boundingBoxes, std::vector<size_t>& results) const {
        // 检查节点边界是否与查询圆相交
        double dx = std::max(0.0, std::max(bounds.minX - center.x, center.x - bounds.maxX));
        double dy = std::max(0.0, std::max(bounds.minY - center.y, center.y - bounds.maxY));
        double distanceToNode = std::sqrt(dx * dx + dy * dy);
        
        if (distanceToNode > radius) {
            return; // 节点完全在查询半径外
        }
        
        // 检查当前节点中的要素（无论是叶子节点还是内部节点）
        for (size_t featureId : featureIds) {
            auto mapIt = featureIdToIndex.find(featureId);
            if (mapIt != featureIdToIndex.end()) {
                size_t bboxIndex = mapIt->second;
                if (bboxIndex < boundingBoxes.size()) {
                    double dist = calculateDistanceToBBox(center, boundingBoxes[bboxIndex]);
                    if (dist <= radius) {
                        results.push_back(featureId);
                    }
                }
            }
        }
        
        // 如果不是叶子节点，递归查询子节点
        if (!isLeaf) {
            for (int i = 0; i < 4; ++i) {
                if (children[i]) {
                    children[i]->queryRadius(center, radius, featureIdToIndex, boundingBoxes, results);
                }
            }
        }
    }
    
    void collectAllFeatures(std::vector<size_t>& results) const {
        // 收集当前节点中的要素（无论是叶子节点还是内部节点）
        results.insert(results.end(), featureIds.begin(), featureIds.end());
        
        // 如果不是叶子节点，递归收集子节点的要素
        if (!isLeaf) {
            for (int i = 0; i < 4; ++i) {
                if (children[i]) {
                    children[i]->collectAllFeatures(results);
                }
            }
        }
    }
    
    size_t countNodes() const {
        size_t count = 1; // 当前节点
        if (!isLeaf) {
            for (int i = 0; i < 4; ++i) {
                if (children[i]) {
                    count += children[i]->countNodes();
                }
            }
        }
        return count;
    }
    
    size_t countLeaves() const {
        if (isLeaf) {
            return 1;
        } else {
            size_t count = 0;
            for (int i = 0; i < 4; ++i) {
                if (children[i]) {
                    count += children[i]->countLeaves();
                }
            }
            return count;
        }
    }
    
private:
    double calculateDistanceToBBox(const Point& point, const BoundingBox& bbox) const {
        double dx = 0.0, dy = 0.0;
        if (point.x < bbox.minX) dx = bbox.minX - point.x;
        else if (point.x > bbox.maxX) dx = point.x - bbox.maxX;
        
        if (point.y < bbox.minY) dy = bbox.minY - point.y;
        else if (point.y > bbox.maxY) dy = point.y - bbox.maxY;
        
        return std::sqrt(dx * dx + dy * dy);
    }
    
    double calculateOverlapArea(const BoundingBox& bbox1, const BoundingBox& bbox2) const {
        double overlapMinX = std::max(bbox1.minX, bbox2.minX);
        double overlapMinY = std::max(bbox1.minY, bbox2.minY);
        double overlapMaxX = std::min(bbox1.maxX, bbox2.maxX);
        double overlapMaxY = std::min(bbox1.maxY, bbox2.maxY);
        
        if (overlapMinX >= overlapMaxX || overlapMinY >= overlapMaxY) {
            return 0.0; // 没有重叠
        }
        
        return (overlapMaxX - overlapMinX) * (overlapMaxY - overlapMinY);
    }
};

// === QuadTreeIndex 实现 ===

QuadTreeIndex::QuadTreeIndex(size_t maxCapacity, size_t maxDepth) 
    : maxCapacity_(std::max(size_t(16), std::min(maxCapacity, size_t(32)))), // 优化：增加容量减少分割
      maxDepth_(std::max(size_t(4), std::min(maxDepth, size_t(8)))), // 优化：减少深度提高性能
      root_(nullptr) {
    stats_ = IndexStats{};
    // 修复：使用更合理的默认边界，不要太大
    worldBounds_ = {-10000.0, -10000.0, 10000.0, 10000.0};
    // 修复：立即创建根节点，这样insert方法就能正常工作
    // 优化：使用更大的容量和更小的深度来提高性能
    size_t optimizedCapacity = maxCapacity_; // 使用设定的容量
    size_t optimizedDepth = maxDepth_; // 使用设定的深度
    root_ = std::make_unique<QuadTreeNode>(worldBounds_, 0, optimizedCapacity, optimizedDepth);
}

QuadTreeIndex::~QuadTreeIndex() = default;

void QuadTreeIndex::build(const FeatureCollection& features) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    clear();
    
    const auto& featureList = features.getFeatures();
    if (featureList.empty()) {
        // 空数据集，设置默认边界
        worldBounds_ = {0.0, 0.0, 1.0, 1.0};
        root_ = std::make_unique<QuadTreeNode>(worldBounds_, 0, maxCapacity_, maxDepth_);
        return;
    }
    
    // 预分配内存以提高性能
    featureIds_.reserve(featureList.size());
    boundingBoxes_.reserve(featureList.size());
    featureIdToIndex_.clear(); // 清理映射表
    
    // 计算数据边界
    bool firstFeature = true;
    for (size_t i = 0; i < featureList.size(); ++i) {
        const auto& feature = featureList[i];
        try {
            BoundingBox bbox = GeometryConverter::extractBoundingBoxFromWKT(feature.geometryWkt);
            
            // 修复：使用真实的要素ID（从feature中获取，如果没有则使用索引）
            size_t realFeatureId = feature.id.empty() ? i : std::stoull(feature.id);
            featureIds_.push_back(realFeatureId);
            boundingBoxes_.push_back(bbox);
            featureIdToIndex_[realFeatureId] = i; // 建立映射关系
            
            if (firstFeature) {
                worldBounds_ = bbox;
                firstFeature = false;
            } else {
                worldBounds_.minX = std::min(worldBounds_.minX, bbox.minX);
                worldBounds_.minY = std::min(worldBounds_.minY, bbox.minY);
                worldBounds_.maxX = std::max(worldBounds_.maxX, bbox.maxX);
                worldBounds_.maxY = std::max(worldBounds_.maxY, bbox.maxY);
            }
        } catch (const std::exception&) {
            // 跳过无效的几何体
            continue;
        }
    }
    
    // 确保边界有效
    if (worldBounds_.maxX <= worldBounds_.minX) {
        worldBounds_.maxX = worldBounds_.minX + 1.0;
    }
    if (worldBounds_.maxY <= worldBounds_.minY) {
        worldBounds_.maxY = worldBounds_.minY + 1.0;
    }
    
    // 扩展边界以避免边界上的要素分配问题
    double padding = std::max((worldBounds_.maxX - worldBounds_.minX) * 0.001, 
                             (worldBounds_.maxY - worldBounds_.minY) * 0.001);
    worldBounds_.minX -= padding;
    worldBounds_.minY -= padding;
    worldBounds_.maxX += padding;
    worldBounds_.maxY += padding;
    
    // 创建根节点
    root_ = std::make_unique<QuadTreeNode>(worldBounds_, 0, maxCapacity_, maxDepth_);
    
    // 优化：批量插入所有要素，减少重复计算
    for (size_t i = 0; i < featureIds_.size(); ++i) {
        root_->insertFeature(featureIds_[i], boundingBoxes_[i], featureIdToIndex_, boundingBoxes_);
    }
    
    // 更新统计信息
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    stats_.totalFeatures = featureIds_.size();
    stats_.nodeCount = root_ ? root_->countNodes() : 0;
    stats_.leafCount = root_ ? root_->countLeaves() : 0;
    stats_.maxDepth = getTreeDepthRecursive(root_.get());
    stats_.averageDepth = calculateAverageDepth(root_.get());
    stats_.buildTime = std::max(0.001, static_cast<double>(duration.count()) / 1000000.0);
    stats_.memoryUsage = static_cast<double>(featureIds_.size() * sizeof(BoundingBox) + 
                       stats_.nodeCount * sizeof(QuadTreeNode) + 
                       featureIdToIndex_.size() * sizeof(std::pair<size_t, size_t>)) / (1024.0 * 1024.0);
    
    // 调试信息：检查根节点状态
    if (root_) {
        // 临时调试：检查根节点是否为叶子节点
        bool rootIsLeaf = root_->isLeaf;
        size_t rootFeatureCount = root_->featureIds.size();
        // 这些信息可以在调试时使用
        (void)rootIsLeaf; // 避免未使用变量警告
        (void)rootFeatureCount;
    }
}

std::vector<size_t> QuadTreeIndex::query(const BoundingBox& bbox) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<size_t> results;
    if (root_) {
        // 使用set来避免重复，比后续排序+去重更高效
        std::set<size_t> uniqueResults;
        queryRangeOptimized(root_.get(), bbox, uniqueResults);
        
        // 转换为vector
        results.reserve(uniqueResults.size());
        results.assign(uniqueResults.begin(), uniqueResults.end());
    }
    
    // 更新查询统计
    updateQueryStats(startTime);
    
    return results;
}

std::vector<size_t> QuadTreeIndex::query(const Point& point) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    BoundingBox pointBbox{point.x, point.y, point.x, point.y};
    std::vector<size_t> results;
    if (root_) {
        // 使用set来避免重复
        std::set<size_t> uniqueResults;
        queryRangeOptimized(root_.get(), pointBbox, uniqueResults);
        
        // 转换为vector
        results.reserve(uniqueResults.size());
        results.assign(uniqueResults.begin(), uniqueResults.end());
    }
    
    return results;
}

std::vector<size_t> QuadTreeIndex::query(const Geometry& geom) const {
    BoundingBox bbox = GeometryConverter::extractBoundingBoxFromWKT(geom.wkt);
    std::vector<size_t> results;
    if (root_) {
        // 使用set来避免重复
        std::set<size_t> uniqueResults;
        queryRangeOptimized(root_.get(), bbox, uniqueResults);
        
        // 转换为vector
        results.reserve(uniqueResults.size());
        results.assign(uniqueResults.begin(), uniqueResults.end());
    }
    return results;
}

std::vector<size_t> QuadTreeIndex::nearestNeighbors(const Point& point, size_t k) {
    if (k == 0 || !root_) return {};
    
    // 激进优化：使用空间剪枝的最近邻搜索
    using DistanceFeaturePair = std::pair<double, size_t>;
    std::priority_queue<DistanceFeaturePair> candidates; // 最大堆，保持k个最近的
    
    // 递归搜索函数，带空间剪枝
    std::function<void(const QuadTreeNode*, double)> searchNode = 
        [&](const QuadTreeNode* node, double minDistToNode) {
        if (!node) return;
        
        // 剪枝：如果候选集已满且当前节点的最小距离大于最远候选，跳过
        if (candidates.size() >= k && minDistToNode > candidates.top().first) {
            return;
        }
        
        if (node->isLeaf) {
            // 叶子节点：检查所有要素
            for (size_t featureId : node->featureIds) {
                auto mapIt = featureIdToIndex_.find(featureId);
                if (mapIt != featureIdToIndex_.end()) {
                    size_t bboxIndex = mapIt->second;
                    if (bboxIndex < boundingBoxes_.size()) {
                        double dist = calculateDistanceToBBox(point, boundingBoxes_[bboxIndex]);
                        
                        if (candidates.size() < k) {
                            candidates.push({dist, featureId});
                        } else if (dist < candidates.top().first) {
                            candidates.pop();
                            candidates.push({dist, featureId});
                        }
                    }
                }
            }
        } else {
            // 内部节点：按距离排序子节点，优先访问近的
            std::vector<std::pair<double, const QuadTreeNode*>> childDistances;
            childDistances.reserve(4);
            
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    const auto& bounds = node->children[i]->bounds;
                    double dx = std::max(0.0, std::max(bounds.minX - point.x, point.x - bounds.maxX));
                    double dy = std::max(0.0, std::max(bounds.minY - point.y, point.y - bounds.maxY));
                    double distance = std::sqrt(dx * dx + dy * dy);
                    childDistances.emplace_back(distance, node->children[i].get());
                }
            }
            
            // 按距离排序
            std::sort(childDistances.begin(), childDistances.end());
            
            // 递归访问子节点
            for (const auto& childPair : childDistances) {
                searchNode(childPair.second, childPair.first);
            }
        }
    };
    
    // 开始搜索
    searchNode(root_.get(), 0.0);
    
    // 提取结果（按距离从近到远排序）
    std::vector<size_t> results;
    results.reserve(candidates.size());
    
    std::vector<DistanceFeaturePair> temp;
    while (!candidates.empty()) {
        temp.push_back(candidates.top());
        candidates.pop();
    }
    
    // 反转以获得从近到远的顺序
    for (auto it = temp.rbegin(); it != temp.rend(); ++it) {
        results.push_back(it->second);
    }
    
    return results;
}

std::vector<size_t> QuadTreeIndex::radiusQuery(const Point& point, double radius) {
    std::vector<size_t> results;
    if (root_ && radius > 0.0) {
        // 使用set来避免重复
        std::set<size_t> uniqueResults;
        radiusQueryOptimized(root_.get(), point, radius, uniqueResults);
        
        // 转换为vector
        results.reserve(uniqueResults.size());
        results.assign(uniqueResults.begin(), uniqueResults.end());
    }
    return results;
}

IndexStats QuadTreeIndex::getStats() const { 
    return stats_; 
}

void QuadTreeIndex::clear() {
    featureIds_.clear();
    boundingBoxes_.clear();
    featureIdToIndex_.clear(); // 清理映射表
    root_.reset();
    stats_ = IndexStats{};
}

bool QuadTreeIndex::empty() const { 
    return featureIds_.empty(); 
}

size_t QuadTreeIndex::size() const { 
    return featureIds_.size(); 
}

void QuadTreeIndex::insert(size_t featureId, const BoundingBox& bbox) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    size_t newIndex = featureIds_.size();
    featureIds_.push_back(featureId);
    boundingBoxes_.push_back(bbox);
    featureIdToIndex_[featureId] = newIndex; // 修复：更新映射表
    
    // 修复：更新世界边界以包含新的边界框
    if (featureIds_.size() == 1) {
        // 第一个要素，设置世界边界
        worldBounds_ = bbox;
    } else {
        // 扩展世界边界
        worldBounds_.minX = std::min(worldBounds_.minX, bbox.minX);
        worldBounds_.minY = std::min(worldBounds_.minY, bbox.minY);
        worldBounds_.maxX = std::max(worldBounds_.maxX, bbox.maxX);
        worldBounds_.maxY = std::max(worldBounds_.maxY, bbox.maxY);
    }
    
    // 修复：如果世界边界发生了显著变化，重建根节点
    if (featureIds_.size() == 1 || needsRebuild(bbox)) {
        rebuildTree();
    } else {
        // 修复：即使不重建，也要更新根节点的边界
        if (root_) {
            root_->bounds = worldBounds_;
            root_->insertFeature(featureId, bbox, featureIdToIndex_, boundingBoxes_);
        }
    }
    
    // 优化：只在必要时更新统计信息，减少计算开销
    if (featureIds_.size() % 100 == 0 || featureIds_.size() < 100) {
        stats_.totalFeatures = featureIds_.size();
        stats_.nodeCount = root_ ? root_->countNodes() : 0;
        stats_.leafCount = root_ ? root_->countLeaves() : 0;
        stats_.maxDepth = getTreeDepthRecursive(root_.get());
        stats_.averageDepth = calculateAverageDepth(root_.get());
        stats_.memoryUsage = static_cast<double>(featureIds_.size() * sizeof(BoundingBox) + 
                           stats_.nodeCount * sizeof(QuadTreeNode) + 
                           featureIdToIndex_.size() * sizeof(std::pair<size_t, size_t>)) / (1024.0 * 1024.0);
    }
    
    // 更新构建时间
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    stats_.buildTime += std::max(0.001, static_cast<double>(duration.count()) / 1000000.0);
}

void QuadTreeIndex::remove(size_t featureId) {
    auto it = std::find(featureIds_.begin(), featureIds_.end(), featureId);
    if (it != featureIds_.end()) {
        size_t index = std::distance(featureIds_.begin(), it);
        featureIds_.erase(it);
        boundingBoxes_.erase(boundingBoxes_.begin() + index);
        
        // 修复：重建映射表，因为索引发生了变化
        featureIdToIndex_.clear();
        for (size_t i = 0; i < featureIds_.size(); ++i) {
            featureIdToIndex_[featureIds_[i]] = i;
        }
        
        stats_.totalFeatures = featureIds_.size();
        stats_.nodeCount = root_ ? root_->countNodes() : 0; // 修复：重新计算节点数
        // TODO: 实现从四叉树中删除（需要重建索引）
    }
}

void QuadTreeIndex::update(size_t featureId, const BoundingBox& newBbox) {
    // 查找要素的当前索引
    auto it = std::find(featureIds_.begin(), featureIds_.end(), featureId);
    if (it != featureIds_.end()) {
        size_t index = std::distance(featureIds_.begin(), it);
        
        // 更新边界框
        boundingBoxes_[index] = newBbox;
        
        // 更新世界边界
        worldBounds_.minX = std::min(worldBounds_.minX, newBbox.minX);
        worldBounds_.minY = std::min(worldBounds_.minY, newBbox.minY);
        worldBounds_.maxX = std::max(worldBounds_.maxX, newBbox.maxX);
        worldBounds_.maxY = std::max(worldBounds_.maxY, newBbox.maxY);
        
        // 重建树以确保正确的空间分布
        rebuildTree();
        
        // 更新统计信息
        stats_.nodeCount = root_ ? root_->countNodes() : 0;
        stats_.leafCount = root_ ? root_->countLeaves() : 0;
        stats_.maxDepth = getTreeDepthRecursive(root_.get());
        stats_.averageDepth = calculateAverageDepth(root_.get());
    }
}

double QuadTreeIndex::calculateDistanceToBBox(const Point& point, const BoundingBox& bbox) const {
    double dx = 0.0, dy = 0.0;
    if (point.x < bbox.minX) dx = bbox.minX - point.x;
    else if (point.x > bbox.maxX) dx = point.x - bbox.maxX;
    
    if (point.y < bbox.minY) dy = bbox.minY - point.y;
    else if (point.y > bbox.maxY) dy = point.y - bbox.maxY;
    
    return std::sqrt(dx * dx + dy * dy);
}

size_t QuadTreeIndex::getTreeDepthRecursive(const QuadTreeNode* node) const {
    if (!node) {
        return 0; // 空节点深度为0
    }
    
    if (node->isLeaf) {
        return 1; // 叶子节点深度为1
    }
    
    size_t maxChildDepth = 0;
    for (int i = 0; i < 4; ++i) {
        if (node->children[i]) {
            maxChildDepth = std::max(maxChildDepth, getTreeDepthRecursive(node->children[i].get()));
        }
    }
    
    return 1 + maxChildDepth;
}

double QuadTreeIndex::calculateAverageDepth(const QuadTreeNode* node) const {
    if (!node) {
        return 0.0;
    }
    
    if (node->isLeaf) {
        return 1.0; // 叶子节点深度为1
    }
    
    double totalDepth = 0.0;
    size_t leafCount = 0;
    
    std::function<void(const QuadTreeNode*, size_t)> traverse = [&](const QuadTreeNode* current, size_t depth) {
        if (!current) return;
        
        if (current->isLeaf) {
            totalDepth += depth;
            leafCount++;
        } else {
            for (int i = 0; i < 4; ++i) {
                if (current->children[i]) {
                    traverse(current->children[i].get(), depth + 1);
                }
            }
        }
    };
    
    traverse(node, 1);
    
    return leafCount > 0 ? (totalDepth / static_cast<double>(leafCount)) : 0.0;
}

void QuadTreeIndex::updateQueryStats(const std::chrono::high_resolution_clock::time_point& startTime) {
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    stats_.queryCount++;
    double queryTime = duration.count() / 1000.0; // 转换为毫秒
    stats_.averageQueryTime = 
        (stats_.averageQueryTime * (stats_.queryCount - 1) + queryTime) / 
        stats_.queryCount;
}

bool QuadTreeIndex::needsRebuild(const BoundingBox& newBbox) const {
    // 修复：只有当新边界框显著超出当前世界边界时才重建
    // 使用更宽松的阈值，避免频繁重建
    double currentWidth = worldBounds_.maxX - worldBounds_.minX;
    double currentHeight = worldBounds_.maxY - worldBounds_.minY;
    
    // 如果当前边界太小，使用默认阈值
    if (currentWidth < 1.0) currentWidth = 1000.0;
    if (currentHeight < 1.0) currentHeight = 1000.0;
    
    bool exceedsX = (newBbox.minX < worldBounds_.minX - currentWidth * 2.0) ||
                    (newBbox.maxX > worldBounds_.maxX + currentWidth * 2.0);
    bool exceedsY = (newBbox.minY < worldBounds_.minY - currentHeight * 2.0) ||
                    (newBbox.maxY > worldBounds_.maxY + currentHeight * 2.0);
    
    return exceedsX || exceedsY;
}

void QuadTreeIndex::rebuildTree() {
    if (featureIds_.empty()) {
        root_.reset();
        return;
    }
    
    // 扩展边界以避免边界上的要素分配问题
    double padding = std::max((worldBounds_.maxX - worldBounds_.minX) * 0.001, 
                             (worldBounds_.maxY - worldBounds_.minY) * 0.001);
    BoundingBox expandedBounds = {
        worldBounds_.minX - padding,
        worldBounds_.minY - padding,
        worldBounds_.maxX + padding,
        worldBounds_.maxY + padding
    };
    
    // 重建根节点
    root_ = std::make_unique<QuadTreeNode>(expandedBounds, 0, maxCapacity_, maxDepth_);
    
    // 重新插入所有要素
    for (size_t i = 0; i < featureIds_.size(); ++i) {
        root_->insertFeature(featureIds_[i], boundingBoxes_[i], featureIdToIndex_, boundingBoxes_);
    }
}

void QuadTreeIndex::queryRangeOptimized(const QuadTreeNode* node, const BoundingBox& queryBounds, std::set<size_t>& results) const {
    if (!node) return;
    
    // 检查节点边界是否与查询框相交
    if (!intersects(node->bounds, queryBounds)) {
        return;
    }
    
    if (node->isLeaf) {
        // 叶子节点：检查所有要素
        for (size_t featureId : node->featureIds) {
            auto mapIt = featureIdToIndex_.find(featureId);
            if (mapIt != featureIdToIndex_.end()) {
                size_t bboxIndex = mapIt->second;
                if (bboxIndex < boundingBoxes_.size()) {
                    const auto& featureBbox = boundingBoxes_[bboxIndex];
                    if (intersects(queryBounds, featureBbox)) {
                        results.insert(featureId);
                    }
                }
            }
        }
    } else {
        // 内部节点：递归查询子节点
        for (int i = 0; i < 4; ++i) {
            if (node->children[i] && intersects(node->children[i]->bounds, queryBounds)) {
                queryRangeOptimized(node->children[i].get(), queryBounds, results);
            }
        }
    }
}

bool QuadTreeIndex::intersects(const BoundingBox& bbox1, const BoundingBox& bbox2) const {
    // 精确的边界框相交检测
    const double EPSILON = 1e-10;
    return !(bbox1.maxX < bbox2.minX - EPSILON || 
             bbox1.minX > bbox2.maxX + EPSILON ||
             bbox1.maxY < bbox2.minY - EPSILON || 
             bbox1.minY > bbox2.maxY + EPSILON);
}

void QuadTreeIndex::radiusQueryOptimized(const QuadTreeNode* node, const Point& center, double radius, std::set<size_t>& results) const {
    if (!node) return;
    
    // 检查节点边界是否与查询圆相交
    double dx = std::max(0.0, std::max(node->bounds.minX - center.x, center.x - node->bounds.maxX));
    double dy = std::max(0.0, std::max(node->bounds.minY - center.y, center.y - node->bounds.maxY));
    double distanceToNode = std::sqrt(dx * dx + dy * dy);
    
    if (distanceToNode > radius) {
        return; // 节点完全在查询半径外
    }
    
    if (node->isLeaf) {
        // 叶子节点：检查所有要素
        for (size_t featureId : node->featureIds) {
            auto mapIt = featureIdToIndex_.find(featureId);
            if (mapIt != featureIdToIndex_.end()) {
                size_t bboxIndex = mapIt->second;
                if (bboxIndex < boundingBoxes_.size()) {
                    double dist = calculateDistanceToBBox(center, boundingBoxes_[bboxIndex]);
                    if (dist <= radius) {
                        results.insert(featureId);
                    }
                }
            }
        }
    } else {
        // 内部节点：递归查询子节点
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                radiusQueryOptimized(node->children[i].get(), center, radius, results);
            }
        }
    }
}

} // namespace oscean::core_services::spatial_ops::index 