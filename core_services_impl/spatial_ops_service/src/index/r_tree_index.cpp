#include "r_tree_index.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "../utils/geometry_converter.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <queue>
#include <cmath>
#include <unordered_map>

using namespace oscean::core_services;
using namespace oscean::core_services::spatial_ops;

namespace oscean::core_services::spatial_ops::index {

// === RTreeNode 结构体定义 ===
struct RTreeNode {
    bool isLeaf = false;
    oscean::core_services::BoundingBox mbr;
    std::vector<size_t> featureIds;  // 叶子节点：存储要素ID
    std::vector<std::unique_ptr<RTreeNode>> children;  // 内部节点：存储子节点
    
    /**
     * @brief 更新节点的最小边界矩形
     */
    void updateMBR() {
        if (children.empty()) return;
        
        mbr = children[0]->mbr;
        for (size_t i = 1; i < children.size(); ++i) {
            const auto& childMbr = children[i]->mbr;
            mbr.minX = std::min(mbr.minX, childMbr.minX);
            mbr.minY = std::min(mbr.minY, childMbr.minY);
            mbr.maxX = std::max(mbr.maxX, childMbr.maxX);
            mbr.maxY = std::max(mbr.maxY, childMbr.maxY);
        }
    }
};

// === RTreeIndex::Impl 结构体定义 ===
struct RTreeIndex::Impl {
    size_t maxEntries = 16;
    size_t minEntries = 4;
    bool useParallelBuild = false;
    size_t numThreads = 1;
    bool needsRebuild = false; // 延迟构建标志
    
    // 真正的R-tree结构
    std::unique_ptr<RTreeNode> root;
    std::vector<size_t> featureIds;
    std::vector<BoundingBox> boundingBoxes;
    std::unordered_map<size_t, size_t> featureIdToIndex; // 优化：featureId到索引的映射
    IndexStats stats;
    
    // 递归查询方法
    void queryRecursive(const RTreeNode* node, const BoundingBox& queryBbox, 
                       std::vector<size_t>& results) const;
    
    // 递归K最近邻查询
    void nearestNeighborsRecursive(const RTreeNode* node, const Point& queryPoint,
                                  size_t k, std::priority_queue<std::pair<double, size_t>>& candidates) const;
    
    // 构建R-tree的递归方法
    std::unique_ptr<RTreeNode> buildRTreeRecursive(
        std::vector<std::pair<size_t, BoundingBox>>& entries, size_t depth);
        
    // 延迟构建方法
    void ensureBuilt() {
        if (needsRebuild && !featureIds.empty()) {
            std::vector<std::pair<size_t, BoundingBox>> entries;
            entries.reserve(featureIds.size());
            for (size_t i = 0; i < featureIds.size(); ++i) {
                entries.emplace_back(featureIds[i], boundingBoxes[i]);
            }
            root = buildRTreeRecursive(entries, 0);
            needsRebuild = false;
        }
    }
};

// === RTreeIndex 方法实现 ===

RTreeIndex::RTreeIndex(size_t maxEntries, size_t minEntries)
    : m_impl(std::make_unique<Impl>()) {
    // 优化：调整默认参数以获得更好的性能
    m_impl->maxEntries = std::max(size_t(4), std::min(maxEntries, size_t(8))); // 减小节点容量，增加树深度
    m_impl->minEntries = std::max(size_t(2), std::min(minEntries, m_impl->maxEntries / 2));
    m_impl->stats = IndexStats{};
}

RTreeIndex::~RTreeIndex() = default;

void RTreeIndex::build(const FeatureCollection& features) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        clear();
        
        const auto& featureList = features.getFeatures();
        
        if (featureList.empty()) {
            return;
        }
        
        // 准备要素数据
        std::vector<std::pair<size_t, BoundingBox>> entries;
        entries.reserve(featureList.size());
        
        // 清理并重建映射表
        m_impl->featureIdToIndex.clear();
        
        for (size_t i = 0; i < featureList.size(); ++i) {
            // 从WKT中提取真实的边界框
            BoundingBox bbox = utils::GeometryConverter::extractBoundingBoxFromWKT(featureList[i].geometryWkt);
            entries.emplace_back(i, bbox);
            
            // 同时保存到线性存储中（用于快速访问）
            m_impl->featureIds.push_back(i);
            m_impl->boundingBoxes.push_back(bbox);
            m_impl->featureIdToIndex[i] = i; // 初始化映射表
        }
        
        // 构建真正的R-tree结构
        m_impl->root = m_impl->buildRTreeRecursive(entries, 0);
        
        // 更新统计信息
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        
        m_impl->stats.totalFeatures = featureList.size();
        m_impl->stats.nodeCount = m_impl->root ? countNodesRecursive(m_impl->root.get()) : 0;
        m_impl->stats.leafCount = m_impl->root ? countLeavesRecursive(m_impl->root.get()) : 0;
        m_impl->stats.maxDepth = calculateTreeDepth(m_impl->root.get());
        m_impl->stats.buildTime = std::max(0.001, static_cast<double>(duration.count()) / 1000000.0);
        m_impl->stats.memoryUsage = featureList.size() * sizeof(BoundingBox);
        
    } catch (const std::exception& e) {
        throw SpatialOpsException("R-tree索引构建失败: " + std::string(e.what()));
    }
}

std::vector<size_t> RTreeIndex::query(const BoundingBox& bbox) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 延迟构建：只在查询时构建索引
    m_impl->ensureBuilt();
    
    std::vector<size_t> results;
    
    // 使用真正的R-tree递归查询
    if (m_impl->root) {
        m_impl->queryRecursive(m_impl->root.get(), bbox, results);
    }
    
    // 更新查询统计
    updateQueryStats(startTime);
    
    return results;
}

std::vector<size_t> RTreeIndex::query(const Point& point) {
    BoundingBox pointBbox;
    pointBbox.minX = pointBbox.maxX = point.x;
    pointBbox.minY = pointBbox.maxY = point.y;
    return query(pointBbox);
}

std::vector<size_t> RTreeIndex::query(const Geometry& geom) const {
    // 从WKT中提取边界框
    BoundingBox bbox = utils::GeometryConverter::extractBoundingBoxFromWKT(geom.wkt);
    
    // 实现const版本的递归查询
    std::vector<size_t> results;
    if (m_impl->root) {
        m_impl->queryRecursive(m_impl->root.get(), bbox, results);
    }
    
    return results;
}

IndexStats RTreeIndex::getStats() const {
    return m_impl->stats;
}

void RTreeIndex::clear() {
    m_impl->featureIds.clear();
    m_impl->boundingBoxes.clear();
    m_impl->featureIdToIndex.clear(); // 清理映射表
    m_impl->root.reset();
    m_impl->stats = IndexStats{};
}

bool RTreeIndex::empty() const {
    return m_impl->featureIds.empty();
}

size_t RTreeIndex::size() const {
    return m_impl->featureIds.size();
}

void RTreeIndex::setParallelBuild(bool useParallel, size_t numThreads) {
    m_impl->useParallelBuild = useParallel;
    m_impl->numThreads = (numThreads == 0) ? std::thread::hardware_concurrency() : numThreads;
}

std::vector<size_t> RTreeIndex::nearestNeighbors(const Point& point, size_t k) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 延迟构建：只在查询时构建索引
    m_impl->ensureBuilt();
    
    // 使用优先队列存储距离和要素ID
    using DistanceFeaturePair = std::pair<double, size_t>;
    std::priority_queue<DistanceFeaturePair> candidates;
    
    // 使用真正的R-tree递归K-NN查询
    if (m_impl->root) {
        m_impl->nearestNeighborsRecursive(m_impl->root.get(), point, k, candidates);
    }
    
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
    
    // 更新查询统计
    updateQueryStats(startTime);
    
    return results;
}

std::vector<size_t> RTreeIndex::radiusQuery(const Point& point, double radius) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 确保索引已完成构建
    if (!m_impl->root && !m_impl->featureIds.empty()) {
        finalizeBuild();
    }
    
    std::vector<size_t> results;
    
    // 创建半径查询的边界框
    BoundingBox radiusBbox;
    radiusBbox.minX = point.x - radius;
    radiusBbox.minY = point.y - radius;
    radiusBbox.maxX = point.x + radius;
    radiusBbox.maxY = point.y + radius;
    
    // 先用边界框查询获取候选要素
    std::vector<size_t> candidates;
    if (m_impl->root) {
        m_impl->queryRecursive(m_impl->root.get(), radiusBbox, candidates);
    }
    
    // 精确距离过滤
    for (size_t featureId : candidates) {
        // 修复：通过featureId找到对应的边界框索引
        auto it = std::find(m_impl->featureIds.begin(), m_impl->featureIds.end(), featureId);
        if (it != m_impl->featureIds.end()) {
            size_t bboxIndex = std::distance(m_impl->featureIds.begin(), it);
            if (bboxIndex < m_impl->boundingBoxes.size()) {
                const auto& bbox = m_impl->boundingBoxes[bboxIndex];
                double distance = calculateDistanceToBBox(point, bbox);
                
                if (distance <= radius) {
                    results.push_back(featureId);
                }
            }
        }
    }
    
    // 更新查询统计
    updateQueryStats(startTime);
    
    return results;
}

void RTreeIndex::insert(size_t featureId, const BoundingBox& bbox) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 检查是否已存在相同的featureId
    auto mapIt = m_impl->featureIdToIndex.find(featureId);
    if (mapIt != m_impl->featureIdToIndex.end()) {
        // 已存在，更新边界框
        size_t index = mapIt->second;
        m_impl->boundingBoxes[index] = bbox;
        
        // 标记需要重建，但不立即重建
        m_impl->needsRebuild = true;
    } else {
        // 不存在，添加新要素
        size_t newIndex = m_impl->featureIds.size();
        m_impl->featureIds.push_back(featureId);
        m_impl->boundingBoxes.push_back(bbox);
        m_impl->featureIdToIndex[featureId] = newIndex; // 维护映射
        
        // 优化：更激进的延迟构建策略
        size_t currentSize = m_impl->featureIds.size();
        bool shouldRebuild = false;
        
        if (currentSize <= 1000) {
            // 小数据集：每1000个重建一次
            shouldRebuild = (currentSize % 1000 == 0);
        } else if (currentSize <= 10000) {
            // 中等数据集：每5000个重建一次
            shouldRebuild = (currentSize % 5000 == 0);
        } else {
            // 大数据集：每10000个重建一次
            shouldRebuild = (currentSize % 10000 == 0);
        }
        
        if (shouldRebuild) {
            std::vector<std::pair<size_t, BoundingBox>> entries;
            entries.reserve(m_impl->featureIds.size());
            for (size_t i = 0; i < m_impl->featureIds.size(); ++i) {
                entries.emplace_back(m_impl->featureIds[i], m_impl->boundingBoxes[i]);
            }
            m_impl->root = m_impl->buildRTreeRecursive(entries, 0);
            m_impl->needsRebuild = false;
        } else {
            // 标记需要重建，但延迟到查询时
            m_impl->needsRebuild = true;
        }
    }
    
    // 轻量级统计更新
    m_impl->stats.totalFeatures = m_impl->featureIds.size();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    m_impl->stats.buildTime = std::max(0.001, static_cast<double>(duration.count()) / 1000000.0);
}

void RTreeIndex::remove(size_t featureId) {
    // 使用映射表快速查找
    auto mapIt = m_impl->featureIdToIndex.find(featureId);
    if (mapIt != m_impl->featureIdToIndex.end()) {
        size_t index = mapIt->second;
        
        // 移除要素
        m_impl->featureIds.erase(m_impl->featureIds.begin() + index);
        m_impl->boundingBoxes.erase(m_impl->boundingBoxes.begin() + index);
        m_impl->featureIdToIndex.erase(mapIt);
        
        // 重建映射表（因为索引发生了变化）
        m_impl->featureIdToIndex.clear();
        for (size_t i = 0; i < m_impl->featureIds.size(); ++i) {
            m_impl->featureIdToIndex[m_impl->featureIds[i]] = i;
        }
        
        // 更新统计信息
        m_impl->stats.totalFeatures = m_impl->featureIds.size();
        
        // 重建树
        std::vector<std::pair<size_t, BoundingBox>> entries;
        for (size_t i = 0; i < m_impl->featureIds.size(); ++i) {
            entries.emplace_back(m_impl->featureIds[i], m_impl->boundingBoxes[i]);
        }
        if (!entries.empty()) {
            m_impl->root = m_impl->buildRTreeRecursive(entries, 0);
        } else {
            m_impl->root.reset();
        }
    }
}

void RTreeIndex::update(size_t featureId, const BoundingBox& newBbox) {
    // 使用映射表快速查找并更新要素的边界框
    auto mapIt = m_impl->featureIdToIndex.find(featureId);
    if (mapIt != m_impl->featureIdToIndex.end()) {
        size_t index = mapIt->second;
        m_impl->boundingBoxes[index] = newBbox;
        
        // 重建树
        std::vector<std::pair<size_t, BoundingBox>> entries;
        for (size_t i = 0; i < m_impl->featureIds.size(); ++i) {
            entries.emplace_back(m_impl->featureIds[i], m_impl->boundingBoxes[i]);
        }
        m_impl->root = m_impl->buildRTreeRecursive(entries, 0);
    }
}

void RTreeIndex::finalizeBuild() {
    if (m_impl->featureIds.empty()) return;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 强制重建整个树结构
    std::vector<std::pair<size_t, BoundingBox>> entries;
    entries.reserve(m_impl->featureIds.size());
    for (size_t i = 0; i < m_impl->featureIds.size(); ++i) {
        entries.emplace_back(m_impl->featureIds[i], m_impl->boundingBoxes[i]);
    }
    m_impl->root = m_impl->buildRTreeRecursive(entries, 0);
    
    // 重新计算统计信息
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    m_impl->stats.nodeCount = m_impl->root ? countNodesRecursive(m_impl->root.get()) : 0;
    m_impl->stats.leafCount = m_impl->root ? countLeavesRecursive(m_impl->root.get()) : 0;
    m_impl->stats.maxDepth = calculateTreeDepth(m_impl->root.get());
    m_impl->stats.averageDepth = m_impl->root ? calculateAverageDepth(m_impl->root.get()) : 0.0;
    m_impl->stats.buildTime = std::max(0.001, static_cast<double>(duration.count()) / 1000000.0);
}

// === 私有辅助方法 ===

double RTreeIndex::calculateDistanceToBBox(const Point& point, const BoundingBox& bbox) const {
    // 计算点到边界框的最小距离
    double dx = 0.0;
    double dy = 0.0;
    
    if (point.x < bbox.minX) {
        dx = bbox.minX - point.x;
    } else if (point.x > bbox.maxX) {
        dx = point.x - bbox.maxX;
    }
    
    if (point.y < bbox.minY) {
        dy = bbox.minY - point.y;
    } else if (point.y > bbox.maxY) {
        dy = point.y - bbox.maxY;
    }
    
    return std::sqrt(dx * dx + dy * dy);
}

void RTreeIndex::updateQueryStats(const std::chrono::high_resolution_clock::time_point& startTime) {
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    m_impl->stats.queryCount++;
    double queryTime = duration.count() / 1000.0; // 转换为毫秒
    m_impl->stats.averageQueryTime = 
        (m_impl->stats.averageQueryTime * (m_impl->stats.queryCount - 1) + queryTime) / 
        m_impl->stats.queryCount;
}

size_t RTreeIndex::calculateTreeDepth(const RTreeNode* node) const {
    if (!node) return 0;
    
    if (node->isLeaf) {
        return 1;
    }
    
    size_t maxChildDepth = 0;
    for (const auto& child : node->children) {
        maxChildDepth = std::max(maxChildDepth, calculateTreeDepth(child.get()));
    }
    
    return 1 + maxChildDepth;
}

size_t RTreeIndex::countNodesRecursive(const RTreeNode* node) const {
    if (!node) return 0;
    
    size_t count = 1; // 当前节点
    for (const auto& child : node->children) {
        count += countNodesRecursive(child.get());
    }
    
    return count;
}

size_t RTreeIndex::countLeavesRecursive(const RTreeNode* node) const {
    if (!node) return 0;
    
    if (node->isLeaf) {
        return 1;
    }
    
    size_t count = 0;
    for (const auto& child : node->children) {
        count += countLeavesRecursive(child.get());
    }
    
    return count;
}

double RTreeIndex::calculateAverageDepth(const RTreeNode* node) const {
    if (!node) return 0.0;
    
    // 计算所有叶子节点的深度总和和叶子节点数量
    size_t totalDepth = 0;
    size_t leafCount = 0;
    calculateDepthSum(node, 1, totalDepth, leafCount);
    
    return leafCount > 0 ? static_cast<double>(totalDepth) / leafCount : 0.0;
}

void RTreeIndex::calculateDepthSum(const RTreeNode* node, size_t currentDepth, 
                                  size_t& totalDepth, size_t& leafCount) const {
    if (!node) return;
    
    if (node->isLeaf) {
        totalDepth += currentDepth;
        leafCount++;
    } else {
        for (const auto& child : node->children) {
            calculateDepthSum(child.get(), currentDepth + 1, totalDepth, leafCount);
        }
    }
}

// === Impl类的递归方法实现 ===

std::unique_ptr<RTreeNode> RTreeIndex::Impl::buildRTreeRecursive(
    std::vector<std::pair<size_t, BoundingBox>>& entries, size_t depth) {
    
    auto node = std::make_unique<RTreeNode>();
    
    if (entries.size() <= maxEntries) {
        // 创建叶子节点
        node->isLeaf = true;
        node->featureIds.reserve(entries.size()); // 预分配内存
        for (const auto& entry : entries) {
            node->featureIds.push_back(entry.first);
        }
        
        // 优化：更高效的MBR计算
        if (!entries.empty()) {
            node->mbr = entries[0].second;
            for (size_t i = 1; i < entries.size(); ++i) {
                const auto& bbox = entries[i].second;
                // 使用更高效的min/max操作
                if (bbox.minX < node->mbr.minX) node->mbr.minX = bbox.minX;
                if (bbox.minY < node->mbr.minY) node->mbr.minY = bbox.minY;
                if (bbox.maxX > node->mbr.maxX) node->mbr.maxX = bbox.maxX;
                if (bbox.maxY > node->mbr.maxY) node->mbr.maxY = bbox.maxY;
            }
        }
    } else {
        // 创建内部节点：优化分割策略
        node->isLeaf = false;
        
        // 优化：使用更智能的分割策略 - STR (Sort-Tile-Recursive)
        size_t numGroups = (entries.size() + maxEntries - 1) / maxEntries;
        size_t sliceSize = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(numGroups))));
        
        // 第一步：按X坐标排序并分片
        std::sort(entries.begin(), entries.end(), 
                 [](const auto& a, const auto& b) {
                     double centerA = (a.second.minX + a.second.maxX) / 2.0;
                     double centerB = (b.second.minX + b.second.maxX) / 2.0;
                     return centerA < centerB;
                 });
        
        std::vector<std::vector<std::pair<size_t, BoundingBox>>> slices;
        for (size_t i = 0; i < entries.size(); i += sliceSize * maxEntries) {
            size_t sliceEnd = std::min(i + sliceSize * maxEntries, entries.size());
            std::vector<std::pair<size_t, BoundingBox>> slice(
                entries.begin() + i, entries.begin() + sliceEnd);
            slices.push_back(std::move(slice));
        }
        
        // 第二步：在每个片内按Y坐标排序并分组
        node->children.reserve(numGroups);
        
        for (auto& slice : slices) {
            std::sort(slice.begin(), slice.end(), 
                     [](const auto& a, const auto& b) {
                         double centerA = (a.second.minY + a.second.maxY) / 2.0;
                         double centerB = (b.second.minY + b.second.maxY) / 2.0;
                         return centerA < centerB;
                     });
            
            // 将片分成多个组
            for (size_t i = 0; i < slice.size(); i += maxEntries) {
                size_t groupEnd = std::min(i + maxEntries, slice.size());
                std::vector<std::pair<size_t, BoundingBox>> group(
                    slice.begin() + i, slice.begin() + groupEnd);
                
                auto child = buildRTreeRecursive(group, depth + 1);
                node->children.push_back(std::move(child));
            }
        }
        
        // 优化：更高效的内部节点MBR更新
        if (!node->children.empty()) {
            node->mbr = node->children[0]->mbr;
            for (size_t i = 1; i < node->children.size(); ++i) {
                const auto& childMbr = node->children[i]->mbr;
                if (childMbr.minX < node->mbr.minX) node->mbr.minX = childMbr.minX;
                if (childMbr.minY < node->mbr.minY) node->mbr.minY = childMbr.minY;
                if (childMbr.maxX > node->mbr.maxX) node->mbr.maxX = childMbr.maxX;
                if (childMbr.maxY > node->mbr.maxY) node->mbr.maxY = childMbr.maxY;
            }
        }
    }
    
    return node;
}

void RTreeIndex::Impl::queryRecursive(const RTreeNode* node, const BoundingBox& queryBbox, 
                                     std::vector<size_t>& results) const {
    if (!node) return;
    
    // 优化：内联的高效边界框相交检测
    if (node->mbr.maxX < queryBbox.minX || node->mbr.minX > queryBbox.maxX ||
        node->mbr.maxY < queryBbox.minY || node->mbr.minY > queryBbox.maxY) {
        return; // 不相交，跳过
    }
    
    if (node->isLeaf) {
        // 叶子节点：优化的要素检查
        const size_t currentSize = results.size();
        results.reserve(currentSize + node->featureIds.size()); // 精确预分配
        
        // 优化：直接使用要素ID作为索引，避免映射表查找
        for (const auto featureId : node->featureIds) {
            // 优化：假设featureId就是boundingBoxes的索引（这是我们构建时的约定）
            if (featureId < boundingBoxes.size()) {
                const auto& featureBbox = boundingBoxes[featureId];
                // 优化：内联相交检测，避免函数调用
                if (!(featureBbox.maxX < queryBbox.minX || featureBbox.minX > queryBbox.maxX ||
                      featureBbox.maxY < queryBbox.minY || featureBbox.minY > queryBbox.maxY)) {
                    results.push_back(featureId);
                }
            }
        }
    } else {
        // 内部节点：优化的子节点遍历
        // 优化：使用基于范围的循环减少开销
        for (const auto& child : node->children) {
            queryRecursive(child.get(), queryBbox, results);
        }
    }
}

void RTreeIndex::Impl::nearestNeighborsRecursive(const RTreeNode* node, const Point& queryPoint,
                                                size_t k, std::priority_queue<std::pair<double, size_t>>& candidates) const {
    if (!node) return;
    
    if (node->isLeaf) {
        // 叶子节点：计算到每个要素的距离
        for (size_t featureId : node->featureIds) {
            // 优化：直接使用要素ID作为索引，避免映射表查找
            if (featureId < boundingBoxes.size()) {
                const auto& bbox = boundingBoxes[featureId];
                
                // 计算点到边界框的距离
                double dx = 0.0, dy = 0.0;
                if (queryPoint.x < bbox.minX) dx = bbox.minX - queryPoint.x;
                else if (queryPoint.x > bbox.maxX) dx = queryPoint.x - bbox.maxX;
                
                if (queryPoint.y < bbox.minY) dy = bbox.minY - queryPoint.y;
                else if (queryPoint.y > bbox.maxY) dy = queryPoint.y - bbox.maxY;
                
                double distance = std::sqrt(dx * dx + dy * dy);
                
                if (candidates.size() < k) {
                    candidates.push(std::make_pair(distance, featureId));
                } else if (distance < candidates.top().first) {
                    candidates.pop();
                    candidates.push(std::make_pair(distance, featureId));
                }
            }
        }
    } else {
        // 内部节点：按距离排序子节点，优先访问近的
        std::vector<std::pair<double, const RTreeNode*>> childDistances;
        childDistances.reserve(node->children.size()); // 预分配内存
        
        for (const auto& child : node->children) {
            // 计算查询点到子节点MBR的距离
            const auto& mbr = child->mbr;
            double dx = 0.0, dy = 0.0;
            if (queryPoint.x < mbr.minX) dx = mbr.minX - queryPoint.x;
            else if (queryPoint.x > mbr.maxX) dx = queryPoint.x - mbr.maxX;
            
            if (queryPoint.y < mbr.minY) dy = mbr.minY - queryPoint.y;
            else if (queryPoint.y > mbr.maxY) dy = queryPoint.y - mbr.maxY;
            
            double distance = std::sqrt(dx * dx + dy * dy);
            childDistances.emplace_back(distance, child.get());
        }
        
        // 按距离排序
        std::sort(childDistances.begin(), childDistances.end());
        
        // 递归访问子节点
        for (const auto& childPair : childDistances) {
            // 优化：如果候选集已满且当前子节点距离大于最远候选，可以剪枝
            if (candidates.size() >= k && childPair.first > candidates.top().first) {
                break;
            }
            nearestNeighborsRecursive(childPair.second, queryPoint, k, candidates);
        }
    }
}

} // namespace oscean::core_services::spatial_ops::index 