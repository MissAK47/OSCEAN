#include "spatial_index_manager.h"
#include "../index/r_tree_index.h"
#include "../index/quad_tree_index.h"
#include "../index/grid_index.h"
#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include "../utils/geometry_converter.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <set>

using namespace oscean::core_services;
using namespace oscean::core_services::spatial_ops;
using namespace oscean::core_services::spatial_ops::utils;

namespace oscean::core_services::spatial_ops::engine {

// === SpatialIndexManager 实现 ===

/**
 * @brief SpatialIndexManager的私有实现
 */
class SpatialIndexManager::Impl {
public:
    explicit Impl(const SpatialOpsConfig& config) 
        : config_(config), nextTempId_(0) {
        resetPerformanceStats();
    }

    // 索引存储
    std::unordered_map<std::string, std::shared_ptr<ISpatialIndex>> persistentIndexes_;
    std::vector<std::shared_ptr<ISpatialIndex>> temporaryIndexes_;
    
    // 配置和统计
    SpatialOpsConfig config_;
    QueryPerformanceStats performanceStats_;
    size_t cacheSize_ = 256; // MB
    size_t nextTempId_;
    
    // 线程安全
    mutable std::mutex indexMutex_;
    mutable std::mutex statsMutex_;

    void updatePerformanceStats(double queryTime) {
        std::lock_guard<std::mutex> lock(statsMutex_);
        performanceStats_.totalQueries++;
        performanceStats_.totalQueryTime += queryTime;
        performanceStats_.averageQueryTime = performanceStats_.totalQueryTime / performanceStats_.totalQueries;
        
        if (performanceStats_.totalQueries == 1) {
            performanceStats_.minQueryTime = queryTime;
            performanceStats_.maxQueryTime = queryTime;
        } else {
            performanceStats_.minQueryTime = std::min(performanceStats_.minQueryTime, queryTime);
            performanceStats_.maxQueryTime = std::max(performanceStats_.maxQueryTime, queryTime);
        }
    }

    void resetPerformanceStats() {
        std::lock_guard<std::mutex> lock(statsMutex_);
        performanceStats_ = QueryPerformanceStats{};
    }
};

SpatialIndexManager::SpatialIndexManager(const SpatialOpsConfig& config)
    : pImpl_(std::make_unique<Impl>(config)) {
}

SpatialIndexManager::~SpatialIndexManager() = default;

std::shared_ptr<ISpatialIndex> SpatialIndexManager::createIndex(
    const oscean::core_services::FeatureCollection& features,
    IndexType indexType,
    const std::string& indexName) {
    
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    
    // 检查是否已存在同名索引
    if (pImpl_->persistentIndexes_.find(indexName) != pImpl_->persistentIndexes_.end()) {
        throw InvalidParameterException("indexName", "Index with name '" + indexName + "' already exists");
    }
    
    // 创建索引
    auto index = createIndexInternal(features, indexType);
    if (!index) {
        throw OperationFailedException("createIndex", "Failed to create index of type " + indexTypeToString(indexType));
    }
    
    // 存储索引
    pImpl_->persistentIndexes_[indexName] = index;
    
    return index;
}

std::shared_ptr<ISpatialIndex> SpatialIndexManager::createTemporaryIndex(
    const oscean::core_services::FeatureCollection& features,
    IndexType indexType) {
    
    auto index = createIndexInternal(features, indexType);
    if (!index) {
        throw OperationFailedException("createTemporaryIndex", "Failed to create temporary index");
    }
    
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    pImpl_->temporaryIndexes_.push_back(index);
    
    return index;
}

std::shared_ptr<ISpatialIndex> SpatialIndexManager::getIndex(const std::string& indexName) {
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    
    auto it = pImpl_->persistentIndexes_.find(indexName);
    if (it != pImpl_->persistentIndexes_.end()) {
        return it->second;
    }
    
    return nullptr;
}

bool SpatialIndexManager::removeIndex(const std::string& indexName) {
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    
    auto it = pImpl_->persistentIndexes_.find(indexName);
    if (it != pImpl_->persistentIndexes_.end()) {
        pImpl_->persistentIndexes_.erase(it);
        return true;
    }
    
    return false;
}

IndexType SpatialIndexManager::selectOptimalIndexType(
    const oscean::core_services::FeatureCollection& features,
    const std::string& expectedQueryType) {
    
    size_t featureCount = features.getFeatures().size();
    
    // 基于要素数量和查询类型的启发式选择
    if (featureCount < 100) {
        return IndexType::GRID; // 小数据集使用网格索引
    } else if (featureCount < 10000) {
        if (expectedQueryType == "point" || expectedQueryType == "nearest") {
            return IndexType::QUADTREE; // 点查询优化
        } else {
            return IndexType::RTREE; // 通用查询
        }
    } else {
        return IndexType::RTREE; // 大数据集使用R-tree
    }
}

bool SpatialIndexManager::rebuildIndex(
    const std::string& indexName,
    const oscean::core_services::FeatureCollection& features) {
    
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    
    auto it = pImpl_->persistentIndexes_.find(indexName);
    if (it == pImpl_->persistentIndexes_.end()) {
        return false;
    }
    
    IndexType indexType = it->second->getType();
    auto newIndex = createIndexInternal(features, indexType);
    if (!newIndex) {
        return false;
    }
    
    it->second = newIndex;
    return true;
}

bool SpatialIndexManager::optimizeIndex(const std::string& indexName) {
    // 索引优化的具体实现取决于索引类型
    // 这里提供基础框架
    auto index = getIndex(indexName);
    if (!index) {
        return false;
    }
    
    // 可以实现重新平衡、压缩等优化操作
    // 具体实现留给各个索引类型
    return true;
}

std::map<std::string, IndexStats> SpatialIndexManager::getAllIndexStats() const {
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    
    std::map<std::string, IndexStats> allStats;
    
    for (const auto& pair : pImpl_->persistentIndexes_) {
        allStats[pair.first] = pair.second->getStats();
    }
    
    return allStats;
}

QueryPerformanceStats SpatialIndexManager::getQueryPerformanceStats() const {
    std::lock_guard<std::mutex> lock(pImpl_->statsMutex_);
    return pImpl_->performanceStats_;
}

void SpatialIndexManager::resetPerformanceStats() {
    pImpl_->resetPerformanceStats();
}

void SpatialIndexManager::setCacheSize(size_t cacheSize) {
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    pImpl_->cacheSize_ = cacheSize;
}

void SpatialIndexManager::clearCache() {
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    pImpl_->temporaryIndexes_.clear();
}

std::vector<IndexType> SpatialIndexManager::getSupportedIndexTypes() {
    return {IndexType::RTREE, IndexType::QUADTREE, IndexType::GRID};
}

std::string SpatialIndexManager::indexTypeToString(IndexType type) {
    switch (type) {
        case IndexType::RTREE: return "R-tree";
        case IndexType::QUADTREE: return "QuadTree";
        case IndexType::GRID: return "Grid";
        case IndexType::AUTO: return "Auto";
        default: return "Unknown";
    }
}

IndexType SpatialIndexManager::stringToIndexType(const std::string& typeStr) {
    if (typeStr == "R-tree" || typeStr == "rtree") return IndexType::RTREE;
    if (typeStr == "QuadTree" || typeStr == "quadtree") return IndexType::QUADTREE;
    if (typeStr == "Grid" || typeStr == "grid") return IndexType::GRID;
    if (typeStr == "Auto" || typeStr == "auto") return IndexType::AUTO;
    
    throw InvalidParameterException("typeStr", "Unknown index type: " + typeStr);
}

// 私有辅助方法
std::shared_ptr<ISpatialIndex> SpatialIndexManager::createIndexInternal(
    const oscean::core_services::FeatureCollection& features,
    IndexType indexType) {
    
    auto startTime = std::chrono::high_resolution_clock::now();

    std::shared_ptr<ISpatialIndex> index;
    switch (indexType) {
        case IndexType::RTREE:
            {
                // 直接创建R-tree索引
                auto rtreeIndex = std::make_shared<index::RTreeIndex>(16, 4);
                rtreeIndex->build(features);
                index = rtreeIndex;
            }
            break;
        case IndexType::QUADTREE:
            {
                // 暂时使用R-tree实现，直到真正的四叉树实现完成
                auto rtreeIndex = std::make_shared<index::RTreeIndex>(8, 2);
                rtreeIndex->build(features);
                index = rtreeIndex;
            }
            break;
        case IndexType::GRID:
            {
                // 暂时使用R-tree实现，直到真正的网格索引实现完成
                auto rtreeIndex = std::make_shared<index::RTreeIndex>(32, 8);
                rtreeIndex->build(features);
                index = rtreeIndex;
            }
            break;
        case IndexType::AUTO:
            {
                IndexType selectedType = selectOptimalIndexType(features);
                // 避免递归调用AUTO
                if (selectedType == IndexType::AUTO) selectedType = IndexType::RTREE; 
                return createIndexInternal(features, selectedType);
            }
        default:
            throw InvalidParameterException("indexType", "Unsupported index type");
    }

    if (!index) {
        throw OperationFailedException("createIndexInternal", "Failed to instantiate index");
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> buildTimeMs = endTime - startTime;
    
    pImpl_->updatePerformanceStats(buildTimeMs.count()); // 记录构建时间作为一次"查询"
    return index;
}

SpatialIndexManager::CacheStats SpatialIndexManager::getCacheStats() const {
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    CacheStats stats;
    // 实现缓存统计逻辑，例如遍历缓存中的索引
    // stats.cachedIndexCount = ...;
    // stats.totalMemoryUsage = ...;
    // stats.cacheHits = pImpl_->performanceStats_.cacheHits;
    // stats.cacheMisses = pImpl_->performanceStats_.cacheMisses;
    return stats;
}

void SpatialIndexManager::setMaxCacheSize(size_t maxSize) {
    std::lock_guard<std::mutex> lock(pImpl_->indexMutex_);
    pImpl_->cacheSize_ = maxSize; // 假设cacheSize_是以字节为单位
    // 可能需要实现LRU或其他缓存淘汰策略来适应新的大小
}


// === IndexBuilder 实现 ===
// IndexBuilder的静态方法通常不需要在这个cpp文件中定义，除非它们非常特定于SpatialIndexManager
// 通常，IndexBuilder会有自己的.cpp文件，或者其方法在各自的索引类实现中。
// 如果IndexBuilder的方法仅仅是工厂方法调用具体索引类的构造函数，它们可以保留在头文件中（如果简单）
// 或在IndexBuilder.cpp中。这里为了演示，假设它们在头文件中或已被处理。
// 例如，IndexBuilder::createSharedRTreeIndex() 等的实现在其他地方

} // namespace oscean::core_services::spatial_ops::engine 