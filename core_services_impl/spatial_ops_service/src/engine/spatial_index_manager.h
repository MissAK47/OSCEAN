#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/spatial_ops/spatial_exceptions.h"

#include <memory>
#include <vector>
#include <string>
#include <future>
#include <optional>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <map>
#include <queue>

namespace oscean::core_services::spatial_ops::engine {

/**
 * @brief 空间索引类型枚举
 */
enum class IndexType {
    RTREE,      // R-tree索引
    QUADTREE,   // 四叉树索引
    GRID,       // 网格索引
    AUTO        // 自动选择最优索引
};

/**
 * @brief 索引统计信息
 */
struct IndexStats {
    size_t nodeCount = 0;           // 节点数量
    size_t leafCount = 0;           // 叶子节点数量
    size_t maxDepth = 0;            // 最大深度
    double averageDepth = 0.0;      // 平均深度
    size_t totalFeatures = 0;       // 总要素数量
    double buildTime = 0.0;         // 构建时间（秒）
    double memoryUsage = 0.0;       // 内存使用（MB）
    size_t queryCount = 0;          // 查询次数
    double averageQueryTime = 0.0;  // 平均查询时间（毫秒）
};

/**
 * @brief 查询性能统计
 */
struct QueryPerformanceStats {
    size_t totalQueries = 0;        // 总查询次数
    double totalQueryTime = 0.0;    // 总查询时间（毫秒）
    double averageQueryTime = 0.0;  // 平均查询时间（毫秒）
    double minQueryTime = 0.0;      // 最小查询时间（毫秒）
    double maxQueryTime = 0.0;      // 最大查询时间（毫秒）
    size_t cacheHits = 0;           // 缓存命中次数
    size_t cacheMisses = 0;         // 缓存未命中次数
};

/**
 * @brief 空间索引接口
 */
class ISpatialIndex {
public:
    virtual ~ISpatialIndex() = default;

    /**
     * @brief 边界框查询
     * @param bbox 查询边界框
     * @return 相交的要素ID列表
     */
    virtual std::vector<size_t> query(const oscean::core_services::BoundingBox& bbox) = 0;

    /**
     * @brief 点查询
     * @param point 查询点
     * @return 包含该点的要素ID列表
     */
    virtual std::vector<size_t> query(const oscean::core_services::Point& point) = 0;

    /**
     * @brief 几何体查询
     * @param geom 查询几何体
     * @return 相交的要素ID列表
     */
    virtual std::vector<size_t> query(const oscean::core_services::Geometry& geom) const = 0;

    /**
     * @brief 半径查询
     * @param center 中心点
     * @param radius 查询半径
     * @return 半径内的要素ID列表
     */
    virtual std::vector<size_t> radiusQuery(const oscean::core_services::Point& center, double radius) = 0;

    /**
     * @brief K最近邻查询
     * @param point 查询点
     * @param k 邻居数量
     * @return K个最近要素的ID列表
     */
    virtual std::vector<size_t> nearestNeighbors(const oscean::core_services::Point& point, size_t k) = 0;

    /**
     * @brief 插入要素
     * @param featureId 要素ID
     * @param bbox 要素边界框
     */
    virtual void insert(size_t featureId, const oscean::core_services::BoundingBox& bbox) = 0;

    /**
     * @brief 删除要素
     * @param featureId 要素ID
     */
    virtual void remove(size_t featureId) = 0;

    /**
     * @brief 更新要素
     * @param featureId 要素ID
     * @param newBbox 新的边界框
     */
    virtual void update(size_t featureId, const oscean::core_services::BoundingBox& newBbox) = 0;

    /**
     * @brief 获取索引统计信息
     * @return 索引统计
     */
    virtual IndexStats getStats() const = 0;

    /**
     * @brief 获取索引类型
     * @return 索引类型
     */
    virtual IndexType getType() const = 0;

    /**
     * @brief 清空索引
     */
    virtual void clear() = 0;

    /**
     * @brief 检查索引是否为空
     * @return 如果为空返回true
     */
    virtual bool empty() const = 0;

    /**
     * @brief 获取索引中的要素数量
     * @return 要素数量
     */
    virtual size_t size() const = 0;
};

/**
 * @brief 空间索引管理器
 * 
 * 负责创建、管理和优化空间索引，提供统一的索引访问接口
 */
class SpatialIndexManager {
public:
    /**
     * @brief 构造函数
     * @param config 空间服务配置
     */
    explicit SpatialIndexManager(const SpatialOpsConfig& config);

    /**
     * @brief 析构函数
     */
    ~SpatialIndexManager();

    /**
     * @brief 创建持久化索引
     * @param features 要素集合
     * @param indexType 索引类型
     * @param indexName 索引名称
     * @return 索引指针
     */
    std::shared_ptr<ISpatialIndex> createIndex(
        const oscean::core_services::FeatureCollection& features,
        IndexType indexType,
        const std::string& indexName);

    /**
     * @brief 创建临时索引
     * @param features 要素集合
     * @param indexType 索引类型
     * @return 索引指针
     */
    std::shared_ptr<ISpatialIndex> createTemporaryIndex(
        const oscean::core_services::FeatureCollection& features,
        IndexType indexType);

    /**
     * @brief 获取已存在的索引
     * @param indexName 索引名称
     * @return 索引指针，如果不存在返回nullptr
     */
    std::shared_ptr<ISpatialIndex> getIndex(const std::string& indexName);

    /**
     * @brief 删除索引
     * @param indexName 索引名称
     * @return 删除成功返回true
     */
    bool removeIndex(const std::string& indexName);

    /**
     * @brief 自动选择最优索引类型
     * @param features 要素集合
     * @param expectedQueryType 预期查询类型
     * @return 推荐的索引类型
     */
    IndexType selectOptimalIndexType(
        const oscean::core_services::FeatureCollection& features,
        const std::string& expectedQueryType = "mixed");

    /**
     * @brief 重建索引
     * @param indexName 索引名称
     * @param features 新的要素集合
     * @return 重建成功返回true
     */
    bool rebuildIndex(
        const std::string& indexName,
        const oscean::core_services::FeatureCollection& features);

    /**
     * @brief 优化索引
     * @param indexName 索引名称
     * @return 优化成功返回true
     */
    bool optimizeIndex(const std::string& indexName);

    /**
     * @brief 获取所有索引的统计信息
     * @return 索引名称到统计信息的映射
     */
    std::map<std::string, IndexStats> getAllIndexStats() const;

    /**
     * @brief 获取查询性能统计
     * @return 查询性能统计
     */
    QueryPerformanceStats getQueryPerformanceStats() const;

    /**
     * @brief 重置性能统计
     */
    void resetPerformanceStats();

    /**
     * @brief 设置缓存大小
     * @param cacheSize 缓存大小（MB）
     */
    void setCacheSize(size_t cacheSize);

    /**
     * @brief 清理缓存
     */
    void clearCache();

    /**
     * @brief 缓存统计信息
     */
    struct CacheStats {
        size_t cacheHits = 0;
        size_t cacheMisses = 0;
        size_t cachedIndexCount = 0;
        size_t totalMemoryUsage = 0;
    };

    /**
     * @brief 获取或创建索引（带缓存）
     * @param features 要素集合
     * @param indexType 索引类型
     * @return 索引指针
     */
    std::shared_ptr<ISpatialIndex> getOrCreateIndex(
        const oscean::core_services::FeatureCollection& features,
        IndexType indexType);

    /**
     * @brief 获取缓存统计信息
     * @return 缓存统计
     */
    CacheStats getCacheStats() const;

    /**
     * @brief 设置最大缓存大小
     * @param maxSize 最大缓存大小（字节）
     */
    void setMaxCacheSize(size_t maxSize);

    /**
     * @brief 获取支持的索引类型列表
     * @return 支持的索引类型
     */
    static std::vector<IndexType> getSupportedIndexTypes();

    /**
     * @brief 获取索引类型的字符串表示
     * @param type 索引类型
     * @return 字符串表示
     */
    static std::string indexTypeToString(IndexType type);

    /**
     * @brief 从字符串解析索引类型
     * @param typeStr 类型字符串
     * @return 索引类型
     */
    static IndexType stringToIndexType(const std::string& typeStr);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;

    /**
     * @brief 内部索引创建方法
     * @param features 要素集合
     * @param indexType 索引类型
     * @return 创建的索引
     */
    std::shared_ptr<ISpatialIndex> createIndexInternal(
        const oscean::core_services::FeatureCollection& features,
        IndexType indexType);
};

/**
 * @brief 索引构建器
 * 
 * 负责具体的索引构建逻辑
 */
class IndexBuilder {
public:
    /**
     * @brief 构建R-tree索引
     * @param features 要素集合
     * @param config 配置
     * @return R-tree索引指针
     */
    static std::unique_ptr<ISpatialIndex> buildRTreeIndex(
        const oscean::core_services::FeatureCollection& features,
        const SpatialOpsConfig& config);

    // Helper methods
    static oscean::core_services::BoundingBox calculateBounds(
        const oscean::core_services::FeatureCollection& features);
    
    static std::map<std::string, double> analyzeFeatureDistribution(
        const oscean::core_services::FeatureCollection& features);
};

} // namespace oscean::core_services::spatial_ops::engine 