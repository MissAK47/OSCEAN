/**
 * @file cache_strategies.h
 * @brief 内部缓存策略实现 - 仅供内部使用，外部请使用CommonServicesFactory
 * @author OSCEAN Team
 * @date 2024
 * 
 * ⚠️  重要提示：
 * 这是内部实现文件，外部代码不应直接使用！
 * 
 * 外部代码应该使用CommonServicesFactory的统一接口：
 * #include "common_utils/infrastructure/common_services_factory.h"
 * 
 * auto factory = CommonServicesFactory::create();
 * auto lruCache = factory->createCache<K,V>("cache1", 1000, "LRU");
 * auto ttlCache = factory->createCache<K,V>("cache2", 500, "TTL");
 * 
 * 支持的策略：LRU, LFU, TTL, FIFO, Adaptive
 */

#pragma once

// 引用统一的boost配置 - 必须在最前面
#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include "icache_manager.h"
#include "cache_config.h"
#include <unordered_map>
#include <list>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <algorithm>
#include <atomic>
#include <queue>
#include <map>
#include <memory>

namespace oscean::common_utils::cache {

/**
 * @brief LRU缓存策略实现 - 支持boost::future异步操作
 * 
 * 使用双向链表+哈希表实现O(1)的插入、删除、访问操作
 * 全面支持异步操作，符合OSCEAN项目架构标准
 */
template<typename Key, typename Value>
class LRUCacheStrategy : public ICacheManager<Key, Value> {
private:
    struct Node {
        Key key;
        Value value;
        std::chrono::steady_clock::time_point accessTime;
        std::chrono::steady_clock::time_point insertTime;
        std::shared_ptr<Node> prev;
        std::shared_ptr<Node> next;
        
        Node(const Key& k, const Value& v) 
            : key(k), value(v)
            , accessTime(std::chrono::steady_clock::now())
            , insertTime(std::chrono::steady_clock::now()) {}
    };
    
    using NodePtr = std::shared_ptr<Node>;
    
    std::unordered_map<Key, NodePtr> hashTable_;
    NodePtr head_;
    NodePtr tail_;
    size_t capacity_;
    mutable std::shared_mutex rwMutex_;
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    
    // 统计信息
    mutable std::atomic<size_t> hitCount_{0};
    mutable std::atomic<size_t> missCount_{0};
    mutable std::atomic<size_t> evictionCount_{0};
    std::chrono::steady_clock::time_point creationTime_;

public:
    explicit LRUCacheStrategy(size_t capacity = 1000, 
                            std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);
    
    // === 同步接口实现 ===
    
    std::optional<Value> get(const Key& key) override;
    bool put(const Key& key, const Value& value) override;
    bool remove(const Key& key) override;
    bool containsKey(const Key& key) const override;
    void clear() override;
    size_t size() const override;
    
    // === 🔴 补充缺失的ICacheManager接口方法声明 ===
    bool contains(const Key& key) const override;
    void putBatch(const std::map<Key, Value>& items) override;
    std::map<Key, Value> getBatch(const std::vector<Key>& keys) override;
    void removeBatch(const std::vector<Key>& keys) override;
    size_t capacity() const override;
    void setCapacity(size_t newCapacity) override;
    void evictExpired() override;
    void optimize() override;
    CacheStatistics getStatistics() const override;
    void resetStatistics() override;
    std::string generateReport() const override;
    void updateConfig(const CacheConfig& config) override;
    CacheConfig getConfig() const override;
    CacheStrategy getStrategy() const override;
    
    // === 异步接口实现 ===
    
    boost::future<std::optional<Value>> getAsync(const Key& key) override;
    boost::future<bool> putAsync(const Key& key, const Value& value) override;
    boost::future<bool> removeAsync(const Key& key) override;
    
    boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
    getBatchAsync(const std::vector<Key>& keys) override;
    
    boost::future<std::vector<bool>> 
    putBatchAsync(const std::vector<std::pair<Key, Value>>& items) override;
    
    boost::future<CacheStatistics> getStatisticsAsync() override;
    boost::future<void> clearAsync() override;
    
    boost::future<size_t> 
    warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) override;
    
    boost::future<std::optional<Value>> 
    refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) override;
    
    boost::future<Value> 
    computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) override;
    
    // === LRU专用方法 ===
    
    /**
     * @brief 异步获取最近访问的N个项目
     */
    boost::future<std::vector<std::pair<Key, Value>>> getMostRecentAsync(size_t count);
    
    /**
     * @brief 异步获取最少访问的N个项目
     */
    boost::future<std::vector<std::pair<Key, Value>>> getLeastRecentAsync(size_t count);
    
    /**
     * @brief 异步优化缓存 - 清理过期项目
     */
    boost::future<size_t> optimizeAsync();

private:
    void moveToFront(NodePtr node);
    void removeNode(NodePtr node);
    void addToFront(NodePtr node);
    NodePtr removeTail();
    void initializeList();
    
    // 异步辅助方法
    template<typename Func>
    auto executeAsync(Func&& func) -> boost::future<decltype(func())>;
};

/**
 * @brief LFU缓存策略实现 - 支持boost::future异步操作
 * 
 * 基于访问频率的缓存淘汰策略，支持异步操作
 */
template<typename Key, typename Value>
class LFUCacheStrategy : public ICacheManager<Key, Value> {
private:
    struct LFUNode {
        Key key;
        Value value;
        size_t frequency;
        std::chrono::steady_clock::time_point lastAccess;
        
        // 默认构造函数
        LFUNode() = default;
        
        LFUNode(const Key& k, const Value& v) 
            : key(k), value(v), frequency(1)
            , lastAccess(std::chrono::steady_clock::now()) {}
    };
    
    std::unordered_map<Key, LFUNode> cache_;
    std::unordered_map<size_t, std::list<Key>> frequencyLists_;
    std::unordered_map<Key, typename std::list<Key>::iterator> keyToIterator_;
    size_t capacity_;
    size_t minFrequency_;
    mutable std::shared_mutex rwMutex_;
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    
    // 统计信息
    mutable std::atomic<size_t> hitCount_{0};
    mutable std::atomic<size_t> missCount_{0};
    mutable std::atomic<size_t> evictionCount_{0};

public:
    explicit LFUCacheStrategy(size_t capacity = 1000,
                            std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);
    
    // === 同步接口实现 ===
    std::optional<Value> get(const Key& key) override;
    bool put(const Key& key, const Value& value) override;
    bool remove(const Key& key) override;
    bool containsKey(const Key& key) const override;
    void clear() override;
    size_t size() const override;
    
    // === 补充缺失的ICacheManager接口方法 ===
    bool contains(const Key& key) const override;
    void putBatch(const std::map<Key, Value>& items) override;
    std::map<Key, Value> getBatch(const std::vector<Key>& keys) override;
    void removeBatch(const std::vector<Key>& keys) override;
    size_t capacity() const override;
    void setCapacity(size_t newCapacity) override;
    void evictExpired() override;
    void optimize() override;
    CacheStatistics getStatistics() const override;
    void resetStatistics() override;
    std::string generateReport() const override;
    void updateConfig(const CacheConfig& config) override;
    CacheConfig getConfig() const override;
    CacheStrategy getStrategy() const override;
    
    // === 异步接口实现 ===
    boost::future<std::optional<Value>> getAsync(const Key& key) override;
    boost::future<bool> putAsync(const Key& key, const Value& value) override;
    boost::future<bool> removeAsync(const Key& key) override;
    
    boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
    getBatchAsync(const std::vector<Key>& keys) override;
    
    boost::future<std::vector<bool>> 
    putBatchAsync(const std::vector<std::pair<Key, Value>>& items) override;
    
    boost::future<CacheStatistics> getStatisticsAsync() override;
    boost::future<void> clearAsync() override;
    
    boost::future<size_t> 
    warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) override;
    
    boost::future<std::optional<Value>> 
    refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) override;
    
    boost::future<Value> 
    computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) override;

private:
    void updateFrequency(const Key& key);
    void evictLFU();
    
    template<typename Func>
    auto executeAsync(Func&& func) -> boost::future<decltype(func())>;
};

/**
 * @brief FIFO缓存策略实现 - 支持boost::future异步操作
 */
template<typename Key, typename Value>
class FIFOCacheStrategy : public ICacheManager<Key, Value> {
private:
    std::unordered_map<Key, Value> cache_;
    std::queue<Key> insertionOrder_;
    size_t capacity_;
    mutable std::shared_mutex rwMutex_;
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    
    // 统计信息
    mutable std::atomic<size_t> hitCount_{0};
    mutable std::atomic<size_t> missCount_{0};
    mutable std::atomic<size_t> evictionCount_{0};

public:
    explicit FIFOCacheStrategy(size_t capacity = 1000,
                             std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);
    
    // === 同步接口实现 ===
    std::optional<Value> get(const Key& key) override;
    bool put(const Key& key, const Value& value) override;
    bool remove(const Key& key) override;
    bool containsKey(const Key& key) const override;
    void clear() override;
    size_t size() const override;
    
    // === 补充缺失的ICacheManager接口方法 ===
    bool contains(const Key& key) const override;
    void putBatch(const std::map<Key, Value>& items) override;
    std::map<Key, Value> getBatch(const std::vector<Key>& keys) override;
    void removeBatch(const std::vector<Key>& keys) override;
    size_t capacity() const override;
    void setCapacity(size_t newCapacity) override;
    void evictExpired() override;
    void optimize() override;
    CacheStatistics getStatistics() const override;
    void resetStatistics() override;
    std::string generateReport() const override;
    void updateConfig(const CacheConfig& config) override;
    CacheConfig getConfig() const override;
    CacheStrategy getStrategy() const override;
    
    // === 异步接口实现 ===
    boost::future<std::optional<Value>> getAsync(const Key& key) override;
    boost::future<bool> putAsync(const Key& key, const Value& value) override;
    boost::future<bool> removeAsync(const Key& key) override;
    
    boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
    getBatchAsync(const std::vector<Key>& keys) override;
    
    boost::future<std::vector<bool>> 
    putBatchAsync(const std::vector<std::pair<Key, Value>>& items) override;
    
    boost::future<CacheStatistics> getStatisticsAsync() override;
    boost::future<void> clearAsync() override;
    
    boost::future<size_t> 
    warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) override;
    
    boost::future<std::optional<Value>> 
    refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) override;
    
    boost::future<Value> 
    computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) override;

private:
    template<typename Func>
    auto executeAsync(Func&& func) -> boost::future<decltype(func())>;
};

/**
 * @brief TTL缓存策略实现 - 支持boost::future异步操作
 * 
 * 基于生存时间的缓存策略，支持自动过期和异步清理
 */
template<typename Key, typename Value>
class TTLCacheStrategy : public ICacheManager<Key, Value> {
private:
    struct TTLNode {
        Value value;
        std::chrono::steady_clock::time_point expiryTime;
        
        TTLNode(const Value& v, std::chrono::seconds ttl)
            : value(v), expiryTime(std::chrono::steady_clock::now() + ttl) {}
        
        bool isExpired() const {
            return std::chrono::steady_clock::now() >= expiryTime;
        }
    };
    
    std::unordered_map<Key, TTLNode> cache_;
    std::chrono::seconds defaultTTL_;
    mutable std::shared_mutex rwMutex_;
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    
    // 自动清理
    std::atomic<bool> autoCleanupEnabled_{true};
    std::chrono::seconds cleanupInterval_{60};
    boost::future<void> cleanupTask_;
    
    // 统计信息
    mutable std::atomic<size_t> hitCount_{0};
    mutable std::atomic<size_t> missCount_{0};
    mutable std::atomic<size_t> expirationCount_{0};

public:
    explicit TTLCacheStrategy(std::chrono::seconds defaultTTL = std::chrono::minutes(30),
                            std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);
    
    ~TTLCacheStrategy();
    
    // === 同步接口实现 ===
    std::optional<Value> get(const Key& key) override;
    bool put(const Key& key, const Value& value) override;
    bool remove(const Key& key) override;
    bool containsKey(const Key& key) const override;
    void clear() override;
    size_t size() const override;
    
    // === 🔴 补充缺失的ICacheManager接口方法声明 ===
    bool contains(const Key& key) const override;
    void putBatch(const std::map<Key, Value>& items) override;
    std::map<Key, Value> getBatch(const std::vector<Key>& keys) override;
    void removeBatch(const std::vector<Key>& keys) override;
    size_t capacity() const override;
    void setCapacity(size_t newCapacity) override;
    void evictExpired() override;
    void optimize() override;
    CacheStatistics getStatistics() const override;
    void resetStatistics() override;
    std::string generateReport() const override;
    void updateConfig(const CacheConfig& config) override;
    CacheConfig getConfig() const override;
    CacheStrategy getStrategy() const override;
    
    // === 异步接口实现 ===
    boost::future<std::optional<Value>> getAsync(const Key& key) override;
    boost::future<bool> putAsync(const Key& key, const Value& value) override;
    boost::future<bool> removeAsync(const Key& key) override;
    
    boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
    getBatchAsync(const std::vector<Key>& keys) override;
    
    boost::future<std::vector<bool>> 
    putBatchAsync(const std::vector<std::pair<Key, Value>>& items) override;
    
    boost::future<CacheStatistics> getStatisticsAsync() override;
    boost::future<void> clearAsync() override;
    
    boost::future<size_t> 
    warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) override;
    
    boost::future<std::optional<Value>> 
    refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) override;
    
    boost::future<Value> 
    computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) override;
    
    // === TTL专用方法 ===
    
    /**
     * @brief 设置项目的自定义TTL
     */
    bool putWithTTL(const Key& key, const Value& value, std::chrono::seconds ttl);
    
    /**
     * @brief 异步设置项目的自定义TTL
     */
    boost::future<bool> putWithTTLAsync(const Key& key, const Value& value, std::chrono::seconds ttl);
    
    /**
     * @brief 异步清理过期项目
     * @return boost::future<size_t> 清理的项目数
     */
    boost::future<size_t> cleanupExpiredAsync();
    
    /**
     * @brief 异步获取即将过期的项目
     */
    boost::future<std::vector<Key>> getExpiringKeysAsync(std::chrono::seconds withinTime);

private:
    void startAutoCleanup();
    void stopAutoCleanup();
    void cleanupExpiredItems();
    
    template<typename Func>
    auto executeAsync(Func&& func) -> boost::future<decltype(func())>;
};

/**
 * @brief 自适应缓存策略 - 支持boost::future异步操作
 * 
 * 根据访问模式自动选择最佳缓存策略，支持动态切换
 */
template<typename Key, typename Value>
class AdaptiveCacheStrategy : public ICacheManager<Key, Value> {
private:
    enum class ActiveStrategy {
        LRU,
        LFU,
        FIFO,
        TTL
    };
    
    std::unique_ptr<ICacheManager<Key, Value>> currentCache_;
    ActiveStrategy currentStrategy_;
    
    // 性能监控
    std::chrono::steady_clock::time_point lastEvaluation_;
    std::chrono::seconds evaluationInterval_{300}; // 5分钟评估一次
    
    struct PerformanceMetrics {
        size_t hits = 0;
        size_t misses = 0;
        
        double hitRatio = 0.0;
        double averageAccessTime = 0.0;
        size_t evictionRate = 0;
        std::chrono::steady_clock::time_point measurementStart;
    };
    
    std::map<ActiveStrategy, PerformanceMetrics> strategyMetrics_;
    mutable std::mutex strategyMutex_;
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    
    // 自动优化
    boost::future<void> optimizationTask_;
    std::atomic<bool> optimizationEnabled_{true};

public:
    explicit AdaptiveCacheStrategy(size_t capacity = 1000,
                                 std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);
    
    ~AdaptiveCacheStrategy();
    
    // === 同步接口实现 ===
    std::optional<Value> get(const Key& key) override;
    bool put(const Key& key, const Value& value) override;
    bool remove(const Key& key) override;
    bool containsKey(const Key& key) const override;
    void clear() override;
    size_t size() const override;
    
    // === 🔴 补充缺失的ICacheManager接口方法声明 ===
    bool contains(const Key& key) const override;
    void putBatch(const std::map<Key, Value>& items) override;
    std::map<Key, Value> getBatch(const std::vector<Key>& keys) override;
    void removeBatch(const std::vector<Key>& keys) override;
    size_t capacity() const override;
    void setCapacity(size_t newCapacity) override;
    void evictExpired() override;
    void optimize() override;
    CacheStatistics getStatistics() const override;
    void resetStatistics() override;
    std::string generateReport() const override;
    void updateConfig(const CacheConfig& config) override;
    CacheConfig getConfig() const override;
    CacheStrategy getStrategy() const override;
    
    // === 异步接口实现 ===
    boost::future<std::optional<Value>> getAsync(const Key& key) override;
    boost::future<bool> putAsync(const Key& key, const Value& value) override;
    boost::future<bool> removeAsync(const Key& key) override;
    
    boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
    getBatchAsync(const std::vector<Key>& keys) override;
    
    boost::future<std::vector<bool>> 
    putBatchAsync(const std::vector<std::pair<Key, Value>>& items) override;
    
    boost::future<CacheStatistics> getStatisticsAsync() override;
    boost::future<void> clearAsync() override;
    
    boost::future<size_t> 
    warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) override;
    
    boost::future<std::optional<Value>> 
    refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) override;
    
    boost::future<Value> 
    computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) override;
    
    // === 自适应专用方法 ===
    
    /**
     * @brief 异步强制策略评估和切换
     */
    boost::future<ActiveStrategy> evaluateAndSwitchAsync();
    
    /**
     * @brief 异步获取当前策略性能报告
     */
    boost::future<std::map<std::string, double>> getPerformanceReportAsync();
    
    /**
     * @brief 手动设置策略
     */
    boost::future<bool> setStrategyAsync(ActiveStrategy strategy);

private:
    void startOptimizationTask();
    void stopOptimizationTask();
    ActiveStrategy selectBestStrategy();
    std::unique_ptr<ICacheManager<Key, Value>> createStrategyCache(ActiveStrategy strategy, size_t capacity);
    void migrateCacheData(ICacheManager<Key, Value>* from, ICacheManager<Key, Value>* to);
    
    template<typename Func>
    auto executeAsync(Func&& func) -> boost::future<decltype(func())>;
};

// === 工厂函数 ===

/**
 * @brief 创建异步LRU缓存
 */
template<typename Key, typename Value>
boost::future<std::unique_ptr<LRUCacheStrategy<Key, Value>>> 
createLRUCacheAsync(size_t capacity, std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);

/**
 * @brief 创建异步LFU缓存
 */
template<typename Key, typename Value>
boost::future<std::unique_ptr<LFUCacheStrategy<Key, Value>>> 
createLFUCacheAsync(size_t capacity, std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);

/**
 * @brief 创建异步自适应缓存
 */
template<typename Key, typename Value>
boost::future<std::unique_ptr<AdaptiveCacheStrategy<Key, Value>>> 
createAdaptiveCacheAsync(size_t capacity, std::shared_ptr<boost::asio::thread_pool> threadPool = nullptr);

} // namespace oscean::common_utils::cache 