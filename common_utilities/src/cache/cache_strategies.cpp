/**
 * @file cache_strategies.cpp
 * @brief 缓存策略核心算法实现
 * @author OSCEAN Team
 * @date 2024
 */

#include "common_utils/cache/cache_strategies.h"
#include "common_utils/streaming/streaming_interfaces.h"  // 为DataChunk类型
#include <algorithm>
#include <chrono>

namespace oscean::common_utils::cache {

// ============================================================================
// LRUCacheStrategy 基础实现
// ============================================================================

template<typename Key, typename Value>
LRUCacheStrategy<Key, Value>::LRUCacheStrategy(size_t capacity, 
                                             std::shared_ptr<boost::asio::thread_pool> threadPool)
    : capacity_(capacity), threadPool_(threadPool), creationTime_(std::chrono::steady_clock::now()) {
    initializeList();
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::initializeList() {
    head_ = std::make_shared<Node>(Key{}, Value{});
    tail_ = std::make_shared<Node>(Key{}, Value{});
    head_->next = tail_;
    tail_->prev = head_;
}

template<typename Key, typename Value>
bool LRUCacheStrategy<Key, Value>::put(const Key& key, const Value& value) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = hashTable_.find(key);
    if (it != hashTable_.end()) {
        // 更新现有节点
        it->second->value = value;
        it->second->accessTime = std::chrono::steady_clock::now();
        moveToFront(it->second);
        hitCount_++;
        return true;
    }
    
    // 检查容量限制
    if (hashTable_.size() >= capacity_) {
        // 移除尾部节点
        auto nodeToRemove = removeTail();
        if (nodeToRemove) {
            hashTable_.erase(nodeToRemove->key);
            evictionCount_++;
        }
    }
    
    // 创建新节点并添加到前端
    auto newNode = std::make_shared<Node>(key, value);
    hashTable_[key] = newNode;
    addToFront(newNode);
    
    missCount_++;
    return true;
}

template<typename Key, typename Value>
std::optional<Value> LRUCacheStrategy<Key, Value>::get(const Key& key) {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = hashTable_.find(key);
    if (it == hashTable_.end()) {
        missCount_++;
        return std::nullopt;
    }
    
    // 更新访问时间并移动到前端
    it->second->accessTime = std::chrono::steady_clock::now();
    
    // 需要升级为写锁来修改链表
    lock.unlock();
    std::unique_lock<std::shared_mutex> writeLock(rwMutex_);
    
    // 重新查找（防止在锁升级期间被修改）
    it = hashTable_.find(key);
    if (it != hashTable_.end()) {
        moveToFront(it->second);
        hitCount_++;
        return it->second->value;
    }
    
    missCount_++;
    return std::nullopt;
}

template<typename Key, typename Value>
bool LRUCacheStrategy<Key, Value>::containsKey(const Key& key) const {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    return hashTable_.find(key) != hashTable_.end();
}

template<typename Key, typename Value>
bool LRUCacheStrategy<Key, Value>::remove(const Key& key) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = hashTable_.find(key);
    if (it == hashTable_.end()) return false;
    
    removeNode(it->second);
    hashTable_.erase(it);
    return true;
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::clear() {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    hashTable_.clear();
    initializeList();
    hitCount_ = 0;
    missCount_ = 0;
    evictionCount_ = 0;
}

template<typename Key, typename Value>
size_t LRUCacheStrategy<Key, Value>::size() const {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    return hashTable_.size();
}

// ============================================================================
// 内部辅助方法
// ============================================================================

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::moveToFront(NodePtr node) {
    if (!node || node == head_->next) return;
    removeNode(node);
    addToFront(node);
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::removeNode(NodePtr node) {
    if (!node) return;
    node->prev->next = node->next;
    node->next->prev = node->prev;
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::addToFront(NodePtr node) {
    if (!node) return;
    node->prev = head_;
    node->next = head_->next;
    head_->next->prev = node;
    head_->next = node;
}

template<typename Key, typename Value>
typename LRUCacheStrategy<Key, Value>::NodePtr LRUCacheStrategy<Key, Value>::removeTail() {
    NodePtr last = tail_->prev;
    if (last == head_) return nullptr;
    removeNode(last);
    return last;
}

// ============================================================================
// 模板显式实例化（常用类型）
// ============================================================================

// 基础类型实例化
template class LRUCacheStrategy<std::string, std::string>;
template class LRUCacheStrategy<std::string, std::vector<uint8_t>>;
template class LRUCacheStrategy<int, std::string>;

// 海洋数据类型实例化
template class LRUCacheStrategy<std::string, std::map<std::string, double>>;

// ============================================================================
// 🔴 补充缺失的ICacheManager接口方法实现
// ============================================================================

template<typename Key, typename Value>
bool LRUCacheStrategy<Key, Value>::contains(const Key& key) const {
    return containsKey(key);  // 直接调用已实现的方法
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::putBatch(const std::map<Key, Value>& items) {
    for (const auto& [key, value] : items) {
        put(key, value);
    }
}

template<typename Key, typename Value>
std::map<Key, Value> LRUCacheStrategy<Key, Value>::getBatch(const std::vector<Key>& keys) {
    std::map<Key, Value> results;
    for (const auto& key : keys) {
        auto value = get(key);
        if (value.has_value()) {
            results[key] = value.value();
        }
    }
    return results;
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::removeBatch(const std::vector<Key>& keys) {
    for (const auto& key : keys) {
        remove(key);
    }
}

template<typename Key, typename Value>
size_t LRUCacheStrategy<Key, Value>::capacity() const {
    return capacity_;
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::setCapacity(size_t newCapacity) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    capacity_ = newCapacity;
    
    // 如果新容量小于当前大小，需要淘汰多余的项目
    while (hashTable_.size() > capacity_) {
        auto nodeToRemove = removeTail();
        if (nodeToRemove) {
            hashTable_.erase(nodeToRemove->key);
            evictionCount_++;
        }
    }
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::evictExpired() {
    // LRU不基于时间过期，此方法为空实现
    // 子类可以重写此方法以实现时间基础的过期
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::optimize() {
    // 简单的优化：重置统计信息
    resetStatistics();
}

template<typename Key, typename Value>
CacheStatistics LRUCacheStrategy<Key, Value>::getStatistics() const {
    CacheStatistics stats;
    stats.hitCount = hitCount_.load();
    stats.missCount = missCount_.load();
    stats.evictionCount = evictionCount_.load();
    
    // 设置新增的字段
    stats.totalRequests = stats.hitCount + stats.missCount;
    stats.hits = stats.hitCount;
    stats.evictions = stats.evictionCount;
    
    size_t total = stats.hitCount + stats.missCount;
    stats.hitRatio = total > 0 ? static_cast<double>(stats.hitCount) / total : 0.0;
    
    stats.totalItems = size();
    stats.creationTime = creationTime_;
    stats.lastAccessTime = std::chrono::steady_clock::now();
    
    return stats;
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::resetStatistics() {
    hitCount_ = 0;
    missCount_ = 0;
    evictionCount_ = 0;
    creationTime_ = std::chrono::steady_clock::now();
}

template<typename Key, typename Value>
std::string LRUCacheStrategy<Key, Value>::generateReport() const {
    auto stats = getStatistics();
    std::ostringstream report;
    
    report << "=== LRU Cache Report ===\\n";
    report << "Capacity: " << capacity_ << "\\n";
    report << "Total Items: " << stats.totalItems << "\\n";
    report << "Hit Count: " << stats.hitCount << "\\n";
    report << "Miss Count: " << stats.missCount << "\\n";
    report << "Hit Ratio: " << (stats.hitRatio * 100) << "%\\n";
    report << "Eviction Count: " << stats.evictionCount << "\\n";
    report << "Total Requests: " << stats.totalRequests << "\\n";
    
    return report.str();
}

template<typename Key, typename Value>
void LRUCacheStrategy<Key, Value>::updateConfig(const CacheConfig& config) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    if (config.capacity > 0) {
        capacity_ = config.capacity;
        
        // 如果新容量小于当前大小，需要清理
        while (hashTable_.size() > capacity_) {
            auto nodeToRemove = removeTail();
            if (nodeToRemove) {
                hashTable_.erase(nodeToRemove->key);
                evictionCount_++;
            }
        }
    }
}

template<typename Key, typename Value>
CacheConfig LRUCacheStrategy<Key, Value>::getConfig() const {
    CacheConfig config;
    config.capacity = capacity_;
    config.strategy = CacheStrategy::LRU;
    config.enableStatistics = true;
    return config;
}

template<typename Key, typename Value>
CacheStrategy LRUCacheStrategy<Key, Value>::getStrategy() const {
    return CacheStrategy::LRU;
}

// ============================================================================
// 🔴 补充缺失的异步方法实现
// ============================================================================

template<typename Key, typename Value>
template<typename Func>
auto LRUCacheStrategy<Key, Value>::executeAsync(Func&& func) -> boost::future<decltype(func())> {
    auto promise = std::make_shared<boost::promise<decltype(func())>>();
    auto future = promise->get_future();
    
    if (threadPool_) {
        boost::asio::post(*threadPool_, [promise, func = std::forward<Func>(func)]() mutable {
            try {
                if constexpr (std::is_void_v<decltype(func())>) {
                    func();
                    promise->set_value();
                } else {
                    promise->set_value(func());
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
    } else {
        // 如果没有线程池，直接同步执行
        try {
            if constexpr (std::is_void_v<decltype(func())>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }
    
    return future;
}

template<typename Key, typename Value>
boost::future<std::optional<Value>> LRUCacheStrategy<Key, Value>::getAsync(const Key& key) {
    return executeAsync([this, key]() {
        return get(key);
    });
}

template<typename Key, typename Value>
boost::future<bool> LRUCacheStrategy<Key, Value>::putAsync(const Key& key, const Value& value) {
    return executeAsync([this, key, value]() {
        return put(key, value);
    });
}

template<typename Key, typename Value>
boost::future<bool> LRUCacheStrategy<Key, Value>::removeAsync(const Key& key) {
    return executeAsync([this, key]() {
        return remove(key);
    });
}

template<typename Key, typename Value>
boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
LRUCacheStrategy<Key, Value>::getBatchAsync(const std::vector<Key>& keys) {
    return executeAsync([this, keys]() {
        std::vector<std::pair<Key, std::optional<Value>>> results;
        results.reserve(keys.size());
        
        for (const auto& key : keys) {
            results.emplace_back(key, get(key));
        }
        
        return results;
    });
}

template<typename Key, typename Value>
boost::future<std::vector<bool>> 
LRUCacheStrategy<Key, Value>::putBatchAsync(const std::vector<std::pair<Key, Value>>& items) {
    return executeAsync([this, items]() {
        std::vector<bool> results;
        results.reserve(items.size());
        
        for (const auto& [key, value] : items) {
            results.push_back(put(key, value));
        }
        
        return results;
    });
}

template<typename Key, typename Value>
boost::future<CacheStatistics> LRUCacheStrategy<Key, Value>::getStatisticsAsync() {
    return executeAsync([this]() {
        return getStatistics();
    });
}

template<typename Key, typename Value>
boost::future<void> LRUCacheStrategy<Key, Value>::clearAsync() {
    return executeAsync([this]() {
        clear();
    });
}

template<typename Key, typename Value>
boost::future<size_t> 
LRUCacheStrategy<Key, Value>::warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) {
    return executeAsync([this, warmupData]() {
        size_t loaded = 0;
        for (const auto& [key, value] : warmupData) {
            if (put(key, value)) {
                loaded++;
            }
        }
        return loaded;
    });
}

template<typename Key, typename Value>
boost::future<std::optional<Value>> 
LRUCacheStrategy<Key, Value>::refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) {
    return executeAsync([this, key, provider = std::move(provider)]() -> std::optional<Value> {
        try {
            auto valueFuture = provider();
            auto value = valueFuture.get();  // 同步等待provider的结果
            
            if (put(key, value)) {
                return value;
            }
        } catch (...) {
            // 刷新失败，返回缓存中的旧值
            return get(key);
        }
        
        return std::nullopt;
    });
}

template<typename Key, typename Value>
boost::future<Value> 
LRUCacheStrategy<Key, Value>::computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) {
    return executeAsync([this, key, computer = std::move(computer)]() -> Value {
        auto existingValue = get(key);
        if (existingValue.has_value()) {
            return existingValue.value();
        }
        
        // 计算新值
        auto computeFuture = computer();
        auto newValue = computeFuture.get();  // 同步等待计算结果
        
        put(key, newValue);
        return newValue;
    });
}

// === LRU专用异步方法 ===

template<typename Key, typename Value>
boost::future<std::vector<std::pair<Key, Value>>> 
LRUCacheStrategy<Key, Value>::getMostRecentAsync(size_t count) {
    return executeAsync([this, count]() {
        std::vector<std::pair<Key, Value>> results;
        std::shared_lock<std::shared_mutex> lock(rwMutex_);
        
        auto current = head_->next;
        while (current != tail_ && results.size() < count) {
            results.emplace_back(current->key, current->value);
            current = current->next;
        }
        
        return results;
    });
}

template<typename Key, typename Value>
boost::future<std::vector<std::pair<Key, Value>>> 
LRUCacheStrategy<Key, Value>::getLeastRecentAsync(size_t count) {
    return executeAsync([this, count]() {
        std::vector<std::pair<Key, Value>> results;
        std::shared_lock<std::shared_mutex> lock(rwMutex_);
        
        auto current = tail_->prev;
        while (current != head_ && results.size() < count) {
            results.emplace_back(current->key, current->value);
            current = current->prev;
        }
        
        return results;
    });
}

template<typename Key, typename Value>
boost::future<size_t> LRUCacheStrategy<Key, Value>::optimizeAsync() {
    return executeAsync([this]() -> size_t {
        size_t initialSize = size();
        optimize();
        return initialSize - size();
    });
}

// === 扩展的模板实例化 ===

// 为测试中使用的类型添加显式实例化
template class LRUCacheStrategy<std::string, int>;
template class LRUCacheStrategy<std::string, std::vector<char>>;
template class LRUCacheStrategy<std::string, std::vector<double>>;
template class LRUCacheStrategy<std::string, std::vector<std::string>>;
template class LRUCacheStrategy<size_t, oscean::common_utils::streaming::DataChunk>;  // 使用正确的命名空间

// ============================================================================
// LFUCacheStrategy 实现
// ============================================================================

template<typename Key, typename Value>
LFUCacheStrategy<Key, Value>::LFUCacheStrategy(size_t capacity, 
                                             std::shared_ptr<boost::asio::thread_pool> threadPool)
    : capacity_(capacity), minFrequency_(1), threadPool_(threadPool) {}

template<typename Key, typename Value>
std::optional<Value> LFUCacheStrategy<Key, Value>::get(const Key& key) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        missCount_++;
        return std::nullopt;
    }
    
    updateFrequency(key);
    hitCount_++;
    return it->second.value;
}

template<typename Key, typename Value>
bool LFUCacheStrategy<Key, Value>::put(const Key& key, const Value& value) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // 更新现有项
        it->second.value = value;
        updateFrequency(key);
        return true;
    }
    
    // 检查容量
    if (cache_.size() >= capacity_) {
        evictLFU();
    }
    
    // 添加新项
    cache_[key] = LFUNode(key, value);
    frequencyLists_[1].push_back(key);
    keyToIterator_[key] = std::prev(frequencyLists_[1].end());
    minFrequency_ = 1;
    
    return true;
}

template<typename Key, typename Value>
bool LFUCacheStrategy<Key, Value>::remove(const Key& key) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) return false;
    
    size_t freq = it->second.frequency;
    auto iterIt = keyToIterator_.find(key);
    if (iterIt != keyToIterator_.end()) {
        frequencyLists_[freq].erase(iterIt->second);
        keyToIterator_.erase(iterIt);
    }
    
    cache_.erase(it);
    return true;
}

template<typename Key, typename Value>
bool LFUCacheStrategy<Key, Value>::containsKey(const Key& key) const {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    return cache_.find(key) != cache_.end();
}

template<typename Key, typename Value>
bool LFUCacheStrategy<Key, Value>::contains(const Key& key) const {
    return containsKey(key);
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::clear() {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    cache_.clear();
    frequencyLists_.clear();
    keyToIterator_.clear();
    minFrequency_ = 1;
    hitCount_ = 0;
    missCount_ = 0;
    evictionCount_ = 0;
}

template<typename Key, typename Value>
size_t LFUCacheStrategy<Key, Value>::size() const {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    return cache_.size();
}

template<typename Key, typename Value>
size_t LFUCacheStrategy<Key, Value>::capacity() const {
    return capacity_;
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::setCapacity(size_t newCapacity) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    capacity_ = newCapacity;
    
    while (cache_.size() > capacity_) {
        evictLFU();
    }
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::putBatch(const std::map<Key, Value>& items) {
    for (const auto& [key, value] : items) {
        put(key, value);
    }
}

template<typename Key, typename Value>
std::map<Key, Value> LFUCacheStrategy<Key, Value>::getBatch(const std::vector<Key>& keys) {
    std::map<Key, Value> results;
    for (const auto& key : keys) {
        auto value = get(key);
        if (value.has_value()) {
            results[key] = value.value();
        }
    }
    return results;
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::removeBatch(const std::vector<Key>& keys) {
    for (const auto& key : keys) {
        remove(key);
    }
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::evictExpired() {
    // LFU不基于时间过期
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::optimize() {
    // 清理空的频率列表
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    auto it = frequencyLists_.begin();
    while (it != frequencyLists_.end()) {
        if (it->second.empty()) {
            it = frequencyLists_.erase(it);
        } else {
            ++it;
        }
    }
}

template<typename Key, typename Value>
CacheStatistics LFUCacheStrategy<Key, Value>::getStatistics() const {
    CacheStatistics stats;
    stats.hitCount = hitCount_.load();
    stats.missCount = missCount_.load();
    stats.evictionCount = evictionCount_.load();
    stats.totalItems = size();
    
    size_t total = stats.hitCount + stats.missCount;
    stats.hitRatio = total > 0 ? static_cast<double>(stats.hitCount) / total : 0.0;
    stats.totalRequests = total;
    stats.hits = stats.hitCount;
    stats.evictions = stats.evictionCount;
    
    return stats;
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::resetStatistics() {
    hitCount_ = 0;
    missCount_ = 0;
    evictionCount_ = 0;
}

template<typename Key, typename Value>
std::string LFUCacheStrategy<Key, Value>::generateReport() const {
    auto stats = getStatistics();
    std::ostringstream report;
    
    report << "=== LFU Cache Report ===\n";
    report << "Capacity: " << capacity_ << "\n";
    report << "Total Items: " << stats.totalItems << "\n";
    report << "Hit Count: " << stats.hitCount << "\n";
    report << "Miss Count: " << stats.missCount << "\n";
    report << "Hit Ratio: " << (stats.hitRatio * 100) << "%\n";
    report << "Eviction Count: " << stats.evictionCount << "\n";
    
    return report.str();
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::updateConfig(const CacheConfig& config) {
    if (config.capacity > 0) {
        setCapacity(config.capacity);
    }
}

template<typename Key, typename Value>
CacheConfig LFUCacheStrategy<Key, Value>::getConfig() const {
    CacheConfig config;
    config.capacity = capacity_;
    config.strategy = CacheStrategy::LFU;
    config.enableStatistics = true;
    return config;
}

template<typename Key, typename Value>
CacheStrategy LFUCacheStrategy<Key, Value>::getStrategy() const {
    return CacheStrategy::LFU;
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::updateFrequency(const Key& key) {
    auto& node = cache_[key];
    size_t oldFreq = node.frequency;
    size_t newFreq = oldFreq + 1;
    
    // 从旧频率列表移除
    auto iterIt = keyToIterator_.find(key);
    if (iterIt != keyToIterator_.end()) {
        frequencyLists_[oldFreq].erase(iterIt->second);
        if (frequencyLists_[oldFreq].empty() && oldFreq == minFrequency_) {
            minFrequency_++;
        }
    }
    
    // 添加到新频率列表
    node.frequency = newFreq;
    node.lastAccess = std::chrono::steady_clock::now();
    frequencyLists_[newFreq].push_back(key);
    keyToIterator_[key] = std::prev(frequencyLists_[newFreq].end());
}

template<typename Key, typename Value>
void LFUCacheStrategy<Key, Value>::evictLFU() {
    if (cache_.empty()) return;
    
    // 找到最小频率的第一个元素
    auto& minFreqList = frequencyLists_[minFrequency_];
    if (minFreqList.empty()) {
        // 找到下一个非空频率
        for (auto& [freq, list] : frequencyLists_) {
            if (!list.empty()) {
                minFrequency_ = freq;
                break;
            }
        }
        return;
    }
    
    Key keyToEvict = minFreqList.front();
    minFreqList.pop_front();
    keyToIterator_.erase(keyToEvict);
    cache_.erase(keyToEvict);
    evictionCount_++;
}

// 异步方法实现
template<typename Key, typename Value>
template<typename Func>
auto LFUCacheStrategy<Key, Value>::executeAsync(Func&& func) -> boost::future<decltype(func())> {
    auto promise = std::make_shared<boost::promise<decltype(func())>>();
    auto future = promise->get_future();
    
    if (threadPool_) {
        boost::asio::post(*threadPool_, [promise, func = std::forward<Func>(func)]() mutable {
            try {
                if constexpr (std::is_void_v<decltype(func())>) {
                    func();
                    promise->set_value();
                } else {
                    promise->set_value(func());
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
    } else {
        try {
            if constexpr (std::is_void_v<decltype(func())>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }
    
    return future;
}

template<typename Key, typename Value>
boost::future<std::optional<Value>> LFUCacheStrategy<Key, Value>::getAsync(const Key& key) {
    return executeAsync([this, key]() { return get(key); });
}

template<typename Key, typename Value>
boost::future<bool> LFUCacheStrategy<Key, Value>::putAsync(const Key& key, const Value& value) {
    return executeAsync([this, key, value]() { return put(key, value); });
}

template<typename Key, typename Value>
boost::future<bool> LFUCacheStrategy<Key, Value>::removeAsync(const Key& key) {
    return executeAsync([this, key]() { return remove(key); });
}

template<typename Key, typename Value>
boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
LFUCacheStrategy<Key, Value>::getBatchAsync(const std::vector<Key>& keys) {
    return executeAsync([this, keys]() {
        std::vector<std::pair<Key, std::optional<Value>>> results;
        results.reserve(keys.size());
        for (const auto& key : keys) {
            results.emplace_back(key, get(key));
        }
        return results;
    });
}

template<typename Key, typename Value>
boost::future<std::vector<bool>> 
LFUCacheStrategy<Key, Value>::putBatchAsync(const std::vector<std::pair<Key, Value>>& items) {
    return executeAsync([this, items]() {
        std::vector<bool> results;
        results.reserve(items.size());
        for (const auto& [key, value] : items) {
            results.push_back(put(key, value));
        }
        return results;
    });
}

template<typename Key, typename Value>
boost::future<CacheStatistics> LFUCacheStrategy<Key, Value>::getStatisticsAsync() {
    return executeAsync([this]() { return getStatistics(); });
}

template<typename Key, typename Value>
boost::future<void> LFUCacheStrategy<Key, Value>::clearAsync() {
    return executeAsync([this]() { clear(); });
}

template<typename Key, typename Value>
boost::future<size_t> 
LFUCacheStrategy<Key, Value>::warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) {
    return executeAsync([this, warmupData]() {
        size_t loaded = 0;
        for (const auto& [key, value] : warmupData) {
            if (put(key, value)) {
                loaded++;
            }
        }
        return loaded;
    });
}

template<typename Key, typename Value>
boost::future<std::optional<Value>> 
LFUCacheStrategy<Key, Value>::refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) {
    return executeAsync([this, key, provider]() -> std::optional<Value> {
        auto future = provider();
        try {
            auto newValue = future.get();
            put(key, newValue);
            return newValue;
        } catch (...) {
            return std::nullopt;
        }
    });
}

template<typename Key, typename Value>
boost::future<Value> 
LFUCacheStrategy<Key, Value>::computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) {
    return executeAsync([this, key, computer]() {
        auto existing = get(key);
        if (existing.has_value()) {
            return existing.value();
        }
        
        auto future = computer();
        auto newValue = future.get();
        put(key, newValue);
        return newValue;
    });
}

// ============================================================================
// 模板显式实例化（LFU策略）
// ============================================================================

template class LFUCacheStrategy<std::string, std::string>;
template class LFUCacheStrategy<std::string, std::vector<uint8_t>>;
template class LFUCacheStrategy<int, std::string>;
template class LFUCacheStrategy<std::string, std::map<std::string, double>>;

// ============================================================================
// FIFOCacheStrategy 实现
// ============================================================================

template<typename Key, typename Value>
FIFOCacheStrategy<Key, Value>::FIFOCacheStrategy(size_t capacity,
                                               std::shared_ptr<boost::asio::thread_pool> threadPool)
    : capacity_(capacity), threadPool_(threadPool) {}

template<typename Key, typename Value>
std::optional<Value> FIFOCacheStrategy<Key, Value>::get(const Key& key) {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        missCount_++;
        return std::nullopt;
    }
    
    hitCount_++;
    return it->second;
}

template<typename Key, typename Value>
bool FIFOCacheStrategy<Key, Value>::put(const Key& key, const Value& value) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // 更新现有项
        it->second = value;
        return true;
    }
    
    // 检查容量
    if (cache_.size() >= capacity_) {
        // 移除最早的项
        if (!insertionOrder_.empty()) {
            Key oldestKey = insertionOrder_.front();
            insertionOrder_.pop();
            cache_.erase(oldestKey);
            evictionCount_++;
        }
    }
    
    // 添加新项
    cache_[key] = value;
    insertionOrder_.push(key);
    
    return true;
}

template<typename Key, typename Value>
bool FIFOCacheStrategy<Key, Value>::remove(const Key& key) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) return false;
    
    cache_.erase(it);
    
    // 从队列中移除（注意：这是O(n)操作，实际实现可能需要优化）
    std::queue<Key> newQueue;
    while (!insertionOrder_.empty()) {
        Key frontKey = insertionOrder_.front();
        insertionOrder_.pop();
        if (frontKey != key) {
            newQueue.push(frontKey);
        }
    }
    insertionOrder_ = std::move(newQueue);
    
    return true;
}

template<typename Key, typename Value>
bool FIFOCacheStrategy<Key, Value>::containsKey(const Key& key) const {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    return cache_.find(key) != cache_.end();
}

template<typename Key, typename Value>
bool FIFOCacheStrategy<Key, Value>::contains(const Key& key) const {
    return containsKey(key);
}

template<typename Key, typename Value>
void FIFOCacheStrategy<Key, Value>::clear() {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    cache_.clear();
    while (!insertionOrder_.empty()) {
        insertionOrder_.pop();
    }
    hitCount_ = 0;
    missCount_ = 0;
    evictionCount_ = 0;
}

template<typename Key, typename Value>
size_t FIFOCacheStrategy<Key, Value>::size() const {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    return cache_.size();
}

template<typename Key, typename Value>
size_t FIFOCacheStrategy<Key, Value>::capacity() const {
    return capacity_;
}

template<typename Key, typename Value>
void FIFOCacheStrategy<Key, Value>::setCapacity(size_t newCapacity) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    capacity_ = newCapacity;
    
    while (cache_.size() > capacity_ && !insertionOrder_.empty()) {
        Key oldestKey = insertionOrder_.front();
        insertionOrder_.pop();
        cache_.erase(oldestKey);
        evictionCount_++;
    }
}

template<typename Key, typename Value>
void FIFOCacheStrategy<Key, Value>::putBatch(const std::map<Key, Value>& items) {
    for (const auto& [key, value] : items) {
        put(key, value);
    }
}

template<typename Key, typename Value>
std::map<Key, Value> FIFOCacheStrategy<Key, Value>::getBatch(const std::vector<Key>& keys) {
    std::map<Key, Value> results;
    for (const auto& key : keys) {
        auto value = get(key);
        if (value.has_value()) {
            results[key] = value.value();
        }
    }
    return results;
}

template<typename Key, typename Value>
void FIFOCacheStrategy<Key, Value>::removeBatch(const std::vector<Key>& keys) {
    for (const auto& key : keys) {
        remove(key);
    }
}

template<typename Key, typename Value>
void FIFOCacheStrategy<Key, Value>::evictExpired() {
    // FIFO不基于时间过期
}

template<typename Key, typename Value>
void FIFOCacheStrategy<Key, Value>::optimize() {
    // FIFO策略优化有限
}

template<typename Key, typename Value>
CacheStatistics FIFOCacheStrategy<Key, Value>::getStatistics() const {
    CacheStatistics stats;
    stats.hitCount = hitCount_.load();
    stats.missCount = missCount_.load();
    stats.evictionCount = evictionCount_.load();
    stats.totalItems = size();
    
    size_t total = stats.hitCount + stats.missCount;
    stats.hitRatio = total > 0 ? static_cast<double>(stats.hitCount) / total : 0.0;
    stats.totalRequests = total;
    stats.hits = stats.hitCount;
    stats.evictions = stats.evictionCount;
    
    return stats;
}

template<typename Key, typename Value>
void FIFOCacheStrategy<Key, Value>::resetStatistics() {
    hitCount_ = 0;
    missCount_ = 0;
    evictionCount_ = 0;
}

template<typename Key, typename Value>
std::string FIFOCacheStrategy<Key, Value>::generateReport() const {
    auto stats = getStatistics();
    std::ostringstream report;
    
    report << "=== FIFO Cache Report ===\n";
    report << "Capacity: " << capacity_ << "\n";
    report << "Total Items: " << stats.totalItems << "\n";
    report << "Hit Count: " << stats.hitCount << "\n";
    report << "Miss Count: " << stats.missCount << "\n";
    report << "Hit Ratio: " << (stats.hitRatio * 100) << "%\n";
    report << "Eviction Count: " << stats.evictionCount << "\n";
    
    return report.str();
}

template<typename Key, typename Value>
void FIFOCacheStrategy<Key, Value>::updateConfig(const CacheConfig& config) {
    if (config.capacity > 0) {
        setCapacity(config.capacity);
    }
}

template<typename Key, typename Value>
CacheConfig FIFOCacheStrategy<Key, Value>::getConfig() const {
    CacheConfig config;
    config.capacity = capacity_;
    config.strategy = CacheStrategy::FIFO;
    config.enableStatistics = true;
    return config;
}

template<typename Key, typename Value>
CacheStrategy FIFOCacheStrategy<Key, Value>::getStrategy() const {
    return CacheStrategy::FIFO;
}

// 异步方法实现
template<typename Key, typename Value>
template<typename Func>
auto FIFOCacheStrategy<Key, Value>::executeAsync(Func&& func) -> boost::future<decltype(func())> {
    auto promise = std::make_shared<boost::promise<decltype(func())>>();
    auto future = promise->get_future();
    
    if (threadPool_) {
        boost::asio::post(*threadPool_, [promise, func = std::forward<Func>(func)]() mutable {
            try {
                if constexpr (std::is_void_v<decltype(func())>) {
                    func();
                    promise->set_value();
                } else {
                    promise->set_value(func());
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
    } else {
        try {
            if constexpr (std::is_void_v<decltype(func())>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }
    
    return future;
}

template<typename Key, typename Value>
boost::future<std::optional<Value>> FIFOCacheStrategy<Key, Value>::getAsync(const Key& key) {
    return executeAsync([this, key]() { return get(key); });
}

template<typename Key, typename Value>
boost::future<bool> FIFOCacheStrategy<Key, Value>::putAsync(const Key& key, const Value& value) {
    return executeAsync([this, key, value]() { return put(key, value); });
}

template<typename Key, typename Value>
boost::future<bool> FIFOCacheStrategy<Key, Value>::removeAsync(const Key& key) {
    return executeAsync([this, key]() { return remove(key); });
}

template<typename Key, typename Value>
boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
FIFOCacheStrategy<Key, Value>::getBatchAsync(const std::vector<Key>& keys) {
    return executeAsync([this, keys]() {
        std::vector<std::pair<Key, std::optional<Value>>> results;
        results.reserve(keys.size());
        for (const auto& key : keys) {
            results.emplace_back(key, get(key));
        }
        return results;
    });
}

template<typename Key, typename Value>
boost::future<std::vector<bool>> 
FIFOCacheStrategy<Key, Value>::putBatchAsync(const std::vector<std::pair<Key, Value>>& items) {
    return executeAsync([this, items]() {
        std::vector<bool> results;
        results.reserve(items.size());
        for (const auto& [key, value] : items) {
            results.push_back(put(key, value));
        }
        return results;
    });
}

template<typename Key, typename Value>
boost::future<CacheStatistics> FIFOCacheStrategy<Key, Value>::getStatisticsAsync() {
    return executeAsync([this]() { return getStatistics(); });
}

template<typename Key, typename Value>
boost::future<void> FIFOCacheStrategy<Key, Value>::clearAsync() {
    return executeAsync([this]() { clear(); });
}

template<typename Key, typename Value>
boost::future<size_t> 
FIFOCacheStrategy<Key, Value>::warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) {
    return executeAsync([this, warmupData]() {
        size_t loaded = 0;
        for (const auto& [key, value] : warmupData) {
            if (put(key, value)) {
                loaded++;
            }
        }
        return loaded;
    });
}

template<typename Key, typename Value>
boost::future<std::optional<Value>> 
FIFOCacheStrategy<Key, Value>::refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) {
    return executeAsync([this, key, provider]() -> std::optional<Value> {
        auto future = provider();
        try {
            auto newValue = future.get();
            put(key, newValue);
            return newValue;
        } catch (...) {
            return std::nullopt;
        }
    });
}

template<typename Key, typename Value>
boost::future<Value> 
FIFOCacheStrategy<Key, Value>::computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) {
    return executeAsync([this, key, computer]() {
        auto existing = get(key);
        if (existing.has_value()) {
            return existing.value();
        }
        
        auto future = computer();
        auto newValue = future.get();
        put(key, newValue);
        return newValue;
    });
}

// ============================================================================
// 模板显式实例化（FIFO策略）
// ============================================================================

template class FIFOCacheStrategy<std::string, std::string>;
template class FIFOCacheStrategy<std::string, std::vector<uint8_t>>;
template class FIFOCacheStrategy<int, std::string>;
template class FIFOCacheStrategy<std::string, std::map<std::string, double>>;

// ============================================================================
// TTLCacheStrategy 实现
// ============================================================================

template<typename Key, typename Value>
TTLCacheStrategy<Key, Value>::TTLCacheStrategy(std::chrono::seconds defaultTTL,
                                             std::shared_ptr<boost::asio::thread_pool> threadPool)
    : defaultTTL_(defaultTTL), threadPool_(threadPool) {
    startAutoCleanup();
}

template<typename Key, typename Value>
TTLCacheStrategy<Key, Value>::~TTLCacheStrategy() {
    stopAutoCleanup();
}

template<typename Key, typename Value>
std::optional<Value> TTLCacheStrategy<Key, Value>::get(const Key& key) {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end() || it->second.isExpired()) {
        if (it != cache_.end()) {
            lock.unlock();
            std::unique_lock<std::shared_mutex> writeLock(rwMutex_);
            cache_.erase(it);
            expirationCount_++;
        }
        missCount_++;
        return std::nullopt;
    }
    
    hitCount_++;
    return it->second.value;
}

template<typename Key, typename Value>
bool TTLCacheStrategy<Key, Value>::put(const Key& key, const Value& value) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    cache_.insert_or_assign(key, TTLNode(value, defaultTTL_));
    return true;
}

template<typename Key, typename Value>
bool TTLCacheStrategy<Key, Value>::putWithTTL(const Key& key, const Value& value, std::chrono::seconds ttl) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    cache_.insert_or_assign(key, TTLNode(value, ttl));
    return true;
}

template<typename Key, typename Value>
bool TTLCacheStrategy<Key, Value>::remove(const Key& key) {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        cache_.erase(it);
        return true;
    }
    return false;
}

template<typename Key, typename Value>
bool TTLCacheStrategy<Key, Value>::containsKey(const Key& key) const {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.find(key);
    return it != cache_.end() && !it->second.isExpired();
}

template<typename Key, typename Value>
bool TTLCacheStrategy<Key, Value>::contains(const Key& key) const {
    return containsKey(key);
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::clear() {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    cache_.clear();
    hitCount_ = 0;
    missCount_ = 0;
    expirationCount_ = 0;
}

template<typename Key, typename Value>
size_t TTLCacheStrategy<Key, Value>::size() const {
    std::shared_lock<std::shared_mutex> lock(rwMutex_);
    
    size_t count = 0;
    for (const auto& [key, node] : cache_) {
        if (!node.isExpired()) {
            count++;
        }
    }
    return count;
}

template<typename Key, typename Value>
size_t TTLCacheStrategy<Key, Value>::capacity() const {
    return SIZE_MAX; // TTL缓存没有硬容量限制
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::setCapacity(size_t newCapacity) {
    // TTL缓存不支持容量设置
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::putBatch(const std::map<Key, Value>& items) {
    for (const auto& [key, value] : items) {
        put(key, value);
    }
}

template<typename Key, typename Value>
std::map<Key, Value> TTLCacheStrategy<Key, Value>::getBatch(const std::vector<Key>& keys) {
    std::map<Key, Value> results;
    for (const auto& key : keys) {
        auto value = get(key);
        if (value.has_value()) {
            results[key] = value.value();
        }
    }
    return results;
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::removeBatch(const std::vector<Key>& keys) {
    for (const auto& key : keys) {
        remove(key);
    }
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::evictExpired() {
    std::unique_lock<std::shared_mutex> lock(rwMutex_);
    
    auto it = cache_.begin();
    while (it != cache_.end()) {
        if (it->second.isExpired()) {
            it = cache_.erase(it);
            expirationCount_++;
        } else {
            ++it;
        }
    }
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::optimize() {
    evictExpired();
}

template<typename Key, typename Value>
CacheStatistics TTLCacheStrategy<Key, Value>::getStatistics() const {
    CacheStatistics stats;
    stats.hitCount = hitCount_.load();
    stats.missCount = missCount_.load();
    stats.evictionCount = expirationCount_.load();
    stats.totalItems = size();
    
    size_t total = stats.hitCount + stats.missCount;
    stats.hitRatio = total > 0 ? static_cast<double>(stats.hitCount) / total : 0.0;
    stats.totalRequests = total;
    stats.hits = stats.hitCount;
    stats.evictions = stats.evictionCount;
    
    return stats;
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::resetStatistics() {
    hitCount_ = 0;
    missCount_ = 0;
    expirationCount_ = 0;
}

template<typename Key, typename Value>
std::string TTLCacheStrategy<Key, Value>::generateReport() const {
    auto stats = getStatistics();
    std::ostringstream report;
    
    report << "=== TTL Cache Report ===\n";
    report << "Default TTL: " << defaultTTL_.count() << " seconds\n";
    report << "Total Items: " << stats.totalItems << "\n";
    report << "Hit Count: " << stats.hitCount << "\n";
    report << "Miss Count: " << stats.missCount << "\n";
    report << "Hit Ratio: " << (stats.hitRatio * 100) << "%\n";
    report << "Expiration Count: " << stats.evictionCount << "\n";
    
    return report.str();
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::updateConfig(const CacheConfig& config) {
    if (config.ttl.count() > 0) {
        defaultTTL_ = std::chrono::duration_cast<std::chrono::seconds>(config.ttl);
    }
}

template<typename Key, typename Value>
CacheConfig TTLCacheStrategy<Key, Value>::getConfig() const {
    CacheConfig config;
    config.ttl = defaultTTL_;
    config.strategy = CacheStrategy::TTL;
    config.enableStatistics = true;
    return config;
}

template<typename Key, typename Value>
CacheStrategy TTLCacheStrategy<Key, Value>::getStrategy() const {
    return CacheStrategy::TTL;
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::startAutoCleanup() {
    if (autoCleanupEnabled_ && threadPool_) {
        cleanupTask_ = executeAsync([this]() {
            while (autoCleanupEnabled_) {
                std::this_thread::sleep_for(cleanupInterval_);
                if (autoCleanupEnabled_) {
                    cleanupExpiredItems();
                }
            }
        });
    }
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::stopAutoCleanup() {
    autoCleanupEnabled_ = false;
    // 等待清理任务结束 (cleanupTask_ future会自动处理)
}

template<typename Key, typename Value>
void TTLCacheStrategy<Key, Value>::cleanupExpiredItems() {
    evictExpired();
}

template<typename Key, typename Value>
template<typename Func>
auto TTLCacheStrategy<Key, Value>::executeAsync(Func&& func) -> boost::future<decltype(func())> {
    auto promise = std::make_shared<boost::promise<decltype(func())>>();
    auto future = promise->get_future();
    
    if (threadPool_) {
        boost::asio::post(*threadPool_, [promise, func = std::forward<Func>(func)]() mutable {
            try {
                if constexpr (std::is_void_v<decltype(func())>) {
                    func();
                    promise->set_value();
                } else {
                    promise->set_value(func());
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
    } else {
        try {
            if constexpr (std::is_void_v<decltype(func())>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }
    
    return future;
}

// 异步方法实现
template<typename Key, typename Value>
boost::future<std::optional<Value>> TTLCacheStrategy<Key, Value>::getAsync(const Key& key) {
    return executeAsync([this, key]() { return get(key); });
}

template<typename Key, typename Value>
boost::future<bool> TTLCacheStrategy<Key, Value>::putAsync(const Key& key, const Value& value) {
    return executeAsync([this, key, value]() { return put(key, value); });
}

template<typename Key, typename Value>
boost::future<bool> TTLCacheStrategy<Key, Value>::putWithTTLAsync(const Key& key, const Value& value, std::chrono::seconds ttl) {
    return executeAsync([this, key, value, ttl]() { return putWithTTL(key, value, ttl); });
}

template<typename Key, typename Value>
boost::future<bool> TTLCacheStrategy<Key, Value>::removeAsync(const Key& key) {
    return executeAsync([this, key]() { return remove(key); });
}

template<typename Key, typename Value>
boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
TTLCacheStrategy<Key, Value>::getBatchAsync(const std::vector<Key>& keys) {
    return executeAsync([this, keys]() {
        std::vector<std::pair<Key, std::optional<Value>>> results;
        for (const auto& key : keys) {
            results.emplace_back(key, get(key));
        }
        return results;
    });
}

template<typename Key, typename Value>
boost::future<std::vector<bool>> 
TTLCacheStrategy<Key, Value>::putBatchAsync(const std::vector<std::pair<Key, Value>>& items) {
    return executeAsync([this, items]() {
        std::vector<bool> results;
        for (const auto& [key, value] : items) {
            results.push_back(put(key, value));
        }
        return results;
    });
}

template<typename Key, typename Value>
boost::future<CacheStatistics> TTLCacheStrategy<Key, Value>::getStatisticsAsync() {
    return executeAsync([this]() { return getStatistics(); });
}

template<typename Key, typename Value>
boost::future<void> TTLCacheStrategy<Key, Value>::clearAsync() {
    return executeAsync([this]() { clear(); });
}

template<typename Key, typename Value>
boost::future<size_t> TTLCacheStrategy<Key, Value>::warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) {
    return executeAsync([this, warmupData]() {
        size_t loaded = 0;
        for (const auto& [key, value] : warmupData) {
            if (put(key, value)) {
                loaded++;
            }
        }
        return loaded;
    });
}

template<typename Key, typename Value>
boost::future<std::optional<Value>> 
TTLCacheStrategy<Key, Value>::refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) {
    return executeAsync([this, key, provider]() -> std::optional<Value> {
        auto future = provider();
        try {
            auto newValue = future.get();
            put(key, newValue);
            return newValue;
        } catch (...) {
            return std::nullopt;
        }
    });
}

template<typename Key, typename Value>
boost::future<Value> 
TTLCacheStrategy<Key, Value>::computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) {
    return executeAsync([this, key, computer]() {
        auto existing = get(key);
        if (existing.has_value()) {
            return existing.value();
        }
        
        auto future = computer();
        auto newValue = future.get();
        put(key, newValue);
        return newValue;
    });
}

template<typename Key, typename Value>
boost::future<size_t> TTLCacheStrategy<Key, Value>::cleanupExpiredAsync() {
    return executeAsync([this]() {
        size_t beforeSize = cache_.size();
        evictExpired();
        return beforeSize - cache_.size();
    });
}

template<typename Key, typename Value>
boost::future<std::vector<Key>> TTLCacheStrategy<Key, Value>::getExpiringKeysAsync(std::chrono::seconds withinTime) {
    return executeAsync([this, withinTime]() {
        std::vector<Key> expiringKeys;
        auto threshold = std::chrono::steady_clock::now() + withinTime;
        
        std::shared_lock<std::shared_mutex> lock(rwMutex_);
        for (const auto& [key, node] : cache_) {
            if (node.expiryTime <= threshold) {
                expiringKeys.push_back(key);
            }
        }
        return expiringKeys;
    });
}

// ============================================================================
// AdaptiveCacheStrategy 实现
// ============================================================================

template<typename Key, typename Value>
AdaptiveCacheStrategy<Key, Value>::AdaptiveCacheStrategy(size_t capacity,
                                                       std::shared_ptr<boost::asio::thread_pool> threadPool)
    : currentStrategy_(ActiveStrategy::LRU), threadPool_(threadPool) {
    currentCache_ = createStrategyCache(currentStrategy_, capacity);
    startOptimizationTask();
}

template<typename Key, typename Value>
AdaptiveCacheStrategy<Key, Value>::~AdaptiveCacheStrategy() {
    stopOptimizationTask();
}

template<typename Key, typename Value>
std::optional<Value> AdaptiveCacheStrategy<Key, Value>::get(const Key& key) {
    auto start = std::chrono::steady_clock::now();
    auto result = currentCache_->get(key);
    auto duration = std::chrono::steady_clock::now() - start;
    
    // 更新策略性能指标
    std::lock_guard<std::mutex> lock(strategyMutex_);
    auto& metrics = strategyMetrics_[currentStrategy_];
    if (result.has_value()) {
        metrics.hits++;
    } else {
        metrics.misses++;
    }
    
    return result;
}

template<typename Key, typename Value>
bool AdaptiveCacheStrategy<Key, Value>::put(const Key& key, const Value& value) {
    return currentCache_->put(key, value);
}

template<typename Key, typename Value>
bool AdaptiveCacheStrategy<Key, Value>::remove(const Key& key) {
    return currentCache_->remove(key);
}

template<typename Key, typename Value>
bool AdaptiveCacheStrategy<Key, Value>::containsKey(const Key& key) const {
    return currentCache_->contains(key);
}

template<typename Key, typename Value>
bool AdaptiveCacheStrategy<Key, Value>::contains(const Key& key) const {
    return containsKey(key);
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::clear() {
    currentCache_->clear();
}

template<typename Key, typename Value>
size_t AdaptiveCacheStrategy<Key, Value>::size() const {
    return currentCache_->size();
}

template<typename Key, typename Value>
size_t AdaptiveCacheStrategy<Key, Value>::capacity() const {
    return currentCache_->capacity();
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::setCapacity(size_t newCapacity) {
    currentCache_->setCapacity(newCapacity);
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::putBatch(const std::map<Key, Value>& items) {
    currentCache_->putBatch(items);
}

template<typename Key, typename Value>
std::map<Key, Value> AdaptiveCacheStrategy<Key, Value>::getBatch(const std::vector<Key>& keys) {
    return currentCache_->getBatch(keys);
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::removeBatch(const std::vector<Key>& keys) {
    currentCache_->removeBatch(keys);
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::evictExpired() {
    currentCache_->evictExpired();
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::optimize() {
    currentCache_->optimize();
    
    // 触发策略评估
    if (std::chrono::steady_clock::now() - lastEvaluation_ > evaluationInterval_) {
        auto bestStrategy = selectBestStrategy();
        if (bestStrategy != currentStrategy_) {
            // 异步切换策略
            setStrategyAsync(bestStrategy);
        }
        lastEvaluation_ = std::chrono::steady_clock::now();
    }
}

template<typename Key, typename Value>
CacheStatistics AdaptiveCacheStrategy<Key, Value>::getStatistics() const {
    return currentCache_->getStatistics();
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::resetStatistics() {
    currentCache_->resetStatistics();
    
    std::lock_guard<std::mutex> lock(strategyMutex_);
    for (auto& [strategy, metrics] : strategyMetrics_) {
        metrics = PerformanceMetrics{};
    }
}

template<typename Key, typename Value>
std::string AdaptiveCacheStrategy<Key, Value>::generateReport() const {
    auto baseReport = currentCache_->generateReport();
    
    std::ostringstream report;
    report << "=== Adaptive Cache Report ===\n";
    report << "Current Strategy: ";
    switch (currentStrategy_) {
        case ActiveStrategy::LRU: report << "LRU"; break;
        case ActiveStrategy::LFU: report << "LFU"; break;
        case ActiveStrategy::FIFO: report << "FIFO"; break;
        case ActiveStrategy::TTL: report << "TTL"; break;
    }
    report << "\n\n" << baseReport;
    
    return report.str();
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::updateConfig(const CacheConfig& config) {
    currentCache_->updateConfig(config);
}

template<typename Key, typename Value>
CacheConfig AdaptiveCacheStrategy<Key, Value>::getConfig() const {
    auto config = currentCache_->getConfig();
    config.strategy = CacheStrategy::ADAPTIVE;
    return config;
}

template<typename Key, typename Value>
CacheStrategy AdaptiveCacheStrategy<Key, Value>::getStrategy() const {
    return CacheStrategy::ADAPTIVE;
}

template<typename Key, typename Value>
typename AdaptiveCacheStrategy<Key, Value>::ActiveStrategy 
AdaptiveCacheStrategy<Key, Value>::selectBestStrategy() {
    std::lock_guard<std::mutex> lock(strategyMutex_);
    
    ActiveStrategy bestStrategy = currentStrategy_;
    double bestHitRatio = 0.0;
    
    for (const auto& [strategy, metrics] : strategyMetrics_) {
        size_t total = metrics.hits + metrics.misses;
        if (total > 0) {
            double hitRatio = static_cast<double>(metrics.hits) / total;
            if (hitRatio > bestHitRatio) {
                bestHitRatio = hitRatio;
                bestStrategy = strategy;
            }
        }
    }
    
    return bestStrategy;
}

template<typename Key, typename Value>
std::unique_ptr<ICacheManager<Key, Value>> 
AdaptiveCacheStrategy<Key, Value>::createStrategyCache(ActiveStrategy strategy, size_t capacity) {
    switch (strategy) {
        case ActiveStrategy::LRU:
            return std::make_unique<LRUCacheStrategy<Key, Value>>(capacity, threadPool_);
        case ActiveStrategy::LFU:
            return std::make_unique<LFUCacheStrategy<Key, Value>>(capacity, threadPool_);
        case ActiveStrategy::FIFO:
            return std::make_unique<FIFOCacheStrategy<Key, Value>>(capacity, threadPool_);
        case ActiveStrategy::TTL:
            return std::make_unique<TTLCacheStrategy<Key, Value>>(std::chrono::minutes(30), threadPool_);
        default:
            return std::make_unique<LRUCacheStrategy<Key, Value>>(capacity, threadPool_);
    }
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::migrateCacheData(ICacheManager<Key, Value>* from, 
                                                        ICacheManager<Key, Value>* to) {
    // 简化的数据迁移 - 在实际实现中可能需要更复杂的逻辑
    // 由于没有标准的"获取所有键值"接口，这里暂时不实现迁移
    // 实际项目中可能需要添加这样的接口
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::startOptimizationTask() {
    if (optimizationEnabled_ && threadPool_) {
        optimizationTask_ = executeAsync([this]() {
            while (optimizationEnabled_) {
                std::this_thread::sleep_for(evaluationInterval_);
                if (optimizationEnabled_) {
                    optimize();
                }
            }
        });
    }
}

template<typename Key, typename Value>
void AdaptiveCacheStrategy<Key, Value>::stopOptimizationTask() {
    optimizationEnabled_ = false;
    // optimizationTask_ future会自动处理
}

template<typename Key, typename Value>
template<typename Func>
auto AdaptiveCacheStrategy<Key, Value>::executeAsync(Func&& func) -> boost::future<decltype(func())> {
    auto promise = std::make_shared<boost::promise<decltype(func())>>();
    auto future = promise->get_future();
    
    if (threadPool_) {
        boost::asio::post(*threadPool_, [promise, func = std::forward<Func>(func)]() mutable {
            try {
                if constexpr (std::is_void_v<decltype(func())>) {
                    func();
                    promise->set_value();
                } else {
                    promise->set_value(func());
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
    } else {
        try {
            if constexpr (std::is_void_v<decltype(func())>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }
    
    return future;
}

// 异步方法实现
template<typename Key, typename Value>
boost::future<std::optional<Value>> AdaptiveCacheStrategy<Key, Value>::getAsync(const Key& key) {
    return executeAsync([this, key]() { return get(key); });
}

template<typename Key, typename Value>
boost::future<bool> AdaptiveCacheStrategy<Key, Value>::putAsync(const Key& key, const Value& value) {
    return executeAsync([this, key, value]() { return put(key, value); });
}

template<typename Key, typename Value>
boost::future<bool> AdaptiveCacheStrategy<Key, Value>::removeAsync(const Key& key) {
    return executeAsync([this, key]() { return remove(key); });
}

template<typename Key, typename Value>
boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
AdaptiveCacheStrategy<Key, Value>::getBatchAsync(const std::vector<Key>& keys) {
    return executeAsync([this, keys]() {
        std::vector<std::pair<Key, std::optional<Value>>> results;
        for (const auto& key : keys) {
            results.emplace_back(key, get(key));
        }
        return results;
    });
}

template<typename Key, typename Value>
boost::future<std::vector<bool>> 
AdaptiveCacheStrategy<Key, Value>::putBatchAsync(const std::vector<std::pair<Key, Value>>& items) {
    return executeAsync([this, items]() {
        std::vector<bool> results;
        for (const auto& [key, value] : items) {
            results.push_back(put(key, value));
        }
        return results;
    });
}

template<typename Key, typename Value>
boost::future<CacheStatistics> AdaptiveCacheStrategy<Key, Value>::getStatisticsAsync() {
    return executeAsync([this]() { return getStatistics(); });
}

template<typename Key, typename Value>
boost::future<void> AdaptiveCacheStrategy<Key, Value>::clearAsync() {
    return executeAsync([this]() { clear(); });
}

template<typename Key, typename Value>
boost::future<size_t> AdaptiveCacheStrategy<Key, Value>::warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) {
    return executeAsync([this, warmupData]() {
        size_t loaded = 0;
        for (const auto& [key, value] : warmupData) {
            if (put(key, value)) {
                loaded++;
            }
        }
        return loaded;
    });
}

template<typename Key, typename Value>
boost::future<std::optional<Value>> 
AdaptiveCacheStrategy<Key, Value>::refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) {
    return executeAsync([this, key, provider]() -> std::optional<Value> {
        auto future = provider();
        try {
            auto newValue = future.get();
            put(key, newValue);
            return newValue;
        } catch (...) {
            return std::nullopt;
        }
    });
}

template<typename Key, typename Value>
boost::future<Value> 
AdaptiveCacheStrategy<Key, Value>::computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) {
    return executeAsync([this, key, computer]() {
        auto existing = get(key);
        if (existing.has_value()) {
            return existing.value();
        }
        
        auto future = computer();
        auto newValue = future.get();
        put(key, newValue);
        return newValue;
    });
}

template<typename Key, typename Value>
boost::future<typename AdaptiveCacheStrategy<Key, Value>::ActiveStrategy> 
AdaptiveCacheStrategy<Key, Value>::evaluateAndSwitchAsync() {
    return executeAsync([this]() {
        auto bestStrategy = selectBestStrategy();
        if (bestStrategy != currentStrategy_) {
            auto newCache = createStrategyCache(bestStrategy, capacity());
            migrateCacheData(currentCache_.get(), newCache.get());
            currentCache_ = std::move(newCache);
            currentStrategy_ = bestStrategy;
        }
        return bestStrategy;
    });
}

template<typename Key, typename Value>
boost::future<std::map<std::string, double>> 
AdaptiveCacheStrategy<Key, Value>::getPerformanceReportAsync() {
    return executeAsync([this]() {
        std::map<std::string, double> report;
        std::lock_guard<std::mutex> lock(strategyMutex_);
        
        for (const auto& [strategy, metrics] : strategyMetrics_) {
            std::string strategyName;
            switch (strategy) {
                case ActiveStrategy::LRU: strategyName = "LRU"; break;
                case ActiveStrategy::LFU: strategyName = "LFU"; break;
                case ActiveStrategy::FIFO: strategyName = "FIFO"; break;
                case ActiveStrategy::TTL: strategyName = "TTL"; break;
            }
            
            size_t total = metrics.hits + metrics.misses;
            double hitRatio = total > 0 ? static_cast<double>(metrics.hits) / total : 0.0;
            report[strategyName + "_hit_ratio"] = hitRatio;
            report[strategyName + "_total_requests"] = static_cast<double>(total);
        }
        
        return report;
    });
}

template<typename Key, typename Value>
boost::future<bool> AdaptiveCacheStrategy<Key, Value>::setStrategyAsync(ActiveStrategy strategy) {
    return executeAsync([this, strategy]() {
        if (strategy != currentStrategy_) {
            auto newCache = createStrategyCache(strategy, capacity());
            migrateCacheData(currentCache_.get(), newCache.get());
            currentCache_ = std::move(newCache);
            currentStrategy_ = strategy;
            return true;
        }
        return false;
    });
}

// ============================================================================
// 模板显式实例化（TTL和Adaptive策略）
// ============================================================================

template class TTLCacheStrategy<std::string, std::string>;
template class TTLCacheStrategy<std::string, std::vector<uint8_t>>;
template class TTLCacheStrategy<int, std::string>;
template class TTLCacheStrategy<std::string, std::map<std::string, double>>;
template class TTLCacheStrategy<std::string, int>;

template class AdaptiveCacheStrategy<std::string, std::string>;
template class AdaptiveCacheStrategy<std::string, std::vector<uint8_t>>;
template class AdaptiveCacheStrategy<int, std::string>;
template class AdaptiveCacheStrategy<std::string, std::map<std::string, double>>;
template class AdaptiveCacheStrategy<std::string, int>;

} // namespace oscean::common_utils::cache 