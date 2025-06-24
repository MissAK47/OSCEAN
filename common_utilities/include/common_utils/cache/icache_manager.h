/**
 * @file icache_manager.h
 * @brief 内部缓存管理器接口定义 - 仅供内部实现使用
 * @author OSCEAN Team
 * @date 2024
 * 
 * ⚠️  重要提示：
 * 这是内部实现接口，外部代码不应直接使用此接口！
 * 
 * 外部代码应该使用：
 * #include "common_utils/infrastructure/common_services_factory.h"
 * 
 * auto factory = CommonServicesFactory::create();
 * auto cache = factory->createCache<Key, Value>("name", capacity);
 * // cache的类型是CommonServices::ICache<Key, Value>
 * 
 * ✅ 正确的使用模式：
 * - 外部只包含 common_services_factory.h
 * - 使用 CommonServices::ICache<K,V> 接口
 * - 通过 factory->createCache() 创建实例
 * 
 * ❌ 错误的使用模式：
 * - 直接包含此文件
 * - 直接使用 cache::ICacheManager<K,V>
 * - 直接创建 cache 模块的实现类
 * 
 * 重构目标：
 * ✅ 提供统一的缓存管理接口
 * ✅ 支持异步操作（基于boost::future）
 * ✅ 简化接口层次，移除领域特定功能
 * ✅ 保持与现有代码的兼容性
 */

#pragma once

#include "../utilities/boost_config.h"
#include "cache_config.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <chrono>
#include <sstream>
#include <functional>

// 立即包含boost::future - 参考CRS模块
#include <boost/thread/future.hpp>

namespace oscean::common_utils::cache {

/**
 * @brief 缓存统计信息
 */
struct CacheStatistics {
    size_t totalItems = 0;
    size_t hitCount = 0;
    size_t missCount = 0;
    size_t evictionCount = 0;
    size_t memoryUsageBytes = 0;
    double hitRatio = 0.0;
    double averageAccessTime = 0.0;
    std::chrono::steady_clock::time_point lastAccessTime;
    std::chrono::steady_clock::time_point creationTime;
    
    // 兼容性字段
    size_t totalRequests = 0;    // 总请求数（hitCount + missCount）
    size_t hits = 0;             // 命中数（与hitCount相同）
    size_t evictions = 0;        // 淘汰数（与evictionCount相同）
    
    /**
     * @brief 计算命中率
     */
    void updateHitRatio() noexcept {
        const size_t totalReqs = hitCount + missCount;
        hitRatio = (totalReqs > 0) ? 
            static_cast<double>(hitCount) / totalReqs : 0.0;
        
        // 同步兼容性字段
        this->totalRequests = totalReqs;
        this->hits = hitCount;
        this->evictions = evictionCount;
    }
    
    /**
     * @brief 转换为字符串表示
     */
    std::string toString() const {
        std::ostringstream oss;
        oss << "CacheStatistics{"
            << "totalItems=" << totalItems
            << ", hitCount=" << hitCount
            << ", missCount=" << missCount
            << ", evictionCount=" << evictionCount
            << ", hitRatio=" << hitRatio
            << ", memoryUsageBytes=" << memoryUsageBytes
            << "}";
        return oss.str();
    }
};

/**
 * @brief 核心缓存管理器接口
 * 
 * 提供统一的缓存管理接口，支持同步和异步操作
 */
template<typename Key, typename Value>
class ICacheManager {
public:
    virtual ~ICacheManager() = default;
    
    // === 基础同步接口 ===
    
    /**
     * @brief 获取缓存项
     * @param key 缓存键
     * @return 缓存值的可选对象
     */
    virtual std::optional<Value> get(const Key& key) = 0;
    
    /**
     * @brief 设置缓存项
     * @param key 缓存键
     * @param value 缓存值
     * @return 是否成功设置
     */
    virtual bool put(const Key& key, const Value& value) = 0;
    
    /**
     * @brief 删除缓存项
     * @param key 缓存键
     * @return 是否成功删除
     */
    virtual bool remove(const Key& key) = 0;
    
    /**
     * @brief 检查键是否存在
     * @param key 缓存键
     * @return 是否存在
     */
    virtual bool containsKey(const Key& key) const = 0;
    
    /**
     * @brief 检查键是否存在（别名方法）
     * @param key 缓存键
     * @return 是否存在
     */
    virtual bool contains(const Key& key) const = 0;
    
    /**
     * @brief 清空缓存
     */
    virtual void clear() = 0;
    
    /**
     * @brief 获取缓存大小
     * @return 当前缓存项数量
     */
    virtual size_t size() const = 0;
    
    /**
     * @brief 获取缓存容量
     * @return 最大容量
     */
    virtual size_t capacity() const = 0;
    
    /**
     * @brief 设置缓存容量
     * @param newCapacity 新的容量
     */
    virtual void setCapacity(size_t newCapacity) = 0;
    
    // === 批量操作接口 ===
    
    /**
     * @brief 批量获取缓存项
     * @param keys 键列表
     * @return 键值对映射
     */
    virtual std::map<Key, Value> getBatch(const std::vector<Key>& keys) = 0;
    
    /**
     * @brief 批量设置缓存项
     * @param items 键值对映射
     */
    virtual void putBatch(const std::map<Key, Value>& items) = 0;
    
    /**
     * @brief 批量删除缓存项
     * @param keys 键列表
     */
    virtual void removeBatch(const std::vector<Key>& keys) = 0;
    
    // === 维护操作接口 ===
    
    /**
     * @brief 清理过期项
     */
    virtual void evictExpired() = 0;
    
    /**
     * @brief 优化缓存
     */
    virtual void optimize() = 0;
    
    /**
     * @brief 获取缓存统计
     * @return 统计信息
     */
    virtual CacheStatistics getStatistics() const = 0;
    
    /**
     * @brief 重置统计信息
     */
    virtual void resetStatistics() = 0;
    
    /**
     * @brief 生成缓存报告
     * @return 报告字符串
     */
    virtual std::string generateReport() const = 0;
    
    // === 配置管理接口 ===
    
    /**
     * @brief 更新缓存配置
     * @param config 新配置
     */
    virtual void updateConfig(const CacheConfig& config) = 0;
    
    /**
     * @brief 获取当前配置
     * @return 当前配置
     */
    virtual CacheConfig getConfig() const = 0;
    
    /**
     * @brief 获取缓存策略
     * @return 当前策略
     */
    virtual CacheStrategy getStrategy() const = 0;
    
    // === 异步接口（基于boost::future）===
    
    /**
     * @brief 异步获取缓存项
     * @param key 缓存键
     * @return 异步结果
     */
    virtual boost::future<std::optional<Value>> getAsync(const Key& key) = 0;
    
    /**
     * @brief 异步设置缓存项
     * @param key 缓存键  
     * @param value 缓存值
     * @return 异步操作结果
     */
    virtual boost::future<bool> putAsync(const Key& key, const Value& value) = 0;
    
    /**
     * @brief 异步删除缓存项
     * @param key 缓存键
     * @return 异步操作结果
     */
    virtual boost::future<bool> removeAsync(const Key& key) = 0;
    
    /**
     * @brief 异步批量获取
     * @param keys 键列表
     * @return 异步结果
     */
    virtual boost::future<std::vector<std::pair<Key, std::optional<Value>>>> 
    getBatchAsync(const std::vector<Key>& keys) = 0;
    
    /**
     * @brief 异步批量设置
     * @param items 键值对列表
     * @return 异步操作结果列表
     */
    virtual boost::future<std::vector<bool>> 
    putBatchAsync(const std::vector<std::pair<Key, Value>>& items) = 0;
    
    /**
     * @brief 异步获取统计信息
     * @return 异步统计结果
     */
    virtual boost::future<CacheStatistics> getStatisticsAsync() = 0;
    
    /**
     * @brief 异步清空缓存
     * @return 异步操作完成标志
     */
    virtual boost::future<void> clearAsync() = 0;
    
    // === 高级功能接口 ===
    
    /**
     * @brief 异步预热缓存
     * @param warmupData 预热数据
     * @return 成功加载的项目数
     */
    virtual boost::future<size_t> 
    warmupAsync(const std::vector<std::pair<Key, Value>>& warmupData) = 0;
    
    /**
     * @brief 异步刷新缓存项
     * @param key 缓存键
     * @param provider 数据提供者函数
     * @return 刷新后的值
     */
    virtual boost::future<std::optional<Value>> 
    refreshAsync(const Key& key, std::function<boost::future<Value>()> provider) = 0;
    
    /**
     * @brief 异步计算并缓存
     * @param key 缓存键
     * @param computer 计算函数
     * @return 计算结果
     */
    virtual boost::future<Value> 
    computeIfAbsentAsync(const Key& key, std::function<boost::future<Value>()> computer) = 0;
};

} // namespace oscean::common_utils::cache 