#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <atomic>
#include <future>

// 缓存系统相关
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/cache/cache_strategies.h"
#include "common_utils/cache/cache_config.h"

// 通用工具
#include "common_utils/utilities/logging_utils.h"

using namespace oscean::common_utils::cache;
using namespace oscean::common_utils::infrastructure;

// =============================================================================
// 测试固件类
// =============================================================================

/**
 * @brief 缓存系统核心测试固件
 */
class CacheSystemCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        LOG_INFO("=== 缓存系统核心测试开始 ===");
        
        // 创建公共服务工厂
        factory = CommonServicesFactory::createForTesting();
        
        // 默认测试参数
        defaultCapacity = 100;
        testDataSize = 50;
        
        // 准备测试数据
        setupTestData();
    }
    
    void TearDown() override {
        LOG_INFO("=== 缓存系统核心测试结束 ===");
        factory.reset();
    }
    
    void setupTestData() {
        // 生成测试键值对
        testData.clear();
        for (int i = 0; i < testDataSize; ++i) {
            std::string key = "key_" + std::to_string(i);
            std::string value = "value_" + std::to_string(i) + "_data";
            testData.emplace_back(key, value);
        }
        
        // 生成大数据集用于压力测试
        largeTestData.clear();
        for (int i = 0; i < 1000; ++i) {
            std::string key = "large_key_" + std::to_string(i);
            std::string value = "large_value_" + std::to_string(i) + "_extended_data_for_testing";
            largeTestData.emplace_back(key, value);
        }
    }
    
    // 创建不同策略的缓存实例
    template<typename Key, typename Value>
    std::shared_ptr<ICache<Key, Value>> createLRUCache(size_t capacity = 0) {
        size_t cap = capacity > 0 ? capacity : defaultCapacity;
        return factory->createCache<Key, Value>("test_lru_cache", cap, "LRU");
    }
    
    template<typename Key, typename Value>
    std::shared_ptr<ICache<Key, Value>> createLFUCache(size_t capacity = 0) {
        size_t cap = capacity > 0 ? capacity : defaultCapacity;
        return factory->createCache<Key, Value>("test_lfu_cache", cap, "LFU");
    }
    
    template<typename Key, typename Value>
    std::shared_ptr<ICache<Key, Value>> createFIFOCache(size_t capacity = 0) {
        size_t cap = capacity > 0 ? capacity : defaultCapacity;
        return factory->createCache<Key, Value>("test_fifo_cache", cap, "FIFO");
    }
    
    template<typename Key, typename Value>
    std::shared_ptr<ICache<Key, Value>> createTTLCache(size_t capacity = 0) {
        size_t cap = capacity > 0 ? capacity : defaultCapacity;
        return factory->createCache<Key, Value>("test_ttl_cache", cap, "TTL");
    }
    
    template<typename Key, typename Value>
    std::shared_ptr<ICache<Key, Value>> createAdaptiveCache(size_t capacity = 0) {
        size_t cap = capacity > 0 ? capacity : defaultCapacity;
        return factory->createCache<Key, Value>("test_adaptive_cache", cap, "ADAPTIVE");
    }
    
    // 验证缓存统计信息的辅助方法
    void validateCacheStatistics(const CacheStatistics& stats, 
                                size_t expectedHits, 
                                size_t expectedMisses) {
        EXPECT_EQ(stats.hitCount, expectedHits) << "缓存命中数不匹配";
        EXPECT_EQ(stats.missCount, expectedMisses) << "缓存未命中数不匹配";
        
        double expectedHitRatio = (expectedHits + expectedMisses > 0) ? 
            static_cast<double>(expectedHits) / (expectedHits + expectedMisses) : 0.0;
        EXPECT_NEAR(stats.hitRatio, expectedHitRatio, 0.001) << "缓存命中率计算错误";
    }
    
protected:
    std::shared_ptr<CommonServicesFactory> factory;
    size_t defaultCapacity;
    size_t testDataSize;
    std::vector<std::pair<std::string, std::string>> testData;
    std::vector<std::pair<std::string, std::string>> largeTestData;
};

// =============================================================================
// 1.2.1 基础缓存操作测试
// =============================================================================

/**
 * @brief 测试基础的Put/Get操作
 */
TEST_F(CacheSystemCoreTest, BasicPutGetOperations) {
    LOG_INFO("--- 测试基础Put/Get操作 ---");
    
    auto cache = createLRUCache<std::string, std::string>();
    ASSERT_NE(cache, nullptr);
    
    // 测试基础put/get操作
    EXPECT_TRUE(cache->put("key1", "value1"));
    EXPECT_TRUE(cache->put("key2", "value2"));
    EXPECT_TRUE(cache->put("key3", "value3"));
    
    // 验证get操作
    auto result1 = cache->get("key1");
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), "value1");
    
    auto result2 = cache->get("key2");
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), "value2");
    
    auto result3 = cache->get("key3");
    ASSERT_TRUE(result3.has_value());
    EXPECT_EQ(result3.value(), "value3");
    
    // 测试不存在的键
    auto resultNone = cache->get("nonexistent");
    EXPECT_FALSE(resultNone.has_value());
    
    // 验证缓存大小
    EXPECT_EQ(cache->size(), 3);
    
    // 测试覆盖更新
    EXPECT_TRUE(cache->put("key1", "updated_value1"));
    auto updatedResult = cache->get("key1");
    ASSERT_TRUE(updatedResult.has_value());
    EXPECT_EQ(updatedResult.value(), "updated_value1");
    EXPECT_EQ(cache->size(), 3); // 大小不变
    
    LOG_INFO("✅ 基础Put/Get操作测试通过");
}

/**
 * @brief 测试TTL过期功能
 */
TEST_F(CacheSystemCoreTest, TTLExpiration) {
    LOG_INFO("--- 测试TTL过期功能 ---");
    
    // 创建带TTL的缓存策略
    auto ttlStrategy = std::make_unique<TTLCacheStrategy<std::string, std::string>>(
        std::chrono::seconds(1), // 1秒TTL
        defaultCapacity
    );
    
    // 添加数据
    EXPECT_TRUE(ttlStrategy->put("temp_key1", "temp_value1"));
    EXPECT_TRUE(ttlStrategy->put("temp_key2", "temp_value2"));
    
    // 立即验证数据存在
    auto result1 = ttlStrategy->get("temp_key1");
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), "temp_value1");
    
    // 等待过期
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    
    // 清理过期项
    ttlStrategy->evictExpired();
    
    // 验证数据已过期
    auto expiredResult1 = ttlStrategy->get("temp_key1");
    EXPECT_FALSE(expiredResult1.has_value());
    
    auto expiredResult2 = ttlStrategy->get("temp_key2");
    EXPECT_FALSE(expiredResult2.has_value());
    
    // 验证缓存已清空
    EXPECT_EQ(ttlStrategy->size(), 0);
    
    LOG_INFO("✅ TTL过期功能测试通过");
}

/**
 * @brief 测试LRU淘汰策略
 */
TEST_F(CacheSystemCoreTest, LRUEviction) {
    LOG_INFO("--- 测试LRU淘汰策略 ---");
    
    size_t smallCapacity = 3;
    auto cache = createLRUCache<std::string, std::string>(smallCapacity);
    ASSERT_NE(cache, nullptr);
    
    // 填充缓存到容量上限
    EXPECT_TRUE(cache->put("key1", "value1"));
    EXPECT_TRUE(cache->put("key2", "value2"));
    EXPECT_TRUE(cache->put("key3", "value3"));
    EXPECT_EQ(cache->size(), 3);
    
    // 访问key1，使其成为最近使用的
    auto result1 = cache->get("key1");
    ASSERT_TRUE(result1.has_value());
    
    // 添加新的键，应该淘汰key2（最久未使用）
    EXPECT_TRUE(cache->put("key4", "value4"));
    EXPECT_EQ(cache->size(), 3);
    
    // 验证key2被淘汰，其他键仍存在
    EXPECT_FALSE(cache->get("key2").has_value());
    EXPECT_TRUE(cache->get("key1").has_value());
    EXPECT_TRUE(cache->get("key3").has_value());
    EXPECT_TRUE(cache->get("key4").has_value());
    
    // 再次访问key1和key3，然后添加key5
    cache->get("key1");
    cache->get("key3");
    EXPECT_TRUE(cache->put("key5", "value5"));
    
    // key4应该被淘汰（最久未使用）
    EXPECT_FALSE(cache->get("key4").has_value());
    EXPECT_TRUE(cache->get("key1").has_value());
    EXPECT_TRUE(cache->get("key3").has_value());
    EXPECT_TRUE(cache->get("key5").has_value());
    
    LOG_INFO("✅ LRU淘汰策略测试通过");
}

/**
 * @brief 测试缓存无效化功能
 */
TEST_F(CacheSystemCoreTest, CacheInvalidation) {
    LOG_INFO("--- 测试缓存无效化功能 ---");
    
    auto cache = createLRUCache<std::string, std::string>();
    ASSERT_NE(cache, nullptr);
    
    // 添加测试数据
    for (const auto& [key, value] : testData) {
        EXPECT_TRUE(cache->put(key, value));
    }
    EXPECT_EQ(cache->size(), testData.size());
    
    // 测试单个键删除
    EXPECT_TRUE(cache->remove("key_5"));
    EXPECT_FALSE(cache->get("key_5").has_value());
    EXPECT_EQ(cache->size(), testData.size() - 1);
    
    // 测试批量删除
    std::vector<std::string> keysToRemove = {"key_1", "key_3", "key_7"};
    cache->removeBatch(keysToRemove);
    
    for (const auto& key : keysToRemove) {
        EXPECT_FALSE(cache->get(key).has_value()) << "键 " << key << " 应该已被删除";
    }
    EXPECT_EQ(cache->size(), testData.size() - 1 - keysToRemove.size());
    
    // 测试全部清空
    cache->clear();
    EXPECT_EQ(cache->size(), 0);
    
    // 验证所有键都不存在
    for (const auto& [key, value] : testData) {
        EXPECT_FALSE(cache->get(key).has_value());
    }
    
    LOG_INFO("✅ 缓存无效化功能测试通过");
}

// =============================================================================
// 1.2.2 缓存策略测试
// =============================================================================

/**
 * @brief 测试LFU策略
 */
TEST_F(CacheSystemCoreTest, LFUStrategy) {
    LOG_INFO("--- 测试LFU策略 ---");
    
    size_t smallCapacity = 3;
    auto lfuStrategy = std::make_unique<LFUCacheStrategy<std::string, std::string>>(smallCapacity);
    
    // 添加初始数据
    EXPECT_TRUE(lfuStrategy->put("key1", "value1"));
    EXPECT_TRUE(lfuStrategy->put("key2", "value2"));
    EXPECT_TRUE(lfuStrategy->put("key3", "value3"));
    
    // 多次访问key1（频率=4）
    lfuStrategy->get("key1");
    lfuStrategy->get("key1");
    lfuStrategy->get("key1");
    
    // 访问key2两次（频率=3）
    lfuStrategy->get("key2");
    lfuStrategy->get("key2");
    
    // key3只访问一次（频率=2，包括put时的初始频率）
    lfuStrategy->get("key3");
    
    // 添加新键，应该淘汰频率最低的key3
    EXPECT_TRUE(lfuStrategy->put("key4", "value4"));
    
    // 验证key3被淘汰
    EXPECT_FALSE(lfuStrategy->get("key3").has_value());
    EXPECT_TRUE(lfuStrategy->get("key1").has_value());
    EXPECT_TRUE(lfuStrategy->get("key2").has_value());
    EXPECT_TRUE(lfuStrategy->get("key4").has_value());
    
    LOG_INFO("✅ LFU策略测试通过");
}

/**
 * @brief 测试FIFO策略
 */
TEST_F(CacheSystemCoreTest, FIFOStrategy) {
    LOG_INFO("--- 测试FIFO策略 ---");
    
    size_t smallCapacity = 3;
    auto fifoStrategy = std::make_unique<FIFOCacheStrategy<std::string, std::string>>(smallCapacity);
    
    // 按顺序添加数据
    EXPECT_TRUE(fifoStrategy->put("first", "value1"));
    EXPECT_TRUE(fifoStrategy->put("second", "value2"));
    EXPECT_TRUE(fifoStrategy->put("third", "value3"));
    EXPECT_EQ(fifoStrategy->size(), 3);
    
    // 访问所有键（不应影响FIFO顺序）
    fifoStrategy->get("first");
    fifoStrategy->get("second");
    fifoStrategy->get("third");
    
    // 添加新键，应该淘汰最先进入的"first"
    EXPECT_TRUE(fifoStrategy->put("fourth", "value4"));
    EXPECT_EQ(fifoStrategy->size(), 3);
    
    // 验证"first"被淘汰，按FIFO顺序
    EXPECT_FALSE(fifoStrategy->get("first").has_value());
    EXPECT_TRUE(fifoStrategy->get("second").has_value());
    EXPECT_TRUE(fifoStrategy->get("third").has_value());
    EXPECT_TRUE(fifoStrategy->get("fourth").has_value());
    
    // 继续添加，应该淘汰"second"
    EXPECT_TRUE(fifoStrategy->put("fifth", "value5"));
    EXPECT_FALSE(fifoStrategy->get("second").has_value());
    EXPECT_TRUE(fifoStrategy->get("third").has_value());
    EXPECT_TRUE(fifoStrategy->get("fourth").has_value());
    EXPECT_TRUE(fifoStrategy->get("fifth").has_value());
    
    LOG_INFO("✅ FIFO策略测试通过");
}

/**
 * @brief 测试自适应策略
 */
TEST_F(CacheSystemCoreTest, AdaptiveStrategy) {
    LOG_INFO("--- 测试自适应策略 ---");
    
    auto adaptiveStrategy = std::make_unique<AdaptiveCacheStrategy<std::string, std::string>>(defaultCapacity);
    
    // 添加测试数据
    for (int i = 0; i < 20; ++i) {
        std::string key = "adaptive_key_" + std::to_string(i);
        std::string value = "adaptive_value_" + std::to_string(i);
        EXPECT_TRUE(adaptiveStrategy->put(key, value));
    }
    
    // 模拟不同的访问模式
    
    // 1. 热点数据访问（高频率）
    for (int round = 0; round < 5; ++round) {
        for (int i = 0; i < 5; ++i) {
            std::string key = "adaptive_key_" + std::to_string(i);
            adaptiveStrategy->get(key);
        }
    }
    
    // 2. 时间局部性访问
    for (int i = 10; i < 15; ++i) {
        std::string key = "adaptive_key_" + std::to_string(i);
        adaptiveStrategy->get(key);
        adaptiveStrategy->get(key); // 重复访问
    }
    
    // 获取初始统计信息
    auto stats1 = adaptiveStrategy->getStatistics();
    EXPECT_GT(stats1.hitCount, 0);
    
    // 添加更多数据，触发策略评估
    for (int i = 20; i < 40; ++i) {
        std::string key = "adaptive_key_" + std::to_string(i);
        std::string value = "adaptive_value_" + std::to_string(i);
        adaptiveStrategy->put(key, value);
    }
    
    // 验证自适应策略能够正常工作
    auto stats2 = adaptiveStrategy->getStatistics();
    EXPECT_GE(stats2.totalItems, stats1.totalItems);
    
    LOG_INFO("✅ 自适应策略测试通过");
}

/**
 * @brief 测试策略切换功能
 */
TEST_F(CacheSystemCoreTest, StrategySwithcing) {
    LOG_INFO("--- 测试策略切换功能 ---");
    
    // 测试从LRU切换到LFU
    auto cache1 = createLRUCache<std::string, std::string>();
    ASSERT_NE(cache1, nullptr);
    
    // 在LRU缓存中添加数据
    for (int i = 0; i < 10; ++i) {
        std::string key = "switch_key_" + std::to_string(i);
        std::string value = "switch_value_" + std::to_string(i);
        cache1->put(key, value);
    }
    
    auto lruStats = cache1->getStatistics();
    
    // 创建新的LFU缓存
    auto cache2 = createLFUCache<std::string, std::string>();
    ASSERT_NE(cache2, nullptr);
    
    // 模拟数据迁移（在实际自适应策略中自动完成）
    for (int i = 0; i < 10; ++i) {
        std::string key = "switch_key_" + std::to_string(i);
        auto value = cache1->get(key);
        if (value.has_value()) {
            cache2->put(key, value.value());
        }
    }
    
    // 验证数据成功迁移
    for (int i = 0; i < 10; ++i) {
        std::string key = "switch_key_" + std::to_string(i);
        auto value1 = cache1->get(key);
        auto value2 = cache2->get(key);
        
        ASSERT_TRUE(value1.has_value());
        ASSERT_TRUE(value2.has_value());
        EXPECT_EQ(value1.value(), value2.value());
    }
    
    LOG_INFO("✅ 策略切换功能测试通过");
}

// =============================================================================
// 1.2.3 并发安全测试
// =============================================================================

/**
 * @brief 测试多线程访问
 */
TEST_F(CacheSystemCoreTest, MultithreadedAccess) {
    LOG_INFO("--- 测试多线程访问 ---");
    
    auto cache = createLRUCache<std::string, std::string>(1000);
    ASSERT_NE(cache, nullptr);
    
    const int numThreads = 8;
    const int operationsPerThread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> successfulOps{0};
    
    // 启动多个线程进行并发操作
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&cache, &successfulOps, t, operationsPerThread]() {
            int localSuccess = 0;
            
            for (int i = 0; i < operationsPerThread; ++i) {
                std::string key = "thread_" + std::to_string(t) + "_key_" + std::to_string(i);
                std::string value = "thread_" + std::to_string(t) + "_value_" + std::to_string(i);
                
                // 执行put操作
                if (cache->put(key, value)) {
                    localSuccess++;
                }
                
                // 执行get操作
                auto result = cache->get(key);
                if (result.has_value() && result.value() == value) {
                    localSuccess++;
                }
                
                // 偶尔执行remove操作
                if (i % 10 == 0) {
                    cache->remove(key);
                }
            }
            
            successfulOps.fetch_add(localSuccess);
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 验证并发操作的正确性
    EXPECT_GT(successfulOps.load(), 0);
    EXPECT_LE(cache->size(), numThreads * operationsPerThread);
    
    // 验证缓存仍然可用
    EXPECT_TRUE(cache->put("final_test", "final_value"));
    auto finalResult = cache->get("final_test");
    ASSERT_TRUE(finalResult.has_value());
    EXPECT_EQ(finalResult.value(), "final_value");
    
    LOG_INFO("✅ 多线程访问测试通过，成功操作数: {}", successfulOps.load());
}

/**
 * @brief 测试并发淘汰
 */
TEST_F(CacheSystemCoreTest, ConcurrentEviction) {
    LOG_INFO("--- 测试并发淘汰 ---");
    
    size_t smallCapacity = 50;
    auto cache = createLRUCache<std::string, std::string>(smallCapacity);
    ASSERT_NE(cache, nullptr);
    
    const int numWriterThreads = 4;
    const int numReaderThreads = 4;
    const int operationsPerThread = 100;
    std::vector<std::thread> threads;
    std::atomic<bool> stopFlag{false};
    
    // 写入线程 - 持续添加数据触发淘汰
    for (int t = 0; t < numWriterThreads; ++t) {
        threads.emplace_back([&cache, &stopFlag, t, operationsPerThread]() {
            for (int i = 0; i < operationsPerThread && !stopFlag.load(); ++i) {
                std::string key = "writer_" + std::to_string(t) + "_" + std::to_string(i);
                std::string value = "value_" + std::to_string(t) + "_" + std::to_string(i);
                cache->put(key, value);
                
                // 偶尔暂停，模拟真实场景
                if (i % 20 == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
    }
    
    // 读取线程 - 持续访问数据
    for (int t = 0; t < numReaderThreads; ++t) {
        threads.emplace_back([&cache, &stopFlag, t, operationsPerThread]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, operationsPerThread - 1);
            
            for (int i = 0; i < operationsPerThread && !stopFlag.load(); ++i) {
                // 随机访问可能存在的键
                int writerThread = dis(gen) % 4;
                int keyIndex = dis(gen);
                std::string key = "writer_" + std::to_string(writerThread) + "_" + std::to_string(keyIndex);
                
                cache->get(key); // 不关心结果，只测试并发安全性
                
                if (i % 30 == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    stopFlag.store(true);
    
    // 验证缓存状态
    EXPECT_LE(cache->size(), smallCapacity) << "缓存大小应该不超过容量限制";
    
    // 验证缓存仍然可用
    EXPECT_TRUE(cache->put("eviction_test", "eviction_value"));
    auto result = cache->get("eviction_test");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "eviction_value");
    
    auto stats = cache->getStatistics();
    EXPECT_GT(stats.hitCount + stats.missCount, 0) << "应该有访问统计";
    
    LOG_INFO("✅ 并发淘汰测试通过，最终缓存大小: {}", cache->size());
}

/**
 * @brief 测试读写一致性
 */
TEST_F(CacheSystemCoreTest, ReadWriteConsistency) {
    LOG_INFO("--- 测试读写一致性 ---");
    
    auto cache = createLRUCache<std::string, std::string>(200);
    ASSERT_NE(cache, nullptr);
    
    const int numOperations = 1000;
    const int numThreads = 6;
    std::vector<std::thread> threads;
    std::atomic<int> inconsistencyCount{0};
    
    // 共享的测试键集
    std::vector<std::string> sharedKeys;
    for (int i = 0; i < 100; ++i) {
        sharedKeys.push_back("shared_key_" + std::to_string(i));
    }
    
    // 启动多个线程进行读写操作
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&cache, &sharedKeys, &inconsistencyCount, t, numOperations]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> keyDis(0, sharedKeys.size() - 1);
            std::uniform_int_distribution<> opDis(0, 100);
            
            for (int i = 0; i < numOperations; ++i) {
                int keyIndex = keyDis(gen);
                std::string key = sharedKeys[keyIndex];
                std::string expectedValue = "thread_" + std::to_string(t) + "_value_" + std::to_string(i);
                
                int operation = opDis(gen);
                
                if (operation < 60) {
                    // 60% 概率执行写操作
                    cache->put(key, expectedValue);
                    
                    // 立即验证写入的数据
                    auto readResult = cache->get(key);
                    if (readResult.has_value() && readResult.value() != expectedValue) {
                        // 检查是否是由于并发更新导致的不一致
                        // 这在多线程环境中是可能的，但应该很少发生
                        inconsistencyCount.fetch_add(1);
                    }
                } else {
                    // 40% 概率执行读操作
                    auto readResult = cache->get(key);
                    // 只测试读操作不会崩溃，不验证具体值（因为可能被其他线程修改）
                }
                
                // 偶尔执行删除操作
                if (operation == 95) {
                    cache->remove(key);
                }
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 验证最终状态
    EXPECT_LT(inconsistencyCount.load(), numOperations * numThreads * 0.01) 
        << "读写不一致次数过多: " << inconsistencyCount.load();
    
    // 验证缓存仍然功能正常
    cache->clear();
    EXPECT_EQ(cache->size(), 0);
    
    EXPECT_TRUE(cache->put("consistency_test", "consistency_value"));
    auto finalResult = cache->get("consistency_test");
    ASSERT_TRUE(finalResult.has_value());
    EXPECT_EQ(finalResult.value(), "consistency_value");
    
    LOG_INFO("✅ 读写一致性测试通过，不一致次数: {}", inconsistencyCount.load());
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    LOG_INFO("🚀 开始运行缓存系统核心测试套件");
    LOG_INFO("测试范围: 基础操作、策略实现、并发安全");
    
    int result = RUN_ALL_TESTS();
    
    LOG_INFO("🏁 缓存系统核心测试套件完成，结果: {}", result == 0 ? "✅ 成功" : "❌ 失败");
    
    return result;
} 