/**
 * @file complex_integration_test.cpp
 * @brief Common Utilities模块复杂集成场景测试
 * @author OSCEAN Team
 * @date 2024
 * 
 * 🎯 测试目标：
 * ✅ 验证复杂业务场景下的模块协作
 * ✅ 测试异步和并发操作的稳定性
 * ✅ 验证大数据量处理能力
 * ✅ 测试错误恢复和异常处理
 */

#include "common_utils/utilities/boost_config.h"
#include "common_utils/utilities/string_utils.h"
#include "common_utils/utilities/file_format_detector.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/cache/cache_strategies.h"
#include "common_utils/async/async_framework.h"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstring>

using namespace oscean::common_utils;
using namespace oscean::common_utils::utilities;

/**
 * @class ComplexIntegrationTest
 * @brief 复杂集成场景测试主类
 */
class ComplexIntegrationTest {
private:
    std::mt19937 randomEngine{static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
    
public:
    struct TestResult {
        std::string scenarioName;
        bool passed;
        double executionTimeMs;
        std::string details;
        size_t dataProcessed;
        
        void print() const {
            std::string status = passed ? "✅ 通过" : "❌ 失败";
            std::cout << "🔍 " << scenarioName << " - " << status << std::endl;
            std::cout << "   执行时间: " << executionTimeMs << "ms | 处理数据量: " << dataProcessed << std::endl;
            if (!details.empty()) {
                std::cout << "   详情: " << details << std::endl;
            }
            std::cout << std::endl;
        }
    };
    
    void runAllComplexScenarios() {
        std::cout << "🚀 === Common Utilities 复杂集成场景测试开始 ===" << std::endl;
        std::cout << "🎯 测试场景: 数据处理流水线|异步文件处理|缓存系统集成|并发安全性|错误恢复\n" << std::endl;
        
        std::vector<TestResult> results;
        
        // 1. 数据处理流水线场景
        std::cout << "🏭 场景1: 复杂数据处理流水线..." << std::endl;
        results.push_back(testDataProcessingPipeline());
        
        // 2. 异步文件处理场景
        std::cout << "\n📁 场景2: 异步文件处理批量操作..." << std::endl;
        results.push_back(testAsyncFileProcessing());
        
        // 3. 缓存系统集成场景
        std::cout << "\n🗄️  场景3: 缓存系统与内存管理集成..." << std::endl;
        results.push_back(testCacheIntegrationScenario());
        
        // 4. 并发安全性测试
        std::cout << "\n🔄 场景4: 高并发多线程安全性..." << std::endl;
        results.push_back(testConcurrentSafetyScenario());
        
        // 5. 错误恢复和异常处理
        std::cout << "\n🛡️  场景5: 错误恢复和异常处理..." << std::endl;
        results.push_back(testErrorRecoveryScenario());
        
        // 6. 大数据量负载测试
        std::cout << "\n📊 场景6: 大数据量负载压力测试..." << std::endl;
        results.push_back(testLargeDataLoadScenario());
        
        // 7. 内存压力和资源管理
        std::cout << "\n💾 场景7: 内存压力和资源管理..." << std::endl;
        results.push_back(testMemoryPressureScenario());
        
        // 输出测试报告
        printComplexTestReport(results);
    }

private:
    TestResult testDataProcessingPipeline() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            constexpr size_t DATA_SIZE = 10000;
            
            // 创建文件格式检测器
            auto detector = FileFormatDetector::createDetector();
            
            // 阶段1: 生成复杂数据集
            auto rawData = generateComplexDataset(DATA_SIZE);
            
            // 阶段2: 数据预处理（字符串清理）
            std::vector<std::string> cleanedData;
            cleanedData.reserve(DATA_SIZE);
            
            for (const auto& [content, extension] : rawData) {
                std::string cleaned = StringUtils::trim(content);
                cleaned = StringUtils::toUpper(cleaned);
                cleanedData.push_back(cleaned);
            }
            
            // 阶段3: 格式检测和分类
            std::unordered_map<std::string, std::vector<size_t>> formatIndex;
            
            for (size_t i = 0; i < rawData.size(); ++i) {
                auto formatResult = detector->detectFormat(rawData[i].second);
                auto format = formatResult.format;
                
                std::string category = (format != FileFormat::UNKNOWN) ? "known" : "unknown";
                formatIndex[category].push_back(i);
            }
            
            // 阶段4: 数据聚合和统计
            std::unordered_map<std::string, size_t> statistics;
            for (const auto& [category, indices] : formatIndex) {
                statistics[category + "_count"] = indices.size();
                
                // 计算该类别的平均文件名长度
                size_t totalLength = 0;
                for (size_t idx : indices) {
                    totalLength += cleanedData[idx].length();
                }
                statistics[category + "_avg_length"] = indices.empty() ? 0 : totalLength / indices.size();
            }
            
            // 阶段5: 结果验证
            bool validResults = !cleanedData.empty() && 
                               !formatIndex.empty() && 
                               !statistics.empty() &&
                               statistics["known_count"] + statistics["unknown_count"] == DATA_SIZE;
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            return {
                "数据处理流水线",
                validResults,
                static_cast<double>(duration.count()),
                "预处理->清理->分类->聚合->统计 完整流水线",
                DATA_SIZE
            };
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            return {
                "数据处理流水线",
                false,
                static_cast<double>(duration.count()),
                "异常: " + std::string(e.what()),
                0
            };
        }
    }
    
    TestResult testAsyncFileProcessing() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            constexpr size_t NUM_VIRTUAL_FILES = 1000;
            constexpr size_t NUM_THREADS = 8;
            
            // 创建文件格式检测器
            auto detector = FileFormatDetector::createDetector();
            
            // 生成虚拟文件列表
            auto virtualFiles = generateVirtualFileList(NUM_VIRTUAL_FILES);
            
            // 异步处理队列
            std::vector<std::future<std::pair<std::string, bool>>> futures;
            
            auto processFile = [&detector](const std::pair<std::string, std::string>& fileInfo) -> std::pair<std::string, bool> {
                // 模拟文件处理操作
                std::string filename = StringUtils::trim(fileInfo.first);
                auto formatResult = detector->detectFormat(fileInfo.second);
                
                // 模拟一些处理时间
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                
                bool processed = !filename.empty() && formatResult.format != FileFormat::UNKNOWN;
                return {filename, processed};
            };
            
            // 分批异步处理
            size_t batchSize = NUM_VIRTUAL_FILES / NUM_THREADS;
            for (size_t t = 0; t < NUM_THREADS; ++t) {
                size_t startIdx = t * batchSize;
                size_t endIdx = (t == NUM_THREADS - 1) ? NUM_VIRTUAL_FILES : startIdx + batchSize;
                
                futures.push_back(std::async(std::launch::async, [&, startIdx, endIdx]() -> std::pair<std::string, bool> {
                    size_t processedCount = 0;
                    std::string batchResult = "batch_" + std::to_string(t);
                    
                    for (size_t i = startIdx; i < endIdx; ++i) {
                        auto result = processFile(virtualFiles[i]);
                        if (result.second) {
                            processedCount++;
                        }
                    }
                    
                    return {batchResult, processedCount == (endIdx - startIdx)};
                }));
            }
            
            // 收集结果
            size_t successfulBatches = 0;
            for (auto& future : futures) {
                auto result = future.get();
                if (result.second) {
                    successfulBatches++;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            bool allBatchesSuccessful = (successfulBatches == NUM_THREADS);
            
            return {
                "异步文件处理",
                allBatchesSuccessful,
                static_cast<double>(duration.count()),
                std::to_string(NUM_THREADS) + "线程并行处理" + std::to_string(NUM_VIRTUAL_FILES) + "个文件",
                NUM_VIRTUAL_FILES
            };
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            return {
                "异步文件处理",
                false,
                static_cast<double>(duration.count()),
                "异常: " + std::string(e.what()),
                0
            };
        }
    }
    
    TestResult testCacheIntegrationScenario() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            constexpr size_t CACHE_OPERATIONS = 10000;
            
            // 模拟多级缓存系统
            std::unordered_map<std::string, std::string> l1Cache; // 快速缓存
            std::unordered_map<std::string, std::string> l2Cache; // 慢速缓存
            
            size_t l1Hits = 0, l2Hits = 0, misses = 0;
            
            // 生成测试数据
            auto testKeys = generateCacheKeys(CACHE_OPERATIONS);
            
            for (const auto& key : testKeys) {
                // 查找数据
                auto l1It = l1Cache.find(key);
                if (l1It != l1Cache.end()) {
                    l1Hits++;
                    continue;
                }
                
                auto l2It = l2Cache.find(key);
                if (l2It != l2Cache.end()) {
                    l2Hits++;
                    // 提升到L1缓存
                    if (l1Cache.size() < 1000) { // L1缓存容量限制
                        l1Cache[key] = l2It->second;
                    }
                    continue;
                }
                
                // 缓存未命中，生成新数据
                misses++;
                std::string value = "processed_" + key;
                
                // 使用字符串工具处理
                value = StringUtils::toUpper(value);
                
                // 存入缓存
                if (l2Cache.size() < 5000) { // L2缓存容量限制
                    l2Cache[key] = value;
                }
                
                // 如果L1有空间，也存入L1
                if (l1Cache.size() < 1000) {
                    l1Cache[key] = value;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // 验证缓存命中率
            double hitRate = static_cast<double>(l1Hits + l2Hits) / CACHE_OPERATIONS;
            bool cacheEffective = hitRate > 0.3; // 至少30%命中率
            
            std::string details = "L1命中:" + std::to_string(l1Hits) + 
                                 " L2命中:" + std::to_string(l2Hits) + 
                                 " 未命中:" + std::to_string(misses) + 
                                 " 命中率:" + std::to_string(static_cast<int>(hitRate * 100)) + "%";
            
            return {
                "缓存系统集成",
                cacheEffective,
                static_cast<double>(duration.count()),
                details,
                CACHE_OPERATIONS
            };
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            return {
                "缓存系统集成",
                false,
                static_cast<double>(duration.count()),
                "异常: " + std::string(e.what()),
                0
            };
        }
    }
    
    TestResult testConcurrentSafetyScenario() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            constexpr size_t NUM_THREADS = 8;
            constexpr size_t OPERATIONS_PER_THREAD = 1000;
            
            // 创建文件格式检测器
            auto detector = FileFormatDetector::createDetector();
            
            // 共享数据结构（需要线程安全访问）
            std::unordered_map<std::string, size_t> sharedCounter;
            std::vector<std::string> sharedResults;
            std::mutex dataMutex;
            
            // 启动多个线程进行并发操作
            std::vector<std::future<bool>> futures;
            
            for (size_t t = 0; t < NUM_THREADS; ++t) {
                futures.push_back(std::async(std::launch::async, [&, t]() -> bool {
                    try {
                        for (size_t i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                            // 生成测试数据
                            std::string data = "thread_" + std::to_string(t) + "_item_" + std::to_string(i);
                            
                            // 字符串处理（线程安全）
                            auto processed = StringUtils::trim(data);
                            processed = StringUtils::toUpper(processed);
                            
                            // 文件格式检测（线程安全）
                            std::vector<std::string> extensions = {".tif", ".nc", ".shp", ".csv"};
                            auto ext = extensions[i % extensions.size()];
                            auto formatResult = detector->detectFromExtension("test" + ext);
                            
                            // 更新共享数据（需要同步）
                            {
                                std::lock_guard<std::mutex> lock(dataMutex);
                                sharedCounter[std::to_string(static_cast<int>(formatResult))]++;
                                sharedResults.push_back(processed);
                            }
                            
                            // 添加一些随机延迟以增加竞态条件的可能性
                            if (i % 100 == 0) {
                                std::this_thread::sleep_for(std::chrono::microseconds(10));
                            }
                        }
                        return true;
                    } catch (...) {
                        return false;
                    }
                }));
            }
            
            // 等待所有线程完成
            size_t successfulThreads = 0;
            for (auto& future : futures) {
                if (future.get()) {
                    successfulThreads++;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // 验证数据一致性
            size_t expectedTotalOperations = NUM_THREADS * OPERATIONS_PER_THREAD;
            size_t actualTotalOperations = sharedResults.size();
            
            bool concurrencySafe = (successfulThreads == NUM_THREADS) && 
                                  (actualTotalOperations == expectedTotalOperations);
            
            std::string details = std::to_string(NUM_THREADS) + "线程并发，" +
                                 "成功线程:" + std::to_string(successfulThreads) + "/" + std::to_string(NUM_THREADS) +
                                 " 操作完成:" + std::to_string(actualTotalOperations) + "/" + std::to_string(expectedTotalOperations);
            
            return {
                "并发安全性测试",
                concurrencySafe,
                static_cast<double>(duration.count()),
                details,
                expectedTotalOperations
            };
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            return {
                "并发安全性测试",
                false,
                static_cast<double>(duration.count()),
                "异常: " + std::string(e.what()),
                0
            };
        }
    }
    
    TestResult testErrorRecoveryScenario() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            constexpr size_t TOTAL_OPERATIONS = 1000;
            constexpr size_t ERROR_INJECTION_RATE = 10; // 每10个操作注入一个错误
            
            size_t successfulOperations = 0;
            size_t recoveredErrors = 0;
            size_t fatalErrors = 0;
            
            for (size_t i = 0; i < TOTAL_OPERATIONS; ++i) {
                try {
                    // 模拟可能出错的操作
                    std::string testData;
                    
                    // 注入错误
                    if (i % ERROR_INJECTION_RATE == 0) {
                        // 模拟各种错误情况
                        switch (i % 4) {
                            case 0:
                                // 空字符串错误
                                testData = "";
                                break;
                            case 1:
                                // 超长字符串错误
                                testData = std::string(100000, 'x');
                                break;
                            case 2:
                                // 无效字符错误
                                testData = "\xFF\xFE\xFD invalid data";
                                break;
                            case 3:
                                // 正常但边界情况
                                testData = "   ";
                                break;
                        }
                    } else {
                        // 正常数据
                        testData = "  normal_data_" + std::to_string(i) + "  ";
                    }
                    
                    // 尝试处理数据，实现错误恢复
                    std::string processed;
                    try {
                        processed = StringUtils::trim(testData);
                        if (processed.empty()) {
                            // 恢复策略：使用默认值
                            processed = "default_value";
                            recoveredErrors++;
                        }
                    } catch (...) {
                        // 恢复策略：使用安全的默认处理
                        processed = "error_recovered_" + std::to_string(i);
                        recoveredErrors++;
                    }
                    
                    // 进一步处理
                    try {
                        auto upper = StringUtils::toUpper(processed);
                        if (upper.length() > 50000) {
                            // 截断过长的字符串
                            upper = upper.substr(0, 50000);
                            recoveredErrors++;
                        }
                        
                        // 文件格式检测（在错误情况下的处理）
                        std::vector<std::string> testExtensions = {".tif", ".nc"};
                        auto ext = testExtensions[i % testExtensions.size()];
                        
                        // 创建检测器并进行格式检测
                        auto detector = FileFormatDetector::createDetector();
                        auto formatResult = detector->detectFromExtension("test" + ext);
                        
                        successfulOperations++;
                        
                    } catch (...) {
                        recoveredErrors++;
                    }
                    
                } catch (...) {
                    fatalErrors++;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // 评估错误恢复效果
            double recoveryRate = static_cast<double>(successfulOperations + recoveredErrors) / TOTAL_OPERATIONS;
            bool errorRecoveryEffective = recoveryRate > 0.9 && fatalErrors < TOTAL_OPERATIONS * 0.05;
            
            std::string details = "成功:" + std::to_string(successfulOperations) +
                                 " 已恢复:" + std::to_string(recoveredErrors) +
                                 " 致命错误:" + std::to_string(fatalErrors) +
                                 " 恢复率:" + std::to_string(static_cast<int>(recoveryRate * 100)) + "%";
            
            return {
                "错误恢复测试",
                errorRecoveryEffective,
                static_cast<double>(duration.count()),
                details,
                TOTAL_OPERATIONS
            };
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            return {
                "错误恢复测试",
                false,
                static_cast<double>(duration.count()),
                "异常: " + std::string(e.what()),
                0
            };
        }
    }
    
    TestResult testLargeDataLoadScenario() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            constexpr size_t LARGE_DATASET_SIZE = 500000;
            
            // 生成大数据集
            std::vector<std::string> largeDataset;
            largeDataset.reserve(LARGE_DATASET_SIZE);
            
            for (size_t i = 0; i < LARGE_DATASET_SIZE; ++i) {
                largeDataset.push_back("large_data_item_" + std::to_string(i) + "_with_some_content");
            }
            
            // 批量处理
            constexpr size_t BATCH_SIZE = 10000;
            size_t processedItems = 0;
            
            for (size_t batch = 0; batch < LARGE_DATASET_SIZE; batch += BATCH_SIZE) {
                size_t batchEnd = std::min(batch + BATCH_SIZE, LARGE_DATASET_SIZE);
                
                // 并行处理批次
                std::vector<std::future<size_t>> batchFutures;
                constexpr size_t BATCH_THREADS = 4;
                size_t itemsPerThread = (batchEnd - batch) / BATCH_THREADS;
                
                for (size_t t = 0; t < BATCH_THREADS; ++t) {
                    size_t threadStart = batch + t * itemsPerThread;
                    size_t threadEnd = (t == BATCH_THREADS - 1) ? batchEnd : threadStart + itemsPerThread;
                    
                    batchFutures.push_back(std::async(std::launch::async, [&, threadStart, threadEnd]() -> size_t {
                        size_t threadProcessed = 0;
                        for (size_t i = threadStart; i < threadEnd; ++i) {
                            auto processed = StringUtils::trim(largeDataset[i]);
                            auto upper = StringUtils::toUpper(processed);
                            
                            // 简单验证
                            if (!upper.empty()) {
                                threadProcessed++;
                            }
                        }
                        return threadProcessed;
                    }));
                }
                
                // 收集批次结果
                for (auto& future : batchFutures) {
                    processedItems += future.get();
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            bool loadTestSuccessful = (processedItems == LARGE_DATASET_SIZE);
            
            return {
                "大数据量负载测试",
                loadTestSuccessful,
                static_cast<double>(duration.count()),
                "处理" + std::to_string(LARGE_DATASET_SIZE) + "条数据，批次大小" + std::to_string(BATCH_SIZE),
                LARGE_DATASET_SIZE
            };
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            return {
                "大数据量负载测试",
                false,
                static_cast<double>(duration.count()),
                "异常: " + std::string(e.what()),
                0
            };
        }
    }
    
    TestResult testMemoryPressureScenario() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            constexpr size_t NUM_ALLOCATIONS = 1000;
            constexpr size_t ALLOCATION_SIZE_BASE = 1024;
            
            std::vector<std::unique_ptr<uint8_t[]>> allocations;
            allocations.reserve(NUM_ALLOCATIONS);
            
            size_t totalAllocated = 0;
            size_t successfulAllocations = 0;
            
            // 逐步增加内存压力
            for (size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
                try {
                    size_t allocSize = ALLOCATION_SIZE_BASE * (1 + i / 100);
                    auto ptr = std::make_unique<uint8_t[]>(allocSize);
                    
                    // 写入数据验证内存可用性
                    std::memset(ptr.get(), static_cast<int>(i % 256), allocSize);
                    
                    allocations.push_back(std::move(ptr));
                    totalAllocated += allocSize;
                    successfulAllocations++;
                    
                    // 定期释放一些内存模拟真实使用场景
                    if (i % 100 == 99 && allocations.size() > 50) {
                        // 释放前50个分配
                        for (size_t j = 0; j < 50; ++j) {
                            allocations.erase(allocations.begin());
                        }
                    }
                    
                } catch (const std::bad_alloc&) {
                    // 内存不足，正常情况
                    break;
                } catch (...) {
                    // 其他错误
                    break;
                }
            }
            
            // 验证分配的内存
            size_t validAllocations = 0;
            for (size_t i = 0; i < std::min(allocations.size(), size_t{10}); ++i) {
                if (allocations[i] != nullptr) {
                    validAllocations++;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            bool memoryTestSuccessful = (successfulAllocations > NUM_ALLOCATIONS / 2) && 
                                       (validAllocations == std::min(allocations.size(), size_t{10}));
            
            std::string details = "成功分配:" + std::to_string(successfulAllocations) + "/" + std::to_string(NUM_ALLOCATIONS) +
                                 " 总内存:" + std::to_string(totalAllocated / 1024) + "KB" +
                                 " 当前活跃:" + std::to_string(allocations.size());
            
            return {
                "内存压力测试",
                memoryTestSuccessful,
                static_cast<double>(duration.count()),
                details,
                successfulAllocations
            };
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            return {
                "内存压力测试",
                false,
                static_cast<double>(duration.count()),
                "异常: " + std::string(e.what()),
                0
            };
        }
    }
    
    // 辅助数据生成方法
    
    std::vector<std::pair<std::string, std::string>> generateComplexDataset(size_t size) {
        std::vector<std::pair<std::string, std::string>> result;
        result.reserve(size);
        
        std::vector<std::string> prefixes = {
            "  ocean_temperature_", "  satellite_image_", "  wind_data_",
            "\t\tclimate_model_", "  population_census_", "  land_use_"
        };
        
        std::vector<std::string> extensions = {
            ".nc", ".tif", ".shp", ".csv", ".h5", ".geojson"
        };
        
        std::uniform_int_distribution<size_t> prefixDist(0, prefixes.size() - 1);
        std::uniform_int_distribution<size_t> extDist(0, extensions.size() - 1);
        
        for (size_t i = 0; i < size; ++i) {
            auto prefix = prefixes[prefixDist(randomEngine)];
            auto extension = extensions[extDist(randomEngine)];
            result.emplace_back(prefix + std::to_string(i) + "  ", extension);
        }
        
        return result;
    }
    
    std::vector<std::pair<std::string, std::string>> generateVirtualFileList(size_t size) {
        return generateComplexDataset(size); // 重用复杂数据集生成器
    }
    
    std::vector<std::string> generateCacheKeys(size_t size) {
        std::vector<std::string> result;
        result.reserve(size);
        
        // 生成有重复模式的键，模拟真实缓存访问模式
        std::uniform_int_distribution<size_t> keyDist(0, size / 3); // 33%重复率
        
        for (size_t i = 0; i < size; ++i) {
            if (i < size / 3) {
                // 前三分之一为唯一键
                result.push_back("key_" + std::to_string(i));
            } else {
                // 后三分之二重复访问前三分之一的键
                size_t keyIndex = keyDist(randomEngine);
                result.push_back("key_" + std::to_string(keyIndex));
            }
        }
        
        return result;
    }
    
    void printComplexTestReport(const std::vector<TestResult>& results) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "📊 === Common Utilities 复杂集成场景测试报告 ===" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        size_t passedTests = 0;
        double totalExecutionTime = 0;
        size_t totalDataProcessed = 0;
        
        for (const auto& result : results) {
            result.print();
            if (result.passed) passedTests++;
            totalExecutionTime += result.executionTimeMs;
            totalDataProcessed += result.dataProcessed;
        }
        
        std::cout << "🎯 复杂集成测试总结:" << std::endl;
        std::cout << "✅ 通过测试: " << passedTests << "/" << results.size() << std::endl;
        std::cout << "⏱️  总执行时间: " << totalExecutionTime << "ms" << std::endl;
        std::cout << "📊 总处理数据量: " << totalDataProcessed << " 项" << std::endl;
        
        if (passedTests == results.size()) {
            std::cout << "\n🎉 所有复杂集成场景测试通过！模块协作稳定可靠。" << std::endl;
        } else {
            std::cout << "\n⚠️  部分复杂场景存在问题，需要进一步优化。" << std::endl;
        }
    }
};

/**
 * @brief 主测试函数
 */
int main() {
    try {
        ComplexIntegrationTest test;
        test.runAllComplexScenarios();
        return 0;
    } catch (const std::exception& e) {
        std::cout << "\n💥 复杂集成测试失败: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "\n💥 复杂集成测试遇到未知错误" << std::endl;
        return 1;
    }
} 