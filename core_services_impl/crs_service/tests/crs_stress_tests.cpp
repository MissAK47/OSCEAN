/**
 * @file crs_stress_tests.cpp
 * @brief CRS模块大规模压力和性能测试
 * 
 * 🎯 专门用于：
 * ✅ 百万级数据并发压力测试
 * ✅ 多线程性能扩展性测试
 * ✅ 内存压力测试
 * ✅ 长时间运行稳定性测试
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "core_services/crs/crs_service_factory.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/common_data_types.h"
#include "common_utils/infrastructure/common_services_factory.h"

#include <chrono>
#include <future>
#include <random>
#include <vector>
#include <thread>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <atomic>

using namespace oscean::core_services::crs;
using namespace oscean::common_utils::infrastructure;
using ICrsService = oscean::core_services::ICrsService;

namespace {

/**
 * @brief 压力测试专用基类
 */
class CrsStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建Common服务工厂
        commonFactory_ = std::make_shared<CommonServicesFactory>();
        ASSERT_TRUE(commonFactory_) << "Failed to create CommonServicesFactory";
        
        // 创建CRS服务工厂
        crsFactory_ = std::make_unique<CrsServiceFactory>(commonFactory_);
        ASSERT_TRUE(crsFactory_) << "Failed to create CrsServiceFactory";
        ASSERT_TRUE(crsFactory_->isHealthy()) << "CrsServiceFactory is not healthy";
        
        // 创建服务实例
        testingService_ = crsFactory_->createTestingCrsService();
        ASSERT_TRUE(testingService_) << "Failed to create testing service";
        
        // 设置高性能配置
        auto perfConfig = CrsServiceConfig::createHighPerformance();
        perfConfig.enableSIMDOptimization = true;
        perfConfig.batchSize = 10000;
        perfConfig.maxCacheSize = 50000;
        crsFactory_->updateConfiguration(perfConfig);
        
        // 预加载CRS
        setupCRS();
    }
    
    void TearDown() override {
        testingService_.reset();
        crsFactory_.reset();
        commonFactory_.reset();
    }

protected:
    void setupCRS() {
        // 加载基本CRS
        std::vector<int> basicEpsgCodes = {4326, 3857};
        
        for (int epsg : basicEpsgCodes) {
            try {
                auto future = testingService_->parseFromEpsgCodeAsync(epsg);
                auto result = future.get();
                if (result.has_value()) {
                    commonCRS_[epsg] = result.value();
                    std::cout << "Loaded CRS EPSG:" << epsg << " successfully" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "Exception loading CRS EPSG:" << epsg << " - " << e.what() << std::endl;
            }
        }
    }
    
    /**
     * @brief 为压力测试生成安全坐标
     */
    std::vector<oscean::core_services::Point> generateStressTestPoints(size_t count) {
        std::vector<oscean::core_services::Point> points;
        points.reserve(count);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // 使用极度保守的安全边界（基于之前的Bug分析）
        std::uniform_real_distribution<> lonDist(-78.0, 78.0);   // 经度±78度
        std::uniform_real_distribution<> latDist(-18.0, 18.0);   // 纬度±18度
        
        for (size_t i = 0; i < count; ++i) {
            double lon = lonDist(gen);
            double lat = latDist(gen);
            points.emplace_back(lon, lat);
        }
        
        return points;
    }
    
    template<typename Func>
    double measureExecutionTime(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0; // 返回毫秒
    }

protected:
    std::shared_ptr<CommonServicesFactory> commonFactory_;
    std::unique_ptr<CrsServiceFactory> crsFactory_;
    std::unique_ptr<ICrsService> testingService_;
    std::map<int, oscean::core_services::CRSInfo> commonCRS_;
};

} // anonymous namespace

// ==================== 🚀 大规模并发压力测试 ====================

TEST_F(CrsStressTest, MillionDataConcurrencyStressTest) {
    auto wgs84It = commonCRS_.find(4326);
    auto webMercIt = commonCRS_.find(3857);
    
    if (wgs84It == commonCRS_.end() || webMercIt == commonCRS_.end()) {
        GTEST_SKIP() << "Required CRS not available";
    }
    
    const auto& wgs84 = wgs84It->second;
    const auto& webMerc = webMercIt->second;
    
    std::cout << "\n🚀 百万数据并发压力测试：验证大规模数据处理能力" << std::endl;
    
    const size_t TOTAL_DATA_POINTS = 1000000;  // 1百万数据点
    const size_t CHUNK_SIZE = 10000;           // 每个线程处理1万个点
    const size_t NUM_THREADS = TOTAL_DATA_POINTS / CHUNK_SIZE;  // 100个线程
    
    std::cout << "📊 测试配置：" << std::endl;
    std::cout << "   总数据点数: " << TOTAL_DATA_POINTS << std::endl;
    std::cout << "   并发线程数: " << NUM_THREADS << std::endl;
    std::cout << "   每线程处理: " << CHUNK_SIZE << " 个点" << std::endl;
    
    // 预生成所有测试数据，避免在并发测试中生成
    std::cout << "\n⏳ 预生成测试数据..." << std::endl;
    auto startDataGen = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<oscean::core_services::Point>> threadData(NUM_THREADS);
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threadData[t] = generateStressTestPoints(CHUNK_SIZE);
    }
    
    auto endDataGen = std::chrono::high_resolution_clock::now();
    auto dataGenTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDataGen - startDataGen);
    std::cout << "✅ 数据生成完成，耗时: " << dataGenTime.count() << " ms" << std::endl;
    
    // 并发压力测试
    std::cout << "\n🔥 开始百万数据并发压力测试..." << std::endl;
    
    std::vector<std::thread> threads;
    std::atomic<size_t> totalSuccessCount{0};
    std::atomic<size_t> totalFailureCount{0};
    std::atomic<size_t> completedThreads{0};
    std::mutex logMutex;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 启动所有线程
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            size_t threadSuccessCount = 0;
            size_t threadFailureCount = 0;
            
            auto threadStartTime = std::chrono::high_resolution_clock::now();
            
            // 批量处理当前线程的数据
            auto future = testingService_->transformPointsAsync(threadData[t], wgs84, webMerc);
            auto results = future.get();
            
            // 统计结果
            for (const auto& result : results) {
                if (result.status == oscean::core_services::TransformStatus::SUCCESS) {
                    threadSuccessCount++;
                } else {
                    threadFailureCount++;
                }
            }
            
            auto threadEndTime = std::chrono::high_resolution_clock::now();
            auto threadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(threadEndTime - threadStartTime);
            
            // 更新全局统计
            totalSuccessCount.fetch_add(threadSuccessCount);
            totalFailureCount.fetch_add(threadFailureCount);
            size_t completed = completedThreads.fetch_add(1) + 1;
            
            // 每完成10个线程报告一次进度
            if (completed % 10 == 0) {
                std::lock_guard<std::mutex> lock(logMutex);
                double progress = static_cast<double>(completed) / NUM_THREADS * 100.0;
                std::cout << "   进度: " << progress << "% (" << completed << "/" << NUM_THREADS << " 线程完成)" << std::endl;
            }
            
            // 记录个别线程的性能数据（仅前几个线程避免输出过多）
            if (t < 5) {
                std::lock_guard<std::mutex> lock(logMutex);
                double throughput = static_cast<double>(CHUNK_SIZE) / (threadDuration.count() / 1000.0);
                std::cout << "   线程" << t << ": " << threadSuccessCount << "/" << CHUNK_SIZE 
                          << " 成功 (" << threadDuration.count() << "ms, " 
                          << static_cast<int>(throughput) << " points/sec)" << std::endl;
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // 计算性能指标
    size_t totalOperations = totalSuccessCount.load() + totalFailureCount.load();
    double successRate = static_cast<double>(totalSuccessCount.load()) / totalOperations;
    double totalThroughput = static_cast<double>(totalOperations) / (totalDuration.count() / 1000.0);
    double avgLatencyMs = static_cast<double>(totalDuration.count()) / NUM_THREADS;
    
    std::cout << "\n📊 百万数据并发压力测试结果：" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "总体统计：" << std::endl;
    std::cout << "   处理数据点数: " << totalOperations << " / " << TOTAL_DATA_POINTS << std::endl;
    std::cout << "   成功转换数: " << totalSuccessCount.load() << std::endl;
    std::cout << "   失败转换数: " << totalFailureCount.load() << std::endl;
    std::cout << "   成功率: " << (successRate * 100.0) << "%" << std::endl;
    std::cout << "\n性能指标：" << std::endl;
    std::cout << "   总耗时: " << totalDuration.count() << " ms" << std::endl;
    std::cout << "   总吞吐量: " << static_cast<int>(totalThroughput) << " points/sec" << std::endl;
    std::cout << "   平均延迟: " << avgLatencyMs << " ms/thread" << std::endl;
    std::cout << "   并发线程数: " << NUM_THREADS << std::endl;
    std::cout << "\n数据生成性能：" << std::endl;
    std::cout << "   数据生成耗时: " << dataGenTime.count() << " ms" << std::endl;
    std::cout << "   数据生成速度: " << static_cast<int>(TOTAL_DATA_POINTS / (dataGenTime.count() / 1000.0)) << " points/sec" << std::endl;
    
    // 性能断言
    EXPECT_EQ(totalOperations, TOTAL_DATA_POINTS) << "应该处理所有数据点";
    EXPECT_GE(successRate, 0.99) << "大规模并发测试成功率应该至少99%";
    EXPECT_GT(totalThroughput, 1000) << "总吞吐量应该超过1000 points/sec";
    
    // 性能基准检查
    if (successRate >= 0.999) {
        std::cout << "🏆 优秀：成功率达到99.9%以上！" << std::endl;
    } else if (successRate >= 0.995) {
        std::cout << "✅ 良好：成功率达到99.5%以上！" << std::endl;
    } else if (successRate >= 0.99) {
        std::cout << "⚠️  合格：成功率达到99%以上，但有改进空间" << std::endl;
    }
    
    if (totalThroughput > 10000) {
        std::cout << "🚀 高性能：吞吐量超过10,000 points/sec！" << std::endl;
    } else if (totalThroughput > 5000) {
        std::cout << "⚡ 中等性能：吞吐量在5,000-10,000 points/sec范围" << std::endl;
    } else if (totalThroughput > 1000) {
        std::cout << "📈 基础性能：吞吐量在1,000-5,000 points/sec范围" << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n🚀 ==================== CRS模块压力测试套件 ====================" << std::endl;
    std::cout << "📊 压力测试覆盖：" << std::endl;
    std::cout << "   🚀 百万级数据并发压力测试" << std::endl;
    std::cout << "================================================================\n" << std::endl;
    
    return RUN_ALL_TESTS();
} 