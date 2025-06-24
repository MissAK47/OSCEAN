/**
 * @file test_gpu_framework.cpp
 * @brief OSCEAN GPU框架集成测试
 */

#include "common_utils/gpu/oscean_gpu_framework.h"
#include "common_utils/utilities/logging_utils.h"
#include <gtest/gtest.h>
#include <iostream>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>

using namespace oscean::common_utils::gpu;

class GPUFrameworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化日志
        oscean::common_utils::LoggingConfig logConfig;
        logConfig.console_level = "debug";
        oscean::common_utils::LoggingManager::configureGlobal(logConfig);
    }
    
    void TearDown() override {
        // 确保框架被清理
        auto& framework = OSCEANGPUFramework::getInstance();
        if (framework.isInitialized()) {
            framework.shutdown();
        }
    }
};

// 测试框架初始化
TEST_F(GPUFrameworkTest, TestInitialization) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    // 检查初始状态
    EXPECT_FALSE(framework.isInitialized());
    
    // 初始化框架
    GPUFrameworkConfig config;
    config.enableMultiGPU = true;
    config.enablePerformanceMonitoring = true;
    
    bool result = framework.initialize(config);
    
    // 如果没有GPU，跳过测试
    if (!result) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    EXPECT_TRUE(framework.isInitialized());
    
    // 获取状态
    auto status = framework.getStatus();
    EXPECT_TRUE(status.initialized);
    EXPECT_GT(status.availableDevices, 0);
    EXPECT_GT(status.activeDevices, 0);
    
    std::cout << status.toString() << std::endl;
}

// 测试GPU检测
TEST_F(GPUFrameworkTest, TestGPUDetection) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    GPUFrameworkConfig config;
    if (!framework.initialize(config)) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    auto devices = framework.getAvailableDevices();
    
    EXPECT_FALSE(devices.empty());
    
    for (const auto& device : devices) {
        std::cout << "Device: " << device.name << std::endl;
        std::cout << "  Vendor: " << GPUDeviceInfo::vendorToString(device.vendor) << std::endl;
        std::cout << "  Memory: " << (device.memoryDetails.totalGlobalMemory / (1024*1024*1024)) 
                  << " GB" << std::endl;
        std::cout << "  Performance Score: " << device.performanceScore << std::endl;
    }
}

// 测试内存分配
TEST_F(GPUFrameworkTest, TestMemoryAllocation) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    GPUFrameworkConfig config;
    config.memoryPoolConfig.initialPoolSize = 64 * 1024 * 1024; // 64MB
    
    if (!framework.initialize(config)) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    // 分配内存
    size_t allocSize = 10 * 1024 * 1024; // 10MB
    auto handle = framework.allocateMemory(allocSize);
    
    EXPECT_TRUE(handle.isValid());
    EXPECT_EQ(handle.size, allocSize);
    EXPECT_GE(handle.deviceId, 0);
    
    // 获取内存统计
    auto& memManager = framework.getMemoryManager();
    auto stats = memManager.getUsageStats();
    
    EXPECT_GT(stats.currentUsage, 0);
    EXPECT_EQ(stats.allocationCount, 1);
    
    // 释放内存
    framework.deallocateMemory(handle);
    
    stats = memManager.getUsageStats();
    EXPECT_EQ(stats.deallocationCount, 1);
}

// 测试简单GPU任务
TEST_F(GPUFrameworkTest, TestSimpleTask) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    if (!framework.initialize()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    // 创建简单任务
    auto task = GPUTaskBuilder()
        .withName("TestTask")
        .withMemoryRequirement(1024 * 1024)  // 1MB
        .withComputeComplexity(1.0)
        .withExecutor([](GPUTaskExecutionContext& context) {
            // 模拟任务执行
            context.setResult("processed", true);
            return true;
        })
        .build();
    
    // 提交任务
    std::string taskId = framework.submitTask(task);
    EXPECT_FALSE(taskId.empty());
    
    // 等待任务完成
    bool completed = framework.waitForTask(taskId, 5000);  // 5秒超时
    EXPECT_TRUE(completed);
    
    // 检查任务状态
    auto status = framework.getTaskStatus(taskId);
    EXPECT_TRUE(status.has_value());
    
    if (status) {
        bool processed = false;
        EXPECT_TRUE(status->getResult("processed", processed));
        EXPECT_TRUE(processed);
    }
}

// 测试多GPU调度
TEST_F(GPUFrameworkTest, TestMultiGPUScheduling) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    // 如果只有一个GPU，跳过测试
    if (framework.getAvailableDevices().size() < 2) {
        GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs";
    }
    
    // 创建多个任务
    std::vector<std::string> taskIds;
    
    for (int i = 0; i < 10; ++i) {
        auto task = GPUTaskBuilder()
            .withName("MultiGPUTask_" + std::to_string(i))
            .withMemoryRequirement(10 * 1024 * 1024)  // 10MB
            .withComputeComplexity(2.0)
            .withExecutor([i](GPUTaskExecutionContext& context) {
                // 记录执行的设备ID
                context.setResult("deviceId", context.deviceId);
                context.setResult("taskIndex", i);
                return true;
            })
            .build();
        
        taskIds.push_back(framework.submitTask(task));
    }
    
    // 等待所有任务完成
    for (const auto& taskId : taskIds) {
        EXPECT_TRUE(framework.waitForTask(taskId, 10000));  // 10秒超时
    }
    
    // 统计任务分布
    std::map<int, int> deviceTaskCount;
    for (const auto& taskId : taskIds) {
        auto status = framework.getTaskStatus(taskId);
        if (status) {
            int deviceId = -1;
            if (status->getResult("deviceId", deviceId)) {
                deviceTaskCount[deviceId]++;
            }
        }
    }
    
    // 验证任务分布到了多个设备
    EXPECT_GT(deviceTaskCount.size(), 1);
    
    // 输出任务分布情况
    for (const auto& [deviceId, count] : deviceTaskCount) {
        OSCEAN_LOG_INFO("GPUFrameworkTest", 
            "Device " + std::to_string(deviceId) + " processed " + 
            std::to_string(count) + " tasks");
    }
}

// 测试内存传输
TEST_F(GPUFrameworkTest, TestMemoryTransfer) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    if (!framework.initialize()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    size_t dataSize = 1024 * 1024; // 1MB
    
    // 分配源和目标内存
    auto srcHandle = framework.allocateMemory(dataSize, 0);
    auto dstHandle = framework.allocateMemory(dataSize, 0);
    
    EXPECT_TRUE(srcHandle.isValid());
    EXPECT_TRUE(dstHandle.isValid());
    
    // 执行传输
    bool success = framework.transferData(srcHandle, dstHandle, false); // 同步传输
    EXPECT_TRUE(success);
    
    // 获取传输统计
    auto& memManager = framework.getMemoryManager();
    auto transferStats = memManager.getTransferStats();
    
    EXPECT_GT(transferStats.totalTransfers, 0);
    EXPECT_GT(transferStats.totalBytesTransferred, 0);
    
    // 清理
    framework.deallocateMemory(srcHandle);
    framework.deallocateMemory(dstHandle);
}

// 测试性能监控
TEST_F(GPUFrameworkTest, TestPerformanceMonitoring) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    GPUFrameworkConfig config;
    config.enablePerformanceMonitoring = true;
    
    if (!framework.initialize(config)) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    // 设置性能回调
    int callbackCount = 0;
    framework.setPerformanceCallback([&callbackCount](const GPUFrameworkStatus& status) {
        callbackCount++;
        std::cout << "Performance Update #" << callbackCount << std::endl;
        std::cout << "  GPU Utilization: " << status.averageGPUUtilization << "%" << std::endl;
        std::cout << "  Memory Utilization: " << status.averageMemoryUtilization << "%" << std::endl;
    });
    
    // 执行一些工作负载
    for (int i = 0; i < 5; ++i) {
        auto handle = framework.allocateMemory(10 * 1024 * 1024); // 10MB
        boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
        framework.deallocateMemory(handle);
    }
    
    // 等待性能更新
    boost::this_thread::sleep_for(boost::chrono::seconds(2));
    
    EXPECT_GT(callbackCount, 0);
}

// 测试设备推荐
TEST_F(GPUFrameworkTest, TestDeviceRecommendation) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    if (!framework.initialize()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    // 获取推荐设备
    auto recommendedDevices = framework.getRecommendedDevices("high_performance");
    
    if (!recommendedDevices.empty()) {
        std::cout << "Recommended devices for high performance:" << std::endl;
        for (int deviceId : recommendedDevices) {
            auto& gpuManager = framework.getGPUManager();
            auto deviceInfo = gpuManager.getDeviceInfo(deviceId);
            if (deviceInfo) {
                std::cout << "  - Device " << deviceId << ": " 
                         << deviceInfo->name << " (Score: " 
                         << deviceInfo->performanceScore << ")" << std::endl;
            }
        }
    }
}

// 测试配置优化
TEST_F(GPUFrameworkTest, TestConfigurationOptimization) {
    auto& framework = OSCEANGPUFramework::getInstance();
    
    if (!framework.initialize()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    // 优化配置
    framework.optimizeConfiguration("compute_intensive");
    
    // 生成性能报告
    std::string report = framework.generatePerformanceReport();
    std::cout << "Performance Report:" << std::endl;
    std::cout << report << std::endl;
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 