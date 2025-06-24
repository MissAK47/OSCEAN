/**
 * @file test_gpu_memory_manager.cpp
 * @brief GPU内存管理器测试
 */

#include <gtest/gtest.h>
#include "common_utils/gpu/oscean_gpu_framework.h"
#include "common_utils/utilities/logging_utils.h"
#include <vector>
#include <thread>

using namespace oscean::common_utils::gpu;

class GPUMemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化日志系统
        oscean::common_utils::LoggingConfig logConfig;
        logConfig.enable_file = true;
        logConfig.log_filename = "test_gpu_memory_manager.log";
        oscean::common_utils::LoggingManager::configureGlobal(logConfig);
        
        // 初始化GPU框架
        GPUFrameworkConfig config;
        config.enablePerformanceMonitoring = false; // 测试时禁用性能监控
        OSCEANGPUFramework::initialize(config);
    }
    
    void TearDown() override {
        OSCEANGPUFramework::shutdown();
    }
};

// 测试基本内存分配和释放
TEST_F(GPUMemoryManagerTest, BasicAllocation) {
    auto& memManager = OSCEANGPUFramework::getMemoryManager();
    
    // 检查是否有可用的GPU
    const auto& devices = OSCEANGPUFramework::getAvailableDevices();
    if (devices.empty()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    // 添加调试信息
    std::cout << "Available devices count: " << devices.size() << std::endl;
    for (const auto& device : devices) {
        std::cout << "  Device " << device.deviceId << ": " << device.name << std::endl;
    }
    
    // 分配1MB内存
    AllocationRequest request;
    request.size = 1024 * 1024; // 1MB
    request.preferredDeviceId = 0;
    request.memoryType = GPUMemoryType::DEVICE;
    
    auto handle = memManager.allocate(request);
    ASSERT_TRUE(handle.isValid());
    EXPECT_EQ(handle.size, request.size);
    EXPECT_GE(handle.deviceId, 0);
    
    // 检查内存统计
    auto stats = memManager.getUsageStats();
    EXPECT_GE(stats.currentUsage, request.size);
    EXPECT_GE(stats.totalAllocated, request.size);
    EXPECT_EQ(stats.allocationCount, 1);
    
    // 释放内存
    ASSERT_TRUE(memManager.deallocate(handle));
    
    // 再次检查统计
    stats = memManager.getUsageStats();
    EXPECT_EQ(stats.deallocationCount, 1);
}

// 测试多设备内存分配
TEST_F(GPUMemoryManagerTest, MultiDeviceAllocation) {
    auto& memManager = OSCEANGPUFramework::getMemoryManager();
    
    const auto& devices = OSCEANGPUFramework::getAvailableDevices();
    if (devices.size() < 2) {
        GTEST_SKIP() << "Less than 2 GPU devices available";
    }
    
    std::vector<GPUMemoryHandle> handles;
    
    // 在每个设备上分配内存
    for (size_t i = 0; i < devices.size(); ++i) {
        AllocationRequest request;
        request.size = 512 * 1024; // 512KB
        request.preferredDeviceId = static_cast<int>(i);
        
        auto handle = memManager.allocate(request);
        ASSERT_TRUE(handle.isValid());
        EXPECT_EQ(handle.deviceId, static_cast<int>(i));
        handles.push_back(handle);
    }
    
    // 释放所有内存
    for (const auto& handle : handles) {
        ASSERT_TRUE(memManager.deallocate(handle));
    }
}

// 测试内存池增长
TEST_F(GPUMemoryManagerTest, MemoryPoolGrowth) {
    auto& memManager = OSCEANGPUFramework::getMemoryManager();
    
    const auto& devices = OSCEANGPUFramework::getAvailableDevices();
    if (devices.empty()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    std::vector<GPUMemoryHandle> handles;
    const size_t blockSize = 1024 * 1024; // 1MB
    const size_t numBlocks = 10;
    
    // 分配多个内存块
    for (size_t i = 0; i < numBlocks; ++i) {
        AllocationRequest request;
        request.size = blockSize;
        request.preferredDeviceId = 0;
        
        auto handle = memManager.allocate(request);
        ASSERT_TRUE(handle.isValid());
        handles.push_back(handle);
    }
    
    // 检查内存池状态
    auto stats = memManager.getUsageStats();
    EXPECT_GE(stats.pooledMemory, blockSize * numBlocks);
    
    // 释放所有内存
    for (const auto& handle : handles) {
        ASSERT_TRUE(memManager.deallocate(handle));
    }
}

// 测试跨GPU数据传输
TEST_F(GPUMemoryManagerTest, CrossGPUTransfer) {
    auto& memManager = OSCEANGPUFramework::getMemoryManager();
    
    const auto& devices = OSCEANGPUFramework::getAvailableDevices();
    if (devices.size() < 2) {
        GTEST_SKIP() << "Less than 2 GPU devices available for transfer test";
    }
    
    const size_t dataSize = 1024 * 1024; // 1MB
    
    // 在设备0上分配源内存
    AllocationRequest srcRequest;
    srcRequest.size = dataSize;
    srcRequest.preferredDeviceId = 0;
    auto srcHandle = memManager.allocate(srcRequest);
    ASSERT_TRUE(srcHandle.isValid());
    
    // 在设备1上分配目标内存
    AllocationRequest dstRequest;
    dstRequest.size = dataSize;
    dstRequest.preferredDeviceId = 1;
    auto dstHandle = memManager.allocate(dstRequest);
    ASSERT_TRUE(dstHandle.isValid());
    
    // 创建测试数据
    std::vector<float> testData(dataSize / sizeof(float), 3.14f);
    
    // 上传数据到源设备
    // 注意：实际的上传需要使用特定API的函数
    // 这里我们只测试传输请求的创建
    
    // 执行跨GPU传输
    TransferRequest transferReq;
    transferReq.source = srcHandle;
    transferReq.destination = dstHandle;
    transferReq.async = false;
    
    bool transferStarted = memManager.transfer(transferReq);
    EXPECT_TRUE(transferStarted);
    
    // 等待传输完成
    if (transferStarted) {
        ASSERT_TRUE(memManager.synchronize(5000)); // 5秒超时
    }
    
    // 验证传输统计
    auto transferStats = memManager.getTransferStats();
    EXPECT_GE(transferStats.totalTransfers, 1);
    EXPECT_GE(transferStats.totalBytesTransferred, dataSize);
    
    // 清理
    ASSERT_TRUE(memManager.deallocate(srcHandle));
    ASSERT_TRUE(memManager.deallocate(dstHandle));
}

// 测试内存池预分配
TEST_F(GPUMemoryManagerTest, MemoryPoolPreallocation) {
    auto& memManager = OSCEANGPUFramework::getMemoryManager();
    
    const auto& devices = OSCEANGPUFramework::getAvailableDevices();
    if (devices.empty()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    const size_t preallocSize = 64 * 1024 * 1024; // 64MB
    
    // 预分配内存池
    ASSERT_TRUE(memManager.preallocatePool(0, preallocSize));
    
    // 快速分配测试
    std::vector<GPUMemoryHandle> handles;
    const size_t blockSize = 1024 * 1024; // 1MB
    const size_t numBlocks = 32;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < numBlocks; ++i) {
        AllocationRequest request;
        request.size = blockSize;
        request.preferredDeviceId = 0;
        
        auto handle = memManager.allocate(request);
        ASSERT_TRUE(handle.isValid());
        handles.push_back(handle);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    OSCEAN_LOG_INFO("GPUMemoryManagerTest", 
                    "Allocated {} blocks in {} ms", numBlocks, duration.count());
    
    // 释放内存
    for (const auto& handle : handles) {
        ASSERT_TRUE(memManager.deallocate(handle));
    }
}

// 测试内存碎片整理
TEST_F(GPUMemoryManagerTest, MemoryDefragmentation) {
    auto& memManager = OSCEANGPUFramework::getMemoryManager();
    
    const auto& devices = OSCEANGPUFramework::getAvailableDevices();
    if (devices.empty()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    // 创建碎片化的内存分配模式
    std::vector<GPUMemoryHandle> handles;
    const size_t smallSize = 256 * 1024;  // 256KB
    const size_t largeSize = 2 * 1024 * 1024; // 2MB
    
    // 交替分配大小内存块
    for (int i = 0; i < 10; ++i) {
        AllocationRequest request;
        request.size = (i % 2 == 0) ? smallSize : largeSize;
        request.preferredDeviceId = 0;
        
        auto handle = memManager.allocate(request);
        ASSERT_TRUE(handle.isValid());
        handles.push_back(handle);
    }
    
    // 释放偶数索引的内存块，创建碎片
    for (size_t i = 0; i < handles.size(); i += 2) {
        ASSERT_TRUE(memManager.deallocate(handles[i]));
    }
    
    // 执行碎片整理
    size_t freedMemory = memManager.defragment(0);
    OSCEAN_LOG_INFO("GPUMemoryManagerTest", 
                    "Defragmentation freed {} bytes", freedMemory);
    
    // 释放剩余的内存
    for (size_t i = 1; i < handles.size(); i += 2) {
        ASSERT_TRUE(memManager.deallocate(handles[i]));
    }
}

// 测试并发内存操作
TEST_F(GPUMemoryManagerTest, ConcurrentOperations) {
    auto& memManager = OSCEANGPUFramework::getMemoryManager();
    
    const auto& devices = OSCEANGPUFramework::getAvailableDevices();
    if (devices.empty()) {
        GTEST_SKIP() << "No GPU devices available";
    }
    
    const int numThreads = 4;
    const int allocationsPerThread = 25;
    std::vector<std::thread> threads;
    std::atomic<int> successCount(0);
    
    auto allocateAndFree = [&](int threadId) {
        std::vector<GPUMemoryHandle> handles;
        
        for (int i = 0; i < allocationsPerThread; ++i) {
            AllocationRequest request;
            request.size = (threadId + 1) * 256 * 1024; // 不同线程分配不同大小
            request.preferredDeviceId = threadId % devices.size();
            
            auto handle = memManager.allocate(request);
            if (handle.isValid()) {
                handles.push_back(handle);
                successCount++;
            }
        }
        
        // 释放所有分配的内存
        for (const auto& handle : handles) {
            memManager.deallocate(handle);
        }
    };
    
    // 启动并发线程
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(allocateAndFree, i);
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successCount.load(), numThreads * allocationsPerThread);
    
    // 验证最终状态
    auto stats = memManager.getUsageStats();
    EXPECT_EQ(stats.currentUsage, 0); // 所有内存应该已释放
}

// 主函数
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 