/**
 * @file test_gpu_detection.cpp
 * @brief GPU检测功能的单元测试
 */

#include <gtest/gtest.h>
#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include <iostream>

using namespace oscean::common_utils::gpu;

class GPUDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化日志系统
        oscean::common_utils::LoggingConfig config;
        config.console_level = "debug";
        oscean::common_utils::LoggingManager::configureGlobal(config);
    }
    
    void TearDown() override {
        // 清理GPU管理器
        UnifiedGPUManager::getInstance().cleanup();
    }
};

/**
 * @test 测试GPU管理器初始化
 */
TEST_F(GPUDetectionTest, ManagerInitialization) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    GPUInitOptions options;
    options.enableMultiGPU = true;
    options.preferHighPerformance = true;
    options.verboseLogging = true;
    
    GPUError result = manager.initialize(options);
    
    // 初始化应该成功（即使没有GPU也应该有CPU fallback）
    EXPECT_EQ(result, GPUError::SUCCESS);
    
    // 获取状态报告
    std::string status = manager.getStatusReport();
    std::cout << "GPU Manager Status:\n" << status << std::endl;
}

/**
 * @test 测试GPU设备检测
 */
TEST_F(GPUDetectionTest, DeviceDetection) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    // 初始化
    ASSERT_EQ(manager.initialize(), GPUError::SUCCESS);
    
    // 检测所有GPU
    auto devices = manager.detectAllGPUs();
    
    std::cout << "Detected " << devices.size() << " GPU device(s)" << std::endl;
    
    // 打印每个设备的信息
    for (const auto& device : devices) {
        std::cout << "\n=== Device " << device.deviceId << " ===" << std::endl;
        std::cout << "Name: " << device.name << std::endl;
        std::cout << "Vendor: " << device.getDescription() << std::endl;
        std::cout << "Performance Score: " << device.performanceScore << std::endl;
        std::cout << "Memory: " << (device.memoryDetails.totalGlobalMemory / (1024*1024*1024)) 
                  << " GB" << std::endl;
        
        std::cout << "Supported APIs:";
        for (const auto& api : device.supportedAPIs) {
            std::cout << " " << computeAPIToString(api);
        }
        std::cout << std::endl;
        
        std::cout << "Best API: " << computeAPIToString(device.getBestAPI()) << std::endl;
    }
}

/**
 * @test 测试设备过滤功能
 */
TEST_F(GPUDetectionTest, DeviceFiltering) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    // 初始化
    ASSERT_EQ(manager.initialize(), GPUError::SUCCESS);
    
    // 测试不同的过滤条件
    {
        // 过滤高性能GPU
        GPUDeviceFilter filter;
        filter.minPerformanceScore = 70;
        
        auto devices = manager.detectAllGPUs(filter);
        std::cout << "\nHigh performance GPUs (score >= 70): " << devices.size() << std::endl;
        
        for (const auto& device : devices) {
            EXPECT_GE(device.performanceScore, 70);
        }
    }
    
    {
        // 过滤支持CUDA的GPU
        GPUDeviceFilter filter;
        filter.requiredAPI = ComputeAPI::CUDA;
        
        auto devices = manager.detectAllGPUs(filter);
        std::cout << "CUDA-capable GPUs: " << devices.size() << std::endl;
        
        for (const auto& device : devices) {
            EXPECT_TRUE(device.hasAPI(ComputeAPI::CUDA));
        }
    }
    
    {
        // 过滤大内存GPU（>= 8GB）
        GPUDeviceFilter filter;
        filter.minMemorySize = 8ULL * 1024 * 1024 * 1024;
        
        auto devices = manager.detectAllGPUs(filter);
        std::cout << "GPUs with >= 8GB memory: " << devices.size() << std::endl;
        
        for (const auto& device : devices) {
            EXPECT_GE(device.memoryDetails.totalGlobalMemory, 8ULL * 1024 * 1024 * 1024);
        }
    }
}

/**
 * @test 测试最优配置选择
 */
TEST_F(GPUDetectionTest, OptimalConfiguration) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    // 初始化
    ASSERT_EQ(manager.initialize(), GPUError::SUCCESS);
    
    // 获取默认最优配置
    auto config = manager.getOptimalConfiguration();
    
    std::cout << "\n=== Optimal GPU Configuration ===" << std::endl;
    std::cout << "Primary Device: " << config.primaryDevice.getDescription() << std::endl;
    std::cout << "Compute API: " << computeAPIToString(config.computeAPI) << std::endl;
    std::cout << "Multi-GPU: " << (config.enableMultiGPU ? "Enabled" : "Disabled") << std::endl;
    
    if (config.enableMultiGPU) {
        std::cout << "Secondary Devices: " << config.secondaryDevices.size() << std::endl;
        for (const auto& device : config.secondaryDevices) {
            std::cout << "  - " << device.getDescription() << std::endl;
        }
    }
    
    // 测试特定需求的配置
    GPUDeviceFilter requirements;
    requirements.requireTensorCores = true;
    
    auto aiConfig = manager.getOptimalConfiguration(requirements);
    if (aiConfig.primaryDevice.deviceId >= 0) {
        std::cout << "\nAI/ML Optimal Configuration: " 
                  << aiConfig.primaryDevice.getDescription() << std::endl;
        EXPECT_TRUE(aiConfig.primaryDevice.hasAIAcceleration());
    }
}

/**
 * @test 测试设备选择功能
 */
TEST_F(GPUDetectionTest, DeviceSelection) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    // 初始化
    ASSERT_EQ(manager.initialize(), GPUError::SUCCESS);
    
    size_t deviceCount = manager.getDeviceCount();
    std::cout << "\nTotal devices: " << deviceCount << std::endl;
    
    if (deviceCount > 0) {
        // 测试设置当前设备
        for (size_t i = 0; i < std::min(deviceCount, size_t(3)); ++i) {
            EXPECT_EQ(manager.setCurrentDevice(static_cast<int>(i)), GPUError::SUCCESS);
            EXPECT_EQ(manager.getCurrentDevice(), static_cast<int>(i));
            
            auto deviceInfo = manager.getDeviceInfo(static_cast<int>(i));
            ASSERT_TRUE(deviceInfo.has_value());
            std::cout << "Selected device " << i << ": " 
                      << deviceInfo->getDescription() << std::endl;
        }
        
        // 测试无效设备ID
        EXPECT_NE(manager.setCurrentDevice(-1), GPUError::SUCCESS);
        EXPECT_NE(manager.setCurrentDevice(static_cast<int>(deviceCount)), GPUError::SUCCESS);
    }
}

/**
 * @test 测试工作负载设备选择
 */
TEST_F(GPUDetectionTest, WorkloadDeviceSelection) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    // 初始化
    ASSERT_EQ(manager.initialize(), GPUError::SUCCESS);
    
    if (manager.getDeviceCount() > 0) {
        // 测试不同工作负载的设备选择
        struct TestCase {
            size_t memoryRequirement;
            double complexity;
            const char* description;
        };
        
        TestCase testCases[] = {
            {512 * 1024 * 1024, 0.3, "Light workload (512MB, low complexity)"},
            {2ULL * 1024 * 1024 * 1024, 0.7, "Medium workload (2GB, high complexity)"},
            {8ULL * 1024 * 1024 * 1024, 0.9, "Heavy workload (8GB, very high complexity)"}
        };
        
        for (const auto& test : testCases) {
            int deviceId = manager.selectOptimalDevice(test.memoryRequirement, test.complexity);
            
            std::cout << "\n" << test.description << std::endl;
            if (deviceId >= 0) {
                auto device = manager.getDeviceInfo(deviceId);
                ASSERT_TRUE(device.has_value());
                std::cout << "  Selected: Device " << deviceId << " - " 
                          << device->getDescription() << std::endl;
                
                // 验证设备有足够的内存
                EXPECT_GE(device->memoryDetails.freeGlobalMemory, test.memoryRequirement);
            } else {
                std::cout << "  No suitable device found" << std::endl;
            }
        }
    }
}

/**
 * @test 测试设备诊断功能
 */
TEST_F(GPUDetectionTest, DeviceDiagnostics) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    // 初始化
    ASSERT_EQ(manager.initialize(), GPUError::SUCCESS);
    
    if (manager.getDeviceCount() > 0) {
        // 获取第一个设备的诊断信息
        std::string diagnostics = manager.getDeviceDiagnostics(0);
        std::cout << "\n" << diagnostics << std::endl;
        
        // 诊断信息应该包含关键信息
        EXPECT_FALSE(diagnostics.empty());
        EXPECT_NE(diagnostics.find("Device ID"), std::string::npos);
        EXPECT_NE(diagnostics.find("Name"), std::string::npos);
        EXPECT_NE(diagnostics.find("Memory"), std::string::npos);
    }
}

/**
 * @test 测试事件回调机制
 */
TEST_F(GPUDetectionTest, EventCallbacks) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    // 初始化
    ASSERT_EQ(manager.initialize(), GPUError::SUCCESS);
    
    // 注册事件回调
    std::vector<std::pair<int, std::string>> events;
    
    manager.registerEventCallback([&events](int deviceId, const std::string& event) {
        events.push_back({deviceId, event});
        std::cout << "Event: Device " << deviceId << " - " << event << std::endl;
    });
    
    if (manager.getDeviceCount() > 0) {
        // 触发一些事件
        manager.setCurrentDevice(0);
        manager.resetDevice(0);
        
        // 检查事件是否被记录
        EXPECT_GE(events.size(), 2u);
        
        // 查找特定事件
        bool foundDeviceSelected = false;
        bool foundDeviceReset = false;
        
        for (const auto& event : events) {
            if (event.second == "device_selected") foundDeviceSelected = true;
            if (event.second == "device_reset") foundDeviceReset = true;
        }
        
        EXPECT_TRUE(foundDeviceSelected);
        EXPECT_TRUE(foundDeviceReset);
    }
    
    // 清除回调
    manager.clearEventCallbacks();
}

/**
 * @test 测试多GPU支持（如果有多个GPU）
 */
TEST_F(GPUDetectionTest, MultiGPUSupport) {
    auto& manager = UnifiedGPUManager::getInstance();
    
    // 启用多GPU初始化
    GPUInitOptions options;
    options.enableMultiGPU = true;
    options.enablePeerAccess = true;
    
    ASSERT_EQ(manager.initialize(options), GPUError::SUCCESS);
    
    if (manager.getDeviceCount() > 1) {
        std::cout << "\n=== Multi-GPU Configuration ===" << std::endl;
        std::cout << "Found " << manager.getDeviceCount() << " GPUs" << std::endl;
        
        // 测试GPU间点对点访问
        for (size_t i = 0; i < manager.getDeviceCount(); ++i) {
            for (size_t j = i + 1; j < manager.getDeviceCount(); ++j) {
                bool canAccess = manager.canAccessPeer(static_cast<int>(i), static_cast<int>(j));
                std::cout << "GPU " << i << " -> GPU " << j 
                          << " peer access: " << (canAccess ? "Yes" : "No") << std::endl;
            }
        }
    } else {
        std::cout << "\nMulti-GPU test skipped (only " 
                  << manager.getDeviceCount() << " GPU found)" << std::endl;
    }
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 