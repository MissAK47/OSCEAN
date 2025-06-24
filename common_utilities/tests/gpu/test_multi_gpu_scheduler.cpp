/**
 * @file test_multi_gpu_scheduler.cpp
 * @brief 多GPU调度器单元测试
 */

#include <gtest/gtest.h>
#include "common_utils/gpu/multi_gpu_scheduler.h"
#include "common_utils/gpu/unified_gpu_manager.h"
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

using namespace oscean::common_utils::gpu;

class MultiGPUSchedulerTest : public ::testing::Test {
protected:
    std::vector<GPUDeviceInfo> createMockDevices() {
        std::vector<GPUDeviceInfo> devices;
        
        // 模拟GPU设备1 - 高性能设备
        GPUDeviceInfo device1;
        device1.deviceId = 0;
        device1.name = "Mock GPU 1 - High Performance";
        device1.vendor = GPUVendor::NVIDIA;
        device1.performanceScore = 90;
        device1.memoryDetails.totalGlobalMemory = 8ULL * 1024 * 1024 * 1024; // 8GB
        device1.memoryDetails.freeGlobalMemory = 7ULL * 1024 * 1024 * 1024;  // 7GB free
        device1.computeUnits.multiprocessorCount = 46;
        device1.computeUnits.totalCores = 5888;
        devices.push_back(device1);
        
        // 模拟GPU设备2 - 中等性能设备
        GPUDeviceInfo device2;
        device2.deviceId = 1;
        device2.name = "Mock GPU 2 - Medium Performance";
        device2.vendor = GPUVendor::NVIDIA;
        device2.performanceScore = 60;
        device2.memoryDetails.totalGlobalMemory = 6ULL * 1024 * 1024 * 1024; // 6GB
        device2.memoryDetails.freeGlobalMemory = 5ULL * 1024 * 1024 * 1024;  // 5GB free
        device2.computeUnits.multiprocessorCount = 30;
        device2.computeUnits.totalCores = 3840;
        devices.push_back(device2);
        
        return devices;
    }
    
    GPUTaskInfo createTask(const std::string& id, size_t memReq, double complexity,
                          GPUTaskPriority priority = GPUTaskPriority::NORMAL) {
        GPUTaskInfo task;
        task.taskId = id;
        task.memoryRequirement = memReq;
        task.computeComplexity = complexity;
        task.priority = priority;
        task.estimatedDuration = boost::chrono::milliseconds(100);
        return task;
    }
};

// 测试调度器初始化
TEST_F(MultiGPUSchedulerTest, Initialization) {
    auto devices = createMockDevices();
    
    SchedulerConfig config;
    config.strategy = SchedulingStrategy::LEAST_LOADED;
    
    MultiGPUScheduler scheduler(devices, config);
    
    auto workloads = scheduler.getAllWorkloads();
    EXPECT_EQ(workloads.size(), 2);
    
    // 验证设备信息正确加载
    EXPECT_EQ(workloads[0].deviceId, 0);
    EXPECT_EQ(workloads[0].deviceInfo.name, "Mock GPU 1 - High Performance");
    EXPECT_EQ(workloads[1].deviceId, 1);
    EXPECT_EQ(workloads[1].deviceInfo.name, "Mock GPU 2 - Medium Performance");
}

// 测试最低负载调度策略
TEST_F(MultiGPUSchedulerTest, LeastLoadedScheduling) {
    auto devices = createMockDevices();
    
    SchedulerConfig config;
    config.strategy = SchedulingStrategy::LEAST_LOADED;
    
    MultiGPUScheduler scheduler(devices, config);
    
    // 创建任务
    auto task1 = createTask("task1", 1024 * 1024 * 100, 0.5); // 100MB
    
    // 应该选择设备0（两个设备负载相同时，选择第一个）
    auto decision = scheduler.selectOptimalGPU(task1);
    EXPECT_EQ(decision.selectedDeviceId, 0);
    EXPECT_GT(decision.confidenceScore, 0.0f);
    
    // 提交任务到设备0
    EXPECT_TRUE(scheduler.submitTask(0, task1));
    scheduler.taskStarted(0, task1.taskId);
    
    // 现在设备0有负载，应该选择设备1
    auto task2 = createTask("task2", 1024 * 1024 * 100, 0.5);
    decision = scheduler.selectOptimalGPU(task2);
    EXPECT_EQ(decision.selectedDeviceId, 1);
}

// 测试性能优先调度策略
TEST_F(MultiGPUSchedulerTest, PerformanceBasedScheduling) {
    auto devices = createMockDevices();
    
    SchedulerConfig config;
    config.strategy = SchedulingStrategy::PERFORMANCE_BASED;
    
    MultiGPUScheduler scheduler(devices, config);
    
    // 创建高计算复杂度任务
    auto task = createTask("compute_heavy", 1024 * 1024 * 500, 0.9); // 500MB, 高复杂度
    
    // 应该选择高性能设备（设备0）
    auto decision = scheduler.selectOptimalGPU(task);
    EXPECT_EQ(decision.selectedDeviceId, 0);
    EXPECT_TRUE(decision.reason.find("Performance-based") != std::string::npos);
}

// 测试内存感知调度
TEST_F(MultiGPUSchedulerTest, MemoryAwareScheduling) {
    auto devices = createMockDevices();
    
    SchedulerConfig config;
    config.strategy = SchedulingStrategy::MEMORY_AWARE;
    
    MultiGPUScheduler scheduler(devices, config);
    
    // 提交大内存任务到设备1，使其内存使用较高
    auto bigTask = createTask("big_task", 4ULL * 1024 * 1024 * 1024, 0.5); // 4GB
    EXPECT_TRUE(scheduler.submitTask(1, bigTask));
    
    // 创建需要大内存的新任务
    auto memTask = createTask("mem_task", 2ULL * 1024 * 1024 * 1024, 0.5); // 2GB
    
    // 应该选择设备0（有更多空闲内存）
    auto decision = scheduler.selectOptimalGPU(memTask);
    EXPECT_EQ(decision.selectedDeviceId, 0);
    EXPECT_TRUE(decision.reason.find("Memory-aware") != std::string::npos);
}

// 测试轮询调度
TEST_F(MultiGPUSchedulerTest, RoundRobinScheduling) {
    auto devices = createMockDevices();
    
    SchedulerConfig config;
    config.strategy = SchedulingStrategy::ROUND_ROBIN;
    
    MultiGPUScheduler scheduler(devices, config);
    
    // 提交多个任务，应该轮流分配
    std::vector<int> selectedDevices;
    for (int i = 0; i < 4; ++i) {
        auto task = createTask("task_" + std::to_string(i), 1024 * 1024 * 10, 0.5);
        auto decision = scheduler.selectOptimalGPU(task);
        selectedDevices.push_back(decision.selectedDeviceId);
        EXPECT_TRUE(scheduler.submitTask(decision.selectedDeviceId, task));
    }
    
    // 验证轮询模式：0, 1, 0, 1
    EXPECT_EQ(selectedDevices[0], 0);
    EXPECT_EQ(selectedDevices[1], 1);
    EXPECT_EQ(selectedDevices[2], 0);
    EXPECT_EQ(selectedDevices[3], 1);
}

// 测试任务优先级
TEST_F(MultiGPUSchedulerTest, TaskPriority) {
    auto devices = createMockDevices();
    
    SchedulerConfig config;
    config.strategy = SchedulingStrategy::LEAST_LOADED;
    config.maxQueuedTasksPerDevice = 3;
    
    MultiGPUScheduler scheduler(devices, config);
    
    // 填满设备0的队列
    for (int i = 0; i < 3; ++i) {
        auto task = createTask("normal_" + std::to_string(i), 1024 * 1024, 0.5, 
                              GPUTaskPriority::NORMAL);
        EXPECT_TRUE(scheduler.submitTask(0, task));
    }
    
    // 高优先级任务仍然可以提交
    auto criticalTask = createTask("critical", 1024 * 1024, 0.5, 
                                  GPUTaskPriority::CRITICAL);
    auto decision = scheduler.selectOptimalGPU(criticalTask);
    EXPECT_NE(decision.selectedDeviceId, -1);
}

// 测试任务生命周期
TEST_F(MultiGPUSchedulerTest, TaskLifecycle) {
    auto devices = createMockDevices();
    MultiGPUScheduler scheduler(devices);
    
    auto task = createTask("lifecycle_test", 1024 * 1024 * 100, 0.5);
    
    // 提交任务
    EXPECT_TRUE(scheduler.submitTask(0, task));
    
    auto workload = scheduler.getWorkload(0);
    ASSERT_TRUE(workload.has_value());
    EXPECT_EQ(workload->queuedTasks, 1);
    EXPECT_EQ(workload->runningTasks, 0);
    
    // 任务开始
    scheduler.taskStarted(0, task.taskId);
    workload = scheduler.getWorkload(0);
    ASSERT_TRUE(workload.has_value());
    EXPECT_EQ(workload->queuedTasks, 0);
    EXPECT_EQ(workload->runningTasks, 1);
    
    // 任务完成
    scheduler.taskCompleted(0, task.taskId, boost::chrono::milliseconds(50));
    workload = scheduler.getWorkload(0);
    ASSERT_TRUE(workload.has_value());
    EXPECT_EQ(workload->runningTasks, 0);
    EXPECT_EQ(workload->completedTasks, 1);
    EXPECT_GT(workload->avgTaskDuration, 0.0);
}

// 测试任务失败处理
TEST_F(MultiGPUSchedulerTest, TaskFailure) {
    auto devices = createMockDevices();
    MultiGPUScheduler scheduler(devices);
    
    auto task = createTask("fail_test", 1024 * 1024, 0.5);
    
    EXPECT_TRUE(scheduler.submitTask(0, task));
    scheduler.taskStarted(0, task.taskId);
    scheduler.taskFailed(0, task.taskId, "Test failure");
    
    auto workload = scheduler.getWorkload(0);
    ASSERT_TRUE(workload.has_value());
    EXPECT_EQ(workload->failedTasks, 1);
    EXPECT_EQ(workload->runningTasks, 0);
}

// 测试事件回调
TEST_F(MultiGPUSchedulerTest, EventCallbacks) {
    auto devices = createMockDevices();
    MultiGPUScheduler scheduler(devices);
    
    std::vector<SchedulerEvent> receivedEvents;
    
    scheduler.registerEventCallback([&receivedEvents](const SchedulerEvent& event) {
        receivedEvents.push_back(event);
    });
    
    auto task = createTask("event_test", 1024 * 1024, 0.5);
    
    scheduler.submitTask(0, task);
    scheduler.taskStarted(0, task.taskId);
    scheduler.taskCompleted(0, task.taskId, boost::chrono::milliseconds(100));
    
    // 应该收到3个事件
    EXPECT_EQ(receivedEvents.size(), 3);
    EXPECT_EQ(receivedEvents[0].type, SchedulerEventType::TASK_SCHEDULED);
    EXPECT_EQ(receivedEvents[1].type, SchedulerEventType::TASK_STARTED);
    EXPECT_EQ(receivedEvents[2].type, SchedulerEventType::TASK_COMPLETED);
}

// 测试负载均衡
TEST_F(MultiGPUSchedulerTest, LoadBalancing) {
    auto devices = createMockDevices();
    
    SchedulerConfig config;
    config.enableDynamicBalancing = true;
    config.updateInterval = boost::chrono::milliseconds(10);
    
    MultiGPUScheduler scheduler(devices, config);
    
    // 给设备0添加大量负载
    for (int i = 0; i < 5; ++i) {
        auto task = createTask("load_" + std::to_string(i), 1024 * 1024 * 100, 0.8);
        scheduler.submitTask(0, task);
        scheduler.taskStarted(0, task.taskId);
    }
    
    // 等待负载均衡触发
    boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
    
    // 新任务应该被调度到设备1
    auto newTask = createTask("balanced", 1024 * 1024 * 50, 0.5);
    auto decision = scheduler.selectOptimalGPU(newTask);
    EXPECT_EQ(decision.selectedDeviceId, 1);
}

// 测试设备过载检测
TEST_F(MultiGPUSchedulerTest, DeviceOverload) {
    auto devices = createMockDevices();
    
    SchedulerConfig config;
    config.loadThreshold = 0.7f;
    config.memoryThreshold = 0.8f;
    
    MultiGPUScheduler scheduler(devices, config);
    
    bool overloadDetected = false;
    scheduler.registerEventCallback([&overloadDetected](const SchedulerEvent& event) {
        if (event.type == SchedulerEventType::DEVICE_OVERLOADED) {
            overloadDetected = true;
        }
    });
    
    // 添加大量任务使设备过载
    for (int i = 0; i < 10; ++i) {
        auto task = createTask("overload_" + std::to_string(i), 
                              512 * 1024 * 1024, 0.9); // 512MB each
        scheduler.submitTask(0, task);
        scheduler.taskStarted(0, task.taskId);
    }
    
    // 等待事件处理
    boost::this_thread::sleep_for(boost::chrono::milliseconds(150));
    
    EXPECT_TRUE(overloadDetected);
}

// 测试统计信息
TEST_F(MultiGPUSchedulerTest, Statistics) {
    auto devices = createMockDevices();
    MultiGPUScheduler scheduler(devices);
    
    // 执行一些任务
    for (int i = 0; i < 3; ++i) {
        auto task = createTask("stat_" + std::to_string(i), 1024 * 1024, 0.5);
        scheduler.submitTask(i % 2, task);
        scheduler.taskStarted(i % 2, task.taskId);
        scheduler.taskCompleted(i % 2, task.taskId, 
                              boost::chrono::milliseconds(50 + i * 10));
    }
    
    auto stats = scheduler.getStatistics();
    
    // 验证统计信息包含关键数据
    EXPECT_TRUE(stats.find("GPU Scheduler Statistics") != std::string::npos);
    EXPECT_TRUE(stats.find("Device 0") != std::string::npos);
    EXPECT_TRUE(stats.find("Device 1") != std::string::npos);
    EXPECT_TRUE(stats.find("Completed Tasks") != std::string::npos);
    EXPECT_TRUE(stats.find("Avg Task Duration") != std::string::npos);
}

// 测试调度器重置
TEST_F(MultiGPUSchedulerTest, Reset) {
    auto devices = createMockDevices();
    MultiGPUScheduler scheduler(devices);
    
    // 添加一些任务和状态
    auto task = createTask("reset_test", 1024 * 1024, 0.5);
    scheduler.submitTask(0, task);
    scheduler.taskStarted(0, task.taskId);
    scheduler.taskCompleted(0, task.taskId, boost::chrono::milliseconds(100));
    
    // 重置
    scheduler.reset();
    
    // 验证所有状态已清除
    auto workloads = scheduler.getAllWorkloads();
    for (const auto& workload : workloads) {
        EXPECT_EQ(workload.currentLoad, 0.0f);
        EXPECT_EQ(workload.queuedTasks, 0);
        EXPECT_EQ(workload.runningTasks, 0);
        EXPECT_EQ(workload.completedTasks, 0);
        EXPECT_EQ(workload.failedTasks, 0);
        EXPECT_EQ(workload.allocatedMemory, 0);
    }
}

// 测试全局调度器管理
TEST_F(MultiGPUSchedulerTest, GlobalSchedulerManager) {
    auto devices = createMockDevices();
    
    // 初始化全局调度器
    GlobalSchedulerManager::initialize(devices);
    
    // 获取实例
    auto& scheduler = GlobalSchedulerManager::getInstance();
    
    // 使用调度器
    auto task = createTask("global_test", 1024 * 1024, 0.5);
    auto decision = scheduler.selectOptimalGPU(task);
    EXPECT_NE(decision.selectedDeviceId, -1);
    
    // 销毁
    GlobalSchedulerManager::destroy();
    
    // 再次获取应该抛出异常
    EXPECT_THROW(GlobalSchedulerManager::getInstance(), std::runtime_error);
} 