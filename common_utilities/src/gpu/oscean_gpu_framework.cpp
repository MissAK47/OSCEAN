/**
 * @file oscean_gpu_framework.cpp
 * @brief OSCEAN统一GPU框架实现
 */

#include "common_utils/gpu/oscean_gpu_framework.h"
#include "common_utils/utilities/logging_utils.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace oscean::common_utils::gpu {

// 静态成员初始化
std::unique_ptr<OSCEANGPUFramework> OSCEANGPUFramework::s_instance;
boost::mutex OSCEANGPUFramework::s_mutex;
boost::once_flag OSCEANGPUFramework::s_onceFlag;

// GPUFrameworkStatus实现
std::string GPUFrameworkStatus::toString() const {
    std::stringstream ss;
    ss << "GPU Framework Status:\n";
    ss << "  Initialized: " << (initialized ? "Yes" : "No") << "\n";
    ss << "  Available Devices: " << availableDevices << "\n";
    ss << "  Active Devices: " << activeDevices << "\n";
    ss << "  Average GPU Utilization: " << std::fixed << std::setprecision(1) 
       << averageGPUUtilization << "%\n";
    ss << "  Average Memory Utilization: " << std::fixed << std::setprecision(1)
       << averageMemoryUtilization << "%\n";
    ss << "  Total Tasks Processed: " << totalTasksProcessed << "\n";
    ss << "  Failed Tasks: " << failedTasks << "\n";
    ss << "  Total Memory Allocated: " << (totalMemoryAllocated / (1024 * 1024)) << " MB\n";
    ss << "  Total Memory Available: " << (totalMemoryAvailable / (1024 * 1024)) << " MB\n";
    
    if (!devices.empty()) {
        ss << "\nDevices:\n";
        for (const auto& device : devices) {
            ss << "  - " << device.name << " (ID: " << device.deviceId << ")\n";
        }
    }
    
    return ss.str();
}

// 简单GPU任务实现
class GPUTaskBuilder::SimpleGPUTask : public IGPUTask {
public:
    std::string name;
    size_t memoryRequirement = 0;
    double computeComplexity = 1.0;
    std::function<bool(GPUTaskExecutionContext&)> executor;
    std::function<bool(const GPUTaskExecutionContext&)> validator;
    
    std::string getName() const override { return name; }
    size_t estimateMemoryRequirement() const override { return memoryRequirement; }
    double estimateComputeComplexity() const override { return computeComplexity; }
    
    bool execute(GPUTaskExecutionContext& context) override {
        if (executor) {
            return executor(context);
        }
        return true;
    }
    
    bool validate(const GPUTaskExecutionContext& context) const override {
        if (validator) {
            return validator(context);
        }
        return true;
    }
};

// GPU任务构建器实现
GPUTaskBuilder& GPUTaskBuilder::withName(const std::string& name) {
    if (!m_task) {
        m_task = std::make_shared<SimpleGPUTask>();
    }
    m_task->name = name;
    return *this;
}

GPUTaskBuilder& GPUTaskBuilder::withMemoryRequirement(size_t size) {
    if (!m_task) {
        m_task = std::make_shared<SimpleGPUTask>();
    }
    m_task->memoryRequirement = size;
    return *this;
}

GPUTaskBuilder& GPUTaskBuilder::withComputeComplexity(double complexity) {
    if (!m_task) {
        m_task = std::make_shared<SimpleGPUTask>();
    }
    m_task->computeComplexity = complexity;
    return *this;
}

GPUTaskBuilder& GPUTaskBuilder::withExecutor(std::function<bool(GPUTaskExecutionContext&)> executor) {
    if (!m_task) {
        m_task = std::make_shared<SimpleGPUTask>();
    }
    m_task->executor = executor;
    return *this;
}

GPUTaskBuilder& GPUTaskBuilder::withValidator(std::function<bool(const GPUTaskExecutionContext&)> validator) {
    if (!m_task) {
        m_task = std::make_shared<SimpleGPUTask>();
    }
    m_task->validator = validator;
    return *this;
}

std::shared_ptr<IGPUTask> GPUTaskBuilder::build() {
    if (!m_task) {
        m_task = std::make_shared<SimpleGPUTask>();
    }
    if (m_task->name.empty()) {
        m_task->name = "UnnamedTask";
    }
    return m_task;
}

// 任务执行信息
struct TaskExecutionInfo {
    std::shared_ptr<IGPUTask> task;
    GPUTaskPriority priority;
    GPUTaskExecutionContext context;
    boost::atomic<bool> completed{false};
    boost::atomic<bool> cancelled{false};
    boost::atomic<bool> success{false};
    std::string errorMessage;
};

// OSCEANGPUFramework内部实现
class OSCEANGPUFramework::Impl {
public:
    Impl() : m_initialized(false), m_performanceMonitorRunning(false) {
        m_status.initialized = false;
        m_status.availableDevices = 0;
        m_status.activeDevices = 0;
    }
    
    ~Impl() {
        shutdown();
    }
    
    bool initialize(const GPUFrameworkConfig& config) {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        if (m_initialized) {
            return true;
        }
        
        m_config = config;
        
        try {
            // 1. 检测GPU设备
            UnifiedGPUManager& gpuManager = getGPUManager();
            auto devices = gpuManager.detectAllGPUs();
            
            if (devices.empty()) {
                OSCEAN_LOG_ERROR("OSCEANGPUFramework", "No GPU devices found");
                return false;
            }
            
            m_status.devices = devices;
            m_status.availableDevices = devices.size();
            
            OSCEAN_LOG_INFO("OSCEANGPUFramework", "Detected {} GPU devices", devices.size());
            
            // 2. 初始化内存管理器
            GlobalMemoryManager::initialize(devices);
            
            // 预分配内存池
            if (config.memoryPoolConfig.initialPoolSize > 0) {
                auto& memManager = GlobalMemoryManager::getInstance();
                for (const auto& device : devices) {
                    memManager.preallocatePool(device.deviceId, 
                                              config.memoryPoolConfig.initialPoolSize);
                    OSCEAN_LOG_INFO("OSCEANGPUFramework", 
                                   "Preallocated {} MB for device {}", 
                                   config.memoryPoolConfig.initialPoolSize / (1024*1024),
                                   device.deviceId);
                }
            }
            
            // 3. 初始化调度器
            GlobalSchedulerManager::initialize(devices);
            auto& scheduler = GlobalSchedulerManager::getInstance();
            scheduler.setSchedulingStrategy(config.schedulerConfig.strategy);
            
            // 4. 启动性能监控
            if (config.enablePerformanceMonitoring) {
                startPerformanceMonitoring();
            }
            
            m_status.initialized = true;
            m_status.activeDevices = devices.size();
            m_initialized = true;
            
            fireEvent(FrameworkEventType::INITIALIZED, -1, 
                     "GPU Framework initialized successfully");
            
            return true;
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("OSCEANGPUFramework", "Initialization failed: {}", e.what());
            return false;
        }
        
        return false;
    }
    
    void shutdown() {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        if (!m_initialized) {
            return;
        }
        
        // 停止性能监控
        stopPerformanceMonitoring();
        
        // 等待所有任务完成
        waitForAllTasks();
        
        // 等待所有任务线程完成
        {
            boost::lock_guard<boost::mutex> threadLock(m_threadMutex);
            for (auto& thread : m_taskThreads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            m_taskThreads.clear();
        }
        
        // 销毁调度器
        GlobalSchedulerManager::destroy();
        
        // 销毁内存管理器
        GlobalMemoryManager::destroy();
        
        // 重置状态
        m_status = GPUFrameworkStatus();
        m_initialized = false;
        
        fireEvent(FrameworkEventType::SHUTDOWN, -1, 
                 "GPU Framework shutdown");
        
        OSCEAN_LOG_INFO("OSCEANGPUFramework", "Shutdown complete");
    }
    
    bool isInitialized() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        return m_initialized;
    }
    
    GPUFrameworkStatus getStatus() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        // 更新实时状态
        if (m_initialized) {
            auto& memManager = GlobalMemoryManager::getInstance();
            auto memStats = memManager.getUsageStats();
            
            m_status.totalMemoryAllocated = memStats.currentUsage;
            m_status.totalMemoryAvailable = 0;
            for (const auto& device : m_status.devices) {
                m_status.totalMemoryAvailable += memManager.getAvailableMemory(device.deviceId);
            }
            
            // 计算内存利用率
            if (m_status.totalMemoryAvailable > 0) {
                m_status.averageMemoryUtilization = 
                    (double)m_status.totalMemoryAllocated / 
                    (m_status.totalMemoryAllocated + m_status.totalMemoryAvailable) * 100.0;
            }
        }
        
        return m_status;
    }
    
    const std::vector<GPUDeviceInfo>& getAvailableDevices() const {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        return m_status.devices;
    }
    
    UnifiedGPUManager& getGPUManager() {
        // 返回静态实例
        static UnifiedGPUManager manager;
        return manager;
    }
    
    MultiGPUScheduler& getScheduler() {
        if (!m_initialized) {
            throw std::runtime_error("GPU Framework not initialized");
        }
        return GlobalSchedulerManager::getInstance();
    }
    
    MultiGPUMemoryManager& getMemoryManager() {
        if (!m_initialized) {
            throw std::runtime_error("GPU Framework not initialized");
        }
        return GlobalMemoryManager::getInstance();
    }
    
    std::string submitTask(std::shared_ptr<IGPUTask> task, GPUTaskPriority priority) {
        if (!m_initialized) {
            throw std::runtime_error("GPU Framework not initialized");
        }
        
        // 生成任务ID
        boost::uuids::random_generator gen;
        std::string taskId = boost::uuids::to_string(gen());
        
        // 创建任务信息
        auto taskInfo = std::make_shared<TaskExecutionInfo>();
        taskInfo->task = task;
        taskInfo->priority = priority;
        
        // 创建GPU任务信息
        GPUTaskInfo gpuTaskInfo;
        gpuTaskInfo.taskId = taskId;
        gpuTaskInfo.memoryRequirement = task->estimateMemoryRequirement();
        gpuTaskInfo.computeComplexity = task->estimateComputeComplexity();
        gpuTaskInfo.priority = priority;
        
        // 提交到调度器
        auto& scheduler = GlobalSchedulerManager::getInstance();
        SchedulingDecision decision = scheduler.selectOptimalGPU(gpuTaskInfo);
        
        if (decision.selectedDeviceId < 0) {
            return "";  // 返回空字符串表示失败
        }
        
        if (!scheduler.submitTask(decision.selectedDeviceId, gpuTaskInfo)) {
            return "";  // 返回空字符串表示失败
        }
        
        // 保存任务信息
        {
            boost::lock_guard<boost::mutex> lock(m_taskMutex);
            m_tasks[taskId] = taskInfo;
        }
        
        fireEvent(FrameworkEventType::TASK_SUBMITTED, -1,
                 "Task submitted: " + task->getName());
        
        // 在新线程中执行任务，但保存线程引用
        {
            boost::lock_guard<boost::mutex> threadLock(m_threadMutex);
            
            // 清理已完成的线程
            m_taskThreads.erase(
                std::remove_if(m_taskThreads.begin(), m_taskThreads.end(),
                    [](boost::thread& t) { return !t.joinable(); }),
                m_taskThreads.end()
            );
            
            // 创建新线程并保存
            m_taskThreads.emplace_back([this, taskInfo, taskId, deviceId = decision.selectedDeviceId]() {
                // 通知调度器任务开始
                auto& scheduler = GlobalSchedulerManager::getInstance();
                scheduler.taskStarted(deviceId, taskId);
                
                // 执行任务
                executeTask(taskInfo, deviceId);
                
                // 通知调度器任务完成
                auto executionTime = boost::chrono::duration_cast<boost::chrono::milliseconds>(
                    taskInfo->context.endTime - taskInfo->context.startTime);
                
                if (taskInfo->success) {
                    scheduler.taskCompleted(deviceId, taskId, executionTime);
                } else {
                    scheduler.taskFailed(deviceId, taskId, taskInfo->errorMessage);
                }
            });
        }
        
        return taskId;
    }
    
    bool waitForTask(const std::string& taskId, int timeoutMs) {
        auto startTime = boost::chrono::steady_clock::now();
        
        while (true) {
            {
                boost::lock_guard<boost::mutex> lock(m_taskMutex);
                auto it = m_tasks.find(taskId);
                if (it != m_tasks.end()) {
                    if (it->second->completed || it->second->cancelled) {
                        return it->second->success;
                    }
                }
            }
            
            if (timeoutMs >= 0) {
                auto elapsed = boost::chrono::duration_cast<boost::chrono::milliseconds>(
                    boost::chrono::steady_clock::now() - startTime).count();
                if (elapsed >= timeoutMs) {
                    return false;
                }
            }
            
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        }
    }
    
    bool cancelTask(const std::string& taskId) {
        boost::lock_guard<boost::mutex> lock(m_taskMutex);
        
        auto it = m_tasks.find(taskId);
        if (it != m_tasks.end()) {
            if (!it->second->completed) {
                it->second->cancelled = true;
                
                // 尝试从调度器取消
                // 注意：当前调度器不支持取消任务
                // scheduler.cancelTask(taskId);
                
                return true;
            }
        }
        
        return false;
    }
    
    boost::optional<GPUTaskExecutionContext> getTaskStatus(const std::string& taskId) const {
        boost::lock_guard<boost::mutex> lock(m_taskMutex);
        
        auto it = m_tasks.find(taskId);
        if (it != m_tasks.end()) {
            return it->second->context;
        }
        
        return boost::none;
    }
    
    GPUMemoryHandle allocateMemory(size_t size, int deviceId) {
        if (!m_initialized) {
            throw std::runtime_error("GPU Framework not initialized");
        }
        
        AllocationRequest request;
        request.size = size;
        request.preferredDeviceId = deviceId;
        
        auto& memManager = GlobalMemoryManager::getInstance();
        return memManager.allocate(request);
    }
    
    void deallocateMemory(const GPUMemoryHandle& handle) {
        if (!m_initialized) {
            return;
        }
        
        auto& memManager = GlobalMemoryManager::getInstance();
        memManager.deallocate(handle);
    }
    
    bool transferData(const GPUMemoryHandle& source,
                     const GPUMemoryHandle& destination,
                     bool async) {
        if (!m_initialized) {
            throw std::runtime_error("GPU Framework not initialized");
        }
        
        TransferRequest request;
        request.source = source;
        request.destination = destination;
        request.async = async;
        
        auto& memManager = GlobalMemoryManager::getInstance();
        return memManager.transfer(request);
    }
    
    void registerEventCallback(FrameworkEventCallback callback) {
        boost::lock_guard<boost::mutex> lock(m_callbackMutex);
        m_eventCallbacks.push_back(callback);
    }
    
    void setPerformanceCallback(std::function<void(const GPUFrameworkStatus&)> callback) {
        boost::lock_guard<boost::mutex> lock(m_callbackMutex);
        m_performanceCallback = callback;
    }
    
    void refreshDevices() {
        if (!m_initialized) {
            return;
        }
        
        UnifiedGPUManager& gpuManager = getGPUManager();
        auto devices = gpuManager.getAllDeviceInfo();
        
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        // 检查设备变化
        for (const auto& newDevice : devices) {
            bool found = false;
            for (const auto& oldDevice : m_status.devices) {
                if (newDevice.deviceId == oldDevice.deviceId) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                fireEvent(FrameworkEventType::DEVICE_ADDED, newDevice.deviceId,
                         "New device detected: " + newDevice.name);
            }
        }
        
        for (const auto& oldDevice : m_status.devices) {
            bool found = false;
            for (const auto& newDevice : devices) {
                if (oldDevice.deviceId == newDevice.deviceId) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                fireEvent(FrameworkEventType::DEVICE_REMOVED, oldDevice.deviceId,
                         "Device removed: " + oldDevice.name);
            }
        }
        
        m_status.devices = devices;
        m_status.availableDevices = devices.size();
    }
    
    std::vector<int> getRecommendedDevices(const std::string& taskType) const {
        std::vector<int> recommended;
        
        if (!m_initialized) {
            return recommended;
        }
        
        // 简单的推荐逻辑
        for (const auto& device : m_status.devices) {
            if (device.performanceScore >= 70) {
                recommended.push_back(device.deviceId);
            }
        }
        
        // 按性能评分排序
        std::sort(recommended.begin(), recommended.end(),
                 [this](int a, int b) {
                     auto deviceA = std::find_if(m_status.devices.begin(), 
                                               m_status.devices.end(),
                                               [a](const GPUDeviceInfo& d) { 
                                                   return d.deviceId == a; 
                                               });
                     auto deviceB = std::find_if(m_status.devices.begin(), 
                                               m_status.devices.end(),
                                               [b](const GPUDeviceInfo& d) { 
                                                   return d.deviceId == b; 
                                               });
                     if (deviceA != m_status.devices.end() && 
                         deviceB != m_status.devices.end()) {
                         return deviceA->performanceScore > deviceB->performanceScore;
                     }
                     return false;
                 });
        
        return recommended;
    }
    
    void optimizeConfiguration(const std::string& workloadProfile) {
        if (!m_initialized) {
            return;
        }
        
        // 根据工作负载特征优化配置
        if (workloadProfile == "compute_intensive") {
            m_config.schedulerConfig.strategy = SchedulingStrategy::PERFORMANCE_BASED;
            m_config.memoryPoolConfig.initialPoolSize = 512 * 1024 * 1024; // 512MB
        } else if (workloadProfile == "memory_intensive") {
            m_config.schedulerConfig.strategy = SchedulingStrategy::MEMORY_AWARE;
            m_config.memoryPoolConfig.initialPoolSize = 1024 * 1024 * 1024; // 1GB
        } else if (workloadProfile == "balanced") {
            m_config.schedulerConfig.strategy = SchedulingStrategy::LEAST_LOADED;
            m_config.memoryPoolConfig.initialPoolSize = 256 * 1024 * 1024; // 256MB
        }
        
        // 应用新配置
        auto& scheduler = GlobalSchedulerManager::getInstance();
        scheduler.setSchedulingStrategy(m_config.schedulerConfig.strategy);
    }
    
    std::string generatePerformanceReport() const {
        std::stringstream ss;
        
        ss << "=== GPU Framework Performance Report ===\n\n";
        
        // 基本状态
        ss << m_status.toString() << "\n";
        
        // 内存统计
        if (m_initialized) {
            auto& memManager = GlobalMemoryManager::getInstance();
            ss << "\n" << memManager.generateMemoryReport() << "\n";
            
            // 传输统计
            auto transferStats = memManager.getTransferStats();
            ss << "\nData Transfer Statistics:\n";
            ss << "  Total Transfers: " << transferStats.totalTransfers << "\n";
            ss << "  Average Transfer Size: " << 
                  (transferStats.averageTransferSize / (1024 * 1024)) << " MB\n";
            ss << "  Average Transfer Time: " << 
                  transferStats.averageTransferTime << " ms\n";
            ss << "  Peak Bandwidth: " << 
                  transferStats.peakBandwidth << " GB/s\n";
        }
        
        // 任务统计
        ss << "\nTask Execution Statistics:\n";
        ss << "  Success Rate: ";
        if (m_status.totalTasksProcessed > 0) {
            double successRate = (double)(m_status.totalTasksProcessed - m_status.failedTasks) / 
                               m_status.totalTasksProcessed * 100.0;
            ss << std::fixed << std::setprecision(1) << successRate << "%\n";
        } else {
            ss << "N/A\n";
        }
        
        return ss.str();
    }
    
    void reset() {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        if (!m_initialized) {
            return;
        }
        
        // 清理任务
        {
            boost::lock_guard<boost::mutex> taskLock(m_taskMutex);
            m_tasks.clear();
        }
        
        // 重置统计
        m_status.totalTasksProcessed = 0;
        m_status.failedTasks = 0;
        m_status.averageGPUUtilization = 0.0;
        m_status.averageMemoryUtilization = 0.0;
        
        // 重置内存管理器
        auto& memManager = GlobalMemoryManager::getInstance();
        memManager.reset(false);
        
        OSCEAN_LOG_INFO("OSCEANGPUFramework", "Reset complete");
    }
    
private:
    // 配置和状态
    GPUFrameworkConfig m_config;
    mutable GPUFrameworkStatus m_status;
    boost::atomic<bool> m_initialized;
    
    // 任务管理
    std::map<std::string, std::shared_ptr<TaskExecutionInfo>> m_tasks;
    
    // 性能监控
    boost::thread m_performanceMonitorThread;
    boost::atomic<bool> m_performanceMonitorRunning;
    
    // 回调
    std::vector<FrameworkEventCallback> m_eventCallbacks;
    std::function<void(const GPUFrameworkStatus&)> m_performanceCallback;
    
    // 同步
    mutable boost::mutex m_mutex;
    mutable boost::mutex m_taskMutex;
    mutable boost::mutex m_callbackMutex;
    
    // 任务执行线程池
    std::vector<boost::thread> m_taskThreads;
    boost::mutex m_threadMutex;
    
    void executeTask(std::shared_ptr<TaskExecutionInfo> taskInfo, int deviceId) {
        taskInfo->context.deviceId = deviceId;
        taskInfo->context.startTime = boost::chrono::steady_clock::now();
        
        try {
            // 分配内存
            if (taskInfo->task->estimateMemoryRequirement() > 0) {
                taskInfo->context.outputMemory = allocateMemory(
                    taskInfo->task->estimateMemoryRequirement(), deviceId);
            }
            
            // 执行任务
            bool success = taskInfo->task->execute(taskInfo->context);
            
            // 验证结果
            if (success) {
                success = taskInfo->task->validate(taskInfo->context);
            }
            
            taskInfo->context.endTime = boost::chrono::steady_clock::now();
            taskInfo->context.executionTimeMs = 
                boost::chrono::duration_cast<boost::chrono::microseconds>(
                    taskInfo->context.endTime - taskInfo->context.startTime).count() / 1000.0;
            
            taskInfo->success = success;
            taskInfo->completed = true;
            
            // 更新统计
            {
                boost::lock_guard<boost::mutex> lock(m_mutex);
                m_status.totalTasksProcessed++;
                if (!success) {
                    m_status.failedTasks++;
                }
            }
            
            fireEvent(success ? FrameworkEventType::TASK_COMPLETED : 
                               FrameworkEventType::TASK_FAILED,
                     deviceId,
                     "Task " + taskInfo->task->getName() + 
                     (success ? " completed" : " failed"));
            
        } catch (const std::exception& e) {
            taskInfo->errorMessage = e.what();
            taskInfo->success = false;
            taskInfo->completed = true;
            
            {
                boost::lock_guard<boost::mutex> lock(m_mutex);
                m_status.totalTasksProcessed++;
                m_status.failedTasks++;
            }
            
            fireEvent(FrameworkEventType::TASK_FAILED, deviceId,
                     "Task " + taskInfo->task->getName() + 
                     " failed: " + e.what());
        }
        
        // 释放内存
        if (taskInfo->context.outputMemory.isValid()) {
            deallocateMemory(taskInfo->context.outputMemory);
        }
    }
    
    void startPerformanceMonitoring() {
        m_performanceMonitorRunning = true;
        m_performanceMonitorThread = boost::thread(
            &Impl::performanceMonitorWorker, this);
    }
    
    void stopPerformanceMonitoring() {
        m_performanceMonitorRunning = false;
        if (m_performanceMonitorThread.joinable()) {
            m_performanceMonitorThread.join();
        }
    }
    
    void performanceMonitorWorker() {
        while (m_performanceMonitorRunning) {
            boost::this_thread::sleep_for(m_config.monitoringInterval);
            
            if (!m_performanceMonitorRunning) break;
            
            // 更新性能指标
            updatePerformanceMetrics();
            
            // 调用性能回调
            if (m_performanceCallback) {
                auto status = getStatus();
                m_performanceCallback(status);
            }
        }
    }
    
    void updatePerformanceMetrics() {
        // TODO: 实现真实的GPU利用率监控
        // 这里使用模拟数据
        
        boost::lock_guard<boost::mutex> lock(m_mutex);
        
        // 模拟GPU利用率
        if (m_status.activeDevices > 0) {
            m_status.averageGPUUtilization = 
                (double)(m_tasks.size() * 20) / m_status.activeDevices;
            m_status.averageGPUUtilization = 
                std::min(100.0, m_status.averageGPUUtilization);
        }
    }
    
    void waitForAllTasks() {
        // 等待所有任务完成
        while (true) {
            bool allCompleted = true;
            {
                boost::lock_guard<boost::mutex> lock(m_taskMutex);
                for (const auto& [_, taskInfo] : m_tasks) {
                    if (!taskInfo->completed && !taskInfo->cancelled) {
                        allCompleted = false;
                        break;
                    }
                }
            }
            
            if (allCompleted) break;
            
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
        }
    }
    
    void fireEvent(FrameworkEventType type, int deviceId, const std::string& message) {
        FrameworkEvent event;
        event.type = type;
        event.message = message;
        event.deviceId = deviceId;
        event.timestamp = boost::chrono::steady_clock::now();
        
        boost::lock_guard<boost::mutex> lock(m_callbackMutex);
        for (const auto& callback : m_eventCallbacks) {
            callback(event);
        }
    }
};

// OSCEANGPUFramework实现
OSCEANGPUFramework::OSCEANGPUFramework() 
    : m_impl(std::make_unique<Impl>()) {
}

OSCEANGPUFramework::~OSCEANGPUFramework() = default;

void OSCEANGPUFramework::createInstance() {
    s_instance.reset(new OSCEANGPUFramework());
}

OSCEANGPUFramework& OSCEANGPUFramework::getInstance() {
    boost::call_once(s_onceFlag, &OSCEANGPUFramework::createInstance);
    return *s_instance;
}

bool OSCEANGPUFramework::initialize(const GPUFrameworkConfig& config) {
    return getInstance().m_impl->initialize(config);
}

void OSCEANGPUFramework::shutdown() {
    getInstance().m_impl->shutdown();
}

void OSCEANGPUFramework::destroy() {
    boost::lock_guard<boost::mutex> lock(s_mutex);
    if (s_instance) {
        s_instance->m_impl->shutdown();
        s_instance.reset();
    }
}

bool OSCEANGPUFramework::isInitialized() {
    return getInstance().m_impl->isInitialized();
}

GPUFrameworkStatus OSCEANGPUFramework::getStatus() {
    return getInstance().m_impl->getStatus();
}

const std::vector<GPUDeviceInfo>& OSCEANGPUFramework::getAvailableDevices() {
    return getInstance().m_impl->getAvailableDevices();
}

UnifiedGPUManager& OSCEANGPUFramework::getGPUManager() {
    return getInstance().m_impl->getGPUManager();
}

MultiGPUScheduler& OSCEANGPUFramework::getScheduler() {
    return getInstance().m_impl->getScheduler();
}

MultiGPUMemoryManager& OSCEANGPUFramework::getMemoryManager() {
    return getInstance().m_impl->getMemoryManager();
}

std::string OSCEANGPUFramework::submitTask(std::shared_ptr<IGPUTask> task,
                                          GPUTaskPriority priority) {
    return m_impl->submitTask(task, priority);
}

bool OSCEANGPUFramework::waitForTask(const std::string& taskId, int timeoutMs) {
    return m_impl->waitForTask(taskId, timeoutMs);
}

bool OSCEANGPUFramework::cancelTask(const std::string& taskId) {
    return m_impl->cancelTask(taskId);
}

boost::optional<GPUTaskExecutionContext> OSCEANGPUFramework::getTaskStatus(const std::string& taskId) const {
    return m_impl->getTaskStatus(taskId);
}

GPUMemoryHandle OSCEANGPUFramework::allocateMemory(size_t size, int deviceId) {
    return m_impl->allocateMemory(size, deviceId);
}

void OSCEANGPUFramework::deallocateMemory(const GPUMemoryHandle& handle) {
    m_impl->deallocateMemory(handle);
}

bool OSCEANGPUFramework::transferData(const GPUMemoryHandle& source,
                                     const GPUMemoryHandle& destination,
                                     bool async) {
    return m_impl->transferData(source, destination, async);
}

void OSCEANGPUFramework::registerEventCallback(FrameworkEventCallback callback) {
    m_impl->registerEventCallback(callback);
}

void OSCEANGPUFramework::setPerformanceCallback(std::function<void(const GPUFrameworkStatus&)> callback) {
    m_impl->setPerformanceCallback(callback);
}

void OSCEANGPUFramework::refreshDevices() {
    m_impl->refreshDevices();
}

std::vector<int> OSCEANGPUFramework::getRecommendedDevices(const std::string& taskType) const {
    return m_impl->getRecommendedDevices(taskType);
}

void OSCEANGPUFramework::optimizeConfiguration(const std::string& workloadProfile) {
    m_impl->optimizeConfiguration(workloadProfile);
}

std::string OSCEANGPUFramework::generatePerformanceReport() const {
    return m_impl->generatePerformanceReport();
}

void OSCEANGPUFramework::reset() {
    m_impl->reset();
}

// GPU框架工具函数实现
namespace GPUFrameworkUtils {

bool checkGPUSupport() {
    UnifiedGPUManager gpuManager;
    auto devices = gpuManager.detectAllGPUs();
    return !devices.empty();
}

GPUFrameworkConfig getBestConfiguration() {
    GPUFrameworkConfig config;
    
    UnifiedGPUManager gpuManager;
    auto devices = gpuManager.detectAllGPUs();
    if (devices.empty()) {
        return config;
    }
    
    // 根据设备特性调整配置
    size_t totalMemory = 0;
    int maxPerformance = 0;
    
    for (const auto& device : devices) {
        totalMemory += device.memoryDetails.totalGlobalMemory;
        maxPerformance = std::max(maxPerformance, device.performanceScore);
    }
    
    // 调整内存池大小
    config.memoryPoolConfig.initialPoolSize = 
        std::min(totalMemory / devices.size() / 4, size_t(512 * 1024 * 1024));
    
    // 选择调度策略
    if (devices.size() > 1) {
        config.schedulerConfig.strategy = SchedulingStrategy::LEAST_LOADED;
    } else if (maxPerformance >= 80) {
        config.schedulerConfig.strategy = SchedulingStrategy::PERFORMANCE_BASED;
    }
    
    return config;
}

bool validateConfiguration(const GPUFrameworkConfig& config) {
    // 验证内存配置
    if (config.memoryPoolConfig.initialPoolSize > config.memoryPoolConfig.maxPoolSize) {
        return false;
    }
    
    if (config.memoryPoolConfig.blockSize == 0) {
        return false;
    }
    
    // 验证传输配置
    if (config.transferConfig.maxConcurrentTransfers <= 0) {
        return false;
    }
    
    return true;
}

std::string formatDeviceInfo(const GPUDeviceInfo& device) {
    std::stringstream ss;
    
    ss << "Device " << device.deviceId << ": " << device.name << "\n";
    ss << "  Vendor: ";
    switch (device.vendor) {
        case GPUVendor::NVIDIA: ss << "NVIDIA"; break;
        case GPUVendor::AMD: ss << "AMD"; break;
        case GPUVendor::INTEL: ss << "Intel"; break;
        case GPUVendor::APPLE: ss << "Apple"; break;
        default: ss << "Unknown"; break;
    }
    ss << "\n";
    
    ss << "  Memory: " << (device.memoryDetails.totalGlobalMemory / (1024 * 1024)) << " MB\n";
    ss << "  Compute Units: " << device.computeUnits.multiprocessorCount << "\n";
    ss << "  Clock Speed: " << device.clockInfo.baseClock << " MHz\n";
    ss << "  Performance Score: " << device.performanceScore << "/100\n";
    
    ss << "  Supported APIs: ";
    for (const auto& api : device.supportedAPIs) {
        switch (api) {
            case ComputeAPI::CUDA: ss << "CUDA "; break;
            case ComputeAPI::ROCM_HIP: ss << "ROCm/HIP "; break;
            case ComputeAPI::OPENCL: ss << "OpenCL "; break;
            case ComputeAPI::LEVEL_ZERO: ss << "Level Zero "; break;
            case ComputeAPI::METAL: ss << "Metal "; break;
            case ComputeAPI::DIRECTCOMPUTE: ss << "DirectCompute "; break;
            default: break;
        }
    }
    ss << "\n";
    
    return ss.str();
}

double estimateExecutionTime(const IGPUTask& task, int deviceId) {
    // 简单的执行时间估算
    double baseTime = 10.0; // 基础时间 (ms)
    
    // 根据内存需求调整
    size_t memReq = task.estimateMemoryRequirement();
    if (memReq > 0) {
        baseTime += (memReq / (1024.0 * 1024.0)) * 0.1; // 每MB增加0.1ms
    }
    
    // 根据计算复杂度调整
    double complexity = task.estimateComputeComplexity();
    baseTime *= (1.0 + complexity);
    
    // TODO: 根据具体设备性能调整
    
    return baseTime;
}

} // namespace GPUFrameworkUtils

} // namespace oscean::common_utils::gpu
