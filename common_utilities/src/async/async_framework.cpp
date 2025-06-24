/**
 * @file async_framework.cpp
 * @brief 统一异步框架实现
 * 
 * 实现统一异步框架的核心功能，整合了原 async_composition、async_context、
 * async_enhanced、async_patterns 等模块的核心逻辑
 */

#include "common_utils/async/async_framework.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/utilities/exceptions.h"
#include <boost/asio/post.hpp>
#include <sstream>
#include <random>
#include <algorithm>
#include <thread>
#include <iostream>

namespace oscean::common_utils::async {

// === TaskMetadata 实现 ===

std::chrono::milliseconds TaskMetadata::getDuration() const {
    auto endTimePoint = (status == TaskStatus::COMPLETED || status == TaskStatus::FAILED) 
                        ? endTime : std::chrono::steady_clock::now();
    
    if (startTime.time_since_epoch().count() == 0) {
        return std::chrono::milliseconds{0};
    }
    
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTime);
}

std::string TaskMetadata::toString() const {
    std::ostringstream oss;
    oss << "Task[" << taskId << "] " << taskName 
        << " Priority:" << static_cast<int>(priority)
        << " Status:" << static_cast<int>(status)
        << " Duration:" << getDuration().count() << "ms";
    
    if (!errorMessage.empty()) {
        oss << " Error:" << errorMessage;
    }
    
    return oss.str();
}

// === AsyncStatistics 实现 ===

double AsyncStatistics::getSuccessRate() const {
    if (totalTasksSubmitted == 0) return 0.0;
    return static_cast<double>(totalTasksCompleted) / totalTasksSubmitted;
}

double AsyncStatistics::getFailureRate() const {
    if (totalTasksSubmitted == 0) return 0.0;
    return static_cast<double>(totalTasksFailed) / totalTasksSubmitted;
}

std::string AsyncStatistics::toString() const {
    std::ostringstream oss;
    oss << "AsyncStats[Submitted:" << totalTasksSubmitted
        << " Completed:" << totalTasksCompleted  
        << " Failed:" << totalTasksFailed
        << " Active:" << currentActiveTasks
        << " AvgTime:" << averageExecutionTime << "ms"
        << " QueueSize:" << queueSize
        << " ThreadUtil:" << (threadPoolUtilization * 100) << "%]";
    return oss.str();
}

// === RetryPolicy 实现 ===

std::chrono::milliseconds RetryPolicy::calculateDelay(size_t retryCount) const {
    if (retryCount == 0) return std::chrono::milliseconds{0};
    
    auto delay = static_cast<double>(baseDelay.count()) * 
                 std::pow(backoffMultiplier, retryCount - 1);
    
    auto result = std::chrono::milliseconds{static_cast<long long>(delay)};
    result = std::min(result, maxDelay);
    
    if (enableJitter) {
        // 添加±25%的随机抖动
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.75, 1.25);
        result = std::chrono::milliseconds{
            static_cast<long long>(result.count() * dis(gen))
        };
    }
    
    return result;
}

std::string RetryPolicy::toString() const {
    std::ostringstream oss;
    oss << "RetryPolicy[MaxRetries:" << maxRetries
        << " BaseDelay:" << baseDelay.count() << "ms"
        << " Multiplier:" << backoffMultiplier
        << " MaxDelay:" << maxDelay.count() << "ms"
        << " Jitter:" << (enableJitter ? "Yes" : "No") << "]";
    return oss.str();
}

// === ResourceLimits 实现 ===

bool ResourceLimits::hasLimits() const {
    return maxConcurrentTasks > 0 || maxMemoryUsage > 0 || 
           maxQueueLength > 0 || maxTaskDuration.count() > 0;
}

std::string ResourceLimits::toString() const {
    std::ostringstream oss;
    oss << "ResourceLimits[";
    if (maxConcurrentTasks > 0) oss << "MaxTasks:" << maxConcurrentTasks << " ";
    if (maxMemoryUsage > 0) oss << "MaxMemory:" << (maxMemoryUsage / 1024 / 1024) << "MB ";
    if (maxQueueLength > 0) oss << "MaxQueue:" << maxQueueLength << " ";
    if (maxTaskDuration.count() > 0) oss << "MaxDuration:" << maxTaskDuration.count() << "ms ";
    oss << "]";
    return oss.str();
}

// === AsyncFramework 实现 ===

AsyncFramework::AsyncFramework(std::shared_ptr<boost::asio::thread_pool> threadPool)
    : threadPool_(std::move(threadPool))
    , taskMonitor_(std::make_unique<TaskMonitor>()) {
    
    if (!threadPool_) {
        throw std::invalid_argument("ThreadPool cannot be null");
    }
    
    // 初始化统计信息
    resetStatistics();
    
    // 完全移除构造函数中的日志调用，避免静态初始化问题
    // 日志记录将在第一次实际使用时进行
}

AsyncFramework::~AsyncFramework() {
    if (!shuttingDown_.load()) {
        emergencyShutdown();
    }
}

std::string AsyncFramework::generateTaskId() {
    return "task_" + std::to_string(nextTaskId_.fetch_add(1));
}

TaskMetadata AsyncFramework::createTaskMetadata(const std::string& taskName, TaskPriority priority) {
    TaskMetadata metadata;
    metadata.taskId = generateTaskId();
    metadata.taskName = taskName.empty() ? "unnamed_task" : taskName;
    metadata.priority = priority;
    metadata.createdTime = std::chrono::steady_clock::now();
    metadata.status = TaskStatus::PENDING;
    
    return metadata;
}

void AsyncFramework::updateStatistics(const TaskMetadata& metadata) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    switch (metadata.status) {
        case TaskStatus::PENDING:
            statistics_.totalTasksSubmitted++;
            statistics_.currentActiveTasks++;
            break;
            
        case TaskStatus::COMPLETED:
            statistics_.totalTasksCompleted++;
            statistics_.currentActiveTasks--;
            
            // 更新平均执行时间
            {
                auto duration = metadata.getDuration().count();
                auto totalCompleted = statistics_.totalTasksCompleted;
                statistics_.averageExecutionTime = 
                    (statistics_.averageExecutionTime * (totalCompleted - 1) + duration) / totalCompleted;
            }
            break;
            
        case TaskStatus::FAILED:
        case TaskStatus::TIMEOUT:
        case TaskStatus::CANCELLED:
            statistics_.totalTasksFailed++;
            statistics_.currentActiveTasks--;
            break;
            
        default:
            break;
    }
}

void AsyncFramework::notifyTaskEvent(const TaskMetadata& metadata, const std::string& event) {
    if (!taskMonitor_) return;
    
    // 这里应该调用 TaskMonitor 的相应回调
    // 由于模板和回调的复杂性，这里只提供基础框架
}

AsyncStatistics AsyncFramework::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return statistics_;
}

void AsyncFramework::resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    statistics_ = AsyncStatistics{};
}

AsyncFramework::TaskMonitor& AsyncFramework::getTaskMonitor() {
    return *taskMonitor_;
}

void AsyncFramework::setConfig(const AsyncConfig& config) {
    config_ = config;
    
    // 暂时移除日志调用，避免静态初始化问题
    // 在确认框架稳定后可以重新添加
}

const AsyncConfig& AsyncFramework::getConfig() const {
    return config_;
}

void AsyncFramework::shutdown() {
    if (shuttingDown_.exchange(true)) {
        return; // 已经在关闭过程中
    }
    
    // 暂时移除日志调用，避免静态初始化问题
    // 在确认框架稳定后可以重新添加
    
    // 等待当前活跃任务完成，但设置较短的超时避免无限等待
    auto start = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(3); // 缩短到3秒超时
    
    // 使用线程安全的方式检查活跃任务
    while (true) {
        size_t activeTasks = 0;
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            activeTasks = statistics_.currentActiveTasks;
        }
        
        if (activeTasks == 0 || std::chrono::steady_clock::now() - start > timeout) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // 停止线程池
    if (threadPool_) {
        threadPool_->stop();
    }
}

void AsyncFramework::emergencyShutdown() {
    shuttingDown_.store(true);
    
    // 暂时移除日志调用，避免静态初始化问题
    // 在确认框架稳定后可以重新添加
    
    // 立即停止线程池并显式调用join以避免析构时卡死
    if (threadPool_) {
        threadPool_->stop();
        
        // 使用boost::thread进行异步清理，避免无限等待
        std::thread cleanupThread([threadPool = threadPool_]() {
            try {
                // 创建一个超时机制 - 使用简单的时间检查
                auto start = std::chrono::steady_clock::now();
                auto timeout = std::chrono::seconds(2);
                
                // 在分离线程中尝试join，设置超时保护
                bool joinCompleted = false;
                
                std::thread joinThread([threadPool, &joinCompleted]() {
                    try {
                        threadPool->join();
                        joinCompleted = true;
                    } catch (...) {
                        // 忽略join过程中的异常
                        joinCompleted = true; // 即使异常也算完成
                    }
                });
                
                // 等待join完成或超时
                while (!joinCompleted && (std::chrono::steady_clock::now() - start) < timeout) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                
                if (!joinCompleted) {
                    std::cout << "Emergency shutdown: join operation timed out, proceeding with cleanup" << std::endl;
                    // 分离join线程，不再等待
                    joinThread.detach();
                } else {
                    // join线程已完成，正常等待其结束
                    if (joinThread.joinable()) {
                        joinThread.join();
                    }
                }
                
            } catch (...) {
                // 忽略所有清理过程中的异常
            }
        });
        
        // 分离清理线程，不等待其完成
        cleanupThread.detach();
    }
}

bool AsyncFramework::isShutdown() const {
    return shuttingDown_.load();
}

// === 静态工厂方法 ===

std::unique_ptr<AsyncFramework> AsyncFramework::createDefault() {
    auto threadPool = std::make_shared<boost::asio::thread_pool>(
        std::thread::hardware_concurrency()
    );
    
    return std::make_unique<AsyncFramework>(threadPool);
}

std::unique_ptr<AsyncFramework> AsyncFramework::createForEnvironment(const std::string& environment) {
    Environment env = Environment::PRODUCTION;
    
    if (environment == "development") env = Environment::DEVELOPMENT;
    else if (environment == "testing") env = Environment::TESTING;
    else if (environment == "production") env = Environment::PRODUCTION;
    else if (environment == "hpc") env = Environment::HPC;
    
    auto config = getEnvironmentSpecificConfig(env);
    auto threadPool = std::make_shared<boost::asio::thread_pool>(config.threadPoolSize);
    
    auto framework = std::make_unique<AsyncFramework>(threadPool);
    framework->setConfig(config);
    
    return framework;
}

std::unique_ptr<AsyncFramework> AsyncFramework::createWithThreadPool(size_t threadCount) {
    auto threadPool = std::make_shared<boost::asio::thread_pool>(threadCount);
    return std::make_unique<AsyncFramework>(threadPool);
}

// === TaskMonitor 实现 ===

void AsyncFramework::TaskMonitor::onTaskStarted(TaskCallback callback) {
    startCallbacks_.push_back(std::move(callback));
}

void AsyncFramework::TaskMonitor::onTaskCompleted(TaskCallback callback) {
    completeCallbacks_.push_back(std::move(callback));
}

void AsyncFramework::TaskMonitor::onTaskFailed(TaskCallback callback) {
    failCallbacks_.push_back(std::move(callback));
}

std::vector<TaskMetadata> AsyncFramework::TaskMonitor::getTaskHistory() const {
    std::lock_guard<std::mutex> lock(historyMutex_);
    return taskHistory_;
}

void AsyncFramework::TaskMonitor::clearHistory() {
    std::lock_guard<std::mutex> lock(historyMutex_);
    taskHistory_.clear();
}

// === CircuitBreaker 实现 ===

AsyncFramework::CircuitBreaker::CircuitBreaker(size_t failureThreshold, 
                                              std::chrono::seconds recoveryTimeout)
    : failureThreshold_(failureThreshold)
    , recoveryTimeout_(recoveryTimeout) {
}

bool AsyncFramework::CircuitBreaker::isOpen() const {
    if (!isOpen_.load()) return false;
    
    auto now = std::chrono::steady_clock::now();
    auto lastFailure = lastFailureTime_.load();
    
    if (now - lastFailure >= recoveryTimeout_) {
        // 尝试从开放状态恢复到半开状态
        const_cast<CircuitBreaker*>(this)->isOpen_.store(false);
        return false;
    }
    
    return true;
}

void AsyncFramework::CircuitBreaker::reset() {
    failureCount_.store(0);
    isOpen_.store(false);
}

void AsyncFramework::CircuitBreaker::recordSuccess() {
    failureCount_.store(0);
}

void AsyncFramework::CircuitBreaker::recordFailure() {
    auto count = failureCount_.fetch_add(1) + 1;
    lastFailureTime_.store(std::chrono::steady_clock::now());
    
    if (count >= failureThreshold_) {
        isOpen_.store(true);
    }
}

std::unique_ptr<AsyncFramework::CircuitBreaker> 
AsyncFramework::createCircuitBreaker(size_t failureThreshold, 
                                    std::chrono::seconds recoveryTimeout) {
    return std::make_unique<CircuitBreaker>(failureThreshold, recoveryTimeout);
}

// === TaskQueue 实现 ===

AsyncFramework::TaskQueue::TaskQueue(size_t maxSize) : maxSize_(maxSize) {
}

size_t AsyncFramework::TaskQueue::size() const {
    return currentSize_.load();
}

size_t AsyncFramework::TaskQueue::capacity() const {
    return maxSize_;
}

bool AsyncFramework::TaskQueue::isFull() const {
    return currentSize_.load() >= maxSize_;
}

void AsyncFramework::TaskQueue::setCapacity(size_t newCapacity) {
    maxSize_ = newCapacity;
}

std::unique_ptr<AsyncFramework::TaskQueue> 
AsyncFramework::createTaskQueue(size_t maxSize) {
    return std::make_unique<TaskQueue>(maxSize);
}

// === AsyncSemaphore 实现 ===

AsyncFramework::AsyncSemaphore::AsyncSemaphore(size_t count) 
    : count_(count), maxPermits_(count), currentPermits_(count) {
}

boost::future<void> AsyncFramework::AsyncSemaphore::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (count_.load() > 0) {
        // 有可用的许可证，直接获取
        count_.fetch_sub(1);
        currentPermits_--;
        
        auto promise = boost::promise<void>();
        auto future = promise.get_future();
        promise.set_value();  // 立即设置为完成状态
        return future;
    } else {
        // 没有可用许可证，创建promise并加入等待队列
        auto promise = boost::promise<void>();
        auto future = promise.get_future();
        waitingQueue_.push(std::move(promise));  // 修复：先创建再入队，而不是从空队列取出
        return future;
    }
}

void AsyncFramework::AsyncSemaphore::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!waitingQueue_.empty()) {
        // 有等待的任务，直接唤醒一个
        auto promise = std::move(waitingQueue_.front());
        waitingQueue_.pop();
        promise.set_value();  // 唤醒等待的任务
    } else {
        // 没有等待任务，增加计数
        if (currentPermits_ < maxPermits_) {
            count_.fetch_add(1);
            currentPermits_++;
        }
    }
}

std::unique_ptr<AsyncFramework::AsyncSemaphore> 
AsyncFramework::createSemaphore(size_t count) {
    return std::make_unique<AsyncSemaphore>(count);
}

} // namespace oscean::common_utils::async 