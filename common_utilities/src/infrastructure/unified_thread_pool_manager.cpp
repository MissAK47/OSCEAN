/**
 * @file unified_thread_pool_manager.cpp
 * @brief 统一线程池管理器实现
 */

#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <thread>

namespace oscean::common_utils::infrastructure {

// === IntelligentTaskScheduler 实现 ===

size_t IntelligentTaskScheduler::selectOptimalPool(const TaskInfo& task, const std::vector<size_t>& availableThreads) const {
    // 根据任务类型选择最优线程池
    (void)availableThreads; // 标记参数未使用，避免警告
    
    switch (task.type) {
        case TaskType::IO_BOUND:
            return 1; // I/O专用池
        case TaskType::CPU_INTENSIVE:
            return 0; // 主要池
        case TaskType::QUICK_TASK:
            return 2; // 快速任务池
        default:
            return 0; // 默认主要池
    }
}

bool IntelligentTaskScheduler::hasResourceConflict(const TaskInfo& newTask, const std::vector<TaskInfo>& runningTasks) const {
    // 检查资源冲突
    for (const auto& runningTask : runningTasks) {
        // 如果有相同的资源提示（如文件路径），可能存在冲突
        for (const auto& newHint : newTask.resourceHints) {
            for (const auto& runningHint : runningTask.resourceHints) {
                if (newHint == runningHint && !newHint.empty()) {
                    return true; // 找到资源冲突
                }
            }
        }
    }
    return false;
}

std::chrono::milliseconds IntelligentTaskScheduler::suggestDelay(const TaskInfo& task) const {
    // 根据任务优先级和类型建议延迟
    switch (task.priority) {
        case TaskPriority::CRITICAL:
            return std::chrono::milliseconds(0);
        case TaskPriority::HIGH:
            return std::chrono::milliseconds(10);
        case TaskPriority::NORMAL:
            return std::chrono::milliseconds(50);
        case TaskPriority::LOW:
            return std::chrono::milliseconds(200);
        default:
            return std::chrono::milliseconds(100);
    }
}

// === ThreadPoolStatistics 实现 ===

std::string ThreadPoolStatistics::toString() const {
    std::ostringstream oss;
    oss << "ThreadPoolStatistics {\n";
    oss << "  Total Threads: " << totalThreads << "\n";
    oss << "  Active Threads: " << activeThreads << "\n";
    oss << "  Idle Threads: " << idleThreads << "\n";
    oss << "  Queued Tasks: " << queuedTasks << "\n";
    oss << "  Completed Tasks: " << completedTasks << "\n";
    oss << "  Failed Tasks: " << failedTasks << "\n";
    oss << "  Utilization: " << utilizationRatio * 100 << "%\n";
    oss << "  Avg Task Time: " << averageTaskTime << "ms\n";
    oss << "}";
    return oss.str();
}

// === UnifiedThreadPoolManager 实现 ===

UnifiedThreadPoolManager::UnifiedThreadPoolManager(const PoolConfiguration& config)
    : config_(config), scheduler_(std::make_unique<IntelligentTaskScheduler>()) {
    
    std::cout << "UnifiedThreadPoolManager: Initializing with " 
              << config_.minThreads << "-" << config_.maxThreads << " threads" << std::endl;
    
    initializePools();
}

UnifiedThreadPoolManager::~UnifiedThreadPoolManager() {
    if (!shutdownRequested_.load()) {
        requestShutdown(std::chrono::seconds{5});
    }
}

void UnifiedThreadPoolManager::setRunMode(RunMode mode) {
    runMode_.store(mode);
    std::cout << "UnifiedThreadPoolManager: Run mode set to " << static_cast<int>(mode) << std::endl;
    
    if (mode == RunMode::SINGLE_THREAD) {
        // 单线程模式：关闭所有线程池
        if (primaryPool_) primaryPool_->stop();
        if (ioPool_) ioPool_->stop();
        if (quickPool_) quickPool_->stop();
    } else if (mode == RunMode::PRODUCTION || mode == RunMode::TESTING) {
        // 重新初始化线程池
        initializePools();
    }
}

ThreadPoolStatistics UnifiedThreadPoolManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    ThreadPoolStatistics stats = stats_;
    stats.totalThreads = config_.maxThreads;
    stats.activeThreads = activeTasks_.load();
    stats.completedTasks = completedTasks_.load();
    stats.utilizationRatio = static_cast<double>(activeTasks_.load()) / config_.maxThreads;
    stats.lastUpdate = std::chrono::steady_clock::now();
    
    return stats;
}

UnifiedThreadPoolManager::HealthStatus UnifiedThreadPoolManager::getHealthStatus() const {
    HealthStatus status;
    
    auto stats = getStatistics();
    status.healthy = !shutdownRequested_.load();
    status.cpuUtilization = stats.utilizationRatio;
    status.pendingTasks = activeTasks_.load();
    
    if (stats.utilizationRatio > 0.9) {
        status.warnings.push_back("High thread pool utilization");
        status.recommendations.push_back("Consider increasing thread pool size");
    }
    
    if (shutdownRequested_.load()) {
        status.healthy = false;
        status.warnings.push_back("System is shutting down");
    }
    
    return status;
}

std::vector<std::string> UnifiedThreadPoolManager::getPerformanceSuggestions() const {
    std::vector<std::string> suggestions;
    
    auto stats = getStatistics();
    
    if (stats.utilizationRatio > 0.8) {
        suggestions.push_back("Thread pool utilization is high. Consider increasing maxThreads.");
    }
    
    if (stats.averageTaskTime > 1000.0) {
        suggestions.push_back("Average task time is high. Consider task optimization.");
    }
    
    if (activeTasks_.load() > config_.maxThreads * 2) {
        suggestions.push_back("Too many queued tasks. Consider load balancing.");
    }
    
    return suggestions;
}

void UnifiedThreadPoolManager::requestShutdown(std::chrono::seconds timeout) {
    shutdownRequested_.store(true);
    
    std::cout << "UnifiedThreadPoolManager: Shutdown requested with " 
              << timeout.count() << "s timeout" << std::endl;
    
    gracefulShutdown(timeout);
}

bool UnifiedThreadPoolManager::waitForCompletion(std::chrono::seconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    while (activeTasks_.load() > 0) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > timeout) {
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return true;
}

// === 私有方法实现 ===

void UnifiedThreadPoolManager::initializePools() {
    if (runMode_.load() == RunMode::SINGLE_THREAD) {
        std::cout << "UnifiedThreadPoolManager: Single-thread mode, skipping pool initialization" << std::endl;
        return;
    }
    
    try {
        // 主线程池
        size_t primaryThreads = std::max(config_.minThreads, config_.maxThreads / 2);
        primaryPool_ = std::make_unique<boost::asio::thread_pool>(primaryThreads);
        
        // I/O专用线程池
        size_t ioThreads = std::max(size_t{2}, config_.maxThreads / 4);
        ioPool_ = std::make_unique<boost::asio::thread_pool>(ioThreads);
        
        // 快速任务线程池
        size_t quickThreads = std::max(size_t{1}, config_.maxThreads / 8);
        quickPool_ = std::make_unique<boost::asio::thread_pool>(quickThreads);
        
        std::cout << "UnifiedThreadPoolManager: Initialized pools - "
                  << "Primary: " << primaryThreads 
                  << ", I/O: " << ioThreads 
                  << ", Quick: " << quickThreads << std::endl;
        std::cout.flush();
                  
    } catch (const std::exception& e) {
        std::cout << "UnifiedThreadPoolManager: Failed to initialize pools: " << e.what() << std::endl;
        throw;
    }
}

boost::asio::thread_pool& UnifiedThreadPoolManager::selectPool(TaskType taskType) {
    switch (taskType) {
        case TaskType::IO_BOUND:
            if (ioPool_) return *ioPool_;
            break;
        case TaskType::QUICK_TASK:
            if (quickPool_) return *quickPool_;
            break;
        case TaskType::CPU_INTENSIVE:
        case TaskType::MEMORY_INTENSIVE:
        case TaskType::BACKGROUND:
        case TaskType::INTERACTIVE:
        default:
            if (primaryPool_) return *primaryPool_;
            break;
    }
    
    // 回退到主池
    if (primaryPool_) return *primaryPool_;
    
    throw std::runtime_error("No available thread pool");
}

void UnifiedThreadPoolManager::updateStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    stats_.activeThreads = activeTasks_.load();
    stats_.completedTasks = completedTasks_.load();
    stats_.lastUpdate = std::chrono::steady_clock::now();
    
    if (config_.maxThreads > 0) {
        stats_.utilizationRatio = static_cast<double>(stats_.activeThreads) / config_.maxThreads;
    }
}

void UnifiedThreadPoolManager::gracefulShutdown(std::chrono::seconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    // 等待当前任务完成
    while (activeTasks_.load() > 0) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > timeout) {
            std::cout << "UnifiedThreadPoolManager: Shutdown timeout, forcing termination" << std::endl;
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // 停止所有线程池
    if (primaryPool_) {
        primaryPool_->stop();
        primaryPool_->join();
        primaryPool_.reset();
    }
    
    if (ioPool_) {
        ioPool_->stop();
        ioPool_->join();
        ioPool_.reset();
    }
    
    if (quickPool_) {
        quickPool_->stop();
        quickPool_->join();
        quickPool_.reset();
    }
    
    std::cout << "UnifiedThreadPoolManager: Graceful shutdown completed" << std::endl;
}

} // namespace oscean::common_utils::infrastructure 