/**
 * @file unified_thread_pool_manager.h
 * @brief 统一线程池管理器 - 智能资源管理和任务调度
 * 
 * 🎯 核心目标：
 * ✅ 统一线程池生命周期管理，解决析构卡死问题
 * ✅ 智能任务调度，避免资源竞争
 * ✅ 支持测试模式的优雅退化
 * ✅ 高效的多文件并发处理
 * 
 * 🔧 设计原则：
 * - 共享而非独占：所有模块共享统一的线程池
 * - 智能而非强制：根据任务特性智能调度
 * - 优雅而非粗暴：渐进式关闭，避免强制等待
 */

#pragma once

// 启用boost::asio支持
#define OSCEAN_ENABLE_BOOST_ASIO
#include "../utilities/boost_config.h"
OSCEAN_ENABLE_BOOST_ASIO_IN_MODULE();  // 线程池管理器需要使用boost::asio

#include "../async/async_framework.h"
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/thread/future.hpp>

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <queue>
#include <condition_variable>

namespace oscean::common_utils::infrastructure {

/**
 * @brief 任务类型 - 用于智能调度
 */
enum class TaskType {
    CPU_INTENSIVE,      // CPU密集型：数值计算、坐标转换
    IO_BOUND,          // I/O密集型：文件读写、网络请求
    MEMORY_INTENSIVE,   // 内存密集型：大数据处理、缓存操作
    QUICK_TASK,        // 快速任务：元数据查询、简单计算
    BACKGROUND,        // 后台任务：清理、统计
    INTERACTIVE        // 交互任务：用户请求响应
};

/**
 * @brief 任务优先级
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief 🔧 智能任务调度器
 */
class IntelligentTaskScheduler {
public:
    struct TaskInfo {
        std::string taskId;
        TaskType type;
        TaskPriority priority;
        std::chrono::steady_clock::time_point submitTime;
        std::vector<std::string> resourceHints;  // 资源提示（如文件路径）
        size_t estimatedDurationMs = 0;
    };
    
    /**
     * @brief 选择最佳线程池
     */
    size_t selectOptimalPool(const TaskInfo& task, const std::vector<size_t>& availableThreads) const;
    
    /**
     * @brief 检查资源冲突
     */
    bool hasResourceConflict(const TaskInfo& newTask, const std::vector<TaskInfo>& runningTasks) const;
    
    /**
     * @brief 建议任务延迟时间（毫秒）
     */
    std::chrono::milliseconds suggestDelay(const TaskInfo& task) const;

private:
    mutable std::mutex mutex_;
    std::vector<TaskInfo> recentTasks_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> resourceLastAccess_;
};

/**
 * @brief 线程池统计信息
 */
struct ThreadPoolStatistics {
    size_t totalThreads = 0;
    size_t activeThreads = 0;
    size_t idleThreads = 0;
    size_t queuedTasks = 0;
    size_t completedTasks = 0;
    size_t failedTasks = 0;
    double averageTaskTime = 0.0;
    double utilizationRatio = 0.0;
    std::chrono::steady_clock::time_point lastUpdate;
    
    std::string toString() const;
};

/**
 * @brief 🔧 统一线程池管理器 - 新架构
 */
class UnifiedThreadPoolManager {
public:
    /**
     * @brief 线程池配置
     */
    struct PoolConfiguration {
        size_t minThreads = 1;
        size_t maxThreads = std::thread::hardware_concurrency();
        std::chrono::seconds threadIdleTimeout{300};  // 5分钟
        bool enableDynamicScaling = true;
        bool enableTaskPriority = true;
    };
    
    /**
     * @brief 构造函数
     */
    explicit UnifiedThreadPoolManager(const PoolConfiguration& config = {});
    
    /**
     * @brief 析构函数 - 🔧 渐进式优雅关闭
     */
    ~UnifiedThreadPoolManager();
    
    // 禁用复制和移动
    UnifiedThreadPoolManager(const UnifiedThreadPoolManager&) = delete;
    UnifiedThreadPoolManager& operator=(const UnifiedThreadPoolManager&) = delete;
    UnifiedThreadPoolManager(UnifiedThreadPoolManager&&) = delete;
    UnifiedThreadPoolManager& operator=(UnifiedThreadPoolManager&&) = delete;
    
    // === 🔧 模式控制 ===
    
    /**
     * @brief 设置运行模式
     */
    enum class RunMode {
        PRODUCTION,    // 生产模式：完整多线程
        TESTING,       // 测试模式：受控并发
        SINGLE_THREAD, // 单线程模式：顺序执行
        DEBUG          // 调试模式：详细日志
    };
    
    void setRunMode(RunMode mode);
    RunMode getRunMode() const { return runMode_.load(); }
    
    // === 🚀 智能任务提交接口 ===
    
    /**
     * @brief 提交智能调度任务
     */
    template<typename Func>
    auto submitTask(Func&& func, 
                   TaskType taskType = TaskType::CPU_INTENSIVE,
                   TaskPriority priority = TaskPriority::NORMAL,
                   const std::vector<std::string>& resourceHints = {})
        -> boost::future<std::invoke_result_t<Func>>;
    
    /**
     * @brief 批量文件处理 - 🔧 智能并发控制
     */
    template<typename Func>
    auto processBatchFiles(const std::vector<std::string>& filePaths,
                          Func&& processor,
                          size_t maxConcurrency = 0)  // 0 = 自动检测
        -> boost::future<std::vector<std::invoke_result_t<Func, std::string>>>;
    
    /**
     * @brief 提交文件处理任务 - 智能资源管理
     */
    template<typename Func>
    auto submitFileTask(Func&& func,
                       const std::string& filePath,
                       TaskPriority priority = TaskPriority::NORMAL)
        -> boost::future<std::invoke_result_t<Func>>;
    
    // === 📊 监控和管理 ===
    
    /**
     * @brief 获取实时统计
     */
    ThreadPoolStatistics getStatistics() const;
    
    /**
     * @brief 获取健康状态
     */
    struct HealthStatus {
        bool healthy = true;
        double cpuUtilization = 0.0;
        size_t pendingTasks = 0;
        std::vector<std::string> warnings;
        std::vector<std::string> recommendations;
    };
    
    HealthStatus getHealthStatus() const;
    
    /**
     * @brief 性能调优建议
     */
    std::vector<std::string> getPerformanceSuggestions() const;
    
    // === 🔧 生命周期管理 ===
    
    /**
     * @brief 请求优雅关闭
     */
    void requestShutdown(std::chrono::seconds timeout = std::chrono::seconds{30});
    
    /**
     * @brief 检查是否正在关闭
     */
    bool isShuttingDown() const { return shutdownRequested_.load(); }
    
    /**
     * @brief 等待所有任务完成
     */
    bool waitForCompletion(std::chrono::seconds timeout = std::chrono::seconds{10});

private:
    // === 内部状态 ===
    std::unique_ptr<boost::asio::thread_pool> primaryPool_;
    std::unique_ptr<boost::asio::thread_pool> ioPool_;        // 专用I/O池
    std::unique_ptr<boost::asio::thread_pool> quickPool_;     // 快速任务池
    
    std::unique_ptr<IntelligentTaskScheduler> scheduler_;
    
    PoolConfiguration config_;
    std::atomic<RunMode> runMode_{RunMode::PRODUCTION};
    
    // 状态控制
    std::atomic<bool> shutdownRequested_{false};
    std::atomic<size_t> activeTasks_{0};
    std::atomic<size_t> completedTasks_{0};
    
    mutable std::mutex statsMutex_;
    ThreadPoolStatistics stats_;
    
    // === 内部方法 ===
    void initializePools();
    boost::asio::thread_pool& selectPool(TaskType taskType);
    void updateStatistics();
    void gracefulShutdown(std::chrono::seconds timeout);
    
    // 任务执行包装器
    template<typename Func>
    void executeTask(Func&& func, const IntelligentTaskScheduler::TaskInfo& taskInfo);
};

// === 🚀 模板实现 ===

template<typename Func>
auto UnifiedThreadPoolManager::submitTask(Func&& func, 
                                         TaskType taskType,
                                         TaskPriority priority,
                                         const std::vector<std::string>& resourceHints)
    -> boost::future<std::invoke_result_t<Func>> {
    
    using ResultType = std::invoke_result_t<Func>;
    
    // 🔧 单线程模式：直接执行
    if (runMode_.load() == RunMode::SINGLE_THREAD) {
        auto promise = std::make_shared<boost::promise<ResultType>>();
        auto future = promise->get_future();
        
        try {
            if constexpr (std::is_void_v<ResultType>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
        
        return future;
    }
    
    auto promise = std::make_shared<boost::promise<ResultType>>();
    auto future = promise->get_future();
    
    // 创建任务信息
    IntelligentTaskScheduler::TaskInfo taskInfo;
    taskInfo.type = taskType;
    taskInfo.priority = priority;
    taskInfo.resourceHints = resourceHints;
    taskInfo.submitTime = std::chrono::steady_clock::now();
    
    // 智能选择线程池
    auto& pool = selectPool(taskType);
    
    // 增加活跃任务计数
    activeTasks_.fetch_add(1);
    
    boost::asio::post(pool, [this, promise, func = std::forward<Func>(func), taskInfo]() mutable {
        try {
            if constexpr (std::is_void_v<ResultType>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
            
            completedTasks_.fetch_add(1);
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
        
        activeTasks_.fetch_sub(1);
    });
    
    return future;
}

template<typename Func>
auto UnifiedThreadPoolManager::submitFileTask(Func&& func,
                                             const std::string& filePath,
                                             TaskPriority priority)
    -> boost::future<std::invoke_result_t<Func>> {
    
    // 文件任务通常是I/O密集型，提供文件路径作为资源提示
    return submitTask(std::forward<Func>(func), 
                     TaskType::IO_BOUND, 
                     priority, 
                     {filePath});
}

template<typename Func>
auto UnifiedThreadPoolManager::processBatchFiles(const std::vector<std::string>& filePaths,
                                                 Func&& processor,
                                                 size_t maxConcurrency)
    -> boost::future<std::vector<std::invoke_result_t<Func, std::string>>> {
    
    using ResultType = std::invoke_result_t<Func, std::string>;
    
    auto promise = std::make_shared<boost::promise<std::vector<ResultType>>>();
    auto future = promise->get_future();
    
    // 🔧 智能并发控制
    if (maxConcurrency == 0) {
        // 自动检测：根据文件数量和系统资源决定并发度
        maxConcurrency = std::min(filePaths.size(), 
                                 static_cast<size_t>(std::thread::hardware_concurrency()));
    }
    
    // 🔧 单线程模式：顺序处理
    if (runMode_.load() == RunMode::SINGLE_THREAD) {
        try {
            std::vector<ResultType> results;
            results.reserve(filePaths.size());
            
            for (const auto& filePath : filePaths) {
                if constexpr (std::is_void_v<ResultType>) {
                    processor(filePath);
                } else {
                    results.push_back(processor(filePath));
                }
            }
            
            if constexpr (std::is_void_v<ResultType>) {
                promise->set_value(std::vector<ResultType>{});
            } else {
                promise->set_value(std::move(results));
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
        
        return future;
    }
    
    // 🚀 多线程模式：智能批处理
    // 使用条件变量和计数器控制并发度，兼容C++17
    auto activeCount = std::make_shared<std::atomic<size_t>>(0);
    auto maxConcur = std::make_shared<size_t>(maxConcurrency);
    auto concurrencyMutex = std::make_shared<std::mutex>();
    auto concurrencyCV = std::make_shared<std::condition_variable>();
    
    auto results = std::make_shared<std::vector<ResultType>>(filePaths.size());
    auto completed = std::make_shared<std::atomic<size_t>>(0);
    auto hasError = std::make_shared<std::atomic<bool>>(false);
    auto errorPtr = std::make_shared<std::exception_ptr>();
    
    for (size_t i = 0; i < filePaths.size(); ++i) {
        submitTask([=, processor = std::forward<Func>(processor)]() {
            // 等待并发控制
            {
                std::unique_lock<std::mutex> lock(*concurrencyMutex);
                concurrencyCV->wait(lock, [=]() { 
                    return activeCount->load() < *maxConcur; 
                });
                activeCount->fetch_add(1);
            }
            
            try {
                if (!hasError->load()) {
                    if constexpr (std::is_void_v<ResultType>) {
                        processor(filePaths[i]);
                    } else {
                        (*results)[i] = processor(filePaths[i]);
                    }
                }
            } catch (...) {
                hasError->store(true);
                *errorPtr = std::current_exception();
            }
            
            // 释放并发控制
            {
                std::lock_guard<std::mutex> lock(*concurrencyMutex);
                activeCount->fetch_sub(1);
            }
            concurrencyCV->notify_one();
            
            // 检查是否完成
            if (completed->fetch_add(1) + 1 == filePaths.size()) {
                if (hasError->load()) {
                    promise->set_exception(*errorPtr);
                } else {
                    if constexpr (std::is_void_v<ResultType>) {
                        promise->set_value(std::vector<ResultType>{});
                    } else {
                        promise->set_value(*results);
                    }
                }
            }
        }, TaskType::IO_BOUND, TaskPriority::NORMAL, {filePaths[i]});
    }
    
    return future;
}

} // namespace oscean::common_utils::infrastructure 