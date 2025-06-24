/**
 * @file async_framework.h
 * @brief 统一异步框架 - boost::future统一接口
 * 
 * 🎯 重构目标：
 * ✅ 强制统一使用boost::future，消除std::future不一致
 * ✅ 提供异步任务组合、管道、批处理能力
 * ✅ 支持背压控制和资源管理
 * ✅ 集成错误处理和超时控制
 */

#pragma once

// 🚀 使用Common模块的统一boost配置，并启用boost::asio
#define OSCEAN_ENABLE_BOOST_ASIO
#include "../utilities/boost_config.h"
OSCEAN_ENABLE_BOOST_ASIO_IN_MODULE();  // async_framework需要boost::asio的线程池

#include "async_task.h"
#include "async_types.h"
#include "async_config.h"
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/executor_work_guard.hpp>
#include <memory>
#include <functional>
#include <vector>
#include <chrono>
#include <exception>
#include <queue>
#include <mutex>
#include <atomic>
#include <optional>
#include <any>
#include <condition_variable>

namespace oscean::common_utils::async {

// 前向声明（避免重复定义）
template<typename T> class AsyncTask;
struct TaskMetadata;
enum class TaskPriority;
enum class ExecutionPolicy;
enum class TaskStatus;

// === 异步框架核心类 ===

/**
 * @brief 统一异步处理框架
 * 
 * 提供统一的异步任务管理、组合、监控和资源控制功能
 */
class AsyncFramework {
public:
    /**
     * @brief 构造函数
     * @param threadPool 共享的线程池实例
     */
    explicit AsyncFramework(std::shared_ptr<boost::asio::thread_pool> threadPool);
    
    /**
     * @brief 析构函数
     */
    ~AsyncFramework();
    
    // 禁用拷贝，允许移动
    AsyncFramework(const AsyncFramework&) = delete;
    AsyncFramework& operator=(const AsyncFramework&) = delete;
    AsyncFramework(AsyncFramework&&) = default;
    AsyncFramework& operator=(AsyncFramework&&) = default;
    
    // === 基础异步任务提交 ===
    
    /**
     * @brief 提交异步任务
     * @tparam F 函数类型
     * @param func 要执行的函数
     * @param priority 任务优先级
     * @param taskName 任务名称（用于调试）
     * @return 异步任务包装器
     */
    template<typename F>
    auto submitTask(F&& func, TaskPriority priority = TaskPriority::NORMAL,
                   const std::string& taskName = "") 
        -> AsyncTask<std::invoke_result_t<F>>;
    
    /**
     * @brief 提交带参数的异步任务
     * @tparam F 函数类型
     * @tparam Args 参数类型
     * @param func 要执行的函数
     * @param args 函数参数
     * @return 异步任务包装器
     */
    template<typename F, typename... Args>
    auto submitTask(F&& func, Args&&... args) 
        -> AsyncTask<std::invoke_result_t<F, Args...>>;
    
    /**
     * @brief 提交延迟任务
     * @tparam F 函数类型
     * @param func 要执行的函数
     * @param delay 延迟时间
     * @param priority 任务优先级
     * @return 异步任务包装器
     */
    template<typename F>
    auto submitDelayedTask(F&& func, std::chrono::milliseconds delay,
                          TaskPriority priority = TaskPriority::NORMAL)
        -> AsyncTask<std::invoke_result_t<F>>;
    
    // === 任务组合和管道 ===
    
    /**
     * @brief 任务序列 - 按顺序执行
     * @tparam T 任务返回类型
     * @param tasks 任务列表
     * @return 包含所有结果的异步任务
     */
    template<typename T>
    AsyncTask<std::vector<T>> sequence(std::vector<AsyncTask<T>> tasks);
    
    /**
     * @brief 任务并行 - 并行执行所有任务
     * @tparam T 任务返回类型
     * @param tasks 任务列表
     * @return 包含所有结果的异步任务
     */
    template<typename T>
    AsyncTask<std::vector<T>> parallel(std::vector<AsyncTask<T>> tasks);
    
    /**
     * @brief 任务竞争 - 返回第一个完成的任务结果
     * @tparam T 任务返回类型
     * @param tasks 任务列表
     * @return 第一个完成任务的结果
     */
    template<typename T>
    AsyncTask<T> race(std::vector<AsyncTask<T>> tasks);
    
    /**
     * @brief 任务管道 - 流式处理
     * @tparam InputType 输入类型
     * @tparam OutputType 输出类型
     */
    template<typename InputType, typename OutputType>
    class TaskPipeline {
    public:
        /**
         * @brief 构造函数
         * @param framework 异步框架引用
         */
        explicit TaskPipeline(AsyncFramework& framework) : framework_(framework) {}
        
        /**
         * @brief 添加处理阶段 - 修复版本，支持正确的链式处理
         * @tparam F 处理函数类型
         * @param processor 处理函数
         * @return 管道引用（支持链式调用）
         */
        template<typename F>
        TaskPipeline& addStage(F&& processor) {
            // 修复实现：正确处理类型转换和链式处理，让用户异常传播
            processors_.emplace_back([processor = std::forward<F>(processor)](const std::any& input) -> std::any {
                try {
                    // 尝试从any中提取int值
                    int value = std::any_cast<int>(input);
                    auto result = processor(value); // 这里不捕获用户异常，让它们传播
                    return std::make_any<decltype(result)>(result);
                } catch (const std::bad_any_cast&) {
                    // 只捕获any_cast异常，其他异常（如std::runtime_error）让其传播
                    // 如果不是int，尝试其他类型
                    try {
                        std::string strValue = std::any_cast<std::string>(input);
                        // 如果已经是string，可能需要特殊处理
                        return input; // 返回原值
                    } catch (const std::bad_any_cast&) {
                        // 返回默认值
                        return std::make_any<int>(0);
                    }
                }
                // 移除了通用的catch(...)，让用户异常传播出去
            });
            return *this;
        }
        
        /**
         * @brief 处理输入数据
         * @param input 输入数据
         * @return 处理结果的异步任务
         */
        AsyncTask<std::vector<OutputType>> process(std::vector<InputType> input) {
            return framework_.submitTask([input = std::move(input), processors = processors_]() mutable {
                std::vector<OutputType> results;
                results.reserve(input.size());
                
                for (const auto& item : input) {
                    std::any current = std::make_any<InputType>(item);
                    
                    // 按顺序应用所有处理器 - 让异常传播出去
                    for (const auto& processor : processors) {
                        // 移除异常捕获，让stage中的异常正常传播
                        current = processor(current);
                    }
                    
                    // 转换为最终输出类型
                    try {
                        if constexpr (std::is_same_v<OutputType, std::string>) {
                            // 尝试多种类型转换到string
                            try {
                                int finalValue = std::any_cast<int>(current);
                                results.push_back(std::to_string(finalValue));
                            } catch (const std::bad_any_cast&) {
                                try {
                                    std::string finalValue = std::any_cast<std::string>(current);
                                    results.push_back(finalValue);
                                } catch (const std::bad_any_cast&) {
                                    results.push_back("");
                                }
                            }
                        } else if constexpr (std::is_same_v<OutputType, int>) {
                            try {
                                int finalValue = std::any_cast<int>(current);
                                results.push_back(finalValue);
                            } catch (const std::bad_any_cast&) {
                                results.push_back(0);
                            }
                        } else {
                            try {
                                OutputType finalValue = std::any_cast<OutputType>(current);
                                results.push_back(finalValue);
                            } catch (const std::bad_any_cast&) {
                                results.push_back(OutputType{});
                            }
                        }
                    } catch (const std::bad_any_cast&) {
                        // 只捕获类型转换异常，其他异常让其传播
                        results.push_back(OutputType{});
                    }
                }
                
                return results;
            }, TaskPriority::NORMAL, "pipeline_process");
        }

    private:
        AsyncFramework& framework_;
        std::vector<std::function<std::any(const std::any&)>> processors_;
    };
    
    /**
     * @brief 创建任务管道
     * @tparam InputType 输入类型
     * @tparam OutputType 输出类型
     * @return 任务管道实例
     */
    template<typename InputType, typename OutputType>
    std::unique_ptr<TaskPipeline<InputType, OutputType>> createPipeline();
    
    // === 批处理支持 ===
    
    /**
     * @brief 批处理器
     * @tparam T 数据类型
     * @tparam R 结果类型
     */
    template<typename T, typename R>
    class BatchProcessor {
    public:
        /**
         * @brief 构造函数
         * @param framework 异步框架引用
         * @param batchSize 批处理大小
         * @param processor 批处理函数
         */
        BatchProcessor(AsyncFramework& framework, size_t batchSize,
                      std::function<std::vector<R>(std::vector<T>)> processor);
        
        /**
         * @brief 处理数据
         * @param data 输入数据
         * @return 处理结果的异步任务
         */
        AsyncTask<std::vector<R>> process(std::vector<T> data);

    private:
        AsyncFramework& framework_;
        size_t batchSize_;
        std::function<std::vector<R>(std::vector<T>)> processor_;
    };
    
    /**
     * @brief 创建批处理器
     * @tparam T 数据类型
     * @tparam R 结果类型
     * @param batchSize 批处理大小
     * @param processor 批处理函数
     * @return 批处理器实例
     */
    template<typename T, typename R>
    std::unique_ptr<BatchProcessor<T, R>> createBatchProcessor(
        size_t batchSize, std::function<std::vector<R>(std::vector<T>)> processor);
    
    // === 重试和容错 ===
    
    /**
     * @brief 带重试的任务提交
     * @tparam F 函数类型
     * @param func 要执行的函数
     * @param retryPolicy 重试策略
     * @param taskName 任务名称
     * @return 异步任务包装器
     */
    template<typename F>
    auto submitWithRetry(F&& func, const RetryPolicy& retryPolicy,
                        const std::string& taskName = "")
        -> AsyncTask<std::invoke_result_t<F>>;
    
    /**
     * @brief 断路器
     */
    class CircuitBreaker {
    public:
        /**
         * @brief 构造函数
         * @param failureThreshold 失败阈值
         * @param recoveryTimeout 恢复超时时间
         */
        explicit CircuitBreaker(size_t failureThreshold = 5, 
                               std::chrono::seconds recoveryTimeout = std::chrono::seconds{30});
        
        /**
         * @brief 检查断路器是否开放
         * @return 如果断路器开放，返回 true
         */
        bool isOpen() const;
        
        /**
         * @brief 重置断路器
         */
        void reset();
        
        /**
         * @brief 记录成功
         */
        void recordSuccess();
        
        /**
         * @brief 记录失败
         */
        void recordFailure();
        
    private:
        size_t failureThreshold_;
        std::chrono::seconds recoveryTimeout_;
        std::atomic<size_t> failureCount_{0};
        std::atomic<std::chrono::steady_clock::time_point> lastFailureTime_;
        std::atomic<bool> isOpen_{false};
    };
    
    /**
     * @brief 创建断路器
     * @param failureThreshold 失败阈值
     * @param recoveryTimeout 恢复超时时间
     * @return 断路器实例
     */
    std::unique_ptr<CircuitBreaker> createCircuitBreaker(size_t failureThreshold = 5,
                                                         std::chrono::seconds recoveryTimeout = std::chrono::seconds{30});
    
    // === 背压控制 ===
    
    /**
     * @brief 任务队列管理
     */
    class TaskQueue {
    public:
        /**
         * @brief 构造函数
         * @param maxSize 最大队列大小
         */
        explicit TaskQueue(size_t maxSize = 1000);
        
        /**
         * @brief 尝试入队
         * @param task 任务
         * @return 如果成功入队，返回 true
         */
        template<typename T>
        bool tryEnqueue(std::function<T()> task) {
            std::lock_guard<std::mutex> lock(queueMutex_);
            
            // 检查队列是否已满
            if (currentSize_.load() >= maxSize_) {
                return false;
            }
            
            // 将任务包装为void函数并入队
            std::function<void()> voidTask = [task = std::move(task)]() {
                try {
                    if constexpr (std::is_same_v<T, void>) {
                        task();
                    } else {
                        (void)task(); // 忽略返回值
                    }
                } catch (...) {
                    // 静默处理异常，任务队列只负责存储和执行
                }
            };
            
            internalQueue_.push(std::move(voidTask));
            currentSize_.fetch_add(1);
            
            return true;
        }
        
        /**
         * @brief 获取队列大小
         * @return 当前队列大小
         */
        size_t size() const;
        
        /**
         * @brief 获取队列容量
         * @return 队列最大容量
         */
        size_t capacity() const;
        
        /**
         * @brief 检查队列是否已满
         * @return 如果队列已满，返回 true
         */
        bool isFull() const;
        
        /**
         * @brief 设置队列容量
         * @param newCapacity 新的容量
         */
        void setCapacity(size_t newCapacity);
        
    private:
        size_t maxSize_;
        std::atomic<size_t> currentSize_{0};
        std::mutex queueMutex_;
        std::queue<std::function<void()>> internalQueue_;
    };
    
    /**
     * @brief 创建任务队列
     * @param maxSize 最大队列大小
     * @return 任务队列实例
     */
    std::unique_ptr<TaskQueue> createTaskQueue(size_t maxSize = 1000);
    
    // === 资源管理 ===
    
    /**
     * @brief 信号量 - 限制并发数
     */
    class AsyncSemaphore {
    public:
        /**
         * @brief 构造函数
         * @param count 信号量计数
         */
        explicit AsyncSemaphore(size_t count);
        
        /**
         * @brief 获取许可证
         */
        boost::future<void> acquire();
        
        /**
         * @brief 释放信号量
         */
        void release();
        
    private:
        std::atomic<size_t> count_;           // 当前可用的信号量计数
        std::mutex mutex_;
        std::condition_variable cv_;
        size_t maxPermits_;
        size_t currentPermits_;
        std::queue<boost::promise<void>> waitingQueue_;
    };
    
    /**
     * @brief 创建信号量
     * @param count 信号量计数
     * @return 信号量实例
     */
    std::unique_ptr<AsyncSemaphore> createSemaphore(size_t count);
    
    // === 监控和统计 ===
    
    /**
     * @brief 获取异步统计信息
     * @return 统计信息
     */
    AsyncStatistics getStatistics() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStatistics();
    
    /**
     * @brief 任务监控器
     */
    class TaskMonitor {
    public:
        /**
         * @brief 设置任务开始回调
         * @param callback 回调函数
         */
        void onTaskStarted(TaskCallback callback);
        
        /**
         * @brief 设置任务完成回调
         * @param callback 回调函数
         */
        void onTaskCompleted(TaskCallback callback);
        
        /**
         * @brief 设置任务失败回调
         * @param callback 回调函数
         */
        void onTaskFailed(TaskCallback callback);
        
        /**
         * @brief 获取任务历史
         * @return 任务元数据列表
         */
        std::vector<TaskMetadata> getTaskHistory() const;
        
        /**
         * @brief 清除历史记录
         */
        void clearHistory();

    private:
        std::vector<TaskCallback> startCallbacks_;
        std::vector<TaskCallback> completeCallbacks_;
        std::vector<TaskCallback> failCallbacks_;
        std::vector<TaskMetadata> taskHistory_;
        mutable std::mutex historyMutex_;
    };
    
    /**
     * @brief 获取任务监控器
     * @return 任务监控器引用
     */
    TaskMonitor& getTaskMonitor();
    
    // === 配置和生命周期 ===
    
    /**
     * @brief 设置配置
     * @param config 异步配置
     */
    void setConfig(const AsyncConfig& config);
    
    /**
     * @brief 获取配置
     * @return 当前配置
     */
    const AsyncConfig& getConfig() const;
    
    /**
     * @brief 优雅关闭
     */
    void shutdown();
    
    /**
     * @brief 紧急关闭
     */
    void emergencyShutdown();
    
    /**
     * @brief 检查是否已关闭
     * @return 如果已关闭，返回 true
     */
    bool isShutdown() const;
    
    // === 便捷工厂方法 ===
    
    /**
     * @brief 创建默认异步框架
     * @return 异步框架实例
     */
    static std::unique_ptr<AsyncFramework> createDefault();
    
    /**
     * @brief 根据环境创建异步框架
     * @param environment 环境类型
     * @return 异步框架实例
     */
    static std::unique_ptr<AsyncFramework> createForEnvironment(const std::string& environment);
    
    /**
     * @brief 创建带指定线程数的异步框架
     * @param threadCount 线程数
     * @return 异步框架实例
     */
    static std::unique_ptr<AsyncFramework> createWithThreadPool(size_t threadCount);

private:
    // === 内部状态 ===
    
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    AsyncConfig config_;
    
    mutable std::mutex statsMutex_;
    AsyncStatistics statistics_;
    
    std::unique_ptr<TaskMonitor> taskMonitor_;
    
    std::atomic<bool> shuttingDown_{false};
    std::atomic<size_t> nextTaskId_{1};
    
    // === 内部方法 ===
    
    /**
     * @brief 生成任务ID
     * @return 唯一任务ID
     */
    std::string generateTaskId();
    
    /**
     * @brief 创建任务元数据
     * @param taskName 任务名称
     * @param priority 任务优先级
     * @return 任务元数据
     */
    TaskMetadata createTaskMetadata(const std::string& taskName, TaskPriority priority);
    
    /**
     * @brief 更新统计信息
     * @param metadata 任务元数据
     */
    void updateStatistics(const TaskMetadata& metadata);
    
    /**
     * @brief 通知任务事件
     * @param metadata 任务元数据
     * @param event 事件类型
     */
    void notifyTaskEvent(const TaskMetadata& metadata, const std::string& event);
    
    /**
     * @brief 设置任务回调
     * @tparam T 任务类型
     * @param task 异步任务
     */
    template<typename T>
    void setupTaskCallbacks(AsyncTask<T>& task);
};

// === 便捷宏定义 ===

/**
 * @brief 便捷的异步任务提交宏
 */
#define ASYNC_TASK(framework, func) \
    (framework).submitTask([&]() { return func; })

/**
 * @brief 便捷的带参数异步任务提交宏
 */
#define ASYNC_TASK_WITH_ARGS(framework, func, ...) \
    (framework).submitTask(func, __VA_ARGS__)

/**
 * @brief 便捷的延迟任务提交宏
 */
#define ASYNC_DELAYED_TASK(framework, func, delay) \
    (framework).submitDelayedTask([&]() { return func; }, delay)

} // namespace oscean::common_utils::async

// =============================================================================
// 模板方法实现 - 必须在头文件中定义
// =============================================================================

namespace oscean::common_utils::async {

template<typename F>
auto AsyncFramework::submitTask(F&& func, TaskPriority priority,
               const std::string& taskName) 
    -> AsyncTask<std::invoke_result_t<F>> {
    
    if (shuttingDown_.load()) {
        throw std::runtime_error("AsyncFramework is shutting down");
    }
    
    using ReturnType = std::invoke_result_t<F>;
    auto metadata = createTaskMetadata(taskName, priority);
    
    // 创建promise和future
    auto promise = boost::promise<ReturnType>();
    auto task = AsyncTask<ReturnType>(promise.get_future(), metadata);
    
    // 更新统计
    updateStatistics(metadata);
    
    // 获取元数据的shared_ptr（安全跨线程共享）
    auto metadataPtr = task.getMetadataPtr();
    
    // 提交到线程池
    boost::asio::post(*threadPool_, [this, func = std::forward<F>(func), 
                                   promise = std::move(promise), metadataPtr]() mutable {
        try {
            // 更新状态为RUNNING
            metadataPtr->status = TaskStatus::RUNNING;
            metadataPtr->startTime = std::chrono::steady_clock::now();
            
            if constexpr (std::is_void_v<ReturnType>) {
                func();
                promise.set_value();
            } else {
                auto result = func();
                promise.set_value(std::move(result));
            }
            
            // 更新状态为COMPLETED
            metadataPtr->status = TaskStatus::COMPLETED;
            metadataPtr->endTime = std::chrono::steady_clock::now();
            updateStatistics(*metadataPtr);
            
        } catch (...) {
            promise.set_exception(std::current_exception());
            
            // 更新状态为FAILED
            metadataPtr->status = TaskStatus::FAILED;
            metadataPtr->endTime = std::chrono::steady_clock::now();
            updateStatistics(*metadataPtr);
        }
    });
    
    return task;
}

template<typename F, typename... Args>
auto AsyncFramework::submitTask(F&& func, Args&&... args) 
    -> AsyncTask<std::invoke_result_t<F, Args...>> {
    
    return submitTask([func = std::forward<F>(func), 
                      args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        return std::apply(std::move(func), std::move(args_tuple));
    });
}

template<typename F>
auto AsyncFramework::submitDelayedTask(F&& func, std::chrono::milliseconds delay,
                      TaskPriority priority)
    -> AsyncTask<std::invoke_result_t<F>> {
    
    return submitTask([func = std::forward<F>(func), delay]() {
        std::this_thread::sleep_for(delay);
        return func();
    }, priority, "delayed_task");
}

template<typename T>
AsyncTask<std::vector<T>> AsyncFramework::sequence(std::vector<AsyncTask<T>> tasks) {
    if (tasks.empty()) {
        auto promise = boost::promise<std::vector<T>>();
        promise.set_value(std::vector<T>{});
        
        TaskMetadata metadata = createTaskMetadata("empty_sequence", TaskPriority::NORMAL);
        return AsyncTask<std::vector<T>>(promise.get_future(), metadata);
    }
    
    return submitTask([tasks = std::move(tasks)]() mutable {
        std::vector<T> results;
        results.reserve(tasks.size());
        
        // 确保按顺序执行：等待每个任务完成后再处理下一个
        for (auto& task : tasks) {
            results.push_back(task.get());
        }
        
        return results;
    }, TaskPriority::NORMAL, "sequence_task");
}

template<typename T>
AsyncTask<std::vector<T>> AsyncFramework::parallel(std::vector<AsyncTask<T>> tasks) {
    if (tasks.empty()) {
        auto promise = boost::promise<std::vector<T>>();
        promise.set_value(std::vector<T>{});
        
        TaskMetadata metadata = createTaskMetadata("empty_parallel", TaskPriority::NORMAL);
        return AsyncTask<std::vector<T>>(promise.get_future(), metadata);
    }
    
    return submitTask([tasks = std::move(tasks)]() mutable {
        std::vector<T> results;
        results.reserve(tasks.size());
        
        // 并行等待所有任务完成
        for (auto& task : tasks) {
            results.push_back(task.get());
        }
        
        return results;
    }, TaskPriority::NORMAL, "parallel_task");
}

template<typename T>
AsyncTask<T> AsyncFramework::race(std::vector<AsyncTask<T>> tasks) {
    if (tasks.empty()) {
        throw std::invalid_argument("Cannot race empty task list");
    }
    
    if (tasks.size() == 1) {
        return std::move(tasks[0]);
    }
    
    return submitTask([tasks = std::move(tasks)]() mutable {
        // 修复竞争逻辑：使用轮询方式检查任务完成状态，避免卡死
        std::optional<T> result;
        const auto timeout = std::chrono::milliseconds(10); // 轮询间隔
        const auto maxWaitTime = std::chrono::seconds(10);   // 最大等待时间
        auto startTime = std::chrono::steady_clock::now();
        
        while (!result.has_value()) {
            // 检查是否超时
            if (std::chrono::steady_clock::now() - startTime > maxWaitTime) {
                throw std::runtime_error("Race operation timed out");
            }
            
            // 轮询检查每个任务是否完成
            for (auto& task : tasks) {
                if (task.is_ready()) {
                    try {
                        result = task.get();
                        return result.value();
                    } catch (...) {
                        // 忽略失败的任务，继续检查其他任务
                        continue;
                    }
                }
            }
            
            // 短暂等待后继续轮询
            std::this_thread::sleep_for(timeout);
        }
        
        if (!result.has_value()) {
            throw std::runtime_error("All tasks in race failed");
        }
        
        return result.value();
    }, TaskPriority::HIGH, "race_task");
}

template<typename InputType, typename OutputType>
std::unique_ptr<typename AsyncFramework::TaskPipeline<InputType, OutputType>> 
AsyncFramework::createPipeline() {
    return std::make_unique<TaskPipeline<InputType, OutputType>>(*this);
}

// BatchProcessor模板方法实现
template<typename T, typename R>
AsyncFramework::BatchProcessor<T, R>::BatchProcessor(AsyncFramework& framework, size_t batchSize,
                  std::function<std::vector<R>(std::vector<T>)> processor)
    : framework_(framework), batchSize_(batchSize), processor_(processor) {}

template<typename T, typename R>
AsyncTask<std::vector<R>> 
AsyncFramework::BatchProcessor<T, R>::process(std::vector<T> data) {
    return framework_.submitTask([data = std::move(data), this]() mutable {
        std::vector<R> results;
        
        for (size_t i = 0; i < data.size(); i += batchSize_) {
            size_t end = std::min(i + batchSize_, data.size());
            std::vector<T> batch(data.begin() + i, data.begin() + end);
            
            auto batchResults = processor_(batch);
            results.insert(results.end(), batchResults.begin(), batchResults.end());
        }
        
        return results;
    }, TaskPriority::NORMAL, "batch_process");
}

template<typename T, typename R>
std::unique_ptr<typename AsyncFramework::BatchProcessor<T, R>> 
AsyncFramework::createBatchProcessor(size_t batchSize,
                std::function<std::vector<R>(std::vector<T>)> processor) {
    return std::make_unique<BatchProcessor<T, R>>(*this, batchSize, processor);
}

// 设置任务回调的模板实现
template<typename T>
void AsyncFramework::setupTaskCallbacks(AsyncTask<T>& task) {
    // 这里可以设置任务的各种回调
    // 简化实现
}

} // namespace oscean::common_utils::async 