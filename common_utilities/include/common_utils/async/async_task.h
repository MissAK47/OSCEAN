/**
 * @file async_task.h
 * @brief 异步任务类型和AsyncTask包装器
 * 
 * 从 async_framework.h 拆分出来的异步任务相关定义
 */

#pragma once

// 🚀 使用Common模块的统一boost配置 - 参考CRS模块成功模式
#include "../utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // async_task模块不使用boost::asio，只使用boost::future

// 立即包含boost::future - 参考CRS模块
#include <boost/thread/future.hpp>

// 标准库头文件
#include <memory>
#include <functional>
#include <chrono>
#include <string>

namespace oscean::common_utils::async {

// === 直接使用boost类型，避免模板别名问题 ===
// 注意：不定义UnifiedFuture等别名，直接使用boost::future<T>

// === 异步任务优先级 ===

enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

// === 异步执行策略 ===

enum class ExecutionPolicy {
    PARALLEL,       // 并行执行
    SEQUENTIAL,     // 串行执行
    CONCURRENT,     // 并发执行
    DEFERRED        // 延迟执行
};

// === 异步任务状态 ===

enum class TaskStatus {
    PENDING,        // 等待中
    RUNNING,        // 运行中
    COMPLETED,      // 已完成
    FAILED,         // 失败
    CANCELLED,      // 已取消
    TIMEOUT         // 超时
};

// === 异步任务元数据 ===

struct TaskMetadata {
    std::string taskId;
    std::string taskName;
    TaskPriority priority = TaskPriority::NORMAL;
    std::chrono::steady_clock::time_point createdTime;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
    TaskStatus status = TaskStatus::PENDING;
    std::string errorMessage;
    
    /**
     * @brief 获取任务执行时长
     * @return 任务执行时长，如果任务未完成则返回从开始到现在的时长
     */
    std::chrono::milliseconds getDuration() const;
    
    /**
     * @brief 转换为字符串表示
     * @return 任务元数据的字符串表示
     */
    std::string toString() const;
};

// === 异步任务包装器 ===

/**
 * @brief 异步任务包装器，提供统一的异步任务接口
 * @tparam T 任务返回值类型
 */
template<typename T>
class AsyncTask {
public:
    /**
     * @brief 构造函数
     * @param future Future对象
     * @param metadata 任务元数据
     */
    explicit AsyncTask(boost::future<T> future, const TaskMetadata& metadata)
        : future_(std::move(future))
        , metadata_(std::make_shared<TaskMetadata>(metadata)) {}
    
    /**
     * @brief 默认构造函数（已删除）
     */
    AsyncTask() = delete;
    
    /**
     * @brief 拷贝构造函数（已删除）
     */
    AsyncTask(const AsyncTask&) = delete;
    
    /**
     * @brief 拷贝赋值运算符（已删除）
     */
    AsyncTask& operator=(const AsyncTask&) = delete;
    
    /**
     * @brief 移动构造函数
     */
    AsyncTask(AsyncTask&&) = default;
    
    /**
     * @brief 移动赋值运算符
     */
    AsyncTask& operator=(AsyncTask&&) = default;
    
    /**
     * @brief 析构函数
     */
    ~AsyncTask() = default;

    /**
     * @brief 检查任务是否已完成
     * @return 如果任务已完成，返回 true
     */
    bool is_ready() const {
        // boost::future没有is_ready()方法，使用wait_for(0)来检查
        return future_.wait_for(boost::chrono::milliseconds(0)) == boost::future_status::ready;
    }
    
    /**
     * @brief 等待任务完成并获取结果
     * @return 任务结果
     */
    T get() {
        return future_.get();
    }
    
    /**
     * @brief 检查任务是否有异常
     * @return 如果任务有异常，返回 true
     */
    bool has_exception() const { return future_.has_exception(); }
    
    /**
     * @brief 检查任务是否有值
     * @return 如果任务有值，返回 true
     */
    bool has_value() const { return future_.has_value(); }
    
    /**
     * @brief 等待任务完成（阻塞）
     */
    void wait() { future_.wait(); }
    
    /**
     * @brief 等待任务完成，带超时
     * @tparam Rep 时间单位类型
     * @tparam Period 时间周期类型
     * @param timeout_duration 超时时长
     * @return future 状态
     */
    template<typename Rep, typename Period>
    boost::future_status wait_for(const std::chrono::duration<Rep, Period>& timeout_duration) {
        // 将std::chrono转换为boost::chrono
        auto boost_duration = boost::chrono::duration_cast<boost::chrono::milliseconds>(
            boost::chrono::milliseconds(std::chrono::duration_cast<std::chrono::milliseconds>(timeout_duration).count())
        );
        return future_.wait_for(boost_duration);
    }
    
    /**
     * @brief 任务组合 - 在当前任务完成后执行下一个任务
     * @tparam F 函数类型
     * @param func 后续处理函数
     * @return 新的异步任务
     */
    template<typename F>
    auto then(F&& func) -> AsyncTask<std::invoke_result_t<F, T>>;
    
    /**
     * @brief 错误处理 - 当任务失败时执行处理函数
     * @tparam F 错误处理函数类型
     * @param handler 错误处理函数
     * @return 带错误处理的异步任务
     */
    template<typename F>
    AsyncTask<T> catch_exception(F&& handler);
    
    /**
     * @brief 设置任务超时
     * @param timeout 超时时长
     * @return 带超时的异步任务
     */
    AsyncTask<T> with_timeout(std::chrono::milliseconds timeout);
    
    /**
     * @brief 取消任务
     */
    void cancel();
    
    /**
     * @brief 获取任务元数据（只读）
     * @return 任务元数据的常量引用
     */
    const TaskMetadata& getMetadata() const { return *metadata_; }
    
    /**
     * @brief 获取任务元数据（可修改）
     * @return 任务元数据的引用
     */
    TaskMetadata& getMetadata() { return *metadata_; }
    
    /**
     * @brief 获取元数据的shared_ptr（内部使用）
     */
    std::shared_ptr<TaskMetadata> getMetadataPtr() const { return metadata_; }

private:
    boost::future<T> future_;
    std::shared_ptr<TaskMetadata> metadata_;
};

} // namespace oscean::common_utils::async

// =============================================================================
// AsyncTask 模板方法实现
// =============================================================================

namespace oscean::common_utils::async {

template<typename T>
template<typename F>
auto AsyncTask<T>::then(F&& func) -> AsyncTask<std::invoke_result_t<F, T>> {
    using ReturnType = std::invoke_result_t<F, T>;
    
    // 创建新的promise和future
    auto promise = std::make_shared<boost::promise<ReturnType>>();
    auto newFuture = promise->get_future();
    
    // 创建新的元数据
    TaskMetadata newMetadata = *metadata_;
    newMetadata.taskName += "_then";
    newMetadata.status = TaskStatus::PENDING;
    
    // 异步执行链式操作
    std::thread([future = std::move(future_), promise, func = std::forward<F>(func)]() mutable {
        try {
            if constexpr (std::is_void_v<T>) {
                future.get();
                if constexpr (std::is_void_v<ReturnType>) {
                    func();
                    promise->set_value();
                } else {
                    auto result = func();
                    promise->set_value(std::move(result));
                }
            } else {
                auto result = future.get();
                if constexpr (std::is_void_v<ReturnType>) {
                    func(std::move(result));
                    promise->set_value();
                } else {
                    auto newResult = func(std::move(result));
                    promise->set_value(std::move(newResult));
                }
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }).detach();
    
    return AsyncTask<ReturnType>(std::move(newFuture), newMetadata);
}

template<typename T>
template<typename F>
AsyncTask<T> AsyncTask<T>::catch_exception(F&& handler) {
    auto promise = std::make_shared<boost::promise<T>>();
    auto newFuture = promise->get_future();
    
    TaskMetadata newMetadata = *metadata_;
    newMetadata.taskName += "_catch";
    
    std::thread([future = std::move(future_), promise, handler = std::forward<F>(handler)]() mutable {
        try {
            if constexpr (std::is_void_v<T>) {
                future.get();
                promise->set_value();
            } else {
                auto result = future.get();
                promise->set_value(std::move(result));
            }
        } catch (...) {
            try {
                if constexpr (std::is_void_v<T>) {
                    handler(std::current_exception());
                    promise->set_value();
                } else {
                    auto result = handler(std::current_exception());
                    promise->set_value(std::move(result));
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    }).detach();
    
    return AsyncTask<T>(std::move(newFuture), newMetadata);
}

template<typename T>
AsyncTask<T> AsyncTask<T>::with_timeout(std::chrono::milliseconds timeout) {
    auto promise = std::make_shared<boost::promise<T>>();
    auto newFuture = promise->get_future();
    
    TaskMetadata newMetadata = *metadata_;
    newMetadata.taskName += "_timeout";
    
    std::thread([future = std::move(future_), promise, timeout]() mutable {
        auto status = future.wait_for(timeout);
        if (status == boost::future_status::timeout) {
            promise->set_exception(std::make_exception_ptr(
                std::runtime_error("Task timed out")
            ));
        } else {
            try {
                if constexpr (std::is_void_v<T>) {
                    future.get();
                    promise->set_value();
                } else {
                    auto result = future.get();
                    promise->set_value(std::move(result));
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    }).detach();
    
    return AsyncTask<T>(std::move(newFuture), newMetadata);
}

template<typename T>
void AsyncTask<T>::cancel() {
    // 设置元数据状态为已取消
    metadata_->status = TaskStatus::CANCELLED;
    
    // 注意：boost::future 没有直接的取消机制
    // 当前简化实现：我们通过检查状态来模拟取消效果
    // 在实际应用中，应该使用cancellation_token或interruption机制
}

} // namespace oscean::common_utils::async 