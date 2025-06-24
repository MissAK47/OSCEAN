/**
 * @file async_types.h
 * @brief 异步操作通用类型定义
 * 
 * 🎯 重构说明：
 * ✅ 参考CRS模块成功模式
 * ✅ 直接使用boost::future，避免模板别名问题
 * ✅ 提供全局异步类型定义
 */

#pragma once

// 🚀 使用Common模块的统一boost配置 - 参考CRS模块成功模式
#include "../utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // async_types模块不使用boost::asio，只使用boost::future

// 立即包含boost::future - 参考CRS模块
#include <boost/thread/future.hpp>

// 标准库头文件
#include <optional>
#include <string>
#include <functional>
#include <chrono>

namespace oscean::common_utils::async {

// === 统一异步类型定义 - 直接使用boost类型，避免别名问题 ===

// 注意：不使用模板别名，直接引用boost::future等类型
// 在实际使用时直接写 boost::future<T>，boost::promise<T> 等

// === 回调函数类型定义 ===

/**
 * @brief 任务回调函数类型
 */
using TaskCallback = std::function<void(const std::string& taskId, const std::string& message)>;

/**
 * @brief 错误回调函数类型
 */
using ErrorCallback = std::function<void(const std::string& taskId, const std::exception& error)>;

/**
 * @brief 进度回调函数类型
 */
using ProgressCallback = std::function<void(const std::string& taskId, double progress)>;

// === 重试策略 ===

/**
 * @brief 重试策略配置
 */
struct RetryPolicy {
    size_t maxRetries = 3;                                  // 最大重试次数
    std::chrono::milliseconds baseDelay{100};              // 基础延迟时间
    double backoffMultiplier = 2.0;                        // 退避乘数
    std::chrono::milliseconds maxDelay{10000};             // 最大延迟时间
    bool enableJitter = true;                               // 启用抖动
    
    /**
     * @brief 计算指定重试次数的延迟时间
     * @param retryCount 当前重试次数
     * @return 延迟时间
     */
    std::chrono::milliseconds calculateDelay(size_t retryCount) const;
    
    /**
     * @brief 转换为字符串表示
     * @return 重试策略的字符串表示
     */
    std::string toString() const;
};

// === 断路器状态 ===

/**
 * @brief 断路器状态枚举
 */
enum class CircuitBreakerState {
    CLOSED,        // 关闭状态（正常）
    OPEN,          // 开放状态（熔断）
    HALF_OPEN      // 半开状态（探测）
};

// === 异步统计信息 ===

/**
 * @brief 异步操作统计信息
 */
struct AsyncStatistics {
    size_t totalTasksSubmitted = 0;        // 总提交任务数
    size_t totalTasksCompleted = 0;        // 总完成任务数
    size_t totalTasksFailed = 0;           // 总失败任务数
    size_t currentActiveTasks = 0;         // 当前活跃任务数
    double averageExecutionTime = 0.0;     // 平均执行时间（毫秒）
    size_t queueSize = 0;                  // 当前队列大小
    double threadPoolUtilization = 0.0;    // 线程池利用率（0.0-1.0）
    
    /**
     * @brief 计算成功率
     * @return 任务成功率（0.0-1.0）
     */
    double getSuccessRate() const;
    
    /**
     * @brief 计算失败率
     * @return 任务失败率（0.0-1.0）
     */
    double getFailureRate() const;
    
    /**
     * @brief 转换为字符串表示
     * @return 统计信息的字符串表示
     */
    std::string toString() const;
};

// === 任务执行结果 ===

/**
 * @brief 任务执行结果枚举
 */
enum class TaskExecutionResult {
    SUCCESS,        // 成功
    FAILED,         // 失败
    TIMEOUT,        // 超时
    CANCELLED,      // 取消
    RETRY_NEEDED    // 需要重试
};

// === 资源限制 ===

/**
 * @brief 资源限制配置
 */
struct ResourceLimits {
    size_t maxConcurrentTasks = 0;         // 最大并发任务数（0表示无限制）
    size_t maxMemoryUsage = 0;             // 最大内存使用量（字节，0表示无限制）
    size_t maxQueueLength = 0;             // 最大队列长度（0表示无限制）
    std::chrono::milliseconds maxTaskDuration{0};  // 最大任务执行时间（0表示无限制）
    
    /**
     * @brief 检查资源限制是否有效
     * @return 如果至少有一个限制被设置，返回 true
     */
    bool hasLimits() const;
    
    /**
     * @brief 转换为字符串表示
     * @return 资源限制的字符串表示
     */
    std::string toString() const;
};

} // namespace oscean::common_utils::async 