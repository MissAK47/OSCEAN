/**
 * @file async_config.h
 * @brief 异步框架配置和后端选择
 * 
 * 定义异步处理框架的配置选项，包括：
 * - boost::future vs std::future 后端选择
 * - 异步执行策略配置
 * - 性能优化选项
 * - 大数据处理专用配置
 */

#pragma once

// 引用统一的boost配置 - 必须在最前面
#include "common_utils/utilities/boost_config.h"

#include <chrono>
#include <thread>
#include <cstddef>

namespace oscean::common_utils::async {

/**
 * @brief 异步后端类型
 */
enum class AsyncBackend {
    BOOST_FUTURE,    // 使用boost::future（默认，功能更丰富）
    STD_FUTURE,      // 使用std::future（标准库）
    HYBRID           // 混合使用（根据场景自动选择）
};

/**
 * @brief 环境类型
 */
enum class Environment {
    DEVELOPMENT,
    TESTING,
    PRODUCTION,
    HPC
};

/**
 * @brief 异步配置结构
 */
struct AsyncConfig {
    // 后端选择
    AsyncBackend backend = AsyncBackend::BOOST_FUTURE;
    
    // 线程池配置
    size_t threadPoolSize = std::thread::hardware_concurrency();
    size_t maxQueueSize = 1000;
    
    // 超时配置
    std::chrono::milliseconds defaultTimeout{30000};  // 30秒
    std::chrono::milliseconds shortTimeout{5000};     // 5秒（快速操作）
    std::chrono::milliseconds longTimeout{300000};    // 5分钟（长时间操作）
    
    // 性能配置
    bool enableContinuation = true;      // 启用future延续
    bool enableWhenAllAny = true;        // 启用when_all/when_any
    bool enableProgressTracking = true;  // 启用进度跟踪
    
    // 大数据处理配置
    size_t chunkSize = 64 * 1024;        // 64KB块大小
    size_t maxMemoryUsage = 512 * 1024 * 1024;  // 512MB最大内存
    bool enableMemoryPressureMonitoring = true;
    
    // 调试配置
    bool enableDetailedLogging = false;
    bool enablePerformanceMetrics = true;
};

/**
 * @brief 获取预定义配置
 */
inline AsyncConfig getDefaultAsyncConfig() noexcept {
    return AsyncConfig{};
}

AsyncConfig getHighPerformanceAsyncConfig() noexcept;
AsyncConfig getLargeDataAsyncConfig() noexcept;
AsyncConfig getEnvironmentSpecificConfig(Environment env) noexcept;

} // namespace oscean::common_utils::async 