/**
 * @file async_config.cpp
 * @brief 异步框架配置实现
 * 
 * 实现异步框架的配置管理功能，包括预定义配置的生成和环境特定配置的适配
 */

#include "common_utils/async/async_config.h"
#include <algorithm>

namespace oscean::common_utils::async {

AsyncConfig getHighPerformanceAsyncConfig() noexcept {
    AsyncConfig config;
    
    // 高性能配置：更多线程，更大队列，更短超时
    config.backend = AsyncBackend::BOOST_FUTURE;
    config.threadPoolSize = std::max(8u, std::thread::hardware_concurrency() * 2);
    config.maxQueueSize = 5000;
    
    // 更激进的超时设置
    config.defaultTimeout = std::chrono::milliseconds{15000};  // 15秒
    config.shortTimeout = std::chrono::milliseconds{2000};     // 2秒
    config.longTimeout = std::chrono::milliseconds{120000};    // 2分钟
    
    // 启用所有性能特性
    config.enableContinuation = true;
    config.enableWhenAllAny = true;
    config.enableProgressTracking = false;  // 关闭进度跟踪以提高性能
    
    // 大数据处理优化
    config.chunkSize = 128 * 1024;  // 128KB
    config.maxMemoryUsage = static_cast<size_t>(1024) * 1024 * 1024;  // 1GB
    config.enableMemoryPressureMonitoring = true;
    
    // 生产环境调试配置
    config.enableDetailedLogging = false;
    config.enablePerformanceMetrics = true;
    
    return config;
}

AsyncConfig getLargeDataAsyncConfig() noexcept {
    AsyncConfig config;
    
    // 大数据处理配置：优化内存使用和吞吐量
    config.backend = AsyncBackend::BOOST_FUTURE;
    config.threadPoolSize = std::thread::hardware_concurrency();
    config.maxQueueSize = 2000;  // 适中的队列大小
    
    // 更宽松的超时设置
    config.defaultTimeout = std::chrono::milliseconds{60000};   // 1分钟
    config.shortTimeout = std::chrono::milliseconds{10000};     // 10秒
    config.longTimeout = std::chrono::milliseconds{600000};     // 10分钟
    
    // 启用延续和批处理
    config.enableContinuation = true;
    config.enableWhenAllAny = true;
    config.enableProgressTracking = true;  // 启用进度跟踪以监控大数据处理
    
    // 大数据特定配置
    config.chunkSize = 256 * 1024;  // 256KB 更大的块
    config.maxMemoryUsage = static_cast<size_t>(2048) * 1024 * 1024;  // 2GB
    config.enableMemoryPressureMonitoring = true;
    
    // 详细日志用于调试大数据处理
    config.enableDetailedLogging = true;
    config.enablePerformanceMetrics = true;
    
    return config;
}

AsyncConfig getEnvironmentSpecificConfig(Environment env) noexcept {
    switch (env) {
        case Environment::DEVELOPMENT:
        {
            AsyncConfig config = getDefaultAsyncConfig();
            // 开发环境：启用详细日志，较小的队列
            config.threadPoolSize = std::min(4u, std::thread::hardware_concurrency());
            config.maxQueueSize = 500;
            config.enableDetailedLogging = true;
            config.enablePerformanceMetrics = true;
            config.enableProgressTracking = true;
            
            // 较短的超时便于快速调试
            config.defaultTimeout = std::chrono::milliseconds{10000};  // 10秒
            config.shortTimeout = std::chrono::milliseconds{3000};     // 3秒
            config.longTimeout = std::chrono::milliseconds{60000};     // 1分钟
            
            return config;
        }
        
        case Environment::TESTING:
        {
            AsyncConfig config = getDefaultAsyncConfig();
            // 测试环境：确定性配置，便于测试
            config.threadPoolSize = 2;  // 固定线程数
            config.maxQueueSize = 100;   // 小队列便于测试边界条件
            config.enableDetailedLogging = false;
            config.enablePerformanceMetrics = false;
            config.enableProgressTracking = false;
            
            // 更短的超时便于快速测试
            config.defaultTimeout = std::chrono::milliseconds{5000};   // 5秒
            config.shortTimeout = std::chrono::milliseconds{1000};     // 1秒
            config.longTimeout = std::chrono::milliseconds{15000};     // 15秒
            
            // 测试环境的内存限制
            config.chunkSize = 32 * 1024;  // 32KB
            config.maxMemoryUsage = static_cast<size_t>(128) * 1024 * 1024;  // 128MB
            
            return config;
        }
        
        case Environment::PRODUCTION:
        {
            // 生产环境使用高性能配置
            AsyncConfig config = getHighPerformanceAsyncConfig();
            // 生产环境特殊调整
            config.enableDetailedLogging = false;  // 关闭详细日志
            config.enablePerformanceMetrics = true; // 保留性能指标
            
            return config;
        }
        
        case Environment::HPC:
        {
            AsyncConfig config;
            // HPC环境：最大化性能和并行度
            config.backend = AsyncBackend::BOOST_FUTURE;
            config.threadPoolSize = std::thread::hardware_concurrency() * 4;  // 超线程
            config.maxQueueSize = 10000;  // 大队列
            
            // HPC环境的长时间任务
            config.defaultTimeout = std::chrono::milliseconds{300000};  // 5分钟
            config.shortTimeout = std::chrono::milliseconds{30000};     // 30秒
            config.longTimeout = std::chrono::milliseconds{3600000};    // 1小时
            
            // 启用所有功能
            config.enableContinuation = true;
            config.enableWhenAllAny = true;
            config.enableProgressTracking = true;
            
            // HPC大数据配置
            config.chunkSize = 512 * 1024;  // 512KB
            config.maxMemoryUsage = static_cast<size_t>(8192) * 1024 * 1024;  // 8GB
            config.enableMemoryPressureMonitoring = true;
            
            // HPC环境的调试配置
            config.enableDetailedLogging = false;  // 性能优先
            config.enablePerformanceMetrics = true;
            
            return config;
        }
        
        default:
            return getDefaultAsyncConfig();
    }
}

} // namespace oscean::common_utils::async 