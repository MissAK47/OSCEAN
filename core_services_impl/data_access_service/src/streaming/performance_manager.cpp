/**
 * @file performance_manager.cpp
 * @brief 数据访问性能管理器实现
 */

#include "performance_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include <algorithm>
#include <numeric>

namespace oscean::core_services::data_access::streaming {

/**
 * @brief 性能管理器内部实现
 */
struct DataAccessPerformanceManager::Impl {
    // 状态管理
    std::atomic<bool> isInitialized{false};
    std::atomic<bool> autoOptimizationEnabled{true};
    
    // 同步原语
    mutable std::mutex metricsMutex;
    mutable std::mutex targetsMutex;
    
    // 数据存储
    oscean::core_services::data_access::api::DataAccessPerformanceTargets targets;
    oscean::core_services::data_access::api::DataAccessMetrics currentMetrics;
    
    // 历史数据
    std::vector<std::chrono::milliseconds> operationHistory;
    static constexpr size_t MAX_HISTORY_SIZE = 1000;
    
    // 计时器
    std::chrono::steady_clock::time_point lastUpdateTime;
    
    /**
     * @brief 更新派生指标
     */
    void updateDerivedMetrics() {
        auto now = std::chrono::steady_clock::now();
        
        // 计算平均响应时间
        if (currentMetrics.totalRequests > 0) {
            currentMetrics.averageResponseTimeMs = 
                static_cast<double>(std::accumulate(operationHistory.begin(), operationHistory.end(), 
                    std::chrono::milliseconds(0)).count()) / operationHistory.size();
        }
        
        // 计算当前吞吐量
        if (currentMetrics.totalBytesRead > 0) {
            auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                now - lastUpdateTime).count();
            if (elapsedSeconds > 0) {
                currentMetrics.currentThroughputMBps = 
                    (currentMetrics.totalBytesRead / (1024.0 * 1024.0)) / elapsedSeconds;
            }
        }
        
        lastUpdateTime = now;
    }
    
    /**
     * @brief 清理过期历史数据
     */
    void cleanupHistory() {
        if (operationHistory.size() > MAX_HISTORY_SIZE) {
            operationHistory.erase(
                operationHistory.begin(), 
                operationHistory.begin() + (operationHistory.size() - MAX_HISTORY_SIZE));
        }
    }
};

DataAccessPerformanceManager::DataAccessPerformanceManager()
    : pImpl_(std::make_unique<Impl>()) {
    LOG_INFO("创建DataAccessPerformanceManager");
}

DataAccessPerformanceManager::~DataAccessPerformanceManager() {
    LOG_INFO("销毁DataAccessPerformanceManager");
    shutdown();
}

void DataAccessPerformanceManager::initialize() {
    if (pImpl_->isInitialized.load()) {
        return;
    }
    
    try {
        std::lock_guard<std::mutex> lock(pImpl_->metricsMutex);
        
        // 初始化指标
        pImpl_->currentMetrics = oscean::core_services::data_access::api::DataAccessMetrics{};
        pImpl_->lastUpdateTime = std::chrono::steady_clock::now();
        
        pImpl_->isInitialized.store(true);
        LOG_INFO("DataAccessPerformanceManager初始化完成");
        
    } catch (const std::exception& e) {
        LOG_ERROR("DataAccessPerformanceManager初始化失败: {}", e.what());
        throw;
    }
}

void DataAccessPerformanceManager::shutdown() {
    if (!pImpl_->isInitialized.load()) {
        return;
    }
    
    try {
        pImpl_->isInitialized.store(false);
        
        std::lock_guard<std::mutex> lock(pImpl_->metricsMutex);
        pImpl_->operationHistory.clear();
        
        LOG_INFO("DataAccessPerformanceManager关闭完成");
        
    } catch (const std::exception& e) {
        LOG_ERROR("DataAccessPerformanceManager关闭异常: {}", e.what());
    }
}

void DataAccessPerformanceManager::setTargets(const oscean::core_services::data_access::api::DataAccessPerformanceTargets& targets) {
    std::lock_guard<std::mutex> lock(pImpl_->targetsMutex);
    pImpl_->targets = targets;
    
    LOG_INFO("更新性能目标: 吞吐量={} MB/s, 最大延迟={} ms", 
             targets.targetThroughputMBps, targets.maxLatencyMs);
}

oscean::core_services::data_access::api::DataAccessMetrics DataAccessPerformanceManager::getCurrentMetrics() const {
    if (!pImpl_->isInitialized.load()) {
        return oscean::core_services::data_access::api::DataAccessMetrics{};
    }
    
    std::lock_guard<std::mutex> lock(pImpl_->metricsMutex);
    
    // 更新派生指标
    const_cast<Impl*>(pImpl_.get())->updateDerivedMetrics();
    
    return pImpl_->currentMetrics;
}

void DataAccessPerformanceManager::recordIOOperation(
    IOOperationType operationType,
    bool success,
    size_t bytesTransferred,
    std::chrono::milliseconds duration) {
    
    if (!pImpl_->isInitialized.load()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pImpl_->metricsMutex);
    
    try {
        // 更新基本指标
        pImpl_->currentMetrics.totalRequests++;
        if (success) {
            pImpl_->currentMetrics.successfulRequests++;
        } else {
            pImpl_->currentMetrics.failedRequests++;
        }
        
        // 更新字节数
        if (operationType == IOOperationType::READ) {
            pImpl_->currentMetrics.totalBytesRead += bytesTransferred;
        }
        
        // 记录历史
        pImpl_->operationHistory.push_back(duration);
        pImpl_->cleanupHistory();
        
        LOG_DEBUG("记录I/O操作: type={}, success={}, size={} bytes, duration={} ms", 
                 static_cast<int>(operationType), success, bytesTransferred, duration.count());
        
    } catch (const std::exception& e) {
        LOG_ERROR("记录I/O操作异常: {}", e.what());
    }
}

void DataAccessPerformanceManager::recordCacheAccess(bool hit, size_t dataSize) {
    if (!pImpl_->isInitialized.load()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pImpl_->metricsMutex);
    
    try {
        auto& cacheStats = pImpl_->currentMetrics.cacheStats;
        
        if (hit) {
            cacheStats.totalHits++;
        } else {
            cacheStats.totalMisses++;
        }
        
        // 计算命中率
        size_t totalRequests = cacheStats.totalHits + cacheStats.totalMisses;
        if (totalRequests > 0) {
            cacheStats.hitRatio = static_cast<double>(cacheStats.totalHits) / totalRequests;
        }
        
        LOG_DEBUG("记录缓存访问: hit={}, size={} bytes, 命中率={}%", 
                 hit, dataSize, cacheStats.hitRatio * 100.0);
        
    } catch (const std::exception& e) {
        LOG_ERROR("记录缓存访问异常: {}", e.what());
    }
}

std::vector<PerformanceOptimizationSuggestion> DataAccessPerformanceManager::getOptimizationSuggestions() const {
    std::vector<PerformanceOptimizationSuggestion> suggestions;
    
    if (!pImpl_->isInitialized.load()) {
        return suggestions;
    }
    
    std::lock_guard<std::mutex> lock(pImpl_->metricsMutex);
    std::lock_guard<std::mutex> targetsLock(pImpl_->targetsMutex);
    
    const auto& metrics = pImpl_->currentMetrics;
    const auto& targets = pImpl_->targets;
    
    try {
        // 检查吞吐量
        if (metrics.currentThroughputMBps < targets.targetThroughputMBps * 0.8) {
            PerformanceOptimizationSuggestion suggestion;
            suggestion.category = "吞吐量优化";
            suggestion.description = "当前吞吐量低于目标，建议增加并发度或优化I/O策略";
            suggestion.priority = 0.8;
            suggestion.action = "增加并发度";
            suggestions.push_back(suggestion);
        }
        
        // 检查缓存命中率
        if (metrics.cacheStats.hitRatio < targets.targetCacheHitRatio) {
            PerformanceOptimizationSuggestion suggestion;
            suggestion.category = "缓存优化";
            suggestion.description = "缓存命中率低于目标，建议调整缓存策略";
            suggestion.priority = 0.7;
            suggestion.action = "优化缓存策略";
            suggestions.push_back(suggestion);
        }
        
        // 检查内存使用
        if (metrics.currentMemoryUsageMB > targets.maxMemoryUsageMB * 0.9) {
            PerformanceOptimizationSuggestion suggestion;
            suggestion.category = "内存优化";
            suggestion.description = "内存使用接近上限，建议释放不必要的缓存";
            suggestion.priority = 0.9;
            suggestion.action = "清理缓存";
            suggestions.push_back(suggestion);
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("生成优化建议异常: {}", e.what());
    }
    
    return suggestions;
}

void DataAccessPerformanceManager::setAutoOptimizationEnabled(bool enabled) {
    pImpl_->autoOptimizationEnabled.store(enabled);
    LOG_INFO("自动优化{}", enabled ? "启用" : "禁用");
}

DataAccessPerformanceManager::MemoryInfo DataAccessPerformanceManager::getMemoryPool() const {
    MemoryInfo info;
    
    if (!pImpl_->isInitialized.load()) {
        return info;
    }
    
    try {
        std::lock_guard<std::mutex> lock(pImpl_->metricsMutex);
        
        // 简化的内存信息
        info.usedBytes = pImpl_->currentMetrics.currentMemoryUsageMB * 1024 * 1024;
        info.totalBytes = pImpl_->targets.maxMemoryUsageMB * 1024 * 1024;
        info.availableBytes = info.totalBytes - info.usedBytes;
        
    } catch (const std::exception& e) {
        LOG_ERROR("获取内存池信息异常: {}", e.what());
    }
    
    return info;
}

} // namespace oscean::core_services::data_access::streaming

