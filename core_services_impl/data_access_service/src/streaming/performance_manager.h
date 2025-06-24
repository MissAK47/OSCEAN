#pragma once

/**
 * @file performance_manager.h
 * @brief Phase 6: 数据访问模块专用性能优化管理器 
 * 
 * 基于common_utilities的性能监控和内存管理功能，
 * 为数据访问模块提供专业化的性能优化扩展：
 * - NetCDF流式读取优化
 * - 数据块处理性能调优
 * - 特定于数据访问的内存模式优化
 */

#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <queue>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <boost/thread/future.hpp>
#include "common_utils/utilities/boost_config.h"
#include "core_services/data_access/unified_data_types.h"
#include "core_services/common_data_types.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/infrastructure/performance_monitor.h"
#include "common_utils/memory/memory_manager_unified.h"

#ifdef _WIN32
#include <windows.h>
#include <pdh.h>
#pragma comment(lib, "pdh.lib")
#endif

namespace oscean::core_services::data_access::streaming {

/**
 * @brief I/O操作类型枚举
 */
enum class IOOperationType {
    READ,
    WRITE,
    SEEK,
    OPEN,
    CLOSE
};

/**
 * @brief 性能优化建议结构
 */
struct PerformanceOptimizationSuggestion {
    std::string category;
    std::string description;
    double priority = 0.0;
    std::string action;
};

/**
 * @brief 数据访问性能管理器
 */
class DataAccessPerformanceManager {
public:
    /**
     * @brief 内存信息结构
     */
    struct MemoryInfo {
        size_t totalBytes = 0;
        size_t usedBytes = 0;
        size_t availableBytes = 0;
    };
    
    /**
     * @brief 磁盘I/O统计结构
     */
    struct DiskIOStats {
        double readMBps = 0.0;
        double writeMBps = 0.0;
        double iops = 0.0;
    };
    
    /**
     * @brief 网络统计结构
     */
    struct NetworkStats {
        double inMBps = 0.0;
        double outMBps = 0.0;
    };
    
    /**
     * @brief 构造函数
     */
    DataAccessPerformanceManager();
    
    /**
     * @brief 析构函数
     */
    ~DataAccessPerformanceManager();
    
    /**
     * @brief 初始化性能管理器
     */
    void initialize();
    
    /**
     * @brief 关闭性能管理器
     */
    void shutdown();
    
    /**
     * @brief 设置性能目标
     */
    void setTargets(const oscean::core_services::data_access::api::DataAccessPerformanceTargets& targets);
    
    /**
     * @brief 获取当前性能指标
     */
    oscean::core_services::data_access::api::DataAccessMetrics getCurrentMetrics() const;
    
    /**
     * @brief 记录I/O操作（带详细参数）
     * @param operationType 操作类型
     * @param success 是否成功
     * @param bytesTransferred 传输字节数
     * @param duration 持续时间
     */
    void recordIOOperation(
        IOOperationType operationType,
        bool success,
        size_t bytesTransferred,
        std::chrono::milliseconds duration);
    
    /**
     * @brief 记录缓存访问
     */
    void recordCacheAccess(bool hit, size_t dataSize = 0);
    
    /**
     * @brief 获取性能优化建议
     */
    std::vector<PerformanceOptimizationSuggestion> getOptimizationSuggestions() const;
    
    /**
     * @brief 启用/禁用自动优化
     */
    void setAutoOptimizationEnabled(bool enabled);
    
    /**
     * @brief 获取内存池信息
     * @return 内存池信息
     */
    MemoryInfo getMemoryPool() const;

private:
    // 私有实现
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * @brief 数据访问性能管理器的便捷别名 (保持向后兼容)
 */
using PerformanceManager = DataAccessPerformanceManager;

} // namespace oscean::core_services::data_access::streaming 