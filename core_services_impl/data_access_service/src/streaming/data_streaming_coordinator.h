#pragma once

/**
 * @file data_streaming_coordinator.h
 * @brief 数据访问专用流式处理协调器
 */

#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <optional>
#include <boost/thread/future.hpp>

#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/async/async_framework.h"
#include "core_services/data_access/unified_data_types.h"

// Forward declarations
struct GDALDataset;
typedef int ncid_t;

namespace oscean::core_services::data_access::readers::impl {

/**
 * @brief 数据块结构
 */
struct DataChunk {
    std::vector<double> data;           ///< 数据内容
    std::vector<size_t> shape;          ///< 数据形状
    std::vector<size_t> offset;         ///< 在原始数据中的偏移
    size_t chunkId;                     ///< 块ID
    bool isLastChunk = false;           ///< 是否为最后一块
};

/**
 * @brief 流式处理配置
 */
struct StreamingConfig {
    size_t chunkSize = 1024 * 1024;        ///< 块大小（字节）
    size_t maxConcurrentChunks = 4;        ///< 最大并发块数
    bool enableBackpressure = true;        ///< 启用背压控制
    double memoryThreshold = 0.8;          ///< 内存阈值
    size_t bufferSize = 8 * 1024 * 1024;   ///< 缓冲区大小
};

/**
 * @brief 格式流式适配器接口
 */
class FormatStreamingAdapter {
public:
    virtual ~FormatStreamingAdapter() = default;
    
    /**
     * @brief 配置流式读取参数
     */
    virtual void configureStreaming(const StreamingConfig& config) = 0;
    
    /**
     * @brief 流式读取数据块
     */
    virtual boost::future<std::optional<DataChunk>> readNextChunk() = 0;
    
    /**
     * @brief 重置流式读取状态
     */
    virtual void resetStreaming() = 0;
};

/**
 * @brief 数据流式处理协调器
 * 
 * 专门针对数据文件读取的流式处理逻辑，基于common的内存管理和异步框架
 */
class DataStreamingCoordinator {
public:
    DataStreamingCoordinator(
        std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager,
        std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework
    );
    
    ~DataStreamingCoordinator() = default;
    
    // =============================================================================
    // 配置接口
    // =============================================================================
    
    /**
     * @brief 配置流式处理参数
     */
    void configureStreaming(const StreamingConfig& config);
    
    /**
     * @brief 配置GDAL数据集的流式适配
     */
    void configureForGdal(GDALDataset* dataset);
    
    /**
     * @brief 配置NetCDF文件的流式适配
     */
    void configureForNetCDF(ncid_t ncid);
    
    // =============================================================================
    // 流式处理接口
    // =============================================================================
    
    /**
     * @brief 流式读取变量数据
     * @param variableName 变量名
     * @param bounds 边界框（可选）
     * @param processor 数据处理函数，返回false表示停止处理
     * @param adapter 格式特定的流式适配器
     */
    boost::future<void> streamVariable(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        std::function<bool(const DataChunk&)> processor,
        std::shared_ptr<FormatStreamingAdapter> adapter
    );
    
    /**
     * @brief 并行流式读取多个变量
     */
    boost::future<void> streamMultipleVariables(
        const std::vector<std::string>& variableNames,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        std::function<bool(const std::string&, const DataChunk&)> processor,
        std::shared_ptr<FormatStreamingAdapter> adapter
    );
    
    // =============================================================================
    // 背压控制
    // =============================================================================
    
    /**
     * @brief 检查是否应该应用背压
     */
    bool shouldApplyBackpressure() const;
    
    /**
     * @brief 等待背压缓解
     */
    void waitForBackpressureRelief();
    
    /**
     * @brief 获取当前内存使用率
     */
    double getCurrentMemoryUsage() const;
    
    // =============================================================================
    // 状态监控
    // =============================================================================
    
    /**
     * @brief 获取活跃块数量
     */
    size_t getActiveChunksCount() const;
    
    /**
     * @brief 获取处理统计信息
     */
    struct StreamingStats {
        size_t totalChunksProcessed = 0;
        size_t totalBytesProcessed = 0;
        std::chrono::milliseconds totalProcessingTime{0};
        double averageChunkProcessingTime = 0.0;
        size_t currentActiveChunks = 0;
    };
    
    StreamingStats getStreamingStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();
    
private:
    // Common组件
    std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager_;
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework_;
    
    // 配置
    StreamingConfig config_;
    
    // 状态管理
    std::atomic<size_t> activeChunks_{0};
    std::atomic<size_t> chunkIdCounter_{0};
    
    // 统计信息
    mutable std::mutex statsMutex_;
    StreamingStats stats_;
    std::chrono::steady_clock::time_point startTime_;
    
    // 内部方法
    void updateStats(size_t bytesProcessed, std::chrono::milliseconds processingTime);
    bool checkMemoryThreshold() const;
    void waitForMemoryAvailable();
};

} // namespace oscean::core_services::data_access::readers::impl 