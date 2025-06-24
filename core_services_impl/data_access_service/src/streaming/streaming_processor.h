#pragma once

/**
 * @file streaming_processor.h
 * @brief 流式数据处理器 - 重构版
 * 
 * 🎯 功能：
 * - 大文件分块流式读取
 * - 自适应分块策略
 * - 背压控制
 * - 流式数据转换
 * - 内存优化管理
 */

#include "core_services/data_access/unified_data_types.h"
#include "core_services/common_data_types.h"
#include "common_utils/async/async_framework.h"
#include "common_utils/utilities/logging_utils.h"

#include <boost/thread/future.hpp>
#include <boost/asio/thread_pool.hpp>
#include <memory>
#include <string>
#include <functional>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <shared_mutex>
#include <unordered_map>

// 前向声明
namespace oscean::core_services::data_access::readers {
    class UnifiedDataReader;
}

namespace oscean::core_services::data_access::streaming {

/**
 * @brief 使用统一的DataChunkKey和IndexRange，避免重复定义
 */
using DataChunk = oscean::core_services::DataChunkKey;
using StreamDataRange = oscean::core_services::IndexRange;

/**
 * @brief 扩展的数据块，包含实际数据内容
 */
struct ProcessingDataChunk {
    oscean::core_services::DataChunkKey chunkKey;  ///< 数据块键值
    std::vector<double> data;                      ///< 实际数据内容
    size_t chunkIndex = 0;                         ///< 块索引
    size_t totalChunks = 0;                        ///< 总块数
    bool isLastChunk = false;                      ///< 是否为最后一块
    
    ProcessingDataChunk() 
        : chunkKey("", "", boost::none, boost::none, boost::none, "double") {}
    ProcessingDataChunk(oscean::core_services::DataChunkKey key, std::vector<double> d, 
                       size_t idx, size_t total)
        : chunkKey(std::move(key)), data(std::move(d)), chunkIndex(idx), totalChunks(total) {}
};

/**
 * @brief 流式处理选项
 */
struct StreamingOptions {
    size_t chunkSize = 1024 * 1024;  // 1MB
    size_t maxConcurrentChunks = 4;
    double backpressureThreshold = 0.8;
    bool enableAdaptiveChunking = true;
    bool enableBackpressure = true;
};

/**
 * @brief 分块类型枚举
 */
enum class ChunkingType {
    SEQUENTIAL,      ///< 顺序分块
    SPATIAL_BLOCKS   ///< 空间分块
};

/**
 * @brief 分块策略结构
 */
struct ChunkingStrategy {
    size_t totalChunks = 0;         ///< 总分块数
    size_t chunkSize = 0;           ///< 分块大小
    size_t lastChunkSize = 0;       ///< 最后一块的大小
    ChunkingType chunkingType = ChunkingType::SEQUENTIAL; ///< 分块类型
};

/**
 * @brief 流式处理统计信息
 */
struct StreamingStatistics {
    size_t processedChunks = 0;     ///< 已处理分块数
    size_t failedChunks = 0;        ///< 失败分块数
    size_t currentConcurrency = 0;  ///< 当前并发数
    size_t maxConcurrency = 0;      ///< 最大并发数
    double averageChunkSize = 0.0;  ///< 平均分块大小
    double successRate = 0.0;       ///< 成功率
};

/**
 * @brief 流状态枚举
 */
enum class StreamState {
    IDLE,           ///< 空闲状态
    ACTIVE,         ///< 活跃状态  
    PAUSED,         ///< 暂停状态
    COMPLETED,      ///< 完成状态
    ERROR_STATE     ///< 错误状态 (避免与ERROR宏冲突)
};

/**
 * @brief 流统计信息
 */
struct StreamStatistics {
    size_t totalChunksProcessed = 0;        ///< 处理的总块数
    size_t totalBytesProcessed = 0;         ///< 处理的总字节数
    std::chrono::milliseconds totalTime{0}; ///< 总处理时间
    double averageChunkProcessingTime = 0.0; ///< 平均块处理时间
    size_t currentQueueSize = 0;            ///< 当前队列大小
    double throughputMBps = 0.0;            ///< 吞吐量 (MB/s)
    
    /**
     * @brief 计算当前吞吐量
     */
    void updateThroughput() {
        if (totalTime.count() > 0) {
            double timeSeconds = totalTime.count() / 1000.0;
            double megabytes = totalBytesProcessed / (1024.0 * 1024.0);
            throughputMBps = megabytes / timeSeconds;
        }
    }
};

/**
 * @brief 自适应分块策略
 */
class AdaptiveChunkingStrategy {
public:
    /**
     * @brief 构造函数
     */
    AdaptiveChunkingStrategy(
        size_t minChunkSize = 64 * 1024,      // 64KB
        size_t maxChunkSize = 10 * 1024 * 1024, // 10MB
        size_t targetChunkSize = 1024 * 1024    // 1MB
    );
    
    /**
     * @brief 根据性能反馈调整块大小
     * @param processingTime 上次处理时间
     * @param memoryPressure 内存压力指标 (0.0-1.0)
     * @return 建议的下一个块大小
     */
    size_t adjustChunkSize(
        std::chrono::milliseconds processingTime,
        double memoryPressure = 0.0);
    
    /**
     * @brief 获取当前建议的块大小
     */
    size_t getCurrentChunkSize() const { return currentChunkSize_; }
    
    /**
     * @brief 重置策略
     */
    void reset();

private:
    size_t minChunkSize_;
    size_t maxChunkSize_;
    size_t targetChunkSize_;
    size_t currentChunkSize_;
    
    // 性能历史记录
    std::vector<std::chrono::milliseconds> processingHistory_;
    static constexpr size_t MAX_HISTORY_SIZE = 10;
    
    /**
     * @brief 计算平均处理时间
     */
    double getAverageProcessingTime() const;
};

/**
 * @brief 背压控制器
 */
class BackpressureController {
public:
    /**
     * @brief 构造函数
     */
    BackpressureController(
        size_t maxQueueSize = 100,
        double backpressureThreshold = 0.8,
        bool dropOnOverflow = false
    );
    
    /**
     * @brief 检查是否应该应用背压
     * @param currentQueueSize 当前队列大小
     * @return 是否需要背压控制
     */
    bool shouldApplyBackpressure(size_t currentQueueSize) const;
    
    /**
     * @brief 等待直到可以继续处理
     * @param currentQueueSize 当前队列大小
     * @param timeout 超时时间
     * @return 是否可以继续
     */
    boost::future<bool> waitForCapacity(
        size_t currentQueueSize, 
        std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));
    
    /**
     * @brief 通知队列容量变化
     */
    void notifyCapacityChange();

private:
    size_t maxQueueSize_;
    double backpressureThreshold_;
    bool dropOnOverflow_;
    
    mutable std::mutex mutex_;
    std::condition_variable capacityCondition_;
};

/**
 * @brief 流式数据处理器
 */
class StreamingProcessor {
public:
    /**
     * @brief 构造函数
     */
    StreamingProcessor();
    
    /**
     * @brief 析构函数
     */
    ~StreamingProcessor();
    
    /**
     * @brief 初始化处理器
     */
    void initialize();
    
    /**
     * @brief 关闭处理器
     */
    void shutdown();
    
    /**
     * @brief 异步处理流式数据
     */
    boost::future<void> processStreamAsync(
        std::shared_ptr<readers::UnifiedDataReader> reader,
        const std::string& variableName,
        std::function<boost::future<bool>(ProcessingDataChunk)> chunkProcessor,
        const StreamingOptions& options);
    
    /**
     * @brief 设置分块大小
     */
    void setChunkSize(size_t chunkSize);
    
    /**
     * @brief 设置最大并发分块数
     */
    void setMaxConcurrentChunks(size_t maxChunks);
    
    /**
     * @brief 设置背压阈值
     */
    void setBackpressureThreshold(double threshold);
    
    /**
     * @brief 获取统计信息
     */
    StreamingStatistics getStatistics() const;
    
    /**
     * @brief 开始流处理
     * @param streamId 流ID
     * @return 是否成功开始
     */
    bool startStreaming(const std::string& streamId);
    
    /**
     * @brief 停止流处理
     * @param streamId 流ID
     * @return 是否成功停止
     */
    bool stopStreaming(const std::string& streamId);
    
    /**
     * @brief 提交数据块到流
     * @param streamId 流ID
     * @param chunk 数据块
     * @return 处理结果
     */
    boost::future<bool> submitChunk(const std::string& streamId, ProcessingDataChunk chunk);
    
    /**
     * @brief 获取流统计信息
     * @param streamId 流ID
     * @return 流统计信息
     */
    StreamStatistics getStreamStatistics(const std::string& streamId) const;

private:
    // 状态变量
    std::atomic<bool> isInitialized_;
    std::atomic<size_t> processedChunks_;
    std::atomic<size_t> failedChunks_;
    std::atomic<size_t> currentConcurrency_;
    
    // 配置参数
    size_t defaultChunkSize_;
    size_t maxConcurrentChunks_;
    double backpressureThreshold_;
    
    // 线程池
    std::unique_ptr<boost::asio::thread_pool> threadPool_;
    
    // 流管理
    mutable std::shared_mutex streamsMutex_;
    std::unordered_map<std::string, StreamState> activeStreams_;
    std::unordered_map<std::string, StreamStatistics> streamStats_;
    
    // 私有方法
    void processStreamImpl(
        std::shared_ptr<readers::UnifiedDataReader> reader,
        const std::string& variableName,
        std::function<boost::future<bool>(ProcessingDataChunk)> chunkProcessor,
        const StreamingOptions& options);
    
    ChunkingStrategy calculateChunkingStrategy(
        const std::string& variableName,
        std::shared_ptr<readers::UnifiedDataReader> reader,
        const StreamingOptions& options);
    
    boost::future<bool> createChunkTask(
        std::shared_ptr<readers::UnifiedDataReader> reader,
        const std::string& variableName,
        size_t chunkIndex,
        const ChunkingStrategy& strategy,
        std::function<boost::future<bool>(ProcessingDataChunk)> chunkProcessor);
    
    bool shouldApplyBackpressure() const;
    void waitForBackpressureRelief();
    
    size_t estimateVariableSize(const std::string& variableName, 
                               std::shared_ptr<readers::UnifiedDataReader> reader) const;
    size_t getDataTypeSize(oscean::core_services::DataType dataType) const;
    
    StreamDataRange calculateChunkRange(size_t chunkIndex, const ChunkingStrategy& strategy) const;
    
    // 分块读取方法
    std::unique_ptr<std::vector<double>> readChunkDataComplete(
        std::shared_ptr<readers::UnifiedDataReader> reader,
        const std::string& variableName,
        const StreamDataRange& range,
        const ChunkingStrategy& strategy) const;
    
    std::unique_ptr<std::vector<double>> readSpatialChunk(
        std::shared_ptr<readers::UnifiedDataReader> reader,
        const std::string& variableName,
        const StreamDataRange& range) const;
    
    std::unique_ptr<std::vector<double>> readSequentialChunk(
        std::shared_ptr<readers::UnifiedDataReader> reader,
        const std::string& variableName,
        const StreamDataRange& range) const;
    
    // 统计方法
    double calculateAverageChunkSize() const;
    double calculateSuccessRate() const;
};

} // namespace oscean::core_services::data_access::streaming 
