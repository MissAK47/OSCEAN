/**
 * @file large_file_processor.h
 * @brief 大文件处理器 - 重构后的核心大文件处理组件
 * @author OSCEAN Team
 * @date 2024
 * 
 * 🎯 重构目标：
 * ✅ 整合streaming模块的核心大文件处理功能
 * ✅ 提供内存高效、高吞吐量的大数据处理能力
 * ✅ 支持异步操作（基于boost::future）
 * ✅ 与内存管理器、线程池管理器集成
 */

#pragma once

#include "common_utils/utilities/boost_config.h"
#include "common_utils/memory/memory_interfaces.h"
#include "unified_thread_pool_manager.h"
#include <boost/thread/future.hpp>
#include <boost/asio/thread_pool.hpp>

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <atomic>
#include <mutex>
#include <optional>

namespace oscean::common_utils::infrastructure {

/**
 * @brief 文件类型枚举
 */
enum class FileType {
    AUTO_DETECT = 0,
    NETCDF,
    HDF5,
    GEOTIFF,
    SHAPEFILE,
    CSV,
    JSON,
    BINARY,
    TEXT
};

/**
 * @brief 处理状态枚举
 */
enum class ProcessingStatus {
    SUCCESS = 0,
    IN_PROGRESS,
    PAUSED,
    CANCELLED,
    FAILED,
    PARTIAL_SUCCESS
};

/**
 * @brief 大文件处理策略
 */
enum class LargeFileStrategy {
    MEMORY_CONSERVATIVE,  // 内存保守：最小内存使用
    BALANCED,            // 平衡：性能与内存平衡
    PERFORMANCE_FIRST,   // 性能优先：最大吞吐量
    ADAPTIVE             // 自适应：根据系统状态调整
};

/**
 * @brief 数据块结构
 */
struct DataChunk {
    std::vector<uint8_t> data;
    size_t offset = 0;
    size_t size = 0;
    bool isLast = false;
    
    DataChunk() = default;
    DataChunk(size_t chunkSize) : data(chunkSize) {}
    DataChunk(const uint8_t* ptr, size_t len) : data(ptr, ptr + len), size(len) {}
};

/**
 * @brief 进度观察者接口
 */
class IProgressObserver {
public:
    virtual ~IProgressObserver() = default;
    virtual void onProgress(double percentage, const std::string& message) = 0;
    virtual void onError(const std::string& error) = 0;
    virtual void onComplete() = 0;
};

/**
 * @brief 大文件处理配置
 */
struct LargeFileConfig {
    LargeFileStrategy strategy = LargeFileStrategy::BALANCED;
    
    // === 内存配置 ===
    size_t maxMemoryUsageMB = 512;         // 最大内存使用
    size_t chunkSizeMB = 32;               // 数据块大小
    size_t bufferPoolSizeMB = 128;         // 缓冲池大小
    
    // === 并发配置 ===
    size_t ioThreads = 2;                  // IO线程数
    size_t processingThreads = 4;          // 处理线程数
    size_t maxConcurrentChunks = 6;        // 最大并发块数
    
    // === 性能优化 ===
    bool enableReadAhead = true;           // 启用预读
    bool enableWriteBehind = true;         // 启用后写
    bool enableCaching = false;            // 启用缓存（大文件通常禁用）
    
    // === 容错配置 ===
    size_t maxRetries = 3;                 // 最大重试次数
    bool enableCheckpointing = true;       // 启用检查点
    std::chrono::minutes checkpointInterval{10}; // 检查点间隔
    
    // === 监控配置 ===
    bool enableDetailedMonitoring = true;  // 启用详细监控
    std::chrono::seconds progressReportInterval{30}; // 进度报告间隔
    
    std::string toString() const;
    static LargeFileConfig createOptimal();
    static LargeFileConfig createForStrategy(LargeFileStrategy strategy);
};

/**
 * @brief 大文件信息结构
 */
struct LargeFileInfo {
    std::string filePath;
    FileType fileType = FileType::AUTO_DETECT;
    size_t fileSizeBytes = 0;
    size_t estimatedRecords = 0;
    
    // === 处理估算 ===
    std::chrono::milliseconds estimatedProcessingTime{0};
    size_t recommendedMemoryMB = 0;
    size_t recommendedChunkSizeMB = 0;
    size_t recommendedParallelism = 0;
    
    // === 文件特征 ===
    bool isCompressed = false;
    bool hasStructuredData = false;
    bool requiresSpecialHandling = false;
    
    std::string toString() const;
};

/**
 * @brief 大文件处理器接口
 */
class ILargeFileProcessor {
public:
    virtual ~ILargeFileProcessor() = default;
    
    /**
     * @brief 处理大文件
     */
    virtual ProcessingStatus processFile(
        const std::string& filePath,
        const LargeFileConfig& config = {},
        std::shared_ptr<IProgressObserver> observer = nullptr) = 0;
    
    /**
     * @brief 异步处理大文件
     */
    virtual boost::future<ProcessingStatus> processFileAsync(
        const std::string& filePath,
        const LargeFileConfig& config = {},
        std::shared_ptr<IProgressObserver> observer = nullptr) = 0;
    
    /**
     * @brief 分块处理文件
     */
    virtual void processInChunks(
        const std::string& filePath,
        std::function<bool(const DataChunk&)> processor,
        const LargeFileConfig& config = {}) = 0;
    
    /**
     * @brief 估算处理时间
     */
    virtual std::chrono::milliseconds estimateProcessingTime(
        const std::string& filePath,
        const LargeFileConfig& config = {}) const = 0;
    
    /**
     * @brief 获取优化配置
     */
    virtual LargeFileConfig getOptimizedConfig(const std::string& filePath) const = 0;
};

/**
 * @brief 大文件处理器实现类
 */
class LargeFileProcessor : public ILargeFileProcessor {
public:
    /**
     * @brief 构造函数
     * @param config 处理配置
     * @param memoryManager 内存管理器（可选，用于优化内存使用）
     * @param threadPoolManager 线程池管理器（可选，用于优化并发处理）
     */
    explicit LargeFileProcessor(
        const LargeFileConfig& config = {},
        std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager = nullptr,
        std::shared_ptr<UnifiedThreadPoolManager> threadPoolManager = nullptr
    );
    
    ~LargeFileProcessor() override;
    
    // === ILargeFileProcessor接口实现 ===
    
    ProcessingStatus processFile(
        const std::string& filePath,
        const LargeFileConfig& config = {},
        std::shared_ptr<IProgressObserver> observer = nullptr) override;
    
    boost::future<ProcessingStatus> processFileAsync(
        const std::string& filePath,
        const LargeFileConfig& config = {},
        std::shared_ptr<IProgressObserver> observer = nullptr) override;
    
    void processInChunks(
        const std::string& filePath,
        std::function<bool(const DataChunk&)> processor,
        const LargeFileConfig& config = {}) override;
    
    std::chrono::milliseconds estimateProcessingTime(
        const std::string& filePath,
        const LargeFileConfig& config = {}) const override;
    
    LargeFileConfig getOptimizedConfig(const std::string& filePath) const override;
    
    // === 扩展功能 ===
    
    /**
     * @brief 分析文件特征
     */
    LargeFileInfo analyzeFile(const std::string& filePath) const;
    
    /**
     * @brief 检查是否能处理该文件
     */
    bool canProcessFile(const std::string& filePath) const;
    
    /**
     * @brief 获取处理建议
     */
    std::vector<std::string> getProcessingRecommendations(
        const std::string& filePath) const;
    
    // === 配置管理 ===
    
    void updateConfig(const LargeFileConfig& config);
    const LargeFileConfig& getConfig() const { return config_; }
    
    // === 处理控制 ===
    
    void pauseProcessing();
    void resumeProcessing();
    void cancelProcessing();
    bool isProcessing() const { return processing_; }
    
    // === 检查点管理 ===
    
    void enableCheckpointing(bool enable = true) { enableCheckpointing_ = enable; }
    bool saveCheckpoint(const std::string& checkpointPath) const;
    bool loadCheckpoint(const std::string& checkpointPath);
    void clearCheckpoints();
    
    // === 性能监控 ===
    
    struct ProcessingStats {
        size_t totalBytesProcessed = 0;
        size_t totalChunksProcessed = 0;
        std::chrono::milliseconds totalProcessingTime{0};
        double averageThroughputMBps = 0.0;
        size_t peakMemoryUsageMB = 0;
    };
    
    ProcessingStats getProcessingStats() const;
    void resetStats();

private:
    // === 配置和依赖 ===
    LargeFileConfig config_;
    std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager_;
    std::shared_ptr<UnifiedThreadPoolManager> threadPoolManager_;
    
    // === 处理状态 ===
    std::atomic<bool> processing_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> cancelled_{false};
    std::atomic<bool> enableCheckpointing_{true};
    
    // === 统计信息 ===
    mutable std::mutex statsMutex_;
    ProcessingStats stats_;
    
    // === 检查点数据 ===
    struct CheckpointData {
        std::string filePath;
        size_t processedBytes = 0;
        size_t processedChunks = 0;
        std::chrono::steady_clock::time_point startTime;
        LargeFileConfig config;
    };
    
    mutable std::mutex checkpointMutex_;
    std::optional<CheckpointData> currentCheckpoint_;
    
    // === 内部处理方法 ===
    
    ProcessingStatus processInternal(
        const std::string& filePath,
        const LargeFileConfig& config,
        std::shared_ptr<IProgressObserver> observer);
    
    void setupProcessingEnvironment(const LargeFileInfo& fileInfo);
    void cleanupProcessingEnvironment();
    
    // === 文件分析方法 ===
    
    FileType detectFileType(const std::string& filePath) const;
    size_t estimateFileRecords(const std::string& filePath, FileType type) const;
    bool isFileCompressed(const std::string& filePath) const;
    
    // === 配置优化方法 ===
    
    LargeFileConfig optimizeConfigForFile(const LargeFileInfo& fileInfo) const;
    size_t calculateOptimalChunkSize(const LargeFileInfo& fileInfo) const;
    size_t calculateOptimalParallelism(const LargeFileInfo& fileInfo) const;
    
    // === 检查点实现 ===
    
    void updateCheckpoint(size_t processedBytes, size_t processedChunks);
    std::string generateCheckpointPath(const std::string& filePath) const;
    
    // === 异步执行辅助 ===
    
    template<typename Func>
    auto executeAsync(Func&& func) -> boost::future<decltype(func())>;
    
    // === 静态工厂方法 ===
    
    /**
     * @brief 创建标准处理器
     */
    static std::unique_ptr<ILargeFileProcessor> createProcessor(
        const LargeFileConfig& config = {}
    );
    
    /**
     * @brief 创建带依赖注入的处理器
     */
    static std::unique_ptr<ILargeFileProcessor> createProcessorWithDependencies(
        const LargeFileConfig& config,
        std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager,
        std::shared_ptr<UnifiedThreadPoolManager> threadPoolManager
    );
    
    /**
     * @brief 创建针对特定文件优化的处理器
     */
    static std::unique_ptr<ILargeFileProcessor> createOptimizedProcessor(
        const std::string& filePath
    );
};

} // namespace oscean::common_utils::infrastructure 