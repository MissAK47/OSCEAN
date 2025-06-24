/**
 * @file streaming_processor.cpp
 * @brief 流式数据处理器实现 - 简化版
 */

#include "streaming_processor.h"
#include "common_utils/utilities/logging_utils.h"
#include "../readers/core/unified_data_reader.h"

#include <boost/thread/future.hpp>
#include <boost/asio/post.hpp>
#include <algorithm>
#include <cmath>

namespace oscean::core_services::data_access::streaming {

// =============================================================================
// AdaptiveChunkingStrategy 实现
// =============================================================================

AdaptiveChunkingStrategy::AdaptiveChunkingStrategy(
    size_t minChunkSize,
    size_t maxChunkSize,
    size_t targetChunkSize)
    : minChunkSize_(minChunkSize)
    , maxChunkSize_(maxChunkSize)
    , targetChunkSize_(targetChunkSize)
    , currentChunkSize_(targetChunkSize) {
}

size_t AdaptiveChunkingStrategy::adjustChunkSize(
    std::chrono::milliseconds processingTime,
    double memoryPressure) {
    
    // 记录处理时间历史
    processingHistory_.push_back(processingTime);
    if (processingHistory_.size() > MAX_HISTORY_SIZE) {
        processingHistory_.erase(processingHistory_.begin());
    }
    
    // 计算平均处理时间
    double avgTime = getAverageProcessingTime();
    
    // 根据处理时间和内存压力调整块大小
    if (avgTime > 1000.0) { // 超过1秒，减小块大小
        currentChunkSize_ = std::max(minChunkSize_, 
                                   static_cast<size_t>(currentChunkSize_ * 0.8));
    } else if (avgTime < 100.0 && memoryPressure < 0.5) { // 处理快且内存充足，增大块大小
        currentChunkSize_ = std::min(maxChunkSize_, 
                                   static_cast<size_t>(currentChunkSize_ * 1.2));
    }
    
    return currentChunkSize_;
}

void AdaptiveChunkingStrategy::reset() {
    currentChunkSize_ = targetChunkSize_;
    processingHistory_.clear();
}

double AdaptiveChunkingStrategy::getAverageProcessingTime() const {
    if (processingHistory_.empty()) {
        return 0.0;
    }
    
    double total = 0.0;
    for (const auto& time : processingHistory_) {
        total += time.count();
    }
    return total / processingHistory_.size();
}

// =============================================================================
// BackpressureController 实现
// =============================================================================

BackpressureController::BackpressureController(
    size_t maxQueueSize,
    double backpressureThreshold,
    bool dropOnOverflow)
    : maxQueueSize_(maxQueueSize)
    , backpressureThreshold_(backpressureThreshold)
    , dropOnOverflow_(dropOnOverflow) {
}

bool BackpressureController::shouldApplyBackpressure(size_t currentQueueSize) const {
    return currentQueueSize >= static_cast<size_t>(maxQueueSize_ * backpressureThreshold_);
}

boost::future<bool> BackpressureController::waitForCapacity(
    size_t currentQueueSize, 
    std::chrono::milliseconds timeout) {
    
    return boost::async(boost::launch::async, [this, currentQueueSize, timeout]() -> bool {
        if (!shouldApplyBackpressure(currentQueueSize)) {
            return true;
        }
        
        std::unique_lock<std::mutex> lock(mutex_);
        return capacityCondition_.wait_for(lock, timeout, [this, currentQueueSize]() {
            return !shouldApplyBackpressure(currentQueueSize);
        });
    });
}

void BackpressureController::notifyCapacityChange() {
    std::lock_guard<std::mutex> lock(mutex_);
    capacityCondition_.notify_all();
}

// =============================================================================
// StreamingProcessor 实现
// =============================================================================

StreamingProcessor::StreamingProcessor()
    : isInitialized_(false)
    , defaultChunkSize_(1024 * 1024) // 1MB默认分块大小
    , maxConcurrentChunks_(8)
    , backpressureThreshold_(0.8) // 80%的缓冲区使用率触发背压
    , processedChunks_(0)
    , failedChunks_(0)
    , currentConcurrency_(0) {
    LOG_INFO("创建StreamingProcessor");
}

StreamingProcessor::~StreamingProcessor() {
    LOG_INFO("StreamingProcessor析构");
    shutdown();
}

void StreamingProcessor::initialize() {
    if (isInitialized_.load()) {
        return;
    }
    
    try {
        // 🔧 检查是否为单线程模式
        const char* runMode = std::getenv("OSCEAN_RUN_MODE");
        bool isSingleThreadMode = (runMode && std::string(runMode) == "SINGLE_THREAD");
        
        if (isSingleThreadMode) {
            // 单线程模式：不创建线程池
            threadPool_ = nullptr;
            LOG_INFO("StreamingProcessor运行在单线程模式，不创建线程池");
        } else {
            // 生产模式：创建线程池，但限制大小
            size_t poolSize = std::min(maxConcurrentChunks_, size_t{4});  // 最多4线程
            threadPool_ = std::make_unique<boost::asio::thread_pool>(poolSize);
            LOG_INFO("StreamingProcessor创建线程池，大小: {}", poolSize);
        }
        
        isInitialized_.store(true);
        LOG_INFO("StreamingProcessor初始化完成");
        
    } catch (const std::exception& e) {
        LOG_ERROR("StreamingProcessor初始化失败: {}", e.what());
        throw;
    }
}

void StreamingProcessor::shutdown() {
    if (!isInitialized_.load()) {
        return;
    }
    
    try {
        isInitialized_.store(false);
        
        // 停止线程池
        if (threadPool_) {
            threadPool_->stop();
            threadPool_->join();
            threadPool_.reset();
        }
        
        LOG_INFO("StreamingProcessor关闭完成");
        
    } catch (const std::exception& e) {
        LOG_ERROR("StreamingProcessor关闭异常: {}", e.what());
    }
}

boost::future<void> StreamingProcessor::processStreamAsync(
    std::shared_ptr<readers::UnifiedDataReader> reader,
    const std::string& variableName,
    std::function<boost::future<bool>(ProcessingDataChunk)> chunkProcessor,
    const StreamingOptions& options) {
    
    if (!isInitialized_.load()) {
        auto exception = std::runtime_error("StreamingProcessor未初始化");
        std::exception_ptr eptr = std::make_exception_ptr(exception);
        boost::promise<void> promise;
        promise.set_exception(eptr);
        return promise.get_future();
    }
    
    if (!reader) {
        auto exception = std::invalid_argument("reader不能为nullptr");
        std::exception_ptr eptr = std::make_exception_ptr(exception);
        boost::promise<void> promise;
        promise.set_exception(eptr);
        return promise.get_future();
    }
    
    if (!chunkProcessor) {
        auto exception = std::invalid_argument("chunkProcessor不能为nullptr");
        std::exception_ptr eptr = std::make_exception_ptr(exception);
        boost::promise<void> promise;
        promise.set_exception(eptr);
        return promise.get_future();
    }
    
    return boost::async(boost::launch::async, [this, reader, variableName, chunkProcessor, options]() {
        processStreamImpl(reader, variableName, chunkProcessor, options);
    });
}

void StreamingProcessor::setChunkSize(size_t chunkSize) {
    if (chunkSize > 0) {
        defaultChunkSize_ = chunkSize;
        LOG_INFO("设置默认分块大小: {}", chunkSize);
    }
}

void StreamingProcessor::setMaxConcurrentChunks(size_t maxChunks) {
    if (maxChunks > 0) {
        maxConcurrentChunks_ = maxChunks;
        LOG_INFO("设置最大并发分块数: {}", maxChunks);
    }
}

void StreamingProcessor::setBackpressureThreshold(double threshold) {
    if (threshold > 0.0 && threshold <= 1.0) {
        backpressureThreshold_ = threshold;
        LOG_INFO("设置背压阈值: {}", threshold);
    }
}

StreamingStatistics StreamingProcessor::getStatistics() const {
    StreamingStatistics stats;
    stats.processedChunks = processedChunks_.load();
    stats.failedChunks = failedChunks_.load();
    stats.currentConcurrency = currentConcurrency_.load();
    stats.maxConcurrency = maxConcurrentChunks_;
    stats.averageChunkSize = calculateAverageChunkSize();
    stats.successRate = calculateSuccessRate();
    
    return stats;
}

bool StreamingProcessor::startStreaming(const std::string& streamId) {
    std::unique_lock<std::shared_mutex> lock(streamsMutex_);
    
    if (activeStreams_.find(streamId) != activeStreams_.end()) {
        LOG_WARN("流已存在: {}", streamId);
        return false;
    }
    
    activeStreams_[streamId] = StreamState::ACTIVE;
    streamStats_[streamId] = StreamStatistics{};
    
    LOG_INFO("开始流处理: {}", streamId);
    return true;
}

bool StreamingProcessor::stopStreaming(const std::string& streamId) {
    std::unique_lock<std::shared_mutex> lock(streamsMutex_);
    
    auto it = activeStreams_.find(streamId);
    if (it == activeStreams_.end()) {
        LOG_WARN("流不存在: {}", streamId);
        return false;
    }
    
    it->second = StreamState::COMPLETED;
    LOG_INFO("停止流处理: {}", streamId);
    return true;
}

boost::future<bool> StreamingProcessor::submitChunk(const std::string& streamId, ProcessingDataChunk chunk) {
    return boost::async(boost::launch::async, [this, streamId, chunk]() -> bool {
        std::shared_lock<std::shared_mutex> lock(streamsMutex_);
        
        auto it = activeStreams_.find(streamId);
        if (it == activeStreams_.end() || it->second != StreamState::ACTIVE) {
            LOG_WARN("流不活跃或不存在: {}", streamId);
            return false;
        }
        
        try {
            // 简化的块处理逻辑
            LOG_DEBUG("处理数据块: stream={}, index={}, size={}", 
                     streamId, chunk.chunkIndex, chunk.data.size());
            
            // 更新统计信息
            {
                std::unique_lock<std::shared_mutex> statsLock(streamsMutex_);
                auto& stats = streamStats_[streamId];
                stats.totalChunksProcessed++;
                stats.totalBytesProcessed += chunk.data.size() * sizeof(double);
                stats.updateThroughput();
            }
            
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR("处理数据块异常: {} - {}", streamId, e.what());
            return false;
        }
    });
}

StreamStatistics StreamingProcessor::getStreamStatistics(const std::string& streamId) const {
    std::shared_lock<std::shared_mutex> lock(streamsMutex_);
    
    auto it = streamStats_.find(streamId);
    if (it != streamStats_.end()) {
        return it->second;
    }
    
    return StreamStatistics{};
}

void StreamingProcessor::processStreamImpl(
    std::shared_ptr<readers::UnifiedDataReader> reader,
    const std::string& variableName,
    std::function<boost::future<bool>(ProcessingDataChunk)> chunkProcessor,
    const StreamingOptions& options) {
    
    try {
        // 计算分块策略
        auto chunkingStrategy = calculateChunkingStrategy(variableName, reader, options);
        
        // 流式读取和处理数据
        size_t totalChunks = chunkingStrategy.totalChunks;
        std::vector<boost::future<bool>> chunkFutures;
        chunkFutures.reserve(totalChunks);
        
        for (size_t chunkIndex = 0; chunkIndex < totalChunks; ++chunkIndex) {
            // 检查背压
            if (shouldApplyBackpressure()) {
                waitForBackpressureRelief();
            }
            
            // 创建分块任务
            auto chunkFuture = createChunkTask(reader, variableName, chunkIndex, chunkingStrategy, chunkProcessor);
            chunkFutures.push_back(std::move(chunkFuture));
            
            currentConcurrency_.fetch_add(1);
        }
        
        // 等待所有分块完成
        for (auto& future : chunkFutures) {
            try {
                bool success = future.get();
                if (success) {
                    processedChunks_.fetch_add(1);
                } else {
                    failedChunks_.fetch_add(1);
                }
            } catch (const std::exception& e) {
                LOG_ERROR("分块处理异常: {}", e.what());
                failedChunks_.fetch_add(1);
            }
            currentConcurrency_.fetch_sub(1);
        }
        
        LOG_INFO("流式处理完成: {} chunks processed, {} failed", 
                processedChunks_.load(), failedChunks_.load());
        
    } catch (const std::exception& e) {
        LOG_ERROR("流式处理失败 [{}::{}]: {}", reader->getFilePath(), variableName, e.what());
        throw;
    }
}

ChunkingStrategy StreamingProcessor::calculateChunkingStrategy(
    const std::string& variableName,
    std::shared_ptr<readers::UnifiedDataReader> reader,
    const StreamingOptions& options) {
    
    ChunkingStrategy strategy;
    
    try {
        // 估算变量大小
        size_t estimatedSize = estimateVariableSize(variableName, reader);
        
        // 计算分块参数
        size_t chunkSize = options.chunkSize > 0 ? options.chunkSize : defaultChunkSize_;
        strategy.chunkSize = chunkSize;
        strategy.totalChunks = (estimatedSize + chunkSize - 1) / chunkSize; // 向上取整
        strategy.lastChunkSize = estimatedSize % chunkSize;
        if (strategy.lastChunkSize == 0) {
            strategy.lastChunkSize = chunkSize;
        }
        strategy.chunkingType = ChunkingType::SEQUENTIAL;
        
        LOG_DEBUG("分块策略: 总大小={}, 块大小={}, 总块数={}", 
                 estimatedSize, chunkSize, strategy.totalChunks);
        
    } catch (const std::exception& e) {
        LOG_ERROR("计算分块策略失败: {}", e.what());
        // 使用默认策略
        strategy.chunkSize = defaultChunkSize_;
        strategy.totalChunks = 1;
        strategy.lastChunkSize = defaultChunkSize_;
        strategy.chunkingType = ChunkingType::SEQUENTIAL;
    }
    
    return strategy;
}

boost::future<bool> StreamingProcessor::createChunkTask(
    std::shared_ptr<readers::UnifiedDataReader> reader,
    const std::string& variableName,
    size_t chunkIndex,
    const ChunkingStrategy& strategy,
    std::function<boost::future<bool>(ProcessingDataChunk)> chunkProcessor) {
    
    return boost::async(boost::launch::async, [this, reader, variableName, chunkIndex, strategy, chunkProcessor]() -> bool {
        try {
            // 计算当前块的范围
            auto range = calculateChunkRange(chunkIndex, strategy);
            
            // 读取块数据
            auto chunkData = readChunkDataComplete(reader, variableName, range, strategy);
            if (!chunkData) {
                LOG_ERROR("读取块数据失败: chunk={}", chunkIndex);
                return false;
            }
            
            // 创建ProcessingDataChunk对象
            ProcessingDataChunk chunk;
            chunk.data = *chunkData;
            chunk.chunkIndex = chunkIndex;
            chunk.totalChunks = strategy.totalChunks;
            chunk.isLastChunk = (chunkIndex == strategy.totalChunks - 1);
            
            // 设置DataChunkKey
            chunk.chunkKey = oscean::core_services::DataChunkKey(
                "", variableName, boost::none, boost::none, boost::none, "double");
            
            // 处理块
            auto processFuture = chunkProcessor(chunk);
            return processFuture.get();
            
        } catch (const std::exception& e) {
            LOG_ERROR("创建分块任务异常: chunk={} - {}", chunkIndex, e.what());
            return false;
        }
    });
}

bool StreamingProcessor::shouldApplyBackpressure() const {
    return currentConcurrency_.load() >= static_cast<size_t>(maxConcurrentChunks_ * backpressureThreshold_);
}

void StreamingProcessor::waitForBackpressureRelief() {
    while (shouldApplyBackpressure()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

size_t StreamingProcessor::estimateVariableSize(const std::string& variableName, 
                                               std::shared_ptr<readers::UnifiedDataReader> reader) const {
    // 简化实现：返回默认大小
    return 10 * 1024 * 1024; // 10MB
}

size_t StreamingProcessor::getDataTypeSize(oscean::core_services::DataType dataType) const {
    switch (dataType) {
        case oscean::core_services::DataType::Byte:
            return sizeof(uint8_t);
        case oscean::core_services::DataType::Int16:
            return sizeof(int16_t);
        case oscean::core_services::DataType::UInt16:
            return sizeof(uint16_t);
        case oscean::core_services::DataType::Int32:
            return sizeof(int32_t);
        case oscean::core_services::DataType::UInt32:
            return sizeof(uint32_t);
        case oscean::core_services::DataType::Float32:
            return sizeof(float);
        case oscean::core_services::DataType::Float64:
            return sizeof(double);
        default:
            return sizeof(double); // 默认
    }
}

oscean::core_services::IndexRange StreamingProcessor::calculateChunkRange(size_t chunkIndex, const ChunkingStrategy& strategy) const {
    oscean::core_services::IndexRange range;
    range.start = static_cast<int>(chunkIndex * strategy.chunkSize);
    
    if (chunkIndex == strategy.totalChunks - 1) {
        // 最后一块
        range.count = static_cast<int>(strategy.lastChunkSize);
    } else {
        range.count = static_cast<int>(strategy.chunkSize);
    }
    
    // IndexRange不需要endIndex字段，已经有start和count
    
    return range;
}

std::unique_ptr<std::vector<double>> StreamingProcessor::readChunkDataComplete(
    std::shared_ptr<readers::UnifiedDataReader> reader,
    const std::string& variableName,
    const oscean::core_services::IndexRange& range,
    const ChunkingStrategy& strategy) const {
    
    try {
        // 简化实现：创建模拟数据
        auto data = std::make_unique<std::vector<double>>(range.count);
        
        // 填充模拟数据
        for (size_t i = 0; i < range.count; ++i) {
            (*data)[i] = static_cast<double>(range.start + i) * 0.1;
        }
        
        return data;
        
    } catch (const std::exception& e) {
        LOG_ERROR("读取块数据异常: {} - {}", variableName, e.what());
        return nullptr;
    }
}

std::unique_ptr<std::vector<double>> StreamingProcessor::readSpatialChunk(
    std::shared_ptr<readers::UnifiedDataReader> reader,
    const std::string& variableName,
    const StreamDataRange& range) const {
    
    // 简化实现：委托给完整读取方法
    ChunkingStrategy strategy;
    strategy.chunkSize = range.count;
    return readChunkDataComplete(reader, variableName, range, strategy);
}

std::unique_ptr<std::vector<double>> StreamingProcessor::readSequentialChunk(
    std::shared_ptr<readers::UnifiedDataReader> reader,
    const std::string& variableName,
    const StreamDataRange& range) const {
    
    // 简化实现：委托给完整读取方法
    ChunkingStrategy strategy;
    strategy.chunkSize = range.count;
    return readChunkDataComplete(reader, variableName, range, strategy);
}

double StreamingProcessor::calculateAverageChunkSize() const {
    size_t totalChunks = processedChunks_.load() + failedChunks_.load();
    if (totalChunks == 0) {
        return 0.0;
    }
    return static_cast<double>(defaultChunkSize_);
}

double StreamingProcessor::calculateSuccessRate() const {
    size_t totalChunks = processedChunks_.load() + failedChunks_.load();
    if (totalChunks == 0) {
        return 1.0;
    }
    return static_cast<double>(processedChunks_.load()) / totalChunks;
}

} // namespace oscean::core_services::data_access::streaming 
