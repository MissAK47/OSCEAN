/**
 * @file data_streaming_coordinator.cpp
 * @brief 数据访问专用流式处理协调器实现
 */

#include "data_streaming_coordinator.h"
#include "common_utils/utilities/logging_utils.h"
#include <thread>
#include <chrono>

namespace oscean::core_services::data_access::readers::impl {

DataStreamingCoordinator::DataStreamingCoordinator(
    std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager,
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework)
    : memoryManager_(memoryManager)
    , asyncFramework_(asyncFramework)
    , startTime_(std::chrono::steady_clock::now()) {
    
    LOG_INFO("DataStreamingCoordinator初始化");
}

void DataStreamingCoordinator::configureStreaming(const StreamingConfig& config) {
    config_ = config;
    LOG_INFO("流式处理配置: 块大小={}MB, 最大并发块={}, 背压={}", 
             config_.chunkSize / (1024 * 1024), 
             config_.maxConcurrentChunks,
             config_.enableBackpressure ? "启用" : "禁用");
}

void DataStreamingCoordinator::configureForGdal(GDALDataset* dataset) {
    // GDAL特定的流式配置
    LOG_INFO("配置GDAL流式处理");
    // 这里可以根据GDAL数据集特性调整配置
}

void DataStreamingCoordinator::configureForNetCDF(ncid_t ncid) {
    // NetCDF特定的流式配置
    LOG_INFO("配置NetCDF流式处理: ncid={}", ncid);
    // 这里可以根据NetCDF文件特性调整配置
}

boost::future<void> DataStreamingCoordinator::streamVariable(
    const std::string& variableName,
    const std::optional<oscean::core_services::BoundingBox>& bounds,
    std::function<bool(const DataChunk&)> processor,
    std::shared_ptr<FormatStreamingAdapter> adapter) {
    
    return boost::async(boost::launch::async, [this, variableName, bounds, processor, adapter]() {
        LOG_INFO("开始流式读取变量: {}", variableName);
        
        if (!adapter) {
            LOG_ERROR("流式适配器为空");
            return;
        }
        
        adapter->configureStreaming(config_);
        
        size_t chunkCount = 0;
        auto startTime = std::chrono::steady_clock::now();
        
        try {
            while (true) {
                // 背压控制
                if (config_.enableBackpressure && shouldApplyBackpressure()) {
                    waitForBackpressureRelief();
                }
                
                // 读取下一个数据块
                auto chunkFuture = adapter->readNextChunk();
                auto chunkOpt = chunkFuture.get();
                
                if (!chunkOpt) {
                    LOG_INFO("变量 {} 流式读取完成，共处理 {} 个块", variableName, chunkCount);
                    break;
                }
                
                auto& chunk = *chunkOpt;
                chunk.chunkId = chunkIdCounter_++;
                activeChunks_++;
                
                auto chunkStartTime = std::chrono::steady_clock::now();
                
                // 处理数据块
                bool shouldContinue = processor(chunk);
                
                auto chunkEndTime = std::chrono::steady_clock::now();
                auto chunkProcessingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    chunkEndTime - chunkStartTime);
                
                // 更新统计信息
                updateStats(chunk.data.size() * sizeof(double), chunkProcessingTime);
                
                activeChunks_--;
                chunkCount++;
                
                if (!shouldContinue) {
                    LOG_INFO("处理器要求停止，变量 {} 处理了 {} 个块", variableName, chunkCount);
                    break;
                }
                
                if (chunk.isLastChunk) {
                    LOG_INFO("变量 {} 最后一个块处理完成，共处理 {} 个块", variableName, chunkCount);
                    break;
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("流式读取变量 {} 时发生异常: {}", variableName, e.what());
            activeChunks_--;
        }
        
        auto endTime = std::chrono::steady_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        LOG_INFO("变量 {} 流式读取完成，总耗时: {}ms", variableName, totalTime.count());
    });
}

boost::future<void> DataStreamingCoordinator::streamMultipleVariables(
    const std::vector<std::string>& variableNames,
    const std::optional<oscean::core_services::BoundingBox>& bounds,
    std::function<bool(const std::string&, const DataChunk&)> processor,
    std::shared_ptr<FormatStreamingAdapter> adapter) {
    
    return boost::async(boost::launch::async, [this, variableNames, bounds, processor, adapter]() {
        LOG_INFO("开始并行流式读取 {} 个变量", variableNames.size());
        
        std::vector<boost::future<void>> futures;
        
        for (const auto& varName : variableNames) {
            auto varProcessor = [processor, varName](const DataChunk& chunk) -> bool {
                return processor(varName, chunk);
            };
            
            auto future = streamVariable(varName, bounds, varProcessor, adapter);
            futures.push_back(std::move(future));
        }
        
        // 等待所有变量处理完成
        for (auto& future : futures) {
            future.wait();
        }
        
        LOG_INFO("所有变量并行流式读取完成");
    });
}

bool DataStreamingCoordinator::shouldApplyBackpressure() const {
    if (!config_.enableBackpressure) {
        return false;
    }
    
    // 检查活跃块数量
    if (activeChunks_.load() >= config_.maxConcurrentChunks) {
        return true;
    }
    
    // 检查内存使用率
    if (checkMemoryThreshold()) {
        return true;
    }
    
    return false;
}

void DataStreamingCoordinator::waitForBackpressureRelief() {
    LOG_DEBUG("应用背压控制，等待资源释放");
    
    while (shouldApplyBackpressure()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    LOG_DEBUG("背压缓解，继续处理");
}

double DataStreamingCoordinator::getCurrentMemoryUsage() const {
    if (memoryManager_) {
        auto stats = memoryManager_->getUsageStats();
        if (stats.totalAllocated > 0) {
            return static_cast<double>(stats.currentUsed) / stats.totalAllocated;
        }
    }
    return 0.0;
}

size_t DataStreamingCoordinator::getActiveChunksCount() const {
    return activeChunks_.load();
}

DataStreamingCoordinator::StreamingStats DataStreamingCoordinator::getStreamingStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    auto stats = stats_;
    stats.currentActiveChunks = activeChunks_.load();
    return stats;
}

void DataStreamingCoordinator::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_ = StreamingStats{};
    startTime_ = std::chrono::steady_clock::now();
    LOG_INFO("流式处理统计信息已重置");
}

void DataStreamingCoordinator::updateStats(size_t bytesProcessed, std::chrono::milliseconds processingTime) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    stats_.totalChunksProcessed++;
    stats_.totalBytesProcessed += bytesProcessed;
    stats_.totalProcessingTime += processingTime;
    
    if (stats_.totalChunksProcessed > 0) {
        stats_.averageChunkProcessingTime = 
            static_cast<double>(stats_.totalProcessingTime.count()) / stats_.totalChunksProcessed;
    }
}

bool DataStreamingCoordinator::checkMemoryThreshold() const {
    if (!memoryManager_) {
        return false;
    }
    
    auto stats = memoryManager_->getUsageStats();
    double currentUsage = 0.0;
    if (stats.totalAllocated > 0) {
        currentUsage = static_cast<double>(stats.currentUsed) / stats.totalAllocated;
    }
    return currentUsage > config_.memoryThreshold;
}

void DataStreamingCoordinator::waitForMemoryAvailable() {
    while (checkMemoryThreshold()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

} // namespace oscean::core_services::data_access::readers::impl 