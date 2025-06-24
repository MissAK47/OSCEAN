/**
 * @file large_file_processor.cpp
 * @brief 大文件处理器实现 - 整合streaming模块核心功能
 * @author OSCEAN Team
 * @date 2024
 */

#include "common_utils/infrastructure/large_file_processor.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/utilities/file_format_detector.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <thread>

namespace oscean::common_utils::infrastructure {

// === LargeFileConfig 实现 ===

std::string LargeFileConfig::toString() const {
    std::ostringstream oss;
    oss << "LargeFileConfig {\n";
    oss << "  Strategy: " << static_cast<int>(strategy) << "\n";
    oss << "  Max Memory: " << maxMemoryUsageMB << "MB\n";
    oss << "  Chunk Size: " << chunkSizeMB << "MB\n";
    oss << "  IO Threads: " << ioThreads << "\n";
    oss << "  Processing Threads: " << processingThreads << "\n";
    oss << "}";
    return oss.str();
}

LargeFileConfig LargeFileConfig::createOptimal() {
    LargeFileConfig config;
    config.strategy = LargeFileStrategy::BALANCED;
    config.maxMemoryUsageMB = 512;
    config.chunkSizeMB = 32;
    config.ioThreads = 2;
    config.processingThreads = std::thread::hardware_concurrency();
    return config;
}

LargeFileConfig LargeFileConfig::createForStrategy(LargeFileStrategy strategy) {
    LargeFileConfig config;
    config.strategy = strategy;
    
    switch (strategy) {
        case LargeFileStrategy::MEMORY_CONSERVATIVE:
            config.maxMemoryUsageMB = 128;
            config.chunkSizeMB = 8;
            config.bufferPoolSizeMB = 32;
            config.maxConcurrentChunks = 2;
            break;
            
        case LargeFileStrategy::PERFORMANCE_FIRST:
            config.maxMemoryUsageMB = 2048;
            config.chunkSizeMB = 128;
            config.bufferPoolSizeMB = 512;
            config.maxConcurrentChunks = 16;
            break;
            
        case LargeFileStrategy::ADAPTIVE:
        case LargeFileStrategy::BALANCED:
        default:
            config = createOptimal();
            break;
    }
    
    return config;
}

// === LargeFileInfo 实现 ===

std::string LargeFileInfo::toString() const {
    std::ostringstream oss;
    oss << "LargeFileInfo {\n";
    oss << "  Path: " << filePath << "\n";
    oss << "  Size: " << fileSizeBytes << " bytes\n";
    oss << "  Type: " << static_cast<int>(fileType) << "\n";
    oss << "  Estimated Records: " << estimatedRecords << "\n";
    oss << "  Processing Time: " << estimatedProcessingTime.count() << "ms\n";
    oss << "}";
    return oss.str();
}

// === LargeFileProcessor 实现 ===

LargeFileProcessor::LargeFileProcessor(
    const LargeFileConfig& config,
    std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager,
    std::shared_ptr<UnifiedThreadPoolManager> threadPoolManager)
    : config_(config), memoryManager_(memoryManager), threadPoolManager_(threadPoolManager) {
    
    std::cout << "LargeFileProcessor: Initialized with config: " << config_.toString() << std::endl;
}

LargeFileProcessor::~LargeFileProcessor() {
    if (processing_) {
        cancelProcessing();
    }
}

ProcessingStatus LargeFileProcessor::processFile(
    const std::string& filePath,
    const LargeFileConfig& config,
    std::shared_ptr<IProgressObserver> observer) {
    
    std::cout << "LargeFileProcessor: Processing file: " << filePath << std::endl;
    
    // 使用提供的配置或默认配置
    LargeFileConfig actualConfig = config.maxMemoryUsageMB > 0 ? config : config_;
    
    return processInternal(filePath, actualConfig, observer);
}

boost::future<ProcessingStatus> LargeFileProcessor::processFileAsync(
    const std::string& filePath,
    const LargeFileConfig& config,
    std::shared_ptr<IProgressObserver> observer) {
    
    return executeAsync([this, filePath, config, observer]() -> ProcessingStatus {
        return processFile(filePath, config, observer);
    });
}

void LargeFileProcessor::processInChunks(
    const std::string& filePath,
    std::function<bool(const DataChunk&)> processor,
    const LargeFileConfig& config) {
    
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error("File does not exist: " + filePath);
    }
    
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }
    
    // 使用提供的配置或默认配置
    LargeFileConfig actualConfig = config.maxMemoryUsageMB > 0 ? config : config_;
    size_t chunkSize = actualConfig.chunkSizeMB * 1024 * 1024;
    
    processing_ = true;
    size_t totalProcessed = 0;
    
    try {
        while (!file.eof() && !cancelled_) {
            while (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            DataChunk chunk(chunkSize);
            file.read(reinterpret_cast<char*>(chunk.data.data()), chunkSize);
            chunk.size = file.gcount();
            chunk.offset = totalProcessed;
            chunk.isLast = file.eof();
            
            if (chunk.size > 0) {
                bool shouldContinue = processor(chunk);
                if (!shouldContinue) {
                    break;
                }
                
                totalProcessed += chunk.size;
                updateCheckpoint(totalProcessed, 0);
            }
        }
    } catch (const std::exception& e) {
        std::cout << "LargeFileProcessor: Error during chunk processing: " << e.what() << std::endl;
        processing_ = false;
        throw;
    }
    
    processing_ = false;
    std::cout << "LargeFileProcessor: Completed chunk processing. Total processed: " 
              << totalProcessed << " bytes" << std::endl;
}

std::chrono::milliseconds LargeFileProcessor::estimateProcessingTime(
    const std::string& filePath,
    const LargeFileConfig& config) const {
    
    if (!std::filesystem::exists(filePath)) {
        return std::chrono::milliseconds(0);
    }
    
    // 🔴 修复：直接计算文件大小，避免调用analyzeFile造成递归
    size_t fileSizeBytes = std::filesystem::file_size(filePath);
    LargeFileConfig actualConfig = config.maxMemoryUsageMB > 0 ? config : config_;
    
    // 简化的时间估算：基于文件大小和配置
    double sizeMB = static_cast<double>(fileSizeBytes) / (1024.0 * 1024.0);
    double throughputMBps = 50.0; // 假设吞吐量50MB/s
    
    // 根据策略调整吞吐量
    switch (actualConfig.strategy) {
        case LargeFileStrategy::MEMORY_CONSERVATIVE:
            throughputMBps *= 0.7; // 保守策略降低30%
            break;
        case LargeFileStrategy::PERFORMANCE_FIRST:
            throughputMBps *= 1.5; // 性能优先提升50%
            break;
        default:
            break;
    }
    
    double estimatedSeconds = sizeMB / throughputMBps;
    return std::chrono::milliseconds(static_cast<long>(estimatedSeconds * 1000));
}

LargeFileConfig LargeFileProcessor::getOptimizedConfig(const std::string& filePath) const {
    auto fileInfo = analyzeFile(filePath);
    return optimizeConfigForFile(fileInfo);
}

LargeFileInfo LargeFileProcessor::analyzeFile(const std::string& filePath) const {
    LargeFileInfo info;
    info.filePath = filePath;
    
    if (!std::filesystem::exists(filePath)) {
        return info;
    }
    
    // 获取文件大小
    info.fileSizeBytes = std::filesystem::file_size(filePath);
    
    // 检测文件类型
    info.fileType = detectFileType(filePath);
    
    // 检查是否压缩
    info.isCompressed = isFileCompressed(filePath);
    
    // 估算记录数
    info.estimatedRecords = estimateFileRecords(filePath, info.fileType);
    
    // 计算推荐配置
    double sizeMB = static_cast<double>(info.fileSizeBytes) / (1024.0 * 1024.0);
    
    info.recommendedMemoryMB = std::min(static_cast<size_t>(512), std::max(static_cast<size_t>(128), static_cast<size_t>(sizeMB * 0.1)));
    info.recommendedChunkSizeMB = calculateOptimalChunkSize(info);
    info.recommendedParallelism = calculateOptimalParallelism(info);
    
    // 估算处理时间
    info.estimatedProcessingTime = estimateProcessingTime(filePath, LargeFileConfig::createOptimal());
    
    return info;
}

bool LargeFileProcessor::canProcessFile(const std::string& filePath) const {
    if (!std::filesystem::exists(filePath)) {
        return false;
    }
    
    // 检查文件大小是否超过限制
    size_t fileSize = std::filesystem::file_size(filePath);
    size_t maxSize = 10ULL * 1024 * 1024 * 1024; // 10GB限制
    
    return fileSize <= maxSize;
}

std::vector<std::string> LargeFileProcessor::getProcessingRecommendations(
    const std::string& filePath) const {
    
    std::vector<std::string> recommendations;
    auto fileInfo = analyzeFile(filePath);
    
    double sizeMB = static_cast<double>(fileInfo.fileSizeBytes) / (1024.0 * 1024.0);
    
    if (sizeMB > 1024) {
        recommendations.push_back("使用MEMORY_CONSERVATIVE策略以减少内存使用");
        recommendations.push_back("启用检查点以防止数据丢失");
    }
    
    if (fileInfo.isCompressed) {
        recommendations.push_back("预分配额外内存用于解压缩");
    }
    
    if (sizeMB > 5000) {
        recommendations.push_back("考虑使用多节点分布式处理");
    }
    
    return recommendations;
}

void LargeFileProcessor::updateConfig(const LargeFileConfig& config) {
    config_ = config;
    std::cout << "LargeFileProcessor: Configuration updated" << std::endl;
}

void LargeFileProcessor::pauseProcessing() {
    paused_ = true;
    std::cout << "LargeFileProcessor: Processing paused" << std::endl;
}

void LargeFileProcessor::resumeProcessing() {
    paused_ = false;
    std::cout << "LargeFileProcessor: Processing resumed" << std::endl;
}

void LargeFileProcessor::cancelProcessing() {
    cancelled_ = true;
    processing_ = false;
    std::cout << "LargeFileProcessor: Processing cancelled" << std::endl;
}

bool LargeFileProcessor::saveCheckpoint(const std::string& checkpointPath) const {
    std::lock_guard<std::mutex> lock(checkpointMutex_);
    
    if (!currentCheckpoint_) {
        return false;
    }
    
    // 简化的检查点保存实现
    std::ofstream checkpoint(checkpointPath);
    if (!checkpoint) {
        return false;
    }
    
    checkpoint << currentCheckpoint_->filePath << "\n";
    checkpoint << currentCheckpoint_->processedBytes << "\n";
    checkpoint << currentCheckpoint_->processedChunks << "\n";
    
    return true;
}

bool LargeFileProcessor::loadCheckpoint(const std::string& checkpointPath) {
    std::lock_guard<std::mutex> lock(checkpointMutex_);
    
    std::ifstream checkpoint(checkpointPath);
    if (!checkpoint) {
        return false;
    }
    
    CheckpointData data;
    checkpoint >> data.filePath >> data.processedBytes >> data.processedChunks;
    
    if (checkpoint.good()) {
        currentCheckpoint_ = data;
        return true;
    }
    
    return false;
}

void LargeFileProcessor::clearCheckpoints() {
    std::lock_guard<std::mutex> lock(checkpointMutex_);
    currentCheckpoint_.reset();
}

LargeFileProcessor::ProcessingStats LargeFileProcessor::getProcessingStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}

void LargeFileProcessor::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_ = ProcessingStats{};
}

// === 私有方法实现 ===

ProcessingStatus LargeFileProcessor::processInternal(
    const std::string& filePath,
    const LargeFileConfig& config,
    std::shared_ptr<IProgressObserver> observer) {
    
    processing_ = true;
    cancelled_ = false;
    paused_ = false;
    
    try {
        auto fileInfo = analyzeFile(filePath);
        setupProcessingEnvironment(fileInfo);
        
        if (observer) {
            observer->onProgress(0.0, "开始处理文件: " + filePath);
        }
        
        // 简化的处理逻辑
        processInChunks(filePath, [observer](const DataChunk& chunk) -> bool {
            if (observer) {
                double progress = static_cast<double>(chunk.offset) / 
                                static_cast<double>(chunk.offset + chunk.size) * 100.0;
                observer->onProgress(progress, "处理数据块");
            }
            
            // 这里应该是实际的数据处理逻辑
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 模拟处理时间
            
            return true; // 继续处理
        }, config);
        
        if (observer) {
            observer->onComplete();
        }
        
        cleanupProcessingEnvironment();
        processing_ = false;
        
        return ProcessingStatus::SUCCESS;
        
    } catch (const std::exception& e) {
        if (observer) {
            observer->onError("处理错误: " + std::string(e.what()));
        }
        
        cleanupProcessingEnvironment();
        processing_ = false;
        
        return ProcessingStatus::FAILED;
    }
}

void LargeFileProcessor::setupProcessingEnvironment(const LargeFileInfo& fileInfo) {
    std::cout << "LargeFileProcessor: Setting up processing environment for: " 
              << fileInfo.filePath << std::endl;
}

void LargeFileProcessor::cleanupProcessingEnvironment() {
    std::cout << "LargeFileProcessor: Cleaning up processing environment" << std::endl;
}

FileType LargeFileProcessor::detectFileType(const std::string& filePath) const {
    // 检测文件格式
    auto detector = oscean::common_utils::utilities::FileFormatDetector::createDetector();
    auto formatResult = detector->detectFormat(filePath);
    auto format = formatResult.format;
    
    switch (format) {
        case oscean::common_utils::utilities::FileFormat::NETCDF3:
        case oscean::common_utils::utilities::FileFormat::NETCDF4:
            return FileType::NETCDF;
        case oscean::common_utils::utilities::FileFormat::HDF5:
            return FileType::HDF5;
        case oscean::common_utils::utilities::FileFormat::GEOTIFF:
            return FileType::GEOTIFF;
        case oscean::common_utils::utilities::FileFormat::SHAPEFILE:
            return FileType::SHAPEFILE;
        case oscean::common_utils::utilities::FileFormat::CSV:
            return FileType::CSV;
        case oscean::common_utils::utilities::FileFormat::JSON:
            return FileType::JSON;
        default:
            return FileType::AUTO_DETECT;
    }
}

size_t LargeFileProcessor::estimateFileRecords(const std::string& filePath, FileType type) const {
    // 简化的记录数估算
    size_t fileSize = std::filesystem::file_size(filePath);
    
    switch (type) {
        case FileType::CSV:
            return fileSize / 100; // 假设每条记录平均100字节
        case FileType::JSON:
            return fileSize / 500; // JSON通常更大
        case FileType::NETCDF:
        case FileType::HDF5:
            return fileSize / 1000; // 科学数据格式
        default:
            return fileSize / 200; // 默认估算
    }
}

bool LargeFileProcessor::isFileCompressed(const std::string& filePath) const {
    std::string extension = std::filesystem::path(filePath).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return extension == ".gz" || extension == ".zip" || extension == ".bz2" || 
           extension == ".xz" || extension == ".lz4";
}

LargeFileConfig LargeFileProcessor::optimizeConfigForFile(const LargeFileInfo& fileInfo) const {
    LargeFileConfig config = config_;
    
    double sizeMB = static_cast<double>(fileInfo.fileSizeBytes) / (1024.0 * 1024.0);
    
    // 根据文件大小调整配置
    if (sizeMB > 5000) { // 大于5GB
        config.strategy = LargeFileStrategy::MEMORY_CONSERVATIVE;
        config.maxMemoryUsageMB = 256;
        config.chunkSizeMB = 16;
    } else if (sizeMB > 1000) { // 大于1GB
        config.strategy = LargeFileStrategy::BALANCED;
        config.maxMemoryUsageMB = 512;
        config.chunkSizeMB = 32;
    } else {
        config.strategy = LargeFileStrategy::PERFORMANCE_FIRST;
        config.maxMemoryUsageMB = 1024;
        config.chunkSizeMB = 64;
    }
    
    return config;
}

size_t LargeFileProcessor::calculateOptimalChunkSize(const LargeFileInfo& fileInfo) const {
    // 基于文件大小和内存计算最优块大小
    size_t sizeMB = fileInfo.fileSizeBytes / (1024 * 1024);
    
    if (sizeMB > 1000) {
        return 128; // 大文件使用大块
    } else if (sizeMB > 100) {
        return 64;  // 中等文件使用中等块
    } else {
        return 16;  // 小文件使用小块
    }
}

size_t LargeFileProcessor::calculateOptimalParallelism(const LargeFileInfo& fileInfo) const {
    // 基于文件大小和系统能力计算最优并行度
    size_t sizeMB = fileInfo.fileSizeBytes / (1024 * 1024);
    size_t maxThreads = std::thread::hardware_concurrency();
    
    if (sizeMB > 1000) {
        return std::min(static_cast<size_t>(8), maxThreads);
    } else if (sizeMB > 100) {
        return std::min(static_cast<size_t>(4), maxThreads);
    } else {
        return std::min(static_cast<size_t>(2), maxThreads);
    }
}

void LargeFileProcessor::updateCheckpoint(size_t processedBytes, size_t processedChunks) {
    if (!enableCheckpointing_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(checkpointMutex_);
    if (currentCheckpoint_) {
        currentCheckpoint_->processedBytes = processedBytes;
        currentCheckpoint_->processedChunks = processedChunks;
    }
}

std::string LargeFileProcessor::generateCheckpointPath(const std::string& filePath) const {
    return filePath + ".checkpoint";
}

template<typename Func>
auto LargeFileProcessor::executeAsync(Func&& func) -> boost::future<decltype(func())> {
    if (threadPoolManager_) {
        // 🔧 使用新的线程池管理器API提交文件I/O任务
        return threadPoolManager_->submitFileTask(std::forward<Func>(func), 
                                                  "", // filePath 可选
                                                  TaskPriority::NORMAL);
    }
    
    // 回退到同步执行
    auto promise = std::make_shared<boost::promise<decltype(func())>>();
    auto future = promise->get_future();
    
    try {
        if constexpr (std::is_void_v<decltype(func())>) {
            func();
            promise->set_value();
        } else {
            promise->set_value(func());
        }
    } catch (...) {
        promise->set_exception(std::current_exception());
    }
    
    return future;
}

} // namespace oscean::common_utils::infrastructure 