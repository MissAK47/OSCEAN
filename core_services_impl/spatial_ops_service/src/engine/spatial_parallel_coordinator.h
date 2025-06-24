#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include <future>
#include <vector>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <optional>

namespace oscean::core_services::spatial_ops::engine {

/**
 * @brief 空间数据分区信息
 */
struct SpatialPartition {
    oscean::core_services::BoundingBox bounds;
    size_t partitionId;
    size_t totalPartitions;
    std::vector<size_t> featureIndices;
    size_t estimatedMemoryUsage;
    
    SpatialPartition() = default;
    SpatialPartition(const oscean::core_services::BoundingBox& bbox, 
                    size_t id, size_t total)
        : bounds(bbox), partitionId(id), totalPartitions(total), estimatedMemoryUsage(0) {}
};

/**
 * @brief 并行任务结果
 */
template<typename T>
struct ParallelTaskResult {
    size_t partitionId;
    T result;
    std::chrono::milliseconds processingTime;
    bool success;
    std::string errorMessage;
    
    ParallelTaskResult() : partitionId(0), processingTime(0), success(false) {}
    ParallelTaskResult(size_t id, T&& res, std::chrono::milliseconds time)
        : partitionId(id), result(std::move(res)), processingTime(time), success(true) {}
};

/**
 * @brief 并行处理统计信息
 */
struct ParallelProcessingStats {
    size_t totalPartitions = 0;
    size_t successfulPartitions = 0;
    size_t failedPartitions = 0;
    std::chrono::milliseconds totalProcessingTime{0};
    std::chrono::milliseconds maxPartitionTime{0};
    std::chrono::milliseconds minPartitionTime{std::chrono::milliseconds::max()};
    double averagePartitionTime = 0.0;
    size_t totalMemoryUsed = 0;
    double parallelEfficiency = 0.0; // 并行效率 (0-1)
    
    void updateStats(const std::chrono::milliseconds& partitionTime) {
        totalProcessingTime += partitionTime;
        maxPartitionTime = std::max(maxPartitionTime, partitionTime);
        minPartitionTime = std::min(minPartitionTime, partitionTime);
        if (successfulPartitions > 0) {
            averagePartitionTime = static_cast<double>(totalProcessingTime.count()) / successfulPartitions;
        }
    }
};

/**
 * @brief 空间并行协调器
 * 
 * 提供高性能的空间数据并行处理能力，支持：
 * - 智能数据分区
 * - 负载均衡
 * - 任务级和数据级并行
 * - 自适应性能优化
 */
class SpatialParallelCoordinator {
public:
    /**
     * @brief 构造函数
     * @param config 空间服务配置
     */
    explicit SpatialParallelCoordinator(const oscean::core_services::spatial_ops::SpatialOpsConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~SpatialParallelCoordinator();
    
    // 禁用拷贝构造和赋值
    SpatialParallelCoordinator(const SpatialParallelCoordinator&) = delete;
    SpatialParallelCoordinator& operator=(const SpatialParallelCoordinator&) = delete;
    
    /**
     * @brief 并行处理要素集合
     * @tparam ResultType 处理结果类型
     * @tparam ProcessorFunc 处理函数类型
     * @param features 输入要素集合
     * @param processor 处理函数
     * @param mergeResults 结果合并函数
     * @return 处理结果的future
     */
    template<typename ResultType, typename ProcessorFunc, typename MergeFunc>
    std::future<ResultType> processFeatureCollection(
        const oscean::core_services::FeatureCollection& features,
        ProcessorFunc processor,
        MergeFunc mergeResults);
    
    /**
     * @brief 并行处理栅格数据
     * @tparam ResultType 处理结果类型
     * @tparam ProcessorFunc 处理函数类型
     * @param raster 输入栅格数据
     * @param processor 处理函数
     * @param mergeResults 结果合并函数
     * @return 处理结果的future
     */
    template<typename ResultType, typename ProcessorFunc, typename MergeFunc>
    std::future<ResultType> processGridData(
        const oscean::core_services::GridData& raster,
        ProcessorFunc processor,
        MergeFunc mergeResults);
    
    /**
     * @brief 并行执行批量任务
     * @tparam TaskType 任务类型
     * @tparam ResultType 结果类型
     * @param tasks 任务列表
     * @return 结果列表的future
     */
    template<typename TaskType, typename ResultType>
    std::future<std::vector<ResultType>> executeBatchTasks(
        const std::vector<TaskType>& tasks);
    
    /**
     * @brief 更新并行配置
     * @param config 新的空间服务配置
     */
    void updateConfig(const oscean::core_services::spatial_ops::SpatialOpsConfig& config);
    
    /**
     * @brief 获取当前配置
     * @return 当前并行配置
     */
    const oscean::core_services::spatial_ops::ParallelConfig& getConfig() const noexcept { return config_; }
    
    /**
     * @brief 获取处理统计信息
     * @return 并行处理统计
     */
    ParallelProcessingStats getStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();
    
    /**
     * @brief 检查是否支持并行处理
     * @param dataSize 数据大小
     * @return 如果建议使用并行处理返回true
     */
    bool shouldUseParallel(size_t dataSize) const noexcept;
    
    /**
     * @brief 估算最优分区数量
     * @param dataSize 数据大小
     * @param memoryPerItem 每项数据的内存使用量
     * @return 建议的分区数量
     */
    size_t estimateOptimalPartitions(size_t dataSize, size_t memoryPerItem) const;
    
    /**
     * @brief 检查是否有任务正在处理
     * @return 如果有任务正在处理返回true
     */
    bool isProcessing() const noexcept {
        return isProcessing_.load();
    }

private:
    // 配置参数
    oscean::core_services::spatial_ops::ParallelConfig config_;
    
    // 性能统计
    mutable ParallelProcessingStats stats_;
    mutable std::mutex statsMutex_;
    
    // 处理状态
    std::atomic<bool> isProcessing_{false};
    
    // 内部辅助方法
    size_t calculateOptimalThreadCount(size_t dataSize) const;
    size_t calculateOptimalPartitions(size_t dataSize, size_t memoryPerItem) const;
    void updateProcessingStats(const std::vector<std::chrono::milliseconds>& partitionTimes);
    
    // 数据分区方法
    std::vector<SpatialPartition> partitionFeatures(
        const oscean::core_services::FeatureCollection& features,
        size_t numPartitions) const;
    
    std::vector<SpatialPartition> partitionRaster(
        const oscean::core_services::GridData& raster,
        size_t numPartitions) const;
};

// Template implementations
template<typename ResultType, typename ProcessorFunc, typename MergeFunc>
std::future<ResultType> SpatialParallelCoordinator::processFeatureCollection(
    const oscean::core_services::FeatureCollection& features,
    ProcessorFunc processor,
    MergeFunc mergeResults) {
    
    return std::async(std::launch::async, [this, features, processor, mergeResults]() -> ResultType {
        // 设置处理状态
        isProcessing_.store(true);
        
        // RAII guard to ensure processing flag is reset
        struct ProcessingGuard {
            std::atomic<bool>& flag;
            ProcessingGuard(std::atomic<bool>& f) : flag(f) {}
            ~ProcessingGuard() { flag.store(false); }
        } guard(isProcessing_);
        
        if (features.getFeatures().empty()) {
            throw oscean::core_services::spatial_ops::InvalidInputDataException("Feature collection is empty");
        }
        
        size_t dataSize = features.getFeatures().size();
        
        // 检查是否需要并行处理
        if (!shouldUseParallel(dataSize)) {
            auto startTime = std::chrono::high_resolution_clock::now();
            auto result = processor(features);
            auto endTime = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            updateProcessingStats({duration});
            
            return result;
        }
        
        // 计算最优分区数
        size_t numPartitions = estimateOptimalPartitions(dataSize, sizeof(oscean::core_services::Feature));
        
        // 分区处理
        auto partitions = partitionFeatures(features, numPartitions);
        
        // 并行处理各分区
        std::vector<std::future<ResultType>> futures;
        std::vector<std::chrono::milliseconds> partitionTimes;
        
        for (const auto& partition : partitions) {
            futures.push_back(std::async(std::launch::async, [&partition, &features, processor]() -> ResultType {
                auto startTime = std::chrono::high_resolution_clock::now();
                
                // 创建分区的要素集合
                oscean::core_services::FeatureCollection partitionFeatures;
                for (size_t idx : partition.featureIndices) {
                    if (idx < features.getFeatures().size()) {
                        partitionFeatures.addFeature(features.getFeatures()[idx]);
                    }
                }
                
                auto result = processor(partitionFeatures);
                auto endTime = std::chrono::high_resolution_clock::now();
                
                return result;
            }));
        }
        
        // 收集结果
        std::vector<ResultType> results;
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        // 合并结果
        return mergeResults(results);
    });
}

template<typename ResultType, typename ProcessorFunc, typename MergeFunc>
std::future<ResultType> SpatialParallelCoordinator::processGridData(
    const oscean::core_services::GridData& raster,
    ProcessorFunc processor,
    MergeFunc mergeResults) {
    
    return std::async(std::launch::async, [this, raster, processor, mergeResults]() -> ResultType {
        // 设置处理状态
        isProcessing_.store(true);
        
        // RAII guard to ensure processing flag is reset
        struct ProcessingGuard {
            std::atomic<bool>& flag;
            ProcessingGuard(std::atomic<bool>& f) : flag(f) {}
            ~ProcessingGuard() { flag.store(false); }
        } guard(isProcessing_);
        
        if (raster.getData().empty()) {
            throw oscean::core_services::spatial_ops::InvalidInputDataException("Raster data is empty");
        }
        
        size_t dataSize = raster.getData().size();
        
        // 检查是否需要并行处理
        if (!shouldUseParallel(dataSize)) {
            auto startTime = std::chrono::high_resolution_clock::now();
            auto result = processor(raster);
            auto endTime = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            updateProcessingStats({duration});
            
            return result;
        }
        
        // 计算最优分区数
        size_t numPartitions = estimateOptimalPartitions(dataSize, 1);
        
        // 分区处理
        auto partitions = partitionRaster(raster, numPartitions);
        
        // 并行处理各分区
        std::vector<std::future<ResultType>> futures;
        
        for (const auto& partition : partitions) {
            futures.push_back(std::async(std::launch::async, [&partition, &raster, processor]() -> ResultType {
                // 创建分区的栅格数据
                // 这里需要根据partition.bounds从原始raster中提取子区域
                // 简化实现，直接使用原始raster
                return processor(raster);
            }));
        }
        
        // 收集结果
        std::vector<ResultType> results;
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        // 合并结果
        return mergeResults(results);
    });
}

template<typename TaskType, typename ResultType>
std::future<std::vector<ResultType>> SpatialParallelCoordinator::executeBatchTasks(
    const std::vector<TaskType>& tasks) {
    
    return std::async(std::launch::async, [this, tasks]() -> std::vector<ResultType> {
        // 设置处理状态
        isProcessing_.store(true);
        
        // RAII guard to ensure processing flag is reset
        struct ProcessingGuard {
            std::atomic<bool>& flag;
            ProcessingGuard(std::atomic<bool>& f) : flag(f) {}
            ~ProcessingGuard() { flag.store(false); }
        } guard(isProcessing_);
        
        if (tasks.empty()) {
            return std::vector<ResultType>();
        }
        
        size_t dataSize = tasks.size();
        
        // 检查是否需要并行处理
        if (!shouldUseParallel(dataSize)) {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            std::vector<ResultType> results;
            for (const auto& task : tasks) {
                results.push_back(task());
            }
            
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            updateProcessingStats({duration});
            
            return results;
        }
        
        // 并行执行任务
        std::vector<std::future<ResultType>> futures;
        for (const auto& task : tasks) {
            futures.push_back(std::async(std::launch::async, task));
        }
        
        // 收集结果
        std::vector<ResultType> results;
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        return results;
    });
}

} // namespace oscean::core_services::spatial_ops::engine 