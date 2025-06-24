/**
 * @file spatial_parallel_coordinator.cpp
 * @brief SpatialParallelCoordinator class implementation
 */

#include "spatial_parallel_coordinator.h"
#include "core_services/spatial_ops/spatial_exceptions.h"

#include <algorithm>
#include <cmath>

namespace oscean::core_services::spatial_ops::engine {

SpatialParallelCoordinator::SpatialParallelCoordinator(const oscean::core_services::spatial_ops::SpatialOpsConfig& config)
    : config_(config.parallelSettings) {
}

SpatialParallelCoordinator::~SpatialParallelCoordinator() = default;

void SpatialParallelCoordinator::updateConfig(const oscean::core_services::spatial_ops::SpatialOpsConfig& config) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    config_ = config.parallelSettings;
}

ParallelProcessingStats SpatialParallelCoordinator::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}

void SpatialParallelCoordinator::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_ = ParallelProcessingStats{};
}

bool SpatialParallelCoordinator::shouldUseParallel(size_t dataSize) const noexcept {
    return dataSize >= config_.minDataSizeForParallelism && config_.maxThreads > 1;
}

size_t SpatialParallelCoordinator::estimateOptimalPartitions(size_t dataSize, size_t memoryPerItem) const {
    if (!shouldUseParallel(dataSize)) {
        return 1;
    }
    
    // 基于数据大小和可用线程数计算最优分区数
    size_t maxPartitions = std::min(config_.maxThreads, dataSize);
    size_t minItemsPerPartition = config_.minDataSizeForParallelism / 4; // 每个分区至少处理一定数量的数据
    
    size_t optimalPartitions = std::max(static_cast<size_t>(1), std::min(maxPartitions, dataSize / minItemsPerPartition));
    
    return optimalPartitions;
}

size_t SpatialParallelCoordinator::calculateOptimalThreadCount(size_t dataSize) const {
    if (!shouldUseParallel(dataSize)) {
        return 1;
    }
    
    size_t hardwareConcurrency = std::thread::hardware_concurrency();
    if (hardwareConcurrency == 0) {
        hardwareConcurrency = 4; // 默认值
    }
    
    return std::min({config_.maxThreads, hardwareConcurrency, dataSize});
}

size_t SpatialParallelCoordinator::calculateOptimalPartitions(size_t dataSize, size_t memoryPerItem) const {
    return estimateOptimalPartitions(dataSize, memoryPerItem);
}

void SpatialParallelCoordinator::updateProcessingStats(const std::vector<std::chrono::milliseconds>& partitionTimes) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    stats_.totalPartitions += partitionTimes.size();
    stats_.successfulPartitions += partitionTimes.size(); // 假设所有分区都成功
    
    for (const auto& time : partitionTimes) {
        stats_.updateStats(time);
    }
    
    // 计算并行效率（简化计算）
    if (stats_.totalPartitions > 1) {
        double serialTime = static_cast<double>(stats_.totalProcessingTime.count());
        double parallelTime = static_cast<double>(stats_.maxPartitionTime.count());
        if (parallelTime > 0) {
            stats_.parallelEfficiency = std::min(1.0, serialTime / (parallelTime * stats_.totalPartitions));
        }
    }
}

std::vector<SpatialPartition> SpatialParallelCoordinator::partitionFeatures(
    const oscean::core_services::FeatureCollection& features,
    size_t numPartitions) const {
    
    std::vector<SpatialPartition> partitions;
    
    if (features.getFeatures().empty() || numPartitions == 0) {
        return partitions;
    }
    
    size_t featuresPerPartition = features.getFeatures().size() / numPartitions;
    size_t remainder = features.getFeatures().size() % numPartitions;
    
    size_t currentIndex = 0;
    for (size_t i = 0; i < numPartitions; ++i) {
        SpatialPartition partition;
        partition.partitionId = i;
        partition.totalPartitions = numPartitions;
        
        // 计算当前分区的要素数量
        size_t currentPartitionSize = featuresPerPartition + (i < remainder ? 1 : 0);
        
        // 添加要素索引
        for (size_t j = 0; j < currentPartitionSize && currentIndex < features.getFeatures().size(); ++j) {
            partition.featureIndices.push_back(currentIndex++);
        }
        
        // 简化的边界框计算（实际应该基于要素的几何边界）
        partition.bounds = oscean::core_services::BoundingBox(-180.0, -90.0, 180.0, 90.0);
        partition.estimatedMemoryUsage = currentPartitionSize * sizeof(oscean::core_services::Feature);
        
        partitions.push_back(std::move(partition));
    }
    
    return partitions;
}

std::vector<SpatialPartition> SpatialParallelCoordinator::partitionRaster(
    const oscean::core_services::GridData& raster,
    size_t numPartitions) const {
    
    std::vector<SpatialPartition> partitions;
    
    if (raster.getData().empty() || numPartitions == 0) {
        return partitions;
    }
    
    // 简化实现：基于行进行分区
    size_t rowsPerPartition = raster.getDefinition().rows / numPartitions;
    size_t remainder = raster.getDefinition().rows % numPartitions;
    
    double cellHeight = (raster.getDefinition().extent.maxY - raster.getDefinition().extent.minY) / raster.getDefinition().rows;
    double cellWidth = (raster.getDefinition().extent.maxX - raster.getDefinition().extent.minX) / raster.getDefinition().cols;
    
    size_t currentRow = 0;
    for (size_t i = 0; i < numPartitions; ++i) {
        SpatialPartition partition;
        partition.partitionId = i;
        partition.totalPartitions = numPartitions;
        
        // 计算当前分区的行数
        size_t currentPartitionRows = rowsPerPartition + (i < remainder ? 1 : 0);
        
        // 计算分区边界
        double minY = raster.getDefinition().extent.maxY - (currentRow + currentPartitionRows) * cellHeight;
        double maxY = raster.getDefinition().extent.maxY - currentRow * cellHeight;
        
        partition.bounds = oscean::core_services::BoundingBox(
            raster.getDefinition().extent.minX, minY, raster.getDefinition().extent.maxX, maxY
        );
        
        partition.estimatedMemoryUsage = currentPartitionRows * raster.getDefinition().cols * sizeof(double);
        
        currentRow += currentPartitionRows;
        partitions.push_back(std::move(partition));
    }
    
    return partitions;
}

} // namespace oscean::core_services::spatial_ops::engine 