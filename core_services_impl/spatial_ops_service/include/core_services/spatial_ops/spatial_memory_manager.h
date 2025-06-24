#pragma once

#include "common_utils/memory_manager.h"
#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/spatial_ops/spatial_exceptions.h"

namespace oscean::core_services::spatial_ops {

/**
 * @brief 空间操作内存管理适配器
 * 
 * 作为common_utils内存管理器的专业化适配器，提供空间操作特有的：
 * - 栅格数据内存分配优化
 * - 要素集合内存管理
 * - 空间索引内存优化
 * - 几何运算临时内存管理
 * 
 * 注意：不继承BaseMemoryManager，而是组合使用common_utils的内存管理器
 */
class SpatialMemoryAdapter {
public:
    explicit SpatialMemoryAdapter(std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager);
    virtual ~SpatialMemoryAdapter();

    /**
     * @brief 为栅格数据分配优化的内存
     * @param gridDef 网格定义
     * @param dataType 数据类型大小
     * @param tag 标签
     * @return 分配的内存指针
     */
    void* allocateForRaster(const GridDefinition& gridDef, 
                           size_t dataType = sizeof(float), 
                           const std::string& tag = "raster");

    /**
     * @brief 为要素集合分配内存
     * @param estimatedFeatureCount 估计的要素数量
     * @param averageFeatureSize 平均要素大小
     * @param tag 标签
     * @return 分配的内存指针
     */
    void* allocateForFeatures(size_t estimatedFeatureCount, 
                             size_t averageFeatureSize = 1024, 
                             const std::string& tag = "features");

    /**
     * @brief 为空间索引分配内存
     * @param indexType 索引类型
     * @param estimatedNodeCount 估计的节点数量
     * @param tag 标签
     * @return 分配的内存指针
     */
    void* allocateForSpatialIndex(const std::string& indexType,
                                 size_t estimatedNodeCount,
                                 const std::string& tag = "spatial_index");

    /**
     * @brief 为几何运算分配临时内存
     * @param operationType 运算类型
     * @param estimatedSize 估计大小
     * @param tag 标签
     * @return 分配的内存指针
     */
    void* allocateForGeometryOperation(const std::string& operationType,
                                      size_t estimatedSize,
                                      const std::string& tag = "geometry_op");

    /**
     * @brief 释放内存
     * @param ptr 要释放的内存指针
     */
    void deallocate(void* ptr);

    /**
     * @brief 获取空间操作特有的内存统计
     * @return 扩展的内存统计信息
     */
    struct SpatialMemoryStats {
        oscean::common_utils::memory::MemoryUsageStats baseStats;
        size_t rasterMemoryUsed = 0;
        size_t featureMemoryUsed = 0;
        size_t indexMemoryUsed = 0;
        size_t geometryOpMemoryUsed = 0;
        size_t temporaryMemoryUsed = 0;
    };
    
    SpatialMemoryStats getSpatialStats() const;

    /**
     * @brief 设置空间操作内存使用阈值
     * @param rasterThreshold 栅格内存阈值
     * @param featureThreshold 要素内存阈值
     * @param indexThreshold 索引内存阈值
     */
    void setSpatialThresholds(size_t rasterThreshold,
                             size_t featureThreshold,
                             size_t indexThreshold);

    /**
     * @brief 获取底层内存管理器
     * @return 内存管理器指针
     */
    std::shared_ptr<oscean::common_utils::memory::IMemoryManager> getMemoryManager() const {
        return memoryManager_;
    }

private:
    // 组合使用common_utils的内存管理器
    std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager_;
    
    // 空间操作特有的统计信息
    mutable std::mutex spatialStatsMutex_;
    SpatialMemoryStats spatialStats_;
    
    // 内存使用阈值
    size_t rasterMemoryThreshold_;
    size_t featureMemoryThreshold_;
    size_t indexMemoryThreshold_;
    
    // 内部辅助方法
    size_t calculateRasterSize(const GridDefinition& gridDef, size_t dataType) const;
    size_t calculateFeatureCollectionSize(size_t featureCount, size_t averageSize) const;
    size_t calculateIndexSize(const std::string& indexType, size_t nodeCount) const;
    
    void updateSpatialStats(const std::string& allocationType, size_t size, bool isAllocation);
    bool checkSpatialThresholds(const std::string& allocationType, size_t size) const;
};

/**
 * @brief 空间内存适配器工厂
 */
class SpatialMemoryAdapterFactory {
public:
    /**
     * @brief 创建空间内存适配器
     * @param memoryManager 底层内存管理器
     * @return 空间内存适配器实例
     */
    static std::shared_ptr<SpatialMemoryAdapter> create(
        std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager);

    /**
     * @brief 从配置创建空间内存适配器
     * @param config 内存池配置
     * @return 空间内存适配器实例
     */
    static std::shared_ptr<SpatialMemoryAdapter> createFromConfig(
        const oscean::common_utils::memory::MemoryPoolConfig& config);

    /**
     * @brief 创建默认配置的空间内存适配器
     * @return 空间内存适配器实例
     */
    static std::shared_ptr<SpatialMemoryAdapter> createDefault();
};

} // namespace oscean::core_services::spatial_ops 