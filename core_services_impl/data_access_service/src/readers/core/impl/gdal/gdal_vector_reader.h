#pragma once

/**
 * @file gdal_vector_reader.h
 * @brief GDAL矢量数据读取器 - 专门处理矢量数据
 * 
 * 🎯 职责:
 * ✅ 继承UnifiedDataReader，专门处理矢量数据
 * ✅ 集成common_utilities高级功能（SIMD、内存、异步、缓存）
 * ✅ 创建并持有专用矢量处理器
 * ✅ 将具体的矢量数据请求委托给Processor
 */

#include "readers/core/unified_data_reader.h"
#include "core_services/common_data_types.h"
#include "gdal_common_types.h"
#include "gdal_vector_processor.h"

// Common Utilities高级功能
#include "common_utils/simd/simd_manager_unified.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/async/async_framework.h"
#include "common_utils/cache/icache_manager.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/utilities/logging_utils.h"

#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <boost/thread/future.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>

namespace oscean::core_services::data_access::readers::impl::gdal {

/**
 * @brief GDAL矢量读取器 - 专门处理矢量数据
 */
class GdalVectorReader final : public UnifiedDataReader {
public:
    /**
     * @brief 构造函数
     * @param filePath 矢量文件路径
     * @param commonServices Common服务工厂（可选）
     */
    explicit GdalVectorReader(
        const std::string& filePath,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~GdalVectorReader() override;

    GdalVectorReader(const GdalVectorReader&) = delete;
    GdalVectorReader& operator=(const GdalVectorReader&) = delete;
    GdalVectorReader(GdalVectorReader&&) = delete;
    GdalVectorReader& operator=(GdalVectorReader&&) = delete;

    // =========================================================================
    // UnifiedDataReader 接口实现
    // =========================================================================
    
    boost::future<bool> openAsync() override;
    boost::future<void> closeAsync() override;
    std::string getReaderType() const override;
    
    boost::future<std::optional<FileMetadata>> getFileMetadataAsync() override;
    boost::future<std::vector<std::string>> getVariableNamesAsync() override;
    
    boost::future<std::shared_ptr<GridData>> readGridDataAsync(
        const std::string& variableName,
        const std::optional<BoundingBox>& bounds = std::nullopt) override;

    // =============================================================================
    // 矢量数据特定接口
    // =============================================================================
    
    /**
     * @brief 获取图层名称列表
     */
    boost::future<std::vector<std::string>> getLayerNamesAsync();
    
    /**
     * @brief 读取图层数据
     */
    boost::future<std::shared_ptr<GridData>> readLayerDataAsync(
        const std::string& layerName,
        const std::optional<BoundingBox>& bounds = std::nullopt);
    
    /**
     * @brief 获取图层要素数量
     */
    boost::future<size_t> getFeatureCountAsync(const std::string& layerName);
    
    /**
     * @brief 获取图层几何类型
     */
    boost::future<std::string> getGeometryTypeAsync(const std::string& layerName);
    
    /**
     * @brief 获取图层字段信息
     */
    boost::future<std::vector<std::map<std::string, std::string>>> getFieldInfoAsync(const std::string& layerName);
    
    /**
     * @brief 空间查询 - 返回与边界框相交的要素
     */
    boost::future<std::shared_ptr<GridData>> spatialQueryAsync(
        const std::string& layerName,
        const BoundingBox& bounds,
        const std::string& spatialRelation = "INTERSECTS");
    
    /**
     * @brief 属性查询 - 基于属性条件查询要素
     */
    boost::future<std::shared_ptr<GridData>> attributeQueryAsync(
        const std::string& layerName,
        const std::string& whereClause);

    // =============================================================================
    // 高级功能接口
    // =============================================================================
    
    /**
     * @brief 启用/禁用SIMD优化
     */
    void enableSIMDOptimization(bool enable = true);
    bool isSIMDOptimizationEnabled() const;
    
    /**
     * @brief 配置SIMD优化参数
     */
    void configureSIMDOptimization(const GdalSIMDConfig& config);
    
    /**
     * @brief 启用/禁用高级缓存
     */
    void enableAdvancedCaching(bool enable = true);
    bool isAdvancedCachingEnabled() const;
    
    /**
     * @brief 启用/禁用性能监控
     */
    void enablePerformanceMonitoring(bool enable = true);
    bool isPerformanceMonitoringEnabled() const;
    
    /**
     * @brief 获取性能统计信息
     */
    GdalPerformanceStats getPerformanceStats() const;
    
    /**
     * @brief 获取性能报告
     */
    std::string getPerformanceReport() const;
    
    /**
     * @brief 流式读取要素数据
     */
    boost::future<void> streamFeaturesAsync(
        const std::string& layerName,
        const std::optional<BoundingBox>& bounds,
        std::function<bool(const std::vector<std::map<std::string, std::string>>&)> processor);

private:
    // =============================================================================
    // 成员变量
    // =============================================================================
    
    std::string filePath_;
    GDALDataset* gdalDataset_ = nullptr;
    std::unique_ptr<GdalVectorProcessor> vectorProcessor_;
    std::atomic<bool> isOpen_{false};
    
    // Common Utilities组件
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices_;
    std::shared_ptr<oscean::common_utils::simd::UnifiedSIMDManager> simdManager_;
    std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager_;
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework_;
    std::shared_ptr<oscean::common_utils::cache::ICacheManager<std::string, std::vector<unsigned char>>> cacheManager_;
    
    // 配置和状态
    GdalSIMDConfig simdConfig_;
    std::atomic<bool> simdEnabled_{false};
    std::atomic<bool> cachingEnabled_{false};
    std::atomic<bool> performanceMonitoringEnabled_{false};
    
    // 性能监控
    mutable GdalPerformanceStats performanceStats_;
    
    // 缓存
    mutable std::unordered_map<std::string, std::vector<std::string>> layerFieldsCache_;
    mutable std::unordered_map<std::string, size_t> featureCountCache_;
    mutable std::unordered_map<std::string, std::string> geometryTypeCache_;
    
    // =============================================================================
    // 内部方法
    // =============================================================================
    
    /**
     * @brief 初始化高级功能组件
     */
    void initializeAdvancedComponents();
    
    /**
     * @brief 初始化GDAL环境
     */
    bool initializeGDAL();
    
    /**
     * @brief 清理资源
     */
    void cleanup();
    
    /**
     * @brief 验证文件是否为有效的矢量文件
     */
    bool validateVectorFile() const;
    
    /**
     * @brief 更新性能统计
     */
    void updatePerformanceStats(size_t bytesRead, bool simdUsed = false, bool cacheHit = false) const;
    
    /**
     * @brief 检查内存使用情况
     */
    bool checkMemoryUsage() const;
    
    /**
     * @brief 计算缓存键
     */
    std::string calculateCacheKey(const std::string& layerName, 
                                  const std::optional<BoundingBox>& bounds = std::nullopt) const;
    
    /**
     * @brief 从缓存获取数据
     */
    std::optional<GridData> getFromCache(const std::string& cacheKey) const;
    
    /**
     * @brief 将数据存入缓存
     */
    void putToCache(const std::string& cacheKey, const GridData& data) const;
};

} // namespace oscean::core_services::data_access::readers::impl::gdal 