#pragma once

/**
 * @file gdal_raster_reader.h
 * @brief GDAL栅格数据读取器 - 完整统一架构实现
 * 
 * 🎯 职责:
 * ✅ 继承UnifiedDataReader，管理文件生命周期和高级功能
 * ✅ 集成common_utilities高级功能（SIMD、内存、异步、缓存）
 * ✅ 创建并持有专用处理器
 * ✅ 将所有具体的数据请求委托给Processor
 */

#include "readers/core/unified_data_reader.h"
#include "core_services/common_data_types.h"
#include "gdal_common_types.h"      // 使用通用类型定义中的结构体定义
#include "gdal_raster_processor.h"

// Common Utilities高级功能
#include "common_utils/simd/simd_manager_unified.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/async/async_framework.h"
#include "common_utils/cache/icache_manager.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/utilities/logging_utils.h"

#include <gdal_priv.h>  // 添加GDAL头文件以定义GDALDataType
#include <boost/thread/future.hpp>
#include <limits>
#include <optional>
#include <atomic>
#include <chrono>
#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <unordered_map>

// Forward declarations for GDAL types
class GDALRasterBand;
class GDALDataset;

namespace oscean::core_services::data_access::readers::impl::gdal {

/**
 * @brief GDAL栅格读取器 - 完全符合UnifiedDataReader接口且集成高级功能
 */
class GdalRasterReader final : public UnifiedDataReader {
public:
    /**
     * @brief 构造函数
     * @param filePath 栅格文件路径
     * @param commonServices Common服务工厂（可选）
     */
    explicit GdalRasterReader(
        const std::string& filePath,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~GdalRasterReader() override;

    GdalRasterReader(const GdalRasterReader&) = delete;
    GdalRasterReader& operator=(const GdalRasterReader&) = delete;
    GdalRasterReader(GdalRasterReader&&) = delete;
    GdalRasterReader& operator=(GdalRasterReader&&) = delete;

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

private:
    // =============================================================================
    // 成员变量
    // =============================================================================
    
    std::string filePath_;
    GDALDataset* gdalDataset_ = nullptr;
    std::unique_ptr<GdalRasterProcessor> rasterProcessor_;
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
     * @brief 更新性能统计
     */
    void updatePerformanceStats(size_t bytesRead, bool simdUsed = false, bool cacheHit = false) const;
    
    /**
     * @brief 检查内存使用情况
     */
    bool checkMemoryUsage() const;
    
    /**
     * @brief 优化读取参数
     */
    void optimizeReadParameters(size_t& blockXSize, size_t& blockYSize, int& bufferType) const;
};

} // namespace oscean::core_services::data_access::readers::impl::gdal 