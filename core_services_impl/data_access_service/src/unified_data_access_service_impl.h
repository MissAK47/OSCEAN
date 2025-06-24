/**
 * @file unified_data_access_service_impl.h
 * @brief 统一数据访问服务实现 - 彻底DI重构版本
 * 
 * 🎯 重构目标：
 * ✅ 移除复杂的抽象工厂 - 直接使用CommonServicesFactory
 * ✅ 简化依赖注入 - 只保留必要的DI组件
 * ✅ 移除过度设计 - 删除不必要的抽象层
 * ✅ 性能优化 - 读取器池化、缓存优化
 * ✅ 延迟初始化 - 避免构造函数阻塞
 * 🆕 真正流式处理 - 实现大文件分块读取
 * 🆕 3D数据支持 - 实现垂直剖面和时间序列读取
 * 🆕 点查询功能 - 实现精确点数据查询
 */

#pragma once

#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>

// 项目接口
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/data_access/unified_data_types.h"
#include "core_services/data_access/i_data_access_service_factory.h"
#include "core_services/common_data_types.h"

// Common基础设施 - 修复接口引用
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/simd/isimd_manager.h"
#include "common_utils/memory/memory_interfaces.h"
#include "common_utils/async/async_framework.h"
#include "common_utils/utilities/file_format_detector.h"

// 数据读取器
#include "readers/core/reader_registry.h"

namespace oscean::core_services {

// 🔧 前向声明，解决循环依赖
class GdalInitializationManager;
class FileAccessLockManager;
class ReaderPoolManager;

// 前向声明
namespace data_access::readers::impl::netcdf {
    class NetCDFAdvancedReader;
}

/**
 * @brief 🆕 高级流式数据处理器实现
 */
class AdvancedStreamProcessor : public data_access::IStreamProcessor {
public:
    explicit AdvancedStreamProcessor(
        std::function<bool(const std::vector<double>&, const std::map<std::string, std::any>&)> processor,
        std::function<void()> onComplete = nullptr,
        std::function<void(const std::string&)> onError = nullptr);

    bool processChunk(
        const std::vector<double>& chunk, 
        const std::map<std::string, std::any>& chunkInfo) override;
        
    void onStreamComplete() override;
    void onStreamError(const std::string& error) override;

private:
    std::function<bool(const std::vector<double>&, const std::map<std::string, std::any>&)> processor_;
    std::function<void()> onComplete_;
    std::function<void(const std::string&)> onError_;
};

/**
 * @brief 统一数据访问服务实现 - DI重构版本
 */
class UnifiedDataAccessServiceImpl : public data_access::IUnifiedDataAccessService {
public:
    /**
     * @brief 构造函数 - 支持完全依赖注入
     * @param servicesFactory Common服务工厂
     * @param config 数据访问配置
     */
    explicit UnifiedDataAccessServiceImpl(
        std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> servicesFactory,
        const data_access::api::DataAccessConfiguration& config = data_access::api::DataAccessConfiguration{});

    /**
     * @brief 析构函数
     */
    ~UnifiedDataAccessServiceImpl() override;

    // 禁用拷贝和移动
    UnifiedDataAccessServiceImpl(const UnifiedDataAccessServiceImpl&) = delete;
    UnifiedDataAccessServiceImpl& operator=(const UnifiedDataAccessServiceImpl&) = delete;
    UnifiedDataAccessServiceImpl(UnifiedDataAccessServiceImpl&&) = delete;
    UnifiedDataAccessServiceImpl& operator=(UnifiedDataAccessServiceImpl&&) = delete;

    // ===============================================================================
    // IUnifiedDataAccessService 接口实现
    // ===============================================================================

    boost::future<data_access::api::UnifiedDataResponse> processDataRequestAsync(
        const data_access::api::UnifiedDataRequest& request) override;

    boost::future<std::vector<data_access::api::UnifiedDataResponse>> processBatchRequestsAsync(
        const std::vector<data_access::api::UnifiedDataRequest>& requests) override;

    boost::future<std::optional<oscean::core_services::FileMetadata>> getFileMetadataAsync(
        const std::string& filePath) override;

    boost::future<std::vector<oscean::core_services::FileMetadata>> extractBatchMetadataAsync(
        const std::vector<std::string>& filePaths,
        size_t maxConcurrency = 4) override;

    boost::future<std::shared_ptr<oscean::core_services::GridData>> readGridDataAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds = std::nullopt) override;

    boost::future<bool> checkVariableExistsAsync(
        const std::string& filePath,
        const std::string& variableName) override;

    boost::future<std::vector<std::string>> getVariableNamesAsync(
        const std::string& filePath) override;

    boost::future<std::shared_ptr<oscean::core_services::GridData>> readGridDataWithCRSAsync(
        const std::string& filePath,
        const std::string& variableName,
        const oscean::core_services::BoundingBox& bounds,
        const std::string& targetCRS) override;

    boost::future<std::optional<double>> readPointDataWithCRSAsync(
        const std::string& filePath,
        const std::string& variableName,
        const oscean::core_services::Point& point,
        const std::string& targetCRS) override;

    // =============================================================================
    // 🆕 3D数据和垂直剖面支持
    // =============================================================================

    boost::future<std::shared_ptr<oscean::core_services::VerticalProfileData>> readVerticalProfileAsync(
        const std::string& filePath,
        const std::string& variableName,
        double longitude,
        double latitude,
        const std::optional<std::chrono::system_clock::time_point>& timePoint = std::nullopt) override;

    boost::future<std::shared_ptr<oscean::core_services::TimeSeriesData>> readTimeSeriesAsync(
        const std::string& filePath,
        const std::string& variableName,
        double longitude,
        double latitude,
        const std::optional<double>& depth = std::nullopt,
        const std::optional<std::pair<std::chrono::system_clock::time_point,
                                      std::chrono::system_clock::time_point>>& timeRange = std::nullopt) override;

    boost::future<std::optional<double>> readPointValueAsync(
        const std::string& filePath,
        const std::string& variableName,
        double longitude,
        double latitude,
        const std::optional<double>& depth = std::nullopt,
        const std::optional<std::chrono::system_clock::time_point>& timePoint = std::nullopt) override;

    boost::future<std::vector<double>> getVerticalLevelsAsync(
        const std::string& filePath) override;

    // =============================================================================
    // 🆕 真正的流式处理 - 大文件优化
    // =============================================================================

    boost::future<void> startAdvancedStreamingAsync(
        const std::string& filePath,
        const std::string& variableName,
        std::shared_ptr<data_access::IStreamProcessor> processor,
        const data_access::LargeFileReadConfig& config = data_access::LargeFileReadConfig{}) override;

    boost::future<void> streamBoundedDataAsync(
        const std::string& filePath,
        const std::string& variableName,
        const oscean::core_services::BoundingBox& bounds,
        std::function<bool(const std::vector<double>&, const std::vector<size_t>&)> chunkProcessor,
        std::function<void(double)> progressCallback = nullptr) override;

    boost::future<std::shared_ptr<oscean::core_services::GridData>> readLargeFileOptimizedAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds = std::nullopt,
        const data_access::LargeFileReadConfig& config = data_access::LargeFileReadConfig{}) override;

    // =============================================================================
    // 传统方法保持兼容性
    // =============================================================================

    boost::future<void> startStreamingAsync(
        const std::string& filePath,
        const std::string& variableName,
        std::function<bool(const std::vector<double>&)> chunkProcessor) override;

    data_access::api::DataAccessMetrics getPerformanceMetrics() const override;

    void configurePerformanceTargets(const data_access::api::DataAccessPerformanceTargets& targets) override;

    void clearCache() override;

    bool isHealthy() const override;

    /**
     * @brief 确保服务已初始化（线程安全）
     *        这是外部模块（如DataManagementService）在调用具体功能前需要调用的方法。
     */
    void ensureInitialized() const;

    // === 🔧 新增：并发优化组件依赖注入支持 ===
    
    /**
     * @brief GDAL初始化管理器 - 依赖注入版本
     */
    class GdalInitializationManager {
    public:
        virtual ~GdalInitializationManager() = default;
        virtual bool warmupInitialization() = 0;
        virtual bool isWarmedUp() const = 0;
        virtual std::shared_ptr<oscean::common_utils::simd::ISIMDManager> getSIMDManager() const = 0;
        virtual std::shared_ptr<oscean::common_utils::memory::IMemoryManager> getMemoryManager() const = 0;
        virtual std::shared_ptr<oscean::common_utils::async::AsyncFramework> getAsyncFramework() const = 0;
    };
    
    /**
     * @brief 文件访问锁管理器 - 依赖注入版本
     */
    class FileAccessLockManager {
    public:
        virtual ~FileAccessLockManager() = default;
        
        class FileAccessGuard {
        public:
            virtual ~FileAccessGuard() = default;
        };
        
        virtual std::unique_ptr<FileAccessGuard> createFileGuard(const std::string& filePath) = 0;
        
        struct FileAccessStats {
            size_t activeReaders = 0;
            size_t queuedRequests = 0;
            std::chrono::system_clock::time_point lastAccess;
        };
        
        virtual FileAccessStats getFileAccessStats(const std::string& filePath) const = 0;
    };
    
    /**
     * @brief 读取器池管理器 - 依赖注入版本
     */
    class ReaderPoolManager {
    public:
        struct PoolConfiguration {
            size_t initialPoolSize = 4;
            size_t maxPoolSize = 16;
            size_t growthIncrement = 2;
            std::chrono::seconds readerTTL{300};
            bool enablePooling = true;
        };
        
        virtual ~ReaderPoolManager() = default;
        virtual bool initializePool(
            const PoolConfiguration& config,
            std::shared_ptr<data_access::readers::ReaderRegistry> readerRegistry,
            std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices) = 0;
        virtual std::shared_ptr<data_access::readers::UnifiedDataReader> getOrCreateReader(
            const std::string& filePath, const std::string& readerType) = 0;
        virtual void returnReaderToPool(std::shared_ptr<data_access::readers::UnifiedDataReader> reader) = 0;
    };

private:
    // ===============================================================================
    // 内部辅助方法
    // ===============================================================================
    
    void initializeServices();
    void initializeInternal();  // 🔧 线程安全的初始化方法
    void shutdown();
    
    std::shared_ptr<data_access::readers::UnifiedDataReader> getReaderForFile(const std::string& filePath);
    bool validateFilePath(const std::string& filePath) const;
    std::string detectFileFormat(const std::string& filePath) const;
    
    template<typename T>
    boost::future<T> createAsyncTask(std::function<T()> task) const;

    // =============================================================================
    // 🆕 NetCDF Advanced Reader支持
    // =============================================================================
    
    std::shared_ptr<data_access::readers::impl::netcdf::NetCDFAdvancedReader> 
    getNetCDFAdvancedReader(const std::string& filePath);
    
    bool isNetCDFFile(const std::string& filePath) const;
    
    // =============================================================================
    // 🆕 大文件处理优化
    // =============================================================================
    
    size_t calculateOptimalChunkSize(const std::string& filePath, const std::string& variableName) const;
    bool shouldUseLargeFileOptimization(const std::string& filePath) const;
    
    // =============================================================================
    // 🆕 3D数据处理辅助方法
    // =============================================================================
    
    std::pair<size_t, size_t> findNearestGridIndices(
        const std::vector<double>& coordinates, double targetValue) const;
    
    double interpolateValue(
        const std::vector<double>& data, 
        const std::vector<size_t>& shape,
        double longitude, double latitude, 
        const std::optional<double>& depth = std::nullopt) const;
    
    // ===============================================================================
    // 成员变量
    // ===============================================================================
    
    // === 🔧 修改：使用依赖注入的组件，而不是全局单例 ===
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> servicesFactory_;
    data_access::api::DataAccessConfiguration config_;
    
    // 缓存管理 - 使用Common模块的ICache接口（定义在CommonServicesFactory中）
    std::shared_ptr<common_utils::infrastructure::ICache<std::string, oscean::core_services::FileMetadata>> metadataCache_;
    std::shared_ptr<common_utils::infrastructure::ICache<std::string, oscean::core_services::GridData>> gridCache_;
    
    // 读取器管理
    std::shared_ptr<data_access::readers::ReaderRegistry> readerRegistry_;
    
    // 状态管理
    std::atomic<bool> isInitialized_;
    mutable std::atomic<size_t> totalRequests_;
    mutable std::atomic<size_t> successfulRequests_;
    
    // 🎯 核心改进：注入的并发优化组件
    std::shared_ptr<GdalInitializationManager> gdalManager_;
    std::shared_ptr<FileAccessLockManager> lockManager_;
    std::shared_ptr<ReaderPoolManager> poolManager_;
    
    // 🔧 修复：使用const指针，与ReaderRegistry::getFormatDetector()返回类型一致
    const common_utils::utilities::FileFormatDetector* fileFormatDetector_;

    // 🆕 NetCDF Advanced Reader支持
    mutable std::shared_mutex netcdfReaderMutex_;
    std::unordered_map<std::string, std::shared_ptr<data_access::readers::impl::netcdf::NetCDFAdvancedReader>> netcdfReaderCache_;

    mutable std::once_flag m_initOnceFlag;
}; // class UnifiedDataAccessServiceImpl

} // namespace oscean::core_services 