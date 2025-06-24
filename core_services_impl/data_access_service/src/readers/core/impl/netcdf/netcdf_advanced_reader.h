/**
 * @file netcdf_advanced_reader.h
 * @brief NetCDF高级读取器 - 统一架构实现
 */

#pragma once

// 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();

// boost线程和异步库
#include <boost/thread/future.hpp>
#include <boost/thread.hpp>

// 核心接口和数据类型
#include "readers/core/unified_data_reader.h"
#include "core_services/common_data_types.h"

// Common Utilities高级功能
#include "common_utils/simd/simd_manager_unified.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/async/async_framework.h"
#include "common_utils/cache/icache_manager.h"
#include "common_utils/infrastructure/common_services_factory.h"

// NetCDF专用处理器
#include "netcdf_variable_processor.h"
#include "netcdf_coordinate_system.h"

// Common模块时间处理
#include "common_utils/time/time_interfaces.h"
#include "common_utils/time/time_services.h"

// NetCDF库
#include <netcdf.h>

// 标准库
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <optional>
#include <chrono>
#include <unordered_map>

namespace oscean::core_services::data_access::readers::impl::netcdf {

// 前向声明
using ncid_t = int;
using varid_t = int;

/**
 * @brief NetCDF高级配置结构
 */
struct NetCDFAdvancedConfig {
    size_t chunkCacheSize = 256 * 1024 * 1024;      // 256MB块缓存
    size_t maxConcurrentReads = 8;                  // 最大并发读取数
    bool enableVariableCache = true;                // 启用变量缓存
    bool enableTimeOptimization = true;             // 启用时间优化
    bool enableCFCompliance = true;                 // 启用CF约定合规
    bool enableStreamingMode = false;               // 启用流式处理
    size_t streamingChunkSize = 64 * 1024;          // 流式块大小
};

/**
 * @brief NetCDF性能统计
 */
struct NetCDFPerformanceStats {
    std::atomic<size_t> totalBytesRead{0};
    std::atomic<size_t> totalVariablesRead{0};
    std::atomic<size_t> cacheHits{0};
    std::atomic<size_t> cacheMisses{0};
    std::atomic<size_t> timeConversions{0};
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point lastAccessTime;
    
    NetCDFPerformanceStats() : startTime(std::chrono::steady_clock::now()) {}
};

/**
 * @brief NetCDF高级读取器 - 完整实现统一架构
 * 
 * 🎯 核心职责：
 * ✅ 继承UnifiedDataReader，实现所有必需接口
 * ✅ 集成common_utilities的高级功能（SIMD、内存、异步、缓存）
 * ✅ 整合现有的NetCDF专用处理器
 * ✅ 提供NetCDF特定的高级功能
 * ✅ 支持流式处理和大数据场景
 */
class NetCDFAdvancedReader final : public UnifiedDataReader {
public:
    /**
     * @brief 构造函数
     * @param filePath NetCDF文件路径
     * @param commonServices Common服务工厂（可选）
     */
    explicit NetCDFAdvancedReader(
        const std::string& filePath,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices = nullptr
    );
    
    /**
     * @brief 析构函数
     */
    ~NetCDFAdvancedReader() override;

    NetCDFAdvancedReader(const NetCDFAdvancedReader&) = delete;
    NetCDFAdvancedReader& operator=(const NetCDFAdvancedReader&) = delete;
    NetCDFAdvancedReader(NetCDFAdvancedReader&&) = delete;
    NetCDFAdvancedReader& operator=(NetCDFAdvancedReader&&) = delete;

    // =============================================================================
    // UnifiedDataReader 接口实现
    // =============================================================================
    
    boost::future<bool> openAsync() override;
    boost::future<void> closeAsync() override;
    std::string getReaderType() const override;
    
    boost::future<std::optional<oscean::core_services::FileMetadata>> getFileMetadataAsync() override;
    boost::future<std::vector<std::string>> getVariableNamesAsync() override;
    
    boost::future<std::shared_ptr<oscean::core_services::GridData>> readGridDataAsync(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds = std::nullopt) override;

    // =============================================================================
    // NetCDF特定高级功能
    // =============================================================================
    
    /**
     * @brief 配置NetCDF高级选项
     */
    void configureAdvancedOptions(const NetCDFAdvancedConfig& config);
    
    /**
     * @brief 启用/禁用SIMD优化
     */
    void enableSIMDOptimization(bool enable = true);
    
    /**
     * @brief 启用/禁用高级缓存
     */
    void enableAdvancedCaching(bool enable = true);
    
    /**
     * @brief 启用/禁用流式处理
     */
    void enableStreamingMode(bool enable = true);
    
    /**
     * @brief 获取性能统计
     */
    const NetCDFPerformanceStats& getPerformanceStats() const;
    
    /**
     * @brief 获取性能报告
     */
    std::string getPerformanceReport() const;

    // =============================================================================
    // NetCDF专用数据访问接口
    // =============================================================================
    
    /**
     * @brief 获取变量详细信息
     */
    boost::future<std::optional<oscean::core_services::VariableMeta>> getVariableInfoAsync(const std::string& variableName);
    
    /**
     * @brief 获取时间范围
     */
    boost::future<std::optional<oscean::core_services::TimeRange>> getTimeRangeAsync();
    
    /**
     * @brief 获取边界框
     */
    boost::future<oscean::core_services::BoundingBox> getBoundingBoxAsync();
    
    /**
     * @brief 获取CRS信息
     */
    boost::future<std::optional<oscean::core_services::CRSInfo>> getCRSInfoAsync();
    
    /**
     * @brief 获取维度信息
     */
    boost::future<std::vector<DimensionCoordinateInfo>> getDimensionInfoAsync();
    
    /**
     * @brief 获取垂直层信息
     */
    boost::future<std::vector<double>> getVerticalLevelsAsync();

    // =============================================================================
    // 流式处理接口
    // =============================================================================
    
    /**
     * @brief 流式读取变量数据
     */
    boost::future<void> streamVariableDataAsync(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        std::function<bool(const std::vector<double>&, const std::vector<size_t>&)> processor
    );
    
    /**
     * @brief 流式读取时间切片
     */
    boost::future<void> streamTimeSlicesAsync(
        const std::string& variableName,
        const std::optional<std::pair<size_t, size_t>>& timeRange,
        std::function<bool(const std::shared_ptr<oscean::core_services::GridData>&, size_t)> processor
    );

    // =============================================================================
    // 高级查询接口
    // =============================================================================
    
    /**
     * @brief 读取时间序列数据
     */
    boost::future<std::shared_ptr<oscean::core_services::GridData>> readTimeSeriesAsync(
        const std::string& variableName,
        double longitude,
        double latitude,
        const std::optional<std::pair<std::chrono::system_clock::time_point,
                                      std::chrono::system_clock::time_point>>& timeRange = std::nullopt
    );
    
    /**
     * @brief 读取垂直剖面数据
     */
    boost::future<std::shared_ptr<oscean::core_services::GridData>> readVerticalProfileAsync(
        const std::string& variableName,
        double longitude,
        double latitude,
        const std::optional<std::chrono::system_clock::time_point>& timePoint = std::nullopt
    );
    
    /**
     * @brief 读取指定时间的数据
     */
    boost::future<std::shared_ptr<oscean::core_services::GridData>> readTimeSliceAsync(
        const std::string& variableName,
        const std::chrono::system_clock::time_point& timePoint,
        const std::optional<oscean::core_services::BoundingBox>& bounds = std::nullopt
    );

    // =============================================================================
    // 🚀 配置化读取接口 (接收工作流层的策略配置)
    // =============================================================================
    
    /**
     * @brief 使用配置参数的数据读取接口
     * @param variableName 变量名
     * @param bounds 边界框
     * @param config 读取配置参数 (由工作流层提供)
     */
    boost::future<std::shared_ptr<oscean::core_services::GridData>> 
    readGridDataWithConfigAsync(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const std::unordered_map<std::string, std::string>& config
    );

private:
    // =============================================================================
    // 私有成员变量
    // =============================================================================
    
    std::string filePath_;                                   ///< 文件路径
    ncid_t ncid_;                                           ///< NetCDF文件ID
    std::atomic<bool> isOpen_{false};                       ///< 文件是否已打开
    
    // Common Utilities高级功能组件
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices_;
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;
    std::shared_ptr<oscean::common_utils::memory::IMemoryManager> memoryManager_;
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework_;
    std::shared_ptr<oscean::common_utils::infrastructure::ICache<std::string, std::vector<unsigned char>>> cacheManager_;
    
    // NetCDF专用处理器
    std::unique_ptr<NetCDFVariableProcessor> variableProcessor_;
    std::unique_ptr<NetCDFCoordinateSystemExtractor> coordinateSystem_;
    
    // 配置和状态
    NetCDFAdvancedConfig config_;
    std::atomic<bool> simdEnabled_{false};
    std::atomic<bool> cachingEnabled_{false};
    std::atomic<bool> streamingEnabled_{false};
    
    // 性能统计
    mutable NetCDFPerformanceStats performanceStats_;
    
    // 缓存
    mutable std::unordered_map<std::string, std::vector<std::string>> cachedVariableNames_;
    mutable std::unordered_map<std::string, oscean::core_services::VariableMeta> cachedVariableInfo_;
    mutable std::optional<oscean::core_services::FileMetadata> cachedFileMetadata_;

    // =============================================================================
    // 私有方法
    // =============================================================================
    
    /**
     * @brief 初始化NetCDF环境
     */
    bool initializeNetCDF();
    
    /**
     * @brief 初始化Common组件
     */
    void initializeCommonComponents();
    
    /**
     * @brief 初始化实例级别的组件（降级模式）
     */
    void initializeInstanceLevelComponents();
    
    /**
     * @brief 初始化NetCDF处理器
     */
    void initializeNetCDFProcessors();
    
    /**
     * @brief 验证NetCDF文件
     */
    bool validateNetCDFFile();
    
    /**
     * @brief 应用高级配置
     */
    void applyAdvancedConfiguration();
    
    /**
     * @brief 清理资源
     */
    void cleanup();
    
    /**
     * @brief 更新性能统计
     */
    void updatePerformanceStats(size_t bytesRead, bool cacheHit = false) const;
    
    /**
     * @brief 检查内存使用情况
     */
    bool checkMemoryUsage() const;
    
    /**
     * @brief 获取变量ID
     */
    varid_t getVariableId(const std::string& variableName) const;
    
    /**
     * @brief 检查变量是否存在
     */
    bool variableExists(const std::string& variableName) const;
    
    /**
     * @brief 创建GridData对象
     */
    std::shared_ptr<oscean::core_services::GridData> createGridData(
        const std::string& variableName,
        const std::vector<double>& data,
        const std::vector<size_t>& shape,
        const oscean::core_services::VariableMeta& varInfo
    ) const;
    
    // =============================================================================
    // 🚀 智能读取策略相关方法
    // =============================================================================
    
    /**
     * @brief 读取策略枚举
     */
    enum class ReadingStrategy {
        SMALL_SUBSET_OPTIMIZED,    // 小数据子集优化读取
        LARGE_DATA_STREAMING,      // 大数据流式读取
        CACHED_READING,            // 缓存优化读取
        SIMD_OPTIMIZED,           // SIMD向量化读取
        MEMORY_EFFICIENT,         // 内存高效读取
        STANDARD_READING          // 标准读取
    };
    
    /**
     * @brief 读取策略信息结构
     */
    struct ReadingStrategyInfo {
        ReadingStrategy strategy;
        std::string strategyName;
        double estimatedDataSizeMB;
        int optimizationLevel;      // 1-5级优化
        bool useCache;
        bool useSIMD;
        bool useStreaming;
        bool useMemoryPool;
        size_t chunkSize;
        int concurrencyLevel;
    };
    
    /**
     * @brief 数据特征分析结构
     */
    struct DataCharacteristics {
        double estimatedSizeMB;
        int dimensionCount;
        double subsetRatio;        // 子集占总数据的比例
        int complexityLevel;       // 1-5级复杂度
        bool isSIMDFriendly;      // 是否适合SIMD优化
        bool isTimeSeriesData;    // 是否为时间序列数据
        bool hasVerticalLayers;   // 是否有垂直层
    };
    
    /**
     * @brief 确保读取器就绪
     */
    bool ensureReaderReady();
    
    /**
     * @brief 带缓存的变量信息获取
     */
    std::optional<oscean::core_services::VariableMeta> getVariableInfoWithCache(const std::string& variableName);
    
    /**
     * @brief 选择最优读取策略
     */
    ReadingStrategyInfo selectOptimalReadingStrategy(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const oscean::core_services::VariableMeta& varInfo);
    
    /**
     * @brief 分析数据特征
     */
    DataCharacteristics analyzeDataCharacteristics(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const oscean::core_services::VariableMeta& varInfo);
    
    /**
     * @brief 执行小数据子集优化读取
     */
    std::shared_ptr<oscean::core_services::GridData> executeSmallSubsetReading(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const oscean::core_services::VariableMeta& varInfo,
        const ReadingStrategyInfo& strategy);
    
    /**
     * @brief 执行大数据流式读取
     */
    std::shared_ptr<oscean::core_services::GridData> executeLargeDataStreamingReading(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const oscean::core_services::VariableMeta& varInfo,
        const ReadingStrategyInfo& strategy);
    
    /**
     * @brief 执行缓存优化读取
     */
    std::shared_ptr<oscean::core_services::GridData> executeCachedReading(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const oscean::core_services::VariableMeta& varInfo,
        const ReadingStrategyInfo& strategy);
    
    /**
     * @brief 执行SIMD优化读取
     */
    std::shared_ptr<oscean::core_services::GridData> executeSIMDOptimizedReading(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const oscean::core_services::VariableMeta& varInfo,
        const ReadingStrategyInfo& strategy);
    
    /**
     * @brief 执行内存高效读取
     */
    std::shared_ptr<oscean::core_services::GridData> executeMemoryEfficientReading(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const oscean::core_services::VariableMeta& varInfo,
        const ReadingStrategyInfo& strategy);
    
    /**
     * @brief 执行标准读取
     */
    std::shared_ptr<oscean::core_services::GridData> executeStandardReading(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        const oscean::core_services::VariableMeta& varInfo,
        const ReadingStrategyInfo& strategy);
    
    /**
     * @brief 应用后处理优化
     */
    void applyPostProcessingOptimizations(
        std::shared_ptr<oscean::core_services::GridData>& gridData,
        const ReadingStrategyInfo& strategy);
    
    /**
     * @brief 丰富GridData元数据
     */
    void enrichGridDataMetadata(
        std::shared_ptr<oscean::core_services::GridData>& gridData,
        const std::string& variableName,
        const oscean::core_services::VariableMeta& varInfo,
        const ReadingStrategyInfo& strategy);
    
    /**
     * @brief 更新高级性能统计
     */
    void updateAdvancedPerformanceStats(
        const std::shared_ptr<oscean::core_services::GridData>& gridData,
        const ReadingStrategyInfo& strategy,
        const std::chrono::steady_clock::time_point& startTime);
    
    /**
     * @brief 辅助方法：计算子集比例
     */
    double calculateSubsetRatio(
        const oscean::core_services::BoundingBox& bounds,
        const std::vector<size_t>& shape);
    
    /**
     * @brief 辅助方法：检查是否为缓存候选
     */
    bool isCacheCandidate(const std::string& variableName, const DataCharacteristics& characteristics);
    
    /**
     * @brief 辅助方法：检查是否有时间维度
     */
    bool hasTimeDimension(const oscean::core_services::VariableMeta& varInfo);
    
    /**
     * @brief 辅助方法：检查是否有垂直维度
     */
    bool hasVerticalDimension(const oscean::core_services::VariableMeta& varInfo);
    
    /**
     * @brief 生成缓存键
     */
    std::string generateCacheKey(const std::string& variableName, const std::optional<oscean::core_services::BoundingBox>& bounds);
    
    /**
     * @brief 检查数据缓存
     */
    std::shared_ptr<oscean::core_services::GridData> checkDataCache(const std::string& cacheKey);
    
    /**
     * @brief 缓存数据结果
     */
    void cacheDataResult(const std::string& cacheKey, std::shared_ptr<oscean::core_services::GridData> gridData);
    
    /**
     * @brief 应用SIMD后处理
     */
    void applySIMDPostProcessing(std::shared_ptr<oscean::core_services::GridData>& gridData);
    
    /**
     * @brief 应用SIMD向量化
     */
    void applySIMDVectorization(std::shared_ptr<oscean::core_services::GridData>& gridData);
    
    /**
     * @brief 使用SIMD验证数据
     */
    void validateDataWithSIMD(std::shared_ptr<oscean::core_services::GridData>& gridData);
    
    /**
     * @brief 执行数据质量检查
     */
    void performDataQualityCheck(std::shared_ptr<oscean::core_services::GridData>& gridData);
    
    /**
     * @brief 应用数据压缩
     */
    void applyDataCompression(std::shared_ptr<oscean::core_services::GridData>& gridData);
    
    /**
     * @brief 优化内存对齐
     */
    void optimizeMemoryAlignment(std::shared_ptr<oscean::core_services::GridData>& gridData);
    
    /**
     * @brief 优化内存使用
     */
    void optimizeMemoryUsage(std::shared_ptr<oscean::core_services::GridData>& gridData, std::shared_ptr<void> memoryPool);
};

} // namespace oscean::core_services::data_access::readers::impl::netcdf 