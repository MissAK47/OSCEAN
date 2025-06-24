/**
 * @file optimized_crs_service_impl.h
 * @brief 优化的CRS服务实现 - 海洋大数据专用
 * 
 * 🎯 核心特性：
 * ✅ 使用Common模块的内存管理 (支持TB级数据处理)
 * ✅ 实际SIMD向量化优化 (AVX2/AVX512支持)
 * ✅ 智能缓存管理 (自适应缓存策略)
 * ✅ 性能监控和优化 (实时性能调优)
 * ✅ 真正的流式处理API (内存高效)
 * ✅ 集成GDAL/OGR功能
 */

#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // CRS服务只使用boost::future，不使用boost::asio

#include "core_services/crs/i_crs_service.h"
#include "core_services/crs/crs_service_factory.h"

// ✅ 只包含Common模块统一接口，不直接包含具体实现
#include "common_utils/infrastructure/common_services_factory.h"

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <boost/thread/future.hpp>

// Forward declarations for GDAL/OGR
#include <ogr_spatialref.h>
#include <ogr_geometry.h>
#include <proj.h>

// === 🔧 类型别名解决命名空间复杂性 ===
namespace crs_types {
    using CacheManager = oscean::common_utils::infrastructure::ICache<std::string, std::vector<double>>;
    using SIMDManager = oscean::common_utils::simd::ISIMDManager;
    using ThreadPoolManager = oscean::common_utils::infrastructure::threading::IThreadPoolManager;
    using PerformanceMonitor = oscean::common_utils::infrastructure::performance::IPerformanceMonitor;
    using MemoryManager = oscean::common_utils::memory::IMemoryManager;
}

namespace oscean::core_services::crs {

// Forward declarations
class NonStandardProjectionManager;

/**
 * @brief 海洋大数据专用的高性能坐标流实现
 */
class HighPerformanceCoordinateStream : public ICrsService::ICoordinateStream {
public:
    explicit HighPerformanceCoordinateStream(
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        size_t bufferSize,
        std::shared_ptr<crs_types::SIMDManager> simdManager,
        std::shared_ptr<crs_types::MemoryManager> memoryManager
    );
    
    ~HighPerformanceCoordinateStream() override;

    boost::future<void> processChunk(const std::vector<Point>& inputChunk) override;
    boost::future<std::vector<TransformedPoint>> getResults() override;
    boost::future<void> flush() override;
    void reset() override;
    size_t getProcessedCount() const override;
    double getCompressionRatio() const override;

private:
    struct StreamContext;
    std::unique_ptr<StreamContext> context_;
    
    void processBatchInternal();
    void processBatchSIMD();
    void processBatchStandard();
    void transformBatchAVX2(const double* inputX, const double* inputY,
                           double* outputX, double* outputY, size_t count);
};

/**
 * @brief 优化的CRS服务实现 - 海洋大数据专用
 */
class OptimizedCrsServiceImpl : public ICrsService,
                                public std::enable_shared_from_this<OptimizedCrsServiceImpl> {
public:
    /**
     * @brief 构造函数 - 注入Common模块服务
     */
    explicit OptimizedCrsServiceImpl(
        const CrsServiceConfig& config,
        std::shared_ptr<crs_types::MemoryManager> memoryManager,
        std::shared_ptr<crs_types::ThreadPoolManager> threadPoolManager,
        std::shared_ptr<crs_types::SIMDManager> simdManager,
        std::shared_ptr<crs_types::PerformanceMonitor> performanceMonitor,
        std::shared_ptr<crs_types::CacheManager> cache = nullptr
    );
    
    ~OptimizedCrsServiceImpl() override;

    // 禁用复制和移动
    OptimizedCrsServiceImpl(const OptimizedCrsServiceImpl&) = delete;
    OptimizedCrsServiceImpl& operator=(const OptimizedCrsServiceImpl&) = delete;
    OptimizedCrsServiceImpl(OptimizedCrsServiceImpl&&) = delete;
    OptimizedCrsServiceImpl& operator=(OptimizedCrsServiceImpl&&) = delete;

    // 移除错误的静态预热方法，GDAL初始化应由数据访问服务负责

    // === ICrsService接口实现 (统一使用boost::future) ===
    
    // 🚀 服务状态接口
    bool isReady() const override;
    std::string getStatus() const override;

    // 🚀 [新] 工作流支持方法
    boost::future<FileMetadata> enrichCrsInfoAsync(const FileMetadata& metadata) override;

    // Parser相关异步方法
    boost::future<boost::optional<CRSInfo>> parseFromWKTAsync(const std::string& wktString) override;
    boost::future<boost::optional<CRSInfo>> parseFromProjStringAsync(const std::string& projString) override;
    boost::future<boost::optional<CRSInfo>> parseFromEpsgCodeAsync(int epsgCode) override;

    // === 🆕 自动坐标系识别功能实现 ===
    boost::future<boost::optional<CRSInfo>> parseFromStringAsync(const std::string& crsString) override;
    boost::future<std::vector<CRSInfo>> suggestCRSFromBoundsAsync(const BoundingBox& bounds) override;
    boost::future<CRSValidationResult> validateCRSAsync(const CRSInfo& crsInfo) override;

    // Transformer相关异步方法
    boost::future<TransformedPoint> transformPointAsync(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) override;
    boost::future<TransformedPoint> transformPointAsync(double x, double y, double z, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) override;
    boost::future<std::vector<TransformedPoint>> transformPointsAsync(const std::vector<Point>& points, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) override;
    boost::future<BoundingBox> transformBoundingBoxAsync(const BoundingBox& sourceBbox, const CRSInfo& targetCRS) override;

    // CRS Analysis相关异步方法
    boost::future<boost::optional<CRSDetailedParameters>> getDetailedParametersAsync(const CRSInfo& crsInfo) override;
    boost::future<boost::optional<std::string>> getUnitAsync(const CRSInfo& crsInfo) override;
    boost::future<boost::optional<std::string>> getProjectionMethodAsync(const CRSInfo& crsInfo) override;
    boost::future<bool> areEquivalentCRSAsync(const CRSInfo& crsInfo1, const CRSInfo& crsInfo2) override;

    // === 🚀 高性能批量处理接口 ===
    
    boost::future<std::shared_ptr<ICoordinateStream>> createCoordinateStreamAsync(
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        size_t bufferSize = 50000
    ) override;

    boost::future<std::vector<TransformedPoint>> transformPointsBatchSIMDAsync(
        const std::vector<Point>& points,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        size_t simdBatchSize = 1000
    ) override;

    boost::future<void> transformPointsStreamAsync(
        const std::vector<Point>& points,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        std::function<void(const std::vector<TransformedPoint>&)> resultCallback,
        std::function<void(double)> progressCallback = nullptr,
        size_t streamBatchSize = 100000
    ) override;

    boost::future<GridData> reprojectGridAsync(
        const GridData& sourceGrid,
        const CRSInfo& targetCRS,
        const std::optional<double>& targetResolution = std::nullopt
    ) override;

    boost::future<CoordinateTransformationResult> transformLargeDatasetAsync(
        const std::vector<Point>& points,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        std::function<void(double)> progressCallback = nullptr
    ) override;

    // === 🔧 集成GDAL/OGR特定功能 ===

    boost::future<std::shared_ptr<OGRSpatialReference>> createOgrSrsAsync(const CRSInfo& crsInfo) override;
    boost::future<bool> canTransformAsync(const OGRSpatialReference* sourceSrs, const OGRSpatialReference* targetSrs) override;
    boost::future<std::vector<Point>> transformGeometryAsync(
        const std::vector<Point>& coords,
        OGRwkbGeometryType wkbType,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS
    );
    boost::future<std::vector<std::vector<unsigned char>>> transformWkbGeometriesAsync(
        const std::vector<std::vector<unsigned char>>& wkbGeometries,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS
    ) override;

    // === 🎯 性能监控和优化接口 ===

    boost::future<ServicePerformanceStats> getPerformanceStatsAsync() override;
    boost::future<void> warmupCacheAsync(
        const std::vector<std::pair<CRSInfo, CRSInfo>>& commonTransformations
    ) override;
    boost::future<void> optimizeConfigurationAsync() override;

    // === 🎯 CF约定投影参数处理接口 ===
    
    /**
     * @brief 从CF约定投影参数创建完整的CRS信息
     * @param cfParams CF约定投影参数
     * @return 完整的CRS信息，如果失败则返回nullopt
     */
    boost::future<boost::optional<CRSInfo>> createCRSFromCFParametersAsync(const CFProjectionParameters& cfParams) override;

private:
    // === Common模块服务实例 ===
    std::shared_ptr<crs_types::ThreadPoolManager> threadManager_;
    std::shared_ptr<crs_types::MemoryManager> memoryManager_;
    std::shared_ptr<crs_types::CacheManager> resultCache_;
    std::shared_ptr<crs_types::SIMDManager> simdManager_;
    std::shared_ptr<crs_types::PerformanceMonitor> perfMonitor_;
    
    // === 配置和状态 ===
    CrsServiceConfig config_;
    mutable std::atomic<size_t> totalTransformations_{0};
    mutable std::atomic<double> totalLatencyMs_{0.0};
    
    // === PROJ上下文管理 ===
    PJ_CONTEXT* projContext_;
    std::mutex projMutex_;
    
    // === 内部转换器缓存 ===
    struct TransformationContext;
    std::unique_ptr<TransformationContext> transformContext_;
    
    // === 私有SIMD实现方法 ===
    
    /**
     * @brief 实际的SIMD向量化坐标转换
     */
    std::vector<TransformedPoint> transformPointsSIMDImpl(
        const std::vector<Point>& points,
        PJ* transformer,
        size_t vectorWidth = 8  // AVX2默认
    );
    
    /**
     * @brief AVX2优化的批量坐标转换
     */
    void transformBatchAVX2(
        const double* inputX, const double* inputY,
        double* outputX, double* outputY,
        size_t count, PJ* transformer
    );
    
    /**
     * @brief AVX512优化的批量坐标转换（如果支持）
     */
    void transformBatchAVX512(
        const double* inputX, const double* inputY,
        double* outputX, double* outputY,
        size_t count, PJ* transformer
    );
    
    /**
     * @brief 检测和选择最佳SIMD指令集
     */
    void detectOptimalSIMDInstructions();
    
    // === 流式处理实现 ===
    
    /**
     * @brief 实际的流式转换核心
     */
    boost::future<void> streamTransformCore(
        const std::vector<Point>& points,
        PJ* transformer,
        std::function<void(const std::vector<TransformedPoint>&)> resultCallback,
        std::function<void(double)> progressCallback,
        size_t batchSize
    );
    
    /**
     * @brief 内存高效的流式批处理
     */
    void processStreamBatch(
        const Point* inputBatch,
        size_t batchSize,
        PJ* transformer,
        std::vector<TransformedPoint>& outputBuffer
    );
    
    // === 辅助方法 ===
    
    /**
     * @brief 获取或创建PROJ转换器
     */
    PJ* getOrCreateTransformer(const CRSInfo& sourceCRS, const CRSInfo& targetCRS);
    
    /**
     * @brief 缓存键生成
     */
    std::string generateCacheKey(const CRSInfo& sourceCRS, const CRSInfo& targetCRS) const;
    
    /**
     * @brief 性能监控点
     */
    void recordPerformanceMetrics(const std::string& operation, double durationMs, size_t dataSize);
    
    /**
     * @brief 内存使用优化
     */
    void optimizeMemoryUsage();
    
    /**
     * @brief 自适应配置调整
     */
    void adaptiveConfigurationUpdate();
    
    /**
     * @brief GDAL/OGR辅助方法
     */
    std::shared_ptr<OGRSpatialReference> createOgrSrsFromCrsInfo(const CRSInfo& crsInfo);
    void ensureGdalInitialized();

    // === 🆕 自动坐标系识别私有方法 ===
    
    /**
     * @brief 创建默认WGS84 CRS
     */
    CRSInfo createDefaultWGS84CRS();
    
    /**
     * @brief 从字符串智能解析CRS（自动检测类型）
     */
    boost::optional<CRSInfo> parseStringInternal(const std::string& crsString);
    
    /**
     * @brief 基于坐标范围推断CRS候选
     */
    std::vector<CRSInfo> generateCRSCandidatesFromBounds(const BoundingBox& bounds);
    
    /**
     * @brief 验证CRS定义内部实现
     */
    CRSValidationResult validateCRSInternal(const CRSInfo& crsInfo);
    


    // 验证坐标是否有效
    bool isValidCoordinate(double x, double y) const;
    
    /**
     * @brief 针对特定CRS的坐标验证
     */
    bool isValidCoordinateForCRS(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) const;

    // 🆕 非标准投影管理器
    std::unique_ptr<NonStandardProjectionManager> nonStandardManager_;
};

} // namespace oscean::core_services::crs 