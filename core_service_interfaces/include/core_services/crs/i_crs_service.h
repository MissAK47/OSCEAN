#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // CRS服务不使用boost::asio，只使用boost::future

#include "core_services/common_data_types.h"
#include "crs_operation_types.h" // For TransformedPoint, CRSDetailedParameters
#include <vector>
#include <optional>
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <boost/thread/future.hpp>

// Forward declarations for GDAL/OGR types
class OGRSpatialReference;
class OGRGeometry;

namespace oscean::core_services {

/**
 * @interface ICrsService
 * @brief 统一坐标参考系统服务接口 (boost::future版本)
 * 
 * 🎯 现代化特性：
 * ✅ 统一使用boost::future异步接口
 * ✅ 集成GDAL/OGR特定功能
 * ✅ 支持海洋大数据处理
 * ✅ SIMD向量化优化
 * ✅ 流式处理API
 */
class ICrsService {
public:
    virtual ~ICrsService() = default;

    /**
     * @brief 获取服务的当前状态。
     * @return std::string 服务状态描述。
     */
    virtual std::string getStatus() const = 0;

    /**
     * @brief 检查服务是否准备就绪。
     * @return bool 如果服务准备就绪，则为 true。
     */
    virtual bool isReady() const = 0;

    /**
     * @brief 🚀 [新] 异步解析并丰富FileMetadata中的CRS信息
     * @param metadata 包含原始CRS字符串的元数据对象
     * @return 返回一个CRS字段被填充后的元数据对象的future
     */
    virtual boost::future<FileMetadata> enrichCrsInfoAsync(const FileMetadata& metadata) = 0;

    // === 🆕 流式坐标转换接口 ===
    
    /**
     * @brief 高性能流式坐标转换接口
     */
    class ICoordinateStream {
    public:
        virtual ~ICoordinateStream() = default;
        virtual boost::future<void> processChunk(const std::vector<Point>& inputChunk) = 0;
        virtual boost::future<std::vector<TransformedPoint>> getResults() = 0;
        virtual boost::future<void> flush() = 0;
        virtual void reset() = 0;
        virtual size_t getProcessedCount() const = 0;
        virtual double getCompressionRatio() const = 0;
    };

    // === 核心Parser相关异步方法 ===
    virtual boost::future<boost::optional<CRSInfo>> parseFromWKTAsync(const std::string& wktString) = 0;
    virtual boost::future<boost::optional<CRSInfo>> parseFromProjStringAsync(const std::string& projString) = 0;
    virtual boost::future<boost::optional<CRSInfo>> parseFromEpsgCodeAsync(int epsgCode) = 0;
    
    /**
     * @brief 从CF（Climate and Forecast）投影参数创建CRS
     * @param cfParams CF格式的投影参数（如从NetCDF文件提取的参数）
     * @return 创建的CRS信息
     */
    virtual boost::future<boost::optional<CRSInfo>> createCRSFromCFParametersAsync(const CFProjectionParameters& cfParams) = 0;

    // === 🆕 自动坐标系识别功能 ===
    
    /**
     * @brief 从WKT或PROJ字符串自动识别CRS类型并解析
     * @param crsString CRS字符串（可能是WKT、PROJ、EPSG代码等）
     * @return 解析后的CRS信息
     */
    virtual boost::future<boost::optional<CRSInfo>> parseFromStringAsync(const std::string& crsString) = 0;

    /**
     * @brief 基于空间范围推断可能的坐标系
     * @param bounds 数据边界框
     * @return 可能的CRS候选列表，按可能性排序
     */
    virtual boost::future<std::vector<CRSInfo>> suggestCRSFromBoundsAsync(const BoundingBox& bounds) = 0;

    /**
     * @brief 验证CRS定义的有效性
     * @param crsInfo 待验证的CRS信息
     * @return 是否有效以及验证详情
     */
    struct CRSValidationResult {
        bool isValid = false;
        std::string errorMessage;
        std::optional<CRSInfo> correctedCRS;  // 如果有修正建议
    };
    virtual boost::future<CRSValidationResult> validateCRSAsync(const CRSInfo& crsInfo) = 0;

    // === 核心Transformer相关异步方法 ===
    virtual boost::future<TransformedPoint> transformPointAsync(double x, double y, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual boost::future<TransformedPoint> transformPointAsync(double x, double y, double z, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual boost::future<std::vector<TransformedPoint>> transformPointsAsync(const std::vector<Point>& points, const CRSInfo& sourceCRS, const CRSInfo& targetCRS) = 0;
    virtual boost::future<BoundingBox> transformBoundingBoxAsync(const BoundingBox& sourceBbox, const CRSInfo& targetCRS) = 0;

    // === CRS Analysis相关异步方法 ===
    virtual boost::future<boost::optional<CRSDetailedParameters>> getDetailedParametersAsync(const CRSInfo& crsInfo) = 0;
    virtual boost::future<boost::optional<std::string>> getUnitAsync(const CRSInfo& crsInfo) = 0;
    virtual boost::future<boost::optional<std::string>> getProjectionMethodAsync(const CRSInfo& crsInfo) = 0;
    virtual boost::future<bool> areEquivalentCRSAsync(const CRSInfo& crsInfo1, const CRSInfo& crsInfo2) = 0;

    // === 🚀 高性能批量处理接口 ===

    /**
     * @brief 创建高性能坐标流，用于海洋大数据集的流式坐标转换
     */
    virtual boost::future<std::shared_ptr<ICoordinateStream>> createCoordinateStreamAsync(
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        size_t bufferSize = 50000  // 海洋数据优化默认值
    ) = 0;

    /**
     * @brief SIMD优化的批量坐标转换
     */
    virtual boost::future<std::vector<TransformedPoint>> transformPointsBatchSIMDAsync(
        const std::vector<Point>& points,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        size_t simdBatchSize = 1000
    ) = 0;

    /**
     * @brief 流式大数据集坐标转换（TB级数据支持）
     */
    virtual boost::future<void> transformPointsStreamAsync(
        const std::vector<Point>& points,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        std::function<void(const std::vector<TransformedPoint>&)> resultCallback,
        std::function<void(double)> progressCallback = nullptr,
        size_t streamBatchSize = 100000
    ) = 0;

    /**
     * @brief 重投影整个栅格数据集
     */
    virtual boost::future<GridData> reprojectGridAsync(
        const GridData& sourceGrid,
        const CRSInfo& targetCRS,
        const std::optional<double>& targetResolution = std::nullopt
    ) = 0;

    /**
     * @brief 大数据集的异步坐标转换（支持进度回调）
     */
    virtual boost::future<CoordinateTransformationResult> transformLargeDatasetAsync(
        const std::vector<Point>& points,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS,
        std::function<void(double)> progressCallback = nullptr
    ) = 0;

    // === 🔧 集成GDAL/OGR特定功能 ===

    /**
     * @brief 根据CRSInfo创建OGR空间参考对象
     */
    virtual boost::future<std::shared_ptr<OGRSpatialReference>> createOgrSrsAsync(const CRSInfo& crsInfo) = 0;

    /**
     * @brief 检查两个OGR空间参考是否可以进行转换
     */
    virtual boost::future<bool> canTransformAsync(const OGRSpatialReference* sourceSrs, const OGRSpatialReference* targetSrs) = 0;

    /**
     * @brief 批量转换WKB几何数据
     */
    virtual boost::future<std::vector<std::vector<unsigned char>>> transformWkbGeometriesAsync(
        const std::vector<std::vector<unsigned char>>& wkbGeometries,
        const CRSInfo& sourceCRS,
        const CRSInfo& targetCRS
    ) = 0;

    // === 🎯 性能监控和优化接口 ===

    /**
     * @brief 获取服务性能统计
     */
    struct ServicePerformanceStats {
        size_t totalTransformations = 0;
        double averageLatencyMs = 0.0;
        double simdAccelerationFactor = 1.0;
        double cacheHitRatio = 0.0;
        size_t memoryUsageMB = 0;
        double throughputPointsPerSecond = 0.0;
    };

    virtual boost::future<ServicePerformanceStats> getPerformanceStatsAsync() = 0;

    /**
     * @brief 预热缓存系统
     */
    virtual boost::future<void> warmupCacheAsync(
        const std::vector<std::pair<CRSInfo, CRSInfo>>& commonTransformations
    ) = 0;

    /**
     * @brief 动态优化服务配置
     */
    virtual boost::future<void> optimizeConfigurationAsync() = 0;
};

} // namespace oscean::core_services 