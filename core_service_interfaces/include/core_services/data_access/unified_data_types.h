/**
 * @file unified_data_types.h
 * @brief 统一数据访问类型定义 - 核心接口层
 * 
 * 🎯 设计原则：
 * ✅ 统一类型定义 - 避免重复定义
 * ✅ 接口层定义 - 所有实现都使用这些类型
 * ✅ 移除CRS依赖 - 只包含原生CRS信息
 * ✅ 类型安全 - 使用强类型枚举和variant
 */

#pragma once

#include "core_services/common_data_types.h"
#include <variant>
#include <optional>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <map>

namespace oscean::core_services::data_access::api {

/**
 * @brief CRS转换请求参数
 * 
 * 为工作流层提供便利，DataAccess可以协调CRS服务进行转换
 * 注意：坐标转换的具体实现仍由CRS服务负责，DataAccess只是协调
 */
struct CRSTransformRequest {
    std::string sourceCRS;          ///< 源坐标系（WKT/PROJ/EPSG:xxxx格式）
    std::string targetCRS;          ///< 目标坐标系（WKT/PROJ/EPSG:xxxx格式）
    bool transformBounds = false;   ///< 是否转换空间边界
    bool transformGeometry = false; ///< 是否转换几何体坐标
    
    /**
     * @brief 检查是否需要坐标转换
     */
    bool needsTransform() const {
        return !sourceCRS.empty() && !targetCRS.empty() && sourceCRS != targetCRS;
    }
    
    /**
     * @brief 检查转换请求是否有效
     */
    bool isValid() const {
        return !sourceCRS.empty() && !targetCRS.empty();
    }
    
    /**
     * @brief 创建EPSG转换请求
     */
    static CRSTransformRequest createEpsgTransform(int sourceEpsg, int targetEpsg) {
        CRSTransformRequest request;
        request.sourceCRS = "EPSG:" + std::to_string(sourceEpsg);
        request.targetCRS = "EPSG:" + std::to_string(targetEpsg);
        request.transformBounds = true;
        request.transformGeometry = true;
        return request;
    }
    
    /**
     * @brief 创建到WGS84的转换请求
     */
    static CRSTransformRequest createToWGS84(const std::string& sourceCrs) {
        CRSTransformRequest request;
        request.sourceCRS = sourceCrs;
        request.targetCRS = "EPSG:4326";
        request.transformBounds = true;
        request.transformGeometry = true;
        return request;
    }
};

/**
 * @brief 读取区域参数 - 统一定义，避免重复
 * 用于指定栅格数据的读取区域和重采样参数
 */
struct ReadRegion {
    int xOff = 0;                    ///< X方向偏移量（像素）
    int yOff = 0;                    ///< Y方向偏移量（像素）
    int xSize = 0;                   ///< X方向大小（像素）
    int ySize = 0;                   ///< Y方向大小（像素）
    int bufXSize = -1;               ///< 缓冲区X大小（重采样用，-1表示与xSize相同）
    int bufYSize = -1;               ///< 缓冲区Y大小（重采样用，-1表示与ySize相同）
    
    /**
     * @brief 检查区域参数是否有效
     */
    bool isValid() const {
        return xSize > 0 && ySize > 0 && xOff >= 0 && yOff >= 0;
    }
    
    /**
     * @brief 获取总像素数
     */
    size_t getPixelCount() const {
        return static_cast<size_t>(xSize) * ySize;
    }
    
    /**
     * @brief 获取缓冲区像素数
     */
    size_t getBufferPixelCount() const {
        int bufX = (bufXSize > 0) ? bufXSize : xSize;
        int bufY = (bufYSize > 0) ? bufYSize : ySize;
        return static_cast<size_t>(bufX) * bufY;
    }
    
    /**
     * @brief 检查是否需要重采样
     */
    bool needsResampling() const {
        return (bufXSize > 0 && bufXSize != xSize) || 
               (bufYSize > 0 && bufYSize != ySize);
    }
    
    /**
     * @brief 创建简单区域（无重采样）
     */
    static ReadRegion create(int x, int y, int width, int height) {
        ReadRegion region;
        region.xOff = x;
        region.yOff = y;
        region.xSize = width;
        region.ySize = height;
        return region;
    }
    
    /**
     * @brief 创建重采样区域
     */
    static ReadRegion createWithResampling(int x, int y, int width, int height, 
                                          int bufWidth, int bufHeight) {
        ReadRegion region;
        region.xOff = x;
        region.yOff = y;
        region.xSize = width;
        region.ySize = height;
        region.bufXSize = bufWidth;
        region.bufYSize = bufHeight;
        return region;
    }
};

/**
 * @brief 统一数据请求类型枚举
 */
enum class UnifiedRequestType {
    FILE_METADATA,          ///< 获取文件元数据
    GRID_DATA,              ///< 读取格点数据
    FEATURE_COLLECTION,     ///< 读取要素集合
    TIME_SERIES,            ///< 读取时间序列
    VERTICAL_PROFILE,       ///< 读取垂直剖面
    VARIABLE_ATTRIBUTES,    ///< 获取变量属性
    GLOBAL_ATTRIBUTES,      ///< 获取全局属性
    FIELD_DEFINITIONS,      ///< 获取字段定义
    VARIABLE_EXISTS_CHECK,  ///< 检查变量是否存在
    STREAMING               ///< 流式数据读取
};

/**
 * @brief 统一数据请求
 */
struct UnifiedDataRequest {
    UnifiedRequestType requestType;         ///< 请求类型
    std::string requestId;                  ///< 请求ID（可选）
    std::string filePath;                   ///< 文件路径
    std::string variableName;               ///< 变量名（可选）
    std::string layerName;                  ///< 图层名（可选）
    
    // 空间参数
    std::optional<oscean::core_services::BoundingBox> spatialBounds;
    std::optional<oscean::core_services::Point> targetPoint;
    
    // 时间参数
    std::optional<oscean::core_services::TimeRange> timeRange;
    std::optional<std::chrono::system_clock::time_point> targetTime;
    
    // 🆕 坐标转换参数 - 为工作流层提供便利
    std::optional<CRSTransformRequest> crsTransform;
    
    // 处理参数
    std::optional<std::vector<double>> targetResolution;
    oscean::core_services::ResampleAlgorithm resampleAlgorithm = oscean::core_services::ResampleAlgorithm::NEAREST;
    std::string interpolationMethod = "nearest";
    
    // 输出参数
    bool includeNativeCrsInfo = true;
    bool includeMetadata = false;
    
    // 流式处理参数
    std::optional<size_t> chunkSize;
    std::function<bool(const std::vector<unsigned char>&)> streamCallback;
    
    /**
     * @brief 默认构造函数
     */
    UnifiedDataRequest() = default;
    
    /**
     * @brief 构造函数
     */
    UnifiedDataRequest(UnifiedRequestType type, const std::string& path)
        : requestType(type), filePath(path) {}
    
    /**
     * @brief 设置坐标转换（便捷方法）
     */
    void setCRSTransform(const std::string& sourceCrs, const std::string& targetCrs) {
        crsTransform = CRSTransformRequest{};
        crsTransform->sourceCRS = sourceCrs;
        crsTransform->targetCRS = targetCrs;
        crsTransform->transformBounds = true;
        crsTransform->transformGeometry = true;
    }
    
    /**
     * @brief 设置到WGS84的转换（便捷方法）
     */
    void setTransformToWGS84(const std::string& sourceCrs) {
        setCRSTransform(sourceCrs, "EPSG:4326");
    }
    
    /**
     * @brief 检查是否需要坐标转换
     */
    bool needsCRSTransform() const {
        return crsTransform.has_value() && crsTransform->needsTransform();
    }
};

/**
 * @brief 统一数据响应状态
 */
enum class UnifiedResponseStatus {
    SUCCESS,
    PARTIAL_SUCCESS,
    FAILED,
    NOT_FOUND,
    FORMAT_ERROR,
    INVALID_REQUEST
};

/**
 * @brief 统一数据响应
 */
struct UnifiedDataResponse {
    UnifiedResponseStatus status = UnifiedResponseStatus::SUCCESS;
    std::string requestId;                  ///< 请求ID
    std::string errorMessage;               ///< 错误消息
    std::string message;                    ///< 响应消息
    
    // 时间戳
    std::chrono::steady_clock::time_point timestamp = 
        std::chrono::steady_clock::now();
    
    // 数据内容（根据请求类型填充对应字段）
    std::variant<
        std::monostate,                     ///< 空状态
        std::shared_ptr<oscean::core_services::GridData>,           ///< 格点数据
        std::shared_ptr<oscean::core_services::FeatureCollection>,  ///< 要素集合
        std::shared_ptr<oscean::core_services::TimeSeriesData>,     ///< 时间序列数据
        std::shared_ptr<oscean::core_services::VerticalProfileData>, ///< 垂直剖面数据
        std::shared_ptr<oscean::core_services::FileMetadata>,       ///< 文件元数据
        std::map<std::string, std::string>, ///< 属性映射
        std::vector<oscean::core_services::FieldDefinition>,        ///< 字段定义列表
        bool,                               ///< 布尔结果
        std::vector<std::string>            ///< 字符串列表
    > data;
    
    // 原生CRS信息（字符串形式，不解析）
    std::optional<std::string> nativeCrsWkt;
    std::optional<std::string> nativeCrsProjString;
    std::optional<int> nativeCrsEpsgCode;
    
    // 处理统计
    std::chrono::milliseconds processingTimeMs{0};
    size_t bytesProcessed = 0;
    bool fromCache = false;
    
    /**
     * @brief 默认构造函数
     */
    UnifiedDataResponse() = default;
    
    /**
     * @brief 构造函数
     */
    UnifiedDataResponse(UnifiedResponseStatus stat, const std::string& msg = "")
        : status(stat), message(msg) {}
    
    /**
     * @brief 检查是否成功
     */
    bool isSuccess() const {
        return status == UnifiedResponseStatus::SUCCESS || 
               status == UnifiedResponseStatus::PARTIAL_SUCCESS;
    }
    
    /**
     * @brief 检查响应是否包含特定类型的数据
     */
    template<typename T>
    bool hasDataType() const {
        return std::holds_alternative<T>(data);
    }
    
    /**
     * @brief 获取特定类型的数据
     */
    template<typename T>
    const T* getDataAs() const {
        return std::get_if<T>(&data);
    }
    
    /**
     * @brief 创建成功响应
     */
    template<typename T>
    static UnifiedDataResponse createSuccess(const T& responseData, const std::string& msg = "Success") {
        UnifiedDataResponse response;
        response.status = UnifiedResponseStatus::SUCCESS;
        response.message = msg;
        response.data = responseData;
        return response;
    }
    
    /**
     * @brief 创建错误响应
     */
    static UnifiedDataResponse createError(const std::string& error) {
        UnifiedDataResponse response;
        response.status = UnifiedResponseStatus::FAILED;
        response.errorMessage = error;
        response.data = std::monostate{};
        return response;
    }
};

/**
 * @brief 数据访问性能指标
 */
struct DataAccessMetrics {
    // 基础指标
    size_t totalRequests = 0;
    size_t successfulRequests = 0;
    size_t failedRequests = 0;
    double averageResponseTimeMs = 0.0;
    
    // 吞吐量指标
    double currentThroughputMBps = 0.0;
    double peakThroughputMBps = 0.0;
    size_t totalBytesRead = 0;
    
    // 系统资源指标
    double memoryUsagePercent = 0.0;
    double cpuUsagePercent = 0.0;
    size_t currentMemoryUsageMB = 0;
    size_t peakMemoryUsageMB = 0;
    
    // 缓存统计
    struct CacheStats {
        double hitRatio = 0.0;
        size_t totalHits = 0;
        size_t totalMisses = 0;
        size_t currentSize = 0;
        size_t maxSize = 0;
    } cacheStats;
    
    // 时间戳
    std::chrono::system_clock::time_point lastUpdated = std::chrono::system_clock::now();
    
    /**
     * @brief 获取成功率
     */
    double getSuccessRate() const {
        return totalRequests > 0 ? 
            static_cast<double>(successfulRequests) / totalRequests : 1.0;
    }
};

/**
 * @brief 性能目标定义
 */
struct DataAccessPerformanceTargets {
    double targetThroughputMBps = 100.0;        ///< 目标吞吐量 (MB/s)
    double maxLatencyMs = 1000.0;               ///< 最大延迟 (毫秒)
    size_t maxMemoryUsageMB = 2048;             ///< 最大内存使用量 (MB)
    double targetCacheHitRatio = 0.85;          ///< 目标缓存命中率
    size_t maxConcurrentOperations = 16;       ///< 最大并发操作数
    size_t targetChunkSizeKB = 1024;            ///< 目标块大小 (KB)
    bool enableAdaptiveOptimization = true;    ///< 启用自适应优化
    double cpuUsageThreshold = 0.8;             ///< CPU使用率阈值
};

} // namespace oscean::core_services::data_access::api 