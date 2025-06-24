#pragma once

/**
 * @file gdal_common_types.h
 * @brief GDAL读取器模块通用类型定义
 * 
 * 统一所有GDAL相关的枚举、结构体和类型定义，
 * 避免重复定义和类型不一致问题。
 */

#include <atomic>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <optional>

// 🔧 引入统一的数据类型定义
#include "core_services/common_data_types.h"

namespace oscean::core_services::data_access::readers::impl::gdal {

// =============================================================================
// 枚举类型定义
// =============================================================================

/**
 * @brief GDAL数据类型枚举
 */
enum class GdalDataType {
    RASTER,     ///< 栅格数据
    VECTOR,     ///< 矢量数据
    UNKNOWN     ///< 未知类型
};

/**
 * @brief GDAL读取器类型枚举
 */
enum class GdalReaderType {
    RASTER,     ///< 栅格数据读取器
    VECTOR      ///< 矢量数据读取器
};

// =============================================================================
// 性能统计结构
// =============================================================================

/**
 * @brief GDAL性能统计信息
 */
struct GdalPerformanceStats {
    std::atomic<size_t> totalBytesRead{0};          ///< 总读取字节数
    std::atomic<size_t> totalBandsRead{0};          ///< 总读取波段数
    std::atomic<size_t> totalFeaturesRead{0};       ///< 总读取要素数
    std::atomic<size_t> simdOperationsCount{0};     ///< SIMD操作次数
    std::atomic<size_t> cacheHits{0};               ///< 缓存命中次数
    std::atomic<size_t> cacheMisses{0};             ///< 缓存未命中次数
    std::chrono::steady_clock::time_point startTime;   ///< 开始时间
    std::chrono::steady_clock::time_point lastAccessTime; ///< 最后访问时间
    
    GdalPerformanceStats() : startTime(std::chrono::steady_clock::now()), 
                            lastAccessTime(std::chrono::steady_clock::now()) {}
    
    // 拷贝构造函数（std::atomic不可拷贝）
    GdalPerformanceStats(const GdalPerformanceStats& other) 
        : totalBytesRead(other.totalBytesRead.load())
        , totalBandsRead(other.totalBandsRead.load())
        , totalFeaturesRead(other.totalFeaturesRead.load())
        , simdOperationsCount(other.simdOperationsCount.load())
        , cacheHits(other.cacheHits.load())
        , cacheMisses(other.cacheMisses.load())
        , startTime(other.startTime)
        , lastAccessTime(other.lastAccessTime) {}
    
    // 赋值操作符
    GdalPerformanceStats& operator=(const GdalPerformanceStats& other) {
        if (this != &other) {
            totalBytesRead.store(other.totalBytesRead.load());
            totalBandsRead.store(other.totalBandsRead.load());
            totalFeaturesRead.store(other.totalFeaturesRead.load());
            simdOperationsCount.store(other.simdOperationsCount.load());
            cacheHits.store(other.cacheHits.load());
            cacheMisses.store(other.cacheMisses.load());
            startTime = other.startTime;
            lastAccessTime = other.lastAccessTime;
        }
        return *this;
    }
};

// =============================================================================
// 配置结构
// =============================================================================

/**
 * @brief GDAL SIMD优化配置
 */
struct GdalSIMDConfig {
    bool enableVectorizedIO = true;              ///< 启用向量化IO
    bool enableParallelProcessing = true;        ///< 启用并行处理
    bool enableNoDataOptimization = true;        ///< 启用NoData优化
    bool enableStatisticsOptimization = true;    ///< 启用统计优化
    size_t vectorSize = 256;                     ///< 向量大小（位）
    size_t chunkSize = 1024 * 1024;              ///< 处理块大小
};

/**
 * @brief GDAL高级配置
 */
struct GdalAdvancedConfig {
    size_t blockCacheSize = 128 * 1024 * 1024;  ///< 块缓存大小（128MB）
    bool enableBlockCache = true;                ///< 启用块缓存
    bool enableOverviews = true;                 ///< 启用概览
    bool enableWarping = true;                   ///< 启用投影变换
    bool enableMultiThreading = true;            ///< 启用多线程
    size_t maxOpenFiles = 100;                   ///< 最大打开文件数
    double noDataTolerance = 1e-9;               ///< NoData值容差
};

/**
 * @brief 流式处理配置
 */
struct GdalStreamingConfig {
    size_t tileSize = 512;                       ///< 瓦片大小（像素）
    size_t maxConcurrentTiles = 4;               ///< 最大并发瓦片数
    size_t bufferSize = 1024 * 1024;             ///< 缓冲区大小
    bool enableOptimization = true;              ///< 启用优化
};

// =============================================================================
// 数据结构定义
// =============================================================================

/**
 * @brief 读取区域结构
 */
struct GdalReadRegion {
    int xOff = 0;          ///< X偏移
    int yOff = 0;          ///< Y偏移
    int xSize = 0;         ///< X大小
    int ySize = 0;         ///< Y大小
    
    bool isValid() const {
        return xSize > 0 && ySize > 0 && xOff >= 0 && yOff >= 0;
    }
};

// 🔧 注意：不再定义 GdalVariableInfo，统一使用 oscean::core_services::VariableMeta
// 对于GDAL特定的字段，我们使用 VariableMeta.attributes 来存储：
// - "band_number": 波段编号
// - "layer_name": 图层名称  
// - "geometry_type": 几何类型
// - "feature_count": 要素数量
// - "no_data_value": NoData值
// - "scale_factor": 缩放因子
// - "add_offset": 偏移量

/**
 * @brief 数据块结构
 */
struct GdalDataChunk {
    std::vector<double> data;                    ///< 数据内容
    std::vector<size_t> shape;                   ///< 数据形状
    std::vector<size_t> offset;                  ///< 在原始数据中的偏移
    size_t chunkId = 0;                          ///< 块ID
    bool isLastChunk = false;                    ///< 是否为最后一块
    std::map<std::string, std::string> metadata; ///< 元数据
};

/**
 * @brief 缓存键结构
 */
struct GdalCacheKey {
    std::string filePath;                        ///< 文件路径
    std::string variableName;                    ///< 变量名
    std::optional<size_t> boundsHash;            ///< 边界框哈希
    
    std::string toString() const {
        std::string key = filePath + ":" + variableName;
        if (boundsHash) {
            key += ":" + std::to_string(*boundsHash);
        }
        return key;
    }
};

// =============================================================================
// 统计信息结构
// =============================================================================

/**
 * @brief SIMD统计结果
 */
struct GdalSIMDStatistics {
    double min = 0.0;                           ///< 最小值
    double max = 0.0;                           ///< 最大值
    double mean = 0.0;                          ///< 平均值
    double sum = 0.0;                           ///< 总和
    double stddev = 0.0;                        ///< 标准差
    size_t validCount = 0;                      ///< 有效值数量
    size_t totalCount = 0;                      ///< 总数量
};

/**
 * @brief 文件信息结构
 */
struct GdalFileInfo {
    std::string driverName;                     ///< 驱动名称
    std::string driverLongName;                 ///< 驱动长名称
    GdalDataType dataType;                      ///< 数据类型
    size_t fileSize = 0;                        ///< 文件大小
    bool hasGeotransform = false;               ///< 是否有地理变换
    bool hasProjection = false;                 ///< 是否有投影
    
    // 栅格特定信息
    int rasterXSize = 0;                        ///< 栅格X大小
    int rasterYSize = 0;                        ///< 栅格Y大小
    int rasterCount = 0;                        ///< 波段数量
    
    // 矢量特定信息
    int layerCount = 0;                         ///< 图层数量
    std::vector<std::string> layerNames;        ///< 图层名称列表
};

// =============================================================================
// 错误处理相关
// =============================================================================

/**
 * @brief GDAL错误类型
 */
enum class GdalErrorType {
    NONE,                   ///< 无错误
    FILE_NOT_FOUND,         ///< 文件未找到
    INVALID_FORMAT,         ///< 无效格式
    PERMISSION_DENIED,      ///< 权限拒绝
    MEMORY_ERROR,           ///< 内存错误
    INVALID_PARAMETER,      ///< 无效参数
    GDAL_ERROR,             ///< GDAL库错误
    PROCESSING_ERROR,       ///< 处理错误
    UNKNOWN_ERROR           ///< 未知错误
};

/**
 * @brief GDAL错误信息
 */
struct GdalErrorInfo {
    GdalErrorType type = GdalErrorType::NONE;   ///< 错误类型
    std::string message;                        ///< 错误消息
    std::string file;                           ///< 相关文件
    int line = 0;                              ///< 行号
    int gdalErrorCode = 0;                     ///< GDAL错误代码
};

// =============================================================================
// 工具函数
// =============================================================================

/**
 * @brief 数据类型转换为字符串
 */
inline std::string gdalDataTypeToString(GdalDataType type) {
    switch (type) {
        case GdalDataType::RASTER: return "RASTER";
        case GdalDataType::VECTOR: return "VECTOR";
        case GdalDataType::UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

/**
 * @brief 读取器类型转换为字符串
 */
inline std::string gdalReaderTypeToString(GdalReaderType type) {
    switch (type) {
        case GdalReaderType::RASTER: return "GDAL_Raster";
        case GdalReaderType::VECTOR: return "GDAL_Vector";
        default: return "GDAL_Unknown";
    }
}

/**
 * @brief 错误类型转换为字符串
 */
inline std::string gdalErrorTypeToString(GdalErrorType type) {
    switch (type) {
        case GdalErrorType::NONE: return "NONE";
        case GdalErrorType::FILE_NOT_FOUND: return "FILE_NOT_FOUND";
        case GdalErrorType::INVALID_FORMAT: return "INVALID_FORMAT";
        case GdalErrorType::PERMISSION_DENIED: return "PERMISSION_DENIED";
        case GdalErrorType::MEMORY_ERROR: return "MEMORY_ERROR";
        case GdalErrorType::INVALID_PARAMETER: return "INVALID_PARAMETER";
        case GdalErrorType::GDAL_ERROR: return "GDAL_ERROR";
        case GdalErrorType::PROCESSING_ERROR: return "PROCESSING_ERROR";
        case GdalErrorType::UNKNOWN_ERROR: return "UNKNOWN_ERROR";
        default: return "INVALID";
    }
}

} // namespace oscean::core_services::data_access::readers::impl::gdal 