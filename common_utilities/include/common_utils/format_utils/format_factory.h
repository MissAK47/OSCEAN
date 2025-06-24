#pragma once

/**
 * @file format_factory.h
 * @brief 格式工具工厂 - 统一工厂模式入口
 * @author OSCEAN Team
 * @date 2024
 * 
 * 🔴 Critical: 提供统一的工厂模式接口，支持异构模式
 * 强制使用boost::future，完全支持依赖注入
 */

// 🚀 使用Common模块的统一boost配置 - 参考CRS模块成功模式
#include "../utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // format_factory模块不使用boost::asio，只使用boost::future

// 立即包含boost::future - 参考CRS模块
#include <boost/thread/future.hpp>

#include "format_detection.h"
#include "format_metadata.h"
#include "netcdf/netcdf_format.h"
#include "netcdf/netcdf_streaming.h"
#include "gdal/gdal_format.h"
#include "gdal/gdal_streaming.h"
#include "../async/async_types.h"
#include "../utilities/file_format_detector.h"
#include "metadata_extractor.h"
#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <map>

namespace oscean::common_utils::format_utils {

// 使用async命名空间中的UnifiedFuture
using oscean::common_utils::async::UnifiedFuture;

// 前向声明
class IFormatReader;
class IFormatWriter;
class IMetadataExtractor;

/**
 * @brief 格式工具类型枚举
 */
enum class FormatToolType {
    DETECTOR,
    METADATA_EXTRACTOR,
    STREAM_READER,
    ALL_TOOLS
};

/**
 * @brief 格式工具环境配置
 */
enum class FormatEnvironment {
    DEVELOPMENT,
    TESTING,
    PRODUCTION,
    HIGH_PERFORMANCE
};

/**
 * @brief 统一格式工具工厂
 */
class UnifiedFormatToolsFactory {
public:
    // === 构造和析构 ===
    UnifiedFormatToolsFactory() = default;
    virtual ~UnifiedFormatToolsFactory() = default;
    
    // 允许拷贝和移动 (工厂模式需要)
    UnifiedFormatToolsFactory(const UnifiedFormatToolsFactory&) = default;
    UnifiedFormatToolsFactory& operator=(const UnifiedFormatToolsFactory&) = default;
    UnifiedFormatToolsFactory(UnifiedFormatToolsFactory&&) = default;
    UnifiedFormatToolsFactory& operator=(UnifiedFormatToolsFactory&&) = default;
    
    // === 通用格式检测器工厂 ===
    
    /**
     * @brief 创建统一格式检测器
     */
    std::unique_ptr<UnifiedFormatDetector> createFormatDetector(
        FormatEnvironment env = FormatEnvironment::PRODUCTION) const;
    
    /**
     * @brief 创建针对特定格式的检测器
     */
    std::unique_ptr<UnifiedFormatDetector> createFormatDetectorFor(
        FileFormat format) const;
    
    // === 元数据提取器工厂 ===
    
    /**
     * @brief 创建通用元数据提取器
     */
    std::unique_ptr<IMetadataExtractor> createMetadataExtractor(
        FileFormat format) const;
    
    /**
     * @brief 创建NetCDF元数据提取器
     */
    std::unique_ptr<netcdf::NetCDFMetadataExtractor> createNetCDFMetadataExtractor() const;
    
    /**
     * @brief 创建GDAL元数据提取器
     */
    std::unique_ptr<gdal::GDALMetadataExtractor> createGDALMetadataExtractor() const;
    
    // === 流式读取器工厂 ===
    
    /**
     * @brief 创建NetCDF流式读取器
     */
    std::unique_ptr<netcdf::INetCDFStreamReader> createNetCDFStreamReader(
        const netcdf::NetCDFStreamingConfig& config = netcdf::NetCDFStreamingConfig{}) const;
    
    /**
     * @brief 创建GDAL栅格流式读取器
     */
    std::unique_ptr<gdal::IGDALStreamReader> createGDALRasterStreamReader(
        const gdal::GDALStreamingConfig& config = gdal::GDALStreamingConfig{}) const;
    
    /**
     * @brief 创建GDAL矢量流式读取器
     */
    std::unique_ptr<gdal::IGDALStreamReader> createGDALVectorStreamReader(
        const gdal::GDALStreamingConfig& config = gdal::GDALStreamingConfig{}) const;
    
    // === 自动检测工厂 ===
    
    /**
     * @brief 自动检测并创建合适的元数据提取器
     */
    std::unique_ptr<IMetadataExtractor> createAutoDetectMetadataExtractor(
        const std::string& filePath) const;
    
    /**
     * @brief 自动检测并创建合适的流式读取器
     */
    boost::future<void*> createAutoDetectStreamReader(
        const std::string& filePath) const;
    
    // === 环境特定工厂 ===
    
    /**
     * @brief 为开发环境创建工具
     */
    std::unique_ptr<UnifiedFormatDetector> createForDevelopment() const;
    
    /**
     * @brief 为高性能环境创建工具
     */
    std::unique_ptr<UnifiedFormatDetector> createForHighPerformance() const;
    
    /**
     * @brief 为大数据处理创建工具
     */
    std::unique_ptr<IMetadataExtractor> createForLargeData(
        size_t expectedFileSizeGB) const;
    
    // === 批量处理工厂 ===
    
    /**
     * @brief 创建批量格式检测器
     */
    boost::future<std::vector<FormatDetectionResult>> detectFormatsInBatch(
        const std::vector<std::string>& filePaths) const;
    
    /**
     * @brief 创建批量元数据提取器
     */
    boost::future<std::vector<FileMetadata>> extractMetadataInBatch(
        const std::vector<std::string>& filePaths) const;
    
    // === 工厂配置 ===
    
    /**
     * @brief 设置默认环境
     */
    void setDefaultEnvironment(FormatEnvironment env);
    
    /**
     * @brief 启用格式支持
     */
    void enableFormatSupport(FileFormat format, bool enable = true);
    
    /**
     * @brief 检查格式支持状态
     */
    bool isFormatSupported(FileFormat format) const;
    
    /**
     * @brief 获取支持的格式列表
     */
    std::vector<FileFormat> getSupportedFormats() const;

private:
    FormatEnvironment defaultEnvironment_ = FormatEnvironment::PRODUCTION;
    std::map<FileFormat, bool> formatSupport_;
    
    void initializeFormatSupport();
};

/**
 * @brief 格式工具服务捆绑包
 */
struct FormatToolsBundle {
    std::unique_ptr<UnifiedFormatDetector> detector;
    std::unique_ptr<IMetadataExtractor> metadataExtractor;
    void* streamReader = nullptr;  // 使用void*代替std::unique_ptr<void>，避免编译错误
    FileFormat detectedFormat;
    
    FormatToolsBundle() : detectedFormat(FileFormat::UNKNOWN) {}
    
    // 禁用拷贝构造和拷贝赋值
    FormatToolsBundle(const FormatToolsBundle&) = delete;
    FormatToolsBundle& operator=(const FormatToolsBundle&) = delete;
    
    // 允许移动构造和移动赋值
    FormatToolsBundle(FormatToolsBundle&& other) noexcept
        : detector(std::move(other.detector))
        , metadataExtractor(std::move(other.metadataExtractor))
        , streamReader(other.streamReader)
        , detectedFormat(other.detectedFormat) {
        other.streamReader = nullptr;
        other.detectedFormat = FileFormat::UNKNOWN;
    }
    
    FormatToolsBundle& operator=(FormatToolsBundle&& other) noexcept {
        if (this != &other) {
            detector = std::move(other.detector);
            metadataExtractor = std::move(other.metadataExtractor);
            streamReader = other.streamReader;
            detectedFormat = other.detectedFormat;
            
            other.streamReader = nullptr;
            other.detectedFormat = FileFormat::UNKNOWN;
        }
        return *this;
    }
    
    ~FormatToolsBundle() {
        // 注意：streamReader需要手动管理或者使用具体类型的deleter
        // 这是简化实现，实际应该使用更安全的方式
    }
    
    bool isValid() const noexcept {
        return detector && metadataExtractor && detectedFormat != FileFormat::UNKNOWN;
    }
};

/**
 * @brief 高级格式工具工厂 - 场景化工厂
 */
class AdvancedFormatToolsFactory {
public:
    /**
     * @brief 为特定文件创建完整工具包
     */
    static boost::future<FormatToolsBundle> createCompleteToolsForFile(
        const std::string& filePath);
    
    /**
     * @brief 为海洋数据处理创建优化工具包
     */
    static FormatToolsBundle createForOceanData(
        const std::string& filePath,
        size_t memoryLimitMB = 256);
    
    /**
     * @brief 为实时数据流创建工具包
     */
    static FormatToolsBundle createForRealTimeStreaming(
        const std::string& filePath);
    
    /**
     * @brief 为元数据分析创建轻量级工具包
     */
    static FormatToolsBundle createForMetadataAnalysis(
        const std::string& filePath);
    
    /**
     * @brief 为批量处理创建高效工具包
     */
    static std::vector<FormatToolsBundle> createForBatchProcessing(
        const std::vector<std::string>& filePaths,
        size_t maxConcurrentFiles = 4);
};

/**
 * @brief 格式工具配置构建器
 */
class FormatToolsConfigBuilder {
public:
    FormatToolsConfigBuilder& setEnvironment(FormatEnvironment env);
    FormatToolsConfigBuilder& enableFormat(FileFormat format);
    FormatToolsConfigBuilder& setMemoryLimit(size_t memoryLimitMB);
    FormatToolsConfigBuilder& enableCaching(bool enable = true);
    FormatToolsConfigBuilder& enableParallelProcessing(bool enable = true);
    
    /**
     * @brief 构建工具工厂配置
     */
    std::unique_ptr<UnifiedFormatToolsFactory> build() const;

private:
    FormatEnvironment environment_ = FormatEnvironment::PRODUCTION;
    std::vector<FileFormat> enabledFormats_;
    size_t memoryLimitMB_ = 256;
    bool cachingEnabled_ = true;
    bool parallelProcessingEnabled_ = true;
};

} // namespace oscean::common_utils::format_utils 