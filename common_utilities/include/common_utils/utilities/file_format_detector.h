/**
 * @file file_format_detector.h
 * @brief 文件格式检测工具 - 移除解析逻辑，专注格式识别
 * @author OSCEAN Team
 * @date 2024
 * 
 * 🔴 Critical: 此文件只包含格式检测工具，禁止包含任何数据解析逻辑
 * 数据解析应在数据访问服务层实现
 */

#pragma once

/**
 * @file file_format_detector.h
 * @brief 轻量级文件格式检测工具
 * 
 * 🎯 重构说明：
 * ✅ 从 format_utils 模块迁移并重构为轻量级检测工具
 * ✅ 只提供基础格式识别，不包含复杂的元数据提取
 * ✅ 专注于文件类型识别和基本验证
 */

// 🚀 使用Common模块的统一boost配置 - 参考CRS模块成功模式
#include "boost_config.h"
// OSCEAN_NO_BOOST_ASIO_MODULE();  // format_detector模块不使用boost::asio，只使用boost::future - 暂时注释

// 立即包含boost::future - 参考CRS模块
#include <boost/thread/future.hpp>

#include "../async/async_types.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <memory>
#include <functional>

namespace oscean::common_utils::utilities {

/**
 * @enum FileFormat
 * @brief 支持的文件格式枚举
 */
enum class FileFormat {
    UNKNOWN,
    NETCDF3,
    NETCDF4,
    HDF5,
    GEOTIFF,
    GDAL_RASTER,
    SHAPEFILE,
    GEOPACKAGE,
    JSON,
    CSV,
    BINARY
};

/**
 * @struct FormatInfo
 * @brief 文件格式信息结构
 */
struct FormatInfo {
    FileFormat format = FileFormat::UNKNOWN;
    std::string formatName;
    std::string description;
    std::vector<std::string> extensions;
    bool isGeospatial = false;
    bool isCompressed = false;
    std::string detectionMethod; // "extension", "header", "content"
    
    // 构造函数
    FormatInfo() = default;
    
    FormatInfo(FileFormat fmt, const std::string& name, const std::string& desc,
               const std::vector<std::string>& exts, bool geo, bool compressed, const std::string& method)
        : format(fmt), formatName(name), description(desc), extensions(exts), 
          isGeospatial(geo), isCompressed(compressed), detectionMethod(method) {}
};

/**
 * @brief 格式检测结果
 */
struct FormatDetectionResult {
    FileFormat format;
    std::string formatName;
    std::string version;
    double confidence;                          // 检测置信度 0.0-1.0
    std::vector<std::string> possibleFormats;  // 可能的格式列表
    std::map<std::string, std::string> basicInfo; // 基础信息（不涉及数据解析）
    
    FormatDetectionResult() : format(FileFormat::UNKNOWN), confidence(0.0) {}
    
    bool isValid() const noexcept { return format != FileFormat::UNKNOWN && confidence > 0.5; }
};

/**
 * @brief 格式能力信息
 */
struct FormatCapabilities {
    bool supportsStreaming;
    bool supportsMetadata;
    bool supportsMultiVariable;
    bool supportsCompression;
    size_t maxFileSize;                        // 最大支持文件大小
    std::vector<std::string> supportedExtensions;
    
    FormatCapabilities() : supportsStreaming(false), supportsMetadata(false),
                          supportsMultiVariable(false), supportsCompression(false),
                          maxFileSize(0) {}
};

/**
 * @class FileFormatDetector
 * @brief 轻量级文件格式检测器
 * 
 * 提供基础的文件格式识别功能，支持：
 * - 基于扩展名的快速识别
 * - 基于文件头的准确识别  
 * - 基于内容采样的深度识别
 */
class FileFormatDetector {
public:
    // === 构造和析构 ===
    FileFormatDetector() = default;
    virtual ~FileFormatDetector() = default;
    
    // 允许拷贝和移动
    FileFormatDetector(const FileFormatDetector&) = default;
    FileFormatDetector& operator=(const FileFormatDetector&) = default;
    FileFormatDetector(FileFormatDetector&&) = default;
    FileFormatDetector& operator=(FileFormatDetector&&) = default;
    
    // === 基础格式检测 ===
    
    /**
     * @brief 检测文件格式
     */
    FormatDetectionResult detectFormat(const std::string& filePath) const;
    
    /**
     * @brief 从扩展名检测格式
     */
    FileFormat detectFromExtension(const std::string& filePath) const;
    
    /**
     * @brief 从文件头检测格式
     */
    FormatDetectionResult detectFromHeader(const std::string& filePath) const;
    
    // === 批量检测 ===
    
    /**
     * @brief 批量格式检测 - 直接使用boost::future
     */
    boost::future<std::vector<FormatDetectionResult>> detectFormatsBatch(
        const std::vector<std::string>& filePaths) const;
    
    // === 格式验证 ===
    
    /**
     * @brief 验证文件格式
     */
    bool validateFormat(const std::string& filePath, FileFormat expectedFormat) const;
    
    /**
     * @brief 检查格式兼容性
     */
    bool isCompatibleFormat(FileFormat format1, FileFormat format2) const;
    
    // === 格式能力查询 ===
    
    /**
     * @brief 获取格式能力
     */
    FormatCapabilities getFormatCapabilities(FileFormat format) const;
    
    /**
     * @brief 检查是否支持流式处理
     */
    bool supportsStreaming(FileFormat format) const;
    
    /**
     * @brief 获取支持的所有格式
     */
    std::vector<FileFormat> getSupportedFormats() const;
    
    /**
     * @brief 获取格式描述
     */
    std::string getFormatDescription(FileFormat format) const;
    
    /**
     * @brief 获取支持的扩展名
     */
    std::vector<std::string> getSupportedExtensions() const;
    
    // === 静态工厂方法 ===
    
    /**
     * @brief 创建标准格式检测器
     */
    static std::unique_ptr<FileFormatDetector> createDetector();
    
    /**
     * @brief 创建针对特定格式的检测器
     */
    static std::unique_ptr<FileFormatDetector> createForFormat(FileFormat format);
    
    /**
     * @brief 创建高性能检测器
     */
    static std::unique_ptr<FileFormatDetector> createHighPerformanceDetector();

private:
    // === 具体格式检测方法 ===
    FormatDetectionResult detectNetCDFFormat(const std::string& filePath) const;
    FormatDetectionResult detectGDALFormat(const std::string& filePath) const;
    FormatDetectionResult detectHDF5Format(const std::string& filePath) const;
    FormatDetectionResult detectShapefileFormat(const std::string& filePath) const;
    FormatDetectionResult detectGeoPackageFormat(const std::string& filePath) const;
    
    // === 文件头检测辅助方法 ===
    std::vector<uint8_t> readFileHeader(const std::string& filePath, size_t bytes = 512) const;
    bool checkMagicBytes(const std::vector<uint8_t>& header, 
                        const std::vector<uint8_t>& signature) const;
    double calculateConfidence(const std::vector<uint8_t>& header, FileFormat format) const;
};

} // namespace oscean::common_utils::utilities 