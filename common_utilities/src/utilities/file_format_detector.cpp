/**
 * @file file_format_detector.cpp
 * @brief 轻量级文件格式检测工具实现
 */

#include "common_utils/utilities/file_format_detector.h"
#include "common_utils/utilities/logging_utils.h"
#include <fstream>
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace oscean::common_utils::utilities {

namespace {
    const std::map<std::string, FileFormat> extensionToFormat{
        {".nc", FileFormat::NETCDF3},
        {".nc4", FileFormat::NETCDF4},
        {".netcdf", FileFormat::NETCDF4},
        {".h5", FileFormat::HDF5},
        {".hdf5", FileFormat::HDF5},
        {".tif", FileFormat::GEOTIFF},
        {".tiff", FileFormat::GEOTIFF},
        {".shp", FileFormat::SHAPEFILE},
        {".geojson", FileFormat::JSON},
        {".json", FileFormat::JSON},
        {".csv", FileFormat::CSV}
    };
    
    const std::map<FileFormat, std::vector<uint8_t>> headerSignatures = {
        {FileFormat::NETCDF3, {0x43, 0x44, 0x46, 0x01}},  // CDF\001
        {FileFormat::NETCDF4, {0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A}}, // NetCDF4使用HDF5格式
        {FileFormat::HDF5, {0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A}},
        {FileFormat::GEOTIFF, {0x49, 0x49, 0x2A, 0x00}},  // 小端TIFF
        {FileFormat::GEOTIFF, {0x4D, 0x4D, 0x00, 0x2A}}   // 大端TIFF
    };
    
    // 最小有效文件大小
    const std::map<FileFormat, size_t> minimumFileSizes = {
        {FileFormat::NETCDF3, 32},      // NetCDF最小头部大小
        {FileFormat::NETCDF4, 32},      // NetCDF4最小头部大小
        {FileFormat::HDF5, 512},        // HDF5最小头部大小
        {FileFormat::GEOTIFF, 8},       // TIFF最小头部大小
        {FileFormat::SHAPEFILE, 100},   // Shapefile最小大小
        {FileFormat::JSON, 2},          // JSON最小大小 "{}"
        {FileFormat::CSV, 1}            // CSV最小大小
    };
}

FormatDetectionResult FileFormatDetector::detectFormat(const std::string& filePath) const {
    FormatDetectionResult result;
    
    // 🔧 修复1: 首先检查文件是否存在和基本有效性
    std::ifstream testFile(filePath, std::ios::binary);
    if (!testFile.good()) {
        LOG_WARN("文件不存在或无法访问: {}", filePath);
        result.format = FileFormat::UNKNOWN;
        result.confidence = 0.0;
        return result;
    }
    
    // 🔧 修复2: 检查文件大小
    testFile.seekg(0, std::ios::end);
    size_t fileSize = static_cast<size_t>(testFile.tellg());
    testFile.close();
    
    // 空文件或过小文件应该被识别为UNKNOWN
    if (fileSize == 0) {
        LOG_WARN("文件为空: {}", filePath);
        result.format = FileFormat::UNKNOWN;
        result.confidence = 0.0;
        return result;
    }
    
    if (fileSize < 4) {  // 最小魔数需要4字节
        LOG_WARN("文件过小，无法确定格式: {} (大小: {} 字节)", filePath, fileSize);
        result.format = FileFormat::UNKNOWN;
        result.confidence = 0.0;
        return result;
    }
    
    // 🔧 修复3: 优先基于文件头检测（更可靠）
    FormatDetectionResult headerResult = detectFromHeader(filePath);
    if (headerResult.isValid()) {
        headerResult.formatName = getFormatDescription(headerResult.format);
        
        // 验证文件大小是否符合格式要求
        auto minSizeIt = minimumFileSizes.find(headerResult.format);
        if (minSizeIt != minimumFileSizes.end() && fileSize < minSizeIt->second) {
            LOG_WARN("文件大小不符合格式要求: {} (实际: {}, 最小: {})", 
                     filePath, fileSize, minSizeIt->second);
            headerResult.confidence *= 0.5;  // 降低置信度
        }
        
        return headerResult;
    }
    
    // 🔧 修复4: 如果文件头检测失败，再尝试扩展名检测（但置信度较低）
    FileFormat extFormat = detectFromExtension(filePath);
    if (extFormat != FileFormat::UNKNOWN) {
        result.format = extFormat;
        result.formatName = getFormatDescription(extFormat);
        
        // 基于扩展名的检测置信度较低，特别是当文件内容不匹配时
        result.confidence = 0.3;  // 降低基于扩展名的置信度
        
        // 进一步验证：检查文件大小是否合理
        auto minSizeIt = minimumFileSizes.find(extFormat);
        if (minSizeIt != minimumFileSizes.end() && fileSize < minSizeIt->second) {
            LOG_WARN("基于扩展名检测的格式与文件大小不符: {} (期望格式: {}, 文件大小: {})", 
                     filePath, result.formatName, fileSize);
            result.confidence = 0.1;  // 进一步降低置信度
        }
        
        // 🔧 修复5: 对于二进制格式，尝试验证基本的文件头
        if (extFormat == FileFormat::GEOTIFF || extFormat == FileFormat::HDF5 || 
            extFormat == FileFormat::NETCDF3 || extFormat == FileFormat::NETCDF4) {
            
            // 读取文件头进行基本验证
            std::vector<uint8_t> header = readFileHeader(filePath, 16);
            if (header.size() < 4) {
                LOG_WARN("无法读取足够的文件头用于验证: {}", filePath);
                result.confidence = 0.05;  // 极低置信度
            } else {
                // 简单检查：文件是否包含可打印字符（可能是损坏的二进制文件）
                bool allPrintable = std::all_of(header.begin(), header.begin() + 4,
                    [](uint8_t byte) { return std::isprint(byte) || std::isspace(byte); });
                
                if (allPrintable && extFormat != FileFormat::JSON && extFormat != FileFormat::CSV) {
                    LOG_WARN("二进制格式文件包含可打印字符，可能已损坏: {}", filePath);
                    result.confidence = 0.1;  // 降低置信度
                }
            }
        }
        
        // 🔧 修复6: Shapefile特殊处理 - 验证配套文件完整性
        if (extFormat == FileFormat::SHAPEFILE) {
            result.confidence = 0.8;  // 基础置信度
            
            // 检查Shapefile配套文件完整性(.shx, .dbf是必需的)
            std::filesystem::path shpPath(filePath);
            std::string basePath = shpPath.parent_path().string() + "/" + shpPath.stem().string();
            
            bool hasShx = std::filesystem::exists(basePath + ".shx");
            bool hasDbf = std::filesystem::exists(basePath + ".dbf");
            bool hasPrj = std::filesystem::exists(basePath + ".prj");
            
            if (hasShx && hasDbf) {
                result.confidence = 0.9;  // 有核心配套文件
                if (hasPrj) {
                    result.confidence = 0.95;  // 有完整配套文件包括投影信息
                }
                LOG_DEBUG("Shapefile配套文件验证成功: {} (shx:{}, dbf:{}, prj:{})", 
                         filePath, hasShx, hasDbf, hasPrj);
            } else {
                result.confidence = 0.3;  // 缺少必要的配套文件
                LOG_WARN("Shapefile配套文件不完整: {} (缺少 shx:{}, dbf:{})", 
                        filePath, !hasShx, !hasDbf);
            }
        }
        
        return result;
    }
    
    // 都无法识别时返回UNKNOWN
    LOG_DEBUG("无法识别文件格式: {}", filePath);
    return result;
}

FileFormat FileFormatDetector::detectFromExtension(const std::string& filePath) const {
    size_t dotPos = filePath.find_last_of('.');
    if (dotPos == std::string::npos) {
        return FileFormat::UNKNOWN;
    }
    
    std::string extension = filePath.substr(dotPos);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    auto it = extensionToFormat.find(extension);
    return (it != extensionToFormat.end()) ? it->second : FileFormat::UNKNOWN;
}

FormatDetectionResult FileFormatDetector::detectFromHeader(const std::string& filePath) const {
    FormatDetectionResult result;
    
    std::vector<uint8_t> header = readFileHeader(filePath, 512);
    if (header.empty()) {
        return result;
    }
    
    // 🔧 特殊处理：NetCDF4使用HDF5格式但扩展名为.nc
    // 首先检查是否为HDF5头部（NetCDF4和HDF5都使用这个魔数）
    const std::vector<uint8_t> hdf5Signature = {0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A};
    if (checkMagicBytes(header, hdf5Signature)) {
        // 检查文件扩展名来区分NetCDF4和HDF5
        size_t dotPos = filePath.find_last_of('.');
        if (dotPos != std::string::npos) {
            std::string extension = filePath.substr(dotPos);
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (extension == ".nc" || extension == ".netcdf") {
                // .nc扩展名的HDF5格式文件认定为NetCDF4
                result.format = FileFormat::NETCDF4;
                result.confidence = 0.95;  // 高置信度
                LOG_DEBUG("检测到NetCDF4文件（HDF5底层）: {}", filePath);
                return result;
            } else if (extension == ".nc4") {
                // .nc4扩展名明确是NetCDF4
                result.format = FileFormat::NETCDF4;
                result.confidence = 0.95;
                LOG_DEBUG("检测到NetCDF4文件: {}", filePath);
                return result;
            } else if (extension == ".h5" || extension == ".hdf5") {
                // .h5或.hdf5扩展名是纯HDF5
                result.format = FileFormat::HDF5;
                result.confidence = 0.9;
                LOG_DEBUG("检测到HDF5文件: {}", filePath);
                return result;
            }
        }
        
        // 默认情况下，有HDF5头部但没有明确扩展名的认定为HDF5
        result.format = FileFormat::HDF5;
        result.confidence = 0.8;
        LOG_DEBUG("检测到HDF5格式文件（默认）: {}", filePath);
        return result;
    }
    
    // 🔧 修复6: 检查其他格式的魔数匹配
    for (const auto& [format, signature] : headerSignatures) {
        // 跳过HDF5和NETCDF4（已在上面特殊处理）
        if (format == FileFormat::HDF5 || format == FileFormat::NETCDF4) {
            continue;
        }
        
        if (checkMagicBytes(header, signature)) {
            result.format = format;
            result.confidence = 0.9;  // 基于文件头的检测置信度高
            
            // 🔧 修复7: 对TIFF格式进行额外验证
            if (format == FileFormat::GEOTIFF) {
                // 验证TIFF文件的基本结构
                if (header.size() >= 8) {
                    // 检查IFD偏移是否合理（应该在文件范围内）
                    uint32_t ifdOffset;
                    if (header[0] == 0x49 && header[1] == 0x49) {  // 小端
                        ifdOffset = header[4] | (header[5] << 8) | (header[6] << 16) | (header[7] << 24);
                    } else {  // 大端
                        ifdOffset = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7];
                    }
                    
                    // IFD偏移应该在合理范围内
                    if (ifdOffset == 0 || ifdOffset > 10000000) {  // 10MB限制
                        LOG_WARN("TIFF文件IFD偏移异常: {} (偏移: {})", filePath, ifdOffset);
                        result.confidence = 0.6;  // 降低置信度
                    }
                }
            }
            
            return result;
        }
    }
    
    return result;
}

bool FileFormatDetector::validateFormat(const std::string& filePath, FileFormat expectedFormat) const {
    FormatDetectionResult result = detectFormat(filePath);
    return result.format == expectedFormat;
}

bool FileFormatDetector::supportsStreaming(FileFormat format) const {
    return format == FileFormat::NETCDF3 || format == FileFormat::NETCDF4 || format == FileFormat::HDF5;
}

std::vector<FileFormat> FileFormatDetector::getSupportedFormats() const {
    return {
        FileFormat::NETCDF3, FileFormat::NETCDF4, FileFormat::HDF5,
        FileFormat::GEOTIFF, FileFormat::SHAPEFILE, FileFormat::JSON, FileFormat::CSV
    };
}

std::string FileFormatDetector::getFormatDescription(FileFormat format) const {
    switch (format) {
        case FileFormat::NETCDF3: return "NETCDF";
        case FileFormat::NETCDF4: return "NETCDF";
        case FileFormat::HDF5: return "HDF5";
        case FileFormat::GEOTIFF: return "GEOTIFF";
        case FileFormat::SHAPEFILE: return "SHAPEFILE";
        case FileFormat::JSON: return "GEOJSON";
        case FileFormat::CSV: return "CSV";
        default: return "UNKNOWN";
    }
}

std::vector<std::string> FileFormatDetector::getSupportedExtensions() const {
    std::vector<std::string> extensions;
    for (const auto& [ext, format] : extensionToFormat) {
        extensions.push_back(ext);
    }
    return extensions;
}

std::vector<uint8_t> FileFormatDetector::readFileHeader(const std::string& filePath, size_t bytes) const {
    std::vector<uint8_t> header;
    std::ifstream file(filePath, std::ios::binary);
    
    if (!file.is_open()) {
        return header;
    }
    
    header.resize(bytes);
    file.read(reinterpret_cast<char*>(header.data()), bytes);
    std::streamsize bytesRead = file.gcount();
    
    if (bytesRead > 0) {
        header.resize(static_cast<size_t>(bytesRead));
    } else {
        header.clear();
    }
    
    return header;
}

bool FileFormatDetector::checkMagicBytes(const std::vector<uint8_t>& header, 
                                        const std::vector<uint8_t>& signature) const {
    if (header.size() < signature.size()) {
        return false;
    }
    return std::equal(signature.begin(), signature.end(), header.begin());
}

std::unique_ptr<FileFormatDetector> FileFormatDetector::createDetector() {
    return std::make_unique<FileFormatDetector>();
}

} // namespace oscean::common_utils::utilities 