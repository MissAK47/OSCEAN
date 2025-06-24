#include "reader_registry.h"
#include "impl/netcdf/netcdf_advanced_reader.h"
#include "impl/gdal/gdal_raster_reader.h"
#include "impl/gdal/gdal_vector_reader.h"
#include <algorithm>

namespace oscean::core_services::data_access::readers {

ReaderRegistry::ReaderRegistry(
    std::unique_ptr<oscean::common_utils::utilities::FileFormatDetector> formatDetector,
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices)
    : formatDetector_(std::move(formatDetector))
    , commonServices_(commonServices) {
    
    if (!formatDetector_) {
        LOG_ERROR("ReaderRegistry构造时格式检测器为空");
        throw std::invalid_argument("格式检测器不能为空");
    }
    
    // 🔧 新增：定义严格的格式支持白名单
    initializeSupportedFormats();
    
    // 注册NetCDF高级读取器 - 传递CommonServices参数
    registerReaderFactory("NETCDF", [](const std::string& filePath, 
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices) {
        return std::static_pointer_cast<UnifiedDataReader>(
            std::make_shared<impl::netcdf::NetCDFAdvancedReader>(filePath, commonServices)
        );
    });

    // 注册GDAL栅格读取器工厂 - 传递CommonServices参数
    ReaderFactory gdalRasterFactory = [](const std::string& filePath,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices) {
        return std::static_pointer_cast<UnifiedDataReader>(
            std::make_shared<impl::gdal::GdalRasterReader>(filePath, commonServices)
        );
    };
    
    // 注册GDAL矢量读取器工厂 - 传递CommonServices参数
    ReaderFactory gdalVectorFactory = [](const std::string& filePath,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices) {
        return std::static_pointer_cast<UnifiedDataReader>(
            std::make_shared<impl::gdal::GdalVectorReader>(filePath, commonServices)
        );
    };
    
    // 注册栅格格式
    registerReaderFactory("GEOTIFF", gdalRasterFactory);
    registerReaderFactory("HDF5", gdalRasterFactory);
    registerReaderFactory("GRIB", gdalRasterFactory);
    registerReaderFactory("GDAL_RASTER", gdalRasterFactory);
    
    // 注册矢量格式
    registerReaderFactory("SHAPEFILE", gdalVectorFactory);
    registerReaderFactory("GDAL_VECTOR", gdalVectorFactory);
    registerReaderFactory("GEOJSON", gdalVectorFactory);
    registerReaderFactory("KML", gdalVectorFactory);
    registerReaderFactory("GPX", gdalVectorFactory);
    registerReaderFactory("GML", gdalVectorFactory);
    registerReaderFactory("WFS", gdalVectorFactory);
    
    // 添加更多NetCDF相关格式别名 - 传递CommonServices参数
    registerReaderFactory("NC", [](const std::string& filePath,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices) {
        return std::static_pointer_cast<UnifiedDataReader>(
            std::make_shared<impl::netcdf::NetCDFAdvancedReader>(filePath, commonServices)
        );
    });
    registerReaderFactory("NC4", [](const std::string& filePath,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices) {
        return std::static_pointer_cast<UnifiedDataReader>(
            std::make_shared<impl::netcdf::NetCDFAdvancedReader>(filePath, commonServices)
        );
    });
    
    LOG_INFO("ReaderRegistry初始化完成 - 已注册 {} 个读取器工厂", readerFactories_.size());
    
    // 🔧 验证所有注册的格式都在白名单中
    validateRegisteredFormats();
}

// 🔧 新增：初始化支持的格式白名单
void ReaderRegistry::initializeSupportedFormats() {
    // 定义严格的GDAL和NetCDF支持的格式白名单
    supportedFormats_ = {
        // === NetCDF 格式族 ===
        "NETCDF",     // 通用NetCDF格式
        "NC",         // NetCDF3格式
        "NC4",        // NetCDF4格式
        
        // === GDAL 栅格格式族 ===
        "GEOTIFF",    // GeoTIFF栅格格式
        "HDF5",       // HDF5栅格格式（GDAL支持）
        "GRIB",       // GRIB气象数据格式
        "GDAL_RASTER", // 通用GDAL栅格格式
        
        // === GDAL 矢量格式族 ===
        "SHAPEFILE",  // ESRI Shapefile
        "GEOJSON",    // GeoJSON矢量格式
        "KML",        // Google KML格式
        "GPX",        // GPS交换格式
        "GML",        // Geography Markup Language
        "WFS",        // Web Feature Service
        "GDAL_VECTOR" // 通用GDAL矢量格式
    };
    
    LOG_INFO("🔧 格式白名单已初始化 - 支持 {} 种格式", supportedFormats_.size());
}

// 🔧 新增：验证注册的格式是否都在白名单中
void ReaderRegistry::validateRegisteredFormats() {
    boost::shared_lock<boost::shared_mutex> lock(registryMutex_);
    
    std::vector<std::string> unsupportedFormats;
    
    for (const auto& [format, factory] : readerFactories_) {
        if (supportedFormats_.find(format) == supportedFormats_.end()) {
            unsupportedFormats.push_back(format);
        }
    }
    
    if (!unsupportedFormats.empty()) {
        LOG_ERROR("🚨 发现不在白名单中的注册格式:");
        for (const auto& format : unsupportedFormats) {
            LOG_ERROR("   ❌ 不支持的格式: {}", format);
        }
        throw std::runtime_error("注册的格式中包含不支持的格式");
    }
    
    LOG_INFO("✅ 所有注册格式都在支持白名单中");
}

// 🔧 新增：检查格式是否被真正支持
bool ReaderRegistry::isFormatTrulySupported(const std::string& format) const {
    if (format.empty()) {
        return false;
    }
    
    // 首先检查是否在白名单中
    if (supportedFormats_.find(format) == supportedFormats_.end()) {
        LOG_WARN("🚫 格式不在支持白名单中: {}", format);
        return false;
    }
    
    // 然后检查是否有对应的读取器工厂
    boost::shared_lock<boost::shared_mutex> lock(registryMutex_);
    bool hasFactory = readerFactories_.find(format) != readerFactories_.end();
    
    if (!hasFactory) {
        LOG_WARN("🚫 格式在白名单中但没有对应的读取器工厂: {}", format);
    }
    
    return hasFactory;
}

bool ReaderRegistry::registerReaderFactory(const std::string& format, ReaderFactory factory) {
    if (format.empty() || !factory) {
        LOG_ERROR("注册读取器工厂时参数无效: format={}, factory={}", format, static_cast<bool>(factory));
        return false;
    }
    
    // 🔧 新增：验证格式是否在支持白名单中
    if (supportedFormats_.find(format) == supportedFormats_.end()) {
        LOG_ERROR("🚫 尝试注册不支持的格式: {} - 格式不在白名单中", format);
        return false;
    }
    
    boost::unique_lock<boost::shared_mutex> lock(registryMutex_);
    
    auto result = readerFactories_.emplace(format, std::move(factory));
    if (result.second) {
        LOG_INFO("读取器工厂已注册: {}", format);
        return true;
    } else {
        LOG_WARN("读取器工厂已存在，覆盖注册: {}", format);
        result.first->second = std::move(factory);
        return true;
    }
}

bool ReaderRegistry::unregisterReaderFactory(const std::string& format) {
    if (format.empty()) {
        LOG_ERROR("取消注册读取器工厂时格式名为空");
        return false;
    }
    
    boost::unique_lock<boost::shared_mutex> lock(registryMutex_);
    
    auto it = readerFactories_.find(format);
    if (it != readerFactories_.end()) {
        readerFactories_.erase(it);
        LOG_INFO("读取器工厂已取消注册: {}", format);
        return true;
    } else {
        LOG_WARN("尝试取消注册不存在的读取器工厂: {}", format);
        return false;
    }
}

std::shared_ptr<UnifiedDataReader> ReaderRegistry::createReader(
    const std::string& filePath,
    const std::optional<std::string>& explicitFormat) {
    
    std::string targetFormat;
    
    if (explicitFormat) {
        targetFormat = *explicitFormat;
    } else {
        auto detectedFormat = detectFileFormat(filePath);
        if (!detectedFormat) {
            LOG_ERROR("🚫 无法检测文件格式: {}", filePath);
            return nullptr;
        }
        targetFormat = *detectedFormat;
    }
    
    // 🔧 新增：严格验证格式是否被真正支持
    if (!isFormatTrulySupported(targetFormat)) {
        LOG_ERROR("🚫 检测到不支持的文件格式: {} - 文件: {}", targetFormat, filePath);
        LOG_INFO("💡 支持的格式列表: {}", getSupportedFormatsString());
        return nullptr;
    }
    
    // 查找对应的工厂
    boost::shared_lock<boost::shared_mutex> lock(registryMutex_);
    auto it = readerFactories_.find(targetFormat);
    if (it == readerFactories_.end()) {
        LOG_ERROR("🚫 不支持的文件格式: {} - 文件: {}", targetFormat, filePath);
        return nullptr;
    }
    
    try {
        // 🔧 创建读取器 - 传递CommonServices参数
        LOG_DEBUG("✅ 创建 {} 格式读取器: {}", targetFormat, filePath);
        return it->second(filePath, commonServices_);
    } catch (const std::exception& e) {
        LOG_ERROR("❌ 创建读取器失败: {} - {} (格式: {})", filePath, e.what(), targetFormat);
        return nullptr;
    }
}

bool ReaderRegistry::supportsFormat(const std::string& format) const {
    if (format.empty()) {
        return false;
    }
    
    // 🔧 修改：使用严格的支持检查
    return isFormatTrulySupported(format);
}

std::vector<std::string> ReaderRegistry::getSupportedFormats() const {
    boost::shared_lock<boost::shared_mutex> lock(registryMutex_);
    
    std::vector<std::string> formats;
    formats.reserve(readerFactories_.size());
    
    for (const auto& pair : readerFactories_) {
        formats.push_back(pair.first);
    }
    
    std::sort(formats.begin(), formats.end());
    return formats;
}

// 🔧 新增：获取支持格式的字符串表示
std::string ReaderRegistry::getSupportedFormatsString() const {
    auto formats = getSupportedFormats();
    std::string result = "";
    for (size_t i = 0; i < formats.size(); ++i) {
        if (i > 0) result += ", ";
        result += formats[i];
    }
    return result;
}

std::optional<std::string> ReaderRegistry::detectFileFormat(const std::string& filePath) const {
    if (!formatDetector_) {
        LOG_ERROR("格式检测器为空");
        return std::nullopt;
    }
    
    try {
        auto result = formatDetector_->detectFormat(filePath);
        if (result.isValid()) {
            // 🔧 修复：先进行格式名称标准化
            std::string detectedFormat = result.formatName;
            std::string standardizedFormat = standardizeFormatName(detectedFormat);
            
            // 🔧 新增：检查置信度阈值 - 低置信度文件应被正确过滤
            constexpr double CONFIDENCE_THRESHOLD = 0.5;
            if (result.confidence < CONFIDENCE_THRESHOLD) {
                LOG_INFO("🚫 文件置信度过低，安全过滤: {} (置信度: {:.2f} < {:.2f}) - 文件: {}", 
                         standardizedFormat, result.confidence, CONFIDENCE_THRESHOLD, filePath);
                return std::nullopt; // 返回nullopt表示格式检测失败，这是正确的安全行为
            }
            
            // 检查是否在白名单中
            if (supportedFormats_.find(standardizedFormat) == supportedFormats_.end()) {
                LOG_WARN("🚫 检测到不支持的格式: {} (标准化后: {}) - 文件: {}", 
                         detectedFormat, standardizedFormat, filePath);
                return std::nullopt;
            }
            
            // 检查工厂是否存在
            boost::shared_lock<boost::shared_mutex> lock(registryMutex_);
            if (readerFactories_.find(standardizedFormat) == readerFactories_.end()) {
                LOG_WARN("🚫 格式在白名单中但没有对应的读取器工厂: {} - 文件: {}", 
                         standardizedFormat, filePath);
                return std::nullopt;
            }
            
            LOG_DEBUG("✅ 格式检测成功: {} -> {} - 文件: {}", 
                     detectedFormat, standardizedFormat, filePath);
            return standardizedFormat;
        }
        return std::nullopt;
    } catch (const std::exception& e) {
        LOG_ERROR("格式检测异常: {} - {}", filePath, e.what());
        return std::nullopt;
    }
}

// 🔧 新增：标准化格式名称，解决命名不一致问题
std::string ReaderRegistry::standardizeFormatName(const std::string& detectedFormat) const {
    // 格式名称映射表，解决FileFormatDetector和ReaderRegistry之间的命名不一致
    static const std::map<std::string, std::string> formatMappings = {
        // FileFormatDetector格式 -> ReaderRegistry标准格式
        {"JSON", "GEOJSON"},           // JSON文件映射到GEOJSON
        {"CSV", "UNSUPPORTED"},        // CSV格式不被支持，明确标记
        {"TIFF", "GEOTIFF"},          // 标准化TIFF名称
        {"NETCDF", "NETCDF"},         // 保持一致
        {"NetCDF", "NETCDF"},         // 🔧 修复：混合大小写到标准格式
        {"HDF5", "HDF5"},             // 保持一致
        {"SHAPEFILE", "SHAPEFILE"},   // 保持一致
        {"Shapefile", "SHAPEFILE"},   // 🔧 修复：混合大小写到标准格式
        {"GEOJSON", "GEOJSON"},       // 保持一致
        {"GeoJSON", "GEOJSON"},       // 🔧 修复：混合大小写到标准格式
        {"GEOTIFF", "GEOTIFF"},       // 保持一致
        {"GeoTIFF", "GEOTIFF"},       // 🔧 修复：混合大小写到标准格式
        {"GTiff", "GEOTIFF"},         // 🔧 新增：GDAL驱动名称映射
        {"GDAL_RASTER", "GDAL_RASTER"}, // 保持一致
        {"GDAL_VECTOR", "GDAL_VECTOR"}, // 保持一致
    };
    
    auto it = formatMappings.find(detectedFormat);
    if (it != formatMappings.end()) {
        std::string mappedFormat = it->second;
        if (mappedFormat == "UNSUPPORTED") {
            LOG_WARN("🚫 明确不支持的格式: {}", detectedFormat);
            return "UNSUPPORTED";
        }
        LOG_DEBUG("🔄 格式标准化: {} -> {}", detectedFormat, mappedFormat);
        return mappedFormat;
    }
    
    // 如果没有映射，返回原格式名（转为大写以保持一致性）
    std::string upperFormat = detectedFormat;
    std::transform(upperFormat.begin(), upperFormat.end(), upperFormat.begin(), ::toupper);
    
    LOG_DEBUG("🔄 格式标准化（转大写）: {} -> {}", detectedFormat, upperFormat);
    return upperFormat;
}

} // namespace oscean::core_services::data_access::readers 
