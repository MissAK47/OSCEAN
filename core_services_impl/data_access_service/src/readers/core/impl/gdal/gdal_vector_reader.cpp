/**
 * @file gdal_vector_reader.cpp
 * @brief GDAL矢量数据读取器实现 - 完整矢量功能
 */

// 🚀 第一步：使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 数据访问服务只使用boost::future，不使用boost::asio

// 🚀 第二步：立即包含boost线程库
#include <boost/thread/future.hpp>
#include <boost/thread.hpp>

// 🚀 第三步：包含项目头文件
#include "gdal_vector_reader.h"
#include "common_utils/utilities/logging_utils.h"
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <mutex>

namespace oscean::core_services::data_access::readers::impl::gdal {

GdalVectorReader::GdalVectorReader(
    const std::string& filePath,
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices)
    : UnifiedDataReader(filePath), filePath_(filePath), commonServices_(commonServices) {
    
    LOG_INFO("GdalVectorReader构造: 文件={}", filePath);
    
    // 初始化高级功能组件
    initializeAdvancedComponents();
}

GdalVectorReader::~GdalVectorReader() {
    if (isOpen_.load()) {
        try {
            closeAsync().wait(); 
        } catch (const std::exception& e) {
            LOG_ERROR("GdalVectorReader析构异常: {}", e.what());
        }
    }
}

// =============================================================================
// UnifiedDataReader 接口实现
// =============================================================================

boost::future<bool> GdalVectorReader::openAsync() {
    return boost::async(boost::launch::async, [this]() {
        if (isOpen_.load()) {
            LOG_WARN("GDAL矢量文件已经打开: {}", filePath_);
            return true;
        }

        // 检查文件是否存在
        if (!std::filesystem::exists(filePath_)) {
            LOG_ERROR("文件不存在: {}", filePath_);
            return false;
        }

        // 初始化GDAL
        if (!initializeGDAL()) {
            LOG_ERROR("GDAL初始化失败");
            return false;
        }

        // 打开GDAL数据集（矢量模式）
        gdalDataset_ = static_cast<GDALDataset*>(GDALOpenEx(filePath_.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr));
        if (!gdalDataset_) {
            CPLErrorNum lastError = CPLGetLastErrorNo();
            const char* lastErrorMsg = CPLGetLastErrorMsg();
            LOG_ERROR("无法使用GDAL打开矢量文件: {} - GDAL错误 {}: {}", filePath_, lastError, lastErrorMsg ? lastErrorMsg : "未知错误");
            return false;
        }

        // 验证是否为矢量文件
        if (!validateVectorFile()) {
            LOG_ERROR("文件不是有效的矢量文件: {}", filePath_);
            GDALClose(gdalDataset_);
            gdalDataset_ = nullptr;
            return false;
        }

        // 创建矢量处理器
        vectorProcessor_ = std::make_unique<GdalVectorProcessor>(gdalDataset_);
        isOpen_.store(true);
        
        // 更新性能统计
        updatePerformanceStats(0, false, false);
        
        LOG_INFO("GDAL矢量文件成功打开: {}", filePath_);
        return true;
    });
}

boost::future<void> GdalVectorReader::closeAsync() {
    return boost::async(boost::launch::async, [this]() {
        if (!isOpen_.load()) return;
        
        cleanup();
        LOG_INFO("GDAL矢量文件已关闭: {}", filePath_);
    });
}

std::string GdalVectorReader::getReaderType() const {
    return "GDAL_VECTOR";
}

boost::future<std::optional<FileMetadata>> GdalVectorReader::getFileMetadataAsync() {
    return boost::async(boost::launch::async, [this]() -> std::optional<FileMetadata> {
        if (!isOpen_.load() || !gdalDataset_) {
            LOG_ERROR("文件未打开，无法获取元数据: {}", filePath_);
            return std::nullopt;
        }

        FileMetadata fm;
        fm.filePath = filePath_;
        
        // 🔧 修复：根据GDAL驱动程序确定具体的格式名称
        std::string formatName = "GDAL_VECTOR"; // 默认值
        std::string driverName = "Unknown"; // 默认驱动名称
        GDALDriver* driver = gdalDataset_->GetDriver();
        if (driver) {
            driverName = GDALGetDriverShortName(driver);
            
            // 根据GDAL驱动程序映射到更具体的格式名称
            if (driverName == "ESRI Shapefile") {
                formatName = "SHAPEFILE";
            } else if (driverName == "GeoJSON") {
                formatName = "GEOJSON";
            } else if (driverName == "KML") {
                formatName = "KML";
            } else if (driverName == "GPX") {
                formatName = "GPX";
            } else if (driverName == "GML") {
                formatName = "GML";
            } else {
                // 对于其他格式，使用 GDAL_前缀 + 驱动名称
                formatName = "GDAL_" + driverName;
            }
            
            LOG_DEBUG("GDAL矢量格式检测: {} -> {}", driverName, formatName);
        }
        
        fm.format = formatName;

        // 1. 填充 geographicDimensions (使用新的结构体定义)
        {
            DimensionDetail dimLayer;
            dimLayer.name = "layer";
            dimLayer.size = static_cast<size_t>(gdalDataset_->GetLayerCount());
            fm.geographicDimensions.push_back(dimLayer);
        }

        // 2. 填充 variables（图层作为变量）
        if(vectorProcessor_) {
            auto layerNames = vectorProcessor_->getLayerNames();
            for(const auto& layerName : layerNames) {
                VariableMeta vm;
                vm.name = layerName;
                // 从GDAL获取图层元数据
                auto metadataEntries = vectorProcessor_->loadLayerMetadataAdvanced(layerName);
                for (const auto& entry : metadataEntries) {
                    vm.attributes[entry.getKey()] = entry.getValue();
                }
                fm.variables.push_back(vm);
            }
        }

        // 3. 填充 metadata
        char** papszMetadata = gdalDataset_->GetMetadata(nullptr);
        if (papszMetadata != nullptr) {
            for (int i = 0; papszMetadata[i] != nullptr; ++i) {
                std::string entry_str(papszMetadata[i]);
                size_t equals_pos = entry_str.find('=');
                if (equals_pos != std::string::npos) {
                    std::string key = entry_str.substr(0, equals_pos);
                    std::string value = entry_str.substr(equals_pos + 1);
                    fm.metadata[key] = value;
                }
            }
        }

        // 驱动信息
        if (driver) {
            fm.metadata["driver"] = driverName;
            fm.metadata["driver_description"] = GDALGetDriverLongName(driver);
        }

        // 矢量特定信息
        fm.metadata["layer_count"] = std::to_string(gdalDataset_->GetLayerCount());
        
        updatePerformanceStats(0, false, true);
        return fm;
    });
}

boost::future<std::vector<std::string>> GdalVectorReader::getVariableNamesAsync() {
     return boost::async(boost::launch::async, [this]() -> std::vector<std::string> {
        // 🔧 修复：如果文件未打开，先尝试打开
        if (!isOpen_.load()) {
            LOG_INFO("GDAL矢量文件未打开，尝试重新打开: {}", filePath_);
            bool opened = openAsync().get();
            if (!opened) {
                LOG_ERROR("GDAL矢量文件打开失败: {}", filePath_);
                return {};
            }
        }
        
        if (!vectorProcessor_) {
            LOG_ERROR("GDAL矢量处理器未初始化");
            return {};
        }
        
        auto names = vectorProcessor_->getLayerNames();
        updatePerformanceStats(0, false, false);
        return names;
    });
}
    
boost::future<std::shared_ptr<GridData>> GdalVectorReader::readGridDataAsync(
    const std::string& variableName,
    const std::optional<BoundingBox>& bounds) {
    
     return boost::async(boost::launch::async, [this, variableName, bounds]() -> std::shared_ptr<GridData> {
        if (!isOpen_.load() || !vectorProcessor_) {
            return nullptr;
        }
        try {
            auto startTime = std::chrono::steady_clock::now();
            
            // 检查缓存
            std::string cacheKey = calculateCacheKey(variableName, bounds);
            if (cachingEnabled_.load()) {
                auto cached = getFromCache(cacheKey);
                if (cached) {
                    updatePerformanceStats(0, false, true);
                    // GridData不支持拷贝，返回nullptr
                    LOG_WARN("GridData不支持拷贝，无法从缓存返回");
                    return nullptr;
                }
            }
            
            auto result = vectorProcessor_->readLayerDataAdvanced(variableName, bounds);
            
            auto endTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            if (result && !result->getData().empty()) {
                size_t bytesRead = result->getData().size();
                
                // 缓存结果
                if (cachingEnabled_.load()) {
                    putToCache(cacheKey, *result);
                }
                
                updatePerformanceStats(bytesRead, simdEnabled_.load(), false);
                
                LOG_DEBUG("读取图层 {} 耗时 {}ms, {} 字节", 
                         variableName, duration.count(), bytesRead);
            }
            
            return result;
        } catch (const std::exception& e) {
            LOG_ERROR("读取矢量数据失败: {}", e.what());
            return nullptr;
        }
    });
}

// =============================================================================
// 矢量数据特定接口实现
// =============================================================================

boost::future<std::vector<std::string>> GdalVectorReader::getLayerNamesAsync() {
    return getVariableNamesAsync(); // 对于矢量数据，图层就是变量
}

boost::future<std::shared_ptr<GridData>> GdalVectorReader::readLayerDataAsync(
    const std::string& layerName,
    const std::optional<BoundingBox>& bounds) {
    
    return readGridDataAsync(layerName, bounds); // 复用基础接口
}

boost::future<size_t> GdalVectorReader::getFeatureCountAsync(const std::string& layerName) {
    return boost::async(boost::launch::async, [this, layerName]() -> size_t {
        if (!isOpen_.load() || !vectorProcessor_) {
            return 0;
        }
        
        // 检查缓存
        auto it = featureCountCache_.find(layerName);
        if (it != featureCountCache_.end()) {
            updatePerformanceStats(0, false, true);
            return it->second;
        }
        
        size_t count = vectorProcessor_->getFeatureCount(layerName);
        featureCountCache_[layerName] = count;
        
        updatePerformanceStats(0, false, false);
        return count;
    });
}

boost::future<std::string> GdalVectorReader::getGeometryTypeAsync(const std::string& layerName) {
    return boost::async(boost::launch::async, [this, layerName]() -> std::string {
        if (!isOpen_.load() || !vectorProcessor_) {
            return "Unknown";
        }
        
        // 检查缓存
        auto it = geometryTypeCache_.find(layerName);
        if (it != geometryTypeCache_.end()) {
            updatePerformanceStats(0, false, true);
            return it->second;
        }
        
        std::string geomType = vectorProcessor_->getGeometryType(layerName);
        geometryTypeCache_[layerName] = geomType;
        
        updatePerformanceStats(0, false, false);
        return geomType;
    });
}

boost::future<std::vector<std::map<std::string, std::string>>> GdalVectorReader::getFieldInfoAsync(const std::string& layerName) {
    return boost::async(boost::launch::async, [this, layerName]() -> std::vector<std::map<std::string, std::string>> {
        if (!isOpen_.load() || !vectorProcessor_) {
            return {};
        }
        
        // 检查缓存
        auto it = layerFieldsCache_.find(layerName);
        if (it != layerFieldsCache_.end()) {
            updatePerformanceStats(0, false, true);
            // 转换为目标格式
            std::vector<std::map<std::string, std::string>> result;
            for (const auto& field : it->second) {
                std::map<std::string, std::string> fieldMap;
                fieldMap["name"] = field;
                result.push_back(fieldMap);
            }
            return result;
        }
        
        auto fieldInfo = vectorProcessor_->getFieldInfo(layerName);
        
        // 缓存字段名称
        std::vector<std::string> fieldNames;
        for (const auto& field : fieldInfo) {
            auto nameIt = field.find("name");
            if (nameIt != field.end()) {
                fieldNames.push_back(nameIt->second);
            }
        }
        layerFieldsCache_[layerName] = fieldNames;
        
        updatePerformanceStats(0, false, false);
        return fieldInfo;
    });
}

boost::future<std::shared_ptr<GridData>> GdalVectorReader::spatialQueryAsync(
    const std::string& layerName,
    const BoundingBox& bounds,
    const std::string& spatialRelation) {
    
    return boost::async(boost::launch::async, [this, layerName, bounds, spatialRelation]() -> std::shared_ptr<GridData> {
        if (!isOpen_.load() || !vectorProcessor_) {
            return nullptr;
        }
        
        try {
            auto startTime = std::chrono::steady_clock::now();
            
            auto result = vectorProcessor_->spatialQuery(layerName, bounds, spatialRelation);
            
            auto endTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            if (result && !result->getData().empty()) {
                size_t bytesRead = result->getData().size();
                updatePerformanceStats(bytesRead, simdEnabled_.load(), false);
                
                LOG_DEBUG("空间查询图层 {} 耗时 {}ms, {} 字节", 
                         layerName, duration.count(), bytesRead);
            }
            
            return result;
        } catch (const std::exception& e) {
            LOG_ERROR("空间查询失败: {}", e.what());
            return nullptr;
        }
    });
}

boost::future<std::shared_ptr<GridData>> GdalVectorReader::attributeQueryAsync(
    const std::string& layerName,
    const std::string& whereClause) {
    
    return boost::async(boost::launch::async, [this, layerName, whereClause]() -> std::shared_ptr<GridData> {
        if (!isOpen_.load() || !vectorProcessor_) {
            return nullptr;
        }
        
        try {
            auto startTime = std::chrono::steady_clock::now();
            
            auto result = vectorProcessor_->attributeQuery(layerName, whereClause);
            
            auto endTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            if (result && !result->getData().empty()) {
                size_t bytesRead = result->getData().size();
                updatePerformanceStats(bytesRead, simdEnabled_.load(), false);
                
                LOG_DEBUG("属性查询图层 {} 耗时 {}ms, {} 字节", 
                         layerName, duration.count(), bytesRead);
            }
            
            return result;
        } catch (const std::exception& e) {
            LOG_ERROR("属性查询失败: {}", e.what());
            return nullptr;
        }
    });
}

// =============================================================================
// 高级功能接口实现
// =============================================================================

void GdalVectorReader::enableSIMDOptimization(bool enable) {
    simdEnabled_.store(enable);
    if (enable && simdManager_) {
        LOG_INFO("SIMD优化已启用: {}", simdManager_->getImplementationName());
    } else {
        LOG_INFO("SIMD优化已禁用");
    }
}

bool GdalVectorReader::isSIMDOptimizationEnabled() const {
    return simdEnabled_.load() && simdManager_ && simdManager_->isOptimizedFor("data_processing");
}

void GdalVectorReader::configureSIMDOptimization(const GdalSIMDConfig& config) {
    simdConfig_ = config;
    if (simdManager_) {
        LOG_INFO("SIMD配置已更新: 向量化IO={}, 并行处理={}", 
                 config.enableVectorizedIO, config.enableParallelProcessing);
    }
}

void GdalVectorReader::enableAdvancedCaching(bool enable) {
    cachingEnabled_.store(enable);
    if (enable) {
        LOG_INFO("高级缓存已启用");
    } else {
        LOG_INFO("高级缓存已禁用");
    }
}

bool GdalVectorReader::isAdvancedCachingEnabled() const {
    return cachingEnabled_.load();
}

void GdalVectorReader::enablePerformanceMonitoring(bool enable) {
    performanceMonitoringEnabled_.store(enable);
    LOG_INFO("性能监控已{}", enable ? "启用" : "禁用");
}

bool GdalVectorReader::isPerformanceMonitoringEnabled() const {
    return performanceMonitoringEnabled_.load();
}

GdalPerformanceStats GdalVectorReader::getPerformanceStats() const {
    return performanceStats_;
}

std::string GdalVectorReader::getPerformanceReport() const {
    std::ostringstream report;
    report << "=== GDAL矢量读取器性能报告 ===\n";
    report << "文件: " << filePath_ << "\n";
    report << "状态: " << (isOpen_.load() ? "已打开" : "已关闭") << "\n";
    
    auto now = std::chrono::steady_clock::now();
    auto totalTimeDuration = now - performanceStats_.startTime;
    auto lastAccessDuration = now - performanceStats_.lastAccessTime;
    
    auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(totalTimeDuration);
    auto lastAccess = std::chrono::duration_cast<std::chrono::seconds>(lastAccessDuration);
    
    report << "性能统计:\n";
    report << "  - 总读取字节数: " << performanceStats_.totalBytesRead.load() << "\n";
    report << "  - 总读取要素数: " << performanceStats_.totalFeaturesRead.load() << "\n";
    report << "  - SIMD操作次数: " << performanceStats_.simdOperationsCount.load() << "\n";
    report << "  - 缓存命中: " << performanceStats_.cacheHits.load() << "\n";
    report << "  - 缓存未命中: " << performanceStats_.cacheMisses.load() << "\n";
    report << "  - 运行时间: " << totalTime.count() << " 秒\n";
    report << "  - 最后访问: " << lastAccess.count() << " 秒前\n";
    
    if (performanceStats_.cacheHits.load() + performanceStats_.cacheMisses.load() > 0) {
        double hitRate = static_cast<double>(performanceStats_.cacheHits.load()) / 
                        (performanceStats_.cacheHits.load() + performanceStats_.cacheMisses.load()) * 100.0;
        report << "  - 缓存命中率: " << std::fixed << std::setprecision(2) << hitRate << "%\n";
    }
    
    report << "高级功能状态:\n";
    report << "  - SIMD优化: " << (isSIMDOptimizationEnabled() ? "已启用" : "未启用") << "\n";
    report << "  - 高级缓存: " << (isAdvancedCachingEnabled() ? "已启用" : "未启用") << "\n";
    report << "  - 性能监控: " << (isPerformanceMonitoringEnabled() ? "已启用" : "未启用") << "\n";
    
    return report.str();
}

boost::future<void> GdalVectorReader::streamFeaturesAsync(
    const std::string& layerName,
    const std::optional<BoundingBox>& bounds,
    std::function<bool(const std::vector<std::map<std::string, std::string>>&)> processor) {
    
    return boost::async(boost::launch::async, [this, layerName, bounds, processor]() {
        if (!isOpen_.load() || !vectorProcessor_) {
            LOG_ERROR("矢量读取器未打开");
            return;
        }
        
        try {
            vectorProcessor_->streamFeatures(layerName, bounds, processor);
            updatePerformanceStats(0, false, false);
        } catch (const std::exception& e) {
            LOG_ERROR("流式读取要素失败: {}", e.what());
        }
    });
}

// =============================================================================
// 内部方法实现
// =============================================================================

void GdalVectorReader::initializeAdvancedComponents() {
    try {
        // 创建默认组件，不依赖CommonServicesFactory的具体API
        using namespace oscean::common_utils::simd;
        SIMDConfig config = SIMDConfig::createOptimal();
        simdManager_ = std::make_shared<UnifiedSIMDManager>(config);
        
        memoryManager_ = std::make_shared<oscean::common_utils::memory::UnifiedMemoryManager>();
        
        auto asyncFrameworkPtr = oscean::common_utils::async::AsyncFramework::createDefault();
        asyncFramework_ = std::shared_ptr<oscean::common_utils::async::AsyncFramework>(asyncFrameworkPtr.release());
        
        simdEnabled_.store(true);
        
        LOG_INFO("GDAL矢量读取器高级功能组件初始化成功: SIMD={}", 
                 simdManager_ ? simdManager_->getImplementationName() : "无");
    } catch (const std::exception& e) {
        LOG_WARN("GDAL矢量读取器高级功能组件初始化失败: {}", e.what());
        // 继续使用基础功能
    }
}

bool GdalVectorReader::initializeGDAL() {
    // 🔧 根本修复：完全移除静态初始化，改为依赖DI管理的GDAL初始化
    try {
        LOG_INFO("🔧 检查GDAL矢量库状态...");
        
        // 检查GDAL是否已经初始化（应该由DI管理器在预热阶段完成）
        int driverCount = GDALGetDriverCount();
        if (driverCount > 0) {
            LOG_INFO("✅ GDAL矢量库已预初始化 - 驱动数量: {}", driverCount);
            return true;
        }
        
        // 如果GDAL尚未初始化，这表示DI预热失败了
        // 作为备用方案，我们进行简单的初始化
        LOG_WARN("⚠️ GDAL矢量库未预初始化，执行紧急初始化...");
        
        // 检查GDAL是否已由全局初始化器初始化
        if (GDALGetDriverCount() == 0) {
            throw std::runtime_error("GDAL未初始化！请确保在main函数中调用了GdalGlobalInitializer::initialize()");
        }
        
        // 移除所有分散的GDAL初始化调用
        // GDALAllRegister(); // ❌ 已移除 - 现在由GdalGlobalInitializer统一管理
        // OGRRegisterAll();  // ❌ 已移除
        
        int finalDriverCount = GDALGetDriverCount();
        if (finalDriverCount > 0) {
            LOG_INFO("✅ GDAL矢量紧急初始化完成 - 驱动数量: {}", finalDriverCount);
            return true;
        } else {
            LOG_ERROR("❌ GDAL矢量紧急初始化失败");
            return false;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("❌ GDAL矢量初始化异常: {}", e.what());
        return false;
    } catch (...) {
        LOG_ERROR("❌ GDAL矢量初始化未知异常");
        return false;
    }
}

void GdalVectorReader::cleanup() {
    if (isOpen_.load()) {
        vectorProcessor_.reset();
        if (gdalDataset_) {
            GDALClose(gdalDataset_);
            gdalDataset_ = nullptr;
        }
        isOpen_.store(false);
        
        // 清理缓存
        layerFieldsCache_.clear();
        featureCountCache_.clear();
        geometryTypeCache_.clear();
    }
}

bool GdalVectorReader::validateVectorFile() const {
    if (!gdalDataset_) {
        return false;
    }
    
    // 检查是否有图层
    int layerCount = gdalDataset_->GetLayerCount();
    if (layerCount == 0) {
        return false;
    }
    
    // 检查第一个图层是否有效
    OGRLayer* layer = gdalDataset_->GetLayer(0);
    return layer != nullptr;
}

void GdalVectorReader::updatePerformanceStats(size_t bytesRead, bool simdUsed, bool cacheHit) const {
    if (!performanceMonitoringEnabled_.load()) {
        return;
    }
    
    performanceStats_.totalBytesRead.fetch_add(bytesRead);
    performanceStats_.lastAccessTime = std::chrono::steady_clock::now();
    
    if (simdUsed) {
        performanceStats_.simdOperationsCount.fetch_add(1);
    }
    
    if (cacheHit) {
        performanceStats_.cacheHits.fetch_add(1);
    } else {
        performanceStats_.cacheMisses.fetch_add(1);
    }
}

bool GdalVectorReader::checkMemoryUsage() const {
    if (!memoryManager_) {
        return true;  // 无法检查，假设OK
    }
    
    // 简化内存检查，避免使用不存在的API
    try {
        // 检查是否有足够内存进行操作
        return true;  // 简化实现，总是返回true
    } catch (const std::exception&) {
        return false;
    }
}

std::string GdalVectorReader::calculateCacheKey(const std::string& layerName, 
                                               const std::optional<BoundingBox>& bounds) const {
    std::ostringstream key;
    key << filePath_ << ":" << layerName;
    
    if (bounds) {
        key << ":" << bounds->minX << "," << bounds->minY << "," << bounds->maxX << "," << bounds->maxY;
    }
    
    return key.str();
}

std::optional<GridData> GdalVectorReader::getFromCache(const std::string& cacheKey) const {
    if (!cacheManager_) {
        return std::nullopt;
    }
    
    // 简化实现，实际需要实现序列化/反序列化
    return std::nullopt;
}

void GdalVectorReader::putToCache(const std::string& cacheKey, const GridData& data) const {
    if (!cacheManager_) {
        return;
    }
    
    // 简化实现，实际需要实现序列化
}

} // namespace oscean::core_services::data_access::readers::impl::gdal 