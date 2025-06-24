/**
 * @file gdal_raster_reader.cpp
 * @brief GDAL栅格读取器实现 - 完整统一架构
 */

#include "gdal_raster_reader.h"
#include "gdal_raster_processor.h"
#include "common_utils/utilities/logging_utils.h"
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <mutex>
#include <shared_mutex>

namespace oscean::core_services::data_access::readers::impl::gdal {

GdalRasterReader::GdalRasterReader(
    const std::string& filePath,
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices)
    : UnifiedDataReader(filePath), filePath_(filePath), commonServices_(commonServices) {
    
    LOG_INFO("GdalRasterReader构造: 文件={}", filePath);
    
    // 初始化高级功能组件
    initializeAdvancedComponents();
}

GdalRasterReader::~GdalRasterReader() {
    if (isOpen_.load()) {
        try {
            closeAsync().wait(); 
        } catch (const std::exception& e) {
            LOG_ERROR("GdalRasterReader析构异常: {}", e.what());
        }
    }
}

// =============================================================================
// UnifiedDataReader 接口实现
// =============================================================================

boost::future<bool> GdalRasterReader::openAsync() {
    return boost::async(boost::launch::async, [this]() -> bool {
        if (isOpen_.load()) {
            LOG_WARN("GDAL文件已经打开: {}", filePath_);
            return true;
        }

        // 检查文件是否存在
        if (!std::filesystem::exists(filePath_)) {
            LOG_ERROR("文件不存在: {}", filePath_);
            return false;
        }

        // 🔧 新增：检查文件大小并应用大文件优化
        std::error_code ec;
        auto fileSize = std::filesystem::file_size(filePath_, ec);
        if (ec) {
            LOG_ERROR("无法获取文件大小: {} - {}", filePath_, ec.message());
            return false;
        }
        
        const size_t MEDIUM_FILE_THRESHOLD = 1 * 1024 * 1024 * 1024ULL;  // 1GB
        const size_t LARGE_FILE_THRESHOLD = 5 * 1024 * 1024 * 1024ULL;   // 5GB
        const size_t HUGE_FILE_THRESHOLD = 10 * 1024 * 1024 * 1024ULL;   // 10GB
        const size_t ULTRA_FILE_THRESHOLD = 15 * 1024 * 1024 * 1024ULL;  // 15GB
        
        if (fileSize > MEDIUM_FILE_THRESHOLD) {
            LOG_INFO("检测到中等大小文件: {} (大小: {:.2f} GB)", filePath_, static_cast<double>(fileSize) / (1024*1024*1024));
            
            // 🔧 修复：使用线程本地配置(CPLSetThreadLocalConfigOption)替代全局配置(CPLSetConfigOption)
            // 避免多线程环境下，不同服务（如DataAccess和CrsService）之间的GDAL配置冲突
            CPLSetThreadLocalConfigOption("GDAL_CACHEMAX", "1024");  // 1GB缓存
            CPLSetThreadLocalConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR");
            
            if (fileSize > LARGE_FILE_THRESHOLD) {
                LOG_INFO("检测到大文件: {} (大小: {:.2f} GB)", filePath_, static_cast<double>(fileSize) / (1024*1024*1024));
                
                // 大文件优化（5-10GB）
                CPLSetThreadLocalConfigOption("GDAL_CACHEMAX", "2048");  // 2GB缓存
                CPLSetThreadLocalConfigOption("VSI_CACHE", "TRUE");
                CPLSetThreadLocalConfigOption("VSI_CACHE_SIZE", "1000000000"); // 1GB VSI缓存
                CPLSetThreadLocalConfigOption("GTIFF_DIRECT_IO", "YES");  // 启用直接I/O
                
                if (fileSize > HUGE_FILE_THRESHOLD) {
                    LOG_WARN("检测到超大文件: {} (大小: {:.2f} GB)", filePath_, static_cast<double>(fileSize) / (1024*1024*1024));
                    
                    // 超大文件优化（10-15GB）
                    CPLSetThreadLocalConfigOption("GDAL_CACHEMAX", "3072");  // 3GB缓存
                    CPLSetThreadLocalConfigOption("VSI_CACHE_SIZE", "2000000000"); // 2GB VSI缓存
                    CPLSetThreadLocalConfigOption("GDAL_TIFF_INTERNAL_MASK", "YES");
                    
                    if (fileSize > ULTRA_FILE_THRESHOLD) {
                        LOG_ERROR("检测到超大文件: {} (大小: {:.2f} GB) - 可能超出系统处理能力", 
                                 filePath_, static_cast<double>(fileSize) / (1024*1024*1024));
                        
                        // 极大文件保守策略（>15GB）
                        CPLSetThreadLocalConfigOption("GDAL_CACHEMAX", "4096");  // 4GB缓存
                        CPLSetThreadLocalConfigOption("VSI_CACHE_SIZE", "3000000000"); // 3GB VSI缓存
                        CPLSetThreadLocalConfigOption("GTIFF_VIRTUAL_MEM_IO", "NO");   // 禁用虚拟内存I/O，避免不稳定
                        CPLSetThreadLocalConfigOption("GDAL_SWATH_SIZE", "67108864");   // 64MB分块大小
                        CPLSetThreadLocalConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", "1024"); // 优化概览图块大小
                    }
                }
            }
        }

        // 初始化GDAL
        if (!initializeGDAL()) {
            LOG_ERROR("GDAL初始化失败");
            return false;
        }

        // 打开GDAL数据集
        LOG_INFO("正在使用GDAL打开文件: {} (大小: {:.2f} GB)", filePath_, static_cast<double>(fileSize) / (1024*1024*1024));
        gdalDataset_ = static_cast<GDALDataset*>(GDALOpen(filePath_.c_str(), GA_ReadOnly));
        if (!gdalDataset_) {
            // 🔧 新增：提供更详细的错误信息
            CPLErrorNum lastError = CPLGetLastErrorNo();
            const char* lastErrorMsg = CPLGetLastErrorMsg();
            LOG_ERROR("无法使用GDAL打开文件: {} - GDAL错误 {}: {}", filePath_, lastError, lastErrorMsg ? lastErrorMsg : "未知错误");
            
            // 重置GDAL配置选项
            if (fileSize > MEDIUM_FILE_THRESHOLD) {
                CPLSetThreadLocalConfigOption("GDAL_CACHEMAX", nullptr);
                CPLSetThreadLocalConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", nullptr);
                if (fileSize > LARGE_FILE_THRESHOLD) {
                    CPLSetThreadLocalConfigOption("VSI_CACHE", nullptr);
                    CPLSetThreadLocalConfigOption("VSI_CACHE_SIZE", nullptr);
                    CPLSetThreadLocalConfigOption("GTIFF_DIRECT_IO", nullptr);
                    if (fileSize > HUGE_FILE_THRESHOLD) {
                        CPLSetThreadLocalConfigOption("GDAL_TIFF_INTERNAL_MASK", nullptr);
                        if (fileSize > ULTRA_FILE_THRESHOLD) {
                            CPLSetThreadLocalConfigOption("GTIFF_VIRTUAL_MEM_IO", nullptr);
                            CPLSetThreadLocalConfigOption("GDAL_SWATH_SIZE", nullptr);
                            CPLSetThreadLocalConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", nullptr);
                        }
                    }
                }
            }
            return false;
        }

        if (gdalDataset_->GetRasterCount() == 0) {
            LOG_ERROR("文件不包含栅格数据: {}", filePath_);
            GDALClose(gdalDataset_);
            gdalDataset_ = nullptr;
            return false;
        }

        // 创建栅格处理器
        rasterProcessor_ = std::make_unique<GdalRasterProcessor>(gdalDataset_);
        isOpen_.store(true);
        
        // 更新性能统计
        updatePerformanceStats(0, false, false);
        
        LOG_INFO("GDAL栅格文件成功打开: {} (包含 {} 个波段)", filePath_, gdalDataset_->GetRasterCount());
        return true;
    });
}

boost::future<void> GdalRasterReader::closeAsync() {
    return boost::async(boost::launch::async, [this]() {
        if (!isOpen_.load()) return;
        
        cleanup();
        LOG_INFO("GDAL栅格文件已关闭: {}", filePath_);
    });
}

std::string GdalRasterReader::getReaderType() const {
    return "GDAL_RASTER";
}

boost::future<std::optional<FileMetadata>> GdalRasterReader::getFileMetadataAsync() {
    return boost::async(boost::launch::async, [this]() -> std::optional<FileMetadata> {
        if (!isOpen_.load() || !gdalDataset_) {
            LOG_ERROR("文件未打开，无法获取元数据: {}", filePath_);
            return std::nullopt;
        }

        FileMetadata fm;
        fm.filePath = filePath_;
        
        // 🔧 修复：根据GDAL驱动程序确定具体的格式名称
        std::string formatName = "GDAL_RASTER"; // 默认值
        std::string driverName = "Unknown"; // 默认驱动名称
        GDALDriver* driver = gdalDataset_->GetDriver();
        if (driver) {
            driverName = GDALGetDriverShortName(driver);
            
            // 根据GDAL驱动程序映射到更具体的格式名称
            if (driverName == "GTiff") {
                formatName = "GEOTIFF";
            } else if (driverName == "HDF5") {
                formatName = "HDF5";
            } else if (driverName == "GRIB") {
                formatName = "GRIB";
            } else if (driverName == "NetCDF") {
                formatName = "NETCDF";
            } else {
                // 对于其他格式，使用 GDAL_前缀 + 驱动名称
                formatName = "GDAL_" + driverName;
            }
            
            LOG_DEBUG("GDAL栅格格式检测: {} -> {}", driverName, formatName);
        }
        
        fm.format = formatName;

        // 1. 填充 geographicDimensions (使用新的结构体定义)
        {
            DimensionDetail dimX;
            dimX.name = "x";
            dimX.size = static_cast<size_t>(gdalDataset_->GetRasterXSize());
            fm.geographicDimensions.push_back(dimX);
        }
        {
            DimensionDetail dimY;
            dimY.name = "y";
            dimY.size = static_cast<size_t>(gdalDataset_->GetRasterYSize());
            fm.geographicDimensions.push_back(dimY);
        }
        {
            DimensionDetail dimBand;
            dimBand.name = "band";
            dimBand.size = static_cast<size_t>(gdalDataset_->GetRasterCount());
            fm.geographicDimensions.push_back(dimBand);
        }

        // 2. 填充 variables
        if(rasterProcessor_) {
            auto varNames = rasterProcessor_->getVariableNames();
            for(const auto& name : varNames) {
                VariableMeta vm;
                vm.name = name;
                // 从GDAL获取变量元数据
                auto metadataEntries = rasterProcessor_->loadBandMetadataAdvanced(name);
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

        // 地理变换信息
        double geoTransform[6];
        if (gdalDataset_->GetGeoTransform(geoTransform) == CE_None) {
            fm.metadata["has_geotransform"] = "true";
            fm.metadata["pixel_size_x"] = std::to_string(geoTransform[1]);
            fm.metadata["pixel_size_y"] = std::to_string(std::abs(geoTransform[5]));
        }

        // 投影信息
        const char* projWKT = gdalDataset_->GetProjectionRef();
        if (projWKT && strlen(projWKT) > 0) {
            fm.metadata["projection_wkt"] = projWKT;
        }
        
        updatePerformanceStats(0, false, true);
        return fm;
    });
}

boost::future<std::vector<std::string>> GdalRasterReader::getVariableNamesAsync() {
     return boost::async(boost::launch::async, [this]() -> std::vector<std::string> {
        // 🔧 修复：如果文件未打开，先尝试打开
        if (!isOpen_.load()) {
            LOG_INFO("GDAL栅格文件未打开，尝试重新打开: {}", filePath_);
            bool opened = openAsync().get();
            if (!opened) {
                LOG_ERROR("GDAL栅格文件打开失败: {}", filePath_);
                return {};
            }
        }
        
        if (!rasterProcessor_) {
            LOG_ERROR("GDAL栅格处理器未初始化");
            return {};
        }
        
        auto names = rasterProcessor_->getVariableNames();
        updatePerformanceStats(0, false, false);
        return names;
    });
}
    
boost::future<std::shared_ptr<GridData>> GdalRasterReader::readGridDataAsync(
    const std::string& variableName,
    const std::optional<BoundingBox>& bounds) {
    
     return boost::async(boost::launch::async, [this, variableName, bounds]() -> std::shared_ptr<GridData> {
        if (!isOpen_.load() || !rasterProcessor_) {
            return nullptr;
        }
        try {
            auto startTime = std::chrono::steady_clock::now();
            
            auto result = rasterProcessor_->readRasterDataAdvanced(variableName, bounds);
            
            auto endTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
                    if (result && !result->getData().empty()) {
            size_t bytesRead = result->getData().size();
                updatePerformanceStats(bytesRead, simdEnabled_.load(), false);
                
                LOG_DEBUG("读取变量 {} 耗时 {}ms, {} 字节", 
                         variableName, duration.count(), bytesRead);
            }
            
            return result;
        } catch (const std::exception& e) {
            LOG_ERROR("读取网格数据失败: {}", e.what());
            return nullptr;
        }
    });
}

// =============================================================================
// 高级功能接口实现
// =============================================================================

void GdalRasterReader::enableSIMDOptimization(bool enable) {
    simdEnabled_.store(enable);
    if (enable && simdManager_) {
        LOG_INFO("SIMD优化已启用: {}", simdManager_->getImplementationName());
    } else {
        LOG_INFO("SIMD优化已禁用");
    }
}

bool GdalRasterReader::isSIMDOptimizationEnabled() const {
    return simdEnabled_.load() && simdManager_ && simdManager_->isOptimizedFor("data_processing");
}

void GdalRasterReader::configureSIMDOptimization(const GdalSIMDConfig& config) {
    simdConfig_ = config;
    if (simdManager_) {
        LOG_INFO("SIMD配置已更新: 向量化IO={}, 并行处理={}", 
                 config.enableVectorizedIO, config.enableParallelProcessing);
    }
}

void GdalRasterReader::enableAdvancedCaching(bool enable) {
    cachingEnabled_.store(enable);
    if (enable) {
        LOG_INFO("高级缓存已启用");
    } else {
        LOG_INFO("高级缓存已禁用");
    }
}

bool GdalRasterReader::isAdvancedCachingEnabled() const {
    return cachingEnabled_.load();
}

void GdalRasterReader::enablePerformanceMonitoring(bool enable) {
    performanceMonitoringEnabled_.store(enable);
    LOG_INFO("性能监控已{}", enable ? "启用" : "禁用");
}

bool GdalRasterReader::isPerformanceMonitoringEnabled() const {
    return performanceMonitoringEnabled_.load();
}

GdalPerformanceStats GdalRasterReader::getPerformanceStats() const {
    return performanceStats_;
}

std::string GdalRasterReader::getPerformanceReport() const {
    std::ostringstream report;
    report << "=== GDAL栅格读取器性能报告 ===\n";
    report << "文件: " << filePath_ << "\n";
    report << "状态: " << (isOpen_.load() ? "已打开" : "已关闭") << "\n";
    
    auto now = std::chrono::steady_clock::now();
    auto totalTimeDuration = now - performanceStats_.startTime;
    auto lastAccessDuration = now - performanceStats_.lastAccessTime;
    
    auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(totalTimeDuration);
    auto lastAccess = std::chrono::duration_cast<std::chrono::seconds>(lastAccessDuration);
    
    report << "性能统计:\n";
    report << "  - 总读取字节数: " << performanceStats_.totalBytesRead.load() << "\n";
    report << "  - 总读取波段数: " << performanceStats_.totalBandsRead.load() << "\n";
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

// =============================================================================
// 内部方法实现 - 更新为使用预热管理器
// =============================================================================

void GdalRasterReader::initializeAdvancedComponents() {
    try {
        // 🔧 修复：使用依赖注入的组件而不是全局单例
        if (commonServices_) {
            // 从依赖注入的 CommonServicesFactory 获取组件
            simdManager_ = std::dynamic_pointer_cast<oscean::common_utils::simd::UnifiedSIMDManager>(
                commonServices_->getSIMDManager());
            memoryManager_ = std::dynamic_pointer_cast<oscean::common_utils::memory::UnifiedMemoryManager>(
                commonServices_->getMemoryManager());
            
            // 注意：CommonServicesFactory 提供的是 IAsyncExecutor 而不是 AsyncFramework
            // 我们暂时设为 nullptr，或者使用直接创建的方式
            asyncFramework_ = nullptr;
            
            if (simdManager_ && memoryManager_) {
                simdEnabled_.store(true);
                LOG_INFO("✅ GDAL高级功能组件分配成功: SIMD={}", 
                         simdManager_ ? simdManager_->getImplementationName() : "无");
            } else {
                simdEnabled_.store(false);
                LOG_WARN("⚠️ 部分高级组件不可用，使用基础功能");
            }
        } else {
            // 如果没有 CommonServicesFactory，创建默认组件
            LOG_INFO("🔧 CommonServicesFactory不可用，创建默认组件...");
            
            using namespace oscean::common_utils::simd;
            SIMDConfig config = SIMDConfig::createOptimal();
            simdManager_ = std::make_shared<UnifiedSIMDManager>(config);
            
            memoryManager_ = std::make_shared<oscean::common_utils::memory::UnifiedMemoryManager>();
            
            // 对于AsyncFramework，暂时设为nullptr或者创建默认实例
            asyncFramework_ = nullptr;
            
            simdEnabled_.store(true);
            LOG_INFO("✅ 默认GDAL高级功能组件创建成功");
        }
        
    } catch (const std::exception& e) {
        LOG_WARN("⚠️ GDAL高级功能组件初始化异常: {}", e.what());
        simdEnabled_.store(false);
    }
}

bool GdalRasterReader::initializeGDAL() {
    // 🔧 根本修复：完全移除静态初始化，改为依赖DI管理的GDAL初始化
    try {
        LOG_INFO("🔧 检查GDAL库状态...");
        
        // 检查GDAL是否已经初始化（应该由DI管理器在预热阶段完成）
        int driverCount = GDALGetDriverCount();
        if (driverCount > 0) {
            LOG_INFO("✅ GDAL库已预初始化 - 驱动数量: {}", driverCount);
            return true;
        }
        
        // 如果GDAL尚未初始化，这表示DI预热失败了
        // 作为备用方案，我们进行简单的初始化
        LOG_WARN("⚠️ GDAL库未预初始化，执行紧急初始化...");
        
        // 检查GDAL是否已由全局初始化器初始化
        if (GDALGetDriverCount() == 0) {
            throw std::runtime_error("GDAL未初始化！请确保在main函数中调用了GdalGlobalInitializer::initialize()");
        }
        
        // 移除所有分散的GDAL初始化和全局配置调用
        // CPLSetConfigOption("GDAL_NUM_THREADS", "1"); // ❌ 已移除
        // CPLSetConfigOption("GDAL_CACHEMAX", "256");  // ❌ 已移除
        // GDALAllRegister(); // ❌ 已移除 - 现在由GdalGlobalInitializer统一管理
        
        int finalDriverCount = GDALGetDriverCount();
        if (finalDriverCount > 0) {
            LOG_INFO("✅ GDAL紧急初始化完成 - 驱动数量: {}", finalDriverCount);
            return true;
        } else {
            LOG_ERROR("❌ GDAL紧急初始化失败");
            return false;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("❌ GDAL初始化异常: {}", e.what());
        return false;
    } catch (...) {
        LOG_ERROR("❌ GDAL初始化未知异常");
        return false;
    }
}

void GdalRasterReader::cleanup() {
    if (isOpen_.load()) {
        rasterProcessor_.reset();
        if (gdalDataset_) {
            GDALClose(gdalDataset_);
            gdalDataset_ = nullptr;
        }
        isOpen_.store(false);
    }
}

void GdalRasterReader::updatePerformanceStats(size_t bytesRead, bool simdUsed, bool cacheHit) const {
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

bool GdalRasterReader::checkMemoryUsage() const {
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

void GdalRasterReader::optimizeReadParameters(size_t& blockXSize, size_t& blockYSize, int& bufferType) const {
    // 默认参数
    blockXSize = 512;
    blockYSize = 512;
    bufferType = GDT_Float64;
    
    if (simdConfig_.enableVectorizedIO) {
        // 调整块大小以适应SIMD
        size_t minSIMDBlockSize = simdConfig_.vectorSize / 8;  // 假设double类型
        blockXSize = std::max(blockXSize, minSIMDBlockSize);
        blockYSize = std::max(blockYSize, minSIMDBlockSize);
    }
    
    if (simdConfig_.chunkSize > 0) {
        size_t elementsPerChunk = simdConfig_.chunkSize / sizeof(double);
        size_t optimalBlockSize = static_cast<size_t>(std::sqrt(elementsPerChunk));
        blockXSize = std::min(blockXSize, optimalBlockSize);
        blockYSize = std::min(blockYSize, optimalBlockSize);
    }
}

} // namespace oscean::core_services::data_access::readers::impl::gdal 