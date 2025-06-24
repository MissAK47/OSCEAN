#include "transformation_cache_pimpl.h"

#include <stdexcept> // For std::runtime_error
#include <utility> // For std::pair
#include <shared_mutex> // Include for std::shared_mutex
#include <vector>      // Include for std::vector if needed for pabSuccess management
#include <cstdio> // For sscanf_s or sscanf
#include <cstring> // For strncmp

// 包含所需的 OGR 头文件
#include <ogr_spatialref.h>
#include <ogr_core.h>       // For OGRERR_NONE etc.
#include <cpl_error.h>      // For CPLGetLastErrorMsg, CPLSetConfigOption

// 包含日志头文件
#include "common_utils/utilities/logging_utils.h"

namespace oscean {
namespace core_services {
namespace crs {
namespace cache {

// 修改后的销毁函数
void SafeDestroyTransformation(OGRCoordinateTransformation* poCT) {
    if (poCT) {
        // Directly destroy the transformation object
        OGRCoordinateTransformation::DestroyCT(poCT);
    }
}

TransformationCacheImpl::TransformationCacheImpl() {
    // 使用正确的函数获取模块日志记录器
    _logger = oscean::common_utils::getModuleLogger("crs_service"); 
    if (!_logger) {
        // 如果无法获取 logger，这通常表示 LoggingManager 未初始化或创建失败
        // 记录到全局默认 logger 或 spdlog 的默认 logger
        spdlog::error("Failed to get module logger 'crs_service' in TransformationCacheImpl. Using default logger as fallback.");
        // 作为后备，尝试使用全局默认 logger（如果 LoggingManager 初始化了的话）
        _logger = oscean::common_utils::getLogger(); 
        if(!_logger) { // 如果全局 logger 也获取失败，使用 spdlog 默认
             _logger = spdlog::default_logger();
        }
    }
    _logger->info("TransformationCacheImpl initialized.");
}

TransformationCacheImpl::~TransformationCacheImpl() {
    _logger->info("Cleaning up TransformationCacheImpl...");
    // Unique lock for write access
    std::unique_lock<std::mutex> lock(_mutex);
    for (auto const& [key, val] : _transformationCache) {
        _logger->debug("Destroying cached transformation for key: {}", key);
        SafeDestroyTransformation(val); // Use the simplified function
    }
    _transformationCache.clear();
    _logger->info("TransformationCacheImpl cleaned up.");
}

bool TransformationCacheImpl::getTransformation(
    const std::string& sourceCRS,
    const std::string& targetCRS,
    OGRCoordinateTransformation** transformationOut)
{
    if (!transformationOut) {
        _logger->error("transformationOut parameter cannot be null.");
        return false;
    }
    *transformationOut = nullptr; // Initialize output parameter

    if (sourceCRS.empty() || targetCRS.empty()) {
        _logger->warn("Source or target CRS string is empty.");
        return false;
    }

    std::string cacheKey = createCacheKey(sourceCRS, targetCRS);
    _logger->debug("Looking for transformation with key: {}", cacheKey);

    // Use unique_lock for potential write access later
    std::unique_lock<std::mutex> lock(_mutex);

    auto it = _transformationCache.find(cacheKey);
    if (it != _transformationCache.end()) {
        _logger->debug("Transformation found in cache for key: {}", cacheKey);
        *transformationOut = it->second;
        return true;
    }

    // Not in cache, need to create (unlock mutex while creating)
    _logger->debug("Transformation not found in cache. Creating new one for key: {}", cacheKey);
    lock.unlock(); // Unlock before potentially long-running operation

    OGRCoordinateTransformation* newTransform = createTransformation(sourceCRS, targetCRS);

    lock.lock(); // Re-lock before accessing cache again

    if (newTransform) {
        _logger->info("Successfully created transformation for key: {}", cacheKey);
        // Double-check if another thread created it while we were unlocked
        it = _transformationCache.find(cacheKey);
        if (it != _transformationCache.end()) {
             _logger->warn("Transformation for key {} was created concurrently by another thread. Using existing.", cacheKey);
            // Destroy the newly created one as it's redundant
             lock.unlock(); // Unlock before destroying
             SafeDestroyTransformation(newTransform); // Use simplified function
             lock.lock();   // Relock
             *transformationOut = it->second;

        } else {
             // Store the newly created transformation in the cache
            _transformationCache[cacheKey] = newTransform;
            *transformationOut = newTransform;
        }
        return true;
    } else {
        _logger->error("Failed to create transformation between '{}' and '{}'", sourceCRS, targetCRS);
        return false;
    }
}

// Release function might not be strictly needed if cache owns lifetime.
// Implementing as no-op for now, could be used for reference counting later if needed.
void TransformationCacheImpl::releaseTransformation(OGRCoordinateTransformation* /*transformation*/) {
     _logger->trace("TransformationCacheImpl::releaseTransformation called (currently no-op).");
    // In a simple ownership model where the cache owns the objects,
    // releasing is handled by the cache cleanup (clear or destructor).
    // If reference counting were used, this would decrement the count.
}

void TransformationCacheImpl::clear() {
    _logger->info("Clearing transformation cache...");
    std::lock_guard<std::mutex> lock(_mutex);

    // Destroy the transformation objects before clearing the map
    for (auto const& [key, transform] : _transformationCache) {
        if (transform != nullptr) {
            SafeDestroyTransformation(transform); // Use the simplified function
        }
    }
    _transformationCache.clear();
    _logger->info("Transformation cache cleared.");
}

OGRCoordinateTransformation* TransformationCacheImpl::createTransformation(
    const std::string& sourceCRS,
    const std::string& targetCRS)
{
    OGRSpatialReference oSourceSRS, oTargetSRS;
    OGRCoordinateTransformation *poCT = nullptr;

    // --- 尝试优先使用 EPSG 代码 ---
    int nSourceEPSG = 0;
    if (strncmp(sourceCRS.c_str(), "EPSG:", 5) == 0) {
        // 使用 sscanf_s (Windows) 或保持 sscanf (跨平台)
        #ifdef _WIN32
        sscanf_s(sourceCRS.c_str() + 5, "%d", &nSourceEPSG);
        #else
        sscanf(sourceCRS.c_str() + 5, "%d", &nSourceEPSG);
        #endif
    }
    int nTargetEPSG = 0;
    if (strncmp(targetCRS.c_str(), "EPSG:", 5) == 0) {
        #ifdef _WIN32
        sscanf_s(targetCRS.c_str() + 5, "%d", &nTargetEPSG);
        #else
        sscanf(targetCRS.c_str() + 5, "%d", &nTargetEPSG);
        #endif
    }

    bool sourceSet = false;
    if (nSourceEPSG > 0) {
        if (oSourceSRS.importFromEPSG(nSourceEPSG) == OGRERR_NONE) {
            _logger->debug("Imported source SRS from EPSG:{}", nSourceEPSG);
            sourceSet = true;
        } else {
            _logger->warn("Failed to import source SRS from EPSG:{}, falling back to user input.", nSourceEPSG);
        }
    }
    if (!sourceSet) {
         if (oSourceSRS.SetFromUserInput(sourceCRS.c_str()) != OGRERR_NONE) {
             _logger->error("Failed to set source SRS from user input: {}. Error: {}", sourceCRS, CPLGetLastErrorMsg());
             return nullptr;
         }
          _logger->debug("Set source SRS from user input.");
    }

    bool targetSet = false;
    if (nTargetEPSG > 0) {
        if (oTargetSRS.importFromEPSG(nTargetEPSG) == OGRERR_NONE) {
            _logger->debug("Imported target SRS from EPSG:{}", nTargetEPSG);
            targetSet = true;
        } else {
             _logger->warn("Failed to import target SRS from EPSG:{}, falling back to user input.", nTargetEPSG);
        }
    }
     if (!targetSet) {
         if (oTargetSRS.SetFromUserInput(targetCRS.c_str()) != OGRERR_NONE) {
             _logger->error("Failed to set target SRS from user input: {}. Error: {}", targetCRS, CPLGetLastErrorMsg());
             return nullptr;
         }
         _logger->debug("Set target SRS from user input.");
     }
    // --- EPSG 代码处理结束 ---

    // --- 关键点：轴顺序设置 ---
    // 统一将源和目标都设置为传统 GIS 顺序 (Lon/Lat, X/Y)
    // 这与 Point 结构 (x=Lon, y=Lat) 和 Transform(1, &x, &y, ...) 的调用方式保持一致
    _logger->debug("Setting axis mapping strategy for source and target SRS to OAMS_TRADITIONAL_GIS_ORDER.");
    oSourceSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    oTargetSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    // 创建变换对象
    poCT = OGRCreateCoordinateTransformation(&oSourceSRS, &oTargetSRS);

    if (poCT == nullptr) {
         _logger->error("OGRCreateCoordinateTransformation failed. Error: {}", CPLGetLastErrorMsg());
        // ... (Error handling) ...
        return nullptr;
    }

    _logger->info("Successfully created transformation from '{}' to '{}'", sourceCRS, targetCRS);
    // Directly return the original transformation object
    return poCT;
}

std::string TransformationCacheImpl::createCacheKey(
    const std::string& sourceCRS,
    const std::string& targetCRS) const
{
    // Simple concatenation with a separator. Consider normalization or hashing for complex keys.
    return sourceCRS + "||" + targetCRS;
}

} // namespace cache
} // namespace crs
} // namespace core_services
} // namespace oscean
