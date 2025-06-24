#include "concurrent_optimization_components.h"
#include "common_utils/utilities/logging_utils.h"

#include <filesystem>
#include <thread>
#include <set>
#include <string>
#include <unordered_map>

// 🆕 添加GDAL支持 - 移到命名空间定义之前
#include <gdal_priv.h>
#include <ogrsf_frmts.h>

namespace oscean::core_services {

// ===============================================================================
// GdalInitializationManagerImpl 实现
// ===============================================================================

GdalInitializationManagerImpl::GdalInitializationManagerImpl(
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory)
    : commonFactory_(commonFactory)
    , isWarmedUp_(false) {
    
    // 构造函数中不执行初始化，而是在需要时延迟初始化
}

bool GdalInitializationManagerImpl::warmupInitialization() {
    if (isWarmedUp_.load()) {
        return true;
    }
    
    std::unique_lock<std::shared_mutex> lock(initMutex_);
    
    // 双重检查锁定模式
    if (isWarmedUp_.load()) {
        return true;
    }
    
    auto start = std::chrono::steady_clock::now();
    
    try {
        // 🔧 检查GDAL全局初始化状态而不是重复初始化
        OSCEAN_LOG_INFO("GdalInit", "🔥 检查GDAL预热状态...");
        
        // 检查GDAL是否已经由全局初始化器初始化
        int existingDrivers = GDALGetDriverCount();
        if (existingDrivers > 0) {
            OSCEAN_LOG_INFO("GdalInit", "✅ GDAL已由全局初始化器初始化 - 驱动数量: {}", existingDrivers);
            isWarmedUp_.store(true);
            return true;
        }
        
        // 如果GDAL未初始化，这是一个错误状态
        OSCEAN_LOG_ERROR("GdalInit", "❌ GDAL未初始化！请确保在main函数中调用了GdalGlobalInitializer::initialize()");
        return false;
        
        // 移除所有分散的GDAL初始化调用
        // CPLSetConfigOption("GDAL_NUM_THREADS", "1");        // ❌ 已移除
        // CPLSetConfigOption("GDAL_CACHEMAX", "256");         // ❌ 已移除
        // CPLSetConfigOption("GDAL_MAX_DATASET_POOL_SIZE", "100"); // ❌ 已移除
        // CPLSetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"); // ❌ 已移除
        // GDALAllRegister(); // ❌ 已移除
        // OGRRegisterAll();  // ❌ 已移除
        
        // 验证初始化结果
        int finalDriverCount = GDALGetDriverCount();
        if (finalDriverCount > 0) {
            isWarmedUp_.store(true);
            
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            
            OSCEAN_LOG_INFO("GdalInit", "✅ GDAL预热初始化完成 ({}ms) - 驱动数量: {}", 
                           elapsed.count(), finalDriverCount);
            return true;
        } else {
            OSCEAN_LOG_ERROR("GdalInit", "❌ GDAL驱动注册失败");
            return false;
        }
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("GdalInit", "❌ GDAL初始化失败: {}", e.what());
        return false;
    } catch (...) {
        OSCEAN_LOG_ERROR("GdalInit", "❌ GDAL初始化未知异常");
        return false;
    }
}

bool GdalInitializationManagerImpl::isWarmedUp() const {
    return isWarmedUp_.load();
}

std::shared_ptr<oscean::common_utils::simd::ISIMDManager> 
GdalInitializationManagerImpl::getSIMDManager() const {
    if (commonFactory_) {
        return commonFactory_->getSIMDManager();
    }
    return nullptr;
}

std::shared_ptr<oscean::common_utils::memory::IMemoryManager> 
GdalInitializationManagerImpl::getMemoryManager() const {
    if (commonFactory_) {
        return commonFactory_->getMemoryManager();
    }
    return nullptr;
}

std::shared_ptr<oscean::common_utils::async::AsyncFramework> 
GdalInitializationManagerImpl::getAsyncFramework() const {
    // 🔧 注意：CommonServicesFactory没有getAsyncFramework()方法
    // 返回nullptr或创建临时的AsyncFramework
    return nullptr;
}

// ===============================================================================
// FileAccessLockManagerImpl 实现
// ===============================================================================

FileAccessLockManagerImpl::FileAccessGuardImpl::FileAccessGuardImpl(
    const std::string& filePath, FileAccessLockManagerImpl& manager)
    : filePath_(filePath), manager_(manager) {
    manager_.acquireFileLock(filePath_);
}

FileAccessLockManagerImpl::FileAccessGuardImpl::~FileAccessGuardImpl() {
    manager_.releaseFileLock(filePath_);
}

// 🔧 修复：实现继承的虚函数接口（移除重复定义）
std::unique_ptr<UnifiedDataAccessServiceImpl::FileAccessLockManager::FileAccessGuard>
FileAccessLockManagerImpl::createFileGuard(const std::string& filePath) {
    return std::unique_ptr<UnifiedDataAccessServiceImpl::FileAccessLockManager::FileAccessGuard>(
        new FileAccessGuardImpl(filePath, *this));
}

UnifiedDataAccessServiceImpl::FileAccessLockManager::FileAccessStats
FileAccessLockManagerImpl::getFileAccessStats(const std::string& filePath) const {
    std::shared_lock<std::shared_mutex> lock(managerMutex_);
    
    auto it = fileLocks_.find(filePath);
    if (it != fileLocks_.end()) {
        UnifiedDataAccessServiceImpl::FileAccessLockManager::FileAccessStats stats;
        stats.activeReaders = it->second->activeReaders.load();
        stats.queuedRequests = it->second->queuedRequests.load();
        stats.lastAccess = it->second->lastAccess;
        return stats;
    }
    
    return {}; // 文件未被访问过
}

void FileAccessLockManagerImpl::acquireFileLock(const std::string& filePath) {
    std::string normalizedPath = normalizePath(filePath);
    
    // 获取或创建文件锁信息
    std::shared_ptr<FileLockInfo> lockInfo;
    {
        std::unique_lock<std::shared_mutex> lock(managerMutex_);
        auto it = fileLocks_.find(normalizedPath);
        if (it == fileLocks_.end()) {
            lockInfo = std::make_shared<FileLockInfo>();
            fileLocks_[normalizedPath] = lockInfo;
        } else {
            lockInfo = it->second;
        }
    }
    
    // 获取文件级共享锁（支持多读）
    lockInfo->queuedRequests.fetch_add(1);
    lockInfo->mutex.lock_shared();
    lockInfo->queuedRequests.fetch_sub(1);
    lockInfo->activeReaders.fetch_add(1);
    lockInfo->lastAccess = std::chrono::system_clock::now();
}

void FileAccessLockManagerImpl::releaseFileLock(const std::string& filePath) {
    std::string normalizedPath = normalizePath(filePath);
    
    std::shared_lock<std::shared_mutex> lock(managerMutex_);
    auto it = fileLocks_.find(normalizedPath);
    if (it != fileLocks_.end()) {
        auto& lockInfo = it->second;
        lockInfo->activeReaders.fetch_sub(1);
        lockInfo->mutex.unlock_shared();
    }
}

std::string FileAccessLockManagerImpl::normalizePath(const std::string& path) const {
    try {
        return std::filesystem::canonical(path).string();
    } catch (const std::exception&) {
        return path; // 如果无法规范化，返回原路径
    }
}

// ===============================================================================
// ReaderPoolManagerImpl 实现
// ===============================================================================

// 🔧 修复：实现继承的虚函数接口（移除重复定义）
bool ReaderPoolManagerImpl::initializePool(
    const UnifiedDataAccessServiceImpl::ReaderPoolManager::PoolConfiguration& config,
    std::shared_ptr<data_access::readers::ReaderRegistry> readerRegistry,
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices) {
    
    std::unique_lock<std::shared_mutex> lock(poolMutex_);
    
    if (isInitialized_.load()) {
        return true; // 已经初始化
    }
    
    // 转换配置类型到内部配置
    config_.initialPoolSize = config.initialPoolSize;
    config_.maxPoolSize = config.maxPoolSize;
    config_.growthIncrement = config.growthIncrement;
    config_.readerTTL = std::chrono::duration_cast<std::chrono::minutes>(config.readerTTL);
    config_.enablePooling = config.enablePooling;
    
    readerRegistry_ = readerRegistry;
    commonServices_ = commonServices;
    
    if (!config_.enablePooling) {
        OSCEAN_LOG_INFO("ReaderPool", "读取器池化已禁用，跳过初始化");
        isInitialized_.store(true);
        return true;
    }
    
    try {
        OSCEAN_LOG_INFO("ReaderPool", "🏊 初始化读取器池 - 初始大小: {}, 最大大小: {}", 
               config_.initialPoolSize, config_.maxPoolSize);
        
        isInitialized_.store(true);
        OSCEAN_LOG_INFO("ReaderPool", "✅ 读取器池初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("ReaderPool", "❌ 读取器池初始化失败: {}", e.what());
        return false;
    }
}

std::shared_ptr<data_access::readers::UnifiedDataReader> 
ReaderPoolManagerImpl::getOrCreateReader(const std::string& filePath, const std::string& readerType) {
    
    if (!config_.enablePooling || !readerRegistry_) {
        // 直接创建读取器
        try {
            std::optional<std::string> explicitFormat = readerType;
            auto reader = readerRegistry_->createReader(filePath, explicitFormat);
            if (reader) {
                totalCreatedReaders_.fetch_add(1);
            }
            return reader;
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("ReaderPool", "❌ 创建读取器失败: {} - {}", filePath, e.what());
            return nullptr;
        }
    }
    
    // 先尝试从池中获取
    {
        std::unique_lock<std::shared_mutex> lock(poolMutex_);
        auto it = readerPools_.find(readerType);
        if (it != readerPools_.end() && !it->second.empty()) {
            auto reader = it->second.back().reader;
            it->second.pop_back();
            totalPoolHits_.fetch_add(1);
            return reader;
        }
    }
    
    // 池中没有，直接创建
    totalPoolMisses_.fetch_add(1);
    try {
        std::optional<std::string> explicitFormat = readerType;
        auto reader = readerRegistry_->createReader(filePath, explicitFormat);
        if (reader) {
            totalCreatedReaders_.fetch_add(1);
        }
        return reader;
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("ReaderPool", "❌ 创建读取器失败: {} - {}", filePath, e.what());
        return nullptr;
    }
}

void ReaderPoolManagerImpl::returnReaderToPool(std::shared_ptr<data_access::readers::UnifiedDataReader> reader) {
    if (!config_.enablePooling || !reader) {
        return;
    }
    
    try {
        // 简化版本：现在暂时不实际入池，只是重置状态
        // 实际实现中需要确定读取器类型并加入对应的池
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_WARN("ReaderPool", "⚠️ 返回读取器到池失败: {}", e.what());
    }
}

namespace data_access {
    
    GdalSafeHandle::GdalSafeHandle(GDALDatasetH hDataset) : hDataset_(hDataset) {}

    GdalSafeHandle::~GdalSafeHandle() {
        if (hDataset_) {
            GDALClose(hDataset_);
        }
    }

    GDALDatasetH GdalSafeHandle::get() const {
        return hDataset_;
    }
    
    GdalSafeHandle::GdalSafeHandle(GdalSafeHandle&& other) noexcept : hDataset_(other.hDataset_) {
        other.hDataset_ = nullptr;
    }

    GdalSafeHandle& GdalSafeHandle::operator=(GdalSafeHandle&& other) noexcept {
        if (this != &other) {
            if (hDataset_) {
                GDALClose(hDataset_);
            }
            hDataset_ = other.hDataset_;
            other.hDataset_ = nullptr;
        }
        return *this;
    }

    GdalGlobalConfigurator::GdalGlobalConfigurator() {
        // Empty, as global initialization is handled by GdalGlobalInitializer
    }
    
    // This static instance might be the cause of some issues, but let's
    // restore it to see if it fixes the include-hell, then we can remove it.
    static GdalGlobalConfigurator gdal_config;

} // namespace data_access

} // namespace oscean::core_services 