#pragma once

#include "unified_data_access_service_impl.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include <unordered_map>
#include <shared_mutex>
#include <chrono>
#include <gdal.h>
#include <memory>
#include <mutex>
#include <iostream>

namespace oscean::core_services {

/**
 * @brief GDAL初始化管理器实现 - 依赖注入版本
 */
class GdalInitializationManagerImpl : public UnifiedDataAccessServiceImpl::GdalInitializationManager {
public:
    explicit GdalInitializationManagerImpl(
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory);
    
    bool warmupInitialization() override;
    bool isWarmedUp() const override;
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> getSIMDManager() const override;
    std::shared_ptr<oscean::common_utils::memory::IMemoryManager> getMemoryManager() const override;
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> getAsyncFramework() const override;

private:
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonFactory_;
    mutable std::shared_mutex initMutex_;
    std::atomic<bool> isWarmedUp_;
};

/**
 * @brief 文件访问锁管理器实现 - 依赖注入版本
 */
class FileAccessLockManagerImpl : public UnifiedDataAccessServiceImpl::FileAccessLockManager {
public:
    class FileAccessGuardImpl : public FileAccessGuard {
    public:
        FileAccessGuardImpl(const std::string& filePath, FileAccessLockManagerImpl& manager);
        ~FileAccessGuardImpl() override;
    private:
        std::string filePath_;
        FileAccessLockManagerImpl& manager_;
    };

    std::unique_ptr<FileAccessGuard> createFileGuard(const std::string& filePath) override;
    FileAccessStats getFileAccessStats(const std::string& filePath) const override;

private:
    struct FileLockInfo {
        mutable std::shared_mutex mutex;
        std::atomic<size_t> activeReaders{0};
        std::atomic<size_t> queuedRequests{0};
        std::chrono::system_clock::time_point lastAccess;
    };

    mutable std::shared_mutex managerMutex_;
    std::unordered_map<std::string, std::shared_ptr<FileLockInfo>> fileLocks_;

    void acquireFileLock(const std::string& filePath);
    void releaseFileLock(const std::string& filePath);
    std::string normalizePath(const std::string& path) const;
};

/**
 * @brief 读取器池管理器实现 - 依赖注入版本
 */
class ReaderPoolManagerImpl : public UnifiedDataAccessServiceImpl::ReaderPoolManager {
public:
    bool initializePool(
        const PoolConfiguration& config,
        std::shared_ptr<data_access::readers::ReaderRegistry> readerRegistry,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices) override;
    
    std::shared_ptr<data_access::readers::UnifiedDataReader> getOrCreateReader(
        const std::string& filePath, const std::string& readerType) override;
    
    void returnReaderToPool(std::shared_ptr<data_access::readers::UnifiedDataReader> reader) override;

private:
    struct PooledReaderInfo {
        std::shared_ptr<data_access::readers::UnifiedDataReader> reader;
        std::chrono::steady_clock::time_point lastUsed;
        std::chrono::steady_clock::time_point createdAt;
    };

    struct InternalPoolConfiguration {
        size_t initialPoolSize = 4;
        size_t maxPoolSize = 16;
        size_t growthIncrement = 2;
        std::chrono::minutes readerTTL{30};
        bool enablePooling = true;
    };

    mutable std::shared_mutex poolMutex_;
    std::unordered_map<std::string, std::vector<PooledReaderInfo>> readerPools_;
    
    InternalPoolConfiguration config_;
    std::shared_ptr<data_access::readers::ReaderRegistry> readerRegistry_;
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices_;
    
    std::atomic<bool> isInitialized_{false};
    std::atomic<size_t> totalCreatedReaders_{0};
    std::atomic<size_t> totalPoolHits_{0};
    std::atomic<size_t> totalPoolMisses_{0};
};

namespace data_access {

    /**
     * @brief GDAL数据集的RAII包装器，确保GDALDatasetH被正确关闭。
     */
    class GdalSafeHandle {
    public:
        explicit GdalSafeHandle(GDALDatasetH hDataset);
        ~GdalSafeHandle();
        
        GDALDatasetH get() const;
        
        // 删除拷贝构造函数和赋值运算符
        GdalSafeHandle(const GdalSafeHandle&) = delete;
        GdalSafeHandle& operator=(const GdalSafeHandle&) = delete;
        
        // 允许移动
        GdalSafeHandle(GdalSafeHandle&& other) noexcept;
        GdalSafeHandle& operator=(GdalSafeHandle&& other) noexcept;

    private:
        GDALDatasetH hDataset_;
    };

    /**
     * @brief 用于确保GDAL配置只被设置一次的辅助类 (即使现在是空的)
     */
    class GdalGlobalConfigurator {
    public:
        GdalGlobalConfigurator();
    };

} // namespace data_access

} // namespace oscean::core_services