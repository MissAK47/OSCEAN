/**
 * @file common_services_factory.cpp
 * @brief CommonServicesFactory完整实现 - 1-9项核心服务
 */

#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "common_utils/memory/memory_manager_unified.h"
#include "common_utils/simd/simd_manager_unified.h"
#include "common_utils/gpu/oscean_gpu_framework.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/utilities/app_config_loader.h"
#include "common_utils/async/async_framework.h"
#include "core_services/exceptions.h"
#include "core_services/common_data_types.h" // For FileMetadata and GridData

// GDAL/PROJ相关的头文件
#include <gdal.h>
#include <cpl_conv.h> // For CPLSetConfigOption
#include <filesystem>

// 📌 C++标准库和第三方库
#include <iostream>
#include <mutex>
#include <thread>

namespace oscean::common_utils::infrastructure {

// === 适配器类定义 ===

/**
 * @brief ThreadPoolManager适配器
 */
class ThreadPoolManagerAdapter : public threading::IThreadPoolManager {
public:
    explicit ThreadPoolManagerAdapter(std::shared_ptr<UnifiedThreadPoolManager> impl)
        : impl_(std::move(impl)) {}
    
    bool initialize() override {
        // UnifiedThreadPoolManager在构造时已初始化
        return impl_ != nullptr;
    }
    
    void shutdown() override { 
        if (impl_) {
            impl_->requestShutdown();
        }
    }
    
    size_t getThreadCount() const override { 
        if (impl_) {
            auto stats = impl_->getStatistics();
            return stats.totalThreads;
        }
        return 0;
    }
    
    bool isHealthy() const override { 
        if (impl_) {
            auto health = impl_->getHealthStatus();
            return health.healthy;
        }
        return false;
    }

private:
    std::shared_ptr<UnifiedThreadPoolManager> impl_;
};

/**
 * @brief Logger适配器
 */
class LoggerAdapter : public logging::ILogger {
public:
    explicit LoggerAdapter(std::shared_ptr<oscean::common_utils::LoggingManager> impl)
        : impl_(std::move(impl)) {}
    
    void trace(const std::string& message) override {
        if (impl_) {
            auto logger = impl_->getLogger();
            if (logger) logger->trace(message);
        }
    }
    
    void debug(const std::string& message) override {
        if (impl_) {
            auto logger = impl_->getLogger();
            if (logger) logger->debug(message);
        }
    }
    
    void info(const std::string& message) override {
        if (impl_) {
            auto logger = impl_->getLogger();
            if (logger) logger->info(message);
        }
    }
    
    void warn(const std::string& message) override {
        if (impl_) {
            auto logger = impl_->getLogger();
            if (logger) logger->warn(message);
        }
    }
    
    void error(const std::string& message) override {
        if (impl_) {
            auto logger = impl_->getLogger();
            if (logger) logger->error(message);
        }
    }
    
    void critical(const std::string& message) override {
        if (impl_) {
            auto logger = impl_->getLogger();
            if (logger) logger->critical(message);
        }
    }
    
    void setLevel(const std::string& level) override {
        if (impl_) {
            impl_->setLevel(level);
        }
    }
    
    void flush() override {
        if (impl_) {
            impl_->flushAll();
        }
    }

private:
    std::shared_ptr<oscean::common_utils::LoggingManager> impl_;
};

/**
 * @brief ConfigurationLoader适配器
 */
class ConfigurationLoaderAdapter : public utilities::IConfigurationLoader {
public:
    explicit ConfigurationLoaderAdapter(std::shared_ptr<oscean::common_utils::AppConfigLoader> impl)
        : impl_(std::move(impl)) {}
    
    bool loadFromFile(const std::string& filePath) override {
        if (impl_) {
            return impl_->loadFromFile(filePath);
        }
        return false;
    }
    
    std::string getString(const std::string& key, const std::string& defaultValue = "") const override {
        if (impl_) {
            return impl_->getString(key, defaultValue);
        }
        return defaultValue;
    }
    
    int getInt(const std::string& key, int defaultValue = 0) const override {
        if (impl_) {
            return impl_->getInt(key, defaultValue);
        }
        return defaultValue;
    }

    bool getBool(const std::string& key, bool defaultValue = false) const override {
        if (impl_) {
            return impl_->getBool(key, defaultValue);
        }
        return defaultValue;
    }

    bool has(const std::string& key) const override {
        if (impl_) {
            return impl_->has(key);
        }
        return false;
    }

private:
    std::shared_ptr<oscean::common_utils::AppConfigLoader> impl_;
};

/**
 * @brief AsyncExecutor适配器
 */
class AsyncExecutorAdapter : public async::IAsyncExecutor {
public:
    explicit AsyncExecutorAdapter(std::shared_ptr<oscean::common_utils::async::AsyncFramework> impl)
        : impl_(std::move(impl)) {}
    
    bool initialize() override {
        return impl_ != nullptr;
    }
    
    void shutdown() override {
        if (impl_) {
            impl_->shutdown();
        }
    }
    
    bool isHealthy() const override {
        return impl_ && !impl_->isShutdown();
    }

private:
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> impl_;
};

// Pimpl (Pointer to implementation) Idiom
class CommonServicesFactory::Impl {
public:
    explicit Impl(const std::string& configPath, const ServiceConfiguration& config = {}) {
        // Step 1: Initialize Configuration Loader
        try {
            configurationLoader_ = std::make_shared<oscean::common_utils::AppConfigLoader>();
            if (!configPath.empty()) {
                if (!configurationLoader_->loadFromFile(configPath)) {
                     std::cerr << "[CommonServicesFactory] WARNING: Failed to load config '" << configPath << "'. Using defaults." << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[CommonServicesFactory] FATAL: Failed to create AppConfigLoader: " << e.what() << std::endl;
            throw oscean::core_services::ServiceCreationException("Failed to create AppConfigLoader: " + std::string(e.what()));
        }

        // Step 2: Initialize Logging Manager
        try {
            oscean::common_utils::LoggingConfig logConfig;
            logConfig.log_filename = configurationLoader_->getString("logging.log_directory", "logs") + "/common_services.log";
            logConfig.file_level = configurationLoader_->getString("logging.level", "info");
            logConfig.enable_file = true;
            loggingManager_ = std::make_shared<oscean::common_utils::LoggingManager>(logConfig);
            oscean::common_utils::LoggingManager::setGlobalInstance(loggingManager_);
        } catch (const std::exception& e) {
            std::cerr << "[CommonServicesFactory] FATAL: Failed to create LoggingManager: " << e.what() << std::endl;
            throw oscean::core_services::ServiceCreationException("Failed to create LoggingManager: " + std::string(e.what()));
        }

        OSCEAN_LOG_INFO("CommonServicesFactory", "Config and Logging initialized. Initializing other services...");

        // Step 3: Initialize Thread Pool - 🔧 优先使用传入的共享线程池管理器
        try {
            if (config.sharedThreadPoolManager) {
                threadPoolManager_ = config.sharedThreadPoolManager;
                OSCEAN_LOG_INFO("CommonServicesFactory", "Using shared UnifiedThreadPoolManager instance");
            } else {
                UnifiedThreadPoolManager::PoolConfiguration poolConfig;
                poolConfig.maxThreads = configurationLoader_->getInt("threading.pool_size", 0);
                if (poolConfig.maxThreads == 0) {
                    poolConfig.maxThreads = std::thread::hardware_concurrency();
                }
                threadPoolManager_ = std::make_shared<UnifiedThreadPoolManager>(poolConfig);
                OSCEAN_LOG_INFO("CommonServicesFactory", "Created new UnifiedThreadPoolManager instance");
            }
        } catch (const std::exception& e) {
            throw oscean::core_services::ServiceCreationException("Failed to initialize UnifiedThreadPoolManager: " + std::string(e.what()));
        }

        // Step 4: Initialize Memory Manager
        try {
            memoryManager_ = std::make_shared<oscean::common_utils::memory::UnifiedMemoryManager>();
        } catch (const std::exception& e) {
            throw oscean::core_services::ServiceCreationException("Failed to create UnifiedMemoryManager: " + std::string(e.what()));
        }

        // Step 5: Initialize SIMD Manager
        try {
            simdManager_ = std::make_shared<oscean::common_utils::simd::UnifiedSIMDManager>();
        } catch (const std::exception& e) {
            throw oscean::core_services::ServiceCreationException("Failed to create SIMDManager: " + std::string(e.what()));
        }

        // Step 6: Initialize Async Framework
        try {
            // 创建独立的线程池给AsyncFramework使用
            size_t asyncThreads = configurationLoader_->getInt("async.thread_count", 2);
            auto asyncThreadPool = std::make_shared<boost::asio::thread_pool>(asyncThreads);
            asyncFramework_ = std::make_shared<oscean::common_utils::async::AsyncFramework>(asyncThreadPool);
        } catch (const std::exception& e) {
            throw oscean::core_services::ServiceCreationException("Failed to create AsyncFramework: " + std::string(e.what()));
        }
        
        OSCEAN_LOG_INFO("CommonServicesFactory", "All services initialized successfully.");
    }

    ~Impl() {
        shutdown();
    }

    void shutdown() {
        OSCEAN_LOG_INFO("CommonServicesFactory", "Shutting down services...");
        if (asyncFramework_) {
            asyncFramework_->shutdown();
        }
        if (threadPoolManager_) {
            threadPoolManager_->requestShutdown();
        }
        if (loggingManager_) {
            loggingManager_->shutdown();
        }
        OSCEAN_LOG_INFO("CommonServicesFactory", "Shutdown complete.");
    }
    
    std::shared_ptr<UnifiedThreadPoolManager> getThreadPool() const { return threadPoolManager_; }
    std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> getMemoryManager() const { return memoryManager_; }
    std::shared_ptr<oscean::common_utils::simd::UnifiedSIMDManager> getSIMDManager() const { return simdManager_; }
    std::shared_ptr<oscean::common_utils::LoggingManager> getLogger() const { return loggingManager_; }
    std::shared_ptr<oscean::common_utils::AppConfigLoader> getConfigurationLoader() const { return configurationLoader_; }
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> getAsyncFramework() const { return asyncFramework_; }

private:
    std::shared_ptr<UnifiedThreadPoolManager> threadPoolManager_;
    std::shared_ptr<oscean::common_utils::memory::UnifiedMemoryManager> memoryManager_;
    std::shared_ptr<oscean::common_utils::simd::UnifiedSIMDManager> simdManager_;
    std::shared_ptr<oscean::common_utils::LoggingManager> loggingManager_;
    std::shared_ptr<oscean::common_utils::AppConfigLoader> configurationLoader_;
    std::shared_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework_;
};


// --- CommonServicesFactory class forwarding to Impl ---

CommonServicesFactory::CommonServicesFactory(const std::string& configPath)
    : pImpl_(std::make_unique<Impl>(configPath)) {}

// This constructor is kept for legacy compatibility but now just forwards to the main one.
CommonServicesFactory::CommonServicesFactory(const ServiceConfiguration& config)
    : pImpl_(std::make_unique<Impl>("", config)) {
    // 现在正确使用传入的配置，包括sharedThreadPoolManager
    OSCEAN_LOG_INFO("CommonServicesFactory", "Created with ServiceConfiguration (sharedThreadPoolManager: {})", 
                   config.sharedThreadPoolManager ? "yes" : "no");
}

CommonServicesFactory::~CommonServicesFactory() = default;

CommonServicesFactory::CommonServicesFactory(CommonServicesFactory&&) noexcept = default;
CommonServicesFactory& CommonServicesFactory::operator=(CommonServicesFactory&&) noexcept = default;

void CommonServicesFactory::shutdown() {
    if (pImpl_) pImpl_->shutdown();
}

// 📌 Return the INTERFACE through adapters
std::shared_ptr<oscean::common_utils::infrastructure::threading::IThreadPoolManager> CommonServicesFactory::getThreadPoolManager() const {
    return std::make_shared<ThreadPoolManagerAdapter>(pImpl_->getThreadPool());
}

std::shared_ptr<oscean::common_utils::memory::IMemoryManager> CommonServicesFactory::getMemoryManager() const {
    return pImpl_->getMemoryManager();
}

std::shared_ptr<oscean::common_utils::simd::ISIMDManager> CommonServicesFactory::getSIMDManager() const {
    return pImpl_->getSIMDManager();
}

std::shared_ptr<oscean::common_utils::gpu::OSCEANGPUFramework> CommonServicesFactory::getGPUFramework() const {
    // GPU框架是全局单例，使用自定义删除器创建shared_ptr
    // 删除器什么都不做，因为单例的生命周期由框架自己管理
    return std::shared_ptr<oscean::common_utils::gpu::OSCEANGPUFramework>(
        &oscean::common_utils::gpu::OSCEANGPUFramework::getInstance(),
        [](oscean::common_utils::gpu::OSCEANGPUFramework*) {} // 空删除器，不删除单例
    );
}

std::shared_ptr<oscean::common_utils::infrastructure::logging::ILogger> CommonServicesFactory::getLogger() const {
    return std::make_shared<LoggerAdapter>(pImpl_->getLogger());
}

std::shared_ptr<oscean::common_utils::infrastructure::utilities::IConfigurationLoader> CommonServicesFactory::getConfigurationLoader() const {
    return std::make_shared<ConfigurationLoaderAdapter>(pImpl_->getConfigurationLoader());
}

std::shared_ptr<oscean::common_utils::infrastructure::async::IAsyncExecutor> CommonServicesFactory::getAsyncExecutor() const {
    return std::make_shared<AsyncExecutorAdapter>(pImpl_->getAsyncFramework());
}

std::shared_ptr<UnifiedThreadPoolManager> CommonServicesFactory::getUnifiedThreadPoolManager() const {
    return pImpl_->getThreadPool();
}

// ✅ 添加缺少的方法实现

/**
 * @brief 性能监控器适配器
 */
class PerformanceMonitorAdapter : public performance::IPerformanceMonitor {
public:
    bool initialize() override {
        return true; // 简单实现，总是返回成功
    }
    
    void shutdown() override {
        // 空实现
    }
    
    bool isHealthy() const override {
        return true; // 简单实现，总是返回健康
    }
};



std::shared_ptr<oscean::common_utils::infrastructure::performance::IPerformanceMonitor> CommonServicesFactory::getPerformanceMonitor() const {
    return std::make_shared<PerformanceMonitorAdapter>();
}

bool CommonServicesFactory::isHealthy() const {
    if (!pImpl_) {
        return false;
    }
    
    // 检查各个服务的健康状态
    try {
        auto threadPool = pImpl_->getThreadPool();
        if (!threadPool) {
            return false;
        }
        auto healthStatus = threadPool->getHealthStatus();
        if (!healthStatus.healthy) {
            return false;
        }
        
        auto memoryManager = pImpl_->getMemoryManager();
        if (!memoryManager) {
            return false;
        }
        
        auto simdManager = pImpl_->getSIMDManager();
        if (!simdManager) {
            return false;
        }
        
        auto logger = pImpl_->getLogger();
        if (!logger) {
            return false;
        }
        
        auto configLoader = pImpl_->getConfigurationLoader();
        if (!configLoader) {
            return false;
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> CommonServicesFactory::getDiagnosticMessages() const {
    std::vector<std::string> messages;
    
    if (!pImpl_) {
        messages.push_back("CommonServicesFactory: Implementation not initialized");
        return messages;
    }
    
    try {
        // 检查各个服务的状态
        auto threadPool = pImpl_->getThreadPool();
        if (threadPool) {
            auto health = threadPool->getHealthStatus();
            auto stats = threadPool->getStatistics();
            if (health.healthy) {
                messages.push_back("ThreadPool: Healthy (" + std::to_string(stats.totalThreads) + " threads, " + 
                                 std::to_string(health.pendingTasks) + " pending tasks)");
            } else {
                std::string warningMsg = "ThreadPool: Unhealthy";
                if (!health.warnings.empty()) {
                    warningMsg += " - " + health.warnings[0];
                }
                messages.push_back(warningMsg);
            }
        } else {
            messages.push_back("ThreadPool: Not initialized");
        }
        
        auto memoryManager = pImpl_->getMemoryManager();
        if (memoryManager) {
            messages.push_back("MemoryManager: Initialized");
        } else {
            messages.push_back("MemoryManager: Not initialized");
        }
        
        auto simdManager = pImpl_->getSIMDManager();
        if (simdManager) {
            messages.push_back("SIMDManager: Initialized");
        } else {
            messages.push_back("SIMDManager: Not initialized");
        }
        
        auto logger = pImpl_->getLogger();
        if (logger) {
            messages.push_back("Logger: Initialized");
        } else {
            messages.push_back("Logger: Not initialized");
        }
        
        auto configLoader = pImpl_->getConfigurationLoader();
        if (configLoader) {
            messages.push_back("ConfigurationLoader: Initialized");
        } else {
            messages.push_back("ConfigurationLoader: Not initialized");
        }
        
    } catch (const std::exception& e) {
        messages.push_back("Error during diagnostic check: " + std::string(e.what()));
    }
    
    return messages;
}

/**
 * @brief 简单的缓存实现
 */
template<typename Key, typename Value>
    class SimpleCacheImpl : public ICache<Key, Value> {
private:
    mutable std::mutex mutex_;
    std::map<Key, std::shared_ptr<Value>> data_;  // 存储shared_ptr而不是值
    size_t capacity_;
    std::string strategy_;
    size_t hits_ = 0;
    size_t misses_ = 0;

    public:
    explicit SimpleCacheImpl(const std::string& /* name */, size_t capacity, const std::string& strategy)
        : capacity_(capacity), strategy_(strategy) {}
        
        bool put(const Key& key, const Value& value) override {
            if constexpr (!std::is_copy_constructible_v<Value>) {
                // 对于不可拷贝的类型，缓存不支持
                return false;
            } else {
            std::lock_guard<std::mutex> lock(mutex_);
        if (data_.size() >= capacity_ && data_.find(key) == data_.end()) {
            // 简单的LRU：删除第一个元素
            if (!data_.empty()) {
                data_.erase(data_.begin());
            }
            }
                data_[key] = std::make_shared<Value>(value);
            return true;
            }
        }
        
        std::optional<Value> get(const Key& key) override {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = data_.find(key);
        if (it != data_.end() && it->second) {
            hits_++;
            // 如果Value是可拷贝的，返回副本；否则需要特殊处理
            if constexpr (std::is_copy_constructible_v<Value>) {
                return *(it->second);
            } else {
                // 对于不可拷贝的类型，我们不能返回副本
                // 这是一个设计问题，需要改变接口或使用其他方法
                return std::nullopt;
            }
        }
        misses_++;
        return std::nullopt;
        }
        
        bool remove(const Key& key) override {
            std::lock_guard<std::mutex> lock(mutex_);
            return data_.erase(key) > 0;
        }
        
        bool contains(const Key& key) const override {
            std::lock_guard<std::mutex> lock(mutex_);
            return data_.find(key) != data_.end();
        }
        
        void clear() override {
            std::lock_guard<std::mutex> lock(mutex_);
            data_.clear();
        }
        
        size_t size() const override {
            std::lock_guard<std::mutex> lock(mutex_);
            return data_.size();
        }
        
    size_t capacity() const override {
        return capacity_;
    }
    
    double hitRate() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = hits_ + misses_;
        return total > 0 ? static_cast<double>(hits_) / total : 0.0;
    }
        
        std::map<Key, Value> getBatch(const std::vector<Key>& keys) override {
            std::map<Key, Value> result;
            if constexpr (std::is_copy_constructible_v<Value>) {
            for (const auto& key : keys) {
            auto value = get(key);
            if (value) {
                        result.emplace(key, std::move(*value));
                    }
                }
            }
            // 对于不可拷贝的类型，返回空map
            return result;
        }
        
        void putBatch(const std::map<Key, Value>& items) override {
            for (const auto& [key, value] : items) {
                put(key, value);
            }
        }
        
        void removeBatch(const std::vector<Key>& keys) override {
            for (const auto& key : keys) {
                remove(key);
            }
        }
        
    void evictExpired() override {
        // 简单实现：不做任何操作
    }
    
    void optimize() override {
        // 简单实现：不做任何操作
    }
    
    void resetStatistics() override {
        std::lock_guard<std::mutex> lock(mutex_);
        hits_ = 0;
        misses_ = 0;
    }
    
    std::string getReport() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return "Cache size: " + std::to_string(data_.size()) + 
               ", capacity: " + std::to_string(capacity_) +
               ", hit rate: " + std::to_string(hitRate());
    }
    
    void setCapacity(size_t newCapacity) override {
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = newCapacity;
    }
    
    void setStrategy(const std::string& strategy) override {
        std::lock_guard<std::mutex> lock(mutex_);
        strategy_ = strategy;
    }
    
    std::string getStrategy() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return strategy_;
    }
};

template<typename Key, typename Value>
std::shared_ptr<ICache<Key, Value>> CommonServicesFactory::createCache(
    const std::string& name,
    size_t capacity,
    const std::string& strategy) const {
    return std::make_shared<SimpleCacheImpl<Key, Value>>(name, capacity, strategy);
}

// 显式实例化常用的模板类型
template std::shared_ptr<ICache<std::string, std::vector<double>>> 
CommonServicesFactory::createCache<std::string, std::vector<double>>(
    const std::string&, size_t, const std::string&) const;

template std::shared_ptr<ICache<std::string, oscean::core_services::FileMetadata>>
CommonServicesFactory::createCache<std::string, oscean::core_services::FileMetadata>(
    const std::string&, size_t, const std::string&) const;

template std::shared_ptr<ICache<std::string, oscean::core_services::GridData>>
CommonServicesFactory::createCache<std::string, oscean::core_services::GridData>(
    const std::string&, size_t, const std::string&) const;

template std::shared_ptr<ICache<std::string, std::vector<unsigned char>>>
CommonServicesFactory::createCache<std::string, std::vector<unsigned char>>(
    const std::string&, size_t, const std::string&) const;

// === ServiceConfiguration 静态方法实现 ===

ServiceConfiguration ServiceConfiguration::createDefault() {
    ServiceConfiguration config;
    config.environment = Environment::PRODUCTION;
    config.threadPoolSize = 0; // auto-detect
    config.maxMemoryUsageMB = 512;
    config.enableSIMD = true;
    config.enableCaching = true;
    config.enablePerformanceMonitoring = true;
    config.logLevel = "info";
    return config;
}

ServiceConfiguration ServiceConfiguration::createForTesting() {
    ServiceConfiguration config;
    config.environment = Environment::TESTING;
    config.threadPoolSize = 2; // 测试环境使用较少线程
    config.maxMemoryUsageMB = 128; // 测试环境使用较少内存
    config.enableSIMD = false; // 测试环境禁用SIMD以提高兼容性
    config.enableCaching = true;
    config.enablePerformanceMonitoring = false; // 测试环境禁用性能监控
    config.logLevel = "debug"; // 测试环境使用详细日志
    return config;
}

ServiceConfiguration ServiceConfiguration::createForHPC() {
    ServiceConfiguration config;
    config.environment = Environment::HPC;
    config.threadPoolSize = 0; // auto-detect，HPC环境通常需要所有可用核心
    config.maxMemoryUsageMB = 2048; // HPC环境使用更多内存
    config.enableSIMD = true; // HPC环境启用SIMD优化
    config.enableCaching = true;
    config.enablePerformanceMonitoring = true; // HPC环境启用性能监控
    config.logLevel = "warn"; // HPC环境使用较少日志以提高性能
    return config;
}

// GdalGlobalInitializer 实现
std::once_flag GdalGlobalInitializer::initFlag_;

void GdalGlobalInitializer::initialize(const std::string& projDataPath) {
    std::call_once(initFlag_, [&]() {
        OSCEAN_LOG_INFO("GdalGlobalInitializer", "Performing one-time GDAL/PROJ global initialization...");

        // 1. 注册所有已知的GDAL驱动程序
        // 这是最关键的全局非可重入调用
        GDALAllRegister();

        // 2. 设置PROJ数据目录
        // 这是PROJ库找到其坐标系定义数据库所必需的
        if (!projDataPath.empty() && std::filesystem::exists(projDataPath)) {
            CPLSetConfigOption("PROJ_LIB", projDataPath.c_str());
            OSCEAN_LOG_INFO("GdalGlobalInitializer", "Set PROJ_LIB to: {}", projDataPath);
        } else {
            OSCEAN_LOG_WARN("GdalGlobalInitializer", "projDataPath is not provided or does not exist. PROJ may fail if it cannot find its data files.");
        }

        // 3. 设置全局共享的GDAL配置
        // 这些是合理的默认值，可以被线程本地配置覆盖
        CPLSetConfigOption("GDAL_PAM_ENABLED", "NO"); // 禁用代理文件写入，通常不需要
        CPLSetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"); // 避免在打开时扫描目录
        CPLSetConfigOption("VSI_CACHE", "TRUE"); // 启用虚拟文件系统缓存
        CPLSetConfigOption("VSI_CACHE_SIZE", "25000000"); // 25MB VSI缓存

        OSCEAN_LOG_INFO("GdalGlobalInitializer", "Global initialization complete.");
    });
}

} // namespace oscean::common_utils::infrastructure