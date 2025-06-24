/**
 * @file common_services_factory.h
 * @brief Common模块统一对外接口 - 纯工厂+依赖注入模式
 * 
 * 🎯 设计原则：
 * ✅ 唯一对外接口 - 外部只需包含此文件
 * ✅ 隐藏内部实现 - 不暴露具体实现类
 * ✅ 统一服务访问 - 所有服务通过工厂获取
 * ✅ 依赖注入支持 - 支持外部依赖注入
 * ✅ 环境适配 - 根据环境自动优化配置
 * 
 * 🚨 重要：这是Common模块的唯一对外接口
 * 
 * ❌ 禁止直接使用的类：
 * - UnifiedMemoryManager（通过 getMemoryManager() 获取）
 * - UnifiedSIMDManager（通过 getSIMDManager() 获取）
 * - LoggingManager（通过 getLogger() 获取）
 * - AppConfigLoader（通过 getConfigurationLoader() 获取）
 * - 其他任何具体实现类
 * 
 * ✅ 正确使用方式：
 * @code
 * // 创建工厂实例
 * auto factory = std::make_unique<CommonServicesFactory>();
 * 
 * // 通过工厂获取服务
 * auto memManager = factory->getMemoryManager();
 * auto logger = factory->getLogger();
 * auto simdManager = factory->getSIMDManager();
 * 
 * // 使用服务
 * void* buffer = memManager->allocate(1024);
 * logger->info("内存分配成功");
 * simdManager->vectorAdd(a, b, result, count);
 * @endcode
 * 
 * 🔄 迁移说明：
 * 如果您之前直接使用了具体实现类，请参考 common_utils.h 中的迁移指南。
 */

#pragma once

#include <memory>
#include <string>
#include <functional>
#include <vector>
#include <map>
#include <mutex>
#include <stdexcept>
#include <optional>
#include <cmath>  // 添加数学常量支持
#include <boost/optional.hpp>

// 引用实际的接口头文件
#include "common_utils/memory/memory_interfaces.h"
#include "common_utils/simd/isimd_manager.h"

// 前向声明
namespace oscean::common_utils::gpu {
    class OSCEANGPUFramework;
}

namespace oscean::common_utils::infrastructure {

// === 前向声明 - 使用正确的命名空间 ===

// 引用现有的类
class UnifiedThreadPoolManager;
class FileFormatDetector;
class AppConfigLoader;
class LoggingManager;

// 本模块的接口定义保持简化
namespace threading {
class IThreadPoolManager {
public:
    virtual ~IThreadPoolManager() = default;
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual size_t getThreadCount() const = 0;
    virtual bool isHealthy() const = 0;
};
}

namespace async {
class IAsyncExecutor {
public:
    virtual ~IAsyncExecutor() = default;
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool isHealthy() const = 0;
};
}

namespace performance {
class IPerformanceMonitor {
public:
    virtual ~IPerformanceMonitor() = default;
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool isHealthy() const = 0;
};
}

namespace utilities {
class IConfigurationLoader {
public:
    virtual ~IConfigurationLoader() = default;
    virtual bool loadFromFile(const std::string& filePath) = 0;
    virtual std::string getString(const std::string& key, const std::string& defaultValue = "") const = 0;
    virtual int getInt(const std::string& key, int defaultValue = 0) const = 0;
    virtual bool getBool(const std::string& key, bool defaultValue = false) const = 0;
    virtual bool has(const std::string& key) const = 0;
};

class IFileFormatDetector {
public:
    virtual ~IFileFormatDetector() = default;
    virtual std::string detectFormat(const std::string& filePath) const = 0;
    virtual bool validateFormat(const std::string& filePath, const std::string& expectedFormat) const = 0;
    virtual std::vector<std::string> getSupportedFormats() const = 0;
};
}

namespace logging {
class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void trace(const std::string& message) = 0;
    virtual void debug(const std::string& message) = 0;
    virtual void info(const std::string& message) = 0;
    virtual void warn(const std::string& message) = 0;
    virtual void error(const std::string& message) = 0;
    virtual void critical(const std::string& message) = 0;
    virtual void setLevel(const std::string& level) = 0;
    virtual void flush() = 0;
};
}

namespace fileio {
class ILargeFileProcessor {
public:
    virtual ~ILargeFileProcessor() = default;
    virtual std::string getFilePath() const = 0;
};
}

namespace time_utils {
class ITimeExtractorFactory {
public:
    virtual ~ITimeExtractorFactory() = default;
};
}

namespace format_utils {
class UnifiedFormatToolsFactory {
public:
    virtual ~UnifiedFormatToolsFactory() = default;
};
}

// === 🔧 核心接口定义 ===

/**
 * @brief 环境类型
 */
enum class Environment {
    DEVELOPMENT,
    TESTING, 
    PRODUCTION,
    HPC
};

/**
 * @brief 服务配置
 */
struct ServiceConfiguration {
    Environment environment = Environment::PRODUCTION;
    size_t threadPoolSize = 0;  // 0 = auto-detect
    size_t maxMemoryUsageMB = 512;
    bool enableSIMD = true;
    bool enableCaching = true;
    bool enablePerformanceMonitoring = true;
    std::string logLevel = "info";
    
    // 🔧 新增：共享线程池管理器支持
    std::shared_ptr<UnifiedThreadPoolManager> sharedThreadPoolManager = nullptr;
    
    static ServiceConfiguration createDefault();
    static ServiceConfiguration createForTesting();
    static ServiceConfiguration createForHPC();
};

/**
 * @brief 系统统计信息
 */
struct SystemStatistics {
    size_t totalMemoryUsageMB = 0;
    double threadPoolUtilization = 0.0;
    std::map<std::string, double> cacheHitRates;
    size_t activeFileProcessors = 0;
    double averageProcessingSpeedMBps = 0.0;
    size_t uptimeSeconds = 0;
    double memoryUsagePercent = 0.0;
};

/**
 * @brief 优化建议
 */
struct OptimizationSuggestion {
    std::string component;
    std::string suggestion;
    std::string category;
    std::string priority;
    double expectedImprovement = 0.0;
    bool isAutoApplicable = false;
};

/**
 * @brief Common服务工厂异常
 */
class CommonServicesException : public std::runtime_error {
public:
    explicit CommonServicesException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief 统一缓存接口 - CommonServices对外接口
 */
template<typename Key, typename Value>
class ICache {
public:
    virtual ~ICache() = default;
    
    // === 基础操作 ===
    virtual bool put(const Key& key, const Value& value) = 0;
    virtual std::optional<Value> get(const Key& key) = 0;
    virtual bool remove(const Key& key) = 0;
    virtual bool contains(const Key& key) const = 0;
    virtual void clear() = 0;
    virtual size_t size() const = 0;
    virtual size_t capacity() const = 0;
    virtual double hitRate() const = 0;
    
    // === 批量操作 ===
    virtual std::map<Key, Value> getBatch(const std::vector<Key>& keys) = 0;
    virtual void putBatch(const std::map<Key, Value>& items) = 0;
    virtual void removeBatch(const std::vector<Key>& keys) = 0;
    
    // === 统计和维护 ===
    virtual void evictExpired() = 0;
    virtual void optimize() = 0;
    virtual void resetStatistics() = 0;
    virtual std::string getReport() const = 0;
    
    // === 配置 ===
    virtual void setCapacity(size_t newCapacity) = 0;
    virtual void setStrategy(const std::string& strategy) = 0;
    virtual std::string getStrategy() const = 0;
};

/**
 * @brief 负责全局GDAL/PROJ库的唯一、线程安全的初始化。
 * 
 * 该类解决了在复杂多线程环境中分散初始化GDAL导致的死锁和竞争条件问题。
 * 它必须在应用程序启动时、任何GDAL/PROJ功能被使用之前，在main函数中被调用。
 * 
 * 使用方法:
 * ```cpp
 * int main() {
 *     oscean::common_utils::infrastructure::GdalGlobalInitializer::getInstance().initialize();
 *     // ... 应用程序的其余部分 ...
 * }
 * ```
 */
class GdalGlobalInitializer {
public:
    /**
     * @brief 获取单例实例。
     */
    static GdalGlobalInitializer& getInstance() {
        static GdalGlobalInitializer instance;
        return instance;
    }

    /**
     * @brief 执行一次性的全局初始化。
     * @param projDataPath PROJ数据目录的路径 (e.g., "path/to/proj/data")。如果为空，将尝试自动检测。
     */
    void initialize(const std::string& projDataPath = "");

private:
    GdalGlobalInitializer() = default;
    ~GdalGlobalInitializer() = default;

    GdalGlobalInitializer(const GdalGlobalInitializer&) = delete;
    GdalGlobalInitializer& operator=(const GdalGlobalInitializer&) = delete;

    static std::once_flag initFlag_;
};

/**
 * @class CommonServicesFactory
 * @brief Common模块统一对外接口
 *
 * 🎯 核心职责：
 * ✅ 提供所有Common服务的统一访问点
 * ✅ 管理服务生命周期和依赖关系
 * ✅ 支持多环境配置和优化
 * ✅ 隐藏内部实现复杂度
 */
class CommonServicesFactory {
public:
    /**
     * @brief 构造函数 - 通过配置文件路径创建
     * @param configPath 配置文件（如.json）的路径
     */
    explicit CommonServicesFactory(const std::string& configPath = "");

    /**
     * @brief 构造函数 - 通过预设配置创建
     * @param config 服务配置对象
     */
    explicit CommonServicesFactory(const ServiceConfiguration& config);
    
    /**
     * @brief 析构函数 - 安全关闭所有服务
     */
    ~CommonServicesFactory();

    // 禁用拷贝构造
    CommonServicesFactory(const CommonServicesFactory&) = delete;
    CommonServicesFactory& operator=(const CommonServicesFactory&) = delete;

    // 启用移动构造
    CommonServicesFactory(CommonServicesFactory&&) noexcept;
    CommonServicesFactory& operator=(CommonServicesFactory&&) noexcept;

    // === 服务获取接口 ===
    
    /**
     * @brief 获取内存管理器
     * @return 内存管理器的共享指针
     */
    std::shared_ptr<oscean::common_utils::memory::IMemoryManager> getMemoryManager() const;
    
    /**
     * @brief 获取线程池管理器
     * @return 线程池管理器的共享指针
     */
    std::shared_ptr<threading::IThreadPoolManager> getThreadPoolManager() const;
    
    /**
     * @brief 获取统一线程池管理器（具体实现）
     * 🔧 新增：支持访问UnifiedThreadPoolManager的具体功能
     */
    std::shared_ptr<UnifiedThreadPoolManager> getUnifiedThreadPoolManager() const;
    
    /**
     * @brief 获取异步执行器
     */
    std::shared_ptr<async::IAsyncExecutor> getAsyncExecutor() const;
    
    /**
     * @brief 获取性能监控器
     */
    std::shared_ptr<performance::IPerformanceMonitor> getPerformanceMonitor() const;
    
    /**
     * @brief 获取SIMD管理器
     * @return SIMD管理器的共享指针
     */
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> getSIMDManager() const;
    
    /**
     * @brief 获取GPU框架
     * @return GPU框架的共享指针
     * @note GPU框架是全局单例，但通过shared_ptr提供以保持接口一致性
     */
    std::shared_ptr<oscean::common_utils::gpu::OSCEANGPUFramework> getGPUFramework() const;
    
    /**
     * @brief 获取日志记录器
     * @return 日志记录器的共享指针
     */
    std::shared_ptr<logging::ILogger> getLogger() const;
    
    // === 📦 专用服务创建 ===
    
    /**
     * @brief 创建缓存实例
     * 
     * 通过CommonServices统一缓存接口提供完整缓存功能：
     * - 基础操作：get, put, remove, contains, clear
     * - 批量操作：getBatch, putBatch, removeBatch  
     * - 统计监控：hitRate, getReport, resetStatistics
     * - 维护优化：evictExpired, optimize
     * - 策略配置：setStrategy, setCapacity
     * 
     * 支持的缓存策略：LRU, LFU, TTL, FIFO, Adaptive
     * 
     * 注意：外部只使用CommonServices的ICache接口，
     * cache模块的ICacheManager等是纯内部实现
     * 
     * @param name 缓存名称，用于标识和监控
     * @param capacity 缓存容量
     * @param strategy 缓存策略，默认为LRU
     * @return CommonServices统一缓存接口实例
     */
    template<typename Key, typename Value>
    std::shared_ptr<ICache<Key, Value>> createCache(
        const std::string& name,
        size_t capacity = 10000,
        const std::string& strategy = "LRU"
    ) const;
    
    /**
     * @brief 创建大文件处理器
     */
    std::shared_ptr<fileio::ILargeFileProcessor> createFileProcessor(
        const std::string& filePath = ""
    ) const;
    
    // === ⏰ 时间处理服务 ===
    
    /**
     * @brief 获取时间提取器工厂
     */
    std::shared_ptr<time_utils::ITimeExtractorFactory> getTimeExtractorFactory() const;
    
    // === 🗂️ 格式处理服务 ===
    
    /**
     * @brief 获取格式工具工厂
     */
    std::shared_ptr<format_utils::UnifiedFormatToolsFactory> getFormatToolsFactory() const;
    
    // === ⚙️ 配置和工具服务 ===
    
    /**
     * @brief 获取配置加载器
     * @return 配置加载器的共享指针
     */
    std::shared_ptr<utilities::IConfigurationLoader> getConfigurationLoader() const;
    
    /**
     * @brief 获取文件格式检测器
     */
    std::shared_ptr<utilities::IFileFormatDetector> getFileFormatDetector() const;
    
    // === 📊 系统监控 ===
    
    SystemStatistics getSystemStatistics() const;
    std::string generateSystemReport() const;
    bool isHealthy() const;
    std::vector<std::string> getDiagnosticMessages() const;
    
    // === ⚡ 性能优化 ===
    
    std::vector<OptimizationSuggestion> getOptimizationSuggestions() const;
    void applyAutomaticOptimizations();
    
    // === 🔧 配置管理 ===
    
    const ServiceConfiguration& getConfiguration() const;
    void updateConfiguration(const ServiceConfiguration& config);
    
    // === 🔒 生命周期管理 ===
    
    /**
     * @brief 安全关闭所有服务
     */
    void shutdown();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace oscean::common_utils::infrastructure 