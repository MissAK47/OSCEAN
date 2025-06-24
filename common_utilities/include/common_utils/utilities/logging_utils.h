/**
 * @file logging.h
 * @brief 定义日志管理系统接口
 */

#pragma once

#include <string>
#include <memory>
#include <unordered_map>

// Try including fmt core explicitly before spdlog
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <mutex>

namespace oscean::common_utils {

/**
 * @brief 日志配置结构体
 */
struct LoggingConfig {
    bool async = false;               ///< 是否使用异步日志
    bool enable_console = true;       ///< 是否启用控制台日志
    bool enable_file = false;         ///< 是否启用文件日志
    std::string console_level = "info";    ///< 控制台日志级别
    std::string file_level = "trace";       ///< 文件日志级别
    std::string log_filename = "app.log";    ///< 日志文件名称
    size_t max_file_size = 1048576 * 5;      ///< 日志文件最大大小 (5MB)
    size_t max_files = 3;                     ///< 最大日志文件数
};

/**
 * @brief 日志管理器类 - 支持静态访问和工厂创建两种模式
 * 
 * 🎯 设计理念：
 * ✅ 静态全局访问 - 提供简单的LOG_XXX宏
 * ✅ 工厂集成支持 - 可通过CommonServicesFactory配置
 * ✅ 线程安全 - 支持多线程环境
 * ✅ 早期可用 - 程序启动即可使用
 * 
 * 使用方式：
 * @code
 * // 方式1：简单静态访问（推荐用于一般日志）
 * LOG_INFO("系统启动");
 * LOG_ERROR("发生错误: {}", errorMsg);
 * 
 * // 方式2：通过工厂配置（推荐用于高级配置）
 * auto factory = std::make_unique<CommonServicesFactory>();
 * auto logger = factory->getLogger(); // 获取配置后的实例
 * logger->info("通过工厂配置的日志");
 * @endcode
 */
class LoggingManager {
public:
    /**
     * @brief 构造函数（支持工厂创建）
     * @param config 日志配置
     */
    explicit LoggingManager(const LoggingConfig& config = LoggingConfig{});
    
    /**
     * @brief 析构函数
     */
    ~LoggingManager();

    // 禁用拷贝，允许移动
    LoggingManager(const LoggingManager&) = delete;
    LoggingManager& operator=(const LoggingManager&) = delete;
    LoggingManager(LoggingManager&&) = default;
    LoggingManager& operator=(LoggingManager&&) = default;

    /**
     * @brief 初始化日志系统
     * @param config 日志配置
     */
    void initialize(const LoggingConfig& config = LoggingConfig());

    /**
     * @brief 获取全局日志器
     * @return 全局日志器指针
     */
    std::shared_ptr<spdlog::logger> getLogger();

    /**
     * @brief 获取指定模块的日志器
     * @param module_name 模块名称
     * @return 模块对应的日志器指针
     */
    std::shared_ptr<spdlog::logger> getModuleLogger(const std::string& module_name);

    /**
     * @brief 设置全局日志级别
     * @param level 日志级别
     */
    void setLevel(const std::string& level);

    /**
     * @brief 刷新所有日志缓冲区
     */
    void flushAll();

    /**
     * @brief 关闭日志系统
     */
    void shutdown();

    /**
     * @brief 获取指定名称的日志记录器
     * @param name 日志记录器名称
     * @return 日志记录器实例
     */
    inline std::shared_ptr<spdlog::logger> getLogger(const std::string& name) {
        auto logger = spdlog::get(name);
        if (!logger) {
            logger = spdlog::stdout_color_mt(name);
        }
        return logger;
    }

    // === 🎯 静态全局访问接口 ===
    
    /**
     * @brief 获取全局日志管理器实例
     * 
     * 注意：这不是传统单例模式，而是提供全局访问点
     * - 支持运行时重新配置
     * - 支持通过CommonServicesFactory替换实例
     * - 线程安全
     */
    static LoggingManager& getGlobalInstance();
    
    /**
     * @brief 全局配置日志系统
     * @param config 全局日志配置
     */
    static void configureGlobal(const LoggingConfig& config);
    
    /**
     * @brief 替换全局实例（用于工厂注入）
     * @param instance 新的日志管理器实例
     */
    static void setGlobalInstance(std::shared_ptr<LoggingManager> instance);

private:
    void earlyInitialize();
    
    std::shared_ptr<spdlog::logger> createLogger(const std::string& name, const LoggingConfig& config);
    
    bool initialized_ = false;
    LoggingConfig config_;
    std::shared_ptr<spdlog::logger> default_logger_ = nullptr;
    std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> module_loggers_;
    mutable std::mutex mutex_;

    // 将字符串日志级别转换为spdlog日志级别
    spdlog::level::level_enum stringToLevel(const std::string& level);
    
    // 静态实例管理
    static std::shared_ptr<LoggingManager> global_instance_;
    static std::mutex global_mutex_;
};

// === 🎯 推荐的日志宏 - 简单易用 ===

/**
 * 全局日志宏 - 推荐的日志使用方式
 * 
 * 特点：
 * - 简单易用，无需创建工厂
 * - 性能优化，编译时优化
 * - 线程安全
 * - 支持格式化字符串
 */
#define LOG_TRACE(...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getLogger()) { \
            logger->trace(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_DEBUG(...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getLogger()) { \
            logger->debug(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_INFO(...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getLogger()) { \
            logger->info(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_WARN(...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getLogger()) { \
            logger->warn(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_ERROR(...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getLogger()) { \
            logger->error(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_CRITICAL(...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getLogger()) { \
            logger->critical(__VA_ARGS__); \
        } \
    } while(0)

// 模块专用日志宏
#define LOG_MODULE_TRACE(module, ...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getModuleLogger(module)) { \
            logger->trace(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_MODULE_DEBUG(module, ...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getModuleLogger(module)) { \
            logger->debug(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_MODULE_INFO(module, ...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getModuleLogger(module)) { \
            logger->info(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_MODULE_WARN(module, ...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getModuleLogger(module)) { \
            logger->warn(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_MODULE_ERROR(module, ...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getModuleLogger(module)) { \
            logger->error(__VA_ARGS__); \
        } \
    } while(0)

#define LOG_MODULE_CRITICAL(module, ...) \
    do { \
        auto& manager = oscean::common_utils::LoggingManager::getGlobalInstance(); \
        if (auto logger = manager.getModuleLogger(module)) { \
            logger->critical(__VA_ARGS__); \
        } \
    } while(0)

// === 🎯 便捷访问函数 ===

/**
 * @brief 获取默认日志器
 * @return 默认日志器实例
 */
inline std::shared_ptr<spdlog::logger> getLogger() {
    return LoggingManager::getGlobalInstance().getLogger();
}

/**
 * @brief 获取模块专用日志器
 * @param module_name 模块名称
 * @return 模块日志器实例
 */
inline std::shared_ptr<spdlog::logger> getModuleLogger(const std::string& module_name) {
    return LoggingManager::getGlobalInstance().getModuleLogger(module_name);
}

/**
 * @brief 获取后备日志器
 * 
 * 当无法获取正常的日志器时，提供一个默认的控制台日志器
 * 
 * @param loggerName 日志器名称
 * @return 日志器实例
 */
inline std::shared_ptr<spdlog::logger> getFallbackLogger(const std::string& loggerName) {
    auto logger = spdlog::get(loggerName);
    if (!logger) {
        logger = spdlog::stdout_color_mt(loggerName);
        logger->set_level(spdlog::level::debug);
    }
    return logger;
}

// === 🎯 项目级别的OSCEAN_LOG宏 - 向后兼容和统一接口 ===

/**
 * OSCEAN项目专用日志宏
 * 
 * 这些宏是OSCEAN项目的标准日志接口，映射到上述LOG_MODULE_*宏
 * 提供与项目现有代码的兼容性
 */
#define OSCEAN_LOG_TRACE(module, ...) LOG_MODULE_TRACE(module, __VA_ARGS__)
#define OSCEAN_LOG_DEBUG(module, ...) LOG_MODULE_DEBUG(module, __VA_ARGS__)
#define OSCEAN_LOG_INFO(module, ...) LOG_MODULE_INFO(module, __VA_ARGS__)
#define OSCEAN_LOG_WARN(module, ...) LOG_MODULE_WARN(module, __VA_ARGS__)
#define OSCEAN_LOG_ERROR(module, ...) LOG_MODULE_ERROR(module, __VA_ARGS__)
#define OSCEAN_LOG_CRITICAL(module, ...) LOG_MODULE_CRITICAL(module, __VA_ARGS__)

// === 🎯 使用示例 ===

/**
 * @example 日志使用示例
 * @code
 * // 基本用法 - 直接使用宏（推荐）
 * LOG_INFO("应用程序启动");
 * LOG_ERROR("发生错误: {}", error_message);
 * LOG_MODULE_DEBUG("DataAccess", "加载文件: {}", filename);
 * 
 * // OSCEAN项目标准用法
 * OSCEAN_LOG_INFO("WorkflowEngine", "工作流开始执行");
 * OSCEAN_LOG_ERROR("MetadataExtractor", "提取失败: {}", error);
 * 
 * // 高级配置 - 通过工厂（可选）
 * LoggingConfig config;
 * config.enable_file = true;
 * config.log_filename = "app.log";
 * LoggingManager::configureGlobal(config);
 * 
 * // 或者通过CommonServicesFactory配置
 * auto factory = std::make_unique<CommonServicesFactory>();
 * auto logger = factory->getLogger(); // 获取工厂配置的日志器
 * 
 * // 获取原始spdlog日志器进行高级操作
 * auto rawLogger = getLogger();
 * rawLogger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
 * @endcode
 */

} // namespace oscean::common_utils 