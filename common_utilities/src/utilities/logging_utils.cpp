#include "common_utils/utilities/logging_utils.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/async.h>
#include <stdexcept>
#include <iostream> // Added for potential error logging in createLogger
#include <mutex>
#include <spdlog/sinks/null_sink.h>
#include <algorithm> // 添加 std::transform 所需的头文件
#include <cctype>    // 添加 std::tolower 所需的头文件

namespace oscean::common_utils {

// --- Forward Declaration for early init ---
// Cannot call initialize directly from getInstance due to potential lock issues
// void initialize(const LoggingConfig& config = LoggingConfig()); 

// === 构造函数和析构函数实现 ===

LoggingManager::LoggingManager(const LoggingConfig& config) : config_(config) {
    // 构造函数保持最小化，实际初始化在earlyInitialize或initialize中完成
}

LoggingManager::~LoggingManager() {
    // 在析构时调用 shutdown 是合理的，以防忘记显式调用
    // 但显式调用 shutdown 仍然是最佳实践
    shutdown();
}

// 私有方法：早期初始化
void LoggingManager::earlyInitialize() {
    if (default_logger_) {
        return; // 已经初始化过了
    }
    
    try {
        // 创建一个基础的控制台日志器，用于早期日志记录
        default_logger_ = spdlog::stdout_color_mt("early_logger");
        default_logger_->set_level(spdlog::level::info);
        spdlog::set_default_logger(default_logger_);
    } catch (const std::exception&) {
        // 如果创建失败，创建一个空日志器避免崩溃
        default_logger_ = spdlog::default_logger();
    }
}

// === 🎯 静态全局访问实现 ===

// 静态成员变量定义
std::shared_ptr<LoggingManager> LoggingManager::global_instance_;
std::mutex LoggingManager::global_mutex_;

// 全局实例访问
LoggingManager& LoggingManager::getGlobalInstance() {
    std::lock_guard<std::mutex> lock(global_mutex_);
    if (!global_instance_) {
        global_instance_ = std::shared_ptr<LoggingManager>(new LoggingManager());
        global_instance_->earlyInitialize();
    }
    return *global_instance_;
}

// 全局配置
void LoggingManager::configureGlobal(const LoggingConfig& config) {
    auto& instance = getGlobalInstance();
    instance.initialize(config);
}

// 替换全局实例（用于工厂注入）
void LoggingManager::setGlobalInstance(std::shared_ptr<LoggingManager> instance) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    if (global_instance_) {
        global_instance_->shutdown();
    }
    global_instance_ = std::move(instance);
}

// 初始化日志系统
void LoggingManager::initialize(const LoggingConfig& config) {
    // Lock to ensure thread-safe initialization
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        // 最好在这里记录一条日志，而不是静默返回
        // 但由于此时 logger 可能还未完全准备好，需要小心处理
        // if(default_logger_) default_logger_->warn("LoggingManager is already initialized.");
        return;
    }
    
    config_ = config;
    
    // 设置异步日志队列（如果启用）
    if (config.async) {
        // 检查 spdlog 线程池是否已初始化，避免重复初始化
        // (注意：spdlog 的全局线程池管理可能需要更细致的考虑，尤其是在多模块场景下)
        // spdlog::init_thread_pool(8192, 1); // 移动到更合适的全局初始化位置可能更好
    }
    
    // 创建默认日志器 (覆盖 earlyInitialize 创建的 logger)
    try {
         // Shutdown previous default logger sinks if necessary? spdlog might handle this.
         default_logger_ = createLogger("app", config);
         spdlog::set_default_logger(default_logger_); // Set as default *after* successful creation
    } catch (const std::exception& e) {
        // 如果创建失败，至少提供一个基础的控制台输出
        std::cerr << "Failed to create default logger: " << e.what() << std::endl;
        // 此时可能无法使用正常的日志宏
        default_logger_ = spdlog::stderr_color_mt("fallback_logger");
        default_logger_->set_level(spdlog::level::warn);
        spdlog::set_default_logger(default_logger_);
        default_logger_->error("Default logger creation failed, using fallback stderr logger.");
    }
    
    // 设置全局刷新间隔（每隔3秒自动刷新）
    // spdlog::flush_every(std::chrono::seconds(3)); // 考虑是否所有 logger 都需要这个策略
    
    initialized_ = true;
}

// 获取默认日志器
std::shared_ptr<spdlog::logger> LoggingManager::getLogger() {
    // getInstance() ensures earlyInitialize was called, so default_logger_ should be valid
    return default_logger_;
}

// 获取指定名称的日志器
std::shared_ptr<spdlog::logger> LoggingManager::getModuleLogger(const std::string& module_name) {
    // Lock is still needed for thread-safe access to module_loggers_
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if default logger exists (it should after getInstance)
    if (!default_logger_) {
         // This case should ideally not happen if getInstance works correctly
         // Maybe return a temporary null logger or throw a different error?
         try {
             auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
             return std::make_shared<spdlog::logger>("temp_null", null_sink);
         } catch (...) { throw std::runtime_error("Default logger is null in getModuleLogger, critical error!"); }
    }

    // 检查模块日志器是否已存在
    auto it = module_loggers_.find(module_name);
    if (it != module_loggers_.end()) {
        return it->second;
    }
    
    // 创建新的模块日志器 (在锁保护下进行)
    std::shared_ptr<spdlog::logger> logger;
    try {
        // --- NOTE: config_ is read here, which is safe if config_ is set during initialize() and not modified later --- 
        logger = createLogger(module_name, config_); 
        module_loggers_[module_name] = logger;
    } catch (const std::exception& e) {
        // 创建模块 logger 失败，记录错误并返回默认 logger
        // --- Need lock to access default_logger_ safely if logging from multiple threads here ---
        // It's generally safer to log outside the critical section if possible, or use the existing default logger carefully.
        // For simplicity, we log using the potentially existing default logger.
        if (default_logger_) { 
            // Use default_logger_ directly as we hold the lock
            default_logger_->error("Failed to create module logger '{}': {}. Returning default logger.", module_name, e.what());
        } else {
            // Fallback if default_logger_ itself failed during init
            std::cerr << "Failed to create module logger \"" << module_name << "\": " << e.what() << ". Returning default logger." << std::endl;
        }
        // --- SIMPLIFIED: Return the existing default_logger_ --- 
        // logger = getLogger(); // Avoid potential recursive lock or re-check for initialized_
        logger = default_logger_; // Return the default logger (which might be the fallback logger)
    }
    return logger;
}

// 设置全局日志级别
void LoggingManager::setLevel(const std::string& level) {
    // --- REMOVED implicit initialization check --- 
    // if (!initialized_) {
    //     initialize(LoggingConfig());
    // }

    // It's safer to only proceed if actually initialized
    if (!initialized_) {
         std::cerr << "Warning: Attempted to set log level before LoggingManager was initialized." << std::endl;
         return;
    }
    
    auto level_enum = stringToLevel(level);
    
    // 设置所有已创建日志器的级别
    if (default_logger_) {
        default_logger_->set_level(level_enum);
    }
    for (auto const& [name, logger] : module_loggers_) { // Use const& for read-only iteration
        if (logger) { // Check if logger pointer is valid
             logger->set_level(level_enum);
        }
    }
    
    // 更新 spdlog 的全局级别（会影响未来创建的 logger，但不会改动已创建的）
    // spdlog::set_level(level_enum);
    // 注意：直接修改 logger 的 level 更可靠
}

// 刷新所有日志缓冲区
void LoggingManager::flushAll() {
    if (initialized_) {
        // 刷新默认日志器
        if (default_logger_) {
            default_logger_->flush();
        }
        
        // 刷新所有模块日志器
        for (auto const& [name, logger] : module_loggers_) { // Use const&
            if (logger) {
                logger->flush();
            }
        }
    }
}

// 关闭日志系统
void LoggingManager::shutdown() {
    if (!initialized_) {
        return;
    }
    
    // 刷新并关闭所有日志器
    // spdlog::shutdown() 会关闭所有注册的 logger 并清理线程池
    spdlog::shutdown();
    
    // 清理内部状态
    module_loggers_.clear();
    default_logger_ = nullptr;
    initialized_ = false;
}

// 转换字符串日志级别到spdlog日志级别
spdlog::level::level_enum LoggingManager::stringToLevel(const std::string& level) {
    std::string lower_level = level;
    // 转换为小写以进行不区分大小写的比较
    std::transform(lower_level.begin(), lower_level.end(), lower_level.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (lower_level == "trace") return spdlog::level::trace;
    if (lower_level == "debug") return spdlog::level::debug;
    if (lower_level == "info") return spdlog::level::info;
    if (lower_level == "warn") return spdlog::level::warn;
    if (lower_level == "error") return spdlog::level::err;
    if (lower_level == "critical") return spdlog::level::critical;
    if (lower_level == "off") return spdlog::level::off;
    
    // 如果无法识别，记录警告并返回默认级别
    // --- MODIFIED: Avoid logging here as default_logger might not be ready ---
    // if(default_logger_) {
    //     default_logger_->warn("Unknown log level string: '{}'. Using default level 'info'.", level);
    // } else {
    //     std::cerr << "Unknown log level string: '" << level << "'. Using default level 'info'." << std::endl;
    // }
    // --- END MODIFICATION ---
    return spdlog::level::info; // 默认级别
}

std::shared_ptr<spdlog::logger> LoggingManager::createLogger(const std::string& name, const LoggingConfig& config) {
    std::vector<spdlog::sink_ptr> sinks;
    
    // 添加控制台输出
    if (config.enable_console) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(stringToLevel(config.console_level));
        // 设置一个通用的模式，或者允许通过 config 配置
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
        sinks.push_back(console_sink);
    }
    
    // 添加文件输出
    if (config.enable_file) {
        try {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                config.log_filename,
                config.max_file_size,
                config.max_files
            );
            file_sink->set_level(stringToLevel(config.file_level));
            // 设置一个通用的模式，或者允许通过 config 配置
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");
            sinks.push_back(file_sink);
        } catch (const spdlog::spdlog_ex& ex) {
            // 创建文件 sink 失败的处理逻辑
            std::cerr << "Failed to create rotating file sink for '" << name << "' at '" << config.log_filename << "': " << ex.what() << std::endl;
            // 根据情况决定是否抛出异常或仅使用其他 sink
            // throw; // 或者不抛出，让 logger 至少能用控制台输出
            if (sinks.empty()) { // 如果连控制台都没有，至少给个错误提示
                 throw std::runtime_error("Cannot create any log sinks for logger: " + name);
            }
        }
    }
    
    // 如果没有任何 sink 配置，至少提供一个 stderr sink，避免 logger 无效
    if (sinks.empty()) {
        auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        stderr_sink->set_level(spdlog::level::warn); // 默认警告级别
        stderr_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [NO SINKS CONFIGURED] %v");
        sinks.push_back(stderr_sink);
    }

    std::shared_ptr<spdlog::logger> logger;
    
    // 创建异步或同步日志器
    if (config.async) {
        // 确保 spdlog 的线程池已初始化。通常全局初始化一次即可。
        // spdlog::init_thread_pool(8192, 1); // 需要放在全局初始化代码中
        logger = std::make_shared<spdlog::async_logger>(
            name,
            sinks.begin(),
            sinks.end(),
            spdlog::thread_pool(), // 获取全局线程池
            spdlog::async_overflow_policy::block // 或 discard
        );
    } else {
        logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
    }
    
    // 设置日志器级别为最低级别，让具体的sink来过滤
    logger->set_level(spdlog::level::trace);
    // 设置自动刷新策略，例如错误级别以上自动刷新
    logger->flush_on(spdlog::level::err);
    
    // 注册到spdlog，这样可以通过 spdlog::get(name) 获取，但也增加了全局状态
    try {
        spdlog::register_logger(logger);
    } catch (const spdlog::spdlog_ex& ex) {
        // 如果注册失败（例如名称冲突），记录日志但不抛出，因为我们已经持有 logger
        if(default_logger_) {
             default_logger_->warn("Failed to register logger '{}': {}. Logger is still functional.", name, ex.what());
        } else {
             std::cerr << "Failed to register logger '" << name << "': " << ex.what() << ". Logger is still functional." << std::endl;
        }
    }
    
    return logger;
}

} // namespace oscean::common_utils 