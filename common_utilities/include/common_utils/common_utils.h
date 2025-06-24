/**
 * @file common_utils.h
 * @brief Common Utilities 模块统一头文件
 * 
 * 🎯 重构说明：
 * ✅ 提供统一的包含入口，简化其他模块的依赖管理
 * ✅ 按功能分组组织头文件包含
 * ✅ 支持选择性包含以减少编译时间
 */

#pragma once

// === 核心基础设施 ===
#include "utilities/boost_config.h"
#include "utilities/logging_utils.h"
#include "utilities/exceptions.h"
#include "utilities/file_format_detector.h"
#include "utilities/string_utils.h"

// === 异步处理框架 ===
#include "async/async_config.h"
#include "async/async_types.h"
#include "async/async_task.h"
#include "async/async_framework.h"

// === 内存管理 ===
#include "memory/memory_config.h"
#include "memory/memory_interfaces.h"
#include "memory/memory_allocators.h"
#include "memory/memory_manager_unified.h"
#include "memory/memory_statistics.h"

// === 缓存系统 ===
#include "cache/cache_config.h"
#include "cache/icache_manager.h"
#include "cache/cache_strategies.h"

// === SIMD 优化 ===
#include "simd/simd_config.h"
#include "simd/isimd_manager.h"
#include "simd/simd_manager_unified.h"

// === 时间处理 ===
#include "time/time_types.h"
#include "time/time_interfaces.h"
#include "time/time_resolution.h"
#include "time/time_range.h"
#include "time/time_calendar.h"
#include "time/time_services.h"

// === 基础设施 ===
#include "infrastructure/unified_thread_pool_manager.h"
#include "infrastructure/performance_monitor.h"
#include "infrastructure/common_services_factory.h"
#include "infrastructure/large_file_processor.h"

/**
 * @namespace oscean::common_utils
 * @brief Common Utilities 模块命名空间
 * 
 * 包含所有通用工具和基础设施组件
 */
namespace oscean::common_utils {

/**
 * @brief 初始化 Common Utilities 模块
 * @param config 可选的配置参数
 * @return 初始化是否成功
 */
bool initialize(const std::string& config = "");

/**
 * @brief 清理 Common Utilities 模块
 */
void cleanup();

/**
 * @brief 获取模块版本信息
 * @return 版本字符串
 */
std::string getVersion();

} // namespace oscean::common_utils

// === 便捷宏定义 ===

/**
 * @brief 便捷的日志记录宏
 */
#define COMMON_UTILS_LOG_INFO(msg) \
    do { \
        auto logger = oscean::common_utils::LoggingUtils::getLogger("CommonUtils"); \
        if (logger) logger->info(msg); \
    } while(0)

#define COMMON_UTILS_LOG_ERROR(msg) \
    do { \
        auto logger = oscean::common_utils::LoggingUtils::getLogger("CommonUtils"); \
        if (logger) logger->error(msg); \
    } while(0)

/**
 * @brief 便捷的异常处理宏
 */
#define COMMON_UTILS_TRY_CATCH(code, fallback) \
    try { \
        code; \
    } catch (const std::exception& e) { \
        COMMON_UTILS_LOG_ERROR("Exception: " + std::string(e.what())); \
        fallback; \
    }

/**
 * @brief 便捷的资源管理宏
 */
// TODO: 实现ResourceManager后再启用此宏
// #define COMMON_UTILS_SCOPED_RESOURCE(type, name, init) \
//     auto name = oscean::common_utils::infrastructure::ResourceManager::getInstance().acquire<type>(init); 