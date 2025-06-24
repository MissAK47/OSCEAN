/**
 * @file boost_config.h
 * @brief 项目统一 Boost 库配置头文件 - 替代所有其他 boost_config.h
 * 
 * 🎯 重构目标：
 * ✅ 这是项目中唯一的 Boost 配置文件
 * ✅ 整合了来自 core_service_interfaces 的所有配置
 * ✅ 支持 boost::future, boost::thread, boost::asio 等核心功能
 * ✅ 跨平台兼容 (Windows/Linux/macOS)
 * 
 * 🔴 重要：core_service_interfaces/include/core_services/boost_config.h 已被移除
 * 
 * 使用方法：
 * ```cpp
 * #include "common_utils/utilities/boost_config.h"  // 项目中唯一的 Boost 配置
 * #include <boost/thread/future.hpp>               // boost::future 支持
 * #include <boost/asio.hpp>                        // boost::asio 支持（可选）
 * ```
 */

#pragma once

// 防止重复定义
#ifndef OSCEAN_BOOST_CONFIG_UNIFIED_H
#define OSCEAN_BOOST_CONFIG_UNIFIED_H

// ============================================================================
// 平台特定配置 - 整合版本
// ============================================================================

#ifdef _WIN32
    // Windows 平台配置 - 整合 core_service_interfaces 和 common_utilities 的配置
    #ifndef _WIN32_WINNT
    #define _WIN32_WINNT 0x0A00  // Windows 10 or later (更新版本)
    #endif
    
    #ifndef NOMINMAX
    #define NOMINMAX  // 避免 Windows.h 的 min/max 宏冲突
    #endif
    
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN  // 减少 Windows.h 包含的内容
    #endif
#endif

// ============================================================================
// Boost.Thread Future 支持宏定义 - 核心功能
// ============================================================================

// 🔴 核心：启用 boost::future 功能 (从 core_service_interfaces 继承)
#ifndef BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE 1
#endif

#ifndef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION 1
#endif

#ifndef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY 1
#endif

#ifndef BOOST_THREAD_PROVIDES_FUTURE_ASYNC
#define BOOST_THREAD_PROVIDES_FUTURE_ASYNC 1
#endif

#ifndef BOOST_THREAD_USES_MOVE
#define BOOST_THREAD_USES_MOVE 1
#endif

// Boost 1.87.0+ 需要的宏 (从 core_service_interfaces 继承)
#ifndef BOOST_THREAD_FUTURE_USES_OPTIONAL
#define BOOST_THREAD_FUTURE_USES_OPTIONAL 1
#endif

#ifndef BOOST_THREAD_VERSION
#define BOOST_THREAD_VERSION 5
#endif

// 启用 boost::thread 的异步特性 (common_utilities 扩展)
#ifndef BOOST_THREAD_PROVIDES_EXECUTORS
#define BOOST_THREAD_PROVIDES_EXECUTORS 1
#endif

// ============================================================================
// 编译器特定配置 - 整合版本
// ============================================================================

#ifdef _MSC_VER
    // MSVC 特定配置 - 整合两个版本的警告处理
    #pragma warning(push)
    #pragma warning(disable: 4996)  // 禁用已弃用函数警告
    #pragma warning(disable: 4251)  // 禁用 DLL 接口警告
    #pragma warning(disable: 4275)  // 禁用 DLL 基类警告
    #pragma warning(disable: 4530)  // 禁用异常处理警告
    #pragma warning(disable: 4834)  // 禁用 [[nodiscard]] 警告
#endif

#ifdef __GNUC__
    // GCC 特定配置
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#ifdef __clang__
    // Clang 特定配置
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

// ============================================================================
// boost::asio 配置 - 条件性启用
// ============================================================================

// 检测是否需要 boost::asio 支持
#if defined(OSCEAN_ENABLE_BOOST_ASIO) || defined(BOOST_ASIO_HPP)

// 只有在明确需要 boost::asio 时才处理 WinSock 冲突
#ifdef _WIN32
    // 强制防止Windows socket头文件冲突
    #ifndef BOOST_ASIO_NO_WIN32_LEAN_AND_MEAN
    #define BOOST_ASIO_NO_WIN32_LEAN_AND_MEAN
    #endif
    
    #ifndef BOOST_ASIO_HAS_WINDOWS_SOCKETS
    #define BOOST_ASIO_HAS_WINDOWS_SOCKETS 1
    #endif
    
    #ifndef BOOST_ASIO_ENABLE_BUFFER_DEBUGGING
    #define BOOST_ASIO_ENABLE_BUFFER_DEBUGGING 0  // 生产环境关闭调试
    #endif
    
    #ifndef BOOST_ASIO_NO_DEPRECATED
    #define BOOST_ASIO_NO_DEPRECATED 0  // 允许使用已弃用功能（兼容性）
    #endif
    
    #ifndef BOOST_ASIO_HEADER_ONLY
    #define BOOST_ASIO_HEADER_ONLY 0  // 使用编译版本，非 header-only
    #endif
    
    #ifndef BOOST_ASIO_SEPARATE_COMPILATION
    #define BOOST_ASIO_SEPARATE_COMPILATION 0  // 不使用分离编译
    #endif
    
    // 防止winsock2.h和mswsock.h冲突
    #ifndef BOOST_ASIO_WINDOWS_RUNTIME
    #define BOOST_ASIO_WINDOWS_RUNTIME 0
    #endif
#endif

#endif // OSCEAN_ENABLE_BOOST_ASIO

// ============================================================================
// 项目特定配置 - 海洋数据处理优化
// ============================================================================

// 异步框架配置 - 基于 boost::thread
#ifndef BOOST_THREAD_POOL_USE_GENERIC_SHARED_PTR
#define BOOST_THREAD_POOL_USE_GENERIC_SHARED_PTR 1
#endif

// ============================================================================
// 使用指导宏定义
// ============================================================================

// 为需要 boost::asio 的模块提供启用宏
#define OSCEAN_ENABLE_BOOST_ASIO_IN_MODULE() \
    _Pragma("message(\"Enabling boost::asio for this module\")")

// 为不需要 boost::asio 的模块提供标记
#define OSCEAN_NO_BOOST_ASIO_MODULE() \
    _Pragma("message(\"This module does not use boost::asio\")")

// ============================================================================
// Boost.SmartPtr 智能指针配置 - 替代 std 智能指针
// ============================================================================

// 启用 boost 智能指针功能
#ifndef BOOST_SMART_PTR_HPP
#define BOOST_SMART_PTR_HPP
#endif

// 项目版本标识
#define OSCEAN_BOOST_CONFIG_VERSION "2.0.0"
#define OSCEAN_BOOST_CONFIG_UNIFIED 1

#endif // OSCEAN_BOOST_CONFIG_UNIFIED_H

/**
 * 🎯 统一使用指南:
 * 
 * 1. 替代所有旧的 boost_config.h 包含：
 *    ```cpp
 *    // 旧的方式（已废弃）：
 *    // #include <core_services/boost_config.h>
 *    
 *    // 新的统一方式：
 *    #include "common_utils/utilities/boost_config.h"
 *    ```
 * 
 * 2. 对于不使用 boost::asio 的模块（如 cache、memory 等）：
 *    ```cpp
 *    #include "common_utils/utilities/boost_config.h"
 *    OSCEAN_NO_BOOST_ASIO_MODULE();
 *    #include <boost/thread/future.hpp>  // 安全使用 boost::future
 *    #include <boost/smart_ptr/scoped_ptr.hpp>  // 替代 std::unique_ptr
 *    #include <boost/smart_ptr/shared_ptr.hpp>  // 使用 boost::shared_ptr
 *    ```
 * 
 * 3. 对于使用 boost::asio 的模块（如 streaming、网络部分）：
 *    ```cpp
 *    #define OSCEAN_ENABLE_BOOST_ASIO
 *    #include "common_utils/utilities/boost_config.h"
 *    OSCEAN_ENABLE_BOOST_ASIO_IN_MODULE();
 *    #include <boost/asio.hpp>
 *    #include <boost/smart_ptr.hpp>  // 全部智能指针功能
 *    ```
 * 
 * 4. 智能指针替换指南：
 *    ```cpp
 *    // 替换前：
 *    std::unique_ptr<Type> ptr = std::make_unique<Type>();
 *    
 *    // 替换后：
 *    boost::scoped_ptr<Type> ptr(new Type());
 *    // 或者使用 boost::make_unique (需要较新版本的 boost)
 *    ```
 */ 