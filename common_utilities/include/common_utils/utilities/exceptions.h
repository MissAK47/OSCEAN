/**
 * @file exceptions.h
 * @brief 项目统一异常体系 - 基础异常定义
 * 
 * 🎯 重构目标：
 * ✅ 这是项目中唯一的基础异常定义文件
 * ✅ 提供所有模块都能使用的通用异常基类
 * ✅ core_services 的异常将从这些基础异常继承
 * ✅ 支持错误码、位置信息等扩展功能
 * 
 * 异常层次结构：
 * - OsceanBaseException (根异常)
 *   ├─ InitializationException
 *   ├─ ConfigurationException  
 *   ├─ IOException
 *   ├─ ResourceNotFoundException
 *   ├─ ValidationException
 *   ├─ TimeoutException
 *   ├─ OperationNotSupportedException
 *   └─ ServiceException (core_services 异常的基类)
 */

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

namespace oscean {
namespace common_utils {

/**
 * @brief 项目根异常类 - 所有自定义异常的基类
 * 
 * 提供统一的异常接口，支持错误码、位置信息等扩展功能
 */
class OsceanBaseException : public std::runtime_error {
public:
    /**
     * @brief 构造函数
     * @param message 异常消息
     */
    explicit OsceanBaseException(const std::string& message)
        : std::runtime_error(message) {}

    /**
     * @brief 带错误码的构造函数
     * @param message 异常消息
     * @param code 错误码
     */
    OsceanBaseException(const std::string& message, int code)
        : std::runtime_error(message), code_(code) {}

    /**
     * @brief 获取错误码
     * @return int 错误码
     */
    int getCode() const noexcept { return code_; }

protected:
    int code_ = 0; // 默认错误码为0
};

// ============================================================================
// 通用基础异常类型
// ============================================================================

/**
 * @brief 初始化异常 - 用于模块或组件初始化失败
 */
class InitializationException : public OsceanBaseException {
public:
    explicit InitializationException(const std::string& message)
        : OsceanBaseException(message) {}

    InitializationException(const std::string& message, int code)
        : OsceanBaseException(message, code) {}
};

/**
 * @brief 配置异常 - 用于配置文件或参数错误
 */
class ConfigurationException : public OsceanBaseException {
public:
    explicit ConfigurationException(const std::string& message)
        : OsceanBaseException(message) {}

    ConfigurationException(const std::string& message, int code)
        : OsceanBaseException(message, code) {}
};

/**
 * @brief I/O 异常 - 用于文件、网络等 I/O 操作错误
 */
class IOException : public OsceanBaseException {
public:
    explicit IOException(const std::string& message)
        : OsceanBaseException(message) {}

    IOException(const std::string& message, int code)
        : OsceanBaseException(message, code) {}
};

/**
 * @brief 资源未找到异常 - 用于文件、数据等资源不存在
 */
class ResourceNotFoundException : public OsceanBaseException {
public:
    explicit ResourceNotFoundException(const std::string& message)
        : OsceanBaseException(message) {}

    ResourceNotFoundException(const std::string& message, int code)
        : OsceanBaseException(message, code) {}
};

/**
 * @brief 验证异常 - 用于参数验证、数据验证失败
 */
class ValidationException : public OsceanBaseException {
public:
    explicit ValidationException(const std::string& message)
        : OsceanBaseException(message) {}

    ValidationException(const std::string& message, int code)
        : OsceanBaseException(message, code) {}
};

/**
 * @brief 超时异常 - 用于操作超时
 */
class TimeoutException : public OsceanBaseException {
public:
    explicit TimeoutException(const std::string& message)
        : OsceanBaseException(message) {}

    TimeoutException(const std::string& message, int code)
        : OsceanBaseException(message, code) {}
};

/**
 * @brief 操作不支持异常 - 用于不支持的功能或操作
 */
class OperationNotSupportedException : public OsceanBaseException {
public:
    explicit OperationNotSupportedException(const std::string& message)
        : OsceanBaseException(message) {}

    OperationNotSupportedException(const std::string& message, int code)
        : OsceanBaseException(message, code) {}
};

/**
 * @brief 服务异常基类 - core_services 模块异常的基类
 * 
 * 这是专为 core_services 层设计的异常基类，
 * core_services 中的所有业务异常都应该从此类继承
 */
class ServiceException : public OsceanBaseException {
public:
    explicit ServiceException(const std::string& message)
        : OsceanBaseException(message) {}

    ServiceException(const std::string& message, int code)
        : OsceanBaseException(message, code) {}
};

// ============================================================================
// 异常工具宏和函数
// ============================================================================

/**
 * @brief 生成带文件、行号、函数名的异常消息辅助宏
 */
#define MAKE_ERROR_MSG(msg) \
    (std::stringstream() << msg << " (at " << __FILE__ << ":" << __LINE__ << ", in " << __FUNCTION__ << ")").str()

/**
 * @brief 抛出带位置信息的异常辅助宏
 */
#define THROW_OSCEAN_EXCEPTION(ExceptionType, msg) \
    throw ExceptionType(MAKE_ERROR_MSG(msg))

#define THROW_OSCEAN_EXCEPTION_CODE(ExceptionType, msg, code) \
    throw ExceptionType(MAKE_ERROR_MSG(msg), code)

// ============================================================================
// 向后兼容的别名 (过渡期使用)
// ============================================================================

// 为了向后兼容，保留旧的异常名称作为别名
using AppBaseException = OsceanBaseException;          // 向后兼容
using InitializationError = InitializationException;   // 向后兼容
using ConfigurationError = ConfigurationException;     // 向后兼容
using FileError = IOException;                          // 向后兼容，文件错误是 I/O 错误的子集
using ResourceNotFoundError = ResourceNotFoundException; // 向后兼容
using ValidationError = ValidationException;           // 向后兼容
using TimeoutError = TimeoutException;                 // 向后兼容

} // namespace common_utils
} // namespace oscean

/**
 * 🎯 使用指南：
 * 
 * 1. 在 common_utilities 模块中：
 *    ```cpp
 *    #include "common_utils/utilities/exceptions.h"
 *    throw InitializationException("Failed to initialize memory manager");
 *    ```
 * 
 * 2. 在 core_services 模块中：
 *    ```cpp
 *    #include "common_utils/utilities/exceptions.h"
 *    
 *    // 业务异常从 ServiceException 继承
 *    class DataAccessException : public ServiceException {
 *        // ... 实现
 *    };
 *    ```
 * 
 * 3. 统一异常捕获：
 *    ```cpp
 *    try {
 *        // ... some operation
 *    } catch (const OsceanBaseException& e) {
 *        // 捕获所有项目异常
 *        logger->error("OSCEAN Exception: {} (code: {})", e.what(), e.getCode());
 *    }
 *    ```
 */