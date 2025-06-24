#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

namespace oscean {
namespace common_utils {

/**
 * @brief Base class for application-specific exceptions.
 * 
 * Provides a common base for catching all application-related errors.
 */
class AppBaseException : public std::runtime_error {
public:
    /**
     * @brief 构造函数
     * @param message 异常消息
     */
    explicit AppBaseException(const std::string& message)
        : std::runtime_error(message) {}

    /**
     * @brief 带错误码的构造函数
     * @param message 异常消息
     * @param code 错误码
     */
    AppBaseException(const std::string& message, int code)
        : std::runtime_error(message), code_(code) {}

    /**
     * @brief 获取错误码
     * @return int 错误码
     */
    int getCode() const { return code_; }

protected:
    int code_ = 0; // 默认错误码为0
};

/**
 * @brief 初始化异常，用于表示组件初始化失败
 */
class InitializationError : public AppBaseException {
public:
    explicit InitializationError(const std::string& message)
        : AppBaseException(message) {}

    InitializationError(const std::string& message, int code)
        : AppBaseException(message, code) {}
};

/**
 * @brief 配置异常，用于表示配置文件错误
 */
class ConfigurationError : public AppBaseException {
public:
    explicit ConfigurationError(const std::string& message)
        : AppBaseException(message) {}

    ConfigurationError(const std::string& message, int code)
        : AppBaseException(message, code) {}
};

/**
 * @brief 文件操作异常，用于表示文件读写错误
 */
class FileError : public AppBaseException {
public:
    explicit FileError(const std::string& message)
        : AppBaseException(message) {}

    FileError(const std::string& message, int code)
        : AppBaseException(message, code) {}
};

/**
 * @brief 资源不存在异常
 */
class ResourceNotFoundError : public AppBaseException {
public:
    explicit ResourceNotFoundError(const std::string& message)
        : AppBaseException(message) {}

    ResourceNotFoundError(const std::string& message, int code)
        : AppBaseException(message, code) {}
};

/**
 * @brief 参数验证异常
 */
class ValidationError : public AppBaseException {
public:
    explicit ValidationError(const std::string& message)
        : AppBaseException(message) {}

    ValidationError(const std::string& message, int code)
        : AppBaseException(message, code) {}
};

/**
 * @brief 操作超时异常
 */
class TimeoutError : public AppBaseException {
public:
    explicit TimeoutError(const std::string& message)
        : AppBaseException(message) {}

    TimeoutError(const std::string& message, int code)
        : AppBaseException(message, code) {}
};

/**
 * @brief 操作不支持异常，用于表示某个功能或操作未被特定对象或在当前状态下支持
 */
class OperationNotSupportedException : public AppBaseException {
public:
    explicit OperationNotSupportedException(const std::string& message)
        : AppBaseException(message) {}

    OperationNotSupportedException(const std::string& message, int code)
        : AppBaseException(message, code) {}
};

/**
 * @brief 生成带文件、行号、函数名的异常消息辅助宏
 */
#define MAKE_ERROR_MSG(msg) \
    (std::stringstream() << msg << " (at " << __FILE__ << ":" << __LINE__ << ", in " << __FUNCTION__ << ")").str()

/**
 * @brief 抛出带位置信息的异常辅助宏
 */
#define THROW_APP_EXCEPTION(ExceptionType, msg) \
    throw ExceptionType(MAKE_ERROR_MSG(msg))

#define THROW_APP_EXCEPTION_CODE(ExceptionType, msg, code) \
    throw ExceptionType(MAKE_ERROR_MSG(msg), code)

} // namespace common_utils
} // namespace oscean