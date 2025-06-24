/**
 * @file exceptions.h
 * @brief Core Services 业务异常定义
 * 
 * 🎯 重构说明：
 * ✅ 所有异常都从 common_utils::ServiceException 继承
 * ✅ 提供业务特定的异常类型
 * ✅ 保持向后兼容的接口
 */

#pragma once

// 引入基础异常定义
#include "common_utils/utilities/exceptions.h"

namespace oscean {
namespace core_services {

// 使用基础服务异常作为所有 core_services 异常的基类
using common_utils::ServiceException;

/**
 * @brief 数据访问异常 - 数据访问层错误
 */
class DataAccessException : public ServiceException {
public:
    explicit DataAccessException(const std::string& message)
        : ServiceException(message) {}
        
    explicit DataAccessException(const char* message)
        : ServiceException(message) {}
        
    DataAccessException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 文件未找到异常 - 数据访问中的文件不存在错误
 */
class FileNotFoundException : public DataAccessException {
public:
    explicit FileNotFoundException(const std::string& message)
        : DataAccessException(message) {}
        
    explicit FileNotFoundException(const char* message)
        : DataAccessException(message) {}
        
    static FileNotFoundException forFile(const std::string& filePath) {
        return FileNotFoundException("File not found: " + filePath);
    }
};

/**
 * @brief CRS 坐标参考系统异常
 */
class CrsException : public ServiceException {
public:
    explicit CrsException(const std::string& message)
        : ServiceException(message) {}
        
    explicit CrsException(const char* message)
        : ServiceException(message) {}
        
    CrsException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 空间操作异常
 */
class SpatialOpsException : public ServiceException {
public:
    explicit SpatialOpsException(const std::string& message)
        : ServiceException(message) {}
        
    explicit SpatialOpsException(const char* message)
        : ServiceException(message) {}
        
    SpatialOpsException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 插值异常
 */
class InterpolationException : public ServiceException {
public:
    explicit InterpolationException(const std::string& message)
        : ServiceException(message) {}
        
    explicit InterpolationException(const char* message)
        : ServiceException(message) {}
        
    InterpolationException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 建模服务异常
 */
class ModelingException : public ServiceException {
public:
    explicit ModelingException(const std::string& message)
        : ServiceException(message) {}
        
    explicit ModelingException(const char* message)
        : ServiceException(message) {}
        
    ModelingException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 服务创建/初始化异常
 */
class ServiceCreationException : public ServiceException {
public:
    explicit ServiceCreationException(const std::string& message)
        : ServiceException(message) {}
        
    explicit ServiceCreationException(const char* message)
        : ServiceException(message) {}
        
    ServiceCreationException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 无效输入异常
 */
class InvalidInputException : public ServiceException {
public:
    explicit InvalidInputException(const std::string& message)
        : ServiceException(message) {}

    explicit InvalidInputException(const char* message)
        : ServiceException(message) {}
        
    InvalidInputException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 操作失败异常
 */
class OperationFailedException : public ServiceException {
public:
    explicit OperationFailedException(const std::string& message)
        : ServiceException(message) {}

    explicit OperationFailedException(const char* message)
        : ServiceException(message) {}
        
    OperationFailedException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 无效参数异常
 */
class InvalidParameterException : public ServiceException {
public:
    explicit InvalidParameterException(const std::string& message)
        : ServiceException(message) {}

    explicit InvalidParameterException(const char* message)
        : ServiceException(message) {}
        
    InvalidParameterException(const std::string& parameterName, const std::string& reason)
        : ServiceException("Invalid parameter '" + parameterName + "': " + reason) {}
        
    InvalidParameterException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 无效几何异常
 */
class InvalidGeometryException : public ServiceException {
public:
    explicit InvalidGeometryException(const std::string& message)
        : ServiceException(message) {}

    explicit InvalidGeometryException(const char* message)
        : ServiceException(message) {}
        
    InvalidGeometryException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 栅格处理异常
 */
class RasterProcessingException : public ServiceException {
public:
    explicit RasterProcessingException(const std::string& message)
        : ServiceException(message) {}

    explicit RasterProcessingException(const char* message)
        : ServiceException(message) {}
        
    RasterProcessingException(const std::string& message, const std::string& operation)
        : ServiceException("Raster processing error in '" + operation + "': " + message) {}
        
    RasterProcessingException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

/**
 * @brief 无效输入几何异常
 */
class InvalidInputGeometryException : public ServiceException {
public:
    explicit InvalidInputGeometryException(const std::string& message)
        : ServiceException(message) {}

    explicit InvalidInputGeometryException(const char* message)
        : ServiceException(message) {}
        
    InvalidInputGeometryException(const std::string& message, int code)
        : ServiceException(message, code) {}
};

} // namespace core_services
} // namespace oscean

/**
 * 🎯 使用指南：
 * 
 * 1. 基础异常捕获（推荐）：
 *    ```cpp
 *    try {
 *        // ... core service operations
 *    } catch (const common_utils::ServiceException& e) {
 *        // 捕获所有 core service 异常
 *    } catch (const common_utils::OsceanBaseException& e) {
 *        // 捕获所有项目异常
 *    }
 *    ```
 * 
 * 2. 具体异常捕获：
 *    ```cpp
 *    try {
 *        // ... data access operations
 *    } catch (const FileNotFoundException& e) {
 *        // 处理文件不存在
 *    } catch (const DataAccessException& e) {
 *        // 处理其他数据访问错误
 *    }
 *    ```
 */ 