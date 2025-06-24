#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <chrono>

namespace oscean::core_services::spatial_ops {

/**
 * @class SpatialOpsException
 * @brief Base exception class for spatial operations service
 */
class SpatialOpsException : public std::runtime_error {
public:
    explicit SpatialOpsException(const std::string& message)
        : std::runtime_error(message), errorCode_("SPATIAL_ERROR") {}
    
    SpatialOpsException(const std::string& message, const std::string& errorCode)
        : std::runtime_error(message), errorCode_(errorCode) {}
    
    const std::string& getErrorCode() const noexcept { return errorCode_; }

private:
    std::string errorCode_;
};

/**
 * @class InvalidInputDataException
 * @brief Exception for invalid input data
 */
class InvalidInputDataException : public SpatialOpsException {
public:
    explicit InvalidInputDataException(const std::string& message)
        : SpatialOpsException(message, "INVALID_INPUT_DATA") {}
    
    InvalidInputDataException(const std::string& message, const std::string& details)
        : SpatialOpsException(message + (details.empty() ? "" : ": " + details), "INVALID_INPUT_DATA") {}
};

/**
 * @class InvalidGeometryException
 * @brief Exception for invalid geometry operations
 */
class InvalidGeometryException : public SpatialOpsException {
public:
    explicit InvalidGeometryException(const std::string& message)
        : SpatialOpsException(message, "INVALID_GEOMETRY") {}
};

/**
 * @class InvalidRasterException
 * @brief Exception for invalid raster operations
 */
class InvalidRasterException : public SpatialOpsException {
public:
    explicit InvalidRasterException(const std::string& message)
        : SpatialOpsException(message, "INVALID_RASTER") {}
};

/**
 * @class InvalidCRSException
 * @brief Exception for invalid coordinate reference system
 */
class InvalidCRSException : public SpatialOpsException {
public:
    explicit InvalidCRSException(const std::string& message)
        : SpatialOpsException(message, "INVALID_CRS") {}
};

/**
 * @class UnsupportedCRSError
 * @brief Exception for unsupported coordinate reference systems
 */
class UnsupportedCRSError : public SpatialOpsException {
public:
    explicit UnsupportedCRSError(const std::string& message)
        : SpatialOpsException(message, "UNSUPPORTED_CRS") {}
    
    UnsupportedCRSError(const std::string& crsIdentifier, const std::string& reason)
        : SpatialOpsException("Unsupported CRS '" + crsIdentifier + "': " + reason, "UNSUPPORTED_CRS") {}
    
    UnsupportedCRSError(const std::string& crsIdentifier, const std::string& reason, const std::string& details)
        : SpatialOpsException("Unsupported CRS '" + crsIdentifier + "': " + reason + (details.empty() ? "" : " (" + details + ")"), "UNSUPPORTED_CRS") {}
};

/**
 * @class OperationTimeoutException
 * @brief Exception for operation timeout
 */
class OperationTimeoutException : public SpatialOpsException {
public:
    OperationTimeoutException(const std::string& message, std::chrono::milliseconds timeout)
        : SpatialOpsException(message, "OPERATION_TIMEOUT"), timeout_(timeout) {}
    
    std::chrono::milliseconds getTimeout() const noexcept { return timeout_; }

private:
    std::chrono::milliseconds timeout_;
};

/**
 * @class InsufficientMemoryException
 * @brief Exception for insufficient memory
 */
class InsufficientMemoryException : public SpatialOpsException {
public:
    explicit InsufficientMemoryException(const std::string& message)
        : SpatialOpsException(message, "INSUFFICIENT_MEMORY") {}
};

/**
 * @class UnsupportedOperationException
 * @brief Exception for unsupported operations
 */
class UnsupportedOperationException : public SpatialOpsException {
public:
    explicit UnsupportedOperationException(const std::string& message)
        : SpatialOpsException(message, "UNSUPPORTED_OPERATION") {}
};

/**
 * @class ConfigurationException
 * @brief Exception for configuration errors
 */
class ConfigurationException : public SpatialOpsException {
public:
    explicit ConfigurationException(const std::string& message)
        : SpatialOpsException(message, "CONFIGURATION_ERROR") {}
};

/**
 * @class InvalidParameterException
 * @brief Exception for invalid parameters
 */
class InvalidParameterException : public SpatialOpsException {
public:
    explicit InvalidParameterException(const std::string& message)
        : SpatialOpsException(message, "INVALID_PARAMETER") {}
    
    InvalidParameterException(const std::string& parameterName, const std::string& reason)
        : SpatialOpsException("Invalid parameter '" + parameterName + "': " + reason, "INVALID_PARAMETER") {}
    
    InvalidParameterException(const std::string& parameterName, const std::string& reason, const std::string& details)
        : SpatialOpsException("Invalid parameter '" + parameterName + "': " + reason + (details.empty() ? "" : " (" + details + ")"), "INVALID_PARAMETER") {}
};

/**
 * @class OperationFailedException
 * @brief Exception for failed operations
 */
class OperationFailedException : public SpatialOpsException {
public:
    explicit OperationFailedException(const std::string& message)
        : SpatialOpsException(message, "OPERATION_FAILED") {}
    
    OperationFailedException(const std::string& operationName, const std::string& reason)
        : SpatialOpsException("Operation '" + operationName + "' failed: " + reason, "OPERATION_FAILED") {}
    
    OperationFailedException(const std::string& operationName, const std::string& reason, const std::string& details)
        : SpatialOpsException("Operation '" + operationName + "' failed: " + reason + (details.empty() ? "" : " (" + details + ")"), "OPERATION_FAILED") {}
};

/**
 * @class ResourceNotFoundException
 * @brief Exception for resource not found errors
 */
class ResourceNotFoundException : public SpatialOpsException {
public:
    explicit ResourceNotFoundException(const std::string& message)
        : SpatialOpsException(message, "RESOURCE_NOT_FOUND") {}
};

} // namespace oscean::core_services::spatial_ops 