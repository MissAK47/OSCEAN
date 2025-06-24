#pragma once

#include <string>

namespace oscean::core_services::data_access {

/**
 * @brief 数据访问模块的错误码枚举
 */
enum class DataAccessErrorCode {
    // 通用错误
    SUCCESS = 0,                    ///< 成功
    UNKNOWN_ERROR = 1,              ///< 未知错误
    INVALID_PARAMETER = 2,          ///< 参数无效
    
    // 文件相关错误
    FILE_NOT_FOUND = 100,           ///< 文件未找到
    FILE_ACCESS_DENIED = 101,       ///< 文件访问被拒绝
    FILE_FORMAT_UNSUPPORTED = 102,  ///< 文件格式不支持
    FILE_CORRUPTED = 103,           ///< 文件损坏
    
    // 读取器相关错误
    READER_NOT_AVAILABLE = 200,     ///< 读取器不可用
    READER_INITIALIZATION_FAILED = 201, ///< 读取器初始化失败
    READER_OPERATION_FAILED = 202,  ///< 读取器操作失败
    
    // 数据相关错误
    VARIABLE_NOT_FOUND = 300,       ///< 变量未找到
    DIMENSION_MISMATCH = 301,       ///< 维度不匹配
    DATA_TYPE_MISMATCH = 302,       ///< 数据类型不匹配
    INDEX_OUT_OF_BOUNDS = 303,      ///< 索引超出范围
    
    // 缓存相关错误
    CACHE_OPERATION_FAILED = 400,   ///< 缓存操作失败
    CACHE_MEMORY_EXHAUSTED = 401,   ///< 缓存内存耗尽
    
    // 流式处理错误
    STREAMING_NOT_SUPPORTED = 500,  ///< 不支持流式处理
    STREAM_INTERRUPTED = 501,       ///< 流被中断
    CHUNK_PROCESSING_FAILED = 502,  ///< 数据块处理失败
    
    // 异步处理错误
    ASYNC_OPERATION_TIMEOUT = 600,  ///< 异步操作超时
    ASYNC_OPERATION_CANCELLED = 601, ///< 异步操作被取消
    
    // CRS相关错误
    CRS_TRANSFORMATION_FAILED = 700, ///< 坐标转换失败
    CRS_NOT_SUPPORTED = 701,        ///< CRS不支持
    
    // 时间处理错误
    TIME_PARSING_FAILED = 800,      ///< 时间解析失败
    TIME_CONVERSION_FAILED = 801,   ///< 时间转换失败
    INVALID_TIME_UNIT = 802         ///< 无效的时间单位
};

/**
 * @brief 将错误码转换为字符串描述
 * @param errorCode 错误码
 * @return 错误描述字符串
 */
inline std::string errorCodeToString(DataAccessErrorCode errorCode) {
    switch (errorCode) {
        case DataAccessErrorCode::SUCCESS:
            return "Success";
        case DataAccessErrorCode::UNKNOWN_ERROR:
            return "Unknown error";
        case DataAccessErrorCode::INVALID_PARAMETER:
            return "Invalid parameter";
        case DataAccessErrorCode::FILE_NOT_FOUND:
            return "File not found";
        case DataAccessErrorCode::FILE_ACCESS_DENIED:
            return "File access denied";
        case DataAccessErrorCode::FILE_FORMAT_UNSUPPORTED:
            return "File format not supported";
        case DataAccessErrorCode::FILE_CORRUPTED:
            return "File corrupted";
        case DataAccessErrorCode::READER_NOT_AVAILABLE:
            return "Reader not available";
        case DataAccessErrorCode::READER_INITIALIZATION_FAILED:
            return "Reader initialization failed";
        case DataAccessErrorCode::READER_OPERATION_FAILED:
            return "Reader operation failed";
        case DataAccessErrorCode::VARIABLE_NOT_FOUND:
            return "Variable not found";
        case DataAccessErrorCode::DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case DataAccessErrorCode::DATA_TYPE_MISMATCH:
            return "Data type mismatch";
        case DataAccessErrorCode::INDEX_OUT_OF_BOUNDS:
            return "Index out of bounds";
        case DataAccessErrorCode::CACHE_OPERATION_FAILED:
            return "Cache operation failed";
        case DataAccessErrorCode::CACHE_MEMORY_EXHAUSTED:
            return "Cache memory exhausted";
        case DataAccessErrorCode::STREAMING_NOT_SUPPORTED:
            return "Streaming not supported";
        case DataAccessErrorCode::STREAM_INTERRUPTED:
            return "Stream interrupted";
        case DataAccessErrorCode::CHUNK_PROCESSING_FAILED:
            return "Chunk processing failed";
        case DataAccessErrorCode::ASYNC_OPERATION_TIMEOUT:
            return "Async operation timeout";
        case DataAccessErrorCode::ASYNC_OPERATION_CANCELLED:
            return "Async operation cancelled";
        case DataAccessErrorCode::CRS_TRANSFORMATION_FAILED:
            return "CRS transformation failed";
        case DataAccessErrorCode::CRS_NOT_SUPPORTED:
            return "CRS not supported";
        case DataAccessErrorCode::TIME_PARSING_FAILED:
            return "Time parsing failed";
        case DataAccessErrorCode::TIME_CONVERSION_FAILED:
            return "Time conversion failed";
        case DataAccessErrorCode::INVALID_TIME_UNIT:
            return "Invalid time unit";
        default:
            return "Unknown error code";
    }
}

} // namespace oscean::core_services::data_access 
