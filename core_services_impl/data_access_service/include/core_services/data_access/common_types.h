#pragma once

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <unordered_map>
#include "core_services/common_data_types.h"

namespace oscean::core_services::data_access {

/**
 * @brief 使用core_service_interfaces中的通用DataChunkKey
 * 
 * 为了避免命名冲突和重复定义，我们使用core_service_interfaces中的
 * DataChunkKey作为基础，并提供类型别名和扩展功能
 */
using DataAccessChunkKey = oscean::core_services::DataChunkKey;

/**
 * @brief 数据访问服务专用的chunk key工厂函数
 * 
 * 提供便捷的方法创建适合数据访问服务的DataChunkKey
 */
namespace chunk_key_factory {

/**
 * @brief 从切片范围创建DataChunkKey
 * @param filePath 文件路径
 * @param variableName 变量名
 * @param sliceRanges 切片范围
 * @param targetCRS 目标坐标系（可选）
 * @return DataChunkKey
 */
inline oscean::core_services::DataChunkKey createFromSliceRanges(
    const std::string& filePath,
    const std::string& variableName, 
    const std::vector<oscean::core_services::IndexRange>& sliceRanges,
    const std::optional<std::string>& targetCRS = std::nullopt) {
    
    oscean::core_services::DataChunkKey key(filePath, variableName, 
                                           std::nullopt, std::nullopt, std::nullopt, 
                                           targetCRS.value_or(""));
    
    // 将sliceRanges信息编码到requestDataType中，以保持兼容性
    std::string requestType = "sliced_data";
    for (const auto& range : sliceRanges) {
        requestType += "[" + std::to_string(range.start) + ":" + std::to_string(range.count) + "]";
    }
    key.requestDataType = requestType;
    
    return key;
}

/**
 * @brief 从时空范围创建DataChunkKey  
 * @param filePath 文件路径
 * @param variableName 变量名
 * @param timeRange 时间范围（可选）
 * @param bbox 空间边界框（可选）
 * @param levelRange 级别范围（可选）
 * @param targetCRS 目标坐标系（可选）
 * @return DataChunkKey
 */
inline oscean::core_services::DataChunkKey createFromSpatioTemporal(
    const std::string& filePath,
    const std::string& variableName,
    const std::optional<oscean::core_services::IndexRange>& timeRange = std::nullopt,
    const std::optional<oscean::core_services::BoundingBox>& bbox = std::nullopt,
    const std::optional<oscean::core_services::IndexRange>& levelRange = std::nullopt,
    const std::optional<std::string>& targetCRS = std::nullopt) {
    
    return oscean::core_services::DataChunkKey(filePath, variableName, 
                                             timeRange, bbox, levelRange,
                                             targetCRS.value_or(""));
}

} // namespace chunk_key_factory

/**
 * @brief 文件元数据缓存键
 */
struct FileMetadataKey {
    std::string filePath;                       ///< 文件路径
    std::optional<std::string> targetCRS;      ///< 目标坐标系（可选）
    
    /**
     * @brief 相等比较运算符
     */
    bool operator==(const FileMetadataKey& other) const {
        return filePath == other.filePath && targetCRS == other.targetCRS;
    }
    
    /**
     * @brief 不等比较运算符
     */
    bool operator!=(const FileMetadataKey& other) const {
        return !(*this == other);
    }
    
    /**
     * @brief 生成字符串表示
     */
    std::string toString() const {
        std::string result = filePath;
        if (targetCRS) {
            result += ":" + *targetCRS;
        }
        return result;
    }
};

} // namespace oscean::core_services::data_access

/**
 * @brief 数据类型转换工具函数
 * 
 * 从旧的data_type_converters.h迁移的有用功能
 */
namespace oscean::core_services::data_access::utils {

/**
 * @brief 将字符串表示的数据类型转换为枚举类型
 * @param typeStr 数据类型字符串
 * @return 数据类型枚举
 */
inline oscean::core_services::DataType translateStringToDataType(const std::string& typeStr) {
    if (typeStr == "Float32") return oscean::core_services::DataType::Float32;
    if (typeStr == "Float64") return oscean::core_services::DataType::Float64;
    if (typeStr == "Int8") return oscean::core_services::DataType::Byte;
    if (typeStr == "Int16") return oscean::core_services::DataType::Int16;
    if (typeStr == "Int32") return oscean::core_services::DataType::Int32;
    if (typeStr == "Int64") return oscean::core_services::DataType::Int64;
    if (typeStr == "UInt8") return oscean::core_services::DataType::UByte;
    if (typeStr == "UInt16") return oscean::core_services::DataType::UInt16;
    if (typeStr == "UInt32") return oscean::core_services::DataType::UInt32;
    if (typeStr == "UInt64") return oscean::core_services::DataType::UInt64;
    if (typeStr == "String") return oscean::core_services::DataType::String;
    return oscean::core_services::DataType::Unknown;
}

/**
 * @brief 将数据类型枚举转换为字符串表示
 * @param dataType 数据类型枚举
 * @return 数据类型字符串
 */
inline std::string translateDataTypeToString(oscean::core_services::DataType dataType) {
    switch (dataType) {
        case oscean::core_services::DataType::Float32: return "Float32";
        case oscean::core_services::DataType::Float64: return "Float64";
        case oscean::core_services::DataType::Byte: return "Int8";
        case oscean::core_services::DataType::Int16: return "Int16";
        case oscean::core_services::DataType::Int32: return "Int32";
        case oscean::core_services::DataType::Int64: return "Int64";
        case oscean::core_services::DataType::UByte: return "UInt8";
        case oscean::core_services::DataType::UInt16: return "UInt16";
        case oscean::core_services::DataType::UInt32: return "UInt32";
        case oscean::core_services::DataType::UInt64: return "UInt64";
        case oscean::core_services::DataType::String: return "String";
        default: return "Unknown";
    }
}

} // namespace oscean::core_services::data_access::utils

/**
 * @brief std::hash特化，支持unordered_map/unordered_set中使用FileMetadataKey
 */
namespace std {
template<>
struct hash<oscean::core_services::data_access::FileMetadataKey> {
    size_t operator()(const oscean::core_services::data_access::FileMetadataKey& key) const {
        size_t h1 = std::hash<std::string>{}(key.filePath);
        size_t h2 = key.targetCRS ? std::hash<std::string>{}(*key.targetCRS) : 0;
        return h1 ^ (h2 << 1);
    }
};
} // namespace std
