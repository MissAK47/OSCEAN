/**
 * @file dimension_converter.h
 * @brief 提供维度转换函数
 */

#pragma once

#include "core_services/common_data_types.h"
#include "data_reader_common.h"

namespace oscean::core_services {

/**
 * @brief 将reader内部的DimensionCoordinateInfo转换为通用的DimensionCoordinateInfo
 * 
 * 使用inline关键字避免多个翻译单元中的重复定义
 * 
 * @param dim 读取器内部使用的维度坐标信息
 * @return 通用的维度坐标信息
 */
inline ::oscean::core_services::DimensionCoordinateInfo convertDimensionCoordinateInfo(const ::oscean::core_services::data_access::readers::DimensionCoordinateInfo& dim) {
    // 直接实现转换逻辑，不再调用外部函数
    ::oscean::core_services::DimensionCoordinateInfo serviceDimInfo;
    
    // 复制基本属性
    serviceDimInfo.name = dim.name;
    serviceDimInfo.standardName = dim.standardName;
    serviceDimInfo.longName = dim.longName;
    serviceDimInfo.units = dim.units;
    
    // 转换维度类型 (在readers命名空间中实现一个类型转换函数)
    // 这里假设CoordinateDimension类型在两个命名空间中是兼容的
    serviceDimInfo.type = static_cast<::oscean::core_services::CoordinateDimension>(dim.type);
    
    // 复制其他属性
    serviceDimInfo.isRegular = dim.isRegular;
    serviceDimInfo.resolution = dim.resolution;
    serviceDimInfo.coordinates = dim.coordinates;
    serviceDimInfo.coordinateLabels = dim.coordinateLabels;
    serviceDimInfo.attributes = dim.attributes;
    
    return serviceDimInfo;
}

} // namespace oscean::core_services

// 声明原始的转换函数，使其可在头文件中调用
namespace oscean::core_services::data_access {
    // 从readers内部到服务层的转换
    DimensionCoordinateInfo convertFromDimensionCoordinateInfo(const readers::DimensionCoordinateInfo& readerDimInfo);
    
    // 从服务层到readers内部的转换
    readers::DimensionCoordinateInfo convertToDimensionCoordinateInfo(const DimensionCoordinateInfo& serviceDimInfo);
} // namespace oscean::core_services::data_access 