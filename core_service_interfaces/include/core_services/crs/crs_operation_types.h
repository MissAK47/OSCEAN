/**
 * @file crs_operation_types.h
 * @brief CRS操作类型定义 - 依赖于common_data_types.h
 */

#pragma once

#include "core_services/common_data_types.h"
#include <string>
#include <optional>
#include <vector>
#include <map>
// #include "core_services/common_data_types.h" // Point/BoundingBox not directly used here, but good for context

namespace oscean {
namespace core_services {

// 枚举定义：轴顺序
enum class AxisOrder {
    XY,   // 标准X,Y顺序 (经度, 纬度)
    YX,   // 反转顺序 (纬度, 经度)
    Other // 其他顺序
};

// Detailed parameters of a CRS, often obtained by inspecting it
struct CRSDetailedParameters { // This struct should start around line 19
    std::string name;
    std::string type; // Geographic, Projected, Vertical, Compound, etc.
    std::string authority;
    std::string code;
    std::string wktext;        // WKT representation
    std::string proj4text;     // Proj string representation

    // 添加缺失的属性
    AxisOrder axisOrder = AxisOrder::XY;  // 坐标轴顺序
    std::string projectionType;           // 投影类型
    std::string units;                    // 单位
    std::string angularUnits;             // 角度单位
    std::map<std::string, std::string> parameters; // 投影参数

    // Optional detailed components
    std::optional<std::string> datumName;
    std::optional<std::string> ellipsoidName;
    std::optional<double> semiMajorAxis;
    std::optional<double> inverseFlattening;
    std::optional<std::string> primeMeridianName;
    std::optional<double> primeMeridianLongitude;
    std::optional<std::string> unitName; // Unit of the axes
    std::optional<double> unitConversionFactor; // To SI unit (meter or radian)
    std::optional<std::string> projectionMethod; // For projected CRS
};

} // namespace core_services
} // namespace oscean 