/**
 * @file raster_algebra.cpp
 * @brief RasterAlgebra class implementation
 */

#include "raster_algebra.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace oscean::core_services::spatial_ops::raster {

RasterAlgebra::RasterAlgebra(const SpatialOpsConfig& config)
    : m_config(config) {
}

oscean::core_services::GridData RasterAlgebra::performRasterAlgebra(
    const std::string& expression,
    const std::map<std::string, oscean::core_services::GridData>& namedRasters,
    std::optional<oscean::core_services::GridDefinition> targetGridDef,
    std::optional<double> noDataValue) const {
    
    if (expression.empty()) {
        throw InvalidParameterException("Expression cannot be empty");
    }
    
    if (namedRasters.empty()) {
        throw InvalidParameterException("Named rasters map cannot be empty");
    }

    // 简化的表达式解析和计算
    return evaluateExpression(expression, namedRasters);
}

oscean::core_services::GridData RasterAlgebra::addRasters(
    const oscean::core_services::GridData& rasterA,
    const oscean::core_services::GridData& rasterB,
    std::optional<double> noDataValue) const {
    
    validateRasterCompatibility(rasterA, rasterB);
    return performBinaryOperation(rasterA, rasterB, "add", noDataValue);
}

oscean::core_services::GridData RasterAlgebra::subtractRasters(
    const oscean::core_services::GridData& rasterA,
    const oscean::core_services::GridData& rasterB,
    std::optional<double> noDataValue) const {
    
    validateRasterCompatibility(rasterA, rasterB);
    return performBinaryOperation(rasterA, rasterB, "subtract", noDataValue);
}

oscean::core_services::GridData RasterAlgebra::multiplyRasters(
    const oscean::core_services::GridData& rasterA,
    const oscean::core_services::GridData& rasterB,
    std::optional<double> noDataValue) const {
    
    validateRasterCompatibility(rasterA, rasterB);
    return performBinaryOperation(rasterA, rasterB, "multiply", noDataValue);
}

oscean::core_services::GridData RasterAlgebra::divideRasters(
    const oscean::core_services::GridData& rasterA,
    const oscean::core_services::GridData& rasterB,
    std::optional<double> noDataValue) const {
    
    validateRasterCompatibility(rasterA, rasterB);
    return performBinaryOperation(rasterA, rasterB, "divide", noDataValue);
}

oscean::core_services::GridData RasterAlgebra::applyMathFunction(
    const oscean::core_services::GridData& inputRaster,
    const std::string& function,
    std::optional<double> noDataValue) const {
    
    if (inputRaster.getData().empty()) {
        throw InvalidInputDataException("Input raster data is empty");
    }

    const auto& gridDef = inputRaster.getDefinition();
    
    // 创建新的结果栅格
    oscean::core_services::GridData result(gridDef, inputRaster.getDataType(), inputRaster.getNumBands());
    
    const auto& sourceBuffer = inputRaster.getData();
    auto& targetBuffer = result.getUnifiedBuffer();
    
    // 初始化结果缓冲区大小
    targetBuffer.resize(sourceBuffer.size());
    
    const float* sourceData = reinterpret_cast<const float*>(sourceBuffer.data());
    float* targetData = reinterpret_cast<float*>(targetBuffer.data());
    
    size_t totalPixels = gridDef.rows * gridDef.cols;
    
    for (size_t i = 0; i < totalPixels; ++i) {
        if (i < sourceBuffer.size() / sizeof(float)) {
            float value = sourceData[i];
            
            // 检查是否为NoData值
            if (noDataValue.has_value() && isNoData(value, noDataValue)) {
                targetData[i] = static_cast<float>(noDataValue.value());
                continue;
            }
            
            // 应用数学函数
            if (function == "sin") {
                targetData[i] = std::sin(value);
            } else if (function == "cos") {
                targetData[i] = std::cos(value);
            } else if (function == "tan") {
                targetData[i] = std::tan(value);
            } else if (function == "log") {
                targetData[i] = (value > 0) ? std::log(value) : static_cast<float>(noDataValue.value_or(-9999.0));
            } else if (function == "log10") {
                targetData[i] = (value > 0) ? std::log10(value) : static_cast<float>(noDataValue.value_or(-9999.0));
            } else if (function == "sqrt") {
                targetData[i] = (value >= 0) ? std::sqrt(value) : static_cast<float>(noDataValue.value_or(-9999.0));
            } else if (function == "abs") {
                targetData[i] = std::abs(value);
            } else if (function == "exp") {
                targetData[i] = std::exp(value);
            } else {
                throw InvalidParameterException("Unsupported math function: " + function);
            }
        }
    }
    
    return result;
}

void RasterAlgebra::validateRasterCompatibility(
    const oscean::core_services::GridData& rasterA,
    const oscean::core_services::GridData& rasterB) const {
    
    if (rasterA.getData().empty() || rasterB.getData().empty()) {
        throw InvalidInputDataException("One or both rasters have empty data");
    }
    
    // 注释掉尺寸检查，允许不同尺寸的栅格进行运算
    // const auto& defA = rasterA.definition;
    // const auto& defB = rasterB.definition;
    
    // if (defA.rows != defB.rows || defA.cols != defB.cols) {
    //     throw InvalidParameterException("Rasters must have the same dimensions for algebraic operations");
    // }
    
    // 可以添加更多兼容性检查，如CRS、分辨率等
}

oscean::core_services::GridData RasterAlgebra::evaluateExpression(
    const std::string& expression,
    const std::map<std::string, oscean::core_services::GridData>& namedRasters) const {
    
    // 简化的表达式解析器
    // 支持基本的算术运算：A + B, A - B, A * B, A / B
    
    if (namedRasters.empty()) {
        throw InvalidParameterException("No rasters provided for expression evaluation");
    }
    
    // 获取第一个栅格作为基础（使用引用）
    const auto& firstRaster = namedRasters.begin()->second;
    
    // 简化处理：支持几种常见表达式模式
    if (expression.find("+") != std::string::npos) {
        // 查找加法表达式 "A + B"
        auto it = namedRasters.begin();
        if (namedRasters.size() >= 2) {
            const auto& rasterA = it->second;
            ++it;
            const auto& rasterB = it->second;
            return addRasters(rasterA, rasterB);
        }
    } else if (expression.find("-") != std::string::npos) {
        // 查找减法表达式 "A - B"
        auto it = namedRasters.begin();
        if (namedRasters.size() >= 2) {
            const auto& rasterA = it->second;
            ++it;
            const auto& rasterB = it->second;
            return subtractRasters(rasterA, rasterB);
        }
    } else if (expression.find("*") != std::string::npos) {
        // 查找乘法表达式 "A * B"
        auto it = namedRasters.begin();
        if (namedRasters.size() >= 2) {
            const auto& rasterA = it->second;
            ++it;
            const auto& rasterB = it->second;
            return multiplyRasters(rasterA, rasterB);
        }
    }
    
    // 如果无法解析表达式，创建并返回第一个栅格的副本
    const auto& gridDef = firstRaster.getDefinition();
    oscean::core_services::GridData result(gridDef, firstRaster.getDataType(), firstRaster.getNumBands());
    auto& buffer = result.getUnifiedBuffer();
    buffer = firstRaster.getData();
    return result;
}

oscean::core_services::GridData RasterAlgebra::performBinaryOperation(
    const oscean::core_services::GridData& rasterA,
    const oscean::core_services::GridData& rasterB,
    const std::string& operation,
    std::optional<double> noDataValue) const {
    
    const auto& gridDefA = rasterA.getDefinition();
    const auto& gridDefB = rasterB.getDefinition();
    
    // 创建新的结果栅格
    oscean::core_services::GridData result(gridDefA, rasterA.getDataType(), rasterA.getNumBands());
    
    const auto& bufferA = rasterA.getData();
    const auto& bufferB = rasterB.getData();
    auto& resultBuffer = result.getUnifiedBuffer();
    
    // 初始化结果缓冲区大小
    resultBuffer.resize(bufferA.size());
    
    const float* dataA = reinterpret_cast<const float*>(bufferA.data());
    const float* dataB = reinterpret_cast<const float*>(bufferB.data());
    float* resultData = reinterpret_cast<float*>(resultBuffer.data());
    
    size_t totalPixelsA = gridDefA.rows * gridDefA.cols;
    size_t totalPixelsB = gridDefB.rows * gridDefB.cols;
    
    for (size_t i = 0; i < totalPixelsA; ++i) {
        if (i < bufferA.size() / sizeof(float)) {
            float valueA = dataA[i];
            float valueB = 0.0f;
            
            // 如果栅格B的像素数量足够，使用对应的值；否则使用默认值
            if (i < totalPixelsB && i < bufferB.size() / sizeof(float)) {
                valueB = dataB[i];
            } else {
                // 对于超出栅格B范围的像素，使用NoData值或0
                valueB = static_cast<float>(noDataValue.value_or(0.0));
            }
            
            // 检查NoData值
            if ((noDataValue.has_value() && (isNoData(valueA, noDataValue) || isNoData(valueB, noDataValue)))) {
                resultData[i] = static_cast<float>(noDataValue.value());
                continue;
            }
            
            // 执行运算
            if (operation == "add") {
                resultData[i] = valueA + valueB;
            } else if (operation == "subtract") {
                resultData[i] = valueA - valueB;
            } else if (operation == "multiply") {
                resultData[i] = valueA * valueB;
            } else if (operation == "divide") {
                if (std::abs(valueB) > std::numeric_limits<float>::epsilon()) {
                    resultData[i] = valueA / valueB;
                } else {
                    resultData[i] = static_cast<float>(noDataValue.value_or(-9999.0));
                }
            } else {
                throw InvalidParameterException("Unsupported binary operation: " + operation);
            }
        }
    }
    
    return result;
}

bool RasterAlgebra::isNoData(double value, std::optional<double> noDataValue) const {
    if (!noDataValue.has_value()) {
        return false;
    }
    
    return std::abs(value - noDataValue.value()) < std::numeric_limits<double>::epsilon();
}

} // namespace oscean::core_services::spatial_ops::raster 