/**
 * @file raster_vectorization.cpp
 * @brief RasterVectorization class implementation
 */

#include "raster_vectorization.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

namespace oscean::core_services::spatial_ops::raster {

RasterVectorization::RasterVectorization(const SpatialOpsConfig& config)
    : m_config(config) {
}

oscean::core_services::GridData RasterVectorization::rasterizeFeatures(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::GridDefinition& targetGridDef,
    const RasterizeOptions& options) const {
    
    if (features.empty()) {
        throw InvalidInputDataException("Feature collection is empty");
    }
    
    validateGridDefinition(targetGridDef);
    
    // 创建输出栅格
    oscean::core_services::GridData result(targetGridDef, DataType::Float32, 1);
    
    size_t totalPixels = targetGridDef.rows * targetGridDef.cols;
    auto& resultBuffer = result.getUnifiedBuffer();
    resultBuffer.resize(totalPixels * sizeof(float));
    
    float* data = reinterpret_cast<float*>(resultBuffer.data());
    
    // 初始化为背景值
    float backgroundValue = static_cast<float>(options.backgroundValue.value_or(0.0));
    std::fill(data, data + totalPixels, backgroundValue);
    
    // 栅格化每个要素
    for (const auto& feature : features.getFeatures()) {
        rasterizeFeature(feature, targetGridDef, data, options);
    }
    
    return result;
}

oscean::core_services::FeatureCollection RasterVectorization::generateContours(
    const oscean::core_services::GridData& raster,
    const ContourOptions& options) const {
    
    if (raster.getData().empty()) {
        throw InvalidInputDataException("Raster data is empty");
    }
    
    oscean::core_services::FeatureCollection result;
    
    // 获取等值线级别
    std::vector<double> levels;
    if (std::holds_alternative<double>(options.intervalOrLevels)) {
        // 使用间隔生成等值线级别
        double interval = std::get<double>(options.intervalOrLevels);
        const auto& rasterBuffer = raster.getData();
        const float* data = reinterpret_cast<const float*>(rasterBuffer.data());
        size_t totalPixels = raster.getDefinition().rows * raster.getDefinition().cols;
        
        // 找到数据范围
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        
        for (size_t i = 0; i < totalPixels; ++i) {
            if (!std::isnan(data[i])) {
                minVal = std::min(minVal, data[i]);
                maxVal = std::max(maxVal, data[i]);
            }
        }
        
        // 生成等值线级别
        for (double level = std::ceil(minVal / interval) * interval; level <= maxVal; level += interval) {
            levels.push_back(level);
        }
    } else {
        levels = std::get<std::vector<double>>(options.intervalOrLevels);
    }
    
    // 为每个级别生成等值线
    for (double level : levels) {
        auto isolines = createIsolines(raster, {level}, options.noDataValueToIgnore);
        
        // 将等值线转换为要素
        for (size_t i = 0; i < isolines.size(); ++i) {
            oscean::core_services::Feature feature;
            feature.geometryWkt = isolines[i];
            feature.attributes[options.outputAttributeName] = level;
            feature.id = "contour_" + std::to_string(level) + "_" + std::to_string(i);
            result.addFeature(feature);
        }
    }
    
    return result;
}

oscean::core_services::FeatureCollection RasterVectorization::vectorizeRaster(
    const oscean::core_services::GridData& raster,
    boost::optional<double> noDataValue) const {
    
    if (raster.getData().empty()) {
        throw InvalidInputDataException("Raster data is empty");
    }
    
    oscean::core_services::FeatureCollection result;
    
    // 简化实现：将栅格转换为点要素
    const auto& gridDef = raster.getDefinition();
    const auto& rasterBuffer = raster.getData();
    const float* data = reinterpret_cast<const float*>(rasterBuffer.data());
    
    for (size_t row = 0; row < gridDef.rows; ++row) {
        for (size_t col = 0; col < gridDef.cols; ++col) {
            size_t index = row * gridDef.cols + col;
            
            if (index < rasterBuffer.size() / sizeof(float)) {
                float value = data[index];
                
                // 跳过NoData值
                if (noDataValue.has_value() && 
                    std::abs(value - noDataValue.value()) < std::numeric_limits<float>::epsilon()) {
                    continue;
                }
                
                // 计算地理坐标
                double x = gridDef.extent.minX + (col + 0.5) * gridDef.xResolution;
                double y = gridDef.extent.maxY - (row + 0.5) * gridDef.yResolution;
                
                // 创建点要素的WKT
                std::ostringstream wkt;
                wkt << "POINT(" << x << " " << y << ")";
                
                oscean::core_services::Feature feature;
                feature.geometryWkt = wkt.str();
                feature.attributes["value"] = value;
                feature.attributes["row"] = static_cast<int>(row);
                feature.attributes["col"] = static_cast<int>(col);
                feature.id = "pixel_" + std::to_string(row) + "_" + std::to_string(col);
                
                result.addFeature(feature);
            }
        }
    }
    
    return result;
}

std::vector<std::string> RasterVectorization::traceBoundaries(
    const oscean::core_services::GridData& raster,
    double threshold,
    boost::optional<double> noDataValue) const {
    
    if (raster.getData().empty()) {
        throw InvalidInputDataException("Raster data is empty");
    }
    
    std::vector<std::string> boundaries;
    
    // 简化实现：返回栅格边界的WKT
    const auto& extent = raster.getDefinition().extent;
    std::ostringstream wkt;
    wkt << "LINESTRING(" 
        << extent.minX << " " << extent.minY << ", "
        << extent.maxX << " " << extent.minY << ", "
        << extent.maxX << " " << extent.maxY << ", "
        << extent.minX << " " << extent.maxY << ", "
        << extent.minX << " " << extent.minY << ")";
    
    boundaries.push_back(wkt.str());
    
    return boundaries;
}

oscean::core_services::FeatureCollection RasterVectorization::rasterToPoints(
    const oscean::core_services::GridData& raster,
    boost::optional<double> noDataValue) const {
    
    if (raster.getData().empty()) {
        throw InvalidInputDataException("Raster data is empty");
    }
    
    oscean::core_services::FeatureCollection result;
    const auto& gridDef = raster.getDefinition();
    const auto& rasterBuffer = raster.getData();
    const float* data = reinterpret_cast<const float*>(rasterBuffer.data());
    
    // 将每个有效像素转换为点要素
    for (size_t row = 0; row < gridDef.rows; ++row) {
        for (size_t col = 0; col < gridDef.cols; ++col) {
            size_t index = row * gridDef.cols + col;
            
            if (index < rasterBuffer.size() / sizeof(float)) {
                float value = data[index];
                
                // 跳过NoData值
                if (noDataValue.has_value() && 
                    std::abs(value - noDataValue.value()) < std::numeric_limits<float>::epsilon()) {
                    continue;
                }
                
                // 计算地理坐标
                double x = gridDef.extent.minX + (col + 0.5) * gridDef.xResolution;
                double y = gridDef.extent.maxY - (row + 0.5) * gridDef.yResolution;
                
                // 创建点要素的WKT
                std::ostringstream wkt;
                wkt << "POINT(" << x << " " << y << ")";
                
                oscean::core_services::Feature feature;
                feature.geometryWkt = wkt.str();
                feature.attributes["value"] = value;
                feature.attributes["row"] = static_cast<int>(row);
                feature.attributes["col"] = static_cast<int>(col);
                feature.id = "point_" + std::to_string(row) + "_" + std::to_string(col);
                
                result.addFeature(feature);
            }
        }
    }
    
    return result;
}

std::vector<std::string> RasterVectorization::createIsolines(
    const oscean::core_services::GridData& raster,
    const std::vector<double>& levels,
    boost::optional<double> noDataValue) const {
    
    if (raster.getData().empty()) {
        throw InvalidInputDataException("Raster data is empty");
    }
    
    if (levels.empty()) {
        throw InvalidParameterException("Isoline levels cannot be empty");
    }
    
    std::vector<std::string> isolines;
    
    // 简化实现：为每个级别创建一个简单的线要素
    const auto& extent = raster.getDefinition().extent;
    
    for (double level : levels) {
        std::ostringstream wkt;
        wkt << "LINESTRING(" 
            << extent.minX << " " << (extent.minY + extent.maxY) / 2.0 << ", "
            << extent.maxX << " " << (extent.minY + extent.maxY) / 2.0 << ")";
        
        isolines.push_back(wkt.str());
    }
    
    return isolines;
}

void RasterVectorization::validateGridDefinition(
    const oscean::core_services::GridDefinition& gridDef) const {
    
    if (gridDef.rows == 0 || gridDef.cols == 0) {
        throw InvalidParameterException("Grid definition has zero dimensions");
    }
    
    if (gridDef.xResolution <= 0 || gridDef.yResolution <= 0) {
        throw InvalidParameterException("Grid definition has invalid resolution");
    }
    
    if (gridDef.extent.minX >= gridDef.extent.maxX || 
        gridDef.extent.minY >= gridDef.extent.maxY) {
        throw InvalidParameterException("Grid definition has invalid extent");
    }
}

void RasterVectorization::rasterizeFeature(
    const oscean::core_services::Feature& feature,
    const oscean::core_services::GridDefinition& gridDef,
    float* data,
    const RasterizeOptions& options) const {
    
    // 获取栅格化值
    float burnValue = static_cast<float>(options.burnValue.value_or(1.0));
    
    // 简化实现：解析WKT几何体并进行基本栅格化
    const std::string& wkt = feature.geometryWkt;
    
    if (wkt.find("POINT") == 0) {
        rasterizePointFromWKT(wkt, gridDef, data, burnValue);
    }
    else if (wkt.find("LINESTRING") == 0) {
        rasterizeLineStringFromWKT(wkt, gridDef, data, burnValue);
    }
    else if (wkt.find("POLYGON") == 0) {
        rasterizePolygonFromWKT(wkt, gridDef, data, burnValue);
    }
}

void RasterVectorization::rasterizePointFromWKT(
    const std::string& wkt,
    const oscean::core_services::GridDefinition& gridDef,
    float* data,
    float burnValue) const {
    
    // 简单的WKT解析：POINT(x y)
    size_t start = wkt.find('(');
    size_t end = wkt.find(')');
    if (start == std::string::npos || end == std::string::npos) {
        return;
    }
    
    std::string coords = wkt.substr(start + 1, end - start - 1);
    std::istringstream iss(coords);
    double x, y;
    if (!(iss >> x >> y)) {
        return;
    }
    
    // 转换为栅格坐标
    int col = static_cast<int>((x - gridDef.extent.minX) / gridDef.xResolution);
    int row = static_cast<int>((gridDef.extent.maxY - y) / gridDef.yResolution);
    
    // 检查边界
    if (col >= 0 && col < static_cast<int>(gridDef.cols) && 
        row >= 0 && row < static_cast<int>(gridDef.rows)) {
        size_t index = row * gridDef.cols + col;
        data[index] = burnValue;
    }
}

void RasterVectorization::rasterizeLineStringFromWKT(
    const std::string& wkt,
    const oscean::core_services::GridDefinition& gridDef,
    float* data,
    float burnValue) const {
    
    // 简化实现：只栅格化线的端点
    size_t start = wkt.find('(');
    size_t end = wkt.find(')');
    if (start == std::string::npos || end == std::string::npos) {
        return;
    }
    
    std::string coords = wkt.substr(start + 1, end - start - 1);
    std::istringstream iss(coords);
    std::string token;
    
    // 解析第一个点
    if (std::getline(iss, token, ',')) {
        std::istringstream pointStream(token);
        double x, y;
        if (pointStream >> x >> y) {
            int col = static_cast<int>((x - gridDef.extent.minX) / gridDef.xResolution);
            int row = static_cast<int>((gridDef.extent.maxY - y) / gridDef.yResolution);
            
            if (col >= 0 && col < static_cast<int>(gridDef.cols) && 
                row >= 0 && row < static_cast<int>(gridDef.rows)) {
                size_t index = row * gridDef.cols + col;
                data[index] = burnValue;
            }
        }
    }
}

void RasterVectorization::rasterizePolygonFromWKT(
    const std::string& wkt,
    const oscean::core_services::GridDefinition& gridDef,
    float* data,
    float burnValue) const {
    
    // 简化实现：栅格化多边形的边界框
    size_t start = wkt.find("((");
    size_t end = wkt.find("))");
    if (start == std::string::npos || end == std::string::npos) {
        return;
    }
    
    std::string coords = wkt.substr(start + 2, end - start - 2);
    std::istringstream iss(coords);
    std::string token;
    
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();
    
    // 找到边界框
    while (std::getline(iss, token, ',')) {
        std::istringstream pointStream(token);
        double x, y;
        if (pointStream >> x >> y) {
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    }
    
    // 栅格化边界框
    int startCol = std::max(0, static_cast<int>((minX - gridDef.extent.minX) / gridDef.xResolution));
    int endCol = std::min(static_cast<int>(gridDef.cols) - 1, 
                         static_cast<int>((maxX - gridDef.extent.minX) / gridDef.xResolution));
    int startRow = std::max(0, static_cast<int>((gridDef.extent.maxY - maxY) / gridDef.yResolution));
    int endRow = std::min(static_cast<int>(gridDef.rows) - 1,
                         static_cast<int>((gridDef.extent.maxY - minY) / gridDef.yResolution));
    
    for (int row = startRow; row <= endRow; ++row) {
        for (int col = startCol; col <= endCol; ++col) {
            size_t index = row * gridDef.cols + col;
            data[index] = burnValue;
        }
    }
}

} // namespace oscean::core_services::spatial_ops::raster 
