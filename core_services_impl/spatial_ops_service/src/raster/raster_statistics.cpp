/**
 * @file raster_statistics.cpp
 * @brief RasterStatistics class implementation
 */

#include "raster_statistics.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <boost/optional.hpp>

namespace oscean::core_services::spatial_ops::raster {

RasterStatistics::RasterStatistics(const SpatialOpsConfig& config)
    : m_config(config) {
}

oscean::core_services::GridData RasterStatistics::computePixelwiseStatistics(
    const std::vector<oscean::core_services::GridData>& inputRasters,
    StatisticalMeasure statisticType) const {
    
    if (inputRasters.empty()) {
        throw InvalidInputDataException("Input rasters vector is empty");
    }
    
    const auto& firstRaster = inputRasters[0];
    const auto& gridDef = firstRaster.getDefinition();
    
    // 创建结果栅格
    oscean::core_services::GridData result(gridDef, DataType::Float32, 1);
    auto& resultBuffer = result.getUnifiedBuffer();
    float* resultData = reinterpret_cast<float*>(resultBuffer.data());
    
    size_t totalPixels = gridDef.rows * gridDef.cols;
    
    for (size_t pixelIndex = 0; pixelIndex < totalPixels; ++pixelIndex) {
        std::vector<double> pixelValues;
        pixelValues.reserve(inputRasters.size());
        
        // 收集所有栅格在该像素位置的值
        for (const auto& raster : inputRasters) {
            const auto& rasterBuffer = raster.getData();
            if (pixelIndex < rasterBuffer.size() / sizeof(float)) {
                const float* data = reinterpret_cast<const float*>(rasterBuffer.data());
                pixelValues.push_back(static_cast<double>(data[pixelIndex]));
            }
        }
        
        // 计算统计值
        double statisticValue = 0.0;
        if (!pixelValues.empty()) {
            switch (statisticType) {
                case StatisticalMeasure::MIN:
                    statisticValue = *std::min_element(pixelValues.begin(), pixelValues.end());
                    break;
                case StatisticalMeasure::MAX:
                    statisticValue = *std::max_element(pixelValues.begin(), pixelValues.end());
                    break;
                case StatisticalMeasure::MEAN:
                    statisticValue = std::accumulate(pixelValues.begin(), pixelValues.end(), 0.0) / pixelValues.size();
                    break;
                case StatisticalMeasure::SUM:
                    statisticValue = std::accumulate(pixelValues.begin(), pixelValues.end(), 0.0);
                    break;
                default:
                    statisticValue = pixelValues[0];
                    break;
            }
        }
        
        resultData[pixelIndex] = static_cast<float>(statisticValue);
    }
    
    return result;
}

std::vector<StatisticsResult> RasterStatistics::calculateZonalStatistics(
    const oscean::core_services::GridData& valueRaster,
    const oscean::core_services::Geometry& zoneGeometry,
    const ZonalStatisticsOptions& options) const {
    
    if (valueRaster.getData().empty()) {
        throw InvalidInputDataException("Value raster data is empty");
    }
    
    std::vector<StatisticsResult> results;
    
    // 简化实现：计算整个栅格的统计值
    auto validPixels = extractValidPixels(valueRaster, options.noDataValueToIgnore);
    if (!validPixels.empty()) {
        StatisticsResult result = computeStatisticsFromValues(validPixels, options.statistics);
        result.zoneIdentifier = "zone_1";
        results.push_back(result);
    }
    
    return results;
}

std::map<FeatureId, StatisticsResult> RasterStatistics::calculateZonalStatistics(
    const oscean::core_services::GridData& valueRaster,
    const oscean::core_services::FeatureCollection& zoneFeatures,
    const ZonalStatisticsOptions& options) const {
    
    if (valueRaster.getData().empty()) {
        throw InvalidInputDataException("Value raster data is empty");
    }
    
    std::map<FeatureId, StatisticsResult> results;
    
    // 简化实现：为每个要素计算相同的统计值
    auto validPixels = extractValidPixels(valueRaster, options.noDataValueToIgnore);
    if (!validPixels.empty()) {
        StatisticsResult result = computeStatisticsFromValues(validPixels, options.statistics);
        
        // 为每个要素创建一个统计结果
        const auto& features = zoneFeatures.getFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            FeatureId featureId = static_cast<int>(i);
            result.zoneIdentifier = "feature_" + std::to_string(i);
            results[featureId] = result;
        }
    }
    
    return results;
}

std::map<int, StatisticsResult> RasterStatistics::calculateZonalStatistics(
    const oscean::core_services::GridData& valueRaster,
    const oscean::core_services::GridData& zoneRaster,
    const ZonalStatisticsOptions& options) const {
    
    if (valueRaster.getData().empty() || zoneRaster.getData().empty()) {
        throw InvalidInputDataException("Value or zone raster data is empty");
    }
    
    std::map<int, StatisticsResult> results;
    std::map<int, std::vector<double>> zoneValues;
    
    const auto& gridDef = valueRaster.getDefinition();
    const auto& valueBuffer = valueRaster.getData();
    const auto& zoneBuffer = zoneRaster.getData();
    const float* valueData = reinterpret_cast<const float*>(valueBuffer.data());
    const float* zoneData = reinterpret_cast<const float*>(zoneBuffer.data());
    
    size_t totalPixels = gridDef.rows * gridDef.cols;
    
    // 按区域ID收集值
    for (size_t i = 0; i < totalPixels; ++i) {
        if (i < valueBuffer.size() / sizeof(float) && i < zoneBuffer.size() / sizeof(float)) {
            int zoneId = static_cast<int>(zoneData[i]);
            double value = static_cast<double>(valueData[i]);
            
            // 跳过NoData值
            if (options.noDataValueToIgnore.has_value() && 
                std::abs(value - options.noDataValueToIgnore.value()) < std::numeric_limits<double>::epsilon()) {
                continue;
            }
            
            zoneValues[zoneId].push_back(value);
        }
    }
    
    // 为每个区域计算统计值
    for (const auto& [zoneId, values] : zoneValues) {
        if (!values.empty()) {
            StatisticsResult result = computeStatisticsFromValues(values, options.statistics);
            result.zoneIdentifier = "zone_" + std::to_string(zoneId);
            results[zoneId] = result;
        }
    }
    
    return results;
}

StatisticsResult RasterStatistics::computeBasicStatistics(
    const oscean::core_services::GridData& raster,
    bool excludeNoData) const {
    
    boost::optional<double> noDataValue;
    if (excludeNoData) {
        auto fillValue = raster.getFillValue();
        if (fillValue) {
            noDataValue = fillValue.value();
        }
    }
    auto validPixels = extractValidPixels(raster, noDataValue);
    
    std::vector<StatisticalMeasure> allMeasures = {
        StatisticalMeasure::MIN,
        StatisticalMeasure::MAX,
        StatisticalMeasure::MEAN,
        StatisticalMeasure::SUM,
        StatisticalMeasure::STDDEV
    };
    
    return computeStatisticsFromValues(validPixels, allMeasures);
}

std::vector<std::pair<double, int>> RasterStatistics::computeHistogram(
    const oscean::core_services::GridData& raster,
    int numBins,
    boost::optional<double> minValue,
    boost::optional<double> maxValue) const {
    
    if (raster.getData().empty()) {
        throw InvalidInputDataException("Raster data is empty");
    }
    
    if (numBins <= 0) {
        throw InvalidParameterException("Number of bins must be positive");
    }
    
    boost::optional<double> fillValue;
    auto rasterFillValue = raster.getFillValue();
    if (rasterFillValue) {
        fillValue = rasterFillValue.value();
    }
    auto validPixels = extractValidPixels(raster, fillValue);
    
    if (validPixels.empty()) {
        return std::vector<std::pair<double, int>>();
    }
    
    // 确定值范围
    double minVal = minValue.get_value_or(*std::min_element(validPixels.begin(), validPixels.end()));
    double maxVal = maxValue.get_value_or(*std::max_element(validPixels.begin(), validPixels.end()));
    
    if (minVal >= maxVal) {
        throw InvalidParameterException("Invalid value range for histogram");
    }
    
    // 创建直方图
    std::vector<std::pair<double, int>> histogram(numBins);
    double binWidth = (maxVal - minVal) / numBins;
    
    // 初始化bin中心值
    for (int i = 0; i < numBins; ++i) {
        histogram[i].first = minVal + (i + 0.5) * binWidth;
        histogram[i].second = 0;
    }
    
    // 统计每个bin的计数
    for (double value : validPixels) {
        int binIndex = static_cast<int>((value - minVal) / binWidth);
        binIndex = std::max(0, std::min(numBins - 1, binIndex));
        histogram[binIndex].second++;
    }
    
    return histogram;
}

std::vector<double> RasterStatistics::computePercentiles(
    const oscean::core_services::GridData& raster,
    const std::vector<double>& percentiles) const {
    
    if (raster.getData().empty()) {
        throw InvalidInputDataException("Raster data is empty");
    }
    
    boost::optional<double> fillValue;
    auto rasterFillValue = raster.getFillValue();
    if (rasterFillValue) {
        fillValue = rasterFillValue.value();
    }
    auto validPixels = extractValidPixels(raster, fillValue);
    
    if (validPixels.empty()) {
        return std::vector<double>(percentiles.size(), std::numeric_limits<double>::quiet_NaN());
    }
    
    // 排序数据
    std::sort(validPixels.begin(), validPixels.end());
    
    std::vector<double> results;
    results.reserve(percentiles.size());
    
    for (double percentile : percentiles) {
        if (percentile < 0.0 || percentile > 100.0) {
            throw InvalidParameterException("Percentile must be between 0 and 100");
        }
        
        double index = (percentile / 100.0) * (validPixels.size() - 1);
        size_t lowerIndex = static_cast<size_t>(std::floor(index));
        size_t upperIndex = static_cast<size_t>(std::ceil(index));
        
        if (lowerIndex == upperIndex) {
            results.push_back(validPixels[lowerIndex]);
        } else {
            double weight = index - lowerIndex;
            double value = validPixels[lowerIndex] * (1.0 - weight) + validPixels[upperIndex] * weight;
            results.push_back(value);
        }
    }
    
    return results;
}

std::vector<double> RasterStatistics::extractValidPixels(
    const oscean::core_services::GridData& raster,
    boost::optional<double> noDataValue) const {
    
    std::vector<double> validPixels;
    
    const auto& gridDef = raster.getDefinition();
    const auto& rasterBuffer = raster.getData();
    const float* data = reinterpret_cast<const float*>(rasterBuffer.data());
    size_t totalPixels = gridDef.rows * gridDef.cols;
    
    for (size_t i = 0; i < totalPixels; ++i) {
        if (i < rasterBuffer.size() / sizeof(float)) {
            double value = static_cast<double>(data[i]);
            
            // 跳过NoData值
            if (noDataValue && 
                std::abs(value - noDataValue.get()) < std::numeric_limits<double>::epsilon()) {
                continue;
            }
            
            // 跳过NaN值
            if (std::isnan(value)) {
                continue;
            }
            
            validPixels.push_back(value);
        }
    }
    
    return validPixels;
}

StatisticsResult RasterStatistics::computeStatisticsFromValues(
    const std::vector<double>& values,
    const std::vector<StatisticalMeasure>& measures) const {
    
    StatisticsResult result;
    
    if (values.empty()) {
        // 返回空结果
        for (auto measure : measures) {
            result.values[measure] = std::numeric_limits<double>::quiet_NaN();
        }
        return result;
    }
    
    for (auto measure : measures) {
        switch (measure) {
            case StatisticalMeasure::MIN:
                result.values[measure] = *std::min_element(values.begin(), values.end());
                break;
                
            case StatisticalMeasure::MAX:
                result.values[measure] = *std::max_element(values.begin(), values.end());
                break;
                
            case StatisticalMeasure::MEAN: {
                double sum = std::accumulate(values.begin(), values.end(), 0.0);
                result.values[measure] = sum / values.size();
                break;
            }
            
            case StatisticalMeasure::SUM:
                result.values[measure] = std::accumulate(values.begin(), values.end(), 0.0);
                break;
                
            case StatisticalMeasure::STDDEV: {
                double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
                double variance = 0.0;
                for (double value : values) {
                    variance += (value - mean) * (value - mean);
                }
                variance /= values.size();
                result.values[measure] = std::sqrt(variance);
                break;
            }
            
            default:
                result.values[measure] = std::numeric_limits<double>::quiet_NaN();
                break;
        }
    }
    
    return result;
}

void RasterStatistics::validateZonalStatisticsInputs(
    const oscean::core_services::GridData& valueRaster,
    const ZonalStatisticsOptions& options) const {
    
    if (valueRaster.getData().empty()) {
        throw InvalidInputDataException("Value raster data is empty");
    }
    
    if (options.statistics.empty()) {
        throw InvalidParameterException("No statistics measures specified");
    }
}

oscean::core_services::GridData RasterStatistics::rasterizeGeometryForZones(
    const oscean::core_services::Geometry& geometry,
    const oscean::core_services::GridDefinition& targetGridDef) const {
    
    // 简化实现：创建一个全为1的栅格
    oscean::core_services::GridData result(targetGridDef, DataType::Float32, 1);
    
    auto& resultBuffer = result.getUnifiedBuffer();
    float* data = reinterpret_cast<float*>(resultBuffer.data());
    size_t totalPixels = targetGridDef.rows * targetGridDef.cols;
    
    std::fill(data, data + totalPixels, 1.0f);
    
    return result;
}

} // namespace oscean::core_services::spatial_ops::raster 