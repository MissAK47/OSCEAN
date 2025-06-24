/**
 * @file simd_manager_ocean.cpp
 * @brief 海洋数据专用SIMD操作实现
 * @author OSCEAN Team
 * @date 2024
 */

#include "common_utils/simd/simd_manager_unified.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace oscean::common_utils::simd {

// === OceanDataSIMDOperations 类实现 ===

OceanDataSIMDOperations::OceanDataSIMDOperations(std::shared_ptr<UnifiedSIMDManager> manager)
    : manager_(manager) {
    if (!manager_) {
        throw std::invalid_argument("UnifiedSIMDManager cannot be null");
    }
}

// === 海洋学数据插值 ===

void OceanDataSIMDOperations::interpolateTemperatureField(
    const float* tempGrid, const float* latCoords, const float* lonCoords,
    float* results, size_t count, size_t gridWidth, size_t gridHeight) {
    
    manager_->bilinearInterpolate(tempGrid, lonCoords, latCoords, results, count, gridWidth, gridHeight);
}

void OceanDataSIMDOperations::interpolateSalinityField(
    const float* salGrid, const float* latCoords, const float* lonCoords,
    float* results, size_t count, size_t gridWidth, size_t gridHeight) {
    
    manager_->bilinearInterpolate(salGrid, lonCoords, latCoords, results, count, gridWidth, gridHeight);
}

void OceanDataSIMDOperations::interpolateCurrentField(
    const float* uGrid, const float* vGrid, 
    const float* latCoords, const float* lonCoords,
    float* uResults, float* vResults, size_t count,
    size_t gridWidth, size_t gridHeight) {
    
    manager_->bilinearInterpolate(uGrid, lonCoords, latCoords, uResults, count, gridWidth, gridHeight);
    manager_->bilinearInterpolate(vGrid, lonCoords, latCoords, vResults, count, gridWidth, gridHeight);
}

// === 海洋学统计计算 ===

void OceanDataSIMDOperations::calculateSeasonalMeans(
    const float* timeSeriesData, float* seasonalMeans,
    size_t timeSteps, size_t spatialPoints) {
    
    // 简化实现 - 实际应按季节分组计算均值
    // 这里假设每4个时间步为一个季节
    const size_t SEASON_LENGTH = 4;
    const size_t numSeasons = timeSteps / SEASON_LENGTH;
    
    for (size_t point = 0; point < spatialPoints; ++point) {
        float totalSum = 0.0f;
        size_t validCount = 0;
        
        for (size_t season = 0; season < numSeasons; ++season) {
            float seasonSum = 0.0f;
            size_t seasonCount = 0;
            
            for (size_t step = 0; step < SEASON_LENGTH; ++step) {
                size_t timeIndex = season * SEASON_LENGTH + step;
                if (timeIndex < timeSteps) {
                    size_t dataIndex = timeIndex * spatialPoints + point;
                    seasonSum += timeSeriesData[dataIndex];
                    seasonCount++;
                }
            }
            
            if (seasonCount > 0) {
                totalSum += seasonSum / seasonCount;
                validCount++;
            }
        }
        
        seasonalMeans[point] = (validCount > 0) ? (totalSum / validCount) : 0.0f;
    }
}

void OceanDataSIMDOperations::calculateAnomalies(
    const float* data, const float* climatology, float* anomalies,
    size_t count) {
    
    manager_->vectorSub(data, climatology, anomalies, count);
}

// === 海洋学空间操作 ===

void OceanDataSIMDOperations::calculateDistanceToCoast(
    const float* pointsX, const float* pointsY,
    const float* coastlineX, const float* coastlineY,
    float* distances, size_t pointCount, size_t coastlineCount) {
    
    // 简化实现 - 计算到海岸线最近点的距离
    for (size_t i = 0; i < pointCount; ++i) {
        float minDistance = std::numeric_limits<float>::max();
        
        for (size_t j = 0; j < coastlineCount; ++j) {
            float dx = pointsX[i] - coastlineX[j];
            float dy = pointsY[i] - coastlineY[j];
            float distance = std::sqrt(dx * dx + dy * dy);
            
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        
        distances[i] = minDistance;
    }
}

void OceanDataSIMDOperations::projectToStereographic(
    const double* lon, const double* lat, double* x, double* y,
    size_t count, double centralLon, double centralLat) {
    
    // 简化的立体投影实现
    const double PI = 3.14159265358979323846;
    const double EARTH_RADIUS = 6371000.0; // 地球半径(米)
    
    // 转换为弧度
    double centralLonRad = centralLon * PI / 180.0;
    double centralLatRad = centralLat * PI / 180.0;
    
    for (size_t i = 0; i < count; ++i) {
        double lonRad = lon[i] * PI / 180.0;
        double latRad = lat[i] * PI / 180.0;
        
        // 立体投影公式
        double k = 2.0 / (1.0 + std::sin(centralLatRad) * std::sin(latRad) +
                         std::cos(centralLatRad) * std::cos(latRad) * std::cos(lonRad - centralLonRad));
        
        x[i] = EARTH_RADIUS * k * std::cos(latRad) * std::sin(lonRad - centralLonRad);
        y[i] = EARTH_RADIUS * k * (std::cos(centralLatRad) * std::sin(latRad) -
                                  std::sin(centralLatRad) * std::cos(latRad) * std::cos(lonRad - centralLonRad));
    }
}

// === 异步海洋学操作 ===

boost::future<void> OceanDataSIMDOperations::interpolateTemperatureFieldAsync(
    const float* tempGrid, const float* latCoords, const float* lonCoords,
    float* results, size_t count, size_t gridWidth, size_t gridHeight) {
    
    return manager_->bilinearInterpolateAsync(tempGrid, lonCoords, latCoords, results, count, gridWidth, gridHeight);
}

boost::future<void> OceanDataSIMDOperations::calculateSeasonalMeansAsync(
    const float* timeSeriesData, float* seasonalMeans,
    size_t timeSteps, size_t spatialPoints) {
    
    return manager_->executeAsync([this, timeSeriesData, seasonalMeans, timeSteps, spatialPoints]() {
        calculateSeasonalMeans(timeSeriesData, seasonalMeans, timeSteps, spatialPoints);
    });
}

// === 数据验证 ===

void OceanDataSIMDOperations::validateOceanographicData(const float* data, size_t count, 
                                                       const std::string& dataType) const {
    
    if (!data || count == 0) {
        std::cout << "OceanDataSIMDOperations: Invalid data for " << dataType << std::endl;
        return;
    }
    
    // 根据数据类型进行特定验证
    if (dataType == "temperature") {
        // 海水温度通常在 -2°C 到 40°C 之间
        for (size_t i = 0; i < count; ++i) {
            if (data[i] < -5.0f || data[i] > 50.0f) {
                std::cout << "Warning: Temperature value " << data[i] 
                         << " at index " << i << " seems out of range" << std::endl;
            }
        }
    } else if (dataType == "salinity") {
        // 海水盐度通常在 0-50 PSU 之间
        for (size_t i = 0; i < count; ++i) {
            if (data[i] < 0.0f || data[i] > 60.0f) {
                std::cout << "Warning: Salinity value " << data[i] 
                         << " at index " << i << " seems out of range" << std::endl;
            }
        }
    } else if (dataType == "current") {
        // 海流速度通常不超过 5 m/s
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(data[i]) > 10.0f) {
                std::cout << "Warning: Current velocity " << data[i] 
                         << " at index " << i << " seems very high" << std::endl;
            }
        }
    } else if (dataType == "depth") {
        // 海洋深度应该为正值
        for (size_t i = 0; i < count; ++i) {
            if (data[i] < 0.0f || data[i] > 12000.0f) {
                std::cout << "Warning: Depth value " << data[i] 
                         << " at index " << i << " seems out of range" << std::endl;
            }
        }
    }
    
    std::cout << "OceanDataSIMDOperations: Validated " << dataType 
             << " data with " << count << " elements" << std::endl;
}

boost::future<void> OceanDataSIMDOperations::validateOceanographicDataAsync(const float* data, size_t count, 
                                                                           const std::string& dataType) const {
    
    return manager_->executeAsync([this, data, count, dataType]() {
        validateOceanographicData(data, count, dataType);
    });
}

// === 地理距离计算 ===

float OceanDataSIMDOperations::calculateHaversineDistance(double lat1, double lon1, double lat2, double lon2) const {
    const double EARTH_RADIUS_KM = 6371.0;
    const double PI = 3.14159265358979323846;
    
    // 转换为弧度
    lat1 = lat1 * PI / 180.0;
    lon1 = lon1 * PI / 180.0;
    lat2 = lat2 * PI / 180.0;
    lon2 = lon2 * PI / 180.0;
    
    double dlat = lat2 - lat1;
    double dlon = lon2 - lon1;
    
    double a = std::sin(dlat / 2) * std::sin(dlat / 2) +
               std::cos(lat1) * std::cos(lat2) *
               std::sin(dlon / 2) * std::sin(dlon / 2);
               
    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
    
    return static_cast<float>(EARTH_RADIUS_KM * c);
}

boost::future<float> OceanDataSIMDOperations::calculateHaversineDistanceAsync(double lat1, double lon1, double lat2, double lon2) const {
    return manager_->executeAsync([this, lat1, lon1, lat2, lon2]() -> float {
        return calculateHaversineDistance(lat1, lon1, lat2, lon2);
    });
}

// === 扩展海洋学方法 ===

/**
 * @brief 计算海水密度 (基于温度和盐度)
 */
void OceanDataSIMDOperations::calculateSeawaterDensity(
    const float* temperature, const float* salinity, float* density, size_t count,
    float pressure) const {
    
    // 简化的海水密度计算 (基于UNESCO公式简化版)
    const float REFERENCE_DENSITY = 1000.0f; // kg/m³
    
    for (size_t i = 0; i < count; ++i) {
        float T = temperature[i];
        float S = salinity[i];
        
        // 简化计算 - 实际应使用完整的状态方程
        float tempEffect = 1.0f - 0.0002f * (T - 4.0f);
        float salinityEffect = 1.0f + 0.0008f * S;
        float pressureEffect = 1.0f + 0.00005f * pressure; // 简化的压力影响
        
        density[i] = REFERENCE_DENSITY * tempEffect * salinityEffect * pressureEffect;
    }
}

/**
 * @brief 计算海水声速
 */
void OceanDataSIMDOperations::calculateSoundSpeed(
    const float* temperature, const float* salinity, const float* depth,
    float* soundSpeed, size_t count) const {
    
    // 简化的声速计算 (基于Chen & Millero公式简化版)
    for (size_t i = 0; i < count; ++i) {
        float T = temperature[i];
        float S = salinity[i];
        float D = depth[i];
        
        // 简化计算
        float tempTerm = 1449.2f + 4.6f * T - 0.055f * T * T + 0.00029f * T * T * T;
        float salinityTerm = (1.34f - 0.01f * T) * (S - 35.0f);
        float depthTerm = 0.016f * D;
        
        soundSpeed[i] = tempTerm + salinityTerm + depthTerm;
    }
}

/**
 * @brief 计算浮力频率 (Brunt-Väisälä频率)
 */
void OceanDataSIMDOperations::calculateBuoyancyFrequency(
    const float* density, const float* depths, float* buoyancyFreq, size_t count) const {
    
    const float GRAVITY = 9.81f; // m/s²
    
    // 计算密度梯度和浮力频率
    for (size_t i = 1; i < count; ++i) {
        float densityGradient = (density[i] - density[i-1]) / (depths[i] - depths[i-1]);
        float meanDensity = (density[i] + density[i-1]) / 2.0f;
        
        float N2 = -(GRAVITY / meanDensity) * densityGradient;
        buoyancyFreq[i] = (N2 > 0.0f) ? std::sqrt(N2) : 0.0f;
    }
    
    // 第一个点设为第二个点的值
    if (count > 1) {
        buoyancyFreq[0] = buoyancyFreq[1];
    }
}

/**
 * @brief 计算混合层深度
 */
float OceanDataSIMDOperations::calculateMixedLayerDepth(
    const float* temperature, const float* depths, size_t count,
    float temperatureCriterion) const {
    
    if (count < 2) return 0.0f;
    
    float surfaceTemp = temperature[0];
    
    // 寻找温度差超过阈值的深度
    for (size_t i = 1; i < count; ++i) {
        if (std::abs(temperature[i] - surfaceTemp) > temperatureCriterion) {
            // 线性插值找到精确深度
            float ratio = temperatureCriterion / std::abs(temperature[i] - surfaceTemp);
            return depths[i-1] + ratio * (depths[i] - depths[i-1]);
        }
    }
    
    // 如果没有找到，返回最大深度
    return depths[count - 1];
}

/**
 * @brief 计算热含量
 */
float OceanDataSIMDOperations::calculateHeatContent(
    const float* temperature, const float* depths, size_t count,
    float referenceTemperature) const {
    
    const float SEAWATER_DENSITY = 1025.0f; // kg/m³
    const float SPECIFIC_HEAT = 3985.0f;    // J/(kg·K)
    
    float heatContent = 0.0f;
    
    for (size_t i = 1; i < count; ++i) {
        float layerThickness = depths[i] - depths[i-1];
        float meanTemp = (temperature[i] + temperature[i-1]) / 2.0f;
        float tempAnomaly = meanTemp - referenceTemperature;
        
        heatContent += SEAWATER_DENSITY * SPECIFIC_HEAT * tempAnomaly * layerThickness;
    }
    
    return heatContent;
}

/**
 * @brief 批量海洋学计算的异步版本
 */
boost::future<void> OceanDataSIMDOperations::calculateSeawaterDensityAsync(
    const float* temperature, const float* salinity, float* density, size_t count,
    float pressure) const {
    
    return manager_->executeAsync([this, temperature, salinity, density, count, pressure]() {
        calculateSeawaterDensity(temperature, salinity, density, count, pressure);
    });
}

boost::future<void> OceanDataSIMDOperations::calculateSoundSpeedAsync(
    const float* temperature, const float* salinity, const float* depth,
    float* soundSpeed, size_t count) const {
    
    return manager_->executeAsync([this, temperature, salinity, depth, soundSpeed, count]() {
        calculateSoundSpeed(temperature, salinity, depth, soundSpeed, count);
    });
}

boost::future<float> OceanDataSIMDOperations::calculateMixedLayerDepthAsync(
    const float* temperature, const float* depths, size_t count,
    float temperatureCriterion) const {
    
    return manager_->executeAsync([this, temperature, depths, count, temperatureCriterion]() -> float {
        return calculateMixedLayerDepth(temperature, depths, count, temperatureCriterion);
    });
}

boost::future<float> OceanDataSIMDOperations::calculateHeatContentAsync(
    const float* temperature, const float* depths, size_t count,
    float referenceTemperature) const {
    
    return manager_->executeAsync([this, temperature, depths, count, referenceTemperature]() -> float {
        return calculateHeatContent(temperature, depths, count, referenceTemperature);
    });
}

} // namespace oscean::common_utils::simd 