/**
 * @file isimd_manager.h
 * @brief 统一SIMD管理器接口 - 重构后的核心SIMD接口
 * @author OSCEAN Team
 * @date 2024
 * 
 * 重构目标：
 * ✅ 提供统一的SIMD管理器接口
 * ✅ 支持异步操作（基于boost::future）
 * ✅ 整合基础、数学、地理、内存操作
 * ✅ 简化接口层次，移除冗余工厂类
 */

#pragma once

#include "simd_config.h"
#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <boost/asio/thread_pool.hpp>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace oscean::common_utils::simd {

/**
 * @brief 统一SIMD管理器接口
 * 
 * 整合了基础操作、数学操作、地理操作、内存操作的统一接口
 * 支持同步和异步操作模式
 */
class ISIMDManager {
public:
    virtual ~ISIMDManager() = default;
    
    // === 基础向量运算（同步接口）===
    
    virtual void vectorAdd(const float* a, const float* b, float* result, size_t count) = 0;
    virtual void vectorSub(const float* a, const float* b, float* result, size_t count) = 0;
    virtual void vectorMul(const float* a, const float* b, float* result, size_t count) = 0;
    virtual void vectorDiv(const float* a, const float* b, float* result, size_t count) = 0;
    
    virtual void vectorScalarAdd(const float* a, float scalar, float* result, size_t count) = 0;
    virtual void vectorScalarMul(const float* a, float scalar, float* result, size_t count) = 0;
    virtual void vectorScalarDiv(const float* a, float scalar, float* result, size_t count) = 0;
    
    // 双精度版本
    virtual void vectorAdd(const double* a, const double* b, double* result, size_t count) = 0;
    virtual void vectorMul(const double* a, const double* b, double* result, size_t count) = 0;
    virtual void vectorScalarMul(const double* a, double scalar, double* result, size_t count) = 0;
    
    // 融合操作
    virtual void vectorFMA(const float* a, const float* b, const float* c, float* result, size_t count) = 0;
    virtual void vectorFMA(const double* a, const double* b, const double* c, double* result, size_t count) = 0;
    
    // === 数学操作（同步接口）===
    
    virtual void vectorSqrt(const float* a, float* result, size_t count) = 0;
    virtual void vectorSqrt(const double* a, double* result, size_t count) = 0;
    virtual void vectorSquare(const float* a, float* result, size_t count) = 0;
    virtual void vectorAbs(const float* a, float* result, size_t count) = 0;
    virtual void vectorFloor(const float* a, float* result, size_t count) = 0;
    virtual void vectorCeil(const float* a, float* result, size_t count) = 0;
    
    // 聚合运算
    virtual float vectorSum(const float* data, size_t count) = 0;
    virtual double vectorSum(const double* data, size_t count) = 0;
    virtual float vectorMin(const float* data, size_t count) = 0;
    virtual float vectorMax(const float* data, size_t count) = 0;
    virtual float vectorMean(const float* data, size_t count) = 0;
    
    // 向量积运算
    virtual float dotProduct(const float* a, const float* b, size_t count) = 0;
    virtual double dotProduct(const double* a, const double* b, size_t count) = 0;
    virtual float vectorLength(const float* a, size_t count) = 0;
    virtual float vectorDistance(const float* a, const float* b, size_t count) = 0;
    
    // 向量比较
    virtual void vectorEqual(const float* a, const float* b, float* result, size_t count) = 0;
    virtual void vectorGreater(const float* a, const float* b, float* result, size_t count) = 0;
    virtual void vectorLess(const float* a, const float* b, float* result, size_t count) = 0;
    
    // === 地理操作（同步接口）===
    
    virtual void bilinearInterpolate(
        const float* gridData, const float* xCoords, const float* yCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    ) = 0;
    
    virtual void bicubicInterpolate(
        const float* gridData, const float* xCoords, const float* yCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    ) = 0;
    
    virtual void linearInterpolate(
        const float* values, const float* weights, float* results, size_t count
    ) = 0;
    
    virtual void transformCoordinates(
        const double* srcX, const double* srcY, double* dstX, double* dstY,
        size_t count, const double* transformMatrix
    ) = 0;
    
    virtual void projectCoordinates(
        const double* lon, const double* lat, double* x, double* y,
        size_t count, int fromCRS, int toCRS
    ) = 0;
    
    virtual void distanceCalculation(
        const float* x1, const float* y1, const float* x2, const float* y2,
        float* distances, size_t count
    ) = 0;
    
    virtual void bufferPoints(
        const float* x, const float* y, const float* distances,
        float* bufferX, float* bufferY, size_t count, int segments = 16
    ) = 0;
    
    virtual void rasterResample(
        const float* srcData, float* dstData,
        size_t srcWidth, size_t srcHeight,
        size_t dstWidth, size_t dstHeight
    ) = 0;
    
    virtual void rasterMaskApply(
        const float* rasterData, const uint8_t* mask, float* output,
        size_t pixelCount, float noDataValue
    ) = 0;
    
    // === 数据转换操作（SIMD特有）===
    
    virtual void convertFloat32ToFloat64(const float* src, double* dst, size_t count) = 0;
    virtual void convertFloat64ToFloat32(const double* src, float* dst, size_t count) = 0;
    virtual void convertIntToFloat(const int32_t* src, float* dst, size_t count) = 0;
    
    virtual void byteSwap16(const uint16_t* src, uint16_t* dst, size_t count) = 0;
    virtual void byteSwap32(const uint32_t* src, uint32_t* dst, size_t count) = 0;
    virtual void byteSwap64(const uint64_t* src, uint64_t* dst, size_t count) = 0;
    
    virtual size_t compressFloats(const float* src, uint8_t* dst, size_t count, float precision) = 0;
    virtual size_t decompressFloats(const uint8_t* src, float* dst, size_t compressedSize) = 0;
    
    // === 异步接口（基于boost::future）===
    
    virtual boost::future<void> vectorAddAsync(const float* a, const float* b, float* result, size_t count) = 0;
    virtual boost::future<void> vectorMulAsync(const float* a, const float* b, float* result, size_t count) = 0;
    virtual boost::future<void> vectorScalarMulAsync(const float* a, float scalar, float* result, size_t count) = 0;
    
    virtual boost::future<float> vectorSumAsync(const float* data, size_t count) = 0;
    virtual boost::future<double> vectorSumAsync(const double* data, size_t count) = 0;
    virtual boost::future<float> dotProductAsync(const float* a, const float* b, size_t count) = 0;
    
    virtual boost::future<void> bilinearInterpolateAsync(
        const float* gridData, const float* xCoords, const float* yCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    ) = 0;
    
    virtual boost::future<void> transformCoordinatesAsync(
        const double* srcX, const double* srcY, double* dstX, double* dstY,
        size_t count, const double* transformMatrix
    ) = 0;
    
    // === 配置和信息接口 ===
    
    virtual SIMDImplementation getImplementationType() const = 0;
    virtual SIMDFeatures getFeatures() const = 0;
    virtual std::string getImplementationName() const = 0;
    virtual size_t getOptimalBatchSize() const = 0;
    virtual size_t getAlignment() const = 0;
    
    virtual double getBenchmarkScore() const = 0;
    virtual bool isOptimizedFor(const std::string& operation) const = 0;
    
    // === 状态和诊断 ===
    
    virtual void warmup() = 0;
    virtual std::string getStatusReport() const = 0;
    virtual void resetCounters() = 0;
    
    // === 配置管理 ===
    
    virtual void updateConfig(const SIMDConfig& config) = 0;
    virtual const SIMDConfig& getConfig() const = 0;
    
    // === 线程池支持 ===
    
    virtual void setThreadPool(std::shared_ptr<boost::asio::thread_pool> threadPool) = 0;
    virtual std::shared_ptr<boost::asio::thread_pool> getThreadPool() const = 0;
};

} // namespace oscean::common_utils::simd 