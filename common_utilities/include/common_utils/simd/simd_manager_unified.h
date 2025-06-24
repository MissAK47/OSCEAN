/**
 * @file simd_manager_unified.h
 * @brief 统一SIMD管理器实现 - 重构后的核心SIMD管理器
 * @author OSCEAN Team
 * @date 2024
 * 
 * 重构目标：
 * ✅ 实现ISIMDManager统一接口
 * ✅ 支持异步操作（基于boost::future）
 * ✅ 整合capabilities和basic operations功能
 * ✅ 提供海洋数据处理专用优化
 */

#pragma once

// 首先包含接口和配置
#include "isimd_manager.h"
#include "simd_config.h"

// 然后包含统一的Boost配置（必须在任何Boost头文件之前）
#include "common_utils/utilities/boost_config.h"

// 接着包含Boost头文件
#include <boost/thread/future.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

// 最后包含标准库头文件
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <atomic>
#include <mutex>
#include <chrono>

namespace oscean::common_utils::simd {

/**
 * @brief 统一SIMD管理器实现类
 * 
 * 实现了ISIMDManager接口，提供完整的SIMD功能：
 * - 基础向量运算
 * - 数学操作
 * - 地理操作 
 * - 内存操作
 * - 异步操作支持
 * - 海洋数据专用优化
 */
class UnifiedSIMDManager : public ISIMDManager {
public:
    explicit UnifiedSIMDManager(const SIMDConfig& config = SIMDConfig::createOptimal());
    virtual ~UnifiedSIMDManager();
    
    // === 线程池支持 ===
    void setThreadPool(std::shared_ptr<boost::asio::thread_pool> threadPool) override;
    std::shared_ptr<boost::asio::thread_pool> getThreadPool() const override;
    boost::future<void> createDefaultThreadPoolAsync(size_t poolSize = 0);
    
    // === 基础向量运算（同步接口实现）===
    
    void vectorAdd(const float* a, const float* b, float* result, size_t count) override;
    void vectorSub(const float* a, const float* b, float* result, size_t count) override;
    void vectorMul(const float* a, const float* b, float* result, size_t count) override;
    void vectorDiv(const float* a, const float* b, float* result, size_t count) override;
    void vectorScalarAdd(const float* a, float scalar, float* result, size_t count) override;
    void vectorScalarMul(const float* a, float scalar, float* result, size_t count) override;
    void vectorScalarDiv(const float* a, float scalar, float* result, size_t count) override;
    void vectorAdd(const double* a, const double* b, double* result, size_t count) override;
    void vectorMul(const double* a, const double* b, double* result, size_t count) override;
    void vectorScalarMul(const double* a, double scalar, double* result, size_t count) override;
    void vectorFMA(const float* a, const float* b, const float* c, float* result, size_t count) override;
    void vectorFMA(const double* a, const double* b, const double* c, double* result, size_t count) override;
    
    // === 数学操作（同步接口实现）===
    
    void vectorSqrt(const float* a, float* result, size_t count) override;
    void vectorSqrt(const double* a, double* result, size_t count) override;
    void vectorSquare(const float* a, float* result, size_t count) override;
    void vectorAbs(const float* a, float* result, size_t count) override;
    void vectorFloor(const float* a, float* result, size_t count) override;
    void vectorCeil(const float* a, float* result, size_t count) override;
    float vectorSum(const float* data, size_t count) override;
    double vectorSum(const double* data, size_t count) override;
    float vectorMin(const float* data, size_t count) override;
    float vectorMax(const float* data, size_t count) override;
    float vectorMean(const float* data, size_t count) override;
    float dotProduct(const float* a, const float* b, size_t count) override;
    double dotProduct(const double* a, const double* b, size_t count) override;
    float vectorLength(const float* a, size_t count) override;
    float vectorDistance(const float* a, const float* b, size_t count) override;
    void vectorEqual(const float* a, const float* b, float* result, size_t count) override;
    void vectorGreater(const float* a, const float* b, float* result, size_t count) override;
    void vectorLess(const float* a, const float* b, float* result, size_t count) override;
    
    // === 地理操作（同步接口实现）===
    
    void bilinearInterpolate(
        const float* gridData, const float* xCoords, const float* yCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    ) override;
    
    void bicubicInterpolate(
        const float* gridData, const float* xCoords, const float* yCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    ) override;
    
    void linearInterpolate(
        const float* values, const float* weights, float* results, size_t count
    ) override;
    
    void transformCoordinates(
        const double* srcX, const double* srcY, double* dstX, double* dstY,
        size_t count, const double* transformMatrix
    ) override;
    
    void projectCoordinates(
        const double* lon, const double* lat, double* x, double* y,
        size_t count, int fromCRS, int toCRS
    ) override;
    
    void distanceCalculation(
        const float* x1, const float* y1, const float* x2, const float* y2,
        float* distances, size_t count
    ) override;
    
    void bufferPoints(
        const float* x, const float* y, const float* distances,
        float* bufferX, float* bufferY, size_t count, int segments = 16
    ) override;
    
    void rasterResample(
        const float* srcData, float* dstData,
        size_t srcWidth, size_t srcHeight,
        size_t dstWidth, size_t dstHeight
    ) override;
    
    void rasterMaskApply(
        const float* rasterData, const uint8_t* mask, float* output,
        size_t pixelCount, float noDataValue
    ) override;
    
    // === 扩展地理操作 ===
    
    void calculateHaversineDistances(
        const double* lat1, const double* lon1, 
        const double* lat2, const double* lon2,
        double* distances, size_t count
    );
    
    void mercatorProjection(
        const double* lon, const double* lat, double* x, double* y,
        size_t count, double centralMeridian = 0.0
    );
    
    void polarProjection(
        const double* lon, const double* lat, double* x, double* y,
        size_t count, double centralLon, double centralLat
    );
    
    // === 数据转换操作（SIMD特有）===
    
    void convertFloat32ToFloat64(const float* src, double* dst, size_t count) override;
    void convertFloat64ToFloat32(const double* src, float* dst, size_t count) override;
    void convertIntToFloat(const int32_t* src, float* dst, size_t count) override;
    
    void byteSwap16(const uint16_t* src, uint16_t* dst, size_t count) override;
    void byteSwap32(const uint32_t* src, uint32_t* dst, size_t count) override;
    void byteSwap64(const uint64_t* src, uint64_t* dst, size_t count) override;
    
    size_t compressFloats(const float* src, uint8_t* dst, size_t count, float precision) override;
    size_t decompressFloats(const uint8_t* src, float* dst, size_t compressedSize) override;
    
    // === 扩展类型转换 ===
    
    void convertInt16ToFloat(const int16_t* src, float* dst, size_t count);
    void convertUInt8ToFloat(const uint8_t* src, float* dst, size_t count, float scale = 1.0f);
    void convertFloatToUInt8(const float* src, uint8_t* dst, size_t count, float scale = 1.0f);
    
    // === 异步接口实现（基于boost::future）===
    
    boost::future<void> vectorAddAsync(const float* a, const float* b, float* result, size_t count) override;
    boost::future<void> vectorMulAsync(const float* a, const float* b, float* result, size_t count) override;
    boost::future<void> vectorScalarMulAsync(const float* a, float scalar, float* result, size_t count) override;
    boost::future<float> vectorSumAsync(const float* data, size_t count) override;
    boost::future<double> vectorSumAsync(const double* data, size_t count) override;
    boost::future<float> dotProductAsync(const float* a, const float* b, size_t count) override;
    
    boost::future<void> bilinearInterpolateAsync(
        const float* gridData, const float* xCoords, const float* yCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    ) override;
    
    boost::future<void> transformCoordinatesAsync(
        const double* srcX, const double* srcY, double* dstX, double* dstY,
        size_t count, const double* transformMatrix
    ) override;
    
    // === 扩展异步接口 ===
    
    boost::future<void> vectorSubAsync(const float* a, const float* b, float* result, size_t count);
    boost::future<void> vectorDivAsync(const float* a, const float* b, float* result, size_t count);
    boost::future<void> vectorScalarAddAsync(const float* a, float scalar, float* result, size_t count);
    boost::future<void> vectorScalarDivAsync(const float* a, float scalar, float* result, size_t count);
    boost::future<void> vectorAddAsync(const double* a, const double* b, double* result, size_t count);
    boost::future<void> vectorScalarMulAsync(const double* a, double scalar, double* result, size_t count);
    boost::future<void> vectorFMAAsync(const float* a, const float* b, const float* c, float* result, size_t count);
    boost::future<void> vectorFMAAsync(const double* a, const double* b, const double* c, double* result, size_t count);
    
    // === 配置和信息接口实现 ===
    
    SIMDImplementation getImplementationType() const override;
    SIMDFeatures getFeatures() const override;
    std::string getImplementationName() const override;
    size_t getOptimalBatchSize() const override;
    size_t getAlignment() const override;
    double getBenchmarkScore() const override;
    bool isOptimizedFor(const std::string& operation) const override;
    
    // === 状态和诊断实现 ===
    
    void warmup() override;
    std::string getStatusReport() const override;
    void resetCounters() override;
    
    // === 配置管理实现 ===
    
    void updateConfig(const SIMDConfig& config) override;
    const SIMDConfig& getConfig() const override { return config_; }
    
    // === 海洋数据专用方法 ===
    
    /**
     * @brief 大数组插值运算 (海洋数据常用)
     */
    void interpolateArrays(
        const float* input1, const float* input2, float* output,
        size_t size, float factor
    );
    
    boost::future<void> interpolateArraysAsync(
        const float* input1, const float* input2, float* output,
        size_t size, float factor
    );
    
    /**
     * @brief 栅格数据统计 (元数据计算常用)
     */
    struct StatisticsResult {
        double min, max, mean, stddev;
    };
    
    StatisticsResult calculateStatistics(const float* data, size_t size);
    boost::future<StatisticsResult> calculateStatisticsAsync(const float* data, size_t size);
    
    // === 性能监控 ===
    
    void enablePerformanceMonitoring(bool enable = true) { performanceMonitoring_ = enable; }
    bool isPerformanceMonitoringEnabled() const { return performanceMonitoring_; }
    void enableAutoOptimization(bool enable = true) { autoOptimization_ = enable; }
    bool isAutoOptimizationEnabled() const { return autoOptimization_; }
    void optimizeForWorkload(const std::string& workloadType);
    bool performSelfTest();
    
    // === 异步配置和管理 ===
    
    boost::future<void> warmupAsync();
    boost::future<std::string> getStatusReportAsync() const;
    boost::future<void> resetCountersAsync();
    boost::future<double> getBenchmarkScoreAsync();
    boost::future<void> updateConfigAsync(const SIMDConfig& config);

    // === 友元类声明 ===
    friend class OceanDataSIMDOperations;

private:
    // === 配置和状态 ===
    SIMDConfig config_;
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    
    // === 性能监控 ===
    mutable std::atomic<bool> performanceMonitoring_;
    mutable std::atomic<bool> autoOptimization_;
    mutable std::atomic<size_t> operationCount_;
    mutable std::mutex timingMutex_;
    mutable std::shared_mutex configMutex_;

    // === 内部辅助方法 ===
    void initialize();
    void selectOptimalImplementation();
    void applyConfiguration();
    
    // === 性能记录 ===
    void recordOperation(const std::string& operation, double timeMs) const;
    void updatePerformanceCounters(const std::string& operation) const;
    
    // === 错误处理 ===
    void handleError(const std::string& operation, const std::exception& e) const;
    bool validateInputs(const void* ptr, size_t size, const std::string& operation) const;
    
    // === 异步执行支持 ===
    template<typename Func>
    auto executeAsync(Func&& func) -> boost::future<decltype(func())>;
    
    // 🔴 修复：添加const版本的executeAsync支持const方法调用
    template<typename Func>
    auto executeAsync(Func&& func) const -> boost::future<decltype(func())>;
    
    // === SIMD实现选择 ===
    void selectImplementation();
    bool testImplementation(SIMDImplementation impl);
    
    // === 内部实现方法 ===
    void vectorAddImpl(const float* a, const float* b, float* result, size_t count);
    void vectorMulImpl(const float* a, const float* b, float* result, size_t count);
    void vectorScalarMulImpl(const float* a, float scalar, float* result, size_t count);
    float vectorSumImpl(const float* data, size_t count);
    void bilinearInterpolateImpl(
        const float* gridData, const float* xCoords, const float* yCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    );
};

/**
 * @brief 海洋数据专用SIMD操作类
 * 
 * 提供针对海洋学数据处理优化的SIMD操作
 */
class OceanDataSIMDOperations {
public:
    explicit OceanDataSIMDOperations(std::shared_ptr<UnifiedSIMDManager> manager);
    
    // === 海洋学数据插值 ===
    
    void interpolateTemperatureField(
        const float* tempGrid, const float* latCoords, const float* lonCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    );
    
    void interpolateSalinityField(
        const float* salGrid, const float* latCoords, const float* lonCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    );
    
    void interpolateCurrentField(
        const float* uGrid, const float* vGrid, 
        const float* latCoords, const float* lonCoords,
        float* uResults, float* vResults, size_t count,
        size_t gridWidth, size_t gridHeight
    );
    
    // === 海洋学统计计算 ===
    
    void calculateSeasonalMeans(
        const float* timeSeriesData, float* seasonalMeans,
        size_t timeSteps, size_t spatialPoints
    );
    
    void calculateAnomalies(
        const float* data, const float* climatology, float* anomalies,
        size_t count
    );
    
    // === 海洋学空间操作 ===
    
    void calculateDistanceToCoast(
        const float* pointsX, const float* pointsY,
        const float* coastlineX, const float* coastlineY,
        float* distances, size_t pointCount, size_t coastlineCount
    );
    
    void projectToStereographic(
        const double* lon, const double* lat, double* x, double* y,
        size_t count, double centralLon, double centralLat
    );
    
    // === 异步海洋学操作 ===
    
    boost::future<void> interpolateTemperatureFieldAsync(
        const float* tempGrid, const float* latCoords, const float* lonCoords,
        float* results, size_t count, size_t gridWidth, size_t gridHeight
    );
    
    boost::future<void> calculateSeasonalMeansAsync(
        const float* timeSeriesData, float* seasonalMeans,
        size_t timeSteps, size_t spatialPoints
    );

    // === 扩展海洋学计算方法 ===
    
    void calculateSeawaterDensity(
        const float* temperature, const float* salinity, float* density, size_t count,
        float pressure = 0.0f
    ) const;
    
    void calculateSoundSpeed(
        const float* temperature, const float* salinity, const float* depth,
        float* soundSpeed, size_t count
    ) const;
    
    void calculateBuoyancyFrequency(
        const float* density, const float* depths, float* buoyancyFreq, size_t count
    ) const;
    
    float calculateMixedLayerDepth(
        const float* temperature, const float* depths, size_t count,
        float temperatureCriterion = 0.2f
    ) const;
    
    float calculateHeatContent(
        const float* temperature, const float* depths, size_t count,
        float referenceTemperature = 0.0f
    ) const;
    
    // === 扩展异步海洋学方法 ===
    
    boost::future<void> calculateSeawaterDensityAsync(
        const float* temperature, const float* salinity, float* density, size_t count,
        float pressure = 0.0f
    ) const;
    
    boost::future<void> calculateSoundSpeedAsync(
        const float* temperature, const float* salinity, const float* depth,
        float* soundSpeed, size_t count
    ) const;
    
    boost::future<float> calculateMixedLayerDepthAsync(
        const float* temperature, const float* depths, size_t count,
        float temperatureCriterion = 0.2f
    ) const;
    
    boost::future<float> calculateHeatContentAsync(
        const float* temperature, const float* depths, size_t count,
        float referenceTemperature = 0.0f
    ) const;

private:
    std::shared_ptr<UnifiedSIMDManager> manager_;
    
    // === 数据验证 ===
    void validateOceanographicData(const float* data, size_t count, 
                                  const std::string& dataType) const;
    boost::future<void> validateOceanographicDataAsync(const float* data, size_t count, 
                                                      const std::string& dataType) const;
    float calculateHaversineDistance(double lat1, double lon1, double lat2, double lon2) const;
    boost::future<float> calculateHaversineDistanceAsync(double lat1, double lon1, double lat2, double lon2) const;
};

} // namespace oscean::common_utils::simd

// =============================================================================
// 模板方法实现 - 必须在头文件中定义
// =============================================================================

namespace oscean::common_utils::simd {

template<typename Func>
auto UnifiedSIMDManager::executeAsync(Func&& func) -> boost::future<decltype(func())> {
    if (!threadPool_) {
        throw std::runtime_error("Thread pool not initialized");
    }
    
    auto task = std::make_shared<boost::packaged_task<decltype(func())()>>(
        std::forward<Func>(func)
    );
    
    auto future = task->get_future();
    
    boost::asio::post(*threadPool_, [task]() {
        try {
            (*task)();
        } catch (...) {
            // 错误已经由packaged_task处理
        }
    });
    
    return future;
}

template<typename Func>
auto UnifiedSIMDManager::executeAsync(Func&& func) const -> boost::future<decltype(func())> {
    if (!threadPool_) {
        throw std::runtime_error("Thread pool not initialized");
    }
    
    auto task = std::make_shared<boost::packaged_task<decltype(func())()>>(
        std::forward<Func>(func)
    );
    
    auto future = task->get_future();
    
    boost::asio::post(*threadPool_, [task]() {
        try {
            (*task)();
        } catch (...) {
            // 错误已经由packaged_task处理
        }
    });
    
    return future;
}

} // namespace oscean::common_utils::simd 