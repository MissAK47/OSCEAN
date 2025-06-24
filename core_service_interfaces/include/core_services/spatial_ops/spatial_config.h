#pragma once

// 防止Windows API宏干扰
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// 取消可能冲突的Windows宏
#ifdef STRICT
#undef STRICT
#endif
#ifdef NONE
#undef NONE
#endif
#ifdef AUTO
#undef AUTO
#endif

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <chrono>

namespace oscean::core_services::spatial_ops {

enum class ParallelStrategy {
    NONE,
    THREAD_POOL,
    TASK_BASED,
    DATA_PARALLEL,
    AUTO
};

enum class IndexStrategy {
    NONE,
    R_TREE,
    QUAD_TREE,
    GRID,
    AUTO
};

enum class MemoryStrategy {
    STANDARD,
    POOLED,
    MAPPED,
    STREAMING,
    AUTO
};

enum class GeometryPrecisionModel {
    FLOATING,
    FLOATING_SINGLE,
    FIXED
};

enum class GeometryValidationLevel {
    NONE,
    BASIC,
    STANDARD,
    STRICT
};

enum class GeometryOptimizationLevel {
    NONE,
    BASIC,
    STANDARD,
    AGGRESSIVE
};

struct ParallelConfig {
    ParallelStrategy strategy;
    std::size_t maxThreads;
    std::size_t minDataSizeForParallelism;
    std::size_t chunkSize;
    bool enableLoadBalancing;
    double loadBalanceThreshold;
    
    ParallelConfig() {
        strategy = ParallelStrategy::AUTO;
        maxThreads = 0;
        minDataSizeForParallelism = 1000000;
        chunkSize = 1024 * 1024;
        enableLoadBalancing = true;
        loadBalanceThreshold = 0.8;
    }
};

struct GdalConfig {
    long long gdalCacheMaxBytes;
    std::string numThreads;
    std::map<std::string, std::string> gdalOptions;
    std::map<std::string, std::string> gdalWarpOptions;
    bool enableGDALOptimizations;
    std::size_t blockCacheSize;
    std::string tempDirectory;
    
    GdalConfig() {
        gdalCacheMaxBytes = 256LL * 1024 * 1024;
        numThreads = "ALL_CPUS";
        enableGDALOptimizations = true;
        blockCacheSize = 40 * 1024 * 1024;
    }
};

struct IndexConfig {
    IndexStrategy strategy;
    std::size_t indexThreshold;
    std::size_t maxIndexDepth;
    std::size_t maxLeafCapacity;
    double indexBuildRatio;
    bool enableIndexCaching;
    std::size_t maxCachedIndices;
    
    IndexConfig() {
        strategy = IndexStrategy::AUTO;
        indexThreshold = 1000;
        maxIndexDepth = 10;
        maxLeafCapacity = 16;
        indexBuildRatio = 0.7;
        enableIndexCaching = true;
        maxCachedIndices = 10;
    }
};

struct MemoryConfig {
    MemoryStrategy strategy;
    std::size_t maxMemoryUsage;
    std::size_t geometryPoolSize;
    std::size_t rasterPoolSize;
    bool enableMemoryPooling;
    bool enableMemoryMapping;
    double memoryPressureThreshold;
    
    MemoryConfig() {
        strategy = MemoryStrategy::AUTO;
        maxMemoryUsage = 0;
        geometryPoolSize = 1000;
        rasterPoolSize = 100;
        enableMemoryPooling = false;
        enableMemoryMapping = true;
        memoryPressureThreshold = 0.85;
    }
};

struct AlgorithmConfig {
    double geometricTolerance;
    double simplificationTolerance;
    std::size_t maxIterations;
    bool enableProgressiveRefinement;
    bool enableApproximateAlgorithms;
    std::string defaultResamplingMethod;
    
    AlgorithmConfig() {
        geometricTolerance = 1e-9;
        simplificationTolerance = 1e-6;
        maxIterations = 1000;
        enableProgressiveRefinement = true;
        enableApproximateAlgorithms = false;
        defaultResamplingMethod = "bilinear";
    }
};

struct PerformanceConfig {
    bool enableSpatialIndex;
    std::size_t spatialIndexThreshold;
    bool enableMemoryPooling;
    bool enablePerformanceMonitoring;
    bool enableMetricsCollection;
    std::size_t metricsBufferSize;
    std::string metricsOutputPath;
    bool enableProfilingMode;
    
    PerformanceConfig() {
        enableSpatialIndex = true;
        spatialIndexThreshold = 1000;
        enableMemoryPooling = false;
        enablePerformanceMonitoring = false;
        enableMetricsCollection = false;
        metricsBufferSize = 10000;
        enableProfilingMode = false;
    }
};

struct LoggingConfig {
    std::string logLevel;
    std::string logFormat;
    std::string logOutputPath;
    bool enableFileLogging;
    std::size_t maxLogFileSize;
    std::size_t maxLogFiles;
    bool enableAsyncLogging;
    
    LoggingConfig() {
        logLevel = "INFO";
        logFormat = "default";
        enableFileLogging = false;
        maxLogFileSize = 100 * 1024 * 1024;
        maxLogFiles = 5;
        enableAsyncLogging = true;
    }
};

struct ValidationConfig {
    bool enableInputValidation;
    bool enableGeometryValidation;
    bool enableCRSValidation;
    bool enableBoundsChecking;
    bool strictMode;
    double validationTolerance;
    
    ValidationConfig() {
        enableInputValidation = true;
        enableGeometryValidation = true;
        enableCRSValidation = true;
        enableBoundsChecking = true;
        strictMode = false;
        validationTolerance = 1e-10;
    }
};

struct SpatialOpsConfig {
    ParallelConfig parallelSettings;
    GdalConfig gdalSettings;
    IndexConfig indexSettings;
    MemoryConfig memorySettings;
    AlgorithmConfig algorithmSettings;
    PerformanceConfig performanceSettings;
    LoggingConfig loggingSettings;
    ValidationConfig validationSettings;
    
    std::string defaultCRS;
    std::string serviceName;
    std::string version;
    
    bool enableDebugMode;
    bool enableExperimentalFeatures;
    std::vector<std::string> enabledFeatures;
    std::map<std::string, std::string> customSettings;
    
    SpatialOpsConfig() {
        defaultCRS = "EPSG:4326";
        serviceName = "SpatialOpsService";
        version = "1.0.0";
        enableDebugMode = false;
        enableExperimentalFeatures = false;
    }
    
    bool validate() const {
        return true;
    }
};

} // namespace oscean::core_services::spatial_ops