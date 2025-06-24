#include "impl/spatial_config_manager.h"
#include "core_services/spatial_ops/spatial_config.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <thread>
#include <iostream>

namespace oscean::core_services::spatial_ops {

// 辅助函数：检查字符串是否以指定后缀结尾
bool endsWith(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

SpatialConfigManager::SpatialConfigManager() {
    initializeDefaultTemplates();
}

SpatialConfigManager::~SpatialConfigManager() = default;

SpatialOpsConfig SpatialConfigManager::loadFromFile(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        throw ConfigurationException("Cannot open configuration file: " + configPath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    SpatialOpsConfig config;
    
    // 尝试解析不同格式
    if (endsWith(configPath, ".yaml") || endsWith(configPath, ".yml")) {
        if (!parseYamlConfig(content, config)) {
            throw ConfigurationException("Failed to parse YAML configuration file: " + configPath);
        }
    } else if (endsWith(configPath, ".json")) {
        if (!parseJsonConfig(content, config)) {
            throw ConfigurationException("Failed to parse JSON configuration file: " + configPath);
        }
    } else {
        // 默认尝试YAML格式
        if (!parseYamlConfig(content, config)) {
            throw ConfigurationException("Failed to parse configuration file (unknown format): " + configPath);
        }
    }

    // 应用环境变量覆盖
    loadFromEnvironment(config);
    
    // 验证配置
    if (!validateAndFix(config)) {
        throw ConfigurationException("Configuration validation failed for file: " + configPath);
    }

    return config;
}

void SpatialConfigManager::loadFromEnvironment(SpatialOpsConfig& config) {
    applyEnvironmentOverrides(config);
}

bool SpatialConfigManager::saveToFile(const SpatialOpsConfig& config, const std::string& configPath) {
    try {
        std::string content;
        
        if (endsWith(configPath, ".yaml") || endsWith(configPath, ".yml")) {
            content = configToYaml(config);
        } else if (endsWith(configPath, ".json")) {
            content = configToJson(config);
        } else {
            // 默认使用YAML格式
            content = configToYaml(config);
        }

        std::ofstream file(configPath);
        if (!file.is_open()) {
            return false;
        }

        file << content;
        return file.good();
    } catch (const std::exception&) {
        return false;
    }
}

SpatialOpsConfig SpatialConfigManager::getDefaultConfig() {
    SpatialOpsConfig config;
    
    // 🔧 检查环境变量是否为单线程模式
    const char* runMode = std::getenv("OSCEAN_RUN_MODE");
    bool isSingleThreadMode = (runMode && std::string(runMode) == "SINGLE_THREAD");
    
    // 并行配置
    config.parallelSettings.strategy = ParallelStrategy::AUTO;
    if (isSingleThreadMode) {
        // 🔧 单线程模式：强制使用1线程
        config.parallelSettings.maxThreads = 1;
        config.parallelSettings.enableLoadBalancing = false;
        std::cout << "🔧 SpatialOpsConfig配置为单线程模式" << std::endl;
    } else {
        // 生产模式：限制最大线程数
        config.parallelSettings.maxThreads = std::min(std::thread::hardware_concurrency(), 32u);
        config.parallelSettings.enableLoadBalancing = true;
    }
    config.parallelSettings.minDataSizeForParallelism = 1000000;
    config.parallelSettings.chunkSize = 1024 * 1024;
    config.parallelSettings.loadBalanceThreshold = 0.8;
    
    // GDAL配置
    config.gdalSettings.gdalCacheMaxBytes = 512LL * 1024 * 1024; // 512MB
    config.gdalSettings.numThreads = isSingleThreadMode ? "1" : "ALL_CPUS";
    config.gdalSettings.enableGDALOptimizations = true;
    config.gdalSettings.blockCacheSize = 40000000;
    
    // 索引配置
    config.indexSettings.strategy = IndexStrategy::AUTO;
    config.indexSettings.indexThreshold = 1000;
    config.indexSettings.maxIndexDepth = 10;
    config.indexSettings.maxLeafCapacity = 16;
    config.indexSettings.indexBuildRatio = 0.7;
    config.indexSettings.enableIndexCaching = true;
    config.indexSettings.maxCachedIndices = 10;
    
    // 内存配置
    config.memorySettings.strategy = MemoryStrategy::AUTO;
    config.memorySettings.maxMemoryUsage = 0; // 无限制
    config.memorySettings.geometryPoolSize = 1000;
    config.memorySettings.rasterPoolSize = 100;
    config.memorySettings.enableMemoryPooling = false;
    config.memorySettings.enableMemoryMapping = true;
    config.memorySettings.memoryPressureThreshold = 0.85;
    
    // 算法配置
    config.algorithmSettings.geometricTolerance = 1e-9;
    config.algorithmSettings.simplificationTolerance = 1e-6;
    config.algorithmSettings.maxIterations = 1000;
    config.algorithmSettings.enableProgressiveRefinement = true;
    config.algorithmSettings.enableApproximateAlgorithms = false;
    config.algorithmSettings.defaultResamplingMethod = "bilinear";
    
    // 性能配置
    config.performanceSettings.enableSpatialIndex = true;
    config.performanceSettings.spatialIndexThreshold = 1000;
    config.performanceSettings.enableMemoryPooling = false;
    config.performanceSettings.enablePerformanceMonitoring = false;
    config.performanceSettings.enableMetricsCollection = false;
    config.performanceSettings.metricsBufferSize = 10000;
    config.performanceSettings.enableProfilingMode = false;
    
    // 日志配置
    config.loggingSettings.logLevel = "INFO";
    config.loggingSettings.logFormat = "default";
    config.loggingSettings.enableFileLogging = false;
    config.loggingSettings.maxLogFileSize = 100 * 1024 * 1024; // 100MB
    config.loggingSettings.maxLogFiles = 5;
    config.loggingSettings.enableAsyncLogging = true;
    
    // 验证配置
    config.validationSettings.enableInputValidation = true;
    config.validationSettings.enableGeometryValidation = true;
    config.validationSettings.enableCRSValidation = true;
    config.validationSettings.enableBoundsChecking = true;
    config.validationSettings.strictMode = false;
    config.validationSettings.validationTolerance = 1e-10;
    
    // 服务级别设置
    config.defaultCRS = "EPSG:4326";
    config.serviceName = "SpatialOpsService";
    config.version = "1.0.0";
    config.enableDebugMode = false;
    config.enableExperimentalFeatures = false;
    
    return config;
}

SpatialOpsConfig SpatialConfigManager::mergeConfigs(const SpatialOpsConfig& base, const SpatialOpsConfig& override) {
    SpatialOpsConfig merged = base;
    
    // 合并并行设置
    if (override.parallelSettings.strategy != ParallelStrategy::AUTO) {
        merged.parallelSettings.strategy = override.parallelSettings.strategy;
    }
    if (override.parallelSettings.maxThreads != 0) {
        merged.parallelSettings.maxThreads = override.parallelSettings.maxThreads;
    }
    
    // 合并GDAL设置
    if (override.gdalSettings.gdalCacheMaxBytes != 256LL * 1024 * 1024) {
        merged.gdalSettings.gdalCacheMaxBytes = override.gdalSettings.gdalCacheMaxBytes;
    }
    if (!override.gdalSettings.numThreads.empty() && override.gdalSettings.numThreads != "ALL_CPUS") {
        merged.gdalSettings.numThreads = override.gdalSettings.numThreads;
    }
    
    // 合并自定义选项
    for (const auto& [key, value] : override.gdalSettings.gdalOptions) {
        merged.gdalSettings.gdalOptions[key] = value;
    }
    
    // 合并其他设置...
    if (!override.defaultCRS.empty() && override.defaultCRS != "EPSG:4326") {
        merged.defaultCRS = override.defaultCRS;
    }
    
    return merged;
}

bool SpatialConfigManager::validateAndFix(SpatialOpsConfig& config) {
    bool isValid = true;
    
    // 验证并修复并行设置
    if (config.parallelSettings.maxThreads == 0) {
        config.parallelSettings.maxThreads = std::thread::hardware_concurrency();
    }
    if (config.parallelSettings.maxThreads > 1000) {
        config.parallelSettings.maxThreads = 1000;
        isValid = false;
    }
    
    // 验证并修复GDAL设置
    if (config.gdalSettings.gdalCacheMaxBytes < 0) {
        config.gdalSettings.gdalCacheMaxBytes = 256LL * 1024 * 1024;
        isValid = false;
    }
    
    // 验证并修复几何容差
    if (config.algorithmSettings.geometricTolerance <= 0) {
        config.algorithmSettings.geometricTolerance = 1e-9;
        isValid = false;
    }
    
    return config.validate() && isValid;
}

std::string SpatialConfigManager::configToString(const SpatialOpsConfig& config) const {
    return configToYaml(config);
}

SpatialOpsConfig SpatialConfigManager::configFromString(const std::string& configString) {
    SpatialOpsConfig config;
    if (!parseYamlConfig(configString, config)) {
        throw ConfigurationException("Failed to parse configuration string");
    }
    return config;
}

void SpatialConfigManager::setConfigTemplate(const std::string& templateName, const SpatialOpsConfig& config) {
    m_configTemplates[templateName] = config;
}

SpatialOpsConfig SpatialConfigManager::getConfigTemplate(const std::string& templateName) {
    auto it = m_configTemplates.find(templateName);
    if (it != m_configTemplates.end()) {
        return it->second;
    }
    return getDefaultConfig();
}

std::vector<std::string> SpatialConfigManager::listConfigTemplates() const {
    std::vector<std::string> templates;
    for (const auto& [name, config] : m_configTemplates) {
        templates.push_back(name);
    }
    return templates;
}

void SpatialConfigManager::initializeDefaultTemplates() {
    // 高性能模板
    auto highPerformanceConfig = getDefaultConfig();
    highPerformanceConfig.parallelSettings.strategy = ParallelStrategy::AUTO;
    highPerformanceConfig.gdalSettings.gdalCacheMaxBytes = 1024LL * 1024 * 1024; // 1GB
    highPerformanceConfig.performanceSettings.enablePerformanceMonitoring = true;
    highPerformanceConfig.algorithmSettings.enableApproximateAlgorithms = true;
    m_configTemplates["high_performance"] = highPerformanceConfig;
    
    // 低内存模板
    auto lowMemoryConfig = getDefaultConfig();
    lowMemoryConfig.memorySettings.strategy = MemoryStrategy::STANDARD;
    lowMemoryConfig.gdalSettings.gdalCacheMaxBytes = 64LL * 1024 * 1024; // 64MB
    lowMemoryConfig.memorySettings.enableMemoryPooling = true;
    lowMemoryConfig.parallelSettings.maxThreads = 2;
    m_configTemplates["low_memory"] = lowMemoryConfig;
    
    // 测试模板
    auto testConfig = getDefaultConfig();
    testConfig.parallelSettings.strategy = ParallelStrategy::NONE;
    testConfig.parallelSettings.maxThreads = 1;
    testConfig.performanceSettings.enableSpatialIndex = false;
    testConfig.loggingSettings.logLevel = "DEBUG";
    testConfig.validationSettings.strictMode = true;
    m_configTemplates["testing"] = testConfig;
}

bool SpatialConfigManager::parseYamlConfig(const std::string& yamlContent, SpatialOpsConfig& config) {
    // 简化的YAML解析实现
    // 在实际项目中，应该使用专业的YAML库如yaml-cpp
    
    std::istringstream stream(yamlContent);
    std::string line;
    
    while (std::getline(stream, line)) {
        // 移除注释
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        
        // 移除前后空白
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty()) continue;
        
        // 简单的键值对解析
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            std::string key = line.substr(0, colonPos);
            std::string value = line.substr(colonPos + 1);
            
            // 移除空白
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            // 应用配置值
            applyConfigValue(config, key, value);
        }
    }
    
    return true;
}

bool SpatialConfigManager::parseJsonConfig(const std::string& jsonContent, SpatialOpsConfig& config) {
    // 简化的JSON解析实现
    // 在实际项目中，应该使用专业的JSON库如nlohmann/json
    
    // 这里提供一个基本的实现框架
    // 实际实现需要完整的JSON解析器
    
    return parseYamlConfig(jsonContent, config); // 临时使用YAML解析器
}

std::string SpatialConfigManager::configToYaml(const SpatialOpsConfig& config) const {
    std::ostringstream yaml;
    
    yaml << "# Spatial Operations Service Configuration\n";
    yaml << "service_name: " << config.serviceName << "\n";
    yaml << "version: " << config.version << "\n";
    yaml << "default_crs: " << config.defaultCRS << "\n\n";
    
    yaml << "# Parallel Processing Configuration\n";
    yaml << "parallel:\n";
    yaml << "  strategy: " << static_cast<int>(config.parallelSettings.strategy) << "\n";
    yaml << "  max_threads: " << config.parallelSettings.maxThreads << "\n";
    yaml << "  min_data_size_for_parallelism: " << config.parallelSettings.minDataSizeForParallelism << "\n";
    yaml << "  chunk_size: " << config.parallelSettings.chunkSize << "\n";
    yaml << "  enable_load_balancing: " << (config.parallelSettings.enableLoadBalancing ? "true" : "false") << "\n\n";
    
    yaml << "# GDAL Configuration\n";
    yaml << "gdal:\n";
    yaml << "  cache_max_bytes: " << config.gdalSettings.gdalCacheMaxBytes << "\n";
    yaml << "  num_threads: " << config.gdalSettings.numThreads << "\n";
    yaml << "  enable_optimizations: " << (config.gdalSettings.enableGDALOptimizations ? "true" : "false") << "\n";
    yaml << "  block_cache_size: " << config.gdalSettings.blockCacheSize << "\n\n";
    
    yaml << "# Logging Configuration\n";
    yaml << "logging:\n";
    yaml << "  log_level: " << config.loggingSettings.logLevel << "\n";
    yaml << "  enable_file_logging: " << (config.loggingSettings.enableFileLogging ? "true" : "false") << "\n";
    yaml << "  max_log_file_size: " << config.loggingSettings.maxLogFileSize << "\n\n";
    
    return yaml.str();
}

std::string SpatialConfigManager::configToJson(const SpatialOpsConfig& config) const {
    std::ostringstream json;
    
    json << "{\n";
    json << "  \"service_name\": \"" << config.serviceName << "\",\n";
    json << "  \"version\": \"" << config.version << "\",\n";
    json << "  \"default_crs\": \"" << config.defaultCRS << "\",\n";
    json << "  \"parallel\": {\n";
    json << "    \"strategy\": " << static_cast<int>(config.parallelSettings.strategy) << ",\n";
    json << "    \"max_threads\": " << config.parallelSettings.maxThreads << ",\n";
    json << "    \"min_data_size_for_parallelism\": " << config.parallelSettings.minDataSizeForParallelism << "\n";
    json << "  },\n";
    json << "  \"gdal\": {\n";
    json << "    \"cache_max_bytes\": " << config.gdalSettings.gdalCacheMaxBytes << ",\n";
    json << "    \"num_threads\": \"" << config.gdalSettings.numThreads << "\",\n";
    json << "    \"enable_optimizations\": " << (config.gdalSettings.enableGDALOptimizations ? "true" : "false") << "\n";
    json << "  },\n";
    json << "  \"logging\": {\n";
    json << "    \"log_level\": \"" << config.loggingSettings.logLevel << "\",\n";
    json << "    \"enable_file_logging\": " << (config.loggingSettings.enableFileLogging ? "true" : "false") << "\n";
    json << "  }\n";
    json << "}\n";
    
    return json.str();
}

void SpatialConfigManager::applyEnvironmentOverrides(SpatialOpsConfig& config) {
    // 检查环境变量并应用覆盖
    
    if (const char* logLevel = std::getenv("SPATIAL_OPS_LOG_LEVEL")) {
        config.loggingSettings.logLevel = logLevel;
    }
    
    if (const char* maxThreads = std::getenv("SPATIAL_OPS_MAX_THREADS")) {
        try {
            config.parallelSettings.maxThreads = std::stoul(maxThreads);
        } catch (const std::exception&) {
            // 忽略无效值
        }
    }
    
    if (const char* gdalCache = std::getenv("SPATIAL_OPS_GDAL_CACHE_MB")) {
        try {
            config.gdalSettings.gdalCacheMaxBytes = std::stoll(gdalCache) * 1024 * 1024;
        } catch (const std::exception&) {
            // 忽略无效值
        }
    }
    
    if (const char* defaultCrs = std::getenv("SPATIAL_OPS_DEFAULT_CRS")) {
        config.defaultCRS = defaultCrs;
    }
    
    if (const char* enableDebug = std::getenv("SPATIAL_OPS_DEBUG")) {
        config.enableDebugMode = (std::string(enableDebug) == "true" || std::string(enableDebug) == "1");
    }
}

bool SpatialConfigManager::validateConfigSection(const SpatialOpsConfig& config, const std::string& section) {
    if (section == "parallel") {
        return config.parallelSettings.maxThreads > 0 && 
               config.parallelSettings.minDataSizeForParallelism > 0;
    } else if (section == "gdal") {
        return config.gdalSettings.gdalCacheMaxBytes >= 0 && 
               config.gdalSettings.blockCacheSize > 0;
    } else if (section == "algorithm") {
        return config.algorithmSettings.geometricTolerance > 0 && 
               config.algorithmSettings.maxIterations > 0;
    }
    
    return true;
}

void SpatialConfigManager::applyConfigValue(SpatialOpsConfig& config, const std::string& key, const std::string& value) {
    // 简化的配置值应用
    if (key == "log_level") {
        config.loggingSettings.logLevel = value;
    } else if (key == "max_threads") {
        try {
            config.parallelSettings.maxThreads = std::stoul(value);
        } catch (const std::exception&) {
            // 忽略无效值
        }
    } else if (key == "default_crs") {
        config.defaultCRS = value;
    } else if (key == "gdal_cache_mb") {
        try {
            config.gdalSettings.gdalCacheMaxBytes = std::stoll(value) * 1024 * 1024;
        } catch (const std::exception&) {
            // 忽略无效值
        }
    }
    // 可以添加更多配置项的处理
}

} // namespace oscean::core_services::spatial_ops 
