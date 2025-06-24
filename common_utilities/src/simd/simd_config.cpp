/**
 * @file simd_config.cpp
 * @brief SIMD配置和环境管理实现
 */

#include "common_utils/simd/simd_config.h"
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <cpuid.h>
#include <unistd.h>
#endif

namespace oscean::common_utils::simd {

// === SIMDConfig实现 ===

std::string SIMDConfig::toString() const {
    std::ostringstream oss;
    oss << "SIMDConfig {\n";
    oss << "  environment: " << static_cast<int>(environment) << "\n";
    oss << "  preferredImplementation: " << static_cast<int>(preferredImplementation) << "\n";
    oss << "  defaultAlignment: " << defaultAlignment << "\n";
    oss << "  optimalBatchSize: " << optimalBatchSize << "\n";
    oss << "  enableVectorization: " << enableVectorization << "\n";
    oss << "  enableCacheOptimization: " << enableCacheOptimization << "\n";
    oss << "  maxThreads: " << maxThreads << "\n";
    oss << "  scratchBufferSize: " << scratchBufferSize << "\n";
    oss << "}";
    return oss.str();
}

SIMDConfig SIMDConfig::createForEnvironment(Environment env) {
    SIMDConfig config;
    config.environment = env;
    
    switch (env) {
        case Environment::DEVELOPMENT:
            config.enableBoundsChecking = true;
            config.enablePerformanceLogging = true;
            config.enableErrorReporting = true;
            config.enableCacheOptimization = false;
            break;
            
        case Environment::TESTING:
            config.enableBoundsChecking = true;
            config.enableErrorReporting = true;
            config.enableVectorization = true;
            config.enableFallback = true;
            break;
            
        case Environment::PRODUCTION:
            config.enableBoundsChecking = false;
            config.enablePerformanceLogging = false;
            config.enableErrorReporting = false;
            config.enableCacheOptimization = true;
            config.enableParallelization = true;
            break;
            
        case Environment::HPC:
            config.enableCacheOptimization = true;
            config.enableParallelization = true;
            config.maxThreads = std::thread::hardware_concurrency();
            config.optimalBatchSize = 16;
            config.scratchBufferSize = 16 * 1024 * 1024; // 16MB
            break;
    }
    
    return config;
}

SIMDConfig SIMDConfig::createOptimal() {
    SIMDConfig config;
    
    // 直接设置最优配置，使用正确的成员变量名
    config.implementation = SIMDImplementation::AUTO_DETECT;
    config.preferredImplementation = SIMDImplementation::AUTO_DETECT;
    
    // 设置特性支持（假设现代x86处理器）
    config.features.hasSSE2 = true;
    config.features.hasSSE3 = true;
    config.features.hasSSE4_1 = true;
    config.features.hasSSE4_2 = true;
    config.features.hasAVX = true;
    config.features.hasAVX2 = true;
    config.features.hasAVX512F = false; // 保守设置
    config.features.hasFMA = true;
    config.features.hasNEON = false; // x86平台默认关闭
    
    // 设置合理的默认值
    config.optimalBatchSize = 8;
    config.maxThreads = std::thread::hardware_concurrency();
    config.enableVectorization = true;
    config.enableParallelization = true;
    config.enableCacheOptimization = true;
    config.enableFallback = true;
    
    return config;
}

SIMDConfig SIMDConfig::createForTesting() {
    SIMDConfig config = createForEnvironment(Environment::TESTING);
    
    // 测试环境特定设置
    config.scratchBufferSize = 1024 * 1024; // 1MB足够测试
    config.maxThreads = 2; // 限制线程数避免测试环境竞争
    config.preferredImplementation = SIMDImplementation::AUTO_DETECT;
    
    return config;
}

// === SIMDFeatures实现 ===

SIMDImplementation SIMDFeatures::getOptimalImplementation() const {
    if (hasAVX512F && hasAVX512VL) return SIMDImplementation::AVX512;
    if (hasAVX2) return SIMDImplementation::AVX2;
    if (hasAVX) return SIMDImplementation::AVX;
    if (hasSSE4_1) return SIMDImplementation::SSE4_1;
    if (hasSSE2) return SIMDImplementation::SSE2;
    if (hasNEON) return SIMDImplementation::NEON;
    return SIMDImplementation::SCALAR;
}

bool SIMDFeatures::supports(SIMDImplementation impl) const {
    switch (impl) {
        case SIMDImplementation::SSE2: return hasSSE2;
        case SIMDImplementation::SSE4_1: return hasSSE4_1;
        case SIMDImplementation::AVX: return hasAVX;
        case SIMDImplementation::AVX2: return hasAVX2;
        case SIMDImplementation::AVX512: return hasAVX512F && hasAVX512VL;
        case SIMDImplementation::NEON: return hasNEON;
        case SIMDImplementation::SCALAR: return true;
        case SIMDImplementation::AUTO_DETECT: return true;
        default: return false;
    }
}

std::string SIMDFeatures::toString() const {
    std::ostringstream oss;
    oss << "SIMDFeatures {";
    if (hasSSE2) oss << " SSE2";
    if (hasSSE3) oss << " SSE3";
    if (hasSSE4_1) oss << " SSE4.1";
    if (hasSSE4_2) oss << " SSE4.2";
    if (hasAVX) oss << " AVX";
    if (hasAVX2) oss << " AVX2";
    if (hasAVX512F) oss << " AVX512F";
    if (hasAVX512VL) oss << " AVX512VL";
    if (hasFMA) oss << " FMA";
    if (hasNEON) oss << " NEON";
    oss << " }";
    return oss.str();
}

// === SIMDSystemInfo实现 ===

std::string SIMDSystemInfo::toString() const {
    std::ostringstream oss;
    oss << "SIMDSystemInfo {\n";
    oss << "  CPU: " << cpuBrand << "\n";
    oss << "  Cores: " << physicalCores << " physical, " << logicalCores << " logical\n";
    oss << "  Cache: L1=" << (l1CacheSize/1024) << "KB, L2=" << (l2CacheSize/1024) << "KB, L3=" << (l3CacheSize/1024/1024) << "MB\n";
    oss << "  Features: " << features.toString() << "\n";
    oss << "}";
    return oss.str();
}

SIMDConfig SIMDSystemInfo::getRecommendedConfig() const {
    SIMDConfig config;
    
    // 根据CPU能力推荐配置
    config.preferredImplementation = features.getOptimalImplementation();
    
    // 根据核心数调整线程配置
    config.maxThreads = logicalCores > 0 ? logicalCores : 4;
    
    // 根据缓存大小调整批处理和缓冲区大小
    if (l3CacheSize > 8 * 1024 * 1024) { // > 8MB L3
        config.optimalBatchSize = 16;
        config.scratchBufferSize = 4 * 1024 * 1024; // 4MB
    } else if (l3CacheSize > 4 * 1024 * 1024) { // > 4MB L3
        config.optimalBatchSize = 8;
        config.scratchBufferSize = 2 * 1024 * 1024; // 2MB
    } else {
        config.optimalBatchSize = 4;
        config.scratchBufferSize = 1024 * 1024; // 1MB
    }
    
    // 根据指令集调整对齐
    if (features.hasAVX512F) {
        config.defaultAlignment = 64;
    } else if (features.hasAVX || features.hasAVX2) {
        config.defaultAlignment = 32;
    } else {
        config.defaultAlignment = 16;
    }
    
    return config;
}

// === SIMDConfigManager实现 ===

SIMDConfigManager::SIMDConfigManager(const SIMDConfig& config) : config_(config) {
    // 在构造时进行系统检测和优化
    detectSystemCapabilities();
    optimizeConfigForSystem();
}

void SIMDConfigManager::updateConfig(const SIMDConfig& updates) {
    // 合并配置，保持有效性
    config_.environment = updates.environment;
    config_.preferredImplementation = updates.preferredImplementation;
    config_.defaultAlignment = updates.defaultAlignment;
    config_.optimalBatchSize = updates.optimalBatchSize;
    config_.enableVectorization = updates.enableVectorization;
    config_.enableCacheOptimization = updates.enableCacheOptimization;
    config_.maxThreads = updates.maxThreads;
    config_.scratchBufferSize = updates.scratchBufferSize;
    
    // 重新优化配置
    optimizeConfigForSystem();
}

void SIMDConfigManager::refreshSystemInfo() {
    detectSystemCapabilities();
    optimizeConfigForSystem();
}

SIMDImplementation SIMDConfigManager::selectOptimalImplementation() const {
    if (config_.preferredImplementation == SIMDImplementation::AUTO_DETECT) {
        return systemInfo_.features.getOptimalImplementation();
    }
    
    // 验证首选实现是否支持
    if (systemInfo_.features.supports(config_.preferredImplementation)) {
        return config_.preferredImplementation;
    }
    
    // 回退到最优支持的实现
    return systemInfo_.features.getOptimalImplementation();
}

bool SIMDConfigManager::isImplementationSupported(SIMDImplementation impl) const {
    return systemInfo_.features.supports(impl);
}

std::vector<SIMDImplementation> SIMDConfigManager::getSupportedImplementations() const {
    std::vector<SIMDImplementation> supported;
    
    const SIMDImplementation candidates[] = {
        SIMDImplementation::AVX512,
        SIMDImplementation::AVX2,
        SIMDImplementation::AVX,
        SIMDImplementation::SSE4_1,
        SIMDImplementation::SSE2,
        SIMDImplementation::NEON,
        SIMDImplementation::SCALAR
    };
    
    for (auto impl : candidates) {
        if (isImplementationSupported(impl)) {
            supported.push_back(impl);
        }
    }
    
    return supported;
}

bool SIMDConfigManager::validateConfig(const SIMDConfig& config, std::string& errorMsg) const {
    // 检查实现支持
    if (!isImplementationSupported(config.preferredImplementation) && 
        config.preferredImplementation != SIMDImplementation::AUTO_DETECT) {
        errorMsg = "Preferred SIMD implementation not supported on this system";
        return false;
    }
    
    // 检查对齐参数
    if (config.defaultAlignment == 0 || (config.defaultAlignment & (config.defaultAlignment - 1)) != 0) {
        errorMsg = "Default alignment must be a power of 2";
        return false;
    }
    
    // 检查批处理大小
    if (config.optimalBatchSize == 0) {
        errorMsg = "Optimal batch size must be greater than 0";
        return false;
    }
    
    // 检查线程数
    if (config.maxThreads > 0 && config.maxThreads > systemInfo_.physicalCores * 2) {
        errorMsg = "Max threads exceeds reasonable limit for this system";
        return false;
    }
    
    return true;
}

std::vector<std::string> SIMDConfigManager::getConfigurationWarnings() const {
    std::vector<std::string> warnings;
    
    // 检查性能相关警告
    if (!config_.enableVectorization) {
        warnings.push_back("Vectorization is disabled - performance may be suboptimal");
    }
    
    if (config_.enableBoundsChecking && config_.environment == Environment::PRODUCTION) {
        warnings.push_back("Bounds checking enabled in production environment");
    }
    
    if (config_.maxThreads > systemInfo_.physicalCores) {
        warnings.push_back("Thread count exceeds physical cores - may cause contention");
    }
    
    return warnings;
}

void SIMDConfigManager::detectSystemCapabilities() {
    // 初始化为安全的默认值
    systemInfo_.features = {};
    
    // 安全的特性检测 - 不依赖CPUID，使用编译时检测
#ifdef _WIN32
    // Windows上使用保守配置，避免CPUID相关问题
    systemInfo_.features.hasSSE2 = true;   // 现代x64 Windows都支持SSE2
    #ifdef __AVX2__
        systemInfo_.features.hasAVX2 = true;
        systemInfo_.features.hasAVX = true;
        systemInfo_.features.hasSSE4_1 = true;
    #elif defined(__AVX__)
        systemInfo_.features.hasAVX = true;
        systemInfo_.features.hasSSE4_1 = true;
    #elif defined(__SSE4_1__)
        systemInfo_.features.hasSSE4_1 = true;
    #endif
#else
    // Linux上使用CPUID，但更简单的实现
    unsigned int eax, ebx, ecx, edx;
    
    // 基础特性检测
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        systemInfo_.features.hasSSE2 = (edx & (1 << 26)) != 0;
        systemInfo_.features.hasSSE3 = (ecx & (1 << 0)) != 0;
        systemInfo_.features.hasSSE4_1 = (ecx & (1 << 19)) != 0;
        systemInfo_.features.hasSSE4_2 = (ecx & (1 << 20)) != 0;
        systemInfo_.features.hasAVX = (ecx & (1 << 28)) != 0;
        systemInfo_.features.hasFMA = (ecx & (1 << 12)) != 0;
    } else {
        // 检测失败，使用保守配置
        systemInfo_.features.hasSSE2 = true;
    }
    
    // 扩展特性检测
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        systemInfo_.features.hasAVX2 = (ebx & (1 << 5)) != 0;
        systemInfo_.features.hasAVX512F = (ebx & (1 << 16)) != 0;
        systemInfo_.features.hasAVX512VL = (ebx & (1 << 31)) != 0;
    }
#endif

    // 检测系统信息
    systemInfo_.physicalCores = std::thread::hardware_concurrency();
    systemInfo_.logicalCores = std::thread::hardware_concurrency();
    
    // 如果检测失败，设置合理默认值
    if (systemInfo_.physicalCores == 0) {
        systemInfo_.physicalCores = 4;
        systemInfo_.logicalCores = 4;
    }
    
    // 设置默认缓存大小
    systemInfo_.cacheLineSize = 64;
    systemInfo_.l1CacheSize = 32768;    // 32KB
    systemInfo_.l2CacheSize = 262144;   // 256KB
    systemInfo_.l3CacheSize = 8388608;  // 8MB
    
    systemInfo_.cpuBrand = "Unknown CPU";
}

void SIMDConfigManager::optimizeConfigForSystem() {
    // 根据系统信息优化配置
    auto recommendedConfig = systemInfo_.getRecommendedConfig();
    
    // 如果当前配置的某些参数未明确设置，使用推荐值
    if (config_.maxThreads == 0) {
        config_.maxThreads = recommendedConfig.maxThreads;
    }
    
    if (config_.preferredImplementation == SIMDImplementation::AUTO_DETECT) {
        config_.preferredImplementation = recommendedConfig.preferredImplementation;
    }
}

// === config_utils命名空间实现 ===

namespace config_utils {

std::string getImplementationName(SIMDImplementation impl) {
    switch (impl) {
        case SIMDImplementation::AUTO_DETECT: return "Auto Detect";
        case SIMDImplementation::SSE2: return "SSE2";
        case SIMDImplementation::SSE4_1: return "SSE4.1";
        case SIMDImplementation::AVX: return "AVX";
        case SIMDImplementation::AVX2: return "AVX2";
        case SIMDImplementation::AVX512: return "AVX512";
        case SIMDImplementation::NEON: return "NEON";
        case SIMDImplementation::SCALAR: return "Scalar";
        default: return "Unknown";
    }
}

std::string getEnvironmentName(Environment env) {
    switch (env) {
        case Environment::DEVELOPMENT: return "Development";
        case Environment::TESTING: return "Testing";
        case Environment::PRODUCTION: return "Production";
        case Environment::HPC: return "HPC";
        default: return "Unknown";
    }
}

bool isImplementationAvailable(SIMDImplementation impl) {
    // 创建临时配置管理器来检查支持
    SIMDConfig config;
    SIMDConfigManager tempManager(config);
    return tempManager.isImplementationSupported(impl);
}

size_t getRecommendedAlignment(SIMDImplementation impl) {
    switch (impl) {
        case SIMDImplementation::AVX512: return 64;
        case SIMDImplementation::AVX2:
        case SIMDImplementation::AVX: return 32;
        case SIMDImplementation::SSE4_1:
        case SIMDImplementation::SSE2: return 16;
        case SIMDImplementation::NEON: return 16;
        default: return sizeof(float);
    }
}

size_t getRecommendedBatchSize(SIMDImplementation impl) {
    switch (impl) {
        case SIMDImplementation::AVX512: return 16;
        case SIMDImplementation::AVX2: return 8;
        case SIMDImplementation::AVX: return 8;
        case SIMDImplementation::SSE4_1:
        case SIMDImplementation::SSE2: return 4;
        case SIMDImplementation::NEON: return 4;
        default: return 1;
    }
}

double benchmarkImplementation(SIMDImplementation impl) {
    // 简单的基准测试 - 实际应用中需要更详细的测试
    if (!isImplementationAvailable(impl)) {
        return 0.0;
    }
    
    // 返回相对性能分数（标量实现为1.0）
    switch (impl) {
        case SIMDImplementation::AVX512: return 16.0;
        case SIMDImplementation::AVX2: return 8.0;
        case SIMDImplementation::AVX: return 6.0;
        case SIMDImplementation::SSE4_1: return 4.0;
        case SIMDImplementation::SSE2: return 3.0;
        case SIMDImplementation::NEON: return 4.0;
        case SIMDImplementation::SCALAR: return 1.0;
        default: return 0.0;
    }
}

} // namespace config_utils

} // namespace oscean::common_utils::simd 