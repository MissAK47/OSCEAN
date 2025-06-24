/**
 * @file simd_config.h
 * @brief SIMD配置和环境定义
 * 
 * 提供SIMD模块的基础配置管理、环境检测和实现类型定义。
 * 支持Development/Testing/Production/HPC等环境的差异化配置。
 */

#pragma once

// 先定义需要boost::asio支持
#define OSCEAN_ENABLE_BOOST_ASIO

// 引用统一的boost配置 - 必须在最前面以避免WinSock冲突
#include "common_utils/utilities/boost_config.h"
OSCEAN_ENABLE_BOOST_ASIO_IN_MODULE()

#include <cstddef>
#include <string>
#include <vector>
#include <optional>

namespace oscean::common_utils::simd {

/**
 * @brief SIMD实现类型
 */
enum class SIMDImplementation {
    AUTO_DETECT,       // 自动检测最优实现
    SSE2,             // SSE2实现
    SSE4_1,           // SSE4.1实现
    AVX,              // AVX实现
    AVX2,             // AVX2实现
    AVX512,           // AVX512实现
    NEON,             // ARM NEON实现
    SCALAR            // 标量回退实现
};

/**
 * @brief SIMD环境类型 - 独立命名空间避免冲突
 */
namespace environment {
    enum class Type {
        DEVELOPMENT,      // 开发环境：调试友好，性能监控详细
        TESTING,          // 测试环境：稳定性优先，错误检测严格
        PRODUCTION,       // 生产环境：性能优先，错误处理快速
        HPC               // 高性能计算：极致性能，硬件专用优化
    };
}

// 为兼容性提供类型别名
using Environment = environment::Type;

/**
 * @brief 🔴 修复：SIMD能力标志位 - 移到SIMDConfig之前
 */
struct SIMDFeatures {
    bool hasSSE2 = false;
    bool hasSSE3 = false;
    bool hasSSE4_1 = false;
    bool hasSSE4_2 = false;
    bool hasAVX = false;
    bool hasAVX2 = false;
    bool hasAVX512F = false;
    bool hasAVX512VL = false;
    bool hasFMA = false;
    bool hasNEON = false;
    
    SIMDImplementation getOptimalImplementation() const;
    bool supports(SIMDImplementation impl) const;
    std::string toString() const;
};

/**
 * @brief SIMD配置参数
 */
struct SIMDConfig {
    Environment environment = Environment::PRODUCTION;
    SIMDImplementation preferredImplementation = SIMDImplementation::AUTO_DETECT;
    
    // === 🔴 修复：添加缺少的成员变量 ===
    SIMDImplementation implementation = SIMDImplementation::AUTO_DETECT;  // 当前使用的实现
    SIMDFeatures features;                                                // 支持的特性集合
    size_t batchSize = 8;                                                // 批处理大小
    size_t alignment = 32;                                               // 内存对齐大小
    
    // === 性能配置 ===
    size_t defaultAlignment = 32;           // 默认内存对齐大小
    size_t optimalBatchSize = 8;            // 最优批处理大小
    bool enableFallback = true;             // 启用标量回退
    bool enableVectorization = true;        // 启用向量化
    
    // === 调试配置 ===
    bool enableBoundsChecking = false;      // 启用边界检查
    bool enablePerformanceLogging = false;  // 启用性能日志
    bool enableErrorReporting = false;      // 启用错误报告
    
    // === 优化配置 ===
    bool enableCacheOptimization = true;    // 启用缓存优化
    bool enableParallelization = true;      // 启用并行化
    size_t maxThreads = 0;                  // 最大线程数(0=自动)
    
    // === 内存配置 ===
    size_t scratchBufferSize = 1024 * 1024; // 临时缓冲区大小(1MB)
    bool useMemoryPool = true;              // 使用内存池
    size_t memoryPoolSize = 8 * 1024 * 1024; // 内存池大小(8MB)
    
    std::string toString() const;
    static SIMDConfig createForEnvironment(Environment env);
    static SIMDConfig createOptimal();
    static SIMDConfig createForTesting();
};

/**
 * @brief SIMD系统信息
 */
struct SIMDSystemInfo {
    SIMDFeatures features;
    size_t cacheLineSize = 64;
    size_t l1CacheSize = 32768;        // 32KB
    size_t l2CacheSize = 262144;       // 256KB
    size_t l3CacheSize = 8388608;      // 8MB
    size_t physicalCores = 1;
    size_t logicalCores = 1;
    std::string cpuBrand;
    
    std::string toString() const;
    SIMDConfig getRecommendedConfig() const;
};

/**
 * @brief SIMD配置管理器 - 通过CommonServicesFactory创建
 * 
 * ⚠️ 重要变更：
 * - 不再使用单例模式
 * - 通过 CommonServicesFactory 内部创建和管理
 * - 支持依赖注入和多实例
 */
class SIMDConfigManager {
public:
    /**
     * @brief 构造函数
     * @param config 初始配置
     */
    explicit SIMDConfigManager(const SIMDConfig& config = SIMDConfig::createOptimal());
    
    /**
     * @brief 析构函数
     */
    ~SIMDConfigManager() = default;
    
    // 禁用拷贝，允许移动
    SIMDConfigManager(const SIMDConfigManager&) = delete;
    SIMDConfigManager& operator=(const SIMDConfigManager&) = delete;
    SIMDConfigManager(SIMDConfigManager&&) = default;
    SIMDConfigManager& operator=(SIMDConfigManager&&) = default;
    
    // === 配置管理 ===
    const SIMDConfig& getConfig() const { return config_; }
    void setConfig(const SIMDConfig& config) { config_ = config; }
    void updateConfig(const SIMDConfig& updates);
    
    // === 环境检测 ===
    const SIMDSystemInfo& getSystemInfo() const { return systemInfo_; }
    void refreshSystemInfo();
    
    // === 实现选择 ===
    SIMDImplementation selectOptimalImplementation() const;
    bool isImplementationSupported(SIMDImplementation impl) const;
    std::vector<SIMDImplementation> getSupportedImplementations() const;
    
    // === 验证 ===
    bool validateConfig(const SIMDConfig& config, std::string& errorMsg) const;
    std::vector<std::string> getConfigurationWarnings() const;

private:
    SIMDConfig config_;
    SIMDSystemInfo systemInfo_;
    
    void detectSystemCapabilities();
    void optimizeConfigForSystem();
};

/**
 * @brief SIMD配置工具函数
 */
namespace config_utils {
    
    /**
     * @brief 获取实现的显示名称
     */
    std::string getImplementationName(SIMDImplementation impl);
    
    /**
     * @brief 获取环境的显示名称
     */
    std::string getEnvironmentName(Environment env);
    
    /**
     * @brief 检查实现是否可用
     */
    bool isImplementationAvailable(SIMDImplementation impl);
    
    /**
     * @brief 获取推荐的对齐大小
     */
    size_t getRecommendedAlignment(SIMDImplementation impl);
    
    /**
     * @brief 获取推荐的批处理大小
     */
    size_t getRecommendedBatchSize(SIMDImplementation impl);
    
    /**
     * @brief 基准测试实现性能
     */
    double benchmarkImplementation(SIMDImplementation impl);
    
} // namespace config_utils

} // namespace oscean::common_utils::simd 