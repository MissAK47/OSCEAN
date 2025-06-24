#pragma once

// 🚀 使用Common模块的统一boost配置（参考CRS服务）
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 插值服务只使用boost::future，不使用boost::asio

#include "core_services/interpolation/i_interpolation_service.h"
#include <boost/smart_ptr/shared_ptr.hpp>
#include <memory>
#include <string>

namespace oscean {
namespace common_utils {
namespace simd {
    class ISIMDManager;
} // namespace simd
} // namespace common_utils

namespace core_services {
namespace interpolation {

/**
 * @struct InterpolationServiceConfig
 * @brief 插值服务配置参数
 */
struct InterpolationServiceConfig {
    bool enableSmartSelection = true;     ///< 启用智能算法选择
    bool enableSIMDOptimization = true;   ///< 启用SIMD优化
    size_t maxCacheSize = 1000;          ///< 最大缓存大小
    double performanceThreshold = 100.0;  ///< 性能阈值（毫秒）
};

/**
 * @class InterpolationServiceFactory
 * @brief 插值服务的工厂类，负责创建和配置插值服务实例。
 * @details 
 * 提供多种创建插值服务实例的方法，支持：
 * - 默认配置的插值服务
 * - 自定义SIMD管理器的插值服务
 * - 从配置文件加载的插值服务
 * 
 * 该工厂类确保创建的插值服务实例正确初始化，并根据系统能力
 * 自动选择最优的实现（如SIMD加速、GPU加速等）。
 */
class InterpolationServiceFactory {
public:
    /**
     * @brief 创建默认配置的插值服务实例
     * @return 插值服务的智能指针
     * @details 
     * 创建一个使用默认配置的插值服务实例：
     * - 自动检测并启用可用的SIMD指令集
     * - 启用智能算法选择
     * - 如果可用，启用GPU加速
     */
    static std::unique_ptr<IInterpolationService> createDefault();

    /**
     * @brief 创建使用指定SIMD管理器的插值服务实例
     * @param simdManager SIMD管理器的智能指针
     * @param enableSmartSelection 是否启用智能算法选择（默认true）
     * @return 插值服务的智能指针
     * @details 
     * 允许用户提供自定义的SIMD管理器，用于：
     * - 控制SIMD指令集的使用
     * - 共享SIMD资源管理
     * - 特定的性能优化策略
     */
    static std::unique_ptr<IInterpolationService> createWithSIMDManager(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager,
        bool enableSmartSelection = true);

    /**
     * @brief 从配置文件创建插值服务实例
     * @param configPath 配置文件路径
     * @param simdManager SIMD管理器的智能指针（可选）
     * @return 插值服务的智能指针
     * @details 
     * 从JSON或YAML配置文件加载插值服务配置，支持：
     * - 算法选择策略
     * - 性能参数调优
     * - GPU使用策略
     * - 缓存配置
     */
    static std::unique_ptr<IInterpolationService> createFromConfig(
        const std::string& configPath,
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr);

    /**
     * @brief 创建默认配置的插值服务实例
     * @return 插值服务的唯一指针
     */
    static std::unique_ptr<IInterpolationService> createService();

    /**
     * @brief 创建带配置的插值服务实例
     * @param config 服务配置参数
     * @return 插值服务的唯一指针
     */
    static std::unique_ptr<IInterpolationService> createService(
        const InterpolationServiceConfig& config);

    /**
     * @brief 创建高性能配置的插值服务实例
     * @param simdManager SIMD管理器共享指针
     * @return 高性能优化的插值服务实例
     */
    static std::unique_ptr<IInterpolationService> createHighPerformanceService(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager);

    /**
     * @brief 创建高精度配置的插值服务实例
     * @return 高精度优化的插值服务实例
     */
    static std::unique_ptr<IInterpolationService> createHighAccuracyService();

    /**
     * @brief 获取默认配置
     * @return 默认的服务配置
     */
    static InterpolationServiceConfig getDefaultConfig();

    /**
     * @brief 获取高性能配置
     * @return 高性能优化的服务配置
     */
    static InterpolationServiceConfig getHighPerformanceConfig();

    /**
     * @brief 获取高精度配置
     * @return 高精度优化的服务配置
     */
    static InterpolationServiceConfig getHighAccuracyConfig();

private:
    // 禁止实例化
    InterpolationServiceFactory() = delete;
    ~InterpolationServiceFactory() = delete;
    InterpolationServiceFactory(const InterpolationServiceFactory&) = delete;
    InterpolationServiceFactory& operator=(const InterpolationServiceFactory&) = delete;
};

} // namespace interpolation
} // namespace core_services
} // namespace oscean 