#pragma once
#include "core_services/interpolation/i_interpolation_service.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "common_utils/cache/multi_level_cache_manager.h"
#include <memory>
#include <unordered_map>
#include <boost/asio/thread_pool.hpp>

namespace oscean::core_services::interpolation {

// 前向声明
class IInterpolationAlgorithm;
class PrecomputedDataCache;

/**
 * @brief 插值服务实现类
 * 
 * 基于现有的线程池和缓存系统，提供高性能的插值服务
 */
class InterpolationServiceImpl : public IInterpolationService {
public:
    /**
     * @brief 构造函数
     * @param threadPool 共享线程池
     * @param cacheManager 共享缓存管理器
     */
    InterpolationServiceImpl(
        std::shared_ptr<boost::asio::thread_pool> threadPool,
        std::shared_ptr<oscean::common_utils::cache::MultiLevelCacheManager> cacheManager);

    ~InterpolationServiceImpl() override = default;

    // IInterpolationService接口实现
    std::future<InterpolationResult> interpolateAsync(
        const InterpolationRequest& request) override;

    std::vector<InterpolationMethod> getSupportedMethods() const override;

private:
    // 核心组件
    std::shared_ptr<boost::asio::thread_pool> threadPool_;
    std::shared_ptr<oscean::common_utils::cache::MultiLevelCacheManager> cacheManager_;
    std::unique_ptr<PrecomputedDataCache> interpolationCache_;

    // 算法注册表
    std::unordered_map<InterpolationMethod, std::unique_ptr<IInterpolationAlgorithm>> algorithms_;

    /**
     * @brief 注册所有支持的插值算法
     */
    void registerAlgorithms();

    /**
     * @brief 执行插值的核心逻辑
     * @param request 插值请求
     * @return 插值结果
     */
    InterpolationResult executeInterpolation(const InterpolationRequest& request);

    /**
     * @brief 生成缓存键
     * @param request 插值请求
     * @return 缓存键字符串
     */
    std::string generateCacheKey(const InterpolationRequest& request) const;

    /**
     * @brief 验证请求的有效性
     * @param request 插值请求
     * @return 验证结果
     */
    bool validateRequest(const InterpolationRequest& request) const;

    /**
     * @brief 验证算法参数
     * @param method 插值方法
     * @param params 算法参数
     * @return 验证结果
     */
    bool validateAlgorithmParameters(
        InterpolationMethod method, 
        const AlgorithmParameters& params) const;
};

} // namespace oscean::core_services::interpolation 