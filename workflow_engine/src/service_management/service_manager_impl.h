#pragma once

// 启用boost::asio支持（因为依赖的线程池管理器需要）
#define OSCEAN_ENABLE_BOOST_ASIO
#include "common_utils/utilities/boost_config.h"
OSCEAN_ENABLE_BOOST_ASIO_IN_MODULE();  // 服务管理器间接使用boost::asio（通过线程池管理器）

#include "workflow_engine/service_management/i_service_manager.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "common_utils/async/async_framework.h"
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <typeindex>
#include <string>


namespace oscean::workflow_engine::service_management {

    /**
     * @brief IServiceManager的线程安全、懒加载实现。
     */
    class ServiceManagerImpl : public IServiceManager, 
                              public std::enable_shared_from_this<ServiceManagerImpl> {
    public:
        /**
         * @brief 构造函数
         * @param threadPoolManager 统一线程池管理器
         */
        explicit ServiceManagerImpl(
            std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPoolManager
        );

        virtual ~ServiceManagerImpl() = default;
        
        // === 🎯 实现统一异步框架接口 ===
        
        /**
         * @brief 获取统一异步框架实例
         * @return 异步框架引用
         */
        oscean::common_utils::async::AsyncFramework& getAsyncFramework() override;
        
        /**
         * @brief 等待所有异步任务完成
         * @param timeoutSeconds 超时时间（秒）
         * @return 是否在超时前完成
         */
        bool waitForAllAsyncTasks(size_t timeoutSeconds = 0) override;

        /**
         * @brief 注册外部服务工厂函数
         * @tparam ServiceInterface 服务接口类型
         * @param factory 服务工厂函数
         */
        template<typename ServiceInterface>
        void registerServiceFactory(std::function<std::shared_ptr<ServiceInterface>()> factory) {
            std::lock_guard<std::mutex> lock(mutex_);
            serviceFactories_[typeid(ServiceInterface)] = [factory]() -> std::shared_ptr<void> {
                return std::static_pointer_cast<void>(factory());
            };
        }

    protected:
        // 实现基类中的纯虚函数
        std::shared_ptr<void> getServiceInternal(std::type_index serviceType) override;

    private:
        // 注册所有核心服务的工厂函数
        void registerServiceFactories();

        // 内部无锁版本，用于避免递归死锁
        std::shared_ptr<void> getServiceInternalNoLock(std::type_index serviceType);
        
        // 模板版本的无锁获取服务方法
        template<typename ServiceInterface>
        std::shared_ptr<ServiceInterface> getServiceNoLock();
        
        // 🎯 异步框架初始化方法
        void initializeAsyncFramework();

        // 核心依赖
        std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPoolManager_;
        // 🎯 统一异步框架实例
        std::unique_ptr<oscean::common_utils::async::AsyncFramework> asyncFramework_;

        // 用于保护服务创建和访问的互斥锁
        std::mutex mutex_;
        
        // 缓存已创建的服务实例
        std::unordered_map<std::type_index, std::shared_ptr<void>> services_;

        // 存储用于创建服务的工厂函数
        using ServiceFactory = std::function<std::shared_ptr<void>()>;
        std::unordered_map<std::type_index, ServiceFactory> serviceFactories_;
    };

} // namespace oscean::workflow_engine::service_management 