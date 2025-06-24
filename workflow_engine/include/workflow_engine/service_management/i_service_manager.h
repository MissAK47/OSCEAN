#pragma once

// 启用boost::asio支持（因为依赖的服务可能需要）
#define OSCEAN_ENABLE_BOOST_ASIO
#include "common_utils/utilities/boost_config.h"
OSCEAN_ENABLE_BOOST_ASIO_IN_MODULE();  // 服务管理器间接使用boost::asio（通过依赖的服务）

#include <memory>
#include <string>
#include <typeindex>
#include <functional>

// 引入统一异步框架
#include "common_utils/async/async_framework.h"
#include "common_utils/async/async_task.h"

// 引入所有核心服务接口
#include "core_services/crs/i_crs_service.h"
#include "core_services/data_access/i_raw_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "core_services/interpolation/i_interpolation_service.h"
#include "core_services/modeling/i_modeling_service.h"

namespace oscean::workflow_engine::service_management {

    /**
     * @brief 统一服务管理器接口
     * @details 负责所有核心服务的生命周期管理、懒加载和单例访问。
     *          这个管理器是线程安全的，并集成了统一异步框架。
     */
    class IServiceManager {
    public:
        virtual ~IServiceManager() = default;

        // === 🎯 统一异步任务管理接口（核心解决方案）===
        
        /**
         * @brief 获取统一异步框架实例
         * @return 异步框架引用
         */
        virtual oscean::common_utils::async::AsyncFramework& getAsyncFramework() = 0;
        
        /**
         * @brief 提交异步任务（主要接口）
         * @tparam Func 任务函数类型
         * @tparam Args 参数类型
         * @param taskName 任务名称
         * @param func 任务函数
         * @param args 任务参数
         * @return 异步任务包装器
         */
        template<typename Func, typename... Args>
        auto submitAsyncTask(const std::string& taskName, Func&& func, Args&&... args) 
            -> oscean::common_utils::async::AsyncTask<std::invoke_result_t<Func, Args...>> {
            return getAsyncFramework().submitTask(
                std::forward<Func>(func), 
                std::forward<Args>(args)...
            );
        }
        
        /**
         * @brief 等待所有异步任务完成（优雅关闭时使用）
         * @param timeoutSeconds 超时时间（秒）
         * @return 是否在超时前完成
         */
        virtual bool waitForAllAsyncTasks(size_t timeoutSeconds = 0) = 0;

        /**
         * @brief 获取指定类型的核心服务实例（线程安全）。
         * @details 如果服务尚未创建，此方法将触发其"懒加载"初始化。
         *          如果已创建，则返回缓存的单例实例。
         * @tparam ServiceInterface 服务的抽象接口类型，例如 ICrsEngine。
         * @return std::shared_ptr<ServiceInterface> 指向服务实例的共享指针。
         * @throw std::runtime_error 如果请求的服务未被注册或创建失败。
         */
        template<typename ServiceInterface>
        std::shared_ptr<ServiceInterface> getService() {
            // 使用 a C-style cast is not safe here, so we must use a static_pointer_cast
            // for safe downcasting of shared_ptr.
            return std::static_pointer_cast<ServiceInterface>(getServiceInternal(typeid(ServiceInterface)));
        }

    protected:
        // 内部实现方法，通过类型索引获取服务
        virtual std::shared_ptr<void> getServiceInternal(std::type_index serviceType) = 0;
    };

} // namespace oscean::workflow_engine::service_management 