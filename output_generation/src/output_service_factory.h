#pragma once

#include <memory>
#include "core_services/output/i_output_service.h"

// 前向声明
namespace oscean::common_utils::infrastructure {
    class UnifiedThreadPoolManager;
}

namespace oscean::output {

/**
 * @class OutputServiceFactory
 * @brief 工厂类，用于创建OutputService实例
 */
class OutputServiceFactory {
public:
    /**
     * @brief 创建OutputService实例
     * @param threadPoolManager 线程池管理器，用于异步操作
     * @return 返回IOutputService接口的实现
     */
    static std::unique_ptr<oscean::core_services::output::IOutputService> createOutputService(
        std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> threadPoolManager
    );
    
    /**
     * @brief 创建默认配置的OutputService实例
     * @param threadPoolManager 线程池管理器
     * @return 返回IOutputService接口的实现
     */
    static std::unique_ptr<oscean::core_services::output::IOutputService> createDefaultOutputService(
        std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> threadPoolManager
    );
};

} // namespace oscean::output 