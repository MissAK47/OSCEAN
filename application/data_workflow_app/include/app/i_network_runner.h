#pragma once

#include <string>
#include <memory>

// 前向声明，避免直接包含重量级头文件
namespace oscean::workflow_engine::service_management {
    class IServiceManager;
}

namespace oscean::application {

/**
 * @brief 网络服务的纯接口，无任何Boost.Asio依赖。
 *        这是应用层与网络服务层之间的防火墙。
 */
class INetworkRunner {
public:
    virtual ~INetworkRunner() = default;

    /**
     * @brief 启动网络服务（此为阻塞调用，会运行事件循环）。
     */
    virtual void run() = 0;

    /**
     * @brief 从另一个线程请求停止网络服务。
     */
    virtual void stop() = 0;
};

/**
 * @brief 创建网络服务运行实例的工厂函数。
 *        此函数的实现位于 network_service 库内部，从而隐藏所有实现细节。
 * @param address 监听地址
 * @param port 监听端口
 * @param num_threads 网络服务的线程数
 * @param service_manager 指向服务管理器的共享指针
 * @return 指向网络服务运行实例的唯一指针
 */
std::unique_ptr<INetworkRunner> create_network_runner(
    const std::string& address,
    unsigned short port,
    int num_threads,
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> service_manager
);

} // namespace oscean::application 