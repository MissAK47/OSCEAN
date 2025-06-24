#include "app/i_network_runner.h"
#include "network_service/network_server.h"
#include <memory>

namespace oscean::application {

// 这是对INetworkRunner接口的具体实现，它包装了真正的NetworkServer
class NetworkRunnerImpl : public INetworkRunner {
public:
    NetworkRunnerImpl(
        const std::string& address,
        unsigned short port,
        int num_threads,
        std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> service_manager)
    {
        // 真正的服务器对象在这里被创建，所有复杂的依赖都被封装在这个模块里
        server_ = std::make_unique<oscean::network_service::NetworkServer>(
            address,
            port,
            num_threads,
            service_manager
        );
    }

    // 实现run方法，直接调用底层服务器的run
    void run() override {
        if (server_) {
            server_->run();
        }
    }

    // 实现stop方法，直接调用底层服务器的stop
    void stop() override {
        if (server_) {
            server_->stop();
        }
    }

private:
    // 持有真正的NetworkServer实例
    std::unique_ptr<oscean::network_service::NetworkServer> server_;
};

// 实现工厂函数，返回接口的实现实例
std::unique_ptr<INetworkRunner> create_network_runner(
    const std::string& address,
    unsigned short port,
    int num_threads,
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> service_manager)
{
    return std::make_unique<NetworkRunnerImpl>(address, port, num_threads, service_manager);
}

} // namespace oscean::application 