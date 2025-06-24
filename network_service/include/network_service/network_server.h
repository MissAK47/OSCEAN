#pragma once

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <memory>
#include <string>
#include <vector>
#include <thread>

#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "workflow_engine/service_management/i_service_manager.h"

namespace oscean::network_service {

namespace asio = boost::asio;
using tcp = asio::ip::tcp;
namespace beast = boost::beast;

class Listener;
class RequestRouter;

/**
 * @class NetworkServer
 * @brief The main class for the network service.
 *
 * This class orchestrates the entire network service. It initializes the
 * I/O context, thread pools, and the listener, and manages their lifetimes.
 */
class NetworkServer {
public:
    /**
     * @brief Constructs the NetworkServer.
     * @param address The IP address to listen on.
     * @param port The port to listen on.
     * @param num_threads The number of I/O threads to run.
     * @param service_manager The central service manager for the application.
     */
    NetworkServer(
        const std::string& address_str,
        unsigned short port,
        int num_threads,
        std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> service_manager
    );

    /**
     * @brief Starts the server and blocks until it is stopped.
     */
    void run();

    /**
     * @brief Stops the server gracefully.
     */
    void stop();

private:
    asio::io_context ioc_;
    int threads_;
    std::shared_ptr<Listener> listener_;
    std::shared_ptr<RequestRouter> router_;
    std::vector<std::thread> io_threads_;

    // 添加 work_guard 来确保 io_context 不会提前退出
    using work_guard_type = asio::executor_work_guard<asio::io_context::executor_type>;
    work_guard_type work_guard_;
};

} // namespace oscean::network_service 