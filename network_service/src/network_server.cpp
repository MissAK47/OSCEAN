#include "network_service/network_server.h"
#include "network_service/listener.h"
#include "network_service/request_router.h"
#include <iostream>

namespace oscean::network_service {

using tcp = boost::asio::ip::tcp;
namespace http = boost::beast::http;

NetworkServer::NetworkServer(
    const std::string& address_str,
    unsigned short port,
    int num_threads,
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> service_manager)
    : ioc_(num_threads > 0 ? num_threads : 1),
      threads_(num_threads > 0 ? num_threads : 1),
      router_(std::make_shared<RequestRouter>(std::move(service_manager))),
      work_guard_(asio::make_work_guard(ioc_))
{
    auto const address = asio::ip::make_address(address_str);

    listener_ = std::make_shared<Listener>(ioc_, tcp::endpoint{address, port}, router_);
}

void NetworkServer::run() {
    if (!listener_) {
        std::cerr << "Error: Listener not initialized." << std::endl;
        return;
    }
    listener_->run();

    io_threads_.reserve(threads_ > 0 ? threads_ - 1 : 0);
    for (auto i = threads_ - 1; i > 0; --i) {
        io_threads_.emplace_back([this] { ioc_.run(); });
    }

    std::cout << "Server started with " << threads_ << " threads..." << std::endl;
    ioc_.run();

    for (auto& t : io_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    std::cout << "Server stopped." << std::endl;
}

void NetworkServer::stop() {
    work_guard_.reset();

    asio::post(ioc_, [this]() {
        if (listener_) {
            // listener_->stop(); // 假设 Listener 有一个 stop 方法来关闭 acceptor
        }
        
        if (!ioc_.stopped()) {
            ioc_.stop();
        }
    });
}

} // namespace oscean::network_service 