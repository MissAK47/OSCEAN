#pragma once

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <memory>
#include <string>

// 前向声明，避免内部依赖
namespace oscean::network_service {
    class RequestRouter;
}

namespace oscean::common_utils::infrastructure {
    class UnifiedThreadPoolManager;
}

namespace oscean::network_service {

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

/**
 * @class Listener
 * @brief Listens for incoming HTTP connections and dispatches them to HttpConnection instances.
 *
 * This class binds to a specific address and port, and continuously accepts new connections.
 * Each accepted connection is wrapped in an HttpConnection object for processing.
 */
class Listener : public std::enable_shared_from_this<Listener> {
public:
    /**
     * @brief Constructs a Listener.
     * @param ioc The I/O context for asynchronous operations.
     * @param endpoint The TCP endpoint to bind to.
     * @param router A shared pointer to the request router.
     */
    Listener(
        asio::io_context& ioc,
        tcp::endpoint endpoint,
        std::shared_ptr<RequestRouter> router
    );

    /**
     * @brief Starts listening for connections.
     */
    void run();

private:
    void do_accept();
    void on_accept(boost::beast::error_code ec, tcp::socket socket);

    asio::io_context& ioc_;
    tcp::acceptor acceptor_;
    asio::strand<asio::any_io_executor> strand_;
    std::shared_ptr<RequestRouter> router_;
};

} // namespace oscean::network_service 