#include "network_service/listener.h"
#include "network_service/http_connection.h"
#include <iostream>

namespace oscean::network_service {

Listener::Listener(
    asio::io_context& ioc,
    tcp::endpoint endpoint,
    std::shared_ptr<RequestRouter> router)
    : ioc_(ioc),
      acceptor_(ioc),
      strand_(asio::make_strand(ioc)),
      router_(std::move(router))
{
    beast::error_code ec;

    // Open the acceptor
    acceptor_.open(endpoint.protocol(), ec);
    if(ec) {
        std::cerr << "open: " << ec.message() << std::endl;
        return;
    }

    // Allow address reuse
    acceptor_.set_option(asio::socket_base::reuse_address(true), ec);
    if(ec) {
        std::cerr << "set_option: " << ec.message() << std::endl;
        return;
    }

    // Bind to the server address
    acceptor_.bind(endpoint, ec);
    if(ec) {
        std::cerr << "bind: " << ec.message() << std::endl;
        return;
    }

    // Start listening for connections
    acceptor_.listen(asio::socket_base::max_listen_connections, ec);
    if(ec) {
        std::cerr << "listen: " << ec.message() << std::endl;
        return;
    }
}

void Listener::run() {
    std::cout << "[Listener] Starting to accept connections..." << std::endl;
    do_accept();
}

void Listener::do_accept() {
    std::cout << "[Listener] Posting a new async_accept operation." << std::endl;
    // The new connection gets its own strand
    acceptor_.async_accept(
        strand_,
        beast::bind_front_handler(
            &Listener::on_accept,
            shared_from_this()));
}

void Listener::on_accept(beast::error_code ec, tcp::socket socket) {
    if(ec) {
        std::cerr << "[Listener] accept error: " << ec.message() << std::endl;
        return; // To avoid infinite loop
    }
    else {
        std::cout << "[Listener] Successfully accepted a new connection." << std::endl;
        // Create the HttpConnection and transfer ownership of the socket
        std::make_shared<HttpConnection>(
            std::move(socket),
            router_)->start();
    }

    // Accept another connection
    do_accept();
}

} // namespace oscean::network_service 