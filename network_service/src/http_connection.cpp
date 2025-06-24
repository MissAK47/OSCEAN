#include "network_service/http_connection.h"
#include "network_service/request_router.h"
#include <boost/beast/version.hpp>
#include <iostream>
#include <utility>

namespace oscean::network_service {

HttpConnection::HttpConnection(
    tcp::socket&& socket,
    std::shared_ptr<RequestRouter> router)
    : stream_(std::move(socket)),
      router_(std::move(router))
{
    std::cout << "[HttpConnection] New connection created." << std::endl;
}

void HttpConnection::start() {
    std::cout << "[HttpConnection] Starting connection processing." << std::endl;
    asio::dispatch(stream_.get_executor(),
        beast::bind_front_handler(&HttpConnection::do_read, shared_from_this()));
}

void HttpConnection::do_read() {
    std::cout << "[HttpConnection] Starting to read request." << std::endl;
    parser_.emplace();
    parser_->body_limit(10000);
    
    http::async_read(stream_, buffer_, *parser_,
        beast::bind_front_handler(&HttpConnection::on_read, shared_from_this()));
}

void HttpConnection::on_read(beast::error_code ec, std::size_t bytes_transferred) {
    boost::ignore_unused(bytes_transferred);
    
    std::cout << "[HttpConnection] Read completed. Bytes: " << bytes_transferred << std::endl;
    
    if (ec == http::error::end_of_stream) {
        std::cout << "[HttpConnection] End of stream detected." << std::endl;
        return do_close();
    }
    
    if (ec) {
        std::cerr << "[HttpConnection] Read error: " << ec.message() << std::endl;
        return;
    }
    
    std::cout << "[HttpConnection] Processing request..." << std::endl;
    process_request();
}

void HttpConnection::process_request() {
    try {
        auto& req = parser_->get();
        std::cout << "[HttpConnection] Request method: " << req.method_string() << ", target: " << req.target() << std::endl;
        std::cout << "[HttpConnection] Request body size: " << req.body().size() << std::endl;
        
        RequestDto dto { req.method(), std::string(req.target()), std::string(req.body()) };

        auto self = shared_from_this();
        auto response_callback = [self](ResponseDto response_dto) {
            std::cout << "[HttpConnection] Response callback invoked. Status: " << response_dto.status_code << std::endl;
            try {
                auto& req_cb = self->parser_->get();
                http::response<http::string_body> res{
                    static_cast<http::status>(response_dto.status_code), req_cb.version()
                };
                res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
                res.set(http::field::content_type, response_dto.content_type);
                res.keep_alive(req_cb.keep_alive());
                res.body() = std::move(response_dto.body);
                res.prepare_payload();
                
                std::cout << "[HttpConnection] Response prepared. Body size: " << res.body().size() << std::endl;
                
                self->response_ = std::move(res);

                std::cout << "[HttpConnection] Starting async write..." << std::endl;
                http::async_write(self->stream_, *self->response_,
                    beast::bind_front_handler(
                        &HttpConnection::on_write,
                        self,
                        self->response_->keep_alive()));
            } catch (const std::exception& e) {
                std::cerr << "[HttpConnection] Exception in response callback: " << e.what() << std::endl;
            }
        };

        std::cout << "[HttpConnection] Routing request to router..." << std::endl;
        router_->route_request(dto, response_callback);
    } catch (const std::exception& e) {
        std::cerr << "[HttpConnection] Exception in process_request: " << e.what() << std::endl;
    }
}

void HttpConnection::on_write(bool keep_alive, beast::error_code ec, std::size_t bytes_transferred) {
    boost::ignore_unused(bytes_transferred);
    
    std::cout << "[HttpConnection] Write completed. Bytes: " << bytes_transferred << std::endl;
    
    if (ec) {
        std::cerr << "[HttpConnection] Write error: " << ec.message() << std::endl;
        return;
    }
    
    std::cout << "[HttpConnection] Write successful. Keep alive: " << keep_alive << std::endl;
    
    if (!keep_alive) {
        std::cout << "[HttpConnection] Closing connection (keep_alive = false)." << std::endl;
        return do_close();
    }
    
    // Reset for next request
    response_.reset();
    parser_.reset();
    std::cout << "[HttpConnection] Ready for next request." << std::endl;
    do_read();
}

void HttpConnection::do_close() {
    std::cout << "[HttpConnection] Closing connection." << std::endl;
    beast::error_code ec;
    stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
    if (ec) {
        std::cerr << "[HttpConnection] Close error: " << ec.message() << std::endl;
    }
}

} // namespace oscean::network_service