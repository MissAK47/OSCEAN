#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <memory>
#include <optional>

// Forward declare dependencies to break include cycles
namespace oscean::common_utils::infrastructure {
    class UnifiedThreadPoolManager;
}

namespace oscean::network_service {

// Forward declaration is crucial here. The full definition will be in the .cpp file.
class RequestRouter;

namespace beast = boost::beast;
namespace http = beast::http;
namespace asio = boost::asio;
using tcp = asio::ip::tcp;

class HttpConnection : public std::enable_shared_from_this<HttpConnection> {
public:
    HttpConnection(
        tcp::socket&& socket,
        std::shared_ptr<RequestRouter> router
    );

    // Start the asynchronous operation
    void start();

private:
    void do_read();
    void on_read(beast::error_code ec, std::size_t bytes_transferred);
    void process_request();
    void on_write(bool keep_alive, beast::error_code ec, std::size_t bytes_transferred);
    void do_close();

    beast::tcp_stream stream_;
    beast::flat_buffer buffer_;
    
    // The parser is stored in an optional because we need to construct it
    // for each new request.
    std::optional<http::request_parser<http::string_body>> parser_;
    
    std::optional<http::response<http::string_body>> response_;

    std::shared_ptr<RequestRouter> router_;
};

} // namespace oscean::network_service 