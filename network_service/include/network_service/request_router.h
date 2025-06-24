#pragma once

#include <memory>
#include <string>
#include <functional>
#include <boost/beast/http/verb.hpp>
#include <nlohmann/json.hpp>

// Forward declarations
namespace oscean::workflow_engine::service_management {
    class IServiceManager;
}

namespace oscean::network_service {

// DTO for request data
struct RequestDto {
    boost::beast::http::verb method;
    std::string target;
    std::string body;
};

// DTO for response data
struct ResponseDto {
    int status_code;
    std::string body;
    std::string content_type = "application/json";
};

using ResponseCallback = std::function<void(ResponseDto)>;

/**
 * @class RequestRouter
 * @brief Dispatches incoming HTTP requests to registered handlers.
 *
 * This class holds a collection of routes and invokes the appropriate handler
 * based on the request's method and target. It is the primary link between
 * the network transport layer and the application's business logic.
 */
class RequestRouter {
public:
    /**
     * @brief Constructs a RequestRouter.
     * @param serviceManager A shared pointer to the application's central service manager,
     *                       which will be used by handlers to access core services.
     */
    explicit RequestRouter(std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> service_manager);

    /**
     * @brief Finds the appropriate handler for a request and invokes it.
     * @param dto The parsed request data.
     * @param callback A callback to send the final ResponseData back to the connection.
     */
    void route_request(const RequestDto& dto, ResponseCallback callback);

private:
    std::shared_ptr<oscean::workflow_engine::service_management::IServiceManager> service_manager_;

    // Handlers
    void handle_status_request(ResponseCallback& callback);
    void handle_workflow_request(const nlohmann::json& request_body, ResponseCallback& callback);
};

} // namespace oscean::network_service 