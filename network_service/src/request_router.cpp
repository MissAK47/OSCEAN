#include "network_service/request_router.h"
#include "workflow_engine/service_management/i_service_manager.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <ctime>

namespace oscean::network_service {

RequestRouter::RequestRouter(std::shared_ptr<workflow_engine::service_management::IServiceManager> service_manager)
    : service_manager_(std::move(service_manager)) {
    std::cout << "[RequestRouter] Router initialized." << std::endl;
}

void RequestRouter::route_request(const RequestDto& dto, ResponseCallback callback) {
    std::cout << "[RequestRouter] Routing request: " << static_cast<int>(dto.method) << " " << dto.target << std::endl;
    std::cout << "[RequestRouter] Request body: " << dto.body << std::endl;
    
    try {
        if (dto.target == "/status") {
            std::cout << "[RequestRouter] Handling status request." << std::endl;
            handle_status_request(callback);
        } else if (dto.target == "/workflow") {
            std::cout << "[RequestRouter] Handling workflow request." << std::endl;
            if (dto.method != boost::beast::http::verb::post) {
                std::cout << "[RequestRouter] Method not allowed for workflow endpoint." << std::endl;
                 callback({405, "{\"error\":\"Method Not Allowed\"}"});
                 return;
            }
            if (dto.body.empty()) {
                std::cout << "[RequestRouter] Empty request body for workflow endpoint." << std::endl;
                callback({400, "{\"error\":\"Request body is empty\"}"});
                return;
            }
            std::cout << "[RequestRouter] Parsing JSON body..." << std::endl;
            handle_workflow_request(nlohmann::json::parse(dto.body), callback);
        } else {
            std::cout << "[RequestRouter] No handler found for target: " << dto.target << std::endl;
            callback({404, "{\"error\":\"Not Found\"}"});
        }
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "[RequestRouter] JSON parse error: " << e.what() << std::endl;
        callback({400, "{\"error\":\"Invalid JSON format\"}"});
    } catch (const std::exception& e) {
        std::cerr << "[RequestRouter] Exception in route_request: " << e.what() << std::endl;
        callback({500, std::string("{\"error\":\"Internal Server Error: ") + e.what() + "\"}"});
    }
}

void RequestRouter::handle_status_request(ResponseCallback& callback) {
    std::cout << "[RequestRouter] Processing status request..." << std::endl;
    nlohmann::json status_response;
    status_response["status"] = "OK";
    status_response["service_name"] = "OSCEAN Network Service";
    status_response["timestamp"] = std::time(nullptr);
    std::cout << "[RequestRouter] Status response prepared. Calling callback..." << std::endl;
    callback({200, status_response.dump()});
}

void RequestRouter::handle_workflow_request(const nlohmann::json& request_body, ResponseCallback& callback) {
    std::cout << "[RequestRouter] Processing workflow request..." << std::endl;
    // 实际的工作流调用逻辑在这里
    // 目前，我们只返回一个确认信息
    nlohmann::json workflow_response;
    workflow_response["message"] = "Workflow request received and is being processed.";
    workflow_response["request_payload"] = request_body;
    std::cout << "[RequestRouter] Workflow response prepared. Calling callback..." << std::endl;
    callback({202, workflow_response.dump()});
}

} // namespace oscean::network_service 