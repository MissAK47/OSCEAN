#pragma once

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <boost/beast/http/verb.hpp>
#include <boost/beast/http/status.hpp>
#include <nlohmann/json.hpp>

namespace oscean::network_service {

/**
 * @brief Represents a parsed client request for consumption by internal services.
 */
struct RequestDTO {
    boost::beast::http::verb method;
    std::string target; // URL path + query string
    unsigned http_version;
    std::map<std::string, std::string> headers;
    std::string body; // Raw body, can be parsed to JSON if needed by the handler
};

/**
 * @brief Represents data needed to build an HTTP response.
 *        This struct is created by the business logic layer and consumed by the network layer.
 */
struct ResponseData {
    boost::beast::http::status status = boost::beast::http::status::ok;
    std::map<std::string, std::string> headers;
    // The body can be a string (e.g., JSON) or a binary blob (e.g., an image).
    std::variant<std::string, std::vector<unsigned char>> body;

    /**
     * @brief A helper function to create a JSON response.
     * @param s The HTTP status code.
     * @param j The nlohmann::json object to be sent.
     * @return A fully formed ResponseData object.
     */
    static ResponseData create_json_response(
        boost::beast::http::status s,
        const nlohmann::json& j
    ) {
        ResponseData resp;
        resp.status = s;
        resp.headers["Content-Type"] = "application/json";
        resp.body = j.dump();
        return resp;
    }

    /**
     * @brief A helper function to create a standard error response in JSON format.
     * @param s The HTTP status code.
     * @param msg The error message.
     * @return A fully formed ResponseData object.
     */
    static ResponseData create_error_response(
        boost::beast::http::status s,
        const std::string& msg
    ) {
        nlohmann::json j;
        j["error"] = msg;
        return create_json_response(s, j);
    }
};

} // namespace oscean::network_service 