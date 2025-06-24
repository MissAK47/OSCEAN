#pragma once

#include <any>
#include <map>
#include <string>
#include <optional>  // 🔧 修复：添加C++17 std::optional支持

namespace ocean {
namespace workflow_engine {

/**
 * @struct WorkflowParameters
 * @brief A structure to hold the initial parameters for starting a workflow.
 *
 * It's essentially a type-safe wrapper around a map of string keys to `std::any` values,
 * allowing for flexible and varied parameter passing to different workflows.
 */
struct WorkflowParameters {
    std::map<std::string, std::any> parameters;

    template<typename T>
    std::optional<T> get(const std::string& key) const {
        auto it = parameters.find(key);
        if (it != parameters.end()) {
            try {
                return std::any_cast<T>(it->second);
            } catch (const std::bad_any_cast&) {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }
};

} // namespace workflow_engine
} // namespace ocean 