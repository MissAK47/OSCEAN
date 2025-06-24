#pragma once

#include <string>
#include <memory>
#include <map>
#include "core_services/output/i_output_service.h" // For request/result structs
#include "boost/thread/future.hpp"
#include <nlohmann/json.hpp> // 直接包含而不使用forward declaration

namespace oscean {
namespace common_utils { namespace cache {
template <typename Key, typename Value>
class ICacheManager;
}}
namespace common_utils{ namespace infrastructure {
class UnifiedThreadPoolManager;
}}

namespace output {

/**
 * @class OutputProfileManager
 * @brief Manages loading, caching, and interpreting output profile JSON files.
 *
 * This class is responsible for the "brains" of the dynamic output system.
 * It reads declarative JSON profiles from the configuration directory,
 * caches them, and uses them to translate a simple, high-level profiled
 * request into a detailed, low-level request ready for an engine to execute.
 */
class OutputProfileManager {
public:
    /**
     * @brief Constructs the manager.
     * @param profileDirectory The absolute path to the directory containing *.json profiles.
     * @param cache A cache instance to store parsed JSON profiles, improving performance.
     * @param threadPool A thread pool for performing file I/O asynchronously.
     */
    OutputProfileManager(
        std::string profileDirectory,
        std::shared_ptr<common_utils::cache::ICacheManager<std::string, nlohmann::json>> cache,
        std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPool
    );

    virtual ~OutputProfileManager() = default;

    /**
     * @brief Resolves a high-level profiled request into a low-level, detailed request.
     * @param profiledRequest The high-level request containing the profile name and variables.
     * @return A future that will contain the fully resolved OutputRequest.
     *
     * This is the primary method of the class. It performs these steps asynchronously:
     * 1. Loads the specified JSON profile from the cache or file system.
     * 2. Validates the profile's structure.
     * 3. Substitutes template variables (e.g., {{jobId}}) in the profile.
     * 4. Constructs and returns a complete OutputRequest object.
     */
    virtual boost::future<core_services::output::OutputRequest> resolveRequest(
        const core_services::output::ProfiledRequest& profiledRequest);

private:
    // Helper method to load a profile from disk
    nlohmann::json loadProfileFromFile(const std::string& profileName);

    // Helper method to substitute variables in the resolved json
    void substituteVariables(nlohmann::json& profileJson, const std::map<std::string, std::string>& variables);

    std::string m_profileDirectory;
    std::shared_ptr<common_utils::cache::ICacheManager<std::string, nlohmann::json>> m_cache;
    std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> m_threadPool;
};

} // namespace output
} // namespace oscean 