#include "profiles/output_profile_manager.h"

#include "common_utils/cache/icache_manager.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "core_services/exceptions.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <regex>

namespace oscean {
namespace output {

OutputProfileManager::OutputProfileManager(
    std::string profileDirectory,
    std::shared_ptr<common_utils::cache::ICacheManager<std::string, nlohmann::json>> cache,
    std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPool)
    : m_profileDirectory(std::move(profileDirectory)),
      m_cache(std::move(cache)),
      m_threadPool(std::move(threadPool)) {
    if (!m_cache || !m_threadPool) {
        throw std::invalid_argument("Cache and ThreadPool must be non-null.");
    }
}

boost::future<core_services::output::OutputRequest> OutputProfileManager::resolveRequest(
    const core_services::output::ProfiledRequest& profiledRequest) {
    
    return m_threadPool->submitTask([this, profiledRequest]() {
        const std::string& profileName = profiledRequest.profileName;
        
        // 1. Get profile from cache or load from file
        auto cachedProfile = m_cache->get(profileName);
        nlohmann::json profileJson;
        if (cachedProfile) {
            profileJson = *cachedProfile;
        } else {
            profileJson = loadProfileFromFile(profileName);
            m_cache->put(profileName, profileJson);
        }

        // 2. Substitute template variables
        substituteVariables(profileJson, profiledRequest.templateVariables);

        // 3. Construct the final OutputRequest from the resolved JSON
        core_services::output::OutputRequest finalRequest;
        finalRequest.dataSource = profiledRequest.dataSource;
        
        // Safely extract values from JSON
        finalRequest.format = profileJson.at("format").get<std::string>();
        
        if (profileJson.contains("streamOutput")) {
            finalRequest.streamOutput = profileJson.at("streamOutput").get<bool>();
        }

        if (profileJson.contains("targetDirectory")) {
            finalRequest.targetDirectory = profileJson.at("targetDirectory").get<std::string>();
        }
        
        if (profileJson.contains("filenameTemplate")) {
            finalRequest.filenameTemplate = profileJson.at("filenameTemplate").get<std::string>();
        }

        if (profileJson.contains("chunking")) {
            core_services::output::ChunkingOptions chunking;
            chunking.maxFileSizeMB = profileJson["chunking"].value("maxFileSizeMB", 100.0);
            chunking.strategy = profileJson["chunking"].value("strategy", "byRow");
            finalRequest.chunking = chunking;
        }

        if (profileJson.contains("creationOptions")) {
             finalRequest.creationOptions = profileJson["creationOptions"].get<std::map<std::string, std::string>>();
        }
        
        if (profileJson.contains("style")) {
            core_services::output::StyleOptions style;
            style.colorMap = profileJson["style"].value("colorMap", "default");
            style.drawContours = profileJson["style"].value("drawContours", false);
            style.contourLevels = profileJson["style"].value("contourLevels", 10);
            finalRequest.style = style;
        }

        return finalRequest;
    });
}

nlohmann::json OutputProfileManager::loadProfileFromFile(const std::string& profileName) {
    // Basic security check to prevent directory traversal
    if (profileName.find("..") != std::string::npos) {
        throw core_services::ServiceException("Invalid profile name: " + profileName);
    }
    
    std::string fullPath = m_profileDirectory + "/" + profileName + ".json";
    std::ifstream profileFile(fullPath);
    if (!profileFile.is_open()) {
        throw core_services::ServiceException("Could not open profile file: " + fullPath);
    }

    try {
        nlohmann::json parsedJson;
        profileFile >> parsedJson;
        return parsedJson;
    } catch (const nlohmann::json::parse_error& e) {
        throw core_services::ServiceException("Failed to parse profile JSON: " + std::string(e.what()));
    }
}

void OutputProfileManager::substituteVariables(nlohmann::json& profileJson, const std::map<std::string, std::string>& variables) {
    static const std::regex varRegex(R"(\{\{([a-zA-Z0-9_]+)\}\})");

    for (auto& element : profileJson.items()) {
        if (element.value().is_string()) {
            std::string val = element.value().get<std::string>();
            
            // 使用sregex_iterator遍历所有匹配并替换
            std::string result = val;
            std::sregex_iterator iter(val.begin(), val.end(), varRegex);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                const std::smatch& match = *iter;
                const std::string& varName = match[1].str();
                auto it = variables.find(varName);
                if (it != variables.end()) {
                    // 替换整个匹配项 {{varName}} 为变量值
                    std::string searchStr = match[0].str();
                    size_t pos = result.find(searchStr);
                    if (pos != std::string::npos) {
                        result.replace(pos, searchStr.length(), it->second);
                    }
                }
            }
            
            element.value() = result;
        } else if (element.value().is_structured()) {
            substituteVariables(element.value(), variables);
        }
    }
}

} // namespace output
} // namespace oscean 