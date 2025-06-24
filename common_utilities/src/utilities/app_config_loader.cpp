/**
 * @file app_config_loader.cpp
 * @brief 简化版应用配置加载器实现
 * 
 * 🎯 重构说明：
 * ✅ 替代复杂的 config_manager，提供轻量级配置加载
 * ✅ 支持JSON和环境变量配置
 * ✅ 专注于通用配置加载，与CommonServicesFactory协同
 */

#include "common_utils/utilities/app_config_loader.h"
#include "common_utils/utilities/logging_utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <iostream>

// 🔧 添加YAML支持
#include <yaml-cpp/yaml.h>

namespace oscean {
namespace common_utils {

// ConfigValue 类型转换方法实现

bool ConfigValue::asBool() const {
    std::string lowerValue = value;
    std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(), ::tolower);
    return lowerValue == "true" || lowerValue == "1" || lowerValue == "yes" || lowerValue == "on";
}

int ConfigValue::asInt() const {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        return 0;
    }
}

double ConfigValue::asDouble() const {
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        return 0.0;
    }
}

std::string ConfigValue::asString() const {
    return value;
}

std::vector<std::string> ConfigValue::asStringList() const {
    std::vector<std::string> result;
    std::stringstream ss(value);
    std::string item;
    
    // 支持逗号分隔和分号分隔
    char delimiter = value.find(',') != std::string::npos ? ',' : ';';
    
    while (std::getline(ss, item, delimiter)) {
        // 去除前后空格
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    
    return result;
}

// AppConfigLoader 实现

AppConfigLoader::AppConfigLoader(const std::string& appName) : m_appName(appName) {
    // 设置一些通用的默认值
    setDefault("log_level", "INFO", "Default logging level");
    setDefault("debug", "false", "Enable debug mode");
    setDefault("config_file", "", "Configuration file path");
}

bool AppConfigLoader::loadFromFile(const std::filesystem::path& configPath) {
    if (!std::filesystem::exists(configPath)) {
        LOG_WARN("Config file does not exist: {}", configPath.string());
        return false;
    }
    
    std::ifstream file(configPath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open config file: {}", configPath.string());
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    try {
        // 🔧 修复：根据文件扩展名选择正确的解析器
        std::string extension = configPath.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        if (extension == ".yaml" || extension == ".yml") {
            parseYamlObject(content, ConfigSource::FILE_JSON);
            LOG_INFO("Loaded YAML configuration from: {}", configPath.string());
        } else if (extension == ".json") {
            parseJsonObject(content, ConfigSource::FILE_JSON);
            LOG_INFO("Loaded JSON configuration from: {}", configPath.string());
        } else {
            // 默认尝试YAML格式
            parseYamlObject(content, ConfigSource::FILE_JSON);
            LOG_INFO("Loaded configuration (default YAML) from: {}", configPath.string());
        }
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse config file {}: {}", configPath.string(), e.what());
        return false;
    }
}

int AppConfigLoader::loadFromEnvironment(const std::string& prefix) {
    int count = 0;
    
    // 在Windows和Unix系统上获取环境变量的方法不同
    // 这里使用标准的getenv方法，适用于常见的环境变量
    
    // 预定义一些常见的配置环境变量
    std::vector<std::string> commonVars = {
        "LOG_LEVEL", "DEBUG", "CONFIG_FILE", "DATA_DIR", "TEMP_DIR",
        "MAX_MEMORY_MB", "THREAD_POOL_SIZE", "CACHE_SIZE", "ENABLE_SIMD"
    };
    
    for (const std::string& var : commonVars) {
        std::string envVar = prefix + var;
        
#ifdef _WIN32
        // Windows: 使用_dupenv_s获取环境变量
        char* envValue = nullptr;
        size_t envValueSize = 0;
        
        if (_dupenv_s(&envValue, &envValueSize, envVar.c_str()) == 0 && envValue) {
            ConfigValue configValue;
            configValue.value = envValue;
            configValue.source = ConfigSource::ENVIRONMENT;
            configValue.description = "Environment variable: " + envVar;
            
            // 将环境变量名转换为配置键（小写，下划线）
            std::string key = toLower(var);
            m_config[key] = configValue;
            count++;
            
            LOG_DEBUG("Loaded env var: {} = {}", envVar, envValue);
            
            free(envValue); // 释放_dupenv_s分配的内存
        }
#else
        // Unix/Linux: 使用getenv（在非Windows系统上是安全的）
        const char* envValue = std::getenv(envVar.c_str());
        
        if (envValue != nullptr) {
            ConfigValue configValue;
            configValue.value = envValue;
            configValue.source = ConfigSource::ENVIRONMENT;
            configValue.description = "Environment variable: " + envVar;
            
            // 将环境变量名转换为配置键（小写，下划线）
            std::string key = toLower(var);
            m_config[key] = configValue;
            count++;
            
            LOG_DEBUG("Loaded env var: {} = {}", envVar, envValue);
        }
#endif
    }
    
    if (count > 0) {
        LOG_INFO("Loaded {} configuration values from environment variables", count);
    }
    
    return count;
}

int AppConfigLoader::loadFromCommandLine(int argc, char* argv[]) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) { // 跳过程序名
        args.emplace_back(argv[i]);
    }
    
    return parseCommandLineArgs(args) ? static_cast<int>(args.size()) : 0;
}

void AppConfigLoader::setDefault(const std::string& key, const std::string& value, 
                                const std::string& description) {
    ConfigValue defaultValue;
    defaultValue.value = value;
    defaultValue.source = ConfigSource::DEFAULT_VALUES;
    defaultValue.description = description;
    
    m_defaults[normalizeKey(key)] = defaultValue;
}

std::optional<ConfigValue> AppConfigLoader::get(const std::string& key) const {
    std::string normalizedKey = normalizeKey(key);
    
    // 首先查找用户配置
    auto it = m_config.find(normalizedKey);
    if (it != m_config.end()) {
        return it->second;
    }
    
    // 然后查找默认值
    auto defaultIt = m_defaults.find(normalizedKey);
    if (defaultIt != m_defaults.end()) {
        return defaultIt->second;
    }
    
    return std::nullopt;
}

std::string AppConfigLoader::getString(const std::string& key, const std::string& defaultValue) const {
    auto configValue = get(key);
    return configValue ? configValue->asString() : defaultValue;
}

int AppConfigLoader::getInt(const std::string& key, int defaultValue) const {
    auto configValue = get(key);
    return configValue ? configValue->asInt() : defaultValue;
}

bool AppConfigLoader::getBool(const std::string& key, bool defaultValue) const {
    auto configValue = get(key);
    return configValue ? configValue->asBool() : defaultValue;
}

double AppConfigLoader::getDouble(const std::string& key, double defaultValue) const {
    auto configValue = get(key);
    return configValue ? configValue->asDouble() : defaultValue;
}

bool AppConfigLoader::has(const std::string& key) const {
    return get(key).has_value();
}

std::vector<std::string> AppConfigLoader::getKeys() const {
    std::vector<std::string> keys;
    
    // 收集所有配置键
    for (const auto& [key, value] : m_config) {
        keys.push_back(key);
    }
    
    // 添加只有默认值的键
    for (const auto& [key, value] : m_defaults) {
        if (m_config.find(key) == m_config.end()) {
            keys.push_back(key);
        }
    }
    
    std::sort(keys.begin(), keys.end());
    return keys;
}

std::map<std::string, ConfigValue> AppConfigLoader::getAll() const {
    std::map<std::string, ConfigValue> result = m_defaults;
    
    // 用户配置覆盖默认值
    for (const auto& [key, value] : m_config) {
        result[key] = value;
    }
    
    return result;
}

void AppConfigLoader::printConfig(bool includeDefaults) const {
    std::cout << "=== " << m_appName << " Configuration ===" << std::endl;
    
    auto allConfig = getAll();
    for (const auto& [key, value] : allConfig) {
        if (!includeDefaults && value.source == ConfigSource::DEFAULT_VALUES) {
            continue;
        }
        
        std::string sourceStr;
        switch (value.source) {
            case ConfigSource::FILE_JSON: sourceStr = "FILE"; break;
            case ConfigSource::ENVIRONMENT: sourceStr = "ENV"; break;
            case ConfigSource::COMMAND_LINE: sourceStr = "CMD"; break;
            case ConfigSource::DEFAULT_VALUES: sourceStr = "DEFAULT"; break;
        }
        
        std::cout << "  " << key << " = " << value.value 
                  << " [" << sourceStr << "]";
        if (!value.description.empty()) {
            std::cout << " - " << value.description;
        }
        std::cout << std::endl;
    }
    std::cout << "===============================" << std::endl;
}

std::vector<std::string> AppConfigLoader::validateRequired(const std::vector<std::string>& requiredKeys) const {
    std::vector<std::string> missing;
    
    for (const std::string& key : requiredKeys) {
        if (!has(key)) {
            missing.push_back(key);
        }
    }
    
    return missing;
}

bool AppConfigLoader::loadStandardConfig() {
    // 尝试加载多个标准位置的配置文件
    std::vector<std::string> config_paths = {
        "config.json",
        "app.config",
        "../config/app.config",
        "../../config/app.config"
    };
    
    for (const auto& path : config_paths) {
        if (std::filesystem::exists(path)) {
            if (loadFromFile(path)) {
                return true;
            }
        }
    }
    
    return false;
}

// 私有方法实现

void AppConfigLoader::parseJsonObject(const std::string& jsonContent, ConfigSource source) {
    // 简化的JSON解析 - 只处理简单的键值对
    auto jsonMap = parseSimpleJson(jsonContent);
    
    for (const auto& [key, value] : jsonMap) {
        ConfigValue configValue;
        configValue.value = value;
        configValue.source = source;
        configValue.description = "Loaded from JSON";
        
        m_config[normalizeKey(key)] = configValue;
    }
}

void AppConfigLoader::parseYamlObject(const std::string& yamlContent, ConfigSource source) {
    try {
        YAML::Node root = YAML::Load(yamlContent);
        parseYamlNode(root, "", source);
        LOG_DEBUG("Successfully parsed YAML configuration");
    } catch (const YAML::Exception& e) {
        LOG_ERROR("YAML parsing error: {}", e.what());
        throw std::runtime_error("Failed to parse YAML configuration: " + std::string(e.what()));
    }
}

void AppConfigLoader::parseYamlNode(const YAML::Node& node, const std::string& prefix, ConfigSource source) {
    if (node.IsMap()) {
        for (const auto& item : node) {
            std::string key = item.first.as<std::string>();
            std::string fullKey = prefix.empty() ? key : prefix + "." + key;
            
            if (item.second.IsScalar()) {
                ConfigValue configValue;
                configValue.value = item.second.as<std::string>();
                configValue.source = source;
                configValue.description = "Loaded from YAML";
                
                m_config[normalizeKey(fullKey)] = configValue;
                LOG_DEBUG("Loaded config: {} = {}", fullKey, configValue.value);
            } else {
                // 递归处理嵌套对象
                parseYamlNode(item.second, fullKey, source);
            }
        }
    } else if (node.IsSequence()) {
        // 处理数组类型，将其转换为逗号分隔的字符串
        std::stringstream ss;
        for (size_t i = 0; i < node.size(); ++i) {
            if (i > 0) ss << ",";
            ss << node[i].as<std::string>();
        }
        
        ConfigValue configValue;
        configValue.value = ss.str();
        configValue.source = source;
        configValue.description = "Loaded from YAML (array)";
        
        m_config[normalizeKey(prefix)] = configValue;
    }
}

std::string AppConfigLoader::normalizeKey(const std::string& key) const {
    std::string normalized = key;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    // 将连字符转换为下划线
    std::replace(normalized.begin(), normalized.end(), '-', '_');
    
    return normalized;
}

std::vector<std::filesystem::path> AppConfigLoader::getStandardConfigPaths() const {
    std::vector<std::filesystem::path> paths;
    
    // 当前目录
    paths.emplace_back(m_appName + ".json");
    paths.emplace_back("config.json");
    
    // 配置目录
    paths.emplace_back("config/" + m_appName + ".json");
    paths.emplace_back("config/config.json");
    
    // 用户主目录（安全处理）
#ifdef _WIN32
    // Windows: 使用_dupenv_s获取环境变量
    char* homeDir = nullptr;
    size_t homeDirSize = 0;
    
    if (_dupenv_s(&homeDir, &homeDirSize, "USERPROFILE") == 0 && homeDir) {
        std::filesystem::path homePath(homeDir);
        paths.emplace_back(homePath / ("." + m_appName) / "config.json");
        paths.emplace_back(homePath / ".config" / m_appName / "config.json");
        free(homeDir); // 释放_dupenv_s分配的内存
    }
#else
    // Unix/Linux: 使用getenv（在非Windows系统上是安全的）
    const char* homeDir = std::getenv("HOME");
    if (homeDir) {
        std::filesystem::path homePath(homeDir);
        paths.emplace_back(homePath / ("." + m_appName) / "config.json");
        paths.emplace_back(homePath / ".config" / m_appName / "config.json");
    }
#endif
    
    return paths;
}

bool AppConfigLoader::parseCommandLineArgs(const std::vector<std::string>& args) {
    int count = 0;
    
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        
        // 处理 --key=value 格式
        if (arg.substr(0, 2) == "--") {
            size_t equalPos = arg.find('=');
            if (equalPos != std::string::npos) {
                std::string key = arg.substr(2, equalPos - 2);
                std::string value = arg.substr(equalPos + 1);
                
                ConfigValue configValue;
                configValue.value = value;
                configValue.source = ConfigSource::COMMAND_LINE;
                configValue.description = "Command line argument";
                
                m_config[normalizeKey(key)] = configValue;
                count++;
            }
            // 处理 --key value 格式
            else if (i + 1 < args.size() && args[i + 1][0] != '-') {
                std::string key = arg.substr(2);
                std::string value = args[i + 1];
                
                ConfigValue configValue;
                configValue.value = value;
                configValue.source = ConfigSource::COMMAND_LINE;
                configValue.description = "Command line argument";
                
                m_config[normalizeKey(key)] = configValue;
                count++;
                i++; // 跳过下一个参数
            }
        }
    }
    
    if (count > 0) {
        LOG_INFO("Loaded {} configuration values from command line", count);
    }
    
    return count > 0;
}

std::string AppConfigLoader::toLower(const std::string& str) const {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::map<std::string, std::string> AppConfigLoader::parseSimpleJson(const std::string& jsonContent) const {
    std::map<std::string, std::string> result;
    
    // 这是一个非常简化的JSON解析器，只处理简单的键值对
    // 格式：{"key": "value", "key2": "value2"}
    
    std::string content = jsonContent;
    
    // 移除空白字符
    content.erase(std::remove_if(content.begin(), content.end(), ::isspace), content.end());
    
    // 简单验证JSON格式
    if (content.empty() || content[0] != '{' || content.back() != '}') {
        return result;
    }
    
    // 去掉外层大括号
    content = content.substr(1, content.length() - 2);
    
    // 分割键值对
    std::stringstream ss(content);
    std::string pair;
    
    while (std::getline(ss, pair, ',')) {
        size_t colonPos = pair.find(':');
        if (colonPos != std::string::npos) {
            std::string key = pair.substr(0, colonPos);
            std::string value = pair.substr(colonPos + 1);
            
            // 去掉引号
            if (key.length() >= 2 && key[0] == '"' && key.back() == '"') {
                key = key.substr(1, key.length() - 2);
            }
            if (value.length() >= 2 && value[0] == '"' && value.back() == '"') {
                value = value.substr(1, value.length() - 2);
            }
            
            result[key] = value;
        }
    }
    
    return result;
}

std::string AppConfigLoader::extractJsonValue(const std::string& jsonContent, const std::string& key) const {
    // 简单的JSON值提取
    std::string searchKey = "\"" + key + "\"";
    size_t keyPos = jsonContent.find(searchKey);
    
    if (keyPos == std::string::npos) {
        return "";
    }
    
    size_t colonPos = jsonContent.find(':', keyPos);
    if (colonPos == std::string::npos) {
        return "";
    }
    
    size_t valueStart = jsonContent.find_first_not_of(" \t\n\r", colonPos + 1);
    if (valueStart == std::string::npos) {
        return "";
    }
    
    size_t valueEnd;
    if (jsonContent[valueStart] == '"') {
        // 字符串值
        valueEnd = jsonContent.find('"', valueStart + 1);
        if (valueEnd == std::string::npos) {
            return "";
        }
        return jsonContent.substr(valueStart + 1, valueEnd - valueStart - 1);
    } else {
        // 数值或布尔值
        valueEnd = jsonContent.find_first_of(",}", valueStart);
        if (valueEnd == std::string::npos) {
            valueEnd = jsonContent.length();
        }
        std::string value = jsonContent.substr(valueStart, valueEnd - valueStart);
        // 去掉尾部空白
        value.erase(value.find_last_not_of(" \t\n\r") + 1);
        return value;
    }
}

} // namespace common_utils
} // namespace oscean 