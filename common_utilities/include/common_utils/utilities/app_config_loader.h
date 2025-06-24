/**
 * @file app_config_loader.h
 * @brief 简化版应用配置加载器
 * 
 * 🎯 重构说明：
 * ✅ 替代复杂的 config_manager，提供轻量级配置加载
 * ✅ 支持JSON和环境变量配置
 * ✅ 专注于通用配置加载，与CommonServicesFactory协同
 */

#pragma once

#include <string>
#include <map>
#include <optional>
#include <filesystem>
#include <vector>
#include <memory>

// 前向声明
namespace YAML {
    class Node;
}

namespace oscean {
namespace common_utils {

/**
 * @enum ConfigSource
 * @brief 配置来源类型
 */
enum class ConfigSource {
    FILE_JSON,      // JSON文件
    ENVIRONMENT,    // 环境变量
    COMMAND_LINE,   // 命令行参数
    DEFAULT_VALUES  // 默认值
};

/**
 * @struct ConfigValue
 * @brief 配置值容器
 */
struct ConfigValue {
    std::string value;
    ConfigSource source = ConfigSource::DEFAULT_VALUES;
    std::string description;
    
    // 类型转换方法
    template<typename T>
    std::optional<T> as() const;
    
    bool asBool() const;
    int asInt() const;
    double asDouble() const;
    std::string asString() const;
    std::vector<std::string> asStringList() const;
};

/**
 * @class AppConfigLoader
 * @brief 简化版应用配置加载器
 * 
 * 提供轻量级的配置管理功能：
 * - JSON配置文件加载
 * - 环境变量读取
 * - 命令行参数解析
 * - 配置值类型转换
 * - 默认值管理
 */
class AppConfigLoader {
public:
    /**
     * @brief 构造函数
     * @param appName 应用程序名称（用于默认配置路径）
     */
    explicit AppConfigLoader(const std::string& appName = "oscean");
    
    /**
     * @brief 析构函数
     */
    ~AppConfigLoader() = default;
    
    // 删除拷贝构造和赋值
    AppConfigLoader(const AppConfigLoader&) = delete;
    AppConfigLoader& operator=(const AppConfigLoader&) = delete;
    
    /**
     * @brief 从JSON文件加载配置
     * @param configPath 配置文件路径
     * @return 是否加载成功
     */
    bool loadFromFile(const std::filesystem::path& configPath);
    
    /**
     * @brief 从环境变量加载配置
     * @param prefix 环境变量前缀（默认为"OSCEAN_"）
     * @return 加载的环境变量数量
     */
    int loadFromEnvironment(const std::string& prefix = "OSCEAN_");
    
    /**
     * @brief 从命令行参数加载配置
     * @param argc 参数数量
     * @param argv 参数数组
     * @return 解析的参数数量
     */
    int loadFromCommandLine(int argc, char* argv[]);
    
    /**
     * @brief 设置默认配置值
     * @param key 配置键
     * @param value 默认值
     * @param description 描述信息
     */
    void setDefault(const std::string& key, const std::string& value, 
                   const std::string& description = "");
    
    /**
     * @brief 获取配置值
     * @param key 配置键
     * @return 配置值（如果存在）
     */
    std::optional<ConfigValue> get(const std::string& key) const;
    
    /**
     * @brief 获取配置值（带默认值）
     * @param key 配置键
     * @param defaultValue 默认值
     * @return 配置值
     */
    std::string getString(const std::string& key, const std::string& defaultValue = "") const;
    
    /**
     * @brief 获取整数配置值
     * @param key 配置键
     * @param defaultValue 默认值
     * @return 配置值
     */
    int getInt(const std::string& key, int defaultValue = 0) const;
    
    /**
     * @brief 获取布尔配置值
     * @param key 配置键
     * @param defaultValue 默认值
     * @return 配置值
     */
    bool getBool(const std::string& key, bool defaultValue = false) const;
    
    /**
     * @brief 获取浮点数配置值
     * @param key 配置键
     * @param defaultValue 默认值
     * @return 配置值
     */
    double getDouble(const std::string& key, double defaultValue = 0.0) const;
    
    /**
     * @brief 检查配置键是否存在
     * @param key 配置键
     * @return 是否存在
     */
    bool has(const std::string& key) const;
    
    /**
     * @brief 获取所有配置键
     * @return 配置键列表
     */
    std::vector<std::string> getKeys() const;
    
    /**
     * @brief 获取所有配置项
     * @return 配置映射
     */
    std::map<std::string, ConfigValue> getAll() const;
    
    /**
     * @brief 打印配置信息（用于调试）
     * @param includeDefaults 是否包含默认值
     */
    void printConfig(bool includeDefaults = false) const;
    
    /**
     * @brief 验证必需的配置项
     * @param requiredKeys 必需的配置键
     * @return 缺失的配置键
     */
    std::vector<std::string> validateRequired(const std::vector<std::string>& requiredKeys) const;
    
    /**
     * @brief 加载标准配置文件路径
     * @return 是否找到并加载了配置文件
     * 
     * ⚠️ 重要变更：
     * - 移除了单例模式 getInstance()
     * - 请通过 CommonServicesFactory::getConfigurationLoader() 获取实例
     * - 支持依赖注入和多实例
     */
    bool loadStandardConfig();

private:
    std::string m_appName;
    std::map<std::string, ConfigValue> m_config;
    std::map<std::string, ConfigValue> m_defaults;
    
    // 内部工具方法
    void parseJsonObject(const std::string& jsonContent, ConfigSource source);
    void parseYamlObject(const std::string& yamlContent, ConfigSource source);
    std::string normalizeKey(const std::string& key) const;
    std::vector<std::filesystem::path> getStandardConfigPaths() const;
    bool parseCommandLineArgs(const std::vector<std::string>& args);
    std::string toLower(const std::string& str) const;
    
    // JSON解析辅助方法（使用简单解析，避免依赖重型JSON库）
    std::map<std::string, std::string> parseSimpleJson(const std::string& jsonContent) const;
    std::string extractJsonValue(const std::string& jsonContent, const std::string& key) const;
    
    // YAML解析辅助方法
    void parseYamlNode(const YAML::Node& node, const std::string& prefix, ConfigSource source);
};

// 模板方法实现
template<typename T>
std::optional<T> ConfigValue::as() const {
    // 这里需要特化处理不同类型的转换
    // 基本实现在cpp文件中
    return std::nullopt;
}

} // namespace common_utils
} // namespace oscean 