#pragma once

#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/spatial_ops/spatial_exceptions.h"
#include <string>
#include <memory>
#include <map>
#include <stdexcept>

namespace oscean::core_services::spatial_ops {

/**
 * @class SpatialConfigManager
 * @brief 管理空间服务配置的加载、验证和应用
 */
class SpatialConfigManager {
public:
    /**
     * @brief 构造函数
     */
    SpatialConfigManager();

    /**
     * @brief 析构函数
     */
    ~SpatialConfigManager();

    /**
     * @brief 从文件加载配置
     * @param configPath 配置文件路径
     * @return 加载的配置对象
     */
    SpatialOpsConfig loadFromFile(const std::string& configPath);

    /**
     * @brief 从环境变量加载配置
     * @param config 要更新的配置对象
     */
    void loadFromEnvironment(SpatialOpsConfig& config);

    /**
     * @brief 保存配置到文件
     * @param config 要保存的配置
     * @param configPath 配置文件路径
     * @return 保存成功返回true
     */
    bool saveToFile(const SpatialOpsConfig& config, const std::string& configPath);

    /**
     * @brief 获取默认配置
     * @return 默认配置对象
     */
    static SpatialOpsConfig getDefaultConfig();

    /**
     * @brief 合并两个配置对象
     * @param base 基础配置
     * @param override 覆盖配置
     * @return 合并后的配置
     */
    static SpatialOpsConfig mergeConfigs(const SpatialOpsConfig& base, const SpatialOpsConfig& override);

    /**
     * @brief 验证配置并应用修正
     * @param config 要验证的配置
     * @return 验证成功返回true
     */
    bool validateAndFix(SpatialOpsConfig& config);

    /**
     * @brief 获取配置的字符串表示
     * @param config 配置对象
     * @return 配置的字符串表示
     */
    std::string configToString(const SpatialOpsConfig& config) const;

    /**
     * @brief 从字符串解析配置
     * @param configString 配置字符串
     * @return 解析的配置对象
     */
    SpatialOpsConfig configFromString(const std::string& configString);

    /**
     * @brief 设置配置模板
     * @param templateName 模板名称
     * @param config 模板配置
     */
    void setConfigTemplate(const std::string& templateName, const SpatialOpsConfig& config);

    /**
     * @brief 获取配置模板
     * @param templateName 模板名称
     * @return 模板配置，如果不存在返回默认配置
     */
    SpatialOpsConfig getConfigTemplate(const std::string& templateName);

    /**
     * @brief 列出所有可用的配置模板
     * @return 模板名称列表
     */
    std::vector<std::string> listConfigTemplates() const;

private:
    // 配置模板存储
    std::map<std::string, SpatialOpsConfig> m_configTemplates;

    // 私有辅助方法
    void initializeDefaultTemplates();
    bool parseYamlConfig(const std::string& yamlContent, SpatialOpsConfig& config);
    bool parseJsonConfig(const std::string& jsonContent, SpatialOpsConfig& config);
    std::string configToYaml(const SpatialOpsConfig& config) const;
    std::string configToJson(const SpatialOpsConfig& config) const;
    void applyEnvironmentOverrides(SpatialOpsConfig& config);
    bool validateConfigSection(const SpatialOpsConfig& config, const std::string& section);
    void applyConfigValue(SpatialOpsConfig& config, const std::string& key, const std::string& value);
};

} // namespace oscean::core_services::spatial_ops 