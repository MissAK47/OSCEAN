#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "core_services/metadata/unified_metadata_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/utilities/logging_utils.h"
#include <memory>
#include <string>
#include <mutex>
#include <boost/thread/future.hpp>

// Forward declare spdlog logger
namespace spdlog {
    class logger;
}

// Forward declare YAML Node
namespace YAML {
    class Node;
}

namespace oscean::core_services::metadata::impl {

using namespace oscean::common_utils;

/**
 * @brief 配置管理器
 * @note 负责管理元数据服务的配置信息
 */
class ConfigurationManager {
public:
    /**
     * @brief 构造函数
     * @param commonServices 通用服务工厂
     */
    ConfigurationManager(
        std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices
    );

    /**
     * @brief 加载所有配置文件
     * @param configBasePath 配置文件基础路径
     */
    void loadAllConfigurations(const std::string& configBasePath = "config");

    /**
     * @brief 更新变量分类配置
     */
    boost::future<AsyncResult<void>> updateVariableClassificationAsync(
        const VariableClassificationConfig& config
    );
    
    /**
     * @brief 更新数据库配置
     */
    boost::future<AsyncResult<void>> updateDatabaseConfigurationAsync(
        const DatabaseConfiguration& config
    );
    
    /**
     * @brief 获取当前配置
     */
    boost::future<AsyncResult<MetadataServiceConfiguration>> getConfigurationAsync();

    // This is now the primary method to get the fully loaded configuration
    const MetadataServiceConfiguration& getFullConfiguration() const;

private:
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices_;
    
    // 配置数据
    MetadataServiceConfiguration currentConfig_;
    
    std::mutex configMutex_;
    
    std::shared_ptr<spdlog::logger> logger_;

    // Specific parsers
    void loadDatabaseConfig(const std::string& filePath);
    void loadClassificationConfig(const std::string& filePath);

    // YAML parsing helpers
    void parseClassificationRules(const YAML::Node& node, VariableClassificationConfig& config);

    // 私有方法
    bool validateConfiguration(const MetadataServiceConfiguration& config) const;
    void saveConfigurationToDisk(const MetadataServiceConfiguration& config);
};

} // namespace oscean::core_services::metadata::impl 