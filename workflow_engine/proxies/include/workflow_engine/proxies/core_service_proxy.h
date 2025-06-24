#pragma once

/**
 * @file core_service_proxy.h
 * @brief 核心服务代理接口
 * @author OSCEAN Team
 * @date 2024
 */

#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <optional>
#include <map>
#include <any>

// 前向声明避免头文件冲突
namespace oscean::core_services {
    struct FileMetadata;
}

namespace oscean::core_services::metadata {
    class IUnifiedMetadataService;
}

namespace oscean::workflow_engine::proxies {

/**
 * @brief 核心服务代理接口
 * 
 * 提供对核心服务的统一访问接口，隔离workflow_engine与具体服务实现的耦合。
 * 当前版本主要支持元数据服务，DAS和CRS服务将在后续版本中集成。
 */
class CoreServiceProxy {
public:
    virtual ~CoreServiceProxy() = default;

    /**
     * @brief 初始化代理
     * @param config 配置参数
     * @return 是否初始化成功
     */
    virtual bool initialize(const std::map<std::string, std::any>& config) = 0;

    /**
     * @brief 从数据访问服务获取文件元数据
     * @param filePath 文件路径
     * @return 文件元数据的future，如果失败返回nullopt
     * @note 当前版本返回nullopt，待DAS服务集成后实现
     */
    virtual boost::future<std::optional<oscean::core_services::FileMetadata>>
    getFileMetadataFromDAS(const std::string& filePath) = 0;

    /**
     * @brief 通过元数据服务识别文件类型
     * @param filePath 文件路径
     * @return 识别的数据类型字符串
     */
    virtual boost::future<std::string>
    recognizeFileWithMDS(const std::string& filePath) = 0;

    /**
     * @brief 检查服务是否可用
     * @return 是否所有必要服务都可用
     */
    virtual bool isServiceAvailable() const = 0;

    /**
     * @brief 获取服务状态描述
     * @return 服务状态字符串
     */
    virtual std::string getServiceStatus() const = 0;

    /**
     * @brief 关闭代理并清理资源
     */
    virtual void shutdown() = 0;
};

/**
 * @brief 创建默认的核心服务代理
 * @return 代理实例
 */
std::shared_ptr<CoreServiceProxy> createCoreServiceProxy();

/**
 * @brief 创建并配置核心服务代理
 * @param mds 元数据服务实例
 * @return 配置好的代理实例
 */
std::shared_ptr<CoreServiceProxy> createCoreServiceProxy(
    std::shared_ptr<oscean::core_services::metadata::IUnifiedMetadataService> mds
);

} // namespace oscean::workflow_engine::proxies 