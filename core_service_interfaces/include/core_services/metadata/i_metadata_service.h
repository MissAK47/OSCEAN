#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "unified_metadata_service.h"  // 包含完整的元数据类型定义
#include "core_services/common_data_types.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <string>
#include <vector>
#include <optional>

// Forward declarations 
namespace oscean {
namespace common_utils {
namespace infrastructure {
    class CommonServicesFactory;
}
}
}

namespace oscean::core_services::metadata {

/**
 * @brief 标准化、重构后的元数据服务接口
 * 
 * 🎯 核心功能：
 * ✅ 提供文件处理、元数据CRUD和查询的统一接口。
 * ✅ 接口设计遵循 C++ 核心准则和现代异步编程模式。
 * ✅ 所有权和生命周期通过智能指针管理。
 */
class IMetadataService {
public:
    virtual ~IMetadataService() = default;

    // === 服务生命周期与状态管理 ===
    
    /**
     * @brief 初始化服务，准备接收请求。
     * @return 如果初始化成功，则返回true。
     */
    virtual bool initialize() = 0;
    
    /**
     * @brief 获取服务版本号。
     */
    virtual std::string getVersion() const = 0;
    
    /**
     * @brief 检查服务是否已初始化并准备好处理请求。
     */
    virtual bool isReady() const = 0;

    // === 核心元数据处理流程 ===
    
    /**
     * @brief 处理单个文件，完成提取、标准化、分类和存储全流程。
     * @param filePath 要处理的文件的完整路径。
     * @return 异步返回一个包含生成的文件元数据ID的结果对象。
     */
    virtual boost::future<AsyncResult<std::string>> processFile(const std::string& filePath) = 0;

    /**
     * @brief 🚀 [新] 批量过滤未处理的文件
     * @param filePaths 待检查的文件路径列表
     * @return 返回一个只包含新文件或已更新文件的列表
     */
    virtual boost::future<std::vector<std::string>> filterUnprocessedFilesAsync(
        const std::vector<std::string>& filePaths) = 0;

    /**
     * @brief 🚀 [新] 对元数据进行分类和最终丰富
     * @param metadata 已经包含原始数据和CRS信息的元数据对象
     * @return 返回完全丰富后的元数据对象
     */
    virtual boost::future<FileMetadata> classifyAndEnrichAsync(
        const FileMetadata& metadata) = 0;

    /**
     * @brief 🚀 [新] 异步保存元数据到持久化存储
     * @param metadata 待保存的完整元数据对象
     * @return 返回一个包含操作结果（如成功/失败信息）的future
     */
    virtual boost::future<AsyncResult<bool>> saveMetadataAsync(
        const FileMetadata& metadata) = 0;

    /**
     * @brief 异步接收一个已存在的 `FileMetadata` 对象进行处理。
     * @note 主要用于外部系统推送元数据，或在内部流程中复用。
     * @param metadata 要处理的文件元数据对象。
     * @return 异步返回一个表示操作完成的结果对象。
     */
    virtual boost::future<AsyncResult<void>> receiveFileMetadataAsync(FileMetadata metadata) = 0;
    
    // === 元数据查询接口 ===
    
    /**
     * @brief 根据通用查询条件异步执行查询。
     * @param criteria 定义查询过滤条件的结构体。
     * @return 异步返回包含匹配文件元数据列表的结果对象。
     */
    virtual boost::future<AsyncResult<std::vector<FileMetadata>>> queryMetadataAsync(const QueryCriteria& criteria) = 0;
    
    /**
     * @brief 根据文件完整路径精确查询元数据。
     * @param filePath 文件的完整路径。
     * @return 异步返回包含单个匹配文件元数据的结果对象，如果未找到则结果包含错误。
     */
    virtual boost::future<AsyncResult<FileMetadata>> queryByFilePathAsync(const std::string& filePath) = 0;
    
    /**
     * @brief 根据主要数据类型进行查询。
     * @param category 要查询的主要数据类型。
     * @param additionalCriteria 可选的附加过滤条件。
     * @return 异步返回包含匹配文件元数据列表的结果对象。
     */
    virtual boost::future<AsyncResult<std::vector<FileMetadata>>> queryByCategoryAsync(
        DataType category,
        const std::optional<QueryCriteria>& additionalCriteria = std::nullopt
    ) = 0;

    // === 元数据修改接口 ===
    
    /**
     * @brief 根据元数据ID异步删除元数据记录。
     * @param metadataId 要删除的元数据记录的唯一ID。
     * @return 异步返回一个布尔值表示操作是否成功的结果对象。
     */
    virtual boost::future<AsyncResult<bool>> deleteMetadataAsync(const std::string& metadataId) = 0;
    
    /**
     * @brief 根据元数据ID异步更新元数据记录。
     * @param metadataId 要更新的元数据记录的唯一ID。
     * @param update 包含要更新字段的结构体。
     * @return 异步返回一个布尔值表示操作是否成功的结果对象。
     */
    virtual boost::future<AsyncResult<bool>> updateMetadataAsync(
        const std::string& metadataId,
        const MetadataUpdate& update
    ) = 0;

    // === 配置管理接口 ===
    
    /**
     * @brief 异步更新服务配置。
     * @param config 新的服务配置。
     * @return 异步返回一个表示操作完成的结果对象。
     */
    virtual boost::future<AsyncResult<void>> updateConfigurationAsync(const MetadataServiceConfiguration& config) = 0;
    
    /**
     * @brief 异步获取当前服务配置。
     * @return 异步返回包含当前服务配置的结果对象。
     */
    virtual boost::future<AsyncResult<MetadataServiceConfiguration>> getConfigurationAsync() = 0;
};

/**
 * @brief 元数据服务工厂接口
 */
class IMetadataServiceFactory {
public:
    virtual ~IMetadataServiceFactory() = default;
    
    /**
     * @brief 创建元数据服务实例
     */
    virtual std::shared_ptr<IMetadataService> createService(
        std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices,
        const MetadataServiceConfiguration& config
    ) = 0;
    
    /**
     * @brief 获取默认配置
     */
    virtual MetadataServiceConfiguration getDefaultConfiguration() const = 0;
};

} // namespace oscean::core_services::metadata 