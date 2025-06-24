#pragma once

#include "core_services/common_data_types.h"
#include <string>
#include <vector>
#include <optional>
#include <chrono> // For removeStale potentially needing timing info, though not strictly required by interface users

namespace oscean {
namespace core_services {
namespace cache { // Place interface in the same namespace level as implementation? (Decided NO based on structure, keeping it under core_services for interface)

/**
 * @brief 元数据缓存接口
 * 定义了缓存操作，用于减少对持久存储的访问。
 * 主要关注文件元数据和查询结果的缓存。
 */
class IMetadataCache {
public:
    virtual ~IMetadataCache() = default;

    /**
     * @brief 获取缓存的文件元数据
     * @param fileId 文件ID
     * @return 可选的元数据，不存在或过期返回std::nullopt
     */
    virtual std::optional<FileMetadata> getMetadata(const std::string& fileId) = 0;

    /**
     * @brief 获取缓存的查询结果
     * @param queryHash 查询条件的哈希值
     * @return 可选的查询结果，不存在或过期返回std::nullopt
     */
    virtual std::optional<std::vector<FileInfo>> getQueryResults(const std::string& queryHash) = 0;

    /**
     * @brief 缓存文件元数据
     * @param fileId 文件ID
     * @param metadata 元数据
     */
    virtual void cacheMetadata(const std::string& fileId, const FileMetadata& metadata) = 0;

    /**
     * @brief 缓存查询结果
     * @param queryHash 查询条件的哈希值
     * @param results 查询结果
     */
    virtual void cacheQueryResults(const std::string& queryHash, std::vector<FileInfo> results) = 0; // Note: results passed by value

    /**
     * @brief 清空缓存
     */
    virtual void clear() = 0;

    /**
     * @brief 移除过期条目
     */
    virtual void removeStale() = 0;

    /**
     * @brief 移除特定键的缓存项
     * @param key 要移除的缓存键 (fileId 或 queryHash)
     */
    virtual void invalidate(const std::string& key) = 0;

    /**
     * @brief 清空所有查询结果缓存
     */
    virtual void clearQueryCache() = 0;

    // Note: Methods like size() or configuration getters (maxSize, ttl) are typically
    // implementation details and might not be part of the core functional interface,
    // unless the service layer needs to dynamically interact with cache capacity/stats.
    // virtual size_t size() const = 0;

    // Note: Methods like putMetadata for simple key-value pairs were present in previous
    // service implementation attempt but don't fit the current MetadataCache design.
    // The service might need a different cache for simple KV pairs if required.
    // virtual std::optional<std::string> getSimpleValue(const std::string& key) const = 0;
    // virtual void putSimpleValue(const std::string& key, const std::string& value) = 0;

};

} // namespace cache
} // namespace core_services
} // namespace oscean 