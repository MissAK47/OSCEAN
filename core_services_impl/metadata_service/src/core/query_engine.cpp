// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "impl/query_engine.h"
#include "impl/unified_database_manager.h"
// #include "common_utils/utilities/logging_utils.h"
// #include "core_services/common_data_types.h"
// #include <boost/thread/future.hpp>
// #include <algorithm>
#include <unordered_set>
#include <sstream>

using namespace oscean::core_services::metadata::impl;
using namespace oscean::core_services::metadata;
using namespace oscean::common_utils;

namespace oscean {
namespace core_services {
namespace metadata {
namespace impl {

// QueryEngine has been temporarily disabled.

} // namespace impl
} // namespace metadata
} // namespace core_services
} // namespace oscean

QueryEngine::QueryEngine(
    std::shared_ptr<UnifiedDatabaseManager> dbManager,
    std::shared_ptr<infrastructure::CommonServicesFactory> commonServices)
    : dbManager_(std::move(dbManager)), 
      commonServices_(std::move(commonServices)) {

    auto logService = commonServices_->getService<log_service::ILogService>();
    if (logService) {
        logger_ = logService->getLogger("metadata_service.QueryEngine");
    }
    
    LOG_WITH_LOGGER(logger_, spdlog::level::info, "QueryEngine初始化完成");
}

boost::future<AsyncResult<std::vector<FileMetadata>>> QueryEngine::executeQueryAsync(
    const QueryCriteria& criteria) {
    
    LOG_WITH_LOGGER(logger_, spdlog::level::info, "开始执行通用查询");
    
    if (!validateQueryCriteria(criteria)) {
        LOG_WITH_LOGGER(logger_, spdlog::level::warn, "查询条件验证失败");
        auto promise = boost::promise<AsyncResult<std::vector<FileMetadata>>>();
        promise.set_value(AsyncResult<std::vector<FileMetadata>>::failure("无效的查询条件"));
        return promise.get_future();
    }

    return dbManager_->queryMetadataAsync(criteria).then([this, criteria](boost::future<AsyncResult<std::vector<FileMetadata>>> f) {
        auto result = f.get();
        if (result.isSuccess()) {
            auto processedResults = postProcessResults(result.getData(), criteria);
            LOG_WITH_LOGGER(logger_, spdlog::level::info, "通用查询成功，返回 {} 条后处理记录", processedResults.size());
            return AsyncResult<std::vector<FileMetadata>>::success(std::move(processedResults));
        }
        LOG_WITH_LOGGER(logger_, spdlog::level::error, "通用查询在数据库层面失败: {}", result.getError());
        return result;
    });
}

boost::future<AsyncResult<std::vector<FileMetadata>>> QueryEngine::executeQueryByFilePathAsync(
    const std::string& filePath) {
    
    LOG_WITH_LOGGER(logger_, spdlog::level::info, "开始按文件路径查询: {}", filePath);

    return dbManager_->queryByFilePathAsync(filePath).then([this, filePath](boost::future<AsyncResult<std::vector<FileMetadata>>> f) {
        auto result = f.get();
        if (result.isSuccess()) {
            LOG_WITH_LOGGER(logger_, spdlog::level::info, "按文件路径查询 '{}' 成功，返回 {} 条记录", filePath, result.getData().size());
        } else {
            LOG_WITH_LOGGER(logger_, spdlog::level::error, "按文件路径查询 '{}' 失败: {}", filePath, result.getError());
        }
        return result;
    });
}

boost::future<AsyncResult<std::vector<FileMetadata>>> QueryEngine::executeQueryByCategoryAsync(
    DataType category,
    const std::optional<QueryCriteria>& additionalCriteria) {
    
    LOG_WITH_LOGGER(logger_, spdlog::level::info, "开始按数据类型查询: {}", static_cast<int>(category));

    QueryCriteria criteria;
    if (additionalCriteria) {
        criteria = *additionalCriteria;
    }
    
    if (std::find(criteria.dataTypes.begin(), criteria.dataTypes.end(), category) == criteria.dataTypes.end()) {
        criteria.dataTypes.push_back(category);
    }

    return executeQueryAsync(criteria);
}

std::vector<oscean::core_services::FileMetadata> QueryEngine::postProcessResults(const std::vector<oscean::core_services::FileMetadata>& results, const QueryCriteria& criteria) {
    std::vector<oscean::core_services::FileMetadata> processed = results;
    
    removeDuplicates(processed);
    
    sortResultsByTimestamp(processed, false);
    
    if (criteria.limit && processed.size() > *criteria.limit) {
        processed.resize(*criteria.limit);
        LOG_WITH_LOGGER(logger_, spdlog::level::debug, "结果已限制为 {} 条记录", *criteria.limit);
    }
    
    return processed;
}

void QueryEngine::removeDuplicates(std::vector<oscean::core_services::FileMetadata>& entries) {
    if (entries.empty()) return;
    std::unordered_set<std::string> seen;
    auto originalSize = entries.size();
    entries.erase(
        std::remove_if(entries.begin(), entries.end(), 
            [&seen](const oscean::core_services::FileMetadata& entry) {
                return !seen.insert(entry.metadataId).second;
            }),
        entries.end()
    );
    if (originalSize > entries.size()) {
        LOG_WITH_LOGGER(logger_, spdlog::level::debug, "移除了 {} 条重复记录", originalSize - entries.size());
    }
}

void QueryEngine::sortResultsByTimestamp(std::vector<oscean::core_services::FileMetadata>& entries, bool ascending) {
    std::sort(entries.begin(), entries.end(), 
        [ascending](const oscean::core_services::FileMetadata& a, const oscean::core_services::FileMetadata& b) {
            if (ascending) {
                return a.extractionTimestamp < b.extractionTimestamp;
            } else {
                return a.extractionTimestamp > b.extractionTimestamp;
            }
        });
}

bool QueryEngine::validateQueryCriteria(const QueryCriteria& criteria) {
    if (criteria.limit && *criteria.limit <= 0) {
        LOG_WITH_LOGGER(logger_, spdlog::level::warn, "查询条件验证失败: limit 值必须大于 0");
        return false;
    }
    
    return true;
}

boost::future<AsyncResult<std::vector<FileMetadata>>> QueryEngine::queryMetadataAsync(
    const UnifiedQueryCriteria& criteria) 
{
    LOG_WARN(logger_, "QueryEngine::queryMetadataAsync is not yet implemented for UnifiedDatabaseManager.");
    auto promise = std::make_shared<boost::promise<AsyncResult<std::vector<FileMetadata>>>>();
    promise->set_value(AsyncResult<std::vector<FileMetadata>>::success({}));
    return promise->get_future();
}

boost::future<AsyncResult<boost::optional<FileMetadata>>> QueryEngine::getFileMetadataAsync(
    const std::string& filePath) 
{
    LOG_WARN(logger_, "QueryEngine::getFileMetadataAsync is not yet implemented for UnifiedDatabaseManager.");
    auto promise = std::make_shared<boost::promise<AsyncResult<boost::optional<FileMetadata>>>>();
    promise->set_value(AsyncResult<boost::optional<FileMetadata>>::success(boost::none));
    return promise->get_future();
}

namespace oscean::core_services::metadata {

} // namespace oscean::core_services::metadata 