#pragma once

#include "common_utils/utilities/boost_config.h"
#include "core_services/common_data_types.h"
#include "dataset_metadata_types.h"
#include <boost/thread/future.hpp>
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace oscean::core_services::metadata {

// 导入类型定义
using MetadataEntry = oscean::core_services::metadata::MetadataEntry;
using MultiDimensionalQueryCriteria = oscean::core_services::metadata::MultiDimensionalQueryCriteria;
using AggregationQueryCriteria = oscean::core_services::metadata::AggregationQueryCriteria;
using AggregatedQueryResult = oscean::core_services::metadata::AggregatedQueryResult;
using RecommendationOptions = oscean::core_services::metadata::RecommendationOptions;
using AsyncResult = oscean::core_services::metadata::AsyncResult;

/**
 * @brief 元数据查询引擎接口
 * 
 * 🎯 负责高性能的元数据查询和分析
 * ✅ 支持多维度并发查询
 * ✅ 跨数据库聚合分析
 * ✅ 智能推荐算法
 * ✅ 异步并发处理
 */
class IMetadataQueryEngine {
public:
    virtual ~IMetadataQueryEngine() = default;

    /**
     * @brief 执行多维度并发查询
     * @param criteria 查询条件
     * @return 异步查询结果
     */
    virtual boost::future<AsyncResult<std::vector<MetadataEntry>>> executeParallelQueryAsync(
        const MultiDimensionalQueryCriteria& criteria
    ) = 0;

    /**
     * @brief 执行跨库聚合查询
     * @param criteria 聚合查询条件
     * @return 异步聚合结果
     */
    virtual boost::future<AsyncResult<AggregatedQueryResult>> executeAggregateQueryAsync(
        const AggregationQueryCriteria& criteria
    ) = 0;

    /**
     * @brief 执行智能推荐查询
     * @param referenceMetadataId 参考元数据ID
     * @param options 推荐选项
     * @return 异步推荐结果
     */
    virtual boost::future<AsyncResult<std::vector<MetadataEntry>>> executeRecommendationQueryAsync(
        const std::string& referenceMetadataId,
        const RecommendationOptions& options
    ) = 0;

    /**
     * @brief 优化查询性能
     * @param criteria 查询条件模式
     * @return 异步优化结果
     */
    virtual boost::future<AsyncResult<void>> optimizeQueryPerformanceAsync(
        const std::vector<MultiDimensionalQueryCriteria>& patterns
    ) = 0;

    /**
     * @brief 预热查询缓存
     * @param commonCriteria 常用查询条件列表
     * @return 异步预热结果
     */
    virtual boost::future<AsyncResult<void>> warmupCacheAsync(
        const std::vector<MultiDimensionalQueryCriteria>& commonCriteria
    ) = 0;

    /**
     * @brief 清除查询缓存
     * @return 异步清除结果
     */
    virtual boost::future<AsyncResult<void>> clearCacheAsync() = 0;
};

} // namespace oscean::core_services::metadata 