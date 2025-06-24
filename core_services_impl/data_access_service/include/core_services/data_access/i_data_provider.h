#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <boost/thread/future.hpp>
#include "core_services/common_data_types.h"
#include "data_access_requests.h"
#include "data_access_responses.h"

namespace oscean::core_services::data_access {

/**
 * @brief 数据提供者接口
 */
class IDataProvider {
public:
    virtual ~IDataProvider() = default;
    
    /**
     * @brief 异步读取格点数据
     * @param request 格点数据读取请求
     * @return 格点数据的异步结果
     */
    virtual boost::future<std::shared_ptr<GridData>> readGridDataAsync(
        const ReadGridDataRequest& request) = 0;
    
    /**
     * @brief 异步读取要素集合
     * @param request 要素集合读取请求
     * @return 要素集合的异步结果
     */
    virtual boost::future<FeatureCollection> readFeatureCollectionAsync(
        const ReadFeatureCollectionRequest& request) = 0;
    
    /**
     * @brief 异步读取时间序列数据
     * @param request 时间序列读取请求
     * @return 时间序列数据的异步结果
     */
    virtual boost::future<TimeSeriesData> readTimeSeriesAsync(
        const ReadTimeSeriesRequest& request) = 0;
    
    /**
     * @brief 异步读取垂直剖面数据
     * @param request 垂直剖面读取请求
     * @return 垂直剖面数据的异步结果
     */
    virtual boost::future<VerticalProfileData> readVerticalProfileAsync(
        const ReadVerticalProfileRequest& request) = 0;
    
    /**
     * @brief 异步读取原始变量数据
     * @param variableName 变量名
     * @param startIndices 起始索引
     * @param counts 数据数量
     * @return 原始变量数据的异步结果
     */
    virtual boost::future<VariableDataVariant> readRawVariableDataAsync(
        const std::string& variableName,
        const std::vector<size_t>& startIndices,
        const std::vector<size_t>& counts) = 0;
};

} // namespace oscean::core_services::data_access 