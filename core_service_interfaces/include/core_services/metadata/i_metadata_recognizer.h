#pragma once

#include "common_utils/utilities/boost_config.h"
#include "core_services/common_data_types.h"
#include "dataset_metadata_types.h"
#include <boost/thread/future.hpp>
#include <string>
#include <vector>
#include <memory>

namespace oscean::core_services::metadata {

// 导入类型定义
using IntelligentRecognitionResult = oscean::core_services::metadata::IntelligentRecognitionResult;
using RecognitionConfiguration = oscean::core_services::metadata::RecognitionConfiguration;
using DataType = oscean::core_services::metadata::DataType;
using VariableInfo = oscean::core_services::metadata::VariableInfo;
using AsyncResult = oscean::core_services::metadata::AsyncResult;

/**
 * @brief 智能元数据识别器接口
 * 
 * 🎯 负责智能识别和分类数据文件
 * ✅ AI驱动的内容识别
 * ✅ 变量映射和分类
 * ✅ 数据质量评估
 * ✅ 异步批量处理
 */
class IMetadataRecognizer {
public:
    virtual ~IMetadataRecognizer() = default;

    /**
     * @brief 智能识别单个文件
     * @param filePath 文件路径
     * @param config 识别配置
     * @return 异步识别结果
     */
    virtual boost::future<AsyncResult<IntelligentRecognitionResult>> recognizeFileAsync(
        const std::string& filePath,
        const RecognitionConfiguration& config = {}
    ) = 0;

    /**
     * @brief 批量智能识别
     * @param filePaths 文件路径列表
     * @param config 识别配置
     * @return 异步批量识别结果
     */
    virtual boost::future<AsyncResult<std::vector<IntelligentRecognitionResult>>> recognizeBatchAsync(
        const std::vector<std::string>& filePaths,
        const RecognitionConfiguration& config = {}
    ) = 0;

    /**
     * @brief 训练识别模型
     * @param trainingData 训练数据集
     * @return 异步训练结果
     */
    virtual boost::future<AsyncResult<void>> trainModelAsync(
        const std::vector<std::pair<std::string, DataType>>& trainingData
    ) = 0;

    /**
     * @brief 更新变量映射规则
     * @param mappingRules 新的映射规则
     * @return 异步更新结果
     */
    virtual boost::future<AsyncResult<void>> updateVariableMappingAsync(
        const std::map<std::string, std::vector<std::string>>& mappingRules
    ) = 0;

    /**
     * @brief 评估识别性能
     * @param testSet 测试数据集
     * @return 异步性能评估结果
     */
    virtual boost::future<AsyncResult<double>> evaluatePerformanceAsync(
        const std::vector<std::pair<std::string, DataType>>& testSet
    ) = 0;

    /**
     * @brief 获取识别器统计信息
     * @return 异步统计结果
     */
    virtual boost::future<AsyncResult<std::map<std::string, double>>> getRecognitionStatisticsAsync() = 0;
};

} // namespace oscean::core_services::metadata 