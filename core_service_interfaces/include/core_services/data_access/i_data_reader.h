#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <boost/optional.hpp>

#include "core_services/common_data_types.h"

namespace oscean {
namespace core_services {

// 导入common_data_types.h中定义的类型
using DimensionDefinition = oscean::core_services::DimensionDefinition;
using ResampleAlgorithm = oscean::core_services::ResampleAlgorithm;
using DatasetIssueType = oscean::core_services::DatasetIssueType;
using DatasetIssue = oscean::core_services::DatasetIssue;
using GridData = oscean::core_services::GridData;
using MetadataEntry = oscean::core_services::MetadataEntry;
using CRSInfo = oscean::core_services::CRSInfo;
using BoundingBox = oscean::core_services::BoundingBox;
using TimeRange = oscean::core_services::TimeRange;
using IndexRange = oscean::core_services::IndexRange;
using FeatureCollection = oscean::core_services::FeatureCollection;

/**
 * @brief 数据读取器接口
 * 
 * 定义所有数据读取器必须实现的基本功能，包括打开/关闭文件、
 * 读取数据、获取元数据等。
 * 
 * 所有格式特定的读取器都应实现此接口。
 */
class IDataReader {
public:
    /**
     * @brief 析构函数
     */
    virtual ~IDataReader() = default;

    /**
     * @brief 打开数据源
     * @param filePath 数据文件路径
     * @param targetCRS 可选的目标坐标参考系统
     * @return 是否成功打开
     */
    virtual bool open(const std::string& filePath, 
                      const boost::optional<std::string>& targetCRS = boost::none) = 0;
    
    /**
     * @brief 打开已设置路径的数据源
     * @return 是否成功打开
     */
    virtual bool open() = 0;

    /**
     * @brief 关闭数据源并释放资源
     */
    virtual void close() = 0;

    /**
     * @brief 检查数据源是否已打开
     * @return 如果已打开返回true，否则返回false
     */
    virtual bool isOpen() const = 0;

    /**
     * @brief 获取关联的文件路径
     * @return 文件路径
     */
    virtual std::string getFilePath() const = 0;

    /**
     * @brief 列出数据源中可读取的数据变量/图层名称
     * @return 变量名称向量
     */
    virtual std::vector<std::string> listDataVariableNames() const = 0;

    /**
     * @brief 读取指定的网格数据变量，可选择读取子区域
     * @param variableName 要读取的变量名称
     * @param targetCRS 可选的目标坐标系统
     * @param targetResolution 可选的目标分辨率
     * @param outputBounds 可选的输出范围
     * @param sliceRanges 可选的切片范围
     * @param resampleAlgo 重采样算法
     * @return 包含数据和完整定义的GridData对象
     */
    virtual std::shared_ptr<GridData> readGridData(
        const std::string& variableName,
        boost::optional<CRSInfo> targetCRS = boost::none,
        boost::optional<std::pair<double, double>> targetResolution = boost::none,
        boost::optional<BoundingBox> outputBounds = boost::none,
        const std::vector<IndexRange>& sliceRanges = {},
        ResampleAlgorithm resampleAlgo = ResampleAlgorithm::NEAREST) = 0;

    /**
     * @brief 获取全局属性
     * @return 属性名称到值的映射
     */
    virtual std::vector<MetadataEntry> getGlobalAttributes() const = 0;

    /**
     * @brief 获取变量的维度定义
     * @param variableName 变量名称
     * @return 维度定义列表，如果不存在则返回std::nullopt
     */
    virtual boost::optional<std::vector<DimensionDefinition>> getVariableDimensions(
        const std::string& variableName) const = 0;

    /**
     * @brief 获取变量的元数据
     * @param variableName 变量名称
     * @return 元数据条目列表，如果不存在则返回std::nullopt
     */
    virtual boost::optional<std::vector<MetadataEntry>> getVariableMetadata(
        const std::string& variableName) const = 0;

    /**
     * @brief 获取数据的原生坐标参考系统
     * @return 坐标参考系统信息
     */
    virtual boost::optional<CRSInfo> getNativeCrs() const = 0;

    /**
     * @brief 获取数据的原生边界框
     * @return 边界框
     */
    virtual BoundingBox getNativeBoundingBox() const = 0;
    
    /**
     * @brief 获取数据的原生时间范围
     * @return 时间范围，如果不适用则返回std::nullopt
     */
    virtual boost::optional<TimeRange> getNativeTimeRange() const = 0;
    
    /**
     * @brief 获取垂直层级
     * @return 垂直坐标值列表，如果不适用则返回空列表
     */
    virtual std::vector<double> getVerticalLevels() const = 0;

    /**
     * @brief 读取特征集合（用于矢量数据）
     * @param layerName 图层名称
     * @param targetCRS 可选的目标坐标系统
     * @param filterBoundingBox 可选的过滤边界框
     * @param bboxCRS 可选的边界框坐标系统
     * @return 特征集合
     */
    virtual FeatureCollection readFeatureCollection(
        const std::string& layerName,
        const boost::optional<CRSInfo>& targetCRS = boost::none,
        const boost::optional<BoundingBox>& filterBoundingBox = boost::none,
        const boost::optional<CRSInfo>& bboxCRS = boost::none) {
        // 默认实现抛出异常，矢量读取器需要重写此方法
        throw std::runtime_error("readFeatureCollection方法未实现");
    }

    /**
     * @brief 读取默认特征集合（用于矢量数据）
     * @return 指向特征集合的共享指针
     */
    virtual std::shared_ptr<FeatureCollection> readFeatureCollection() {
        // 默认实现抛出异常，矢量读取器需要重写此方法
        throw std::runtime_error("readFeatureCollection方法未实现");
    }

    /**
     * @brief 验证数据集
     * @param comprehensive 是否执行全面验证
     * @return 数据集问题列表
     */
    virtual std::vector<DatasetIssue> validateDataset(bool comprehensive = false) const {
        // 默认实现返回空列表，具体读取器可以重写
        return {};
    }
    
    /**
     * @brief 获取所有变量名称
     * @return 变量名称向量
     */
    virtual std::vector<std::string> getVariableNames() const {
        // 默认实现调用listDataVariableNames
        return listDataVariableNames();
    }
};

} // namespace core_services
} // namespace oscean 