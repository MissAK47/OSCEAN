/**
 * @file i_data_reader_impl.h
 * @brief 定义数据读取器实现接口
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>

// 包含公共数据类型定义
#include "core_services/common_data_types.h"
// 添加对i_raw_data_access_service.h的引用，因为这里定义了VariableDataVariant
#include "core_services/data_access/i_raw_data_access_service.h"

// 包含核心接口定义
// #include "../../../../../core_service_interfaces/include/core_services/data_access/i_data_reader.h" // 用户确认此文件已删除，且其内容与 IDataReaderImpl 重复或被其取代

namespace oscean {
namespace core_services {
namespace data_access {

/**
 * @brief 数据读取器内部接口定义
 */
class IDataReaderImpl {
public:
    /**
     * @brief 虚析构函数
     */
    virtual ~IDataReaderImpl() = default;
    
    /**
     * @brief 打开数据源
     * @param filePath 数据文件路径
     * @param targetCRS 可选的目标坐标系
     * @return 是否成功打开
     */
    virtual bool open(const std::string& filePath, 
                     const std::optional<std::string>& targetCRS = std::nullopt) = 0;
    
    /**
     * @brief 打开已设置路径的数据源
     * @return 是否成功打开
     */
    virtual bool open() = 0;
    
    /**
     * @brief 关闭数据源
     */
    virtual void close() = 0;
    
    /**
     * @brief 检查数据源是否已打开
     * @return 是否已打开
     */
    virtual bool isOpen() const = 0;
    
    /**
     * @brief 获取文件路径
     * @return 文件路径
     */
    virtual std::string getFilePath() const = 0;
    
    /**
     * @brief 列出所有可读取的数据变量名称
     * @return 变量名列表
     */
    virtual std::vector<std::string> listDataVariableNames() const = 0;
    
    /**
     * @brief 列出所有变量名称（包括非数据变量）
     * @return 变量名列表
     */
    virtual std::vector<std::string> getVariableNames() const = 0;
    
    /**
     * @brief 获取原生坐标参考系统信息
     * @return 坐标参考系统信息
     */
    virtual std::optional<oscean::core_services::CRSInfo> getNativeCrs() const = 0;
    
    /**
     * @brief 获取原生边界框
     * @return 边界框
     */
    virtual oscean::core_services::BoundingBox getNativeBoundingBox() const = 0;
    
    /**
     * @brief 获取原生时间范围
     * @return 时间范围
     */
    virtual std::optional<oscean::core_services::TimeRange> getNativeTimeRange() const = 0;
    
    /**
     * @brief 获取垂直层次
     * @return 垂直层次值列表
     */
    virtual std::vector<double> getVerticalLevels() const = 0;
    
    /**
     * @brief 获取全局属性
     * @return 元数据条目列表
     */
    virtual std::vector<oscean::core_services::MetadataEntry> getGlobalAttributes() const = 0;
    
    /**
     * @brief 获取变量的维度定义
     * @param variableName 变量名
     * @return 维度定义列表
     */
    virtual std::optional<std::vector<oscean::core_services::DimensionDefinition>> getVariableDimensions(
        const std::string& variableName) const = 0;
    
    /**
     * @brief 获取变量的元数据
     * @param variableName 变量名
     * @return 元数据条目列表
     */
    virtual std::optional<std::vector<oscean::core_services::MetadataEntry>> getVariableMetadata(
        const std::string& variableName) const = 0;
    
    /**
     * @brief 读取网格数据
     * @param variableName 变量名
     * @param sliceRanges 可选的切片范围
     * @param targetResolution 可选的目标分辨率
     * @param targetCRS 可选的目标坐标系
     * @param resampleAlgo 重采样算法
     * @param outputBounds 可选的输出边界
     * @return 网格数据
     */
    virtual std::shared_ptr<oscean::core_services::GridData> readGridData(
        const std::string& variableName,
        const std::vector<oscean::core_services::IndexRange>& sliceRanges = {},
        const std::optional<std::vector<double>>& targetResolution = std::nullopt,
        const std::optional<oscean::core_services::CRSInfo>& targetCRS = std::nullopt,
        oscean::core_services::ResampleAlgorithm resampleAlgo = oscean::core_services::ResampleAlgorithm::NEAREST,
        const std::optional<oscean::core_services::BoundingBox>& outputBounds = std::nullopt) = 0;
    
    /**
     * @brief 读取矢量特征集合
     * @param layerName 图层名
     * @param targetCRS 可选的目标坐标系
     * @param filterBoundingBox 可选的过滤边界框
     * @param bboxCRS 可选的边界框坐标系
     * @return 特征集合
     */
    virtual oscean::core_services::FeatureCollection readFeatureCollection(
        const std::string& layerName,
        const std::optional<oscean::core_services::CRSInfo>& targetCRS = std::nullopt,
        const std::optional<oscean::core_services::BoundingBox>& filterBoundingBox = std::nullopt,
        const std::optional<oscean::core_services::CRSInfo>& bboxCRS = std::nullopt) = 0;
    
    /**
     * @brief 验证数据集
     * @param comprehensive 是否进行全面验证
     * @return 问题列表
     */
    virtual std::vector<oscean::core_services::DatasetIssue> validateDataset(bool comprehensive = false) const = 0;

    /**
     * @brief 读取变量数据
     * @param variableName 变量名
     * @param startIndices 各维度的起始索引
     * @param counts 各维度的元素数量
     * @return 包含变量数据的变体类型
     */
    virtual oscean::core_services::VariableDataVariant readVariableData(
        const std::string& variableName,
        const std::vector<size_t>& startIndices,
        const std::vector<size_t>& counts) = 0;
};

/** // Temporarily commenting out DataReaderAdapter as it depends on the deleted IDataReader
 * @brief 创建一个适配器类，从IDataReaderImpl适配到IDataReader
 * 
 * 这个类允许我们使用现有的IDataReaderImpl实现来提供IDataReader接口。
 * 
class DataReaderAdapter : public oscean::core_services::IDataReader {
private:
    std::shared_ptr<IDataReaderImpl> m_impl;

public:
    explicit DataReaderAdapter(std::shared_ptr<IDataReaderImpl> impl) : m_impl(impl) {}
    
    ~DataReaderAdapter() override = default;
    
    bool open(const std::string& filePath,
              const std::optional<std::string>& targetCRS = std::nullopt) override {
        return m_impl->open(filePath, targetCRS);
    }
    
    bool open() override {
        return m_impl->open();
    }
    
    void close() override {
        m_impl->close();
    }
    
    bool isOpen() const override {
        return m_impl->isOpen();
    }
    
    std::string getFilePath() const override {
        return m_impl->getFilePath();
    }
    
    std::vector<std::string> listDataVariableNames() const override {
        return m_impl->listDataVariableNames();
    }
    
    std::vector<std::string> getVariableNames() const override {
        // IDataReaderImpl has getVariableNames(), use it.
        return m_impl->getVariableNames(); 
    }

    std::optional<oscean::core_services::CRSInfo> getNativeCrs() const override {
        return m_impl->getNativeCrs();
    }

    oscean::core_services::BoundingBox getNativeBoundingBox() const override {
        return m_impl->getNativeBoundingBox();
    }

    std::optional<oscean::core_services::TimeRange> getNativeTimeRange() const override {
        return m_impl->getNativeTimeRange();
    }

    std::vector<double> getVerticalLevels() const override {
        return m_impl->getVerticalLevels();
    }

    std::vector<oscean::core_services::MetadataEntry> getGlobalAttributes() const override {
        return m_impl->getGlobalAttributes();
    }

    std::optional<std::vector<oscean::core_services::DimensionDefinition>> getVariableDimensions(
        const std::string& variableName) const override {
        return m_impl->getVariableDimensions(variableName);
    }

    std::optional<std::vector<oscean::core_services::MetadataEntry>> getVariableMetadata(
        const std::string& variableName) const override {
        return m_impl->getVariableMetadata(variableName);
    }

    std::shared_ptr<oscean::core_services::GridData> readGridData(
        const std::string& variableName,
        std::optional<oscean::core_services::CRSInfo> targetCRS = std::nullopt, // Corrected parameter name from IDataReader
        std::optional<std::pair<double, double>> targetResolution = std::nullopt,
        std::optional<oscean::core_services::BoundingBox> outputBounds = std::nullopt,
        const std::vector<oscean::core_services::IndexRange>& sliceRanges = {},
        oscean::core_services::ResampleAlgorithm resampleAlgo = oscean::core_services::ResampleAlgorithm::NEAREST) override {
        return m_impl->readGridData(variableName, sliceRanges, targetResolution, targetCRS, resampleAlgo, outputBounds);
    }

    oscean::core_services::FeatureCollection readFeatureCollection(
        const std::string& layerName,
        const std::optional<oscean::core_services::CRSInfo>& targetCRS = std::nullopt,
        const std::optional<oscean::core_services::BoundingBox>& filterBoundingBox = std::nullopt,
        const std::optional<oscean::core_services::CRSInfo>& bboxCRS = std::nullopt) override {
        return m_impl->readFeatureCollection(layerName, targetCRS, filterBoundingBox, bboxCRS);
    }

    std::vector<oscean::core_services::DatasetIssue> validateDataset(bool comprehensive = false) const override {
        return m_impl->validateDataset(comprehensive);
    }
};
**/

} // namespace data_access
} // namespace core_services
} // namespace oscean 