#pragma once

#include "core_services/data_access/i_data_reader.h"
#include "core_services/common_data_types.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <boost/thread/future.hpp>
#include <boost/optional.hpp>

namespace oscean {
namespace output {

/**
 * @class InMemoryDataReader
 * @brief 内存数据读取器，用于零磁盘I/O的数据切片和处理
 * 
 * 这个类允许直接从内存中的GridData对象构造IDataReader，
 * 提供了行子集、波段子集、空间和时间边界处理等功能，
 * 避免了不必要的磁盘I/O操作。
 */
class InMemoryDataReader : public oscean::core_services::IDataReader {
public:
    /**
     * @brief 简化构造函数，使用GridData的变量名
     * @param gridData 源GridData对象
     */
    explicit InMemoryDataReader(
        std::shared_ptr<oscean::core_services::GridData> gridData);
        
    /**
     * @brief 构造函数
     * @param gridData 源GridData对象
     * @param variableName 变量名称
     */
    explicit InMemoryDataReader(
        std::shared_ptr<oscean::core_services::GridData> gridData,
        const std::string& variableName);

    /**
     * @brief 析构函数
     */
    virtual ~InMemoryDataReader() = default;

    // ===== IDataReader接口实现 =====
    
    /**
     * @brief 打开数据源（对于内存读取器总是成功）
     */
    bool open(const std::string& filePath, 
              const boost::optional<std::string>& targetCRS = boost::none) override;
    
    /**
     * @brief 打开已设置路径的数据源
     */
    bool open() override;
    
    /**
     * @brief 关闭数据源
     */
    void close() override;
    
    /**
     * @brief 检查是否已打开
     */
    bool isOpen() const override;
    
    /**
     * @brief 获取文件路径（内存读取器返回虚拟路径）
     */
    std::string getFilePath() const override;

    /**
     * @brief 列出可用的数据变量名称
     */
    std::vector<std::string> listDataVariableNames() const override;

    /**
     * @brief 读取网格数据
     */
    std::shared_ptr<oscean::core_services::GridData> readGridData(
        const std::string& variableName,
        boost::optional<oscean::core_services::CRSInfo> targetCRS = boost::none,
        boost::optional<std::pair<double, double>> targetResolution = boost::none,
        boost::optional<oscean::core_services::BoundingBox> outputBounds = boost::none,
        const std::vector<oscean::core_services::IndexRange>& sliceRanges = {},
        oscean::core_services::ResampleAlgorithm resampleAlgo = oscean::core_services::ResampleAlgorithm::NEAREST) override;

    /**
     * @brief 读取特征集合（内存读取器不支持）
     */
    oscean::core_services::FeatureCollection readFeatureCollection(
        const std::string& layerName,
        const boost::optional<oscean::core_services::CRSInfo>& targetCRS = boost::none,
        const boost::optional<oscean::core_services::BoundingBox>& filterBoundingBox = boost::none,
        const boost::optional<oscean::core_services::CRSInfo>& bboxCRS = boost::none) override;

    /**
     * @brief 获取全局属性
     */
    std::vector<oscean::core_services::MetadataEntry> getGlobalAttributes() const override;

    /**
     * @brief 获取变量维度定义
     */
    boost::optional<std::vector<oscean::core_services::DimensionDefinition>> getVariableDimensions(
        const std::string& variableName) const override;

    /**
     * @brief 获取变量元数据
     */
    boost::optional<std::vector<oscean::core_services::MetadataEntry>> getVariableMetadata(
        const std::string& variableName) const override;

    /**
     * @brief 获取原生坐标参考系统
     */
    boost::optional<oscean::core_services::CRSInfo> getNativeCrs() const override;

    /**
     * @brief 获取原生边界框
     */
    oscean::core_services::BoundingBox getNativeBoundingBox() const override;

    /**
     * @brief 获取原生时间范围
     */
    boost::optional<oscean::core_services::TimeRange> getNativeTimeRange() const override;

    /**
     * @brief 获取垂直层级
     */
    std::vector<double> getVerticalLevels() const override;

    // ===== 专用切片方法 =====

    /**
     * @brief 创建行子集数据
     * @param variableName 变量名称
     * @param startRow 起始行
     * @param endRow 结束行（不包含）
     * @return 新的GridData子集
     */
    std::shared_ptr<oscean::core_services::GridData> createRowSubset(
        const std::string& variableName, 
        size_t startRow, 
        size_t endRow);

    /**
     * @brief 创建波段子集数据
     * @param variableName 变量名称
     * @param startBand 起始波段
     * @param endBand 结束波段（不包含）
     * @return 新的GridData子集
     */
    std::shared_ptr<oscean::core_services::GridData> createBandSubset(
        const std::string& variableName, 
        size_t startBand, 
        size_t endBand);

    /**
     * @brief 创建空间边界子集
     * @param variableName 变量名称
     * @param bounds 空间边界
     * @return 新的GridData子集
     */
    std::shared_ptr<oscean::core_services::GridData> createSpatialSubset(
        const std::string& variableName, 
        const oscean::core_services::BoundingBox& bounds);

    /**
     * @brief 创建时间范围子集
     * @param variableName 变量名称
     * @param timeRange 时间范围
     * @return 新的GridData子集
     */
    std::shared_ptr<oscean::core_services::GridData> createTemporalSubset(
        const std::string& variableName, 
        const oscean::core_services::TimeRange& timeRange);

private:
    std::shared_ptr<oscean::core_services::GridData> m_gridData;  ///< 源数据
    std::string m_variableName;                                   ///< 变量名称
    bool m_isOpen;                                                ///< 是否已打开

    /**
     * @brief 验证变量名称是否有效
     */
    void validateVariableName(const std::string& variableName) const;
};

} // namespace output
} // namespace oscean 