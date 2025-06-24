#pragma once

/**
 * @file gdal_vector_processor.h
 * @brief GDAL矢量数据专用处理器 - 负责矢量数据的核心处理逻辑
 */

#include "core_services/common_data_types.h"
#include "gdal_common_types.h"
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace oscean::core_services::data_access::readers::impl::gdal {

/**
 * @brief GDAL矢量数据专用处理器
 */
class GdalVectorProcessor {
public:
    explicit GdalVectorProcessor(GDALDataset* dataset);
    ~GdalVectorProcessor() = default;

    GdalVectorProcessor(const GdalVectorProcessor&) = delete;
    GdalVectorProcessor& operator=(const GdalVectorProcessor&) = delete;
    GdalVectorProcessor(GdalVectorProcessor&&) = delete;
    GdalVectorProcessor& operator=(GdalVectorProcessor&&) = delete;

    // =============================================================================
    // 图层基础操作
    // =============================================================================
    
    /**
     * @brief 获取图层名称列表
     */
    std::vector<std::string> getLayerNames() const;
    
    /**
     * @brief 获取图层对象
     */
    OGRLayer* getLayer(const std::string& layerName) const;
    
    /**
     * @brief 获取图层要素数量
     */
    size_t getFeatureCount(const std::string& layerName) const;
    
    /**
     * @brief 获取图层几何类型
     */
    std::string getGeometryType(const std::string& layerName) const;

    // =============================================================================
    // 数据读取功能
    // =============================================================================
    
    /**
     * @brief 读取图层数据为GridData格式
     */
    std::shared_ptr<oscean::core_services::GridData> readLayerDataAdvanced(
        const std::string& layerName,
        const std::optional<oscean::core_services::BoundingBox>& bounds = std::nullopt);
    
    /**
     * @brief 空间查询
     */
    std::shared_ptr<oscean::core_services::GridData> spatialQuery(
        const std::string& layerName,
        const oscean::core_services::BoundingBox& bounds,
        const std::string& spatialRelation = "INTERSECTS");
    
    /**
     * @brief 属性查询
     */
    std::shared_ptr<oscean::core_services::GridData> attributeQuery(
        const std::string& layerName,
        const std::string& whereClause);

    // =============================================================================
    // 元数据功能
    // =============================================================================
    
    /**
     * @brief 获取图层字段信息
     */
    std::vector<std::map<std::string, std::string>> getFieldInfo(const std::string& layerName) const;
    
    /**
     * @brief 加载图层元数据
     */
    std::vector<oscean::core_services::MetadataEntry> loadLayerMetadataAdvanced(const std::string& layerName) const;
    
    /**
     * @brief 获取图层边界框
     */
    std::optional<oscean::core_services::BoundingBox> getLayerBounds(const std::string& layerName) const;

    // =============================================================================
    // 流式处理
    // =============================================================================
    
    /**
     * @brief 流式读取要素
     */
    void streamFeatures(
        const std::string& layerName,
        const std::optional<oscean::core_services::BoundingBox>& bounds,
        std::function<bool(const std::vector<std::map<std::string, std::string>>&)> processor) const;

private:
    // =============================================================================
    // 私有方法
    // =============================================================================
    
    /**
     * @brief 验证图层名称
     */
    bool validateLayerName(const std::string& layerName) const;
    
    /**
     * @brief 将OGR要素转换为属性映射
     */
    std::map<std::string, std::string> featureToAttributes(OGRFeature* feature) const;
    
    /**
     * @brief 将要素几何转换为WKT格式
     */
    std::string geometryToWKT(OGRGeometry* geometry) const;
    
    /**
     * @brief 应用空间过滤器
     */
    void applySpatialFilter(OGRLayer* layer, const oscean::core_services::BoundingBox& bounds) const;
    
    /**
     * @brief 应用属性过滤器
     */
    bool applyAttributeFilter(OGRLayer* layer, const std::string& whereClause) const;
    
    /**
     * @brief 将要素集合转换为GridData
     */
    std::shared_ptr<oscean::core_services::GridData> featuresToGridData(
        const std::vector<std::map<std::string, std::string>>& features,
        const std::string& layerName) const;
    
    /**
     * @brief 获取图层定义
     */
    OGRFeatureDefn* getLayerDefinition(const std::string& layerName) const;

private:
    GDALDataset* dataset_; // Non-owning pointer
};

} // namespace oscean::core_services::data_access::readers::impl::gdal 