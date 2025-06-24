/**
 * @file gdal_vector_processor.cpp
 * @brief GDAL矢量数据专用处理器实现
 */

#include "gdal_vector_processor.h"
#include "common_utils/utilities/logging_utils.h"
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <stdexcept>
#include <sstream>
#include <algorithm>

namespace oscean::core_services::data_access::readers::impl::gdal {

GdalVectorProcessor::GdalVectorProcessor(GDALDataset* dataset) : dataset_(dataset) {
    if (!dataset_) {
        throw std::invalid_argument("GdalVectorProcessor requires a non-null GDALDataset.");
    }
    
    // 验证数据集包含矢量图层
    if (dataset_->GetLayerCount() == 0) {
        throw std::invalid_argument("Dataset contains no vector layers.");
    }
    
    LOG_INFO("GdalVectorProcessor初始化成功，图层数量: {}", dataset_->GetLayerCount());
}

// =============================================================================
// 图层基础操作
// =============================================================================

std::vector<std::string> GdalVectorProcessor::getLayerNames() const {
    std::vector<std::string> layerNames;
    if (!dataset_) {
        return layerNames;
    }
    
    int layerCount = dataset_->GetLayerCount();
    layerNames.reserve(layerCount);
    
    for (int i = 0; i < layerCount; ++i) {
        OGRLayer* layer = dataset_->GetLayer(i);
        if (layer) {
            layerNames.push_back(layer->GetName());
        }
    }
    
    LOG_DEBUG("获取图层名称列表，数量: {}", layerNames.size());
    return layerNames;
}

OGRLayer* GdalVectorProcessor::getLayer(const std::string& layerName) const {
    if (!dataset_) {
        return nullptr;
    }
    
    // 首先尝试按名称获取
    OGRLayer* layer = dataset_->GetLayerByName(layerName.c_str());
    if (layer) {
        return layer;
    }
    
    // 尝试按索引获取（如果layerName是数字）
    try {
        int index = std::stoi(layerName);
        if (index >= 0 && index < dataset_->GetLayerCount()) {
            return dataset_->GetLayer(index);
        }
    } catch (const std::exception&) {
        // 不是有效的数字索引，继续
    }
    
    LOG_WARN("未找到图层: {}", layerName);
    return nullptr;
}

size_t GdalVectorProcessor::getFeatureCount(const std::string& layerName) const {
    OGRLayer* layer = getLayer(layerName);
    if (!layer) {
        return 0;
    }
    
    // 使用GDAL的快速计数方法
    GIntBig count = layer->GetFeatureCount();
    if (count < 0) {
        // 如果快速计数不可用，手动计数
        layer->ResetReading();
        count = 0;
        OGRFeature* feature;
        while ((feature = layer->GetNextFeature()) != nullptr) {
            count++;
            OGRFeature::DestroyFeature(feature);
        }
        layer->ResetReading();
    }
    
    return static_cast<size_t>(std::max(static_cast<GIntBig>(0), count));
}

std::string GdalVectorProcessor::getGeometryType(const std::string& layerName) const {
    OGRLayer* layer = getLayer(layerName);
    if (!layer) {
        return "Unknown";
    }
    
    OGRFeatureDefn* layerDefn = layer->GetLayerDefn();
    if (!layerDefn) {
        return "Unknown";
    }
    
    OGRwkbGeometryType geomType = layerDefn->GetGeomType();
    const char* typeName = OGRGeometryTypeToName(geomType);
    return typeName ? std::string(typeName) : "Unknown";
}

// =============================================================================
// 数据读取功能
// =============================================================================

std::shared_ptr<oscean::core_services::GridData> GdalVectorProcessor::readLayerDataAdvanced(
    const std::string& layerName,
    const std::optional<oscean::core_services::BoundingBox>& bounds) {
    
    OGRLayer* layer = getLayer(layerName);
    if (!layer) {
        LOG_ERROR("无法找到图层: {}", layerName);
        return nullptr;
    }
    
    // 应用空间过滤器
    if (bounds) {
        applySpatialFilter(layer, *bounds);
    }
    
    // 读取所有要素
    std::vector<std::map<std::string, std::string>> features;
    layer->ResetReading();
    
    OGRFeature* feature;
    while ((feature = layer->GetNextFeature()) != nullptr) {
        auto attributes = featureToAttributes(feature);
        features.push_back(attributes);
        OGRFeature::DestroyFeature(feature);
    }
    
    // 清除空间过滤器
    layer->SetSpatialFilter(nullptr);
    
    LOG_INFO("读取图层 {} 完成，要素数量: {}", layerName, features.size());
    return featuresToGridData(features, layerName);
}

std::shared_ptr<oscean::core_services::GridData> GdalVectorProcessor::spatialQuery(
    const std::string& layerName,
    const oscean::core_services::BoundingBox& bounds,
    const std::string& spatialRelation) {
    
    // 目前只支持INTERSECTS关系
    if (spatialRelation != "INTERSECTS") {
        LOG_WARN("暂不支持空间关系: {}，使用INTERSECTS", spatialRelation);
    }
    
    return readLayerDataAdvanced(layerName, bounds);
}

std::shared_ptr<oscean::core_services::GridData> GdalVectorProcessor::attributeQuery(
    const std::string& layerName,
    const std::string& whereClause) {
    
    OGRLayer* layer = getLayer(layerName);
    if (!layer) {
        LOG_ERROR("无法找到图层: {}", layerName);
        return nullptr;
    }
    
    // 应用属性过滤器
    if (!applyAttributeFilter(layer, whereClause)) {
        LOG_ERROR("属性查询失败: {}", whereClause);
        return nullptr;
    }
    
    // 读取过滤后的要素
    std::vector<std::map<std::string, std::string>> features;
    layer->ResetReading();
    
    OGRFeature* feature;
    while ((feature = layer->GetNextFeature()) != nullptr) {
        auto attributes = featureToAttributes(feature);
        features.push_back(attributes);
        OGRFeature::DestroyFeature(feature);
    }
    
    // 清除属性过滤器
    layer->SetAttributeFilter(nullptr);
    
    LOG_INFO("属性查询完成，图层: {}，条件: {}，结果数量: {}", 
             layerName, whereClause, features.size());
    
    return featuresToGridData(features, layerName);
}

// =============================================================================
// 元数据功能
// =============================================================================

std::vector<std::map<std::string, std::string>> GdalVectorProcessor::getFieldInfo(const std::string& layerName) const {
    std::vector<std::map<std::string, std::string>> fieldInfo;
    
    OGRFeatureDefn* layerDefn = getLayerDefinition(layerName);
    if (!layerDefn) {
        return fieldInfo;
    }
    
    int fieldCount = layerDefn->GetFieldCount();
    fieldInfo.reserve(fieldCount);
    
    for (int i = 0; i < fieldCount; ++i) {
        OGRFieldDefn* fieldDefn = layerDefn->GetFieldDefn(i);
        if (!fieldDefn) continue;
        
        std::map<std::string, std::string> field;
        field["name"] = fieldDefn->GetNameRef();
        field["type"] = OGRFieldDefn::GetFieldTypeName(fieldDefn->GetType());
        field["width"] = std::to_string(fieldDefn->GetWidth());
        field["precision"] = std::to_string(fieldDefn->GetPrecision());
        field["nullable"] = fieldDefn->IsNullable() ? "true" : "false";
        
        fieldInfo.push_back(field);
    }
    
    return fieldInfo;
}

std::vector<oscean::core_services::MetadataEntry> GdalVectorProcessor::loadLayerMetadataAdvanced(const std::string& layerName) const {
    std::vector<oscean::core_services::MetadataEntry> metadata;
    
    OGRLayer* layer = getLayer(layerName);
    if (!layer) {
        return metadata;
    }
    
    // 基本信息
    metadata.emplace_back("layer_name", layerName);
    metadata.emplace_back("geometry_type", getGeometryType(layerName));
    metadata.emplace_back("feature_count", std::to_string(getFeatureCount(layerName)));
    
    // 空间参考系统
    OGRSpatialReference* srs = layer->GetSpatialRef();
    if (srs) {
        char* wktString = nullptr;
        if (srs->exportToWkt(&wktString) == OGRERR_NONE && wktString) {
            metadata.emplace_back("spatial_reference", wktString);
            CPLFree(wktString);
        }
        
        const char* authName = srs->GetAuthorityName(nullptr);
        const char* authCode = srs->GetAuthorityCode(nullptr);
        if (authName && authCode) {
            metadata.emplace_back("epsg_code", std::string(authName) + ":" + authCode);
        }
    }
    
    // 边界框
    auto bounds = getLayerBounds(layerName);
    if (bounds) {
        metadata.emplace_back("extent_minx", std::to_string(bounds->minX));
        metadata.emplace_back("extent_miny", std::to_string(bounds->minY));
        metadata.emplace_back("extent_maxx", std::to_string(bounds->maxX));
        metadata.emplace_back("extent_maxy", std::to_string(bounds->maxY));
    }
    
    // 字段数量
    OGRFeatureDefn* layerDefn = layer->GetLayerDefn();
    if (layerDefn) {
        metadata.emplace_back("field_count", std::to_string(layerDefn->GetFieldCount()));
    }
    
    return metadata;
}

std::optional<oscean::core_services::BoundingBox> GdalVectorProcessor::getLayerBounds(const std::string& layerName) const {
    OGRLayer* layer = getLayer(layerName);
    if (!layer) {
        return std::nullopt;
    }
    
    OGREnvelope envelope;
    if (layer->GetExtent(&envelope) != OGRERR_NONE) {
        return std::nullopt;
    }
    
    oscean::core_services::BoundingBox bounds;
    bounds.minX = envelope.MinX;
    bounds.maxX = envelope.MaxX;
    bounds.minY = envelope.MinY;
    bounds.maxY = envelope.MaxY;
    
    return bounds;
}

// =============================================================================
// 流式处理
// =============================================================================

void GdalVectorProcessor::streamFeatures(
    const std::string& layerName,
    const std::optional<oscean::core_services::BoundingBox>& bounds,
    std::function<bool(const std::vector<std::map<std::string, std::string>>&)> processor) const {
    
    OGRLayer* layer = getLayer(layerName);
    if (!layer) {
        LOG_ERROR("无法找到图层: {}", layerName);
        return;
    }
    
    // 应用空间过滤器
    if (bounds) {
        applySpatialFilter(layer, *bounds);
    }
    
    layer->ResetReading();
    
    const size_t batchSize = 1000; // 批处理大小
    std::vector<std::map<std::string, std::string>> batch;
    batch.reserve(batchSize);
    
    OGRFeature* feature;
    while ((feature = layer->GetNextFeature()) != nullptr) {
        auto attributes = featureToAttributes(feature);
        batch.push_back(attributes);
        OGRFeature::DestroyFeature(feature);
        
        // 当批次满了时，调用处理器
        if (batch.size() >= batchSize) {
            if (!processor(batch)) {
                break; // 处理器要求停止
            }
            batch.clear();
        }
    }
    
    // 处理剩余的要素
    if (!batch.empty()) {
        processor(batch);
    }
    
    // 清除空间过滤器
    layer->SetSpatialFilter(nullptr);
    
    LOG_INFO("图层 {} 流式处理完成", layerName);
}

// =============================================================================
// 私有方法实现
// =============================================================================

bool GdalVectorProcessor::validateLayerName(const std::string& layerName) const {
    return getLayer(layerName) != nullptr;
}

std::map<std::string, std::string> GdalVectorProcessor::featureToAttributes(OGRFeature* feature) const {
    std::map<std::string, std::string> attributes;
    
    if (!feature) {
        return attributes;
    }
    
    OGRFeatureDefn* layerDefn = feature->GetDefnRef();
    if (!layerDefn) {
        return attributes;
    }
    
    // 添加FID
    attributes["fid"] = std::to_string(feature->GetFID());
    
    // 添加几何信息
    OGRGeometry* geometry = feature->GetGeometryRef();
    if (geometry) {
        attributes["geometry_type"] = geometry->getGeometryName();
        attributes["geometry_wkt"] = geometryToWKT(geometry);
    }
    
    // 添加属性字段
    int fieldCount = layerDefn->GetFieldCount();
    for (int i = 0; i < fieldCount; ++i) {
        OGRFieldDefn* fieldDefn = layerDefn->GetFieldDefn(i);
        if (!fieldDefn) continue;
        
        const char* fieldName = fieldDefn->GetNameRef();
        if (!fieldName) continue;
        
        if (feature->IsFieldSet(i)) {
            const char* fieldValue = feature->GetFieldAsString(i);
            attributes[fieldName] = fieldValue ? fieldValue : "";
        } else {
            attributes[fieldName] = "";
        }
    }
    
    return attributes;
}

std::string GdalVectorProcessor::geometryToWKT(OGRGeometry* geometry) const {
    if (!geometry) {
        return "";
    }
    
    char* wktString = nullptr;
    if (geometry->exportToWkt(&wktString) == OGRERR_NONE && wktString) {
        std::string result(wktString);
        CPLFree(wktString);
        return result;
    }
    
    return "";
}

void GdalVectorProcessor::applySpatialFilter(OGRLayer* layer, const oscean::core_services::BoundingBox& bounds) const {
    if (!layer) {
        return;
    }
    
    layer->SetSpatialFilterRect(bounds.minX, bounds.minY, bounds.maxX, bounds.maxY);
}

bool GdalVectorProcessor::applyAttributeFilter(OGRLayer* layer, const std::string& whereClause) const {
    if (!layer) {
        return false;
    }
    
    OGRErr result = layer->SetAttributeFilter(whereClause.c_str());
    return result == OGRERR_NONE;
}

std::shared_ptr<oscean::core_services::GridData> GdalVectorProcessor::featuresToGridData(
    const std::vector<std::map<std::string, std::string>>& features,
    const std::string& layerName) const {
    
    auto gridData = std::make_shared<oscean::core_services::GridData>();
    
    // 设置基本信息
    gridData->definition.cols = 1;
    gridData->definition.rows = features.size();
    gridData->dataType = oscean::core_services::DataType::String;
    
    // 设置元数据
    gridData->metadata["layer_name"] = layerName;
    gridData->metadata["data_type"] = "vector_features";
    gridData->metadata["feature_count"] = std::to_string(features.size());
    
    // 将要素数据序列化为JSON字符串并存储在data中
    std::ostringstream jsonStream;
    jsonStream << "[";
    
    for (size_t i = 0; i < features.size(); ++i) {
        if (i > 0) jsonStream << ",";
        jsonStream << "{";
        
        bool first = true;
        for (const auto& [key, value] : features[i]) {
            if (!first) jsonStream << ",";
            jsonStream << "\"" << key << "\":\"" << value << "\"";
            first = false;
        }
        
        jsonStream << "}";
    }
    
    jsonStream << "]";
    
    std::string jsonString = jsonStream.str();
    auto& buffer = gridData->getUnifiedBuffer();
    buffer.resize(jsonString.size());
    std::memcpy(buffer.data(), jsonString.data(), jsonString.size());
    
    LOG_DEBUG("转换要素为GridData完成，图层: {}，要素数量: {}", layerName, features.size());
    return gridData;
}

OGRFeatureDefn* GdalVectorProcessor::getLayerDefinition(const std::string& layerName) const {
    OGRLayer* layer = getLayer(layerName);
    return layer ? layer->GetLayerDefn() : nullptr;
}

} // namespace oscean::core_services::data_access::readers::impl::gdal 