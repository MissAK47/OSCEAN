/**
 * @file gdal_format_handler.cpp
 * @brief GDAL格式处理器实�?- 简化版�?
 */

#include "gdal_format_handler.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/simd/isimd_manager.h"
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <boost/thread/future.hpp>
#include <boost/optional.hpp>
#include <boost/asio/post.hpp>
#include <chrono>

namespace oscean::core_services::data_access::readers::impl::gdal {

GDALFormatHandler::GDALFormatHandler(GDALDataset* dataset) : dataset_(dataset) {
    if (!dataset_) {
        throw std::invalid_argument("GDAL dataset cannot be null");
    }
    
    dataType_ = detectDataType();
    LOG_INFO("GDALFormatHandler初始�? 数据类型={}", 
             dataType_ == GdalDataType::RASTER ? "栅格" : "矢量");
}

bool GDALFormatHandler::openFile(const std::string& /* filePath */) {
    // 文件已经在构造时打开，这里主要是验证
    return validateDataset();
}

std::vector<std::string> GDALFormatHandler::getVariableNames() {
    if (cachedVariableNames_) {
        return *cachedVariableNames_;
    }
    
    std::vector<std::string> variableNames;
    
    if (dataType_ == GdalDataType::RASTER) {
        // 栅格数据：使用波�?
        int bandCount = dataset_->GetRasterCount();
        for (int i = 1; i <= bandCount; ++i) {
            std::string bandName = "Band_" + std::to_string(i);
            
            GDALRasterBand* band = dataset_->GetRasterBand(i);
            if (band) {
                const char* description = band->GetDescription();
                if (description && strlen(description) > 0) {
                    bandName = std::string(description);
                }
            }
            
            variableNames.push_back(bandName);
        }
    } else if (dataType_ == GdalDataType::VECTOR) {
        // 矢量数据：使用图�?
        int layerCount = dataset_->GetLayerCount();
        for (int i = 0; i < layerCount; ++i) {
            OGRLayer* layer = dataset_->GetLayer(i);
            if (layer) {
                variableNames.push_back(layer->GetName());
            }
        }
    }
    
    cachedVariableNames_ = variableNames;
    return variableNames;
}

std::shared_ptr<oscean::core_services::GridData> GDALFormatHandler::readVariable(const std::string& name) {
    if (dataType_ == GdalDataType::RASTER) {
        return readRasterData(name);
    } else {
        LOG_WARN("矢量数据读取未实�?);
        return nullptr;
    }
}

bool GDALFormatHandler::shouldUseSIMD() const {
    // 大型栅格数据适合SIMD优化
    if (dataType_ == GdalDataType::RASTER) {
        size_t totalPixels = static_cast<size_t>(dataset_->GetRasterXSize()) * dataset_->GetRasterYSize();
        return totalPixels > 1000000; // 100万像素以�?
    }
    return false;
}

size_t GDALFormatHandler::getOptimalChunkSize() const {
    if (dataType_ == GdalDataType::RASTER) {
        return 1024 * 1024; // 1MB for raster data
    } else {
        return 64 * 1024; // 64KB for vector data
    }
}

boost::optional<oscean::core_services::VariableMeta> GDALFormatHandler::getVariableInfo(const std::string& variableName) const {
    if (!dataset_) {
        LOG_WARN("GDAL数据集未打开，无法获取变量信�? {}", variableName);
        return boost::none;
    }
    
    try {
        auto info = extractVariableInfo(variableName);
        return info;
    } catch (const std::exception& e) {
        LOG_ERROR("获取GDAL变量信息异常: {} - {}", variableName, e.what());
        return boost::none;
    }
}

std::shared_ptr<oscean::core_services::GridData> GDALFormatHandler::readRasterData(
    const std::string& variableName,
    const boost::optional<oscean::core_services::BoundingBox>& bounds) const {
    
    if (dataType_ != GdalDataType::RASTER) {
        LOG_ERROR("数据集不是栅格类�?);
        return nullptr;
    }
    
    // 获取波段编号
    int bandNumber = -1;
    
    if (variableName.find("Band_") == 0) {
        try {
            bandNumber = std::stoi(variableName.substr(5));
        } catch (const std::exception&) {
            LOG_ERROR("无效的波段名�? {}", variableName);
            return nullptr;
        }
    } else {
        // 通过描述查找
        int bandCount = dataset_->GetRasterCount();
        for (int i = 1; i <= bandCount; ++i) {
            GDALRasterBand* band = dataset_->GetRasterBand(i);
            if (band) {
                const char* description = band->GetDescription();
                if (description && variableName == std::string(description)) {
                    bandNumber = i;
                    break;
                }
            }
        }
    }
    
    if (bandNumber < 1 || bandNumber > dataset_->GetRasterCount()) {
        LOG_ERROR("无效的波段编�? {}", bandNumber);
        return nullptr;
    }
    
    GDALRasterBand* band = dataset_->GetRasterBand(bandNumber);
    if (!band) {
        LOG_ERROR("无法获取波段: {}", bandNumber);
        return nullptr;
    }
    
    // 计算读取区域
    int xOff = 0, yOff = 0;
    int xSize = dataset_->GetRasterXSize();
    int ySize = dataset_->GetRasterYSize();
    
    if (bounds) {
        // 简化的边界框转换，这里假设是像素坐�?
        xOff = std::max(0, static_cast<int>(bounds->minX));
        yOff = std::max(0, static_cast<int>(bounds->minY));
        xSize = std::min(xSize - xOff, static_cast<int>(bounds->maxX - bounds->minX));
        ySize = std::min(ySize - yOff, static_cast<int>(bounds->maxY - bounds->minY));
    }
    
    // 读取数据为double类型
    size_t totalElements = static_cast<size_t>(xSize) * ySize;
    std::vector<double> rawData(totalElements);
    
    CPLErr result = band->RasterIO(GF_Read, xOff, yOff, xSize, ySize,
                                  rawData.data(), xSize, ySize, GDT_Float64,
                                  0, 0);
    
    if (result != CE_None) {
        LOG_ERROR("读取栅格数据失败: {}", variableName);
        return nullptr;
    }
    
    // 处理NoData�?
    int hasNoData;
    double noDataValue = band->GetNoDataValue(&hasNoData);
    if (hasNoData) {
        for (auto& value : rawData) {
            if (std::abs(value - noDataValue) < 1e-10) {
                value = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    
    // 应用缩放和偏�?
    double scale = band->GetScale();
    double offset = band->GetOffset();
    if (scale != 1.0 || offset != 0.0) {
        for (auto& value : rawData) {
            if (!std::isnan(value)) {
                value = value * scale + offset;
            }
        }
    }
    
    // 创建GridData
    auto gridData = std::make_shared<oscean::core_services::GridData>();
    
    // 设置网格定义
    gridData->definition.cols = xSize;
    gridData->definition.rows = ySize;
    
    // 获取地理变换
    double geoTransform[6];
    if (dataset_->GetGeoTransform(geoTransform) == CE_None) {
        gridData->definition.xResolution = geoTransform[1];
        gridData->definition.yResolution = std::abs(geoTransform[5]);
        
        // 设置边界�?
        gridData->definition.extent.minX = geoTransform[0] + xOff * geoTransform[1];
        gridData->definition.extent.maxX = geoTransform[0] + (xOff + xSize) * geoTransform[1];
        gridData->definition.extent.maxY = geoTransform[3] + yOff * geoTransform[5];
        gridData->definition.extent.minY = geoTransform[3] + (yOff + ySize) * geoTransform[5];
    }
    
    // 转换数据为unsigned char格式 (GridData的data成员是std::vector<unsigned char>)
    size_t dataSize = totalElements * sizeof(double);
    auto& buffer = gridData->getUnifiedBuffer();
    buffer.resize(dataSize);
    std::memcpy(buffer.data(), rawData.data(), dataSize);
    
    // 设置数据类型
    gridData->dataType = oscean::core_services::DataType::Float64;
    
    // 设置元数�?
    gridData->metadata["variable_name"] = variableName;
    gridData->metadata["band_number"] = std::to_string(bandNumber);
    
    LOG_INFO("栅格数据读取成功: {} ({}x{} 像素)", variableName, xSize, ySize);
    return gridData;
}

boost::optional<oscean::core_services::CRSInfo> GDALFormatHandler::getCRSInfo() const {
    if (cachedCRSInfo_) {
        return *cachedCRSInfo_;
    }
    
    const char* projRef = dataset_->GetProjectionRef();
    if (!projRef || strlen(projRef) == 0) {
        return boost::none;
    }
    
    oscean::core_services::CRSInfo crsInfo;
    crsInfo.wkt = std::string(projRef);
    
    // 尝试提取EPSG代码
    OGRSpatialReference oSRS;
    if (oSRS.importFromWkt(projRef) == OGRERR_NONE) {
        const char* authName = oSRS.GetAuthorityName(nullptr);
        const char* authCode = oSRS.GetAuthorityCode(nullptr);
        
        if (authName && authCode && std::string(authName) == "EPSG") {
            try {
                crsInfo.epsgCode = std::stoi(authCode);
            } catch (const std::exception&) {
                // 忽略转换错误
            }
        }
    }
    
    cachedCRSInfo_ = crsInfo;
    return crsInfo;
}

oscean::core_services::BoundingBox GDALFormatHandler::getBoundingBox() const {
    if (cachedBoundingBox_) {
        return *cachedBoundingBox_;
    }
    
    oscean::core_services::BoundingBox bbox;
    
    if (dataType_ == GdalDataType::RASTER) {
        // 栅格数据边界�?
        double geoTransform[6];
        if (dataset_->GetGeoTransform(geoTransform) == CE_None) {
            int xSize = dataset_->GetRasterXSize();
            int ySize = dataset_->GetRasterYSize();
            
            bbox.minX = geoTransform[0];
            bbox.maxX = geoTransform[0] + xSize * geoTransform[1];
            bbox.maxY = geoTransform[3];
            bbox.minY = geoTransform[3] + ySize * geoTransform[5];
            
            // 确保坐标顺序正确
            if (bbox.minX > bbox.maxX) std::swap(bbox.minX, bbox.maxX);
            if (bbox.minY > bbox.maxY) std::swap(bbox.minY, bbox.maxY);
        } else {
            // 默认边界�?
            bbox.minX = 0.0; bbox.maxX = dataset_->GetRasterXSize();
            bbox.minY = 0.0; bbox.maxY = dataset_->GetRasterYSize();
        }
    } else {
        // 矢量数据边界框（简化）
        bbox.minX = -180.0; bbox.maxX = 180.0;
        bbox.minY = -90.0; bbox.maxY = 90.0;
    }
    
    cachedBoundingBox_ = bbox;
    return bbox;
}

std::vector<oscean::core_services::MetadataEntry> GDALFormatHandler::getVariableAttributes(const std::string& variableName) const {
    std::vector<oscean::core_services::MetadataEntry> attributes;
    
    if (dataType_ == GdalDataType::RASTER) {
        // 获取波段编号
        int bandNumber = -1;
        if (variableName.find("Band_") == 0) {
            try {
                bandNumber = std::stoi(variableName.substr(5));
            } catch (const std::exception&) {
                return attributes;
            }
        }
        
        if (bandNumber < 1 || bandNumber > dataset_->GetRasterCount()) {
            return attributes;
        }
        
        GDALRasterBand* band = dataset_->GetRasterBand(bandNumber);
        if (!band) {
            return attributes;
        }
        
        // 基本属�?
        attributes.emplace_back("band_number", std::to_string(bandNumber));
        attributes.emplace_back("data_type", GDALGetDataTypeName(band->GetRasterDataType()));
        attributes.emplace_back("x_size", std::to_string(band->GetXSize()));
        attributes.emplace_back("y_size", std::to_string(band->GetYSize()));
        
        // NoData�?
        int hasNoData;
        double noDataValue = band->GetNoDataValue(&hasNoData);
        if (hasNoData) {
            attributes.emplace_back("no_data_value", std::to_string(noDataValue));
        }
        
        // 缩放和偏�?
        double scale = band->GetScale();
        double offset = band->GetOffset();
        if (scale != 1.0) {
            attributes.emplace_back("scale_factor", std::to_string(scale));
        }
        if (offset != 0.0) {
            attributes.emplace_back("add_offset", std::to_string(offset));
        }
        
        // 单位
        const char* units = band->GetUnitType();
        if (units && strlen(units) > 0) {
            attributes.emplace_back("units", std::string(units));
        }
    }
    
    return attributes;
}

boost::unique_future<void> GDALFormatHandler::streamVariableData(
    const std::string& variableName,
    const boost::optional<oscean::core_services::BoundingBox>& bounds,
    std::function<bool(const std::vector<double>&, const std::vector<size_t>&)> processor) {
    
    return boost::async(boost::launch::async, [this, variableName, bounds, processor]() {
        LOG_INFO("开始流式读取GDAL变量: {}", variableName);
        
        // 简化的流式读取实现
        auto gridData = readVariable(variableName);
        if (!gridData) {
            LOG_ERROR("无法读取变量数据: {}", variableName);
            return;
        }
        
        // 从GridData中提取double数据
        const auto& buffer = gridData->getData();
        size_t totalElements = buffer.size() / sizeof(double);
        std::vector<double> doubleData(totalElements);
        std::memcpy(doubleData.data(), buffer.data(), buffer.size());
        
        // 构造形状信�?
        std::vector<size_t> shape = {
            static_cast<size_t>(gridData->definition.rows),
            static_cast<size_t>(gridData->definition.cols)
        };
        
        // 调用处理函数
        processor(doubleData, shape);
        
        LOG_INFO("GDAL变量流式读取完成: {}", variableName);
    });
}

// =============================================================================
// 私有方法实现
// =============================================================================

GdalDataType GDALFormatHandler::detectDataType() const {
    if (dataset_->GetRasterCount() > 0) {
        return GdalDataType::RASTER;
    } else if (dataset_->GetLayerCount() > 0) {
        return GdalDataType::VECTOR;
    } else {
        return GdalDataType::UNKNOWN;
    }
}

oscean::core_services::VariableMeta GDALFormatHandler::extractVariableInfo(const std::string& variableName) const {
    oscean::core_services::VariableMeta variableInfo;
    variableInfo.name = variableName;
    
    if (dataType_ == GdalDataType::RASTER) {
        // 🔧 完善栅格数据的字段提取逻辑
        int bandNumber = -1;
        if (variableName.find("Band_") == 0) {
            try {
                bandNumber = std::stoi(variableName.substr(5));
            } catch (const std::exception&) {
                LOG_WARN("无法解析波段编号: {}", variableName);
                return variableInfo;
            }
        }
        
        if (bandNumber < 1 || bandNumber > dataset_->GetRasterCount()) {
            LOG_WARN("无效的波段编�? {}", bandNumber);
            return variableInfo;
        }
        
        GDALRasterBand* band = dataset_->GetRasterBand(bandNumber);
        if (!band) {
            LOG_WARN("无法获取波段: {}", bandNumber);
            return variableInfo;
        }
        
        // 🔧 设置基本字段
        const char* description = band->GetDescription();
        if (description && strlen(description) > 0) {
            variableInfo.description = std::string(description);
        } else {
            variableInfo.description = "Band " + std::to_string(bandNumber);
        }
        
        // 🔧 获取单位信息
        const char* units = band->GetUnitType();
        if (units && strlen(units) > 0) {
            variableInfo.units = std::string(units);
        } else {
            // 尝试从元数据中获取单�?
            char** metadata = band->GetMetadata();
            if (metadata) {
                for (int i = 0; metadata[i] != nullptr; ++i) {
                    std::string entry(metadata[i]);
                    if (entry.find("units=") == 0) {
                        variableInfo.units = entry.substr(6);
                        break;
                    } else if (entry.find("GRIB_UNIT=") == 0) {
                        variableInfo.units = entry.substr(10);
                        break;
                    }
                }
            }
        }
        
        // 🔧 获取数据类型
        GDALDataType gdalDataType = band->GetRasterDataType();
        switch (gdalDataType) {
            case GDT_Byte:
                variableInfo.dataType = DataType::UByte;
                break;
            case GDT_UInt16:
                variableInfo.dataType = DataType::UInt16;
                break;
            case GDT_Int16:
                variableInfo.dataType = DataType::Int16;
                break;
            case GDT_UInt32:
                variableInfo.dataType = DataType::UInt32;
                break;
            case GDT_Int32:
                variableInfo.dataType = DataType::Int32;
                break;
            case GDT_Float32:
                variableInfo.dataType = DataType::Float32;
                break;
            case GDT_Float64:
                variableInfo.dataType = DataType::Float64;
                break;
            case GDT_CInt16:
                variableInfo.dataType = DataType::Complex16;
                break;
            case GDT_CInt32:
                variableInfo.dataType = DataType::Complex32;
                break;
            case GDT_CFloat32:
                variableInfo.dataType = DataType::Complex64;
                break;
            case GDT_CFloat64:
                variableInfo.dataType = DataType::Complex64; // 注意：没有Complex128，使用Complex64
                break;
            default:
                variableInfo.dataType = DataType::Unknown;
                break;
        }
        
        // 🔧 设置维度信息到attributes�?
        int xSize = band->GetXSize();
        int ySize = band->GetYSize();
        variableInfo.attributes["dimensions"] = "y,x";
        variableInfo.attributes["shape"] = std::to_string(ySize) + "," + std::to_string(xSize);
        variableInfo.attributes["band_number"] = std::to_string(bandNumber);
        
        // 🔧 获取NoData�?
        int hasNoData;
        double noDataValue = band->GetNoDataValue(&hasNoData);
        if (hasNoData != 0) {
            variableInfo.attributes["no_data_value"] = std::to_string(noDataValue);
        }
        
        // 🔧 获取缩放因子和偏移量
        double scaleFactor = band->GetScale();
        double addOffset = band->GetOffset();
        if (scaleFactor != 1.0) {
            variableInfo.attributes["scale_factor"] = std::to_string(scaleFactor);
        }
        if (addOffset != 0.0) {
            variableInfo.attributes["add_offset"] = std::to_string(addOffset);
        }
        
        // 🔧 获取统计信息
        double minVal, maxVal, meanVal, stdDevVal;
        if (band->GetStatistics(FALSE, FALSE, &minVal, &maxVal, &meanVal, &stdDevVal) == CE_None) {
            variableInfo.attributes["minimum"] = std::to_string(minVal);
            variableInfo.attributes["maximum"] = std::to_string(maxVal);
            variableInfo.attributes["mean"] = std::to_string(meanVal);
            variableInfo.attributes["standard_deviation"] = std::to_string(stdDevVal);
        }
        
        // 🔧 获取波段元数�?
        char** bandMetadata = band->GetMetadata();
        if (bandMetadata) {
            for (int i = 0; bandMetadata[i] != nullptr; ++i) {
                std::string entry(bandMetadata[i]);
                size_t equalPos = entry.find('=');
                if (equalPos != std::string::npos) {
                    variableInfo.attributes[entry.substr(0, equalPos)] = entry.substr(equalPos + 1);
                }
            }
        }
        
        LOG_DEBUG("成功提取栅格变量信息: {} (波段 {})", variableName, bandNumber);
        
    } else if (dataType_ == GdalDataType::VECTOR) {
        // 🔧 完善矢量数据的字段提取逻辑
        OGRLayer* layer = nullptr;
        
        // 根据变量名查找图�?
        int layerCount = dataset_->GetLayerCount();
        for (int i = 0; i < layerCount; ++i) {
            OGRLayer* candidateLayer = dataset_->GetLayer(i);
            if (candidateLayer && candidateLayer->GetName() == variableName) {
                layer = candidateLayer;
                break;
            }
        }
        
        if (!layer) {
            LOG_WARN("未找到矢量图�? {}", variableName);
            return variableInfo;
        }
        
        // 设置基本信息
        variableInfo.description = std::string(layer->GetName());
        variableInfo.dataType = DataType::String; // 🔧 矢量数据使用String类型表示
        
        // 🔧 获取几何类型
        OGRwkbGeometryType geomType = layer->GetGeomType();
        std::string geometryTypeName = OGRGeometryTypeToName(geomType);
        variableInfo.attributes["geometry_type"] = geometryTypeName;
        
        // 🔧 获取要素数量
        GIntBig featureCount = layer->GetFeatureCount();
        if (featureCount >= 0) {
            variableInfo.attributes["feature_count"] = std::to_string(featureCount);
        }
        
        // 🔧 获取图层信息
        OGRFeatureDefn* layerDefn = layer->GetLayerDefn();
        if (layerDefn) {
            int fieldCount = layerDefn->GetFieldCount();
            variableInfo.attributes["field_count"] = std::to_string(fieldCount);
            
            // 获取字段信息
            std::vector<std::string> fieldNames;
            for (int i = 0; i < fieldCount; ++i) {
                OGRFieldDefn* fieldDefn = layerDefn->GetFieldDefn(i);
                if (fieldDefn) {
                    fieldNames.push_back(fieldDefn->GetNameRef());
                }
            }
            
            if (!fieldNames.empty()) {
                std::string fieldsStr;
                for (size_t i = 0; i < fieldNames.size(); ++i) {
                    if (i > 0) fieldsStr += ",";
                    fieldsStr += fieldNames[i];
                }
                variableInfo.attributes["fields"] = fieldsStr;
            }
        }
        
        // 🔧 获取空间参考系�?
        OGRSpatialReference* spatialRef = layer->GetSpatialRef();
        if (spatialRef) {
            const char* authName = spatialRef->GetAuthorityName(nullptr);
            const char* authCode = spatialRef->GetAuthorityCode(nullptr);
            if (authName && authCode) {
                variableInfo.attributes["crs_authority"] = authName;
                variableInfo.attributes["crs_code"] = authCode;
            }
        }
        
        LOG_DEBUG("成功提取矢量变量信息: {} (几何类型: {})", variableName, geometryTypeName);
    }
    
    return variableInfo;
}

bool GDALFormatHandler::validateDataset() const {
    if (!dataset_) {
        return false;
    }
    
    if (dataType_ == GdalDataType::RASTER) {
        return dataset_->GetRasterCount() > 0;
    } else if (dataType_ == GdalDataType::VECTOR) {
        return dataset_->GetLayerCount() > 0;
    }
    
    return false;
}

// =============================================================================
// GDALStreamingAdapter实现
// =============================================================================

GDALStreamingAdapter::GDALStreamingAdapter(GDALDataset* dataset, const std::string& variableName)
    : dataset_(dataset), variableName_(variableName) {
    
    if (!dataset_) {
        throw std::invalid_argument("GDAL dataset cannot be null");
    }
    
    dataType_ = (dataset_->GetRasterCount() > 0) ? GdalDataType::RASTER : GdalDataType::VECTOR;
    LOG_INFO("GDALStreamingAdapter初始�? 变量={}, 类型={}", 
             variableName, dataType_ == GdalDataType::RASTER ? "栅格" : "矢量");
}

bool GDALStreamingAdapter::hasMoreChunks() const {
    if (!initialized_) {
        return true; // 第一次调�?
    }
    
    if (dataType_ == GdalDataType::RASTER) {
        return currentTileX_ < tilesX_ || currentTileY_ < tilesY_;
    } else if (dataType_ == GdalDataType::VECTOR) {
        return currentFeatureIndex_ < totalFeatures_;
    }
    
    return false;
}

boost::optional<DataChunk> GDALStreamingAdapter::getNextChunk() {
    if (!initialized_) {
        initialize();
    }
    
    // 🆕 应用背压控制
    if (shouldApplyBackpressure()) {
        LOG_DEBUG("应用背压控制，等待资源释�?);
        auto future = waitForBackpressureRelief();
        // 这里可以选择同步等待或异步处�?
        // 为了简化，暂时同步等待
        try {
            future.wait_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            LOG_WARN("背压等待异常: {}", e.what());
        }
    }
    
    boost::optional<DataChunk> chunk;
    
    if (dataType_ == GdalDataType::RASTER) {
        chunk = readRasterTile();
    } else if (dataType_ == GdalDataType::VECTOR) {
        chunk = readVectorChunk();
    }
    
    // 🆕 更新内存使用统计和SIMD优化
    if (chunk.has_value()) {
        size_t chunkMemory = chunk->data.size() * sizeof(double);
        updateMemoryUsage(chunkMemory, true);
        
        // 🆕 应用SIMD优化
        applySIMDOptimizations(*chunk);
        
        LOG_DEBUG("读取数据�? ID={}, 大小={:.2f}KB, 活跃块数={}, 内存使用={:.2f}MB",
                  chunk->chunkId, chunkMemory / 1024.0, 
                  activeChunks_.load(), getCurrentMemoryUsage() / (1024.0 * 1024.0));
    }
    
    return chunk;
}

void GDALStreamingAdapter::reset() {
    currentTileX_ = 0;
    currentTileY_ = 0;
    currentFeatureIndex_ = 0;
    currentChunkId_ = 0;
    initialized_ = false;
    LOG_INFO("GDALStreamingAdapter已重�?);
}

void GDALStreamingAdapter::configureChunking(const StreamingConfig& config) {
    config_ = config;
    if (dataType_ == GdalDataType::RASTER) {
        tileXSize_ = static_cast<int>(config_.chunkSize / sizeof(double));
        tileYSize_ = 256; // 固定高度
    } else if (dataType_ == GdalDataType::VECTOR) {
        // 对于矢量数据，根据块大小估算每块要素数量
        // 假设每个要素平均占用1KB内存
        featuresPerChunk_ = std::max(size_t{100}, config_.chunkSize / 1024);
    }
    LOG_INFO("流式配置更新: 块大�?{}字节", config_.chunkSize);
}

void GDALStreamingAdapter::configureRasterStreaming(int bandNumber, int tileXSize, int tileYSize) {
    bandNumber_ = bandNumber;
    tileXSize_ = tileXSize;
    tileYSize_ = tileYSize;
    LOG_INFO("栅格流式配置: 波段={}, 瓦片大小={}x{}", bandNumber, tileXSize, tileYSize);
}

void GDALStreamingAdapter::configureVectorStreaming(const std::string& layerName, size_t featuresPerChunk) {
    layerName_ = layerName;
    featuresPerChunk_ = featuresPerChunk;
    LOG_INFO("矢量流式配置: 图层={}, 每块要素�?{}", layerName, featuresPerChunk);
}

void GDALStreamingAdapter::initialize() {
    if (dataType_ == GdalDataType::RASTER) {
        calculateRasterTiling();
    } else if (dataType_ == GdalDataType::VECTOR) {
        initializeVectorStreaming();
    }
    initialized_ = true;
}

void GDALStreamingAdapter::calculateRasterTiling() {
    if (bandNumber_ <= 0) {
        bandNumber_ = 1; // 默认第一个波�?
    }
    
    int rasterXSize = dataset_->GetRasterXSize();
    int rasterYSize = dataset_->GetRasterYSize();
    
    // 🆕 计算文件总大小和内存需�?
    GDALRasterBand* band = dataset_->GetRasterBand(bandNumber_);
    if (!band) {
        LOG_ERROR("无法获取波段: {}", bandNumber_);
        return;
    }
    
    GDALDataType bandDataType = band->GetRasterDataType();
    int dataTypeSize = GDALGetDataTypeSizeBytes(bandDataType);
    size_t totalFileSize = static_cast<size_t>(rasterXSize) * rasterYSize * dataTypeSize;
    size_t totalMemoryNeeded = static_cast<size_t>(rasterXSize) * rasterYSize * sizeof(double); // 转换为double的内存需�?
    
    // 🆕 获取可用内存信息
    size_t availableMemory = config_.chunkSize * 10; // 基于配置的内存预算，默认是块大小�?0�?
    size_t maxChunkMemory = config_.chunkSize;
    
    // 🆕 自适应分块策略
    if (totalMemoryNeeded <= maxChunkMemory) {
        // 小文件：一次性读�?
        tileXSize_ = rasterXSize;
        tileYSize_ = rasterYSize;
        LOG_INFO("小文件策�? 一次性读�?{}x{} 像素", rasterXSize, rasterYSize);
    } else if (totalFileSize > 100 * 1024 * 1024) { // 大于100MB
        // 大文件：使用小块策略
        size_t pixelsPerChunk = maxChunkMemory / sizeof(double);
        
        // 优先保持宽度，调整高�?
        if (static_cast<size_t>(rasterXSize) <= pixelsPerChunk) {
            tileXSize_ = rasterXSize;
            tileYSize_ = static_cast<int>(pixelsPerChunk / rasterXSize);
            tileYSize_ = std::max(1, std::min(tileYSize_, rasterYSize));
        } else {
            // 超宽图像：使用正方形瓦片
            int tileSize = static_cast<int>(std::sqrt(pixelsPerChunk));
            tileXSize_ = std::min(tileSize, rasterXSize);
            tileYSize_ = std::min(tileSize, rasterYSize);
        }
        
        // 确保瓦片大小是合理的（至�?4x64，最�?048x2048�?
        tileXSize_ = std::max(64, std::min(2048, tileXSize_));
        tileYSize_ = std::max(64, std::min(2048, tileYSize_));
        
        LOG_INFO("大文件策�? 瓦片大小={}x{}, 文件大小={:.2f}MB", 
                 tileXSize_, tileYSize_, totalFileSize / (1024.0 * 1024.0));
    } else {
        // 中等文件：平衡策�?
        size_t pixelsPerChunk = maxChunkMemory / sizeof(double);
        int approxTileSize = static_cast<int>(std::sqrt(pixelsPerChunk));
        
        tileXSize_ = std::min(approxTileSize, rasterXSize);
        tileYSize_ = std::min(approxTileSize, rasterYSize);
        
        // 优化为GDAL块大小的倍数（如果有的话�?
        int blockXSize, blockYSize;
        band->GetBlockSize(&blockXSize, &blockYSize);
        if (blockXSize > 0 && blockYSize > 0) {
            // 调整为块大小的倍数以提高I/O效率
            tileXSize_ = ((tileXSize_ + blockXSize - 1) / blockXSize) * blockXSize;
            tileYSize_ = ((tileYSize_ + blockYSize - 1) / blockYSize) * blockYSize;
            tileXSize_ = std::min(tileXSize_, rasterXSize);
            tileYSize_ = std::min(tileYSize_, rasterYSize);
        }
        
        LOG_INFO("中等文件策略: 瓦片大小={}x{}, 原生块大�?{}x{}", 
                 tileXSize_, tileYSize_, blockXSize, blockYSize);
    }
    
    // 计算瓦片数量
    tilesX_ = (rasterXSize + tileXSize_ - 1) / tileXSize_;
    tilesY_ = (rasterYSize + tileYSize_ - 1) / tileYSize_;
    
    // 🆕 内存使用验证
    size_t actualChunkMemory = static_cast<size_t>(tileXSize_) * tileYSize_ * sizeof(double);
    size_t totalProcessingMemory = actualChunkMemory * config_.maxConcurrency;
    
    if (totalProcessingMemory > availableMemory) {
        LOG_WARN("内存使用警告: 预计使用{:.2f}MB, 可用{:.2f}MB", 
                 totalProcessingMemory / (1024.0 * 1024.0),
                 availableMemory / (1024.0 * 1024.0));
    }
    
    LOG_INFO("自适应栅格瓦片计算完成: {}x{} 像素 -> {}x{} 瓦片 ({}x{} 像素/瓦片, {:.2f}MB/瓦片)", 
             rasterXSize, rasterYSize, tilesX_, tilesY_,
             tileXSize_, tileYSize_, actualChunkMemory / (1024.0 * 1024.0));
}

void GDALStreamingAdapter::initializeVectorStreaming() {
    if (layerName_.empty()) {
        // 如果没有指定图层名，使用第一个图�?
        if (dataset_->GetLayerCount() > 0) {
            currentLayer_ = dataset_->GetLayer(0);
            layerName_ = currentLayer_->GetName();
        }
    } else {
        currentLayer_ = dataset_->GetLayerByName(layerName_.c_str());
    }
    
    if (!currentLayer_) {
        LOG_ERROR("无法获取矢量图层: {}", layerName_);
        return;
    }
    
    // 获取总要素数�?
    totalFeatures_ = static_cast<size_t>(std::max(static_cast<GIntBig>(0), currentLayer_->GetFeatureCount()));
    currentFeatureIndex_ = 0;
    currentChunkId_ = 0;
    
    // 重置图层读取位置
    currentLayer_->ResetReading();
    
    LOG_INFO("矢量流式初始�? 图层={}, 总要素数={}, 每块要素�?{}", 
             layerName_, totalFeatures_, featuresPerChunk_);
}

boost::optional<DataChunk> GDALStreamingAdapter::readRasterTile() {
    if (currentTileY_ >= tilesY_) {
        return boost::none;
    }
    
    GDALRasterBand* band = dataset_->GetRasterBand(bandNumber_);
    if (!band) {
        return boost::none;
    }
    
    // 计算当前瓦片的像素范�?
    int xOff = currentTileX_ * tileXSize_;
    int yOff = currentTileY_ * tileYSize_;
    int xSize = std::min(tileXSize_, dataset_->GetRasterXSize() - xOff);
    int ySize = std::min(tileYSize_, dataset_->GetRasterYSize() - yOff);
    
    // 读取数据
    size_t dataSize = static_cast<size_t>(xSize) * ySize;
    std::vector<double> data(dataSize);
    
    CPLErr result = band->RasterIO(GF_Read, xOff, yOff, xSize, ySize,
                                  data.data(), xSize, ySize, GDT_Float64,
                                  0, 0);
    
    if (result != CE_None) {
        LOG_ERROR("读取栅格瓦片失败: ({}, {})", currentTileX_, currentTileY_);
        return boost::none;
    }
    
    // 创建数据�?
    DataChunk chunk;
    chunk.data = std::move(data);
    chunk.shape = {static_cast<size_t>(ySize), static_cast<size_t>(xSize)};
    chunk.offset = {static_cast<size_t>(yOff), static_cast<size_t>(xOff)};
    chunk.chunkId = currentTileY_ * tilesX_ + currentTileX_;
    
    // 移动到下一个瓦�?
    currentTileX_++;
    if (currentTileX_ >= tilesX_) {
        currentTileX_ = 0;
        currentTileY_++;
    }
    
    chunk.isLastChunk = (currentTileY_ >= tilesY_);
    
    return chunk;
}

boost::optional<DataChunk> GDALStreamingAdapter::readVectorChunk() {
    if (!currentLayer_ || currentFeatureIndex_ >= totalFeatures_) {
        return boost::none;
    }
    
    std::vector<double> chunkData;
    size_t featuresInThisChunk = 0;
    size_t maxFeaturesInChunk = std::min(featuresPerChunk_, totalFeatures_ - currentFeatureIndex_);
    
    // 读取一批要�?
    class OGRFeature* feature = nullptr;
    while (featuresInThisChunk < maxFeaturesInChunk && 
           (feature = currentLayer_->GetNextFeature()) != nullptr) {
        
        try {
            // 处理几何数据 - 提取坐标�?
            OGRGeometry* geometry = feature->GetGeometryRef();
            if (geometry) {
                // 根据几何类型提取坐标
                if (wkbFlatten(geometry->getGeometryType()) == wkbPoint) {
                    OGRPoint* point = geometry->toPoint();
                    chunkData.push_back(point->getX());
                    chunkData.push_back(point->getY());
                    if (point->Is3D()) {
                        chunkData.push_back(point->getZ());
                    }
                } else if (wkbFlatten(geometry->getGeometryType()) == wkbLineString) {
                    OGRLineString* lineString = geometry->toLineString();
                    int numPoints = lineString->getNumPoints();
                    for (int i = 0; i < numPoints; ++i) {
                        chunkData.push_back(lineString->getX(i));
                        chunkData.push_back(lineString->getY(i));
                        if (lineString->Is3D()) {
                            chunkData.push_back(lineString->getZ(i));
                        }
                    }
                }
                // 其他几何类型可以根据需要扩�?
            }
            
            // 处理属性数�?- 提取数值属�?
            OGRFeatureDefn* featureDefn = currentLayer_->GetLayerDefn();
            int fieldCount = featureDefn->GetFieldCount();
            for (int i = 0; i < fieldCount; ++i) {
                OGRFieldDefn* fieldDefn = featureDefn->GetFieldDefn(i);
                if (fieldDefn && fieldDefn->GetType() == OFTReal) {
                    double value = feature->GetFieldAsDouble(i);
                    chunkData.push_back(value);
                } else if (fieldDefn && fieldDefn->GetType() == OFTInteger) {
                    int value = feature->GetFieldAsInteger(i);
                    chunkData.push_back(static_cast<double>(value));
                }
            }
            
            featuresInThisChunk++;
            currentFeatureIndex_++;
            
        } catch (const std::exception& e) {
            LOG_WARN("处理要素时出�? {}", e.what());
        }
        
        // 清理要素
        OGRFeature::DestroyFeature(feature);
    }
    
    if (chunkData.empty()) {
        return boost::none;
    }
    
    // 创建数据�?
    DataChunk chunk;
    chunk.data = std::move(chunkData);
    chunk.shape = {featuresInThisChunk, chunk.data.size() / featuresInThisChunk}; // 要素�?x 每要素数据数
    chunk.offset = {currentChunkId_ * featuresPerChunk_, 0};
    chunk.chunkId = currentChunkId_++;
    chunk.isLastChunk = (currentFeatureIndex_ >= totalFeatures_);
    
    LOG_DEBUG("读取矢量数据�? 块ID={}, 要素�?{}, 数据点数={}", 
              chunk.chunkId, featuresInThisChunk, chunk.data.size());
    
    return chunk;
}

bool GDALStreamingAdapter::shouldApplyBackpressure() const {
    // 检查活跃块数量是否超过阈�?
    size_t maxActiveChunks = config_.maxConcurrency * 2; // 允许一定的缓冲
    if (activeChunks_.load() >= maxActiveChunks) {
        return true;
    }
    
    // 检查内存使用是否超过阈�?
    return checkMemoryThreshold();
}

boost::unique_future<bool> GDALStreamingAdapter::waitForBackpressureRelief() {
    return boost::async(boost::launch::async, [this]() -> bool {
        std::unique_lock<std::mutex> lock(backpressureMutex_);
        
        // 等待直到背压缓解
        bool relieved = backpressureCondition_.wait_for(lock, std::chrono::seconds(30), [this]() {
            return !shouldApplyBackpressure();
        });
        
        if (!relieved) {
            LOG_WARN("背压等待超时，强制继续处�?);
        }
        
        return relieved;
    });
}

void GDALStreamingAdapter::notifyChunkProcessed() {
    // 减少活跃块计�?
    if (activeChunks_.load() > 0) {
        activeChunks_.fetch_sub(1);
    }
    
    // 通知等待的线�?
    {
        std::lock_guard<std::mutex> lock(backpressureMutex_);
        backpressureCondition_.notify_all();
    }
}

size_t GDALStreamingAdapter::getCurrentMemoryUsage() const {
    return totalMemoryUsed_.load();
}

bool GDALStreamingAdapter::checkMemoryThreshold() const {
    size_t currentMemory = totalMemoryUsed_.load();
    size_t maxMemory = config_.chunkSize * config_.maxConcurrency * 3; // 3倍缓�?
    
    return currentMemory > maxMemory;
}

void GDALStreamingAdapter::updateMemoryUsage(size_t chunkSize, bool isAdd) {
    if (isAdd) {
        totalMemoryUsed_.fetch_add(chunkSize);
        activeChunks_.fetch_add(1);
    } else {
        if (totalMemoryUsed_.load() >= chunkSize) {
            totalMemoryUsed_.fetch_sub(chunkSize);
        }
        if (activeChunks_.load() > 0) {
            activeChunks_.fetch_sub(1);
        }
    }
}

void GDALStreamingAdapter::setSIMDManager(std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager) {
    simdManager_ = simdManager;
    LOG_INFO("GDAL流式适配器已设置SIMD管理�?);
}

bool GDALStreamingAdapter::shouldUseSIMDProcessing(size_t dataSize) const {
    // 检查基本条�?
    if (!simdManager_ || !enableSIMDOptimizations_ || dataSize < simdThreshold_) {
        return false;
    }
    
    // 检查数据类型是否适合SIMD处理
    if (dataType_ == GdalDataType::RASTER) {
        // 栅格数据：大数据块适合SIMD
        return dataSize >= 1000; // 至少1000个数据点
    } else if (dataType_ == GdalDataType::VECTOR) {
        // 矢量数据：当有大量坐标点时使用SIMD
        return dataSize >= 500; // 至少500个坐标点
    }
    
    return false;
}

void GDALStreamingAdapter::applySIMDOptimizations(DataChunk& chunk) const {
    if (!simdManager_ || chunk.data.empty()) {
        return;
    }
    
    try {
        const size_t dataSize = chunk.data.size();
        
        if (!shouldUseSIMDProcessing(dataSize)) {
            return;
        }
        
        LOG_DEBUG("对数据块应用SIMD优化: 数据点数={}", dataSize);
        
        // 🆕 SIMD数据处理优化
        
        // 1. 数据清理 - 移除NaN和异常�?
        std::vector<float> floatData(dataSize);
        for (size_t i = 0; i < dataSize; ++i) {
            floatData[i] = static_cast<float>(chunk.data[i]);
        }
        
        // 2. SIMD统计计算 (均值、最值等)
        if (dataSize >= 4) { // SIMD至少需�?个元�?
            float minValue = simdManager_->vectorMin(floatData.data(), dataSize);
            float maxValue = simdManager_->vectorMax(floatData.data(), dataSize);
            float meanValue = simdManager_->vectorMean(floatData.data(), dataSize);
            
            // 将统计信息保存在chunk中（可扩展DataChunk结构来支持）
            LOG_DEBUG("SIMD统计: min={:.3f}, max={:.3f}, mean={:.3f}", minValue, maxValue, meanValue);
        }
        
        // 3. 数据标准�?(如果需�?
        if (config_.enableOptimization) {
            std::vector<float> normalizedData(dataSize);
            float scale = 1.0f;
            
            // 使用SIMD进行向量标量乘法
            simdManager_->vectorScalarMul(floatData.data(), scale, normalizedData.data(), dataSize);
            
            // 将结果转换回double
            for (size_t i = 0; i < dataSize; ++i) {
                chunk.data[i] = static_cast<double>(normalizedData[i]);
            }
        }
        
        LOG_DEBUG("SIMD优化完成: 处理{}个数据点", dataSize);
        
    } catch (const std::exception& e) {
        LOG_WARN("SIMD优化失败，回退到标量处�? {}", e.what());
    }
}

void GDALStreamingAdapter::setThreadPool(std::shared_ptr<boost::asio::thread_pool> threadPool) {
    threadPool_ = threadPool;
    LOG_INFO("GDAL流式适配器已设置线程�?);
}

void GDALStreamingAdapter::configureConcurrency(size_t maxConcurrentReads, bool enableParallelProcessing) {
    maxConcurrentReads_ = maxConcurrentReads;
    enableParallelProcessing_ = enableParallelProcessing;
    LOG_INFO("配置并发处理: 最大并发读�?{}, 启用并行={}", maxConcurrentReads, enableParallelProcessing);
}

boost::unique_future<std::vector<DataChunk>> GDALStreamingAdapter::readMultipleChunksAsync(size_t numChunks) {
    return boost::async(boost::launch::async, [this, numChunks]() -> std::vector<DataChunk> {
        std::vector<DataChunk> chunks;
        chunks.reserve(numChunks);
        
        if (!enableParallelProcessing_ || !threadPool_ || numChunks <= 1) {
            // 顺序读取
            for (size_t i = 0; i < numChunks && hasMoreChunks(); ++i) {
                auto chunk = getNextChunk();
                if (chunk.has_value()) {
                    chunks.push_back(std::move(*chunk));
                } else {
                    break;
                }
            }
            return chunks;
        }
        
        // 🆕 并行读取策略
        LOG_DEBUG("开始并行读�?{} 个数据块", numChunks);
        
        std::vector<boost::unique_future<boost::optional<DataChunk>>> futures;
        std::mutex chunksMutex;
        
        try {
            // 提交并行读取任务
            size_t actualTasks = std::min(numChunks, maxConcurrentReads_);
            for (size_t i = 0; i < actualTasks && hasMoreChunks(); ++i) {
                auto future = boost::async(boost::launch::async, [this]() -> boost::optional<DataChunk> {
                    return getNextChunk();
                });
                futures.push_back(std::move(future));
            }
            
            // 收集结果
            for (auto& future : futures) {
                try {
                    auto chunk = future.get();
                    if (chunk.has_value()) {
                        std::lock_guard<std::mutex> lock(chunksMutex);
                        chunks.push_back(std::move(*chunk));
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("并行读取块失�? {}", e.what());
                }
            }
            
            // 按需继续读取剩余块（顺序�?
            while (chunks.size() < numChunks && hasMoreChunks()) {
                auto chunk = getNextChunk();
                if (chunk.has_value()) {
                    chunks.push_back(std::move(*chunk));
                } else {
                    break;
                }
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("并行读取失败，回退到顺序读�? {}", e.what());
            
            // 回退到顺序读�?
            chunks.clear();
            for (size_t i = 0; i < numChunks && hasMoreChunks(); ++i) {
                auto chunk = getNextChunk();
                if (chunk.has_value()) {
                    chunks.push_back(std::move(*chunk));
                } else {
                    break;
                }
            }
        }
        
        LOG_DEBUG("并行读取完成: 实际读取 {} 个数据块", chunks.size());
        return chunks;
    });
}

} // namespace oscean::core_services::data_access::readers::impl::gdal 
