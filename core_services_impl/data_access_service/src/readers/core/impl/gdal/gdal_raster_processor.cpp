/**
 * @file gdal_raster_processor.cpp
 * @brief GDAL栅格数据专用处理器实现
 */

#include "gdal_raster_processor.h"
#include "common_utils/utilities/logging_utils.h"
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace oscean::core_services::data_access::readers::impl::gdal {

GdalRasterProcessor::GdalRasterProcessor(GDALDataset* dataset) : dataset_(dataset) {
    if (!dataset_) {
        throw std::invalid_argument("GdalRasterProcessor requires a non-null GDALDataset.");
    }
}

int GdalRasterProcessor::getBandNumber(const std::string& variableName) const {
    if (!dataset_) {
        return -1;
    }
    
    // 尝试直接解析波段编号
    if (variableName.find("band_") == 0) {
        try {
            int bandNum = std::stoi(variableName.substr(5));
            if (bandNum >= 1 && bandNum <= dataset_->GetRasterCount()) {
                return bandNum;
            }
        } catch (const std::exception&) {
            // 继续其他方法
        }
    }
    
    // 通过描述查找
    int bandCount = dataset_->GetRasterCount();
    for (int i = 1; i <= bandCount; ++i) {
        GDALRasterBand* band = dataset_->GetRasterBand(i);
        if (band) {
            const char* description = band->GetDescription();
            if (description && variableName == std::string(description)) {
                return i;
            }
        }
    }
    
    return -1;
}

std::vector<std::string> GdalRasterProcessor::getVariableNames() const {
    std::vector<std::string> variableNames;
    if (!dataset_) {
        return variableNames;
    }
    int bandCount = dataset_->GetRasterCount();
    for (int i = 1; i <= bandCount; ++i) {
        std::string variableName = "band_" + std::to_string(i);
        variableNames.push_back(variableName);
    }
    return variableNames;
}

std::shared_ptr<oscean::core_services::GridData> GdalRasterProcessor::readRasterDataAdvanced(
    const std::string& variableName,
    const std::optional<oscean::core_services::BoundingBox>& bounds) {
    
    int bandNumber = getBandNumber(variableName);
    if (bandNumber <= 0) {
        throw std::runtime_error("Invalid variable name: " + variableName);
    }
    
    GDALRasterBand* band = dataset_->GetRasterBand(bandNumber);
    if (!band) {
        throw std::runtime_error("Cannot get band: " + std::to_string(bandNumber));
    }
    
    auto gridData = std::make_shared<oscean::core_services::GridData>();
    
    ReadRegion readRegion;
    if (bounds) {
        auto optRegion = calculateReadRegion(*bounds, band);
        if (!optRegion) {
            LOG_WARN("Cannot calculate read region from bounding box, using full extent.");
            readRegion = {0, 0, band->GetXSize(), band->GetYSize()};
        } else {
            readRegion = *optRegion;
        }
    } else {
        readRegion = {0, 0, band->GetXSize(), band->GetYSize()};
    }
    
    gridData->definition.cols = readRegion.xSize;
    gridData->definition.rows = readRegion.ySize;
    gridData->definition.xResolution = 1.0; 
    gridData->definition.yResolution = 1.0;
    
    const size_t dataSize = static_cast<size_t>(readRegion.xSize) * readRegion.ySize;
    std::vector<double> doubleData(dataSize);
    
    CPLErr err = band->RasterIO(GF_Read, readRegion.xOff, readRegion.yOff, readRegion.xSize, readRegion.ySize,
                              doubleData.data(), readRegion.xSize, readRegion.ySize, GDT_Float64, 0, 0);
    
    if (err != CE_None) {
        throw std::runtime_error("Failed to read raster data");
    }
    
    double noDataValue = 0.0;
    int hasNoData = 0;
    noDataValue = band->GetNoDataValue(&hasNoData);
    
    double scaleFactor = band->GetScale();
    double offset = band->GetOffset();
    
    if (hasNoData || scaleFactor != 1.0 || offset != 0.0) {
        processDataStandard(doubleData, noDataValue, scaleFactor, offset, hasNoData != 0);
    }
    
    auto& buffer = gridData->getUnifiedBuffer();
    buffer.resize(dataSize * sizeof(double));
    std::memcpy(buffer.data(), doubleData.data(), buffer.size());
    gridData->dataType = oscean::core_services::DataType::Float64;

    gridData->metadata["band_number"] = std::to_string(bandNumber);
    
    auto metadataEntries = loadBandMetadataAdvanced(variableName);
    for (const auto& entry : metadataEntries) {
        gridData->metadata.emplace(entry.getKey(), entry.getValue());
    }
    
    return gridData;
}

std::optional<ReadRegion> GdalRasterProcessor::calculateReadRegion(
    const oscean::core_services::BoundingBox& bounds, GDALRasterBand* band) {
    
    if (!dataset_) {
        return std::nullopt;
    }
    
    double geoTransform[6];
    if (dataset_->GetGeoTransform(geoTransform) != CE_None) {
        LOG_WARN("Cannot get geotransform, unable to calculate region from bounds.");
        return std::nullopt;
    }
    
    double invGeoTransform[6];
    if (GDALInvGeoTransform(geoTransform, invGeoTransform) == 0) {
        LOG_WARN("Cannot calculate inverse geotransform.");
        return std::nullopt;
    }
    
    double minPixelX, minPixelY, maxPixelX, maxPixelY;
    GDALApplyGeoTransform(invGeoTransform, bounds.minX, bounds.maxY, &minPixelX, &minPixelY);
    GDALApplyGeoTransform(invGeoTransform, bounds.maxX, bounds.minY, &maxPixelX, &maxPixelY);
    
    if (minPixelX > maxPixelX) std::swap(minPixelX, maxPixelX);
    if (minPixelY > maxPixelY) std::swap(minPixelY, maxPixelY);
    
    ReadRegion region;
    region.xOff = std::max(0, static_cast<int>(std::floor(minPixelX)));
    region.yOff = std::max(0, static_cast<int>(std::floor(minPixelY)));
    region.xSize = std::min(dataset_->GetRasterXSize() - region.xOff, 
                           static_cast<int>(std::ceil(maxPixelX)) - region.xOff);
    region.ySize = std::min(dataset_->GetRasterYSize() - region.yOff, 
                           static_cast<int>(std::ceil(maxPixelY)) - region.yOff);
    
    if (region.xSize <= 0 || region.ySize <= 0) {
        LOG_WARN("Calculated read region is invalid: offset=({},{}), size=({},{})", 
                 region.xOff, region.yOff, region.xSize, region.ySize);
        return std::nullopt;
    }
    
    return region;
}

std::vector<oscean::core_services::MetadataEntry> GdalRasterProcessor::loadBandMetadataAdvanced(const std::string& variableName) {
    std::vector<oscean::core_services::MetadataEntry> metadata;
    
    int bandNumber = getBandNumber(variableName);
    if (bandNumber <= 0 || !dataset_) {
        return metadata;
    }
    
    GDALRasterBand* band = dataset_->GetRasterBand(bandNumber);
    if (!band) {
        return metadata;
    }
    
    metadata.emplace_back("ColorInterpretation", std::to_string(band->GetColorInterpretation()));
    metadata.emplace_back("DataType", GDALGetDataTypeName(band->GetRasterDataType()));
    
    int blockXSize, blockYSize;
    band->GetBlockSize(&blockXSize, &blockYSize);
    metadata.emplace_back("BlockXSize", std::to_string(blockXSize));
    metadata.emplace_back("BlockYSize", std::to_string(blockYSize));
    
    return metadata;
}


void GdalRasterProcessor::processDataStandard(std::vector<double>& data, double noDataValue, 
                                          double scaleFactor, double offset, bool hasNoData) {
    for (auto& value : data) {
        if (hasNoData && std::abs(value - noDataValue) < 1e-9) {
            value = std::numeric_limits<double>::quiet_NaN();
        } else if (!std::isnan(value)) {
            value = value * scaleFactor + offset;
        }
    }
}

} // namespace oscean::core_services::data_access::readers::impl::gdal 