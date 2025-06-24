/**
 * @file gdal_raster_processor.h
 * @brief GDAL栅格数据专用处理器 - 负责栅格数据的核心处理逻辑
 */

#pragma once

#include "readers/core/unified_data_reader.h"
#include "core_services/common_data_types.h"
#include "gdal_common_types.h" // For GdalPerformanceStats and other types
#include <gdal_priv.h>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <map>

// Forward declarations to avoid extra includes
struct ReadRegion;

namespace oscean::core_services::data_access::readers::impl::gdal {


/**
 * @brief 读取区域结构
 */
struct ReadRegion {
    int xOff = 0;
    int yOff = 0;
    int xSize = 0;
    int ySize = 0;
};


class GdalRasterProcessor {
public:
    explicit GdalRasterProcessor(GDALDataset* dataset);
    ~GdalRasterProcessor() = default;

    GdalRasterProcessor(const GdalRasterProcessor&) = delete;
    GdalRasterProcessor& operator=(const GdalRasterProcessor&) = delete;
    GdalRasterProcessor(GdalRasterProcessor&&) = delete;
    GdalRasterProcessor& operator=(GdalRasterProcessor&&) = delete;

    std::shared_ptr<oscean::core_services::GridData> readRasterDataAdvanced(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds);

    std::vector<oscean::core_services::MetadataEntry> loadBandMetadataAdvanced(const std::string& variableName);

    int getBandNumber(const std::string& variableName) const;

    std::vector<std::string> getVariableNames() const;

private:
    std::optional<ReadRegion> calculateReadRegion(
        const oscean::core_services::BoundingBox& bounds, GDALRasterBand* band);
    
    void processDataStandard(std::vector<double>& data, double noDataValue, 
                           double scaleFactor, double offset, bool hasNoData);

    GDALDataset* dataset_; // Non-owning pointer
};

} // namespace oscean::core_services::data_access::readers::impl::gdal 