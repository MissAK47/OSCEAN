#pragma once

#include "core_services/common_data_types.h"
#include "core_services/crs/i_crs_service.h" // Re-adding dependency to call CRS service
#include "common_utils/utilities/logging_utils.h"
#include <memory>
#include <string>
#include <map>
#include <vector>

namespace oscean::core_services::metadata::impl {

/**
 * @class MetadataStandardizer
 * @brief 负责将从不同数据源读取的原始元数据转换为统一、标准化的格式。
 *
 * 🎯 **核心职责**
 * - **结构化转换**: 将扁平的键值对属性（如NetCDF全局属性）解析并填充到结构化的
 *   `SpatialInfo`, `TemporalInfo` 等字段中。
 * - **坐标系标准化**: 解析多种格式的CRS信息（如PROJ字符串、EPSG代码、WKT），
 *   并统一为标准的 `CRSInfo` 格式。
 * - **时空范围计算**: 根据原始坐标轴数据（如经度、纬度、时间数组）计算出标准
 *   的 `BoundingBox` 和 `TimeRange`。
 * - **变量元数据规范化**: 统一不同来源的变量属性，例如单位、标准名称等。
 * - **完整性校验与修复**: 检查关键元数据字段是否缺失，并尝试根据可用信息进行
 *   推断或填充默认值。
 *
 * ⚙️ **工作流程**
 * 1. 接收一个包含原始、未处理数据的 `FileMetadata` 对象。
 * 2. 根据 `readerType`（如 "NetCDF", "GDAL"）选择特定的标准化规则集。
 * 3. 应用规则，逐步填充和转换 `FileMetadata` 对象的各个字段。
 * 4. 返回一个经过完全处理、可直接入库的标准化 `FileMetadata` 对象。
 */
class MetadataStandardizer {
public:
    /**
     * @brief 构造函数，注入所需的服务依赖。
     * @param crsService 一个CRS服务实例，用于所有坐标系相关的操作。
     */
    explicit MetadataStandardizer(std::shared_ptr<oscean::core_services::ICrsService> crsService);

    /**
     * @brief 标准化文件元数据的主入口点。
     * @param rawMetadata 从读取器获取的、包含原始信息的元数据对象。
     * @param readerType 产生此元数据的读取器类型（例如 "NetCDF_Advanced"）。
     * @return 一个经过完全处理和标准化的`FileMetadata`对象。
     */
    oscean::core_services::FileMetadata standardizeMetadata(
        const oscean::core_services::FileMetadata& rawMetadata,
        const std::string& readerType) const;

private:
    /**
     * @brief 应用NetCDF特定的标准化规则。
     * @param metadata 要被修改和填充的元数据对象。
     */
    void applyNetCDFStandardization(oscean::core_services::FileMetadata& metadata) const;
    
    /**
     * @brief 应用GDAL特定的标准化规则。
     * @param metadata 要被修改和填充的元数据对象。
     */
    void applyGDALStandardization(oscean::core_services::FileMetadata& metadata) const;
    
    /**
     * @brief 最终的验证和修复步骤，确保元数据的逻辑一致性和完整性。
     * @param metadata 要被验证和修复的元数据对象。
     */
    void validateAndRepair(oscean::core_services::FileMetadata& metadata) const;

    // 服务依赖
    std::shared_ptr<oscean::core_services::ICrsService> crsService_;
};

} // namespace oscean::core_services::metadata::impl 