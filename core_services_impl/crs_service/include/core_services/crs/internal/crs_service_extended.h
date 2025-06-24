#pragma once

#include "core_services/crs/i_crs_service.h"
#include <memory>

// Include actual GDAL headers instead of forward declarations
#include <ogr_spatialref.h> // For OGRSpatialReference
#include <ogr_geometry.h>   // For OGRGeometry (assuming this is the header)

// class OGRSpatialReference; // Removed forward declaration
// class OGRGeometry;         // Removed forward declaration

namespace oscean::core_services {

/**
 * @class ICrsServiceGdalExtended
 * @brief 扩展ICrsService接口，添加GDAL特定的功能以支持矢量数据处理
 */
class ICrsServiceGdalExtended : public ICrsService {
public:
    /**
     * @brief 析构函数
     */
    virtual ~ICrsServiceGdalExtended() = default;

    /**
     * @brief 根据CRSInfo创建OGR空间参考对象
     * @param crsInfo CRS信息
     * @return 创建的OGRSpatialReference对象的共享指针，如果失败则返回nullptr
     */
    virtual std::shared_ptr<OGRSpatialReference> createOgrSrs(const CRSInfo& crsInfo) const = 0;

    /**
     * @brief 检查两个OGR空间参考是否可以进行转换
     * @param sourceSrs 源空间参考
     * @param targetSrs 目标空间参考
     * @return 如果可以转换则返回true，否则返回false
     */
    virtual bool canTransform(const OGRSpatialReference* sourceSrs, const OGRSpatialReference* targetSrs) const = 0;

    /**
     * @brief 转换OGR几何对象
     * @param sourceGeom 源几何对象
     * @param sourceSrs 源空间参考
     * @param targetSrs 目标空间参考
     * @return 转换后的几何对象的唯一指针，如果失败则返回nullptr
     */
    virtual std::unique_ptr<OGRGeometry> transformGeometry(
        OGRGeometry* sourceGeom, 
        OGRSpatialReference* sourceSrs, 
        OGRSpatialReference* targetSrs) const = 0;
};

} // namespace oscean::core_services 