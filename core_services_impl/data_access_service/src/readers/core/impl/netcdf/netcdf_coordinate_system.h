#pragma once

/**
 * @file netcdf_coordinate_system.h
 * @brief NetCDF坐标系统信息提取器 - 专注于元数据提取，不进行坐标转换
 * 
 * 架构定位：数据访问服务的坐标系统元数据提取组件
 * 职责：
 * 1. 提取NetCDF文件中的CRS元数据信息
 * 2. 解析CF约定的坐标系统定义
 * 3. 提供给CRS服务进行坐标转换
 * 
 * 不负责：
 * - 坐标转换计算（由CRS服务负责）
 * - 空间几何运算（由空间服务负责）
 */

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <boost/optional.hpp>
#include <map>

#include "core_services/common_data_types.h"
#include "core_services/data_access/unified_data_types.h"

// Forward declarations for NetCDF handles
typedef int ncid_t;

namespace oscean::core_services::data_access::readers::impl::netcdf {

/**
 * @brief NetCDF坐标维度类型
 */
enum class CoordinateDimension {
    LON,        ///< 经度维度
    LAT,        ///< 纬度维度
    VERTICAL,   ///< 垂直维度
    TIME,       ///< 时间维度
    UNKNOWN     ///< 未知维度
};

/**
 * @brief 维度坐标信息
 */
struct DimensionCoordinateInfo {
    std::string name;                           ///< 维度名称
    std::string standardName;                   ///< CF标准名称
    std::string longName;                       ///< 长名称
    std::string units;                          ///< 单位
    CoordinateDimension type;                   ///< 维度类型
    std::vector<double> coordinates;            ///< 坐标值
    bool isRegular = false;                     ///< 是否规则间隔
    double resolution = 0.0;                    ///< 分辨率（如果规则）
    std::map<std::string, std::string> attributes; ///< 其他属性
};

/**
 * @brief NetCDF坐标系统信息提取器
 * 
 * 负责从NetCDF文件中提取坐标系统元数据，遵循单一职责原则：
 * - 只提取CRS信息，不进行坐标转换
 * - 解析CF约定的坐标维度信息
 * - 为其他服务提供标准化的CRS元数据
 */
class NetCDFCoordinateSystemExtractor {
public:
    explicit NetCDFCoordinateSystemExtractor(ncid_t ncid);
    virtual ~NetCDFCoordinateSystemExtractor() = default;
    
    // =============================================================================
    // CRS元数据提取 - 核心职责
    // =============================================================================
    
    /**
     * @brief 提取CRS信息
     * @return 标准化的CRS信息，供CRS服务使用
     */
    oscean::core_services::CRSInfo extractCRSInfo() const;
    
    /**
     * @brief 检测网格映射信息
     */
    boost::optional<std::string> detectGridMapping(const std::string& variableName) const;
    
    /**
     * @brief 解析CRS的WKT表示
     */
    boost::optional<std::string> extractWKTFromCRS() const;
    
    // =============================================================================
    // 维度元数据提取
    // =============================================================================
    
    /**
     * @brief 提取维度坐标信息
     */
    boost::optional<DimensionCoordinateInfo> extractDimensionInfo(const std::string& dimName) const;
    
    /**
     * @brief 获取所有维度信息
     */
    std::vector<DimensionCoordinateInfo> getAllDimensionInfo() const;
    
    /**
     * @brief 查找特定类型的维度
     */
    std::vector<std::string> findDimensionsByType(CoordinateDimension type) const;
    
    /**
     * @brief 查找时间维度
     */
    std::string findTimeDimension() const;
    
    /**
     * @brief 查找经度维度
     */
    std::string findLongitudeDimension() const;
    
    /**
     * @brief 查找纬度维度
     */
    std::string findLatitudeDimension() const;
    
    /**
     * @brief 查找垂直维度
     */
    std::string findVerticalDimension() const;
    
    // =============================================================================
    // 空间元数据提取 - 不进行坐标转换，只提供原始信息
    // =============================================================================
    
    /**
     * @brief 提取原始边界框信息（数据坐标系下）
     * @note 不进行坐标转换，只提取原始坐标范围
     */
    oscean::core_services::BoundingBox extractRawBoundingBox() const;
    
    /**
     * @brief 提取变量的空间范围（数据坐标系下）
     */
    boost::optional<oscean::core_services::BoundingBox> extractVariableRawBounds(const std::string& variableName) const;
    
    /**
     * @brief 检查坐标是否规则间隔
     */
    bool isRegularGrid() const;
    
    // 🚫 已移除：getRawSpatialResolution() - 无任何调用，功能已由空间服务提供
    // 原功能：获取原始空间分辨率
    // 替代方案：使用 GridDefinition 中的分辨率信息或空间服务计算
    // std::pair<double, double> getRawSpatialResolution() const;
    
    // =============================================================================
    // CF约定支持
    // =============================================================================
    
    /**
     * @brief 解析CF约定的坐标属性
     */
    std::vector<std::string> parseCFCoordinates(const std::string& coordinatesAttribute) const;
    
    /**
     * @brief 检测CF约定的轴类型
     */
    CoordinateDimension detectCFAxisType(const std::string& dimName) const;
    
    // 🚫 已移除：validateCFCompliance() - 无任何调用，功能已由CRS服务提供
    // 原功能：验证CF约定合规性
    // 替代方案：使用 ICrsService::validateCRSAsync()
    // bool validateCFCompliance() const;
    
    // =============================================================================
    // 缓存管理
    // =============================================================================
    
    /**
     * @brief 清除缓存
     */
    void clearCache();
    
    /**
     * @brief 预加载维度信息
     */
    void preloadDimensionInfo();
    
private:
    ncid_t ncid_;
    
    // 缓存
    mutable std::unordered_map<std::string, DimensionCoordinateInfo> dimensionCache_;
    mutable boost::optional<oscean::core_services::CRSInfo> cachedCRS_;
    mutable boost::optional<oscean::core_services::BoundingBox> cachedBoundingBox_;
    
    // 内部方法
    CoordinateDimension classifyDimension(const std::string& dimName) const;
    bool isDimensionCoordinate(const std::string& dimName) const;
    std::vector<double> readCoordinateValues(const std::string& dimName) const;
    
    // CF约定解析
    std::string readStringAttribute(int varid, const std::string& attName) const;
    bool hasAttribute(int varid, const std::string& attName) const;
    
    // 坐标系统特定方法
    boost::optional<std::string> findCRSVariable() const;
    boost::optional<std::string> extractProjectionWKT(int crsVarid) const;
    
    // CF投影参数提取方法
    boost::optional<oscean::core_services::CFProjectionParameters> extractCFProjectionParameters(int varid, const std::string& gridMappingName) const;
    double readDoubleAttribute(int varid, const std::string& attName, double defaultValue = 0.0) const;
    
    // 维度分类辅助方法
    bool isLongitudeDimension(const std::string& dimName, const DimensionCoordinateInfo& info) const;
    bool isLatitudeDimension(const std::string& dimName, const DimensionCoordinateInfo& info) const;
    bool isTimeDimension(const std::string& dimName, const DimensionCoordinateInfo& info) const;
    bool isVerticalDimension(const std::string& dimName, const DimensionCoordinateInfo& info) const;
    
    // 🚫 已移除：PROJ字符串清理方法 - 无任何调用，功能已统一到CRS服务
    // 原功能：cleanNetCDFProjString() - 清理NetCDF PROJ字符串
    // 原功能：tryMapToEPSG() - 尝试映射到EPSG代码
    // 替代方案：使用 ICrsService::parseFromStringAsync() 进行统一处理
    // std::string cleanNetCDFProjString(const std::string& projString) const;
    // std::string tryMapToEPSG(const std::string& projString) const;
};

} // namespace oscean::core_services::data_access::readers::impl::netcdf 
