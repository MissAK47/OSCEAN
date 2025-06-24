/**
 * @file netcdf_coordinate_system.cpp
 * @brief NetCDF坐标系统信息提取器实现 - 专注于元数据提取
 * 
 * 重构原则：
 * 1. 移除所有坐标转换功能
 * 2. 只保留CRS元数据提取功能
 * 3. 为CRS服务提供标准化的元数据接口
 */

#include "netcdf_coordinate_system.h"
#include "common_utils/utilities/logging_utils.h"
#include <netcdf.h>
#include <algorithm>
#include <regex>
#include <sstream>

namespace oscean::core_services::data_access::readers::impl::netcdf {

NetCDFCoordinateSystemExtractor::NetCDFCoordinateSystemExtractor(ncid_t ncid) : ncid_(ncid) {
    LOG_INFO("NetCDFCoordinateSystemExtractor初始化: ncid={}", ncid);
}

oscean::core_services::CRSInfo NetCDFCoordinateSystemExtractor::extractCRSInfo() const {
    if (cachedCRS_) {
        return *cachedCRS_;
    }
    
    oscean::core_services::CRSInfo crsInfo;
    
    // 设置默认值
    crsInfo.authorityName = "AUTO";
    crsInfo.authorityCode = "DETECTED";
    crsInfo.isGeographic = true;
    crsInfo.isProjected = false;
    
    // 兼容字段
    crsInfo.authority = crsInfo.authorityName;
    crsInfo.code = crsInfo.authorityCode;
    crsInfo.id = crsInfo.authorityName + ":" + crsInfo.authorityCode;
    
    bool crsDetected = false;
    
    // 步骤1：查找CRS/投影变量 - CF约定支持
    auto crsVariable = findCRSVariable();
    if (crsVariable) {
        int varid;
        if (nc_inq_varid(ncid_, crsVariable->c_str(), &varid) == NC_NOERR) {
            LOG_INFO("发现CRS变量: {}", *crsVariable);
            
            // 优先级1：读取PROJ4字符串（最直接）
            std::string proj4Str = readStringAttribute(varid, "proj4");
            if (proj4Str.empty()) {
                proj4Str = readStringAttribute(varid, "proj4text");
            }
            if (proj4Str.empty()) {
                proj4Str = readStringAttribute(varid, "proj_string");
            }
            
            if (!proj4Str.empty()) {
                LOG_INFO("检测到PROJ4字符串: {}", proj4Str);
                
                // 🔧 修复：只保存原始PROJ字符串，不进行清理处理
                // CRS处理应该在元数据服务中进行，而不是在文件读取阶段
                crsInfo.projString = proj4Str;
                crsInfo.proj4text = proj4Str;  // 兼容字段
                crsInfo.isProjected = true;
                crsInfo.isGeographic = false;
                crsInfo.authorityName = "PROJ4";
                crsInfo.authorityCode = "CUSTOM";
                crsDetected = true;
                
                LOG_INFO("✅ 原始PROJ字符串已保存，将由元数据服务进行标准化处理");
            }
            
            // 优先级2：读取WKT
            if (!crsDetected) {
                std::string wkt = readStringAttribute(varid, "spatial_ref");
                if (wkt.empty()) {
                    wkt = readStringAttribute(varid, "crs_wkt");
                }
                if (wkt.empty()) {
                    wkt = readStringAttribute(varid, "wkt");
                }
                
                if (!wkt.empty()) {
                    LOG_INFO("检测到WKT定义");
                    crsInfo.wktext = wkt;
                    crsInfo.wkt = wkt;  // 兼容字段
                    crsInfo.isProjected = (wkt.find("PROJCS") != std::string::npos);
                    crsInfo.isGeographic = !crsInfo.isProjected;
                    crsDetected = true;
                }
            }
            
            // 🔧 修复：总是尝试提取CF参数，不依赖crsDetected状态
            // CF参数可以与PROJ4/WKT共存，为CRS服务提供更多选择
            std::string gridMappingName = readStringAttribute(varid, "grid_mapping_name");
            LOG_INFO("🔍 查找grid_mapping_name属性: '{}'", gridMappingName.empty() ? "未找到" : gridMappingName);
            
            if (!gridMappingName.empty()) {
                LOG_INFO("检测到CF投影: {}", gridMappingName);
                auto cfParams = extractCFProjectionParameters(varid, gridMappingName);
                if (cfParams.has_value()) {
                    crsInfo.cfParameters = cfParams;
                    LOG_INFO("✅ CF投影参数已提取并保存: {}", gridMappingName);
                    
                    // 🔧 只有在没有其他CRS信息时，才将CF设为主要CRS类型
                    if (!crsDetected) {
                        crsInfo.isProjected = true;
                        crsInfo.isGeographic = false;
                        crsInfo.authorityName = "CF";
                        crsInfo.authorityCode = gridMappingName;
                        crsDetected = true;
                        LOG_INFO("CF投影设为主要CRS类型: {}", gridMappingName);
                    }
                } else {
                    LOG_WARN("⚠️ CF参数提取失败: {}", gridMappingName);
                }
            } else {
                // 🔧 尝试另一种方法：直接使用变量名作为grid_mapping_name
                LOG_INFO("🔧 尝试使用变量名作为CF投影类型: {}", *crsVariable);
                auto cfParams = extractCFProjectionParameters(varid, *crsVariable);
                if (cfParams.has_value()) {
                    crsInfo.cfParameters = cfParams;
                    LOG_INFO("✅ 使用变量名成功提取CF参数: {}", *crsVariable);
                    
                    if (!crsDetected) {
                        crsInfo.isProjected = true;
                        crsInfo.isGeographic = false;
                        crsInfo.authorityName = "CF";
                        crsInfo.authorityCode = *crsVariable;
                        crsDetected = true;
                        LOG_INFO("CF投影设为主要CRS类型: {}", *crsVariable);
                    }
                } else {
                    LOG_INFO("ℹ️ 使用变量名提取CF参数失败: {}", *crsVariable);
                }
            }
            
            // 读取EPSG代码（如果存在）
            int epsgCode;
            if (nc_get_att_int(ncid_, varid, "epsg_code", &epsgCode) == NC_NOERR) {
                crsInfo.epsgCode = epsgCode;
                crsInfo.authorityName = "EPSG";
                crsInfo.authorityCode = std::to_string(epsgCode);
                LOG_INFO("检测到EPSG代码: {}", epsgCode);
            }
            
            // 读取单位信息
            std::string units = readStringAttribute(varid, "units");
            if (!units.empty()) {
                if (units == "m" || units == "meter" || units == "metres") {
                    crsInfo.linearUnitName = "metre";
                    crsInfo.linearUnitToMeter = 1.0;
                } else if (units == "degree" || units == "degrees") {
                    crsInfo.angularUnitName = "degree";
                    crsInfo.angularUnitToRadian = 3.14159265358979323846 / 180.0;
                }
            }
        }
    }
    
    // 步骤2：如果没有找到CRS变量，检查全局属性
    if (!crsDetected) {
        LOG_INFO("未找到CRS变量，检查全局属性...");
        
        // 检查全局CRS WKT属性
        size_t attlen;
        if (nc_inq_attlen(ncid_, NC_GLOBAL, "crs_wkt", &attlen) == NC_NOERR) {
            std::vector<char> wkt(attlen + 1, 0);
            if (nc_get_att_text(ncid_, NC_GLOBAL, "crs_wkt", wkt.data()) == NC_NOERR) {
                crsInfo.wktext = std::string(wkt.data());
                crsInfo.wkt = crsInfo.wktext;
                crsDetected = true;
                LOG_INFO("从全局属性获取CRS WKT: {}", crsInfo.wkt.substr(0, 100) + "...");
            }
        }
        
        // 检查全局PROJ4属性
        if (!crsDetected) {
            std::vector<std::string> proj4Attrs = {"proj4", "proj4text", "proj_string", "projection"};
            for (const auto& attr : proj4Attrs) {
                if (nc_inq_attlen(ncid_, NC_GLOBAL, attr.c_str(), &attlen) == NC_NOERR) {
                    std::vector<char> proj4(attlen + 1, 0);
                    if (nc_get_att_text(ncid_, NC_GLOBAL, attr.c_str(), proj4.data()) == NC_NOERR) {
                        std::string proj4Str(proj4.data());
                        if (!proj4Str.empty()) {
                            crsInfo.projString = proj4Str;
                            crsInfo.proj4text = proj4Str;
                            crsInfo.isProjected = true;
                            crsInfo.isGeographic = false;
                            crsInfo.authorityName = "PROJ4";
                            crsInfo.authorityCode = "CUSTOM";
                            crsDetected = true;
                            LOG_INFO("从全局属性{}获取PROJ4: {}", attr, proj4Str);
                            break;
                        }
                    }
                }
            }
        }
        
        // 检查EPSG代码属性
        if (!crsDetected) {
            std::vector<std::string> epsgAttrs = {"EPSG_code", "epsg_code", "epsg", "EPSG"};
            for (const auto& attr : epsgAttrs) {
                int epsgCode;
                if (nc_get_att_int(ncid_, NC_GLOBAL, attr.c_str(), &epsgCode) == NC_NOERR) {
                    crsInfo.epsgCode = epsgCode;
                    crsInfo.authorityName = "EPSG";
                    crsInfo.authorityCode = std::to_string(epsgCode);
                    
                    // 根据EPSG代码判断是否为地理坐标系
                    if (epsgCode == 4326 || epsgCode == 4269 || epsgCode == 4267) {
                        crsInfo.isGeographic = true;
                        crsInfo.isProjected = false;
                        crsInfo.angularUnitName = "degree";
                        crsInfo.angularUnitToRadian = 3.14159265358979323846 / 180.0;
                    } else {
                        crsInfo.isProjected = true;
                        crsInfo.isGeographic = false;
                        crsInfo.linearUnitName = "metre";
                        crsInfo.linearUnitToMeter = 1.0;
                    }
                    
                    crsDetected = true;
                    LOG_INFO("从全局属性{}获取EPSG代码: {}", attr, epsgCode);
                    break;
                }
            }
        }
        
        // 🔧 新增：检查坐标变量的CRS属性
        if (!crsDetected) {
            LOG_INFO("检查坐标变量的CRS属性...");
            
            // 查找经纬度变量
            std::vector<std::string> lonNames = {"longitude", "lon", "x"};
            std::vector<std::string> latNames = {"latitude", "lat", "y"};
            
            for (const auto& lonName : lonNames) {
                int lonVarId;
                if (nc_inq_varid(ncid_, lonName.c_str(), &lonVarId) == NC_NOERR) {
                    // 检查经度变量的CRS相关属性
                    std::string gridMapping = readStringAttribute(lonVarId, "grid_mapping");
                    if (!gridMapping.empty()) {
                        // 找到grid_mapping变量
                        int crsVarId;
                        if (nc_inq_varid(ncid_, gridMapping.c_str(), &crsVarId) == NC_NOERR) {
                            LOG_INFO("通过经度变量找到grid_mapping: {}", gridMapping);
                            
                            // 从grid_mapping变量提取CRS信息
                            std::string proj4Str = readStringAttribute(crsVarId, "proj4");
                            if (proj4Str.empty()) {
                                proj4Str = readStringAttribute(crsVarId, "proj4text");
                            }
                            
                            if (!proj4Str.empty()) {
                                crsInfo.projString = proj4Str;
                                crsInfo.proj4text = proj4Str;
                                crsInfo.isProjected = true;
                                crsInfo.isGeographic = false;
                                crsInfo.authorityName = "PROJ4";
                                crsInfo.authorityCode = "CUSTOM";
                                crsDetected = true;
                                LOG_INFO("从grid_mapping变量获取PROJ4: {}", proj4Str);
                                break;
                            }
                            
                            // 检查WKT
                            std::string wktStr = readStringAttribute(crsVarId, "crs_wkt");
                            if (wktStr.empty()) {
                                wktStr = readStringAttribute(crsVarId, "spatial_ref");
                            }
                            
                            if (!wktStr.empty()) {
                                crsInfo.wktext = wktStr;
                                crsInfo.wkt = wktStr;
                                crsDetected = true;
                                LOG_INFO("从grid_mapping变量获取WKT: {}", wktStr.substr(0, 100) + "...");
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    
    // 步骤3：如果仍未检测到，使用默认值
    if (!crsDetected) {
        LOG_INFO("未检测到CRS信息，使用默认WGS84");
        crsInfo.authorityName = "EPSG";
        crsInfo.authorityCode = "4326";
        crsInfo.epsgCode = 4326;
        crsInfo.wktext = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]";
        crsInfo.wkt = crsInfo.wktext;
        crsInfo.isGeographic = true;
        crsInfo.isProjected = false;
        crsInfo.angularUnitName = "degree";
        crsInfo.angularUnitToRadian = 3.14159265358979323846 / 180.0;
    }
    
    // 更新兼容字段
    crsInfo.authority = crsInfo.authorityName;
    crsInfo.code = crsInfo.authorityCode;
    crsInfo.id = crsInfo.authorityName + ":" + crsInfo.authorityCode;
    
    LOG_INFO("CRS元数据提取完成 - Authority: {}, Code: {}, Projected: {}", 
             crsInfo.authorityName, crsInfo.authorityCode, crsInfo.isProjected);
    
    cachedCRS_ = crsInfo;
    return crsInfo;
}

boost::optional<std::string> NetCDFCoordinateSystemExtractor::detectGridMapping(const std::string& variableName) const {
    int varid;
    if (nc_inq_varid(ncid_, variableName.c_str(), &varid) != NC_NOERR) {
        return boost::none;
    }
    
    return readStringAttribute(varid, "grid_mapping");
}

boost::optional<std::string> NetCDFCoordinateSystemExtractor::extractWKTFromCRS() const {
    auto crsVariable = findCRSVariable();
    if (!crsVariable) {
        return boost::none;
    }
    
    int varid;
    if (nc_inq_varid(ncid_, crsVariable->c_str(), &varid) != NC_NOERR) {
        return boost::none;
    }
    
    return extractProjectionWKT(varid);
}

boost::optional<DimensionCoordinateInfo> NetCDFCoordinateSystemExtractor::extractDimensionInfo(const std::string& dimName) const {
    // 检查缓存
    auto it = dimensionCache_.find(dimName);
    if (it != dimensionCache_.end()) {
        return it->second;
    }
    
    int varid;
    if (nc_inq_varid(ncid_, dimName.c_str(), &varid) != NC_NOERR) {
        return boost::none;
    }
    
    DimensionCoordinateInfo info;
    info.name = dimName;
    
    // 读取属性
    info.standardName = readStringAttribute(varid, "standard_name");
    info.longName = readStringAttribute(varid, "long_name");
    info.units = readStringAttribute(varid, "units");
    
    // 确定维度类型
    info.type = classifyDimension(dimName);
    
    // 读取坐标值
    info.coordinates = readCoordinateValues(dimName);
    
    // 检查是否规则间隔
    if (info.coordinates.size() > 1) {
        double diff1 = info.coordinates[1] - info.coordinates[0];
        bool regular = true;
        for (size_t i = 2; i < info.coordinates.size() && regular; ++i) {
            double diff = info.coordinates[i] - info.coordinates[i-1];
            if (std::abs(diff - diff1) > 1e-10) {
                regular = false;
            }
        }
        info.isRegular = regular;
        if (regular) {
            info.resolution = diff1;
        }
    }
    
    // 缓存结果
    dimensionCache_[dimName] = info;
    return info;
}

std::vector<DimensionCoordinateInfo> NetCDFCoordinateSystemExtractor::getAllDimensionInfo() const {
    std::vector<DimensionCoordinateInfo> dimensions;
    
    int nvars;
    if (nc_inq_nvars(ncid_, &nvars) == NC_NOERR) {
        for (int varid = 0; varid < nvars; ++varid) {
            char varname[NC_MAX_NAME + 1];
            if (nc_inq_varname(ncid_, varid, varname) == NC_NOERR) {
                if (isDimensionCoordinate(std::string(varname))) {
                    auto dimInfo = extractDimensionInfo(std::string(varname));
                    if (dimInfo) {
                        dimensions.push_back(*dimInfo);
                    }
                }
            }
        }
    }
    
    return dimensions;
}

std::vector<std::string> NetCDFCoordinateSystemExtractor::findDimensionsByType(CoordinateDimension type) const {
    std::vector<std::string> dimensions;
    
    int nvars;
    if (nc_inq_nvars(ncid_, &nvars) == NC_NOERR) {
        for (int varid = 0; varid < nvars; ++varid) {
            char varname[NC_MAX_NAME + 1];
            if (nc_inq_varname(ncid_, varid, varname) == NC_NOERR) {
                std::string dimName(varname);
                if (isDimensionCoordinate(dimName)) {
                    auto dimInfo = extractDimensionInfo(dimName);
                    if (dimInfo && dimInfo->type == type) {
                        dimensions.push_back(dimName);
                    }
                }
            }
        }
    }
    
    return dimensions;
}

std::string NetCDFCoordinateSystemExtractor::findTimeDimension() const {
    auto timeDims = findDimensionsByType(CoordinateDimension::TIME);
    return timeDims.empty() ? "" : timeDims[0];
}

std::string NetCDFCoordinateSystemExtractor::findLongitudeDimension() const {
    auto lonDims = findDimensionsByType(CoordinateDimension::LON);
    return lonDims.empty() ? "" : lonDims[0];
}

std::string NetCDFCoordinateSystemExtractor::findLatitudeDimension() const {
    auto latDims = findDimensionsByType(CoordinateDimension::LAT);
    return latDims.empty() ? "" : latDims[0];
}

std::string NetCDFCoordinateSystemExtractor::findVerticalDimension() const {
    auto vertDims = findDimensionsByType(CoordinateDimension::VERTICAL);
    return vertDims.empty() ? "" : vertDims[0];
}

oscean::core_services::BoundingBox NetCDFCoordinateSystemExtractor::extractRawBoundingBox() const {
    if (cachedBoundingBox_) {
        return *cachedBoundingBox_;
    }
    
    oscean::core_services::BoundingBox bbox;
    bbox.minX = -180.0; bbox.maxX = 180.0;
    bbox.minY = -90.0; bbox.maxY = 90.0;
    
    // 查找经纬度维度
    std::string lonDim = findLongitudeDimension();
    std::string latDim = findLatitudeDimension();
    
    if (!lonDim.empty()) {
        auto lonInfo = extractDimensionInfo(lonDim);
        if (lonInfo && !lonInfo->coordinates.empty()) {
            bbox.minX = *std::min_element(lonInfo->coordinates.begin(), lonInfo->coordinates.end());
            bbox.maxX = *std::max_element(lonInfo->coordinates.begin(), lonInfo->coordinates.end());
        }
    }
    
    if (!latDim.empty()) {
        auto latInfo = extractDimensionInfo(latDim);
        if (latInfo && !latInfo->coordinates.empty()) {
            bbox.minY = *std::min_element(latInfo->coordinates.begin(), latInfo->coordinates.end());
            bbox.maxY = *std::max_element(latInfo->coordinates.begin(), latInfo->coordinates.end());
        }
    }
    
    cachedBoundingBox_ = bbox;
    return bbox;
}

boost::optional<oscean::core_services::BoundingBox> NetCDFCoordinateSystemExtractor::extractVariableRawBounds(const std::string& variableName) const {
    int varid;
    if (nc_inq_varid(ncid_, variableName.c_str(), &varid) != NC_NOERR) {
        return boost::none;
    }
    
    // 检查变量的coordinates属性
    std::string coordinates = readStringAttribute(varid, "coordinates");
    if (!coordinates.empty()) {
        auto coordVars = parseCFCoordinates(coordinates);
        // 基于coordinates属性提取边界
        // 这里可以实现更精确的变量边界提取
    }
    
    // 默认返回全局边界框
    return extractRawBoundingBox();
}

bool NetCDFCoordinateSystemExtractor::isRegularGrid() const {
    std::string lonDim = findLongitudeDimension();
    std::string latDim = findLatitudeDimension();
    
    if (lonDim.empty() || latDim.empty()) {
        return false;
    }
    
    auto lonInfo = extractDimensionInfo(lonDim);
    auto latInfo = extractDimensionInfo(latDim);
    
    return lonInfo && latInfo && lonInfo->isRegular && latInfo->isRegular;
}

// 🚫 已移除：getRawSpatialResolution() - 无任何调用，功能已由空间服务提供
// 原功能：计算原始空间分辨率
// 替代方案：使用 GridDefinition 中的分辨率信息或空间服务计算
// 如需要此功能，请使用空间服务的相关接口

std::vector<std::string> NetCDFCoordinateSystemExtractor::parseCFCoordinates(const std::string& coordinatesAttribute) const {
    std::vector<std::string> coordinates;
    std::istringstream iss(coordinatesAttribute);
    std::string coord;
    
    while (iss >> coord) {
        coordinates.push_back(coord);
    }
    
    return coordinates;
}

CoordinateDimension NetCDFCoordinateSystemExtractor::detectCFAxisType(const std::string& dimName) const {
    return classifyDimension(dimName);
}

// 🚫 已移除：validateCFCompliance() - 无任何调用，功能已由CRS服务提供
// 原功能：验证CF约定合规性
// 替代方案：使用 ICrsService::validateCRSAsync()
// 如需要此功能，请使用CRS服务的验证接口

void NetCDFCoordinateSystemExtractor::clearCache() {
    dimensionCache_.clear();
    cachedCRS_.reset();
    cachedBoundingBox_.reset();
    LOG_INFO("NetCDF坐标系统缓存已清除");
}

void NetCDFCoordinateSystemExtractor::preloadDimensionInfo() {
    LOG_INFO("预加载NetCDF维度信息");
    getAllDimensionInfo();
}

// =============================================================================
// 私有方法实现
// =============================================================================

CoordinateDimension NetCDFCoordinateSystemExtractor::classifyDimension(const std::string& dimName) const {
    int varid;
    if (nc_inq_varid(ncid_, dimName.c_str(), &varid) != NC_NOERR) {
        return CoordinateDimension::UNKNOWN;
    }
    
    // 检查axis属性
    std::string axis = readStringAttribute(varid, "axis");
    if (axis == "X") return CoordinateDimension::LON;
    if (axis == "Y") return CoordinateDimension::LAT;
    if (axis == "Z") return CoordinateDimension::VERTICAL;
    if (axis == "T") return CoordinateDimension::TIME;
    
    // 检查standard_name
    std::string standardName = readStringAttribute(varid, "standard_name");
    if (standardName.find("longitude") != std::string::npos) {
        return CoordinateDimension::LON;
    }
    if (standardName.find("latitude") != std::string::npos) {
        return CoordinateDimension::LAT;
    }
    if (standardName.find("time") != std::string::npos) {
        return CoordinateDimension::TIME;
    }
    
    // 🔧 增强：检查变量名（支持更多常见变量名）
    std::string lowerName = dimName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    
    // 经度识别
    if (lowerName == "lon" || lowerName == "longitude" || lowerName == "x" || 
        lowerName == "long" || lowerName == "xc" || lowerName == "xi_rho" ||
        lowerName.find("lon") != std::string::npos) {
        return CoordinateDimension::LON;
    }
    
    // 纬度识别
    if (lowerName == "lat" || lowerName == "latitude" || lowerName == "y" ||
        lowerName == "lati" || lowerName == "yc" || lowerName == "eta_rho" ||
        lowerName.find("lat") != std::string::npos) {
        return CoordinateDimension::LAT;
    }
    
    // 时间识别
    if (lowerName == "time" || lowerName == "t" || lowerName == "time_counter" ||
        lowerName.find("time") != std::string::npos) {
        return CoordinateDimension::TIME;
    }
    
    // 垂直维度识别
    if (lowerName == "level" || lowerName == "z" || lowerName == "height" || 
        lowerName == "depth" || lowerName == "lev" || lowerName == "plev" ||
        lowerName.find("level") != std::string::npos || lowerName.find("depth") != std::string::npos) {
        return CoordinateDimension::VERTICAL;
    }
    
    return CoordinateDimension::UNKNOWN;
}

bool NetCDFCoordinateSystemExtractor::isDimensionCoordinate(const std::string& dimName) const {
    // 检查是否存在同名的维度和变量
    int dimid, varid;
    return (nc_inq_dimid(ncid_, dimName.c_str(), &dimid) == NC_NOERR) &&
           (nc_inq_varid(ncid_, dimName.c_str(), &varid) == NC_NOERR);
}

std::vector<double> NetCDFCoordinateSystemExtractor::readCoordinateValues(const std::string& dimName) const {
    std::vector<double> values;
    
    int varid;
    if (nc_inq_varid(ncid_, dimName.c_str(), &varid) != NC_NOERR) {
        return values;
    }
    
    // 获取维度长度
    int dimid;
    if (nc_inq_dimid(ncid_, dimName.c_str(), &dimid) != NC_NOERR) {
        return values;
    }
    
    size_t dimlen;
    if (nc_inq_dimlen(ncid_, dimid, &dimlen) != NC_NOERR) {
        return values;
    }
    
    values.resize(dimlen);
    if (nc_get_var_double(ncid_, varid, values.data()) != NC_NOERR) {
        values.clear();
    }
    
    return values;
}

std::string NetCDFCoordinateSystemExtractor::readStringAttribute(int varid, const std::string& attName) const {
    size_t attlen;
    if (nc_inq_attlen(ncid_, varid, attName.c_str(), &attlen) != NC_NOERR) {
        return "";
    }
    
    std::vector<char> attvalue(attlen + 1, 0);
    if (nc_get_att_text(ncid_, varid, attName.c_str(), attvalue.data()) != NC_NOERR) {
        return "";
    }
    
    return std::string(attvalue.data());
}

bool NetCDFCoordinateSystemExtractor::hasAttribute(int varid, const std::string& attName) const {
    size_t attlen;
    return nc_inq_attlen(ncid_, varid, attName.c_str(), &attlen) == NC_NOERR;
}

boost::optional<std::string> NetCDFCoordinateSystemExtractor::findCRSVariable() const {
    int nvars;
    if (nc_inq_nvars(ncid_, &nvars) != NC_NOERR) {
        return boost::none;
    }
    
    for (int varid = 0; varid < nvars; ++varid) {
        char varname[NC_MAX_NAME + 1];
        if (nc_inq_varname(ncid_, varid, varname) == NC_NOERR) {
            std::string name(varname);
            if (name.find("crs") != std::string::npos ||
                name.find("projection") != std::string::npos ||
                hasAttribute(varid, "grid_mapping_name")) {
                return name;
            }
        }
    }
    
    return boost::none;
}

boost::optional<std::string> NetCDFCoordinateSystemExtractor::extractProjectionWKT(int crsVarid) const {
    std::string wkt = readStringAttribute(crsVarid, "spatial_ref");
    if (!wkt.empty()) {
        return wkt;
    }
    
    wkt = readStringAttribute(crsVarid, "crs_wkt");
    if (!wkt.empty()) {
        return wkt;
    }
    
    return boost::none;
}

boost::optional<oscean::core_services::CFProjectionParameters> NetCDFCoordinateSystemExtractor::extractCFProjectionParameters(int varid, const std::string& gridMappingName) const {
    oscean::core_services::CFProjectionParameters cfParams;
    cfParams.gridMappingName = gridMappingName;
    
    // 读取数值属性
    std::vector<std::string> numericAttrs = {
        "latitude_of_projection_origin",
        "longitude_of_projection_origin", 
        "straight_vertical_longitude_from_pole",
        "semi_major_axis",
        "semi_minor_axis",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
        "standard_parallel",
        "longitude_of_central_meridian",
        "latitude_of_standard_parallel"
    };
    
    for (const auto& attr : numericAttrs) {
        // 🔧 修复：使用安全的属性读取方法
        double value = readDoubleAttribute(varid, attr, std::numeric_limits<double>::quiet_NaN());
        if (!std::isnan(value)) {
            cfParams.numericParameters[attr] = value;
        }
    }
    
    // 读取字符串属性
    std::vector<std::string> stringAttrs = {
        "units",
        "proj4",
        "proj4text",
        "crs_wkt",
        "spatial_ref"
    };
    
    for (const auto& attr : stringAttrs) {
        std::string value = readStringAttribute(varid, attr);
        if (!value.empty()) {
            cfParams.stringParameters[attr] = value;
        }
    }
    
    // 验证是否获取到了有效的投影参数
    if (!cfParams.numericParameters.empty() || !cfParams.stringParameters.empty()) {
        LOG_INFO("CF投影参数提取完成: {} 数值参数: {}, 字符串参数: {}", 
                gridMappingName, cfParams.numericParameters.size(), cfParams.stringParameters.size());
        return cfParams;
    }
    
    LOG_WARN("未能提取到有效的CF投影参数: {}", gridMappingName);
    return boost::none;
}

double NetCDFCoordinateSystemExtractor::readDoubleAttribute(int varid, const std::string& attName, double defaultValue) const {
    // 🔧 修复：先检查属性长度，避免栈溢出
    size_t attlen;
    nc_type atttype;
    if (nc_inq_att(ncid_, varid, attName.c_str(), &atttype, &attlen) != NC_NOERR) {
        return defaultValue;
    }
    
    // 如果属性是数组，只读取第一个值
    if (attlen == 1) {
        // 单个值，安全读取
    double value;
    if (nc_get_att_double(ncid_, varid, attName.c_str(), &value) == NC_NOERR) {
        return value;
    }
    } else if (attlen > 1) {
        // 数组属性，读取第一个值
        std::vector<double> values(attlen);
        if (nc_get_att_double(ncid_, varid, attName.c_str(), values.data()) == NC_NOERR) {
            return values[0];
        }
    }
    
    return defaultValue;
}

bool NetCDFCoordinateSystemExtractor::isLongitudeDimension(const std::string& dimName, const DimensionCoordinateInfo& info) const {
    return info.type == CoordinateDimension::LON;
}

bool NetCDFCoordinateSystemExtractor::isLatitudeDimension(const std::string& dimName, const DimensionCoordinateInfo& info) const {
    return info.type == CoordinateDimension::LAT;
}

bool NetCDFCoordinateSystemExtractor::isTimeDimension(const std::string& dimName, const DimensionCoordinateInfo& info) const {
    return info.type == CoordinateDimension::TIME;
}

bool NetCDFCoordinateSystemExtractor::isVerticalDimension(const std::string& dimName, const DimensionCoordinateInfo& info) const {
    return info.type == CoordinateDimension::VERTICAL;
}

// 🚫 已移除：NetCDF PROJ字符串清理和EPSG映射功能 - 无任何调用，功能已统一到CRS服务
// 原功能：cleanNetCDFProjString() - 清理NetCDF PROJ字符串中的冲突参数
// 原功能：tryMapToEPSG() - 尝试将PROJ字符串映射到标准EPSG代码
// 
// 设计决策：
// 1. PROJ字符串处理统一由CRS服务负责
// 2. NetCDF读取器只负责提取原始CRS信息
// 3. 使用 ICrsService::parseFromStringAsync() 进行统一的CRS解析和验证
//
// 替代方案：
// - 原始PROJ字符串保存在 crsInfo.projString 中
// - 由元数据服务调用CRS服务进行标准化处理
// - 确保职责分离和功能统一

} // namespace oscean::core_services::data_access::readers::impl::netcdf 
