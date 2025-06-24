#pragma once

#include <string>
#include <vector>

namespace oscean::core_services::crs::testing {

// WGS84地理坐标系 (EPSG:4326) - WKT1格式
inline const std::string WGS84_WKT = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]";

// WGS84 地理坐标系 (EPSG:4326) - PROJ字符串
inline const std::string WGS84_PROJ = "+proj=longlat +datum=WGS84 +no_defs";

// Web墨卡托投影坐标系 (EPSG:3857) - WKT1格式
inline const std::string WEB_MERCATOR_WKT = "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"X\",EAST],AXIS[\"Y\",NORTH],AUTHORITY[\"EPSG\",\"3857\"]]";

// Web墨卡托投影坐标系 (EPSG:3857) - PROJ字符串
inline const std::string WEB_MERCATOR_PROJ = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs";

// 北极极射投影坐标系 (EPSG:3413) - WKT1格式
inline const std::string POLAR_STEREO_NORTH_WKT = "PROJCS[\"WGS 84 / NSIDC Sea Ice Polaris North Stereographic\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Stereographic_North_Pole\"],PARAMETER[\"latitude_of_origin\",70],PARAMETER[\"central_meridian\",-45],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AUTHORITY[\"EPSG\",\"3413\"]]";

// 北极极射投影坐标系 (EPSG:3413) - PROJ字符串
inline const std::string POLAR_STEREO_NORTH_PROJ = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";

// 南极极射投影坐标系 (EPSG:3031) - PROJ字符串
inline const std::string POLAR_STEREO_SOUTH_PROJ = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";

// 兰伯特等角圆锥投影 (美国州平面坐标系常用，例如 EPSG:2264 - NAD83 / North Carolina) - PROJ字符串
inline const std::string LAMBERT_CONF_NC_PROJ = "+proj=lcc +lat_1=36.16666666666666 +lat_2=34.33333333333334 +lat_0=33.75 +lon_0=-79 +x_0=609601.22 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs";

// 阿尔伯斯等积圆锥投影 (常用于美国全国地图，例如 EPSG:5070 - NAD83 / Conus Albers) - PROJ字符串
inline const std::string ALBERS_USA_PROJ = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs";

// UTM投影坐标系 - 32N区带 (EPSG:32632) - PROJ字符串
inline const std::string UTM_32N_PROJ = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs";

// UTM投影坐标系 - 50N区带 (EPSG:32650) - PROJ字符串
inline const std::string UTM_50N_PROJ = "+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs";

// UTM投影坐标系 - 10S区带 (EPSG:32710) - PROJ字符串
inline const std::string UTM_10S_PROJ = "+proj=utm +zone=10 +south +datum=WGS84 +units=m +no_defs";

// 中国国家2000大地坐标系 (EPSG:4490) - PROJ字符串
inline const std::string CGCS2000_PROJ = "+proj=longlat +ellps=GRS80 +no_defs";

// 中国国家2000投影坐标系 - 3度带 (EPSG:4547) - PROJ字符串
inline const std::string CGCS2000_3_PROJ = "+proj=tmerc +lat_0=0 +lon_0=120 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs";

// 常用椭球体参数表
struct EllipsoidParams {
    std::string name;       // 椭球体名称
    double a;               // 长半轴，单位：米
    double invf;            // 扁率倒数(1/f)
    std::string authority;  // 权威机构编码
};

// 定义常用椭球体参数列表
inline const std::vector<EllipsoidParams> COMMON_ELLIPSOIDS = {
    {"WGS 84", 6378137.0, 298.257223563, "EPSG:7030"},
    {"GRS 1980", 6378137.0, 298.257222101, "EPSG:7019"},
    {"Clarke 1866", 6378206.4, 294.9786982, "EPSG:7008"},
    {"CGCS2000", 6378137.0, 298.257222101, "EPSG:1024"},
    {"Airy 1830", 6377563.396, 299.3249646, "EPSG:7001"},
    {"Bessel 1841", 6377397.155, 299.1528128, "EPSG:7004"},
    {"International 1924", 6378388.0, 297.0, "EPSG:7022"},
    {"Krasovsky 1940", 6378245.0, 298.3, "EPSG:7024"}
};

// 常用大地基准面
struct DatumParams {
    std::string name;       // 大地基准面名称
    std::string ellipsoid;  // 使用的椭球体
    std::string description; // 描述信息
    std::string authority;  // 权威机构编码
};

// 定义常用大地基准面
inline const std::vector<DatumParams> COMMON_DATUMS = {
    {"WGS_1984", "WGS 84", "World Geodetic System 1984", "EPSG:6326"},
    {"NAD83", "GRS 1980", "North American Datum 1983", "EPSG:6269"},
    {"NAD27", "Clarke 1866", "North American Datum 1927", "EPSG:6267"},
    {"CGCS2000", "CGCS2000", "China Geodetic Coordinate System 2000", "EPSG:1043"},
    {"OSGB 1936", "Airy 1830", "Ordnance Survey Great Britain 1936", "EPSG:6277"},
    {"ED50", "International 1924", "European Datum 1950", "EPSG:6230"}
};

} // namespace oscean::core_services::crs::testing 