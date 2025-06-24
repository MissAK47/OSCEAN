/**
 * @file common_data_types.h
 * @brief Defines shared data types for core service interfaces
 */

#pragma once

#ifndef OSCEAN_CORE_SERVICES_COMMON_DATA_TYPES_H
#define OSCEAN_CORE_SERVICES_COMMON_DATA_TYPES_H

// 防止Windows API宏干扰
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// 在任何Windows头文件被包含前取消这些宏定义
#undef min
#undef max
#undef DOMAIN
#undef KEY
#undef TYPE
#undef VALUE
#undef OPTIONAL
#undef ERROR
#undef SUCCESS
#undef FAILED
#undef key
#undef value
#undef domain
#undef type

// --- Standard Library Headers FIRST (Ensure these are before hash specializations) ---
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <memory>     // For std::shared_ptr
#include <cstdint>    // For uint64_t, int64_t etc.
#include <stdexcept>  // For std::out_of_range
#include <any>        // For ModelInput/Output
#include <chrono>     // For Timestamp
#include <functional> // For std::hash
#include <utility>    // For std::pair
#include <cmath>      // For std::isnan, std::isinf etc if needed in GridData or elsewhere
#include <limits>     // For std::numeric_limits
#include <iostream>   // 添加输出流头文件
#include <numeric>    // For std::accumulate if needed for strides
#include <sstream>    // Added for std::basic_stringstream
#include <algorithm>
#include <type_traits>
#include <boost/any.hpp> // 使用 boost::any
#include <boost/variant.hpp> // 确保包含

// --- 🔧 C++17兼容：使用Boost库替代std库 ---
#include <boost/optional.hpp>  // 🔧 C++17兼容：使用boost::optional替代boost::optional

// 在所有包含之后再次取消定义可能的宏
#undef min
#undef max
#undef DOMAIN
#undef KEY
#undef TYPE
#undef VALUE
#undef OPTIONAL
#undef ERROR
#undef SUCCESS
#undef FAILED
#undef key
#undef value
#undef domain
#undef type

// --- Now open the project namespace ---
namespace oscean::core_services
{
    // ======== ADDED DEFINITIONS FROM I_DATA_READER_H ========
    /**
     * @brief 重采样算法枚举
     */
    enum class ResampleAlgorithm
    {
        NEAREST,     ///< Nearest neighbor resampling
        BILINEAR,    ///< Bilinear interpolation
        CUBIC,       ///< Cubic spline interpolation
        CUBICSPLINE, ///< Cubic spline curve
        LANCZOS,     ///< Lanczos window function
        AVERAGE,     ///< Average value resampling
        MODE,        ///< Mode resampling (for categorical data)
        MIN,         ///< Minimum value resampling
        MAX,         ///< Maximum value resampling
        MEDIAN,      ///< Median resampling
        Q1,          ///< First quartile resampling
        Q3           ///< Third quartile resampling
    };

    /**
     * @brief 维度定义结构体
     */
    struct DimensionDefinition
    {
        std::string name;                ///< Dimension name
        size_t size;                     ///< Dimension size
        std::string units;               ///< Units
        std::vector<double> coordinates; ///< Coordinate values
        bool isRegular;                  ///< Whether regularly spaced
        double resolution;               ///< Resolution (if regular)
        std::string type;                ///< Dimension type
        bool isUnlimited;                ///< Whether unlimited dimension

        /**
         * @brief 默认构造函数
         */
        DimensionDefinition() : size(0), isRegular(false), resolution(0.0), type("UNKNOWN"), isUnlimited(false) {}

        /**
         * @brief 构造函数
         */
        DimensionDefinition(const std::string &name, size_t size, const std::string &units = "",
                            bool isRegular = false, double resolution = 0.0,
                            const std::string &type = "UNKNOWN", bool isUnlimited = false)
            : name(name), size(size), units(units), isRegular(isRegular),
              resolution(resolution), type(type), isUnlimited(isUnlimited) {}
    };

    /**
     * @brief 数据集问题类型枚举
     */
    enum class DatasetIssueType
    {
        INFO,    ///< Informational
        WARNING, ///< Warning
        ERROR    ///< Error
    };

    /**
     * @brief 数据集问题结构体
     */
    struct DatasetIssue
    {
        DatasetIssueType type; ///< Issue type
        std::string code;      ///< Issue code
        std::string message;   ///< Issue description
        std::string component; ///< Related component/variable

        /**
         * @brief 构造函数
         */
        DatasetIssue(DatasetIssueType issueType, const std::string &issueCode,
                     const std::string &issueMessage, const std::string &relatedComponent = "")
            : type(issueType), code(issueCode), message(issueMessage), component(relatedComponent) {}
    };
    // ======== END ADDED DEFINITIONS ========

    // For test_main.cpp only
    struct MyTestStruct
    {
        std::string testStringMember;
    };

    /**
     * @brief 值范围模板结构体，用于表示最小和最大值
     */
    template<typename T>
    struct ValueRange {
        // 值范围的属性
        T min;                    ///< [Chinese comment removed for encoding compatibility]
        T max;                    ///< [Chinese comment removed for encoding compatibility]
        bool valid = false;       ///< [Chinese comment removed for encoding compatibility]
        
        // 默认构造函数
        ValueRange() : min(T()), max(T()), valid(false) {}
        
        // 参数构造函数
        ValueRange(const T& minVal, const T& maxVal) 
            : min(minVal), max(maxVal), valid(true) {}
        
        // 判断范围是否有效的方法
        bool isValid() const { return valid && min <= max; }
        
        // 等值运算符
        bool operator==(const ValueRange<T>& other) const {
            return min == other.min && max == other.max && valid == other.valid;
        }
        
        // 不等运算符
        bool operator!=(const ValueRange<T>& other) const {
            return !(*this == other);
        }
    };

    // ======== ENSURE AttributeValue IS DEFINED EARLY AND CORRECTLY ========
    /**
     * @brief 属性值，可以是多种类型之一
     *
     * 用于元数据、要素属性等。
     * 增加了对布尔类型和各种向量类型的支持。
     * Also includes std::monostate for empty/null representation.
     */
    using AttributeValue = std::variant<
        std::monostate, // Added for representing null/empty
        std::string,
        int,
        double,
        bool,
        std::vector<std::string>,
        std::vector<int>,
        std::vector<double>>;

    /**
     * @brief 元数据条目类，表示键值对元数据
     *
     * 该类使用md_前缀命名成员变量以避免与Windows宏或预处理器定义冲突
     */
    class MetadataEntry
    {
    private:
        std::string md_key;    ///< [Chinese comment removed for encoding compatibility]
        std::string md_value;  ///< [Chinese comment removed for encoding compatibility]
        std::string md_domain; ///< [Chinese comment removed for encoding compatibility]
        std::string md_type;   ///< [Chinese comment removed for encoding compatibility]

    public:
        /**
         * @brief 默认构造函数
         */
        MetadataEntry() = default;

        /**
         * @brief 构造函数
         * @param k 键名
         * @param v 值
         * @param d 域名（可选）
         * @param t 类型（可选）
         */
        MetadataEntry(std::string k, std::string v = "", std::string d = "", std::string t = "")
        {
            md_key = std::move(k);
            md_value = std::move(v);
            md_domain = std::move(d);
            md_type = std::move(t);
        }

        /**
         * @brief 获取键名
         * @return 键名
         */
        std::string getKey() const
        {
            return md_key;
        }

        /**
         * @brief 设置键名
         * @param k 键名
         */
        void setKey(std::string k)
        {
            md_key = std::move(k);
        }

        /**
         * @brief 获取值
         * @return 值
         */
        std::string getValue() const
        {
            return md_value;
        }

        /**
         * @brief 设置值
         * @param v 值
         */
        void setValue(std::string v)
        {
            md_value = std::move(v);
        }

        /**
         * @brief 获取域
         * @return 域
         */
        std::string getDomain() const
        {
            return md_domain;
        }

        /**
         * @brief 设置域
         * @param d 域
         */
        void setDomain(std::string d)
        {
            md_domain = std::move(d);
        }

        /**
         * @brief 获取类型
         * @return 类型
         */
        std::string getType() const
        {
            return md_type;
        }

        /**
         * @brief 设置类型
         * @param t 类型
         */
        void setType(std::string t)
        {
            md_type = std::move(t);
        }

        /**
         * @brief 比较运算符
         */
        bool operator==(const MetadataEntry &other) const
        {
            return md_key == other.md_key &&
                   md_value == other.md_value &&
                   md_domain == other.md_domain &&
                   md_type == other.md_type;
        }

        /**
         * @brief 不等运算符
         */
        bool operator!=(const MetadataEntry &other) const
        {
            return !(*this == other);
        }
    };

    /**
     * @brief 数据格式枚举，用于标识不同的数据文件格式
     */
    enum class DataFormat {
        UNKNOWN = 0,         ///< 未知格式
        NETCDF,              ///< NetCDF格式 (.nc, .nc4)
        GDAL_RASTER,         ///< GDAL栅格格式 (.tif, .tiff, .img等)
        GDAL_VECTOR,         ///< GDAL矢量格式 (.shp, .geojson, .gpkg等)
        HDF5,                ///< HDF5格式
        GRIB,                ///< GRIB格式
        ASCII_GRID,          ///< ASCII格式栅格
        CSV,                 ///< CSV格式
        JSON,                ///< JSON格式
        GEOJSON,             ///< GeoJSON格式
        KML,                 ///< KML格式
        GPX,                 ///< GPX格式
        TEXT                 ///< 纯文本格式
    };

    /**
     * @brief 数据类型枚举
     */
    enum class DataType
    {
        Unknown,   ///< [Chinese comment removed for encoding compatibility]
        Byte,      ///< [Chinese comment removed for encoding compatibility]
        UByte,     ///< [Chinese comment removed for encoding compatibility]
        UInt16,    ///< [Chinese comment removed for encoding compatibility]
        Int16,     ///< [Chinese comment removed for encoding compatibility]
        UInt32,    ///< [Chinese comment removed for encoding compatibility]
        Int32,     ///< [Chinese comment removed for encoding compatibility]
        UInt64,    ///< [Chinese comment removed for encoding compatibility]
        Int64,     ///< [Chinese comment removed for encoding compatibility]
        Float32,   ///< [Chinese comment removed for encoding compatibility]
        Float64,   ///< [Chinese comment removed for encoding compatibility]
        String,    ///< [Chinese comment removed for encoding compatibility]
        Boolean,   ///< [Chinese comment removed for encoding compatibility]
        Binary,    ///< [Chinese comment removed for encoding compatibility]
        Complex16, ///< [Chinese comment removed for encoding compatibility]
        Complex32, ///< [Chinese comment removed for encoding compatibility]
        Complex64  ///< [Chinese comment removed for encoding compatibility]
    };

    // --- NEW: Semantic Dimension Type Enum ---
    /**
     * @brief 语义维度类型枚举
     */
    enum class SemanticDimensionType
    {
        UNKNOWN,   ///< [Chinese comment removed for encoding compatibility]
        LONGITUDE, ///< [Chinese comment removed for encoding compatibility]
        LATITUDE,  ///< [Chinese comment removed for encoding compatibility]
        VERTICAL,  ///< [Chinese comment removed for encoding compatibility]/高度维度 (通常是 Z 轴)
        TIME,      ///< [Chinese comment removed for encoding compatibility]
        OTHER      ///< [Chinese comment removed for encoding compatibility]
    };



    // 🔧 操作符重载前向声明，实现在dataTypeToString函数定义之后

    /**
     * @brief 将SemanticDimensionType转换为字符串
     * @param type 语义维度类型枚举值
     * @return 对应的字符串表示
     */
    inline std::string semanticDimensionTypeToString(SemanticDimensionType type)
    {
        switch (type)
        {
        case SemanticDimensionType::LONGITUDE:
            return "LONGITUDE";
        case SemanticDimensionType::LATITUDE:
            return "LATITUDE";
        case SemanticDimensionType::VERTICAL:
            return "VERTICAL";
        case SemanticDimensionType::TIME:
            return "TIME";
        case SemanticDimensionType::OTHER:
            return "OTHER";
        case SemanticDimensionType::UNKNOWN:
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @enum CoordinateDimension
     * @brief 坐标维度类型
     */
    enum class CoordinateDimension
    {
        LON,      ///< [Chinese comment removed for encoding compatibility]
        LAT,      ///< [Chinese comment removed for encoding compatibility]
        TIME,     ///< [Chinese comment removed for encoding compatibility]
        VERTICAL, ///< [Chinese comment removed for encoding compatibility]
        SPECTRAL, ///< [Chinese comment removed for encoding compatibility]/波段维度
        OTHER,    ///< [Chinese comment removed for encoding compatibility]
        NONE      ///< [Chinese comment removed for encoding compatibility]
    };

    /**
     * @brief 维度坐标信息结构体
     * 用于统一表示各类维度的坐标信息
     */
    struct DimensionCoordinateInfo
    {
        std::string name;                                     ///< [Chinese comment removed for encoding compatibility]
        std::string standardName;                             ///< [Chinese comment removed for encoding compatibility]
        std::string longName;                                 ///< [Chinese comment removed for encoding compatibility]
        std::string units;                                    ///< [Chinese comment removed for encoding compatibility]
        CoordinateDimension type = CoordinateDimension::NONE; ///< [Chinese comment removed for encoding compatibility]
        bool isRegular = false;                               ///< [Chinese comment removed for encoding compatibility]
        double resolution = 0.0;                              ///< [Chinese comment removed for encoding compatibility]
        std::vector<double> coordinates;                      ///< [Chinese comment removed for encoding compatibility]
        std::vector<std::string> coordinateLabels;            ///< [Chinese comment removed for encoding compatibility]/标签型维度)
        std::map<std::string, std::string> attributes;        ///< [Chinese comment removed for encoding compatibility]
        double minValue = 0.0;                                ///< [Chinese comment removed for encoding compatibility]
        double maxValue = 0.0;                                ///< [Chinese comment removed for encoding compatibility]
        bool hasValueRange = false;                           ///< [Chinese comment removed for encoding compatibility]
        ValueRange<double> valueRange;                        ///< [Chinese comment removed for encoding compatibility]

        /**
         * @brief 默认构造函数
         */
        DimensionCoordinateInfo() = default;

        /**
         * @brief 获取维度中的级别/点数
         * @return 坐标或标签的数量
         */
        size_t getNumberOfLevels() const
        {
            if (!coordinates.empty())
                return coordinates.size();
            if (!coordinateLabels.empty())
                return coordinateLabels.size();
            return 0;
        }

        /**
         * @brief 检查是否有数值坐标
         * @return 如果有数值坐标则返回true
         */
        bool hasNumericCoordinates() const { return !coordinates.empty(); }

        /**
         * @brief 检查是否有文本标签
         * @return 如果有文本标签则返回true
         */
        bool hasTextualLabels() const { return !coordinateLabels.empty(); }

        /**
         * @brief 比较运算符
         */
        bool operator==(const DimensionCoordinateInfo &other) const
        {
            return name == other.name &&
                   standardName == other.standardName &&
                   longName == other.longName &&
                   units == other.units &&
                   type == other.type &&
                   isRegular == other.isRegular &&
                   resolution == other.resolution &&
                   coordinates == other.coordinates &&
                   coordinateLabels == other.coordinateLabels &&
                   minValue == other.minValue &&
                   maxValue == other.maxValue &&
                   hasValueRange == other.hasValueRange &&
                   valueRange == other.valueRange;
            // 注：attributes 未包含在比较中，因为它可能包含非关键信息
        }
    };

    // --- NEW: Dimension Detail Structure ---
    /**
     * @brief 通用维度详细信息结构体
     */
    struct DimensionDetail
    {
        std::string name;                                   ///< 维度名称
        size_t size = 0;                                    ///< 维度大小
        std::string units;                                  ///< 关联坐标变量的单位（原始）
        std::vector<double> coordinates;                    ///< 关联坐标变量的数值（原始）
        std::map<std::string, std::string> attributes;      ///< 关联坐标变量的属性（原始）

        bool operator==(const DimensionDetail& other) const {
            return name == other.name && size == other.size;
        }
    };

    /**
     * @struct Point
     * @brief 表示一个二维或三维空间点。
     */
    struct Point
    {
        double x;
        double y;
        boost::optional<double> z;          // 可选的Z坐标
        boost::optional<std::string> crsId; // 点坐标对应的CRS标识符 (可选，因为有时CRS是上下文已知的)

        // 构造函数
        Point(double x_val, double y_val, boost::optional<double> z_val = boost::none, boost::optional<std::string> crs_id_val = boost::none)
            : x(x_val), y(y_val), z(z_val), crsId(std::move(crs_id_val)) {}

        // 添加比较操作符
        bool operator==(const Point &other) const
        {
            return x == other.x && y == other.y && z == other.z && crsId == other.crsId;
        }

        bool operator!=(const Point &other) const
        {
            return !(*this == other);
        }
    };

    /**
     * @struct BoundingBox
     * @brief 表示一个地理或投影坐标系下的边界框。
     */
    struct BoundingBox
    {
        double minX;
        double minY;
        double maxX;
        double maxY;
        boost::optional<double> minZ; // 可选的最小Z值
        boost::optional<double> maxZ; // 可选的最大Z值
        std::string crsId;          // 边界框坐标对应的CRS标识符

        // 构造函数
        BoundingBox(double min_x = 0.0, double min_y = 0.0, double max_x = 0.0, double max_y = 0.0,
                    boost::optional<double> min_z = boost::none, boost::optional<double> max_z = boost::none,
                    std::string crs_id = "")
            : minX(min_x), minY(min_y), maxX(max_x), maxY(max_y), minZ(min_z), maxZ(max_z), crsId(std::move(crs_id)) {}

        // 工具方法
        bool isValid() const
        {
            return (maxX > minX) && (maxY > minY) &&
                   (!minZ.has_value() || !maxZ.has_value() || maxZ.value() > minZ.value());
        }

        // 添加比较操作符
        bool operator==(const BoundingBox &other) const
        {
            return minX == other.minX && minY == other.minY &&
                   maxX == other.maxX && maxY == other.maxY &&
                   minZ == other.minZ && maxZ == other.maxZ &&
                   crsId == other.crsId;
        }

        bool operator!=(const BoundingBox &other) const
        {
            return !(*this == other);
        }
    };

    /**
     * @brief CRS定义类型枚举
     */
    enum class CRSType
    {
        UNKNOWN,        ///< [Chinese comment removed for encoding compatibility]
        WKT1,           ///< [Chinese comment removed for encoding compatibility]
        WKT2,           ///< [Chinese comment removed for encoding compatibility]
        PROJ_STRING,    ///< [Chinese comment removed for encoding compatibility]
        EPSG_CODE,      ///< [Chinese comment removed for encoding compatibility]
        URN_OGC_DEF_CRS ///< [Chinese comment removed for encoding compatibility]
    };

    /**
     * @brief CF约定投影参数结构
     * 用于在数据访问服务和CRS服务之间传递原始CF投影参数
     */
    struct CFProjectionParameters {
        std::string gridMappingName;                      ///< CF投影类型名称(如polar_stereographic)
        std::map<std::string, double> numericParameters;  ///< 数值参数(如latitude_of_projection_origin)
        std::map<std::string, std::string> stringParameters; ///< 字符串参数(如units)
        
        // 常用投影参数的便捷访问器
        boost::optional<double> getLatitudeOfProjectionOrigin() const {
            auto it = numericParameters.find("latitude_of_projection_origin");
            return it != numericParameters.end() ? boost::optional<double>(it->second) : boost::none;
        }
        
        boost::optional<double> getLongitudeOfProjectionOrigin() const {
            auto it = numericParameters.find("longitude_of_projection_origin");
            return it != numericParameters.end() ? boost::optional<double>(it->second) : boost::none;
        }
        
        boost::optional<double> getSemiMajorAxis() const {
            auto it = numericParameters.find("semi_major_axis");
            return it != numericParameters.end() ? boost::optional<double>(it->second) : boost::none;
        }
        
        boost::optional<double> getSemiMinorAxis() const {
            auto it = numericParameters.find("semi_minor_axis");
            return it != numericParameters.end() ? boost::optional<double>(it->second) : boost::none;
        }
        
        boost::optional<double> getScaleFactor() const {
            auto it = numericParameters.find("scale_factor_at_projection_origin");
            return it != numericParameters.end() ? boost::optional<double>(it->second) : boost::none;
        }
        
        boost::optional<double> getFalseEasting() const {
            auto it = numericParameters.find("false_easting");
            return it != numericParameters.end() ? boost::optional<double>(it->second) : boost::none;
        }
        
        boost::optional<double> getFalseNorthing() const {
            auto it = numericParameters.find("false_northing");
            return it != numericParameters.end() ? boost::optional<double>(it->second) : boost::none;
        }
        
        boost::optional<std::string> getUnits() const {
            auto it = stringParameters.find("units");
            return it != stringParameters.end() ? boost::optional<std::string>(it->second) : boost::none;
        }
        
        // 比较运算符
        bool operator==(const CFProjectionParameters& other) const {
            return gridMappingName == other.gridMappingName &&
                   numericParameters == other.numericParameters &&
                   stringParameters == other.stringParameters;
        }
    };

    /**
     * @struct CRSInfo
     * @brief 坐标参考系统信息
     */
    struct CRSInfo
    {
        // 新的字段定义
        std::string authorityName;                  // 权威组织名称，如"EPSG"
        std::string authorityCode;                  // 权威代码，如"4326"
        std::string wktext;                         // WKT格式的完整CRS描述
        std::string wkt;                            // WKT格式的完整CRS描述（别名，兼容旧代码）
        std::string projString;                     // PROJ格式的字符串描述
        std::string proj4text;                      // PROJ格式的字符串描述（别名，兼容旧代码）
        bool isGeographic = false;                  // 是否为地理坐标系(经纬度)
        bool isProjected = false;                   // 是否为投影坐标系
        boost::optional<int> epsgCode = boost::none; // EPSG代码，如果可用

        // 单位信息
        std::string linearUnitName;       // 线性单位名称
        double linearUnitToMeter = 1.0;   // 线性单位到米的转换系数
        std::string angularUnitName;      // 角度单位名称
        double angularUnitToRadian = 1.0; // 角度单位到弧度的转换系数

        // 兼容旧代码的字段
        std::string id;                            // 唯一标识符，可以是EPSG代码或其他标识
        std::string name;                          // 名称描述
        std::string authority;                     // 权威机构（如EPSG、ESRI等）
        std::string code;                          // 权威机构提供的代码
        std::string type;                          // CRS类型（如地理、投影、复合等）
        CRSType definitionType = CRSType::UNKNOWN; // 定义类型，兼容旧代码
        
        // 扩展参数字段，用于存储投影特定参数和元数据
        std::map<std::string, std::string> parameters; // 投影参数和扩展信息

        // CF约定投影参数（用于数据访问服务和CRS服务之间传递原始投影参数）
        boost::optional<CFProjectionParameters> cfParameters; // CF约定投影参数

        // 构造函数
        CRSInfo() = default;

        // 构造函数，从基本信息初始化
        CRSInfo(const std::string &auth, const std::string &authCode, const std::string &wktText)
            : authorityName(auth), authorityCode(authCode), wktext(wktText), wkt(wktText),
              id(auth + ":" + authCode), authority(auth), code(authCode)
        {
            if (auth == "EPSG" && !authCode.empty())
            {
                try
                {
                    epsgCode = std::stoi(authCode);
                }
                catch (...)
                {
                    // 转换失败，保持nullopt
                }
            }
        }

        // 比较运算符，用于容器等操作
        bool operator==(const CRSInfo &other) const
        {
            // 如果有EPSG代码，优先比较EPSG代码
            if (epsgCode.has_value() && other.epsgCode.has_value())
            {
                return epsgCode.value() == other.epsgCode.value();
            }

            // 比较权威名称和代码
            if (!authorityName.empty() && !authorityCode.empty() &&
                !other.authorityName.empty() && !other.authorityCode.empty())
            {
                return (authorityName == other.authorityName &&
                        authorityCode == other.authorityCode);
            }

            // 否则比较WKT文本
            if (!wktext.empty() && !other.wktext.empty())
            {
                return wktext == other.wktext;
            }

            // 检查兼容性字段
            if (!wkt.empty() && !other.wkt.empty())
            {
                return wkt == other.wkt;
            }

            // 兼容旧代码的比较
            return id == other.id && name == other.name &&
                   authority == other.authority && code == other.code &&
                   proj4text == other.proj4text && type == other.type &&
                   parameters == other.parameters;
        }

        bool operator!=(const CRSInfo &other) const
        {
            return !(*this == other);
        }

        bool operator<(const CRSInfo &other) const
        {
            return id < other.id;
        }
    };

    /**
     * @brief 索引范围结构体，用于表示数据访问的索引范围
     */
    struct IndexRange
    {
        int start = 0;  ///< [Chinese comment removed for encoding compatibility]
        int count = -1; ///< [Chinese comment removed for encoding compatibility]

        /**
         * @brief 默认构造函数
         */
        IndexRange() = default;

        /**
         * @brief 构造函数
         * @param start 起始索引
         * @param count 数量
         */
        IndexRange(int start, int count) : start(start), count(count) {}

        /**
         * @brief 判断索引范围是否有效
         * @return 如果有效返回true，否则返回false
         */
        bool isValid() const
        {
            // start 必须非负，count 不能是 0 (可以是负数表示读取到末尾)
            return count != 0 && start >= 0;
        }

        /**
         * @brief 判断索引范围是否为空
         * @return 如果为空返回true，否则返回false
         */
        bool isEmpty() const
        {
            return count == 0;
        }

        /**
         * @brief 判断索引范围是否为全部
         * @return 如果count为-1返回true，否则返回false
         */
        bool isAll() const
        {
            return count == -1;
        }

        bool operator==(const IndexRange &other) const
        {
            return start == other.start && count == other.count;
        }
    };

    /**
     * @brief 时间单位枚举
     */
    enum class TimeUnit
    {
        Unknown,
        Seconds,
        Minutes,
        Hours,
        Days,
        Months, // Note: Month duration can be ambiguous
        Years   // Note: Year duration can be ambiguous (e.g. leap year)
    };

    /**
     * @brief 坐标轴的语义类型
     */
    enum class CoordinateType
    {
        Unknown, ///< [Chinese comment removed for encoding compatibility]
        X,       ///< [Chinese comment removed for encoding compatibility]
        Y,       ///< [Chinese comment removed for encoding compatibility]
        Z,       ///< [Chinese comment removed for encoding compatibility]
        Time,    ///< [Chinese comment removed for encoding compatibility]
        Generic  ///< [Chinese comment removed for encoding compatibility]
    };

    /**
     * @brief 表示一个坐标变量/轴的信息
     */
    struct CoordinateVariable
    {
        std::string name;                              ///< [Chinese comment removed for encoding compatibility]
        std::string standardName;                      ///< [Chinese comment removed for encoding compatibility]
        std::string longName;                          ///< [Chinese comment removed for encoding compatibility]
        std::string units;                             ///< [Chinese comment removed for encoding compatibility]
        CoordinateType type = CoordinateType::Unknown; ///< [Chinese comment removed for encoding compatibility]
        std::vector<double> values;                    ///< [Chinese comment removed for encoding compatibility]
        // Potentially other attributes like `axis`, `positive` for CF

        CoordinateVariable() = default;

        bool operator==(const CoordinateVariable &other) const
        {
            return name == other.name &&
                   standardName == other.standardName &&
                   longName == other.longName &&
                   units == other.units &&
                   type == other.type &&
                   values == other.values;
        }
    };

    /**
     * @brief 表示一个坐标系统
     */
    struct CoordinateSystem
    {
        std::string name;                        ///< [Chinese comment removed for encoding compatibility]
        CRSInfo crs;                             ///< [Chinese comment removed for encoding compatibility]
        boost::optional<CoordinateVariable> xAxis; ///< [Chinese comment removed for encoding compatibility]
        boost::optional<CoordinateVariable> yAxis; ///< [Chinese comment removed for encoding compatibility]
        boost::optional<CoordinateVariable> zAxis; ///< [Chinese comment removed for encoding compatibility]
        boost::optional<CoordinateVariable> tAxis; ///< [Chinese comment removed for encoding compatibility]

        CoordinateSystem() = default;

        bool operator==(const CoordinateSystem &other) const
        {
            return name == other.name &&
                   crs == other.crs &&
                   xAxis == other.xAxis &&
                   yAxis == other.yAxis &&
                   zAxis == other.zAxis &&
                   tAxis == other.tAxis;
        }
    };

    /**
     * @brief 时间戳类型 (使用64位无符号整数)
     */
    using Timestamp = uint64_t; // Unified to uint64_t

    /**
     * @brief 辅助函数：解析时间戳字符串 (示例，具体实现可能需要日期库)
     * @param timestampStr 时间戳字符串
     * @return Timestamp 时间戳
     */
    inline Timestamp parseTimestamp(const std::string &timestampStr)
    {
        // 实际实现应使用日期时间库解析各种格式
        try
        {
            // 简化示例：假设是简单的Unix时间戳数字
            return std::stoull(timestampStr);
        }
        catch (const std::exception & /*e*/)
        { // 省略未使用的变量 e
            // 处理解析错误，例如返回0或抛出异常
            // std::cerr << "Error parsing timestamp: " << e.what() << std::endl;
            throw std::invalid_argument("Invalid timestamp format");
        }
        // return 0; // Placeholder - Removed unreachable code
    }

      /**
     * @brief 属性过滤器（用于矢量数据查询）
     */
    struct AttributeFilter
    {
        std::string attributeName; ///< [Chinese comment removed for encoding compatibility]
        std::string operation;     ///< [Chinese comment removed for encoding compatibility]
        std::string value;         ///< [Chinese comment removed for encoding compatibility]

        /**
         * @brief 构造函数
         * @param attributeName 属性名称
         * @param operation 操作
         * @param value 属性值
         */
        AttributeFilter(const std::string &attributeName,
                        const std::string &operation,
                        const std::string &value)
            : attributeName(attributeName), operation(operation), value(value) {}
    };

    /**
     * @brief 定义网格的结构
     */
    struct GridDefinition
    {
        size_t rows = 0;
        size_t cols = 0;
        BoundingBox extent; // 使用 BoundingBox
        double xResolution = 0.0;
        double yResolution = 0.0;
        CRSInfo crs; // CRS信息也可以放在这里，或者依赖 extent 中的 CRS

        // --- NEW: 维度信息 ---
        std::string gridName;                                        ///< [Chinese comment removed for encoding compatibility]
        DimensionCoordinateInfo xDimension;                          ///< [Chinese comment removed for encoding compatibility]
        DimensionCoordinateInfo yDimension;                          ///< [Chinese comment removed for encoding compatibility]
        DimensionCoordinateInfo zDimension;                          ///< [Chinese comment removed for encoding compatibility]
        DimensionCoordinateInfo tDimension;                          ///< [Chinese comment removed for encoding compatibility]
        std::vector<DimensionCoordinateInfo> dimensions;             ///< [Chinese comment removed for encoding compatibility]
        DataType originalDataType = DataType::Unknown;               ///< [Chinese comment removed for encoding compatibility]
        std::vector<CoordinateDimension> dimensionOrderInDataLayout; ///< [Chinese comment removed for encoding compatibility]
        std::map<std::string, std::string> globalAttributes;         ///< [Chinese comment removed for encoding compatibility]

        // --- NEW: 辅助方法 ---
        /**
         * @brief 检查是否有X维度
         * @return 如果X维度有效则返回true
         */
        bool hasXDimension() const
        {
            return xDimension.type != CoordinateDimension::NONE && xDimension.getNumberOfLevels() > 0;
        }

        /**
         * @brief 检查是否有Y维度
         * @return 如果Y维度有效则返回true
         */
        bool hasYDimension() const
        {
            return yDimension.type != CoordinateDimension::NONE && yDimension.getNumberOfLevels() > 0;
        }

        /**
         * @brief 检查是否有Z维度
         * @return 如果Z维度有效则返回true
         */
        bool hasZDimension() const
        {
            return zDimension.type != CoordinateDimension::NONE && zDimension.getNumberOfLevels() > 0;
        }

        /**
         * @brief 检查是否有T维度
         * @return 如果T维度有效则返回true
         */
        bool hasTDimension() const
        {
            return tDimension.type != CoordinateDimension::NONE && tDimension.getNumberOfLevels() > 0;
        }

        /**
         * @brief 获取指定维度类型的级别数
         * @param dimType 维度类型
         * @return 级别数量，如果维度不存在则返回0
         */
        size_t getLevelsForDimension(CoordinateDimension dimType) const
        {
            if (xDimension.type == dimType)
                return xDimension.getNumberOfLevels();
            if (yDimension.type == dimType)
                return yDimension.getNumberOfLevels();
            if (zDimension.type == dimType)
                return zDimension.getNumberOfLevels();
            if (tDimension.type == dimType)
                return tDimension.getNumberOfLevels();
            return 0;
        }

        /**
         * @brief 比较运算符
         */
        bool operator==(const GridDefinition &other) const
        {
            return rows == other.rows &&
                   cols == other.cols &&
                   extent == other.extent &&
                   xResolution == other.xResolution &&
                   yResolution == other.yResolution &&
                   crs == other.crs &&
                   gridName == other.gridName &&
                   xDimension == other.xDimension &&
                   yDimension == other.yDimension &&
                   zDimension == other.zDimension &&
                   tDimension == other.tDimension;
            // 注：globalAttributes 未包含在比较中
        }
    };

    /**
     * @brief 表示网格数据的类，包含数据、元数据和坐标信息
     */
    class GridData : public std::enable_shared_from_this<GridData>
    {
    public:
        // ===== 新增1：内存布局标识（轻量级） =====
        enum class MemoryLayout : uint8_t {
            ROW_MAJOR = 0,     // 默认，C风格
            COLUMN_MAJOR = 1,  // Fortran风格
            UNKNOWN = 2        // 未知/自定义
        };
        
        // ===== 新增2：访问模式提示（用于优化） =====
        enum class AccessPattern : uint8_t {
            RANDOM = 0,        // 随机访问
            SEQUENTIAL_X = 1,  // X方向顺序访问
            SEQUENTIAL_Y = 2,  // Y方向顺序访问
            SEQUENTIAL_Z = 3,  // Z方向（深度）顺序访问
            BLOCK_2D = 4,      // 2D块访问
            UNKNOWN = 5        // 未知模式
        };

        // --- 构造函数 ---
        GridData() = default;
        GridData(const GridData &) = delete;  // 删除拷贝构造函数，因为包含unique_ptr
        GridData(GridData &&) = default;
        GridData &operator=(const GridData &) = delete;  // 删除拷贝赋值运算符，因为包含unique_ptr
        GridData &operator=(GridData &&) = default;

        /**
         * @brief 构造函数
         * @param width 宽度（列数）
         * @param height 高度（行数）
         * @param bands 波段数
         * @param type 数据类型
         */
        inline GridData(size_t width, size_t height, size_t bands, DataType type)
            : _dataType(type), _bandCount(bands)
        {
            _definition.cols = width;
            _definition.rows = height;
            // Important: _definition.crs, _definition.extent etc. are NOT set here by this constructor
            // _crs is also not set here.

            size_t elementSize = getElementSizeBytes();
            if (elementSize == 0 && (width > 0 || height > 0 || bands > 0) && type != DataType::Unknown)
            {
                throw std::runtime_error("GridData: Attempted to create grid with zero element size for type: " + std::to_string(static_cast<int>(type)));
            }

            size_t totalSizeBytes = width * height * bands * elementSize;
            if (totalSizeBytes > 0)
            {
                try
                {
                    _buffer.resize(totalSizeBytes);
                }
                catch (const std::bad_alloc &e)
                {
                    throw std::runtime_error("GridData: Failed to allocate buffer of size " + std::to_string(totalSizeBytes) + ". " + std::string(e.what()));
                }
            }
            else
            {
                _buffer.clear();
            }

            // 🔧 优化：仅在需要时同步public成员，避免完整拷贝
            // data缓冲区延迟同步，仅在访问时进行
            this->dataType = _dataType;
            this->definition = _definition; // Public definition gets PARTIALLY initialized private _definition
            this->crs = _crs;               // Public crs gets default private _crs
            
            // 🔧 性能优化：避免完整缓冲区拷贝
            // this->data = _buffer;  // 移除：避免内存双倍占用

            // 初始化Z维度（波段）信息
            // 当使用此构造函数时，通常使用简单的波段索引作为Z坐标
            if (bands > 0)
            {
                _definition.zDimension.name = "band";
                _definition.zDimension.type = CoordinateDimension::SPECTRAL;
                _definition.zDimension.coordinates.resize(bands);
                for (size_t i = 0; i < bands; ++i)
                {
                    _definition.zDimension.coordinates[i] = static_cast<double>(i);
                }
            }
        }

        /**
         * @brief 使用完整的网格定义构造GridData
         * @param complete_definition 网格定义
         * @param data_type_param 数据类型
         * @param band_count_param 波段数（默认为1）
         */
        inline GridData(const GridDefinition &complete_definition, DataType data_type_param, size_t band_count_param = 1)
            : _definition(complete_definition),
              _dataType(data_type_param),
              _bandCount(band_count_param),
              _crs(complete_definition.crs)
        { // Initialize _crs from the provided complete_definition's CRS

            size_t elementSize = getElementSizeBytes(); // Uses private _dataType (now set)
            if (elementSize == 0 && (_definition.rows > 0 || _definition.cols > 0 || _bandCount > 0) && _dataType != DataType::Unknown)
            {
                throw std::runtime_error("GridData (new ctor): Attempted to create grid with zero element size for type: " + std::to_string(static_cast<int>(_dataType)));
            }

            size_t totalSizeBytes = _definition.rows * _definition.cols * _bandCount * elementSize;
            if (totalSizeBytes > 0)
            {
                try
                {
                    _buffer.resize(totalSizeBytes);
                }
                catch (const std::bad_alloc &e)
                {
                    throw std::runtime_error("GridData (new ctor): Failed to allocate buffer of size " + std::to_string(totalSizeBytes) + ". " + std::string(e.what()));
                }
            }
            else
            {
                _buffer.clear();
            }

            // 🔧 优化：仅同步非缓冲区成员，避免内存双倍占用
            // IF public members are to be kept. Ideally, they would be removed.
            // this->data = _buffer;  // 移除：避免完整缓冲区拷贝
            this->dataType = _dataType;
            this->definition = _definition;
            this->crs = _crs;
        }

        /**
         * @brief 获取波段坐标值
         * @return 波段坐标值数组
         * 注：此方法是为了保持向后兼容性，实际应使用 definition.zDimension.coordinates
         */
        const std::vector<double> &getBandCoordinates() const
        {
            return _definition.zDimension.coordinates;
        }

        /**
         * @brief 获取指定索引的波段坐标值
         * @param index 波段索引
         * @return 波段坐标值，如果超出范围则返回NaN
         * 注：此方法是为了保持向后兼容性，实际应使用 definition.zDimension.coordinates
         */
        double getBandCoordinate(size_t index) const
        {
            if (index < _definition.zDimension.coordinates.size())
            {
                return _definition.zDimension.coordinates[index];
            }
            return std::numeric_limits<double>::quiet_NaN();
        }

        /**
         * @brief 设置波段坐标
         * @param coords 坐标值数组
         * @param name 坐标名称（默认为"band"）
         * @param unit 坐标单位（默认为空）
         * 注：此方法是为了保持向后兼容性，实际应直接修改 definition.zDimension
         */
        void setBandCoordinates(const std::vector<double> &coords,
                                const std::string &name = "band",
                                const std::string &unit = "")
        {
            _definition.zDimension.coordinates = coords;
            _definition.zDimension.name = name;
            _definition.zDimension.units = unit;
            _definition.zDimension.type = CoordinateDimension::SPECTRAL;

            // 同步到公共成员
            definition.zDimension = _definition.zDimension;
        }

        // --- 访问器 ---
        inline const GridDefinition &getDefinition() const
        {
            return _definition;
        }

        inline DataType getDataType() const
        {
            return _dataType;
        }
        
        /**
         * @brief 获取网格宽度（列数）
         * @return 宽度
         */
        inline size_t getWidth() const
        {
            return _definition.cols;
        }
        
        /**
         * @brief 获取网格高度（行数）
         * @return 高度
         */
        inline size_t getHeight() const
        {
            return _definition.rows;
        }
        
        /**
         * @brief 获取波段数量
         * @return 波段数量
         */
        inline size_t getBandCount() const
        {
            return _bandCount;
        }
        
        /**
         * @brief 获取坐标系统信息
         * @return 坐标系统信息
         */
        inline const CRSInfo& getCoordinateSystem() const
        {
            return _crs;
        }
        
        /**
         * @brief 获取空间范围
         * @return 空间范围
         */
        inline const BoundingBox& getSpatialExtent() const
        {
            return _definition.extent;
        }
        
        /**
         * @brief 获取元数据
         * @return 元数据映射
         */
        inline const std::map<std::string, std::string>& getMetadata() const
        {
            return _definition.globalAttributes;
        }
        
        /**
         * @brief 检查是否有颜色表
         * @return 如果有颜色表返回true
         */
        inline bool hasColorTable() const
        {
            // 简化实现，实际应该检查颜色表数据
            return false;
        }
        
        /**
         * @brief 检查是否有NoData值
         * @return 如果有NoData值返回true
         */
        inline bool hasNoDataValue() const
        {
            return _fillValue.has_value();
        }

        /**
         * @brief 获取单个元素的大小（字节）
         * @return 单个元素的大小（字节）
         */
        inline size_t getElementSizeBytes() const
        { // New public method
            switch (getDataType())
            { // Uses public getDataType()
            case DataType::Byte:
                return sizeof(uint8_t);
            case DataType::UInt16:
                return sizeof(uint16_t);
            case DataType::Int16:
                return sizeof(int16_t);
            case DataType::UInt32:
                return sizeof(uint32_t);
            case DataType::Int32:
                return sizeof(int32_t);
            case DataType::UInt64:
                return sizeof(uint64_t);
            case DataType::Int64:
                return sizeof(int64_t);
            case DataType::Float32:
                return sizeof(float); // Same as GDT_Float32
            case DataType::Float64:
                return sizeof(double); // Same as GDT_Float64
            case DataType::Complex16:
                return sizeof(int16_t) * 2;
            case DataType::Complex32:
                return sizeof(float) * 2;
            case DataType::Complex64:
                return sizeof(double) * 2;
            // String, Boolean, Binary, Unknown might not have a fixed size in this context
            // or their size is handled differently (e.g., for String, it's variable).
            // For buffer allocation purposes, 0 might be appropriate for these or throw an error.
            case DataType::String:
            case DataType::Boolean: // Often stored as byte, but depends on convention
            case DataType::Binary:
            case DataType::Unknown:
            default:
                return 0; // Or throw std::runtime_error for unhandled types
            }
        }

        size_t getDataSizeBytes() const
        {
            // 根据数据类型返回每个元素的大小
            switch (dataType)
            {
            case DataType::Byte:
                return sizeof(char);
            case DataType::UInt16:
            case DataType::Int16:
                return sizeof(short);
            case DataType::UInt32:
            case DataType::Int32:
                return sizeof(int);
            case DataType::UInt64:
            case DataType::Int64:
                return sizeof(long long);
            case DataType::Float32:
                return sizeof(float);
            case DataType::Float64:
                return sizeof(double);
            case DataType::Complex16:
                return 2 * sizeof(int16_t);
            case DataType::Complex32:
                return 2 * sizeof(float);
            case DataType::Complex64:
                return 2 * sizeof(double);
            default:
                return sizeof(char); // 默认返回1字节
            }
        }

        size_t getTotalDataSize() const; // 获取总数据大小（字节）
        const void *getDataPtr() const
        {
            return _buffer.data();
        }
        void *getDataPtrMutable()
        {
            // Example implementation: Requires _buffer to be the actual data store
            return _buffer.data();
        }
        size_t getSizeInBytes() const;
        inline const std::vector<double> &getGeoTransform() const
        {
            return _geoTransform;
        }
        inline const CRSInfo &getCRS() const
        {
            return _crs;
        }
        inline const std::string &getVariableName() const { return _variableName; }
        inline const std::string &getUnits() const { return _units; }
        inline boost::optional<double> getFillValue() const { return _fillValue; }

        // 获取特定位置的值 (需要类型转换)
        template <typename T>
        T getValue(size_t row, size_t col, size_t band) const;

        // --- 修改器 ---
        inline void setGeoTransform(const std::vector<double> &transform) {
            _geoTransform = transform;
            // definition中没有geoTransform成员，不需要同步
        }
        
        inline void setCrs(const CRSInfo &newCrs) {
            _crs = newCrs;
            // 同步到公共成员以保持兼容性
            this->crs = newCrs;
            // 更新网格定义中的CRS
            _definition.crs = newCrs;
            this->definition.crs = newCrs;
        }
        inline void setVariableName(const std::string &name) { _variableName = name; }
        inline void setUnits(const std::string &units) { _units = units; }
        inline void setFillValue(boost::optional<double> fillValue)
        {
            _fillValue = fillValue;
        }
        inline void setNoDataValue(double noDataValue)
        {
            setFillValue(noDataValue);
        }

        // 设置特定位置的值 (需要类型转换)
        template <typename T>
        void setValue(size_t row, size_t col, size_t band, T value);

        // 根据地理变换和起始索引填充坐标 (假设规则网格)
        void populateCoordinates(int xStartIndex = 0, int yStartIndex = 0);

        // 🔧 统一缓冲区访问接口（优化的第一步）
        /**
         * @brief 获取统一数据缓冲区引用（推荐使用）
         * @return 内部缓冲区的引用
         */
        std::vector<unsigned char>& getUnifiedBuffer() { return _buffer; }
        
        /**
         * @brief 获取统一数据缓冲区引用（只读）
         * @return 内部缓冲区的常量引用
         */
        const std::vector<unsigned char>& getUnifiedBuffer() const { return _buffer; }
        
        /**
         * @brief 获取缓冲区数据指针（性能关键路径）
         * @return 缓冲区数据指针
         */
        unsigned char* getUnifiedBufferData() { return _buffer.data(); }
        
        /**
         * @brief 获取缓冲区数据指针（只读）
         * @return 缓冲区数据指针（常量）
         */
        const unsigned char* getUnifiedBufferData() const { return _buffer.data(); }
        
        /**
         * @brief 获取统一缓冲区大小
         * @return 缓冲区大小（字节）
         */
        size_t getUnifiedBufferSize() const { return _buffer.size(); }
        
        /**
         * @brief 调整统一缓冲区大小
         * @param newSize 新的缓冲区大小（字节）
         */
        void resizeUnifiedBuffer(size_t newSize) { 
            _buffer.resize(newSize);
        }

        // 🔧 直接数据缓冲区访问（消除冗余）
        /**
         * @brief 直接获取数据缓冲区（替代原data成员）
         * @return 数据缓冲区引用
         */
        std::vector<unsigned char>& getData() { return _buffer; }
        const std::vector<unsigned char>& getData() const { return _buffer; }

        // (可选) 坐标访问器
        const std::vector<double> &getLonValues() const { // 经度 (X) 坐标值
            return _definition.xDimension.coordinates;
        }
        const std::vector<double> &getLatValues() const { // 纬度 (Y) 坐标值
            return _definition.yDimension.coordinates;
        }

        // --- 🔧 统一的公共成员（消除重复） ---
        DataType dataType = DataType::Unknown;       // 数据类型
        std::map<std::string, std::string> metadata; // 元数据键值对
        CRSInfo crs;                                 // 坐标参考系统
        GridDefinition definition;                   // 网格定义

        // Example of how getNumBands() could be implemented
        inline size_t getNumBands() const
        {
            // 首先检查 zDimension 中的坐标数量
            if (_definition.zDimension.getNumberOfLevels() > 0)
            {
                return _definition.zDimension.getNumberOfLevels();
            }
            // 如果 zDimension 未定义，则返回原始波段计数
            return _bandCount;
        }

        /**
         * @brief 获取Z维度类型
         * @return Z维度的坐标类型
         */
        inline CoordinateDimension getZDimensionType() const
        {
            return _definition.zDimension.type;
        }

        /**
         * @brief 获取Z维度名称
         * @return Z维度的名称
         */
        inline const std::string &getZDimensionName() const
        {
            return _definition.zDimension.name;
        }

        /**
         * @brief 获取Z维度单位
         * @return Z维度的单位
         */
        inline const std::string &getZDimensionUnits() const
        {
            return _definition.zDimension.units;
        }

        // Creates a new GridData object representing a horizontal slice of the original.
        // The new object copies the relevant data and metadata.
        std::shared_ptr<GridData> createSlice(size_t startRow, size_t numRows) const;

        // ===== 布局相关方法 =====
        MemoryLayout getMemoryLayout() const { return _memoryLayout; }
        
        void setMemoryLayout(MemoryLayout layout) { 
            _memoryLayout = layout; 
        }
        
        // ===== 访问模式提示 =====
        AccessPattern getPreferredAccessPattern() const { 
            return _preferredAccess; 
        }
        
        void setPreferredAccessPattern(AccessPattern pattern) { 
            _preferredAccess = pattern; 
        }
        
        // ===== 新增3：快速索引计算（内联优化） =====
        /**
         * @brief 获取线性索引（考虑内存布局）
         * @note 这是最关键的性能优化点
         */
        inline size_t getLinearIndex(size_t x, size_t y, size_t z = 0, size_t t = 0) const {
            if (_memoryLayout == MemoryLayout::ROW_MAJOR) {
                // 标准C布局：t * (Z * Y * X) + z * (Y * X) + y * X + x
                size_t xSize = _definition.cols;
                size_t ySize = _definition.rows;
                size_t zSize = _definition.zDimension.getNumberOfLevels();
                if (zSize == 0) zSize = 1;
                size_t tSize = _definition.tDimension.getNumberOfLevels();
                if (tSize == 0) tSize = 1;
                
                return t * (zSize * ySize * xSize) +
                       z * (ySize * xSize) +
                       y * xSize + x;
            } else {
                // 列主序：x * (T * Z * Y) + y * (T * Z) + z * T + t
                size_t xSize = _definition.cols;
                size_t ySize = _definition.rows;
                size_t zSize = _definition.zDimension.getNumberOfLevels();
                if (zSize == 0) zSize = 1;
                size_t tSize = _definition.tDimension.getNumberOfLevels();
                if (tSize == 0) tSize = 1;
                
                return x * (tSize * zSize * ySize) +
                       y * (tSize * zSize) +
                       z * tSize + t;
            }
        }

        // ===== 新增4：优化提示（延迟初始化） =====
        /**
         * @brief 优化提示结构（延迟计算）
         */
        struct OptimizationHints {
            bool isContiguousX = true;      // X方向是否连续
            bool isContiguousY = true;      // Y方向是否连续
            size_t cacheLineSize = 64;      // 缓存行大小
            size_t optimalBlockSizeX = 32;  // 最优块大小
            size_t optimalBlockSizeY = 32;
            bool hasUniformSpacing = true;  // 是否均匀间隔
            double avgSpacingX = 1.0;       // 平均间隔
            double avgSpacingY = 1.0;
            double avgSpacingZ = 1.0;
            bool hasPrecomputedDerivatives = false;  // 是否有预计算的导数
        };
        
        /**
         * @brief 获取优化提示（首次调用时创建）
         */
        const OptimizationHints& getOptimizationHints() const {
            if (!_optimizationHints) {
                _optimizationHints = std::make_unique<GridData::OptimizationHints>();
                computeOptimizationHints(*_optimizationHints);
            }
            return *_optimizationHints;
        }
        
        // ===== 新增5：批量访问支持（零拷贝） =====
        /**
         * @brief 获取数据切片视图（不复制数据）
         */
        struct DataSlice {
            const unsigned char* data;
            size_t offset;
            size_t stride[4];
            size_t count[4];
            DataType dataType;
            
            template<typename T>
            inline T getValue(size_t idx) const {
                return *reinterpret_cast<const T*>(data + offset + idx * sizeof(T));
            }
        };
        
        DataSlice getSlice(size_t xStart, size_t xCount,
                          size_t yStart = 0, size_t yCount = 1,
                          size_t zStart = 0, size_t zCount = 1) const {
            DataSlice slice;
            slice.data = _buffer.data();
            slice.dataType = _dataType;
            slice.offset = getLinearIndex(xStart, yStart, zStart) * getElementSizeBytes();
            
            // 计算步长
            if (_memoryLayout == MemoryLayout::ROW_MAJOR) {
                slice.stride[0] = getElementSizeBytes();
                slice.stride[1] = slice.stride[0] * _definition.cols;
                slice.stride[2] = slice.stride[1] * _definition.rows;
                slice.stride[3] = slice.stride[2] * std::max(size_t(1), _definition.zDimension.getNumberOfLevels());
            } else {
                // 列主序步长计算
                size_t ySize = _definition.rows;
                size_t zSize = std::max(size_t(1), _definition.zDimension.getNumberOfLevels());
                size_t tSize = std::max(size_t(1), _definition.tDimension.getNumberOfLevels());
                
                slice.stride[0] = getElementSizeBytes() * ySize * zSize * tSize;
                slice.stride[1] = getElementSizeBytes() * zSize * tSize;
                slice.stride[2] = getElementSizeBytes() * tSize;
                slice.stride[3] = getElementSizeBytes();
            }
            
            slice.count[0] = xCount;
            slice.count[1] = yCount;
            slice.count[2] = zCount;
            slice.count[3] = 1;
            
            return slice;
        }

        // ===== 新增6：插值优化辅助方法 =====
        /**
         * @brief 获取用于插值的优化数据视图
         * @return 插值优化的数据视图，包含布局信息和快速访问方法
         */
        struct InterpolationView {
            const unsigned char* data;
            size_t rows;
            size_t cols; 
            size_t bands;
            size_t elementSize;
            DataType dataType;
            MemoryLayout layout;
            
            // 维度步长（用于快速索引计算）
            size_t rowStride;    // 行间字节步长
            size_t colStride;    // 列间字节步长
            size_t bandStride;   // 波段间字节步长
            
            // 坐标信息（如果可用）
            const std::vector<double>* xCoords = nullptr;
            const std::vector<double>* yCoords = nullptr;
            const std::vector<double>* zCoords = nullptr;
            
            // 快速值访问（考虑内存布局）
            template<typename T>
            inline T getValue(size_t row, size_t col, size_t band = 0) const {
                size_t offset;
                if (layout == MemoryLayout::ROW_MAJOR) {
                    offset = band * bandStride + row * rowStride + col * colStride;
                } else {
                    // 列主序：col变化最快
                    offset = band * bandStride + col * colStride + row * rowStride;
                }
                return *reinterpret_cast<const T*>(data + offset);
            }
            
            // 获取2x2邻域（用于双线性插值）
            template<typename T>
            inline void getNeighborhood2x2(size_t row, size_t col, size_t band, T neighbors[4]) const {
                neighbors[0] = getValue<T>(row, col, band);
                neighbors[1] = getValue<T>(row, col + 1, band);
                neighbors[2] = getValue<T>(row + 1, col, band);
                neighbors[3] = getValue<T>(row + 1, col + 1, band);
            }
            
            // 获取4x4邻域（用于双三次/PCHIP插值）
            template<typename T>
            inline void getNeighborhood4x4(size_t row, size_t col, size_t band, T neighbors[16]) const {
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        neighbors[i * 4 + j] = getValue<T>(row + i - 1, col + j - 1, band);
                    }
                }
            }
        };
        
        /**
         * @brief 创建插值优化视图
         */
        InterpolationView createInterpolationView() const {
            InterpolationView view;
            view.data = _buffer.data();
            view.rows = _definition.rows;
            view.cols = _definition.cols;
            view.bands = _bandCount;
            view.elementSize = getElementSizeBytes();
            view.dataType = _dataType;
            view.layout = _memoryLayout;
            
            // 计算步长
            if (_memoryLayout == MemoryLayout::ROW_MAJOR) {
                view.colStride = view.elementSize;
                view.rowStride = view.colStride * view.cols;
                view.bandStride = view.rowStride * view.rows;
            } else {
                // 列主序
                view.rowStride = view.elementSize;
                view.colStride = view.rowStride * view.rows;
                view.bandStride = view.colStride * view.cols;
            }
            
            // 设置坐标指针（如果可用）
            if (_definition.hasXDimension()) {
                view.xCoords = &_definition.xDimension.coordinates;
            }
            if (_definition.hasYDimension()) {
                view.yCoords = &_definition.yDimension.coordinates;
            }
            if (_definition.hasZDimension()) {
                view.zCoords = &_definition.zDimension.coordinates;
            }
            
            return view;
        }
        
        /**
         * @brief 准备用于GPU的数据（确保行主序和内存对齐）
         */
        std::shared_ptr<GridData> prepareForGPU() const {
            // 如果已经是行主序且对齐，直接返回共享指针
            if (_memoryLayout == MemoryLayout::ROW_MAJOR && isMemoryAligned()) {
                // 注意：这要求GridData对象本身是通过shared_ptr管理的
                // 如果不是，则需要外部代码处理
                try {
                    return std::const_pointer_cast<GridData>(shared_from_this());
                } catch (const std::bad_weak_ptr&) {
                    // 如果对象不是通过shared_ptr管理的，返回nullptr
                    // 调用者需要自行处理数据准备
                    return nullptr;
                }
            }
            
            // 如果需要转换，返回nullptr，让调用者处理
            // 因为GridData不支持拷贝，无法在这里创建副本
            return nullptr;
        }
        
        /**
         * @brief 预计算插值导数（用于PCHIP等高阶插值）
         */
        void precomputeInterpolationDerivatives() const {
            if (!_optimizationHints) {
                _optimizationHints = std::make_unique<OptimizationHints>();
            }
            
            // 这里可以预计算PCHIP所需的导数
            // 实际实现将在插值服务中完成
            _optimizationHints->hasPrecomputedDerivatives = true;
        }
        
        /**
         * @brief 检查是否有预计算的插值数据
         */
        bool hasPrecomputedInterpolationData() const {
            return _optimizationHints && 
                   _optimizationHints->hasPrecomputedDerivatives;
        }

    private:
        // --- Helper methods ---
        template <typename T>
        T _getValueInternal(size_t index) const;
        template <typename T>
        void _setValueInternal(size_t index, T value);

        inline size_t calculateOffset(size_t row, size_t col, size_t band) const
        {
            // 检查索引是否越界
            if (row >= _definition.rows || col >= _definition.cols)
            {
                throw std::out_of_range("GridData row/col indices out of range. Requested row: " + std::to_string(row) +
                                        " (max: " + std::to_string(_definition.rows - 1) + "), col: " + std::to_string(col) +
                                        " (max: " + std::to_string(_definition.cols - 1) + ")");
            }

            // 验证波段索引
            size_t numBands = getNumBands();
            if (band >= numBands)
            {
                throw std::out_of_range("GridData band index (" + std::to_string(band) +
                                        ") out of range. Max band index: " + std::to_string(numBands - 1));
            }

            // 使用BSQ (Band Sequential) 格式计算偏移量:
            // element_offset = (band_index * rows * cols) + (row_index * cols) + col_index
            size_t elements_per_band = _definition.rows * _definition.cols;
            size_t element_offset = (band * elements_per_band) + (row * _definition.cols) + col;

            // 乘以每个元素的字节大小获取最终字节偏移量
            size_t element_size_bytes = getElementSizeBytes();
            if (element_size_bytes == 0)
            {
                throw std::runtime_error("Element size is 0 for data type " + std::to_string(static_cast<int>(_dataType)));
            }

            return element_offset * element_size_bytes;
        }

        inline size_t getElementSize() const
        {
            switch (_dataType)
            {
            case DataType::Byte:
                return sizeof(uint8_t);
            case DataType::UInt16:
                return sizeof(uint16_t);
            case DataType::Int16:
                return sizeof(int16_t);
            case DataType::UInt32:
                return sizeof(uint32_t);
            case DataType::Int32:
                return sizeof(int32_t);
            case DataType::UInt64:
                return sizeof(uint64_t);
            case DataType::Int64:
                return sizeof(int64_t);
            case DataType::Float32:
                return sizeof(float);
            case DataType::Float64:
                return sizeof(double);
            case DataType::String:
                return 0; // Or some other convention if strings are in buffer
            case DataType::Boolean:
                return sizeof(bool); // Or typically sizeof(char)
            case DataType::Binary:
                return 0; // Depends on how binary data is handled
            case DataType::Complex16:
                return sizeof(int16_t) * 2;
            case DataType::Complex32:
                return sizeof(float) * 2;
            case DataType::Complex64:
                return sizeof(double) * 2;
            case DataType::Unknown:
                return 0;
            default:
                return 0;
            }
        }

        // --- 成员变量 ---
        GridDefinition _definition;
        DataType _dataType = DataType::Unknown;
        
        // ===== 新增的轻量级成员 =====
        MemoryLayout _memoryLayout = MemoryLayout::ROW_MAJOR;
        AccessPattern _preferredAccess = AccessPattern::UNKNOWN;
        
        // 延迟创建的优化数据（使用unique_ptr避免默认分配）
        mutable std::unique_ptr<GridData::OptimizationHints> _optimizationHints;
        
        // 🚀 实用的内存对齐优化：使用alignas确保SIMD友好的32字节对齐
        #ifdef OSCEAN_HAS_HIGH_PERFORMANCE_INTERFACES
        #pragma warning(push)
        #pragma warning(disable: 4324) // 抑制结构填充警告
        alignas(32) std::vector<unsigned char> _buffer; ///< 32字节对齐的数据缓冲区（SIMD友好）
        #pragma warning(pop)
        #else
        std::vector<unsigned char> _buffer; ///< 标准分配器
        #endif
        
        std::vector<double> _geoTransform;  // GDAL-style GeoTransform
        CRSInfo _crs;
        std::string _variableName;
        std::string _units;
        boost::optional<double> _fillValue = boost::none;
        size_t _bandCount = 1; // Number of bands, default to 1

        // (可选) 存储计算出的坐标
        std::vector<double> _lonCoordinates;
        std::vector<double> _latCoordinates;

        friend class DataChunkCache;
        
    public:
        /**
         * @brief 检查数据缓冲区是否内存对齐
         * @return 如果缓冲区按SIMD要求对齐则返回true
         */
        inline bool isMemoryAligned() const noexcept {
            #ifdef OSCEAN_HAS_HIGH_PERFORMANCE_INTERFACES
            // 检查是否按32字节对齐（AVX要求）
            return (reinterpret_cast<std::uintptr_t>(_buffer.data()) % 32) == 0;
            #else
            // 标准模式：检查是否自然对齐到较小边界
            return (reinterpret_cast<std::uintptr_t>(_buffer.data()) % alignof(std::max_align_t)) == 0;
            #endif
        }
        
        /**
         * @brief 获取内存对齐信息
         * @return 对齐字节数
         */
        inline size_t getMemoryAlignment() const noexcept {
            #ifdef OSCEAN_HAS_HIGH_PERFORMANCE_INTERFACES
            return 32; // AVX对齐
            #else
            return alignof(std::max_align_t); // 标准对齐
            #endif
        }
        
        /**
         * @brief 获取内存优化状态描述
         * @return 内存优化状态的字符串描述
         */
        inline std::string getMemoryOptimizationStatus() const {
            #ifdef OSCEAN_HAS_HIGH_PERFORMANCE_INTERFACES
            return isMemoryAligned() ? 
                "高性能对齐模式 (32字节对齐)" : 
                "高性能模式 (意外未对齐)";
            #else
            return isMemoryAligned() ? 
                "标准模式 (自然对齐)" : 
                "标准模式 (未对齐)";
            #endif
        }
        
        /**
         * @brief 获取缓冲区对齐状态的技术细节
         * @return 技术细节字符串
         */
        inline std::string getAlignmentDetails() const {
            std::stringstream ss;
            ss << "缓冲区地址: 0x" << std::hex << reinterpret_cast<std::uintptr_t>(_buffer.data());
            ss << ", 要求对齐: " << std::dec << getMemoryAlignment() << "字节";
            ss << ", 实际对齐: " << (isMemoryAligned() ? "是" : "否");
            return ss.str();
        }
        
        /**
         * @brief 强制重新分配对齐内存（仅高性能模式）
         * @details 在高性能模式下，如果缓冲区未对齐则重新分配；标准模式下为无操作
         */
        inline void reallocateAligned() {
            #ifdef OSCEAN_HAS_HIGH_PERFORMANCE_INTERFACES
            if (!_buffer.empty() && !isMemoryAligned()) {
                // 保存当前数据
                auto currentData = _buffer;
                // 重新分配对齐内存（SIMD对齐分配器自动处理）
                _buffer.clear();
                _buffer.resize(currentData.size());
                // 复制数据到内部缓冲区
                std::copy(currentData.begin(), currentData.end(), _buffer.begin());
            }
            #endif
            // 标准模式下不执行任何操作，避免不必要的重新分配
        }

        /**
         * @brief 简化的内存对齐检查（仅高性能模式有意义）
         * @details 在高性能模式下提供对齐状态检查；标准模式下为无操作
         */
        inline void checkAlignmentOptimal() const {
            #ifdef OSCEAN_HAS_HIGH_PERFORMANCE_INTERFACES
            if (!isMemoryAligned()) {
                // 在高性能模式下，记录对齐状态但不强制重新分配
                // 因为重新分配可能很昂贵，而且不保证对齐
                // 建议在数据初始化时确保对齐
            }
            #endif
            // 标准模式下不执行任何操作
        }

        /**
         * @brief 计算优化提示（只在需要时调用）
         */
        void computeOptimizationHints(OptimizationHints& hints) const {
            // 检查维度连续性
            if (_definition.xDimension.getNumberOfLevels() > 1) {
                hints.isContiguousX = isUniformSpacing(_definition.xDimension.coordinates);
                hints.avgSpacingX = computeAverageSpacing(_definition.xDimension.coordinates);
            }
            
            if (_definition.yDimension.getNumberOfLevels() > 1) {
                hints.isContiguousY = isUniformSpacing(_definition.yDimension.coordinates);
                hints.avgSpacingY = computeAverageSpacing(_definition.yDimension.coordinates);
            }
            
            // 计算最优块大小（考虑缓存）
            size_t elementSize = getElementSizeBytes();
            size_t elementsPerCacheLine = hints.cacheLineSize / elementSize;
            
            hints.optimalBlockSizeX = std::min(size_t(32), 
                std::max(elementsPerCacheLine, size_t(16)));
            hints.optimalBlockSizeY = std::min(size_t(32), 
                std::max(size_t(4), hints.cacheLineSize / (hints.optimalBlockSizeX * elementSize)));
        }
        
        /**
         * @brief 检查坐标是否均匀间隔
         */
        bool isUniformSpacing(const std::vector<double>& coords) const {
            if (coords.size() < 2) return true;
            
            double firstSpacing = coords[1] - coords[0];
            const double tolerance = 1e-6;
            
            for (size_t i = 2; i < coords.size(); ++i) {
                double spacing = coords[i] - coords[i-1];
                if (std::abs(spacing - firstSpacing) > tolerance) {
                    return false;
                }
            }
            return true;
        }
        
        /**
         * @brief 计算平均间隔
         */
        double computeAverageSpacing(const std::vector<double>& coords) const {
            if (coords.size() < 2) return 1.0;
            return (coords.back() - coords.front()) / (coords.size() - 1);
        }

        // 内部辅助方法：转换为行主序
        void convertToRowMajor() {
            if (_memoryLayout == MemoryLayout::ROW_MAJOR) return;
            
            // 创建新缓冲区
            std::vector<unsigned char> newBuffer(_buffer.size());
            
            // 执行转置
            size_t elementSize = getElementSizeBytes();
            for (size_t band = 0; band < _bandCount; ++band) {
                for (size_t row = 0; row < _definition.rows; ++row) {
                    for (size_t col = 0; col < _definition.cols; ++col) {
                        // 计算源和目标偏移
                        size_t srcOffset = calculateOffsetColumnMajor(row, col, band);
                        size_t dstOffset = calculateOffset(row, col, band);
                        
                        // 复制元素
                        std::memcpy(newBuffer.data() + dstOffset,
                                   _buffer.data() + srcOffset,
                                   elementSize);
                    }
                }
            }
            
            // 替换缓冲区
            _buffer = std::move(newBuffer);
            _memoryLayout = MemoryLayout::ROW_MAJOR;
        }
        
        // 列主序偏移计算
        inline size_t calculateOffsetColumnMajor(size_t row, size_t col, size_t band) const {
            size_t element_offset = band * (_definition.rows * _definition.cols) +
                                   col * _definition.rows + row;
            return element_offset * getElementSizeBytes();
        }

    };

    /**
     * @brief Template helper to get value from GridData safely
     */
    template <typename T>
    T GridData::getValue(size_t row, size_t col, size_t band) const
    {
        // ADD DEBUG LOGGING (temporarily, assuming iostream is available)
        // std::cout << "[GridData::getValue DEBUG] Request: (" << row << "," << col << "," << band << ")" << std::endl;
        // std::cout << "[GridData::getValue DEBUG] _definition.rows: " << _definition.rows << ", _definition.cols: " << _definition.cols << ", _bandCount: " << _bandCount << std::endl;
        // std::cout << "[GridData::getValue DEBUG] _dataType: " << static_cast<int>(_dataType) << ", elementSize: " << getElementSize() << std::endl;

        if (row >= _definition.rows || col >= _definition.cols || band >= _bandCount)
        { // Added band >= _bandCount check
            // std::cout << "[GridData::getValue ERROR] Indices out of range." << std::endl;
            throw std::out_of_range("GridData indices out of range. Requested (" +
                                    std::to_string(row) + "," + std::to_string(col) + "," + std::to_string(band) +
                                    ") vs Max (" + std::to_string(_definition.rows - 1) + "," +
                                    std::to_string(_definition.cols - 1) + "," + std::to_string(_bandCount - 1) + ")");
        }

        size_t offset = calculateOffset(row, col, band);
        // std::cout << "[GridData::getValue DEBUG] Calculated offset: " << offset << ", sizeof(T): " << sizeof(T) << ", _buffer.size(): " << _buffer.size() << std::endl;

        if (offset + sizeof(T) > _buffer.size())
        {
            // std::cout << "[GridData::getValue ERROR] Calculated offset out of buffer bounds." << std::endl;
            throw std::out_of_range("Calculated offset out of buffer bounds. Offset: " + std::to_string(offset) +
                                    ", sizeof(T): " + std::to_string(sizeof(T)) + ", BufferSize: " + std::to_string(_buffer.size()));
        }

        const T *value_ptr = reinterpret_cast<const T *>(_buffer.data() + offset);
        // T actual_value = *value_ptr;
        // std::cout << "[GridData::getValue DEBUG] Returning value: " << actual_value << std::endl;

        // TODO: Add check for actual DataType vs requested T
        // if (getDataType() != /* Map T to DataType enum */) {
        //     throw std::runtime_error("Type mismatch when reading GridData value");
        // }

        return *value_ptr;
    }

    /**
     * @brief Template implementation to set value in GridData safely
     * 🔧 优化：消除双重写入，统一使用内部缓冲区
     */
    template <typename T>
    void GridData::setValue(size_t row, size_t col, size_t band, T value)
    {
        if (row >= _definition.rows || col >= _definition.cols /* || band >= numBands */)
        {
            throw std::out_of_range("GridData indices out of range");
        }

        size_t offset = calculateOffset(row, col, band);
        if (offset + sizeof(T) > _buffer.size())
        {
            throw std::out_of_range("Calculated offset out of buffer bounds");
        }

        // 🔧 统一写入：仅操作内部缓冲区
        *reinterpret_cast<T *>(_buffer.data() + offset) = value;
    }

    // 显式实例化常用类型以确保链接时可用
    template void GridData::setValue<float>(size_t row, size_t col, size_t band, float value);
    template void GridData::setValue<double>(size_t row, size_t col, size_t band, double value);
    template void GridData::setValue<int>(size_t row, size_t col, size_t band, int value);
    template void GridData::setValue<short>(size_t row, size_t col, size_t band, short value);
    template void GridData::setValue<unsigned char>(size_t row, size_t col, size_t band, unsigned char value);

    /**
     * @brief 获取内部缓冲区大小（字节）
     */
    inline size_t GridData::getSizeInBytes() const
    {
        // The most straightforward implementation is returning the size of the internal buffer.
        // Assumes the buffer accurately represents the stored data.
        return _buffer.size();
        // Alternative: Calculate based on definition and data type
        // return _definition.rows * _definition.cols * /*bands?*/ * getElementSize();
    }

     /**
     * @brief 矢量要素属性值
     */
    using VectorFeatureAttribute = AttributeValue; // NEW DEFINITION: Alias to the main AttributeValue

    /**
     * @brief 矢量要素结构
     */
    struct VectorFeature
    {
        std::string id;                                           ///< [Chinese comment removed for encoding compatibility]
        int geometryType = 0;                                  ///< [Chinese comment removed for encoding compatibility] (simplified from GeometryType)
        std::string geometryWKT;                                  ///< [Chinese comment removed for encoding compatibility]
        std::map<std::string, VectorFeatureAttribute> attributes; ///< [Chinese comment removed for encoding compatibility]
        CRSInfo crs;                                              ///< [Chinese comment removed for encoding compatibility]

        /**
         * @brief 获取指定名称的属性值
         * @param name 属性名称
         * @return 属性值
         * @throws std::out_of_range 如果属性不存在
         */
        const VectorFeatureAttribute &getAttribute(const std::string &name) const
        {
            return attributes.at(name);
        }

        /**
         * @brief 设置属性值
         * @param name 属性名称
         * @param value 属性值
         */
        void setAttribute(const std::string &name, const VectorFeatureAttribute &value)
        {
            attributes[name] = value;
        }

        /**
         * @brief 设置所有属性
         * @param newAttributes 属性映射
         */
        void setAttributes(const std::map<std::string, VectorFeatureAttribute> &newAttributes)
        {
            this->attributes = newAttributes;
        }
    };

    /**
     * @brief 查询标准结构体
     */
    struct QueryCriteria
    {
        boost::optional<BoundingBox> spatialExtent;                 ///< [Chinese comment removed for encoding compatibility]
        boost::optional<std::pair<Timestamp, Timestamp>> timeRange; ///< [Chinese comment removed for encoding compatibility]
        std::vector<std::string> variables;                       ///< [Chinese comment removed for encoding compatibility]
        std::string textFilter;                                   ///< [Chinese comment removed for encoding compatibility]
        std::string formatFilter;                                 ///< [Chinese comment removed for encoding compatibility]
        std::map<std::string, std::string> metadataFilters;       ///< [Chinese comment removed for encoding compatibility]

        // *** ADDED: operator== for comparison in tests ***
        bool operator==(const QueryCriteria &other) const
        {
            // Compare optionals: check both present/absent, then compare values if present
            if (spatialExtent.has_value() != other.spatialExtent.has_value())
                return false;
            if (spatialExtent.has_value() && !(spatialExtent.value() == other.spatialExtent.value()))
                return false;

            if (timeRange.has_value() != other.timeRange.has_value())
                return false;
            if (timeRange.has_value() && !(timeRange.value() == other.timeRange.value()))
                return false;

            // Compare vectors, maps, and strings directly
            return variables == other.variables &&
                   textFilter == other.textFilter &&
                   formatFilter == other.formatFilter &&
                   metadataFilters == other.metadataFilters;
        }
    };

    // --- 🔧 前向声明（C++17版本）---
    // 为了在VariableMeta中使用，需要提前声明这些函数
    inline std::string dataTypeToString(DataType dataType);
    inline DataType stringToDataType(const std::string& typeStr);

    // --- Common Querying & Modeling Types ---

    // Moved TimeRange definition earlier
    struct TimeRange
    {
        std::chrono::system_clock::time_point startTime;
        std::chrono::system_clock::time_point endTime;  ///< 添加结束时间成员
        
        TimeRange() = default;
        TimeRange(std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end)
            : startTime(start), endTime(end) {}
            
        /**
         * @brief 检查时间范围是否有效
         */
        bool isValid() const {
            return startTime <= endTime;
        }
        
        /**
         * @brief 获取时间范围的持续时间
         */
        std::chrono::duration<double> getDuration() const {
            return endTime - startTime;
        }
        
        /**
         * @brief 🔧 新增：等值运算符（C++17版本）
         */
        bool operator==(const TimeRange& other) const {
            return startTime == other.startTime && endTime == other.endTime;
        }
    };

    /**
     * @brief 🔧 新增：统一查询条件结构（C++17版本）
     * 🎯 用于替代所有分散的查询条件定义，统一查询接口
     */
    struct UnifiedQueryCriteria {
        boost::optional<BoundingBox> spatialBounds;      ///< 空间范围限制（C++17: boost::optional）
        boost::optional<TimeRange> timeRange;           ///< 时间范围限制（C++17: boost::optional）
        std::vector<DataType> dataTypes;                ///< 数据类型过滤
        std::vector<std::string> variableNames;         ///< 变量名称过滤
        std::vector<std::string> formats;               ///< 文件格式过滤
        std::map<std::string, std::string> attributes;  ///< 属性过滤条件
        size_t maxResults = 1000;                       ///< 最大结果数量
        bool includeDetails = true;                     ///< 是否包含详细信息
        std::string sortBy = "lastModified";            ///< 排序字段
        bool ascending = false;                         ///< 排序方向（默认降序）
        
        /**
         * @brief 验证查询条件是否有效
         * @return 如果有效返回true
         */
        bool isValid() const {
            if (maxResults == 0) return false;
            if (spatialBounds && !spatialBounds->isValid()) return false;
            if (timeRange && !timeRange->isValid()) return false;
            return true;
        }
        
        /**
         * @brief 检查是否为空查询（无任何限制条件）
         * @return 如果为空查询返回true
         */
        bool isEmpty() const {
            return !spatialBounds && !timeRange && 
                   dataTypes.empty() && variableNames.empty() && 
                   formats.empty() && attributes.empty();
        }
        
        /**
         * @brief 等值运算符
         */
        bool operator==(const UnifiedQueryCriteria& other) const {
            return spatialBounds == other.spatialBounds &&
                   timeRange == other.timeRange &&
                   dataTypes == other.dataTypes &&
                   variableNames == other.variableNames &&
                   formats == other.formats &&
                   attributes == other.attributes &&
                   maxResults == other.maxResults &&
                   includeDetails == other.includeDetails &&
                   sortBy == other.sortBy &&
                   ascending == other.ascending;
        }
    };

    /**
     * @brief 变量元数据结构体
     */
    struct VariableMeta
    {
        std::string name;                                   ///< 变量名称
        DataType dataType = DataType::Unknown;              ///< 数据类型
        std::vector<std::string> dimensionNames;            ///< 维度名称列表（原始）
        std::map<std::string, std::string> attributes;      ///< 属性（原始）
        std::string units;                                  ///< 单位（从属性中提取的快捷方式）
        std::string description;                            ///< 描述（从属性中提取的快捷方式）

        // --- 新增的标准化字段 ---
        boost::optional<double> noDataValue;                ///< 无效/填充值
        boost::optional<double> scaleFactor;                ///< 缩放因子
        boost::optional<double> addOffset;                  ///< 偏移量
        boost::optional<ValueRange<double>> validRange;       ///< 有效值范围

        bool operator==(const VariableMeta& other) const {
            return name == other.name;
        }
    };

    /**
     * @brief 定义文件元数据
     * 包含文件信息、格式、范围、投影等
     */
    struct FileMetadata
    {
        /**
         * @brief 默认构造函数
         */
        FileMetadata() = default;

        // --- 🔧 第四阶段新增：metadata_service兼容字段 ---
        std::string metadataId;                      ///< metadata服务的唯一ID标识符
        std::string fileId;                          ///< [Chinese comment removed for encoding compatibility]
        std::string fileName;                        ///< [Chinese comment removed for encoding compatibility]
        std::string filePath;                        ///< [Chinese comment removed for encoding compatibility]
        std::string format;                          ///< [Chinese comment removed for encoding compatibility]
        
        // --- 原始CRS信息 ---
        boost::optional<std::string> rawCrsWkt;      ///< 从文件提取的原始WKT字符串
        boost::optional<std::string> rawCrsProj;     ///< 从文件提取的原始PROJ字符串

        // --- 🔧 metadata_service生命周期字段 ---
        int64_t extractionTimestamp = 0;            ///< 提取时间戳
        std::string lastIndexedTime;                ///< 最后索引时间（ISO格式）
        DataType dataType = DataType::Unknown;      ///< 数据类型（与metadata服务兼容）
        
        // --- 修复：恢复被错误删除的 primaryCategory 字段 ---
        DataType primaryCategory = DataType::Unknown; ///< 智能识别器确定的主要数据类型
        
        // --- 🔧 修复编译错误：新增缺失的classifications字段 ---
        std::vector<std::string> classifications;   ///< 文件分类标签（智能识别器生成）

        CRSInfo crs;                                 ///< [Chinese comment removed for encoding compatibility]
        BoundingBox spatialCoverage;                 ///< [Chinese comment removed for encoding compatibility]
        TimeRange timeRange;                         ///< [Chinese comment removed for encoding compatibility]
        std::vector<VariableMeta> variables;         ///< [Chinese comment removed for encoding compatibility]/图层信息列表
        std::map<std::string, std::string> metadata; ///< [Chinese comment removed for encoding compatibility]

        // --- 🔧 C++17兼容: 使用boost::optional替代boost::optional ---
        boost::optional<std::string> mainVariableName;       ///< 主变量名称 (boost::optional for C++17)
        std::vector<DimensionDetail> geographicDimensions;   ///< 地理维度详情
        
        // --- 🔧 新增字段：统一扩展字段（按方案要求）---
        size_t fileSizeBytes = 0;                           ///< 文件大小（字节）
        std::string lastModified;                           ///< 最后修改时间（ISO格式）
        std::string fileType;                               ///< 文件类型描述（如NetCDF, GeoTIFF等）
        DataType inferredDataType = DataType::Unknown;      ///< 推断的数据类型
        
        // --- 🔧 第二阶段新增：兼容数据库适配器字段 ---
        std::map<std::string, std::string> attributes;      ///< 文件属性集合
        boost::optional<double> dataQuality;                ///< 数据质量评分（0.0-1.0）
        boost::optional<double> completeness;               ///< 数据完整性评分（0.0-1.0）
        
        // --- 🔧 第三阶段新增：统一字段结构（修复metadata_service兼容性）---
        struct SpatialInfo {
            BoundingBox bounds;                              ///< 空间边界（使用标准BoundingBox）
            boost::optional<double> spatialResolution;      ///< 空间分辨率（C++17: boost::optional）
            std::string coordinateSystem = "WGS84";         ///< 坐标系统标识符
            std::string crsWkt;                             ///< CRS的WKT表示
            std::string proj4;                              ///< PROJ4字符串
            double resolutionX = 0.0;                       ///< X方向分辨率
            double resolutionY = 0.0;                       ///< Y方向分辨率
        } spatialInfo;
        
        struct TemporalInfo {
            // --- 🔧 metadata_service直接访问字段 ---
            std::string startTime;    ///< 开始时间（ISO格式）- metadata_service直接字段
            std::string endTime;      ///< 结束时间（ISO格式）- metadata_service直接字段
            
            struct TimeRange {
                std::string startTime;  ///< ISO格式时间字符串，如"2023-01-01T00:00:00Z"
                std::string endTime;    ///< ISO格式时间字符串，如"2023-12-31T23:59:59Z"
                std::string timeUnits = "ISO8601";  ///< 时间单位，统一使用ISO8601
            } timeRange;
            
            // 时间分辨率（秒）- 统一标准字段
            boost::optional<int> temporalResolutionSeconds;
            
            // 日历类型
            std::string calendar;
            
            // === 📅 时间范围验证方法 ===
            
            /**
             * @brief 验证时间范围是否有效
             */
            bool isValid() const {
                return !startTime.empty() && !endTime.empty();
            }
            
            /**
             * @brief 获取时间跨度（秒）
             */
            boost::optional<double> getDurationSeconds() const;
        } temporalInfo;

        // Helper methods can be added here
        // ... (removed operator[] for clarity, prefer explicit getters if needed)
    };

    /**
     * @brief 变量信息结构体
     */
    struct VariableInfo
    {
        std::string name;                              ///< [Chinese comment removed for encoding compatibility]
        DataType dataType = DataType::Unknown;         ///< [Chinese comment removed for encoding compatibility]
        std::vector<std::string> dimensions;           ///< [Chinese comment removed for encoding compatibility]
        std::vector<size_t> shape;                     ///< [Chinese comment removed for encoding compatibility]
        std::map<std::string, std::string> attributes; ///< [Chinese comment removed for encoding compatibility]
        boost::optional<double> fillValue;               ///< [Chinese comment removed for encoding compatibility]
        boost::optional<double> scaleFactor;             ///< [Chinese comment removed for encoding compatibility]
        boost::optional<double> addOffset;               ///< [Chinese comment removed for encoding compatibility]
        CRSInfo crs;                                   ///< [Chinese comment removed for encoding compatibility]

        /**
         * @brief 默认构造函数
         */
        VariableInfo() = default;
    };

    /**
     * @brief 将数据类型转换为字符串
     * @param dataType 数据类型
     * @return 数据类型字符串
     */
    inline std::string dataTypeToString(DataType dataType)
    {
        switch (dataType)
        {
        case DataType::Byte:
            return "Byte";
        case DataType::UInt16:
            return "UInt16";
        case DataType::Int16:
            return "Int16";
        case DataType::UInt32:
            return "UInt32";
        case DataType::Int32:
            return "Int32";
        case DataType::Float32:
            return "Float32";
        case DataType::Float64:
            return "Float64";
        case DataType::String:
            return "String";
        case DataType::Boolean:
            return "Boolean";
        default:
            return "Unknown";
        }
    }

    /**
     * @brief 🔧 新增：从字符串转换为数据类型枚举（C++17版本）
     * @param typeStr 数据类型字符串
     * @return 数据类型枚举
     */
    inline DataType stringToDataType(const std::string& typeStr)
    {
        if (typeStr == "Byte" || typeStr == "byte" || typeStr == "uint8") return DataType::Byte;
        if (typeStr == "UInt16" || typeStr == "uint16") return DataType::UInt16;
        if (typeStr == "Int16" || typeStr == "int16" || typeStr == "short") return DataType::Int16;
        if (typeStr == "UInt32" || typeStr == "uint32") return DataType::UInt32;
        if (typeStr == "Int32" || typeStr == "int32" || typeStr == "int") return DataType::Int32;
        if (typeStr == "Float32" || typeStr == "float32" || typeStr == "float") return DataType::Float32;
        if (typeStr == "Float64" || typeStr == "float64" || typeStr == "double") return DataType::Float64;
        if (typeStr == "String" || typeStr == "string" || typeStr == "char") return DataType::String;
        if (typeStr == "Boolean" || typeStr == "boolean" || typeStr == "bool") return DataType::Boolean;
        return DataType::Unknown;
    }

    // 🔧 操作符重载，用于支持流输出和比较（现在dataTypeToString函数已定义）
    inline std::ostream& operator<<(std::ostream& os, DataType type) {
        os << dataTypeToString(type);
        return os;
    }

    // --- Feature Data ---
    // Opaque handle example:
    struct OpaqueGeometryHandle;
    using GeometryPtr = std::shared_ptr<OpaqueGeometryHandle>; // Actual definition elsewhere

    /**
     * @struct Feature
     * @brief Represents a single geographic feature with geometry and attributes.
     */
    struct Feature
    {
        std::string id;                                   ///< Optional unique identifier for the feature
        std::string geometryWkt;                          ///< The geometry of the feature (using WKT format string)
        std::map<std::string, AttributeValue> attributes; ///< Key-value pairs of feature attributes

        /**
         * @brief Default constructor
         */
        Feature() = default;

        /**
         * @brief Parameterized constructor
         */
        Feature(std::string feature_id, std::string geom_wkt, std::map<std::string, AttributeValue> attrs)
            : id(std::move(feature_id)), geometryWkt(std::move(geom_wkt)), attributes(std::move(attrs)) {}

        // Parameterized constructor without ID
        Feature(std::string geom_wkt, std::map<std::string, AttributeValue> attrs)
            : geometryWkt(std::move(geom_wkt)), attributes(std::move(attrs)) {}

        // Add comparison operator
        bool operator==(const Feature &other) const
        {
            return id == other.id && geometryWkt == other.geometryWkt && attributes == other.attributes;
        }

        // Getter for geometry
        const std::string &getGeometry() const
        {
            return geometryWkt;
        }

        /**
         * @brief 从WKT几何体中提取边界框
         * @return 要素的边界框
         */
        BoundingBox getBoundingBox() const;
    };

    struct FileInfo
    {
        std::string id;   ///< [Chinese comment removed for encoding compatibility]
        std::string path; ///< [Chinese comment removed for encoding compatibility]
        // Add other potential summary fields if needed later, e.g.:
        // uint64_t sizeBytes = 0;
        // Timestamp lastModified = 0;

        // 添加相等运算符
        bool operator==(const FileInfo &other) const
        {
            return id == other.id && path == other.path;
        }
    };

    /**
     * @brief 索引进度状态枚举
     */
    enum class IndexingProgress
    {
        IDLE,         ///< [Chinese comment removed for encoding compatibility]
        INDEXING,     ///< [Chinese comment removed for encoding compatibility]
        REBUILDING,   ///< [Chinese comment removed for encoding compatibility]
        ERROR,        ///< [Chinese comment removed for encoding compatibility]
        MAINTENANCE   ///< [Chinese comment removed for encoding compatibility]
    };

    /**
     * @brief 索引状态信息
     */
    struct IndexingStatus
    {
        IndexingProgress status = IndexingProgress::IDLE; ///< [Chinese comment removed for encoding compatibility]
        size_t totalFiles = 0;                            ///< [Chinese comment removed for encoding compatibility]
        size_t processedFiles = 0;                        ///< [Chinese comment removed for encoding compatibility]
        std::string currentFile = "";                     ///< [Chinese comment removed for encoding compatibility]
        std::string errorMessage = "";                    ///< [Chinese comment removed for encoding compatibility]
        std::chrono::system_clock::time_point startTime;  // Added startTime
        std::chrono::system_clock::time_point endTime;    // Added endTime

        // 可以添加其他字段，如开始时间、结束时间等
    };

    /**
     * @brief 索引统计信息
     */
    struct IndexStatistics
    {
        size_t totalFiles = 0;                               ///< 索引中的文件总数
        size_t totalMetadataEntries = 0;                     ///< 元数据条目总数
        size_t indexSize = 0;                                ///< 索引大小 (字节)
        std::chrono::system_clock::time_point lastUpdated;   ///< 最后更新时间
        double averageQueryTime = 0.0;                       ///< 平均查询时间 (毫秒)
        size_t queryCount = 0;                               ///< 查询次数
        double indexUtilization = 0.0;                       ///< 索引利用率 (0.0-1.0)
        double fragmentationRatio = 0.0;                     ///< 碎片率
        size_t cacheHits = 0;                                ///< 缓存命中次数
        size_t cacheMisses = 0;                              ///< 缓存未命中次数
        
        /**
         * @brief 计算缓存命中率
         */
        double getCacheHitRatio() const {
            if (cacheHits + cacheMisses == 0) return 0.0;
            return static_cast<double>(cacheHits) / (cacheHits + cacheMisses);
        }
        
        /**
         * @brief 等值运算符
         */
        bool operator==(const IndexStatistics& other) const {
            return totalFiles == other.totalFiles &&
                   totalMetadataEntries == other.totalMetadataEntries &&
                   indexSize == other.indexSize &&
                   lastUpdated == other.lastUpdated &&
                   averageQueryTime == other.averageQueryTime &&
                   queryCount == other.queryCount &&
                   indexUtilization == other.indexUtilization;
        }
    };



    /**
     * @brief 转换状态枚举
     */
    enum class TransformStatus {
        SUCCESS,        ///< 转换成功
        FAILED,         ///< 转换失败
        NOT_APPLICABLE, ///< 转换不适用
        PARTIAL_SUCCESS ///< 部分成功
    };

    /**
     * @brief 已转换的点
     */
    struct TransformedPoint
    {
        double x = 0.0;                                    ///< X坐标
        double y = 0.0;                                    ///< Y坐标
        boost::optional<double> z = boost::none;            ///< Z坐标 (可选)
        TransformStatus status = TransformStatus::FAILED;  ///< 转换状态
        boost::optional<std::string> errorMessage = boost::none; ///< 错误信息 (可选)
        
        /**
         * @brief 构造函数
         */
        TransformedPoint() = default;
        
        /**
         * @brief 参数构造函数
         */
        TransformedPoint(double x_val, double y_val, 
                        boost::optional<double> z_val = boost::none,
                        TransformStatus transform_status = TransformStatus::SUCCESS)
            : x(x_val), y(y_val), z(z_val), status(transform_status) {}
        
        /**
         * @brief 检查转换是否成功
         */
        bool isValid() const {
            return status == TransformStatus::SUCCESS || status == TransformStatus::PARTIAL_SUCCESS;
        }
        
        /**
         * @brief 等值运算符
         */
        bool operator==(const TransformedPoint& other) const {
            return x == other.x && y == other.y && z == other.z && 
                   status == other.status && errorMessage == other.errorMessage;
        }
    };

    /**
     * @brief 坐标转换结果
     */
    struct CoordinateTransformationResult
    {
        std::vector<TransformedPoint> transformedPoints;    ///< 转换后的点集合
        size_t successCount = 0;                            ///< 成功转换的点数
        size_t failureCount = 0;                            ///< 转换失败的点数
        double averageTransformTime = 0.0;                  ///< 平均转换时间 (毫秒)
        std::string sourceCRS;                              ///< 源坐标系统
        std::string targetCRS;                              ///< 目标坐标系统
        std::vector<std::string> errors;                    ///< 错误信息列表
        std::chrono::milliseconds totalTime{0};             ///< 总转换时间
        double totalDistance = 0.0;                         ///< 总变换距离
        
        /**
         * @brief 计算成功率
         */
        double getSuccessRate() const {
            if (successCount + failureCount == 0) return 0.0;
            return static_cast<double>(successCount) / (successCount + failureCount);
        }
        
        /**
         * @brief 检查是否有错误
         */
        bool hasErrors() const {
            return failureCount > 0 || !errors.empty();
        }
        
        /**
         * @brief 等值运算符
         */
        bool operator==(const CoordinateTransformationResult& other) const {
            return transformedPoints == other.transformedPoints &&
                   successCount == other.successCount &&
                   failureCount == other.failureCount &&
                   sourceCRS == other.sourceCRS &&
                   targetCRS == other.targetCRS &&
                   errors == other.errors;
        }
    };

    // Using std::map<std::string, std::any> for flexible Model I/O
    // Requires careful type checking on usage.
    // Alternatively, define specific input/output structs per model type.
    using ModelInput = std::map<std::string, boost::any>;
    using ModelOutput = std::map<std::string, boost::any>;

    /**
     * @brief Represents time series data at a single point or grid cell.
     */
    struct TimeSeriesData
    {
        std::string variableName;
        std::string units;
        std::vector<Timestamp> timePoints;
        std::vector<double> values;                     // Assumes double for simplicity
        boost::optional<double> fillValue = boost::none; // Optional fill value indicator

        bool empty() const { return timePoints.empty(); }
        size_t size() const { return timePoints.size(); }
    };

    /**
     * @brief Represents vertical profile data at a single point/cell and time.
     */
    struct VerticalProfileData
    {
        std::string variableName;
        std::string units;
        std::vector<double> verticalLevels; // Depth, Height, Pressure Level, etc.
        std::string verticalUnits;
        std::vector<double> values;
        boost::optional<double> fillValue = boost::none;

        bool empty() const { return verticalLevels.empty(); }
        size_t size() const { return verticalLevels.size(); }
    };

    /**
     * @brief Opaque handle or wrapper around OGRGeometry*, Boost.Geometry object etc.
     * Using std::shared_ptr<void> allows storing different geometry types,
     * but requires careful casting or type identification elsewhere.
     * A more type-safe approach might use std::variant or a dedicated geometry class hierarchy.
     */
    // using GeometryPtr = std::shared_ptr<void>; // Example, concrete type needed

    // Let's define a placeholder or require a concrete geometry library type later.
    // For now, avoid defining GeometryPtr to prevent forcing a specific library prematurely.

    /**
     * @brief Key structure for the data chunk cache.
     * Moved here from data_chunk_cache.h to resolve circular dependencies.
     */
    struct DataChunkKey
    {
        std::string filePath;                      ///< [Chinese comment removed for encoding compatibility]
        std::string variableName;                  ///< [Chinese comment removed for encoding compatibility]
        boost::optional<IndexRange> timeIndexRange;  ///< [Chinese comment removed for encoding compatibility]
        boost::optional<BoundingBox> boundingBox;    ///< [Chinese comment removed for encoding compatibility]
        boost::optional<IndexRange> levelIndexRange; ///< [Chinese comment removed for encoding compatibility]/层级索引范围 (用于网格数据)
        // boost::optional<AttributeFilter> attributeFilter; // (用于矢量数据, 但可能过于复杂不适合做key)
        // boost::optional<CRSInfo> targetCrs;             // (目标CRS，如果Reader转换了数据)
        std::string requestDataType; ///< [Chinese comment removed for encoding compatibility]

        // 构造函数 (可选, 但为了方便可以添加一个)
        DataChunkKey(std::string fp, std::string varName,
                     boost::optional<IndexRange> tRange,
                     boost::optional<BoundingBox> bbox,
                     boost::optional<IndexRange> lRange,
                     std::string reqType)
            : filePath(std::move(fp)), variableName(std::move(varName)),
              timeIndexRange(std::move(tRange)), boundingBox(std::move(bbox)),
              levelIndexRange(std::move(lRange)), requestDataType(std::move(reqType)) {}

        // Equality operator
        bool operator==(const DataChunkKey &other) const
        {
            return filePath == other.filePath &&
                   variableName == other.variableName &&
                   timeIndexRange == other.timeIndexRange && // boost::optional handles comparison correctly
                   boundingBox == other.boundingBox &&       // BoundingBox needs operator==
                   levelIndexRange == other.levelIndexRange &&
                   requestDataType == other.requestDataType;
        }

        // Inequality operator
        bool operator!=(const DataChunkKey &other) const
        {
            return !(*this == other);
        }

        // 添加一个toString方法，用于日志格式化
        std::string toString() const
        {
            std::stringstream ss;
            ss << "DataChunkKey[Path: " << filePath
               << ", Var: " << variableName
               << ", RequestType: " << requestDataType;
            if (timeIndexRange)
            {
                ss << ", TimeIdxRange:{" << timeIndexRange->start << "," << timeIndexRange->count << "}";
            }
            if (boundingBox)
            {
                ss << ", BBox:{" << boundingBox->minX << "," << boundingBox->minY << "...}"; // Simplified
            }
            if (levelIndexRange)
            {
                ss << ", LevelIdxRange:{" << levelIndexRange->start << "," << levelIndexRange->count << "}";
            }
            ss << "]";
            return ss.str();
        }
    };

    /**
     * @brief 进度回调函数类型，用于报告长时间操作的进度
     * @param progress 进度值，范围 0.0-1.0
     * @param message 进度消息
     */
    using ProgressCallback = std::function<void(float progress, const std::string &message)>;

    /**
     * @struct FieldDefinition
     * @brief 表示矢量数据中的一个字段定义
     */
    struct FieldDefinition
    {
        std::string name;        ///< [Chinese comment removed for encoding compatibility]
        std::string description; ///< [Chinese comment removed for encoding compatibility]
        std::string dataType;    ///< [Chinese comment removed for encoding compatibility]
        std::string type;        ///< [Chinese comment removed for encoding compatibility]
        bool isNullable = true;  ///< [Chinese comment removed for encoding compatibility]
        int width = 0;           ///< [Chinese comment removed for encoding compatibility]
        int precision = 0;       ///< [Chinese comment removed for encoding compatibility]

        // 比较运算符
        bool operator==(const FieldDefinition &other) const
        {
            return name == other.name &&
                   description == other.description &&
                   dataType == other.dataType &&
                   type == other.type &&
                   isNullable == other.isNullable &&
                   width == other.width &&
                   precision == other.precision;
        }
    };

    /**
     * @class FeatureCollection
     * @brief 表示矢量要素集合
     */
    class FeatureCollection
    {
    public:
        // 构造函数
        FeatureCollection() = default;

        // 新增字段
        std::string name;                              ///< [Chinese comment removed for encoding compatibility]
        std::vector<FieldDefinition> fieldDefinitions; ///< [Chinese comment removed for encoding compatibility]
        boost::optional<CRSInfo> crs;                    ///< [Chinese comment removed for encoding compatibility]
        boost::optional<BoundingBox> extent;             ///< [Chinese comment removed for encoding compatibility]

        // 添加要素
        void addFeature(const Feature &feature)
        {
            mFeatures.push_back(feature);
        }

        // 获取所有要素
        const std::vector<Feature> &getFeatures() const
        {
            return mFeatures;
        }

        // 获取要素数量
        size_t size() const
        {
            return mFeatures.size();
        }

        // 是否为空
        bool empty() const
        {
            return mFeatures.empty();
        }

        // 清空要素
        void clear()
        {
            mFeatures.clear();
        }

        // 迭代器访问
        typename std::vector<Feature>::iterator begin() { return mFeatures.begin(); }
        typename std::vector<Feature>::iterator end() { return mFeatures.end(); }
        typename std::vector<Feature>::const_iterator begin() const { return mFeatures.begin(); }
        typename std::vector<Feature>::const_iterator end() const { return mFeatures.end(); }

        /**
         * @brief 生成要素集合的字符串表示
         * @return 字符串表示
         */
        std::string toString() const;

    private:
        std::vector<Feature> mFeatures;
    };

    /**
     * @enum ColorInterpretation
     * @brief 定义颜色解释类型
     */
    enum class ColorInterpretation
    {
        UNKNOWN_TYPE = 0,
        GRAY = 1,
        PALETTE = 2,
        RED = 3,
        GREEN = 4,
        BLUE = 5,
        ALPHA = 6,
        HUE = 7,
        SATURATION = 8,
        LIGHTNESS = 9,
        CYAN = 10,
        MAGENTA = 11,
        YELLOW = 12,
        BLACK = 13,
        YCbCr_Y = 14,
        YCbCr_Cb = 15,
        YCbCr_Cr = 16,
        COMPRESSED = 17
    };

    /**
     * @class Geometry
     * @brief 表示几何对象的类
     */
    class Geometry
    {
    public:
        enum class Type
        {
            UNKNOWN = 0,
            POINT,
            LINESTRING,
            POLYGON,
            MULTIPOINT,
            MULTILINESTRING,
            MULTIPOLYGON,
            GEOMETRYCOLLECTION
        };

        explicit Geometry(Type type = Type::UNKNOWN) : type_(type) {}

        Type getType() const { return type_; }
        void setType(Type type) { type_ = type; }

        // 添加成员变量
        std::string wkt; // WKT 格式的几何数据
        int wkb = 0;     // 原始WKB几何类型

    private:
        Type type_;
    };

    /**
     * @struct RasterWindow
     * @brief 表示栅格数据窗口的结构体
     */
    struct RasterWindow
    {
        int x;      ///< [Chinese comment removed for encoding compatibility]
        int y;      ///< [Chinese comment removed for encoding compatibility]
        int width;  ///< [Chinese comment removed for encoding compatibility]
        int height; ///< [Chinese comment removed for encoding compatibility]

        /**
         * @brief 默认构造函数
         */
        RasterWindow() : x(0), y(0), width(0), height(0) {}

        /**
         * @brief 构造函数
         * @param x_val X坐标
         * @param y_val Y坐标
         * @param w 宽度
         * @param h 高度
         */
        RasterWindow(int x_val, int y_val, int w, int h)
            : x(x_val), y(y_val), width(w), height(h) {}
    };

    /**
     * @brief 获取指定数据类型的字节大小
     * @param dataType 数据类型枚举值
     * @return 字节大小
     */
    inline size_t getDataTypeSize(DataType dataType)
    {
        switch (dataType)
        {
        case DataType::Byte:
            return sizeof(unsigned char);
        case DataType::UInt16:
            return sizeof(uint16_t);
        case DataType::Int16:
            return sizeof(int16_t);
        case DataType::UInt32:
            return sizeof(uint32_t);
        case DataType::Int32:
            return sizeof(int32_t);
        case DataType::Float32:
            return sizeof(float);
        case DataType::Float64:
            return sizeof(double);
        case DataType::Complex16: // 修正: 复数类型使用枚举值名称
            return 2 * sizeof(int16_t);
        case DataType::Complex32: // 修正: 复数类型使用枚举值名称
            return 2 * sizeof(int32_t);
        case DataType::Complex64: // 修正: 复数类型使用枚举值名称
            return 2 * sizeof(float);
        case DataType::String:
            // 字符串类型返回sizeof(char*)，实际大小由内容决定
            return sizeof(char *);
        case DataType::Boolean:
            return sizeof(bool);
        case DataType::Unknown:
        case DataType::Binary:
        default:
            return 0;
        }
    }

    /**
     * @brief 网格索引结构体
     * 用于表示网格数据中的索引位置
     */
    struct GridIndex {
        int x;                    ///< [Chinese comment removed for encoding compatibility]
        int y;                    ///< [Chinese comment removed for encoding compatibility]
        boost::optional<int> z;     ///< [Chinese comment removed for encoding compatibility]
        boost::optional<int> t;     ///< [Chinese comment removed for encoding compatibility]
        
        /**
         * @brief 构造函数
         * @param xIdx X维度索引
         * @param yIdx Y维度索引
         * @param zIdx Z维度索引(可选)
         * @param tIdx 时间维度索引(可选)
         */
        GridIndex(int xIdx, int yIdx, boost::optional<int> zIdx = boost::none, boost::optional<int> tIdx = boost::none)
            : x(xIdx), y(yIdx), z(zIdx), t(tIdx) {}
        
        /**
         * @brief 等值运算符
         */
        bool operator==(const GridIndex& other) const {
            return x == other.x && y == other.y && z == other.z && t == other.t;
        }
        
        /**
         * @brief 不等运算符
         */
        bool operator!=(const GridIndex& other) const {
            return !(*this == other);
        }
    };

    /**
     * @brief 边界框结构
     */

    // === UNIFIED OPERATION RESULT TEMPLATE ===
    
    /**
     * @template OperationResult
     * @brief 统一的操作结果模板，用于所有服务的返回值
     * @tparam T 结果数据类型
     */
    template<typename T>
    struct OperationResult {
        bool success = false;                           ///< 操作是否成功
        T data;                                         ///< 结果数据
        std::string errorMessage;                       ///< 错误信息
        std::vector<std::string> warnings;              ///< 警告信息
        std::chrono::milliseconds executionTime{0};     ///< 执行时间
        std::map<std::string, std::string> metadata;    ///< 附加元数据
        
        /**
         * @brief 默认构造函数
         */
        OperationResult() = default;
        
        /**
         * @brief 成功结果构造函数
         * @param result_data 结果数据
         */
        explicit OperationResult(T result_data) 
            : success(true), data(std::move(result_data)) {}
        
        /**
         * @brief 失败结果构造函数
         * @param error_message 错误信息
         */
        explicit OperationResult(const std::string& error_message)
            : success(false), errorMessage(error_message) {}
        
        /**
         * @brief 检查操作是否成功
         */
        bool isSuccess() const { return success; }
        
        /**
         * @brief 检查是否有警告
         */
        bool hasWarnings() const { return !warnings.empty(); }
        
        /**
         * @brief 添加警告信息
         */
        void addWarning(const std::string& warning) {
            warnings.push_back(warning);
        }
        
        /**
         * @brief 设置元数据
         */
        void setMetadata(const std::string& key, const std::string& value) {
            metadata[key] = value;
        }
        
        /**
         * @brief 获取元数据
         */
        boost::optional<std::string> getMetadata(const std::string& key) const {
            auto it = metadata.find(key);
            return (it != metadata.end()) ? boost::optional<std::string>(it->second) : boost::none;
        }
    };

    // 参数: 文件导出
    struct ExportParameters {
        std::string targetPath; // 完整文件路径
        std::string format;     // e.g., "txt", "csv", "nc", "geojson"
        // 用于向特定Writer传递额外指令, e.g., {"columns", vector<string>{"lat", "lon"}}
        std::map<std::string, boost::any> formatSpecifics;
    };

    inline std::shared_ptr<GridData> GridData::createSlice(size_t startRow, size_t numRows) const {
        if (startRow + numRows > _definition.rows) {
            throw std::out_of_range("Slice dimensions are out of range for GridData.");
        }

        // 1. Create a new GridDefinition for the slice
        GridDefinition sliceDef = _definition;
        sliceDef.rows = numRows;
        
        // 2. Slice the Y coordinates
        if (!_definition.yDimension.coordinates.empty()) {
            sliceDef.yDimension.coordinates.assign(
                _definition.yDimension.coordinates.begin() + startRow,
                _definition.yDimension.coordinates.begin() + startRow + numRows
            );
        }

        // 3. Create a new GridData object for the slice
        auto sliceGridData = std::make_shared<GridData>(sliceDef, _dataType, _bandCount);
        sliceGridData->setCrs(_crs);
        sliceGridData->setGeoTransform(_geoTransform);
        // Note: GeoTransform might need adjustment if it's not just an affine transform, but for now we copy it.
        sliceGridData->setVariableName(_variableName);
        sliceGridData->setUnits(_units);
        sliceGridData->setFillValue(_fillValue);
        
        // 4. Copy the data buffer for the slice
        const size_t rowPitchBytes = _definition.cols * _bandCount * getElementSizeBytes();
        const size_t sliceDataSizeBytes = rowPitchBytes * numRows;
        sliceGridData->resizeUnifiedBuffer(sliceDataSizeBytes);
        
        const unsigned char* sourceDataStart = _buffer.data() + startRow * rowPitchBytes;
        
        std::memcpy(sliceGridData->getData().data(), sourceDataStart, sliceDataSizeBytes);

        return sliceGridData;
    }

    // ===== 🔧 类型统一别名（解决重复定义问题）=====
    // 使用BoundingBox作为统一的空间边界类型
    using SpatialBounds = BoundingBox;               ///< 统一的空间边界类型
    using TemporalBounds = TimeRange;                ///< 时间边界统一使用TimeRange

    // 🆕 添加缺失的插值方法枚举（从插值服务接口引入）
    /**
     * @brief 插值方法枚举
     */
    enum class InterpolationMethod {
        UNKNOWN,
        LINEAR_1D,                // 1D 线性插值
        CUBIC_SPLINE_1D,          // 1D 立方样条插值
        NEAREST_NEIGHBOR,         // N-D 最近邻插值
        BILINEAR,                 // 2D 双线性插值 (通常用于规则网格)
        BICUBIC,                  // 2D 双三次插值 (通常用于规则网格)
        TRILINEAR,                // 3D 三线性插值 (通常用于规则网格)
        TRICUBIC,                 // 3D 三次插值 (通常用于规则网格)
        PCHIP_RECURSIVE_NDIM,     // N-D 分段三次 Hermite 插值 (PCHIP), 递归实现
        PCHIP_MULTIGRID_NDIM,     // N-D PCHIP, 基于预计算和多重网格思想
        PCHIP_OPTIMIZED_2D_BATHY, // 针对2D水深优化PCHIP
        PCHIP_OPTIMIZED_3D_SVP,   // 针对3D声速剖面优化PCHIP
        PCHIP_FAST_2D,            // 2D PCHIP (高性能预计算版)
        PCHIP_FAST_3D             // 3D PCHIP (高性能预计算版)
    };

    // 🆕 添加输出格式枚举
    namespace output {
        /**
         * @brief 输出格式枚举
         */
        enum class OutputFormat {
            UNKNOWN,
            NETCDF,          // NetCDF格式
            GEOTIFF,         // GeoTIFF格式
            PNG,             // PNG图像格式
            JPEG,            // JPEG图像格式
            CSV,             // CSV文本格式
            JSON,            // JSON格式
            GEOJSON,         // GeoJSON格式
            SHAPEFILE,       // Shapefile格式
            HDF5,            // HDF5格式
            ZARR,            // Zarr格式
            PARQUET          // Parquet格式
        };
    }

} // namespace oscean::core_services

// 🔧 添加 fmt formatter 特化支持 DataType（在命名空间外）
#include <fmt/format.h>
#include <cstring> // for std::memcpy

template <>
struct fmt::formatter<oscean::core_services::DataType> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const oscean::core_services::DataType& type, FormatContext& ctx) const -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{}", oscean::core_services::dataTypeToString(type));
    }
};

#include "hash_specializations.h" // Include the new header for hash specializations

#endif // OSCEAN_CORE_SERVICES_COMMON_DATA_TYPES_H
