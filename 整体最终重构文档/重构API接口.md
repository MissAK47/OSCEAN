# OSCEAN核心服务API接口定义

本文档定义了OSCEAN项目核心服务中使用的主要数据结构。这些结构旨在为不同类型的数据（特别是与海洋环境相关的NetCDF文件和地理空间矢量文件）提供统一的表示。目标是实现高度的灵活性和可扩展性，以适应多样化的数据格式和未来的需求。

## 文件位置

这些定义应位于 `core_service_interfaces/include/core_services/common_data_types.h`。

## 核心数据结构

```cpp
#ifndef OSCEAN_CORE_SERVICES_COMMON_DATA_TYPES_H
#define OSCEAN_CORE_SERVICES_COMMON_DATA_TYPES_H

#include <string>
#include <vector>
#include <variant>
#include <map>
#include <optional>
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <numeric>   // For std::accumulate if needed for strides
#include <algorithm> // For std::find_if

// Forward declarations for types assumed to be defined elsewhere or later in this file
// These would typically be in their own headers or a common utility header.
namespace oscean {
namespace core_services {
    // DataType enum is crucial and assumed to be defined.
    // Example:
    enum class DataType {
        Unknown, Byte, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
        Float32, Float64, String, ComplexFloat32, ComplexFloat64
        // Note: String type size calculation in getDataTypeSizeBytes needs careful handling.
    };

    // Helper function to get data type size. Needs robust implementation.
    inline size_t getDataTypeSizeBytes(DataType dt) {
        switch (dt) {
            case DataType::Byte: case DataType::Int8: case DataType::UInt8: return 1;
            case DataType::Int16: case DataType::UInt16: return 2;
            case DataType::Int32: case DataType::UInt32: case DataType::Float32: return 4;
            case DataType::Int64: case DataType::UInt64: case DataType::Float64: return 8;
            case DataType::ComplexFloat32: return 8;  // 2 * float32
            case DataType::ComplexFloat64: return 16; // 2 * float64
            case DataType::String: return 0; // Size of std::string is variable, this indicates an array of strings.
                                           // Actual memory is managed by std::string itself.
                                           // For buffer calculations, this often implies an array of pointers or offsets.
            case DataType::Unknown: default: return 0; // Or throw an exception for unknown types.
        }
    }
}
}

namespace oscean {
namespace core_services {

/**
 * @enum CoordinateDimension
 * @brief Defines the type of a coordinate dimension.
 * Provides a standardized way to identify common geophysical and other dimensions.
 */
enum class CoordinateDimension {
    LON,         ///< Longitude, typically representing the X-axis in geographic contexts.
    LAT,         ///< Latitude, typically representing the Y-axis in geographic contexts.
    VERTICAL,    ///< Vertical dimension (e.g., height, depth, pressure level).
    TIME,        ///< Temporal dimension.
    SPECTRAL,    ///< Spectral dimension (e.g., wavelength, band for rasters).
    BAND,        ///< Explicitly for raster bands if SPECTRAL is used for something else or for clarity.
    INSTANCE,    ///< For instance or ensemble members.
    FEATURE_ID,  ///< For discrete feature identifiers (e.g., in vector data represented as a grid).
    STRING_CHAR, ///< For dimensions representing characters of strings (e.g. NetCDF char arrays for strings).
    OTHER,       ///< A custom or otherwise categorized dimension. Use 'name' in DimensionCoordinateInfo to specify.
    NONE         ///< Represents an unspecified, non-existent, or uninitialized dimension.
};

/**
 * @typedef AttributeValue
 * @brief A variant type to store various kinds of metadata attribute values.
 *
 * This allows for flexible storage of common attribute types found in scientific datasets,
 * including single values and arrays of values.
 */
using AttributeValue = std::variant<
    std::monostate,          ///< Represents an empty or unset state.
    bool,                    ///< Boolean value.
    char,                    ///< Character value. (Added for char attributes)
    long long,               ///< Integer value (using long long for wider range).
    unsigned long long,      ///< Unsigned integer value. (Added for unsigned attributes)
    double,                  ///< Double-precision floating-point value.
    std::string,             ///< String value.
    std::vector<char>,       ///< Vector of char values.
    std::vector<long long>,  ///< Vector of integer values.
    std::vector<unsigned long long>, ///< Vector of unsigned integer values.
    std::vector<double>,     ///< Vector of double-precision floating-point values.
    std::vector<std::string> ///< Vector of string values
>;

/**
 * @struct ValueRange
 * @brief Represents a range with optional minimum and maximum values.
 */
struct ValueRange {
    std::optional<double> min_val; ///< Optional minimum value of the range.
    std::optional<double> max_val; ///< Optional maximum value of the range.

    ValueRange() = default;
    ValueRange(double min_v, double max_v) : min_val(min_v), max_val(max_v) {}

    bool isWithin(double value) const {
        if (min_val && value < *min_val) return false;
        if (max_val && value > *max_val) return false;
        return true;
    }
};

/**
 * @struct DimensionCoordinateInfo
 * @brief Provides comprehensive information about a single coordinate dimension.
 */
struct DimensionCoordinateInfo {
    // === Basic Identification ===
    std::string name;                               ///< Dimension name as defined in the source (e.g., "lat", "time", "band1").
                                                    ///< For dimensions not having an explicit name but identified by type (e.g. an X dimension),
                                                    ///< a canonical name can be assigned (e.g. "x", "longitude").
    CoordinateDimension type = CoordinateDimension::NONE; ///< Semantic type of the dimension.
    std::string units;                              ///< Units of the coordinate values (e.g., "degrees_north", "m", "seconds since 1970-01-01").
    std::string standardName;                       ///< CF Standard Name, if available (e.g., "latitude", "projection_x_coordinate").
    std::string longName;                           ///< Descriptive long name, if available.
    size_t length = 0;                              ///< Number of levels or points along this dimension. Updated from coordinates.size().

    // === Coordinate Data ===
    // It's one or the other, or neither if it's an abstract dimension without explicit coordinates.
    std::vector<double> coordinates;                ///< Numeric coordinate values (e.g., [0.0, 10.0, 20.0]).
    std::vector<std::string> coordinateLabels;      ///< Textual labels for coordinates (e.g., ["CategoryA", "CategoryB"]).
                                                    ///< Use if the dimension is categorical.
    std::vector<std::vector<double>> coordinateBounds; ///< Bounds for each coordinate level (e.g., for cell boundaries).
                                                    ///< Outer vector size matches coordinates.size(). Inner vector typically size 2 [min, max].

    // === Coordinate Characteristics ===
    bool isRegular = false;                         ///< True if coordinate spacing is uniform.
    std::optional<double> resolution;               ///< Coordinate resolution if regular, otherwise std::nullopt or an average.
    bool isAscending = true;                        ///< True if coordinate values are strictly increasing.
    bool isCyclic = false;                          ///< True if the dimension is cyclic (e.g., longitude).
    std::optional<double> cycleLength;              ///< Length of the cycle if isCyclic is true (e.g., 360.0 for longitude).
    bool isDiscrete = true;                         ///< True if values are discrete points rather than continuous ranges (default for most coordinates).

    // === Data Quality and Range ===
    std::optional<ValueRange> validRange;           ///< Recommended valid range for coordinate values.
    std::optional<double> missingValue;             ///< Specific value indicating missing numeric coordinates (use with 'coordinates').
    std::optional<double> fillValue;                ///< Specific fill value for numeric coordinates.
    std::optional<double> accuracy;                 ///< Accuracy of the coordinate values.

    // === Coordinate Transformations (Common CF attributes for data variables, but can apply to coordinates too) ===
    std::optional<double> scale_factor;             ///< Scale factor to apply to coordinate values.
    std::optional<double> add_offset;               ///< Offset to apply after scaling.
    std::optional<std::string> formula_terms;       ///< CF formula_terms attribute string.

    // === Reference System (Optional, for specific dimension types) ===
    std::optional<std::string> datum;               ///< Datum information, if applicable (e.g., for vertical dimensions).

    // === Dimension-Specific Information (Using std::variant for type-safe specific properties) ===
    struct TimeSpecificInfo {
        std::string calendar;                       ///< CF Calendar attribute (e.g., "gregorian", "proleptic_gregorian", "noleap").
        std::string referenceEpochString;           ///< Original reference epoch string from units (e.g., "days since 1970-01-01 00:00:00").
        // Consider adding a parsed epoch time_point here for easier use, if needed frequently.
        std::map<std::string, AttributeValue> customAttributes; ///< For future or non-standard time attributes.
    };
    struct VerticalSpecificInfo {
        bool positiveUp = true;                     ///< CF "positive" attribute ("up" or "down").
        std::string verticalDatumName;              ///< Name of the vertical datum, if specified.
        std::optional<std::string> formula;         ///< E.g., for computed vertical coordinates like sigma levels.
        std::map<std::string, std::string> formulaParameters; ///< Parameters for the formula.
        std::map<std::string, AttributeValue> customAttributes; ///< For future or non-standard vertical attributes.
    };
    struct SpectralSpecificInfo {
        std::string spectralUnitOverride;           ///< If units for spectral bands differ from main 'units'.
        bool isWavelength = true;                   ///< True if coordinates are wavelengths, false for wavenumbers, etc.
        std::map<std::string, AttributeValue> customAttributes; ///< For future or non-standard spectral attributes.
    };
    // Can add more *SpecificInfo structs for other CoordinateDimension types if they have unique, structured properties.
    struct OtherSpecificInfo {
        std::map<std::string, AttributeValue> attributes; ///< Generic attributes for OTHER dimension types.
    };

    std::variant<std::monostate, TimeSpecificInfo, VerticalSpecificInfo, SpectralSpecificInfo, OtherSpecificInfo> specificInfo;

    // === General Metadata Extension ===
    std::map<std::string, AttributeValue> attributes; ///< Any other attributes associated with this dimension.

    DimensionCoordinateInfo() : specificInfo(std::monostate{}) {}

    size_t getNumberOfLevels() const { // Renamed from getLength for clarity
        if (!coordinates.empty()) return coordinates.size();
        if (!coordinateLabels.empty()) return coordinateLabels.size();
        // If length is explicitly set and no coordinates/labels, use that.
        // This might be for an abstract dimension whose size is known but coordinates are implicit or unnecessary.
        if (length > 0) return length;
        return 0; // Or 1 if it's a scalar dimension implied by presence but no explicit levels.
    }
    bool hasNumericCoordinates() const { return !coordinates.empty(); }
    bool hasTextualLabels() const { return !coordinateLabels.empty(); }
    bool hasBoundaries() const { return !coordinateBounds.empty(); }

    template<typename T> const T* getSpecificInfoAs() const { return std::get_if<T>(&specificInfo); }
    template<typename T> T* getSpecificInfoAs() { return std::get_if<T>(&specificInfo); }

    void updateLengthFromCoordinates() {
        if (!coordinates.empty()) length = coordinates.size();
        else if (!coordinateLabels.empty()) length = coordinateLabels.size();
        // Do not set length to 0 if it was already set and coordinates are empty.
    }
};


/**
 * @enum CRSType
 * @brief Indicates the type or format of the CRS definition string.
 * Suggested optional enhancement for CRSInfo.
 */
enum class CRSType {
    UNKNOWN,        ///< CRS type is not known or not specified.
    WKT2,           ///< OGC Well-Known Text 2.x string.
    WKT1_GDAL,      ///< OGC Well-Known Text 1 (GDAL/ESRI dialect).
    PROJ_STRING,    ///< PROJ.4 or PROJ string (e.g., "+proj=utm +zone=10 +datum=WGS84").
    EPSG_CODE,      ///< EPSG code (e.g., "EPSG:4326"). The srsDefinition would be "EPSG:4326".
    URN_OGC_DEF_CRS ///< OGC URN (e.g. "urn:ogc:def:crs:EPSG::4326")
    // ESRI_WKT (if needing to distinguish from WKT1_GDAL specifically)
};

/**
 * @class CRSInfo
 * @brief Represents Coordinate Reference System information.
 */
class CRSInfo {
public:
    std::string srsDefinition; ///< The CRS definition string (e.g., WKT, Proj string, "EPSG:XXXX").
    
    // --- Optional Enhanced Fields (for better CRS context) ---
    CRSType definitionType = CRSType::UNKNOWN; ///< Type of the srsDefinition string.
    std::string name;                          ///< Human-readable name of the CRS (e.g., "WGS 84", "NAD83 / UTM zone 10N").
    std::string authorityName;                 ///< Authority that defined the CRS (e.g., "EPSG", "ESRI").
    std::string authorityCode;                 ///< Code within the authority (e.g., "4326", "26910").
    std::string areaOfUse;                     ///< Description of the area where this CRS is valid.
    std::optional<BoundingBox> geographicBounds; ///< Optional geographic bounding box of the CRS validity area.

    CRSInfo(const std::string& crsString = "", CRSType type = CRSType::UNKNOWN)
        : srsDefinition(crsString), definitionType(type) {
        // Basic parsing if type is EPSG_CODE and crsString matches "EPSG:XXXX"
        if (type == CRSType::EPSG_CODE) {
            if (crsString.rfind("EPSG:", 0) == 0 || crsString.rfind("epsg:", 0) == 0) {
                authorityName = "EPSG";
                authorityCode = crsString.substr(5);
            }
        }
        // More advanced parsing (e.g. WKT to extract name) could be done here or by a utility.
    }

    bool isValid() const {
        return !srsDefinition.empty() && definitionType != CRSType::UNKNOWN;
    }

    // Equality operator for comparisons
    bool operator==(const CRSInfo& other) const {
        return srsDefinition == other.srsDefinition &&
               definitionType == other.definitionType && // Include other fields if they are considered defining
               authorityCode == other.authorityCode &&
               authorityName == other.authorityName;
    }
    bool operator!=(const CRSInfo& other) const {
        return !(*this == other);
    }
};

/**
 * @class BoundingBox
 * @brief Represents a bounding box, potentially with a Z range and associated CRS.
 */
class BoundingBox {
public:
    double minX = 0.0, minY = 0.0; // Required 2D extent
    double maxX = 0.0, maxY = 0.0;
    std::optional<double> minZ, maxZ; // Optional vertical extent
    CRSInfo crs;                      // CRS of these coordinate values

    BoundingBox() = default;
    BoundingBox(double lonMin, double latMin, double lonMax, double latMax, 
                const CRSInfo& boxCrs = CRSInfo())
        : minX(lonMin), minY(latMin), maxX(lonMax), maxY(latMax), crs(boxCrs) {}

    bool hasZ() const { return minZ.has_value() && maxZ.has_value(); }
};


/**
 * @class GridDefinition
 * @brief Defines the N-dimensional structure and metadata of a grid or variable.
 * This structure is designed to be flexible and self-describing, driven by the actual
 * dimensions and metadata present in the source data (e.g., a NetCDF variable).
 */
class GridDefinition {
public:
    std::string gridName;                                 ///< Name of the grid or variable (e.g., "sea_surface_temperature").
    CRSInfo crs;                                          ///< Coordinate Reference System of the grid's spatial dimensions.
                                                          ///< This applies to dimensions identified as LON, LAT, or projected X, Y.
    std::optional<BoundingBox> extent;                    ///< Overall bounding box of the grid in its native CRS.
                                                          ///< May be calculated from coordinate dimensions.
    DataType originalDataType = DataType::Unknown;        ///< Data type of the variable in its original source file.
    std::map<std::string, AttributeValue> globalAttributes; ///< Attributes associated with this grid/variable (not dimension-specific).

    // --- Flexible Dimension Representation ---
    std::vector<DimensionCoordinateInfo> dimensions;      ///< Detailed information for ALL dimensions of this grid.
                                                          ///< The order in this vector is typically the logical order or order of definition.

    // --- Data Layout Information ---
    // Defines the physical storage order of dimensions in GridData::_buffer.
    // The elements are the *types* of the dimensions in that order.
    // For example: {CoordinateDimension::TIME, CoordinateDimension::LAT, CoordinateDimension::LON}
    // means TIME is the slowest varying, LON is the fastest.
    // If multiple dimensions share a type (e.g. two CoordinateDimension::OTHER),
    // then 'name' in DimensionCoordinateInfo must be used to disambiguate if necessary for layout.
    // For robust mapping, this could also be std::vector<std::string> storing dimension names
    // if CoordinateDimension enum is insufficient for unique layout definition in all cases.
    // Sticking to CoordinateDimension for now, assuming 'type' is usually sufficient for layout,
    // or names are used by convention/reader logic to map to these types.
    std::vector<CoordinateDimension> dimensionOrderInDataLayout;

    GridDefinition() = default;

    // --- Helper methods for accessing dimension information ---

    /**
     * @brief Gets information for a dimension by its semantic type.
     * @param type The CoordinateDimension type to search for.
     * @return Pointer to DimensionCoordinateInfo if found, nullptr otherwise.
     * @note If multiple dimensions have the same type (e.g. two 'OTHER' types),
     *       this will return the first one found. Use getDimensionInfoByName for uniqueness.
     */
    const DimensionCoordinateInfo* getDimensionInfoByType(CoordinateDimension type) const {
        auto it = std::find_if(dimensions.begin(), dimensions.end(),
                               [type](const DimensionCoordinateInfo& dimInfo){ return dimInfo.type == type; });
        return (it != dimensions.end()) ? &(*it) : nullptr;
    }
    DimensionCoordinateInfo* getDimensionInfoByType(CoordinateDimension type) {
        auto it = std::find_if(dimensions.begin(), dimensions.end(),
                               [type](const DimensionCoordinateInfo& dimInfo){ return dimInfo.type == type; });
        return (it != dimensions.end()) ? &(*it) : nullptr;
    }

    /**
     * @brief Gets information for a dimension by its name.
     * @param name The name of the dimension (DimensionCoordinateInfo::name).
     * @return Pointer to DimensionCoordinateInfo if found, nullptr otherwise.
     */
    const DimensionCoordinateInfo* getDimensionInfoByName(const std::string& name) const {
        auto it = std::find_if(dimensions.begin(), dimensions.end(),
                               [&name](const DimensionCoordinateInfo& dimInfo){ return dimInfo.name == name; });
        return (it != dimensions.end()) ? &(*it) : nullptr;
    }
    DimensionCoordinateInfo* getDimensionInfoByName(const std::string& name) {
        auto it = std::find_if(dimensions.begin(), dimensions.end(),
                               [&name](const DimensionCoordinateInfo& dimInfo){ return dimInfo.name == name; });
        return (it != dimensions.end()) ? &(*it) : nullptr;
    }

    /**
     * @brief Gets the number of levels (length) for a dimension of a specific type.
     * @param dimType The CoordinateDimension type.
     * @return Number of levels, or 0 if dimension not found or has no levels.
     */
    size_t getLevelsForDimension(CoordinateDimension dimType) const {
        const DimensionCoordinateInfo* dimInfo = getDimensionInfoByType(dimType);
        return dimInfo ? dimInfo->getNumberOfLevels() : 0;
    }
     // Overload for by name, if needed
    size_t getLevelsForDimension(const std::string& dimName) const {
        const DimensionCoordinateInfo* dimInfo = getDimensionInfoByName(dimName);
        return dimInfo ? dimInfo->getNumberOfLevels() : 0;
    }


    /**
     * @brief Calculates the total number of elements in the grid based on dimensionOrderInDataLayout.
     * @return Total number of elements. Returns 0 if layout is empty or any dimension in layout has 0 levels.
     *         Returns 1 if layout is empty but it represents a scalar (e.g., a single global attribute as data).
     *         This logic needs careful implementation based on how scalars are represented.
     */
    size_t getTotalElements() const {
        if (dimensionOrderInDataLayout.empty()) {
            // Handle scalar case: if 'dimensions' vector is empty or all dimensions have length 1, it might be a scalar.
            // Or if this GridDefinition is for a single value (e.g. a global attribute being read as data).
            // A common convention for scalars is for 'dimensions' to be empty or have one dimension of length 1.
            if (dimensions.empty()) return 1; // Scalar by absence of dimensions
            bool allLengthOne = true;
            for(const auto& dimInfo : dimensions) {
                if(dimInfo.getNumberOfLevels() > 1) {
                    allLengthOne = false;
                    break;
                }
            }
            if(allLengthOne && !dimensions.empty()) return 1; // Scalar if all existing dimensions are length 1

            return 0; // Or 1, depending on convention for "dimension-less" data. Let's assume 0 if layout is key.
        }

        size_t totalElements = 1;
        bool hasValidDimensionInLayout = false;
        for (const auto& dimLayoutType : dimensionOrderInDataLayout) {
            // Need to map dimLayoutType to the actual DimensionCoordinateInfo to get its length.
            // This assumes dimLayoutType uniquely identifies a dimension in the 'dimensions' vector.
            // If multiple dimensions can have the same 'type', then 'dimensionOrderInDataLayout'
            // should perhaps store names or indices into the 'dimensions' vector.
            // For now, assuming getLevelsForDimension(dimLayoutType) works.
            size_t levels = getLevelsForDimension(dimLayoutType);

            if (levels == 0) {
                // If any dimension in the layout has zero size, the total is zero (unless it's a special case like a string array).
                // For a 0-sized dimension in a data variable, it usually means no data.
                return 0;
            }
            totalElements *= levels;
            hasValidDimensionInLayout = true;
        }
        // If dimensionOrderInDataLayout is not empty but found no valid dimensions (e.g. all had 0 levels indirectly)
        return hasValidDimensionInLayout ? totalElements : 0;
    }
};

/**
 * @class GridData
 * @brief Holds the actual N-dimensional grid data along with its definition.
 */
class GridData {
private:
    GridDefinition _definition;
    DataType _internalDataType = DataType::Unknown; ///< The actual data type stored in _buffer.
                                                    ///< This might differ from _definition.originalDataType if conversion occurred.
    std::vector<unsigned char> _buffer;             ///< Raw byte buffer storing the grid data.

    /**
     * @brief Calculates the byte offset into the _buffer for a given set of N-D indices.
     * The order of indices in the 'indices' vector must correspond to the order
     * specified in _definition.dimensionOrderInDataLayout.
     * @param indices A vector of 0-based indices, one for each dimension in data layout order.
     * @return The byte offset.
     * @throws std::invalid_argument If indices.size() doesn't match layout rank.
     * @throws std::out_of_range If any index is out of bounds for its dimension.
     * @throws std::logic_error If data type size is zero or dimension in layout has zero size.
     */
    size_t calculateByteOffset(const std::vector<size_t>& indices) const {
        if (indices.size() != _definition.dimensionOrderInDataLayout.size()) {
            throw std::invalid_argument("Number of indices must match data layout rank.");
        }

        size_t elementOffset = 0;
        size_t stride = 1;
        size_t typeSize = getDataTypeSizeBytes(_internalDataType);

        if (typeSize == 0 && _internalDataType != DataType::String) { 
             throw std::logic_error("Data type size is zero for non-string type during offset calculation.");
        }

        // Iterate from the fastest varying dimension (last in layout) to slowest (first in layout)
        // to correctly calculate the linear offset.
        for (int i = _definition.dimensionOrderInDataLayout.size() - 1; i >= 0; --i) {
            CoordinateDimension currentDimLayoutType = _definition.dimensionOrderInDataLayout[i];
            size_t currentIndexForLayoutDim = indices[i]; // The index provided for this layout dimension

            // Get the actual size of this dimension
            // This assumes currentDimLayoutType uniquely identifies a dimension.
            // If not, dimensionOrderInDataLayout might need to store names or direct indices.
            const DimensionCoordinateInfo* dimInfo = _definition.getDimensionInfoByType(currentDimLayoutType);
            if (!dimInfo) {
                 throw std::logic_error("Dimension type in layout not found in GridDefinition's dimensions list.");
            }
            size_t currentDimSize = dimInfo->getNumberOfLevels();

            if (currentDimSize == 0 && _definition.dimensionOrderInDataLayout.size() > 0) {
                 // This case should ideally be caught by getTotalElements() returning 0.
                 throw std::logic_error("Dimension in layout has zero size during offset calculation.");
            }
            if (currentIndexForLayoutDim >= currentDimSize && currentDimSize > 0) {
                // Construct a more informative error message
                std::string errorMsg = "Index " + std::to_string(currentIndexForLayoutDim) +
                                       " out of bounds for dimension of type " + std::to_string(static_cast<int>(currentDimLayoutType)) +
                                       " (name: " + (dimInfo ? dimInfo->name : "unknown") +
                                       ", size: " + std::to_string(currentDimSize) +
                                       ") at layout position " + std::to_string(i) + ".";
                throw std::out_of_range(errorMsg);
            }

            elementOffset += currentIndexForLayoutDim * stride;

            // Stride for the *next* (slower) dimension is the total number of elements in *all faster* dimensions.
            if (currentDimSize > 0) {
                 stride *= currentDimSize;
            } else if (_definition.dimensionOrderInDataLayout.size() == 1 && currentDimSize == 0) {
                // Special case for a single, 0-sized dimension (e.g. an empty array). Stride remains 1.
                // This might occur if getTotalElements() allowed a 0-size single dimension to be 1.
                 stride = 1;
            }
            // If currentDimSize is 0 in a multi-dimensional layout, it's an issue unless typeSize is also 0 (e.g. array of 0-sized strings)
            // This should ideally be prevented by getTotalElements() check.
        }
        return elementOffset * typeSize;
    }

public:
    GridData() = default;
    GridData(const GridDefinition& gridDef, DataType actualStorageType)
        : _definition(gridDef), _internalDataType(actualStorageType) {
        size_t totalElements = _definition.getTotalElements();
        size_t typeSize = getDataTypeSizeBytes(_internalDataType);

        if (totalElements > 0 && typeSize > 0) {
            _buffer.resize(totalElements * typeSize);
        } else if (totalElements > 0 && _internalDataType == DataType::String) {
             // For String, _buffer might store offsets or pointers if it's a packed array of strings.
             // Or, it might be empty if strings are handled elsewhere (e.g. vector<string>).
             // For now, assume if _internalDataType is String, GridData is not managing a flat byte buffer for it.
             // This part needs careful design if `getValue<std::string>` is to be supported directly from buffer.
             _buffer.clear();
        } else {
            // totalElements is 0, or typeSize is 0 for a non-string type.
            _buffer.clear();
        }
    }
    
    const GridDefinition& getDefinition() const { return _definition; }
    DataType getInternalDataType() const { return _internalDataType; }
    const std::vector<unsigned char>& getRawDataBuffer() const { return _buffer; }
    std::vector<unsigned char>& getRawDataBufferMutable() { return _buffer; }
    size_t getTotalSizeInBytes() const { return _buffer.size(); } // Size of the raw buffer

    template<typename T>
    T getValue(const std::vector<size_t>& indices) const {
        if (getDataTypeSizeBytes(_internalDataType) == 0 || _buffer.empty()) {
             // Special handling for std::string if it's not directly in _buffer
            if constexpr (std::is_same_v<T, std::string>) {
                // Logic to retrieve string would go here, potentially involving other member variables
                // if strings are stored separately. This example assumes direct buffer access.
                throw std::runtime_error("Cannot retrieve std::string: String storage mechanism not fully defined for direct buffer access.");
            }
            throw std::runtime_error("Cannot retrieve value: data type unknown, zero size, or buffer empty.");
        }
        // Add check: if (sizeof(T) != getDataTypeSizeBytes(_internalDataType)) throw type_mismatch_error;
        // This check is tricky if T is different from _internalDataType but convertible.
        // Assume T matches _internalDataType for direct read.
        size_t offset = calculateByteOffset(indices);
        if (offset + sizeof(T) > _buffer.size()) { // Ensure read is within bounds
             throw std::out_of_range("Calculated offset plus type size is out of buffer bounds for reading.");
        }
        return *reinterpret_cast<const T*>(&_buffer[offset]);
    }

    template<typename T>
    void setValue(const std::vector<size_t>& indices, T value) {
        if (getDataTypeSizeBytes(_internalDataType) == 0) {
            if constexpr (std::is_same_v<T, std::string>) {
                 throw std::runtime_error("Cannot set std::string: String storage mechanism not fully defined for direct buffer access.");
            }
             throw std::runtime_error("Cannot set value: data type unknown or zero size.");
        }
        // Add check: if (sizeof(T) != getDataTypeSizeBytes(_internalDataType)) throw type_mismatch_error;
        size_t offset = calculateByteOffset(indices);
        if (offset + sizeof(T) > _buffer.size()) { // Ensure write is within bounds
            throw std::out_of_range("Calculated offset plus type size is out of buffer bounds for writing.");
        }
        *reinterpret_cast<T*>(&_buffer[offset]) = value;
    }
};


// --- Vector Data Structures (New Additions) ---

/**
 * @enum GeometryType
 * @brief Defines standard_services_impl/data_access_service/src/impl/readers/gdal_vector_reader.cpp vector geometry types.
 * Corresponds to OGC WKB geometry types for interoperability.
 */
enum class GeometryType {
    UNKNOWN,
    POINT,                 // OGR: wkbPoint, wkbPoint25D
    LINESTRING,            // OGR: wkbLineString, wkbLineString25D
    POLYGON,               // OGR: wkbPolygon, wkbPolygon25D
    MULTIPOINT,            // OGR: wkbMultiPoint, wkbMultiPoint25D
    MULTILINESTRING,       // OGR: wkbMultiLineString, wkbMultiLineString25D
    MULTIPOLYGON,          // OGR: wkbMultiPolygon, wkbMultiPolygon25D
    GEOMETRYCOLLECTION     // OGR: wkbGeometryCollection, wkbGeometryCollection25D
    // Other OGR types like wkbCircularString, wkbCompoundCurve etc. can be added if needed.
};

/**
 * @struct Geometry
 * @brief Represents a single vector geometry object.
 */
struct Geometry {
    std::vector<unsigned char> wkb; ///< Geometry as Well-Known Binary.
    GeometryType type = GeometryType::UNKNOWN; ///< The primary type of the geometry.
    std::optional<CRSInfo> crs;     ///< Optional CRS for this specific geometry, if it differs from FeatureCollection's CRS.
                                    ///< Usually, geometry shares CRS with its parent FeatureCollection.

    Geometry() = default;
    explicit Geometry(std::vector<unsigned char> wkb_data, GeometryType geom_type = GeometryType::UNKNOWN) // Updated constructor
        : wkb(std::move(wkb_data)), type(geom_type) {}
};

/**
 * @struct FieldDefinition
 * @brief Describes a single field (attribute column) in a FeatureCollection.
 */
struct FieldDefinition {
    std::string name;                           ///< Name of the field.
    DataType type = DataType::Unknown;          ///< Oscean data type of the field.
    // std::string originalTypeName;            ///< Optional: Original type name from source (e.g., "Integer", "Date").
    // int sourceIndex = -1;                    ///< Optional: Index of the field in the source data if relevant.
    // std::map<std::string, AttributeValue> attributes; ///< Optional: Additional metadata for this field.

    FieldDefinition(std::string n = "", DataType dt = DataType::Unknown) 
        : name(std::move(n)), type(dt) {}
};

/**
 * @struct Feature
 * @brief Represents a single vector feature, combining geometry with attributes.
 * Corresponds to a single record in a Shapefile or a feature in GeoJSON.
 */
struct Feature {
    std::optional<std::string> id;                    ///< Optional unique identifier for the feature.
    Geometry geometry;                                ///< The geometry of the feature.
    std::map<std::string, AttributeValue> attributes; ///< Key-value map of feature attributes.
                                                      ///< Keys are field/property names.
};

/**
 * @struct FeatureCollection
 * @brief Represents a collection of vector features, typically from a single layer or file.
 */
struct FeatureCollection {
    std::string name;                                 ///< Name of the feature collection (e.g., layer name).
    CRSInfo crs;                                      ///< The primary Coordinate Reference System for all features in this collection.
                                                      ///< Individual features can override this if their Geometry::crs is set.
    std::optional<BoundingBox> extent;                ///< Overall bounding box of all features in this collection, in the collection's CRS.
    std::vector<Feature> features;                    ///< List of features.

    // Optional: Metadata about the fields/attributes present in the features.
    // Key: field name, Value: data type of the field.
    std::vector<FieldDefinition> fieldDefinitions;    ///< Definitions of the fields (attributes) for the features in this collection.

    // Optional: Global attributes or metadata for the entire collection.
    std::map<std::string, AttributeValue> globalAttributes;
};


} // namespace core_services
} // namespace oscean

#endif // OSCEAN_CORE_SERVICES_COMMON_DATA_TYPES_H
