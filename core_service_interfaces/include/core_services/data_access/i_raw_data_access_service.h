/**
 * @file i_raw_data_access_service.h
 * @brief 原始数据访问服务接口
 */

#pragma once

#include "core_services/common_data_types.h"
#include "core_services/exceptions.h"
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 数据访问服务不使用boost::asio，只使用boost::future

#include <boost/thread/future.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <variant>

namespace oscean::core_services {

// NO FORWARD DECLARATIONS HERE - They are in common_data_types.h

/**
 * @brief 变量数据变体类型，用于存储不同类型的变量值
 */
using VariableDataVariant = std::variant<
    std::monostate, 
    std::vector<unsigned char>, 
    std::vector<int>, 
    std::vector<double>, 
    std::vector<float>,
    std::string,
    std::vector<std::string>
>;

/**
 * @interface IRawDataAccessService
 * @brief 原始数据访问服务接口
 */
class IRawDataAccessService {
public:
    virtual ~IRawDataAccessService() = default;
    
    /**
     * @brief 获取服务版本
     * @return 服务版本字符串
     */
    virtual std::string getVersion() const = 0;
    
    /**
     * @brief Reads a subset of a grid variable from a raster file (e.g., NetCDF, GeoTIFF).
     *
     * @param filePath Path to the raster file.
     * @param variableName Name of the variable to read.
     * @param timeRange Optional time range (e.g., indices or timestamps) to select.
     * @param spatialExtent Optional bounding box in the *native* CRS of the file to select.
     *                    If provided, the service should calculate corresponding pixel/row/col ranges.
     * @param levelRange Optional vertical level range (indices) if applicable.
     * @return A future containing the requested GridData subset.
     *         The future might contain an exception if the file/variable is not found, 
     *         or if the requested subset cannot be read.
     */
    virtual boost::future<GridData> readGridVariableSubsetAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::optional<IndexRange>& timeRange = std::nullopt, // Or TimePeriod struct?
        const std::optional<BoundingBox>& spatialExtent = std::nullopt, // Native CRS
        const std::optional<IndexRange>& levelRange = std::nullopt
    ) = 0;
    
    /**
     * @brief Reads vector features from a file (e.g., Shapefile, GeoJSON, GeoPackage).
     *
     * @param filePath Path to the vector file.
     * @param layerName Optional name or index of the layer to read. If empty, reads the first layer.
     * @param spatialFilter Optional spatial filter (BoundingBox) in a *specified* CRS.
     *                     The service needs to handle potential CRS transformation for filtering.
     * @param attributeFilter Optional attribute filter.
     * @param targetCRS Optional: The desired CRS for the output features' geometries.
     *                  If provided, the service should transform geometries before returning.
     *                  If not provided, geometries are returned in their native CRS.
     * @return A future containing a vector of Feature objects matching the criteria.
     *         The future might contain an exception on read or filter failure.
     */
    virtual boost::future<std::vector<Feature>> readFeaturesAsync(
        const std::string& filePath,
        const std::string& layerName = "",
        const std::optional<BoundingBox>& spatialFilter = std::nullopt, // Specify CRS in BBox
        const std::optional<AttributeFilter>& attributeFilter = std::nullopt,
        const std::optional<CRSInfo>& targetCRS = std::nullopt
    ) = 0;
    
    /**
     * @brief Asynchronously reads time series data for a specific variable at a given point.
     * 
     * The service implementation should handle finding the nearest grid cell 
     * or interpolating between cells based on the specified method.
     * 
     * @param filePath Path to the data file.
     * @param varName Name of the variable.
     * @param point The geographic point (including CRS) for which to retrieve the time series.
     * @param method Interpolation method (e.g., "nearest", "bilinear"). Defaults to nearest neighbour.
     * @return A future containing the TimeSeriesData.
     */
    virtual boost::future<TimeSeriesData> readTimeSeriesAtPointAsync(
        const std::string& filePath,
        const std::string& varName,
        const core_services::Point& point,
        const std::string& method = "nearest") = 0;
    
    /**
     * @brief Asynchronously reads vertical profile data for a specific variable at a given point and time.
     *
     * The service implementation should handle finding the nearest grid cell and time step 
     * or interpolating based on the specified method.
     *
     * @param filePath Path to the data file.
     * @param varName Name of the variable.
     * @param point The geographic point (including CRS).
     * @param time The specific timestamp for the profile.
     * @param spatialMethod Spatial interpolation method (e.g., "nearest", "bilinear").
     * @param timeMethod Temporal interpolation method (e.g., "nearest", "linear").
     * @return A future containing the VerticalProfileData.
     */
    virtual boost::future<VerticalProfileData> readVerticalProfileAsync(
        const std::string& filePath,
        const std::string& varName,
        const core_services::Point& point,
        const core_services::Timestamp& time,
        const std::string& spatialMethod = "nearest",
        const std::string& timeMethod = "nearest") = 0;
    
    /**
     * @brief 异步检查文件是否存在并包含指定变量
     * @return boost::future<bool> 异步检查结果
     */
    virtual boost::future<bool> checkVariableExistsAsync(
        const std::string& filePath,
        const std::string& varName) = 0;
    
    /**
     * @brief 获取文件支持的格式列表
     * @return std::vector<std::string> 支持的格式列表
     */
    virtual std::vector<std::string> getSupportedFormats() const = 0;

    /**
     * @brief Extracts metadata summary for a given file.
     * 
     * This method is optimized for quickly extracting essential metadata 
     * needed for indexing, without reading large amounts of actual data.
     * For NetCDF files, it should extract BBOX, CRS, time information (prioritizing 
     * file internal metadata like time variable and units, potentially falling 
     * back to filename conventions if internal info is missing), and variable names.
     *
     * @param filePath Path to the file.
     * @param targetCrs Optional target CRS for coordinate transformation.
     * @return A future containing an optional FileMetadata object. 
     *         Returns std::nullopt within the future if the file cannot be read, 
     *         is not a supported format for metadata extraction, or if essential 
     *         metadata (like BBOX or CRS) cannot be determined.
     *         The future itself might contain an exception for severe errors.
     *         Note: The returned FileMetadata might be partially filled (e.g., 
     *         filePath, fileName, format, CRS, spatialCoverage, timeRange, variables), 
     *         but fields like `metadata` (key-value map) might be empty unless 
     *         easily obtainable during the summary extraction.
     */
    virtual boost::future<std::optional<FileMetadata>> extractFileMetadataAsync(
        const std::string& filePath,
        const std::optional<CRSInfo>& targetCrs = std::nullopt
    ) = 0;

    /**
     * @brief Asynchronously retrieves grid data with the specified parameters.
     * 
     * @param filePath Path to the data file.
     * @param variableName Name of the variable to read.
     * @param sliceRanges Ranges for each dimension to slice.
     * @param targetResolution Optional target resolution for resampling.
     * @param targetCRS Optional target CRS for reprojection.
     * @param resampleAlgo Algorithm to use for resampling.
     * @param outputBounds Optional bounds for the output grid.
     * @return A future containing the GridData.
     */
    virtual boost::future<std::shared_ptr<GridData>> getGridDataAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::vector<IndexRange>& sliceRanges,
        const std::optional<std::vector<double>>& targetResolution,
        const std::optional<CRSInfo>& targetCRS,
        ResampleAlgorithm resampleAlgo,
        const std::optional<BoundingBox>& outputBounds) = 0;

    /**
     * @brief Asynchronously retrieves feature data from a vector file.
     * 
     * @param filePath Path to the vector file.
     * @param layerName Name of the layer to read.
     * @param targetCRS Optional target CRS for reprojection.
     * @param filterBoundingBox Optional bounding box for spatial filtering.
     * @param bboxCRS Optional CRS of the filter bounding box.
     * @return A future containing the FeatureCollection.
     */
    virtual boost::future<FeatureCollection> getFeatureDataAsync(
        const std::string& filePath,
        const std::string& layerName,
        const std::optional<CRSInfo>& targetCRS,
        const std::optional<BoundingBox>& filterBoundingBox,
        const std::optional<CRSInfo>& bboxCRS) = 0;

    /**
     * @brief Asynchronously retrieves metadata entries for a dataset.
     * 
     * @param filePath Path to the data file.
     * @param domain Optional metadata domain.
     * @return A future containing the metadata entries.
     */
    virtual boost::future<std::vector<MetadataEntry>> getDatasetMetadataAsync(
        const std::string& filePath,
        const std::optional<std::string>& domain) = 0;

    /**
     * @brief Asynchronously retrieves metadata entries for a layer.
     * 
     * @param filePath Path to the data file.
     * @param layerName Name of the layer.
     * @param domain Optional metadata domain.
     * @return A future containing the metadata entries.
     */
    virtual boost::future<std::vector<MetadataEntry>> getLayerMetadataAsync(
        const std::string& filePath,
        const std::string& layerName,
        const std::optional<std::string>& domain) = 0;

    /**
     * @brief Asynchronously retrieves field definitions for a layer.
     * 
     * @param filePath Path to the data file.
     * @param layerName Name of the layer.
     * @return A future containing the field definitions.
     */
    virtual boost::future<std::vector<FieldDefinition>> getFieldDefinitionsAsync(
        const std::string& filePath,
        const std::string& layerName) = 0;

    /**
     * @brief Asynchronously retrieves raw file content as bytes.
     * 
     * @param filePath Path to the file.
     * @return A future containing the file content as bytes.
     */
    virtual boost::future<std::vector<unsigned char>> getRawFileContentAsync(
        const std::string& filePath) = 0;

    /**
     * @brief Asynchronously retrieves a variable value as a variant.
     * 
     * @param filePath Path to the data file.
     * @param variableName Name of the variable.
     * @param startIndices Start indices for each dimension.
     * @param counts Count of elements to read for each dimension.
     * @return A future containing the variable value as a variant.
     */
    virtual boost::future<VariableDataVariant> getVariableValueAsync(
        const std::string& filePath,
        const std::string& variableName,
        const std::vector<size_t>& startIndices,
        const std::vector<size_t>& counts) = 0;
};

} // namespace oscean::core_services 