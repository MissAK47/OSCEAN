#pragma once

// 首先包含基础类型定义
#include "core_services/common_data_types.h"

// 然后包含boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();

// 包含boost头文件
#include <boost/thread/future.hpp>

// 包含标准库头文件
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <map>

// 最后包含空间服务特定的类型
#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_config.h"

namespace oscean::core_services::spatial_ops {

// 前向声明基本类型
struct SpatialOpsConfig;

/**
 * @brief Interface for Spatial Operations Service.
 *
 * Provides methods for spatial analysis, geometric operations, and 
 * interactions between different geometry types and raster data.
 * All methods are asynchronous and return OSCEAN_FUTURE.
 */
class ISpatialOpsService {
public:
    virtual ~ISpatialOpsService() = default;
    
    // --- 服务管理与配置 (Service Management & Configuration) ---
    /**
     * @brief 异步设置空间服务的运行配置。
     * @param config 包含并行设置、GDAL优化参数等的配置对象。
     * @return OSCEAN_FUTURE<void> 操作完成的 future。
     * @throw ConfigurationException 如果配置无效 (通过future传递)。
     */
    virtual boost::future<void> setConfiguration(const SpatialOpsConfig& config) = 0;

    /**
     * @brief 异步获取当前的空间服务运行配置。
     * @return OSCEAN_FUTURE<SpatialOpsConfig> 包含当前配置对象的 future。
     */
    virtual boost::future<SpatialOpsConfig> getConfiguration() const = 0;

    /**
     * @brief 异步获取服务支持的功能列表。
     * @return OSCEAN_FUTURE<std::vector<std::string>> 包含支持操作名称的字符串向量的 future。
     */
    virtual boost::future<std::vector<std::string>> getCapabilities() const = 0;

    /**
     * @brief 获取服务版本
     * @return 服务版本字符串
     */
    virtual std::string getVersion() const = 0;
    
    /**
     * @brief 检查服务是否就绪
     * @return 如果服务就绪返回true
     */
    virtual bool isReady() const = 0;

    // --- 基础几何运算 (Basic Geometry Operations) ---
    /**
     * @brief 异步计算输入几何体的缓冲区。
     * @param geom 输入几何体。
     * @param distance 缓冲距离。
     * @param options 缓冲区计算选项。
     * @return OSCEAN_FUTURE<Geometry> 包含计算得到的缓冲区几何体的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<Geometry> buffer(
        const Geometry& geom,
        double distance,
        const BufferOptions& options = {}) const = 0;
    
    /**
     * @brief 异步计算两个几何体的交集。
     * @param geom1 第一个几何体。
     * @param geom2 第二个几何体。
     * @return OSCEAN_FUTURE<Geometry> 包含两个几何体交集部分的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<Geometry> intersection(
        const Geometry& geom1,
        const Geometry& geom2) const = 0;

    /**
     * @brief 异步计算两个几何体的差集 (geom1 - geom2)。
     * @param geom1 第一个几何体。
     * @param geom2 第二个几何体。
     * @return OSCEAN_FUTURE<Geometry> 包含两个几何体差集部分的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<Geometry> difference(
        const Geometry& geom1,
        const Geometry& geom2) const = 0;
    
    /**
     * @brief 异步计算两个几何体的并集。
     * @param geom1 第一个几何体。
     * @param geom2 第二个几何体。
     * @return OSCEAN_FUTURE<Geometry> 包含两个几何体并集的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<Geometry> unionGeometries(
        const Geometry& geom1,
        const Geometry& geom2) const = 0;

    /**
     * @brief 异步计算几何体的凸包。
     * @param geom 输入几何体。
     * @return OSCEAN_FUTURE<Geometry> 包含几何体凸包的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<Geometry> convexHull(
        const Geometry& geom) const = 0;

    /**
     * @brief 异步简化几何体。
     * @param geom 输入几何体。
     * @param tolerance 简化容差。
     * @return OSCEAN_FUTURE<Geometry> 包含简化后几何体的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<Geometry> simplify(
        const Geometry& geom,
        double tolerance) const = 0;

    /**
     * @brief 异步获取任意几何体的边界框。
     * @param geom 输入几何体。
     * @return OSCEAN_FUTURE<BoundingBox> 包含几何体边界框的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     */
    virtual boost::future<BoundingBox> getBoundingBoxForGeometry(
        const Geometry& geom) const = 0;

    /**
     * @brief 异步计算两个几何体之间的距离。
     * @param geom1 第一个几何体。
     * @param geom2 第二个几何体。
     * @param type 距离计算类型 (例如：欧几里得、大地测量)。
     * @return OSCEAN_FUTURE<double> 包含计算得到的距离的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<double> calculateDistance(
        const Geometry& geom1,
        const Geometry& geom2,
        DistanceType type = DistanceType::EUCLIDEAN) const = 0;

    // --- 空间关系与查询 (Spatial Predicates & Queries) ---
    /**
     * @brief 异步评估两个几何体之间的空间关系。
     * @param geom1 第一个几何体。
     * @param geom2 第二个几何体。
     * @param predicate 要评估的空间谓词（例如 INTERSECTS, CONTAINS）。
     * @return OSCEAN_FUTURE<bool> 如果空间关系为真，则返回 true 的 future，否则返回 false 的 future。
     * @throw InvalidInputGeometryException 如果输入几何无效 (通过future传递)。
     */
    virtual boost::future<bool> evaluatePredicate(
        SpatialPredicate predicate,
        const Geometry& geom1,
        const Geometry& geom2) const = 0;
    
    /**
     * @brief 异步根据边界框查询要素集合。
     * @param features 要查询的要素集合。
     * @param bbox 查询边界框。
     * @return OSCEAN_FUTURE<FeatureCollection> 包含与边界框相交要素子集的 future。
     */
    virtual boost::future<FeatureCollection> queryByBoundingBox(
        const FeatureCollection& features,
        const BoundingBox& bbox) const = 0;

    /**
     * @brief 异步根据查询几何体和空间谓词查询要素集合。
     * @param features 要查询的要素集合。
     * @param queryGeom 用于查询的几何体。
     * @param predicate 应用于 queryGeom 和每个要素几何体的空间谓词。
     * @return OSCEAN_FUTURE<FeatureCollection> 包含满足条件要素子集的 future。
     */
    virtual boost::future<FeatureCollection> queryByGeometry(
        const FeatureCollection& features,
        const Geometry& queryGeom,
        SpatialPredicate predicate) const = 0;
    
    /**
     * @brief Finds the grid cell index for a single point.
     * @param point The point (must have CRS defined).
     * @param gridDef Definition of the target grid (must have CRS defined).
     * @return OSCEAN_FUTURE<std::optional<GridIndex>> A future containing an optional GridIndex.
     *         std::nullopt indicates the point is outside the grid extent or transformation failed.
     */
    virtual boost::future<std::optional<GridIndex>> findGridCell(
        const Point& point,
        const GridDefinition& gridDef) const = 0;

    /**
     * @brief 异步在候选要素集合中查找距离给定点最近的要素。
     * @param point 参考点。
     * @param candidates 候选要素集合。
     * @return OSCEAN_FUTURE<Feature> 包含最近要素的 future。如果候选集为空，可能返回包含无效要素的 future。
     */
    virtual boost::future<std::optional<Feature>> findNearestNeighbor(
        const Point& point,
        const FeatureCollection& candidates) const = 0;

    /**
     * @brief 异步根据起点、方位角和距离计算目标点（大地测量计算）。
     * @param startPoint 起始点（必须是WGS84地理坐标）。
     * @param bearing 方位角，从正北方向顺时针测量（0-360度）。
     * @param distance 距离（米）。
     * @return OSCEAN_FUTURE<Point> 包含计算得到的目标点（WGS84）的 future。
     * @throw InvalidInputException 如果输入参数无效 (通过future传递)。
     */
    virtual boost::future<Point> calculateDestinationPointAsync(
        const Point& startPoint,
        double bearing,
        double distance) const = 0;

    // --- 栅格操作 (Raster Operations) ---
    /**
     * @brief 异步裁剪栅格数据。
     * @param source 源栅格数据。
     * @param clipGeom 用于裁剪的几何体。
     * @param options 裁剪选项。
     * @return OSCEAN_FUTURE<std::shared_ptr<GridData>> 包含裁剪后栅格数据的 future。
     */
    virtual boost::future<std::shared_ptr<GridData>> clipRaster(
        std::shared_ptr<GridData> source,
        const Geometry& clipGeom,
        const RasterClipOptions& options = {}) const = 0;
    
    /**
     * @brief 异步根据边界框裁剪栅格数据。
     * @param raster 源栅格数据。
     * @param bbox 裁剪边界框。
     * @return OSCEAN_FUTURE<std::shared_ptr<GridData>> 包含裁剪后栅格数据的 future。
     */
    virtual boost::future<std::shared_ptr<GridData>> clipRasterByBoundingBox(
        const GridData& raster,
        const BoundingBox& bbox) const = 0;

    /**
     * @brief 异步将矢量要素栅格化为栅格数据。
     * @param features 要栅格化的矢量要素。
     * @param targetGridDef 目标栅格的定义。
     * @param options （可选）栅格化选项，包括属性字段、烧录值、输出数据类型等。
     * @return OSCEAN_FUTURE<GridData> 包含栅格化后栅格数据的 future。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<std::shared_ptr<GridData>> rasterizeFeatures(
        const FeatureCollection& features,
        const GridDefinition& targetGridDef,
        const RasterizeOptions& options = {}) const = 0;

    /**
     * @brief 异步应用一个掩膜栅格到另一个栅格数据。
     * @param raster 要被掩膜的栅格数据。
     * @param maskRaster 掩膜栅格。
     * @param options 掩膜选项，例如是否反转掩膜。
     * @return OSCEAN_FUTURE<GridData> 包含应用掩膜后栅格数据的 future。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<std::shared_ptr<GridData>> applyRasterMask(
        const GridData& raster,
        const GridData& maskRaster,
        const MaskOptions& options = {}) const = 0;
    
    /**
     * @brief 🆕 异步将多个源栅格融合成一个目标栅格。
     * @details 此函数用于将多个栅格数据拼接成一个单一的、无缝的数据集。
     * @param sources 要融合的源栅格数据列表。
     * @param options 融合选项，例如重叠区域的处理方法、输出分辨率等。
     * @return boost::future<std::shared_ptr<GridData>> 包含融合后栅格数据的 future。
     * @throw OperationFailedException 如果融合失败。
     */
    virtual boost::future<std::shared_ptr<GridData>> mosaicRastersAsync(
        const std::vector<std::shared_ptr<const GridData>>& sources,
        const MosaicOptions& options = {}) const = 0;

    // --- 矢量化与栅格化 (Vectorization & Rasterization) ---
    /**
     * @brief 异步根据栅格数据生成等值线。
     * @param raster 输入栅格数据。
     * @param options 等值线生成选项 (例如，等值线间隔、层级列表)。
     * @return OSCEAN_FUTURE<FeatureCollection> 包含表示等值线的要素集合 (MultiLineString Features) 的 future。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<FeatureCollection> generateContours(
        const GridData& raster,
        const ContourOptions& options) const = 0;

    // --- 高级分析 (Advanced Analysis) ---
    /**
     * @brief 异步计算区域统计。
     * @param valueRaster 包含值的栅格数据。
     * @param zoneGeometry 定义区域的几何体。
     * @param options 区域统计选项 (例如，要计算的统计量列表)。
     * @return OSCEAN_FUTURE<StatisticsResult> 包含每个区域统计结果的 future。
     * @throw OperationFailedException 如果操作失败 (通过future传递)。
     */
    virtual boost::future<StatisticsResult> calculateStatistics(
        std::shared_ptr<GridData> source,
        const StatisticsOptions& options = {}) const = 0;

    // Overload for zonal statistics using FeatureCollection for zones
    virtual boost::future<std::map<std::string, StatisticsResult>> calculateZonalStatistics(
        const GridData& valueRaster,
        const FeatureCollection& zoneFeatures,
        const std::string& zoneIdField,
        const ZonalStatisticsOptions& options = {}) const = 0;

    // Overload for zonal statistics using a raster for zones
    virtual boost::future<std::map<int, StatisticsResult>> calculateZonalStatistics(
        const GridData& valueRaster,
        const GridData& zoneRaster,
        const ZonalStatisticsOptions& options = {}) const = 0;
    
    /**
     * @brief 异步执行栅格代数运算。
     * @param expression 代数表达式字符串 (例如 "raster1 * 2 + raster2")。
     * @param namedRasters 一个从名称到栅格数据的映射，用于表达式中的变量。
     * @return OSCEAN_FUTURE<GridData> 包含运算结果栅格数据的 future。
     * @throw OperationFailedException 如果表达式无效或运算失败 (通过future传递)。
     * @throw RasterAlgebraSyntaxException 如果表达式语法错误。
     */
    virtual boost::future<std::shared_ptr<GridData>> rasterAlgebra(
        const std::vector<std::shared_ptr<GridData>>& sources,
        const std::string& expression,
        const RasterAlgebraOptions& options = {}) const = 0;

    /**
     * @brief Finds the grid cell indices corresponding to a list of points.
     * @param points Vector of points (must have CRS defined).
     * @param gridDef Definition of the target grid (must have CRS defined).
     * @return OSCEAN_FUTURE<std::vector<std::optional<GridIndex>>> A future containing a vector of optional GridIndex objects.
     */
    virtual boost::future<std::vector<std::optional<GridIndex>>> findGridCellsForPointsAsync(
        const std::vector<Point>& points,
        const GridDefinition& gridDef
    ) const = 0;

    virtual boost::future<std::map<std::string, StatisticsResult>> zonalStatistics(
        const GridData& valueRaster,
        const GridData& zoneRaster,
        const ZonalStatisticsOptions& options = {}) const = 0;

    /**
     * @brief 异步使用栅格代数（map algebra）处理一个或多个栅格。
     * @param sources 参与计算的源栅格列表。
     * @param expression 定义计算的代数表达式 (例如 "raster1 > 100 ? (raster1 - raster2) : -9999")。
     * @param options 栅格代数计算选项。
     * @return OSCEAN_FUTURE<std::shared_ptr<GridData>> 包含计算结果栅格的 future。
     * @throw InvalidInputException 如果输入参数无效 (通过future传递)。
     * @throw OperationFailedException 如果表达式解析或计算失败 (通过future传递)。
     */
    virtual boost::future<std::shared_ptr<GridData>> performRasterAlgebra(
        const std::vector<std::shared_ptr<GridData>>& sources,
        const std::string& expression,
        const RasterAlgebraOptions& options = {}) const = 0;
        
    // --- 高级分析 (Advanced Analysis) ---
    /**
     * @brief Creates a grid from a set of points using interpolation.
     * @param points The input points.
     * @param gridDef The target grid definition.
     * @return A future containing the resulting grid data.
     */
    virtual boost::future<std::shared_ptr<GridData>> createGridFromPoints(
        const std::vector<Point>& points,
        const GridDefinition& gridDef
    ) const = 0;
};

} // namespace oscean::core_services::spatial_ops 