/**
 * @file spatial_ops_service_impl.cpp
 * @brief Implementation of the SpatialOpsServiceImpl class.
 */

#include "spatial_ops_service_impl.h"
#include "core_services/exceptions.h"
#include "common_utils/utilities/logging_utils.h"

// GDAL/OGR includes
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <ogr_geometry.h>
#include <cpl_conv.h>

#include <stdexcept>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <variant>

namespace oscean::core_services::spatial_ops::impl {

SpatialOpsServiceImpl::SpatialOpsServiceImpl(const SpatialOpsConfig& config)
    : m_config(config)
    , m_rasterEngine(std::make_unique<engine::RasterEngine>(config)) {
    // 移除 GDALAllRegister() 调用。该操作现在由GdalGlobalInitializer在程序启动时统一管理。
}

SpatialOpsServiceImpl::~SpatialOpsServiceImpl() = default;

// --- Service Management & Configuration ---
boost::future<void> SpatialOpsServiceImpl::setConfiguration(const SpatialOpsConfig& config) {
    return boost::async(boost::launch::async, [this, config]() {
        std::lock_guard<std::mutex> lock(m_configMutex);
        m_config = config;
    });
}

boost::future<SpatialOpsConfig> SpatialOpsServiceImpl::getConfiguration() const {
    return boost::async(boost::launch::async, [this]() {
        std::lock_guard<std::mutex> lock(m_configMutex);
        return m_config;
    });
}

boost::future<std::vector<std::string>> SpatialOpsServiceImpl::getCapabilities() const {
    return boost::async(boost::launch::async, []() {
        return std::vector<std::string>{
            "buffer", "intersection", "difference", "union", "convexHull", "simplify",
            "calculateDistance", "getBoundingBox", "evaluatePredicate", "queryByBoundingBox",
            "queryByGeometry", "findGridCell", "findNearestNeighbor", "calculateDestinationPoint",
            "clipRaster", "clipRasterByBoundingBox", "rasterizeFeatures", "applyRasterMask",
            "mosaicRasters", "generateContours", "calculateStatistics", "calculateZonalStatistics",
            "rasterAlgebra", "performRasterAlgebra", "createGridFromPoints"
        };
    });
}

std::string SpatialOpsServiceImpl::getVersion() const {
    return "1.0.0";
}

bool SpatialOpsServiceImpl::isReady() const {
    return m_rasterEngine != nullptr;
}

// --- Basic Geometry Operations ---
boost::future<Geometry> SpatialOpsServiceImpl::buffer(const Geometry& geom, double distance, const BufferOptions& options) const {
    return boost::async(boost::launch::async, [geom, distance, options]() -> Geometry {
        OGRGeometry *poGeom = nullptr;
        OGRGeometryFactory::createFromWkt(geom.wkt.c_str(), nullptr, &poGeom);
        if (!poGeom) throw InvalidInputException("Failed to create geometry from WKT for buffer.");

        OGRGeometry* poBufferedGeom = poGeom->Buffer(distance, options.quadrantSegments);
        OGRGeometryFactory::destroyGeometry(poGeom);
        if (!poBufferedGeom) throw OperationFailedException("Buffer operation failed.");
        
        char* pszWkt = nullptr;
        poBufferedGeom->exportToWkt(&pszWkt);
        std::string bufferedWkt(pszWkt);
        CPLFree(pszWkt);
        OGRGeometryFactory::destroyGeometry(poBufferedGeom);

        Geometry result;
        result.wkt = bufferedWkt;
        return result;
    });
}

boost::future<Geometry> SpatialOpsServiceImpl::intersection(const Geometry& geom1, const Geometry& geom2) const {
    return boost::async(boost::launch::async, [geom1, geom2]() -> Geometry {
        OGRGeometry *poGeom1 = nullptr, *poGeom2 = nullptr;
        OGRGeometryFactory::createFromWkt(geom1.wkt.c_str(), nullptr, &poGeom1);
        OGRGeometryFactory::createFromWkt(geom2.wkt.c_str(), nullptr, &poGeom2);
        if (!poGeom1 || !poGeom2) {
            if (poGeom1) OGRGeometryFactory::destroyGeometry(poGeom1);
            if (poGeom2) OGRGeometryFactory::destroyGeometry(poGeom2);
            throw InvalidInputException("Failed to create one or both geometries from WKT for intersection.");
        }
        OGRGeometry* poIntersection = poGeom1->Intersection(poGeom2);
        OGRGeometryFactory::destroyGeometry(poGeom1);
        OGRGeometryFactory::destroyGeometry(poGeom2);
        if (!poIntersection) throw OperationFailedException("Intersection operation failed.");
        
        char* pszWkt = nullptr;
        poIntersection->exportToWkt(&pszWkt);
        std::string intersectionWkt(pszWkt);
        CPLFree(pszWkt);
        OGRGeometryFactory::destroyGeometry(poIntersection);

        Geometry result;
        result.wkt = intersectionWkt;
        return result;
    });
}

boost::future<Geometry> SpatialOpsServiceImpl::difference(const Geometry& geom1, const Geometry& geom2) const {
    return boost::async(boost::launch::async, [geom1, geom2]() -> Geometry {
        OGRGeometry *poGeom1 = nullptr, *poGeom2 = nullptr;
        OGRGeometryFactory::createFromWkt(geom1.wkt.c_str(), nullptr, &poGeom1);
        OGRGeometryFactory::createFromWkt(geom2.wkt.c_str(), nullptr, &poGeom2);
        if (!poGeom1 || !poGeom2) {
            if (poGeom1) OGRGeometryFactory::destroyGeometry(poGeom1);
            if (poGeom2) OGRGeometryFactory::destroyGeometry(poGeom2);
            throw InvalidInputException("Failed to create one or both geometries from WKT for difference.");
        }
        OGRGeometry* poDifference = poGeom1->Difference(poGeom2);
        OGRGeometryFactory::destroyGeometry(poGeom1);
        OGRGeometryFactory::destroyGeometry(poGeom2);
        if (!poDifference) throw OperationFailedException("Difference operation failed.");
        
        char* pszWkt = nullptr;
        poDifference->exportToWkt(&pszWkt);
        std::string differenceWkt(pszWkt);
        CPLFree(pszWkt);
        OGRGeometryFactory::destroyGeometry(poDifference);

        Geometry result;
        result.wkt = differenceWkt;
        return result;
    });
}

boost::future<Geometry> SpatialOpsServiceImpl::unionGeometries(const Geometry& geom1, const Geometry& geom2) const {
    return boost::async(boost::launch::async, [geom1, geom2]() -> Geometry {
        OGRGeometry *poGeom1 = nullptr, *poGeom2 = nullptr;
        OGRGeometryFactory::createFromWkt(geom1.wkt.c_str(), nullptr, &poGeom1);
        OGRGeometryFactory::createFromWkt(geom2.wkt.c_str(), nullptr, &poGeom2);
        if (!poGeom1 || !poGeom2) {
            if (poGeom1) OGRGeometryFactory::destroyGeometry(poGeom1);
            if (poGeom2) OGRGeometryFactory::destroyGeometry(poGeom2);
            throw InvalidInputException("Failed to create one or both geometries from WKT for union.");
        }
        OGRGeometry* poUnion = poGeom1->Union(poGeom2);
        OGRGeometryFactory::destroyGeometry(poGeom1);
        OGRGeometryFactory::destroyGeometry(poGeom2);
        if (!poUnion) throw OperationFailedException("Union operation failed.");

        char* pszWkt = nullptr;
        poUnion->exportToWkt(&pszWkt);
        std::string unionWkt(pszWkt);
        CPLFree(pszWkt);
        OGRGeometryFactory::destroyGeometry(poUnion);

        Geometry result;
        result.wkt = unionWkt;
        return result;
    });
}

boost::future<Geometry> SpatialOpsServiceImpl::convexHull(const Geometry& geom) const {
    return boost::async(boost::launch::async, [geom]() -> Geometry {
        OGRGeometry *poGeom = nullptr;
        OGRGeometryFactory::createFromWkt(geom.wkt.c_str(), nullptr, &poGeom);
        if (!poGeom) throw InvalidInputException("Failed to create geometry from WKT for convex hull.");

        OGRGeometry* poConvexHull = poGeom->ConvexHull();
        OGRGeometryFactory::destroyGeometry(poGeom);
        if (!poConvexHull) throw OperationFailedException("ConvexHull operation failed.");
        
        char* pszWkt = nullptr;
        poConvexHull->exportToWkt(&pszWkt);
        std::string convexHullWkt(pszWkt);
        CPLFree(pszWkt);
        OGRGeometryFactory::destroyGeometry(poConvexHull);

        Geometry result;
        result.wkt = convexHullWkt;
        return result;
    });
}

boost::future<Geometry> SpatialOpsServiceImpl::simplify(const Geometry& geom, double tolerance) const {
    return boost::async(boost::launch::async, [geom, tolerance]() -> Geometry {
        OGRGeometry *poGeom = nullptr;
        OGRGeometryFactory::createFromWkt(geom.wkt.c_str(), nullptr, &poGeom);
        if (!poGeom) throw InvalidInputException("Failed to create geometry from WKT for simplification.");

        OGRGeometry* poSimplifiedGeom = poGeom->SimplifyPreserveTopology(tolerance);
        OGRGeometryFactory::destroyGeometry(poGeom);
        if (!poSimplifiedGeom) throw OperationFailedException("Simplify operation failed.");
        
        char* pszWkt = nullptr;
        poSimplifiedGeom->exportToWkt(&pszWkt);
        std::string simplifiedWkt(pszWkt);
        CPLFree(pszWkt);
        OGRGeometryFactory::destroyGeometry(poSimplifiedGeom);

        Geometry result;
        result.wkt = simplifiedWkt;
        return result;
    });
}

boost::future<BoundingBox> SpatialOpsServiceImpl::getBoundingBoxForGeometry(const Geometry& geom) const {
    return boost::async(boost::launch::async, [geom]() -> BoundingBox {
        OGRGeometry *poGeom = nullptr;
        if (OGRGeometryFactory::createFromWkt(geom.wkt.c_str(), nullptr, &poGeom) != OGRERR_NONE || !poGeom) {
            throw InvalidInputException("Failed to create geometry from WKT for bounding box.");
        }

        OGREnvelope envelope;
        poGeom->getEnvelope(&envelope);
        OGRGeometryFactory::destroyGeometry(poGeom);

        return BoundingBox{envelope.MinX, envelope.MinY, envelope.MaxX, envelope.MaxY};
    });
}

boost::future<double> SpatialOpsServiceImpl::calculateDistance(const Geometry& geom1, const Geometry& geom2, DistanceType type) const {
    return boost::async(boost::launch::async, [geom1, geom2, type]() -> double {
        OGRGeometry *poGeom1 = nullptr, *poGeom2 = nullptr;
        OGRGeometryFactory::createFromWkt(geom1.wkt.c_str(), nullptr, &poGeom1);
        OGRGeometryFactory::createFromWkt(geom2.wkt.c_str(), nullptr, &poGeom2);
        if (!poGeom1 || !poGeom2) {
            if(poGeom1) OGRGeometryFactory::destroyGeometry(poGeom1);
            if(poGeom2) OGRGeometryFactory::destroyGeometry(poGeom2);
            throw InvalidInputException("Failed to parse one or both geometries for distance calculation.");
        }
        double distance = poGeom1->Distance(poGeom2);
        OGRGeometryFactory::destroyGeometry(poGeom1);
        OGRGeometryFactory::destroyGeometry(poGeom2);
        return distance;
    });
}

// --- Spatial Predicates & Queries ---
boost::future<bool> SpatialOpsServiceImpl::evaluatePredicate(SpatialPredicate predicate, const Geometry& geom1, const Geometry& geom2) const {
    return boost::async(boost::launch::async, [predicate, geom1, geom2]() -> bool {
        // Stub implementation
        return false;
    });
}

boost::future<FeatureCollection> SpatialOpsServiceImpl::queryByBoundingBox(const FeatureCollection& features, const BoundingBox& bbox) const {
    return boost::async(boost::launch::async, [features, bbox]() -> FeatureCollection {
        // Stub implementation
        return FeatureCollection{};
    });
}

boost::future<FeatureCollection> SpatialOpsServiceImpl::queryByGeometry(const FeatureCollection& features, const Geometry& queryGeom, SpatialPredicate predicate) const {
    return boost::async(boost::launch::async, [features, queryGeom, predicate]() -> FeatureCollection {
        // Stub implementation
        return FeatureCollection{};
    });
}

boost::future<std::optional<GridIndex>> SpatialOpsServiceImpl::findGridCell(const Point& point, const GridDefinition& gridDef) const {
    return boost::async(boost::launch::async, [point, gridDef]() -> std::optional<GridIndex> {
        // Stub implementation
        return std::nullopt;
    });
}

boost::future<std::optional<Feature>> SpatialOpsServiceImpl::findNearestNeighbor(const Point& point, const FeatureCollection& candidates) const {
    return boost::async(boost::launch::async, [point, candidates]() -> std::optional<Feature> {
        // Stub implementation
        return std::nullopt;
    });
}

boost::future<Point> SpatialOpsServiceImpl::calculateDestinationPointAsync(const Point& startPoint, double bearing, double distance) const {
    return boost::async(boost::launch::async, [startPoint, bearing, distance]() -> Point {
        // 简化的目标点计算（实际应该使用大地测量学计算）
        double radianBearing = bearing * M_PI / 180.0;
        double deltaX = distance * sin(radianBearing);
        double deltaY = distance * cos(radianBearing);
        
        // 修复：使用正确的Point构造函数
        return Point(startPoint.x + deltaX, startPoint.y + deltaY, startPoint.z, startPoint.crsId);
    });
}

// --- Raster Operations ---
boost::future<std::shared_ptr<GridData>> SpatialOpsServiceImpl::clipRaster(std::shared_ptr<GridData> source, const Geometry& clipGeom, const RasterClipOptions& options) const {
    return boost::async(boost::launch::async, [this, source, clipGeom, options]() -> std::shared_ptr<GridData> {
        if (!source) throw InvalidInputException("Source grid data is null");
        
        MaskOptions maskOptions;
        maskOptions.outputNoDataValue = options.noDataValue;
        
        auto resultGrid = m_rasterEngine->clipRasterByGeometry(*source, clipGeom, maskOptions);
        return std::make_shared<GridData>(std::move(resultGrid));
    });
}

boost::future<std::shared_ptr<GridData>> SpatialOpsServiceImpl::clipRasterByBoundingBox(const GridData& raster, const BoundingBox& bbox) const {
    return boost::async(boost::launch::async, [this, &raster, bbox]() -> std::shared_ptr<GridData> {
        auto resultGrid = m_rasterEngine->clipRasterByBoundingBox(raster, bbox, -9999.0);
        return std::make_shared<GridData>(std::move(resultGrid));
    });
}

boost::future<std::shared_ptr<GridData>> SpatialOpsServiceImpl::rasterizeFeatures(const FeatureCollection& features, const GridDefinition& targetGridDef, const RasterizeOptions& options) const {
    return boost::async(boost::launch::async, [features, targetGridDef, options]() -> std::shared_ptr<GridData> {
        // Stub implementation
        return std::make_shared<GridData>();
    });
}

boost::future<std::shared_ptr<GridData>> SpatialOpsServiceImpl::applyRasterMask(const GridData& raster, const GridData& maskRaster, const MaskOptions& options) const {
    return boost::async(boost::launch::async, [&raster, &maskRaster, options]() -> std::shared_ptr<GridData> {
        // Stub implementation
        return std::make_shared<GridData>();
    });
}

boost::future<std::shared_ptr<GridData>> SpatialOpsServiceImpl::mosaicRastersAsync(const std::vector<std::shared_ptr<const GridData>>& sources, const MosaicOptions& options) const {
    return boost::async(boost::launch::async, [sources, options]() -> std::shared_ptr<GridData> {
        // Stub implementation
        return std::make_shared<GridData>();
    });
}

boost::future<FeatureCollection> SpatialOpsServiceImpl::generateContours(const GridData& raster, const ContourOptions& options) const {
    return boost::async(boost::launch::async, [&raster, options]() -> FeatureCollection {
        // Stub implementation
        return FeatureCollection{};
    });
}

// --- Advanced Analysis ---
boost::future<StatisticsResult> SpatialOpsServiceImpl::calculateStatistics(std::shared_ptr<GridData> source, const StatisticsOptions& options) const {
    return boost::async(boost::launch::async, [this, source, options]() -> StatisticsResult {
        // 修复：直接实现统计计算，不调用不存在的RasterEngine方法
        StatisticsResult result;
        
        if (!source || source->getData().empty()) {
            throw InvalidInputException("Input raster is null or empty");
        }
        
        // 简化的统计计算实现
        const auto& data = source->getData();
        size_t validCount = 0;
        double sum = 0.0;
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::lowest();
        
        // 根据数据类型处理数据
        for (size_t i = 0; i < data.size(); ++i) {
            double value = static_cast<double>(data[i]);
            if (source->hasNoDataValue() && value == source->getFillValue().value_or(-9999.0)) {
                continue; // 跳过NoData值
            }
            
            validCount++;
            sum += value;
            min = std::min(min, value);
            max = std::max(max, value);
        }
        
        if (validCount > 0) {
            double mean = sum / validCount;
            
            // 使用正确的StatisticsResult结构体格式
            result.values[StatisticalMeasure::COUNT] = static_cast<double>(validCount);
            result.values[StatisticalMeasure::SUM] = sum;
            result.values[StatisticalMeasure::MEAN] = mean;
            result.values[StatisticalMeasure::MIN] = min;
            result.values[StatisticalMeasure::MAX] = max;
            result.values[StatisticalMeasure::RANGE] = max - min;
            
            // 计算标准差
            double sumSquaredDiff = 0.0;
            for (size_t i = 0; i < data.size(); ++i) {
                double value = static_cast<double>(data[i]);
                if (source->hasNoDataValue() && value == source->getFillValue().value_or(-9999.0)) {
                    continue;
                }
                double diff = value - mean;
                sumSquaredDiff += diff * diff;
            }
            result.values[StatisticalMeasure::STDDEV] = std::sqrt(sumSquaredDiff / validCount);
        }
        
        return result;
    });
}

boost::future<std::map<std::string, StatisticsResult>> SpatialOpsServiceImpl::calculateZonalStatistics(const GridData& valueRaster, const FeatureCollection& zoneFeatures, const std::string& zoneIdField, const ZonalStatisticsOptions& options) const {
    return boost::async(boost::launch::async, [this, &valueRaster, &zoneFeatures, &zoneIdField, &options]() -> std::map<std::string, StatisticsResult> {
        // 修复：返回正确的类型 std::map<std::string, StatisticsResult>
        std::map<std::string, StatisticsResult> results;
        
        // 调用RasterEngine的方法获取FeatureId映射的结果
        auto featureIdResults = m_rasterEngine->calculateZonalStatistics(valueRaster, zoneFeatures, options);
        
        // 转换FeatureId到string
        for (const auto& [featureId, stats] : featureIdResults) {
            std::string stringId;
            if (std::holds_alternative<std::string>(featureId)) {
                stringId = std::get<std::string>(featureId);
            } else if (std::holds_alternative<int>(featureId)) {
                stringId = std::to_string(std::get<int>(featureId));
            }
            results[stringId] = stats;
        }
        
        return results;
    });
}

boost::future<std::map<int, StatisticsResult>> SpatialOpsServiceImpl::calculateZonalStatistics(const GridData& valueRaster, const GridData& zoneRaster, const ZonalStatisticsOptions& options) const {
    return boost::async(boost::launch::async, [this, &valueRaster, &zoneRaster, options]() -> std::map<int, StatisticsResult> {
        return m_rasterEngine->calculateZonalStatistics(valueRaster, zoneRaster, options);
    });
}

boost::future<std::shared_ptr<GridData>> SpatialOpsServiceImpl::rasterAlgebra(const std::vector<std::shared_ptr<GridData>>& sources, const std::string& expression, const RasterAlgebraOptions& options) const {
    return boost::async(boost::launch::async, [sources, expression, options]() -> std::shared_ptr<GridData> {
        // Stub implementation
        return std::make_shared<GridData>();
    });
}

boost::future<std::vector<std::optional<GridIndex>>> SpatialOpsServiceImpl::findGridCellsForPointsAsync(const std::vector<Point>& points, const GridDefinition& gridDef) const {
    return boost::async(boost::launch::async, [points, gridDef]() -> std::vector<std::optional<GridIndex>> {
        // Stub implementation
        return std::vector<std::optional<GridIndex>>(points.size(), std::nullopt);
    });
}

boost::future<std::map<std::string, StatisticsResult>> SpatialOpsServiceImpl::zonalStatistics(const GridData& valueRaster, const GridData& zoneRaster, const ZonalStatisticsOptions& options) const {
    return boost::async(boost::launch::async, [this, &valueRaster, &zoneRaster, options]() -> std::map<std::string, StatisticsResult> {
        // Convert int-based result to string-based result
        auto intResult = m_rasterEngine->calculateZonalStatistics(valueRaster, zoneRaster, options);
        std::map<std::string, StatisticsResult> stringResult;
        for (const auto& [key, value] : intResult) {
            stringResult[std::to_string(key)] = value;
        }
        return stringResult;
    });
}

boost::future<std::shared_ptr<GridData>> SpatialOpsServiceImpl::performRasterAlgebra(const std::vector<std::shared_ptr<GridData>>& sources, const std::string& expression, const RasterAlgebraOptions& options) const {
    return boost::async(boost::launch::async, [sources, expression, options]() -> std::shared_ptr<GridData> {
        // Stub implementation
        return std::make_shared<GridData>();
    });
}

boost::future<std::shared_ptr<GridData>> SpatialOpsServiceImpl::createGridFromPoints(const std::vector<Point>& points, const GridDefinition& gridDef) const {
    return boost::async(boost::launch::async, [points, gridDef]() -> std::shared_ptr<GridData> {
        // Stub implementation
        return std::make_shared<GridData>();
    });
}

} // namespace oscean::core_services::spatial_ops::impl 