/**
 * @file spatial_ops_service_impl.h
 * @brief Final, correct declaration of the SpatialOpsServiceImpl class.
 */
#pragma once

#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/exceptions.h"
#include "common_utils/utilities/logging_utils.h"
#include "engine/raster_engine.h"
#include <memory>

// Forward declarations
namespace oscean::core_services::spatial_ops::engine {
    class RasterEngine;
}

namespace oscean::core_services::spatial_ops::impl {

class SpatialOpsServiceImpl : public ISpatialOpsService {
public:
    explicit SpatialOpsServiceImpl(const SpatialOpsConfig& config);
    ~SpatialOpsServiceImpl() override;

    // --- Service Management & Configuration ---
    boost::future<void> setConfiguration(const SpatialOpsConfig& config) override;
    boost::future<SpatialOpsConfig> getConfiguration() const override;
    boost::future<std::vector<std::string>> getCapabilities() const override;
    std::string getVersion() const override;
    bool isReady() const override;

    // --- Basic Geometry Operations ---
    boost::future<Geometry> buffer(const Geometry& geom, double distance, const BufferOptions& options = {}) const override;
    boost::future<Geometry> intersection(const Geometry& geom1, const Geometry& geom2) const override;
    boost::future<Geometry> difference(const Geometry& geom1, const Geometry& geom2) const override;
    boost::future<Geometry> unionGeometries(const Geometry& geom1, const Geometry& geom2) const override;
    boost::future<Geometry> convexHull(const Geometry& geom) const override;
    boost::future<Geometry> simplify(const Geometry& geom, double tolerance) const override;
    boost::future<BoundingBox> getBoundingBoxForGeometry(const Geometry& geom) const override;
    boost::future<double> calculateDistance(const Geometry& geom1, const Geometry& geom2, DistanceType type = DistanceType::EUCLIDEAN) const override;

    // --- Spatial Predicates & Queries ---
    boost::future<bool> evaluatePredicate(SpatialPredicate predicate, const Geometry& geom1, const Geometry& geom2) const override;
    boost::future<FeatureCollection> queryByBoundingBox(const FeatureCollection& features, const BoundingBox& bbox) const override;
    boost::future<FeatureCollection> queryByGeometry(const FeatureCollection& features, const Geometry& queryGeom, SpatialPredicate predicate) const override;
    boost::future<std::optional<GridIndex>> findGridCell(const Point& point, const GridDefinition& gridDef) const override;
    boost::future<std::optional<Feature>> findNearestNeighbor(const Point& point, const FeatureCollection& candidates) const override;
    boost::future<Point> calculateDestinationPointAsync(const Point& startPoint, double bearing, double distance) const override;

    // --- Raster Operations ---
    boost::future<std::shared_ptr<GridData>> clipRaster(std::shared_ptr<GridData> source, const Geometry& clipGeom, const RasterClipOptions& options = {}) const override;
    boost::future<std::shared_ptr<GridData>> clipRasterByBoundingBox(const GridData& raster, const BoundingBox& bbox) const override;
    boost::future<std::shared_ptr<GridData>> rasterizeFeatures(const FeatureCollection& features, const GridDefinition& targetGridDef, const RasterizeOptions& options = {}) const override;
    boost::future<std::shared_ptr<GridData>> applyRasterMask(const GridData& raster, const GridData& maskRaster, const MaskOptions& options = {}) const override;
    boost::future<std::shared_ptr<GridData>> mosaicRastersAsync(const std::vector<std::shared_ptr<const GridData>>& sources, const MosaicOptions& options = {}) const override;
    boost::future<FeatureCollection> generateContours(const GridData& raster, const ContourOptions& options) const override;

    // --- Advanced Analysis ---
    boost::future<StatisticsResult> calculateStatistics(std::shared_ptr<GridData> source, const StatisticsOptions& options = {}) const override;
    boost::future<std::map<std::string, StatisticsResult>> calculateZonalStatistics(const GridData& valueRaster, const FeatureCollection& zoneFeatures, const std::string& zoneIdField, const ZonalStatisticsOptions& options = {}) const override;
    boost::future<std::map<int, StatisticsResult>> calculateZonalStatistics(const GridData& valueRaster, const GridData& zoneRaster, const ZonalStatisticsOptions& options = {}) const override;
    boost::future<std::shared_ptr<GridData>> rasterAlgebra(const std::vector<std::shared_ptr<GridData>>& sources, const std::string& expression, const RasterAlgebraOptions& options = {}) const override;
    boost::future<std::vector<std::optional<GridIndex>>> findGridCellsForPointsAsync(const std::vector<Point>& points, const GridDefinition& gridDef) const override;
    boost::future<std::map<std::string, StatisticsResult>> zonalStatistics(const GridData& valueRaster, const GridData& zoneRaster, const ZonalStatisticsOptions& options = {}) const override;
    boost::future<std::shared_ptr<GridData>> performRasterAlgebra(const std::vector<std::shared_ptr<GridData>>& sources, const std::string& expression, const RasterAlgebraOptions& options = {}) const override;
    boost::future<std::shared_ptr<GridData>> createGridFromPoints(const std::vector<Point>& points, const GridDefinition& gridDef) const override;

private:
    SpatialOpsConfig m_config;
    std::unique_ptr<engine::RasterEngine> m_rasterEngine;
    mutable std::mutex m_configMutex;
};

} // namespace oscean::core_services::spatial_ops::impl 