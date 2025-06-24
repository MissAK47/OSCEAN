#pragma once

#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/common_data_types.h"
#include <string>
#include <vector>
#include <memory>

// Forward declare GEOS types
typedef struct GEOSContextHandle_HS *GEOSContextHandle_t;
typedef struct GEOSGeom_t GEOSGeometry;
typedef struct GEOSWKTReader_t GEOSWKTReader;
typedef struct GEOSWKTWriter_t GEOSWKTWriter;

namespace oscean::core_services::spatial_ops::engine { // Updated namespace

/**
 * @class GeometryEngine
 * @brief Handles various geometry processing tasks using GEOS library.
 *
 * This class encapsulates logic for performing common spatial operations on
 * vector geometries, such as buffering, overlay analysis (intersection, union, difference),
 * simplification, and predicate evaluation. It uses GEOS for robust geometric computations
 * and OGR for coordinate transformations.
 */
class GeometryEngine {
public:
    /**
     * @brief Constructs a GeometryEngine instance.
     * @param config The spatial operations configuration, containing settings like
     *               geometric tolerance.
     */
    explicit GeometryEngine(const oscean::core_services::spatial_ops::SpatialOpsConfig& config); // Added oscean::core_services::spatial_ops namespace

    ~GeometryEngine();

    // Disable copy constructor and assignment operator
    GeometryEngine(const GeometryEngine&) = delete;
    GeometryEngine& operator=(const GeometryEngine&) = delete;

    /**
     * @brief Creates a buffer around a given geometry.
     * @param geom The input geometry.
     * @param distance The buffer distance.
     * @param options Buffer creation options (e.g., quadrant segments).
     * @return The buffered geometry.
     * @throws oscean::core_services::InvalidInputException if input is invalid.
     * @throws oscean::core_services::OperationFailedException if buffering fails.
     */
    oscean::core_services::Geometry buffer(
        const oscean::core_services::Geometry& geom,
        double distance,
        const oscean::core_services::spatial_ops::BufferOptions& options) const;

    /**
     * @brief Computes the intersection of two geometries.
     * @param geom1 The first geometry.
     * @param geom2 The second geometry.
     * @return The intersection geometry. Returns an empty geometry if no intersection.
     * @throws oscean::core_services::InvalidInputException if inputs are invalid.
     * @throws oscean::core_services::OperationFailedException if intersection fails.
     */
    oscean::core_services::Geometry intersection(
        const oscean::core_services::Geometry& geom1,
        const oscean::core_services::Geometry& geom2) const;

    /**
     * @brief Computes the union of two geometries.
     * @param geom1 The first geometry.
     * @param geom2 The second geometry.
     * @return The union geometry.
     * @throws oscean::core_services::InvalidInputException if inputs are invalid.
     * @throws oscean::core_services::OperationFailedException if union calculation fails.
     */
    oscean::core_services::Geometry unionGeometries(
        const oscean::core_services::Geometry& geom1,
        const oscean::core_services::Geometry& geom2) const;

    /**
     * @brief Computes the difference between two geometries (geom1 - geom2).
     * @param geom1 The geometry from which to subtract.
     * @param geom2 The geometry to subtract.
     * @return The difference geometry.
     * @throws oscean::core_services::InvalidInputException if inputs are invalid.
     * @throws oscean::core_services::OperationFailedException if difference calculation fails.
     */
    oscean::core_services::Geometry difference(
        const oscean::core_services::Geometry& geom1,
        const oscean::core_services::Geometry& geom2) const;

    /**
     * @brief Computes the symmetric difference between two geometries.
     * @param geom1 The first geometry.
     * @param geom2 The second geometry.
     * @return The symmetric difference geometry.
     * @throws oscean::core_services::InvalidInputException if inputs are invalid.
     * @throws oscean::core_services::OperationFailedException if symmetric difference calculation fails.
     */
    oscean::core_services::Geometry symmetricDifference(
        const oscean::core_services::Geometry& geom1,
        const oscean::core_services::Geometry& geom2) const;

    /**
     * @brief Simplifies a geometry using the specified tolerance.
     * @param geom The input geometry.
     * @param tolerance The simplification tolerance.
     * @return The simplified geometry.
     * @throws oscean::core_services::InvalidInputException if input is invalid.
     * @throws oscean::core_services::OperationFailedException if simplification fails.
     */
    oscean::core_services::Geometry simplify(
        const oscean::core_services::Geometry& geom,
        double tolerance) const;

    /**
     * @brief Computes the convex hull of a geometry.
     * @param geom Input geometry.
     * @return Convex hull geometry.
     * @throws InvalidInputDataException if geometry is invalid.
     * @throws OperationFailedException if GEOS operation fails.
     */
    oscean::core_services::Geometry convexHull(
        const oscean::core_services::Geometry& geom) const;

    /**
     * @brief Calculates the distance between two geometries.
     * @param geom1 The first geometry.
     * @param geom2 The second geometry.
     * @param type The type of distance to calculate (e.g., Euclidean, Geodesic).
     *             Note: Geodesic distance might not be fully supported.
     * @return The calculated distance.
     * @throws oscean::core_services::InvalidInputException if inputs are invalid.
     * @throws oscean::core_services::OperationFailedException if distance calculation fails or type is unsupported.
     */
    double calculateDistance(
        const oscean::core_services::Geometry& geom1,
        const oscean::core_services::Geometry& geom2,
        oscean::core_services::spatial_ops::DistanceType type) const;

    /**
     * @brief Evaluates a spatial predicate between two geometries.
     * @param geom1 The first geometry.
     * @param geom2 The second geometry.
     * @param predicate The spatial predicate to evaluate (e.g., INTERSECTS, CONTAINS).
     * @return True if the predicate holds, false otherwise.
     * @throws oscean::core_services::InvalidInputException if inputs are invalid or predicate is unsupported.
     * @throws oscean::core_services::OperationFailedException if predicate evaluation fails.
     */
    bool evaluatePredicate(
        const oscean::core_services::Geometry& geom1,
        const oscean::core_services::Geometry& geom2,
        oscean::core_services::spatial_ops::SpatialPredicate predicate) const;

    /**
     * @brief Tests if a geometry is valid according to OGC specifications.
     * @param geom Geometry to test.
     * @return true if geometry is valid, false otherwise.
     */
    bool isValid(const oscean::core_services::Geometry& geom) const;

    /**
     * @brief Gets the reason why a geometry is invalid.
     * @param geom The geometry to check.
     * @return A string describing the validation issue, or empty string if valid.
     */
    std::string getValidationReason(const oscean::core_services::Geometry& geom) const;

    /**
     * @brief Attempts to make an invalid geometry valid.
     * @param geom The geometry to fix.
     * @return A valid geometry, or the original geometry if it was already valid.
     */
    oscean::core_services::Geometry makeValid(const oscean::core_services::Geometry& geom) const;

    // --- Methods identified as missing by Linter for ISpatialOpsService calls ---
    // These might belong in QueryEngine or SpatialOpsServiceImpl depending on final design

    oscean::core_services::FeatureCollection queryByBoundingBox(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::BoundingBox& bbox) const;

    oscean::core_services::FeatureCollection queryByGeometry(
        const oscean::core_services::FeatureCollection& features,
        const oscean::core_services::Geometry& queryGeom,
        oscean::core_services::spatial_ops::SpatialPredicate predicate) const;

    oscean::core_services::Feature findNearestNeighbor(
        const oscean::core_services::Point& point,
        const oscean::core_services::FeatureCollection& candidates) const;
        
    // ❌ 坐标转换功能已移除 - 使用CRS服务进行坐标转换
    // 使用 ICrsService::transformGeometryAsync() 替代

    oscean::core_services::Point centroid(
        const oscean::core_services::Geometry& geom) const;

    oscean::core_services::BoundingBox envelope(
        const oscean::core_services::Geometry& geom) const;
            
    double area(
        const oscean::core_services::Geometry& geom) const;

    double length(
        const oscean::core_services::Geometry& geom) const;

private:
    oscean::core_services::spatial_ops::SpatialOpsConfig config_; // Added oscean::core_services::spatial_ops namespace
    GEOSContextHandle_t geosContext_;
    GEOSWKTReader* wktReader_;
    GEOSWKTWriter* wktWriter_;

    // Store relevant config values, e.g., tolerance
    double m_geometricTolerance;
    
    // Helper methods
    std::string geosToWkt(const GEOSGeometry* geosGeom) const;
    GEOSGeometry* wktToGeos(const std::string& wkt) const;
    void cleanupGeosGeometry(GEOSGeometry* geom) const;

    // Error handlers
    static void geosErrorHandler(const char* fmt, ...);
    static void geosNoticeHandler(const char* fmt, ...);

    // Initialization and cleanup
    void initializeGeos();
    void cleanupGeos();
    
    // Additional geometry operation helper functions
    double calculateLineParameter(const GEOSGeometry* line, const GEOSGeometry* point) const;
    double pointToSegmentDistance(double px, double py, double x1, double y1, double x2, double y2, double& t) const;
};

} // namespace oscean::core_services::spatial_ops::engine 