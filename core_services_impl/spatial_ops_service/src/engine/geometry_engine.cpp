#include "engine/geometry_engine.h" // Updated path
#include "core_services/exceptions.h" // This should be OsceanException or spatial_exceptions.h
#include "core_services/spatial_ops/spatial_exceptions.h"

// GEOS includes
#include <geos_c.h>

// OGR includes for coordinate transformations
#include <ogr_geometry.h>
#include <ogr_spatialref.h>
#include <ogr_api.h>

#include <stdexcept>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <limits>

// Using the new namespace from the header
namespace oscean::core_services::spatial_ops::engine {

// Static error handlers for GEOS
void GeometryEngine::geosErrorHandler(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    
    // Consider using a proper logger from common_utils here
    std::cerr << "GEOS Error: " << buffer << std::endl;
}

void GeometryEngine::geosNoticeHandler(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    
    // Consider using a proper logger from common_utils here
    std::cout << "GEOS Notice: " << buffer << std::endl;
}

GeometryEngine::GeometryEngine(const oscean::core_services::spatial_ops::SpatialOpsConfig& config)
    : config_(config)
    , geosContext_(nullptr)
    , wktReader_(nullptr)
    , wktWriter_(nullptr)
    , m_geometricTolerance(0.0) // Initialize with default value
    {
    
    initializeGeos();
    // Get tolerance from algorithm settings
    m_geometricTolerance = config_.algorithmSettings.geometricTolerance;
}

GeometryEngine::~GeometryEngine() {
    cleanupGeos();
}

void GeometryEngine::initializeGeos() {
    // Initialize GEOS with custom error handlers
    geosContext_ = GEOS_init_r();
    if (!geosContext_) {
        throw oscean::core_services::spatial_ops::OperationFailedException("Failed to initialize GEOS context");
    }
    
    // Set error and notice handlers
    GEOSContext_setErrorHandler_r(geosContext_, geosErrorHandler);
    GEOSContext_setNoticeHandler_r(geosContext_, geosNoticeHandler);
    
    // Create WKT reader and writer
    wktReader_ = GEOSWKTReader_create_r(geosContext_);
    if (!wktReader_) {
        cleanupGeos(); // Clean up partially initialized GEOS
        throw oscean::core_services::spatial_ops::OperationFailedException("Failed to create GEOS WKT reader");
    }
    
    wktWriter_ = GEOSWKTWriter_create_r(geosContext_);
    if (!wktWriter_) {
        cleanupGeos(); // Clean up partially initialized GEOS
        throw oscean::core_services::spatial_ops::OperationFailedException("Failed to create GEOS WKT writer");
    }
    
    // Configure WKT writer for better output
    GEOSWKTWriter_setTrim_r(geosContext_, wktWriter_, 1);
    GEOSWKTWriter_setRoundingPrecision_r(geosContext_, wktWriter_, 15); // Consider making precision configurable
}

void GeometryEngine::cleanupGeos() {
    if (wktWriter_) {
        GEOSWKTWriter_destroy_r(geosContext_, wktWriter_);
        wktWriter_ = nullptr;
    }
    
    if (wktReader_) {
        GEOSWKTReader_destroy_r(geosContext_, wktReader_);
        wktReader_ = nullptr;
    }
    
    if (geosContext_) {
        GEOS_finish_r(geosContext_);
        geosContext_ = nullptr;
    }
}

GEOSGeometry* GeometryEngine::wktToGeos(const std::string& wkt) const {
    if (wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidParameterException("Input WKT string is empty");
    }
    
    GEOSGeometry* geom = GEOSWKTReader_read_r(geosContext_, wktReader_, wkt.c_str());
    if (!geom) {
        // Try to get more detailed error from GEOS if possible, though GEOS error handler should have logged it.
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Failed to parse WKT: " + wkt);
    }
    
    return geom;
}

std::string GeometryEngine::geosToWkt(const GEOSGeometry* geom) const {
    if (!geom) {
        throw oscean::core_services::spatial_ops::InvalidParameterException("Input GEOS geometry is null");
    }
    
    char* wktCStr = GEOSWKTWriter_write_r(geosContext_, wktWriter_, geom);
    if (!wktCStr) {
        throw oscean::core_services::spatial_ops::OperationFailedException("Failed to convert GEOS geometry to WKT string");
    }
    
    std::string wkt(wktCStr);
    GEOSFree_r(geosContext_, wktCStr);
    
    return wkt;
}

void GeometryEngine::cleanupGeosGeometry(GEOSGeometry* geom) const {
    if (geom) {
        GEOSGeom_destroy_r(geosContext_, geom);
    }
}

oscean::core_services::Geometry GeometryEngine::buffer(
    const oscean::core_services::Geometry& geom,
    double distance,
    const oscean::core_services::spatial_ops::BufferOptions& options) const {
    
    if (geom.wkt.empty()) {
         throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for buffer operation");
    }

    GEOSGeometry* inputGeom = nullptr;
    GEOSGeometry* bufferedGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        
        // Create buffer with specified parameters
        // Use simple GEOSBuffer_r for now, can be enhanced later with GEOSBufferWithStyle_r
        bufferedGeom = GEOSBuffer_r(geosContext_, inputGeom, distance, options.quadrantSegments);
        
        if (!bufferedGeom) {
            throw oscean::core_services::spatial_ops::OperationFailedException("GEOS buffer operation returned null");
        }
        
        std::string resultWkt = geosToWkt(bufferedGeom);
        oscean::core_services::Geometry result;
        result.wkt = resultWkt;
        // Note: Geometry class doesn't have srid member, so we don't preserve it
        
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(bufferedGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(bufferedGeom);
        throw; // Re-throw the caught exception (could be InvalidInputDataException, OperationFailedException, etc.)
    }
}

oscean::core_services::Geometry GeometryEngine::intersection(
    const oscean::core_services::Geometry& geom1,
    const oscean::core_services::Geometry& geom2) const {
    if (geom1.wkt.empty() || geom2.wkt.empty()) {
         throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for intersection operation");
    }
    // Note: Cannot check SRID compatibility since Geometry class doesn't have srid member

    GEOSGeometry* g1 = nullptr;
    GEOSGeometry* g2 = nullptr;
    GEOSGeometry* resultGeom = nullptr;
    try {
        g1 = wktToGeos(geom1.wkt);
        g2 = wktToGeos(geom2.wkt);
        
        resultGeom = GEOSIntersection_r(geosContext_, g1, g2);
        
        if (!resultGeom) {
            throw oscean::core_services::spatial_ops::OperationFailedException("GEOS intersection operation returned null");
        }
        
        std::string resultWkt = geosToWkt(resultGeom);
        oscean::core_services::Geometry result;
        result.wkt = resultWkt;
        // Note: Geometry class doesn't have srid member, so we don't preserve it

        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        cleanupGeosGeometry(resultGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        cleanupGeosGeometry(resultGeom);
        throw;
    }
}

oscean::core_services::Geometry GeometryEngine::unionGeometries(
    const oscean::core_services::Geometry& geom1,
    const oscean::core_services::Geometry& geom2) const {
    
    if (geom1.wkt.empty() || geom2.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for union operation");
    }
    
    GEOSGeometry* g1 = nullptr;
    GEOSGeometry* g2 = nullptr;
    GEOSGeometry* resultGeom = nullptr;
    try {
        g1 = wktToGeos(geom1.wkt);
        g2 = wktToGeos(geom2.wkt);
        
        resultGeom = GEOSUnion_r(geosContext_, g1, g2);
        if (!resultGeom) {
            throw oscean::core_services::spatial_ops::OperationFailedException("GEOS union operation returned null");
        }
        
        std::string resultWkt = geosToWkt(resultGeom);
        oscean::core_services::Geometry result;
        result.wkt = resultWkt;
        
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        cleanupGeosGeometry(resultGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        cleanupGeosGeometry(resultGeom);
        throw;
    }
}

// ❌ 坐标转换功能已移除 - 避免与CRS服务功能重复
// 
// 空间服务专注于几何运算，坐标转换功能统一由CRS服务提供：
// - 使用 ICrsService::transformGeometryAsync() 进行几何转换
// - 使用 ICrsService::transformPointAsync() 进行点转换
// - 使用 ICrsService::parseFromEpsgCodeAsync() 获取CRS信息

// ... (Implementation for centroid, envelope, area, length using GEOS)

oscean::core_services::Point GeometryEngine::centroid(
    const oscean::core_services::Geometry& geom) const {
    if (geom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for centroid operation");
    }
    GEOSGeometry* inputGeom = nullptr;
    GEOSGeometry* centroidGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        centroidGeom = GEOSGetCentroid_r(geosContext_, inputGeom);
        if (!centroidGeom || GEOSisEmpty_r(geosContext_, centroidGeom)) {
             cleanupGeosGeometry(inputGeom);
             cleanupGeosGeometry(centroidGeom);
            throw oscean::core_services::spatial_ops::OperationFailedException("Failed to calculate centroid or centroid is empty");
        }

        double x, y;
        if (GEOSGeomGetX_r(geosContext_, centroidGeom, &x) != 1 || GEOSGeomGetY_r(geosContext_, centroidGeom, &y) != 1) {
            cleanupGeosGeometry(inputGeom);
            cleanupGeosGeometry(centroidGeom);
            throw oscean::core_services::spatial_ops::OperationFailedException("Failed to get coordinates from centroid geometry");
        }
        
        oscean::core_services::Point pt(x, y);
        // Note: Point class has optional crsId member, but we don't have source CRS info
        
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(centroidGeom);
        return pt;
    } catch(...) {
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(centroidGeom);
        throw;
    }
}

oscean::core_services::BoundingBox GeometryEngine::envelope(
    const oscean::core_services::Geometry& geom) const {
    if (geom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for envelope operation");
    }
    GEOSGeometry* inputGeom = nullptr;
    GEOSGeometry* envelopeGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        envelopeGeom = GEOSEnvelope_r(geosContext_, inputGeom);
        if (!envelopeGeom || GEOSisEmpty_r(geosContext_, envelopeGeom) || (strcmp(GEOSGeomType_r(geosContext_, envelopeGeom), "Polygon") != 0) ){
            cleanupGeosGeometry(inputGeom);
            cleanupGeosGeometry(envelopeGeom);
            throw oscean::core_services::spatial_ops::OperationFailedException("Failed to calculate envelope or envelope is not a polygon");
        }

        const GEOSGeometry* exteriorRing = GEOSGetExteriorRing_r(geosContext_, envelopeGeom);
        if (!exteriorRing) {
            cleanupGeosGeometry(inputGeom);
            cleanupGeosGeometry(envelopeGeom);
            throw oscean::core_services::spatial_ops::OperationFailedException("Failed to get exterior ring of envelope");
        }

        const GEOSCoordSequence* coordSeq = GEOSGeom_getCoordSeq_r(geosContext_, exteriorRing);
        if (!coordSeq) {
            cleanupGeosGeometry(inputGeom);
            cleanupGeosGeometry(envelopeGeom);
            throw oscean::core_services::spatial_ops::OperationFailedException("Failed to get coordinate sequence of envelope ring");
        }

        double minX, minY, maxX, maxY;
        unsigned int numCoords;
        GEOSCoordSeq_getSize_r(geosContext_, coordSeq, &numCoords);
        if (numCoords < 1) {
             cleanupGeosGeometry(inputGeom);
             cleanupGeosGeometry(envelopeGeom);
             throw oscean::core_services::spatial_ops::OperationFailedException("Envelope coordinate sequence is empty");
        }
        
        // Initialize with first point
        GEOSCoordSeq_getX_r(geosContext_, coordSeq, 0, &minX);
        GEOSCoordSeq_getY_r(geosContext_, coordSeq, 0, &minY);
        maxX = minX;
        maxY = minY;

        for (unsigned int i = 1; i < numCoords; ++i) {
            double x, y;
            GEOSCoordSeq_getX_r(geosContext_, coordSeq, i, &x);
            GEOSCoordSeq_getY_r(geosContext_, coordSeq, i, &y);
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }
        
        oscean::core_services::BoundingBox bbox(minX, minY, maxX, maxY);
        // Note: BoundingBox constructor takes crsId as last parameter, but we don't have source CRS info

        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(envelopeGeom);
        // GEOSGetExteriorRing_r returns a pointer within envelopeGeom, no separate cleanup
        // GEOSGeom_getCoordSeq_r also internal, no separate cleanup
        return bbox;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(envelopeGeom);
        throw;
    }
}

double GeometryEngine::area(
    const oscean::core_services::Geometry& geom) const {
    if (geom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for area operation");
    }
    GEOSGeometry* inputGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        double areaVal = 0.0;
        if (GEOSArea_r(geosContext_, inputGeom, &areaVal) != 1) {
            cleanupGeosGeometry(inputGeom);
            throw oscean::core_services::spatial_ops::OperationFailedException("Failed to calculate area using GEOS");
        }
        cleanupGeosGeometry(inputGeom);
        return areaVal;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        throw;
    }
}

double GeometryEngine::length(
    const oscean::core_services::Geometry& geom) const {
    if (geom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for length operation");
    }
     GEOSGeometry* inputGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        double lengthVal = 0.0;
        if (GEOSLength_r(geosContext_, inputGeom, &lengthVal) != 1) {
            cleanupGeosGeometry(inputGeom);
            throw oscean::core_services::spatial_ops::OperationFailedException("Failed to calculate length using GEOS");
        }
        cleanupGeosGeometry(inputGeom);
        return lengthVal;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        throw;
    }
}

double GeometryEngine::calculateDistance(
    const oscean::core_services::Geometry& geom1,
    const oscean::core_services::Geometry& geom2,
    oscean::core_services::spatial_ops::DistanceType distanceType) const {
    if (geom1.wkt.empty() || geom2.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for distance calculation");
    }
    // Note: Cannot check SRID compatibility since Geometry class doesn't have srid member

    GEOSGeometry* g1 = nullptr;
    GEOSGeometry* g2 = nullptr;
    try {
        g1 = wktToGeos(geom1.wkt);
        g2 = wktToGeos(geom2.wkt);
        
        double distance = 0.0;
        switch (distanceType) {
            case oscean::core_services::spatial_ops::DistanceType::EUCLIDEAN:
                if (GEOSDistance_r(geosContext_, g1, g2, &distance) != 1) {
                    cleanupGeosGeometry(g1);
                    cleanupGeosGeometry(g2);
                    throw oscean::core_services::spatial_ops::OperationFailedException("Failed to calculate Euclidean distance using GEOS");
                }
                break;
            case oscean::core_services::spatial_ops::DistanceType::GEODESIC:
                // For geodesic distance, we'd need to use OGR or a specialized library like GeographicLib
                // For now, fall back to Euclidean
                if (GEOSDistance_r(geosContext_, g1, g2, &distance) != 1) {
                    cleanupGeosGeometry(g1);
                    cleanupGeosGeometry(g2);
                    throw oscean::core_services::spatial_ops::OperationFailedException("Failed to calculate geodesic distance (using Euclidean fallback)");
                }
                break;
            default:
                cleanupGeosGeometry(g1);
                cleanupGeosGeometry(g2);
                throw oscean::core_services::spatial_ops::InvalidParameterException("Unsupported distance type");
        }
        
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        return distance;
    } catch (...) {
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        throw;
    }
}

// --- Query methods that were in GeometryEngine header, might move to QueryEngine ---
// For now, providing stub implementations or noting they should be in QueryEngine.

oscean::core_services::FeatureCollection GeometryEngine::queryByBoundingBox(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::BoundingBox& bbox) const {
    // This logic should ideally be in QueryEngine and use spatial indexing.
    // Basic implementation without index for now, if it must remain here.
    oscean::core_services::FeatureCollection result;
    result.crs = features.crs;

    for (const auto& feature : features.getFeatures()) {
        const std::string& featureWkt = feature.getGeometry();
        if (featureWkt.empty()) continue;

        // Rough check: Get BBox of feature geometry and see if it intersects query bbox
        // This is expensive if done repeatedly without an index.
        oscean::core_services::Geometry featureGeom;
        featureGeom.wkt = featureWkt;
        oscean::core_services::BoundingBox featureBBox = envelope(featureGeom);
        
        // Simple BBox intersection check
        if (!(featureBBox.maxX < bbox.minX || featureBBox.minX > bbox.maxX ||
              featureBBox.maxY < bbox.minY || featureBBox.minY > bbox.maxY)) {
            result.addFeature(feature);
        }
    }
    return result;
}

oscean::core_services::FeatureCollection GeometryEngine::queryByGeometry(
    const oscean::core_services::FeatureCollection& features,
    const oscean::core_services::Geometry& queryGeom,
    oscean::core_services::spatial_ops::SpatialPredicate predicate) const {
    // This logic should ideally be in QueryEngine and use spatial indexing.
    oscean::core_services::FeatureCollection result;
    result.crs = features.crs;

    if (queryGeom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Query geometry WKT is empty.");
    }

    for (const auto& feature : features.getFeatures()) {
        const std::string& featureWkt = feature.getGeometry();
        if (featureWkt.empty()) continue;
        
        oscean::core_services::Geometry featureGeom;
        featureGeom.wkt = featureWkt;
        if (evaluatePredicate(featureGeom, queryGeom, predicate)) {
            result.addFeature(feature);
        }
    }
    return result;
}

oscean::core_services::Feature GeometryEngine::findNearestNeighbor(
    const oscean::core_services::Point& point,
    const oscean::core_services::FeatureCollection& candidates) const {
    // This logic should ideally be in QueryEngine and use spatial indexing (e.g., k-NN on R-tree).
    if (candidates.getFeatures().empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Candidate feature collection is empty for nearest neighbor search");
    }

    oscean::core_services::Feature nearestFeature = candidates.getFeatures()[0]; // Initialize with first
    double minDistance = std::numeric_limits<double>::max();

    oscean::core_services::Geometry pointGeom;
    pointGeom.wkt = "POINT (" + std::to_string(point.x) + " " + std::to_string(point.y) + ")";
    // Note: Geometry class doesn't have srid or crsWkt members

    bool first = true;
    for (const auto& candidateFeature : candidates.getFeatures()) {
        const std::string& candidateWkt = candidateFeature.getGeometry();
        if (candidateWkt.empty()) continue;

        oscean::core_services::Geometry candidateGeom;
        candidateGeom.wkt = candidateWkt;
        
        double dist = calculateDistance(pointGeom, candidateGeom, oscean::core_services::spatial_ops::DistanceType::EUCLIDEAN); // Assuming Euclidean
        if (first || dist < minDistance) {
            minDistance = dist;
            nearestFeature = candidateFeature;
            first = false;
        }
    }
    if(first) { // No valid candidates found
         throw oscean::core_services::spatial_ops::OperationFailedException("No valid candidate features found or distance calculation failed for all");
    }
    return nearestFeature;
}

// === 缺失的方法实现 ===

bool GeometryEngine::isValid(const oscean::core_services::Geometry& geom) const {
    if (geom.wkt.empty()) {
        return false;
    }
    
    GEOSGeometry* inputGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        char isValidResult = GEOSisValid_r(geosContext_, inputGeom);
        cleanupGeosGeometry(inputGeom);
        return isValidResult == 1;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        return false;
    }
}

std::string GeometryEngine::getValidationReason(const oscean::core_services::Geometry& geom) const {
    if (geom.wkt.empty()) {
        return "Empty WKT string";
    }
    
    GEOSGeometry* inputGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        char* reason = GEOSisValidReason_r(geosContext_, inputGeom);
        std::string result;
        if (reason) {
            result = std::string(reason);
            GEOSFree_r(geosContext_, reason);
        }
        cleanupGeosGeometry(inputGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        return "Failed to validate geometry";
    }
}

oscean::core_services::Geometry GeometryEngine::makeValid(const oscean::core_services::Geometry& geom) const {
    if (geom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for makeValid operation");
    }
    
    GEOSGeometry* inputGeom = nullptr;
    GEOSGeometry* validGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        
        // Check if already valid
        if (GEOSisValid_r(geosContext_, inputGeom) == 1) {
            cleanupGeosGeometry(inputGeom);
            return geom; // Already valid
        }
        
        // Try to make valid using GEOS buffer(0) trick
        validGeom = GEOSBuffer_r(geosContext_, inputGeom, 0.0, 8);
        if (!validGeom) {
            cleanupGeosGeometry(inputGeom);
            throw oscean::core_services::spatial_ops::OperationFailedException("Failed to make geometry valid using GEOS");
        }
        
        std::string resultWkt = geosToWkt(validGeom);
        oscean::core_services::Geometry result;
        result.wkt = resultWkt;
        
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(validGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(validGeom);
        throw;
    }
}

bool GeometryEngine::evaluatePredicate(
    const oscean::core_services::Geometry& geom1,
    const oscean::core_services::Geometry& geom2,
    oscean::core_services::spatial_ops::SpatialPredicate predicate) const {
    
    if (geom1.wkt.empty() || geom2.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for predicate evaluation");
    }
    
    GEOSGeometry* g1 = nullptr;
    GEOSGeometry* g2 = nullptr;
    try {
        g1 = wktToGeos(geom1.wkt);
        g2 = wktToGeos(geom2.wkt);
        
        char result = 0;
        switch (predicate) {
            case oscean::core_services::spatial_ops::SpatialPredicate::INTERSECTS:
                result = GEOSIntersects_r(geosContext_, g1, g2);
                break;
            case oscean::core_services::spatial_ops::SpatialPredicate::CONTAINS:
                result = GEOSContains_r(geosContext_, g1, g2);
                break;
            case oscean::core_services::spatial_ops::SpatialPredicate::WITHIN:
                result = GEOSWithin_r(geosContext_, g1, g2);
                break;
            case oscean::core_services::spatial_ops::SpatialPredicate::TOUCHES:
                result = GEOSTouches_r(geosContext_, g1, g2);
                break;
            case oscean::core_services::spatial_ops::SpatialPredicate::CROSSES:
                result = GEOSCrosses_r(geosContext_, g1, g2);
                break;
            case oscean::core_services::spatial_ops::SpatialPredicate::OVERLAPS:
                result = GEOSOverlaps_r(geosContext_, g1, g2);
                break;
            case oscean::core_services::spatial_ops::SpatialPredicate::DISJOINT:
                result = GEOSDisjoint_r(geosContext_, g1, g2);
                break;
            case oscean::core_services::spatial_ops::SpatialPredicate::EQUALS:
                result = GEOSEquals_r(geosContext_, g1, g2);
                break;
            default:
                cleanupGeosGeometry(g1);
                cleanupGeosGeometry(g2);
                throw oscean::core_services::spatial_ops::InvalidParameterException("Unsupported spatial predicate");
        }
        
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        return result == 1;
    } catch (...) {
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        throw;
    }
}

oscean::core_services::Geometry GeometryEngine::difference(
    const oscean::core_services::Geometry& geom1,
    const oscean::core_services::Geometry& geom2) const {
    
    if (geom1.wkt.empty() || geom2.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for difference operation");
    }
    
    GEOSGeometry* g1 = nullptr;
    GEOSGeometry* g2 = nullptr;
    GEOSGeometry* resultGeom = nullptr;
    try {
        g1 = wktToGeos(geom1.wkt);
        g2 = wktToGeos(geom2.wkt);
        
        resultGeom = GEOSDifference_r(geosContext_, g1, g2);
        if (!resultGeom) {
            throw oscean::core_services::spatial_ops::OperationFailedException("GEOS difference operation returned null");
        }
        
        std::string resultWkt = geosToWkt(resultGeom);
        oscean::core_services::Geometry result;
        result.wkt = resultWkt;
        
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        cleanupGeosGeometry(resultGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        cleanupGeosGeometry(resultGeom);
        throw;
    }
}

oscean::core_services::Geometry GeometryEngine::simplify(
    const oscean::core_services::Geometry& geom,
    double tolerance) const {
    
    if (geom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for simplify operation");
    }
    
    if (tolerance < 0.0) {
        throw oscean::core_services::spatial_ops::InvalidParameterException("Simplification tolerance must be non-negative");
    }
    
    GEOSGeometry* inputGeom = nullptr;
    GEOSGeometry* simplifiedGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        
        simplifiedGeom = GEOSSimplify_r(geosContext_, inputGeom, tolerance);
        if (!simplifiedGeom) {
            throw oscean::core_services::spatial_ops::OperationFailedException("GEOS simplify operation returned null");
        }
        
        std::string resultWkt = geosToWkt(simplifiedGeom);
        oscean::core_services::Geometry result;
        result.wkt = resultWkt;
        
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(simplifiedGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(simplifiedGeom);
        throw;
    }
}

oscean::core_services::Geometry GeometryEngine::symmetricDifference(
    const oscean::core_services::Geometry& geom1,
    const oscean::core_services::Geometry& geom2) const {
    
    if (geom1.wkt.empty() || geom2.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for symmetric difference operation");
    }
    
    GEOSGeometry* g1 = nullptr;
    GEOSGeometry* g2 = nullptr;
    GEOSGeometry* resultGeom = nullptr;
    try {
        g1 = wktToGeos(geom1.wkt);
        g2 = wktToGeos(geom2.wkt);
        
        resultGeom = GEOSSymDifference_r(geosContext_, g1, g2);
        if (!resultGeom) {
            throw oscean::core_services::spatial_ops::OperationFailedException("GEOS symmetric difference operation returned null");
        }
        
        std::string resultWkt = geosToWkt(resultGeom);
        oscean::core_services::Geometry result;
        result.wkt = resultWkt;
        
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        cleanupGeosGeometry(resultGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(g1);
        cleanupGeosGeometry(g2);
        cleanupGeosGeometry(resultGeom);
        throw;
    }
}

oscean::core_services::Geometry GeometryEngine::convexHull(
    const oscean::core_services::Geometry& geom) const {
    
    if (geom.wkt.empty()) {
        throw oscean::core_services::spatial_ops::InvalidInputDataException("Input geometry WKT is empty for convexHull operation");
    }
    
    GEOSGeometry* inputGeom = nullptr;
    GEOSGeometry* hullGeom = nullptr;
    try {
        inputGeom = wktToGeos(geom.wkt);
        
        hullGeom = GEOSConvexHull_r(geosContext_, inputGeom);
        if (!hullGeom) {
            throw oscean::core_services::spatial_ops::OperationFailedException("GEOS convex hull operation returned null");
        }
        
        std::string resultWkt = geosToWkt(hullGeom);
        oscean::core_services::Geometry result;
        result.wkt = resultWkt;
        
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(hullGeom);
        return result;
    } catch (...) {
        cleanupGeosGeometry(inputGeom);
        cleanupGeosGeometry(hullGeom);
        throw;
    }
}

} // namespace oscean::core_services::spatial_ops::engine
