#pragma once

#include "modeling_spatial_support.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/spatial_ops/spatial_exceptions.h"

namespace oscean::core_services::spatial_ops::algorithms {

/**
 * @brief Implementation of spatial support for modeling service
 * 
 * This implementation is split across multiple files for maintainability:
 * - modeling_grid_generator.cpp - Grid generation operations
 * - modeling_mesh_operations.cpp - Mesh operations
 * - modeling_boundary_conditions.cpp - Boundary condition handling
 * - modeling_field_initialization.cpp - Field initialization
 * - modeling_discretization.cpp - Spatial discretization
 * - modeling_coupling.cpp - Model coupling operations
 * - modeling_refinement.cpp - Adaptive refinement
 * - modeling_quality_assessment.cpp - Quality assessment
 */
class ModelingSpatialSupportImpl : public IModelingSpatialSupport {
public:
    explicit ModelingSpatialSupportImpl(const SpatialOpsConfig& config);
    ~ModelingSpatialSupportImpl() override = default;

    // --- Domain Creation ---
    std::future<ComputationalDomain> createComputationalDomain(
        const Geometry& domainGeometry,
        double resolution,
        const DomainCreationOptions& options = {}) const override;

    // --- Mesh Generation ---
    std::future<SpatialMesh> generateSpatialMesh(
        const ComputationalDomain& domain,
        MeshType meshType,
        const MeshGenerationOptions& options = {}) const override;

    std::future<SpatialMesh> refineMesh(
        const SpatialMesh& mesh,
        const std::vector<RefinementCriterion>& refinementCriteria,
        const MeshRefinementOptions& options = {}) const override;

    std::future<MeshValidationResult> validateMesh(
        const SpatialMesh& mesh,
        const std::vector<MeshQualityMetric>& qualityMetrics) const override;

    // --- Configuration ---
    void setConfiguration(const ModelingSpatialSupportConfig& config) override;

    ModelingSpatialSupportConfig getConfiguration() const override;

    std::future<std::vector<std::string>> getSupportedDiscretizationMethods() const;

    std::future<PerformanceMetrics> getPerformanceMetrics() const;

private:
    const SpatialOpsConfig& m_config;
    ModelingSpatialSupportConfig m_supportConfig;
    
    // Helper methods (implemented in respective files)
    std::vector<Point> generateGridPoints(
        const BoundingBox& extent, double cellSizeX, double cellSizeY) const;
    
    double calculateCellQuality(const std::vector<Point>& cellVertices) const;
};

} // namespace oscean::core_services::spatial_ops::algorithms 