/**
 * @file raster_algebra.h
 * @brief RasterAlgebra class for raster algebraic operations
 */

#pragma once

#include "core_services/common_data_types.h"
#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/spatial_ops/spatial_config.h"
#include <map>
#include <string>
#include <optional>

namespace oscean::core_services::spatial_ops::raster {

/**
 * @brief Handles raster algebraic operations and expressions
 */
class RasterAlgebra {
public:
    explicit RasterAlgebra(const SpatialOpsConfig& config);
    ~RasterAlgebra() = default;

    // Non-copyable, non-movable
    RasterAlgebra(const RasterAlgebra&) = delete;
    RasterAlgebra& operator=(const RasterAlgebra&) = delete;
    RasterAlgebra(RasterAlgebra&&) = delete;
    RasterAlgebra& operator=(RasterAlgebra&&) = delete;

    /**
     * @brief Performs raster algebra based on an expression and named input rasters
     * @param expression The mathematical expression to evaluate
     * @param namedRasters A map of names to GridData objects, used as variables in the expression
     * @param targetGridDef Optional target grid definition for the output raster
     * @param noDataValue Optional NoData value for the output raster
     * @return A new GridData object representing the result of the raster algebra
     */
    oscean::core_services::GridData performRasterAlgebra(
        const std::string& expression,
        const std::map<std::string, oscean::core_services::GridData>& namedRasters,
        std::optional<oscean::core_services::GridDefinition> targetGridDef = std::nullopt,
        std::optional<double> noDataValue = std::nullopt) const;

    /**
     * @brief Adds two rasters pixel by pixel
     * @param rasterA First input raster
     * @param rasterB Second input raster
     * @param noDataValue Optional NoData value for the output
     * @return Result of A + B
     */
    oscean::core_services::GridData addRasters(
        const oscean::core_services::GridData& rasterA,
        const oscean::core_services::GridData& rasterB,
        std::optional<double> noDataValue = std::nullopt) const;

    /**
     * @brief Subtracts two rasters pixel by pixel
     * @param rasterA First input raster
     * @param rasterB Second input raster
     * @param noDataValue Optional NoData value for the output
     * @return Result of A - B
     */
    oscean::core_services::GridData subtractRasters(
        const oscean::core_services::GridData& rasterA,
        const oscean::core_services::GridData& rasterB,
        std::optional<double> noDataValue = std::nullopt) const;

    /**
     * @brief Multiplies two rasters pixel by pixel
     * @param rasterA First input raster
     * @param rasterB Second input raster
     * @param noDataValue Optional NoData value for the output
     * @return Result of A * B
     */
    oscean::core_services::GridData multiplyRasters(
        const oscean::core_services::GridData& rasterA,
        const oscean::core_services::GridData& rasterB,
        std::optional<double> noDataValue = std::nullopt) const;

    /**
     * @brief Divides two rasters pixel by pixel
     * @param rasterA First input raster (dividend)
     * @param rasterB Second input raster (divisor)
     * @param noDataValue Optional NoData value for the output
     * @return Result of A / B
     */
    oscean::core_services::GridData divideRasters(
        const oscean::core_services::GridData& rasterA,
        const oscean::core_services::GridData& rasterB,
        std::optional<double> noDataValue = std::nullopt) const;

    /**
     * @brief Applies a mathematical function to a raster
     * @param inputRaster Input raster
     * @param function Function name (e.g., "sin", "cos", "log", "sqrt")
     * @param noDataValue Optional NoData value for the output
     * @return Result raster with function applied
     */
    oscean::core_services::GridData applyMathFunction(
        const oscean::core_services::GridData& inputRaster,
        const std::string& function,
        std::optional<double> noDataValue = std::nullopt) const;

private:
    const SpatialOpsConfig& m_config;

    /**
     * @brief Validates that two rasters are compatible for algebraic operations
     */
    void validateRasterCompatibility(
        const oscean::core_services::GridData& rasterA,
        const oscean::core_services::GridData& rasterB) const;

    /**
     * @brief Parses and evaluates a mathematical expression
     */
    oscean::core_services::GridData evaluateExpression(
        const std::string& expression,
        const std::map<std::string, oscean::core_services::GridData>& namedRasters) const;

    /**
     * @brief Performs binary operation on two rasters
     */
    oscean::core_services::GridData performBinaryOperation(
        const oscean::core_services::GridData& rasterA,
        const oscean::core_services::GridData& rasterB,
        const std::string& operation,
        std::optional<double> noDataValue) const;

    /**
     * @brief Checks if a value should be treated as NoData
     */
    bool isNoData(double value, std::optional<double> noDataValue) const;
};

} // namespace oscean::core_services::spatial_ops::raster 