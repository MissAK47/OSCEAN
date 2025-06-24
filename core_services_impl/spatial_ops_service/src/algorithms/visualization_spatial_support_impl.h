#pragma once

#include "visualization_spatial_support.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/spatial_ops/spatial_exceptions.h"

namespace oscean::core_services::spatial_ops::algorithms {

/**
 * @brief Implementation of spatial support for visualization service
 */
class VisualizationSpatialSupportImpl : public IVisualizationSpatialSupport {
public:
    explicit VisualizationSpatialSupportImpl(const SpatialOpsConfig& config);
    ~VisualizationSpatialSupportImpl() override = default;

    // --- Rendering Operations ---
    std::future<RenderedImage> renderSpatialData(
        const SpatialData& spatialData,
        const RenderOptions& renderOptions = {}) const override;

    std::future<RenderedImage> renderLayers(
        const std::vector<SpatialLayer>& layers,
        const std::vector<LayerStyle>& layerStyles,
        const RenderOptions& renderOptions = {}) const override;

    std::future<TileGenerationResult> generateVisualizationTiles(
        const SpatialData& spatialData,
        TileScheme tileScheme,
        const std::vector<int>& zoomLevels,
        const StyleOptions& styleOptions = {}) const override;

    // --- Symbolization ---
    std::future<SymbolizedData> applySymbolization(
        const VectorData& vectorData,
        const std::vector<SymbolizationRule>& symbolizationRules,
        const SymbolOptions& symbolOptions = {}) const override;

    std::future<ThematicVisualization> createThematicVisualization(
        const SpatialData& spatialData,
        const std::string& thematicField,
        ThematicVisualizationType visualizationType,
        const ThematicOptions& thematicOptions = {}) const override;

    // --- Styling ---
    std::future<LayerStyle> createStyleFromSLD(
        const std::string& sldContent,
        const SLDValidationOptions& validationOptions = {}) const override;

    std::future<LayerStyle> generateAutomaticStyle(
        const SpatialData& spatialData,
        AutoStyleType styleType,
        const AutoStyleOptions& autoStyleOptions = {}) const override;

    // --- Legend and Annotation ---
    std::future<LegendImage> generateLegend(
        const std::vector<LayerStyle>& layerStyles,
        const LegendOptions& legendOptions = {}) const override;

    std::future<RenderedImage> addAnnotations(
        const RenderedImage& baseImage,
        const std::vector<Annotation>& annotations,
        const AnnotationOptions& annotationOptions = {}) const override;

    // --- Interactive Visualization ---
    std::future<InteractiveMapConfig> createInteractiveMap(
        const std::vector<SpatialLayer>& spatialLayers,
        const InteractiveOptions& interactiveOptions = {}) const override;

    std::future<WMSConfiguration> generateWMSConfiguration(
        const SpatialData& spatialData,
        const WMSOptions& wmsOptions = {}) const override;

    // --- 3D Visualization ---
    std::future<Visualization3D> create3DVisualization(
        const GridData& elevationData,
        const std::optional<SpatialData>& overlayData = std::nullopt,
        const Visualization3DOptions& visualization3DOptions = {}) const override;

    // --- Animation ---
    std::future<SpatialAnimation> createTemporalAnimation(
        const std::vector<TimestampedSpatialData>& temporalData,
        const AnimationOptions& animationOptions = {}) const override;

    // --- Configuration ---
    void setConfiguration(const VisualizationSpatialSupportConfig& config) override;

    VisualizationSpatialSupportConfig getConfiguration() const override;

    std::future<std::vector<std::string>> getSupportedOutputFormats() const override;

    std::future<PerformanceMetrics> getPerformanceMetrics() const override;

private:
    const SpatialOpsConfig& m_config;
    VisualizationSpatialSupportConfig m_supportConfig;
    
    // Helper methods
    RenderedImage createEmptyImage(int width, int height, const std::string& format) const;
    
    std::vector<uint8_t> generatePlaceholderImageData(int width, int height) const;
    
    LayerStyle createDefaultStyle(const SpatialData& spatialData) const;
};

} // namespace oscean::core_services::spatial_ops::algorithms 