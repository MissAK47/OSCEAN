#pragma once

#include "core_services/spatial_ops/spatial_types.h"
#include <future>
#include <vector>
#include <string>
#include <map>
#include <optional>

namespace oscean::core_services::spatial_ops::algorithms {

/**
 * @interface IVisualizationSpatialSupport
 * @brief Interface for spatial visualization support operations
 * 
 * This interface provides comprehensive spatial visualization capabilities
 * including rendering, styling, symbolization, and interactive visualization.
 */
class IVisualizationSpatialSupport {
public:
    virtual ~IVisualizationSpatialSupport() = default;

    // --- Rendering Operations ---
    
    /**
     * @brief Render spatial data to image
     * @param spatialData Input spatial data
     * @param renderOptions Rendering options
     * @return Future containing rendered image
     */
    virtual std::future<RenderedImage> renderSpatialData(
        const SpatialData& spatialData,
        const RenderOptions& renderOptions = {}) const = 0;

    /**
     * @brief Render multiple layers
     * @param layers Vector of spatial layers
     * @param layerStyles Styling for each layer
     * @param renderOptions Global rendering options
     * @return Future containing composite rendered image
     */
    virtual std::future<RenderedImage> renderLayers(
        const std::vector<SpatialLayer>& layers,
        const std::vector<LayerStyle>& layerStyles,
        const RenderOptions& renderOptions = {}) const = 0;

    /**
     * @brief Create map tiles for visualization
     * @param spatialData Input spatial data
     * @param tileScheme Tile scheme to use
     * @param zoomLevels Zoom levels to generate
     * @param styleOptions Styling options
     * @return Future containing tile generation result
     */
    virtual std::future<TileGenerationResult> generateVisualizationTiles(
        const SpatialData& spatialData,
        TileScheme tileScheme,
        const std::vector<int>& zoomLevels,
        const StyleOptions& styleOptions = {}) const = 0;

    // --- Symbolization ---
    
    /**
     * @brief Apply symbolization to vector data
     * @param vectorData Input vector data
     * @param symbolizationRules Symbolization rules
     * @param symbolOptions Symbolization options
     * @return Future containing symbolized data
     */
    virtual std::future<SymbolizedData> applySymbolization(
        const VectorData& vectorData,
        const std::vector<SymbolizationRule>& symbolizationRules,
        const SymbolOptions& symbolOptions = {}) const = 0;

    /**
     * @brief Create thematic visualization
     * @param spatialData Input spatial data
     * @param thematicField Field to visualize
     * @param visualizationType Type of thematic visualization
     * @param thematicOptions Thematic visualization options
     * @return Future containing thematic visualization
     */
    virtual std::future<ThematicVisualization> createThematicVisualization(
        const SpatialData& spatialData,
        const std::string& thematicField,
        ThematicVisualizationType visualizationType,
        const ThematicOptions& thematicOptions = {}) const = 0;

    // --- Styling ---
    
    /**
     * @brief Create style from SLD (Styled Layer Descriptor)
     * @param sldContent SLD XML content
     * @param validationOptions Validation options
     * @return Future containing parsed style
     */
    virtual std::future<LayerStyle> createStyleFromSLD(
        const std::string& sldContent,
        const SLDValidationOptions& validationOptions = {}) const = 0;

    /**
     * @brief Generate style automatically
     * @param spatialData Input spatial data
     * @param styleType Type of automatic styling
     * @param autoStyleOptions Auto-styling options
     * @return Future containing generated style
     */
    virtual std::future<LayerStyle> generateAutomaticStyle(
        const SpatialData& spatialData,
        AutoStyleType styleType,
        const AutoStyleOptions& autoStyleOptions = {}) const = 0;

    // --- Legend and Annotation ---
    
    /**
     * @brief Generate legend for visualization
     * @param layerStyles Styles for which to generate legend
     * @param legendOptions Legend generation options
     * @return Future containing legend image
     */
    virtual std::future<LegendImage> generateLegend(
        const std::vector<LayerStyle>& layerStyles,
        const LegendOptions& legendOptions = {}) const = 0;

    /**
     * @brief Add annotations to visualization
     * @param baseImage Base visualization image
     * @param annotations Annotations to add
     * @param annotationOptions Annotation options
     * @return Future containing annotated image
     */
    virtual std::future<RenderedImage> addAnnotations(
        const RenderedImage& baseImage,
        const std::vector<Annotation>& annotations,
        const AnnotationOptions& annotationOptions = {}) const = 0;

    // --- Interactive Visualization ---
    
    /**
     * @brief Create interactive map configuration
     * @param spatialLayers Spatial layers for interactive map
     * @param interactiveOptions Interactive map options
     * @return Future containing interactive map configuration
     */
    virtual std::future<InteractiveMapConfig> createInteractiveMap(
        const std::vector<SpatialLayer>& spatialLayers,
        const InteractiveOptions& interactiveOptions = {}) const = 0;

    /**
     * @brief Generate web map service configuration
     * @param spatialData Input spatial data
     * @param wmsOptions WMS configuration options
     * @return Future containing WMS configuration
     */
    virtual std::future<WMSConfiguration> generateWMSConfiguration(
        const SpatialData& spatialData,
        const WMSOptions& wmsOptions = {}) const = 0;

    // --- 3D Visualization ---
    
    /**
     * @brief Create 3D visualization
     * @param elevationData Elevation/height data
     * @param overlayData Optional overlay data
     * @param visualization3DOptions 3D visualization options
     * @return Future containing 3D visualization
     */
    virtual std::future<Visualization3D> create3DVisualization(
        const GridData& elevationData,
        const std::optional<SpatialData>& overlayData = std::nullopt,
        const Visualization3DOptions& visualization3DOptions = {}) const = 0;

    // --- Animation ---
    
    /**
     * @brief Create temporal animation
     * @param temporalData Time-series spatial data
     * @param animationOptions Animation options
     * @return Future containing animation
     */
    virtual std::future<SpatialAnimation> createTemporalAnimation(
        const std::vector<TimestampedSpatialData>& temporalData,
        const AnimationOptions& animationOptions = {}) const = 0;

    // --- Configuration ---
    
    /**
     * @brief Set visualization configuration
     * @param config Visualization configuration
     */
    virtual void setConfiguration(const VisualizationSpatialSupportConfig& config) = 0;

    /**
     * @brief Get current configuration
     * @return Current visualization configuration
     */
    virtual VisualizationSpatialSupportConfig getConfiguration() const = 0;

    /**
     * @brief Get supported output formats
     * @return Future containing list of supported formats
     */
    virtual std::future<std::vector<std::string>> getSupportedOutputFormats() const = 0;

    /**
     * @brief Get performance metrics
     * @return Future containing performance metrics
     */
    virtual std::future<PerformanceMetrics> getPerformanceMetrics() const = 0;
};

// --- Supporting Types ---

/**
 * @struct RenderedImage
 * @brief Represents a rendered spatial visualization
 */
struct RenderedImage {
    std::vector<uint8_t> imageData;         ///< Image data in specified format
    std::string format;                     ///< Image format (PNG, JPEG, etc.)
    int width;                              ///< Image width in pixels
    int height;                             ///< Image height in pixels
    BoundingBox geographicExtent;           ///< Geographic extent of image
    std::string crs;                        ///< Coordinate reference system
    std::map<std::string, std::string> metadata; ///< Image metadata
};

/**
 * @struct SpatialLayer
 * @brief Represents a spatial data layer
 */
struct SpatialLayer {
    std::string layerId;                    ///< Layer identifier
    SpatialData data;                       ///< Spatial data
    LayerStyle style;                       ///< Layer styling
    bool visible = true;                    ///< Layer visibility
    double opacity = 1.0;                   ///< Layer opacity (0.0-1.0)
    int zOrder = 0;                         ///< Z-order for layer stacking
    std::map<std::string, std::string> properties; ///< Layer properties
};

/**
 * @struct RenderOptions
 * @brief Options for rendering operations
 */
struct RenderOptions {
    int width = 800;                        ///< Output width in pixels
    int height = 600;                       ///< Output height in pixels
    std::string outputFormat = "PNG";       ///< Output image format
    BoundingBox extent;                     ///< Geographic extent to render
    std::string crs = "EPSG:4326";          ///< Coordinate reference system
    double dpi = 96.0;                      ///< Dots per inch
    bool antialiasing = true;               ///< Enable antialiasing
    std::string backgroundColor = "#FFFFFF"; ///< Background color
    bool transparent = false;               ///< Enable transparency
};

/**
 * @struct LayerStyle
 * @brief Styling information for a spatial layer
 */
struct LayerStyle {
    std::string styleId;                    ///< Style identifier
    std::string styleName;                  ///< Human-readable style name
    std::vector<StyleRule> rules;           ///< Styling rules
    std::map<std::string, std::string> parameters; ///< Style parameters
    std::optional<std::string> sldContent;  ///< SLD XML content
};

/**
 * @struct StyleRule
 * @brief Individual styling rule
 */
struct StyleRule {
    std::string filter;                     ///< Filter expression
    SymbolStyle symbolStyle;                ///< Symbol styling
    double minScale = 0.0;                  ///< Minimum scale for rule
    double maxScale = 0.0;                  ///< Maximum scale for rule (0 = no limit)
};

/**
 * @struct SymbolStyle
 * @brief Symbol styling properties
 */
struct SymbolStyle {
    std::string fillColor = "#000000";      ///< Fill color
    std::string strokeColor = "#000000";    ///< Stroke color
    double strokeWidth = 1.0;               ///< Stroke width
    double opacity = 1.0;                   ///< Symbol opacity
    std::string symbolType = "circle";      ///< Symbol type
    double size = 5.0;                      ///< Symbol size
    std::map<std::string, std::string> customProperties; ///< Custom properties
};

/**
 * @struct VisualizationSpatialSupportConfig
 * @brief Configuration for visualization spatial support
 */
struct VisualizationSpatialSupportConfig {
    std::string defaultOutputFormat = "PNG"; ///< Default output format
    int defaultWidth = 800;                 ///< Default image width
    int defaultHeight = 600;                ///< Default image height
    double defaultDPI = 96.0;               ///< Default DPI
    bool enableCaching = true;              ///< Enable rendering cache
    std::size_t cacheSize = 1000;           ///< Cache size
    bool enableParallelProcessing = true;   ///< Enable parallel processing
    std::size_t maxThreads = 0;             ///< Maximum threads (0 = auto)
    std::string tempDirectory;              ///< Temporary directory for processing
    std::map<std::string, std::string> customSettings; ///< Custom settings
};

// --- Enumerations ---

/**
 * @enum ThematicVisualizationType
 * @brief Types of thematic visualization
 */
enum class ThematicVisualizationType {
    CHOROPLETH,                 ///< Choropleth map
    PROPORTIONAL_SYMBOLS,       ///< Proportional symbol map
    DOT_DENSITY,               ///< Dot density map
    ISOLINE,                   ///< Isoline/contour map
    HEAT_MAP,                  ///< Heat map
    GRADUATED_COLORS,          ///< Graduated color map
    GRADUATED_SYMBOLS          ///< Graduated symbol map
};

/**
 * @enum AutoStyleType
 * @brief Types of automatic styling
 */
enum class AutoStyleType {
    SIMPLE,                    ///< Simple default styling
    QUANTILE,                  ///< Quantile-based styling
    EQUAL_INTERVAL,            ///< Equal interval styling
    NATURAL_BREAKS,            ///< Natural breaks (Jenks) styling
    STANDARD_DEVIATION         ///< Standard deviation styling
};

// Forward declarations for complex types
struct TileGenerationResult;
struct SymbolizedData;
struct ThematicVisualization;
struct ThematicOptions;
struct SymbolizationRule;
struct SymbolOptions;
struct SLDValidationOptions;
struct AutoStyleOptions;
struct LegendImage;
struct LegendOptions;
struct Annotation;
struct AnnotationOptions;
struct InteractiveMapConfig;
struct InteractiveOptions;
struct WMSConfiguration;
struct WMSOptions;
struct Visualization3D;
struct Visualization3DOptions;
struct SpatialAnimation;
struct TimestampedSpatialData;
struct AnimationOptions;
struct StyleOptions;
struct VectorData;

} // namespace oscean::core_services::spatial_ops::algorithms 