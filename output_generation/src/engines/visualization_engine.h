#pragma once

#include "core_services/output/i_output_service.h"
#include "core_services/common_data_types.h"
#include "common_utils/simd/simd_manager_unified.h"
#include "font_renderer.h"  // 包含完整定义
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <atomic>
#include <boost/thread/future.hpp>

// 添加GPU相关头文件
#ifdef OSCEAN_CUDA_ENABLED
#include "output_generation/gpu/gpu_visualization_engine.h"
#include "output_generation/gpu/gpu_color_mapper.h"
#include "output_generation/gpu/gpu_tile_generator.h"
#endif

namespace oscean {
namespace common_utils { 
namespace infrastructure {
    class UnifiedThreadPoolManager;
} // namespace infrastructure
namespace gpu {
    class OSCEANGPUFramework;
    class MultiGPUCoordinator;
} // namespace gpu
} // namespace common_utils

namespace output {

/**
 * @class VisualizationEngine
 * @brief 负责数据可视化和图像生成的引擎
 * 
 * 此引擎专门处理以下类型的输出：
 * - 栅格图像（PNG、JPEG、TIFF等）
 * - 瓦片服务（XYZ、WMTS）
 * - 等值线图
 * - 彩色地图渲染
 * - 图例生成
 * 
 * v2.0 性能优化特性：
 * - SIMD加速的颜色映射（4-8倍加速）
 * - 并行瓦片生成（4-16倍加速）
 * - 智能内存管理
 * - 缓存优化
 * 
 * v3.0 GPU加速特性：
 * - GPU颜色映射（25-40倍加速）
 * - GPU瓦片生成（30-50倍加速）
 * - 多GPU负载均衡
 * - GPU内存优化
 */
class VisualizationEngine {
public:
    /**
     * @brief 构造函数
     * @param threadPool 线程池，用于异步图像处理
     * @param simdManager SIMD管理器，用于加速计算（可选）
     */
    explicit VisualizationEngine(
        std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPool,
        std::shared_ptr<common_utils::simd::UnifiedSIMDManager> simdManager = nullptr
    );

    ~VisualizationEngine() = default;

    /**
     * @brief 处理可视化请求
     * @param request 可视化输出请求
     * @return 输出结果的future
     */
    boost::future<core_services::output::OutputResult> process(
        const core_services::output::OutputRequest& request);

    /**
     * @brief 渲染GridData为图像
     * @param gridData 要渲染的网格数据
     * @param outputPath 输出图像路径
     * @param style 渲染样式选项
     * @return 生成的图像文件路径
     */
    boost::future<std::string> renderToImage(
        std::shared_ptr<core_services::GridData> gridData,
        const std::string& outputPath,
        const core_services::output::StyleOptions& style);

    /**
     * @brief 高性能渲染（使用SIMD优化）
     * @param gridData 要渲染的网格数据
     * @param outputPath 输出图像路径
     * @param style 渲染样式选项
     * @return 生成的图像文件路径
     */
    boost::future<std::string> renderToImageOptimized(
        std::shared_ptr<core_services::GridData> gridData,
        const std::string& outputPath,
        const core_services::output::StyleOptions& style);

    /**
     * @brief 生成瓦片金字塔
     * @param gridData 源数据
     * @param outputDirectory 瓦片输出目录
     * @param style 渲染样式
     * @param minZoom 最小缩放级别
     * @param maxZoom 最大缩放级别
     * @return 瓦片信息结果
     */
    boost::future<core_services::output::OutputResult> generateTiles(
        std::shared_ptr<core_services::GridData> gridData,
        const std::string& outputDirectory,
        const core_services::output::StyleOptions& style,
        int minZoom = 0, int maxZoom = 18);

    /**
     * @brief 并行生成瓦片金字塔（性能优化版本）
     * @param gridData 源数据
     * @param outputDirectory 瓦片输出目录
     * @param style 渲染样式
     * @param minZoom 最小缩放级别
     * @param maxZoom 最大缩放级别
     * @return 瓦片信息结果
     */
    boost::future<core_services::output::OutputResult> generateTilesParallel(
        std::shared_ptr<core_services::GridData> gridData,
        const std::string& outputDirectory,
        const core_services::output::StyleOptions& style,
        int minZoom = 0, int maxZoom = 18);

    /**
     * @brief 生成颜色图例
     * @param colorMap 颜色映射名称
     * @param minValue 最小值
     * @param maxValue 最大值
     * @param title 图例标题
     * @param outputPath 输出路径
     * @param width 图例宽度
     * @param height 图例高度
     * @return 生成的图例文件路径
     */
    boost::future<std::string> generateLegend(
        const std::string& colorMap,
        double minValue, double maxValue,
        const std::string& title,
        const std::string& outputPath,
        int width = 200, int height = 600);

    /**
     * @brief 生成等值线
     * @param gridData 网格数据
     * @param levels 等值线级别
     * @return 生成的等值线要素集合
     */
    std::shared_ptr<core_services::FeatureCollection> generateContours(
        std::shared_ptr<core_services::GridData> gridData,
        const std::vector<double>& levels);

    // === 性能监控和配置 ===
    
    /**
     * @brief 启用/禁用SIMD优化
     * @param enable 是否启用SIMD
     */
    void enableSIMDOptimization(bool enable = true) { useSIMDOptimization_ = enable; }
    
    /**
     * @brief 检查是否启用SIMD优化
     * @return 是否启用SIMD
     */
    bool isSIMDOptimizationEnabled() const { return useSIMDOptimization_ && simdManager_; }
    
    /**
     * @brief 获取性能统计
     * @return 性能报告字符串
     */
    std::string getPerformanceReport() const;

    // === 新增：高性能优化方法 ===
    
    /**
     * @brief SIMD优化的单瓦片生成
     * @param gridData 源数据
     * @param x 瓦片X坐标
     * @param y 瓦片Y坐标
     * @param z 缩放级别
     * @param style 渲染样式
     * @param outputPath 输出路径
     * @param tileBounds 瓦片边界
     * @return 瓦片文件路径
     */
    std::string generateSingleTileOptimized(
        std::shared_ptr<core_services::GridData> gridData,
        int x, int y, int z,
        const core_services::output::StyleOptions& style,
        const std::string& outputPath,
        const core_services::BoundingBox& tileBounds);
    
    /**
     * @brief SIMD优化的网格数据重采样
     * @param gridData 源网格数据
     * @param targetBounds 目标边界
     * @param targetWidth 目标宽度
     * @param targetHeight 目标高度
     * @param outputData 输出数据缓冲区
     */
    void resampleGridDataSIMD(
        std::shared_ptr<core_services::GridData> gridData,
        const core_services::BoundingBox& targetBounds,
        int targetWidth, int targetHeight,
        float* outputData);
        
    /**
     * @brief 标量版本的网格数据重采样（回退实现）
     * @param gridData 源网格数据
     * @param targetBounds 目标边界
     * @param targetWidth 目标宽度
     * @param targetHeight 目标高度
     * @param outputData 输出数据缓冲区
     */
    void resampleGridDataScalar(
        std::shared_ptr<core_services::GridData> gridData,
        const core_services::BoundingBox& targetBounds,
        int targetWidth, int targetHeight,
        float* outputData);

    // === GPU加速方法 ===
    
    /**
     * @brief 启用/禁用GPU优化
     * @param enable 是否启用GPU
     */
    void enableGPUOptimization(bool enable = true);
    
    /**
     * @brief 检查GPU是否可用
     * @return 如果GPU可用返回true
     */
    bool isGPUAvailable() const { return gpuAvailable_; }
    
    /**
     * @brief 检查是否启用GPU优化
     * @return 是否启用GPU
     */
    bool isGPUOptimizationEnabled() const { return useGPUOptimization_ && gpuAvailable_; }
    
    /**
     * @brief 设置GPU框架（用于外部注入）
     * @param gpuFramework GPU框架实例
     */
    void setGPUFramework(std::shared_ptr<common_utils::gpu::OSCEANGPUFramework> gpuFramework);
    
    /**
     * @brief GPU加速的渲染方法
     * @param gridData 要渲染的网格数据
     * @param outputPath 输出图像路径
     * @param style 渲染样式选项
     * @return 生成的图像文件路径
     */
    boost::future<std::string> renderToImageGPU(
        std::shared_ptr<core_services::GridData> gridData,
        const std::string& outputPath,
        const core_services::output::StyleOptions& style);
    
    /**
     * @brief GPU加速的瓦片生成
     * @param gridData 源数据
     * @param outputDirectory 瓦片输出目录
     * @param style 渲染样式
     * @param minZoom 最小缩放级别
     * @param maxZoom 最大缩放级别
     * @return 瓦片信息结果
     */
    boost::future<core_services::output::OutputResult> generateTilesGPU(
        std::shared_ptr<core_services::GridData> gridData,
        const std::string& outputDirectory,
        const core_services::output::StyleOptions& style,
        int minZoom = 0, int maxZoom = 18);
    
    /**
     * @brief GPU加速的等值线生成
     * @param gridData 网格数据
     * @param levels 等值线级别
     * @return 生成的等值线要素集合
     */
    std::shared_ptr<core_services::FeatureCollection> generateContoursGPU(
        std::shared_ptr<core_services::GridData> gridData,
        const std::vector<double>& levels);

private:
    // 基础成员变量
    std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> m_threadPool;
    std::shared_ptr<common_utils::simd::UnifiedSIMDManager> simdManager_;
    bool useSIMDOptimization_;
    
    // GPU相关成员变量
    bool useGPUOptimization_;
    bool gpuAvailable_;
    std::shared_ptr<common_utils::gpu::OSCEANGPUFramework> gpuFramework_;
    
#ifdef OSCEAN_CUDA_ENABLED
    std::unique_ptr<output_generation::gpu::GPUVisualizationEngine> gpuEngine_;
    std::shared_ptr<output_generation::gpu::IGPUColorMapper> gpuColorMapper_;
    std::unique_ptr<output_generation::gpu::IGPUTileGenerator> gpuTileGenerator_;
#endif

    // === 核心渲染方法 ===
    
    /**
     * @brief 将数据值映射到颜色
     * @param values 数据值数组
     * @param colorMap 颜色映射名称
     * @param minValue 最小值
     * @param maxValue 最大值
     * @return RGBA颜色数组
     */
    std::vector<uint32_t> mapDataToColors(
        const std::vector<double>& values,
        const std::string& colorMap,
        double minValue, double maxValue);

    /**
     * @brief SIMD优化的颜色映射（高性能版本）
     * @param values 数据值数组（float格式）
     * @param colorMap 颜色映射名称
     * @param minValue 最小值
     * @param maxValue 最大值
     * @return RGBA颜色数组
     */
    std::vector<uint32_t> mapDataToColorsSIMD(
        const std::vector<float>& values,
        const std::string& colorMap,
        float minValue, float maxValue);

    /**
     * @brief 生成彩色图像数据
     * @param gridData 网格数据
     * @param style 样式选项
     * @return 图像像素数据（RGBA格式）
     */
    std::vector<uint32_t> generateImageData(
        std::shared_ptr<core_services::GridData> gridData,
        const core_services::output::StyleOptions& style);

    /**
     * @brief SIMD优化的图像数据生成
     * @param gridData 网格数据
     * @param style 样式选项
     * @return 图像像素数据（RGBA格式）
     */
    std::vector<uint32_t> generateImageDataSIMD(
        std::shared_ptr<core_services::GridData> gridData,
        const core_services::output::StyleOptions& style);

    /**
     * @brief 保存图像数据到文件
     * @param filename 输出文件名
     * @param imageData 图像数据 (RGBA)
     * @param width 图像宽度
     * @param height 图像高度
     */
    void saveImageToFile(const std::string& filename, 
                         const std::vector<uint8_t>& imageData, 
                         int width, int height);

    // === 瓦片生成方法 ===
    
    /**
     * @brief 计算瓦片边界
     * @param x 瓦片X坐标
     * @param y 瓦片Y坐标
     * @param z 缩放级别
     * @return 瓦片的地理边界（WGS84经纬度坐标）
     */
    core_services::BoundingBox calculateTileBounds(int x, int y, int z);

    /**
     * @brief 生成单个瓦片
     * @param gridData 源数据
     * @param x 瓦片X坐标
     * @param y 瓦片Y坐标
     * @param z 缩放级别
     * @param style 渲染样式
     * @param outputPath 输出路径
     * @return 瓦片文件路径
     */
    std::string generateSingleTile(
        std::shared_ptr<core_services::GridData> gridData,
        int x, int y, int z,
        const core_services::output::StyleOptions& style,
        const std::string& outputPath);

    // === 样式和颜色映射 ===
    
    /**
     * @brief 获取颜色映射
     * @param colorMapName 颜色映射名称
     * @return 颜色值数组（RGB）
     */
    std::vector<std::array<uint8_t, 3>> getColorMap(const std::string& colorMapName);

    /**
     * @brief 在颜色映射中插值
     * @param value 归一化值（0-1）
     * @param colorMap 颜色映射
     * @return RGB颜色
     */
    std::array<uint8_t, 3> interpolateColor(double value, const std::vector<std::array<uint8_t, 3>>& colorMap);

    /**
     * @brief 绘制图例刻度和标签
     * @param imageData 图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param minValue 最小值
     * @param maxValue 最大值
     * @param numTicks 刻度数量
     */
    void drawLegendTicks(
        std::vector<uint8_t>& imageData,
        int width, int height,
        double minValue, double maxValue,
        int numTicks = 5);

    /**
     * @brief 绘制图例标题
     * @param imageData 图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param title 标题文本
     */
    void drawLegendTitle(
        std::vector<uint8_t>& imageData,
        int width, int height,
        const std::string& title);

    // === 工具方法 ===
    
    /**
     * @brief 检查格式是否为有效的可视化格式
     * @param format 格式名称
     * @return 是否有效
     */
    bool isValidVisualizationFormat(const std::string& format);

    /**
     * @brief 数据统计结构
     */
    struct DataStatistics {
        double minValue;
        double maxValue;
        double meanValue;
        double stdDev;
    };

    /**
     * @brief 计算数据统计信息
     * @param gridData 网格数据
     * @return 统计信息
     */
    DataStatistics calculateDataStatistics(std::shared_ptr<core_services::GridData> gridData);

    /**
     * @brief SIMD优化的统计计算
     * @param gridData 网格数据
     * @return 统计信息
     */
    DataStatistics calculateDataStatisticsSIMD(std::shared_ptr<core_services::GridData> gridData);

    /**
     * @brief SIMD优化的RGBA转换
     * @param imageData 图像数据（uint32_t格式）
     * @param rgbaData 输出的RGBA数据（uint8_t格式）
     */
    void convertToRGBASIMD(
        const std::vector<uint32_t>& imageData, 
        std::vector<uint8_t>& rgbaData);

    // === 缓存和内存管理 ===
    
    mutable std::map<std::string, std::vector<std::array<uint8_t, 3>>> colorMapCache_;
    mutable std::map<size_t, DataStatistics> statisticsCache_;
    
    // 性能计数器
    mutable std::atomic<size_t> renderCount_{0};
    mutable std::atomic<double> totalRenderTime_{0.0};
    mutable std::atomic<size_t> simdOperationCount_{0};

    // === GPU内部方法 ===
    
    /**
     * @brief 初始化GPU组件
     */
    void initializeGPUComponents();
    
    /**
     * @brief GPU优化的颜色映射
     * @param values 数据值数组
     * @param colorMap 颜色映射名称
     * @param minValue 最小值
     * @param maxValue 最大值
     * @return RGBA颜色数组
     */
    std::vector<uint32_t> mapDataToColorsGPU(
        const std::vector<float>& values,
        const std::string& colorMap,
        float minValue, float maxValue);
    
    /**
     * @brief GPU优化的图像数据生成
     * @param gridData 网格数据
     * @param style 样式选项
     * @return 图像像素数据（RGBA格式）
     */
    std::vector<uint32_t> generateImageDataGPU(
        std::shared_ptr<core_services::GridData> gridData,
        const core_services::output::StyleOptions& style);
    
    /**
     * @brief GPU优化的单瓦片生成
     * @param gridData 源数据
     * @param x 瓦片X坐标
     * @param y 瓦片Y坐标
     * @param z 缩放级别
     * @param style 渲染样式
     * @param outputPath 输出路径
     * @param tileBounds 瓦片边界
     * @return 瓦片文件路径
     */
    std::string generateSingleTileGPU(
        std::shared_ptr<core_services::GridData> gridData,
        int x, int y, int z,
        const core_services::output::StyleOptions& style,
        const std::string& outputPath,
        const core_services::BoundingBox& tileBounds);

    // === 私有成员变量 ===
    // 字体渲染器
    std::unique_ptr<FontRenderer> m_fontRenderer;
};

} // namespace output
} // namespace oscean 