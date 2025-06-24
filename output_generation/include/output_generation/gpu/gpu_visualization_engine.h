/**
 * @file gpu_visualization_engine.h
 * @brief GPU加速的可视化引擎接口
 */

#pragma once

#include "common_utils/utilities/boost_config.h"
#include "common_utils/gpu/gpu_algorithm_interface.h"
#include "common_utils/gpu/gpu_performance_monitor.h"
#include "core_services/common_data_types.h"
#include "core_services/output/i_output_service.h"
#include <boost/thread/future.hpp>
#include <memory>
#include <vector>

namespace oscean::output_generation::gpu {

using namespace oscean::common_utils::gpu;
using namespace oscean::core_services;

/**
 * @brief GPU颜色映射参数
 */
struct GPUColorMappingParams {
    std::string colormap = "viridis";      // 颜色映射方案
    double minValue = 0.0;                 // 最小值
    double maxValue = 1.0;                 // 最大值
    bool autoScale = true;                 // 自动缩放
    float opacity = 1.0f;                  // 不透明度
    int outputFormat = 0;                  // 0=RGBA, 1=RGB, 2=Grayscale
};

/**
 * @brief GPU瓦片生成参数
 */
struct GPUTileGenerationParams {
    int zoomLevel = 0;                     // 缩放级别
    int tileSize = 256;                    // 瓦片大小
    std::string format = "PNG";            // 输出格式
    int quality = 95;                      // 质量(JPEG)
    bool parallelTiles = true;             // 并行生成瓦片
};

/**
 * @brief GPU可视化结果
 */
struct GPUVisualizationResult {
    std::vector<unsigned char> imageData;  // 图像数据
    int width = 0;                         // 宽度
    int height = 0;                        // 高度
    int channels = 0;                      // 通道数
    std::string format;                    // 格式
    
    // 性能统计
    struct Stats {
        double gpuTime = 0.0;              // GPU处理时间(ms)
        double transferTime = 0.0;         // 数据传输时间(ms)
        double totalTime = 0.0;            // 总时间(ms)
        size_t memoryUsed = 0;             // 使用的GPU内存
    } stats;
};

/**
 * @brief GPU颜色映射算法
 */
class IGPUColorMapper : public IGPUAlgorithm<std::shared_ptr<GridData>, GPUVisualizationResult> {
public:
    virtual ~IGPUColorMapper() = default;
    
    /**
     * @brief 设置颜色映射参数
     */
    virtual void setParameters(const GPUColorMappingParams& params) = 0;
    
    /**
     * @brief 获取支持的颜色映射方案
     */
    virtual std::vector<std::string> getSupportedColormaps() const = 0;
};

/**
 * @brief GPU瓦片生成算法
 */
class IGPUTileGenerator : public IGPUAlgorithm<std::shared_ptr<GridData>, std::vector<GPUVisualizationResult>> {
public:
    virtual ~IGPUTileGenerator() = default;
    
    /**
     * @brief 设置瓦片生成参数
     */
    virtual void setParameters(const GPUTileGenerationParams& params) = 0;
    
    /**
     * @brief 计算指定缩放级别的瓦片数量
     */
    virtual std::pair<int, int> calculateTileCount(int zoomLevel, const GridData& data) const = 0;
};

/**
 * @brief GPU可视化引擎
 */
class GPUVisualizationEngine {
public:
    GPUVisualizationEngine();
    ~GPUVisualizationEngine();
    
    /**
     * @brief 初始化GPU可视化引擎
     */
    bool initialize();
    
    /**
     * @brief 执行颜色映射
     */
    boost::future<GPUVisualizationResult> applyColorMapping(
        std::shared_ptr<GridData> gridData,
        const GPUColorMappingParams& params);
    
    /**
     * @brief 生成瓦片
     */
    boost::future<std::vector<GPUVisualizationResult>> generateTiles(
        std::shared_ptr<GridData> gridData,
        const GPUTileGenerationParams& params);
    
    /**
     * @brief 获取GPU设备信息
     */
    std::vector<GPUDeviceInfo> getAvailableDevices() const;
    
    /**
     * @brief 设置使用的GPU设备
     */
    bool setDevice(int deviceId);
    
    /**
     * @brief 获取性能监控器
     */
    IGPUPerformanceMonitor* getPerformanceMonitor() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief GPU可视化算法工厂
 */
class GPUVisualizationFactory {
public:
    /**
     * @brief 创建颜色映射器
     */
    static std::unique_ptr<IGPUColorMapper> createColorMapper(ComputeAPI api);
    
    /**
     * @brief 创建瓦片生成器
     */
    static std::unique_ptr<IGPUTileGenerator> createTileGenerator(ComputeAPI api);
    
    /**
     * @brief 根据设备创建最优颜色映射器
     */
    static std::unique_ptr<IGPUColorMapper> createOptimalColorMapper(const GPUDeviceInfo& device);
    
    /**
     * @brief 根据设备创建最优瓦片生成器
     */
    static std::unique_ptr<IGPUTileGenerator> createOptimalTileGenerator(const GPUDeviceInfo& device);
};

} // namespace oscean::output_generation::gpu 