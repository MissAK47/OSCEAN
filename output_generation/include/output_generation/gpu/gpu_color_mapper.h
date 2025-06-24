/**
 * @file gpu_color_mapper.h
 * @brief GPU颜色映射器详细接口
 */

#pragma once

#include "gpu_visualization_engine.h"
#include <array>

namespace oscean::output_generation::gpu {

/**
 * @brief 颜色映射查找表
 */
struct ColorLUT {
    static constexpr int LUT_SIZE = 256;
    std::array<float, LUT_SIZE * 4> data;  // RGBA格式
    
    void loadColormap(const std::string& name);
};

/**
 * @brief GPU颜色映射配置
 */
struct GPUColorMapperConfig {
    // 性能选项
    bool useTextureMemory = true;          // 使用纹理内存加速
    bool enableSIMD = true;                // 启用SIMD优化
    int threadsPerBlock = 256;             // 每个块的线程数
    
    // 质量选项
    bool enableDithering = false;          // 启用抖动
    bool enableAntialiasing = false;       // 启用抗锯齿
    float gamma = 1.0f;                    // 伽马校正
};

/**
 * @brief 基础GPU颜色映射器实现
 */
class GPUColorMapperBase : public IGPUColorMapper {
public:
    GPUColorMapperBase();
    virtual ~GPUColorMapperBase();
    
    // IGPUColorMapper接口实现
    void setParameters(const GPUColorMappingParams& params) override;
    std::vector<std::string> getSupportedColormaps() const override;
    
    // IGPUAlgorithm接口实现
    std::string getAlgorithmName() const override { return "GPUColorMapper"; }
    std::string getVersion() const override { return "1.0.0"; }
    
    /**
     * @brief 设置配置
     */
    void setConfig(const GPUColorMapperConfig& config);
    
protected:
    GPUColorMappingParams m_params;
    GPUColorMapperConfig m_config;
    ColorLUT m_colorLUT;
    
    /**
     * @brief 预处理输入数据
     */
    virtual GPUError preprocessData(const GridData& input);
    
    /**
     * @brief 在CPU上计算最小最大值（作为GPU的后备方案）
     */
    void computeMinMaxCPU(const GridData& input);
    
    /**
     * @brief 上传数据到GPU
     */
    virtual GPUError uploadToGPU(const GridData& input, void* devicePtr) = 0;
    
    /**
     * @brief 执行颜色映射核函数
     */
    virtual GPUError executeKernel(
        void* inputDevice,
        void* outputDevice,
        int width,
        int height,
        const GPUExecutionContext& context) = 0;
    
    /**
     * @brief 下载结果从GPU
     */
    virtual GPUError downloadFromGPU(void* devicePtr, GPUVisualizationResult& result) = 0;
};

/**
 * @brief 高级GPU颜色映射功能
 */
class AdvancedGPUColorMapper : public GPUColorMapperBase {
public:
    /**
     * @brief 设置自定义颜色映射
     */
    void setCustomColormap(const std::vector<std::array<float, 4>>& colors);
    
    /**
     * @brief 启用多变量可视化
     */
    void enableMultiVariable(bool enable);
    
    /**
     * @brief 设置数据变换函数
     */
    enum class DataTransform {
        LINEAR,
        LOG,
        SQRT,
        POWER,
        CUSTOM
    };
    void setDataTransform(DataTransform transform, float param = 1.0f);
    
    /**
     * @brief 设置等值线叠加
     */
    void setContourOverlay(bool enable, int levels = 10);
    
protected:
    bool m_multiVariable = false;
    DataTransform m_transform = DataTransform::LINEAR;
    float m_transformParam = 1.0f;
    bool m_contourOverlay = false;
    int m_contourLevels = 10;
};

// 工厂函数声明
std::unique_ptr<IGPUColorMapper> createCUDAColorMapper();
std::unique_ptr<IGPUColorMapper> createCUDAColorMapperReal();

} // namespace oscean::output_generation::gpu 