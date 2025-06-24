/**
 * @file gpu_tile_generator_enhanced.cpp
 * @brief 增强的GPU瓦片生成器实现，集成图像重采样
 */

#include "output_generation/gpu/gpu_tile_generator.h"
#include "output_generation/gpu/gpu_tile_types.h"
#include "output_generation/gpu/gpu_visualization_engine.h"
#include "output_generation/gpu/gpu_grid_data_helper.h"
#include "core_services/common_data_types.h"
#include "engines/gpu/color_maps.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <cmath>
#include <memory>

#ifdef OSCEAN_CUDA_ENABLED
#include <cuda_runtime.h>

// CUDA外部函数声明
extern "C" {
    cudaError_t launchTileGeneration(
        const float* d_gridData,
        uint8_t* d_tileData,
        int tileX, int tileY, int zoomLevel,
        int tileSize,
        int gridWidth, int gridHeight,
        float minLon, float maxLon,
        float minLat, float maxLat,
        float minValue, float maxValue,
        cudaStream_t stream);
        
    cudaError_t uploadColorLUT(const float* colorLUT, size_t size);
    
    // 图像重采样函数
    cudaError_t resampleBilinearGPU(
        const float* d_input,
        float* d_output,
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        cudaStream_t stream);
        
    cudaError_t resampleBicubicGPU(
        const float* d_input,
        float* d_output,
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        cudaStream_t stream);
        
    cudaError_t resampleLanczosGPU(
        const float* d_input,
        float* d_output,
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        int radius,
        cudaStream_t stream);
}
#endif

namespace oscean::output_generation::gpu {

/**
 * @brief 重采样方法枚举
 */
enum class ResampleMethod {
    NONE,       // 不重采样
    BILINEAR,   // 双线性
    BICUBIC,    // 双三次
    LANCZOS     // Lanczos
};

/**
 * @brief 增强的GPU瓦片生成器实现
 */
class EnhancedTileGenerator : public IGPUTileGenerator {
private:
    int m_deviceId;
    GPUTileParams m_tileParams;
    GPUTileGenerationParams m_params;
    ResampleMethod m_resampleMethod;
    int m_lanczosRadius;
    std::shared_ptr<spdlog::logger> m_logger;
    
    // 缓存的GPU资源
    float* m_d_resampledData;
    size_t m_resampledDataSize;
    
public:
    explicit EnhancedTileGenerator(int deviceId = 0) 
        : m_deviceId(deviceId),
          m_resampleMethod(ResampleMethod::BILINEAR),
          m_lanczosRadius(3),
          m_d_resampledData(nullptr),
          m_resampledDataSize(0),
          m_logger(spdlog::default_logger()) {
        
        #ifdef OSCEAN_CUDA_ENABLED
        cudaSetDevice(m_deviceId);
        #endif
    }
    
    ~EnhancedTileGenerator() {
        #ifdef OSCEAN_CUDA_ENABLED
        if (m_d_resampledData) {
            cudaFree(m_d_resampledData);
        }
        #endif
    }
    
    /**
     * @brief 设置重采样方法
     */
    void setResampleMethod(ResampleMethod method, int lanczosRadius = 3) {
        m_resampleMethod = method;
        m_lanczosRadius = lanczosRadius;
    }
    
    /**
     * @brief 设置瓦片生成参数
     */
    void setParameters(const GPUTileGenerationParams& params) override {
        m_params = params;
        m_tileParams.tileSize = params.tileSize;
        m_tileParams.colormap = "viridis";
        m_tileParams.autoScale = true;
    }
    
    /**
     * @brief 计算指定缩放级别的瓦片数量
     */
    std::pair<int, int> calculateTileCount(int zoomLevel, const GridData& data) const override {
        int tilesPerDim = 1 << zoomLevel;
        return {tilesPerDim, tilesPerDim};
    }
    
    /**
     * @brief 执行异步瓦片生成
     */
    boost::future<GPUAlgorithmResult<std::vector<GPUVisualizationResult>>> executeAsync(
        const std::shared_ptr<GridData>& input,
        const GPUExecutionContext& context) override {
        
        return boost::async(boost::launch::async, [this, input, context]() {
            return generateTilesWithResampling(input, context);
        });
    }
    
    /**
     * @brief 获取支持的API
     */
    std::vector<ComputeAPI> getSupportedAPIs() const override {
        return {ComputeAPI::CUDA};
    }
    
    /**
     * @brief 检查设备支持
     */
    bool supportsDevice(const GPUDeviceInfo& device) const override {
        return device.vendor == GPUVendor::NVIDIA;
    }
    
    /**
     * @brief 估算内存需求
     */
    size_t estimateMemoryRequirement(const std::shared_ptr<GridData>& input) const override {
        size_t gridSize = input->getWidth() * input->getHeight() * sizeof(float);
        size_t tileSize = m_params.tileSize * m_params.tileSize * 4;
        
        // 如果需要重采样，还需要额外的缓冲区
        if (m_resampleMethod != ResampleMethod::NONE) {
            // 假设最大需要2倍的网格大小用于重采样
            gridSize *= 2;
        }
        
        return gridSize + tileSize + (256 * 4 * sizeof(float));
    }
    
    /**
     * @brief 获取算法名称
     */
    std::string getAlgorithmName() const override {
        return "EnhancedGPUTileGenerator";
    }
    
    /**
     * @brief 获取算法版本
     */
    std::string getVersion() const override {
        return "2.0.0";
    }
    
private:
    /**
     * @brief 执行重采样
     */
    cudaError_t performResampling(
        const float* d_input,
        float* d_output,
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        cudaStream_t stream) {
        
        #ifdef OSCEAN_CUDA_ENABLED
        switch (m_resampleMethod) {
            case ResampleMethod::BILINEAR:
                return resampleBilinearGPU(d_input, d_output, 
                                         srcWidth, srcHeight, 
                                         dstWidth, dstHeight, stream);
                
            case ResampleMethod::BICUBIC:
                return resampleBicubicGPU(d_input, d_output, 
                                        srcWidth, srcHeight, 
                                        dstWidth, dstHeight, stream);
                
            case ResampleMethod::LANCZOS:
                return resampleLanczosGPU(d_input, d_output, 
                                        srcWidth, srcHeight, 
                                        dstWidth, dstHeight, 
                                        m_lanczosRadius, stream);
                
            default:
                return cudaSuccess;
        }
        #else
        return cudaSuccess;
        #endif
    }
    
    /**
     * @brief 计算重采样后的尺寸
     */
    std::pair<int, int> calculateResampledSize(
        int srcWidth, int srcHeight,
        int zoomLevel) const {
        
        // 根据缩放级别计算目标分辨率
        int targetResolution = m_params.tileSize * (1 << zoomLevel);
        
        // 保持宽高比
        double aspectRatio = static_cast<double>(srcWidth) / srcHeight;
        int dstWidth, dstHeight;
        
        if (aspectRatio > 1.0) {
            dstWidth = targetResolution;
            dstHeight = static_cast<int>(targetResolution / aspectRatio);
        } else {
            dstHeight = targetResolution;
            dstWidth = static_cast<int>(targetResolution * aspectRatio);
        }
        
        // 确保至少和原始大小一样
        dstWidth = std::max(dstWidth, srcWidth);
        dstHeight = std::max(dstHeight, srcHeight);
        
        return {dstWidth, dstHeight};
    }
    
    /**
     * @brief 生成带重采样的瓦片
     */
    GPUAlgorithmResult<std::vector<GPUVisualizationResult>> generateTilesWithResampling(
        const std::shared_ptr<GridData>& input,
        const GPUExecutionContext& context) {
        
        GPUAlgorithmResult<std::vector<GPUVisualizationResult>> result;
        std::vector<GPUVisualizationResult> tileResults;
        
        #ifdef OSCEAN_CUDA_ENABLED
        try {
            auto totalStart = std::chrono::high_resolution_clock::now();
            
            // 获取原始数据
            const float* hostData = GPUGridDataHelper::getFloat32DataPtr(input);
            int srcWidth = input->getWidth();
            int srcHeight = input->getHeight();
            size_t srcDataSize = srcWidth * srcHeight * sizeof(float);
            
            // 计算数据范围
            float minValue, maxValue;
            GPUGridDataHelper::computeMinMax(input, minValue, maxValue);
            
            // 上传颜色查找表
            const auto& colorMap = ColorMapManager::getInstance().getColorMap(m_tileParams.colormap);
            uploadColorLUT(reinterpret_cast<const float*>(colorMap.data.data()), 256 * 4);
            
            // 分配原始数据GPU内存
            float* d_srcData = nullptr;
            cudaMalloc(&d_srcData, srcDataSize);
            cudaMemcpy(d_srcData, hostData, srcDataSize, cudaMemcpyHostToDevice);
            
            // 创建CUDA流
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            
            // 决定是否需要重采样
            float* d_processedData = d_srcData;
            int processedWidth = srcWidth;
            int processedHeight = srcHeight;
            
            if (m_resampleMethod != ResampleMethod::NONE) {
                // 计算重采样后的尺寸
                auto [dstWidth, dstHeight] = calculateResampledSize(
                    srcWidth, srcHeight, m_params.zoomLevel);
                
                // 只在需要时重采样
                if (dstWidth != srcWidth || dstHeight != srcHeight) {
                    size_t dstDataSize = dstWidth * dstHeight * sizeof(float);
                    
                    // 重新分配缓冲区（如果需要）
                    if (dstDataSize > m_resampledDataSize) {
                        if (m_d_resampledData) {
                            cudaFree(m_d_resampledData);
                        }
                        cudaMalloc(&m_d_resampledData, dstDataSize);
                        m_resampledDataSize = dstDataSize;
                    }
                    
                    // 执行重采样
                    auto resampleStart = std::chrono::high_resolution_clock::now();
                    
                    cudaError_t err = performResampling(
                        d_srcData, m_d_resampledData,
                        srcWidth, srcHeight,
                        dstWidth, dstHeight,
                        stream);
                    
                    cudaStreamSynchronize(stream);
                    
                    auto resampleEnd = std::chrono::high_resolution_clock::now();
                    double resampleTime = std::chrono::duration<double, std::milli>(
                        resampleEnd - resampleStart).count();
                    
                    if (err != cudaSuccess) {
                        throw std::runtime_error("Resampling failed");
                    }
                    
                    m_logger->info("Resampled from {}x{} to {}x{} in {:.2f}ms",
                                 srcWidth, srcHeight, dstWidth, dstHeight, resampleTime);
                    
                    d_processedData = m_d_resampledData;
                    processedWidth = dstWidth;
                    processedHeight = dstHeight;
                }
            }
            
            // 计算瓦片数量
            auto [tilesX, tilesY] = calculateTileCount(m_params.zoomLevel, *input);
            
            // 生成所有瓦片
            for (int y = 0; y < tilesY; ++y) {
                for (int x = 0; x < tilesX; ++x) {
                    GPUVisualizationResult vizResult;
                    vizResult.width = m_params.tileSize;
                    vizResult.height = m_params.tileSize;
                    vizResult.channels = 4;
                    vizResult.format = "RGBA";
                    
                    size_t tileDataSize = m_params.tileSize * m_params.tileSize * 4;
                    vizResult.imageData.resize(tileDataSize);
                    
                    // 分配瓦片GPU内存
                    uint8_t* d_tileData = nullptr;
                    cudaMalloc(&d_tileData, tileDataSize);
                    
                    // 生成瓦片
                    const auto& extent = input->getDefinition().extent;
                    
                    auto tileStart = std::chrono::high_resolution_clock::now();
                    
                    launchTileGeneration(
                        d_processedData,
                        d_tileData,
                        x, y, m_params.zoomLevel,
                        m_params.tileSize,
                        processedWidth,
                        processedHeight,
                        static_cast<float>(extent.minX),
                        static_cast<float>(extent.maxX),
                        static_cast<float>(extent.minY),
                        static_cast<float>(extent.maxY),
                        minValue, maxValue,
                        stream);
                    
                    cudaStreamSynchronize(stream);
                    
                    auto tileEnd = std::chrono::high_resolution_clock::now();
                    
                    // 下载瓦片数据
                    cudaMemcpy(vizResult.imageData.data(), d_tileData, 
                             tileDataSize, cudaMemcpyDeviceToHost);
                    
                    cudaFree(d_tileData);
                    
                    vizResult.stats.gpuTime = std::chrono::duration<double, std::milli>(
                        tileEnd - tileStart).count();
                    vizResult.stats.totalTime = vizResult.stats.gpuTime;
                    
                    tileResults.push_back(std::move(vizResult));
                }
            }
            
            // 清理
            cudaFree(d_srcData);
            cudaStreamDestroy(stream);
            
            auto totalEnd = std::chrono::high_resolution_clock::now();
            double totalTime = std::chrono::duration<double, std::milli>(
                totalEnd - totalStart).count();
            
            m_logger->info("Generated {} tiles with {} resampling in {:.2f}ms",
                         tileResults.size(),
                         m_resampleMethod == ResampleMethod::NONE ? "no" :
                         m_resampleMethod == ResampleMethod::BILINEAR ? "bilinear" :
                         m_resampleMethod == ResampleMethod::BICUBIC ? "bicubic" : "lanczos",
                         totalTime);
            
            result.success = true;
            result.data = std::move(tileResults);
            result.error = GPUError::SUCCESS;
            
        } catch (const std::exception& e) {
            result.success = false;
            result.error = GPUError::KERNEL_LAUNCH_FAILED;
            result.errorMessage = e.what();
            m_logger->error("Enhanced tile generation failed: {}", e.what());
        }
        #else
        result.success = false;
        result.error = GPUError::NOT_SUPPORTED;
        result.errorMessage = "CUDA not enabled";
        #endif
        
        return result;
    }
};

// 工厂函数实现
std::unique_ptr<IGPUTileGenerator> createGPUTileGenerator(int deviceId) {
    return std::make_unique<EnhancedTileGenerator>(deviceId);
}

} // namespace oscean::output_generation::gpu 