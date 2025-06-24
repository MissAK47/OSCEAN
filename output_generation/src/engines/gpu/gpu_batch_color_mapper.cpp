/**
 * @file gpu_batch_color_mapper.cpp
 * @brief GPU批处理颜色映射器实现
 */

#include "output_generation/gpu/gpu_color_mapper.h"
#include "common_utils/gpu/oscean_gpu_framework.h"
#include "color_maps.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <vector>

#ifdef OSCEAN_CUDA_ENABLED
#include <cuda_runtime.h>

// CUDA外部函数声明
extern "C" {
    cudaError_t launchBatchColorMapping(
        const float** d_inputs,
        uint8_t** d_outputs,
        const int* d_widths,
        const int* d_heights,
        const float* d_minValues,
        const float* d_maxValues,
        int batchSize,
        cudaStream_t stream);
        
    cudaError_t uploadColorLUT(const float* colorLUT, size_t size);
}
#endif

namespace oscean::output_generation::gpu {

using namespace oscean::common_utils::gpu;

/**
 * @brief 批处理颜色映射结果
 */
struct BatchColorMappingResult {
    std::vector<GPUVisualizationResult> results;
    double totalGpuTime = 0.0;
    double totalTransferTime = 0.0;
    double totalTime = 0.0;
};

/**
 * @brief GPU批处理颜色映射器实现
 */
class GPUBatchColorMapper {
public:
    GPUBatchColorMapper() : m_logger(spdlog::default_logger()) {
        // 使用默认logger，避免创建新的logger
    }
    
    ~GPUBatchColorMapper() {
#ifdef OSCEAN_CUDA_ENABLED
        cleanupCUDA();
#endif
    }
    
    /**
     * @brief 批量处理多个图像
     */
    BatchColorMappingResult processBatch(
        const std::vector<std::shared_ptr<GridData>>& gridDataList,
        const GPUColorMappingParams& params) {
        
        BatchColorMappingResult batchResult;
        auto startTime = std::chrono::high_resolution_clock::now();
        
#ifdef OSCEAN_CUDA_ENABLED
        if (!m_cudaInitialized) {
            initializeCUDA();
            uploadColorLUT(params.colormap);
        }
        
        int batchSize = static_cast<int>(gridDataList.size());
        
        // 准备批处理数据
        std::vector<const float*> h_inputs(batchSize);
        std::vector<uint8_t*> h_outputs(batchSize);
        std::vector<int> h_widths(batchSize);
        std::vector<int> h_heights(batchSize);
        std::vector<float> h_minValues(batchSize);
        std::vector<float> h_maxValues(batchSize);
        
        // 分配设备内存
        size_t totalInputSize = 0;
        size_t totalOutputSize = 0;
        std::vector<size_t> inputSizes(batchSize);
        std::vector<size_t> outputSizes(batchSize);
        
        for (int i = 0; i < batchSize; ++i) {
            auto& gridData = gridDataList[i];
            int width = gridData->getDefinition().cols;
            int height = gridData->getDefinition().rows;
            
            h_widths[i] = width;
            h_heights[i] = height;
            
            inputSizes[i] = width * height * sizeof(float);
            outputSizes[i] = width * height * 4 * sizeof(uint8_t);
            
            totalInputSize += inputSizes[i];
            totalOutputSize += outputSizes[i];
            
            // 计算最小最大值
            if (params.autoScale) {
                auto minmax = calculateMinMax(gridData);
                h_minValues[i] = minmax.first;
                h_maxValues[i] = minmax.second;
            } else {
                h_minValues[i] = static_cast<float>(params.minValue);
                h_maxValues[i] = static_cast<float>(params.maxValue);
            }
        }
        
        // 分配连续的设备内存块
        float* d_inputBlock;
        uint8_t* d_outputBlock;
        cudaMalloc(&d_inputBlock, totalInputSize);
        cudaMalloc(&d_outputBlock, totalOutputSize);
        
        // 设置指针数组
        std::vector<float*> d_inputs(batchSize);
        std::vector<uint8_t*> d_outputs(batchSize);
        
        size_t inputOffset = 0;
        size_t outputOffset = 0;
        
        // 传输数据到GPU
        auto transferStart = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < batchSize; ++i) {
            d_inputs[i] = d_inputBlock + inputOffset / sizeof(float);
            d_outputs[i] = d_outputBlock + outputOffset;
            
            // 复制输入数据
            cudaMemcpy(d_inputs[i], gridDataList[i]->getData().data(),
                      inputSizes[i], cudaMemcpyHostToDevice);
            
            h_inputs[i] = d_inputs[i];
            h_outputs[i] = d_outputs[i];
            
            inputOffset += inputSizes[i];
            outputOffset += outputSizes[i];
        }
        
        // 复制元数据到设备
        float** d_inputPtrs;
        uint8_t** d_outputPtrs;
        int* d_widths;
        int* d_heights;
        float* d_minValues;
        float* d_maxValues;
        
        cudaMalloc(&d_inputPtrs, batchSize * sizeof(float*));
        cudaMalloc(&d_outputPtrs, batchSize * sizeof(uint8_t*));
        cudaMalloc(&d_widths, batchSize * sizeof(int));
        cudaMalloc(&d_heights, batchSize * sizeof(int));
        cudaMalloc(&d_minValues, batchSize * sizeof(float));
        cudaMalloc(&d_maxValues, batchSize * sizeof(float));
        
        cudaMemcpy(d_inputPtrs, h_inputs.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outputPtrs, h_outputs.data(), batchSize * sizeof(uint8_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_widths, h_widths.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_heights, h_heights.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_minValues, h_minValues.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_maxValues, h_maxValues.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice);
        
        auto transferEnd = std::chrono::high_resolution_clock::now();
        batchResult.totalTransferTime = std::chrono::duration<double, std::milli>(
            transferEnd - transferStart).count();
        
        // 执行批处理核函数
        auto gpuStart = std::chrono::high_resolution_clock::now();
        
        launchBatchColorMapping(
            (const float**)d_inputPtrs,
            d_outputPtrs,
            d_widths,
            d_heights,
            d_minValues,
            d_maxValues,
            batchSize,
            0);
        
        cudaDeviceSynchronize();
        
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        batchResult.totalGpuTime = std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart).count();
        
        // 复制结果回主机
        transferStart = std::chrono::high_resolution_clock::now();
        
        outputOffset = 0;
        for (int i = 0; i < batchSize; ++i) {
            GPUVisualizationResult result;
            result.width = h_widths[i];
            result.height = h_heights[i];
            result.channels = 4;
            result.format = "RGBA";
            result.imageData.resize(outputSizes[i]);
            
            cudaMemcpy(result.imageData.data(), d_outputs[i],
                      outputSizes[i], cudaMemcpyDeviceToHost);
            
            result.stats.gpuTime = batchResult.totalGpuTime / batchSize;
            result.stats.transferTime = batchResult.totalTransferTime / batchSize;
            result.stats.memoryUsed = inputSizes[i] + outputSizes[i];
            
            batchResult.results.push_back(std::move(result));
        }
        
        transferEnd = std::chrono::high_resolution_clock::now();
        batchResult.totalTransferTime += std::chrono::duration<double, std::milli>(
            transferEnd - transferStart).count();
        
        // 清理设备内存
        cudaFree(d_inputBlock);
        cudaFree(d_outputBlock);
        cudaFree(d_inputPtrs);
        cudaFree(d_outputPtrs);
        cudaFree(d_widths);
        cudaFree(d_heights);
        cudaFree(d_minValues);
        cudaFree(d_maxValues);
        
#else
        m_logger->warn("CUDA not enabled, batch processing falling back to CPU");
        // CPU后备实现...
#endif
        
        auto endTime = std::chrono::high_resolution_clock::now();
        batchResult.totalTime = std::chrono::duration<double, std::milli>(
            endTime - startTime).count();
        
        m_logger->info("Batch processing completed: {} images in {:.2f}ms (GPU: {:.2f}ms)",
                      batchSize, batchResult.totalTime, batchResult.totalGpuTime);
        
        return batchResult;
    }
    
private:
    std::shared_ptr<spdlog::logger> m_logger;
    bool m_cudaInitialized = false;
    
#ifdef OSCEAN_CUDA_ENABLED
    void initializeCUDA() {
        cudaSetDevice(0);
        m_cudaInitialized = true;
        m_logger->info("CUDA initialized for batch processing");
    }
    
    void cleanupCUDA() {
        if (m_cudaInitialized) {
            cudaDeviceReset();
            m_cudaInitialized = false;
        }
    }
    
    void uploadColorLUT(const std::string& colormap) {
        const auto& cmap = ColorMapManager::getInstance().getColorMap(colormap);
        ::uploadColorLUT(cmap.data.data(), cmap.data.size() * sizeof(float));
    }
    
    std::pair<float, float> calculateMinMax(const std::shared_ptr<GridData>& gridData) {
        const auto& data = gridData->getData();
        if (data.empty()) return {0.0f, 1.0f};
        
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        
        for (float val : data) {
            if (!std::isnan(val)) {
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
        
        if (minVal > maxVal) {
            minVal = 0.0f;
            maxVal = 1.0f;
        }
        
        return {minVal, maxVal};
    }
#endif
};

} // namespace oscean::output_generation::gpu 