/**
 * @file gpu_color_mapper.cpp
 * @brief GPU颜色映射器实现
 */

#include "output_generation/gpu/gpu_color_mapper.h"
#include "common_utils/gpu/oscean_gpu_framework.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <limits>

#ifdef OSCEAN_CUDA_ENABLED
#include <cuda_runtime.h>
// 声明外部CUDA函数
extern "C" cudaError_t computeMinMaxGPU(
    const float* d_data,
    size_t numElements,
    float* h_min,
    float* h_max);
#endif

namespace oscean::output_generation::gpu {

// 预定义的颜色映射方案
namespace {
    // Viridis颜色映射数据（简化版）
    const std::vector<std::array<float, 4>> VIRIDIS_COLORS = {
        {0.267f, 0.004f, 0.329f, 1.0f},
        {0.283f, 0.141f, 0.458f, 1.0f},
        {0.253f, 0.265f, 0.530f, 1.0f},
        {0.207f, 0.371f, 0.553f, 1.0f},
        {0.164f, 0.471f, 0.558f, 1.0f},
        {0.128f, 0.567f, 0.551f, 1.0f},
        {0.135f, 0.659f, 0.518f, 1.0f},
        {0.267f, 0.749f, 0.441f, 1.0f},
        {0.478f, 0.821f, 0.318f, 1.0f},
        {0.741f, 0.873f, 0.150f, 1.0f},
        {0.993f, 0.906f, 0.144f, 1.0f}
    };
    
    // Inferno颜色映射数据（简化版）
    const std::vector<std::array<float, 4>> INFERNO_COLORS = {
        {0.001f, 0.000f, 0.014f, 1.0f},
        {0.088f, 0.042f, 0.141f, 1.0f},
        {0.232f, 0.060f, 0.234f, 1.0f},
        {0.391f, 0.071f, 0.294f, 1.0f},
        {0.550f, 0.089f, 0.320f, 1.0f},
        {0.705f, 0.128f, 0.316f, 1.0f},
        {0.847f, 0.190f, 0.283f, 1.0f},
        {0.963f, 0.282f, 0.223f, 1.0f},
        {0.993f, 0.435f, 0.209f, 1.0f},
        {0.973f, 0.621f, 0.351f, 1.0f},
        {0.988f, 0.998f, 0.645f, 1.0f}
    };
}

// ColorLUT实现
void ColorLUT::loadColormap(const std::string& name) {
    const std::vector<std::array<float, 4>>* colors = nullptr;
    
    if (name == "viridis") {
        colors = &VIRIDIS_COLORS;
    } else if (name == "inferno") {
        colors = &INFERNO_COLORS;
    } else {
        // 默认使用灰度
        for (int i = 0; i < LUT_SIZE; ++i) {
            float gray = static_cast<float>(i) / (LUT_SIZE - 1);
            data[i * 4 + 0] = gray;
            data[i * 4 + 1] = gray;
            data[i * 4 + 2] = gray;
            data[i * 4 + 3] = 1.0f;
        }
        return;
    }
    
    // 插值生成完整的LUT
    if (colors) {
        int numColors = static_cast<int>(colors->size());
        for (int i = 0; i < LUT_SIZE; ++i) {
            float t = static_cast<float>(i) / (LUT_SIZE - 1) * (numColors - 1);
            int idx0 = static_cast<int>(t);
            int idx1 = std::min(idx0 + 1, numColors - 1);
            float frac = t - idx0;
            
            // 线性插值
            for (int c = 0; c < 4; ++c) {
                data[i * 4 + c] = (*colors)[idx0][c] * (1.0f - frac) + 
                                  (*colors)[idx1][c] * frac;
            }
        }
    }
}

// GPUColorMapperBase实现
GPUColorMapperBase::GPUColorMapperBase() {
    m_colorLUT.loadColormap("viridis");
}

GPUColorMapperBase::~GPUColorMapperBase() = default;

void GPUColorMapperBase::setParameters(const GPUColorMappingParams& params) {
    m_params = params;
    
    // 加载对应的颜色映射
    m_colorLUT.loadColormap(params.colormap);
}

std::vector<std::string> GPUColorMapperBase::getSupportedColormaps() const {
    return {
        "viridis", "inferno", "plasma", "magma", "cividis",
        "turbo", "hot", "cool", "spring", "summer",
        "autumn", "winter", "gray", "bone", "copper",
        "jet", "hsv", "rainbow"
    };
}

void GPUColorMapperBase::setConfig(const GPUColorMapperConfig& config) {
    m_config = config;
}

GPUError GPUColorMapperBase::preprocessData(const GridData& input) {
    // 检查输入数据
    if (input.getUnifiedBufferSize() == 0) {
        spdlog::error("Empty input data for color mapping");
        return GPUError::INVALID_KERNEL;
    }
    
    // 如果需要自动缩放，计算最小值和最大值
    if (m_params.autoScale) {
        #ifdef OSCEAN_CUDA_ENABLED
        // 获取数据信息
        const float* hostData = reinterpret_cast<const float*>(input.getUnifiedBuffer().data());
        size_t numElements = input.getDefinition().cols * input.getDefinition().rows;
        size_t dataSize = numElements * sizeof(float);
        
        // 分配GPU内存
        float* d_data = nullptr;
        cudaError_t err = cudaMalloc(&d_data, dataSize);
        if (err != cudaSuccess) {
            spdlog::warn("Failed to allocate GPU memory for min/max computation, falling back to CPU");
            // CPU fallback
            computeMinMaxCPU(input);
            return GPUError::SUCCESS;
        }
        
        // 上传数据到GPU
        err = cudaMemcpy(d_data, hostData, dataSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_data);
            spdlog::warn("Failed to upload data for min/max computation, falling back to CPU");
            // CPU fallback
            computeMinMaxCPU(input);
            return GPUError::SUCCESS;
        }
        
        // 在GPU上计算最小最大值
        float minValue, maxValue;
        err = computeMinMaxGPU(d_data, numElements, &minValue, &maxValue);
        
        // 清理GPU内存
        cudaFree(d_data);
        
        if (err == cudaSuccess) {
            m_params.minValue = minValue;
            m_params.maxValue = maxValue;
            spdlog::debug("GPU auto-scaling data: min={}, max={}", 
                         m_params.minValue, m_params.maxValue);
        } else {
            spdlog::warn("GPU min/max computation failed, falling back to CPU");
            // CPU fallback
            computeMinMaxCPU(input);
        }
        #else
        // 没有CUDA时使用CPU计算
        computeMinMaxCPU(input);
        #endif
    }
    
    return GPUError::SUCCESS;
}

void GPUColorMapperBase::computeMinMaxCPU(const GridData& input) {
    const float* data = reinterpret_cast<const float*>(input.getUnifiedBuffer().data());
    size_t numElements = input.getDefinition().cols * input.getDefinition().rows;
    
    if (numElements == 0) {
        m_params.minValue = 0.0f;
        m_params.maxValue = 1.0f;
        return;
    }
    
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < numElements; ++i) {
        float val = data[i];
        if (!std::isnan(val) && !std::isinf(val)) {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }
    
    // 处理特殊情况
    if (minVal == maxVal) {
        maxVal = minVal + 1.0f;
    }
    
    m_params.minValue = minVal;
    m_params.maxValue = maxVal;
    
    spdlog::debug("CPU auto-scaling data: min={}, max={}", 
                 m_params.minValue, m_params.maxValue);
}

// AdvancedGPUColorMapper实现
void AdvancedGPUColorMapper::setCustomColormap(const std::vector<std::array<float, 4>>& colors) {
    if (colors.empty()) {
        spdlog::warn("Empty custom colormap provided");
        return;
    }
    
    // 插值生成LUT
    int numColors = static_cast<int>(colors.size());
    for (int i = 0; i < ColorLUT::LUT_SIZE; ++i) {
        float t = static_cast<float>(i) / (ColorLUT::LUT_SIZE - 1) * (numColors - 1);
        int idx0 = static_cast<int>(t);
        int idx1 = std::min(idx0 + 1, numColors - 1);
        float frac = t - idx0;
        
        // 线性插值
        for (int c = 0; c < 4; ++c) {
            m_colorLUT.data[i * 4 + c] = colors[idx0][c] * (1.0f - frac) + 
                                          colors[idx1][c] * frac;
        }
    }
}

void AdvancedGPUColorMapper::enableMultiVariable(bool enable) {
    m_multiVariable = enable;
}

void AdvancedGPUColorMapper::setDataTransform(DataTransform transform, float param) {
    m_transform = transform;
    m_transformParam = param;
}

void AdvancedGPUColorMapper::setContourOverlay(bool enable, int levels) {
    m_contourOverlay = enable;
    m_contourLevels = levels;
}

} // namespace oscean::output_generation::gpu 