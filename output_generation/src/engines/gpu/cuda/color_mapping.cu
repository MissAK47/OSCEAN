/**
 * @file color_mapping.cu
 * @brief CUDA颜色映射核函数实现
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace oscean {
namespace output_generation {
namespace gpu {
namespace cuda {

// 设备端颜色查找表（纹理内存）
__constant__ float d_colorLUT[256 * 4];  // RGBA格式

/**
 * @brief 线性插值函数
 */
__device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

/**
 * @brief 数据变换函数
 */
__device__ float applyTransform(float value, int transformType, float param) {
    switch (transformType) {
        case 0: // LINEAR
            return value;
        case 1: // LOG
            return (value > 0.0f) ? logf(value + 1.0f) : 0.0f;
        case 2: // SQRT
            return (value >= 0.0f) ? sqrtf(value) : 0.0f;
        case 3: // POWER
            return powf(value, param);
        default:
            return value;
    }
}

/**
 * @brief 基础颜色映射核函数
 * @param input 输入数据
 * @param output 输出RGBA数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param minValue 最小值
 * @param maxValue 最大值
 * @param nanColor NaN值的颜色
 */
__global__ void colorMappingKernel(
    const float* input,
    uint8_t* output,
    int width,
    int height,
    float minValue,
    float maxValue,
    uint32_t nanColor) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = input[idx];
    
    // 处理NaN值
    if (isnan(value)) {
        uint32_t* outPtr = reinterpret_cast<uint32_t*>(output);
        outPtr[idx] = nanColor;
        return;
    }
    
    // 归一化到[0, 1]
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = fmaxf(0.0f, fminf(1.0f, normalizedValue));
    
    // 映射到LUT索引
    float lutIndex = normalizedValue * 255.0f;
    int index0 = __float2int_rd(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    // 从LUT中获取颜色并插值
    float r = lerp(d_colorLUT[index0 * 4 + 0], d_colorLUT[index1 * 4 + 0], frac);
    float g = lerp(d_colorLUT[index0 * 4 + 1], d_colorLUT[index1 * 4 + 1], frac);
    float b = lerp(d_colorLUT[index0 * 4 + 2], d_colorLUT[index1 * 4 + 2], frac);
    float a = lerp(d_colorLUT[index0 * 4 + 3], d_colorLUT[index1 * 4 + 3], frac);
    
    // 转换为uint8_t并写入输出
    output[idx * 4 + 0] = __float2uint_rn(r * 255.0f);
    output[idx * 4 + 1] = __float2uint_rn(g * 255.0f);
    output[idx * 4 + 2] = __float2uint_rn(b * 255.0f);
    output[idx * 4 + 3] = __float2uint_rn(a * 255.0f);
}

/**
 * @brief 高级颜色映射核函数（支持数据变换和伽马校正）
 */
__global__ void advancedColorMappingKernel(
    const float* input,
    uint8_t* output,
    int width,
    int height,
    float minValue,
    float maxValue,
    int transformType,
    float transformParam,
    float gamma,
    uint32_t nanColor) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = input[idx];
    
    // 处理NaN值
    if (isnan(value)) {
        uint32_t* outPtr = reinterpret_cast<uint32_t*>(output);
        outPtr[idx] = nanColor;
        return;
    }
    
    // 应用数据变换
    value = applyTransform(value, transformType, transformParam);
    
    // 归一化
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = fmaxf(0.0f, fminf(1.0f, normalizedValue));
    
    // 应用伽马校正
    if (gamma != 1.0f) {
        normalizedValue = powf(normalizedValue, gamma);
    }
    
    // 映射到LUT索引
    float lutIndex = normalizedValue * 255.0f;
    int index0 = __float2int_rd(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    // 从LUT中获取颜色并插值
    float r = lerp(d_colorLUT[index0 * 4 + 0], d_colorLUT[index1 * 4 + 0], frac);
    float g = lerp(d_colorLUT[index0 * 4 + 1], d_colorLUT[index1 * 4 + 1], frac);
    float b = lerp(d_colorLUT[index0 * 4 + 2], d_colorLUT[index1 * 4 + 2], frac);
    float a = lerp(d_colorLUT[index0 * 4 + 3], d_colorLUT[index1 * 4 + 3], frac);
    
    // 转换为uint8_t并写入输出
    output[idx * 4 + 0] = __float2uint_rn(r * 255.0f);
    output[idx * 4 + 1] = __float2uint_rn(g * 255.0f);
    output[idx * 4 + 2] = __float2uint_rn(b * 255.0f);
    output[idx * 4 + 3] = __float2uint_rn(a * 255.0f);
}

/**
 * @brief 批量颜色映射核函数（处理多个图像）
 */
__global__ void batchColorMappingKernel(
    const float** inputs,
    uint8_t** outputs,
    const int* widths,
    const int* heights,
    const float* minValues,
    const float* maxValues,
    int batchSize) {
    
    int batchIdx = blockIdx.z;
    if (batchIdx >= batchSize) return;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int width = widths[batchIdx];
    int height = heights[batchIdx];
    
    if (x >= width || y >= height) return;
    
    const float* input = inputs[batchIdx];
    uint8_t* output = outputs[batchIdx];
    float minValue = minValues[batchIdx];
    float maxValue = maxValues[batchIdx];
    
    int idx = y * width + x;
    float value = input[idx];
    
    // 处理NaN值
    if (isnan(value)) {
        output[idx * 4 + 0] = 0;
        output[idx * 4 + 1] = 0;
        output[idx * 4 + 2] = 0;
        output[idx * 4 + 3] = 0;
        return;
    }
    
    // 归一化
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = fmaxf(0.0f, fminf(1.0f, normalizedValue));
    
    // 映射到LUT索引
    float lutIndex = normalizedValue * 255.0f;
    int index0 = __float2int_rd(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    // 从LUT中获取颜色并插值
    float r = lerp(d_colorLUT[index0 * 4 + 0], d_colorLUT[index1 * 4 + 0], frac);
    float g = lerp(d_colorLUT[index0 * 4 + 1], d_colorLUT[index1 * 4 + 1], frac);
    float b = lerp(d_colorLUT[index0 * 4 + 2], d_colorLUT[index1 * 4 + 2], frac);
    float a = lerp(d_colorLUT[index0 * 4 + 3], d_colorLUT[index1 * 4 + 3], frac);
    
    // 转换为uint8_t并写入输出
    output[idx * 4 + 0] = __float2uint_rn(r * 255.0f);
    output[idx * 4 + 1] = __float2uint_rn(g * 255.0f);
    output[idx * 4 + 2] = __float2uint_rn(b * 255.0f);
    output[idx * 4 + 3] = __float2uint_rn(a * 255.0f);
}

// C++接口函数
extern "C" {

/**
 * @brief 上传颜色查找表到常量内存
 */
cudaError_t uploadColorLUT(const float* colorLUT, size_t size) {
    return cudaMemcpyToSymbol(d_colorLUT, colorLUT, size);
}

/**
 * @brief 执行基础颜色映射
 */
cudaError_t launchColorMapping(
    const float* d_input,
    uint8_t* d_output,
    int width,
    int height,
    float minValue,
    float maxValue,
    uint32_t nanColor,
    cudaStream_t stream) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    colorMappingKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, width, height, minValue, maxValue, nanColor);
    
    return cudaGetLastError();
}

/**
 * @brief 执行高级颜色映射
 */
cudaError_t launchAdvancedColorMapping(
    const float* d_input,
    uint8_t* d_output,
    int width,
    int height,
    float minValue,
    float maxValue,
    int transformType,
    float transformParam,
    float gamma,
    uint32_t nanColor,
    cudaStream_t stream) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    advancedColorMappingKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, width, height, minValue, maxValue,
        transformType, transformParam, gamma, nanColor);
    
    return cudaGetLastError();
}

/**
 * @brief 执行批量颜色映射
 */
cudaError_t launchBatchColorMapping(
    const float** d_inputs,
    uint8_t** d_outputs,
    const int* d_widths,
    const int* d_heights,
    const float* d_minValues,
    const float* d_maxValues,
    int batchSize,
    cudaStream_t stream) {
    
    // 找出最大尺寸以确定网格大小
    int maxWidth = 0, maxHeight = 0;
    std::vector<int> h_widths(batchSize), h_heights(batchSize);
    cudaMemcpy(h_widths.data(), d_widths, batchSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_heights.data(), d_heights, batchSize * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < batchSize; ++i) {
                    maxWidth = std::max(maxWidth, h_widths[i]);
            maxHeight = std::max(maxHeight, h_heights[i]);
    }
    
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((maxWidth + blockSize.x - 1) / blockSize.x,
                  (maxHeight + blockSize.y - 1) / blockSize.y,
                  batchSize);
    
    batchColorMappingKernel<<<gridSize, blockSize, 0, stream>>>(
        d_inputs, d_outputs, d_widths, d_heights, 
        d_minValues, d_maxValues, batchSize);
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace gpu
} // namespace output_generation
} // namespace oscean

// 外部接口函数
extern "C" {
    
cudaError_t launchColorMappingKernel(
    const float* d_input,
    uint8_t* d_output,
    int width,
    int height,
    float minValue,
    float maxValue,
    const float* d_colorLUT,
    int lutSize,
    cudaStream_t stream) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    oscean::output_generation::gpu::cuda::colorMappingKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, width, height, minValue, maxValue, 0xFF000000);
    
    return cudaGetLastError();
}

cudaError_t launchAdvancedColorMappingKernel(
    const float* d_input,
    uint8_t* d_output,
    int width,
    int height,
    float minValue,
    float maxValue,
    const float* d_colorLUT,
    int lutSize,
    int transformType,
    float transformParam,
    float gamma,
    cudaStream_t stream) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    oscean::output_generation::gpu::cuda::advancedColorMappingKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, width, height, minValue, maxValue,
        transformType, transformParam, gamma, 0xFF000000);
    
    return cudaGetLastError();
}

} // extern "C" 