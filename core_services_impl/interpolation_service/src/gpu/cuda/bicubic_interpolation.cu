/**
 * @file bicubic_interpolation.cu
 * @brief CUDA双三次插值核函数实现
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace oscean {
namespace interpolation {
namespace gpu {
namespace cuda {

// 常量定义
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

/**
 * @brief 三次插值权重计算
 */
__device__ float cubicWeight(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return -0.5f * t3 + t2 - 0.5f * t;
}

/**
 * @brief 计算三次插值的四个权重
 */
__device__ void getCubicWeights(float t, float weights[4]) {
    float t2 = t * t;
    float t3 = t2 * t;
    
    weights[0] = -0.5f * t3 + t2 - 0.5f * t;
    weights[1] = 1.5f * t3 - 2.5f * t2 + 1.0f;
    weights[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    weights[3] = 0.5f * t3 - 0.5f * t2;
}

/**
 * @brief 双三次插值设备函数
 */
__device__ float bicubicInterpolate(
    const float* data,
    float x, float y,
    int width, int height,
    float fillValue) {
    
    // 边界检查
    if (x < 1 || x > width - 2 || y < 1 || y > height - 2) {
        return fillValue;
    }
    
    // 计算整数坐标
    int x0 = __float2int_rd(x);
    int y0 = __float2int_rd(y);
    
    // 计算分数部分
    float fx = x - x0;
    float fy = y - y0;
    
    // 计算权重
    float wx[4], wy[4];
    getCubicWeights(fx, wx);
    getCubicWeights(fy, wy);
    
    // 双三次插值
    float result = 0.0f;
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            int xi = x0 + i;
            int yi = y0 + j;
            
            // 边界处理
            xi = max(0, min(xi, width - 1));
            yi = max(0, min(yi, height - 1));
            
            float value = data[yi * width + xi];
            if (isnan(value)) value = fillValue;
            
            result += value * wx[i + 1] * wy[j + 1];
        }
    }
    
    return result;
}

/**
 * @brief 基础双三次插值核函数
 */
__global__ void bicubicInterpolationKernel(
    const float* sourceData,
    float* outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue) {
    
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outX >= outputWidth || outY >= outputHeight) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    
    // 转换到源数据的像素坐标
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    
    // 执行双三次插值
    float result = bicubicInterpolate(sourceData, srcX, srcY, 
                                     sourceWidth, sourceHeight, fillValue);
    
    // 写入结果
    outputData[outY * outputWidth + outX] = result;
}

/**
 * @brief 优化的双三次插值核函数（使用共享内存）
 */
__global__ void bicubicInterpolationOptimizedKernel(
    const float* sourceData,
    float* outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue) {
    
    // 共享内存用于缓存源数据块（需要更大的边界）
    __shared__ float sharedData[BLOCK_SIZE_Y + 6][BLOCK_SIZE_X + 6];
    
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 计算对应的源坐标范围
    float blockMinX = minX + (maxX - minX) * (blockIdx.x * blockDim.x - 1) / (outputWidth - 1);
    float blockMaxX = minX + (maxX - minX) * ((blockIdx.x + 1) * blockDim.x + 1) / (outputWidth - 1);
    float blockMinY = minY + (maxY - minY) * (blockIdx.y * blockDim.y - 1) / (outputHeight - 1);
    float blockMaxY = minY + (maxY - minY) * ((blockIdx.y + 1) * blockDim.y + 1) / (outputHeight - 1);
    
    // 转换到源像素坐标
    int srcBlockMinX = max(0, (int)((blockMinX - minX) / (maxX - minX) * (sourceWidth - 1)) - 1);
    int srcBlockMaxX = min(sourceWidth - 1, (int)((blockMaxX - minX) / (maxX - minX) * (sourceWidth - 1)) + 2);
    int srcBlockMinY = max(0, (int)((blockMinY - minY) / (maxY - minY) * (sourceHeight - 1)) - 1);
    int srcBlockMaxY = min(sourceHeight - 1, (int)((blockMaxY - minY) / (maxY - minY) * (sourceHeight - 1)) + 2);
    
    // 协作加载数据到共享内存
    int sharedWidth = srcBlockMaxX - srcBlockMinX + 1;
    int sharedHeight = srcBlockMaxY - srcBlockMinY + 1;
    
    for (int sy = threadIdx.y; sy < sharedHeight; sy += blockDim.y) {
        for (int sx = threadIdx.x; sx < sharedWidth; sx += blockDim.x) {
            int srcX = srcBlockMinX + sx;
            int srcY = srcBlockMinY + sy;
            
            if (srcX >= 0 && srcX < sourceWidth && srcY >= 0 && srcY < sourceHeight) {
                sharedData[sy][sx] = sourceData[srcY * sourceWidth + srcX];
            } else {
                sharedData[sy][sx] = fillValue;
            }
        }
    }
    
    __syncthreads();
    
    if (outX >= outputWidth || outY >= outputHeight) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    
    // 相对于共享内存块的坐标
    float localX = srcX - srcBlockMinX;
    float localY = srcY - srcBlockMinY;
    
    // 从共享内存执行双三次插值
    float result = fillValue;
    if (localX >= 1 && localX < sharedWidth - 2 && 
        localY >= 1 && localY < sharedHeight - 2) {
        
        int x0 = (int)localX;
        int y0 = (int)localY;
        
        float fx = localX - x0;
        float fy = localY - y0;
        
        float wx[4], wy[4];
        getCubicWeights(fx, wx);
        getCubicWeights(fy, wy);
        
        result = 0.0f;
        for (int j = -1; j <= 2; ++j) {
            for (int i = -1; i <= 2; ++i) {
                float value = sharedData[y0 + j][x0 + i];
                result += value * wx[i + 1] * wy[j + 1];
            }
        }
    }
    
    outputData[outY * outputWidth + outX] = result;
}

// C++接口函数
extern "C" {

/**
 * @brief 执行双三次插值
 */
cudaError_t launchBicubicInterpolation(
    const float* d_sourceData,
    float* d_outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);
    
    bicubicInterpolationKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceData, d_outputData,
        sourceWidth, sourceHeight,
        outputWidth, outputHeight,
        minX, maxX, minY, maxY,
        fillValue);
    
    return cudaGetLastError();
}

/**
 * @brief 执行优化的双三次插值
 */
cudaError_t launchBicubicInterpolationOptimized(
    const float* d_sourceData,
    float* d_outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);
    
    bicubicInterpolationOptimizedKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceData, d_outputData,
        sourceWidth, sourceHeight,
        outputWidth, outputHeight,
        minX, maxX, minY, maxY,
        fillValue);
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace gpu
} // namespace interpolation
} // namespace oscean 