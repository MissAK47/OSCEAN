/**
 * @file bilinear_interpolation.cu
 * @brief CUDA双线性插值核函数实现
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

// 注意：CUDA 12+中纹理引用已被弃用，使用常规内存访问代替

/**
 * @brief 双线性插值设备函数
 */
__device__ float bilinearInterpolate(
    const float* data,
    float x, float y,
    int width, int height,
    float fillValue) {
    
    // 边界检查
    if (x < 0 || x > width - 1 || y < 0 || y > height - 1) {
        return fillValue;
    }
    
    // 计算整数坐标
    int x0 = __float2int_rd(x);  // floor
    int y0 = __float2int_rd(y);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    // 计算分数部分
    float fx = x - x0;
    float fy = y - y0;
    
    // 获取四个角点的值
    float v00 = data[y0 * width + x0];
    float v10 = data[y0 * width + x1];
    float v01 = data[y1 * width + x0];
    float v11 = data[y1 * width + x1];
    
    // 处理NaN值
    if (isnan(v00)) v00 = fillValue;
    if (isnan(v10)) v10 = fillValue;
    if (isnan(v01)) v01 = fillValue;
    if (isnan(v11)) v11 = fillValue;
    
    // 双线性插值计算
    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;
    
    return v0 * (1.0f - fy) + v1 * fy;
}



/**
 * @brief 基础双线性插值核函数
 */
__global__ void bilinearInterpolationKernel(
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
    
    // 执行双线性插值
    float result = bilinearInterpolate(sourceData, srcX, srcY, 
                                     sourceWidth, sourceHeight, fillValue);
    
    // 写入结果
    outputData[outY * outputWidth + outX] = result;
}

/**
 * @brief 优化的双线性插值核函数（使用共享内存）
 */
__global__ void bilinearInterpolationOptimizedKernel(
    const float* sourceData,
    float* outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue) {
    
    // 共享内存用于缓存源数据块
    __shared__ float sharedData[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 计算对应的源坐标范围
    float blockMinX = minX + (maxX - minX) * (blockIdx.x * blockDim.x) / (outputWidth - 1);
    float blockMaxX = minX + (maxX - minX) * ((blockIdx.x + 1) * blockDim.x - 1) / (outputWidth - 1);
    float blockMinY = minY + (maxY - minY) * (blockIdx.y * blockDim.y) / (outputHeight - 1);
    float blockMaxY = minY + (maxY - minY) * ((blockIdx.y + 1) * blockDim.y - 1) / (outputHeight - 1);
    
    // 转换到源像素坐标
    int srcBlockMinX = max(0, (int)((blockMinX - minX) / (maxX - minX) * (sourceWidth - 1)));
    int srcBlockMaxX = min(sourceWidth - 1, (int)((blockMaxX - minX) / (maxX - minX) * (sourceWidth - 1)) + 1);
    int srcBlockMinY = max(0, (int)((blockMinY - minY) / (maxY - minY) * (sourceHeight - 1)));
    int srcBlockMaxY = min(sourceHeight - 1, (int)((blockMaxY - minY) / (maxY - minY) * (sourceHeight - 1)) + 1);
    
    // 协作加载数据到共享内存
    int sharedWidth = srcBlockMaxX - srcBlockMinX + 1;
    int sharedHeight = srcBlockMaxY - srcBlockMinY + 1;
    
    for (int sy = threadIdx.y; sy < sharedHeight; sy += blockDim.y) {
        for (int sx = threadIdx.x; sx < sharedWidth; sx += blockDim.x) {
            int srcX = srcBlockMinX + sx;
            int srcY = srcBlockMinY + sy;
            
            if (srcX < sourceWidth && srcY < sourceHeight) {
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
    
    // 从共享内存执行双线性插值
    float result = fillValue;
    if (localX >= 0 && localX < sharedWidth - 1 && 
        localY >= 0 && localY < sharedHeight - 1) {
        
        int x0 = (int)localX;
        int y0 = (int)localY;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        float fx = localX - x0;
        float fy = localY - y0;
        
        float v00 = sharedData[y0][x0];
        float v10 = sharedData[y0][x1];
        float v01 = sharedData[y1][x0];
        float v11 = sharedData[y1][x1];
        
        float v0 = v00 * (1.0f - fx) + v10 * fx;
        float v1 = v01 * (1.0f - fx) + v11 * fx;
        
        result = v0 * (1.0f - fy) + v1 * fy;
    }
    
    outputData[outY * outputWidth + outX] = result;
}

// C++接口函数
extern "C" {



/**
 * @brief 执行双线性插值
 */
cudaError_t launchBilinearInterpolation(
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
    
    bilinearInterpolationKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceData, d_outputData,
        sourceWidth, sourceHeight,
        outputWidth, outputHeight,
        minX, maxX, minY, maxY,
        fillValue);
    
    return cudaGetLastError();
}

/**
 * @brief 执行优化的双线性插值
 */
cudaError_t launchBilinearInterpolationOptimized(
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
    
    bilinearInterpolationOptimizedKernel<<<gridSize, blockSize, 0, stream>>>(
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