/**
 * @file trilinear_interpolation.cu
 * @brief CUDA三线性插值核函数实现
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
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

/**
 * @brief 三线性插值设备函数
 */
__device__ float trilinearInterpolate(
    const float* data,
    float x, float y, float z,
    int width, int height, int depth,
    float fillValue) {
    
    // 边界检查
    if (x < 0 || x > width - 1 || y < 0 || y > height - 1 || z < 0 || z > depth - 1) {
        return fillValue;
    }
    
    // 计算整数坐标
    int x0 = __float2int_rd(x);  // floor
    int y0 = __float2int_rd(y);
    int z0 = __float2int_rd(z);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    int z1 = min(z0 + 1, depth - 1);
    
    // 计算分数部分
    float fx = x - x0;
    float fy = y - y0;
    float fz = z - z0;
    
    // 获取八个角点的值
    float v000 = data[z0 * height * width + y0 * width + x0];
    float v100 = data[z0 * height * width + y0 * width + x1];
    float v010 = data[z0 * height * width + y1 * width + x0];
    float v110 = data[z0 * height * width + y1 * width + x1];
    float v001 = data[z1 * height * width + y0 * width + x0];
    float v101 = data[z1 * height * width + y0 * width + x1];
    float v011 = data[z1 * height * width + y1 * width + x0];
    float v111 = data[z1 * height * width + y1 * width + x1];
    
    // 处理NaN值
    if (isnan(v000)) v000 = fillValue;
    if (isnan(v100)) v100 = fillValue;
    if (isnan(v010)) v010 = fillValue;
    if (isnan(v110)) v110 = fillValue;
    if (isnan(v001)) v001 = fillValue;
    if (isnan(v101)) v101 = fillValue;
    if (isnan(v011)) v011 = fillValue;
    if (isnan(v111)) v111 = fillValue;
    
    // 三线性插值计算
    // 先在X方向插值
    float v00 = v000 * (1.0f - fx) + v100 * fx;
    float v10 = v010 * (1.0f - fx) + v110 * fx;
    float v01 = v001 * (1.0f - fx) + v101 * fx;
    float v11 = v011 * (1.0f - fx) + v111 * fx;
    
    // 再在Y方向插值
    float v0 = v00 * (1.0f - fy) + v10 * fy;
    float v1 = v01 * (1.0f - fy) + v11 * fy;
    
    // 最后在Z方向插值
    return v0 * (1.0f - fz) + v1 * fz;
}

/**
 * @brief 基础三线性插值核函数
 */
__global__ void trilinearInterpolationKernel(
    const float* sourceData,
    float* outputData,
    int sourceWidth, int sourceHeight, int sourceDepth,
    int outputWidth, int outputHeight, int outputDepth,
    float minX, float maxX,
    float minY, float maxY,
    float minZ, float maxZ,
    float fillValue) {
    
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (outX >= outputWidth || outY >= outputHeight || outZ >= outputDepth) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    float z = minZ + (maxZ - minZ) * outZ / (outputDepth - 1);
    
    // 转换到源数据的像素坐标
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    float srcZ = (z - minZ) / (maxZ - minZ) * (sourceDepth - 1);
    
    // 执行三线性插值
    float result = trilinearInterpolate(sourceData, srcX, srcY, srcZ,
                                       sourceWidth, sourceHeight, sourceDepth, fillValue);
    
    // 写入结果
    outputData[outZ * outputHeight * outputWidth + outY * outputWidth + outX] = result;
}

/**
 * @brief 优化的三线性插值核函数（使用共享内存）
 */
__global__ void trilinearInterpolationOptimizedKernel(
    const float* sourceData,
    float* outputData,
    int sourceWidth, int sourceHeight, int sourceDepth,
    int outputWidth, int outputHeight, int outputDepth,
    float minX, float maxX,
    float minY, float maxY,
    float minZ, float maxZ,
    float fillValue) {
    
    // 共享内存用于缓存源数据块
    __shared__ float sharedData[BLOCK_SIZE_Z + 2][BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z * blockDim.z + threadIdx.z;
    
    // 计算对应的源坐标范围
    float blockMinX = minX + (maxX - minX) * (blockIdx.x * blockDim.x) / (outputWidth - 1);
    float blockMaxX = minX + (maxX - minX) * ((blockIdx.x + 1) * blockDim.x - 1) / (outputWidth - 1);
    float blockMinY = minY + (maxY - minY) * (blockIdx.y * blockDim.y) / (outputHeight - 1);
    float blockMaxY = minY + (maxY - minY) * ((blockIdx.y + 1) * blockDim.y - 1) / (outputHeight - 1);
    float blockMinZ = minZ + (maxZ - minZ) * (blockIdx.z * blockDim.z) / (outputDepth - 1);
    float blockMaxZ = minZ + (maxZ - minZ) * ((blockIdx.z + 1) * blockDim.z - 1) / (outputDepth - 1);
    
    // 转换到源像素坐标
    int srcBlockMinX = max(0, (int)((blockMinX - minX) / (maxX - minX) * (sourceWidth - 1)));
    int srcBlockMaxX = min(sourceWidth - 1, (int)((blockMaxX - minX) / (maxX - minX) * (sourceWidth - 1)) + 1);
    int srcBlockMinY = max(0, (int)((blockMinY - minY) / (maxY - minY) * (sourceHeight - 1)));
    int srcBlockMaxY = min(sourceHeight - 1, (int)((blockMaxY - minY) / (maxY - minY) * (sourceHeight - 1)) + 1);
    int srcBlockMinZ = max(0, (int)((blockMinZ - minZ) / (maxZ - minZ) * (sourceDepth - 1)));
    int srcBlockMaxZ = min(sourceDepth - 1, (int)((blockMaxZ - minZ) / (maxZ - minZ) * (sourceDepth - 1)) + 1);
    
    // 协作加载数据到共享内存
    int sharedWidth = srcBlockMaxX - srcBlockMinX + 1;
    int sharedHeight = srcBlockMaxY - srcBlockMinY + 1;
    int sharedDepth = srcBlockMaxZ - srcBlockMinZ + 1;
    
    for (int sz = threadIdx.z; sz < sharedDepth; sz += blockDim.z) {
        for (int sy = threadIdx.y; sy < sharedHeight; sy += blockDim.y) {
            for (int sx = threadIdx.x; sx < sharedWidth; sx += blockDim.x) {
                int srcX = srcBlockMinX + sx;
                int srcY = srcBlockMinY + sy;
                int srcZ = srcBlockMinZ + sz;
                
                if (srcX < sourceWidth && srcY < sourceHeight && srcZ < sourceDepth) {
                    sharedData[sz][sy][sx] = sourceData[srcZ * sourceHeight * sourceWidth + 
                                                       srcY * sourceWidth + srcX];
                } else {
                    sharedData[sz][sy][sx] = fillValue;
                }
            }
        }
    }
    
    __syncthreads();
    
    if (outX >= outputWidth || outY >= outputHeight || outZ >= outputDepth) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    float z = minZ + (maxZ - minZ) * outZ / (outputDepth - 1);
    
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    float srcZ = (z - minZ) / (maxZ - minZ) * (sourceDepth - 1);
    
    // 相对于共享内存块的坐标
    float localX = srcX - srcBlockMinX;
    float localY = srcY - srcBlockMinY;
    float localZ = srcZ - srcBlockMinZ;
    
    // 从共享内存执行三线性插值
    float result = fillValue;
    if (localX >= 0 && localX < sharedWidth - 1 && 
        localY >= 0 && localY < sharedHeight - 1 &&
        localZ >= 0 && localZ < sharedDepth - 1) {
        
        int x0 = (int)localX;
        int y0 = (int)localY;
        int z0 = (int)localZ;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;
        
        float fx = localX - x0;
        float fy = localY - y0;
        float fz = localZ - z0;
        
        // 获取八个角点的值
        float v000 = sharedData[z0][y0][x0];
        float v100 = sharedData[z0][y0][x1];
        float v010 = sharedData[z0][y1][x0];
        float v110 = sharedData[z0][y1][x1];
        float v001 = sharedData[z1][y0][x0];
        float v101 = sharedData[z1][y0][x1];
        float v011 = sharedData[z1][y1][x0];
        float v111 = sharedData[z1][y1][x1];
        
        // 三线性插值
        float v00 = v000 * (1.0f - fx) + v100 * fx;
        float v10 = v010 * (1.0f - fx) + v110 * fx;
        float v01 = v001 * (1.0f - fx) + v101 * fx;
        float v11 = v011 * (1.0f - fx) + v111 * fx;
        
        float v0 = v00 * (1.0f - fy) + v10 * fy;
        float v1 = v01 * (1.0f - fy) + v11 * fy;
        
        result = v0 * (1.0f - fz) + v1 * fz;
    }
    
    outputData[outZ * outputHeight * outputWidth + outY * outputWidth + outX] = result;
}

// C++接口函数
extern "C" {

/**
 * @brief 执行三线性插值
 */
cudaError_t launchTrilinearInterpolation(
    const float* d_sourceData,
    float* d_outputData,
    int sourceWidth, int sourceHeight, int sourceDepth,
    int outputWidth, int outputHeight, int outputDepth,
    float minX, float maxX,
    float minY, float maxY,
    float minZ, float maxZ,
    float fillValue,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y,
                  (outputDepth + blockSize.z - 1) / blockSize.z);
    
    trilinearInterpolationKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceData, d_outputData,
        sourceWidth, sourceHeight, sourceDepth,
        outputWidth, outputHeight, outputDepth,
        minX, maxX, minY, maxY, minZ, maxZ,
        fillValue);
    
    return cudaGetLastError();
}

/**
 * @brief 执行优化的三线性插值
 */
cudaError_t launchTrilinearInterpolationOptimized(
    const float* d_sourceData,
    float* d_outputData,
    int sourceWidth, int sourceHeight, int sourceDepth,
    int outputWidth, int outputHeight, int outputDepth,
    float minX, float maxX,
    float minY, float maxY,
    float minZ, float maxZ,
    float fillValue,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y,
                  (outputDepth + blockSize.z - 1) / blockSize.z);
    
    trilinearInterpolationOptimizedKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceData, d_outputData,
        sourceWidth, sourceHeight, sourceDepth,
        outputWidth, outputHeight, outputDepth,
        minX, maxX, minY, maxY, minZ, maxZ,
        fillValue);
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace gpu
} // namespace interpolation
} // namespace oscean 