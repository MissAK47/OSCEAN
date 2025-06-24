/**
 * @file nearest_neighbor_interpolation.cu
 * @brief CUDA最近邻插值核函数实现
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace oscean {
namespace interpolation {
namespace gpu {
namespace cuda {

// 常量定义
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16

/**
 * @brief 最近邻插值设备函数
 */
__device__ float nearestNeighborInterpolate(
    const float* data,
    float x, float y,
    int width, int height,
    float fillValue) {
    
    // 边界检查
    if (x < -0.5f || x > width - 0.5f || y < -0.5f || y > height - 0.5f) {
        return fillValue;
    }
    
    // 最近邻 - 四舍五入到最近的整数坐标
    int nearestX = __float2int_rn(x);  // round to nearest
    int nearestY = __float2int_rn(y);
    
    // 确保在有效范围内
    nearestX = max(0, min(nearestX, width - 1));
    nearestY = max(0, min(nearestY, height - 1));
    
    // 获取最近点的值
    float value = data[nearestY * width + nearestX];
    
    // 处理NaN值
    return isnan(value) ? fillValue : value;
}

/**
 * @brief 基础最近邻插值核函数
 */
__global__ void nearestNeighborInterpolationKernel(
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
    
    // 执行最近邻插值
    float result = nearestNeighborInterpolate(sourceData, srcX, srcY, 
                                            sourceWidth, sourceHeight, fillValue);
    
    // 写入结果
    outputData[outY * outputWidth + outX] = result;
}

/**
 * @brief 优化的最近邻插值核函数（使用共享内存）
 */
__global__ void nearestNeighborInterpolationOptimizedKernel(
    const float* sourceData,
    float* outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue) {
    
    // 使用共享内存预取数据块
    extern __shared__ float sharedData[];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 计算该块对应的源数据范围
    float blockStartX = minX + (maxX - minX) * (blockIdx.x * blockDim.x) / (outputWidth - 1);
    float blockEndX = minX + (maxX - minX) * ((blockIdx.x + 1) * blockDim.x - 1) / (outputWidth - 1);
    float blockStartY = minY + (maxY - minY) * (blockIdx.y * blockDim.y) / (outputHeight - 1);
    float blockEndY = minY + (maxY - minY) * ((blockIdx.y + 1) * blockDim.y - 1) / (outputHeight - 1);
    
    // 转换到源像素坐标并扩展边界（考虑舍入）
    int srcStartX = max(0, (int)((blockStartX - minX) / (maxX - minX) * (sourceWidth - 1) - 0.5f));
    int srcEndX = min(sourceWidth - 1, (int)((blockEndX - minX) / (maxX - minX) * (sourceWidth - 1) + 0.5f));
    int srcStartY = max(0, (int)((blockStartY - minY) / (maxY - minY) * (sourceHeight - 1) - 0.5f));
    int srcEndY = min(sourceHeight - 1, (int)((blockEndY - minY) / (maxY - minY) * (sourceHeight - 1) + 0.5f));
    
    int sharedWidth = srcEndX - srcStartX + 1;
    int sharedHeight = srcEndY - srcStartY + 1;
    int sharedSize = sharedWidth * sharedHeight;
    
    // 协作加载数据到共享内存
    for (int i = tid; i < sharedSize; i += blockDim.x * blockDim.y) {
        int localX = i % sharedWidth;
        int localY = i / sharedWidth;
        int srcX = srcStartX + localX;
        int srcY = srcStartY + localY;
        
        if (srcX < sourceWidth && srcY < sourceHeight) {
            sharedData[i] = sourceData[srcY * sourceWidth + srcX];
        }
    }
    
    __syncthreads();
    
    if (outX >= outputWidth || outY >= outputHeight) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    
    // 最近邻坐标
    int nearestX = __float2int_rn(srcX);
    int nearestY = __float2int_rn(srcY);
    
    float result = fillValue;
    
    // 检查是否在共享内存范围内
    if (nearestX >= srcStartX && nearestX <= srcEndX &&
        nearestY >= srcStartY && nearestY <= srcEndY) {
        int localIdx = (nearestY - srcStartY) * sharedWidth + (nearestX - srcStartX);
        result = sharedData[localIdx];
        if (isnan(result)) result = fillValue;
    } else if (nearestX >= 0 && nearestX < sourceWidth &&
               nearestY >= 0 && nearestY < sourceHeight) {
        // 从全局内存读取
        result = sourceData[nearestY * sourceWidth + nearestX];
        if (isnan(result)) result = fillValue;
    }
    
    outputData[outY * outputWidth + outX] = result;
}

/**
 * @brief 批量最近邻插值核函数
 */
__global__ void batchNearestNeighborInterpolationKernel(
    const float** sourceDatas,
    float** outputDatas,
    const int* sourceWidths,
    const int* sourceHeights,
    const int* outputWidths,
    const int* outputHeights,
    const float* bounds,  // minX, maxX, minY, maxY for each batch
    float fillValue,
    int batchSize) {
    
    int batchIdx = blockIdx.z;
    if (batchIdx >= batchSize) return;
    
    const float* sourceData = sourceDatas[batchIdx];
    float* outputData = outputDatas[batchIdx];
    int sourceWidth = sourceWidths[batchIdx];
    int sourceHeight = sourceHeights[batchIdx];
    int outputWidth = outputWidths[batchIdx];
    int outputHeight = outputHeights[batchIdx];
    
    float minX = bounds[batchIdx * 4 + 0];
    float maxX = bounds[batchIdx * 4 + 1];
    float minY = bounds[batchIdx * 4 + 2];
    float maxY = bounds[batchIdx * 4 + 3];
    
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outX >= outputWidth || outY >= outputHeight) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    
    // 执行最近邻插值
    float result = nearestNeighborInterpolate(sourceData, srcX, srcY, 
                                            sourceWidth, sourceHeight, fillValue);
    
    outputData[outY * outputWidth + outX] = result;
}

// C++接口函数
extern "C" {

/**
 * @brief 执行最近邻插值
 */
cudaError_t launchNearestNeighborInterpolation(
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
    
    nearestNeighborInterpolationKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceData, d_outputData,
        sourceWidth, sourceHeight,
        outputWidth, outputHeight,
        minX, maxX, minY, maxY,
        fillValue);
    
    return cudaGetLastError();
}

/**
 * @brief 执行优化的最近邻插值
 */
cudaError_t launchNearestNeighborInterpolationOptimized(
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
    
    // 计算共享内存大小（保守估计）
    int maxSharedSize = 48 * 1024; // 48KB共享内存
    int elementsPerBlock = maxSharedSize / sizeof(float);
    size_t sharedMemSize = elementsPerBlock * sizeof(float);
    
    nearestNeighborInterpolationOptimizedKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_sourceData, d_outputData,
        sourceWidth, sourceHeight,
        outputWidth, outputHeight,
        minX, maxX, minY, maxY,
        fillValue);
    
    return cudaGetLastError();
}

/**
 * @brief 执行批量最近邻插值
 */
cudaError_t launchBatchNearestNeighborInterpolation(
    const float** d_sourceDatas,
    float** d_outputDatas,
    const int* d_sourceWidths,
    const int* d_sourceHeights,
    const int* d_outputWidths,
    const int* d_outputHeights,
    const float* d_bounds,
    float fillValue,
    int batchSize,
    cudaStream_t stream) {
    
    // 找到最大的输出尺寸（需要主机端临时数组）
    int maxOutputWidth = 0, maxOutputHeight = 0;
    
    // 这里简化处理，假设所有批次的输出尺寸相近
    // 实际使用时可以传入预计算的最大值
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(256, 256, batchSize); // 足够大的网格
    
    batchNearestNeighborInterpolationKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceDatas, d_outputDatas,
        d_sourceWidths, d_sourceHeights,
        d_outputWidths, d_outputHeights,
        d_bounds, fillValue, batchSize);
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace gpu
} // namespace interpolation
} // namespace oscean 