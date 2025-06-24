/**
 * @file batch_interpolation.cu
 * @brief 批量插值CUDA核函数实现
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cstdio>

/**
 * @brief 批量双线性插值核函数
 * 
 * 每个线程处理一个输出像素，但可以处理批中的多个图像
 */
__global__ void bilinearInterpolationBatchKernel(
    const float** sourceArrays,     // 指向多个源图像的指针数组
    float** outputArrays,           // 指向多个输出图像的指针数组
    const int* sourceWidths,        // 每个源图像的宽度
    const int* sourceHeights,       // 每个源图像的高度
    const int* outputWidths,        // 每个输出图像的宽度
    const int* outputHeights,       // 每个输出图像的高度
    const float* minXArray,         // 每个图像的X最小值
    const float* maxXArray,         // 每个图像的X最大值
    const float* minYArray,         // 每个图像的Y最小值
    const float* maxYArray,         // 每个图像的Y最大值
    const float* fillValues,        // 每个图像的填充值
    int batchSize                   // 批处理大小
) {
    // 使用3D网格：x,y是图像坐标，z是批次索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z;
    
    if (batchIdx >= batchSize) return;
    
    int outputWidth = outputWidths[batchIdx];
    int outputHeight = outputHeights[batchIdx];
    
    if (x >= outputWidth || y >= outputHeight) return;
    
    const float* sourceData = sourceArrays[batchIdx];
    float* outputData = outputArrays[batchIdx];
    int sourceWidth = sourceWidths[batchIdx];
    int sourceHeight = sourceHeights[batchIdx];
    
    // 计算源坐标
    float srcX = minXArray[batchIdx] + (x / (float)(outputWidth - 1)) * (maxXArray[batchIdx] - minXArray[batchIdx]);
    float srcY = minYArray[batchIdx] + (y / (float)(outputHeight - 1)) * (maxYArray[batchIdx] - minYArray[batchIdx]);
    
    // 边界检查
    if (srcX < 0 || srcX > sourceWidth - 1 || srcY < 0 || srcY > sourceHeight - 1) {
        outputData[y * outputWidth + x] = fillValues[batchIdx];
        return;
    }
    
    // 双线性插值
    int x0 = (int)srcX;
    int y0 = (int)srcY;
    int x1 = min(x0 + 1, sourceWidth - 1);
    int y1 = min(y0 + 1, sourceHeight - 1);
    
    float wx = srcX - x0;
    float wy = srcY - y0;
    
    float v00 = sourceData[y0 * sourceWidth + x0];
    float v10 = sourceData[y0 * sourceWidth + x1];
    float v01 = sourceData[y1 * sourceWidth + x0];
    float v11 = sourceData[y1 * sourceWidth + x1];
    
    float result = (1 - wx) * (1 - wy) * v00 +
                   wx * (1 - wy) * v10 +
                   (1 - wx) * wy * v01 +
                   wx * wy * v11;
    
    outputData[y * outputWidth + x] = result;
}

/**
 * @brief 批量双三次插值核函数
 */
__device__ inline float cubicWeight(float t) {
    float a = -0.5f;
    float t2 = t * t;
    float t3 = t2 * t;
    
    if (fabsf(t) <= 1.0f) {
        return (a + 2.0f) * t3 - (a + 3.0f) * t2 + 1.0f;
    } else if (fabsf(t) <= 2.0f) {
        return a * t3 - 5.0f * a * t2 + 8.0f * a * fabsf(t) - 4.0f * a;
    }
    return 0.0f;
}

__global__ void bicubicInterpolationBatchKernel(
    const float** sourceArrays,
    float** outputArrays,
    const int* sourceWidths,
    const int* sourceHeights,
    const int* outputWidths,
    const int* outputHeights,
    const float* scaleX,
    const float* scaleY,
    const float* offsetX,
    const float* offsetY,
    const float* fillValues,
    int batchSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z;
    
    if (batchIdx >= batchSize) return;
    
    int outputWidth = outputWidths[batchIdx];
    int outputHeight = outputHeights[batchIdx];
    
    if (x >= outputWidth || y >= outputHeight) return;
    
    const float* sourceData = sourceArrays[batchIdx];
    float* outputData = outputArrays[batchIdx];
    int sourceWidth = sourceWidths[batchIdx];
    int sourceHeight = sourceHeights[batchIdx];
    
    // 使用预计算的缩放和偏移
    float srcX = x * scaleX[batchIdx] + offsetX[batchIdx];
    float srcY = y * scaleY[batchIdx] + offsetY[batchIdx];
    
    // 边界检查
    if (srcX < 1 || srcX > sourceWidth - 2 || srcY < 1 || srcY > sourceHeight - 2) {
        outputData[y * outputWidth + x] = fillValues[batchIdx];
        return;
    }
    
    // Catmull-Rom cubic interpolation
    int x0 = (int)srcX;
    int y0 = (int)srcY;
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    float result = 0.0f;
    
    // 4x4 kernel
    for (int j = -1; j <= 2; j++) {
        float wy = 0.0f;
        float t = fy - j;
        float t2 = t * t;
        float t3 = t2 * t;
        
        if (j == -1) wy = -0.5f * t3 + t2 - 0.5f * t;
        else if (j == 0) wy = 1.5f * t3 - 2.5f * t2 + 1.0f;
        else if (j == 1) wy = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
        else if (j == 2) wy = 0.5f * t3 - 0.5f * t2;
        
        float rowResult = 0.0f;
        for (int i = -1; i <= 2; i++) {
            float wx = 0.0f;
            t = fx - i;
            t2 = t * t;
            t3 = t2 * t;
            
            if (i == -1) wx = -0.5f * t3 + t2 - 0.5f * t;
            else if (i == 0) wx = 1.5f * t3 - 2.5f * t2 + 1.0f;
            else if (i == 1) wx = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
            else if (i == 2) wx = 0.5f * t3 - 0.5f * t2;
            
            int px = x0 + i;
            int py = y0 + j;
            
            px = max(0, min(px, sourceWidth - 1));
            py = max(0, min(py, sourceHeight - 1));
            
            rowResult += wx * sourceData[py * sourceWidth + px];
        }
        result += wy * rowResult;
    }
    
    outputData[y * outputWidth + x] = result;
}

// 最近邻批量插值核函数
__global__ void nearestNeighborBatchKernel(
    const float** sourceArrays,
    float** outputArrays,
    const int* sourceWidths,
    const int* sourceHeights,
    const int* outputWidths,
    const int* outputHeights,
    const float* scaleX,
    const float* scaleY,
    const float* offsetX,
    const float* offsetY,
    const float* fillValues,
    int batchSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z;
    
    if (batchIdx >= batchSize) return;
    
    int outputWidth = outputWidths[batchIdx];
    int outputHeight = outputHeights[batchIdx];
    
    if (x >= outputWidth || y >= outputHeight) return;
    
    const float* sourceData = sourceArrays[batchIdx];
    float* outputData = outputArrays[batchIdx];
    int sourceWidth = sourceWidths[batchIdx];
    int sourceHeight = sourceHeights[batchIdx];
    
    // 使用预计算的缩放和偏移
    float srcX = x * scaleX[batchIdx] + offsetX[batchIdx];
    float srcY = y * scaleY[batchIdx] + offsetY[batchIdx];
    
    // 四舍五入到最近的整数坐标
    int nearestX = (int)(srcX + 0.5f);
    int nearestY = (int)(srcY + 0.5f);
    
    // 边界检查
    if (nearestX < 0 || nearestX >= sourceWidth || nearestY < 0 || nearestY >= sourceHeight) {
        outputData[y * outputWidth + x] = fillValues[batchIdx];
    } else {
        outputData[y * outputWidth + x] = sourceData[nearestY * sourceWidth + nearestX];
    }
}

// C接口函数实现
extern "C" {
    
cudaError_t launchBatchBilinearInterpolation(
    const float** d_sourceArrays,
    float** d_outputArrays,
    const int* d_sourceWidths,
    const int* d_sourceHeights,
    const int* d_outputWidths,
    const int* d_outputHeights,
    const float* d_minXArray,
    const float* d_maxXArray,
    const float* d_minYArray,
    const float* d_maxYArray,
    const float* d_fillValues,
    int batchSize,
    cudaStream_t stream
) {
    // 找到最大的输出尺寸以确定网格大小
    int maxOutputWidth = 0;
    int maxOutputHeight = 0;
    
    // 这需要在主机端预计算
    // 暂时使用固定值
    maxOutputWidth = 2048;
    maxOutputHeight = 2048;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (maxOutputWidth + blockSize.x - 1) / blockSize.x,
        (maxOutputHeight + blockSize.y - 1) / blockSize.y,
        batchSize
    );
    
    bilinearInterpolationBatchKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceArrays, d_outputArrays,
        d_sourceWidths, d_sourceHeights,
        d_outputWidths, d_outputHeights,
        d_minXArray, d_maxXArray,
        d_minYArray, d_maxYArray,
        d_fillValues,
        batchSize
    );
    
    return cudaGetLastError();
}

// 优化版本的批量双线性插值启动函数
extern "C" cudaError_t launchBatchBilinearInterpolationOptimized(
    const float** d_sourceArrays,
    float** d_outputArrays,
    const int* d_sourceWidths,
    const int* d_sourceHeights,
    const int* d_outputWidths,
    const int* d_outputHeights,
    const float* d_scaleX,
    const float* d_scaleY,
    const float* d_offsetX,
    const float* d_offsetY,
    const float* d_fillValues,
    int batchSize,
    int maxOutputWidth,
    int maxOutputHeight,
    cudaStream_t stream
) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (maxOutputWidth + blockSize.x - 1) / blockSize.x,
        (maxOutputHeight + blockSize.y - 1) / blockSize.y,
        batchSize
    );
    
    size_t sharedMemSize = 18 * 18 * sizeof(float); // TILE_SIZE = 18
    
    bilinearInterpolationBatchKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_sourceArrays, d_outputArrays,
        d_sourceWidths, d_sourceHeights,
        d_outputWidths, d_outputHeights,
        d_scaleX, d_scaleY,
        d_offsetX, d_offsetY,
        d_fillValues,
        batchSize
    );
    
    return cudaGetLastError();
}

cudaError_t launchBicubicInterpolationBatch(
    const float** d_sourceArrays,
    float** d_outputArrays,
    const int* d_sourceWidths,
    const int* d_sourceHeights,
    const int* d_outputWidths,
    const int* d_outputHeights,
    const float* d_scaleX,
    const float* d_scaleY,
    const float* d_offsetX,
    const float* d_offsetY,
    const float* d_fillValues,
    int batchSize,
    cudaStream_t stream
) {
    if (batchSize <= 0) return cudaSuccess;
    
    // 找出最大的输出尺寸来设置网格
    int maxWidth = 0, maxHeight = 0;
    int* h_widths = new int[batchSize];
    int* h_heights = new int[batchSize];
    
    cudaMemcpy(h_widths, d_outputWidths, batchSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_heights, d_outputHeights, batchSize * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < batchSize; ++i) {
        maxWidth = max(maxWidth, h_widths[i]);
        maxHeight = max(maxHeight, h_heights[i]);
    }
    
    delete[] h_widths;
    delete[] h_heights;
    
    // 设置线程块和网格尺寸
    dim3 blockSize(8, 8);  // 双三次插值计算量大，使用较小的块
    dim3 gridSize(
        (maxWidth + blockSize.x - 1) / blockSize.x,
        (maxHeight + blockSize.y - 1) / blockSize.y,
        batchSize  // Z维度用于批处理
    );
    
    // 启动核函数
    bicubicInterpolationBatchKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceArrays, d_outputArrays,
        d_sourceWidths, d_sourceHeights,
        d_outputWidths, d_outputHeights,
        d_scaleX, d_scaleY,
        d_offsetX, d_offsetY,
        d_fillValues,
        batchSize
    );
    
    return cudaGetLastError();
}

} // extern "C" 