/**
 * @file image_resampling.cu
 * @brief CUDA图像重采样核函数实现
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace oscean {
namespace output_generation {
namespace gpu {
namespace cuda {

// 常量定义
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define LANCZOS_RADIUS 3

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief 计算Lanczos核函数值
 */
__device__ float lanczosKernel(float x, int a) {
    if (x == 0.0f) return 1.0f;
    if (fabsf(x) >= a) return 0.0f;
    
    float pix = M_PI * x;
    float pix_a = pix / a;
    return (sinf(pix) / pix) * (sinf(pix_a) / pix_a);
}

/**
 * @brief 双线性插值重采样核函数
 */
__global__ void bilinearResampleKernel(
    const float* input,
    float* output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    // 计算源图像坐标
    float srcX = (float)x * srcWidth / dstWidth;
    float srcY = (float)y * srcHeight / dstHeight;
    
    // 获取整数坐标和小数部分
    int x0 = (int)floorf(srcX);
    int y0 = (int)floorf(srcY);
    int x1 = min(x0 + 1, srcWidth - 1);
    int y1 = min(y0 + 1, srcHeight - 1);
    
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    // 边界检查
    x0 = max(0, x0);
    y0 = max(0, y0);
    
    // 双线性插值
    float v00 = input[y0 * srcWidth + x0];
    float v10 = input[y0 * srcWidth + x1];
    float v01 = input[y1 * srcWidth + x0];
    float v11 = input[y1 * srcWidth + x1];
    
    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;
    float result = v0 * (1.0f - fy) + v1 * fy;
    
    output[y * dstWidth + x] = result;
}

/**
 * @brief 双三次插值重采样核函数
 */
__global__ void bicubicResampleKernel(
    const float* input,
    float* output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    // 计算源图像坐标
    float srcX = (float)x * srcWidth / dstWidth;
    float srcY = (float)y * srcHeight / dstHeight;
    
    int x0 = (int)floorf(srcX);
    int y0 = (int)floorf(srcY);
    
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    // Mitchell-Netravali 双三次核函数
    auto cubicKernel = [](float x) -> float {
        float ax = fabsf(x);
        if (ax < 1.0f) {
            return (1.5f * ax - 2.5f) * ax * ax + 1.0f;
        } else if (ax < 2.0f) {
            return ((-0.5f * ax + 2.5f) * ax - 4.0f) * ax + 2.0f;
        }
        return 0.0f;
    };
    
    // 4x4 采样
    float sum = 0.0f;
    float weight = 0.0f;
    
    for (int j = -1; j <= 2; j++) {
        for (int i = -1; i <= 2; i++) {
            int sx = x0 + i;
            int sy = y0 + j;
            
            // 边界处理
            sx = max(0, min(sx, srcWidth - 1));
            sy = max(0, min(sy, srcHeight - 1));
            
            float w = cubicKernel(i - fx) * cubicKernel(j - fy);
            sum += input[sy * srcWidth + sx] * w;
            weight += w;
        }
    }
    
    output[y * dstWidth + x] = sum / weight;
}

/**
 * @brief Lanczos重采样核函数
 */
__global__ void lanczosResampleKernel(
    const float* input,
    float* output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    int radius) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    // 计算源图像坐标
    float srcX = (float)x * srcWidth / dstWidth;
    float srcY = (float)y * srcHeight / dstHeight;
    
    int x0 = (int)floorf(srcX);
    int y0 = (int)floorf(srcY);
    
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    // Lanczos采样
    float sum = 0.0f;
    float weight = 0.0f;
    
    for (int j = -radius + 1; j <= radius; j++) {
        for (int i = -radius + 1; i <= radius; i++) {
            int sx = x0 + i;
            int sy = y0 + j;
            
            // 边界处理
            if (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight) {
                continue;
            }
            
            float wx = lanczosKernel(i - fx, radius);
            float wy = lanczosKernel(j - fy, radius);
            float w = wx * wy;
            
            sum += input[sy * srcWidth + sx] * w;
            weight += w;
        }
    }
    
    output[y * dstWidth + x] = (weight > 0.0f) ? (sum / weight) : 0.0f;
}

/**
 * @brief 批量双线性重采样（优化版本）
 */
__global__ void bilinearResampleBatchKernel(
    const float* input,
    float* output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    int batchSize) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = dstWidth * dstHeight;
    
    if (idx >= totalPixels * batchSize) return;
    
    int batchIdx = idx / totalPixels;
    int pixelIdx = idx % totalPixels;
    
    int x = pixelIdx % dstWidth;
    int y = pixelIdx / dstWidth;
    
    // 计算源图像坐标
    float srcX = (float)x * srcWidth / dstWidth;
    float srcY = (float)y * srcHeight / dstHeight;
    
    int x0 = (int)floorf(srcX);
    int y0 = (int)floorf(srcY);
    int x1 = min(x0 + 1, srcWidth - 1);
    int y1 = min(y0 + 1, srcHeight - 1);
    
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    x0 = max(0, x0);
    y0 = max(0, y0);
    
    // 计算批次偏移
    int batchOffset = batchIdx * srcWidth * srcHeight;
    const float* batchInput = input + batchOffset;
    
    // 双线性插值
    float v00 = batchInput[y0 * srcWidth + x0];
    float v10 = batchInput[y0 * srcWidth + x1];
    float v01 = batchInput[y1 * srcWidth + x0];
    float v11 = batchInput[y1 * srcWidth + x1];
    
    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;
    float result = v0 * (1.0f - fy) + v1 * fy;
    
    output[idx] = result;
}

// C接口函数
extern "C" {

/**
 * @brief 执行双线性重采样
 */
cudaError_t resampleBilinearGPU(
    const float* d_input,
    float* d_output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(
        (dstWidth + blockSize.x - 1) / blockSize.x,
        (dstHeight + blockSize.y - 1) / blockSize.y
    );
    
    bilinearResampleKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, srcWidth, srcHeight, dstWidth, dstHeight
    );
    
    return cudaGetLastError();
}

/**
 * @brief 执行双三次重采样
 */
cudaError_t resampleBicubicGPU(
    const float* d_input,
    float* d_output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(
        (dstWidth + blockSize.x - 1) / blockSize.x,
        (dstHeight + blockSize.y - 1) / blockSize.y
    );
    
    bicubicResampleKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, srcWidth, srcHeight, dstWidth, dstHeight
    );
    
    return cudaGetLastError();
}

/**
 * @brief 执行Lanczos重采样
 */
cudaError_t resampleLanczosGPU(
    const float* d_input,
    float* d_output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    int radius,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(
        (dstWidth + blockSize.x - 1) / blockSize.x,
        (dstHeight + blockSize.y - 1) / blockSize.y
    );
    
    lanczosResampleKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, srcWidth, srcHeight, dstWidth, dstHeight, radius
    );
    
    return cudaGetLastError();
}

/**
 * @brief 批量双线性重采样
 */
cudaError_t resampleBilinearBatchGPU(
    const float* d_input,
    float* d_output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    int batchSize,
    cudaStream_t stream) {
    
    int totalPixels = dstWidth * dstHeight * batchSize;
    int blockSize = 256;
    int gridSize = (totalPixels + blockSize - 1) / blockSize;
    
    bilinearResampleBatchKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, srcWidth, srcHeight, dstWidth, dstHeight, batchSize
    );
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace gpu
} // namespace output_generation
} // namespace oscean 