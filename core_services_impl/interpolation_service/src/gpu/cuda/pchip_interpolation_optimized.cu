/**
 * @file pchip_interpolation_optimized.cu
 * @brief 优化的CUDA PCHIP快速2D插值核函数实现（完整版本）
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace oscean {
namespace interpolation {
namespace gpu {
namespace cuda {

// 常量定义
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define WARP_SIZE 32

// 共享内存大小（用于缓存数据块）
#define SHARED_MEMORY_SIZE 18  // 16x16 + 边界 = 18x18

/**
 * @brief 优化的PCHIP斜率计算（无分支版本）
 */
__device__ __forceinline__ float pchipSlopeOptimized(float h1, float h2, float m1, float m2) {
    // 使用无分支计算
    float sign_product = m1 * m2;
    
    // 如果符号不同（乘积为负或零），返回0
    if (sign_product <= 0.0f) {
        return 0.0f;
    }
    
    // 调和平均数
    float wh1 = 2.0f * h1 + h2;
    float wh2 = h1 + 2.0f * h2;
    return (wh1 + wh2) / (wh1 / m1 + wh2 / m2);
}

/**
 * @brief 优化的Hermite多项式计算（使用FMA指令）
 */
__device__ __forceinline__ float evaluateHermiteOptimized(
    float t, float y0, float y1, float m0, float m1) {
    
    float t2 = t * t;
    float mt = 1.0f - t;
    float mt2 = mt * mt;
    
    // 使用FMA（Fused Multiply-Add）指令优化
    float h00 = fmaf(2.0f, t, 1.0f) * mt2;        // (1 + 2t) * (1-t)^2
    float h10 = t * mt2;                           // t * (1-t)^2
    float h01 = t2 * fmaf(-2.0f, t, 3.0f);        // t^2 * (3 - 2t)
    float h11 = t2 * (t - 1.0f);                  // t^2 * (t - 1)
    
    // 使用FMA组合结果
    return fmaf(h00, y0, fmaf(h10, m0, fmaf(h01, y1, h11 * m1)));
}

// 注：纹理内存在CUDA 12中已被弃用，改用直接内存访问和共享内存优化

/**
 * @brief 完整的PCHIP 2D插值核函数（使用共享内存优化）
 */
__global__ void pchip2DInterpolationOptimizedKernel(
    const float* __restrict__ sourceData,
    const float* __restrict__ derivX,
    const float* __restrict__ derivY,
    const float* __restrict__ derivXY,
    float* __restrict__ outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float scaleX, float scaleY,    // 预计算的缩放因子
    float offsetX, float offsetY,   // 预计算的偏移
    float fillValue) {
    
    // 共享内存用于缓存局部数据块
    __shared__ float sharedData[SHARED_MEMORY_SIZE][SHARED_MEMORY_SIZE];
    __shared__ float sharedDerivX[SHARED_MEMORY_SIZE][SHARED_MEMORY_SIZE];
    __shared__ float sharedDerivY[SHARED_MEMORY_SIZE][SHARED_MEMORY_SIZE];
    __shared__ float sharedDerivXY[SHARED_MEMORY_SIZE][SHARED_MEMORY_SIZE];
    
    // 全局索引
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 计算对应的源坐标（使用预计算的缩放因子）
    float srcX = fmaf((float)outX, scaleX, offsetX);
    float srcY = fmaf((float)outY, scaleY, offsetY);
    
    // 计算源数据的基础索引
    int baseX = __float2int_rd(srcX) - 1;  // 需要额外的边界点
    int baseY = __float2int_rd(srcY) - 1;
    
    // 协作加载数据到共享内存
    int localX = threadIdx.x;
    int localY = threadIdx.y;
    
    // 每个线程负责加载一部分数据
    for (int dy = localY; dy < SHARED_MEMORY_SIZE; dy += blockDim.y) {
        for (int dx = localX; dx < SHARED_MEMORY_SIZE; dx += blockDim.x) {
            int globalX = baseX + dx;
            int globalY = baseY + dy;
            
            // 边界检查
            if (globalX >= 0 && globalX < sourceWidth &&
                globalY >= 0 && globalY < sourceHeight) {
                int srcIdx = globalY * sourceWidth + globalX;
                sharedData[dy][dx] = sourceData[srcIdx];
                sharedDerivX[dy][dx] = derivX[srcIdx];
                sharedDerivY[dy][dx] = derivY[srcIdx];
                sharedDerivXY[dy][dx] = derivXY[srcIdx];
            } else {
                sharedData[dy][dx] = fillValue;
                sharedDerivX[dy][dx] = 0.0f;
                sharedDerivY[dy][dx] = 0.0f;
                sharedDerivXY[dy][dx] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // 主计算
    if (outX < outputWidth && outY < outputHeight) {
        // 边界检查
        if (srcX < 0 || srcX > sourceWidth - 1 || 
            srcY < 0 || srcY > sourceHeight - 1) {
            outputData[outY * outputWidth + outX] = fillValue;
            return;
        }
        
        // 转换到共享内存坐标
        float localSrcX = srcX - baseX;
        float localSrcY = srcY - baseY;
        
        // 计算整数坐标（在共享内存中）
        int x0 = __float2int_rd(localSrcX);
        int y0 = __float2int_rd(localSrcY);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        // 计算分数部分
        float tx = localSrcX - x0;
        float ty = localSrcY - y0;
        
        // 从共享内存读取4个角点的数据
        float v00 = sharedData[y0][x0];
        float v10 = sharedData[y0][x1];
        float v01 = sharedData[y1][x0];
        float v11 = sharedData[y1][x1];
        
        // 处理NaN值
        if (isnan(v00)) v00 = fillValue;
        if (isnan(v10)) v10 = fillValue;
        if (isnan(v01)) v01 = fillValue;
        if (isnan(v11)) v11 = fillValue;
        
        // 读取导数（已经缩放过）
        float fx00 = sharedDerivX[y0][x0];
        float fx10 = sharedDerivX[y0][x1];
        float fx01 = sharedDerivX[y1][x0];
        float fx11 = sharedDerivX[y1][x1];
        
        float fy00 = sharedDerivY[y0][x0];
        float fy10 = sharedDerivY[y0][x1];
        float fy01 = sharedDerivY[y1][x0];
        float fy11 = sharedDerivY[y1][x1];
        
        float fxy00 = sharedDerivXY[y0][x0];
        float fxy10 = sharedDerivXY[y0][x1];
        float fxy01 = sharedDerivXY[y1][x0];
        float fxy11 = sharedDerivXY[y1][x1];
        
        // X方向的PCHIP插值（对y=y0和y=y1）
        float v0 = evaluateHermiteOptimized(tx, v00, v10, fx00, fx10);
        float v1 = evaluateHermiteOptimized(tx, v01, v11, fx01, fx11);
        
        // Y方向导数的插值
        float m0y = evaluateHermiteOptimized(tx, fy00, fy10, fxy00, fxy10);
        float m1y = evaluateHermiteOptimized(tx, fy01, fy11, fxy01, fxy11);
        
        // 最终Y方向的PCHIP插值
        float result = evaluateHermiteOptimized(ty, v0, v1, m0y, m1y);
        
        // 写入结果
        outputData[outY * outputWidth + outX] = result;
    }
}

/**
 * @brief 一体化的PCHIP插值核函数（导数即时计算，完整版本）
 */
__global__ void pchip2DInterpolationIntegratedKernel(
    const float* __restrict__ sourceData,
    float* __restrict__ outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float scaleX, float scaleY,
    float offsetX, float offsetY,
    float fillValue) {
    
    // 共享内存用于缓存4x4数据块
    __shared__ float sharedData[BLOCK_SIZE_Y + 3][BLOCK_SIZE_X + 3];
    
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 计算源坐标
    float srcX = fmaf((float)outX, scaleX, offsetX);
    float srcY = fmaf((float)outY, scaleY, offsetY);
    
    // 计算需要的数据块的起始位置
    int blockStartX = __float2int_rd(fmaf((float)(blockIdx.x * blockDim.x), scaleX, offsetX)) - 1;
    int blockStartY = __float2int_rd(fmaf((float)(blockIdx.y * blockDim.y), scaleY, offsetY)) - 1;
    
    // 协作加载4x4邻域数据到共享内存
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;
    int sharedWidth = BLOCK_SIZE_X + 3;
    int sharedHeight = BLOCK_SIZE_Y + 3;
    int totalElements = sharedWidth * sharedHeight;
    
    for (int i = tid; i < totalElements; i += totalThreads) {
        int localX = i % sharedWidth;
        int localY = i / sharedWidth;
        int globalX = blockStartX + localX;
        int globalY = blockStartY + localY;
        
        if (globalX >= 0 && globalX < sourceWidth &&
            globalY >= 0 && globalY < sourceHeight) {
            sharedData[localY][localX] = sourceData[globalY * sourceWidth + globalX];
        } else {
            sharedData[localY][localX] = fillValue;
        }
    }
    
    __syncthreads();
    
    if (outX >= outputWidth || outY >= outputHeight) return;
    
    // 边界检查
    if (srcX < 1 || srcX >= sourceWidth - 2 || 
        srcY < 1 || srcY >= sourceHeight - 2) {
        outputData[outY * outputWidth + outX] = fillValue;
        return;
    }
    
    // 在共享内存中的局部坐标
    float localX = srcX - blockStartX;
    float localY = srcY - blockStartY;
    
    int x0 = __float2int_rd(localX);
    int y0 = __float2int_rd(localY);
    
    // 计算分数部分
    float tx = localX - x0;
    float ty = localY - y0;
    
    // 从共享内存读取4x4邻域数据
    float data[4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            data[j][i] = sharedData[y0 - 1 + j][x0 - 1 + i];
        }
    }
    
    // 计算中心4个点的导数
    float fx[2][2], fy[2][2], fxy[2][2];
    
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            // X方向导数
            float dxLeft = data[j+1][i+1] - data[j+1][i];
            float dxRight = data[j+1][i+2] - data[j+1][i+1];
            fx[j][i] = pchipSlopeOptimized(1.0f, 1.0f, dxLeft, dxRight);
            
            // Y方向导数
            float dyUp = data[j+1][i+1] - data[j][i+1];
            float dyDown = data[j+2][i+1] - data[j+1][i+1];
            fy[j][i] = pchipSlopeOptimized(1.0f, 1.0f, dyUp, dyDown);
        }
    }
    
    // 计算交叉导数
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            // 使用相邻点的X导数差计算XY导数
            float fxyUp = (i < 1) ? 
                (fx[j][i+1] - fx[j][i]) : 
                (fx[j][i] - fx[j][i-1]);
            float fxyDown = (i < 1) ? 
                (fx[j+1 < 2 ? j+1 : j][i+1] - fx[j+1 < 2 ? j+1 : j][i]) :
                (fx[j+1 < 2 ? j+1 : j][i] - fx[j+1 < 2 ? j+1 : j][i-1]);
            fxy[j][i] = 0.5f * (fxyUp + fxyDown);
        }
    }
    
    // 获取中心4个点的值
    float v00 = data[1][1], v10 = data[1][2];
    float v01 = data[2][1], v11 = data[2][2];
    
    // X方向的PCHIP插值
    float v0 = evaluateHermiteOptimized(tx, v00, v10, fx[0][0], fx[0][1]);
    float v1 = evaluateHermiteOptimized(tx, v01, v11, fx[1][0], fx[1][1]);
    
    // Y方向导数的插值
    float m0y = evaluateHermiteOptimized(tx, fy[0][0], fy[0][1], fxy[0][0], fxy[0][1]);
    float m1y = evaluateHermiteOptimized(tx, fy[1][0], fy[1][1], fxy[1][0], fxy[1][1]);
    
    // 最终Y方向的PCHIP插值
    float result = evaluateHermiteOptimized(ty, v0, v1, m0y, m1y);
    
    // 写入结果
    outputData[outY * outputWidth + outX] = result;
}

/**
 * @brief 高度优化的导数计算核函数
 */
__global__ void computePCHIPDerivativesOptimizedKernel(
    const float* __restrict__ data,
    float* __restrict__ derivX,
    float* __restrict__ derivY,
    float* __restrict__ derivXY,
    int width, int height,
    float dx, float dy) {
    
    // 使用共享内存缓存3x3邻域
    __shared__ float sharedData[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 协作加载数据到共享内存（包括边界）
    int localX = threadIdx.x + 1;
    int localY = threadIdx.y + 1;
    
    // 加载中心点
    if (globalX < width && globalY < height) {
        sharedData[localY][localX] = data[globalY * width + globalX];
    }
    
    // 加载边界
    if (threadIdx.x == 0 && globalX > 0) {
        sharedData[localY][0] = data[globalY * width + globalX - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && globalX < width - 1) {
        sharedData[localY][blockDim.x + 1] = data[globalY * width + globalX + 1];
    }
    if (threadIdx.y == 0 && globalY > 0) {
        sharedData[0][localX] = data[(globalY - 1) * width + globalX];
    }
    if (threadIdx.y == blockDim.y - 1 && globalY < height - 1) {
        sharedData[blockDim.y + 1][localX] = data[(globalY + 1) * width + globalX];
    }
    
    // 加载角点
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (globalX > 0 && globalY > 0)
            sharedData[0][0] = data[(globalY - 1) * width + globalX - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        if (globalX < width - 1 && globalY > 0)
            sharedData[0][blockDim.x + 1] = data[(globalY - 1) * width + globalX + 1];
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        if (globalX > 0 && globalY < height - 1)
            sharedData[blockDim.y + 1][0] = data[(globalY + 1) * width + globalX - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        if (globalX < width - 1 && globalY < height - 1)
            sharedData[blockDim.y + 1][blockDim.x + 1] = data[(globalY + 1) * width + globalX + 1];
    }
    
    __syncthreads();
    
    if (globalX >= width || globalY >= height) return;
    
    int idx = globalY * width + globalX;
    
    // 计算X方向导数
    if (globalX == 0) {
        // 前向差分
        derivX[idx] = (sharedData[localY][localX + 1] - sharedData[localY][localX]) * dx;
    } else if (globalX == width - 1) {
        // 后向差分
        derivX[idx] = (sharedData[localY][localX] - sharedData[localY][localX - 1]) * dx;
    } else {
        // PCHIP斜率
        float dxLeft = sharedData[localY][localX] - sharedData[localY][localX - 1];
        float dxRight = sharedData[localY][localX + 1] - sharedData[localY][localX];
        derivX[idx] = pchipSlopeOptimized(1.0f, 1.0f, dxLeft, dxRight) * dx;
    }
    
    // 计算Y方向导数
    if (globalY == 0) {
        // 前向差分
        derivY[idx] = (sharedData[localY + 1][localX] - sharedData[localY][localX]) * dy;
    } else if (globalY == height - 1) {
        // 后向差分
        derivY[idx] = (sharedData[localY][localX] - sharedData[localY - 1][localX]) * dy;
    } else {
        // PCHIP斜率
        float dyUp = sharedData[localY][localX] - sharedData[localY - 1][localX];
        float dyDown = sharedData[localY + 1][localX] - sharedData[localY][localX];
        derivY[idx] = pchipSlopeOptimized(1.0f, 1.0f, dyUp, dyDown) * dy;
    }
    
    // 计算交叉导数（使用导数的导数）
    float dxySum = 0.0f;
    int count = 0;
    
    // 左上
    if (globalX > 0 && globalY > 0) {
        float dxy = (sharedData[localY][localX] - sharedData[localY][localX - 1] -
                     sharedData[localY - 1][localX] + sharedData[localY - 1][localX - 1]);
        dxySum += dxy;
        count++;
    }
    
    // 右上
    if (globalX < width - 1 && globalY > 0) {
        float dxy = (sharedData[localY][localX + 1] - sharedData[localY][localX] -
                     sharedData[localY - 1][localX + 1] + sharedData[localY - 1][localX]);
        dxySum += dxy;
        count++;
    }
    
    // 左下
    if (globalX > 0 && globalY < height - 1) {
        float dxy = (sharedData[localY + 1][localX] - sharedData[localY + 1][localX - 1] -
                     sharedData[localY][localX] + sharedData[localY][localX - 1]);
        dxySum += dxy;
        count++;
    }
    
    // 右下
    if (globalX < width - 1 && globalY < height - 1) {
        float dxy = (sharedData[localY + 1][localX + 1] - sharedData[localY + 1][localX] -
                     sharedData[localY][localX + 1] + sharedData[localY][localX]);
        dxySum += dxy;
        count++;
    }
    
    derivXY[idx] = (count > 0) ? (dxySum / count) * dx * dy : 0.0f;
}

// C++接口函数
extern "C" {

/**
 * @brief 执行优化的PCHIP 2D插值（预计算导数版本）
 */
cudaError_t launchPCHIP2DInterpolationOptimized(
    const float* d_sourceData,
    const float* d_derivX,
    const float* d_derivY,
    const float* d_derivXY,
    float* d_outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue,
    cudaStream_t stream) {
    
    // 预计算缩放因子和偏移
    float scaleX = (float)(sourceWidth - 1) / (outputWidth - 1);
    float scaleY = (float)(sourceHeight - 1) / (outputHeight - 1);
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);
    
    // 使用更大的共享内存
    size_t sharedMemSize = 4 * SHARED_MEMORY_SIZE * SHARED_MEMORY_SIZE * sizeof(float);
    
    pchip2DInterpolationOptimizedKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_sourceData, d_derivX, d_derivY, d_derivXY,
        d_outputData,
        sourceWidth, sourceHeight,
        outputWidth, outputHeight,
        scaleX, scaleY, offsetX, offsetY,
        fillValue);
    
    return cudaGetLastError();
}

/**
 * @brief 执行一体化的PCHIP 2D插值（即时计算导数版本）
 */
cudaError_t launchPCHIP2DInterpolationIntegrated(
    const float* d_sourceData,
    float* d_outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue,
    cudaStream_t stream) {
    
    // 预计算缩放因子
    float scaleX = (float)(sourceWidth - 1) / (outputWidth - 1);
    float scaleY = (float)(sourceHeight - 1) / (outputHeight - 1);
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);
    
    pchip2DInterpolationIntegratedKernel<<<gridSize, blockSize, 0, stream>>>(
        d_sourceData, d_outputData,
        sourceWidth, sourceHeight,
        outputWidth, outputHeight,
        scaleX, scaleY, offsetX, offsetY,
        fillValue);
    
    return cudaGetLastError();
}

/**
 * @brief 计算优化的PCHIP导数
 */
cudaError_t computePCHIPDerivativesOptimized(
    const float* d_data,
    float* d_derivX,
    float* d_derivY,
    float* d_derivXY,
    int width, int height,
    float dx, float dy,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    computePCHIPDerivativesOptimizedKernel<<<gridSize, blockSize, 0, stream>>>(
        d_data, d_derivX, d_derivY, d_derivXY,
        width, height, dx, dy);
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace gpu
} // namespace interpolation
} // namespace oscean 