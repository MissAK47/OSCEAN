/**
 * @file min_max_reduction.cu
 * @brief CUDA核函数实现GPU上的最小最大值计算
 */

#include <cuda_runtime.h>
#include <limits>
#include <cfloat>

namespace oscean::output_generation::gpu {

/**
 * @brief CUDA核函数：并行归约计算最小最大值
 * 
 * 使用共享内存优化的并行归约算法
 */
__global__ void minMaxReductionKernel(
    const float* __restrict__ input,
    float* __restrict__ minOutput,
    float* __restrict__ maxOutput,
    size_t numElements)
{
    extern __shared__ float sharedMem[];
    float* sMin = sharedMem;
    float* sMax = &sharedMem[blockDim.x];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    
    // 初始化为极值
    float localMin = FLT_MAX;
    float localMax = -FLT_MAX;
    
    // 网格跨步循环，每个线程处理多个元素
    while (i < numElements) {
        // 每个线程处理两个元素（提高内存带宽利用率）
        if (i < numElements) {
            float val = input[i];
            localMin = fminf(localMin, val);
            localMax = fmaxf(localMax, val);
        }
        if (i + blockDim.x < numElements) {
            float val = input[i + blockDim.x];
            localMin = fminf(localMin, val);
            localMax = fmaxf(localMax, val);
        }
        i += gridSize;
    }
    
    // 将结果存储到共享内存
    sMin[tid] = localMin;
    sMax[tid] = localMax;
    __syncthreads();
    
    // 在共享内存中进行归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sMin[tid] = fminf(sMin[tid], sMin[tid + s]);
            sMax[tid] = fmaxf(sMax[tid], sMax[tid + s]);
        }
        __syncthreads();
    }
    
    // 线程0将块结果写入全局内存
    if (tid == 0) {
        minOutput[blockIdx.x] = sMin[0];
        maxOutput[blockIdx.x] = sMax[0];
    }
}

/**
 * @brief 最终归约核函数
 * 
 * 对各个块的结果进行最终归约
 */
__global__ void finalReductionKernel(
    float* __restrict__ minData,
    float* __restrict__ maxData,
    float* __restrict__ finalMin,
    float* __restrict__ finalMax,
    unsigned int numBlocks)
{
    extern __shared__ float sharedMem[];
    float* sMin = sharedMem;
    float* sMax = &sharedMem[blockDim.x];
    
    unsigned int tid = threadIdx.x;
    
    // 初始化
    float localMin = FLT_MAX;
    float localMax = -FLT_MAX;
    
    // 每个线程处理多个块结果
    for (unsigned int i = tid; i < numBlocks; i += blockDim.x) {
        localMin = fminf(localMin, minData[i]);
        localMax = fmaxf(localMax, maxData[i]);
    }
    
    sMin[tid] = localMin;
    sMax[tid] = localMax;
    __syncthreads();
    
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sMin[tid] = fminf(sMin[tid], sMin[tid + s]);
            sMax[tid] = fmaxf(sMax[tid], sMax[tid + s]);
        }
        __syncthreads();
    }
    
    // 写入最终结果
    if (tid == 0) {
        *finalMin = sMin[0];
        *finalMax = sMax[0];
    }
}

extern "C" {

/**
 * @brief 在GPU上计算数组的最小值和最大值
 * 
 * @param d_data 设备端数据指针
 * @param numElements 元素数量
 * @param h_min 主机端最小值结果指针
 * @param h_max 主机端最大值结果指针
 * @return cudaError_t CUDA错误码
 */
cudaError_t computeMinMaxGPU(
    const float* d_data,
    size_t numElements,
    float* h_min,
    float* h_max)
{
    if (numElements == 0) {
        return cudaErrorInvalidValue;
    }
    
    // 计算网格和块大小
    const int blockSize = 256;
    const int maxBlocks = 1024;
    int numBlocks = (numElements + blockSize * 2 - 1) / (blockSize * 2);
    numBlocks = (numBlocks > maxBlocks) ? maxBlocks : numBlocks;
    
    // 分配中间结果内存
    float *d_blockMin, *d_blockMax;
    cudaError_t err = cudaMalloc(&d_blockMin, numBlocks * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_blockMax, numBlocks * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_blockMin);
        return err;
    }
    
    // 分配最终结果内存
    float *d_finalMin, *d_finalMax;
    err = cudaMalloc(&d_finalMin, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_blockMin);
        cudaFree(d_blockMax);
        return err;
    }
    
    err = cudaMalloc(&d_finalMax, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_blockMin);
        cudaFree(d_blockMax);
        cudaFree(d_finalMin);
        return err;
    }
    
    // 共享内存大小
    size_t sharedMemSize = blockSize * 2 * sizeof(float);
    
    // 第一步：并行归约
    minMaxReductionKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        d_data, d_blockMin, d_blockMax, numElements);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_blockMin);
        cudaFree(d_blockMax);
        cudaFree(d_finalMin);
        cudaFree(d_finalMax);
        return err;
    }
    
    // 第二步：最终归约
    finalReductionKernel<<<1, blockSize, sharedMemSize>>>(
        d_blockMin, d_blockMax, d_finalMin, d_finalMax, numBlocks);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_blockMin);
        cudaFree(d_blockMax);
        cudaFree(d_finalMin);
        cudaFree(d_finalMax);
        return err;
    }
    
    // 将结果复制回主机
    err = cudaMemcpy(h_min, d_finalMin, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_blockMin);
        cudaFree(d_blockMax);
        cudaFree(d_finalMin);
        cudaFree(d_finalMax);
        return err;
    }
    
    err = cudaMemcpy(h_max, d_finalMax, sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理内存
    cudaFree(d_blockMin);
    cudaFree(d_blockMax);
    cudaFree(d_finalMin);
    cudaFree(d_finalMax);
    
    return err;
}

} // extern "C"

} // namespace oscean::output_generation::gpu 