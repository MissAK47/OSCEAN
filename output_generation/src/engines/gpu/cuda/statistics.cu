/**
 * @file statistics.cu
 * @brief CUDA统计计算核函数实现
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cmath>
#include <cfloat>

namespace oscean {
namespace output_generation {
namespace gpu {
namespace cuda {

// 块大小定义
#define BLOCK_SIZE 256
#define WARP_SIZE 32

/**
 * @brief 均值计算核函数
 */
__global__ void meanKernel(
    const float* data,
    float* partialSums,
    int* partialCounts,
    int size) {
    
    extern __shared__ float sharedData[];
    float* sSum = sharedData;
    int* sCount = (int*)&sSum[blockDim.x];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 初始化
    float sum = 0.0f;
    int count = 0;
    
    // 每个线程处理多个元素
    for (int i = gid; i < size; i += gridDim.x * blockDim.x) {
        float val = data[i];
        if (!isnan(val)) {
            sum += val;
            count++;
        }
    }
    
    sSum[tid] = sum;
    sCount[tid] = count;
    __syncthreads();
    
    // 块内规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sSum[tid] += sSum[tid + s];
            sCount[tid] += sCount[tid + s];
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        partialSums[blockIdx.x] = sSum[0];
        partialCounts[blockIdx.x] = sCount[0];
    }
}

/**
 * @brief 标准差计算核函数（第一步：计算平方差和）
 */
__global__ void stdDevKernel(
    const float* data,
    float mean,
    float* partialSums,
    int* partialCounts,
    int size) {
    
    extern __shared__ float sharedData[];
    float* sSum = sharedData;
    int* sCount = (int*)&sSum[blockDim.x];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int i = gid; i < size; i += gridDim.x * blockDim.x) {
        float val = data[i];
        if (!isnan(val)) {
            float diff = val - mean;
            sum += diff * diff;
            count++;
        }
    }
    
    sSum[tid] = sum;
    sCount[tid] = count;
    __syncthreads();
    
    // 块内规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sSum[tid] += sSum[tid + s];
            sCount[tid] += sCount[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partialSums[blockIdx.x] = sSum[0];
        partialCounts[blockIdx.x] = sCount[0];
    }
}

/**
 * @brief 直方图计算核函数
 */
__global__ void histogramKernel(
    const float* data,
    int* histogram,
    float minValue,
    float maxValue,
    int numBins,
    int size) {
    
    extern __shared__ int sharedHist[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 初始化共享内存直方图
    for (int i = tid; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();
    
    // 计算直方图
    float binWidth = (maxValue - minValue) / numBins;
    
    for (int i = gid; i < size; i += gridDim.x * blockDim.x) {
        float val = data[i];
        if (!isnan(val) && val >= minValue && val <= maxValue) {
            int bin = min((int)((val - minValue) / binWidth), numBins - 1);
            atomicAdd(&sharedHist[bin], 1);
        }
    }
    __syncthreads();
    
    // 写入全局内存
    for (int i = tid; i < numBins; i += blockDim.x) {
        atomicAdd(&histogram[i], sharedHist[i]);
    }
}

/**
 * @brief 百分位数计算的辅助核函数（排序后选择）
 */
__global__ void percentileSelectKernel(
    const float* sortedData,
    float* percentiles,
    const float* percentileValues,
    int numPercentiles,
    int validCount) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPercentiles) return;
    
    float p = percentileValues[tid];
    int index = (int)(p * (validCount - 1) / 100.0f);
    index = min(max(index, 0), validCount - 1);
    
    percentiles[tid] = sortedData[index];
}

/**
 * @brief 数据有效性检查和压缩核函数
 */
__global__ void compactValidDataKernel(
    const float* input,
    float* output,
    int* validIndices,
    int size) {
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    
    float val = input[gid];
    if (!isnan(val)) {
        int idx = validIndices[gid];
        output[idx] = val;
    }
}

// C++接口函数
extern "C" {

/**
 * @brief 计算均值
 */
cudaError_t computeMeanGPU(
    const float* d_data,
    float* h_mean,
    int size,
    cudaStream_t stream) {
    
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024); // 限制块数
    
    // 分配部分和缓冲区
    float* d_partialSums;
    int* d_partialCounts;
    cudaMalloc(&d_partialSums, numBlocks * sizeof(float));
    cudaMalloc(&d_partialCounts, numBlocks * sizeof(int));
    
    // 第一步：部分求和
    size_t sharedSize = BLOCK_SIZE * (sizeof(float) + sizeof(int));
    meanKernel<<<numBlocks, BLOCK_SIZE, sharedSize, stream>>>(
        d_data, d_partialSums, d_partialCounts, size);
    
    // 第二步：最终规约（可以在CPU上完成或使用另一个kernel）
    float* h_partialSums = new float[numBlocks];
    int* h_partialCounts = new int[numBlocks];
    
    cudaMemcpyAsync(h_partialSums, d_partialSums, 
                    numBlocks * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_partialCounts, d_partialCounts, 
                    numBlocks * sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // CPU上完成最终计算
    double totalSum = 0.0;
    int totalCount = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += h_partialSums[i];
        totalCount += h_partialCounts[i];
    }
    
    *h_mean = (totalCount > 0) ? (float)(totalSum / totalCount) : 0.0f;
    
    // 清理
    delete[] h_partialSums;
    delete[] h_partialCounts;
    cudaFree(d_partialSums);
    cudaFree(d_partialCounts);
    
    return cudaGetLastError();
}

/**
 * @brief 计算标准差
 */
cudaError_t computeStdDevGPU(
    const float* d_data,
    float mean,
    float* h_stddev,
    int size,
    cudaStream_t stream) {
    
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024);
    
    // 分配缓冲区
    float* d_partialSums;
    int* d_partialCounts;
    cudaMalloc(&d_partialSums, numBlocks * sizeof(float));
    cudaMalloc(&d_partialCounts, numBlocks * sizeof(int));
    
    // 计算平方差和
    size_t sharedSize = BLOCK_SIZE * (sizeof(float) + sizeof(int));
    stdDevKernel<<<numBlocks, BLOCK_SIZE, sharedSize, stream>>>(
        d_data, mean, d_partialSums, d_partialCounts, size);
    
    // 最终规约
    float* h_partialSums = new float[numBlocks];
    int* h_partialCounts = new int[numBlocks];
    
    cudaMemcpyAsync(h_partialSums, d_partialSums, 
                    numBlocks * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_partialCounts, d_partialCounts, 
                    numBlocks * sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // CPU计算最终结果
    double totalSum = 0.0;
    int totalCount = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += h_partialSums[i];
        totalCount += h_partialCounts[i];
    }
    
    *h_stddev = (totalCount > 1) ? 
                sqrtf((float)(totalSum / (totalCount - 1))) : 0.0f;
    
    // 清理
    delete[] h_partialSums;
    delete[] h_partialCounts;
    cudaFree(d_partialSums);
    cudaFree(d_partialCounts);
    
    return cudaGetLastError();
}

/**
 * @brief 计算直方图
 */
cudaError_t computeHistogramGPU(
    const float* d_data,
    int* h_histogram,
    float minValue,
    float maxValue,
    int numBins,
    int size,
    cudaStream_t stream) {
    
    // 分配设备端直方图
    int* d_histogram;
    cudaMalloc(&d_histogram, numBins * sizeof(int));
    cudaMemsetAsync(d_histogram, 0, numBins * sizeof(int), stream);
    
    // 计算块数和共享内存大小
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024);
    size_t sharedSize = numBins * sizeof(int);
    
    // 执行直方图计算
    histogramKernel<<<numBlocks, BLOCK_SIZE, sharedSize, stream>>>(
        d_data, d_histogram, minValue, maxValue, numBins, size);
    
    // 复制结果到主机
    cudaMemcpyAsync(h_histogram, d_histogram, 
                    numBins * sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaFree(d_histogram);
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace gpu
} // namespace output_generation
} // namespace oscean 