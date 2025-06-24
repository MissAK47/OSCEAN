/**
 * @file image_compression.cu
 * @brief GPU加速的图像压缩实现
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>

namespace oscean::output_generation::gpu::cuda {

// 常量定义
#define BLOCK_SIZE 16
#define DCT_SIZE 8

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief DCT系数矩阵（预计算）
 */
__constant__ float c_dctMatrix[DCT_SIZE * DCT_SIZE];
__constant__ float c_dctMatrixT[DCT_SIZE * DCT_SIZE];

/**
 * @brief JPEG量化表
 */
__constant__ int c_quantTable[DCT_SIZE * DCT_SIZE] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
};

/**
 * @brief RGB到YCbCr转换核函数
 */
__global__ void rgbToYCbCrKernel(
    const uint8_t* rgbData,
    float* yData, float* cbData, float* crData,
    int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int rgbIdx = idx * 3;
    
    float r = rgbData[rgbIdx + 0];
    float g = rgbData[rgbIdx + 1];
    float b = rgbData[rgbIdx + 2];
    
    // ITU-R BT.601转换
    yData[idx]  =  0.299f * r + 0.587f * g + 0.114f * b;
    cbData[idx] = -0.169f * r - 0.331f * g + 0.500f * b + 128.0f;
    crData[idx] =  0.500f * r - 0.419f * g - 0.081f * b + 128.0f;
}

/**
 * @brief 2D DCT变换核函数
 */
__global__ void dct2dKernel(
    const float* input,
    float* output,
    int width, int height,
    int quality) {
    
    __shared__ float block[DCT_SIZE][DCT_SIZE];
    __shared__ float temp[DCT_SIZE][DCT_SIZE];
    
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 块的起始位置
    int startX = blockX * DCT_SIZE;
    int startY = blockY * DCT_SIZE;
    
    // 加载8x8块到共享内存
    if (tx < DCT_SIZE && ty < DCT_SIZE) {
        int x = startX + tx;
        int y = startY + ty;
        
        if (x < width && y < height) {
            block[ty][tx] = input[y * width + x] - 128.0f;
        } else {
            block[ty][tx] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // 行DCT
    if (tx < DCT_SIZE && ty < DCT_SIZE) {
        float sum = 0.0f;
        for (int k = 0; k < DCT_SIZE; k++) {
            sum += c_dctMatrix[ty * DCT_SIZE + k] * block[k][tx];
        }
        temp[ty][tx] = sum;
    }
    
    __syncthreads();
    
    // 列DCT
    if (tx < DCT_SIZE && ty < DCT_SIZE) {
        float sum = 0.0f;
        for (int k = 0; k < DCT_SIZE; k++) {
            sum += temp[ty][k] * c_dctMatrixT[k * DCT_SIZE + tx];
        }
        
        // 量化
        int quantValue = c_quantTable[ty * DCT_SIZE + tx];
        quantValue = (quantValue * quality + 50) / 100;
        quantValue = max(1, min(255, quantValue));
        
        int quantized = __float2int_rn(sum / quantValue);
        
        // 写回结果
        int x = startX + tx;
        int y = startY + ty;
        if (x < width && y < height) {
            output[y * width + x] = (float)quantized;
        }
    }
}

/**
 * @brief 游程编码(RLE)核函数
 */
__global__ void runLengthEncodeKernel(
    const int16_t* input,
    uint8_t* output,
    int* outputSize,
    int dataSize) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= dataSize) return;
    
    // 简化的RLE实现
    // 实际实现需要更复杂的并行算法
    
    int value = input[tid];
    int count = 1;
    
    // 向前查找相同值
    int next = tid + 1;
    while (next < dataSize && input[next] == value && count < 255) {
        count++;
        next++;
    }
    
    // 原子操作写入输出
    if (tid == 0 || input[tid] != input[tid - 1]) {
        int pos = atomicAdd(outputSize, 2);
        if (pos + 1 < dataSize * 2) {
            output[pos] = (uint8_t)count;
            output[pos + 1] = (uint8_t)value;
        }
    }
}

/**
 * @brief PNG预测滤波器核函数
 */
__global__ void pngFilterKernel(
    const uint8_t* input,
    uint8_t* output,
    int width, int height,
    int stride) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * stride + x;
    
    // Sub filter (预测器类型1)
    uint8_t left = (x > 0) ? input[idx - 1] : 0;
    output[idx] = input[idx] - left;
}

/**
 * @brief Huffman编码准备核函数（计算频率）
 */
__global__ void calculateFrequencyKernel(
    const uint8_t* data,
    int* frequency,
    int dataSize) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= dataSize) return;
    
    atomicAdd(&frequency[data[tid]], 1);
}

/**
 * @brief 差分编码核函数（用于无损压缩）
 */
__global__ void differentialEncodeKernel(
    const float* input,
    int16_t* output,
    int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float current = input[idx];
    float predictor = 0.0f;
    
    // 使用左侧和上方像素进行预测
    if (x > 0 && y > 0) {
        float left = input[idx - 1];
        float top = input[idx - width];
        float topLeft = input[idx - width - 1];
        
        // Paeth预测器
        float p = left + top - topLeft;
        float pa = fabsf(p - left);
        float pb = fabsf(p - top);
        float pc = fabsf(p - topLeft);
        
        if (pa <= pb && pa <= pc) {
            predictor = left;
        } else if (pb <= pc) {
            predictor = top;
        } else {
            predictor = topLeft;
        }
    } else if (x > 0) {
        predictor = input[idx - 1];
    } else if (y > 0) {
        predictor = input[idx - width];
    }
    
    output[idx] = (int16_t)(current - predictor);
}

/**
 * @brief 色度子采样核函数（4:2:0）
 */
__global__ void chromaSubsampleKernel(
    const float* cbInput, const float* crInput,
    float* cbOutput, float* crOutput,
    int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int outWidth = width / 2;
    int outHeight = height / 2;
    
    if (x >= outWidth || y >= outHeight) return;
    
    // 2x2块平均
    int inX = x * 2;
    int inY = y * 2;
    
    float cbSum = 0.0f;
    float crSum = 0.0f;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int idx = (inY + dy) * width + (inX + dx);
            cbSum += cbInput[idx];
            crSum += crInput[idx];
        }
    }
    
    int outIdx = y * outWidth + x;
    cbOutput[outIdx] = cbSum * 0.25f;
    crOutput[outIdx] = crSum * 0.25f;
}

// 初始化DCT矩阵的主机函数
void initializeDCTMatrix() {
    float dctMatrix[DCT_SIZE * DCT_SIZE];
    float dctMatrixT[DCT_SIZE * DCT_SIZE];
    
    // 计算DCT系数矩阵
    for (int i = 0; i < DCT_SIZE; i++) {
        for (int j = 0; j < DCT_SIZE; j++) {
            float c = (i == 0) ? 1.0f / sqrtf(2.0f) : 1.0f;
            dctMatrix[i * DCT_SIZE + j] = c * sqrtf(2.0f / DCT_SIZE) * 
                cosf((2.0f * j + 1.0f) * i * M_PI / (2.0f * DCT_SIZE));
            dctMatrixT[j * DCT_SIZE + i] = dctMatrix[i * DCT_SIZE + j];
        }
    }
    
    // 复制到常量内存
    cudaMemcpyToSymbol(c_dctMatrix, dctMatrix, sizeof(dctMatrix));
    cudaMemcpyToSymbol(c_dctMatrixT, dctMatrixT, sizeof(dctMatrixT));
}

// C接口函数
extern "C" {

/**
 * @brief GPU加速的JPEG压缩预处理
 */
cudaError_t preprocessJPEGCompression(
    const uint8_t* d_rgbData,
    float* d_yData, float* d_cbData, float* d_crData,
    float* d_yDCT, float* d_cbDCT, float* d_crDCT,
    int width, int height, int quality,
    cudaStream_t stream) {
    
    // RGB到YCbCr转换
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    rgbToYCbCrKernel<<<gridSize, blockSize, 0, stream>>>(
        d_rgbData, d_yData, d_cbData, d_crData, width, height);
    
    // 色度子采样
    dim3 subsampleGrid((width/2 + blockSize.x - 1) / blockSize.x,
                       (height/2 + blockSize.y - 1) / blockSize.y);
    
    chromaSubsampleKernel<<<subsampleGrid, blockSize, 0, stream>>>(
        d_cbData, d_crData, d_cbData, d_crData, width, height);
    
    // DCT变换
    dim3 dctGrid((width + DCT_SIZE - 1) / DCT_SIZE,
                 (height + DCT_SIZE - 1) / DCT_SIZE);
    dim3 dctBlock(DCT_SIZE, DCT_SIZE);
    
    dct2dKernel<<<dctGrid, dctBlock, 0, stream>>>(
        d_yData, d_yDCT, width, height, quality);
    
    // 对子采样后的色度通道进行DCT
    dim3 chromaDctGrid((width/2 + DCT_SIZE - 1) / DCT_SIZE,
                       (height/2 + DCT_SIZE - 1) / DCT_SIZE);
    
    dct2dKernel<<<chromaDctGrid, dctBlock, 0, stream>>>(
        d_cbData, d_cbDCT, width/2, height/2, quality);
    
    dct2dKernel<<<chromaDctGrid, dctBlock, 0, stream>>>(
        d_crData, d_crDCT, width/2, height/2, quality);
    
    return cudaGetLastError();
}

/**
 * @brief GPU加速的PNG压缩预处理
 */
cudaError_t preprocessPNGCompression(
    const uint8_t* d_imageData,
    uint8_t* d_filteredData,
    int16_t* d_diffData,
    int width, int height, int channels,
    cudaStream_t stream) {
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // PNG滤波
    pngFilterKernel<<<gridSize, blockSize, 0, stream>>>(
        d_imageData, d_filteredData, width, height, width * channels);
    
    // 差分编码（用于更好的压缩）
    differentialEncodeKernel<<<gridSize, blockSize, 0, stream>>>(
        (float*)d_imageData, d_diffData, width, height);
    
    return cudaGetLastError();
}

/**
 * @brief 计算数据频率分布（用于Huffman编码）
 */
cudaError_t calculateFrequencyDistribution(
    const uint8_t* d_data,
    int* d_frequency,
    int dataSize,
    cudaStream_t stream) {
    
    // 清零频率数组
    cudaMemsetAsync(d_frequency, 0, 256 * sizeof(int), stream);
    
    // 计算频率
    int blockSize = 256;
    int gridSize = (dataSize + blockSize - 1) / blockSize;
    
    calculateFrequencyKernel<<<gridSize, blockSize, 0, stream>>>(
        d_data, d_frequency, dataSize);
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace oscean::output_generation::gpu::cuda 