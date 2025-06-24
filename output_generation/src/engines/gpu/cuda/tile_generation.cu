/**
 * @file tile_generation.cu
 * @brief CUDA瓦片生成核函数实现
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

// 定义M_PI（如果未定义）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace oscean {
namespace output_generation {
namespace gpu {
namespace cuda {

// 常量定义
#define TILE_SIZE 256
#define WARP_SIZE 32

// 设备端颜色查找表（从color_mapping.cu共享）
extern __constant__ float d_colorLUT[256 * 4];

/**
 * @brief 双线性插值
 */
__device__ float bilinearInterpolate(
    const float* data,
    float x, float y,
    int width, int height) {
    
    // 边界检查
    x = fmaxf(0.0f, fminf(x, width - 1.0f));
    y = fmaxf(0.0f, fminf(y, height - 1.0f));
    
    int x0 = __float2int_rd(x);
    int y0 = __float2int_rd(y);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    float fx = x - x0;
    float fy = y - y0;
    
    // 获取四个角的值
    float v00 = data[y0 * width + x0];
    float v10 = data[y0 * width + x1];
    float v01 = data[y1 * width + x0];
    float v11 = data[y1 * width + x1];
    
    // 双线性插值
    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;
    
    return v0 * (1.0f - fy) + v1 * fy;
}

/**
 * @brief Web Mercator投影转换
 */
__device__ void mercatorToLatLon(
    float x, float y, int z,
    float& lon, float& lat) {
    
    float n = powf(2.0f, z);
    lon = x / n * 360.0f - 180.0f;
    
    float latRad = atanf(sinhf(M_PI * (1.0f - 2.0f * y / n)));
    lat = latRad * 180.0f / M_PI;
}

/**
 * @brief 经纬度到数据网格坐标转换
 */
__device__ void latLonToGrid(
    float lon, float lat,
    float minLon, float maxLon,
    float minLat, float maxLat,
    int gridWidth, int gridHeight,
    float& gridX, float& gridY) {
    
    gridX = (lon - minLon) / (maxLon - minLon) * (gridWidth - 1);
    gridY = (lat - minLat) / (maxLat - minLat) * (gridHeight - 1);
}

/**
 * @brief 基础瓦片生成核函数
 */
__global__ void tileGenerationKernel(
    const float* gridData,
    uint8_t* tileData,
    int tileX, int tileY, int zoomLevel,
    int tileSize,
    int gridWidth, int gridHeight,
    float minLon, float maxLon,
    float minLat, float maxLat,
    float minValue, float maxValue) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= tileSize || y >= tileSize) return;
    
    // 计算瓦片中像素对应的经纬度
    float pixelX = tileX + (float)x / tileSize;
    float pixelY = tileY + (float)y / tileSize;
    
    float lon, lat;
    mercatorToLatLon(pixelX, pixelY, zoomLevel, lon, lat);
    
    // 转换到网格坐标
    float gridX, gridY;
    latLonToGrid(lon, lat, minLon, maxLon, minLat, maxLat,
                 gridWidth, gridHeight, gridX, gridY);
    
    // 双线性插值获取数据值
    float value = bilinearInterpolate(gridData, gridX, gridY, gridWidth, gridHeight);
    
    // 处理NaN值
    if (isnan(value)) {
        int idx = y * tileSize + x;
        tileData[idx * 4 + 0] = 0;
        tileData[idx * 4 + 1] = 0;
        tileData[idx * 4 + 2] = 0;
        tileData[idx * 4 + 3] = 0;
        return;
    }
    
    // 归一化到[0, 1]
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = fmaxf(0.0f, fminf(1.0f, normalizedValue));
    
    // 映射到颜色
    float lutIndex = normalizedValue * 255.0f;
    int index0 = __float2int_rd(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    // 从LUT中获取颜色
    float r = d_colorLUT[index0 * 4 + 0] * (1.0f - frac) + d_colorLUT[index1 * 4 + 0] * frac;
    float g = d_colorLUT[index0 * 4 + 1] * (1.0f - frac) + d_colorLUT[index1 * 4 + 1] * frac;
    float b = d_colorLUT[index0 * 4 + 2] * (1.0f - frac) + d_colorLUT[index1 * 4 + 2] * frac;
    float a = d_colorLUT[index0 * 4 + 3] * (1.0f - frac) + d_colorLUT[index1 * 4 + 3] * frac;
    
    // 写入输出
    int idx = y * tileSize + x;
    tileData[idx * 4 + 0] = __float2uint_rn(r * 255.0f);
    tileData[idx * 4 + 1] = __float2uint_rn(g * 255.0f);
    tileData[idx * 4 + 2] = __float2uint_rn(b * 255.0f);
    tileData[idx * 4 + 3] = __float2uint_rn(a * 255.0f);
}

/**
 * @brief 高级瓦片生成核函数（支持抗锯齿）
 */
__global__ void tileGenerationAntialiasingKernel(
    const float* gridData,
    uint8_t* tileData,
    int tileX, int tileY, int zoomLevel,
    int tileSize,
    int gridWidth, int gridHeight,
    float minLon, float maxLon,
    float minLat, float maxLat,
    float minValue, float maxValue,
    int sampleCount) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= tileSize || y >= tileSize) return;
    
    // 超采样抗锯齿
    float r = 0, g = 0, b = 0, a = 0;
    float sampleStep = 1.0f / sampleCount;
    
    for (int sy = 0; sy < sampleCount; ++sy) {
        for (int sx = 0; sx < sampleCount; ++sx) {
            // 计算子像素位置
            float subX = x + (sx + 0.5f) * sampleStep;
            float subY = y + (sy + 0.5f) * sampleStep;
            
            float pixelX = tileX + subX / tileSize;
            float pixelY = tileY + subY / tileSize;
            
            float lon, lat;
            mercatorToLatLon(pixelX, pixelY, zoomLevel, lon, lat);
            
            float gridX, gridY;
            latLonToGrid(lon, lat, minLon, maxLon, minLat, maxLat,
                        gridWidth, gridHeight, gridX, gridY);
            
            float value = bilinearInterpolate(gridData, gridX, gridY, gridWidth, gridHeight);
            
            if (!isnan(value)) {
                float normalizedValue = (value - minValue) / (maxValue - minValue);
                normalizedValue = fmaxf(0.0f, fminf(1.0f, normalizedValue));
                
                float lutIndex = normalizedValue * 255.0f;
                int index0 = __float2int_rd(lutIndex);
                int index1 = min(index0 + 1, 255);
                float frac = lutIndex - index0;
                
                r += d_colorLUT[index0 * 4 + 0] * (1.0f - frac) + d_colorLUT[index1 * 4 + 0] * frac;
                g += d_colorLUT[index0 * 4 + 1] * (1.0f - frac) + d_colorLUT[index1 * 4 + 1] * frac;
                b += d_colorLUT[index0 * 4 + 2] * (1.0f - frac) + d_colorLUT[index1 * 4 + 2] * frac;
                a += d_colorLUT[index0 * 4 + 3] * (1.0f - frac) + d_colorLUT[index1 * 4 + 3] * frac;
            }
        }
    }
    
    // 平均化
    float invSamples = 1.0f / (sampleCount * sampleCount);
    r *= invSamples;
    g *= invSamples;
    b *= invSamples;
    a *= invSamples;
    
    // 写入输出
    int idx = y * tileSize + x;
    tileData[idx * 4 + 0] = __float2uint_rn(r * 255.0f);
    tileData[idx * 4 + 1] = __float2uint_rn(g * 255.0f);
    tileData[idx * 4 + 2] = __float2uint_rn(b * 255.0f);
    tileData[idx * 4 + 3] = __float2uint_rn(a * 255.0f);
}

/**
 * @brief 批量瓦片生成核函数
 */
__global__ void batchTileGenerationKernel(
    const float* gridData,
    uint8_t** tileDatas,
    const int* tileXs,
    const int* tileYs,
    int zoomLevel,
    int tileSize,
    int gridWidth, int gridHeight,
    float minLon, float maxLon,
    float minLat, float maxLat,
    float minValue, float maxValue,
    int batchSize) {
    
    int tileIdx = blockIdx.z;
    if (tileIdx >= batchSize) return;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= tileSize || y >= tileSize) return;
    
    int tileX = tileXs[tileIdx];
    int tileY = tileYs[tileIdx];
    uint8_t* tileData = tileDatas[tileIdx];
    
    // 计算瓦片中像素对应的经纬度
    float pixelX = tileX + (float)x / tileSize;
    float pixelY = tileY + (float)y / tileSize;
    
    float lon, lat;
    mercatorToLatLon(pixelX, pixelY, zoomLevel, lon, lat);
    
    // 转换到网格坐标
    float gridX, gridY;
    latLonToGrid(lon, lat, minLon, maxLon, minLat, maxLat,
                 gridWidth, gridHeight, gridX, gridY);
    
    // 双线性插值获取数据值
    float value = bilinearInterpolate(gridData, gridX, gridY, gridWidth, gridHeight);
    
    // 颜色映射（同单瓦片版本）
    if (isnan(value)) {
        int idx = y * tileSize + x;
        tileData[idx * 4 + 0] = 0;
        tileData[idx * 4 + 1] = 0;
        tileData[idx * 4 + 2] = 0;
        tileData[idx * 4 + 3] = 0;
        return;
    }
    
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = fmaxf(0.0f, fminf(1.0f, normalizedValue));
    
    float lutIndex = normalizedValue * 255.0f;
    int index0 = __float2int_rd(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    float r = d_colorLUT[index0 * 4 + 0] * (1.0f - frac) + d_colorLUT[index1 * 4 + 0] * frac;
    float g = d_colorLUT[index0 * 4 + 1] * (1.0f - frac) + d_colorLUT[index1 * 4 + 1] * frac;
    float b = d_colorLUT[index0 * 4 + 2] * (1.0f - frac) + d_colorLUT[index1 * 4 + 2] * frac;
    float a = d_colorLUT[index0 * 4 + 3] * (1.0f - frac) + d_colorLUT[index1 * 4 + 3] * frac;
    
    int idx = y * tileSize + x;
    tileData[idx * 4 + 0] = __float2uint_rn(r * 255.0f);
    tileData[idx * 4 + 1] = __float2uint_rn(g * 255.0f);
    tileData[idx * 4 + 2] = __float2uint_rn(b * 255.0f);
    tileData[idx * 4 + 3] = __float2uint_rn(a * 255.0f);
}

// C++接口函数
extern "C" {

/**
 * @brief 执行瓦片生成
 */
cudaError_t launchTileGeneration(
    const float* d_gridData,
    uint8_t* d_tileData,
    int tileX, int tileY, int zoomLevel,
    int tileSize,
    int gridWidth, int gridHeight,
    float minLon, float maxLon,
    float minLat, float maxLat,
    float minValue, float maxValue,
    cudaStream_t stream) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((tileSize + blockSize.x - 1) / blockSize.x,
                  (tileSize + blockSize.y - 1) / blockSize.y);
    
    tileGenerationKernel<<<gridSize, blockSize, 0, stream>>>(
        d_gridData, d_tileData,
        tileX, tileY, zoomLevel, tileSize,
        gridWidth, gridHeight,
        minLon, maxLon, minLat, maxLat,
        minValue, maxValue);
    
    return cudaGetLastError();
}

/**
 * @brief 执行抗锯齿瓦片生成
 */
cudaError_t launchTileGenerationAntialiasing(
    const float* d_gridData,
    uint8_t* d_tileData,
    int tileX, int tileY, int zoomLevel,
    int tileSize,
    int gridWidth, int gridHeight,
    float minLon, float maxLon,
    float minLat, float maxLat,
    float minValue, float maxValue,
    int sampleCount,
    cudaStream_t stream) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((tileSize + blockSize.x - 1) / blockSize.x,
                  (tileSize + blockSize.y - 1) / blockSize.y);
    
    tileGenerationAntialiasingKernel<<<gridSize, blockSize, 0, stream>>>(
        d_gridData, d_tileData,
        tileX, tileY, zoomLevel, tileSize,
        gridWidth, gridHeight,
        minLon, maxLon, minLat, maxLat,
        minValue, maxValue, sampleCount);
    
    return cudaGetLastError();
}

/**
 * @brief 执行批量瓦片生成
 */
cudaError_t launchBatchTileGeneration(
    const float* d_gridData,
    uint8_t** d_tileDatas,
    const int* d_tileXs,
    const int* d_tileYs,
    int zoomLevel,
    int tileSize,
    int gridWidth, int gridHeight,
    float minLon, float maxLon,
    float minLat, float maxLat,
    float minValue, float maxValue,
    int batchSize,
    cudaStream_t stream) {
    
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((tileSize + blockSize.x - 1) / blockSize.x,
                  (tileSize + blockSize.y - 1) / blockSize.y,
                  batchSize);
    
    batchTileGenerationKernel<<<gridSize, blockSize, 0, stream>>>(
        d_gridData, d_tileDatas,
        d_tileXs, d_tileYs,
        zoomLevel, tileSize,
        gridWidth, gridHeight,
        minLon, maxLon, minLat, maxLat,
        minValue, maxValue, batchSize);
    
    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace gpu
} // namespace output_generation
} // namespace oscean 