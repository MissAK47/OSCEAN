#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>

namespace oscean::core_services::interpolation::gpu {

// 设备端辅助函数
__device__ inline float hermite(float t, float p0, float p1, float m0, float m1) {
    float t2 = t * t;
    float t3 = t2 * t;
    float h00 = 2*t3 - 3*t2 + 1;
    float h10 = t3 - 2*t2 + t;
    float h01 = -2*t3 + 3*t2;
    float h11 = t3 - t2;
    return h00*p0 + h10*m0 + h01*p1 + h11*m1;
}

// 双线性插值核函数
__global__ void bilinearKernel(
    const float* __restrict__ gridData,
    const float* __restrict__ xCoords,
    const float* __restrict__ yCoords,
    float* __restrict__ results,
    int numPoints,
    int gridWidth,
    int gridHeight) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;
    
    float x = xCoords[tid];
    float y = yCoords[tid];
    
    // 计算网格索引
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    
    // 边界检查
    if (x0 < 0 || x0 >= gridWidth - 1 || y0 < 0 || y0 >= gridHeight - 1) {
        results[tid] = 0.0f;
        return;
    }
    
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // 计算权重
    float wx = x - x0;
    float wy = y - y0;
    
    // 获取四个角点的值
    float v00 = gridData[y0 * gridWidth + x0];
    float v10 = gridData[y0 * gridWidth + x1];
    float v01 = gridData[y1 * gridWidth + x0];
    float v11 = gridData[y1 * gridWidth + x1];
    
    // 双线性插值
    float v0 = v00 * (1 - wx) + v10 * wx;
    float v1 = v01 * (1 - wx) + v11 * wx;
    results[tid] = v0 * (1 - wy) + v1 * wy;
}

// 2D PCHIP插值核函数（使用共享内存优化）
__global__ void pchip2DKernel(
    const float* __restrict__ gridData,
    const float* __restrict__ xCoords,
    const float* __restrict__ yCoords,
    float* __restrict__ results,
    int numPoints,
    int gridWidth,
    int gridHeight,
    const float* __restrict__ dervX,
    const float* __restrict__ dervY,
    const float* __restrict__ dervXY) {
    
    // 共享内存用于缓存局部数据
    extern __shared__ float sharedMem[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;
    
    float x = xCoords[tid];
    float y = yCoords[tid];
    
    // 计算网格索引
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    
    // 边界检查
    if (x0 < 0 || x0 >= gridWidth - 1 || y0 < 0 || y0 >= gridHeight - 1) {
        results[tid] = 0.0f;
        return;
    }
    
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // 归一化坐标
    float tx = x - x0;
    float ty = y - y0;
    
    // 获取四个角点的值和导数
    int idx00 = y0 * gridWidth + x0;
    int idx10 = y0 * gridWidth + x1;
    int idx01 = y1 * gridWidth + x0;
    int idx11 = y1 * gridWidth + x1;
    
    float p[4] = {
        gridData[idx00], gridData[idx10],
        gridData[idx01], gridData[idx11]
    };
    
    float fx[4] = {
        dervX[idx00], dervX[idx10],
        dervX[idx01], dervX[idx11]
    };
    
    float fy[4] = {
        dervY[idx00], dervY[idx10],
        dervY[idx01], dervY[idx11]
    };
    
    float fxy[4] = {
        dervXY[idx00], dervXY[idx10],
        dervXY[idx01], dervXY[idx11]
    };
    
    // X方向的PCHIP插值
    float v0 = hermite(tx, p[0], p[1], fx[0], fx[1]);
    float v1 = hermite(tx, p[2], p[3], fx[2], fx[3]);
    
    // Y方向导数的插值
    float m0y = hermite(tx, fy[0], fy[1], fxy[0], fxy[1]);
    float m1y = hermite(tx, fy[2], fy[3], fxy[2], fxy[3]);
    
    // Y方向的最终PCHIP插值
    results[tid] = hermite(ty, v0, v1, m0y, m1y);
}

// 三线性插值核函数
__global__ void trilinearKernel(
    const float* __restrict__ gridData,
    const float* __restrict__ xCoords,
    const float* __restrict__ yCoords,
    const float* __restrict__ zCoords,
    float* __restrict__ results,
    int numPoints,
    int gridWidth,
    int gridHeight,
    int gridDepth) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;
    
    float x = xCoords[tid];
    float y = yCoords[tid];
    float z = zCoords[tid];
    
    // 计算网格索引
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int z0 = (int)floorf(z);
    
    // 边界检查
    if (x0 < 0 || x0 >= gridWidth - 1 || 
        y0 < 0 || y0 >= gridHeight - 1 ||
        z0 < 0 || z0 >= gridDepth - 1) {
        results[tid] = 0.0f;
        return;
    }
    
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    
    // 计算权重
    float wx = x - x0;
    float wy = y - y0;
    float wz = z - z0;
    
    // 获取8个角点的值
    float v000 = gridData[(z0 * gridHeight + y0) * gridWidth + x0];
    float v100 = gridData[(z0 * gridHeight + y0) * gridWidth + x1];
    float v010 = gridData[(z0 * gridHeight + y1) * gridWidth + x0];
    float v110 = gridData[(z0 * gridHeight + y1) * gridWidth + x1];
    float v001 = gridData[(z1 * gridHeight + y0) * gridWidth + x0];
    float v101 = gridData[(z1 * gridHeight + y0) * gridWidth + x1];
    float v011 = gridData[(z1 * gridHeight + y1) * gridWidth + x0];
    float v111 = gridData[(z1 * gridHeight + y1) * gridWidth + x1];
    
    // 三线性插值
    float v00 = v000 * (1 - wx) + v100 * wx;
    float v10 = v010 * (1 - wx) + v110 * wx;
    float v01 = v001 * (1 - wx) + v101 * wx;
    float v11 = v011 * (1 - wx) + v111 * wx;
    
    float v0 = v00 * (1 - wy) + v10 * wy;
    float v1 = v01 * (1 - wy) + v11 * wy;
    
    results[tid] = v0 * (1 - wz) + v1 * wz;
}

// 批量处理的优化版本
__global__ void bilinearBatchKernel(
    const float* __restrict__ gridData,
    const float2* __restrict__ points,  // x,y坐标打包
    float* __restrict__ results,
    int numPoints,
    int gridWidth,
    int gridHeight,
    int gridPitch) {  // 内存对齐的行宽度
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;
    
    float2 point = points[tid];
    float x = point.x;
    float y = point.y;
    
    // 使用纹理内存的版本会更快，但这里先用全局内存
    int x0 = __float2int_rd(x);
    int y0 = __float2int_rd(y);
    
    if (x0 < 0 || x0 >= gridWidth - 1 || y0 < 0 || y0 >= gridHeight - 1) {
        results[tid] = 0.0f;
        return;
    }
    
    float wx = x - x0;
    float wy = y - y0;
    
    // 使用pitch优化内存访问
    const float* row0 = gridData + y0 * gridPitch;
    const float* row1 = gridData + (y0 + 1) * gridPitch;
    
    float v00 = row0[x0];
    float v10 = row0[x0 + 1];
    float v01 = row1[x0];
    float v11 = row1[x0 + 1];
    
    float v0 = fmaf(wx, v10 - v00, v00);
    float v1 = fmaf(wx, v11 - v01, v01);
    results[tid] = fmaf(wy, v1 - v0, v0);
}

// 复数场插值核函数（用于RAM）
__global__ void complexFieldPCHIPKernel(
    const float2* __restrict__ complexData,  // 复数数据（实部，虚部）
    const float* __restrict__ ranges,
    const float* __restrict__ depths,
    float2* __restrict__ results,
    int numPoints,
    int numRanges,
    int numDepths,
    const float* __restrict__ dervR,
    const float* __restrict__ dervD) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;
    
    float r = ranges[tid];
    float d = depths[tid];
    
    int r0 = (int)floorf(r);
    int d0 = (int)floorf(d);
    
    if (r0 < 0 || r0 >= numRanges - 1 || d0 < 0 || d0 >= numDepths - 1) {
        results[tid] = make_float2(0.0f, 0.0f);
        return;
    }
    
    float tr = r - r0;
    float td = d - d0;
    
    // 获取四个角点的复数值
    int idx00 = d0 * numRanges + r0;
    int idx10 = d0 * numRanges + r0 + 1;
    int idx01 = (d0 + 1) * numRanges + r0;
    int idx11 = (d0 + 1) * numRanges + r0 + 1;
    
    float2 c00 = complexData[idx00];
    float2 c10 = complexData[idx10];
    float2 c01 = complexData[idx01];
    float2 c11 = complexData[idx11];
    
    // 分别对实部和虚部进行PCHIP插值
    // 实部
    float realResult = hermite(tr, 
        hermite(td, c00.x, c01.x, dervD[idx00], dervD[idx01]),
        hermite(td, c10.x, c11.x, dervD[idx10], dervD[idx11]),
        dervR[idx00], dervR[idx10]);
    
    // 虚部
    float imagResult = hermite(tr,
        hermite(td, c00.y, c01.y, dervD[idx00], dervD[idx01]),
        hermite(td, c10.y, c11.y, dervD[idx10], dervD[idx11]),
        dervR[idx00], dervR[idx10]);
    
    results[tid] = make_float2(realResult, imagResult);
}

} // namespace oscean::core_services::interpolation::gpu 