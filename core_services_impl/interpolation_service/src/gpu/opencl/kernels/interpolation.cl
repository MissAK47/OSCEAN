/**
 * @file interpolation.cl
 * @brief OpenCL插值核函数实现
 */

// 双线性插值核函数
__kernel void bilinearInterpolation(
    __global const float* sourceData,
    __global float* outputData,
    const int sourceWidth,
    const int sourceHeight,
    const int outputWidth,
    const int outputHeight,
    const float minX,
    const float maxX,
    const float minY,
    const float maxY,
    const float fillValue) {
    
    int outX = get_global_id(0);
    int outY = get_global_id(1);
    
    if (outX >= outputWidth || outY >= outputHeight) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    
    // 转换到源数据的像素坐标
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    
    // 边界检查
    if (srcX < 0 || srcX > sourceWidth - 1 || srcY < 0 || srcY > sourceHeight - 1) {
        outputData[outY * outputWidth + outX] = fillValue;
        return;
    }
    
    // 计算整数坐标
    int x0 = (int)floor(srcX);
    int y0 = (int)floor(srcY);
    int x1 = min(x0 + 1, sourceWidth - 1);
    int y1 = min(y0 + 1, sourceHeight - 1);
    
    // 计算分数部分
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    // 获取四个角点的值
    float v00 = sourceData[y0 * sourceWidth + x0];
    float v10 = sourceData[y0 * sourceWidth + x1];
    float v01 = sourceData[y1 * sourceWidth + x0];
    float v11 = sourceData[y1 * sourceWidth + x1];
    
    // 处理NaN值
    if (isnan(v00)) v00 = fillValue;
    if (isnan(v10)) v10 = fillValue;
    if (isnan(v01)) v01 = fillValue;
    if (isnan(v11)) v11 = fillValue;
    
    // 双线性插值计算
    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;
    float result = v0 * (1.0f - fy) + v1 * fy;
    
    // 写入结果
    outputData[outY * outputWidth + outX] = result;
}

// 三次插值权重计算
float cubicWeight(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return -0.5f * t3 + t2 - 0.5f * t;
}

// 计算三次插值的四个权重
void getCubicWeights(float t, float weights[4]) {
    float t2 = t * t;
    float t3 = t2 * t;
    
    weights[0] = -0.5f * t3 + t2 - 0.5f * t;
    weights[1] = 1.5f * t3 - 2.5f * t2 + 1.0f;
    weights[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    weights[3] = 0.5f * t3 - 0.5f * t2;
}

// 双三次插值核函数
__kernel void bicubicInterpolation(
    __global const float* sourceData,
    __global float* outputData,
    const int sourceWidth,
    const int sourceHeight,
    const int outputWidth,
    const int outputHeight,
    const float minX,
    const float maxX,
    const float minY,
    const float maxY,
    const float fillValue) {
    
    int outX = get_global_id(0);
    int outY = get_global_id(1);
    
    if (outX >= outputWidth || outY >= outputHeight) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    
    // 转换到源数据的像素坐标
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    
    // 边界检查
    if (srcX < 1 || srcX > sourceWidth - 2 || srcY < 1 || srcY > sourceHeight - 2) {
        outputData[outY * outputWidth + outX] = fillValue;
        return;
    }
    
    // 计算整数坐标
    int x0 = (int)floor(srcX);
    int y0 = (int)floor(srcY);
    
    // 计算分数部分
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    // 计算权重
    float wx[4], wy[4];
    getCubicWeights(fx, wx);
    getCubicWeights(fy, wy);
    
    // 双三次插值
    float result = 0.0f;
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            int xi = x0 + i;
            int yi = y0 + j;
            
            // 边界处理
            xi = max(0, min(xi, sourceWidth - 1));
            yi = max(0, min(yi, sourceHeight - 1));
            
            float value = sourceData[yi * sourceWidth + xi];
            if (isnan(value)) value = fillValue;
            
            result += value * wx[i + 1] * wy[j + 1];
        }
    }
    
    outputData[outY * outputWidth + outX] = result;
}

// 三线性插值核函数
__kernel void trilinearInterpolation(
    __global const float* sourceData,
    __global float* outputData,
    const int sourceWidth,
    const int sourceHeight,
    const int sourceDepth,
    const int outputWidth,
    const int outputHeight,
    const int outputDepth,
    const float minX,
    const float maxX,
    const float minY,
    const float maxY,
    const float minZ,
    const float maxZ,
    const float fillValue) {
    
    int outX = get_global_id(0);
    int outY = get_global_id(1);
    int outZ = get_global_id(2);
    
    if (outX >= outputWidth || outY >= outputHeight || outZ >= outputDepth) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    float z = minZ + (maxZ - minZ) * outZ / (outputDepth - 1);
    
    // 转换到源数据的像素坐标
    float srcX = (x - minX) / (maxX - minX) * (sourceWidth - 1);
    float srcY = (y - minY) / (maxY - minY) * (sourceHeight - 1);
    float srcZ = (z - minZ) / (maxZ - minZ) * (sourceDepth - 1);
    
    // 边界检查
    if (srcX < 0 || srcX > sourceWidth - 1 || 
        srcY < 0 || srcY > sourceHeight - 1 ||
        srcZ < 0 || srcZ > sourceDepth - 1) {
        outputData[outZ * outputHeight * outputWidth + outY * outputWidth + outX] = fillValue;
        return;
    }
    
    // 计算整数坐标
    int x0 = (int)floor(srcX);
    int y0 = (int)floor(srcY);
    int z0 = (int)floor(srcZ);
    int x1 = min(x0 + 1, sourceWidth - 1);
    int y1 = min(y0 + 1, sourceHeight - 1);
    int z1 = min(z0 + 1, sourceDepth - 1);
    
    // 计算分数部分
    float fx = srcX - x0;
    float fy = srcY - y0;
    float fz = srcZ - z0;
    
    // 获取八个角点的值
    float v000 = sourceData[z0 * sourceHeight * sourceWidth + y0 * sourceWidth + x0];
    float v100 = sourceData[z0 * sourceHeight * sourceWidth + y0 * sourceWidth + x1];
    float v010 = sourceData[z0 * sourceHeight * sourceWidth + y1 * sourceWidth + x0];
    float v110 = sourceData[z0 * sourceHeight * sourceWidth + y1 * sourceWidth + x1];
    float v001 = sourceData[z1 * sourceHeight * sourceWidth + y0 * sourceWidth + x0];
    float v101 = sourceData[z1 * sourceHeight * sourceWidth + y0 * sourceWidth + x1];
    float v011 = sourceData[z1 * sourceHeight * sourceWidth + y1 * sourceWidth + x0];
    float v111 = sourceData[z1 * sourceHeight * sourceWidth + y1 * sourceWidth + x1];
    
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
    float v00 = v000 * (1.0f - fx) + v100 * fx;
    float v10 = v010 * (1.0f - fx) + v110 * fx;
    float v01 = v001 * (1.0f - fx) + v101 * fx;
    float v11 = v011 * (1.0f - fx) + v111 * fx;
    
    float v0 = v00 * (1.0f - fy) + v10 * fy;
    float v1 = v01 * (1.0f - fy) + v11 * fy;
    
    float result = v0 * (1.0f - fz) + v1 * fz;
    
    outputData[outZ * outputHeight * outputWidth + outY * outputWidth + outX] = result;
}

// PCHIP斜率计算
float pchipSlope(float h1, float h2, float m1, float m2) {
    if (m1 * m2 <= 0.0f) {
        return 0.0f;
    }
    
    float wh1 = 2.0f * h1 + h2;
    float wh2 = h1 + 2.0f * h2;
    return (wh1 + wh2) / (wh1 / m1 + wh2 / m2);
}

// Hermite多项式计算
float evaluateHermitePolynomial(float t, float y0, float y1, float m0, float m1, float h) {
    float t2 = t * t;
    float t3 = t2 * t;
    
    float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 = t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 = t3 - t2;
    
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
}

// PCHIP 2D插值核函数
__kernel void pchip2DInterpolation(
    __global const float* sourceData,
    __global const float* derivX,
    __global const float* derivY,
    __global const float* derivXY,
    __global float* outputData,
    const int sourceWidth,
    const int sourceHeight,
    const int outputWidth,
    const int outputHeight,
    const float minX,
    const float maxX,
    const float minY,
    const float maxY,
    const float fillValue) {
    
    int outX = get_global_id(0);
    int outY = get_global_id(1);
    
    if (outX >= outputWidth || outY >= outputHeight) return;
    
    // 计算输出坐标对应的源坐标
    float x = minX + (maxX - minX) * outX / (outputWidth - 1);
    float y = minY + (maxY - minY) * outY / (outputHeight - 1);
    
    // 计算网格间距
    float dx = (maxX - minX) / (sourceWidth - 1);
    float dy = (maxY - minY) / (sourceHeight - 1);
    
    // 转换到源数据的像素坐标
    float srcX = (x - minX) / dx;
    float srcY = (y - minY) / dy;
    
    // 边界检查
    if (srcX < 0 || srcX > sourceWidth - 1 || srcY < 0 || srcY > sourceHeight - 1) {
        outputData[outY * outputWidth + outX] = fillValue;
        return;
    }
    
    // 计算整数坐标
    int x0 = (int)floor(srcX);
    int y0 = (int)floor(srcY);
    int x1 = min(x0 + 1, sourceWidth - 1);
    int y1 = min(y0 + 1, sourceHeight - 1);
    
    // 计算归一化的分数部分
    float tx = srcX - x0;
    float ty = srcY - y0;
    
    // 获取四个角点的值和导数
    int idx00 = y0 * sourceWidth + x0;
    int idx10 = y0 * sourceWidth + x1;
    int idx01 = y1 * sourceWidth + x0;
    int idx11 = y1 * sourceWidth + x1;
    
    float v00 = sourceData[idx00];
    float v10 = sourceData[idx10];
    float v01 = sourceData[idx01];
    float v11 = sourceData[idx11];
    
    // 处理NaN值
    if (isnan(v00)) v00 = fillValue;
    if (isnan(v10)) v10 = fillValue;
    if (isnan(v01)) v01 = fillValue;
    if (isnan(v11)) v11 = fillValue;
    
    // 获取导数
    float fx00 = derivX[idx00] * dx;
    float fx10 = derivX[idx10] * dx;
    float fx01 = derivX[idx01] * dx;
    float fx11 = derivX[idx11] * dx;
    
    float fy00 = derivY[idx00] * dy;
    float fy10 = derivY[idx10] * dy;
    float fy01 = derivY[idx01] * dy;
    float fy11 = derivY[idx11] * dy;
    
    float fxy00 = derivXY[idx00] * dx * dy;
    float fxy10 = derivXY[idx10] * dx * dy;
    float fxy01 = derivXY[idx01] * dx * dy;
    float fxy11 = derivXY[idx11] * dx * dy;
    
    // 1D PCHIP在X方向
    float v0 = evaluateHermitePolynomial(tx, v00, v10, fx00, fx10, 1.0f);
    float v1 = evaluateHermitePolynomial(tx, v01, v11, fx01, fx11, 1.0f);
    
    // 1D PCHIP对Y方向的导数
    float m0y = evaluateHermitePolynomial(tx, fy00, fy10, fxy00, fxy10, 1.0f);
    float m1y = evaluateHermitePolynomial(tx, fy01, fy11, fxy01, fxy11, 1.0f);
    
    // 最终的1D PCHIP在Y方向
    float result = evaluateHermitePolynomial(ty, v0, v1, m0y, m1y, 1.0f);
    
    outputData[outY * outputWidth + outX] = result;
} 