/**
 * @file tile_generation.cl
 * @brief OpenCL瓦片生成内核实现
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 颜色查找表（从color_mapping.cl共享）
__constant float4 colorLUT[256];

/**
 * @brief 双线性插值
 */
float bilinearInterpolate(
    __global const float* data,
    float x, float y,
    int width, int height) {
    
    // 边界检查
    x = clamp(x, 0.0f, (float)(width - 1));
    y = clamp(y, 0.0f, (float)(height - 1));
    
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
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
void mercatorToLatLon(
    float x, float y, int z,
    float* lon, float* lat) {
    
    float n = pow(2.0f, (float)z);
    *lon = x / n * 360.0f - 180.0f;
    
    float latRad = atan(sinh(M_PI * (1.0f - 2.0f * y / n)));
    *lat = latRad * 180.0f / M_PI;
}

/**
 * @brief 经纬度到数据网格坐标转换
 */
void latLonToGrid(
    float lon, float lat,
    float minLon, float maxLon,
    float minLat, float maxLat,
    int gridWidth, int gridHeight,
    float* gridX, float* gridY) {
    
    *gridX = (lon - minLon) / (maxLon - minLon) * (gridWidth - 1);
    *gridY = (lat - minLat) / (maxLat - minLat) * (gridHeight - 1);
}

/**
 * @brief 基础瓦片生成内核
 */
__kernel void tileGenerationKernel(
    __global const float* gridData,
    __global uchar4* tileData,
    const int tileX,
    const int tileY,
    const int zoomLevel,
    const int tileSize,
    const int gridWidth,
    const int gridHeight,
    const float minLon,
    const float maxLon,
    const float minLat,
    const float maxLat,
    const float minValue,
    const float maxValue) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= tileSize || y >= tileSize) return;
    
    // 计算瓦片中像素对应的经纬度
    float pixelX = tileX + (float)x / tileSize;
    float pixelY = tileY + (float)y / tileSize;
    
    float lon, lat;
    mercatorToLatLon(pixelX, pixelY, zoomLevel, &lon, &lat);
    
    // 转换到网格坐标
    float gridX, gridY;
    latLonToGrid(lon, lat, minLon, maxLon, minLat, maxLat,
                 gridWidth, gridHeight, &gridX, &gridY);
    
    // 双线性插值获取数据值
    float value = bilinearInterpolate(gridData, gridX, gridY, gridWidth, gridHeight);
    
    // 处理NaN值
    if (isnan(value)) {
        int idx = y * tileSize + x;
        tileData[idx] = (uchar4)(0, 0, 0, 0);
        return;
    }
    
    // 归一化到[0, 1]
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
    
    // 映射到颜色
    float lutIndex = normalizedValue * 255.0f;
    int index0 = (int)floor(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    // 从LUT中获取颜色
    float4 color0 = colorLUT[index0];
    float4 color1 = colorLUT[index1];
    float4 color = mix(color0, color1, frac);
    
    // 写入输出
    int idx = y * tileSize + x;
    tileData[idx] = (uchar4)(
        (uchar)(color.x * 255.0f),
        (uchar)(color.y * 255.0f),
        (uchar)(color.z * 255.0f),
        (uchar)(color.w * 255.0f)
    );
}

/**
 * @brief 高级瓦片生成内核（支持抗锯齿）
 */
__kernel void tileGenerationAntialiasingKernel(
    __global const float* gridData,
    __global uchar4* tileData,
    const int tileX,
    const int tileY,
    const int zoomLevel,
    const int tileSize,
    const int gridWidth,
    const int gridHeight,
    const float minLon,
    const float maxLon,
    const float minLat,
    const float maxLat,
    const float minValue,
    const float maxValue,
    const int sampleCount) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= tileSize || y >= tileSize) return;
    
    // 超采样抗锯齿
    float4 accColor = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float sampleStep = 1.0f / sampleCount;
    int validSamples = 0;
    
    for (int sy = 0; sy < sampleCount; ++sy) {
        for (int sx = 0; sx < sampleCount; ++sx) {
            // 计算子像素位置
            float subX = x + (sx + 0.5f) * sampleStep;
            float subY = y + (sy + 0.5f) * sampleStep;
            
            float pixelX = tileX + subX / tileSize;
            float pixelY = tileY + subY / tileSize;
            
            float lon, lat;
            mercatorToLatLon(pixelX, pixelY, zoomLevel, &lon, &lat);
            
            float gridX, gridY;
            latLonToGrid(lon, lat, minLon, maxLon, minLat, maxLat,
                        gridWidth, gridHeight, &gridX, &gridY);
            
            float value = bilinearInterpolate(gridData, gridX, gridY, gridWidth, gridHeight);
            
            if (!isnan(value)) {
                float normalizedValue = (value - minValue) / (maxValue - minValue);
                normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
                
                float lutIndex = normalizedValue * 255.0f;
                int index0 = (int)floor(lutIndex);
                int index1 = min(index0 + 1, 255);
                float frac = lutIndex - index0;
                
                float4 color0 = colorLUT[index0];
                float4 color1 = colorLUT[index1];
                float4 color = mix(color0, color1, frac);
                
                accColor += color;
                validSamples++;
            }
        }
    }
    
    // 平均化
    if (validSamples > 0) {
        accColor /= (float)validSamples;
    }
    
    // 写入输出
    int idx = y * tileSize + x;
    tileData[idx] = (uchar4)(
        (uchar)(accColor.x * 255.0f),
        (uchar)(accColor.y * 255.0f),
        (uchar)(accColor.z * 255.0f),
        (uchar)(accColor.w * 255.0f)
    );
}

/**
 * @brief 批量瓦片生成内核
 */
__kernel void batchTileGenerationKernel(
    __global const float* gridData,
    __global uchar4* tileDataBatch,  // 所有瓦片数据连续存储
    __global const int2* tileCoords, // (tileX, tileY) pairs
    const int zoomLevel,
    const int tileSize,
    const int gridWidth,
    const int gridHeight,
    const float minLon,
    const float maxLon,
    const float minLat,
    const float maxLat,
    const float minValue,
    const float maxValue,
    const int batchSize) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int tileIdx = get_global_id(2);
    
    if (x >= tileSize || y >= tileSize || tileIdx >= batchSize) return;
    
    int2 tileCoord = tileCoords[tileIdx];
    int tileX = tileCoord.x;
    int tileY = tileCoord.y;
    
    // 计算瓦片中像素对应的经纬度
    float pixelX = tileX + (float)x / tileSize;
    float pixelY = tileY + (float)y / tileSize;
    
    float lon, lat;
    mercatorToLatLon(pixelX, pixelY, zoomLevel, &lon, &lat);
    
    // 转换到网格坐标
    float gridX, gridY;
    latLonToGrid(lon, lat, minLon, maxLon, minLat, maxLat,
                 gridWidth, gridHeight, &gridX, &gridY);
    
    // 双线性插值获取数据值
    float value = bilinearInterpolate(gridData, gridX, gridY, gridWidth, gridHeight);
    
    // 颜色映射
    uchar4 outputColor;
    if (isnan(value)) {
        outputColor = (uchar4)(0, 0, 0, 0);
    } else {
        float normalizedValue = (value - minValue) / (maxValue - minValue);
        normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
        
        float lutIndex = normalizedValue * 255.0f;
        int index0 = (int)floor(lutIndex);
        int index1 = min(index0 + 1, 255);
        float frac = lutIndex - index0;
        
        float4 color0 = colorLUT[index0];
        float4 color1 = colorLUT[index1];
        float4 color = mix(color0, color1, frac);
        
        outputColor = (uchar4)(
            (uchar)(color.x * 255.0f),
            (uchar)(color.y * 255.0f),
            (uchar)(color.z * 255.0f),
            (uchar)(color.w * 255.0f)
        );
    }
    
    // 写入批量输出缓冲区
    int tileOffset = tileIdx * tileSize * tileSize;
    int pixelIdx = y * tileSize + x;
    tileDataBatch[tileOffset + pixelIdx] = outputColor;
} 