/**
 * @file gpu_tile_types.h
 * @brief GPU瓦片生成相关数据结构定义
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace oscean::output_generation::gpu {

/**
 * @brief 瓦片请求结构
 */
struct TileRequest {
    int tileX;      // 瓦片X坐标
    int tileY;      // 瓦片Y坐标
    int zoomLevel;  // 缩放级别
};

/**
 * @brief GPU瓦片生成参数
 */
struct GPUTileParams {
    int tileSize = 256;           // 瓦片大小（像素）
    std::string colormap = "viridis"; // 颜色映射方案
    bool autoScale = true;        // 自动缩放数据范围
    double minValue = 0.0;        // 手动设置最小值
    double maxValue = 1.0;        // 手动设置最大值
    bool antialiasing = false;    // 抗锯齿
    int sampleCount = 1;          // 抗锯齿采样数
};

/**
 * @brief GPU瓦片生成结果
 */
struct GPUTileResult {
    bool success = false;
    int tileX = 0;
    int tileY = 0;
    int zoomLevel = 0;
    int width = 0;
    int height = 0;
    std::string format = "RGBA";
    std::vector<uint8_t> tileData;
    
    // 性能统计
    struct {
        double gpuTime = 0.0;      // GPU计算时间(ms)
        double transferTime = 0.0;  // 数据传输时间(ms)
        double totalTime = 0.0;     // 总时间(ms)
        size_t memoryUsed = 0;      // 内存使用(bytes)
        double throughput = 0.0;    // 吞吐量(GB/s)
    } stats;
};

} // namespace oscean::output_generation::gpu 