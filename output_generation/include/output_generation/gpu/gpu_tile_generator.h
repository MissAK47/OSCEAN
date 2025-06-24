/**
 * @file gpu_tile_generator.h
 * @brief GPU瓦片生成器工厂函数
 */

#pragma once

#include <memory>

namespace oscean::output_generation::gpu {

// 前向声明
class IGPUTileGenerator;

/**
 * @brief 创建GPU瓦片生成器
 * @param deviceId GPU设备ID
 */
std::unique_ptr<IGPUTileGenerator> createGPUTileGenerator(int deviceId = 0);

} // namespace oscean::output_generation::gpu 