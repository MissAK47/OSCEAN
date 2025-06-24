/**
 * @file gpu_tile_generator_wrapper.h
 * @brief GPU瓦片生成器包装类，提供简化的测试接口
 */

#pragma once

#include "output_generation/gpu/gpu_tile_types.h"
#include "core_services/common_data_types.h"
#include <memory>
#include <vector>

namespace oscean::output_generation::gpu {

/**
 * @brief GPU瓦片生成器包装类
 * 提供简化的接口用于测试，隐藏IGPUTileGenerator的复杂性
 */
class GPUTileGeneratorWrapper {
public:
    GPUTileGeneratorWrapper(int deviceId = 0);
    ~GPUTileGeneratorWrapper();
    
    /**
     * @brief 生成单个瓦片
     * @param gridData 网格数据
     * @param request 瓦片请求
     * @param params 生成参数
     * @return 瓦片生成结果
     */
    GPUTileResult generateTile(
        const std::shared_ptr<core_services::GridData>& gridData,
        const TileRequest& request,
        const GPUTileParams& params);
    
    /**
     * @brief 批量生成瓦片
     * @param gridData 网格数据
     * @param requests 瓦片请求列表
     * @param params 生成参数
     * @return 瓦片生成结果列表
     */
    std::vector<GPUTileResult> generateTileBatch(
        const std::shared_ptr<core_services::GridData>& gridData,
        const std::vector<TileRequest>& requests,
        const GPUTileParams& params);
        
private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief 创建瓦片生成器包装实例
 */
std::unique_ptr<GPUTileGeneratorWrapper> createGPUTileGeneratorWrapper(int deviceId = 0);

} // namespace oscean::output_generation::gpu 