/**
 * @file gpu_grid_data_helper.h
 * @brief GridData GPU处理辅助工具
 */

#pragma once

#include "core_services/common_data_types.h"
#include <memory>
#include <cmath>

namespace oscean::output_generation::gpu {

using namespace oscean::core_services;

/**
 * @brief GridData GPU辅助工具类
 * 提供创建和操作GridData的便捷方法，专门优化用于GPU处理
 */
class GPUGridDataHelper {
public:
    /**
     * @brief 创建测试用的GridData
     * @param width 宽度
     * @param height 高度
     * @param extent 空间范围（默认全球范围）
     * @return GridData智能指针
     */
    static std::shared_ptr<GridData> createTestGridData(
        size_t width, size_t height,
        const BoundingBox& extent = BoundingBox(-180.0, -90.0, 180.0, 90.0)) {
        
        // 使用简单构造函数
        auto gridData = std::make_shared<GridData>(width, height, 1, DataType::Float32);
        
        // 设置空间范围（通过修改definition）
        gridData->definition.extent = extent;
        gridData->definition.xResolution = (extent.maxX - extent.minX) / width;
        gridData->definition.yResolution = (extent.maxY - extent.minY) / height;
        
        // 设置CRS
        gridData->definition.crs.id = "EPSG:4326";
        gridData->definition.crs.name = "WGS 84";
        gridData->definition.crs.isGeographic = true;
        
        return gridData;
    }
    
    /**
     * @brief 生成测试数据模式
     * @param gridData GridData对象
     * @param pattern 模式类型："gradient", "sinwave", "ocean_temp"
     * @param minValue 最小值
     * @param maxValue 最大值
     */
    static void generateTestPattern(
        std::shared_ptr<GridData>& gridData,
        const std::string& pattern = "gradient",
        float minValue = 0.0f,
        float maxValue = 1.0f) {
        
        size_t width = gridData->getWidth();
        size_t height = gridData->getHeight();
        float* data = reinterpret_cast<float*>(gridData->getUnifiedBufferData());
        
        if (pattern == "gradient") {
            // 简单梯度
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    float value = static_cast<float>(x + y) / (width + height);
                    data[y * width + x] = minValue + value * (maxValue - minValue);
                }
            }
        } else if (pattern == "sinwave") {
            // 正弦波模式
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    float fx = static_cast<float>(x) / width * 2.0f * 3.14159f;
                    float fy = static_cast<float>(y) / height * 2.0f * 3.14159f;
                    float value = (std::sin(fx) + std::cos(fy)) * 0.5f + 0.5f;
                    data[y * width + x] = minValue + value * (maxValue - minValue);
                }
            }
        } else if (pattern == "ocean_temp") {
            // 模拟海洋温度场
            const auto& extent = gridData->getDefinition().extent;
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    // 计算地理坐标
                    float lon = extent.minX + (extent.maxX - extent.minX) * x / width;
                    float lat = extent.minY + (extent.maxY - extent.minY) * y / height;
                    
                    // 简单的温度模型：赤道热，两极冷
                    float latFactor = std::cos(lat * 3.14159f / 180.0f);
                    float temp = 5.0f + 25.0f * latFactor; // 5°C到30°C
                    
                    // 添加一些噪声
                    float noise = (std::sin(lon * 0.1f) * std::cos(lat * 0.1f)) * 2.0f;
                    temp += noise;
                    
                    data[y * width + x] = temp;
                }
            }
        }
    }
    
    /**
     * @brief 获取GridData的Float32数据指针（用于CUDA）
     * @param gridData GridData对象
     * @return Float32数据指针
     */
    static const float* getFloat32DataPtr(const std::shared_ptr<GridData>& gridData) {
        if (gridData->getDataType() != DataType::Float32) {
            throw std::runtime_error("GridData is not Float32 type");
        }
        return reinterpret_cast<const float*>(gridData->getUnifiedBufferData());
    }
    
    /**
     * @brief 获取可修改的Float32数据指针
     * @param gridData GridData对象
     * @return 可修改的Float32数据指针
     */
    static float* getMutableFloat32DataPtr(std::shared_ptr<GridData>& gridData) {
        if (gridData->getDataType() != DataType::Float32) {
            throw std::runtime_error("GridData is not Float32 type");
        }
        return reinterpret_cast<float*>(gridData->getUnifiedBufferData());
    }
    
    /**
     * @brief 检查GridData是否适合GPU处理
     * @param gridData GridData对象
     * @return 如果适合返回true
     */
    static bool isGPUCompatible(const std::shared_ptr<GridData>& gridData) {
        // 检查数据类型
        DataType dtype = gridData->getDataType();
        if (dtype != DataType::Float32 && dtype != DataType::Float64 &&
            dtype != DataType::Int32 && dtype != DataType::UInt16) {
            return false;
        }
        
        // 检查内存对齐（如果启用了高性能模式）
        if (!gridData->isMemoryAligned()) {
            // 内存未对齐，但仍可使用，只是性能可能降低
        }
        
        // 检查数据大小
        size_t totalSize = gridData->getUnifiedBufferSize();
        if (totalSize == 0) {
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief 计算数据统计信息
     * @param gridData GridData对象
     * @param minValue 输出最小值
     * @param maxValue 输出最大值
     */
    static void computeMinMax(const std::shared_ptr<GridData>& gridData,
                            float& minValue, float& maxValue) {
        if (gridData->getDataType() != DataType::Float32) {
            throw std::runtime_error("Only Float32 supported for now");
        }
        
        const float* data = getFloat32DataPtr(gridData);
        size_t count = gridData->getWidth() * gridData->getHeight() * gridData->getBandCount();
        
        if (count == 0) {
            minValue = 0.0f;
            maxValue = 0.0f;
            return;
        }
        
        minValue = maxValue = data[0];
        for (size_t i = 1; i < count; ++i) {
            if (data[i] < minValue) minValue = data[i];
            if (data[i] > maxValue) maxValue = data[i];
        }
    }
};

} // namespace oscean::output_generation::gpu 