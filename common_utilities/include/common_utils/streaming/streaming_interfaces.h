/**
 * @file streaming_interfaces.h
 * @brief 流式处理基础接口 - 支持LargeFileProcessor
 * @author OSCEAN Team
 * @date 2024
 * 
 * 🎯 整合说明：
 * ✅ 此文件保留了streaming模块中最基础的类型定义
 * ✅ 主要为infrastructure/large_file_processor提供支持
 * ✅ 大部分streaming功能已迁移到LargeFileProcessor
 */

#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <string>

namespace oscean::common_utils::streaming {

/**
 * @brief 数据块结构 - 基础流处理单元
 */
struct DataChunk {
    std::vector<uint8_t> data;
    size_t offset = 0;
    size_t size = 0;
    bool isLast = false;
    
    DataChunk() = default;
    DataChunk(size_t chunkSize) : data(chunkSize) {}
    DataChunk(const uint8_t* ptr, size_t len) : data(ptr, ptr + len), size(len) {}
};

/**
 * @brief 进度回调函数类型
 */
using ProgressCallback = std::function<void(double progress, const std::string& message)>;

/**
 * @brief 数据处理回调函数类型
 */
using DataProcessor = std::function<bool(const DataChunk& chunk)>;

/**
 * @brief 基础大数据处理器接口
 * 
 * 注意：此接口的具体实现已迁移到infrastructure::LargeFileProcessor
 */
class ILargeDataProcessor {
public:
    virtual ~ILargeDataProcessor() = default;
    
    /**
     * @brief 处理大文件
     * @param filePath 文件路径
     * @param processor 数据处理回调
     * @param progress 进度回调（可选）
     */
    virtual void processFile(
        const std::string& filePath,
        DataProcessor processor,
        ProgressCallback progress = nullptr) = 0;
};

} // namespace oscean::common_utils::streaming 