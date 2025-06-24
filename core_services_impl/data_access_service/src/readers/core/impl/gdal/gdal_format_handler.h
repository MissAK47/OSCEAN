#pragma once

/**
 * @file gdal_format_handler.h
 * @brief GDAL格式处理器 - 简化版本
 */

#include <gdal_priv.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <boost/optional.hpp>
#include <boost/thread/future.hpp>
#include <boost/asio/thread_pool.hpp>

#include "readers/core/unified_data_reader.h"
#include "core_services/common_data_types.h"
#include "gdal_common_types.h"  // 包含完整的枚举定义

// 前向声明
class GDALDataset;

namespace oscean::common_utils::simd {
    class ISIMDManager;
}

namespace oscean::core_services::data_access::readers::impl::gdal {

// 简化的流式配置结构
struct StreamingConfig {
    size_t chunkSize = 1024 * 1024;    ///< 数据块大小
    size_t maxConcurrency = 4;         ///< 最大并发数
    bool enableOptimization = true;    ///< 启用优化
};

/**
 * @brief 数据块结构 - 简化版本
 */
struct DataChunk {
    std::vector<double> data;           ///< 数据内容
    std::vector<size_t> shape;          ///< 数据形状
    std::vector<size_t> offset;         ///< 在原始数据中的偏移
    size_t chunkId;                     ///< 块ID
    bool isLastChunk = false;           ///< 是否为最后一块
};

/**
 * @brief GDAL格式处理器 - 简化版本
 */
class GDALFormatHandler {
public:
    explicit GDALFormatHandler(GDALDataset* dataset);
    virtual ~GDALFormatHandler() = default;
    
    // =============================================================================
    // 基本接口
    // =============================================================================
    
    bool openFile(const std::string& filePath);
    std::vector<std::string> getVariableNames();
    std::shared_ptr<oscean::core_services::GridData> readVariable(const std::string& name);
    
    // 格式特定的优化提示
    bool shouldUseSIMD() const;
    size_t getOptimalChunkSize() const;
    
    // =============================================================================
    // GDAL特定接口
    // =============================================================================
    
    /**
     * @brief 获取GDAL数据集
     */
    GDALDataset* getDataset() const { return dataset_; }
    
    /**
     * @brief 获取变量详细信息
     */
    boost::optional<oscean::core_services::VariableMeta> getVariableInfo(const std::string& variableName) const;
    
    /**
     * @brief 获取数据类型
     */
    GdalDataType getDataType() const { return dataType_; }
    
    /**
     * @brief 检查是否为栅格数据
     */
    bool isRasterData() const { return dataType_ == oscean::core_services::data_access::readers::impl::gdal::GdalDataType::RASTER; }
    
    /**
     * @brief 检查是否为矢量数据
     */
    bool isVectorData() const { return dataType_ == oscean::core_services::data_access::readers::impl::gdal::GdalDataType::VECTOR; }
    
    /**
     * @brief 读取栅格数据
     */
    std::shared_ptr<oscean::core_services::GridData> readRasterData(
        const std::string& variableName,
        const boost::optional<oscean::core_services::BoundingBox>& bounds = boost::none) const;
    
    /**
     * @brief 获取CRS信息
     */
    boost::optional<oscean::core_services::CRSInfo> getCRSInfo() const;
    
    /**
     * @brief 获取边界框
     */
    oscean::core_services::BoundingBox getBoundingBox() const;
    
    /**
     * @brief 获取变量属性
     */
    std::vector<oscean::core_services::MetadataEntry> getVariableAttributes(const std::string& variableName) const;
    
    /**
     * @brief 流式读取变量数据
     */
    boost::future<void> streamVariableData(
        const std::string& variableName,
        const boost::optional<oscean::core_services::BoundingBox>& bounds,
        std::function<bool(const std::vector<double>&, const std::vector<size_t>&)> processor
    );
    
private:
    // =============================================================================
    // 私有成员变量
    // =============================================================================
    
    GDALDataset* dataset_;                                          ///< GDAL数据集
    GdalDataType dataType_;                                         ///< 数据类型
    
    // 🔧 简化缓存，移除过时的变量信息缓存
    mutable boost::optional<std::vector<std::string>> cachedVariableNames_;
    mutable boost::optional<oscean::core_services::CRSInfo> cachedCRSInfo_;
    mutable boost::optional<oscean::core_services::BoundingBox> cachedBoundingBox_;
    
    // =============================================================================
    // 私有方法
    // =============================================================================
    
    /**
     * @brief 检测数据类型
     */
    GdalDataType detectDataType() const;
    
    /**
     * @brief 提取变量信息 - 统一使用VariableMeta
     */
    oscean::core_services::VariableMeta extractVariableInfo(const std::string& variableName) const;
    
    /**
     * @brief 验证数据集
     */
    bool validateDataset() const;
};

/**
 * @brief GDAL流式适配器 - 增强版本，支持矢量流式处理和背压控制
 */
class GDALStreamingAdapter {
public:
    GDALStreamingAdapter(GDALDataset* dataset, const std::string& variableName);
    virtual ~GDALStreamingAdapter() = default;
    
    // =============================================================================
    // 基本接口
    // =============================================================================
    
    bool hasMoreChunks() const;
    boost::optional<DataChunk> getNextChunk();
    void reset();
    void configureChunking(const StreamingConfig& config);
    
    // =============================================================================
    // 🆕 背压控制接口
    // =============================================================================
    
    /**
     * @brief 检查是否应该应用背压
     */
    bool shouldApplyBackpressure() const;
    
    /**
     * @brief 等待背压缓解
     */
    boost::future<bool> waitForBackpressureRelief();
    
    /**
     * @brief 通知处理完成，释放背压
     */
    void notifyChunkProcessed();
    
    /**
     * @brief 获取当前内存使用统计
     */
    size_t getCurrentMemoryUsage() const;
    
    // =============================================================================
    // 🆕 SIMD优化接口
    // =============================================================================
    
    /**
     * @brief 设置SIMD管理器
     */
    void setSIMDManager(std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager);
    
    /**
     * @brief 检查是否应该使用SIMD优化
     */
    bool shouldUseSIMDProcessing(size_t dataSize) const;
    
    /**
     * @brief 对数据块应用SIMD优化处理
     */
    void applySIMDOptimizations(DataChunk& chunk) const;
    
    // =============================================================================
    // 🆕 并发处理接口
    // =============================================================================
    
    /**
     * @brief 设置线程池
     */
    void setThreadPool(std::shared_ptr<boost::asio::thread_pool> threadPool);
    
    /**
     * @brief 并行读取多个数据块
     */
    boost::future<std::vector<DataChunk>> readMultipleChunksAsync(size_t numChunks);
    
    /**
     * @brief 配置并发处理参数
     */
    void configureConcurrency(size_t maxConcurrentReads, bool enableParallelProcessing = true);
    
    // =============================================================================
    // GDAL特定接口
    // =============================================================================
    
    /**
     * @brief 配置栅格流式读取
     */
    void configureRasterStreaming(int bandNumber, int tileXSize, int tileYSize);
    
    /**
     * @brief 配置矢量流式读取
     */
    void configureVectorStreaming(const std::string& layerName, size_t featuresPerChunk = 1000);
    
private:
    GDALDataset* dataset_;
    std::string variableName_;
    GdalDataType dataType_;
    
    // 栅格流式状态
    int bandNumber_ = -1;
    int tileXSize_ = 512;
    int tileYSize_ = 512;
    int currentTileX_ = 0;
    int currentTileY_ = 0;
    int tilesX_ = 0;
    int tilesY_ = 0;
    
    // 矢量流式状态
    std::string layerName_;
    size_t featuresPerChunk_ = 1000;    ///< 每块的要素数量
    size_t currentFeatureIndex_ = 0;    ///< 当前要素索引
    size_t totalFeatures_ = 0;          ///< 总要素数量
    size_t currentChunkId_ = 0;         ///< 当前块ID
    class OGRLayer* currentLayer_ = nullptr;  ///< 当前图层指针
    
    // 🆕 背压控制状态
    mutable std::atomic<size_t> activeChunks_{0};      ///< 当前活跃块数量
    mutable std::atomic<size_t> totalMemoryUsed_{0};   ///< 当前内存使用量
    mutable std::mutex backpressureMutex_;             ///< 背压控制互斥锁
    mutable std::condition_variable backpressureCondition_; ///< 背压条件变量
    
    // 🆕 SIMD优化状态
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_; ///< SIMD管理器
    bool enableSIMDOptimizations_ = true;               ///< 是否启用SIMD优化
    size_t simdThreshold_ = 1000;                      ///< SIMD处理的最小数据量阈值
    
    // 通用状态
    StreamingConfig config_;
    bool initialized_ = false;
    
    // 🆕 并发处理状态
    std::shared_ptr<boost::asio::thread_pool> threadPool_; ///< 线程池
    size_t maxConcurrentReads_ = 4;                        ///< 最大并发读取数
    bool enableParallelProcessing_ = true;                   ///< 是否启用并行处理
    
    /**
     * @brief 初始化流式状态
     */
    void initialize();
    
    /**
     * @brief 计算栅格瓦片参数
     */
    void calculateRasterTiling();
    
    /**
     * @brief 初始化矢量流式参数
     */
    void initializeVectorStreaming();
    
    /**
     * @brief 读取栅格瓦片
     */
    boost::optional<DataChunk> readRasterTile();
    
    /**
     * @brief 读取矢量数据块
     */
    boost::optional<DataChunk> readVectorChunk();
    
    /**
     * @brief 🆕 检查内存阈值
     */
    bool checkMemoryThreshold() const;
    
    /**
     * @brief 🆕 更新内存使用统计
     */
    void updateMemoryUsage(size_t chunkSize, bool isAdd);
};

} // namespace oscean::core_services::data_access::readers::impl::gdal 