#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();

#include "core_services/common_data_types.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/time/time_resolution.h"
#include "common_utils/utilities/logging_utils.h"
#include <boost/thread/future.hpp>
#include <boost/optional.hpp>
#include <memory>
#include <string>
#include <vector>

namespace oscean::core_services::metadata::unified {

/**
 * @brief 元数据验证结果
 */
struct ValidationResult {
    bool isValid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::vector<std::string> suggestions;
    double qualityScore = 1.0;
    
    /**
     * @brief 合并多个验证结果
     */
    void mergeResults(const std::vector<ValidationResult>& results);
};

/**
 * @brief 元数据增强选项
 */
struct MetadataEnhancementOptions {
    bool validateSpatialInfo = true;    ///< 验证空间信息
    bool validateTemporalInfo = true;   ///< 验证时间信息
    bool validateCRS = true;            ///< 验证坐标系
    bool calculateQualityMetrics = true; ///< 计算质量指标
    bool standardizeFormats = true;     ///< 标准化格式
    bool extractAdditionalInfo = false; ///< 提取额外信息
    double qualityThreshold = 0.8;      ///< 质量阈值
};

/**
 * @brief 统一元数据适配器 - 核心协调组件（C++17版本）
 * 🎯 职责：统一管理所有元数据转换、验证和增强逻辑
 * 
 * 直接使用现有专业服务：
 * - common_utils::time 用于时间处理
 * - ICrsService 用于CRS验证和转换
 * - ISpatialOpsService 用于空间计算和验证
 * - metadata::util::SpatialResolutionExtractor 用于分辨率计算
 */
class UnifiedMetadataAdapter {
public:
    /**
     * @brief 构造函数 - 注入必要的依赖服务
     */
    UnifiedMetadataAdapter(
        std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices,
        std::shared_ptr<oscean::core_services::ICrsService> crsService,
        std::shared_ptr<oscean::core_services::spatial_ops::ISpatialOpsService> spatialService
    );

    /**
     * @brief 析构函数
     */
    ~UnifiedMetadataAdapter();

    /**
     * @brief 统一元数据验证和标准化（C++17: 使用boost::future）
     * @param rawMetadata 原始元数据
     * @return 验证结果
     */
    boost::future<ValidationResult> validateAndStandardizeAsync(
        const core_services::FileMetadata& rawMetadata
    );

    /**
     * @brief 统一元数据增强处理（C++17: 使用boost::future）
     * @param basicMetadata 基础元数据
     * @param options 增强选项
     * @return 增强后的元数据
     */
    boost::future<core_services::FileMetadata> enhanceMetadataAsync(
        const core_services::FileMetadata& basicMetadata,
        const MetadataEnhancementOptions& options
    );

    /**
     * @brief 批量元数据处理（C++17: 使用boost::future）
     * @param metadataList 元数据列表
     * @param options 增强选项
     * @return 处理后的元数据列表
     */
    boost::future<std::vector<core_services::FileMetadata>> processBatchMetadataAsync(
        const std::vector<core_services::FileMetadata>& metadataList,
        const MetadataEnhancementOptions& options
    );

    /**
     * @brief 计算元数据质量评分
     * @param metadata 元数据
     * @return 质量评分 (0-1)
     */
    double calculateQualityScore(const core_services::FileMetadata& metadata);

private:
    std::shared_ptr<common_utils::infrastructure::CommonServicesFactory> commonServices_;
    std::shared_ptr<oscean::core_services::ICrsService> crsService_;
    std::shared_ptr<oscean::core_services::spatial_ops::ISpatialOpsService> spatialService_;

    // === 内部处理方法 ===

    /**
     * @brief 验证基础元数据完整性
     */
    ValidationResult validateBasicMetadata(const core_services::FileMetadata& metadata);

    /**
     * @brief 验证时间信息
     */
    boost::future<ValidationResult> validateTemporalInfoAsync(
        const core_services::FileMetadata& metadata
    );

    /**
     * @brief 验证空间信息 - 使用CRS和空间服务
     */
    boost::future<ValidationResult> validateSpatialInfoAsync(
        const core_services::FileMetadata& metadata
    );

    /**
     * @brief 增强时间信息 - 使用common_utils::time
     */
    void enhanceTemporalInfo(
        core_services::FileMetadata& metadata,
        const MetadataEnhancementOptions& options
    );

    /**
     * @brief 增强空间信息 - 使用CRS和空间服务
     */
    boost::future<void> enhanceSpatialInfoAsync(
        core_services::FileMetadata& metadata,
        const MetadataEnhancementOptions& options
    );

    /**
     * @brief 计算并设置分辨率信息 - 使用现有分辨率提取器
     */
    void calculateAndSetResolution(core_services::FileMetadata& metadata);

    /**
     * @brief 标准化变量信息
     */
    void standardizeVariableInfo(core_services::FileMetadata& metadata);

    /**
     * @brief 验证数据类型一致性
     */
    ValidationResult validateDataTypeConsistency(const core_services::FileMetadata& metadata);

    /**
     * @brief 验证边界框有效性 - 使用空间服务
     */
    boost::future<bool> validateBoundingBoxAsync(
        const core_services::BoundingBox& bbox,
        const std::string& crsId
    );

    /**
     * @brief 标准化CRS信息 - 使用CRS服务
     */
    boost::future<core_services::CRSInfo> standardizeCRSAsync(
        const core_services::CRSInfo& rawCRS
    );
};

} // namespace oscean::core_services::metadata::unified 