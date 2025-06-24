#pragma once

// 🚀 使用Common模块的统一boost配置
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // metadata服务不使用boost::asio，只使用boost::future

#include "core_services/metadata/unified_metadata_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "common_utils/utilities/logging_utils.h"
#include <boost/thread/future.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <set>
#include "common_utils/infrastructure/common_services_factory.h"
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/data_access/i_unified_data_access_service.h"
#include "core_services/crs/i_crs_service.h"

// Forward declare spdlog logger
namespace spdlog {
    class logger;
}

namespace oscean::core_services::metadata::impl {

// 变量分类规则定义
struct VariableClassificationRule {
    std::string standardName;
    std::string longName;
    std::string variableName;
    std::string units;
    DataType dataType;
    double weight = 1.0;
};

// 用于存储分类规则
struct ClassificationRules {
    std::vector<VariableClassificationRule> rules;
    std::map<DataType, std::vector<std::string>> categoryKeywords;
    std::map<std::string, std::string> variableNameMapping;
    std::map<std::string, std::vector<std::string>> variableClassificationRules;
    bool fuzzyMatchingEnabled = false;
    double fuzzyMatchingThreshold = 0.8;
};

// ✅ 新增：用于加载 file_format_mapping.yaml 的规则
struct FileFormatMappingRules {
    // 格式优先规则
    std::map<std::string, std::string> formatToDatabase; // e.g., ".tif" -> "topography_bathymetry"
    // 需要内容分析的格式
    std::set<std::string> contentAnalysisFormats; // e.g., ".nc"
    // 变量到数据库的映射
    std::map<std::string, DatabaseType> variableToDatabase; // e.g., "temperature" -> DatabaseType::OCEAN_ENVIRONMENT
    // 变量分类到数据库的映射
    std::map<std::string, std::string> variableCategoryToDatabase;
};

// ✅ 新增：统一的分类结果
struct ClassificationResult {
    DataType primaryCategory = DataType::UNKNOWN;              // ✅ 第一层分类: 主要归属
    std::vector<DataType> detailedDataTypes;                // ✅ 第二层分类: 所有详细类型
    std::map<DataType, double> confidenceScores;             // ✅ 每种类型的置信度
    std::string reason;                                     // 分类原因
    
    // --- 🔧 修复编译错误：新增缺失的tags字段 ---
    std::vector<std::string> tags;                          // 分类标签（向后兼容）
};

/**
 * @brief 智能识别器 - 增强版
 * @note 负责根据文件格式和变量内容，将文件分类到合适的数据库。
 */
class IntelligentRecognizer {
public:
    /**
     * @brief 构造函数
     * @param logger 日志服务实例
     * @param dataAccessService 数据访问服务实例
     * @param crsService CRS服务实例
     * @param loadClassificationRules 是否加载分类规则
     */
    IntelligentRecognizer(
        std::shared_ptr<oscean::common_utils::infrastructure::logging::ILogger> logger,
        std::shared_ptr<core_services::data_access::IUnifiedDataAccessService> dataAccessService,
        std::shared_ptr<core_services::ICrsService> crsService,
        bool loadClassificationRules = true
    );

    /**
     * @brief 对文件元数据进行分类和丰富化处理
     * @param metadata 从读取器获取的，包含原始属性的元数据对象
     * @return 包含分类结果和原因的ClassificationResult对象
     */
    ClassificationResult classifyFile(const core_services::FileMetadata& metadata) const;

    /**
     * @brief 对单个变量进行分类
     * @param variableName 变量名
     * @return 变量类型字符串
     */
    std::string classifyVariable(const std::string& variableName) const;
    
    /**
     * @brief 批量变量分类
     * @param variableNames 变量名列表
     * @return 变量类型列表
     */
    std::vector<std::string> classifyVariables(const std::vector<std::string>& variableNames) const;

    /**
     * @brief 更新变量分类配置
     */
    void updateClassificationConfig(const VariableClassificationConfig& config);

    /**
     * @brief 从变量确定数据类型
     */
    std::vector<DataType> determineDataTypeFromVariables(const std::vector<oscean::core_services::VariableMeta>& variables) const;
    
    /**
     * @brief 获取标准化变量名
     */
    std::string getNormalizedVariableName(const std::string& originalName) const;
    
    /**
     * @brief 判断变量是否应该包含在指定数据类型中
     */
    bool shouldIncludeVariableForDataType(const std::string& varType, DataType dataType) const;

    /**
     * @brief 延迟加载YAML配置文件（用于服务初始化后调用）
     */
    void loadConfigurationFiles();

private:
    void loadDefaultClassificationRules();
    void loadFileFormatRules(const std::string& path);
    void loadVariableClassificationRules(const std::string& path);
    DataType determinePrimaryCategory(const std::map<DataType, double>& confidenceScores) const;
    std::map<DataType, double> determineDetailedDataTypes(
        const std::vector<core_services::VariableMeta>& variables) const;
    
    /**
     * @brief 根据原始元数据填充结构化的空间信息 (BoundingBox, CRS等)
     * @param metadata 要被丰富的元数据对象 (输入/输出)
     */
    void enrichWithSpatialInfo(core_services::FileMetadata& metadata) const;

    /**
     * @brief 根据原始元数据填充结构化的时间信息 (TimeRange)
     * @param metadata 要被丰富的元数据对象 (输入/输出)
     */
    void enrichWithTemporalInfo(core_services::FileMetadata& metadata) const;

    std::shared_ptr<oscean::common_utils::infrastructure::logging::ILogger> m_logger;
    std::shared_ptr<core_services::data_access::IUnifiedDataAccessService> m_dataAccessService;
    std::shared_ptr<core_services::ICrsService> m_crsService;
    std::unique_ptr<YAML::Node> m_formatRules;
    std::unique_ptr<YAML::Node> m_variableRules;
    std::atomic<bool> m_rulesLoaded;
};

} // namespace oscean::core_services::metadata::impl 