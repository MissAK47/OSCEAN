#pragma once

#include "core_services/common_data_types.h"
#include <string>
#include <optional>

namespace oscean::core_services::metadata::impl {

/**
 * @brief CRS处理器 - 负责处理和优化CRS信息
 * 
 * 职责：
 * 1. 清理和修复PROJ字符串
 * 2. 映射到标准EPSG代码
 * 3. 验证CRS信息完整性
 * 4. 优化CRS表示
 */
class CRSProcessor {
public:
    CRSProcessor() = default;
    ~CRSProcessor() = default;

    /**
     * @brief 处理原始CRS信息，进行清理和优化
     * @param rawCRS 从文件读取器获得的原始CRS信息
     * @return 处理后的优化CRS信息
     */
    oscean::core_services::CRSInfo processCRSInfo(const oscean::core_services::CRSInfo& rawCRS) const;

    /**
     * @brief 清理NetCDF文件中的PROJ字符串
     * @param projString 原始PROJ字符串
     * @return 清理后的PROJ字符串
     */
    std::string cleanNetCDFProjString(const std::string& projString) const;

    /**
     * @brief 尝试将PROJ字符串映射到标准EPSG代码
     * @param projString PROJ字符串
     * @return 如果找到匹配的EPSG代码则返回，否则返回空字符串
     */
    std::string tryMapToEPSG(const std::string& projString) const;

    /**
     * @brief 验证CRS信息的完整性
     * @param crsInfo CRS信息
     * @return 验证结果和建议
     */
    struct ValidationResult {
        bool isValid = false;
        std::vector<std::string> warnings;
        std::vector<std::string> suggestions;
    };
    ValidationResult validateCRSInfo(const oscean::core_services::CRSInfo& crsInfo) const;

private:
    /**
     * @brief 检测并修复PROJ字符串中的参数冲突
     */
    std::string fixParameterConflicts(const std::string& projString) const;

    /**
     * @brief 为自定义球体参数添加必要的标识
     */
    std::string enhanceCustomSphere(const std::string& projString) const;

    /**
     * @brief 标准化PROJ字符串格式
     */
    std::string normalizeProjString(const std::string& projString) const;
};

} // namespace oscean::core_services::metadata::impl 
 