#pragma once

/**
 * @file memory_layout_analyzer.h
 * @brief NetCDF内存布局分析器
 */

#include <string>
#include <vector>
#include <memory>
#include "core_services/common_data_types.h"
#include "core_services/data_access/unified_data_types.h"

namespace oscean::core_services::data_access::readers::impl::netcdf {

/**
 * @brief 内存布局分析器
 * 
 * 根据NetCDF变量的元数据和使用场景，分析最优的内存布局
 */
class MemoryLayoutAnalyzer {
public:
    /**
     * @brief 分析结果结构
     */
    struct LayoutAnalysisResult {
        GridData::MemoryLayout recommendedLayout;
        GridData::AccessPattern recommendedAccessPattern;
        std::vector<oscean::core_services::CoordinateDimension> dimensionOrder;
        std::string rationale;  // 推荐理由
        bool shouldConvertLayout;  // 是否建议转换布局
    };
    
    /**
     * @brief 分析变量的最优内存布局
     * 
     * @param varInfo 变量元数据
     * @param targetUsage 目标用途（插值、可视化、计算等）
     * @return 布局分析结果
     */
    static LayoutAnalysisResult analyzeOptimalLayout(
        const VariableMeta& varInfo,
        const std::string& targetUsage = "general");
    
    /**
     * @brief 检测NetCDF文件的实际存储布局
     * 
     * @param varInfo 变量元数据
     * @return 检测到的内存布局
     */
    static GridData::MemoryLayout detectStorageLayout(const VariableMeta& varInfo);
    
    /**
     * @brief 判断是否需要为特定用途转换布局
     * 
     * @param currentLayout 当前布局
     * @param varInfo 变量信息
     * @param targetUsage 目标用途
     * @return 是否需要转换
     */
    static bool shouldConvertForUsage(
        GridData::MemoryLayout currentLayout,
        const VariableMeta& varInfo,
        const std::string& targetUsage);
    
private:
    /**
     * @brief 分析维度顺序
     */
    static std::vector<oscean::core_services::CoordinateDimension> analyzeDimensionOrder(
        const std::vector<std::string>& dimensionNames);
    
    /**
     * @brief 判断是否是深度优先的数据
     */
    static bool isDepthFirstData(const VariableMeta& varInfo);
    
    /**
     * @brief 判断是否是时间序列数据
     */
    static bool isTimeSeriesData(const VariableMeta& varInfo);
    
    /**
     * @brief 根据数据特征推荐访问模式
     */
    static GridData::AccessPattern recommendAccessPattern(
        const VariableMeta& varInfo,
        const std::string& targetUsage);
};

} // namespace 