/**
 * @file memory_layout_analyzer.cpp
 * @brief NetCDF内存布局分析器实现
 */

#include "memory_layout_analyzer.h"
#include "common_utils/utilities/logging_utils.h"
#include <algorithm>

namespace oscean::core_services::data_access::readers::impl::netcdf {

using namespace oscean::core_services;

MemoryLayoutAnalyzer::LayoutAnalysisResult 
MemoryLayoutAnalyzer::analyzeOptimalLayout(
    const VariableMeta& varInfo,
    const std::string& targetUsage) {
    
    LayoutAnalysisResult result;
    
    // 1. 检测存储布局
    result.recommendedLayout = detectStorageLayout(varInfo);
    
    // 2. 分析维度顺序
    result.dimensionOrder = analyzeDimensionOrder(varInfo.dimensionNames);
    
    // 3. 根据用途推荐访问模式
    result.recommendedAccessPattern = recommendAccessPattern(varInfo, targetUsage);
    
    // 4. 决定是否需要转换布局
    result.shouldConvertLayout = false;
    
    // 特殊用途的布局优化
    if (targetUsage == "interpolation") {
        if (isDepthFirstData(varInfo) && result.recommendedLayout == GridData::MemoryLayout::ROW_MAJOR) {
            // 深度优先的插值（如声速剖面）可能受益于列主序
            result.shouldConvertLayout = true;
            result.recommendedLayout = GridData::MemoryLayout::COLUMN_MAJOR;
            result.rationale = "深度优先插值，列主序可提高缓存命中率";
        }
    } else if (targetUsage == "gpu_processing") {
        if (result.recommendedLayout == GridData::MemoryLayout::COLUMN_MAJOR) {
            // GPU总是需要行主序
            result.shouldConvertLayout = true;
            result.recommendedLayout = GridData::MemoryLayout::ROW_MAJOR;
            result.rationale = "GPU处理需要行主序以实现coalesced memory access";
        }
    } else if (targetUsage == "visualization") {
        // 可视化通常需要水平切片，行主序更优
        if (result.recommendedLayout == GridData::MemoryLayout::COLUMN_MAJOR && 
            varInfo.dimensionNames.size() >= 2) {
            result.shouldConvertLayout = true;
            result.recommendedLayout = GridData::MemoryLayout::ROW_MAJOR;
            result.rationale = "可视化需要频繁的水平切片操作";
        }
    }
    
    // 如果不需要转换，提供默认理由
    if (!result.shouldConvertLayout) {
        result.rationale = "保持原始布局，避免转换开销";
    }
    
    LOG_INFO("内存布局分析: 变量={}, 用途={}, 推荐布局={}, 需要转换={}",
             varInfo.name, targetUsage, 
             static_cast<int>(result.recommendedLayout),
             result.shouldConvertLayout);
    
    return result;
}

GridData::MemoryLayout 
MemoryLayoutAnalyzer::detectStorageLayout(const VariableMeta& varInfo) {
    // NetCDF-C API 默认返回行主序
    // 但如果数据是从Fortran程序写入的，可能是列主序
    
    // 检查属性中是否有布局提示
    auto it = varInfo.attributes.find("storage_layout");
    if (it != varInfo.attributes.end()) {
        if (it->second == "column_major" || it->second == "fortran") {
            return GridData::MemoryLayout::COLUMN_MAJOR;
        }
    }
    
    // 检查是否有Fortran约定的维度顺序
    if (!varInfo.dimensionNames.empty()) {
        // Fortran约定：最快变化的维度在前
        // C约定：最快变化的维度在后
        
        // 如果第一个维度是深度或垂直维度，可能是Fortran风格
        const auto& firstDim = varInfo.dimensionNames[0];
        if (firstDim == "depth" || firstDim == "z" || firstDim == "level" ||
            firstDim == "altitude" || firstDim == "pressure") {
            return GridData::MemoryLayout::COLUMN_MAJOR;
        }
    }
    
    // 默认假设是行主序（C风格）
    return GridData::MemoryLayout::ROW_MAJOR;
}

bool MemoryLayoutAnalyzer::shouldConvertForUsage(
    GridData::MemoryLayout currentLayout,
    const VariableMeta& varInfo,
    const std::string& targetUsage) {
    
    auto analysisResult = analyzeOptimalLayout(varInfo, targetUsage);
    return analysisResult.shouldConvertLayout && 
           analysisResult.recommendedLayout != currentLayout;
}

std::vector<oscean::core_services::CoordinateDimension> 
MemoryLayoutAnalyzer::analyzeDimensionOrder(
    const std::vector<std::string>& dimensionNames) {
    
    std::vector<oscean::core_services::CoordinateDimension> order;
    
    for (const auto& dimName : dimensionNames) {
        if (dimName == "lon" || dimName == "longitude" || dimName == "x") {
            order.push_back(oscean::core_services::CoordinateDimension::LON);
        } else if (dimName == "lat" || dimName == "latitude" || dimName == "y") {
            order.push_back(oscean::core_services::CoordinateDimension::LAT);
        } else if (dimName == "depth" || dimName == "z" || dimName == "level" ||
                   dimName == "altitude" || dimName == "pressure") {
            order.push_back(oscean::core_services::CoordinateDimension::VERTICAL);
        } else if (dimName == "time" || dimName == "t") {
            order.push_back(oscean::core_services::CoordinateDimension::TIME);
        }
    }
    
    return order;
}

bool MemoryLayoutAnalyzer::isDepthFirstData(const VariableMeta& varInfo) {
    // 检查是否是深度优先的数据（如声速剖面）
    
    // 1. 检查变量名
    const std::string& name = varInfo.name;
    if (name.find("sound_speed") != std::string::npos ||
        name.find("svp") != std::string::npos ||
        name.find("velocity") != std::string::npos ||
        name.find("temperature") != std::string::npos ||
        name.find("salinity") != std::string::npos) {
        
        // 2. 检查是否有深度维度
        for (const auto& dim : varInfo.dimensionNames) {
            if (dim == "depth" || dim == "z" || dim == "level") {
                return true;
            }
        }
    }
    
    return false;
}

bool MemoryLayoutAnalyzer::isTimeSeriesData(const VariableMeta& varInfo) {
    // 检查是否是时间序列数据
    
    // 1. 检查是否有时间维度
    bool hasTimeDim = false;
    for (const auto& dim : varInfo.dimensionNames) {
        if (dim == "time" || dim == "t") {
            hasTimeDim = true;
            break;
        }
    }
    
    if (!hasTimeDim) {
        return false;
    }
    
    // 2. 检查时间维度是否是主要维度（维度大小较大）
    // 这里简化处理，如果有时间维度就认为是时间序列
    return true;
}

GridData::AccessPattern 
MemoryLayoutAnalyzer::recommendAccessPattern(
    const VariableMeta& varInfo,
    const std::string& targetUsage) {
    
    if (targetUsage == "interpolation") {
        if (isDepthFirstData(varInfo)) {
            return GridData::AccessPattern::SEQUENTIAL_Z;
        } else if (varInfo.dimensionNames.size() == 2) {
            return GridData::AccessPattern::SEQUENTIAL_X;
        } else {
            return GridData::AccessPattern::BLOCK_2D;
        }
    } else if (targetUsage == "visualization") {
        // 可视化通常需要2D块访问
        return GridData::AccessPattern::BLOCK_2D;
    } else if (targetUsage == "time_series_analysis") {
        // 时间序列分析需要沿时间轴访问
        return GridData::AccessPattern::SEQUENTIAL_X;  // 假设时间是X轴
    } else {
        // 默认随机访问
        return GridData::AccessPattern::RANDOM;
    }
}

} // namespace 