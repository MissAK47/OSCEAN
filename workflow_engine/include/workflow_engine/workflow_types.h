#pragma once

/**
 * @file workflow_types.h
 * @brief 工作流基础类型定义 - 简化版本
 * @author OSCEAN Team
 * @date 2024
 */

#include <string>
#include <memory>
#include <map>
#include <functional>  // 🔧 修复：添加C++17 std::function支持

namespace oscean::workflow_engine {

/**
 * @brief 工作流类型枚举
 */
enum class WorkflowType {
    DATA_MANAGEMENT = 0,    // 数据管理工作流
    SPATIAL_ANALYSIS = 1,   // 空间分析工作流
    TIME_SERIES = 2,        // 时间序列工作流
    MULTI_FUSION = 3,       // 多源融合工作流
    CUSTOM = 999            // 自定义工作流
};

/**
 * @brief 工作流信息结构
 */
struct WorkflowInfo {
    WorkflowType type;
    std::string name;
    std::string version;
    std::string description;
    bool isAvailable = false;
    
    WorkflowInfo() = default;
    
    WorkflowInfo(WorkflowType t, const std::string& n, const std::string& v, const std::string& d)
        : type(t), name(n), version(v), description(d), isAvailable(true) {
    }
};

/**
 * @brief 简化的工作流工厂函数类型
 */
template<typename ServiceType>
using WorkflowFactory = std::function<std::shared_ptr<ServiceType>()>;

} // namespace oscean::workflow_engine 