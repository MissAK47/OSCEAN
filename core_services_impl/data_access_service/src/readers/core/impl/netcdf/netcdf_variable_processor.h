#pragma once

/**
 * @file netcdf_variable_processor.h
 * @brief NetCDF变量专用处理器
 */

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <chrono>
#include <map>

#include "core_services/data_access/unified_data_types.h"
#include "core_services/data_access/i_data_reader.h"
#include "core_services/common_data_types.h"
#include "common_utils/utilities/logging_utils.h"

// Forward declarations for NetCDF handles
typedef int ncid_t;
typedef int varid_t;

// 🚀 新增：坐标系统前向声明
namespace oscean::core_services::data_access::readers::impl::netcdf {
    class NetCDFCoordinateSystemExtractor;
}

// 🚀 添加std::pair的hash特化声明，支持属性缓存
namespace std {
template<>
struct hash<std::pair<std::string, std::string>> {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
    }
};
}

namespace oscean::core_services::data_access::readers::impl::netcdf {

/**
 * @brief 变量读取选项
 */
struct VariableReadOptions {
    std::optional<oscean::core_services::BoundingBox> bounds;  ///< 空间边界
    std::optional<std::pair<std::chrono::system_clock::time_point, std::chrono::system_clock::time_point>> timeRange; ///< 时间范围
    std::optional<std::pair<size_t, size_t>> verticalRange;   ///< 垂直范围（索引）
    std::vector<size_t> stride;                               ///< 步长
    bool applyScaleOffset = true;                             ///< 是否应用缩放和偏移
    bool handleNoData = true;                                 ///< 是否处理NoData值
};

/**
 * @brief 🚀 空间索引结构体 - 用于高效的空间子集读取
 */
struct SpatialIndices {
    size_t lonStartIndex = 0;  ///< 经度起始索引
    size_t lonEndIndex = 0;    ///< 经度结束索引
    size_t latStartIndex = 0;  ///< 纬度起始索引  
    size_t latEndIndex = 0;    ///< 纬度结束索引
    size_t lonDimIndex = 0;    ///< 经度维度在变量中的位置
    size_t latDimIndex = 0;    ///< 纬度维度在变量中的位置
};

/**
 * @brief NetCDF变量专用处理器
 * 
 * 负责NetCDF变量的读取、处理和转换
 */
class NetCDFVariableProcessor {
public:
    explicit NetCDFVariableProcessor(ncid_t ncid);
    virtual ~NetCDFVariableProcessor() = default;
    
    // =============================================================================
    // 变量信息获取
    // =============================================================================
    
    /**
     * @brief 获取所有变量名
     */
    std::vector<std::string> getVariableNames() const;
    
    /**
     * @brief 获取变量详细信息
     */
    std::optional<oscean::core_services::VariableMeta> getVariableInfo(const std::string& variableName) const;
    
    /**
     * @brief 检查变量是否存在
     */
    bool variableExists(const std::string& variableName) const;
    
    /**
     * @brief 获取变量ID
     */
    varid_t getVariableId(const std::string& variableName) const;
    
    /**
     * @brief 获取变量维度数量
     */
    int getVariableDimensionCount(const std::string& variableName) const;
    
    /**
     * @brief 获取变量形状
     */
    std::vector<size_t> getVariableShape(const std::string& variableName) const;
    
    // =============================================================================
    // 变量数据读取
    // =============================================================================
    
    /**
     * @brief 读取完整变量数据
     */
    std::shared_ptr<oscean::core_services::GridData> readVariable(
        const std::string& variableName,
        const VariableReadOptions& options = {}
    ) const;
    
    /**
     * @brief 读取变量子集
     */
    std::vector<double> readVariableSubset(
        const std::string& variableName,
        const std::vector<size_t>& start,
        const std::vector<size_t>& count,
        const std::vector<size_t>& stride = {}
    ) const;
    
    /**
     * @brief 读取变量的单个时间步
     */
    std::shared_ptr<oscean::core_services::GridData> readVariableTimeStep(
        const std::string& variableName,
        size_t timeIndex,
        const VariableReadOptions& options = {}
    ) const;
    
    /**
     * @brief 读取变量的单个垂直层
     */
    std::shared_ptr<oscean::core_services::GridData> readVariableLevel(
        const std::string& variableName,
        size_t levelIndex,
        const VariableReadOptions& options = {}
    ) const;
    
    // =============================================================================
    // 变量属性处理
    // =============================================================================
    
    /**
     * @brief 获取变量属性
     */
    std::vector<oscean::core_services::MetadataEntry> getVariableAttributes(const std::string& variableName) const;
    
    /**
     * @brief 读取字符串属性
     */
    std::string readStringAttribute(const std::string& variableName, const std::string& attributeName) const;
    
    /**
     * @brief 读取数值属性
     */
    double readNumericAttribute(const std::string& variableName, const std::string& attributeName, double defaultValue = 0.0) const;
    
    /**
     * @brief 检查属性是否存在
     */
    bool hasAttribute(const std::string& variableName, const std::string& attributeName) const;
    
    // =============================================================================
    // 数据处理和转换
    // =============================================================================
    
    /**
     * @brief 应用缩放因子和偏移量
     */
    void applyScaleAndOffset(std::vector<double>& data, double scaleFactor, double addOffset) const;
    
    /**
     * @brief 处理NoData值
     */
    void handleNoDataValues(std::vector<double>& data, double noDataValue) const;
    
    /**
     * @brief 转换数据类型
     */
    std::vector<double> convertToDouble(const void* data, int ncType, size_t count) const;
    
    /**
     * @brief 验证数据完整性
     */
    bool validateData(const std::vector<double>& data, const oscean::core_services::VariableMeta& varInfo) const;
    
    // =============================================================================
    // 空间子集处理
    // =============================================================================
    
    /**
     * @brief 🚀 计算空间子集的索引范围 - 优化版本
     */
    std::optional<SpatialIndices> calculateSpatialIndices(
        const std::string& variableName,
        const oscean::core_services::BoundingBox& bounds
    ) const;
    
    /**
     * @brief 🚀 应用空间子集到读取参数
     */
    void applySpatialSubset(
        const std::vector<std::string>& dimensions,
        const std::vector<size_t>& shape,
        const SpatialIndices& spatialIndices,
        std::vector<size_t>& start,
        std::vector<size_t>& count
    ) const;
    
    /**
     * @brief 🚀 应用时间子集到读取参数
     */
    void applyTimeSubset(
        const std::vector<std::string>& dimensions,
        const std::vector<size_t>& shape,
        const std::pair<std::chrono::system_clock::time_point, std::chrono::system_clock::time_point>& timeRange,
        std::vector<size_t>& start,
        std::vector<size_t>& count
    ) const;
    
    /**
     * @brief 🚀 读取坐标数据
     */
    std::vector<double> readCoordinateData(const std::string& coordName) const;
    
    // =============================================================================
    // 缓存管理
    // =============================================================================
    
    /**
     * @brief 清除变量信息缓存
     */
    void clearCache();
    
    /**
     * @brief 预加载变量信息到缓存
     */
    void preloadVariableInfo();
    
    /**
     * @brief 🚀 积极预加载所有基础数据，避免重复IO
     * 
     * 该方法在构造函数中被调用，一次性加载：
     * - 所有变量的基本信息和形状
     * - 所有坐标数据（longitude, latitude, depth, time等）
     * - 所有变量的核心属性
     * 
     * 预加载完成后，后续的所有读取操作都将使用缓存，实现零IO。
     */
    void preloadEssentialData();
    
    /**
     * @brief 获取缓存统计
     */
    struct CacheStats {
        size_t cachedVariables = 0;
        size_t totalVariables = 0;
        size_t cacheHits = 0;
        size_t cacheMisses = 0;
    };
    
    CacheStats getCacheStats() const;
    
private:
    ncid_t ncid_;
    
    // 🚀 新增：坐标系统处理器
    std::shared_ptr<NetCDFCoordinateSystemExtractor> coordinateSystem_;
    
    // 缓存
    mutable std::optional<std::vector<std::string>> cachedVariableNames_;
    mutable std::unordered_map<std::string, oscean::core_services::VariableMeta> cachedVariableInfo_;
    mutable std::unordered_map<std::string, std::vector<oscean::core_services::MetadataEntry>> cachedVariableAttributes_;
    
    // 🔧 新增：增强的缓存容器，避免重复IO
    mutable std::unordered_map<std::string, std::vector<size_t>> cachedVariableShapes_;
    mutable std::unordered_map<std::pair<std::string, std::string>, std::string> cachedAttributesMap_;
    mutable std::unordered_map<std::string, std::vector<double>> coordinateDataCache_;
    
    // 统计
    mutable CacheStats cacheStats_;
    
    // 内部方法
    oscean::core_services::VariableMeta extractVariableInfo(const std::string& variableName) const;
    std::vector<oscean::core_services::MetadataEntry> extractVariableAttributes(const std::string& variableName) const;
    
    // 🔧 新增：批量读取方法，避免重复IO
    std::map<std::string, std::string> batchReadVariableAttributes(const std::string& variableName) const;
    
    // 数据类型转换
    oscean::core_services::DataType convertNetCDFDataType(int ncType) const;
    
    // 数据读取辅助方法
    std::vector<double> readVariableData(varid_t varid, const std::vector<size_t>& start, const std::vector<size_t>& count) const;
    std::shared_ptr<oscean::core_services::GridData> createGridData(
        const std::string& variableName,
        const std::vector<double>& data,
        const std::vector<size_t>& shape,
        const oscean::core_services::VariableMeta& varInfo
    ) const;
    
    // CF约定处理
    void parseCFConventions(const std::string& variableName, oscean::core_services::VariableMeta& varInfo) const;
    
    // 坐标处理
    std::vector<double> extractCoordinateValues(const std::string& dimName) const;
    
    // 验证方法
    bool validateVariableName(const std::string& variableName) const;
    bool validateReadParameters(const std::vector<size_t>& start, const std::vector<size_t>& count) const;
};

} // namespace oscean::core_services::data_access::readers::impl::netcdf 