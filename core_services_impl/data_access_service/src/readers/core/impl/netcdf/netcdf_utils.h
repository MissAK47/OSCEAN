#pragma once

/**
 * @file netcdf_utils.h
 * @brief NetCDF通用工具函数 - 消除重复代码
 */

#include "core_services/common_data_types.h"
#include <netcdf.h>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <initializer_list>
#include <chrono>

namespace oscean::core_services::data_access::readers::impl::netcdf::NetCDFUtils {

/**
 * @brief 转换NetCDF数据类型到统一数据类型
 * @param ncType NetCDF数据类型标识符
 * @return 统一的数据类型枚举
 */
oscean::core_services::DataType convertNetCDFType(int ncType);

/**
 * @brief 检查NetCDF错误并记录日志
 * @param status NetCDF函数返回的状态码
 * @param operation 操作描述
 * @return 是否成功（status == NC_NOERR）
 */
bool checkNetCDFError(int status, const std::string& operation);

/**
 * @brief 读取数值属性
 * @param ncid NetCDF文件ID
 * @param varid 变量ID（NC_GLOBAL表示全局属性）
 * @param attName 属性名称
 * @param defaultValue 默认值
 * @return 属性值，如果不存在返回默认值
 */
double readNumericAttribute(int ncid, int varid, const std::string& attName, double defaultValue);

/**
 * @brief 检查属性是否存在
 * @param ncid NetCDF文件ID
 * @param varid 变量ID
 * @param attName 属性名称
 * @return 属性是否存在
 */
bool hasAttribute(int ncid, int varid, const std::string& attName);

/**
 * @brief 检查变量是否存在
 * @param ncid NetCDF文件ID
 * @param varName 变量名称
 * @return 变量是否存在
 */
bool variableExists(int ncid, const std::string& varName);

/**
 * @brief 获取变量ID
 * @param ncid NetCDF文件ID
 * @param varName 变量名称
 * @return 变量ID，如果不存在返回-1
 */
int getVariableId(int ncid, const std::string& varName);

/**
 * @brief 获取维度长度
 * @param ncid NetCDF文件ID
 * @param dimName 维度名称
 * @return 维度长度，如果不存在返回0
 */
size_t getDimensionLength(int ncid, const std::string& dimName);

/**
 * @brief 检查NetCDF文件格式
 * @param ncid NetCDF文件ID
 * @return 格式字符串（"Classic", "64-bit", "NetCDF-4", etc.）
 */
std::string getNetCDFFormat(int ncid);

/**
 * @brief 计算数据大小（字节）
 * @param dataType NetCDF数据类型
 * @param elementCount 元素数量
 * @return 总字节数
 */
size_t calculateDataSize(int dataType, size_t elementCount);

/**
 * @brief 检查NetCDF操作的状态并记录错误
 * @param status NetCDF函数返回的状态码
 * @param context 描述操作上下文的字符串
 * @return 如果操作成功则返回true，否则返回false
 */
bool checkNetCDFError(int status, const std::string& context);

/**
 * @brief 从NetCDF变量或全局属性中读取字符串属性
 * @param ncid NetCDF文件ID
 * @param varid 变量ID (NC_GLOBAL表示全局属性)
 * @param attName 属性名称
 * @param defaultValue 获取失败时返回的默认值
 * @return 属性的字符串值
 */
std::string readStringAttribute(int ncid, int varid, const std::string& attName, const std::string& defaultValue);

/**
 * @brief 获取变量的ID，处理未找到的情况
 * @param ncid NetCDF文件ID
 * @param varName 变量名
 * @return 变量ID，如果未找到则返回-1
 */
int getVarId(int ncid, const std::string& varName);

/**
 * @brief 读取NetCDF文件中的所有全局属性
 * @param ncid NetCDF文件ID
 * @return 包含所有全局属性的map
 */
std::map<std::string, std::string> readGlobalAttributes(int ncid);

/**
 * @brief 读取NetCDF文件中的所有维度及其详细信息（包括坐标值）
 * @param ncid NetCDF文件ID
 * @return 包含所有维度详情的vector
 */
std::vector<oscean::core_services::DimensionDetail> readDimensionDetails(int ncid);

/**
 * @brief 读取NetCDF文件中的所有变量及其元数据
 * @param ncid NetCDF文件ID
 * @return 包含所有变量元数据的vector
 */
std::vector<oscean::core_services::VariableMeta> readAllVariablesMetadata(int ncid);

/**
 * @brief 从指定的NetCDF变量中读取所有数据并返回为double类型的vector
 * @param ncid NetCDF文件ID
 * @param varName 变量名
 * @return 包含变量数据的vector，如果失败则为空
 */
std::vector<double> readVariableDataDouble(int ncid, const std::string& varName);

/**
 * @brief 将 std::chrono::time_point 转换为 ISO 8601 格式的字符串
 * @param tp 要转换的时间点
 * @return ISO 8601 格式的字符串
 */
std::string timePointToISOString(const std::chrono::system_clock::time_point& tp);

} // namespace oscean::core_services::data_access::readers::impl::netcdf::NetCDFUtils 