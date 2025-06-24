#pragma once

#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <boost/thread/shared_mutex.hpp>
#include "unified_data_reader.h"
#include "common_utils/utilities/file_format_detector.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/infrastructure/common_services_factory.h"

namespace oscean::core_services::data_access::readers {

/**
 * @brief 读取器工厂函数类型定义 - 支持依赖注入
 */
using ReaderFactory = std::function<std::shared_ptr<UnifiedDataReader>(
    const std::string& filePath,
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices)>;

/**
 * @brief 读取器注册表
 * 
 * 管理不同文件格式的读取器工厂，支持动态注册和创建
 * 使用统一的格式检测器，并确保只支持GDAL和NetCDF真正能处理的格式
 */
class ReaderRegistry {
public:
    /**
     * @brief 构造函数
     * @param formatDetector 统一格式检测器
     * @param commonServices Common服务工厂
     */
    explicit ReaderRegistry(
        std::unique_ptr<oscean::common_utils::utilities::FileFormatDetector> formatDetector,
        std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~ReaderRegistry() = default;

    /**
     * @brief 注册读取器工厂
     * @param format 格式名称（如"GDAL_RASTER", "NETCDF"）
     * @param factory 读取器工厂函数
     * @return 是否注册成功
     */
    bool registerReaderFactory(const std::string& format, ReaderFactory factory);
    
    /**
     * @brief 取消注册读取器工厂
     * @param format 格式名称
     * @return 是否取消成功
     */
    bool unregisterReaderFactory(const std::string& format);
    
    /**
     * @brief 创建读取器 - 移除CRS参数
     * @param filePath 文件路径
     * @param explicitFormat 显式指定的格式（可选）
     * @return 读取器实例，如果无法创建则返回nullptr
     */
    std::shared_ptr<UnifiedDataReader> createReader(
        const std::string& filePath,
        const std::optional<std::string>& explicitFormat = std::nullopt);
    
    /**
     * @brief 检查是否支持指定格式
     * @param format 格式名称
     * @return 是否支持
     */
    bool supportsFormat(const std::string& format) const;
    
    /**
     * @brief 获取所有支持的格式列表
     * @return 格式名称列表
     */
    std::vector<std::string> getSupportedFormats() const;
    
    /**
     * @brief 获取格式检测器
     * @return 格式检测器
     */
    const oscean::common_utils::utilities::FileFormatDetector* getFormatDetector() const { 
        return formatDetector_.get(); 
    }

    /**
     * @brief 检查格式是否被真正支持（在白名单中且有对应读取器）
     * @param format 格式名称
     * @return 是否被真正支持
     */
    bool isFormatTrulySupported(const std::string& format) const;
    
    /**
     * @brief 获取支持格式的字符串表示
     * @return 支持格式的逗号分隔字符串
     */
    std::string getSupportedFormatsString() const;

private:
    /**
     * @brief 检测文件格式
     * @param filePath 文件路径
     * @return 检测到的格式，如果无法检测则返回nullopt
     */
    std::optional<std::string> detectFileFormat(const std::string& filePath) const;

    /**
     * @brief 初始化支持的格式白名单
     */
    void initializeSupportedFormats();
    
    /**
     * @brief 验证注册的格式是否都在白名单中
     */
    void validateRegisteredFormats();
    
    /**
     * @brief 标准化格式名称，解决命名不一致问题
     * @param detectedFormat FileFormatDetector检测到的格式名称
     * @return 标准化后的格式名称
     */
    std::string standardizeFormatName(const std::string& detectedFormat) const;

    // 成员变量
    std::unique_ptr<oscean::common_utils::utilities::FileFormatDetector> formatDetector_; ///< 统一格式检测器
    std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServices_;
    std::unordered_map<std::string, ReaderFactory> readerFactories_;                      ///< 读取器工厂映射
    mutable boost::shared_mutex registryMutex_;                                           ///< 注册表保护互斥锁
    
    std::unordered_set<std::string> supportedFormats_;                                    ///< 严格的格式支持白名单
};

} // namespace oscean::core_services::data_access::readers 
