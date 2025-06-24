#pragma once

#include "core_services/spatial_ops/i_spatial_ops_service.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/crs/i_crs_service.h"
#include "core_services/data_access/i_raw_data_access_service.h"
#include "core_services/interpolation/i_interpolation_service.h"
#include <memory>
#include <vector>
#include <string>

// GDAL头文件包含
#ifdef GDAL_FOUND
#include "cpl_error.h"
#endif

namespace oscean::core_services::spatial_ops {

/**
 * @brief 空间服务工厂类
 * 负责创建和配置空间服务实例，管理依赖注入和初始化过程
 */
class SpatialOpsServiceFactory {
public:
    /**
     * @brief 创建空间服务实例（使用默认配置）
     * @return 空间服务实例的智能指针
     */
    static std::unique_ptr<ISpatialOpsService> createService();
    
    /**
     * @brief 创建空间服务实例（使用指定配置）
     * @param config 空间服务配置
     * @return 空间服务实例的智能指针
     */
    static std::unique_ptr<ISpatialOpsService> createService(
        const SpatialOpsConfig& config);
    
    /**
     * @brief 创建空间服务实例（完整依赖注入）
     * @param config 空间服务配置
     * @param crsService CRS服务实例（用于坐标转换）
     * @param dataAccessService 数据访问服务实例（用于GDAL基础设施重用）
     * @param interpolationService 插值服务实例（用于重采样等操作）
     * @return 空间服务实例的智能指针
     */
    static std::unique_ptr<ISpatialOpsService> createService(
        const SpatialOpsConfig& config,
        std::shared_ptr<oscean::core_services::ICrsService> crsService,
        std::shared_ptr<oscean::core_services::IRawDataAccessService> dataAccessService,
        std::shared_ptr<oscean::core_services::interpolation::IInterpolationService> interpolationService = nullptr);
    
    /**
     * @brief 创建用于测试的Mock服务实例
     * @return Mock空间服务实例的智能指针
     */
    static std::unique_ptr<ISpatialOpsService> createMockService();
    
    /**
     * @brief 验证配置的有效性
     * @param config 要验证的配置
     * @return 如果配置有效返回true，否则返回false
     */
    static bool validateConfig(const SpatialOpsConfig& config);
    
    /**
     * @brief 获取默认配置
     * @return 默认的空间服务配置
     */
    static SpatialOpsConfig getDefaultConfig();

    /**
     * @brief 清理工厂资源（主要是GDAL资源）
     */
    static void cleanup();

    /**
     * @brief 获取GDAL版本信息
     * @return GDAL版本字符串
     */
    static std::string getGDALVersion();

    /**
     * @brief 检查GDAL是否可用
     * @return 如果GDAL可用返回true
     */
    static bool isGDALAvailable();

    /**
     * @brief 获取支持的数据格式列表
     * @return 支持的格式列表
     */
    static std::vector<std::string> getSupportedFormats();

private:
    // 私有辅助方法
    static void initializeGDAL(const SpatialOpsConfig& config);
    static void setupPerformanceOptimizations(const SpatialOpsConfig& config);
    static void initializeGDALForTesting(const SpatialOpsConfig& config);
    static void validateDependencies(
        std::shared_ptr<oscean::core_services::ICrsService> crsService,
        std::shared_ptr<oscean::core_services::IRawDataAccessService> dataAccessService,
        std::shared_ptr<oscean::core_services::interpolation::IInterpolationService> interpolationService = nullptr);

#ifdef GDAL_FOUND
    /**
     * @brief GDAL错误处理回调函数
     * @param eErrClass 错误类别
     * @param nError 错误编号
     * @param pszErrorMsg 错误消息
     */
    static void gdalErrorHandler(CPLErr eErrClass, CPLErrorNum nError, const char* pszErrorMsg);
#endif
    
    // 防止实例化
    SpatialOpsServiceFactory() = delete;
    ~SpatialOpsServiceFactory() = delete;
    SpatialOpsServiceFactory(const SpatialOpsServiceFactory&) = delete;
    SpatialOpsServiceFactory& operator=(const SpatialOpsServiceFactory&) = delete;
};

} // namespace oscean::core_services::spatial_ops 