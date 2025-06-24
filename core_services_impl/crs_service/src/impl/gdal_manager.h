#pragma once

#include <mutex>

namespace oscean::core_services::crs::impl {

/**
 * @brief GDAL环境管理器 - 智能懒初始化方案
 * 
 * 🎯 设计目标：
 * ✅ 线程安全的单次初始化
 * ✅ 真正的懒加载 - 只在需要时初始化
 * ✅ 简单可靠 - 无复杂依赖关系
 * ✅ 高性能 - 后续调用零开销
 */
class GDALManager {
public:
    /**
     * @brief 确保GDAL环境已初始化
     * 
     * 使用std::call_once确保线程安全的单次初始化。
     * 后续调用只有原子标志位检查的微小开销。
     * 
     * @return true 如果GDAL成功初始化或已经初始化
     * @return false 如果GDAL初始化失败
     * @throws std::runtime_error 如果初始化过程中发生严重错误
     */
    static bool ensureInitialized();

    /**
     * @brief 检查GDAL是否已经初始化
     * 
     * @return true 如果GDAL已初始化并且可用
     * @return false 如果GDAL未初始化
     */
    static bool isInitialized();

    /**
     * @brief 获取GDAL驱动程序数量
     * 
     * 这是检查GDAL是否正确初始化的标准方法。
     * 
     * @return GDAL驱动程序数量，0表示未初始化
     */
    static int getDriverCount();

private:
    /**
     * @brief 执行实际的GDAL初始化
     * 
     * 这个方法只会被std::call_once调用一次。
     * 包含所有必要的GDAL和PROJ配置。
     * 
     * @return true 如果初始化成功
     * @return false 如果初始化失败
     */
    static bool performInitialization();

    // 确保这是一个纯静态工具类
    GDALManager() = delete;
    ~GDALManager() = delete;
    GDALManager(const GDALManager&) = delete;
    GDALManager& operator=(const GDALManager&) = delete;
};

} // namespace oscean::core_services::crs::impl 