#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <spdlog/spdlog.h>

// 包含正确的 GDAL/OGR 头文件
#include <ogr_spatialref.h>
#include <ogr_core.h>
#include <ogrsf_frmts.h>

#include "common_utils/utilities/logging_utils.h"

namespace oscean {
namespace core_services {
namespace crs {
namespace cache {

/**
 * @brief TransformationCache 的 PIMPL 实现
 */
class TransformationCacheImpl {
public:
    TransformationCacheImpl();
    ~TransformationCacheImpl();

    // 禁止拷贝和移动
    TransformationCacheImpl(const TransformationCacheImpl&) = delete;
    TransformationCacheImpl& operator=(const TransformationCacheImpl&) = delete;
    TransformationCacheImpl(TransformationCacheImpl&&) = delete;
    TransformationCacheImpl& operator=(TransformationCacheImpl&&) = delete;

    /**
     * @brief 获取或创建 OGRCoordinateTransformation 对象
     * @param sourceCRS 源坐标系字符串 (WKT 或 PROJ.4)
     * @param targetCRS 目标坐标系字符串 (WKT 或 PROJ.4)
     * @param transformationOut 输出参数，存储转换对象
     * @return 成功返回 true，失败返回 false
     */
    bool getTransformation(
        const std::string& sourceCRS,
        const std::string& targetCRS,
        OGRCoordinateTransformation** transformationOut);

    /**
     * @brief 释放 OGRCoordinateTransformation 对象
     * @param transformation 转换对象指针
     */
    void releaseTransformation(OGRCoordinateTransformation* transformation);

    /**
     * @brief 清空缓存
     */
    void clear();

private:
    /**
     * @brief 创建新的坐标转换对象
     * @param sourceCRS 源坐标系
     * @param targetCRS 目标坐标系
     * @return 成功返回转换对象指针，失败返回 nullptr
     */
    OGRCoordinateTransformation* createTransformation(
        const std::string& sourceCRS,
        const std::string& targetCRS);

    /**
     * @brief 创建缓存键
     * @param sourceCRS 源坐标系
     * @param targetCRS 目标坐标系
     * @return 缓存键
     */
    std::string createCacheKey(
        const std::string& sourceCRS,
        const std::string& targetCRS) const;

    // 线程安全的缓存映射，保存转换对象
    std::unordered_map<std::string, OGRCoordinateTransformation*> _transformationCache;
    std::mutex _mutex; // 保护缓存的互斥锁
    std::shared_ptr<spdlog::logger> _logger; // 日志记录器
};

} // namespace cache
} // namespace crs
} // namespace core_services
} // namespace oscean 