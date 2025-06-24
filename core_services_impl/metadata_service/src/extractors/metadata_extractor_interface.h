#pragma once

#include "core_services/metadata/unified_metadata_service.h"
#include <string>
#include <boost/thread/future.hpp>

namespace oscean {
namespace core_services {
namespace metadata {
namespace extractors {

/**
 * @class IMetadataExtractor
 * @brief 定义了所有元数据提取器必须实现的通用接口。
 *
 * 这个接口确保了无论是处理NetCDF, GRIB, GeoTIFF还是其他格式，
 * 上层服务都能以统一的方式调用提取逻辑。
 */
class IMetadataExtractor {
public:
    virtual ~IMetadataExtractor() = default;

    /**
     * @brief 检查此提取器是否能够处理指定的文件。
     * @param filePath 文件的路径。
     * @return 如果可以处理，返回true；否则返回false。
     */
    virtual bool canProcess(const std::string& filePath) const = 0;

    /**
     * @brief 🔧 第三阶段：从指定文件异步提取标准化文件元数据
     * @param filePath 文件的路径。
     * @return 一个包含FileMetadata的异步结果的future。
     */
    virtual boost::future<AsyncResult<::oscean::core_services::FileMetadata>> extract(const std::string& filePath) const = 0;

    /**
     * @brief 指示此提取器是否需要GDAL库的支持。
     * @return 如果需要GDAL，返回true。
     */
    virtual bool requiresGdalSupport() const = 0;
};

} // namespace extractors
} // namespace metadata
} // namespace core_services
} // namespace oscean 