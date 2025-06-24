#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include "core_services/common_data_types.h"

namespace oscean::core_services::data_access::readers {

/**
 * @brief GDAL读取器类型枚举
 */
enum class GdalReaderType {
    RASTER,  ///< 栅格数据读取器
    VECTOR   ///< 矢量数据读取器
};

} // namespace oscean::core_services::data_access::readers 