#include "readers/core/unified_data_reader.h"
#include "common_utils/utilities/logging_utils.h"

namespace oscean::core_services::data_access::readers {

UnifiedDataReader::UnifiedDataReader(const std::string& filePath)
    : filePath_(filePath)
    , isOpen_(false) {
    LOG_DEBUG("创建UnifiedDataReader，文件路径: {}", filePath);
}

} // namespace oscean::core_services::data_access::readers 