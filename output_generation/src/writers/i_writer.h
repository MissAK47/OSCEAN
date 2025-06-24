#pragma once

#include <memory>
#include <string>
#include <vector>

#include "boost/thread/future.hpp"
#include "boost/variant.hpp"
#include "core_services/output/i_output_service.h"
#include "core_services/data_access/i_data_reader.h"
#include "core_services/common_data_types.h"

namespace oscean {
namespace output {
namespace internal {

/// @brief Defines the contract for all specific file format writers for chunked and non-chunked writing.
class IWriter {
public:
    virtual ~IWriter() = default;

    // --- 新的分块写入接口 ---

    /**
     * @brief Opens the output file and writes any necessary headers.
     * @param path The full path of the file to be created.
     * @param request The original output request containing all parameters.
     * @return A future that becomes ready when the file is opened.
     */
    virtual boost::future<void> open(const std::string& path, const core_services::output::OutputRequest& request) = 0;

    /**
     * @brief Writes a chunk of data to the currently open file.
     * @param dataChunk A variant holding the data to be written (e.g., a GridData subset).
     * @return A future that becomes ready when the chunk is written.
     */
    virtual boost::future<void> writeChunk(const boost::variant<std::shared_ptr<core_services::GridData>, std::shared_ptr<core_services::FeatureCollection>>& dataChunk) = 0;

    /**
     * @brief Finalizes the file, writes any footers, and closes it.
     * @return A future that becomes ready when the file is closed.
     */
    virtual boost::future<void> close() = 0;

    // --- 旧的非分块写入接口（保留用于简单场景或向后兼容）---
    
    /**
     * @brief Performs a simple, non-chunked writing operation.
     * @param reader A shared pointer to the data source.
     * @param request The detailed, low-level request specifying what and how to write.
     * @return A future containing a vector of generated absolute file paths (usually one).
     */
    virtual boost::future<std::vector<std::string>> write(
        std::shared_ptr<oscean::core_services::IDataReader> reader,
        const core_services::output::OutputRequest& request) = 0;
};

} // namespace internal
} // namespace output
} // namespace oscean 