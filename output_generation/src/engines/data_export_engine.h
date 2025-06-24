#pragma once

#include "core_services/output/i_output_service.h"
#include <memory>
#include <string>
#include <boost/thread/future.hpp>

// Forward declarations
namespace oscean {
namespace common_utils { 
namespace infrastructure {
    class UnifiedThreadPoolManager;
} // namespace infrastructure
} // namespace common_utils

namespace core_services {
    struct GridData;
    struct FeatureCollection;
    class IDataReader;
} // namespace core_services

namespace output {
namespace internal {
    class WriterFactory;
    class IWriter;
} // namespace internal

/**
 * @class DataExportEngine
 * @brief Orchestrates the process of exporting data to file formats like CSV, NetCDF, etc.
 *
 * This engine is responsible for handling all non-visualization output requests.
 * It acts as a controller that determines which specific writer is needed for a
 * given format, obtains it from a factory, and then delegates the actual writing
 * task. It does not contain any format-specific logic itself.
 */
class DataExportEngine {
public:
    /**
     * @brief Constructs the engine using Dependency Injection.
     * @param writerFactory A factory for creating format-specific writer instances.
     * @param threadPool A thread pool for executing the writing task asynchronously.
     */
    DataExportEngine(
        std::shared_ptr<internal::WriterFactory> writerFactory,
        std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPool
    );

    /**
     * @brief Processes a data export request.
     * @param request The detailed, low-level request specifying what to export.
     * @return A future that will contain the OutputResult.
     *
     * The method performs the following steps:
     * 1. Asynchronously runs the task on the thread pool.
     * 2. Obtains the appropriate IWriter from the WriterFactory based on the request's format.
     * 3. Delegates the writing task to the obtained writer.
     * 4. Wraps the list of file paths from the writer into an OutputResult.
     */
    boost::future<oscean::core_services::output::OutputResult> process(
        const oscean::core_services::output::OutputRequest& request);

private:
    std::shared_ptr<internal::WriterFactory> m_writerFactory;
    std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> m_threadPool;

    /**
     * @brief 处理分块导出的主要逻辑
     */
    std::vector<std::string> processChunkedExport(
        std::shared_ptr<internal::IWriter> writer,
        std::shared_ptr<oscean::core_services::IDataReader> reader,
        const oscean::core_services::output::OutputRequest& request);

    /**
     * @brief 按行分块处理数据
     */
    std::vector<std::string> processChunkedByRow(
        std::shared_ptr<internal::IWriter> writer,
        std::shared_ptr<oscean::core_services::IDataReader> reader,
        const oscean::core_services::output::OutputRequest& request,
        const std::string& variableName,
        double maxSizeBytes);

    /**
     * @brief 按切片分块处理数据（多波段数据）
     */
    std::vector<std::string> processChunkedBySlice(
        std::shared_ptr<internal::IWriter> writer,
        std::shared_ptr<oscean::core_services::IDataReader> reader,
        const oscean::core_services::output::OutputRequest& request,
        const std::string& variableName,
        double maxSizeBytes);

    /**
     * @brief 生成分块文件名
     */
    std::string generateChunkFilename(
        const oscean::core_services::output::OutputRequest& request,
        size_t chunkIndex);

    /**
     * @brief 创建GridData的行子集
     */
    std::shared_ptr<oscean::core_services::GridData> createGridDataSubset(
        std::shared_ptr<oscean::core_services::GridData> fullData,
        size_t startRow, size_t endRow);

    /**
     * @brief 创建GridData的波段子集
     */
    std::shared_ptr<oscean::core_services::GridData> createGridDataBandSubset(
        std::shared_ptr<oscean::core_services::GridData> fullData,
        size_t startBand, size_t endBand);

    /**
     * @brief 处理Grid数据的分块写入（遗留方法）
     */
    std::vector<std::string> processGridDataInChunks(
        std::shared_ptr<internal::IWriter> writer, 
        const oscean::core_services::output::OutputRequest& request,
        std::shared_ptr<oscean::core_services::GridData> fullGridData);

    /**
     * @brief 处理FeatureCollection数据的分块写入（遗留方法）
     */
    std::vector<std::string> processFeatureCollectionInChunks(
        std::shared_ptr<internal::IWriter> writer, 
        const oscean::core_services::output::OutputRequest& request,
        std::shared_ptr<oscean::core_services::FeatureCollection> featureCollection);
};

} // namespace output
} // namespace oscean 