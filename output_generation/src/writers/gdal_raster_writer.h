#pragma once
#include "writers/i_writer.h"
#include <gdal_priv.h>
#include <memory>

namespace oscean {
namespace output {
namespace internal {

// 使用自定义删除器来安全关闭 GDALDataset
struct GdalDatasetDeleter {
    void operator()(GDALDataset* ds) const {
        if (ds) {
            GDALClose(ds);
        }
    }
};

class GdalRasterWriter : public IWriter {
public:
    GdalRasterWriter();
    ~GdalRasterWriter() override;

    // 实现新的分块接口
    boost::future<void> open(const std::string& path, const core_services::output::OutputRequest& request) override;
    boost::future<void> writeChunk(const boost::variant<std::shared_ptr<core_services::GridData>, std::shared_ptr<core_services::FeatureCollection>>& dataChunk) override;
    boost::future<void> close() override;

    // 实现旧的整体写入接口
    boost::future<std::vector<std::string>> write(
        std::shared_ptr<oscean::core_services::IDataReader> reader,
        const core_services::output::OutputRequest& request) override;

private:
    std::unique_ptr<GDALDataset, GdalDatasetDeleter> m_dataset;
    core_services::output::OutputRequest m_request;
    std::string m_path;
};

} // namespace internal
} // namespace output
} // namespace oscean 