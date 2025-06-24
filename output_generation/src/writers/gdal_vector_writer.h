#pragma once
#include "writers/i_writer.h"
#include <gdal_priv.h>
#include <memory>

namespace oscean {
namespace output {
namespace internal {

class GdalVectorWriter : public IWriter {
public:
    GdalVectorWriter();
    ~GdalVectorWriter() override;

    // 实现新的分块接口
    boost::future<void> open(const std::string& path, const core_services::output::OutputRequest& request) override;
    boost::future<void> writeChunk(const boost::variant<std::shared_ptr<core_services::GridData>, std::shared_ptr<core_services::FeatureCollection>>& dataChunk) override;
    boost::future<void> close() override;

    // 实现旧的整体写入接口
    boost::future<std::vector<std::string>> write(
        std::shared_ptr<oscean::core_services::IDataReader> reader,
        const core_services::output::OutputRequest& request) override;

private:
    // 使用lambda删除器避免重复定义问题
    std::unique_ptr<GDALDataset, void(*)(GDALDataset*)> m_dataSource;
    // OGR图层指针，不由unique_ptr管理，其生命周期由m_dataSource拥有
    OGRLayer* m_layer = nullptr; 
    core_services::output::OutputRequest m_request;
    std::string m_path;
};

} // namespace internal
} // namespace output
} // namespace oscean 