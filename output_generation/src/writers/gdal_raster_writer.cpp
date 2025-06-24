// 首先强制包含IDataReader的完整定义
#include "core_services/data_access/i_data_reader.h"
#include "core_services/common_data_types.h"
#include "core_services/exceptions.h"
// 然后包含自己的头文件
#include "writers/gdal_raster_writer.h"

#include <boost/thread/future.hpp>
#include <boost/log/trivial.hpp>
#include <gdal_priv.h>
#include <cpl_conv.h> // For CPLMalloc()
#include <memory>

namespace oscean {
namespace output {
namespace internal {

GDALDataType toGdalDataType(core_services::DataType dataType); // Forward declaration

// 辅助函数：将GridData写入栅格文件
void writeGridToGdalRaster(const std::string& path, std::shared_ptr<core_services::GridData> gridData, const std::string& format) {
    if (!gridData) {
        throw core_services::ServiceException("GdalRasterWriter received null GridData.");
    }

    const auto& def = gridData->getDefinition();
    const size_t width = def.cols;
    const size_t height = def.rows;
    const size_t bands = gridData->getBandCount();

    if (width == 0 || height == 0 || bands == 0) {
        BOOST_LOG_TRIVIAL(warning) << "GridData has zero dimensions or bands. Raster file will not be created.";
        return;
    }

    GDALAllRegister();
    GDALDriverManager* driverManager = GetGDALDriverManager();
    GDALDriver* driver = driverManager->GetDriverByName(format.c_str());
    if (!driver) {
        throw core_services::ServiceException("GDAL driver not available for format: " + format);
    }
    
    GDALDataType gdalType = toGdalDataType(gridData->getDataType());
    if (gdalType == GDT_Unknown) {
        throw core_services::ServiceException("Unsupported data type for GDAL raster export.");
    }
    
    // 使用智能指针管理 GDALDataset
    auto deleter = [](GDALDataset* ds){ if(ds) GDALClose(ds); };
    std::unique_ptr<GDALDataset, decltype(deleter)> dataset(
        driver->Create(path.c_str(), width, height, bands, gdalType, NULL),
        deleter
    );

    if (!dataset) {
        throw core_services::ServiceException("Failed to create GDAL dataset for path: " + path);
    }

    // 设置地理转换和投影
    auto geoTransform = gridData->getGeoTransform();
    if (geoTransform.size() == 6) {
        dataset->SetGeoTransform(geoTransform.data());
    }
    const auto& crs = gridData->getCRS();
    if (!crs.wktext.empty()) {
        dataset->SetProjection(crs.wktext.c_str());
    }

    // 写入波段数据
    for (size_t b = 0; b < bands; ++b) {
        GDALRasterBand* band = dataset->GetRasterBand(b + 1);
        if (!band) continue;
        
        // 计算当前波段数据的偏移量
        const size_t bandOffsetBytes = width * height * gridData->getElementSizeBytes() * b;
        const void* bandData = static_cast<const void*>(gridData->getData().data() + bandOffsetBytes);
        
        CPLErr e = band->RasterIO(GF_Write, 0, 0, width, height, 
                                const_cast<void*>(bandData), width, height, gdalType, 
                                0, 0);
        if (e != CE_None) {
            throw core_services::ServiceException("Failed to write data to GDAL raster band.");
        }
    }
     BOOST_LOG_TRIVIAL(info) << "GDAL Raster data written successfully to " << path;
}

GdalRasterWriter::GdalRasterWriter() {
    GDALAllRegister();
}
GdalRasterWriter::~GdalRasterWriter() = default;

boost::future<void> GdalRasterWriter::open(const std::string& path, const core_services::output::OutputRequest& request) {
    return boost::async(boost::launch::async, [this, path, request]() {
        m_request = request;
        m_path = path; // Store path for dataset creation in the first writeChunk
    });
}

boost::future<void> GdalRasterWriter::writeChunk(const boost::variant<std::shared_ptr<core_services::GridData>, std::shared_ptr<core_services::FeatureCollection>>& dataChunk) {
    return boost::async(boost::launch::async, [this, dataChunk]() {
        auto gridDataPtr = boost::get<std::shared_ptr<core_services::GridData>>(&dataChunk);
        if (!gridDataPtr) {
             BOOST_LOG_TRIVIAL(warning) << "GdalRasterWriter received non-GridData chunk, skipping.";
             return;
        }
        auto& gridData = *gridDataPtr;

        // First chunk: create the dataset
        if (!m_dataset) {
            std::string format = "GTiff";
             if(m_request.format == "geotiff" || m_request.format == "tif") {
                 format = "GTiff";
             }
            
            GDALDriver* driver = GetGDALDriverManager()->GetDriverByName(format.c_str());
            if (!driver) throw core_services::ServiceException("GDAL driver not found for: " + format);

            GDALDataType gdalType = toGdalDataType(gridData->getDataType());
            const auto& def = gridData->getDefinition();
            m_dataset.reset(driver->Create(m_path.c_str(), def.cols, def.rows, gridData->getBandCount(), gdalType, NULL));
            if (!m_dataset) throw core_services::ServiceException("Failed to create GDAL dataset for: " + m_path);
            
            if (gridData->getGeoTransform().size() == 6) {
                m_dataset->SetGeoTransform(const_cast<double*>(gridData->getGeoTransform().data()));
            }
            if (!gridData->getCRS().wktext.empty()) {
                m_dataset->SetProjection(gridData->getCRS().wktext.c_str());
            }
        }
        
        // Write data for each band
        const auto& def = gridData->getDefinition();
        for (size_t b = 0; b < gridData->getBandCount(); ++b) {
            GDALRasterBand* band = m_dataset->GetRasterBand(b + 1);
            const size_t bandOffsetBytes = def.cols * def.rows * gridData->getElementSizeBytes() * b;
            const void* bandData = static_cast<const void*>(gridData->getData().data() + bandOffsetBytes);
            
            CPLErr err = band->RasterIO(GF_Write, 0, 0, def.cols, def.rows, 
                           const_cast<void*>(bandData), def.cols, def.rows, 
                           toGdalDataType(gridData->getDataType()), 0, 0);
            if (err != CE_None) {
                throw core_services::ServiceException("Failed to write raster band data");
            }
        }
    });
}

boost::future<void> GdalRasterWriter::close() {
    return boost::async(boost::launch::async, [this]() {
        m_dataset.reset();
        BOOST_LOG_TRIVIAL(info) << "GDAL Raster file closed.";
    });
}

boost::future<std::vector<std::string>> GdalRasterWriter::write(
    std::shared_ptr<oscean::core_services::IDataReader> reader,
    const core_services::output::OutputRequest& request) {

    return boost::async(boost::launch::async, [this, reader, request]() -> std::vector<std::string> {
        std::string extension = ".tif";
        if(request.format != "GTiff" && request.format != "geotiff" && request.format != "tif") {
            // A real implementation would map format to extension
        }
        std::string baseFilename = request.filenameTemplate ? *request.filenameTemplate : "output";
        std::string targetDir = request.targetDirectory ? *request.targetDirectory : ".";
        std::string outputPath = targetDir + "/" + baseFilename + extension;

        if(reader->listDataVariableNames().empty()){
             throw core_services::ServiceException("Data source has no readable variables.");
        }
        auto gridData = reader->readGridData(reader->listDataVariableNames().at(0));
        
        this->open(outputPath, request).get();
        this->writeChunk(gridData).get();
        this->close().get();

        return std::vector<std::string>{outputPath};
    });
}

// Definition of the conversion function
GDALDataType toGdalDataType(core_services::DataType dataType) {
    switch (dataType) {
        case core_services::DataType::Byte: return GDT_Byte;
        case core_services::DataType::UInt16: return GDT_UInt16;
        case core_services::DataType::Int16: return GDT_Int16;
        case core_services::DataType::UInt32: return GDT_UInt32;
        case core_services::DataType::Int32: return GDT_Int32;
        case core_services::DataType::Float32: return GDT_Float32;
        case core_services::DataType::Float64: return GDT_Float64;
        default: return GDT_Unknown;
    }
}

} // namespace internal
} // namespace output
} // namespace oscean 