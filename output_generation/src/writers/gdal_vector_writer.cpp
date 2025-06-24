// 首先强制包含IDataReader的完整定义
#include "core_services/data_access/i_data_reader.h"
#include "core_services/common_data_types.h"
#include "core_services/exceptions.h"
// 然后包含自己的头文件
#include "writers/gdal_vector_writer.h"
#include "writers/gdal_raster_writer.h"  // 为了获取GdalDatasetDeleter定义

#include <boost/thread/future.hpp>
#include <boost/log/trivial.hpp>
#include <boost/variant.hpp>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <memory>
#include <stdexcept>
#include <cpl_conv.h>
#include <variant>
#include <type_traits>

namespace oscean {
namespace output {
namespace internal {

// 辅助函数：将字符串类型转换为OGR字段类型
OGRFieldType toOgrFieldType(const std::string& typeStr) {
    if (typeStr == "string" || typeStr == "text") return OFTString;
    if (typeStr == "integer" || typeStr == "int") return OFTInteger;
    if (typeStr == "real" || typeStr == "double" || typeStr == "float") return OFTReal;
    if (typeStr == "date") return OFTDate;
    if (typeStr == "time") return OFTTime;
    if (typeStr == "datetime") return OFTDateTime;
    return OFTString; // 默认为字符串
}



// 辅助函数：将FeatureCollection写入矢量文件
void writeFeaturesToGdalVector(const std::string& path, std::shared_ptr<core_services::FeatureCollection> features, const std::string& format) {
    if (!features || features->empty()) {
        BOOST_LOG_TRIVIAL(warning) << "FeatureCollection is null or empty. Vector file will not be created.";
        return;
    }

    GDALAllRegister();
    GDALDriverManager* driverManager = GetGDALDriverManager();
    GDALDriver* driver = driverManager->GetDriverByName(format.c_str());
    if (!driver) {
        throw core_services::ServiceException("GDAL/OGR driver not available for format: " + format);
    }
    
    // 删除已存在的文件，因为一些驱动（如Shapefile）不支持覆盖
    VSIStatBufL sStat;
    if (VSIStatL(path.c_str(), &sStat) == 0) {
        driver->Delete(path.c_str());
    }

    auto dsDeleter = [](GDALDataset* ds){ if(ds) GDALClose(ds); };
    std::unique_ptr<GDALDataset, decltype(dsDeleter)> ds(
        driver->Create(path.c_str(), 0, 0, 0, GDT_Unknown, NULL),
        dsDeleter
    );

    if (!ds) {
        throw core_services::ServiceException("Failed to create GDAL/OGR datasource for path: " + path);
    }

    OGRSpatialReference oSRS;
    if (features->crs.has_value() && !features->crs->wktext.empty()) {
        oSRS.importFromWkt(features->crs->wktext.c_str());
    } else {
        oSRS.SetWellKnownGeogCS("WGS84"); // 默认使用WGS84
    }

    OGRLayer* layer = ds->CreateLayer(features->name.c_str(), &oSRS, wkbUnknown, NULL);
    if (!layer) {
        throw core_services::ServiceException("Failed to create layer in OGR datasource.");
    }
    
    // 创建字段
    for (const auto& fieldDef : features->fieldDefinitions) {
        OGRFieldDefn oField(fieldDef.name.c_str(), toOgrFieldType(fieldDef.type));
        if (layer->CreateField(&oField) != OGRERR_NONE) {
            throw core_services::ServiceException("Failed to create field in layer: " + fieldDef.name);
        }
    }
    
    // 写入要素
    for (const auto& feature : features->getFeatures()) {
        OGRFeature* oFeature = OGRFeature::CreateFeature(layer->GetLayerDefn());
        
        // 设置几何
        OGRGeometry* geom = nullptr;
        char* wkt = const_cast<char*>(feature.geometryWkt.c_str());
        OGRGeometryFactory::createFromWkt(&wkt, nullptr, &geom);
        if (geom) {
            oFeature->SetGeometry(geom);
            OGRGeometryFactory::destroyGeometry(geom);
        }
        
        // 设置属性
        for (const auto& attr : feature.attributes) {
            int fieldIndex = oFeature->GetFieldIndex(attr.first.c_str());
            if (fieldIndex != -1) {
                std::visit([&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::string>) oFeature->SetField(fieldIndex, arg.c_str());
                    else if constexpr (std::is_same_v<T, int>) oFeature->SetField(fieldIndex, arg);
                    else if constexpr (std::is_same_v<T, double>) oFeature->SetField(fieldIndex, arg);
                }, attr.second);
            }
        }

        if (layer->CreateFeature(oFeature) != OGRERR_NONE) {
            OGRFeature::DestroyFeature(oFeature);
            throw core_services::ServiceException("Failed to create feature in layer.");
        }
        OGRFeature::DestroyFeature(oFeature);
    }
    BOOST_LOG_TRIVIAL(info) << "GDAL Vector data written successfully to " << path;
}

GdalVectorWriter::GdalVectorWriter() 
    : m_dataSource(nullptr, [](GDALDataset* ds) { if (ds) GDALClose(ds); }) {
    GDALAllRegister();
}
GdalVectorWriter::~GdalVectorWriter() = default;

boost::future<void> GdalVectorWriter::open(const std::string& path, const core_services::output::OutputRequest& request) {
    return boost::async(boost::launch::async, [this, path, request]() {
        m_request = request;
        m_path = path;
    });
}

boost::future<void> GdalVectorWriter::writeChunk(const boost::variant<std::shared_ptr<core_services::GridData>, std::shared_ptr<core_services::FeatureCollection>>& dataChunk) {
    return boost::async(boost::launch::async, [this, dataChunk]() {
        auto featureCollPtr = boost::get<std::shared_ptr<core_services::FeatureCollection>>(&dataChunk);
        if (!featureCollPtr) {
            BOOST_LOG_TRIVIAL(warning) << "GdalVectorWriter received non-FeatureCollection chunk, skipping.";
            return;
        }
        auto& featureColl = *featureCollPtr;

        // First chunk: create data source, layer, and fields
        if (!m_dataSource) {
            std::string format = "ESRI Shapefile";
             if(m_request.format == "shapefile" || m_request.format == "shp") {
                 format = "ESRI Shapefile";
             }
            
            GDALDriver* driver = GetGDALDriverManager()->GetDriverByName(format.c_str());
            if (!driver) throw core_services::ServiceException("OGR driver not found for: " + format);
            
            m_dataSource.reset(driver->Create(m_path.c_str(), 0, 0, 0, GDT_Unknown, NULL));
            if (!m_dataSource) throw core_services::ServiceException("Failed to create OGR data source for: " + m_path);
            
            OGRSpatialReference srs;
            if(featureColl->crs.has_value() && !featureColl->crs->wktext.empty()) {
                srs.importFromWkt(featureColl->crs->wktext.c_str());
            }

            m_layer = m_dataSource->CreateLayer("features", &srs, wkbUnknown, NULL);
            if (!m_layer) throw core_services::ServiceException("Failed to create OGR layer.");
            
            for(const auto& fieldDef : featureColl->fieldDefinitions) {
                OGRFieldDefn field(fieldDef.name.c_str(), toOgrFieldType(fieldDef.type));
                if(m_layer->CreateField(&field) != OGRERR_NONE) {
                    throw core_services::ServiceException("Failed to create field: " + fieldDef.name);
                }
            }
        }

        // Write features from the chunk
        for (const auto& feature : featureColl->getFeatures()) {
            OGRFeature* ogrFeature = OGRFeature::CreateFeature(m_layer->GetLayerDefn());
            
            // Set geometry from WKT
            OGRGeometry* geom = nullptr;
            char* wkt = const_cast<char*>(feature.geometryWkt.c_str());
            if (OGRGeometryFactory::createFromWkt(&wkt, nullptr, &geom) == OGRERR_NONE && geom) {
                ogrFeature->SetGeometry(geom);
                OGRGeometryFactory::destroyGeometry(geom);
            }

            // Set fields from attributes
            for(const auto& attr : feature.attributes) {
                int fieldIndex = ogrFeature->GetFieldIndex(attr.first.c_str());
                if (fieldIndex != -1) {
                    std::visit([&](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, std::string>) ogrFeature->SetField(fieldIndex, arg.c_str());
                        else if constexpr (std::is_same_v<T, int>) ogrFeature->SetField(fieldIndex, arg);
                        else if constexpr (std::is_same_v<T, double>) ogrFeature->SetField(fieldIndex, arg);
                    }, attr.second);
                }
            }

            if(m_layer->CreateFeature(ogrFeature) != OGRERR_NONE) {
                 OGRFeature::DestroyFeature(ogrFeature);
                 throw core_services::ServiceException("Failed to create feature in OGR layer.");
            }
            OGRFeature::DestroyFeature(ogrFeature);
        }
    });
}

boost::future<void> GdalVectorWriter::close() {
    return boost::async(boost::launch::async, [this]() {
        m_dataSource.reset();
        BOOST_LOG_TRIVIAL(info) << "GDAL Vector file closed.";
    });
}

boost::future<std::vector<std::string>> GdalVectorWriter::write(
    std::shared_ptr<oscean::core_services::IDataReader> reader,
    const core_services::output::OutputRequest& request) {
    
    return boost::async(boost::launch::async, [this, reader, request]() -> std::vector<std::string> {
        std::string extension = ".shp";
         if(request.format != "ESRI Shapefile" && request.format != "shapefile" && request.format != "shp") {
            // map format to extension
        }
        std::string baseFilename = request.filenameTemplate ? *request.filenameTemplate : "output";
        std::string targetDir = request.targetDirectory ? *request.targetDirectory : ".";
        std::string outputPath = targetDir + "/" + baseFilename + extension;

        auto featureCollection = reader->readFeatureCollection();

        this->open(outputPath, request).get();
        this->writeChunk(featureCollection).get();
        this->close().get();

        return std::vector<std::string>{outputPath};
    });
}

} // namespace internal
} // namespace output
} // namespace oscean 