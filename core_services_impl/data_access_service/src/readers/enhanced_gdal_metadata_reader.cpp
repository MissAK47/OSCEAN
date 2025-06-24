#include "../include/readers/enhanced_gdal_metadata_reader.h"
#include <algorithm>
#include <filesystem>
#include <regex>
#include <mutex>

// GDAL头文件包含（这里使用条件编译，如果没有GDAL则提供模拟实现）
#ifdef GDAL_AVAILABLE
#include <gdal.h>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <ogr_spatialref.h>
#include <cpl_conv.h>
#else
// 模拟GDAL类型定义
typedef void* GDALDatasetH;
typedef void* OGRDataSourceH;
typedef void* OGRSpatialReferenceH;
typedef int GDALDataType;
#define GDT_Unknown 0
#define GDT_Byte 1
#define GDT_UInt16 2
#define GDT_Int16 3
#define GDT_UInt32 4
#define GDT_Int32 5
#define GDT_Float32 6
#define GDT_Float64 7
#define CE_None 0
#define CE_Warning 1
#define CE_Failure 2
#define CE_Fatal 3
#endif

namespace oscean::core_services::data_access::readers {

// 静态成员初始化
bool EnhancedGDALMetadataReader::gdalInitialized_ = false;
std::mutex EnhancedGDALMetadataReader::gdalMutex_;

EnhancedGDALMetadataReader::EnhancedGDALMetadataReader() {
    // 移除 GDALAllRegister() 调用。该操作现在由GdalGlobalInitializer在程序启动时统一管理。
}

EnhancedGDALMetadataReader::~EnhancedGDALMetadataReader() = default;

bool EnhancedGDALMetadataReader::initializeGDAL() {
    std::lock_guard<std::mutex> lock(gdalMutex_);
    
    if (gdalInitialized_) {
        return true;
    }
    
    try {
#ifdef GDAL_AVAILABLE
        // 移除GDALAllRegister()调用 - 现在由GdalGlobalInitializer统一管理
        // GDALAllRegister(); // ❌ 已移除
        // OGRRegisterAll();  // ❌ 已移除
        
        // 检查GDAL是否已经初始化
        if (GDALGetDriverCount() > 0) {
            OSCEAN_LOG_INFO("EnhancedGDALMetadataReader", "GDAL库已由全局初始化器初始化，驱动数量: {}", GDALGetDriverCount());
            gdalInitialized_ = true;
            return true;
        } else {
            OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "GDAL库未初始化，请检查GdalGlobalInitializer是否已调用");
            return false;
        }
#else
        OSCEAN_LOG_WARN("EnhancedGDALMetadataReader", "GDAL库未可用，使用模拟实现");
        gdalInitialized_ = true;
        return false;
#endif
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "GDAL库初始化检查失败: {}", e.what());
        return false;
    }
}

void EnhancedGDALMetadataReader::cleanupGDAL() {
    std::lock_guard<std::mutex> lock(gdalMutex_);
    
    if (gdalInitialized_) {
#ifdef GDAL_AVAILABLE
        OGRCleanupAll();
        GDALDestroyDriverManager();
#endif
        gdalInitialized_ = false;
        OSCEAN_LOG_INFO("EnhancedGDALMetadataReader", "GDAL库清理完成");
    }
}

boost::future<boost::optional<oscean::core_services::FileMetadata>> 
EnhancedGDALMetadataReader::extractStandardMetadataAsync(const std::string& filePath) {
    
    return boost::async(boost::launch::async, [this, filePath]() -> boost::optional<oscean::core_services::FileMetadata> {
        try {
            if (!supportsFile(filePath)) {
                OSCEAN_LOG_WARN("EnhancedGDALMetadataReader", "文件格式不支持: {}", filePath);
                return boost::none;
            }
            
            // 判断是栅格还是矢量数据
            if (isRasterFile(filePath)) {
                auto result = extractRasterMetadataAsync(filePath).get();
                if (result) {
                    OSCEAN_LOG_INFO("EnhancedGDALMetadataReader", "成功提取栅格元数据: {}", filePath);
                }
                return result;
            } else if (isVectorFile(filePath)) {
                auto result = extractVectorMetadataAsync(filePath).get();
                if (result) {
                    OSCEAN_LOG_INFO("EnhancedGDALMetadataReader", "成功提取矢量元数据: {}", filePath);
                }
                return result;
            } else {
                OSCEAN_LOG_WARN("EnhancedGDALMetadataReader", "无法确定文件类型: {}", filePath);
                return boost::none;
            }
            
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "元数据提取失败: {} - {}", filePath, e.what());
            return boost::none;
        }
    });
}

boost::future<boost::optional<oscean::core_services::FileMetadata>> 
EnhancedGDALMetadataReader::extractRasterMetadataAsync(const std::string& filePath) {
    
    return boost::async(boost::launch::async, [this, filePath]() -> boost::optional<oscean::core_services::FileMetadata> {
        try {
#ifdef GDAL_AVAILABLE
            // 打开栅格数据集
            GDALDataset* dataset = static_cast<GDALDataset*>(
                GDALOpen(filePath.c_str(), GA_ReadOnly)
            );
            
            if (!dataset) {
                OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "无法打开栅格文件: {} - {}", 
                    filePath, getLastGDALError());
                return boost::none;
            }
            
            // 创建元数据对象
            oscean::core_services::FileMetadata metadata;
            metadata.filePath = filePath;
            metadata.fileName = std::filesystem::path(filePath).filename().string();
            metadata.format = "GDAL/" + std::string(dataset->GetDriverName());
            
            // 获取栅格基本信息
            int width = dataset->GetRasterXSize();
            int height = dataset->GetRasterYSize();
            int bandCount = dataset->GetRasterCount();
            
            // 提取地理变换信息
            double geoTransform[6];
            if (dataset->GetGeoTransform(geoTransform) == CE_None) {
                metadata.spatialCoverage = calculateBoundingBoxFromGeoTransform(
                    geoTransform, width, height
                );
            }
            
            // 提取空间参考信息
            const OGRSpatialReference* srs = dataset->GetSpatialRef();
            if (srs) {
                metadata.crs = extractCRSInfo(const_cast<OGRSpatialReference*>(srs));
            }
            
            // 提取波段信息
            metadata.variables = extractRasterBandInfo(dataset);
            
            // 创建维度信息
            oscean::core_services::DimensionDetail xDim;
            xDim.name = "x";
            xDim.size = width;
            xDim.dataType = oscean::core_services::DataType::Float64;
            
            oscean::core_services::DimensionDetail yDim;
            yDim.name = "y";
            yDim.size = height;
            yDim.dataType = oscean::core_services::DataType::Float64;
            
            oscean::core_services::DimensionDetail bandDim;
            bandDim.name = "band";
            bandDim.size = bandCount;
            bandDim.dataType = oscean::core_services::DataType::Int32;
            
            metadata.dimensions = {xDim, yDim, bandDim};
            
            // 提取元数据属性
            metadata.metadata = extractMetadataAttributes(dataset);
            
            // 设置文件大小
            try {
                metadata.fileSizeBytes = std::filesystem::file_size(filePath);
            } catch (...) {
                metadata.fileSizeBytes = 0;
            }
            
            // 清理资源
            GDALClose(dataset);
            
            return metadata;
            
#else
            // 模拟实现
            OSCEAN_LOG_WARN("EnhancedGDALMetadataReader", "GDAL不可用，返回基础元数据: {}", filePath);
            
            oscean::core_services::FileMetadata metadata;
            metadata.filePath = filePath;
            metadata.fileName = std::filesystem::path(filePath).filename().string();
            metadata.format = "Unknown_Raster";
            
            // 设置默认边界框
            metadata.spatialCoverage.minX = -180.0;
            metadata.spatialCoverage.maxX = 180.0;
            metadata.spatialCoverage.minY = -90.0;
            metadata.spatialCoverage.maxY = 90.0;
            
            // 设置默认CRS
            metadata.crs.id = "EPSG:4326";
            metadata.crs.epsgCode = 4326;
            
            return metadata;
#endif
            
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "栅格元数据提取异常: {}", e.what());
            return boost::none;
        }
    });
}

boost::future<boost::optional<oscean::core_services::FileMetadata>> 
EnhancedGDALMetadataReader::extractVectorMetadataAsync(const std::string& filePath) {
    
    return boost::async(boost::launch::async, [this, filePath]() -> boost::optional<oscean::core_services::FileMetadata> {
        try {
#ifdef GDAL_AVAILABLE
            // 打开矢量数据源
            GDALDataset* dataset = static_cast<GDALDataset*>(
                GDALOpenEx(filePath.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)
            );
            
            if (!dataset) {
                OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "无法打开矢量文件: {} - {}", 
                    filePath, getLastGDALError());
                return boost::none;
            }
            
            // 创建元数据对象
            oscean::core_services::FileMetadata metadata;
            metadata.filePath = filePath;
            metadata.fileName = std::filesystem::path(filePath).filename().string();
            metadata.format = "OGR/" + std::string(dataset->GetDriverName());
            
            // 获取图层数量
            int layerCount = dataset->GetLayerCount();
            if (layerCount > 0) {
                // 获取第一个图层的信息
                OGRLayer* layer = dataset->GetLayer(0);
                if (layer) {
                    // 获取空间范围
                    OGREnvelope envelope;
                    if (layer->GetExtent(&envelope) == OGRERR_NONE) {
                        metadata.spatialCoverage.minX = envelope.MinX;
                        metadata.spatialCoverage.maxX = envelope.MaxX;
                        metadata.spatialCoverage.minY = envelope.MinY;
                        metadata.spatialCoverage.maxY = envelope.MaxY;
                    }
                    
                    // 获取空间参考
                    OGRSpatialReference* srs = layer->GetSpatialRef();
                    if (srs) {
                        metadata.crs = extractCRSInfo(srs);
                    }
                    
                    // 获取要素数量
                    metadata.metadata["feature_count"] = std::to_string(layer->GetFeatureCount());
                    metadata.metadata["layer_count"] = std::to_string(layerCount);
                }
            }
            
            // 提取图层信息作为变量
            metadata.variables = extractVectorLayerInfo(dataset);
            
            // 提取元数据属性
            auto additionalMetadata = extractMetadataAttributes(dataset);
            metadata.metadata.insert(additionalMetadata.begin(), additionalMetadata.end());
            
            // 设置文件大小
            try {
                metadata.fileSizeBytes = std::filesystem::file_size(filePath);
            } catch (...) {
                metadata.fileSizeBytes = 0;
            }
            
            // 清理资源
            GDALClose(dataset);
            
            return metadata;
            
#else
            // 模拟实现
            OSCEAN_LOG_WARN("EnhancedGDALMetadataReader", "GDAL不可用，返回基础矢量元数据: {}", filePath);
            
            oscean::core_services::FileMetadata metadata;
            metadata.filePath = filePath;
            metadata.fileName = std::filesystem::path(filePath).filename().string();
            metadata.format = "Unknown_Vector";
            
            // 设置默认边界框
            metadata.spatialCoverage.minX = -180.0;
            metadata.spatialCoverage.maxX = 180.0;
            metadata.spatialCoverage.minY = -90.0;
            metadata.spatialCoverage.maxY = 90.0;
            
            // 设置默认CRS
            metadata.crs.id = "EPSG:4326";
            metadata.crs.epsgCode = 4326;
            
            return metadata;
#endif
            
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "矢量元数据提取异常: {}", e.what());
            return boost::none;
        }
    });
}

ReaderCapabilities EnhancedGDALMetadataReader::getCapabilities() const {
    ReaderCapabilities caps;
    caps.readerName = "EnhancedGDALMetadataReader";
    caps.version = "1.0.0";
    caps.supportedFormats = getSupportedRasterExtensions();
    
    // 添加矢量格式
    auto vectorFormats = getSupportedVectorExtensions();
    caps.supportedFormats.insert(caps.supportedFormats.end(), 
        vectorFormats.begin(), vectorFormats.end());
    
    caps.capabilities.push_back("raster_metadata_extraction");
    caps.capabilities.push_back("vector_metadata_extraction");
    caps.capabilities.push_back("crs_information");
    caps.capabilities.push_back("spatial_extent");
    caps.capabilities.push_back("band_information");
    caps.capabilities.push_back("layer_information");
    
    return caps;
}

bool EnhancedGDALMetadataReader::supportsFile(const std::string& filePath) const {
    try {
        std::filesystem::path path(filePath);
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        // 检查栅格格式
        auto rasterExts = getSupportedRasterExtensions();
        if (std::find(rasterExts.begin(), rasterExts.end(), extension) != rasterExts.end()) {
            return true;
        }
        
        // 检查矢量格式
        auto vectorExts = getSupportedVectorExtensions();
        if (std::find(vectorExts.begin(), vectorExts.end(), extension) != vectorExts.end()) {
            return true;
        }
        
        return false;
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "文件支持检查失败: {}", e.what());
        return false;
    }
}

oscean::core_services::CRSInfo EnhancedGDALMetadataReader::extractCRSInfo(void* spatialRef) {
    oscean::core_services::CRSInfo crsInfo;
    
    try {
#ifdef GDAL_AVAILABLE
        OGRSpatialReference* srs = static_cast<OGRSpatialReference*>(spatialRef);
        if (!srs) return crsInfo;
        
        // 获取EPSG代码
        const char* authorityCode = srs->GetAuthorityCode(nullptr);
        if (authorityCode) {
            crsInfo.epsgCode = std::stoi(authorityCode);
            crsInfo.id = "EPSG:" + std::string(authorityCode);
            crsInfo.authority = "EPSG";
            crsInfo.code = authorityCode;
        }
        
        // 获取WKT
        char* wktString = nullptr;
        if (srs->exportToWkt(&wktString) == OGRERR_NONE && wktString) {
            crsInfo.wkt = wktString;
            CPLFree(wktString);
        }
        
        // 获取PROJ4
        char* proj4String = nullptr;
        if (srs->exportToProj4(&proj4String) == OGRERR_NONE && proj4String) {
            crsInfo.proj4text = proj4String;
            crsInfo.projString = proj4String;
            CPLFree(proj4String);
        }
        
        // 判断是否为地理坐标系
        crsInfo.isGeographic = srs->IsGeographic();
        
#endif
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "CRS信息提取失败: {}", e.what());
    }
    
    return crsInfo;
}

GDALTypeMapping EnhancedGDALMetadataReader::mapGDALDataType(int gdalType) {
    GDALTypeMapping mapping;
    
    switch (gdalType) {
        case GDT_Byte:
            mapping.osceanType = oscean::core_services::DataType::UInt8;
            mapping.typeString = "uint8";
            mapping.sizeInBytes = 1;
            mapping.isSigned = false;
            break;
        case GDT_UInt16:
            mapping.osceanType = oscean::core_services::DataType::UInt16;
            mapping.typeString = "uint16";
            mapping.sizeInBytes = 2;
            mapping.isSigned = false;
            break;
        case GDT_Int16:
            mapping.osceanType = oscean::core_services::DataType::Int16;
            mapping.typeString = "int16";
            mapping.sizeInBytes = 2;
            mapping.isSigned = true;
            break;
        case GDT_UInt32:
            mapping.osceanType = oscean::core_services::DataType::UInt32;
            mapping.typeString = "uint32";
            mapping.sizeInBytes = 4;
            mapping.isSigned = false;
            break;
        case GDT_Int32:
            mapping.osceanType = oscean::core_services::DataType::Int32;
            mapping.typeString = "int32";
            mapping.sizeInBytes = 4;
            mapping.isSigned = true;
            break;
        case GDT_Float32:
            mapping.osceanType = oscean::core_services::DataType::Float32;
            mapping.typeString = "float32";
            mapping.sizeInBytes = 4;
            mapping.isSigned = true;
            break;
        case GDT_Float64:
            mapping.osceanType = oscean::core_services::DataType::Float64;
            mapping.typeString = "float64";
            mapping.sizeInBytes = 8;
            mapping.isSigned = true;
            break;
        default:
            mapping.osceanType = oscean::core_services::DataType::Unknown;
            mapping.typeString = "unknown";
            mapping.sizeInBytes = 0;
            mapping.isSigned = false;
            break;
    }
    
    return mapping;
}

std::vector<oscean::core_services::VariableMeta> 
EnhancedGDALMetadataReader::extractRasterBandInfo(void* dataset) {
    std::vector<oscean::core_services::VariableMeta> variables;
    
    try {
#ifdef GDAL_AVAILABLE
        GDALDataset* gdalDataset = static_cast<GDALDataset*>(dataset);
        if (!gdalDataset) return variables;
        
        int bandCount = gdalDataset->GetRasterCount();
        for (int i = 1; i <= bandCount; ++i) {
            GDALRasterBand* band = gdalDataset->GetRasterBand(i);
            if (!band) continue;
            
            oscean::core_services::VariableMeta variable;
            variable.name = "Band_" + std::to_string(i);
            
            // 获取波段描述
            const char* description = band->GetDescription();
            if (description && strlen(description) > 0) {
                variable.longName = std::string(description);
            }
            
            // 获取数据类型
            GDALDataType bandType = band->GetRasterDataType();
            auto typeMapping = mapGDALDataType(bandType);
            variable.dataType = typeMapping.osceanType;
            variable.dataTypeString = typeMapping.typeString;
            
            // 获取无数据值
            int hasNoData;
            double noDataValue = band->GetNoDataValue(&hasNoData);
            if (hasNoData) {
                variable.attributes["_FillValue"] = std::to_string(noDataValue);
                variable.attributes["missing_value"] = std::to_string(noDataValue);
            }
            
            // 获取统计信息
            double min, max, mean, stddev;
            if (band->GetStatistics(TRUE, TRUE, &min, &max, &mean, &stddev) == CE_None) {
                variable.attributes["actual_min"] = std::to_string(min);
                variable.attributes["actual_max"] = std::to_string(max);
                variable.attributes["mean"] = std::to_string(mean);
                variable.attributes["std_dev"] = std::to_string(stddev);
            }
            
            // 获取单位信息
            const char* units = band->GetUnitType();
            if (units && strlen(units) > 0) {
                variable.units = std::string(units);
            }
            
            // 设置维度
            variable.dimensions = {"y", "x"};
            
            variables.push_back(variable);
        }
#endif
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "栅格波段信息提取失败: {}", e.what());
    }
    
    return variables;
}

std::vector<oscean::core_services::VariableMeta> 
EnhancedGDALMetadataReader::extractVectorLayerInfo(void* dataset) {
    std::vector<oscean::core_services::VariableMeta> variables;
    
    try {
#ifdef GDAL_AVAILABLE
        GDALDataset* gdalDataset = static_cast<GDALDataset*>(dataset);
        if (!gdalDataset) return variables;
        
        int layerCount = gdalDataset->GetLayerCount();
        for (int i = 0; i < layerCount; ++i) {
            OGRLayer* layer = gdalDataset->GetLayer(i);
            if (!layer) continue;
            
            oscean::core_services::VariableMeta variable;
            variable.name = layer->GetName();
            variable.dataType = oscean::core_services::DataType::Geometry;
            variable.dataTypeString = "geometry";
            
            // 获取几何类型
            OGRwkbGeometryType geomType = layer->GetGeomType();
            variable.attributes["geometry_type"] = std::to_string(geomType);
            
            // 获取要素数量
            variable.attributes["feature_count"] = std::to_string(layer->GetFeatureCount());
            
            // 获取字段信息
            OGRFeatureDefn* layerDefn = layer->GetLayerDefn();
            if (layerDefn) {
                int fieldCount = layerDefn->GetFieldCount();
                variable.attributes["field_count"] = std::to_string(fieldCount);
                
                // 记录字段名称
                std::string fieldNames;
                for (int j = 0; j < fieldCount; ++j) {
                    OGRFieldDefn* fieldDefn = layerDefn->GetFieldDefn(j);
                    if (fieldDefn) {
                        if (!fieldNames.empty()) fieldNames += ", ";
                        fieldNames += fieldDefn->GetNameRef();
                    }
                }
                if (!fieldNames.empty()) {
                    variable.attributes["field_names"] = fieldNames;
                }
            }
            
            variables.push_back(variable);
        }
#endif
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "矢量图层信息提取失败: {}", e.what());
    }
    
    return variables;
}

oscean::core_services::BoundingBox 
EnhancedGDALMetadataReader::calculateBoundingBoxFromGeoTransform(
    const double* geoTransform, int width, int height) {
    
    oscean::core_services::BoundingBox bbox;
    
    if (!geoTransform) {
        // 设置默认值
        bbox.minX = -180.0;
        bbox.maxX = 180.0;
        bbox.minY = -90.0;
        bbox.maxY = 90.0;
        return bbox;
    }
    
    // 计算四个角点的坐标
    double minX = geoTransform[0];
    double maxY = geoTransform[3];
    double maxX = geoTransform[0] + width * geoTransform[1] + height * geoTransform[2];
    double minY = geoTransform[3] + width * geoTransform[4] + height * geoTransform[5];
    
    bbox.minX = std::min(minX, maxX);
    bbox.maxX = std::max(minX, maxX);
    bbox.minY = std::min(minY, maxY);
    bbox.maxY = std::max(minY, maxY);
    
    return bbox;
}

std::map<std::string, std::string> 
EnhancedGDALMetadataReader::extractMetadataAttributes(void* gdalObject) {
    std::map<std::string, std::string> attributes;
    
    try {
#ifdef GDAL_AVAILABLE
        GDALMajorObject* majorObject = static_cast<GDALMajorObject*>(gdalObject);
        if (!majorObject) return attributes;
        
        char** metadata = majorObject->GetMetadata();
        if (metadata) {
            for (int i = 0; metadata[i] != nullptr; ++i) {
                std::string metaItem(metadata[i]);
                size_t pos = metaItem.find('=');
                if (pos != std::string::npos) {
                    std::string key = metaItem.substr(0, pos);
                    std::string value = metaItem.substr(pos + 1);
                    attributes[key] = value;
                }
            }
        }
#endif
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "元数据属性提取失败: {}", e.what());
    }
    
    return attributes;
}

bool EnhancedGDALMetadataReader::isRasterFile(const std::string& filePath) {
    try {
        std::filesystem::path path(filePath);
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        auto rasterExts = getSupportedRasterExtensions();
        return std::find(rasterExts.begin(), rasterExts.end(), extension) != rasterExts.end();
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "栅格文件检测失败: {}", e.what());
        return false;
    }
}

bool EnhancedGDALMetadataReader::isVectorFile(const std::string& filePath) {
    try {
        std::filesystem::path path(filePath);
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        auto vectorExts = getSupportedVectorExtensions();
        return std::find(vectorExts.begin(), vectorExts.end(), extension) != vectorExts.end();
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("EnhancedGDALMetadataReader", "矢量文件检测失败: {}", e.what());
        return false;
    }
}

std::vector<std::string> EnhancedGDALMetadataReader::getSupportedRasterExtensions() {
    return {
        ".tif", ".tiff", ".geotiff",    // GeoTIFF
        ".img",                         // ERDAS Imagine
        ".ecw",                         // ECW
        ".jp2",                         // JPEG 2000
        ".sid",                         // MrSID
        ".hdf", ".h4",                  // HDF4
        ".hdf5", ".h5",                 // HDF5
        ".nc", ".netcdf",               // NetCDF
        ".grb", ".grib", ".grib2",      // GRIB
        ".asc",                         // ASCII Grid
        ".dem",                         // USGS DEM
        ".bil", ".bip", ".bsq",         // ENVI
        ".rst",                         // Idrisi
        ".pix"                          // PCI Geomatics
    };
}

std::vector<std::string> EnhancedGDALMetadataReader::getSupportedVectorExtensions() {
    return {
        ".shp",                         // Shapefile
        ".kml", ".kmz",                 // KML
        ".gpx",                         // GPX
        ".json", ".geojson",            // GeoJSON
        ".gml",                         // GML
        ".dxf",                         // DXF
        ".dwg",                         // DWG
        ".tab",                         // MapInfo TAB
        ".mif",                         // MapInfo MIF
        ".sqlite", ".db",               // SQLite
        ".gpkg"                         // GeoPackage
    };
}

bool EnhancedGDALMetadataReader::isValidGDALObject(void* gdalObject) {
    return gdalObject != nullptr;
}

std::string EnhancedGDALMetadataReader::getLastGDALError() {
    try {
#ifdef GDAL_AVAILABLE
        const char* errorMsg = CPLGetLastErrorMsg();
        return errorMsg ? std::string(errorMsg) : "未知GDAL错误";
#else
        return "GDAL不可用";
#endif
    } catch (...) {
        return "获取GDAL错误信息失败";
    }
}

} // namespace oscean::core_services::data_access::readers 