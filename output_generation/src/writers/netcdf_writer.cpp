// 强制包含完整的IDataReader定义
#include "core_services/data_access/i_data_reader.h"

#include "writers/netcdf_writer.h"
#include "core_services/common_data_types.h"
#include "core_services/exceptions.h"

// Boost配置必须在boost库包含之前
#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <boost/log/trivial.hpp>
#include <boost/format.hpp>

// NetCDF C库
#include <netcdf.h>

#include <vector>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace oscean {
namespace output {
namespace internal {

NetCdfWriter::NetCdfWriter() = default;

NetCdfWriter::~NetCdfWriter() noexcept {
    try {
        cleanup();
    } catch (...) {
        // 析构函数中不能抛出异常
    }
}

boost::future<void> NetCdfWriter::open(const std::string& path, const core_services::output::OutputRequest& request) {
    return boost::async(boost::launch::async, [this, path, request]() {
        m_filePath = path;
        m_request = request;
        
        try {
            createNetCDFFile(path);
            m_isOpen = true;
            BOOST_LOG_TRIVIAL(info) << "NetCDF file opened: " << path;
        } catch (const std::exception& e) {
            cleanup();
            throw core_services::ServiceException("Failed to open NetCDF file: " + std::string(e.what()));
        }
    });
}

boost::future<void> NetCdfWriter::writeChunk(const boost::variant<std::shared_ptr<core_services::GridData>, std::shared_ptr<core_services::FeatureCollection>>& dataChunk) {
     return boost::async(boost::launch::async, [this, dataChunk]() {
        try {
            // 使用visitor模式处理variant
            struct ChunkWriteVisitor : boost::static_visitor<void> {
                NetCdfWriter* writer;
                
                ChunkWriteVisitor(NetCdfWriter* w) : writer(w) {}
                
                void operator()(std::shared_ptr<core_services::GridData> gridData) {
                    if (!gridData) {
                        throw core_services::ServiceException("Received null GridData in NetCDF writeChunk");
                    }
                    
                    // 如果是第一个chunk，需要定义文件结构
                    if (writer->m_chunksWritten == 0) {
                        writer->defineDimensions(gridData);
                        writer->defineVariable(gridData);
                        writer->writeGlobalAttributes();
                        
                        // 从变量映射中获取变量ID
                        std::string varName = writer->getVariableNameFromGridData(gridData);
                        auto it = writer->m_variableIds.find(varName);
                        if (it != writer->m_variableIds.end()) {
                            writer->writeVariableAttributes(it->second, gridData);
                        }
                        
                        writer->writeCoordinateData(gridData);
                        
                        // 结束定义模式
                        int status = nc_enddef(writer->m_ncid);
                        writer->checkNetCDFError(status, "nc_enddef");
                    }
                    
                    writer->writeGridData(gridData);
                    writer->m_chunksWritten++;
                }
                
                void operator()(std::shared_ptr<core_services::FeatureCollection> features) {
                    throw core_services::ServiceException("NetCDF writer does not support FeatureCollection data");
                }
            };
            
            ChunkWriteVisitor visitor(this);
            boost::apply_visitor(visitor, dataChunk);
            
            BOOST_LOG_TRIVIAL(debug) << "NetCDF chunk " << m_chunksWritten << " written";
            
        } catch (const std::exception& e) {
            throw core_services::ServiceException("Failed to write NetCDF chunk: " + std::string(e.what()));
        }
    });
}

boost::future<void> NetCdfWriter::close() {
    return boost::async(boost::launch::async, [this]() {
        try {
            if (m_ncid >= 0) {
                int status = nc_close(m_ncid);
                checkNetCDFError(status, "nc_close");
                m_ncid = -1;
            }
            m_isOpen = false;
            BOOST_LOG_TRIVIAL(info) << "NetCDF file closed: " << m_filePath;
        } catch (const std::exception& e) {
            throw core_services::ServiceException("Failed to close NetCDF file: " + std::string(e.what()));
        }
    });
}

boost::future<std::vector<std::string>> NetCdfWriter::write(
    std::shared_ptr<oscean::core_services::IDataReader> reader,
    const core_services::output::OutputRequest& request) {
    
    return boost::async(boost::launch::async, [this, reader, request]() -> std::vector<std::string> {
        try {
            if (!reader) {
                throw core_services::ServiceException("NetCdfWriter received a null data reader.");
            }

            // 生成输出文件路径
            std::string extension = ".nc";
        std::string baseFilename = request.filenameTemplate ? *request.filenameTemplate : "output";
        std::string targetDir = request.targetDirectory ? *request.targetDirectory : ".";
            std::string outputPath = targetDir + "/" + baseFilename + extension;

            // 打开文件进行写入
            open(outputPath, request).get();

            // 获取可用的变量名
            auto variableNames = reader->listDataVariableNames();
            if (variableNames.empty()) {
                throw core_services::ServiceException("No variables found in data source");
            }

            // 使用第一个变量或指定的变量
            std::string variableName = variableNames[0];
            
            // 读取数据并写入
            auto gridData = reader->readGridData(variableName);
            
            if (gridData) {
                // 写入数据chunk
                boost::variant<std::shared_ptr<core_services::GridData>, std::shared_ptr<core_services::FeatureCollection>> dataVariant = gridData;
                writeChunk(dataVariant).get();
            } else {
                BOOST_LOG_TRIVIAL(warning) << "No grid data found for variable: " << variableName;
            }

            // 关闭文件
            close().get();

        return std::vector<std::string>{outputPath};
            
        } catch (const std::exception& e) {
            cleanup();
            throw core_services::ServiceException("NetCDF write operation failed: " + std::string(e.what()));
        }
    });
}

// 私有方法实现

void NetCdfWriter::createNetCDFFile(const std::string& path) {
    int status = nc_create(path.c_str(), NC_CLOBBER | NC_NETCDF4, &m_ncid);
    checkNetCDFError(status, "nc_create");
    
    BOOST_LOG_TRIVIAL(info) << "Created NetCDF file: " << path << " (ncid=" << m_ncid << ")";
}

void NetCdfWriter::defineDimensions(std::shared_ptr<core_services::GridData> gridData) {
    if (!gridData) return;
    
    // 从GridData定义中获取维度信息
    const auto& definition = gridData->getDefinition();
    
    // X维度 (longitude/columns)
    if (definition.hasXDimension()) {
        int dimid;
        int status = nc_def_dim(m_ncid, "lon", definition.cols, &dimid);
        checkNetCDFError(status, "nc_def_dim for lon");
        m_dimensionIds["lon"] = dimid;
        m_dimensionSizes["lon"] = definition.cols;
        BOOST_LOG_TRIVIAL(debug) << "Defined dimension: lon (size=" << definition.cols << ", id=" << dimid << ")";
    }
    
    // Y维度 (latitude/rows)
    if (definition.hasYDimension()) {
        int dimid;
        int status = nc_def_dim(m_ncid, "lat", definition.rows, &dimid);
        checkNetCDFError(status, "nc_def_dim for lat");
        m_dimensionIds["lat"] = dimid;
        m_dimensionSizes["lat"] = definition.rows;
        BOOST_LOG_TRIVIAL(debug) << "Defined dimension: lat (size=" << definition.rows << ", id=" << dimid << ")";
    }
    
    // Z维度 (bands/levels)
    size_t bandCount = gridData->getBandCount();
    if (bandCount > 1) {
        int dimid;
        int status = nc_def_dim(m_ncid, "band", bandCount, &dimid);
        checkNetCDFError(status, "nc_def_dim for band");
        m_dimensionIds["band"] = dimid;
        m_dimensionSizes["band"] = bandCount;
        BOOST_LOG_TRIVIAL(debug) << "Defined dimension: band (size=" << bandCount << ", id=" << dimid << ")";
    }
}

void NetCdfWriter::defineVariable(std::shared_ptr<core_services::GridData> gridData) {
    if (!gridData) return;
    
    // 获取维度ID数组 - 按NetCDF约定顺序 (time, level, lat, lon)
    std::vector<int> dimids;
    std::vector<std::string> orderedDimNames;
    
    // 添加band维度（如果存在）
    if (m_dimensionIds.find("band") != m_dimensionIds.end()) {
        dimids.push_back(m_dimensionIds["band"]);
        orderedDimNames.push_back("band");
    }
    
    // 添加lat维度（如果存在）
    if (m_dimensionIds.find("lat") != m_dimensionIds.end()) {
        dimids.push_back(m_dimensionIds["lat"]);
        orderedDimNames.push_back("lat");
    }
    
    // 添加lon维度（如果存在）
    if (m_dimensionIds.find("lon") != m_dimensionIds.end()) {
        dimids.push_back(m_dimensionIds["lon"]);
        orderedDimNames.push_back("lon");
    }
    
    // 生成变量名
    std::string variableName = getVariableNameFromGridData(gridData);
    
    // 定义主数据变量
    nc_type ncType = convertToNetCDFType(gridData->getDataType());
    int varid;
    int status = nc_def_var(m_ncid, variableName.c_str(), ncType, 
                           static_cast<int>(dimids.size()), dimids.data(), &varid);
    checkNetCDFError(status, "nc_def_var for " + variableName);
    
    m_variableIds[variableName] = varid;
    m_currentVariableName = variableName;
    
    // 定义坐标变量
    for (const std::string& dimName : orderedDimNames) {
        if (m_variableIds.find(dimName) == m_variableIds.end()) {
            int coord_varid;
            int coord_dimid = m_dimensionIds[dimName];
            status = nc_def_var(m_ncid, dimName.c_str(), NC_DOUBLE, 1, &coord_dimid, &coord_varid);
            checkNetCDFError(status, "nc_def_var for coordinate " + dimName);
            
            m_variableIds[dimName] = coord_varid;
            
            // 添加坐标变量属性
            std::string axis = getAxisAttribute(dimName);
            if (!axis.empty()) {
                status = nc_put_att_text(m_ncid, coord_varid, "axis", axis.length(), axis.c_str());
                checkNetCDFError(status, "nc_put_att_text axis for " + dimName);
            }
        }
    }
    
    BOOST_LOG_TRIVIAL(info) << "Defined variable: " << variableName << " (id=" << varid << ")";
}

void NetCdfWriter::writeGlobalAttributes() {
    // 写入CF约定属性
    std::string conventions = "CF-1.8";
    int status = nc_put_att_text(m_ncid, NC_GLOBAL, "Conventions", conventions.length(), conventions.c_str());
    checkNetCDFError(status, "nc_put_att_text Conventions");
    
    // 写入创建时间
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S UTC");
    std::string history = "Created by OSCEAN NetCDF Writer on " + ss.str();
    
    status = nc_put_att_text(m_ncid, NC_GLOBAL, "history", history.length(), history.c_str());
    checkNetCDFError(status, "nc_put_att_text history");
    
    // 写入创建工具信息
    std::string source = "OSCEAN Ocean Environment Data Processing System";
    status = nc_put_att_text(m_ncid, NC_GLOBAL, "source", source.length(), source.c_str());
    checkNetCDFError(status, "nc_put_att_text source");
    
    BOOST_LOG_TRIVIAL(debug) << "Wrote global attributes";
}

void NetCdfWriter::writeVariableAttributes(int varid, std::shared_ptr<core_services::GridData> gridData) {
    if (!gridData) return;
    
    // 写入标准名称 - 从GridData元数据中获取
    auto metadataIt = gridData->metadata.find("standard_name");
    if (metadataIt != gridData->metadata.end()) {
        int status = nc_put_att_text(m_ncid, varid, "standard_name", 
                                   metadataIt->second.length(), metadataIt->second.c_str());
        checkNetCDFError(status, "nc_put_att_text standard_name");
    }
    
    // 写入长名称
    metadataIt = gridData->metadata.find("long_name");
    if (metadataIt != gridData->metadata.end()) {
        int status = nc_put_att_text(m_ncid, varid, "long_name", 
                                   metadataIt->second.length(), metadataIt->second.c_str());
        checkNetCDFError(status, "nc_put_att_text long_name");
    }
    
    // 写入单位
    metadataIt = gridData->metadata.find("units");
    if (metadataIt != gridData->metadata.end()) {
        int status = nc_put_att_text(m_ncid, varid, "units", 
                                   metadataIt->second.length(), metadataIt->second.c_str());
        checkNetCDFError(status, "nc_put_att_text units");
    }
    
    // 写入填充值（如果有）
    if (gridData->hasNoDataValue()) {
        // 注意：GridData可能没有直接的getFillValue方法，我们需要从元数据中获取
        metadataIt = gridData->metadata.find("_FillValue");
        if (metadataIt != gridData->metadata.end()) {
            try {
                double fillValue = std::stod(metadataIt->second);
                int status = nc_put_att_double(m_ncid, varid, "_FillValue", NC_DOUBLE, 1, &fillValue);
                checkNetCDFError(status, "nc_put_att_double _FillValue");
            } catch (const std::exception&) {
                // 如果转换失败，忽略填充值
            }
        }
    }
    
    std::string varName = getVariableNameFromGridData(gridData);
    BOOST_LOG_TRIVIAL(debug) << "Wrote variable attributes for: " << varName;
}

void NetCdfWriter::writeCoordinateData(std::shared_ptr<core_services::GridData> gridData) {
    if (!gridData) return;
    
    const auto& definition = gridData->getDefinition();
    
    // 写入经度坐标
    if (m_variableIds.find("lon") != m_variableIds.end()) {
        int varid = m_variableIds["lon"];
        std::vector<double> lonValues = gridData->getLonValues();
        if (!lonValues.empty()) {
            int status = nc_put_var_double(m_ncid, varid, lonValues.data());
            checkNetCDFError(status, "nc_put_var_double for coordinate lon");
            BOOST_LOG_TRIVIAL(debug) << "Wrote coordinate data for: lon (" << lonValues.size() << " values)";
        }
    }
    
    // 写入纬度坐标
    if (m_variableIds.find("lat") != m_variableIds.end()) {
        int varid = m_variableIds["lat"];
        std::vector<double> latValues = gridData->getLatValues();
        if (!latValues.empty()) {
            int status = nc_put_var_double(m_ncid, varid, latValues.data());
            checkNetCDFError(status, "nc_put_var_double for coordinate lat");
            BOOST_LOG_TRIVIAL(debug) << "Wrote coordinate data for: lat (" << latValues.size() << " values)";
        }
    }
    
    // 写入band坐标（如果存在）
    if (m_variableIds.find("band") != m_variableIds.end()) {
        int varid = m_variableIds["band"];
        const auto& bandCoords = gridData->getBandCoordinates();
        if (!bandCoords.empty()) {
            int status = nc_put_var_double(m_ncid, varid, bandCoords.data());
            checkNetCDFError(status, "nc_put_var_double for coordinate band");
            BOOST_LOG_TRIVIAL(debug) << "Wrote coordinate data for: band (" << bandCoords.size() << " values)";
        }
    }
}

void NetCdfWriter::writeGridData(std::shared_ptr<core_services::GridData> gridData) {
    if (!gridData) return;
    
    std::string variableName = getVariableNameFromGridData(gridData);
    auto it = m_variableIds.find(variableName);
    if (it == m_variableIds.end()) {
        throw core_services::ServiceException("Variable not defined: " + variableName);
    }
    
    int varid = it->second;
    
    // 获取数据指针
    const void* dataPtr = gridData->getDataPtr();
    if (!dataPtr) {
        throw core_services::ServiceException("GridData contains no data");
    }
    
    // 根据数据类型写入数据
    int status;
    switch (gridData->getDataType()) {
        case core_services::DataType::Float64:
            status = nc_put_var_double(m_ncid, varid, static_cast<const double*>(dataPtr));
            break;
        case core_services::DataType::Float32:
            status = nc_put_var_float(m_ncid, varid, static_cast<const float*>(dataPtr));
            break;
        case core_services::DataType::Int32:
            status = nc_put_var_int(m_ncid, varid, static_cast<const int*>(dataPtr));
            break;
        default:
            // 默认转换为double
            status = nc_put_var_double(m_ncid, varid, static_cast<const double*>(dataPtr));
            break;
    }
    
    checkNetCDFError(status, "nc_put_var for " + variableName);
    
    BOOST_LOG_TRIVIAL(info) << "Wrote grid data for variable: " << variableName 
                           << " (" << gridData->getDataSizeBytes() << " bytes)";
}

nc_type NetCdfWriter::convertToNetCDFType(core_services::DataType dataType) {
    switch (dataType) {
        case core_services::DataType::Byte:     return NC_BYTE;
        case core_services::DataType::UByte:    return NC_UBYTE;
        case core_services::DataType::Int16:    return NC_SHORT;
        case core_services::DataType::UInt16:   return NC_USHORT;
        case core_services::DataType::Int32:    return NC_INT;
        case core_services::DataType::UInt32:   return NC_UINT;
        case core_services::DataType::Int64:    return NC_INT64;
        case core_services::DataType::UInt64:   return NC_UINT64;
        case core_services::DataType::Float32:  return NC_FLOAT;
        case core_services::DataType::Float64:  return NC_DOUBLE;
        case core_services::DataType::String:   return NC_STRING;
        default:
            return NC_DOUBLE; // 默认使用double
    }
}

std::string NetCdfWriter::getAxisAttribute(const std::string& dimensionName) {
    std::string lowerDim = dimensionName;
    std::transform(lowerDim.begin(), lowerDim.end(), lowerDim.begin(), ::tolower);
    
    if (lowerDim.find("lon") != std::string::npos || lowerDim.find("x") != std::string::npos) {
        return "X";
    } else if (lowerDim.find("lat") != std::string::npos || lowerDim.find("y") != std::string::npos) {
        return "Y";
    } else if (lowerDim.find("time") != std::string::npos || lowerDim.find("t") != std::string::npos) {
        return "T";
    } else if (lowerDim.find("depth") != std::string::npos || lowerDim.find("level") != std::string::npos || 
               lowerDim.find("band") != std::string::npos || lowerDim.find("z") != std::string::npos) {
        return "Z";
    }
    return "";
}

std::string NetCdfWriter::generateStandardName(const std::string& variableName) {
    // 简单的标准名称生成规则
    std::string lowerVar = variableName;
    std::transform(lowerVar.begin(), lowerVar.end(), lowerVar.begin(), ::tolower);
    
    if (lowerVar.find("temp") != std::string::npos) {
        return "sea_water_temperature";
    } else if (lowerVar.find("sal") != std::string::npos) {
        return "sea_water_salinity";
    } else if (lowerVar.find("depth") != std::string::npos) {
        return "depth";
    }
    
    return variableName; // 返回原名称作为fallback
}

std::string NetCdfWriter::getVariableNameFromGridData(std::shared_ptr<core_services::GridData> gridData) {
    if (!gridData) return "data";
    
    // 从元数据中获取变量名
    auto metadataIt = gridData->metadata.find("variable_name");
    if (metadataIt != gridData->metadata.end()) {
        return metadataIt->second;
    }
    
    // 或从其他可能的字段获取
    metadataIt = gridData->metadata.find("name");
    if (metadataIt != gridData->metadata.end()) {
        return metadataIt->second;
    }
    
    // 默认名称
    return "data";
}

void NetCdfWriter::checkNetCDFError(int status, const std::string& operation) {
    if (status != NC_NOERR) {
        std::string errorMsg = "NetCDF error in " + operation + ": " + nc_strerror(status);
        BOOST_LOG_TRIVIAL(error) << errorMsg;
        throw std::runtime_error(errorMsg);
    }
}

void NetCdfWriter::cleanup() {
    if (m_ncid >= 0) {
        nc_close(m_ncid);
        m_ncid = -1;
    }
    m_isOpen = false;
    m_dimensionIds.clear();
    m_variableIds.clear();
    m_dimensionSizes.clear();
    m_chunksWritten = 0;
    m_currentVariableName.clear();
}

} // namespace internal
} // namespace output
} // namespace oscean