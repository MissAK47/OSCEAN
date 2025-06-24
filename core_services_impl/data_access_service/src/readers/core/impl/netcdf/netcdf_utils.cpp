/**
 * @file netcdf_utils.cpp
 * @brief NetCDF通用工具函数实现
 */

#include "netcdf_utils.h"
#include "common_utils/utilities/logging_utils.h"
#include <netcdf.h>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <limits>
#include <regex>

namespace oscean::core_services::data_access::readers::impl::netcdf::NetCDFUtils {

// ===================================================================================
// Implementation
// ===================================================================================

bool checkNetCDFError(int status, const std::string& context) {
    if (status != NC_NOERR) {
        LOG_ERROR("NetCDF Error in [{}]: {}", context, nc_strerror(status));
        return false;
    }
    return true;
}

oscean::core_services::DataType convertNetCDFType(int ncType) {
    switch (ncType) {
        case NC_BYTE:   return oscean::core_services::DataType::Byte;
        case NC_UBYTE:  return oscean::core_services::DataType::UByte;
        case NC_SHORT:  return oscean::core_services::DataType::Int16;
        case NC_USHORT: return oscean::core_services::DataType::UInt16;
        case NC_INT:    return oscean::core_services::DataType::Int32;
        case NC_UINT:   return oscean::core_services::DataType::UInt32;
        case NC_INT64:  return oscean::core_services::DataType::Int64;
        case NC_UINT64: return oscean::core_services::DataType::UInt64;
        case NC_FLOAT:  return oscean::core_services::DataType::Float32;
        case NC_DOUBLE: return oscean::core_services::DataType::Float64;
        case NC_STRING: return oscean::core_services::DataType::String;
        case NC_CHAR:   return oscean::core_services::DataType::String;
        default:
            LOG_WARN("Unknown NetCDF data type: {}. Defaulting to Float64.", ncType);
            return oscean::core_services::DataType::Float64;
    }
}

std::string readStringAttribute(int ncid, int varid, const std::string& attName, const std::string& defaultValue) {
    nc_type att_type;
    size_t att_len;

    if (nc_inq_att(ncid, varid, attName.c_str(), &att_type, &att_len) != NC_NOERR) {
        return defaultValue; 
    }

    if (att_type == NC_STRING) {
        char* stringValue = nullptr;
        if(checkNetCDFError(nc_get_att_string(ncid, varid, attName.c_str(), &stringValue), "readStringAttribute: nc_get_att_string")) {
            std::string result(stringValue);
            nc_free_string(1, &stringValue);
            return result;
        }
        return defaultValue;
    }
    
    if (att_type == NC_CHAR) {
        std::string val(att_len, '\0');
        if (checkNetCDFError(nc_get_att_text(ncid, varid, attName.c_str(), &val[0]), "readStringAttribute: nc_get_att_text")) {
            val.erase(std::find(val.begin(), val.end(), '\0'), val.end());
            return val;
        }
        return defaultValue;
    }
    
    return defaultValue;
}

double readNumericAttribute(int ncid, int varid, const std::string& attName, double defaultValue) {
    nc_type atttype;
    if (nc_inq_atttype(ncid, varid, attName.c_str(), &atttype) != NC_NOERR) {
        return defaultValue;
    }

    switch (atttype) {
        case NC_DOUBLE: {
            double value;
            if (nc_get_att_double(ncid, varid, attName.c_str(), &value) == NC_NOERR) return value;
            break;
        }
        case NC_FLOAT: {
            float value;
            if (nc_get_att_float(ncid, varid, attName.c_str(), &value) == NC_NOERR) return static_cast<double>(value);
            break;
        }
        case NC_INT: {
            int value;
            if (nc_get_att_int(ncid, varid, attName.c_str(), &value) == NC_NOERR) return static_cast<double>(value);
            break;
        }
        // Add other numeric types as needed (short, byte, etc.)
    }
    return defaultValue;
}

bool hasAttribute(int ncid, int varid, const std::string& attName) {
    return nc_inq_attid(ncid, varid, attName.c_str(), nullptr) == NC_NOERR;
}

bool variableExists(int ncid, const std::string& varName) {
    int varid;
    return nc_inq_varid(ncid, varName.c_str(), &varid) == NC_NOERR;
}

int getVariableId(int ncid, const std::string& varName) {
    int varid;
    if (nc_inq_varid(ncid, varName.c_str(), &varid) != NC_NOERR) {
        return -1;
    }
    return varid;
}

int getVarId(int ncid, const std::string& varName) {
    return getVariableId(ncid, varName);  // 保持向后兼容性
}

std::map<std::string, std::string> readGlobalAttributes(int ncid) {
    std::map<std::string, std::string> attributes;
    int natts;
    if (!checkNetCDFError(nc_inq_natts(ncid, &natts), "readGlobalAttributes: nc_inq_natts")) return {};

    for (int i = 0; i < natts; ++i) {
        char name[NC_MAX_NAME + 1] = {0};
        if (checkNetCDFError(nc_inq_attname(ncid, NC_GLOBAL, i, name), "readGlobalAttributes: nc_inq_attname")) {
            attributes[name] = readStringAttribute(ncid, NC_GLOBAL, name, "");
        }
    }
    return attributes;
}

std::vector<double> readVariableDataDouble(int ncid, const std::string& varName) {
    int varid = getVarId(ncid, varName);
    if (varid == -1) return {};

    int ndims;
    nc_inq_varndims(ncid, varid, &ndims);
    
    std::vector<int> dimids(ndims);
    nc_inq_vardimid(ncid, varid, dimids.data());

    size_t total_size = 1;
    for (int dimid : dimids) {
        size_t len;
        nc_inq_dimlen(ncid, dimid, &len);
        total_size *= len;
    }

    if (total_size == 0) return {};

    std::vector<double> data(total_size);
    if (!checkNetCDFError(nc_get_var_double(ncid, varid, data.data()), "readVariableDataDouble: nc_get_var_double")) {
        return {};
    }
    return data;
}

std::vector<oscean::core_services::DimensionDetail> readDimensionDetails(int ncid) {
    std::vector<oscean::core_services::DimensionDetail> dimensions;
    int ndims;
    if (!checkNetCDFError(nc_inq_ndims(ncid, &ndims), "readDimensionDetails: nc_inq_ndims")) return {};

    for (int i = 0; i < ndims; ++i) {
        char name[NC_MAX_NAME + 1] = {0};
        size_t len;
        if (checkNetCDFError(nc_inq_dim(ncid, i, name, &len), "readDimensionDetails: nc_inq_dim")) {
            oscean::core_services::DimensionDetail detail;
            detail.name = name;
            detail.size = len;

            int varid = getVarId(ncid, detail.name);
            if (varid != -1) {
                detail.coordinates = readVariableDataDouble(ncid, detail.name);
                detail.units = readStringAttribute(ncid, varid, "units", "");

                int natts;
                nc_inq_varnatts(ncid, varid, &natts);
                for (int j = 0; j < natts; ++j) {
                    char attname[NC_MAX_NAME + 1] = {0};
                    nc_inq_attname(ncid, varid, j, attname);
                    detail.attributes[attname] = readStringAttribute(ncid, varid, attname, "");
                }
            }
            dimensions.push_back(detail);
        }
    }
    return dimensions;
}

std::vector<oscean::core_services::VariableMeta> readAllVariablesMetadata(int ncid) {
    std::vector<oscean::core_services::VariableMeta> variables;
    int nvars;
    if (!checkNetCDFError(nc_inq_nvars(ncid, &nvars), "readAllVariablesMetadata: nc_inq_nvars")) return {};

    for (int i = 0; i < nvars; ++i) {
        char name[NC_MAX_NAME + 1] = {0};
        if (!checkNetCDFError(nc_inq_varname(ncid, i, name), "readAllVariablesMetadata: nc_inq_varname")) continue;
        
        oscean::core_services::VariableMeta varMeta;
        varMeta.name = name;
        
        int natts;
        nc_inq_varnatts(ncid, i, &natts);
        for (int j = 0; j < natts; ++j) {
            char attname[NC_MAX_NAME + 1] = {0};
            nc_inq_attname(ncid, i, j, attname);
            varMeta.attributes[attname] = readStringAttribute(ncid, i, attname, "");
        }

        if (auto units_it = varMeta.attributes.find("units"); units_it != varMeta.attributes.end()) {
            varMeta.units = units_it->second;
        }
        if (auto long_name_it = varMeta.attributes.find("long_name"); long_name_it != varMeta.attributes.end()) {
            varMeta.description = long_name_it->second;
        } else if (auto std_name_it = varMeta.attributes.find("standard_name"); std_name_it != varMeta.attributes.end()) {
            varMeta.description = std_name_it->second;
        }

        int ndims;
        nc_inq_varndims(ncid, i, &ndims);
        std::vector<int> dimids(ndims);
        nc_inq_vardimid(ncid, i, dimids.data());
        for (int dimid : dimids) {
            char dimname[NC_MAX_NAME + 1] = {0};
            nc_inq_dimname(ncid, dimid, dimname);
            varMeta.dimensionNames.push_back(dimname);
        }
        
        int ncType;
        nc_inq_vartype(ncid, i, &ncType);
        varMeta.dataType = convertNetCDFType(ncType);

        variables.push_back(varMeta);
    }
    return variables;
}

std::string timePointToISOString(const std::chrono::system_clock::time_point& tp) {
    std::time_t time = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

} // namespace oscean::core_services::data_access::readers::impl::netcdf::NetCDFUtils 