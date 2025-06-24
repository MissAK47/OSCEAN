/**
 * @file netcdf_variable_processor.cpp
 * @brief NetCDF变量专用处理器实现
 */

#include "netcdf_variable_processor.h"
#include "netcdf_utils.h"
#include "common_utils/utilities/logging_utils.h"
#include <netcdf.h>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace oscean::core_services::data_access::readers::impl::netcdf {

// =============================================================================
// Constructor & Destructor
// =============================================================================

NetCDFVariableProcessor::NetCDFVariableProcessor(ncid_t ncid) : ncid_(ncid) {
    if (ncid_ < 0) {
        throw std::invalid_argument("NetCDFVariableProcessor: Provided NetCDF ID is invalid.");
    }
}

// =============================================================================
// Variable Information Retrieval
// =============================================================================

std::vector<std::string> NetCDFVariableProcessor::getVariableNames() const {
    int nvars;
    if (nc_inq_nvars(ncid_, &nvars) != NC_NOERR) {
        LOG_ERROR("nc_inq_nvars failed for ncid: {}", ncid_);
        return {};
    }

    std::vector<std::string> names;
    names.reserve(nvars);
    for (int i = 0; i < nvars; ++i) {
        char name[NC_MAX_NAME + 1] = {0};
        if (nc_inq_varname(ncid_, i, name) == NC_NOERR) {
            names.emplace_back(name);
        }
    }
    return names;
}

std::optional<oscean::core_services::VariableMeta> NetCDFVariableProcessor::getVariableInfo(const std::string& variableName) const {
    return extractVariableInfo(variableName);
}

bool NetCDFVariableProcessor::variableExists(const std::string& variableName) const {
    return getVariableId(variableName) >= 0;
}

varid_t NetCDFVariableProcessor::getVariableId(const std::string& variableName) const {
    varid_t varid;
    if (nc_inq_varid(ncid_, variableName.c_str(), &varid) != NC_NOERR) {
        return -1;
    }
    return varid;
}

int NetCDFVariableProcessor::getVariableDimensionCount(const std::string& variableName) const {
    int varid = getVariableId(variableName);
    if (varid < 0) return -1;
    
    int ndims;
    if (nc_inq_varndims(ncid_, varid, &ndims) != NC_NOERR) {
        return -1;
    }
    return ndims;
}

std::vector<size_t> NetCDFVariableProcessor::getVariableShape(const std::string& variableName) const {
    int varid = getVariableId(variableName);
    if (varid < 0) return {};

    int ndims;
    if (nc_inq_varndims(ncid_, varid, &ndims) != NC_NOERR) return {};
    if (ndims == 0) return {};

    std::vector<int> dimids(ndims);
    if (nc_inq_vardimid(ncid_, varid, dimids.data()) != NC_NOERR) return {};

    std::vector<size_t> shape(ndims);
    for (int i = 0; i < ndims; ++i) {
        if (nc_inq_dimlen(ncid_, dimids[i], &shape[i]) != NC_NOERR) {
            return {};
        }
    }
    return shape;
}

// =============================================================================
// Variable Data Reading (Stubs)
// =============================================================================

std::shared_ptr<oscean::core_services::GridData> NetCDFVariableProcessor::readVariable(
    const std::string& variableName, const VariableReadOptions& options) const {
    LOG_WARN("NetCDFVariableProcessor::readVariable for '{}' is not fully implemented yet.", variableName);
    return nullptr;
}

std::vector<double> NetCDFVariableProcessor::readVariableSubset(
    const std::string& variableName, const std::vector<size_t>& start,
    const std::vector<size_t>& count, const std::vector<size_t>& stride) const {
    LOG_WARN("NetCDFVariableProcessor::readVariableSubset for '{}' is not fully implemented yet.", variableName);
    return {};
}

std::shared_ptr<oscean::core_services::GridData> NetCDFVariableProcessor::readVariableTimeStep(
    const std::string& variableName, size_t timeIndex, const VariableReadOptions& options) const {
    LOG_WARN("NetCDFVariableProcessor::readVariableTimeStep for '{}' is not fully implemented yet.", variableName);
    return nullptr;
}

std::shared_ptr<oscean::core_services::GridData> NetCDFVariableProcessor::readVariableLevel(
    const std::string& variableName, size_t levelIndex, const VariableReadOptions& options) const {
    LOG_WARN("NetCDFVariableProcessor::readVariableLevel for '{}' is not fully implemented yet.", variableName);
    return nullptr;
}

// =============================================================================
// Variable Attribute Handling
// =============================================================================

std::vector<oscean::core_services::MetadataEntry> NetCDFVariableProcessor::getVariableAttributes(const std::string& variableName) const {
    int varid = getVariableId(variableName);
    if (varid < 0) return {};

    int natts;
    if (nc_inq_varnatts(ncid_, varid, &natts) != NC_NOERR) return {};

    std::vector<oscean::core_services::MetadataEntry> entries;
    entries.reserve(natts);
    for (int i = 0; i < natts; ++i) {
        char name[NC_MAX_NAME + 1] = {0};
        if (nc_inq_attname(ncid_, varid, i, name) == NC_NOERR) {
            std::string value = NetCDFUtils::readStringAttribute(ncid_, varid, std::string(name), "");
            entries.emplace_back(std::string(name), value, "variable", "string");
        }
    }
    return entries;
}

std::string NetCDFVariableProcessor::readStringAttribute(const std::string& variableName, const std::string& attributeName) const {
    int varid = getVariableId(variableName);
    if (varid < 0) return "";
    return NetCDFUtils::readStringAttribute(ncid_, varid, attributeName, "");
}

double NetCDFVariableProcessor::readNumericAttribute(const std::string& variableName, const std::string& attributeName, double defaultValue) const {
    int varid = getVariableId(variableName);
    if (varid < 0) return defaultValue;
    return NetCDFUtils::readNumericAttribute(ncid_, varid, attributeName, defaultValue);
}

bool NetCDFVariableProcessor::hasAttribute(const std::string& variableName, const std::string& attributeName) const {
    int varid = getVariableId(variableName);
    if (varid < 0) return false;
    return NetCDFUtils::hasAttribute(ncid_, varid, attributeName);
}

// =============================================================================
// Data Processing and Conversion (Stubs)
// =============================================================================

void NetCDFVariableProcessor::applyScaleAndOffset(std::vector<double>& data, double scaleFactor, double addOffset) const {
    // Stub
}

void NetCDFVariableProcessor::handleNoDataValues(std::vector<double>& data, double noDataValue) const {
    // Stub
}

std::vector<double> NetCDFVariableProcessor::convertToDouble(const void* data, int ncType, size_t count) const {
    // Stub
    return {};
}

bool NetCDFVariableProcessor::validateData(const std::vector<double>& data, const oscean::core_services::VariableMeta& varInfo) const {
    // Stub
    return true;
}

// =============================================================================
// Spatial & Temporal Subset Handling (Stubs)
// =============================================================================

std::optional<SpatialIndices> NetCDFVariableProcessor::calculateSpatialIndices(
    const std::string& variableName, const oscean::core_services::BoundingBox& bounds) const {
    return std::nullopt; // Stub
}

void NetCDFVariableProcessor::applySpatialSubset(
    const std::vector<std::string>& dimensions, const std::vector<size_t>& shape,
    const SpatialIndices& spatialIndices, std::vector<size_t>& start, std::vector<size_t>& count) const {
    // Stub
}

void NetCDFVariableProcessor::applyTimeSubset(
    const std::vector<std::string>& dimensions, const std::vector<size_t>& shape,
    const std::pair<std::chrono::system_clock::time_point, std::chrono::system_clock::time_point>& timeRange,
    std::vector<size_t>& start, std::vector<size_t>& count) const {
    // Stub
}

std::vector<double> NetCDFVariableProcessor::readCoordinateData(const std::string& coordName) const {
    return {}; // Stub
}

// =============================================================================
// Caching (Stubs)
// =============================================================================

void NetCDFVariableProcessor::clearCache() {
    // Stub
}

void NetCDFVariableProcessor::preloadVariableInfo() {
    // Stub
}

NetCDFVariableProcessor::CacheStats NetCDFVariableProcessor::getCacheStats() const {
    return {}; // Stub
}

// =============================================================================
// Private Helper Methods
// =============================================================================

oscean::core_services::VariableMeta NetCDFVariableProcessor::extractVariableInfo(const std::string& variableName) const {
    oscean::core_services::VariableMeta varInfo;
    varid_t varid = getVariableId(variableName);
    
    if (varid < 0) {
        return varInfo; // Return empty meta
    }

    varInfo.name = variableName;

    nc_type type;
    if (nc_inq_vartype(ncid_, varid, &type) == NC_NOERR) {
        varInfo.dataType = convertNetCDFDataType(type);
    }

    int ndims;
    if (nc_inq_varndims(ncid_, varid, &ndims) == NC_NOERR && ndims > 0) {
        std::vector<int> dimids(ndims);
        if (nc_inq_vardimid(ncid_, varid, dimids.data()) == NC_NOERR) {
            for (int dimid : dimids) {
                char dimname[NC_MAX_NAME + 1] = {0};
                if (nc_inq_dimname(ncid_, dimid, dimname) == NC_NOERR) {
                    varInfo.dimensionNames.push_back(dimname);
                }
            }
        }
    }

    auto attributes = getVariableAttributes(variableName);
    for(const auto& entry : attributes) {
        varInfo.attributes[entry.getKey()] = entry.getValue();
    }
    
    parseCFConventions(variableName, varInfo);

    return varInfo;
}

oscean::core_services::DataType NetCDFVariableProcessor::convertNetCDFDataType(int ncType) const {
    return NetCDFUtils::convertNetCDFType(ncType);
}

void NetCDFVariableProcessor::parseCFConventions(const std::string& variableName, oscean::core_services::VariableMeta& varInfo) const {
    // Extract common fields from attributes for convenience
    if (auto it = varInfo.attributes.find("units"); it != varInfo.attributes.end()) {
        varInfo.units = it->second;
    }
     if (auto it = varInfo.attributes.find("long_name"); it != varInfo.attributes.end()) {
        varInfo.description = it->second;
    } else if (auto it = varInfo.attributes.find("standard_name"); it != varInfo.attributes.end()) {
        varInfo.description = it->second;
    }
}

bool NetCDFVariableProcessor::validateVariableName(const std::string& variableName) const {
    // Stub
    return true;
}

bool NetCDFVariableProcessor::validateReadParameters(const std::vector<size_t>& start, const std::vector<size_t>& count) const {
    // Stub
    return true;
}

} // namespace oscean::core_services::data_access::readers::impl::netcdf 