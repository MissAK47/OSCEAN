#include "engines/in_memory_data_reader.h"
#include "core_services/exceptions.h"
#include <stdexcept>
#include <cstring>
#include <boost/log/trivial.hpp>

namespace oscean {
namespace output {

// 添加简化构造函数实现
InMemoryDataReader::InMemoryDataReader(
    std::shared_ptr<oscean::core_services::GridData> gridData)
    : m_gridData(std::move(gridData))
    , m_isOpen(true) {
    
    if (!m_gridData) {
        throw std::invalid_argument("GridData cannot be null");
    }
    
    // 使用GridData的变量名，如果没有则使用默认名称
    m_variableName = m_gridData->getVariableName();
    if (m_variableName.empty()) {
        m_variableName = "data";
    }
    
    BOOST_LOG_TRIVIAL(info) << "InMemoryDataReader created with variable name: " << m_variableName;
}

InMemoryDataReader::InMemoryDataReader(
    std::shared_ptr<oscean::core_services::GridData> gridData,
    const std::string& variableName)
    : m_gridData(std::move(gridData))
    , m_variableName(variableName)
    , m_isOpen(true) {
    
    if (!m_gridData) {
        throw std::invalid_argument("GridData cannot be null");
    }
    if (variableName.empty()) {
        throw std::invalid_argument("Variable name cannot be empty");
    }
    
    BOOST_LOG_TRIVIAL(info) << "InMemoryDataReader created for variable: " << variableName;
}

// ===== IDataReader接口实现 =====

bool InMemoryDataReader::open(const std::string& filePath, 
                               const boost::optional<std::string>& targetCRS) {
    // 内存读取器总是成功打开
    m_isOpen = true;
    return true;
}

bool InMemoryDataReader::open() {
    m_isOpen = true;
    return true;
}

void InMemoryDataReader::close() {
    m_isOpen = false;
}

bool InMemoryDataReader::isOpen() const {
    return m_isOpen;
}

std::string InMemoryDataReader::getFilePath() const {
    return "memory://" + m_variableName;
}

std::vector<std::string> InMemoryDataReader::listDataVariableNames() const {
    return {m_variableName};
}

std::shared_ptr<oscean::core_services::GridData> InMemoryDataReader::readGridData(
    const std::string& variableName,
    boost::optional<oscean::core_services::CRSInfo> targetCRS,
    boost::optional<std::pair<double, double>> targetResolution,
    boost::optional<oscean::core_services::BoundingBox> outputBounds,
    const std::vector<oscean::core_services::IndexRange>& sliceRanges,
    oscean::core_services::ResampleAlgorithm resampleAlgo) {
    
    validateVariableName(variableName);
    
    if (!m_isOpen) {
        throw std::runtime_error("Data reader is not open");
    }
    
    // 对于内存读取器，我们直接返回数据（忽略重采样等参数）
    return m_gridData;
}

oscean::core_services::FeatureCollection InMemoryDataReader::readFeatureCollection(
    const std::string& layerName,
    const boost::optional<oscean::core_services::CRSInfo>& targetCRS,
    const boost::optional<oscean::core_services::BoundingBox>& filterBoundingBox,
    const boost::optional<oscean::core_services::CRSInfo>& bboxCRS) {
    
    throw std::runtime_error("InMemoryDataReader does not support FeatureCollection reading");
}

std::vector<oscean::core_services::MetadataEntry> InMemoryDataReader::getGlobalAttributes() const {
    std::vector<oscean::core_services::MetadataEntry> attrs;
    
    // 添加一些基本的全局属性 - 使用正确的MetadataEntry构造函数
    oscean::core_services::MetadataEntry sourceAttr("source", "InMemoryDataReader");
    attrs.push_back(sourceAttr);
    
    oscean::core_services::MetadataEntry variableAttr("primary_variable", m_variableName);
    attrs.push_back(variableAttr);
    
    return attrs;
}

boost::optional<std::vector<oscean::core_services::DimensionDefinition>> 
InMemoryDataReader::getVariableDimensions(const std::string& variableName) const {
    validateVariableName(variableName);
    
    if (!m_gridData) {
        return boost::none;
    }
    
    const auto& definition = m_gridData->getDefinition();
    std::vector<oscean::core_services::DimensionDefinition> dimensions;
    
    // 创建维度定义
    oscean::core_services::DimensionDefinition lonDim;
    lonDim.name = "longitude";
    lonDim.size = definition.cols;
    dimensions.push_back(lonDim);
    
    oscean::core_services::DimensionDefinition latDim;
    latDim.name = "latitude";
    latDim.size = definition.rows;
    dimensions.push_back(latDim);
    
    // GridDefinition没有bands字段，使用getBandCount()
    size_t bands = m_gridData->getBandCount();
    if (bands > 1) {
        oscean::core_services::DimensionDefinition bandDim;
        bandDim.name = "band";
        bandDim.size = bands;
        dimensions.push_back(bandDim);
    }
    
    return dimensions;
}

boost::optional<std::vector<oscean::core_services::MetadataEntry>> 
InMemoryDataReader::getVariableMetadata(const std::string& variableName) const {
    validateVariableName(variableName);
    
    std::vector<oscean::core_services::MetadataEntry> metadata;
    
    if (m_gridData) {
        const auto& definition = m_gridData->getDefinition();
        
        // 添加变量的基本信息 - 使用正确的MetadataEntry构造函数
        oscean::core_services::MetadataEntry nameEntry("variable_name", variableName);
        metadata.push_back(nameEntry);
        
        size_t bands = m_gridData->getBandCount();
        std::string dimStr = std::to_string(definition.cols) + "x" + std::to_string(definition.rows) + "x" + std::to_string(bands);
        oscean::core_services::MetadataEntry dimEntry("dimensions", dimStr);
        metadata.push_back(dimEntry);
    }
    
    return metadata;
}

boost::optional<oscean::core_services::CRSInfo> InMemoryDataReader::getNativeCrs() const {
    if (!m_gridData) {
        return boost::none;
    }
    
    // 返回默认的WGS84 CRS - 使用正确的字段名
    oscean::core_services::CRSInfo crs;
    crs.wkt = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]";
    crs.epsgCode = 4326;  // 使用int类型而不是字符串
    return crs;
}

oscean::core_services::BoundingBox InMemoryDataReader::getNativeBoundingBox() const {
    if (!m_gridData) {
        return oscean::core_services::BoundingBox{};
    }
    
    const auto& definition = m_gridData->getDefinition();
    return definition.extent;  // GridDefinition使用extent而不是spatialBounds
}

boost::optional<oscean::core_services::TimeRange> InMemoryDataReader::getNativeTimeRange() const {
    if (!m_gridData) {
        return boost::none;
    }
    
    // GridDefinition没有temporalBounds字段，返回空
    return boost::none;
}

std::vector<double> InMemoryDataReader::getVerticalLevels() const {
    // 内存读取器不支持垂直层级
    return {};
}

// ===== 专用切片方法 =====

std::shared_ptr<oscean::core_services::GridData> InMemoryDataReader::createRowSubset(
    const std::string& variableName, 
    size_t startRow, 
    size_t endRow) {
    
    validateVariableName(variableName);
    
    if (!m_gridData) {
        throw std::runtime_error("No source data available in InMemoryDataReader");
    }
    
    const auto& sourceDef = m_gridData->getDefinition();
    
    if (startRow >= sourceDef.rows || endRow > sourceDef.rows || startRow >= endRow) {
        throw std::invalid_argument("Invalid row range for subsetting.");
    }
    
    BOOST_LOG_TRIVIAL(info) << "Creating row subset: rows " << startRow << " to " << (endRow - 1);

    size_t numRows = endRow - startRow;
    
    // 1. 创建新的GridDefinition定义
    oscean::core_services::GridDefinition subsetDef;
    subsetDef.rows = numRows;
    subsetDef.cols = sourceDef.cols;
    subsetDef.xResolution = sourceDef.xResolution;
    subsetDef.yResolution = sourceDef.yResolution;
    subsetDef.crs = sourceDef.crs;
    subsetDef.gridName = sourceDef.gridName;
    subsetDef.originalDataType = sourceDef.originalDataType;
    subsetDef.dimensionOrderInDataLayout = sourceDef.dimensionOrderInDataLayout;
    subsetDef.globalAttributes = sourceDef.globalAttributes;

    // 2. 调整空间范围 (简单的线性插值)
    if (sourceDef.extent.isValid() && sourceDef.rows > 0) {
        double totalLatSpan = sourceDef.extent.maxY - sourceDef.extent.minY;
        double latPerRow = totalLatSpan / sourceDef.rows;
        subsetDef.extent.minX = sourceDef.extent.minX;
        subsetDef.extent.maxX = sourceDef.extent.maxX;
        subsetDef.extent.maxY = sourceDef.extent.maxY - startRow * latPerRow;
        subsetDef.extent.minY = subsetDef.extent.maxY - numRows * latPerRow;
        subsetDef.extent.crsId = sourceDef.extent.crsId;
        if (sourceDef.extent.minZ.has_value() && sourceDef.extent.maxZ.has_value()) {
            subsetDef.extent.minZ = sourceDef.extent.minZ;
            subsetDef.extent.maxZ = sourceDef.extent.maxZ;
        }
    }
    
    // 复制维度信息
    if (sourceDef.hasXDimension()) {
        subsetDef.xDimension.name = sourceDef.xDimension.name;
        subsetDef.xDimension.standardName = sourceDef.xDimension.standardName;
        subsetDef.xDimension.longName = sourceDef.xDimension.longName;
        subsetDef.xDimension.units = sourceDef.xDimension.units;
        subsetDef.xDimension.type = sourceDef.xDimension.type;
        subsetDef.xDimension.isRegular = sourceDef.xDimension.isRegular;
        subsetDef.xDimension.resolution = sourceDef.xDimension.resolution;
        subsetDef.xDimension.coordinates = sourceDef.xDimension.coordinates;
        subsetDef.xDimension.coordinateLabels = sourceDef.xDimension.coordinateLabels;
        subsetDef.xDimension.attributes = sourceDef.xDimension.attributes;
        subsetDef.xDimension.minValue = sourceDef.xDimension.minValue;
        subsetDef.xDimension.maxValue = sourceDef.xDimension.maxValue;
        subsetDef.xDimension.hasValueRange = sourceDef.xDimension.hasValueRange;
        subsetDef.xDimension.valueRange = sourceDef.xDimension.valueRange;
    }
    
    if (sourceDef.hasYDimension()) {
        subsetDef.yDimension.name = sourceDef.yDimension.name;
        subsetDef.yDimension.standardName = sourceDef.yDimension.standardName;
        subsetDef.yDimension.longName = sourceDef.yDimension.longName;
        subsetDef.yDimension.units = sourceDef.yDimension.units;
        subsetDef.yDimension.type = sourceDef.yDimension.type;
        subsetDef.yDimension.isRegular = sourceDef.yDimension.isRegular;
        subsetDef.yDimension.resolution = sourceDef.yDimension.resolution;
        
        // 对于Y坐标，只取子集
        if (sourceDef.yDimension.hasNumericCoordinates() && sourceDef.yDimension.coordinates.size() == sourceDef.rows) {
            std::vector<double> newYCoords;
            auto& srcYCoords = sourceDef.yDimension.coordinates;
            newYCoords.assign(srcYCoords.begin() + startRow, srcYCoords.begin() + endRow);
            subsetDef.yDimension.coordinates = newYCoords;
        }
        
        subsetDef.yDimension.coordinateLabels = sourceDef.yDimension.coordinateLabels;
        subsetDef.yDimension.attributes = sourceDef.yDimension.attributes;
        subsetDef.yDimension.minValue = sourceDef.yDimension.minValue;
        subsetDef.yDimension.maxValue = sourceDef.yDimension.maxValue;
        subsetDef.yDimension.hasValueRange = sourceDef.yDimension.hasValueRange;
        subsetDef.yDimension.valueRange = sourceDef.yDimension.valueRange;
    }
    
    // 复制Z维度和其他维度
    if (sourceDef.hasZDimension()) {
        subsetDef.zDimension = sourceDef.zDimension;
    }
    
    if (sourceDef.hasTDimension()) {
        subsetDef.tDimension = sourceDef.tDimension;
    }
    
    subsetDef.dimensions = sourceDef.dimensions;
    
    // 3. 创建新的GridData对象
    auto subsetData = std::make_shared<oscean::core_services::GridData>(
        subsetDef, m_gridData->getDataType(), m_gridData->getBandCount());

    // 4. 复制子集数据
    const auto* sourceBuffer = m_gridData->getUnifiedBuffer().data();
    auto& destBuffer = subsetData->getUnifiedBuffer();

    size_t bytesPerElement = m_gridData->getElementSizeBytes();
    size_t rowStrideBytes = sourceDef.cols * m_gridData->getBandCount() * bytesPerElement;
    size_t subsetSizeBytes = numRows * rowStrideBytes;
    
    destBuffer.resize(subsetSizeBytes);
    
    const unsigned char* sourceStartPtr = sourceBuffer + startRow * rowStrideBytes;
    std::memcpy(destBuffer.data(), sourceStartPtr, subsetSizeBytes);
    
    return subsetData;
}

std::shared_ptr<oscean::core_services::GridData> InMemoryDataReader::createBandSubset(
    const std::string& variableName, 
    size_t startBand, 
    size_t endBand) {
    
    validateVariableName(variableName);
    
    if (!m_gridData) {
        throw std::runtime_error("No source data available in InMemoryDataReader");
    }
    
    size_t sourceBandCount = m_gridData->getBandCount();
    
    if (startBand >= sourceBandCount || endBand > sourceBandCount || startBand >= endBand) {
        throw std::invalid_argument("Invalid band range for subsetting.");
    }

    BOOST_LOG_TRIVIAL(info) << "Creating band subset: bands " << startBand << " to " << (endBand - 1);
    
    size_t numBands = endBand - startBand;
    const auto& sourceDef = m_gridData->getDefinition();

    // 1. 创建新的GridDefinition定义
    oscean::core_services::GridDefinition subsetDef;
    subsetDef.rows = sourceDef.rows;
    subsetDef.cols = sourceDef.cols;
    subsetDef.xResolution = sourceDef.xResolution;
    subsetDef.yResolution = sourceDef.yResolution;
    subsetDef.crs = sourceDef.crs;
    subsetDef.gridName = sourceDef.gridName;
    subsetDef.originalDataType = sourceDef.originalDataType;
    subsetDef.dimensionOrderInDataLayout = sourceDef.dimensionOrderInDataLayout;
    subsetDef.globalAttributes = sourceDef.globalAttributes;
    subsetDef.extent = sourceDef.extent;
    
    // 复制所有维度信息
    if (sourceDef.hasXDimension()) {
        subsetDef.xDimension = sourceDef.xDimension;
    }
    
    if (sourceDef.hasYDimension()) {
        subsetDef.yDimension = sourceDef.yDimension;
    }
    
    if (sourceDef.hasZDimension()) {
        subsetDef.zDimension = sourceDef.zDimension;
    }
    
    if (sourceDef.hasTDimension()) {
        subsetDef.tDimension = sourceDef.tDimension;
    }
    
    subsetDef.dimensions = sourceDef.dimensions;
    
    // 2. 创建新的GridData对象
    auto subsetData = std::make_shared<oscean::core_services::GridData>(
        subsetDef, m_gridData->getDataType(), numBands);
        
    // 3. 复制子集数据 (假设BIP - Band Interleaved by Pixel)
    const auto* sourceBuffer = m_gridData->getUnifiedBuffer().data();
    auto& destBuffer = subsetData->getUnifiedBuffer();

    size_t bytesPerElement = m_gridData->getElementSizeBytes();
    size_t numPixels = sourceDef.rows * sourceDef.cols;
    size_t destPixelStride = numBands * bytesPerElement;
    size_t sourcePixelStride = sourceBandCount * bytesPerElement;
    
    destBuffer.resize(numPixels * destPixelStride);
    
    for (size_t i = 0; i < numPixels; ++i) {
        const unsigned char* sourcePixelStart = sourceBuffer + i * sourcePixelStride;
        unsigned char* destPixelStart = destBuffer.data() + i * destPixelStride;
        
        // 从源像素的起始波段开始复制
        const unsigned char* sourceBandStart = sourcePixelStart + startBand * bytesPerElement;
        
        // 复制连续的波段数据
        std::memcpy(destPixelStart, sourceBandStart, destPixelStride);
    }

    return subsetData;
}

std::shared_ptr<oscean::core_services::GridData> InMemoryDataReader::createSpatialSubset(
    const std::string& variableName, 
    const oscean::core_services::BoundingBox& bounds) {
    
    validateVariableName(variableName);
    
    if (!m_gridData) {
        throw std::runtime_error("No source data available");
    }
    
    // 简化实现：返回原始数据
    // 完整实现需要根据bounds计算行列索引
    BOOST_LOG_TRIVIAL(warning) << "Spatial subset not fully implemented, returning original data";
    return m_gridData;
}

std::shared_ptr<oscean::core_services::GridData> InMemoryDataReader::createTemporalSubset(
    const std::string& variableName, 
    const oscean::core_services::TimeRange& timeRange) {
    
    validateVariableName(variableName);
    
    if (!m_gridData) {
        throw std::runtime_error("No source data available");
    }
    
    // 简化实现：返回原始数据
    // 完整实现需要根据时间范围处理时间维度
    BOOST_LOG_TRIVIAL(warning) << "Temporal subset not fully implemented, returning original data";
    return m_gridData;
}

void InMemoryDataReader::validateVariableName(const std::string& variableName) const {
    if (variableName != m_variableName) {
        throw std::invalid_argument("Variable '" + variableName + "' not found. Available: " + m_variableName);
    }
}

} // namespace output
} // namespace oscean 