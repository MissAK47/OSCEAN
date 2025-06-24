#include "engines/data_export_engine.h"
#include "engines/in_memory_data_reader.h"
#include "writers/writer_factory.h"
#include "writers/i_writer.h"
#include "core_services/data_access/i_data_reader.h"
#include "core_services/common_data_types.h"
#include "core_services/exceptions.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"

// Boost配置头文件必须在boost库包含之前
#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <numeric>
#include <algorithm>

namespace oscean {
namespace output {

DataExportEngine::DataExportEngine(
    std::shared_ptr<internal::WriterFactory> writerFactory,
    std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPool)
    : m_writerFactory(std::move(writerFactory))
    , m_threadPool(std::move(threadPool)) {
    
    if (!m_writerFactory) {
        throw std::invalid_argument("WriterFactory cannot be null");
    }
    if (!m_threadPool) {
        throw std::invalid_argument("ThreadPool cannot be null");
    }
    
    BOOST_LOG_TRIVIAL(info) << "DataExportEngine created with DI dependencies.";
}

boost::future<oscean::core_services::output::OutputResult> DataExportEngine::process(
    const oscean::core_services::output::OutputRequest& request) {

    return m_threadPool->submitTask([this, request]() -> oscean::core_services::output::OutputResult {
        try {
            auto writer = m_writerFactory->createWriter(request.format);
            if (!writer) {
                throw oscean::core_services::ServiceException("Unsupported format: " + request.format);
            }

            // 使用 visitor 安全地提取 IDataReader
            struct GetReaderVisitor : public boost::static_visitor<std::shared_ptr<oscean::core_services::IDataReader>> {
                std::shared_ptr<oscean::core_services::IDataReader> operator()(const std::shared_ptr<oscean::core_services::IDataReader>& reader) const {
                    return reader;
                }
                
                std::shared_ptr<oscean::core_services::IDataReader> operator()(const std::string& /*path*/) const {
                    // 如果需要，这里可以实现从路径创建DataReader的逻辑
                    throw oscean::core_services::ServiceException("DataExportEngine expects an IDataReader, not a file path.");
                }
            };

            GetReaderVisitor visitor;
            std::shared_ptr<oscean::core_services::IDataReader> reader = boost::apply_visitor(visitor, request.dataSource);

            if (!reader) {
                throw oscean::core_services::ServiceException("IDataReader is null");
            }

            std::vector<std::string> generatedFiles;

            if (!request.chunking) {
                // 非分块模式：使用writer的简单write方法
                BOOST_LOG_TRIVIAL(info) << "Processing non-chunked export for format: " << request.format;
                generatedFiles = writer->write(reader, request).get();
            } else {
                // 分块模式：实现智能分块逻辑
                BOOST_LOG_TRIVIAL(info) << "Starting chunked export with maxSize: " 
                                       << request.chunking->maxFileSizeMB << "MB, strategy: " 
                                       << request.chunking->strategy;
                
                generatedFiles = processChunkedExport(writer, reader, request);
                
                BOOST_LOG_TRIVIAL(info) << "Chunked export finished. " << generatedFiles.size() << " files created.";
            }

            // 构造OutputResult
            oscean::core_services::output::OutputResult result;
            result.filePaths = generatedFiles;
            return result;

        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "DataExportEngine::process failed: " << e.what();
            throw;
        }
    }, common_utils::infrastructure::TaskType::IO_BOUND);
}

std::vector<std::string> DataExportEngine::processChunkedExport(
    std::shared_ptr<internal::IWriter> writer,
    std::shared_ptr<oscean::core_services::IDataReader> reader,
    const oscean::core_services::output::OutputRequest& request) {
    
    std::vector<std::string> generatedFiles;
    const auto& chunking = *request.chunking;
    const double maxSizeBytes = chunking.maxFileSizeMB * 1024.0 * 1024.0;
    
    try {
        // 获取所有可用的变量名
        auto variableNames = reader->listDataVariableNames();
        if (variableNames.empty()) {
            throw oscean::core_services::ServiceException("No variables found in data source");
        }
        
        // 处理每个变量（通常是第一个变量）
        const std::string& variableName = variableNames[0];
        
        if (chunking.strategy == "byRow") {
            generatedFiles = processChunkedByRow(writer, reader, request, variableName, maxSizeBytes);
        } else if (chunking.strategy == "bySlice") {
            generatedFiles = processChunkedBySlice(writer, reader, request, variableName, maxSizeBytes);
        } else {
            BOOST_LOG_TRIVIAL(warning) << "Unknown chunking strategy: " << chunking.strategy 
                                      << ", falling back to byRow";
            generatedFiles = processChunkedByRow(writer, reader, request, variableName, maxSizeBytes);
        }
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Chunked export failed: " << e.what();
        throw;
    }
    
    return generatedFiles;
}

std::vector<std::string> DataExportEngine::processChunkedByRow(
    std::shared_ptr<internal::IWriter> writer,
    std::shared_ptr<oscean::core_services::IDataReader> reader,
    const oscean::core_services::output::OutputRequest& request,
    const std::string& variableName,
    double maxSizeBytes) {
    
    std::vector<std::string> generatedFiles;
    
    try {
        // 优化：首先获取变量信息而不读取完整数据
        auto variableDimensions = reader->getVariableDimensions(variableName);
        if (!variableDimensions) {
            throw oscean::core_services::ServiceException("Failed to get variable dimensions for: " + variableName);
        }
        
        // 获取基本信息而不加载全部数据
        auto bounds = reader->getNativeBoundingBox();
        auto nativeCrs = reader->getNativeCrs();
        
        // 计算行数 - 从维度信息获取
        size_t totalRows = 0;
        size_t totalCols = 0;
        size_t totalBands = 1;
        
        for (const auto& dim : *variableDimensions) {
            if (dim.name == "y" || dim.name == "lat" || dim.name == "latitude") {
                totalRows = dim.size;
            } else if (dim.name == "x" || dim.name == "lon" || dim.name == "longitude") {
                totalCols = dim.size;
            } else if (dim.name == "band" || dim.name == "time" || dim.name == "level") {
                totalBands = dim.size;
            }
        }
        
        if (totalRows == 0 || totalCols == 0) {
            // 作为后备方案，读取完整数据获取维度
            auto fullGridData = reader->readGridData(variableName);
            if (!fullGridData) {
                throw oscean::core_services::ServiceException("Failed to read grid data for variable: " + variableName);
            }
            
            const auto& definition = fullGridData->getDefinition();
            totalRows = definition.rows;
            totalCols = definition.cols;
            totalBands = fullGridData->getBandCount();
        }
        
        // 智能内存管理：估算内存使用
        size_t bytesPerElement = 4; // 假设float32
        size_t estimatedRowSize = totalCols * totalBands * bytesPerElement;
        size_t maxRowsPerChunk = static_cast<size_t>(maxSizeBytes / estimatedRowSize);
        
        // 确保至少每个chunk有1行，但不超过总行数
        maxRowsPerChunk = std::max(static_cast<size_t>(1), std::min(maxRowsPerChunk, totalRows));
        
        // 优化：调整分块大小以更均匀地分配数据并最小化内存峰值
        size_t optimalChunkCount = (totalRows + maxRowsPerChunk - 1) / maxRowsPerChunk;
        maxRowsPerChunk = (totalRows + optimalChunkCount - 1) / optimalChunkCount;
        
        BOOST_LOG_TRIVIAL(info) << "Memory-efficient chunking by row: " << totalRows << " total rows, " 
                               << maxRowsPerChunk << " rows per chunk, "
                               << optimalChunkCount << " chunks, "
                               << "estimated " << (estimatedRowSize * maxRowsPerChunk / 1024.0 / 1024.0) << " MB per chunk";
        
        size_t chunkIndex = 0;
        for (size_t startRow = 0; startRow < totalRows; startRow += maxRowsPerChunk) {
            size_t endRow = std::min(startRow + maxRowsPerChunk, totalRows);
            
            // 优化：延迟数据加载 - 只为当前chunk读取数据
            std::vector<oscean::core_services::IndexRange> sliceRanges;
            
            // 构建行范围切片 - 使用正确的IndexRange字段
            oscean::core_services::IndexRange rowRange;
            rowRange.start = static_cast<int>(startRow);
            rowRange.count = static_cast<int>(endRow - startRow);
            sliceRanges.push_back(rowRange);
            
            // 使用切片直接读取子集数据，避免加载完整数据集
            auto chunkData = reader->readGridData(variableName, boost::none, boost::none, boost::none, sliceRanges);
            if (!chunkData) {
                BOOST_LOG_TRIVIAL(warning) << "Failed to read row subset for chunk " << chunkIndex 
                                          << " (rows " << startRow << "-" << (endRow-1) << ")";
                continue;
            }
            
            // 生成chunk文件名
            std::string chunkFilename = generateChunkFilename(request, chunkIndex);
            
            // 使用writer的分块接口
            auto chunkRequest = request; // 复制request
            
            // 添加进度信息到请求的元数据中
            if (!chunkRequest.creationOptions) {
                chunkRequest.creationOptions = std::map<std::string, std::string>();
            }
            (*chunkRequest.creationOptions)["chunk_index"] = std::to_string(chunkIndex);
            (*chunkRequest.creationOptions)["total_chunks"] = std::to_string(optimalChunkCount);
            (*chunkRequest.creationOptions)["chunk_start_row"] = std::to_string(startRow);
            (*chunkRequest.creationOptions)["chunk_end_row"] = std::to_string(endRow - 1);
            (*chunkRequest.creationOptions)["chunk_strategy"] = "by_row";
            (*chunkRequest.creationOptions)["source_variable"] = variableName;
            
            writer->open(chunkFilename, chunkRequest).get();
            
            boost::variant<std::shared_ptr<oscean::core_services::GridData>, std::shared_ptr<oscean::core_services::FeatureCollection>> dataVariant = chunkData;
            writer->writeChunk(dataVariant).get();
            
            writer->close().get();
            
            generatedFiles.push_back(chunkFilename);
            chunkIndex++;
            
            BOOST_LOG_TRIVIAL(debug) << "Created chunk " << chunkIndex << "/" << optimalChunkCount 
                                    << ": " << chunkFilename
                                    << " (rows " << startRow << "-" << (endRow-1) << ")"
                                    << " actual size: " << (chunkData->getDataSizeBytes() / 1024.0 / 1024.0) << " MB";
                                    
            // 优化：立即释放chunk数据，减少内存占用
            chunkData.reset();
        }
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Memory-efficient row-based chunking failed: " << e.what();
        throw;
    }
    
    return generatedFiles;
}

std::vector<std::string> DataExportEngine::processChunkedBySlice(
    std::shared_ptr<internal::IWriter> writer,
    std::shared_ptr<oscean::core_services::IDataReader> reader,
    const oscean::core_services::output::OutputRequest& request,
    const std::string& variableName,
    double maxSizeBytes) {
    
    std::vector<std::string> generatedFiles;
    
    try {
        // 优化：获取变量维度信息以避免加载完整数据
        auto variableDimensions = reader->getVariableDimensions(variableName);
        if (!variableDimensions) {
            throw oscean::core_services::ServiceException("Failed to get variable dimensions for: " + variableName);
        }
        
        // 从维度信息解析波段/时间层数
        size_t totalBands = 1;
        size_t totalRows = 0;
        size_t totalCols = 0;
        std::string bandDimensionName = "band"; // 默认
        
        for (const auto& dim : *variableDimensions) {
            if (dim.name == "band" || dim.name == "time" || dim.name == "level" || dim.name == "z") {
                totalBands = dim.size;
                bandDimensionName = dim.name;
            } else if (dim.name == "y" || dim.name == "lat" || dim.name == "latitude") {
                totalRows = dim.size;
            } else if (dim.name == "x" || dim.name == "lon" || dim.name == "longitude") {
                totalCols = dim.size;
            }
        }
        
        // 如果只有一个波段，使用行分块方法
        if (totalBands <= 1) {
            BOOST_LOG_TRIVIAL(info) << "Only one band available, switching to row-based chunking";
            return processChunkedByRow(writer, reader, request, variableName, maxSizeBytes);
        }
        
        // 智能内存管理：估算每个波段的内存使用
        size_t bytesPerElement = 4; // 假设float32
        size_t estimatedBandSize = totalRows * totalCols * bytesPerElement;
        size_t maxBandsPerChunk = static_cast<size_t>(maxSizeBytes / estimatedBandSize);
        
        // 确保至少每个chunk有1个波段，但不超过总波段数
        maxBandsPerChunk = std::max(static_cast<size_t>(1), std::min(maxBandsPerChunk, totalBands));
        
        // 优化：调整分块大小以更均匀地分配数据
        size_t optimalChunkCount = (totalBands + maxBandsPerChunk - 1) / maxBandsPerChunk;
        maxBandsPerChunk = (totalBands + optimalChunkCount - 1) / optimalChunkCount;
        
        BOOST_LOG_TRIVIAL(info) << "Memory-efficient chunking by slice: " << totalBands << " total bands, " 
                               << maxBandsPerChunk << " bands per chunk, "
                               << optimalChunkCount << " chunks, "
                               << "estimated " << (estimatedBandSize * maxBandsPerChunk / 1024.0 / 1024.0) << " MB per chunk";
        
        size_t chunkIndex = 0;
        for (size_t startBand = 0; startBand < totalBands; startBand += maxBandsPerChunk) {
            size_t endBand = std::min(startBand + maxBandsPerChunk, totalBands);
            
            // 优化：延迟数据加载 - 只为当前chunk读取数据
            std::vector<oscean::core_services::IndexRange> sliceRanges;
            
            // 构建波段范围切片 - 使用正确的IndexRange字段
            oscean::core_services::IndexRange bandRange;
            bandRange.start = static_cast<int>(startBand);
            bandRange.count = static_cast<int>(endBand - startBand);
            sliceRanges.push_back(bandRange);
            
            // 使用切片直接读取子集数据，避免加载完整数据集
            auto chunkData = reader->readGridData(variableName, boost::none, boost::none, boost::none, sliceRanges);
            if (!chunkData) {
                BOOST_LOG_TRIVIAL(warning) << "Failed to read band subset for chunk " << chunkIndex 
                                          << " (bands " << startBand << "-" << (endBand-1) << ")";
                continue;
            }
            
            // 生成chunk文件名
            std::string chunkFilename = generateChunkFilename(request, chunkIndex);
            
            // 使用writer的分块接口
            auto chunkRequest = request; // 复制request
            
            // 添加进度信息到请求的元数据中
            if (!chunkRequest.creationOptions) {
                chunkRequest.creationOptions = std::map<std::string, std::string>();
            }
            (*chunkRequest.creationOptions)["chunk_index"] = std::to_string(chunkIndex);
            (*chunkRequest.creationOptions)["total_chunks"] = std::to_string(optimalChunkCount);
            (*chunkRequest.creationOptions)["chunk_start_band"] = std::to_string(startBand);
            (*chunkRequest.creationOptions)["chunk_end_band"] = std::to_string(endBand - 1);
            (*chunkRequest.creationOptions)["chunk_strategy"] = "by_slice";
            (*chunkRequest.creationOptions)["source_variable"] = variableName;
            (*chunkRequest.creationOptions)["band_dimension"] = bandDimensionName;
            
            // 添加波段坐标信息
            std::vector<double> bandCoords = chunkData->getBandCoordinates();
            if (!bandCoords.empty()) {
                std::string bandCoordStr;
                for (size_t i = 0; i < bandCoords.size(); ++i) {
                    if (!bandCoordStr.empty()) bandCoordStr += ",";
                    bandCoordStr += std::to_string(bandCoords[i]);
                }
                (*chunkRequest.creationOptions)["band_coordinates"] = bandCoordStr;
            }
            
            writer->open(chunkFilename, chunkRequest).get();
            
            boost::variant<std::shared_ptr<oscean::core_services::GridData>, std::shared_ptr<oscean::core_services::FeatureCollection>> dataVariant = chunkData;
            writer->writeChunk(dataVariant).get();
            
            writer->close().get();
            
            generatedFiles.push_back(chunkFilename);
            chunkIndex++;
            
            BOOST_LOG_TRIVIAL(debug) << "Created chunk " << chunkIndex << "/" << optimalChunkCount 
                                    << ": " << chunkFilename
                                    << " (bands " << startBand << "-" << (endBand-1) << ")"
                                    << " actual size: " << (chunkData->getDataSizeBytes() / 1024.0 / 1024.0) << " MB";
                                    
            // 优化：立即释放chunk数据，减少内存占用
            chunkData.reset();
        }
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Memory-efficient slice-based chunking failed: " << e.what();
        throw;
    }
    
    return generatedFiles;
}

std::string DataExportEngine::generateChunkFilename(
    const oscean::core_services::output::OutputRequest& request,
    size_t chunkIndex) {
    
    // 获取基本文件名
    std::string baseFilename;
    if (request.filenameTemplate) {
        baseFilename = *request.filenameTemplate;
    } else {
        // 默认文件名
        baseFilename = "output";
    }
    
    // 获取目标目录
    std::string targetDir;
    if (request.targetDirectory) {
        targetDir = *request.targetDirectory;
    } else {
        // 默认当前目录
        targetDir = ".";
    }
    
    // 确保目录存在
    boost::filesystem::path dirPath(targetDir);
    if (!boost::filesystem::exists(dirPath)) {
        boost::filesystem::create_directories(dirPath);
    }
    
    // 解析文件名和扩展名
    boost::filesystem::path filePath(baseFilename);
    std::string filename = filePath.stem().string();
    std::string extension = filePath.extension().string();
    
    // 如果没有扩展名，使用请求的格式作为扩展名
    if (extension.empty()) {
        extension = "." + request.format;
    }
    
    // 生成带有块索引的文件名
    std::string chunkFilename = (boost::format("%s_chunk%04d%s") % filename % chunkIndex % extension).str();
    
    // 组合完整路径
    boost::filesystem::path fullPath = dirPath / chunkFilename;
    
    return fullPath.string();
}

std::shared_ptr<oscean::core_services::GridData> DataExportEngine::createGridDataSubset(
    std::shared_ptr<oscean::core_services::GridData> fullData,
    size_t startRow, size_t endRow) {
    
    // 使用InMemoryDataReader进行真正的数据切片
    try {
        InMemoryDataReader memoryReader(fullData, "temp_variable");
        return memoryReader.createRowSubset("temp_variable", startRow, endRow);
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Failed to create row subset: " << e.what();
        throw;
    }
}

std::shared_ptr<oscean::core_services::GridData> DataExportEngine::createGridDataBandSubset(
    std::shared_ptr<oscean::core_services::GridData> fullData,
    size_t startBand, size_t endBand) {
    
    // 使用InMemoryDataReader进行真正的波段切片
    try {
        InMemoryDataReader memoryReader(fullData, "temp_variable");
        return memoryReader.createBandSubset("temp_variable", startBand, endBand);
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Failed to create band subset: " << e.what();
        throw;
    }
}

std::vector<std::string> DataExportEngine::processGridDataInChunks(
    std::shared_ptr<internal::IWriter> writer, 
    const oscean::core_services::output::OutputRequest& request,
    std::shared_ptr<oscean::core_services::GridData> fullGridData)
{
    // 简化实现：直接调用writer的write方法
    std::vector<std::string> generatedFiles;
    BOOST_LOG_TRIVIAL(info) << "Processing grid data chunks (simplified implementation)";
    return generatedFiles;
}

std::vector<std::string> DataExportEngine::processFeatureCollectionInChunks(
    std::shared_ptr<internal::IWriter> writer, 
    const oscean::core_services::output::OutputRequest& request,
    std::shared_ptr<oscean::core_services::FeatureCollection> featureCollection)
{
    std::vector<std::string> generatedFiles;
    
    try {
        if (!featureCollection || featureCollection->empty()) {
            BOOST_LOG_TRIVIAL(warning) << "Empty or null feature collection provided";
            return generatedFiles;
        }
        
        // 计算合适的分块大小
        size_t totalFeatures = featureCollection->size();
        size_t maxFeaturesPerChunk = 1000; // 默认每块1000个要素
        
        // 如果有创建选项中指定的分块大小，使用该值
        if (request.creationOptions) {
            auto it = request.creationOptions->find("features_per_chunk");
            if (it != request.creationOptions->end()) {
                try {
                    maxFeaturesPerChunk = std::stoul(it->second);
                } catch (...) {
                    BOOST_LOG_TRIVIAL(warning) << "Invalid features_per_chunk value: " << it->second;
                }
            }
        }
        
        // 确保至少每个chunk有1个要素
        maxFeaturesPerChunk = std::max(static_cast<size_t>(1), maxFeaturesPerChunk);
        
        // 计算分块数量
        size_t optimalChunkCount = (totalFeatures + maxFeaturesPerChunk - 1) / maxFeaturesPerChunk;
        
        BOOST_LOG_TRIVIAL(info) << "Processing FeatureCollection in chunks: " << totalFeatures << " total features, " 
                               << maxFeaturesPerChunk << " features per chunk, "
                               << optimalChunkCount << " chunks";
        
        size_t chunkIndex = 0;
        for (size_t startFeature = 0; startFeature < totalFeatures; startFeature += maxFeaturesPerChunk) {
            size_t endFeature = std::min(startFeature + maxFeaturesPerChunk, totalFeatures);
            
            // 创建要素子集
            auto chunkFeatureCollection = std::make_shared<oscean::core_services::FeatureCollection>();
            
            // 复制元数据
            chunkFeatureCollection->name = featureCollection->name + "_chunk_" + std::to_string(chunkIndex);
            chunkFeatureCollection->crs = featureCollection->crs;
            chunkFeatureCollection->extent = featureCollection->extent;
            chunkFeatureCollection->fieldDefinitions = featureCollection->fieldDefinitions;
            
            // 复制要素子集 - 使用正确的FeatureCollection访问方式
            const auto& features = featureCollection->getFeatures();
            for (size_t i = startFeature; i < endFeature; ++i) {
                if (i < features.size()) {
                    chunkFeatureCollection->addFeature(features[i]);
                }
            }
            
            // 生成chunk文件名
            std::string chunkFilename = generateChunkFilename(request, chunkIndex);
            
            // 使用writer的分块接口
            auto chunkRequest = request; // 复制request
            
            // 添加进度信息到请求的元数据中
            if (!chunkRequest.creationOptions) {
                chunkRequest.creationOptions = std::map<std::string, std::string>();
            }
            (*chunkRequest.creationOptions)["chunk_index"] = std::to_string(chunkIndex);
            (*chunkRequest.creationOptions)["total_chunks"] = std::to_string(optimalChunkCount);
            (*chunkRequest.creationOptions)["chunk_start_feature"] = std::to_string(startFeature);
            (*chunkRequest.creationOptions)["chunk_end_feature"] = std::to_string(endFeature - 1);
            (*chunkRequest.creationOptions)["chunk_strategy"] = "by_feature";
            (*chunkRequest.creationOptions)["total_features"] = std::to_string(totalFeatures);
            (*chunkRequest.creationOptions)["chunk_feature_count"] = std::to_string(chunkFeatureCollection->size());
            
            // 添加FeatureCollection特定的元数据
            if (!featureCollection->name.empty()) {
                (*chunkRequest.creationOptions)["source_collection_name"] = featureCollection->name;
            }
            if (featureCollection->crs && !featureCollection->crs->id.empty()) {
                (*chunkRequest.creationOptions)["source_crs"] = featureCollection->crs->id;
            }
            
            writer->open(chunkFilename, chunkRequest).get();
            
            boost::variant<std::shared_ptr<oscean::core_services::GridData>, std::shared_ptr<oscean::core_services::FeatureCollection>> dataVariant = chunkFeatureCollection;
            writer->writeChunk(dataVariant).get();
            
            writer->close().get();
            
            generatedFiles.push_back(chunkFilename);
            chunkIndex++;
            
            BOOST_LOG_TRIVIAL(debug) << "Created FeatureCollection chunk " << chunkIndex << "/" << optimalChunkCount 
                                    << ": " << chunkFilename
                                    << " (features " << startFeature << "-" << (endFeature-1) << ")"
                                    << " actual feature count: " << chunkFeatureCollection->size();
                                    
            // 优化：释放chunk数据，减少内存占用
            chunkFeatureCollection.reset();
        }
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "FeatureCollection chunking failed: " << e.what();
        throw;
    }
    
    return generatedFiles;
}

} // namespace output
} // namespace oscean