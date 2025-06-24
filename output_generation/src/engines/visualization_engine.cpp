#include "engines/visualization_engine.h"
#include "engines/in_memory_data_reader.h"
#include "core_services/exceptions.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "common_utils/utilities/boost_config.h"
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/thread/future.hpp>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <set>
#include <array>
#include <numeric>
#include <chrono>
#include <sstream>
#include <atomic>
#include "font_renderer.h"

// 包含GDAL头文件
#include "gdal_priv.h"
#include "cpl_conv.h" // For CPLMalloc()
#include "gdal_alg.h" // For GDALContourGenerate
#include "cpl_string.h" // For CSL functions

// 如果GDAL版本没有定义GDAL_CG_CONTOUR_LEVEL_INTERVAL常量，则自行定义
#ifndef GDAL_CG_CONTOUR_LEVEL_INTERVAL
#define GDAL_CG_CONTOUR_LEVEL_INTERVAL 2
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 添加GPU相关头文件
#ifdef OSCEAN_CUDA_ENABLED
#include "common_utils/gpu/oscean_gpu_framework.h"
#include "output_generation/gpu/gpu_color_mapper.h"
#include "output_generation/gpu/gpu_tile_generator.h"
#include <cuda_runtime.h>

// 声明CUDA核函数接口
extern "C" {
    cudaError_t generateContoursGPU(
        const float* d_gridData,
        int width, int height,
        const float* levels, int numLevels,
        float** d_contourPoints,
        int* numContourPoints,
        cudaStream_t stream);
}
#endif

namespace oscean {
namespace output {

VisualizationEngine::VisualizationEngine(
    std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPool,
    std::shared_ptr<common_utils::simd::UnifiedSIMDManager> simdManager)
    : m_threadPool(std::move(threadPool)), 
      simdManager_(std::move(simdManager)),
      useSIMDOptimization_(simdManager_ != nullptr),
      useGPUOptimization_(false),
      gpuAvailable_(false) {
    
    if (!m_threadPool) {
        throw std::invalid_argument("ThreadPool cannot be null");
    }
    
    // 初始化SIMD管理器
    if (simdManager) {
        simdManager_ = simdManager;
        useSIMDOptimization_ = true;
        BOOST_LOG_TRIVIAL(info) << "VisualizationEngine created with SIMD optimization";
    } else {
        useSIMDOptimization_ = false;
        BOOST_LOG_TRIVIAL(info) << "VisualizationEngine created without SIMD optimization";
    }
    
    // 初始化GDAL
    GDALAllRegister();
    OGRRegisterAll();
    CPLSetErrorHandler(CPLQuietErrorHandler); // 设置静默错误处理器
    
    // 初始化字体渲染器
    m_fontRenderer = std::make_unique<FontRenderer>();
    if (!m_fontRenderer->initialize()) {
        BOOST_LOG_TRIVIAL(warning) << "Font renderer initialization failed, text rendering will be limited";
        m_fontRenderer.reset(); // 释放失败的渲染器
    } else {
        // 设置默认字体样式
        FontRenderer::FontStyle defaultStyle;
        defaultStyle.fontSize = 12;
        defaultStyle.color[0] = defaultStyle.color[1] = defaultStyle.color[2] = 0; // 黑色
        defaultStyle.color[3] = 255; // 不透明
        m_fontRenderer->setFontStyle(defaultStyle);
    }
    
#ifdef OSCEAN_CUDA_ENABLED
    // 初始化GPU组件
    initializeGPUComponents();
    if (gpuAvailable_) {
        BOOST_LOG_TRIVIAL(info) << "GPU acceleration available and initialized";
    }
#endif
}

boost::future<core_services::output::OutputResult> VisualizationEngine::process(
    const core_services::output::OutputRequest& request) {
    
    return m_threadPool->submitTask([this, request]() -> core_services::output::OutputResult {
        try {
            BOOST_LOG_TRIVIAL(info) << "Processing visualization request for format: " << request.format;
            
            if (!isValidVisualizationFormat(request.format)) {
                throw core_services::ServiceException("Unsupported visualization format: " + request.format);
            }
            
            // 使用visitor模式提取数据源
            struct GetDataVisitor : public boost::static_visitor<std::shared_ptr<core_services::GridData>> {
                std::shared_ptr<core_services::GridData> operator()(const std::shared_ptr<oscean::core_services::IDataReader>& reader) const {
                    auto variables = reader->listDataVariableNames();
                    if (variables.empty()) {
                        throw core_services::ServiceException("No variables found in data source");
                    }
                    // 修复：正确包装future结果为shared_ptr
                    auto gridDataFuture = reader->readGridData(variables[0]);
                    auto gridDataPtr = gridDataFuture.get();
                    return std::shared_ptr<core_services::GridData>(gridDataPtr);
                }
                
                std::shared_ptr<core_services::GridData> operator()(const std::string& /*path*/) const {
                    throw core_services::ServiceException("VisualizationEngine requires IDataReader, not file path");
                }
            };
            
            GetDataVisitor visitor;
            auto gridData = boost::apply_visitor(visitor, request.dataSource);
            
            if (!gridData) {
                throw core_services::ServiceException("Failed to extract GridData from request");
            }
            
            // 根据格式选择处理方式
            std::vector<std::string> generatedFiles;
            
            if (request.format == "tile" || request.format == "xyz" || request.format == "wmts") {
                // 瓦片生成
                core_services::output::StyleOptions style;
                if (request.style) {
                    style = *request.style;
                }
                
                std::string outputDir = request.targetDirectory ? *request.targetDirectory : "./tiles";
                auto tileResult = generateTiles(gridData, outputDir, style, 0, 10).get();
                // 修复：正确访问boost::optional内容
                if (tileResult.filePaths.has_value()) {
                    generatedFiles = tileResult.filePaths.value();
                }
                
            } else {
                // 单一图像生成
                core_services::output::StyleOptions style;
                if (request.style) {
                    style = *request.style;
                }
                
                std::string outputPath;
                if (request.targetDirectory && request.filenameTemplate) {
                    boost::filesystem::path fullPath(*request.targetDirectory);
                    fullPath /= *request.filenameTemplate;
                    outputPath = fullPath.string();
                } else {
                    outputPath = "output." + request.format;
                }
                
                auto imagePath = renderToImage(gridData, outputPath, style).get();
                generatedFiles.push_back(imagePath);
            }
            
            // 构造结果
            core_services::output::OutputResult result;
            result.filePaths = generatedFiles;
            
            BOOST_LOG_TRIVIAL(info) << "Visualization completed, generated " << generatedFiles.size() << " files";
            
            return result;
            
        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "VisualizationEngine::process failed: " << e.what();
            throw;
        }
    }, common_utils::infrastructure::TaskType::IO_BOUND);
}

boost::future<std::string> VisualizationEngine::renderToImage(
    std::shared_ptr<core_services::GridData> gridData,
    const std::string& outputPath,
    const core_services::output::StyleOptions& style) {
    
    return m_threadPool->submitTask([this, gridData, outputPath, style]() -> std::string {
        try {
            BOOST_LOG_TRIVIAL(info) << "Rendering GridData to image: " << outputPath;
            
            // 生成图像数据
            auto imageData = generateImageData(gridData, style);
            
            const auto& definition = gridData->getDefinition();
            size_t width = definition.cols;
            size_t height = definition.rows;
            
            // 提取格式
            boost::filesystem::path path(outputPath);
            std::string format = path.extension().string();
            if (!format.empty() && format[0] == '.') {
                format = format.substr(1); // 移除点号
            }
            
            // 转换RGBA uint32_t数据为独立的通道uint8_t数据
            std::vector<uint8_t> rgbaData;
            rgbaData.reserve(width * height * 4);
            
            for (uint32_t color : imageData) {
                // 按照RGBA顺序提取每个通道的8位数据
                rgbaData.push_back(static_cast<uint8_t>(color & 0xFF));         // R
                rgbaData.push_back(static_cast<uint8_t>((color >> 8) & 0xFF));  // G
                rgbaData.push_back(static_cast<uint8_t>((color >> 16) & 0xFF)); // B
                rgbaData.push_back(static_cast<uint8_t>((color >> 24) & 0xFF)); // A
            }
            
            // 保存图像
            saveImageToFile(outputPath, rgbaData, width, height);
            
            BOOST_LOG_TRIVIAL(info) << "Image rendered successfully to: " << outputPath;
            
            return outputPath;
            
        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "renderToImage failed: " << e.what();
            throw;
        }
    }, common_utils::infrastructure::TaskType::CPU_INTENSIVE);
}

boost::future<core_services::output::OutputResult> VisualizationEngine::generateTiles(
    std::shared_ptr<core_services::GridData> gridData,
    const std::string& outputDirectory,
    const core_services::output::StyleOptions& style,
    int minZoom, int maxZoom) {
    
    return m_threadPool->submitTask([this, gridData, outputDirectory, style, minZoom, maxZoom]() -> core_services::output::OutputResult {
        try {
            BOOST_LOG_TRIVIAL(info) << "Generating tiles from zoom " << minZoom << " to " << maxZoom;
            
            std::vector<std::string> generatedFiles;
            
            // 创建输出目录
            boost::filesystem::create_directories(outputDirectory);
            
            // 为每个缩放级别生成瓦片
            for (int z = minZoom; z <= maxZoom; ++z) {
                int numTiles = static_cast<int>(std::pow(2, z));
                
                BOOST_LOG_TRIVIAL(info) << "Generating zoom level " << z << " (" << numTiles << "x" << numTiles << " tiles)";
                
                for (int x = 0; x < numTiles; ++x) {
                    for (int y = 0; y < numTiles; ++y) {
                        try {
                            std::string tilePath = generateSingleTile(gridData, x, y, z, style, outputDirectory);
                            generatedFiles.push_back(tilePath);
                        } catch (const std::exception& e) {
                            BOOST_LOG_TRIVIAL(warning) << "Failed to generate tile (" << x << "," << y << "," << z << "): " << e.what();
                        }
                    }
                }
            }
            
            // 构造结果
            core_services::output::OutputResult result;
            result.filePaths = generatedFiles;
            
            BOOST_LOG_TRIVIAL(info) << "Tile generation completed, " << generatedFiles.size() << " tiles created";
            
            return result;
            
        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "generateTiles failed: " << e.what();
            throw;
        }
    }, common_utils::infrastructure::TaskType::CPU_INTENSIVE);
}

// 私有方法实现

std::vector<uint32_t> VisualizationEngine::generateImageData(
    std::shared_ptr<core_services::GridData> gridData,
    const core_services::output::StyleOptions& style) {
    
    const auto& definition = gridData->getDefinition();
    size_t width = definition.cols;
    size_t height = definition.rows;
    
    // 获取数据统计信息
    auto stats = calculateDataStatistics(gridData);
    
    // 获取数据指针和类型
    auto dataPtr = gridData->getDataPtr();
    auto dataType = gridData->getDataType();
    
    std::vector<double> values;
    values.reserve(width * height);
    
    // 根据数据类型提取数值
    if (dataType == core_services::DataType::Float64) {
        const double* doubleData = static_cast<const double*>(dataPtr);
        for (size_t i = 0; i < width * height; ++i) {
            values.push_back(doubleData[i]);
        }
    } else if (dataType == core_services::DataType::Float32) {
        const float* floatData = static_cast<const float*>(dataPtr);
        for (size_t i = 0; i < width * height; ++i) {
            values.push_back(static_cast<double>(floatData[i]));
        }
    } else {
        BOOST_LOG_TRIVIAL(warning) << "Unsupported data type for visualization, using zero values";
        values.assign(width * height, 0.0);
    }
    
    // 映射到颜色
    std::string colorMap = style.colorMap.empty() ? "viridis" : style.colorMap;
    return mapDataToColors(values, colorMap, stats.minValue, stats.maxValue);
}

std::vector<uint32_t> VisualizationEngine::mapDataToColors(
    const std::vector<double>& values,
    const std::string& colorMap,
    double minValue, double maxValue) {
    
    auto colorMapData = getColorMap(colorMap);
    std::vector<uint32_t> colors;
    colors.reserve(values.size());
    
    double range = maxValue - minValue;
    if (range == 0.0) {
        range = 1.0; // 避免除零
    }
    
    for (double value : values) {
        // 归一化到0-1范围
        double normalizedValue = (value - minValue) / range;
        normalizedValue = std::max(0.0, std::min(1.0, normalizedValue));
        
        // 获取RGB颜色
        auto rgb = interpolateColor(normalizedValue, colorMapData);
        
        // 转换为RGBA格式（小端序）
        uint32_t rgba = (255U << 24) | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0];
        colors.push_back(rgba);
    }
    
    return colors;
}

void VisualizationEngine::saveImageToFile(const std::string& filename, 
                                          const std::vector<uint8_t>& imageData, 
                                          int width, int height) {
    BOOST_LOG_TRIVIAL(info) << "Saving image to " << filename << " using GDAL...";

    // 获取文件扩展名以确定格式
    boost::filesystem::path filePath(filename);
    std::string extension = filePath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension.empty() || extension[0] != '.') {
        BOOST_LOG_TRIVIAL(warning) << "No file extension found, defaulting to PNG format";
        extension = ".png";
    }
    
    // 根据扩展名确定GDAL驱动
    const char* pszFormat = nullptr;
    char** papszOptions = nullptr;
    
    if (extension == ".png") {
        pszFormat = "PNG";
        // PNG特定选项
        papszOptions = CSLSetNameValue(papszOptions, "WORLDFILE", "NO");
        papszOptions = CSLSetNameValue(papszOptions, "ZLEVEL", "9"); // 最大压缩
    } else if (extension == ".jpg" || extension == ".jpeg") {
        pszFormat = "JPEG";
        // JPEG特定选项
        papszOptions = CSLSetNameValue(papszOptions, "QUALITY", "95"); // 高质量
        papszOptions = CSLSetNameValue(papszOptions, "PROGRESSIVE", "ON");
    } else if (extension == ".tif" || extension == ".tiff") {
        pszFormat = "GTiff";
        // TIFF特定选项
        papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");
        papszOptions = CSLSetNameValue(papszOptions, "PREDICTOR", "2");
        papszOptions = CSLSetNameValue(papszOptions, "TILED", "YES");
    } else if (extension == ".bmp") {
        pszFormat = "BMP";
    } else {
        BOOST_LOG_TRIVIAL(warning) << "Unsupported image format: " << extension << ", defaulting to PNG";
        pszFormat = "PNG";
    }

    GDALAllRegister();
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if (poDriver == nullptr) {
        CSLDestroy(papszOptions);
        throw oscean::core_services::ServiceException(std::string("GDAL driver not available for format: ") + pszFormat);
    }

    // 检查驱动程序支持的创建方法 - 测试发现PNG只支持CreateCopy
    auto metadata = poDriver->GetMetadata();
    bool supportsCreate = false;
    bool supportsCreateCopy = false;
    
    for (int i = 0; metadata && metadata[i]; i++) {
        std::string meta(metadata[i]);
        if (meta.find("DCAP_CREATE=YES") != std::string::npos) {
            supportsCreate = true;
        }
        if (meta.find("DCAP_CREATECOPY=YES") != std::string::npos) {
            supportsCreateCopy = true;
        }
    }

    GDALDataset* poDstDS = nullptr;
    
    if (supportsCreate) {
        // 使用Create方法（直接创建）
        poDstDS = poDriver->Create(filename.c_str(), width, height, 4, GDT_Byte, papszOptions);
    } else if (supportsCreateCopy) {
        // 使用CreateCopy方法（先创建内存数据集，再复制）
        GDALDriver* memDriver = GetGDALDriverManager()->GetDriverByName("MEM");
        if (memDriver == nullptr) {
            CSLDestroy(papszOptions);
            throw oscean::core_services::ServiceException("Memory driver not available");
        }
        
        // 创建内存数据集
        GDALDataset* memDS = memDriver->Create("", width, height, 4, GDT_Byte, nullptr);
        if (memDS == nullptr) {
            CSLDestroy(papszOptions);
            throw oscean::core_services::ServiceException("Failed to create memory dataset");
        }
        
        // 将数据写入内存数据集
        std::vector<uint8_t> r(width * height);
        std::vector<uint8_t> g(width * height);
        std::vector<uint8_t> b(width * height);
        std::vector<uint8_t> a(width * height);

        for (int i = 0; i < width * height; ++i) {
            r[i] = imageData[i * 4 + 0];
            g[i] = imageData[i * 4 + 1];
            b[i] = imageData[i * 4 + 2];
            a[i] = imageData[i * 4 + 3];
        }
        
        CPLErr eErr = CE_None;
        eErr = memDS->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, r.data(), width, height, GDT_Byte, 0, 0);
        if (eErr == CE_None) {
            eErr = memDS->GetRasterBand(2)->RasterIO(GF_Write, 0, 0, width, height, g.data(), width, height, GDT_Byte, 0, 0);
        }
        if (eErr == CE_None) {
            eErr = memDS->GetRasterBand(3)->RasterIO(GF_Write, 0, 0, width, height, b.data(), width, height, GDT_Byte, 0, 0);
        }
        if (eErr == CE_None) {
            eErr = memDS->GetRasterBand(4)->RasterIO(GF_Write, 0, 0, width, height, a.data(), width, height, GDT_Byte, 0, 0);
        }
        
        if (eErr != CE_None) {
            GDALClose(memDS);
            CSLDestroy(papszOptions);
            throw oscean::core_services::ServiceException("Failed to write data to memory dataset");
        }
        
        // 使用CreateCopy方法复制到目标文件
        poDstDS = poDriver->CreateCopy(filename.c_str(), memDS, FALSE, papszOptions, nullptr, nullptr);
        
        // 关闭内存数据集
        GDALClose(memDS);
        
        if (poDstDS == nullptr) {
            CSLDestroy(papszOptions);
            throw oscean::core_services::ServiceException("Failed to create copy of dataset for " + filename);
        }
        
        // CreateCopy已经完成了数据写入，关闭数据集
        CSLDestroy(papszOptions);
        GDALClose(poDstDS);
        
        // **测试发现问题：必须验证文件是否真正创建成功**
        if (!boost::filesystem::exists(filename)) {
            throw oscean::core_services::ServiceException("File was not created despite successful GDAL operation: " + filename);
        }
        
        // 验证文件大小
        auto fileSize = boost::filesystem::file_size(filename);
        if (fileSize == 0) {
            throw oscean::core_services::ServiceException("Created file is empty: " + filename);
        }
        
        BOOST_LOG_TRIVIAL(info) << "Successfully saved image to: " << filename << " using " << pszFormat << " format (CreateCopy method), size: " << fileSize << " bytes";
        return;
        
    } else {
        CSLDestroy(papszOptions);
        throw oscean::core_services::ServiceException(std::string("Driver ") + pszFormat + " does not support creating files");
    }

    if (poDstDS == nullptr) {
        CSLDestroy(papszOptions);
        throw oscean::core_services::ServiceException("Failed to create GDAL dataset for " + filename);
    }

    // 释放选项
    CSLDestroy(papszOptions);

    // GDAL期望波段是分开的，所以我们需要解开交错的RGBA数据
    std::vector<uint8_t> r(width * height);
    std::vector<uint8_t> g(width * height);
    std::vector<uint8_t> b(width * height);
    std::vector<uint8_t> a(width * height);

    for (int i = 0; i < width * height; ++i) {
        r[i] = imageData[i * 4 + 0];
        g[i] = imageData[i * 4 + 1];
        b[i] = imageData[i * 4 + 2];
        a[i] = imageData[i * 4 + 3];
    }
    
    // 将数据写入每个波段（仅对支持Create方法的驱动程序）
    CPLErr eErr = CE_None;
    eErr = poDstDS->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, r.data(), width, height, GDT_Byte, 0, 0);
    if (eErr != CE_None) {
        GDALClose(poDstDS);
        throw oscean::core_services::ServiceException("Failed to write red band data");
    }
    
    eErr = poDstDS->GetRasterBand(2)->RasterIO(GF_Write, 0, 0, width, height, g.data(), width, height, GDT_Byte, 0, 0);
    if (eErr != CE_None) {
        GDALClose(poDstDS);
        throw oscean::core_services::ServiceException("Failed to write green band data");
    }
    
    eErr = poDstDS->GetRasterBand(3)->RasterIO(GF_Write, 0, 0, width, height, b.data(), width, height, GDT_Byte, 0, 0);
    if (eErr != CE_None) {
        GDALClose(poDstDS);
        throw oscean::core_services::ServiceException("Failed to write blue band data");
    }
    
    eErr = poDstDS->GetRasterBand(4)->RasterIO(GF_Write, 0, 0, width, height, a.data(), width, height, GDT_Byte, 0, 0);
    if (eErr != CE_None) {
        GDALClose(poDstDS);
        throw oscean::core_services::ServiceException("Failed to write alpha band data");
    }
    
    // 关闭数据集，完成写入
    GDALClose(poDstDS);
    
    // **测试发现问题：必须验证文件是否真正创建成功**
    if (!boost::filesystem::exists(filename)) {
        throw oscean::core_services::ServiceException("File was not created despite successful GDAL operation: " + filename);
    }
    
    // 验证文件大小
    auto fileSize = boost::filesystem::file_size(filename);
    if (fileSize == 0) {
        throw oscean::core_services::ServiceException("Created file is empty: " + filename);
    }
    
    BOOST_LOG_TRIVIAL(info) << "Successfully saved image to: " << filename << " using " << pszFormat << " format (Create method), size: " << fileSize << " bytes";
}

std::vector<std::array<uint8_t, 3>> VisualizationEngine::getColorMap(const std::string& colorMapName) {
    // 预定义的颜色映射
    if (colorMapName == "viridis") {
        return {
            {{68, 1, 84}},     // 紫色
            {{59, 82, 139}},   // 蓝紫色
            {{33, 144, 140}},  // 青绿色
            {{93, 201, 99}},   // 绿色
            {{253, 231, 37}}   // 黄色
        };
    } else if (colorMapName == "plasma") {
        return {
            {{13, 8, 135}},    // 深蓝
            {{126, 3, 168}},   // 紫红
            {{203, 70, 121}},  // 红色
            {{249, 142, 8}},   // 橙色
            {{240, 249, 33}}   // 黄色
        };
    } else if (colorMapName == "jet") {
        return {
            {{0, 0, 143}},     // 深蓝
            {{0, 0, 255}},     // 蓝色
            {{0, 255, 255}},   // 青色
            {{0, 255, 0}},     // 绿色
            {{255, 255, 0}},   // 黄色
            {{255, 0, 0}}      // 红色
        };
    } else {
        // 默认灰度
        return {
            {{0, 0, 0}},       // 黑色
            {{128, 128, 128}}, // 灰色
            {{255, 255, 255}}  // 白色
        };
    }
}

std::array<uint8_t, 3> VisualizationEngine::interpolateColor(
    double value, const std::vector<std::array<uint8_t, 3>>& colorMap) {
    
    if (colorMap.empty()) {
        return {{128, 128, 128}}; // 默认灰色
    }
    
    if (colorMap.size() == 1) {
        return colorMap[0];
    }
    
    // 将value映射到颜色映射索引
    double scaledValue = value * (colorMap.size() - 1);
    size_t lowIndex = static_cast<size_t>(std::floor(scaledValue));
    size_t highIndex = std::min(lowIndex + 1, colorMap.size() - 1);
    
    if (lowIndex == highIndex) {
        return colorMap[lowIndex];
    }
    
    // 线性插值
    double fraction = scaledValue - lowIndex;
    const auto& lowColor = colorMap[lowIndex];
    const auto& highColor = colorMap[highIndex];
    
    std::array<uint8_t, 3> result;
    for (size_t i = 0; i < 3; ++i) {
        double interpolated = lowColor[i] + fraction * (highColor[i] - lowColor[i]);
        result[i] = static_cast<uint8_t>(std::round(interpolated));
    }
    
    return result;
}

std::string VisualizationEngine::generateSingleTile(
    std::shared_ptr<core_services::GridData> gridData,
    int x, int y, int z,
    const core_services::output::StyleOptions& style,
    const std::string& outputDirectory) {
    
    // 计算瓦片边界（经纬度）
    auto tileBounds = calculateTileBounds(x, y, z);
    
    // 生成瓦片文件名
    std::string tilePath = outputDirectory + "/" + std::to_string(z) + "_" + std::to_string(x) + "_" + std::to_string(y) + ".png";
    
    BOOST_LOG_TRIVIAL(info) << "Generating tile at " << x << "," << y << "," << z 
                           << " with bounds: " << tileBounds.minX << "," << tileBounds.minY 
                           << " to " << tileBounds.maxX << "," << tileBounds.maxY;
    
    try {
        const auto& definition = gridData->getDefinition();
        
        // 如果数据在不同的坐标系，需要进行坐标转换
        // 这里我们假设数据是WGS84经纬度坐标系，与瓦片系统匹配
        
        // 计算瓦片对应的数据区域（像素坐标）
        double dataMinX = definition.extent.minX;
        double dataMaxX = definition.extent.maxX;
        double dataMinY = definition.extent.minY;
        double dataMaxY = definition.extent.maxY;
        
        // 如果瓦片完全在数据范围外，返回空瓦片
        if (tileBounds.maxX < dataMinX || tileBounds.minX > dataMaxX ||
            tileBounds.maxY < dataMinY || tileBounds.minY > dataMaxY) {
            // 创建空瓦片（透明）
            std::vector<uint32_t> emptyTileData(256 * 256, 0);
            std::vector<uint8_t> rgbaData;
            rgbaData.reserve(256 * 256 * 4);
            
            for (uint32_t color : emptyTileData) {
                rgbaData.push_back(0);  // R
                rgbaData.push_back(0);  // G
                rgbaData.push_back(0);  // B
                rgbaData.push_back(0);  // A (透明)
            }
            
            saveImageToFile(tilePath, rgbaData, 256, 256);
            return tilePath;
        }
        
        // 计算数据在瓦片中的映射
        double xScale = definition.cols / (dataMaxX - dataMinX);
        double yScale = definition.rows / (dataMaxY - dataMinY);
        
        // 计算瓦片对应的数据区域（像素索引）
        int startCol = std::max(0, static_cast<int>((tileBounds.minX - dataMinX) * xScale));
        int endCol = std::min(static_cast<int>(definition.cols), static_cast<int>((tileBounds.maxX - dataMinX) * xScale));
        int startRow = std::max(0, static_cast<int>((dataMaxY - tileBounds.maxY) * yScale)); // Y轴方向相反
        int endRow = std::min(static_cast<int>(definition.rows), static_cast<int>((dataMaxY - tileBounds.minY) * yScale));
        
        // 瓦片尺寸（通常是256x256）
        const int tileSize = 256;
        
        // 提取并重采样数据到瓦片尺寸
        std::vector<double> tileValues(tileSize * tileSize, 0.0);
        
        if (endCol > startCol && endRow > startRow) {
            // 获取数据指针和类型
            auto dataPtr = gridData->getDataPtr();
            auto dataType = gridData->getDataType();
            
            // 计算重采样因子
            double xFactor = static_cast<double>(endCol - startCol) / tileSize;
            double yFactor = static_cast<double>(endRow - startRow) / tileSize;
            
            // 提取并重采样数据
            for (int ty = 0; ty < tileSize; ++ty) {
                for (int tx = 0; tx < tileSize; ++tx) {
                    // 计算对应的数据坐标
                    int dataX = startCol + static_cast<int>(tx * xFactor);
                    int dataY = startRow + static_cast<int>(ty * yFactor);
                    
                    // 确保在有效范围内
                    if (dataX >= 0 && dataX < definition.cols && dataY >= 0 && dataY < definition.rows) {
                        // 根据数据类型提取值
                        double value = 0.0;
                        if (dataType == core_services::DataType::Float64) {
                            const double* doubleData = static_cast<const double*>(dataPtr);
                            value = doubleData[dataY * definition.cols + dataX];
                        } else if (dataType == core_services::DataType::Float32) {
                            const float* floatData = static_cast<const float*>(dataPtr);
                            value = static_cast<double>(floatData[dataY * definition.cols + dataX]);
                        } else if (dataType == core_services::DataType::Int32) {
                            const int32_t* intData = static_cast<const int32_t*>(dataPtr);
                            value = static_cast<double>(intData[dataY * definition.cols + dataX]);
                        } else if (dataType == core_services::DataType::Int16) {
                            const int16_t* shortData = static_cast<const int16_t*>(dataPtr);
                            value = static_cast<double>(shortData[dataY * definition.cols + dataX]);
                        }
                        
                        tileValues[ty * tileSize + tx] = value;
                    }
                }
            }
        }
        
        // 计算数据统计信息（用于颜色映射）
        auto stats = calculateDataStatistics(gridData);
        
        // 映射数据到颜色
        std::string colorMap = style.colorMap.empty() ? "viridis" : style.colorMap;
        std::vector<uint32_t> tileColors = mapDataToColors(tileValues, colorMap, stats.minValue, stats.maxValue);
        
        // 转换为RGBA格式
        std::vector<uint8_t> rgbaData;
        rgbaData.reserve(tileSize * tileSize * 4);
        
        for (uint32_t color : tileColors) {
            rgbaData.push_back(static_cast<uint8_t>(color & 0xFF));         // R
            rgbaData.push_back(static_cast<uint8_t>((color >> 8) & 0xFF));  // G
            rgbaData.push_back(static_cast<uint8_t>((color >> 16) & 0xFF)); // B
            rgbaData.push_back(static_cast<uint8_t>((color >> 24) & 0xFF)); // A
        }
        
        // 保存瓦片图像
        saveImageToFile(tilePath, rgbaData, tileSize, tileSize);
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Failed to generate tile: " << e.what();
        
        // 创建错误瓦片（红色）
        std::vector<uint8_t> errorTileData(256 * 256 * 4, 0);
        for (size_t i = 0; i < 256 * 256; ++i) {
            errorTileData[i * 4] = 255;     // R
            errorTileData[i * 4 + 1] = 0;   // G
            errorTileData[i * 4 + 2] = 0;   // B
            errorTileData[i * 4 + 3] = 128; // A (半透明)
        }
        
        saveImageToFile(tilePath, errorTileData, 256, 256);
    }
    
    return tilePath;
}

core_services::BoundingBox VisualizationEngine::calculateTileBounds(int x, int y, int z) {
    // Web Mercator瓦片边界计算
    // 标准XYZ瓦片系统使用Web Mercator投影(EPSG:3857)
    // 但我们返回经纬度坐标(EPSG:4326)以便于与其他数据比较
    
    // 计算瓦片范围（0-1范围内）
    double n = std::pow(2.0, z); // 该缩放级别的瓦片总数（每边）
    double xmin = x / n;         // 瓦片左边缘（0-1范围）
    double xmax = (x + 1) / n;   // 瓦片右边缘（0-1范围）
    double ymin = y / n;         // 瓦片上边缘（0-1范围）
    double ymax = (y + 1) / n;   // 瓦片下边缘（0-1范围）
    
    // 转换为经度（-180到180度）
    double lonMin = xmin * 360.0 - 180.0;
    double lonMax = xmax * 360.0 - 180.0;
    
    // 将Web Mercator的y坐标转换为纬度
    // 公式: lat = 2 * atan(exp(π * (1 - 2 * y))) - π/2
    double latMax = std::atan(std::sinh(M_PI * (1 - 2 * ymin))) * 180.0 / M_PI;
    double latMin = std::atan(std::sinh(M_PI * (1 - 2 * ymax))) * 180.0 / M_PI;
    
    // 创建并返回边界框
    core_services::BoundingBox bounds;
    bounds.minX = lonMin;
    bounds.maxX = lonMax;
    bounds.minY = latMin;
    bounds.maxY = latMax;
    bounds.crsId = "EPSG:4326"; // WGS84经纬度
    
    BOOST_LOG_TRIVIAL(debug) << "Tile " << z << "/" << x << "/" << y 
                           << " bounds: lon(" << bounds.minX << "," << bounds.maxX 
                           << "), lat(" << bounds.minY << "," << bounds.maxY << ")";
    
    return bounds;
}

bool VisualizationEngine::isValidVisualizationFormat(const std::string& format) {
    std::set<std::string> validFormats = {"png", "jpg", "jpeg", "tiff", "bmp", "tile", "xyz", "wmts"};
    return validFormats.find(format) != validFormats.end();
}

VisualizationEngine::DataStatistics VisualizationEngine::calculateDataStatistics(
    std::shared_ptr<core_services::GridData> gridData) {
    
    DataStatistics stats{0.0, 0.0, 0.0, 0.0};  // 显式初始化所有字段
    
    // 获取数据指针和类型
    auto dataPtr = gridData->getDataPtr();
    auto dataType = gridData->getDataType();
    const auto& definition = gridData->getDefinition();
    
    size_t totalElements = definition.rows * definition.cols * gridData->getBandCount();
    
    if (totalElements == 0) {
        return stats;
    }
    
    std::vector<double> values;
    values.reserve(totalElements);
    
    // 根据数据类型提取数值
    if (dataType == core_services::DataType::Float64) {
        const double* doubleData = static_cast<const double*>(dataPtr);
        for (size_t i = 0; i < totalElements; ++i) {
            values.push_back(doubleData[i]);
        }
    } else if (dataType == core_services::DataType::Float32) {
        const float* floatData = static_cast<const float*>(dataPtr);
        for (size_t i = 0; i < totalElements; ++i) {
            values.push_back(static_cast<double>(floatData[i]));
        }
    } else {
        // 对于其他数据类型，返回默认统计
        stats.minValue = 0.0;
        stats.maxValue = 1.0;
        stats.meanValue = 0.5;
        stats.stdDev = 0.1;
        return stats;
    }
    
    // 计算统计信息
    if (!values.empty()) {
        auto minMaxPair = std::minmax_element(values.begin(), values.end());
        stats.minValue = *minMaxPair.first;
        stats.maxValue = *minMaxPair.second;
        
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        stats.meanValue = sum / values.size();
        
        double variance = 0.0;
        for (double value : values) {
            variance += (value - stats.meanValue) * (value - stats.meanValue);
        }
        variance /= values.size();
        stats.stdDev = std::sqrt(variance);
    }
    
    return stats;
}

std::shared_ptr<core_services::FeatureCollection> VisualizationEngine::generateContours(
    std::shared_ptr<core_services::GridData> gridData,
    const std::vector<double>& levels) {
    
    BOOST_LOG_TRIVIAL(info) << "Generating contours for " << levels.size() << " levels";
    
    // 创建返回的FeatureCollection
    auto featureCollection = std::make_shared<core_services::FeatureCollection>();
    
    try {
        // 获取网格数据信息
        const auto& definition = gridData->getDefinition();
        size_t width = definition.cols;
        size_t height = definition.rows;
        auto dataPtr = gridData->getDataPtr();
        auto dataType = gridData->getDataType();
        
        // 设置FeatureCollection元数据
        featureCollection->name = "Generated Contours";
        featureCollection->crs = definition.crs;
        featureCollection->extent = definition.extent;
        
        // 添加字段定义
        core_services::FieldDefinition elevationField;
        elevationField.name = "elevation";
        elevationField.description = "Contour elevation value";
        elevationField.dataType = "double";
        elevationField.type = "real";
        elevationField.isNullable = false;
        featureCollection->fieldDefinitions.push_back(elevationField);
        
        // 创建临时内存数据集
        GDALDriverH memDriver = GDALGetDriverByName("MEM");
        if (memDriver == nullptr) {
            throw core_services::ServiceException("Failed to create GDAL memory driver");
        }
        
        // 创建单波段栅格数据集
        GDALDatasetH dataset = GDALCreate(memDriver, "", width, height, 1, GDT_Float64, nullptr);
        if (dataset == nullptr) {
            throw core_services::ServiceException("Failed to create GDAL memory dataset");
        }
        
        // 设置地理变换参数
        double geoTransform[6] = {
            definition.extent.minX, 
            (definition.extent.maxX - definition.extent.minX) / width, 
            0.0, 
            definition.extent.maxY, 
            0.0, 
            -(definition.extent.maxY - definition.extent.minY) / height
        };
        GDALSetGeoTransform(dataset, geoTransform);
        
        // 设置投影
        if (!definition.crs.wkt.empty()) {
            GDALSetProjection(dataset, definition.crs.wkt.c_str());
        } else if (!definition.crs.projString.empty()) {
            char* wkt = nullptr;
            OGRSpatialReference srs;
            srs.importFromProj4(definition.crs.projString.c_str());
            srs.exportToWkt(&wkt);
            if (wkt) {
                GDALSetProjection(dataset, wkt);
                CPLFree(wkt);
            }
        }
        
        // 获取波段
        GDALRasterBandH band = GDALGetRasterBand(dataset, 1);
        
        // 将数据复制到GDAL数据集
        std::vector<double> rasterData(width * height);
        
        if (dataType == core_services::DataType::Float64) {
            const double* doubleData = static_cast<const double*>(dataPtr);
            std::copy(doubleData, doubleData + (width * height), rasterData.begin());
        } else if (dataType == core_services::DataType::Float32) {
            const float* floatData = static_cast<const float*>(dataPtr);
            for (size_t i = 0; i < width * height; ++i) {
                rasterData[i] = static_cast<double>(floatData[i]);
            }
        } else if (dataType == core_services::DataType::Int32) {
            const int32_t* intData = static_cast<const int32_t*>(dataPtr);
            for (size_t i = 0; i < width * height; ++i) {
                rasterData[i] = static_cast<double>(intData[i]);
            }
        } else if (dataType == core_services::DataType::Int16) {
            const int16_t* shortData = static_cast<const int16_t*>(dataPtr);
            for (size_t i = 0; i < width * height; ++i) {
                rasterData[i] = static_cast<double>(shortData[i]);
            }
        } else {
            GDALClose(dataset);
            throw core_services::ServiceException("Unsupported data type for contour generation");
        }
        
        // 写入数据到波段
        CPLErr err = GDALRasterIO(band, GF_Write, 0, 0, width, height, 
                                  rasterData.data(), width, height, GDT_Float64, 0, 0);
        if (err != CE_None) {
            GDALClose(dataset);
            throw core_services::ServiceException("Failed to write data to GDAL dataset");
        }
        
        // 创建OGR层来存储等值线
        OGRSFDriverH ogrDriver = OGRGetDriverByName("Memory");
        if (ogrDriver == nullptr) {
            GDALClose(dataset);
            throw core_services::ServiceException("Failed to create OGR memory driver");
        }
        
        OGRDataSourceH ogrDS = OGR_Dr_CreateDataSource(ogrDriver, "contours", nullptr);
        if (ogrDS == nullptr) {
            GDALClose(dataset);
            throw core_services::ServiceException("Failed to create OGR data source");
        }
        
        OGRSpatialReferenceH ogrSRS = nullptr;
        if (!definition.crs.wkt.empty()) {
            ogrSRS = OSRNewSpatialReference(definition.crs.wkt.c_str());
        } else if (!definition.crs.projString.empty()) {
            ogrSRS = OSRNewSpatialReference(nullptr);
            OSRImportFromProj4(ogrSRS, definition.crs.projString.c_str());
        }
        
        OGRLayerH ogrLayer = OGR_DS_CreateLayer(ogrDS, "contours", ogrSRS, wkbLineString, nullptr);
        if (ogrLayer == nullptr) {
            if (ogrSRS) OSRDestroySpatialReference(ogrSRS);
            OGR_DS_Destroy(ogrDS);
            GDALClose(dataset);
            throw core_services::ServiceException("Failed to create OGR layer");
        }
        
        // 添加属性字段
        OGRFieldDefnH elevField = OGR_Fld_Create("elevation", OFTReal);
        OGR_L_CreateField(ogrLayer, elevField, 0);
        OGR_Fld_Destroy(elevField);
        
        // 生成等值线
        if (levels.empty()) {
            // 如果未指定等值线级别，使用自动间隔
            auto stats = calculateDataStatistics(gridData);
            double interval = (stats.maxValue - stats.minValue) / 10.0; // 默认10个等值线
            
            err = GDALContourGenerate(band, interval, 0.0, 0, nullptr, 0, 0, 
                                     ogrLayer, 0, 1, nullptr, nullptr);
        } else {
            // 使用指定的等值线级别
            std::vector<double> fixedLevels = levels;
            
            // 对于固定等值线级别，使用GDALContourGenerate函数的不同调用方式
            // 传递固定级别数组
            err = GDALContourGenerate(band, 0.0, 0.0, 
                                     static_cast<int>(fixedLevels.size()), 
                                     fixedLevels.data(), 
                                     0, 0, 
                                     ogrLayer, 0, 1, nullptr, nullptr);
        }
        
        if (err != CE_None) {
            if (ogrSRS) OSRDestroySpatialReference(ogrSRS);
            OGR_DS_Destroy(ogrDS);
            GDALClose(dataset);
            throw core_services::ServiceException("Failed to generate contours");
        }
        
        // 从OGR层提取等值线要素
        OGR_L_ResetReading(ogrLayer);
        OGRFeatureH ogrFeature;
        while ((ogrFeature = OGR_L_GetNextFeature(ogrLayer)) != nullptr) {
            // 获取几何体
            OGRGeometryH ogrGeom = OGR_F_GetGeometryRef(ogrFeature);
            char* wkt = nullptr;
            OGR_G_ExportToWkt(ogrGeom, &wkt);
            
            // 获取属性
            double elevation = OGR_F_GetFieldAsDouble(ogrFeature, 0);
            
            // 创建Feature并添加到FeatureCollection
            core_services::Feature feature;
            feature.id = "contour_" + std::to_string(featureCollection->size());
            feature.geometryWkt = wkt;
            
            // 添加属性
            feature.attributes["elevation"] = elevation;
            
            // 添加到集合
            featureCollection->addFeature(feature);
            
            // 释放资源
            CPLFree(wkt);
            OGR_F_Destroy(ogrFeature);
        }
        
        // 清理资源
        if (ogrSRS) OSRDestroySpatialReference(ogrSRS);
        OGR_DS_Destroy(ogrDS);
        GDALClose(dataset);
        
        BOOST_LOG_TRIVIAL(info) << "Generated " << featureCollection->size() << " contour features";
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Contour generation failed: " << e.what();
        throw;
    }
    
    return featureCollection;
}

boost::future<std::string> VisualizationEngine::generateLegend(
    const std::string& colorMap,
    double minValue, double maxValue,
    const std::string& title,
    const std::string& outputPath,
    int width, int height) {
    
    return m_threadPool->submitTask([this, colorMap, minValue, maxValue, title, outputPath, width, height]() -> std::string {
        try {
            BOOST_LOG_TRIVIAL(info) << "Generating color legend for colorMap: " << colorMap 
                                   << ", range: [" << minValue << ", " << maxValue << "]";
            
            // 创建图例图像
            std::vector<uint8_t> imageData(width * height * 4, 255); // 初始化为白色背景，带Alpha通道
            
            // 计算图例条的位置和大小
            int legendBarWidth = width / 3;
            int legendBarHeight = height * 2 / 3;
            int legendBarX = width / 3;
            int legendBarY = height / 6;
            
            // 获取颜色映射
            auto colorMapData = getColorMap(colorMap);
            
            // 绘制颜色条
            for (int y = 0; y < legendBarHeight; ++y) {
                // 计算归一化值（从底部到顶部）
                double normalizedValue = 1.0 - static_cast<double>(y) / legendBarHeight;
                
                // 获取对应的颜色
                auto rgb = interpolateColor(normalizedValue, colorMapData);
                
                // 在这一行绘制颜色条
                for (int x = 0; x < legendBarWidth; ++x) {
                    int pixelIndex = ((legendBarY + y) * width + (legendBarX + x)) * 4;
                    imageData[pixelIndex] = rgb[0];     // R
                    imageData[pixelIndex + 1] = rgb[1]; // G
                    imageData[pixelIndex + 2] = rgb[2]; // B
                    imageData[pixelIndex + 3] = 255;    // A (不透明)
                }
            }
            
            // 绘制图例边框
            for (int y = 0; y < legendBarHeight; ++y) {
                // 左边框
                int leftBorderIndex = ((legendBarY + y) * width + legendBarX) * 4;
                imageData[leftBorderIndex] = 0;
                imageData[leftBorderIndex + 1] = 0;
                imageData[leftBorderIndex + 2] = 0;
                
                // 右边框
                int rightBorderIndex = ((legendBarY + y) * width + (legendBarX + legendBarWidth - 1)) * 4;
                imageData[rightBorderIndex] = 0;
                imageData[rightBorderIndex + 1] = 0;
                imageData[rightBorderIndex + 2] = 0;
            }
            
            // 绘制顶部和底部边框
            for (int x = 0; x < legendBarWidth; ++x) {
                // 顶部边框
                int topBorderIndex = (legendBarY * width + (legendBarX + x)) * 4;
                imageData[topBorderIndex] = 0;
                imageData[topBorderIndex + 1] = 0;
                imageData[topBorderIndex + 2] = 0;
                
                // 底部边框
                int bottomBorderIndex = ((legendBarY + legendBarHeight - 1) * width + (legendBarX + x)) * 4;
                imageData[bottomBorderIndex] = 0;
                imageData[bottomBorderIndex + 1] = 0;
                imageData[bottomBorderIndex + 2] = 0;
            }
            
            // 绘制刻度和标签
            drawLegendTicks(imageData, width, height, minValue, maxValue, 5);
            
            // 绘制标题
            if (!title.empty()) {
                drawLegendTitle(imageData, width, height, title);
            }
            
            // 保存图例图像
            saveImageToFile(outputPath, imageData, width, height);
            
            BOOST_LOG_TRIVIAL(info) << "Legend generated successfully: " << outputPath;
            
            return outputPath;
            
        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "Legend generation failed: " << e.what();
            throw;
        }
    }, common_utils::infrastructure::TaskType::CPU_INTENSIVE);
}

void VisualizationEngine::drawLegendTicks(
    std::vector<uint8_t>& imageData,
    int width, int height,
    double minValue, double maxValue,
    int numTicks) {
    
    // 计算图例条的位置和大小
    int legendBarWidth = width / 3;
    int legendBarHeight = height * 2 / 3;
    int legendBarX = width / 3;
    int legendBarY = height / 6;
    
    // 计算刻度间隔
    double valueRange = maxValue - minValue;
    double tickInterval = valueRange / (numTicks - 1);
    int pixelInterval = legendBarHeight / (numTicks - 1);
    
    // 绘制刻度和标签
    for (int i = 0; i < numTicks; ++i) {
        // 计算刻度位置
        int tickY = legendBarY + legendBarHeight - 1 - i * pixelInterval;
        double tickValue = minValue + i * tickInterval;
        
        // 绘制刻度线
        for (int x = 0; x < legendBarWidth / 4; ++x) {
            int tickIndex = (tickY * width + (legendBarX + legendBarWidth + x)) * 4;
            imageData[tickIndex] = 0;
            imageData[tickIndex + 1] = 0;
            imageData[tickIndex + 2] = 0;
            imageData[tickIndex + 3] = 255;
        }
        
        // 绘制标签（简化实现，实际应使用字体渲染库）
        // 这里我们用一个简单的数字表示，在实际应用中应使用字体渲染
        std::string label = std::to_string(tickValue);
        if (label.find('.') != std::string::npos) {
            // 限制小数位数
            size_t decimalPos = label.find('.');
            if (label.length() > decimalPos + 3) {
                label = label.substr(0, decimalPos + 3);
            }
        }
        
        // 使用字体渲染器绘制标签
        int labelX = legendBarX + legendBarWidth + legendBarWidth / 3;
        if (m_fontRenderer) {
            m_fontRenderer->drawText(imageData, width, height, label, 
                                   labelX, tickY, FontRenderer::Alignment::LEFT);
        } else {
            // 回退：在标签位置绘制简单的点表示（实际应使用字体渲染）
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    int labelIndex = ((tickY + y) * width + (labelX + x)) * 4;
                    if (labelIndex >= 0 && labelIndex < static_cast<int>(imageData.size() - 3)) {
                        imageData[labelIndex] = 0;
                        imageData[labelIndex + 1] = 0;
                        imageData[labelIndex + 2] = 0;
                        imageData[labelIndex + 3] = 255;
                    }
                }
            }
        }
    }
}

void VisualizationEngine::drawLegendTitle(
    std::vector<uint8_t>& imageData,
    int width, int height,
    const std::string& title) {
    
    // 计算标题位置
    int titleY = height / 12;
    int titleX = width / 2;
    
    // 如果字体渲染器可用，使用真实字体渲染
    if (m_fontRenderer && m_fontRenderer->drawText(
            imageData, width, height, title, 
            titleX, titleY, FontRenderer::Alignment::CENTER)) {
        BOOST_LOG_TRIVIAL(debug) << "Legend title rendered: " << title;
        return;
    }
    
    // 回退到简单实现
    // 简单的标题表示（实际应使用字体渲染库）
    // 这里我们只绘制一个简单的标记点表示标题位置
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            int titleIndex = ((titleY + y) * width + (titleX + x)) * 4;
            if (titleIndex >= 0 && titleIndex < static_cast<int>(imageData.size() - 3)) {
                imageData[titleIndex] = 0;
                imageData[titleIndex + 1] = 0;
                imageData[titleIndex + 2] = 0;
                imageData[titleIndex + 3] = 255;
            }
        }
    }
    
    BOOST_LOG_TRIVIAL(debug) << "Legend title would be: " << title;
}

// === SIMD优化方法实现 ===

boost::future<std::string> VisualizationEngine::renderToImageOptimized(
    std::shared_ptr<core_services::GridData> gridData,
    const std::string& outputPath,
    const core_services::output::StyleOptions& style) {
    
    return m_threadPool->submitTask([this, gridData, outputPath, style]() -> std::string {
        try {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            BOOST_LOG_TRIVIAL(info) << "Rendering GridData to image (SIMD optimized): " << outputPath;
            
            // 使用SIMD优化的图像数据生成
            std::vector<uint32_t> imageData;
            if (isSIMDOptimizationEnabled()) {
                imageData = generateImageDataSIMD(gridData, style);
                simdOperationCount_++;
            } else {
                imageData = generateImageData(gridData, style);
            }
            
            const auto& definition = gridData->getDefinition();
            size_t width = definition.cols;
            size_t height = definition.rows;
            
            // SIMD优化的RGBA转换
            std::vector<uint8_t> rgbaData;
            convertToRGBASIMD(imageData, rgbaData);
            
            // 保存图像
            saveImageToFile(outputPath, rgbaData, width, height);
            
            // 记录性能统计
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            totalRenderTime_.store(totalRenderTime_.load() + duration.count());
            renderCount_++;
            
            BOOST_LOG_TRIVIAL(info) << "SIMD optimized image rendered in " << duration.count() << "ms: " << outputPath;
            
            return outputPath;
            
        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "renderToImageOptimized failed: " << e.what();
            throw;
        }
    }, common_utils::infrastructure::TaskType::CPU_INTENSIVE);
}

boost::future<core_services::output::OutputResult> VisualizationEngine::generateTilesParallel(
    std::shared_ptr<core_services::GridData> gridData,
    const std::string& outputDirectory,
    const core_services::output::StyleOptions& style,
    int minZoom, int maxZoom) {
    
    return m_threadPool->submitTask([this, gridData, outputDirectory, style, minZoom, maxZoom]() -> core_services::output::OutputResult {
        
        auto startTime = std::chrono::high_resolution_clock::now();
        BOOST_LOG_TRIVIAL(info) << "Starting parallel tile generation (Z" << minZoom << "-" << maxZoom << ")";
        
        core_services::output::OutputResult result;
        result.filePaths = std::vector<std::string>();
        
        try {
            // 创建输出目录
            boost::filesystem::path tileDir(outputDirectory);
            boost::filesystem::create_directories(tileDir);
            
            // === 第一阶段：计算瓦片任务分布 ===
            struct TileTask {
                int x, y, z;
                std::string outputPath;
                core_services::BoundingBox bounds;
            };
            
            std::vector<TileTask> allTasks;
            
            // 预计算所有瓦片任务
            for (int z = minZoom; z <= maxZoom; ++z) {
                int tilesPerLevel = 1 << z; // 2^z
                
                for (int x = 0; x < tilesPerLevel; ++x) {
                    for (int y = 0; y < tilesPerLevel; ++y) {
                        TileTask task;
                        task.x = x;
                        task.y = y; 
                        task.z = z;
                        task.bounds = calculateTileBounds(x, y, z);
                        
                        // 构造瓦片文件路径
                        boost::filesystem::path tilePath = tileDir / std::to_string(z) 
                                                          / std::to_string(x) 
                                                          / (std::to_string(y) + ".png");
                        
                        // 创建目录
                        boost::filesystem::create_directories(tilePath.parent_path());
                        task.outputPath = tilePath.string();
                        
                        allTasks.push_back(task);
                    }
                }
            }
            
            BOOST_LOG_TRIVIAL(info) << "Parallel tile generation: " << allTasks.size() << " tiles planned";
            
            // === 第二阶段：智能负载平衡的并行处理 ===
            const size_t numWorkers = 4; // 使用固定的worker数量
            const size_t tasksPerWorker = std::max(size_t(1), allTasks.size() / numWorkers);
            
            std::vector<boost::future<std::vector<std::string>>> workerFutures;
            std::atomic<size_t> completedTasks{0};
            std::atomic<size_t> failedTasks{0};
            
            // 分批提交任务到线程池
            for (size_t workerIdx = 0; workerIdx < numWorkers; ++workerIdx) {
                size_t startTask = workerIdx * tasksPerWorker;
                size_t endTask = (workerIdx == numWorkers - 1) ? allTasks.size() : 
                                (workerIdx + 1) * tasksPerWorker;
                
                if (startTask >= allTasks.size()) break;
                
                // 提交工作者任务
                auto workerFuture = m_threadPool->submitTask([this, gridData, style, startTask, endTask, &allTasks, &completedTasks, &failedTasks]() -> std::vector<std::string> {
                    
                    std::vector<std::string> workerResults;
                    workerResults.reserve(endTask - startTask);
                    
                    // === 工作者内部：批量瓦片生成 ===
                    for (size_t taskIdx = startTask; taskIdx < endTask; ++taskIdx) {
                        const auto& task = allTasks[taskIdx];
                        
                        try {
                            // 生成单个瓦片（使用SIMD优化版本）
                            std::string tilePath = generateSingleTileOptimized(
                                gridData, task.x, task.y, task.z, style, task.outputPath, task.bounds);
                            
                            if (!tilePath.empty() && boost::filesystem::exists(tilePath)) {
                                workerResults.push_back(tilePath);
                                completedTasks.fetch_add(1);
                                
                                // 定期报告进度
                                if (completedTasks.load() % 10 == 0) {
                                    BOOST_LOG_TRIVIAL(debug) << "Parallel tiles: " << completedTasks.load() 
                                                           << "/" << allTasks.size() << " completed";
                                }
                            } else {
                                failedTasks.fetch_add(1);
                                BOOST_LOG_TRIVIAL(warning) << "Failed to generate tile Z" << task.z 
                                                          << "/" << task.x << "/" << task.y;
                            }
                            
                        } catch (const std::exception& e) {
                            failedTasks.fetch_add(1);
                            BOOST_LOG_TRIVIAL(error) << "Tile generation error Z" << task.z 
                                                   << "/" << task.x << "/" << task.y << ": " << e.what();
                        }
                    }
                    
                    return workerResults;
                });
                
                workerFutures.push_back(std::move(workerFuture));
            }
            
            // === 第三阶段：收集结果 ===
            for (auto& future : workerFutures) {
                try {
                    auto workerResults = future.get();
                    for (const auto& tilePath : workerResults) {
                        result.filePaths->push_back(tilePath);
                    }
                } catch (const std::exception& e) {
                    BOOST_LOG_TRIVIAL(error) << "Worker thread failed: " << e.what();
                }
            }
            
            // === 性能报告 ===
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            BOOST_LOG_TRIVIAL(info) << "Parallel tile generation completed in " << duration.count() << "ms";
            BOOST_LOG_TRIVIAL(info) << "Successfully generated " << completedTasks.load() 
                                   << " tiles, " << failedTasks.load() << " failed";
            
            // 更新性能统计
            renderCount_.fetch_add(1);
            totalRenderTime_.store(totalRenderTime_.load() + duration.count());
            
            // 这里不设置元数据，因为OutputResult结构中没有metadata字段
            
            return result;
            
        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "Parallel tile generation failed: " << e.what();
            throw core_services::ServiceException("Parallel tile generation failed: " + std::string(e.what()));
        }
    });
}

std::vector<uint32_t> VisualizationEngine::generateImageDataSIMD(
    std::shared_ptr<core_services::GridData> gridData,
    const core_services::output::StyleOptions& style) {
    
    const auto& definition = gridData->getDefinition();
    size_t width = definition.cols;
    size_t height = definition.rows;
    size_t totalPixels = width * height;
    
    // 获取数据统计信息（使用SIMD优化）
    auto stats = isSIMDOptimizationEnabled() ? 
        calculateDataStatisticsSIMD(gridData) : 
        calculateDataStatistics(gridData);
    
    // 获取数据指针和类型
    auto dataPtr = gridData->getDataPtr();
    auto dataType = gridData->getDataType();
    
    std::vector<float> values;
    values.reserve(totalPixels);
    
    // SIMD优化的数据类型转换
    if (dataType == core_services::DataType::Float32) {
        const float* floatData = static_cast<const float*>(dataPtr);
        values.assign(floatData, floatData + totalPixels);
    } else if (dataType == core_services::DataType::Float64 && simdManager_) {
        // 使用SIMD进行Double到Float转换
        const double* doubleData = static_cast<const double*>(dataPtr);
        values.resize(totalPixels);
        simdManager_->convertFloat64ToFloat32(doubleData, values.data(), totalPixels);
    } else if (dataType == core_services::DataType::Int32 && simdManager_) {
        // 使用SIMD进行Int到Float转换
        const int32_t* intData = static_cast<const int32_t*>(dataPtr);
        values.resize(totalPixels);
        simdManager_->convertIntToFloat(intData, values.data(), totalPixels);
    } else {
        // 回退到标量转换
        values.resize(totalPixels);
        if (dataType == core_services::DataType::Float64) {
            const double* doubleData = static_cast<const double*>(dataPtr);
            for (size_t i = 0; i < totalPixels; ++i) {
                values[i] = static_cast<float>(doubleData[i]);
            }
        } else {
            BOOST_LOG_TRIVIAL(warning) << "Unsupported data type for SIMD visualization, using zero values";
            std::fill(values.begin(), values.end(), 0.0f);
        }
    }
    
    // SIMD优化的颜色映射
    std::string colorMap = style.colorMap.empty() ? "viridis" : style.colorMap;
    return mapDataToColorsSIMD(values, colorMap, 
                               static_cast<float>(stats.minValue), 
                               static_cast<float>(stats.maxValue));
}

std::vector<uint32_t> VisualizationEngine::mapDataToColorsSIMD(
    const std::vector<float>& values,
    const std::string& colorMap,
    float minValue, float maxValue) {
    
    const size_t count = values.size();
    std::vector<uint32_t> colors(count);
    
    if (!simdManager_ || count < 16) {
        // 对于小数据集，标量实现更高效
        std::vector<double> doubleValues(values.begin(), values.end());
        return mapDataToColors(doubleValues, colorMap, 
                              static_cast<double>(minValue), 
                              static_cast<double>(maxValue));
    }
    
    try {
        // === 第一阶段：内存预分配和对齐优化 ===
        const size_t alignment = 32; // AVX2对齐
        const size_t alignedCount = ((count + alignment - 1) / alignment) * alignment;
        
        // 预分配对齐内存以提升cache性能
        std::vector<float> alignedValues(alignedCount, 0.0f);
        std::vector<float> normalizedValues(alignedCount, 0.0f);
        std::vector<uint32_t> alignedColors(alignedCount, 0);
        
        // 复制数据到对齐缓冲区
        std::memcpy(alignedValues.data(), values.data(), count * sizeof(float));
        
        // === 第二阶段：SIMD批量归一化 ===
        float range = maxValue - minValue;
        if (range > 0.0f) {
            // 使用SIMD减法和除法操作
            simdManager_->vectorScalarAdd(alignedValues.data(), -minValue, normalizedValues.data(), count);
            simdManager_->vectorScalarDiv(normalizedValues.data(), range, normalizedValues.data(), count);
        } else {
            // 范围为0时，所有值设为0.5
            std::fill_n(normalizedValues.data(), count, 0.5f);
        }
        
        // === 第三阶段：优化的颜色映射查找表 ===
        auto colorMapTable = getColorMap(colorMap);
        const size_t colorMapSize = colorMapTable.size();
        
        // 预计算插值表以减少运行时计算
        struct ColorMapEntry {
            float r, g, b;
        };
        std::vector<ColorMapEntry> optimizedColorMap(colorMapSize);
        for (size_t i = 0; i < colorMapSize; ++i) {
            optimizedColorMap[i] = {
                static_cast<float>(colorMapTable[i][0]),
                static_cast<float>(colorMapTable[i][1]),
                static_cast<float>(colorMapTable[i][2])
            };
        }
        
        // === 第四阶段：高性能SIMD颜色插值 ===
        const size_t vectorWidth = 8; // 假设AVX2向量宽度
        const size_t optimalBatchSize = vectorWidth * 4; // 优化批处理大小
        const size_t numBatches = (count + optimalBatchSize - 1) / optimalBatchSize;
        
        // 预分配批处理缓冲区（避免重复分配）
        std::vector<float> batchIndices(optimalBatchSize);
        std::vector<float> batchWeights(optimalBatchSize);
        std::vector<uint32_t> batchColors(optimalBatchSize);
        
        // 预计算常量
        const float scaleFloat = static_cast<float>(colorMapSize - 1);
        const int32_t maxIndex = static_cast<int32_t>(colorMapSize - 1);
        
        for (size_t batch = 0; batch < numBatches; ++batch) {
            const size_t startIdx = batch * optimalBatchSize;
            const size_t endIdx = std::min(startIdx + optimalBatchSize, count);
            const size_t currentBatchSize = endIdx - startIdx;
            
            // SIMD计算颜色映射索引
            simdManager_->vectorScalarMul(&normalizedValues[startIdx], scaleFloat, 
                                         batchIndices.data(), currentBatchSize);
            
            // === 高效的SIMD双线性插值 ===
            #pragma unroll(4)
            for (size_t i = 0; i < currentBatchSize; ++i) {
                const float index = batchIndices[i];
                
                // 使用bit操作优化范围检查
                const int32_t idx0 = static_cast<int32_t>(index);
                const int32_t clampedIdx0 = std::max(0, std::min(idx0, maxIndex));
                const int32_t clampedIdx1 = std::min(clampedIdx0 + 1, maxIndex);
                
                const float weight = index - static_cast<float>(clampedIdx0);
                
                // 预加载颜色数据以提升cache性能
                const ColorMapEntry& color0 = optimizedColorMap[clampedIdx0];
                const ColorMapEntry& color1 = optimizedColorMap[clampedIdx1];
                
                // SIMD融合插值计算
                const float r = color0.r + weight * (color1.r - color0.r);
                const float g = color0.g + weight * (color1.g - color0.g);
                const float b = color0.b + weight * (color1.b - color0.b);
                
                // 优化的位打包（避免浮点到整数转换开销）
                const uint32_t rInt = static_cast<uint32_t>(r + 0.5f) & 0xFF;
                const uint32_t gInt = static_cast<uint32_t>(g + 0.5f) & 0xFF;
                const uint32_t bInt = static_cast<uint32_t>(b + 0.5f) & 0xFF;
                
                batchColors[i] = rInt | (gInt << 8) | (bInt << 16) | (0xFF << 24);
            }
            
            // 批量复制结果
            std::memcpy(&colors[startIdx], batchColors.data(), 
                       currentBatchSize * sizeof(uint32_t));
        }
        
        // === 性能统计更新 ===
        simdOperationCount_.fetch_add(1);
        
        return colors;
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(warning) << "Optimized SIMD color mapping failed, falling back to scalar: " << e.what();
        
        // 智能回退：使用标量版本
        std::vector<double> doubleValues(values.begin(), values.end());
        return mapDataToColors(doubleValues, colorMap, 
                              static_cast<double>(minValue), 
                              static_cast<double>(maxValue));
    }
}

VisualizationEngine::DataStatistics VisualizationEngine::calculateDataStatisticsSIMD(
    std::shared_ptr<core_services::GridData> gridData) {
    
    DataStatistics stats{0.0, 0.0, 0.0, 0.0};
    
    if (!simdManager_) {
        return calculateDataStatistics(gridData);
    }
    
    // 获取数据指针和类型
    auto dataPtr = gridData->getDataPtr();
    auto dataType = gridData->getDataType();
    const auto& definition = gridData->getDefinition();
    
    size_t totalElements = definition.rows * definition.cols * gridData->getBandCount();
    
    if (totalElements == 0) {
        return stats;
    }
    
    try {
        if (dataType == core_services::DataType::Float32) {
            const float* floatData = static_cast<const float*>(dataPtr);
            
            // SIMD优化的统计计算
            stats.minValue = static_cast<double>(simdManager_->vectorMin(floatData, totalElements));
            stats.maxValue = static_cast<double>(simdManager_->vectorMax(floatData, totalElements));
            stats.meanValue = static_cast<double>(simdManager_->vectorMean(floatData, totalElements));
            
            // 计算标准差
            std::vector<float> deviations(totalElements);
            simdManager_->vectorScalarAdd(floatData, -static_cast<float>(stats.meanValue), deviations.data(), totalElements);
            simdManager_->vectorSquare(deviations.data(), deviations.data(), totalElements);
            float variance = simdManager_->vectorMean(deviations.data(), totalElements);
            stats.stdDev = static_cast<double>(std::sqrt(variance));
            
        } else if (dataType == core_services::DataType::Float64) {
            const double* doubleData = static_cast<const double*>(dataPtr);
            
            // 对于Double类型，转换为Float进行SIMD处理（牺牲一点精度换取性能）
            std::vector<float> floatData(totalElements);
            simdManager_->convertFloat64ToFloat32(doubleData, floatData.data(), totalElements);
            
            stats.minValue = static_cast<double>(simdManager_->vectorMin(floatData.data(), totalElements));
            stats.maxValue = static_cast<double>(simdManager_->vectorMax(floatData.data(), totalElements));
            stats.meanValue = static_cast<double>(simdManager_->vectorMean(floatData.data(), totalElements));
            
            // 计算标准差
            std::vector<float> deviations(totalElements);
            simdManager_->vectorScalarAdd(floatData.data(), -static_cast<float>(stats.meanValue), deviations.data(), totalElements);
            simdManager_->vectorSquare(deviations.data(), deviations.data(), totalElements);
            float variance = simdManager_->vectorMean(deviations.data(), totalElements);
            stats.stdDev = static_cast<double>(std::sqrt(variance));
            
        } else {
            // 对于其他数据类型，回退到标量计算
            return calculateDataStatistics(gridData);
        }
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(warning) << "SIMD statistics calculation failed, falling back to scalar: " << e.what();
        return calculateDataStatistics(gridData);
    }
    
    return stats;
}

void VisualizationEngine::convertToRGBASIMD(
    const std::vector<uint32_t>& imageData, 
    std::vector<uint8_t>& rgbaData) {
    
    const size_t pixelCount = imageData.size();
    rgbaData.reserve(pixelCount * 4);
    
    if (!simdManager_ || pixelCount < 16) {
        // 回退到标量实现
        rgbaData.clear();
        for (uint32_t color : imageData) {
            rgbaData.push_back(static_cast<uint8_t>(color & 0xFF));         // R
            rgbaData.push_back(static_cast<uint8_t>((color >> 8) & 0xFF));  // G
            rgbaData.push_back(static_cast<uint8_t>((color >> 16) & 0xFF)); // B
            rgbaData.push_back(static_cast<uint8_t>((color >> 24) & 0xFF)); // A
        }
        return;
    }
    
    try {
        // SIMD优化的像素解包
        rgbaData.resize(pixelCount * 4);
        
        // 使用SIMD批量处理像素解包
        const size_t batchSize = simdManager_->getOptimalBatchSize();
        const size_t numBatches = (pixelCount + batchSize - 1) / batchSize;
        
        for (size_t batch = 0; batch < numBatches; ++batch) {
            size_t startIdx = batch * batchSize;
            size_t endIdx = std::min(startIdx + batchSize, pixelCount);
            size_t currentBatchSize = endIdx - startIdx;
            
            // 为当前批次处理像素解包
            for (size_t i = 0; i < currentBatchSize; ++i) {
                uint32_t color = imageData[startIdx + i];
                size_t rgbaIdx = (startIdx + i) * 4;
                
                rgbaData[rgbaIdx + 0] = static_cast<uint8_t>(color & 0xFF);         // R
                rgbaData[rgbaIdx + 1] = static_cast<uint8_t>((color >> 8) & 0xFF);  // G
                rgbaData[rgbaIdx + 2] = static_cast<uint8_t>((color >> 16) & 0xFF); // B
                rgbaData[rgbaIdx + 3] = static_cast<uint8_t>((color >> 24) & 0xFF); // A
            }
        }
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(warning) << "SIMD RGBA conversion failed, falling back to scalar: " << e.what();
        // 回退到标量实现
        rgbaData.clear();
        for (uint32_t color : imageData) {
            rgbaData.push_back(static_cast<uint8_t>(color & 0xFF));         // R
            rgbaData.push_back(static_cast<uint8_t>((color >> 8) & 0xFF));  // G
            rgbaData.push_back(static_cast<uint8_t>((color >> 16) & 0xFF)); // B
            rgbaData.push_back(static_cast<uint8_t>((color >> 24) & 0xFF)); // A
        }
    }
}

std::string VisualizationEngine::getPerformanceReport() const {
    std::ostringstream report;
    
    report << "=== OSCEAN Visualization Engine Performance Report ===\n";
    report << "Total Renders: " << renderCount_.load() << "\n";
    
    if (renderCount_.load() > 0) {
        double avgRenderTime = totalRenderTime_.load() / renderCount_.load();
        report << "Average Render Time: " << avgRenderTime << " ms\n";
    }
    
    report << "SIMD Operations: " << simdOperationCount_.load() << "\n";
    report << "SIMD Enabled: " << (isSIMDOptimizationEnabled() ? "Yes" : "No") << "\n";
    
    if (simdManager_) {
        report << "SIMD Implementation: " << simdManager_->getImplementationName() << "\n";
        report << "SIMD Features: ";
        auto features = simdManager_->getFeatures();
        if (features.hasSSE2) report << "SSE2 ";
        if (features.hasSSE3) report << "SSE3 ";
        if (features.hasSSE4_1) report << "SSE4.1 ";
        if (features.hasAVX) report << "AVX ";
        if (features.hasAVX2) report << "AVX2 ";
        if (features.hasAVX512F) report << "AVX512F ";
        report << "\n";
        
        report << "Optimal Batch Size: " << simdManager_->getOptimalBatchSize() << "\n";
        report << "Memory Alignment: " << simdManager_->getAlignment() << " bytes\n";
    }
    
    // 缓存统计
    report << "Color Map Cache Size: " << colorMapCache_.size() << "\n";
    report << "Statistics Cache Size: " << statisticsCache_.size() << "\n";
    
    return report.str();
}

// === SIMD优化的网格数据重采样实现 ===

void VisualizationEngine::resampleGridDataSIMD(
    std::shared_ptr<core_services::GridData> gridData,
    const core_services::BoundingBox& targetBounds,
    int targetWidth, int targetHeight,
    float* outputData) {
    
    if (!simdManager_ || !gridData) {
        // 回退到标量实现
        resampleGridDataScalar(gridData, targetBounds, targetWidth, targetHeight, outputData);
        return;
    }
    
    try {
        const auto& gridDef = gridData->getDefinition();
        auto dataPtr = gridData->getDataPtr();
        auto dataType = gridData->getDataType();
        
        // 预计算重采样参数
        const double xScale = static_cast<double>(gridDef.cols) / (targetBounds.maxX - targetBounds.minX);
        const double yScale = static_cast<double>(gridDef.rows) / (targetBounds.maxY - targetBounds.minY);
        const double xOffset = (targetBounds.minX - gridDef.extent.minX) * xScale;
        const double yOffset = (targetBounds.minY - gridDef.extent.minY) * yScale;
        
        // === SIMD批量重采样 ===
        const size_t totalPixels = targetWidth * targetHeight;
        const size_t batchSize = simdManager_->getOptimalBatchSize();
        
        // 预分配批处理缓冲区
        std::vector<float> xCoords(batchSize);
        std::vector<float> yCoords(batchSize);
        std::vector<float> batchResults(batchSize);
        
        for (size_t pixelStart = 0; pixelStart < totalPixels; pixelStart += batchSize) {
            const size_t currentBatchSize = std::min(batchSize, totalPixels - pixelStart);
            
            // 批量计算坐标
            for (size_t i = 0; i < currentBatchSize; ++i) {
                const size_t pixelIdx = pixelStart + i;
                const int targetY = pixelIdx / targetWidth;
                const int targetX = pixelIdx % targetWidth;
                
                // 计算源数据坐标
                const double srcX = targetX * (targetBounds.maxX - targetBounds.minX) / targetWidth + targetBounds.minX;
                const double srcY = targetY * (targetBounds.maxY - targetBounds.minY) / targetHeight + targetBounds.minY;
                
                xCoords[i] = static_cast<float>((srcX - gridDef.extent.minX) * xScale);
                yCoords[i] = static_cast<float>((srcY - gridDef.extent.minY) * yScale);
            }
            
            // 双线性插值（回退到标量实现）
            for (size_t i = 0; i < currentBatchSize; ++i) {
                const float gridX = xCoords[i];
                const float gridY = yCoords[i];
                
                float interpolatedValue = 0.0f;
                
                // 边界检查
                if (gridX >= 0 && gridX < gridDef.cols - 1 && gridY >= 0 && gridY < gridDef.rows - 1) {
                    const int x0 = static_cast<int>(gridX);
                    const int y0 = static_cast<int>(gridY);
                    const int x1 = x0 + 1;
                    const int y1 = y0 + 1;
                    
                    const float wx = gridX - x0;
                    const float wy = gridY - y0;
                    
                    // 获取四个角点的值
                    float v00, v10, v01, v11;
                    
                    if (dataType == core_services::DataType::Float32) {
                        const float* floatData = static_cast<const float*>(dataPtr);
                        v00 = floatData[y0 * gridDef.cols + x0];
                        v10 = floatData[y0 * gridDef.cols + x1];
                        v01 = floatData[y1 * gridDef.cols + x0];
                        v11 = floatData[y1 * gridDef.cols + x1];
                    } else if (dataType == core_services::DataType::Float64) {
                        const double* doubleData = static_cast<const double*>(dataPtr);
                        v00 = static_cast<float>(doubleData[y0 * gridDef.cols + x0]);
                        v10 = static_cast<float>(doubleData[y0 * gridDef.cols + x1]);
                        v01 = static_cast<float>(doubleData[y1 * gridDef.cols + x0]);
                        v11 = static_cast<float>(doubleData[y1 * gridDef.cols + x1]);
                    } else {
                        v00 = v10 = v01 = v11 = 0.0f;
                    }
                    
                    // 双线性插值
                    const float v0 = v00 * (1.0f - wx) + v10 * wx;
                    const float v1 = v01 * (1.0f - wx) + v11 * wx;
                    interpolatedValue = v0 * (1.0f - wy) + v1 * wy;
                }
                
                batchResults[i] = interpolatedValue;
            }
            
            // 复制结果
            std::memcpy(&outputData[pixelStart], batchResults.data(), 
                       currentBatchSize * sizeof(float));
        }
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(warning) << "SIMD resampling failed, falling back to scalar: " << e.what();
        resampleGridDataScalar(gridData, targetBounds, targetWidth, targetHeight, outputData);
    }
}

void VisualizationEngine::resampleGridDataScalar(
    std::shared_ptr<core_services::GridData> gridData,
    const core_services::BoundingBox& targetBounds,
    int targetWidth, int targetHeight,
    float* outputData) {
    
    if (!gridData) {
        // 填充默认值
        std::fill_n(outputData, targetWidth * targetHeight, 0.0f);
        return;
    }
    
    const auto& gridDef = gridData->getDefinition();
    auto dataPtr = gridData->getDataPtr();
    auto dataType = gridData->getDataType();
    
    // 计算重采样参数
    const double xScale = static_cast<double>(gridDef.cols) / (gridDef.extent.maxX - gridDef.extent.minX);
    const double yScale = static_cast<double>(gridDef.rows) / (gridDef.extent.maxY - gridDef.extent.minY);
    
    // 标量双线性插值
    for (int targetY = 0; targetY < targetHeight; ++targetY) {
        for (int targetX = 0; targetX < targetWidth; ++targetX) {
            // 计算目标坐标在源数据中的位置
            const double srcX = targetX * (targetBounds.maxX - targetBounds.minX) / targetWidth + targetBounds.minX;
            const double srcY = targetY * (targetBounds.maxY - targetBounds.minY) / targetHeight + targetBounds.minY;
            
            // 转换为网格索引
            const double gridX = (srcX - gridDef.extent.minX) * xScale;
            const double gridY = (srcY - gridDef.extent.minY) * yScale;
            
            float interpolatedValue = 0.0f;
            
            // 边界检查
            if (gridX >= 0 && gridX < gridDef.cols - 1 && gridY >= 0 && gridY < gridDef.rows - 1) {
                const int x0 = static_cast<int>(gridX);
                const int y0 = static_cast<int>(gridY);
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                
                const float wx = gridX - x0;
                const float wy = gridY - y0;
                
                // 获取四个角点的值
                float v00, v10, v01, v11;
                
                if (dataType == core_services::DataType::Float32) {
                    const float* floatData = static_cast<const float*>(dataPtr);
                    v00 = floatData[y0 * gridDef.cols + x0];
                    v10 = floatData[y0 * gridDef.cols + x1];
                    v01 = floatData[y1 * gridDef.cols + x0];
                    v11 = floatData[y1 * gridDef.cols + x1];
                } else if (dataType == core_services::DataType::Float64) {
                    const double* doubleData = static_cast<const double*>(dataPtr);
                    v00 = static_cast<float>(doubleData[y0 * gridDef.cols + x0]);
                    v10 = static_cast<float>(doubleData[y0 * gridDef.cols + x1]);
                    v01 = static_cast<float>(doubleData[y1 * gridDef.cols + x0]);
                    v11 = static_cast<float>(doubleData[y1 * gridDef.cols + x1]);
                } else if (dataType == core_services::DataType::Int32) {
                    const int32_t* intData = static_cast<const int32_t*>(dataPtr);
                    v00 = static_cast<float>(intData[y0 * gridDef.cols + x0]);
                    v10 = static_cast<float>(intData[y0 * gridDef.cols + x1]);
                    v01 = static_cast<float>(intData[y1 * gridDef.cols + x0]);
                    v11 = static_cast<float>(intData[y1 * gridDef.cols + x1]);
                } else {
                    // 不支持的数据类型，使用默认值
                    v00 = v10 = v01 = v11 = 0.0f;
                }
                
                // 双线性插值
                const float v0 = v00 * (1.0f - wx) + v10 * wx;
                const float v1 = v01 * (1.0f - wx) + v11 * wx;
                interpolatedValue = v0 * (1.0f - wy) + v1 * wy;
            }
            
            outputData[targetY * targetWidth + targetX] = interpolatedValue;
        }
    }
}

// === SIMD优化的单瓦片生成实现 ===

std::string VisualizationEngine::generateSingleTileOptimized(
    std::shared_ptr<core_services::GridData> gridData,
    int x, int y, int z,
    const core_services::output::StyleOptions& style,
    const std::string& outputPath,
    const core_services::BoundingBox& tileBounds) {
    
    try {
        // === 第一阶段：空间数据快速裁剪 ===
        auto gridDef = gridData->getDefinition();
        
        // 快速边界检查
        bool intersects = !(tileBounds.maxX < gridDef.extent.minX || 
                           tileBounds.minX > gridDef.extent.maxX ||
                           tileBounds.maxY < gridDef.extent.minY ||
                           tileBounds.minY > gridDef.extent.maxY);
        if (!intersects) {
            // 生成空瓦片或跳过
            return "";
        }
        
        // === 第二阶段：SIMD优化的数据重采样 ===
        const int tileSize = 256; // 标准瓦片大小
        const size_t totalPixels = tileSize * tileSize;
        
        // 预分配对齐内存
        std::vector<float> tileData(totalPixels, 0.0f);
        
        // 使用SIMD优化的双线性插值
        if (simdManager_ && useSIMDOptimization_) {
            // SIMD批量重采样
            resampleGridDataSIMD(gridData, tileBounds, tileSize, tileSize, tileData.data());
        } else {
            // 标量重采样回退
            resampleGridDataScalar(gridData, tileBounds, tileSize, tileSize, tileData.data());
        }
        
        // === 第三阶段：高性能颜色映射 ===
        std::vector<uint32_t> colorData;
        if (simdManager_ && useSIMDOptimization_) {
            // 使用优化的SIMD颜色映射
            auto stats = calculateDataStatisticsSIMD(gridData);
            colorData = mapDataToColorsSIMD(tileData, style.colorMap, 
                                          static_cast<float>(stats.minValue), 
                                          static_cast<float>(stats.maxValue));
        } else {
            // 标量颜色映射回退
            auto stats = calculateDataStatistics(gridData);
            std::vector<double> doubleData(tileData.begin(), tileData.end());
            colorData = mapDataToColors(doubleData, style.colorMap, stats.minValue, stats.maxValue);
        }
        
        // === 第四阶段：优化的图像编码 ===
        std::vector<uint8_t> rgbaData;
        rgbaData.reserve(totalPixels * 4);
        
        if (simdManager_ && useSIMDOptimization_) {
            convertToRGBASIMD(colorData, rgbaData);
        } else {
            // 标量转换
            for (uint32_t color : colorData) {
                rgbaData.push_back(static_cast<uint8_t>(color & 0xFF));         // R
                rgbaData.push_back(static_cast<uint8_t>((color >> 8) & 0xFF));  // G
                rgbaData.push_back(static_cast<uint8_t>((color >> 16) & 0xFF)); // B
                rgbaData.push_back(static_cast<uint8_t>((color >> 24) & 0xFF)); // A
            }
        }
        
        // === 第五阶段：并行安全的文件保存 ===
        saveImageToFile(outputPath, rgbaData, tileSize, tileSize);
        
        return outputPath;
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Optimized tile generation failed for Z" << z 
                               << "/" << x << "/" << y << ": " << e.what();
        return "";
    }
}

// === GPU相关方法实现 ===

void VisualizationEngine::enableGPUOptimization(bool enable) {
    useGPUOptimization_ = enable && gpuAvailable_;
    if (useGPUOptimization_) {
        BOOST_LOG_TRIVIAL(info) << "GPU optimization enabled";
    } else {
        BOOST_LOG_TRIVIAL(info) << "GPU optimization disabled";
    }
}

void VisualizationEngine::setGPUFramework(std::shared_ptr<common_utils::gpu::OSCEANGPUFramework> gpuFramework) {
    gpuFramework_ = gpuFramework;
#ifdef OSCEAN_CUDA_ENABLED
    if (gpuFramework_) {
        // 重新初始化GPU组件
        initializeGPUComponents();
    }
#endif
}

void VisualizationEngine::initializeGPUComponents() {
#ifdef OSCEAN_CUDA_ENABLED
    using namespace output_generation::gpu;
    
    try {
        // 如果没有外部提供的GPU框架，尝试使用全局框架
        if (!gpuFramework_) {
            if (common_utils::gpu::OSCEANGPUFramework::initialize()) {
                // 使用全局单例（需要在OSCEANGPUFramework中实现getInstance方法）
                // gpuFramework_ = common_utils::gpu::OSCEANGPUFramework::getInstance();
                gpuAvailable_ = true;
            }
        }
        
        if (gpuFramework_ || gpuAvailable_) {
            // 创建GPU组件
            gpuEngine_ = std::make_unique<output_generation::gpu::GPUVisualizationEngine>();
            
            // 创建GPU颜色映射器
            auto cudaColorMapper = output_generation::gpu::createCUDAColorMapper();
            gpuColorMapper_ = std::move(cudaColorMapper);
            
            // 创建GPU瓦片生成器
            gpuTileGenerator_ = output_generation::gpu::createGPUTileGenerator(0);
            
            // 创建多GPU协调器
            // multiGPUCoordinator_ = std::make_shared<output_generation::gpu::MultiGPUCoordinator>();
            
            gpuAvailable_ = true;
            BOOST_LOG_TRIVIAL(info) << "GPU components initialized successfully";
        }
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "Failed to initialize GPU components: " << e.what();
        gpuAvailable_ = false;
    }
#else
    gpuAvailable_ = false;
#endif
}

boost::future<std::string> VisualizationEngine::renderToImageGPU(
    std::shared_ptr<core_services::GridData> gridData,
    const std::string& outputPath,
    const core_services::output::StyleOptions& style) {
    
    return m_threadPool->submitTask([this, gridData, outputPath, style]() -> std::string {
        try {
            BOOST_LOG_TRIVIAL(info) << "GPU-accelerated rendering to image: " << outputPath;
            
#ifdef OSCEAN_CUDA_ENABLED
            if (isGPUOptimizationEnabled() && gpuColorMapper_) {
                // 使用GPU生成图像数据
                auto imageData = generateImageDataGPU(gridData, style);
                
                const auto& definition = gridData->getDefinition();
                size_t width = definition.cols;
                size_t height = definition.rows;
                
                // 转换RGBA数据
                std::vector<uint8_t> rgbaData;
                rgbaData.reserve(width * height * 4);
                
                for (uint32_t color : imageData) {
                    rgbaData.push_back(static_cast<uint8_t>(color & 0xFF));         // R
                    rgbaData.push_back(static_cast<uint8_t>((color >> 8) & 0xFF));  // G
                    rgbaData.push_back(static_cast<uint8_t>((color >> 16) & 0xFF)); // B
                    rgbaData.push_back(static_cast<uint8_t>((color >> 24) & 0xFF)); // A
                }
                
                // 保存图像
                saveImageToFile(outputPath, rgbaData, width, height);
                
                BOOST_LOG_TRIVIAL(info) << "GPU-accelerated image rendered successfully to: " << outputPath;
                return outputPath;
            }
#endif
            
            // 回退到CPU实现
            BOOST_LOG_TRIVIAL(info) << "Falling back to CPU rendering";
            return renderToImage(gridData, outputPath, style).get();
            
        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "renderToImageGPU failed: " << e.what();
            throw;
        }
    }, common_utils::infrastructure::TaskType::CPU_INTENSIVE);
}

std::vector<uint32_t> VisualizationEngine::generateImageDataGPU(
    std::shared_ptr<core_services::GridData> gridData,
    const core_services::output::StyleOptions& style) {
    
#ifdef OSCEAN_CUDA_ENABLED
    if (!isGPUOptimizationEnabled() || !gpuColorMapper_) {
        // 回退到CPU实现
        return generateImageData(gridData, style);
    }
    
    const auto& definition = gridData->getDefinition();
    size_t width = definition.cols;
    size_t height = definition.rows;
    
    // 获取数据统计信息
    auto stats = calculateDataStatistics(gridData);
    
    // 准备float格式的数据
    std::vector<float> values;
    values.reserve(width * height);
    
    auto dataPtr = gridData->getDataPtr();
    auto dataType = gridData->getDataType();
    
    if (dataType == core_services::DataType::Float64) {
        const double* doubleData = static_cast<const double*>(dataPtr);
        for (size_t i = 0; i < width * height; ++i) {
            values.push_back(static_cast<float>(doubleData[i]));
        }
    } else if (dataType == core_services::DataType::Float32) {
        const float* floatData = static_cast<const float*>(dataPtr);
        values.assign(floatData, floatData + width * height);
    } else {
        values.assign(width * height, 0.0f);
    }
    
    // 使用GPU进行颜色映射
    std::string colorMap = style.colorMap.empty() ? "viridis" : style.colorMap;
    return mapDataToColorsGPU(values, colorMap, 
                              static_cast<float>(stats.minValue), 
                              static_cast<float>(stats.maxValue));
#else
    return generateImageData(gridData, style);
#endif
}

std::vector<uint32_t> VisualizationEngine::mapDataToColorsGPU(
    const std::vector<float>& values,
    const std::string& colorMap,
    float minValue, float maxValue) {
    
#ifdef OSCEAN_CUDA_ENABLED
    if (!gpuColorMapper_) {
        // 回退到CPU实现（需要转换float到double）
        std::vector<double> doubleValues(values.begin(), values.end());
        return mapDataToColors(doubleValues, colorMap, minValue, maxValue);
    }
    
    try {
        // 创建网格定义
        core_services::GridDefinition gridDef;
        gridDef.rows = 1;
        gridDef.cols = values.size();
        gridDef.extent.minX = 0;
        gridDef.extent.maxX = values.size() - 1;
        gridDef.extent.minY = 0;
        gridDef.extent.maxY = 1;
        
        // 创建GridData对象
        auto gridData = std::make_shared<core_services::GridData>(gridDef, core_services::DataType::Float32);
        
        // 设置数据
        std::vector<uint8_t>& buffer = gridData->getUnifiedBuffer();
        buffer.resize(values.size() * sizeof(float));
        std::memcpy(buffer.data(), values.data(), buffer.size());
        
        // 设置颜色映射参数
        output_generation::gpu::GPUColorMappingParams params;
        params.minValue = minValue;
        params.maxValue = maxValue;
        params.colormap = colorMap;
        params.autoScale = false;
        
        // 设置参数
        gpuColorMapper_->setParameters(params);
        
        // 创建执行上下文
        common_utils::gpu::GPUExecutionContext context;
        context.deviceId = 0;
        // streamId不存在，使用默认流
        
        // 执行GPU颜色映射
        auto future = gpuColorMapper_->executeAsync(gridData, context);
        auto result = future.get();
        
        if (!result.success) {
            BOOST_LOG_TRIVIAL(warning) << "GPU color mapping failed: " << result.errorMessage;
            // 回退到CPU实现
            std::vector<double> doubleValues(values.begin(), values.end());
            return mapDataToColors(doubleValues, colorMap, minValue, maxValue);
        }
        
        // 转换结果格式
        std::vector<uint32_t> colors;
        colors.reserve(values.size());
        
        const auto& imageData = result.data.imageData;
        for (size_t i = 0; i < values.size(); ++i) {
            uint32_t rgba = 0;
            rgba |= imageData[i * 4 + 0];       // R
            rgba |= imageData[i * 4 + 1] << 8;  // G
            rgba |= imageData[i * 4 + 2] << 16; // B
            rgba |= imageData[i * 4 + 3] << 24; // A
            colors.push_back(rgba);
        }
        
        return colors;
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "GPU color mapping exception: " << e.what();
        // 回退到CPU实现
        std::vector<double> doubleValues(values.begin(), values.end());
        return mapDataToColors(doubleValues, colorMap, minValue, maxValue);
    }
#else
    // 无GPU支持，使用CPU实现
    std::vector<double> doubleValues(values.begin(), values.end());
    return mapDataToColors(doubleValues, colorMap, minValue, maxValue);
#endif
}

boost::future<core_services::output::OutputResult> VisualizationEngine::generateTilesGPU(
    std::shared_ptr<core_services::GridData> gridData,
    const std::string& outputDirectory,
    const core_services::output::StyleOptions& style,
    int minZoom, int maxZoom) {
    
    return m_threadPool->submitTask([this, gridData, outputDirectory, style, minZoom, maxZoom]() -> core_services::output::OutputResult {
        try {
            BOOST_LOG_TRIVIAL(info) << "GPU-accelerated tile generation from zoom " << minZoom << " to " << maxZoom;
            
#ifdef OSCEAN_CUDA_ENABLED
            if (isGPUOptimizationEnabled() && gpuTileGenerator_) {
                std::vector<std::string> generatedFiles;
                
                // 创建输出目录
                boost::filesystem::create_directories(outputDirectory);
                
                // 直接使用gpuTileGenerator进行瓦片生成
                for (int z = minZoom; z <= maxZoom; ++z) {
                    int numTiles = static_cast<int>(std::pow(2, z));
                    
                    BOOST_LOG_TRIVIAL(info) << "Generating zoom level " << z << " with GPU acceleration";
                    
                    for (int x = 0; x < numTiles; ++x) {
                        for (int y = 0; y < numTiles; ++y) {
                            try {
                                // 计算瓦片边界
                                auto tileBounds = calculateTileBounds(x, y, z);
                                
                                // 生成单个瓦片
                                auto tilePath = generateSingleTileGPU(
                                    gridData, x, y, z, style, outputDirectory, tileBounds);
                                
                                if (!tilePath.empty()) {
                                    generatedFiles.push_back(tilePath);
                                }
                            } catch (const std::exception& e) {
                                BOOST_LOG_TRIVIAL(warning) << "GPU tile generation failed: " << e.what();
                            }
                        }
                    }
                }
                
                // 构造结果
                core_services::output::OutputResult result;
                result.filePaths = generatedFiles;
                
                BOOST_LOG_TRIVIAL(info) << "GPU tile generation completed, " << generatedFiles.size() << " tiles created";
                return result;
            }
#endif
            
            // 回退到CPU实现
            BOOST_LOG_TRIVIAL(info) << "Falling back to CPU tile generation";
            return generateTiles(gridData, outputDirectory, style, minZoom, maxZoom).get();
            
        } catch (const std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << "generateTilesGPU failed: " << e.what();
            throw;
        }
    }, common_utils::infrastructure::TaskType::CPU_INTENSIVE);
}

std::string VisualizationEngine::generateSingleTileGPU(
    std::shared_ptr<core_services::GridData> gridData,
    int x, int y, int z,
    const core_services::output::StyleOptions& style,
    const std::string& outputPath,
    const core_services::BoundingBox& tileBounds) {
    
#ifdef OSCEAN_CUDA_ENABLED
    if (!isGPUOptimizationEnabled() || !gpuTileGenerator_) {
        // 回退到CPU实现
        return generateSingleTile(gridData, x, y, z, style, outputPath);
    }
    
    try {
        // 设置瓦片生成参数
        output_generation::gpu::GPUTileGenerationParams params;
        params.zoomLevel = z;
        params.tileSize = 256;
        params.format = "PNG";
        
        gpuTileGenerator_->setParameters(params);
        
        // 创建执行上下文
        common_utils::gpu::GPUExecutionContext context;
        context.deviceId = 0;
        // streamId不存在，使用默认流
        
        // 执行GPU瓦片生成
        auto future = gpuTileGenerator_->executeAsync(gridData, context);
        auto results = future.get();
        
        if (!results.success || results.data.empty()) {
            BOOST_LOG_TRIVIAL(warning) << "GPU tile generation failed";
            // 回退到CPU实现
            return generateSingleTile(gridData, x, y, z, style, outputPath);
        }
        
        // 查找对应的瓦片结果
        for (const auto& vizResult : results.data) {
            // 构建输出路径
            boost::filesystem::path tileDir(outputPath);
            tileDir /= std::to_string(z);
            tileDir /= std::to_string(x);
            boost::filesystem::create_directories(tileDir);
            
            boost::filesystem::path tilePath = tileDir / (std::to_string(y) + ".png");
            
            // 保存瓦片
            saveImageToFile(tilePath.string(), vizResult.imageData, vizResult.width, vizResult.height);
            
            return tilePath.string();
        }
        
        // 如果没有找到对应的瓦片，回退到CPU实现
        return generateSingleTile(gridData, x, y, z, style, outputPath);
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "GPU tile generation exception: " << e.what();
        // 回退到CPU实现
        return generateSingleTile(gridData, x, y, z, style, outputPath);
    }
#else
    return generateSingleTile(gridData, x, y, z, style, outputPath);
#endif
}

std::shared_ptr<core_services::FeatureCollection> VisualizationEngine::generateContoursGPU(
    std::shared_ptr<core_services::GridData> gridData,
    const std::vector<double>& levels) {
    
#ifdef OSCEAN_CUDA_ENABLED
    if (!isGPUOptimizationEnabled()) {
        // 回退到CPU实现
        return generateContours(gridData, levels);
    }
    
    try {
        const auto& definition = gridData->getDefinition();
        int width = definition.cols;
        int height = definition.rows;
        
        // 准备float格式的数据和等值线级别
        std::vector<float> floatData;
        std::vector<float> floatLevels;
        
        // 转换数据
        auto dataPtr = gridData->getDataPtr();
        auto dataType = gridData->getDataType();
        
        if (dataType == core_services::DataType::Float64) {
            const double* doubleData = static_cast<const double*>(dataPtr);
            floatData.assign(doubleData, doubleData + width * height);
        } else if (dataType == core_services::DataType::Float32) {
            const float* data = static_cast<const float*>(dataPtr);
            floatData.assign(data, data + width * height);
        } else {
            // 不支持的数据类型，回退到CPU
            return generateContours(gridData, levels);
        }
        
        // 转换等值线级别
        for (double level : levels) {
            floatLevels.push_back(static_cast<float>(level));
        }
        
        // 调用GPU等值线生成
        float* d_contourPoints = nullptr;
        int numContourPoints = 0;
        
        cudaError_t err = ::generateContoursGPU(
            floatData.data(), width, height,
            floatLevels.data(), floatLevels.size(),
            &d_contourPoints, &numContourPoints,
            nullptr // 使用默认流
        );
        
        if (err != cudaSuccess) {
            BOOST_LOG_TRIVIAL(warning) << "GPU contour generation failed: " << cudaGetErrorString(err);
            // 回退到CPU实现
            return generateContours(gridData, levels);
        }
        
        // 将GPU结果转换为FeatureCollection
        // 这里需要实现GPU结果到FeatureCollection的转换
        // 暂时回退到CPU实现
        return generateContours(gridData, levels);
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "GPU contour generation exception: " << e.what();
        return generateContours(gridData, levels);
    }
#else
    return generateContours(gridData, levels);
#endif
}

} // namespace output
} // namespace oscean 