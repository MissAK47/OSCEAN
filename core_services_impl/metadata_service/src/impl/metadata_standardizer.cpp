#include "metadata_standardizer.h"
#include "common_utils/time/time_interfaces.h" 
#include "common_utils/time/time_calendar.h"
#include "core_services/crs/i_crs_service.h" // For ICrsService
#include "common_utils/utilities/logging_utils.h" // 确保LOG宏可用
#include <algorithm>
#include <string>
#include <vector>
#include <iostream> // For std::cout debugging
#include <cctype> // For ::tolower

namespace oscean::core_services::metadata::impl {

// 恢复构造函数实现
MetadataStandardizer::MetadataStandardizer(std::shared_ptr<oscean::core_services::ICrsService> crsService)
    : crsService_(std::move(crsService)) {
    // 🔧 允许CRS服务为空，在使用时进行检查
    if (!crsService_) {
        LOG_WARN("MetadataStandardizer initialized without CRS service - CRS operations will be skipped");
    } else {
        LOG_INFO("MetadataStandardizer initialized with CRS service");
    }
}

// 主分发函数
oscean::core_services::FileMetadata MetadataStandardizer::standardizeMetadata(
    const oscean::core_services::FileMetadata& rawMetadata,
    const std::string& readerType) const {
    
    LOG_INFO("开始标准化元数据，读取器类型: {}", readerType);
    std::cout << "🔧 [MetadataStandardizer] 开始标准化元数据，读取器类型: " << readerType << std::endl;
    std::cout << "🔧 [MetadataStandardizer] 📊 输入FileMetadata统计:" << std::endl;
    std::cout << "🔧 [MetadataStandardizer]   - 文件路径: " << rawMetadata.filePath << std::endl;
    std::cout << "🔧 [MetadataStandardizer]   - 格式: " << rawMetadata.format << std::endl;
    std::cout << "🔧 [MetadataStandardizer]   - 变量数量: " << rawMetadata.variables.size() << std::endl;
    std::cout << "🔧 [MetadataStandardizer]   - geographicDimensions数量: " << rawMetadata.geographicDimensions.size() << std::endl;
    std::cout << "🔧 [MetadataStandardizer]   - 时间开始: " << rawMetadata.temporalInfo.startTime << std::endl;
    std::cout << "🔧 [MetadataStandardizer]   - 时间结束: " << rawMetadata.temporalInfo.endTime << std::endl;
    std::cout << "🔧 [MetadataStandardizer]   - 时间分辨率(秒): " << (rawMetadata.temporalInfo.temporalResolutionSeconds ? 
             std::to_string(*rawMetadata.temporalInfo.temporalResolutionSeconds) : "NULL") << std::endl;
    
    LOG_INFO("📊 输入FileMetadata统计:");
    LOG_INFO("  - 文件路径: {}", rawMetadata.filePath);
    LOG_INFO("  - 格式: {}", rawMetadata.format);
    LOG_INFO("  - 变量数量: {}", rawMetadata.variables.size());
    LOG_INFO("  - geographicDimensions数量: {}", rawMetadata.geographicDimensions.size());
    LOG_INFO("  - 时间开始: {}", rawMetadata.temporalInfo.startTime);
    LOG_INFO("  - 时间结束: {}", rawMetadata.temporalInfo.endTime);
    LOG_INFO("  - 时间分辨率(秒): {}", rawMetadata.temporalInfo.temporalResolutionSeconds ? 
             std::to_string(*rawMetadata.temporalInfo.temporalResolutionSeconds) : "NULL");
    
    auto standardized = rawMetadata;
    
    if (readerType.find("NetCDF") != std::string::npos) {
        applyNetCDFStandardization(standardized);
    } else if (readerType.find("GDAL") != std::string::npos) {
        applyGDALStandardization(standardized);
    } else {
        LOG_WARN("未知的读取器类型 '{}'，将应用通用标准化规则。", readerType);
    }
    
    validateAndRepair(standardized);
    
    LOG_INFO("元数据标准化完成: {}", standardized.filePath);
    return standardized;
}

// 恢复并重构NetCDF特定实现
void MetadataStandardizer::applyNetCDFStandardization(oscean::core_services::FileMetadata& metadata) const {
    LOG_DEBUG("应用NetCDF特定标准化规则...");

    // 1. 标准化CRS信息
    // 优先使用WKT，其次是PROJ字符串，最后是EPSG代码
    std::string crsString = !metadata.crs.wkt.empty() ? metadata.crs.wkt 
                           : !metadata.crs.projString.empty() ? metadata.crs.projString
                           : metadata.crs.epsgCode.has_value() ? "EPSG:" + std::to_string(*metadata.crs.epsgCode)
                           : "";

    if (!crsString.empty() && crsService_) {
        try {
            auto parseResult = crsService_->parseFromStringAsync(crsString).get();
            if (parseResult) {
                metadata.crs = *parseResult;
                LOG_INFO("CRS解析成功: {}", metadata.crs.id);
            } else {
                LOG_WARN("CRS解析失败，文件: {}. CRS String: '{}'.", metadata.filePath, crsString);
            }
        } catch (const std::exception& e) {
            LOG_WARN("CRS解析异常，文件: {}. 错误: {}", metadata.filePath, e.what());
        }
    } else if (!crsString.empty()) {
        LOG_DEBUG("CRS服务不可用，跳过CRS解析: {}", crsString);
    }

    // 2. 转换空间范围 (Bounding Box) - 仅在CRS有效且CRS服务可用时进行
    if (!metadata.crs.id.empty() && crsService_) {
        try {
            auto targetCrsOpt = crsService_->parseFromStringAsync("EPSG:4326").get();
            if (targetCrsOpt) {
                auto transformResult = crsService_->transformBoundingBoxAsync(metadata.spatialCoverage, *targetCrsOpt).get();
                // 注意: IDataAccessReader的结果类型和ICrsService的参数类型可能需要适配
                // 假设 transformBoundingBoxAsync 接受并返回 BoundingBox
                metadata.spatialCoverage = transformResult;
                LOG_INFO("空间边界框成功转换为WGS84");
            }
        } catch (const std::exception& e) {
            LOG_WARN("空间边界框转换失败: {}", e.what());
        }
    }

    // 3. 时间标准化逻辑
    LOG_INFO("开始处理时间标准化，geographicDimensions数量: {}", metadata.geographicDimensions.size());
    
    for (size_t i = 0; i < metadata.geographicDimensions.size(); ++i) {
        const auto& dim = metadata.geographicDimensions[i];
        LOG_INFO("处理维度 {}: name={}, 坐标数量={}, 属性数量={}", 
                 i, dim.name, dim.coordinates.size(), dim.attributes.size());
        
        // 打印所有属性
        for (const auto& [key, value] : dim.attributes) {
            LOG_INFO("  属性: {} = {}", key, value);
        }
        
        bool isTimeDimension = false;
        if (dim.attributes.count("units")) {
            std::string units = dim.attributes.at("units");
            LOG_INFO("检查units属性: '{}'", units);
            if (units.find("since") != std::string::npos) {
                isTimeDimension = true;
                LOG_INFO("✅ 检测到时间维度: {}", dim.name);
            }
        } else {
            LOG_INFO("维度 {} 没有units属性", dim.name);
        }
        
        if (isTimeDimension && !dim.coordinates.empty() && dim.attributes.count("units")) {
            const auto& units = dim.attributes.at("units");
            LOG_INFO("开始处理时间维度: {}, units: {}, 坐标范围: [{}, {}]", 
                     dim.name, units, dim.coordinates.front(), dim.coordinates.back());
                     
            auto startTimeOpt = oscean::common_utils::time::CFTimeConverter::convertCFTime(dim.coordinates.front(), units);
            auto endTimeOpt = oscean::common_utils::time::CFTimeConverter::convertCFTime(dim.coordinates.back(), units);

            if (startTimeOpt) {
                metadata.temporalInfo.startTime = oscean::common_utils::time::CalendarUtils::toISO8601(*startTimeOpt);
                LOG_INFO("✅ 开始时间转换成功: {}", metadata.temporalInfo.startTime);
            } else {
                LOG_WARN("❌ 开始时间转换失败");
            }
            if (endTimeOpt) {
                metadata.temporalInfo.endTime = oscean::common_utils::time::CalendarUtils::toISO8601(*endTimeOpt);
                LOG_INFO("✅ 结束时间转换成功: {}", metadata.temporalInfo.endTime);
            } else {
                LOG_WARN("❌ 结束时间转换失败");
            }

            // 计算时间分辨率
            if (dim.coordinates.size() > 1) {
                LOG_INFO("开始计算时间分辨率，坐标数量: {}", dim.coordinates.size());
                auto timeResOpt = oscean::common_utils::time::CFTimeConverter::calculateTimeResolution(dim.coordinates, units);
                if (timeResOpt) {
                    metadata.temporalInfo.temporalResolutionSeconds = static_cast<int>(*timeResOpt);
                    LOG_INFO("✅ 计算得到时间分辨率: {} 秒", *timeResOpt);
                } else {
                    LOG_WARN("❌ 时间分辨率计算失败，时间单位: {}, 坐标数量: {}", units, dim.coordinates.size());
                }
            } else {
                LOG_INFO("单时间点文件，从其他信息推断时间分辨率，坐标数量: {}", dim.coordinates.size());
                
                // 从全局属性推断时间分辨率
                bool foundResolution = false;
                for (const auto& [key, value] : metadata.attributes) {
                    std::cout << "🔧 检查全局属性: " << key << " = " << value << std::endl;
                    std::string lowerValue = value;
                    std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(), ::tolower);
                    
                    if (lowerValue.find("monthly") != std::string::npos || 
                        lowerValue.find("month") != std::string::npos) {
                        metadata.temporalInfo.temporalResolutionSeconds = 2592000; // 30天 * 24小时 * 3600秒
                        LOG_INFO("✅ 从全局属性检测到月度数据，设置时间分辨率: 2592000 秒 (30天)");
                        std::cout << "✅ 从全局属性检测到月度数据，设置时间分辨率: 2592000 秒" << std::endl;
                        foundResolution = true;
                        break;
                    } else if (lowerValue.find("daily") != std::string::npos || 
                               lowerValue.find("day") != std::string::npos) {
                        metadata.temporalInfo.temporalResolutionSeconds = 86400; // 24小时 * 3600秒
                        LOG_INFO("✅ 从全局属性检测到日度数据，设置时间分辨率: 86400 秒 (1天)");
                        foundResolution = true;
                        break;
                    }
                }
                
                // 如果全局属性未能确定，从文件名推断
                if (!foundResolution) {
                    std::string filename = metadata.filePath;
                    std::cout << "🔧 从文件名推断时间分辨率: " << filename << std::endl;
                    
                    // 检查文件名中的年月模式（如 cs_2023_01_00_00.nc）
                    if (filename.find("_2023_") != std::string::npos || 
                        filename.find("_2024_") != std::string::npos ||
                        filename.find("_2022_") != std::string::npos) {
                        // 假设这种模式是月度数据
                        metadata.temporalInfo.temporalResolutionSeconds = 2592000; // 30天
                        LOG_INFO("✅ 从文件名模式推断为月度数据，设置时间分辨率: 2592000 秒");
                        std::cout << "✅ 从文件名模式推断为月度数据" << std::endl;
                        foundResolution = true;
                    }
                }
                
                // 最后的默认值
                if (!foundResolution) {
                    metadata.temporalInfo.temporalResolutionSeconds = 86400; // 默认日度
                    LOG_INFO("⚠️ 无法确定时间分辨率，使用默认值: 86400 秒 (1天)");
                    std::cout << "⚠️ 使用默认时间分辨率: 86400 秒" << std::endl;
                }
            }

            break;
        }
    }

    // 4. 计算空间分辨率
    for (const auto& dim : metadata.geographicDimensions) {
        if (dim.coordinates.size() > 1) {
            std::string stdName = dim.attributes.count("standard_name") ? dim.attributes.at("standard_name") : "";
            std::string units = dim.attributes.count("units") ? dim.attributes.at("units") : "";

            double resolution = std::abs((dim.coordinates.back() - dim.coordinates.front()) / (dim.coordinates.size() - 1));

            if (stdName == "longitude" || units == "degrees_east") {
                metadata.spatialInfo.resolutionX = resolution;
                LOG_DEBUG("计算得到经度分辨率: {}", resolution);
            } else if (stdName == "latitude" || units == "degrees_north") {
                metadata.spatialInfo.resolutionY = resolution;
                LOG_DEBUG("计算得到纬度分辨率: {}", resolution);
            }
        }
    }

    // 5. 变量元数据标准化
    LOG_DEBUG("开始标准化变量元数据...");
    for (auto& var : metadata.variables) {
        if(var.attributes.count("_FillValue")) {
            try {
                var.noDataValue = std::stod(var.attributes.at("_FillValue"));
            } catch (const std::exception& e) {
                LOG_WARN("无法将_FillValue '{}' 解析为double. 变量: {}. 错误: {}", var.attributes.at("_FillValue"), var.name, e.what());
            }
        }
        if(var.attributes.count("scale_factor")) {
             try {
                var.scaleFactor = std::stod(var.attributes.at("scale_factor"));
            } catch (const std::exception& e) {
                LOG_WARN("无法将scale_factor '{}' 解析为double. 变量: {}. 错误: {}", var.attributes.at("scale_factor"), var.name, e.what());
            }
        }
        if(var.attributes.count("add_offset")) {
            try {
                var.addOffset = std::stod(var.attributes.at("add_offset"));
            } catch (const std::exception& e) {
                LOG_WARN("无法将add_offset '{}' 解析为double. 变量: {}. 错误: {}", var.attributes.at("add_offset"), var.name, e.what());
            }
        }

        double minVal, maxVal;
        bool hasMin = false, hasMax = false;
        if(var.attributes.count("valid_min")) {
            try {
                minVal = std::stod(var.attributes.at("valid_min"));
                hasMin = true;
            } catch (const std::exception& e) {
                 LOG_WARN("无法将valid_min '{}' 解析为double. 变量: {}. 错误: {}", var.attributes.at("valid_min"), var.name, e.what());
            }
        }
        if(var.attributes.count("valid_max")) {
            try {
                maxVal = std::stod(var.attributes.at("valid_max"));
                hasMax = true;
            } catch (const std::exception& e) {
                 LOG_WARN("无法将valid_max '{}' 解析为double. 变量: {}. 错误: {}", var.attributes.at("valid_max"), var.name, e.what());
            }
        }

        if(hasMin && hasMax) {
            var.validRange = oscean::core_services::ValueRange<double>(minVal, maxVal);
        }
    }
}

void MetadataStandardizer::applyGDALStandardization(oscean::core_services::FileMetadata& metadata) const {
    LOG_WARN("GDAL标准化功能尚未实现。");
    (void)metadata; // 避免未使用参数的警告
}

void MetadataStandardizer::validateAndRepair(oscean::core_services::FileMetadata& metadata) const {
    LOG_DEBUG("验证和修复元数据...");
    if (metadata.crs.wkt.empty() && metadata.crs.projString.empty() && !metadata.crs.epsgCode.has_value()) {
        LOG_WARN("元数据中CRS信息完全缺失，无法进行标准化。");
    }
}

} // namespace oscean::core_services::metadata::impl 