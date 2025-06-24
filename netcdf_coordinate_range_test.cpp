#include <iostream>
#include <memory>
#include <iomanip>

// 项目头文件
#include "core_services/crs/crs_service_factory.h"
#include "core_services/crs/i_crs_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "core_services/common_data_types.h"
#include "core_services/data_access/unified_data_types.h"
#include "core_services_impl/crs_service/src/impl/optimized_crs_service_impl.h"

using namespace oscean::core_services::crs;
using namespace oscean::common_utils::infrastructure;

int main() {
    std::cout << "=== NetCDF坐标范围转换测试 ===" << std::endl;
    
    try {
        // 初始化服务
        auto commonFactory = std::make_shared<CommonServicesFactory>();
        auto crsFactory = std::make_unique<CrsServiceFactory>(commonFactory);
        auto crsService = crsFactory->createCrsService();
        
        // 创建NetCDF的非标准极地立体投影参数
        oscean::core_services::CFProjectionParameters cfParams;
        cfParams.gridMappingName = "polar_stereographic";
        cfParams.numericParameters["latitude_of_projection_origin"] = 90.0;
        cfParams.numericParameters["straight_vertical_longitude_from_pole"] = -45.0;
        cfParams.numericParameters["standard_parallel"] = 90.0;
        cfParams.numericParameters["false_easting"] = 0.0;
        cfParams.numericParameters["false_northing"] = 0.0;
        cfParams.numericParameters["semi_major_axis"] = 6378273.0;
        cfParams.numericParameters["semi_minor_axis"] = 6378273.0;
        cfParams.stringParameters["units"] = "m";
        
        // 获取OptimizedCrsServiceImpl实例  
        auto optimizedService = dynamic_cast<OptimizedCrsServiceImpl*>(crsService.get());
        if (!optimizedService) {
            std::cerr << "无法获取OptimizedCrsServiceImpl实例" << std::endl;
            return 1;
        }
        
        // 从CF参数创建CRS
        auto sourceCRSResult = optimizedService->createCRSFromCFParametersAsync(cfParams).get();
        if (!sourceCRSResult.has_value()) {
            std::cerr << "从CF参数创建CRS失败" << std::endl;
            return 1;
        }
        
        auto sourceCRS = sourceCRSResult.value();
        std::cout << "源CRS创建成功:" << std::endl;
        std::cout << "  ID: " << sourceCRS.id << std::endl;
        std::cout << "  PROJ字符串: " << sourceCRS.projString << std::endl;
        
        // 创建WGS84目标坐标系
        auto targetCRSResult = crsService->parseFromEpsgCodeAsync(4326).get();
        if (!targetCRSResult.has_value()) {
            std::cerr << "WGS84坐标系创建失败" << std::endl;
            return 1;
        }
        
        auto targetCRS = targetCRSResult.value();
        std::cout << "目标CRS (WGS84) 创建成功" << std::endl;
        
        // NetCDF文件的坐标范围（基于ncdump结果）
        // Y: -4,300,000m 到 +2,798,000m
        // X: -3,600,000m 到 +3,798,000m
        
        struct BoundaryPoint {
            std::string name;
            double x, y;  // 投影坐标 (米)
        };
        
        std::vector<BoundaryPoint> boundaryPoints = {
            {"左下角 (西南)", -3600000.0, -4300000.0},
            {"右下角 (东南)", +3798000.0, -4300000.0},
            {"左上角 (西北)", -3600000.0, +2798000.0},
            {"右上角 (东北)", +3798000.0, +2798000.0},
            {"中心点", 0.0, 0.0}
        };
        
        std::cout << "\n=== NetCDF边界点坐标转换 ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        
        double minLon = 180.0, maxLon = -180.0;
        double minLat = 90.0, maxLat = -90.0;
        
        for (const auto& point : boundaryPoints) {
            std::cout << "\n" << point.name << ":" << std::endl;
            std::cout << "  投影坐标: (" << point.x << ", " << point.y << ") m" << std::endl;
            
            auto result = crsService->transformPointAsync(point.x, point.y, sourceCRS, targetCRS).get();
            
            if (result.status == oscean::core_services::TransformStatus::SUCCESS) {
                double lon = result.x;
                double lat = result.y;
                
                std::cout << "  WGS84坐标: (" << lon << "°, " << lat << "°)" << std::endl;
                
                // 更新范围
                minLon = std::min(minLon, lon);
                maxLon = std::max(maxLon, lon);
                minLat = std::min(minLat, lat);
                maxLat = std::max(maxLat, lat);
                
                std::cout << "  ✅ 转换成功" << std::endl;
            } else {
                std::cout << "  ❌ 转换失败: " << result.errorMessage.value_or("未知错误") << std::endl;
            }
        }
        
        std::cout << "\n=== NetCDF文件经纬度覆盖范围 ===" << std::endl;
        std::cout << "经度范围: " << minLon << "° 到 " << maxLon << "°" << std::endl;
        std::cout << "纬度范围: " << minLat << "° 到 " << maxLat << "°" << std::endl;
        std::cout << "经度跨度: " << (maxLon - minLon) << "°" << std::endl;
        std::cout << "纬度跨度: " << (maxLat - minLat) << "°" << std::endl;
        
        // 测试一些中间点以验证覆盖区域
        std::cout << "\n=== 中间点验证 ===" << std::endl;
        std::vector<BoundaryPoint> midPoints = {
            {"南边界中点", 0.0, -4300000.0},
            {"北边界中点", 0.0, +2798000.0},
            {"西边界中点", -3600000.0, 0.0},
            {"东边界中点", +3798000.0, 0.0}
        };
        
        for (const auto& point : midPoints) {
            std::cout << "\n" << point.name << ":" << std::endl;
            auto result = crsService->transformPointAsync(point.x, point.y, sourceCRS, targetCRS).get();
            
            if (result.status == oscean::core_services::TransformStatus::SUCCESS) {
                std::cout << "  (" << point.x << ", " << point.y << ") m -> (" 
                          << result.x << "°, " << result.y << "°)" << std::endl;
            } else {
                std::cout << "  转换失败" << std::endl;
            }
        }
        
        std::cout << "\n=== 测试完成 ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }
} 