/**
 * @file simple_netcdf_test.cpp
 * @brief 简化的NetCDF非标准极地投影测试
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

// 项目头文件
#include "core_services/crs/crs_service_factory.h"
#include "core_services/crs/i_crs_service.h"
#include "common_utils/infrastructure/common_services_factory.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services;
using namespace oscean::core_services::crs;
using namespace oscean::common_utils::infrastructure;

class SimpleNetCDFTest : public ::testing::Test {
protected:
    std::unique_ptr<ICrsService> crsService_;
    
    void SetUp() override {
        // 创建CRS服务
        auto commonFactory = std::make_shared<CommonServicesFactory>();
        auto crsFactory = std::make_unique<CrsServiceFactory>(commonFactory);
        crsService_ = crsFactory->createCrsService();
        ASSERT_TRUE(crsService_ != nullptr) << "Failed to create CRS service";
    }
};

/**
 * @brief 测试NetCDF非标准极地立体投影的基本转换 
 */
TEST_F(SimpleNetCDFTest, BasicPolarStereographicConversion) {
    std::cout << "\n=== NetCDF非标准极地立体投影基本测试 ===" << std::endl;
    
    // 1. 解决NetCDF特定的极地立体投影问题
    CFProjectionParameters cfParams;
    cfParams.gridMappingName = "polar_stereographic";
    
    // 设置CF投影参数（对应您的NetCDF文件中的实际参数）
    cfParams.numericParameters["latitude_of_projection_origin"] = 90.0;
    cfParams.numericParameters["straight_vertical_longitude_from_pole"] = -45.0;
    cfParams.numericParameters["false_easting"] = 0.0;
    cfParams.numericParameters["false_northing"] = 0.0;
    cfParams.numericParameters["earth_radius"] = 6378273.0;
    cfParams.stringParameters["units"] = "m";
    
    std::cout << "使用CF参数创建极地立体投影CRS..." << std::endl;
    
    auto sourceCRSResult = crsService_->createCRSFromCFParametersAsync(cfParams).get();
    ASSERT_TRUE(sourceCRSResult.has_value()) << "从CF参数创建极地投影CRS失败";
    
    std::cout << "生成的PROJ字符串: " << sourceCRSResult->projString << std::endl;
    
    CRSInfo sourceCRS = sourceCRSResult.value();
    std::cout << "源CRS创建成功: " << sourceCRS.id << std::endl;
    
    // 2. 创建WGS84目标坐标系
    auto targetCRSResult = crsService_->parseFromEpsgCodeAsync(4326).get();
    ASSERT_TRUE(targetCRSResult.has_value()) << "WGS84坐标系创建失败";
    
    CRSInfo targetCRS = targetCRSResult.value();
    std::cout << "目标CRS (WGS84) 创建成功" << std::endl;
    
    // 3. 测试几个关键点的转换
    struct TestPoint {
        std::string name;
        double x, y;           // 投影坐标 (米)
        double expectedLon, expectedLat; // 期望的WGS84坐标 (度) - 大致估计
    };
    
    std::vector<TestPoint> testPoints = {
        {"中心点", 0.0, 0.0, -45.0, 90.0},
        {"北美方向", -1000000.0, -1000000.0, -55.0, 85.0},
        {"欧洲方向", 1000000.0, -1000000.0, -35.0, 85.0},
        {"边界点", -2000000.0, -2000000.0, -70.0, 75.0}
    };
    
    std::cout << "\n--- 坐标转换测试结果 ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    int successCount = 0;
    
    for (const auto& testPoint : testPoints) {
        auto transformResult = crsService_->transformPointAsync(
            testPoint.x, testPoint.y, sourceCRS, targetCRS
        ).get();
        
        std::cout << "\n" << testPoint.name << ":" << std::endl;
        std::cout << "  投影坐标: (" << testPoint.x << ", " << testPoint.y << ") m" << std::endl;
        
        if (transformResult.status == TransformStatus::SUCCESS) {
            double actualLon = transformResult.x;
            double actualLat = transformResult.y;
            
            std::cout << "  转换结果: (" << actualLon << "°, " << actualLat << "°)" << std::endl;
            std::cout << "  ✅ 转换成功" << std::endl;
            successCount++;
            
            // 基本合理性检查
            EXPECT_GE(actualLat, 60.0) << "纬度应该在北极地区";
            EXPECT_LE(actualLat, 90.0) << "纬度不应超过90°";
            EXPECT_GE(actualLon, -180.0) << "经度应该在有效范围内";
            EXPECT_LE(actualLon, 180.0) << "经度应该在有效范围内";
            
        } else {
            std::cout << "  ❌ 转换失败: " << transformResult.errorMessage.value_or("未知错误") << std::endl;
        }
    }
    
    // 至少一半的点应该转换成功
    EXPECT_GE(successCount, testPoints.size() / 2) << "转换成功率过低";
    
    std::cout << "\n转换成功: " << successCount << "/" << testPoints.size() << " 点" << std::endl;
    std::cout << "=== NetCDF非标准极地立体投影基本测试完成 ===" << std::endl;
}

/**
 * @brief 测试反向转换
 */
TEST_F(SimpleNetCDFTest, ReversePolarStereographicConversion) {
    std::cout << "\n=== 反向转换测试 ===" << std::endl;
    
    // 使用EPSG:3995作为测试CRS
    auto sourceCRSResult = crsService_->parseFromEpsgCodeAsync(3995).get();
    ASSERT_TRUE(sourceCRSResult.has_value());
    
    // 创建WGS84
    auto targetCRSResult = crsService_->parseFromEpsgCodeAsync(4326).get();
    ASSERT_TRUE(targetCRSResult.has_value());
    
    CRSInfo polarCRS = sourceCRSResult.value();
    CRSInfo wgs84CRS = targetCRSResult.value();
    
    // 测试从WGS84到极地投影的转换
    double testLon = -45.0; // 中央经线
    double testLat = 80.0;  // 高纬度点
    
    auto transformResult = crsService_->transformPointAsync(
        testLon, testLat, wgs84CRS, polarCRS
    ).get();
    
    if (transformResult.status == TransformStatus::SUCCESS) {
        double projX = transformResult.x;
        double projY = transformResult.y;
        
        std::cout << "WGS84坐标 (" << testLon << "°, " << testLat << "°) -> " 
                  << "投影坐标 (" << projX << ", " << projY << ") m" << std::endl;
        
        // 验证反向转换
        auto reverseResult = crsService_->transformPointAsync(
            projX, projY, polarCRS, wgs84CRS
        ).get();
        
        if (reverseResult.status == TransformStatus::SUCCESS) {
            double backLon = reverseResult.x;
            double backLat = reverseResult.y;
            
            std::cout << "反向转换: (" << backLon << "°, " << backLat << "°)" << std::endl;
            
            // 检查往返转换的精度
            double lonError = std::abs(backLon - testLon);
            double latError = std::abs(backLat - testLat);
            
            std::cout << "经度误差: " << lonError << "°" << std::endl;
            std::cout << "纬度误差: " << latError << "°" << std::endl;
            
            EXPECT_LT(lonError, 0.01) << "经度往返误差过大";
            EXPECT_LT(latError, 0.01) << "纬度往返误差过大";
            
            std::cout << "✅ 反向转换测试成功" << std::endl;
        } else {
            std::cout << "❌ 反向转换失败" << std::endl;
            FAIL() << "反向转换应该成功";
        }
    } else {
        std::cout << "❌ 正向转换失败" << std::endl;
        FAIL() << "正向转换应该成功";
    }
    
    std::cout << "=== 反向转换测试完成 ===" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n🧪 NetCDF非标准极地立体投影简化测试" << std::endl;
    std::cout << "🎯 目标：验证CRS服务对NetCDF特殊投影的支持" << std::endl;
    std::cout << "=============================================\n" << std::endl;
    
    return RUN_ALL_TESTS();
} 