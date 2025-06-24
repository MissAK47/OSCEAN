#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../src/impl/algorithms/pchip_interpolator.h"
#include "core_services/common_data_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace oscean::core_services::interpolation;
using namespace oscean::core_services;

class PCHIPBasicSIMDPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_ = std::mt19937(42);
        dist_ = std::uniform_real_distribution<double>(0.0, 100.0);
        coord_dist_ = std::uniform_real_distribution<double>(10.0, 90.0);
    }

    // 创建测试网格数据
    std::shared_ptr<GridData> createTestGrid(size_t width, size_t height) {
        GridDefinition def;
        def.cols = width;
        def.rows = height;
        
        auto grid = std::make_shared<GridData>(def, DataType::Float64, 1);
        
        // 设置地理变换参数
        std::vector<double> geoTransform = {
            0.0,    // 左上角X坐标
            1.0,    // X方向像素大小
            0.0,    // X方向旋转
            0.0,    // 左上角Y坐标
            0.0,    // Y方向旋转
            1.0     // Y方向像素大小
        };
        grid->setGeoTransform(geoTransform);
        
        // 填充测试数据 - 使用平滑函数
        auto* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                // 使用平滑的2D函数
                double fx = static_cast<double>(x) / width;
                double fy = static_cast<double>(y) / height;
                data[y * width + x] = 50.0 + 30.0 * std::sin(2 * M_PI * fx) * 
                                             std::cos(2 * M_PI * fy);
            }
        }
        
        return grid;
    }

    // 生成随机测试点
    std::vector<TargetPoint> generateRandomPoints(size_t count, double maxX, double maxY) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            point.coordinates = {coord_dist_(gen_) * maxX / 100.0, 
                               coord_dist_(gen_) * maxY / 100.0};
            points.push_back(point);
        }
        
        return points;
    }

private:
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dist_;
    std::uniform_real_distribution<double> coord_dist_;
};

TEST_F(PCHIPBasicSIMDPerformanceTest, PerformanceComparison) {
    std::cout << "\n=== 基础PCHIP插值SIMD性能测试 ===\n" << std::endl;
    
    // 测试不同大小的网格
    std::vector<std::pair<size_t, size_t>> gridSizes = {
        {100, 100},    // 小网格
        {500, 500},    // 中等网格
        {1000, 1000}   // 大网格
    };
    
    std::vector<size_t> pointCounts = {100, 1000, 10000};
    
    auto interpolator = std::make_unique<PCHIPInterpolator>(nullptr);
    
    for (const auto& [width, height] : gridSizes) {
        auto grid = createTestGrid(width, height);
        
        std::cout << "网格大小: " << width << "x" << height << std::endl;
        
        for (size_t numPoints : pointCounts) {
            auto points = generateRandomPoints(numPoints, width - 1, height - 1);
            
            // 预热
            InterpolationRequest warmupReq;
            warmupReq.sourceGrid = grid;
            warmupReq.target = std::vector<TargetPoint>(points.begin(), points.begin() + 10);
            interpolator->execute(warmupReq, nullptr);
            
            // 测试标量版本（强制使用小批量）
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<std::optional<double>> scalarResults;
            for (const auto& point : points) {
                InterpolationRequest req;
                req.sourceGrid = grid;
                req.target = std::vector<TargetPoint>{point};
                auto result = interpolator->execute(req, nullptr);
                if (std::holds_alternative<std::vector<std::optional<double>>>(result.data)) {
                    auto& values = std::get<std::vector<std::optional<double>>>(result.data);
                    if (!values.empty()) {
                        scalarResults.push_back(values[0]);
                    }
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto scalarTime = std::chrono::duration<double, std::milli>(end - start).count();
            
            // 测试SIMD版本
            InterpolationRequest simdReq;
            simdReq.sourceGrid = grid;
            simdReq.target = points;
            
            start = std::chrono::high_resolution_clock::now();
            auto simdResult = interpolator->execute(simdReq, nullptr);
            end = std::chrono::high_resolution_clock::now();
            auto simdTime = std::chrono::duration<double, std::milli>(end - start).count();
            
            std::vector<std::optional<double>> simdResults;
            if (std::holds_alternative<std::vector<std::optional<double>>>(simdResult.data)) {
                simdResults = std::get<std::vector<std::optional<double>>>(simdResult.data);
            }
            
            // 计算性能指标
            double scalarNsPerPoint = (scalarTime * 1e6) / numPoints;
            double simdNsPerPoint = (simdTime * 1e6) / numPoints;
            double speedup = scalarTime / simdTime;
            double throughput = numPoints / (simdTime * 1e-3) / 1e6; // M点/秒
            
            std::cout << "  插值点数: " << numPoints << std::endl;
            std::cout << "    标量路径: " << std::fixed << std::setprecision(2) 
                      << scalarNsPerPoint << " ns/点" << std::endl;
            std::cout << "    SIMD路径: " << simdNsPerPoint << " ns/点" << std::endl;
            std::cout << "    性能提升: " << speedup << "倍" << std::endl;
            std::cout << "    吞吐量: " << throughput << " M点/秒" << std::endl;
            
            // 验证结果一致性
            size_t mismatches = 0;
            double maxError = 0.0;
            for (size_t i = 0; i < numPoints; ++i) {
                if (scalarResults[i].has_value() && simdResults[i].has_value()) {
                    double error = std::abs(scalarResults[i].value() - simdResults[i].value());
                    maxError = std::max(maxError, error);
                    if (error > 1e-10) {
                        mismatches++;
                    }
                } else if (scalarResults[i].has_value() != simdResults[i].has_value()) {
                    mismatches++;
                }
            }
            
            std::cout << "    结果验证: ";
            if (mismatches == 0) {
                std::cout << "通过 (最大误差: " << std::scientific << maxError << ")" << std::endl;
            } else {
                std::cout << "失败 (" << mismatches << " 个不匹配)" << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

TEST_F(PCHIPBasicSIMDPerformanceTest, AccuracyValidation) {
    std::cout << "\n=== PCHIP插值精度验证 ===\n" << std::endl;
    
    // 创建一个已知函数的网格
    size_t gridSize = 20;
    auto grid = createTestGrid(gridSize, gridSize);
    
    auto interpolator = std::make_unique<PCHIPInterpolator>(nullptr);
    
    // 在内部网格点上测试（避开边缘，因为PCHIP在边缘的行为不同）
    std::vector<TargetPoint> gridPoints;
    for (size_t y = 1; y < gridSize - 1; ++y) {
        for (size_t x = 1; x < gridSize - 1; ++x) {
            TargetPoint point;
            point.coordinates = {static_cast<double>(x), static_cast<double>(y)};
            gridPoints.push_back(point);
        }
    }
    
    InterpolationRequest req;
    req.sourceGrid = grid;
    req.target = gridPoints;
    
    auto result = interpolator->execute(req, nullptr);
    
    std::vector<std::optional<double>> results;
    if (std::holds_alternative<std::vector<std::optional<double>>>(result.data)) {
        results = std::get<std::vector<std::optional<double>>>(result.data);
    }
    
    // 验证精度
    auto* data = static_cast<const double*>(grid->getDataPtr());
    size_t exactMatches = 0;
    double maxError = 0.0;
    size_t validResults = 0;
    
    for (size_t i = 0; i < gridPoints.size(); ++i) {
        if (results[i].has_value()) {
            validResults++;
            size_t x = static_cast<size_t>(gridPoints[i].coordinates[0]);
            size_t y = static_cast<size_t>(gridPoints[i].coordinates[1]);
            double expected = data[y * gridSize + x];
            double actual = results[i].value();
            double error = std::abs(expected - actual);
            
            if (error < 1e-10) {
                exactMatches++;
            }
            maxError = std::max(maxError, error);
            
            if (i < 5) {  // 打印前几个结果
                std::cout << "  点 " << i << ": 期望=" << std::fixed << std::setprecision(2) 
                          << expected << ", 实际=" << actual 
                          << ", 误差=" << std::scientific << error << std::endl;
            }
        }
    }
    
    std::cout << "\n精度统计:" << std::endl;
    std::cout << "  有效结果: " << validResults << "/" << gridPoints.size() << std::endl;
    std::cout << "  精确匹配的网格点: " << exactMatches << "/" << validResults << std::endl;
    std::cout << "  最大绝对误差: " << std::scientific << maxError << std::endl;
    
    // 对于PCHIP，我们期望所有内部点都应该精确匹配
    EXPECT_EQ(exactMatches, validResults);
    EXPECT_LT(maxError, 1e-10);
}

TEST_F(PCHIPBasicSIMDPerformanceTest, BoundaryConditionTest) {
    std::cout << "\n=== 边界条件测试 ===\n" << std::endl;
    
    auto grid = createTestGrid(10, 10);
    auto interpolator = std::make_unique<PCHIPInterpolator>(nullptr);
    
    // 测试边界附近的点
    // 注意：PCHIP需要在边界内至少1个单元格，所以有效范围是[0, 8]x[0, 8]
    std::vector<TargetPoint> boundaryPoints = {
        {{0.1, 0.1}},      // 接近原点（有效）
        {{8.5, 8.5}},      // 接近最大有效值（有效）
        {{8.9, 8.9}},      // 接近边界（有效）
        {{9.0, 5.0}},      // X在边界上（无效）
        {{5.0, 9.0}},      // Y在边界上（无效）
        {{-0.1, 5.0}},     // X越界（无效）
        {{5.0, -0.1}},     // Y越界（无效）
        {{10.1, 5.0}},     // X越界（无效）
        {{5.0, 10.1}},     // Y越界（无效）
    };
    
    InterpolationRequest req;
    req.sourceGrid = grid;
    req.target = boundaryPoints;
    
    auto result = interpolator->execute(req, nullptr);
    
    std::vector<std::optional<double>> results;
    if (std::holds_alternative<std::vector<std::optional<double>>>(result.data)) {
        results = std::get<std::vector<std::optional<double>>>(result.data);
    }
    
    // 验证边界处理
    EXPECT_TRUE(results[0].has_value());   // 内部点应该有值
    EXPECT_TRUE(results[1].has_value());   // 内部点应该有值
    EXPECT_TRUE(results[2].has_value());   // 内部点应该有值
    EXPECT_FALSE(results[3].has_value());  // 边界点应该无值
    EXPECT_FALSE(results[4].has_value());  // 边界点应该无值
    EXPECT_FALSE(results[5].has_value());  // 越界点应该无值
    EXPECT_FALSE(results[6].has_value());  // 越界点应该无值
    EXPECT_FALSE(results[7].has_value());  // 越界点应该无值
    EXPECT_FALSE(results[8].has_value());  // 越界点应该无值
    
    std::cout << "  边界条件处理正确" << std::endl;
    
    // 打印详细信息以调试
    for (size_t i = 0; i < boundaryPoints.size(); ++i) {
        std::cout << "  点 " << i << " (" << boundaryPoints[i].coordinates[0] 
                  << ", " << boundaryPoints[i].coordinates[1] << "): "
                  << (results[i].has_value() ? "有值" : "无值") << std::endl;
    }
} 