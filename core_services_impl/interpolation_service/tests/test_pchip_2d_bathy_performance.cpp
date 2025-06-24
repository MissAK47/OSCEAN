#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../src/impl/algorithms/pchip_interpolator_2d_bathy.h"
#include "core_services/common_data_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace oscean::core_services::interpolation;
using namespace oscean::core_services;

class PCHIP2DBathyPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_ = std::mt19937(42);
        dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }
    
    // 创建模拟海底地形的测试网格
    GridData createBathymetryGrid(size_t width, size_t height) {
        GridDefinition def;
        def.cols = width;
        def.rows = height;
        def.bands = 1;
        def.dataType = DataType::Float32;
        
        std::vector<float> data(width * height);
        
        // 生成类似海底地形的数据：基础深度 + 海山/海沟特征
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                double nx = static_cast<double>(x) / width;
                double ny = static_cast<double>(y) / height;
                
                // 基础深度（-3000米到-5000米）
                double depth = -4000.0;
                
                // 添加大尺度地形特征
                depth += 1000.0 * std::sin(2 * M_PI * nx) * std::cos(2 * M_PI * ny);
                
                // 添加海山（高斯分布）
                double cx = 0.3, cy = 0.7;
                double dist = std::sqrt((nx - cx) * (nx - cx) + (ny - cy) * (ny - cy));
                depth += 1500.0 * std::exp(-dist * dist / 0.01);
                
                // 添加海沟
                if (std::abs(nx - 0.7) < 0.05) {
                    depth -= 1000.0 * (1.0 - std::abs(nx - 0.7) / 0.05);
                }
                
                // 添加小尺度噪声
                depth += 50.0 * (dist_() - 0.5);
                
                data[y * width + x] = static_cast<float>(depth);
            }
        }
        
        return GridData(def, std::move(data));
    }
    
    // 生成随机测试点
    std::vector<TargetPoint> generateRandomPoints(size_t count, size_t gridWidth, size_t gridHeight) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            // 生成在有效范围内的坐标（避开边界）
            point.coordinates = {
                2.0 + dist_() * (gridWidth - 4.0),
                2.0 + dist_() * (gridHeight - 4.0)
            };
            points.push_back(point);
        }
        
        return points;
    }
    
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dist_;
};

TEST_F(PCHIP2DBathyPerformanceTest, PerformanceComparison) {
    std::cout << "\n=== PCHIP 2D Bathymetry 性能测试 ===\n" << std::endl;
    
    // 测试不同大小的海底地形网格
    std::vector<std::pair<size_t, size_t>> gridSizes = {
        {50, 50},     // 小型：沿海区域
        {200, 200},   // 中型：区域海底
        {500, 500}    // 大型：大洋海底
    };
    
    std::vector<size_t> pointCounts = {100, 1000, 10000};
    
    for (const auto& [width, height] : gridSizes) {
        std::cout << "\n网格大小: " << width << "x" << height 
                  << " (深度点: " << width * height << ")" << std::endl;
        
        auto grid = createBathymetryGrid(width, height);
        auto interpolator = std::make_unique<PCHIPInterpolator2DBathy>(nullptr);
        
        for (size_t numPoints : pointCounts) {
            auto points = generateRandomPoints(numPoints, width, height);
            
            InterpolationRequest req;
            req.sourceGrid = &grid;
            req.target = points;
            
            // 预热
            for (int i = 0; i < 3; ++i) {
                interpolator->execute(req, nullptr);
            }
            
            // 性能测试
            const int iterations = 10;
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; ++i) {
                auto result = interpolator->execute(req, nullptr);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            double avgTimeMs = duration / 1000.0 / iterations;
            double timePerPoint = duration / static_cast<double>(iterations * numPoints);
            double throughput = numPoints / (avgTimeMs / 1000.0) / 1e6;
            
            std::cout << "  " << std::setw(6) << numPoints << " 点: "
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << avgTimeMs << " ms, "
                      << std::setw(8) << timePerPoint << " ns/点, "
                      << std::setw(6) << throughput << " M点/秒" << std::endl;
        }
    }
}

TEST_F(PCHIP2DBathyPerformanceTest, AccuracyValidation) {
    std::cout << "\n=== 精度验证 ===\n" << std::endl;
    
    // 创建一个已知函数的网格（海底地形模型）
    size_t gridSize = 20;
    GridDefinition def;
    def.cols = gridSize;
    def.rows = gridSize;
    def.bands = 1;
    def.dataType = DataType::Float32;
    
    std::vector<float> data(gridSize * gridSize);
    
    // 使用简单的二次函数模拟海底地形
    for (size_t y = 0; y < gridSize; ++y) {
        for (size_t x = 0; x < gridSize; ++x) {
            double nx = static_cast<double>(x) / (gridSize - 1);
            double ny = static_cast<double>(y) / (gridSize - 1);
            // 碗状地形
            double depth = -3000.0 - 1000.0 * (nx - 0.5) * (nx - 0.5) 
                                  - 1000.0 * (ny - 0.5) * (ny - 0.5);
            data[y * gridSize + x] = static_cast<float>(depth);
        }
    }
    
    GridData grid(def, std::move(data));
    auto interpolator = std::make_unique<PCHIPInterpolator2DBathy>(nullptr);
    
    // 在非网格点上测试
    std::vector<TargetPoint> testPoints;
    std::vector<double> expectedValues;
    
    for (int i = 0; i < 10; ++i) {
        double x = 2.5 + i * 1.5;
        double y = 2.5 + i * 1.5;
        
        if (x < gridSize - 2 && y < gridSize - 2) {
            TargetPoint point;
            point.coordinates = {x, y};
            testPoints.push_back(point);
            
            // 计算期望值
            double nx = x / (gridSize - 1);
            double ny = y / (gridSize - 1);
            double expected = -3000.0 - 1000.0 * (nx - 0.5) * (nx - 0.5) 
                                     - 1000.0 * (ny - 0.5) * (ny - 0.5);
            expectedValues.push_back(expected);
        }
    }
    
    InterpolationRequest req;
    req.sourceGrid = &grid;
    req.target = testPoints;
    
    auto result = interpolator->execute(req, nullptr);
    
    std::cout << "测试点数: " << testPoints.size() << std::endl;
    
    double maxError = 0.0;
    double avgError = 0.0;
    
    for (size_t i = 0; i < testPoints.size(); ++i) {
        if (result.values[i].has_value()) {
            double error = std::abs(result.values[i].value() - expectedValues[i]);
            double relError = error / std::abs(expectedValues[i]) * 100.0;
            
            maxError = std::max(maxError, error);
            avgError += error;
            
            std::cout << "点 " << i << ": "
                      << "期望=" << std::fixed << std::setprecision(1) << expectedValues[i]
                      << ", 实际=" << result.values[i].value()
                      << ", 误差=" << std::setprecision(2) << error
                      << " (" << relError << "%)" << std::endl;
        }
    }
    
    avgError /= testPoints.size();
    
    std::cout << "\n最大误差: " << maxError << " 米" << std::endl;
    std::cout << "平均误差: " << avgError << " 米" << std::endl;
    
    // 对于海底地形，误差在100米以内是可接受的
    EXPECT_LT(maxError, 100.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 