#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include "../src/impl/algorithms/fast_pchip_interpolator_2d.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services::interpolation;
using namespace oscean::core_services;

class PCHIP2DSIMDPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置随机数生成器
        gen_ = std::mt19937(42);
        dist_ = std::uniform_real_distribution<double>(0.0, 100.0);
        coord_dist_ = std::uniform_real_distribution<double>(10.0, 90.0);
    }

    // 创建测试网格数据（必须是Float64，因为FastPchipInterpolator2D要求）
    std::shared_ptr<GridData> createTestGrid(size_t width, size_t height) {
        GridDefinition def;
        def.cols = width;
        def.rows = height;
        
        auto grid = std::make_shared<GridData>(def, DataType::Float64, 1);
        
        // 设置地理变换参数
        std::vector<double> geoTransform = {
            0.0,    // 左上角X坐标
            0.1,    // X方向像素大小
            0.0,    // X方向旋转
            100.0,  // 左上角Y坐标
            0.0,    // Y方向旋转
            -0.1    // Y方向像素大小
        };
        grid->setGeoTransform(geoTransform);
        
        // 填充数据（创建一个平滑的函数，适合PCHIP插值）
        double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t row = 0; row < height; ++row) {
            for (size_t col = 0; col < width; ++col) {
                // 使用一个平滑的2D函数
                double x = col * 0.1;
                double y = row * 0.1;
                data[row * width + col] = std::sin(x * 0.5) * std::cos(y * 0.3) * 50.0 + 50.0;
            }
        }
        
        return grid;
    }

    // 创建随机目标点
    std::vector<TargetPoint> createRandomPoints(size_t count) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            point.coordinates = {coord_dist_(gen_), coord_dist_(gen_)};
            points.push_back(point);
        }
        
        return points;
    }

    // 计时辅助函数
    template<typename Func>
    double measureTime(Func&& func, int iterations = 1) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return static_cast<double>(duration.count()) / iterations;
    }

private:
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dist_;
    std::uniform_real_distribution<double> coord_dist_;
};

// PCHIP 2D SIMD性能测试
TEST_F(PCHIP2DSIMDPerformanceTest, FastPCHIP2DPerformance) {
    std::cout << "\n=== Fast PCHIP 2D SIMD性能测试 ===" << std::endl;
    
    // 创建不同大小的测试网格
    std::vector<std::pair<size_t, std::shared_ptr<GridData>>> grids = {
        {100, createTestGrid(100, 100)},
        {500, createTestGrid(500, 500)},
        {1000, createTestGrid(1000, 1000)}
    };
    
    // 测试不同数量的点
    std::vector<size_t> pointCounts = {10, 100, 1000, 10000};
    
    for (const auto& [gridSize, grid] : grids) {
        std::cout << "\n网格大小: " << gridSize << "x" << gridSize << std::endl;
        
        // 创建插值器（预计算导数）
        auto setupTime = measureTime([&]() {
            auto interpolator = std::make_unique<FastPchipInterpolator2D>(grid);
        });
        
        std::cout << "  预计算时间: " << std::fixed << std::setprecision(2) 
                 << setupTime / 1000.0 << " ms" << std::endl;
        
        // 创建插值器用于测试
        auto interpolator = std::make_unique<FastPchipInterpolator2D>(grid);
        
        for (size_t pointCount : pointCounts) {
            auto points = createRandomPoints(pointCount);
            
            // 创建插值请求
            InterpolationRequest request;
            request.sourceGrid = grid;  // 虽然FastPchipInterpolator2D不使用这个
            request.target = points;
            request.method = interpolation::InterpolationMethod::PCHIP_RECURSIVE_NDIM;
            
            // 测试标量路径（少量点）
            if (pointCount <= 10) {
                auto scalarTime = measureTime([&]() {
                    auto smallPoints = std::vector<TargetPoint>(points.begin(), points.begin() + 4);
                    InterpolationRequest smallRequest = request;
                    smallRequest.target = smallPoints;
                    interpolator->execute(smallRequest, nullptr);
                }, 100);
                
                std::cout << "  标量路径 (" << 4 << " 点): " 
                         << std::fixed << std::setprecision(2) 
                         << scalarTime / 4 << " ns/点" << std::endl;
            }
            
            // 测试SIMD路径
            auto simdTime = measureTime([&]() {
                interpolator->execute(request, nullptr);
            }, pointCount > 1000 ? 5 : 10);
            
            std::cout << "  SIMD路径 (" << pointCount << " 点): " 
                     << std::fixed << std::setprecision(2) 
                     << simdTime / pointCount << " ns/点" << std::endl;
        }
    }
}

// 精度验证测试
TEST_F(PCHIP2DSIMDPerformanceTest, AccuracyValidation) {
    std::cout << "\n=== PCHIP 2D精度验证测试 ===" << std::endl;
    
    // 创建一个较大的测试网格（10x10），以便有更好的插值精度
    GridDefinition def;
    def.cols = 10;
    def.rows = 10;
    
    auto grid = std::make_shared<GridData>(def, DataType::Float64, 1);
    
    // 设置地理变换参数
    std::vector<double> geoTransform = {
        0.0,    // 左上角X坐标
        1.0,    // X方向像素大小
        0.0,    // X方向旋转
        0.0,    // 左上角Y坐标
        0.0,    // Y方向旋转
        1.0     // Y方向像素大小（正值，因为我们使用行主序）
    };
    grid->setGeoTransform(geoTransform);
    
    // 填充已知数据（使用平滑的函数，适合PCHIP插值）
    double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
    for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < 10; ++col) {
            // 使用平滑的函数：f(x,y) = sin(x/2) * cos(y/2) * 10 + 20
            double x = col;
            double y = row;
            data[row * 10 + col] = std::sin(x * 0.5) * std::cos(y * 0.5) * 10.0 + 20.0;
        }
    }
    
    // 创建插值器
    auto interpolator = std::make_unique<FastPchipInterpolator2D>(grid);
    
    // 测试点（在网格范围内）
    std::vector<TargetPoint> testPoints;
    std::vector<double> expectedValues;
    
    // 测试1：网格点上（应该精确匹配）
    testPoints.push_back({{2.0, 3.0}});
    expectedValues.push_back(std::sin(2.0 * 0.5) * std::cos(3.0 * 0.5) * 10.0 + 20.0);
    
    // 测试2：网格中心
    testPoints.push_back({{2.5, 3.5}});
    expectedValues.push_back(std::sin(2.5 * 0.5) * std::cos(3.5 * 0.5) * 10.0 + 20.0);
    
    // 测试3：接近原点
    testPoints.push_back({{0.5, 0.5}});
    expectedValues.push_back(std::sin(0.5 * 0.5) * std::cos(0.5 * 0.5) * 10.0 + 20.0);
    
    // 测试4：接近边界（但在范围内）
    testPoints.push_back({{8.5, 8.5}});
    expectedValues.push_back(std::sin(8.5 * 0.5) * std::cos(8.5 * 0.5) * 10.0 + 20.0);
    
    InterpolationRequest request;
    request.sourceGrid = grid;
    request.target = testPoints;
    request.method = interpolation::InterpolationMethod::PCHIP_RECURSIVE_NDIM;
    
    auto result = interpolator->execute(request, nullptr);
    auto values = std::get<std::vector<std::optional<double>>>(result.data);
    
    std::cout << "\nPCHIP 2D插值结果:" << std::endl;
    std::cout << "点坐标 | 插值结果 | 期望值 | 绝对误差 | 相对误差(%)" << std::endl;
    std::cout << "-------|---------|--------|----------|------------" << std::endl;
    
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    
    for (size_t i = 0; i < testPoints.size(); ++i) {
        std::cout << "(" << std::setw(3) << testPoints[i].coordinates[0] << ", " 
                 << std::setw(3) << testPoints[i].coordinates[1] << ") | ";
        
        if (values[i].has_value()) {
            double interpolated = values[i].value();
            double expected = expectedValues[i];
            double absError = std::abs(interpolated - expected);
            double relError = std::abs(absError / expected) * 100.0;
            
            maxAbsError = std::max(maxAbsError, absError);
            maxRelError = std::max(maxRelError, relError);
            
            std::cout << std::setw(7) << std::fixed << std::setprecision(4) << interpolated
                     << " | " << std::setw(6) << expected
                     << " | " << std::setw(8) << absError
                     << " | " << std::setw(10) << relError << std::endl;
        } else {
            std::cout << "  无效  |    -   |    -     |     -" << std::endl;
        }
    }
    
    std::cout << "\n最大绝对误差: " << maxAbsError << std::endl;
    std::cout << "最大相对误差: " << maxRelError << "%" << std::endl;
    
    // 验证精度
    EXPECT_LT(maxAbsError, 0.15) << "绝对误差应小于0.15";
    EXPECT_LT(maxRelError, 1.0) << "相对误差应小于1%";
}

// 导数预计算性能分析
TEST_F(PCHIP2DSIMDPerformanceTest, DerivativePrecomputationAnalysis) {
    std::cout << "\n=== 导数预计算性能分析 ===" << std::endl;
    
    std::vector<size_t> gridSizes = {50, 100, 200, 500, 1000};
    
    std::cout << "\n网格大小 | 预计算时间(ms) | 每点时间(μs)" << std::endl;
    std::cout << "---------|----------------|-------------" << std::endl;
    
    for (size_t size : gridSizes) {
        auto grid = createTestGrid(size, size);
        
        auto precomputeTime = measureTime([&]() {
            auto interpolator = std::make_unique<FastPchipInterpolator2D>(grid);
        }, 5);
        
        double timePerPoint = precomputeTime / (size * size);
        
        std::cout << std::setw(8) << size << " | " 
                 << std::setw(14) << std::fixed << std::setprecision(2) << precomputeTime / 1000.0
                 << " | " 
                 << std::setw(11) << std::fixed << std::setprecision(3) << timePerPoint
                 << std::endl;
    }
    
    std::cout << "\n注：预计算包括X、Y方向导数和XY混合导数的计算" << std::endl;
} 