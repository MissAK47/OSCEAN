#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include "../src/impl/algorithms/cubic_spline_interpolator.h"
#include "../src/impl/algorithms/nearest_neighbor_interpolator.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services::interpolation;
using namespace oscean::core_services;

class CubicNearestSIMDPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置随机数生成器
        gen_ = std::mt19937(42);
        dist_ = std::uniform_real_distribution<float>(0.0f, 100.0f);
        coord_dist_ = std::uniform_real_distribution<double>(10.0, 90.0);
    }

    // 创建测试网格数据
    std::shared_ptr<GridData> createTestGrid(size_t width, size_t height, size_t bands = 1, DataType dataType = DataType::Float32) {
        GridDefinition def;
        def.cols = width;
        def.rows = height;
        
        auto grid = std::make_shared<GridData>(def, dataType, bands);
        
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
        
        // 填充数据
        size_t totalSize = width * height * bands;
        if (dataType == DataType::Float32) {
            float* data = static_cast<float*>(const_cast<void*>(grid->getDataPtr()));
            for (size_t i = 0; i < totalSize; ++i) {
                data[i] = dist_(gen_);
            }
        } else if (dataType == DataType::Float64) {
            double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
            for (size_t i = 0; i < totalSize; ++i) {
                data[i] = static_cast<double>(dist_(gen_));
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
    std::uniform_real_distribution<float> dist_;
    std::uniform_real_distribution<double> coord_dist_;
};

// 立方样条插值SIMD性能测试
TEST_F(CubicNearestSIMDPerformanceTest, CubicSplineSIMDPerformance) {
    std::cout << "\n=== 立方样条插值SIMD性能测试 ===" << std::endl;
    
    // 创建不同大小的测试网格
    auto smallGrid = createTestGrid(100, 100, 1, DataType::Float32);
    auto mediumGrid = createTestGrid(500, 500, 1, DataType::Float32);
    auto largeGrid = createTestGrid(1000, 1000, 1, DataType::Float32);
    
    // 创建插值器
    auto interpolator = std::make_unique<CubicSplineInterpolator>();
    
    // 测试不同数量的点
    std::vector<size_t> pointCounts = {10, 100, 1000, 10000};
    
    for (auto grid : {smallGrid, mediumGrid, largeGrid}) {
        size_t gridSize = grid->getDefinition().cols;
        std::cout << "\n网格大小: " << gridSize << "x" << gridSize << std::endl;
        
        for (size_t pointCount : pointCounts) {
            auto points = createRandomPoints(pointCount);
            
            // 创建插值请求
            InterpolationRequest request;
            request.sourceGrid = grid;
            request.target = points;
            request.method = interpolation::InterpolationMethod::CUBIC_SPLINE_1D;
            
            // 测试标量路径（少量点）
            if (pointCount <= 10) {
                auto scalarTime = measureTime([&]() {
                    auto smallPoints = std::vector<TargetPoint>(points.begin(), points.begin() + 4);
                    InterpolationRequest smallRequest = request;
                    smallRequest.target = smallPoints;
                    interpolator->execute(smallRequest);
                }, 100);
                
                std::cout << "  标量路径 (" << 4 << " 点): " 
                         << std::fixed << std::setprecision(2) 
                         << scalarTime / 4 << " ns/点" << std::endl;
            }
            
            // 测试SIMD路径
            auto simdTime = measureTime([&]() {
                interpolator->execute(request);
            }, 10);
            
            std::cout << "  SIMD路径 (" << pointCount << " 点): " 
                     << std::fixed << std::setprecision(2) 
                     << simdTime / pointCount << " ns/点" << std::endl;
        }
    }
}

// 最近邻插值SIMD性能测试
TEST_F(CubicNearestSIMDPerformanceTest, NearestNeighborSIMDPerformance) {
    std::cout << "\n=== 最近邻插值SIMD性能测试 ===" << std::endl;
    
    // 创建不同大小的测试网格
    auto smallGrid = createTestGrid(100, 100, 1, DataType::Float32);
    auto mediumGrid = createTestGrid(500, 500, 1, DataType::Float32);
    auto largeGrid = createTestGrid(1000, 1000, 1, DataType::Float32);
    
    // 创建插值器
    auto interpolator = std::make_unique<NearestNeighborInterpolator>();
    
    // 测试不同数量的点
    std::vector<size_t> pointCounts = {10, 100, 1000, 10000, 100000};
    
    for (auto grid : {smallGrid, mediumGrid, largeGrid}) {
        size_t gridSize = grid->getDefinition().cols;
        std::cout << "\n网格大小: " << gridSize << "x" << gridSize << std::endl;
        
        for (size_t pointCount : pointCounts) {
            auto points = createRandomPoints(pointCount);
            
            // 创建插值请求
            InterpolationRequest request;
            request.sourceGrid = grid;
            request.target = points;
            request.method = interpolation::InterpolationMethod::NEAREST_NEIGHBOR;
            
            // 测试标量路径（少量点）
            if (pointCount <= 10) {
                auto scalarTime = measureTime([&]() {
                    auto smallPoints = std::vector<TargetPoint>(points.begin(), points.begin() + 4);
                    InterpolationRequest smallRequest = request;
                    smallRequest.target = smallPoints;
                    interpolator->execute(smallRequest);
                }, 100);
                
                std::cout << "  标量路径 (" << 4 << " 点): " 
                         << std::fixed << std::setprecision(2) 
                         << scalarTime / 4 << " ns/点" << std::endl;
            }
            
            // 测试SIMD路径
            auto simdTime = measureTime([&]() {
                interpolator->execute(request);
            }, pointCount > 10000 ? 1 : 10);
            
            std::cout << "  SIMD路径 (" << pointCount << " 点): " 
                     << std::fixed << std::setprecision(2) 
                     << simdTime / pointCount << " ns/点" << std::endl;
        }
    }
}

// 不同数据类型的性能对比
TEST_F(CubicNearestSIMDPerformanceTest, DataTypePerformanceComparison) {
    std::cout << "\n=== 数据类型性能对比 ===" << std::endl;
    
    size_t gridSize = 1000;
    size_t pointCount = 10000;
    
    // Float32网格
    auto float32Grid = createTestGrid(gridSize, gridSize, 1, DataType::Float32);
    // Float64网格
    auto float64Grid = createTestGrid(gridSize, gridSize, 1, DataType::Float64);
    
    auto points = createRandomPoints(pointCount);
    
    // 测试立方样条
    {
        auto interpolator = std::make_unique<CubicSplineInterpolator>();
        
        std::cout << "\n立方样条插值:" << std::endl;
        
        // Float32
        InterpolationRequest request32;
        request32.sourceGrid = float32Grid;
        request32.target = points;
        request32.method = interpolation::InterpolationMethod::CUBIC_SPLINE_1D;
        
        auto time32 = measureTime([&]() {
            interpolator->execute(request32);
        }, 10);
        
        std::cout << "  Float32: " << std::fixed << std::setprecision(2) 
                 << time32 / pointCount << " ns/点" << std::endl;
        
        // Float64
        InterpolationRequest request64;
        request64.sourceGrid = float64Grid;
        request64.target = points;
        request64.method = interpolation::InterpolationMethod::CUBIC_SPLINE_1D;
        
        auto time64 = measureTime([&]() {
            interpolator->execute(request64);
        }, 10);
        
        std::cout << "  Float64: " << std::fixed << std::setprecision(2) 
                 << time64 / pointCount << " ns/点" << std::endl;
    }
    
    // 测试最近邻
    {
        auto interpolator = std::make_unique<NearestNeighborInterpolator>();
        
        std::cout << "\n最近邻插值:" << std::endl;
        
        // Float32
        InterpolationRequest request32;
        request32.sourceGrid = float32Grid;
        request32.target = points;
        request32.method = interpolation::InterpolationMethod::NEAREST_NEIGHBOR;
        
        auto time32 = measureTime([&]() {
            interpolator->execute(request32);
        }, 10);
        
        std::cout << "  Float32: " << std::fixed << std::setprecision(2) 
                 << time32 / pointCount << " ns/点" << std::endl;
        
        // Float64
        InterpolationRequest request64;
        request64.sourceGrid = float64Grid;
        request64.target = points;
        request64.method = interpolation::InterpolationMethod::NEAREST_NEIGHBOR;
        
        auto time64 = measureTime([&]() {
            interpolator->execute(request64);
        }, 10);
        
        std::cout << "  Float64: " << std::fixed << std::setprecision(2) 
                 << time64 / pointCount << " ns/点" << std::endl;
    }
}

// 精度验证测试
TEST_F(CubicNearestSIMDPerformanceTest, AccuracyValidation) {
    std::cout << "\n=== 精度验证测试 ===" << std::endl;
    
    // 创建一个简单的测试网格（4x4）
    GridDefinition def;
    def.cols = 4;
    def.rows = 4;
    
    auto grid = std::make_shared<GridData>(def, DataType::Float32, 1);
    
    // 设置地理变换参数
    std::vector<double> geoTransform = {0.0, 1.0, 0.0, 4.0, 0.0, -1.0};
    grid->setGeoTransform(geoTransform);
    
    // 填充已知数据
    float* data = static_cast<float*>(const_cast<void*>(grid->getDataPtr()));
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            data[row * 4 + col] = static_cast<float>(row * 10 + col);
        }
    }
    
    // 测试点
    std::vector<TargetPoint> testPoints;
    testPoints.push_back({{1.5, 2.5}});  // 网格中心
    testPoints.push_back({{0.0, 0.0}});  // 角点
    testPoints.push_back({{3.0, 3.0}});  // 另一个角点
    
    // 立方样条插值
    {
        auto interpolator = std::make_unique<CubicSplineInterpolator>();
        
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = testPoints;
        request.method = interpolation::InterpolationMethod::CUBIC_SPLINE_1D;
        
        auto result = interpolator->execute(request);
        auto values = std::get<std::vector<std::optional<double>>>(result.data);
        
        std::cout << "\n立方样条插值结果:" << std::endl;
        for (size_t i = 0; i < testPoints.size(); ++i) {
            std::cout << "  点(" << testPoints[i].coordinates[0] << ", " 
                     << testPoints[i].coordinates[1] << "): ";
            if (values[i].has_value()) {
                std::cout << values[i].value() << std::endl;
            } else {
                std::cout << "无效" << std::endl;
            }
        }
    }
    
    // 最近邻插值
    {
        auto interpolator = std::make_unique<NearestNeighborInterpolator>();
        
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = testPoints;
        request.method = interpolation::InterpolationMethod::NEAREST_NEIGHBOR;
        
        auto result = interpolator->execute(request);
        auto values = std::get<std::vector<std::optional<double>>>(result.data);
        
        std::cout << "\n最近邻插值结果:" << std::endl;
        for (size_t i = 0; i < testPoints.size(); ++i) {
            std::cout << "  点(" << testPoints[i].coordinates[0] << ", " 
                     << testPoints[i].coordinates[1] << "): ";
            if (values[i].has_value()) {
                std::cout << values[i].value() << std::endl;
            } else {
                std::cout << "无效" << std::endl;
            }
        }
    }
} 