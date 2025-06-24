#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include "../src/impl/algorithms/linear_1d_interpolator.h"
#include "../src/impl/algorithms/trilinear_interpolator.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services::interpolation;
// 不再使用 using namespace oscean::core_services; 避免冲突
using oscean::core_services::GridData;
using oscean::core_services::GridDefinition;
using oscean::core_services::DataType;

class Linear1DTrilinearSIMDPerformanceTest : public ::testing::Test {
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
            0.0,    // 左上角X
            1.0,    // X方向像素大小
            0.0,    // X旋转
            100.0,  // 左上角Y
            0.0,    // Y旋转
            -1.0    // Y方向像素大小
        };
        grid->setGeoTransform(geoTransform);
        
        // 填充测试数据
        size_t totalSize = width * height * bands;
        if (dataType == DataType::Float32) {
            float* data = new float[totalSize];
            for (size_t i = 0; i < totalSize; ++i) {
                data[i] = dist_(gen_);
            }
            std::memcpy(const_cast<void*>(grid->getDataPtr()), data, totalSize * sizeof(float));
            delete[] data;
        } else if (dataType == DataType::Float64) {
            double* data = new double[totalSize];
            for (size_t i = 0; i < totalSize; ++i) {
                data[i] = static_cast<double>(dist_(gen_));
            }
            std::memcpy(const_cast<void*>(grid->getDataPtr()), data, totalSize * sizeof(double));
            delete[] data;
        }
        
        return grid;
    }

    // 创建随机目标点
    std::vector<TargetPoint> createRandomPoints(size_t count, bool is3D = false) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            point.coordinates.push_back(coord_dist_(gen_));
            point.coordinates.push_back(coord_dist_(gen_));
            if (is3D) {
                point.coordinates.push_back(std::uniform_real_distribution<double>(0.0, 3.0)(gen_));
            }
            points.push_back(point);
        }
        
        return points;
    }

    // 性能测试辅助函数
    template<typename InterpolatorType>
    double measurePerformance(
        const InterpolatorType& interpolator,
        const InterpolationRequest& request,
        int iterations = 10) {
        
        // 预热
        for (int i = 0; i < 3; ++i) {
            interpolator.execute(request);
        }
        
        // 实际测试
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            auto result = interpolator.execute(request);
            if (result.statusCode != 0) {
                std::cerr << "插值失败: " << result.message << std::endl;
                return -1.0;
            }
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

// 测试1D线性插值SIMD性能
TEST_F(Linear1DTrilinearSIMDPerformanceTest, Linear1D_SIMD_Performance) {
    std::cout << "\n=== 1D线性插值SIMD性能测试 ===" << std::endl;
    
    // 创建不同大小的测试网格
    std::vector<std::pair<size_t, size_t>> gridSizes = {
        {1000, 1000},
        {2000, 2000},
        {4000, 4000}
    };
    
    // 测试不同数量的插值点
    std::vector<size_t> pointCounts = {100, 1000, 10000, 100000};
    
    // 创建插值器（不使用SIMD管理器，让它使用内联SIMD）
    Linear1DInterpolator interpolator;
    
    for (const auto& [width, height] : gridSizes) {
        std::cout << "\n网格大小: " << width << "x" << height << std::endl;
        
        // 测试Float32
        {
            auto grid = createTestGrid(width, height, 1, DataType::Float32);
            std::cout << "\nFloat32数据类型:" << std::endl;
            std::cout << std::setw(15) << "点数" 
                     << std::setw(20) << "总时间(μs)" 
                     << std::setw(20) << "每点时间(ns)" << std::endl;
            std::cout << std::string(55, '-') << std::endl;
            
            for (size_t pointCount : pointCounts) {
                auto points = createRandomPoints(pointCount);
                
                InterpolationRequest request;
                request.sourceGrid = grid;
                request.target = points;
                request.method = InterpolationMethod::LINEAR_1D;
                
                double time = measurePerformance(interpolator, request);
                if (time > 0) {
                    double timePerPoint = (time * 1000.0) / pointCount;
                    std::cout << std::setw(15) << pointCount
                             << std::setw(20) << std::fixed << std::setprecision(2) << time
                             << std::setw(20) << std::fixed << std::setprecision(2) << timePerPoint
                             << std::endl;
                }
            }
        }
        
        // 测试Float64
        {
            auto grid = createTestGrid(width, height, 1, DataType::Float64);
            std::cout << "\nFloat64数据类型:" << std::endl;
            std::cout << std::setw(15) << "点数" 
                     << std::setw(20) << "总时间(μs)" 
                     << std::setw(20) << "每点时间(ns)" << std::endl;
            std::cout << std::string(55, '-') << std::endl;
            
            for (size_t pointCount : pointCounts) {
                auto points = createRandomPoints(pointCount);
                
                InterpolationRequest request;
                request.sourceGrid = grid;
                request.target = points;
                request.method = InterpolationMethod::LINEAR_1D;
                
                double time = measurePerformance(interpolator, request);
                if (time > 0) {
                    double timePerPoint = (time * 1000.0) / pointCount;
                    std::cout << std::setw(15) << pointCount
                             << std::setw(20) << std::fixed << std::setprecision(2) << time
                             << std::setw(20) << std::fixed << std::setprecision(2) << timePerPoint
                             << std::endl;
                }
            }
        }
    }
}

// 测试三线性插值SIMD性能
TEST_F(Linear1DTrilinearSIMDPerformanceTest, Trilinear_SIMD_Performance) {
    std::cout << "\n=== 三线性插值SIMD性能测试 ===" << std::endl;
    
    // 创建不同大小的测试网格
    std::vector<std::tuple<size_t, size_t, size_t>> gridSizes = {
        {100, 100, 10},   // 小型3D数据
        {200, 200, 20},   // 中型3D数据
        {500, 500, 5}     // 大型2D，少量波段
    };
    
    // 测试不同数量的插值点
    std::vector<size_t> pointCounts = {100, 1000, 10000};
    
    // 创建插值器
    TrilinearInterpolator interpolator;
    
    for (const auto& [width, height, bands] : gridSizes) {
        std::cout << "\n网格大小: " << width << "x" << height << "x" << bands << std::endl;
        
        // 测试Float32
        {
            auto grid = createTestGrid(width, height, bands, DataType::Float32);
            std::cout << "\nFloat32数据类型:" << std::endl;
            std::cout << std::setw(15) << "点数" 
                     << std::setw(20) << "总时间(μs)" 
                     << std::setw(20) << "每点时间(ns)" << std::endl;
            std::cout << std::string(55, '-') << std::endl;
            
            for (size_t pointCount : pointCounts) {
                auto points = createRandomPoints(pointCount, bands > 1);
                
                InterpolationRequest request;
                request.sourceGrid = grid;
                request.target = points;
                request.method = InterpolationMethod::TRILINEAR;
                
                double time = measurePerformance(interpolator, request);
                if (time > 0) {
                    double timePerPoint = (time * 1000.0) / pointCount;
                    std::cout << std::setw(15) << pointCount
                             << std::setw(20) << std::fixed << std::setprecision(2) << time
                             << std::setw(20) << std::fixed << std::setprecision(2) << timePerPoint
                             << std::endl;
                }
            }
        }
        
        // 测试2D情况（只有一个波段）
        if (bands == 1) {
            std::cout << "\n2D双线性插值（通过三线性插值器）:" << std::endl;
            auto grid = createTestGrid(width * 5, height * 5, 1, DataType::Float32);
            
            std::cout << std::setw(15) << "点数" 
                     << std::setw(20) << "总时间(μs)" 
                     << std::setw(20) << "每点时间(ns)" << std::endl;
            std::cout << std::string(55, '-') << std::endl;
            
            for (size_t pointCount : pointCounts) {
                auto points = createRandomPoints(pointCount, false);
                
                InterpolationRequest request;
                request.sourceGrid = grid;
                request.target = points;
                request.method = InterpolationMethod::TRILINEAR;
                
                double time = measurePerformance(interpolator, request);
                if (time > 0) {
                    double timePerPoint = (time * 1000.0) / pointCount;
                    std::cout << std::setw(15) << pointCount
                             << std::setw(20) << std::fixed << std::setprecision(2) << time
                             << std::setw(20) << std::fixed << std::setprecision(2) << timePerPoint
                             << std::endl;
                }
            }
        }
    }
}

// 对比测试：SIMD vs 非SIMD
TEST_F(Linear1DTrilinearSIMDPerformanceTest, SIMD_vs_NonSIMD_Comparison) {
    std::cout << "\n=== SIMD vs 非SIMD性能对比 ===" << std::endl;
    
    const size_t width = 1000;
    const size_t height = 1000;
    const size_t pointCount = 10000;
    
    // 1D线性插值对比
    {
        std::cout << "\n1D线性插值对比 (1000x1000网格, 10000点):" << std::endl;
        
        auto grid = createTestGrid(width, height, 1, DataType::Float32);
        auto points = createRandomPoints(pointCount);
        
        // 创建请求
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = points;
        request.method = InterpolationMethod::LINEAR_1D;
        
        // 测试小批量（强制使用标量路径）
        {
            auto smallPoints = std::vector<TargetPoint>(points.begin(), points.begin() + 4);
            request.target = smallPoints;
            
            Linear1DInterpolator interpolator;
            double smallBatchTime = measurePerformance(interpolator, request, 1000);
            double timePerPoint = (smallBatchTime * 1000.0) / smallPoints.size();
            
            std::cout << "标量路径（4点）: " << std::fixed << std::setprecision(2) 
                     << timePerPoint << " ns/点" << std::endl;
        }
        
        // 测试大批量（使用SIMD路径）
        {
            request.target = points;
            Linear1DInterpolator interpolator;
            double largeBatchTime = measurePerformance(interpolator, request);
            double timePerPoint = (largeBatchTime * 1000.0) / pointCount;
            
            std::cout << "SIMD路径（10000点）: " << std::fixed << std::setprecision(2) 
                     << timePerPoint << " ns/点" << std::endl;
        }
    }
    
    // 三线性插值对比
    {
        std::cout << "\n三线性插值对比 (200x200x10网格, 10000点):" << std::endl;
        
        auto grid = createTestGrid(200, 200, 10, DataType::Float32);
        auto points = createRandomPoints(pointCount, true);
        
        // 创建请求
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = points;
        request.method = InterpolationMethod::TRILINEAR;
        
        // 测试小批量
        {
            auto smallPoints = std::vector<TargetPoint>(points.begin(), points.begin() + 3);
            request.target = smallPoints;
            
            TrilinearInterpolator interpolator;
            double smallBatchTime = measurePerformance(interpolator, request, 1000);
            double timePerPoint = (smallBatchTime * 1000.0) / smallPoints.size();
            
            std::cout << "标量路径（3点）: " << std::fixed << std::setprecision(2) 
                     << timePerPoint << " ns/点" << std::endl;
        }
        
        // 测试大批量
        {
            request.target = points;
            TrilinearInterpolator interpolator;
            double largeBatchTime = measurePerformance(interpolator, request);
            double timePerPoint = (largeBatchTime * 1000.0) / pointCount;
            
            std::cout << "SIMD路径（10000点）: " << std::fixed << std::setprecision(2) 
                     << timePerPoint << " ns/点" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 