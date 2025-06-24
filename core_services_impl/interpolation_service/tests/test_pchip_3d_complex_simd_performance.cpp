#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <complex>
#include "../src/impl/algorithms/fast_pchip_interpolator_3d.h"
#include "../src/impl/algorithms/complex_field_interpolator.h"
#include "../src/impl/algorithms/pchip_interpolator.h"
#include "../src/impl/algorithms/recursive_ndim_pchip_interpolator.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services::interpolation;
using namespace oscean::core_services;

class PCHIP3DComplexSIMDPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_ = std::mt19937(42);
        dist_ = std::uniform_real_distribution<double>(0.0, 100.0);
    }

    // 创建3D测试网格（如声速剖面数据）
    std::shared_ptr<GridData> create3DTestGrid(size_t width, size_t height, size_t depth) {
        GridDefinition def;
        def.cols = width;
        def.rows = height;
        // 使用多个band来模拟深度维度
        
        auto grid = std::make_shared<GridData>(def, DataType::Float64, depth);
        
        // 设置地理变换参数
        std::vector<double> geoTransform = {
            0.0,    // 左上角X坐标（经度）
            1.0,    // X方向像素大小
            0.0,    // X方向旋转
            0.0,    // 左上角Y坐标（纬度）
            0.0,    // Y方向旋转
            1.0     // Y方向像素大小
        };
        grid->setGeoTransform(geoTransform);
        
        // 设置Z维度（深度）坐标
        grid->getDefinition().zDimension.coordinates.resize(depth);
        for (size_t z = 0; z < depth; ++z) {
            grid->getDefinition().zDimension.coordinates[z] = z * 10.0; // 每10米一层
        }
        
        // 填充声速剖面数据（典型的海洋声速剖面）
        double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    // 模拟声速随深度变化
                    double depth_m = z * 10.0;
                    double lat = y * 1.0;
                    double lon = x * 1.0;
                    // 典型声速剖面：表层快，中层慢，深层再快
                    double sound_speed = 1500.0 + 
                                       20.0 * std::exp(-depth_m / 100.0) - 
                                       30.0 * std::exp(-(depth_m - 300.0) * (depth_m - 300.0) / 10000.0) +
                                       0.1 * std::sin(lat * 0.1) * std::cos(lon * 0.1);
                    data[z * width * height + y * width + x] = sound_speed;
                }
            }
        }
        
        return grid;
    }

    // 创建复数场测试数据（如声压场）
    std::shared_ptr<GridData> createComplexFieldGrid(size_t width, size_t height) {
        GridDefinition def;
        def.cols = width;
        def.rows = height;
        
        // 复数需要2个band（实部和虚部）
        auto grid = std::make_shared<GridData>(def, DataType::Float64, 2);
        
        std::vector<double> geoTransform = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        grid->setGeoTransform(geoTransform);
        
        // 填充复数场数据
        double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                // 创建一个复杂的声场模式
                double r = std::sqrt(x * x + y * y);
                double theta = std::atan2(y, x);
                
                // 实部：衰减的柱面波
                data[y * width + x] = std::exp(-r * 0.01) * std::cos(r * 0.5 - theta);
                
                // 虚部
                data[width * height + y * width + x] = std::exp(-r * 0.01) * std::sin(r * 0.5 - theta);
            }
        }
        
        return grid;
    }

    // 创建随机3D目标点
    std::vector<TargetPoint> create3DRandomPoints(size_t count, double maxX, double maxY, double maxZ) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        std::uniform_real_distribution<double> distX(0.5, maxX - 0.5);
        std::uniform_real_distribution<double> distY(0.5, maxY - 0.5);
        std::uniform_real_distribution<double> distZ(0.5, maxZ - 0.5);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            point.coordinates = {distX(gen_), distY(gen_), distZ(gen_)};
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
};

// Fast PCHIP 3D性能测试
TEST_F(PCHIP3DComplexSIMDPerformanceTest, FastPCHIP3DPerformance) {
    std::cout << "\n=== Fast PCHIP 3D 性能测试（声速剖面插值）===" << std::endl;
    
    // 测试不同规模的3D网格
    std::vector<std::tuple<size_t, size_t, size_t>> gridSizes = {
        {50, 50, 20},    // 小规模：50x50x20（2万个点）
        {100, 100, 50},  // 中规模：100x100x50（50万个点）
        {200, 200, 100}  // 大规模：200x200x100（400万个点）
    };
    
    std::vector<size_t> pointCounts = {100, 1000, 10000};
    
    for (const auto& [width, height, depth] : gridSizes) {
        std::cout << "\n3D网格大小: " << width << "x" << height << "x" << depth 
                 << " (总计" << width * height * depth << "个点)" << std::endl;
        
        auto grid = create3DTestGrid(width, height, depth);
        
        // 创建插值器（预计算导数）
        auto setupTime = measureTime([&]() {
            auto interpolator = std::make_unique<FastPchipInterpolator3D>(grid);
        });
        
        std::cout << "  预计算时间: " << std::fixed << std::setprecision(2) 
                 << setupTime / 1000.0 << " ms" << std::endl;
        
        auto interpolator = std::make_unique<FastPchipInterpolator3D>(grid);
        
        for (size_t pointCount : pointCounts) {
            auto points = create3DRandomPoints(pointCount, width - 1, height - 1, depth - 1);
            
            InterpolationRequest request;
            request.sourceGrid = grid;
            request.target = points;
            request.method = interpolation::InterpolationMethod::PCHIP_FAST_3D;
            
            auto time = measureTime([&]() {
                interpolator->execute(request, nullptr);
            }, pointCount > 1000 ? 3 : 10);
            
            std::cout << "  插值 " << pointCount << " 个点: " 
                     << std::fixed << std::setprecision(2) 
                     << time / pointCount << " ns/点" << std::endl;
        }
    }
}

// 复数场插值性能测试
TEST_F(PCHIP3DComplexSIMDPerformanceTest, ComplexFieldInterpolationPerformance) {
    std::cout << "\n=== 复数场插值性能测试（声场计算）===" << std::endl;
    
    std::vector<size_t> gridSizes = {100, 500, 1000};
    std::vector<size_t> pointCounts = {1000, 10000, 100000};
    
    for (size_t gridSize : gridSizes) {
        std::cout << "\n复数场网格大小: " << gridSize << "x" << gridSize << std::endl;
        
        auto grid = createComplexFieldGrid(gridSize, gridSize);
        
        // 创建复数场插值器
        auto interpolator = std::make_unique<ComplexFieldInterpolator>();
        
        for (size_t pointCount : pointCounts) {
            auto points = create3DRandomPoints(pointCount, gridSize - 1, gridSize - 1, 1.0);
            // 只使用前两个坐标
            for (auto& point : points) {
                point.coordinates.resize(2);
            }
            
            InterpolationRequest request;
            request.sourceGrid = grid;
            request.target = points;
            request.method = interpolation::InterpolationMethod::BILINEAR; // 复数场通常使用双线性
            
            auto time = measureTime([&]() {
                interpolator->execute(request, nullptr);
            }, pointCount > 10000 ? 3 : 10);
            
            std::cout << "  插值 " << pointCount << " 个复数点: " 
                     << std::fixed << std::setprecision(2) 
                     << time / pointCount << " ns/点" << std::endl;
        }
    }
}

// 基础PCHIP性能测试（1D情况）
TEST_F(PCHIP3DComplexSIMDPerformanceTest, BasicPCHIPPerformance) {
    std::cout << "\n=== 基础PCHIP插值器性能测试 ===" << std::endl;
    
    // 创建2D网格用于测试
    std::vector<size_t> gridSizes = {100, 500, 1000};
    
    for (size_t gridSize : gridSizes) {
        std::cout << "\n网格大小: " << gridSize << "x" << gridSize << std::endl;
        
        GridDefinition def;
        def.cols = gridSize;
        def.rows = gridSize;
        
        auto grid = std::make_shared<GridData>(def, DataType::Float64, 1);
        std::vector<double> geoTransform = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        grid->setGeoTransform(geoTransform);
        
        // 填充测试数据
        double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t y = 0; y < gridSize; ++y) {
            for (size_t x = 0; x < gridSize; ++x) {
                data[y * gridSize + x] = std::sin(x * 0.1) * std::cos(y * 0.1) * 100.0;
            }
        }
        
        auto interpolator = std::make_unique<PCHIPInterpolator>();
        
        // 测试不同数量的点
        std::vector<size_t> pointCounts = {100, 1000, 10000};
        
        for (size_t pointCount : pointCounts) {
            auto points = create3DRandomPoints(pointCount, gridSize - 1, gridSize - 1, 1.0);
            for (auto& point : points) {
                point.coordinates.resize(2);
            }
            
            InterpolationRequest request;
            request.sourceGrid = grid;
            request.target = points;
            request.method = interpolation::InterpolationMethod::PCHIP_RECURSIVE_NDIM;
            
            auto time = measureTime([&]() {
                interpolator->execute(request, nullptr);
            }, pointCount > 1000 ? 5 : 10);
            
            std::cout << "  插值 " << pointCount << " 个点: " 
                     << std::fixed << std::setprecision(2) 
                     << time / pointCount << " ns/点" << std::endl;
        }
    }
}

// 递归N维PCHIP性能测试
TEST_F(PCHIP3DComplexSIMDPerformanceTest, RecursiveNDimPCHIPPerformance) {
    std::cout << "\n=== 递归N维PCHIP性能测试（USML兼容）===" << std::endl;
    
    // 测试2D和3D情况
    std::cout << "\n2D递归PCHIP测试:" << std::endl;
    {
        size_t gridSize = 100;
        GridDefinition def;
        def.cols = gridSize;
        def.rows = gridSize;
        
        auto grid = std::make_shared<GridData>(def, DataType::Float64, 1);
        std::vector<double> geoTransform = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        grid->setGeoTransform(geoTransform);
        
        // 填充数据
        double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t i = 0; i < gridSize * gridSize; ++i) {
            data[i] = dist_(gen_);
        }
        
        auto interpolator = std::make_unique<RecursiveNDimPCHIPInterpolator>();
        
        auto points = create3DRandomPoints(1000, gridSize - 1, gridSize - 1, 1.0);
        for (auto& point : points) {
            point.coordinates.resize(2);
        }
        
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = points;
        
        auto time = measureTime([&]() {
            interpolator->execute(request, nullptr);
        }, 10);
        
        std::cout << "  2D递归PCHIP (1000点): " 
                 << std::fixed << std::setprecision(2) 
                 << time / 1000.0 << " ns/点" << std::endl;
    }
    
    std::cout << "\n3D递归PCHIP测试:" << std::endl;
    {
        auto grid = create3DTestGrid(50, 50, 20);
        auto interpolator = std::make_unique<RecursiveNDimPCHIPInterpolator>();
        
        auto points = create3DRandomPoints(1000, 49, 49, 19);
        
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = points;
        
        auto time = measureTime([&]() {
            interpolator->execute(request, nullptr);
        }, 5);
        
        std::cout << "  3D递归PCHIP (1000点): " 
                 << std::fixed << std::setprecision(2) 
                 << time / 1000.0 << " ns/点" << std::endl;
    }
} 