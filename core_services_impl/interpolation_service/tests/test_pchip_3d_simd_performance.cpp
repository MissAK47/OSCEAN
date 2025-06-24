#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include "../src/impl/algorithms/fast_pchip_interpolator_3d.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services::interpolation;
using namespace oscean::core_services;

class FastPCHIP3DSIMDPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_ = std::mt19937(42);
        dist_ = std::uniform_real_distribution<double>(0.0, 100.0);
        coord_dist_ = std::uniform_real_distribution<double>(10.0, 90.0);
    }

    // 创建3D测试网格（模拟声速剖面数据）
    std::shared_ptr<GridData> create3DTestGrid(size_t width, size_t height, size_t depth) {
        GridDefinition def;
        def.cols = width;
        def.rows = height;
        
        // 设置Z维度坐标（深度）
        def.zDimension.coordinates.clear();
        for (size_t z = 0; z < depth; ++z) {
            def.zDimension.coordinates.push_back(z * 10.0); // 深度间隔10米
        }
        
        auto grid = std::make_shared<GridData>(def, DataType::Float64, depth);
        
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
        
        // 填充数据（模拟声速剖面）
        double* data = static_cast<double*>(const_cast<void*>(grid->getDataPtr()));
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    // 声速随深度变化的简单模型
                    double soundSpeed = 1500.0 + 0.05 * z * 10.0 - 0.001 * x * y;
                    data[z * width * height + y * width + x] = soundSpeed;
                }
            }
        }
        
        return grid;
    }

    // 生成随机测试点
    std::vector<TargetPoint> generateRandomPoints(size_t count, size_t maxX, size_t maxY, size_t maxZ) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            point.coordinates = {
                coord_dist_(gen_) * maxX / 100.0,
                coord_dist_(gen_) * maxY / 100.0,
                coord_dist_(gen_) * maxZ * 10.0 / 100.0  // Z是深度，单位不同
            };
            points.push_back(point);
        }
        
        return points;
    }

    // 测量执行时间
    template<typename Func>
    double measureTime(Func&& func, int iterations = 10) {
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

// 性能对比测试
TEST_F(FastPCHIP3DSIMDPerformanceTest, PerformanceComparison) {
    std::cout << "\n=== Fast PCHIP 3D SIMD性能测试 ===" << std::endl;
    
    // 测试不同大小的3D网格
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> testCases = {
        {50, 50, 20, 1000},     // 小网格，1000个点
        {100, 100, 50, 10000},  // 中等网格，10000个点
        {200, 200, 100, 50000}  // 大网格，50000个点
    };
    
    for (const auto& [width, height, depth, pointCount] : testCases) {
        std::cout << "\n网格大小: " << width << "x" << height << "x" << depth 
                  << ", 插值点数: " << pointCount << std::endl;
        
        // 创建测试数据
        auto grid = create3DTestGrid(width, height, depth);
        auto points = generateRandomPoints(pointCount, width, height, depth);
        
        // 创建插值器
        auto interpolator = std::make_shared<FastPchipInterpolator3D>(grid);
        
        // 准备插值请求
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = points;
        request.method = interpolation::InterpolationMethod::PCHIP_FAST_3D;
        
        // 测试标量路径（少量点）
        if (pointCount <= 10) {
            auto scalarTime = measureTime([&]() {
                auto smallPoints = std::vector<TargetPoint>(points.begin(), points.begin() + 4);
                InterpolationRequest smallRequest = request;
                smallRequest.target = smallPoints;
                interpolator->execute(smallRequest, nullptr);
            }, 100);
            
            std::cout << "  标量路径 (4 点): " 
                     << std::fixed << std::setprecision(2) 
                     << scalarTime / 4 << " ns/点" << std::endl;
        }
        
        // 测试SIMD路径
        auto simdTime = measureTime([&]() {
            interpolator->execute(request, nullptr);
        }, pointCount > 10000 ? 5 : 10);
        
        std::cout << "  SIMD路径 (" << pointCount << " 点): " 
                 << std::fixed << std::setprecision(2) 
                 << simdTime * 1000.0 / pointCount << " ns/点" << std::endl;
        
        // 计算总时间和吞吐量
        std::cout << "  总时间: " << simdTime / 1000.0 << " ms" << std::endl;
        std::cout << "  吞吐量: " << std::fixed << std::setprecision(2) 
                 << pointCount / (simdTime / 1e6) / 1e6 << " M点/秒" << std::endl;
        
        // 预计算时间测试
        auto precomputeTime = measureTime([&]() {
            auto newInterpolator = std::make_shared<FastPchipInterpolator3D>(grid);
        }, 3);
        
        std::cout << "  预计算时间: " << precomputeTime / 1000.0 << " ms" << std::endl;
        std::cout << "  每点预计算时间: " << precomputeTime / (width * height * depth) 
                 << " μs/点" << std::endl;
    }
}

// 精度验证测试
TEST_F(FastPCHIP3DSIMDPerformanceTest, AccuracyValidation) {
    std::cout << "\n=== Fast PCHIP 3D精度验证测试 ===" << std::endl;
    
    // 创建一个小的测试网格
    size_t width = 10, height = 10, depth = 10;
    auto grid = create3DTestGrid(width, height, depth);
    
    // 创建插值器
    auto interpolator = std::make_shared<FastPchipInterpolator3D>(grid);
    
    // 测试已知点的插值精度
    std::vector<TargetPoint> testPoints;
    std::vector<double> expectedValues;
    
    // 网格点上的精确值
    for (int i = 1; i < 9; ++i) {
        for (int j = 1; j < 9; ++j) {
            for (int k = 1; k < 9; ++k) {
                TargetPoint point;
                point.coordinates = {
                    static_cast<double>(i),
                    static_cast<double>(j),
                    static_cast<double>(k * 10)  // Z坐标是深度
                };
                testPoints.push_back(point);
                
                // 期望值（从create3DTestGrid的公式计算）
                double expected = 1500.0 + 0.05 * k * 10.0 - 0.001 * i * j;
                expectedValues.push_back(expected);
            }
        }
    }
    
    // 网格点之间的插值
    for (int i = 0; i < 5; ++i) {
        TargetPoint point;
        point.coordinates = {
            1.5 + i * 1.7,
            2.3 + i * 1.4,
            15.0 + i * 17.0  // 深度
        };
        testPoints.push_back(point);
        
        // 对于非网格点，我们只检查结果是否合理
        expectedValues.push_back(-1.0); // 标记为需要范围检查
    }
    
    // 执行插值
    InterpolationRequest request;
    request.sourceGrid = grid;
    request.target = testPoints;
    request.method = interpolation::InterpolationMethod::PCHIP_FAST_3D;
    
    auto result = interpolator->execute(request, nullptr);
    auto values = std::get<std::vector<std::optional<double>>>(result.data);
    
    // 验证结果
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    int exactCount = 0;
    
    for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_TRUE(values[i].has_value()) << "点 " << i << " 插值失败";
        
        double actual = values[i].value();
        
        if (expectedValues[i] > 0) {
            // 网格点，应该精确
            double absError = std::abs(actual - expectedValues[i]);
            double relError = absError / std::abs(expectedValues[i]);
            
            if (absError < 1e-10) {
                exactCount++;
            }
            
            maxAbsError = std::max(maxAbsError, absError);
            maxRelError = std::max(maxRelError, relError);
            
            if (i < 10) {  // 只打印前几个
                std::cout << "  点 " << i << ": 期望=" << expectedValues[i] 
                         << ", 实际=" << actual
                         << ", 误差=" << absError << std::endl;
            }
        } else {
            // 非网格点，检查范围
            EXPECT_GE(actual, 1490.0) << "声速值过低";
            EXPECT_LE(actual, 1550.0) << "声速值过高";
        }
    }
    
    std::cout << "\n精度统计:" << std::endl;
    std::cout << "  精确匹配的网格点: " << exactCount << "/" << 512 << std::endl;
    std::cout << "  最大绝对误差: " << maxAbsError << std::endl;
    std::cout << "  最大相对误差: " << maxRelError * 100 << "%" << std::endl;
    
    // 验证精度
    EXPECT_LT(maxAbsError, 1e-6) << "绝对误差过大";
    EXPECT_LT(maxRelError, 1e-6) << "相对误差过大";
}

// 声速剖面专用测试
TEST_F(FastPCHIP3DSIMDPerformanceTest, SoundVelocityProfileTest) {
    std::cout << "\n=== 声速剖面插值测试 ===" << std::endl;
    
    // 创建典型的声速剖面网格（水平分辨率低，垂直分辨率高）
    size_t numLat = 20;   // 纬度方向
    size_t numLon = 30;   // 经度方向
    size_t numDepth = 100; // 深度方向（高分辨率）
    
    auto grid = create3DTestGrid(numLon, numLat, numDepth);
    auto interpolator = std::make_shared<FastPchipInterpolator3D>(grid);
    
    // 生成垂直剖面查询（固定水平位置，不同深度）
    std::vector<TargetPoint> profilePoints;
    for (double depth = 0; depth < 990; depth += 5.0) {
        TargetPoint point;
        point.coordinates = {15.5, 10.5, depth}; // 固定水平位置
        profilePoints.push_back(point);
    }
    
    std::cout << "  垂直剖面查询点数: " << profilePoints.size() << std::endl;
    
    // 测试性能
    InterpolationRequest request;
    request.sourceGrid = grid;
    request.target = profilePoints;
    
    auto profileTime = measureTime([&]() {
        interpolator->execute(request, nullptr);
    }, 20);
    
    std::cout << "  垂直剖面插值时间: " << profileTime / 1000.0 << " ms" << std::endl;
    std::cout << "  每点时间: " << profileTime * 1000.0 / profilePoints.size() << " ns/点" << std::endl;
    
    // 测试水平切片查询（固定深度，不同水平位置）
    std::vector<TargetPoint> slicePoints;
    for (double lat = 1; lat < 19; lat += 0.5) {
        for (double lon = 1; lon < 29; lon += 0.5) {
            TargetPoint point;
            point.coordinates = {lon, lat, 500.0}; // 固定深度500米
            slicePoints.push_back(point);
        }
    }
    
    std::cout << "\n  水平切片查询点数: " << slicePoints.size() << std::endl;
    
    request.target = slicePoints;
    auto sliceTime = measureTime([&]() {
        interpolator->execute(request, nullptr);
    }, 10);
    
    std::cout << "  水平切片插值时间: " << sliceTime / 1000.0 << " ms" << std::endl;
    std::cout << "  每点时间: " << sliceTime * 1000.0 / slicePoints.size() << " ns/点" << std::endl;
}

// 边界条件测试
TEST_F(FastPCHIP3DSIMDPerformanceTest, BoundaryConditionTest) {
    std::cout << "\n=== 边界条件测试 ===" << std::endl;
    
    auto grid = create3DTestGrid(10, 10, 10);
    auto interpolator = std::make_shared<FastPchipInterpolator3D>(grid);
    
    // 测试边界附近的点
    std::vector<TargetPoint> boundaryPoints = {
        {{0.1, 0.1, 0.1}},      // 接近原点
        {{9.9, 9.9, 90.0}},     // 接近最大值（注意：Z坐标是深度，最大90米）
        {{-0.1, 5.0, 50.0}},    // X越界
        {{5.0, -0.1, 50.0}},    // Y越界
        {{5.0, 5.0, -1.0}},     // Z越界
        {{10.1, 5.0, 50.0}},    // X越界
        {{5.0, 10.1, 50.0}},    // Y越界
        {{5.0, 5.0, 101.0}}     // Z越界
    };
    
    InterpolationRequest request;
    request.sourceGrid = grid;
    request.target = boundaryPoints;
    
    auto result = interpolator->execute(request, nullptr);
    auto values = std::get<std::vector<std::optional<double>>>(result.data);
    
    // 验证边界处理
    EXPECT_TRUE(values[0].has_value()) << "边界内的点应该有值";
    EXPECT_TRUE(values[1].has_value()) << "边界内的点应该有值";
    EXPECT_FALSE(values[2].has_value()) << "X越界的点应该返回nullopt";
    EXPECT_FALSE(values[3].has_value()) << "Y越界的点应该返回nullopt";
    EXPECT_FALSE(values[4].has_value()) << "Z越界的点应该返回nullopt";
    EXPECT_FALSE(values[5].has_value()) << "X越界的点应该返回nullopt";
    EXPECT_FALSE(values[6].has_value()) << "Y越界的点应该返回nullopt";
    EXPECT_FALSE(values[7].has_value()) << "Z越界的点应该返回nullopt";
    
    std::cout << "  边界条件处理正确" << std::endl;
} 