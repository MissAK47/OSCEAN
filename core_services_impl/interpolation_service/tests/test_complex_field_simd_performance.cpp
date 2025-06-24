#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <complex>
#include <iostream>
#include <iomanip>
#include "../src/impl/algorithms/complex_field_interpolator.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services::interpolation;
using namespace oscean::core_services;

class ComplexFieldSIMDPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_ = std::mt19937(42);
        dist_ = std::uniform_real_distribution<double>(-100.0, 100.0);
        coord_dist_ = std::uniform_real_distribution<double>(10.0, 90.0);
    }

    // 创建测试用的复数场网格（模拟RAM声场数据）
    std::pair<std::shared_ptr<GridData>, std::shared_ptr<GridData>> 
    createComplexFieldGrid(size_t numRanges, size_t numDepths) {
        GridDefinition def;
        def.cols = numRanges;
        def.rows = numDepths;
        
        // 设置距离维度（X）
        def.xDimension.coordinates.clear();
        for (size_t r = 0; r < numRanges; ++r) {
            def.xDimension.coordinates.push_back(r * 100.0); // 距离间隔100米
        }
        def.xDimension.minValue = 0.0;
        def.xDimension.maxValue = (numRanges - 1) * 100.0;
        def.xDimension.name = "range";
        def.xDimension.units = "m";
        
        // 设置深度维度（Y）
        def.yDimension.coordinates.clear();
        for (size_t d = 0; d < numDepths; ++d) {
            def.yDimension.coordinates.push_back(d * 5.0); // 深度间隔5米
        }
        def.yDimension.minValue = 0.0;
        def.yDimension.maxValue = (numDepths - 1) * 5.0;
        def.yDimension.name = "depth";
        def.yDimension.units = "m";
        
        // 创建实部和虚部网格
        auto realGrid = std::make_shared<GridData>(def, DataType::Float64, 1);
        auto imagGrid = std::make_shared<GridData>(def, DataType::Float64, 1);
        
        // 设置地理变换参数
        std::vector<double> geoTransform = {
            0.0, 100.0, 0.0, 0.0, 0.0, 5.0
        };
        realGrid->setGeoTransform(geoTransform);
        imagGrid->setGeoTransform(geoTransform);
        
        // 填充复数场数据（模拟声压场）
        double* realData = static_cast<double*>(const_cast<void*>(realGrid->getDataPtr()));
        double* imagData = static_cast<double*>(const_cast<void*>(imagGrid->getDataPtr()));
        
        for (size_t d = 0; d < numDepths; ++d) {
            for (size_t r = 0; r < numRanges; ++r) {
                size_t idx = d * numRanges + r;
                // 简单的声场模型：幅度随距离衰减，相位随距离变化
                double amplitude = 100.0 * std::exp(-0.001 * r * 100.0);
                double phase = 0.01 * r * 100.0;
                realData[idx] = amplitude * std::cos(phase);
                imagData[idx] = amplitude * std::sin(phase);
            }
        }
        
        return {realGrid, imagGrid};
    }

    // 生成随机测试点
    std::vector<TargetPoint> generateRandomPoints(size_t count, double maxRange, double maxDepth) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            point.coordinates = {
                coord_dist_(gen_) * maxRange / 100.0,
                coord_dist_(gen_) * maxDepth / 100.0
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
TEST_F(ComplexFieldSIMDPerformanceTest, PerformanceComparison) {
    std::cout << "\n=== 复数场插值SIMD性能测试 ===" << std::endl;
    
    // 测试不同大小的声场网格
    std::vector<std::tuple<size_t, size_t, size_t>> testCases = {
        {100, 50, 1000},     // 小网格，1000个点
        {500, 100, 10000},   // 中等网格，10000个点
        {1000, 200, 50000}   // 大网格，50000个点
    };
    
    for (const auto& [numRanges, numDepths, pointCount] : testCases) {
        std::cout << "\n声场网格大小: " << numRanges << "x" << numDepths 
                  << " (距离x深度), 插值点数: " << pointCount << std::endl;
        
        // 创建测试数据
        auto [realGrid, imagGrid] = createComplexFieldGrid(numRanges, numDepths);
        auto points = generateRandomPoints(pointCount, (numRanges-1)*100.0, (numDepths-1)*5.0);
        
        // 创建插值器
        auto interpolator = std::make_shared<ComplexFieldInterpolator>();
        
        // 测试标量路径（少量点）
        if (pointCount <= 1000) {
            auto scalarTime = measureTime([&]() {
                auto smallPoints = std::vector<TargetPoint>(points.begin(), points.begin() + 4);
                // 禁用SIMD，强制使用标量路径
                for (const auto& pt : smallPoints) {
                    interpolator->interpolateComplex(*realGrid, *imagGrid, 
                                                   pt.coordinates[0], pt.coordinates[1]);
                }
            }, 100);
            
            std::cout << "  标量路径 (4 点): " 
                     << std::fixed << std::setprecision(2) 
                     << scalarTime / 4 << " ns/点" << std::endl;
        }
        
        // 测试SIMD路径（批量）
        auto simdTime = measureTime([&]() {
            interpolator->interpolateComplexBatch(*realGrid, *imagGrid, points);
        }, pointCount > 10000 ? 5 : 10);
        
        std::cout << "  SIMD路径 (" << pointCount << " 点): " 
                 << std::fixed << std::setprecision(2) 
                 << simdTime * 1000.0 / pointCount << " ns/点" << std::endl;
        
        // 计算总时间和吞吐量
        std::cout << "  总时间: " << simdTime / 1000.0 << " ms" << std::endl;
        std::cout << "  吞吐量: " << std::fixed << std::setprecision(2) 
                 << pointCount / (simdTime / 1e6) / 1e6 << " M点/秒" << std::endl;
        
        // 性能提升比
        if (pointCount <= 1000) {
            auto singleTime = measureTime([&]() {
                for (const auto& pt : points) {
                    interpolator->interpolateComplex(*realGrid, *imagGrid, 
                                                   pt.coordinates[0], pt.coordinates[1]);
                }
            }, 10);
            
            double speedup = singleTime / simdTime;
            std::cout << "  性能提升: " << std::fixed << std::setprecision(1) 
                     << speedup << "倍" << std::endl;
        }
    }
}

// 精度验证测试
TEST_F(ComplexFieldSIMDPerformanceTest, AccuracyValidation) {
    std::cout << "\n=== 复数场插值精度验证 ===" << std::endl;
    
    // 创建一个小的测试网格
    auto [realGrid, imagGrid] = createComplexFieldGrid(10, 10);
    auto interpolator = std::make_shared<ComplexFieldInterpolator>();
    
    // 测试网格点上的插值精度
    std::vector<TargetPoint> testPoints;
    std::vector<std::complex<double>> expectedValues;
    
    // 网格点上的精确值
    for (int r = 1; r < 9; ++r) {
        for (int d = 1; d < 9; ++d) {
            TargetPoint point;
            point.coordinates = {r * 100.0, d * 5.0};
            testPoints.push_back(point);
            
            // 期望值
            double amplitude = 100.0 * std::exp(-0.001 * r * 100.0);
            double phase = 0.01 * r * 100.0;
            expectedValues.emplace_back(amplitude * std::cos(phase), 
                                      amplitude * std::sin(phase));
        }
    }
    
    // 执行批量插值
    auto results = interpolator->interpolateComplexBatch(*realGrid, *imagGrid, testPoints);
    
    // 验证结果
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    int exactCount = 0;
    
    for (size_t i = 0; i < results.size(); ++i) {
        auto expected = expectedValues[i];
        auto actual = results[i];
        
        double absError = std::abs(actual - expected);
        double relError = absError / std::abs(expected);
        
        if (absError < 1e-10) {
            exactCount++;
        }
        
        maxAbsError = std::max(maxAbsError, absError);
        maxRelError = std::max(maxRelError, relError);
        
        if (i < 5) {  // 只打印前几个
            std::cout << "  点 " << i << ": 期望=" << expected 
                     << ", 实际=" << actual
                     << ", 误差=" << absError << std::endl;
        }
    }
    
    std::cout << "\n精度统计:" << std::endl;
    std::cout << "  精确匹配的网格点: " << exactCount << "/" << results.size() << std::endl;
    std::cout << "  最大绝对误差: " << maxAbsError << std::endl;
    std::cout << "  最大相对误差: " << maxRelError * 100 << "%" << std::endl;
    
    // 验证精度
    EXPECT_LT(maxAbsError, 1e-10) << "绝对误差过大";
    EXPECT_LT(maxRelError, 1e-10) << "相对误差过大";
}

// RAM声场适配器测试
TEST_F(ComplexFieldSIMDPerformanceTest, RAMFieldAdapterTest) {
    std::cout << "\n=== RAM声场适配器测试 ===" << std::endl;
    
    // 创建模拟的RAM声场数据
    std::vector<double> ranges, depths;
    for (int r = 0; r < 100; ++r) ranges.push_back(r * 50.0);
    for (int d = 0; d < 50; ++d) depths.push_back(d * 2.0);
    
    std::vector<std::complex<double>> pressureField;
    for (size_t d = 0; d < depths.size(); ++d) {
        for (size_t r = 0; r < ranges.size(); ++r) {
            double amplitude = 100.0 * std::exp(-0.001 * ranges[r]);
            double phase = 0.01 * ranges[r] - 0.005 * depths[d];
            pressureField.emplace_back(amplitude * std::cos(phase), 
                                     amplitude * std::sin(phase));
        }
    }
    
    // 测试适配器
    auto [realGrid, imagGrid] = RAMFieldAdapter::createFromRAMField(
        pressureField, ranges, depths);
    
    EXPECT_EQ(realGrid->getDefinition().cols, ranges.size());
    EXPECT_EQ(realGrid->getDefinition().rows, depths.size());
    EXPECT_EQ(imagGrid->getDefinition().cols, ranges.size());
    EXPECT_EQ(imagGrid->getDefinition().rows, depths.size());
    
    // 测试交错存储格式
    auto interleavedGrid = RAMFieldAdapter::createInterleavedComplexGrid(
        pressureField, ranges, depths);
    
    EXPECT_EQ(interleavedGrid->getDefinition().cols, ranges.size() * 2);
    EXPECT_EQ(interleavedGrid->getDefinition().rows, depths.size());
    
    std::cout << "  RAM声场适配器测试通过" << std::endl;
    std::cout << "  分离格式网格大小: " << realGrid->getDefinition().cols 
              << "x" << realGrid->getDefinition().rows << std::endl;
    std::cout << "  交错格式网格大小: " << interleavedGrid->getDefinition().cols 
              << "x" << interleavedGrid->getDefinition().rows << std::endl;
}

// 边界条件测试
TEST_F(ComplexFieldSIMDPerformanceTest, BoundaryConditionTest) {
    std::cout << "\n=== 边界条件测试 ===" << std::endl;
    
    auto [realGrid, imagGrid] = createComplexFieldGrid(10, 10);
    auto interpolator = std::make_shared<ComplexFieldInterpolator>();
    
    // 测试边界附近的点
    std::vector<TargetPoint> boundaryPoints = {
        {{50.0, 2.5}},      // 边界内
        {{850.0, 42.5}},    // 边界内
        {{-50.0, 25.0}},    // X越界
        {{500.0, -5.0}},    // Y越界
        {{1000.0, 25.0}},   // X越界
        {{500.0, 50.0}}     // Y越界
    };
    
    auto results = interpolator->interpolateComplexBatch(*realGrid, *imagGrid, boundaryPoints);
    
    // 验证边界处理
    EXPECT_NE(std::abs(results[0]), 0.0) << "边界内的点应该有值";
    EXPECT_NE(std::abs(results[1]), 0.0) << "边界内的点应该有值";
    EXPECT_EQ(results[2], std::complex<double>(0, 0)) << "越界点应该返回0";
    EXPECT_EQ(results[3], std::complex<double>(0, 0)) << "越界点应该返回0";
    EXPECT_EQ(results[4], std::complex<double>(0, 0)) << "越界点应该返回0";
    EXPECT_EQ(results[5], std::complex<double>(0, 0)) << "越界点应该返回0";
    
    std::cout << "  边界条件处理正确" << std::endl;
} 