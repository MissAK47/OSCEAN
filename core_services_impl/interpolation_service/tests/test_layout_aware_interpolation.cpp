#include <gtest/gtest.h>
#include "../src/impl/algorithms/pchip_interpolator.h"
#include "../src/impl/algorithms/bilinear_interpolator.h"
#include "interpolation/layout_converter.h"
#include <common_utils/simd/isimd_manager.h>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>

using namespace oscean::core_services;
using namespace oscean::core_services::interpolation;

class LayoutAwareInterpolationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试网格数据（行主序）
        GridDefinition def;
        def.rows = 10;
        def.cols = 10;
        def.xDimension.coordinates = std::vector<double>(10);
        def.yDimension.coordinates = std::vector<double>(10);
        
        // 初始化坐标
        for (int i = 0; i < 10; ++i) {
            def.xDimension.coordinates[i] = i * 1.0;
            def.yDimension.coordinates[i] = i * 1.0;
        }
        
        rowMajorGrid_ = std::make_shared<GridData>(def, DataType::Float64, 1);
        
        // 填充测试数据：z = x^2 + y^2
        auto* data = static_cast<double*>(const_cast<void*>(rowMajorGrid_->getDataPtr()));
        for (int row = 0; row < 10; ++row) {
            for (int col = 0; col < 10; ++col) {
                data[row * 10 + col] = col * col + row * row;
            }
        }
        
        // 设置地理变换
        std::vector<double> geoTransform = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        rowMajorGrid_->setGeoTransform(geoTransform);
        
        // 创建列主序版本（模拟）
        columnMajorGrid_ = std::make_shared<GridData>(def, DataType::Float64, 1);
        auto* colData = static_cast<double*>(const_cast<void*>(columnMajorGrid_->getDataPtr()));
        
        // 转置数据以模拟列主序
        for (int row = 0; row < 10; ++row) {
            for (int col = 0; col < 10; ++col) {
                colData[col * 10 + row] = col * col + row * row;
            }
        }
        columnMajorGrid_->setGeoTransform(geoTransform);
    }
    
    std::shared_ptr<GridData> rowMajorGrid_;
    std::shared_ptr<GridData> columnMajorGrid_;
};

TEST_F(LayoutAwareInterpolationTest, TestBilinearInterpolation) {
    // 创建双线性插值器
    BilinearInterpolator interpolator;
    
    // 测试单点插值
    std::vector<TargetPoint> points;
    TargetPoint pt;
    pt.coordinates = {2.5, 3.5};  // 应该得到约 2.5^2 + 3.5^2 = 18.5
    points.push_back(pt);
    
    // 执行插值
    auto result = interpolator.interpolateAtPoints(*rowMajorGrid_, points);
    
    EXPECT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].has_value());
    
    if (result[0].has_value()) {
        // 双线性插值应该给出准确的结果（对于二次函数）
        double expected = 2.5 * 2.5 + 3.5 * 3.5;  // 18.5
        EXPECT_NEAR(result[0].value(), expected, 0.1);
    }
}

TEST_F(LayoutAwareInterpolationTest, TestLayoutConverter) {
    // 跳过此测试，因为LayoutConverter是抽象类
    GTEST_SKIP() << "LayoutConverter is abstract class";
}

TEST_F(LayoutAwareInterpolationTest, TestLayoutAdapterView) {
    // 创建布局适配视图
    LayoutAdapterView<double> rowMajorView(
        static_cast<const double*>(rowMajorGrid_->getDataPtr()),
        10, 10, MemoryLayout::ROW_MAJOR
    );
    
    LayoutAdapterView<double> colMajorView(
        static_cast<const double*>(columnMajorGrid_->getDataPtr()),
        10, 10, MemoryLayout::COLUMN_MAJOR
    );
    
    // 测试访问相同逻辑位置的值
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 10; ++y) {
            double rowValue = rowMajorView.getValue(y, x);  // 注意：getValue(row, col)
            double colValue = colMajorView.getValue(y, x);
            
            EXPECT_DOUBLE_EQ(rowValue, colValue);
            EXPECT_DOUBLE_EQ(rowValue, x * x + y * y);
        }
    }
}

TEST_F(LayoutAwareInterpolationTest, TestBatchInterpolation) {
    BilinearInterpolator interpolator;
    
    // 创建一批测试点
    std::vector<TargetPoint> points;
    for (double x = 0.5; x < 9.0; x += 1.0) {
        for (double y = 0.5; y < 9.0; y += 1.0) {
            TargetPoint pt;
            pt.coordinates = {x, y};
            points.push_back(pt);
        }
    }
    
    // 执行批量插值
    auto results = interpolator.interpolateAtPoints(*rowMajorGrid_, points);
    
    EXPECT_EQ(results.size(), points.size());
    
    // 验证结果
    for (size_t i = 0; i < points.size(); ++i) {
        EXPECT_TRUE(results[i].has_value());
        if (results[i].has_value()) {
            double x = points[i].coordinates[0];
            double y = points[i].coordinates[1];
            double expected = x * x + y * y;
            EXPECT_NEAR(results[i].value(), expected, 0.1);
        }
    }
}

TEST_F(LayoutAwareInterpolationTest, TestSIMDOptimization) {
    BilinearInterpolator interpolator;
    
    // 创建大批量测试点以触发SIMD优化
    std::vector<TargetPoint> points;
    for (int i = 0; i < 1000; ++i) {
        TargetPoint pt;
        pt.coordinates = {
            0.5 + (i % 9),
            0.5 + ((i / 9) % 9)
        };
        points.push_back(pt);
    }
    
    // 执行插值并测量时间
    auto start = std::chrono::high_resolution_clock::now();
    auto results = interpolator.interpolateAtPoints(*rowMajorGrid_, points);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    EXPECT_EQ(results.size(), points.size());
    
    // 输出性能信息
    std::cout << "Batch interpolation of " << points.size() 
              << " points took " << duration.count() << " microseconds" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 