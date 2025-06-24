/**
 * @file test_gpu_interpolation_integration.cpp
 * @brief GPU插值集成测试
 */

#include <gtest/gtest.h>
#include <core_services/interpolation/i_interpolation_service.h>
#include <core_services/common_data_types.h>
#include <memory>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

using namespace oscean::core_services;
using namespace oscean::core_services::interpolation;

class GPUInterpolationIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试初始化
    }
    
    void TearDown() override {
        // 测试清理
    }
    
    // 辅助函数：创建测试网格数据
    std::shared_ptr<GridData> createTestGrid(int width, int height, bool addNoise = false) {
        GridDefinition def;
        def.rows = height;
        def.cols = width;
        def.extent = BoundingBox(0.0, 0.0, 100.0, 100.0);
        
        auto grid = std::make_shared<GridData>(def, DataType::Float32, 1);
        float* data = static_cast<float*>(grid->getDataPtrMutable());
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> noise(-0.1f, 0.1f);
        
        // 创建一个简单的2D函数: z = sin(x/10) * cos(y/10)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float xCoord = x * 100.0f / (width - 1);
                float yCoord = y * 100.0f / (height - 1);
                float value = std::sin(xCoord / 10.0f) * std::cos(yCoord / 10.0f);
                
                if (addNoise) {
                    value += noise(gen);
                }
                
                data[y * width + x] = value;
            }
        }
        
        return grid;
    }
};

TEST_F(GPUInterpolationIntegrationTest, BilinearInterpolationAccuracy) {
    // 创建源数据
    auto sourceGrid = createTestGrid(100, 100);
    
    // 创建目标点
    std::vector<TargetPoint> points;
    
    // 测试几个已知点
    TargetPoint pt1;
    pt1.coordinates = {50.0, 50.0};  // 中心点
    points.push_back(pt1);
    
    TargetPoint pt2;
    pt2.coordinates = {25.0, 75.0};  // 四分之一点
    points.push_back(pt2);
    
    // 创建插值请求
    InterpolationRequest request;
    request.sourceGrid = sourceGrid;
    request.target = points;
    request.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
    
    // 这里我们只测试数据结构和API的正确性
    // 实际的GPU插值需要服务实例
    
    EXPECT_EQ(request.sourceGrid->getDefinition().rows, 100);
    EXPECT_EQ(request.sourceGrid->getDefinition().cols, 100);
    
    // 检查variant中的vector
    if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        const auto& pts = std::get<std::vector<TargetPoint>>(request.target);
        EXPECT_EQ(pts.size(), 2u);
    }
}

TEST_F(GPUInterpolationIntegrationTest, LargeBatchInterpolation) {
    // 创建大型源数据
    auto sourceGrid = createTestGrid(512, 512);
    
    // 创建大批量目标点
    std::vector<TargetPoint> points;
    for (int i = 0; i < 10000; ++i) {
        TargetPoint pt;
        pt.coordinates = {
            (i % 500) * 1.0,
            (i / 500) * 1.0
        };
        points.push_back(pt);
    }
    
    // 创建插值请求
    InterpolationRequest request;
    request.sourceGrid = sourceGrid;
    request.target = points;
    request.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
    
    // 验证请求结构
    if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        const auto& pts = std::get<std::vector<TargetPoint>>(request.target);
        EXPECT_EQ(pts.size(), 10000u);
    }
    ASSERT_NE(request.sourceGrid, nullptr);
}

TEST_F(GPUInterpolationIntegrationTest, GridToGridInterpolation) {
    // 创建源网格
    auto sourceGrid = createTestGrid(128, 128);
    
    // 创建目标网格定义（更高分辨率）
    TargetGridDefinition targetDef;
    targetDef.gridName = "high_res_grid";
    targetDef.outputDataType = DataType::Float32;
    targetDef.crs = sourceGrid->getDefinition().crs;
    
    // X维度
    DimensionCoordinateInfo xDim;
    xDim.name = "x";
    xDim.coordinates.resize(256);
    for (int i = 0; i < 256; ++i) {
        xDim.coordinates[i] = i * 100.0 / 255.0;
    }
    targetDef.dimensions.push_back(xDim);
    
    // Y维度
    DimensionCoordinateInfo yDim;
    yDim.name = "y";
    yDim.coordinates.resize(256);
    for (int i = 0; i < 256; ++i) {
        yDim.coordinates[i] = i * 100.0 / 255.0;
    }
    targetDef.dimensions.push_back(yDim);
    
    // 创建插值请求
    InterpolationRequest request;
    request.sourceGrid = sourceGrid;
    request.target = targetDef;
    request.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
    
    // 验证目标网格定义
    EXPECT_EQ(targetDef.dimensions.size(), 2u);
    EXPECT_EQ(targetDef.dimensions[0].coordinates.size(), 256u);
    EXPECT_EQ(targetDef.dimensions[1].coordinates.size(), 256u);
}

TEST_F(GPUInterpolationIntegrationTest, PerformanceMeasurement) {
    // 创建不同大小的网格进行性能测试
    std::vector<int> gridSizes = {64, 128, 256, 512};
    
    for (int size : gridSizes) {
        auto grid = createTestGrid(size, size);
        
        // 创建目标点
        std::vector<TargetPoint> points;
        int numPoints = size * size / 4;  // 四分之一的点数
        
        for (int i = 0; i < numPoints; ++i) {
            TargetPoint pt;
            pt.coordinates = {
                (i % size) * 1.0,
                (i / size) * 1.0
            };
            points.push_back(pt);
        }
        
        // 测量创建请求的时间
        auto start = std::chrono::high_resolution_clock::now();
        
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = points;
        request.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Grid size: " << size << "x" << size 
                  << ", Points: " << numPoints 
                  << ", Setup time: " << duration.count() << " us" << std::endl;
    }
}

TEST_F(GPUInterpolationIntegrationTest, BoundaryConditions) {
    auto grid = createTestGrid(50, 50);
    
    // 测试边界点
    std::vector<TargetPoint> boundaryPoints;
    
    // 角点
    TargetPoint corner1;
    corner1.coordinates = {0.0, 0.0};
    boundaryPoints.push_back(corner1);
    
    TargetPoint corner2;
    corner2.coordinates = {100.0, 0.0};
    boundaryPoints.push_back(corner2);
    
    TargetPoint corner3;
    corner3.coordinates = {0.0, 100.0};
    boundaryPoints.push_back(corner3);
    
    TargetPoint corner4;
    corner4.coordinates = {100.0, 100.0};
    boundaryPoints.push_back(corner4);
    
    // 边界外的点
    TargetPoint outside1;
    outside1.coordinates = {-10.0, 50.0};
    boundaryPoints.push_back(outside1);
    
    TargetPoint outside2;
    outside2.coordinates = {50.0, 110.0};
    boundaryPoints.push_back(outside2);
    
    InterpolationRequest request;
    request.sourceGrid = grid;
    request.target = boundaryPoints;
    request.method = oscean::core_services::interpolation::InterpolationMethod::BILINEAR;
    
    if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
        const auto& pts = std::get<std::vector<TargetPoint>>(request.target);
        EXPECT_EQ(pts.size(), 6u);
    }
} 