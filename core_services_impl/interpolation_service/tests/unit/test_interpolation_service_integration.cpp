#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>

// 插值服务接口和实现
#include "core_services/interpolation/i_interpolation_service.h"
#include "../../src/impl/interpolation_service_impl.cpp"

// Common Utilities
#include "common_utils/simd/isimd_manager.h"
#include "common_utils/simd/simd_manager_unified.h"

using namespace oscean::core_services;
using namespace oscean::core_services::interpolation;

/**
 * @brief 插值服务集成测试类
 * @details 测试服务层的完整功能，包括依赖注入、算法选择、异步处理等
 */
class InterpolationServiceIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建真实的SIMD管理器
        simdManager_ = std::make_shared<oscean::common_utils::simd::UnifiedSIMDManager>();
        
        // 创建插值服务实例（启用智能选择）
        serviceWithSIMD_ = std::make_unique<InterpolationServiceImpl>(simdManager_, true);
        
        // 创建不使用SIMD的服务实例
        serviceWithoutSIMD_ = std::make_unique<InterpolationServiceImpl>(nullptr, true);
        
        // 创建测试数据
        createTestData();
    }
    
    void createTestData() {
        // 创建小规模测试网格 (10x10)
        smallGrid_ = createTestGrid(10, 10, 1);
        
        // 创建中等规模测试网格 (100x100)
        mediumGrid_ = createTestGrid(100, 100, 1);
        
        // 创建大规模测试网格 (500x500)
        largeGrid_ = createTestGrid(500, 500, 1);
        
        // 创建3D测试网格 (50x50x10)
        grid3D_ = createTestGrid(50, 50, 10);
        
        // 创建测试目标点
        smallTargetPoints_ = createTestPoints(10);
        mediumTargetPoints_ = createTestPoints(100);
        largeTargetPoints_ = createTestPoints(1000);
    }
    
    std::shared_ptr<GridData> createTestGrid(size_t cols, size_t rows, size_t bands) {
        auto grid = std::make_shared<GridData>(cols, rows, bands, DataType::Float32);
        
        // 设置地理变换
        std::vector<double> geoTransform = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        grid->setGeoTransform(geoTransform);
        
        // 设置CRS信息
        CRSInfo crs;
        crs.wkt = "EPSG:4326";
        crs.isGeographic = true;
        grid->setCrs(crs);
        
        // 设置边界框
        auto& definition = const_cast<GridDefinition&>(grid->getDefinition());
        definition.extent.minX = 0.0;
        definition.extent.maxX = static_cast<double>(cols - 1);
        definition.extent.minY = 0.0;
        definition.extent.maxY = static_cast<double>(rows - 1);
        definition.extent.crsId = "EPSG:4326";
        
        // 填充测试数据 f(x,y,z) = sin(x/10) * cos(y/10) * (z+1)
        for (size_t band = 0; band < bands; ++band) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    double x = static_cast<double>(col);
                    double y = static_cast<double>(row);
                    double z = static_cast<double>(band);
                    double value = std::sin(x / 10.0) * std::cos(y / 10.0) * (z + 1.0);
                    grid->setValue(row, col, band, static_cast<float>(value));
                }
            }
        }
        
        return grid;
    }
    
    std::vector<TargetPoint> createTestPoints(size_t count) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            point.coordinates = {
                static_cast<double>(i % 10) + 0.5,  // X坐标
                static_cast<double>(i / 10) + 0.5   // Y坐标
            };
            points.push_back(point);
        }
        
        return points;
    }

protected:
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;
    std::unique_ptr<IInterpolationService> serviceWithSIMD_;
    std::unique_ptr<IInterpolationService> serviceWithoutSIMD_;
    
    std::shared_ptr<GridData> smallGrid_;
    std::shared_ptr<GridData> mediumGrid_;
    std::shared_ptr<GridData> largeGrid_;
    std::shared_ptr<GridData> grid3D_;
    
    std::vector<TargetPoint> smallTargetPoints_;
    std::vector<TargetPoint> mediumTargetPoints_;
    std::vector<TargetPoint> largeTargetPoints_;
};

// === 服务基础功能测试 ===

TEST_F(InterpolationServiceIntegrationTest, ServiceConstruction) {
    // 测试服务构造
    ASSERT_NE(serviceWithSIMD_, nullptr);
    ASSERT_NE(serviceWithoutSIMD_, nullptr);
    
    // 测试支持的方法
    auto methodsWithSIMD = serviceWithSIMD_->getSupportedMethods();
    auto methodsWithoutSIMD = serviceWithoutSIMD_->getSupportedMethods();
    
    EXPECT_GT(methodsWithSIMD.size(), 0);
    EXPECT_EQ(methodsWithSIMD.size(), methodsWithoutSIMD.size());
    
    // 验证包含主要插值方法
    EXPECT_TRUE(std::find(methodsWithSIMD.begin(), methodsWithSIMD.end(), 
                         InterpolationMethod::BILINEAR) != methodsWithSIMD.end());
    EXPECT_TRUE(std::find(methodsWithSIMD.begin(), methodsWithSIMD.end(), 
                         InterpolationMethod::NEAREST_NEIGHBOR) != methodsWithSIMD.end());
    
    std::cout << "支持的插值方法数量: " << methodsWithSIMD.size() << std::endl;
}

TEST_F(InterpolationServiceIntegrationTest, BasicAsyncInterpolation) {
    // 测试基本的异步插值功能
    InterpolationRequest request;
    request.sourceGrid = smallGrid_;
    request.target = smallTargetPoints_;
    request.method = InterpolationMethod::BILINEAR;
    
    auto future = serviceWithSIMD_->interpolateAsync(request);
    auto result = future.get();
    
    EXPECT_EQ(result.statusCode, 0);
    EXPECT_TRUE(std::holds_alternative<std::vector<std::optional<double>>>(result.data));
    
    auto values = std::get<std::vector<std::optional<double>>>(result.data);
    EXPECT_EQ(values.size(), smallTargetPoints_.size());
    
    // 验证结果的合理性
    size_t validCount = 0;
    for (const auto& value : values) {
        if (value.has_value()) {
            validCount++;
            EXPECT_FALSE(std::isnan(value.value()));
            EXPECT_FALSE(std::isinf(value.value()));
        }
    }
    EXPECT_GT(validCount, 0);
    
    std::cout << "基础异步插值测试: " << validCount << "/" << values.size() << " 有效结果" << std::endl;
}

// === 智能算法选择测试 ===

TEST_F(InterpolationServiceIntegrationTest, SmartAlgorithmSelection) {
    // 测试智能算法选择功能
    InterpolationRequest request;
    request.sourceGrid = mediumGrid_;
    request.target = mediumTargetPoints_;
    request.method = InterpolationMethod::UNKNOWN; // 让服务自动选择
    
    auto future = serviceWithSIMD_->interpolateAsync(request);
    auto result = future.get();
    
    EXPECT_EQ(result.statusCode, 0);
    EXPECT_FALSE(result.message.empty());
    
    auto values = std::get<std::vector<std::optional<double>>>(result.data);
    EXPECT_EQ(values.size(), mediumTargetPoints_.size());
    
    std::cout << "智能算法选择测试: " << result.message << std::endl;
}

// === SIMD性能对比测试 ===

TEST_F(InterpolationServiceIntegrationTest, SIMDPerformanceComparison) {
    // 使用大数据集测试SIMD性能提升
    InterpolationRequest request;
    request.sourceGrid = largeGrid_;
    request.target = largeTargetPoints_;
    request.method = InterpolationMethod::BILINEAR;
    
    // 测试SIMD版本
    auto simdStartTime = std::chrono::high_resolution_clock::now();
    auto simdFuture = serviceWithSIMD_->interpolateAsync(request);
    auto simdResult = simdFuture.get();
    auto simdEndTime = std::chrono::high_resolution_clock::now();
    
    // 测试非SIMD版本
    auto scalarStartTime = std::chrono::high_resolution_clock::now();
    auto scalarFuture = serviceWithoutSIMD_->interpolateAsync(request);
    auto scalarResult = scalarFuture.get();
    auto scalarEndTime = std::chrono::high_resolution_clock::now();
    
    auto simdDuration = std::chrono::duration_cast<std::chrono::milliseconds>(simdEndTime - simdStartTime);
    auto scalarDuration = std::chrono::duration_cast<std::chrono::milliseconds>(scalarEndTime - scalarStartTime);
    
    EXPECT_EQ(simdResult.statusCode, 0);
    EXPECT_EQ(scalarResult.statusCode, 0);
    
    auto simdValues = std::get<std::vector<std::optional<double>>>(simdResult.data);
    auto scalarValues = std::get<std::vector<std::optional<double>>>(scalarResult.data);
    
    EXPECT_EQ(simdValues.size(), scalarValues.size());
    
    // 验证结果一致性
    size_t consistentResults = 0;
    for (size_t i = 0; i < simdValues.size(); ++i) {
        if (simdValues[i].has_value() && scalarValues[i].has_value()) {
            double diff = std::abs(simdValues[i].value() - scalarValues[i].value());
            if (diff < 1e-6) {
                consistentResults++;
            }
        }
    }
    
    double speedup = static_cast<double>(scalarDuration.count()) / simdDuration.count();
    
    std::cout << "SIMD性能对比测试:" << std::endl;
    std::cout << "  SIMD时间: " << simdDuration.count() << "ms" << std::endl;
    std::cout << "  标量时间: " << scalarDuration.count() << "ms" << std::endl;
    std::cout << "  加速比: " << speedup << "x" << std::endl;
    std::cout << "  结果一致性: " << consistentResults << "/" << simdValues.size() << std::endl;
    
    // SIMD应该提供性能提升（至少不应该更慢）
    EXPECT_LE(simdDuration.count(), scalarDuration.count() * 1.2);
    
    // 结果应该基本一致
    EXPECT_GT(consistentResults, simdValues.size() * 0.95);
}

// === 多维数据测试 ===

TEST_F(InterpolationServiceIntegrationTest, MultiDimensionalDataInterpolation) {
    // 测试3D数据插值
    InterpolationRequest request;
    request.sourceGrid = grid3D_;
    request.target = mediumTargetPoints_;
    request.method = InterpolationMethod::TRILINEAR;
    
    auto future = serviceWithSIMD_->interpolateAsync(request);
    auto result = future.get();
    
    EXPECT_EQ(result.statusCode, 0);
    
    auto values = std::get<std::vector<std::optional<double>>>(result.data);
    EXPECT_EQ(values.size(), mediumTargetPoints_.size());
    
    // 验证3D插值结果
    size_t validCount = 0;
    for (const auto& value : values) {
        if (value.has_value()) {
            validCount++;
            EXPECT_FALSE(std::isnan(value.value()));
            EXPECT_FALSE(std::isinf(value.value()));
        }
    }
    
    std::cout << "3D数据插值测试: " << validCount << "/" << values.size() << " 有效结果" << std::endl;
    EXPECT_GT(validCount, values.size() * 0.8);
}

// === 错误处理测试 ===

TEST_F(InterpolationServiceIntegrationTest, ErrorHandling) {
    // 测试空数据处理
    InterpolationRequest emptyRequest;
    emptyRequest.sourceGrid = nullptr;
    emptyRequest.target = smallTargetPoints_;
    emptyRequest.method = InterpolationMethod::BILINEAR;
    
    auto emptyFuture = serviceWithSIMD_->interpolateAsync(emptyRequest);
    auto emptyResult = emptyFuture.get();
    
    EXPECT_NE(emptyResult.statusCode, 0);
    EXPECT_FALSE(emptyResult.message.empty());
    
    // 测试无效方法处理
    InterpolationRequest invalidRequest;
    invalidRequest.sourceGrid = smallGrid_;
    invalidRequest.target = smallTargetPoints_;
    invalidRequest.method = static_cast<InterpolationMethod>(999); // 无效方法
    
    auto invalidFuture = serviceWithSIMD_->interpolateAsync(invalidRequest);
    auto invalidResult = invalidFuture.get();
    
    EXPECT_NE(invalidResult.statusCode, 0);
    EXPECT_FALSE(invalidResult.message.empty());
    
    std::cout << "错误处理测试:" << std::endl;
    std::cout << "  空数据错误: " << emptyResult.message << std::endl;
    std::cout << "  无效方法错误: " << invalidResult.message << std::endl;
}

// === 并发处理测试 ===

TEST_F(InterpolationServiceIntegrationTest, ConcurrentProcessing) {
    // 测试并发处理能力
    const size_t numConcurrentRequests = 10;
    std::vector<boost::future<InterpolationResult>> futures;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 创建多个并发请求
    for (size_t i = 0; i < numConcurrentRequests; ++i) {
        InterpolationRequest request;
        request.sourceGrid = mediumGrid_;
        request.target = mediumTargetPoints_;
        request.method = (i % 2 == 0) ? InterpolationMethod::BILINEAR : InterpolationMethod::NEAREST_NEIGHBOR;
        
        futures.push_back(serviceWithSIMD_->interpolateAsync(request));
    }
    
    // 等待所有请求完成
    size_t successCount = 0;
    for (auto& future : futures) {
        auto result = future.get();
        if (result.statusCode == 0) {
            successCount++;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    EXPECT_EQ(successCount, numConcurrentRequests);
    
    std::cout << "并发处理测试:" << std::endl;
    std::cout << "  并发请求数: " << numConcurrentRequests << std::endl;
    std::cout << "  成功处理数: " << successCount << std::endl;
    std::cout << "  总处理时间: " << totalDuration.count() << "ms" << std::endl;
    std::cout << "  平均处理时间: " << totalDuration.count() / numConcurrentRequests << "ms/请求" << std::endl;
    
    // 并发处理应该在合理时间内完成
    EXPECT_LT(totalDuration.count(), 5000); // 不超过5秒
} 