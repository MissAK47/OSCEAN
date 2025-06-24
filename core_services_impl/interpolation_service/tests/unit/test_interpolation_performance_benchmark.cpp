#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>

// 插值服务接口和实现
#include "core_services/interpolation/i_interpolation_service.h"
#include "../../src/impl/interpolation_service_impl.cpp"

// Common Utilities
#include "common_utils/simd/isimd_manager.h"
#include "common_utils/simd/simd_manager_unified.h"

using namespace oscean::core_services;
using namespace oscean::core_services::interpolation;

/**
 * @brief 性能基准测试结果
 */
struct BenchmarkResult {
    std::string testName;
    std::string algorithmName;
    size_t dataPoints;
    size_t targetPoints;
    double executionTimeMs;
    double throughputPointsPerSec;
    double memoryUsageMB;
    size_t validResults;
    double successRate;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "测试: " << testName << std::endl;
        std::cout << "  算法: " << algorithmName << std::endl;
        std::cout << "  数据点: " << dataPoints << std::endl;
        std::cout << "  目标点: " << targetPoints << std::endl;
        std::cout << "  执行时间: " << executionTimeMs << "ms" << std::endl;
        std::cout << "  吞吐量: " << throughputPointsPerSec << " 点/秒" << std::endl;
        std::cout << "  内存使用: " << memoryUsageMB << "MB" << std::endl;
        std::cout << "  成功率: " << (successRate * 100) << "%" << std::endl;
        std::cout << std::endl;
    }
};

/**
 * @brief 插值性能基准测试类
 */
class InterpolationPerformanceBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建SIMD管理器
        simdManager_ = std::make_shared<oscean::common_utils::simd::UnifiedSIMDManager>();
        
        // 创建服务实例
        serviceWithSIMD_ = std::make_unique<InterpolationServiceImpl>(simdManager_, true);
        serviceWithoutSIMD_ = std::make_unique<InterpolationServiceImpl>(nullptr, true);
        
        std::cout << "=== 插值性能基准测试 ===" << std::endl;
        std::cout << "SIMD支持: " << (simdManager_ ? "是" : "否") << std::endl;
        std::cout << std::endl;
    }
    
    /**
     * @brief 创建大规模测试网格
     */
    std::shared_ptr<GridData> createLargeTestGrid(size_t cols, size_t rows, size_t bands = 1) {
        auto grid = std::make_shared<GridData>(cols, rows, bands, DataType::Float32);
        
        // 设置地理变换
        std::vector<double> geoTransform = {0.0, 0.01, 0.0, 0.0, 0.0, 0.01};
        grid->setGeoTransform(geoTransform);
        
        // 设置CRS信息
        CRSInfo crs;
        crs.wkt = "EPSG:4326";
        crs.isGeographic = true;
        grid->setCrs(crs);
        
        // 设置边界框
        auto& definition = const_cast<GridDefinition&>(grid->getDefinition());
        definition.extent.minX = 0.0;
        definition.extent.maxX = static_cast<double>(cols - 1) * 0.01;
        definition.extent.minY = 0.0;
        definition.extent.maxY = static_cast<double>(rows - 1) * 0.01;
        definition.extent.crsId = "EPSG:4326";
        
        std::cout << "生成测试网格: " << cols << "x" << rows;
        if (bands > 1) std::cout << "x" << bands;
        std::cout << " = " << (cols * rows * bands) << " 数据点..." << std::endl;
        
        // 填充复杂的测试数据，模拟真实地理数据
        for (size_t band = 0; band < bands; ++band) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    double x = static_cast<double>(col) * 0.01;
                    double y = static_cast<double>(row) * 0.01;
                    double z = static_cast<double>(band);
                    
                    // 复杂的多频率函数，模拟地形数据
                    double value = 100.0 * std::sin(x * 10.0) * std::cos(y * 10.0) +
                                  50.0 * std::sin(x * 5.0) +
                                  25.0 * std::cos(y * 7.0) +
                                  10.0 * std::sin(x * 20.0) * std::sin(y * 20.0) +
                                  (z + 1.0) * 5.0;
                    
                    grid->setValue(row, col, band, static_cast<float>(value));
                }
                
                // 进度显示
                if (row % (rows / 10) == 0) {
                    std::cout << "  进度: " << (row * 100 / rows) << "%" << std::endl;
                }
            }
        }
        
        std::cout << "测试网格生成完成！" << std::endl;
        return grid;
    }
    
    /**
     * @brief 创建大量测试目标点
     */
    std::vector<TargetPoint> createLargeTestPoints(const BoundingBox& bounds, size_t count) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        std::cout << "生成 " << count << " 个测试目标点..." << std::endl;
        
        // 在边界内生成分布均匀的点
        double marginX = (bounds.maxX - bounds.minX) * 0.05;
        double marginY = (bounds.maxY - bounds.minY) * 0.05;
        
        double safeMinX = bounds.minX + marginX;
        double safeMaxX = bounds.maxX - marginX;
        double safeMinY = bounds.minY + marginY;
        double safeMaxY = bounds.maxY - marginY;
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            
            // 使用更好的分布算法
            double u = static_cast<double>(i) / (count - 1);
            double v = static_cast<double>(i % 1000) / 999.0;
            
            point.coordinates = {
                safeMinX + (safeMaxX - safeMinX) * u,
                safeMinY + (safeMaxY - safeMinY) * v
            };
            points.push_back(point);
        }
        
        std::cout << "目标点生成完成！" << std::endl;
        return points;
    }
    
    /**
     * @brief 执行性能基准测试
     */
    BenchmarkResult runBenchmark(
        const std::string& testName,
        IInterpolationService* service,
        std::shared_ptr<GridData> grid,
        const std::vector<TargetPoint>& targetPoints,
        InterpolationMethod method) {
        
        BenchmarkResult result;
        result.testName = testName;
        result.algorithmName = getAlgorithmName(method);
        result.dataPoints = grid->getDefinition().cols * grid->getDefinition().rows * grid->getBandCount();
        result.targetPoints = targetPoints.size();
        
        // 创建插值请求
        InterpolationRequest request;
        request.sourceGrid = grid;
        request.target = targetPoints;
        request.method = method;
        
        std::cout << "执行基准测试: " << testName << " - " << result.algorithmName << std::endl;
        
        // 预热运行
        auto warmupFuture = service->interpolateAsync(request);
        warmupFuture.get();
        
        // 正式测试运行
        auto startTime = std::chrono::high_resolution_clock::now();
        auto future = service->interpolateAsync(request);
        auto interpolationResult = future.get();
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        result.executionTimeMs = duration.count() / 1000.0;
        result.throughputPointsPerSec = result.executionTimeMs > 0 ? 
            (result.targetPoints * 1000.0 / result.executionTimeMs) : 0;
        
        // 估算内存使用
        result.memoryUsageMB = (result.dataPoints * sizeof(float) + 
                               result.targetPoints * sizeof(double) * 2) / (1024.0 * 1024.0);
        
        // 验证结果
        if (interpolationResult.statusCode == 0 && 
            std::holds_alternative<std::vector<std::optional<double>>>(interpolationResult.data)) {
            
            const auto& values = std::get<std::vector<std::optional<double>>>(interpolationResult.data);
            result.validResults = 0;
            for (const auto& value : values) {
                if (value.has_value() && !std::isnan(value.value()) && !std::isinf(value.value())) {
                    result.validResults++;
                }
            }
            result.successRate = static_cast<double>(result.validResults) / result.targetPoints;
        } else {
            result.validResults = 0;
            result.successRate = 0.0;
        }
        
        return result;
    }
    
    std::string getAlgorithmName(InterpolationMethod method) const {
        switch (method) {
            case InterpolationMethod::BILINEAR: return "双线性插值";
            case InterpolationMethod::NEAREST_NEIGHBOR: return "最近邻插值";
            case InterpolationMethod::TRILINEAR: return "三线性插值";
            case InterpolationMethod::CUBIC_SPLINE_1D: return "立方样条插值";
            case InterpolationMethod::PCHIP_RECURSIVE_NDIM: return "PCHIP插值";
            default: return "未知算法";
        }
    }

protected:
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;
    std::unique_ptr<IInterpolationService> serviceWithSIMD_;
    std::unique_ptr<IInterpolationService> serviceWithoutSIMD_;
};

// === 小规模性能基准测试 ===

TEST_F(InterpolationPerformanceBenchmarkTest, SmallScaleBenchmark) {
    // 100x100 网格，1000个目标点
    auto grid = createLargeTestGrid(100, 100);
    auto bounds = grid->getSpatialExtent();
    auto targetPoints = createLargeTestPoints(bounds, 1000);
    
    std::vector<InterpolationMethod> methods = {
        InterpolationMethod::NEAREST_NEIGHBOR,
        InterpolationMethod::BILINEAR,
        InterpolationMethod::CUBIC_SPLINE_1D
    };
    
    std::vector<BenchmarkResult> results;
    
    for (auto method : methods) {
        auto result = runBenchmark("小规模测试", serviceWithSIMD_.get(), grid, targetPoints, method);
        results.push_back(result);
        result.print();
        
        // 验证基本性能要求
        EXPECT_GT(result.successRate, 0.9) << "成功率应该大于90%";
        EXPECT_LT(result.executionTimeMs, 1000) << "执行时间应该小于1秒";
        EXPECT_GT(result.throughputPointsPerSec, 100) << "吞吐量应该大于100点/秒";
    }
}

// === 中等规模性能基准测试 ===

TEST_F(InterpolationPerformanceBenchmarkTest, MediumScaleBenchmark) {
    // 500x500 网格，5000个目标点
    auto grid = createLargeTestGrid(500, 500);
    auto bounds = grid->getSpatialExtent();
    auto targetPoints = createLargeTestPoints(bounds, 5000);
    
    std::vector<InterpolationMethod> methods = {
        InterpolationMethod::NEAREST_NEIGHBOR,
        InterpolationMethod::BILINEAR
    };
    
    std::vector<BenchmarkResult> results;
    
    for (auto method : methods) {
        auto result = runBenchmark("中等规模测试", serviceWithSIMD_.get(), grid, targetPoints, method);
        results.push_back(result);
        result.print();
        
        // 验证中等规模性能要求
        EXPECT_GT(result.successRate, 0.85) << "成功率应该大于85%";
        EXPECT_LT(result.executionTimeMs, 5000) << "执行时间应该小于5秒";
        EXPECT_GT(result.throughputPointsPerSec, 50) << "吞吐量应该大于50点/秒";
    }
}

// === 大规模性能基准测试 ===

TEST_F(InterpolationPerformanceBenchmarkTest, LargeScaleBenchmark) {
    // 1000x1000 网格，10000个目标点
    auto grid = createLargeTestGrid(1000, 1000);
    auto bounds = grid->getSpatialExtent();
    auto targetPoints = createLargeTestPoints(bounds, 10000);
    
    std::vector<InterpolationMethod> methods = {
        InterpolationMethod::NEAREST_NEIGHBOR,
        InterpolationMethod::BILINEAR
    };
    
    std::vector<BenchmarkResult> results;
    
    for (auto method : methods) {
        auto result = runBenchmark("大规模测试", serviceWithSIMD_.get(), grid, targetPoints, method);
        results.push_back(result);
        result.print();
        
        // 验证大规模性能要求
        EXPECT_GT(result.successRate, 0.8) << "成功率应该大于80%";
        EXPECT_LT(result.executionTimeMs, 15000) << "执行时间应该小于15秒";
        EXPECT_GT(result.throughputPointsPerSec, 20) << "吞吐量应该大于20点/秒";
    }
}

// === SIMD性能对比基准测试 ===

TEST_F(InterpolationPerformanceBenchmarkTest, SIMDComparisonBenchmark) {
    // 使用足够大的数据集来测量SIMD效果
    auto grid = createLargeTestGrid(800, 800);
    auto bounds = grid->getSpatialExtent();
    auto targetPoints = createLargeTestPoints(bounds, 8000);
    
    InterpolationMethod method = InterpolationMethod::BILINEAR;
    
    // SIMD版本测试
    auto simdResult = runBenchmark("SIMD版本", serviceWithSIMD_.get(), grid, targetPoints, method);
    simdResult.print();
    
    // 标量版本测试
    auto scalarResult = runBenchmark("标量版本", serviceWithoutSIMD_.get(), grid, targetPoints, method);
    scalarResult.print();
    
    // 性能对比分析
    double speedup = scalarResult.executionTimeMs / simdResult.executionTimeMs;
    double throughputImprovement = simdResult.throughputPointsPerSec / scalarResult.throughputPointsPerSec;
    
    std::cout << "=== SIMD性能对比分析 ===" << std::endl;
    std::cout << "加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "吞吐量提升: " << std::fixed << std::setprecision(2) << throughputImprovement << "x" << std::endl;
    
    // 验证SIMD性能提升
    EXPECT_GT(simdResult.successRate, 0.9) << "SIMD版本成功率应该大于90%";
    EXPECT_GT(scalarResult.successRate, 0.9) << "标量版本成功率应该大于90%";
    
    // SIMD应该提供性能提升或至少不降低性能
    EXPECT_GE(speedup, 0.8) << "SIMD版本不应该显著慢于标量版本";
    
    // 结果一致性检查
    EXPECT_NEAR(simdResult.successRate, scalarResult.successRate, 0.05) 
        << "SIMD和标量版本的成功率应该相近";
}

// === 3D数据性能基准测试 ===

TEST_F(InterpolationPerformanceBenchmarkTest, ThreeDimensionalBenchmark) {
    // 100x100x20 3D网格，2000个目标点
    auto grid = createLargeTestGrid(100, 100, 20);
    auto bounds = grid->getSpatialExtent();
    auto targetPoints = createLargeTestPoints(bounds, 2000);
    
    auto result = runBenchmark("3D数据测试", serviceWithSIMD_.get(), grid, targetPoints, 
                              InterpolationMethod::TRILINEAR);
    result.print();
    
    // 验证3D性能要求
    EXPECT_GT(result.successRate, 0.8) << "3D插值成功率应该大于80%";
    EXPECT_LT(result.executionTimeMs, 10000) << "3D插值执行时间应该小于10秒";
    EXPECT_GT(result.throughputPointsPerSec, 10) << "3D插值吞吐量应该大于10点/秒";
}

// === 综合性能报告 ===

TEST_F(InterpolationPerformanceBenchmarkTest, ComprehensivePerformanceReport) {
    std::cout << "\n=== 综合性能基准测试报告 ===" << std::endl;
    
    // 测试不同规模的数据
    std::vector<std::pair<std::string, std::pair<size_t, size_t>>> testCases = {
        {"超小规模", {50, 100}},
        {"小规模", {100, 500}},
        {"中规模", {200, 1000}},
        {"大规模", {500, 2000}}
    };
    
    std::vector<BenchmarkResult> allResults;
    
    for (const auto& testCase : testCases) {
        const std::string& name = testCase.first;
        size_t gridSize = testCase.second.first;
        size_t targetCount = testCase.second.second;
        
        auto grid = createLargeTestGrid(gridSize, gridSize);
        auto bounds = grid->getSpatialExtent();
        auto targetPoints = createLargeTestPoints(bounds, targetCount);
        
        auto result = runBenchmark(name, serviceWithSIMD_.get(), grid, targetPoints, 
                                  InterpolationMethod::BILINEAR);
        allResults.push_back(result);
    }
    
    // 生成性能报告
    std::cout << "\n=== 性能扩展性分析 ===" << std::endl;
    std::cout << std::setw(12) << "测试规模" 
              << std::setw(12) << "数据点" 
              << std::setw(12) << "目标点" 
              << std::setw(12) << "时间(ms)" 
              << std::setw(15) << "吞吐量(点/秒)" 
              << std::setw(12) << "成功率(%)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    for (const auto& result : allResults) {
        std::cout << std::setw(12) << result.testName
                  << std::setw(12) << result.dataPoints
                  << std::setw(12) << result.targetPoints
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.executionTimeMs
                  << std::setw(15) << std::fixed << std::setprecision(0) << result.throughputPointsPerSec
                  << std::setw(12) << std::fixed << std::setprecision(1) << (result.successRate * 100)
                  << std::endl;
    }
    
    // 验证性能扩展性
    for (const auto& result : allResults) {
        EXPECT_GT(result.successRate, 0.8) << result.testName << " 成功率应该大于80%";
        EXPECT_GT(result.throughputPointsPerSec, 10) << result.testName << " 吞吐量应该大于10点/秒";
    }
} 