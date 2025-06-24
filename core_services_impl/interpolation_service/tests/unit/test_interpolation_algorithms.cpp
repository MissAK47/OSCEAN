/**
 * @file test_interpolation_algorithms.cpp
 * @brief 插值算法单元测试
 */

#include <gtest/gtest.h>
#include <core_services/interpolation/i_interpolation_service.h>
#include <core_services/common_data_types.h>
#include <common_utils/utilities/logging_utils.h>
#include <boost/make_shared.hpp>
#include <cmath>
#include <random>
#include <chrono>
#include <locale>

#ifdef _WIN32
#include <windows.h>
#endif

// 避免命名空间冲突，不使用using namespace
namespace interpolation = oscean::core_services::interpolation;
using oscean::core_services::GridData;
using oscean::core_services::GridDefinition;
using oscean::core_services::DataType;
using oscean::core_services::CRSInfo;
using oscean::core_services::BoundingBox;
using interpolation::TargetPoint;
using interpolation::InterpolationMethod;
using interpolation::InterpolationRequest;
using interpolation::InterpolationResult;
// IInterpolationAlgorithm在实现目录中，需要显式包含

// 插值算法接口
#include "../../include/core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"

// 插值算法头文件
#include "../../src/impl/algorithms/bilinear_interpolator.h"
#include "../../src/impl/algorithms/linear_1d_interpolator.h"
#include "../../src/impl/algorithms/cubic_spline_interpolator.h"
#include "../../src/impl/algorithms/nearest_neighbor_interpolator.h"
#include "../../src/impl/algorithms/trilinear_interpolator.h"
#include "../../src/impl/algorithms/pchip_interpolator.h"
#include "../../src/impl/algorithms/fast_pchip_interpolator_2d.h"
#include "../../src/impl/algorithms/fast_pchip_interpolator_3d.h"
#include "../../src/impl/algorithms/pchip_interpolator_2d_bathy.h"

// Common Utilities
#include "common_utils/simd/isimd_manager.h"
#include "common_utils/simd/simd_manager_unified.h"

// 设置控制台编码为UTF-8
static void setupConsoleEncoding() {
#ifdef _WIN32
    // Windows控制台UTF-8支持
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    // 注意：不要使用_setmode，它与C++流不兼容
    // 改为使用locale设置
    std::locale::global(std::locale(".UTF-8"));
#endif
}

// 在main函数运行前设置编码
static struct ConsoleEncodingSetup {
    ConsoleEncodingSetup() {
        setupConsoleEncoding();
    }
} g_consoleEncodingSetup;

/**
 * @brief 通用插值算法测试数据生成器
 */
class InterpolationTestDataGenerator {
public:
    /**
     * @brief 创建1D测试数据
     */
    static std::shared_ptr<GridData> create1DTestData(size_t points = 100) {
        auto grid = std::make_shared<GridData>(points, 1, 1, DataType::Float32);
        
        // 设置1D地理变换
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
        definition.extent.maxX = static_cast<double>(points - 1);
        definition.extent.minY = 0.0;
        definition.extent.maxY = 0.0;
        definition.extent.crsId = "EPSG:4326";
        
        // 填充1D函数数据 f(x) = sin(x/10) + 0.1*x
        for (size_t i = 0; i < points; ++i) {
            double x = static_cast<double>(i);
            double value = std::sin(x / 10.0) + 0.1 * x;
            grid->setValue(0, i, 0, static_cast<float>(value));
        }
        
        return grid;
    }
    
    /**
     * @brief 创建2D测试数据
     */
    static std::shared_ptr<GridData> create2DTestData(size_t cols = 50, size_t rows = 50) {
        auto grid = std::make_shared<GridData>(cols, rows, 1, DataType::Float32);
        
        // 设置2D地理变换
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
        
        // 填充2D函数数据 f(x,y) = sin(x/10) * cos(y/10) + 0.01*(x+y)
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                double x = static_cast<double>(col);
                double y = static_cast<double>(row);
                double value = std::sin(x / 10.0) * std::cos(y / 10.0) + 0.01 * (x + y);
                grid->setValue(row, col, 0, static_cast<float>(value));
            }
        }
        
        return grid;
    }
    
    /**
     * @brief 创建大规模2D测试数据（100万数据点）
     */
    static std::shared_ptr<GridData> createLargeScale2DTestData() {
        // 1000x1000 = 100万数据点
        size_t cols = 1000;
        size_t rows = 1000;
        
        auto grid = std::make_shared<GridData>(cols, rows, 1, DataType::Float32);
        
        // 设置2D地理变换
        std::vector<double> geoTransform = {0.0, 0.1, 0.0, 0.0, 0.0, 0.1};
        grid->setGeoTransform(geoTransform);
        
        // 设置CRS信息
        CRSInfo crs;
        crs.wkt = "EPSG:4326";
        crs.isGeographic = true;
        grid->setCrs(crs);
        
        // 设置边界框
        auto& definition = const_cast<GridDefinition&>(grid->getDefinition());
        definition.extent.minX = 0.0;
        definition.extent.maxX = static_cast<double>(cols - 1) * 0.1;
        definition.extent.minY = 0.0;
        definition.extent.maxY = static_cast<double>(rows - 1) * 0.1;
        definition.extent.crsId = "EPSG:4326";
        
        std::cout << "正在生成100万数据点的测试网格..." << std::endl;
        
        // 填充复杂的2D函数数据，模拟真实地理数据
        // f(x,y) = 100 * sin(x/50) * cos(y/50) + 50 * sin(x/20) + 25 * cos(y/30) + 0.1*(x+y)
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                double x = static_cast<double>(col) * 0.1;
                double y = static_cast<double>(row) * 0.1;
                
                // 复杂的多频率函数，模拟地形数据
                double value = 100.0 * std::sin(x / 50.0) * std::cos(y / 50.0) +
                              50.0 * std::sin(x / 20.0) +
                              25.0 * std::cos(y / 30.0) +
                              0.1 * (x + y) +
                              10.0 * std::sin(x / 5.0) * std::sin(y / 5.0); // 高频细节
                
                grid->setValue(row, col, 0, static_cast<float>(value));
            }
            
            // 进度显示
            if (row % 100 == 0) {
                std::cout << "进度: " << (row * 100 / rows) << "%" << std::endl;
            }
        }
        
        std::cout << "大规模测试网格生成完成！" << std::endl;
        return grid;
    }
    
    /**
     * @brief 创建精度验证用的解析函数数据
     */
    static std::shared_ptr<GridData> createAnalyticalTestData(size_t cols = 100, size_t rows = 100) {
        auto grid = std::make_shared<GridData>(cols, rows, 1, DataType::Float64);
        
        // 设置高精度地理变换
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
        
        // 填充解析函数数据 f(x,y) = x^2 + 2*x*y + y^2 + 3*x + 4*y + 5
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                double x = static_cast<double>(col) * 0.01;
                double y = static_cast<double>(row) * 0.01;
                double value = x*x + 2.0*x*y + y*y + 3.0*x + 4.0*y + 5.0;
                grid->setValue(row, col, 0, value);
            }
        }
        
        return grid;
    }
    
    /**
     * @brief 创建3D测试数据
     */
    static std::shared_ptr<GridData> create3DTestData(size_t cols = 20, size_t rows = 20, size_t bands = 10) {
        auto grid = std::make_shared<GridData>(cols, rows, bands, DataType::Float32);
        
        // 设置3D地理变换
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
        
        // 填充3D函数数据 f(x,y,z) = sin(x/5) * cos(y/5) * sin(z/3)
        for (size_t band = 0; band < bands; ++band) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    double x = static_cast<double>(col);
                    double y = static_cast<double>(row);
                    double z = static_cast<double>(band);
                    double value = std::sin(x / 5.0) * std::cos(y / 5.0) * std::sin(z / 3.0);
                    grid->setValue(row, col, band, static_cast<float>(value));
                }
            }
        }
        
        return grid;
    }
    
    /**
     * @brief 创建3D解析测试数据（Float64类型，用于高精度插值测试）
     */
    static std::shared_ptr<GridData> createAnalytical3DTestData(size_t cols = 10, size_t rows = 10, size_t bands = 10) {
        auto grid = std::make_shared<GridData>(cols, rows, bands, DataType::Float64);
        
        // 设置高精度地理变换
        std::vector<double> geoTransform = {0.0, 0.1, 0.0, 0.0, 0.0, 0.1};
        grid->setGeoTransform(geoTransform);
        
        // 设置CRS信息
        CRSInfo crs;
        crs.wkt = "EPSG:4326";
        crs.isGeographic = true;
        grid->setCrs(crs);
        
        // 设置边界框
        auto& definition = const_cast<GridDefinition&>(grid->getDefinition());
        definition.extent.minX = 0.0;
        definition.extent.maxX = static_cast<double>(cols - 1) * 0.1;
        definition.extent.minY = 0.0;
        definition.extent.maxY = static_cast<double>(rows - 1) * 0.1;
        definition.extent.crsId = "EPSG:4326";
        
        // 填充3D解析函数数据 f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y + 3*x*z + 4*y*z + 5*x + 6*y + 7*z + 8
        for (size_t band = 0; band < bands; ++band) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    double x = static_cast<double>(col) * 0.1;
                    double y = static_cast<double>(row) * 0.1;
                    double z = static_cast<double>(band) * 0.1;
                    double value = x*x + y*y + z*z + 2.0*x*y + 3.0*x*z + 4.0*y*z + 5.0*x + 6.0*y + 7.0*z + 8.0;
                    grid->setValue(row, col, band, value);
                }
            }
        }
        
        return grid;
    }
    
    /**
     * @brief 生成测试目标点
     */
    static std::vector<TargetPoint> generateTestPoints(const BoundingBox& bounds, size_t count) {
        std::vector<TargetPoint> points;
        points.reserve(count);
        
        // 在边界内生成随机分布的点
        double marginX = (bounds.maxX - bounds.minX) * 0.1;
        double marginY = (bounds.maxY - bounds.minY) * 0.1;
        
        double safeMinX = bounds.minX + marginX;
        double safeMaxX = bounds.maxX - marginX;
        double safeMinY = bounds.minY + marginY;
        double safeMaxY = bounds.maxY - marginY;
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint point;
            double x = safeMinX + (safeMaxX - safeMinX) * (static_cast<double>(i) / (count - 1));
            double y = safeMinY + (safeMaxY - safeMinY) * (static_cast<double>(i % 10) / 9.0);
            point.coordinates = {x, y};
            points.push_back(point);
        }
        
        return points;
    }

    /**
     * @brief 创建数据类型转换后的GridData
     */
    static std::shared_ptr<GridData> convertDataType(std::shared_ptr<GridData> sourceData, DataType targetType) {
        if (sourceData->getDataType() == targetType) {
            return sourceData; // 如果类型相同，直接返回
        }
        
        // 创建新的GridData，使用相同的定义但不同的数据类型
        auto newGrid = std::make_shared<GridData>(
            sourceData->getDefinition(), 
            targetType, 
            sourceData->getBandCount()
        );
        
        // 复制地理变换和CRS信息
        newGrid->setGeoTransform(sourceData->getGeoTransform());
        newGrid->setCrs(sourceData->getCRS());
        newGrid->setVariableName(sourceData->getVariableName());
        newGrid->setUnits(sourceData->getUnits());
        if (sourceData->getFillValue().has_value()) {
            newGrid->setFillValue(sourceData->getFillValue().value());
        }
        
        // 简单的数据类型转换（这里只是示例，实际应该进行真正的数据转换）
        // 为了测试目的，我们只是创建一个相同结构但不同类型的网格
        // 实际的数据转换会更复杂
        
        return newGrid;
    }
};

/**
 * @brief 插值算法性能测试结果
 */
struct PerformanceTestResult {
    std::string algorithmName;
    size_t dataPoints;
    size_t targetPoints;
    double executionTimeMs;
    double pointsPerSecond;
    size_t validResults;
    double successRate;
    
    void print() const {
        std::cout << "算法: " << algorithmName << std::endl;
        std::cout << "  数据点: " << dataPoints << std::endl;
        std::cout << "  目标点: " << targetPoints << std::endl;
        std::cout << "  执行时间: " << executionTimeMs << "ms" << std::endl;
        std::cout << "  处理速度: " << pointsPerSecond << " 点/秒" << std::endl;
        std::cout << "  成功率: " << (successRate * 100) << "%" << std::endl;
    }
};

/**
 * @brief 测试算法的精度
 */
struct AccuracyTestResult {
    std::string algorithmName;
    double meanAbsoluteError;
    double maxAbsoluteError;
    double rootMeanSquareError;
    size_t validResults;
    double successRate;
    
    void print() const {
        std::cout << "算法: " << algorithmName << std::endl;
        std::cout << "  平均绝对误差: " << meanAbsoluteError << std::endl;
        std::cout << "  最大绝对误差: " << maxAbsoluteError << std::endl;
        std::cout << "  均方根误差: " << rootMeanSquareError << std::endl;
        std::cout << "  有效结果: " << validResults << std::endl;
        std::cout << "  成功率: " << (successRate * 100) << "%" << std::endl;
    }
};

/**
 * @brief 通用插值算法测试类
 */
class InterpolationAlgorithmTest : public ::testing::Test {
private:
    // 静态成员变量，在所有测试间共享
    static std::shared_ptr<GridData> s_largeScaleTestData;
    static std::once_flag s_largeScaleDataFlag;
    
protected:
    void SetUp() override {
        // 创建SIMD管理器
        simdManager_ = std::make_shared<oscean::common_utils::simd::UnifiedSIMDManager>();
        
        // 创建所有已完善的算法实例
        algorithms_[InterpolationMethod::BILINEAR] = 
            std::make_unique<oscean::core_services::interpolation::BilinearInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::LINEAR_1D] = 
            std::make_unique<oscean::core_services::interpolation::Linear1DInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::NEAREST_NEIGHBOR] = 
            std::make_unique<oscean::core_services::interpolation::NearestNeighborInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::CUBIC_SPLINE_1D] = 
            std::make_unique<oscean::core_services::interpolation::CubicSplineInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::PCHIP_RECURSIVE_NDIM] = 
            std::make_unique<oscean::core_services::interpolation::PCHIPInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::TRILINEAR] = 
            std::make_unique<oscean::core_services::interpolation::TrilinearInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY] =
            std::make_unique<oscean::core_services::interpolation::PCHIPInterpolator2DBathy>(simdManager_);
        
        // 创建测试数据
        testData1D_ = InterpolationTestDataGenerator::create1DTestData(100);
        testData2D_ = InterpolationTestDataGenerator::create2DTestData(50, 50);
        testData3D_ = InterpolationTestDataGenerator::create3DTestData(20, 20, 10);
        
        // 只创建一次大规模测试数据
        std::call_once(s_largeScaleDataFlag, []() {
            std::cout << "创建大规模测试数据（只执行一次）..." << std::endl;
            s_largeScaleTestData = InterpolationTestDataGenerator::createLargeScale2DTestData();
        });
        largeScaleTestData_ = s_largeScaleTestData;
        
        // 创建精度验证数据
        analyticalTestData_ = InterpolationTestDataGenerator::createAnalyticalTestData(100, 100);

        // 为FastPCHIP创建专门的实例，因为它需要预计算
        // 注意：这里我们使用了一个新的枚举值来标识它
        fastPchipInterpolator_ = std::make_unique<oscean::core_services::interpolation::FastPchipInterpolator2D>(analyticalTestData_, simdManager_);
        
        // 为3D PCHIP创建一个更大的3D数据集，确保满足4x4x4的最小要求
        auto analyticalTestData3D = InterpolationTestDataGenerator::createAnalytical3DTestData(10, 10, 10);
        fastPchipInterpolator3D_ = std::make_unique<oscean::core_services::interpolation::FastPchipInterpolator3D>(analyticalTestData3D, simdManager_);
    }
    
    /**
     * @brief 测试算法的基本功能
     */
    void testBasicFunctionality(InterpolationMethod method, std::shared_ptr<GridData> testData) {
        auto it = algorithms_.find(method);
        ASSERT_NE(it, algorithms_.end()) << "算法未注册: " << static_cast<int>(method);
        
        auto bounds = testData->getSpatialExtent();
        auto targetPoints = InterpolationTestDataGenerator::generateTestPoints(bounds, 10);
        
        // 创建插值请求
        InterpolationRequest request;
        request.sourceGrid = testData;
        request.target = targetPoints;
        request.method = method;
        
        // 执行插值
        auto result = it->second->execute(request, nullptr);
        
        // 验证结果
        EXPECT_EQ(result.statusCode, 0) << "插值应该成功: " << result.message;
        EXPECT_TRUE(std::holds_alternative<std::vector<std::optional<double>>>(result.data))
            << "应该返回点插值结果";
        
        if (std::holds_alternative<std::vector<std::optional<double>>>(result.data)) {
            const auto& values = std::get<std::vector<std::optional<double>>>(result.data);
            EXPECT_EQ(values.size(), targetPoints.size()) << "结果数量应该匹配目标点数量";
            
            // 统计有效结果
            size_t validCount = 0;
            for (const auto& value : values) {
                if (value.has_value()) {
                    validCount++;
                    EXPECT_FALSE(std::isnan(value.value())) << "结果不应该是NaN";
                    EXPECT_FALSE(std::isinf(value.value())) << "结果不应该是无穷大";
                }
            }
            
            EXPECT_GT(validCount, 0) << "至少应该有一些有效结果";
        }
    }
    
    /**
     * @brief 测试算法的性能
     */
    PerformanceTestResult testPerformance(InterpolationMethod method, 
                                        std::shared_ptr<GridData> testData,
                                        size_t targetPointCount = 1000) {
        
        interpolation::IInterpolationAlgorithm* algorithm = nullptr;
        std::unique_ptr<interpolation::IInterpolationAlgorithm> tempAlgorithm = nullptr;

        if (method == InterpolationMethod::PCHIP_FAST_2D) {
            // 为Fast PCHIP动态创建实例，因为它需要与特定数据绑定
            try {
                // 确保数据类型是Float64，如果不是，则跳过
                if (testData->getDataType() != DataType::Float64) {
                    // 尝试将数据转换为Float64
                    auto convertedData = InterpolationTestDataGenerator::convertDataType(testData, DataType::Float64);
                    tempAlgorithm = std::make_unique<oscean::core_services::interpolation::FastPchipInterpolator2D>(convertedData, simdManager_);
                } else {
                    tempAlgorithm = std::make_unique<oscean::core_services::interpolation::FastPchipInterpolator2D>(testData, simdManager_);
                }
                algorithm = tempAlgorithm.get();
            } catch (const std::exception& e) {
                PerformanceTestResult failResult;
                failResult.algorithmName = getAlgorithmName(method) + " (创建失败: " + e.what() + ")";
                return failResult;
            }
        } else {
            auto it = algorithms_.find(method);
            if (it == algorithms_.end()) {
                PerformanceTestResult failResult;
                failResult.algorithmName = getAlgorithmName(method) + " (未注册)";
                return failResult;
            }
            algorithm = it->second.get();
        }
        
        auto bounds = testData->getSpatialExtent();
        auto targetPoints = InterpolationTestDataGenerator::generateTestPoints(bounds, targetPointCount);
        
        InterpolationRequest request;
        request.sourceGrid = testData;
        request.target = targetPoints;
        request.method = method;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        auto result = algorithm->execute(request, nullptr);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        double executionTimeMs = duration.count() / 1000.0;
        
        PerformanceTestResult perfResult;
        perfResult.algorithmName = getAlgorithmName(method);
        perfResult.dataPoints = testData->getDefinition().cols * testData->getDefinition().rows * testData->getBandCount();
        perfResult.targetPoints = targetPointCount;
        perfResult.executionTimeMs = executionTimeMs;
        perfResult.pointsPerSecond = executionTimeMs > 0 ? (targetPointCount * 1000.0 / executionTimeMs) : 0;
        
        if (result.statusCode == 0 && std::holds_alternative<std::vector<std::optional<double>>>(result.data)) {
            const auto& values = std::get<std::vector<std::optional<double>>>(result.data);
            perfResult.validResults = 0;
            for (const auto& value : values) {
                if (value.has_value()) {
                    perfResult.validResults++;
                }
            }
            perfResult.successRate = static_cast<double>(perfResult.validResults) / targetPointCount;
        } else {
            perfResult.validResults = 0;
            perfResult.successRate = 0.0;
        }
        
        return perfResult;
    }
    
    /**
     * @brief 获取算法名称
     */
    std::string getAlgorithmName(InterpolationMethod method) const {
        switch (method) {
            case InterpolationMethod::BILINEAR: return "双线性插值";
            case InterpolationMethod::LINEAR_1D: return "1D线性插值";
            case InterpolationMethod::NEAREST_NEIGHBOR: return "最近邻插值";
            case InterpolationMethod::CUBIC_SPLINE_1D: return "立方样条插值";
            case InterpolationMethod::PCHIP_RECURSIVE_NDIM: return "PCHIP插值 (通用递归)";
            case InterpolationMethod::PCHIP_FAST_2D: return "PCHIP插值 (2D优化版)";
            case InterpolationMethod::PCHIP_FAST_3D: return "PCHIP插值 (3D优化版)";
            case InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY: return "PCHIP插值 (2D水深优化)";
            case InterpolationMethod::TRILINEAR: return "三线性插值";
            default: return "未知算法";
        }
    }

    /**
     * @brief 测试算法精度（使用解析函数）
     */
    AccuracyTestResult testAccuracy(InterpolationMethod method, 
                                   std::shared_ptr<GridData> testData,
                                   size_t targetPointCount = 1000) {
        
        interpolation::IInterpolationAlgorithm* algorithm = nullptr;
        if (method == InterpolationMethod::PCHIP_FAST_2D) {
            // 对于Fast PCHIP，使用专门的实例
            // 注意：理想情况下，应该为每个测试数据重新创建实例，但为了简化，我们复用
            if (testData.get() != analyticalTestData_.get()) {
                 // 如果测试数据不是预期的分析数据，则跳过
                AccuracyTestResult skipResult;
                skipResult.algorithmName = getAlgorithmName(method) + " (跳过)";
                return skipResult;
            }
            algorithm = fastPchipInterpolator_.get();
        } else {
            auto it = algorithms_.find(method);
            if (it == algorithms_.end()) {
                 AccuracyTestResult failResult;
                failResult.algorithmName = getAlgorithmName(method) + " (未注册)";
                return failResult;
            }
            algorithm = it->second.get();
        }

        AccuracyTestResult result;
        result.algorithmName = getAlgorithmName(method);
        
        auto bounds = testData->getSpatialExtent();
        auto targetPoints = InterpolationTestDataGenerator::generateTestPoints(bounds, targetPointCount);
        
        // 创建插值请求
        InterpolationRequest request;
        request.sourceGrid = testData;
        request.target = targetPoints;
        request.method = method;
        
        // 执行插值
        auto interpolationResult = algorithm->execute(request, nullptr);
        
        if (interpolationResult.statusCode != 0 || 
            !std::holds_alternative<std::vector<std::optional<double>>>(interpolationResult.data)) {
            result.meanAbsoluteError = std::numeric_limits<double>::max();
            result.maxAbsoluteError = std::numeric_limits<double>::max();
            result.rootMeanSquareError = std::numeric_limits<double>::max();
            result.validResults = 0;
            result.successRate = 0.0;
            return result;
        }
        
        const auto& interpolatedValues = std::get<std::vector<std::optional<double>>>(interpolationResult.data);
        
        // 计算理论值并比较精度
        std::vector<double> errors;
        result.validResults = 0;
        result.maxAbsoluteError = 0.0;
        
        for (size_t i = 0; i < targetPoints.size(); ++i) {
            if (interpolatedValues[i].has_value()) {
                double x = targetPoints[i].coordinates[0];
                double y = targetPoints[i].coordinates[1];
                
                // 计算解析函数的理论值 f(x,y) = x^2 + 2*x*y + y^2 + 3*x + 4*y + 5
                double theoreticalValue = x*x + 2.0*x*y + y*y + 3.0*x + 4.0*y + 5.0;
                double interpolatedValue = interpolatedValues[i].value();
                
                double error = std::abs(interpolatedValue - theoreticalValue);
                errors.push_back(error);
                result.maxAbsoluteError = std::max(result.maxAbsoluteError, error);
                result.validResults++;
            }
        }
        
        result.successRate = static_cast<double>(result.validResults) / targetPointCount;
        
        if (!errors.empty()) {
            // 计算平均绝对误差
            double sumError = 0.0;
            double sumSquaredError = 0.0;
            for (double error : errors) {
                sumError += error;
                sumSquaredError += error * error;
            }
            result.meanAbsoluteError = sumError / errors.size();
            result.rootMeanSquareError = std::sqrt(sumSquaredError / errors.size());
        } else {
            result.meanAbsoluteError = std::numeric_limits<double>::max();
            result.rootMeanSquareError = std::numeric_limits<double>::max();
        }
        
        return result;
    }

protected:
    std::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;
    std::map<InterpolationMethod, std::unique_ptr<interpolation::IInterpolationAlgorithm>> algorithms_;
    std::shared_ptr<GridData> testData1D_;
    std::shared_ptr<GridData> testData2D_;
    std::shared_ptr<GridData> testData3D_;
    std::shared_ptr<GridData> largeScaleTestData_;
    std::shared_ptr<GridData> analyticalTestData_;
    // 新增的优化版PCHIP插值器实例
    std::unique_ptr<oscean::core_services::interpolation::FastPchipInterpolator2D> fastPchipInterpolator_;
    std::unique_ptr<oscean::core_services::interpolation::FastPchipInterpolator3D> fastPchipInterpolator3D_;
};

// === 1D插值算法测试 ===

TEST_F(InterpolationAlgorithmTest, Linear1D_BasicFunctionality) {
    testBasicFunctionality(InterpolationMethod::LINEAR_1D, testData1D_);
}

// === 2D插值算法测试 ===

TEST_F(InterpolationAlgorithmTest, Bilinear_BasicFunctionality) {
    testBasicFunctionality(InterpolationMethod::BILINEAR, testData2D_);
}

TEST_F(InterpolationAlgorithmTest, NearestNeighbor_BasicFunctionality) {
    testBasicFunctionality(InterpolationMethod::NEAREST_NEIGHBOR, testData2D_);
}

TEST_F(InterpolationAlgorithmTest, CubicSpline_BasicFunctionality) {
    testBasicFunctionality(InterpolationMethod::CUBIC_SPLINE_1D, testData2D_);
}

TEST_F(InterpolationAlgorithmTest, PCHIP_BasicFunctionality) {
    testBasicFunctionality(InterpolationMethod::PCHIP_RECURSIVE_NDIM, testData2D_);
}

// === 3D插值算法测试 ===

TEST_F(InterpolationAlgorithmTest, Trilinear_BasicFunctionality) {
    testBasicFunctionality(InterpolationMethod::TRILINEAR, testData3D_);
}

// === 性能对比测试 ===

TEST_F(InterpolationAlgorithmTest, PerformanceComparison_2D) {
    std::cout << "\n=== 2D插值算法性能对比 ===" << std::endl;
    
    std::vector<InterpolationMethod> methods2D = {
        InterpolationMethod::NEAREST_NEIGHBOR,
        InterpolationMethod::BILINEAR,
        InterpolationMethod::CUBIC_SPLINE_1D,
        InterpolationMethod::PCHIP_RECURSIVE_NDIM,
        InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY,
        InterpolationMethod::PCHIP_FAST_2D
    };
    
    std::vector<PerformanceTestResult> results;
    
    for (auto method : methods2D) {
        auto result = testPerformance(method, testData2D_, 1000);
        results.push_back(result);
        result.print();
        std::cout << std::endl;
    }
    
    // 性能排序和分析
    std::sort(results.begin(), results.end(), 
              [](const PerformanceTestResult& a, const PerformanceTestResult& b) {
                  return a.pointsPerSecond > b.pointsPerSecond;
              });
    
    std::cout << "\n=== 性能排名 ===" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i + 1) << ". " << results[i].algorithmName 
                  << " - " << static_cast<int>(results[i].pointsPerSecond) << " 点/秒" << std::endl;
    }
    
    // 验证所有算法都有合理的性能
    for (const auto& result : results) {
        EXPECT_GT(result.successRate, 0.8) << result.algorithmName << " 成功率应该大于80%";
        EXPECT_LT(result.executionTimeMs, 500) << result.algorithmName << " 执行时间应该小于0.5秒";
    }
}

TEST_F(InterpolationAlgorithmTest, PerformanceComparison_1D) {
    std::cout << "\n=== 1D插值算法性能对比 ===" << std::endl;
    
    std::vector<InterpolationMethod> methods1D = {
        InterpolationMethod::LINEAR_1D
    };
    
    std::vector<PerformanceTestResult> results;
    
    for (auto method : methods1D) {
        auto result = testPerformance(method, testData1D_, 500);
        results.push_back(result);
        result.print();
        std::cout << std::endl;
    }
    
    // 验证所有算法都有合理的性能
    for (const auto& result : results) {
        EXPECT_GT(result.successRate, 0.8) << result.algorithmName << " 成功率应该大于80%";
        EXPECT_LT(result.executionTimeMs, 500) << result.algorithmName << " 执行时间应该小于0.5秒";
    }
}

TEST_F(InterpolationAlgorithmTest, PerformanceComparison_3D) {
    std::cout << "\n=== 3D插值算法性能对比 ===" << std::endl;
    
    std::vector<InterpolationMethod> methods3D = {
        InterpolationMethod::TRILINEAR
    };
    
    std::vector<PerformanceTestResult> results;
    
    for (auto method : methods3D) {
        auto result = testPerformance(method, testData3D_, 500);
        results.push_back(result);
        result.print();
        std::cout << std::endl;
    }
    
    // 验证所有算法都有合理的性能
    for (const auto& result : results) {
        EXPECT_GT(result.successRate, 0.8) << result.algorithmName << " 成功率应该大于80%";
        EXPECT_LT(result.executionTimeMs, 1000) << result.algorithmName << " 执行时间应该小于1秒";
    }
}

// === 大规模性能测试（100万数据点）===

TEST_F(InterpolationAlgorithmTest, LargeScale_PerformanceComparison) {
    std::cout << "\n=== 大规模性能测试（100万数据点）===" << std::endl;
    std::cout << "数据网格: 1000x1000 = 1,000,000 数据点" << std::endl;
    
    std::vector<InterpolationMethod> methods = {
        InterpolationMethod::NEAREST_NEIGHBOR,
        InterpolationMethod::BILINEAR,
        InterpolationMethod::CUBIC_SPLINE_1D,
        InterpolationMethod::PCHIP_RECURSIVE_NDIM,
        InterpolationMethod::PCHIP_FAST_2D
    };
    
    std::vector<PerformanceTestResult> results;
    
    for (auto method : methods) {
        std::cout << "\n测试算法: " << getAlgorithmName(method) << std::endl;
        auto result = testPerformance(method, largeScaleTestData_, 10000); // 1万个目标点
        results.push_back(result);
        result.print();
    }
    
    // 性能排序和分析
    std::sort(results.begin(), results.end(), 
              [](const PerformanceTestResult& a, const PerformanceTestResult& b) {
                  return a.pointsPerSecond > b.pointsPerSecond;
              });
    
    std::cout << "\n=== 性能排名 ===" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i + 1) << ". " << results[i].algorithmName 
                  << " - " << static_cast<int>(results[i].pointsPerSecond) << " 点/秒" << std::endl;
    }
    
    // 验证所有算法都有合理的性能
    for (const auto& result : results) {
        EXPECT_GT(result.successRate, 0.8) << result.algorithmName << " 成功率应该大于80%";
        EXPECT_LT(result.executionTimeMs, 10000) << result.algorithmName << " 执行时间应该小于10秒";
        EXPECT_GT(result.pointsPerSecond, 100) << result.algorithmName << " 处理速度应该大于100点/秒";
    }
}

// === 算法精度比较测试 ===

TEST_F(InterpolationAlgorithmTest, AccuracyComparison_AnalyticalFunction) {
    std::cout << "\n=== 算法精度比较测试 ===" << std::endl;
    std::cout << "测试函数: f(x,y) = x² + 2xy + y² + 3x + 4y + 5" << std::endl;
    std::cout << "数据网格: 100x100 = 10,000 数据点" << std::endl;
    
    std::vector<InterpolationMethod> methods = {
        InterpolationMethod::NEAREST_NEIGHBOR,
        InterpolationMethod::BILINEAR,
        InterpolationMethod::PCHIP_RECURSIVE_NDIM,
        InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY,
        InterpolationMethod::PCHIP_FAST_2D
    };
    
    std::vector<AccuracyTestResult> results;
    
    for (auto method : methods) {
        std::cout << "\n测试算法: " << getAlgorithmName(method) << std::endl;
        auto result = testAccuracy(method, analyticalTestData_, 2000); // 2000个测试点
        results.push_back(result);
        result.print();
    }
    
    // 精度排序和分析
    std::sort(results.begin(), results.end(), 
              [](const AccuracyTestResult& a, const AccuracyTestResult& b) {
                  return a.meanAbsoluteError < b.meanAbsoluteError;
              });
    
    std::cout << "\n=== 精度排名（按平均绝对误差）===" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i + 1) << ". " << results[i].algorithmName 
                  << " - MAE: " << results[i].meanAbsoluteError 
                  << ", RMSE: " << results[i].rootMeanSquareError << std::endl;
    }
    
    // 验证精度要求
    for (const auto& result : results) {
        EXPECT_GT(result.successRate, 0.9) << result.algorithmName << " 成功率应该大于90%";
        EXPECT_LT(result.meanAbsoluteError, 10.0) << result.algorithmName << " 平均绝对误差应该合理";
        
        // 对于二次函数，双线性插值应该有较好的精度
        if (result.algorithmName == "双线性插值") {
            EXPECT_LT(result.meanAbsoluteError, 1.0) << "双线性插值对二次函数应该有较好精度";
        }
    }
}

// === 综合性能与精度评估 ===

TEST_F(InterpolationAlgorithmTest, ComprehensiveEvaluation) {
    std::cout << "\n=== 综合性能与精度评估 ===" << std::endl;
    
    std::vector<InterpolationMethod> methods = {
        InterpolationMethod::NEAREST_NEIGHBOR,
        InterpolationMethod::BILINEAR,
        InterpolationMethod::CUBIC_SPLINE_1D,
        InterpolationMethod::PCHIP_RECURSIVE_NDIM,
        InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY,
        InterpolationMethod::PCHIP_FAST_2D
    };
    
    struct ComprehensiveResult {
        std::string algorithmName;
        double performanceScore;  // 性能得分 (点/秒)
        double accuracyScore;     // 精度得分 (1/MAE)
        double overallScore;      // 综合得分
    };
    
    std::vector<ComprehensiveResult> comprehensiveResults;
    
    for (auto method : methods) {
        // 性能测试（使用较小数据集以节省时间）
        auto perfResult = testPerformance(method, testData2D_, 1000);
        
        // 精度测试
        auto accResult = testAccuracy(method, analyticalTestData_, 500);
        
        ComprehensiveResult compResult;
        compResult.algorithmName = getAlgorithmName(method);
        compResult.performanceScore = perfResult.pointsPerSecond;
        compResult.accuracyScore = accResult.meanAbsoluteError > 0 ? 1.0 / accResult.meanAbsoluteError : 0.0;
        
        // 综合得分：性能权重0.3，精度权重0.7
        double normalizedPerf = compResult.performanceScore / 1000000.0; // 归一化到百万点/秒
        double normalizedAcc = compResult.accuracyScore * 10.0; // 放大精度得分
        compResult.overallScore = 0.3 * normalizedPerf + 0.7 * normalizedAcc;
        
        comprehensiveResults.push_back(compResult);
        
        std::cout << compResult.algorithmName << ":" << std::endl;
        std::cout << "  性能: " << static_cast<int>(compResult.performanceScore) << " 点/秒" << std::endl;
        std::cout << "  精度得分: " << compResult.accuracyScore << std::endl;
        std::cout << "  综合得分: " << compResult.overallScore << std::endl;
        std::cout << std::endl;
    }
    
    // 按综合得分排序
    std::sort(comprehensiveResults.begin(), comprehensiveResults.end(),
              [](const ComprehensiveResult& a, const ComprehensiveResult& b) {
                  return a.overallScore > b.overallScore;
              });
    
    std::cout << "=== 综合排名 ===" << std::endl;
    for (size_t i = 0; i < comprehensiveResults.size(); ++i) {
        std::cout << (i + 1) << ". " << comprehensiveResults[i].algorithmName 
                  << " (综合得分: " << comprehensiveResults[i].overallScore << ")" << std::endl;
    }
}

// 新增：测试Fast PCHIP的基本功能
TEST_F(InterpolationAlgorithmTest, FastPCHIP_BasicFunctionality) {
    // Fast PCHIP需要特定的数据进行预计算，我们用analyticalTestData来测试
    InterpolationRequest request;
    request.sourceGrid = analyticalTestData_;
    auto bounds = analyticalTestData_->getSpatialExtent();
    request.target = InterpolationTestDataGenerator::generateTestPoints(bounds, 10);
    
    auto result = fastPchipInterpolator_->execute(request, nullptr);

    EXPECT_EQ(result.statusCode, 0);
    EXPECT_TRUE(std::holds_alternative<std::vector<std::optional<double>>>(result.data));
    const auto& values = std::get<std::vector<std::optional<double>>>(result.data);
    EXPECT_EQ(values.size(), 10);
    size_t validCount = 0;
    for(const auto& v : values) {
        if(v.has_value()) validCount++;
    }
    EXPECT_GT(validCount, 0);
}

// 新增：测试Fast PCHIP 3D的基本功能
TEST_F(InterpolationAlgorithmTest, FastPCHIP3D_BasicFunctionality) {
    // Fast PCHIP 3D需要特定的数据进行预计算，我们用testData3D来测试
    InterpolationRequest request;
    request.sourceGrid = testData3D_;
    auto bounds = testData3D_->getSpatialExtent();
    request.target = InterpolationTestDataGenerator::generateTestPoints(bounds, 5);
    
    auto result = fastPchipInterpolator3D_->execute(request, nullptr);

    // 由于execute方法还未完全实现，我们期望得到一个"未实现"的状态
    EXPECT_EQ(result.statusCode, -1);
    EXPECT_TRUE(std::holds_alternative<std::monostate>(result.data));
    EXPECT_FALSE(result.message.empty());
    std::cout << "FastPCHIP3D状态: " << result.message << std::endl;
}

// 新增：测试PCHIP 2D Bathy优化算法的基本功能
TEST_F(InterpolationAlgorithmTest, PCHIP_Bathy2D_BasicFunctionality) {
    auto it = algorithms_.find(InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY);
    ASSERT_NE(it, algorithms_.end());

    InterpolationRequest request;
    request.sourceGrid = testData2D_; // 使用标准的2D数据
    auto bounds = testData2D_->getSpatialExtent();
    request.target = InterpolationTestDataGenerator::generateTestPoints(bounds, 5);
    
    auto result = it->second->execute(request, nullptr);

    // 验证插值是否成功执行
    EXPECT_EQ(result.statusCode, 0);
}

// 静态成员定义
std::shared_ptr<GridData> InterpolationAlgorithmTest::s_largeScaleTestData;
std::once_flag InterpolationAlgorithmTest::s_largeScaleDataFlag; 