#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iomanip>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include "../src/impl/algorithms/bilinear_interpolator.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services::interpolation;
using namespace oscean::core_services;

class BilinearSIMDPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试网格数据
        createTestGrid(1000, 1000);
        
        // 生成随机测试点
        generateRandomPoints(10000);
    }
    
    void createTestGrid(size_t width, size_t height) {
        GridDefinition def;
        def.cols = width;
        def.rows = height;
        
        // 创建Float64和Float32两种类型的网格
        gridDouble_ = std::make_shared<GridData>(def, DataType::Float64, 1);
        gridFloat_ = std::make_shared<GridData>(def, DataType::Float32, 1);
        
        // 填充测试数据（简单的二维函数）
        double* dataDouble = const_cast<double*>(
            static_cast<const double*>(gridDouble_->getDataPtr()));
        float* dataFloat = const_cast<float*>(
            static_cast<const float*>(gridFloat_->getDataPtr()));
            
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                double value = std::sin(x * 0.01) * std::cos(y * 0.01);
                dataDouble[idx] = value;
                dataFloat[idx] = static_cast<float>(value);
            }
        }
        
        // 设置地理变换
        std::vector<double> geoTransform = {
            0.0,    // 左上角X
            0.1,    // X分辨率
            0.0,    // 旋转
            100.0,  // 左上角Y
            0.0,    // 旋转
            -0.1    // Y分辨率（负值）
        };
        gridDouble_->setGeoTransform(geoTransform);
        gridFloat_->setGeoTransform(geoTransform);
    }
    
    void generateRandomPoints(size_t count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> xDist(0.0, 99.0);
        std::uniform_real_distribution<> yDist(1.0, 99.0);
        
        testPoints_.clear();
        testPoints_.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            TargetPoint pt;
            pt.coordinates = {xDist(gen), yDist(gen)};
            testPoints_.push_back(pt);
        }
    }
    
    template<typename Func>
    double measureTime(const std::string& name, Func&& func, int iterations = 10) {
        // 预热
        func();
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        std::cout << std::setw(30) << std::left << name 
                  << ": " << std::fixed << std::setprecision(3) 
                  << ms << " ms" << std::endl;
        return ms;
    }
    
    std::shared_ptr<GridData> gridDouble_;
    std::shared_ptr<GridData> gridFloat_;
    std::vector<TargetPoint> testPoints_;
};

TEST_F(BilinearSIMDPerformanceTest, CompareScalarVsSIMD) {
    std::cout << "\n=== 双线性插值性能对比测试 ===" << std::endl;
    std::cout << "网格大小: 1000x1000" << std::endl;
    std::cout << "测试点数: " << testPoints_.size() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // 创建插值器
    BilinearInterpolator interpolator;
    
    // 测试Float64数据
    std::cout << "\n[Float64 数据类型]" << std::endl;
    
    // 标量版本
    double scalarTimeDouble = measureTime("标量实现", [&]() {
        auto results = interpolator.interpolateAtPoints(*gridDouble_, testPoints_);
    });
    
    // SIMD版本
    double simdTimeDouble = measureTime("SIMD优化实现", [&]() {
        auto results = interpolator.interpolateAtPointsSIMD(*gridDouble_, testPoints_);
    });
    
    double speedupDouble = scalarTimeDouble / simdTimeDouble;
    std::cout << "加速比: " << std::fixed << std::setprecision(2) 
              << speedupDouble << "x" << std::endl;
    
    // 测试Float32数据
    std::cout << "\n[Float32 数据类型]" << std::endl;
    
    // 标量版本
    double scalarTimeFloat = measureTime("标量实现", [&]() {
        auto results = interpolator.interpolateAtPoints(*gridFloat_, testPoints_);
    });
    
    // SIMD版本
    double simdTimeFloat = measureTime("SIMD优化实现", [&]() {
        auto results = interpolator.interpolateAtPointsSIMD(*gridFloat_, testPoints_);
    });
    
    double speedupFloat = scalarTimeFloat / simdTimeFloat;
    std::cout << "加速比: " << std::fixed << std::setprecision(2) 
              << speedupFloat << "x" << std::endl;
    
    // 验证加速效果
    EXPECT_GT(speedupDouble, 2.0) << "Float64 SIMD加速比应大于2倍";
    EXPECT_GT(speedupFloat, 3.0) << "Float32 SIMD加速比应大于3倍";
}

TEST_F(BilinearSIMDPerformanceTest, BatchSizeImpact) {
    std::cout << "\n=== 批量大小对性能的影响 ===" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    BilinearInterpolator interpolator;
    std::vector<size_t> batchSizes = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 10000};
    
    for (size_t batchSize : batchSizes) {
        // 生成指定大小的点集
        generateRandomPoints(batchSize);
        
        std::cout << "\n批量大小: " << std::setw(6) << batchSize << std::endl;
        
        // 标量版本
        double scalarTime = measureTime("  标量", [&]() {
            auto results = interpolator.interpolateAtPoints(*gridDouble_, testPoints_);
        }, 100);
        
        // SIMD版本
        double simdTime = measureTime("  SIMD", [&]() {
            auto results = interpolator.interpolateAtPointsSIMD(*gridDouble_, testPoints_);
        }, 100);
        
        double speedup = scalarTime / simdTime;
        std::cout << "  加速比: " << std::fixed << std::setprecision(2) 
                  << speedup << "x" << std::endl;
    }
}

TEST_F(BilinearSIMDPerformanceTest, AccuracyVerification) {
    std::cout << "\n=== SIMD精度验证 ===" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    BilinearInterpolator interpolator;
    
    // 生成少量测试点进行精度验证
    generateRandomPoints(100);
    
    // 获取标量和SIMD结果
    auto scalarResults = interpolator.interpolateAtPoints(*gridDouble_, testPoints_);
    auto simdResults = interpolator.interpolateAtPointsSIMD(*gridDouble_, testPoints_);
    
    ASSERT_EQ(scalarResults.size(), simdResults.size());
    
    double maxError = 0.0;
    double avgError = 0.0;
    int validCount = 0;
    
    for (size_t i = 0; i < scalarResults.size(); ++i) {
        if (scalarResults[i].has_value() && simdResults[i].has_value()) {
            double error = std::abs(scalarResults[i].value() - simdResults[i].value());
            maxError = std::max(maxError, error);
            avgError += error;
            validCount++;
        } else {
            // 确保两个版本对无效点的处理一致
            EXPECT_EQ(scalarResults[i].has_value(), simdResults[i].has_value());
        }
    }
    
    if (validCount > 0) {
        avgError /= validCount;
    }
    
    std::cout << "最大误差: " << std::scientific << maxError << std::endl;
    std::cout << "平均误差: " << std::scientific << avgError << std::endl;
    std::cout << "有效点数: " << validCount << "/" << testPoints_.size() << std::endl;
    
    // 验证精度在可接受范围内
    EXPECT_LT(maxError, 1e-12) << "SIMD和标量版本的最大误差应小于1e-12";
    EXPECT_LT(avgError, 1e-13) << "SIMD和标量版本的平均误差应小于1e-13";
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // 检测CPU能力
    std::cout << "CPU SIMD能力检测:" << std::endl;
    
    #ifdef _MSC_VER
    // MSVC方式检测
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    
    bool hasSSE42 = false;
    bool hasAVX2 = false;
    bool hasAVX512F = false;
    
    if (nIds >= 1) {
        __cpuid(cpuInfo, 1);
        hasSSE42 = (cpuInfo[2] & (1 << 20)) != 0;
    }
    
    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;
        hasAVX512F = (cpuInfo[1] & (1 << 16)) != 0;
    }
    
    std::cout << "  SSE4.2: " << (hasSSE42 ? "支持" : "不支持") << std::endl;
    std::cout << "  AVX2: " << (hasAVX2 ? "支持" : "不支持") << std::endl;
    std::cout << "  AVX-512: " << (hasAVX512F ? "支持" : "不支持") << std::endl;
    #else
    // GCC/Clang方式
    #ifdef __SSE4_2__
    std::cout << "  SSE4.2: " << (__builtin_cpu_supports("sse4.2") ? "支持" : "不支持") << std::endl;
    #endif
    #ifdef __AVX2__
    std::cout << "  AVX2: " << (__builtin_cpu_supports("avx2") ? "支持" : "不支持") << std::endl;
    #endif
    #ifdef __AVX512F__
    std::cout << "  AVX-512: " << (__builtin_cpu_supports("avx512f") ? "支持" : "不支持") << std::endl;
    #endif
    #endif
    
    return RUN_ALL_TESTS();
} 