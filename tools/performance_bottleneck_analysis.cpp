#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <thread>
#include <future>
#include <vector>
#include <iostream>
#include <filesystem>

#include "core_services_impl/data_access_service/src/readers/core/impl/gdal_unified_reader.h"
#include "core_services_impl/crs_service/src/impl/crs_service_gdal_proj.h"
#include "common_utils/utilities/logging_utils.h"

using namespace oscean::core_services::data_access::readers::impl;
using namespace oscean::core_services::data_access::api;

class PerformanceBottleneckAnalysis : public ::testing::Test {
protected:
    void SetUp() override {
        LOG_INFO("=== GDAL性能瓶颈分析测试环境初始化 ===");
        
        // 使用真实的GEBCO测试数据
        gdalTestFile_ = "E:\\Ocean_data\\gebco_2024_tid_geotiff\\gebco_2024_tid_n0.0_s-90.0_w-180.0_e-90.0.tif";
        
        // 验证测试文件存在
        if (!std::filesystem::exists(gdalTestFile_)) {
            GTEST_SKIP() << "GEBCO测试文件不存在: " << gdalTestFile_;
        }
        
        LOG_INFO("使用GEBCO测试数据文件: {}", gdalTestFile_);
        LOG_INFO("GDAL性能瓶颈分析测试环境初始化完成");
    }
    
    void TearDown() override {
        LOG_INFO("GDAL性能瓶颈分析测试环境清理完成");
    }

protected:
    std::string gdalTestFile_;
};

TEST_F(PerformanceBottleneckAnalysis, CPUUsageAnalysis) {
    LOG_INFO("=== CPU使用率分析 ===");
    
    auto reader = std::make_unique<GdalUnifiedReader>(gdalTestFile_, nullptr, GdalReaderType::RASTER);
    
    auto openFuture = reader->openAsync(gdalTestFile_);
    ASSERT_TRUE(openFuture.get());
    
    ReadGridDataRequest request;
    request.variableName = "Band_1";
    
    // 多次测试CPU使用模式
    std::vector<double> processingTimes;
    constexpr int iterations = 5;
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto dataFuture = reader->readGridDataAsync(request);
        auto result = dataFuture.get();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        processingTimes.push_back(duration);
        
        ASSERT_TRUE(result->requestStatus.success);
        LOG_INFO("第{}次读取耗时: {}ms", i+1, duration);
    }
    
    // 分析CPU使用模式
    double avgTime = 0;
    for (auto time : processingTimes) avgTime += time;
    avgTime /= iterations;
    
    double variance = 0;
    for (auto time : processingTimes) {
        variance += (time - avgTime) * (time - avgTime);
    }
    variance /= iterations;
    
    LOG_INFO("CPU使用分析结果:");
    LOG_INFO("  平均处理时间: {:.2f}ms", avgTime);
    LOG_INFO("  时间方差: {:.2f}", variance);
    LOG_INFO("  性能稳定性: {}", variance < 1000 ? "良好" : "需改进");
    
    // 评估是否CPU密集
    if (avgTime > 200) {
        LOG_INFO("  优化建议: CPU密集型操作，考虑SIMD优化");
    }
}

TEST_F(PerformanceBottleneckAnalysis, MemoryBandwidthAnalysis) {
    LOG_INFO("=== 内存带宽利用率分析 ===");
    
    auto reader = std::make_unique<GdalUnifiedReader>(gdalTestFile_, nullptr, GdalReaderType::RASTER);
    
    auto openFuture = reader->openAsync(gdalTestFile_);
    ASSERT_TRUE(openFuture.get());
    
    ReadGridDataRequest request;
    request.variableName = "Band_1";
    
    auto start = std::chrono::high_resolution_clock::now();
    auto dataFuture = reader->readGridDataAsync(request);
    auto result = dataFuture.get();
    auto end = std::chrono::high_resolution_clock::now();
    
    ASSERT_TRUE(result->requestStatus.success);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // 计算内存带宽利用率
    size_t dataSize = result->grid->data.size();
    double mbps = (static_cast<double>(dataSize) / (1024.0 * 1024.0)) / (duration / 1000000.0);
    
    LOG_INFO("内存带宽分析结果:");
    LOG_INFO("  数据大小: {:.2f} MB", static_cast<double>(dataSize) / (1024.0 * 1024.0));
    LOG_INFO("  处理时间: {:.2f}ms", duration / 1000.0);
    LOG_INFO("  内存带宽: {:.2f} MB/s", mbps);
    
    // 典型DDR4内存带宽约25-50 GB/s，实际应用中能达到5-15 GB/s算不错
    if (mbps < 2000) {
        LOG_INFO("  优化建议: 内存带宽利用率较低，考虑内存对齐和缓存优化");
    } else if (mbps > 5000) {
        LOG_INFO("  评估: 内存带宽利用率良好");
    }
}

TEST_F(PerformanceBottleneckAnalysis, IOBottleneckAnalysis) {
    LOG_INFO("=== I/O瓶颈分析 ===");
    
    auto reader = std::make_unique<GdalUnifiedReader>(gdalTestFile_, nullptr, GdalReaderType::RASTER);
    
    // 测试文件打开时间
    auto openStart = std::chrono::high_resolution_clock::now();
    auto openFuture = reader->openAsync(gdalTestFile_);
    ASSERT_TRUE(openFuture.get());
    auto openEnd = std::chrono::high_resolution_clock::now();
    auto openTime = std::chrono::duration_cast<std::chrono::milliseconds>(openEnd - openStart).count();
    
    ReadGridDataRequest request;
    request.variableName = "Band_1";
    
    // 测试纯数据读取时间
    auto readStart = std::chrono::high_resolution_clock::now();
    auto dataFuture = reader->readGridDataAsync(request);
    auto result = dataFuture.get();
    auto readEnd = std::chrono::high_resolution_clock::now();
    auto readTime = std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - readStart).count();
    
    ASSERT_TRUE(result->requestStatus.success);
    
    // 获取文件大小
    auto fileSize = std::filesystem::file_size(gdalTestFile_);
    double fileSizeMB = static_cast<double>(fileSize) / (1024.0 * 1024.0);
    
    LOG_INFO("I/O性能分析结果:");
    LOG_INFO("  文件大小: {:.2f} MB", fileSizeMB);
    LOG_INFO("  文件打开时间: {}ms", openTime);
    LOG_INFO("  数据读取时间: {}ms", readTime);
    LOG_INFO("  I/O速度: {:.2f} MB/s", fileSizeMB / (readTime / 1000.0));
    
    if (openTime > 50) {
        LOG_INFO("  优化建议: 文件打开较慢，考虑连接池或预热策略");
    }
    
    // 现代SSD顺序读取应该在500+ MB/s
    double ioSpeed = fileSizeMB / (readTime / 1000.0);
    if (ioSpeed < 200) {
        LOG_INFO("  优化建议: I/O速度较慢，考虑异步I/O或预读优化");
    }
}

TEST_F(PerformanceBottleneckAnalysis, ParallelProcessingPotential) {
    LOG_INFO("=== 并行处理潜力分析 ===");
    
    auto reader = std::make_unique<GdalUnifiedReader>(gdalTestFile_, nullptr, GdalReaderType::RASTER);
    auto openFuture = reader->openAsync(gdalTestFile_);
    ASSERT_TRUE(openFuture.get());
    
    // 测试不同大小子集的处理时间
    std::vector<std::pair<std::string, BoundingBox>> testCases = {
        {"小块(1/16)", {-180, -90, -135, -45}},  // 1/16大小
        {"中块(1/4)", {-180, -90, -90, 0}},      // 1/4大小  
        {"大块(1/2)", {-180, -90, 0, 90}},       // 1/2大小
    };
    
    for (const auto& testCase : testCases) {
        ReadGridDataRequest request;
        request.variableName = "Band_1";
        request.outputBounds = testCase.second;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto dataFuture = reader->readGridDataAsync(request);
        auto result = dataFuture.get();
        auto end = std::chrono::high_resolution_clock::now();
        
        ASSERT_TRUE(result->requestStatus.success);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        size_t pixels = result->grid->getWidth() * result->grid->getHeight();
        double pixelsPerMs = static_cast<double>(pixels) / duration;
        
        LOG_INFO("{}并行处理分析: {}x{} = {} 像素, {}ms, {:.0f} 像素/ms", 
                testCase.first, result->grid->getWidth(), result->grid->getHeight(), 
                pixels, duration, pixelsPerMs);
    }
    
    LOG_INFO("  优化建议: 如果小块处理效率高，考虑实现瓦片并行处理");
}

TEST_F(PerformanceBottleneckAnalysis, MemoryAccessPatternAnalysis) {
    LOG_INFO("=== 内存访问模式分析 ===");
    
    auto reader = std::make_unique<GdalUnifiedReader>(gdalTestFile_, nullptr, GdalReaderType::RASTER);
    auto openFuture = reader->openAsync(gdalTestFile_);
    ASSERT_TRUE(openFuture.get());
    
    ReadGridDataRequest request;
    request.variableName = "Band_1";
    
    // 测试不同的空间访问模式
    std::vector<std::pair<std::string, BoundingBox>> accessPatterns = {
        {"顺序访问(左上)", {-180, 45, -90, 90}},     // 左上角
        {"随机访问(中间)", {-45, -22.5, 45, 22.5}}, // 中间区域
        {"条带访问(水平)", {-180, 0, 180, 22.5}},    // 水平条带
    };
    
    for (const auto& pattern : accessPatterns) {
        ReadGridDataRequest testRequest = request;
        testRequest.outputBounds = pattern.second;
        
        // 多次测试获取稳定结果
        std::vector<double> times;
        for (int i = 0; i < 3; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto dataFuture = reader->readGridDataAsync(testRequest);
            auto result = dataFuture.get();
            auto end = std::chrono::high_resolution_clock::now();
            
            ASSERT_TRUE(result->requestStatus.success);
            times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        }
        
        double avgTime = (times[0] + times[1] + times[2]) / 3.0;
        LOG_INFO("{}访问模式: 平均 {:.1f}ms", pattern.first, avgTime);
    }
    
    LOG_INFO("  优化建议: 如果随机访问明显慢于顺序访问，考虑缓存优化");
}

TEST_F(PerformanceBottleneckAnalysis, DataTypeConversionOverhead) {
    LOG_INFO("=== 数据类型转换开销分析 ===");
    
    auto reader = std::make_unique<GdalUnifiedReader>(gdalTestFile_, nullptr, GdalReaderType::RASTER);
    auto openFuture = reader->openAsync(gdalTestFile_);
    ASSERT_TRUE(openFuture.get());
    
    ReadGridDataRequest request;
    request.variableName = "Band_1";
    // 使用较小的区域减少I/O影响
    request.outputBounds = BoundingBox{-180, -90, -90, 0}; // 1/4大小
    
    auto start = std::chrono::high_resolution_clock::now();
    auto dataFuture = reader->readGridDataAsync(request);
    auto result = dataFuture.get();
    auto end = std::chrono::high_resolution_clock::now();
    
    ASSERT_TRUE(result->requestStatus.success);
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    size_t dataSize = result->grid->data.size();
    double conversionRate = static_cast<double>(dataSize) / (1024.0 * 1024.0) / (totalTime / 1000.0);
    
    LOG_INFO("数据类型转换分析:");
    LOG_INFO("  处理数据量: {:.2f} MB", static_cast<double>(dataSize) / (1024.0 * 1024.0));
    LOG_INFO("  总处理时间: {}ms", totalTime);
    LOG_INFO("  转换速度: {:.2f} MB/s", conversionRate);
    
    if (conversionRate < 1000) {
        LOG_INFO("  优化建议: 数据转换较慢，考虑SIMD或零拷贝优化");
    }
}

int main(int argc, char** argv) {
    LoggingManager::initialize();
    LoggingManager::setLevel(spdlog::level::info);
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 