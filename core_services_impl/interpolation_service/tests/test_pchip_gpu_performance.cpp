#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

// 包含boost thread和future支持
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <boost/make_shared.hpp>

// 包含项目头文件
#include "interpolation/gpu/gpu_interpolation_engine.h"
#include "../src/impl/algorithms/fast_pchip_interpolator_2d.h"
#include "core_services/common_data_types.h"
#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/gpu/multi_gpu_memory_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/simd/simd_manager_unified.h"

// 定义M_PI（如果未定义）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std::chrono;
using namespace oscean::core_services;
using namespace oscean::core_services::interpolation;

/**
 * @brief 生成测试数据（模拟海洋深度数据）
 */
boost::shared_ptr<GridData> generateOceanDepthData(int width, int height) {
    GridDefinition def;
    def.cols = width;
    def.rows = height;
    def.extent.minX = 0.0;
    def.extent.maxX = width - 1.0;
    def.extent.minY = 0.0;
    def.extent.maxY = height - 1.0;
    def.srs = "";
    
    // 分配内存对齐的数据
    size_t dataSize = width * height;
    std::vector<double> data(dataSize);
    
    // 生成模拟海洋深度数据（使用多个正弦波叠加）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> noise(-50.0, 50.0);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // 基础深度
            double depth = -2000.0;
            
            // 大尺度特征
            depth += 500.0 * std::sin(2.0 * M_PI * x / width) * 
                     std::cos(2.0 * M_PI * y / height);
            
            // 中尺度特征
            depth += 200.0 * std::sin(4.0 * M_PI * x / width) * 
                     std::sin(4.0 * M_PI * y / height);
            
            // 小尺度特征
            depth += 100.0 * std::sin(8.0 * M_PI * x / width) * 
                     std::cos(8.0 * M_PI * y / height);
            
            // 随机噪声
            depth += noise(gen);
            
            data[y * width + x] = depth;
        }
    }
    
    // 创建GridData
    double geoTransform[6] = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    auto gridData = boost::make_shared<GridData>(
        def, DataType::Float64, 1, data.data(), geoTransform);
    
    return gridData;
}

/**
 * @brief PCHIP GPU性能测试
 */
void testPCHIPPerformance() {
    std::cout << "\n=== PCHIP GPU vs CPU性能测试 ===" << std::endl;
    std::cout << "测试环境: " << std::endl;
    
    // 初始化GPU
    auto& gpuFramework = oscean::common_utils::gpu::OSCEANGPUFramework::getInstance();
    if (!gpuFramework.isInitialized()) {
        gpuFramework.initialize();
    }
    
    auto devices = gpuFramework.getAvailableDevices();
    if (devices.empty()) {
        std::cout << "未检测到GPU设备，跳过GPU测试" << std::endl;
        return;
    }
    
    std::cout << "GPU设备: " << devices[0].name << std::endl;
    std::cout << "计算能力: " << devices[0].major << "." << devices[0].minor << std::endl;
    std::cout << "全局内存: " << (devices[0].totalMemory / 1024 / 1024) << " MB" << std::endl;
    
    // CPU信息
    std::cout << "\nCPU架构: x64" << std::endl;
    std::cout << "SIMD支持: AVX2 (假定)" << std::endl;
    
    // 测试不同规模
    struct TestCase {
        int srcWidth, srcHeight;
        int dstWidth, dstHeight;
        const char* name;
    };
    
    std::vector<TestCase> testCases = {
        {256, 256, 512, 512, "小规模(256x256 -> 512x512)"},
        {512, 512, 1024, 1024, "中规模(512x512 -> 1024x1024)"},
        {1024, 1024, 2048, 2048, "大规模(1024x1024 -> 2048x2048)"},
        {2048, 2048, 4096, 4096, "超大规模(2048x2048 -> 4096x4096)"}
    };
    
    // 结果汇总表
    std::cout << "\n性能测试结果:" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    std::cout << std::left 
              << std::setw(30) << "测试规模"
              << std::setw(15) << "CPU SIMD(ms)"
              << std::setw(15) << "GPU(ms)"
              << std::setw(15) << "GPU核函数(ms)"
              << std::setw(15) << "加速比"
              << std::setw(15) << "吞吐量(MP/s)"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    for (const auto& testCase : testCases) {
        std::cout << std::setw(30) << testCase.name << std::flush;
        
        try {
            // 生成源数据
            auto sourceData = generateOceanDepthData(testCase.srcWidth, testCase.srcHeight);
            
            // 准备目标点
            std::vector<TargetPoint> targetPoints;
            targetPoints.reserve(testCase.dstWidth * testCase.dstHeight);
            
            for (int y = 0; y < testCase.dstHeight; ++y) {
                for (int x = 0; x < testCase.dstWidth; ++x) {
                    TargetPoint pt;
                    pt.coordinates = {
                        x * (testCase.srcWidth - 1.0) / (testCase.dstWidth - 1.0),
                        y * (testCase.srcHeight - 1.0) / (testCase.dstHeight - 1.0)
                    };
                    targetPoints.push_back(pt);
                }
            }
            
            // 1. 测试CPU SIMD版本
            double cpuTime = 0.0;
            {
                // 创建PCHIP插值器（不使用SIMD管理器）
                FastPchipInterpolator2D pchipInterpolator(sourceData, nullptr);
                
                // 预热
                InterpolationRequest request;
                request.method = InterpolationMethod::PCHIP_FAST_2D;
                request.sourceGrid = sourceData;
                request.target = targetPoints;
                
                pchipInterpolator.execute(request);
                
                // 正式测试
                auto start = high_resolution_clock::now();
                auto result = pchipInterpolator.execute(request);
                auto end = high_resolution_clock::now();
                
                cpuTime = duration_cast<microseconds>(end - start).count() / 1000.0;
                
                // 验证结果
                if (std::holds_alternative<std::vector<std::optional<double>>>(result.data)) {
                    auto& values = std::get<std::vector<std::optional<double>>>(result.data);
                    int validCount = 0;
                    for (const auto& val : values) {
                        if (val.has_value()) validCount++;
                    }
                    if (validCount != targetPoints.size()) {
                        std::cout << "[CPU警告:部分插值失败] ";
                    }
                }
            }
            
            // 2. 测试GPU版本
            double gpuTime = 0.0;
            double gpuKernelTime = 0.0;
            {
                // 创建GPU插值引擎
                auto gpuEngine = oscean::core_services::interpolation::gpu::createGPUInterpolationEngine(
                    oscean::common_utils::gpu::ComputeAPI::CUDA
                );
                
                // 设置插值方法
                gpuEngine->setInterpolationMethod(InterpolationMethod::PCHIP_FAST_2D);
                
                // 准备GPU参数
                GPUInterpolationParams params;
                params.sourceData = sourceData;
                params.outputWidth = testCase.dstWidth;
                params.outputHeight = testCase.dstHeight;
                params.method = InterpolationMethod::PCHIP_FAST_2D;
                
                // 预热
                gpuEngine->execute(params);
                
                // 正式测试
                auto start = high_resolution_clock::now();
                auto result = gpuEngine->execute(params);
                auto end = high_resolution_clock::now();
                
                gpuTime = duration_cast<microseconds>(end - start).count() / 1000.0;
                
                if (result.status == oscean::common_utils::gpu::GPUError::SUCCESS) {
                    gpuKernelTime = result.data.gpuTimeMs;
                }
            }
            
            // 计算性能指标
            double speedup = cpuTime / gpuTime;
            double throughput = (testCase.dstWidth * testCase.dstHeight) / 
                               (gpuKernelTime / 1000.0) / (1024.0 * 1024.0);
            
            // 输出结果
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) << cpuTime
                      << std::setw(15) << gpuTime
                      << std::setw(15) << gpuKernelTime
                      << std::setw(15) << std::setprecision(2) << speedup << "x"
                      << std::setw(15) << std::setprecision(1) << throughput
                      << std::endl;
                      
        } catch (const std::exception& e) {
            std::cout << "错误: " << e.what() << std::endl;
        }
    }
    
    std::cout << std::string(100, '-') << std::endl;
    
    // 性能分析
    std::cout << "\n性能分析:" << std::endl;
    std::cout << "1. PCHIP算法特点：" << std::endl;
    std::cout << "   - 需要计算导数（X、Y、XY方向）" << std::endl;
    std::cout << "   - 使用Hermite多项式插值" << std::endl;
    std::cout << "   - 保持单调性，适合海洋深度等物理量" << std::endl;
    
    std::cout << "\n2. GPU优化策略：" << std::endl;
    std::cout << "   - 共享内存缓存数据块和导数" << std::endl;
    std::cout << "   - FMA指令优化Hermite计算" << std::endl;
    std::cout << "   - 一体化版本避免导数预计算开销" << std::endl;
    
    std::cout << "\n3. 性能瓶颈分析：" << std::endl;
    std::cout << "   - PCHIP需要4x4邻域数据（内存访问密集）" << std::endl;
    std::cout << "   - 导数计算增加了额外开销" << std::endl;
    std::cout << "   - 条件分支影响GPU效率" << std::endl;
}

int main() {
    try {
        // 初始化日志系统
        OSCEAN_LOG_INIT();
        
        // 初始化全局GPU框架
        auto& gpuFramework = oscean::common_utils::gpu::OSCEANGPUFramework::getInstance();
        if (!gpuFramework.isInitialized()) {
            gpuFramework.initialize();
        }
        
        // 运行PCHIP性能测试
        testPCHIPPerformance();
        
        std::cout << "\n测试完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "发生错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 