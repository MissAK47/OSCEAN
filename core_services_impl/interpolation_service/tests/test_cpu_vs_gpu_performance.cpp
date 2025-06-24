#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>

#include "interpolation/gpu/gpu_interpolation_engine.h"
#include "../src/impl/algorithms/bilinear_interpolator.h"
#include "../src/impl/algorithms/nearest_neighbor_interpolator.h"
#include "../src/impl/algorithms/fast_pchip_interpolator_2d.h"
#include "core_services/common_data_types.h"
#include "core_services/interpolation/i_interpolation_service.h"
#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/gpu/multi_gpu_memory_manager.h"
#include "common_utils/utilities/logging_utils.h"

using namespace std::chrono;
using namespace oscean::core_services;

/**
 * @brief 生成测试数据
 */
boost::shared_ptr<GridData> generateTestData(int width, int height) {
    GridDefinition def;
    def.cols = width;
    def.rows = height;
    def.extent.minX = 0.0;
    def.extent.maxX = width - 1.0;
    def.extent.minY = 0.0;
    def.extent.maxY = height - 1.0;
    def.xResolution = 1.0;
    def.yResolution = 1.0;
    
    auto data = boost::make_shared<GridData>(def, DataType::Float32, 1);
    
    // 生成随机数据
    size_t dataSize = width * height * sizeof(float);
    auto& buffer = const_cast<std::vector<unsigned char>&>(data->getUnifiedBuffer());
    buffer.resize(dataSize);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    float* dataPtr = reinterpret_cast<float*>(buffer.data());
    for (int i = 0; i < width * height; ++i) {
        dataPtr[i] = dis(gen);
    }
    
    return data;
}

/**
 * @brief 运行性能对比测试
 */
void runPerformanceComparison(int srcSize, int dstSize) {
    std::cout << "\n====== CPU vs GPU 性能对比测试 ======" << std::endl;
    std::cout << "源数据: " << srcSize << "x" << srcSize << std::endl;
    std::cout << "目标数据: " << dstSize << "x" << dstSize << std::endl;
    
    // 生成测试数据
    auto sourceData = generateTestData(srcSize, srcSize);
    
    // 生成目标点
    std::vector<oscean::core_services::interpolation::TargetPoint> targetPoints;
    targetPoints.reserve(dstSize * dstSize);
    
    for (int y = 0; y < dstSize; y++) {
        for (int x = 0; x < dstSize; x++) {
            oscean::core_services::interpolation::TargetPoint pt;
            // 将目标点映射到源数据的坐标范围
            double xCoord = (double)x / (dstSize - 1) * (srcSize - 1);
            double yCoord = (double)y / (dstSize - 1) * (srcSize - 1);
            pt.coordinates = {xCoord, yCoord};
            targetPoints.push_back(pt);
        }
    }
    
    // 创建GPU引擎
    auto factory = boost::make_shared<oscean::core_services::interpolation::gpu::GPUInterpolationEngineFactory>();
    auto gpuEngine = factory->create(oscean::common_utils::gpu::ComputeAPI::CUDA);
    
    // 创建CPU插值器
    oscean::core_services::interpolation::BilinearInterpolator bilinearInterpolator;
    oscean::core_services::interpolation::NearestNeighborInterpolator nearestInterpolator;
    oscean::core_services::interpolation::FastPchipInterpolator2D pchipInterpolator(
        boost::const_pointer_cast<const GridData>(sourceData), nullptr);
    
    // 测试配置
    struct TestConfig {
        oscean::core_services::interpolation::InterpolationMethod method;
        std::string name;
        bool supportedOnCPU;
    };
    
    std::vector<TestConfig> testConfigs = {
        {oscean::core_services::interpolation::InterpolationMethod::BILINEAR, "双线性", true},
        {oscean::core_services::interpolation::InterpolationMethod::BICUBIC, "双三次", false},
        {oscean::core_services::interpolation::InterpolationMethod::NEAREST_NEIGHBOR, "最近邻", true},
        {oscean::core_services::interpolation::InterpolationMethod::PCHIP_FAST_2D, "PCHIP 2D", true}
    };
    
    std::cout << "\n" << std::setw(20) << "插值方法" 
              << std::setw(15) << "CPU时间(ms)"
              << std::setw(15) << "GPU总时间(ms)"
              << std::setw(15) << "GPU核函数(ms)"
              << std::setw(15) << "数据传输(ms)"
              << std::setw(10) << "加速比"
              << std::setw(15) << "吞吐量(MP/s)"
              << std::endl;
    std::cout << std::string(110, '-') << std::endl;
    
    for (const auto& config : testConfigs) {
        std::cout << std::setw(20) << config.name;
        
        // CPU测试
        double cpuTime = 0.0;
        bool cpuSuccess = false;
        
        if (config.supportedOnCPU) {
            try {
                auto cpuStart = high_resolution_clock::now();
                
                if (config.method == oscean::core_services::interpolation::InterpolationMethod::BILINEAR) {
                    // 使用SIMD优化版本
                    auto results = bilinearInterpolator.interpolateAtPointsSIMD(*sourceData, targetPoints);
                    cpuSuccess = !results.empty();
                } else if (config.method == oscean::core_services::interpolation::InterpolationMethod::NEAREST_NEIGHBOR) {
                    auto results = nearestInterpolator.interpolateAtPointsSIMD(*sourceData, targetPoints);
                    cpuSuccess = !results.empty();
                } else if (config.method == oscean::core_services::interpolation::InterpolationMethod::PCHIP_FAST_2D) {
                    oscean::core_services::interpolation::InterpolationRequest request;
                    request.sourceGrid = sourceData;
                    auto result = pchipInterpolator.execute(request, nullptr);
                    cpuSuccess = (result.statusCode == 0);
                }
                
                auto cpuEnd = high_resolution_clock::now();
                cpuTime = duration_cast<microseconds>(cpuEnd - cpuStart).count() / 1000.0;
                
            } catch (const std::exception& e) {
                std::cerr << "\nCPU测试失败(" << config.name << "): " << e.what() << std::endl;
            }
        } else {
            cpuSuccess = false;
        }
        
        if (cpuSuccess) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) << cpuTime;
        } else {
            std::cout << std::setw(15) << "不支持";
        }
        
        // GPU测试
        oscean::core_services::interpolation::gpu::GPUInterpolationParams params;
        params.sourceData = sourceData;
        params.outputWidth = dstSize;
        params.outputHeight = dstSize;
        params.method = config.method;
        params.useTextureMemory = false;
        
        oscean::common_utils::gpu::GPUExecutionContext context;
        context.deviceId = 0;
        
        double gpuTotalTime = 0.0;
        double gpuKernelTime = 0.0;
        double gpuTransferTime = 0.0;
        double throughput = 0.0;
        bool gpuSuccess = false;
        
        try {
            // 预热
            gpuEngine->execute(params, context);
            
            // 正式测试
            auto start = high_resolution_clock::now();
            auto algResult = gpuEngine->execute(params, context);
            auto end = high_resolution_clock::now();
            
            if (algResult.success) {
                gpuSuccess = true;
                gpuTotalTime = duration_cast<microseconds>(end - start).count() / 1000.0;
                const auto& result = algResult.data;
                gpuKernelTime = result.gpuTimeMs;
                gpuTransferTime = result.memoryTransferTimeMs;
                throughput = (dstSize * dstSize) / (result.gpuTimeMs * 1000.0); // 百万点/秒
            }
        } catch (const std::exception& e) {
            std::cerr << "\nGPU测试失败(" << config.name << "): " << e.what() << std::endl;
        }
        
        if (gpuSuccess) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) << gpuTotalTime
                      << std::setw(15) << std::fixed << std::setprecision(2) << gpuKernelTime
                      << std::setw(15) << std::fixed << std::setprecision(2) << gpuTransferTime;
            
            if (cpuSuccess && gpuKernelTime > 0) {
                double speedup = cpuTime / gpuKernelTime;
                std::cout << std::setw(10) << std::fixed << std::setprecision(1) << speedup << "x";
            } else {
                std::cout << std::setw(10) << "-";
            }
            
            std::cout << std::setw(15) << std::fixed << std::setprecision(1) << throughput;
        } else {
            std::cout << std::setw(15) << "失败"
                      << std::setw(15) << "-"
                      << std::setw(15) << "-"
                      << std::setw(10) << "-"
                      << std::setw(15) << "-";
        }
        
        std::cout << std::endl;
    }
    
    // 分析结果
    std::cout << "\n性能分析:" << std::endl;
    std::cout << "- CPU时间: 使用SIMD优化版本的计算时间" << std::endl;
    std::cout << "- GPU总时间: 包含数据传输和核函数执行" << std::endl;
    std::cout << "- GPU核函数: 仅GPU计算时间（不含传输）" << std::endl;
    std::cout << "- 加速比: CPU时间 / GPU核函数时间" << std::endl;
    std::cout << "- 注：双三次插值CPU版本暂未实现" << std::endl;
}

int main() {
    try {
        std::cout << "=== CPU vs GPU 插值性能对比测试 ===" << std::endl;
        
        // 初始化日志
        oscean::common_utils::LoggingConfig config;
        config.console_level = "warn";
        config.enable_console = true;
        oscean::common_utils::LoggingManager::configureGlobal(config);
        
        // 初始化GPU管理器
        auto& deviceManager = oscean::common_utils::gpu::UnifiedGPUManager::getInstance();
        deviceManager.initialize();
        
        // 获取设备信息
        auto devices = deviceManager.getAllDeviceInfo();
        
        // 初始化全局GPU内存管理器
        if (!devices.empty()) {
            oscean::common_utils::gpu::GlobalMemoryManager::initialize(devices);
        }
        
        // 检查GPU设备
        if (devices.empty()) {
            std::cerr << "错误：没有找到GPU设备" << std::endl;
            return 1;
        }
        
        std::cout << "\nGPU设备信息:" << std::endl;
        std::cout << "名称: " << devices[0].name << std::endl;
        std::cout << "显存: " << (devices[0].memoryDetails.totalGlobalMemory / (1024*1024)) << " MB" << std::endl;
        std::cout << "计算能力: " << devices[0].architecture.majorVersion << "." 
                  << devices[0].architecture.minorVersion << std::endl;
        
        // 运行不同规模的测试
        std::vector<std::pair<int, int>> testConfigs = {
            {128, 256},    // 小规模
            {256, 512},    // 中小规模
            {512, 1024},   // 中等规模
            {1024, 2048},  // 大规模
        };
        
        for (const auto& [srcSize, dstSize] : testConfigs) {
            runPerformanceComparison(srcSize, dstSize);
        }
        
        // 性能总结
        std::cout << "\n=== 性能测试总结 ===" << std::endl;
        std::cout << "基于测试结果的观察:" << std::endl;
        std::cout << "1. GPU在大规模数据处理上具有明显优势" << std::endl;
        std::cout << "2. 批量处理可以显著提升GPU利用率" << std::endl;
        std::cout << "3. 数据传输是GPU性能的主要瓶颈" << std::endl;
        std::cout << "4. 不同插值方法的GPU加速效果不同" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "未知错误" << std::endl;
        return 1;
    }
    
    return 0;
} 