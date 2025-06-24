/**
 * @file gpu_visualization_engine.cpp
 * @brief GPU加速的可视化引擎实现
 */

#include "output_generation/gpu/gpu_visualization_engine.h"
#include "output_generation/gpu/gpu_color_mapper.h"
#include "output_generation/gpu/gpu_tile_generator.h"
#include "common_utils/gpu/oscean_gpu_framework.h"
#include "common_utils/gpu/multi_gpu_scheduler.h"
#include "common_utils/gpu/multi_gpu_memory_manager.h"
#include "common_utils/gpu/gpu_performance_monitor_impl.h"
#include <spdlog/spdlog.h>

// 包含具体实现
#include "gpu_color_mapper_impl.h"

namespace oscean::output_generation::gpu {

using namespace oscean::common_utils::gpu;

/**
 * @brief GPU可视化引擎实现类
 */
class GPUVisualizationEngine::Impl {
public:
    Impl() 
        : m_initialized(false)
        , m_currentDevice(-1) {
    }
    
    ~Impl() = default;
    
    bool initialize() {
        if (m_initialized) {
            return true;
        }
        
        // 初始化GPU框架
        if (!OSCEANGPUFramework::initialize()) {
            spdlog::error("Failed to initialize OSCEAN GPU framework");
            return false;
        }
        
        // 获取可用设备
        m_availableDevices = OSCEANGPUFramework::getAvailableDevices();
        if (m_availableDevices.empty()) {
            spdlog::warn("No GPU devices available");
            return false;
        }
        
        // 默认使用第一个设备
        m_currentDevice = 0;
        
        // 创建性能监控器
        m_performanceMonitor = std::make_unique<GPUPerformanceMonitor>();
        
        m_initialized = true;
        spdlog::info("GPU visualization engine initialized with {} devices", 
                     m_availableDevices.size());
        
        return true;
    }
    
    boost::future<GPUVisualizationResult> applyColorMapping(
        std::shared_ptr<GridData> gridData,
        const GPUColorMappingParams& params) {
        
        return boost::async(boost::launch::async, 
            [this, gridData, params]() -> GPUVisualizationResult {
                
            GPUVisualizationResult result;
            
            // 参数验证
            if (!gridData || gridData->getUnifiedBufferSize() == 0) {
                spdlog::error("Invalid grid data for color mapping");
                return result;
            }
            
            // 获取调度器和内存管理器
            auto& scheduler = OSCEANGPUFramework::getScheduler();
            auto& memManager = OSCEANGPUFramework::getMemoryManager();
            
            // 估算内存需求
            size_t numElements = gridData->getWidth() * gridData->getHeight() * gridData->getBandCount();
            size_t dataSize = numElements * gridData->getElementSizeBytes();
            size_t outputSize = numElements * 4; // RGBA
            
            // 选择最优GPU
            GPUTaskInfo taskInfo;
            taskInfo.taskId = "ColorMapping_" + std::to_string(boost::chrono::steady_clock::now().time_since_epoch().count());
            taskInfo.memoryRequirement = dataSize + outputSize;
            taskInfo.computeComplexity = 0.7;  // 颜色映射是中等复杂度任务
            taskInfo.priority = GPUTaskPriority::NORMAL;
            
            auto decision = scheduler.selectOptimalGPU(taskInfo);
            if (decision.selectedDeviceId == -1) {
                spdlog::error("No suitable GPU device for color mapping: {}", decision.reason);
                return result;
            }
            
            // 创建颜色映射器
            auto colorMapper = GPUVisualizationFactory::createOptimalColorMapper(
                m_availableDevices[decision.selectedDeviceId]);
            
            if (!colorMapper) {
                spdlog::error("Failed to create GPU color mapper");
                return result;
            }
            
            // 设置参数
            colorMapper->setParameters(params);
            
            // 创建执行上下文
            GPUExecutionContext context;
            context.deviceId = decision.selectedDeviceId;
            // 其他字段使用默认值
            
            // 执行颜色映射
            auto future = colorMapper->executeAsync(gridData, context);
            
            try {
                auto algorithmResult = future.get();
                if (algorithmResult.success) {
                    result = algorithmResult.data;
                    
                    // 更新性能统计
                    if (m_performanceMonitor) {
                        dynamic_cast<GPUPerformanceMonitor*>(m_performanceMonitor.get())->recordTaskExecution(
                            decision.selectedDeviceId,
                            "ColorMapping",
                            algorithmResult.stats.totalTime,
                            dataSize + outputSize
                        );
                    }
                } else {
                    spdlog::error("Color mapping failed: {}", algorithmResult.errorMessage);
                }
                
            } catch (const std::exception& e) {
                spdlog::error("Color mapping failed: {}", e.what());
            }
            
            return result;
        });
    }
    
    boost::future<std::vector<GPUVisualizationResult>> generateTiles(
        std::shared_ptr<GridData> gridData,
        const GPUTileGenerationParams& params) {
        
        return boost::async(boost::launch::async, 
            [this, gridData, params]() -> std::vector<GPUVisualizationResult> {
                
            std::vector<GPUVisualizationResult> results;
            
            // 参数验证
            if (!gridData || gridData->getUnifiedBufferSize() == 0) {
                spdlog::error("Invalid grid data for tile generation");
                return results;
            }
            
            // 创建瓦片生成器
            auto tileGenerator = GPUVisualizationFactory::createOptimalTileGenerator(
                m_availableDevices[m_currentDevice]);
            
            if (!tileGenerator) {
                spdlog::error("Failed to create GPU tile generator");
                return results;
            }
            
            // 设置瓦片生成参数
            (*tileGenerator).setParameters(params);
            
            // 计算瓦片数量
            auto tileCount = (*tileGenerator).calculateTileCount(
                params.zoomLevel, *gridData);
            
            spdlog::info("Generating tiles at zoom level {}", params.zoomLevel);
            
            // 创建执行上下文
            GPUExecutionContext context;
            context.deviceId = m_currentDevice;
            // 其他字段使用默认值
            
            // 执行瓦片生成
            auto future = (*tileGenerator).executeAsync(gridData, context);
            
            try {
                auto algorithmResult = future.get();
                if (algorithmResult.success) {
                    results = algorithmResult.data;
                    
                    // 更新性能统计
                    size_t numElements = gridData->getWidth() * gridData->getHeight() * gridData->getBandCount();
                    size_t totalMemory = numElements * gridData->getElementSizeBytes() + 
                                       results.size() * params.tileSize * params.tileSize * 4;
                    
                    if (m_performanceMonitor) {
                        dynamic_cast<GPUPerformanceMonitor*>(m_performanceMonitor.get())->recordTaskExecution(
                            m_currentDevice,
                            "TileGeneration",
                            algorithmResult.stats.totalTime,
                            totalMemory
                        );
                    }
                } else {
                    spdlog::error("Tile generation failed: {}", algorithmResult.errorMessage);
                }
                
            } catch (const std::exception& e) {
                spdlog::error("Tile generation failed: {}", e.what());
            }
            
            return results;
        });
    }
    
    std::vector<GPUDeviceInfo> getAvailableDevices() const {
        return m_availableDevices;
    }
    
    bool setDevice(int deviceId) {
        if (deviceId < 0 || deviceId >= static_cast<int>(m_availableDevices.size())) {
            spdlog::error("Invalid device ID: {}", deviceId);
            return false;
        }
        
        m_currentDevice = deviceId;
        spdlog::info("Set current GPU device to: {}", 
                     m_availableDevices[deviceId].name);
        return true;
    }
    
    IGPUPerformanceMonitor* getPerformanceMonitor() const {
        return m_performanceMonitor.get();
    }
    
private:
    bool m_initialized;
    int m_currentDevice;
    std::vector<GPUDeviceInfo> m_availableDevices;
    std::unique_ptr<GPUPerformanceMonitor> m_performanceMonitor;
};

// GPU可视化引擎实现
GPUVisualizationEngine::GPUVisualizationEngine()
    : m_impl(std::make_unique<Impl>()) {
}

GPUVisualizationEngine::~GPUVisualizationEngine() = default;

bool GPUVisualizationEngine::initialize() {
    return m_impl->initialize();
}

boost::future<GPUVisualizationResult> GPUVisualizationEngine::applyColorMapping(
    std::shared_ptr<GridData> gridData,
    const GPUColorMappingParams& params) {
    return m_impl->applyColorMapping(gridData, params);
}

boost::future<std::vector<GPUVisualizationResult>> GPUVisualizationEngine::generateTiles(
    std::shared_ptr<GridData> gridData,
    const GPUTileGenerationParams& params) {
    return m_impl->generateTiles(gridData, params);
}

std::vector<GPUDeviceInfo> GPUVisualizationEngine::getAvailableDevices() const {
    return m_impl->getAvailableDevices();
}

bool GPUVisualizationEngine::setDevice(int deviceId) {
    return m_impl->setDevice(deviceId);
}

IGPUPerformanceMonitor* GPUVisualizationEngine::getPerformanceMonitor() const {
    return m_impl->getPerformanceMonitor();
}

// 声明工厂函数
std::unique_ptr<IGPUColorMapper> createCUDAColorMapper();
std::unique_ptr<IGPUColorMapper> createCUDAColorMapperReal();

// GPU可视化工厂实现
std::unique_ptr<IGPUColorMapper> GPUVisualizationFactory::createColorMapper(ComputeAPI api) {
    switch (api) {
        case ComputeAPI::CUDA:
            // 尝试创建真正的CUDA实现
            try {
                return createCUDAColorMapperReal();
            } catch (...) {
                // 如果失败，回退到简化实现
                spdlog::warn("Failed to create real CUDA color mapper, falling back to simplified implementation");
                return std::make_unique<GPUColorMapperImpl>();
            }
        case ComputeAPI::OPENCL:
            // TODO: 实现OpenCL版本
            return std::make_unique<GPUColorMapperImpl>();
        default:
            // 默认使用CPU实现
            return std::make_unique<GPUColorMapperImpl>();
    }
}

std::unique_ptr<IGPUTileGenerator> GPUVisualizationFactory::createTileGenerator(ComputeAPI api) {
    // 暂时只使用一个实现
    return createGPUTileGenerator(0);
}

std::unique_ptr<IGPUColorMapper> GPUVisualizationFactory::createOptimalColorMapper(
    const GPUDeviceInfo& device) {
    // 根据设备选择最优API
    ComputeAPI api = ComputeAPI::AUTO_DETECT;
    
    if (device.vendor == GPUVendor::NVIDIA) {
        for (const auto& supportedAPI : device.supportedAPIs) {
            if (supportedAPI == ComputeAPI::CUDA) {
                api = ComputeAPI::CUDA;
                break;
            }
        }
    }
    
    if (api == ComputeAPI::AUTO_DETECT) {
        // 使用OpenCL作为通用后备
        for (const auto& supportedAPI : device.supportedAPIs) {
            if (supportedAPI == ComputeAPI::OPENCL) {
                api = ComputeAPI::OPENCL;
                break;
            }
        }
    }
    
    return createColorMapper(api);
}

std::unique_ptr<IGPUTileGenerator> GPUVisualizationFactory::createOptimalTileGenerator(
    const GPUDeviceInfo& device) {
    // 根据设备选择最优API
    ComputeAPI api = ComputeAPI::AUTO_DETECT;
    
    if (device.vendor == GPUVendor::NVIDIA) {
        for (const auto& supportedAPI : device.supportedAPIs) {
            if (supportedAPI == ComputeAPI::CUDA) {
                api = ComputeAPI::CUDA;
                break;
            }
        }
    }
    
    if (api == ComputeAPI::AUTO_DETECT) {
        // 使用OpenCL作为通用后备
        for (const auto& supportedAPI : device.supportedAPIs) {
            if (supportedAPI == ComputeAPI::OPENCL) {
                api = ComputeAPI::OPENCL;
                break;
            }
        }
    }
    
    return createTileGenerator(api);
}

} // namespace oscean::output_generation::gpu 