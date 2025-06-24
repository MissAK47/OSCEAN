/**
 * @file gpu_performance_monitor.cpp
 * @brief GPU性能监控器实现
 */

#include "common_utils/utilities/boost_config.h"
#include "common_utils/gpu/gpu_performance_monitor.h"
#include "common_utils/utilities/logging_utils.h"
#include <boost/thread/lock_guard.hpp>
#include <sstream>

namespace oscean::common_utils::gpu {

// GPUPerformanceTimer 实现
GPUPerformanceTimer::GPUPerformanceTimer(
    const std::string& kernelName,
    IGPUPerformanceMonitor* monitor)
    : m_kernelName(kernelName)
    , m_monitor(monitor)
    , m_startTime(boost::chrono::steady_clock::now())
    , m_data(std::make_unique<KernelPerformanceData>()) {
    
    m_data->kernelName = kernelName;
    m_data->timestamp = m_startTime;
}

GPUPerformanceTimer::~GPUPerformanceTimer() {
    if (m_monitor && m_data) {
        auto endTime = boost::chrono::steady_clock::now();
        auto duration = boost::chrono::duration_cast<boost::chrono::milliseconds>(
            endTime - m_startTime);
        m_data->timing.totalTime = duration.count();
        
        // 记录性能数据
        m_monitor->recordKernelPerformance(*m_data);
    }
}

void GPUPerformanceTimer::setKernelTime(double ms) {
    if (m_data) {
        m_data->timing.kernelTime = ms;
    }
}

void GPUPerformanceTimer::setMemoryTransfer(
    size_t bytesH2D, size_t bytesD2H,
    double timeH2D, double timeD2H) {
    
    if (m_data) {
        m_data->memory.bytesH2D = bytesH2D;
        m_data->memory.bytesD2H = bytesD2H;
        m_data->timing.memcpyH2D = timeH2D;
        m_data->timing.memcpyD2H = timeD2H;
        
        // 计算带宽 (GB/s)
        if (timeH2D > 0) {
            m_data->memory.bandwidthH2D = (bytesH2D / (1024.0 * 1024.0 * 1024.0)) / (timeH2D / 1000.0);
        }
        if (timeD2H > 0) {
            m_data->memory.bandwidthD2H = (bytesD2H / (1024.0 * 1024.0 * 1024.0)) / (timeD2H / 1000.0);
        }
    }
}

void GPUPerformanceTimer::setWorkload(const size_t gridSize[3], const size_t blockSize[3]) {
    if (m_data) {
        for (int i = 0; i < 3; ++i) {
            m_data->workload.gridSize[i] = gridSize[i];
            m_data->workload.blockSize[i] = blockSize[i];
        }
        m_data->workload.totalThreads = 
            gridSize[0] * gridSize[1] * gridSize[2] * 
            blockSize[0] * blockSize[1] * blockSize[2];
    }
}

// GPUPerformanceAnalyzer 静态方法实现
GPUPerformanceAnalyzer::AnalysisResult GPUPerformanceAnalyzer::analyzeKernelPerformance(
    const KernelPerformanceData& kernelData,
    const GPUDeviceInfo& deviceInfo) {
    
    AnalysisResult result;
    
    // 简单的分析逻辑
    double totalTime = kernelData.timing.totalTime;
    double kernelTime = kernelData.timing.kernelTime;
    double transferTime = kernelData.timing.memcpyH2D + kernelData.timing.memcpyD2H;
    
    // 判断瓶颈
    if (transferTime > kernelTime) {
        result.primaryBottleneck = BottleneckType::MEMORY_BOUND;
    } else if (kernelData.efficiency.occupancy < 50.0f) {
        result.primaryBottleneck = BottleneckType::LATENCY_BOUND;
    } else {
        result.primaryBottleneck = BottleneckType::COMPUTE_BOUND;
    }
    
    // 计算效率
    result.overallEfficiency = (kernelTime / totalTime) * 100.0;
    result.details.computeUtilization = kernelData.efficiency.smEfficiency;
    result.details.memoryUtilization = kernelData.efficiency.memoryEfficiency;
    result.details.transferEfficiency = (kernelData.memory.bandwidthH2D + kernelData.memory.bandwidthD2H) / 2.0;
    result.details.kernelEfficiency = kernelData.efficiency.occupancy;
    result.details.occupancyRatio = kernelData.efficiency.occupancy / 100.0;
    
    // 生成优化建议
    if (result.primaryBottleneck == BottleneckType::MEMORY_BOUND) {
        result.hints.push_back({
            "Memory Optimization",
            "Consider using shared memory or texture memory to reduce global memory access",
            8,
            15.0
        });
    }
    
    return result;
}

GPUPerformanceAnalyzer::AnalysisResult GPUPerformanceAnalyzer::analyzeDevicePerformance(
    const GPUPerformanceMetrics& metrics,
    const std::vector<GPUPerformanceMetrics>& history) {
    
    AnalysisResult result;
    
    // 检查温度限制
    if (metrics.thermal.temperature > 80) {
        result.primaryBottleneck = BottleneckType::THERMAL_THROTTLE;
    }
    // 检查功耗限制
    else if (metrics.thermal.power >= metrics.thermal.powerLimit * 0.95f) {
        result.primaryBottleneck = BottleneckType::POWER_THROTTLE;
    }
    // 检查GPU利用率
    else if (metrics.utilization.gpu < 50.0f) {
        result.primaryBottleneck = BottleneckType::LATENCY_BOUND;
    }
    else if (metrics.utilization.memory > metrics.utilization.gpu) {
        result.primaryBottleneck = BottleneckType::MEMORY_BOUND;
    }
    else {
        result.primaryBottleneck = BottleneckType::COMPUTE_BOUND;
    }
    
    // 计算整体效率
    result.overallEfficiency = metrics.utilization.gpu;
    
    return result;
}

std::string GPUPerformanceAnalyzer::generatePerformanceReport(
    const IGPUPerformanceMonitor& monitor,
    int deviceId) {
    
    std::stringstream report;
    report << "# GPU Performance Report\n\n";
    report << "## Device " << deviceId << " Metrics\n\n";
    
    auto metrics = monitor.getDeviceMetrics(deviceId);
    report << "- GPU Utilization: " << metrics.utilization.gpu << "%\n";
    report << "- Memory Utilization: " << metrics.utilization.memory << "%\n";
    report << "- Temperature: " << metrics.thermal.temperature << "°C\n";
    report << "- Power: " << metrics.thermal.power << "W / " << metrics.thermal.powerLimit << "W\n";
    report << "- GPU Clock: " << metrics.clocks.graphics << " MHz\n";
    report << "- Memory Clock: " << metrics.clocks.memory << " MHz\n";
    
    return report.str();
}

// GPUPerformanceMonitorFactory 实现
std::unique_ptr<IGPUPerformanceMonitor> GPUPerformanceMonitorFactory::createMonitor(ComputeAPI api) {
    // TODO: 根据不同的API创建不同的监控器实现
    return nullptr;
}

std::unique_ptr<IGPUPerformanceMonitor> GPUPerformanceMonitorFactory::createOptimalMonitor(
    const GPUDeviceInfo& device) {
    return createMonitor(device.getBestAPI());
}

// GlobalPerformanceManager 实现
void GlobalPerformanceManager::setGlobalMonitor(std::shared_ptr<IGPUPerformanceMonitor> monitor) {
    boost::lock_guard<boost::mutex> lock(m_mutex);
    m_globalMonitor = monitor;
}

std::shared_ptr<IGPUPerformanceMonitor> GlobalPerformanceManager::getGlobalMonitor() const {
    boost::lock_guard<boost::mutex> lock(m_mutex);
    return m_globalMonitor;
}

size_t GlobalPerformanceManager::registerPerformanceCallback(PerformanceCallback callback) {
    boost::lock_guard<boost::mutex> lock(m_mutex);
    size_t id = m_nextCallbackId++;
    m_callbacks[id] = callback;
    return id;
}

void GlobalPerformanceManager::unregisterPerformanceCallback(size_t callbackId) {
    boost::lock_guard<boost::mutex> lock(m_mutex);
    m_callbacks.erase(callbackId);
}

} // namespace oscean::common_utils::gpu 