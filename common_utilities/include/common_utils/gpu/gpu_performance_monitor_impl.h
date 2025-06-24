/**
 * @file gpu_performance_monitor_impl.h
 * @brief GPU性能监控器简单实现
 */

#pragma once

#include "gpu_performance_monitor.h"
#include <boost/thread/mutex.hpp>
#include <unordered_map>

namespace oscean::common_utils::gpu {

/**
 * @brief GPU性能监控器的简单实现
 */
class GPUPerformanceMonitor : public IGPUPerformanceMonitor {
public:
    GPUPerformanceMonitor() = default;
    ~GPUPerformanceMonitor() override = default;
    
    bool startMonitoring(const std::vector<int>& deviceIds = {}) override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        m_monitoring = true;
        m_monitoredDevices = deviceIds;
        return true;
    }
    
    void stopMonitoring() override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        m_monitoring = false;
    }
    
    bool isMonitoring() const override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        return m_monitoring;
    }
    
    GPUPerformanceMetrics getDeviceMetrics(int deviceId) const override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        auto it = m_deviceMetrics.find(deviceId);
        if (it != m_deviceMetrics.end()) {
            return it->second;
        }
        return GPUPerformanceMetrics{};
    }
    
    std::unordered_map<int, GPUPerformanceMetrics> getAllMetrics() const override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        return m_deviceMetrics;
    }
    
    void recordKernelPerformance(const KernelPerformanceData& data) override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        m_kernelHistory.push_back(data);
        if (m_kernelHistory.size() > m_maxHistorySize) {
            m_kernelHistory.erase(m_kernelHistory.begin());
        }
    }
    
    std::vector<KernelPerformanceData> getKernelHistory(
        const std::string& kernelName = "",
        size_t maxRecords = 100) const override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        std::vector<KernelPerformanceData> result;
        
        for (const auto& data : m_kernelHistory) {
            if (kernelName.empty() || data.kernelName == kernelName) {
                result.push_back(data);
                if (result.size() >= maxRecords) break;
            }
        }
        
        return result;
    }
    
    void recordEvent(const PerformanceEvent& event) override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        m_eventHistory.push_back(event);
        if (m_eventHistory.size() > m_maxHistorySize) {
            m_eventHistory.erase(m_eventHistory.begin());
        }
    }
    
    std::vector<PerformanceEvent> getEventHistory(
        boost::optional<PerformanceEventType> type = boost::none,
        size_t maxRecords = 100) const override {
        boost::lock_guard<boost::mutex> lock(m_mutex);
        std::vector<PerformanceEvent> result;
        
        for (const auto& event : m_eventHistory) {
            if (!type || event.type == *type) {
                result.push_back(event);
                if (result.size() >= maxRecords) break;
            }
        }
        
        return result;
    }
    
    /**
     * @brief 记录任务执行
     */
    void recordTaskExecution(int deviceId, const std::string& taskName, 
                           double executionTime, size_t memoryUsed) {
        KernelPerformanceData data;
        data.kernelName = taskName;
        data.deviceId = deviceId;
        data.timing.totalTime = executionTime;
        data.memory.bytesH2D = memoryUsed;
        data.timestamp = boost::chrono::steady_clock::now();
        
        recordKernelPerformance(data);
    }
    
private:
    mutable boost::mutex m_mutex;
    bool m_monitoring = false;
    std::vector<int> m_monitoredDevices;
    std::unordered_map<int, GPUPerformanceMetrics> m_deviceMetrics;
    std::vector<KernelPerformanceData> m_kernelHistory;
    std::vector<PerformanceEvent> m_eventHistory;
    size_t m_maxHistorySize = 1000;
};

} // namespace oscean::common_utils::gpu 