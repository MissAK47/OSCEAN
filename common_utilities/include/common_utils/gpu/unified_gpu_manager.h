/**
 * @file unified_gpu_manager.h
 * @brief 统一GPU管理器接口 - 自动适配所有GPU类型
 * 
 * 负责检测、初始化和管理各种厂商的GPU设备
 */

#pragma once

#include "gpu_common.h"
#include "gpu_device_info.h"
#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include <functional>

namespace oscean {
namespace common_utils {
namespace gpu {

/**
 * @brief GPU初始化选项
 */
struct GPUInitOptions {
    bool enableMultiGPU = true;         ///< 是否启用多GPU支持
    bool preferHighPerformance = true;  ///< 优先选择高性能GPU
    bool enablePeerAccess = true;       ///< 启用GPU间直接访问
    bool autoSelectAPI = true;          ///< 自动选择最优API
    ComputeAPI preferredAPI = ComputeAPI::AUTO_DETECT; ///< 首选API
    size_t memoryPoolSize = 0;          ///< 内存池大小（0表示自动）
    bool enableProfiling = false;       ///< 启用性能分析
    bool verboseLogging = false;        ///< 详细日志输出
};

/**
 * @brief GPU设备过滤器
 */
struct GPUDeviceFilter {
    boost::optional<GPUVendor> vendor;              ///< 厂商过滤
    boost::optional<ComputeAPI> requiredAPI;        ///< 必需的API
    boost::optional<size_t> minMemorySize;          ///< 最小内存要求
    boost::optional<int> minPerformanceScore;       ///< 最小性能评分
    boost::optional<bool> requireTensorCores;       ///< 需要Tensor核心
    boost::optional<bool> requireDoublePrecision;   ///< 需要双精度支持
    
    /**
     * @brief 检查设备是否满足过滤条件
     */
    bool matches(const GPUDeviceInfo& device) const {
        if (vendor && device.vendor != *vendor) return false;
        if (requiredAPI && !device.hasAPI(*requiredAPI)) return false;
        if (minMemorySize && device.memoryDetails.totalGlobalMemory < *minMemorySize) return false;
        if (minPerformanceScore && device.performanceScore < *minPerformanceScore) return false;
        if (requireTensorCores && *requireTensorCores && !device.hasAIAcceleration()) return false;
        if (requireDoublePrecision && *requireDoublePrecision && 
            !device.capabilities.supportsDoublePrecision) return false;
        return true;
    }
};

/**
 * @brief GPU事件回调类型
 */
using GPUEventCallback = std::function<void(int deviceId, const std::string& event)>;

/**
 * @brief 统一GPU管理器接口
 * 
 * 提供跨平台、多厂商的GPU设备管理功能
 */
class IUnifiedGPUManager {
public:
    virtual ~IUnifiedGPUManager() = default;
    
    // === 初始化和检测 ===
    
    /**
     * @brief 初始化GPU管理器
     * @param options 初始化选项
     * @return 成功返回SUCCESS
     */
    virtual GPUError initialize(const GPUInitOptions& options = {}) = 0;
    
    /**
     * @brief 检测所有可用的GPU设备
     * @param filter 可选的设备过滤器
     * @return GPU设备列表
     */
    virtual std::vector<GPUDeviceInfo> detectAllGPUs(
        const GPUDeviceFilter& filter = {}) = 0;
    
    /**
     * @brief 刷新GPU设备列表
     * @return 成功返回SUCCESS
     */
    virtual GPUError refreshDeviceList() = 0;
    
    // === 设备管理 ===
    
    /**
     * @brief 获取设备数量
     * @return GPU设备数量
     */
    virtual size_t getDeviceCount() const = 0;
    
    /**
     * @brief 获取指定设备信息
     * @param deviceId 设备ID
     * @return 设备信息
     */
    virtual boost::optional<GPUDeviceInfo> getDeviceInfo(int deviceId) const = 0;
    
    /**
     * @brief 获取所有设备信息
     * @return 所有设备信息列表
     */
    virtual std::vector<GPUDeviceInfo> getAllDeviceInfo() const = 0;
    
    /**
     * @brief 设置当前活动设备
     * @param deviceId 设备ID
     * @return 成功返回SUCCESS
     */
    virtual GPUError setCurrentDevice(int deviceId) = 0;
    
    /**
     * @brief 获取当前活动设备ID
     * @return 当前设备ID
     */
    virtual int getCurrentDevice() const = 0;
    
    // === 最优配置选择 ===
    
    /**
     * @brief 获取最优GPU配置
     * @param requirements 需求过滤器
     * @return GPU配置
     */
    virtual GPUConfiguration getOptimalConfiguration(
        const GPUDeviceFilter& requirements = {}) = 0;
    
    /**
     * @brief 根据工作负载选择最优设备
     * @param workloadMemory 工作负载内存需求
     * @param workloadComplexity 工作负载复杂度(0-1)
     * @return 推荐的设备ID，-1表示无合适设备
     */
    virtual int selectOptimalDevice(size_t workloadMemory, 
                                   double workloadComplexity = 0.5) = 0;
    
    // === 设备状态监控 ===
    
    /**
     * @brief 获取设备内存信息
     * @param deviceId 设备ID
     * @return 内存信息
     */
    virtual boost::optional<GPUMemoryInfo> getMemoryInfo(int deviceId) const = 0;
    
    /**
     * @brief 获取设备温度信息
     * @param deviceId 设备ID
     * @return 温度信息
     */
    virtual boost::optional<GPUThermalInfo> getThermalInfo(int deviceId) const = 0;
    
    /**
     * @brief 获取设备功耗信息
     * @param deviceId 设备ID
     * @return 功耗信息
     */
    virtual boost::optional<GPUPowerInfo> getPowerInfo(int deviceId) const = 0;
    
    /**
     * @brief 获取设备利用率
     * @param deviceId 设备ID
     * @return 利用率(0-100)
     */
    virtual boost::optional<float> getDeviceUtilization(int deviceId) const = 0;
    
    // === 多GPU支持 ===
    
    /**
     * @brief 启用GPU间点对点访问
     * @param srcDevice 源设备ID
     * @param dstDevice 目标设备ID
     * @return 成功返回SUCCESS
     */
    virtual GPUError enablePeerAccess(int srcDevice, int dstDevice) = 0;
    
    /**
     * @brief 禁用GPU间点对点访问
     * @param srcDevice 源设备ID
     * @param dstDevice 目标设备ID
     * @return 成功返回SUCCESS
     */
    virtual GPUError disablePeerAccess(int srcDevice, int dstDevice) = 0;
    
    /**
     * @brief 检查GPU间是否可以点对点访问
     * @param srcDevice 源设备ID
     * @param dstDevice 目标设备ID
     * @return 如果可以访问返回true
     */
    virtual bool canAccessPeer(int srcDevice, int dstDevice) const = 0;
    
    // === 事件和回调 ===
    
    /**
     * @brief 注册设备事件回调
     * @param callback 回调函数
     */
    virtual void registerEventCallback(GPUEventCallback callback) = 0;
    
    /**
     * @brief 清除所有事件回调
     */
    virtual void clearEventCallbacks() = 0;
    
    // === 诊断和调试 ===
    
    /**
     * @brief 获取设备诊断信息
     * @param deviceId 设备ID
     * @return 诊断信息字符串
     */
    virtual std::string getDeviceDiagnostics(int deviceId) const = 0;
    
    /**
     * @brief 运行设备自检
     * @param deviceId 设备ID
     * @return 成功返回SUCCESS
     */
    virtual GPUError runDeviceSelfTest(int deviceId) = 0;
    
    /**
     * @brief 获取管理器状态报告
     * @return 状态报告字符串
     */
    virtual std::string getStatusReport() const = 0;
    
    // === 资源清理 ===
    
    /**
     * @brief 重置设备
     * @param deviceId 设备ID
     * @return 成功返回SUCCESS
     */
    virtual GPUError resetDevice(int deviceId) = 0;
    
    /**
     * @brief 清理所有资源
     */
    virtual void cleanup() = 0;
};

/**
 * @brief 统一GPU管理器实现类
 */
class UnifiedGPUManager : public IUnifiedGPUManager {
public:
    /**
     * @brief 获取单例实例
     * @return 管理器实例引用
     */
    static UnifiedGPUManager& getInstance();
    
    /**
     * @brief 构造函数
     */
    UnifiedGPUManager();
    
    /**
     * @brief 析构函数
     */
    ~UnifiedGPUManager() override;
    
    // 实现IUnifiedGPUManager接口
    GPUError initialize(const GPUInitOptions& options = {}) override;
    std::vector<GPUDeviceInfo> detectAllGPUs(const GPUDeviceFilter& filter = {}) override;
    GPUError refreshDeviceList() override;
    
    size_t getDeviceCount() const override;
    boost::optional<GPUDeviceInfo> getDeviceInfo(int deviceId) const override;
    std::vector<GPUDeviceInfo> getAllDeviceInfo() const override;
    GPUError setCurrentDevice(int deviceId) override;
    int getCurrentDevice() const override;
    
    GPUConfiguration getOptimalConfiguration(const GPUDeviceFilter& requirements = {}) override;
    int selectOptimalDevice(size_t workloadMemory, double workloadComplexity = 0.5) override;
    
    boost::optional<GPUMemoryInfo> getMemoryInfo(int deviceId) const override;
    boost::optional<GPUThermalInfo> getThermalInfo(int deviceId) const override;
    boost::optional<GPUPowerInfo> getPowerInfo(int deviceId) const override;
    boost::optional<float> getDeviceUtilization(int deviceId) const override;
    
    GPUError enablePeerAccess(int srcDevice, int dstDevice) override;
    GPUError disablePeerAccess(int srcDevice, int dstDevice) override;
    bool canAccessPeer(int srcDevice, int dstDevice) const override;
    
    void registerEventCallback(GPUEventCallback callback) override;
    void clearEventCallbacks() override;
    
    std::string getDeviceDiagnostics(int deviceId) const override;
    GPUError runDeviceSelfTest(int deviceId) override;
    std::string getStatusReport() const override;
    
    GPUError resetDevice(int deviceId) override;
    void cleanup() override;
    
private:
    // === 平台特定的检测方法 ===
    std::vector<GPUDeviceInfo> detectNVIDIAGPUs();
    std::vector<GPUDeviceInfo> detectAMDGPUs();
    std::vector<GPUDeviceInfo> detectIntelGPUs();
    std::vector<GPUDeviceInfo> detectAppleGPUs();
    std::vector<GPUDeviceInfo> detectOpenCLGPUs();
    
    // === 内部辅助方法 ===
    void removeDuplicatesAndSort(std::vector<GPUDeviceInfo>& devices);
    int calculatePerformanceScore(const GPUDeviceInfo& device);
    GPUConfiguration createCPUFallbackConfig();
    void notifyEvent(int deviceId, const std::string& event);
    
    // === 成员变量 ===
    mutable std::mutex m_mutex;                         ///< 线程安全锁
    bool m_initialized;                                 ///< 是否已初始化
    GPUInitOptions m_options;                           ///< 初始化选项
    std::vector<GPUDeviceInfo> m_devices;              ///< 设备列表
    int m_currentDevice;                                ///< 当前活动设备
    std::vector<GPUEventCallback> m_eventCallbacks;    ///< 事件回调列表
    
    // 平台特定的实现指针
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace gpu
} // namespace common_utils
} // namespace oscean 