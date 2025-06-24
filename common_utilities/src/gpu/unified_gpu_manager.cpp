/**
 * @file unified_gpu_manager.cpp
 * @brief 统一GPU管理器实现
 */

#include "common_utils/gpu/unified_gpu_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include <boost/optional.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace oscean::common_utils::gpu {

// === 平台特定检测函数的外部声明 ===
// 这些函数在各自的检测器文件中实现
extern std::vector<GPUDeviceInfo> detectNVIDIAGPUsImpl();
extern std::vector<GPUDeviceInfo> detectAMDGPUsImpl();
extern std::vector<GPUDeviceInfo> detectOpenCLGPUsImpl();

// === 平台特定实现的前向声明 ===
class UnifiedGPUManager::Impl {
public:
    virtual ~Impl() = default;
    virtual bool isAvailable() const = 0;
};

// === 单例实现 ===
UnifiedGPUManager& UnifiedGPUManager::getInstance() {
    static UnifiedGPUManager instance;
    return instance;
}

// === 构造和析构 ===
UnifiedGPUManager::UnifiedGPUManager()
    : m_initialized(false)
    , m_currentDevice(-1) {
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "UnifiedGPUManager created"); // TODO: Fix log format
}

UnifiedGPUManager::~UnifiedGPUManager() {
    cleanup();
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "UnifiedGPUManager destroyed"); // TODO: Fix log format
}

// === 初始化和检测 ===
GPUError UnifiedGPUManager::initialize(const GPUInitOptions& options) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_initialized) {
        // OSCEAN_LOG_WARN("UnifiedGPUManager", "Manager already initialized"); // TODO: Fix log format
        return GPUError::SUCCESS;
    }
    
    m_options = options;
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "Initializing GPU Manager with options:"); // TODO: Fix log format
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "  - Multi-GPU: {}", options.enableMultiGPU ? "enabled" : "disabled"); // TODO: Fix log format
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "  - Preferred API: {}", computeAPIToString(options.preferredAPI)); // TODO: Fix log format
    
    // 检测所有GPU
    auto devices = detectAllGPUs();
    
    if (devices.empty()) {
        // OSCEAN_LOG_WARN("UnifiedGPUManager", "No GPU devices found"); // TODO: Fix log format
        // 创建CPU后备配置
        if (options.autoSelectAPI) {
            // OSCEAN_LOG_INFO("UnifiedGPUManager", "Creating CPU fallback configuration"); // TODO: Fix log format
            m_devices.clear();
            m_currentDevice = -1;
            m_initialized = true;
            return GPUError::SUCCESS;
        }
        return GPUError::DEVICE_NOT_FOUND;
    }
    
    m_devices = std::move(devices);
    
    // 选择默认设备
    if (!m_devices.empty()) {
        m_currentDevice = 0;
        // OSCEAN_LOG_INFO("UnifiedGPUManager", "Selected device 0 as default: {}", m_devices[0].getDescription()); // TODO: Fix log format
    }
    
    // 初始化多GPU支持
    if (options.enableMultiGPU && m_devices.size() > 1) {
        // OSCEAN_LOG_INFO("UnifiedGPUManager", "Enabling multi-GPU support for {} devices", 
        //                m_devices.size()); // TODO: Fix log format
        
        if (options.enablePeerAccess) {
            // 尝试启用GPU间点对点访问
            for (size_t i = 0; i < m_devices.size(); ++i) {
                for (size_t j = i + 1; j < m_devices.size(); ++j) {
                    if (canAccessPeer(static_cast<int>(i), static_cast<int>(j))) {
                        enablePeerAccess(static_cast<int>(i), static_cast<int>(j));
                        enablePeerAccess(static_cast<int>(j), static_cast<int>(i));
                    }
                }
            }
        }
    }
    
    m_initialized = true;
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "GPU Manager initialized successfully"); // TODO: Fix log format
    return GPUError::SUCCESS;
}

std::vector<GPUDeviceInfo> UnifiedGPUManager::detectAllGPUs(const GPUDeviceFilter& filter) {
    std::vector<GPUDeviceInfo> allDevices;
    
    OSCEAN_LOG_INFO("UnifiedGPUManager", "Starting GPU detection...");
    
    // 1. 检测NVIDIA GPU (CUDA)
    try {
        auto nvidiaDevices = detectNVIDIAGPUs();
        OSCEAN_LOG_INFO("UnifiedGPUManager", "Found " + std::to_string(nvidiaDevices.size()) + " NVIDIA GPU(s)");
        allDevices.insert(allDevices.end(), nvidiaDevices.begin(), nvidiaDevices.end());
    } catch (const std::exception& e) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", std::string("NVIDIA detection failed: ") + e.what());
    } catch (...) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", "NVIDIA detection failed: unknown error");
    }
    
    // 2. 检测AMD GPU (ROCm)
    try {
        auto amdDevices = detectAMDGPUs();
        OSCEAN_LOG_INFO("UnifiedGPUManager", "Found " + std::to_string(amdDevices.size()) + " AMD GPU(s)");
        allDevices.insert(allDevices.end(), amdDevices.begin(), amdDevices.end());
    } catch (const std::exception& e) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", std::string("AMD detection failed: ") + e.what());
    } catch (...) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", "AMD detection failed: unknown error");
    }
    
    // 3. 检测Intel GPU
    try {
        auto intelDevices = detectIntelGPUs();
        OSCEAN_LOG_INFO("UnifiedGPUManager", "Found " + std::to_string(intelDevices.size()) + " Intel GPU(s)");
        allDevices.insert(allDevices.end(), intelDevices.begin(), intelDevices.end());
    } catch (const std::exception& e) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", std::string("Intel detection failed: ") + e.what());
    } catch (...) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", "Intel detection failed: unknown error");
    }
    
    // 4. 检测Apple GPU (Metal)
    #ifdef __APPLE__
    try {
        auto appleDevices = detectAppleGPUs();
        OSCEAN_LOG_INFO("UnifiedGPUManager", "Found " + std::to_string(appleDevices.size()) + " Apple GPU(s)");
        allDevices.insert(allDevices.end(), appleDevices.begin(), appleDevices.end());
    } catch (const std::exception& e) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", std::string("Apple detection failed: ") + e.what());
    } catch (...) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", "Apple detection failed: unknown error");
    }
    #endif
    
    // 5. OpenCL通用检测（总是尝试）
    try {
        auto openclDevices = detectOpenCLGPUs();
        OSCEAN_LOG_INFO("UnifiedGPUManager", "Found " + std::to_string(openclDevices.size()) + " OpenCL GPU(s)");
        allDevices.insert(allDevices.end(), openclDevices.begin(), openclDevices.end());
    } catch (const std::exception& e) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", std::string("OpenCL detection failed: ") + e.what());
    } catch (...) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", "OpenCL detection failed: unknown error");
    }
    
    // 去重并排序
    try {
        removeDuplicatesAndSort(allDevices);
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("UnifiedGPUManager", std::string("Error during device sorting: ") + e.what());
    } catch (...) {
        OSCEAN_LOG_ERROR("UnifiedGPUManager", "Error during device sorting: unknown error");
    }
    
    // 应用过滤器
    if (filter.vendor || filter.requiredAPI || filter.minMemorySize || 
        filter.minPerformanceScore || filter.requireTensorCores || filter.requireDoublePrecision) {
        allDevices.erase(
            std::remove_if(allDevices.begin(), allDevices.end(),
                          [&filter](const GPUDeviceInfo& device) {
                              return !filter.matches(device);
                          }),
            allDevices.end()
        );
    }
    
    OSCEAN_LOG_INFO("UnifiedGPUManager", "Detection complete. Found " + std::to_string(allDevices.size()) + " GPU(s) matching criteria");
    
    return allDevices;
}

GPUError UnifiedGPUManager::refreshDeviceList() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_initialized) {
        return GPUError::NOT_SUPPORTED;
    }
    
    auto newDevices = detectAllGPUs();
    
    // 检查设备列表是否发生变化
    bool changed = (newDevices.size() != m_devices.size());
    if (!changed) {
        for (size_t i = 0; i < newDevices.size(); ++i) {
            if (newDevices[i].deviceId != m_devices[i].deviceId ||
                newDevices[i].name != m_devices[i].name) {
                changed = true;
                break;
            }
        }
    }
    
    if (changed) {
        // OSCEAN_LOG_WARN("UnifiedGPUManager", "Device list changed during refresh"); // TODO: Fix log format
        m_devices = std::move(newDevices);
        
        // 验证当前设备是否仍然有效
        if (m_currentDevice >= static_cast<int>(m_devices.size())) {
            m_currentDevice = m_devices.empty() ? -1 : 0;
        }
        
        // 通知设备变化
        notifyEvent(-1, "device_list_changed");
    }
    
    return GPUError::SUCCESS;
}

// === 设备管理 ===
size_t UnifiedGPUManager::getDeviceCount() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_devices.size();
}

boost::optional<GPUDeviceInfo> UnifiedGPUManager::getDeviceInfo(int deviceId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return boost::none;
    }
    
    return m_devices[deviceId];
}

std::vector<GPUDeviceInfo> UnifiedGPUManager::getAllDeviceInfo() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_devices;
}

GPUError UnifiedGPUManager::setCurrentDevice(int deviceId) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return GPUError::INVALID_DEVICE;
    }
    
    m_currentDevice = deviceId;
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "Current device set to {}: {}", deviceId, m_devices[deviceId].getDescription()); // TODO: Fix log format
    
    notifyEvent(deviceId, "device_selected");
    return GPUError::SUCCESS;
}

int UnifiedGPUManager::getCurrentDevice() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_currentDevice;
}

// === 最优配置选择 ===
GPUConfiguration UnifiedGPUManager::getOptimalConfiguration(const GPUDeviceFilter& requirements) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    GPUConfiguration config;
    
    // 筛选符合要求的设备
    std::vector<GPUDeviceInfo> eligibleDevices;
    for (const auto& device : m_devices) {
        if (requirements.matches(device)) {
            eligibleDevices.push_back(device);
        }
    }
    
    if (eligibleDevices.empty()) {
        // OSCEAN_LOG_WARN("UnifiedGPUManager", "No devices match requirements, using CPU fallback"); // TODO: Fix log format
        return createCPUFallbackConfig();
    }
    
    // 按性能评分排序
    std::sort(eligibleDevices.begin(), eligibleDevices.end(),
              [](const GPUDeviceInfo& a, const GPUDeviceInfo& b) {
                  return a.performanceScore > b.performanceScore;
              });
    
    // 选择最优设备作为主设备
    config.primaryDevice = eligibleDevices[0];
    config.computeAPI = config.primaryDevice.getBestAPI();
    
    // 如果启用多GPU，添加其他设备
    if (m_options.enableMultiGPU && eligibleDevices.size() > 1) {
        config.enableMultiGPU = true;
        for (size_t i = 1; i < eligibleDevices.size(); ++i) {
            config.secondaryDevices.push_back(eligibleDevices[i]);
        }
    } else {
        config.enableMultiGPU = false;
    }
    
    config.enablePeerAccess = m_options.enablePeerAccess;
    config.memoryPoolSize = m_options.memoryPoolSize;
    config.maxConcurrentKernels = 8; // 默认值
    
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "Optimal configuration: {} with {}", config.primaryDevice.getDescription(), computeAPIToString(config.computeAPI)); // TODO: Fix log format
    
    return config;
}

int UnifiedGPUManager::selectOptimalDevice(size_t workloadMemory, double workloadComplexity) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    int bestDevice = -1;
    double bestScore = -1.0;
    
    for (size_t i = 0; i < m_devices.size(); ++i) {
        const auto& device = m_devices[i];
        
        // 检查内存是否足够
        if (device.memoryDetails.freeGlobalMemory < workloadMemory) {
            continue;
        }
        
        // 计算适合度评分
        double memoryScore = static_cast<double>(device.memoryDetails.freeGlobalMemory) / 
                           device.memoryDetails.totalGlobalMemory;
        double performanceScore = device.performanceScore / 100.0;
        double complexityMatch = 1.0 - std::abs(performanceScore - workloadComplexity);
        
        double totalScore = memoryScore * 0.3 + performanceScore * 0.5 + complexityMatch * 0.2;
        
        if (totalScore > bestScore) {
            bestScore = totalScore;
            bestDevice = static_cast<int>(i);
        }
    }
    
    if (bestDevice >= 0) {
        // OSCEAN_LOG_INFO("UnifiedGPUManager", "Selected device {} for workload (memory: {} MB, complexity: {})", bestDevice, workloadMemory / (1024*1024), workloadComplexity); // TODO: Fix log format
    }
    
    return bestDevice;
}

// === 辅助方法实现 ===
void UnifiedGPUManager::removeDuplicatesAndSort(std::vector<GPUDeviceInfo>& devices) {
    if (devices.empty()) {
        return;
    }
    
    // 先计算性能评分
    for (auto& device : devices) {
        device.performanceScore = calculatePerformanceScore(device);
    }
    
    // 基于名称和厂商去重（更可靠）
    std::sort(devices.begin(), devices.end(),
              [](const GPUDeviceInfo& a, const GPUDeviceInfo& b) {
                  if (a.vendor != b.vendor) {
                      return a.vendor < b.vendor;
                  }
                  return a.name < b.name;
              });
    
    devices.erase(
        std::unique(devices.begin(), devices.end(),
                   [](const GPUDeviceInfo& a, const GPUDeviceInfo& b) {
                       return a.vendor == b.vendor && a.name == b.name;
                   }),
        devices.end()
    );
    
    // 按性能评分排序
    std::sort(devices.begin(), devices.end(),
              [](const GPUDeviceInfo& a, const GPUDeviceInfo& b) {
                  return a.performanceScore > b.performanceScore;
              });
    
    // 重新分配设备ID
    for (size_t i = 0; i < devices.size(); ++i) {
        devices[i].deviceId = static_cast<int>(i);
    }
}

int UnifiedGPUManager::calculatePerformanceScore(const GPUDeviceInfo& device) {
    // 基于多个因素计算性能评分
    double score = 0.0;
    
    // 计算单元评分 (30%)
    double computeScore = std::min(100.0, device.computeUnits.totalCores / 50.0);
    score += computeScore * 0.3;
    
    // 内存带宽评分 (25%)
    double bandwidthScore = std::min(100.0, device.memoryDetails.memoryBandwidth / 10.0);
    score += bandwidthScore * 0.25;
    
    // 内存容量评分 (20%)
    double memoryScore = std::min(100.0, device.memoryDetails.totalGlobalMemory / 
                                        (16.0 * 1024 * 1024 * 1024));
    score += memoryScore * 0.2;
    
    // 时钟频率评分 (15%)
    double clockScore = std::min(100.0, device.clockInfo.boostClock / 20.0);
    score += clockScore * 0.15;
    
    // 特性加分 (10%)
    double featureScore = 0.0;
    if (device.capabilities.supportsTensorCores) featureScore += 30.0;
    if (device.capabilities.supportsDoublePrecision) featureScore += 20.0;
    if (device.capabilities.supportsUnifiedMemory) featureScore += 20.0;
    if (device.capabilities.supportsConcurrentKernels) featureScore += 15.0;
    if (device.capabilities.supportsAsyncTransfer) featureScore += 15.0;
    score += std::min(100.0, featureScore) * 0.1;
    
    return static_cast<int>(std::round(score));
}

GPUConfiguration UnifiedGPUManager::createCPUFallbackConfig() {
    GPUConfiguration config;
    
    // 创建虚拟CPU设备信息
    GPUDeviceInfo cpuDevice;
    cpuDevice.deviceId = -1;
    cpuDevice.name = "CPU Fallback";
    cpuDevice.vendor = GPUVendor::UNKNOWN;
    cpuDevice.performanceScore = 10;
    
    config.primaryDevice = cpuDevice;
    config.computeAPI = ComputeAPI::AUTO_DETECT;
    config.enableMultiGPU = false;
    config.enablePeerAccess = false;
    
    return config;
}

void UnifiedGPUManager::notifyEvent(int deviceId, const std::string& event) {
    for (const auto& callback : m_eventCallbacks) {
        try {
            callback(deviceId, event);
        } catch (const std::exception& e) {
            // OSCEAN_LOG_ERROR("UnifiedGPUManager", "Event callback error: {}", e.what()); // TODO: Fix log format
        }
    }
}

// === 状态监控方法（基础实现） ===
boost::optional<GPUMemoryInfo> UnifiedGPUManager::getMemoryInfo(int deviceId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return boost::none;
    }
    
    GPUMemoryInfo info;
    info.totalMemory = m_devices[deviceId].memoryDetails.totalGlobalMemory;
    info.freeMemory = m_devices[deviceId].memoryDetails.freeGlobalMemory;
    info.usedMemory = info.totalMemory - info.freeMemory;
    
    return info;
}

boost::optional<GPUThermalInfo> UnifiedGPUManager::getThermalInfo(int deviceId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return boost::none;
    }
    
    return m_devices[deviceId].thermalInfo;
}

boost::optional<GPUPowerInfo> UnifiedGPUManager::getPowerInfo(int deviceId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return boost::none;
    }
    
    return m_devices[deviceId].powerInfo;
}

boost::optional<float> UnifiedGPUManager::getDeviceUtilization(int deviceId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return boost::none;
    }
    
    // 基础实现：基于内存使用估算利用率
    const auto& device = m_devices[deviceId];
    float memoryUtilization = 1.0f - (static_cast<float>(device.memoryDetails.freeGlobalMemory) / 
                                      static_cast<float>(device.memoryDetails.totalGlobalMemory));
    
    // 返回内存利用率作为设备利用率的近似值
    // 实际的GPU核心利用率需要平台特定的API（NVML、ADL等）
    return memoryUtilization * 100.0f;
}

// === 多GPU支持（基础实现） ===
GPUError UnifiedGPUManager::enablePeerAccess(int srcDevice, int dstDevice) {
    if (srcDevice == dstDevice) {
        return GPUError::INVALID_DEVICE;
    }
    
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "Enabling peer access from device {} to device {}", srcDevice, dstDevice); // TODO: Fix log format
    
    // TODO: 实现平台特定的点对点访问
    return GPUError::SUCCESS;
}

GPUError UnifiedGPUManager::disablePeerAccess(int srcDevice, int dstDevice) {
    if (srcDevice == dstDevice) {
        return GPUError::INVALID_DEVICE;
    }
    
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "Disabling peer access from device {} to device {}", srcDevice, dstDevice); // TODO: Fix log format
    
    // TODO: 实现平台特定的点对点访问禁用
    return GPUError::SUCCESS;
}

bool UnifiedGPUManager::canAccessPeer(int srcDevice, int dstDevice) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (srcDevice == dstDevice) {
        return false;
    }
    
    if (srcDevice < 0 || srcDevice >= static_cast<int>(m_devices.size()) ||
        dstDevice < 0 || dstDevice >= static_cast<int>(m_devices.size())) {
        return false;
    }
    
    // 检查是否为同一厂商的GPU
    if (m_devices[srcDevice].vendor != m_devices[dstDevice].vendor) {
        return false;
    }
    
    // NVIDIA GPU通常支持同一系统内的P2P访问
    if (m_devices[srcDevice].vendor == GPUVendor::NVIDIA) {
        // 简化实现：假设同一系统内的NVIDIA GPU可以互相访问
        // 实际需要调用cudaDeviceCanAccessPeer
        return true;
    }
    
    // 其他厂商暂不支持
    return false;
}

// === 事件和回调 ===
void UnifiedGPUManager::registerEventCallback(GPUEventCallback callback) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_eventCallbacks.push_back(callback);
}

void UnifiedGPUManager::clearEventCallbacks() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_eventCallbacks.clear();
}

// === 诊断和调试 ===
std::string UnifiedGPUManager::getDeviceDiagnostics(int deviceId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return "Invalid device ID";
    }
    
    const auto& device = m_devices[deviceId];
    std::stringstream ss;
    
    ss << "=== GPU Device Diagnostics ===" << std::endl;
    ss << "Device ID: " << device.deviceId << std::endl;
    ss << "Name: " << device.name << std::endl;
    ss << "Vendor: " << device.vendorToString(device.vendor) << std::endl;
    ss << "Driver: " << device.driverVersion << std::endl;
    ss << "PCIe Bus: " << device.pcieBusId << std::endl;
    ss << "Architecture: " << device.architecture.name << std::endl;
    ss << "Performance Score: " << device.performanceScore << "/100" << std::endl;
    ss << std::endl;
    
    ss << "=== Compute Units ===" << std::endl;
    ss << "Multiprocessors: " << device.computeUnits.multiprocessorCount << std::endl;
    ss << "Total Cores: " << device.computeUnits.totalCores << std::endl;
    ss << "Tensor Cores: " << device.computeUnits.tensorCores << std::endl;
    ss << std::endl;
    
    ss << "=== Memory ===" << std::endl;
    ss << "Total: " << (device.memoryDetails.totalGlobalMemory / (1024*1024*1024)) << " GB" << std::endl;
    ss << "Free: " << (device.memoryDetails.freeGlobalMemory / (1024*1024*1024)) << " GB" << std::endl;
    ss << "Bandwidth: " << device.memoryDetails.memoryBandwidth << " GB/s" << std::endl;
    ss << std::endl;
    
    ss << "=== Supported APIs ===" << std::endl;
    for (const auto& api : device.supportedAPIs) {
        ss << "  - " << computeAPIToString(api) << std::endl;
    }
    
    return ss.str();
}

GPUError UnifiedGPUManager::runDeviceSelfTest(int deviceId) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return GPUError::INVALID_DEVICE;
    }
    
    OSCEAN_LOG_INFO("UnifiedGPUManager", "Running self-test on device " + std::to_string(deviceId));
    
    const auto& device = m_devices[deviceId];
    
    // 基础自检实现
    // 1. 检查设备是否可用
    if (device.memoryDetails.totalGlobalMemory == 0) {
        OSCEAN_LOG_ERROR("UnifiedGPUManager", "Device memory size is 0");
        return GPUError::INVALID_DEVICE;
    }
    
    // 2. 检查计算单元
    if (device.computeUnits.multiprocessorCount == 0) {
        OSCEAN_LOG_ERROR("UnifiedGPUManager", "No compute units found");
        return GPUError::INVALID_DEVICE;
    }
    
    // 3. 检查支持的API
    if (device.supportedAPIs.empty()) {
        OSCEAN_LOG_ERROR("UnifiedGPUManager", "No supported APIs");
        return GPUError::NOT_SUPPORTED;
    }
    
    // 4. 检查性能评分
    if (device.performanceScore <= 0) {
        OSCEAN_LOG_WARN("UnifiedGPUManager", "Performance score is too low");
    }
    
    OSCEAN_LOG_INFO("UnifiedGPUManager", "Device self-test passed");
    notifyEvent(deviceId, "self_test_passed");
    
    return GPUError::SUCCESS;
}

std::string UnifiedGPUManager::getStatusReport() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    std::stringstream ss;
    ss << "=== GPU Manager Status Report ===" << std::endl;
    ss << "Initialized: " << (m_initialized ? "Yes" : "No") << std::endl;
    ss << "Device Count: " << m_devices.size() << std::endl;
    ss << "Current Device: " << m_currentDevice << std::endl;
    ss << "Multi-GPU: " << (m_options.enableMultiGPU ? "Enabled" : "Disabled") << std::endl;
    ss << "Preferred API: " << computeAPIToString(m_options.preferredAPI) << std::endl;
    ss << std::endl;
    
    if (!m_devices.empty()) {
        ss << "=== Detected Devices ===" << std::endl;
        for (const auto& device : m_devices) {
            ss << "[" << device.deviceId << "] " << device.getDescription() 
               << " (Score: " << device.performanceScore << ")" << std::endl;
        }
    }
    
    return ss.str();
}

// === 资源清理 ===
GPUError UnifiedGPUManager::resetDevice(int deviceId) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (deviceId < 0 || deviceId >= static_cast<int>(m_devices.size())) {
        return GPUError::INVALID_DEVICE;
    }
    
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "Resetting device {}", deviceId); // TODO: Fix log format
    
    // TODO: 实现平台特定的设备重置
    
    notifyEvent(deviceId, "device_reset");
    return GPUError::SUCCESS;
}

void UnifiedGPUManager::cleanup() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_initialized) {
        return;
    }
    
    // OSCEAN_LOG_INFO("UnifiedGPUManager", "Cleaning up GPU Manager"); // TODO: Fix log format
    
    // 禁用所有点对点访问
    if (m_options.enablePeerAccess) {
        for (size_t i = 0; i < m_devices.size(); ++i) {
            for (size_t j = i + 1; j < m_devices.size(); ++j) {
                disablePeerAccess(static_cast<int>(i), static_cast<int>(j));
                disablePeerAccess(static_cast<int>(j), static_cast<int>(i));
            }
        }
    }
    
    // 清理设备列表
    m_devices.clear();
    m_currentDevice = -1;
    m_initialized = false;
    
    // 清理回调
    m_eventCallbacks.clear();
}

// === 平台特定检测方法 ===
// NVIDIA GPU检测在cuda_device_detector.cpp中实现
// AMD GPU检测在rocm_device_detector.cpp中实现  
// Intel GPU通过OpenCL检测
// Apple GPU需要Metal API（macOS特定）
// OpenCL检测在opencl_device_detector.cpp中实现

std::vector<GPUDeviceInfo> UnifiedGPUManager::detectNVIDIAGPUs() {
#ifdef OSCEAN_CUDA_ENABLED
    try {
        return detectNVIDIAGPUsImpl();
    } catch (const std::exception& e) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", std::string("NVIDIA detection implementation error: ") + e.what());
        return std::vector<GPUDeviceInfo>();
    }
#else
    OSCEAN_LOG_DEBUG("UnifiedGPUManager", "CUDA support not enabled");
    return std::vector<GPUDeviceInfo>();
#endif
}

std::vector<GPUDeviceInfo> UnifiedGPUManager::detectAMDGPUs() {
    try {
        return detectAMDGPUsImpl();
    } catch (const std::exception& e) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", std::string("AMD detection implementation error: ") + e.what());
        return std::vector<GPUDeviceInfo>();
    }
}

std::vector<GPUDeviceInfo> UnifiedGPUManager::detectOpenCLGPUs() {
#ifdef OSCEAN_OPENCL_ENABLED
    try {
        return detectOpenCLGPUsImpl();
    } catch (const std::exception& e) {
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", std::string("OpenCL detection implementation error: ") + e.what());
        return std::vector<GPUDeviceInfo>();
    }
#else
    OSCEAN_LOG_DEBUG("UnifiedGPUManager", "OpenCL support not enabled");
    return std::vector<GPUDeviceInfo>();
#endif
}

std::vector<GPUDeviceInfo> UnifiedGPUManager::detectIntelGPUs() {
    // Intel GPU检测通过OpenCL实现
    // Intel GPU通常通过OpenCL API检测，不需要单独的Level Zero实现
    OSCEAN_LOG_DEBUG("UnifiedGPUManager", "Intel GPU detection delegated to OpenCL");
    
    // Intel GPU会在OpenCL检测中被识别
    // 这里返回空列表避免重复
    return std::vector<GPUDeviceInfo>();
}

std::vector<GPUDeviceInfo> UnifiedGPUManager::detectAppleGPUs() {
    // Apple Metal检测实现
    #ifdef __APPLE__
        // macOS平台特定的Metal检测
        OSCEAN_LOG_DEBUG("UnifiedGPUManager", "Detecting Apple Metal GPUs...");
        
        // TODO: 实现Metal API调用
        // 需要Objective-C++和Metal框架
        return std::vector<GPUDeviceInfo>();
    #else
        // 非macOS平台返回空列表
        return std::vector<GPUDeviceInfo>();
    #endif
}

} // namespace oscean::common_utils::gpu 