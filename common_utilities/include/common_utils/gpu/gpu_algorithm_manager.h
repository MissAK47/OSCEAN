#pragma once

#include "gpu_algorithm_interface.h"
#include "gpu_device_info.h"
#include "oscean_gpu_framework.h"
#include <boost/shared_ptr.hpp>
#include <boost/thread/future.hpp>
#include <map>
#include <string>
#include <vector>

namespace oscean {
namespace common_utils {
namespace gpu {

/**
 * @brief GPU算法注册信息
 */
struct GPUAlgorithmInfo {
    std::string name;
    std::string version;
    std::vector<ComputeAPI> supportedAPIs;
    std::string category;  // 例如: "visualization", "interpolation", "spatial"
    std::string description;
};

/**
 * @brief GPU算法工厂函数类型
 */
template<typename InputType, typename OutputType>
using GPUAlgorithmFactory = std::function<
    boost::shared_ptr<IGPUAlgorithm<InputType, OutputType>>(ComputeAPI)
>;

/**
 * @brief 统一GPU算法管理器
 * 负责管理所有GPU算法的注册、创建和执行
 */
class GPUAlgorithmManager {
public:
    /**
     * @brief 获取单例实例
     */
    static GPUAlgorithmManager& getInstance() {
        static GPUAlgorithmManager instance;
        return instance;
    }
    
    /**
     * @brief 注册GPU算法
     * @tparam InputType 输入类型
     * @tparam OutputType 输出类型
     * @param info 算法信息
     * @param factory 算法工厂函数
     */
    template<typename InputType, typename OutputType>
    void registerAlgorithm(
        const GPUAlgorithmInfo& info,
        GPUAlgorithmFactory<InputType, OutputType> factory) {
        
        std::string key = createKey<InputType, OutputType>(info.name);
        m_algorithmInfos[key] = info;
        m_factories[key] = boost::any(factory);
        
        BOOST_LOG_TRIVIAL(info) << "Registered GPU algorithm: " << info.name
                               << " (version " << info.version << ")";
    }
    
    /**
     * @brief 创建GPU算法实例
     * @tparam InputType 输入类型
     * @tparam OutputType 输出类型
     * @param algorithmName 算法名称
     * @param api 计算API（可选，默认自动选择）
     * @return 算法实例
     */
    template<typename InputType, typename OutputType>
    boost::shared_ptr<IGPUAlgorithm<InputType, OutputType>> createAlgorithm(
        const std::string& algorithmName,
        ComputeAPI api = ComputeAPI::AUTO_DETECT) {
        
        std::string key = createKey<InputType, OutputType>(algorithmName);
        
        auto infoIt = m_algorithmInfos.find(key);
        if (infoIt == m_algorithmInfos.end()) {
            BOOST_LOG_TRIVIAL(error) << "Algorithm not found: " << algorithmName;
            return nullptr;
        }
        
        auto factoryIt = m_factories.find(key);
        if (factoryIt == m_factories.end()) {
            BOOST_LOG_TRIVIAL(error) << "Factory not found for: " << algorithmName;
            return nullptr;
        }
        
        // 如果是自动检测，选择最优API
        if (api == ComputeAPI::AUTO_DETECT) {
            api = selectBestAPI(infoIt->second);
        }
        
        // 检查API是否支持
        const auto& supportedAPIs = infoIt->second.supportedAPIs;
        if (std::find(supportedAPIs.begin(), supportedAPIs.end(), api) == supportedAPIs.end()) {
            BOOST_LOG_TRIVIAL(error) << "API not supported for algorithm: " << algorithmName;
            return nullptr;
        }
        
        try {
            auto factory = boost::any_cast<GPUAlgorithmFactory<InputType, OutputType>>(
                factoryIt->second);
            return factory(api);
        } catch (const boost::bad_any_cast& e) {
            BOOST_LOG_TRIVIAL(error) << "Type mismatch for algorithm: " << algorithmName;
            return nullptr;
        }
    }
    
    /**
     * @brief 执行GPU算法（同步）
     * @tparam InputType 输入类型
     * @tparam OutputType 输出类型
     * @param algorithmName 算法名称
     * @param input 输入数据
     * @param deviceId GPU设备ID（-1表示自动选择）
     * @return 执行结果
     */
    template<typename InputType, typename OutputType>
    OutputType execute(
        const std::string& algorithmName,
        const InputType& input,
        int deviceId = -1) {
        
        auto algorithm = createAlgorithm<InputType, OutputType>(algorithmName);
        if (!algorithm) {
            throw std::runtime_error("Failed to create algorithm: " + algorithmName);
        }
        
        // 创建执行上下文
        GPUExecutionContext context;
        context.deviceId = (deviceId >= 0) ? deviceId : selectOptimalDevice(algorithm, input);
        
        // 执行算法
        auto future = algorithm->executeAsync(input, context);
        return future.get();
    }
    
    /**
     * @brief 执行GPU算法（异步）
     * @tparam InputType 输入类型
     * @tparam OutputType 输出类型
     * @param algorithmName 算法名称
     * @param input 输入数据
     * @param deviceId GPU设备ID（-1表示自动选择）
     * @return Future对象
     */
    template<typename InputType, typename OutputType>
    boost::future<OutputType> executeAsync(
        const std::string& algorithmName,
        const InputType& input,
        int deviceId = -1) {
        
        auto algorithm = createAlgorithm<InputType, OutputType>(algorithmName);
        if (!algorithm) {
            boost::promise<OutputType> promise;
            promise.set_exception(
                std::runtime_error("Failed to create algorithm: " + algorithmName));
            return promise.get_future();
        }
        
        // 创建执行上下文
        GPUExecutionContext context;
        context.deviceId = (deviceId >= 0) ? deviceId : selectOptimalDevice(algorithm, input);
        
        // 异步执行算法
        return algorithm->executeAsync(input, context);
    }
    
    /**
     * @brief 获取所有注册的算法信息
     */
    std::vector<GPUAlgorithmInfo> getAllAlgorithms() const {
        std::vector<GPUAlgorithmInfo> algorithms;
        for (const auto& pair : m_algorithmInfos) {
            algorithms.push_back(pair.second);
        }
        return algorithms;
    }
    
    /**
     * @brief 获取特定类别的算法
     */
    std::vector<GPUAlgorithmInfo> getAlgorithmsByCategory(
        const std::string& category) const {
        
        std::vector<GPUAlgorithmInfo> algorithms;
        for (const auto& pair : m_algorithmInfos) {
            if (pair.second.category == category) {
                algorithms.push_back(pair.second);
            }
        }
        return algorithms;
    }
    
    /**
     * @brief 检查算法是否已注册
     */
    template<typename InputType, typename OutputType>
    bool isAlgorithmRegistered(const std::string& algorithmName) const {
        std::string key = createKey<InputType, OutputType>(algorithmName);
        return m_algorithmInfos.find(key) != m_algorithmInfos.end();
    }
    
private:
    GPUAlgorithmManager() = default;
    ~GPUAlgorithmManager() = default;
    
    // 禁止拷贝
    GPUAlgorithmManager(const GPUAlgorithmManager&) = delete;
    GPUAlgorithmManager& operator=(const GPUAlgorithmManager&) = delete;
    
    // 创建算法键值
    template<typename InputType, typename OutputType>
    std::string createKey(const std::string& name) const {
        return name + "_" + typeid(InputType).name() + "_" + typeid(OutputType).name();
    }
    
    // 选择最佳API
    ComputeAPI selectBestAPI(const GPUAlgorithmInfo& info) const {
        auto devices = OSCEANGPUFramework::getAvailableDevices();
        if (devices.empty()) {
            return ComputeAPI::AUTO_DETECT;
        }
        
        // 选择第一个可用设备的最佳API
        const auto& device = devices[0];
        for (const auto& api : info.supportedAPIs) {
            if (device.hasAPI(api)) {
                return api;
            }
        }
        
        return info.supportedAPIs.empty() ? ComputeAPI::AUTO_DETECT : info.supportedAPIs[0];
    }
    
    // 选择最优设备
    template<typename Algorithm, typename InputType>
    int selectOptimalDevice(
        const boost::shared_ptr<Algorithm>& algorithm,
        const InputType& input) const {
        
        auto& scheduler = OSCEANGPUFramework::getScheduler();
        size_t memoryReq = algorithm->estimateMemoryRequirement(input);
        
        // 简单的任务复杂度估算（可以根据实际情况改进）
        double complexity = 1.0;
        
        return scheduler.selectOptimalGPU(memoryReq, complexity);
    }
    
private:
    std::map<std::string, GPUAlgorithmInfo> m_algorithmInfos;
    std::map<std::string, boost::any> m_factories;
};

/**
 * @brief GPU算法自动注册器
 * 用于在程序启动时自动注册算法
 */
template<typename InputType, typename OutputType>
class GPUAlgorithmRegistrar {
public:
    GPUAlgorithmRegistrar(
        const GPUAlgorithmInfo& info,
        GPUAlgorithmFactory<InputType, OutputType> factory) {
        
        GPUAlgorithmManager::getInstance().registerAlgorithm<InputType, OutputType>(
            info, factory);
    }
};

// 辅助宏：简化算法注册
#define REGISTER_GPU_ALGORITHM(InputType, OutputType, AlgorithmClass, Name, Version, Category, Description, ...) \
    static GPUAlgorithmRegistrar<InputType, OutputType> g_##AlgorithmClass##Registrar( \
        GPUAlgorithmInfo{Name, Version, {__VA_ARGS__}, Category, Description}, \
        [](ComputeAPI api) -> boost::shared_ptr<IGPUAlgorithm<InputType, OutputType>> { \
            return boost::make_shared<AlgorithmClass>(api); \
        });

} // namespace gpu
} // namespace common_utils
} // namespace oscean 