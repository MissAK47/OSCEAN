// 🚀 使用Common模块的统一boost配置（参考CRS服务）
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 插值服务只使用boost::future，不使用boost::asio

// 首先包含项目接口定义（包含所有类型定义）
#include "core_services/interpolation/i_interpolation_service.h"
#include "core_services/interpolation/impl/algorithms/i_interpolation_algorithm.h"

// 然后包含Boost头文件
#include <boost/thread/future.hpp>
#include <boost/thread.hpp>

// 最后包含标准库头文件
#include <unordered_map>
#include <memory>
#include <chrono>
#include <algorithm>

// 包含算法实现
#include "algorithms/bilinear_interpolator.h"
#include "algorithms/pchip_interpolator.h"
#include "algorithms/nearest_neighbor_interpolator.h"
#include "algorithms/cubic_spline_interpolator.h"
#include "algorithms/trilinear_interpolator.h"
#include "algorithms/linear_1d_interpolator.h"
#include "algorithms/recursive_ndim_pchip_interpolator.h"
#include "algorithms/complex_field_interpolator.h"
#include "algorithms/fast_pchip_interpolator_2d.h"
#include "algorithms/fast_pchip_interpolator_3d.h"
#include "algorithms/pchip_interpolator_2d_bathy.h"

// GPU加速支持
#include "interpolation/gpu/gpu_interpolation_engine.h"
#include "interpolation/interpolation_method_mapping.h"
#include <common_utils/gpu/oscean_gpu_framework.h>
#include <boost/log/trivial.hpp>

#include "common_utils/simd/isimd_manager.h"
#include "common_utils/simd/simd_manager_unified.h"
#include "common_utils/simd/simd_config.h"

namespace oscean::core_services::interpolation {

/**
 * @brief 算法选择策略
 */
struct AlgorithmSelectionCriteria {
    size_t dataSize = 0;           // 数据点数量
    size_t dimensions = 2;         // 数据维度
    double noiseLevel = 0.0;       // 噪声水平估计
    bool preserveMonotonicity = false;  // 是否需要保持单调性
    bool highAccuracy = false;     // 是否需要高精度
    bool fastComputation = false;  // 是否需要快速计算
};

/**
 * @brief 完整的插值服务实现
 * @details 支持SIMD优化、智能算法选择、依赖注入的高性能插值服务
 */
class InterpolationServiceImpl : public IInterpolationService {
private:
    std::unordered_map<InterpolationMethod, std::unique_ptr<IInterpolationAlgorithm>> algorithms_;
    boost::shared_ptr<common_utils::simd::ISIMDManager> simdManager_;
    bool enableSmartSelection_;
    bool enableGPUAcceleration_;
    boost::shared_ptr<gpu::IGPUInterpolationEngine> gpuEngine_;
    bool m_initialized;
    bool m_gpuEnabled;
    InterpolationMethod m_defaultMethod;
    bool m_enableGPUAcceleration;
    size_t m_gpuMemoryLimit;
    size_t m_batchSize;
    bool m_enableCaching;
    size_t m_cacheSize;
    bool m_enableAutoMethodSelection;
    double m_performanceThreshold;
    bool m_enableAsyncProcessing;
    size_t m_threadPoolSize;
    bool m_enableMemoryOptimization;
    size_t m_memoryAlignmentBytes;
    bool m_enableSIMD;
    common_utils::simd::SIMDImplementation m_simdImplementation;
    boost::shared_ptr<common_utils::simd::ISIMDManager> m_simdManager;

public:
    /**
     * @brief 构造函数，支持依赖注入
     * @param simdManager SIMD管理器（可选）
     * @param enableSmartSelection 是否启用智能算法选择
     * @param enableGPUAcceleration 是否启用GPU加速
     */
    InterpolationServiceImpl(
        boost::shared_ptr<common_utils::simd::ISIMDManager> simdManager = nullptr,
        bool enableSmartSelection = true,
        bool enableGPUAcceleration = true)
        : simdManager_(std::move(simdManager))
        , enableSmartSelection_(enableSmartSelection)
        , enableGPUAcceleration_(enableGPUAcceleration)
        , m_initialized(false)
        , m_gpuEnabled(false)
        , m_defaultMethod(InterpolationMethod::BILINEAR)
        , m_enableGPUAcceleration(enableGPUAcceleration)
        , m_gpuMemoryLimit(1024 * 1024 * 1024) // 1GB
        , m_batchSize(16)
        , m_enableCaching(true)
        , m_cacheSize(100 * 1024 * 1024) // 100MB
        , m_enableAutoMethodSelection(true)
        , m_performanceThreshold(0.8)
        , m_enableAsyncProcessing(true)
        , m_threadPoolSize(4)
        , m_enableMemoryOptimization(true)
        , m_memoryAlignmentBytes(32)
        , m_enableSIMD(true)
        , m_simdImplementation(common_utils::simd::SIMDImplementation::AUTO_DETECT) {
        
        // 如果没有提供SIMD管理器，创建一个
        if (!simdManager_) {
            // 创建统一的SIMD管理器
            auto unifiedManager = boost::make_shared<common_utils::simd::UnifiedSIMDManager>();
            simdManager_ = boost::static_pointer_cast<common_utils::simd::ISIMDManager>(unifiedManager);
            m_simdManager = simdManager_;
        } else {
            m_simdManager = simdManager_;
        }
        
        registerAlgorithms();
        initializeGPUSupport();
        m_initialized = true;
    }

    ~InterpolationServiceImpl() override = default;

    boost::future<InterpolationResult> interpolateAsync(
        const InterpolationRequest& request) override {
        
        // 检查是否可以使用GPU加速
        if (shouldUseGPUAcceleration(request)) {
            return executeGPUInterpolation(request);
        }
        
        // 对于大数据集，使用异步SIMD批量处理
        if (shouldUseAsyncProcessing(request)) {
            return executeAsyncSIMDInterpolation(request);
        }
        
        // 对于小数据集，使用传统异步处理
        boost::promise<InterpolationResult> promise;
        
        try {
            // 智能算法选择（如果启用）
            InterpolationRequest optimizedRequest = request;
            if (enableSmartSelection_ && request.method == InterpolationMethod::UNKNOWN) {
                optimizedRequest.method = selectOptimalAlgorithm(request);
            }
            
            auto result = executeInterpolation(optimizedRequest);
            promise.set_value(std::move(result));
        } catch (const std::exception& e) {
            InterpolationResult errorResult;
            errorResult.statusCode = -1;
            errorResult.message = std::string("插值算法执行期间发生异常: ") + e.what();
            promise.set_value(std::move(errorResult));
        } catch (...) {
            InterpolationResult errorResult;
            errorResult.statusCode = -1;
            errorResult.message = "插值算法执行期间发生未知异常";
            promise.set_value(std::move(errorResult));
        }
        
        return promise.get_future();
    }

    std::vector<InterpolationMethod> getSupportedMethods() const override {
        // 使用统一的CPU支持方法列表
        return InterpolationMethodMapping::getCPUSupportedMethods();
    }

    /**
     * @brief 获取算法性能预测
     * @param request 插值请求
     * @param method 算法类型
     * @return 预估的计算时间（毫秒）和内存使用（MB）
     */
    std::pair<double, double> predictPerformance(
        const InterpolationRequest& request,
        InterpolationMethod method) const {
        
        size_t dataPoints = 0;
        if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
            dataPoints = std::get<std::vector<TargetPoint>>(request.target).size();
        } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
            const auto& gridDef = std::get<TargetGridDefinition>(request.target);
            for (const auto& dim : gridDef.dimensions) {
                dataPoints = std::max(dataPoints, dim.getNumberOfLevels());
            }
        }
        
        // 基于算法复杂度的性能预测
        double timeMs = 0.0;
        double memoryMB = 0.0;
        
        switch (method) {
            case InterpolationMethod::NEAREST_NEIGHBOR:
                timeMs = dataPoints * 0.001;  // O(n)
                memoryMB = dataPoints * 0.008 / 1024.0;
                break;
            case InterpolationMethod::BILINEAR:
                timeMs = dataPoints * 0.005;  // O(n)
                memoryMB = dataPoints * 0.032 / 1024.0;
                break;
            case InterpolationMethod::TRILINEAR:
                timeMs = dataPoints * 0.008;  // O(n)
                memoryMB = dataPoints * 0.064 / 1024.0;
                break;
            case InterpolationMethod::CUBIC_SPLINE_1D:
                timeMs = dataPoints * 0.02;   // O(n log n)
                memoryMB = dataPoints * 0.128 / 1024.0;
                break;
            case InterpolationMethod::PCHIP_RECURSIVE_NDIM:
                timeMs = dataPoints * 0.015;  // O(n log n)
                memoryMB = dataPoints * 0.096 / 1024.0;
                break;
            default:
                timeMs = dataPoints * 0.01;
                memoryMB = dataPoints * 0.064 / 1024.0;
                break;
        }
        
        // SIMD加速因子
        if (simdManager_ && dataPoints > 8) {
            timeMs *= 0.3;  // SIMD可以提供约3倍加速
        }
        
        return {timeMs, memoryMB};
    }

    /**
     * @brief 判断是否应该使用异步SIMD处理
     * @param request 插值请求
     * @return 如果数据量大且支持SIMD则返回true
     */
    bool shouldUseAsyncProcessing(const InterpolationRequest& request) const {
        if (!simdManager_) {
            return false;
        }
        
        size_t dataPoints = 0;
        if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
            dataPoints = std::get<std::vector<TargetPoint>>(request.target).size();
        } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
            const auto& gridDef = std::get<TargetGridDefinition>(request.target);
            for (const auto& dim : gridDef.dimensions) {
                dataPoints = std::max(dataPoints, dim.getNumberOfLevels());
            }
        }
        
        // 大于1000个点时使用异步SIMD处理
        return dataPoints > 1000;
    }

    /**
     * @brief 执行异步SIMD插值
     * @param request 插值请求
     * @return 异步插值结果
     */
    boost::future<InterpolationResult> executeAsyncSIMDInterpolation(
        const InterpolationRequest& request) {
        
        // 使用boost::async在线程池中执行
        return boost::async(boost::launch::async, [this, request]() -> InterpolationResult {
            InterpolationResult result;
            result.statusCode = -1;
            
            try {
                // 智能算法选择
                InterpolationRequest optimizedRequest = request;
                if (enableSmartSelection_ && request.method == InterpolationMethod::UNKNOWN) {
                    optimizedRequest.method = selectOptimalAlgorithm(request);
                }
                
                // 执行异步SIMD插值
                result = executeAsyncSIMDInterpolationImpl(optimizedRequest);
                
            } catch (const std::exception& e) {
                result.message = std::string("异步SIMD插值失败: ") + e.what();
            } catch (...) {
                result.message = "异步SIMD插值发生未知异常";
            }
            
            return result;
        });
    }

    /**
     * @brief 异步SIMD插值实现
     * @param request 插值请求
     * @return 插值结果
     */
    InterpolationResult executeAsyncSIMDInterpolationImpl(const InterpolationRequest& request) {
        InterpolationResult result;
        result.statusCode = -1;
        
        if (!request.sourceGrid) {
            result.message = "源网格数据为空";
            return result;
        }
        
        // 检查目标类型
        if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
            // 点插值的异步SIMD处理
            const auto& targetPoints = std::get<std::vector<TargetPoint>>(request.target);
            
            auto asyncResult = executeAsyncPointInterpolation(
                request.sourceGrid, targetPoints, request.method);
            
            // 等待异步结果
            auto values = asyncResult.get();
            result.data = std::move(values);
            result.statusCode = 0;
            result.message = "异步SIMD点插值成功完成";
            
        } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
            // 网格到网格的异步SIMD处理
            const auto& targetGridDef = std::get<TargetGridDefinition>(request.target);
            
            auto asyncResult = executeAsyncGridInterpolation(
                request.sourceGrid, targetGridDef, request.method);
            
            // 等待异步结果
            auto gridResult = asyncResult.get();
            result.data = std::move(gridResult);
            result.statusCode = 0;
            result.message = "异步SIMD网格插值成功完成";
            
        } else {
            result.message = "未知的目标类型";
        }
        
        return result;
    }

    /**
     * @brief 异步点插值处理
     * @param sourceGrid 源网格数据
     * @param targetPoints 目标点集合
     * @param method 插值方法
     * @return 异步插值结果
     */
    boost::future<std::vector<std::optional<double>>> executeAsyncPointInterpolation(
        boost::shared_ptr<GridData> sourceGrid,
        const std::vector<TargetPoint>& targetPoints,
        InterpolationMethod method) {
        
        // 分块处理大数据集
        const size_t chunkSize = simdManager_->getOptimalBatchSize() * 4; // 4倍批处理大小
        const size_t numChunks = (targetPoints.size() + chunkSize - 1) / chunkSize;
        
        // 多块并行处理
        return boost::async(boost::launch::async, [this, sourceGrid, &targetPoints, method, chunkSize, numChunks]() -> std::vector<std::optional<double>> {
            std::vector<boost::future<std::vector<std::optional<double>>>> chunkFutures;
            chunkFutures.reserve(numChunks);
            
            // 启动所有块的异步处理
            for (size_t chunk = 0; chunk < numChunks; ++chunk) {
                size_t startIdx = chunk * chunkSize;
                size_t endIdx = std::min(startIdx + chunkSize, targetPoints.size());
                
                std::vector<TargetPoint> chunkPoints(
                    targetPoints.begin() + startIdx,
                    targetPoints.begin() + endIdx
                );
                
                // 为每个块启动异步处理
                auto chunkFuture = boost::async(boost::launch::async, [this, sourceGrid, chunkPoints, method]() -> std::vector<std::optional<double>> {
                    // 查找算法并执行
                    auto it = algorithms_.find(method);
                    if (it != algorithms_.end()) {
                        InterpolationRequest chunkRequest;
                        // 使用传入的sourceGrid shared_ptr
                        chunkRequest.sourceGrid = sourceGrid;
                        chunkRequest.target = chunkPoints;
                        chunkRequest.method = method;
                        
                        // TODO: 需要修改算法实现，让它们能够接受GridData的引用而不是shared_ptr
                        auto chunkResult = it->second->execute(chunkRequest);
                        if (chunkResult.statusCode == 0 && 
                            std::holds_alternative<std::vector<std::optional<double>>>(chunkResult.data)) {
                            return std::get<std::vector<std::optional<double>>>(chunkResult.data);
                        }
                    }
                    
                    // 失败时返回空结果
                    return std::vector<std::optional<double>>(chunkPoints.size(), std::nullopt);
                });
                
                chunkFutures.push_back(std::move(chunkFuture));
            }
            
            // 收集所有块的结果
            std::vector<std::optional<double>> finalResults;
            finalResults.reserve(targetPoints.size());
            
            for (auto& chunkFuture : chunkFutures) {
                auto chunkResults = chunkFuture.get();
                finalResults.insert(finalResults.end(), chunkResults.begin(), chunkResults.end());
            }
            
            return finalResults;
        });
    }

    /**
     * @brief 异步网格插值处理
     * @param sourceGrid 源网格数据
     * @param targetGridDef 目标网格定义
     * @param method 插值方法
     * @return 异步插值结果
     */
    boost::future<GridData> executeAsyncGridInterpolation(
        boost::shared_ptr<GridData> sourceGrid,
        const TargetGridDefinition& targetGridDef,
        InterpolationMethod method) {
        
        return boost::async(boost::launch::async, [this, sourceGrid, &targetGridDef, method]() -> GridData {
            // 查找算法并执行网格到网格插值
            auto it = algorithms_.find(method);
            if (it != algorithms_.end()) {
                InterpolationRequest gridRequest;
                // 使用传入的sourceGrid shared_ptr
                gridRequest.sourceGrid = sourceGrid;
                gridRequest.target = targetGridDef;
                gridRequest.method = method;
                
                auto gridResult = it->second->execute(gridRequest);
                if (gridResult.statusCode == 0 && 
                    std::holds_alternative<GridData>(gridResult.data)) {
                    return std::move(std::get<GridData>(gridResult.data));
                }
            }
            
            // 失败时返回空网格
            GridDefinition emptyDef;
            return GridData(emptyDef, DataType::Float64, 1);
        });
    }

private:
    void registerAlgorithms() {
        // 注册所有算法，传入SIMD管理器
        algorithms_[InterpolationMethod::BILINEAR] = 
            std::make_unique<BilinearInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::TRILINEAR] = 
            std::make_unique<TrilinearInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::NEAREST_NEIGHBOR] = 
            std::make_unique<NearestNeighborInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::CUBIC_SPLINE_1D] = 
            std::make_unique<CubicSplineInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::PCHIP_RECURSIVE_NDIM] = 
            std::make_unique<RecursiveNDimPCHIPInterpolator>(simdManager_);
        algorithms_[InterpolationMethod::LINEAR_1D] = 
            std::make_unique<Linear1DInterpolator>(simdManager_);
        
        // 注册特殊优化的算法
        // 注意：FastPchipInterpolator2D和FastPchipInterpolator3D需要sourceGrid参数
        // 这里暂时注释掉，需要在interpolate方法中动态创建
        // algorithms_[InterpolationMethod::PCHIP_FAST_2D] = 
        //     std::make_unique<FastPchipInterpolator2D>(sourceGrid, simdManager_);
        // algorithms_[InterpolationMethod::PCHIP_FAST_3D] = 
        //     std::make_unique<FastPchipInterpolator3D>(sourceGrid, simdManager_);
        // algorithms_[InterpolationMethod::PCHIP_OPTIMIZED_2D_BATHY] = 
        //     std::make_unique<PCHIPInterpolator2DBathy>(simdManager_);
        
        // 注册复数场插值器（用于RAM声场数据）
        algorithms_[InterpolationMethod::COMPLEX_FIELD_BILINEAR] = 
            std::make_unique<ComplexFieldInterpolator>(simdManager_, InterpolationMethod::BILINEAR);
        algorithms_[InterpolationMethod::COMPLEX_FIELD_BICUBIC] = 
            std::make_unique<ComplexFieldInterpolator>(simdManager_, InterpolationMethod::BICUBIC);
        algorithms_[InterpolationMethod::COMPLEX_FIELD_TRILINEAR] = 
            std::make_unique<ComplexFieldInterpolator>(simdManager_, InterpolationMethod::TRILINEAR);
        algorithms_[InterpolationMethod::COMPLEX_FIELD_PCHIP] = 
            std::make_unique<ComplexFieldInterpolator>(simdManager_, InterpolationMethod::PCHIP_FAST_2D);
    }

    /**
     * @brief 智能算法选择
     * @param request 插值请求
     * @return 推荐的算法类型
     */
    InterpolationMethod selectOptimalAlgorithm(const InterpolationRequest& request) const {
        AlgorithmSelectionCriteria criteria = analyzeCriteria(request);
        
        // 基于数据特征的智能选择逻辑
        if (criteria.dimensions == 1) {
            if (criteria.preserveMonotonicity) {
                return InterpolationMethod::PCHIP_RECURSIVE_NDIM;
            } else if (criteria.highAccuracy) {
                return InterpolationMethod::CUBIC_SPLINE_1D;
            } else {
                return InterpolationMethod::LINEAR_1D;
            }
        } else if (criteria.dimensions == 2) {
            if (criteria.fastComputation) {
                return InterpolationMethod::NEAREST_NEIGHBOR;
            } else if (criteria.highAccuracy) {
                return InterpolationMethod::BICUBIC;
            } else {
                return InterpolationMethod::BILINEAR;
            }
        } else if (criteria.dimensions == 3) {
            if (criteria.fastComputation) {
                return InterpolationMethod::NEAREST_NEIGHBOR;
            } else {
                return InterpolationMethod::TRILINEAR;
            }
        }
        
        // 默认选择双线性插值
        return InterpolationMethod::BILINEAR;
    }

    /**
     * @brief 分析数据特征
     * @param request 插值请求
     * @return 选择标准
     */
    AlgorithmSelectionCriteria analyzeCriteria(const InterpolationRequest& request) const {
        AlgorithmSelectionCriteria criteria;
        
        if (!request.sourceGrid) {
            return criteria;
        }
        
        const auto& gridDef = request.sourceGrid->getDefinition();
        
        // 分析数据维度
        criteria.dimensions = 2; // 默认2D
        if (request.sourceGrid->getBandCount() > 1) {
            criteria.dimensions = 3;
        }
        
        // 分析数据大小
        criteria.dataSize = gridDef.cols * gridDef.rows * request.sourceGrid->getBandCount();
        
        // 基于数据大小决定性能偏好
        if (criteria.dataSize > 1000000) {  // 大数据集
            criteria.fastComputation = true;
        } else if (criteria.dataSize < 10000) {  // 小数据集
            criteria.highAccuracy = true;
        }
        
        // 分析目标点数量
        if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
            const auto& points = std::get<std::vector<TargetPoint>>(request.target);
            if (points.size() > 100000) {
                criteria.fastComputation = true;
            }
        }
        
        return criteria;
    }

    InterpolationResult executeInterpolation(const InterpolationRequest& request) {
        InterpolationResult result;
        result.statusCode = -1; // 默认为失败
        
        // 验证请求
        if (!request.sourceGrid) {
            result.message = "源网格数据为空";
            return result;
        }
        
        // 查找算法
        auto it = algorithms_.find(request.method);
        if (it == algorithms_.end()) {
            result.message = "不支持的插值算法类型: " + std::to_string(static_cast<int>(request.method));
            return result;
        }
        
        // 执行插值
        try {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            result = it->second->execute(request);
            
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            if (result.statusCode == 0) {
                result.message += " (执行时间: " + std::to_string(duration.count()) + "ms)";
            }
        } catch (const std::exception& e) {
            result.statusCode = -1;
            result.message = std::string("算法执行失败: ") + e.what();
        }
        
        return result;
    }
    
    /**
     * @brief 初始化GPU支持
     */
    void initializeGPUSupport() {
        try {
            // 尝试初始化GPU框架
            if (common_utils::gpu::OSCEANGPUFramework::initialize()) {
                auto devices = common_utils::gpu::OSCEANGPUFramework::getAvailableDevices();
                if (!devices.empty()) {
                    m_gpuEnabled = true;
                    OSCEAN_LOG_INFO("InterpolationService", 
                        "GPU support initialized with {} device(s)", devices.size());
                    
                    // 创建GPU插值引擎
                    gpuEngine_ = gpu::GPUInterpolationEngineFactory::create(
                        common_utils::gpu::ComputeAPI::CUDA);
                    
                    // 注册GPU加速的算法
                    registerGPUAlgorithms();
                } else {
                    m_gpuEnabled = false;
                    OSCEAN_LOG_INFO("InterpolationService", 
                        "No GPU devices found, using CPU-only algorithms");
                }
            } else {
                m_gpuEnabled = false;
                OSCEAN_LOG_INFO("InterpolationService", 
                    "GPU framework initialization failed, using CPU-only algorithms");
            }
        } catch (const std::exception& e) {
            m_gpuEnabled = false;
            OSCEAN_LOG_WARN("InterpolationService", 
                "GPU initialization error: {}, using CPU-only algorithms", e.what());
        }
    }
    
    /**
     * @brief 判断是否应该使用GPU加速
     */
    bool shouldUseGPUAcceleration(const InterpolationRequest& request) const {
        // 首先检查GPU是否可用
        if (!m_gpuEnabled || !enableGPUAcceleration_) {
            return false;
        }
        
        // 检查是否为GPU支持的方法
        auto gpuMethods = InterpolationMethodMapping::getGPUSupportedMethods();
        if (std::find(gpuMethods.begin(), gpuMethods.end(), request.method) == gpuMethods.end()) {
            return false;
        }
        
        // 检查数据规模
        size_t dataSize = 0;
        if (request.sourceGrid) {
            dataSize = request.sourceGrid->getUnifiedBufferSize();
        }
        
        // 小数据集不值得使用GPU
        const size_t MIN_GPU_DATA_SIZE = 1024 * 1024; // 1MB
        if (dataSize < MIN_GPU_DATA_SIZE) {
            return false;
        }
        
        // 检查目标点数量
        size_t targetPoints = 0;
        if (std::holds_alternative<std::vector<TargetPoint>>(request.target)) {
            targetPoints = std::get<std::vector<TargetPoint>>(request.target).size();
        } else if (std::holds_alternative<TargetGridDefinition>(request.target)) {
            const auto& gridDef = std::get<TargetGridDefinition>(request.target);
            targetPoints = 1;
            for (const auto& dim : gridDef.dimensions) {
                targetPoints *= dim.coordinates.size();
            }
        }
        
        // 目标点太少不值得使用GPU
        const size_t MIN_GPU_TARGET_POINTS = 10000;
        if (targetPoints < MIN_GPU_TARGET_POINTS) {
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief 执行GPU插值
     */
    boost::future<InterpolationResult> executeGPUInterpolation(
        const InterpolationRequest& request) {
        
        return boost::async(boost::launch::async, [this, request]() -> InterpolationResult {
            InterpolationResult result;
            
            try {
                // 准备GPU插值参数
                gpu::GPUInterpolationParams gpuParams;
                // 直接使用原始的shared_ptr，避免引用计数问题
                gpuParams.sourceData = request.sourceGrid;
                gpuParams.method = static_cast<gpu::InterpolationMethod>(request.method);
                gpuParams.fillValue = 0.0f;
                
                // 处理不同的目标类型
                if (std::holds_alternative<TargetGridDefinition>(request.target)) {
                    const auto& targetGrid = std::get<TargetGridDefinition>(request.target);
                    
                    // 假设2D插值
                    if (targetGrid.dimensions.size() >= 2) {
                        gpuParams.outputWidth = targetGrid.dimensions[0].getNumberOfLevels();
                        gpuParams.outputHeight = targetGrid.dimensions[1].getNumberOfLevels();
                        
                        // 计算输出边界
                        gpuParams.outputBounds.minX = targetGrid.dimensions[0].minValue;
                        gpuParams.outputBounds.maxX = targetGrid.dimensions[0].maxValue;
                        gpuParams.outputBounds.minY = targetGrid.dimensions[1].minValue;
                        gpuParams.outputBounds.maxY = targetGrid.dimensions[1].maxValue;
                    }
                    
                    if (targetGrid.fillValue.has_value()) {
                        gpuParams.fillValue = static_cast<float>(targetGrid.fillValue.value());
                    }
                } else {
                    // 点插值暂时不支持GPU
                    return executeInterpolation(request);
                }
                
                // 执行GPU插值
                common_utils::gpu::GPUExecutionContext context;
                context.deviceId = 0;
                auto gpuAlgoResult = gpuEngine_->execute(gpuParams, context);
                
                if (gpuAlgoResult.success && gpuAlgoResult.error == common_utils::gpu::GPUError::SUCCESS) {
                    // 从GPUAlgorithmResult中提取GPUInterpolationResult
                    const auto& gpuResult = gpuAlgoResult.data;
                    // 转换结果为GridData
                    GridDefinition outputDef;
                    outputDef.rows = gpuResult.height;
                    outputDef.cols = gpuResult.width;
                    outputDef.extent = gpuParams.outputBounds;
                    
                    GridData outputGrid(outputDef, DataType::Float32, 1);
                    
                    // 复制数据
                    float* outputPtr = static_cast<float*>(const_cast<void*>(outputGrid.getDataPtr()));
                    std::copy(gpuResult.interpolatedData.begin(), 
                             gpuResult.interpolatedData.end(), 
                             outputPtr);
                    
                    result.data = std::move(outputGrid);
                    result.statusCode = 0;
                    result.message = "GPU插值成功完成 (GPU时间: " + 
                                   std::to_string(gpuResult.gpuTimeMs) + "ms)";
                } else {
                    result.statusCode = -1;
                    result.message = "GPU插值失败";
                }
                
            } catch (const std::exception& e) {
                result.statusCode = -1;
                result.message = std::string("GPU插值异常: ") + e.what();
            }
            
            return result;
        });
    }

    void registerGPUAlgorithms() {
        // 注册GPU加速的算法
        // 实现GPU加速算法的注册逻辑
    }
};

} // namespace oscean::core_services::interpolation 