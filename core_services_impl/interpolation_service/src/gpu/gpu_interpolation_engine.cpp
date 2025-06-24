/**
 * @file gpu_interpolation_engine.cpp
 * @brief GPU加速的插值引擎实现
 */

#include "interpolation/gpu/gpu_interpolation_engine.h"
#include "interpolation/interpolation_method_mapping.h"
#include <common_utils/gpu/multi_gpu_memory_manager.h>
#include <common_utils/gpu/multi_gpu_scheduler.h>
#include <common_utils/gpu/oscean_gpu_framework.h>
#include <common_utils/utilities/logging_utils.h>
#include <boost/make_shared.hpp>
#include <boost/thread/future.hpp>
#include <cstring>
#include <queue>
#include <mutex>
#include <condition_variable>

#ifdef OSCEAN_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda.h>

// CUDA核函数的外部声明（全局作用域）
extern "C" {
    cudaError_t launchBilinearInterpolation(
        const float* d_sourceData,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
    
    cudaError_t launchBicubicInterpolation(
        const float* d_sourceData,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
    
    cudaError_t computePCHIPDerivatives(
        const float* d_data,
        float* d_derivX,
        float* d_derivY,
        float* d_derivXY,
        int width, int height,
        cudaStream_t stream);
    
    cudaError_t computePCHIPDerivativesOptimized(
        const float* d_data,
        float* d_derivX,
        float* d_derivY,
        float* d_derivXY,
        int width, int height,
        float dx, float dy,
        cudaStream_t stream);
    
    cudaError_t launchPCHIP2DInterpolation(
        const float* d_sourceData,
        const float* d_derivX,
        const float* d_derivY,
        const float* d_derivXY,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
    
    cudaError_t launchPCHIP2DInterpolationOptimized(
        const float* d_sourceData,
        const float* d_derivX,
        const float* d_derivY,
        const float* d_derivXY,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
    
    cudaError_t launchPCHIP2DInterpolationIntegrated(
        const float* d_sourceData,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
    
    cudaError_t launchTrilinearInterpolation(
        const float* d_sourceData,
        float* d_outputData,
        int sourceWidth, int sourceHeight, int sourceDepth,
        int outputWidth, int outputHeight, int outputDepth,
        float minX, float maxX,
        float minY, float maxY,
        float minZ, float maxZ,
        float fillValue,
        cudaStream_t stream);
    
    cudaError_t launchNearestNeighborInterpolation(
        const float* d_sourceData,
        float* d_outputData,
        int sourceWidth, int sourceHeight,
        int outputWidth, int outputHeight,
        float minX, float maxX,
        float minY, float maxY,
        float fillValue,
        cudaStream_t stream);
    
    // 批量处理函数
    cudaError_t launchBatchBilinearInterpolation(
        const float** d_sourceArrays,
        float** d_outputArrays,
        const int* d_sourceWidths,
        const int* d_sourceHeights,
        const int* d_outputWidths,
        const int* d_outputHeights,
        const float* d_minXArray,
        const float* d_maxXArray,
        const float* d_minYArray,
        const float* d_maxYArray,
        const float* d_fillValues,
        int batchSize,
        cudaStream_t stream);
    
    cudaError_t launchBatchBilinearInterpolationOptimized(
        const float** d_sourceArrays,
        float** d_outputArrays,
        const int* d_sourceWidths,
        const int* d_sourceHeights,
        const int* d_outputWidths,
        const int* d_outputHeights,
        const float* d_scaleX,
        const float* d_scaleY,
        const float* d_offsetX,
        const float* d_offsetY,
        const float* d_fillValues,
        int batchSize,
        int maxOutputWidth,
        int maxOutputHeight,
        cudaStream_t stream);
}
#endif

namespace oscean {
namespace core_services {
namespace interpolation {
namespace gpu {

using namespace common_utils::gpu;

// GPU核函数将在特定实现文件中定义

/**
 * @brief 将InterpolationMethod转换为字符串
 */
static std::string interpolationMethodToString(InterpolationMethod method) {
    return InterpolationMethodMapping::toString(method);
}

/**
 * @brief 具体的GPU插值引擎实现
 */
class GPUInterpolationEngine : public common_utils::gpu::GPUAlgorithmBase<GPUInterpolationParams, GPUInterpolationResult>,
                               public IGPUInterpolationEngine {
private:
    InterpolationMethod m_currentMethod;
    std::vector<InterpolationMethod> m_supportedMethods;
    MultiGPUMemoryManager* m_memoryManager;
    OSCEANGPUFramework* m_gpuFramework;
    
    // CUDA核函数声明
    void launchBilinearKernel(const float* input, float* output, 
                             int srcWidth, int srcHeight,
                             int dstWidth, int dstHeight,
                             const GPUExecutionContext& context);
    
    void launchBicubicKernel(const float* input, float* output,
                            int srcWidth, int srcHeight,
                            int dstWidth, int dstHeight,
                            const GPUExecutionContext& context);
    
    void launchPCHIPKernel(const float* input, float* output,
                          int srcWidth, int srcHeight,
                          int dstWidth, int dstHeight,
                          const GPUExecutionContext& context);
    
    void launchTrilinearKernel(const float* input, float* output,
                              int srcWidth, int srcHeight,
                              int dstWidth, int dstHeight,
                              const GPUExecutionContext& context);
    
    void launchNearestNeighborKernel(const float* input, float* output,
                          int srcWidth, int srcHeight,
                          int dstWidth, int dstHeight,
                          const GPUExecutionContext& context);
    
public:
    GPUInterpolationEngine()
        : GPUAlgorithmBase("GPUInterpolation", "1.0.0", 
            {ComputeAPI::CUDA, ComputeAPI::OPENCL})
        , m_currentMethod(InterpolationMethod::BILINEAR) {
        
        // 初始化支持的方法
        m_supportedMethods = {
            InterpolationMethod::BILINEAR,
            InterpolationMethod::BICUBIC,
            InterpolationMethod::TRILINEAR,
            InterpolationMethod::PCHIP_FAST_2D,
            InterpolationMethod::NEAREST_NEIGHBOR
        };
    
        // 获取全局管理器
        m_memoryManager = &GlobalMemoryManager::getInstance();
        m_gpuFramework = &OSCEANGPUFramework::getInstance();
    }
    
    /**
     * @brief 设置插值方法
     */
    void setInterpolationMethod(InterpolationMethod method) override {
        m_currentMethod = method;
    }
    
    /**
     * @brief 获取支持的插值方法
     */
    std::vector<InterpolationMethod> getSupportedMethods() const override {
        // 使用统一的GPU支持方法列表
        return InterpolationMethodMapping::getGPUSupportedMethods();
    }
    
    /**
     * @brief 验证插值参数
     */
    bool validateParams(const GPUInterpolationParams& params) const override {
        if (!params.sourceData || params.sourceData->getUnifiedBufferSize() == 0) {
            OSCEAN_LOG_ERROR("GPUInterpolationEngine", "源数据为空");
            return false;
        }
        
        if (params.outputWidth <= 0 || params.outputHeight <= 0) {
            OSCEAN_LOG_ERROR("GPUInterpolationEngine", "输出尺寸无效");
            return false;
        }
        
        auto it = std::find(m_supportedMethods.begin(), m_supportedMethods.end(), params.method);
        if (it == m_supportedMethods.end()) {
            OSCEAN_LOG_ERROR("GPUInterpolationEngine", "不支持的插值方法");
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief 估算插值的内存需求
     */
    size_t estimateInterpolationMemory(
        int sourceWidth, int sourceHeight,
        int targetWidth, int targetHeight,
        InterpolationMethod method) const override {
        
        size_t sourceSize = sourceWidth * sourceHeight * sizeof(float);
        size_t targetSize = targetWidth * targetHeight * sizeof(float);
        
        // 根据方法估算额外内存
        size_t extraMemory = 0;
        switch (method) {
            case InterpolationMethod::BICUBIC:
            case InterpolationMethod::PCHIP_FAST_2D:
                // 需要额外的系数存储
                extraMemory = sourceSize * 0.5;
                break;
            case InterpolationMethod::TRILINEAR:
                // 3D插值需要更多内存
                extraMemory = sourceSize;
                break;
            default:
                break;
        }
        
        return sourceSize + targetSize + extraMemory;
    }
    
    /**
     * @brief 估算内存需求（IGPUAlgorithm接口）
     */
    size_t estimateMemoryRequirement(const GPUInterpolationParams& input) const override {
        auto& def = input.sourceData->getDefinition();
        int srcWidth = def.cols;
        int srcHeight = def.rows;
        
        return estimateInterpolationMemory(
            srcWidth, srcHeight,
            input.outputWidth, input.outputHeight,
            input.method
        );
    }
    
    /**
     * @brief 执行GPU插值（解决多继承歧义）
     */
    GPUAlgorithmResult<GPUInterpolationResult> execute(
        const GPUInterpolationParams& params,
        const GPUExecutionContext& context) override {
        // 调用基类的execute方法
        return GPUAlgorithmBase<GPUInterpolationParams, GPUInterpolationResult>::execute(params, context);
    }
    
protected:
    /**
     * @brief 内部执行方法
     */
    GPUAlgorithmResult<GPUInterpolationResult> executeInternal(
        const GPUInterpolationParams& params,
        const GPUExecutionContext& context) override {
        
        // 验证参数
        if (!validateParams(params)) {
            return createErrorResult(
                GPUError::UNKNOWN_ERROR,
                "无效的插值参数"
            );
        }
        
        // 创建性能统计
        typename GPUAlgorithmResult<GPUInterpolationResult>::PerformanceStats stats;
        
        try {
            // 准备输入数据
            const auto& sourceBuffer = params.sourceData->getUnifiedBuffer();
            auto& def = params.sourceData->getDefinition();
            int srcWidth = def.cols;
            int srcHeight = def.rows;
            
            // 初始化内存管理器（如果需要）
            if (!m_memoryManager || !OSCEANGPUFramework::isInitialized()) {
                OSCEAN_LOG_WARN("GPUInterpolationEngine", "内存管理器未初始化，尝试初始化");
                if (!OSCEANGPUFramework::initialize()) {
                    OSCEAN_LOG_ERROR("GPUInterpolationEngine", "GPU框架初始化失败");
                    return createErrorResult(
                        GPUError::INITIALIZATION_FAILED,
                        "GPU框架未初始化"
                    );
                }
                m_memoryManager = &GlobalMemoryManager::getInstance();
            }
            
            // 分配GPU内存 - 使用改进的分配策略
            AllocationRequest allocReq;
            allocReq.size = sourceBuffer.size();
            allocReq.preferredDeviceId = context.deviceId;
            allocReq.memoryType = params.useTextureMemory ? 
                GPUMemoryType::TEXTURE : GPUMemoryType::DEVICE;
            allocReq.strategy = AllocationStrategy::BEST_FIT;
            allocReq.allowFallback = true;
            allocReq.tag = "interpolation_input";
            
            ScopedTimer transferTimer(stats.transferTime);
            
            // 尝试使用内存管理器分配
            GPUMemoryHandle inputHandle;
            GPUMemoryHandle outputHandle;
            bool useMemoryManager = true;
            
            try {
                inputHandle = m_memoryManager->allocate(allocReq);
            if (!inputHandle.isValid()) {
                    throw std::runtime_error("输入内存分配失败");
            }
            
            // 分配输出内存
            allocReq.size = params.outputWidth * params.outputHeight * sizeof(float);
                allocReq.tag = "interpolation_output";
                outputHandle = m_memoryManager->allocate(allocReq);
            if (!outputHandle.isValid()) {
                m_memoryManager->deallocate(inputHandle);
                    throw std::runtime_error("输出内存分配失败");
            }
            
            // 传输输入数据到GPU
            TransferRequest transferReq;
                // 创建主机内存句柄
            GPUMemoryHandle hostHandle;
            hostHandle.devicePtr = const_cast<void*>(static_cast<const void*>(sourceBuffer.data()));
            hostHandle.size = sourceBuffer.size();
            hostHandle.memoryType = GPUMemoryType::HOST;
                hostHandle.deviceId = -1;  // 主机内存
            
            transferReq.source = hostHandle;
            transferReq.destination = inputHandle;
                transferReq.size = sourceBuffer.size();
                transferReq.async = false;  // 同步传输
            
            if (!m_memoryManager->transfer(transferReq)) {
                    throw std::runtime_error("数据传输失败");
                }
            } catch (const std::exception& e) {
                OSCEAN_LOG_WARN("GPUInterpolationEngine", 
                    "内存管理器操作失败: {}，回退到直接CUDA操作", e.what());
                useMemoryManager = false;
                
                // 清理已分配的内存
                if (inputHandle.isValid()) {
                m_memoryManager->deallocate(inputHandle);
                }
                if (outputHandle.isValid()) {
                m_memoryManager->deallocate(outputHandle);
                }
            }
            
            // 如果内存管理器失败，使用直接CUDA操作作为回退
            if (!useMemoryManager) {
#ifdef OSCEAN_CUDA_ENABLED
                // 直接使用CUDA内存分配
                float* d_input = nullptr;
                float* d_output = nullptr;
                cudaError_t err = cudaMalloc(&d_input, sourceBuffer.size());
                if (err != cudaSuccess) {
                    return createErrorResult(
                        GPUError::OUT_OF_MEMORY,
                        "CUDA内存分配失败: " + std::string(cudaGetErrorString(err))
                    );
                }
                
                err = cudaMalloc(&d_output, params.outputWidth * params.outputHeight * sizeof(float));
                if (err != cudaSuccess) {
                    cudaFree(d_input);
                    return createErrorResult(
                        GPUError::OUT_OF_MEMORY,
                        "输出内存分配失败: " + std::string(cudaGetErrorString(err))
                    );
                }
                
                // 传输数据
                err = cudaMemcpy(d_input, sourceBuffer.data(), sourceBuffer.size(), 
                               cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    cudaFree(d_input);
                    cudaFree(d_output);
                return createErrorResult(
                    GPUError::TRANSFER_FAILED,
                        "数据传输失败: " + std::string(cudaGetErrorString(err))
                );
            }
                
            transferTimer.~ScopedTimer();
            
            // 执行GPU核函数
            ScopedTimer kernelTimer(stats.kernelTime);
                
                // 根据插值方法调用相应的核函数
            switch (params.method) {
                case InterpolationMethod::BILINEAR:
                        launchBilinearKernel(d_input, d_output, srcWidth, srcHeight,
                                           params.outputWidth, params.outputHeight, context);
                    break;
                case InterpolationMethod::BICUBIC:
                        launchBicubicKernel(d_input, d_output, srcWidth, srcHeight,
                                          params.outputWidth, params.outputHeight, context);
                    break;
                case InterpolationMethod::PCHIP_FAST_2D:
                        launchPCHIPKernel(d_input, d_output, srcWidth, srcHeight,
                                        params.outputWidth, params.outputHeight, context);
                    break;
                    case InterpolationMethod::TRILINEAR:
                        launchTrilinearKernel(d_input, d_output, srcWidth, srcHeight,
                                            params.outputWidth, params.outputHeight, context);
                        break;
                    case InterpolationMethod::NEAREST_NEIGHBOR:
                        launchNearestNeighborKernel(d_input, d_output, srcWidth, srcHeight,
                                                  params.outputWidth, params.outputHeight, context);
                        break;
                    default:
                        cudaFree(d_input);
                        cudaFree(d_output);
                        return createErrorResult(
                            GPUError::NOT_SUPPORTED,
                            "GPU不支持该插值方法: " + interpolationMethodToString(params.method)
                        );
                }
                
                // 同步GPU
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    cudaFree(d_input);
                    cudaFree(d_output);
                    return createErrorResult(
                        GPUError::KERNEL_LAUNCH_FAILED,
                        "GPU核函数执行失败: " + std::string(cudaGetErrorString(err))
                    );
                }
                
                kernelTimer.~ScopedTimer();
                
                // 准备结果
                GPUInterpolationResult result;
                result.width = params.outputWidth;
                result.height = params.outputHeight;
                result.interpolatedData.resize(params.outputWidth * params.outputHeight);
                
                // 传输结果回主机
                ScopedTimer transferBackTimer(stats.transferTime);
                err = cudaMemcpy(result.interpolatedData.data(), d_output,
                               result.interpolatedData.size() * sizeof(float),
                               cudaMemcpyDeviceToHost);
                
                // 释放CUDA内存
                cudaFree(d_input);
                cudaFree(d_output);
                
                if (err != cudaSuccess) {
                    return createErrorResult(
                        GPUError::TRANSFER_FAILED,
                        "结果传输失败: " + std::string(cudaGetErrorString(err))
                    );
                }
                
                transferBackTimer.~ScopedTimer();
                
                // 计算结果统计
                result.gpuTimeMs = stats.kernelTime;
                result.memoryTransferTimeMs = stats.transferTime;
                result.memoryUsedBytes = sourceBuffer.size() + result.interpolatedData.size() * sizeof(float);
                
                // 计算数据统计
                float minVal = std::numeric_limits<float>::max();
                float maxVal = std::numeric_limits<float>::lowest();
                float sum = 0.0f;
                int nanCount = 0;
                
                for (const auto& val : result.interpolatedData) {
                    if (std::isnan(val)) {
                        nanCount++;
                    } else {
                        minVal = std::min(minVal, val);
                        maxVal = std::max(maxVal, val);
                        sum += val;
                    }
                }
                
                result.minValue = minVal;
                result.maxValue = maxVal;
                result.meanValue = sum / (result.interpolatedData.size() - nanCount);
                result.nanCount = nanCount;
                result.status = GPUError::SUCCESS;
                
                return createSuccessResult(std::move(result), stats);
#else
                return createErrorResult(
                    GPUError::NOT_SUPPORTED,
                    "CUDA未启用"
                );
#endif
            } else {
                // 使用内存管理器成功，执行GPU操作
                transferTimer.~ScopedTimer();
                
                // 执行GPU核函数
                ScopedTimer kernelTimer(stats.kernelTime);
                
                float* d_input = static_cast<float*>(inputHandle.devicePtr);
                float* d_output = static_cast<float*>(outputHandle.devicePtr);
                
                // 根据插值方法调用相应的核函数
                switch (params.method) {
                    case InterpolationMethod::BILINEAR:
                        launchBilinearKernel(d_input, d_output, srcWidth, srcHeight,
                                           params.outputWidth, params.outputHeight, context);
                        break;
                    case InterpolationMethod::BICUBIC:
                        launchBicubicKernel(d_input, d_output, srcWidth, srcHeight,
                                          params.outputWidth, params.outputHeight, context);
                        break;
                    case InterpolationMethod::PCHIP_FAST_2D:
                        launchPCHIPKernel(d_input, d_output, srcWidth, srcHeight,
                                        params.outputWidth, params.outputHeight, context);
                        break;
                    case InterpolationMethod::TRILINEAR:
                        launchTrilinearKernel(d_input, d_output, srcWidth, srcHeight,
                                            params.outputWidth, params.outputHeight, context);
                        break;
                    case InterpolationMethod::NEAREST_NEIGHBOR:
                        launchNearestNeighborKernel(d_input, d_output, srcWidth, srcHeight,
                                                  params.outputWidth, params.outputHeight, context);
                        break;
                default:
                    m_memoryManager->deallocate(inputHandle);
                    m_memoryManager->deallocate(outputHandle);
                    return createErrorResult(
                        GPUError::NOT_SUPPORTED,
                        "GPU不支持该插值方法: " + interpolationMethodToString(params.method)
                    );
            }
                
                // 同步GPU（通过内存管理器）
                if (!m_memoryManager->synchronize(5000)) {  // 5秒超时
                    m_memoryManager->deallocate(inputHandle);
                    m_memoryManager->deallocate(outputHandle);
                    return createErrorResult(
                        GPUError::KERNEL_LAUNCH_FAILED,
                        "GPU同步超时"
                    );
                }
                
            kernelTimer.~ScopedTimer();
            
            // 准备结果
            GPUInterpolationResult result;
            result.width = params.outputWidth;
            result.height = params.outputHeight;
            result.interpolatedData.resize(params.outputWidth * params.outputHeight);
                
                // 创建主机内存句柄用于接收结果
                GPUMemoryHandle hostResultHandle;
                hostResultHandle.devicePtr = result.interpolatedData.data();
                hostResultHandle.size = result.interpolatedData.size() * sizeof(float);
                hostResultHandle.memoryType = GPUMemoryType::HOST;
                hostResultHandle.deviceId = -1;
            
            // 传输结果回主机
                TransferRequest transferBackReq;
                transferBackReq.source = outputHandle;
                transferBackReq.destination = hostResultHandle;
                transferBackReq.size = hostResultHandle.size;
                transferBackReq.async = false;
                
                ScopedTimer transferBackTimer(stats.transferTime);
                if (!m_memoryManager->transfer(transferBackReq)) {
                m_memoryManager->deallocate(inputHandle);
                m_memoryManager->deallocate(outputHandle);
                return createErrorResult(
                    GPUError::TRANSFER_FAILED,
                    "结果传输失败"
                );
            }
            transferBackTimer.~ScopedTimer();
            
            // 释放GPU内存
            m_memoryManager->deallocate(inputHandle);
            m_memoryManager->deallocate(outputHandle);
            
                // 计算结果统计
            result.gpuTimeMs = stats.kernelTime;
            result.memoryTransferTimeMs = stats.transferTime;
                result.memoryUsedBytes = sourceBuffer.size() + result.interpolatedData.size() * sizeof(float);
            
            // 计算数据统计
            float minVal = std::numeric_limits<float>::max();
            float maxVal = std::numeric_limits<float>::lowest();
            float sum = 0.0f;
            int nanCount = 0;
            
            for (const auto& val : result.interpolatedData) {
                if (std::isnan(val)) {
                    nanCount++;
                } else {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                    sum += val;
                }
            }
            
            result.minValue = minVal;
            result.maxValue = maxVal;
            result.meanValue = sum / (result.interpolatedData.size() - nanCount);
            result.nanCount = nanCount;
            result.status = GPUError::SUCCESS;
            
            return createSuccessResult(std::move(result), stats);
            }
        } catch (const std::exception& e) {
            return createErrorResult(
                GPUError::UNKNOWN_ERROR,
                std::string("GPU插值执行失败: ") + e.what()
            );
    }
    }
    
};

// CUDA核函数实现（在cpp文件中的简单实现）
void GPUInterpolationEngine::launchBilinearKernel(const float* input, float* output,
                         int srcWidth, int srcHeight,
                         int dstWidth, int dstHeight,
                         const GPUExecutionContext& context) {
#ifdef OSCEAN_CUDA_ENABLED
    // 使用默认CUDA流
    cudaStream_t stream = nullptr;  // 默认流
    
    // 调用CUDA核函数
    cudaError_t err = launchBilinearInterpolation(
        input, output,
        srcWidth, srcHeight,
        dstWidth, dstHeight,
        0.0f, (float)(srcWidth - 1),
        0.0f, (float)(srcHeight - 1),
        0.0f,  // fillValue
        stream
    );
    
    if (err != cudaSuccess) {
        OSCEAN_LOG_ERROR("GPUInterpolationEngine", "CUDA双线性插值失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 同步等待完成
    cudaDeviceSynchronize();
#else
    // 当CUDA未启用时，使用CPU版本
    OSCEAN_LOG_WARN("GPUInterpolationEngine", "CUDA未启用，使用CPU双线性插值");
    // 简单的CPU实现
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            float srcX = x * (float)(srcWidth - 1) / (dstWidth - 1);
            float srcY = y * (float)(srcHeight - 1) / (dstHeight - 1);
            
            int x0 = (int)srcX;
            int y0 = (int)srcY;
            int x1 = std::min(x0 + 1, srcWidth - 1);
            int y1 = std::min(y0 + 1, srcHeight - 1);
            
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            float v00 = input[y0 * srcWidth + x0];
            float v10 = input[y0 * srcWidth + x1];
            float v01 = input[y1 * srcWidth + x0];
            float v11 = input[y1 * srcWidth + x1];
            
            float v0 = v00 * (1 - fx) + v10 * fx;
            float v1 = v01 * (1 - fx) + v11 * fx;
            
            output[y * dstWidth + x] = v0 * (1 - fy) + v1 * fy;
        }
    }
#endif
}
    
void GPUInterpolationEngine::launchBicubicKernel(const float* input, float* output,
                        int srcWidth, int srcHeight,
                        int dstWidth, int dstHeight,
                            const GPUExecutionContext& context) {
#ifdef OSCEAN_CUDA_ENABLED
    // 使用默认CUDA流
    cudaStream_t stream = nullptr;  // 默认流
    
    // 调用CUDA核函数
    cudaError_t err = launchBicubicInterpolation(
        input, output,
        srcWidth, srcHeight,
        dstWidth, dstHeight,
        0.0f, (float)(srcWidth - 1),
        0.0f, (float)(srcHeight - 1),
        0.0f,  // fillValue
        stream
    );
    
    if (err != cudaSuccess) {
        OSCEAN_LOG_ERROR("GPUInterpolationEngine", "CUDA双三次插值失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 同步等待完成
    cudaDeviceSynchronize();
#else
    // 当CUDA未启用时，使用简化的双三次插值实现
    OSCEAN_LOG_WARN("GPUInterpolationEngine", "CUDA未启用，使用CPU双三次插值");
    
    // 立方权重函数
    auto cubic = [](float t) {
        float a = -0.5f;
        float t2 = t * t;
        float t3 = t2 * t;
        if (std::abs(t) <= 1.0f) {
            return (a + 2.0f) * t3 - (a + 3.0f) * t2 + 1.0f;
        } else if (std::abs(t) <= 2.0f) {
            return a * t3 - 5.0f * a * t2 + 8.0f * a * std::abs(t) - 4.0f * a;
        }
        return 0.0f;
    };
    
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            float srcX = x * (float)(srcWidth - 1) / (dstWidth - 1);
            float srcY = y * (float)(srcHeight - 1) / (dstHeight - 1);
            
            int x0 = (int)srcX;
            int y0 = (int)srcY;
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            // 4x4邻域
            for (int j = -1; j <= 2; ++j) {
                for (int i = -1; i <= 2; ++i) {
                    int xi = x0 + i;
                    int yi = y0 + j;
                    
                    // 边界处理
                    xi = std::max(0, std::min(xi, srcWidth - 1));
                    yi = std::max(0, std::min(yi, srcHeight - 1));
                    
                    float wx = cubic(fx - i);
                    float wy = cubic(fy - j);
                    float weight = wx * wy;
                    
                    sum += input[yi * srcWidth + xi] * weight;
                    weightSum += weight;
                }
            }
            
            output[y * dstWidth + x] = weightSum > 0 ? sum / weightSum : 0.0f;
        }
    }
#endif
}

void GPUInterpolationEngine::launchPCHIPKernel(const float* input, float* output,
                      int srcWidth, int srcHeight,
                      int dstWidth, int dstHeight,
                                    const GPUExecutionContext& context) {
#ifdef OSCEAN_CUDA_ENABLED
    cudaStream_t stream = nullptr;  // 默认流
    
    // 使用一体化的PCHIP插值（导数即时计算版本）
    // 这个版本避免了预计算导数的开销
    cudaError_t err = launchPCHIP2DInterpolationIntegrated(
        input, output,
        srcWidth, srcHeight,
        dstWidth, dstHeight,
        0.0f, (float)(srcWidth - 1),
        0.0f, (float)(srcHeight - 1),
        0.0f,  // fillValue
        stream
    );
    
    if (err != cudaSuccess) {
        OSCEAN_LOG_ERROR("GPUInterpolationEngine", "CUDA PCHIP插值失败: " + std::string(cudaGetErrorString(err)));
        
        // 如果一体化版本失败，尝试传统版本
        OSCEAN_LOG_INFO("GPUInterpolationEngine", "尝试使用传统PCHIP插值");
        
        // PCHIP需要先计算导数
        size_t derivSize = srcWidth * srcHeight * sizeof(float);
        float* d_derivX = nullptr;
        float* d_derivY = nullptr;
        float* d_derivXY = nullptr;
        
        cudaMalloc(&d_derivX, derivSize);
        cudaMalloc(&d_derivY, derivSize);
        cudaMalloc(&d_derivXY, derivSize);
        
        // 计算导数（使用优化版本）
        float dx = 1.0f;  // 假设单位网格间距
        float dy = 1.0f;
        err = computePCHIPDerivativesOptimized(
            input, d_derivX, d_derivY, d_derivXY,
            srcWidth, srcHeight, dx, dy, stream
        );
        
        if (err == cudaSuccess) {
            // 执行优化的PCHIP插值
            err = launchPCHIP2DInterpolationOptimized(
                input, d_derivX, d_derivY, d_derivXY, output,
                srcWidth, srcHeight,
                dstWidth, dstHeight,
                0.0f, (float)(srcWidth - 1),
                0.0f, (float)(srcHeight - 1),
                0.0f,  // fillValue
                stream
            );
            
            if (err != cudaSuccess) {
                OSCEAN_LOG_ERROR("GPUInterpolationEngine", "优化PCHIP插值也失败: " + std::string(cudaGetErrorString(err)));
            }
        }
        
        // 清理GPU内存
        cudaFree(d_derivX);
        cudaFree(d_derivY);
        cudaFree(d_derivXY);
    }
    
    // 同步等待完成
    cudaDeviceSynchronize();
#else
    // 当CUDA未启用时，使用CPU版本
    OSCEAN_LOG_WARN("GPUInterpolationEngine", "CUDA未启用，使用CPU双线性插值代替PCHIP");
    // 使用双线性插值作为合理的回退方案
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            float srcX = x * (float)(srcWidth - 1) / (dstWidth - 1);
            float srcY = y * (float)(srcHeight - 1) / (dstHeight - 1);
            
            int x0 = (int)srcX;
            int y0 = (int)srcY;
            int x1 = std::min(x0 + 1, srcWidth - 1);
            int y1 = std::min(y0 + 1, srcHeight - 1);
            
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            float v00 = input[y0 * srcWidth + x0];
            float v10 = input[y0 * srcWidth + x1];
            float v01 = input[y1 * srcWidth + x0];
            float v11 = input[y1 * srcWidth + x1];
            
            float v0 = v00 * (1 - fx) + v10 * fx;
            float v1 = v01 * (1 - fx) + v11 * fx;
            
            output[y * dstWidth + x] = v0 * (1 - fy) + v1 * fy;
        }
    }
#endif
}

void GPUInterpolationEngine::launchTrilinearKernel(const float* input, float* output,
                              int srcWidth, int srcHeight,
                              int dstWidth, int dstHeight,
                              const GPUExecutionContext& context) {
#ifdef OSCEAN_CUDA_ENABLED
    // 注意：这是2D到2D的简化版本，真正的3D需要更多参数
    // 暂时使用双线性插值
    OSCEAN_LOG_WARN("GPUInterpolationEngine", "三线性插值需要3D数据，暂时使用双线性插值");
    launchBilinearKernel(input, output, srcWidth, srcHeight, dstWidth, dstHeight, context);
#else
    // CPU版本
    OSCEAN_LOG_WARN("GPUInterpolationEngine", "CUDA未启用，使用CPU双线性插值代替三线性");
    // 使用双线性插值作为回退
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            float srcX = x * (float)(srcWidth - 1) / (dstWidth - 1);
            float srcY = y * (float)(srcHeight - 1) / (dstHeight - 1);
            
            int x0 = (int)srcX;
            int y0 = (int)srcY;
            int x1 = std::min(x0 + 1, srcWidth - 1);
            int y1 = std::min(y0 + 1, srcHeight - 1);
            
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            float v00 = input[y0 * srcWidth + x0];
            float v10 = input[y0 * srcWidth + x1];
            float v01 = input[y1 * srcWidth + x0];
            float v11 = input[y1 * srcWidth + x1];
            
            float v0 = v00 * (1 - fx) + v10 * fx;
            float v1 = v01 * (1 - fx) + v11 * fx;
            
            output[y * dstWidth + x] = v0 * (1 - fy) + v1 * fy;
        }
    }
#endif
}

void GPUInterpolationEngine::launchNearestNeighborKernel(const float* input, float* output,
                                    int srcWidth, int srcHeight,
                                    int dstWidth, int dstHeight,
                                    const GPUExecutionContext& context) {
#ifdef OSCEAN_CUDA_ENABLED
    // 使用默认CUDA流
    cudaStream_t stream = nullptr;  // 默认流
    
    // 执行CUDA最近邻插值
    cudaError_t err = launchNearestNeighborInterpolation(
        input, output,
        srcWidth, srcHeight,
        dstWidth, dstHeight,
        0.0f, (float)(srcWidth - 1),
        0.0f, (float)(srcHeight - 1),
        0.0f,  // fillValue
        stream
    );
    
    if (err != cudaSuccess) {
        OSCEAN_LOG_ERROR("GPUInterpolationEngine", "CUDA最近邻插值失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 同步等待完成
    cudaDeviceSynchronize();
#else
    // CPU最近邻插值
    OSCEAN_LOG_INFO("GPUInterpolationEngine", "使用CPU最近邻插值");
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            // 计算最近的源像素
            int srcX = std::round(x * (float)(srcWidth - 1) / (dstWidth - 1));
            int srcY = std::round(y * (float)(srcHeight - 1) / (dstHeight - 1));
            
            // 确保在边界内
            srcX = std::max(0, std::min(srcX, srcWidth - 1));
            srcY = std::max(0, std::min(srcY, srcHeight - 1));
            
            output[y * dstWidth + x] = input[srcY * srcWidth + srcX];
        }
    }
#endif
}
    

    
/**
 * @brief 固定内存池管理器
 */
#ifdef OSCEAN_CUDA_ENABLED
class PinnedMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool inUse;
    };
    
    std::vector<MemoryBlock> m_blocks;
    std::mutex m_mutex;
    size_t m_totalAllocated = 0;
    const size_t MAX_POOL_SIZE = 1024 * 1024 * 1024; // 1GB上限
    
public:
    ~PinnedMemoryPool() {
        // 清理所有固定内存
        for (auto& block : m_blocks) {
            if (block.ptr) {
                cudaFreeHost(block.ptr);
            }
        }
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 首先查找可重用的块
        for (auto& block : m_blocks) {
            if (!block.inUse && block.size >= size) {
                block.inUse = true;
                return block.ptr;
            }
        }
        
        // 检查是否超过内存池限制
        if (m_totalAllocated + size > MAX_POOL_SIZE) {
            return nullptr;
        }
        
        // 分配新的固定内存
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, size);
        if (err != cudaSuccess) {
            OSCEAN_LOG_ERROR("PinnedMemoryPool", "固定内存分配失败: " + std::string(cudaGetErrorString(err)));
            return nullptr;
        }
        
        m_blocks.push_back({ptr, size, true});
        m_totalAllocated += size;
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        for (auto& block : m_blocks) {
            if (block.ptr == ptr) {
                block.inUse = false;
                return;
            }
        }
    }
};

/**
 * @brief 优化的批量GPU插值引擎实现
 * 
 * 包含以下优化：
 * 1. 真正的批量处理（不是逐个处理）
 * 2. 固定内存（pinned memory）优化
 * 3. 异步流水线处理
 * 4. CUDA流并发执行
 */
class BatchGPUInterpolationEngine : public IBatchGPUInterpolationEngine {
private:
    int m_batchSize = 32;
    int m_numStreams = 4;
    InterpolationMethod m_currentMethod = InterpolationMethod::BILINEAR;
    std::string m_algorithmName = "OptimizedBatchGPUInterpolation";
    std::string m_version = "2.0.0";
    std::vector<ComputeAPI> m_supportedAPIs = {ComputeAPI::CUDA};
    
    // CUDA资源
    std::vector<cudaStream_t> m_streams;
    std::vector<cudaEvent_t> m_events;
    
    // 内存池
    std::unique_ptr<PinnedMemoryPool> m_pinnedMemoryPool;
    
    // 异步处理队列
    struct BatchJob {
        std::vector<GPUInterpolationParams> params;
        boost::promise<GPUAlgorithmResult<std::vector<GPUInterpolationResult>>> promise;
    };
    
    std::queue<BatchJob> m_jobQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    boost::thread m_workerThread;
    bool m_shouldStop = false;
    
public:
    BatchGPUInterpolationEngine() {
        // 初始化CUDA流
        m_streams.resize(m_numStreams);
        m_events.resize(m_numStreams);
        
        for (int i = 0; i < m_numStreams; ++i) {
            cudaStreamCreate(&m_streams[i]);
            cudaEventCreate(&m_events[i]);
        }
        
        // 初始化内存池
        m_pinnedMemoryPool = std::make_unique<PinnedMemoryPool>();
        
        // 启动工作线程
        m_workerThread = boost::thread(&BatchGPUInterpolationEngine::workerThreadFunc, this);
    }
    
    ~BatchGPUInterpolationEngine() {
        // 停止工作线程
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_shouldStop = true;
            m_queueCV.notify_all();
        }
        
        if (m_workerThread.joinable()) {
            m_workerThread.join();
        }
        
        // 清理CUDA资源
        for (auto& stream : m_streams) {
            cudaStreamDestroy(stream);
        }
        for (auto& event : m_events) {
            cudaEventDestroy(event);
        }
    }
    
    void setBatchSize(int size) override {
        m_batchSize = size;
    }
    
    int getOptimalBatchSize(const GPUDeviceInfo& device) const override {
        // 基于设备内存和计算能力优化批大小
        size_t availableMemory = device.memoryDetails.freeGlobalMemory;
        
        // 假设每个插值任务需要的内存
        size_t memPerTask = 1024 * 1024 * sizeof(float) * 2; // 输入+输出
        
        // 考虑多流并发，预留一些内存
        int maxBatchFromMemory = static_cast<int>((availableMemory * 0.8) / memPerTask / m_numStreams);
        
        // 基于SM数量的批大小
        int maxBatchFromSM = device.computeUnits.multiprocessorCount * 2;
        
        return std::min({maxBatchFromMemory, maxBatchFromSM, 64});
    }
    
    /**
     * @brief 获取最优批处理大小（IBatchGPUAlgorithm接口）
     */
    size_t getOptimalBatchSize(
        size_t totalItems,
        const GPUDeviceInfo& device) const override {
        return std::min(totalItems, static_cast<size_t>(getOptimalBatchSize(device)));
    }
    
    /**
     * @brief 是否支持流式处理
     */
    bool supportsStreaming() const override {
        return true; // 支持流式处理
    }
    
    /**
     * @brief 异步执行批量插值
     */
    boost::future<GPUAlgorithmResult<std::vector<GPUInterpolationResult>>> executeAsync(
        const std::vector<GPUInterpolationParams>& input,
        const GPUExecutionContext& context) override {
        
        auto promise = boost::make_shared<boost::promise<GPUAlgorithmResult<std::vector<GPUInterpolationResult>>>>();
        auto future = promise->get_future();
        
        // 将任务添加到队列
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_jobQueue.push({input, std::move(*promise)});
            m_queueCV.notify_one();
        }
        
        return future;
    }

    /**
     * @brief 同步执行批量插值
     */
    GPUAlgorithmResult<std::vector<GPUInterpolationResult>> execute(
        const std::vector<GPUInterpolationParams>& input,
        const GPUExecutionContext& context) override {
        
        auto future = executeAsync(input, context);
        return future.get();
    }
    
private:
    /**
     * @brief 工作线程函数
     */
    void workerThreadFunc() {
        while (!m_shouldStop) {
            BatchJob job;
            
            // 从队列获取任务
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_queueCV.wait(lock, [this] { return !m_jobQueue.empty() || m_shouldStop; });
                
                if (m_shouldStop) break;
                
                job = std::move(m_jobQueue.front());
                m_jobQueue.pop();
            }
            
            // 处理批任务
            auto result = processBatchOptimized(job.params);
            job.promise.set_value(result);
        }
    }
    
    /**
     * @brief 优化的批处理实现
     */
    GPUAlgorithmResult<std::vector<GPUInterpolationResult>> processBatchOptimized(
        const std::vector<GPUInterpolationParams>& input) {
        
        GPUAlgorithmResult<std::vector<GPUInterpolationResult>> result;
        result.data.resize(input.size());
        
        // 分组处理，每组使用一个CUDA流
        int groupsPerStream = (input.size() + m_numStreams - 1) / m_numStreams;
        
        std::vector<std::vector<size_t>> streamGroups(m_numStreams);
        for (size_t i = 0; i < input.size(); ++i) {
            int streamIdx = i / groupsPerStream;
            if (streamIdx >= m_numStreams) streamIdx = m_numStreams - 1;
            streamGroups[streamIdx].push_back(i);
        }
        
        // 并发处理每个流的任务
        std::vector<boost::thread> threads;
        std::mutex resultMutex;
        
        for (int streamIdx = 0; streamIdx < m_numStreams; ++streamIdx) {
            if (streamGroups[streamIdx].empty()) continue;
            
            threads.emplace_back([this, &input, &result, &resultMutex, &streamGroups, streamIdx]() {
                // 使用批量处理函数
                processStreamGroupBatch(
                    input, 
                    streamGroups[streamIdx],
                    result.data,
                    m_streams[streamIdx],
                    resultMutex
                );
            });
        }
        
        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }
        
        // 同步所有流
        for (auto& stream : m_streams) {
            cudaStreamSynchronize(stream);
        }
        
        result.success = true;
        result.error = GPUError::SUCCESS;
        
        // 计算总体统计
        GPUAlgorithmResult<std::vector<GPUInterpolationResult>>::PerformanceStats stats;
        stats.totalTime = 0;
        stats.kernelTime = 0;
        stats.transferTime = 0;
        
        for (const auto& res : result.data) {
            stats.totalTime = std::max(stats.totalTime, res.gpuTimeMs + res.memoryTransferTimeMs);
            stats.kernelTime += res.gpuTimeMs;
            stats.transferTime += res.memoryTransferTimeMs;
        }
        
        result.stats = stats;
        
        return result;
    }
    
    /**
     * @brief 使用批量核函数处理
     */
    void processStreamGroupBatch(
        const std::vector<GPUInterpolationParams>& allParams,
        const std::vector<size_t>& indices,
        std::vector<GPUInterpolationResult>& results,
        cudaStream_t stream,
        std::mutex& resultMutex) {
        
        if (indices.empty()) return;
        
        size_t batchSize = indices.size();
        
        // 检查是否所有参数都使用相同的插值方法
        InterpolationMethod method = allParams[indices[0]].method;
        for (size_t idx : indices) {
            if (allParams[idx].method != method) {
                // 回退到单个处理
                processStreamGroup(allParams, indices, results, stream, resultMutex);
                return;
            }
        }
        
        // 分配设备内存用于批处理参数
        std::vector<const float*> h_sourceArrays;
        std::vector<float*> h_outputArrays;
        std::vector<int> h_sourceWidths;
        std::vector<int> h_sourceHeights;
        std::vector<int> h_outputWidths;
        std::vector<int> h_outputHeights;
        std::vector<float> h_minX, h_maxX, h_minY, h_maxY;
        std::vector<float> h_fillValues;
        std::vector<float> h_scaleX, h_scaleY, h_offsetX, h_offsetY;
        
        // 预分配所有GPU内存
        std::vector<float*> d_sourceBuffers;
        std::vector<float*> d_outputBuffers;
        std::vector<size_t> srcSizes;
        std::vector<size_t> dstSizes;
        int maxOutputWidth = 0;
        int maxOutputHeight = 0;
        
        for (size_t idx : indices) {
            const auto& params = allParams[idx];
            if (!params.sourceData) continue;
            
            auto& def = params.sourceData->getDefinition();
            size_t srcSize = def.cols * def.rows * sizeof(float);
            size_t dstSize = params.outputWidth * params.outputHeight * sizeof(float);
            
            srcSizes.push_back(srcSize);
            dstSizes.push_back(dstSize);
            
            // 分配GPU内存
            float* d_src = nullptr;
            float* d_dst = nullptr;
            cudaMalloc(&d_src, srcSize);
            cudaMalloc(&d_dst, dstSize);
            d_sourceBuffers.push_back(d_src);
            d_outputBuffers.push_back(d_dst);
            
            h_sourceArrays.push_back(d_src);
            h_outputArrays.push_back(d_dst);
            h_sourceWidths.push_back(def.cols);
            h_sourceHeights.push_back(def.rows);
            h_outputWidths.push_back(params.outputWidth);
            h_outputHeights.push_back(params.outputHeight);
            
            // 计算缩放参数
            float minX = 0.0f;
            float maxX = (float)(def.cols - 1);
            float minY = 0.0f;
            float maxY = (float)(def.rows - 1);
            h_minX.push_back(minX);
            h_maxX.push_back(maxX);
            h_minY.push_back(minY);
            h_maxY.push_back(maxY);
            h_fillValues.push_back(0.0f);
            
            // 预计算缩放因子和偏移
            float scaleX = (maxX - minX) / (params.outputWidth - 1);
            float scaleY = (maxY - minY) / (params.outputHeight - 1);
            h_scaleX.push_back(scaleX);
            h_scaleY.push_back(scaleY);
            h_offsetX.push_back(minX);
            h_offsetY.push_back(minY);
            
            maxOutputWidth = std::max(maxOutputWidth, params.outputWidth);
            maxOutputHeight = std::max(maxOutputHeight, params.outputHeight);
        }
        
        // 分配设备内存用于参数数组
        const float** d_sourceArrays = nullptr;
        float** d_outputArrays = nullptr;
        int* d_sourceWidths = nullptr;
        int* d_sourceHeights = nullptr;
        int* d_outputWidths = nullptr;
        int* d_outputHeights = nullptr;
        float* d_minX = nullptr;
        float* d_maxX = nullptr;
        float* d_minY = nullptr;
        float* d_maxY = nullptr;
        float* d_fillValues = nullptr;
        float* d_scaleX = nullptr;
        float* d_scaleY = nullptr;
        float* d_offsetX = nullptr;
        float* d_offsetY = nullptr;
        
        cudaMalloc(&d_sourceArrays, batchSize * sizeof(const float*));
        cudaMalloc(&d_outputArrays, batchSize * sizeof(float*));
        cudaMalloc(&d_sourceWidths, batchSize * sizeof(int));
        cudaMalloc(&d_sourceHeights, batchSize * sizeof(int));
        cudaMalloc(&d_outputWidths, batchSize * sizeof(int));
        cudaMalloc(&d_outputHeights, batchSize * sizeof(int));
        cudaMalloc(&d_minX, batchSize * sizeof(float));
        cudaMalloc(&d_maxX, batchSize * sizeof(float));
        cudaMalloc(&d_minY, batchSize * sizeof(float));
        cudaMalloc(&d_maxY, batchSize * sizeof(float));
        cudaMalloc(&d_fillValues, batchSize * sizeof(float));
        cudaMalloc(&d_scaleX, batchSize * sizeof(float));
        cudaMalloc(&d_scaleY, batchSize * sizeof(float));
        cudaMalloc(&d_offsetX, batchSize * sizeof(float));
        cudaMalloc(&d_offsetY, batchSize * sizeof(float));
        
        // 复制参数到设备
        cudaMemcpyAsync(d_sourceArrays, h_sourceArrays.data(), batchSize * sizeof(const float*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_outputArrays, h_outputArrays.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sourceWidths, h_sourceWidths.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sourceHeights, h_sourceHeights.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_outputWidths, h_outputWidths.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_outputHeights, h_outputHeights.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_minX, h_minX.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_maxX, h_maxX.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_minY, h_minY.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_maxY, h_maxY.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_fillValues, h_fillValues.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_scaleX, h_scaleX.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_scaleY, h_scaleY.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_offsetX, h_offsetX.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_offsetY, h_offsetY.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        
        // 预分配固定内存
        std::vector<void*> pinnedSrcBlocks;
        std::vector<void*> pinnedDstBlocks;
        
        for (size_t i = 0; i < batchSize; ++i) {
            void* pinnedSrc = m_pinnedMemoryPool->allocate(srcSizes[i]);
            void* pinnedDst = m_pinnedMemoryPool->allocate(dstSizes[i]);
            pinnedSrcBlocks.push_back(pinnedSrc);
            pinnedDstBlocks.push_back(pinnedDst);
        }
        
        // 批量上传数据
        auto uploadStart = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < batchSize; ++i) {
            size_t idx = indices[i];
            const auto& params = allParams[idx];
            
            if (pinnedSrcBlocks[i]) {
                const unsigned char* buffer = params.sourceData->getUnifiedBufferData();
                memcpy(pinnedSrcBlocks[i], buffer, srcSizes[i]);
                cudaMemcpyAsync(d_sourceBuffers[i], pinnedSrcBlocks[i], 
                               srcSizes[i], cudaMemcpyHostToDevice, stream);
            } else {
                const unsigned char* buffer = params.sourceData->getUnifiedBufferData();
                cudaMemcpyAsync(d_sourceBuffers[i], buffer, 
                               srcSizes[i], cudaMemcpyHostToDevice, stream);
            }
        }
        
        auto uploadEnd = std::chrono::high_resolution_clock::now();
        auto uploadTime = std::chrono::duration_cast<std::chrono::microseconds>(uploadEnd - uploadStart).count() / 1000.0f;
        
        // 调用批量核函数
        auto kernelStart = std::chrono::high_resolution_clock::now();
        
        switch (method) {
            case InterpolationMethod::BILINEAR:
                launchBatchBilinearInterpolationOptimized(
                    d_sourceArrays, d_outputArrays,
                    d_sourceWidths, d_sourceHeights,
                    d_outputWidths, d_outputHeights,
                    d_scaleX, d_scaleY,
                    d_offsetX, d_offsetY,
                    d_fillValues,
                    batchSize,
                    maxOutputWidth, maxOutputHeight,
                    stream
                );
                break;
                
            default:
                // 使用通用批量双线性插值
                launchBatchBilinearInterpolation(
                    d_sourceArrays, d_outputArrays,
                    d_sourceWidths, d_sourceHeights,
                    d_outputWidths, d_outputHeights,
                    d_minX, d_maxX,
                    d_minY, d_maxY,
                    d_fillValues,
                    batchSize,
                    stream
                );
                break;
        }
        
        cudaStreamSynchronize(stream);
        auto kernelEnd = std::chrono::high_resolution_clock::now();
        auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelEnd - kernelStart).count() / 1000.0f;
        
        // 批量下载结果
        auto downloadStart = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < batchSize; ++i) {
            if (pinnedDstBlocks[i]) {
                cudaMemcpyAsync(pinnedDstBlocks[i], d_outputBuffers[i], 
                               dstSizes[i], cudaMemcpyDeviceToHost, stream);
            }
        }
        
        cudaStreamSynchronize(stream);
        auto downloadEnd = std::chrono::high_resolution_clock::now();
        auto downloadTime = std::chrono::duration_cast<std::chrono::microseconds>(downloadEnd - downloadStart).count() / 1000.0f;
        
        // 处理结果
        for (size_t i = 0; i < batchSize; ++i) {
            size_t idx = indices[i];
            const auto& params = allParams[idx];
            
            GPUInterpolationResult res;
            res.width = params.outputWidth;
            res.height = params.outputHeight;
            res.interpolatedData.resize(params.outputWidth * params.outputHeight);
            
            if (pinnedDstBlocks[i]) {
                memcpy(res.interpolatedData.data(), pinnedDstBlocks[i], dstSizes[i]);
            } else {
                cudaMemcpy(res.interpolatedData.data(), d_outputBuffers[i], 
                          dstSizes[i], cudaMemcpyDeviceToHost);
            }
            
            res.status = GPUError::SUCCESS;
            res.gpuTimeMs = kernelTime / batchSize;
            res.memoryTransferTimeMs = (uploadTime + downloadTime) / batchSize;
            
            {
                std::lock_guard<std::mutex> lock(resultMutex);
                results[idx] = std::move(res);
            }
        }
        
        // 清理资源
        for (size_t i = 0; i < pinnedSrcBlocks.size(); ++i) {
            if (pinnedSrcBlocks[i]) m_pinnedMemoryPool->deallocate(pinnedSrcBlocks[i]);
            if (pinnedDstBlocks[i]) m_pinnedMemoryPool->deallocate(pinnedDstBlocks[i]);
        }
        
        for (auto ptr : d_sourceBuffers) {
            cudaFree(ptr);
        }
        for (auto ptr : d_outputBuffers) {
            cudaFree(ptr);
        }
        
        cudaFree(d_sourceArrays);
        cudaFree(d_outputArrays);
        cudaFree(d_sourceWidths);
        cudaFree(d_sourceHeights);
        cudaFree(d_outputWidths);
        cudaFree(d_outputHeights);
        cudaFree(d_minX);
        cudaFree(d_maxX);
        cudaFree(d_minY);
        cudaFree(d_maxY);
        cudaFree(d_fillValues);
        cudaFree(d_scaleX);
        cudaFree(d_scaleY);
        cudaFree(d_offsetX);
        cudaFree(d_offsetY);
    }
    
    /**
     * @brief 处理单个流的任务组（非批量版本，作为后备）
     */
    void processStreamGroup(
        const std::vector<GPUInterpolationParams>& allParams,
        const std::vector<size_t>& indices,
        std::vector<GPUInterpolationResult>& results,
        cudaStream_t stream,
        std::mutex& resultMutex) {
        
        for (size_t idx : indices) {
            const auto& params = allParams[idx];
            if (!params.sourceData) continue;
            
            auto& def = params.sourceData->getDefinition();
            size_t srcSize = def.cols * def.rows * sizeof(float);
            size_t dstSize = params.outputWidth * params.outputHeight * sizeof(float);
            
            // 分配GPU内存
            float* d_src = nullptr;
            float* d_dst = nullptr;
            cudaMalloc(&d_src, srcSize);
            cudaMalloc(&d_dst, dstSize);
            
            // 上传数据
            const unsigned char* buffer = params.sourceData->getUnifiedBufferData();
            cudaMemcpyAsync(d_src, buffer, srcSize, cudaMemcpyHostToDevice, stream);
            
            // 调用核函数
            switch (params.method) {
                case InterpolationMethod::BILINEAR:
                    launchBilinearInterpolation(
                        d_src, d_dst,
                        def.cols, def.rows,
                        params.outputWidth, params.outputHeight,
                        0.0f, (float)(def.cols - 1),
                        0.0f, (float)(def.rows - 1),
                        0.0f, stream
                    );
                    break;
                    
                case InterpolationMethod::BICUBIC:
                    launchBicubicInterpolation(
                        d_src, d_dst,
                        def.cols, def.rows,
                        params.outputWidth, params.outputHeight,
                        0.0f, (float)(def.cols - 1),
                        0.0f, (float)(def.rows - 1),
                        0.0f, stream
                    );
                    break;
                    
                case InterpolationMethod::NEAREST_NEIGHBOR:
                    launchNearestNeighborInterpolation(
                        d_src, d_dst,
                        def.cols, def.rows,
                        params.outputWidth, params.outputHeight,
                        0.0f, (float)(def.cols - 1),
                        0.0f, (float)(def.rows - 1),
                        0.0f, stream
                    );
                    break;
                    
                default:
                    launchBilinearInterpolation(
                        d_src, d_dst,
                        def.cols, def.rows,
                        params.outputWidth, params.outputHeight,
                        0.0f, (float)(def.cols - 1),
                        0.0f, (float)(def.rows - 1),
                        0.0f, stream
                    );
                    break;
            }
            
            // 同步流
            cudaStreamSynchronize(stream);
            
            // 下载结果
            GPUInterpolationResult res;
            res.width = params.outputWidth;
            res.height = params.outputHeight;
            res.interpolatedData.resize(params.outputWidth * params.outputHeight);
            
            cudaMemcpy(res.interpolatedData.data(), d_dst, dstSize, cudaMemcpyDeviceToHost);
            
            res.status = GPUError::SUCCESS;
            res.gpuTimeMs = 1.0f; // 估算值
            res.memoryTransferTimeMs = 0.5f; // 估算值
            
            // 更新结果
            {
                std::lock_guard<std::mutex> lock(resultMutex);
                results[idx] = std::move(res);
            }
            
            // 清理GPU内存
            cudaFree(d_src);
            cudaFree(d_dst);
        }
    }
    
public:
    
    /**
     * @brief 获取支持的GPU API列表
     */
    std::vector<ComputeAPI> getSupportedAPIs() const override {
        return m_supportedAPIs;
    }
    
    /**
     * @brief 检查是否支持指定的GPU设备
     */
    bool supportsDevice(const GPUDeviceInfo& device) const override {
        for (const auto& api : m_supportedAPIs) {
            if (std::find(device.supportedAPIs.begin(), device.supportedAPIs.end(), api) != device.supportedAPIs.end()) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief 估算内存需求
     */
    size_t estimateMemoryRequirement(const std::vector<GPUInterpolationParams>& input) const override {
        size_t totalMemory = 0;
        for (const auto& params : input) {
            if (params.sourceData) {
                auto& def = params.sourceData->getDefinition();
                size_t srcSize = def.cols * def.rows * sizeof(float);
                size_t dstSize = params.outputWidth * params.outputHeight * sizeof(float);
                totalMemory += srcSize + dstSize;
        }
        }
        return totalMemory;
    }
    
    /**
     * @brief 获取算法名称
     */
    std::string getAlgorithmName() const override {
        return m_algorithmName;
    }
    
    /**
     * @brief 获取算法版本
     */
    std::string getVersion() const override {
        return m_version;
    }
};
#endif // OSCEAN_CUDA_ENABLED

// 简化的批处理引擎（不依赖CUDA）
class SimpleBatchGPUInterpolationEngine : public IBatchGPUInterpolationEngine {
private:
    int m_batchSize = 16;
    InterpolationMethod m_currentMethod = InterpolationMethod::BILINEAR;
    std::string m_algorithmName = "SimpleBatchGPUInterpolation";
    std::string m_version = "1.0.0";
    std::vector<ComputeAPI> m_supportedAPIs = {ComputeAPI::CUDA, ComputeAPI::OPENCL};
    
public:
    SimpleBatchGPUInterpolationEngine() = default;
    
    void setBatchSize(int size) override {
        m_batchSize = size;
    }
    
    int getOptimalBatchSize(const GPUDeviceInfo& device) const override {
        // 基于设备内存估算最优批处理大小
        size_t availableMemory = device.memoryDetails.freeGlobalMemory;
        size_t itemSize = 1024 * 1024 * sizeof(float); // 假设每个项目1MB
        
        int optimalBatch = static_cast<int>(availableMemory / itemSize / 4);
        return std::max(1, std::min(optimalBatch, 64));
    }
    
    size_t getOptimalBatchSize(
        size_t totalItems,
        const GPUDeviceInfo& device) const override {
        return std::min(totalItems, static_cast<size_t>(getOptimalBatchSize(device)));
    }
    
    bool supportsStreaming() const override {
        return false; // 暂不支持流式处理
    }
    
    boost::future<GPUAlgorithmResult<std::vector<GPUInterpolationResult>>> executeAsync(
        const std::vector<GPUInterpolationParams>& input,
        const GPUExecutionContext& context) override {
        
        auto promise = boost::make_shared<boost::promise<GPUAlgorithmResult<std::vector<GPUInterpolationResult>>>>();
        auto future = promise->get_future();
        
        // 在线程中执行
        boost::thread([this, input, context, promise]() {
            try {
                auto result = this->execute(input, context);
                promise->set_value(result);
            } catch (const std::exception& e) {
                GPUAlgorithmResult<std::vector<GPUInterpolationResult>> errorResult;
                errorResult.success = false;
                errorResult.error = GPUError::UNKNOWN_ERROR;
                errorResult.errorMessage = e.what();
                promise->set_value(errorResult);
    }
        }).detach();
        
        return future;
        }

    GPUAlgorithmResult<std::vector<GPUInterpolationResult>> execute(
        const std::vector<GPUInterpolationParams>& input,
        const GPUExecutionContext& context) override {
        
        GPUAlgorithmResult<std::vector<GPUInterpolationResult>> result;
        result.data.reserve(input.size());
        
        // 简单实现：逐个处理
        for (const auto& params : input) {
            GPUInterpolationEngine engine;
            engine.setInterpolationMethod(params.method);
            
            auto singleResult = engine.execute(params, context);
            if (singleResult.success) {
                result.data.push_back(std::move(singleResult.data));
            } else {
                result.success = false;
                result.error = singleResult.error;
                result.errorMessage = "批处理中的插值失败: " + singleResult.errorMessage;
                return result;
            }
        }
        
        result.success = true;
        result.error = GPUError::SUCCESS;
        return result;
    }
    
    std::vector<ComputeAPI> getSupportedAPIs() const override {
        return m_supportedAPIs;
    }
    
    bool supportsDevice(const GPUDeviceInfo& device) const override {
        for (const auto& api : m_supportedAPIs) {
            if (std::find(device.supportedAPIs.begin(), device.supportedAPIs.end(), api) != device.supportedAPIs.end()) {
                return true;
            }
        }
        return false;
    }
    
    size_t estimateMemoryRequirement(const std::vector<GPUInterpolationParams>& input) const override {
        size_t totalMemory = 0;
        for (const auto& params : input) {
            if (params.sourceData) {
                auto& def = params.sourceData->getDefinition();
                size_t srcSize = def.cols * def.rows * sizeof(float);
                size_t dstSize = params.outputWidth * params.outputHeight * sizeof(float);
                totalMemory += srcSize + dstSize;
        }
        }
        return totalMemory;
    }
    
    std::string getAlgorithmName() const override {
        return m_algorithmName;
    }
    
    std::string getVersion() const override {
        return m_version;
    }
};

// 工厂方法实现
boost::shared_ptr<IGPUInterpolationEngine> GPUInterpolationEngineFactory::create(
    ComputeAPI api) {
    return boost::make_shared<GPUInterpolationEngine>();
}

boost::shared_ptr<IGPUInterpolationEngine> GPUInterpolationEngineFactory::createOptimal(
    const GPUDeviceInfo& device) {
    // 根据设备选择最优API
    if (device.hasAPI(ComputeAPI::CUDA)) {
        return create(ComputeAPI::CUDA);
    } else if (device.hasAPI(ComputeAPI::OPENCL)) {
        return create(ComputeAPI::OPENCL);
    }
    return boost::shared_ptr<IGPUInterpolationEngine>();
}

boost::shared_ptr<IBatchGPUInterpolationEngine> GPUInterpolationEngineFactory::createBatch(
    ComputeAPI api) {
#ifdef OSCEAN_CUDA_ENABLED
    return boost::make_shared<BatchGPUInterpolationEngine>();
#else
    return boost::make_shared<SimpleBatchGPUInterpolationEngine>();
#endif
}

boost::shared_ptr<IBatchGPUInterpolationEngine> GPUInterpolationEngineFactory::createOptimizedBatch() {
#ifdef OSCEAN_CUDA_ENABLED
    // 返回优化的批处理引擎实例
    return boost::make_shared<BatchGPUInterpolationEngine>();
#else
    return boost::make_shared<SimpleBatchGPUInterpolationEngine>();
#endif
}

} // namespace gpu
} // namespace interpolation
} // namespace core_services
} // namespace oscean 