/**
 * @file gpu_color_mapper_cuda.cpp
 * @brief CUDA颜色映射器实现
 */

#include "output_generation/gpu/gpu_color_mapper.h"
#include "common_utils/gpu/oscean_gpu_framework.h"
#include <cuda_runtime.h>
#include <boost/log/trivial.hpp>
#include <cstring>
#include <chrono>

// 声明CUDA核函数接口
extern "C" {
    cudaError_t uploadColorLUT(const float* colorLUT, size_t size);
    
    cudaError_t launchColorMapping(
        const float* d_input,
        uint8_t* d_output,
        int width,
        int height,
        float minValue,
        float maxValue,
        uint32_t nanColor,
        cudaStream_t stream);
    
    cudaError_t launchAdvancedColorMapping(
        const float* d_input,
        uint8_t* d_output,
        int width,
        int height,
        float minValue,
        float maxValue,
        int transformType,
        float transformParam,
        float gamma,
        uint32_t nanColor,
        cudaStream_t stream);
}

namespace oscean::output_generation::gpu {

using namespace oscean::common_utils::gpu;

/**
 * @brief CUDA颜色映射器实现 - 使用真正的GPU核函数
 */
class GPUColorMapperCUDA : public GPUColorMapperBase {
public:
    GPUColorMapperCUDA() : m_cudaInitialized(false), m_cudaStream(nullptr) {
        initializeCUDA();
    }
    
    virtual ~GPUColorMapperCUDA() {
        cleanup();
    }
    
    // IGPUAlgorithm接口实现
    boost::future<GPUAlgorithmResult<GPUVisualizationResult>> executeAsync(
        const std::shared_ptr<GridData>& input,
        const GPUExecutionContext& context) override {
        
        return boost::async(boost::launch::async, 
            [this, input, context]() -> GPUAlgorithmResult<GPUVisualizationResult> {
                
            GPUAlgorithmResult<GPUVisualizationResult> result;
            result.success = false;
            result.error = GPUError::SUCCESS;
            
            // 记录总时间开始
            auto totalStartTime = std::chrono::high_resolution_clock::now();
            
            try {
                // 参数验证
                if (!input || input->getUnifiedBufferSize() == 0) {
                    result.error = GPUError::INVALID_KERNEL;
                    result.errorMessage = "Invalid input data";
                    return result;
                }
                
                if (!m_cudaInitialized) {
                    result.error = GPUError::DEVICE_NOT_FOUND;
                    result.errorMessage = "CUDA not initialized";
                    return result;
                }
                
                // 获取网格定义
                const auto& gridDef = input->getDefinition();
                int width = gridDef.cols;
                int height = gridDef.rows;
                
                // 分配GPU内存
                size_t inputSize = width * height * sizeof(float);
                size_t outputSize = width * height * 4 * sizeof(uint8_t);
                
                float* d_input = nullptr;
                uint8_t* d_output = nullptr;
                
                // 创建CUDA事件用于计时
                cudaEvent_t startEvent, stopEvent;
                cudaEventCreate(&startEvent);
                cudaEventCreate(&stopEvent);
                
                cudaError_t err = cudaMalloc(&d_input, inputSize);
                if (err != cudaSuccess) {
                    result.error = GPUError::OUT_OF_MEMORY;
                    result.errorMessage = "Failed to allocate input memory";
                    cudaEventDestroy(startEvent);
                    cudaEventDestroy(stopEvent);
                    return result;
                }
                
                err = cudaMalloc(&d_output, outputSize);
                if (err != cudaSuccess) {
                    cudaFree(d_input);
                    result.error = GPUError::OUT_OF_MEMORY;
                    result.errorMessage = "Failed to allocate output memory";
                    cudaEventDestroy(startEvent);
                    cudaEventDestroy(stopEvent);
                    return result;
                }
                
                // 上传输入数据
                const float* inputData = reinterpret_cast<const float*>(input->getUnifiedBuffer().data());
                err = cudaMemcpy(d_input, inputData, inputSize, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    cudaFree(d_input);
                    cudaFree(d_output);
                    result.error = GPUError::TRANSFER_FAILED;
                    result.errorMessage = "Failed to upload input data";
                    cudaEventDestroy(startEvent);
                    cudaEventDestroy(stopEvent);
                    return result;
                }
                
                // 记录内核执行开始时间
                cudaEventRecord(startEvent, m_cudaStream);
                
                // 调用真正的CUDA核函数
                uint32_t nanColor = 0xFF000000; // 黑色，不透明
                
                // 使用基础颜色映射
                err = launchColorMapping(
                    d_input, d_output, width, height,
                    m_params.minValue, m_params.maxValue,
                    nanColor,
                    m_cudaStream
                );
                
                if (err != cudaSuccess) {
                    cudaFree(d_input);
                    cudaFree(d_output);
                    result.error = GPUError::KERNEL_LAUNCH_FAILED;
                    result.errorMessage = "Failed to launch color mapping kernel";
                    cudaEventDestroy(startEvent);
                    cudaEventDestroy(stopEvent);
                    return result;
                }
                
                // 记录内核执行结束时间
                cudaEventRecord(stopEvent, m_cudaStream);
                
                // 等待核函数完成
                cudaStreamSynchronize(m_cudaStream);
                
                // 计算内核执行时间
                float kernelTimeMs = 0.0f;
                cudaEventElapsedTime(&kernelTimeMs, startEvent, stopEvent);
                
                // 创建输出结果
                GPUVisualizationResult vizResult;
                vizResult.width = width;
                vizResult.height = height;
                vizResult.channels = 4;
                vizResult.imageData.resize(outputSize);
                
                // 下载结果
                err = cudaMemcpy(vizResult.imageData.data(), d_output, outputSize, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    cudaFree(d_input);
                    cudaFree(d_output);
                    result.error = GPUError::TRANSFER_FAILED;
                    result.errorMessage = "Failed to download output data";
                    cudaEventDestroy(startEvent);
                    cudaEventDestroy(stopEvent);
                    return result;
                }
                
                // 清理GPU内存和事件
                cudaFree(d_input);
                cudaFree(d_output);
                cudaEventDestroy(startEvent);
                cudaEventDestroy(stopEvent);
                
                // 计算总时间
                auto totalEndTime = std::chrono::high_resolution_clock::now();
                auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(
                    totalEndTime - totalStartTime).count();
                
                // 设置结果
                result.success = true;
                result.data = vizResult;
                result.stats.kernelTime = kernelTimeMs;
                result.stats.totalTime = totalDuration / 1000.0; // 转换为毫秒
                result.stats.transferTime = result.stats.totalTime - result.stats.kernelTime;
                result.stats.memoryUsed = inputSize + outputSize;
                result.stats.throughput = (inputSize + outputSize) / (result.stats.totalTime * 1e6); // GB/s
                
                BOOST_LOG_TRIVIAL(debug) << "CUDA color mapping completed: kernel=" << result.stats.kernelTime 
                                        << "ms, total=" << result.stats.totalTime 
                                        << "ms, throughput=" << result.stats.throughput << "GB/s";
                
            } catch (const std::exception& e) {
                result.error = GPUError::UNKNOWN_ERROR;
                result.errorMessage = e.what();
            }
            
            return result;
        });
    }
    
    std::vector<ComputeAPI> getSupportedAPIs() const override {
        return {ComputeAPI::CUDA};
    }
    
    bool supportsDevice(const GPUDeviceInfo& device) const override {
        return device.vendor == GPUVendor::NVIDIA && 
               std::find(device.supportedAPIs.begin(), device.supportedAPIs.end(), 
                        ComputeAPI::CUDA) != device.supportedAPIs.end();
    }
    
    size_t estimateMemoryRequirement(const std::shared_ptr<GridData>& input) const override {
        if (!input) return 0;
        const auto& gridDef = input->getDefinition();
        size_t inputSize = gridDef.cols * gridDef.rows * sizeof(float);
        size_t outputSize = gridDef.cols * gridDef.rows * 4 * sizeof(uint8_t);
        return inputSize + outputSize;
    }
    
    // 设置颜色查找表
    void setColorLUT(const ColorLUT& lut) {
        m_colorLUT = lut;
        
        if (m_cudaInitialized) {
            // 上传颜色查找表到GPU常量内存
            cudaError_t err = uploadColorLUT(m_colorLUT.data.data(), sizeof(m_colorLUT.data));
            if (err != cudaSuccess) {
                BOOST_LOG_TRIVIAL(error) << "Failed to upload color LUT to GPU: " << cudaGetErrorString(err);
            }
        }
    }
    
protected:
    // 实现纯虚函数
    GPUError uploadToGPU(const GridData& input, void* devicePtr) override {
        // 在executeAsync中直接处理
        return GPUError::SUCCESS;
    }
    
    GPUError executeKernel(
        void* inputDevice,
        void* outputDevice,
        int width,
        int height,
        const GPUExecutionContext& context) override {
        // 在executeAsync中直接处理
        return GPUError::SUCCESS;
    }
    
    GPUError downloadFromGPU(void* devicePtr, GPUVisualizationResult& result) override {
        // 在executeAsync中直接处理
        return GPUError::SUCCESS;
    }
    
private:
    bool m_cudaInitialized;
    cudaStream_t m_cudaStream;
    
    void initializeCUDA() {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err == cudaSuccess && deviceCount > 0) {
            // 创建CUDA流
            err = cudaStreamCreate(&m_cudaStream);
            if (err != cudaSuccess) {
                BOOST_LOG_TRIVIAL(warning) << "Failed to create CUDA stream: " << cudaGetErrorString(err);
                m_cudaStream = nullptr;
            }
            
            m_cudaInitialized = true;
            BOOST_LOG_TRIVIAL(info) << "CUDA color mapper initialized with " << deviceCount << " device(s)";
            
            // 初始化默认颜色查找表并上传到GPU
            initializeDefaultColorLUT();
            setColorLUT(m_colorLUT);
            
        } else {
            BOOST_LOG_TRIVIAL(warning) << "CUDA initialization failed or no devices found";
        }
    }
    
    void cleanup() {
        if (m_cudaStream) {
            cudaStreamDestroy(m_cudaStream);
            m_cudaStream = nullptr;
        }
        m_cudaInitialized = false;
    }
    
    void initializeDefaultColorLUT() {
        // 创建一个简单的灰度颜色表
        for (int i = 0; i < 256; ++i) {
            float value = i / 255.0f;
            m_colorLUT.data[i * 4 + 0] = value;  // R
            m_colorLUT.data[i * 4 + 1] = value;  // G
            m_colorLUT.data[i * 4 + 2] = value;  // B
            m_colorLUT.data[i * 4 + 3] = 1.0f;   // A
        }
    }
};

// 创建CUDA颜色映射器的工厂函数
std::unique_ptr<IGPUColorMapper> createCUDAColorMapper() {
    return std::make_unique<GPUColorMapperCUDA>();
}

// 创建真实CUDA颜色映射器的工厂函数（别名）
std::unique_ptr<IGPUColorMapper> createCUDAColorMapperReal() {
    return std::make_unique<GPUColorMapperCUDA>();
}

} // namespace oscean::output_generation::gpu 