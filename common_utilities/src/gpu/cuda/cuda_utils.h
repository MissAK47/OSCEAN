/**
 * @file cuda_utils.h
 * @brief CUDA工具函数和宏定义
 */

#pragma once

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace oscean::common_utils::gpu::cuda {

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " - " + \
                                   cudaGetErrorString(error)); \
        } \
    } while(0)

// CUDA安全调用（不抛异常）
#define CUDA_SAFE_CALL(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            OSCEAN_LOG_ERROR("CUDA error: {}", cudaGetErrorString(error)); \
            return GPUError::CUDA_ERROR; \
        } \
    } while(0)

// 获取CUDA块和网格大小
inline void getCudaLaunchConfig(
    int totalElements,
    int& gridSize,
    int& blockSize,
    int maxBlockSize = 256) {
    
    blockSize = (totalElements < maxBlockSize) ? totalElements : maxBlockSize;
    gridSize = (totalElements + blockSize - 1) / blockSize;
}

// 2D块和网格配置
inline void getCuda2DLaunchConfig(
    int width,
    int height,
    dim3& gridSize,
    dim3& blockSize,
    int blockX = 16,
    int blockY = 16) {
    
    blockSize = dim3(blockX, blockY);
    gridSize = dim3(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
}

// 3D块和网格配置
inline void getCuda3DLaunchConfig(
    int width,
    int height,
    int depth,
    dim3& gridSize,
    dim3& blockSize,
    int blockX = 8,
    int blockY = 8,
    int blockZ = 8) {
    
    blockSize = dim3(blockX, blockY, blockZ);
    gridSize = dim3(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z
    );
}

// 获取设备属性
inline cudaDeviceProp getCudaDeviceProperties(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    return prop;
}

// 计算共享内存大小
inline size_t getSharedMemorySize(int deviceId) {
    cudaDeviceProp prop = getCudaDeviceProperties(deviceId);
    return prop.sharedMemPerBlock;
}

// 检查是否支持统一内存
inline bool supportsUnifiedMemory(int deviceId) {
    cudaDeviceProp prop = getCudaDeviceProperties(deviceId);
    return prop.unifiedAddressing && prop.concurrentManagedAccess;
}

// 获取最大线程数
inline int getMaxThreadsPerBlock(int deviceId) {
    cudaDeviceProp prop = getCudaDeviceProperties(deviceId);
    return prop.maxThreadsPerBlock;
}

// CUDA流创建辅助类
class CudaStream {
private:
    cudaStream_t m_stream;
    
public:
    CudaStream(unsigned int flags = cudaStreamDefault) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, flags));
    }
    
    ~CudaStream() {
        cudaStreamDestroy(m_stream);
    }
    
    // 禁止拷贝
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // 允许移动
    CudaStream(CudaStream&& other) noexcept : m_stream(other.m_stream) {
        other.m_stream = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (m_stream) {
                cudaStreamDestroy(m_stream);
            }
            m_stream = other.m_stream;
            other.m_stream = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const { return m_stream; }
    operator cudaStream_t() const { return m_stream; }
    
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
    }
    
    bool isCompleted() {
        return cudaStreamQuery(m_stream) == cudaSuccess;
    }
};

// CUDA事件辅助类
class CudaEvent {
private:
    cudaEvent_t m_event;
    
public:
    CudaEvent(unsigned int flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&m_event, flags));
    }
    
    ~CudaEvent() {
        cudaEventDestroy(m_event);
    }
    
    // 禁止拷贝
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    cudaEvent_t get() const { return m_event; }
    operator cudaEvent_t() const { return m_event; }
    
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(m_event, stream));
    }
    
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(m_event));
    }
    
    float elapsedTime(const CudaEvent& start) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.m_event, m_event));
        return ms;
    }
};

// CUDA计时器
class CudaTimer {
private:
    CudaEvent m_start;
    CudaEvent m_stop;
    cudaStream_t m_stream;
    
public:
    explicit CudaTimer(cudaStream_t stream = 0) 
        : m_start(cudaEventDefault)
        , m_stop(cudaEventDefault)
        , m_stream(stream) {
    }
    
    void start() {
        m_start.record(m_stream);
    }
    
    void stop() {
        m_stop.record(m_stream);
    }
    
    float getElapsedMilliseconds() {
        m_stop.synchronize();
        return m_stop.elapsedTime(m_start);
    }
};

// 设备内存自动管理
template<typename T>
class CudaDeviceBuffer {
private:
    T* m_data = nullptr;
    size_t m_size = 0;
    
public:
    CudaDeviceBuffer() = default;
    
    explicit CudaDeviceBuffer(size_t size) : m_size(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&m_data, size * sizeof(T)));
        }
    }
    
    ~CudaDeviceBuffer() {
        if (m_data) {
            cudaFree(m_data);
        }
    }
    
    // 禁止拷贝
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
    
    // 允许移动
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept 
        : m_data(other.m_data), m_size(other.m_size) {
        other.m_data = nullptr;
        other.m_size = 0;
    }
    
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (m_data) {
                cudaFree(m_data);
            }
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
            other.m_size = 0;
        }
        return *this;
    }
    
    T* get() { return m_data; }
    const T* get() const { return m_data; }
    size_t size() const { return m_size; }
    
    void resize(size_t newSize) {
        if (newSize == m_size) return;
        
        if (m_data) {
            cudaFree(m_data);
            m_data = nullptr;
        }
        
        m_size = newSize;
        if (newSize > 0) {
            CUDA_CHECK(cudaMalloc(&m_data, newSize * sizeof(T)));
        }
    }
    
    void copyFrom(const T* hostData, size_t count, cudaStream_t stream = 0) {
        if (count > m_size) {
            resize(count);
        }
        CUDA_CHECK(cudaMemcpyAsync(m_data, hostData, count * sizeof(T), 
                                   cudaMemcpyHostToDevice, stream));
    }
    
    void copyTo(T* hostData, size_t count, cudaStream_t stream = 0) const {
        count = std::min(count, m_size);
        CUDA_CHECK(cudaMemcpyAsync(hostData, m_data, count * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream));
    }
};

} // namespace oscean::common_utils::gpu::cuda 