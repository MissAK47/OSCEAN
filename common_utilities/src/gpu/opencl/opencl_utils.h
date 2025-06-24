/**
 * @file opencl_utils.h
 * @brief OpenCL工具函数和宏定义
 */

#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

namespace oscean::common_utils::gpu::opencl {

// OpenCL错误码转字符串
inline const char* getOpenCLErrorString(cl_int error) {
    switch(error) {
        case CL_SUCCESS:                            return "Success";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown error";
    }
}

// OpenCL错误检查宏
#define CL_CHECK(call) \
    do { \
        cl_int error = call; \
        if (error != CL_SUCCESS) { \
            throw std::runtime_error(std::string("OpenCL error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " - " + \
                                   getOpenCLErrorString(error)); \
        } \
    } while(0)

// OpenCL安全调用（不抛异常）
#define CL_SAFE_CALL(call) \
    do { \
        cl_int error = call; \
        if (error != CL_SUCCESS) { \
            OSCEAN_LOG_ERROR("OpenCL error: {}", getOpenCLErrorString(error)); \
            return GPUError::OPENCL_ERROR; \
        } \
    } while(0)

// 获取平台信息
inline std::string getOpenCLPlatformInfo(cl_platform_id platform, cl_platform_info param) {
    size_t size;
    clGetPlatformInfo(platform, param, 0, nullptr, &size);
    std::vector<char> info(size);
    clGetPlatformInfo(platform, param, size, info.data(), nullptr);
    return std::string(info.data());
}

// 获取设备信息（字符串）
inline std::string getOpenCLDeviceInfo(cl_device_id device, cl_device_info param) {
    size_t size;
    clGetDeviceInfo(device, param, 0, nullptr, &size);
    std::vector<char> info(size);
    clGetDeviceInfo(device, param, size, info.data(), nullptr);
    return std::string(info.data());
}

// 获取设备信息（数值）
template<typename T>
inline T getOpenCLDeviceValue(cl_device_id device, cl_device_info param) {
    T value;
    clGetDeviceInfo(device, param, sizeof(T), &value, nullptr);
    return value;
}

// 读取OpenCL内核源文件
inline std::string readKernelFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open kernel file: " + filename);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// OpenCL上下文RAII包装
class CLContext {
private:
    cl_context m_context = nullptr;
    
public:
    CLContext(cl_platform_id platform, cl_device_id device) {
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
            0
        };
        
        cl_int err;
        m_context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
        CL_CHECK(err);
    }
    
    ~CLContext() {
        if (m_context) {
            clReleaseContext(m_context);
        }
    }
    
    // 禁止拷贝
    CLContext(const CLContext&) = delete;
    CLContext& operator=(const CLContext&) = delete;
    
    // 允许移动
    CLContext(CLContext&& other) noexcept : m_context(other.m_context) {
        other.m_context = nullptr;
    }
    
    CLContext& operator=(CLContext&& other) noexcept {
        if (this != &other) {
            if (m_context) {
                clReleaseContext(m_context);
            }
            m_context = other.m_context;
            other.m_context = nullptr;
        }
        return *this;
    }
    
    cl_context get() const { return m_context; }
    operator cl_context() const { return m_context; }
};

// OpenCL命令队列RAII包装
class CLCommandQueue {
private:
    cl_command_queue m_queue = nullptr;
    
public:
    CLCommandQueue(cl_context context, cl_device_id device, 
                   cl_command_queue_properties properties = 0) {
        cl_int err;
        m_queue = clCreateCommandQueue(context, device, properties, &err);
        CL_CHECK(err);
    }
    
    ~CLCommandQueue() {
        if (m_queue) {
            clReleaseCommandQueue(m_queue);
        }
    }
    
    // 禁止拷贝
    CLCommandQueue(const CLCommandQueue&) = delete;
    CLCommandQueue& operator=(const CLCommandQueue&) = delete;
    
    // 允许移动
    CLCommandQueue(CLCommandQueue&& other) noexcept : m_queue(other.m_queue) {
        other.m_queue = nullptr;
    }
    
    cl_command_queue get() const { return m_queue; }
    operator cl_command_queue() const { return m_queue; }
    
    void finish() {
        CL_CHECK(clFinish(m_queue));
    }
    
    void flush() {
        CL_CHECK(clFlush(m_queue));
    }
};

// OpenCL程序RAII包装
class CLProgram {
private:
    cl_program m_program = nullptr;
    
public:
    CLProgram(cl_context context, const std::string& source) {
        const char* src = source.c_str();
        size_t length = source.length();
        cl_int err;
        m_program = clCreateProgramWithSource(context, 1, &src, &length, &err);
        CL_CHECK(err);
    }
    
    ~CLProgram() {
        if (m_program) {
            clReleaseProgram(m_program);
        }
    }
    
    // 禁止拷贝
    CLProgram(const CLProgram&) = delete;
    CLProgram& operator=(const CLProgram&) = delete;
    
    cl_program get() const { return m_program; }
    operator cl_program() const { return m_program; }
    
    void build(cl_device_id device, const std::string& options = "") {
        cl_int err = clBuildProgram(m_program, 1, &device, options.c_str(), nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            // 获取构建日志
            size_t logSize;
            clGetProgramBuildInfo(m_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(m_program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            
            throw std::runtime_error("OpenCL program build failed:\n" + std::string(log.data()));
        }
    }
};

// OpenCL内核RAII包装
class CLKernel {
private:
    cl_kernel m_kernel = nullptr;
    
public:
    CLKernel(cl_program program, const std::string& name) {
        cl_int err;
        m_kernel = clCreateKernel(program, name.c_str(), &err);
        CL_CHECK(err);
    }
    
    ~CLKernel() {
        if (m_kernel) {
            clReleaseKernel(m_kernel);
        }
    }
    
    // 禁止拷贝
    CLKernel(const CLKernel&) = delete;
    CLKernel& operator=(const CLKernel&) = delete;
    
    cl_kernel get() const { return m_kernel; }
    operator cl_kernel() const { return m_kernel; }
    
    template<typename T>
    void setArg(cl_uint index, const T& value) {
        CL_CHECK(clSetKernelArg(m_kernel, index, sizeof(T), &value));
    }
    
    void setArg(cl_uint index, size_t size, const void* value) {
        CL_CHECK(clSetKernelArg(m_kernel, index, size, value));
    }
};

// OpenCL缓冲区RAII包装
template<typename T>
class CLBuffer {
private:
    cl_mem m_buffer = nullptr;
    size_t m_size = 0;
    
public:
    CLBuffer() = default;
    
    CLBuffer(cl_context context, cl_mem_flags flags, size_t count) : m_size(count) {
        cl_int err;
        m_buffer = clCreateBuffer(context, flags, count * sizeof(T), nullptr, &err);
        CL_CHECK(err);
    }
    
    ~CLBuffer() {
        if (m_buffer) {
            clReleaseMemObject(m_buffer);
        }
    }
    
    // 禁止拷贝
    CLBuffer(const CLBuffer&) = delete;
    CLBuffer& operator=(const CLBuffer&) = delete;
    
    // 允许移动
    CLBuffer(CLBuffer&& other) noexcept 
        : m_buffer(other.m_buffer), m_size(other.m_size) {
        other.m_buffer = nullptr;
        other.m_size = 0;
    }
    
    CLBuffer& operator=(CLBuffer&& other) noexcept {
        if (this != &other) {
            if (m_buffer) {
                clReleaseMemObject(m_buffer);
            }
            m_buffer = other.m_buffer;
            m_size = other.m_size;
            other.m_buffer = nullptr;
            other.m_size = 0;
        }
        return *this;
    }
    
    cl_mem get() const { return m_buffer; }
    operator cl_mem() const { return m_buffer; }
    size_t size() const { return m_size; }
    
    void write(cl_command_queue queue, const T* data, size_t count, bool blocking = true) {
        CL_CHECK(clEnqueueWriteBuffer(queue, m_buffer, blocking ? CL_TRUE : CL_FALSE,
                                     0, count * sizeof(T), data, 0, nullptr, nullptr));
    }
    
    void read(cl_command_queue queue, T* data, size_t count, bool blocking = true) {
        CL_CHECK(clEnqueueReadBuffer(queue, m_buffer, blocking ? CL_TRUE : CL_FALSE,
                                    0, count * sizeof(T), data, 0, nullptr, nullptr));
    }
};

// OpenCL事件RAII包装
class CLEvent {
private:
    cl_event m_event = nullptr;
    
public:
    CLEvent() = default;
    
    ~CLEvent() {
        if (m_event) {
            clReleaseEvent(m_event);
        }
    }
    
    // 禁止拷贝
    CLEvent(const CLEvent&) = delete;
    CLEvent& operator=(const CLEvent&) = delete;
    
    cl_event* ptr() { return &m_event; }
    cl_event get() const { return m_event; }
    operator cl_event() const { return m_event; }
    
    void wait() {
        if (m_event) {
            CL_CHECK(clWaitForEvents(1, &m_event));
        }
    }
    
    cl_ulong getProfilingInfo(cl_profiling_info param) {
        cl_ulong value;
        CL_CHECK(clGetEventProfilingInfo(m_event, param, sizeof(cl_ulong), &value, nullptr));
        return value;
    }
    
    double getElapsedTime() {
        cl_ulong start = getProfilingInfo(CL_PROFILING_COMMAND_START);
        cl_ulong end = getProfilingInfo(CL_PROFILING_COMMAND_END);
        return (end - start) / 1e6;  // 转换为毫秒
    }
};

// 计算工作组大小
inline void getOptimalWorkGroupSize(
    cl_device_id device,
    cl_kernel kernel,
    size_t globalSize,
    size_t& localSize) {
    
    size_t maxWorkGroupSize;
    CL_CHECK(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(size_t), &maxWorkGroupSize, nullptr));
    
    // 选择一个合适的本地工作组大小
    localSize = 64;  // 默认值
    while (localSize > maxWorkGroupSize) {
        localSize /= 2;
    }
    
    // 确保全局大小是本地大小的倍数
    if (globalSize % localSize != 0) {
        globalSize = ((globalSize + localSize - 1) / localSize) * localSize;
    }
}

} // namespace oscean::common_utils::gpu::opencl 