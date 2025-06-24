/**
 * @file opencl_kernels.cpp
 * @brief OpenCL内核管理器实现（使用纯C API）
 */

#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <vector>
#include <boost/log/trivial.hpp>

namespace oscean {
namespace output_generation {
namespace gpu {
namespace opencl {

/**
 * @brief OpenCL错误检查宏
 */
#define CL_CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        BOOST_LOG_TRIVIAL(error) << "OpenCL error: " << getErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
        return false; \
    }

/**
 * @brief 获取OpenCL错误字符串
 */
const char* getErrorString(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "Success!";
        case CL_DEVICE_NOT_FOUND: return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Program build failure";
        case CL_MAP_FAILURE: return "Map failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
        case CL_INVALID_PLATFORM: return "Invalid platform";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
        case CL_INVALID_SAMPLER: return "Invalid sampler";
        case CL_INVALID_BINARY: return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid argument index";
        case CL_INVALID_ARG_VALUE: return "Invalid argument value";
        case CL_INVALID_ARG_SIZE: return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid mip-map level";
        default: return "Unknown";
    }
}

/**
 * @brief OpenCL内核管理器（使用纯C API）
 */
class OpenCLKernelManager {
private:
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    std::unordered_map<std::string, cl_program> programs_;
    std::unordered_map<std::string, cl_kernel> kernels_;
    
    // 颜色查找表缓冲区
    cl_mem colorLUTBuffer_;
    
public:
    OpenCLKernelManager() 
        : platform_(nullptr), device_(nullptr), context_(nullptr), 
          queue_(nullptr), colorLUTBuffer_(nullptr) {}
    
    ~OpenCLKernelManager() {
        cleanup();
    }
    
    /**
     * @brief 初始化OpenCL环境
     */
    bool initialize() {
        cl_int err;
        
        // 获取平台
        cl_uint numPlatforms;
        err = clGetPlatformIDs(0, nullptr, &numPlatforms);
        CL_CHECK_ERROR(err);
        
        if (numPlatforms == 0) {
            BOOST_LOG_TRIVIAL(error) << "No OpenCL platforms found";
            return false;
        }
        
        std::vector<cl_platform_id> platforms(numPlatforms);
        err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        CL_CHECK_ERROR(err);
        
        // 选择第一个平台
        platform_ = platforms[0];
        
        // 获取GPU设备
        cl_uint numDevices;
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) {
            // 尝试CPU设备
            err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
            CL_CHECK_ERROR(err);
        }
        
        if (numDevices == 0) {
            BOOST_LOG_TRIVIAL(error) << "No OpenCL devices found";
            return false;
        }
        
        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, numDevices, devices.data(), nullptr);
            CL_CHECK_ERROR(err);
        }
        
        // 选择第一个设备
        device_ = devices[0];
        
        // 获取设备信息
        char deviceName[256];
        err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        CL_CHECK_ERROR(err);
        
        BOOST_LOG_TRIVIAL(info) << "OpenCL device: " << deviceName;
        
        // 创建上下文
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        CL_CHECK_ERROR(err);
        
        // 创建命令队列
        #ifdef CL_VERSION_2_0
            cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
            queue_ = clCreateCommandQueueWithProperties(context_, device_, properties, &err);
        #else
            queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
        #endif
        CL_CHECK_ERROR(err);
        
        // 加载内核
        if (!loadKernels()) {
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief 加载OpenCL内核
     */
    bool loadKernels() {
        // 加载颜色映射内核
        if (!loadKernelFromFile("color_mapping", "kernels/color_mapping.cl")) {
            return false;
        }
        
        // 加载瓦片生成内核
        if (!loadKernelFromFile("tile_generation", "kernels/tile_generation.cl")) {
            return false;
        }
        
        // 创建内核对象
        if (!createKernel("color_mapping", "colorMappingKernel")) {
            return false;
        }
        
        if (!createKernel("color_mapping", "colorMappingWithAntialiasing")) {
            return false;
        }
        
        if (!createKernel("tile_generation", "generateTileKernel")) {
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief 从文件加载内核源码
     */
    bool loadKernelFromFile(const std::string& name, const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            BOOST_LOG_TRIVIAL(error) << "Failed to open kernel file: " << filename;
            return false;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source = buffer.str();
        
        const char* sourcePtr = source.c_str();
        size_t sourceSize = source.length();
        
        cl_int err;
        cl_program program = clCreateProgramWithSource(
            context_, 1, &sourcePtr, &sourceSize, &err);
        CL_CHECK_ERROR(err);
        
        // 编译程序
        err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // 获取编译错误信息
            size_t logSize;
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            
            BOOST_LOG_TRIVIAL(error) << "OpenCL build error:\n" << log.data();
            clReleaseProgram(program);
            return false;
        }
        
        programs_[name] = program;
        return true;
    }
    
    /**
     * @brief 创建内核对象
     */
    bool createKernel(const std::string& programName, const std::string& kernelName) {
        auto it = programs_.find(programName);
        if (it == programs_.end()) {
            BOOST_LOG_TRIVIAL(error) << "Program not found: " << programName;
            return false;
        }
        
        cl_int err;
        cl_kernel kernel = clCreateKernel(it->second, kernelName.c_str(), &err);
        CL_CHECK_ERROR(err);
        
        kernels_[kernelName] = kernel;
        return true;
    }
    
    /**
     * @brief 设置颜色查找表
     */
    bool setColorLUT(const float* lut, size_t size) {
        cl_int err;
        
        if (colorLUTBuffer_) {
            clReleaseMemObject(colorLUTBuffer_);
        }
        
        colorLUTBuffer_ = clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            size * sizeof(cl_float4), (void*)lut, &err);
        CL_CHECK_ERROR(err);
        
        // 为所有相关内核设置颜色LUT
        for (const auto& [name, kernel] : kernels_) {
            if (name.find("colorMapping") != std::string::npos ||
                name.find("generateTile") != std::string::npos) {
                err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &colorLUTBuffer_);
                if (err != CL_SUCCESS) {
                    BOOST_LOG_TRIVIAL(warning) << "Failed to set color LUT for kernel: " << name;
                }
            }
        }
        
        return true;
    }
    
    /**
     * @brief 执行颜色映射
     */
    bool executeColorMapping(
        const float* input, uint8_t* output,
        int width, int height,
        float minValue, float maxValue,
        bool antialiasing = false) {
        
        cl_int err;
        
        // 创建缓冲区
        size_t dataSize = width * height * sizeof(float);
        size_t outputSize = width * height * 4 * sizeof(uint8_t);
        
        cl_mem inputBuffer = clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            dataSize, (void*)input, &err);
        CL_CHECK_ERROR(err);
        
        cl_mem outputBuffer = clCreateBuffer(
            context_, CL_MEM_WRITE_ONLY,
            outputSize, nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(inputBuffer);
            return false;
        }
        
        // 选择内核
        std::string kernelName = antialiasing ? 
            "colorMappingWithAntialiasing" : "colorMappingKernel";
        
        auto it = kernels_.find(kernelName);
        if (it == kernels_.end()) {
            BOOST_LOG_TRIVIAL(error) << "Kernel not found: " << kernelName;
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(outputBuffer);
            return false;
        }
        
        cl_kernel kernel = it->second;
        
        // 设置内核参数
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
        err |= clSetKernelArg(kernel, 4, sizeof(float), &minValue);
        err |= clSetKernelArg(kernel, 5, sizeof(float), &maxValue);
        
        if (err != CL_SUCCESS) {
            BOOST_LOG_TRIVIAL(error) << "Failed to set kernel arguments";
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(outputBuffer);
            return false;
        }
        
        // 执行内核
        size_t globalSize[2] = {(size_t)width, (size_t)height};
        size_t localSize[2] = {16, 16};
        
        err = clEnqueueNDRangeKernel(
            queue_, kernel, 2, nullptr,
            globalSize, localSize,
            0, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            BOOST_LOG_TRIVIAL(error) << "Failed to execute kernel: " << getErrorString(err);
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(outputBuffer);
            return false;
        }
        
        // 读取结果
        err = clEnqueueReadBuffer(
            queue_, outputBuffer, CL_TRUE,
            0, outputSize, output,
            0, nullptr, nullptr);
        
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        
        return err == CL_SUCCESS;
    }
    
    /**
     * @brief 获取设备信息
     */
    bool getDeviceInfo(std::string& name, size_t& maxMemory, size_t& maxWorkGroupSize) {
        cl_int err;
        
        char deviceName[256];
        err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        CL_CHECK_ERROR(err);
        name = deviceName;
        
        cl_ulong memSize;
        err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr);
        CL_CHECK_ERROR(err);
        maxMemory = (size_t)memSize;
        
        size_t workGroupSize;
        err = clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, nullptr);
        CL_CHECK_ERROR(err);
        maxWorkGroupSize = workGroupSize;
        
        return true;
    }
    
    /**
     * @brief 清理资源
     */
    void cleanup() {
        // 释放内核
        for (auto& [name, kernel] : kernels_) {
            if (kernel) clReleaseKernel(kernel);
        }
        kernels_.clear();
        
        // 释放程序
        for (auto& [name, program] : programs_) {
            if (program) clReleaseProgram(program);
        }
        programs_.clear();
        
        // 释放其他资源
        if (colorLUTBuffer_) clReleaseMemObject(colorLUTBuffer_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
        
        colorLUTBuffer_ = nullptr;
        queue_ = nullptr;
        context_ = nullptr;
        device_ = nullptr;
        platform_ = nullptr;
    }
};

// 全局OpenCL管理器实例
static std::unique_ptr<OpenCLKernelManager> g_openclManager;

/**
 * @brief 获取OpenCL管理器实例
 */
OpenCLKernelManager& getOpenCLManager() {
    if (!g_openclManager) {
        g_openclManager = std::make_unique<OpenCLKernelManager>();
        if (!g_openclManager->initialize()) {
            BOOST_LOG_TRIVIAL(error) << "Failed to initialize OpenCL";
        }
    }
    return *g_openclManager;
}

} // namespace opencl
} // namespace gpu
} // namespace output_generation
} // namespace oscean 