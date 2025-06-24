/**
 * @file opencl_kernels.cpp
 * @brief OpenCL核函数管理器实现
 */

#include "interpolation/gpu/gpu_interpolation_engine.h"
#include <common_utils/gpu/opencl_utils.h>
#include <boost/log/trivial.hpp>
#include <fstream>
#include <sstream>

namespace oscean {
namespace interpolation {
namespace gpu {
namespace opencl {

/**
 * @brief OpenCL核函数管理器
 */
class OpenCLKernelManager {
private:
    cl_context m_context;
    cl_device_id m_device;
    cl_program m_program;
    
    // 核函数
    cl_kernel m_bilinearKernel;
    cl_kernel m_bicubicKernel;
    cl_kernel m_trilinearKernel;
    cl_kernel m_pchipKernel;
    
public:
    OpenCLKernelManager(cl_context context, cl_device_id device) 
        : m_context(context), m_device(device), m_program(nullptr) {
        
        m_bilinearKernel = nullptr;
        m_bicubicKernel = nullptr;
        m_trilinearKernel = nullptr;
        m_pchipKernel = nullptr;
    }
    
    ~OpenCLKernelManager() {
        if (m_bilinearKernel) clReleaseKernel(m_bilinearKernel);
        if (m_bicubicKernel) clReleaseKernel(m_bicubicKernel);
        if (m_trilinearKernel) clReleaseKernel(m_trilinearKernel);
        if (m_pchipKernel) clReleaseKernel(m_pchipKernel);
        if (m_program) clReleaseProgram(m_program);
    }
    
    /**
     * @brief 加载并编译核函数
     */
    bool loadKernels(const std::string& kernelPath) {
        // 读取核函数源代码
        std::ifstream file(kernelPath);
        if (!file.is_open()) {
            BOOST_LOG_TRIVIAL(error) << "Failed to open kernel file: " << kernelPath;
            return false;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source = buffer.str();
        
        // 创建程序
        const char* sourcePtr = source.c_str();
        size_t sourceSize = source.length();
        cl_int err;
        
        m_program = clCreateProgramWithSource(m_context, 1, &sourcePtr, &sourceSize, &err);
        if (err != CL_SUCCESS) {
            BOOST_LOG_TRIVIAL(error) << "Failed to create program: " << err;
            return false;
        }
        
        // 编译程序
        err = clBuildProgram(m_program, 1, &m_device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // 获取编译错误信息
            size_t logSize;
            clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            
            BOOST_LOG_TRIVIAL(error) << "Kernel compilation failed: " << log.data();
            return false;
        }
        
        // 创建核函数
        m_bilinearKernel = clCreateKernel(m_program, "bilinearInterpolation", &err);
        if (err != CL_SUCCESS) {
            BOOST_LOG_TRIVIAL(error) << "Failed to create bilinear kernel: " << err;
            return false;
        }
        
        m_bicubicKernel = clCreateKernel(m_program, "bicubicInterpolation", &err);
        if (err != CL_SUCCESS) {
            BOOST_LOG_TRIVIAL(error) << "Failed to create bicubic kernel: " << err;
            return false;
        }
        
        m_trilinearKernel = clCreateKernel(m_program, "trilinearInterpolation", &err);
        if (err != CL_SUCCESS) {
            BOOST_LOG_TRIVIAL(error) << "Failed to create trilinear kernel: " << err;
            return false;
        }
        
        m_pchipKernel = clCreateKernel(m_program, "pchip2DInterpolation", &err);
        if (err != CL_SUCCESS) {
            BOOST_LOG_TRIVIAL(error) << "Failed to create PCHIP kernel: " << err;
            return false;
        }
        
        BOOST_LOG_TRIVIAL(info) << "OpenCL kernels loaded successfully";
        return true;
    }
    
    /**
     * @brief 获取双线性插值核函数
     */
    cl_kernel getBilinearKernel() const { return m_bilinearKernel; }
    
    /**
     * @brief 获取双三次插值核函数
     */
    cl_kernel getBicubicKernel() const { return m_bicubicKernel; }
    
    /**
     * @brief 获取三线性插值核函数
     */
    cl_kernel getTrilinearKernel() const { return m_trilinearKernel; }
    
    /**
     * @brief 获取PCHIP插值核函数
     */
    cl_kernel getPCHIPKernel() const { return m_pchipKernel; }
};

// C++接口函数
extern "C" {

/**
 * @brief 执行OpenCL双线性插值
 */
cl_int launchBilinearInterpolationCL(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_sourceData,
    cl_mem d_outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue) {
    
    // 设置核函数参数
    cl_int err = CL_SUCCESS;
    int argIdx = 0;
    
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_sourceData);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_outputData);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceWidth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceHeight);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputWidth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputHeight);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &fillValue);
    
    if (err != CL_SUCCESS) {
        return err;
    }
    
    // 设置工作组大小
    size_t globalSize[2] = { (size_t)outputWidth, (size_t)outputHeight };
    size_t localSize[2] = { 16, 16 };
    
    // 执行核函数
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    
    return err;
}

/**
 * @brief 执行OpenCL双三次插值
 */
cl_int launchBicubicInterpolationCL(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_sourceData,
    cl_mem d_outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue) {
    
    // 设置核函数参数
    cl_int err = CL_SUCCESS;
    int argIdx = 0;
    
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_sourceData);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_outputData);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceWidth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceHeight);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputWidth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputHeight);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &fillValue);
    
    if (err != CL_SUCCESS) {
        return err;
    }
    
    // 设置工作组大小
    size_t globalSize[2] = { (size_t)outputWidth, (size_t)outputHeight };
    size_t localSize[2] = { 16, 16 };
    
    // 执行核函数
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    
    return err;
}

/**
 * @brief 执行OpenCL三线性插值
 */
cl_int launchTrilinearInterpolationCL(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_sourceData,
    cl_mem d_outputData,
    int sourceWidth, int sourceHeight, int sourceDepth,
    int outputWidth, int outputHeight, int outputDepth,
    float minX, float maxX,
    float minY, float maxY,
    float minZ, float maxZ,
    float fillValue) {
    
    // 设置核函数参数
    cl_int err = CL_SUCCESS;
    int argIdx = 0;
    
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_sourceData);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_outputData);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceWidth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceHeight);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceDepth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputWidth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputHeight);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputDepth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minZ);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxZ);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &fillValue);
    
    if (err != CL_SUCCESS) {
        return err;
    }
    
    // 设置工作组大小
    size_t globalSize[3] = { (size_t)outputWidth, (size_t)outputHeight, (size_t)outputDepth };
    size_t localSize[3] = { 8, 8, 8 };
    
    // 执行核函数
    err = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    
    return err;
}

/**
 * @brief 执行OpenCL PCHIP 2D插值
 */
cl_int launchPCHIP2DInterpolationCL(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_sourceData,
    cl_mem d_derivX,
    cl_mem d_derivY,
    cl_mem d_derivXY,
    cl_mem d_outputData,
    int sourceWidth, int sourceHeight,
    int outputWidth, int outputHeight,
    float minX, float maxX,
    float minY, float maxY,
    float fillValue) {
    
    // 设置核函数参数
    cl_int err = CL_SUCCESS;
    int argIdx = 0;
    
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_sourceData);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_derivX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_derivY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_derivXY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_outputData);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceWidth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &sourceHeight);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputWidth);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(int), &outputHeight);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxX);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &minY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &maxY);
    err |= clSetKernelArg(kernel, argIdx++, sizeof(float), &fillValue);
    
    if (err != CL_SUCCESS) {
        return err;
    }
    
    // 设置工作组大小
    size_t globalSize[2] = { (size_t)outputWidth, (size_t)outputHeight };
    size_t localSize[2] = { 16, 16 };
    
    // 执行核函数
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    
    return err;
}

} // extern "C"

} // namespace opencl
} // namespace gpu
} // namespace interpolation
} // namespace oscean 