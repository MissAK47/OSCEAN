/**
 * @file test_gpu_conditional_compilation.cpp
 * @brief Test program to verify GPU conditional compilation setup
 */

#include <iostream>
#include <string>
#include <common_utils/gpu/gpu_config.h>

// Test function with GPU/CPU branches
void testProcessing() {
    std::cout << "\n=== Processing Test ===" << std::endl;
    
    OSCEAN_GPU_CODE(
        std::cout << "Running GPU implementation" << std::endl;
    ,
        std::cout << "Running CPU fallback implementation" << std::endl;
    )
}

// Test platform-specific code
void testPlatformSpecific() {
    std::cout << "\n=== Platform-Specific Test ===" << std::endl;
    
    #if OSCEAN_CUDA_ENABLED
        std::cout << "CUDA code path selected" << std::endl;
        std::cout << "CUDA Version: " << OSCEAN_CUDA_VERSION_MAJOR << "." 
                  << OSCEAN_CUDA_VERSION_MINOR << std::endl;
    #elif OSCEAN_OPENCL_ENABLED
        std::cout << "OpenCL code path selected" << std::endl;
    #elif OSCEAN_ROCM_ENABLED
        std::cout << "ROCm code path selected" << std::endl;
    #elif OSCEAN_ONEAPI_ENABLED
        std::cout << "Intel oneAPI code path selected" << std::endl;
    #else
        std::cout << "CPU-only code path selected" << std::endl;
    #endif
}

// Test class with conditional members
class TestEngine {
public:
    TestEngine() : dataSize(1024) {
        OSCEAN_GPU_CODE(
            std::cout << "TestEngine: GPU mode initialized" << std::endl;
        ,
            std::cout << "TestEngine: CPU mode initialized" << std::endl;
        )
    }
    
    void showMembers() {
        std::cout << "\n=== Class Members ===" << std::endl;
        std::cout << "dataSize: " << dataSize << std::endl;
        
        #if OSCEAN_GPU_AVAILABLE
            std::cout << "GPU-specific members are included" << std::endl;
        #else
            std::cout << "GPU-specific members are excluded" << std::endl;
        #endif
    }
    
private:
    size_t dataSize;
    OSCEAN_GPU_MEMBER(void* d_deviceMemory;)
    OSCEAN_GPU_MEMBER(int deviceId;)
};

// Main test program
int main(int argc, char* argv[]) {
    std::cout << "OSCEAN GPU Conditional Compilation Test" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Print compile-time configuration
    std::cout << "\n=== Compile-Time Configuration ===" << std::endl;
    std::cout << "OSCEAN_GPU_AVAILABLE: " << OSCEAN_GPU_AVAILABLE << std::endl;
    std::cout << "OSCEAN_CUDA_ENABLED: " << OSCEAN_CUDA_ENABLED << std::endl;
    std::cout << "OSCEAN_OPENCL_ENABLED: " << OSCEAN_OPENCL_ENABLED << std::endl;
    std::cout << "OSCEAN_ROCM_ENABLED: " << OSCEAN_ROCM_ENABLED << std::endl;
    std::cout << "OSCEAN_ONEAPI_ENABLED: " << OSCEAN_ONEAPI_ENABLED << std::endl;
    
    // Test runtime GPU detection
    std::cout << "\n=== Runtime GPU Detection ===" << std::endl;
    auto backend = oscean::gpu::getPreferredBackend();
    std::cout << "Preferred backend: " << oscean::gpu::getBackendName(backend) << std::endl;
    std::cout << "GPU available at runtime: " 
              << (oscean::gpu::isGpuRuntimeAvailable() ? "Yes" : "No") << std::endl;
    
    if (oscean::gpu::isGpuRuntimeAvailable()) {
        std::cout << "GPU device count: " << oscean::gpu::getDeviceCount() << std::endl;
    }
    
    // Run tests
    testProcessing();
    testPlatformSpecific();
    
    // Test class with conditional members
    TestEngine engine;
    engine.showMembers();
    
    // Final summary
    std::cout << "\n=== Summary ===" << std::endl;
    if (oscean::gpu::isGpuAvailable()) {
        std::cout << "This build includes GPU support" << std::endl;
        
        if (oscean::gpu::isGpuRuntimeAvailable()) {
            std::cout << "GPU devices are available and can be used" << std::endl;
        } else {
            std::cout << "GPU support compiled but no devices found at runtime" << std::endl;
            std::cout << "The application will use CPU fallback" << std::endl;
        }
    } else {
        std::cout << "This is a CPU-only build" << std::endl;
        std::cout << "All processing will use CPU implementations" << std::endl;
    }
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
} 