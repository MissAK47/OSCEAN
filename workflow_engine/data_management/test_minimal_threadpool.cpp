#include <iostream>
#include <memory>
#include "common_utils/infrastructure/unified_thread_pool_manager.h"

int main() {
    std::cout << "=== 最小线程池测试 ===" << std::endl;
    
    try {
        std::cout << "1. 开始创建线程池..." << std::endl;
        std::cout.flush();
        
        // 直接创建线程池，不使用任何其他组件
        auto threadPool = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>();
        
        std::cout << "2. 线程池创建完成" << std::endl;
        std::cout.flush();
        
        // 获取统计信息
        std::cout << "3. 获取统计信息..." << std::endl;
        auto stats = threadPool->getStatistics();
        std::cout << "   总线程数: " << stats.totalThreads << std::endl;
        std::cout.flush();
        
        std::cout << "4. 请求关闭..." << std::endl;
        threadPool->requestShutdown(std::chrono::seconds(5));
        
        std::cout << "5. 测试完成！" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "未知错误!" << std::endl;
        return 1;
    }
} 