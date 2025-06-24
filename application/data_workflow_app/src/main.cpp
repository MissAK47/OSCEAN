#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <csignal>
#include <thread>
#include <boost/asio/signal_set.hpp>

#include "common_utils/infrastructure/common_services_factory.h"
#include "workflow_engine/service_management/service_manager_impl.h"
#include "app/i_network_runner.h"

// 全局指针，用于信号处理器
std::unique_ptr<oscean::application::INetworkRunner> server_ptr;
boost::asio::io_context signals_ioc;

// 信号处理器，用于优雅关闭
void signal_handler(const boost::system::error_code& error, int signum) {
    if (!error) {
        std::cout << "\n收到信号 " << signum << ". 正在关闭服务器..." << std::endl;
        if (server_ptr) {
            // 通过接口停止服务
            server_ptr->stop();
            signals_ioc.stop();
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        // --- 配置 ---
        const std::string address = "0.0.0.0";
        const unsigned short port = 8080;
        const int num_threads = 4;

        std::cout << "🚀 启动 OSCEAN 后端服务..." << std::endl;

        // --- 初始化GDAL ---
        // 注意：GDAL初始化应该由数据访问服务在需要时内部处理
        // oscean::common_utils::infrastructure::GdalInitializer gdal_init;
        std::cout << "🌍 GDAL 环境将由服务按需初始化。" << std::endl;

        // --- 初始化服务管理器 ---
        std::cout << "🛠️ 初始化服务管理器..." << std::endl;
        oscean::common_utils::infrastructure::UnifiedThreadPoolManager::PoolConfiguration poolConfig;
        poolConfig.minThreads = 2;
        poolConfig.maxThreads = 8;
        auto threadPoolManager = std::make_shared<oscean::common_utils::infrastructure::UnifiedThreadPoolManager>(poolConfig);
        auto service_manager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>(threadPoolManager);
        std::cout << "✅ 服务管理器已初始化。" << std::endl;

        // --- 创建并启动网络服务 ---
        std::cout << "🌐 创建网络服务..." << std::endl;
        // 通过工厂函数创建网络服务实例，完全隐藏实现细节
        server_ptr = oscean::application::create_network_runner(address, port, num_threads, service_manager);
        std::cout << "✅ 网络服务已创建。正在监听 " << address << ":" << port << std::endl;

        // --- 设置信号处理 ---
        boost::asio::signal_set signals(signals_ioc, SIGINT, SIGTERM);
        signals.async_wait(signal_handler);

        // 在一个新线程中运行信号io_context
        std::thread signals_thread([&]() { signals_ioc.run(); });

        // --- 启动服务器（阻塞） ---
        server_ptr->run(); // 通过接口启动

        // --- 等待关闭 ---
        std::cout << "服务已停止。正在清理..." << std::endl;
        signals_thread.join(); // 等待信号处理线程结束
        std::cout << "👋 OSCEAN 后端服务已关闭。" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cerr << "!!    关键错误: " << e.what() << std::endl;
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        return 1;
    }
    return 0;
}