#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <csignal>
#include <thread>
#include <boost/asio/signal_set.hpp>

#include "common_utils/infrastructure/gdal_initializer.h"
#include "workflow_engine/service_management/service_manager_impl.h"
#include "../../../network_service/src/network_server.h" // Our new network server

// A global pointer to the server for the signal handler
std::unique_ptr<oscean::network_service::NetworkServer> server_ptr;
boost::asio::io_context signals_ioc;


// Signal handler for graceful shutdown
void signal_handler(const boost::system::error_code& error, int signum) {
    if (!error) {
        std::cout << "\nCaught signal " << signum << ". Shutting down server..." << std::endl;
        if (server_ptr) {
            // It's better to post the stop command to an io_context
            // to avoid issues with stopping from within a signal handler.
            // For simplicity here, we call stop directly, but a more robust
            // solution would use an io_context.
            server_ptr->stop();
            signals_ioc.stop();
        }
    }
}


int main(int argc, char* argv[]) {
    try {
        // --- Configuration ---
        const std::string address = "0.0.0.0";
        const unsigned short port = 8080;
        const int io_threads = 4;

        std::cout << "🚀 Starting OSCEAN Backend Server..." << std::endl;

        // --- Initialize Core Systems ---
        std::cout << "🌍 Initializing GDAL..." << std::endl;
        
        // 设置 PROJ 数据路径 (从项目配置中获取)
        const std::string projDataPath = "C:/Users/Administrator/vcpkg/installed/x64-windows/share/proj";
        
        // 执行 GDAL 全局初始化
        oscean::common_utils::infrastructure::GdalGlobalInitializer::getInstance()
            .initialize(projDataPath);
            
        std::cout << "✅ GDAL Initialized with PROJ data path: " << projDataPath << std::endl;

        std::cout << "🛠️ Initializing Service Manager..." << std::endl;
        auto service_manager = std::make_shared<oscean::workflow_engine::service_management::ServiceManagerImpl>();
        // Initialize the service manager, which might discover plugins or load configs
        service_manager->initialize();
        std::cout << "✅ Service Manager Initialized." << std::endl;
        
        // --- Create and Run Network Server ---
        std::cout << "🌐 Creating Network Server..." << std::endl;
        server_ptr = std::make_unique<oscean::network_service::NetworkServer>(
            address,
            port,
            io_threads,
            service_manager
        );
        std::cout << "✅ Network Server Created." << std::endl;

        // --- Register Signal Handler for Graceful Shutdown ---
        boost::asio::signal_set signals(signals_ioc, SIGINT, SIGTERM);
        signals.async_wait(signal_handler);
        
        std::thread signals_thread([&](){ signals_ioc.run(); });

        std::cout << "🏃‍ Server is running on " << address << ":" << port << std::endl;
        std::cout << "   Press Ctrl+C to shut down." << std::endl;
        
        server_ptr->run(); // This will block until the server is stopped

        if(signals_thread.joinable()){
            signals_thread.join();
        }

        std::cout << "🛑 Server has been shut down gracefully." << std::endl;

        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "💥 An unhandled exception occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}