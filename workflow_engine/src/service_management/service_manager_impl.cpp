// 🚀 强制禁用boost::asio
#define OSCEAN_NO_BOOST_ASIO_MODULE 1

#include "service_manager_impl.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/gpu/oscean_gpu_framework.h"

// 核心服务接口头文件
#include "core_services/crs/i_crs_service.h"

#include <thread>
#include <chrono>
#include <filesystem>

// 引入所有核心服务的工厂头文件
#include "core_services/crs/crs_service_factory.h"  // CrsServiceConfig在这里定义
#include "core_services/data_access/i_data_access_service_factory.h"
#include "core_services/spatial_ops/spatial_ops_service_factory.h"
#include "core_services/metadata/i_metadata_service.h"
#include "impl/metadata_service_factory.h"
#include "impl/metadata_service_impl.h"
#include "impl/unified_database_manager.h"
#include "impl/intelligent_recognizer.h"
#include "core_services/output/i_output_service.h"
#include "common_utils/infrastructure/common_services_factory.h"

// 插值服务工厂需要直接包含实现文件的头文件
#include "../../../core_services_impl/interpolation_service/src/factory/interpolation_service_factory.h"

// 输出服务工厂
#include "../../../output_generation/src/output_service_factory.h"

// 注意：工作流服务通过外部注册，不在这里直接包含实现

namespace oscean::workflow_engine::service_management {

ServiceManagerImpl::ServiceManagerImpl(
    std::shared_ptr<common_utils::infrastructure::UnifiedThreadPoolManager> threadPoolManager)
    : threadPoolManager_(std::move(threadPoolManager)) {
    
    if (!threadPoolManager_) {
        throw std::invalid_argument("Thread pool manager cannot be null.");
    }
    
    // 🎯 初始化统一异步框架
    initializeAsyncFramework();
    
    registerServiceFactories();
    OSCEAN_LOG_INFO("ServiceManager", "ServiceManager initialized with AsyncFramework.");
}

std::shared_ptr<void> ServiceManagerImpl::getServiceInternal(std::type_index serviceType) {
    std::lock_guard<std::mutex> lock(mutex_);
    return getServiceInternalNoLock(serviceType);
}

std::shared_ptr<void> ServiceManagerImpl::getServiceInternalNoLock(std::type_index serviceType) {
    // 1. 检查服务是否已在缓存中
    auto it = services_.find(serviceType);
    if (it != services_.end()) {
        OSCEAN_LOG_DEBUG("ServiceManager", "Returning cached service for '{}'", serviceType.name());
        return it->second;
    }

    // 2. 服务未创建，使用预注册的工厂进行懒加载
    OSCEAN_LOG_INFO("ServiceManager", "Lazily loading new service for '{}'", serviceType.name());
    auto factory_it = serviceFactories_.find(serviceType);
    if (factory_it == serviceFactories_.end()) {
        auto errorMsg = "Service not registered: " + std::string(serviceType.name());
        OSCEAN_LOG_ERROR("ServiceManager", errorMsg);
        // 对于未注册的服务，返回nullptr而不是抛异常
        return nullptr;
    }

    // 3. 调用工厂函数创建服务实例
    try {
        auto instance = factory_it->second();
        services_[serviceType] = instance; // 缓存实例（即使是nullptr）
        if (instance) {
            OSCEAN_LOG_INFO("ServiceManager", "Service '{}' created and cached successfully.", serviceType.name());
        } else {
            OSCEAN_LOG_WARN("ServiceManager", "Service '{}' factory returned nullptr (service unavailable).", serviceType.name());
        }
        return instance;
    } catch (const std::exception& e) {
        auto errorMsg = "Failed to create service '" + std::string(serviceType.name()) + "': " + e.what();
        OSCEAN_LOG_ERROR("ServiceManager", errorMsg);
        // 缓存失败结果为nullptr，避免重复尝试
        services_[serviceType] = nullptr;
        return nullptr;
    }
}

template<typename ServiceInterface>
std::shared_ptr<ServiceInterface> ServiceManagerImpl::getServiceNoLock() {
    return std::static_pointer_cast<ServiceInterface>(getServiceInternalNoLock(typeid(ServiceInterface)));
}

void ServiceManagerImpl::registerServiceFactories() {
    // 注册所有核心服务的创建逻辑。
    // 服务间的依赖关系在此处通过调用 getServiceNoLock<T>() 来解决。
    
    OSCEAN_LOG_INFO("ServiceManager", "Registering service factories...");
    
    // 首先注册CommonServicesFactory（作为基础依赖）- 使用共享的线程池管理器
    serviceFactories_[typeid(common_utils::infrastructure::CommonServicesFactory)] = [this]() {
        // 🔧 关键修复：使用配置根目录来定位配置文件
        std::string configPath = "config/database_config.yaml";
        if (!std::filesystem::exists(configPath)) {
            // 恢复原来的逻辑，尽管它可能有问题，但遵循不自己造路径的原则
            configPath = "./config/database_config.yaml";
        }
        
        if (std::filesystem::exists(configPath)) {
            // 🎯 使用配置文件创建
            std::cout << "[ServiceManager] 🔧 检测到配置文件，使用配置文件创建CommonServicesFactory: " << configPath << std::endl;
            auto factory = std::make_shared<common_utils::infrastructure::CommonServicesFactory>(configPath);
            OSCEAN_LOG_INFO("ServiceManager", "CommonServicesFactory created with config file: {}", configPath);
            return factory;
        } else {
            // 🔧 使用ServiceConfiguration传入已有的线程池管理器，避免创建新的UnifiedThreadPoolManager
            std::cout << "[ServiceManager] ⚠️ 未找到配置文件，使用默认配置创建CommonServicesFactory" << std::endl;
            common_utils::infrastructure::ServiceConfiguration config;
            config.sharedThreadPoolManager = threadPoolManager_;  // 复用现有的线程池管理器
            config.environment = common_utils::infrastructure::Environment::PRODUCTION;  // 🎯 使用生产环境配置
            
            auto factory = std::make_shared<common_utils::infrastructure::CommonServicesFactory>(config);
            OSCEAN_LOG_INFO("ServiceManager", "CommonServicesFactory created with shared thread pool manager.");
            return factory;
        }
    };
    
    // 注册UnifiedThreadPoolManager（从CommonServicesFactory获取）
    serviceFactories_[typeid(common_utils::infrastructure::UnifiedThreadPoolManager)] = [this]() {
        return threadPoolManager_;
    };
    
    // 注册OSCEANGPUFramework（全局单例）
    serviceFactories_[typeid(common_utils::gpu::OSCEANGPUFramework)] = []() {
        // GPU框架是全局单例，使用自定义删除器创建shared_ptr
        // 删除器什么都不做，因为单例的生命周期由框架自己管理
        return std::shared_ptr<void>(
            &common_utils::gpu::OSCEANGPUFramework::getInstance(),
            [](void*) {} // 空删除器，不删除单例
        );
    };
    
    // 1. CRS 服务 - 🔍 重新启用并深度调试以彻底解决问题
    serviceFactories_[typeid(core_services::ICrsService)] = [this]() {
        std::cout << "[DEBUG ServiceManager] 🔍 开始CRS服务深度诊断与创建..." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            // 步骤1: 验证CommonServicesFactory
            std::cout << "[DEBUG ServiceManager] 步骤1: 获取CommonServicesFactory..." << std::endl;
            auto commonFactory = getServiceNoLock<common_utils::infrastructure::CommonServicesFactory>();
            if (!commonFactory) {
                std::cout << "[DEBUG ServiceManager] ❌ CommonServicesFactory为空" << std::endl;
                OSCEAN_LOG_ERROR("ServiceManager", "[CRS] 获取CommonServicesFactory失败");
                return std::shared_ptr<core_services::ICrsService>();
            }
            std::cout << "[DEBUG ServiceManager] ✅ CommonServicesFactory获取成功" << std::endl;
            
            // 步骤2: 检查依赖服务的可用性
            std::cout << "[DEBUG ServiceManager] 步骤2: 检查依赖服务..." << std::endl;
            
            auto memoryManager = commonFactory->getMemoryManager();
            std::cout << "[DEBUG ServiceManager] 内存管理器: " << (memoryManager ? "✅ 可用" : "❌ 不可用") << std::endl;
            
            auto threadPoolManager = commonFactory->getThreadPoolManager();
            std::cout << "[DEBUG ServiceManager] 线程池管理器: " << (threadPoolManager ? "✅ 可用" : "❌ 不可用") << std::endl;
            
            auto simdManager = commonFactory->getSIMDManager();
            std::cout << "[DEBUG ServiceManager] SIMD管理器: " << (simdManager ? "✅ 可用" : "❌ 不可用") << std::endl;
            
            auto logger = commonFactory->getLogger();
            std::cout << "[DEBUG ServiceManager] 日志管理器: " << (logger ? "✅ 可用" : "❌ 不可用") << std::endl;
            
            if (!memoryManager || !threadPoolManager || !logger) {
                std::cout << "[DEBUG ServiceManager] ❌ 必需的依赖服务不可用" << std::endl;
                return std::shared_ptr<core_services::ICrsService>();
            }
            
            // 步骤3: 创建CRS服务配置
            std::cout << "[DEBUG ServiceManager] 步骤3: 创建CRS服务配置..." << std::endl;
            auto crsConfig = oscean::core_services::crs::CrsServiceConfig::createHighPerformance();
            std::cout << "[DEBUG ServiceManager] CRS配置创建成功" << std::endl;
            
            // 步骤4: 创建CRS服务工厂
            std::cout << "[DEBUG ServiceManager] 步骤4: 创建CRS服务工厂..." << std::endl;
            auto crsFactory = std::make_unique<oscean::core_services::crs::CrsServiceFactory>(commonFactory, crsConfig);
            std::cout << "[DEBUG ServiceManager] CRS工厂创建成功" << std::endl;
            
            // 步骤5: 使用超时保护创建CRS服务
            std::cout << "[DEBUG ServiceManager] 步骤5: 使用超时保护创建CRS服务..." << std::endl;
            
            std::unique_ptr<oscean::core_services::ICrsService> crsServiceUnique;
            std::exception_ptr creation_exception = nullptr;
            std::atomic<bool> creation_complete{false};
            
            // 在独立线程中创建服务
            std::thread creation_thread([&]() {
                try {
                    std::cout << "[DEBUG ServiceManager] [创建线程] 开始调用createCrsService()..." << std::endl;
                    crsServiceUnique = crsFactory->createCrsService();
                    std::cout << "[DEBUG ServiceManager] [创建线程] createCrsService()调用完成" << std::endl;
                    creation_complete.store(true);
                } catch (const std::exception& e) {
                    std::cout << "[DEBUG ServiceManager] [创建线程] 创建异常: " << e.what() << std::endl;
                    creation_exception = std::current_exception();
                    creation_complete.store(true);
                } catch (...) {
                    std::cout << "[DEBUG ServiceManager] [创建线程] 未知异常" << std::endl;
                    creation_exception = std::current_exception();
                    creation_complete.store(true);
                }
            });
            
            // 等待创建完成，超时时间20秒
            const int timeout_seconds = 20;
            auto creation_start = std::chrono::steady_clock::now();
            
            while (!creation_complete.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                auto elapsed = std::chrono::steady_clock::now() - creation_start;
                if (elapsed > std::chrono::seconds(timeout_seconds)) {
                    std::cout << "[DEBUG ServiceManager] ⏰ CRS服务创建超时(" << timeout_seconds << "秒)" << std::endl;
                    OSCEAN_LOG_ERROR("ServiceManager", "[CRS] 服务创建超时");
                    
                    // 尝试优雅终止线程
                    creation_thread.detach();
                    return std::shared_ptr<core_services::ICrsService>();
                }
            }
            
            // 等待线程完成
            if (creation_thread.joinable()) {
                creation_thread.join();
            }
            
            // 检查创建结果
            if (creation_exception) {
                std::cout << "[DEBUG ServiceManager] 重新抛出创建异常..." << std::endl;
                std::rethrow_exception(creation_exception);
            }
            
            if (!crsServiceUnique) {
                std::cout << "[DEBUG ServiceManager] ❌ CRS服务创建返回空指针" << std::endl;
                OSCEAN_LOG_ERROR("ServiceManager", "[CRS] 服务创建返回空指针");
                return std::shared_ptr<core_services::ICrsService>();
            }
            
            // 步骤6: 转换为shared_ptr并验证
            std::cout << "[DEBUG ServiceManager] 步骤6: 转换为shared_ptr..." << std::endl;
            auto crsServiceShared = std::shared_ptr<core_services::ICrsService>(crsServiceUnique.release());
            
            if (!crsServiceShared) {
                std::cout << "[DEBUG ServiceManager] ❌ shared_ptr转换失败" << std::endl;
                return std::shared_ptr<core_services::ICrsService>();
            }
            
            auto end_time = std::chrono::steady_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "[DEBUG ServiceManager] ✅ CRS服务创建成功！总耗时: " << total_duration.count() << "ms" << std::endl;
            OSCEAN_LOG_INFO("ServiceManager", "[CRS] 服务创建成功，耗时: {}ms", total_duration.count());
            
            return crsServiceShared;
            
        } catch (const std::exception& e) {
            auto end_time = std::chrono::steady_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "[DEBUG ServiceManager] ❌ CRS服务创建失败: " << e.what() << " (耗时: " << total_duration.count() << "ms)" << std::endl;
            OSCEAN_LOG_ERROR("ServiceManager", "[CRS] 服务创建失败: {} (耗时: {}ms)", e.what(), total_duration.count());
            return std::shared_ptr<core_services::ICrsService>();
        }
    };

    // 2. 数据访问服务 (通过工厂创建) - 🔧 恢复简单工厂模式 
    serviceFactories_[typeid(core_services::data_access::IUnifiedDataAccessService)] = [this]() {
        OSCEAN_LOG_INFO("ServiceManager", "[DataAccess] 创建数据访问服务 - 懒加载架构");
        auto commonFactory = getServiceNoLock<common_utils::infrastructure::CommonServicesFactory>();
        auto dataAccessFactory = core_services::data_access::createDataAccessServiceFactoryWithDependencies(commonFactory);
        
        // 🔧 **架构修复**: 恢复纯粹的懒加载，GDAL将在需要时由各服务自己初始化
        auto dataAccessService = dataAccessFactory->createForProduction();
        OSCEAN_LOG_INFO("ServiceManager", "[DataAccess] 数据访问服务创建完成");
        return dataAccessService;
    };
    
    // 3. 元数据服务 (通过工厂创建) - 🛡️ 支持CRS服务可选
    serviceFactories_[typeid(core_services::metadata::IMetadataService)] = [this]() {
        std::cout << "[DEBUG] 开始创建元数据服务..." << std::endl;
        try {
            std::cout << "[DEBUG] 步骤1: 获取CommonServicesFactory..." << std::endl;
            auto commonFactory = getServiceNoLock<common_utils::infrastructure::CommonServicesFactory>();
            
            if (!commonFactory) {
                std::cout << "[DEBUG] CommonServicesFactory为null，创建失败" << std::endl;
                OSCEAN_LOG_ERROR("ServiceManager", "Cannot create MetadataService due to missing CommonServicesFactory.");
                return std::shared_ptr<core_services::metadata::IMetadataService>();
            }
            std::cout << "[DEBUG] CommonServicesFactory获取成功" << std::endl;

            // 🎯 使用数据管理工作流优化配置：禁用分类规则加载以提高启动速度和稳定性
            std::cout << "[DEBUG] 步骤2: 获取默认配置..." << std::endl;
            auto config = core_services::metadata::impl::MetadataServiceFactory::getDefaultConfiguration();
            config.classificationConfig.loadClassificationRules = false;  // 🔧 关键修复：避免YAML文件加载阻塞
            std::cout << "[DEBUG] 配置获取完成，分类规则加载已禁用" << std::endl;
            OSCEAN_LOG_INFO("ServiceManager", "MetadataService config: 禁用分类规则加载以提高数据管理工作流启动速度");
            
            // 🔧 核心修复：注入服务管理器自身，而不是具体的服务实例
            std::cout << "[DEBUG] 步骤3: 创建MetadataServiceFactory..." << std::endl;
            auto metadataFactory = std::make_unique<core_services::metadata::impl::MetadataServiceFactory>(
                commonFactory, 
                this->shared_from_this(), // 注入IServiceManager
                config
            );
            std::cout << "[DEBUG] MetadataServiceFactory创建成功" << std::endl;
            
            std::cout << "[DEBUG] 步骤4: 调用createMetadataService()..." << std::endl;
            auto serviceUniquePtr = metadataFactory->createMetadataService();
            
            if (serviceUniquePtr) {
                std::cout << "[DEBUG] MetadataService创建成功，转换为shared_ptr..." << std::endl;
                OSCEAN_LOG_INFO("ServiceManager", "MetadataService created successfully via its factory.");
                // 转换unique_ptr为shared_ptr
                return std::shared_ptr<core_services::metadata::IMetadataService>(serviceUniquePtr.release());
            } else {
                std::cout << "[DEBUG] MetadataServiceFactory返回了nullptr" << std::endl;
                OSCEAN_LOG_ERROR("ServiceManager", "MetadataServiceFactory failed to create a service instance.");
                return std::shared_ptr<core_services::metadata::IMetadataService>();
            }

        } catch (const std::exception& e) {
            std::cout << "[DEBUG] 元数据服务创建异常: " << e.what() << std::endl;
            OSCEAN_LOG_ERROR("ServiceManager", "Failed to create MetadataService: {}", e.what());
            return std::shared_ptr<core_services::metadata::IMetadataService>();
        }
    };

    // 4. 空间操作服务 (通过工厂创建)
    serviceFactories_[typeid(core_services::spatial_ops::ISpatialOpsService)] = [this]() {
        OSCEAN_LOG_INFO("ServiceManager", "[SpatialOps] 创建空间操作服务...");
        try {
            // 创建独立的空间操作服务，不依赖其他服务
            auto spatialOpsService = core_services::spatial_ops::SpatialOpsServiceFactory::createService();
            if (spatialOpsService) {
                OSCEAN_LOG_INFO("ServiceManager", "[SpatialOps] 空间操作服务创建成功");
                return std::shared_ptr<core_services::spatial_ops::ISpatialOpsService>(spatialOpsService.release());
            } else {
                OSCEAN_LOG_ERROR("ServiceManager", "[SpatialOps] 空间操作服务创建返回nullptr");
                return std::shared_ptr<core_services::spatial_ops::ISpatialOpsService>(nullptr);
            }
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("ServiceManager", "[SpatialOps] 空间操作服务创建失败: {}", e.what());
            return std::shared_ptr<core_services::spatial_ops::ISpatialOpsService>(nullptr);
        }
    };

    // 5. 插值服务 (通过工厂创建)
    serviceFactories_[typeid(core_services::interpolation::IInterpolationService)] = [this]() {
        auto commonFactory = getServiceNoLock<common_utils::infrastructure::CommonServicesFactory>();
        auto stdSimdManager = commonFactory->getSIMDManager();
        
        // 转换 std::shared_ptr 到 boost::shared_ptr
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> boostSimdManager;
        if (stdSimdManager) {
            // 使用原始指针构造 boost::shared_ptr，并使用自定义删除器来确保正确的内存管理
            boostSimdManager = boost::shared_ptr<oscean::common_utils::simd::ISIMDManager>(
                stdSimdManager.get(),
                [stdSimdManager](oscean::common_utils::simd::ISIMDManager*) mutable { 
                    // 保持std::shared_ptr活着，让它管理生命周期
                    stdSimdManager.reset();
                }
            );
        }
        
        auto interpolationServicePtr = core_services::interpolation::InterpolationServiceFactory::createHighPerformanceService(boostSimdManager);
        return std::shared_ptr<core_services::interpolation::IInterpolationService>(interpolationServicePtr.release());
    };
    
    // 6. 输出服务 (通过工厂创建) - 🆕 实现完整的输出服务
    serviceFactories_[typeid(core_services::output::IOutputService)] = [this]() {
        OSCEAN_LOG_INFO("ServiceManager", "[Output] 创建输出服务...");
        try {
            // 使用输出服务工厂创建服务，并传入配置根目录
            auto outputService = oscean::output::OutputServiceFactory::createDefaultOutputService(threadPoolManager_);
            
            if (outputService) {
                OSCEAN_LOG_INFO("ServiceManager", "[Output] 输出服务创建成功");
                return std::shared_ptr<core_services::output::IOutputService>(outputService.release());
            } else {
                OSCEAN_LOG_ERROR("ServiceManager", "[Output] 输出服务工厂返回nullptr");
                return std::shared_ptr<core_services::output::IOutputService>(nullptr);
            }
        } catch (const std::exception& e) {
            OSCEAN_LOG_ERROR("ServiceManager", "[Output] 输出服务创建失败: {}", e.what());
            return std::shared_ptr<core_services::output::IOutputService>(nullptr);
        }
    };
    
    // 注意：工作流服务（数据工作流、数据管理等）通过外部注册机制添加
    // 这样可以避免核心服务管理器对具体工作流实现的直接依赖
    
    OSCEAN_LOG_INFO("ServiceManager", "Core service factories registration completed. Registered {} factories.", serviceFactories_.size());
}

// =============================================================================
// 🎯 统一异步框架实现 (核心解决方案)
// =============================================================================

void ServiceManagerImpl::initializeAsyncFramework() {
    try {
        // 🎯 使用AsyncFramework的工厂方法创建实例
        // 这样避免了直接访问UnifiedThreadPoolManager内部的线程池
        asyncFramework_ = oscean::common_utils::async::AsyncFramework::createDefault();
        
        OSCEAN_LOG_INFO("ServiceManager", "AsyncFramework initialized successfully using factory method");
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("ServiceManager", "Failed to initialize AsyncFramework: {}", e.what());
        throw;
    }
}

oscean::common_utils::async::AsyncFramework& ServiceManagerImpl::getAsyncFramework() {
    if (!asyncFramework_) {
        throw std::runtime_error("AsyncFramework not initialized");
    }
    return *asyncFramework_;
}

bool ServiceManagerImpl::waitForAllAsyncTasks(size_t timeoutSeconds) {
    if (!asyncFramework_) {
        OSCEAN_LOG_WARN("ServiceManager", "AsyncFramework not initialized, no tasks to wait for");
        return true;
    }
    
    try {
        OSCEAN_LOG_INFO("ServiceManager", "等待所有异步任务完成，超时时间: {} 秒", timeoutSeconds);
        
        // AsyncFramework没有直接的waitForAll方法，我们需要实现一个简单的等待逻辑
        // 这里使用统计信息来检查活跃任务数
        auto startTime = std::chrono::steady_clock::now();
        auto timeout = std::chrono::seconds(timeoutSeconds);
        
        while (true) {
            auto stats = asyncFramework_->getStatistics();
            if (stats.currentActiveTasks == 0) {
                OSCEAN_LOG_INFO("ServiceManager", "所有异步任务已完成");
                return true;
            }
            
            // 检查超时
            if (timeoutSeconds > 0) {
                auto elapsed = std::chrono::steady_clock::now() - startTime;
                if (elapsed >= timeout) {
                    OSCEAN_LOG_WARN("ServiceManager", "等待异步任务超时，仍有 {} 个活跃任务", stats.currentActiveTasks);
                    return false;
                }
            }
            
            // 短暂等待后重新检查
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("ServiceManager", "等待异步任务时发生错误: {}", e.what());
        return false;
    }
}

// 🎯 模板方法实现需要在头文件中
// submitAsyncTask的实现已在头文件中定义

} // namespace oscean::workflow_engine::service_management 