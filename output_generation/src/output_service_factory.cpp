#include "output_service_factory.h"
#include "output_service_impl.h"
#include "profiles/output_profile_manager.h"
#include "engines/data_export_engine.h"
#include "engines/visualization_engine.h"
#include "writers/writer_factory.h"
#include "common_utils/infrastructure/unified_thread_pool_manager.h"
#include "common_utils/utilities/logging_utils.h"
#include "common_utils/cache/cache_strategies.h"
#include <nlohmann/json.hpp>
#include <filesystem>

namespace oscean::output {

std::unique_ptr<oscean::core_services::output::IOutputService> 
OutputServiceFactory::createOutputService(
    std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> threadPoolManager) {
    
    OSCEAN_LOG_INFO("OutputServiceFactory", "创建输出服务...");
    
    try {
        // 创建缓存管理器用于OutputProfileManager
        // 使用空指针，让OutputProfileManager使用内部缓存
        std::shared_ptr<oscean::common_utils::cache::ICacheManager<std::string, nlohmann::json>> cache = nullptr;
        
        // 🆕 使用根目录构建正确的配置文件路径
        std::filesystem::path rootPath("config");
        std::filesystem::path profilePath = rootPath / "output_profiles";
        std::string profileDirectory = profilePath.string();
        OSCEAN_LOG_DEBUG("OutputServiceFactory", "使用配置文件目录: {}", profileDirectory);
        if (!std::filesystem::exists(profileDirectory)) {
            OSCEAN_LOG_WARN("OutputServiceFactory", "输出配置文件目录不存在: {}", profileDirectory);
            // 尽管目录不存在，但构造函数可能不会立即失败，继续尝试
        }
        
        // 创建输出配置文件管理器
        auto profileManager = std::make_shared<OutputProfileManager>(
            profileDirectory,
            cache,
            threadPoolManager
        );
        OSCEAN_LOG_DEBUG("OutputServiceFactory", "输出配置文件管理器创建成功");
        
        // 创建WriterFactory
        auto writerFactory = std::make_shared<internal::WriterFactory>();
        
        // 创建数据导出引擎
        auto exportEngine = std::make_shared<DataExportEngine>(
            writerFactory,
            threadPoolManager
        );
        OSCEAN_LOG_DEBUG("OutputServiceFactory", "数据导出引擎创建成功");
        
        // 创建可视化引擎
        auto visualizationEngine = std::make_shared<VisualizationEngine>(threadPoolManager);
        OSCEAN_LOG_DEBUG("OutputServiceFactory", "可视化引擎创建成功");
        
        // 创建输出服务实现
        auto outputService = std::make_unique<OutputServiceImpl>(
            profileManager,
            exportEngine,
            visualizationEngine
        );
        
        OSCEAN_LOG_INFO("OutputServiceFactory", "输出服务创建成功");
        return outputService;
        
    } catch (const std::exception& e) {
        OSCEAN_LOG_ERROR("OutputServiceFactory", "创建输出服务失败: {}", e.what());
        throw;
    }
}

std::unique_ptr<oscean::core_services::output::IOutputService> 
OutputServiceFactory::createDefaultOutputService(
    std::shared_ptr<oscean::common_utils::infrastructure::UnifiedThreadPoolManager> threadPoolManager) {
    return createOutputService(threadPoolManager);
}

} // namespace oscean::output 