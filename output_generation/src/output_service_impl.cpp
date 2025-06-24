#include "output_service_impl.h"
#include "profiles/output_profile_manager.h"
#include "engines/data_export_engine.h"
#include "engines/visualization_engine.h"  // 添加完整头文件
#include "engines/in_memory_data_reader.h"
#include "core_services/exceptions.h"
#include "common_utils/utilities/boost_config.h"
#include <boost/log/trivial.hpp>
#include <stdexcept>
#include <thread>

// Note: The headers for the dependent components (OutputProfileManager, etc.) are
// intentionally not included here yet. The methods are stubbed out and do not
// require the full definition of these classes to compile.

namespace oscean {
namespace output {

OutputServiceImpl::OutputServiceImpl(
    std::shared_ptr<OutputProfileManager> profileManager,
    std::shared_ptr<DataExportEngine> exportEngine,
    std::shared_ptr<VisualizationEngine> visualizationEngine)
    : m_profileManager(std::move(profileManager)),
      m_exportEngine(std::move(exportEngine)),
      m_visualizationEngine(std::move(visualizationEngine)) {
    // Per our strict architectural rules, all dependencies are mandatory.
    // This check ensures a valid service state upon construction.
    if (!m_profileManager || !m_exportEngine) {
        throw std::invalid_argument("ProfileManager and ExportEngine must be non-null for OutputServiceImpl.");
    }
    
    // VisualizationEngine 暂时可以为null，因为尚未实现
    if (!m_visualizationEngine) {
        BOOST_LOG_TRIVIAL(warning) << "VisualizationEngine is null - visualization features will be unavailable.";
    }
}

boost::future<core_services::output::OutputResult> OutputServiceImpl::processFromProfile(
    const core_services::output::ProfiledRequest& request) {
    
    BOOST_LOG_TRIVIAL(info) << "Processing profiled request with profile: " << request.profileName;
    
    try {
        // 创建一个promise来处理最终结果
        auto promise = std::make_shared<boost::promise<core_services::output::OutputResult>>();
        auto resultFuture = promise->get_future();
        
        // 1. 使用ProfileManager解析请求
        auto resolvedRequestFuture = m_profileManager->resolveRequest(request);
        
        // 2. 在后台线程中处理链式调用
        std::thread([this, promise, resolvedRequestFuture = std::move(resolvedRequestFuture)]() mutable {
            try {
                // 等待profile解析完成
                auto resolvedRequest = resolvedRequestFuture.get();
                BOOST_LOG_TRIVIAL(debug) << "Profile resolved successfully, format: " << resolvedRequest.format;
                
                // 3. 处理解析后的请求
                auto outputFuture = this->processRequest(resolvedRequest);
                auto result = outputFuture.get();
                
                // 设置最终结果
                promise->set_value(result);
                
            } catch (const std::exception& e) {
                BOOST_LOG_TRIVIAL(error) << "Failed in profile processing chain: " << e.what();
                promise->set_exception(boost::copy_exception(
                    core_services::ServiceException("Profile processing failed: " + std::string(e.what()))
                ));
            }
        }).detach();
        
        return resultFuture;
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "processFromProfile failed: " << e.what();
        boost::promise<core_services::output::OutputResult> promise;
        promise.set_exception(boost::copy_exception(
            core_services::ServiceException("processFromProfile failed: " + std::string(e.what()))
        ));
        return promise.get_future();
    }
}

boost::future<core_services::output::OutputResult> OutputServiceImpl::processRequest(
    const core_services::output::OutputRequest& request) {
    
    BOOST_LOG_TRIVIAL(info) << "Processing output request, format: " << request.format;
    
    try {
        // 检查是否为可视化请求（图像格式）
        if (isVisualizationFormat(request.format)) {
            if (!m_visualizationEngine) {
                boost::promise<core_services::output::OutputResult> promise;
                promise.set_exception(boost::copy_exception(
                    core_services::ServiceException("Visualization engine not available for format: " + request.format)
                ));
                return promise.get_future();
            }
            
            BOOST_LOG_TRIVIAL(info) << "Delegating to VisualizationEngine for format: " << request.format;
            return m_visualizationEngine->process(request);
        } else {
            // 文件导出请求
            BOOST_LOG_TRIVIAL(info) << "Delegating to DataExportEngine for format: " << request.format;
            return m_exportEngine->process(request);
        }
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "processRequest failed: " << e.what();
        boost::promise<core_services::output::OutputResult> promise;
        promise.set_exception(boost::copy_exception(
            core_services::ServiceException("processRequest failed: " + std::string(e.what()))
        ));
        return promise.get_future();
    }
}

boost::future<void> OutputServiceImpl::writeGridAsync(
    std::shared_ptr<const core_services::GridData> gridDataPtr,
    const std::string& filePath,
    const core_services::output::WriteOptions& options) {
    
    BOOST_LOG_TRIVIAL(info) << "Writing GridData to: " << filePath << ", format: " << options.format;
    
    try {
        // 创建InMemoryDataReader包装GridData
        // 注意：InMemoryDataReader 可能需要非const指针，需要确认其接口
        auto mutableGridDataPtr = std::const_pointer_cast<core_services::GridData>(gridDataPtr);
        auto memoryReader = std::make_shared<InMemoryDataReader>(mutableGridDataPtr, "output_data");
        
        // 创建OutputRequest
        core_services::output::OutputRequest request;
        request.format = options.format;
        request.streamOutput = false;
        request.dataSource = memoryReader; // 使用InMemoryDataReader作为数据源
        
        // 从文件路径中提取目录和文件名
        boost::filesystem::path path(filePath);
        request.targetDirectory = path.parent_path().string();
        request.filenameTemplate = path.filename().string();
        
        if (!options.creationOptions.empty()) {
            request.creationOptions = options.creationOptions;
        }
        
        // 创建promise来处理结果转换
        auto promise = std::make_shared<boost::promise<void>>();
        auto resultFuture = promise->get_future();
        
        // 调用processRequest并在后台线程中处理结果
        auto outputFuture = this->processRequest(request);
        std::thread([promise, outputFuture = std::move(outputFuture)]() mutable {
            try {
                auto result = outputFuture.get();
                if (result.filePaths.has_value() && !result.filePaths.value().empty()) {
                    BOOST_LOG_TRIVIAL(info) << "Successfully wrote file: " << result.filePaths.value()[0];
                }
                promise->set_value();
            } catch (const std::exception& e) {
                promise->set_exception(boost::copy_exception(e));
            }
        }).detach();
        
        return resultFuture;
        
    } catch (const std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "writeGridAsync failed: " << e.what();
        boost::promise<void> promise;
        promise.set_exception(boost::copy_exception(e));
        return promise.get_future();
    }
}

// 私有辅助方法
bool OutputServiceImpl::isVisualizationFormat(const std::string& format) const {
    // 定义可视化格式列表
    static const std::set<std::string> visualFormats = {
        "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp",
        "tile", "xyz", "wmts" // 瓦片格式
    };
    
    std::string lowerFormat = format;
    std::transform(lowerFormat.begin(), lowerFormat.end(), lowerFormat.begin(), ::tolower);
    
    return visualFormats.find(lowerFormat) != visualFormats.end();
}

} // namespace output
} // namespace oscean 