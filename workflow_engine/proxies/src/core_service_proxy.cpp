#include "workflow_engine/proxies/core_service_proxy.h"
#include "common_utils/utilities/logging_utils.h"
#include "core_services/common_data_types.h"

// 只包含实际存在的核心服务头文件
#include "core_services/metadata/unified_metadata_service.h"

#include <boost/thread/future.hpp>
#include <future>
#include <thread>

namespace oscean::workflow_engine::proxies {

/**
 * @brief CoreServiceProxy的具体实现
 */
class CoreServiceProxyImpl : public CoreServiceProxy {
public:
    CoreServiceProxyImpl() = default;
    
    ~CoreServiceProxyImpl() override {
        shutdown();
    }

    bool initialize(const std::map<std::string, std::any>& config) override {
        LOG_INFO("初始化核心服务代理");
        
        try {
            // 检查是否已经通过其他方式设置了服务
            if (metadataService_) {
                LOG_INFO("服务已经通过外部设置，跳过内部初始化");
                initialized_ = true;
                return true;
            }

            // 从配置中获取服务实例或初始化新的服务
            // 这里假设服务实例通过configure方法外部传入
            
            LOG_WARN("服务未通过外部设置，需要调用 setServices() 方法");
            return false;
            
        } catch (const std::exception& e) {
            LOG_ERROR("初始化核心服务代理失败: {}", e.what());
            return false;
        }
    }

    /**
     * @brief 设置服务实例（外部注入）
     */
    void setMetadataService(
        std::shared_ptr<oscean::core_services::metadata::IUnifiedMetadataService> mds) {
        
        metadataService_ = mds;
        initialized_ = (mds != nullptr);
        
        LOG_INFO("元数据服务实例设置完成，初始化状态: {}", initialized_);
    }

    boost::future<std::optional<oscean::core_services::FileMetadata>>
    getFileMetadataFromDAS(const std::string& filePath) override {
        boost::promise<std::optional<oscean::core_services::FileMetadata>> promise;
        auto future = promise.get_future();
        
        // 暂时返回空值，待DAS服务集成
        LOG_WARN("DAS服务暂未集成，返回空元数据: {}", filePath);
        promise.set_value(std::nullopt);
        
        return future;
    }

    boost::future<std::string>
    recognizeFileWithMDS(const std::string& filePath) override {
        boost::promise<std::string> promise;
        auto future = promise.get_future();
        
        if (!initialized_ || !metadataService_) {
            LOG_ERROR("元数据服务未初始化");
            promise.set_value("Unknown");
            return future;
        }
        
        try {
            LOG_DEBUG("通过MDS识别文件类型: {}", filePath);
            
            // 暂时使用简单的文件扩展名识别
            // TODO: 等待MDS服务提供识别接口后，再调用真实的方法
            std::string extension = filePath.substr(filePath.find_last_of('.') + 1);
            
            std::string dataType = "Unknown";
            if (extension == "nc" || extension == "netcdf") {
                dataType = "OceanEnvironment";  // NetCDF通常是海洋环境数据
            } else if (extension == "tif" || extension == "tiff") {
                dataType = "TopographyBathymetry";  // GeoTIFF通常是地形数据
            } else if (extension == "shp") {
                dataType = "BoundaryLines";  // Shapefile通常是边界数据
            }
            
            LOG_DEBUG("简单文件类型识别: {} -> {}", filePath, dataType);
            promise.set_value(dataType);
            
        } catch (const std::exception& e) {
            LOG_ERROR("文件类型识别异常 [{}]: {}", filePath, e.what());
            promise.set_value("Unknown");
        }
        
        return future;
    }

    bool isServiceAvailable() const override {
        return initialized_ && metadataService_;
    }

    std::string getServiceStatus() const override {
        if (!initialized_) {
            return "未初始化";
        }
        
        std::vector<std::string> status;
        status.push_back("DAS:暂未集成");
        status.push_back(metadataService_ ? "MDS:可用" : "MDS:不可用");
        status.push_back("CRS:暂未集成");
        
        std::string result;
        for (size_t i = 0; i < status.size(); ++i) {
            if (i > 0) result += ", ";
            result += status[i];
        }
        
        return result;
    }

    void shutdown() override {
        LOG_INFO("关闭核心服务代理");
        
        metadataService_.reset();
        initialized_ = false;
    }

private:
    std::shared_ptr<oscean::core_services::metadata::IUnifiedMetadataService> metadataService_;
    bool initialized_ = false;
};

// === 工厂函数实现 ===

std::shared_ptr<CoreServiceProxy> createCoreServiceProxy() {
    return std::make_shared<CoreServiceProxyImpl>();
}

std::shared_ptr<CoreServiceProxy> createCoreServiceProxy(
    std::shared_ptr<oscean::core_services::metadata::IUnifiedMetadataService> mds) {
    
    auto proxy = std::make_shared<CoreServiceProxyImpl>();
    static_cast<CoreServiceProxyImpl*>(proxy.get())->setMetadataService(mds);
    
    return proxy;
}

} // namespace oscean::workflow_engine::proxies 