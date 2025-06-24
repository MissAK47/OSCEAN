# 元数据服务延迟初始化使用指南

## 概述

元数据服务现在支持**延迟初始化**模式，数据库不会在服务启动时全部创建，而是根据实际存储需求按需创建。这样可以避免创建不需要的数据库，提高资源利用效率。

## 设计原则

✅ **职责分离**：元数据服务不进行文件扫描，只响应存储请求  
✅ **延迟初始化**：只在需要时创建数据库  
✅ **动态管理**：支持运行时添加新的数据库类型  
✅ **资源优化**：避免创建不需要的数据库

## 工作流程

### 1. 外部扫描模块进行文件扫描和分类

```cpp
// 示例：数据访问服务或应用程序进行文件扫描
#include "core_services/data_access/i_raw_data_access_service.h"
#include "core_services/metadata/i_metadata_service.h"

class DataIndexer {
private:
    std::shared_ptr<IMetadataService> metadataService_;
    std::shared_ptr<IRawDataAccessService> dataAccessService_;
    
public:
    /**
     * @brief 扫描指定目录并处理文件
     */
    void processDirectory(const std::string& directoryPath) {
        // 1. 扫描目录获取文件列表
        auto files = scanDirectory(directoryPath);
        
        // 2. 对每个文件进行元数据提取和存储
        for (const auto& filePath : files) {
            processFile(filePath);
        }
    }
    
private:
    std::vector<std::string> scanDirectory(const std::string& dir) {
        std::vector<std::string> files;
        
        for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                // 只处理支持的文件格式
                if (ext == ".nc" || ext == ".netcdf" || ext == ".shp" || 
                    ext == ".tif" || ext == ".tiff" || ext == ".h5") {
                    files.push_back(entry.path().string());
                }
            }
        }
        
        return files;
    }
    
    void processFile(const std::string& filePath) {
        try {
            // 1. 使用数据访问服务提取元数据
            auto extractResult = dataAccessService_->extractMetadataAsync(filePath).get();
            
            if (!extractResult.isSuccess()) {
                std::cerr << "提取元数据失败: " << filePath << std::endl;
                return;
            }
            
            auto extractedMetadata = extractResult.getData();
            
            // 2. 将元数据存储到元数据服务
            // 元数据服务会根据数据类型自动创建相应的数据库
            auto storeResult = metadataService_->storeMetadataAsync(extractedMetadata).get();
            
            if (storeResult.isSuccess()) {
                std::cout << "✅ 文件处理完成: " << filePath 
                         << " -> 元数据ID: " << storeResult.getData() << std::endl;
            } else {
                std::cerr << "❌ 元数据存储失败: " << filePath 
                         << " 错误: " << storeResult.getError() << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "❌ 处理文件异常: " << filePath << " 错误: " << e.what() << std::endl;
        }
    }
};
```

### 2. 元数据服务自动处理数据库创建

```cpp
// 元数据服务内部流程 (用户无需关心)
class MetadataServiceImpl : public IMetadataService {
public:
    boost::future<AsyncResult<std::string>> storeMetadataAsync(
        const ExtractedMetadata& metadata,
        const StorageOptions& options = {}) override {
        
        // 1. 分析元数据，确定数据库类型
        DataType dataType = determineDataType(metadata);
        DatabaseType dbType = mapDataTypeToDatabaseType(dataType);
        
        // 2. 自动确保数据库存在（延迟初始化）
        return databaseManager_->storeMetadataAsync(dbType, metadata);
    }
    
private:
    DataType determineDataType(const ExtractedMetadata& metadata) {
        return fileClassifier_->classifyFileAsync(metadata.filePath, metadata.variables).get().getData();
    }
    
    DatabaseType mapDataTypeToDatabaseType(DataType dataType) {
        switch (dataType) {
            case DataType::OCEAN_ENVIRONMENT:
                return DatabaseType::OCEAN_ENVIRONMENT;
            case DataType::TOPOGRAPHY_BATHYMETRY:
                return DatabaseType::TOPOGRAPHY_BATHYMETRY;
            case DataType::BOUNDARY_LINES:
                return DatabaseType::BOUNDARY_LINES;
            case DataType::SONAR_PROPAGATION:
                return DatabaseType::SONAR_PROPAGATION;
            case DataType::TACTICAL_ENVIRONMENT:
                return DatabaseType::TACTICAL_ENVIRONMENT;
            default:
                return DatabaseType::OCEAN_ENVIRONMENT;  // 默认类型
        }
    }
};
```

### 3. 数据库管理器实现延迟初始化

```cpp
class MultiDatabaseManager {
public:
    boost::future<AsyncResult<std::string>> storeMetadataAsync(
        DatabaseType dbType,
        const ExtractedMetadata& metadata) {
        
        return boost::async([this, dbType, metadata]() {
            // 🔥 核心：确保数据库存在（如果不存在则创建）
            if (!ensureDatabaseExists(dbType)) {
                return AsyncResult<std::string>::failure("无法创建数据库");
            }
            
            // 使用已存在的数据库进行存储
            auto adapter = databaseAdapters_[dbType];
            return adapter->storeMetadataAsync(metadata).get();
        });
    }
    
private:
    bool ensureDatabaseExists(DatabaseType dbType) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 如果已初始化，直接返回
        if (initializedDatabases_.find(dbType) != initializedDatabases_.end()) {
            return true;
        }
        
        // 按需初始化数据库
        return initializeDatabaseOnDemand(dbType);
    }
};
```

## 实际使用示例

### 完整的应用程序示例

```cpp
#include "core_services/metadata/i_metadata_service.h"
#include "core_services/data_access/i_raw_data_access_service.h"
#include "core_services/metadata/metadata_service_factory.h"
#include "common_utils/infrastructure/common_services_factory.h"

int main() {
    try {
        // 1. 创建通用服务工厂
        auto commonFactory = std::make_shared<CommonServicesFactory>();
        
        // 2. 创建元数据服务（延迟初始化模式）
        auto metadataFactory = MetadataServiceFactory::createHighPerformance(commonFactory);
        auto metadataService = metadataFactory->createMetadataService();
        
        // 3. 创建数据访问服务
        auto dataAccessService = createDataAccessService();
        
        // 4. 处理用户指定的目录
        std::string userDirectory = "/path/to/user/data";
        
        std::cout << "🔍 开始扫描目录: " << userDirectory << std::endl;
        
        // 5. 扫描并处理文件
        DataIndexer indexer(metadataService, dataAccessService);
        indexer.processDirectory(userDirectory);
        
        // 6. 查看已创建的数据库统计
        auto dbManager = getMultiDatabaseManager(metadataService);
        auto stats = dbManager->getStatistics();
        
        std::cout << "\n📊 数据库创建统计:" << std::endl;
        std::cout << "已初始化数据库数量: " << stats.totalInitializedDatabases << std::endl;
        
        for (const auto& [dbType, creationTime] : stats.creationTimes) {
            std::cout << "数据库类型 " << static_cast<int>(dbType) 
                     << " 创建时间: " << formatTime(creationTime) << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 应用程序异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## 优势

### 1. 资源优化
- **只创建需要的数据库**：如果用户目录中只有海洋环境数据，则只会创建 `ocean_environment.db`
- **避免空数据库**：不会预先创建所有5种类型的数据库

### 2. 灵活性
- **动态适应**：根据实际数据类型动态创建对应数据库
- **支持扩展**：新增数据类型时，可以轻松添加新的数据库类型

### 3. 性能
- **启动快速**：服务启动时不需要初始化所有数据库
- **按需优化**：只对实际使用的数据库进行性能优化

## 数据库文件说明

根据实际数据内容，可能创建的数据库文件：

```
./databases/
├── ocean_environment.db      # 海洋环境数据（温度、盐度、流速等）
├── topography_bathymetry.db  # 地形底质数据（深度、高程等）
├── boundary_lines.db         # 边界线数据（海岸线、边界等）
├── sonar_propagation.db      # 声纳传播数据（传播损失、探测概率等）
└── tactical_environment.db   # 战术环境数据（声道深度、汇聚区等）
```

**注意**：只有当实际存储对应类型的数据时，数据库文件才会被创建。

## 监控和管理

### 查询已初始化的数据库

```cpp
auto dbManager = getMultiDatabaseManager(metadataService);

// 获取已初始化的数据库类型
auto initializedTypes = dbManager->getInitializedDatabaseTypes();

std::cout << "已初始化的数据库:" << std::endl;
for (auto dbType : initializedTypes) {
    std::cout << "- " << getDatabaseTypeName(dbType) << std::endl;
}
```

### 预热数据库（可选）

```cpp
// 如果知道将要处理特定类型的数据，可以预先创建数据库
std::vector<DatabaseType> expectedTypes = {
    DatabaseType::OCEAN_ENVIRONMENT,
    DatabaseType::TOPOGRAPHY_BATHYMETRY
};

size_t preWarmedCount = dbManager->preWarmDatabases(expectedTypes);
std::cout << "预热了 " << preWarmedCount << " 个数据库" << std::endl;
```

### 获取统计信息

```cpp
auto stats = dbManager->getStatistics();

std::cout << "数据库管理器统计:" << std::endl;
std::cout << "启动时间: " << formatTime(stats.managerStartTime) << std::endl;
std::cout << "已初始化数据库: " << stats.totalInitializedDatabases << std::endl;

for (const auto& [dbType, count] : stats.recordCounts) {
    std::cout << "数据库 " << static_cast<int>(dbType) 
             << " 记录数: " << count << std::endl;
}
```

## 总结

新的延迟初始化系统实现了以下目标：

1. **职责分离**：元数据服务专注于元数据管理，不进行文件扫描
2. **资源优化**：只创建实际需要的数据库
3. **使用简单**：外部调用者无需关心数据库创建细节
4. **自动化**：数据库根据元数据类型自动创建和管理

这种设计既保持了模块间的清晰职责分离，又实现了高效的资源利用。 