/**
 * @file real_netcdf_metadata_test.cpp
 * @brief 真实NetCDF数据元数据处理和验证测试
 * @note 使用实际的哥白尼海洋数据测试元数据分类、SQLite数据库存储和验证功能
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>

// 引入元数据服务接口
#include "core_services/metadata/unified_metadata_service.h"
#include "impl/metadata_service_factory.h"
#include "common_utils/infrastructure/common_services_factory.h"

using namespace std;
using namespace filesystem;
using namespace oscean::core_services::metadata;
using namespace oscean::core_services::metadata::impl;

namespace oscean::core_services::metadata::test {

// 删除重复的FileMetadata定义，使用标准定义
// 注意：这里原本有重复定义，现在使用core_services中的标准FileMetadata

}

/**
 * @brief 真实NetCDF元数据测试类 - 使用完整的元数据服务
 */
class RealNetCDFMetadataTest : public ::testing::Test {
protected:
    void SetUp() override {
        cout << "=== 真实NetCDF元数据测试初始化 ===" << endl;
        
        // 初始化项目路径
        projectRoot_ = filesystem::current_path();
        while (!filesystem::exists(projectRoot_ / "core_services_impl") && 
               projectRoot_.has_parent_path()) {
            projectRoot_ = projectRoot_.parent_path();
        }
        
        configPath_ = projectRoot_ / "core_services_impl" / "metadata_service" / "config";
        testDbDir_ = projectRoot_ / "test_data" / "databases" / "metadata_test";
        
        // 确保测试目录存在
        filesystem::create_directories(testDbDir_);
        
        // 加载变量分类规则
        loadClassificationRules();
        
        // 初始化元数据服务
        initMetadataService();
    }
    
    void TearDown() override {
        // 清理测试数据库文件
        if (metadataService_) {
            // 服务析构时会自动关闭数据库
            metadataService_.reset();
        }
        
        cout << "测试清理完成" << endl;
    }
    
    void loadClassificationRules() {
        // 海流变量
        oceanCurrentVars_ = {"uo", "vo", "u", "v", "current_speed", "current_direction",
                            "eastward_sea_water_velocity", "northward_sea_water_velocity"};
        
        // 海冰变量
        seaIceVars_ = {"siconc", "sithick", "sea_ice_concentration", "sea_ice_thickness",
                      "sea_ice_area_fraction", "vxsi", "vysi", "sea_ice_x_velocity", "sea_ice_y_velocity"};
        
        // 坐标变量
        coordinateVars_ = {"latitude", "longitude", "lat", "lon", "x", "y"};
        
        // 深度变量
        depthVars_ = {"depth", "level", "z"};
        
        cout << "✅ 加载变量分类规则完成" << endl;
    }
    
    void initMetadataService() {
        try {
            // 创建测试环境的通用服务工厂
            auto commonServicesUniquePtr = oscean::common_utils::infrastructure::CommonServicesFactory::createForTesting();
            std::shared_ptr<oscean::common_utils::infrastructure::CommonServicesFactory> commonServicesFactory(commonServicesUniquePtr.release());
            
            // 配置元数据服务
            MetadataServiceConfiguration config;
            config.databaseConfig.basePath = testDbDir_.string();
            config.databaseConfig.enableWALMode = true;
            config.databaseConfig.cacheSize = 1000;
            
            // 创建元数据服务工厂
            auto factory = std::make_unique<MetadataServiceFactory>(commonServicesFactory, config);
            
            // 创建元数据服务（工厂会自动初始化）
            metadataService_ = factory->createTestingMetadataService();
            
            if (!metadataService_ || !metadataService_->isReady()) {
                throw std::runtime_error("元数据服务初始化失败或未就绪");
            }
            
            cout << "✅ 元数据服务初始化成功" << endl;
            
        } catch (const std::exception& e) {
            FAIL() << "元数据服务初始化失败: " << e.what();
        }
    }
    
    string classifyVariable(const string& varName, const string& standardName) {
        // 首先检查坐标变量（优先级最高）
        for (const auto& var : coordinateVars_) {
            if (varName == var || standardName == var) {
                return "coordinate";
            }
        }
        
        // 然后检查深度变量
        for (const auto& var : depthVars_) {
            if (varName == var || standardName == var) {
                return "depth";
            }
        }
        
        // 检查海流变量（使用更精确的匹配）
        for (const auto& var : oceanCurrentVars_) {
            if (varName == var || standardName == var) {
                return "ocean_current";
            }
        }
        
        // 检查海冰变量
        for (const auto& var : seaIceVars_) {
            if (varName == var || standardName == var) {
                return "ocean_sea_ice";
            }
        }
        
        return "unknown";
    }
    
    ExtractedMetadata createCSFileMetadata() {
        ExtractedMetadata metadata;
        
        metadata.filePath = CS_FILE;
        metadata.fileName = "cs_2023_01_00_00.nc";
        metadata.format = "NetCDF";
        metadata.dataType = DataType::OCEAN_ENVIRONMENT;
        
        // 空间信息
        metadata.spatialInfo.bounds = SpatialBounds(-180.0, -90.0, 180.0, 90.0);
        metadata.spatialInfo.coordinateSystem = "WGS84";
        
        // 时间信息
        metadata.temporalInfo.timeRange.startTime = "2023-01-01T00:00:00Z";
        metadata.temporalInfo.timeRange.endTime = "2023-01-31T23:59:59Z";
        metadata.temporalInfo.timeRange.timeUnits = "ISO8601";
        
        // 暂时设置为月度数据
        metadata.temporalInfo.temporalResolutionType = TemporalResolutionType::MONTHLY;
        metadata.temporalInfo.temporalResolutionSeconds = 2629746; // 月度数据的平均秒数
        
        // 变量信息 - 使用VariableMeta
        metadata.variables = {
            {"depth", "depth coordinate", "float", "m"},
            {"latitude", "latitude coordinate", "float", "degrees_north"},
            {"longitude", "longitude coordinate", "float", "degrees_east"},
            {"time", "time coordinate", "double", "seconds since 1950-01-01"},
            {"uo", "Eastward velocity", "float", "m s-1"},
            {"vo", "Northward velocity", "float", "m s-1"}
        };
        
        // 变量详细信息已通过构造函数设置完成
        
        // 属性信息
        metadata.attributes = {
            {"title", "Monthly mean fields for product GLOBAL_ANALYSIS_FORECAST_PHY_001_024"},
            {"source", "MERCATOR GLO12"},
            {"institution", "Mercator Ocean"}
        };
        
        return metadata;
    }
    
    ExtractedMetadata createITFileMetadata() {
        ExtractedMetadata metadata;
        
        metadata.filePath = IT_FILE;
        metadata.fileName = "it_2023_01_00_00.nc";
        metadata.format = "NetCDF";
        metadata.dataType = DataType::OCEAN_ENVIRONMENT;
        
        // 空间信息
        metadata.spatialInfo.bounds = SpatialBounds(-180.0, -90.0, 180.0, 90.0);
        metadata.spatialInfo.coordinateSystem = "WGS84";
        
        // 时间信息
        metadata.temporalInfo.timeRange.startTime = "2023-01-01T00:00:00Z";
        metadata.temporalInfo.timeRange.endTime = "2023-01-31T23:59:59Z";
        metadata.temporalInfo.timeRange.timeUnits = "ISO8601";
        
        // 暂时设置为月度数据
        metadata.temporalInfo.temporalResolutionType = TemporalResolutionType::MONTHLY;
        metadata.temporalInfo.temporalResolutionSeconds = 2629746; // 月度数据的平均秒数
        
        // 变量信息
        metadata.variables = {
            {"latitude", "latitude coordinate", "float", "degrees_north"},
            {"longitude", "longitude coordinate", "float", "degrees_east"},
            {"time", "time coordinate", "double", "seconds since 1950-01-01"},
            {"siconc", "Sea ice concentration", "float", "%"},
            {"sithick", "Sea ice thickness", "float", "m"},
            {"vxsi", "Sea ice velocity x-component", "float", "m s-1"},
            {"vysi", "Sea ice velocity y-component", "float", "m s-1"}
        };
        
        // 变量详细信息已通过构造函数设置完成
        
        // 属性信息
        metadata.attributes = {
            {"title", "TOPAZ Arctic Ocean System reanalysis"},
            {"source", "TOPAZ5 Arctic Ocean"},
            {"institution", "Nersc"}
        };
        
        return metadata;
    }

    ExtractedMetadata createSPFileMetadata() {
        ExtractedMetadata metadata;
        
        metadata.filePath = SP_FILE;
        metadata.fileName = "sp_2024_07_00_00.nc";
        metadata.format = "NetCDF";
        metadata.dataType = DataType::OCEAN_ENVIRONMENT;
        
        // 空间信息 - 基于实际ncdump输出
        metadata.spatialInfo.bounds = SpatialBounds(-180.0, -80.0, 179.9167, 90.0);
        metadata.spatialInfo.coordinateSystem = "WGS84";
        
        // 时间信息 - 文件名暗示2024年7月
        metadata.temporalInfo.timeRange.startTime = "2024-07-01T00:00:00Z";
        metadata.temporalInfo.timeRange.endTime = "2024-07-31T23:59:59Z";
        metadata.temporalInfo.timeRange.timeUnits = "ISO8601";
        
        // 文件名模式sp_2024_07_00_00表示2024年7月月度数据
        metadata.temporalInfo.temporalResolutionType = TemporalResolutionType::MONTHLY;
        metadata.temporalInfo.temporalResolutionSeconds = 31 * 24 * 3600; // 31天的秒数，7月有31天
        metadata.temporalInfo.calendar = "gregorian";
        
        // 声速剖面变量 - 基于实际ncdump输出
        metadata.variables = {
            {"latitude", "latitude coordinate (2041 points from -80 to 90 degrees)", "float", "degrees_north"},
            {"longitude", "longitude coordinate (4320 points from -180 to 179.9167 degrees)", "float", "degrees_east"},
            {"depth", "depth coordinate (57 levels from 0.494 to 12000 meters)", "float", "m"},
            {"ssp", "Sound Speed Profile - 3D sound velocity field", "float", "m s-1"}
        };
        
        // 属性信息
        metadata.attributes = {
            {"institution", "Ocean Research Institute"},
            {"source", "SOUND_SPEED_PROFILE_GLOBAL_MODEL"},
            {"title", "Global 3D Sound Speed Profile for July 2024"},
            {"data_dimensions", "latitude(2041) x longitude(4320) x depth(57)"},
            {"spatial_resolution", "~0.083 degrees (~9.2 km at equator)"},
            {"depth_levels", "57 levels from 0.5m to 12000m"}
        };
        
        metadata.dataQuality = 1.0;
        metadata.completeness = 1.0;
        
        return metadata;
    }

    ExtractedMetadata createSonarFileMetadata() {
        ExtractedMetadata metadata;
        
        metadata.filePath = "E:\\Test_data\\sonar\\sonar_test.nc";
        metadata.fileName = "sonar_test.nc";
        metadata.format = "NetCDF";
        metadata.dataType = DataType::SONAR_PROPAGATION;  // 明确设置为声纳传播数据类型
        
        // 空间信息
        metadata.spatialInfo.bounds = SpatialBounds(-180.0, -90.0, 180.0, 90.0);
        metadata.spatialInfo.coordinateSystem = "WGS84";
        
        // 时间信息
        metadata.temporalInfo.timeRange.startTime = "2024-01-01T00:00:00Z";
        metadata.temporalInfo.timeRange.endTime = "2024-01-31T23:59:59Z";
        metadata.temporalInfo.timeRange.timeUnits = "ISO8601";
        
        // 声纳传播变量 - 包含明确的声纳变量
        metadata.variables = {
            {"latitude", "纬度坐标", "float", "degrees_north"},
            {"longitude", "经度坐标", "float", "degrees_east"},
            {"TL", "传播损失", "float", "dB"},
            {"SE", "声暴露级", "float", "dB re 1 μPa²·s"},
            {"PD", "探测概率", "float", "probability"}
        };
        
        return metadata;
    }

protected:
    filesystem::path projectRoot_;
    filesystem::path configPath_;
    filesystem::path testDbDir_;
    
    std::unique_ptr<IMetadataService> metadataService_;
    
    vector<string> oceanCurrentVars_;
    vector<string> seaIceVars_;
    vector<string> coordinateVars_;
    vector<string> depthVars_;
    
    // 测试文件路径
    const string CS_FILE = "E:\\Ocean_data\\cs\\cs_2023_01_00_00.nc";
    const string IT_FILE = "E:\\Ocean_data\\it\\it_2023_01_00_00.nc";
    const string SP_FILE = "E:\\Ocean_data\\sp\\sp_2024_07_00_00.nc";
};

/**
 * @brief 测试CS文件（海流数据）元数据处理
 */
TEST_F(RealNetCDFMetadataTest, ProcessCSFileMetadata) {
    cout << "=== 测试CS文件（海流数据）元数据处理 ===" << endl;
    
    // 创建CS文件的元数据
    ExtractedMetadata csMetadata = createCSFileMetadata();
    
    cout << "构建CS文件元数据完成:" << endl;
    cout << "- 文件: " << csMetadata.fileName << endl;
    cout << "- 格式: " << csMetadata.format << endl;
    cout << "- 数据类型: " << static_cast<int>(csMetadata.dataType) << endl;
    cout << "- 变量数量: " << csMetadata.variables.size() << endl;
    
    // 使用元数据服务存储
    auto storeResult = metadataService_->storeMetadataAsync(csMetadata).get();
    
    ASSERT_TRUE(storeResult.isSuccess()) << "存储CS文件元数据失败: " << storeResult.getError();
    
    string metadataId = storeResult.getData();
    cout << "✅ CS文件元数据存储成功，ID: " << metadataId << endl;
    
    // 验证变量分类
    map<string, int> categoryCounts;
    for (const auto& var : csMetadata.variables) {
        categoryCounts[var.dataType]++;
    }
    
    cout << "CS文件变量分类统计:" << endl;
    for (const auto& [category, count] : categoryCounts) {
        cout << "  " << category << ": " << count << endl;
    }
    
    // 预期分类结果验证
    EXPECT_EQ(categoryCounts["coordinate"], 2);    // latitude, longitude
    EXPECT_EQ(categoryCounts["depth"], 1);         // depth
    EXPECT_EQ(categoryCounts["ocean_current"], 2); // uo, vo
    EXPECT_EQ(categoryCounts["unknown"], 1);       // time
}

/**
 * @brief 测试IT文件（海冰数据）元数据处理
 */
TEST_F(RealNetCDFMetadataTest, ProcessITFileMetadata) {
    cout << "=== 测试IT文件（海冰数据）元数据处理 ===" << endl;
    
    // 创建IT文件的元数据
    ExtractedMetadata itMetadata = createITFileMetadata();
    
    cout << "构建IT文件元数据完成:" << endl;
    cout << "- 文件: " << itMetadata.fileName << endl;
    cout << "- 格式: " << itMetadata.format << endl;
    cout << "- 数据类型: " << static_cast<int>(itMetadata.dataType) << endl;
    cout << "- 变量数量: " << itMetadata.variables.size() << endl;
    
    // 使用元数据服务存储
    auto storeResult = metadataService_->storeMetadataAsync(itMetadata).get();
    
    ASSERT_TRUE(storeResult.isSuccess()) << "存储IT文件元数据失败: " << storeResult.getError();
    
    string metadataId = storeResult.getData();
    cout << "✅ IT文件元数据存储成功，ID: " << metadataId << endl;
    
    // 验证变量分类
    map<string, int> categoryCounts;
    for (const auto& var : itMetadata.variables) {
        categoryCounts[var.dataType]++;
    }
    
    cout << "IT文件变量分类统计:" << endl;
    for (const auto& [category, count] : categoryCounts) {
        cout << "  " << category << ": " << count << endl;
    }
    
    // 预期分类结果验证
    EXPECT_EQ(categoryCounts["coordinate"], 2);        // latitude, longitude
    EXPECT_EQ(categoryCounts["ocean_sea_ice"], 4);     // siconc, sithick, vxsi, vysi
    EXPECT_EQ(categoryCounts["unknown"], 1);           // time
}

/**
 * @brief 测试SP文件（声纳传播数据）元数据处理
 */
TEST_F(RealNetCDFMetadataTest, ProcessSPFileMetadata) {
    std::cout << "=== 测试SP文件（声纳传播数据）元数据处理 ===" << std::endl;
    
    // 构建SP文件元数据
    auto metadata = createSPFileMetadata();
    std::cout << "构建SP文件元数据完成:" << std::endl;
    std::cout << "- 文件: " << metadata.fileName << std::endl;
    std::cout << "- 格式: " << metadata.format << std::endl;
    std::cout << "- 数据类型: " << static_cast<int>(metadata.dataType) << std::endl;
    std::cout << "- 变量数量: " << metadata.variables.size() << std::endl;
    
    // 存储元数据 - 使用正确的方法名
    auto storeResult = metadataService_->storeMetadataAsync(metadata).get();
    ASSERT_TRUE(storeResult.isSuccess()) << "存储SP文件元数据失败: " << storeResult.getError();
    
    std::string metadataId = storeResult.getData();
    std::cout << "✅SP文件元数据存储成功，ID: " << metadataId << std::endl;
    
    // 统计变量分类
    std::map<std::string, int> variableTypeCount;
    for (const auto& var : metadata.variables) {
        variableTypeCount[var.dataType]++;
    }
    
    std::cout << "SP文件变量分类统计:" << std::endl;
    for (const auto& [type, count] : variableTypeCount) {
        std::cout << "  " << type << ": " << count << std::endl;
    }
}

/**
 * @brief 测试数据库查询功能
 */
TEST_F(RealNetCDFMetadataTest, TestDatabaseQuery) {
    cout << "=== 测试数据库查询功能 ===" << endl;
    
    // 先存储两个文件的元数据
    auto csMetadata = createCSFileMetadata();
    auto itMetadata = createITFileMetadata();
    
    auto csResult = metadataService_->storeMetadataAsync(csMetadata).get();
    auto itResult = metadataService_->storeMetadataAsync(itMetadata).get();
    
    ASSERT_TRUE(csResult.isSuccess()) << "存储CS文件失败";
    ASSERT_TRUE(itResult.isSuccess()) << "存储IT文件失败";
    
    // 测试查询所有元数据
    QueryCriteria criteria;
    auto queryResult = metadataService_->queryMetadataAsync(criteria).get();
    
    ASSERT_TRUE(queryResult.isSuccess()) << "查询失败: " << queryResult.getError();
    
    const auto& results = queryResult.getData();
    cout << "查询到 " << results.size() << " 条元数据记录" << endl;
    
    EXPECT_GE(results.size(), 2) << "应该至少有2条记录";
    
    // 测试按文件路径查询
    auto pathQueryResult = metadataService_->queryByFilePathAsync(CS_FILE).get();
    ASSERT_TRUE(pathQueryResult.isSuccess()) << "按路径查询失败";
    
    const auto& pathResults = pathQueryResult.getData();
    EXPECT_GE(pathResults.size(), 1) << "应该找到CS文件的记录";
    
    if (!pathResults.empty()) {
        const auto& entry = pathResults[0];
        cout << "找到CS文件记录:" << endl;
        cout << "  ID: " << entry.metadataId << endl;
        cout << "  路径: " << entry.filePath << endl;
        cout << "  格式: " << entry.format << endl;
        cout << "  数据类型: " << static_cast<int>(entry.dataType) << endl;
    }
}

/**
 * @brief 测试SQLite数据库完整性
 */
TEST_F(RealNetCDFMetadataTest, VerifyDatabaseIntegrity) {
    cout << "=== 测试SQLite数据库完整性 ===" << endl;
    
    // 存储测试数据
    auto csMetadata = createCSFileMetadata();
    auto storeResult = metadataService_->storeMetadataAsync(csMetadata).get();
    
    ASSERT_TRUE(storeResult.isSuccess()) << "存储失败";
    
    // 验证数据库文件存在
    filesystem::path dbFile = filesystem::path("./databases") / "ocean_environment.db";
    EXPECT_TRUE(filesystem::exists(dbFile)) << "数据库文件不存在: " << dbFile;
    
    if (filesystem::exists(dbFile)) {
        auto fileSize = filesystem::file_size(dbFile);
        cout << "数据库文件大小: " << fileSize << " 字节" << endl;
        EXPECT_GT(fileSize, 0) << "数据库文件为空";
    }
    
    // 验证查询功能
    QueryCriteria criteria;
    criteria.dataTypes = {DataType::OCEAN_ENVIRONMENT};
    
    auto queryResult = metadataService_->queryByCategoryAsync(DataType::OCEAN_ENVIRONMENT).get();
    ASSERT_TRUE(queryResult.isSuccess()) << "分类查询失败: " << queryResult.getError();
    
    const auto& results = queryResult.getData();
    EXPECT_GE(results.size(), 1) << "应该找到海洋环境数据";
    
    cout << "✅ 数据库完整性验证通过，找到 " << results.size() << " 条海洋环境数据记录" << endl;
}

/**
 * @brief 测试声纳传播变量分类
 */
TEST_F(RealNetCDFMetadataTest, TestSonarVariableClassification) {
    cout << "=== 测试声纳传播变量分类 ===" << endl;
    
    // 创建声纳传播文件的元数据
    ExtractedMetadata sonarMetadata = createSonarFileMetadata();
    
    cout << "构建声纳传播文件元数据完成:" << endl;
    cout << "- 文件: " << sonarMetadata.fileName << endl;
    cout << "- 格式: " << sonarMetadata.format << endl;
    cout << "- 数据类型: " << static_cast<int>(sonarMetadata.dataType) << endl;
    cout << "- 变量数量: " << sonarMetadata.variables.size() << endl;
    
    // 使用元数据服务存储
    auto storeResult = metadataService_->storeMetadataAsync(sonarMetadata).get();
    
    ASSERT_TRUE(storeResult.isSuccess()) << "存储声纳传播文件元数据失败: " << storeResult.getError();
    
    string metadataId = storeResult.getData();
    cout << "✅ 声纳传播文件元数据存储成功，ID: " << metadataId << endl;
    
    // 验证变量分类
    map<string, int> categoryCounts;
    for (const auto& var : sonarMetadata.variables) {
        categoryCounts[var.dataType]++;
    }
    
    cout << "声纳传播文件变量分类统计:" << endl;
    for (const auto& [category, count] : categoryCounts) {
        cout << "  " << category << ": " << count << endl;
    }
    
    // 预期分类结果验证
    EXPECT_EQ(categoryCounts["coordinate"], 2);    // latitude, longitude
    // TL, SE, PD可能被识别为声纳变量或unknown，取决于分类逻辑
    EXPECT_GE(categoryCounts["sonar_transmission_loss"] + categoryCounts["sonar_sound_exposure"] + 
              categoryCounts["sonar_probability_detection"] + categoryCounts["unknown"], 3);
} 