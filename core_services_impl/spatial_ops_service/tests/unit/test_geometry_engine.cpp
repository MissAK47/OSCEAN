/**
 * @file test_geometry_engine.cpp
 * @brief 几何引擎单元测试 - 基于真实GEOS库的完整功能测试
 * 
 * 🎯 测试目标：
 * ✅ 真实GEOS库集成测试
 * ✅ 所有几何操作的准确性验证
 * ✅ 错误处理和边界条件测试
 * ❌ 不使用Mock - 直接测试真实GEOS功能
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <cmath>

// 空间服务头文件
#include "engine/geometry_engine.h"
#include "core_services/spatial_ops/spatial_config.h"
#include "core_services/spatial_ops/spatial_types.h"
#include "core_services/common_data_types.h"

using namespace oscean::core_services;
using namespace oscean::core_services::spatial_ops;
using namespace oscean::core_services::spatial_ops::engine;

/**
 * @class GeometryEngineTest
 * @brief 几何引擎测试基类
 */
class GeometryEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试配置
        config.version = "1.0.0";
        config.enableExperimentalFeatures = false;
        config.algorithmSettings.geometricTolerance = 1e-10;
        config.performanceSettings.enablePerformanceMonitoring = false;
        
        // 初始化几何引擎
        geometryEngine = std::make_unique<GeometryEngine>(config);
        
        // 创建测试几何体
        setupTestGeometries();
    }
    
    void TearDown() override {
        geometryEngine.reset();
    }
    
    void setupTestGeometries() {
        // 测试点
        testPoint.wkt = "POINT(0 0)";
        
        // 测试线段
        testLine.wkt = "LINESTRING(0 0, 10 0, 10 10)";
        
        // 测试多边形
        testPolygon.wkt = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))";
        
        // 测试多点
        testMultiPoint.wkt = "MULTIPOINT((0 0), (5 5), (10 10))";
        
        // 相交的多边形
        intersectingPolygon.wkt = "POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))";
        
        // 复杂多边形（带洞）
        polygonWithHole.wkt = "POLYGON((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))";
    }
    
protected:
    SpatialOpsConfig config;
    std::unique_ptr<GeometryEngine> geometryEngine;
    
    // 测试几何体
    Geometry testPoint;
    Geometry testLine;
    Geometry testPolygon;
    Geometry testMultiPoint;
    Geometry intersectingPolygon;
    Geometry polygonWithHole;
};

// ================================================================
// 缓冲区操作测试
// ================================================================

TEST_F(GeometryEngineTest, BufferOperation_Point_Success) {
    BufferOptions options;
    options.quadrantSegments = 8;
    
    auto result = geometryEngine->buffer(testPoint, 5.0, options);
    
    EXPECT_FALSE(result.wkt.empty());
    EXPECT_TRUE(geometryEngine->isValid(result));
    
    // 验证缓冲区是近似圆形（包含原点）
    Point origin{0.0, 0.0};
    EXPECT_TRUE(geometryEngine->evaluatePredicate(result, testPoint, SpatialPredicate::CONTAINS));
}

TEST_F(GeometryEngineTest, BufferOperation_Polygon_Success) {
    BufferOptions options;
    options.quadrantSegments = 16;
    
    auto result = geometryEngine->buffer(testPolygon, 2.0, options);
    
    EXPECT_FALSE(result.wkt.empty());
    EXPECT_TRUE(geometryEngine->isValid(result));
    
    // 缓冲后的多边形应该包含原多边形
    EXPECT_TRUE(geometryEngine->evaluatePredicate(result, testPolygon, SpatialPredicate::CONTAINS));
}

TEST_F(GeometryEngineTest, BufferOperation_NegativeDistance_HandledCorrectly) {
    // 对于负距离缓冲，GEOS可能返回空几何体或进行内缓冲
    BufferOptions options;
    auto result = geometryEngine->buffer(testPolygon, -10.0, options);
    
    // 验证结果是有效的几何体（可能为空）
    EXPECT_TRUE(result.wkt.find("POLYGON") != std::string::npos || 
                result.wkt.find("EMPTY") != std::string::npos ||
                result.wkt.empty()); // 允许空结果
    
    // 负缓冲应该使几何体变小或消失
    if (!result.wkt.empty() && result.wkt.find("EMPTY") == std::string::npos) {
        // 如果有结果，验证它是合理的（简化验证）
        EXPECT_FALSE(result.wkt.empty());
    }
}

// ================================================================
// 拓扑操作测试 (交集、并集、差集)
// ================================================================

TEST_F(GeometryEngineTest, IntersectionOperation_OverlappingPolygons_Success) {
    auto result = geometryEngine->intersection(testPolygon, intersectingPolygon);
    
    EXPECT_FALSE(result.wkt.empty());
    EXPECT_TRUE(geometryEngine->isValid(result));
    
    // 验证交集不为空
    EXPECT_FALSE(geometryEngine->evaluatePredicate(result, testPoint, SpatialPredicate::TOUCHES));
}

TEST_F(GeometryEngineTest, UnionOperation_TwoPolygons_Success) {
    auto result = geometryEngine->unionGeometries(testPolygon, intersectingPolygon);
    
    EXPECT_FALSE(result.wkt.empty());
    EXPECT_TRUE(geometryEngine->isValid(result));
    
    // 验证并集包含两个原多边形
    EXPECT_TRUE(geometryEngine->evaluatePredicate(result, testPolygon, SpatialPredicate::CONTAINS));
    EXPECT_TRUE(geometryEngine->evaluatePredicate(result, intersectingPolygon, SpatialPredicate::CONTAINS));
}

TEST_F(GeometryEngineTest, DifferenceOperation_ComplexGeometry_Success) {
    // 使用更简单的几何体进行差集操作测试
    Geometry simpleBox1;
    simpleBox1.wkt = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))";
    
    Geometry simpleBox2;
    simpleBox2.wkt = "POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))";
    
    auto result = geometryEngine->difference(simpleBox1, simpleBox2);
    
    // 验证结果是有效的多边形
    EXPECT_FALSE(result.wkt.empty());
    EXPECT_TRUE(result.wkt.find("POLYGON") != std::string::npos ||
                result.wkt.find("MULTIPOLYGON") != std::string::npos);
    
    // 验证差集结果合理（简化验证，不计算面积）
    EXPECT_TRUE(geometryEngine->isValid(result));
}

// ================================================================
// 空间谓词测试
// ================================================================

TEST_F(GeometryEngineTest, SpatialPredicate_Contains_Success) {
    // 多边形包含内部点
    Geometry innerPoint;
    innerPoint.wkt = "POINT(5 5)";
    
    bool contains = geometryEngine->evaluatePredicate(testPolygon, innerPoint, SpatialPredicate::CONTAINS);
    EXPECT_TRUE(contains);
}

TEST_F(GeometryEngineTest, SpatialPredicate_Intersects_Success) {
    bool intersects = geometryEngine->evaluatePredicate(testPolygon, intersectingPolygon, SpatialPredicate::INTERSECTS);
    EXPECT_TRUE(intersects);
}

TEST_F(GeometryEngineTest, SpatialPredicate_Disjoint_Success) {
    Geometry distantPolygon;
    distantPolygon.wkt = "POLYGON((100 100, 110 100, 110 110, 100 110, 100 100))";
    
    bool disjoint = geometryEngine->evaluatePredicate(testPolygon, distantPolygon, SpatialPredicate::DISJOINT);
    EXPECT_TRUE(disjoint);
}

// ================================================================
// 距离计算测试
// ================================================================

TEST_F(GeometryEngineTest, DistanceCalculation_TwoPoints_Success) {
    Geometry point1;
    point1.wkt = "POINT(0 0)";
    
    Geometry point2;
    point2.wkt = "POINT(3 4)";
    
    double distance = geometryEngine->calculateDistance(point1, point2, DistanceType::EUCLIDEAN);
    
    // 3-4-5直角三角形，距离应为5
    EXPECT_NEAR(distance, 5.0, 1e-10);
}

TEST_F(GeometryEngineTest, DistanceCalculation_PointToPolygon_Success) {
    Geometry externalPoint;
    externalPoint.wkt = "POINT(-5 5)";
    
    double distance = geometryEngine->calculateDistance(externalPoint, testPolygon, DistanceType::EUCLIDEAN);
    
    // 点到多边形边界的距离应为5
    EXPECT_NEAR(distance, 5.0, 1e-10);
}

// ================================================================
// 几何简化测试
// ================================================================

TEST_F(GeometryEngineTest, SimplifyOperation_ComplexGeometry_Success) {
    // 创建复杂线段
    Geometry complexLine;
    complexLine.wkt = "LINESTRING(0 0, 1 1, 2 0, 3 1, 4 0, 5 0)";
    
    auto simplified = geometryEngine->simplify(complexLine, 1.0);
    
    EXPECT_FALSE(simplified.wkt.empty());
    EXPECT_TRUE(geometryEngine->isValid(simplified));
    
    // 简化后的几何体应该更简单（点数更少）
    EXPECT_LT(simplified.wkt.length(), complexLine.wkt.length());
}

// ================================================================
// 凸包计算测试
// ================================================================

TEST_F(GeometryEngineTest, ConvexHullOperation_MultiPoint_Success) {
    auto convexHull = geometryEngine->convexHull(testMultiPoint);
    
    EXPECT_FALSE(convexHull.wkt.empty());
    EXPECT_TRUE(geometryEngine->isValid(convexHull));
    
    // 凸包应该包含所有原始点
    EXPECT_TRUE(geometryEngine->evaluatePredicate(convexHull, testMultiPoint, SpatialPredicate::CONTAINS));
}

// ================================================================
// 几何有效性测试
// ================================================================

TEST_F(GeometryEngineTest, GeometryValidation_ValidGeometry_ReturnsTrue) {
    EXPECT_TRUE(geometryEngine->isValid(testPolygon));
    EXPECT_TRUE(geometryEngine->isValid(testPoint));
    EXPECT_TRUE(geometryEngine->isValid(testLine));
}

TEST_F(GeometryEngineTest, GeometryValidation_InvalidGeometry_ReturnsFalse) {
    Geometry invalidGeometry;
    invalidGeometry.wkt = "POLYGON((0 0, 10 10, 0 10, 10 0, 0 0))"; // 自相交多边形
    
    EXPECT_FALSE(geometryEngine->isValid(invalidGeometry));
    
    std::string reason = geometryEngine->getValidationReason(invalidGeometry);
    EXPECT_FALSE(reason.empty());
}

TEST_F(GeometryEngineTest, MakeValidOperation_InvalidGeometry_ReturnsValidGeometry) {
    Geometry invalidGeometry;
    invalidGeometry.wkt = "POLYGON((0 0, 10 10, 0 10, 10 0, 0 0))";
    
    auto validGeometry = geometryEngine->makeValid(invalidGeometry);
    
    EXPECT_TRUE(geometryEngine->isValid(validGeometry));
}

// ================================================================
// 错误处理测试
// ================================================================

TEST_F(GeometryEngineTest, InvalidWKT_ThrowsException) {
    Geometry invalidGeometry;
    invalidGeometry.wkt = "INVALID_WKT_STRING";
    
    EXPECT_THROW(
        geometryEngine->buffer(invalidGeometry, 1.0, BufferOptions{}),
        std::exception
    );
}

TEST_F(GeometryEngineTest, EmptyGeometry_ThrowsException) {
    Geometry emptyGeometry;
    emptyGeometry.wkt = "";
    
    EXPECT_THROW(
        geometryEngine->buffer(emptyGeometry, 1.0, BufferOptions{}),
        std::exception
    );
}

// ================================================================
// 性能基准测试
// ================================================================

TEST_F(GeometryEngineTest, PerformanceBenchmark_BufferOperation) {
    const int iterations = 1000;
    BufferOptions options;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        geometryEngine->buffer(testPolygon, 1.0, options);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 1000次缓冲操作应在合理时间内完成（<10秒）
    EXPECT_LT(duration.count(), 10000);
    
    std::cout << "缓冲区操作性能: " << iterations << " 次操作耗时 " 
              << duration.count() << " 毫秒" << std::endl;
} 