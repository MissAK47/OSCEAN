# 空间缓存正确使用示例

## 架构分层说明

### ❌ 错误架构（缓存层做太多）
```cpp
// ❌ 错误：缓存层实现空间算法
class SpatialCache {
    RTree spatialIndex_;  // ❌ 不应该在缓存层
    std::vector<Point> getInRadius(...) {
        return spatialIndex_.rangeQuery(...);  // ❌ 缓存不做算法
    }
};
```

### ✅ 正确架构（各司其职）

#### 1. 空间服务层（在 core_services_impl/spatial_ops_service/）
```cpp
#include "core_services/spatial_ops/spatial_operations.h"
#include "common_utils/cache/cache_spatial.h"

class SpatialOperationsService {
private:
    // 🎯 空间算法在这里实现
    std::unique_ptr<RTree> spatialIndex_;
    std::unique_ptr<QuadTree> quadTree_;
    
    // 📦 使用缓存优化性能
    std::unique_ptr<SpatialCache<std::vector<OceanDataPoint>>> pointCache_;
    std::unique_ptr<ComputationCache<SpatialQuery, SpatialResult>> queryCache_;

public:
    SpatialOperationsService() {
        // 初始化空间算法结构
        spatialIndex_ = std::make_unique<RTree>();
        quadTree_ = std::make_unique<QuadTree>();
        
        // 初始化缓存
        pointCache_ = std::make_unique<SpatialCache<std::vector<OceanDataPoint>>>(1000, 10.0);
        queryCache_ = std::make_unique<ComputationCache<SpatialQuery, SpatialResult>>(500);
    }
    
    /**
     * @brief 查找半径内的海洋数据点
     * 🎯 空间算法 + 缓存优化的正确结合
     */
    std::vector<OceanDataPoint> findNearbyPoints(double lon, double lat, double radiusKm) {
        // 1️⃣ 先检查缓存
        std::string queryId = "nearby_points";
        if (auto cached = pointCache_->getRadiusQuery(queryId, lon, lat, radiusKm)) {
            return cached.value();
        }
        
        // 2️⃣ 缓存未命中，执行空间算法
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // 🔥 这里是真正的空间算法实现
        std::vector<OceanDataPoint> result;
        spatialIndex_->rangeQuery(lon, lat, radiusKm, result);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto computationTime = std::chrono::duration<double>(endTime - startTime).count();
        
        // 3️⃣ 缓存结果以供后续使用
        pointCache_->putRadiusQuery(queryId, result, lon, lat, radiusKm);
        
        // 4️⃣ 记录计算成本到计算缓存
        SpatialQuery query{lon, lat, radiusKm, "nearby_points"};
        SpatialResult queryResult{result.size(), computationTime};
        std::vector<std::string> dependencies = {"ocean_station_data", "spatial_index"};
        queryCache_->putWithCost(query, queryResult, computationTime, dependencies);
        
        return result;
    }
    
    /**
     * @brief 边界框查询
     */
    std::vector<OceanDataPoint> findInRegion(double minLon, double minLat, 
                                           double maxLon, double maxLat) {
        // 1️⃣ 检查缓存
        std::string queryId = "region_query";
        if (auto cached = pointCache_->getRegionQuery(queryId, minLon, minLat, maxLon, maxLat)) {
            return cached.value();
        }
        
        // 2️⃣ 执行空间算法
        std::vector<OceanDataPoint> result;
        spatialIndex_->boundingBoxQuery(minLon, minLat, maxLon, maxLat, result);
        
        // 3️⃣ 缓存结果
        pointCache_->putRegionQuery(queryId, result, minLon, minLat, maxLon, maxLat);
        
        return result;
    }
    
    /**
     * @brief 数据更新时的缓存失效
     * 🎯 展示如何维护缓存一致性
     */
    void updateOceanData(const std::string& dataSource, 
                        const std::vector<OceanDataPoint>& newData) {
        // 1️⃣ 更新空间索引
        for (const auto& point : newData) {
            spatialIndex_->insert(point);
        }
        
        // 2️⃣ 失效相关的缓存
        for (const auto& point : newData) {
            // 失效该点周围的空间查询缓存
            pointCache_->invalidateAroundPoint(point.longitude, point.latitude, 50.0);
        }
        
        // 3️⃣ 失效依赖该数据源的计算缓存
        queryCache_->invalidateByDataSource(dataSource);
    }

private:
    struct SpatialQuery {
        double lon, lat, radius;
        std::string queryType;
        
        bool operator==(const SpatialQuery& other) const {
            return lon == other.lon && lat == other.lat && 
                   radius == other.radius && queryType == other.queryType;
        }
    };
    
    struct SpatialResult {
        size_t resultCount;
        double computationTime;
    };
    
    struct OceanDataPoint {
        double longitude, latitude, depth;
        std::map<std::string, double> variables;
        std::chrono::system_clock::time_point timestamp;
    };
};
```

#### 2. 缓存层（在 common_utilities/cache/）
```cpp
// ✅ 缓存层专注于缓存策略，不实现空间算法

template<typename Value>
class SpatialCache : public LRUCacheStrategy<std::string, Value> {
public:
    // 🎯 核心职责：为空间查询提供缓存机制
    void putRadiusQuery(const std::string& queryId, const Value& result,
                       double centerLon, double centerLat, double radiusKm);
    
    std::optional<Value> getRadiusQuery(const std::string& queryId,
                                       double centerLon, double centerLat, double radiusKm);
    
    // 🎯 空间感知的失效策略
    size_t invalidateAroundPoint(double lon, double lat, double radiusKm);
    
    // ❌ 不实现空间算法
    // std::vector<Point> rangeQuery(...);  // 这个在SpatialOperationsService中
};
```

## 关键优势分析

### 🎯 **专用缓存的独特价值**

**1. SpatialCache（空间缓存）**
- ✅ **空间感知键生成**：根据经纬度和精度生成一致的缓存键
- ✅ **网格化失效策略**：当某区域数据更新时，智能失效相关查询
- ✅ **空间查询优化**：针对点查询、区域查询、半径查询的专门优化
- ❌ **不实现空间算法**：R-Tree、QuadTree等在空间服务层

**2. ComputationCache（计算缓存）**
- ✅ **计算成本感知**：基于计算时间优化缓存策略
- ✅ **依赖关系管理**：数据源更新时自动失效相关计算
- ✅ **计算统计**：跟踪缓存节省的计算时间
- ❌ **不实现科学计算**：插值、统计算法在相应服务层

### 🔄 **数据流程**

```
用户请求
    ↓
空间服务层 (SpatialOperationsService)
    ↓ 检查缓存
空间缓存 (SpatialCache)
    ↓ 缓存未命中
空间算法 (R-Tree/QuadTree)
    ↓ 计算结果
空间缓存 (存储结果)
    ↓ 返回给用户
用户得到结果
```

### 🏗️ **为什么不能用通用缓存替代？**

**通用LRU缓存的局限性：**
```cpp
// ❌ 通用缓存无法处理空间相关性
LRUCache<string, vector<Point>> generalCache;

// 问题1：缓存键无法表达空间关系
generalCache.put("query1", result1);  // 不知道这是什么位置的查询
generalCache.put("query2", result2);  // 不知道与query1的空间关系

// 问题2：无法进行空间失效
// 当(120.5, 30.2)附近的数据更新时，应该失效哪些查询？
// 通用缓存无法回答这个问题

// 问题3：无法优化空间访问模式
// 无法知道用户经常查询哪些区域，无法进行预取优化
```

**专用空间缓存的优势：**
```cpp
// ✅ 专用缓存理解空间关系
spatialCache.putRadiusQuery("ocean_temp", result, 120.5, 30.2, 10.0);

// ✅ 智能空间失效
spatialCache.invalidateAroundPoint(120.5, 30.2, 5.0);  // 精确失效

// ✅ 空间访问模式优化
spatialCache.optimizeForSpatialWorkload();  // 基于访问模式调整网格
```

## 结论

**✅ 应该实现专用缓存，因为：**

1. **空间缓存**能提供通用缓存无法实现的**空间感知失效策略**
2. **计算缓存**能提供基于**计算成本的智能缓存策略**
3. **架构清晰**：空间算法在空间服务层，缓存策略在缓存层
4. **性能优化**：针对OSCEAN的空间查询模式进行专门优化
5. **可维护性**：职责分离，便于独立优化和测试

**🎯 实施建议：**
- 保持当前的架构设计
- 修复专用缓存的编译问题
- 在空间服务层正确使用这些缓存
- 不要用通用缓存替代专用缓存的功能 