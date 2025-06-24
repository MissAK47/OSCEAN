# Infrastructure模块简化设计方案

## 📋 **设计目标**

### **核心原则**
1. **最小化复杂度**：从20+文件简化到5个核心文件
2. **消除重复功能**：删除performance模块后，统一性能监控
3. **保持必要功能**：保留线程池、性能监控、服务工厂
4. **避免过度设计**：删除不必要的抽象层和接口

### **功能范围**
- ✅ 统一线程池管理
- ✅ 统一性能监控（唯一实现）
- ✅ 基础服务工厂
- ❌ 删除：过多的接口和抽象层
- ❌ 删除：与其他模块重复的功能

---

## 🏗️ **简化后的模块结构**

### **文件结构（仅5个文件）**
```
common_utilities/
├── include/common_utils/infrastructure/
│   ├── unified_thread_pool_manager.h    # 统一线程池管理器
│   ├── unified_performance_monitor.h    # 统一性能监控器
│   └── common_services_factory.h        # 通用服务工厂
└── src/infrastructure/
    ├── unified_thread_pool_manager.cpp
    ├── unified_performance_monitor.cpp
    └── common_services_factory.cpp
```

### **删除的文件**
- ❌ `thread_pool_interface.h` - 过度抽象，直接实现即可
- ❌ `application_context.h` - 功能可以合并到factory中
- ❌ 所有performance目录文件 - 避免重复实现

---

## 📄 **核心文件设计**

### **1. unified_thread_pool_manager.h（保持现有设计）**

```cpp
#pragma once

#include "../utilities/boost_config.h"
#include <boost/asio/thread_pool.hpp>
#include <memory>
#include <atomic>
#include <vector>

namespace oscean::common_utils::infrastructure {

/**
 * @brief 统一线程池管理器 - 简化实现
 * 
 * 提供全局线程池管理，支持：
 * - 单例模式的全局线程池
 * - 任务提交和执行
 * - 基本的性能统计
 */
class UnifiedThreadPoolManager {
public:
    static UnifiedThreadPoolManager& getInstance();
    
    // 初始化和关闭
    void initialize(size_t threadCount = 0); // 0表示自动检测
    void shutdown();
    bool isInitialized() const { return initialized_; }
    
    // 任务提交
    template<typename Func>
    void submitTask(Func&& func);
    
    template<typename Func>
    auto submitTaskWithResult(Func&& func) 
        -> boost::future<std::invoke_result_t<Func>>;
    
    // 获取线程池（供需要直接访问的模块使用）
    boost::asio::thread_pool& getThreadPool();
    
    // 基本统计
    struct Statistics {
        size_t threadCount = 0;
        std::atomic<size_t> tasksSubmitted{0};
        std::atomic<size_t> tasksCompleted{0};
        std::atomic<size_t> tasksActive{0};
    };
    
    const Statistics& getStatistics() const { return stats_; }

private:
    UnifiedThreadPoolManager() = default;
    ~UnifiedThreadPoolManager();
    
    // 禁用拷贝和移动
    UnifiedThreadPoolManager(const UnifiedThreadPoolManager&) = delete;
    UnifiedThreadPoolManager& operator=(const UnifiedThreadPoolManager&) = delete;
    
    std::unique_ptr<boost::asio::thread_pool> threadPool_;
    std::atomic<bool> initialized_{false};
    Statistics stats_;
    mutable std::mutex mutex_;
};

// 模板实现
template<typename Func>
void UnifiedThreadPoolManager::submitTask(Func&& func) {
    if (!initialized_) {
        throw std::runtime_error("ThreadPoolManager not initialized");
    }
    
    stats_.tasksSubmitted++;
    stats_.tasksActive++;
    
    boost::asio::post(*threadPool_, [this, func = std::forward<Func>(func)]() {
        func();
        stats_.tasksActive--;
        stats_.tasksCompleted++;
    });
}

template<typename Func>
auto UnifiedThreadPoolManager::submitTaskWithResult(Func&& func) 
    -> boost::future<std::invoke_result_t<Func>> {
    
    using ResultType = std::invoke_result_t<Func>;
    auto promise = std::make_shared<boost::promise<ResultType>>();
    auto future = promise->get_future();
    
    submitTask([promise, func = std::forward<Func>(func)]() {
        try {
            if constexpr (std::is_void_v<ResultType>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    
    return future;
}

} // namespace oscean::common_utils::infrastructure
```

### **2. unified_performance_monitor.h（简化版本）**

```cpp
#pragma once

#include <string>
#include <chrono>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace oscean::common_utils::infrastructure {

/**
 * @brief 统一性能监控器 - 简化实现
 * 
 * 提供基本的性能监控功能：
 * - 计时器
 * - 计数器
 * - 内存使用跟踪
 * - 简单的报告生成
 */
class UnifiedPerformanceMonitor {
public:
    static UnifiedPerformanceMonitor& getInstance();
    
    // === 计时器功能 ===
    class Timer {
    public:
        Timer(const std::string& name);
        ~Timer();
        
        void stop();
        double getElapsedSeconds() const;
        
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point startTime_;
        bool stopped_ = false;
    };
    
    // 创建计时器
    std::unique_ptr<Timer> createTimer(const std::string& name);
    
    // === 计数器功能 ===
    void incrementCounter(const std::string& name, int64_t value = 1);
    int64_t getCounter(const std::string& name) const;
    void resetCounter(const std::string& name);
    
    // === 内存跟踪 ===
    void recordMemoryAllocation(const std::string& category, size_t bytes);
    void recordMemoryDeallocation(const std::string& category, size_t bytes);
    size_t getCurrentMemoryUsage(const std::string& category) const;
    
    // === 性能指标 ===
    struct PerformanceMetrics {
        // 计时统计
        struct TimerStats {
            size_t count = 0;
            double totalTime = 0.0;
            double minTime = std::numeric_limits<double>::max();
            double maxTime = 0.0;
            double avgTime = 0.0;
        };
        std::unordered_map<std::string, TimerStats> timers;
        
        // 计数器
        std::unordered_map<std::string, int64_t> counters;
        
        // 内存使用
        std::unordered_map<std::string, size_t> memoryUsage;
        
        // 生成报告
        std::string toString() const;
    };
    
    // 获取性能指标
    PerformanceMetrics getMetrics() const;
    
    // 重置所有统计
    void reset();
    
    // === 便捷宏 ===
    #define PERF_TIMER(name) \
        auto _perf_timer_##__LINE__ = \
            oscean::common_utils::infrastructure::UnifiedPerformanceMonitor::getInstance().createTimer(name)
    
    #define PERF_COUNT(name, value) \
        oscean::common_utils::infrastructure::UnifiedPerformanceMonitor::getInstance().incrementCounter(name, value)
    
    #define PERF_MEMORY_ALLOC(category, bytes) \
        oscean::common_utils::infrastructure::UnifiedPerformanceMonitor::getInstance().recordMemoryAllocation(category, bytes)
    
    #define PERF_MEMORY_FREE(category, bytes) \
        oscean::common_utils::infrastructure::UnifiedPerformanceMonitor::getInstance().recordMemoryDeallocation(category, bytes)

private:
    UnifiedPerformanceMonitor() = default;
    ~UnifiedPerformanceMonitor() = default;
    
    // 禁用拷贝和移动
    UnifiedPerformanceMonitor(const UnifiedPerformanceMonitor&) = delete;
    UnifiedPerformanceMonitor& operator=(const UnifiedPerformanceMonitor&) = delete;
    
    // 内部数据结构
    mutable std::mutex mutex_;
    PerformanceMetrics metrics_;
    
    // 记录计时结果
    void recordTimerResult(const std::string& name, double seconds);
    
    friend class Timer;
};

} // namespace oscean::common_utils::infrastructure
```

### **3. common_services_factory.h（大幅简化）**

```cpp
#pragma once

#include <memory>
#include <string>

namespace oscean::common_utils {
    
// 前向声明
namespace async {
    class UnifiedAsyncContext;
}
namespace memory {
    class UnifiedMemoryManager;
}
namespace cache {
    class CacheFactory;
}
namespace simd {
    class SIMDFactory;
}
namespace parallel {
    class ParallelProcessingFactory;
}
namespace streaming {
    class StreamingFactory;
}

namespace infrastructure {

/**
 * @brief 环境枚举
 */
enum class Environment {
    DEVELOPMENT,
    TESTING,
    PRODUCTION,
    HIGH_PERFORMANCE
};

/**
 * @brief 服务配置
 */
struct ServiceConfig {
    Environment environment = Environment::PRODUCTION;
    size_t maxMemoryMB = 512;
    size_t threadPoolSize = 0; // 0表示自动
    bool enableProfiling = false;
    
    // 便捷工厂方法
    static ServiceConfig forDevelopment();
    static ServiceConfig forTesting();
    static ServiceConfig forProduction();
    static ServiceConfig forHighPerformance();
};

/**
 * @brief 通用服务工厂 - 简化版本
 * 
 * 提供对所有common_utils服务的统一访问点
 */
class CommonServicesFactory {
public:
    // 工厂创建
    static std::unique_ptr<CommonServicesFactory> create(
        const ServiceConfig& config = ServiceConfig::forProduction()
    );
    
    ~CommonServicesFactory();
    
    // === 获取服务实例 ===
    
    // 基础设施服务
    UnifiedThreadPoolManager& getThreadPoolManager() const;
    UnifiedPerformanceMonitor& getPerformanceMonitor() const;
    
    // 其他模块的工厂（延迟创建）
    async::UnifiedAsyncContext& getAsyncContext() const;
    memory::UnifiedMemoryManager& getMemoryManager() const;
    cache::CacheFactory& getCacheFactory() const;
    simd::SIMDFactory& getSIMDFactory() const;
    parallel::ParallelProcessingFactory& getParallelFactory() const;
    streaming::StreamingFactory& getStreamingFactory() const;
    
    // === 便捷服务组合 ===
    
    /**
     * @brief 为数据访问服务创建优化的服务组合
     */
    struct DataAccessServices {
        std::shared_ptr<memory::UnifiedMemoryManager> memoryManager;
        std::shared_ptr<streaming::StreamingFactory> streamingFactory;
        std::shared_ptr<cache::CacheFactory> cacheFactory;
    };
    DataAccessServices createDataAccessServices() const;
    
    /**
     * @brief 为空间操作服务创建优化的服务组合
     */
    struct SpatialOpsServices {
        std::shared_ptr<memory::UnifiedMemoryManager> memoryManager;
        std::shared_ptr<parallel::ParallelProcessingFactory> parallelFactory;
        std::shared_ptr<simd::SIMDFactory> simdFactory;
    };
    SpatialOpsServices createSpatialOpsServices() const;
    
    /**
     * @brief 获取当前配置
     */
    const ServiceConfig& getConfig() const { return config_; }
    
private:
    explicit CommonServicesFactory(const ServiceConfig& config);
    
    ServiceConfig config_;
    
    // 延迟初始化的服务实例
    mutable std::unique_ptr<async::UnifiedAsyncContext> asyncContext_;
    mutable std::unique_ptr<memory::UnifiedMemoryManager> memoryManager_;
    mutable std::unique_ptr<cache::CacheFactory> cacheFactory_;
    mutable std::unique_ptr<simd::SIMDFactory> simdFactory_;
    mutable std::unique_ptr<parallel::ParallelProcessingFactory> parallelFactory_;
    mutable std::unique_ptr<streaming::StreamingFactory> streamingFactory_;
    
    // 初始化基础设施
    void initializeInfrastructure();
};

} // namespace infrastructure
} // namespace oscean::common_utils
```

---

## 🔧 **实现策略**

### **第1步：备份现有代码**
```bash
# 备份现有infrastructure代码
cp -r common_utilities/include/common_utils/infrastructure \
      common_utilities/include/common_utils/infrastructure_backup

cp -r common_utilities/src/infrastructure \
      common_utilities/src/infrastructure_backup
```

### **第2步：简化实现**

#### **保留的文件**
1. `unified_thread_pool_manager.h/cpp` - 直接使用现有实现
2. `unified_performance_monitor.h/cpp` - 简化现有实现
3. `common_services_factory.h/cpp` - 大幅简化

#### **删除的文件**
```bash
# 删除过度设计的文件
rm common_utilities/include/common_utils/infrastructure/thread_pool_interface.h
rm common_utilities/include/common_utils/infrastructure/application_context.h

# 删除performance目录（用户手动删除）
# rm -rf common_utilities/include/common_utils/performance/
# rm -rf common_utilities/src/performance/
```

### **第3步：更新依赖**

#### **统一性能监控命名空间**
```cpp
// 在 common_utils/common_utils.h 中添加
namespace oscean::common_utils {
    // 统一使用infrastructure的性能监控器
    using UnifiedPerformanceMonitor = infrastructure::UnifiedPerformanceMonitor;
    
    // 便捷宏（兼容旧代码）
    #define OSCEAN_PERF_TIMER(name) PERF_TIMER(name)
    #define OSCEAN_PERFORMANCE_TIMER(name) PERF_TIMER(name)
}
```

#### **更新CMakeLists.txt**
```cmake
# common_utilities/CMakeLists.txt
# 删除performance相关的源文件
# 只保留infrastructure的3个源文件
set(INFRASTRUCTURE_SOURCES
    src/infrastructure/unified_thread_pool_manager.cpp
    src/infrastructure/unified_performance_monitor.cpp
    src/infrastructure/common_services_factory.cpp
)
```

---

## 📊 **简化效果对比**

### **代码复杂度降低**
| 指标 | 原设计 | 简化后 | 降低比例 |
|-----|--------|--------|---------|
| **文件数量** | 20+ | 3 | **85%** |
| **代码行数** | 3000+ | 800 | **73%** |
| **接口数量** | 15+ | 3 | **80%** |
| **依赖关系** | 复杂 | 简单 | **90%** |

### **功能保持**
| 功能 | 原设计 | 简化后 | 状态 |
|-----|--------|--------|------|
| **线程池管理** | ✅ | ✅ | 保持 |
| **性能监控** | 3个实现 | 1个实现 | 统一 |
| **服务工厂** | 复杂 | 简化 | 优化 |
| **依赖注入** | ✅ | ✅ | 保持 |

---

## 🎯 **核心价值**

### **1. 消除重复**
- 删除performance模块，统一性能监控实现
- 避免多处重复定义相同功能

### **2. 降低复杂度**
- 从20+文件减少到3个核心文件
- 删除不必要的抽象层和接口

### **3. 保持兼容性**
- 提供命名空间别名，兼容旧代码
- 保持核心功能不变

### **4. 提高可维护性**
- 代码结构清晰简单
- 易于理解和修改

---

## 🚀 **下一步行动**

1. **确认设计方案**
2. **备份现有代码**
3. **实施简化重构**
4. **更新相关文档**
5. **测试兼容性**

这个简化方案保留了infrastructure模块的核心价值，同时大幅降低了复杂度，使代码更加清晰和易于维护。 