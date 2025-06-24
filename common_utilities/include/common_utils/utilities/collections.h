/**
 * @file collections.h
 * @brief 容器操作工具库
 */

#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <optional>
#include <random>
#include <numeric>
#include <sstream>

namespace oscean::common_utils {

/**
 * @brief 将哈希值与另一个值的哈希合并，基于Boost的hash_combine实现
 * @tparam T 要合并的值类型
 * @param seed 已有的哈希种子
 * @param v 要组合的值
 */
template <typename T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/**
 * @brief 提供各种容器操作的通用工具函数
 */
class Collections {
public:
    /**
     * @brief 检查容器是否包含指定元素
     * @tparam Container 容器类型
     * @tparam T 值类型
     * @param container 容器
     * @param value 要查找的值
     * @return 如果找到则返回true
     */
    template<typename Container, typename T>
    static bool contains(const Container& container, const T& value) {
        return std::find(std::begin(container), std::end(container), value) != std::end(container);
    }
    
    /**
     * @brief 从映射中获取值或返回默认值
     * @tparam MapType 映射类型
     * @tparam K 键类型
     * @tparam V 值类型
     * @param map 映射
     * @param key 键
     * @param defaultValue 默认值
     * @return 找到的值或默认值
     */
    template<typename MapType, typename K, typename V>
    static V getOrDefault(const MapType& map, const K& key, const V& defaultValue) {
        auto it = map.find(key);
        return (it != map.end()) ? it->second : defaultValue;
    }
    
    /**
     * @brief 从映射中获取值或返回std::nullopt
     * @tparam MapType 映射类型
     * @tparam K 键类型
     * @tparam V 值类型
     * @param map 映射
     * @param key 键
     * @return 包含值的optional或std::nullopt
     */
    template<typename MapType, typename K, typename V = typename MapType::mapped_type>
    static std::optional<V> getOptional(const MapType& map, const K& key) {
        auto it = map.find(key);
        return (it != map.end()) ? std::make_optional(it->second) : std::nullopt;
    }
    
    /**
     * @brief 使用函数转换容器元素
     * @tparam SourceContainer 源容器类型
     * @tparam Mapper 映射函数类型
     * @param source 源容器
     * @param mapper 映射函数
     * @return 转换后的容器
     */
    template<template<typename...> class ResultContainer = std::vector, 
             typename SourceContainer, 
             typename Mapper>
    static auto map(const SourceContainer& source, Mapper mapper) {
        using ResultType = std::invoke_result_t<Mapper, decltype(*std::begin(source))>;
        ResultContainer<ResultType> result;
        
        if constexpr (has_reserve<ResultContainer<ResultType>>::value) {
            result.reserve(source.size());
        }
        
        std::transform(std::begin(source), std::end(source), 
                      std::inserter(result, result.end()), mapper);
        return result;
    }
    
    /**
     * @brief 过滤容器元素
     * @tparam Container 容器类型
     * @tparam Predicate 谓词函数类型
     * @param container 容器
     * @param predicate 谓词函数
     * @return 过滤后的容器
     */
    template<typename Container, typename Predicate>
    static Container filter(const Container& container, Predicate predicate) {
        Container result;
        std::copy_if(std::begin(container), std::end(container), 
                     std::inserter(result, result.end()), predicate);
        return result;
    }
    
    /**
     * @brief 连接容器元素为字符串
     * @tparam Container 容器类型
     * @param container 容器
     * @param delimiter 分隔符
     * @return 连接后的字符串
     */
    template<typename Container>
    static std::string join(const Container& container, const std::string& delimiter) {
        if (container.empty()) {
            return "";
        }
        
        std::ostringstream result;
        auto it = std::begin(container);
        result << *it;
        
        for (++it; it != std::end(container); ++it) {
            result << delimiter << *it;
        }
        
        return result.str();
    }
    
    /**
     * @brief 使用转换函数将容器元素连接为字符串
     * @tparam Container 容器类型
     * @tparam Converter 转换函数类型
     * @param container 容器
     * @param delimiter 分隔符
     * @param converter 转换函数
     * @return 连接后的字符串
     */
    template<typename Container, typename Converter>
    static std::string join(const Container& container, const std::string& delimiter, Converter converter) {
        if (container.empty()) {
            return "";
        }
        
        std::ostringstream result;
        auto it = std::begin(container);
        result << converter(*it);
        
        for (++it; it != std::end(container); ++it) {
            result << delimiter << converter(*it);
        }
        
        return result.str();
    }
    
    /**
     * @brief 对容器元素排序
     * @tparam Container 容器类型
     * @param container 容器
     * @return 排序后的容器
     */
    template<typename Container>
    static Container sorted(const Container& container) {
        Container result = container;
        std::sort(std::begin(result), std::end(result));
        return result;
    }
    
    /**
     * @brief 使用比较函数对容器元素排序
     * @tparam Container 容器类型
     * @tparam Comparator 比较函数类型
     * @param container 容器
     * @param comparator 比较函数
     * @return 排序后的容器
     */
    template<typename Container, typename Comparator>
    static Container sorted(const Container& container, Comparator comparator) {
        Container result = container;
        std::sort(std::begin(result), std::end(result), comparator);
        return result;
    }
    
    /**
     * @brief 查找容器中满足条件的第一个元素
     * @tparam Container 容器类型
     * @tparam Predicate 谓词函数类型
     * @param container 容器
     * @param predicate 谓词函数
     * @return 找到的元素或std::nullopt
     */
    template<typename Container, typename Predicate>
    static auto findFirst(const Container& container, Predicate predicate) {
        auto it = std::find_if(std::begin(container), std::end(container), predicate);
        using value_type = typename std::iterator_traits<decltype(std::begin(container))>::value_type;
        return it != std::end(container) ? std::make_optional(*it) : std::optional<value_type>{};
    }
    
    /**
     * @brief 判断容器中是否任意元素满足条件
     * @tparam Container 容器类型
     * @tparam Predicate 谓词函数类型
     * @param container 容器
     * @param predicate 谓词函数
     * @return 如果有元素满足条件则返回true
     */
    template<typename Container, typename Predicate>
    static bool anyMatch(const Container& container, Predicate predicate) {
        return std::any_of(std::begin(container), std::end(container), predicate);
    }
    
    /**
     * @brief 判断容器中是否所有元素满足条件
     * @tparam Container 容器类型
     * @tparam Predicate 谓词函数类型
     * @param container 容器
     * @param predicate 谓词函数
     * @return 如果所有元素满足条件则返回true
     */
    template<typename Container, typename Predicate>
    static bool allMatch(const Container& container, Predicate predicate) {
        return std::all_of(std::begin(container), std::end(container), predicate);
    }
    
    /**
     * @brief 判断容器中是否没有元素满足条件
     * @tparam Container 容器类型
     * @tparam Predicate 谓词函数类型
     * @param container 容器
     * @param predicate 谓词函数
     * @return 如果没有元素满足条件则返回true
     */
    template<typename Container, typename Predicate>
    static bool noneMatch(const Container& container, Predicate predicate) {
        return std::none_of(std::begin(container), std::end(container), predicate);
    }
    
    /**
     * @brief 统计满足条件的元素数量
     * @tparam Container 容器类型
     * @tparam Predicate 谓词函数类型
     * @param container 容器
     * @param predicate 谓词函数
     * @return 满足条件的元素数量
     */
    template<typename Container, typename Predicate>
    static size_t countIf(const Container& container, Predicate predicate) {
        return std::count_if(std::begin(container), std::end(container), predicate);
    }
    
    /**
     * @brief 对容器元素进行归约
     * @tparam Container 容器类型
     * @tparam T 结果类型
     * @tparam BinaryOp 二元操作函数类型
     * @param container 容器
     * @param init 初始值
     * @param op 二元操作函数
     * @return 归约结果
     */
    template<typename Container, typename T, typename BinaryOp>
    static T reduce(const Container& container, T init, BinaryOp op) {
        return std::accumulate(std::begin(container), std::end(container), init, op);
    }
    
    /**
     * @brief 获取容器的子范围
     * @tparam Container 容器类型
     * @param container 容器
     * @param fromIndex 起始索引
     * @param toIndex 结束索引
     * @return 子范围
     */
    template<typename Container>
    static Container subrange(const Container& container, size_t fromIndex, size_t toIndex) {
        if (fromIndex >= container.size() || fromIndex > toIndex) {
            return Container();
        }
        
        size_t endPos = std::min(toIndex, container.size());
        auto first = std::begin(container);
        std::advance(first, fromIndex);
        
        auto last = std::begin(container);
        std::advance(last, endPos);
        
        return Container(first, last);
    }
    
    /**
     * @brief 根据键函数对容器元素进行分组
     * @tparam Container 容器类型
     * @tparam KeySelector 键选择函数类型
     * @tparam K 键类型
     * @param container 容器
     * @param keySelector 键选择函数
     * @return 分组后的映射
     */
    template<typename Container, typename KeySelector, 
             typename K = std::invoke_result_t<KeySelector, decltype(*std::begin(std::declval<Container>()))>>
    static std::map<K, Container> groupBy(const Container& container, KeySelector keySelector) {
        std::map<K, Container> result;
        
        for (const auto& item : container) {
            K key = keySelector(item);
            result[key].push_back(item);
        }
        
        return result;
    }

private:
    // SFINAE检测容器是否有reserve方法
    template<typename T, typename = void>
    struct has_reserve : std::false_type {};
    
    template<typename T>
    struct has_reserve<T, std::void_t<decltype(std::declval<T>().reserve(0))>> : std::true_type {};
};

} // namespace oscean::common_utils 