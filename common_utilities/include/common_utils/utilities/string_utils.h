/**
 * @file string_utils.h
 * @brief 字符串处理工具函数
 */

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <locale>
#include <regex>
#include <functional>

namespace oscean::common_utils {

/**
 * @brief 字符串工具类
 */
class StringUtils {
public:
    /**
     * @brief 去除字符串左侧空白字符
     * @param s 输入字符串
     * @return 处理后的字符串
     */
    static std::string trimLeft(const std::string& s);
    
    /**
     * @brief 去除字符串右侧空白字符
     * @param s 输入字符串
     * @return 处理后的字符串
     */
    static std::string trimRight(const std::string& s);
    
    /**
     * @brief 去除字符串两侧空白字符
     * @param s 输入字符串
     * @return 处理后的字符串
     */
    static std::string trim(const std::string& s);
    
    /**
     * @brief 转换字符串为小写
     * @param s 输入字符串
     * @return 小写字符串
     */
    static std::string toLower(const std::string& s);
    
    /**
     * @brief 转换字符串为大写
     * @param s 输入字符串
     * @return 大写字符串
     */
    static std::string toUpper(const std::string& s);
    
    /**
     * @brief 按分隔符分割字符串
     * @param s 输入字符串
     * @param delimiter 分隔符
     * @param trimTokens 是否去除每个部分的空白
     * @return 分割后的字符串向量
     */
    static std::vector<std::string> split(const std::string& s, 
                                         const std::string& delimiter,
                                         bool trimTokens = true);
    
    /**
     * @brief 连接字符串向量
     * @param v 字符串向量
     * @param delimiter 分隔符
     * @return 连接后的字符串
     */
    static std::string join(const std::vector<std::string>& v, 
                           const std::string& delimiter);
    
    /**
     * @brief 检查字符串是否以指定前缀开始
     * @param s 输入字符串
     * @param prefix 前缀
     * @param caseSensitive 是否区分大小写
     * @return 是否以前缀开始
     */
    static bool startsWith(const std::string& s, 
                          const std::string& prefix,
                          bool caseSensitive = true);
    
    /**
     * @brief 检查字符串是否以指定后缀结束
     * @param s 输入字符串
     * @param suffix 后缀
     * @param caseSensitive 是否区分大小写
     * @return 是否以后缀结束
     */
    static bool endsWith(const std::string& s, 
                        const std::string& suffix,
                        bool caseSensitive = true);
    
    /**
     * @brief 替换字符串中所有指定子串
     * @param s 输入字符串
     * @param from 要替换的子串
     * @param to 替换为的子串
     * @return 替换后的字符串
     */
    static std::string replace(const std::string& s, 
                              const std::string& from, 
                              const std::string& to);
    
    /**
     * @brief 替换字符串中首个指定子串
     * @param s 输入字符串
     * @param from 要替换的子串
     * @param to 替换为的子串
     * @return 替换后的字符串
     */
    static std::string replaceFirst(const std::string& s, 
                                   const std::string& from, 
                                   const std::string& to);
    
    /**
     * @brief 检查字符串是否包含子串
     * @param s 输入字符串
     * @param substring 子串
     * @param caseSensitive 是否区分大小写
     * @return 是否包含子串
     */
    static bool contains(const std::string& s, 
                        const std::string& substring,
                        bool caseSensitive = true);
    
    /**
     * @brief 使用正则表达式查找所有匹配项
     * @param s 输入字符串
     * @param regex 正则表达式
     * @return 匹配的子串向量
     */
    static std::vector<std::string> findAll(const std::string& s, 
                                           const std::string& regex);
    
    /**
     * @brief 使用正则表达式查找第一个匹配项
     * @param s 输入字符串
     * @param regex 正则表达式
     * @param result 匹配结果
     * @return 是否找到匹配
     */
    static bool findFirst(const std::string& s, 
                         const std::string& regex, 
                         std::string& result);
    
    /**
     * @brief 使用正则表达式替换所有匹配项
     * @param s 输入字符串
     * @param regex 正则表达式
     * @param replacement 替换字符串
     * @return 替换后的字符串
     */
    static std::string regexReplace(const std::string& s, 
                                   const std::string& regex, 
                                   const std::string& replacement);
    
    /**
     * @brief 检查字符串是否匹配正则表达式
     * @param s 输入字符串
     * @param regex 正则表达式
     * @return 是否匹配
     */
    static bool regexMatch(const std::string& s, const std::string& regex);
    
    /**
     * @brief 格式化字符串 (类似于printf)
     * @tparam Args 参数类型包
     * @param format 格式字符串
     * @param args 参数
     * @return 格式化后的字符串
     */
    template<typename... Args>
    static std::string format(const std::string& format, Args&&... args);
    
    /**
     * @brief 将字符串转换为其他类型
     * @tparam T 目标类型
     * @param s 输入字符串
     * @param defaultValue 转换失败时的默认值
     * @return 转换后的值
     */
    template<typename T>
    static T convert(const std::string& s, const T& defaultValue = T());
    
    /**
     * @brief 将各种类型转换为字符串
     * @tparam T 源类型
     * @param value 输入值
     * @return 转换后的字符串
     */
    template<typename T>
    static std::string toString(const T& value);
    
    /**
     * @brief 编码字符串为URL编码
     * @param s 输入字符串
     * @return URL编码后的字符串
     */
    static std::string urlEncode(const std::string& s);
    
    /**
     * @brief 解码URL编码的字符串
     * @param s URL编码的字符串
     * @return 解码后的字符串
     */
    static std::string urlDecode(const std::string& s);
    
    /**
     * @brief Base64编码
     * @param s 输入字符串
     * @return Base64编码后的字符串
     */
    static std::string base64Encode(const std::string& s);
    
    /**
     * @brief Base64解码
     * @param s Base64编码的字符串
     * @return 解码后的字符串
     */
    static std::string base64Decode(const std::string& s);
    
    /**
     * @brief 填充字符串至指定长度
     * @param s 输入字符串
     * @param length 目标长度
     * @param fillChar 填充字符
     * @param fillLeft 是否从左侧填充
     * @return 填充后的字符串
     */
    static std::string pad(const std::string& s, 
                          size_t length, 
                          char fillChar = ' ', 
                          bool fillLeft = false);
    
    /**
     * @brief 从左侧填充字符串至指定长度
     * @param s 输入字符串
     * @param length 目标长度
     * @param fillChar 填充字符
     * @return 填充后的字符串
     */
    static std::string padLeft(const std::string& s,
                              size_t length,
                              char fillChar = ' ');
    
    /**
     * @brief 截取字符串的一部分
     * @param s 输入字符串
     * @param start 起始位置
     * @param length 长度，默认为直到字符串结束
     * @return 截取后的字符串
     */
    static std::string substring(const std::string& s, 
                                size_t start, 
                                size_t length = std::string::npos);
};

// 模板函数实现
template<typename... Args>
std::string StringUtils::format(const std::string& format, Args&&... args) {
    // 计算所需缓冲区大小
    int size_s = std::snprintf(nullptr, 0, format.c_str(), std::forward<Args>(args)...) + 1;
    if (size_s <= 0) { 
        return ""; 
    }
    
    // 分配缓冲区
    auto size = static_cast<size_t>(size_s);
    auto buf = std::make_unique<char[]>(size);
    
    // 格式化字符串
    std::snprintf(buf.get(), size, format.c_str(), std::forward<Args>(args)...);
    
    // 返回格式化后的字符串，不包含结尾的空字符
    return std::string(buf.get(), buf.get() + size - 1);
}

template<typename T>
T StringUtils::convert(const std::string& s, const T& defaultValue) {
    T result = defaultValue;
    std::istringstream iss(s);
    iss >> result;
    return iss.fail() ? defaultValue : result;
}

template<typename T>
std::string StringUtils::toString(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

// 特化常见类型
template<>
inline bool StringUtils::convert<bool>(const std::string& s, const bool& defaultValue) {
    std::string lower = toLower(trim(s));
    if (lower == "true" || lower == "yes" || lower == "1" || lower == "y" || lower == "t") {
        return true;
    }
    if (lower == "false" || lower == "no" || lower == "0" || lower == "n" || lower == "f") {
        return false;
    }
    return defaultValue;
}

template<>
inline int StringUtils::convert<int>(const std::string& s, const int& defaultValue) {
    try {
        return std::stoi(s);
    } catch (...) {
        return defaultValue;
    }
}

template<>
inline double StringUtils::convert<double>(const std::string& s, const double& defaultValue) {
    try {
        return std::stod(s);
    } catch (...) {
        return defaultValue;
    }
}

// 特化 std::vector<std::string> 的 toString 方法
template<>
inline std::string StringUtils::toString(const std::vector<std::string>& value) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        oss << value[i];
        if (i < value.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

// 特化 std::vector<size_t> 的 toString 方法
template<>
inline std::string StringUtils::toString(const std::vector<size_t>& value) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        oss << value[i];
        if (i < value.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

} // namespace oscean::common_utils 