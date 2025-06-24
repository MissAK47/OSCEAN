#include "common_utils/utilities/error_handling.h"
#include <sstream>
#include <iostream>

namespace oscean::common_utils::utilities {

// 这个文件为异常类提供额外的实用功能
// 由于大部分异常类都是在头文件中内联实现的，这里主要提供一些辅助函数

/**
 * @brief 错误处理工具类的实现
 */
class ErrorHandler {
public:
    /**
     * @brief 格式化异常信息
     */
    static std::string formatException(const std::exception& e, const std::string& context = "") {
        std::ostringstream oss;
        if (!context.empty()) {
            oss << "[" << context << "] ";
        }
        oss << "Exception: " << e.what();
        return oss.str();
    }
    
    /**
     * @brief 记录异常信息
     */
    static void logException(const std::exception& e, const std::string& context = "") {
        // 这里可以集成日志系统
        // 暂时使用标准输出
        std::cerr << formatException(e, context) << std::endl;
    }
    
    /**
     * @brief 安全执行函数，捕获所有异常
     */
    template<typename Func>
    static bool safeExecute(Func&& func, const std::string& context = "") {
        try {
            func();
            return true;
        } catch (const std::exception& e) {
            logException(e, context);
            return false;
        } catch (...) {
            std::cerr << "[" << context << "] Unknown exception caught" << std::endl;
            return false;
        }
    }
};

} // namespace oscean::common_utils::utilities 