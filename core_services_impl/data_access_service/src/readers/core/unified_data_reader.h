#pragma once

/**
 * @file unified_data_reader.h
 * @brief 统一数据读取器抽象基类 - 简化版
 */

#include "common_utils/utilities/boost_config.h"
#include <boost/thread/future.hpp>
#include <string>
#include <memory>
#include <optional>
#include <atomic>
#include "core_services/common_data_types.h"
#include "core_services/data_access/unified_data_types.h"

namespace oscean::core_services::data_access::readers {

/**
 * @brief 统一数据读取器抽象基类
 */
class UnifiedDataReader : public std::enable_shared_from_this<UnifiedDataReader> {
public:
    /**
     * @brief 构造函数
     */
    explicit UnifiedDataReader(const std::string& filePath);
    
    /**
     * @brief 析构函数
     */
    virtual ~UnifiedDataReader() = default;

    // 基本信息
    std::string getFilePath() const noexcept { return filePath_; }
    bool isOpen() const noexcept { return isOpen_; }
    
    // 核心方法
    virtual boost::future<bool> openAsync() = 0;
    virtual boost::future<void> closeAsync() = 0;
    virtual std::string getReaderType() const = 0;
    
    // 元数据方法
    virtual boost::future<std::optional<oscean::core_services::FileMetadata>> getFileMetadataAsync() = 0;
    virtual boost::future<std::vector<std::string>> getVariableNamesAsync() = 0;
    
    // 数据读取方法
    virtual boost::future<std::shared_ptr<oscean::core_services::GridData>> readGridDataAsync(
        const std::string& variableName,
        const std::optional<oscean::core_services::BoundingBox>& bounds = std::nullopt) = 0;

protected:
    void setOpenState(bool isOpen) noexcept { isOpen_ = isOpen; }

private:
    std::string filePath_;
    std::atomic<bool> isOpen_{false};
};

} // namespace oscean::core_services::data_access::readers 