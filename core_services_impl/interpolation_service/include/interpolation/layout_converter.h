#pragma once

#include <core_services/common_data_types.h>
#include <memory>
#include <vector>

namespace oscean::core_services::interpolation {

/**
 * @brief 内存布局枚举
 */
enum class MemoryLayout {
    ROW_MAJOR,     // 行主序（C风格）
    COLUMN_MAJOR,  // 列主序（Fortran风格）
    CUSTOM         // 自定义布局
};

/**
 * @brief 访问模式
 */
enum class AccessPattern {
    RANDOM,                    // 随机访问
    ROW_MAJOR_SEQUENTIAL,      // 行顺序访问
    COLUMN_MAJOR_SEQUENTIAL,   // 列顺序访问
    HORIZONTAL_SLICE,          // 水平切片
    VERTICAL_SLICE            // 垂直切片
};

/**
 * @brief 布局适配视图
 */
template<typename T>
class LayoutAdapterView {
private:
    const T* data_;
    size_t rows_;
    size_t cols_;
    MemoryLayout layout_;
    
public:
    LayoutAdapterView(const T* data, size_t rows, size_t cols, MemoryLayout layout)
        : data_(data), rows_(rows), cols_(cols), layout_(layout) {}
    
    /**
     * @brief 获取值（自动处理布局差异）
     */
    T getValue(size_t row, size_t col) const {
        if (layout_ == MemoryLayout::ROW_MAJOR) {
            return data_[row * cols_ + col];
        } else {
            return data_[col * rows_ + row];
        }
    }
    
    /**
     * @brief 获取原始数据指针
     */
    const T* getData() const { return data_; }
    
    /**
     * @brief 获取行数
     */
    size_t getRows() const { return rows_; }
    
    /**
     * @brief 获取列数
     */
    size_t getCols() const { return cols_; }
    
    /**
     * @brief 获取布局
     */
    MemoryLayout getLayout() const { return layout_; }
};

/**
 * @brief 内存布局转换器
 */
class LayoutConverter {
public:
    virtual ~LayoutConverter() = default;
    
    /**
     * @brief 转换内存布局
     * @param src 源数据
     * @param dst 目标数据
     * @param rows 行数
     * @param cols 列数
     * @param dataType 数据类型
     * @param srcLayout 源布局
     * @param dstLayout 目标布局
     */
    virtual void convertLayout(
        const void* src,
        void* dst,
        size_t rows,
        size_t cols,
        DataType dataType,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) = 0;
    
    /**
     * @brief 批量转换内存布局
     */
    virtual void convertLayoutBatch(
        const std::vector<const void*>& srcBatch,
        std::vector<void*>& dstBatch,
        size_t rows,
        size_t cols,
        DataType dataType,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) = 0;
    
    /**
     * @brief 创建布局适配视图
     */
    virtual std::unique_ptr<LayoutAdapterView<double>> createAdapterView(
        const GridData& grid) = 0;
    
    /**
     * @brief 判断是否应该转换布局以获得更好的性能
     */
    virtual bool shouldConvertForPerformance(
        size_t dataSize,
        size_t numAccesses,
        MemoryLayout currentLayout,
        AccessPattern pattern) const = 0;
    
    /**
     * @brief 获取最优布局
     */
    virtual MemoryLayout getOptimalLayout(AccessPattern pattern) const = 0;
    
    /**
     * @brief 创建布局转换器实例
     */
    static std::unique_ptr<LayoutConverter> create();
};

} // namespace oscean::core_services::interpolation 