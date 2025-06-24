#include "interpolation/layout_converter.h"
#include <common_utils/simd/isimd_manager.h>
#include <common_utils/utilities/logging_utils.h>
#include <algorithm>
#include <cstring>
#include <omp.h>

#ifdef _MSC_VER
#include <immintrin.h>  // AVX2
#endif

namespace oscean::core_services::interpolation {

using namespace common_utils::simd;

/**
 * @brief 高性能内存布局转换器实现
 */
class LayoutConverterImpl : public LayoutConverter {
private:
    // 块大小常量（针对缓存优化）
    static constexpr size_t BLOCK_SIZE = 64;
    static constexpr size_t L1_CACHE_SIZE = 32 * 1024;  // 32KB
    static constexpr size_t L2_CACHE_SIZE = 256 * 1024; // 256KB
    
public:
    void convertLayout(
        const void* src,
        void* dst,
        size_t rows,
        size_t cols,
        DataType dataType,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) override {
        
        if (srcLayout == dstLayout) {
            // 相同布局，直接复制
            size_t totalSize = rows * cols * getDataTypeSize(dataType);
            std::memcpy(dst, src, totalSize);
            return;
        }
        
        // 根据数据类型调用相应的转换函数
        switch (dataType) {
            case DataType::Float32:
                convertLayoutTyped<float>(
                    static_cast<const float*>(src),
                    static_cast<float*>(dst),
                    rows, cols, srcLayout, dstLayout);
                break;
                
            case DataType::Float64:
                convertLayoutTyped<double>(
                    static_cast<const double*>(src),
                    static_cast<double*>(dst),
                    rows, cols, srcLayout, dstLayout);
                break;
                
            case DataType::Int32:
                convertLayoutTyped<int32_t>(
                    static_cast<const int32_t*>(src),
                    static_cast<int32_t*>(dst),
                    rows, cols, srcLayout, dstLayout);
                break;
                
            default:
                throw std::runtime_error("Unsupported data type for layout conversion");
        }
    }
    
    void convertLayoutBatch(
        const std::vector<const void*>& srcBatch,
        std::vector<void*>& dstBatch,
        size_t rows,
        size_t cols,
        DataType dataType,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) override {
        
        if (srcBatch.size() != dstBatch.size()) {
            throw std::invalid_argument("Source and destination batch sizes must match");
        }
        
        // 并行处理批量转换
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(srcBatch.size()); ++i) {
            convertLayout(srcBatch[i], dstBatch[i], rows, cols, 
                         dataType, srcLayout, dstLayout);
        }
    }
    
    std::unique_ptr<LayoutAdapterView<double>> createAdapterView(
        const GridData& grid) override {
        
        return std::make_unique<LayoutAdapterView<double>>(
            static_cast<const double*>(grid.getDataPtr()),
            grid.getDefinition().rows,
            grid.getDefinition().cols,
            determineLayout(grid)
        );
    }
    
    bool shouldConvertForPerformance(
        size_t dataSize,
        size_t numAccesses,
        MemoryLayout currentLayout,
        AccessPattern pattern) const override {
        
        // 基于访问模式和数据大小的启发式决策
        
        // 小数据集不值得转换
        if (dataSize < 1000) {
            return false;
        }
        
        // 计算转换成本
        double conversionCost = dataSize * 0.001; // 假设每个元素1微秒
        
        // 计算访问成本差异
        double currentAccessCost = estimateAccessCost(
            dataSize, numAccesses, currentLayout, pattern);
        
        MemoryLayout optimalLayout = getOptimalLayout(pattern);
        double optimalAccessCost = estimateAccessCost(
            dataSize, numAccesses, optimalLayout, pattern);
        
        // 如果优化后的访问成本加上转换成本仍然更低，则转换
        return (optimalAccessCost + conversionCost) < currentAccessCost * 0.8;
    }
    
    MemoryLayout getOptimalLayout(AccessPattern pattern) const override {
        switch (pattern) {
            case AccessPattern::ROW_MAJOR_SEQUENTIAL:
            case AccessPattern::HORIZONTAL_SLICE:
                return MemoryLayout::ROW_MAJOR;
                
            case AccessPattern::COLUMN_MAJOR_SEQUENTIAL:
            case AccessPattern::VERTICAL_SLICE:
                return MemoryLayout::COLUMN_MAJOR;
                
            case AccessPattern::RANDOM:
            default:
                return MemoryLayout::ROW_MAJOR; // 默认行主序
        }
    }
    
private:
    template<typename T>
    void convertLayoutTyped(
        const T* src,
        T* dst,
        size_t rows,
        size_t cols,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) {
        
        // 选择最优的转换策略
        if (rows * cols < L1_CACHE_SIZE / sizeof(T)) {
            // 小数据集，使用简单转换
            convertLayoutSimple(src, dst, rows, cols, srcLayout, dstLayout);
        } else if (rows * cols < L2_CACHE_SIZE / sizeof(T)) {
            // 中等数据集，使用块转换
            convertLayoutBlocked(src, dst, rows, cols, srcLayout, dstLayout);
        } else {
            // 大数据集，使用并行块转换
            convertLayoutParallelBlocked(src, dst, rows, cols, srcLayout, dstLayout);
        }
    }
    
    template<typename T>
    void convertLayoutSimple(
        const T* src,
        T* dst,
        size_t rows,
        size_t cols,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) {
        
        if (srcLayout == MemoryLayout::ROW_MAJOR && 
            dstLayout == MemoryLayout::COLUMN_MAJOR) {
            // 行主序到列主序
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        } else {
            // 列主序到行主序
            for (size_t j = 0; j < cols; ++j) {
                for (size_t i = 0; i < rows; ++i) {
                    dst[i * cols + j] = src[j * rows + i];
                }
            }
        }
    }
    
    template<typename T>
    void convertLayoutBlocked(
        const T* src,
        T* dst,
        size_t rows,
        size_t cols,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) {
        
        // 使用分块算法优化缓存局部性
        for (size_t i0 = 0; i0 < rows; i0 += BLOCK_SIZE) {
            for (size_t j0 = 0; j0 < cols; j0 += BLOCK_SIZE) {
                // 处理一个块
                size_t iMax = std::min(i0 + BLOCK_SIZE, rows);
                size_t jMax = std::min(j0 + BLOCK_SIZE, cols);
                
                if (srcLayout == MemoryLayout::ROW_MAJOR && 
                    dstLayout == MemoryLayout::COLUMN_MAJOR) {
                    for (size_t i = i0; i < iMax; ++i) {
                        for (size_t j = j0; j < jMax; ++j) {
                            dst[j * rows + i] = src[i * cols + j];
                        }
                    }
                } else {
                    for (size_t j = j0; j < jMax; ++j) {
                        for (size_t i = i0; i < iMax; ++i) {
                            dst[i * cols + j] = src[j * rows + i];
                        }
                    }
                }
            }
        }
    }
    
    template<typename T>
    void convertLayoutParallelBlocked(
        const T* src,
        T* dst,
        size_t rows,
        size_t cols,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) {
        
        // 并行分块转换
        #pragma omp parallel for collapse(2)
        for (size_t i0 = 0; i0 < rows; i0 += BLOCK_SIZE) {
            for (size_t j0 = 0; j0 < cols; j0 += BLOCK_SIZE) {
                // 处理一个块
                size_t iMax = std::min(i0 + BLOCK_SIZE, rows);
                size_t jMax = std::min(j0 + BLOCK_SIZE, cols);
                
                if (srcLayout == MemoryLayout::ROW_MAJOR && 
                    dstLayout == MemoryLayout::COLUMN_MAJOR) {
                    for (size_t i = i0; i < iMax; ++i) {
                        for (size_t j = j0; j < jMax; ++j) {
                            dst[j * rows + i] = src[i * cols + j];
                        }
                    }
                } else {
                    for (size_t j = j0; j < jMax; ++j) {
                        for (size_t i = i0; i < iMax; ++i) {
                            dst[i * cols + j] = src[j * rows + i];
                        }
                    }
                }
            }
        }
    }
    
    // 特化的SIMD优化版本（仅用于float）
    template<>
    void convertLayoutBlocked<float>(
        const float* src,
        float* dst,
        size_t rows,
        size_t cols,
        MemoryLayout srcLayout,
        MemoryLayout dstLayout) {
        
        // 使用AVX2进行8x8块的快速转置
        const size_t SIMD_BLOCK = 8;
        
        // 处理完整的8x8块
        for (size_t i = 0; i < rows - 7; i += SIMD_BLOCK) {
            for (size_t j = 0; j < cols - 7; j += SIMD_BLOCK) {
                if (srcLayout == MemoryLayout::ROW_MAJOR && 
                    dstLayout == MemoryLayout::COLUMN_MAJOR) {
                    transpose8x8AVX2(&src[i * cols + j], &dst[j * rows + i], 
                                    cols, rows);
                } else {
                    transpose8x8AVX2(&src[j * rows + i], &dst[i * cols + j], 
                                    rows, cols);
                }
            }
        }
        
        // 处理剩余的边界
        for (size_t i = (rows / SIMD_BLOCK) * SIMD_BLOCK; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if (srcLayout == MemoryLayout::ROW_MAJOR && 
                    dstLayout == MemoryLayout::COLUMN_MAJOR) {
                    dst[j * rows + i] = src[i * cols + j];
                } else {
                    dst[i * cols + j] = src[j * rows + i];
                }
            }
        }
        
        for (size_t j = (cols / SIMD_BLOCK) * SIMD_BLOCK; j < cols; ++j) {
            for (size_t i = 0; i < (rows / SIMD_BLOCK) * SIMD_BLOCK; ++i) {
                if (srcLayout == MemoryLayout::ROW_MAJOR && 
                    dstLayout == MemoryLayout::COLUMN_MAJOR) {
                    dst[j * rows + i] = src[i * cols + j];
                } else {
                    dst[i * cols + j] = src[j * rows + i];
                }
            }
        }
    }
    
    void transpose8x8AVX2(const float* src, float* dst, 
                         size_t srcStride, size_t dstStride) {
        // 加载8行数据
        __m256 row0 = _mm256_loadu_ps(&src[0 * srcStride]);
        __m256 row1 = _mm256_loadu_ps(&src[1 * srcStride]);
        __m256 row2 = _mm256_loadu_ps(&src[2 * srcStride]);
        __m256 row3 = _mm256_loadu_ps(&src[3 * srcStride]);
        __m256 row4 = _mm256_loadu_ps(&src[4 * srcStride]);
        __m256 row5 = _mm256_loadu_ps(&src[5 * srcStride]);
        __m256 row6 = _mm256_loadu_ps(&src[6 * srcStride]);
        __m256 row7 = _mm256_loadu_ps(&src[7 * srcStride]);
        
        // 8x8转置使用AVX2指令
        __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
        
        // 第一阶段：2x2块转置
        tmp0 = _mm256_unpacklo_ps(row0, row1);
        tmp1 = _mm256_unpackhi_ps(row0, row1);
        tmp2 = _mm256_unpacklo_ps(row2, row3);
        tmp3 = _mm256_unpackhi_ps(row2, row3);
        tmp4 = _mm256_unpacklo_ps(row4, row5);
        tmp5 = _mm256_unpackhi_ps(row4, row5);
        tmp6 = _mm256_unpacklo_ps(row6, row7);
        tmp7 = _mm256_unpackhi_ps(row6, row7);
        
        // 第二阶段：4x4块转置
        row0 = _mm256_shuffle_ps(tmp0, tmp2, 0x44);
        row1 = _mm256_shuffle_ps(tmp0, tmp2, 0xEE);
        row2 = _mm256_shuffle_ps(tmp1, tmp3, 0x44);
        row3 = _mm256_shuffle_ps(tmp1, tmp3, 0xEE);
        row4 = _mm256_shuffle_ps(tmp4, tmp6, 0x44);
        row5 = _mm256_shuffle_ps(tmp4, tmp6, 0xEE);
        row6 = _mm256_shuffle_ps(tmp5, tmp7, 0x44);
        row7 = _mm256_shuffle_ps(tmp5, tmp7, 0xEE);
        
        // 第三阶段：最终排列
        tmp0 = _mm256_permute2f128_ps(row0, row4, 0x20);
        tmp1 = _mm256_permute2f128_ps(row1, row5, 0x20);
        tmp2 = _mm256_permute2f128_ps(row2, row6, 0x20);
        tmp3 = _mm256_permute2f128_ps(row3, row7, 0x20);
        tmp4 = _mm256_permute2f128_ps(row0, row4, 0x31);
        tmp5 = _mm256_permute2f128_ps(row1, row5, 0x31);
        tmp6 = _mm256_permute2f128_ps(row2, row6, 0x31);
        tmp7 = _mm256_permute2f128_ps(row3, row7, 0x31);
        
        // 存储转置后的数据
        _mm256_storeu_ps(&dst[0 * dstStride], tmp0);
        _mm256_storeu_ps(&dst[1 * dstStride], tmp1);
        _mm256_storeu_ps(&dst[2 * dstStride], tmp2);
        _mm256_storeu_ps(&dst[3 * dstStride], tmp3);
        _mm256_storeu_ps(&dst[4 * dstStride], tmp4);
        _mm256_storeu_ps(&dst[5 * dstStride], tmp5);
        _mm256_storeu_ps(&dst[6 * dstStride], tmp6);
        _mm256_storeu_ps(&dst[7 * dstStride], tmp7);
    }
    
    size_t getDataTypeSize(DataType type) const {
        switch (type) {
            case DataType::Float32: return sizeof(float);
            case DataType::Float64: return sizeof(double);
            case DataType::Int32: return sizeof(int32_t);
            case DataType::Int64: return sizeof(int64_t);
            default: return 0;
        }
    }
    
    MemoryLayout determineLayout(const GridData& grid) const {
        // 基于GridData的dimensionOrderInDataLayout判断
        const auto& order = grid.getDefinition().dimensionOrderInDataLayout;
        
        if (order.empty()) {
            return MemoryLayout::ROW_MAJOR; // 默认
        }
        
        // 如果最快变化的维度是垂直/深度，则为列主序
        if (!order.empty() && 
            (order.back() == CoordinateDimension::VERTICAL)) {
            return MemoryLayout::COLUMN_MAJOR;
        }
        
        return MemoryLayout::ROW_MAJOR;
    }
    
    double estimateAccessCost(
        size_t dataSize,
        size_t numAccesses,
        MemoryLayout layout,
        AccessPattern pattern) const {
        
        // 简化的成本模型
        double baseCost = numAccesses * 0.001; // 基础访问成本
        
        // 根据布局和访问模式调整成本
        if ((layout == MemoryLayout::ROW_MAJOR && 
             pattern == AccessPattern::ROW_MAJOR_SEQUENTIAL) ||
            (layout == MemoryLayout::COLUMN_MAJOR && 
             pattern == AccessPattern::COLUMN_MAJOR_SEQUENTIAL)) {
            // 最优情况：顺序访问
            return baseCost;
        } else if (pattern == AccessPattern::RANDOM) {
            // 随机访问，布局影响较小
            return baseCost * 5;
        } else {
            // 不匹配的访问模式
            return baseCost * 10;
        }
    }
};

// 工厂函数
std::unique_ptr<LayoutConverter> LayoutConverter::create() {
    return std::make_unique<LayoutConverterImpl>();
}

} // namespace oscean::core_services::interpolation 