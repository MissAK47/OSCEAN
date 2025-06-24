#pragma once

#include "core_services/interpolation/i_interpolation_service.h"
#include "core_services/common_data_types.h"
#include "common_utils/simd/isimd_manager.h"
#include <boost/smart_ptr/shared_ptr.hpp>
#include <memory>
#include <vector>
#include <functional>

namespace oscean::core_services::interpolation {

/**
 * @brief 内存布局感知的插值器基类
 * @details 提供对不同内存布局（行主序/列主序）的透明访问支持
 */
class LayoutAwareInterpolatorBase {
public:
    /**
     * @brief 内存布局感知的数据访问器
     */
    class LayoutAwareAccessor {
    private:
        const GridData& grid_;
        GridData::MemoryLayout layout_;
        size_t dims_[4];  // 最多支持4维
        size_t strides_[4];  // 各维度步长
        size_t numDims_;
        
        friend class LayoutAwareInterpolatorBase;
        
    public:
        explicit LayoutAwareAccessor(const GridData& grid);
        
        /**
         * @brief 获取数据值（自动处理布局差异）
         * @param x X坐标索引（逻辑坐标）
         * @param y Y坐标索引（逻辑坐标）
         * @param z Z坐标索引（逻辑坐标）
         * @param band 波段索引
         * @return 数据值
         */
        template<typename T>
        T getValue(size_t x, size_t y, size_t z = 0, size_t band = 0) const {
            size_t physicalIndex = 0;
            
            if (layout_ == GridData::MemoryLayout::ROW_MAJOR) {
                // 行主序：最后一个维度变化最快
                // 标准顺序：band, z, y, x
                physicalIndex = ((band * dims_[2] + z) * dims_[1] + y) * dims_[0] + x;
            } else if (layout_ == GridData::MemoryLayout::COLUMN_MAJOR) {
                // 列主序：第一个维度变化最快
                // Fortran顺序：x, y, z, band
                physicalIndex = x + dims_[0] * (y + dims_[1] * (z + dims_[2] * band));
            } else {
                // 自定义布局：使用步长计算
                physicalIndex = x * strides_[0] + y * strides_[1] + 
                               z * strides_[2] + band * strides_[3];
            }
            
            // 使用GridData的标准访问方法
            // 需要将线性索引转换回多维索引
            size_t row = physicalIndex / dims_[0];
            size_t col = physicalIndex % dims_[0];
            return grid_.getValue<T>(row, col, band);
        }
        
        /**
         * @brief 批量获取数据块（优化缓存访问）
         */
        template<typename T>
        void getBlock(size_t x0, size_t y0, size_t width, size_t height, 
                     T* output, size_t z = 0, size_t band = 0) const {
            if (shouldTranspose()) {
                getBlockTransposed(x0, y0, width, height, output, z, band);
            } else {
                getBlockDirect(x0, y0, width, height, output, z, band);
            }
        }
        
        /**
         * @brief 获取一维切片（用于PCHIP等算法）
         */
        template<typename T>
        void getSlice(size_t dim, const std::vector<size_t>& fixedIndices, 
                     T* output, size_t length) const;
        
        /**
         * @brief 判断是否需要转置访问
         */
        bool shouldTranspose() const {
            return layout_ == GridData::MemoryLayout::COLUMN_MAJOR;
        }
        
        GridData::MemoryLayout getLayout() const { return layout_; }
        
    private:
        void calculateStrides();
        
        template<typename T>
        void getBlockDirect(size_t x0, size_t y0, size_t width, size_t height,
                           T* output, size_t z, size_t band) const;
        
        template<typename T>
        void getBlockTransposed(size_t x0, size_t y0, size_t width, size_t height,
                               T* output, size_t z, size_t band) const;
    };
    
    /**
     * @brief SIMD优化的数据访问
     */
    class SIMDOptimizedAccessor : public LayoutAwareAccessor {
    private:
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;
        
    public:
        SIMDOptimizedAccessor(const GridData& grid, 
                             boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager);
        
        /**
         * @brief SIMD向量化的块访问
         */
        void getBlockSIMD(size_t x0, size_t y0, size_t width, size_t height,
                         float* output, size_t z = 0, size_t band = 0) const;
        
        /**
         * @brief SIMD向量化的插值计算
         */
        void interpolateBlockSIMD(const float* srcBlock, size_t srcWidth, size_t srcHeight,
                                 float* dstBlock, size_t dstWidth, size_t dstHeight,
                                 InterpolationMethod method) const;
    };
    
protected:
    boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager_;
    mutable std::unique_ptr<LayoutAwareAccessor> accessor_;
    mutable std::unique_ptr<SIMDOptimizedAccessor> simdAccessor_;
    
    /**
     * @brief 获取布局感知的访问器
     */
    const LayoutAwareAccessor& getAccessor(const GridData& grid) const {
        if (!accessor_ || &accessor_->grid_ != &grid) {
            accessor_ = std::make_unique<LayoutAwareAccessor>(grid);
        }
        return *accessor_;
    }
    
    /**
     * @brief 获取SIMD优化的访问器
     */
    const SIMDOptimizedAccessor& getSIMDAccessor(const GridData& grid) const {
        if (!simdAccessor_ || &simdAccessor_->grid_ != &grid) {
            simdAccessor_ = std::make_unique<SIMDOptimizedAccessor>(grid, simdManager_);
        }
        return *simdAccessor_;
    }
    
    /**
     * @brief 判断是否应该使用SIMD优化
     */
    bool shouldUseSIMD(size_t dataSize) const {
        return simdManager_ && dataSize >= 16;
    }
    
public:
    explicit LayoutAwareInterpolatorBase(
        boost::shared_ptr<oscean::common_utils::simd::ISIMDManager> simdManager = nullptr)
        : simdManager_(simdManager) {}
    
    virtual ~LayoutAwareInterpolatorBase() = default;
};

} // namespace oscean::core_services::interpolation 