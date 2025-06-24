#pragma once

#include "core_services/common_data_types.h"
#include "core_services/exceptions.h"
#include "common_utils/utilities/boost_config.h"
OSCEAN_NO_BOOST_ASIO_MODULE();  // 插值服务不使用boost::asio，只使用boost::future

#include <boost/thread/future.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace oscean {
namespace core_services {
namespace interpolation {

// 使用 common_data_types.h 中已有的 GridData, CRSInfo, DataType, DimensionCoordinateInfo 等

/**
 * @enum InterpolationMethod
 * @brief 定义可用的插值算法。
 */
enum class InterpolationMethod {
    UNKNOWN,
    LINEAR_1D,                // 1D 线性插值
    CUBIC_SPLINE_1D,          // 1D 立方样条插值
    NEAREST_NEIGHBOR,         // N-D 最近邻插值
    BILINEAR,                 // 2D 双线性插值 (通常用于规则网格)
    BICUBIC,                  // 2D 双三次插值 (通常用于规则网格)
    TRILINEAR,                // 3D 三线性插值 (通常用于规则网格)
    TRICUBIC,                 // 3D 三次插值 (通常用于规则网格)
    PCHIP_RECURSIVE_NDIM,     // N-D 分段三次 Hermite 插值 (PCHIP), 递归实现
    PCHIP_MULTIGRID_NDIM,     // N-D PCHIP, 基于预计算和多重网格思想
    PCHIP_OPTIMIZED_2D_BATHY, // 针对2D水深优化PCHIP
    PCHIP_OPTIMIZED_3D_SVP,   // 针对3D声速剖面优化PCHIP (例如，深度PCHIP + 水平双线性)
    PCHIP_FAST_2D,            // 2D PCHIP (高性能预计算版)
    PCHIP_FAST_3D,            // 3D PCHIP (高性能预计算版)
    // 复数场插值方法
    COMPLEX_FIELD_BILINEAR,   // 复数场双线性插值（用于RAM声场数据）
    COMPLEX_FIELD_BICUBIC,    // 复数场双三次插值
    COMPLEX_FIELD_TRILINEAR,  // 复数场三线性插值
    COMPLEX_FIELD_PCHIP       // 复数场PCHIP插值
    // 后续可添加: IDW, KRIGING_ORDINARY, RBF_THIN_PLATE_SPLINE
};

/**
 * @struct TargetPoint
 * @brief 定义插值目标的单个N维点。
 */
struct TargetPoint {
    std::vector<double> coordinates; ///< N维坐标值。
                                     ///< 其顺序必须与源GridData的维度顺序 (例如 sourceGrid.getDefinition().dimensionOrderInDataLayout) 一致。
    std::optional<CRSInfo> crs;      ///< 可选：目标点的CRS。如果与源GridData的CRS不同，则可能需要转换。
                                     ///< 通常，目标点坐标应预先转换为源GridData的CRS。
};

/**
 * @struct TargetGridDefinition
 * @brief 定义插值目标网格的结构。
 */
struct TargetGridDefinition {
    std::string gridName = "interpolated_grid"; ///< 输出网格的名称。
    std::vector<DimensionCoordinateInfo> dimensions; ///< 定义目标网格的各个维度轴。
                                                     ///< 每个 DimensionCoordinateInfo 来自 common_data_types.h。
    CRSInfo crs;                                     ///< 目标网格的CRS。源数据将根据需要重投影到此CRS。
    DataType outputDataType = DataType::Float32;     ///< 期望输出GridData的数据类型。
    std::optional<double> fillValue;                 ///< 目标网格中无法插值时代的填充值。
                                                     ///< (源GridData的_fillValue可以作为默认参考)
    // 输出网格的 dimensionOrderInDataLayout 可由 'dimensions' 中的顺序隐式定义，
    // 或在此处显式添加该成员如果需要更精细控制。
};

/**
 * @brief 算法特定参数的变体类型。
 * PCHIP本身通常不需要参数，但优化版本可能需要配置，或IDW/Kriging等算法需要参数。
 */
using AlgorithmParameters = std::variant<
    std::monostate // 默认空状态
    // , IdwParams, KrigingParams 等具体参数结构体可以后续添加
>;

/**
 * @struct InterpolationRequest
 * @brief 封装插值操作所需的所有信息。
 */
struct InterpolationRequest {
    boost::shared_ptr<GridData> sourceGrid;        ///< 源数据，来自 common_data_types.h
    std::variant<std::vector<TargetPoint>, TargetGridDefinition> target; ///< 插值目标
    InterpolationMethod method = InterpolationMethod::UNKNOWN;           ///< 插值方法
    AlgorithmParameters algorithmParams;         ///< 算法特定参数
    DataType desiredOutputValueType = DataType::Float32; ///< 用于点插值时，输出值的期望类型。
};

/**
 * @brief 插值结果数据。
 */
using InterpolationResultData = std::variant<
    std::monostate,                           // 空结果
    GridData,                                 // 用于 interpolateToGrid 的结果 (来自 common_data_types.h)
    std::vector<std::optional<double>>        // 用于 interpolateAtPoints 的结果 (double类型，可选表示插值失败的点)
                                              // 如果 desiredOutputValueType 更灵活，可以是 std::vector<std::optional<std::any>>
>;

/**
 * @struct InterpolationResult
 * @brief 表示插值操作的结果。
 */
struct InterpolationResult {
    InterpolationResultData data;
    // common_utils::ErrorCode status = common_utils::ErrorCode::Success; // 使用项目统一错误码
    int statusCode = 0; // 0 表示成功，其他表示错误 (临时，应替换为统一错误码)
    std::string message; // 错误或状态消息
};

/**
 * @class IInterpolationService
 * @brief 插值服务的核心接口。
 */
class IInterpolationService {
public:
    virtual ~IInterpolationService() = default;

    /**
     * @brief 异步执行插值操作
     * 
     * @param request 插值请求对象，包含输入数据、目标点、算法选择等信息。
     * @return 一个 boost::future，最终将包含 InterpolationResult。
     */
    virtual boost::future<InterpolationResult> interpolateAsync(const InterpolationRequest& request) = 0;

    /**
     * @brief 获取此服务支持的插值方法列表。
     * @return 支持的 InterpolationMethod 枚举向量。
     */
    virtual std::vector<InterpolationMethod> getSupportedMethods() const = 0;
};

} // namespace interpolation
} // namespace core_services
} // namespace oscean 