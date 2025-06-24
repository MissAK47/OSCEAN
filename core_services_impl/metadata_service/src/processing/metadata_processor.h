#ifndef METADATA_PROCESSOR_H
#define METADATA_PROCESSOR_H

#include "core_services/common_data_types.h"
#include <memory>

namespace oscean::core_services::metadata {

/**
 * @class MetadataProcessor
 * @brief 负责解析和丰富FileMetadata对象的工具类
 * 
 * 此类包含了从原始元数据（如原始坐标数组、CF时间单位等）
 * 计算和生成结构化元数据（如空间范围、时间范围、CRS等）的所有业务逻辑。
 * 遵循"处理器"的设计原则。
 */
class MetadataProcessor {
public:
    /**
     * @brief 对FileMetadata对象进行完整的处理和丰富
     * 
     * @param metadata 一个可能只包含原始信息的FileMetadata对象的引用。
     *                 此对象将被就地修改和丰富。
     * @return 如果处理成功则返回true，否则返回false。
     */
    static bool processAndEnrich(oscean::core_services::FileMetadata& metadata);

private:
    /**
     * @brief 解析空间相关的元数据
     * @param metadata FileMetadata对象的引用
     */
    static void processSpatialInfo(oscean::core_services::FileMetadata& metadata);

    /**
     * @brief 解析时间相关的元数据
     * @param metadata FileMetadata对象的引用
     */
    static void processTemporalInfo(oscean::core_services::FileMetadata& metadata);

    /**
     * @brief 解析坐标参考系统（CRS）信息
     * @param metadata FileMetadata对象的引用
     */
    static void processCrsInfo(oscean::core_services::FileMetadata& metadata);

    /**
     * @brief 将CF时间（数值、单位、日历）转换为ISO 8601字符串范围
     * @param values CF时间数值数组
     * @param units CF时间单位
     * @param calendar CF日历
     * @return 一个包含开始和结束ISO字符串的pair，如果失败则为空
     */
    static std::optional<std::pair<std::string, std::string>> convertCFTimeToISO(
        const std::vector<double>& values,
        const std::string& units,
        const std::string& calendar
    );

    /**
     * @brief 根据时间值和单位计算时间分辨率
     * @param values 时间值数组
     * @param units 时间单位
     * @return 表示时间分辨率的字符串（例如 "1 day"），如果失败则为空
     */
    static std::optional<std::string> calculateTimeResolution(
        const std::vector<double>& values,
        const std::string& units
    );
};

} // namespace oscean::core_services::metadata

#endif // METADATA_PROCESSOR_H 