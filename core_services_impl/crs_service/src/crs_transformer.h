#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <core_services/crs/crs_operation_types.h>
#include <core_services/common_data_types.h>

// 使用void*避免类型冲突
typedef void* ProjObject;
typedef void* ProjContext;

namespace icrs::transformer
{
    class CrsTransformer
    {
    public:
        CrsTransformer();
        ~CrsTransformer();
        
        // 转换点坐标
        bool transform(const oscean::core_services::Point& srcPoint, 
                      const std::string& srcCrs, 
                      const std::string& dstCrs, 
                      oscean::core_services::TransformedPoint& dstPoint);
                      
        // 检测是否已加载指定CRS
        bool hasCrs(const std::string& crsId);
        
    private:
        bool initialize();
        ProjObject createTransformer(const std::string& srcCrs, const std::string& dstCrs);
        std::string getTransformerKey(const std::string& srcCrs, const std::string& dstCrs);
        
    private:
        // transformers缓存容器
        std::unordered_map<std::string, ProjObject> m_Transformers;
        
        // PROJ上下文
        ProjContext m_Context;
    };

} // namespace icrs::transformer 