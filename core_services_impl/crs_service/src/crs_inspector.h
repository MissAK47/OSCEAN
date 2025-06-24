#pragma once

#include <string>
#include <memory>
#include <core_services/common_data_types.h>
// #include "proj.h"  // 直接包含PROJ库头文件，在.cpp文件中包含

// 使用void*避免类型冲突
typedef void* ProjObject;
typedef void* ProjContext;

namespace icrs::inspector
{
    class CrsInspector
    {
    public:
        CrsInspector();
        ~CrsInspector();
        
        // 检查CRS是否有效
        bool isValid(const std::string& crsDefinition);
        
        // 获取CRS相关信息
        bool getInfo(const std::string& crsDefinition, oscean::core_services::CRSInfo& crsInfo);
        
        // 检查两个CRS是否兼容（可以进行转换）
        bool isCompatible(const std::string& srcCrs, const std::string& dstCrs);
        
    private:
        bool initialize();
        ProjObject createProjObj(const std::string& crsDefinition);
        void cleanup(ProjContext ctx, ProjObject crs);
        
    private:
        // PROJ上下文
        ProjContext m_Context;
    };

} // namespace icrs::inspector 