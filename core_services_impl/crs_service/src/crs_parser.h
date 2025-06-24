#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <core_services/common_data_types.h>

// 使用void*避免类型冲突
typedef void* ProjObject;
typedef void* ProjContext;

namespace icrs::parser
{
    class CrsParser
    {
    public:
        CrsParser();
        
        // 返回WKT坐标系统
        bool parseWKT(const std::string& wkt, oscean::core_services::CRSInfo& crsInfo);

        // 返回投影串坐标系统
        bool parseProjString(const std::string& projString, oscean::core_services::CRSInfo& crsInfo);

    private:
        bool createCRSInfo(ProjObject crs, oscean::core_services::CRSInfo& crsInfo);
        ProjObject createProjObj(const std::string& crsDefinition, int crsType);

        void setupProjLib(ProjContext ctx);
        void cleanupProjLib(ProjContext ctx, ProjObject crs);

    private:
        // 这个map是经纬度-投影方向的转换函数Map
        std::unordered_map<std::string, ProjObject> m_Transformers;
    };

} // namespace icrs::parser 