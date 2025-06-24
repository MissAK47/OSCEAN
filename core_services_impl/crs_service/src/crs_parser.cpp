#include "crs_parser.h"
#include <stdexcept>
#include <string>
#include <mutex>
#include <proj.h>
// GDAL头文件移除 - CRS解析器只使用PROJ库
#include <spdlog/spdlog.h>
#include "common_utils/utilities/logging_utils.h"

// 添加可能缺失的宏
#ifndef PJ_WKT
#define PJ_WKT 1
#endif

#ifndef PJ_PROJ_STRING 
#define PJ_PROJ_STRING 2
#endif

namespace icrs::parser
{
    CrsParser::CrsParser()
    {
        // CRS解析器只处理坐标系统定义解析，不负责GDAL初始化
        // GDAL初始化由数据访问服务负责
    }

    bool CrsParser::parseWKT(const std::string& wkt, oscean::core_services::CRSInfo& crsInfo)
    {
        if (wkt.empty())
        {
            return false;
        }
        
        ProjContext ctx = (ProjContext)proj_context_create();
        if (!ctx)
        {
            return false;
        }
        
        // 设置PROJ_LIB环境变量确保数据路径正确
        #ifdef _WIN32
        _putenv_s("PROJ_LIB", "C:\\Users\\Administrator\\vcpkg\\installed\\x64-windows\\share\\proj");
        #else
        setenv("PROJ_LIB", "C:/Users/Administrator/vcpkg/installed/x64-windows/share/proj", 1);
        #endif
        
        ProjObject crs = createProjObj(wkt, PJ_WKT);
        if (!crs)
        {
            proj_context_destroy((PJ_CONTEXT*)ctx);
            return false;
        }
        
        bool result = createCRSInfo(crs, crsInfo);
        
        cleanupProjLib(ctx, crs);
        return result;
    }
    
    bool CrsParser::parseProjString(const std::string& projString, oscean::core_services::CRSInfo& crsInfo)
    {
        if (projString.empty())
        {
            return false;
        }
        
        ProjContext ctx = (ProjContext)proj_context_create();
        if (!ctx)
        {
            return false;
        }
        
        // 设置PROJ_LIB环境变量确保数据路径正确
        #ifdef _WIN32
        _putenv_s("PROJ_LIB", "C:\\Users\\Administrator\\vcpkg\\installed\\x64-windows\\share\\proj");
        #else
        setenv("PROJ_LIB", "C:/Users/Administrator/vcpkg/installed/x64-windows/share/proj", 1);
        #endif
        
        ProjObject crs = createProjObj(projString, PJ_PROJ_STRING);
        if (!crs)
        {
            proj_context_destroy((PJ_CONTEXT*)ctx);
            return false;
        }
        
        bool result = createCRSInfo(crs, crsInfo);
        
        cleanupProjLib(ctx, crs);
        return result;
    }
    
    ProjObject CrsParser::createProjObj(const std::string& crsDefinition, int crsType)
    {
        ProjContext ctx = (ProjContext)proj_context_create();
        if (!ctx)
        {
            return nullptr;
        }
        
        setupProjLib(ctx);
        
        ProjObject crs = nullptr;
        if (crsType == PJ_WKT)
        {
            crs = (ProjObject)proj_create((PJ_CONTEXT*)ctx, crsDefinition.c_str());
        }
        else if (crsType == PJ_PROJ_STRING)
        {
            crs = (ProjObject)proj_create((PJ_CONTEXT*)ctx, crsDefinition.c_str());
        }
        
        if (!crs)
        {
            proj_context_destroy((PJ_CONTEXT*)ctx);
            return nullptr;
        }
        
        return crs;
    }
    
    void CrsParser::setupProjLib([[maybe_unused]] ProjContext ctx)
    {
        // 设置PROJ库相关配置
        // 如有必要可添加具体实现
    }
    
    void CrsParser::cleanupProjLib(ProjContext ctx, ProjObject crs)
    {
        if (crs)
        {
            proj_destroy((PJ*)crs);
        }
        
        if (ctx)
        {
            proj_context_destroy((PJ_CONTEXT*)ctx);
        }
    }
    
    bool CrsParser::createCRSInfo(ProjObject crs, oscean::core_services::CRSInfo& crsInfo)
    {
        if (!crs)
        {
            return false;
        }
        
        // 获取WKT
        const char* wkt = proj_as_wkt(NULL, (PJ*)crs, PJ_WKT2_2019, NULL);
        if (wkt)
        {
            crsInfo.wkt = wkt;
            crsInfo.wktext = wkt; // 兼容性字段
        }
        
        // 获取PROJ字符串
        const char* proj_str = proj_as_proj_string(NULL, (PJ*)crs, PJ_PROJ_5, NULL);
        if (proj_str)
        {
            crsInfo.projString = proj_str;
            crsInfo.proj4text = proj_str; // 兼容性字段
        }
        
        // 获取EPSG代码
        const char* code = proj_get_id_code((PJ*)crs, 0);
        if (code)
        {
            try {
                crsInfo.epsgCode = std::stoi(code);
                // 设置兼容性字段
                crsInfo.code = code;
                const char* auth = proj_get_id_auth_name((PJ*)crs, 0);
                if (auth) {
                    crsInfo.authority = auth;
                    crsInfo.authorityCode = code;
                    crsInfo.authorityName = auth;
                    // 设置ID
                    crsInfo.id = std::string(auth) + ":" + code;
                }
            } catch (const std::exception&) {
                crsInfo.epsgCode = -1;
            }
        }
        else
        {
            crsInfo.epsgCode = -1;
        }
        
        // 获取名称
        const char* name = proj_get_name((PJ*)crs);
        if (name)
        {
            crsInfo.name = name;
        }
        
        // 获取类型
        PJ_TYPE type = proj_get_type((PJ*)crs);
        switch (type) {
            case PJ_TYPE_GEOGRAPHIC_2D_CRS:
                crsInfo.type = "Geographic 2D CRS";
                crsInfo.isGeographic = true;
                break;
            case PJ_TYPE_GEOGRAPHIC_3D_CRS:
                crsInfo.type = "Geographic 3D CRS";
                crsInfo.isGeographic = true;
                break;
            case PJ_TYPE_PROJECTED_CRS:
                crsInfo.type = "Projected CRS";
                crsInfo.isProjected = true;
                break;
            case PJ_TYPE_COMPOUND_CRS:
                crsInfo.type = "Compound CRS";
                break;
            case PJ_TYPE_VERTICAL_CRS:
                crsInfo.type = "Vertical CRS";
                break;
            default:
                crsInfo.type = "Other CRS";
                break;
        }
        
        return true;
    }
    
} // namespace icrs::parser 
