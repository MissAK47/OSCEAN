#include "crs_inspector.h"
#include "core_services/crs/crs_operation_types.h" // For CRSDetailedParameters
#include "core_services/common_data_types.h"     // For CRSInfo
#include "crs_parser.h" // for helper methods
#include <proj.h>
// GDAL头文件移除 - CRS检查器只使用PROJ库
#include <spdlog/spdlog.h>
#include <string>
#include <mutex>
#include <algorithm>
#include "common_utils/utilities/logging_utils.h"

#include <stdexcept>

// Placeholder for PROJ library headers (e.g., #include <proj.h>)

namespace icrs::inspector {

CrsInspector::CrsInspector()
    : m_Context(nullptr)
{
    // CRS检查器只处理坐标系统验证，不负责GDAL初始化
    // GDAL初始化由数据访问服务负责
    initialize();
}

CrsInspector::~CrsInspector() {
    // 清理上下文
    if (m_Context) {
        proj_context_destroy((PJ_CONTEXT*)m_Context);
        m_Context = nullptr;
    }
}

bool CrsInspector::initialize() {
    // 创建PROJ上下文
    m_Context = (ProjContext)proj_context_create();
    if (!m_Context) {
        return false;
    }
    
    // 设置PROJ数据路径
    #ifdef _WIN32
    _putenv_s("PROJ_LIB", "C:\\Users\\Administrator\\vcpkg\\installed\\x64-windows\\share\\proj");
    #else
    setenv("PROJ_LIB", "C:/Users/Administrator/vcpkg/installed/x64-windows/share/proj", 1);
    #endif
    
    return true;
}

bool CrsInspector::isValid(const std::string& crsDefinition) {
    if (crsDefinition.empty()) {
        return false;
    }
    
    ProjObject crs = createProjObj(crsDefinition);
    if (!crs) {
        return false;
    }
    
    proj_destroy((PJ*)crs);
    return true;
}

bool CrsInspector::getInfo(const std::string& crsDefinition, oscean::core_services::CRSInfo& crsInfo) {
    if (crsDefinition.empty()) {
        return false;
    }
    
    ProjObject crs = createProjObj(crsDefinition);
    if (!crs) {
        return false;
    }
    
    // 获取WKT
    const char* wkt = proj_as_wkt(NULL, (PJ*)crs, PJ_WKT2_2019, NULL);
    if (wkt) {
        crsInfo.wkt = wkt;
        crsInfo.wktext = wkt; // 兼容性字段
    }
    
    // 获取PROJ字符串
    const char* proj_str = proj_as_proj_string(NULL, (PJ*)crs, PJ_PROJ_5, NULL);
    if (proj_str) {
        crsInfo.projString = proj_str;
        crsInfo.proj4text = proj_str; // 兼容性字段
    }
    
    // 获取EPSG代码
    const char* code = proj_get_id_code((PJ*)crs, 0);
    if (code) {
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
    } else {
        crsInfo.epsgCode = -1;
    }
    
    // 获取名称
    const char* name = proj_get_name((PJ*)crs);
    if (name) {
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
    
    proj_destroy((PJ*)crs);
    return true;
}

bool CrsInspector::isCompatible(const std::string& srcCrs, const std::string& dstCrs) {
    if (srcCrs.empty() || dstCrs.empty()) {
        return false;
    }
    
    PJ* sourcePj = proj_create((PJ_CONTEXT*)m_Context, srcCrs.c_str());
    if (!sourcePj) {
        return false;
    }
    
    PJ* targetPj = proj_create((PJ_CONTEXT*)m_Context, dstCrs.c_str());
    if (!targetPj) {
        proj_destroy(sourcePj);
        return false;
    }
    
    // 检查两个CRS是否可以进行转换
    PJ* transformer = proj_create_crs_to_crs_from_pj((PJ_CONTEXT*)m_Context, sourcePj, targetPj, nullptr, nullptr);
    
    proj_destroy(sourcePj);
    proj_destroy(targetPj);
    
    if (!transformer) {
        return false;
    }
    
    proj_destroy(transformer);
    return true;
}

ProjObject CrsInspector::createProjObj(const std::string& crsDefinition) {
    if (!m_Context) {
        initialize();
    }
    
    if (crsDefinition.empty()) {
        return nullptr;
    }
    
    PJ* crs = proj_create((PJ_CONTEXT*)m_Context, crsDefinition.c_str());
    return (ProjObject)crs;
}

void CrsInspector::cleanup(ProjContext ctx, ProjObject crs) {
    if (crs) {
        proj_destroy((PJ*)crs);
    }
    
    if (ctx && ctx != m_Context) {
        proj_context_destroy((PJ_CONTEXT*)ctx);
    }
}

} // namespace icrs::inspector 
