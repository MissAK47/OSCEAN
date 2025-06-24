#define _USE_MATH_DEFINES  // 必须放在包含cmath之前
#include "crs_transformer.h"
#include <cmath>
#include <mutex>
#include <proj.h>
#include <stdexcept>
#include "common_utils/utilities/logging_utils.h"
#include <gdal_priv.h>

namespace icrs::transformer
{
    CrsTransformer::CrsTransformer()
        : m_Context(nullptr)
    {
        // 🚀 检查GDAL是否已由全局初始化器初始化
        static std::once_flag gdalCheckFlag;
        std::call_once(gdalCheckFlag, []() {
            if (GDALGetDriverCount() == 0) {
                throw std::runtime_error("GDAL未初始化！请确保在main函数中调用了GdalGlobalInitializer::initialize()");
            }
            // GDALAllRegister(); // ❌ 已移除 - 现在由GdalGlobalInitializer统一管理
            spdlog::info("GDAL已由全局初始化器初始化，驱动数量: {}", GDALGetDriverCount());
        });
        
        initialize();
    }
    
    CrsTransformer::~CrsTransformer()
    {
        // 清理所有转换器资源
        for (auto& pair : m_Transformers)
        {
            if (pair.second)
            {
                proj_destroy((PJ*)pair.second);
                pair.second = nullptr;
            }
        }
        m_Transformers.clear();
        
        // 清理上下文
        if (m_Context)
        {
            proj_context_destroy((PJ_CONTEXT*)m_Context);
            m_Context = nullptr;
        }
    }
    
    bool CrsTransformer::initialize()
    {
        // 创建PROJ上下文
        m_Context = (ProjContext)proj_context_create();
        if (!m_Context)
        {
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
    
    bool CrsTransformer::transform(const oscean::core_services::Point& srcPoint, 
                                 const std::string& srcCrs, 
                                 const std::string& dstCrs, 
                                 oscean::core_services::TransformedPoint& dstPoint)
    {
        // 如果源和目标CRS相同，直接返回原始点
        if (srcCrs == dstCrs)
        {
            dstPoint.x = srcPoint.x;
            dstPoint.y = srcPoint.y;
            dstPoint.z = srcPoint.z;
            dstPoint.status = oscean::core_services::TransformStatus::SUCCESS;
            return true;
        }
        
        // 获取或创建转换器
        std::string key = getTransformerKey(srcCrs, dstCrs);
        ProjObject transformer = nullptr;
        
        // 查找缓存的转换器
        auto it = m_Transformers.find(key);
        if (it != m_Transformers.end() && it->second)
        {
            transformer = it->second;
        }
        else
        {
            // 创建新的转换器
            transformer = createTransformer(srcCrs, dstCrs);
            if (!transformer)
            {
                dstPoint.status = oscean::core_services::TransformStatus::FAILED;
                dstPoint.errorMessage = "Failed to create transformer";
                return false;
            }
            
            // 存入缓存
            m_Transformers[key] = transformer;
        }
        
        // 执行坐标转换
        PJ_COORD srcCoord;
        srcCoord.xyzt.x = srcPoint.x;
        srcCoord.xyzt.y = srcPoint.y;
        srcCoord.xyzt.z = srcPoint.z.value_or(0.0);
        srcCoord.xyzt.t = 0.0;  // 时间不使用
        
        PJ_COORD dstCoord = proj_trans((PJ*)transformer, PJ_FWD, srcCoord);
        
        // 检查转换结果
        if (proj_errno((PJ*)transformer) != 0)
        {
            dstPoint.status = oscean::core_services::TransformStatus::FAILED;
            dstPoint.errorMessage = proj_errno_string(proj_errno((PJ*)transformer));
            return false;
        }
        
        // 设置转换结果
        dstPoint.x = dstCoord.xyzt.x;
        dstPoint.y = dstCoord.xyzt.y;
        if (srcPoint.z.has_value()) {
            dstPoint.z = dstCoord.xyzt.z;
        }
        dstPoint.status = oscean::core_services::TransformStatus::SUCCESS;
        
        return true;
    }
    
    bool CrsTransformer::hasCrs(const std::string& crsId)
    {
        // 检查CRS是否有效
        PJ* crs = proj_create((PJ_CONTEXT*)m_Context, crsId.c_str());
        if (!crs)
        {
            return false;
        }
        
        // 清理资源
        proj_destroy(crs);
        return true;
    }
    
    ProjObject CrsTransformer::createTransformer(const std::string& srcCrs, const std::string& dstCrs)
    {
        // 创建源和目标CRS
        PJ* sourcePj = proj_create((PJ_CONTEXT*)m_Context, srcCrs.c_str());
        if (!sourcePj)
        {
            return nullptr;
        }
        
        PJ* targetPj = proj_create((PJ_CONTEXT*)m_Context, dstCrs.c_str());
        if (!targetPj)
        {
            proj_destroy(sourcePj);
            return nullptr;
        }
        
        // 创建转换器
        PJ* transformer = proj_create_crs_to_crs_from_pj((PJ_CONTEXT*)m_Context, sourcePj, targetPj, nullptr, nullptr);
        
        // 清理临时资源
        proj_destroy(sourcePj);
        proj_destroy(targetPj);
        
        if (!transformer)
        {
            return nullptr;
        }
        
        // 规范化转换器以获得最佳精度
        PJ* normalizedTransformer = proj_normalize_for_visualization((PJ_CONTEXT*)m_Context, transformer);
        if (normalizedTransformer)
        {
            proj_destroy(transformer);
            return (ProjObject)normalizedTransformer;
        }
        
        return (ProjObject)transformer;
    }
    
    std::string CrsTransformer::getTransformerKey(const std::string& srcCrs, const std::string& dstCrs)
    {
        return srcCrs + " -> " + dstCrs;
    }
    
} // namespace icrs::transformer 
