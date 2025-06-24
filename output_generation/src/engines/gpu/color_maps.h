/**
 * @file color_maps.h
 * @brief 预定义的颜色映射方案
 */

#ifndef OSCEAN_GPU_COLOR_MAPS_H
#define OSCEAN_GPU_COLOR_MAPS_H

#include <array>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace oscean {
namespace output_generation {
namespace gpu {

// 颜色映射结构体
struct ColorMap {
    static constexpr int SIZE = 256;
    std::array<float, SIZE * 4> data; // RGBA格式
    
    ColorMap() = default;
    ColorMap(const std::array<float, SIZE * 4>& colorData) : data(colorData) {}
};

// 颜色映射生成函数
namespace colormaps {

// Viridis颜色映射
inline ColorMap generateViridis() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        // Viridis颜色映射的近似公式
        cmap.data[i * 4 + 0] = 0.267004f + 0.004874f * t + 0.329415f * t * t - 0.255635f * t * t * t;
        cmap.data[i * 4 + 1] = 0.004874f + 0.761241f * t + 0.247462f * t * t - 0.148581f * t * t * t;
        cmap.data[i * 4 + 2] = 0.329415f + 0.185371f * t - 1.048072f * t * t + 0.614231f * t * t * t;
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Plasma颜色映射
inline ColorMap generatePlasma() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        // Plasma颜色映射的近似公式
        cmap.data[i * 4 + 0] = 0.050383f + 2.021084f * t - 1.781748f * t * t + 0.553282f * t * t * t;
        cmap.data[i * 4 + 1] = 0.029803f + 0.224816f * t + 0.878260f * t * t - 0.376563f * t * t * t;
        cmap.data[i * 4 + 2] = 0.527975f + 1.914169f * t - 4.023257f * t * t + 2.165586f * t * t * t;
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Inferno颜色映射
inline ColorMap generateInferno() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        // Inferno颜色映射的近似公式
        cmap.data[i * 4 + 0] = 0.001462f + 1.087595f * t + 0.810693f * t * t - 0.684280f * t * t * t;
        cmap.data[i * 4 + 1] = 0.000466f + 0.540948f * t + 0.344863f * t * t - 0.177636f * t * t * t;
        cmap.data[i * 4 + 2] = 0.013866f + 1.296073f * t - 3.018556f * t * t + 1.616894f * t * t * t;
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Magma颜色映射
inline ColorMap generateMagma() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        // Magma颜色映射的近似公式
        cmap.data[i * 4 + 0] = 0.001462f + 1.078260f * t + 0.843661f * t * t - 0.713544f * t * t * t;
        cmap.data[i * 4 + 1] = 0.000466f + 0.427802f * t + 0.688223f * t * t - 0.384981f * t * t * t;
        cmap.data[i * 4 + 2] = 0.013866f + 1.384508f * t - 3.290936f * t * t + 1.786953f * t * t * t;
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Cividis颜色映射（色盲友好）
inline ColorMap generateCividis() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        // Cividis颜色映射的近似公式
        cmap.data[i * 4 + 0] = 0.0f + 0.267331f * t + 0.311261f * t * t + 0.183270f * t * t * t;
        cmap.data[i * 4 + 1] = 0.135112f + 0.662569f * t + 0.202285f * t * t - 0.047847f * t * t * t;
        cmap.data[i * 4 + 2] = 0.329898f + 0.780776f * t - 1.182017f * t * t + 0.361665f * t * t * t;
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Turbo颜色映射（Google的改进版jet）
inline ColorMap generateTurbo() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float r, g, b;
        
        // Turbo颜色映射的分段定义
        if (t < 0.13f) {
            r = 0.18995f + 4.53537f * t;
            g = 0.0f;
            b = 0.34167f + 1.88433f * t;
        } else if (t < 0.37f) {
            r = 0.73333f + 0.87502f * (t - 0.13f);
            g = 0.0f + 3.95833f * (t - 0.13f);
            b = 0.57100f + 0.89583f * (t - 0.13f);
        } else if (t < 0.65f) {
            r = 0.94333f + 0.19643f * (t - 0.37f);
            g = 0.95000f + 0.17857f * (t - 0.37f);
            b = 0.78600f - 2.21429f * (t - 0.37f);
        } else if (t < 0.87f) {
            r = 0.99833f - 0.72727f * (t - 0.65f);
            g = 1.00000f - 1.04545f * (t - 0.65f);
            b = 0.16600f + 0.15909f * (t - 0.65f);
        } else {
            r = 0.83833f - 1.21154f * (t - 0.87f);
            g = 0.77000f - 1.84615f * (t - 0.87f);
            b = 0.20100f - 0.53846f * (t - 0.87f);
        }
        
        cmap.data[i * 4 + 0] = std::max(0.0f, std::min(1.0f, r));
        cmap.data[i * 4 + 1] = std::max(0.0f, std::min(1.0f, g));
        cmap.data[i * 4 + 2] = std::max(0.0f, std::min(1.0f, b));
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Jet颜色映射（经典但有争议）
inline ColorMap generateJet() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float r, g, b;
        
        // Jet颜色映射的分段定义
        if (t < 0.125f) {
            r = 0.0f;
            g = 0.0f;
            b = 0.5f + 4.0f * t;
        } else if (t < 0.375f) {
            r = 0.0f;
            g = 4.0f * (t - 0.125f);
            b = 1.0f;
        } else if (t < 0.625f) {
            r = 4.0f * (t - 0.375f);
            g = 1.0f;
            b = 1.0f - 4.0f * (t - 0.375f);
        } else if (t < 0.875f) {
            r = 1.0f;
            g = 1.0f - 4.0f * (t - 0.625f);
            b = 0.0f;
        } else {
            r = 1.0f - 4.0f * (t - 0.875f);
            g = 0.0f;
            b = 0.0f;
        }
        
        cmap.data[i * 4 + 0] = std::max(0.0f, std::min(1.0f, r));
        cmap.data[i * 4 + 1] = std::max(0.0f, std::min(1.0f, g));
        cmap.data[i * 4 + 2] = std::max(0.0f, std::min(1.0f, b));
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Hot颜色映射
inline ColorMap generateHot() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float r, g, b;
        
        // Hot颜色映射的定义
        if (t < 0.365f) {
            r = 2.747f * t;
            g = 0.0f;
            b = 0.0f;
        } else if (t < 0.746f) {
            r = 1.0f;
            g = 2.625f * (t - 0.365f);
            b = 0.0f;
        } else {
            r = 1.0f;
            g = 1.0f;
            b = 3.906f * (t - 0.746f);
        }
        
        cmap.data[i * 4 + 0] = std::max(0.0f, std::min(1.0f, r));
        cmap.data[i * 4 + 1] = std::max(0.0f, std::min(1.0f, g));
        cmap.data[i * 4 + 2] = std::max(0.0f, std::min(1.0f, b));
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Cool颜色映射
inline ColorMap generateCool() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        cmap.data[i * 4 + 0] = t;
        cmap.data[i * 4 + 1] = 1.0f - t;
        cmap.data[i * 4 + 2] = 1.0f;
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

// Gray颜色映射
inline ColorMap generateGray() {
    ColorMap cmap;
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        cmap.data[i * 4 + 0] = t;
        cmap.data[i * 4 + 1] = t;
        cmap.data[i * 4 + 2] = t;
        cmap.data[i * 4 + 3] = 1.0f;
    }
    return cmap;
}

} // namespace colormaps

// 颜色映射管理器
class ColorMapManager {
public:
    static ColorMapManager& getInstance() {
        static ColorMapManager instance;
        return instance;
    }
    
    const ColorMap& getColorMap(const std::string& name) {
        auto it = m_colorMaps.find(name);
        if (it != m_colorMaps.end()) {
            return it->second;
        }
        // 默认返回viridis
        return m_colorMaps["viridis"];
    }
    
    std::vector<std::string> getAvailableColorMaps() const {
        std::vector<std::string> names;
        for (const auto& pair : m_colorMaps) {
            names.push_back(pair.first);
        }
        return names;
    }
    
private:
    ColorMapManager() {
        // 初始化所有颜色映射
        m_colorMaps["viridis"] = colormaps::generateViridis();
        m_colorMaps["plasma"] = colormaps::generatePlasma();
        m_colorMaps["inferno"] = colormaps::generateInferno();
        m_colorMaps["magma"] = colormaps::generateMagma();
        m_colorMaps["cividis"] = colormaps::generateCividis();
        m_colorMaps["turbo"] = colormaps::generateTurbo();
        m_colorMaps["jet"] = colormaps::generateJet();
        m_colorMaps["hot"] = colormaps::generateHot();
        m_colorMaps["cool"] = colormaps::generateCool();
        m_colorMaps["gray"] = colormaps::generateGray();
    }
    
    std::map<std::string, ColorMap> m_colorMaps;
};

} // namespace gpu
} // namespace output_generation
} // namespace oscean

#endif // OSCEAN_GPU_COLOR_MAPS_H 