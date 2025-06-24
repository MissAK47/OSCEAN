/**
 * @file font_renderer.cpp
 * @brief 字体渲染器实现
 */

#include "font_renderer.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace oscean {
namespace output {

FontRenderer::FontRenderer() 
    : m_library(nullptr)
    , m_face(nullptr)
    , m_initialized(false) {
}

FontRenderer::~FontRenderer() {
    if (m_face) {
        FT_Done_Face(m_face);
    }
    
    for (auto& pair : m_fontCache) {
        FT_Done_Face(pair.second);
    }
    
    if (m_library) {
        FT_Done_FreeType(m_library);
    }
}

bool FontRenderer::initialize(const std::string& defaultFontPath) {
    if (m_initialized) {
        return true;
    }
    
    // 初始化FreeType库
    if (FT_Init_FreeType(&m_library)) {
        BOOST_LOG_TRIVIAL(error) << "Failed to initialize FreeType library";
        return false;
    }
    
    // 确定要使用的字体路径
    std::string fontPath = defaultFontPath;
    if (fontPath.empty()) {
        fontPath = findSystemFont();
    }
    
    if (fontPath.empty()) {
        BOOST_LOG_TRIVIAL(error) << "No font file specified and no system font found";
        FT_Done_FreeType(m_library);
        m_library = nullptr;
        return false;
    }
    
    // 加载默认字体
    if (!loadFont(fontPath)) {
        FT_Done_FreeType(m_library);
        m_library = nullptr;
        return false;
    }
    
    m_currentStyle.fontPath = fontPath;
    m_initialized = true;
    
    BOOST_LOG_TRIVIAL(info) << "FontRenderer initialized with font: " << fontPath;
    return true;
}

bool FontRenderer::loadFont(const std::string& fontPath) {
    if (!boost::filesystem::exists(fontPath)) {
        BOOST_LOG_TRIVIAL(error) << "Font file not found: " << fontPath;
        return false;
    }
    
    // 检查缓存
    auto it = m_fontCache.find(fontPath);
    if (it != m_fontCache.end()) {
        m_face = it->second;
        return true;
    }
    
    // 加载新字体
    FT_Face newFace;
    if (FT_New_Face(m_library, fontPath.c_str(), 0, &newFace)) {
        BOOST_LOG_TRIVIAL(error) << "Failed to load font: " << fontPath;
        return false;
    }
    
    // 设置字体大小
    FT_Set_Pixel_Sizes(newFace, 0, m_currentStyle.fontSize);
    
    // 缓存字体
    m_fontCache[fontPath] = newFace;
    m_face = newFace;
    
    return true;
}

void FontRenderer::setFontStyle(const FontStyle& style) {
    m_currentStyle = style;
    
    if (!style.fontPath.empty() && style.fontPath != m_currentStyle.fontPath) {
        loadFont(style.fontPath);
    }
    
    if (m_face) {
        FT_Set_Pixel_Sizes(m_face, 0, style.fontSize);
    }
}

bool FontRenderer::drawText(
    std::vector<uint8_t>& imageData,
    int width, int height,
    const std::string& text,
    int x, int y,
    Alignment alignment) {
    
    if (!m_initialized || !m_face || text.empty()) {
        return false;
    }
    
    // 计算文本尺寸以处理对齐
    int textWidth, textHeight;
    getTextSize(text, textWidth, textHeight);
    
    // 调整起始位置
    int penX = x;
    int penY = y;
    
    switch (alignment) {
        case Alignment::CENTER:
            penX = x - textWidth / 2;
            break;
        case Alignment::RIGHT:
            penX = x - textWidth;
            break;
        case Alignment::LEFT:
        default:
            break;
    }
    
    // 渲染每个字符
    for (char c : text) {
        if (!renderGlyph(imageData, width, height, penX, penY, static_cast<unsigned char>(c))) {
            BOOST_LOG_TRIVIAL(warning) << "Failed to render character: " << c;
        }
    }
    
    return true;
}

bool FontRenderer::renderGlyph(
    std::vector<uint8_t>& imageData,
    int imgWidth, int imgHeight,
    int& penX, int& penY,
    unsigned long charCode) {
    
    // 加载字形
    if (FT_Load_Char(m_face, charCode, FT_LOAD_RENDER)) {
        return false;
    }
    
    FT_GlyphSlot slot = m_face->glyph;
    
    // 计算绘制位置
    int drawX = penX + slot->bitmap_left;
    int drawY = penY - slot->bitmap_top;
    
    // 绘制位图到图像
    for (unsigned int row = 0; row < slot->bitmap.rows; ++row) {
        for (unsigned int col = 0; col < slot->bitmap.width; ++col) {
            int x = drawX + col;
            int y = drawY + row;
            
            // 边界检查
            if (x < 0 || x >= imgWidth || y < 0 || y >= imgHeight) {
                continue;
            }
            
            // 获取字形像素的alpha值
            unsigned char alpha = slot->bitmap.buffer[row * slot->bitmap.width + col];
            if (alpha == 0) continue;
            
            // 计算目标像素位置
            int pixelIndex = (y * imgWidth + x) * 4;
            
            // Alpha混合
            blendPixel(&imageData[pixelIndex], m_currentStyle.color, alpha);
        }
    }
    
    // 更新笔位置
    penX += slot->advance.x >> 6;
    penY += slot->advance.y >> 6;
    
    return true;
}

void FontRenderer::blendPixel(uint8_t* dst, const uint8_t* src, uint8_t alpha) {
    float a = alpha / 255.0f;
    float oneMinusA = 1.0f - a;
    
    dst[0] = static_cast<uint8_t>(src[0] * a + dst[0] * oneMinusA);
    dst[1] = static_cast<uint8_t>(src[1] * a + dst[1] * oneMinusA);
    dst[2] = static_cast<uint8_t>(src[2] * a + dst[2] * oneMinusA);
    dst[3] = std::min(255, dst[3] + alpha);
}

void FontRenderer::getTextSize(const std::string& text, int& width, int& height) const {
    width = 0;
    height = 0;
    
    if (!m_face || text.empty()) {
        return;
    }
    
    int maxHeight = 0;
    int totalWidth = 0;
    
    for (char c : text) {
        if (FT_Load_Char(m_face, static_cast<unsigned char>(c), FT_LOAD_DEFAULT)) {
            continue;
        }
        
        totalWidth += m_face->glyph->advance.x >> 6;
        
        int glyphHeight = m_face->glyph->bitmap.rows;
        if (glyphHeight > maxHeight) {
            maxHeight = glyphHeight;
        }
    }
    
    width = totalWidth;
    height = maxHeight + m_currentStyle.fontSize / 4; // 添加一些额外空间
}

bool FontRenderer::drawTextWithBackground(
    std::vector<uint8_t>& imageData,
    int width, int height,
    const std::string& text,
    int x, int y,
    const uint8_t bgColor[4],
    int padding) {
    
    // 获取文本尺寸
    int textWidth, textHeight;
    getTextSize(text, textWidth, textHeight);
    
    // 绘制背景矩形
    int bgX = x - padding;
    int bgY = y - textHeight - padding;
    int bgWidth = textWidth + 2 * padding;
    int bgHeight = textHeight + 2 * padding;
    
    for (int row = 0; row < bgHeight; ++row) {
        for (int col = 0; col < bgWidth; ++col) {
            int px = bgX + col;
            int py = bgY + row;
            
            if (px >= 0 && px < width && py >= 0 && py < height) {
                int idx = (py * width + px) * 4;
                std::memcpy(&imageData[idx], bgColor, 4);
            }
        }
    }
    
    // 绘制文本
    return drawText(imageData, width, height, text, x, y);
}

std::string FontRenderer::findSystemFont() const {
#ifdef _WIN32
    // Windows系统字体路径
    char winDir[MAX_PATH];
    GetWindowsDirectoryA(winDir, MAX_PATH);
    std::string fontPath = std::string(winDir) + "\\Fonts\\arial.ttf";
    if (boost::filesystem::exists(fontPath)) {
        return fontPath;
    }
    // 尝试其他常见字体
    fontPath = std::string(winDir) + "\\Fonts\\calibri.ttf";
    if (boost::filesystem::exists(fontPath)) {
        return fontPath;
    }
#elif defined(__linux__)
    // Linux系统字体路径
    std::vector<std::string> fontPaths = {
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (boost::filesystem::exists(path)) {
            return path;
        }
    }
#elif defined(__APPLE__)
    // macOS系统字体路径
    std::vector<std::string> fontPaths = {
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (boost::filesystem::exists(path)) {
            return path;
        }
    }
#endif
    
    return "";
}

bool FontRenderer::drawRotatedText(
    std::vector<uint8_t>& imageData,
    int width, int height,
    const std::string& text,
    int x, int y,
    double angle) {
    
    // 简化实现：暂时只支持90度的倍数旋转
    // 完整实现需要使用FreeType的矩阵变换功能
    
    int normalizedAngle = static_cast<int>(angle) % 360;
    if (normalizedAngle < 0) normalizedAngle += 360;
    
    if (normalizedAngle == 0) {
        return drawText(imageData, width, height, text, x, y);
    } else if (normalizedAngle == 90) {
        // 90度旋转：从下到上绘制
        int penX = x;
        int penY = y;
        
        for (char c : text) {
            if (!renderGlyph(imageData, width, height, penX, penY, static_cast<unsigned char>(c))) {
                BOOST_LOG_TRIVIAL(warning) << "Failed to render rotated character: " << c;
            }
            penY -= m_face->glyph->advance.x >> 6; // 使用水平前进作为垂直偏移
        }
        return true;
    }
    
    // 其他角度暂不支持
    BOOST_LOG_TRIVIAL(warning) << "Rotation angle " << angle << " not supported, drawing normal text";
    return drawText(imageData, width, height, text, x, y);
}

} // namespace output
} // namespace oscean 