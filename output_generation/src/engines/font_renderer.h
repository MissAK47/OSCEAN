/**
 * @file font_renderer.h
 * @brief 字体渲染器，用于在图像上绘制文本
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// 前向声明FreeType类型
typedef struct FT_LibraryRec_* FT_Library;
typedef struct FT_FaceRec_* FT_Face;

namespace oscean {
namespace output {

/**
 * @brief 字体渲染器类
 * 
 * 使用FreeType库在图像上渲染文本
 */
class FontRenderer {
public:
    /**
     * @brief 文本对齐方式
     */
    enum class Alignment {
        LEFT,
        CENTER,
        RIGHT
    };

    /**
     * @brief 字体样式
     */
    struct FontStyle {
        std::string fontPath;      // 字体文件路径
        int fontSize;              // 字体大小（像素）
        uint8_t color[4];          // RGBA颜色
        bool bold;                 // 是否粗体
        bool italic;               // 是否斜体
        
        FontStyle() : fontSize(12), bold(false), italic(false) {
            color[0] = color[1] = color[2] = 0;  // 默认黑色
            color[3] = 255;  // 不透明
        }
    };

    /**
     * @brief 构造函数
     */
    FontRenderer();

    /**
     * @brief 析构函数
     */
    ~FontRenderer();

    /**
     * @brief 初始化字体渲染器
     * @param defaultFontPath 默认字体文件路径
     * @return 初始化是否成功
     */
    bool initialize(const std::string& defaultFontPath = "");

    /**
     * @brief 设置字体样式
     * @param style 字体样式
     */
    void setFontStyle(const FontStyle& style);

    /**
     * @brief 在图像上绘制文本
     * @param imageData 图像数据（RGBA格式）
     * @param width 图像宽度
     * @param height 图像高度
     * @param text 要绘制的文本
     * @param x X坐标
     * @param y Y坐标
     * @param alignment 文本对齐方式
     * @return 绘制是否成功
     */
    bool drawText(
        std::vector<uint8_t>& imageData,
        int width, int height,
        const std::string& text,
        int x, int y,
        Alignment alignment = Alignment::LEFT
    );

    /**
     * @brief 获取文本的渲染尺寸
     * @param text 文本内容
     * @param width 输出宽度
     * @param height 输出高度
     */
    void getTextSize(const std::string& text, int& width, int& height) const;

    /**
     * @brief 绘制带背景的文本
     * @param imageData 图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param text 文本内容
     * @param x X坐标
     * @param y Y坐标
     * @param bgColor 背景颜色（RGBA）
     * @param padding 内边距
     */
    bool drawTextWithBackground(
        std::vector<uint8_t>& imageData,
        int width, int height,
        const std::string& text,
        int x, int y,
        const uint8_t bgColor[4],
        int padding = 2
    );

    /**
     * @brief 绘制旋转的文本
     * @param imageData 图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param text 文本内容
     * @param x X坐标
     * @param y Y坐标
     * @param angle 旋转角度（度）
     */
    bool drawRotatedText(
        std::vector<uint8_t>& imageData,
        int width, int height,
        const std::string& text,
        int x, int y,
        double angle
    );

private:
    // FreeType库和字体
    FT_Library m_library;
    FT_Face m_face;
    
    // 当前字体样式
    FontStyle m_currentStyle;
    
    // 字体缓存
    std::map<std::string, FT_Face> m_fontCache;
    
    // 是否已初始化
    bool m_initialized;
    
    // 加载字体
    bool loadFont(const std::string& fontPath);
    
    // 渲染单个字符
    bool renderGlyph(
        std::vector<uint8_t>& imageData,
        int imgWidth, int imgHeight,
        int& penX, int& penY,
        unsigned long charCode
    );
    
    // 混合像素（alpha混合）
    void blendPixel(
        uint8_t* dst,
        const uint8_t* src,
        uint8_t alpha
    );
    
    // 查找系统字体
    std::string findSystemFont() const;
};

} // namespace output
} // namespace oscean 