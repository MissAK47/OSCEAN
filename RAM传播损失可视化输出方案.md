# RAM传播损失可视化输出方案

## 1. 概述

本方案描述如何将RAM-PE计算得到的传播损失数据通过OSCEAN的output模块生成图像，并以压缩的二进制流形式输出给前端显示，实现零磁盘I/O的高性能可视化。

## 2. 数据流程

```mermaid
graph TB
    subgraph "RAM计算结果"
        A[传播损失场<br/>TL(x,y,z)]
        B[相位场<br/>Phase(x,y,z)]
    end
    
    subgraph "数据转换"
        C[GridData构建<br/>将TL数据封装]
        D[数据范围分析<br/>计算min/max]
        E[切片选择<br/>选择显示深度]
    end
    
    subgraph "图像生成"
        F[颜色映射<br/>TL值→RGB]
        G[图像渲染<br/>生成RGBA数组]
        H[添加标注<br/>图例/标题/等值线]
    end
    
    subgraph "压缩输出"
        I[PNG编码<br/>内存中压缩]
        J[Base64编码<br/>用于传输]
        K[二进制流<br/>直接传输]
    end
    
    subgraph "前端显示"
        L[WebSocket/HTTP<br/>传输]
        M[Canvas/WebGL<br/>渲染]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    I --> K
    J --> L
    K --> L
    L --> M
```

## 3. 具体实现步骤

### 3.1 RAM计算结果到GridData的转换

```cpp
/**
 * @brief 将RAM传播损失结果转换为GridData格式
 */
class RamResultToGridDataConverter {
public:
    /**
     * @brief 转换传播损失场到GridData
     * @param result RAM多方位计算结果
     * @param depthIndex 要显示的深度索引（-1表示垂直平均）
     * @return 包含传播损失的GridData
     */
    static boost::shared_ptr<GridData> convertTLField(
        const RamPeMultiAzimuthResult& result,
        int depthIndex = -1) {
        
        // 确定输出网格尺寸
        size_t nx = result.cartesianField->getColumns();
        size_t ny = result.cartesianField->getRows();
        
        // 创建GridData
        GridDefinition def;
        def.cols = nx;
        def.rows = ny;
        def.bands = 1;  // 单波段：传播损失值
        
        auto gridData = boost::make_shared<GridData>(
            def, DataType::Float32, 1);
        
        // 设置空间参考
        gridData->setCRS(result.cartesianField->getCRS());
        gridData->setBoundingBox(result.cartesianField->getBoundingBox());
        
        // 设置元数据
        gridData->setVariableName("TransmissionLoss");
        gridData->setUnit("dB");
        gridData->setLongName("Transmission Loss");
        
        // 提取指定深度的传播损失数据
        if (depthIndex >= 0) {
            // 特定深度切片
            extractDepthSlice(result, depthIndex, gridData);
            gridData->setAttribute("depth", 
                std::to_string(result.polarField.depths[depthIndex]));
        } else {
            // 垂直平均或最大值
            extractVerticalAggregate(result, gridData);
            gridData->setAttribute("depth", "vertically_averaged");
        }
        
        // 设置时间戳
        gridData->setTimestamp(boost::posix_time::microsec_clock::local_time());
        
        return gridData;
    }
    
private:
    static void extractDepthSlice(
        const RamPeMultiAzimuthResult& result,
        int depthIndex,
        boost::shared_ptr<GridData>& gridData) {
        
        // 从笛卡尔场中提取指定深度的数据
        float* data = gridData->getData<float>();
        const auto& field = result.cartesianField;
        
        size_t nx = field->getColumns();
        size_t ny = field->getRows();
        
        for (size_t i = 0; i < ny; ++i) {
            for (size_t j = 0; j < nx; ++j) {
                // 获取复数场值
                Complex fieldValue = field->getValue(i, j, depthIndex);
                
                // 计算传播损失: TL = -20*log10(|field|)
                double magnitude = std::abs(fieldValue);
                double tl = -20.0 * std::log10(std::max(magnitude, 1e-10));
                
                data[i * nx + j] = static_cast<float>(tl);
            }
        }
    }
};
```

### 3.2 使用Output模块生成内存图像

```cpp
/**
 * @brief RAM可视化输出管理器
 */
class RamVisualizationManager {
private:
    boost::shared_ptr<IOutputService> outputService_;
    boost::shared_ptr<VisualizationEngine> vizEngine_;
    
public:
    /**
     * @brief 生成传播损失图像的二进制流
     * @param tlGridData 传播损失GridData
     * @param options 可视化选项
     * @return 压缩的图像二进制数据
     */
    boost::future<std::vector<unsigned char>> generateTLImageStream(
        boost::shared_ptr<GridData> tlGridData,
        const TLVisualizationOptions& options) {
        
        // 创建内存数据读取器
        auto memReader = boost::make_shared<InMemoryDataReader>(tlGridData);
        
        // 构建输出请求
        OutputRequest request;
        request.dataSource = memReader;
        request.format = "png";  // 或 "jpeg" 根据需要
        request.streamOutput = true;  // 关键：输出到内存流而不是文件
        
        // 设置样式选项
        request.style = StyleOptions();
        request.style->colorMap = options.colorMap;  // 如 "jet", "viridis"
        request.style->drawContours = options.drawContours;
        request.style->contourLevels = options.contourLevels;
        
        // 设置PNG压缩选项
        request.creationOptions = {
            {"COMPRESS", "DEFLATE"},
            {"ZLEVEL", std::to_string(options.compressionLevel)},
            {"WORLDFILE", "NO"}  // 不需要地理参考文件
        };
        
        // 异步处理请求
        return outputService_->processRequest(request)
            .then([](boost::future<OutputResult> resultFuture) {
                auto result = resultFuture.get();
                if (result.dataStream.has_value()) {
                    return result.dataStream.value();
                } else {
                    throw std::runtime_error("Failed to generate image stream");
                }
            });
    }
    
    /**
     * @brief 生成带图例的传播损失图像
     */
    boost::future<std::vector<unsigned char>> generateTLImageWithLegend(
        boost::shared_ptr<GridData> tlGridData,
        const TLVisualizationOptions& options) {
        
        // 先生成主图像
        auto mainImageFuture = generateTLImageStream(tlGridData, options);
        
        // 获取数据统计
        auto stats = calculateStatistics(tlGridData);
        
        // 生成图例
        auto legendFuture = vizEngine_->generateLegend(
            options.colorMap,
            stats.minValue,
            stats.maxValue,
            "Transmission Loss (dB)",
            "memory://legend",  // 内存路径
            200, 600
        );
        
        // 组合主图像和图例
        return boost::when_all(mainImageFuture, legendFuture)
            .then([this](auto futures) {
                auto mainImage = std::get<0>(futures).get();
                auto legendPath = std::get<1>(futures).get();
                
                // 在内存中组合图像
                return combineImagesInMemory(mainImage, legendPath);
            });
    }
    
private:
    /**
     * @brief 高性能的内存图像生成（使用SIMD/GPU加速）
     */
    std::vector<unsigned char> generateOptimizedImage(
        boost::shared_ptr<GridData> tlGridData,
        const TLVisualizationOptions& options) {
        
        // 直接使用VisualizationEngine的优化方法
        size_t width = tlGridData->getColumns();
        size_t height = tlGridData->getRows();
        
        // 1. 提取数据为float数组
        std::vector<float> tlValues(width * height);
        const float* srcData = tlGridData->getData<float>();
        std::copy(srcData, srcData + width * height, tlValues.begin());
        
        // 2. 计算数据范围
        auto [minVal, maxVal] = std::minmax_element(
            tlValues.begin(), tlValues.end());
        
        // 3. SIMD/GPU加速的颜色映射
        std::vector<uint32_t> rgbaData;
        if (vizEngine_->isGPUOptimizationEnabled()) {
            // GPU加速路径
            rgbaData = mapDataToColorsGPU(
                tlValues, options.colorMap, *minVal, *maxVal);
        } else if (vizEngine_->isSIMDOptimizationEnabled()) {
            // SIMD加速路径
            rgbaData = vizEngine_->mapDataToColorsSIMD(
                tlValues, options.colorMap, *minVal, *maxVal);
        } else {
            // 标量路径
            std::vector<double> doubleValues(tlValues.begin(), tlValues.end());
            rgbaData = vizEngine_->mapDataToColors(
                doubleValues, options.colorMap, *minVal, *maxVal);
        }
        
        // 4. 添加等值线（如果需要）
        if (options.drawContours) {
            drawContours(rgbaData, width, height, tlValues, options);
        }
        
        // 5. PNG编码（内存中）
        return encodePNGInMemory(rgbaData, width, height);
    }
};
```

### 3.3 压缩和编码优化

```cpp
/**
 * @brief PNG编码器（内存操作）
 */
class MemoryPNGEncoder {
private:
    // libpng写入回调
    static void pngWriteCallback(png_structp png_ptr, 
                                png_bytep data, 
                                png_size_t length) {
        auto* output = static_cast<std::vector<unsigned char>*>(
            png_get_io_ptr(png_ptr));
        output->insert(output->end(), data, data + length);
    }
    
public:
    /**
     * @brief 将RGBA数据编码为PNG格式
     * @param rgbaData RGBA像素数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param compressionLevel 压缩级别（0-9）
     * @return PNG格式的二进制数据
     */
    static std::vector<unsigned char> encode(
        const std::vector<uint32_t>& rgbaData,
        int width, int height,
        int compressionLevel = 6) {
        
        std::vector<unsigned char> pngData;
        
        // 初始化libpng
        png_structp png_ptr = png_create_write_struct(
            PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        png_infop info_ptr = png_create_info_struct(png_ptr);
        
        // 设置错误处理
        if (setjmp(png_jmpbuf(png_ptr))) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            throw std::runtime_error("PNG encoding failed");
        }
        
        // 设置自定义写入函数
        png_set_write_fn(png_ptr, &pngData, pngWriteCallback, nullptr);
        
        // 设置图像信息
        png_set_IHDR(png_ptr, info_ptr, width, height,
                     8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        
        // 设置压缩级别
        png_set_compression_level(png_ptr, compressionLevel);
        
        // 写入头信息
        png_write_info(png_ptr, info_ptr);
        
        // 准备行指针
        std::vector<png_bytep> row_pointers(height);
        for (int y = 0; y < height; ++y) {
            row_pointers[y] = reinterpret_cast<png_bytep>(
                const_cast<uint32_t*>(&rgbaData[y * width]));
        }
        
        // 写入图像数据
        png_write_image(png_ptr, row_pointers.data());
        png_write_end(png_ptr, nullptr);
        
        // 清理
        png_destroy_write_struct(&png_ptr, &info_ptr);
        
        return pngData;
    }
    
    /**
     * @brief 使用快速压缩算法（如LZ4）进一步压缩
     */
    static std::vector<unsigned char> compressWithLZ4(
        const std::vector<unsigned char>& pngData) {
        
        // 分配压缩缓冲区
        int maxCompressedSize = LZ4_compressBound(pngData.size());
        std::vector<unsigned char> compressed(maxCompressedSize + 4);
        
        // 存储原始大小（用于解压）
        *reinterpret_cast<uint32_t*>(compressed.data()) = pngData.size();
        
        // 压缩
        int compressedSize = LZ4_compress_default(
            reinterpret_cast<const char*>(pngData.data()),
            reinterpret_cast<char*>(compressed.data() + 4),
            pngData.size(),
            maxCompressedSize
        );
        
        compressed.resize(compressedSize + 4);
        return compressed;
    }
};
```

### 3.4 Web传输优化

```cpp
/**
 * @brief Web传输管理器
 */
class WebTransmissionManager {
private:
    // WebSocket连接池
    std::map<std::string, WebSocketConnection> connections_;
    
public:
    /**
     * @brief 通过WebSocket发送图像流
     * @param clientId 客户端ID
     * @param imageData 图像二进制数据
     * @param metadata 图像元数据
     */
    void sendImageStream(
        const std::string& clientId,
        const std::vector<unsigned char>& imageData,
        const ImageMetadata& metadata) {
        
        // 构建消息头
        MessageHeader header;
        header.type = MessageType::IMAGE_STREAM;
        header.dataSize = imageData.size();
        header.timestamp = boost::posix_time::microsec_clock::universal_time();
        header.metadata = metadata;
        
        // 分块传输大图像
        if (imageData.size() > MAX_WEBSOCKET_FRAME_SIZE) {
            sendChunked(clientId, header, imageData);
        } else {
            sendDirect(clientId, header, imageData);
        }
    }
    
    /**
     * @brief 支持HTTP流式响应
     */
    void streamImageHTTP(
        HttpResponse& response,
        const std::vector<unsigned char>& imageData,
        const std::string& contentType = "image/png") {
        
        // 设置响应头
        response.setHeader("Content-Type", contentType);
        response.setHeader("Content-Length", std::to_string(imageData.size()));
        response.setHeader("Cache-Control", "no-cache");
        
        // 支持分块传输编码
        if (imageData.size() > HTTP_CHUNK_SIZE) {
            response.setHeader("Transfer-Encoding", "chunked");
            
            // 分块发送
            size_t offset = 0;
            while (offset < imageData.size()) {
                size_t chunkSize = std::min(
                    HTTP_CHUNK_SIZE, imageData.size() - offset);
                response.writeChunk(&imageData[offset], chunkSize);
                offset += chunkSize;
            }
            response.endChunkedResponse();
        } else {
            // 一次性发送
            response.write(imageData.data(), imageData.size());
        }
    }
    
    /**
     * @brief 生成Base64编码（用于嵌入JSON）
     */
    static std::string toBase64(const std::vector<unsigned char>& data) {
        using namespace boost::archive::iterators;
        using It = base64_from_binary<
            transform_width<std::vector<unsigned char>::const_iterator, 6, 8>>;
        
        std::string base64(It(data.begin()), It(data.end()));
        
        // 添加填充
        base64.append((3 - data.size() % 3) % 3, '=');
        
        return base64;
    }
};
```

### 3.5 前端显示实现

```javascript
/**
 * RAM传播损失可视化前端组件
 */
class TLVisualizationComponent {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);
        
        // WebSocket连接
        this.ws = null;
        this.imageBuffer = [];
    }
    
    /**
     * 连接到服务器
     */
    connect(wsUrl) {
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';
        
        this.ws.onmessage = (event) => {
            this.handleMessage(event.data);
        };
    }
    
    /**
     * 处理接收到的图像数据
     */
    handleMessage(data) {
        const view = new DataView(data);
        const header = this.parseHeader(view);
        
        if (header.type === 'IMAGE_STREAM') {
            if (header.isChunked) {
                // 处理分块数据
                this.imageBuffer.push(new Uint8Array(
                    data, header.headerSize));
                
                if (header.isFinalChunk) {
                    const fullImage = this.combineChunks();
                    this.displayImage(fullImage);
                    this.imageBuffer = [];
                }
            } else {
                // 直接显示
                const imageData = new Uint8Array(
                    data, header.headerSize);
                this.displayImage(imageData);
            }
        }
    }
    
    /**
     * 显示PNG图像
     */
    displayImage(pngData) {
        const blob = new Blob([pngData], {type: 'image/png'});
        const url = URL.createObjectURL(blob);
        
        const img = new Image();
        img.onload = () => {
            // 调整画布大小
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            
            // 绘制图像
            this.ctx.drawImage(img, 0, 0);
            
            // 清理
            URL.revokeObjectURL(url);
        };
        img.src = url;
    }
    
    /**
     * 使用WebGL加速显示（用于高频更新）
     */
    initWebGLDisplay() {
        const gl = this.canvas.getContext('webgl2');
        
        // 创建纹理
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        
        // 设置纹理参数
        gl.texParameteri(gl.TEXTURE_2D, 
            gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, 
            gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        
        // 着色器程序...
    }
}
```

## 4. 性能优化策略

### 4.1 计算优化
- **GPU加速**：使用GPU进行颜色映射和图像渲染
- **SIMD优化**：使用AVX2/AVX512加速数据处理
- **并行处理**：多线程处理不同深度切片

### 4.2 内存优化
- **零拷贝**：直接在GridData上操作，避免数据复制
- **内存池**：预分配图像缓冲区
- **流式处理**：大数据集分块处理

### 4.3 传输优化
- **压缩**：PNG压缩 + LZ4二次压缩
- **差分编码**：只传输变化的区域
- **多分辨率**：根据缩放级别传输不同分辨率

### 4.4 缓存策略
```cpp
class TLImageCache {
private:
    struct CacheEntry {
        std::vector<unsigned char> imageData;
        boost::posix_time::ptime timestamp;
        size_t accessCount;
    };
    
    std::map<std::string, CacheEntry> cache_;
    size_t maxCacheSize_ = 100 * 1024 * 1024; // 100MB
    
public:
    /**
     * @brief 智能缓存策略
     */
    boost::optional<std::vector<unsigned char>> get(
        const std::string& key) {
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // 更新访问计数和时间
            it->second.accessCount++;
            it->second.timestamp = 
                boost::posix_time::microsec_clock::local_time();
            return it->second.imageData;
        }
        return boost::none;
    }
    
    void put(const std::string& key, 
             const std::vector<unsigned char>& data) {
        // LRU淘汰策略
        ensureCapacity(data.size());
        
        CacheEntry entry;
        entry.imageData = data;
        entry.timestamp = boost::posix_time::microsec_clock::local_time();
        entry.accessCount = 1;
        
        cache_[key] = std::move(entry);
    }
};
```

## 5. 使用示例

```cpp
// 完整的使用流程
class RamVisualizationWorkflow {
public:
    boost::future<void> visualizeRAMResults(
        const RamPeMultiAzimuthResult& ramResult,
        const std::string& clientId) {
        
        // 1. 转换RAM结果到GridData
        auto tlGrid = RamResultToGridDataConverter::convertTLField(
            ramResult, 10);  // 选择第10层深度
        
        // 2. 设置可视化选项
        TLVisualizationOptions vizOptions;
        vizOptions.colorMap = "jet";
        vizOptions.drawContours = true;
        vizOptions.contourLevels = 10;
        vizOptions.compressionLevel = 6;
        
        // 3. 生成图像流
        return vizManager_->generateTLImageWithLegend(tlGrid, vizOptions)
            .then([this, clientId](std::vector<unsigned char> imageData) {
                // 4. 发送到前端
                ImageMetadata metadata;
                metadata.width = 800;
                metadata.height = 600;
                metadata.format = "png";
                metadata.timestamp = 
                    boost::posix_time::microsec_clock::universal_time();
                
                webTransmitter_->sendImageStream(
                    clientId, imageData, metadata);
            });
    }
};
```

## 6. 总结

本方案实现了从RAM传播损失计算结果到前端可视化的完整流程：

1. **高效数据转换**：直接从RAM结果构建GridData，避免中间文件
2. **内存图像生成**：使用Output模块的streamOutput功能，在内存中生成PNG
3. **性能优化**：支持SIMD/GPU加速，智能缓存，压缩传输
4. **灵活传输**：支持WebSocket/HTTP，分块传输，Base64编码
5. **前端显示**：Canvas/WebGL渲染，支持实时更新

该方案充分利用了OSCEAN现有的基础设施，实现了高性能、低延迟的传播损失可视化。 