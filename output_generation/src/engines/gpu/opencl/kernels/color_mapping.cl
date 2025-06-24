/**
 * @file color_mapping.cl
 * @brief OpenCL颜色映射内核实现
 */

// 颜色查找表（需要在主机端设置）
__constant float4 colorLUT[256];

/**
 * @brief 基础颜色映射内核
 * @param input 输入数据
 * @param output 输出RGBA数据
 * @param width 数据宽度
 * @param height 数据高度
 * @param minValue 最小值
 * @param maxValue 最大值
 */
__kernel void colorMappingKernel(
    __global const float* input,
    __global uchar4* output,
    const int width,
    const int height,
    const float minValue,
    const float maxValue) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = input[idx];
    
    // 处理NaN
    if (isnan(value)) {
        output[idx] = (uchar4)(0, 0, 0, 0);
        return;
    }
    
    // 归一化到[0, 1]
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
    
    // 映射到LUT索引
    float lutIndex = normalizedValue * 255.0f;
    int index0 = (int)floor(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    // 线性插值颜色
    float4 color0 = colorLUT[index0];
    float4 color1 = colorLUT[index1];
    float4 color = mix(color0, color1, frac);
    
    // 转换为8位颜色
    output[idx] = (uchar4)(
        (uchar)(color.x * 255.0f),
        (uchar)(color.y * 255.0f),
        (uchar)(color.z * 255.0f),
        (uchar)(color.w * 255.0f)
    );
}

/**
 * @brief 带透明度调整的颜色映射
 */
__kernel void colorMappingWithAlphaKernel(
    __global const float* input,
    __global uchar4* output,
    const int width,
    const int height,
    const float minValue,
    const float maxValue,
    const float alphaScale) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = input[idx];
    
    if (isnan(value)) {
        output[idx] = (uchar4)(0, 0, 0, 0);
        return;
    }
    
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
    
    float lutIndex = normalizedValue * 255.0f;
    int index0 = (int)floor(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    float4 color0 = colorLUT[index0];
    float4 color1 = colorLUT[index1];
    float4 color = mix(color0, color1, frac);
    
    // 应用透明度缩放
    color.w *= alphaScale;
    
    output[idx] = (uchar4)(
        (uchar)(color.x * 255.0f),
        (uchar)(color.y * 255.0f),
        (uchar)(color.z * 255.0f),
        (uchar)(color.w * 255.0f)
    );
}

/**
 * @brief 批量颜色映射（处理多个数据集）
 */
__kernel void batchColorMappingKernel(
    __global const float* inputs,
    __global uchar4* outputs,
    __global const float2* valueRanges,  // (min, max) pairs
    const int width,
    const int height,
    const int batchSize) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int b = get_global_id(2);
    
    if (x >= width || y >= height || b >= batchSize) return;
    
    int pixelIdx = y * width + x;
    int batchOffset = b * width * height;
    int idx = batchOffset + pixelIdx;
    
    float value = inputs[idx];
    float2 range = valueRanges[b];
    
    if (isnan(value)) {
        outputs[idx] = (uchar4)(0, 0, 0, 0);
        return;
    }
    
    float normalizedValue = (value - range.x) / (range.y - range.x);
    normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
    
    float lutIndex = normalizedValue * 255.0f;
    int index0 = (int)floor(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    float4 color0 = colorLUT[index0];
    float4 color1 = colorLUT[index1];
    float4 color = mix(color0, color1, frac);
    
    outputs[idx] = (uchar4)(
        (uchar)(color.x * 255.0f),
        (uchar)(color.y * 255.0f),
        (uchar)(color.z * 255.0f),
        (uchar)(color.w * 255.0f)
    );
}

/**
 * @brief Min/Max reduction内核
 */
__kernel void minMaxReductionKernel(
    __global const float* input,
    __global float2* output,  // (min, max) pairs
    __local float* sharedMin,
    __local float* sharedMax,
    const int size) {
    
    int tid = get_local_id(0);
    int gid = get_global_id(0);
    int blockSize = get_local_size(0);
    
    // 初始化共享内存
    sharedMin[tid] = FLT_MAX;
    sharedMax[tid] = -FLT_MAX;
    
    // 每个线程处理多个元素
    for (int i = gid; i < size; i += get_global_size(0)) {
        float val = input[i];
        if (!isnan(val)) {
            sharedMin[tid] = min(sharedMin[tid], val);
            sharedMax[tid] = max(sharedMax[tid], val);
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 块内规约
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMin[tid] = min(sharedMin[tid], sharedMin[tid + s]);
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // 写入全局结果
    if (tid == 0) {
        int blockIdx = get_group_id(0);
        output[blockIdx] = (float2)(sharedMin[0], sharedMax[0]);
    }
}

/**
 * @brief 高级颜色映射内核（支持数据变换和伽马校正）
 */
__kernel void advancedColorMapping(
    __global const float* input,
    __global uchar4* output,
    int width,
    int height,
    float minValue,
    float maxValue,
    int transformType,
    float transformParam,
    float gamma,
    uint nanColor) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = input[idx];
    
    // 处理NaN值
    if (isnan(value)) {
        output[idx] = (uchar4)(
            (nanColor >> 0) & 0xFF,
            (nanColor >> 8) & 0xFF,
            (nanColor >> 16) & 0xFF,
            (nanColor >> 24) & 0xFF
        );
        return;
    }
    
    // 应用数据变换
    value = applyTransform(value, transformType, transformParam);
    
    // 归一化
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
    
    // 应用伽马校正
    if (gamma != 1.0f) {
        normalizedValue = pow(normalizedValue, gamma);
    }
    
    // 映射到LUT索引
    float lutIndex = normalizedValue * 255.0f;
    int index0 = (int)floor(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    // 从LUT中获取颜色并插值
    float4 color0 = colorLUT[index0];
    float4 color1 = colorLUT[index1];
    
    float r = lerp(color0.x, color1.x, frac);
    float g = lerp(color0.y, color1.y, frac);
    float b = lerp(color0.z, color1.z, frac);
    float a = lerp(color0.w, color1.w, frac);
    
    // 转换为uchar4并写入输出
    output[idx] = (uchar4)(
        (uchar)(r * 255.0f),
        (uchar)(g * 255.0f),
        (uchar)(b * 255.0f),
        (uchar)(a * 255.0f)
    );
}

/**
 * @brief 带抖动的颜色映射（提高视觉质量）
 */
__kernel void colorMappingWithDithering(
    __global const float* input,
    __global uchar4* output,
    int width,
    int height,
    float minValue,
    float maxValue,
    uint seed) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = input[idx];
    
    // 处理NaN值
    if (isnan(value)) {
        output[idx] = (uchar4)(0, 0, 0, 0);
        return;
    }
    
    // 归一化
    float normalizedValue = (value - minValue) / (maxValue - minValue);
    normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
    
    // 添加抖动噪声
    uint hash = (x * 73856093u) ^ (y * 19349663u) ^ seed;
    float noise = (float)(hash & 0xFFFF) / 65535.0f - 0.5f;
    normalizedValue += noise * 0.01f; // 1%的抖动
    normalizedValue = clamp(normalizedValue, 0.0f, 1.0f);
    
    // 映射到LUT索引
    float lutIndex = normalizedValue * 255.0f;
    int index0 = (int)floor(lutIndex);
    int index1 = min(index0 + 1, 255);
    float frac = lutIndex - index0;
    
    // 从LUT中获取颜色并插值
    float4 color0 = colorLUT[index0];
    float4 color1 = colorLUT[index1];
    
    float r = lerp(color0.x, color1.x, frac);
    float g = lerp(color0.y, color1.y, frac);
    float b = lerp(color0.z, color1.z, frac);
    float a = lerp(color0.w, color1.w, frac);
    
    // 转换为uchar4并写入输出
    output[idx] = (uchar4)(
        (uchar)(r * 255.0f),
        (uchar)(g * 255.0f),
        (uchar)(b * 255.0f),
        (uchar)(a * 255.0f)
    );
} 