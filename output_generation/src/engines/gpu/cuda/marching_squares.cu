/**
 * @file marching_squares.cu
 * @brief Marching Squares GPU implementation for contour generation
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>

namespace oscean {
namespace output_generation {
namespace gpu {
namespace cuda {

// Constants
#define BLOCK_SIZE 16
#define MAX_CONTOURS_PER_CELL 2

/**
 * @brief Marching Squares lookup table
 * 16 possible configurations, each defines line segment endpoints
 */
__constant__ int edgeTable[16] = {
    0x0,   // 0000
    0x9,   // 0001
    0x3,   // 0010
    0xa,   // 0011
    0x6,   // 0100
    0xf,   // 0101
    0x5,   // 0110
    0xc,   // 0111
    0xc,   // 1000
    0x5,   // 1001
    0xf,   // 1010
    0x6,   // 1011
    0xa,   // 1100
    0x3,   // 1101
    0x9,   // 1110
    0x0    // 1111
};

/**
 * @brief Line endpoints lookup table
 * For each configuration, defines line segment start and end points
 */
__constant__ int lineTable[16][4] = {
    {-1, -1, -1, -1},  // 0
    { 0,  3, -1, -1},  // 1
    { 0,  1, -1, -1},  // 2
    { 1,  3, -1, -1},  // 3
    { 1,  2, -1, -1},  // 4
    { 0,  3,  1,  2},  // 5
    { 0,  2, -1, -1},  // 6
    { 2,  3, -1, -1},  // 7
    { 2,  3, -1, -1},  // 8
    { 0,  2, -1, -1},  // 9
    { 0,  3,  1,  2},  // 10
    { 1,  2, -1, -1},  // 11
    { 1,  3, -1, -1},  // 12
    { 0,  1, -1, -1},  // 13
    { 0,  3, -1, -1},  // 14
    {-1, -1, -1, -1}   // 15
};

/**
 * @brief Contour vertex structure
 */
struct ContourVertex {
    float x, y;
    float value;
};

/**
 * @brief Contour segment structure
 */
struct ContourSegment {
    ContourVertex start;
    ContourVertex end;
    int cellX, cellY;
};

// Forward declaration
__global__ void convertSegmentsToPoints(
    const ContourSegment* segments,
    float* points,
    int numSegments);

/**
 * @brief Calculate linear interpolation position
 */
__device__ float interpolate(float val1, float val2, float isoValue) {
    if (fabsf(val2 - val1) < 1e-6f) {
        return 0.5f;
    }
    return (isoValue - val1) / (val2 - val1);
}

/**
 * @brief Get edge vertex position
 */
__device__ void getEdgeVertex(
    int edge, float x, float y, float cellSize,
    float val0, float val1, float val2, float val3,
    float isoValue, float& vx, float& vy) {
    
    float t;
    switch (edge) {
        case 0:  // Bottom edge
            t = interpolate(val0, val1, isoValue);
            vx = x + t * cellSize;
            vy = y;
            break;
        case 1:  // Right edge
            t = interpolate(val1, val2, isoValue);
            vx = x + cellSize;
            vy = y + t * cellSize;
            break;
        case 2:  // Top edge
            t = interpolate(val3, val2, isoValue);
            vx = x + (1.0f - t) * cellSize;
            vy = y + cellSize;
            break;
        case 3:  // Left edge
            t = interpolate(val0, val3, isoValue);
            vx = x;
            vy = y + t * cellSize;
            break;
    }
}

/**
 * @brief Marching Squares kernel
 */
__global__ void marchingSquaresKernel(
    const float* gridData,
    ContourSegment* segments,
    int* segmentCount,
    int gridWidth, int gridHeight,
    float cellSize, float isoValue,
    float originX, float originY,
    int maxSegments) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (x >= gridWidth - 1 || y >= gridHeight - 1) return;
    
    // Get cell corner values
    int idx00 = y * gridWidth + x;
    int idx10 = y * gridWidth + (x + 1);
    int idx01 = (y + 1) * gridWidth + x;
    int idx11 = (y + 1) * gridWidth + (x + 1);
    
    float val0 = gridData[idx00];  // Bottom left
    float val1 = gridData[idx10];  // Bottom right
    float val2 = gridData[idx11];  // Top right
    float val3 = gridData[idx01];  // Top left
    
    // Check NaN
    if (isnan(val0) || isnan(val1) || isnan(val2) || isnan(val3)) {
        return;
    }
    
    // Calculate configuration index
    int config = 0;
    if (val0 >= isoValue) config |= 1;
    if (val1 >= isoValue) config |= 2;
    if (val2 >= isoValue) config |= 4;
    if (val3 >= isoValue) config |= 8;
    
    // Generate line segments based on configuration
    if (edgeTable[config] != 0) {
        // Calculate cell world coordinates
        float cellX = originX + x * cellSize;
        float cellY = originY + y * cellSize;
        
        // Generate segments
        for (int i = 0; i < 4; i += 2) {
            int edge1 = lineTable[config][i];
            int edge2 = lineTable[config][i + 1];
            
            if (edge1 == -1) break;
            
            // Get segment endpoints
            float x1, y1, x2, y2;
            getEdgeVertex(edge1, cellX, cellY, cellSize, 
                         val0, val1, val2, val3, isoValue, x1, y1);
            getEdgeVertex(edge2, cellX, cellY, cellSize, 
                         val0, val1, val2, val3, isoValue, x2, y2);
            
            // Atomically increment counter and get index
            int segIdx = atomicAdd(segmentCount, 1);
            
            if (segIdx < maxSegments) {
                ContourSegment& seg = segments[segIdx];
                seg.start.x = x1;
                seg.start.y = y1;
                seg.start.value = isoValue;
                seg.end.x = x2;
                seg.end.y = y2;
                seg.end.value = isoValue;
                seg.cellX = x;
                seg.cellY = y;
            }
        }
    }
}

/**
 * @brief Multi-contour generation kernel
 */
__global__ void multiContourKernel(
    const float* gridData,
    ContourSegment* segments,
    int* segmentCounts,
    int gridWidth, int gridHeight,
    float cellSize,
    const float* isoValues, int numIsoValues,
    float originX, float originY,
    int maxSegmentsPerLevel) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int levelIdx = blockIdx.z;
    
    if (x >= gridWidth - 1 || y >= gridHeight - 1 || levelIdx >= numIsoValues) {
        return;
    }
    
    float isoValue = isoValues[levelIdx];
    int* levelSegmentCount = &segmentCounts[levelIdx];
    ContourSegment* levelSegments = &segments[levelIdx * maxSegmentsPerLevel];
    
    // Get cell corner values
    int idx00 = y * gridWidth + x;
    int idx10 = y * gridWidth + (x + 1);
    int idx01 = (y + 1) * gridWidth + x;
    int idx11 = (y + 1) * gridWidth + (x + 1);
    
    float val0 = gridData[idx00];
    float val1 = gridData[idx10];
    float val2 = gridData[idx11];
    float val3 = gridData[idx01];
    
    // Check NaN
    if (isnan(val0) || isnan(val1) || isnan(val2) || isnan(val3)) {
        return;
    }
    
    // Calculate configuration index
    int config = 0;
    if (val0 >= isoValue) config |= 1;
    if (val1 >= isoValue) config |= 2;
    if (val2 >= isoValue) config |= 4;
    if (val3 >= isoValue) config |= 8;
    
    // Generate line segments based on configuration
    if (edgeTable[config] != 0) {
        float cellX = originX + x * cellSize;
        float cellY = originY + y * cellSize;
        
        for (int i = 0; i < 4; i += 2) {
            int edge1 = lineTable[config][i];
            int edge2 = lineTable[config][i + 1];
            
            if (edge1 == -1) break;
            
            float x1, y1, x2, y2;
            getEdgeVertex(edge1, cellX, cellY, cellSize, 
                         val0, val1, val2, val3, isoValue, x1, y1);
            getEdgeVertex(edge2, cellX, cellY, cellSize, 
                         val0, val1, val2, val3, isoValue, x2, y2);
            
            int segIdx = atomicAdd(levelSegmentCount, 1);
            
            if (segIdx < maxSegmentsPerLevel) {
                ContourSegment& seg = levelSegments[segIdx];
                seg.start.x = x1;
                seg.start.y = y1;
                seg.start.value = isoValue;
                seg.end.x = x2;
                seg.end.y = y2;
                seg.end.value = isoValue;
                seg.cellX = x;
                seg.cellY = y;
            }
        }
    }
}

/**
 * @brief Contour smoothing kernel (using Chaikin algorithm)
 */
__global__ void smoothContourKernel(
    const ContourSegment* inputSegments,
    ContourSegment* outputSegments,
    int segmentCount,
    float smoothingFactor) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= segmentCount) return;
    
    const ContourSegment& seg = inputSegments[idx];
    ContourSegment& outSeg = outputSegments[idx];
    
    // Chaikin smoothing algorithm
    float t = smoothingFactor;  // Usually 0.25
    
    outSeg.start.x = (1.0f - t) * seg.start.x + t * seg.end.x;
    outSeg.start.y = (1.0f - t) * seg.start.y + t * seg.end.y;
    outSeg.start.value = seg.start.value;
    
    outSeg.end.x = t * seg.start.x + (1.0f - t) * seg.end.x;
    outSeg.end.y = t * seg.start.y + (1.0f - t) * seg.end.y;
    outSeg.end.value = seg.end.value;
    
    outSeg.cellX = seg.cellX;
    outSeg.cellY = seg.cellY;
}

// C interface functions
extern "C" {

/**
 * @brief Generate single contour
 */
cudaError_t generateContourGPU(
    const float* d_gridData,
    ContourSegment* d_segments,
    int* d_segmentCount,
    int gridWidth, int gridHeight,
    float cellSize, float isoValue,
    float originX, float originY,
    int maxSegments,
    cudaStream_t stream) {
    
    // Calculate grid size
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (gridWidth + blockSize.x - 1) / blockSize.x,
        (gridHeight + blockSize.y - 1) / blockSize.y
    );
    
    // Clear counter
    cudaMemsetAsync(d_segmentCount, 0, sizeof(int), stream);
    
    // Launch kernel
    marchingSquaresKernel<<<gridSize, blockSize, 0, stream>>>(
        d_gridData, d_segments, d_segmentCount,
        gridWidth, gridHeight,
        cellSize, isoValue,
        originX, originY,
        maxSegments
    );
    
    return cudaGetLastError();
}

/**
 * @brief Generate multiple contours
 */
cudaError_t generateMultiContoursGPU(
    const float* d_gridData,
    ContourSegment* d_segments,
    int* d_segmentCounts,
    int gridWidth, int gridHeight,
    float cellSize,
    const float* d_isoValues, int numIsoValues,
    float originX, float originY,
    int maxSegmentsPerLevel,
    cudaStream_t stream) {
    
    // Calculate grid size
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize(
        (gridWidth + blockSize.x - 1) / blockSize.x,
        (gridHeight + blockSize.y - 1) / blockSize.y,
        numIsoValues
    );
    
    // Clear counters
    cudaMemsetAsync(d_segmentCounts, 0, numIsoValues * sizeof(int), stream);
    
    // Launch kernel
    multiContourKernel<<<gridSize, blockSize, 0, stream>>>(
        d_gridData, d_segments, d_segmentCounts,
        gridWidth, gridHeight,
        cellSize,
        d_isoValues, numIsoValues,
        originX, originY,
        maxSegmentsPerLevel
    );
    
    return cudaGetLastError();
}

/**
 * @brief Smooth contours
 */
cudaError_t smoothContoursGPU(
    const ContourSegment* d_inputSegments,
    ContourSegment* d_outputSegments,
    int segmentCount,
    float smoothingFactor,
    cudaStream_t stream) {
    
    if (segmentCount == 0) return cudaSuccess;
    
    // Calculate grid size
    int blockSize = 256;
    int gridSize = (segmentCount + blockSize - 1) / blockSize;
    
    // Launch kernel
    smoothContourKernel<<<gridSize, blockSize, 0, stream>>>(
        d_inputSegments, d_outputSegments,
        segmentCount, smoothingFactor
    );
    
    return cudaGetLastError();
}

/**
 * @brief Generate multiple contours
 */
cudaError_t generateContoursGPU(
    const float* d_gridData,
    int width, int height,
    const float* levels, int numLevels,
    float** d_contourPoints,
    int* numContourPoints,
    cudaStream_t stream) {
    
    if (numLevels == 0 || !d_gridData || !levels) {
        *numContourPoints = 0;
        *d_contourPoints = nullptr;
        return cudaSuccess;
    }
    
    // Allocate device memory for contour levels
    float* d_levels;
    cudaMalloc(&d_levels, numLevels * sizeof(float));
    cudaMemcpyAsync(d_levels, levels, numLevels * sizeof(float), 
                    cudaMemcpyHostToDevice, stream);
    
    // Estimate maximum possible segments
    int maxSegmentsPerLevel = width * height * 2; // Max 2 segments per cell
    int totalMaxSegments = maxSegmentsPerLevel * numLevels;
    
    // Allocate device memory
    ContourSegment* d_segments;
    int* d_segmentCounts;
    cudaMalloc(&d_segments, totalMaxSegments * sizeof(ContourSegment));
    cudaMalloc(&d_segmentCounts, numLevels * sizeof(int));
    cudaMemsetAsync(d_segmentCounts, 0, numLevels * sizeof(int), stream);
    
    // Call multi-contour generation kernel
    cudaError_t err = generateMultiContoursGPU(
        d_gridData, d_segments, d_segmentCounts,
        width, height,
        1.0f,  // Assume cell size is 1
        d_levels, numLevels,
        0.0f, 0.0f,  // Origin coordinates
        maxSegmentsPerLevel,
        stream
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_levels);
        cudaFree(d_segments);
        cudaFree(d_segmentCounts);
        return err;
    }
    
    // Get total segment count
    int* h_segmentCounts = new int[numLevels];
    cudaMemcpyAsync(h_segmentCounts, d_segmentCounts, 
                    numLevels * sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    int totalSegments = 0;
    for (int i = 0; i < numLevels; ++i) {
        totalSegments += h_segmentCounts[i];
    }
    
    if (totalSegments == 0) {
        *numContourPoints = 0;
        *d_contourPoints = nullptr;
        delete[] h_segmentCounts;
        cudaFree(d_levels);
        cudaFree(d_segments);
        cudaFree(d_segmentCounts);
        return cudaSuccess;
    }
    
    // Allocate output memory (2 points per segment, 3 coords per point)
    *numContourPoints = totalSegments * 2;
    cudaMalloc(d_contourPoints, totalSegments * 2 * 3 * sizeof(float));
    
    // Convert segments to points array
    dim3 blockSize(256);
    dim3 gridSize((totalSegments + blockSize.x - 1) / blockSize.x);
    
    // Launch conversion kernel
    convertSegmentsToPoints<<<gridSize, blockSize, 0, stream>>>(
        d_segments, *d_contourPoints, totalSegments
    );
    
    // Clean up temporary memory
    delete[] h_segmentCounts;
    cudaFree(d_levels);
    cudaFree(d_segments);
    cudaFree(d_segmentCounts);
    
    return cudaGetLastError();
}

} // extern "C"

/**
 * @brief Convert segments to points array
 */
__global__ void convertSegmentsToPoints(
    const ContourSegment* segments,
    float* points,
    int numSegments) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments) return;
    
    const ContourSegment& seg = segments[idx];
    int baseIdx = idx * 6;  // 6 values per segment (2 points, 3 coords each)
    
    // Start point
    points[baseIdx + 0] = seg.start.x;
    points[baseIdx + 1] = seg.start.y;
    points[baseIdx + 2] = seg.start.value;
    
    // End point
    points[baseIdx + 3] = seg.end.x;
    points[baseIdx + 4] = seg.end.y;
    points[baseIdx + 5] = seg.end.value;
}

} // namespace cuda
} // namespace gpu
} // namespace output_generation
} // namespace oscean 