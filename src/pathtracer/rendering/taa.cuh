// taa.cuh
// Temporal Anti-Aliasing support for real-time path tracing
// Uses Halton low-discrepancy sequences for optimal sub-pixel jitter
// Particularly effective for thin geometry like ropes, wires, and hair

#ifndef TAA_CUH
#define TAA_CUH

#include <cuda_runtime.h>

// Number of unique jitter positions before the sequence repeats.
// 16 is a good balance: enough variety for convergence, small enough
// to avoid floating point precision issues in the Halton sequence.
// For more aggressive AA, increase to 64 or 256.
constexpr int TAA_SEQUENCE_LENGTH = 16;

// Precomputed Halton sequence (base 2, base 3) for first 16 samples
// These provide excellent low-discrepancy coverage of the unit square
__constant__ float2 d_halton_sequence[TAA_SEQUENCE_LENGTH] = {
    {0.500000f, 0.333333f},  // 0
    {0.250000f, 0.666667f},  // 1
    {0.750000f, 0.111111f},  // 2
    {0.125000f, 0.444444f},  // 3
    {0.625000f, 0.777778f},  // 4
    {0.375000f, 0.222222f},  // 5
    {0.875000f, 0.555556f},  // 6
    {0.062500f, 0.888889f},  // 7
    {0.562500f, 0.037037f},  // 8
    {0.312500f, 0.370370f},  // 9
    {0.812500f, 0.703704f},  // 10
    {0.187500f, 0.148148f},  // 11
    {0.687500f, 0.481481f},  // 12
    {0.437500f, 0.814815f},  // 13
    {0.937500f, 0.259259f},  // 14
    {0.062500f, 0.592593f},  // 15
};

// Get TAA jitter offset for a given frame
// Returns sub-pixel offset in range [-0.5, 0.5] for both x and y
// This centers the jitter around the pixel center
__device__ __host__ __forceinline__ float2 getTAAJitter(int frame_index) {
    int idx = frame_index % TAA_SEQUENCE_LENGTH;
#ifdef __CUDA_ARCH__
    float2 h = d_halton_sequence[idx];
#else
    // Host-side fallback (for CPU validation)
    const float2 halton_host[TAA_SEQUENCE_LENGTH] = {
        {0.500000f, 0.333333f}, {0.250000f, 0.666667f},
        {0.750000f, 0.111111f}, {0.125000f, 0.444444f},
        {0.625000f, 0.777778f}, {0.375000f, 0.222222f},
        {0.875000f, 0.555556f}, {0.062500f, 0.888889f},
        {0.562500f, 0.037037f}, {0.312500f, 0.370370f},
        {0.812500f, 0.703704f}, {0.187500f, 0.148148f},
        {0.687500f, 0.481481f}, {0.437500f, 0.814815f},
        {0.937500f, 0.259259f}, {0.062500f, 0.592593f},
    };
    float2 h = halton_host[idx];
#endif
    // Center the jitter around 0 (range becomes [-0.5, 0.5])
    return make_float2(h.x - 0.5f, h.y - 0.5f);
}

// Get TAA jitter for camera ray generation
// width/height are screen dimensions
// Returns the jitter in normalized device coordinates (NDC)
__device__ __host__ __forceinline__ float2 getTAAJitterNDC(int frame_index,
                                                            int width,
                                                            int height) {
    float2 pixel_jitter = getTAAJitter(frame_index);
    // Convert to NDC space
    return make_float2(pixel_jitter.x / (float)width,
                       pixel_jitter.y / (float)height);
}

// Compute Halton sequence value at runtime (for extended sequences)
// base should be a prime number (2 or 3 typically)
__device__ __host__ __forceinline__ float halton(int index, int base) {
    float result = 0.0f;
    float f = 1.0f / (float)base;
    int i = index;
    while (i > 0) {
        result += f * (float)(i % base);
        i = i / base;
        f = f / (float)base;
    }
    return result;
}

// Get extended Halton jitter (for sequences longer than 16)
// Use this if you want more than 16 unique samples
__device__ __host__ __forceinline__ float2 getTAAJitterExtended(int frame_index) {
    float x = halton(frame_index + 1, 2); // +1 to avoid (0,0) at index 0
    float y = halton(frame_index + 1, 3);
    return make_float2(x - 0.5f, y - 0.5f);
}

// R2 sequence - alternative to Halton, sometimes better distribution
// Based on the plastic constant (generalized golden ratio for 2D)
__device__ __host__ __forceinline__ float2 getR2Jitter(int frame_index) {
    // Plastic constant for 2D: solutions to x^3 = x + 1
    const float g = 1.32471795724f;
    const float a1 = 1.0f / g;
    const float a2 = 1.0f / (g * g);
    
    float x = fmodf(0.5f + a1 * frame_index, 1.0f);
    float y = fmodf(0.5f + a2 * frame_index, 1.0f);
    
    return make_float2(x - 0.5f, y - 0.5f);
}

#endif // TAA_CUH
