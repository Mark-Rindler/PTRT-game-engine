// mathutils.cuh
// lightweight math utilities shared across host and device code
// provides constants scalar helpers random helpers and small bit packing
// utilities

#ifndef MATH_UTILS_CUH
#define MATH_UTILS_CUH

#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f
#define PI_OVER_TWO 1.57079632679489661923f
#define PI_OVER_FOUR 0.78539816339744830961f
#define INV_PI 0.31830988618379067154f
#define INV_TWO_PI 0.15915494309189533577f
#define EPSILON 1e-6f

#define DEG_TO_RAD 0.01745329251994329577f // PI / 180
#define RAD_TO_DEG 57.2957795130823208768f // 180 / PI

// converts an angle in degrees to radians
__host__ __device__ __forceinline__ float degrees_to_radians(float degrees) {
    return degrees * DEG_TO_RAD;
}

// converts an angle in radians to degrees
__host__ __device__ __forceinline__ float radians_to_degrees(float radians) {
    return radians * RAD_TO_DEG;
}

// clamps a value to the inclusive range min to max
__host__ __device__ __forceinline__ float clamp(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

// clamps a value to the inclusive range min to max
__host__ __device__ __forceinline__ int clamp(int x, int min, int max) {
    return x < min ? min : (x > max ? max : x);
}

// linearly interpolates between a and b using t
__host__ __device__ __forceinline__ float lerp(float a, float b, float t) {
    return fmaf(t, b - a, a);
}

// computes smoothstep interpolation for x between edge0 and edge1
__host__ __device__ __forceinline__ float smoothstep(float edge0, float edge1,
                                                     float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// computes smootherstep interpolation for x between edge0 and edge1
__host__ __device__ __forceinline__ float smootherstep(float edge0, float edge1,
                                                       float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// returns a uniform random float in the requested range using curand
__device__ __forceinline__ float random_float(curandState *state) {
    return curand_uniform(state);
}

// returns a uniform random float in the requested range using curand
__device__ __forceinline__ float random_float(curandState *state, float min,
                                              float max) {
    return fmaf(max - min, curand_uniform(state), min);
}

// returns a uniform random integer in the inclusive range min to max using
// curand
__device__ __forceinline__ int random_int(curandState *state, int min,
                                          int max) {
    int range = max - min + 1;
    float u = curand_uniform(state);
    int r = static_cast<int>(u * range);
    if (r >= range)
        r = range - 1;
    return min + r;
}

// initializes a curand state for a given thread id and seed
__device__ __forceinline__ void init_rand_state(curandState *state, int tid,
                                                int seed = 1984) {
    curand_init(seed, tid, 0, state);
}

// computes a^b using fast device intrinsics when available
__host__ __device__ __forceinline__ float fast_pow(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __powf(a, b);
#else
    return static_cast<float>(std::pow(a, b));
#endif
}

// computes exp(x) using fast device intrinsics when available
__host__ __device__ __forceinline__ float fast_exp(float x) {
#if defined(__CUDA_ARCH__)
    return __expf(x);
#else
    return static_cast<float>(std::exp(x));
#endif
}

// computes log(x) using fast device intrinsics when available
__host__ __device__ __forceinline__ float fast_log(float x) {
#if defined(__CUDA_ARCH__)
    return __logf(x);
#else
    return static_cast<float>(std::log(x));
#endif
}

// computes 1 over sqrt(x) using rsqrtf
__device__ __forceinline__ float fast_rsqrt(float x) { return rsqrtf(x); }

// computes an approximate sqrt(x) using rsqrtf
__device__ __forceinline__ float fast_sqrt(float x) {
    return x * rsqrtf(x + 1e-10f); // Avoid division, use rsqrt
}

// computes euclidean distance between two 2d points
__host__ __device__ __forceinline__ float distance(float x1, float y1, float x2,
                                                   float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

// computes squared euclidean distance between two 2d points
__host__ __device__ __forceinline__ float distance_squared(float x1, float y1,
                                                           float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return dx * dx + dy * dy;
}

// applies gamma correction to convert linear to gamma space
__host__ __device__ __forceinline__ float linear_to_gamma(float linear,
                                                          float gamma = 2.2f) {
    return fast_pow(linear, 1.0f / gamma);
}

// removes gamma correction to convert gamma to linear space
__host__ __device__ __forceinline__ float gamma_to_linear(float gamma_val,
                                                          float gamma = 2.2f) {
    return fast_pow(gamma_val, gamma);
}

// approximates linear to srgb conversion for a single channel
__device__ __forceinline__ float linear_to_srgb_approx(float x) {
    return sqrtf(fmaxf(0.0f, x));
}

// packs four normalized floats into a 32 bit rgba8 integer
__host__ __device__ __forceinline__ unsigned int pack_float4(float x, float y,
                                                             float z, float w) {
    unsigned int r = (unsigned int)(fminf(fmaxf(x, 0.0f), 1.0f) * 255.0f);
    unsigned int g = (unsigned int)(fminf(fmaxf(y, 0.0f), 1.0f) * 255.0f);
    unsigned int b = (unsigned int)(fminf(fmaxf(z, 0.0f), 1.0f) * 255.0f);
    unsigned int a = (unsigned int)(fminf(fmaxf(w, 0.0f), 1.0f) * 255.0f);

    return (a << 24) | (b << 16) | (g << 8) | r;
}

// unpacks a 32 bit rgba8 integer into four normalized floats
__host__ __device__ __forceinline__ void
unpack_float4(unsigned int packed, float &x, float &y, float &z, float &w) {
    const float inv255 = 1.0f / 255.0f;
    x = (packed & 0xFF) * inv255;
    y = ((packed >> 8) & 0xFF) * inv255;
    z = ((packed >> 16) & 0xFF) * inv255;
    w = ((packed >> 24) & 0xFF) * inv255;
}

// returns the fractional part of x
__host__ __device__ __forceinline__ float fract(float x) {
    return x - floorf(x);
}

// returns a floating modulus similar to fmodf
__host__ __device__ __forceinline__ float mod(float x, float y) {
    return x - y * floorf(x / y);
}

// returns the sign of x as -1 0 or 1
__host__ __device__ __forceinline__ float sign(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

// divides a by b with an epsilon guard and a fallback value
__host__ __device__ __forceinline__ float
safe_divide(float a, float b, float default_val = 0.0f) {
    return (fabsf(b) > EPSILON) ? (a / b) : default_val;
}

// checks whether x is finite
__host__ __device__ __forceinline__ bool is_finite(float x) {
    return isfinite(x);
}

// solves a quadratic or linear equation and returns sorted real roots
__host__ __device__ __forceinline__ bool
solve_quadratic(float a, float b, float c, float &x0, float &x1) {
    // Linear fallback when a is nearly zero: b*x + c = 0
    if (fabsf(a) <= EPSILON) {
        if (fabsf(b) <= EPSILON)
            return false;
        float x = -c / b;
        x0 = x;
        x1 = x;
        return true;
    }

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f)
        return false;

    float sqrt_disc = sqrtf(discriminant);

    // Numerically stable form to avoid catastrophic cancellation
    float q = -0.5f * (b + copysignf(sqrt_disc, b));
    float r0 = q / a;
    float r1 = c / q;

    if (r0 > r1) {
        float tmp = r0;
        r0 = r1;
        r1 = tmp;
    }

    x0 = r0;
    x1 = r1;
    return true;
}

// bit casts a float to a 32 bit unsigned integer
__device__ __forceinline__ unsigned int float_to_uint_bits(float f) {
    return __float_as_uint(f);
}

// bit casts a 32 bit unsigned integer to a float
__device__ __forceinline__ float uint_bits_to_float(unsigned int u) {
    return __uint_as_float(u);
}

// returns the maximum of two floats using fmaxf
__device__ __forceinline__ float fast_max(float a, float b) {
    return fmaxf(a, b); // CUDA intrinsic
}

// returns the minimum of two floats using fminf
__device__ __forceinline__ float fast_min(float a, float b) {
    return fminf(a, b); // CUDA intrinsic
}

#endif // MATH_UTILS_CUH
