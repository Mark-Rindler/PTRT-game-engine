#ifndef MATH_UTILS_CUH
#define MATH_UTILS_CUH

#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

static constexpr float PI = 3.14159265358979323846f;
static constexpr float TWO_PI = 6.28318530717958647692f;
static constexpr float PI_OVER_TWO = 1.57079632679489661923f;
static constexpr float PI_OVER_FOUR = 0.78539816339744830961f;
static constexpr float INV_PI = 0.31830988618379067154f;
static constexpr float INV_TWO_PI = 0.15915494309189533577f;
static constexpr float EPSILON = 1e-6f;
static constexpr float DEG_TO_RAD = 0.01745329251994329577f; // PI/180
static constexpr float RAD_TO_DEG = 57.2957795130823208768f; // 180/PI

// MATH FUNCTIONS

__host__ __device__ __forceinline__ float degrees_to_radians(float degrees) {
    return degrees * DEG_TO_RAD;
}

__host__ __device__ __forceinline__ float radians_to_degrees(float radians) {
    return radians * RAD_TO_DEG;
}

__host__ __device__ __forceinline__ float clamp(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__host__ __device__ __forceinline__ int clamp(int x, int lo, int hi) {
    return min(max(x, lo), hi);
}

// FMA lerp
__host__ __device__ __forceinline__ float lerp(float a, float b, float t) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(t, b - a, a);
#else
    return a + t * (b - a);
#endif
}

__host__ __device__ __forceinline__ float smoothstep(float edge0, float edge1,
                                                     float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// RANDOM NUMBER GENERATION

__device__ __forceinline__ float random_float(curandState *state) {
    return curand_uniform(state);
}

__device__ __forceinline__ float random_float(curandState *state, float lo,
                                              float hi) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(hi - lo, curand_uniform(state), lo);
#else
    return lo + (hi - lo) * curand_uniform(state);
#endif
}

__device__ __forceinline__ int random_int(curandState *state, int lo, int hi) {
    return lo + static_cast<int>((hi - lo + 1) * curand_uniform(state));
}

__device__ __forceinline__ void init_rand_state(curandState *state, int tid,
                                                int seed = 1984) {
    curand_init(seed, tid, 0, state);
}

// FAST TRANSCENDENTAL APPROXIMATIONS

__host__ __device__ __forceinline__ float fast_pow(float a, float b) {
#ifdef __CUDA_ARCH__
    return __powf(a, b);
#else
    return powf(a, b);
#endif
}

__host__ __device__ __forceinline__ float fast_exp(float x) {
#ifdef __CUDA_ARCH__
    return __expf(x);
#else
    return expf(x);
#endif
}

__host__ __device__ __forceinline__ float fast_log(float x) {
#ifdef __CUDA_ARCH__
    return __logf(x);
#else
    return logf(x);
#endif
}

// Fast reciprocal  single precision
__device__ __forceinline__ float fast_rcp(float x) {
#ifdef __CUDA_ARCH__
    return __frcp_rn(x);
#else
    return 1.0f / x;
#endif
}

// Fast inverse square root
__device__ __forceinline__ float fast_rsqrt(float x) {
#ifdef __CUDA_ARCH__
    return rsqrtf(x);
#else
    return 1.0f / sqrtf(x);
#endif
}

// Fast square root using rsqrt
__device__ __forceinline__ float fast_sqrt(float x) {
#ifdef __CUDA_ARCH__
    return x * rsqrtf(x + 1e-30f); // Avoid div by zero
#else
    return sqrtf(x);
#endif
}

// Fast sin/cos using CUDA intrinsics
__device__ __forceinline__ void fast_sincos(float x, float *s, float *c) {
#ifdef __CUDA_ARCH__
    __sincosf(x, s, c);
#else
    *s = sinf(x);
    *c = cosf(x);
#endif
}

__device__ __forceinline__ float fast_sin(float x) {
#ifdef __CUDA_ARCH__
    return __sinf(x);
#else
    return sinf(x);
#endif
}

__device__ __forceinline__ float fast_cos(float x) {
#ifdef __CUDA_ARCH__
    return __cosf(x);
#else
    return cosf(x);
#endif
}

// UTILITY FUNCTIONS

__host__ __device__ __forceinline__ float distance(float x1, float y1, float x2,
                                                   float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

__host__ __device__ __forceinline__ float distance_squared(float x1, float y1,
                                                           float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return dx * dx + dy * dy;
}

// Gamma correction using fast pow
__host__ __device__ __forceinline__ float linear_to_gamma(float linear,
                                                          float gamma = 2.2f) {
#ifdef __CUDA_ARCH__
    return __powf(linear, 1.0f / gamma);
#else
    return powf(linear, 1.0f / gamma);
#endif
}

__host__ __device__ __forceinline__ float gamma_to_linear(float gamma_val,
                                                          float gamma = 2.2f) {
#ifdef __CUDA_ARCH__
    return __powf(gamma_val, gamma);
#else
    return powf(gamma_val, gamma);
#endif
}

// Fast gamma 2.2 approximation using sqrt (gamma 2.0)
__device__ __forceinline__ float fast_linear_to_srgb(float x) {
    return sqrtf(fmaxf(x, 0.0f));
}

// Pack float4 to RGBA8 - optimized
__host__ __device__ __forceinline__ unsigned int pack_float4(float x, float y,
                                                             float z, float w) {
    unsigned int r =
        static_cast<unsigned int>(fminf(fmaxf(x, 0.0f), 1.0f) * 255.0f);
    unsigned int g =
        static_cast<unsigned int>(fminf(fmaxf(y, 0.0f), 1.0f) * 255.0f);
    unsigned int b =
        static_cast<unsigned int>(fminf(fmaxf(z, 0.0f), 1.0f) * 255.0f);
    unsigned int a =
        static_cast<unsigned int>(fminf(fmaxf(w, 0.0f), 1.0f) * 255.0f);
    return (a << 24) | (b << 16) | (g << 8) | r;
}

__host__ __device__ __forceinline__ void
unpack_float4(unsigned int packed, float &x, float &y, float &z, float &w) {
    constexpr float inv255 = 1.0f / 255.0f;
    x = (packed & 0xFF) * inv255;
    y = ((packed >> 8) & 0xFF) * inv255;
    z = ((packed >> 16) & 0xFF) * inv255;
    w = ((packed >> 24) & 0xFF) * inv255;
}

__host__ __device__ __forceinline__ float fract(float x) {
    return x - floorf(x);
}

__host__ __device__ __forceinline__ float mod(float x, float y) {
    return x - y * floorf(x / y);
}

__host__ __device__ __forceinline__ float sign(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

__host__ __device__ __forceinline__ float
safe_divide(float a, float b, float default_val = 0.0f) {
    return (fabsf(b) > EPSILON) ? (a / b) : default_val;
}

__host__ __device__ __forceinline__ bool is_finite(float x) {
    return isfinite(x);
}

// quadratic solver
__host__ __device__ __forceinline__ bool
solve_quadratic(float a, float b, float c, float &x0, float &x1) {
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f)
        return false;

    float sqrt_disc = sqrtf(discriminant);
    float inv_2a = 0.5f / a;
    x0 = (-b - sqrt_disc) * inv_2a;
    x1 = (-b + sqrt_disc) * inv_2a;

    if (x0 > x1) {
        float temp = x0;
        x0 = x1;
        x1 = temp;
    }
    return true;
}

// FAST PCG RANDOM

__device__ __forceinline__ unsigned int pcg_hash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float pcg_float(unsigned int &seed) {
    seed = pcg_hash(seed);
    return static_cast<float>(seed) * 2.3283064365386963e-10f; // / 2^32
}

// BRANCHLESS MIN/MAX FOR INDICES

__device__ __forceinline__ int min3_index(float a, float b, float c) {
    return (a < b) ? ((a < c) ? 0 : 2) : ((b < c) ? 1 : 2);
}

__device__ __forceinline__ int max3_index(float a, float b, float c) {
    return (a > b) ? ((a > c) ? 0 : 2) : ((b > c) ? 1 : 2);
}

#endif
