/*
this file provides rendering utility functions used across the renderer
it includes sampling helpers tone mapping color conversions and small math
utilities it centralizes helpers that are shared between kernels to keep code
consistent it is intended for cuda device code and host side glue where
applicable it expects common math types and camera or scene definitions to be
included by the caller
*/

#ifndef RENDER_UTILS_CUH
#define RENDER_UTILS_CUH

#include "common/mat4.cuh"
#include "common/matrix.cuh"
#include "common/ray.cuh"
#include "common/vec3.cuh"
#include "pathtracer/math/mathutils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float attenuate(float distance, float range) {
    float att = range / (range + distance);
    return att * att;
}

__device__ __forceinline__ float attenuate_physical(float distance_sq,
                                                    float range_sq) {
    if (distance_sq > range_sq)
        return 0.0f;
    return 1.0f / (distance_sq + 1.0f);
}

__host__ __device__ __forceinline__ float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__host__ __device__ __forceinline__ vec3 clamp01(const vec3 &v) {
    return vec3(clamp01(v.x), clamp01(v.y), clamp01(v.z));
}

__host__ __device__ __forceinline__ vec3 reflectVec(const vec3 &I,
                                                    const vec3 &N) {
    return I - 2.0f * dot(I, N) * N;
}

__host__ __device__ __forceinline__ bool
refractVec(const vec3 &I, const vec3 &N, float eta, vec3 &T) {
    float NdotI = dot(N, I);
    float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);
    if (k < 0.0f)
        return false;
    T = eta * I - (eta * NdotI + sqrtf(k)) * N;
    return true;
}

__host__ __device__ __forceinline__ vec3 faceForward(const vec3 &N,
                                                     const vec3 &I) {
    return (dot(N, I) < 0.0f) ? N : (-N);
}

__host__ __device__ __forceinline__ vec3 fmaxf(const vec3 &a, const vec3 &b) {
    return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ __forceinline__ vec3 fminf(const vec3 &a, const vec3 &b) {
    return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ __forceinline__ vec3 absf(const vec3 &v) {
    return vec3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

__device__ __forceinline__ vec3 reinhard_tonemap(const vec3 &color) {
    return color / (color + vec3(1.0f));
}

__device__ __forceinline__ vec3 aces_tonemap(vec3 color) {
    const mat3 ACES_INPUT_MAT =
        mat3(0.59719f, 0.35458f, 0.04823f, 0.07600f, 0.90834f, 0.01566f,
             0.02840f, 0.13383f, 0.83777f);

    const mat3 ACES_OUTPUT_MAT =
        mat3(1.60475f, -0.53108f, -0.07367f, -0.10208f, 1.10813f, -0.00605f,
             -0.00327f, -0.07276f, 1.07602f);

    vec3 aces_color = ACES_INPUT_MAT * color;

    vec3 a = aces_color * (aces_color + 0.0245786f) - 0.000090537f;
    vec3 b = aces_color * (0.983729f * aces_color + 0.4329510f) + 0.238081f;
    aces_color = clamp(a / b, 0.0f, 1.0f);

    aces_color = ACES_OUTPUT_MAT * aces_color;

    return clamp(aces_color, 0.0f, 1.0f);
}

__device__ __forceinline__ vec3 uncharted2_tonemap_partial(vec3 x) {
    const float A = 0.15f;
    const float B = 0.50f;
    const float C = 0.10f;
    const float D = 0.20f;
    const float E = 0.02f;
    const float F = 0.30f;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

__device__ __forceinline__ vec3 uncharted2_tonemap(vec3 color,
                                                   float exposure = 2.0f) {
    const float W = 11.2f;
    vec3 curr = uncharted2_tonemap_partial(color * exposure);
    vec3 whiteScale = vec3(1.0f) / uncharted2_tonemap_partial(vec3(W));
    return curr * whiteScale;
}

__device__ __forceinline__ vec3 sampleSky(const Ray &r, const vec3 &top,
                                          const vec3 &bottom, bool useSky,
                                          cudaTextureObject_t envMap) {
    if (!useSky)
        return vec3(0.0f);

    if (envMap == 0) {

        float t = 0.5f * (r.direction().y + 1.0f);
        return lerp(bottom, top, t);
    }

    vec3 dir = r.direction();
    float phi = atan2f(dir.z, dir.x);

    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));

    float u = (phi + PI) * (1.0f / TWO_PI);
    float v = theta * (1.0f / PI);

    float4 color = tex2D<float4>(envMap, u, v);
    return vec3(color.x, color.y, color.z);
}

__device__ __forceinline__ vec3 sampleSkyDir(const vec3 &dir, const vec3 &top,
                                             const vec3 &bottom, bool useSky,
                                             cudaTextureObject_t envMap) {
    if (!useSky)
        return vec3(0.0f);

    if (envMap == 0) {
        float t = 0.5f * (dir.y + 1.0f);
        return lerp(bottom, top, t);
    }

    float phi = atan2f(dir.z, dir.x);
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));

    float u = (phi + PI) * (1.0f / TWO_PI);
    float v = theta * (1.0f / PI);

    float4 color = tex2D<float4>(envMap, u, v);
    return vec3(color.x, color.y, color.z);
}

__device__ __forceinline__ vec3 linear_to_srgb(const vec3 &color) {
    return vec3(powf(color.x, 1.0f / 2.2f), powf(color.y, 1.0f / 2.2f),
                powf(color.z, 1.0f / 2.2f));
}

__device__ __forceinline__ vec3 srgb_to_linear(const vec3 &color) {
    return vec3(powf(color.x, 2.2f), powf(color.y, 2.2f), powf(color.z, 2.2f));
}

__device__ __forceinline__ vec3 linear_to_srgb_fast(const vec3 &color) {

    return vec3(sqrtf(fmaxf(0.0f, color.x)), sqrtf(fmaxf(0.0f, color.y)),
                sqrtf(fmaxf(0.0f, color.z)));
}

#endif
