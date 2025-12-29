/*
this file provides physically based rendering utility functions for shading
it includes microfacet distribution geometry and fresnel helper routines
it provides common scalar and vector math used by brdf implementations
it is designed for cuda device code and can be included in multiple translation
units it aims to keep brdf related formulas in one place to avoid duplication it
expects vec3 and related math types to be available from the common math headers
*/

#ifndef PBR_UTILS_CUH
#define PBR_UTILS_CUH

#include "common/vec3.cuh"
#include "pathtracer/math/mathutils.cuh"

__device__ __forceinline__ vec3 fresnelSchlick(float cosTheta, const vec3 &F0) {
    cosTheta = clamp01(cosTheta);
    float f = 1.0f - cosTheta;
    float f2 = f * f;
    float f5 = f2 * f2 * f;
    return F0 + (vec3(1.0f) - F0) * f5;
}

__device__ __forceinline__ vec3 fresnelSchlickRoughness(float cosTheta,
                                                        const vec3 &F0,
                                                        float roughness) {
    cosTheta = clamp01(cosTheta);
    float f = 1.0f - cosTheta;
    float f2 = f * f;
    float f5 = f2 * f2 * f;
    vec3 maxRefl =
        vec3(fmaxf(1.0f - roughness, F0.x), fmaxf(1.0f - roughness, F0.y),
             fmaxf(1.0f - roughness, F0.z));
    return F0 + (maxRefl - F0) * f5;
}

__device__ __forceinline__ float distributionGGX(const vec3 &N, const vec3 &H,
                                                 float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    denom = PI * denom * denom;

    return a2 / fmaxf(denom, 1e-6f);
}

__device__ __forceinline__ float distributionGGX_fast(float NdotH, float a2) {
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom + 1e-6f);
}

__device__ __forceinline__ float geometrySchlickGGX(float NdotV,
                                                    float roughness) {
    float r = (roughness + 1.0f);
    float k = (r * r) * 0.125f;

    return NdotV / (NdotV * (1.0f - k) + k + 1e-6f);
}

__device__ __forceinline__ float geometrySmith(const vec3 &N, const vec3 &V,
                                               const vec3 &L, float roughness) {
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

__device__ __forceinline__ float geometrySmith_fast(float NdotV, float NdotL,
                                                    float roughness) {
    float r = (roughness + 1.0f);
    float k = (r * r) * 0.125f;

    float ggx1 = NdotL / (NdotL * (1.0f - k) + k + 1e-6f);
    float ggx2 = NdotV / (NdotV * (1.0f - k) + k + 1e-6f);

    return ggx1 * ggx2;
}

__device__ __forceinline__ vec3 calculateIridescence(float thickness,
                                                     float cosTheta,
                                                     float filmIOR = 1.3f,
                                                     float baseIOR = 1.5f) {
    cosTheta = clamp01(cosTheta);

    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float sinThetaFilm = sinTheta / filmIOR;

    if (sinThetaFilm * sinThetaFilm > 1.0f) {
        return vec3(1.0f);
    }

    float cosThetaFilm = sqrtf(1.0f - sinThetaFilm * sinThetaFilm);
    float OPD = 2.0f * filmIOR * thickness * cosThetaFilm;

    float R_air_film_s = (1.0f - filmIOR) / (1.0f + filmIOR);
    R_air_film_s *= R_air_film_s;

    float R_film_base_s = (filmIOR - baseIOR) / (filmIOR + baseIOR);
    R_film_base_s *= R_film_base_s;

    const float inv_wavelengths[3] = {1.0f / 650.0f, 1.0f / 550.0f,
                                      1.0f / 450.0f};
    vec3 result;

    float sqrtR1R2 = sqrtf(R_air_film_s * R_film_base_s);
    float R_max = (sqrtf(R_air_film_s) + sqrtf(R_film_base_s));
    R_max *= R_max;
    float inv_R_max = 1.0f / (R_max + 1e-6f);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        float delta = TWO_PI * OPD * inv_wavelengths[i];
        float R_total =
            R_air_film_s + R_film_base_s + 2.0f * sqrtR1R2 * cosf(delta);
        result[i] = clamp01(R_total * inv_R_max);
    }

    return result;
}

__device__ __forceinline__ float schlick_dielectric(float cosTheta, float ior_i,
                                                    float ior_t) {
    cosTheta = clamp01(cosTheta);

    float r0 = (ior_i - ior_t) / (ior_i + ior_t);
    r0 = r0 * r0;

    float f = 1.0f - cosTheta;
    float f2 = f * f;
    float f5 = f2 * f2 * f;
    return r0 + (1.0f - r0) * f5;
}

__device__ __forceinline__ vec3 log_vec3(const vec3 &v) {
    const float EPS = 1e-12f;
    return vec3(logf(fmaxf(v.x, EPS)), logf(fmaxf(v.y, EPS)),
                logf(fmaxf(v.z, EPS)));
}

__device__ __forceinline__ float schlick_dielectric_oneIOR(float cosTheta,
                                                           float ior) {
    return schlick_dielectric(cosTheta, 1.0f, ior);
}

__device__ __forceinline__ vec3 log(const vec3 &v) {
    return vec3(logf(v.x), logf(v.y), logf(v.z));
}

__device__ __forceinline__ vec3 beerLambert(const vec3 &absorption_coefficient,
                                            float dist) {
    vec3 coeff = vec3(fmaxf(absorption_coefficient.x, 0.0f),
                      fmaxf(absorption_coefficient.y, 0.0f),
                      fmaxf(absorption_coefficient.z, 0.0f));
    return vec3(expf(-coeff.x * dist), expf(-coeff.y * dist),
                expf(-coeff.z * dist));
}

#endif
