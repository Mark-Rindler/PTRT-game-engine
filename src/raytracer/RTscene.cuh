#ifndef RT_SCENE_CUH
#define RT_SCENE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "common/matrix.cuh"
#include "common/ray.cuh"
#include "common/triangle.cuh"
#include "common/vec3.cuh"
#include "raytracer/RTcamera.cuh"
#include "raytracer/RTmathutils.cuh"
#include "raytracer/RTmesh.cuh"

// MATERIAL STRUCTURE

struct Material {
    vec3 albedo;
    vec3 specular;
    float metallic;
    float roughness;
    vec3 emission;

    float ior;
    float transmission;
    float transmissionRoughness;

    float clearcoat;
    float clearcoatRoughness;

    vec3 subsurfaceColor;
    float subsurfaceRadius;

    float anisotropy;
    float sheen;
    vec3 sheenTint;

    float iridescence;
    float iridescenceThickness;

    __host__ __device__ Material()
        : albedo(vec3(0.8f)), specular(vec3(0.04f)), metallic(0.0f),
          roughness(0.5f), emission(vec3(0.0f)), ior(1.5f), transmission(0.0f),
          transmissionRoughness(0.0f), clearcoat(0.0f),
          clearcoatRoughness(0.03f), subsurfaceColor(vec3(1.0f)),
          subsurfaceRadius(0.0f), anisotropy(0.0f), sheen(0.0f),
          sheenTint(vec3(0.5f)), iridescence(0.0f),
          iridescenceThickness(550.0f) {}

    __host__ __device__ Material(const vec3 &alb, float rough = 0.5f,
                                 float met = 0.0f)
        : Material() {
        albedo = alb;
        roughness = rough;
        metallic = met;
        specular = lerp(vec3(0.04f), albedo, metallic);
    }
};
// LIGHT TYPES

enum LightType { LIGHT_POINT = 0, LIGHT_DIRECTIONAL = 1, LIGHT_SPOT = 2 };

struct Light {
    LightType type;
    vec3 position;
    vec3 direction;
    vec3 color;
    float intensity;
    float range;
    float innerCone;
    float outerCone;

    __host__ __device__ Light()
        : type(LIGHT_POINT), position(vec3(0, 10, 0)),
          direction(vec3(0, -1, 0)), color(vec3(1.0f)), intensity(1.0f),
          range(100.0f), innerCone(0.5f), outerCone(0.7f) {}
};

// DEVICE MESH DESCRIPTOR

struct DeviceMesh {
    vec3 *verts;
    Tri *faces;
    int faceCount;
    Material material;
    DeviceBVHNode *bvhNodes;
    int nodeCount;
    int *primIndices;

    vec3 translation;
    mat3 rotation;
    mat3 invRotation;

    __host__ __device__ void setIdentity() {
        translation = vec3(0.0f);
        rotation = mat3();
        invRotation = mat3();
    }
};

// HIT INFO

struct HitInfo {
    bool hit;
    float t;
    vec3 point;
    vec3 normal;
    Material material;

    __device__ __forceinline__ HitInfo() : hit(false), t(1e30f) {}
};

// forward declarations
__global__ void render_kernel(unsigned char *out, int W, int H, Camera cam,
                              DeviceMesh *meshes, int nMeshes, Light *lights,
                              int nLights, vec3 ambientLight, vec3 skyColorTop,
                              vec3 skyColorBottom, bool useSky);

// SHADING FUNCTIONS

__device__ __forceinline__ float attenuate(float distance, float range) {
    float att = range / (range + distance);
    return att * att;
}

// Fast Fresnel using approximation
__device__ __forceinline__ vec3 fresnelSchlick(float cosTheta, const vec3 &F0) {
    float x = 1.0f - cosTheta;
    float x2 = x * x;
    float x5 = x2 * x2 * x; // (1-cos)^5 without pow()
    return F0 + (vec3(1.0f) - F0) * x5;
}

__device__ __forceinline__ vec3 fresnelSchlickRoughness(float cosTheta,
                                                        const vec3 &F0,
                                                        float roughness) {
    float x = fmaxf(1.0f - cosTheta, 0.0f);
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    vec3 maxRefl =
        vec3(fmaxf(1.0f - roughness, F0.x), fmaxf(1.0f - roughness, F0.y),
             fmaxf(1.0f - roughness, F0.z));
    return F0 + (maxRefl - F0) * x5;
}

__device__ __forceinline__ float distributionGGX(const vec3 &N, const vec3 &H,
                                                 float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    denom = PI * denom * denom;

    return a2 / fmaxf(denom, 0.001f);
}

__device__ __forceinline__ float geometrySchlickGGX(float NdotV,
                                                    float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) * 0.125f; // /8
    return NdotV / (NdotV * (1.0f - k) + k + 0.001f);
}

__device__ __forceinline__ float geometrySmith(const vec3 &N, const vec3 &V,
                                               const vec3 &L, float roughness) {
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotL = fmaxf(dot(N, L), 0.0f);
    return geometrySchlickGGX(NdotV, roughness) *
           geometrySchlickGGX(NdotL, roughness);
}

__device__ __forceinline__ void buildTangentFrame(const vec3 &N, vec3 &T,
                                                  vec3 &B) {
    if (fabsf(N.z) < 0.9999f) {
        T = normalize(cross(vec3(0.0f, 0.0f, 1.0f), N));
    } else {
        T = normalize(cross(vec3(1.0f, 0.0f, 0.0f), N));
    }
    B = cross(N, T);
}

__device__ __forceinline__ float
distributionGGXAnisotropic(const vec3 &N, const vec3 &H, const vec3 &T,
                           const vec3 &B, float ax, float ay) {
    float NdotH = dot(N, H);
    if (NdotH <= 0.0f)
        return 0.0f;

    float TdotH = dot(T, H);
    float BdotH = dot(B, H);
    float ax2 = ax * ax;
    float ay2 = ay * ay;

    float denom =
        (TdotH * TdotH / ax2) + (BdotH * BdotH / ay2) + (NdotH * NdotH);
    denom = PI * ax * ay * denom * denom;
    return 1.0f / fmaxf(denom, 0.001f);
}

__device__ __forceinline__ float
geometrySchlickGGXAnisotropic(float NdotV, float TdotV, float BdotV, float ax,
                              float ay) {
    float ax2 = ax * ax;
    float ay2 = ay * ay;
    float lambda =
        sqrtf(ax2 * TdotV * TdotV + ay2 * BdotV * BdotV + NdotV * NdotV);
    return 2.0f * NdotV / (NdotV + lambda + 0.001f);
}

__device__ __forceinline__ float
geometrySmithAnisotropic(const vec3 &N, const vec3 &V, const vec3 &L,
                         const vec3 &T, const vec3 &B, float ax, float ay) {
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float TdotV = dot(T, V), BdotV = dot(B, V);
    float TdotL = dot(T, L), BdotL = dot(B, L);

    return geometrySchlickGGXAnisotropic(NdotV, TdotV, BdotV, ax, ay) *
           geometrySchlickGGXAnisotropic(NdotL, TdotL, BdotL, ax, ay);
}

__device__ __forceinline__ void
anisotropyToAlpha(float roughness, float anisotropy, float &ax, float &ay) {
    float r2 = roughness * roughness;
    float aspect = sqrtf(1.0f - 0.9f * fabsf(anisotropy));
    if (anisotropy >= 0.0f) {
        ax = r2 / aspect;
        ay = r2 * aspect;
    } else {
        ax = r2 * aspect;
        ay = r2 / aspect;
    }
    ax = fmaxf(ax, 0.001f);
    ay = fmaxf(ay, 0.001f);
}

__device__ __forceinline__ vec3 perturbDirectionGGX(const vec3 &dir,
                                                    const vec3 &N,
                                                    float roughness,
                                                    unsigned int &seed) {
    if (roughness < 0.01f)
        return dir;

    seed = seed * 747796405u + 2891336453u;
    float u1 = float(seed) * 2.3283064365386963e-10f;
    seed = seed * 747796405u + 2891336453u;
    float u2 = float(seed) * 2.3283064365386963e-10f;

    float a = roughness * roughness;
    float phi = TWO_PI * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (a * a - 1.0f) * u2));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    vec3 T, B;
    buildTangentFrame(dir, T, B);

    float sinPhi, cosPhi;
#ifdef __CUDA_ARCH__
    __sincosf(phi, &sinPhi, &cosPhi);
#else
    sinPhi = sinf(phi);
    cosPhi = cosf(phi);
#endif

    return normalize(T * (cosPhi * sinTheta) + B * (sinPhi * sinTheta) +
                     dir * cosTheta);
}

__host__ __device__ __forceinline__ float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ __forceinline__ vec3 calculateIridescence(float thickness,
                                                     float cosTheta,
                                                     float filmIOR = 1.3f,
                                                     float baseIOR = 1.5f) {
    cosTheta = clamp01(cosTheta);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float sinThetaFilm = sinTheta / filmIOR;

    if (sinThetaFilm * sinThetaFilm > 1.0f)
        return vec3(1.0f);

    float cosThetaFilm = sqrtf(1.0f - sinThetaFilm * sinThetaFilm);
    float OPD = 2.0f * filmIOR * thickness * cosThetaFilm;

    float R_air_film_s = (1.0f - filmIOR) / (1.0f + filmIOR);
    R_air_film_s *= R_air_film_s;
    float R_film_base_s = (filmIOR - baseIOR) / (filmIOR + baseIOR);
    R_film_base_s *= R_film_base_s;

    const float wavelengths[3] = {650.0f, 550.0f, 450.0f};
    vec3 result;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        float lambda = wavelengths[i];
        float delta = TWO_PI * OPD / lambda;
        float sqrtR1R2 = sqrtf(R_air_film_s * R_film_base_s);

#ifdef __CUDA_ARCH__
        float R_total =
            R_air_film_s + R_film_base_s + 2.0f * sqrtR1R2 * __cosf(delta);
#else
        float R_total =
            R_air_film_s + R_film_base_s + 2.0f * sqrtR1R2 * cosf(delta);
#endif

        float R_max = sqrtf(R_air_film_s) + sqrtf(R_film_base_s);
        R_max *= R_max;
        result[i] = clamp01(R_total / (R_max + 1e-6f));
    }
    return result;
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

__device__ __forceinline__ vec3 beerLambert(const vec3 &transRGBPerUnit,
                                            float dist) {
    vec3 t = clamp(transRGBPerUnit, 0.0f, 1.0f);
#ifdef __CUDA_ARCH__
    return vec3(__powf(t.x, dist), __powf(t.y, dist), __powf(t.z, dist));
#else
    return vec3(powf(t.x, dist), powf(t.y, dist), powf(t.z, dist));
#endif
}

__device__ __forceinline__ vec3 sampleSky(const vec3 &dir, const vec3 &top,
                                          const vec3 &bottom, bool useSky) {
    if (!useSky)
        return vec3(0.0f);
    float t = 0.5f * (dir.y + 1.0f);
    return lerp(bottom, top, t);
}

// BVH TRAVERSAL

__device__ __forceinline__ bool bvh_any_hit(const RayOpt &ray,
                                            const DeviceMesh &M, float tMax) {
    if (!M.bvhNodes || M.nodeCount <= 0 || !M.primIndices || !M.verts ||
        !M.faces || M.faceCount <= 0)
        return false;

    // transform ray to local space
    vec3 worldOrig = vec3(ray.orig.x, ray.orig.y, ray.orig.z);
    vec3 worldDir = vec3(ray.dir.x, ray.dir.y, ray.dir.z);
    vec3 localOrig = M.invRotation * (worldOrig - M.translation);
    vec3 localDir = M.invRotation * worldDir;
    RayOpt localRay(localOrig, localDir);

    // stack based traversal
    constexpr int STACK_SIZE = 32;
    int stack[STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int ni = stack[--sp];
        const DeviceBVHNode &N = M.bvhNodes[ni];

        float tEntry;
        if (!N.bbox.hit_fast(localRay, tMax, tEntry))
            continue;

        if (N.count > 0) {
            // leaf node  test triangles
            for (int i = 0; i < N.count; ++i) {
                const int fidx = M.primIndices[N.start + i];
                const Tri &idx = M.faces[fidx];
                const vec3 &v0 = M.verts[idx.v0];
                const vec3 &v1 = M.verts[idx.v1];
                const vec3 &v2 = M.verts[idx.v2];

                float t, u, v;
                if (intersect_triangle_mt(localRay, v0, v1, v2, t, u, v)) {
                    if (t > 1e-4f && t < tMax)
                        return true;
                }
            }
        } else {
            // Internal node  push children
            if (N.left >= 0 && sp < STACK_SIZE)
                stack[sp++] = N.left;
            if (N.right >= 0 && sp < STACK_SIZE)
                stack[sp++] = N.right;
        }
    }
    return false;
}

__device__ __forceinline__ HitInfo bvh_trace(const RayOpt &ray,
                                             const DeviceMesh &M) {
    HitInfo out;
    out.hit = false;
    out.t = 1e30f;

    if (!M.bvhNodes || M.nodeCount == 0 || !M.primIndices)
        return out;

    // Transform ray to local space
    vec3 worldOrig = vec3(ray.orig.x, ray.orig.y, ray.orig.z);
    vec3 worldDir = vec3(ray.dir.x, ray.dir.y, ray.dir.z);
    vec3 localOrig = M.invRotation * (worldOrig - M.translation);
    vec3 localDir = M.invRotation * worldDir;
    RayOpt localRay(localOrig, localDir);

    constexpr int MaxStack = 32;
    int stack[MaxStack];
    int sp = 0;
    int ni = 0;

    while (true) {
        const DeviceBVHNode &N = M.bvhNodes[ni];

        float tEntry;
        if (!N.bbox.hit_fast(localRay, out.t, tEntry)) {
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        if (N.count > 0) {
            // Leaf  test triangles
            for (int i = 0; i < N.count; ++i) {
                const int fidx = M.primIndices[N.start + i];
                const Tri &idx = M.faces[fidx];
                const vec3 &v0 = M.verts[idx.v0];
                const vec3 &v1 = M.verts[idx.v1];
                const vec3 &v2 = M.verts[idx.v2];

                float tHit, uHit, vHit;
                if (intersect_triangle_mt(localRay, v0, v1, v2, tHit, uHit,
                                          vHit) &&
                    tHit > 1e-4f && tHit < out.t) {
                    out.hit = true;
                    out.t = tHit;
                    out.point = ray.at(tHit);

                    vec3 e1 = v1 - v0, e2 = v2 - v0;
                    vec3 localNormal = normalize(cross(e1, e2));
                    out.normal = normalize(M.rotation * localNormal);
                    out.material = M.material;
                }
            }

            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        // Internal node  ordered traversal
        const int L = N.left, R = N.right;
        float tL = 1e30f, tR = 1e30f;
        bool hL = (L >= 0) && M.bvhNodes[L].bbox.hit_fast(localRay, out.t, tL);
        bool hR = (R >= 0) && M.bvhNodes[R].bbox.hit_fast(localRay, out.t, tR);

        if (!hL && !hR) {
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        // Traverse near child first
        if (hL && hR) {
            if (tL <= tR) {
                if (sp < MaxStack)
                    stack[sp++] = R;
                ni = L;
            } else {
                if (sp < MaxStack)
                    stack[sp++] = L;
                ni = R;
            }
        } else if (hL) {
            ni = L;
        } else {
            ni = R;
        }
    }
    return out;
}

// RAY TRACING CORE

__device__ inline vec3 shadeOneBounce(const RayOpt &r, DeviceMesh *meshes,
                                      int nMeshes, Light *lights, int nLights,
                                      const vec3 &ambientLight,
                                      const vec3 &skyTop, const vec3 &skyBottom,
                                      bool useSky);

__device__ __forceinline__ HitInfo traceRay(const RayOpt &ray,
                                            DeviceMesh *meshes, int nMeshes) {
    HitInfo best;
    best.hit = false;
    best.t = 1e30f;

    for (int m = 0; m < nMeshes; ++m) {
        HitInfo h = bvh_trace(ray, meshes[m]);
        if (h.hit && h.t < best.t)
            best = h;
    }
    return best;
}

__device__ inline vec3 calculatePBRLightingCore(
    const HitInfo &hit, const RayOpt &ray, DeviceMesh *meshes, int nMeshes,
    Light *lights, int nLights, const vec3 &ambientLight, const vec3 &skyTop,
    const vec3 &skyBottom, bool useSky, bool allowSpecTransmission);

__device__ inline vec3 calculatePBRLightingCore(
    const HitInfo &hit, const RayOpt &ray, DeviceMesh *meshes, int nMeshes,
    Light *lights, int nLights, const vec3 &ambientLight, const vec3 &skyTop,
    const vec3 &skyBottom, bool useSky, bool allowSpecTransmission) {

    vec3 color = vec3(0.0f);
    vec3 V = vec3(-ray.dir.x, -ray.dir.y, -ray.dir.z);
    const vec3 &Ng = hit.normal;
    const Material &mat = hit.material;

    const float rough = fminf(fmaxf(mat.roughness, 0.02f), 1.0f);
    const float metal = fminf(fmaxf(mat.metallic, 0.0f), 1.0f);
    const bool isGlass = (mat.transmission > 0.0f) && (metal < 0.1f);

    vec3 F0 = mat.specular;
    F0 = lerp(F0, mat.albedo, metal);

    color = color + mat.emission;

    float NdotV = fmaxf(dot(Ng, V), 0.0f);
    vec3 F_ambient = fresnelSchlickRoughness(NdotV, F0, rough);
    vec3 kD_ambient = (vec3(1.0f) - F_ambient) * (1.0f - metal);
    if (isGlass)
        kD_ambient = vec3(0.0f);
    color = color + kD_ambient * mat.albedo * ambientLight;

    for (int i = 0; i < nLights; ++i) {
        const Light &light = lights[i];
        vec3 L;
        float attenuation = 1.0f;

        if (light.type == LIGHT_DIRECTIONAL) {
            L = -light.direction;
        } else {
            vec3 toLight = light.position - hit.point;
            float distance = toLight.length();
            L = toLight / fmaxf(distance, 1e-6f);
            float att = attenuate(distance, light.range);
            if (light.type == LIGHT_SPOT) {
                float theta = dot(L, -light.direction);
                float epsilon = light.innerCone - light.outerCone;
                float spotIntensity =
                    clamp((theta - light.outerCone) / epsilon, 0.0f, 1.0f);
                att *= spotIntensity;
            }
            attenuation = att;
        }

        // Shadow ray with optimized structure
        bool inShadow = false;
        const float eps = 1e-3f * fmaxf(1.0f, hit.t);
        RayOpt shadowRay(hit.point + Ng * eps, L);
        float lightDistance = (light.type == LIGHT_DIRECTIONAL)
                                  ? 1e30f
                                  : (light.position - hit.point).length();

        for (int m = 0; m < nMeshes && !inShadow; ++m) {
            if (meshes[m].material.transmission > 0.0f)
                continue;
            if (bvh_any_hit(shadowRay, meshes[m], lightDistance))
                inShadow = true;
        }
        if (inShadow)
            continue;

        vec3 H = (L + V).normalized();
        float NdotL = fmaxf(dot(Ng, L), 0.0f);
        float VdotH = fmaxf(dot(V, H), 0.0f);

        float D, G;
        if (fabsf(mat.anisotropy) > 0.01f) {
            vec3 T, B;
            buildTangentFrame(Ng, T, B);
            float ax, ay;
            anisotropyToAlpha(rough, mat.anisotropy, ax, ay);
            D = distributionGGXAnisotropic(Ng, H, T, B, ax, ay);
            G = geometrySmithAnisotropic(Ng, V, L, T, B, ax, ay);
        } else {
            D = distributionGGX(Ng, H, rough);
            G = geometrySmith(Ng, V, L, rough);
        }

        vec3 F = fresnelSchlick(VdotH, F0);

        if (mat.iridescence > 0.0f) {
            vec3 iridColor =
                calculateIridescence(mat.iridescenceThickness, VdotH);
            F = lerp(F, F * iridColor, mat.iridescence);
        }

        vec3 specular =
            (D * G * F) / (4.0f * fmaxf(dot(Ng, V), 0.0f) * NdotL + 0.001f);

        vec3 kS = F;
        vec3 kD = (vec3(1.0f) - kS) * (1.0f - metal);
        vec3 diffuse = mat.albedo * INV_PI;

        if (mat.sheen > 0.0f) {
            float x = 1.0f - VdotH;
            float x2 = x * x;
            float FH = x2 * x2 * x;
            vec3 sheenColor = lerp(vec3(1.0f), mat.sheenTint, FH);
            kD = kD + sheenColor * mat.sheen * (1.0f - metal);
        }

        if (mat.subsurfaceRadius > 0.0f) {
            float sss = fmaxf(dot(V, -L), 0.0f);
            sss = sss * sss * mat.subsurfaceRadius;
            diffuse = lerp(diffuse, mat.subsurfaceColor * INV_PI, sss);
        }

        vec3 thinTrans = vec3(0.0f);
        if (isGlass && !allowSpecTransmission) {
            kD = vec3(0.0f);
            thinTrans = (vec3(1.0f) - F) * mat.transmission;
        }

        vec3 Lo = (kD * diffuse + specular + thinTrans) * light.color *
                  light.intensity * 20.0f * NdotL * attenuation;

        if (mat.clearcoat > 0.0f) {
            float ccD = distributionGGX(Ng, H, mat.clearcoatRoughness);
            float ccG = geometrySmith(Ng, V, L, mat.clearcoatRoughness);
            vec3 ccF = fresnelSchlick(VdotH, vec3(0.04f));
            vec3 ccBRDF = (ccD * ccG * ccF) /
                          (4.0f * fmaxf(dot(Ng, V), 0.0f) * NdotL + 0.001f);
            Lo = Lo * (vec3(1.0f) - mat.clearcoat * ccF) +
                 ccBRDF * light.color * light.intensity * 20.0f * NdotL *
                     attenuation * mat.clearcoat;
        }

        color = color + Lo;
    }

    // Glass/transmission handling
    if (isGlass && allowSpecTransmission) {
        vec3 I = vec3(ray.dir.x, ray.dir.y, ray.dir.z);
        vec3 Nf = faceForward(Ng, I);
        float n1 = 1.0f, n2 = mat.ior;
        if (dot(Ng, I) > 0.0f) {
            float tmp = n1;
            n1 = n2;
            n2 = tmp;
            Nf = faceForward(Ng, I);
        }
        float eta = n1 / n2;

        float F0s = (n2 - n1) / (n2 + n1);
        F0s = F0s * F0s;
        float cosTheta = fmaxf(dot(-I, Nf), 0.0f);
        vec3 Fr = fresnelSchlick(cosTheta, vec3(F0s));

        const float eps = 1e-3f * fmaxf(1.0f, hit.t);

        unsigned int seed =
            __float_as_uint(hit.point.x * 12.9898f + hit.point.y * 78.233f +
                            hit.point.z * 45.164f);
        seed = seed * 747796405u + 2891336453u;

        vec3 Rdir = reflectVec(I, Nf).normalized();
        float reflRoughness = fmaxf(mat.roughness, mat.transmissionRoughness);
        if (reflRoughness > 0.02f) {
            Rdir = perturbDirectionGGX(Rdir, Nf, reflRoughness, seed);
        }
        vec3 Rcol = shadeOneBounce(RayOpt(hit.point + Nf * eps, Rdir), meshes,
                                   nMeshes, lights, nLights, ambientLight,
                                   skyTop, skyBottom, useSky);

        vec3 Tdir;
        vec3 Tcol = vec3(0.0f);
        bool refrOk = refractVec(I, Nf, eta, Tdir);
        if (refrOk) {
            Tdir = Tdir.normalized();
            if (mat.transmissionRoughness > 0.02f) {
                Tdir = perturbDirectionGGX(Tdir, -Nf, mat.transmissionRoughness,
                                           seed);
            }

            HitInfo h2 =
                traceRay(RayOpt(hit.point - Nf * eps, Tdir), meshes, nMeshes);
            float thickness = h2.hit ? h2.t : 1.0f;

            vec3 behind =
                h2.hit ? calculatePBRLightingCore(
                             h2, RayOpt(hit.point - Nf * eps, Tdir), meshes,
                             nMeshes, lights, nLights, ambientLight, skyTop,
                             skyBottom, useSky, false)
                       : sampleSky(Tdir, skyTop, skyBottom, useSky);

            vec3 absorb = beerLambert(clamp(mat.albedo, 0.0f, 1.0f), thickness);
            Tcol = absorb * behind;
        } else {
            Fr = vec3(1.0f);
        }

        color = color + Fr * Rcol + (vec3(1.0f) - Fr) * mat.transmission * Tcol;
    }

    return color;
}

__device__ inline vec3
calculatePBRLighting(const HitInfo &hit, const RayOpt &ray, DeviceMesh *meshes,
                     int nMeshes, Light *lights, int nLights,
                     const vec3 &ambientLight, const vec3 &skyTop,
                     const vec3 &skyBottom, bool useSky) {
    return calculatePBRLightingCore(hit, ray, meshes, nMeshes, lights, nLights,
                                    ambientLight, skyTop, skyBottom, useSky,
                                    true);
}

__device__ inline vec3 shadeOneBounce(const RayOpt &r, DeviceMesh *meshes,
                                      int nMeshes, Light *lights, int nLights,
                                      const vec3 &ambientLight,
                                      const vec3 &skyTop, const vec3 &skyBottom,
                                      bool useSky) {
    HitInfo h = traceRay(r, meshes, nMeshes);
    if (!h.hit)
        return sampleSky(vec3(r.dir.x, r.dir.y, r.dir.z), skyTop, skyBottom,
                         useSky);

    return calculatePBRLightingCore(h, r, meshes, nMeshes, lights, nLights,
                                    ambientLight, skyTop, skyBottom, useSky,
                                    false);
}

// SCENE CLASS

class Scene {
  private:
    int width;
    int height;

    int bvhLeafTarget_ = 4;
    int bvhLeafTol_ = 2;

    std::vector<std::unique_ptr<Mesh>> meshes;
    std::vector<Material> mesh_materials;
    std::vector<Light> lights;
    Camera camera;

    DeviceMesh *d_mesh_descriptors = nullptr;
    Light *d_lights = nullptr;
    int d_light_count = 0;
    unsigned char *d_pixels = nullptr;

    vec3 ambient_light = vec3(0.1f);
    bool use_sky = true;
    vec3 sky_color_top = vec3(0.6f, 0.7f, 1.0f);
    vec3 sky_color_bottom = vec3(1.0f, 1.0f, 1.0f);

  public:
    Scene(int w, int h)
        : width(w), height(h), camera(static_cast<float>(w) / h, 2.0f, 1.0f) {
        size_t nBytes = static_cast<size_t>(width) * height * 3;
        cudaError_t err = cudaMalloc(&d_pixels, nBytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU pixel buffer");
        }
    }

    ~Scene() {
        if (d_mesh_descriptors)
            cudaFree(d_mesh_descriptors);
        if (d_lights) {
            cudaFree(d_lights);
            d_lights = nullptr;
        }
        d_light_count = 0;
        if (d_pixels)
            cudaFree(d_pixels);
    }

    Scene(const Scene &) = delete;
    Scene &operator=(const Scene &) = delete;

    Mesh *getMesh(size_t index) {
        return (index < meshes.size()) ? meshes[index].get() : nullptr;
    }

    size_t getMeshCount() const { return meshes.size(); }

    void setBVHLeafTarget(int target, int tol = 2) {
        bvhLeafTarget_ = (target < 1 ? 1 : target);
        bvhLeafTol_ = (tol < 0 ? 0 : tol);
        for (auto &m : meshes)
            m->bvhDirty = true;
    }

    void setMeshMaterial(size_t index, const Material &mat) {
        if (index < mesh_materials.size())
            mesh_materials[index] = mat;
    }

    void setCamera(const vec3 &lookfrom, const vec3 &lookat, const vec3 &vup,
                   float vfov, float aperture = 0.0f, float focus_dist = 1.0f) {
        float aspect = static_cast<float>(width) / height;
        camera =
            Camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    }

    void setCameraSimple(float viewport_height = 2.0f,
                         float focal_length = 1.0f) {
        float aspect = static_cast<float>(width) / height;
        camera = Camera(aspect, viewport_height, focal_length);
    }

    vec3 cameraOrigin() const { return camera.get_origin(); }

    vec3 cameraForward() const {
        vec3 dir = camera.get_lower_left_corner() +
                   camera.get_horizontal() * 0.5f +
                   camera.get_vertical() * 0.5f - camera.get_origin();
        return dir.normalized();
    }

    void moveCamera(const vec3 &pos) { camera.set_position(pos); }
    void lookCameraAt(const vec3 &target, const vec3 &vup = vec3(0, 1, 0)) {
        camera.look_at(target, vup);
    }

    Mesh *addMesh(const std::string &obj_path,
                  const Material &mat = Material()) {
        meshes.push_back(std::make_unique<Mesh>(obj_path));
        mesh_materials.push_back(mat);
        return meshes.back().get();
    }

    inline Mesh *addTriangles(const std::vector<Triangle> &tris,
                              const Material &mat = Material()) {
        meshes.push_back(std::make_unique<Mesh>());
        mesh_materials.push_back(mat);
        Mesh *m = meshes.back().get();

        m->vertices.clear();
        m->faces.clear();
        m->vertices.reserve(tris.size() * 3);
        m->faces.reserve(tris.size());

        for (const Triangle &t : tris) {
            const int base = static_cast<int>(m->vertices.size());
            m->vertices.push_back(t.v0);
            m->vertices.push_back(t.v1);
            m->vertices.push_back(t.v2);
            m->faces.push_back(Tri{base + 0, base + 1, base + 2});
        }
        return m;
    }

    inline Mesh *addPlaneXZ(float planeY, float halfSize,
                            const Material &mat = Material(vec3(0.8f))) {
        const vec3 A(-halfSize, planeY, -halfSize);
        const vec3 B(halfSize, planeY, -halfSize);
        const vec3 C(halfSize, planeY, halfSize);
        const vec3 D(-halfSize, planeY, halfSize);

        std::vector<Triangle> tris;
        tris.reserve(2);
        tris.emplace_back(A, C, B);
        tris.emplace_back(A, D, C);
        return addTriangles(tris, mat);
    }

    Mesh *addSphere(int segments = 32,
                    const Material &mat = Material(vec3(1.0f, 0.0f, 0.0f))) {
        auto sphereMesh = std::make_unique<Mesh>();
        sphereMesh->vertices.clear();
        sphereMesh->faces.clear();

        const int rings = segments;
        const int sectors = segments;
        const float radius = 0.5f;

        for (int r = 0; r <= rings; ++r) {
            float phi = PI * float(r) / float(rings);
            float y = cosf(phi) * radius;
            float ringRadius = sinf(phi) * radius;

            for (int s = 0; s <= sectors; ++s) {
                float theta = TWO_PI * float(s) / float(sectors);
                float x = ringRadius * cosf(theta);
                float z = ringRadius * sinf(theta);
                sphereMesh->vertices.push_back(vec3(x, y, z));
            }
        }

        for (int r = 0; r < rings; ++r) {
            for (int s = 0; s < sectors; ++s) {
                int curr = r * (sectors + 1) + s;
                int next = curr + sectors + 1;
                sphereMesh->faces.push_back({curr, next, curr + 1});
                sphereMesh->faces.push_back({curr + 1, next, next + 1});
            }
        }

        meshes.push_back(std::move(sphereMesh));
        mesh_materials.push_back(mat);
        return meshes.back().get();
    }

    inline void addCheckerboardPlaneXZ(float planeY, int tilesPerSide,
                                       float tileSize, const Material &whiteMat,
                                       const Material &blackMat) {
        std::vector<Triangle> whiteTris, blackTris;
        whiteTris.reserve(tilesPerSide * tilesPerSide * 2);
        blackTris.reserve(tilesPerSide * tilesPerSide * 2);

        const int N = tilesPerSide;
        const float start = -N * tileSize;

        for (int iz = 0; iz < 2 * N; ++iz) {
            for (int ix = 0; ix < 2 * N; ++ix) {
                const float x0 = start + ix * tileSize;
                const float x1 = x0 + tileSize;
                const float z0 = start + iz * tileSize;
                const float z1 = z0 + tileSize;

                const vec3 A(x0, planeY, z0), B(x1, planeY, z0);
                const vec3 C(x1, planeY, z1), D(x0, planeY, z1);

                auto &bucket = ((ix + iz) & 1) == 0 ? whiteTris : blackTris;
                bucket.emplace_back(A, C, B);
                bucket.emplace_back(A, D, C);
            }
        }

        if (!whiteTris.empty())
            addTriangles(whiteTris, whiteMat);
        if (!blackTris.empty())
            addTriangles(blackTris, blackMat);
    }

    Mesh *addCube(const Material &mat = Material(vec3(1.0f, 0.0f, 0.0f))) {
        meshes.push_back(std::make_unique<Mesh>());
        mesh_materials.push_back(mat);
        return meshes.back().get();
    }

    void addPointLight(const vec3 &position, const vec3 &color,
                       float intensity = 1.0f, float range = 100.0f) {
        Light light;
        light.type = LIGHT_POINT;
        light.position = position;
        light.color = color;
        light.intensity = intensity;
        light.range = range;
        lights.push_back(light);
    }

    void addDirectionalLight(const vec3 &direction, const vec3 &color,
                             float intensity = 1.0f) {
        Light light;
        light.type = LIGHT_DIRECTIONAL;
        light.direction = direction.normalized();
        light.color = color;
        light.intensity = intensity;
        lights.push_back(light);
    }

    void addSpotLight(const vec3 &position, const vec3 &direction,
                      const vec3 &color, float intensity = 1.0f,
                      float innerCone = 0.5f, float outerCone = 0.7f,
                      float range = 100.0f) {
        Light light;
        light.type = LIGHT_SPOT;
        light.position = position;
        light.direction = direction.normalized();
        light.color = color;
        light.intensity = intensity;
        light.innerCone = cosf(innerCone);
        light.outerCone = cosf(outerCone);
        light.range = range;
        lights.push_back(light);
    }

    void setAmbientLight(const vec3 &ambient) { ambient_light = ambient; }

    void setSkyGradient(const vec3 &top, const vec3 &bottom) {
        sky_color_top = top;
        sky_color_bottom = bottom;
        use_sky = true;
    }

    void disableSky() { use_sky = false; }

    void uploadToGPU() {
        if (meshes.empty()) {
            std::cerr << "Warning: no meshes in scene\n";
            return;
        }

        std::vector<DeviceMesh> h_mesh_desc(meshes.size());

        for (size_t i = 0; i < meshes.size(); ++i) {
            Mesh *m = meshes[i].get();
            m->upload();

            if (m->bvhDirty || m->d_bvhNodes == nullptr) {
                m->setBVHLeafParams(bvhLeafTarget_, bvhLeafTol_);
                m->buildBVH();
                m->uploadBVH();
            }

            DeviceMesh &desc = h_mesh_desc[i];
            desc.verts = m->d_vertices;
            desc.faces = m->d_faces;
            desc.faceCount = static_cast<int>(m->faces.size());
            desc.material = mesh_materials[i];

            mat3 rotX = rotation_x(m->rotationEuler.x);
            mat3 rotY = rotation_y(m->rotationEuler.y);
            mat3 rotZ = rotation_z(m->rotationEuler.z);
            mat3 rotationMat = rotY * rotX * rotZ;
            mat3 invRotationMat = rotationMat.transpose();

            desc.translation = m->position;
            desc.rotation = rotationMat;
            desc.invRotation = invRotationMat;
            desc.bvhNodes = m->d_bvhNodes;
            desc.nodeCount = static_cast<int>(m->bvhNodes.size());
            desc.primIndices = m->d_bvhPrim;
        }

        if (d_mesh_descriptors) {
            cudaFree(d_mesh_descriptors);
            d_mesh_descriptors = nullptr;
        }

        cudaError_t err = cudaMalloc(&d_mesh_descriptors,
                                     h_mesh_desc.size() * sizeof(DeviceMesh));
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to allocate mesh descriptors");

        cudaMemcpy(d_mesh_descriptors, h_mesh_desc.data(),
                   h_mesh_desc.size() * sizeof(DeviceMesh),
                   cudaMemcpyHostToDevice);

        if (lights.empty()) {
            if (d_lights) {
                cudaFree(d_lights);
                d_lights = nullptr;
            }
            d_light_count = 0;
        } else {
            const int want = static_cast<int>(lights.size());
            if (!d_lights || d_light_count != want) {
                if (d_lights) {
                    cudaFree(d_lights);
                    d_lights = nullptr;
                }
                err = cudaMalloc(&d_lights, lights.size() * sizeof(Light));
                if (err != cudaSuccess)
                    throw std::runtime_error("Failed to allocate lights");
                d_light_count = want;
            }
            cudaMemcpy(d_lights, lights.data(), lights.size() * sizeof(Light),
                       cudaMemcpyHostToDevice);
        }
    }

    void render(unsigned char *output_pixels) {
        if (!d_mesh_descriptors || meshes.empty()) {
            std::cerr << "Error: Scene not uploaded to GPU\n";
            return;
        }

        // Optimized block size for better occupancy
        dim3 block(8, 8); // 64 threads per block  good for register pressure
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        render_kernel<<<grid, block>>>(
            d_pixels, width, height, camera, d_mesh_descriptors,
            static_cast<int>(meshes.size()), d_lights,
            static_cast<int>(lights.size()), ambient_light, sky_color_top,
            sky_color_bottom, use_sky);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel launch failed: ") +
                                     cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();

        size_t nBytes = static_cast<size_t>(width) * height * 3;
        err =
            cudaMemcpy(output_pixels, d_pixels, nBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to copy pixels from GPU");
    }

    void render_to_device(unsigned char *device_pixels) {
        if (meshes.empty()) {
            std::cerr << "Error: no meshes in scene\n";
            return;
        }

        std::vector<DeviceMesh> h_mesh_desc(meshes.size());
        for (size_t i = 0; i < meshes.size(); ++i) {
            Mesh *m = meshes[i].get();
            m->upload();

            if (m->bvhDirty || m->d_bvhNodes == nullptr) {
                m->setBVHLeafParams(bvhLeafTarget_, bvhLeafTol_);
                m->buildBVH();
                m->uploadBVH();
            }

            h_mesh_desc[i].verts = m->d_vertices;
            h_mesh_desc[i].faces = m->d_faces;
            h_mesh_desc[i].faceCount = static_cast<int>(m->faces.size());
            h_mesh_desc[i].material = mesh_materials[i];

            mat3 rotX = rotation_x(m->rotationEuler.x);
            mat3 rotY = rotation_y(m->rotationEuler.y);
            mat3 rotZ = rotation_z(m->rotationEuler.z);
            mat3 rotationMat = rotY * rotX * rotZ;
            mat3 invRotationMat = rotationMat.transpose();

            h_mesh_desc[i].translation = m->position;
            h_mesh_desc[i].rotation = rotationMat;
            h_mesh_desc[i].invRotation = invRotationMat;
            h_mesh_desc[i].bvhNodes = m->d_bvhNodes;
            h_mesh_desc[i].nodeCount = static_cast<int>(m->bvhNodes.size());
            h_mesh_desc[i].primIndices = m->d_bvhPrim;
        }

        if (d_mesh_descriptors) {
            cudaFree(d_mesh_descriptors);
            d_mesh_descriptors = nullptr;
        }
        cudaMalloc(&d_mesh_descriptors,
                   sizeof(DeviceMesh) * h_mesh_desc.size());
        cudaMemcpy(d_mesh_descriptors, h_mesh_desc.data(),
                   sizeof(DeviceMesh) * h_mesh_desc.size(),
                   cudaMemcpyHostToDevice);

        if (lights.empty()) {
            if (d_lights) {
                cudaFree(d_lights);
                d_lights = nullptr;
            }
            d_light_count = 0;
        } else {
            const int want = static_cast<int>(lights.size());
            if (!d_lights || d_light_count != want) {
                if (d_lights) {
                    cudaFree(d_lights);
                    d_lights = nullptr;
                }
                cudaMalloc(&d_lights, lights.size() * sizeof(Light));
                d_light_count = want;
            }
            cudaMemcpy(d_lights, lights.data(), lights.size() * sizeof(Light),
                       cudaMemcpyHostToDevice);
        }

        dim3 block(8, 8);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        render_kernel<<<grid, block>>>(
            device_pixels, width, height, camera, d_mesh_descriptors,
            static_cast<int>(meshes.size()), d_lights,
            static_cast<int>(lights.size()), ambient_light, sky_color_top,
            sky_color_bottom, use_sky);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel launch failed: ") +
                                     cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }

    void saveAsPPM(const std::string &filename, unsigned char *pixels) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs)
            throw std::runtime_error("Cannot open file: " + filename);

        ofs << "P3\n" << width << ' ' << height << "\n255\n";
        size_t idx = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ofs << int(pixels[idx]) << ' ' << int(pixels[idx + 1]) << ' '
                    << int(pixels[idx + 2]) << '\n';
                idx += 3;
            }
        }
        ofs.close();
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    size_t getPixelBufferSize() const {
        return static_cast<size_t>(width) * height * 3;
    }
    Camera &getCamera() { return camera; }
};

// RENDER KERNEL

__global__ void render_kernel(unsigned char *out, int W, int H, Camera cam,
                              DeviceMesh *meshes, int nMeshes, Light *lights,
                              int nLights, vec3 ambientLight, vec3 skyColorTop,
                              vec3 skyColorBottom, bool useSky) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    // Compute UV with half pixel offset
    const float invW = 1.0f / W;
    const float invH = 1.0f / H;
    const float u = (x + 0.5f) * invW;
    const float v = 1.0f - (y + 0.5f) * invH;

    // Generate primary ray
    Ray rawRay = cam.get_ray(u, v);
    RayOpt ray(rawRay);

    // Trace
    HitInfo hit = traceRay(ray, meshes, nMeshes);

    vec3 color;
    if (hit.hit) {
        color = calculatePBRLighting(hit, ray, meshes, nMeshes, lights, nLights,
                                     ambientLight, skyColorTop, skyColorBottom,
                                     useSky);
    } else {
        color = useSky ? lerp(skyColorBottom, skyColorTop,
                              0.5f * (ray.dir.y + 1.0f))
                       : vec3(0.0f);
    }

    // Tone mapping (Reinhard) and gamma correction
    color = color / (color + vec3(1.0f));

// Fast gamma using pow intrinsic
#ifdef __CUDA_ARCH__
    color = vec3(__powf(color.x, 0.4545454545f), __powf(color.y, 0.4545454545f),
                 __powf(color.z, 0.4545454545f));
#else
    color = vec3(powf(color.x, 0.4545454545f), powf(color.y, 0.4545454545f),
                 powf(color.z, 0.4545454545f));
#endif

    // Clamp and convert to bytes
    const vec3 rgb = clamp(color, 0.0f, 1.0f) * 255.0f;
    const int y_out = H - 1 - y;
    const size_t idx = (static_cast<size_t>(y_out) * W + x) * 3;

    out[idx + 0] = static_cast<unsigned char>(rgb.x);
    out[idx + 1] = static_cast<unsigned char>(rgb.y);
    out[idx + 2] = static_cast<unsigned char>(rgb.z);
}

// MATERIALS NAMESPACE  All original materials preserved

namespace Materials {

inline Material Gold() {
    Material m(vec3(1.0f, 0.766f, 0.336f), 0.1f, 1.0f);
    m.specular = vec3(1.0f, 0.782f, 0.344f);
    return m;
}

inline Material Silver() {
    Material m(vec3(0.972f, 0.960f, 0.915f), 0.1f, 1.0f);
    m.specular = vec3(0.972f, 0.960f, 0.915f);
    return m;
}

inline Material Copper() {
    Material m(vec3(0.955f, 0.637f, 0.538f), 0.1f, 1.0f);
    m.specular = vec3(0.955f, 0.637f, 0.538f);
    return m;
}

inline Material Bronze() {
    Material m(vec3(0.8f, 0.5f, 0.2f), 0.25f, 0.9f);
    m.specular = vec3(0.7f, 0.4f, 0.15f);
    return m;
}

inline Material Aluminum() {
    Material m(vec3(0.913f, 0.921f, 0.925f), 0.2f, 1.0f);
    m.specular = vec3(0.913f, 0.921f, 0.925f);
    return m;
}

inline Material BrushedAluminum() {
    Material m(vec3(0.913f, 0.921f, 0.925f), 0.35f, 1.0f);
    m.specular = vec3(0.913f, 0.921f, 0.925f);
    m.anisotropy = 0.8f;
    return m;
}

inline Material Glass() {
    Material m(vec3(1.0f), 0.0f, 0.0f);
    m.transmission = 1.0f;
    m.ior = 1.5f;
    m.specular = vec3(0.04f);
    return m;
}

inline Material FrostedGlass() {
    Material m(vec3(1.0f), 0.0f, 0.0f);
    m.transmission = 1.0f;
    m.ior = 1.5f;
    m.transmissionRoughness = 0.3f;
    m.roughness = 0.3f;
    m.specular = vec3(0.04f);
    return m;
}

inline Material Diamond() {
    Material m(vec3(1.0f), 0.0f, 0.0f);
    m.transmission = 1.0f;
    m.ior = 2.42f;
    m.specular = vec3(0.17f);
    return m;
}

inline Material Water() {
    Material m(vec3(0.8f, 0.95f, 1.0f), 0.01f, 0.0f);
    m.transmission = 0.9f;
    m.ior = 1.33f;
    m.specular = vec3(0.02f);
    return m;
}

inline Material PlasticRed() {
    Material m(vec3(0.8f, 0.1f, 0.1f), 0.2f, 0.0f);
    m.specular = vec3(0.04f);
    return m;
}

inline Material RubberBlack() {
    Material m(vec3(0.05f), 0.8f, 0.0f);
    m.specular = vec3(0.03f);
    return m;
}

inline Material CarPaint(const vec3 &baseColor) {
    Material m(baseColor, 0.2f, 0.3f);
    m.clearcoat = 1.0f;
    m.clearcoatRoughness = 0.03f;
    m.specular = vec3(0.05f);
    return m;
}

inline Material PearlescentPaint(const vec3 &baseColor) {
    Material m = CarPaint(baseColor);
    m.iridescence = 0.8f;
    m.iridescenceThickness = 400.0f;
    return m;
}

inline Material Skin() {
    Material m(vec3(0.95f, 0.75f, 0.67f), 0.4f, 0.0f);
    m.subsurfaceColor = vec3(1.0f, 0.4f, 0.3f);
    m.subsurfaceRadius = 0.5f;
    m.specular = vec3(0.028f);
    return m;
}

inline Material Wax() {
    Material m(vec3(0.95f, 0.93f, 0.88f), 0.3f, 0.0f);
    m.subsurfaceColor = vec3(1.0f, 0.9f, 0.7f);
    m.subsurfaceRadius = 0.8f;
    m.specular = vec3(0.03f);
    return m;
}

inline Material Jade() {
    Material m(vec3(0.2f, 0.6f, 0.4f), 0.1f, 0.0f);
    m.subsurfaceColor = vec3(0.3f, 0.8f, 0.5f);
    m.subsurfaceRadius = 0.3f;
    m.specular = vec3(0.05f);
    return m;
}

inline Material Velvet(const vec3 &color) {
    Material m(color, 0.8f, 0.0f);
    m.sheen = 1.0f;
    m.sheenTint = color * 1.2f;
    m.specular = vec3(0.02f);
    return m;
}

inline Material Silk(const vec3 &color) {
    Material m(color, 0.2f, 0.0f);
    m.sheen = 0.6f;
    m.sheenTint = vec3(1.0f);
    m.anisotropy = 0.5f;
    m.specular = vec3(0.04f);
    return m;
}

inline Material SoapBubble() {
    Material m(vec3(1.0f), 0.0f, 0.0f);
    m.transmission = 0.95f;
    m.ior = 1.33f;
    m.iridescence = 1.0f;
    m.iridescenceThickness = 380.0f;
    m.specular = vec3(0.04f);
    return m;
}

inline Material OilSlick() {
    Material m(vec3(0.01f), 0.0f, 0.95f);
    m.iridescence = 1.0f;
    m.iridescenceThickness = 450.0f;
    return m;
}

inline Material EmissiveLamp(const vec3 &color, float intensity = 5.0f) {
    Material m(vec3(1.0f), 0.0f, 0.0f);
    m.emission = color * intensity;
    return m;
}

inline Material NeonLight(const vec3 &color) {
    Material m(color * 0.1f, 0.0f, 0.0f);
    m.emission = color * 10.0f;
    return m;
}

__host__ __device__ inline Material MarbleCarrara(bool polished = true) {
    const float baseRough = polished ? 0.15f : 0.35f;
    const float coatAmt = polished ? 0.70f : 0.15f;
    const float coatRough = polished ? 0.05f : 0.20f;

    Material m(vec3(0.93f, 0.94f, 0.96f), baseRough, 0.0f);
    m.ior = 1.49f;
    m.clearcoat = coatAmt;
    m.clearcoatRoughness = coatRough;
    m.subsurfaceColor = vec3(0.98f, 0.98f, 0.96f);
    m.subsurfaceRadius = 1.0f;
    m.transmission = 0.0f;
    m.transmissionRoughness = 0.0f;
    m.anisotropy = 0.0f;
    m.sheen = 0.0f;
    m.sheenTint = vec3(0.5f);
    m.iridescence = 0.0f;
    m.iridescenceThickness = 550.0f;
    return m;
}

__host__ __device__ inline Material MarbleNero(bool polished = true) {
    const float baseRough = polished ? 0.12f : 0.28f;
    const float coatAmt = polished ? 0.85f : 0.20f;
    const float coatRough = polished ? 0.04f : 0.18f;

    Material m(vec3(0.04f, 0.045f, 0.05f), baseRough, 0.0f);
    m.ior = 1.49f;
    m.clearcoat = coatAmt;
    m.clearcoatRoughness = coatRough;
    m.subsurfaceColor = vec3(0.15f, 0.15f, 0.16f);
    m.subsurfaceRadius = 0.6f;
    m.transmission = 0.0f;
    m.transmissionRoughness = 0.0f;
    m.anisotropy = 0.0f;
    m.sheen = 0.0f;
    m.iridescence = 0.0f;
    return m;
}

__host__ __device__ inline Material MarbleVerde(bool polished = true) {
    const float baseRough = polished ? 0.14f : 0.30f;
    const float coatAmt = polished ? 0.75f : 0.18f;
    const float coatRough = polished ? 0.05f : 0.19f;

    Material m(vec3(0.10f, 0.18f, 0.14f), baseRough, 0.0f);
    m.ior = 1.49f;
    m.clearcoat = coatAmt;
    m.clearcoatRoughness = coatRough;
    m.subsurfaceColor = vec3(0.12f, 0.20f, 0.16f);
    m.subsurfaceRadius = 0.8f;
    m.transmission = 0.0f;
    m.transmissionRoughness = 0.0f;
    m.anisotropy = 0.0f;
    m.sheen = 0.0f;
    m.iridescence = 0.0f;
    return m;
}

inline Material Iron() {
    Material m(vec3(0.560f, 0.570f, 0.580f), 0.4f, 1.0f);
    m.specular = vec3(0.560f, 0.570f, 0.580f);
    return m;
}

inline Material Chrome() {
    Material m(vec3(0.549f, 0.556f, 0.554f), 0.02f, 1.0f);
    m.specular = vec3(0.549f, 0.556f, 0.554f);
    return m;
}

inline Material Ice() {
    Material m(vec3(0.9f, 0.95f, 1.0f), 0.1f, 0.0f);
    m.transmission = 0.7f;
    m.ior = 1.31f;
    m.subsurfaceColor = vec3(0.8f, 0.9f, 1.0f);
    m.subsurfaceRadius = 0.3f;
    return m;
}

inline Material PlasticBlue() {
    Material m(vec3(0.1f, 0.2f, 0.8f), 0.2f, 0.0f);
    m.specular = vec3(0.04f);
    return m;
}

inline Material PlasticGreen() {
    Material m(vec3(0.1f, 0.7f, 0.2f), 0.2f, 0.0f);
    m.specular = vec3(0.04f);
    return m;
}

inline Material Cotton(const vec3 &color) {
    Material m(color, 0.9f, 0.0f);
    m.specular = vec3(0.02f);
    return m;
}

inline Material Concrete() {
    Material m(vec3(0.5f, 0.5f, 0.5f), 0.9f, 0.0f);
    m.specular = vec3(0.02f);
    return m;
}

inline Material WoodOak() {
    Material m(vec3(0.6f, 0.4f, 0.2f), 0.5f, 0.0f);
    m.specular = vec3(0.04f);
    return m;
}

inline Material WoodCherry() {
    Material m(vec3(0.5f, 0.2f, 0.1f), 0.4f, 0.0f);
    m.clearcoat = 0.3f;
    m.clearcoatRoughness = 0.1f;
    return m;
}

inline Material WoodWalnut() {
    Material m(vec3(0.3f, 0.2f, 0.15f), 0.45f, 0.0f);
    m.specular = vec3(0.04f);
    return m;
}

} // namespace Materials

// EXAMPLE SCENES

namespace Scenes {

inline std::unique_ptr<Scene> createLitTestScene(int width = 800,
                                                 int height = 600) {
    auto scene = std::make_unique<Scene>(width, height);

    Material redMat(vec3(0.8f, 0.2f, 0.2f), 0.2f);
    redMat.specular = vec3(0.5f);

    Material blueMat(vec3(0.2f, 0.2f, 0.8f), 0.3f);
    blueMat.specular = vec3(0.3f);

    Material goldMat(vec3(0.9f, 0.7f, 0.3f), 0.15f, 1.0f);
    goldMat.specular = vec3(0.8f, 0.6f, 0.2f);

    Mesh *cube = scene->addCube(redMat);
    cube->moveTo(vec3(-2, 0, -5));
    cube->scale(0.8f);

    Mesh *cube2 = scene->addCube(blueMat);
    cube2->moveTo(vec3(2, 0, -5));
    cube2->scale(0.8f);

    Mesh *cube3 = scene->addCube(goldMat);
    cube3->moveTo(vec3(0, 2, -5));
    cube3->scale(0.8f);

    scene->addPointLight(vec3(5, 5, 0), vec3(1.0f, 0.9f, 0.8f), 2.0f, 50.0f);
    scene->addDirectionalLight(vec3(-0.3f, -0.8f, -0.5f),
                               vec3(0.9f, 0.9f, 1.0f), 0.5f);
    scene->addSpotLight(vec3(0, 4, -2), vec3(0, -1, -0.3f),
                        vec3(1.0f, 0.8f, 0.6f), 3.0f, 0.3f, 0.5f, 20.0f);

    scene->setAmbientLight(vec3(0.05f, 0.05f, 0.08f));
    scene->setSkyGradient(vec3(0.5f, 0.6f, 0.9f), vec3(0.9f, 0.9f, 0.95f));

    return scene;
}

} // namespace Scenes

#endif
