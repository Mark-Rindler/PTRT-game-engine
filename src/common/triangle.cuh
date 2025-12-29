#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "common\ray.cuh"
#include "common\vec3.cuh"
#include <cmath>
#include <cuda_runtime.h>

#ifndef RT_CULL_BACKFACES
#define RT_CULL_BACKFACES                                                      \
    0 // set to 0 if you want two-sided hits (this is only if you use the old
      // intersection code, the new intersection code is always two-sided)
#endif

struct Triangle {
    vec3 v0, v1, v2; // vertices
    vec3 e1, e2;     // pre‑computed edges (v1‑v0, v2‑v0)
    vec3 n;          // geometric normal (not unit length)

    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(const vec3 &_v0, const vec3 &_v1,
                                 const vec3 &_v2)
        : v0(_v0), v1(_v1), v2(_v2) {
        e1 = v1 - v0;
        e2 = v2 - v0;
        n = cross(e1, e2); // outward normal (not normalized)
    }

    /** Return a *unit‑length* normal. */
    __forceinline__ __host__ __device__ vec3 normal() const {
        return normalize(n);
    }

    /** Triangle area = 0.5 * |n| */
    __forceinline__ __host__ __device__ float area() const {
        return 0.5f * length(n);
    }

    /** Axis‑aligned bounding box (AABB) */
    __forceinline__ __host__ __device__ void bounds(vec3 &bmin,
                                                    vec3 &bmax) const {
        bmin.x = fminf(v0.x, fminf(v1.x, v2.x));
        bmin.y = fminf(v0.y, fminf(v1.y, v2.y));
        bmin.z = fminf(v0.z, fminf(v1.z, v2.z));

        bmax.x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
        bmax.y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
        bmax.z = fmaxf(v0.z, fmaxf(v1.z, v2.z));
    }

    /**
     * Möller–Trumbore ray‑triangle intersection test.
     * @param ray   Ray to test against
     * @param t     (out) distance along the ray to the hit point
     * @param u,v   (out) barycentric coordinates of the hit point
     * @return      true if the ray hits the front face of the triangle
     */
    __forceinline__ __host__ __device__ bool
    intersect(const Ray &ray, float &t, float &u, float &v) const {
        const float EPS = 1e-6f;

        vec3 pvec = cross(ray.direction(), e2);
        float det = dot(e1, pvec);

#if RT_CULL_BACKFACES
        // Front-face only: negative det => backface (or parallel)
        if (det <= EPS)
            return false;
#else
        // Two-sided
        if (fabsf(det) < EPS)
            return false;
#endif

        float invDet = 1.0f / det;

        vec3 tvec = ray.origin() - v0;
        u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f)
            return false;

        vec3 qvec = cross(tvec, e1);
        v = dot(ray.direction(), qvec) * invDet;
        if (v < 0.0f || (u + v) > 1.0f)
            return false;

        t = dot(e2, qvec) * invDet;
        return t > EPS;
    }
};

#endif