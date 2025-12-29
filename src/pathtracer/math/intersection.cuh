// intersection.cuh
// ray geometry intersection utilities and bvh traversal helpers for cuda path
// tracing contains optimized ray representations aabb tests triangle tests and
// mesh traversal entry points

#ifndef INTERSECTION_CUH
#define INTERSECTION_CUH

#include "common/mat4.cuh"
#include "common/ray.cuh"
#include "common/triangle.cuh"
#include "common/vec3.cuh"
#include "pathtracer/scene/material_lib.cuh"
#include "pathtracer/scene/mesh.cuh"
#include "pathtracer/scene/transform.cuh"

constexpr int BVH_STACK_SIZE = 24;

constexpr float MAX_RAY_DISTANCE = 1e30f;

// loads a vec3 through the readonly cache when available
__device__ __forceinline__ vec3 ldg_vec3_inter(const vec3 *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return vec3(__ldg(&ptr->x), __ldg(&ptr->y), __ldg(&ptr->z));
#else
    return *ptr;
#endif
}

// loads a triangle index triple through the readonly cache when available
__device__ __forceinline__ Tri ldg_tri(const Tri *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return Tri{__ldg(&ptr->v0), __ldg(&ptr->v1), __ldg(&ptr->v2)};
#else
    return *ptr;
#endif
}

struct RayOptimized {
    vec3 origin;
    vec3 direction;
    vec3 invDirection;
    int dirSign[3];

    // constructs a cached ray representation with inverse direction and sign
    // bits for slab tests
    __device__ __forceinline__ RayOptimized(const Ray &r) {
        origin = r.origin();
        direction = r.direction();

        invDirection.x = (fabsf(direction.x) > 1e-8f)
                             ? __frcp_rn(direction.x)
                             : ((direction.x >= 0) ? 1e30f : -1e30f);
        invDirection.y = (fabsf(direction.y) > 1e-8f)
                             ? __frcp_rn(direction.y)
                             : ((direction.y >= 0) ? 1e30f : -1e30f);
        invDirection.z = (fabsf(direction.z) > 1e-8f)
                             ? __frcp_rn(direction.z)
                             : ((direction.z >= 0) ? 1e30f : -1e30f);

        dirSign[0] = invDirection.x < 0 ? 1 : 0;
        dirSign[1] = invDirection.y < 0 ? 1 : 0;
        dirSign[2] = invDirection.z < 0 ? 1 : 0;
    }

    // constructs a cached ray representation with inverse direction and sign
    // bits for slab tests
    __device__ __forceinline__ RayOptimized(const vec3 &o, const vec3 &d) {
        origin = o;
        direction = d;

        invDirection.x = (fabsf(d.x) > 1e-8f) ? __frcp_rn(d.x)
                                              : ((d.x >= 0) ? 1e30f : -1e30f);
        invDirection.y = (fabsf(d.y) > 1e-8f) ? __frcp_rn(d.y)
                                              : ((d.y >= 0) ? 1e30f : -1e30f);
        invDirection.z = (fabsf(d.z) > 1e-8f) ? __frcp_rn(d.z)
                                              : ((d.z >= 0) ? 1e30f : -1e30f);

        dirSign[0] = invDirection.x < 0 ? 1 : 0;
        dirSign[1] = invDirection.y < 0 ? 1 : 0;
        dirSign[2] = invDirection.z < 0 ? 1 : 0;
    }

    // computes a point along a ray at parameter t
    __device__ __forceinline__ vec3 at(float t) const {
        return origin + t * direction;
    }
};

struct DeviceMesh {
    vec3 *verts;
    Tri *faces;
    int faceCount;

    DeviceBVHNode *bvhNodes;
    int nodeCount;
    int *primIndices;

    mat4 worldMatrix;
    mat4 inverseMatrix;
    mat4 normalMatrix;
    AABB worldAABB;
    bool hasTransform;

    mat4 prevWorldMatrix;
};

struct HitInfo {
    bool hit;
    float t;
    vec3 point;
    vec3 normal;
    int mesh_index;
    bool front_face;

    float u;
    float v;
    int face_index;

    vec3 localPoint;

    __host__ __device__ HitInfo()
        : hit(false), t(MAX_RAY_DISTANCE), mesh_index(-1), front_face(true),
          u(0.0f), v(0.0f), face_index(-1), localPoint(0.0f) {}

    // orients a shading normal to oppose the incoming ray direction
    __device__ __forceinline__ void
    set_face_normal(const vec3 &rayDir, const vec3 &outward_normal) {
        front_face = dot(rayDir, outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// tests a ray against an axis aligned bounding box using a slab method with
// early out
__device__ __forceinline__ bool
aabb_hit_fast(const AABB &box, const RayOptimized &ray, float tMax) {
    float t0x = (box.bmin.x - ray.origin.x) * ray.invDirection.x;
    float t1x = (box.bmax.x - ray.origin.x) * ray.invDirection.x;
    if (ray.dirSign[0]) {
        float tmp = t0x;
        t0x = t1x;
        t1x = tmp;
    }

    float t0y = (box.bmin.y - ray.origin.y) * ray.invDirection.y;
    float t1y = (box.bmax.y - ray.origin.y) * ray.invDirection.y;
    if (ray.dirSign[1]) {
        float tmp = t0y;
        t0y = t1y;
        t1y = tmp;
    }

    float tmin = fmaxf(t0x, t0y);
    float tmax = fminf(t1x, t1y);

    if (tmin > tmax)
        return false;

    float t0z = (box.bmin.z - ray.origin.z) * ray.invDirection.z;
    float t1z = (box.bmax.z - ray.origin.z) * ray.invDirection.z;
    if (ray.dirSign[2]) {
        float tmp = t0z;
        t0z = t1z;
        t1z = tmp;
    }

    tmin = fmaxf(tmin, t0z);
    tmax = fminf(tmax, t1z);

    return (tmax >= 0.0f) && (tmin <= tmax) && (tmin < tMax);
}

// tests a ray against an aabb and also returns the entry distance tmin
__device__ __forceinline__ bool aabb_hit_fast_t(const AABB &box,
                                                const RayOptimized &ray,
                                                float tMax, float &tHit) {
    float t0x = (box.bmin.x - ray.origin.x) * ray.invDirection.x;
    float t1x = (box.bmax.x - ray.origin.x) * ray.invDirection.x;
    if (ray.dirSign[0]) {
        float tmp = t0x;
        t0x = t1x;
        t1x = tmp;
    }

    float t0y = (box.bmin.y - ray.origin.y) * ray.invDirection.y;
    float t1y = (box.bmax.y - ray.origin.y) * ray.invDirection.y;
    if (ray.dirSign[1]) {
        float tmp = t0y;
        t0y = t1y;
        t1y = tmp;
    }

    float tmin = fmaxf(t0x, t0y);
    float tmax = fminf(t1x, t1y);

    if (tmin > tmax)
        return false;

    float t0z = (box.bmin.z - ray.origin.z) * ray.invDirection.z;
    float t1z = (box.bmax.z - ray.origin.z) * ray.invDirection.z;
    if (ray.dirSign[2]) {
        float tmp = t0z;
        t0z = t1z;
        t1z = tmp;
    }

    tmin = fmaxf(tmin, t0z);
    tmax = fminf(tmax, t1z);

    if ((tmax < 0.0f) || (tmin > tmax) || (tmin >= tMax))
        return false;

    tHit = fmaxf(tmin, 0.0f);
    return true;
}

// intersects a ray with a single triangle returning barycentrics and distance
__device__ __forceinline__ bool
triangle_intersect_fast(const vec3 &v0, const vec3 &v1, const vec3 &v2,
                        const RayOptimized &ray, float tMax, float &t_out,
                        float &u_out, float &v_out) {

    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 h = cross(ray.direction, e2);
    float a = dot(e1, h);

    if (fabsf(a) < EPSILON)
        return false;

    float f = 1.0f / a;
    vec3 s = ray.origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return false;

    vec3 q = cross(s, e1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = f * dot(e2, q);

    if (t > EPSILON && t < tMax) {
        t_out = t;
        u_out = u;
        v_out = v;
        return true;
    }

    return false;
}

// transforms a point by a matrix treating it as position with w=1
__device__ __host__ __forceinline__ vec3 transformPoint(const mat4 &m,
                                                        const vec3 &p) {
    return vec3(m.m[0] * p.x + m.m[1] * p.y + m.m[2] * p.z + m.m[3],
                m.m[4] * p.x + m.m[5] * p.y + m.m[6] * p.z + m.m[7],
                m.m[8] * p.x + m.m[9] * p.y + m.m[10] * p.z + m.m[11]);
}

// transforms a direction by a matrix treating it as vector with w=0
__device__ __host__ __forceinline__ vec3 transformDirection(const mat4 &m,
                                                            const vec3 &d) {
    return vec3(m.m[0] * d.x + m.m[1] * d.y + m.m[2] * d.z,
                m.m[4] * d.x + m.m[5] * d.y + m.m[6] * d.z,
                m.m[8] * d.x + m.m[9] * d.y + m.m[10] * d.z);
}

// transforms a normal by the inverse transpose matrix
__device__ __host__ __forceinline__ vec3 transformNormal(const mat4 &normalMat,
                                                         const vec3 &n) {
    vec3 transformed(
        normalMat.m[0] * n.x + normalMat.m[1] * n.y + normalMat.m[2] * n.z,
        normalMat.m[4] * n.x + normalMat.m[5] * n.y + normalMat.m[6] * n.z,
        normalMat.m[8] * n.x + normalMat.m[9] * n.y + normalMat.m[10] * n.z);
    return normalize(transformed);
}

// transforms a world space ray into a local space ray for instanced meshes
__device__ __forceinline__ RayOptimized
transformRayToLocal(const RayOptimized &worldRay, const mat4 &invMatrix) {
    vec3 localOrigin = transformPoint(invMatrix, worldRay.origin);
    vec3 localDir = transformDirection(invMatrix, worldRay.direction);
    return RayOptimized(localOrigin, normalize(localDir));
}

// computes how local space scaling affects ray direction length for tmax
// adjustment
__device__ __forceinline__ float getDirectionScale(const mat4 &invMatrix,
                                                   const vec3 &worldDir) {
    vec3 localDir = transformDirection(invMatrix, worldDir);
    return localDir.length();
}

// traverses a mesh bvh and returns true on the first hit within tmax
__device__ __forceinline__ bool bvh_any_hit_local(const RayOptimized &localRay,
                                                  const DeviceMesh &M,
                                                  float tMax) {
    if (!M.bvhNodes || M.nodeCount == 0)
        return false;

    int stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int ni = stack[--sp];
        const DeviceBVHNode &N = M.bvhNodes[ni];

        if (!aabb_hit_fast(N.bbox, localRay, tMax))
            continue;

        if (N.count > 0) {
            for (int i = 0; i < N.count; ++i) {
                int fidx = M.primIndices[N.start + i];

                const Tri idx = ldg_tri(&M.faces[fidx]);
                vec3 v0 = ldg_vec3_inter(&M.verts[idx.v0]);
                vec3 v1 = ldg_vec3_inter(&M.verts[idx.v1]);
                vec3 v2 = ldg_vec3_inter(&M.verts[idx.v2]);

                float t, u, v;
                if (triangle_intersect_fast(v0, v1, v2, localRay, tMax, t, u,
                                            v)) {
                    if (t > 1e-5f)
                        return true;
                }
            }
        } else {
            if (N.right >= 0 && sp < BVH_STACK_SIZE)
                stack[sp++] = N.right;
            if (N.left >= 0 && sp < BVH_STACK_SIZE)
                stack[sp++] = N.left;
        }
    }
    return false;
}

// traverses a mesh bvh and returns the closest hit information
__device__ __forceinline__ HitInfo bvh_trace_local(const RayOptimized &localRay,
                                                   const DeviceMesh &M) {
    HitInfo out;
    out.hit = false;
    out.t = MAX_RAY_DISTANCE;

    if (!M.bvhNodes || M.nodeCount == 0 || !M.primIndices)
        return out;

    int stack[BVH_STACK_SIZE];
    int sp = 0;
    int ni = 0;

    while (true) {
        const DeviceBVHNode &N = M.bvhNodes[ni];

        if (!aabb_hit_fast(N.bbox, localRay, out.t)) {
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        if (N.count > 0) {
            for (int i = 0; i < N.count; ++i) {
                const int fidx = M.primIndices[N.start + i];

                const Tri triIdx = ldg_tri(&M.faces[fidx]);
                vec3 v0 = ldg_vec3_inter(&M.verts[triIdx.v0]);
                vec3 v1 = ldg_vec3_inter(&M.verts[triIdx.v1]);
                vec3 v2 = ldg_vec3_inter(&M.verts[triIdx.v2]);

                float tHit, uHit, vHit;
                if (triangle_intersect_fast(v0, v1, v2, localRay, out.t, tHit,
                                            uHit, vHit)) {
                    if (tHit > 1e-5f) {
                        out.hit = true;
                        out.t = tHit;
                        out.localPoint = localRay.at(tHit);
                        out.point = out.localPoint;

                        vec3 e1 = v1 - v0;
                        vec3 e2 = v2 - v0;
                        vec3 geom_normal = normalize(cross(e1, e2));
                        out.set_face_normal(localRay.direction, geom_normal);

                        out.u = uHit;
                        out.v = vHit;
                        out.face_index = fidx;
                    }
                }
            }

            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        const int L = N.left, R = N.right;
        float tL = 0.f, tR = 0.f;
        bool hL = (L >= 0) &&
                  aabb_hit_fast_t(M.bvhNodes[L].bbox, localRay, out.t, tL);
        bool hR = (R >= 0) &&
                  aabb_hit_fast_t(M.bvhNodes[R].bbox, localRay, out.t, tR);

        if (!hL && !hR) {
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        int nearIdx, farIdx;
        bool hitFar;
        if (hL && (!hR || tL <= tR)) {
            nearIdx = L;
            farIdx = R;
            hitFar = hR;
        } else {
            nearIdx = R;
            farIdx = L;
            hitFar = hL;
        }

        if (hitFar && sp < BVH_STACK_SIZE) {
            stack[sp++] = farIdx;
        }
        ni = nearIdx;
    }
    return out;
}

// tests a world ray against a mesh instance including local transforms
__device__ __forceinline__ bool bvh_any_hit(const RayOptimized &worldRay,
                                            const DeviceMesh &M,
                                            float worldTMax) {
    if (!M.hasTransform) {
        return bvh_any_hit_local(worldRay, M, worldTMax);
    }

    RayOptimized localRay = transformRayToLocal(worldRay, M.inverseMatrix);
    float dirScale = getDirectionScale(M.inverseMatrix, worldRay.direction);
    float localTMax = worldTMax * dirScale;

    return bvh_any_hit_local(localRay, M, localTMax);
}

// traces a world ray against a mesh instance and returns the closest hit in
// world space
__device__ __forceinline__ HitInfo bvh_trace(const RayOptimized &worldRay,
                                             const DeviceMesh &M) {
    HitInfo result;

    if (!M.hasTransform) {
        return bvh_trace_local(worldRay, M);
    }

    RayOptimized localRay = transformRayToLocal(worldRay, M.inverseMatrix);
    result = bvh_trace_local(localRay, M);

    if (result.hit) {
        result.point = transformPoint(M.worldMatrix, result.localPoint);

        float dirScale = getDirectionScale(M.inverseMatrix, worldRay.direction);
        result.t = result.t / dirScale;

        vec3 localNormal = result.normal;
        vec3 worldNormal = transformNormal(M.normalMatrix, localNormal);

        result.front_face = dot(worldRay.direction, worldNormal) < 0.0f;
        result.normal = result.front_face ? worldNormal : -worldNormal;
    }

    return result;
}

__device__ inline bool bvh_any_hit_tlas(const Ray &ray, float tMax,
                                        DeviceMesh *meshes,
                                        DeviceBVHNode *tlasNodes,
                                        int *tlasMeshIndices,
                                        const DeviceMaterials &materials) {
    if (!tlasNodes)
        return false;

    RayOptimized optRay(ray);

    if (!aabb_hit_fast(tlasNodes[0].bbox, optRay, tMax))
        return false;

    int stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int ni = stack[--sp];
        const DeviceBVHNode &N = tlasNodes[ni];

        if (!aabb_hit_fast(N.bbox, optRay, tMax))
            continue;

        if (N.count > 0) {
            for (int i = 0; i < N.count; ++i) {
                int mesh_id = tlasMeshIndices[N.start + i];

                float trans = __ldg(&materials.transmission[mesh_id]);
                if (trans > 0.5f)
                    continue;

                if (bvh_any_hit(optRay, meshes[mesh_id], tMax))
                    return true;
            }
        } else {
            if (N.right >= 0 && sp < BVH_STACK_SIZE)
                stack[sp++] = N.right;
            if (N.left >= 0 && sp < BVH_STACK_SIZE)
                stack[sp++] = N.left;
        }
    }
    return false;
}

__device__ inline HitInfo traceRay(const Ray &ray, DeviceMesh *meshes,
                                   DeviceBVHNode *tlasNodes,
                                   int *tlasMeshIndices) {

    HitInfo best;
    best.hit = false;
    best.t = MAX_RAY_DISTANCE;

    if (!tlasNodes)
        return best;

    RayOptimized optRay(ray);

    if (!aabb_hit_fast(tlasNodes[0].bbox, optRay, best.t))
        return best;

    int stack[BVH_STACK_SIZE];
    int sp = 0;
    int ni = 0;

    while (true) {
        const DeviceBVHNode &N = tlasNodes[ni];

        if (!aabb_hit_fast(N.bbox, optRay, best.t)) {
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        if (N.count > 0) {
            for (int i = 0; i < N.count; ++i) {
                int mesh_id = tlasMeshIndices[N.start + i];
                HitInfo h = bvh_trace(optRay, meshes[mesh_id]);

                if (h.hit && h.t < best.t) {
                    best = h;
                    best.mesh_index = mesh_id;
                }
            }
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        const int L = N.left, R = N.right;
        float tL = 0.f, tR = 0.f;

        bool hL =
            (L >= 0) && aabb_hit_fast_t(tlasNodes[L].bbox, optRay, best.t, tL);
        bool hR =
            (R >= 0) && aabb_hit_fast_t(tlasNodes[R].bbox, optRay, best.t, tR);

        if (!hL && !hR) {
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        int nearIdx, farIdx;
        bool hitFar;
        if (hL && (!hR || tL <= tR)) {
            nearIdx = L;
            farIdx = R;
            hitFar = hR;
        } else {
            nearIdx = R;
            farIdx = L;
            hitFar = hL;
        }

        if (hitFar && sp < BVH_STACK_SIZE) {
            stack[sp++] = farIdx;
        }
        ni = nearIdx;
    }
    return best;
}

#endif // INTERSECTION_CUH
