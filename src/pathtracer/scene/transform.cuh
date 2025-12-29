// transform cuh
// axis aligned bounding boxes and object transform helpers
// provides robust ray aabb tests and matrix based transform utilities
// used for dynamic scenes where transforms change without rebuilding geometry
#ifndef TRANSFORM_CUH
#define TRANSFORM_CUH

#include "common/mat4.cuh"
#include "common/ray.cuh"
#include "common/vec3.cuh"
#include <cmath>
#include <cuda_runtime.h>

struct AABB {
    vec3 bmin;
    vec3 bmax;

    // hit
    // tests intersection between ray and bounds and updates interval
    // inputs r tmax
    // returns bool

    __host__ __device__ inline bool hit(const Ray &r, float tMax) const {
        float tmin = 1e-5f, tmax = tMax;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            const float dir = r.direction()[a];
            const float orig = r.origin()[a];
            // handle near zero direction to avoid inf and nan
            if (fabsf(dir) < 1e-12f) {
                if (orig < bmin[a] || orig > bmax[a])
                    return false;
                continue;
            }
            const float invD = 1.0f / dir;
            float t0 = (bmin[a] - orig) * invD;
            float t1 = (bmax[a] - orig) * invD;
            if (invD < 0.0f) {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
            if (tmax <= tmin)
                return false;
        }
        return true;
    }

    // hit t
    // tests intersection between ray and bounds and updates interval
    // inputs r tmax tenter
    // returns bool

    __host__ __device__ inline bool hit_t(const Ray &r, float tMax,
                                          float &tEnter) const {
        float tmin = 1e-5f, tmax = tMax;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            const float dir = r.direction()[a];
            const float orig = r.origin()[a];
            // handle near zero direction to avoid inf and nan
            if (fabsf(dir) < 1e-12f) {
                if (orig < bmin[a] || orig > bmax[a])
                    return false;
                continue;
            }
            const float invD = 1.0f / dir;
            float t0 = (bmin[a] - orig) * invD;
            float t1 = (bmax[a] - orig) * invD;
            if (invD < 0.0f) {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
            if (tmax <= tmin)
                return false;
        }
        tEnter = tmin;
        return true;
    }

    // extent
    // implements a unit of behavior used by higher level scene code
    // inputs none
    // returns vec3

    __host__ __device__ inline vec3 extent() const { return bmax - bmin; }
    // center
    // implements a unit of behavior used by higher level scene code
    // inputs none
    // returns vec3

    __host__ __device__ inline vec3 center() const {
        return (bmin + bmax) * 0.5f;
    }
    // radius
    // implements a unit of behavior used by higher level scene code
    // inputs none
    // returns float

    __host__ __device__ inline float radius() const {
        return 0.5f * extent().length();
    }

    // make invalid
    // implements a unit of behavior used by higher level scene code
    // inputs none
    // returns aabb

    __host__ __device__ static inline AABB make_invalid() {

        return {vec3(1e30f), vec3(-1e30f)};
    }

    // expand
    // implements a unit of behavior used by higher level scene code
    // inputs b
    // returns none

    __host__ __device__ inline void expand(const AABB &b) {
        bmin.x = fminf(bmin.x, b.bmin.x);
        bmin.y = fminf(bmin.y, b.bmin.y);
        bmin.z = fminf(bmin.z, b.bmin.z);
        bmax.x = fmaxf(bmax.x, b.bmax.x);
        bmax.y = fmaxf(bmax.y, b.bmax.y);
        bmax.z = fmaxf(bmax.z, b.bmax.z);
    }

    // expand
    // implements a unit of behavior used by higher level scene code
    // inputs p
    // returns none

    __host__ __device__ inline void expand(const vec3 &p) {
        bmin.x = fminf(bmin.x, p.x);
        bmin.y = fminf(bmin.y, p.y);
        bmin.z = fminf(bmin.z, p.z);
        bmax.x = fmaxf(bmax.x, p.x);
        bmax.y = fmaxf(bmax.y, p.y);
        bmax.z = fmaxf(bmax.z, p.z);
    }
};

struct Transform3D {
    vec3 position;
    vec3 rotation;
    vec3 scale;

    mat4 worldMatrix;
    mat4 inverseMatrix;
    mat4 normalMatrix;

    bool dirty;

    // transform3d
    // implements a unit of behavior used by higher level scene code
    // inputs none
    // returns device

    __host__ __device__ Transform3D()
        // position
        // implements a unit of behavior used by higher level scene code
        // inputs 0 0f rotation 0 0f scale 1 0f dirty true
        // returns value

        : position(0.0f), rotation(0.0f), scale(1.0f), dirty(true) {
        updateMatrices();
    }

    // transform3d
    // implements a unit of behavior used by higher level scene code
    // inputs pos rot scl
    // returns device

    __host__ __device__ Transform3D(vec3 pos, vec3 rot = vec3(0.0f),
                                    // vec3
                                    // implements a unit of behavior used by
                                    // higher level scene code inputs 1 0f
                                    // returns value

                                    vec3 scl = vec3(1.0f))
        // position
        // implements a unit of behavior used by higher level scene code
        // inputs pos rotation rot scale scl dirty true
        // returns value

        : position(pos), rotation(rot), scale(scl), dirty(true) {
        updateMatrices();
    }

    // setposition
    // updates internal state and marks dependent data as dirty
    // inputs p
    // returns none

    __host__ __device__ void setPosition(const vec3 &p) {
        position = p;
        dirty = true;
    }

    // setrotation
    // updates internal state and marks dependent data as dirty
    // inputs r
    // returns none

    __host__ __device__ void setRotation(const vec3 &r) {
        rotation = r;
        dirty = true;
    }

    // setscale
    // updates internal state and marks dependent data as dirty
    // inputs s
    // returns none

    __host__ __device__ void setScale(const vec3 &s) {
        scale = s;
        dirty = true;
    }

    // setscale
    // updates internal state and marks dependent data as dirty
    // inputs s
    // returns none

    __host__ __device__ void setScale(float s) {
        scale = vec3(s);
        dirty = true;
    }

    // translate
    // implements a unit of behavior used by higher level scene code
    // inputs delta
    // returns none

    __host__ __device__ void translate(const vec3 &delta) {
        position = position + delta;
        dirty = true;
    }

    // rotate
    // implements a unit of behavior used by higher level scene code
    // inputs deltaradians
    // returns none

    __host__ __device__ void rotate(const vec3 &deltaRadians) {
        rotation = rotation + deltaRadians;
        dirty = true;
    }

    // updatematrices
    // implements a unit of behavior used by higher level scene code
    // inputs none
    // returns none

    __host__ __device__ void updateMatrices() {
        if (!dirty)
            return;

        float cx = cosf(rotation.x), sx = sinf(rotation.x);
        float cy = cosf(rotation.y), sy = sinf(rotation.y);
        float cz = cosf(rotation.z), sz = sinf(rotation.z);

        mat4 rotMat;

        rotMat.m[0] = cy * cz;
        rotMat.m[1] = cz * sx * sy - cx * sz;
        rotMat.m[2] = cx * cz * sy + sx * sz;
        rotMat.m[3] = 0.0f;

        rotMat.m[4] = cy * sz;
        rotMat.m[5] = cx * cz + sx * sy * sz;
        rotMat.m[6] = cx * sy * sz - cz * sx;
        rotMat.m[7] = 0.0f;

        rotMat.m[8] = -sy;
        rotMat.m[9] = cy * sx;
        rotMat.m[10] = cx * cy;
        rotMat.m[11] = 0.0f;

        rotMat.m[12] = 0.0f;
        rotMat.m[13] = 0.0f;
        rotMat.m[14] = 0.0f;
        rotMat.m[15] = 1.0f;

        worldMatrix = mat4::identity();
        worldMatrix.m[0] = scale.x;
        worldMatrix.m[5] = scale.y;
        worldMatrix.m[10] = scale.z;

        worldMatrix = rotMat * worldMatrix;

        worldMatrix.m[3] = position.x;
        worldMatrix.m[7] = position.y;
        worldMatrix.m[11] = position.z;

        inverseMatrix = worldMatrix.inverse();

        normalMatrix = inverseMatrix.transpose();

        dirty = false;
    }

    // transformpoint
    // implements a unit of behavior used by higher level scene code
    // inputs p
    // returns vec3

    __host__ __device__ vec3 transformPoint(const vec3 &p) const {
        float x = worldMatrix.m[0] * p.x + worldMatrix.m[1] * p.y +
                  worldMatrix.m[2] * p.z + worldMatrix.m[3];
        float y = worldMatrix.m[4] * p.x + worldMatrix.m[5] * p.y +
                  worldMatrix.m[6] * p.z + worldMatrix.m[7];
        float z = worldMatrix.m[8] * p.x + worldMatrix.m[9] * p.y +
                  worldMatrix.m[10] * p.z + worldMatrix.m[11];
        return vec3(x, y, z);
    }

    // inversetransformpoint
    // updates internal state and marks dependent data as dirty
    // inputs p
    // returns vec3

    __host__ __device__ vec3 inverseTransformPoint(const vec3 &p) const {
        float x = inverseMatrix.m[0] * p.x + inverseMatrix.m[1] * p.y +
                  inverseMatrix.m[2] * p.z + inverseMatrix.m[3];
        float y = inverseMatrix.m[4] * p.x + inverseMatrix.m[5] * p.y +
                  inverseMatrix.m[6] * p.z + inverseMatrix.m[7];
        float z = inverseMatrix.m[8] * p.x + inverseMatrix.m[9] * p.y +
                  inverseMatrix.m[10] * p.z + inverseMatrix.m[11];
        return vec3(x, y, z);
    }

    // transformdirection
    // implements a unit of behavior used by higher level scene code
    // inputs d
    // returns vec3

    __host__ __device__ vec3 transformDirection(const vec3 &d) const {
        float x = worldMatrix.m[0] * d.x + worldMatrix.m[1] * d.y +
                  worldMatrix.m[2] * d.z;
        float y = worldMatrix.m[4] * d.x + worldMatrix.m[5] * d.y +
                  worldMatrix.m[6] * d.z;
        float z = worldMatrix.m[8] * d.x + worldMatrix.m[9] * d.y +
                  worldMatrix.m[10] * d.z;
        return vec3(x, y, z);
    }

    // inversetransformdirection
    // updates internal state and marks dependent data as dirty
    // inputs d
    // returns vec3

    __host__ __device__ vec3 inverseTransformDirection(const vec3 &d) const {
        float x = inverseMatrix.m[0] * d.x + inverseMatrix.m[1] * d.y +
                  inverseMatrix.m[2] * d.z;
        float y = inverseMatrix.m[4] * d.x + inverseMatrix.m[5] * d.y +
                  inverseMatrix.m[6] * d.z;
        float z = inverseMatrix.m[8] * d.x + inverseMatrix.m[9] * d.y +
                  inverseMatrix.m[10] * d.z;
        return vec3(x, y, z);
    }

    // transformnormal
    // implements a unit of behavior used by higher level scene code
    // inputs n
    // returns vec3

    __host__ __device__ vec3 transformNormal(const vec3 &n) const {
        float x = normalMatrix.m[0] * n.x + normalMatrix.m[1] * n.y +
                  normalMatrix.m[2] * n.z;
        float y = normalMatrix.m[4] * n.x + normalMatrix.m[5] * n.y +
                  normalMatrix.m[6] * n.z;
        float z = normalMatrix.m[8] * n.x + normalMatrix.m[9] * n.y +
                  normalMatrix.m[10] * n.z;
        return normalize(vec3(x, y, z));
    }

    // transformraytolocal
    // implements a unit of behavior used by higher level scene code
    // inputs worldray
    // returns ray

    __host__ __device__ Ray transformRayToLocal(const Ray &worldRay) const {
        vec3 localOrigin = inverseTransformPoint(worldRay.origin());
        vec3 localDir = inverseTransformDirection(worldRay.direction());
        return Ray(localOrigin, normalize(localDir));
    }

    // transformaabb
    // implements a unit of behavior used by higher level scene code
    // inputs localaabb
    // returns aabb

    __host__ __device__ AABB transformAABB(const AABB &localAABB) const {

        vec3 corners[8] = {
            vec3(localAABB.bmin.x, localAABB.bmin.y, localAABB.bmin.z),
            vec3(localAABB.bmax.x, localAABB.bmin.y, localAABB.bmin.z),
            vec3(localAABB.bmin.x, localAABB.bmax.y, localAABB.bmin.z),
            vec3(localAABB.bmax.x, localAABB.bmax.y, localAABB.bmin.z),
            vec3(localAABB.bmin.x, localAABB.bmin.y, localAABB.bmax.z),
            vec3(localAABB.bmax.x, localAABB.bmin.y, localAABB.bmax.z),
            vec3(localAABB.bmin.x, localAABB.bmax.y, localAABB.bmax.z),
            vec3(localAABB.bmax.x, localAABB.bmax.y, localAABB.bmax.z)};

        AABB worldAABB = AABB::make_invalid();
        for (int i = 0; i < 8; i++) {
            worldAABB.expand(transformPoint(corners[i]));
        }
        return worldAABB;
    }
};

struct DeviceTransform {
    mat4 worldMatrix;
    mat4 inverseMatrix;
    mat4 normalMatrix;
    AABB worldAABB;

    // transformpoint
    // implements a unit of behavior used by higher level scene code
    // inputs p
    // returns vec3

    __device__ vec3 transformPoint(const vec3 &p) const {
        float x = worldMatrix.m[0] * p.x + worldMatrix.m[1] * p.y +
                  worldMatrix.m[2] * p.z + worldMatrix.m[3];
        float y = worldMatrix.m[4] * p.x + worldMatrix.m[5] * p.y +
                  worldMatrix.m[6] * p.z + worldMatrix.m[7];
        float z = worldMatrix.m[8] * p.x + worldMatrix.m[9] * p.y +
                  worldMatrix.m[10] * p.z + worldMatrix.m[11];
        return vec3(x, y, z);
    }

    // inversetransformpoint
    // updates internal state and marks dependent data as dirty
    // inputs p
    // returns vec3

    __device__ vec3 inverseTransformPoint(const vec3 &p) const {
        float x = inverseMatrix.m[0] * p.x + inverseMatrix.m[1] * p.y +
                  inverseMatrix.m[2] * p.z + inverseMatrix.m[3];
        float y = inverseMatrix.m[4] * p.x + inverseMatrix.m[5] * p.y +
                  inverseMatrix.m[6] * p.z + inverseMatrix.m[7];
        float z = inverseMatrix.m[8] * p.x + inverseMatrix.m[9] * p.y +
                  inverseMatrix.m[10] * p.z + inverseMatrix.m[11];
        return vec3(x, y, z);
    }

    // inversetransformdirection
    // updates internal state and marks dependent data as dirty
    // inputs d
    // returns vec3

    __device__ vec3 inverseTransformDirection(const vec3 &d) const {
        float x = inverseMatrix.m[0] * d.x + inverseMatrix.m[1] * d.y +
                  inverseMatrix.m[2] * d.z;
        float y = inverseMatrix.m[4] * d.x + inverseMatrix.m[5] * d.y +
                  inverseMatrix.m[6] * d.z;
        float z = inverseMatrix.m[8] * d.x + inverseMatrix.m[9] * d.y +
                  inverseMatrix.m[10] * d.z;
        return vec3(x, y, z);
    }

    // transformnormal
    // implements a unit of behavior used by higher level scene code
    // inputs n
    // returns vec3

    __device__ vec3 transformNormal(const vec3 &n) const {
        float x = normalMatrix.m[0] * n.x + normalMatrix.m[1] * n.y +
                  normalMatrix.m[2] * n.z;
        float y = normalMatrix.m[4] * n.x + normalMatrix.m[5] * n.y +
                  normalMatrix.m[6] * n.z;
        float z = normalMatrix.m[8] * n.x + normalMatrix.m[9] * n.y +
                  normalMatrix.m[10] * n.z;
        return normalize(vec3(x, y, z));
    }

    // transformraytolocal
    // implements a unit of behavior used by higher level scene code
    // inputs worldray
    // returns ray

    __device__ Ray transformRayToLocal(const Ray &worldRay) const {
        vec3 localOrigin = inverseTransformPoint(worldRay.origin());
        vec3 localDir = inverseTransformDirection(worldRay.direction());
        return Ray(localOrigin, normalize(localDir));
    }
};

__host__ __device__ inline Transform3D
// lerptransform
// implements a unit of behavior used by higher level scene code
// inputs a b t
// returns value

lerpTransform(const Transform3D &a, const Transform3D &b, float t) {
    Transform3D result;
    result.position = lerp(a.position, b.position, t);
    result.rotation = lerp(a.rotation, b.rotation, t);
    result.scale = lerp(a.scale, b.scale, t);
    result.dirty = true;
    return result;
}

// orbitaround
// implements a unit of behavior used by higher level scene code
// inputs transform center radius angle height
// returns none

__host__ inline void orbitAround(Transform3D &transform, const vec3 &center,
                                 float radius, float angle,
                                 float height = 0.0f) {

    transform.position.x = center.x + radius * cosf(angle);
    transform.position.y = center.y + height;
    transform.position.z = center.z + radius * sinf(angle);
    transform.dirty = true;
}

// oscillate
// implements a unit of behavior used by higher level scene code
// inputs transform axis amplitude time frequency
// returns none

__host__ inline void oscillate(Transform3D &transform, const vec3 &axis,
                               float amplitude, float time,
                               float frequency = 1.0f) {

    float offset = amplitude * sinf(time * frequency * 2.0f * 3.14159265f);
    transform.position = transform.position + axis * offset;
    transform.dirty = true;
}

#endif