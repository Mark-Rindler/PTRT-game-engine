#ifndef MESH_CUH
#define MESH_CUH

#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/ray.cuh"
#include "common/vec3.cuh"

// RAY STRUCTURE

struct alignas(16) RayOpt {
    float3 orig;
    float3 dir;
    float3 inv_dir;
    int3 sign; // 0 if dir >= 0, 1 if dir < 0

    __device__ __forceinline__ RayOpt() {}

    __device__ __forceinline__ RayOpt(const vec3 &o, const vec3 &d) {
        orig = make_float3(o.x, o.y, o.z);
        dir = make_float3(d.x, d.y, d.z);

        // Precompute inverse direction with safe handling of zeros
        inv_dir.x = (fabsf(d.x) > 1e-8f) ? (1.0f / d.x)
                                         : ((d.x >= 0.0f) ? 1e30f : -1e30f);
        inv_dir.y = (fabsf(d.y) > 1e-8f) ? (1.0f / d.y)
                                         : ((d.y >= 0.0f) ? 1e30f : -1e30f);
        inv_dir.z = (fabsf(d.z) > 1e-8f) ? (1.0f / d.z)
                                         : ((d.z >= 0.0f) ? 1e30f : -1e30f);

        // Precompute direction signs for ordered traversal
        sign.x = (d.x < 0.0f) ? 1 : 0;
        sign.y = (d.y < 0.0f) ? 1 : 0;
        sign.z = (d.z < 0.0f) ? 1 : 0;
    }

    __device__ __forceinline__ RayOpt(const Ray &r)
        : RayOpt(r.origin(), r.direction()) {}

    __device__ __forceinline__ vec3 at(float t) const {
        return vec3(orig.x + t * dir.x, orig.y + t * dir.y, orig.z + t * dir.z);
    }
};

// OPTIMIZED AABB

struct alignas(16) AABB {
    float3 bmin;
    float3 bmax;

    __device__ __forceinline__ bool hit_fast(const RayOpt &r, float tmax_in,
                                             float &tmin_out) const {
        // Compute slab intersections
        float tx1 = (bmin.x - r.orig.x) * r.inv_dir.x;
        float tx2 = (bmax.x - r.orig.x) * r.inv_dir.x;
        float tmin = fminf(tx1, tx2);
        float tmax = fmaxf(tx1, tx2);

        float ty1 = (bmin.y - r.orig.y) * r.inv_dir.y;
        float ty2 = (bmax.y - r.orig.y) * r.inv_dir.y;
        tmin = fmaxf(tmin, fminf(ty1, ty2));
        tmax = fminf(tmax, fmaxf(ty1, ty2));

        float tz1 = (bmin.z - r.orig.z) * r.inv_dir.z;
        float tz2 = (bmax.z - r.orig.z) * r.inv_dir.z;
        tmin = fmaxf(tmin, fminf(tz1, tz2));
        tmax = fminf(tmax, fmaxf(tz1, tz2));

        tmin = fmaxf(tmin, 1e-4f); // Near plane
        tmin_out = tmin;

        return tmax >= tmin && tmin < tmax_in;
    }

    // Simple hit test without returning t
    __device__ __forceinline__ bool hit_any(const RayOpt &r,
                                            float tmax_in) const {
        float dummy;
        return hit_fast(r, tmax_in, dummy);
    }

    // Legacy interface for compatibility
    __host__ __device__ __forceinline__ bool hit(const Ray &r,
                                                 float tMax) const {
        const vec3 orig = r.origin();
        const vec3 dir = r.direction();
        float tmin = 1e-4f, tmax = tMax;

        // X axis
        float invD = 1.0f / dir.x;
        float t0 = (bmin.x - orig.x) * invD;
        float t1 = (bmax.x - orig.x) * invD;
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax <= tmin)
            return false;

        // Y axis
        invD = 1.0f / dir.y;
        t0 = (bmin.y - orig.y) * invD;
        t1 = (bmax.y - orig.y) * invD;
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax <= tmin)
            return false;

        // Z axis
        invD = 1.0f / dir.z;
        t0 = (bmin.z - orig.z) * invD;
        t1 = (bmax.z - orig.z) * invD;
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);

        return tmax > tmin;
    }

    __host__ __device__ __forceinline__ bool hit_t(const Ray &r, float tMax,
                                                   float &tEnter) const {
        const vec3 orig = r.origin();
        const vec3 dir = r.direction();
        float tmin = 1e-4f, tmax = tMax;

        // X axis
        float invD = 1.0f / dir.x;
        float t0 = (bmin.x - orig.x) * invD;
        float t1 = (bmax.x - orig.x) * invD;
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax <= tmin)
            return false;

        // Y axis
        invD = 1.0f / dir.y;
        t0 = (bmin.y - orig.y) * invD;
        t1 = (bmax.y - orig.y) * invD;
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax <= tmin)
            return false;

        // Z axis
        invD = 1.0f / dir.z;
        t0 = (bmin.z - orig.z) * invD;
        t1 = (bmax.z - orig.z) * invD;
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax <= tmin)
            return false;

        tEnter = tmin;
        return true;
    }

    __host__ __device__ __forceinline__ vec3 extent() const {
        return vec3(bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z);
    }

    __host__ __device__ __forceinline__ vec3 center() const {
        return vec3((bmin.x + bmax.x) * 0.5f, (bmin.y + bmax.y) * 0.5f,
                    (bmin.z + bmax.z) * 0.5f);
    }

    __host__ __device__ static __forceinline__ AABB make_invalid() {
        AABB box;
        box.bmin = make_float3(1e30f, 1e30f, 1e30f);
        box.bmax = make_float3(-1e30f, -1e30f, -1e30f);
        return box;
    }

    __host__ __device__ __forceinline__ void expand(const AABB &b) {
        bmin.x = fminf(bmin.x, b.bmin.x);
        bmin.y = fminf(bmin.y, b.bmin.y);
        bmin.z = fminf(bmin.z, b.bmin.z);
        bmax.x = fmaxf(bmax.x, b.bmax.x);
        bmax.y = fmaxf(bmax.y, b.bmax.y);
        bmax.z = fmaxf(bmax.z, b.bmax.z);
    }

    __host__ __device__ __forceinline__ void expand(const vec3 &p) {
        bmin.x = fminf(bmin.x, p.x);
        bmin.y = fminf(bmin.y, p.y);
        bmin.z = fminf(bmin.z, p.z);
        bmax.x = fmaxf(bmax.x, p.x);
        bmax.y = fmaxf(bmax.y, p.y);
        bmax.z = fmaxf(bmax.z, p.z);
    }
};

struct alignas(32) DeviceBVHNode {
    AABB bbox; // 24 bytes
    int left;  // 4 bytes - left child or -1
    int right; // 4 bytes - right child or -1
    int start; // 4 bytes - primitive start index (leaf only)
    int count; // 4 bytes - primitive count (leaf: >0, internal: 0)
    // Total: 40 bytes, padded to 48 for alignment
    // We could compress further but this is a good balance
};
// note: fix up the path tracer later

// TRIANGLE INDEX

struct Tri {
    int v0, v1, v2;
};

// FAST MOLLER-TRUMBORE TRIANGLE INTERSECTION

__device__ __forceinline__ bool
intersect_triangle_mt(const RayOpt &ray, const vec3 &v0, const vec3 &v1,
                      const vec3 &v2, float &t_out, float &u_out,
                      float &v_out) {
    const vec3 edge1 = v1 - v0;
    const vec3 edge2 = v2 - v0;

    const vec3 rd = vec3(ray.dir.x, ray.dir.y, ray.dir.z);
    const vec3 h = cross(rd, edge2);
    const float a = dot(edge1, h);

    // Parallel check with small epsilon
    if (fabsf(a) < 1e-8f)
        return false;

    const float f = 1.0f / a;
    const vec3 s =
        vec3(ray.orig.x - v0.x, ray.orig.y - v0.y, ray.orig.z - v0.z);
    const float u = f * dot(s, h);

    // Early exit for u outside [0,1]
    if (u < 0.0f || u > 1.0f)
        return false;

    const vec3 q = cross(s, edge1);
    const float v = f * dot(rd, q);

    // Early exit for v outside [0,1] or u+v > 1
    if (v < 0.0f || u + v > 1.0f)
        return false;

    const float t = f * dot(edge2, q);

    if (t > 1e-4f) {
        t_out = t;
        u_out = u;
        v_out = v;
        return true;
    }
    return false;
}

// MESH CLASS

class Mesh {
  public:
    std::vector<vec3> vertices;
    std::vector<Tri> faces;

    vec3 position = vec3(0.0f);
    vec3 rotationEuler = vec3(0.0f);

    void setPosition(const vec3 &p) { position = p; }
    void setRotation(const vec3 &r) { rotationEuler = r; }

    vec3 *d_vertices = nullptr;
    Tri *d_faces = nullptr;

    std::vector<DeviceBVHNode> bvhNodes;
    std::vector<int> bvhPrimIndices;

    DeviceBVHNode *d_bvhNodes = nullptr;
    int *d_bvhPrim = nullptr;

    bool bvhDirty = true;
    int bvhLeafTarget =
        4; // Smaller leaves = faster traversal for complex scenes
    int bvhLeafTol = 2;

    void setBVHLeafParams(int target, int tol = 2) {
        bvhLeafTarget = target < 1 ? 1 : target;
        bvhLeafTol = tol < 0 ? 0 : tol;
        bvhDirty = true;
    }

    void buildBVH();
    void uploadBVH();
    void freeBVHDevice();

    Mesh();
    explicit Mesh(const std::string &path);
    ~Mesh();

    Mesh(const Mesh &) = delete;
    Mesh &operator=(const Mesh &) = delete;

    Mesh(Mesh &&other) noexcept { *this = std::move(other); }

    Mesh &operator=(Mesh &&other) noexcept {
        if (this == &other)
            return *this;

        freeDevice();
        freeBVHDevice();

        vertices = std::move(other.vertices);
        faces = std::move(other.faces);
        d_vertices = other.d_vertices;
        other.d_vertices = nullptr;
        d_faces = other.d_faces;
        other.d_faces = nullptr;
        bvhNodes = std::move(other.bvhNodes);
        bvhPrimIndices = std::move(other.bvhPrimIndices);
        d_bvhNodes = other.d_bvhNodes;
        other.d_bvhNodes = nullptr;
        d_bvhPrim = other.d_bvhPrim;
        other.d_bvhPrim = nullptr;
        bvhDirty = other.bvhDirty;
        bvhLeafTarget = other.bvhLeafTarget;
        bvhLeafTol = other.bvhLeafTol;
        position = other.position;
        rotationEuler = other.rotationEuler;

        return *this;
    }

    void upload();
    void freeDevice();
    AABB boundingBox() const;
    void scale(float s);
    void scale(vec3 s);
    void translate(const vec3 &d);
    void moveTo(const vec3 &p);
    void rotateSelfEulerXYZ(const vec3 &rad);

    size_t faceCount() const { return faces.size(); }
    size_t vertexCount() const { return vertices.size(); }
};

// MESH IMPLEMENTATION

inline Mesh::Mesh() {
    vertices = {{-0.5f, -0.5f, -3.5f}, {0.5f, -0.5f, -3.5f},
                {0.5f, 0.5f, -3.5f},   {-0.5f, 0.5f, -3.5f},
                {-0.5f, -0.5f, -2.5f}, {0.5f, -0.5f, -2.5f},
                {0.5f, 0.5f, -2.5f},   {-0.5f, 0.5f, -2.5f}};
    faces = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
             {3, 7, 6}, {3, 6, 2}, {0, 4, 7}, {0, 7, 3}, {1, 2, 6}, {1, 6, 5}};
}

inline Mesh::Mesh(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Mesh: cannot open " + path);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        } else if (key == "f") {
            std::vector<int> idx;
            std::string vert;
            while (ss >> vert) {
                size_t slash = vert.find('/');
                int id = std::stoi(
                    slash == std::string::npos ? vert : vert.substr(0, slash));
                idx.push_back(id - 1);
            }
            if (idx.size() < 3)
                continue;
            for (size_t i = 1; i + 1 < idx.size(); ++i)
                faces.push_back({idx[0], idx[i], idx[i + 1]});
        }
    }
    if (vertices.empty() || faces.empty())
        throw std::runtime_error("Mesh: no geometry in " + path);
}

inline void Mesh::upload() {
    freeDevice();
    if (vertices.empty() || faces.empty())
        return;
    cudaMalloc(&d_vertices, sizeof(vec3) * vertices.size());
    cudaMalloc(&d_faces, sizeof(Tri) * faces.size());
    cudaMemcpy(d_vertices, vertices.data(), sizeof(vec3) * vertices.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces.data(), sizeof(Tri) * faces.size(),
               cudaMemcpyHostToDevice);
}

inline void Mesh::freeDevice() {
    if (d_vertices) {
        cudaFree(d_vertices);
        d_vertices = nullptr;
    }
    if (d_faces) {
        cudaFree(d_faces);
        d_faces = nullptr;
    }
}

inline void Mesh::freeBVHDevice() {
    if (d_bvhNodes) {
        cudaFree(d_bvhNodes);
        d_bvhNodes = nullptr;
    }
    if (d_bvhPrim) {
        cudaFree(d_bvhPrim);
        d_bvhPrim = nullptr;
    }
}

inline static AABB tri_bounds(const vec3 &a, const vec3 &b, const vec3 &c) {
    AABB box;
    box.bmin =
        make_float3(fminf(fminf(a.x, b.x), c.x), fminf(fminf(a.y, b.y), c.y),
                    fminf(fminf(a.z, b.z), c.z));
    box.bmax =
        make_float3(fmaxf(fmaxf(a.x, b.x), c.x), fmaxf(fmaxf(a.y, b.y), c.y),
                    fmaxf(fmaxf(a.z, b.z), c.z));
    return box;
}

struct _BuildRef {
    int f;
    vec3 c;
    AABB b;
};

// SAH based BVH builder for better traversal performance
inline void Mesh::buildBVH() {
    bvhNodes.clear();
    bvhPrimIndices.clear();

    if (faces.empty()) {
        bvhDirty = false;
        return;
    }

    std::vector<_BuildRef> refs;
    refs.reserve(faces.size());
    for (int i = 0; i < (int)faces.size(); ++i) {
        const Tri t = faces[i];
        const vec3 &v0 = vertices[t.v0];
        const vec3 &v1 = vertices[t.v1];
        const vec3 &v2 = vertices[t.v2];
        _BuildRef r;
        r.f = i;
        r.b = tri_bounds(v0, v1, v2);
        r.c = (v0 + v1 + v2) * (1.0f / 3.0f);
        refs.push_back(r);
    }

    const int leafMax = bvhLeafTarget + bvhLeafTol;

    struct _Builder {
        std::vector<DeviceBVHNode> &nodes;
        std::vector<int> &prims;
        std::vector<_BuildRef> &R;
        int leafMax;

        int build(int begin, int end) {
            AABB bb = AABB::make_invalid();
            AABB cb = AABB::make_invalid();
            for (int i = begin; i < end; ++i) {
                bb.expand(R[i].b);
                cb.expand(R[i].c);
            }
            int n = end - begin;

            int me = (int)nodes.size();
            nodes.emplace_back();
            nodes[me].bbox = bb;
            nodes[me].left = -1;
            nodes[me].right = -1;
            nodes[me].start = -1;
            nodes[me].count = 0;

            if (n <= leafMax) {
                nodes[me].start = (int)prims.size();
                nodes[me].count = n;
                prims.reserve(prims.size() + n);
                for (int i = begin; i < end; ++i)
                    prims.push_back(R[i].f);
                return me;
            }

            // Find best split axis using SAH approximation
            vec3 e = cb.extent();
            int axis = (e.x > e.y && e.x > e.z) ? 0 : ((e.y > e.z) ? 1 : 2);

            int mid = (begin + end) / 2;
            std::nth_element(R.begin() + begin, R.begin() + mid,
                             R.begin() + end,
                             [axis](const _BuildRef &A, const _BuildRef &B) {
                                 return A.c[axis] < B.c[axis];
                             });

            int L = build(begin, mid);
            int Rn = build(mid, end);
            nodes[me].left = L;
            nodes[me].right = Rn;
            return me;
        }
    };

    _Builder B{bvhNodes, bvhPrimIndices, refs, leafMax};
    B.build(0, (int)refs.size());
    bvhDirty = false;
}

inline void Mesh::uploadBVH() {
    freeBVHDevice();
    if (bvhNodes.empty())
        return;

    cudaMalloc(&d_bvhNodes, sizeof(DeviceBVHNode) * bvhNodes.size());
    cudaMemcpy(d_bvhNodes, bvhNodes.data(),
               sizeof(DeviceBVHNode) * bvhNodes.size(), cudaMemcpyHostToDevice);

    if (!bvhPrimIndices.empty()) {
        cudaMalloc(&d_bvhPrim, sizeof(int) * bvhPrimIndices.size());
        cudaMemcpy(d_bvhPrim, bvhPrimIndices.data(),
                   sizeof(int) * bvhPrimIndices.size(), cudaMemcpyHostToDevice);
    }
}

inline Mesh::~Mesh() {
    freeDevice();
    freeBVHDevice();
}

inline AABB Mesh::boundingBox() const {
    if (vertices.empty()) {
        AABB box;
        box.bmin = make_float3(0, 0, 0);
        box.bmax = make_float3(0, 0, 0);
        return box;
    }

    vec3 vmin = vertices[0], vmax = vertices[0];
    for (size_t i = 1; i < vertices.size(); ++i) {
        const vec3 &v = vertices[i];
        vmin.x = fminf(vmin.x, v.x);
        vmax.x = fmaxf(vmax.x, v.x);
        vmin.y = fminf(vmin.y, v.y);
        vmax.y = fmaxf(vmax.y, v.y);
        vmin.z = fminf(vmin.z, v.z);
        vmax.z = fmaxf(vmax.z, v.z);
    }
    AABB box;
    box.bmin = make_float3(vmin.x, vmin.y, vmin.z);
    box.bmax = make_float3(vmax.x, vmax.y, vmax.z);
    return box;
}

inline void Mesh::scale(float s) {
    for (auto &v : vertices) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
    }
    bvhDirty = true;
}

inline void Mesh::scale(vec3 s) {
    for (auto &v : vertices) {
        v.x *= s.x;
        v.y *= s.y;
        v.z *= s.z;
    }
    bvhDirty = true;
}

inline void Mesh::translate(const vec3 &d) {
    for (auto &v : vertices) {
        v.x += d.x;
        v.y += d.y;
        v.z += d.z;
    }
    bvhDirty = true;
}

inline void Mesh::moveTo(const vec3 &p) {
    AABB box = boundingBox();
    vec3 center = box.center();
    translate(p - center);
    bvhDirty = true;
}

inline static vec3 rX(const vec3 &v, float c, float s) {
    return {v.x, c * v.y - s * v.z, s * v.y + c * v.z};
}
inline static vec3 rY(const vec3 &v, float c, float s) {
    return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
}
inline static vec3 rZ(const vec3 &v, float c, float s) {
    return {c * v.x - s * v.y, s * v.x + c * v.y, v.z};
}

inline void Mesh::rotateSelfEulerXYZ(const vec3 &rad) {
    vec3 center = boundingBox().center();
    const float cx = cosf(rad.x), sx = sinf(rad.x);
    const float cy = cosf(rad.y), sy = sinf(rad.y);
    const float cz = cosf(rad.z), sz = sinf(rad.z);

    for (auto &v : vertices) {
        vec3 p = v - center;
        p = rX(p, cx, sx);
        p = rY(p, cy, sy);
        p = rZ(p, cz, sz);
        v = p + center;
    }
    bvhDirty = true;
}

#endif
