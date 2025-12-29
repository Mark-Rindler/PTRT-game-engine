// mesh cuh
// triangle mesh container with cpu storage and optional gpu buffers
// supports uploading vertices faces and bvh nodes to device memory
// provides simple transforms and bounding boxes for tlas building
#ifndef MESH_CUH
#define MESH_CUH

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "common/ray.cuh"
#include "common/vec3.cuh"
#include "pathtracer/scene/transform.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("cuda error: ") +             \
                                     cudaGetErrorString(_e));                  \
        }                                                                      \
    } while (0)
#endif

struct DeviceBVHNode {
    AABB bbox;
    int left;
    int right;
    int start;
    int count;
};

struct Tri {
    int v0, v1, v2;
};

class Mesh {
  public:
    std::vector<vec3> vertices;
    std::vector<Tri> faces;

    vec3 *d_vertices = nullptr;
    Tri *d_faces = nullptr;

    std::vector<DeviceBVHNode> bvhNodes;
    std::vector<int> bvhPrimIndices;

    DeviceBVHNode *d_bvhNodes = nullptr;
    int *d_bvhPrim = nullptr;

    bool bvhDirty = true;
    bool vertsDirty = true;
    int bvhLeafTarget = 12;
    int bvhLeafTol = 5;

    // setbvhleafparams
    // updates internal state and marks dependent data as dirty
    // inputs target tol
    // returns none

    void setBVHLeafParams(int target, int tol = 5) {
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

    // mesh
    // implements a unit of behavior used by higher level scene code
    // inputs this
    // returns value

    Mesh(Mesh &&other) noexcept { *this = std::move(other); }

    // operator
    // implements a unit of behavior used by higher level scene code
    // inputs other
    // returns mesh

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
        vertsDirty = other.vertsDirty;

        bvhLeafTarget = other.bvhLeafTarget;
        bvhLeafTol = other.bvhLeafTol;

        transform = other.transform;
        localAABB = other.localAABB;

        other.bvhDirty = true;
        other.vertsDirty = true;
        other.localAABB = AABB::make_invalid();

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

    // facecount
    // implements a unit of behavior used by higher level scene code
    // inputs faces size
    // returns size t

    __host__ size_t faceCount() const { return faces.size(); }
    // vertexcount
    // implements a unit of behavior used by higher level scene code
    // inputs vertices size
    // returns size t

    __host__ size_t vertexCount() const { return vertices.size(); }

    Transform3D transform;
    AABB localAABB;

    // settransform
    // updates internal state and marks dependent data as dirty
    // inputs t
    // returns none

    void setTransform(const Transform3D &t) {
        transform = t;
        transform.updateMatrices();
    }

    // setposition
    // updates internal state and marks dependent data as dirty
    // inputs pos
    // returns none

    void setPosition(const vec3 &pos) {
        transform.setPosition(pos);
        transform.updateMatrices();
    }

    // setrotation
    // updates internal state and marks dependent data as dirty
    // inputs rot
    // returns none

    void setRotation(const vec3 &rot) {
        transform.setRotation(rot);
        transform.updateMatrices();
    }

    // getworldaabb
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns aabb

    AABB getWorldAABB() const {
        if (!transform.dirty && localAABB.bmin.x < 1e20f) {
            return transform.transformAABB(localAABB);
        }
        return boundingBox();
    }

    // computelocalaabb
    // implements a unit of behavior used by higher level scene code
    // inputs localaabb
    // returns none

    void computeLocalAABB() { localAABB = boundingBox(); }
};

// mesh
// implements a unit of behavior used by higher level scene code
// inputs none
// returns inline

inline Mesh::Mesh() {
    vertices = {{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},
                {0.5f, 0.5f, -0.5f},   {-0.5f, 0.5f, -0.5f},
                {-0.5f, -0.5f, 0.5f},  {0.5f, -0.5f, 0.5f},
                {0.5f, 0.5f, 0.5f},    {-0.5f, 0.5f, 0.5f}};
    faces = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
             {3, 7, 6}, {3, 6, 2}, {0, 4, 7}, {0, 7, 3}, {1, 2, 6}, {1, 6, 5}};
}

// mesh
// implements a unit of behavior used by higher level scene code
// inputs path
// returns inline

inline Mesh::Mesh(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Mesh: cannot open " + path);

    vertices.reserve(1000000);
    faces.reserve(1000000);

    double sumX = 0.0;
    double sumY = 0.0;
    double sumZ = 0.0;

    std::string line;
    line.reserve(512);
    std::istringstream ss;
    std::string key;
    std::vector<int> idx;
    size_t vertex_count = 0;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        ss.clear();
        ss.str(line);
        ss >> key;

        if (key == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            if (!ss.fail()) {
                vertices.emplace_back(x, y, z);

                sumX += x;
                sumY += y;
                sumZ += z;
                vertex_count++;
            }
            // if
            // implements a unit of behavior used by higher level scene code
            // inputs key
            // returns else

        } else if (key == "f") {
            idx.clear();
            int id;
            char dummy;
            while (ss >> id) {
                if (id < 0)
                    id = static_cast<int>(vertex_count) + id;
                else
                    id = id - 1;
                idx.push_back(id);
                while (ss.peek() == '/') {
                    ss.get(dummy);
                    if (ss.peek() == '/')
                        ss.get(dummy);
                    int temp_id;
                    ss >> temp_id;
                }
            }
            if (idx.size() >= 3) {
                for (size_t i = 1; i + 1 < idx.size(); ++i) {
                    faces.push_back({idx[0], idx[i], idx[i + 1]});
                }
            }
        }
    }

    if (in.bad())
        throw std::runtime_error("Mesh: hardware error reading " + path);
    if (vertices.empty() || faces.empty())
        throw std::runtime_error("Mesh: no valid geometry in " + path);

    if (vertex_count > 0) {
        float centerX = static_cast<float>(sumX / vertex_count);
        float centerY = static_cast<float>(sumY / vertex_count);
        float centerZ = static_cast<float>(sumZ / vertex_count);

        for (auto &v : vertices) {
            v.x -= centerX;
            v.y -= centerY;
            v.z -= centerZ;
        }
    }
}

// upload
// uploads host data to device memory and updates device pointers
// inputs none
// returns none

inline void Mesh::upload() {

    if (!vertsDirty && d_vertices != nullptr && d_faces != nullptr)
        return;

    freeDevice();
    if (vertices.empty() || faces.empty())
        return;
    CUDA_CHECK(cudaMalloc(&d_vertices, sizeof(vec3) * vertices.size()));
    CUDA_CHECK(cudaMalloc(&d_faces, sizeof(Tri) * faces.size()));
    CUDA_CHECK(cudaMemcpy(d_vertices, vertices.data(),
                          sizeof(vec3) * vertices.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_faces, faces.data(), sizeof(Tri) * faces.size(),
                          cudaMemcpyHostToDevice));
    vertsDirty = false;
}

// freedevice
// releases owned resources and resets pointers to safe values
// inputs none
// returns none

inline void Mesh::freeDevice() {
    if (d_vertices) {
        CUDA_CHECK(cudaFree(d_vertices));
        d_vertices = nullptr;
    }
    if (d_faces) {
        CUDA_CHECK(cudaFree(d_faces));
        d_faces = nullptr;
    }
}

// freebvhdevice
// releases owned resources and resets pointers to safe values
// inputs none
// returns none

inline void Mesh::freeBVHDevice() {
    if (d_bvhNodes) {
        CUDA_CHECK(cudaFree(d_bvhNodes));
        d_bvhNodes = nullptr;
    }
    if (d_bvhPrim) {
        CUDA_CHECK(cudaFree(d_bvhPrim));
        d_bvhPrim = nullptr;
    }
}

// tri bounds
// implements a unit of behavior used by higher level scene code
// inputs a b c
// returns aabb

inline static AABB tri_bounds(const vec3 &a, const vec3 &b, const vec3 &c) {
    AABB box = {a, a};
    box.expand(b);
    box.expand(c);
    return box;
}

struct _BuildRef {
    int f;
    vec3 c;
    AABB b;
};

// buildbvh
// builds a bvh over mesh triangles for ray traversal on the gpu
// inputs none
// returns none

inline void Mesh::buildBVH() {
    bvhNodes.clear();
    bvhPrimIndices.clear();

    // build per triangle references with bounds and centroids
    if (faces.empty()) {
        bvhDirty = false;
        return;
    }

    std::vector<_BuildRef> refs;
    refs.reserve(faces.size());

    // fill refs then build a recursive bvh with a leaf size target
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

        // build
        // implements a unit of behavior used by higher level scene code
        // inputs begin end
        // returns int

        int build(int begin, int end) {
            // compute bounds for this range then either emit a leaf or split
            // into children
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

// uploadbvh
// uploads host data to device memory and updates device pointers
// inputs none
// returns none

inline void Mesh::uploadBVH() {
    freeBVHDevice();
    if (bvhNodes.empty())
        return;
    CUDA_CHECK(
        cudaMalloc(&d_bvhNodes, sizeof(DeviceBVHNode) * bvhNodes.size()));
    CUDA_CHECK(cudaMemcpy(d_bvhNodes, bvhNodes.data(),
                          sizeof(DeviceBVHNode) * bvhNodes.size(),
                          cudaMemcpyHostToDevice));
    if (!bvhPrimIndices.empty()) {
        CUDA_CHECK(cudaMalloc(&d_bvhPrim, sizeof(int) * bvhPrimIndices.size()));
        CUDA_CHECK(cudaMemcpy(d_bvhPrim, bvhPrimIndices.data(),
                              sizeof(int) * bvhPrimIndices.size(),
                              cudaMemcpyHostToDevice));
    } else {
        d_bvhPrim = nullptr;
    }
}

// mesh
// implements a unit of behavior used by higher level scene code
// inputs none
// returns inline

inline Mesh::~Mesh() {
    freeDevice();
    freeBVHDevice();
}

// boundingbox
// implements a unit of behavior used by higher level scene code
// inputs none
// returns aabb

inline AABB Mesh::boundingBox() const {
    if (vertices.empty())
        return {vec3(0.0f), vec3(0.0f)};
    vec3 vmin = vertices[0];
    vec3 vmax = vertices[0];
    for (size_t i = 1; i < vertices.size(); ++i) {
        const vec3 &v = vertices[i];
        vmin.x = fminf(vmin.x, v.x);
        vmax.x = fmaxf(vmax.x, v.x);
        vmin.y = fminf(vmin.y, v.y);
        vmax.y = fmaxf(vmax.y, v.y);
        vmin.z = fminf(vmin.z, v.z);
        vmax.z = fmaxf(vmax.z, v.z);
    }
    return {vmin, vmax};
}

// scale
// implements a unit of behavior used by higher level scene code
// inputs s
// returns none

inline void Mesh::scale(float s) {
    for (auto &v : vertices) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
    }
    bvhDirty = true;
    vertsDirty = true;
}

// scale
// implements a unit of behavior used by higher level scene code
// inputs s
// returns none

inline void Mesh::scale(vec3 s) {
    for (auto &v : vertices) {
        v.x *= s.x;
        v.y *= s.y;
        v.z *= s.z;
    }
    bvhDirty = true;
    vertsDirty = true;
}

// translate
// implements a unit of behavior used by higher level scene code
// inputs d
// returns none

inline void Mesh::translate(const vec3 &d) {
    for (auto &v : vertices) {
        v = v + d;
    }
    bvhDirty = true;
    vertsDirty = true;
}

// moveto
// implements a unit of behavior used by higher level scene code
// inputs p
// returns none

inline void Mesh::moveTo(const vec3 &p) {
    AABB bb = boundingBox();
    vec3 center = (bb.bmin + bb.bmax) * 0.5f;
    vec3 diff = p - center;
    translate(diff);
}

// rotateselfeulerxyz
// implements a unit of behavior used by higher level scene code
// inputs rad
// returns none

inline void Mesh::rotateSelfEulerXYZ(const vec3 &rad) {
    AABB bb = boundingBox();
    vec3 center = (bb.bmin + bb.bmax) * 0.5f;

    float cx = cosf(rad.x), sx = sinf(rad.x);
    float cy = cosf(rad.y), sy = sinf(rad.y);
    float cz = cosf(rad.z), sz = sinf(rad.z);

    for (auto &v : vertices) {
        vec3 p = v - center;

        float y1 = cx * p.y - sx * p.z;
        float z1 = sx * p.y + cx * p.z;
        p.y = y1;
        p.z = z1;

        float x2 = cy * p.x + sy * p.z;
        float z2 = -sy * p.x + cy * p.z;
        p.x = x2;
        p.z = z2;

        float x3 = cz * p.x - sz * p.y;
        float y3 = sz * p.x + cz * p.y;
        p.x = x3;
        p.y = y3;

        v = p + center;
    }
    bvhDirty = true;
    vertsDirty = true;
}

#endif // MESH_CUH