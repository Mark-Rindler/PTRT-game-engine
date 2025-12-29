#ifndef VISUALIZATION_CUH
#define VISUALIZATION_CUH

#include "common\triangle.cuh"
#include "common\vec3.cuh"
#include "pathtracer\scene\camera.cuh"
#include "pathtracer\scene\mesh.cuh"
#include <vector>

namespace Visualization {

// Host Functions (Declarations)

// Create a simple quad for a line segment
std::vector<Triangle> createLineQuad(const vec3 &start, const vec3 &end,
                                     float thickness = 0.01f,
                                     const vec3 &viewDir = vec3(0, 0, 1));

// Generate a cylinder mesh
std::vector<Triangle> generateCylinder(float radius, float height,
                                       int segments = 8);

// Generate a cone mesh
std::vector<Triangle> generateCone(float radius, float height,
                                   int segments = 8);

// Transform triangles by a matrix-like operation
void transformTriangles(std::vector<Triangle> &tris, const vec3 &translation,
                        const vec3 &scale = vec3(1.0f),
                        const vec3 &rotationAxis = vec3(0, 1, 0),
                        float rotationAngle = 0.0f);

// Generate arrow with LOD support
std::vector<Triangle> generateArrow(const vec3 &origin, const vec3 &direction,
                                    float length, float shaftRadius = 0.02f,
                                    float headRadius = 0.06f,
                                    float headLength = 0.15f, int lodLevel = 0);

// Camera Frustum Generation
std::vector<Triangle> generateFrustumWireframe(const Camera &cam, float aspect,
                                               float nearPlane = 0.1f,
                                               float farPlane = 100.0f,
                                               float lineThickness = 0.01f,
                                               bool useSimpleLines = false);

// Image Plane Generation
std::vector<Triangle> generateImagePlane(float width, float height,
                                         const vec3 &position,
                                         const vec3 &normal = vec3(0, 0, 1));

// Device Functions

// Kept in header to ensure visibility for CUDA Kernels
__device__ inline bool intersectTriangleWireframe(const Ray &ray,
                                                  const Triangle &tri, float &t,
                                                  float &u, float &v, float &w,
                                                  float edgeThreshold = 0.02f) {

    vec3 edge1 = tri.v1 - tri.v0;
    vec3 edge2 = tri.v2 - tri.v0;
    vec3 pvec = cross(ray.direction(), edge2);
    float det = dot(edge1, pvec);

    if (fabsf(det) < 1e-8f)
        return false;

    float invDet = 1.0f / det;
    vec3 tvec = ray.origin() - tri.v0;
    u = dot(tvec, pvec) * invDet;

    if (u < 0.0f || u > 1.0f)
        return false;

    vec3 qvec = cross(tvec, edge1);
    v = dot(ray.direction(), qvec) * invDet;

    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = dot(edge2, qvec) * invDet;
    if (t <= 0.0f)
        return false;

    // Compute barycentric coordinate w
    w = 1.0f - u - v;

    // Check if we're near an edge
    bool nearEdge =
        (u < edgeThreshold || v < edgeThreshold || w < edgeThreshold);

    return nearEdge;
}

} // namespace Visualization

#endif // VISUALIZATION_CUH