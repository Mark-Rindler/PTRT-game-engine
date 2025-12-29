#ifndef RAY_CUH
#define RAY_CUH

#include "common\vec3.cuh"

// Use vec3 as point3
using point3 = vec3;

class Ray {
  public:
    // Data Members
    point3 orig;
    vec3 dir;
    bool spec;

    // Constructors
    __host__ __device__ Ray() {}

    __host__ __device__ Ray(const point3 &origin, const vec3 &direction)
        : orig(origin), dir(direction), spec(false) // Default to not specular
    {}

    // Constructor for path tracing
    __host__ __device__ Ray(const point3 &origin, const vec3 &direction,
                            bool is_specular)
        : orig(origin), dir(direction), spec(is_specular) {}

    // Member Functions
    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }
    __host__ __device__ bool isSpecular() const { return spec; }

    __host__ __device__ point3 at(float t) const { return orig + t * dir; }
};

#endif