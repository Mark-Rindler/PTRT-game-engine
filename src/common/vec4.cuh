#ifndef VEC4_CUH
#define VEC4_CUH

#include "common\vec3.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Forward declaration for mat4
// This avoids a circular dependency, as mat4 will need to include vec4
class mat4;

class vec4 {
  public:
    float x, y, z, w;

    __host__ __device__ vec4() : x(0), y(0), z(0), w(0) {}
    __host__ __device__ vec4(float s) : x(s), y(s), z(s), w(s) {}
    __host__ __device__ vec4(float _x, float _y, float _z, float _w)
        : x(_x), y(_y), z(_z), w(_w) {}
    __host__ __device__ vec4(const vec3 &v, float _w)
        : x(v.x), y(v.y), z(v.z), w(_w) {}

    __host__ __device__ vec3 xyz() const { return vec3(x, y, z); }

    __host__ __device__ float operator[](int i) const {
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        if (i == 2)
            return z;
        return w;
    }

    __host__ __device__ float &operator[](int i) {
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        if (i == 2)
            return z;
        return w;
    }

    __host__ __device__ vec4 operator-() const { return vec4(-x, -y, -z, -w); }

    __host__ __device__ vec4 &operator+=(const vec4 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }
    __host__ __device__ vec4 &operator-=(const vec4 &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }
    __host__ __device__ vec4 &operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return *this;
    }
    __host__ __device__ vec4 &operator/=(float s) {
        float inv = 1.0f / s;
        x *= inv;
        y *= inv;
        z *= inv;
        w *= inv;
        return *this;
    }
};

// Utility functions for vec4
__host__ __device__ inline vec4 operator+(const vec4 &a, const vec4 &b) {
    return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ inline vec4 operator-(const vec4 &a, const vec4 &b) {
    return vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ inline vec4 operator*(const vec4 &v, float s) {
    return vec4(v.x * s, v.y * s, v.z * s, v.w * s);
}

__host__ __device__ inline vec4 operator*(float s, const vec4 &v) {
    return v * s;
}

__host__ __device__ inline vec4 operator/(const vec4 &v, float s) {
    return v * (1.0f / s);
}

__host__ __device__ inline float dot(const vec4 &a, const vec4 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ inline float length(const vec4 &v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline vec4 normalize(const vec4 &v) {
    return v / length(v);
}

#endif // VEC4_CUH