#ifndef VEC3_CUH
#define VEC3_CUH

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

class vec3 {
  public:
    float x, y, z;

    // Constructors
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3(float val) : x(val), y(val), z(val) {}

    // Accessors
    __host__ __device__ float operator[](int i) const {
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    __host__ __device__ float &operator[](int i) {
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    // Unary operators
    __host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }

    // Binary operators
    __host__ __device__ vec3 operator+(const vec3 &v) const {
        return vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ vec3 operator-(const vec3 &v) const {
        return vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ vec3 operator*(const vec3 &v) const {
        return vec3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__ vec3 operator*(float t) const {
        return vec3(x * t, y * t, z * t);
    }

    __host__ __device__ vec3 operator/(float t) const {
        return vec3(x / t, y / t, z / t);
    }

    __host__ __device__ vec3 operator/(vec3 &t) const {
        return vec3(x / t.x, y / t.y, z / t.z);
    }

    // Compound assignment operators
    __host__ __device__ vec3 &operator+=(const vec3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ vec3 &operator-=(const vec3 &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    __host__ __device__ vec3 &operator*=(float t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(float t) {
        x /= t;
        y /= t;
        z /= t;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(const vec3 &t) {
        x /= t.x;
        y /= t.y;
        z /= t.z;
        return *this;
    }

    // Utility functions
    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__ float length_squared() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ vec3 normalized() const {
        float len = length();
        return (len > 0) ? (*this / len) : vec3(0, 0, 0);
    }

    __host__ __device__ void normalize() {
        float len = length();
        if (len > 0) {
            x /= len;
            y /= len;
            z /= len;
        }
    }
};

// Non-member functions
__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return v * t;
}

__host__ __device__ inline float dot(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(const vec3 &v) { return v.length(); }

__host__ __device__ inline vec3 normalize(const vec3 &v) {
    return v.normalized();
}

__host__ __device__ inline vec3 cross(const vec3 &a, const vec3 &b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__host__ __device__ inline vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3 &v, const vec3 &n,
                                        float eta) {
    float cos_theta = fminf(dot(-v, n), 1.0f);
    vec3 r_out_perp = eta * (v + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// Utility functions
__host__ __device__ inline vec3 lerp(const vec3 &a, const vec3 &b, float t) {
    return (1.0f - t) * a + t * b;
}

__host__ __device__ inline vec3 clamp(const vec3 &v, float min_val,
                                      float max_val) {
    return vec3(fminf(fmaxf(v.x, min_val), max_val),
                fminf(fmaxf(v.y, min_val), max_val),
                fminf(fmaxf(v.z, min_val), max_val));
}

// Type aliases
using point3 = vec3;
using color = vec3;

// Stream output (host only)
inline std::ostream &operator<<(std::ostream &os, const vec3 &v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

#endif