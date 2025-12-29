#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "common\vec3.cuh"
#include <cmath>
#include <cuda_runtime.h>

class mat3 {
  public:
    float m[3][3];

    // Constructors
    __host__ __device__ mat3() {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    __host__ __device__ mat3(float m00, float m01, float m02, float m10,
                             float m11, float m12, float m20, float m21,
                             float m22) {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
    }

    // Matrix-vector multiplication
    __host__ __device__ vec3 operator*(const vec3 &v) const {
        return vec3(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                    m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                    m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
    }

    // Matrix-matrix multiplication
    __host__ __device__ mat3 operator*(const mat3 &other) const {
        mat3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }

    // Transpose
    __host__ __device__ mat3 transpose() const {
        return mat3(m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1],
                    m[0][2], m[1][2], m[2][2]);
    }

    // Determinant
    __host__ __device__ float determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    // Inverse
    __host__ __device__ mat3 inverse() const {
        float det = determinant();
        if (fabsf(det) < 1e-6f)
            return mat3(); // Return identity if not invertible

        float inv_det = 1.0f / det;
        return mat3((m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                    (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                    (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
                    (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                    (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                    (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
                    (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                    (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                    (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det);
    }
};

// Factory functions for common transformations
__host__ __device__ inline mat3 rotation_x(float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return mat3(1, 0, 0, 0, c, -s, 0, s, c);
}

__host__ __device__ inline mat3 rotation_y(float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return mat3(c, 0, s, 0, 1, 0, -s, 0, c);
}

__host__ __device__ inline mat3 rotation_z(float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return mat3(c, -s, 0, s, c, 0, 0, 0, 1);
}

__host__ __device__ inline mat3 scale(float sx, float sy, float sz) {
    return mat3(sx, 0, 0, 0, sy, 0, 0, 0, sz);
}

__host__ __device__ inline mat3 scale(float s) { return scale(s, s, s); }

// Rotation matrix from axis and angle
__host__ __device__ inline mat3 rotation_axis_angle(const vec3 &axis,
                                                    float angle) {
    vec3 a = axis.normalized();
    float c = cosf(angle);
    float s = sinf(angle);
    float t = 1.0f - c;

    return mat3(
        t * a.x * a.x + c, t * a.x * a.y - s * a.z, t * a.x * a.z + s * a.y,
        t * a.x * a.y + s * a.z, t * a.y * a.y + c, t * a.y * a.z - s * a.x,
        t * a.x * a.z - s * a.y, t * a.y * a.z + s * a.x, t * a.z * a.z + c);
}

// Look-at matrix (for camera transformations)
__host__ __device__ inline mat3 look_at(const vec3 &forward, const vec3 &up) {
    vec3 f = forward.normalized();
    vec3 r = cross(f, up).normalized();
    vec3 u = cross(r, f);

    return mat3(r.x, r.y, r.z, u.x, u.y, u.z, -f.x, -f.y, -f.z);
}

#endif // MATRIX_CUH