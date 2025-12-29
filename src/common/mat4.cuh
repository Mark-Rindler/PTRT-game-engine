#ifndef MAT4_CUH
#define MAT4_CUH

#include "common\vec3.cuh"
#include "common\vec4.cuh"
#include "pathtracer\math\mathutils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

/**
 * @class mat4
 * @brief A 4x4 matrix class for CUDA, usable on host and device.
 *
 * Stores data in column-major order, which is standard for OpenGL/Vulkan
 * and generally efficient for matrix-vector multiplication.
 */
class mat4 {
  public:
    // Column-major storage
    float m[16];

    // Constructors

    /**
     * @brief Default constructor. Initializes to an identity matrix.
     */
    __host__ __device__ mat4() {
        m[0] = 1.f;
        m[4] = 0.f;
        m[8] = 0.f;
        m[12] = 0.f;
        m[1] = 0.f;
        m[5] = 1.f;
        m[9] = 0.f;
        m[13] = 0.f;
        m[2] = 0.f;
        m[6] = 0.f;
        m[10] = 1.f;
        m[14] = 0.f;
        m[3] = 0.f;
        m[7] = 0.f;
        m[11] = 0.f;
        m[15] = 1.f;
    }

    /**
     * @brief Constructs from 16 floats in column-major order.
     */
    __host__ __device__ mat4(float m0, float m1, float m2, float m3, float m4,
                             float m5, float m6, float m7, float m8, float m9,
                             float m10, float m11, float m12, float m13,
                             float m14, float m15) {
        m[0] = m0;
        m[1] = m1;
        m[2] = m2;
        m[3] = m3;
        m[4] = m4;
        m[5] = m5;
        m[6] = m6;
        m[7] = m7;
        m[8] = m8;
        m[9] = m9;
        m[10] = m10;
        m[11] = m11;
        m[12] = m12;
        m[13] = m13;
        m[14] = m14;
        m[15] = m15;
    }

    /**
     * @brief Constructs from four column vectors.
     */
    __host__ __device__ mat4(const vec4 &col0, const vec4 &col1,
                             const vec4 &col2, const vec4 &col3) {
        m[0] = col0.x;
        m[1] = col0.y;
        m[2] = col0.z;
        m[3] = col0.w;
        m[4] = col1.x;
        m[5] = col1.y;
        m[6] = col1.z;
        m[7] = col1.w;
        m[8] = col2.x;
        m[9] = col2.y;
        m[10] = col2.z;
        m[11] = col2.w;
        m[12] = col3.x;
        m[13] = col3.y;
        m[14] = col3.z;
        m[15] = col3.w;
    }

    // Accessors

    /**
     * @brief Access a column by index.
     */
    __host__ __device__ vec4 get_col(int i) const {
        const float *col_ptr = &m[i * 4];
        return vec4(col_ptr[0], col_ptr[1], col_ptr[2], col_ptr[3]);
    }

    /**
     * @brief Access a row by index (less efficient).
     */
    __host__ __device__ vec4 get_row(int i) const {
        return vec4(m[i], m[i + 4], m[i + 8], m[i + 12]);
    }

    // Static Creation Methods

    __host__ __device__ static mat4 identity() {
        return mat4(); // Default constructor is identity
    }

    /**
     * @brief Creates a translation matrix.
     */
    __host__ __device__ static mat4 translate(const vec3 &t) {
        mat4 r = mat4::identity();
        r.m[12] = t.x;
        r.m[13] = t.y;
        r.m[14] = t.z;
        return r;
    }

    /**
     * @brief Creates a scale matrix.
     */
    __host__ __device__ static mat4 scale(const vec3 &s) {
        mat4 r = mat4::identity();
        r.m[0] = s.x;
        r.m[5] = s.y;
        r.m[10] = s.z;
        return r;
    }

    /**
     * @brief Creates a right-handed view matrix.
     */
    __host__ __device__ static mat4 lookAt(const vec3 &eye, const vec3 &center,
                                           const vec3 &up) {
        vec3 f = normalize(center - eye); // Forward (points to center)
        vec3 s = normalize(cross(f, up)); // Right
        vec3 u = cross(s, f);             // Up

        mat4 r = mat4::identity();
        r.m[0] = s.x;
        r.m[4] = s.y;
        r.m[8] = s.z;
        r.m[1] = u.x;
        r.m[5] = u.y;
        r.m[9] = u.z;
        r.m[2] = -f.x; // Note: -f for right-handed look-at
        r.m[6] = -f.y;
        r.m[10] = -f.z;
        r.m[12] = -dot(s, eye);
        r.m[13] = -dot(u, eye);
        r.m[14] = dot(f, eye); // Note: dot(f, eye)
        return r;
    }

    /**
     * @brief Creates a right-handed perspective projection matrix.
     */
    __host__ __device__ static mat4
    perspective(float fov_y_radians, float aspect, float zNear, float zFar) {
        mat4 r = mat4::identity();
        float const tanHalfFovy = tanf(fov_y_radians / 2.0f);

        r.m[0] = 1.0f / (aspect * tanHalfFovy);
        r.m[1] = 0.0f;
        r.m[2] = 0.0f;
        r.m[3] = 0.0f;

        r.m[4] = 0.0f;
        r.m[5] = 1.0f / (tanHalfFovy);
        r.m[6] = 0.0f;
        r.m[7] = 0.0f;

        r.m[8] = 0.0f;
        r.m[9] = 0.0f;
        // Standard perspective matrix (maps z to [-1, 1])
        r.m[10] = -(zFar + zNear) / (zFar - zNear);
        r.m[11] = -1.0f;

        r.m[12] = 0.0f;
        r.m[13] = 0.0f;
        r.m[14] = -(2.0f * zFar * zNear) / (zFar - zNear);
        r.m[15] = 0.0f;

        return r;
    }

    // Member Functions

    /**
     * @brief Returns a transposed copy of this matrix.
     */
    __host__ __device__ mat4 transpose() const {
        return mat4(m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2],
                    m[6], m[10], m[14], m[3], m[7], m[11], m[15]);
    }

    /**
     * @brief Returns an inverted copy of this matrix.
     * This is a full analytic inverse, crucial for un-projection.
     */
    __host__ __device__ mat4 inverse() const {
        float A2323 = m[10] * m[15] - m[11] * m[14];
        float A1323 = m[9] * m[15] - m[11] * m[13];
        float A1223 = m[9] * m[14] - m[10] * m[13];
        float A0323 = m[8] * m[15] - m[11] * m[12];
        float A0223 = m[8] * m[14] - m[10] * m[12];
        float A0123 = m[8] * m[13] - m[9] * m[12];
        float A2313 = m[6] * m[15] - m[7] * m[14];
        float A1313 = m[5] * m[15] - m[7] * m[13];
        float A1213 = m[5] * m[14] - m[6] * m[13];
        float A0313 = m[4] * m[15] - m[7] * m[12];
        float A0213 = m[4] * m[14] - m[6] * m[12];
        float A0113 = m[4] * m[13] - m[5] * m[12];
        float A2312 = m[6] * m[11] - m[7] * m[10];
        float A1312 = m[5] * m[11] - m[7] * m[9];
        float A1212 = m[5] * m[10] - m[6] * m[9];
        float A0312 = m[4] * m[11] - m[7] * m[8];
        float A0212 = m[4] * m[10] - m[6] * m[8];
        float A0112 = m[4] * m[9] - m[5] * m[8];

        float det = m[0] * (m[5] * A2323 - m[6] * A1323 + m[7] * A1223) -
                    m[1] * (m[4] * A2323 - m[6] * A0323 + m[7] * A0223) +
                    m[2] * (m[4] * A1323 - m[5] * A0323 + m[7] * A0123) -
                    m[3] * (m[4] * A1223 - m[5] * A0223 + m[6] * A0123);

        if (fabsf(det) < 1e-10f) {
            // Return identity or some error matrix
            return mat4::identity();
        }

        float invDet = 1.0f / det;

        return mat4(invDet * (m[5] * A2323 - m[6] * A1323 + m[7] * A1223),
                    invDet * -(m[1] * A2323 - m[2] * A1323 + m[3] * A1223),
                    invDet * (m[1] * A2313 - m[2] * A1313 + m[3] * A1213),
                    invDet * -(m[1] * A2312 - m[2] * A1312 + m[3] * A1212),

                    invDet * -(m[4] * A2323 - m[6] * A0323 + m[7] * A0223),
                    invDet * (m[0] * A2323 - m[2] * A0323 + m[3] * A0223),
                    invDet * -(m[0] * A2313 - m[2] * A0313 + m[3] * A0113),
                    invDet * (m[0] * A2312 - m[2] * A0312 + m[3] * A0112),

                    invDet * (m[4] * A1323 - m[5] * A0323 + m[7] * A0123),
                    invDet * -(m[0] * A1323 - m[1] * A0323 + m[3] * A0123),
                    invDet * (m[0] * A1313 - m[1] * A0313 + m[3] * A0113),
                    invDet * -(m[0] * A1312 - m[1] * A0312 + m[3] * A0112),

                    invDet * -(m[4] * A1223 - m[5] * A0223 + m[6] * A0123),
                    invDet * (m[0] * A1223 - m[1] * A0223 + m[2] * A0123),
                    invDet * -(m[0] * A1213 - m[1] * A0213 + m[2] * A0113),
                    invDet * (m[0] * A1212 - m[1] * A0212 + m[2] * A0112));
    }
};

// Operator Overloads

/**
 * @brief Matrix-Vector multiplication (column-major).
 */
__host__ __device__ inline vec4 operator*(const mat4 &m, const vec4 &v) {
    return vec4(m.m[0] * v.x + m.m[4] * v.y + m.m[8] * v.z + m.m[12] * v.w,
                m.m[1] * v.x + m.m[5] * v.y + m.m[9] * v.z + m.m[13] * v.w,
                m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
                m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w);
}

/**
 * @brief Matrix-Matrix multiplication.
 */
__host__ __device__ inline mat4 operator*(const mat4 &a, const mat4 &b) {
    mat4 r;
    // r.col[0] = a * b.col[0]
    r.m[0] =
        a.m[0] * b.m[0] + a.m[4] * b.m[1] + a.m[8] * b.m[2] + a.m[12] * b.m[3];
    r.m[1] =
        a.m[1] * b.m[0] + a.m[5] * b.m[1] + a.m[9] * b.m[2] + a.m[13] * b.m[3];
    r.m[2] =
        a.m[2] * b.m[0] + a.m[6] * b.m[1] + a.m[10] * b.m[2] + a.m[14] * b.m[3];
    r.m[3] = a.m[3] * b.m[0] + a.m[7] * b.m[11] + a.m[11] * b.m[2] +
             a.m[15] * b.m[3];

    // r.col[1] = a * b.col[1]
    r.m[4] =
        a.m[0] * b.m[4] + a.m[4] * b.m[5] + a.m[8] * b.m[6] + a.m[12] * b.m[7];
    r.m[5] =
        a.m[1] * b.m[4] + a.m[5] * b.m[5] + a.m[9] * b.m[6] + a.m[13] * b.m[7];
    r.m[6] =
        a.m[2] * b.m[4] + a.m[6] * b.m[5] + a.m[10] * b.m[6] + a.m[14] * b.m[7];
    r.m[7] =
        a.m[3] * b.m[4] + a.m[7] * b.m[5] + a.m[11] * b.m[6] + a.m[15] * b.m[7];

    // r.col[2] = a * b.col[2]
    r.m[8] = a.m[0] * b.m[8] + a.m[4] * b.m[9] + a.m[8] * b.m[10] +
             a.m[12] * b.m[11];
    r.m[9] = a.m[1] * b.m[8] + a.m[5] * b.m[9] + a.m[9] * b.m[10] +
             a.m[13] * b.m[11];
    r.m[10] = a.m[2] * b.m[8] + a.m[6] * b.m[9] + a.m[10] * b.m[10] +
              a.m[14] * b.m[11];
    r.m[11] = a.m[3] * b.m[8] + a.m[7] * b.m[9] + a.m[11] * b.m[10] +
              a.m[15] * b.m[11];

    // r.col[3] = a * b.col[3]
    r.m[12] = a.m[0] * b.m[12] + a.m[4] * b.m[13] + a.m[8] * b.m[14] +
              a.m[12] * b.m[15];
    r.m[13] = a.m[1] * b.m[12] + a.m[5] * b.m[13] + a.m[9] * b.m[14] +
              a.m[13] * b.m[15];
    r.m[14] = a.m[2] * b.m[12] + a.m[6] * b.m[13] + a.m[10] * b.m[14] +
              a.m[14] * b.m[15];
    r.m[15] = a.m[3] * b.m[12] + a.m[7] * b.m[13] + a.m[11] * b.m[14] +
              a.m[15] * b.m[15];
    return r;
}

#endif // MAT4_CUH