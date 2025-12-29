// sampling.cuh
// sampling helpers for cuda path tracing including random number generation and
// direction sampling provides orthonormal basis construction and common
// hemisphere sphere cone and ggx sampling routines

#ifndef SAMPLING_CUH
#define SAMPLING_CUH

#include "common/bluenoise.cuh"
#include "common/vec3.cuh"
#include "pathtracer/math/mathutils.cuh"
#include <curand_kernel.h>

// samples a precomputed blue noise texture using integer pixel coordinates
__device__ __forceinline__ float2 next_blue_noise(int x, int y, int frame) {
    int bx = x & (BLUE_NOISE_SIZE - 1); // equivalent to x % 64
    int by = y & (BLUE_NOISE_SIZE - 1);

    float val_x = d_blue_noise[by][bx][0];
    float val_y = d_blue_noise[by][bx][1];

    uint32_t hash = (uint32_t)frame * 0x9e3779b9u; // Golden Ratio hash

    hash ^= (hash >> 15);
    hash *= 0x85ebca6bu;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35u;
    hash ^= (hash >> 16);
    float shift_x = (hash & 0xFFFFFF) * (1.0f / 16777216.0f);

    hash *= 0x85ebca6bu;
    float shift_y = (hash & 0xFFFFFF) * (1.0f / 16777216.0f);

    float u = val_x + shift_x;
    float v = val_y + shift_y;

    if (u >= 1.0f)
        u -= 1.0f;
    if (v >= 1.0f)
        v -= 1.0f;

    return make_float2(u, v);
}

struct FastRNG {
    uint32_t state;

    // constructs a small fast random number generator for sampling on device
    __device__ __forceinline__ FastRNG(uint32_t seed) : state(seed) {}

    // constructs a small fast random number generator for sampling on device
    __device__ __forceinline__ FastRNG(int x, int y, int frame) {
        state = (x * 1973u) ^ (y * 9277u) ^ (frame * 26699u) ^ 0x9e3779b9u;
        uniform();
        uniform();
    }

    // returns a uniform random float in the range 0 to 1
    __device__ __forceinline__ float uniform() {
        state = state * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        word = (word >> 22u) ^ word;
        return (float)word * 2.3283064365386963e-10f; // / 4294967296.0f
    }

    // returns a pair of uniform random floats in the range 0 to 1
    __device__ __forceinline__ float2 uniform2() {
        return make_float2(uniform(), uniform());
    }
};

// builds an orthonormal tangent frame from a normal vector
__device__ __forceinline__ void createOrthoNormalBasis(const vec3 &N, vec3 &T,
                                                       vec3 &B) {
    float len2 = dot(N, N);
    if (len2 < 1e-20f) {
        T = vec3(1.0f, 0.0f, 0.0f);
        B = vec3(0.0f, 1.0f, 0.0f);
        return;
    }

    vec3 Nn = N * rsqrtf(len2);

    // Frisvad style basis construction for a unit normal
    float s = copysignf(1.0f, Nn.z);
    float a = -1.0f / (s + Nn.z);
    float b = Nn.x * Nn.y * a;

    T = vec3(1.0f + s * Nn.x * Nn.x * a, s * b, -s * Nn.x);
    B = cross(Nn, T);
}

// builds an orthonormal frame from a normal using a robust fallback method
__device__ __forceinline__ void createOrthoNormalBasis_safe(const vec3 &N,
                                                            vec3 &T, vec3 &B) {
    if (fabsf(N.x) > fabsf(N.z)) {
        T = vec3(-N.y, N.x, 0.0f).normalized();
    } else {
        T = vec3(0.0f, -N.z, N.y).normalized();
    }
    B = cross(N, T);
}

// samples a direction inside a cone around a given axis
__device__ __forceinline__ vec3 sample_cone_direction(curandState *state,
                                                      const vec3 &cone_dir,
                                                      float cos_theta_max) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);

    float cos_theta = 1.0f - u1 * (1.0f - cos_theta_max);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = TWO_PI * u2;

    vec3 sample_dir(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);

    vec3 T, B;
    createOrthoNormalBasis(cone_dir, T, B);
    return sample_dir.x * T + sample_dir.y * B + sample_dir.z * cone_dir;
}

// samples a direction inside a cone around a given axis
__device__ __forceinline__ vec3 sample_cone_direction(FastRNG &rng,
                                                      const vec3 &cone_dir,
                                                      float cos_theta_max) {
    float u1 = rng.uniform();
    float u2 = rng.uniform();

    float cos_theta = 1.0f - u1 * (1.0f - cos_theta_max);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = TWO_PI * u2;

    vec3 sample_dir(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);

    vec3 T, B;
    createOrthoNormalBasis(cone_dir, T, B);
    return sample_dir.x * T + sample_dir.y * B + sample_dir.z * cone_dir;
}

// samples a cosine weighted direction in the local hemisphere
__device__ __forceinline__ vec3 sample_cosine_hemisphere(curandState *state) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    float r = sqrtf(u1);
    float phi = TWO_PI * u2;
    return vec3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1)));
}

// samples a cosine weighted direction in the local hemisphere
__device__ __forceinline__ vec3 sample_cosine_hemisphere(FastRNG &rng) {
    float u1 = rng.uniform();
    float u2 = rng.uniform();
    float r = sqrtf(u1);
    float phi = TWO_PI * u2;
    return vec3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1)));
}

// converts a local hemisphere direction into world space using a normal frame
__device__ __forceinline__ vec3 hemisphere_to_world(const vec3 &sample,
                                                    const vec3 &N) {
    vec3 T, B;
    createOrthoNormalBasis(N, T, B);
    return sample.x * T + sample.y * B + sample.z * N;
}

// samples a uniform direction on the unit sphere
__device__ __forceinline__ vec3 sample_unit_sphere(curandState *state) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    float z = 1.0f - 2.0f * u1;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * u2;
    return vec3(r * cosf(phi), r * sinf(phi), z);
}

// samples a uniform direction on the unit sphere
__device__ __forceinline__ vec3 sample_unit_sphere(FastRNG &rng) {
    float u1 = rng.uniform();
    float u2 = rng.uniform();
    float z = 1.0f - 2.0f * u1;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * u2;
    return vec3(r * cosf(phi), r * sinf(phi), z);
}

// importance samples a ggx microfacet normal distribution
__device__ __forceinline__ vec3 importance_sample_ggx(curandState *state,
                                                      const vec3 &N,
                                                      float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;

    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);

    u2 = fminf(u2, 0.9999999f);

    float phi = TWO_PI * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (a2 - 1.0f) * u2));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    vec3 H;
    H.x = sinTheta * cosf(phi);
    H.y = sinTheta * sinf(phi);
    H.z = cosTheta;

    return hemisphere_to_world(H, N);
}

// importance samples a ggx microfacet normal distribution
__device__ __forceinline__ vec3 importance_sample_ggx(FastRNG &rng,
                                                      const vec3 &N,
                                                      float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;

    float u1 = rng.uniform();
    float u2 = fminf(rng.uniform(), 0.9999999f);

    float phi = TWO_PI * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (a2 - 1.0f) * u2));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    vec3 H(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

    return hemisphere_to_world(H, N);
}

// computes sine and cosine together using a fast intrinsic when available
__device__ __forceinline__ void sincosf_fast(float angle, float *s, float *c) {
    sincosf(angle, s, c); // CUDA intrinsic - computes both in one call
}

#endif // SAMPLING_CUH
