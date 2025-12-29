#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "common/ray.cuh"
#include "common/vec3.cuh"
#include <cuda_runtime.h>

#include "common/bluenoise.cuh"

class Camera {
  private:
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;

    // Precomputed for fast ray generation
    vec3 corner_minus_origin;

    __device__ __forceinline__ vec3
    random_in_unit_disk_blue(int pixel_x, int pixel_y, int sample_index) const {
        int tx = (pixel_x + sample_index * 17) & 63;
        int ty = (pixel_y + sample_index * 29) & 63;

        float r1 = d_blue_noise[ty][tx][0];
        float r2 = d_blue_noise[ty][tx][1];

        float phi_offset = sample_index * 0.618033988749895f;
        r1 = fmodf(r1 + phi_offset, 1.0f);
        r2 = fmodf(r2 + phi_offset * 0.381966011250105f, 1.0f);

        float r = sqrtf(r1);
        float theta = 6.28318530718f * r2;

        float s, c;
#ifdef __CUDA_ARCH__
        __sincosf(theta, &s, &c);
#else
        s = sinf(theta);
        c = cosf(theta);
#endif

        return vec3(r * c, r * s, 0.0f);
    }

    __host__ __forceinline__ vec3 random_in_unit_disk_hash(uint32_t x,
                                                           uint32_t y) const {
        uint32_t seed = (x * 1973u) ^ (y * 9277u) ^ 0x9e3779b9u;
        seed ^= seed >> 17;
        seed *= 0xed5ad4bbu;
        seed ^= seed >> 11;
        seed *= 0xac4c1b51u;
        seed ^= seed >> 15;
        seed *= 0x31848babu;
        seed ^= seed >> 14;

        float r1 = ((seed & 0xFFFFu) + 0.5f) / 65536.0f;
        float r2 = (((seed * 0x343fdu + 0xc0f5u) & 0xFFFFu) + 0.5f) / 65536.0f;

        float r = sqrtf(r1);
        float phi = 6.2831853f * r2;
        return vec3(r * cosf(phi), r * sinf(phi), 0.0f);
    }

  public:
    __host__ __device__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov,
                               float aspect_ratio, float aperture = 0.0f,
                               float focus_dist = 1.0f) {
        float theta = vfov * 0.01745329251994329577f; // deg to rad
        float h = tanf(theta * 0.5f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = (lookfrom - lookat).normalized();
        u = cross(vup, w).normalized();
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner =
            origin - horizontal * 0.5f - vertical * 0.5f - focus_dist * w;
        corner_minus_origin = lower_left_corner - origin;

        lens_radius = aperture * 0.5f;
    }

    __host__ __device__ Camera(float aspect_ratio, float viewport_height = 2.0f,
                               float focal_length = 1.0f) {
        origin = vec3(0.0f, 0.0f, 0.0f);
        float viewport_width = viewport_height * aspect_ratio;

        horizontal = vec3(viewport_width, 0.0f, 0.0f);
        vertical = vec3(0.0f, -viewport_height, 0.0f);
        lower_left_corner = origin - horizontal * 0.5f - vertical * 0.5f -
                            vec3(0.0f, 0.0f, focal_length);
        corner_minus_origin = lower_left_corner - origin;

        lens_radius = 0.0f;
        u = vec3(1, 0, 0);
        v = vec3(0, 1, 0);
        w = vec3(0, 0, 1);
    }

    // Fast ray generation with DOF
    __device__ __forceinline__ Ray get_ray(float s, float t, int pixel_x,
                                           int pixel_y,
                                           int sample_index = 0) const {
        if (lens_radius <= 0.0f) {
// No DOF - fast path using FMA
#ifdef __CUDA_ARCH__
            vec3 dir;
            dir.x = __fmaf_rn(s, horizontal.x,
                              __fmaf_rn(t, vertical.x, corner_minus_origin.x));
            dir.y = __fmaf_rn(s, horizontal.y,
                              __fmaf_rn(t, vertical.y, corner_minus_origin.y));
            dir.z = __fmaf_rn(s, horizontal.z,
                              __fmaf_rn(t, vertical.z, corner_minus_origin.z));
#else
            vec3 dir = corner_minus_origin + s * horizontal + t * vertical;
#endif
            return Ray(origin, dir.normalized());
        }

        vec3 rd = lens_radius *
                  random_in_unit_disk_blue(pixel_x, pixel_y, sample_index);
        vec3 offset = u * rd.x + v * rd.y;
        vec3 ray_dir =
            lower_left_corner + s * horizontal + t * vertical - origin - offset;
        return Ray(origin + offset, ray_dir.normalized());
    }

    // Simple ray generation (no DOF parameters)
    __host__ __device__ __forceinline__ Ray get_ray(float s, float t) const {
        if (lens_radius <= 0.0f) {
            vec3 dir = corner_minus_origin + s * horizontal + t * vertical;
            return Ray(origin, dir.normalized());
        }

#ifdef __CUDA_ARCH__
        // On device without pixel coords, just use simple ray
        vec3 dir = corner_minus_origin + s * horizontal + t * vertical;
        return Ray(origin, dir.normalized());
#else
        vec3 rd = lens_radius * random_in_unit_disk_hash((uint32_t)(s * 1e6f),
                                                         (uint32_t)(t * 1e6f));
        vec3 offset = u * rd.x + v * rd.y;
        vec3 ray_dir =
            lower_left_corner + s * horizontal + t * vertical - origin - offset;
        return Ray(origin + offset, ray_dir.normalized());
#endif
    }

    __host__ __device__ __forceinline__ Ray get_ray_simple(float s,
                                                           float t) const {
        vec3 ray_direction =
            corner_minus_origin + s * horizontal + t * vertical;
        return Ray(origin, ray_direction.normalized());
    }

    __host__ __device__ __forceinline__ vec3 get_origin() const {
        return origin;
    }
    __host__ __device__ __forceinline__ vec3 get_lower_left_corner() const {
        return lower_left_corner;
    }
    __host__ __device__ __forceinline__ vec3 get_horizontal() const {
        return horizontal;
    }
    __host__ __device__ __forceinline__ vec3 get_vertical() const {
        return vertical;
    }

    __host__ __device__ void set_position(const vec3 &pos) {
        vec3 delta = pos - origin;
        origin = pos;
        lower_left_corner = lower_left_corner + delta;
        corner_minus_origin = lower_left_corner - origin;
    }

    __host__ __device__ void look_at(const vec3 &target,
                                     const vec3 &vup = vec3(0, 1, 0)) {
        w = (origin - target).normalized();
        u = cross(vup, w).normalized();
        v = cross(w, u);

        float viewport_height = vertical.length();
        float viewport_width = horizontal.length();
        float focus_dist = (origin - target).length();

        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner =
            origin - horizontal * 0.5f - vertical * 0.5f - focus_dist * w;
        corner_minus_origin = lower_left_corner - origin;
    }
};

#endif
