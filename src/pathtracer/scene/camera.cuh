// camera cuh
// camera model and ray generation utilities for the renderer
// supports depth of field and basic orbit style animation
// stores view projection matrices for downstream motion vectors and denoiser
// inputs
#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "common/ray.cuh"
#include "common/vec3.cuh"
#include <cuda_runtime.h>

#include <curand_kernel.h>

#include "common/mat4.cuh"
#include "pathtracer/math/mathutils.cuh"

// random in unit disk
// implements a unit of behavior used by higher level scene code
// inputs state
// returns vec3

__device__ inline vec3 random_in_unit_disk(curandState *state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(state), curand_uniform(state), 0.0f) -
            vec3(1.0f, 1.0f, 0.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class Camera {
  private:
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;

    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 inv_view_proj_matrix;

    float fov;
    float aspect;
    float near_clip;
    float far_clip;

    // random in unit disk hash
    // implements a unit of behavior used by higher level scene code
    // inputs x y
    // returns vec3

    __host__ __device__ inline vec3 random_in_unit_disk_hash(uint32_t x,
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

    __host__ __device__ void
    // update matrices
    // implements a unit of behavior used by higher level scene code
    // inputs lookfrom lookat vup
    // returns value

    update_matrices(const vec3 &lookfrom, const vec3 &lookat, const vec3 &vup) {
        view_matrix = mat4::lookAt(lookfrom, lookat, vup);
        proj_matrix =
            mat4::perspective(fov * (PI / 180.0f), aspect, near_clip, far_clip);

        mat4 view_proj = proj_matrix * view_matrix;
        inv_view_proj_matrix = view_proj.inverse();
    }

  public:
    // camera
    // implements a unit of behavior used by higher level scene code
    // inputs lookfrom lookat vup vfov aspect ratio aperture focus dist znear
    // zfar returns device

    __host__ __device__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov,
                               float aspect_ratio, float aperture = 0.0f,
                               float focus_dist = 1.0f, float znear = 0.1f,
                               float zfar = 1000.0f) {
        float theta = vfov * (PI / 180.0f);
        float h = tanf(theta / 2.0f);
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

        lens_radius = aperture / 2.0f;

        fov = vfov;
        aspect = aspect_ratio;
        near_clip = znear;
        far_clip = zfar;
        update_matrices(lookfrom, lookat, vup);
    }

    // camera
    // implements a unit of behavior used by higher level scene code
    // inputs aspect ratio viewport height focal length
    // returns device

    __host__ __device__ Camera(float aspect_ratio, float viewport_height = 2.0f,
                               float focal_length = 1.0f) {
        origin = vec3(0.0f, 0.0f, 0.0f);

        float viewport_width = viewport_height * aspect_ratio;

        horizontal = vec3(viewport_width, 0.0f, 0.0f);
        vertical = vec3(0.0f, viewport_height, 0.0f);
        lower_left_corner = origin - horizontal * 0.5f - vertical * 0.5f -
                            vec3(0.0f, 0.0f, focal_length);

        lens_radius = 0.0f;

        u = vec3(1, 0, 0);
        v = vec3(0, 1, 0);
        w = vec3(0, 0, 1);
        fov = 90.0f;
        aspect = aspect_ratio;
        near_clip = 0.1f;
        far_clip = 100.0f;
        update_matrices(origin, vec3(0, 0, -1), vec3(0, 1, 0));
    }

    // get ray
    // returns cached state for downstream code without modifying scene
    // inputs s t state
    // returns ray

    __device__ Ray get_ray(float s, float t, curandState *state) const {
        if (lens_radius <= 0)
            return get_ray_simple(s, t);

        vec3 rd = lens_radius * random_in_unit_disk(state);
        vec3 offset = u * rd.x + v * rd.y;
        vec3 ray_dir =
            lower_left_corner + s * horizontal + t * vertical - origin - offset;

        return Ray(origin + offset, ray_dir.normalized(), true);
    }

    // get ray
    // returns cached state for downstream code without modifying scene
    // inputs s t
    // returns ray

    __host__ __device__ Ray get_ray(float s, float t) const {
        if (lens_radius <= 0)
            return get_ray_simple(s, t);

#ifdef __CUDA_ARCH__
        uint32_t x = (uint32_t)(s * 10000.0f) + (uint32_t)(t * 5000.0f);
        uint32_t y = (uint32_t)(t * 10000.0f) + (uint32_t)(s * 5000.0f);
        vec3 rd = lens_radius * random_in_unit_disk_hash(x, y);
        vec3 offset = u * rd.x + v * rd.y;
        vec3 ray_dir =
            lower_left_corner + s * horizontal + t * vertical - origin - offset;

        return Ray(origin + offset, ray_dir.normalized(), false);
#else
        vec3 rd = lens_radius * random_in_unit_disk_hash((uint32_t)(s * 1e6f),
                                                         (uint32_t)(t * 1e6f));
        vec3 offset = u * rd.x + v * rd.y;
        vec3 ray_dir =
            lower_left_corner + s * horizontal + t * vertical - origin - offset;
        return Ray(origin + offset, ray_dir.normalized(), false);
#endif
    }

    // get ray simple
    // returns cached state for downstream code without modifying scene
    // inputs s t
    // returns ray

    __host__ __device__ Ray get_ray_simple(float s, float t) const {
        vec3 ray_direction =
            lower_left_corner + s * horizontal + t * vertical - origin;
        return Ray(origin, ray_direction.normalized(), true);
    }

    // get origin
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns vec3

    __host__ __device__ vec3 get_origin() const { return origin; }
    // get lower left corner
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns vec3

    __host__ __device__ vec3 get_lower_left_corner() const {
        return lower_left_corner;
    }
    // get horizontal
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns vec3

    __host__ __device__ vec3 get_horizontal() const { return horizontal; }
    // get vertical
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns vec3

    __host__ __device__ vec3 get_vertical() const { return vertical; }

    // get view
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns mat4

    __host__ __device__ const mat4 &get_view() const { return view_matrix; }
    // get proj
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns mat4

    __host__ __device__ const mat4 &get_proj() const { return proj_matrix; }
    // get inv view proj
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns mat4

    __host__ __device__ const mat4 &get_inv_view_proj() const {
        return inv_view_proj_matrix;
    }
    // get view proj
    // returns cached state for downstream code without modifying scene
    // inputs none
    // returns mat4

    __host__ __device__ mat4 get_view_proj() const {
        return proj_matrix * view_matrix;
    }

    // set position
    // updates internal state and marks dependent data as dirty
    // inputs pos
    // returns none

    __host__ __device__ void set_position(const vec3 &pos) {
        vec3 old_center =
            lower_left_corner + 0.5f * horizontal + 0.5f * vertical;
        float focus_dist = (origin - old_center).length();
        vec3 lookat = origin - w * focus_dist;
        vec3 vup = v;

        origin = pos;

        w = (origin - lookat).normalized();
        u = cross(vup, w).normalized();
        v = cross(w, u);

        float theta = fov * (PI / 180.0f);
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect * viewport_height;

        focus_dist = (origin - lookat).length();

        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner =
            origin - horizontal * 0.5f - vertical * 0.5f - focus_dist * w;

        update_matrices(origin, lookat, vup);
    }

    // look at
    // implements a unit of behavior used by higher level scene code
    // inputs target vup 1 0
    // returns none

    __host__ __device__ void look_at(const vec3 &target,
                                     // vec3
                                     // implements a unit of behavior used by
                                     // higher level scene code inputs 0 1 0
                                     // returns value

                                     const vec3 &vup = vec3(0, 1, 0)) {
        w = (origin - target).normalized();
        u = cross(vup, w).normalized();
        v = cross(w, u);

        float focus_dist = (origin - target).length();
        float theta = fov * (PI / 180.0f);
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect * viewport_height;

        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner =
            origin - horizontal * 0.5f - vertical * 0.5f - focus_dist * w;

        update_matrices(origin, target, vup);
    }
};

#endif