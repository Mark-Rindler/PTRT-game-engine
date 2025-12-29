/*
this file implements a cuda based denoiser for a path traced renderer
it defines settings inputs and a denoiser class that owns all gpu buffers
it provides kernels for firefly suppression temporal accumulation variance
estimation and atrous wavelet filtering it uses pixel center coordinates where a
pixel center is x plus one half and y plus one half it reprojects history using
motion vectors in uv space and rejects history using depth normal and optional
object id tests it performs edge aware sampling using depth and normal
thresholds to avoid bleeding across geometry edges it maintains per pixel
temporal moments and a history length value for adaptive blending it expects
depth to be linear ray distance and normals to be unit length with sky pixels
identified by a depth threshold it supports optional object id buffers to reduce
ghosting during disocclusions when ids are available it is intended to be
included in cuda translation units and launched with standard grid and block
configurations
*/

#ifndef DENOISER_CUH
#define DENOISER_CUH

#include "common/mat4.cuh"
#include "common/vec3.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " at "  \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
        }                                                                      \
    } while (0)

struct DenoiserSettings {
    int width;
    int height;

    float diffuse_tau = 0.06f;
    float diffuse_min_alpha = 0.05f;
    float diffuse_max_history = 32.0f;
    float diffuse_sigma_luminance = 4.0f;
    float diffuse_sigma_normal = 64.0f;
    float diffuse_sigma_depth = 0.5f;
    int diffuse_atrous_iterations = 5;
    float diffuse_clamp_scale = 1.2f;
    float diffuse_firefly_threshold = 3.0f;

    float specular_tau = 0.12f;
    float specular_min_alpha = 0.2f;
    float specular_max_history = 6.0f;
    float specular_sigma_luminance = 1.0f;
    float specular_sigma_normal = 128.0f;
    float specular_sigma_depth = 0.2f;
    int specular_atrous_iterations = 2;
    float specular_clamp_scale = 2.0f;
    float specular_firefly_threshold = 8.0f;

    float depth_reject_absolute = 0.1f;
    float depth_reject_relative = 0.005f;
    float normal_reject_threshold = 0.95f;
    float sky_depth_threshold = 1e9f;

    float edge_depth_threshold = 0.01f;
    float edge_normal_threshold = 0.95f;
    bool use_edge_aware_sampling = true;

    bool use_object_ids = true;

    bool enable_firefly_suppression = true;
    bool enable_split_denoising = true;
};

struct DenoiserInputs {
    vec3 *noisyColor;
    vec3 *normal;
    float *depth;
    float2 *motion;
    int *objectId;

    vec3 *diffuseColor;
    vec3 *specularColor;
    vec3 *emissionColor;

    float *transmission;
    float *roughness;

    DenoiserInputs()
        : noisyColor(nullptr), normal(nullptr), depth(nullptr), motion(nullptr),
          objectId(nullptr), diffuseColor(nullptr), specularColor(nullptr),
          emissionColor(nullptr), transmission(nullptr), roughness(nullptr) {}
};

struct DenoiserCommonSettings {
    mat4 viewProj;
    mat4 prevViewProj;
};

__device__ __host__ __forceinline__ float lerpf(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ __host__ __forceinline__ int mini(int a, int b) {
    return a < b ? a : b;
}

__device__ __forceinline__ float luminance(const vec3 &c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ __forceinline__ vec3 safe_normalize(const vec3 &v) {
    float len_sq = dot(v, v);
    if (len_sq < 1e-8f)
        return vec3(0.0f, 0.0f, 1.0f);
    return v * rsqrtf(len_sq);
}

__device__ __forceinline__ bool is_sky(float depth, const vec3 &normal,
                                       float sky_threshold) {
    return (depth > sky_threshold) || (dot(normal, normal) < 0.1f);
}

__device__ __forceinline__ float clampf(float x, float a, float b) {
    return fminf(fmaxf(x, a), b);
}

__device__ __forceinline__ vec3 max_vec3(const vec3 &a, const vec3 &b) {
    return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ vec3 min_vec3(const vec3 &a, const vec3 &b) {
    return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ int clamp_int(int v, int a, int b) {
    return v < a ? a : (v > b ? b : v);
}

__constant__ static float atrous_kernel[25] = {
    1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f,
    1.0f / 256.0f,  4.0f / 256.0f,  16.0f / 256.0f, 24.0f / 256.0f,
    16.0f / 256.0f, 4.0f / 256.0f,  6.0f / 256.0f,  24.0f / 256.0f,
    36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,  4.0f / 256.0f,
    16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
    1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f,
    1.0f / 256.0f};

__device__ __forceinline__ vec3 bilinear_sample_vec3(const vec3 *buf, int width,
                                                     int height, float u,
                                                     float v) {
    float fx = u - 0.5f;
    float fy = v - 0.5f;

    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float sx = fx - x0, sy = fy - y0;

    x0 = clamp_int(x0, 0, width - 1);
    y0 = clamp_int(y0, 0, height - 1);
    x1 = clamp_int(x1, 0, width - 1);
    y1 = clamp_int(y1, 0, height - 1);

    vec3 c00 = buf[y0 * width + x0], c10 = buf[y0 * width + x1];
    vec3 c01 = buf[y1 * width + x0], c11 = buf[y1 * width + x1];

    vec3 c0 = c00 * (1.0f - sx) + c10 * sx;
    vec3 c1 = c01 * (1.0f - sx) + c11 * sx;
    return c0 * (1.0f - sy) + c1 * sy;
}

__device__ __forceinline__ float bilinear_sample_float(const float *buf,
                                                       int width, int height,
                                                       float u, float v) {
    float fx = u - 0.5f;
    float fy = v - 0.5f;

    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float sx = fx - x0, sy = fy - y0;

    x0 = clamp_int(x0, 0, width - 1);
    y0 = clamp_int(y0, 0, height - 1);
    x1 = clamp_int(x1, 0, width - 1);
    y1 = clamp_int(y1, 0, height - 1);

    float c00 = buf[y0 * width + x0], c10 = buf[y0 * width + x1];
    float c01 = buf[y1 * width + x0], c11 = buf[y1 * width + x1];

    float c0 = c00 * (1.0f - sx) + c10 * sx;
    float c1 = c01 * (1.0f - sx) + c11 * sx;
    return c0 * (1.0f - sy) + c1 * sy;
}

__device__ __forceinline__ bool
is_edge_discontinuity(float d0, float d1, const vec3 &n0, const vec3 &n1,
                      int obj0, int obj1, float depth_threshold,
                      float normal_threshold, bool use_obj_id) {

    if (use_obj_id && obj0 != obj1 && obj0 >= 0 && obj1 >= 0) {
        return true;
    }

    float max_d = fmaxf(d0, d1);
    float depth_diff = fabsf(d0 - d1);
    if (max_d > 1e-6f && depth_diff / max_d > depth_threshold) {
        return true;
    }

    float n_dot = dot(n0, n1);
    if (n_dot < normal_threshold) {
        return true;
    }

    return false;
}

__device__ __forceinline__ int
nearest_sample_int(const int *buf, int width, int height, float u, float v) {
    int ix = clamp_int((int)floorf(u), 0, width - 1);
    int iy = clamp_int((int)floorf(v), 0, height - 1);
    return buf[iy * width + ix];
}

__device__ __forceinline__ vec3 edge_aware_bilinear_sample_vec3(
    const vec3 *color_buf, const float *depth_buf, const vec3 *normal_buf,
    const int *obj_buf, int width, int height, float u, float v,
    float center_depth, const vec3 &center_normal, int center_obj,
    float edge_depth_threshold, float edge_normal_threshold, bool use_obj_id) {
    bool use_obj = use_obj_id && (obj_buf != nullptr);

    float fx = u - 0.5f;
    float fy = v - 0.5f;

    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float sx = fx - x0, sy = fy - y0;

    x0 = clamp_int(x0, 0, width - 1);
    y0 = clamp_int(y0, 0, height - 1);
    x1 = clamp_int(x1, 0, width - 1);
    y1 = clamp_int(y1, 0, height - 1);

    int idx00 = y0 * width + x0;
    int idx10 = y0 * width + x1;
    int idx01 = y1 * width + x0;
    int idx11 = y1 * width + x1;

    vec3 c00 = color_buf[idx00], c10 = color_buf[idx10];
    vec3 c01 = color_buf[idx01], c11 = color_buf[idx11];

    float d00 = depth_buf[idx00], d10 = depth_buf[idx10];
    float d01 = depth_buf[idx01], d11 = depth_buf[idx11];

    vec3 n00 = normal_buf[idx00], n10 = normal_buf[idx10];
    vec3 n01 = normal_buf[idx01], n11 = normal_buf[idx11];

    int o00 = use_obj ? obj_buf[idx00] : -1;
    int o10 = use_obj ? obj_buf[idx10] : -1;
    int o01 = use_obj ? obj_buf[idx01] : -1;
    int o11 = use_obj ? obj_buf[idx11] : -1;

    bool valid00 = !is_edge_discontinuity(center_depth, d00, center_normal, n00,
                                          center_obj, o00, edge_depth_threshold,
                                          edge_normal_threshold, use_obj);
    bool valid10 = !is_edge_discontinuity(center_depth, d10, center_normal, n10,
                                          center_obj, o10, edge_depth_threshold,
                                          edge_normal_threshold, use_obj);
    bool valid01 = !is_edge_discontinuity(center_depth, d01, center_normal, n01,
                                          center_obj, o01, edge_depth_threshold,
                                          edge_normal_threshold, use_obj);
    bool valid11 = !is_edge_discontinuity(center_depth, d11, center_normal, n11,
                                          center_obj, o11, edge_depth_threshold,
                                          edge_normal_threshold, use_obj);

    float w00 = valid00 ? (1.0f - sx) * (1.0f - sy) : 0.0f;
    float w10 = valid10 ? sx * (1.0f - sy) : 0.0f;
    float w01 = valid01 ? (1.0f - sx) * sy : 0.0f;
    float w11 = valid11 ? sx * sy : 0.0f;

    float total_w = w00 + w10 + w01 + w11;

    if (total_w < 1e-6f) {
        if (valid00)
            return c00;
        if (valid10)
            return c10;
        if (valid01)
            return c01;
        if (valid11)
            return c11;

        int nearest_x = clamp_int((int)floorf(u), 0, width - 1);
        int nearest_y = clamp_int((int)floorf(v), 0, height - 1);
        return color_buf[nearest_y * width + nearest_x];
    }

    return (c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11) * (1.0f / total_w);
}

__device__ __forceinline__ float edge_aware_bilinear_sample_float(
    const float *buf, const float *depth_buf, const vec3 *normal_buf,
    const int *obj_buf, int width, int height, float u, float v,
    float center_depth, const vec3 &center_normal, int center_obj,
    float edge_depth_threshold, float edge_normal_threshold, bool use_obj_id) {
    bool use_obj = use_obj_id && (obj_buf != nullptr);

    float fx = u - 0.5f;
    float fy = v - 0.5f;

    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float sx = fx - x0, sy = fy - y0;

    x0 = clamp_int(x0, 0, width - 1);
    y0 = clamp_int(y0, 0, height - 1);
    x1 = clamp_int(x1, 0, width - 1);
    y1 = clamp_int(y1, 0, height - 1);

    int idx00 = y0 * width + x0;
    int idx10 = y0 * width + x1;
    int idx01 = y1 * width + x0;
    int idx11 = y1 * width + x1;

    float v00 = buf[idx00], v10 = buf[idx10];
    float v01 = buf[idx01], v11 = buf[idx11];

    float d00 = depth_buf[idx00], d10 = depth_buf[idx10];
    float d01 = depth_buf[idx01], d11 = depth_buf[idx11];

    vec3 n00 = normal_buf[idx00], n10 = normal_buf[idx10];
    vec3 n01 = normal_buf[idx01], n11 = normal_buf[idx11];

    int o00 = use_obj ? obj_buf[idx00] : -1;
    int o10 = use_obj ? obj_buf[idx10] : -1;
    int o01 = use_obj ? obj_buf[idx01] : -1;
    int o11 = use_obj ? obj_buf[idx11] : -1;

    bool valid00 = !is_edge_discontinuity(center_depth, d00, center_normal, n00,
                                          center_obj, o00, edge_depth_threshold,
                                          edge_normal_threshold, use_obj);
    bool valid10 = !is_edge_discontinuity(center_depth, d10, center_normal, n10,
                                          center_obj, o10, edge_depth_threshold,
                                          edge_normal_threshold, use_obj);
    bool valid01 = !is_edge_discontinuity(center_depth, d01, center_normal, n01,
                                          center_obj, o01, edge_depth_threshold,
                                          edge_normal_threshold, use_obj);
    bool valid11 = !is_edge_discontinuity(center_depth, d11, center_normal, n11,
                                          center_obj, o11, edge_depth_threshold,
                                          edge_normal_threshold, use_obj);

    float w00 = valid00 ? (1.0f - sx) * (1.0f - sy) : 0.0f;
    float w10 = valid10 ? sx * (1.0f - sy) : 0.0f;
    float w01 = valid01 ? (1.0f - sx) * sy : 0.0f;
    float w11 = valid11 ? sx * sy : 0.0f;

    float total_w = w00 + w10 + w01 + w11;

    if (total_w < 1e-6f) {
        if (valid00)
            return v00;
        if (valid10)
            return v10;
        if (valid01)
            return v01;
        if (valid11)
            return v11;
        int nearest_x = clamp_int((int)floorf(u), 0, width - 1);
        int nearest_y = clamp_int((int)floorf(v), 0, height - 1);
        return buf[nearest_y * width + nearest_x];
    }

    return (v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11) * (1.0f / total_w);
}

__global__ void firefly_suppression_kernel(
    vec3 *__restrict__ output, const vec3 *__restrict__ input,
    const float *__restrict__ depth, const vec3 *__restrict__ normal,
    float firefly_threshold, float sky_threshold, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;

    vec3 center = input[idx];
    float center_d = depth[idx];
    vec3 center_n = normal[idx];

    if (is_sky(center_d, center_n, sky_threshold)) {
        output[idx] = center;
        return;
    }

    vec3 max_neighbor = vec3(0.0f);
    bool valid_neighbors = false;

#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0)
                continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                vec3 val = input[ny * width + nx];
                max_neighbor = max_vec3(max_neighbor, val);
                valid_neighbors = true;
            }
        }
    }

    if (valid_neighbors) {
        float relaxation = 1.25f;
        vec3 clamped = min_vec3(center, max_neighbor * relaxation);
        const float MAX_RADIANCE = 10.0f;
        clamped = min_vec3(clamped, vec3(MAX_RADIANCE));
        output[idx] = clamped;
    } else {
        output[idx] = center;
    }
}

__global__ void temporal_accumulation_kernel(
    vec3 *__restrict__ out_mean, vec3 *__restrict__ out_m2,
    float *__restrict__ out_history_length,
    const vec3 *__restrict__ current_color, const vec3 *__restrict__ prev_mean,
    const vec3 *__restrict__ prev_m2,
    const float *__restrict__ prev_history_length,
    const float2 *__restrict__ motion, const float *__restrict__ depth,
    const float *__restrict__ prev_depth, const vec3 *__restrict__ normal,
    const vec3 *__restrict__ prev_normal, const int *__restrict__ object_id,
    const int *__restrict__ prev_object_id, float tau, float min_alpha,
    float max_history_length, float depth_reject_abs, float depth_reject_rel,
    float normal_reject_thresh, float clamp_scale, float sky_threshold,
    float edge_depth_thresh, float edge_normal_thresh, bool use_obj_id,
    int width, int height) {
    bool use_obj =
        use_obj_id && (object_id != nullptr) && (prev_object_id != nullptr);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;

    vec3 cur_c = current_color[idx];
    float d = depth[idx];
    vec3 n = normal[idx];
    int obj_id = (use_obj && object_id) ? object_id[idx] : -1;

    if (is_sky(d, n, sky_threshold)) {
        out_mean[idx] = cur_c;
        out_m2[idx] = cur_c * cur_c;
        out_history_length[idx] = 1.0f;
        return;
    }

    vec3 neighborhood_mean = vec3(0.0f);
    vec3 neighborhood_m2 = vec3(0.0f);
    int neighbor_count = 0;

#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = clamp_int(x + dx, 0, width - 1);
            int ny = clamp_int(y + dy, 0, height - 1);
            int n_idx = ny * width + nx;

            vec3 nc = current_color[n_idx];
            float nd = depth[n_idx];
            vec3 nn = normal[n_idx];
            int no = use_obj ? object_id[n_idx] : -1;

            bool same_surface = !is_edge_discontinuity(
                d, nd, n, nn, obj_id, no, edge_depth_thresh, edge_normal_thresh,
                use_obj);

            if (same_surface) {
                neighborhood_mean += nc;
                neighborhood_m2 += nc * nc;
                neighbor_count++;
            }
        }
    }

    if (neighbor_count == 0) {
        neighborhood_mean = cur_c;
        neighborhood_m2 = cur_c * cur_c;
        neighbor_count = 1;
    }

    float inv_neighbor = 1.0f / (float)neighbor_count;
    neighborhood_mean *= inv_neighbor;
    neighborhood_m2 *= inv_neighbor;
    vec3 neighborhood_var = max_vec3(
        neighborhood_m2 - neighborhood_mean * neighborhood_mean, vec3(0.0f));
    vec3 neighborhood_std =
        vec3(sqrtf(neighborhood_var.x), sqrtf(neighborhood_var.y),
             sqrtf(neighborhood_var.z));

    vec3 soft_min = neighborhood_mean - neighborhood_std * clamp_scale;
    vec3 soft_max = neighborhood_mean + neighborhood_std * clamp_scale;

    float2 mv = motion[idx];
    float prev_u = (float)x + 0.5f - mv.x * width;
    float prev_v = (float)y + 0.5f - mv.y * height;

    bool valid_history = true;
    if (prev_u < 0.5f || prev_v < 0.5f || prev_u >= (float)(width - 0.5f) ||
        prev_v >= (float)(height - 0.5f)) {
        valid_history = false;
    }

    vec3 hist_mean = vec3(0.0f);
    vec3 hist_m2 = vec3(0.0f);
    float hist_len = 0.0f;

    if (valid_history) {

        hist_mean = edge_aware_bilinear_sample_vec3(
            prev_mean, prev_depth, prev_normal, prev_object_id, width, height,
            prev_u, prev_v, d, n, obj_id, edge_depth_thresh, edge_normal_thresh,
            use_obj);
        hist_m2 = edge_aware_bilinear_sample_vec3(
            prev_m2, prev_depth, prev_normal, prev_object_id, width, height,
            prev_u, prev_v, d, n, obj_id, edge_depth_thresh, edge_normal_thresh,
            use_obj);
        hist_len = edge_aware_bilinear_sample_float(
            prev_history_length, prev_depth, prev_normal, prev_object_id, width,
            height, prev_u, prev_v, d, n, obj_id, edge_depth_thresh,
            edge_normal_thresh, use_obj);

        float hist_d = edge_aware_bilinear_sample_float(
            prev_depth, prev_depth, prev_normal, prev_object_id, width, height,
            prev_u, prev_v, d, n, obj_id, edge_depth_thresh, edge_normal_thresh,
            use_obj);

        if (use_obj && prev_object_id != nullptr) {
            int hist_obj = nearest_sample_int(prev_object_id, width, height,
                                              prev_u, prev_v);
            if (hist_obj != obj_id)
                valid_history = false;
        }

        float depth_abs_diff = fabsf(d - hist_d);
        if (depth_abs_diff > depth_reject_abs ||
            depth_abs_diff > depth_reject_rel * fmaxf(1e-6f, d)) {
            valid_history = false;
        }

        int nearest_x = clamp_int((int)floorf(prev_u), 0, width - 1);
        int nearest_y = clamp_int((int)floorf(prev_v), 0, height - 1);
        vec3 hist_n = prev_normal[nearest_y * width + nearest_x];
        if (dot(n, hist_n) < normal_reject_thresh) {
            valid_history = false;
        }
    }

    if (valid_history) {
        hist_mean = min_vec3(max_vec3(hist_mean, soft_min), soft_max);
    }

    float alpha = 1.0f;
    float new_history_len = 1.0f;

    if (valid_history) {
        vec3 var = max_vec3(hist_m2 - (hist_mean * hist_mean), vec3(0.0f));
        float std_approx =
            (sqrtf(var.x) + sqrtf(var.y) + sqrtf(var.z)) * (1.0f / 3.0f);
        float variance_alpha = std_approx / (std_approx + tau);
        float history_alpha = 1.0f / (hist_len + 1.0f);

        alpha = clampf(fmaxf(variance_alpha, history_alpha), min_alpha, 1.0f);
        new_history_len = fminf(hist_len + 1.0f, max_history_length);
    }

    out_mean[idx] = hist_mean * (1.0f - alpha) + cur_c * alpha;
    out_m2[idx] = hist_m2 * (1.0f - alpha) + (cur_c * cur_c) * alpha;
    out_history_length[idx] = new_history_len;
}

__global__ void estimate_variance_kernel(
    float *__restrict__ out_variance, const vec3 *__restrict__ color,
    const vec3 *__restrict__ m2, const float *__restrict__ history_length,
    const float *__restrict__ depth, const vec3 *__restrict__ normal,
    const int *__restrict__ object_id, float sky_threshold, int width,
    int height, bool use_obj_id) {
    bool use_obj = use_obj_id && (object_id != nullptr);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;

    float d = depth[idx];
    vec3 n = normal[idx];
    int obj = (use_obj && object_id) ? object_id[idx] : -1;

    if (is_sky(d, n, sky_threshold)) {
        out_variance[idx] = 0.0f;
        return;
    }

    vec3 c = color[idx];
    vec3 c_m2 = m2[idx];
    float hist_len = history_length[idx];

    vec3 var = max_vec3(c_m2 - (c * c), vec3(0.0f));
    float reliability = fminf(hist_len * 0.25f, 1.0f);
    float variance_boost = 1.0f + (1.0f - reliability) * 3.0f;

    vec3 spatial_mean = vec3(0.0f);
    vec3 spatial_m2 = vec3(0.0f);
    int count = 0;

#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = clamp_int(x + dx, 0, width - 1);
            int ny = clamp_int(y + dy, 0, height - 1);
            int nidx = ny * width + nx;

            if (use_obj && object_id[nidx] != obj)
                continue;

            vec3 nc = color[nidx];
            spatial_mean += nc;
            spatial_m2 += nc * nc;
            count++;
        }
    }

    const float inv_count = 1.0f / (float)count;
    spatial_mean *= inv_count;
    spatial_m2 *= inv_count;
    vec3 spatial_var =
        max_vec3(spatial_m2 - spatial_mean * spatial_mean, vec3(0.0f));

    vec3 combined_var = max_vec3(var * variance_boost, spatial_var);
    out_variance[idx] = 0.2126f * combined_var.x + 0.7152f * combined_var.y +
                        0.0722f * combined_var.z;
}

__global__ void atrous_filter_kernel(
    vec3 *__restrict__ output, float *__restrict__ out_variance,
    const vec3 *__restrict__ input, const float *__restrict__ in_variance,
    const vec3 *__restrict__ normal, const float *__restrict__ depth,
    const int *__restrict__ object_id, int step_size, float sigma_luminance,
    float sigma_normal, float sigma_depth, float sky_threshold,
    float edge_depth_threshold, float edge_normal_threshold, bool use_object_id,
    int width, int height) {
    bool use_obj = use_object_id && (object_id != nullptr);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;

    vec3 center_c = input[idx];
    vec3 center_n = normal[idx];
    float center_d = depth[idx];
    int center_obj = (use_obj && object_id) ? object_id[idx] : -1;
    float center_var = in_variance[idx];
    float center_lum = luminance(center_c);

    if (is_sky(center_d, center_n, sky_threshold)) {
        output[idx] = center_c;
        out_variance[idx] = center_var;
        return;
    }

    float var_scale = sqrtf(fmaxf(center_var, 1e-6f));
    float adaptive_sigma_lum = sigma_luminance * (1.0f + var_scale * 2.0f);
    float inv_sigma_lum_sq =
        1.0f / (2.0f * adaptive_sigma_lum * adaptive_sigma_lum + 1e-6f);

    vec3 sum = vec3(0.0f);
    float sum_var = 0.0f;
    float total_w = 0.0f;

#pragma unroll
    for (int dy = -2; dy <= 2; ++dy) {
#pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            int k = (dy + 2) * 5 + (dx + 2);
            int nx = x + dx * step_size;
            int ny = y + dy * step_size;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;

            int n_idx = ny * width + nx;

            if (use_obj) {
                int n_obj = object_id[n_idx];
                if (center_obj != n_obj && center_obj >= 0 && n_obj >= 0) {
                    continue;
                }
            }

            float n_d = depth[n_idx];

            float max_d = fmaxf(center_d, n_d);
            float depth_diff = fabsf(center_d - n_d);
            if (max_d > 1e-6f && depth_diff / max_d > edge_depth_threshold) {
                continue;
            }

            vec3 n_n = normal[n_idx];
            float n_dot = dot(center_n, n_n);

            if (n_dot < edge_normal_threshold) {
                continue;
            }

            if (is_sky(n_d, n_n, sky_threshold))
                continue;

            vec3 n_c = input[n_idx];
            float n_var = in_variance[n_idx];
            float n_lum = luminance(n_c);

            float lum_diff = fabsf(center_lum - n_lum);
            float w_l = __expf(-lum_diff * lum_diff * inv_sigma_lum_sq);

            float weight = atrous_kernel[k] * w_l;

            sum += n_c * weight;
            sum_var += n_var * weight;
            total_w += weight;
        }
    }

    if (total_w < 1e-6f) {
        output[idx] = center_c;
        out_variance[idx] = center_var;
    } else {
        float inv_w = 1.0f / total_w;
        output[idx] = sum * inv_w;
        out_variance[idx] = sum_var * inv_w;
    }
}

__global__ void init_moments_kernel(const vec3 *__restrict__ color,
                                    vec3 *__restrict__ out_m2,
                                    float *__restrict__ out_history_length,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;
    vec3 c = color[idx];
    out_m2[idx] = c * c;
    out_history_length[idx] = 1.0f;
}

__global__ void combine_split_channels_kernel(vec3 *__restrict__ output,
                                              const vec3 *__restrict__ diffuse,
                                              const vec3 *__restrict__ specular,
                                              const vec3 *__restrict__ emission,
                                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;
    vec3 combined = diffuse[idx] + specular[idx];
    if (emission != nullptr)
        combined += emission[idx];
    output[idx] = combined;
}

class Denoiser {
  public:
    DenoiserSettings settings;
    bool initialized = false;
    bool first_frame = true;

    vec3 *d_ping = nullptr;
    vec3 *d_pong = nullptr;
    float *d_variance_ping = nullptr;
    float *d_variance_pong = nullptr;

    vec3 *d_diffuse_history_mean = nullptr;
    vec3 *d_diffuse_history_m2 = nullptr;
    float *d_diffuse_history_length = nullptr;

    vec3 *d_specular_history_mean = nullptr;
    vec3 *d_specular_history_m2 = nullptr;
    float *d_specular_history_length = nullptr;

    vec3 *d_history_normal = nullptr;
    float *d_history_depth = nullptr;
    int *d_history_objectId = nullptr;

    vec3 *d_denoised_diffuse = nullptr;
    vec3 *d_denoised_specular = nullptr;

    Denoiser() = default;
    Denoiser(const DenoiserSettings &s) { initialize(s); }
    ~Denoiser() { destroy(); }

    void initialize(const DenoiserSettings &s) {
        settings = s;
        int width = s.width;
        int height = s.height;
        size_t pixels = (size_t)width * height;

        CUDA_CHECK(cudaMalloc(&d_ping, pixels * sizeof(vec3)));
        CUDA_CHECK(cudaMalloc(&d_pong, pixels * sizeof(vec3)));
        CUDA_CHECK(cudaMalloc(&d_variance_ping, pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_variance_pong, pixels * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_diffuse_history_mean, pixels * sizeof(vec3)));
        CUDA_CHECK(cudaMalloc(&d_diffuse_history_m2, pixels * sizeof(vec3)));
        CUDA_CHECK(
            cudaMalloc(&d_diffuse_history_length, pixels * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_specular_history_mean, pixels * sizeof(vec3)));
        CUDA_CHECK(cudaMalloc(&d_specular_history_m2, pixels * sizeof(vec3)));
        CUDA_CHECK(
            cudaMalloc(&d_specular_history_length, pixels * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_history_normal, pixels * sizeof(vec3)));
        CUDA_CHECK(cudaMalloc(&d_history_depth, pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_history_objectId, pixels * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&d_denoised_diffuse, pixels * sizeof(vec3)));
        CUDA_CHECK(cudaMalloc(&d_denoised_specular, pixels * sizeof(vec3)));

        CUDA_CHECK(
            cudaMemset(d_diffuse_history_length, 0, pixels * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(d_specular_history_length, 0, pixels * sizeof(float)));

        initialized = true;
        first_frame = true;
    }

    void release() {
        if (d_ping)
            cudaFree(d_ping);
        if (d_pong)
            cudaFree(d_pong);
        if (d_variance_ping)
            cudaFree(d_variance_ping);
        if (d_variance_pong)
            cudaFree(d_variance_pong);
        if (d_diffuse_history_mean)
            cudaFree(d_diffuse_history_mean);
        if (d_diffuse_history_m2)
            cudaFree(d_diffuse_history_m2);
        if (d_diffuse_history_length)
            cudaFree(d_diffuse_history_length);
        if (d_specular_history_mean)
            cudaFree(d_specular_history_mean);
        if (d_specular_history_m2)
            cudaFree(d_specular_history_m2);
        if (d_specular_history_length)
            cudaFree(d_specular_history_length);
        if (d_history_normal)
            cudaFree(d_history_normal);
        if (d_history_depth)
            cudaFree(d_history_depth);
        if (d_history_objectId)
            cudaFree(d_history_objectId);
        if (d_denoised_diffuse)
            cudaFree(d_denoised_diffuse);
        if (d_denoised_specular)
            cudaFree(d_denoised_specular);
        initialized = false;
    }

    void destroy() { release(); }

    void denoiseChannel(vec3 *source_color, vec3 *history_mean,
                        vec3 *history_m2, float *history_length,
                        vec3 *output_color, const DenoiserInputs &inputs,
                        float tau, float min_alpha, float max_history,
                        float sigma_lum, float sigma_norm, float sigma_depth,
                        int atrous_iters, float clamp_scale,
                        float firefly_threshold, dim3 grid, dim3 block) {
        int width = settings.width;
        int height = settings.height;
        bool use_obj_id =
            settings.use_object_ids && (inputs.objectId != nullptr);

        if (settings.enable_firefly_suppression) {
            firefly_suppression_kernel<<<grid, block>>>(
                d_ping, source_color, inputs.depth, inputs.normal,
                firefly_threshold, settings.sky_depth_threshold, width, height);
            CUDA_CHECK(cudaGetLastError());
        } else {
            CUDA_CHECK(cudaMemcpy(d_ping, source_color,
                                  (size_t)width * height * sizeof(vec3),
                                  cudaMemcpyDeviceToDevice));
        }

        if (first_frame) {
            CUDA_CHECK(cudaMemcpy(history_mean, d_ping,
                                  (size_t)width * height * sizeof(vec3),
                                  cudaMemcpyDeviceToDevice));
            init_moments_kernel<<<grid, block>>>(d_ping, history_m2,
                                                 history_length, width, height);
        }

        temporal_accumulation_kernel<<<grid, block>>>(
            d_ping, d_pong, d_variance_ping, d_ping, history_mean, history_m2,
            history_length, inputs.motion, inputs.depth, d_history_depth,
            inputs.normal, d_history_normal, inputs.objectId,
            d_history_objectId, tau, min_alpha, max_history,
            settings.depth_reject_absolute, settings.depth_reject_relative,
            settings.normal_reject_threshold, clamp_scale,
            settings.sky_depth_threshold, settings.edge_depth_threshold,
            settings.edge_normal_threshold, use_obj_id, width, height);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(history_mean, d_ping,
                              (size_t)width * height * sizeof(vec3),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(history_m2, d_pong,
                              (size_t)width * height * sizeof(vec3),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(history_length, d_variance_ping,
                              (size_t)width * height * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        estimate_variance_kernel<<<grid, block>>>(
            d_variance_ping, d_ping, history_m2, history_length, inputs.depth,
            inputs.normal, inputs.objectId, settings.sky_depth_threshold, width,
            height, settings.use_object_ids);
        CUDA_CHECK(cudaGetLastError());

        vec3 *spatial_input = d_ping;
        vec3 *spatial_output = d_pong;
        float *var_input = d_variance_ping;
        float *var_output = d_variance_pong;

        int step_sizes[] = {1, 2, 4, 8, 16};
        int num_iterations = mini(atrous_iters, 5);

        for (int i = 0; i < num_iterations; ++i) {
            atrous_filter_kernel<<<grid, block>>>(
                spatial_output, var_output, spatial_input, var_input,
                inputs.normal, inputs.depth, inputs.objectId, step_sizes[i],
                sigma_lum, sigma_norm, sigma_depth,
                settings.sky_depth_threshold, settings.edge_depth_threshold,
                settings.edge_normal_threshold, settings.use_object_ids, width,
                height);
            CUDA_CHECK(cudaGetLastError());

            vec3 *tmp = spatial_input;
            spatial_input = spatial_output;
            spatial_output = tmp;
            float *tmp_v = var_input;
            var_input = var_output;
            var_output = tmp_v;
        }

        CUDA_CHECK(cudaMemcpy(output_color, spatial_input,
                              (size_t)width * height * sizeof(vec3),
                              cudaMemcpyDeviceToDevice));
    }

    void denoise(const DenoiserInputs &inputs,
                 const DenoiserCommonSettings &common, vec3 *output_buffer) {
        if (!initialized)
            return;

        if (inputs.normal == nullptr || inputs.depth == nullptr ||
            inputs.motion == nullptr) {
            std::cerr << "Denoiser: Missing required inputs!" << std::endl;
            return;
        }

        int width = settings.width;
        int height = settings.height;
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        if (first_frame) {
            CUDA_CHECK(cudaMemcpy(d_history_normal, inputs.normal,
                                  (size_t)width * height * sizeof(vec3),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_history_depth, inputs.depth,
                                  (size_t)width * height * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            if (inputs.objectId != nullptr) {
                CUDA_CHECK(cudaMemcpy(d_history_objectId, inputs.objectId,
                                      (size_t)width * height * sizeof(int),
                                      cudaMemcpyDeviceToDevice));
            }
        }

        bool have_split =
            (inputs.diffuseColor != nullptr && inputs.specularColor != nullptr);

        if (settings.enable_split_denoising && have_split) {
            denoiseChannel(
                inputs.diffuseColor, d_diffuse_history_mean,
                d_diffuse_history_m2, d_diffuse_history_length,
                d_denoised_diffuse, inputs, settings.diffuse_tau,
                settings.diffuse_min_alpha, settings.diffuse_max_history,
                settings.diffuse_sigma_luminance, settings.diffuse_sigma_normal,
                settings.diffuse_sigma_depth,
                settings.diffuse_atrous_iterations,
                settings.diffuse_clamp_scale,
                settings.diffuse_firefly_threshold, grid, block);

            denoiseChannel(
                inputs.specularColor, d_specular_history_mean,
                d_specular_history_m2, d_specular_history_length,
                d_denoised_specular, inputs, settings.specular_tau,
                settings.specular_min_alpha, settings.specular_max_history,
                settings.specular_sigma_luminance,
                settings.specular_sigma_normal, settings.specular_sigma_depth,
                settings.specular_atrous_iterations,
                settings.specular_clamp_scale,
                settings.specular_firefly_threshold, grid, block);

            combine_split_channels_kernel<<<grid, block>>>(
                output_buffer, d_denoised_diffuse, d_denoised_specular,
                inputs.emissionColor, width, height);
            CUDA_CHECK(cudaGetLastError());

        } else {
            if (inputs.noisyColor == nullptr) {
                std::cerr << "Denoiser: No input color provided!" << std::endl;
                return;
            }

            denoiseChannel(
                inputs.noisyColor, d_diffuse_history_mean, d_diffuse_history_m2,
                d_diffuse_history_length, output_buffer, inputs,
                settings.diffuse_tau, settings.diffuse_min_alpha,
                settings.diffuse_max_history, settings.diffuse_sigma_luminance,
                settings.diffuse_sigma_normal, settings.diffuse_sigma_depth,
                settings.diffuse_atrous_iterations,
                settings.diffuse_clamp_scale,
                settings.diffuse_firefly_threshold, grid, block);
        }

        CUDA_CHECK(cudaMemcpy(d_history_normal, inputs.normal,
                              (size_t)width * height * sizeof(vec3),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_history_depth, inputs.depth,
                              (size_t)width * height * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        if (inputs.objectId != nullptr) {
            CUDA_CHECK(cudaMemcpy(d_history_objectId, inputs.objectId,
                                  (size_t)width * height * sizeof(int),
                                  cudaMemcpyDeviceToDevice));
        }

        first_frame = false;
    }

    void updateSettings(const DenoiserSettings &new_settings) {
        settings = new_settings;
    }
};

#endif
