/*
this file provides a cuda kernel that computes screen space motion vectors
it is meant to support temporal reprojection based denoisers such as nrd style
pipelines it reconstructs a world space position from a camera ray and a linear
depth value it projects that position with the previous frame view projection
matrix it outputs motion vectors in uv space as the difference between current
uv and previous uv it treats very large depth values as sky pixels and writes
zero motion for those pixels it expects depth to be a linear ray distance
matching the camera ray parameterization it is intended to be included by cuda
code that launches the kernel with a two dimensional grid the sky depth
threshold can be overridden at compile time by defining denoiser sky depth
threshold
*/

#ifndef DENOISER_KERNELS_CUH
#define DENOISER_KERNELS_CUH

#include "common/mat4.cuh"
#include "common/vec3.cuh"
#include "pathtracer/scene/camera.cuh"
#include <cuda_runtime.h>

#ifndef DENOISER_SKY_DEPTH_THRESHOLD
#define DENOISER_SKY_DEPTH_THRESHOLD 1e29f
#endif

#ifndef __CUDACC__
struct float2 {
    float x, y;
};
#endif

__global__ void motion_vector_kernel(float2 *out_motion_vectors,
                                     const float *in_depth, int W, int H,
                                     Camera cam, mat4 prevViewProj) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const size_t idx = (static_cast<size_t>(y) * W + x);
    float depth = in_depth[idx];

    if (depth >= DENOISER_SKY_DEPTH_THRESHOLD) {
        out_motion_vectors[idx] = {0.0f, 0.0f};
        return;
    }

    float u = (x + 0.5f) / W;
    float v = (y + 0.5f) / H;

    Ray ray = cam.get_ray(u, 1.0f - v);
    vec3 world_pos = ray.origin() + ray.direction() * depth;
    vec4 world_pos_h = vec4(world_pos, 1.0f);

    vec4 prev_clip_h = prevViewProj * world_pos_h;

    vec3 prev_ndc =
        vec3(prev_clip_h.x, prev_clip_h.y, prev_clip_h.z) / prev_clip_h.w;

    float prev_u = (prev_ndc.x + 1.0f) * 0.5f;
    float prev_v = (1.0f - prev_ndc.y) * 0.5f;

    float mv_x = u - prev_u;
    float mv_y = v - prev_v;

    out_motion_vectors[idx] = {mv_x, mv_y};
}

#endif
