// scene_kernels.cuh
#ifndef SCENE_KERNELS_CUH
#define SCENE_KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// TAA support for better anti-aliasing of thin geometry (ropes, wires, etc.)
#include "pathtracer/rendering/taa.cuh"

#ifndef PT_BLOCK_X
#define PT_BLOCK_X 8
#endif
#ifndef PT_BLOCK_Y
#define PT_BLOCK_Y 8
#endif

#ifndef SIMPLE_BLOCK_X
#define SIMPLE_BLOCK_X 16
#endif
#ifndef SIMPLE_BLOCK_Y
#define SIMPLE_BLOCK_Y 16
#endif

// curand state initialization
static __global__ void init_curand_kernel(curandState *states, int W, int H,
                                          unsigned long long seed) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const size_t idx = (static_cast<size_t>(y) * W + x);
    curand_init(seed, idx, 0, &states[idx]);
}

// Single ray trace kernel (for debugging)
static __global__ void trace_single_ray_kernel(vec3 origin, vec3 direction,
                                               DeviceMesh *meshes, int nMeshes,
                                               DeviceBVHNode *tlasNodes,
                                               int *tlasMeshIndices,
                                               HitInfo *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    Ray ray(origin, direction);
    HitInfo hit = traceRay(ray, meshes, tlasNodes, tlasMeshIndices);
    *result = hit;
}

// ═══════════════════════════════════════════════════════════════════════════
// Wireframe rendering kernel
// ═══════════════════════════════════════════════════════════════════════════
static __global__ void render_kernel_wireframe(
    unsigned char *out, int W, int H, Camera cam, DeviceMesh *meshes,
    int nMeshes, Light *lights, int nLights, vec3 skyColorTop,
    vec3 skyColorBottom, bool useSky, bool wireframeMode,
    float wireframeThickness, DeviceMaterials *materials_ptr,
    DeviceBVHNode *tlasNodes, int *tlasMeshIndices,
    cudaTextureObject_t envMap) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const float u = (x + 0.5f) / W;
    const float v = 1.0f - (y + 0.5f) / H;

    Ray ray = cam.get_ray(u, v);

    vec3 color(0.0f);

    const DeviceMaterials *materials_nullable = materials_ptr;

    if (wireframeMode) {
        bool hitEdge = false;
        HitInfo hit = traceRay(ray, meshes, tlasNodes, tlasMeshIndices);

        if (hit.hit) {
            float w_bary = 1.0f - hit.u - hit.v;

            if (hit.u < wireframeThickness || hit.v < wireframeThickness ||
                w_bary < wireframeThickness) {
                hitEdge = true;
                if (materials_nullable) {
                    vec3 emission =
                        materials_nullable->emission[hit.mesh_index];
                    color = emission.x > 0 ? emission : vec3(1.0f);
                } else {
                    color = vec3(1.0f);
                }
            }
        }

        if (!hitEdge) {
            color = sampleSky(ray, skyColorTop, skyColorBottom, useSky, envMap);
        }
    } else {
        HitInfo hit = traceRay(ray, meshes, tlasNodes, tlasMeshIndices);
        if (!hit.hit) {
            color = sampleSky(ray, skyColorTop, skyColorBottom, useSky, envMap);
        } else {
            color = vec3(0.0f);
        }
    }

    // Tone mapping and gamma
    color = color / (color + vec3(1.0f));
    color = vec3(powf(color.x, 1.0f / 2.2f), powf(color.y, 1.0f / 2.2f),
                 powf(color.z, 1.0f / 2.2f));

    const vec3 rgb = clamp(color, 0.f, 1.f) * 255.99f;
    const int y_out = H - 1 - y;
    const size_t idx = (static_cast<size_t>(y_out) * W + x) * 3;
    out[idx + 0] = static_cast<unsigned char>(rgb.x);
    out[idx + 1] = static_cast<unsigned char>(rgb.y);
    out[idx + 2] = static_cast<unsigned char>(rgb.z);
}

// Main path tracing kernel
// Uses 1 SPP by default (rely on temporal denoiser)
static __global__ void
path_trace_kernel(vec3 *accum_buffer, vec3 *normal_buffer, float *depth_buffer,
                  int *objectId_buffer, int width, int height, Camera cam,
                  DeviceMesh *meshes, int nMeshes, Light *lights, int nLights,
                  vec3 skyColorTop, vec3 skyColorBottom, bool useSky, int spp,
                  int max_depth, DeviceBVHNode *tlasNodes, int *tlasMeshIndices,
                  DeviceMaterials *materials, cudaTextureObject_t envMap,
                  curandState *rand_states, int frame_count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const size_t idx = (static_cast<size_t>(y) * width + x);
    curandState local_rand_state = rand_states[idx];

    if (!materials)
        return;

    vec3 avg_color(0.0f);
    vec3 first_normal(0.0f);
    float first_depth = 1e30f;
    int first_objectId = -1;

    for (int s = 0; s < spp; ++s) {
        // TAA: Use Halton sequence for temporally stable sub-pixel jitter
        // This provides better coverage for thin geometry like ropes/wires
        // The jitter is the same for all pixels in a frame (temporal coherence)
        // but varies between frames for progressive refinement
        float2 taa_jitter = getTAAJitter(frame_count + s);

        // Combine TAA jitter with per-pixel blue noise for variance reduction
        // Blue noise decorrelates samples spatially while Halton handles
        // temporal
        float2 bn = next_blue_noise(x, y, frame_count + s);

        // Use TAA jitter as primary offset, with small blue noise perturbation
        // The 0.25 factor keeps blue noise subtle so TAA pattern dominates
        float jitter_x = taa_jitter.x + (bn.x - 0.5f) * 0.25f;
        float jitter_y = taa_jitter.y + (bn.y - 0.5f) * 0.25f;

        const float u = (x + 0.5f + jitter_x) / width;
        const float v = 1.0f - (y + 0.5f + jitter_y) / height;

        Ray ray = cam.get_ray(u, v, &local_rand_state);

        vec3 sample_normal;
        float sample_depth;
        int sample_objectId = -1;

        vec3 sample_color =
            tracePath(ray, meshes, nMeshes, tlasNodes, tlasMeshIndices,
                      *materials, lights, nLights, skyColorTop, skyColorBottom,
                      useSky, &local_rand_state, max_depth, envMap,
                      sample_normal, sample_depth, sample_objectId);

        avg_color += sample_color;

        if (s == 0) {
            first_normal = sample_normal;
            first_depth = sample_depth;
            first_objectId = sample_objectId;
        }
    }

    rand_states[idx] = local_rand_state;

    accum_buffer[idx] = avg_color / (float)spp;
    normal_buffer[idx] = first_normal;
    depth_buffer[idx] = first_depth;
    objectId_buffer[idx] = first_objectId;
}

// Split path tracing kernel

static __global__ __launch_bounds__(256, 4) void path_trace_split_kernel(
    vec3 *out_diffuse, vec3 *out_specular, vec3 *out_emission,
    vec3 *out_world_normal, float *out_depth, float *out_roughness,
    float *out_transmission, int W, int H, Camera cam, DeviceMesh *meshes,
    int nMeshes, Light *lights, int nLights, vec3 skyColorTop,
    vec3 skyColorBottom, bool useSky, int samples_per_pixel, int max_depth,
    DeviceBVHNode *tlasNodes, int *tlasMeshIndices,
    DeviceMaterials *materials_ptr, cudaTextureObject_t envMap,
    curandState *global_rand_states) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const size_t idx = (static_cast<size_t>(y) * W + x);

    curandState local_rand_state = global_rand_states[idx];

    if (!materials_ptr)
        return;
    const DeviceMaterials &materials = *materials_ptr;

    vec3 avg_diffuse(0.0f);
    vec3 avg_specular(0.0f);
    vec3 avg_emission(0.0f);
    vec3 first_normal(0.0f);
    float first_depth = 1e30f;
    float first_roughness = 1.0f;
    float first_transmission = 0.0f;

    for (int s = 0; s < samples_per_pixel; ++s) {
        // TAA: Use Halton sequence for temporally stable sub-pixel jitter
        // Combined with small random perturbation for variance reduction
        float2 taa_jitter = getTAAJitter(s); // Use sample index for multi-SPP
        float rand_x = curand_uniform(&local_rand_state);
        float rand_y = curand_uniform(&local_rand_state);

        // Primary jitter from Halton, small random perturbation for
        // decorrelation
        float jitter_x = taa_jitter.x + (rand_x - 0.5f) * 0.25f;
        float jitter_y = taa_jitter.y + (rand_y - 0.5f) * 0.25f;

        const float u = (x + 0.5f + jitter_x) / W;
        const float v = 1.0f - (y + 0.5f + jitter_y) / H;

        Ray ray = cam.get_ray(u, v, &local_rand_state);

        vec3 sample_normal;
        float sample_depth;
        float sample_roughness;
        float sample_transmission;

        SplitPathOutput result = tracePathSplit(
            ray, meshes, nMeshes, tlasNodes, tlasMeshIndices, materials, lights,
            nLights, skyColorTop, skyColorBottom, useSky, &local_rand_state,
            max_depth, envMap, sample_normal, sample_depth, sample_roughness,
            sample_transmission);

        avg_diffuse += result.diffuse;
        avg_specular += result.specular;
        avg_emission += result.emission;

        if (s == 0) {
            first_normal = sample_normal;
            first_depth = sample_depth;
            first_roughness = sample_roughness;
            first_transmission = sample_transmission;
        }
    }

    global_rand_states[idx] = local_rand_state;

    float inv_spp = 1.0f / (float)samples_per_pixel;
    out_diffuse[idx] = avg_diffuse * inv_spp;
    out_specular[idx] = avg_specular * inv_spp;
    out_emission[idx] = avg_emission * inv_spp;
    out_world_normal[idx] = first_normal;
    out_depth[idx] = first_depth;
    out_roughness[idx] = first_roughness;
    out_transmission[idx] = first_transmission;
}

// Bloom Kernels

static __global__ __launch_bounds__(256, 4) void bloom_bright_pass_kernel(
    vec3 *__restrict__ out_bright, const vec3 *__restrict__ in_hdr, int W,
    int H, float threshold, float knee) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const size_t idx = (size_t)y * W + x;
    vec3 color = in_hdr[idx];

    float brightness = fmaxf(color.x, fmaxf(color.y, color.z));
    float soft_t = brightness - threshold + knee;
    float bloom = clamp(soft_t / (2.0f * knee) + 0.5f, 0.0f, 1.0f);

    out_bright[idx] = color * bloom;
}

static __global__
__launch_bounds__(256, 4) void bloom_blur_h_kernel(vec3 *out, const vec3 *in,
                                                   int W, int H, float radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const float weights[3] = {0.227027f, 0.316216f, 0.070270f};
    const size_t idx = (size_t)y * W + x;

    vec3 color = in[idx] * weights[0];

#pragma unroll
    for (int i = 1; i <= 2; ++i) {
        int x_l = max(x - i, 0);
        int x_r = min(x + i, W - 1);
        color += in[(size_t)y * W + x_l] * weights[i];
        color += in[(size_t)y * W + x_r] * weights[i];
    }
    out[idx] = color;
}

static __global__ __launch_bounds__(256, 4) void bloom_downsample_v_kernel(
    vec3 *out, const vec3 *in, int in_W, int in_H, float radius) {
    const int out_W = in_W / 2;
    const int out_H = in_H / 2;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_W || y >= out_H)
        return;

    const int in_x = x * 2;
    const int in_y = y * 2;

    const float weights[3] = {0.227027f, 0.316216f, 0.070270f};

    vec3 color = vec3(0.0f);

#pragma unroll
    for (int j = -2; j <= 2; ++j) {
        int in_y_tap = min(max(in_y + j, 0), in_H - 1);
        float w = weights[abs(j)];
        color += in[(size_t)in_y_tap * in_W + in_x] * w;
    }

    out[(size_t)y * out_W + x] = color;
}

static __global__ __launch_bounds__(256, 4) void bloom_upsample_add_kernel(
    vec3 *out_color, const vec3 *in_bloom, int W_low, int H_low) {
    const int W_high = W_low * 2;
    const int H_high = H_low * 2;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W_high || y >= H_high)
        return;

    float u = (x + 0.5f) / (float)W_high;
    float v = (y + 0.5f) / (float)H_high;

    float u_low = u * W_low - 0.5f;
    float v_low = v * H_low - 0.5f;

    int x0 = (int)floorf(u_low);
    int y0 = (int)floorf(v_low);
    float u_frac = u_low - x0;
    float v_frac = v_low - y0;

    int x1 = min(x0 + 1, W_low - 1);
    int y1 = min(y0 + 1, H_low - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    vec3 s00 = in_bloom[(size_t)y0 * W_low + x0];
    vec3 s10 = in_bloom[(size_t)y0 * W_low + x1];
    vec3 s01 = in_bloom[(size_t)y1 * W_low + x0];
    vec3 s11 = in_bloom[(size_t)y1 * W_low + x1];

    vec3 bloom = lerp(lerp(s00, s10, u_frac), lerp(s01, s11, u_frac), v_frac);

    const size_t idx_high = (size_t)y * W_high + x;
    out_color[idx_high] = out_color[idx_high] + bloom;
}

// Helper function to get optimal grid/block configuration
inline void getOptimalLaunchConfig(int width, int height, dim3 &grid,
                                   dim3 &block, bool isPathTracing = true) {
    if (isPathTracing) {
        // Use smaller blocks for path tracing (high register usage)
        block = dim3(PT_BLOCK_X, PT_BLOCK_Y);
    } else {
        // Use larger blocks for simpler kernels
        block = dim3(SIMPLE_BLOCK_X, SIMPLE_BLOCK_Y);
    }

    grid =
        dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

// Bilinear upscale kernel for resolution scaling
static __global__ __launch_bounds__(256, 4) void upscale_bilinear_kernel(
    vec3 *__restrict__ out, const vec3 *__restrict__ in, int out_w, int out_h,
    int in_w, int in_h) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h)
        return;

    // Map output pixel to input coordinates
    float u = (x + 0.5f) * (float)in_w / (float)out_w - 0.5f;
    float v = (y + 0.5f) * (float)in_h / (float)out_h - 0.5f;

    // Clamp to valid range
    u = fmaxf(0.0f, fminf((float)(in_w - 1), u));
    v = fmaxf(0.0f, fminf((float)(in_h - 1), v));

    int x0 = (int)floorf(u);
    int y0 = (int)floorf(v);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);

    float fx = u - x0;
    float fy = v - y0;

    // Bilinear interpolation
    vec3 s00 = in[(size_t)y0 * in_w + x0];
    vec3 s10 = in[(size_t)y0 * in_w + x1];
    vec3 s01 = in[(size_t)y1 * in_w + x0];
    vec3 s11 = in[(size_t)y1 * in_w + x1];

    vec3 top = s00 * (1.0f - fx) + s10 * fx;
    vec3 bot = s01 * (1.0f - fx) + s11 * fx;
    vec3 result = top * (1.0f - fy) + bot * fy;

    out[(size_t)y * out_w + x] = result;
}

#endif // SCENE_KERNELS_CUH
