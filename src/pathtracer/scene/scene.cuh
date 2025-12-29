// scene.cuh
#ifndef PT_SCENE_CUH
#define PT_SCENE_CUH

// render config constants shared across host code and kernels
// these are intentionally exposed as a namespace so call sites can use
// pt_render_config colon colon names macros are optionally provided for older
// code paths that expect PT_BLOCK_X style names

namespace pt_render_config {
static constexpr int pt_block_x = 8;
static constexpr int pt_block_y = 8;
static constexpr int simple_block_x = 16;
static constexpr int simple_block_y = 16;
} // namespace pt_render_config

#ifndef PT_BLOCK_X
#define PT_BLOCK_X pt_render_config::pt_block_x
#endif
#ifndef PT_BLOCK_Y
#define PT_BLOCK_Y pt_render_config::pt_block_y
#endif
#ifndef SIMPLE_BLOCK_X
#define SIMPLE_BLOCK_X pt_render_config::simple_block_x
#endif
#ifndef SIMPLE_BLOCK_Y
#define SIMPLE_BLOCK_Y pt_render_config::simple_block_y
#endif

#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <curand_kernel.h>

#include "common\mat4.cuh"
#include "common\matrix.cuh"
#include "pathtracer\math\intersection.cuh"
#include "pathtracer\math\pdf.cuh"
#include "pathtracer\rendering\denoiser.cuh"
#include "pathtracer\rendering\denoiser_kernels.cuh"
#include "pathtracer\scene\lights.cuh"

#include <stb_image.h>

#include "common\ray.cuh"
#include "common\triangle.cuh"
#include "common\vec3.cuh"
#include "common\visualization.cuh"
#include "pathtracer\math\intersection.cuh"
#include "pathtracer\math\mathutils.cuh"
#include "pathtracer\math\sampling.cuh"
#include "pathtracer\rendering\path_logic.cuh"
#include "pathtracer\rendering\pbr_utils.cuh"
#include "pathtracer\rendering\render_utils.cuh"
#include "pathtracer\scene\camera.cuh"
#include "pathtracer\scene\material_lib.cuh"
#include "pathtracer\scene\mesh.cuh"
#include "pathtracer\scene\scene_kernels.cuh"

__global__ void tonemap_kernel(unsigned char *out, const vec3 *in_buffer, int W,
                               int H, int total_samples);

__global__ void render_kernel_wireframe(
    unsigned char *out, int W, int H, Camera cam, DeviceMesh *meshes,
    int nMeshes, Light *lights, int nLights, vec3 skyColorTop,
    vec3 skyColorBottom, bool useSky, bool wireframeMode,
    float wireframeThickness, DeviceMaterials *materials,
    DeviceBVHNode *tlasNodes, int *tlasMeshIndices, cudaTextureObject_t envMap);

// Scene class
class Scene {
  private:
    // Image settings
    int width;
    int height;

    // Path Tracing Settings (old but i am scared it will mess something up if I
    // delete it)
    int samples_per_pixel_ = 16; // This will be IGNORED by render_to_device
    int max_depth_ = 8;          // This will be IGNORED by render_to_device
    int frame_count_ = 0;        // For progressive accumulation / random seed

    int bvhLeafTarget_ = 12;
    int bvhLeafTol_ = 5;

    // Scene components
    std::vector<std::unique_ptr<Mesh>> meshes;
    std::vector<Material> mesh_materials; // HOST-SIDE AoS list
    std::vector<Light> lights;
    Camera camera;

    size_t d_mesh_count_ = 0;  // Tracks allocated size
    size_t d_light_count_ = 0; // Tracks allocated size

    // GPU resources
    DeviceMesh *d_mesh_descriptors = nullptr;
    std::vector<DeviceMesh> h_mesh_descriptors_;

    Light *d_lights = nullptr;
    unsigned char *d_pixels = nullptr; // This is the 8-bit output buffer

    vec3 *d_accum_buffer = nullptr;
    curandState *d_rand_states = nullptr;

    // GBuffers for Denoiser
    vec3 *d_normal_buffer = nullptr;  // World-space normals
    float *d_depth_buffer = nullptr;  // Scene depth (hit_t)
    int *d_objectId_buffer = nullptr; // NEW: Object ID G-Buffer

    // Motion Vectors
    float2 *d_motion_vectors = nullptr; // Buffer for NRD motion vectors
    mat4 prev_view_proj;                // Previous frame's view-proj matrix

    //  TLAS device resources
    DeviceBVHNode *d_tlasNodes = nullptr;
    int *d_tlasMeshIndices = nullptr;

    // Host-side TLAS builder data
    std::vector<DeviceBVHNode> h_tlasNodes;
    std::vector<int> h_tlasMeshIndices; // Stores mesh_id for TLAS leaves

    // Material SoA device resources
    // This struct holds all the device pointers
    DeviceMaterials *d_materials_ptr_struct = nullptr;

    // Individual device pointers
    vec3 *d_mat_albedo = nullptr;
    vec3 *d_mat_specular = nullptr;
    float *d_mat_metallic = nullptr;
    float *d_mat_roughness = nullptr;
    vec3 *d_mat_emission = nullptr;
    float *d_mat_ior = nullptr;
    float *d_mat_transmission = nullptr;
    float *d_mat_transmissionRoughness = nullptr;
    float *d_mat_clearcoat = nullptr;
    float *d_mat_clearcoatRoughness = nullptr;
    vec3 *d_mat_subsurfaceColor = nullptr;
    float *d_mat_subsurfaceRadius = nullptr;
    float *d_mat_anisotropy = nullptr;
    float *d_mat_sheen = nullptr;
    vec3 *d_mat_sheenTint = nullptr;
    float *d_mat_iridescence = nullptr;
    float *d_mat_iridescenceThickness = nullptr;

    // HDRI / IBL resources
    cudaTextureObject_t d_env_texture = 0;
    float *d_env_data = nullptr;
    int env_width = 0;
    int env_height = 0;

    // Bloom Buffers
    static const int BLOOM_MIP_LEVELS = 6;
    vec3 *d_bright_pass_buffer = nullptr;
    vec3 *d_bloom_mip_chain[BLOOM_MIP_LEVELS]; // For downsampling
    vec3 *d_bloom_temp_buffer = nullptr;       // For separable blur

    // Background settings
    bool use_sky = true;
    vec3 sky_color_top = vec3(0.6f, 0.7f, 1.0f);
    vec3 sky_color_bottom = vec3(1.0f, 1.0f, 1.0f);

    bool wireframe_mode = false;
    bool show_frustum = false;
    bool show_rays = false;
    std::vector<std::pair<vec3, vec3>> debug_rays; // origin, direction pairs
    float ray_length = 5.0f;

    std::vector<size_t> visualization_mesh_indices;

    Denoiser *denoiser_ = nullptr;     // The wrapper for the NRD library
    vec3 *d_denoised_buffer = nullptr; // The output buffer for the clean image

    // Low-res buffers for rendering at scale < 1.0
    vec3 *d_scaled_accum = nullptr;    // Color
    vec3 *d_scaled_normal = nullptr;   // Normals
    float *d_scaled_depth = nullptr;   // Depth
    int *d_scaled_objectId = nullptr;  // Object IDs
    float2 *d_scaled_motion = nullptr; // Motion Vectors
    vec3 *d_scaled_denoised = nullptr; // Denoised output (low res)

    // PERFORMANCE SETTINGS - Configurable quality/speed tradeoffs
    struct PerformanceSettings {
        bool enableDenoiser = true;        // Disable for 30% speedup
        bool enableBloom = true;           // Disable for 10% speedup
        bool enableMotionVectors = true;   // Only needed with denoiser
        int maxBounceDepth = 4;            // Lower = faster (2-4 for realtime)
        int samplesPerPixel = 1;           // Always 1 for realtime
        float resolutionScale = 1.0f;      // 0.5 = half res (4x faster tracing)
        bool fastBVHUpdates = true;        // Skip unnecessary BVH rebuilds
        bool enableRussianRoulette = true; // Early ray termination
        int russianRouletteStartBounce = 1; // When to start RR
    };
    PerformanceSettings perfSettings;

    // Track if we're using scaled resolution
    int render_width = 0;
    int render_height = 0;
    vec3 *d_scaled_buffer = nullptr; // For resolution scaling

    struct DebugRayInfo {
        vec3 origin;
        vec3 direction;
        float length;
    };
    std::vector<DebugRayInfo> debug_rays_with_length;

    size_t frustum_mesh_index_ = size_t(-1);

    bool validateGPUResources() const {
        // Check critical buffers
        // Check object ID buffer
        if (!d_accum_buffer || !d_normal_buffer || !d_depth_buffer ||
            !d_objectId_buffer) {
            std::cerr << "ERROR: G-buffers not allocated!\n";
            return false;
        }

        if (!d_rand_states) {
            std::cerr << "ERROR: Random states not initialized!\n";
            return false;
        }

        if (meshes.empty()) {
            std::cerr << "ERROR: No meshes in scene!\n";
            return false;
        }

        if (!d_mesh_descriptors) {
            std::cerr << "ERROR: Mesh descriptors not allocated!\n";
            return false;
        }

        if (!d_materials_ptr_struct) {
            std::cerr << "ERROR: Materials not uploaded!\n";
            return false;
        }

        if (!d_tlasNodes) {
            std::cerr << "ERROR: TLAS not built!\n";
            return false;
        }

        return true;
    }

    // Flag to check if GPU resources are initialized
    bool gpu_resources_initialized = false;

    // Helper to free SoA memory
    void freeMaterialSoA() {
        auto free_if = [&](auto *&p) {
            if (p) {
                cudaFree(p);
                p = nullptr;
            }
        };

        free_if(d_mat_albedo);
        free_if(d_mat_specular);
        free_if(d_mat_metallic);
        free_if(d_mat_roughness);
        free_if(d_mat_emission);
        free_if(d_mat_ior);
        free_if(d_mat_transmission);
        free_if(d_mat_transmissionRoughness);
        free_if(d_mat_clearcoat);
        free_if(d_mat_clearcoatRoughness);
        free_if(d_mat_subsurfaceColor);
        free_if(d_mat_subsurfaceRadius);
        free_if(d_mat_anisotropy);
        free_if(d_mat_sheen);
        free_if(d_mat_sheenTint);
        free_if(d_mat_iridescence);
        free_if(d_mat_iridescenceThickness);

        free_if(d_materials_ptr_struct);
    }

    void uploadMaterialSoA() {
        if (meshes.empty() || mesh_materials.empty())
            return;

        size_t n = mesh_materials.size();

        std::cout << "Uploading " << n << " materials to GPU..." << std::endl;

        // 1. Create host side SoA buffers
        std::vector<vec3> h_albedo(n);
        std::vector<vec3> h_specular(n);
        std::vector<float> h_metallic(n);
        std::vector<float> h_roughness(n);
        std::vector<vec3> h_emission(n);
        std::vector<float> h_ior(n);
        std::vector<float> h_transmission(n);
        std::vector<float> h_transmissionRoughness(n);
        std::vector<float> h_clearcoat(n);
        std::vector<float> h_clearcoatRoughness(n);
        std::vector<vec3> h_subsurfaceColor(n);
        std::vector<float> h_subsurfaceRadius(n);
        std::vector<float> h_anisotropy(n);
        std::vector<float> h_sheen(n);
        std::vector<vec3> h_sheenTint(n);
        std::vector<float> h_iridescence(n);
        std::vector<float> h_iridescenceThickness(n);

        // 2. Unpack AoS into host SoA
        for (size_t i = 0; i < n; ++i) {
            const Material &m = mesh_materials[i];
            h_albedo[i] = m.albedo;
            h_specular[i] = m.specular;
            h_metallic[i] = m.metallic;
            h_roughness[i] = m.roughness;
            h_emission[i] = m.emission;
            h_ior[i] = m.ior;
            h_transmission[i] = m.transmission;
            h_transmissionRoughness[i] = m.transmissionRoughness;
            h_clearcoat[i] = m.clearcoat;
            h_clearcoatRoughness[i] = m.clearcoatRoughness;
            h_subsurfaceColor[i] = m.subsurfaceColor;
            h_subsurfaceRadius[i] = m.subsurfaceRadius;
            h_anisotropy[i] = m.anisotropy;
            h_sheen[i] = m.sheen;
            h_sheenTint[i] = m.sheenTint;
            h_iridescence[i] = m.iridescence;
            h_iridescenceThickness[i] = m.iridescenceThickness;
        }

        // 3. Free old device buffers
        freeMaterialSoA();

        // 4. Malloc and Memcpy new device buffers
        cudaError_t err;

        auto alloc_and_copy = [&](auto *&d_ptr, const auto &h_vec,
                                  const char *label) {
            using VecT = typename std::remove_reference<decltype(h_vec)>::type;
            using T = typename VecT::value_type;

            err = cudaMalloc(reinterpret_cast<void **>(&d_ptr), n * sizeof(T));
            if (err != cudaSuccess) {
                std::cerr << "failed to alloc " << label << " "
                          << cudaGetErrorString(err) << std::endl;
                freeMaterialSoA();
                throw std::runtime_error("material allocation failed");
            }

            err = cudaMemcpy(d_ptr, h_vec.data(), n * sizeof(T),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "failed to copy " << label << " "
                          << cudaGetErrorString(err) << std::endl;
                freeMaterialSoA();
                throw std::runtime_error("material copy failed");
            }
        };

        alloc_and_copy(d_mat_albedo, h_albedo, "d_mat_albedo");
        alloc_and_copy(d_mat_specular, h_specular, "d_mat_specular");
        alloc_and_copy(d_mat_metallic, h_metallic, "d_mat_metallic");
        alloc_and_copy(d_mat_roughness, h_roughness, "d_mat_roughness");
        alloc_and_copy(d_mat_emission, h_emission, "d_mat_emission");
        alloc_and_copy(d_mat_ior, h_ior, "d_mat_ior");
        alloc_and_copy(d_mat_transmission, h_transmission,
                       "d_mat_transmission");
        alloc_and_copy(d_mat_transmissionRoughness, h_transmissionRoughness,
                       "d_mat_transmissionRoughness");
        alloc_and_copy(d_mat_clearcoat, h_clearcoat, "d_mat_clearcoat");
        alloc_and_copy(d_mat_clearcoatRoughness, h_clearcoatRoughness,
                       "d_mat_clearcoatRoughness");
        alloc_and_copy(d_mat_subsurfaceColor, h_subsurfaceColor,
                       "d_mat_subsurfaceColor");
        alloc_and_copy(d_mat_subsurfaceRadius, h_subsurfaceRadius,
                       "d_mat_subsurfaceRadius");
        alloc_and_copy(d_mat_anisotropy, h_anisotropy, "d_mat_anisotropy");
        alloc_and_copy(d_mat_sheen, h_sheen, "d_mat_sheen");
        alloc_and_copy(d_mat_sheenTint, h_sheenTint, "d_mat_sheenTint");
        alloc_and_copy(d_mat_iridescence, h_iridescence, "d_mat_iridescence");
        alloc_and_copy(d_mat_iridescenceThickness, h_iridescenceThickness,
                       "d_mat_iridescenceThickness");

        // 5. Create the device-side struct of pointers
        DeviceMaterials h_materials_ptr_struct;
        h_materials_ptr_struct.albedo = d_mat_albedo;
        h_materials_ptr_struct.specular = d_mat_specular;
        h_materials_ptr_struct.metallic = d_mat_metallic;
        h_materials_ptr_struct.roughness = d_mat_roughness;
        h_materials_ptr_struct.emission = d_mat_emission;
        h_materials_ptr_struct.ior = d_mat_ior;
        h_materials_ptr_struct.transmission = d_mat_transmission;
        h_materials_ptr_struct.transmissionRoughness =
            d_mat_transmissionRoughness;
        h_materials_ptr_struct.clearcoat = d_mat_clearcoat;
        h_materials_ptr_struct.clearcoatRoughness = d_mat_clearcoatRoughness;
        h_materials_ptr_struct.subsurfaceColor = d_mat_subsurfaceColor;
        h_materials_ptr_struct.subsurfaceRadius = d_mat_subsurfaceRadius;
        h_materials_ptr_struct.anisotropy = d_mat_anisotropy;
        h_materials_ptr_struct.sheen = d_mat_sheen;
        h_materials_ptr_struct.sheenTint = d_mat_sheenTint;
        h_materials_ptr_struct.iridescence = d_mat_iridescence;
        h_materials_ptr_struct.iridescenceThickness =
            d_mat_iridescenceThickness;

        // 6. Upload the struct of pointers to the device
        err = cudaMalloc(&d_materials_ptr_struct, sizeof(DeviceMaterials));
        if (err != cudaSuccess) {
            std::cerr << "Failed to alloc d_materials_ptr_struct: "
                      << cudaGetErrorString(err) << std::endl;
            freeMaterialSoA();
            throw std::runtime_error("Material struct allocation failed");
        }

        err = cudaMemcpy(d_materials_ptr_struct, &h_materials_ptr_struct,
                         sizeof(DeviceMaterials), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy d_materials_ptr_struct: "
                      << cudaGetErrorString(err) << std::endl;
            freeMaterialSoA();
            throw std::runtime_error("Material struct copy failed");
        }

        std::cout << "Materials uploaded successfully." << std::endl;
        std::cout << "d_materials_ptr_struct = " << d_materials_ptr_struct
                  << std::endl;
    }

    void initRandomStates() {
        if (d_rand_states)
            cudaFree(d_rand_states);

        size_t nPixels = static_cast<size_t>(width) * height;
        cudaError_t err =
            cudaMalloc(&d_rand_states, nPixels * sizeof(curandState));
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to alloc rand states");

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        // Seed with clock or constant
        unsigned long long seed = 12345ULL;
        init_curand_kernel<<<grid, block>>>(d_rand_states, width, height, seed);

        err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to init rand states");

        cudaDeviceSynchronize();
    }

    void buildAndUploadTLAS() {
        if (meshes.empty())
            return;

        // This is the equivalent of 'BVHPrimitive'
        // It's a temporary host-side struct for building the TLAS
        struct TLASBuildRef {
            int mesh_id; // The primitive index is the mesh index
            vec3 c;      // Centroid of the AABB
            AABB b;      // The AABB of the mesh
        };

        // 1. Get AABBs and create build refs
        std::vector<TLASBuildRef> tlas_prims(meshes.size());

        for (size_t i = 0; i < meshes.size(); ++i) {
            Mesh *m = meshes[i].get();

            // Use transformed AABB instead of BLAS root AABB
            AABB mesh_aabb;
            if (m->transform.dirty) {
                m->transform.updateMatrices();
            }

            if (m->bvhNodes.empty()) {
                throw std::runtime_error("Mesh BVH not built before TLAS");
            }

            // Get local AABB from BLAS root, transform to world space
            AABB localAABB = m->bvhNodes[0].bbox;
            mesh_aabb = m->transform.transformAABB(localAABB);

            tlas_prims[i].mesh_id = static_cast<int>(i);
            tlas_prims[i].b = mesh_aabb;
            tlas_prims[i].c = mesh_aabb.center();
        }

        // This is the equivalent of the 'BVH' class
        // It's an adaptation of the '_Builder' from mesh.cuh
        struct _TLASBuilder {
            std::vector<DeviceBVHNode> &nodes;
            std::vector<int> &prims; // This will store the mesh indices
            std::vector<TLASBuildRef> &R;
            int leafMax;

            int build(int begin, int end) {
                // compute bounds + centroid bounds
                AABB bb = AABB::make_invalid();
                AABB cb = AABB::make_invalid();
                for (int i = begin; i < end; ++i) {
                    bb.expand(R[i].b);
                    cb.expand(R[i].c);
                }
                int n = end - begin;

                int me = (int)nodes.size();
                nodes.emplace_back(); // placeholder

                nodes[me].bbox = bb;
                nodes[me].left = -1;
                nodes[me].right = -1;
                nodes[me].start = -1;
                nodes[me].count = 0;

                if (n <= leafMax) {
                    nodes[me].start = (int)prims.size();
                    nodes[me].count = n;
                    prims.reserve(prims.size() + n);
                    for (int i = begin; i < end; ++i)
                        prims.push_back(R[i].mesh_id); // Store mesh_id
                    return me;
                }

                // choose split axis = longest centroid axis
                vec3 e = cb.extent();
                int axis = (e.x > e.y && e.x > e.z) ? 0 : ((e.y > e.z) ? 1 : 2);

                int mid = (begin + end) / 2;
                std::nth_element(
                    R.begin() + begin, R.begin() + mid, R.begin() + end,
                    [axis](const TLASBuildRef &A, const TLASBuildRef &B) {
                        return A.c[axis] < B.c[axis];
                    });

                int L = build(begin, mid);
                int Rn = build(mid, end);
                nodes[me].left = L;
                nodes[me].right = Rn;
                return me;
            }
        };

        // 2. Build host-side TLAS
        // Clear old host data
        h_tlasNodes.clear();
        h_tlasMeshIndices.clear();

        // Use the scene's BVH params for the TLAS as well
        const int tlasLeafMax = bvhLeafTarget_ + bvhLeafTol_;

        _TLASBuilder builder{h_tlasNodes, h_tlasMeshIndices, tlas_prims,
                             tlasLeafMax};
        builder.build(0, (int)tlas_prims.size());

        // 3. Upload TLAS to GPU
        if (d_tlasNodes)
            cudaFree(d_tlasNodes);
        if (d_tlasMeshIndices)
            cudaFree(d_tlasMeshIndices);

        cudaError_t err;
        err = cudaMalloc(&d_tlasNodes,
                         h_tlasNodes.size() * sizeof(DeviceBVHNode));
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to alloc d_tlasNodes");

        err = cudaMemcpy(d_tlasNodes, h_tlasNodes.data(),
                         h_tlasNodes.size() * sizeof(DeviceBVHNode),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to copy d_tlasNodes");

        if (!h_tlasMeshIndices.empty()) { // Add safety check
            err = cudaMalloc(&d_tlasMeshIndices,
                             h_tlasMeshIndices.size() * sizeof(int));
            if (err != cudaSuccess)
                throw std::runtime_error("Failed to alloc d_tlasMeshIndices");

            err = cudaMemcpy(d_tlasMeshIndices, h_tlasMeshIndices.data(),
                             h_tlasMeshIndices.size() * sizeof(int),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error("Failed to copy d_tlasMeshIndices");
        } else {
            d_tlasMeshIndices = nullptr;
        }
    }

    void updateAccelerationStructures() {
        if (meshes.empty())
            return;

        bool descriptors_reallocated = false;

        // Manage Memory (Host & Device)
        // Ensure host shadow buffer is the right size
        if (h_mesh_descriptors_.size() != meshes.size()) {
            h_mesh_descriptors_.resize(meshes.size());
            // Initialize with default values to avoid garbage data on first
            // frame
            memset(h_mesh_descriptors_.data(), 0,
                   meshes.size() * sizeof(DeviceMesh));
        }

        // Ensure Device buffer is allocated (standard Device Memory, NOT
        // Managed)
        if (d_mesh_descriptors == nullptr || d_mesh_count_ != meshes.size()) {
            if (d_mesh_descriptors) {
                cudaFree(d_mesh_descriptors);
            }
            cudaError_t err = cudaMalloc(&d_mesh_descriptors,
                                         meshes.size() * sizeof(DeviceMesh));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate mesh descriptors");
            }
            d_mesh_count_ = meshes.size();
            descriptors_reallocated = true;
        }

        // Always upload materials if topology changed
        if (descriptors_reallocated || d_materials_ptr_struct == nullptr) {
            uploadMaterialSoA();
        }

        // 2. Manage Lights
        if (d_lights == nullptr || d_light_count_ != lights.size()) {
            if (d_lights)
                cudaFree(d_lights);
            if (!lights.empty()) {
                // Lights are small, Managed is fine, or switch to Malloc/Memcpy
                // like above
                cudaMallocManaged(&d_lights, lights.size() * sizeof(Light));
                for (size_t i = 0; i < lights.size(); ++i)
                    d_lights[i] = lights[i];
            } else {
                d_lights = nullptr;
            }
            d_light_count_ = lights.size();
        }

        bool tlas_dirty = false;

        // 3. Update Meshes & Descriptors (CPU ONLY LOOP)
        for (size_t i = 0; i < meshes.size(); ++i) {
            Mesh *m = meshes[i].get();
            m->upload(); // Ensure Verts/Faces are on Device

            // A. Rebuild BLAS if needed
            if (m->bvhDirty || m->d_bvhNodes == nullptr) {
                m->setBVHLeafParams(bvhLeafTarget_, bvhLeafTol_);
                m->buildBVH();
                m->uploadBVH();
                m->bvhDirty = false;
                tlas_dirty = true;
            }

            // B. Handle Transforms and History
            // READ from HOST shadow copy, not GPU
            mat4 old_matrix;

            if (descriptors_reallocated) {
                if (m->transform.dirty)
                    m->transform.updateMatrices();
                old_matrix =
                    m->transform.worldMatrix; // No motion on first frame
            } else {
                // We read the matrix we stored in the shadow copy last frame
                old_matrix = h_mesh_descriptors_[i].worldMatrix;
            }

            bool matrix_changed = false;
            if (m->transform.dirty) {
                m->transform.updateMatrices();
                matrix_changed = true;
            } else {
                // Compare Host Shadow vs Current Transform
                if (memcmp(&h_mesh_descriptors_[i].worldMatrix,
                           &m->transform.worldMatrix, sizeof(mat4)) != 0) {
                    matrix_changed = true;
                }
            }

            if (matrix_changed) {
                tlas_dirty = true;
            }

            // C. Update Host Shadow Descriptor
            h_mesh_descriptors_[i].verts = m->d_vertices;
            h_mesh_descriptors_[i].faces = m->d_faces;
            h_mesh_descriptors_[i].faceCount =
                static_cast<int>(m->faces.size());
            h_mesh_descriptors_[i].bvhNodes = m->d_bvhNodes;
            h_mesh_descriptors_[i].nodeCount =
                static_cast<int>(m->bvhNodes.size());
            h_mesh_descriptors_[i].primIndices = m->d_bvhPrim;

            h_mesh_descriptors_[i].worldMatrix = m->transform.worldMatrix;
            h_mesh_descriptors_[i].inverseMatrix = m->transform.inverseMatrix;
            h_mesh_descriptors_[i].normalMatrix = m->transform.normalMatrix;

            // Update Previous Matrix for Motion Vectors
            h_mesh_descriptors_[i].prevWorldMatrix = old_matrix;

            if (!m->bvhNodes.empty()) {
                h_mesh_descriptors_[i].worldAABB =
                    m->transform.transformAABB(m->bvhNodes[0].bbox);
            } else {
                h_mesh_descriptors_[i].worldAABB = AABB::make_invalid();
            }

            h_mesh_descriptors_[i].hasTransform =
                (m->transform.position.length() > 0.001f ||
                 m->transform.rotation.length() > 0.001f ||
                 fabsf(m->transform.scale.x - 1.0f) > 0.001f);
        }

        // 4. Bulk Upload to GPU
        // This is the ONLY interaction with the GPU for mesh descriptors
        cudaMemcpy(d_mesh_descriptors, h_mesh_descriptors_.data(),
                   meshes.size() * sizeof(DeviceMesh), cudaMemcpyHostToDevice);

        // Rebuild TLAS
        if (tlas_dirty || descriptors_reallocated ||
            (d_tlasNodes == nullptr && !meshes.empty())) {
            buildAndUploadTLAS();
        }

        // Safety check for lights
        if (d_lights == nullptr && !lights.empty()) {
            cudaMallocManaged(&d_lights, lights.size() * sizeof(Light));
            for (size_t i = 0; i < lights.size(); ++i)
                d_lights[i] = lights[i];
        }

        gpu_resources_initialized = true;
    }

  public:
    // Constructor
    Scene(int w, int h)
        : width(w), height(h), camera(static_cast<float>(w) / h, 2.0f, 1.0f) {

        initRandomStates();

        size_t nPixels = static_cast<size_t>(width) * height;

        size_t pixel_bytes = nPixels * 3 * sizeof(unsigned char);
        cudaError_t err = cudaMalloc(&d_pixels, pixel_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU pixel buffer: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // This is now the 1-SPP noisy color buffer
        err = cudaMalloc(&d_accum_buffer, nPixels * sizeof(vec3));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "Failed to allocate GPU noisy color buffer");
        }

        // Allocate G-Buffers
        err = cudaMalloc(&d_normal_buffer, nPixels * sizeof(vec3));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU normal buffer");
        }
        err = cudaMalloc(&d_depth_buffer, nPixels * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU depth buffer");
        }
        // Object ID Buffer Allocation
        err = cudaMalloc(&d_objectId_buffer, nPixels * sizeof(int));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU object ID buffer");
        }
        // Initialize to -1 (no object)
        cudaMemset(d_objectId_buffer, -1, nPixels * sizeof(int));

        // Allocate Motion Vector Buffer
        err = cudaMalloc(&d_motion_vectors, nPixels * sizeof(float2));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "Failed to allocate GPU motion vector buffer");
        }

        cudaMalloc(&d_denoised_buffer, nPixels * sizeof(vec3));

        // Initialize wrapper
        DenoiserSettings settings;
        settings.width = width;
        settings.height = height;
        denoiser_ = new Denoiser(settings);

        // Allocate Bloom Buffers
        err = cudaMalloc(&d_bright_pass_buffer, nPixels * sizeof(vec3));
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to alloc d_bright_pass_buffer");

        err = cudaMalloc(&d_bloom_temp_buffer, nPixels * sizeof(vec3));
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to alloc d_bloom_temp_buffer");

        int mip_w = w;
        int mip_h = h;
        for (int i = 0; i < BLOOM_MIP_LEVELS; i++) {
            mip_w = mip_w / 2;
            mip_h = mip_h / 2;
            if (mip_w == 0 || mip_h == 0) {
                d_bloom_mip_chain[i] = nullptr;
                continue;
            }
            size_t mip_pixels = (size_t)mip_w * mip_h;
            err = cudaMalloc(&d_bloom_mip_chain[i], mip_pixels * sizeof(vec3));
            if (err != cudaSuccess)
                throw std::runtime_error("Failed to alloc bloom mip");
        }
        // End Bloom Alloc

        // Clear accumulation/noisy buffer
        cudaMemset(d_accum_buffer, 0, nPixels * sizeof(vec3));

        // Set the initial "previous" matrix to the current one.
        // This ensures the first frame has 0 motion.
        prev_view_proj = camera.get_view_proj();
    }

    // Destructor
    ~Scene() {
        std::cout << "Destroying scene..." << std::endl;

        cudaDeviceSynchronize();

        if (d_rand_states) {
            cudaFree(d_rand_states);
            d_rand_states = nullptr;
        }

        if (d_mesh_descriptors) {
            cudaFree(d_mesh_descriptors);
            d_mesh_descriptors = nullptr;
        }
        if (d_lights) {
            cudaFree(d_lights);
            d_lights = nullptr;
        }
        if (d_pixels) {
            cudaFree(d_pixels);
            d_pixels = nullptr;
        }
        if (d_accum_buffer) {
            cudaFree(d_accum_buffer);
            d_accum_buffer = nullptr;
        }
        if (d_normal_buffer) {
            cudaFree(d_normal_buffer);
            d_normal_buffer = nullptr;
        }
        if (d_depth_buffer) {
            cudaFree(d_depth_buffer);
            d_depth_buffer = nullptr;
        }
        if (d_objectId_buffer) { // Free Object ID Buffer
            cudaFree(d_objectId_buffer);
            d_objectId_buffer = nullptr;
        }

        if (d_denoised_buffer) {
            cudaFree(d_denoised_buffer);
            d_denoised_buffer = nullptr;
        }

        if (d_scaled_accum) {
            cudaFree(d_scaled_accum);
            d_scaled_accum = nullptr;
        }
        if (d_scaled_normal) {
            cudaFree(d_scaled_normal);
            d_scaled_normal = nullptr;
        }
        if (d_scaled_depth) {
            cudaFree(d_scaled_depth);
            d_scaled_depth = nullptr;
        }
        if (d_scaled_objectId) {
            cudaFree(d_scaled_objectId);
            d_scaled_objectId = nullptr;
        }
        if (d_scaled_motion) {
            cudaFree(d_scaled_motion);
            d_scaled_motion = nullptr;
        }
        if (d_scaled_denoised) {
            cudaFree(d_scaled_denoised);
            d_scaled_denoised = nullptr;
        }

        if (denoiser_) {
            denoiser_->destroy();
            delete denoiser_;
            denoiser_ = nullptr;
        }

        if (d_motion_vectors) {
            cudaFree(d_motion_vectors);
            d_motion_vectors = nullptr;
        }

        if (d_bright_pass_buffer) {
            cudaFree(d_bright_pass_buffer);
            d_bright_pass_buffer = nullptr;
        }
        if (d_bloom_temp_buffer) {
            cudaFree(d_bloom_temp_buffer);
            d_bloom_temp_buffer = nullptr;
        }
        for (int i = 0; i < BLOOM_MIP_LEVELS; i++) {
            if (d_bloom_mip_chain[i]) {
                cudaFree(d_bloom_mip_chain[i]);
                d_bloom_mip_chain[i] = nullptr;
            }
        }

        if (d_tlasNodes) {
            cudaFree(d_tlasNodes);
            d_tlasNodes = nullptr;
        }
        if (d_tlasMeshIndices) {
            cudaFree(d_tlasMeshIndices);
            d_tlasMeshIndices = nullptr;
        }
        freeMaterialSoA();
        freeHDRI();

        std::cout << "Scene destroyed." << std::endl;
    }

    Scene(const Scene &) = delete;
    Scene &operator=(const Scene &) = delete;

    // Helper to free HDRI resources
    void freeHDRI() {
        if (d_env_texture) {
            cudaDestroyTextureObject(d_env_texture);
            d_env_texture = 0;
        }
        if (d_env_data) {
            cudaFree(d_env_data);
            d_env_data = nullptr;
        }
    }

    // Public method to load an HDRI
    void loadHDRI(const std::string &filepath) {
        std::cout << "Loading HDRI: " << filepath << "..." << std::endl;

        freeHDRI(); // Cleanup old data

        // 1. Load from disk
        stbi_set_flip_vertically_on_load(true);
        int n_channels;
        // Force 4 channels (RGBA) to ensure 16-byte pixel alignment
        float *h_data = stbi_loadf(filepath.c_str(), &env_width, &env_height,
                                   &n_channels, 4);

        if (!h_data) {
            throw std::runtime_error("Failed to load HDRI file: " + filepath);
        }
        std::cout << "Loaded HDRI: " << env_width << "x" << env_height
                  << std::endl;

        size_t row_bytes = env_width * 4 * sizeof(float);
        size_t pitch_bytes = 0;

        // Use cudaMallocPitch to get the correct hardware alignment
        cudaError_t err =
            cudaMallocPitch(&d_env_data, &pitch_bytes, row_bytes, env_height);
        if (err != cudaSuccess) {
            stbi_image_free(h_data);
            throw std::runtime_error("Failed to alloc HDRI pitch memory");
        }

        // Use cudaMemcpy2D to copy row-by-row into the padded memory
        err = cudaMemcpy2D(d_env_data, pitch_bytes, h_data, row_bytes,
                           row_bytes, env_height, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            stbi_image_free(h_data);
            throw std::runtime_error("Failed to copy HDRI data");
        }

        stbi_image_free(h_data);

        // CREATE TEXTURE OBJECT
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = d_env_data;
        res_desc.res.pitch2D.width = env_width;
        res_desc.res.pitch2D.height = env_height;
        res_desc.res.pitch2D.pitchInBytes = pitch_bytes;
        res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float4>();

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        err = cudaCreateTextureObject(&d_env_texture, &res_desc, &tex_desc,
                                      nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to create env texture: ") +
                cudaGetErrorString(err));
        }

        use_sky = true;
        resetAccumulation();
    }

    void render_to_device(unsigned char *device_pixels) {
        if (!validateGPUResources()) {
            return;
        }

        updateAccelerationStructures();
        updateScaledBuffers();

        auto cudaCheckLaunch = [&](const char *where) {
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                std::cerr << "CUDA kernel launch failed at " << where << ": "
                          << cudaGetErrorString(e) << "\n";
            }
        };

        const int spp = perfSettings.samplesPerPixel;
        const int depth = perfSettings.maxBounceDepth;
        bool is_scaled = (render_width != width || render_height != height);

        // 1. Setup Pointers
        int cur_w = render_width;
        int cur_h = render_height;

        vec3 *ptr_accum = is_scaled ? d_scaled_accum : d_accum_buffer;
        vec3 *ptr_normal = is_scaled ? d_scaled_normal : d_normal_buffer;
        float *ptr_depth = is_scaled ? d_scaled_depth : d_depth_buffer;
        int *ptr_objectId = is_scaled ? d_scaled_objectId : d_objectId_buffer;
        float2 *ptr_motion = is_scaled ? d_scaled_motion : d_motion_vectors;

        // Defensive guards: if the caller changed perfSettings directly without
        // going through setters, render_width/render_height or scaled buffers
        // may be stale/null and would cause a crash.
        if (cur_w <= 0 || cur_h <= 0) {
            cur_w = width;
            cur_h = height;
            is_scaled = false;
        }

        if (is_scaled) {
            if (!ptr_accum || !ptr_normal || !ptr_depth || !ptr_objectId) {
                std::cerr << "WARNING: scaled buffers not allocated; falling "
                             "back to full resolution\n";
                is_scaled = false;
                cur_w = width;
                cur_h = height;
                ptr_accum = d_accum_buffer;
                ptr_normal = d_normal_buffer;
                ptr_depth = d_depth_buffer;
                ptr_objectId = d_objectId_buffer;
                ptr_motion = d_motion_vectors;
            }
        }

        vec3 *current_image = ptr_accum;

        // CONFIG 1: Path Tracing (Heavy Register Usage -> Use 8x8)
        dim3 blockPT(pt_render_config::pt_block_x,
                     pt_render_config::pt_block_y);
        dim3 gridPT((cur_w + blockPT.x - 1) / blockPT.x,
                    (cur_h + blockPT.y - 1) / blockPT.y);

        // 2. Trace Rays (Low Res)
        path_trace_kernel<<<gridPT, blockPT>>>(
            ptr_accum, ptr_normal, ptr_depth, ptr_objectId, cur_w, cur_h,
            camera, d_mesh_descriptors, static_cast<int>(meshes.size()),
            d_lights, static_cast<int>(lights.size()), sky_color_top,
            sky_color_bottom, use_sky, spp, depth, d_tlasNodes,
            d_tlasMeshIndices, d_materials_ptr_struct, d_env_texture,
            d_rand_states, frame_count_);
        cudaCheckLaunch("path_trace_kernel");

        frame_count_++;

        // 3. Motion Vectors (Low Res)
        if (perfSettings.enableMotionVectors && perfSettings.enableDenoiser) {
            motion_vector_kernel<<<gridPT, blockPT>>>(
                ptr_motion, ptr_depth, cur_w, cur_h, camera, prev_view_proj);
            cudaCheckLaunch("motion_vector_kernel");
        }

        // 4. Denoiser (Low Res)
        if (perfSettings.enableDenoiser && denoiser_) {
            DenoiserInputs inputs;
            inputs.noisyColor = ptr_accum;
            inputs.normal = ptr_normal;
            inputs.depth = ptr_depth;
            inputs.motion = ptr_motion;
            inputs.objectId = ptr_objectId;

            DenoiserCommonSettings common;
            common.viewProj = camera.get_view_proj();
            common.prevViewProj = prev_view_proj;

            vec3 *denoise_target =
                is_scaled ? d_scaled_denoised : d_denoised_buffer;

            denoiser_->denoise(inputs, common, denoise_target);
            current_image = denoise_target;
        }
        // CONFIG 2: Post-Processing (Low Register Usage -> Use 16x16)

        dim3 blockPost(pt_render_config::simple_block_x,
                       pt_render_config::simple_block_y);

        // Grid for current resolution (low res if scaling is on)
        dim3 gridPost((cur_w + blockPost.x - 1) / blockPost.x,
                      (cur_h + blockPost.y - 1) / blockPost.y);

        // 5. Bloom (Low Res)
        if (perfSettings.enableBloom) {
            // Use gridPost/blockPost here
            bloom_bright_pass_kernel<<<gridPost, blockPost>>>(
                d_bright_pass_buffer, current_image, cur_w, cur_h, 1.5f, 0.5f);
            cudaCheckLaunch("bloom_bright_pass_kernel");

            int mip_w = cur_w, mip_h = cur_h;
            vec3 *last_mip = d_bright_pass_buffer;

            for (int i = 0; i < BLOOM_MIP_LEVELS; i++) {
                if (!d_bloom_mip_chain[i])
                    break;
                int next_w = mip_w / 2;
                int next_h = mip_h / 2;
                if (next_w == 0 || next_h == 0)
                    break;

                dim3 g_h((mip_w + 15) / 16, (mip_h + 15) / 16);
                bloom_blur_h_kernel<<<g_h, blockPost>>>(
                    d_bloom_temp_buffer, last_mip, mip_w, mip_h, 1.0f);

                dim3 g_v((next_w + 15) / 16, (next_h + 15) / 16);
                bloom_downsample_v_kernel<<<g_v, blockPost>>>(
                    d_bloom_mip_chain[i], d_bloom_temp_buffer, mip_w, mip_h,
                    1.0f);

                last_mip = d_bloom_mip_chain[i];
                mip_w = next_w;
                mip_h = next_h;
            }

            for (int i = BLOOM_MIP_LEVELS - 2; i >= 0; i--) {
                if (!d_bloom_mip_chain[i])
                    continue;
                mip_w *= 2;
                mip_h *= 2;
                dim3 g_up((mip_w + 15) / 16, (mip_h + 15) / 16);
                bloom_upsample_add_kernel<<<g_up, blockPost>>>(
                    d_bloom_mip_chain[i], d_bloom_mip_chain[i + 1], mip_w / 2,
                    mip_h / 2);
            }

            // Apply bloom to the CURRENT image
            bloom_upsample_add_kernel<<<gridPost, blockPost>>>(
                current_image, d_bloom_mip_chain[0], cur_w / 2, cur_h / 2);
        }

        // 6. Upscale (Low Res -> Full Res)
        vec3 *final_hdr_image = current_image;

        // Grid for FULL resolution output
        dim3 fullGridPost((width + blockPost.x - 1) / blockPost.x,
                          (height + blockPost.y - 1) / blockPost.y);

        if (is_scaled) {
            upscale_bilinear_kernel<<<fullGridPost, blockPost>>>(
                d_accum_buffer, // Dest: Full Res Main Buffer
                current_image,  // Source: Low Res Processed Image
                width, height,  // Dest Dims
                cur_w, cur_h    // Source Dims
            );
            cudaCheckLaunch("upscale_bilinear_kernel");
            final_hdr_image = d_accum_buffer;
        }

        // 7. Tonemap (Full Res)
        tonemap_kernel<<<fullGridPost, blockPost>>>(
            device_pixels, final_hdr_image, width, height, 1);
        cudaCheckLaunch("tonemap_kernel");

        prev_view_proj = camera.get_view_proj();
    }

    void render_to_device_wireframe(unsigned char *device_pixels,
                                    float wireframeThickness) {
        if (meshes.empty()) {
            std::cerr << "Error: no meshes in scene\n";
            return;
        }

        // Ensure all GPU resources are valid
        updateAccelerationStructures();

        // Configure kernel
        dim3 block(pt_render_config::pt_block_x, pt_render_config::pt_block_y);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        // Launch Wireframe Kernel
        render_kernel_wireframe<<<grid, block>>>(
            device_pixels, width, height, camera, d_mesh_descriptors,
            static_cast<int>(meshes.size()), d_lights,
            static_cast<int>(lights.size()), sky_color_top, sky_color_bottom,
            use_sky,
            true,               // wireframeMode
            wireframeThickness, // Pass the thickness
            d_materials_ptr_struct, d_tlasNodes, d_tlasMeshIndices,
            d_env_texture // Pass HDRI map
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Wireframe kernel failed: ") +
                                     cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();
    }

    // setters
    void setSamplesPerPixel(int spp) {
        samples_per_pixel_ = spp;
        resetAccumulation();
    }
    void setMaxDepth(int depth) {
        max_depth_ = depth;
        resetAccumulation();
    }

    void setMeshMaterial(size_t index, const Material &mat) {
        if (index < mesh_materials.size()) {
            mesh_materials[index] = mat;
        }
    }

    void commitMaterialChanges() {
        // Re-uploads the material buffers to the GPU
        uploadMaterialSoA();
        // Reset accumulation so the image updates immediately
        resetAccumulation();
    }

    void resetAccumulation() {
        frame_count_ = 0;
        // Clear main buffer
        if (d_accum_buffer) {
            cudaMemset(d_accum_buffer, 0,
                       static_cast<size_t>(width) * height * sizeof(vec3));
        }

        // Clear scaled buffer if it exists
        if (d_scaled_accum && render_width > 0 && render_height > 0) {
            cudaMemset(d_scaled_accum, 0,
                       static_cast<size_t>(render_width) * render_height *
                           sizeof(vec3));
        }

        prev_view_proj = camera.get_view_proj();
    }

    void setBVHLeafTarget(int target, int tol = 5) {
        bvhLeafTarget_ = (target < 1 ? 1 : target);
        bvhLeafTol_ = (tol < 0 ? 0 : tol);
        // mark meshes dirty
        for (auto &m : meshes)
            m->bvhDirty = true;
        resetAccumulation(); // BVH change requires reset
    }

    // Camera setup methods
    void setCamera(const vec3 &lookfrom, const vec3 &lookat, const vec3 &vup,
                   float vfov, float aperture = 0.0f, float focus_dist = 1.0f) {
        float aspect = static_cast<float>(width) / height;
        // Pass near/far to NRD-ready camera constructor
        camera =
            Camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist,
                   0.1f, 1000.0f); // Added 0.1f, 1000.0f for near/far
        resetAccumulation();       // Camera change requires reset
    }

    void setCameraSimple(float viewport_height = 2.0f,
                         float focal_length = 1.0f) {
        float aspect = static_cast<float>(width) / height;
        camera = Camera(aspect, viewport_height, focal_length);
        resetAccumulation();
    }

    // Camera helpers for runtime control
    vec3 cameraOrigin() const { return camera.get_origin(); }
    vec3 cameraForward() const {
        // Direction of ray through the center pixel
        vec3 dir = camera.get_lower_left_corner() +
                   camera.get_horizontal() * 0.5f +
                   camera.get_vertical() * 0.5f - camera.get_origin();
        return dir.normalized();
    }
    void moveCamera(const vec3 &pos) {
        camera.set_position(pos);
        resetAccumulation();
    }
    void lookCameraAt(const vec3 &target, const vec3 &vup = vec3(0, 1, 0)) {
        camera.look_at(target, vup);
        resetAccumulation();
    }

    // Mesh management with materials
    Mesh *addMesh(const std::string &obj_path,
                  const Material &mat = Material()) {
        meshes.push_back(std::make_unique<Mesh>(obj_path));
        mesh_materials.push_back(mat);
        resetAccumulation();
        return meshes.back().get();
    }

    inline Mesh *addTriangles(const std::vector<Triangle> &tris,
                              const Material &mat = Material()) {
        meshes.push_back(std::make_unique<Mesh>());
        mesh_materials.push_back(mat);

        Mesh *m = meshes.back().get();

        m->vertices.clear();
        m->faces.clear();

        m->vertices.reserve(tris.size() * 3);
        m->faces.reserve(tris.size());

        for (const Triangle &t : tris) {
            const int base = static_cast<int>(m->vertices.size());
            m->vertices.push_back(t.v0);
            m->vertices.push_back(t.v1);
            m->vertices.push_back(t.v2);

            m->faces.push_back(Tri{base + 0, base + 1, base + 2});
        }
        resetAccumulation();
        return m;
    }

    HitInfo traceSingleRay(const vec3 &origin, const vec3 &direction) {
        HitInfo h_result;
        HitInfo *d_result;

        // Allocate device memory for result
        cudaMalloc(&d_result, sizeof(HitInfo));

        // Launch kernel with single thread
        trace_single_ray_kernel<<<1, 1>>>(origin, direction, d_mesh_descriptors,
                                          static_cast<int>(meshes.size()),
                                          d_tlasNodes, // Pass TLAS
                                          d_tlasMeshIndices, d_result);

        // Wait for kernel to complete
        cudaDeviceSynchronize();

        // Copy result back to host
        cudaMemcpy(&h_result, d_result, sizeof(HitInfo),
                   cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_result);

        return h_result;
    }

    inline Mesh *addPlaneXZ(float planeY, float halfSize,
                            const Material &mat = Material(vec3(0.8f))) {
        // Square in XZ at y=planeY
        const vec3 A(-halfSize, planeY, -halfSize);
        const vec3 B(halfSize, planeY, -halfSize);
        const vec3 C(halfSize, planeY, halfSize);
        const vec3 D(-halfSize, planeY, halfSize);

        std::vector<Triangle> tris;
        tris.reserve(2);

        // Wind CCW as seen from +Y so normal is +Y
        tris.emplace_back(A, C, B);
        tris.emplace_back(A, D, C);

        return addTriangles(tris, mat);
    }

    inline void addCheckerboardPlaneXZ(float planeY, int tilesPerSide,
                                       float tileSize, const Material &whiteMat,
                                       const Material &blackMat) {
        std::vector<Triangle> whiteTris, blackTris;
        whiteTris.reserve(tilesPerSide * tilesPerSide * 2);
        blackTris.reserve(tilesPerSide * tilesPerSide * 2);

        const int N = tilesPerSide;
        const float start = -N * tileSize;

        for (int iz = 0; iz < 2 * N; ++iz) {
            for (int ix = 0; ix < 2 * N; ++ix) {
                const float x0 = start + ix * tileSize;
                const float x1 = x0 + tileSize;
                const float z0 = start + iz * tileSize;
                const float z1 = z0 + tileSize;

                const vec3 A(x0, planeY, z0);
                const vec3 B(x1, planeY, z0);
                const vec3 C(x1, planeY, z1);
                const vec3 D(x0, planeY, z1);

                const bool white = ((ix + iz) & 1) == 0;
                auto &bucket = white ? whiteTris : blackTris;

                // CCW from +Y: (A,C,B) and (A,D,C)
                bucket.emplace_back(A, C, B);
                bucket.emplace_back(A, D, C);
            }
        }

        if (!whiteTris.empty())
            addTriangles(whiteTris, whiteMat);
        if (!blackTris.empty())
            addTriangles(blackTris, blackMat);
    }

    Mesh *addCube(const Material &mat = Material(vec3(1.0f, 0.0f, 0.0f))) {
        meshes.push_back(std::make_unique<Mesh>());
        mesh_materials.push_back(mat);
        resetAccumulation();
        return meshes.back().get();
    }

    // Add a UV sphere with the given number of segments
    Mesh *addSphere(int segments = 32,
                    const Material &mat = Material(vec3(1.0f, 0.0f, 0.0f))) {
        auto sphereMesh = std::make_unique<Mesh>();

        // Clear default cube geometry
        sphereMesh->vertices.clear();
        sphereMesh->faces.clear();

        const int rings = segments;
        const int sectors = segments;
        const float radius = 0.5f; // Unit sphere (diameter 1)

        // Generate vertices
        for (int r = 0; r <= rings; ++r) {
            float phi = PI * float(r) / float(rings); // 0 to PI
            float y = cosf(phi) * radius;
            float ringRadius = sinf(phi) * radius;

            for (int s = 0; s <= sectors; ++s) {
                float theta = TWO_PI * float(s) / float(sectors); // 0 to 2*PI
                float x = ringRadius * cosf(theta);
                float z = ringRadius * sinf(theta);
                sphereMesh->vertices.push_back(vec3(x, y, z));
            }
        }

        // Generate faces
        for (int r = 0; r < rings; ++r) {
            for (int s = 0; s < sectors; ++s) {
                int curr = r * (sectors + 1) + s;
                int next = curr + sectors + 1;

                // Two triangles per quad
                // First triangle
                sphereMesh->faces.push_back({curr, next, curr + 1});
                // Second triangle
                sphereMesh->faces.push_back({curr + 1, next, next + 1});
            }
        }

        meshes.push_back(std::move(sphereMesh));
        mesh_materials.push_back(mat);
        resetAccumulation();
        return meshes.back().get();
    }

    // Add light methods now include radius
    void addPointLight(const vec3 &position, const vec3 &color,
                       float intensity = 1.0f, float range = 100.0f,
                       float radius = 0.0f) { // Added radius
        Light light;
        light.type = LIGHT_POINT;
        light.position = position;
        light.color = color;
        light.intensity = intensity;
        light.range = range;
        light.radius = radius; // Set radius
        lights.push_back(light);
        resetAccumulation();
    }

    void addDirectionalLight(const vec3 &direction, const vec3 &color,
                             float intensity = 1.0f) {
        Light light;
        light.type = LIGHT_DIRECTIONAL;
        light.direction = direction.normalized();
        light.color = color;
        light.intensity = intensity;
        lights.push_back(light);
        resetAccumulation();
    }

    void addSpotLight(const vec3 &position, const vec3 &direction,
                      const vec3 &color, float intensity = 1.0f,
                      float innerCone = 0.5f, float outerCone = 0.7f,
                      float range = 100.0f,
                      float radius = 0.0f) { // Added radius
        Light light;
        light.type = LIGHT_SPOT;
        light.position = position;
        light.direction = direction.normalized();
        light.color = color;
        light.intensity = intensity;
        light.innerCone = cosf(innerCone);
        light.outerCone = cosf(outerCone);
        light.range = range;
        light.radius = radius; // Set radius
        lights.push_back(light);
        resetAccumulation();
    }

    // Background settings
    void setSkyGradient(const vec3 &top, const vec3 &bottom) {
        sky_color_top = top;
        sky_color_bottom = bottom;
        use_sky = true;

        // If user sets a gradient, disable the HDRI
        freeHDRI();

        resetAccumulation();
    }

    void disableSky() {
        use_sky = false;
        resetAccumulation();
    }

    void setWireframeMode(bool enabled) { wireframe_mode = enabled; }
    bool isWireframeMode() const { return wireframe_mode; }
    void toggleWireframeMode() { wireframe_mode = !wireframe_mode; }
    void setShowFrustum(bool show) { show_frustum = show; }
    void toggleFrustum() { show_frustum = !show_frustum; }
    void addDebugRay(const vec3 &origin, const vec3 &direction) {
        debug_rays.push_back({origin, direction});
    }
    void addDebugRayWithLength(const vec3 &origin, const vec3 &direction,
                               float length) {
        debug_rays_with_length.push_back(
            {origin, normalize(direction), length});
    }
    void clearDebugRays() {
        debug_rays.clear();
        debug_rays_with_length.clear();
    }
    void clearVisualizationMeshes() {
        // Remove visualization meshes in reverse order to maintain indices
        for (auto it = visualization_mesh_indices.rbegin();
             it != visualization_mesh_indices.rend(); ++it) {
            if (*it < meshes.size()) {
                meshes.erase(meshes.begin() + *it);
                mesh_materials.erase(mesh_materials.begin() + *it);
            }
        }
        visualization_mesh_indices.clear();
        resetAccumulation(); // Visual meshes were removed
    }
    Mesh *addVisualizationTriangles(const std::vector<Triangle> &tris,
                                    const Material &mat) {
        visualization_mesh_indices.push_back(meshes.size());
        return addTriangles(tris, mat);
    }
    void setRayLength(float length) { ray_length = length; }
    void setShowRays(bool show) { show_rays = show; }

    // Gave the unnamed function a name
    // Add visualization meshes to scene
    void updateVisualizationMeshes() {
        clearVisualizationMeshes(); // This will call resetAccumulation

        if (show_frustum) {
            if (frustum_mesh_index_ == size_t(-1)) {
                Material frustumMat(vec3(0.2f, 0.8f, 0.2f), 0.0f, 0.0f);
                frustumMat.emission = vec3(0.1f, 0.4f, 0.1f);
                auto frustumTris = Visualization::generateFrustumWireframe(
                    camera, static_cast<float>(width) / height, 0.1f, 100.0f,
                    0.005f);
                if (!frustumTris.empty()) {
                    Mesh *frustumMesh =
                        addVisualizationTriangles(frustumTris, frustumMat);
                    frustum_mesh_index_ = visualization_mesh_indices.back();
                }
            }
            show_frustum = false;
        }
        if (show_rays &&
            (!debug_rays.empty() || !debug_rays_with_length.empty())) {
            Material rayMat(vec3(1.0f, 0.2f, 0.2f), 0.0f, 0.0f);
            rayMat.emission = vec3(0.5f, 0.1f, 0.1f);
            for (const auto &ray : debug_rays) {
                auto arrowTris = Visualization::generateArrow(
                    ray.first, ray.second, ray_length, 0.01f, 0.03f, 0.1f);
                if (!arrowTris.empty()) {
                    addVisualizationTriangles(arrowTris, rayMat);
                }
            }
            for (const auto &ray : debug_rays_with_length) {
                auto arrowTris = Visualization::generateArrow(
                    ray.origin, ray.direction, ray.length, 0.01f, 0.03f, 0.1f);
                if (!arrowTris.empty()) {
                    addVisualizationTriangles(arrowTris, rayMat);
                }
            }
        }
    }

    // Upload scene to GPU
    void uploadToGPU() {
        if (meshes.empty()) {
            std::cerr << "Warning: No meshes in scene\n";
            return;
        }

        if (meshes.size() != mesh_materials.size()) {
            throw std::runtime_error("Mesh count and material count mismatch!");
        }

        // Upload/Build BLAS (Bottom-Level)
        updateAccelerationStructures();
        gpu_resources_initialized = true; // Mark as done
        resetAccumulation();
    }

    void generatePrimaryRayVisualization(Camera cam, int numRays = 10) {
        clearDebugRays();
        int gridSize = (int)sqrt(numRays);
        for (int y = 0; y < gridSize; ++y) {
            for (int x = 0; x < gridSize; ++x) {
                float u = (x + 0.5f) / gridSize;
                float v = (y + 0.5f) / gridSize;
                Ray ray = cam.get_ray(u, v); // Assumes old get_ray
                addDebugRay(ray.origin(), ray.direction());
            }
        }
        std::cout << "generated primary rays" << '\n';
    }
    void generateReflectionRayVisualization(const vec3 &hitPoint,
                                            const vec3 &normal,
                                            const vec3 &incidentDir) {
        vec3 reflectedDir = reflectVec(incidentDir, normal); // Use reflectVec
        addDebugRay(hitPoint + normal * 0.01f, reflectedDir);
    }
    void generateRefractionRayVisualization(const vec3 &hitPoint,
                                            const vec3 &normal,
                                            const vec3 &incidentDir,
                                            float ior) {
        vec3 refractedDir;
        vec3 I = incidentDir;
        vec3 N = normal; // Assume normal is already face-forward
        float eta = (dot(N, I) > 0.0f) ? ior : (1.0f / ior); // Simple check
        if (refractVec(I, N, eta, refractedDir)) {
            addDebugRay(hitPoint - normal * 0.01f, refractedDir);
        }
    }

    // Main render loop

    // Save to PPM file
    void saveAsPPM(const std::string &filename, unsigned char *pixels) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        ofs << "P3\n" << width << ' ' << height << "\n255\n";
        size_t idx = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ofs << int(pixels[idx]) << ' ' << int(pixels[idx + 1]) << ' '
                    << int(pixels[idx + 2]) << '\n';
                idx += 3;
            }
        }
        ofs.close();
    }

    // Getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    size_t getPixelBufferSize() const {
        return static_cast<size_t>(width) * height * 3;
    }
    Camera &getCamera() { return camera; }

    int getFrameCount() const { return frame_count_; }
    int getSamplesPerPixel() const { return samples_per_pixel_; }

    vec3 *getNoisyColorBuffer() { return d_accum_buffer; }
    vec3 *getNormalBuffer() { return d_normal_buffer; }
    float *getDepthBuffer() { return d_depth_buffer; }
    float2 *getMotionVectorBuffer() { return d_motion_vectors; }
    mat4 getPrevViewProjMatrix() { return prev_view_proj; }

    Mesh *getMesh(size_t index) {
        if (index < meshes.size()) {
            return meshes[index].get();
        }
        return nullptr;
    }

    /// Get const mesh pointer
    const Mesh *getMesh(size_t index) const {
        if (index < meshes.size()) {
            return meshes[index].get();
        }
        return nullptr;
    }

    /// Get number of meshes in scene
    size_t getMeshCount() const { return meshes.size(); }

    /// Move a mesh to an absolute world position
    /// Automatically triggers BVH and TLAS rebuild on next update
    void moveMeshTo(size_t index, const vec3 &position) {
        if (Mesh *m = getMesh(index)) {
            m->moveTo(position);
            // bvhDirty is set by moveTo()
        }
    }

    /// Translate a mesh by a delta
    void translateMesh(size_t index, const vec3 &delta) {
        if (Mesh *m = getMesh(index)) {
            m->translate(delta);
        }
    }

    /// Rotate a mesh around its center
    /// @param radians Euler angles in radians (X, Y, Z rotation order)
    void rotateMesh(size_t index, const vec3 &radians) {
        if (Mesh *m = getMesh(index)) {
            m->rotateSelfEulerXYZ(radians);
        }
    }

    /// Scale a mesh uniformly
    void scaleMesh(size_t index, float scale) {
        if (Mesh *m = getMesh(index)) {
            m->scale(scale);
        }
    }

    /// Scale a mesh non-uniformly
    void scaleMesh(size_t index, const vec3 &scale) {
        if (Mesh *m = getMesh(index)) {
            m->scale(scale);
        }
    }

    void commitObjectChanges() {
        updateAccelerationStructures();
        resetAccumulation(); // Restart progressive rendering
    }

    /// Check if any mesh has pending changes
    bool hasObjectChanges() const {
        for (const auto &mesh : meshes) {
            if (mesh->bvhDirty)
                return true;
        }
        return false;
    }

    /// Get a light for manipulation
    Light *getLight(size_t index) {
        if (index < lights.size()) {
            return &lights[index];
        }
        return nullptr;
    }

    size_t getLightCount() const { return lights.size(); }

    /// Move a light to a new position
    void moveLightTo(size_t index, const vec3 &position) {
        if (Light *light = getLight(index)) {
            light->position = position;
            // Lights don't need BVH rebuild, just re-upload
            if (d_lights && index < d_light_count_) {
                cudaMemcpy(&d_lights[index], light, sizeof(Light),
                           cudaMemcpyHostToDevice);
            }
            resetAccumulation();
        }
    }

    /// Update all lights on GPU (call after modifying light properties)
    void commitLightChanges() {
        if (!lights.empty() && d_lights) {
            cudaMemcpy(d_lights, lights.data(), lights.size() * sizeof(Light),
                       cudaMemcpyHostToDevice);
            resetAccumulation();
        }
    }

    // PERFORMANCE CONFIGURATION METHODS

    /// Set performance preset for different use cases
    void setPerformancePreset(const std::string &preset) {
        if (preset == "ultra") {
            // Best quality, slowest
            perfSettings.enableDenoiser = false;
            perfSettings.enableBloom = true;
            perfSettings.enableMotionVectors = true;
            perfSettings.samplesPerPixel = 128;
            perfSettings.maxBounceDepth = 32;
            perfSettings.resolutionScale = 1.0f;
            perfSettings.russianRouletteStartBounce = 8;
        } else if (preset == "quality") {
            // excellent quality, slow
            perfSettings.enableDenoiser = true;
            perfSettings.enableBloom = true;
            perfSettings.enableMotionVectors = true;
            perfSettings.maxBounceDepth = 6;
            perfSettings.resolutionScale = 1.0f;
            perfSettings.russianRouletteStartBounce = 2;
        } else if (preset == "balanced") {
            // Good balance of quality and speed
            perfSettings.enableDenoiser = true;
            perfSettings.enableBloom = true;
            perfSettings.enableMotionVectors = true;
            perfSettings.maxBounceDepth = 4;
            perfSettings.resolutionScale = 1.0f;
            perfSettings.russianRouletteStartBounce = 1;
        } else if (preset == "performance") {
            // Fast with acceptable quality
            perfSettings.enableDenoiser = true;
            perfSettings.enableBloom = false;
            perfSettings.enableMotionVectors = true;
            perfSettings.maxBounceDepth = 3;
            perfSettings.resolutionScale = 0.75f;
            perfSettings.russianRouletteStartBounce = 1;
        } else if (preset == "fast") {
            // Maximum speed, reduced quality
            perfSettings.enableDenoiser = false;
            perfSettings.enableBloom = false;
            perfSettings.enableMotionVectors = false;
            perfSettings.maxBounceDepth = 2;
            perfSettings.resolutionScale = 0.35f;
            perfSettings.russianRouletteStartBounce = 1;
        }

        // Reallocate scaled buffer if resolution changed
        updateScaledBuffers();
    }

    /// Enable/disable denoiser (significant performance impact)
    void setDenoiserEnabled(bool enabled) {
        if (perfSettings.enableDenoiser != enabled) {
            perfSettings.enableDenoiser = enabled;
            // Recreate or destroy denoiser and any scale-dependent resources.
            updateScaledBuffers();
        }
    }

    /// Enable/disable bloom (moderate performance impact)
    void setBloomEnabled(bool enabled) { perfSettings.enableBloom = enabled; }

    /// Set maximum bounce depth (2-4 for realtime, 6-8 for quality)
    void setMaxBounceDepth(int depth) {
        perfSettings.maxBounceDepth =
            (depth < 1) ? 1 : ((depth > 16) ? 16 : depth);
    }

    /// Set resolution scale (0.25 to 1.0, lower = faster)
    void setResolutionScale(float scale) {
        scale = fmaxf(0.25f, fminf(1.0f, scale));
        if (fabsf(scale - perfSettings.resolutionScale) > 0.01f) {
            perfSettings.resolutionScale = scale;
            updateScaledBuffers();
        }
    }

    /// Get current performance settings
    const PerformanceSettings &getPerformanceSettings() const {
        return perfSettings;
    }

  private:
    void updateScaledBuffers() {
        // Calculate new dimensions
        int new_render_width =
            static_cast<int>(width * perfSettings.resolutionScale);
        int new_render_height =
            static_cast<int>(height * perfSettings.resolutionScale);

        // Clamping to safe minimums
        new_render_width = (new_render_width < 64) ? 64 : new_render_width;
        new_render_height = (new_render_height < 64) ? 64 : new_render_height;

        // Only reallocate if dimensions actually changed
        if (new_render_width != render_width ||
            new_render_height != render_height) {
            render_width = new_render_width;
            render_height = new_render_height;

            bool is_scaled = (render_width != width || render_height != height);

            auto cudaFreeSafe = [](void *&p) {
                if (p) {
                    cudaError_t e = cudaFree(p);
                    // optional: handle/log e
                    p = nullptr;
                }
            };

            auto cudaMallocSafe = [](void **p, size_t bytes) {
                cudaError_t e = cudaMalloc(p, bytes);
                // optional: handle/log e
                return e;
            };

            // Free existing scaled buffers
            cudaFreeSafe((void *&)d_scaled_accum);
            cudaFreeSafe((void *&)d_scaled_normal);
            cudaFreeSafe((void *&)d_scaled_depth);
            cudaFreeSafe((void *&)d_scaled_objectId);
            cudaFreeSafe((void *&)d_scaled_motion);
            cudaFreeSafe((void *&)d_scaled_denoised);

            // Allocate new scaled buffers ONLY if scaling is active
            if (is_scaled) {
                size_t pixel_count =
                    (size_t)render_width * (size_t)render_height;

                // If any of these fails, you should bail or keep pointers null.
                cudaMallocSafe((void **)&d_scaled_accum,
                               pixel_count * sizeof(vec3));
                cudaMallocSafe((void **)&d_scaled_normal,
                               pixel_count * sizeof(vec3));
                cudaMallocSafe((void **)&d_scaled_depth,
                               pixel_count * sizeof(float));
                cudaMallocSafe((void **)&d_scaled_objectId,
                               pixel_count * sizeof(int));
                cudaMallocSafe((void **)&d_scaled_motion,
                               pixel_count * sizeof(float2));
                cudaMallocSafe((void **)&d_scaled_denoised,
                               pixel_count * sizeof(vec3));
            }

            // DENOISER: only exist when enabled
            if (!perfSettings.enableDenoiser) {
                // If preset disables denoiser, make sure it's destroyed
                if (denoiser_) {
                    denoiser_->destroy();
                    delete denoiser_;
                    denoiser_ = nullptr;
                }
            } else {
                // enabled -> (re)create at current render resolution
                if (denoiser_) {
                    denoiser_->destroy();
                    delete denoiser_;
                    denoiser_ = nullptr;
                }

                DenoiserSettings settings;
                settings.width = render_width;
                settings.height = render_height;
                denoiser_ = new Denoiser(settings);
            }

            // Clear the accumulation to start fresh
            resetAccumulation();
        }
    }
};

// Tonemapping Kernel (with ACES)
__global__ void tonemap_kernel(unsigned char *__restrict__ out,
                               const vec3 *__restrict__ in_buffer, int W, int H,
                               int total_samples) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const size_t idx = (static_cast<size_t>(y) * W + x);
    const size_t out_idx =
        (static_cast<size_t>(H - 1 - y) * W + x) * 3; // Flipped Y

    if (total_samples == 0) {
        out[out_idx + 0] = 0;
        out[out_idx + 1] = 0;
        out[out_idx + 2] = 0;
        return;
    }

    // 1. Average
    vec3 color = in_buffer[idx] / (float)total_samples;

    // 2. Tone Map (ACES)
    color = aces_tonemap(color); // CHANGED

    // 3. Gamma Correction (sRGB OETF)
    // ACES output is linear, so we must apply gamma
    color.x = (color.x <= 0.0031308f)
                  ? 12.92f * color.x
                  : 1.055f * powf(color.x, 1.0f / 2.4f) - 0.055f;
    color.y = (color.y <= 0.0031308f)
                  ? 12.92f * color.y
                  : 1.055f * powf(color.y, 1.0f / 2.4f) - 0.055f;
    color.z = (color.z <= 0.0031308f)
                  ? 12.92f * color.z
                  : 1.055f * powf(color.z, 1.0f / 2.4f) - 0.055f;

    // 4. Convert to 8-bit
    const vec3 rgb = clamp(color, 0.f, 1.f) * 255.99f;
    out[out_idx + 0] = static_cast<unsigned char>(rgb.x);
    out[out_idx + 1] = static_cast<unsigned char>(rgb.y);
    out[out_idx + 2] = static_cast<unsigned char>(rgb.z);
}

#endif // SCENE_CUH
