# CUDA Path Tracer

A GPU-accelerated physically based **path tracer** written in **C++/CUDA**, designed for “real-time-ish” interactive rendering: **1 spp per frame**, progressive accumulation, a **spatiotemporal denoiser** (motion vectors + A-Trous), optional **bloom**, and a small scene API for meshes, primitives, materials, lights, and camera control.

---

## Features

### Rendering
- **CUDA path tracing integrator** with:
  - BSDF sampling + **next-event estimation** (direct light sampling)
  - Throughput clamping / contribution clamping to reduce fireflies
  - Optional **Russian roulette** early termination (Scene-level toggle)
- **Split-path rendering** (useful for denoising):
  - Separates **diffuse**, **specular**, and **emission** contributions (`SplitPathOutput`)

### Materials (PBR)
Backed by a **Material struct (inferred)** uploaded to GPU as a **SoA (structure-of-arrays)** for fast device reads (`DeviceMaterials`).
Observed fields include:
- `albedo`, `specular`, `metallic`, `roughness`, `emission`
- `ior`
- `transmission`, `transmissionRoughness`
- `clearcoat`, `clearcoatRoughness`
- `subsurfaceColor`, `subsurfaceRadius`
- `sheen`, `sheenTint`
- `anisotropy`
- `iridescence`, `iridescenceThickness`

### Geometry + Acceleration
- **Triangle meshes** with:
  - CPU storage + GPU upload of vertices/faces
  - Per-mesh **BVH (BLAS)** build on CPU + upload to GPU
  - Simple transform controls (move/translate/rotate/scale)
- Scene-level **TLAS** (top-level BVH) built from per-mesh AABBs for fast instancing / dynamic transforms.

### Lighting
(Inferred from shading code)
- **Directional**, **Point**, **Spot** lights
- Optional **light radius** for softer point/spot lighting
- Spot cone shaping via `innerCone` / `outerCone` and distance attenuation via `range`.

### Environment + Post
- **Sky gradient** (top/bottom) toggleable
- **HDRI environment map** loading (`stbi_loadf`) uploaded as a CUDA texture object
- Tone mapping helpers (Reinhard / Uncharted2 / ACES) + sRGB conversions
- Optional **bloom** pipeline (mip-style blur/compose)

### Denoiser (Built-in)
A custom **spatiotemporal denoiser**:
- Temporal accumulation with history buffers (mean / M2 variance) and reprojection
- **Motion vectors** computed from current and previous view-projection matrices
- Spatial **A-Trous wavelet filter** guided by normal/depth/objectID edges
- Supports either:
  - single-color denoise (`noisyColor`), or
  - split-channel denoise (`diffuseColor`, `specularColor`, `emissionColor`)

---

## Requirements

- **NVIDIA GPU** + **CUDA Toolkit**
  - uses `cuda_runtime.h`, `device_launch_parameters.h`, `curand_kernel.h`
- **C++17** (recommended)
- **stb_image** (or compatible) for HDRI loading (`stbi_loadf`, `stbi_image_free`)
- Your project also needs the “missing” headers referenced below (common math, light/material definitions, BVH traversal kernels).

> Output format: the renderer writes **RGB8** into an `unsigned char*` buffer (3 bytes per pixel). The included `saveAsPPM()` expects that layout.

---

## Project Layout (from includes)

These are the major headers in the path tracer:

### Rendering / Integration
- `pathtracer/rendering/path_logic.cuh`  
  Core shading + path tracing logic (`tracePath`, `tracePathSplit`), direct light sampling, BSDF sampling and PDFs, clamps, etc.
- `pathtracer/rendering/render_utils.cuh`  
  Tone mapping (incl. ACES), gamma/sRGB conversions, small render math utilities.
- `pathtracer/rendering/pbr_utils.cuh`  
  Fresnel/microfacet helpers (GGX distribution, geometry terms, Beer-Lambert absorption, etc).

### Scene / Geometry
- `pathtracer/scene/scene.cuh`  
  High-level `Scene` API, buffer management, kernel launches, accumulation, presets, HDRI, bloom, denoiser wiring.
- `pathtracer/scene/mesh.cuh`  
  `Mesh` container: OBJ parsing, vertex/triangle storage, BVH build/upload, transforms, AABBs.
- `pathtracer/scene/transform.cuh`  
  `AABB` + transform helpers (ray/AABB tests, matrices, device transform data).
- `pathtracer/scene/camera.cuh`  
  Camera model, ray generation helpers, stores view/proj matrices for motion vectors.
- `pathtracer/math/sampling.cuh`  
  RNG + sampling routines (blue-noise fetch, hemisphere/cone/GGX sampling, ONB creation).
- `pathtracer/rendering/denoiser.cuh` + `pathtracer/rendering/denoiser_kernels.cuh`  
  Denoiser implementation + motion vector kernel.

### Missing but Referenced (inferred responsibilities)
- `common/vec3.cuh`, `common/ray.cuh`, `common/mat4.cuh`, `common/matrix.cuh`, `common/bluenoise.cuh`  
  Vector/matrix math, ray type, blue-noise storage.
- `pathtracer/math/mathutils.cuh`, `pathtracer/math/intersection.cuh`, `pathtracer/math/pdf.cuh`  
  Misc math, ray/triangle + BVH traversal, PDF computations for BSDF sampling.
- `pathtracer/scene/lights.cuh`  
  `Light` struct + enums (`LIGHT_POINT`, `LIGHT_DIRECTIONAL`, `LIGHT_SPOT`) and parameters.
- `pathtracer/scene/material_lib.cuh`  
  `Material` (AoS) + `DeviceMaterials` (SoA) definitions.
- `pathtracer/scene/scene_kernels.cuh`  
  Actual render kernels that call into `tracePath*` and write G-buffers/accum buffers.

---

## Quick Start (Typical Flow)

### 1 Create a Scene
```cpp
#include "pathtracer/scene/scene.cuh"

int W = 1280, H = 720;
Scene scene(W, H);
scene.setPerformancePreset("balanced");   // "quality", "balanced", "performance", "fast"
