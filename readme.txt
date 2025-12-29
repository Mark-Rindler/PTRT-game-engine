******IF YOU WISH TO COMPILE
This project uses a makefile to compile. It currently requires users to be on windows, have visual studios 2022 installed, CUDA toolkit installed,
and an NVIDIA GPU that matches the sm_86 architecture (30 series NVIDIA GPU). If you do not have an NVIDIA GPU, you cannot run this program because CUDA
is NVIDIA only, but if you have an NVIDIA GPU without any of the afformentioned requirements all you need to do is either install them or refactor 
the Makefile.

NOTE: If your project does not compile, it is likely an issue with NRD. These project files do not integrate NRD so you can safely delete any mention of
NRD from the project files, and delete the source code in the library folder.


# CUDA Rendering Suite (Ray Tracer + Path Tracer)

A single codebase that contains **two GPU renderers**:

- **CUDA Path Tracer (PT)**: physically based renderer with **progressive accumulation**, **spatiotemporal denoising**, optional **bloom**, and **HDRI** support
- **CUDA Ray Tracer (RT)**: fast, renderer, not as comprehensive as the path tracer but still functional 

Both renderers share a common math / geometry foundation (`common/`) and can be driven through a shared **Unified Scene** layer (`common/PTRTtransfer.cuh`) so you can swap backends without rewriting your scene construction code.
Both renderes have their respective readmes, for more detailed information on either you can view those.

---

## What’s in here

### Path Tracer (PT)
- CUDA path tracing integrator (multi-bounce, progressive)
- Next-event estimation (direct light sampling) + BSDF sampling
- Split-path output (diffuse / specular / emission) for denoising
- Scene-level optimizations (TLAS over meshes, material SoA)
- **Spatiotemporal denoiser** (motion vectors + history + A-Trous edge-aware filter)
- Optional bloom and HDRI environment map

### Ray Tracer (RT)
- CUDA kernel renderer writing an **8-bit RGB** buffer
- CPU-built BVH, GPU traversal (iterative stack traversal)
- OBJ mesh loading + transforms
- PBR-ish material set (metal/rough + extras)
- Lights: point / directional / spot (+ ambient)
- Sky gradient
- In-kernel tonemap + gamma

### Shared (Common)
- CUDA-friendly math types: `vec3`, `vec4`, `mat4`, `matrix`
- Ray + triangle helpers
- Blue-noise sampling support
- Optional CUDA/OpenGL interop viewer (`glfw_view_interop.hpp`) using a mapped PBO (no CPU copy)

---

## Repository layout (high level)

> Exact folder names below match how headers include each other (some headers use Windows-style slashes).

### `common/`
- `vec3.cuh`, `vec4.cuh` — vector math (host+device)
- `ray.cuh` — ray type
- `triangle.cuh` — triangle intersection helpers
- `mat4.cuh`, `matrix.cuh` — transforms and matrix math
- `bluenoise.cuh` — blue-noise table + sampling helpers
- `glfw_view_interop.hpp` — GLFW + OpenGL + CUDA PBO interop viewer
- `PTRTtransfer.cuh` — **Unified Scene wrapper** that can build either RT or PT scenes

### `raytracer/`
- `RTscene.cuh` — Scene container, shading + intersection, CUDA `render_kernel`
- `RTmesh.cuh` — OBJ parser, CPU BVH build, device uploads
- `RTcamera.cuh` — ray generation + DOF sampling
- `RTmathutils.cuh` — CUDA math/random helpers
- `RTapp_utils.cuh` — CLI + demo scenes + GLFW/OpenGL glue (interactive app path)

### `pathtracer/`
- `scene/scene.cuh` — PT scene API, accumulation, TLAS/material uploads, render entry points
- `scene/mesh.cuh`, `scene/transform.cuh`, `scene/camera.cuh`
- `math/sampling.cuh` — RNG + sampling utilities (blue-noise, GGX, hemispheres, etc.)
- `rendering/path_logic.cuh` — integrator + BSDF/light sampling
- `rendering/pbr_utils.cuh`, `rendering/render_utils.cuh`
- `rendering/denoiser.cuh`, `rendering/denoiser_kernels.cuh`
- `rendering/taa.cuh`

---

## The “Unified Scene” bridge (RT ↔ PT)

`common/PTRTtransfer.cuh` defines a renderer-agnostic scene model:
- `UnifiedScene` (camera, meshes, lights, sky, material library, presets)
- Handle-based editing (`ObjectHandle`, `LightHandle`)
- Keyframe animation helpers (time + easing)
- Builder functions that convert `UnifiedScene` into either backend

### Choosing a backend at compile time
You **must** define exactly one of:
- `UNIFIED_SCENE_ENABLE_RT`
- `UNIFIED_SCENE_ENABLE_PT`

Example:
```cpp
#define UNIFIED_SCENE_ENABLE_PT
#include "common/PTRTtransfer.cuh"


## Example Games

A separate repository hosted by Mark Rindler includes **multiple example games** built on top of the rendering framework (ray tracer / path tracer).

Each example is fully self-contained and demonstrates how to use the engine for an interactive or game-like workload.

### Running an example game (These are in a separate github)

1. Navigate to the `test games/` directory.
2. Choose one of the three example game folders.
3. Copy **all files** from that game folder.
4. Paste them into the `src/` directory.
5. Compile the project normally.

No additional configuration is required.  
Each game is designed to build directly against the existing renderer and shared infrastructure.

Gameplay and screenshots of all three games are in Test game screenshots/

> Only one example game should be placed in `src/` at a time.