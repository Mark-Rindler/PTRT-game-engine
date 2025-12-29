// material lib cuh
// material definitions and helpers for pbr shading
// contains host side material struct and device side soa material buffers
// provides small conversion helpers used during shading
#ifndef MATERIAL_LIB_CUH
#define MATERIAL_LIB_CUH

#include "common/vec3.cuh"
#include "pathtracer/math/mathutils.cuh"
#include "pathtracer/rendering/render_utils.cuh"

struct Material {

    vec3 albedo;
    vec3 specular;
    float metallic;
    float roughness;
    vec3 emission;

    float ior;
    float transmission;
    float transmissionRoughness;

    float clearcoat;
    float clearcoatRoughness;

    vec3 subsurfaceColor;
    float subsurfaceRadius;

    float anisotropy;
    float sheen;
    vec3 sheenTint;

    float iridescence;
    float iridescenceThickness;

    // material
    // implements a unit of behavior used by higher level scene code
    // inputs none
    // returns device

    __host__ __device__ Material()
        // albedo
        // implements a unit of behavior used by higher level scene code
        // inputs vec3 0 8f specular vec3 0 04f metallic 0 0f
        // returns value

        : albedo(vec3(0.8f)), specular(vec3(0.04f)), metallic(0.0f),
          // roughness
          // implements a unit of behavior used by higher level scene code
          // inputs 0 5f emission vec3 0 0f ior 1 5f transmission 0 0f
          // returns value

          roughness(0.5f), emission(vec3(0.0f)), ior(1.5f), transmission(0.0f),
          // transmissionroughness
          // implements a unit of behavior used by higher level scene code
          // inputs 0 0f clearcoat 0 0f
          // returns value

          transmissionRoughness(0.0f), clearcoat(0.0f),
          // clearcoatroughness
          // implements a unit of behavior used by higher level scene code
          // inputs 0 03f subsurfacecolor vec3 1 0f
          // returns value

          clearcoatRoughness(0.03f), subsurfaceColor(vec3(1.0f)),
          // subsurfaceradius
          // implements a unit of behavior used by higher level scene code
          // inputs 0 0f anisotropy 0 0f sheen 0 0f
          // returns value

          subsurfaceRadius(0.0f), anisotropy(0.0f), sheen(0.0f),
          // sheentint
          // implements a unit of behavior used by higher level scene code
          // inputs vec3 0 5f iridescence 0 0f
          // returns value

          sheenTint(vec3(0.5f)), iridescence(0.0f),
          // iridescencethickness
          // implements a unit of behavior used by higher level scene code
          // inputs 550 0f
          // returns value

          iridescenceThickness(550.0f) {}

    // material
    // implements a unit of behavior used by higher level scene code
    // inputs alb rough met
    // returns device

    __host__ __device__ Material(const vec3 &alb, float rough = 0.5f,
                                 float met = 0.0f)
        // material
        // implements a unit of behavior used by higher level scene code
        // inputs none
        // returns value

        : Material() {
        albedo = alb;
        roughness = rough;
        metallic = met;
        specular = lerp(vec3(0.04f), albedo, metallic);
        transmissionRoughness = fmaxf(transmissionRoughness, roughness);
    }
};

struct DeviceMaterials {
    vec3 *albedo;
    vec3 *specular;
    float *metallic;
    float *roughness;
    vec3 *emission;
    float *ior;
    float *transmission;
    float *transmissionRoughness;
    float *clearcoat;
    float *clearcoatRoughness;
    vec3 *subsurfaceColor;
    float *subsurfaceRadius;
    float *anisotropy;
    float *sheen;
    vec3 *sheenTint;
    float *iridescence;
    float *iridescenceThickness;
};

// phongshininesstoroughness
// implements a unit of behavior used by higher level scene code
// inputs n
// returns float

__host__ __device__ inline float phongShininessToRoughness(float n) {
    float alpha = sqrtf(2.0f / (fmaxf(n, 1.0f) + 2.0f));
    return clamp01(fmaxf(alpha, 0.02f));
}

// iortof0
// implements a unit of behavior used by higher level scene code
// inputs ior
// returns float

__host__ __device__ inline float iorToF0(float ior) {
    float a = (ior - 1.0f) / (ior + 1.0f);
    return a * a;
}

#endif