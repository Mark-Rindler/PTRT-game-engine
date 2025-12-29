// pdf.cuh
// probability density function evaluation for light sampling and bsdf sampling
// provides mis weights and pdf evaluators for cosine ggx reflection ggx
// refraction and light solid angle sampling

#ifndef PDF_CUH
#define PDF_CUH

#include "pathtracer/math/intersection.cuh"
#include "pathtracer/math/sampling.cuh"
#include "pathtracer/rendering/pbr_utils.cuh"
#include "pathtracer/scene/lights.cuh"
#include "pathtracer/scene/material_lib.cuh"
#include "pathtracer/scene/mesh.cuh"

// loads a vec3 through the readonly cache when available
__device__ __forceinline__ vec3 ldg_vec3_pdf(const vec3 *ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return vec3(__ldg(&ptr->x), __ldg(&ptr->y), __ldg(&ptr->z));
#else
    return *ptr;
#endif
}

// computes a multiple importance sampling weight for two competing pdf values
__device__ __forceinline__ float mis_weight(float pdf1, float pdf2) {
    float pdf1_sq = pdf1 * pdf1;
    float pdf2_sq = pdf2 * pdf2;
    return pdf1_sq / (pdf1_sq + pdf2_sq + 1e-10f);
}

// computes the solid angle pdf for sampling a specific light from a point
__device__ __forceinline__ float
light_pdf(const HitInfo &hit, const vec3 &V, int mat_id,
          const DeviceMaterials &materials, const vec3 &L, DeviceMesh *meshes,
          int nMeshes, DeviceBVHNode *tlasNodes, int *tlasMeshIndices,
          Light *lights, int nLights, cudaTextureObject_t envMap) {
    if (nLights == 0)
        return 0.0f;

    float total_pdf = 0.0f;
    float prob_pick_one_light = 1.0f / (float)nLights;

    for (int i = 0; i < nLights; ++i) {
        const Light &light = lights[i];

        if (light.type == LIGHT_DIRECTIONAL) {
            continue; // Delta light
        }

        vec3 toLight = light.position - hit.point;
        float light_dist_sq = toLight.length_squared();
        vec3 light_dir = normalize(toLight);

        if (light.radius <= 0.0f) {
            continue; // Point light (Delta)
        }

        float cos_theta_max = sqrtf(
            fmaxf(0.f, 1.f - (light.radius * light.radius) / light_dist_sq));
        float pdf_solid_angle =
            1.0f / (TWO_PI * (1.0f - cos_theta_max) + 1e-6f);

        if (dot(L, light_dir) > cos_theta_max) {
            total_pdf += prob_pick_one_light * pdf_solid_angle;
        }
    }

    return total_pdf;
}

// computes the cosine hemisphere pdf for a given normal and direction
__device__ __forceinline__ float pdf_cosine_hemisphere(const vec3 &N,
                                                       const vec3 &L) {
    float NdotL = fmaxf(dot(N, L), 0.0f);
    return NdotL * (1.0f / PI);
}

// computes the ggx microfacet reflection pdf for a sampled outgoing direction
__device__ __forceinline__ float
pdf_ggx_reflect(const vec3 &N, const vec3 &V, const vec3 &L, float roughness) {
    float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotV == 0.0f)
        return 0.0f;

    vec3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);

    float D = distributionGGX(N, H, roughness);

    float pdf_H = D * NdotH;
    return pdf_H / (4.0f * VdotH + 1e-6f);
}

// computes the ggx microfacet transmission pdf for a sampled outgoing direction
__device__ __forceinline__ float pdf_ggx_refract(const vec3 &N, const vec3 &V,
                                                 const vec3 &L, float roughness,
                                                 float ior_ratio) {
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotL = dot(N, L);

    // For refraction, L should be on opposite side of surface (NdotL < 0)
    if (NdotV <= 0.0f || NdotL >= 0.0f)
        return 0.0f;

    float NdotL_abs = fabsf(NdotL);

    float eta = ior_ratio;
    vec3 H = normalize(-(V * eta + L));
    if (dot(N, H) < 0.0f)
        H = -H;

    float VdotH = fmaxf(dot(V, H), 0.0f);
    float LdotH = fabsf(dot(L, H)); // Use absolute value
    float NdotH = fmaxf(dot(N, H), 0.0f);

    float D = distributionGGX(N, H, roughness);

    float pdf_H = D * NdotH;
    float dwh_dwo = (eta * eta * LdotH) / powf(eta * VdotH + LdotH, 2.0f);
    return pdf_H * fabsf(dwh_dwo);
}

// computes the overall bsdf pdf used by the path tracer for the chosen
// scattering event
__device__ __forceinline__ float material_pdf(const HitInfo &hit, int mat_id,
                                              const DeviceMaterials &materials,
                                              const vec3 &V, const vec3 &L) {

    const vec3 N = hit.normal;
    const float NdotV = fmaxf(dot(N, V), 0.0f);
    const float NdotL = fmaxf(dot(N, L), 0.0f);

    if (NdotV == 0.0f)
        return 0.0f;

    const float metal = clamp01(__ldg(&materials.metallic[mat_id]));
    const float rough = fmaxf(__ldg(&materials.roughness[mat_id]), 0.02f);
    const float trans = clamp01(__ldg(&materials.transmission[mat_id]));
    const vec3 albedo = ldg_vec3_pdf(&materials.albedo[mat_id]);
    const vec3 specular = ldg_vec3_pdf(&materials.specular[mat_id]);

    vec3 F0_base = lerp(specular, albedo, metal);

    const float iridescence = clamp01(__ldg(&materials.iridescence[mat_id]));
    if (iridescence > 0.0f) {
        const float thickness = __ldg(&materials.iridescenceThickness[mat_id]);
        const float ior = __ldg(&materials.ior[mat_id]);
        vec3 iridescence_color =
            calculateIridescence(thickness, NdotV, 1.3f, ior);
        F0_base = lerp(F0_base, iridescence_color, iridescence);
    }

    vec3 F_base = fresnelSchlick(NdotV, F0_base);

    float total_pdf = 0.0f;
    float prob_base = 1.0f;

    const float clearcoat = clamp01(__ldg(&materials.clearcoat[mat_id]));
    if (clearcoat > 0.0f) {
        const float clearcoatRough =
            fmaxf(__ldg(&materials.clearcoatRoughness[mat_id]), 0.001f);
        const vec3 F0_coat = vec3(0.04f);
        const vec3 F_coat = fresnelSchlick(NdotV, F0_coat);
        float F_coat_avg = (F_coat.x + F_coat.y + F_coat.z) * (1.0f / 3.0f);
        float P_coat = clamp01(F_coat_avg * clearcoat);

        if (NdotL > 0.0f) {
            total_pdf += P_coat * pdf_ggx_reflect(N, V, L, clearcoatRough);
        }

        prob_base = (1.0f - P_coat);
    }

    if (trans > 0.0f && metal < 0.1f) {
        const float ior = __ldg(&materials.ior[mat_id]);
        const float transRough =
            fmaxf(__ldg(&materials.transmissionRoughness[mat_id]), rough);
        float ior_ratio = hit.front_face ? (1.0f / ior) : ior;
        float reflect_prob = schlick_dielectric_oneIOR(NdotV, ior_ratio);

        if (NdotL > 0.0f) {
            // Reflection side
            float pdf_reflect = pdf_ggx_reflect(N, V, L, rough);
            total_pdf += prob_base * reflect_prob * pdf_reflect;

            // Check for TIR case (refraction that became reflection)
            vec3 H = normalize(V + L);
            float VdotH = fmaxf(dot(V, H), 0.0f);
            float k = 1.0f - ior_ratio * ior_ratio * (1.0f - VdotH * VdotH);
            if (k < 0.0f) {
                // TIR: refraction samples ended up as reflection
                float pdf_refract_as_reflect =
                    pdf_ggx_reflect(N, V, L, transRough);
                total_pdf +=
                    prob_base * (1.0f - reflect_prob) * pdf_refract_as_reflect;
            }
        } else {
            // Refraction side (NdotL < 0)
            float pdf_refract = pdf_ggx_refract(N, V, L, transRough, ior_ratio);
            total_pdf += prob_base * (1.0f - reflect_prob) * pdf_refract;
        }

        return total_pdf;
    }

    if (NdotL > 0.0f) {
        float max_fresnel = fmaxf(F_base.x, fmaxf(F_base.y, F_base.z));
        float specular_prob = (metal > 0.0f) ? 1.0f : max_fresnel;

        float pdf_spec = pdf_ggx_reflect(N, V, L, rough);
        float pdf_diffuse = pdf_cosine_hemisphere(N, L);

        total_pdf += prob_base * (specular_prob * pdf_spec +
                                  (1.0f - specular_prob) * pdf_diffuse);
    }

    return total_pdf;
}

#endif // PDF_CUH
