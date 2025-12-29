/*
this file contains core path tracing shading logic used by the renderer
it defines helper functions for loading data evaluating bsdf terms and sampling
lights it implements light selection attenuation and spotlight cone shaping it
provides routines for computing direct lighting contributions and pdf values it
assumes inputs such as materials lights and hit records are provided by the
caller it uses cuda device functions and may use ldg loads for read only data
when available it is intended to be included in cuda translation units that
implement ray traversal and integration loops it relies on consistent
conventions for normals directions and linear depth measured along camera rays
*/

#ifndef PATH_LOGIC_CUH
#define PATH_LOGIC_CUH

#include "pathtracer/math/intersection.cuh"
#include "pathtracer/math/pdf.cuh"
#include "pathtracer/math/sampling.cuh"
#include "pathtracer/rendering/pbr_utils.cuh"
#include "pathtracer/rendering/render_utils.cuh"
#include "pathtracer/scene/lights.cuh"
#include "pathtracer/scene/material_lib.cuh"

constexpr int RUSSIAN_ROULETTE_START_BOUNCE = 2;
constexpr float RUSSIAN_ROULETTE_MIN_PROB = 0.05f;

constexpr float MAX_BOUNCE_WEIGHT = 50.0f;
constexpr float MAX_NEE_CONTRIBUTION = 500.0f;
constexpr float MAX_FINAL_RADIANCE = 100.0f;

constexpr float SIMPLE_MATERIAL_THRESHOLD = 0.01f;

__device__ __forceinline__ float geometrySmithTransmission(const vec3 &N,
                                                           const vec3 &V,
                                                           const vec3 &L,
                                                           float roughness) {
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotL = fabsf(dot(N, L));
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

__device__ __forceinline__ vec3 clamp_vector_soft(const vec3 &v,
                                                  float max_lum) {
    float lum = 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
    if (lum > max_lum && lum > 0.0f) {
        float scale = max_lum / lum;
        return v * scale;
    }
    return v;
}

__device__ __forceinline__ vec3 clamp_vector(const vec3 &v, float max_len) {
    float len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq > max_len * max_len) {
        float scale = max_len * rsqrtf(len_sq);
        return v * scale;
    }
    return v;
}

struct SplitPathOutput {
    vec3 diffuse;
    vec3 specular;
    vec3 emission;
};

__device__ __forceinline__ vec3 ldg_vec3(const vec3 *ptr) {
    return vec3(__ldg(&ptr->x), __ldg(&ptr->y), __ldg(&ptr->z));
}

struct MaterialProps {
    vec3 albedo;
    vec3 specular;
    vec3 emission;
    float metallic;
    float roughness;
    float transmission;
    float ior;
    float transmissionRoughness;
    float clearcoat;
    float clearcoatRoughness;
    float iridescence;
    float iridescenceThickness;
    float sheen;
    vec3 sheenTint;

    __device__ __forceinline__ void load(const DeviceMaterials &materials,
                                         int mat_id) {
        albedo = ldg_vec3(&materials.albedo[mat_id]);
        specular = ldg_vec3(&materials.specular[mat_id]);
        emission = ldg_vec3(&materials.emission[mat_id]);
        metallic = __ldg(&materials.metallic[mat_id]);
        roughness = __ldg(&materials.roughness[mat_id]);
        transmission = __ldg(&materials.transmission[mat_id]);
        ior = __ldg(&materials.ior[mat_id]);
        transmissionRoughness = __ldg(&materials.transmissionRoughness[mat_id]);
        clearcoat = __ldg(&materials.clearcoat[mat_id]);
        clearcoatRoughness = __ldg(&materials.clearcoatRoughness[mat_id]);
        iridescence = __ldg(&materials.iridescence[mat_id]);
        iridescenceThickness = __ldg(&materials.iridescenceThickness[mat_id]);
        sheen = __ldg(&materials.sheen[mat_id]);
        sheenTint = ldg_vec3(&materials.sheenTint[mat_id]);
    }

    __device__ __forceinline__ bool isSimple() const {
        return (transmission < SIMPLE_MATERIAL_THRESHOLD &&
                clearcoat < SIMPLE_MATERIAL_THRESHOLD &&
                iridescence < SIMPLE_MATERIAL_THRESHOLD &&
                metallic < SIMPLE_MATERIAL_THRESHOLD);
    }

    __device__ __forceinline__ bool isEmissive() const {
        return (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f);
    }

    __device__ __forceinline__ float emissionLuminance() const {
        return 0.2126f * emission.x + 0.7152f * emission.y +
               0.0722f * emission.z;
    }
};

__device__ __forceinline__ vec3 evaluateBSDF_fast(const HitInfo &hit,
                                                  const MaterialProps &mat,
                                                  const vec3 &L,
                                                  const vec3 &V) {
    const vec3 N = hit.normal;
    const float NdotL = fmaxf(dot(N, L), 0.0f);
    const float NdotV = fmaxf(dot(N, V), 0.0f);

    if (NdotL <= 0.0f || NdotV <= 0.0f)
        return vec3(0.0f);

    vec3 H = normalize(L + V);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);

    float rough = fmaxf(mat.roughness, 0.02f);
    float a = rough * rough;
    float a2 = a * a;

    float D = distributionGGX_fast(NdotH, a2);
    float G = geometrySmith_fast(NdotV, NdotL, rough);

    vec3 F0 = mat.specular;
    vec3 F = fresnelSchlick(VdotH, F0);

    vec3 spec = (D * G * F) / (4.0f * NdotV * NdotL + 0.001f);

    vec3 kD = (vec3(1.0f) - F) * (1.0f - mat.metallic);
    vec3 diffuse = kD * mat.albedo * (1.0f / PI);

    return (diffuse + spec) * NdotL;
}

__device__ __forceinline__ vec3 evaluateBSDF(const HitInfo &hit,
                                             const MaterialProps &mat,
                                             const vec3 &L, const vec3 &V) {
    const vec3 N = hit.normal;
    const float NdotV = fmaxf(dot(N, V), 0.0f);

    if (NdotV <= 0.0f)
        return vec3(0.0f);

    const float metal = clamp01(mat.metallic);
    const float rough = fmaxf(mat.roughness, 0.02f);
    const float trans = clamp01(mat.transmission);
    const vec3 albedo = mat.albedo;
    vec3 specular = mat.specular;

    vec3 F0_base = lerp(specular, albedo, metal);

    const float iridescence = clamp01(mat.iridescence);
    if (iridescence > 0.0f) {
        const float thickness = mat.iridescenceThickness;
        const float ior = mat.ior;
        vec3 iridescence_color =
            calculateIridescence(thickness, NdotV, 1.3f, ior);
        F0_base = lerp(F0_base, iridescence_color, iridescence);
    }

    if (trans > 0.0f && metal < 0.1f) {
        const float ior = mat.ior;
        const float transRough = fmaxf(mat.transmissionRoughness, rough);
        float ior_ratio = hit.front_face ? (1.0f / ior) : ior;
        float NdotL = dot(N, L);

        if (NdotL > 0.0f) {
            vec3 H = normalize(L + V);
            float NdotH = fmaxf(dot(N, H), 0.0f);
            float VdotH = fmaxf(dot(V, H), 0.0f);

            float D = distributionGGX(N, H, rough);
            float G = geometrySmith(N, V, L, rough);
            vec3 F = fresnelSchlick(VdotH, F0_base);

            vec3 spec = (D * G * F) / (4.0f * NdotV * NdotL + 1e-6f);
            return spec * NdotL;
        } else {
            float eta = ior_ratio;
            vec3 H = normalize(-(V * eta + L));
            if (dot(N, H) < 0.0f)
                H = -H;

            float VdotH = fmaxf(dot(V, H), 0.0f);
            float LdotH = fabsf(dot(L, H));
            float NdotH = fmaxf(dot(N, H), 0.0f);
            float NdotL_abs = fabsf(NdotL);

            float k = 1.0f - eta * eta * (1.0f - VdotH * VdotH);
            if (k < 0.0f)
                return vec3(0.0f);

            float D = distributionGGX(N, H, transRough);
            float G = geometrySmithTransmission(N, V, L, transRough);

            vec3 F_fresnel = fresnelSchlick(VdotH, F0_base);
            vec3 F = vec3(1.0f) - F_fresnel;

            float numerator =
                (eta * eta * (1.0f - metal) * G * D * VdotH * LdotH);
            float denominator =
                NdotV * NdotL_abs * powf(eta * VdotH + LdotH, 2.0f);
            vec3 btdf = (albedo * F * numerator) / (denominator + 1e-6f);

            return btdf * NdotL_abs;
        }
    }

    float NdotL = fmaxf(dot(N, L), 0.0f);
    if (NdotL <= 0.0f)
        return vec3(0.0f);

    vec3 H = normalize(L + V);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);

    float D = distributionGGX(N, H, rough);
    float G = geometrySmith(N, V, L, rough);
    vec3 F = fresnelSchlick(VdotH, F0_base);

    specular = (D * G * F) / (4.0f * NdotV * NdotL + 0.001f);

    vec3 kS = F;
    vec3 kD = (vec3(1.0f) - kS) * (1.0f - metal);
    vec3 diffuse = kD * albedo / PI;

    return (diffuse + specular) * NdotL;
}

__device__ __forceinline__ void
evaluateBSDF_split(const HitInfo &hit, const MaterialProps &mat, const vec3 &L,
                   const vec3 &V, vec3 &out_diffuse, vec3 &out_specular) {
    out_diffuse = vec3(0.0f);
    out_specular = vec3(0.0f);

    const vec3 N = hit.normal;
    const float NdotV = fmaxf(dot(N, V), 0.0f);

    if (NdotV <= 0.0f)
        return;

    const float metal = clamp01(mat.metallic);
    const float rough = fmaxf(mat.roughness, 0.02f);
    const float trans = clamp01(mat.transmission);
    const vec3 albedo = mat.albedo;
    vec3 specular_color = mat.specular;

    vec3 F0_base = lerp(specular_color, albedo, metal);

    const float iridescence = clamp01(mat.iridescence);
    if (iridescence > 0.0f) {
        const float thickness = mat.iridescenceThickness;
        const float ior = mat.ior;
        vec3 iridescence_color =
            calculateIridescence(thickness, NdotV, 1.3f, ior);
        F0_base = lerp(F0_base, iridescence_color, iridescence);
    }

    if (trans > 0.0f && metal < 0.1f) {
        out_specular = evaluateBSDF(hit, mat, L, V);
        return;
    }

    float NdotL = fmaxf(dot(N, L), 0.0f);
    if (NdotL <= 0.0f)
        return;

    vec3 H = normalize(L + V);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);

    float D = distributionGGX(N, H, rough);
    float G = geometrySmith(N, V, L, rough);
    vec3 F = fresnelSchlick(VdotH, F0_base);

    out_specular = (D * G * F) / (4.0f * NdotV * NdotL + 0.001f) * NdotL;

    vec3 kS = F;
    vec3 kD = (vec3(1.0f) - kS) * (1.0f - metal);
    out_diffuse = (kD * albedo / PI) * NdotL;
}

__device__ __forceinline__ vec3 sample_direct_lighting_with_mat(
    const HitInfo &hit, const MaterialProps &mat,
    const DeviceMaterials &materials, const Ray &ray_in, DeviceMesh *meshes,
    int nMeshes, DeviceBVHNode *tlasNodes, int *tlasMeshIndices, Light *lights,
    int nLights, curandState *state, vec3 &out_L, float &out_pdf) {

    if (nLights == 0) {
        out_L = vec3(0.0f);
        out_pdf = 0.0f;
        return vec3(0.0f);
    }

    vec3 direct_light(0.0f);
    const vec3 V = -ray_in.direction();

    float r = curand_uniform(state);
    r = fminf(r, 0.99999994f);
    int light_index = (int)(r * nLights);
    const Light &light = lights[light_index];

    float pdf_pick = 1.0f / (float)nLights;

    vec3 L;
    float attenuation = 1.0f;
    float light_dist = 1e30f;
    vec3 light_radiance = light.color * light.intensity;
    float pdf_sample = 1.0f;

    if (light.type == LIGHT_DIRECTIONAL) {
        L = -light.direction;
        pdf_sample = pdf_pick;
    } else {
        vec3 toLight = light.position - hit.point;
        float light_dist_sq = toLight.length_squared();
        light_dist = sqrtf(light_dist_sq);

        if (light.radius <= 0.0f) {
            L = toLight / light_dist;
            pdf_sample = pdf_pick;
        } else {
            float sin_theta_max_sq =
                (light.radius * light.radius) / light_dist_sq;
            sin_theta_max_sq = fminf(sin_theta_max_sq, 0.9999f);
            float cos_theta_max = sqrtf(1.0f - sin_theta_max_sq);

            L = sample_cone_direction(state, toLight / light_dist,
                                      cos_theta_max);

            float solid_angle = TWO_PI * (1.0f - cos_theta_max);
            pdf_sample =
                (solid_angle > 1e-6f) ? (pdf_pick / solid_angle) : pdf_pick;
        }

        attenuation = attenuate(light_dist, light.range);

        if (light.type == LIGHT_SPOT) {
            float theta = dot(L, -light.direction);
            float epsilon = light.innerCone - light.outerCone;
            float spotIntensity;
            if (epsilon <= 1e-6f) {
                spotIntensity = (theta >= light.outerCone) ? 1.0f : 0.0f;
            } else {
                spotIntensity =
                    clamp((theta - light.outerCone) / epsilon, 0.0f, 1.0f);
            }
            attenuation *= spotIntensity;
        }
    }

    out_L = L;
    out_pdf = pdf_sample;

    vec3 shadow_offset =
        dot(hit.normal, L) > 0.0f ? hit.normal * 1e-4f : -hit.normal * 1e-4f;
    Ray shadowRay(hit.point + shadow_offset, L);

    bool inShadow = bvh_any_hit_tlas(shadowRay, light_dist - 1e-3f, meshes,
                                     tlasNodes, tlasMeshIndices, materials);

    if (!inShadow) {
        vec3 bsdf = evaluateBSDF(hit, mat, L, V);
        if (pdf_sample > 0.0f) {
            direct_light = bsdf * light_radiance * attenuation / pdf_sample;
            direct_light =
                clamp_vector_soft(direct_light, MAX_NEE_CONTRIBUTION);
        }
    }
    return direct_light;
}

__device__ __forceinline__ void sample_direct_lighting_split(
    const HitInfo &hit, const MaterialProps &mat,
    const DeviceMaterials &materials, const Ray &ray_in, DeviceMesh *meshes,
    int nMeshes, DeviceBVHNode *tlasNodes, int *tlasMeshIndices, Light *lights,
    int nLights, curandState *state, vec3 &out_L, float &out_pdf,
    vec3 &out_diffuse, vec3 &out_specular) {

    out_diffuse = vec3(0.0f);
    out_specular = vec3(0.0f);

    if (nLights == 0) {
        out_L = vec3(0.0f);
        out_pdf = 0.0f;
        return;
    }

    const vec3 V = -ray_in.direction();

    float r = curand_uniform(state);
    r = fminf(r, 0.99999994f);
    int light_index = (int)(r * nLights);
    const Light &light = lights[light_index];

    float pdf_pick = 1.0f / (float)nLights;

    vec3 L;
    float attenuation = 1.0f;
    float light_dist = 1e30f;
    vec3 light_radiance = light.color * light.intensity;
    float pdf_sample = 1.0f;

    if (light.type == LIGHT_DIRECTIONAL) {
        L = -light.direction;
        pdf_sample = pdf_pick;
    } else {
        vec3 toLight = light.position - hit.point;
        float light_dist_sq = toLight.length_squared();
        light_dist = sqrtf(light_dist_sq);

        if (light.radius <= 0.0f) {
            L = toLight / light_dist;
            pdf_sample = pdf_pick;
        } else {
            float sin_theta_max_sq =
                (light.radius * light.radius) / light_dist_sq;
            sin_theta_max_sq = fminf(sin_theta_max_sq, 0.9999f);
            float cos_theta_max = sqrtf(1.0f - sin_theta_max_sq);

            L = sample_cone_direction(state, toLight / light_dist,
                                      cos_theta_max);

            float solid_angle = TWO_PI * (1.0f - cos_theta_max);
            pdf_sample =
                (solid_angle > 1e-6f) ? (pdf_pick / solid_angle) : pdf_pick;
        }

        attenuation = attenuate(light_dist, light.range);

        if (light.type == LIGHT_SPOT) {
            float theta = dot(L, -light.direction);
            float epsilon = light.innerCone - light.outerCone;
            float spotIntensity;
            if (epsilon <= 1e-6f) {
                spotIntensity = (theta >= light.outerCone) ? 1.0f : 0.0f;
            } else {
                spotIntensity =
                    clamp((theta - light.outerCone) / epsilon, 0.0f, 1.0f);
            }
            attenuation *= spotIntensity;
        }
    }

    out_L = L;
    out_pdf = pdf_sample;

    vec3 shadow_offset =
        dot(hit.normal, L) > 0.0f ? hit.normal * 1e-4f : -hit.normal * 1e-4f;
    Ray shadowRay(hit.point + shadow_offset, L);

    bool inShadow = bvh_any_hit_tlas(shadowRay, light_dist - 1e-3f, meshes,
                                     tlasNodes, tlasMeshIndices, materials);

    if (!inShadow && pdf_sample > 0.0f) {
        vec3 bsdf_diffuse, bsdf_specular;
        evaluateBSDF_split(hit, mat, L, V, bsdf_diffuse, bsdf_specular);

        float scale = attenuation / pdf_sample;
        out_diffuse = bsdf_diffuse * light_radiance * scale;
        out_specular = bsdf_specular * light_radiance * scale;

        out_diffuse = clamp_vector_soft(out_diffuse, MAX_NEE_CONTRIBUTION);
        out_specular = clamp_vector_soft(out_specular, MAX_NEE_CONTRIBUTION);
    }
}

__device__ __forceinline__ bool
material_scatter(const HitInfo &hit, const MaterialProps &mat,
                 const Ray &ray_in, curandState *state, vec3 &scattered_dir,
                 vec3 &attenuation, bool &is_specular_bounce, float &out_pdf) {

    const vec3 V = -ray_in.direction();
    const vec3 N = hit.normal;
    const float NdotV = fmaxf(dot(N, V), 0.0f);

    const float metal = clamp01(mat.metallic);
    const float rough = fmaxf(mat.roughness, 0.02f);
    const float trans = clamp01(mat.transmission);
    const vec3 albedo = mat.albedo;
    const vec3 specular = mat.specular;

    vec3 F0_base = lerp(specular, albedo, metal);

    const float iridescence = clamp01(mat.iridescence);
    if (iridescence > 0.0f) {
        const float thickness = mat.iridescenceThickness;
        const float ior = mat.ior;
        vec3 iridescence_color =
            calculateIridescence(thickness, NdotV, 1.3f, ior);
        F0_base = lerp(F0_base, iridescence_color, iridescence);
    }

    vec3 F_base_for_NdotV = fresnelSchlick(NdotV, F0_base);

    const float clearcoat = clamp01(mat.clearcoat);
    float P_coat = 0.0f;
    float prob_base = 1.0f;
    float clearcoatRough = 0.0f;
    vec3 F0_coat = vec3(0.0f);

    if (clearcoat > 0.0f) {
        clearcoatRough = fmaxf(mat.clearcoatRoughness, 0.001f);
        F0_coat = vec3(0.04f);
        vec3 F_coat = fresnelSchlick(NdotV, F0_coat);
        float F_coat_avg = (F_coat.x + F_coat.y + F_coat.z) * (1.0f / 3.0f);
        P_coat = clamp01(F_coat_avg * clearcoat);
        prob_base = (1.0f - P_coat);
    }

    if (trans > 0.0f && metal < 0.1f) {
        const float ior = mat.ior;
        const float transRough = fmaxf(mat.transmissionRoughness, rough);
        float ior_ratio = hit.front_face ? (1.0f / ior) : ior;
        float ior_incident = hit.front_face ? 1.0f : ior;
        float ior_transmitted = hit.front_face ? ior : 1.0f;
        float reflect_prob =
            schlick_dielectric(NdotV, ior_incident, ior_transmitted);
        float refract_prob = 1.0f - reflect_prob;

        float P_trans_reflect = prob_base * reflect_prob;
        float P_trans_refract = prob_base * refract_prob;

        float u = curand_uniform(state);

        vec3 H;
        bool is_refraction = false;
        float sample_roughness;
        float eta = ior_ratio;

        if (u < P_coat) {
            sample_roughness = clearcoatRough;
            H = importance_sample_ggx(state, N, sample_roughness);
            scattered_dir = reflectVec(-V, H);
            is_specular_bounce = (sample_roughness < 0.1f);
        } else if (u < P_coat + P_trans_reflect) {
            sample_roughness = rough;
            H = importance_sample_ggx(state, N, sample_roughness);
            scattered_dir = reflectVec(-V, H);
            is_specular_bounce = (sample_roughness < 0.1f);
        } else {
            sample_roughness = transRough;
            H = importance_sample_ggx(state, N, sample_roughness);
            is_refraction = true;
            is_specular_bounce = (sample_roughness < 0.1f);

            float VdotH_tir = dot(V, H);
            float VdotH_abs = fabsf(VdotH_tir);
            if (VdotH_tir < 0.0f)
                H = -H; // Ensure H is in same hemisphere as V
            VdotH_tir = fabsf(dot(V, H));

            float k = 1.0f - eta * eta * (1.0f - VdotH_tir * VdotH_tir);
            if (k < 0.0f) {
                // Total internal reflection
                scattered_dir = reflectVec(-V, H);
                is_specular_bounce = true; // TIR is always specular
            } else {
                // Refraction using correct Snell's law
                float cos_t = sqrtf(k);
                scattered_dir =
                    normalize(eta * (-V) + (eta * VdotH_tir - cos_t) * H);
            }
        }

        float NdotL = dot(N, scattered_dir);
        vec3 f_total(0.0f);
        float pdf_total = 0.0f;

        vec3 F_coat_atten;
        if (is_refraction) {
            vec3 H_refr_base = normalize(eta * V + scattered_dir);
            float VdotH_refr_base = fmaxf(dot(V, H_refr_base), 0.0f);
            F_coat_atten = fresnelSchlick(VdotH_refr_base, F0_coat);
        } else {
            vec3 H_refl_base = normalize(V + scattered_dir);
            float VdotH_refl_base = fmaxf(dot(V, H_refl_base), 0.0f);
            F_coat_atten = fresnelSchlick(VdotH_refl_base, F0_coat);
        }
        vec3 base_attenuation = vec3(1.0f) - clearcoat * F_coat_atten;

        if (P_coat > 0.0f && NdotL > 0.0f) {
            vec3 H_coat = normalize(V + scattered_dir);
            float NdotH_coat = fmaxf(dot(N, H_coat), 0.0f);
            float VdotH_coat = fmaxf(dot(V, H_coat), 0.0f);
            float D_coat = distributionGGX(N, H_coat, clearcoatRough);
            float G_coat = geometrySmith(N, V, scattered_dir, clearcoatRough);
            vec3 F_coat = fresnelSchlick(VdotH_coat, F0_coat);

            float pdf_L_coat =
                (D_coat * NdotH_coat) / (4.0f * VdotH_coat + 1e-6f);
            pdf_total += P_coat * pdf_L_coat;

            vec3 brdf_coat =
                (D_coat * G_coat * F_coat) / (4.0f * NdotV * NdotL + 1e-6f);
            f_total += clearcoat * brdf_coat * NdotL;
        }

        if (P_trans_reflect > 0.0f && NdotL > 0.0f) {
            vec3 H_refl = normalize(V + scattered_dir);
            float NdotH_refl = fmaxf(dot(N, H_refl), 0.0f);
            float VdotH_refl = fmaxf(dot(V, H_refl), 0.0f);
            float D_refl = distributionGGX(N, H_refl, rough);
            float G_refl = geometrySmith(N, V, scattered_dir, rough);
            vec3 F_refl = fresnelSchlick(VdotH_refl, F0_base);

            float pdf_L_refl =
                (D_refl * NdotH_refl) / (4.0f * VdotH_refl + 1e-6f);
            pdf_total += P_trans_reflect * pdf_L_refl;

            vec3 brdf_refl =
                (D_refl * G_refl * F_refl) / (4.0f * NdotV * NdotL + 1e-6f);
            f_total += brdf_refl * NdotL * base_attenuation;
        }

        if (P_trans_refract > 0.0f && NdotL < 0.0f) {
            vec3 H_refr = normalize(-(V * eta + scattered_dir));
            if (dot(N, H_refr) < 0.0f)
                H_refr = -H_refr;

            float VdotH_refr = fmaxf(dot(V, H_refr), 0.0f);
            float LdotH_refr = fabsf(dot(scattered_dir, H_refr));
            float NdotH_refr = fmaxf(dot(N, H_refr), 0.0f);
            float NdotL_abs = fabsf(NdotL);

            float k = 1.0f - eta * eta * (1.0f - VdotH_refr * VdotH_refr);
            if (k >= 0.0f) {
                float D_refr = distributionGGX(N, H_refr, transRough);
                float G_refr =
                    geometrySmithTransmission(N, V, scattered_dir, transRough);

                float dwh_dwo = (eta * eta * LdotH_refr) /
                                powf(eta * VdotH_refr + LdotH_refr, 2.0f);
                float pdf_L_refr = (D_refr * NdotH_refr * fabsf(dwh_dwo));
                pdf_total += P_trans_refract * pdf_L_refr;

                vec3 F_refr = vec3(1.0f) - fresnelSchlick(VdotH_refr, F0_base);
                float numerator = (eta * eta * (1.0f - metal) * G_refr *
                                   D_refr * VdotH_refr * LdotH_refr);
                float denominator = NdotV * NdotL_abs *
                                    powf(eta * VdotH_refr + LdotH_refr, 2.0f);
                vec3 btdf =
                    (albedo * F_refr * numerator) / (denominator + 1e-6f);

                f_total += btdf * NdotL_abs * base_attenuation;
            }
        }

        if (is_refraction && NdotL > 0.0f) {
            vec3 H_refl = normalize(V + scattered_dir);
            float NdotH_refl = fmaxf(dot(N, H_refl), 0.0f);
            float VdotH_refl = fmaxf(dot(V, H_refl), 0.0f);
            float D_refl = distributionGGX(N, H_refl, transRough);
            float G_refl = geometrySmith(N, V, scattered_dir, transRough);

            float pdf_L_refl =
                (D_refl * NdotH_refl) / (4.0f * VdotH_refl + 1e-6f);
            pdf_total += P_trans_refract * pdf_L_refl;

            vec3 brdf_refl =
                (D_refl * G_refl * vec3(1.0f)) / (4.0f * NdotV * NdotL + 1e-6f);
            f_total += brdf_refl * NdotL * base_attenuation;
        }

        out_pdf = fmaxf(pdf_total, 1e-6f);
        attenuation = f_total / out_pdf;
        return true;
    }

    float max_fresnel = fmaxf(F_base_for_NdotV.x,
                              fmaxf(F_base_for_NdotV.y, F_base_for_NdotV.z));
    float specular_prob = (metal > 0.0f) ? 1.0f : max_fresnel;

    float P_opaque_spec = prob_base * specular_prob;
    float P_opaque_diff = prob_base * (1.0f - specular_prob);

    float u = curand_uniform(state);
    if (u < P_coat) {
        vec3 H = importance_sample_ggx(state, N, clearcoatRough);
        scattered_dir = reflectVec(-V, H);
        is_specular_bounce = (clearcoatRough < 0.1f);
    } else if (u < P_coat + P_opaque_spec) {
        vec3 H = importance_sample_ggx(state, N, rough);
        scattered_dir = reflectVec(-V, H);
        is_specular_bounce = (rough < 0.1f);
    } else if (P_opaque_diff > 1e-6f) {
        vec3 hemi_sample = sample_cosine_hemisphere(state);
        scattered_dir = hemisphere_to_world(hemi_sample, N);
        is_specular_bounce = false;
    } else {
        return false;
    }

    scattered_dir = normalize(scattered_dir);

    float NdotL = fmaxf(dot(N, scattered_dir), 0.0f);
    vec3 f_total(0.0f);
    float pdf_total = 0.0f;

    if (P_coat > 0.0f) {
        vec3 H_coat = normalize(V + scattered_dir);
        float NdotH_coat = fmaxf(dot(N, H_coat), 0.0f);
        float VdotH_coat = fmaxf(dot(V, H_coat), 0.0f);
        float D_coat = distributionGGX(N, H_coat, clearcoatRough);
        float G_coat = geometrySmith(N, V, scattered_dir, clearcoatRough);
        vec3 F_coat = fresnelSchlick(VdotH_coat, F0_coat);

        float pdf_L_coat = (D_coat * NdotH_coat) / (4.0f * VdotH_coat + 1e-6f);
        pdf_total += P_coat * pdf_L_coat;

        vec3 brdf_coat =
            (D_coat * G_coat * F_coat) / (4.0f * NdotV * NdotL + 1e-6f);
        f_total += clearcoat * brdf_coat * NdotL;
    }

    vec3 H_for_base = normalize(V + scattered_dir);
    float VdotH_for_base = fmaxf(dot(V, H_for_base), 0.0f);
    vec3 F_coat_atten = fresnelSchlick(VdotH_for_base, F0_coat);
    vec3 base_attenuation = vec3(1.0f) - clearcoat * F_coat_atten;

    vec3 H_spec = H_for_base;
    float NdotH_spec = fmaxf(dot(N, H_spec), 0.0f);
    float VdotH_spec = VdotH_for_base;
    float D_spec = distributionGGX(N, H_spec, rough);
    float G_spec = geometrySmith(N, V, scattered_dir, rough);
    vec3 F_spec = fresnelSchlick(VdotH_spec, F0_base);

    float pdf_L_spec = (D_spec * NdotH_spec) / (4.0f * VdotH_spec + 1e-6f);
    pdf_total += P_opaque_spec * pdf_L_spec;

    vec3 brdf_spec =
        (D_spec * G_spec * F_spec) / (4.0f * NdotV * NdotL + 1e-6f);
    f_total += brdf_spec * NdotL * base_attenuation;

    if (P_opaque_diff > 1e-6f) {
        float pdf_L_diff = NdotL / PI;
        pdf_total += P_opaque_diff * pdf_L_diff;

        const float sheen = clamp01(mat.sheen);
        vec3 kD = (vec3(1.0f) - F_base_for_NdotV) * (1.0f - metal);
        vec3 f_diff = (kD * albedo / PI) * NdotL;

        if (sheen > 0.0f) {
            const vec3 sheenTint = mat.sheenTint;
            float FH = 1.0f - fmaxf(dot(V, H_for_base), 0.0f);
            float FH5 = FH * FH * FH * FH * FH;
            vec3 Csheen = lerp(vec3(1.0f), sheenTint, 0.5f);
            f_diff = f_diff + sheen * Csheen * FH5 * NdotL;
        }

        f_total += f_diff * base_attenuation;
    }

    out_pdf = pdf_total;
    attenuation = f_total / fmaxf(pdf_total, 1e-6f);

    return true;
}

__device__ vec3 tracePath(Ray ray, DeviceMesh *meshes, int nMeshes,
                          DeviceBVHNode *tlasNodes, int *tlasMeshIndices,
                          const DeviceMaterials &materials, Light *lights,
                          int nLights, vec3 skyColorTop, vec3 skyColorBottom,
                          bool useSky, curandState *state, int max_depth,
                          cudaTextureObject_t envMap, vec3 &out_first_normal,
                          float &out_first_depth, int &out_first_objectId) {

    vec3 accumulated_color(0.0f);
    vec3 throughput(1.0f);

    bool prev_was_specular = true;

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        HitInfo hit = traceRay(ray, meshes, tlasNodes, tlasMeshIndices);

        if (bounce == 0) {
            if (!hit.hit) {
                out_first_normal = vec3(0.0f);
                out_first_depth = 1e30f;
                out_first_objectId = -1;
            } else {
                out_first_normal = hit.normal;
                out_first_depth = hit.t;
                out_first_objectId = hit.mesh_index;
            }
        }

        if (!hit.hit) {
            vec3 sky =
                sampleSky(ray, skyColorTop, skyColorBottom, useSky, envMap);
            accumulated_color = accumulated_color + throughput * sky;
            break;
        }

        MaterialProps mat;
        mat.load(materials, hit.mesh_index);

        const int mat_id = hit.mesh_index;
        const vec3 V = -ray.direction();

        if (!hit.front_face) {
            vec3 transmittance_color = mat.albedo;
            vec3 T_unit = fmaxf(vec3(1e-6f), transmittance_color);
            vec3 absorption_coefficient = -log(T_unit);
            throughput =
                throughput * beerLambert(absorption_coefficient, hit.t);
        }

        vec3 emission = mat.emission;
        if (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f) {
            if (bounce == 0 || prev_was_specular) {
                accumulated_color = accumulated_color + throughput * emission;
            }
        }

        const float transmission = mat.transmission;

        if (!ray.isSpecular()) {
            vec3 L_nee;
            float pdf_nee;

            vec3 brdf_nee = sample_direct_lighting_with_mat(
                hit, mat, materials, ray, meshes, nMeshes, tlasNodes,
                tlasMeshIndices, lights, nLights, state, L_nee, pdf_nee);

            if (brdf_nee.x > 0.0f || brdf_nee.y > 0.0f || brdf_nee.z > 0.0f) {
                if (pdf_nee > 0.0f) {
                    float pdf_brdf =
                        material_pdf(hit, mat_id, materials, V, L_nee);
                    float w = mis_weight(pdf_nee, pdf_brdf);
                    accumulated_color =
                        accumulated_color + throughput * brdf_nee * w;
                }
            }
        }

        vec3 scatter_dir;
        vec3 attenuation;
        bool is_specular;
        float pdf_brdf;

        if (!material_scatter(hit, mat, ray, state, scatter_dir, attenuation,
                              is_specular, pdf_brdf)) {
            break;
        }

        prev_was_specular = is_specular;

        if (bounce >= RUSSIAN_ROULETTE_START_BOUNCE) {
            float p =
                fmaxf(RUSSIAN_ROULETTE_MIN_PROB,
                      fminf(0.95f, fmaxf(throughput.x,
                                         fmaxf(throughput.y, throughput.z))));
            if (curand_uniform(state) > p) {
                break;
            }
            throughput = throughput / p;
        }

        throughput = throughput * attenuation;
        throughput = clamp_vector_soft(throughput, MAX_BOUNCE_WEIGHT);

        vec3 offset_origin;
        if (dot(scatter_dir, hit.normal) > 0.0f) {
            offset_origin = hit.point + hit.normal * 1e-4f;
        } else {
            offset_origin = hit.point - hit.normal * 1e-4f;
        }

        ray = Ray(offset_origin, scatter_dir, is_specular);
    }

    accumulated_color =
        clamp_vector_soft(accumulated_color, MAX_FINAL_RADIANCE);

    return accumulated_color;
}

__device__ SplitPathOutput tracePathSplit(
    Ray ray, DeviceMesh *meshes, int nMeshes, DeviceBVHNode *tlasNodes,
    int *tlasMeshIndices, const DeviceMaterials &materials, Light *lights,
    int nLights, vec3 skyColorTop, vec3 skyColorBottom, bool useSky,
    curandState *state, int max_depth, cudaTextureObject_t envMap,
    vec3 &out_first_normal, float &out_first_depth, float &out_first_roughness,
    float &out_first_transmission) {

    SplitPathOutput result;
    result.diffuse = vec3(0.0f);
    result.specular = vec3(0.0f);
    result.emission = vec3(0.0f);

    vec3 throughput(1.0f);
    bool path_still_specular = true;

    bool prev_was_specular = true;

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        HitInfo hit = traceRay(ray, meshes, tlasNodes, tlasMeshIndices);

        MaterialProps mat;
        if (hit.hit) {
            mat.load(materials, hit.mesh_index);
        }

        if (bounce == 0) {
            if (!hit.hit) {
                out_first_normal = vec3(0.0f);
                out_first_depth = 1e30f;
                out_first_roughness = 1.0f;
                out_first_transmission = 0.0f;
            } else {
                out_first_normal = hit.normal;
                out_first_depth = hit.t;
                out_first_roughness = mat.roughness;
                out_first_transmission = mat.transmission;
            }
        }

        if (!hit.hit) {
            vec3 sky =
                sampleSky(ray, skyColorTop, skyColorBottom, useSky, envMap);
            vec3 contribution = throughput * sky;

            if (path_still_specular) {
                result.specular += contribution;
            } else {
                result.diffuse += contribution;
            }
            break;
        }

        const int mat_id = hit.mesh_index;
        const vec3 V = -ray.direction();

        if (!hit.front_face) {
            vec3 transmittance_color = mat.albedo;
            vec3 T_unit = fmaxf(vec3(1e-6f), transmittance_color);
            vec3 absorption_coefficient = -log(T_unit);
            throughput =
                throughput * beerLambert(absorption_coefficient, hit.t);
        }

        vec3 mat_emission = mat.emission;
        if (mat_emission.x > 0.0f || mat_emission.y > 0.0f ||
            mat_emission.z > 0.0f) {

            if (bounce == 0 || prev_was_specular) {
                vec3 contribution = throughput * mat_emission;

                if (bounce == 0) {
                    result.emission += contribution;
                } else if (path_still_specular) {
                    result.specular += contribution;
                } else {
                    result.diffuse += contribution;
                }
            }
        }

        if (!ray.isSpecular()) {
            vec3 L_nee;
            float pdf_nee;
            vec3 nee_diffuse, nee_specular;

            sample_direct_lighting_split(hit, mat, materials, ray, meshes,
                                         nMeshes, tlasNodes, tlasMeshIndices,
                                         lights, nLights, state, L_nee, pdf_nee,
                                         nee_diffuse, nee_specular);

            if (pdf_nee > 0.0f) {
                float pdf_brdf = material_pdf(hit, mat_id, materials, V, L_nee);
                float w = mis_weight(pdf_nee, pdf_brdf);

                result.diffuse += throughput * nee_diffuse * w;
                result.specular += throughput * nee_specular * w;
            }
        }

        vec3 scatter_dir;
        vec3 attenuation;
        bool is_specular_bounce;
        float pdf_brdf;

        if (!material_scatter(hit, mat, ray, state, scatter_dir, attenuation,
                              is_specular_bounce, pdf_brdf)) {
            break;
        }

        prev_was_specular = is_specular_bounce;

        if (!is_specular_bounce) {
            path_still_specular = false;
        }

        if (bounce >= RUSSIAN_ROULETTE_START_BOUNCE) {
            float p =
                fmaxf(RUSSIAN_ROULETTE_MIN_PROB,
                      fminf(0.95f, fmaxf(throughput.x,
                                         fmaxf(throughput.y, throughput.z))));
            if (curand_uniform(state) > p) {
                break;
            }
            throughput = throughput / p;
        }

        throughput = throughput * attenuation;
        throughput = clamp_vector_soft(throughput, MAX_BOUNCE_WEIGHT);

        vec3 offset_origin;
        if (dot(scatter_dir, hit.normal) > 0.0f) {
            offset_origin = hit.point + hit.normal * 1e-4f;
        } else {
            offset_origin = hit.point - hit.normal * 1e-4f;
        }

        ray = Ray(offset_origin, scatter_dir, is_specular_bounce);
    }

    return result;
}

#endif
