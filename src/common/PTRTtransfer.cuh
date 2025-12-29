#ifndef UNIFIED_SCENE_CUH
// file overview
// unified scene wrapper shared between the simple ray tracer and the path
// tracer stores materials lights camera mesh descriptors transforms and
// animation state provides builder functions that convert unified data into
// renderer specific scenes provides incremental update helpers for runtime
// animation and interaction

#define UNIFIED_SCENE_CUH

#if defined(UNIFIED_SCENE_ENABLE_RT) && defined(UNIFIED_SCENE_ENABLE_PT)
#error                                                                         \
    "Cannot enable both UNIFIED_SCENE_ENABLE_RT and UNIFIED_SCENE_ENABLE_PT. Choose one renderer."
#endif

#if !defined(UNIFIED_SCENE_ENABLE_RT) && !defined(UNIFIED_SCENE_ENABLE_PT)

#endif

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/vec3.cuh"

struct UnifiedTransform {
    vec3 position{0.0f, 0.0f, 0.0f};
    vec3 rotation{0.0f, 0.0f, 0.0f};
    vec3 scale{1.0f, 1.0f, 1.0f};

    UnifiedTransform() = default;

    UnifiedTransform(const vec3 &pos, const vec3 &rot = vec3(0.0f),
                     const vec3 &scl = vec3(1.0f))
        // function position

        : position(pos), rotation(rot), scale(scl) {}

    // function set position

    UnifiedTransform &setPosition(const vec3 &pos) {
        position = pos;
        return *this;
    }

    // function set position

    UnifiedTransform &setPosition(float x, float y, float z) {
        position = vec3(x, y, z);
        return *this;
    }

    // function set rotation

    UnifiedTransform &setRotation(const vec3 &rot) {
        rotation = rot;
        return *this;
    }

    // function set rotation degrees

    UnifiedTransform &setRotationDegrees(const vec3 &deg) {
        const float toRad = 3.14159265358979f / 180.0f;
        rotation = deg * toRad;
        return *this;
    }

    // function set rotation degrees

    UnifiedTransform &setRotationDegrees(float x, float y, float z) {
        const float toRad = 3.14159265358979f / 180.0f;
        rotation = vec3(x * toRad, y * toRad, z * toRad);
        return *this;
    }

    // function set scale

    UnifiedTransform &setScale(const vec3 &scl) {
        scale = scl;
        return *this;
    }

    // function set scale

    UnifiedTransform &setScale(float uniform) {
        scale = vec3(uniform, uniform, uniform);
        return *this;
    }

    // function translate

    UnifiedTransform &translate(const vec3 &delta) {
        position = position + delta;
        return *this;
    }

    // function rotate

    UnifiedTransform &rotate(const vec3 &deltaRadians) {
        rotation = rotation + deltaRadians;
        return *this;
    }

    // function rotate degrees

    UnifiedTransform &rotateDegrees(const vec3 &deltaDegrees) {
        const float toRad = 3.14159265358979f / 180.0f;
        rotation = rotation + deltaDegrees * toRad;
        return *this;
    }

    // function is identity

    bool isIdentity() const {
        const float eps = 1e-6f;
        return (fabsf(position.x) < eps && fabsf(position.y) < eps &&
                fabsf(position.z) < eps && fabsf(rotation.x) < eps &&
                fabsf(rotation.y) < eps && fabsf(rotation.z) < eps &&
                fabsf(scale.x - 1.0f) < eps && fabsf(scale.y - 1.0f) < eps &&
                fabsf(scale.z - 1.0f) < eps);
    }

    static UnifiedTransform lerp(const UnifiedTransform &a,
                                 const UnifiedTransform &b, float t) {
        return UnifiedTransform(a.position + (b.position - a.position) * t,
                                a.rotation + (b.rotation - a.rotation) * t,
                                a.scale + (b.scale - a.scale) * t);
    }
};

enum class UnifiedLightType { Point = 0, Directional = 1, Spot = 2, Area = 3 };

struct UnifiedLight {
    UnifiedLightType type = UnifiedLightType::Point;
    vec3 position{0, 10, 0};
    vec3 direction{0, -1, 0};
    vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float range = 100.0f;
    float innerCone = 0.5f;
    float outerCone = 0.7f;
    float radius = 0.0f;

    vec3 areaU{1, 0, 0};
    vec3 areaV{0, 0, 1};
    float areaWidth = 1.0f;
    float areaHeight = 1.0f;

    bool animated = false;
    std::string name;

    static UnifiedLight Point(const vec3 &pos, const vec3 &col,
                              float intensity = 1.0f, float range = 100.0f,
                              float radius = 0.0f) {
        UnifiedLight l;
        l.type = UnifiedLightType::Point;
        l.position = pos;
        l.color = col;
        l.intensity = intensity;
        l.range = range;
        l.radius = radius;
        return l;
    }

    static UnifiedLight Directional(const vec3 &dir, const vec3 &col,
                                    float intensity = 1.0f) {
        UnifiedLight l;
        l.type = UnifiedLightType::Directional;
        l.direction = dir.normalized();
        l.color = col;
        l.intensity = intensity;
        return l;
    }

    static UnifiedLight Spot(const vec3 &pos, const vec3 &dir, const vec3 &col,
                             float intensity = 1.0f, float innerCone = 0.5f,
                             float outerCone = 0.7f, float range = 100.0f,
                             float radius = 0.0f) {
        UnifiedLight l;
        l.type = UnifiedLightType::Spot;
        l.position = pos;
        l.direction = dir.normalized();
        l.color = col;
        l.intensity = intensity;
        l.innerCone = innerCone;
        l.outerCone = outerCone;
        l.range = range;
        l.radius = radius;
        return l;
    }

    static UnifiedLight Area(const vec3 &pos, const vec3 &dir, const vec3 &col,
                             float width, float height,
                             float intensity = 1.0f) {
        UnifiedLight l;
        l.type = UnifiedLightType::Area;
        l.position = pos;
        l.direction = dir.normalized();
        l.color = col;
        l.intensity = intensity;
        l.areaWidth = width;
        l.areaHeight = height;

        vec3 up = (fabsf(dir.y) < 0.999f) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        l.areaU = cross(up, dir).normalized() * width;
        l.areaV = cross(dir, l.areaU).normalized() * height;
        return l;
    }

    // function move to

    UnifiedLight &moveTo(const vec3 &pos) {
        position = pos;
        return *this;
    }

    // function set direction

    UnifiedLight &setDirection(const vec3 &dir) {
        direction = dir.normalized();
        return *this;
    }

    // function set color

    UnifiedLight &setColor(const vec3 &col) {
        color = col;
        return *this;
    }

    // function set intensity

    UnifiedLight &setIntensity(float i) {
        intensity = i;
        return *this;
    }
};

struct UnifiedMaterial {

    vec3 albedo{0.8f, 0.8f, 0.8f};
    vec3 specular{0.04f, 0.04f, 0.04f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    vec3 emission{0.0f, 0.0f, 0.0f};

    float ior = 1.5f;
    float transmission = 0.0f;
    float transmissionRoughness = 0.0f;

    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.03f;

    vec3 subsurfaceColor{1.0f, 1.0f, 1.0f};
    float subsurfaceRadius = 0.0f;

    float anisotropy = 0.0f;
    float sheen = 0.0f;
    vec3 sheenTint{0.5f, 0.5f, 0.5f};

    float iridescence = 0.0f;
    float iridescenceThickness = 550.0f;

    std::string name;

    UnifiedMaterial() = default;

    UnifiedMaterial(const vec3 &alb, float rough = 0.5f, float met = 0.0f)
        // function albedo

        : albedo(alb), roughness(rough), metallic(met) {
        specular = ::lerp(vec3(0.04f), albedo, metallic);
    }

    // function gold
    static UnifiedMaterial Gold() {
        UnifiedMaterial m(vec3(1.0f, 0.766f, 0.336f), 0.1f, 1.0f);
        m.specular = vec3(1.0f, 0.782f, 0.344f);
        m.name = "Gold";
        return m;
    }

    // function plain clay

    static UnifiedMaterial PlainClay() {
        UnifiedMaterial m(vec3(0.5f, 0.5f, 0.5f), 1.0f, 0.0f);
        m.name = "PlainClay";
        return m;
    }

    // function silver

    static UnifiedMaterial Silver() {
        UnifiedMaterial m(vec3(0.972f, 0.960f, 0.915f), 0.05f, 1.0f);
        m.specular = vec3(0.972f, 0.960f, 0.915f);
        m.name = "Silver";
        return m;
    }

    // function copper

    static UnifiedMaterial Copper() {
        UnifiedMaterial m(vec3(0.955f, 0.637f, 0.538f), 0.15f, 1.0f);
        m.specular = vec3(0.955f, 0.637f, 0.538f);
        m.name = "Copper";
        return m;
    }

    // function brushed aluminum

    static UnifiedMaterial BrushedAluminum() {
        UnifiedMaterial m(vec3(0.913f, 0.921f, 0.925f), 0.3f, 1.0f);
        m.anisotropy = 0.8f;
        m.name = "BrushedAluminum";
        return m;
    }

    // function iron

    static UnifiedMaterial Iron() {
        UnifiedMaterial m(vec3(0.560f, 0.570f, 0.580f), 0.4f, 1.0f);
        m.specular = vec3(0.560f, 0.570f, 0.580f);
        m.name = "Iron";
        return m;
    }

    // function chrome

    static UnifiedMaterial Chrome() {
        UnifiedMaterial m(vec3(0.549f, 0.556f, 0.554f), 0.02f, 1.0f);
        m.specular = vec3(0.549f, 0.556f, 0.554f);
        m.name = "Chrome";
        return m;
    }

    // function glass

    static UnifiedMaterial Glass() {
        UnifiedMaterial m(vec3(1.0f), 0.02f, 0.0f);
        m.transmission = 0.98f;
        m.ior = 1.5f;
        m.specular = vec3(0.04f);
        m.name = "Glass";
        return m;
    }

    // function frosted glass

    static UnifiedMaterial FrostedGlass() {
        UnifiedMaterial m = Glass();
        m.roughness = 0.3f;
        m.transmissionRoughness = 0.5f;
        m.name = "FrostedGlass";
        return m;
    }

    // function diamond

    static UnifiedMaterial Diamond() {
        UnifiedMaterial m(vec3(1.0f), 0.0f, 0.0f);
        m.transmission = 0.95f;
        m.ior = 2.42f;
        m.specular = vec3(0.17f);
        m.name = "Diamond";
        return m;
    }

    // function water

    static UnifiedMaterial Water() {
        UnifiedMaterial m(vec3(0.8f, 0.95f, 1.0f), 0.01f, 0.0f);
        m.transmission = 0.9f;
        m.ior = 1.33f;
        m.specular = vec3(0.02f);
        m.name = "Water";
        return m;
    }

    // function ice

    static UnifiedMaterial Ice() {
        UnifiedMaterial m(vec3(0.9f, 0.95f, 1.0f), 0.1f, 0.0f);
        m.transmission = 0.7f;
        m.ior = 1.31f;
        m.subsurfaceColor = vec3(0.8f, 0.9f, 1.0f);
        m.subsurfaceRadius = 0.3f;
        m.name = "Ice";
        return m;
    }

    // function plastic red

    static UnifiedMaterial PlasticRed() {
        UnifiedMaterial m(vec3(0.8f, 0.1f, 0.1f), 0.2f, 0.0f);
        m.specular = vec3(0.04f);
        m.name = "PlasticRed";
        return m;
    }

    // function plastic blue

    static UnifiedMaterial PlasticBlue() {
        UnifiedMaterial m(vec3(0.1f, 0.2f, 0.8f), 0.2f, 0.0f);
        m.specular = vec3(0.04f);
        m.name = "PlasticBlue";
        return m;
    }

    // function plastic green

    static UnifiedMaterial PlasticGreen() {
        UnifiedMaterial m(vec3(0.1f, 0.7f, 0.2f), 0.2f, 0.0f);
        m.specular = vec3(0.04f);
        m.name = "PlasticGreen";
        return m;
    }

    // function rubber black

    static UnifiedMaterial RubberBlack() {
        UnifiedMaterial m(vec3(0.05f), 0.8f, 0.0f);
        m.specular = vec3(0.03f);
        m.name = "RubberBlack";
        return m;
    }

    // function car paint

    static UnifiedMaterial CarPaint(const vec3 &baseColor) {
        UnifiedMaterial m(baseColor, 0.2f, 0.3f);
        m.clearcoat = 1.0f;
        m.clearcoatRoughness = 0.03f;
        m.specular = vec3(0.05f);
        m.name = "CarPaint";
        return m;
    }

    // function pearlescent paint

    static UnifiedMaterial PearlescentPaint(const vec3 &baseColor) {
        UnifiedMaterial m = CarPaint(baseColor);
        m.iridescence = 0.8f;
        m.iridescenceThickness = 400.0f;
        m.name = "PearlescentPaint";
        return m;
    }

    // function skin

    static UnifiedMaterial Skin() {
        UnifiedMaterial m(vec3(0.95f, 0.75f, 0.67f), 0.4f, 0.0f);
        m.subsurfaceColor = vec3(1.0f, 0.4f, 0.3f);
        m.subsurfaceRadius = 0.5f;
        m.specular = vec3(0.028f);
        m.name = "Skin";
        return m;
    }

    // function wax

    static UnifiedMaterial Wax() {
        UnifiedMaterial m(vec3(0.95f, 0.93f, 0.88f), 0.3f, 0.0f);
        m.subsurfaceColor = vec3(1.0f, 0.9f, 0.7f);
        m.subsurfaceRadius = 0.8f;
        m.specular = vec3(0.03f);
        m.name = "Wax";
        return m;
    }

    // function jade

    static UnifiedMaterial Jade() {
        UnifiedMaterial m(vec3(0.2f, 0.6f, 0.4f), 0.1f, 0.0f);
        m.subsurfaceColor = vec3(0.3f, 0.8f, 0.5f);
        m.subsurfaceRadius = 0.3f;
        m.specular = vec3(0.05f);
        m.name = "Jade";
        return m;
    }

    // function velvet

    static UnifiedMaterial Velvet(const vec3 &color) {
        UnifiedMaterial m(color, 0.8f, 0.0f);
        m.sheen = 1.0f;
        m.sheenTint = color * 1.2f;
        m.specular = vec3(0.02f);
        m.name = "Velvet";
        return m;
    }

    // function silk

    static UnifiedMaterial Silk(const vec3 &color) {
        UnifiedMaterial m(color, 0.2f, 0.0f);
        m.sheen = 0.6f;
        m.sheenTint = vec3(1.0f);
        m.anisotropy = 0.5f;
        m.specular = vec3(0.04f);
        m.name = "Silk";
        return m;
    }

    // function cotton

    static UnifiedMaterial Cotton(const vec3 &color) {
        UnifiedMaterial m(color, 0.9f, 0.0f);
        m.specular = vec3(0.02f);
        m.name = "Cotton";
        return m;
    }

    // function soap bubble

    static UnifiedMaterial SoapBubble() {
        UnifiedMaterial m(vec3(1.0f), 0.0f, 0.0f);
        m.transmission = 0.95f;
        m.ior = 1.33f;
        m.iridescence = 1.0f;
        m.iridescenceThickness = 380.0f;
        m.specular = vec3(0.04f);
        m.name = "SoapBubble";
        return m;
    }

    // function oil slick

    static UnifiedMaterial OilSlick() {
        UnifiedMaterial m(vec3(0.01f), 0.0f, 0.95f);
        m.iridescence = 1.0f;
        m.iridescenceThickness = 450.0f;
        m.name = "OilSlick";
        return m;
    }

    static UnifiedMaterial EmissiveLamp(const vec3 &color,
                                        float intensity = 5.0f) {
        UnifiedMaterial m(vec3(1.0f), 0.0f, 0.0f);
        m.emission = color * intensity;
        m.name = "EmissiveLamp";
        return m;
    }

    // function neon light

    static UnifiedMaterial NeonLight(const vec3 &color) {
        UnifiedMaterial m(color * 0.1f, 0.0f, 0.0f);
        m.emission = color * 1.5f;
        m.name = "NeonLight";
        return m;
    }

    // function marble carrara

    static UnifiedMaterial MarbleCarrara(bool polished = false) {
        const float baseRough = polished ? 0.15f : 0.35f;
        const float coatAmt = polished ? 0.70f : 0.15f;
        const float coatRough = polished ? 0.05f : 0.20f;

        UnifiedMaterial m(vec3(0.93f, 0.94f, 0.96f), baseRough, 0.0f);
        m.ior = 1.49f;
        m.clearcoat = coatAmt;
        m.clearcoatRoughness = coatRough;
        m.subsurfaceColor = vec3(0.98f, 0.98f, 0.96f);
        m.subsurfaceRadius = 1.0f;
        m.name = "MarbleCarrara";
        return m;
    }

    // function marble nero

    static UnifiedMaterial MarbleNero(bool polished = true) {
        const float baseRough = polished ? 0.12f : 0.28f;
        const float coatAmt = polished ? 0.85f : 0.20f;
        const float coatRough = polished ? 0.04f : 0.18f;

        UnifiedMaterial m(vec3(0.04f, 0.045f, 0.05f), baseRough, 0.0f);
        m.ior = 1.49f;
        m.clearcoat = coatAmt;
        m.clearcoatRoughness = coatRough;
        m.subsurfaceColor = vec3(0.15f, 0.15f, 0.16f);
        m.subsurfaceRadius = 0.6f;
        m.name = "MarbleNero";
        return m;
    }

    // function marble verde

    static UnifiedMaterial MarbleVerde(bool polished = true) {
        const float baseRough = polished ? 0.14f : 0.30f;
        const float coatAmt = polished ? 0.75f : 0.18f;
        const float coatRough = polished ? 0.05f : 0.19f;

        UnifiedMaterial m(vec3(0.10f, 0.18f, 0.14f), baseRough, 0.0f);
        m.ior = 1.49f;
        m.clearcoat = coatAmt;
        m.clearcoatRoughness = coatRough;
        m.subsurfaceColor = vec3(0.12f, 0.20f, 0.16f);
        m.subsurfaceRadius = 0.8f;
        m.name = "MarbleVerde";
        return m;
    }

    // function concrete

    static UnifiedMaterial Concrete() {
        UnifiedMaterial m(vec3(0.5f, 0.5f, 0.5f), 0.9f, 0.0f);
        m.specular = vec3(0.02f);
        m.name = "Concrete";
        return m;
    }

    // function wood oak

    static UnifiedMaterial WoodOak() {
        UnifiedMaterial m(vec3(0.6f, 0.4f, 0.2f), 0.5f, 0.0f);
        m.specular = vec3(0.04f);
        m.name = "WoodOak";
        return m;
    }

    // function wood cherry

    static UnifiedMaterial WoodCherry() {
        UnifiedMaterial m(vec3(0.5f, 0.2f, 0.1f), 0.4f, 0.0f);
        m.clearcoat = 0.3f;
        m.clearcoatRoughness = 0.1f;
        m.name = "WoodCherry";
        return m;
    }

    // function wood walnut

    static UnifiedMaterial WoodWalnut() {
        UnifiedMaterial m(vec3(0.3f, 0.2f, 0.15f), 0.45f, 0.0f);
        m.specular = vec3(0.04f);
        m.name = "WoodWalnut";
        return m;
    }
};

struct UnifiedCameraConfig {
    vec3 lookfrom{0, 0, 0};
    vec3 lookat{0, 0, -1};
    vec3 vup{0, 1, 0};
    float vfov = 60.0f;
    float aperture = 0.003125f;
    float focusDist = 1.0f;
    float nearClip = 0.1f;
    float farClip = 1000.0f;

    UnifiedCameraConfig() = default;

    UnifiedCameraConfig(const vec3 &from, const vec3 &at, const vec3 &up,
                        float fov, float apt = 0.0f, float focus = 1.0f)
        : lookfrom(from), lookat(at), vup(up), vfov(fov), aperture(apt),
          // function focus dist

          focusDist(focus) {}

    // function set position

    UnifiedCameraConfig &setPosition(const vec3 &pos) {
        lookfrom = pos;
        return *this;
    }

    // function set target

    UnifiedCameraConfig &setTarget(const vec3 &target) {
        lookat = target;
        return *this;
    }

    // function set fov

    UnifiedCameraConfig &setFOV(float fov) {
        vfov = fov;
        return *this;
    }

    // function set dof

    UnifiedCameraConfig &setDOF(float aperture_, float focusDist_) {
        aperture = aperture_;
        focusDist = focusDist_;
        return *this;
    }

    UnifiedCameraConfig &orbit(const vec3 &center, float distance,
                               float azimuth, float elevation) {
        float ca = cosf(azimuth), sa = sinf(azimuth);
        float ce = cosf(elevation), se = sinf(elevation);
        lookfrom = center +
                   vec3(distance * ce * ca, distance * se, distance * ce * sa);
        lookat = center;
        return *this;
    }
};

struct UnifiedMeshDesc {
    enum class Type {
        ObjFile,
        Cube,
        PlaneXZ,
        PlaneXY,
        PlaneYZ,
        Sphere,
        Triangles
    };

    Type type = Type::Cube;
    std::string objPath;
    std::vector<vec3> triangleVerts;
    float planeY = 0.0f;
    float planeHalfSize = 50.0f;
    int sphereSegments = 32;

    UnifiedMaterial material;
    std::string materialRef;
    UnifiedTransform transform;

    UnifiedTransform bakedTransform;

    std::string name;
    bool isDynamic = false;
    bool castsShadows = true;
    bool receivesShadows = true;

    static UnifiedMeshDesc
    FromOBJ(const std::string &path,

            const UnifiedMaterial &mat = UnifiedMaterial()) {
        UnifiedMeshDesc d;
        d.type = Type::ObjFile;
        d.objPath = path;
        d.material = mat;
        return d;
    }

    static UnifiedMeshDesc
    // function cube

    Cube(const UnifiedMaterial &mat = UnifiedMaterial()) {
        UnifiedMeshDesc d;
        d.type = Type::Cube;
        d.material = mat;
        return d;
    }

    static UnifiedMeshDesc
    PlaneXZ(float y, float halfSize,

            const UnifiedMaterial &mat = UnifiedMaterial()) {
        UnifiedMeshDesc d;
        d.type = Type::PlaneXZ;
        d.planeY = y;
        d.planeHalfSize = halfSize;
        d.material = mat;
        return d;
    }

    static UnifiedMeshDesc
    // function sphere

    Sphere(int segments = 32, const UnifiedMaterial &mat = UnifiedMaterial()) {
        UnifiedMeshDesc d;
        d.type = Type::Sphere;
        d.sphereSegments = segments;
        d.material = mat;
        return d;
    }

    // function set position

    UnifiedMeshDesc &setPosition(const vec3 &pos) {
        transform.setPosition(pos);
        return *this;
    }

    // function set rotation

    UnifiedMeshDesc &setRotation(const vec3 &rot) {
        transform.setRotation(rot);
        return *this;
    }

    // function set rotation degrees

    UnifiedMeshDesc &setRotationDegrees(const vec3 &deg) {
        transform.setRotationDegrees(deg);
        return *this;
    }

    // function set scale

    UnifiedMeshDesc &setScale(const vec3 &s) {
        transform.setScale(s);
        return *this;
    }

    // function set scale

    UnifiedMeshDesc &setScale(float s) {
        transform.setScale(s);
        return *this;
    }

    // function set transform

    UnifiedMeshDesc &setTransform(const UnifiedTransform &t) {
        transform = t;
        return *this;
    }

    // function set name

    UnifiedMeshDesc &setName(const std::string &n) {
        name = n;
        return *this;
    }

    // function set dynamic

    UnifiedMeshDesc &setDynamic(bool dynamic = true) {
        isDynamic = dynamic;
        return *this;
    }

    // function set material

    UnifiedMeshDesc &setMaterial(const UnifiedMaterial &mat) {
        material = mat;
        return *this;
    }
};

struct UnifiedSkyConfig {
    bool enabled = true;
    vec3 topColor{0.6f, 0.7f, 1.0f};
    vec3 bottomColor{1.0f, 1.0f, 1.0f};
    std::string hdriPath;
    float hdriIntensity = 1.0f;
    float hdriRotation = 0.0f;
};

class UnifiedScene;

class ObjectHandle {
  public:
    size_t index;
    UnifiedScene *scene;

    // function object handle

    ObjectHandle() : index(SIZE_MAX), scene(nullptr) {}
    // function object handle

    ObjectHandle(size_t idx, UnifiedScene *s) : index(idx), scene(s) {}

    // function is valid

    bool isValid() const { return scene != nullptr && index != SIZE_MAX; }

    ObjectHandle &setPosition(const vec3 &pos);
    ObjectHandle &setRotation(const vec3 &rot);
    ObjectHandle &setRotationDegrees(const vec3 &deg);
    ObjectHandle &setScale(const vec3 &s);
    ObjectHandle &setScale(float s);
    ObjectHandle &translate(const vec3 &delta);
    ObjectHandle &rotate(const vec3 &deltaRad);
    ObjectHandle &rotateDegrees(const vec3 &deltaDeg);

    UnifiedTransform getTransform() const;
    vec3 getPosition() const;

    ObjectHandle &setMaterial(const UnifiedMaterial &mat);

    ObjectHandle &useLibraryMaterial(const std::string &materialName);

    ObjectHandle &setName(const std::string &name);

    ObjectHandle &setDynamic(bool dynamic);
};

class LightHandle {
  public:
    size_t index;
    UnifiedScene *scene;

    // function light handle

    LightHandle() : index(SIZE_MAX), scene(nullptr) {}
    // function light handle

    LightHandle(size_t idx, UnifiedScene *s) : index(idx), scene(s) {}

    // function is valid

    bool isValid() const { return scene != nullptr && index != SIZE_MAX; }

    LightHandle &setPosition(const vec3 &pos);
    LightHandle &setDirection(const vec3 &dir);
    LightHandle &setColor(const vec3 &col);
    LightHandle &setIntensity(float i);
    LightHandle &setName(const std::string &name);

    vec3 getPosition() const;
};

enum class EaseType { Linear, EaseIn, EaseOut, EaseInOut, Bounce, Elastic };

template <typename T> struct Keyframe {
    float time;
    T value;
    EaseType ease = EaseType::Linear;

    // function keyframe

    Keyframe() : time(0.0f) {}
    Keyframe(float t, const T &v, EaseType e = EaseType::Linear)
        // function time

        : time(t), value(v), ease(e) {}
};

// function apply easing

inline float applyEasing(float t, EaseType ease) {
    switch (ease) {
    case EaseType::Linear:
        return t;
    case EaseType::EaseIn:
        return t * t;
    case EaseType::EaseOut:
        return t * (2.0f - t);
    case EaseType::EaseInOut:
        return t < 0.5f ? 2.0f * t * t : -1.0f + (4.0f - 2.0f * t) * t;
    case EaseType::Bounce: {
        if (t < 1.0f / 2.75f)
            return 7.5625f * t * t;
        else if (t < 2.0f / 2.75f) {
            t -= 1.5f / 2.75f;
            return 7.5625f * t * t + 0.75f;
        } else if (t < 2.5f / 2.75f) {
            t -= 2.25f / 2.75f;
            return 7.5625f * t * t + 0.9375f;
        } else {
            t -= 2.625f / 2.75f;
            return 7.5625f * t * t + 0.984375f;
        }
    }
    case EaseType::Elastic: {
        if (t == 0.0f || t == 1.0f)
            return t;
        float p = 0.3f;
        float s = p / 4.0f;
        return powf(2.0f, -10.0f * t) *
                   sinf((t - s) * (2.0f * 3.14159265f) / p) +
               1.0f;
    }
    }
    return t;
}

class TransformAnimation {
  public:
    std::vector<Keyframe<vec3>> positionKeys;
    std::vector<Keyframe<vec3>> rotationKeys;
    std::vector<Keyframe<vec3>> scaleKeys;

    bool looping = false;
    float duration = 0.0f;

    TransformAnimation &addPositionKey(float time, const vec3 &pos,
                                       EaseType ease = EaseType::Linear) {
        positionKeys.emplace_back(time, pos, ease);
        duration = fmaxf(duration, time);
        return *this;
    }

    TransformAnimation &addRotationKey(float time, const vec3 &rot,
                                       EaseType ease = EaseType::Linear) {
        rotationKeys.emplace_back(time, rot, ease);
        duration = fmaxf(duration, time);
        return *this;
    }

    TransformAnimation &addScaleKey(float time, const vec3 &scl,
                                    EaseType ease = EaseType::Linear) {
        scaleKeys.emplace_back(time, scl, ease);
        duration = fmaxf(duration, time);
        return *this;
    }

    // function set looping

    TransformAnimation &setLooping(bool loop) {
        looping = loop;
        return *this;
    }

    // function evaluate

    UnifiedTransform evaluate(float time) const {
        if (looping && duration > 0.0f) {
            time = fmodf(time, duration);
        }

        UnifiedTransform result;

        if (!positionKeys.empty()) {
            result.position = interpolateKeys(positionKeys, time);
        }

        if (!rotationKeys.empty()) {
            result.rotation = interpolateKeys(rotationKeys, time);
        }

        if (!scaleKeys.empty()) {
            result.scale = interpolateKeys(scaleKeys, time);
        }

        return result;
    }

  private:
    static vec3 interpolateKeys(const std::vector<Keyframe<vec3>> &keys,
                                float time) {
        if (keys.empty())
            return vec3(0.0f);
        if (keys.size() == 1 || time <= keys[0].time)
            return keys[0].value;
        if (time >= keys.back().time)
            return keys.back().value;

        size_t i = 0;
        for (; i < keys.size() - 1; ++i) {
            if (time < keys[i + 1].time)
                break;
        }

        const Keyframe<vec3> &k0 = keys[i];
        const Keyframe<vec3> &k1 = keys[i + 1];

        float t = (time - k0.time) / (k1.time - k0.time);
        t = applyEasing(t, k0.ease);

        return k0.value + (k1.value - k0.value) * t;
    }
};

class UnifiedScene {
  public:
    int width = 800;
    int height = 600;

    UnifiedCameraConfig camera;

    std::vector<UnifiedMeshDesc> meshes;
    std::vector<UnifiedLight> lights;

    UnifiedSkyConfig sky;
    vec3 ambientLight{0.03f, 0.03f, 0.03f};

    int bvhLeafTarget = 12;
    int bvhLeafTolerance = 5;

    int samplesPerPixel = 16;
    int maxBounceDepth = 8;

    std::unordered_map<std::string, TransformAnimation> animations;

    std::unordered_map<std::string, UnifiedMaterial> materialLibrary;

    bool transformsDirty = false;
    bool lightsDirty = false;
    bool materialsDirty = false;
    std::vector<bool> meshDirtyFlags;

    UnifiedScene &addLibraryMaterial(const std::string &name,
                                     const UnifiedMaterial &mat) {
        materialLibrary[name] = mat;

        materialsDirty = true;
        return *this;
    }

    // function get library material

    UnifiedMaterial *getLibraryMaterial(const std::string &name) {
        auto it = materialLibrary.find(name);
        if (it != materialLibrary.end())
            return &it->second;
        return nullptr;
    }

    UnifiedScene() = default;
    // function unified scene

    UnifiedScene(int w, int h) : width(w), height(h) {}

    UnifiedScene &setCamera(const vec3 &from, const vec3 &at, const vec3 &up,
                            float fov, float aperture = 0.0f,
                            float focusDist = 1.0f) {
        camera.lookfrom = from;
        camera.lookat = at;
        camera.vup = up;
        camera.vfov = fov;
        camera.aperture = aperture;
        camera.focusDist = focusDist;
        return *this;
    }

    // function set camera

    UnifiedScene &setCamera(const UnifiedCameraConfig &cam) {
        camera = cam;
        return *this;
    }

    // function add mesh

    ObjectHandle addMesh(const UnifiedMeshDesc &mesh) {
        size_t idx = meshes.size();
        meshes.push_back(mesh);
        meshDirtyFlags.push_back(true);
        return ObjectHandle(idx, this);
    }

    ObjectHandle instantiateObject(const UnifiedMeshDesc &meshDesc,
                                   const std::string &name = "") {
        UnifiedMeshDesc newMesh = meshDesc;
        if (!name.empty()) {
            newMesh.name = name;
        }

        newMesh.isDynamic = true;

        return addMesh(newMesh);
    }

    ObjectHandle
    addMeshFromOBJ(const std::string &path,

                   const UnifiedMaterial &mat = UnifiedMaterial()) {
        return addMesh(UnifiedMeshDesc::FromOBJ(path, mat));
    }

    // function add cube

    ObjectHandle addCube(const UnifiedMaterial &mat = UnifiedMaterial()) {
        return addMesh(UnifiedMeshDesc::Cube(mat));
    }

    ObjectHandle addPlaneXZ(float y, float halfSize,

                            const UnifiedMaterial &mat = UnifiedMaterial()) {
        return addMesh(UnifiedMeshDesc::PlaneXZ(y, halfSize, mat));
    }

    ObjectHandle addSphere(int segments = 32,

                           const UnifiedMaterial &mat = UnifiedMaterial()) {
        return addMesh(UnifiedMeshDesc::Sphere(segments, mat));
    }

    // function add light

    LightHandle addLight(const UnifiedLight &light) {
        size_t idx = lights.size();
        lights.push_back(light);
        lightsDirty = true;
        return LightHandle(idx, this);
    }

    LightHandle addPointLight(const vec3 &pos, const vec3 &color,
                              float intensity = 1.0f, float range = 100.0f,
                              float radius = 0.0f) {
        return addLight(
            UnifiedLight::Point(pos, color, intensity, range, radius));
    }

    LightHandle addDirectionalLight(const vec3 &dir, const vec3 &color,
                                    float intensity = 1.0f) {
        return addLight(UnifiedLight::Directional(dir, color, intensity));
    }

    LightHandle addSpotLight(const vec3 &pos, const vec3 &dir,
                             const vec3 &color, float intensity = 1.0f,
                             float innerCone = 0.5f, float outerCone = 0.7f,
                             float range = 100.0f, float radius = 0.0f) {
        return addLight(UnifiedLight::Spot(
            pos, dir, color, intensity, innerCone, outerCone, range, radius));
    }

    LightHandle addAreaLight(const vec3 &pos, const vec3 &dir,
                             const vec3 &color, float width, float height,
                             float intensity = 1.0f) {
        return addLight(
            UnifiedLight::Area(pos, dir, color, width, height, intensity));
    }

    // function set sky gradient

    UnifiedScene &setSkyGradient(const vec3 &top, const vec3 &bottom) {
        sky.enabled = true;
        sky.topColor = top;
        sky.bottomColor = bottom;
        sky.hdriPath.clear();
        return *this;
    }

    UnifiedScene &setHDRI(const std::string &path, float intensity = 1.0f,
                          float rotation = 0.0f) {
        sky.enabled = true;
        sky.hdriPath = path;
        sky.hdriIntensity = intensity;
        sky.hdriRotation = rotation;
        return *this;
    }

    // function disable sky

    UnifiedScene &disableSky() {
        sky.enabled = false;
        return *this;
    }

    // function set ambient light

    UnifiedScene &setAmbientLight(const vec3 &ambient) {
        ambientLight = ambient;
        return *this;
    }

    // function set bvh params

    UnifiedScene &setBVHParams(int leafTarget, int tolerance = 5) {
        bvhLeafTarget = leafTarget;
        bvhLeafTolerance = tolerance;
        return *this;
    }

    // function set path tracer params

    UnifiedScene &setPathTracerParams(int spp, int maxDepth) {
        samplesPerPixel = spp;
        maxBounceDepth = maxDepth;
        return *this;
    }

    // function find object

    ObjectHandle findObject(const std::string &name) {
        for (size_t i = 0; i < meshes.size(); ++i) {
            if (meshes[i].name == name) {
                return ObjectHandle(i, this);
            }
        }
        return ObjectHandle();
    }

    // function find light

    LightHandle findLight(const std::string &name) {
        for (size_t i = 0; i < lights.size(); ++i) {
            if (lights[i].name == name) {
                return LightHandle(i, this);
            }
        }
        return LightHandle();
    }

    UnifiedScene &addAnimation(const std::string &objectName,
                               const TransformAnimation &anim) {
        animations[objectName] = anim;
        return *this;
    }

    // function update animations

    void updateAnimations(float time) {
        for (const auto &[name, anim] : animations) {
            ObjectHandle obj = findObject(name);
            if (obj.isValid()) {
                UnifiedTransform newTransform = anim.evaluate(time);
                meshes[obj.index].transform = newTransform;
                markMeshDirty(obj.index);
            }
        }
    }

    // function mark mesh dirty

    void markMeshDirty(size_t index) {
        if (index < meshDirtyFlags.size()) {
            meshDirtyFlags[index] = true;
            transformsDirty = true;
        }
    }

    // function mark all meshes dirty

    void markAllMeshesDirty() {
        for (size_t i = 0; i < meshDirtyFlags.size(); ++i) {
            meshDirtyFlags[i] = true;
        }
        transformsDirty = true;
    }

    // function clear dirty flags

    void clearDirtyFlags() {
        for (size_t i = 0; i < meshDirtyFlags.size(); ++i) {
            meshDirtyFlags[i] = false;
        }
        transformsDirty = false;
        lightsDirty = false;
        materialsDirty = false;
    }

    // function has dirty meshes

    bool hasDirtyMeshes() const {
        for (bool dirty : meshDirtyFlags) {
            if (dirty)
                return true;
        }
        return false;
    }

    // function get dirty mesh indices

    std::vector<size_t> getDirtyMeshIndices() const {
        std::vector<size_t> indices;
        for (size_t i = 0; i < meshDirtyFlags.size(); ++i) {
            if (meshDirtyFlags[i]) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    // function get mesh count

    size_t getMeshCount() const { return meshes.size(); }
    // function get light count

    size_t getLightCount() const { return lights.size(); }
    // function get dynamic mesh count

    size_t getDynamicMeshCount() const {
        size_t count = 0;
        for (const auto &m : meshes) {
            if (m.isDynamic)
                ++count;
        }
        return count;
    }

    // function get mesh

    UnifiedMeshDesc *getMesh(size_t index) {
        return (index < meshes.size()) ? &meshes[index] : nullptr;
    }

    // function get mesh

    const UnifiedMeshDesc *getMesh(size_t index) const {
        return (index < meshes.size()) ? &meshes[index] : nullptr;
    }

    // function get light

    UnifiedLight *getLight(size_t index) {
        return (index < lights.size()) ? &lights[index] : nullptr;
    }

    // function get light

    const UnifiedLight *getLight(size_t index) const {
        return (index < lights.size()) ? &lights[index] : nullptr;
    }
};

// function set position

inline ObjectHandle &ObjectHandle::setPosition(const vec3 &pos) {
    if (isValid()) {
        scene->meshes[index].transform.setPosition(pos);
        scene->markMeshDirty(index);
    }
    return *this;
}

// function set rotation

inline ObjectHandle &ObjectHandle::setRotation(const vec3 &rot) {
    if (isValid()) {
        scene->meshes[index].transform.setRotation(rot);
        scene->markMeshDirty(index);
    }
    return *this;
}

// function set rotation degrees

inline ObjectHandle &ObjectHandle::setRotationDegrees(const vec3 &deg) {
    if (isValid()) {
        scene->meshes[index].transform.setRotationDegrees(deg);
        scene->markMeshDirty(index);
    }
    return *this;
}

// function set scale

inline ObjectHandle &ObjectHandle::setScale(const vec3 &s) {
    if (isValid()) {
        scene->meshes[index].transform.setScale(s);
        scene->markMeshDirty(index);
    }
    return *this;
}

// function set scale

inline ObjectHandle &ObjectHandle::setScale(float s) {
    if (isValid()) {
        scene->meshes[index].transform.setScale(s);
        scene->markMeshDirty(index);
    }
    return *this;
}

// function translate

inline ObjectHandle &ObjectHandle::translate(const vec3 &delta) {
    if (isValid()) {
        scene->meshes[index].transform.translate(delta);
        scene->markMeshDirty(index);
    }
    return *this;
}

// function rotate

inline ObjectHandle &ObjectHandle::rotate(const vec3 &deltaRad) {
    if (isValid()) {
        scene->meshes[index].transform.rotate(deltaRad);
        scene->markMeshDirty(index);
    }
    return *this;
}

// function rotate degrees

inline ObjectHandle &ObjectHandle::rotateDegrees(const vec3 &deltaDeg) {
    if (isValid()) {
        scene->meshes[index].transform.rotateDegrees(deltaDeg);
        scene->markMeshDirty(index);
    }
    return *this;
}

// function get transform

inline UnifiedTransform ObjectHandle::getTransform() const {
    if (isValid()) {
        return scene->meshes[index].transform;
    }
    return UnifiedTransform();
}

// function get position

inline vec3 ObjectHandle::getPosition() const {
    if (isValid()) {
        return scene->meshes[index].transform.position;
    }
    return vec3(0.0f);
}

// function set material

inline ObjectHandle &ObjectHandle::setMaterial(const UnifiedMaterial &mat) {
    if (isValid()) {
        scene->meshes[index].material = mat;
        scene->materialsDirty = true;
    }
    return *this;
}

inline ObjectHandle &
// function use library material

ObjectHandle::useLibraryMaterial(const std::string &materialName) {
    if (isValid()) {
        scene->meshes[index].materialRef = materialName;
        scene->materialsDirty = true;
    }
    return *this;
}

// function set name

inline ObjectHandle &ObjectHandle::setName(const std::string &name) {
    if (isValid()) {
        scene->meshes[index].name = name;
    }
    return *this;
}

// function set dynamic

inline ObjectHandle &ObjectHandle::setDynamic(bool dynamic) {
    if (isValid()) {
        scene->meshes[index].isDynamic = dynamic;
    }
    return *this;
}

// function set position

inline LightHandle &LightHandle::setPosition(const vec3 &pos) {
    if (isValid()) {
        scene->lights[index].position = pos;
        scene->lightsDirty = true;
    }
    return *this;
}

// function set direction

inline LightHandle &LightHandle::setDirection(const vec3 &dir) {
    if (isValid()) {
        scene->lights[index].direction = dir.normalized();
        scene->lightsDirty = true;
    }
    return *this;
}

// function set color

inline LightHandle &LightHandle::setColor(const vec3 &col) {
    if (isValid()) {
        scene->lights[index].color = col;
        scene->lightsDirty = true;
    }
    return *this;
}

// function set intensity

inline LightHandle &LightHandle::setIntensity(float i) {
    if (isValid()) {
        scene->lights[index].intensity = i;
        scene->lightsDirty = true;
    }
    return *this;
}

// function set name

inline LightHandle &LightHandle::setName(const std::string &name) {
    if (isValid()) {
        scene->lights[index].name = name;
    }
    return *this;
}

// function get position

inline vec3 LightHandle::getPosition() const {
    if (isValid()) {
        return scene->lights[index].position;
    }
    return vec3(0.0f);
}

#ifdef UNIFIED_SCENE_ENABLE_RT
#include "raytracer/RTscene.cuh"
#endif

#ifdef UNIFIED_SCENE_ENABLE_PT
#include "pathtracer/scene/scene.cuh"
#endif

namespace UnifiedSceneBuilder {

inline const UnifiedMaterial &resolveMaterial(const UnifiedScene &scene,
                                              const UnifiedMeshDesc &mesh) {
    if (!mesh.materialRef.empty()) {
        auto it = scene.materialLibrary.find(mesh.materialRef);
        if (it != scene.materialLibrary.end()) {
            return it->second;
        }
    }
    return mesh.material;
}

// function is scale collapsed

inline bool isScaleCollapsed(const vec3 &scale) {
    const float threshold = 0.0001f;
    return (scale.x < threshold || scale.y < threshold || scale.z < threshold);
}

inline void resetMeshToDefaultGeometry(Mesh *mesh, UnifiedMeshDesc::Type type,
                                       int sphereSegments = 32) {
    if (!mesh)
        return;

    switch (type) {
    case UnifiedMeshDesc::Type::Cube:

        mesh->vertices = {{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},
                          {0.5f, 0.5f, -0.5f},   {-0.5f, 0.5f, -0.5f},
                          {-0.5f, -0.5f, 0.5f},  {0.5f, -0.5f, 0.5f},
                          {0.5f, 0.5f, 0.5f},    {-0.5f, 0.5f, 0.5f}};
        mesh->faces = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
                       {0, 1, 5}, {0, 5, 4}, {3, 7, 6}, {3, 6, 2},
                       {0, 4, 7}, {0, 7, 3}, {1, 2, 6}, {1, 6, 5}};
        break;

    case UnifiedMeshDesc::Type::Sphere: {

        mesh->vertices.clear();
        mesh->faces.clear();

        const int rings = sphereSegments;
        const int sectors = sphereSegments;
        const float radius = 0.5f;

        for (int r = 0; r <= rings; ++r) {
            float phi = PI * float(r) / float(rings);
            float y = cosf(phi) * radius;
            float ringRadius = sinf(phi) * radius;

            for (int s = 0; s <= sectors; ++s) {
                float theta = TWO_PI * float(s) / float(sectors);
                float x = ringRadius * cosf(theta);
                float z = ringRadius * sinf(theta);
                mesh->vertices.push_back(vec3(x, y, z));
            }
        }

        for (int r = 0; r < rings; ++r) {
            for (int s = 0; s < sectors; ++s) {
                int curr = r * (sectors + 1) + s;
                int next = curr + sectors + 1;
                // function mesh->faces.push back

                mesh->faces.push_back({curr, next, curr + 1});
                // function mesh->faces.push back

                mesh->faces.push_back({curr + 1, next, next + 1});
            }
        }
        break;
    }

    case UnifiedMeshDesc::Type::PlaneXZ:

        mesh->vertices = {{-0.5f, 0.0f, -0.5f},
                          {0.5f, 0.0f, -0.5f},
                          {0.5f, 0.0f, 0.5f},
                          {-0.5f, 0.0f, 0.5f}};
        mesh->faces = {{0, 2, 1}, {0, 3, 2}};
        break;

    default:

        break;
    }

    mesh->bvhDirty = true;
}

#ifdef UNIFIED_SCENE_ENABLE_RT

// function to rt material

inline Material toRTMaterial(const UnifiedMaterial &um) {
    Material m;
    m.albedo = um.albedo;
    m.specular = um.specular;
    m.metallic = um.metallic;
    m.roughness = um.roughness;
    m.emission = um.emission;
    m.ior = um.ior;
    m.transmission = um.transmission;
    m.transmissionRoughness = um.transmissionRoughness;
    m.clearcoat = um.clearcoat;
    m.clearcoatRoughness = um.clearcoatRoughness;
    m.subsurfaceColor = um.subsurfaceColor;
    m.subsurfaceRadius = um.subsurfaceRadius;
    m.anisotropy = um.anisotropy;
    m.sheen = um.sheen;
    m.sheenTint = um.sheenTint;
    m.iridescence = um.iridescence;
    m.iridescenceThickness = um.iridescenceThickness;
    return m;
}

// function to rt light

inline Light toRTLight(const UnifiedLight &ul) {
    Light l;
    l.type = static_cast<LightType>(static_cast<int>(ul.type));
    l.position = ul.position;
    l.direction = ul.direction;
    l.color = ul.color;
    l.intensity = ul.intensity;
    l.range = ul.range;
    l.innerCone = ul.innerCone;
    l.outerCone = ul.outerCone;
    return l;
}

inline Mesh *addMeshToRTScene(Scene &scene, const UnifiedMeshDesc &meshDesc,
                              const UnifiedScene &unified) {
    Mesh *mesh = nullptr;
    Material mat = toRTMaterial(resolveMaterial(unified, meshDesc));

    switch (meshDesc.type) {
    case UnifiedMeshDesc::Type::ObjFile:
        mesh = scene.addMesh(meshDesc.objPath, mat);
        break;
    case UnifiedMeshDesc::Type::Cube:
        mesh = scene.addCube(mat);
        break;
    case UnifiedMeshDesc::Type::PlaneXZ:
        mesh = scene.addPlaneXZ(meshDesc.planeY, meshDesc.planeHalfSize, mat);
        break;
    case UnifiedMeshDesc::Type::Sphere:
        mesh = scene.addSphere(meshDesc.sphereSegments, mat);
        break;
    case UnifiedMeshDesc::Type::PlaneXY:
    case UnifiedMeshDesc::Type::PlaneYZ:
        break;
    case UnifiedMeshDesc::Type::Triangles: {
        // Convert triangleVerts (every 3 consecutive vec3s form a triangle) to
        // Triangle objects
        const auto &verts = meshDesc.triangleVerts;
        if (verts.size() >= 3) {
            std::vector<Triangle> tris;
            tris.reserve(verts.size() / 3);
            for (size_t i = 0; i + 2 < verts.size(); i += 3) {
                tris.emplace_back(verts[i], verts[i + 1], verts[i + 2]);
            }
            mesh = scene.addTriangles(tris, mat);
        }
        break;
    }
    }
    return mesh;
}

// function build rt scene

inline std::unique_ptr<Scene> buildRTScene(UnifiedScene &unified) {
    auto scene = std::make_unique<Scene>(unified.width, unified.height);

    scene->setCamera(unified.camera.lookfrom, unified.camera.lookat,
                     unified.camera.vup, unified.camera.vfov,
                     unified.camera.aperture, unified.camera.focusDist);

    scene->setBVHLeafTarget(unified.bvhLeafTarget, unified.bvhLeafTolerance);

    for (size_t i = 0; i < unified.meshes.size(); ++i) {
        const auto &meshDesc = unified.meshes[i];
        Mesh *mesh = addMeshToRTScene(*scene, meshDesc, unified);

        if (mesh) {
            const UnifiedTransform &t = meshDesc.transform;

            if (meshDesc.isDynamic) {
                AABB box = mesh->boundingBox();
                vec3 center = box.center();
                if (center.length_squared() > 1e-10f) {
                    mesh->translate(vec3(-center.x, -center.y, -center.z));
                }

                if (t.scale.x != 1.0f || t.scale.y != 1.0f ||
                    t.scale.z != 1.0f) {
                    mesh->scale(t.scale);
                }

                mesh->setPosition(t.position);
                mesh->setRotation(t.rotation);
            } else {
                if (t.scale.x != 1.0f || t.scale.y != 1.0f ||
                    t.scale.z != 1.0f) {
                    mesh->scale(t.scale);
                }
                if (t.rotation.x != 0.0f || t.rotation.y != 0.0f ||
                    t.rotation.z != 0.0f) {
                    mesh->rotateSelfEulerXYZ(t.rotation);
                }
                if (t.position.x != 0.0f || t.position.y != 0.0f ||
                    t.position.z != 0.0f) {
                    mesh->moveTo(t.position);
                }

                mesh->setPosition(vec3(0.0f));
                mesh->setRotation(vec3(0.0f));
            }

            unified.meshes[i].bakedTransform = t;
        }
    }

    for (const auto &light : unified.lights) {
        switch (light.type) {
        case UnifiedLightType::Point:
            scene->addPointLight(light.position, light.color, light.intensity,
                                 light.range);
            break;
        case UnifiedLightType::Directional:
            scene->addDirectionalLight(light.direction, light.color,
                                       light.intensity);
            break;
        case UnifiedLightType::Spot:
            scene->addSpotLight(light.position, light.direction, light.color,
                                light.intensity, light.innerCone,
                                light.outerCone, light.range);
            break;
        case UnifiedLightType::Area:

            scene->addPointLight(light.position, light.color, light.intensity,
                                 light.range);
            break;
        }
    }

    if (unified.sky.enabled) {
        scene->setSkyGradient(unified.sky.topColor, unified.sky.bottomColor);
    } else {
        scene->disableSky();
    }

    scene->setAmbientLight(unified.ambientLight);

    unified.clearDirtyFlags();
    return scene;
}

// function update rt scene

inline void updateRTScene(Scene &scene, UnifiedScene &unified) {
    // function update r t scene
    // purpose push dirty unified scene changes into an already built ray tracer
    // scene inputs renderer scene reference and unified scene reference outputs
    // updated renderer state including meshes materials and camera notes
    // dynamic meshes update position and rotation through per mesh transform
    // fields while static meshes bake transforms into vertices

    size_t currentRenderCount = scene.getMeshCount();
    size_t targetCount = unified.meshes.size();

    if (targetCount > currentRenderCount) {
        for (size_t i = currentRenderCount; i < targetCount; ++i) {
            const auto &meshDesc = unified.meshes[i];
            Mesh *mesh = addMeshToRTScene(scene, meshDesc, unified);
            if (!mesh)
                continue;

            const UnifiedTransform &t = meshDesc.transform;

            if (meshDesc.isDynamic) {
                AABB box = mesh->boundingBox();
                vec3 center = box.center();
                if (center.length_squared() > 1e-10f) {
                    mesh->translate(vec3(-center.x, -center.y, -center.z));
                }

                if (t.scale.x != 1.0f || t.scale.y != 1.0f || t.scale.z != 1.0f)
                    mesh->scale(t.scale);

                mesh->setPosition(t.position);
                mesh->setRotation(t.rotation);
            } else {
                if (t.scale.x != 1.0f || t.scale.y != 1.0f || t.scale.z != 1.0f)
                    mesh->scale(t.scale);
                if (t.rotation.x != 0.0f || t.rotation.y != 0.0f ||
                    t.rotation.z != 0.0f)
                    mesh->rotateSelfEulerXYZ(t.rotation);
                if (t.position.x != 0.0f || t.position.y != 0.0f ||
                    t.position.z != 0.0f)
                    mesh->moveTo(t.position);

                mesh->setPosition(vec3(0.0f));
                mesh->setRotation(vec3(0.0f));
            }

            unified.meshes[i].bakedTransform = t;
        }
    }

    auto dirtyIndices = unified.getDirtyMeshIndices();

    for (size_t idx : dirtyIndices) {
        if (idx >= unified.meshes.size())
            continue;
        if (idx >= scene.getMeshCount())
            continue;

        auto &meshDesc = unified.meshes[idx];
        Mesh *mesh = scene.getMesh(idx);
        if (!mesh)
            continue;

        // Handle Triangles type meshes - update vertices from triangleVerts
        if (meshDesc.type == UnifiedMeshDesc::Type::Triangles) {
            const auto &verts = meshDesc.triangleVerts;
            if (verts.size() >= 3) {
                mesh->vertices.clear();
                mesh->faces.clear();
                mesh->vertices.reserve(verts.size());
                mesh->faces.reserve(verts.size() / 3);
                for (size_t i = 0; i + 2 < verts.size(); i += 3) {
                    int base = static_cast<int>(mesh->vertices.size());
                    mesh->vertices.push_back(verts[i]);
                    mesh->vertices.push_back(verts[i + 1]);
                    mesh->vertices.push_back(verts[i + 2]);
                    mesh->faces.push_back(Tri{base, base + 1, base + 2});
                }
                mesh->bvhDirty = true;
                mesh->vertsDirty = true;
                mesh->computeLocalAABB();
            }
            continue; // Skip transform updates for Triangles type - vertices
                      // are in world space
        }

        const UnifiedTransform &newT = meshDesc.transform;
        const UnifiedTransform &oldT = meshDesc.bakedTransform;

        if (meshDesc.isDynamic) {
            vec3 scaleRatio(
                (oldT.scale.x > 0.0001f) ? (newT.scale.x / oldT.scale.x) : 1.0f,
                (oldT.scale.y > 0.0001f) ? (newT.scale.y / oldT.scale.y) : 1.0f,
                (oldT.scale.z > 0.0001f) ? (newT.scale.z / oldT.scale.z)
                                         : 1.0f);

            if (fabsf(scaleRatio.x - 1.0f) > 1e-4f ||
                fabsf(scaleRatio.y - 1.0f) > 1e-4f ||
                // function fabsf

                fabsf(scaleRatio.z - 1.0f) > 1e-4f) {
                mesh->scale(scaleRatio);
            }

            mesh->setPosition(newT.position);
            mesh->setRotation(newT.rotation);

            meshDesc.bakedTransform = newT;
            continue;
        }

        if (oldT.scale.x == 0.0f && oldT.scale.y == 0.0f &&
            oldT.scale.z == 0.0f) {
            if (newT.scale.x != 1.0f || newT.scale.y != 1.0f ||
                newT.scale.z != 1.0f)
                mesh->scale(newT.scale);
            if (newT.rotation.x != 0.0f || newT.rotation.y != 0.0f ||
                newT.rotation.z != 0.0f)
                mesh->rotateSelfEulerXYZ(newT.rotation);
            if (newT.position.x != 0.0f || newT.position.y != 0.0f ||
                newT.position.z != 0.0f)
                mesh->moveTo(newT.position);

            mesh->setPosition(vec3(0.0f));
            mesh->setRotation(vec3(0.0f));

            meshDesc.bakedTransform = newT;
            continue;
        }

        vec3 rotDelta = newT.rotation - oldT.rotation;
        if (rotDelta.length_squared() > 1e-6f)
            mesh->rotateSelfEulerXYZ(rotDelta);

        vec3 scaleRatio(
            (oldT.scale.x > 0.0001f) ? (newT.scale.x / oldT.scale.x) : 1.0f,
            (oldT.scale.y > 0.0001f) ? (newT.scale.y / oldT.scale.y) : 1.0f,
            (oldT.scale.z > 0.0001f) ? (newT.scale.z / oldT.scale.z) : 1.0f);

        if (fabsf(scaleRatio.x - 1.0f) > 1e-4f ||
            fabsf(scaleRatio.y - 1.0f) > 1e-4f ||
            // function fabsf

            fabsf(scaleRatio.z - 1.0f) > 1e-4f) {
            mesh->scale(scaleRatio);
        }

        vec3 posDelta = newT.position - oldT.position;
        if (posDelta.length_squared() > 1e-6f)
            mesh->moveTo(newT.position);

        mesh->setPosition(vec3(0.0f));
        mesh->setRotation(vec3(0.0f));

        meshDesc.bakedTransform = newT;
    }

    if (unified.materialsDirty) {
        for (size_t i = 0; i < unified.meshes.size(); ++i) {
            if (i < scene.getMeshCount()) {
                scene.setMeshMaterial(i, toRTMaterial(resolveMaterial(
                                             unified, unified.meshes[i])));
            }
        }
    }

    unified.clearDirtyFlags();
}

// function update rt camera

inline void updateRTCamera(Scene &scene, const UnifiedScene &unified) {
    scene.setCamera(unified.camera.lookfrom, unified.camera.lookat,
                    unified.camera.vup, unified.camera.vfov,
                    unified.camera.aperture, unified.camera.focusDist);
}

#endif

#ifdef UNIFIED_SCENE_ENABLE_PT

// function to pt material

inline Material toPTMaterial(const UnifiedMaterial &um) {
    Material m;
    m.albedo = um.albedo;
    m.specular = um.specular;
    m.metallic = um.metallic;
    m.roughness = um.roughness;
    m.emission = um.emission;
    m.ior = um.ior;
    m.transmission = um.transmission;
    m.transmissionRoughness = um.transmissionRoughness;
    m.clearcoat = um.clearcoat;
    m.clearcoatRoughness = um.clearcoatRoughness;
    m.subsurfaceColor = um.subsurfaceColor;
    m.subsurfaceRadius = um.subsurfaceRadius;
    m.anisotropy = um.anisotropy;
    m.sheen = um.sheen;
    m.sheenTint = um.sheenTint;
    m.iridescence = um.iridescence;
    m.iridescenceThickness = um.iridescenceThickness;
    return m;
}

// function to pt light

inline Light toPTLight(const UnifiedLight &ul) {
    Light l{};
    l.type = static_cast<LightType>(static_cast<int>(ul.type));
    l.position = ul.position;
    l.direction = ul.direction.normalized(); // always normalize
    l.color = ul.color;
    l.intensity = ul.intensity;
    l.range = ul.range;
    l.radius = ul.radius;

    // PT scene stores cos(coneAngle)
    l.innerCone = cosf(ul.innerCone);
    l.outerCone = cosf(ul.outerCone);

    return l;
}

inline Mesh *addMeshToPTScene(Scene &scene, const UnifiedMeshDesc &meshDesc,
                              const UnifiedScene &unified) {
    Mesh *mesh = nullptr;
    Material mat = toPTMaterial(resolveMaterial(unified, meshDesc));

    switch (meshDesc.type) {
    case UnifiedMeshDesc::Type::ObjFile:
        mesh = scene.addMesh(meshDesc.objPath, mat);
        break;
    case UnifiedMeshDesc::Type::Cube:
        mesh = scene.addCube(mat);
        break;
    case UnifiedMeshDesc::Type::PlaneXZ:
        mesh = scene.addPlaneXZ(meshDesc.planeY, meshDesc.planeHalfSize, mat);
        break;
    case UnifiedMeshDesc::Type::Sphere:
        mesh = scene.addSphere(meshDesc.sphereSegments, mat);
        break;
    case UnifiedMeshDesc::Type::PlaneXY:
    case UnifiedMeshDesc::Type::PlaneYZ:
        break;
    case UnifiedMeshDesc::Type::Triangles: {
        // Convert triangleVerts (every 3 consecutive vec3s form a triangle) to
        // Triangle objects
        const auto &verts = meshDesc.triangleVerts;
        if (verts.size() >= 3) {
            std::vector<Triangle> tris;
            tris.reserve(verts.size() / 3);
            for (size_t i = 0; i + 2 < verts.size(); i += 3) {
                tris.emplace_back(verts[i], verts[i + 1], verts[i + 2]);
            }
            mesh = scene.addTriangles(tris, mat);
        }
        break;
    }
    }
    return mesh;
}

// function build pt scene

inline std::unique_ptr<Scene> buildPTScene(UnifiedScene &unified) {
    auto scene = std::make_unique<Scene>(unified.width, unified.height);

    scene->setCamera(unified.camera.lookfrom, unified.camera.lookat,
                     unified.camera.vup, unified.camera.vfov,
                     unified.camera.aperture, unified.camera.focusDist);

    scene->setBVHLeafTarget(unified.bvhLeafTarget, unified.bvhLeafTolerance);

    for (size_t i = 0; i < unified.meshes.size(); ++i) {
        const auto &meshDesc = unified.meshes[i];
        Mesh *mesh = addMeshToPTScene(*scene, meshDesc, unified);

        if (mesh) {
            const UnifiedTransform &t = meshDesc.transform;

            if (meshDesc.isDynamic) {
                mesh->transform.setPosition(t.position);
                mesh->transform.setRotation(t.rotation);
                mesh->transform.setScale(t.scale);
                mesh->transform.updateMatrices();
            } else {

                if (t.scale.x != 1.0f || t.scale.y != 1.0f ||
                    t.scale.z != 1.0f) {
                    mesh->scale(t.scale);
                }
                if (t.rotation.x != 0.0f || t.rotation.y != 0.0f ||
                    t.rotation.z != 0.0f) {
                    mesh->rotateSelfEulerXYZ(t.rotation);
                }
                if (t.position.x != 0.0f || t.position.y != 0.0f ||
                    t.position.z != 0.0f) {
                    mesh->moveTo(t.position);
                }

                unified.meshes[i].bakedTransform = t;
            }

            mesh->computeLocalAABB();
        }
    }

    for (const auto &light : unified.lights) {
        switch (light.type) {
        case UnifiedLightType::Point:
            scene->addPointLight(light.position, light.color, light.intensity,
                                 light.range, light.radius);
            break;
        case UnifiedLightType::Directional:
            scene->addDirectionalLight(light.direction, light.color,
                                       light.intensity);
            break;
        case UnifiedLightType::Spot:
            scene->addSpotLight(light.position, light.direction, light.color,
                                light.intensity, light.innerCone,
                                light.outerCone, light.range, light.radius);
            break;
        case UnifiedLightType::Area:

            scene->addPointLight(light.position, light.color, light.intensity,
                                 light.range,
                                 fmaxf(light.areaWidth, light.areaHeight));
            break;
        }
    }

    if (unified.sky.enabled) {
        if (!unified.sky.hdriPath.empty()) {
            scene->loadHDRI(unified.sky.hdriPath);
        } else {
            scene->setSkyGradient(unified.sky.topColor,
                                  unified.sky.bottomColor);
        }
    } else {
        scene->disableSky();
    }

    unified.clearDirtyFlags();
    return scene;
}

// function update pt scene

inline void updatePTScene(Scene &scene, UnifiedScene &unified) {

    size_t currentRenderCount = scene.getMeshCount();
    size_t targetCount = unified.meshes.size();

    if (targetCount > currentRenderCount) {
        for (size_t i = currentRenderCount; i < targetCount; ++i) {
            const auto &meshDesc = unified.meshes[i];
            Mesh *mesh = addMeshToPTScene(scene, meshDesc, unified);

            if (mesh) {
                const UnifiedTransform &t = meshDesc.transform;
                if (meshDesc.isDynamic) {
                    mesh->transform.setPosition(t.position);
                    mesh->transform.setRotation(t.rotation);
                    mesh->transform.setScale(t.scale);
                    mesh->transform.updateMatrices();
                } else {
                    if (t.scale.x != 1.0f || t.scale.y != 1.0f ||
                        t.scale.z != 1.0f)
                        mesh->scale(t.scale);
                    if (t.rotation.x != 0.0f || t.rotation.y != 0.0f ||
                        t.rotation.z != 0.0f)
                        mesh->rotateSelfEulerXYZ(t.rotation);
                    if (t.position.x != 0.0f || t.position.y != 0.0f ||
                        t.position.z != 0.0f)
                        mesh->moveTo(t.position);
                    unified.meshes[i].bakedTransform = t;
                }
                mesh->computeLocalAABB();
            }
        }
    }

    auto dirtyIndices = unified.getDirtyMeshIndices();

    for (size_t idx : dirtyIndices) {
        if (idx >= currentRenderCount)
            continue;

        auto &meshDesc = unified.meshes[idx];
        Mesh *mesh = scene.getMesh(idx);
        if (!mesh)
            continue;

        // Handle Triangles type meshes - update vertices from triangleVerts
        if (meshDesc.type == UnifiedMeshDesc::Type::Triangles) {
            const auto &verts = meshDesc.triangleVerts;
            if (verts.size() >= 3) {
                mesh->vertices.clear();
                mesh->faces.clear();
                mesh->vertices.reserve(verts.size());
                mesh->faces.reserve(verts.size() / 3);
                for (size_t i = 0; i + 2 < verts.size(); i += 3) {
                    int base = static_cast<int>(mesh->vertices.size());
                    mesh->vertices.push_back(verts[i]);
                    mesh->vertices.push_back(verts[i + 1]);
                    mesh->vertices.push_back(verts[i + 2]);
                    mesh->faces.push_back(Tri{base, base + 1, base + 2});
                }
                mesh->bvhDirty = true;
                mesh->vertsDirty = true;
                mesh->computeLocalAABB();
            }
            continue; // Skip transform updates for Triangles type - vertices
                      // are in world space
        }

        const UnifiedTransform &newT = meshDesc.transform;

        if (meshDesc.isDynamic) {
            mesh->transform.setPosition(newT.position);
            mesh->transform.setRotation(newT.rotation);
            mesh->transform.setScale(newT.scale);
            mesh->transform.updateMatrices();
        } else {
            const UnifiedTransform &oldT = meshDesc.bakedTransform;
            bool wasHidden = isScaleCollapsed(oldT.scale);
            bool nowVisible = !isScaleCollapsed(newT.scale);

            if (wasHidden && nowVisible) {
                resetMeshToDefaultGeometry(mesh, meshDesc.type,
                                           meshDesc.sphereSegments);
                if (newT.scale.x != 1.0f || newT.scale.y != 1.0f ||
                    newT.scale.z != 1.0f)
                    mesh->scale(newT.scale);
                if (newT.rotation.x != 0.0f || newT.rotation.y != 0.0f ||
                    newT.rotation.z != 0.0f)
                    mesh->rotateSelfEulerXYZ(newT.rotation);
                if (newT.position.x != 0.0f || newT.position.y != 0.0f ||
                    newT.position.z != 0.0f)
                    mesh->moveTo(newT.position);
            } else {
                vec3 rotDelta = newT.rotation - oldT.rotation;
                if (rotDelta.length_squared() > 1e-6f)
                    mesh->rotateSelfEulerXYZ(rotDelta);

                vec3 scaleRatio(
                    (oldT.scale.x > 0.0001f) ? (newT.scale.x / oldT.scale.x)
                                             : 1.0f,
                    (oldT.scale.y > 0.0001f) ? (newT.scale.y / oldT.scale.y)
                                             : 1.0f,
                    (oldT.scale.z > 0.0001f) ? (newT.scale.z / oldT.scale.z)
                                             : 1.0f);
                if (fabsf(scaleRatio.x - 1.0f) > 1e-4f ||
                    fabsf(scaleRatio.y - 1.0f) > 1e-4f ||
                    // function fabsf

                    fabsf(scaleRatio.z - 1.0f) > 1e-4f) {
                    mesh->scale(scaleRatio);
                }

                vec3 posDelta = newT.position - oldT.position;
                if (posDelta.length_squared() > 1e-6f)
                    mesh->moveTo(newT.position);
            }
            meshDesc.bakedTransform = newT;
            mesh->computeLocalAABB();
        }
    }

    if (unified.materialsDirty) {
        for (size_t i = 0; i < unified.meshes.size(); ++i) {
            if (i < scene.getMeshCount()) {
                scene.setMeshMaterial(i, toPTMaterial(resolveMaterial(
                                             unified, unified.meshes[i])));
            }
        }

        scene.commitMaterialChanges();
    }

    if (unified.lightsDirty) {
        const size_t ptCount = scene.getLightCount();
        const size_t uCount = unified.lights.size();

        // 1) If new lights were added, create them in the PT Scene.
        //    (PT Scene has no addAreaLight, so treat Area as Point for now.)
        if (uCount > ptCount) {
            for (size_t i = ptCount; i < uCount; ++i) {
                const auto &ul = unified.lights[i];
                switch (ul.type) {
                case UnifiedLightType::Point:
                    scene.addPointLight(ul.position, ul.color, ul.intensity,
                                        ul.range, ul.radius);
                    break;
                case UnifiedLightType::Directional:
                    scene.addDirectionalLight(ul.direction, ul.color,
                                              ul.intensity);
                    break;
                case UnifiedLightType::Spot:
                    scene.addSpotLight(ul.position, ul.direction, ul.color,
                                       ul.intensity, ul.innerCone, ul.outerCone,
                                       ul.range, ul.radius);
                    break;
                case UnifiedLightType::Area:
                    // not supported in PT scene right now
                    scene.addPointLight(ul.position, ul.color, ul.intensity,
                                        ul.range, ul.radius);
                    break;
                }
            }
            scene.commitObjectChanges();
        }

        // 2) Update properties for existing lights (not just position)
        for (size_t i = 0; i < uCount && i < scene.getLightCount(); ++i) {
            if (Light *pl = scene.getLight(i)) {
                *pl = toPTLight(unified.lights[i]); // overwrite everything
            }
        }

        // 3) Upload to GPU
        scene.commitLightChanges();
    }

    if (unified.hasDirtyMeshes() || targetCount > currentRenderCount) {
        scene.commitObjectChanges();
    }

    unified.clearDirtyFlags();
}

// function update pt camera

inline void updatePTCamera(Scene &scene, const UnifiedScene &unified) {
    scene.setCamera(unified.camera.lookfrom, unified.camera.lookat,
                    unified.camera.vup, unified.camera.vfov,
                    unified.camera.aperture, unified.camera.focusDist);
}

#endif

} // namespace UnifiedSceneBuilder

namespace UnifiedScenePresets {

// function cornell box

inline UnifiedScene CornellBox(int width = 800, int height = 800) {
    UnifiedScene scene(width, height);

    scene.setCamera(vec3(278, 273, -800), vec3(278, 273, 0), vec3(0, 1, 0),
                    40.0f);

    UnifiedMaterial white(vec3(0.73f, 0.73f, 0.73f), 0.9f, 0.0f);
    UnifiedMaterial red(vec3(0.65f, 0.05f, 0.05f), 0.9f, 0.0f);
    UnifiedMaterial green(vec3(0.12f, 0.45f, 0.15f), 0.9f, 0.0f);

    scene.addPlaneXZ(0, 278, white);
    scene.addPlaneXZ(548.8f, 278, white);

    scene.addPointLight(vec3(278, 530, 279.5f), vec3(1.0f), 50.0f);
    scene.setSkyGradient(vec3(0.0f), vec3(0.0f));

    return scene;
}

// function material showcase

inline UnifiedScene MaterialShowcase(int width = 1280, int height = 720) {
    UnifiedScene scene(width, height);

    scene.setCamera(vec3(0, 5, 15), vec3(0, 0, 0), vec3(0, 1, 0), 45.0f);

    scene.addPlaneXZ(-1.0f, 50.0f, UnifiedMaterial::MarbleCarrara());

    scene.addDirectionalLight(vec3(-0.5f, -1.0f, -0.3f),
                              vec3(1.0f, 0.95f, 0.9f), 2.0f);
    scene.addPointLight(vec3(5, 8, 5), vec3(1.0f, 0.9f, 0.8f), 100.0f);

    scene.setSkyGradient(vec3(0.6f, 0.7f, 1.0f), vec3(1.0f, 1.0f, 1.0f));

    return scene;
}

// function empty

inline UnifiedScene Empty(int width = 800, int height = 600) {
    UnifiedScene scene(width, height);

    scene.setCamera(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0), 60.0f);
    scene.addPlaneXZ(-1.0f, 50.0f, UnifiedMaterial(vec3(0.8f)));
    scene.addDirectionalLight(vec3(-0.5f, -1.0f, -0.3f), vec3(1.0f), 1.0f);
    scene.setSkyGradient(vec3(0.6f, 0.7f, 1.0f), vec3(1.0f, 1.0f, 1.0f));

    return scene;
}

// function dynamic objects demo

inline UnifiedScene DynamicObjectsDemo(int width = 1280, int height = 720) {
    UnifiedScene scene(width, height);

    scene.setCamera(vec3(0, 8, 15), vec3(0, 2, 0), vec3(0, 1, 0), 50.0f);

    scene.addPlaneXZ(-0.5f, 30.0f, UnifiedMaterial::Concrete());

    for (int i = 0; i < 5; ++i) {
        auto cube = scene.addCube(UnifiedMaterial::PlasticRed());
        cube.setPosition(vec3((i - 2) * 3.0f, 1.0f, 0.0f))
            .setScale(1.5f)
            .setName("cube_" + std::to_string(i));
        scene.meshes[cube.index].isDynamic = true;
    }

    auto sphere = scene.addSphere(32, UnifiedMaterial::Chrome());
    sphere.setPosition(vec3(0, 3, 5)).setScale(1.0f).setName("sphere_main");
    scene.meshes[sphere.index].isDynamic = true;

    TransformAnimation sphereAnim;
    sphereAnim.addPositionKey(0.0f, vec3(0, 3, 5))
        .addPositionKey(2.0f, vec3(5, 5, 5), EaseType::EaseInOut)
        .addPositionKey(4.0f, vec3(0, 3, -5), EaseType::EaseInOut)
        .addPositionKey(6.0f, vec3(-5, 5, 5), EaseType::EaseInOut)
        .addPositionKey(8.0f, vec3(0, 3, 5), EaseType::EaseInOut)
        .setLooping(true);
    scene.addAnimation("sphere_main", sphereAnim);

    auto light =
        scene.addPointLight(vec3(5, 10, 5), vec3(1.0f, 0.9f, 0.8f), 200.0f);
    light.setName("main_light");
    scene.lights[light.index].animated = true;

    scene.addDirectionalLight(vec3(-0.3f, -1.0f, -0.5f), vec3(0.5f, 0.6f, 0.8f),
                              0.5f);

    scene.setSkyGradient(vec3(0.4f, 0.5f, 0.8f), vec3(0.9f, 0.9f, 1.0f));

    return scene;
}

// function glass demo

inline UnifiedScene GlassDemo(int width = 1280, int height = 720) {
    UnifiedScene scene(width, height);

    scene.setCamera(vec3(0, 4, 12), vec3(0, 1.5f, 0), vec3(0, 1, 0), 45.0f);

    scene.addPlaneXZ(0.0f, 20.0f, UnifiedMaterial::MarbleCarrara());

    scene.addSphere(64, UnifiedMaterial::Glass())
        .setPosition(vec3(0, 2, 0))
        .setScale(2.0f)
        .setName("glass_sphere");

    scene.addSphere(64, UnifiedMaterial::Diamond())
        .setPosition(vec3(-4, 1.5f, 0))
        .setScale(1.5f)
        .setName("diamond");

    scene.addSphere(48, UnifiedMaterial::Water())
        .setPosition(vec3(4, 1.5f, 0))
        .setScale(1.5f)
        .setName("water");

    scene.addCube(UnifiedMaterial::FrostedGlass())
        .setPosition(vec3(0, 1, -4))
        .setScale(vec3(6, 2, 0.3f))
        .setName("frosted_panel");

    scene.addPointLight(vec3(5, 10, 5), vec3(1.0f), 150.0f, 50.0f, 0.5f);
    scene.addPointLight(vec3(-5, 8, -3), vec3(0.9f, 0.9f, 1.0f), 100.0f);
    scene.addDirectionalLight(vec3(-0.5f, -1.0f, 0.2f), vec3(1.0f, 0.95f, 0.9f),
                              1.0f);

    scene.setSkyGradient(vec3(0.7f, 0.8f, 1.0f), vec3(1.0f, 1.0f, 1.0f));

    return scene;
}

// function metal demo

inline UnifiedScene MetalDemo(int width = 1280, int height = 720) {
    UnifiedScene scene(width, height);

    scene.setCamera(vec3(0, 6, 14), vec3(0, 2, 0), vec3(0, 1, 0), 45.0f);

    scene.addPlaneXZ(0.0f, 30.0f, UnifiedMaterial::MarbleNero());

    scene.addSphere(48, UnifiedMaterial::Gold())
        .setPosition(vec3(-6, 1.5f, 0))
        .setScale(1.5f);

    scene.addSphere(48, UnifiedMaterial::Silver())
        .setPosition(vec3(-3, 1.5f, 0))
        .setScale(1.5f);

    scene.addSphere(48, UnifiedMaterial::Copper())
        .setPosition(vec3(0, 1.5f, 0))
        .setScale(1.5f);

    scene.addSphere(48, UnifiedMaterial::Chrome())
        .setPosition(vec3(3, 1.5f, 0))
        .setScale(1.5f);

    scene.addSphere(48, UnifiedMaterial::BrushedAluminum())
        .setPosition(vec3(6, 1.5f, 0))
        .setScale(1.5f);

    scene.addCube(UnifiedMaterial::CarPaint(vec3(0.8f, 0.1f, 0.1f)))
        .setPosition(vec3(-2, 1, 4))
        .setScale(2.0f)
        .setRotationDegrees(vec3(0, 30, 0));

    scene.addCube(UnifiedMaterial::PearlescentPaint(vec3(0.1f, 0.2f, 0.8f)))
        .setPosition(vec3(2, 1, 4))
        .setScale(2.0f)
        .setRotationDegrees(vec3(0, -30, 0));

    scene.addPointLight(vec3(0, 12, 8), vec3(1.0f), 300.0f);
    scene.addPointLight(vec3(-8, 8, -5), vec3(0.9f, 0.95f, 1.0f), 150.0f);
    scene.addPointLight(vec3(8, 8, -5), vec3(1.0f, 0.95f, 0.9f), 150.0f);

    scene.setSkyGradient(vec3(0.2f, 0.2f, 0.3f), vec3(0.5f, 0.5f, 0.6f));

    return scene;
}

} // namespace UnifiedScenePresets

#endif
