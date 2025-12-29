#ifndef APP_UTILS_CUH_
#define APP_UTILS_CUH_

// Includes
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Project Includes
#include "pathtracer/scene/material_lib.cuh"
#include "pathtracer/scene/scene.cuh"

#define GLFW_INCLUDE_NONE
#include "common/glfw_view_interop.hpp"

// CUDA Check Macro
#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t err = (stmt);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Config Struct

struct RenderConfig {
    int sceneId = 0;
    int width = 800;
    int height = 600;
    std::string outputName = "output";
    bool showHelp = false;
    bool headlessOnce = false;
    int bvhLeafTarget = 12;
    int bvhLeafTol = 5;
};

// Materials Helper

struct Materials {
    // METALS
    static Material Silver() {
        return Material(vec3(0.97, 0.96, 0.91), 0.05f, 1.0f);
    }
    static Material BrushedAluminum() {
        return Material(vec3(0.91, 0.92, 0.92), 0.3f, 1.0f);
    }
    static Material Gold() {
        return Material(vec3(1.00, 0.78, 0.34), 0.1f, 1.0f);
    }
    static Material Copper() {
        return Material(vec3(0.95, 0.64, 0.54), 0.2f, 1.0f);
    }
    static Material Titanium() {
        return Material(vec3(0.542, 0.497, 0.449), 0.15f, 1.0f);
    }

    // DIELECTRICS / GLASS
    static Material Glass() {
        Material m(vec3(1.0f), 0.0f);
        m.transmission = 1.0f;
        m.ior = 1.5f;
        m.specular = vec3(iorToF0(1.5f));
        return m;
    }
    static Material FrostedGlass() {
        Material m = Glass();
        m.roughness = 0.2f;
        return m;
    }
    static Material Water() {
        Material m = Glass();
        m.ior = 1.33f;
        return m;
    }
    static Material Diamond() {
        Material m = Glass();
        m.ior = 2.417f;
        m.specular = vec3(iorToF0(2.417f));
        return m;
    }

    // IRIDESCENCE & THIN FILMS
    static Material SoapBubble() {
        Material m(vec3(1.0f), 0.0f);
        m.transmission = 0.95f;
        m.ior = 1.01f; // Very thin air interface
        m.iridescence = 1.0f;
        m.iridescenceThickness = 400.0f; // Nanometers
        return m;
    }

    static Material OilSlick() {
        Material m(vec3(0.1f), 0.4f, 0.8f); // Dark metallic base
        m.iridescence = 1.0f;
        m.iridescenceThickness = 600.0f;
        return m;
    }

    // FABRICS (SHEEN)
    static Material VelvetRed() {
        Material m(vec3(0.4f, 0.01f, 0.05f), 0.8f); // Dark red base
        m.sheen = 1.0f;
        m.sheenTint = vec3(1.0f, 0.5f, 0.5f); // Pinkish sheen
        return m;
    }

    static Material SatinBlue() {
        Material m(vec3(0.1f, 0.1f, 0.6f), 0.3f);
        m.sheen = 0.8f;
        m.anisotropy = 0.6f; // Directional highlight
        return m;
    }

    // CLEARCOAT
    static Material CarPaintMidnight() {
        Material m(vec3(0.02f, 0.02f, 0.15f), 0.5f); // Dark blue rough base
        m.metallic = 0.4f;                           // Semi-metallic
        m.clearcoat = 1.0f;
        m.clearcoatRoughness = 0.01f; // Super shiny coat
        return m;
    }

    static Material LacqueredWood() {
        Material m(vec3(0.2f, 0.1f, 0.02f), 0.6f);
        m.clearcoat = 1.0f;
        m.clearcoatRoughness = 0.05f;
        return m;
    }

    // PLASTICS / RUBBERS
    static Material PlasticRed() {
        return Material(vec3(0.8f, 0.1f, 0.1f), 0.3f);
    }
    static Material RubberBlack() { return Material(vec3(0.05f), 0.8f); }

    // SUBSURFACE
    static Material Wax() {
        Material m(vec3(0.9f, 0.8f, 0.5f), 0.3f);
        m.transmission = 0.2f;
        return m;
    }

    static Material Jade() {
        Material m(vec3(0.1f, 0.6f, 0.3f), 0.4f);
        m.subsurfaceRadius = 1.0f;
        m.subsurfaceColor = vec3(0.1f, 0.8f, 0.4f);
        return m;
    }

    // CUSTOM / ARTISTIC
    static Material PearlescentPaint(vec3 color) {
        Material m(color, 0.4f, 0.8f);
        m.iridescence = 0.5f;
        return m;
    }
    static Material GlowingNeon(vec3 color) {
        Material m(vec3(0.0f));
        m.emission = color * 10.0f;
        return m;
    }

    // MARBLES
    static Material MarbleCarrara() {
        return Material(vec3(0.95f), 0.1f, 0.5f);
    }
    static Material MarbleVerde() {
        return Material(vec3(0.1f, 0.4f, 0.2f), 0.1f, 0.6f);
    }
    static Material MarbleNero() { return Material(vec3(0.05f), 0.1f, 0.7f); }
};

// Default Scene Helper

namespace Scenes {
inline std::unique_ptr<Scene> createLitTestScene(int w, int h) {
    auto scene = std::make_unique<Scene>(w, h);
    Material floorMat(vec3(0.8f), 0.5f);
    scene->addPlaneXZ(-1.0f, 50.0f, floorMat);
    Mesh *sphere = scene->addCube(Materials::Silver());
    sphere->moveTo(vec3(0, 0.5, 3));
    scene->addSpotLight(vec3(-3, 5, 2), vec3(1, -1, 1), vec3(1.0f), 5.0f);
    scene->addPointLight(vec3(2, 3, 1), vec3(0.8f, 0.8f, 1.0f), 2.0f);
    scene->setCamera(vec3(0, 1.5, -2), vec3(0, 0.5, 3), vec3(0, 1, 0), 60.0f);
    return scene;
}
} // namespace Scenes

// Controller Structs

struct CameraController {
    vec3 pos{0, 0, 3};
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 1.0f;
    float sensitivity = 0.12f;
    bool captureMouse = true;
    double lastX = 0.0, lastY = 0.0;
    bool firstMouse = true;

    void initFromScene(Scene &s, int W, int H) {
        pos = s.cameraOrigin();
        vec3 f = s.cameraForward();
        yaw = std::atan2(f.z, f.x) * 180.0f / 3.14159265f;
        pitch = std::asin(std::clamp(f.y, -1.0f, 1.0f)) * 180.0f / 3.14159265f;
        lastX = W * 0.5;
        lastY = H * 0.5;
        firstMouse = true;
    }

    static vec3 rightFromForward(const vec3 &f) {
        vec3 up(0, 1, 0);
        return cross(f, up).normalized();
    }
    static vec3 forwardFromYawPitch(float yawDeg, float pitchDeg) {
        float cy = std::cos(yawDeg * 3.14159265f / 180.0f);
        float sy = std::sin(yawDeg * 3.14159265f / 180.0f);
        float cp = std::cos(pitchDeg * 3.14159265f / 180.0f);
        float sp = std::sin(pitchDeg * 3.14159265f / 180.0f);
        return vec3(cy * cp, sp, sy * cp).normalized();
    }

    void applyMouse(GLFWwindow *win, float /*dt*/) {
        if (!captureMouse)
            return;
        double x, y;
        glfwGetCursorPos(win, &x, &y);
        if (firstMouse) {
            lastX = x;
            lastY = y;
            firstMouse = false;
        }
        double dx = x - lastX;
        double dy = lastY - y;
        lastX = x;
        lastY = y;
        yaw += float(dx) * sensitivity;
        pitch += float(dy) * sensitivity;
        pitch = std::clamp(pitch, -89.9f, 89.9f);
    }

    void applyKeyboard(GLFWwindow *win, float dt) {
        float boost =
            (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? 2.5f : 1.0f;
        float v = speed * boost * dt * 5;
        vec3 fwd = forwardFromYawPitch(yaw, pitch);
        vec3 right = rightFromForward(fwd);
        vec3 up(0, 1, 0);

        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS)
            pos = pos + fwd * v;
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS)
            pos = pos - fwd * v;
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS)
            pos = pos - right * v;
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS)
            pos = pos + right * v;
        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS)
            pos = pos + up * v;
        if (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            pos = pos - up * v;

        static bool cPrev = false;
        bool cNow = (glfwGetKey(win, GLFW_KEY_C) == GLFW_PRESS);
        if (cNow && !cPrev) {
            captureMouse = !captureMouse;
            glfwSetInputMode(win, GLFW_CURSOR,
                             captureMouse ? GLFW_CURSOR_DISABLED
                                          : GLFW_CURSOR_NORMAL);
            firstMouse = true;
        }
        cPrev = cNow;
    }

    void update(Scene &s, GLFWwindow *win, float dt) {
        applyMouse(win, dt);
        applyKeyboard(win, dt);
        vec3 fwd = forwardFromYawPitch(yaw, pitch);
        s.moveCamera(pos);
        s.lookCameraAt(pos + fwd, vec3(0, 1, 0));
    }
};

struct VisualizationController {
    bool showFrustum = false;
    bool showRays = false;
    bool showPrimaryRays = false;
    float rayLength = 5.0f;
    int numDebugRays = 16;
    int maxBounces = 10;
    Camera camera;
    bool keys[GLFW_KEY_LAST + 1] = {};

    explicit VisualizationController(float aspect = 16.0f / 9.0f)
        : camera(aspect) {}

    void handleKeyPress(GLFWwindow *window, Scene &scene) {
        const bool fKey = (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS);
        if (fKey && !keys[GLFW_KEY_F]) {
            showFrustum = true;
            scene.setShowFrustum(showFrustum);
        }
        keys[GLFW_KEY_F] = fKey;

        const bool vKey = (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS);
        if (vKey && !keys[GLFW_KEY_V]) {
            showRays = !showRays;
            scene.setShowRays(showRays);
            camera = scene.getCamera();
        }
        keys[GLFW_KEY_V] = vKey;

        const bool pKey = (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS);
        if (pKey && !keys[GLFW_KEY_P]) {
            scene.generatePrimaryRayVisualization(camera, numDebugRays);
        }
        keys[GLFW_KEY_P] = pKey;

        const bool plusKey = (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS);
        const bool minusKey =
            (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS);
        if (plusKey && !keys[GLFW_KEY_EQUAL]) {
            rayLength += 0.5f;
            scene.setRayLength(rayLength);
        }
        if (minusKey && !keys[GLFW_KEY_MINUS]) {
            rayLength = fmaxf(0.5f, rayLength - 0.5f);
            scene.setRayLength(rayLength);
        }
        keys[GLFW_KEY_EQUAL] = plusKey;
        keys[GLFW_KEY_MINUS] = minusKey;

        const bool hKey = (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS);
        if (hKey && !keys[GLFW_KEY_H])
            printVisualizationHelp();
        keys[GLFW_KEY_H] = hKey;
    }

    void printVisualizationHelp() {
        std::cout << "\nVisualization Controls\n"
                  << "F - Toggle camera frustum display\n"
                  << "V - Toggle ray visualization\n"
                  << "P - Generate primary rays from camera\n"
                  << "+/- - Increase/decrease ray length\n"
                  << "H - Show this help\n"
                  << std::endl;
    }
};

// Utility Functions

inline void printUsage(const char *programName) {
    std::cout << "\nUsage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -s, --scene <id>    Select scene (0-10)\n";
    std::cout << "  -w, --width <size>  Set image width (default: 800)\n";
    std::cout << "  -h, --height <size> Set image height (default: 600)\n";
    std::cout << "  -o, --output <name> Output filename (without extension)\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Scenes:\n";
    std::cout << "  0: Lit Test Scene\n";
    std::cout << "  1: Presidents\n";
    std::cout << "  2: Statues\n";
    std::cout << "  3: X3D Components\n";
    std::cout << "  4: Abstract Pair\n";
    std::cout << "  5: Vase\n";
    std::cout << "  6: USNM Object\n";
    std::cout << "  7: Custom Scene\n";
    std::cout << "  8: Ultimate Model Showcase (Updated)\n";
    std::cout << "  9: Cornell Gems\n";
    std::cout << " 10: Material Matrix (Cubes)\n";
}

inline RenderConfig parseArguments(int argc, char *argv[]) {
    RenderConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            config.showHelp = true;
            return config;
        } else if ((arg == "-s" || arg == "--scene") && i + 1 < argc)
            config.sceneId = std::atoi(argv[++i]);
        else if ((arg == "-w" || arg == "--width") && i + 1 < argc)
            config.width = std::atoi(argv[++i]);
        else if ((arg == "-h" || arg == "--height") && i + 1 < argc)
            config.height = std::atoi(argv[++i]);
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc)
            config.outputName = argv[++i];
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            config.showHelp = true;
            return config;
        }
    }
    return config;
}

inline void printRenderInfo(const std::string &sceneName, int width,
                            int height) {
    std::cout << "Scene:      " << std::left << std::setw(26) << sceneName
              << "\n";
    std::cout << "Resolution: " << std::setw(26)
              << (std::to_string(width) + " x " + std::to_string(height))
              << "\n";
}

// Scene Builder

inline std::pair<std::unique_ptr<Scene>, std::string>
buildSceneById(RenderConfig configs) {
    std::unique_ptr<Scene> scene;
    std::string sceneName;
    Material floorMat(vec3(0.8f), 0.7f);

    // Default camera settings
    vec3 lookfrom(0, 0, 0);
    vec3 lookat(0, 3.5, 5);
    float focus_dist = (lookat - lookfrom).length();

    switch (configs.sceneId) {
    case 0:
        sceneName = "Lit Test Scene";
        scene = Scenes::createLitTestScene(configs.width, configs.height);
        break;

    case 1: {
        sceneName = "Presidents";
        scene = std::make_unique<Scene>(configs.width, configs.height);
        scene->setSkyGradient(vec3(0.1f), vec3(0.0f));
        Mesh *lincoln =
            scene->addMesh("models/abraham-lincoln-mills-life-mask-150k.obj",
                           Materials::Copper());
        lincoln->scale(0.8f / 50.0f);
        lincoln->moveTo(vec3(-2, 0, 4));
        lincoln->rotateSelfEulerXYZ(vec3(0, 0.5, 0));
        Mesh *washington = scene->addMesh(
            "models/george-washington-greenough-statue-(1840)-150k.obj",
            Materials::MarbleCarrara());
        washington->scale(0.6f / 500.0f);
        washington->moveTo(vec3(2, -1, 4));
        washington->rotateSelfEulerXYZ(vec3(0, -0.5, 0));
        scene->addSpotLight(vec3(-3, 5, 2), vec3(0.5f, -1, 0.5f), vec3(1.0f),
                            5.0f, 0.1f, 0.3f, 1.75f, 0.2f);
        scene->addPointLight(vec3(3, 4, 2), vec3(0.5f, 0.5f, 1.0f), 1.0f, 1.0f,
                             0.2f);
        scene->setCamera(lookfrom, lookat, vec3(0, 1, 0), 60.0f, 0.0001f,
                         focus_dist);
        scene->addPlaneXZ(-3.0f, 50.0f, floorMat);
        break;
    }
    case 2: {
        sceneName = "Statues";
        scene = std::make_unique<Scene>(configs.width, configs.height);
        scene->setSkyGradient(vec3(0.6f, 0.7f, 0.9f), vec3(0.9f, 0.95f, 1.0f));
        Mesh *jackson = scene->addMesh(
            "models/andrew-jackson-zinc-sculpture-150k.obj", Materials::Jade());
        jackson->scale(0.7f / 50.0f);
        jackson->moveTo(vec3(0, 0, 4.5));
        jackson->rotateSelfEulerXYZ(vec3(0, 0.3, 0));
        scene->addSpotLight(vec3(0, 7, 5), vec3(0, -1, 0), vec3(1.0f), 4.0f,
                            0.1f, 0.4f, 2.0f, 0.3f);
        scene->addPointLight(vec3(0, 4, 0), vec3(1.0f, 0.7f, 0.3f), 0.5f, 1.0f,
                             0.1f);
        scene->setCamera(lookfrom, lookat, vec3(0, 1, 0), 60.0f, 0.0001f,
                         focus_dist);
        scene->addPlaneXZ(-3.0f, 50.0f, floorMat);
        break;
    }
    case 3: {
        sceneName = "X3D Components";
        scene = std::make_unique<Scene>(configs.width, configs.height);
        Mesh *shell = scene->addMesh("models/x3d-cm-exterior-shell-90k-uvs.obj",
                                     Materials::FrostedGlass());
        shell->scale(0.5f / 50.0f);
        shell->moveTo(vec3(-2, 0, 4));
        shell->rotateSelfEulerXYZ(vec3(0, 0.3, 0));
        Mesh *top = scene->addMesh("models/x3d-cm-exterior-top-160k-uvs.obj",
                                   Materials::Titanium());
        top->scale(0.5f / 50.0f);
        top->moveTo(vec3(2, 0, 4));
        top->rotateSelfEulerXYZ(vec3(0, 0.3, 0));
        scene->addPointLight(vec3(-4, 5, 2), vec3(1.0f), 2.0f, 1.0f, 0.5f);
        scene->addPointLight(vec3(3, 4, 1), vec3(0.5f), 1.0f, 1.0f, 0.5f);
        scene->addPointLight(vec3(0, 4, 8), vec3(0.7f), 1.5f, 1.0f, 0.5f);
        scene->setCamera(lookfrom, lookat, vec3(0, 1, 0), 60.0f, 0.0001f,
                         focus_dist);
        scene->addPlaneXZ(-3.0f, 50.0f, floorMat);
        break;
    }
    case 4: {
        sceneName = "Abstract Pair";
        scene = std::make_unique<Scene>(configs.width, configs.height);
        Mesh *full =
            scene->addMesh("models/full.obj", Materials::CarPaintMidnight());
        full->scale(0.5f * 30.0f);
        full->moveTo(vec3(-2.5, 0, 4));
        full->rotateSelfEulerXYZ(vec3(0, 0.5, 0));
        Mesh *lowteira =
            scene->addMesh("models/cosmic-buddha-laser-scan-150k.obj",
                           Materials::RubberBlack());
        lowteira->scale(0.7f / 100.0f);
        lowteira->moveTo(vec3(2, -1, 4));
        lowteira->rotateSelfEulerXYZ(vec3(0, -0.5, 0));
        scene->addSpotLight(vec3(0, 4, 2), vec3(0, -1, -0.5f), vec3(1.0f), 5.0f,
                            0.1f, 0.3f, 1.75f, 0.2f);
        scene->addPointLight(vec3(0, 4.5, 2), vec3(0.5f, 0.5f, 1.0f), 1.0f,
                             1.0f, 0.2f);
        scene->setCamera(lookfrom, lookat, vec3(0, 1, 0), 60.0f, 0.0001f,
                         focus_dist);
        scene->addPlaneXZ(-3.0f, 50.0f, floorMat);
        break;
    }
    case 5: {
        sceneName = "Vase";
        scene = std::make_unique<Scene>(configs.width, configs.height);
        Mesh *vase = scene->addMesh("models/vase.obj", Materials::Wax());
        vase->scale(0.7f / 100.0f);
        vase->moveTo(vec3(0, 0, 4));
        vase->rotateSelfEulerXYZ(vec3(0, 0.3, 0));
        scene->addPointLight(vec3(-2, 4, 2), vec3(0.8f), 1.5f, 1.0f, 0.3f);
        scene->addSpotLight(vec3(0, 4, 8), vec3(0, -0.1f, -1), vec3(1.0f), 6.0f,
                            0.05f, 0.2f, 2.0f, 0.3f);
        scene->setCamera(lookfrom, lookat, vec3(0, 1, 0), 60.0f, 0.0001f,
                         focus_dist);
        scene->addPlaneXZ(-3.0f, 50.0f, floorMat);
        break;
    }
    case 6: {
        sceneName = "USNM Object";
        scene = std::make_unique<Scene>(configs.width, configs.height);
        Mesh *usnm =
            scene->addMesh("models/usnm_346-01-100k.obj",
                           Materials::PearlescentPaint(vec3(0.8f, 0.2f, 0.5f)));
        usnm->scale(0.6f / 50.0f);
        usnm->moveTo(vec3(0, 0, 4));
        usnm->rotateSelfEulerXYZ(vec3(0, 0.3, 0));
        scene->addSpotLight(vec3(0, 4, 2), vec3(0, -1, -0.5f), vec3(1.0f), 5.0f,
                            0.1f, 0.3f, 1.75f, 0.2f);
        scene->addPointLight(vec3(0, 4.5, 2), vec3(0.5f, 0.5f, 1.0f), 1.0f,
                             1.0f, 0.2f);
        scene->setCamera(lookfrom, lookat, vec3(0, 1, 0), 60.0f, 0.0001f,
                         focus_dist);
        scene->addPlaneXZ(-3.0f, 50.0f, floorMat);
        break;
    }
    case 7: {
        sceneName = "Custom Scene (lowteiradam)";
        scene = std::make_unique<Scene>(configs.width, configs.height);
        Mesh *fancyguy = scene->addMesh("models/subhumanchoppedahhdude.obj",
                                        Materials::VelvetRed());
        fancyguy->scale(0.6f / 100.0f);
        fancyguy->moveTo(vec3(0, 0, 4));
        fancyguy->rotateSelfEulerXYZ(vec3(0, 0.3, 0));
        scene->addSpotLight(vec3(0, 4, 2), vec3(0, -1, -0.5f),
                            vec3(1.0f, 1.0f, 1.0f), 5.0f, 0.1f, 0.3f, 1.75f,
                            0.2f);
        scene->addPointLight(vec3(0, 4.5, 2), vec3(0.5f, 0.5f, 1.0f), 1.0f,
                             1.0f, 0.2f);
        scene->setCamera(lookfrom, lookat, vec3(0, 1, 0), 60.0f, 0.0001f,
                         focus_dist);
        floorMat.specular = vec3(0.1f);
        scene->addPlaneXZ(-3.0f, 50.0f, floorMat);
        break;
    }
    case 8: {
        sceneName = "Ultimate Model Showcase";
        scene = std::make_unique<Scene>(configs.width, configs.height);

        // Enclosed Room
        Material roomMat(vec3(0.4f), 0.9f);
        roomMat.specular = vec3(0.0f);
        float roomHeight = 10.0f;
        float roomYCenter = 2.0f;
        float floorY = roomYCenter - roomHeight / 2.0f;
        vec3 rotation = vec3(0, 0.3, 0);

        // Back Row (z = -12)
        Mesh *lincoln =
            scene->addMesh("models/abraham-lincoln-mills-life-mask-150k.obj",
                           Materials::Copper());
        lincoln->scale(0.8f / 50.0f);
        lincoln->moveTo(vec3(-8, floorY + 3.0f, -12));
        lincoln->rotateSelfEulerXYZ(rotation);

        Mesh *washington = scene->addMesh(
            "models/george-washington-greenough-statue-(1840)-150k.obj",
            Materials::MarbleCarrara());
        washington->scale(0.6f / 500.0f);
        washington->moveTo(vec3(-4, floorY + 2.0f, -12));
        washington->rotateSelfEulerXYZ(rotation);

        Mesh *jackson = scene->addMesh(
            "models/andrew-jackson-zinc-sculpture-150k.obj", Materials::Jade());
        jackson->scale(0.7f / 50.0f);
        jackson->moveTo(vec3(0, floorY + 3.0f, -12));
        jackson->rotateSelfEulerXYZ(rotation);

        Mesh *shell = scene->addMesh("models/x3d-cm-exterior-shell-90k-uvs.obj",
                                     Materials::SoapBubble());
        shell->scale(0.5f / 50.0f);
        shell->moveTo(vec3(4, floorY + 3.0f, -12));
        shell->rotateSelfEulerXYZ(rotation);

        Mesh *top = scene->addMesh("models/x3d-cm-exterior-top-160k-uvs.obj",
                                   Materials::Titanium());
        top->scale(0.5f / 50.0f);
        top->moveTo(vec3(8, floorY + 3.0f, -12));
        top->rotateSelfEulerXYZ(rotation);

        // Front Row (z = -8)
        Mesh *full =
            scene->addMesh("models/full.obj", Materials::CarPaintMidnight());
        full->scale(0.5f * 30.0f);
        full->moveTo(vec3(-8, floorY + 3.0f, -8));
        full->rotateSelfEulerXYZ(rotation);

        Mesh *lowteira = scene->addMesh(
            "models/cosmic-buddha-laser-scan-150k.obj", Materials::Gold());
        lowteira->scale(0.7f / 100.0f);
        lowteira->moveTo(vec3(-4, floorY + 2.0f, -8));
        lowteira->rotateSelfEulerXYZ(rotation);

        Mesh *vase = scene->addMesh("models/vase.obj", Materials::Wax());
        vase->scale(0.7f / 100.0f);
        vase->moveTo(vec3(0, floorY + 3.0f, -8));
        vase->rotateSelfEulerXYZ(rotation);

        Mesh *usnm = scene->addMesh("models/usnm_346-01-100k.obj",
                                    Materials::VelvetRed());
        usnm->scale(0.6f / 50.0f);
        usnm->moveTo(vec3(4, floorY + 3.0f, -8));
        usnm->rotateSelfEulerXYZ(rotation);

        Mesh *fancy = scene->addMesh("models/lowteiradamlookindude.obj",
                                     Materials::Glass());
        fancy->scale(0.6f / 100.0f);
        fancy->moveTo(vec3(8, floorY + 3.0f, -8));
        fancy->rotateSelfEulerXYZ(rotation);

        // Lighting
        scene->setSkyGradient(vec3(0.5f), vec3(0.5f));
        vec3 lightColor = vec3(1.0f);
        scene->addSpotLight(vec3(0, 6.5, -10), vec3(0, -1, 0), lightColor,
                            15.0f, 0.1f, 0.8f, 2.0f, 0.1f);
        scene->addSpotLight(vec3(-6, 6.5, -10), vec3(0, -1, 0), lightColor,
                            12.0f, 0.1f, 0.8f, 2.0f, 0.1f);
        scene->addSpotLight(vec3(6, 6.5, -10), vec3(0, -1, 0), lightColor,
                            12.0f, 0.1f, 0.8f, 2.0f, 0.1f);
        scene->addPointLight(vec3(0, 2, 4), vec3(0.8f), 5.0f, 20.0f, 0.1f);
        scene->addPointLight(vec3(-8, 1, 4), vec3(0.5f), 3.0f, 20.0f, 0.1f);
        scene->addPointLight(vec3(8, 1, 4), vec3(0.5f), 3.0f, 20.0f, 0.1f);

        // Camera
        vec3 camPos = vec3(0, 2, 5);
        vec3 camAt = vec3(0, 0, -10);
        scene->setCamera(camPos, camAt, vec3(0, 1, 0), 50.0f, 0.0f,
                         (camAt - camPos).length());
        break;
    }
    case 9: {
        sceneName = "Custom Scene1 (Cornell Gems)";
        scene = std::make_unique<Scene>(configs.width, configs.height);
        Mesh *gem1 = scene->addMesh("models/ugly.obj", Materials::OilSlick());
        gem1->scale(20.0f);
        gem1->moveTo(vec3(-2.5, -2, -10));
        gem1->rotateSelfEulerXYZ(vec3(0, 0.5, 0));
        Mesh *gem2 =
            scene->addMesh("models/halfway.obj", Materials::SatinBlue());
        gem2->scale(20.0f);
        gem2->moveTo(vec3(0, -2, -10));
        gem2->rotateSelfEulerXYZ(vec3(0, -0.2, 0));
        Mesh *gem3 = scene->addMesh("models/full.obj", Materials::Diamond());
        gem3->scale(20.0f);
        gem3->moveTo(vec3(2.5, -2, -10));
        gem3->rotateSelfEulerXYZ(vec3(0, -0.5, 0));

        Material wallMat = Materials::Silver();
        wallMat.roughness = 0.5f;
        Mesh *backWall = scene->addCube(wallMat);
        backWall->scale(vec3(6, 6, 0.1));
        backWall->moveTo(vec3(0, 2, -13));
        Mesh *leftWall = scene->addCube(wallMat);
        leftWall->scale(vec3(0.1, 6, 6));
        leftWall->moveTo(vec3(-5, 2, -7));
        Mesh *rightWall = scene->addCube(wallMat);
        rightWall->scale(vec3(0.1, 6, 6));
        rightWall->moveTo(vec3(5, 2, -7));
        Mesh *floor = scene->addCube(wallMat);
        floor->scale(vec3(6, 0.1, 6));
        floor->moveTo(vec3(0, -2, -7));
        Mesh *roof = scene->addCube(wallMat);
        roof->scale(vec3(6, 0.1, 6));
        roof->moveTo(vec3(0, 8, -7));

        scene->addPointLight(vec3(3, 0, -10), vec3(0.5f, 0.5f, 1.0f), 1.0f,
                             2.0f, 0.3f);
        scene->addPointLight(vec3(1, 1, -9), vec3(0.5f, 0.5f, 1.0f), 1.0f,
                             1.75f, 0.3f);
        scene->addPointLight(vec3(4, 1, -9), vec3(0.5f, 0.5f, 1.0f), 1.0f, 2.0f,
                             0.3f);
        scene->addPointLight(vec3(0, 7.5, -7), vec3(1.0f, 0.9f, 0.8f), 15.0f,
                             10.0f, 0.5f);
        scene->setCamera(vec3(0, 0, 0), vec3(0, 0, -10), vec3(0, 1, 0), 60.0f,
                         0.0001f, 10.0f);
        scene->setSkyGradient(vec3(0.05f, 0.05f, 0.08f),
                              vec3(0.02f, 0.02f, 0.03f));
        break;
    }
    case 10: {
        sceneName = "Material Matrix (Cubes)";
        scene = std::make_unique<Scene>(configs.width, configs.height);

        // Floor
        Material tileMat(vec3(0.2f), 0.8f);
        scene->addPlaneXZ(-1.0f, 50.0f, tileMat);

        // Grid parameters
        int rows = 4;
        int cols = 4;
        float spacing = 2.0f;
        float startX = -((cols - 1) * spacing) / 2.0f;
        float startZ = -((rows - 1) * spacing) / 2.0f - 5.0f; // Push back

        // Material Palette
        std::vector<Material> palette = {
            Materials::Silver(),
            Materials::Gold(),
            Materials::Copper(),
            Materials::Titanium(),
            Materials::CarPaintMidnight(),
            Materials::PlasticRed(),
            Materials::RubberBlack(),
            Materials::LacqueredWood(),
            Materials::Glass(),
            Materials::FrostedGlass(),
            Materials::SoapBubble(),
            Materials::OilSlick(),
            Materials::VelvetRed(),
            Materials::SatinBlue(),
            Materials::Jade(),
            Materials::GlowingNeon(vec3(0.2f, 1.0f, 0.2f))};

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int idx = r * cols + c;
                if (idx >= palette.size())
                    break;

                Mesh *cube = scene->addCube(palette[idx]);
                float x = startX + c * spacing;
                float z = startZ + r * spacing;

                // Add a small bevel/scale
                cube->scale(vec3(0.7f));
                cube->moveTo(vec3(x, 0.0f, z));
                cube->moveTo(vec3(x, -1.0f + 0.7f, z));

                // Rotate them slightly so they aren't boring
                cube->rotateSelfEulerXYZ(vec3(0, 0.7f, 0));
            }
        }

        // Lighting
        scene->addSpotLight(vec3(0, 8, -5), vec3(0, -1, 0), vec3(1.0f), 10.0f,
                            0.1f, 0.5f, 2.0f, 0.1f);
        scene->addPointLight(vec3(-5, 2, -2), vec3(1.0f, 0.8f, 0.8f), 2.0f,
                             10.0f, 0.2f);
        scene->addPointLight(vec3(5, 2, -2), vec3(0.8f, 0.8f, 1.0f), 2.0f,
                             10.0f, 0.2f);

        // Camera
        scene->setCamera(vec3(0, 6, 4), vec3(0, 0, -5), vec3(0, 1, 0), 50.0f);
        scene->setSkyGradient(vec3(0.1f), vec3(0.02f));
        break;
    }
    default:
        std::cerr << "Invalid scene ID. Using default.\n";
        sceneName = "Lit Test Scene";
        scene = Scenes::createLitTestScene(configs.width, configs.height);
        break;
    }

    scene->setBVHLeafTarget(configs.bvhLeafTarget, configs.bvhLeafTol);
    return {std::move(scene), sceneName};
}

#endif // APP_UTILS_CUH_