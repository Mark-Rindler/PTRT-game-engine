#ifndef APP_UTILS_CUH_
#define APP_UTILS_CUH_

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
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>

// file overview
// utilities used by the interactive ray tracing app
// provides small data structures helpers and glue code between glfw opengl and
// cuda keeps host only logic and gpu dispatch helpers in one place

#include "raytracer\RTscene.cuh"

#define GLFW_INCLUDE_NONE

// glad include handling
// this header avoids hard coding a single glad include path
// some projects ship glad as glad h some ship glad slash glad h and some wrap
// glad inside the interop header if your build already provides glad through
// the interop header these checks are harmless if your build provides glad as a
// standalone header one of these includes should be found

#if defined(__has_include)
#if __has_include(<glad/glad.h>)
#include <glad/glad.h>
#elif __has_include(<glad.h>)
#include <glad.h>
#elif __has_include("glad/glad.h")
#include "glad/glad.h"
#elif __has_include("glad.h")
#include "glad.h"
#endif
#endif

#include "common\glfw_view_interop.hpp"

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t err = (stmt);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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

// function print usage

inline void printUsage(const char *programName) {
    std::cout << "\nUsage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -s, --scene <id>    Select scene (0-" << 7 << ")\n";
    std::cout << "  -w, --width <size>  Set image width (default: 800)\n";
    std::cout << "  -h, --height <size> Set image height (default: 600)\n";
    std::cout << "  -o, --output <name> Output filename (without extension)\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Scenes:\n";
    std::cout << "  0: Lit Test Scene (basic lighting demo)\n";
    std::cout << "  1: Character Showcase (default)\n";
    std::cout << "  2: Presidents Showcase\n";
    std::cout << "  3: Statues Showcase\n";
    std::cout << "  4: Fancy Dudes Showcase\n";
    std::cout << "  5: Scans Showcase\n";
    std::cout << "  6: Artifacts Showcase\n";
    std::cout << "  7: Dynamic Circle Animation\n\n";
}

// function parse arguments

inline RenderConfig parseArguments(int argc, char *argv[]) {
    RenderConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            config.showHelp = true;
            return config;
        } else if ((arg == "-s" || arg == "--scene") && i + 1 < argc) {
            config.sceneId = std::atoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            config.width = std::atoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            config.height = std::atoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.outputName = argv[++i];
        } else {
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

struct CameraController {

    vec3 pos{0, 0, 3};
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 5.0f;
    float sensitivity = 0.12f;
    bool captureMouse = true;

    double lastX = 0.0, lastY = 0.0;
    bool firstMouse = true;

    // function init from scene

    void initFromScene(Scene &s, int W, int H) {
        pos = s.cameraOrigin();
        vec3 f = s.cameraForward();

        yaw = std::atan2(f.z, f.x) * 180.0f / 3.14159265f;
        pitch = std::asin(std::clamp(f.y, -1.0f, 1.0f)) * 180.0f / 3.14159265f;
        lastX = W * 0.5;
        lastY = H * 0.5;
        firstMouse = true;
    }

    // function right from forward

    static vec3 rightFromForward(const vec3 &f) {
        vec3 up(0, 1, 0);
        return cross(f, up).normalized();
    }
    // function forward from yaw pitch

    static vec3 forwardFromYawPitch(float yawDeg, float pitchDeg) {
        float cy = std::cos(yawDeg * 3.14159265f / 180.0f);
        float sy = std::sin(yawDeg * 3.14159265f / 180.0f);
        float cp = std::cos(pitchDeg * 3.14159265f / 180.0f);
        float sp = std::sin(pitchDeg * 3.14159265f / 180.0f);

        return vec3(cy * cp, sp, sy * cp).normalized();
    }

    // function apply mouse

    void applyMouse(GLFWwindow *win, float) {
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

    // function apply keyboard

    void applyKeyboard(GLFWwindow *win, float dt) {
        float boost =
            (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? 2.5f : 1.0f;
        float v = speed * boost * dt;

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

    // function update

    void update(Scene &s, GLFWwindow *win, float dt) {
        applyMouse(win, dt);
        applyKeyboard(win, dt);
        vec3 fwd = forwardFromYawPitch(yaw, pitch);
        s.moveCamera(pos);
        s.lookCameraAt(pos + fwd, vec3(0, 1, 0));
    }
};

namespace DemoScenes {

inline std::unique_ptr<Scene> createCornellBox(int width = 800,
                                               int height = 800) {
    auto scene = std::make_unique<Scene>(width, height);

    Material whiteMat(vec3(0.73f, 0.73f, 0.73f), 0.6f, 0.0f);
    whiteMat.specular = vec3(0.04f);

    Material redMat(vec3(0.65f, 0.05f, 0.05f), 0.6f, 0.0f);
    redMat.specular = vec3(0.04f);

    Material greenMat(vec3(0.12f, 0.45f, 0.15f), 0.6f, 0.0f);
    greenMat.specular = vec3(0.04f);

    Material lightMat(vec3(0.0f), 0.0f, 0.0f);
    lightMat.emission = vec3(15.0f);

    Mesh *back = scene->addCube(whiteMat);
    back->scale(vec3(10.0f, 10.0f, 0.1f));
    back->moveTo(vec3(0, 0, -10));

    Mesh *left = scene->addCube(redMat);
    left->scale(vec3(0.1f, 10.0f, 10.0f));
    left->moveTo(vec3(-5, 0, -5));

    Mesh *right = scene->addCube(greenMat);
    right->scale(vec3(0.1f, 10.0f, 10.0f));
    right->moveTo(vec3(5, 0, -5));

    Mesh *floor = scene->addCube(whiteMat);
    floor->scale(vec3(10.0f, 0.1f, 10.0f));
    floor->moveTo(vec3(0, -5, -5));

    Mesh *ceiling = scene->addCube(whiteMat);
    ceiling->scale(vec3(10.0f, 0.1f, 10.0f));
    ceiling->moveTo(vec3(0, 5, -5));

    Mesh *light = scene->addCube(lightMat);
    light->scale(vec3(2.0f, 0.1f, 2.0f));
    light->moveTo(vec3(0, 4.9f, -5));

    Material boxMat(vec3(0.9f), 0.2f, 0.0f);
    boxMat.specular = vec3(0.04f);

    Mesh *box1 = scene->addCube(boxMat);
    box1->scale(vec3(1.5f, 3.0f, 1.5f));
    box1->moveTo(vec3(-1.5f, -3.5f, -6));
    box1->rotateSelfEulerXYZ(vec3(0, 0.3f, 0));

    Mesh *box2 = scene->addCube(boxMat);
    box2->scale(vec3(1.5f, 1.5f, 1.5f));
    box2->moveTo(vec3(1.5f, -4.25f, -4));
    box2->rotateSelfEulerXYZ(vec3(0, -0.4f, 0));

    scene->addPointLight(vec3(0, 4.5f, -5), vec3(1.0f, 0.9f, 0.8f), 3.0f,
                         20.0f);
    scene->setAmbientLight(vec3(0.02f));

    scene->setCamera(vec3(0, 0, 5), vec3(0, 0, -5), vec3(0, 1, 0), 40.0f);

    scene->disableSky();
    return scene;
}

inline std::unique_ptr<Scene> createMaterialShowcase1(int width = 1200,
                                                      int height = 800) {
    auto scene = std::make_unique<Scene>(width, height);

    const int rows = 3;
    const int cols = 5;
    const float spacing = 2.5f;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float metallic = static_cast<float>(j) / (cols - 1);
            float roughness = static_cast<float>(i) / (rows - 1);

            Material mat(vec3(0.8f, 0.3f, 0.2f), roughness, metallic);
            mat.specular = vec3(0.04f);

            Mesh *sphere = scene->addCube(mat);
            sphere->scale(0.8f);
            float x = (j - cols / 2.0f) * spacing;
            float y = (i - rows / 2.0f) * spacing;
            sphere->moveTo(vec3(x, y, -10));
        }
    }

    scene->addPointLight(vec3(10, 10, 0), vec3(1.0f, 0.95f, 0.9f), 3.0f, 50.0f);
    scene->addPointLight(vec3(-10, 5, 5), vec3(0.4f, 0.4f, 0.5f), 2.0f, 40.0f);
    scene->addPointLight(vec3(0, 15, -15), vec3(0.8f, 0.8f, 1.0f), 1.5f, 40.0f);

    scene->setAmbientLight(vec3(0.03f));
    scene->setCamera(vec3(0, 0, 5), vec3(0, 0, -10), vec3(0, 1, 0), 45.0f);

    Material floorMat(vec3(0.8f), 0.4f, 0.0f);
    floorMat.specular = vec3(0.04f);
    scene->addPlaneXZ(-10.0f, 50.0f, floorMat);

    return scene;
}

inline std::unique_ptr<Scene> createLightShow(int width = 1024,
                                              int height = 768) {
    auto scene = std::make_unique<Scene>(width, height);

    Mesh *centerSphere = scene->addCube(Materials::Water());
    centerSphere->scale(2.0f);
    centerSphere->moveTo(vec3(0, 0, -10));

    const int numObjects = 12;
    const float radius = 6.0f;

    for (int i = 0; i < numObjects; ++i) {
        float angle = (TWO_PI * i) / numObjects;
        float hue = static_cast<float>(i) / numObjects;

        vec3 color(0.5f + 0.5f * cosf(TWO_PI * hue),
                   0.5f + 0.5f * cosf(TWO_PI * hue + TWO_PI / 3),
                   0.5f + 0.5f * cosf(TWO_PI * hue + 2 * TWO_PI / 3));

        Material mat(color, 0.25f, (i % 2) ? 0.8f : 0.2f);
        mat.specular = vec3(0.04f);

        Mesh *obj = scene->addCube(mat);
        obj->scale(0.7f);
        obj->moveTo(vec3(radius * cosf(angle), 2.0f * sinf(angle * 2),
                         -10 + radius * sinf(angle)));
        obj->rotateSelfEulerXYZ(vec3(angle, angle * 0.5f, 0));
    }

    scene->addPointLight(vec3(5, 3, -5), vec3(1.0f, 0.2f, 0.2f), 3.0f, 30.0f);
    scene->addPointLight(vec3(-5, 3, -5), vec3(0.2f, 1.0f, 0.2f), 3.0f, 30.0f);
    scene->addPointLight(vec3(0, -3, -5), vec3(0.2f, 0.2f, 1.0f), 3.0f, 30.0f);
    scene->addPointLight(vec3(0, 8, -10), vec3(1.0f, 1.0f, 1.0f), 2.0f, 40.0f);
    scene->addSpotLight(vec3(0, 10, 0), vec3(0, -1, -0.5f),
                        vec3(1.0f, 0.9f, 0.7f), 4.0f, 0.2f, 0.4f, 30.0f);

    scene->setAmbientLight(vec3(0.01f));
    scene->setCamera(vec3(8, 5, 8), vec3(0, 0, -10), vec3(0, 1, 0), 50.0f);

    Material floorMat(vec3(0.8f), 0.4f, 0.0f);
    floorMat.specular = vec3(0.04f);
    scene->addPlaneXZ(-5.0f, 50.0f, floorMat);

    return scene;
}

inline std::unique_ptr<Scene> createArchitectural(int width = 1280,
                                                  int height = 720) {
    auto scene = std::make_unique<Scene>(width, height);

    Material concrete(vec3(0.7f, 0.7f, 0.65f), 0.6f, 0.0f);
    concrete.specular = vec3(0.04f);

    Material glass(vec3(0.98f), 0.02f, 0.0f);
    glass.specular = vec3(0.04f);
    glass.transmission = 0.98f;
    glass.ior = 1.5f;

    Material wood(vec3(0.55f, 0.35f, 0.2f), 0.45f, 0.0f);
    wood.specular = vec3(0.04f);

    for (int i = 0; i < 5; ++i) {
        Mesh *pillar = scene->addCube(concrete);
        pillar->scale(vec3(0.5f, 8.0f, 0.5f));
        pillar->moveTo(vec3(-8.0f + i * 4.0f, 0.0f, -15.0f));
    }

    for (int i = 0; i < 4; ++i) {
        Mesh *panel = scene->addCube(glass);
        panel->scale(vec3(3.8f, 6.0f, 0.1f));
        panel->moveTo(vec3(-6.0f + i * 4.0f, 0.0f, -14.5f));
    }

    Mesh *floor = scene->addCube(wood);
    floor->scale(vec3(20.0f, 0.2f, 20.0f));
    floor->moveTo(vec3(0, -4, -15));

    Mesh *ceiling = scene->addCube(concrete);
    ceiling->scale(vec3(20.0f, 0.5f, 20.0f));
    ceiling->moveTo(vec3(0, 4, -15));

    scene->addDirectionalLight(vec3(-0.3f, -0.6f, -0.5f),
                               vec3(1.0f, 0.95f, 0.8f), 1.5f);
    for (int i = 0; i < 3; ++i) {
        scene->addPointLight(vec3(-4.0f + i * 4.0f, 3, -12.0f),
                             vec3(1.0f, 0.9f, 0.7f), 0.8f, 15.0f);
    }

    scene->setAmbientLight(vec3(0.15f, 0.15f, 0.2f));
    scene->setCamera(vec3(10, 2, 0), vec3(0, 0, -15), vec3(0, 1, 0), 60.0f);

    Material ground(vec3(0.8f), 0.4f, 0.0f);
    ground.specular = vec3(0.04f);
    scene->addPlaneXZ(-10.0f, 50.0f, ground);

    return scene;
}

inline std::unique_ptr<Scene> createMaterialShowcase(int width = 1024,
                                                     int height = 768) {
    auto scene = std::make_unique<Scene>(width, height);

    const int gridSize = 5;
    const float spacing = 2.5f;
    const float startX = -(gridSize - 1) * spacing / 2.0f;
    const float startZ = -10.0f;

    Mesh *m1 = scene->addCube(Materials::Gold());
    m1->moveTo(vec3(startX + 0 * spacing, 0, startZ));
    m1->scale(0.8f);
    Mesh *m2 = scene->addCube(Materials::Silver());
    m2->moveTo(vec3(startX + 1 * spacing, 0, startZ));
    m2->scale(0.8f);
    Mesh *m3 = scene->addCube(Materials::Copper());
    m3->moveTo(vec3(startX + 2 * spacing, 0, startZ));
    m3->scale(0.8f);
    Mesh *m4 = scene->addCube(Materials::BrushedAluminum());
    m4->moveTo(vec3(startX + 3 * spacing, 0, startZ));
    m4->scale(0.8f);
    Mesh *m5 = scene->addCube(Materials::OilSlick());
    m5->moveTo(vec3(startX + 4 * spacing, 0, startZ));
    m5->scale(0.8f);

    Mesh *m6 = scene->addCube(Materials::Glass());
    m6->moveTo(vec3(startX + 0 * spacing, 0, startZ - spacing));
    m6->scale(0.8f);
    Mesh *m7 = scene->addCube(Materials::FrostedGlass());
    m7->moveTo(vec3(startX + 1 * spacing, 0, startZ - spacing));
    m7->scale(0.8f);
    Mesh *m8 = scene->addCube(Materials::Diamond());
    m8->moveTo(vec3(startX + 2 * spacing, 0, startZ - spacing));
    m8->scale(0.8f);
    Mesh *m9 = scene->addCube(Materials::SoapBubble());
    m9->moveTo(vec3(startX + 3 * spacing, 0, startZ - spacing));
    m9->scale(0.8f);
    Mesh *m10 = scene->addCube(Materials::Water());
    m10->moveTo(vec3(startX + 4 * spacing, 0, startZ - spacing));
    m10->scale(0.8f);

    Mesh *m11 = scene->addCube(Materials::CarPaint(vec3(0.8f, 0.1f, 0.1f)));
    m11->moveTo(vec3(startX + 0 * spacing, 0, startZ - 2 * spacing));
    m11->scale(0.8f);

    Mesh *m12 =
        scene->addCube(Materials::PearlescentPaint(vec3(0.9f, 0.9f, 1.0f)));
    m12->moveTo(vec3(startX + 1 * spacing, 0, startZ - 2 * spacing));
    m12->scale(0.8f);

    Mesh *m13 = scene->addCube(Materials::Skin());
    m13->moveTo(vec3(startX + 2 * spacing, 0, startZ - 2 * spacing));
    m13->scale(0.8f);

    Mesh *m14 = scene->addCube(Materials::Jade());
    m14->moveTo(vec3(startX + 3 * spacing, 0, startZ - 2 * spacing));
    m14->scale(0.8f);

    Mesh *m15 = scene->addCube(Materials::Wax());
    m15->moveTo(vec3(startX + 4 * spacing, 0, startZ - 2 * spacing));
    m15->scale(0.8f);

    Mesh *m16 = scene->addCube(Materials::Velvet(vec3(0.5f, 0.1f, 0.6f)));
    m16->moveTo(vec3(startX + 0 * spacing, 0, startZ - 3 * spacing));
    m16->scale(0.8f);

    Mesh *m17 = scene->addCube(Materials::Silk(vec3(0.1f, 0.3f, 0.8f)));
    m17->moveTo(vec3(startX + 1 * spacing, 0, startZ - 3 * spacing));
    m17->scale(0.8f);

    Mesh *m18 = scene->addCube(Materials::PlasticRed());
    m18->moveTo(vec3(startX + 2 * spacing, 0, startZ - 3 * spacing));
    m18->scale(0.8f);

    Mesh *m19 = scene->addCube(Materials::RubberBlack());
    m19->moveTo(vec3(startX + 3 * spacing, 0, startZ - 3 * spacing));
    m19->scale(0.8f);

    Mesh *m20 = scene->addCube(Materials::NeonLight(vec3(0.3f, 0.8f, 1.0f)));
    m20->moveTo(vec3(startX + 4 * spacing, 0, startZ - 3 * spacing));
    m20->scale(0.8f);

    scene->addPointLight(vec3(0, 8, -8), vec3(1.0f), 3.0f, 50.0f);
    scene->addPointLight(vec3(-8, 4, -4), vec3(1.0f, 0.9f, 0.8f), 2.0f, 30.0f);
    scene->addPointLight(vec3(8, 4, -12), vec3(0.8f, 0.9f, 1.0f), 2.0f, 30.0f);

    scene->setAmbientLight(vec3(0.03f));

    Material floorMat(vec3(0.9f), 0.05f, 0.0f);
    floorMat.specular = vec3(0.04f);
    floorMat.clearcoat = 0.5f;
    floorMat.clearcoatRoughness = 0.1f;
    scene->addPlaneXZ(-1.5f, 50.0f, floorMat);

    scene->setCamera(vec3(0, 6, 5), vec3(0, -0.5f, -10), vec3(0, 1, 0), 45.0f);
    scene->setSkyGradient(vec3(0.05f, 0.05f, 0.08f), vec3(0.02f, 0.02f, 0.03f));

    return scene;
}

} // namespace DemoScenes

// function create base showcase scene

inline std::unique_ptr<Scene> createBaseShowcaseScene(int width, int height) {
    auto scene = std::make_unique<Scene>(width, height);

    scene->setCamera(vec3(0, 2.0, 6.0), vec3(0, 1.0, 0), vec3(0, 1, 0), 60.0f);

    scene->addSpotLight(vec3(0, 6, 6), vec3(0, -1, -1), vec3(1.0f), 8.0f, 0.4f,
                        0.8f, 50.0f);

    scene->addPointLight(vec3(-5, 3, 5), vec3(0.3f), 1.0f, 30.0f);

    scene->setAmbientLight(vec3(0.15f));

    Material floorMat(vec3(0.5f));
    floorMat.specular = vec3(0.1f);
    scene->addPlaneXZ(0.0f, 50.0f, floorMat);

    return scene;
}

inline std::pair<std::unique_ptr<Scene>, std::string>
buildSceneById(RenderConfig configs) {
    std::unique_ptr<Scene> scene;
    std::string sceneName;

    switch (configs.sceneId) {

    default:
        std::cerr << "Invalid scene ID. Using default Character Showcase "
                     "(Scene 1).\n";

    case 1: {
        sceneName = "Character Showcase";
        scene = std::make_unique<Scene>(configs.width, configs.height);

        Mesh *gem1 = scene->addMesh("models/ugly.obj", Materials::Glass());
        gem1->scale(10.5f);
        gem1->moveTo(vec3(-3.0f, 0.0f, 0.0f));

        Mesh *gem2 =
            scene->addMesh("models/halfway.obj", Materials::MarbleNero());
        gem2->scale(10.5f);
        gem2->moveTo(vec3(0.0f, 0.0f, 0.0f));

        Mesh *gem3 =
            scene->addMesh("models/full.obj", Materials::MarbleVerde());
        gem3->scale(10.5f);
        gem3->moveTo(vec3(3.0f, 0.0f, 0.0f));

        scene->addSpotLight(vec3(0, 4, 2), vec3(0, -1, -0.5f),
                            vec3(1.0f, 1.0f, 1.0f), 5.0f, 0.1f, 0.3f, 1.75f);
        scene->addPointLight(vec3(0, 4.5, 2), vec3(0.5f, 0.5f, 1.0f), 1.0f,
                             1.0f);
        scene->addSpotLight(vec3(0.0f, 5.0f, -4.0f), vec3(0.0f, -0.6f, -1.0f),
                            vec3(1.0f, 1.0f, 1.0f), 6.0f, 0.2f, 0.8f, 2.0f);
        scene->setAmbientLight(vec3(0.08f));

        scene->setCamera(vec3(0, 3, 0), vec3(0, 3.5, 5), vec3(0, 1, 0), 60.0f);

        Material floorMat(vec3(0.8f));
        floorMat.specular = vec3(0.1f);
        scene->addPlaneXZ(-3.0f, 50.0f, floorMat);

        break;
    }

    case 2:
        sceneName = "Presidents Showcase";
        scene = createBaseShowcaseScene(configs.width, configs.height);
        {
            Mesh *m1 = scene->addMesh(
                "models/abraham-lincoln-mills-life-mask-150k.obj",
                Materials::MarbleNero());
            m1->scale(.01f);
            m1->moveTo(vec3(-1.2f, 0.0f, 0.0f));
            m1->rotateSelfEulerXYZ(vec3(0, 0, 0));

            Mesh *m2 =
                scene->addMesh("models/andrew-jackson-zinc-sculpture-150k.obj",
                               Materials::MarbleNero());
            m2->scale(0.01f);
            m2->moveTo(vec3(1.2f, 0.0f, 0.0f));
            m2->rotateSelfEulerXYZ(vec3(0, 0, 0));
        }
        break;

    case 3:
        sceneName = "Statues Showcase";
        scene = createBaseShowcaseScene(configs.width, configs.height);
        {
            Mesh *m1 = scene->addMesh(
                "models/cosmic-buddha-laser-scan-150k.obj", Materials::Gold());
            m1->scale(0.001f);
            m1->moveTo(vec3(-1.2f, 0.0f, 0.0f));
            m1->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, 0, 0));

            Mesh *m2 = scene->addMesh(
                "models/george-washington-greenough-statue-(1840)-150k.obj",
                Materials::MarbleNero());
            m2->scale(0.001f);
            m2->moveTo(vec3(1.2f, 0.0f, 0.0f));
            m2->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, 0, 0));
        }
        break;

    case 4:
        sceneName = "Fancy Dudes Showcase";
        scene = createBaseShowcaseScene(configs.width, configs.height);
        {
            Mesh *m1 = scene->addMesh(
                "models/george-washington-greenough-statue-(1840)-150k.obj",
                Materials::Skin());
            m1->scale(0.001f);
            m1->moveTo(vec3(-1.2f, 0.0f, 0.0f));
            m1->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, PI, 0));

            Mesh *m2 = scene->addMesh("models/x3d-cm-exterior-top-160k-uvs.obj",
                                      Materials::Skin());
            m2->scale(0.01f);
            m2->moveTo(vec3(1.2f, 0.0f, 0.0f));
            m2->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, PI, 0));
        }
        break;

    case 5:
        sceneName = "Scans Showcase";
        scene = createBaseShowcaseScene(configs.width, configs.height);
        {
            Mesh *m1 = scene->addMesh("models/x3d-cm-exterior-top-160k-uvs.obj",
                                      Materials::FrostedGlass());
            m1->scale(0.01f);
            m1->moveTo(vec3(-1.2f, 0.0f, 0.0f));
            m1->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, 0, 0));

            Mesh *m2 =
                scene->addMesh("models/x3d-cm-exterior-shell-90k-uvs.obj",
                               Materials::FrostedGlass());
            m2->scale(0.01f);
            m2->moveTo(vec3(1.2f, 0.0f, 0.0f));
            m2->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, 0, 0));
        }
        break;

    case 6:
        sceneName = "Artifacts Showcase";
        scene = createBaseShowcaseScene(configs.width, configs.height);
        {
            Mesh *m1 = scene->addMesh("models/usnm_346-01-100k.obj",
                                      Materials::FrostedGlass());
            m1->scale(0.01f);
            m1->moveTo(vec3(-1.2f, 0.0f, 0.0f));
            m1->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, 0, 0));

            Mesh *m2 = scene->addMesh("models/vase.obj", Materials::Glass());
            m2->scale(0.005f);
            m2->moveTo(vec3(1.2f, 0.0f, 0.0f));
            m2->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, 0, 0));
        }
        break;

    case 7:
        sceneName = "Dynamic Circle Animation";
        scene = createBaseShowcaseScene(configs.width, configs.height);
        {

            Mesh *m1 = scene->addMesh("models/ugly.obj", Materials::Glass());
            m1->scale(10.5f);
            m1->moveTo(vec3(3.0f, 0.0f, 0.0f));

            Mesh *m2 =
                scene->addMesh("models/halfway.obj", Materials::MarbleNero());
            m2->scale(10.5f);
            m2->moveTo(vec3(-1.5f, 0.0f, 2.6f));

            Mesh *m3 =
                scene->addMesh("models/full.obj", Materials::MarbleVerde());
            m3->scale(10.5f);
            m3->moveTo(vec3(-1.5f, 0.0f, -2.6f));
        }
        break;
    }

    scene->setBVHLeafTarget(configs.bvhLeafTarget, configs.bvhLeafTol);

    return {std::move(scene), sceneName};
}

#endif