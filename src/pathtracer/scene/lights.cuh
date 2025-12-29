// lights cuh
// light data structures used by the renderer
// supports point lights and directional lights with simple parameters
// meant to be copied to device memory and accessed in kernels
#ifndef LIGHTS_CUH
#define LIGHTS_CUH

#include "common/vec3.cuh"

enum LightType { LIGHT_POINT = 0, LIGHT_DIRECTIONAL = 1, LIGHT_SPOT = 2 };

struct Light {
    LightType type;
    vec3 position;
    vec3 direction;
    vec3 color;
    float intensity;
    float range;
    float innerCone;
    float outerCone;

    float radius;

    // light
    // implements a unit of behavior used by higher level scene code
    // inputs none
    // returns device

    __host__ __device__ Light()
        // type
        // implements a unit of behavior used by higher level scene code
        // inputs position vec3 0 10 0
        // returns value

        : type(LIGHT_POINT), position(vec3(0, 10, 0)),
          // direction
          // implements a unit of behavior used by higher level scene code
          // inputs vec3 0 1 0 color vec3 1 0f intensity 1 0f
          // returns value

          direction(vec3(0, -1, 0)), color(vec3(1.0f)), intensity(1.0f),
          // range
          // implements a unit of behavior used by higher level scene code
          // inputs 100 0f innercone 0 5f outercone 0 7f
          // returns value

          range(100.0f), innerCone(0.5f), outerCone(0.7f),
          // radius
          // implements a unit of behavior used by higher level scene code
          // inputs 0 0f
          // returns value

          radius(0.0f) {}
};

#endif