// blue_noise.cuh
#ifndef BLUE_NOISE_CUH
#define BLUE_NOISE_CUH

#include <cuda_runtime.h>
#include <string> // For loadBlueNoise prototype
#include <vector> // For generateBlueNoise2D prototype

// Blue noise texture size (power of 2 for fast wrapping)
#define BLUE_NOISE_SIZE 64
#define BLUE_NOISE_CHANNELS 2 // For 2D disk sampling

// Device constant memory for blue noise (fast cached access)
extern __constant__ float d_blue_noise[BLUE_NOISE_SIZE][BLUE_NOISE_SIZE]
                                      [BLUE_NOISE_CHANNELS];

class BlueNoiseGenerator {
  public:
    /**
     * @brief Generates a 2D blue noise point set using Particle Relaxation.
     *
     * This starts with jittered stratified sampling and then runs a
     * particle simulation to "relax" the points into a more uniform
     * (blue noise) distribution.
     *
     * @param size The width and height of the grid (e.g., 64 for 64x64).
     * @param relaxation_iterations The number of simulation steps. More
     * iterations cost more time but produce higher quality results.
     * @return A vector containing (x, y) coordinates.
     */
    static std::vector<float>
    generateBlueNoise2D(int size, int relaxation_iterations = 25);

    /**
     * @brief Loads precomputed blue noise from a file (or generates it).
     */
    static std::vector<float> loadBlueNoise(const std::string &filename);
};

/**
 * @brief Allocates and copies the blue noise data to the __constant__
 * memory array on the GPU.
 */
void initBlueNoise();

#ifdef BLUE_NOISE_IMPLEMENTATION

#include <algorithm> // For std::max
#include <cmath>     // For fmod, sqrt
#include <cstdio>    // For printf in error checking
#include <cstdlib>   // For exit in error checking
#include <random>    // For std::mt19937

// Helper macro for checking CUDA calls
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__,     \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/**
 * @brief Helper for fmod that correctly handles negative numbers for
 * wrapping.
 */

__constant__ float d_blue_noise[BLUE_NOISE_SIZE][BLUE_NOISE_SIZE]
                               [BLUE_NOISE_CHANNELS];
inline float fmod_wrap(float a, float b) {
    float r = fmod(a, b);
    return r < 0.0f ? r + b : r;
}

// BlueNoiseGenerator Implementation

std::vector<float>
BlueNoiseGenerator::generateBlueNoise2D(int size, int relaxation_iterations) {
    std::vector<float> noise(size * size * BLUE_NOISE_CHANNELS);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float cellSize = 1.0f / size;
    int numPoints = size * size;

    std::vector<std::pair<float, float>> points;
    points.reserve(numPoints);

    // 1. Generate initial Jittered Stratified Sampling points
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float px = (x + dist(rng)) * cellSize;
            float py = (y + dist(rng)) * cellSize;
            points.push_back({px, py});
        }
    }

    // 2. Run Particle Relaxation
    // This is a simple O(N^2) particle simulation.
    // Each point repels every other point, pushing them into a more
    // uniform configuration.

    // Tunable parameters for the simulation
    float stepSize = 0.0001f;    // How much to move each point per iteration
    float min_dist_sq = 0.0001f; // A small value to prevent division by zero

    std::vector<std::pair<float, float>> forces(numPoints);

    for (int iter = 0; iter < relaxation_iterations; ++iter) {
        // Reset forces for this iteration
        std::fill(forces.begin(), forces.end(),
                  std::pair<float, float>{0.0f, 0.0f});

        // Calculate N^2 repulsive forces
        for (int i = 0; i < numPoints; ++i) {
            for (int j = 0; j < numPoints; ++j) {
                if (i == j)
                    continue;

                // Get vector from j to i, with toroidal (wrapping) distance
                float dx = points[i].first - points[j].first;
                float dy = points[i].second - points[j].second;

                // Account for wrapping
                if (dx > 0.5f)
                    dx -= 1.0f;
                if (dx < -0.5f)
                    dx += 1.0f;
                if (dy > 0.5f)
                    dy -= 1.0f;
                if (dy < -0.5f)
                    dy += 1.0f;

                float distSq = dx * dx + dy * dy;

                // Clamp to avoid extreme forces at close range
                distSq = std::max(distSq, min_dist_sq);

                // Calculate repulsive force (F = 1/r^2)
                float invDistSq = 1.0f / distSq;

                forces[i].first += dx * invDistSq;
                forces[i].second += dy * invDistSq;
            }
        }

        // Apply forces to move points
        for (int i = 0; i < numPoints; ++i) {
            // Normalize force (crude normalization)
            float forceMag = sqrt(forces[i].first * forces[i].first +
                                  forces[i].second * forces[i].second);

            // Avoid division by zero if force is zero
            if (forceMag < 1e-6f)
                continue;

            float norm_force_x = forces[i].first / forceMag;
            float norm_force_y = forces[i].second / forceMag;

            // Apply movement and wrap around the 0.0-1.0 domain
            points[i].first =
                fmod_wrap(points[i].first + norm_force_x * stepSize, 1.0f);
            points[i].second =
                fmod_wrap(points[i].second + norm_force_y * stepSize, 1.0f);
        }
    }

    // 3. Fill the final noise vector
    for (int i = 0; i < numPoints; ++i) {
        noise[i * BLUE_NOISE_CHANNELS + 0] = points[i].first;
        noise[i * BLUE_NOISE_CHANNELS + 1] = points[i].second;
    }

    return noise;
}

std::vector<float>
BlueNoiseGenerator::loadBlueNoise(const std::string &filename) {
    (void)filename; // Suppress unused parameter warning

    // Generate with 25 relaxation iterations
    return generateBlueNoise2D(BLUE_NOISE_SIZE, 25);
}

// Standalone Function Implementation

void initBlueNoise() {
    // Generate with 25 relaxation iterations
    std::vector<float> blueNoise =
        BlueNoiseGenerator::generateBlueNoise2D(BLUE_NOISE_SIZE, 25);

    // Copy to constant memory with error checking
    CUDA_CHECK(cudaMemcpyToSymbol(d_blue_noise, blueNoise.data(),
                                  sizeof(float) * BLUE_NOISE_SIZE *
                                      BLUE_NOISE_SIZE * BLUE_NOISE_CHANNELS));
}

#endif // BLUE_NOISE_IMPLEMENTATION
#endif // BLUE_NOISE_CUH