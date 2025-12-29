#include "common/visualization.cuh"
#include <cmath>

#ifndef PI
#define PI 3.14159265358979323846f
#endif

namespace Visualization {

std::vector<Triangle> createLineQuad(const vec3 &start, const vec3 &end,
                                     float thickness, const vec3 &viewDir) {
    std::vector<Triangle> quad;

    vec3 lineDir = (end - start).normalized();

    // Calculate perpendicular to both line direction and view direction
    vec3 perp = cross(lineDir, viewDir);
    if (perp.length() < 0.001f) {
        // Fallback if line is parallel to view
        perp = cross(lineDir, vec3(0, 1, 0));
        if (perp.length() < 0.001f) {
            perp = cross(lineDir, vec3(1, 0, 0));
        }
    }
    perp = perp.normalized() * thickness;

    // Create billboard quad
    vec3 v0 = start - perp;
    vec3 v1 = start + perp;
    vec3 v2 = end + perp;
    vec3 v3 = end - perp;

    quad.emplace_back(v0, v1, v2);
    quad.emplace_back(v0, v2, v3);
    quad.emplace_back(v0, v2, v1);
    quad.emplace_back(v0, v3, v2);

    return quad;
}

// Arrow Generation with LOD

std::vector<Triangle> generateCylinder(float radius, float height,
                                       int segments) {
    std::vector<Triangle> tris;
    std::vector<vec3> verts;

    // Generate vertices for top and bottom circles
    for (int i = 0; i < segments; ++i) {
        float angle = (2.0f * PI * i) / segments;
        float x = radius * cosf(angle);
        float z = radius * sinf(angle);

        verts.push_back(vec3(x, -height / 2, z)); // bottom
        verts.push_back(vec3(x, height / 2, z));  // top
    }

    // Generate side triangles
    for (int i = 0; i < segments; ++i) {
        int idx = i * 2;
        int next_idx = ((i + 1) % segments) * 2;

        // Two triangles per quad
        tris.emplace_back(verts[idx], verts[next_idx], verts[idx + 1]);
        tris.emplace_back(verts[next_idx], verts[next_idx + 1], verts[idx + 1]);
    }

    // Add caps
    vec3 bottomCenter(0, -height / 2, 0);
    vec3 topCenter(0, height / 2, 0);

    for (int i = 0; i < segments; ++i) {
        int idx = i * 2;
        int next_idx = ((i + 1) % segments) * 2;

        // Bottom cap
        tris.emplace_back(bottomCenter, verts[idx], verts[next_idx]);
        // Top cap
        tris.emplace_back(topCenter, verts[next_idx + 1], verts[idx + 1]);
    }

    return tris;
}

std::vector<Triangle> generateCone(float radius, float height, int segments) {
    std::vector<Triangle> tris;
    std::vector<vec3> baseVerts;

    vec3 apex(0, height, 0);
    vec3 baseCenter(0, 0, 0);

    // Generate base vertices
    for (int i = 0; i < segments; ++i) {
        float angle = (2.0f * PI * i) / segments;
        float x = radius * cosf(angle);
        float z = radius * sinf(angle);
        baseVerts.push_back(vec3(x, 0, z));
    }

    // Side triangles
    for (int i = 0; i < segments; ++i) {
        int next = (i + 1) % segments;
        tris.emplace_back(baseVerts[i], baseVerts[next], apex);
    }

    // Base cap
    for (int i = 0; i < segments; ++i) {
        int next = (i + 1) % segments;
        tris.emplace_back(baseCenter, baseVerts[next], baseVerts[i]);
    }

    return tris;
}

void transformTriangles(std::vector<Triangle> &tris, const vec3 &translation,
                        const vec3 &scale, const vec3 &rotationAxis,
                        float rotationAngle) {

    // Compute rotation matrix components if needed
    float c = cosf(rotationAngle);
    float s = sinf(rotationAngle);
    vec3 axis = rotationAxis.normalized();
    float t = 1.0f - c;

    for (auto &tri : tris) {
        // Apply transformations to each vertex
        for (vec3 *v : {&tri.v0, &tri.v1, &tri.v2}) {
            // Scale
            *v = vec3(v->x * scale.x, v->y * scale.y, v->z * scale.z);

            // Rotate (Rodrigues' rotation formula)
            if (rotationAngle != 0.0f) {
                vec3 orig = *v;
                *v = orig * c + cross(axis, orig) * s +
                     axis * dot(axis, orig) * t;
            }

            // Translate
            *v = *v + translation;
        }
    }
}

std::vector<Triangle> generateArrow(const vec3 &origin, const vec3 &direction,
                                    float length, float shaftRadius,
                                    float headRadius, float headLength,
                                    int lodLevel) {
    std::vector<Triangle> arrow;

    // Adjust segment count based on LOD
    int shaftSegments = 6;
    int headSegments = 8;

    switch (lodLevel) {
    case 0: // High quality
        shaftSegments = 8;
        headSegments = 12;
        break;
    case 1: // Medium quality
        shaftSegments = 6;
        headSegments = 8;
        break;
    case 2: // Low quality
        shaftSegments = 4;
        headSegments = 6;
        break;
    case 3: // Minimum quality
        shaftSegments = 3;
        headSegments = 4;
        break;
    }

    // Guard: zero-length direction
    vec3 dir = direction.length() > 0 ? direction.normalized() : vec3(0, 1, 0);

    // 1) Build geometry aligned to +Y, tail at y=0
    const float shaftLength = fmaxf(0.0f, length - headLength);

    // Shaft
    auto cylinder = generateCylinder(shaftRadius, shaftLength, shaftSegments);
    transformTriangles(cylinder, vec3(0, +shaftLength * 0.5f, 0));

    // Head
    auto cone = generateCone(headRadius, headLength, headSegments);
    transformTriangles(cone, vec3(0, shaftLength, 0));

    // Combine
    arrow.insert(arrow.end(), cylinder.begin(), cylinder.end());
    arrow.insert(arrow.end(), cone.begin(), cone.end());

    // 2) Rotate from +Y into `dir`
    const vec3 up(0, 1, 0);
    float c = dot(up, dir);
    c = fminf(fmaxf(c, -1.0f), 1.0f);
    float rotAngle = acosf(c);

    vec3 rotAxis = cross(up, dir);
    float axisLen = rotAxis.length();

    if (axisLen < 1e-6f) {
        if (c < 0.0f) {
            rotAxis = vec3(1, 0, 0);
            transformTriangles(arrow, vec3(0), vec3(1), rotAxis, (float)PI);
        }
    } else {
        transformTriangles(arrow, vec3(0), vec3(1), rotAxis / axisLen,
                           rotAngle);
    }

    // 3) Translate to origin
    transformTriangles(arrow, origin);

    return arrow;
}

// Optimized Camera Frustum Generation

std::vector<Triangle> generateFrustumWireframe(const Camera &cam, float aspect,
                                               float nearPlane, float farPlane,
                                               float lineThickness,
                                               bool useSimpleLines) {
    std::vector<Triangle> lines;

    // Get camera basis vectors
    vec3 origin = cam.get_origin();
    vec3 horizontal = cam.get_horizontal();
    vec3 vertical = cam.get_vertical();
    vec3 lower_left = cam.get_lower_left_corner();

    // Calculate the view direction
    vec3 viewDir = (lower_left + horizontal * 0.5f + vertical * 0.5f - origin)
                       .normalized();

    // Calculate right and up vectors
    vec3 right = horizontal.normalized();
    vec3 up = vertical.normalized();

    // Calculate FOV from camera vectors
    float halfHeight = vertical.length() * 0.5f;
    float halfWidth = horizontal.length() * 0.5f;

    // Near plane corners
    float nearHeight = nearPlane * halfHeight;
    float nearWidth = nearPlane * halfWidth;

    vec3 nearCenter = origin + viewDir * nearPlane;
    vec3 nearTL = nearCenter - right * nearWidth + up * nearHeight;
    vec3 nearTR = nearCenter + right * nearWidth + up * nearHeight;
    vec3 nearBL = nearCenter - right * nearWidth - up * nearHeight;
    vec3 nearBR = nearCenter + right * nearWidth - up * nearHeight;

    // Far plane corners
    float farHeight = farPlane * halfHeight;
    float farWidth = farPlane * halfWidth;

    vec3 farCenter = origin + viewDir * farPlane;
    vec3 farTL = farCenter - right * farWidth + up * farHeight;
    vec3 farTR = farCenter + right * farWidth + up * farHeight;
    vec3 farBL = farCenter - right * farWidth - up * farHeight;
    vec3 farBR = farCenter + right * farWidth - up * farHeight;

    // Helper to add a line segment
    auto addLine = [&lines, lineThickness, useSimpleLines,
                    &viewDir](const vec3 &start, const vec3 &end) {
        if (useSimpleLines) {
            // Use simple quad (2 triangles)
            auto quad = createLineQuad(start, end, lineThickness, viewDir);
            lines.insert(lines.end(), quad.begin(), quad.end());
        } else {
            // Use full cylinder (only if absolutely needed)
            vec3 dir = (end - start).normalized();
            float length = (end - start).length();

            auto cylinder =
                generateCylinder(lineThickness, length, 4); // Minimal segments

            vec3 up(0, 1, 0);
            vec3 rotAxis = cross(up, dir);
            float rotAngle = acosf(fminf(fmaxf(dot(up, dir), -1.0f), 1.0f));

            if (rotAxis.length() > 0.001f) {
                transformTriangles(cylinder, vec3(0), vec3(1.0f),
                                   rotAxis.normalized(), rotAngle);
            }

            vec3 midpoint = (start + end) * 0.5f;
            transformTriangles(cylinder, midpoint);

            lines.insert(lines.end(), cylinder.begin(), cylinder.end());
        }
    };

    // Near plane edges (4 lines)
    addLine(nearTL, nearTR);
    addLine(nearTR, nearBR);
    addLine(nearBR, nearBL);
    addLine(nearBL, nearTL);

    // Far plane edges (4 lines)
    addLine(farTL, farTR);
    addLine(farTR, farBR);
    addLine(farBR, farBL);
    addLine(farBL, farTL);

    // Connecting edges (4 lines)
    addLine(nearTL, farTL);
    addLine(nearTR, farTR);
    addLine(nearBL, farBL);
    addLine(nearBR, farBR);

    return lines;
}

// Image Plane Generation

std::vector<Triangle> generateImagePlane(float width, float height,
                                         const vec3 &position,
                                         const vec3 &normal) {
    std::vector<Triangle> quad;

    // Create a quad in XY plane
    vec3 tl(-width / 2, height / 2, 0);
    vec3 tr(width / 2, height / 2, 0);
    vec3 bl(-width / 2, -height / 2, 0);
    vec3 br(width / 2, -height / 2, 0);

    // Two triangles for the quad
    quad.emplace_back(bl, br, tr);
    quad.emplace_back(bl, tr, tl);

    // Rotate to face the desired normal
    vec3 currentNormal(0, 0, 1);
    vec3 rotAxis = cross(currentNormal, normal);
    if (rotAxis.length() > 0.001f) {
        float angle = acosf(dot(currentNormal, normal));
        transformTriangles(quad, vec3(0), vec3(1.0f), rotAxis.normalized(),
                           angle);
    }

    // Translate to position
    transformTriangles(quad, position);

    return quad;
}

} // namespace Visualization