#pragma once
#ifndef VIEWER_SCENE_62A40FE2_FD3F_4D78_A324_76DAE0A4A10B
#define VIEWER_SCENE_62A40FE2_FD3F_4D78_A324_76DAE0A4A10B

#include <vector>
#include "meshview/mesh.hpp"
#include "meshview/camera.hpp"

namespace meshview {

// A set of meshes + lighting
class Scene {
public:
    Scene();

    // Draw all meshes to shader wrt camera
    void draw(const Shader& shader, const Camera& camera);

    // [INTERNAL] Call before first render for each GLFW context to ensure
    // textures are reconstructed
    void reset();

    // Add mesh
    Mesh& add(Mesh&& mesh);
    Mesh& add(const Mesh& mesh);

    // The meshes
    std::vector<Mesh> meshes;

    // Ambient light color, default 0.1 0.1 0.1
    Vector3f ambient_light_color;

    // Point light position
    Vector3f light_pos;
    // Light color diffuse/specular, default white
    Vector3f light_color_diffuse;
    Vector3f light_color_specular;
private:
    bool initialized = false;
};

}  // namespace meshview

#endif  // ifndef VIEWER_SCENE_62A40FE2_FD3F_4D78_A324_76DAE0A4A10B
