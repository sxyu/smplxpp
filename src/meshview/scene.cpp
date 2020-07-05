#include "meshview/scene.hpp"

#include <iostream>
#include <Eigen/Geometry>

namespace meshview {

Scene::Scene() {
    ambient_light_color.setConstant(0.2f);
    light_color_diffuse.setConstant(0.8f);
    light_color_specular.setConstant(1.f);
    light_pos << 1.2f, 1.0f, 2.0f;
}

void Scene::draw(const Shader& shader, const Camera& camera) {
    if (!initialized) {
        initialized = true;
        for (auto& mesh : meshes) {
            mesh.reset();
        }
    }
    shader.set_vec3("light.ambient", ambient_light_color);
    shader.set_vec3("light.diffuse", light_color_diffuse);
    shader.set_vec3("light.specular", light_color_specular);
    shader.set_vec3("light.position", (camera.view.inverse() * light_pos.homogeneous()).head<3>());
    shader.set_vec3("viewPos", camera.get_pos());
    for (auto& mesh : meshes) {
        mesh.draw(shader, camera);
    }
}

void Scene::reset() {
    initialized = false;
}

Mesh& Scene::add(Mesh&& mesh) {
    meshes.push_back(mesh);
    return meshes.back();
}
Mesh& Scene::add(const Mesh& mesh) {
    meshes.push_back(mesh);
    return meshes.back();
}

}  // namespace meshview
