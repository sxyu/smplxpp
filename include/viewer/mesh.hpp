#pragma once
#ifndef SMPL_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4
#define SMPL_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4

#include <vector>
#include <string>
#include <Eigen/Core>

namespace smpl {

// Represents a texture
struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
};

// A generic mesh
class Mesh {
public:
    explicit Mesh(size_t num_verts, size_t num_triangles);

    void draw();

    // Constants
    const size_t num_verts, num_triangles;

    // 3 x vertex position
    // 2 x uv
    // 3 x normal
    Eigen::Matrix<float, Eigen::Dynamic, 8, Eigen::RowMajor> verts;

    // Triangle indices
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor> triangles;

    // Textures
    std::vector<Texture> textures;

    // Vertex Array Object index
    unsigned int VAO;

private:
    // Must call before first draw
    void init_or_update();

    unsigned int VBO, EBO;
};

}  // namespace smpl

#endif  // ifndef SMPL_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4
