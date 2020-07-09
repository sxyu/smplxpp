#pragma once
#ifndef VIEWER_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4
#define VIEWER_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4

// Contains definitions of Mesh, PointCloud, and Line

#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <cstddef>

#include "meshview/common.hpp"
#include "meshview/texture.hpp"
#include "meshview/shader.hpp"
#include "meshview/camera.hpp"

namespace meshview {

// Represents a triangle mesh with vertices (including uv, normals),
// triangular faces, and textures
class Mesh {
public:
    explicit Mesh(size_t num_verts, size_t num_triangles = -1);
    // Construct with given points and faces
    explicit Mesh(const Eigen::Ref<const Points>& pos,
                  const Eigen::Ref<const Triangles>& tri_faces,
                  const Eigen::Ref<const Points2D>& uv = Points2D(),
                  const Eigen::Ref<const Points>& normals = Points());
    // Construct without EBO (triangles will be 0 1 2, 3 4 5 etc)
    explicit Mesh(const Eigen::Ref<const Points>& pos,
                  const Eigen::Ref<const Points2D>& uv = Points2D(),
                  const Eigen::Ref<const Points>& normals = Points());
    ~Mesh();

    // Draw mesh to shader wrt camera
    void draw(const Shader& shader, const Camera& camera);

    // Compute normals automatically from verts_pos
    // usage: first set vertex positions using verts_pos()
    // then call estimate_normals()
    Mesh& estimate_normals();

    // Add a texture from a file
    // type: one of diffuse specular normal height
    // returns self (for chaining)
    template<int Type = Texture::TYPE_DIFFUSE>
    Mesh& add_texture(const std::string& path) {
        textures[Type].emplace_back(path, Type);
        return *this;
    }
    // Add solid texture
    template<int Type = Texture::TYPE_DIFFUSE>
    Mesh& add_texture_solid(const Eigen::Ref<const Vector3f>& color) {
        textures[Type].emplace_back(color, Type);
        return *this;
    }
    template<int Type = Texture::TYPE_DIFFUSE>
    Mesh& add_texture_solid(float r, float g, float b) {
        textures[Type].emplace_back(Vector3f(r, g, b), Type);
        return *this;
    }

    // Set specular shininess parameter
    Mesh& set_shininess(float val);

    // Apply translation
    Mesh& translate(const Eigen::Ref<const Vector3f>& vec);

    // Apply rotation
    Mesh& rotate(const Eigen::Ref<const Matrix3f>& mat);

    // Apply scaling
    Mesh& scale(const Eigen::Ref<const Vector3f>& vec);
    // Apply uniform scaling
    Mesh& scale(float val);

    // Set transform
    Mesh& set_transform(const Eigen::Ref<const Matrix4f>& mat);

    // Init or update VAO/VBO/EBO buffers from current vertex and triangle data
    // Must called before first draw for each GLFW context to ensure
    // textures are reconstructed.
    void update(bool force_init = false);

    // ADVANCED: Free buffers. Used automatically in destructor.
    void free_bufs();

    // *Accessors
    // Position part of verts
    inline Eigen::Ref<Points> verts_pos() { return verts.leftCols<3>(); }

    // UV part of verts
    inline Eigen::Ref<Points2D> verts_uv() { return verts.middleCols<2>(3); }

    // Normal part of verts
    inline Eigen::Ref<Points> verts_norm() { return verts.rightCols<3>(); }

    // Enable/disable object
    Mesh& enable(bool val = true);

    // * Example meshes
    // Triangle
    static Mesh Triangle(const Eigen::Ref<const Vector3f>& a,
                         const Eigen::Ref<const Vector3f>& b,
                         const Eigen::Ref<const Vector3f>& c);
    // Square centered at 0,0,0 with normal in z direction and side length 'side_len'
    static Mesh Square(float side_len);

    // Cube centered at 0,0,0 with side length 'side_len'
    static Mesh Cube(float side_len);

    // * Per-instance constants; can't be changed since we want to update VBO in-place
    // num_verts: # verts
    // num_triangles: # triangles, -1 if not using EBO
    const size_t num_verts, num_triangles;

    // Shape (num_verts, 8)
    // 3 x vertex position
    // 2 x uv
    // 3 x normal
    PointsUVN verts;

    // Shape (num_triangles, 3)
    // Triangle indices, empty if num_triangles = -1 (not using EBO)
    Triangles faces;

    // Whether this mesh is enabled; if false, does not draw anything
    bool enabled = true;

    // Textures
    std::array<std::vector<Texture>, Texture::__TYPE_COUNT> textures;

    // Shininess
    float shininess = 32.f;

    // Model local transfom
    Matrix4f transform;

private:
    // Generate a white 1x1 texture to blank_tex_id
    // used to fill maps if no texture provided
    void gen_blank_texture();

    // Vertex Array Object index
    Index VAO = -1;

    Index VBO = -1, EBO = -1;
    Index blank_tex_id = -1;
};

// Represents a 3D point cloud with vertices (including uv, normals)
// where each vertex has a color.
// Also supports drawing the points as a polyline (call draw_lines()).
class PointCloud {
public:
    explicit PointCloud(size_t num_verts);
    // Set vertices with positions pos with colors rgb
    explicit PointCloud(const Eigen::Ref<const Points>& pos,
                        const Eigen::Ref<const Points>& rgb);
    // Set all points to same color
    // (can't put Eigen::Vector3f due 'ambiguity' with above)
    explicit PointCloud(const Eigen::Ref<const Points>& pos,
                        float r = 1.f, float g = 1.f, float b = 1.f);
    ~PointCloud();

    // Draw mesh to shader wrt camera
    void draw(const Shader& shader, const Camera& camera);

    // Position part of verts
    inline Eigen::Ref<Points> verts_pos() { return verts.leftCols<3>(); }
    // RGB part of verts
    inline Eigen::Ref<Points> verts_rgb() { return verts.rightCols<3>(); }

    // Enable/disable object
    PointCloud& enable(bool val = true);
    // Set the point size for drawing
    inline PointCloud & set_point_size(float val) { point_size = val; return *this; }
    // Draw polylines between consecutive points
    inline PointCloud & draw_lines() { lines = true; return *this; }

    // Apply translation
    PointCloud& translate(const Eigen::Ref<const Vector3f>& vec);

    // Apply rotation
    PointCloud& rotate(const Eigen::Ref<const Matrix3f>& mat);
    // Apply scaling
    PointCloud& scale(const Eigen::Ref<const Vector3f>& vec);
    // Apply uniform scaling
    PointCloud& scale(float val);

    // Set transform
    PointCloud& set_transform(const Eigen::Ref<const Matrix4f>& mat);

    // Init or update VAO/VBO buffers from current vertex data
    // Must called before first draw for each GLFW context to ensure
    // textures are reconstructed.
    // force_init: INTERNAL, whether to force recreating buffers, DO NOT use this
    void update(bool force_init = false);

    // ADVANCED: Free buffers. Used automatically in destructor.
    void free_bufs();

    // * Example point clouds/lines
    static PointCloud Line(const Eigen::Ref<const Vector3f>& a,
                           const Eigen::Ref<const Vector3f>& b,
                           const Eigen::Ref<const Vector3f>& color = Vector3f(1.f, 1.f, 1.f));

    const size_t num_verts;

    // Data store
    PointsRGB verts;

    // Whether this point cloud is enabled; if false, does not draw anything
    bool enabled = true;

    // If true, draws polylines between vertices
    // If false (default), draws points only
    bool lines = false;

    // Point size (if lines = false)
    float point_size = 1.f;

    // Model local transfom
    Matrix4f transform;

private:
    // Buffer indices
    Index VAO = -1, VBO = -1;
};

}  // namespace meshview

#endif  // ifndef VIEWER_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4
