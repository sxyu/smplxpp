#pragma once
#ifndef VIEWER_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4
#define VIEWER_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4

#include <vector>
#include <array>
#include <string>
#include <cstdint>

#include "meshview/common.hpp"
#include "meshview/shader.hpp"
#include "meshview/camera.hpp"

namespace meshview {

// Represents a texture/material
struct Texture {
    // Texture types
    enum {
        TYPE_DIFFUSE,
        TYPE_SPECULAR,
        // TYPE_NORMAL,
        // TYPE_HEIGHT,
        __TYPE_COUNT
    };
    // Texture type names (TYPE_DIFFUSE -> "diffuse" etc)
    static constexpr inline const char* type_to_name(int type) {
        switch(type) {
            case TYPE_SPECULAR: return "specular";
            // case TYPE_NORMAL: return "normal";
            // case TYPE_HEIGHT: return "height";
            default: return "diffuse";
        }
    }

    // Texture from path
    Texture(const std::string& path, bool flip = true, int type = TYPE_DIFFUSE);
    // Texture from solid color (1x1)
    Texture(const Eigen::Ref<const Vector3f>& color, int type = TYPE_DIFFUSE);

    ~Texture();

    // Load texture; need to be called before first use in each context
    // (called by Mesh::reset())
    void load();

    // GL texture id; -1 if unavailable
    MeshIndex id = -1;

    // File path (optional)
    std::string path;

    // Color to use if path empty OR failed to load the texture image
    Vector3f fallback_color;

    // Texture type
    int type;

    // Vertical flip on load?
    bool flip;
};

// A mesh object with vertices (including uv, normals), triangular faces, and textures
class Mesh {
public:
    explicit Mesh(size_t num_verts, size_t num_triangles = -1);
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
    inline Eigen::Ref<PointCloud> verts_pos() { return verts.leftCols<3>(); }

    // UV part of verts
    inline Eigen::Ref<PointCloud2D> verts_uv() { return verts.middleCols<2>(3); }

    // Normal part of verts
    inline Eigen::Ref<PointCloud> verts_norm() { return verts.rightCols<3>(); }

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
    PointCloudUVN verts;

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
    MeshIndex VAO = -1;

    MeshIndex VBO = -1, EBO = -1;
    MeshIndex blank_tex_id = -1;
};

}  // namespace meshview

#endif  // ifndef VIEWER_MESH_5872C703_91C0_48F0_AB16_333F916F9FF4
